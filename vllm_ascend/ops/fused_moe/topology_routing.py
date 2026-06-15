# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Topology-aware MoE routing integration for vLLM Ascend.

The solver works on a global token view. AllGather already prepares a global
view before expert selection; MC2/All2All/FusedMC2 use local prepared token
slices, so this module gathers logits across EP ranks before routing and slices
results back to the caller rank.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch_npu.distributed
from vllm.forward_context import get_forward_context
from vllm.logger import logger

import vllm_ascend.envs as envs
from vllm_ascend.ascend_forward_context import MoECommType


_STEP_BY_LAYER: dict[tuple[int, str], int] = {}
_TOPOLOGY_CONFIG_CACHE: tuple[str, Any | None] | None = None
_SOLVER_CONFIG_CACHE: tuple[str, Any | None] | None = None
_TOKEN_TOPOLOGY_CONFIG_CACHE: tuple[str, Any | None] | None = None


def topology_aware_routing_enabled() -> bool:
    return envs.VLLM_ENABLE_TOPOLOGY_AWARE_ROUTING


def validate_topology_routing_runtime(*, multistream_overlap_gate: bool) -> None:
    if not envs.VLLM_ENABLE_TOPOLOGY_AWARE_ROUTING:
        return
    if envs.VLLM_ENABLE_TOKEN_DROP:
        raise ValueError(
            "VLLM_ENABLE_TOPOLOGY_AWARE_ROUTING and VLLM_ENABLE_TOKEN_DROP "
            "cannot both be enabled.")
    if multistream_overlap_gate:
        raise ValueError(
            "Topology-aware routing does not support multistream_overlap_gate. "
            "Disable multistream overlap gate before enabling "
            "VLLM_ENABLE_TOPOLOGY_AWARE_ROUTING.")


def apply_topology_aware_routing(
    *,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    scoring_func: str,
    renormalize: bool,
    global_num_experts: int,
    num_local_experts: int,
    ep_rank: int,
    ep_size: int,
    ep_group,
    moe_instance_id: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not envs.VLLM_ENABLE_TOPOLOGY_AWARE_ROUTING:
        return topk_weights, topk_ids
    if envs.VLLM_ENABLE_TOKEN_DROP:
        raise ValueError(
            "Topology-aware routing and token drop are mutually exclusive. "
            "Set only one of VLLM_ENABLE_TOPOLOGY_AWARE_ROUTING or "
            "VLLM_ENABLE_TOKEN_DROP.")

    ctx = get_forward_context()
    if envs.VLLM_TOPOLOGY_AWARE_ROUTING_DECODE_ONLY and not bool(
            getattr(ctx, "uniform_decode", False)):
        return topk_weights, topk_ids

    if scoring_func == "sigmoid":
        raise ValueError(
            "Topology-aware routing currently supports softmax/logits/raw "
            "scoring only; got scoring_func='sigmoid'.")

    if router_logits is None:
        return topk_weights, topk_ids

    route_method = envs.VLLM_TOPOLOGY_AWARE_ROUTING_STRATEGY
    comm_type = getattr(ctx, "moe_comm_type", None)
    num_experts = _num_router_experts(router_logits, global_num_experts)
    local_experts = _resolve_num_local_experts(num_local_experts, num_experts,
                                               ep_size)
    token_counts = _resolve_token_counts(router_logits, ep_size, ep_rank,
                                         ep_group, comm_type)

    if comm_type == MoECommType.ALLGATHER:
        global_router_logits = router_logits
        global_topk_ids_before = topk_ids
        global_token_counts = _counts_for_global_view(router_logits,
                                                      token_counts, ep_size)
        rank_start = 0
        rank_end = int(router_logits.shape[0])
    else:
        global_token_counts = token_counts
        global_router_logits = _gather_uneven(router_logits,
                                              global_token_counts, ep_group)
        global_topk_ids_before = _gather_uneven(topk_ids.contiguous(),
                                                global_token_counts, ep_group)
        rank_start, rank_end = _rank_range(global_token_counts, ep_rank)

    topology_config = _load_topology_config()
    token_topology_config = _load_token_topology_config()
    solver_config = _load_solver_config(topology_config)
    solver_ep_size = _resolve_solver_ep_size(topology_config, ep_size)
    token_source_ranks, token_source_rank_mapping = _build_token_source_ranks_for_routing(
        router_logits=global_router_logits,
        ep_size=solver_ep_size,
        token_counts=global_token_counts,
        token_topology_config=token_topology_config,
    )
    expert_physical_ranks = _expert_physical_ranks(num_experts, local_experts,
                                                   topology_config,
                                                   router_logits.device)
    topology_context: dict[str, Any] = {
        "token_source_ranks": token_source_ranks,
        "expert_physical_ranks": expert_physical_ranks,
    }
    if topology_config is not None:
        topology_context["rank_to_node"] = topology_config.rank_to_node

    route = _load_route_function()
    routed_weights, routed_ids = route(
        global_router_logits,
        top_k,
        scoring_func=scoring_func,
        renormalize=renormalize,
        topology_context=topology_context,
        route_method=route_method,
        solver_config=solver_config,
    )
    routed_ids = routed_ids.to(device=topk_ids.device, dtype=topk_ids.dtype)
    routed_weights = routed_weights.to(device=topk_weights.device,
                                       dtype=topk_weights.dtype)

    if envs.VLLM_TOPOLOGY_AWARE_ROUTING_LOGGING:
        _save_routing_log(
            router_logits=global_router_logits,
            before_topk_ids=global_topk_ids_before,
            after_topk_ids=routed_ids,
            token_counts=global_token_counts,
            token_source_ranks=token_source_ranks,
            token_source_rank_mapping=token_source_rank_mapping,
            expert_physical_ranks=expert_physical_ranks,
            top_k=top_k,
            num_experts=num_experts,
            num_local_experts=local_experts,
            moe_instance_id=moe_instance_id,
            layer_name=layer_name,
            comm_type=comm_type,
            route_method=route_method,
            topology_config=topology_config,
            token_topology_config=token_topology_config,
            solver_config=solver_config,
        )

    return routed_weights[rank_start:rank_end], routed_ids[rank_start:rank_end]


def _load_route_function():
    try:
        from topology_aware_routing import route
        return route
    except ModuleNotFoundError:
        project_root = Path(__file__).resolve().parents[5]
        lib_src = project_root / "topology_aware_routing" / "src"
        if str(lib_src) not in sys.path:
            sys.path.insert(0, str(lib_src))
        from topology_aware_routing import route
        return route


def _load_topology_config():
    global _TOPOLOGY_CONFIG_CACHE
    config_path = envs.VLLM_TOPOLOGY_AWARE_ROUTING_CONFIG
    if _TOPOLOGY_CONFIG_CACHE is not None and _TOPOLOGY_CONFIG_CACHE[0] == config_path:
        return _TOPOLOGY_CONFIG_CACHE[1]
    if not config_path:
        _TOPOLOGY_CONFIG_CACHE = (config_path, None)
        return None
    try:
        from topology_aware_routing import load_topology_config
    except ModuleNotFoundError:
        _load_route_function()
        from topology_aware_routing import load_topology_config
    config = load_topology_config(config_path)
    _TOPOLOGY_CONFIG_CACHE = (config_path, config)
    return config


def _load_token_topology_config():
    global _TOKEN_TOPOLOGY_CONFIG_CACHE
    config_path = envs.VLLM_TOPOLOGY_AWARE_ROUTING_TOKEN_CONFIG
    if _TOKEN_TOPOLOGY_CONFIG_CACHE is not None and _TOKEN_TOPOLOGY_CONFIG_CACHE[0] == config_path:
        return _TOKEN_TOPOLOGY_CONFIG_CACHE[1]
    if not config_path:
        _TOKEN_TOPOLOGY_CONFIG_CACHE = (config_path, None)
        return None
    try:
        from topology_aware_routing import load_token_topology_config
    except ModuleNotFoundError:
        _load_route_function()
        from topology_aware_routing import load_token_topology_config
    config = load_token_topology_config(config_path)
    _TOKEN_TOPOLOGY_CONFIG_CACHE = (config_path, config)
    return config


def _load_solver_config(topology_config):
    global _SOLVER_CONFIG_CACHE
    raw = envs.VLLM_TOPOLOGY_AWARE_ROUTING_SOLVER_CONFIG_JSON
    if _SOLVER_CONFIG_CACHE is not None and _SOLVER_CONFIG_CACHE[0] == raw:
        return _SOLVER_CONFIG_CACHE[1]
    if raw:
        config = json.loads(raw)
        if not isinstance(config, dict):
            raise ValueError(
                "VLLM_TOPOLOGY_AWARE_ROUTING_SOLVER_CONFIG_JSON must be a JSON object")
    elif topology_config is not None:
        config = topology_config.solver_config
    else:
        config = None
    _SOLVER_CONFIG_CACHE = (raw, config)
    return config


def _resolve_solver_ep_size(topology_config, runtime_ep_size: int) -> int:
    if topology_config is None:
        return int(runtime_ep_size)

    rank_to_node = getattr(topology_config, "rank_to_node", None)
    if rank_to_node is not None and int(rank_to_node.numel()) > 0:
        return int(rank_to_node.numel())

    expert_physical_ranks = getattr(topology_config, "expert_physical_ranks", None)
    if expert_physical_ranks is not None and int(expert_physical_ranks.numel()) > 0:
        return int(expert_physical_ranks.max().item()) + 1

    return int(runtime_ep_size)


def _build_token_source_ranks_for_routing(
    *,
    router_logits: torch.Tensor,
    ep_size: int,
    token_counts: torch.Tensor,
    token_topology_config,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if token_topology_config is None:
        return _build_token_source_ranks(token_counts), {
            "mode": "capture",
            "ep_size": int(token_counts.numel()),
        }

    try:
        from topology_aware_routing import build_token_source_ranks_from_config
    except ModuleNotFoundError:
        _load_route_function()
        from topology_aware_routing import build_token_source_ranks_from_config

    token_source_ranks = build_token_source_ranks_from_config(
        num_tokens=int(router_logits.shape[0]),
        ep_size=ep_size,
        token_topology_config=token_topology_config,
        device=router_logits.device,
    )
    return token_source_ranks, {
        "mode": token_topology_config.token_source_policy,
        "name": token_topology_config.name,
        "remainder_policy": token_topology_config.remainder_policy,
        "order": token_topology_config.order,
        "ep_size": int(ep_size),
    }


def _num_router_experts(router_logits: torch.Tensor,
                        global_num_experts: int) -> int:
    del global_num_experts
    return int(router_logits.shape[1])


def _resolve_num_local_experts(num_local_experts: int, num_experts: int,
                               ep_size: int) -> int:
    local_experts = int(num_local_experts or 0)
    if local_experts > 0:
        return local_experts
    if ep_size <= 0:
        return num_experts
    return max(1, num_experts // ep_size)


def _resolve_token_counts(router_logits: torch.Tensor, ep_size: int,
                          ep_rank: int, ep_group,
                          comm_type: Optional[MoECommType]) -> torch.Tensor:
    local_tokens = int(router_logits.shape[0])
    ctx = get_forward_context()
    counts = None
    try:
        counts = ctx.dp_metadata.num_tokens_across_dp_cpu.to(torch.int64).cpu()
    except (AssertionError, AttributeError):
        counts = None

    if comm_type == MoECommType.ALLGATHER:
        if counts is not None and int(counts.sum().item()) == local_tokens:
            return counts
        return _counts_for_global_view(router_logits, counts, ep_size)

    if counts is not None and counts.numel() == ep_size:
        if ep_rank < counts.numel() and int(counts[ep_rank].item()) == local_tokens:
            return counts
    return _gather_local_token_counts(local_tokens, router_logits.device,
                                      ep_size, ep_group)


def _counts_for_global_view(router_logits: torch.Tensor,
                            counts: Optional[torch.Tensor],
                            ep_size: int) -> torch.Tensor:
    total_tokens = int(router_logits.shape[0])
    if counts is not None and counts.numel() > 0:
        if int(counts.sum().item()) == total_tokens:
            return counts.to(torch.int64).cpu()
        if total_tokens % int(counts.numel()) == 0:
            return torch.full((int(counts.numel()), ),
                              total_tokens // int(counts.numel()),
                              dtype=torch.int64)
    if ep_size > 0 and total_tokens % ep_size == 0:
        return torch.full((ep_size, ), total_tokens // ep_size, dtype=torch.int64)
    return torch.tensor([total_tokens], dtype=torch.int64)


def _gather_local_token_counts(local_tokens: int, device: torch.device,
                               ep_size: int, ep_group) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()) or ep_size <= 1:
        return torch.tensor([local_tokens], dtype=torch.int64)
    local = torch.tensor([local_tokens], dtype=torch.int64, device=device)
    gathered = [torch.zeros_like(local) for _ in range(ep_size)]
    dist.all_gather(gathered, local, group=ep_group)
    return torch.cat(gathered).cpu()


def _gather_uneven(tensor: torch.Tensor, counts: torch.Tensor, ep_group):
    total = int(counts.sum().item())
    out = torch.empty((total, *tensor.shape[1:]),
                      dtype=tensor.dtype,
                      device=tensor.device)
    torch_npu.distributed.all_gather_into_tensor_uneven(
        out, tensor.contiguous(), counts.numpy(), group=ep_group)
    return out


def _rank_range(counts: torch.Tensor, rank: int) -> tuple[int, int]:
    start = int(counts[:rank].sum().item()) if rank > 0 else 0
    end = start + int(counts[rank].item())
    return start, end


def _build_token_source_ranks(counts: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=torch.long),
        counts.to(torch.long),
    )


def _expert_physical_ranks(num_experts: int, num_local_experts: int,
                           topology_config, device: torch.device) -> torch.Tensor:
    if topology_config is not None and topology_config.expert_physical_ranks is not None:
        if int(topology_config.expert_physical_ranks.numel()) != num_experts:
            raise ValueError(
                "topology config expert_physical_ranks length does not match router logits experts")
        return topology_config.expert_physical_ranks.to(device=device,
                                                        dtype=torch.long)
    return (torch.arange(num_experts, dtype=torch.long, device=device) //
            int(num_local_experts))


def _is_global_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _safe_layer_name(layer_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", layer_name).strip("_") or "layer"


def _next_step(moe_instance_id: int, layer_name: str) -> int:
    key = (moe_instance_id, layer_name)
    step = _STEP_BY_LAYER.get(key, 0)
    _STEP_BY_LAYER[key] = step + 1
    return step


def _save_routing_log(
    *,
    router_logits: torch.Tensor,
    before_topk_ids: torch.Tensor,
    after_topk_ids: torch.Tensor,
    token_counts: torch.Tensor,
    token_source_ranks: torch.Tensor,
    token_source_rank_mapping: dict[str, Any],
    expert_physical_ranks: torch.Tensor,
    top_k: int,
    num_experts: int,
    num_local_experts: int,
    moe_instance_id: Optional[int],
    layer_name: Optional[str],
    comm_type: Optional[MoECommType],
    route_method: str,
    topology_config,
    token_topology_config,
    solver_config,
) -> None:
    if not _is_global_rank0():
        return

    instance_id = int(moe_instance_id if moe_instance_id is not None else -1)
    layer = str(layer_name or "unknown_layer")
    step = _next_step(instance_id, layer)
    log_root = Path(envs.VLLM_TOPOLOGY_AWARE_ROUTING_LOG_DIR)
    safe_layer = _safe_layer_name(layer)
    file_name = f"router_logits_layer_{instance_id}_step_{step}_{safe_layer}.pt"

    common_metadata = {
        "moe_instance_id": instance_id,
        "layer_name": layer,
        "step": step,
        "top_k": int(top_k),
        "num_tokens": int(router_logits.shape[0]),
        "num_experts": int(num_experts),
        "route_method": route_method,
        "moe_comm_type": comm_type.name if comm_type is not None else None,
        "uniform_decode": bool(getattr(get_forward_context(), "uniform_decode", False)),
        "topology_config": envs.VLLM_TOPOLOGY_AWARE_ROUTING_CONFIG or None,
        "token_topology_config": envs.VLLM_TOPOLOGY_AWARE_ROUTING_TOKEN_CONFIG or None,
        "token_source_rank_mapping": token_source_rank_mapping,
        "solver_config": solver_config,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    capture_payload = {
        **common_metadata,
        "rank": 0,
        "shape": tuple(router_logits.shape),
        "dtype": str(router_logits.dtype),
        "router_logits": router_logits.detach().cpu().contiguous(),
        "token_source_ranks": token_source_ranks.to(torch.int32).contiguous(),
        "num_tokens_across_ranks": token_counts.detach().cpu().contiguous(),
        "actual_token_counts": token_counts.detach().cpu().to(torch.int32).contiguous(),
        "expert_physical_ranks": expert_physical_ranks.detach().cpu().to(torch.int32).contiguous(),
        "expert_rank_semantics": "physical",
        "ep_size": int(token_counts.numel()),
        "num_local_experts": int(num_local_experts),
        "global_num_experts": int(num_experts),
    }
    before_payload = {
        **common_metadata,
        "source_capture_path": str(log_root / "router_logits" / file_name),
        "route_method": "baseline",
        "topk_ids": before_topk_ids.detach().cpu().contiguous(),
    }
    after_payload = {
        **common_metadata,
        "source_capture_path": str(log_root / "router_logits" / file_name),
        "topk_ids": after_topk_ids.detach().cpu().contiguous(),
    }

    for subdir, payload in (
            ("router_logits", capture_payload),
            ("topk_before", before_payload),
            ("topk_after", after_payload),
    ):
        out_dir = log_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_dir / file_name)
    logger.info(
        "[TopologyRouting] Saved routing log for layer=%s step=%d to %s",
        layer, step, log_root)
