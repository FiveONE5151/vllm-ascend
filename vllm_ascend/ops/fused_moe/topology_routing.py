# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Topology-aware MoE routing integration for vLLM Ascend.

The solver works on a global token view. AllGather already prepares a global
view before expert selection; MC2/All2All/FusedMC2 use local prepared token
slices, so this module gathers logits across EP ranks before routing and slices
results back to the caller rank.
"""

from __future__ import annotations

from dataclasses import dataclass
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
_LOG_SKIP_WARNINGS: set[str] = set()


@dataclass
class PreparedTokenLayout:
    max_local_tokens: int
    max_global_tokens: int
    token_source_ranks: torch.Tensor
    local_start: int
    local_end: int
    row_ids: torch.Tensor


@dataclass
class TopologyRoutingState:
    ep_rank: int
    runtime_ep_size: int
    virtual_ep_size: int
    top_k: int
    runtime_num_local_experts: int
    virtual_num_local_experts: int
    global_num_experts: int
    route_method: str
    solver_config: dict[str, Any]
    token_topology_config: Any
    rank_to_node: torch.Tensor | None
    expert_physical_ranks: torch.Tensor
    prepared_layouts: dict[int, PreparedTokenLayout]

    @classmethod
    def from_layer(cls, layer) -> Optional["TopologyRoutingState"]:
        if not envs.VLLM_ENABLE_TOPOLOGY_AWARE_ROUTING:
            return None

        topology_config = _load_topology_config()
        token_topology_config = _load_token_topology_config()
        if token_topology_config is None:
            raise ValueError(
                "Token topology config is required for topology-aware routing. "
                "Set VLLM_TOPOLOGY_AWARE_ROUTING_TOKEN_CONFIG.")
        if getattr(token_topology_config, "token_source_policy",
                   None) != "uniform_ep_rank":
            raise ValueError(
                "Graph-compatible topology-aware routing only supports "
                "token_source_policy='uniform_ep_rank'.")

        runtime_ep_size = int(getattr(layer, "ep_size", 1))
        ep_rank = int(getattr(layer, "ep_rank", 0))
        global_num_experts = int(getattr(layer, "global_num_experts",
                                         getattr(layer, "num_experts", 0)))
        runtime_num_local_experts = int(getattr(layer, "local_num_experts", 0))
        if runtime_num_local_experts <= 0:
            runtime_num_local_experts = _resolve_num_local_experts(
                runtime_num_local_experts, global_num_experts, runtime_ep_size)
        virtual_ep_size = _resolve_solver_ep_size(topology_config, runtime_ep_size)
        virtual_num_local_experts = _resolve_virtual_num_local_experts(
            global_num_experts, virtual_ep_size, topology_config)

        solver_config = _load_solver_config(topology_config) or {}
        device = _initial_state_device()
        rank_to_node = None
        if topology_config is not None and topology_config.rank_to_node is not None:
            rank_to_node = topology_config.rank_to_node.to(device=device,
                                                           dtype=torch.long)
            _validate_rank_to_node_covers_virtual_topology(
                rank_to_node, virtual_ep_size)
        expert_physical_ranks = _expert_physical_ranks(
            global_num_experts, virtual_num_local_experts, topology_config,
            device).to(dtype=torch.long)
        _validate_expert_ranks_cover_virtual_topology(
            expert_physical_ranks, rank_to_node, virtual_ep_size)

        # logger.info(
        #     "TAR from_layer: runtime_ep_size=%d ep_rank=%d virtual_ep_size=%d "
        #     "runtime_num_local_experts=%d virtual_num_local_experts=%d "
        #     "global_num_experts=%d expert_physical_ranks=%s rank_to_node=%s",
        #     runtime_ep_size,
        #     ep_rank,
        #     virtual_ep_size,
        #     runtime_num_local_experts,
        #     virtual_num_local_experts,
        #     global_num_experts,
        #     expert_physical_ranks.detach().cpu().tolist(),
        #     None if rank_to_node is None else rank_to_node.detach().cpu().tolist(),
        # )

        return cls(
            ep_rank=ep_rank,
            runtime_ep_size=runtime_ep_size,
            virtual_ep_size=virtual_ep_size,
            top_k=int(layer.top_k),
            runtime_num_local_experts=runtime_num_local_experts,
            virtual_num_local_experts=virtual_num_local_experts,
            global_num_experts=global_num_experts,
            route_method=envs.VLLM_TOPOLOGY_AWARE_ROUTING_STRATEGY,
            solver_config=solver_config,
            token_topology_config=token_topology_config,
            rank_to_node=rank_to_node,
            expert_physical_ranks=expert_physical_ranks,
            prepared_layouts={},
        )

    def prepare_for_tokens(self, max_local_tokens: int) -> PreparedTokenLayout:
        max_local_tokens = int(max_local_tokens)
        if max_local_tokens <= 0:
            raise ValueError("max_local_tokens must be positive")
        prepared = self.prepared_layouts.get(max_local_tokens)
        if prepared is not None:
            return prepared

        device = self.expert_physical_ranks.device
        max_global_tokens = self.runtime_ep_size * max_local_tokens
        token_source_ranks = _build_token_source_ranks_for_layout(
            num_tokens=max_global_tokens,
            virtual_ep_size=self.virtual_ep_size,
            token_topology_config=self.token_topology_config,
            device=device,
        )
        # logger.info(
        #     "TAR token source rank for max_local_tokens=%d, max_global_tokens=%d: %s",
        #     max_local_tokens,
        #     max_global_tokens,
        #     token_source_ranks.detach().cpu().tolist()
        # )
        local_start = self.ep_rank * max_local_tokens
        local_end = local_start + max_local_tokens
        row_ids = torch.arange(max_local_tokens, dtype=torch.long, device=device)
        prepared = PreparedTokenLayout(
            max_local_tokens=max_local_tokens,
            max_global_tokens=max_global_tokens,
            token_source_ranks=token_source_ranks,
            local_start=local_start,
            local_end=local_end,
            row_ids=row_ids,
        )
        self.prepared_layouts[max_local_tokens] = prepared
        return prepared

    def get_prepared(self, max_local_tokens: int,
                     *, allow_prepare: bool) -> PreparedTokenLayout:
        max_local_tokens = int(max_local_tokens)
        prepared = self.prepared_layouts.get(max_local_tokens)
        if prepared is not None:
            return prepared
        if not allow_prepare:
            raise RuntimeError(
                "Topology-aware routing token layout was not prepared before "
                f"graph capture for max_local_tokens={max_local_tokens}.")
        return self.prepare_for_tokens(max_local_tokens)


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
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    top_k: int,
    use_grouped_topk: bool,
    scoring_func: str,
    renormalize: bool,
    global_num_experts: int,
    num_local_experts: int,
    ep_rank: int,
    ep_size: int,
    ep_group,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Any] = None,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    moe_instance_id: Optional[int] = None,
    layer_name: Optional[str] = None,
    topology_routing_state: Optional[TopologyRoutingState] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del global_num_experts, num_local_experts, ep_rank, ep_size
    if envs.VLLM_ENABLE_TOKEN_DROP:
        raise ValueError(
            "Topology-aware routing and token drop are mutually exclusive. "
            "Set only one of VLLM_ENABLE_TOPOLOGY_AWARE_ROUTING or "
            "VLLM_ENABLE_TOKEN_DROP.")
    if topology_routing_state is None:
        raise RuntimeError(
            "Topology-aware routing is enabled but no TopologyRoutingState was "
            "passed from the AscendFusedMoE layer.")

    ctx = get_forward_context()
    if scoring_func == "sigmoid":
        raise ValueError(
            "Topology-aware routing currently supports softmax/logits/raw "
            "scoring only; got scoring_func='sigmoid'.")
    if int(top_k) != topology_routing_state.top_k:
        raise RuntimeError(
            f"TAR state top_k={topology_routing_state.top_k} does not match "
            f"runtime top_k={top_k}.")

    runtime_mode = getattr(ctx, "cudagraph_runtime_mode", None)
    runtime_mode_name = getattr(runtime_mode, "name", "NONE")
    graph_mode = bool(getattr(ctx, "my_capturing", False) or
                      getattr(ctx, "capturing", False) or
                      runtime_mode_name != "NONE")
    logging_requested = bool(envs.VLLM_TOPOLOGY_AWARE_ROUTING_LOGGING)
    logging_enabled = logging_requested
    if logging_enabled and graph_mode:
        _warn_skip_routing_log_once(
            "graph_mode",
            "Skip TAR routing log because graph/full-cudagraph mode does not "
            "provide per-step online logging semantics.")
        logging_enabled = False

    local_valid_mask = getattr(ctx, "tar_valid_token_mask", None)
    comm_type = getattr(ctx, "moe_comm_type", None)
    if local_valid_mask is None:
        if graph_mode:
            raise RuntimeError(
                "Graph-compatible topology-aware routing requires "
                "forward_context.tar_valid_token_mask. Ensure the forward "
                "context is initialized with the graph bucket size before "
                "MoE routing.")
        max_local_tokens = int(router_logits.shape[0])
    else:
        max_local_tokens = int(local_valid_mask.shape[0])
        _validate_fixed_local_shape(router_logits, "router_logits",
                                    max_local_tokens)
        local_valid_mask = local_valid_mask.to(device=router_logits.device,
                                               dtype=torch.bool)
    layout = topology_routing_state.get_prepared(
        max_local_tokens, allow_prepare=not graph_mode)

    output_rows = int(router_logits.shape[0])
    if comm_type == MoECommType.ALLGATHER:
        global_router_logits = router_logits
        global_valid_mask = local_valid_mask
        rank_start = 0
        rank_end = max_local_tokens
    else:
        global_router_logits = _fixed_gather_first_dim(
            router_logits, topology_routing_state.runtime_ep_size, ep_group)
        global_valid_mask = (
            None if local_valid_mask is None else _fixed_gather_first_dim(
                local_valid_mask, topology_routing_state.runtime_ep_size, ep_group))
        rank_start = layout.local_start
        rank_end = layout.local_end

    route = _load_route_function()
    global_rows = int(global_router_logits.shape[0])
    route_kwargs = {
        "route_method": topology_routing_state.route_method,
        "scoring_func": scoring_func,
        "renormalize": renormalize,
        "topology_context": {
            "token_source_ranks": layout.token_source_ranks[:global_rows],
            "expert_physical_ranks": topology_routing_state.expert_physical_ranks,
            "rank_to_node": topology_routing_state.rank_to_node,
        },
        "solver_config": topology_routing_state.solver_config,
    }
    if global_valid_mask is not None:
        route_kwargs["valid_token_mask"] = global_valid_mask
    logger.info(
        "[TopologyRouting] mask valid=%d total=%d logits_shape=%s comm=%s",
        int(global_valid_mask.sum().item()) if global_valid_mask is not None else -1,
        int(global_valid_mask.numel()) if global_valid_mask is not None else -1,
        tuple(global_router_logits.shape),
        comm_type.name if comm_type is not None else None,
    )

    routed_weights, routed_ids = route(
        global_router_logits,
        top_k,
        **route_kwargs,
    )
    routed_ids = routed_ids.to(device=router_logits.device, dtype=torch.int32)
    routed_weights = routed_weights.to(device=router_logits.device,
                                       dtype=torch.bfloat16)

    if logging_enabled:
        baseline_topk_weights = topk_weights
        baseline_topk_ids = topk_ids
        if baseline_topk_ids is None or baseline_topk_weights is None:
            baseline_topk_weights, baseline_topk_ids = _compute_local_baseline_topk(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias,
                global_num_experts=topology_routing_state.global_num_experts,
            )
        runtime_token_counts = _resolve_token_counts(
            router_logits,
            topology_routing_state.runtime_ep_size,
            topology_routing_state.ep_rank,
            comm_type,
        )
        if comm_type == MoECommType.ALLGATHER:
            global_before_topk_ids = baseline_topk_ids
        else:
            global_before_topk_ids = _gather_uneven(
                baseline_topk_ids, runtime_token_counts, ep_group)
        token_source_ranks = layout.token_source_ranks[:global_rows]
        _save_routing_log(
            router_logits=global_router_logits,
            before_topk_ids=global_before_topk_ids,
            after_topk_ids=routed_ids,
            valid_token_mask=global_valid_mask,
            token_counts=_token_counts_from_source_ranks(
                token_source_ranks, topology_routing_state.virtual_ep_size),
            actual_token_counts=runtime_token_counts,
            token_source_ranks=token_source_ranks,
            token_source_rank_mapping=_token_source_rank_mapping(
                topology_routing_state.token_topology_config,
                topology_routing_state.virtual_ep_size),
            expert_physical_ranks=topology_routing_state.expert_physical_ranks,
            top_k=top_k,
            num_experts=int(global_router_logits.shape[1]),
            num_local_experts=topology_routing_state.virtual_num_local_experts,
            moe_instance_id=moe_instance_id,
            layer_name=layer_name,
            comm_type=comm_type,
            route_method=topology_routing_state.route_method,
            solver_config=topology_routing_state.solver_config,
        )


    local_routed_weights = routed_weights[rank_start:rank_end]
    local_routed_ids = routed_ids[rank_start:rank_end]
    return (local_routed_weights[:output_rows],
            local_routed_ids[:output_rows])


def _initial_state_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu", torch.npu.current_device())
    return torch.device("cpu")


def _validate_fixed_local_shape(tensor: torch.Tensor, name: str,
                                expected_rows: int) -> None:
    rows = int(tensor.shape[0])
    if rows != int(expected_rows):
        raise RuntimeError(
            f"Topology-aware routing expects {name} to be pre-padded to "
            f"the graph bucket size {expected_rows}, but got {rows} rows.")


def _fixed_gather_first_dim(tensor: torch.Tensor, ep_size: int, ep_group):
    if int(ep_size) == 1:
        return tensor.contiguous()
    out = torch.empty((int(ep_size) * int(tensor.shape[0]), *tensor.shape[1:]),
                      dtype=tensor.dtype,
                      device=tensor.device)
    dist.all_gather_into_tensor(out, tensor.contiguous(), group=ep_group)
    return out


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


def _resolve_virtual_num_local_experts(
    num_experts: int,
    virtual_ep_size: int,
    topology_config,
) -> int:
    if virtual_ep_size <= 0:
        raise ValueError("virtual_ep_size must be positive")
    expert_physical_ranks = getattr(topology_config, "expert_physical_ranks", None)
    if expert_physical_ranks is not None:
        ranks = expert_physical_ranks.detach().cpu().to(torch.long)
        if int(ranks.numel()) != int(num_experts):
            raise ValueError(
                "topology config expert_physical_ranks length does not match "
                "router logits experts")
        if ranks.numel() and int(ranks.max().item()) >= int(virtual_ep_size):
            raise ValueError(
                "topology config expert_physical_ranks contains rank outside "
                "virtual_ep_size")
        counts = torch.bincount(ranks, minlength=int(virtual_ep_size))
        if counts.numel() != int(virtual_ep_size) or not bool((counts == counts[0]).all().item()):
            raise ValueError(
                "topology config expert_physical_ranks must assign the same "
                "number of experts to each virtual EP rank")
        return int(counts[0].item())
    if int(num_experts) % int(virtual_ep_size) != 0:
        raise ValueError(
            "global_num_experts must be divisible by virtual_ep_size when "
            "topology config does not provide expert_physical_ranks")
    return int(num_experts) // int(virtual_ep_size)


def _validate_rank_to_node_covers_virtual_topology(
    rank_to_node: torch.Tensor, virtual_ep_size: int
) -> None:
    if int(rank_to_node.numel()) < int(virtual_ep_size):
        raise ValueError(
            "topology config rank_to_node length must cover virtual_ep_size")


def _validate_expert_ranks_cover_virtual_topology(
    expert_physical_ranks: torch.Tensor,
    rank_to_node: torch.Tensor | None,
    virtual_ep_size: int,
) -> None:
    if expert_physical_ranks.numel() and int(expert_physical_ranks.min().item()) < 0:
        raise ValueError("expert_physical_ranks contains negative values")
    if expert_physical_ranks.numel() and int(expert_physical_ranks.max().item()) >= int(virtual_ep_size):
        raise ValueError("expert_physical_ranks contains rank outside virtual_ep_size")
    if rank_to_node is not None and expert_physical_ranks.numel():
        max_rank = int(expert_physical_ranks.max().item())
        if max_rank >= int(rank_to_node.numel()):
            raise ValueError("rank_to_node does not cover expert_physical_ranks")


def _build_token_source_ranks_for_layout(
    *,
    num_tokens: int,
    virtual_ep_size: int,
    token_topology_config,
    device: torch.device,
) -> torch.Tensor:
    if token_topology_config is None:
        raise ValueError("Token topology config is required for topology-aware routing.")
    try:
        from topology_aware_routing import build_token_source_ranks_from_config
    except ModuleNotFoundError:
        _load_route_function()
        from topology_aware_routing import build_token_source_ranks_from_config
    return build_token_source_ranks_from_config(
        num_tokens=int(num_tokens),
        ep_size=int(virtual_ep_size),
        token_topology_config=token_topology_config,
        device=device,
    )


def _build_token_source_ranks_for_routing(
    *,
    router_logits: torch.Tensor,
    ep_size: int,
    token_counts: torch.Tensor,
    token_topology_config,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if token_topology_config is None:
        raise ValueError(
            "Token topology config is required for topology-aware routing. Please set VLLM_TOPOLOGY_AWARE_ROUTING_TOKEN_CONFIG to a valid config file path or provide a config with token topology settings."
        )

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
                          ep_rank: int,
                          comm_type: Optional[MoECommType]) -> torch.Tensor:
    local_tokens = int(router_logits.shape[0])
    ctx = get_forward_context()

    try:
        counts = ctx.dp_metadata.num_tokens_across_dp_cpu.to(torch.int64).cpu()
    except (AssertionError, AttributeError) as exc:
        raise RuntimeError(
            "Topology-aware routing requires dp_metadata.num_tokens_across_dp_cpu "
            "in pure dp+ep mode.") from exc

    if counts.numel() != ep_size:
        raise RuntimeError(
            "Topology-aware routing expects pure dp+ep mode with matching "
            f"dp/ep sizes, but got len(num_tokens_across_dp_cpu)={counts.numel()} "
            f"and ep_size={ep_size}.")
    if not 0 <= ep_rank < ep_size:
        raise RuntimeError(
            f"Invalid ep_rank={ep_rank} for ep_size={ep_size}.")

    if comm_type == MoECommType.ALLGATHER:
        total_tokens = int(counts.sum().item())
        if total_tokens != local_tokens:
            raise RuntimeError(
                "Topology-aware routing expects ALLGATHER router_logits to "
                "already be the global token view in pure dp+ep mode, but got "
                f"sum(num_tokens_across_dp_cpu)={total_tokens} and "
                f"router_logits.shape[0]={local_tokens}.")
        return counts

    expected_local_tokens = int(counts[ep_rank].item())
    if expected_local_tokens != local_tokens:
        raise RuntimeError(
            "Topology-aware routing expects local router_logits to match "
            "dp_metadata for non-ALLGATHER communication in pure dp+ep mode, "
            f"but got num_tokens_across_dp_cpu[{ep_rank}]="
            f"{expected_local_tokens} and router_logits.shape[0]={local_tokens}.")
    return counts

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


def _expert_physical_ranks(num_experts: int, virtual_num_local_experts: int,
                           topology_config, device: torch.device) -> torch.Tensor:
    if topology_config is not None and topology_config.expert_physical_ranks is not None:
        if int(topology_config.expert_physical_ranks.numel()) != num_experts:
            raise ValueError(
                "topology config expert_physical_ranks length does not match router logits experts")
        return topology_config.expert_physical_ranks.to(device=device,
                                                        dtype=torch.long)
    return (torch.arange(num_experts, dtype=torch.long, device=device) //
            int(virtual_num_local_experts))


def _compute_local_baseline_topk(
    *,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int],
    num_expert_group: Optional[int],
    custom_routing_function: Optional[Any],
    scoring_func: str,
    routed_scaling_factor: float,
    e_score_correction_bias: Optional[torch.Tensor],
    global_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_ascend.ops.fused_moe import experts_selector as experts_selector_mod

    is_support_npu_moe_gating_top_k =         experts_selector_mod.check_npu_moe_gating_top_k(
            hidden_states=hidden_states,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            custom_routing_function=custom_routing_function)
    if is_support_npu_moe_gating_top_k:
        return experts_selector_mod._select_experts_with_fusion_ops(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            global_num_experts=global_num_experts)
    return experts_selector_mod._native_select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias,
        global_num_experts=global_num_experts,
    )


def _warn_skip_routing_log_once(key: str, message: str) -> None:
    if key in _LOG_SKIP_WARNINGS or not _is_global_rank0():
        return
    _LOG_SKIP_WARNINGS.add(key)
    logger.warning("[TopologyRouting] %s", message)


def _token_counts_from_source_ranks(token_source_ranks: torch.Tensor,
                                    ep_size: int) -> torch.Tensor:
    return torch.bincount(token_source_ranks.to(dtype=torch.long).cpu(),
                          minlength=int(ep_size)).to(dtype=torch.int32)


def _token_source_rank_mapping(token_topology_config,
                               ep_size: int) -> dict[str, Any]:
    if token_topology_config is None:
        return {"mode": "unknown", "ep_size": int(ep_size)}
    return {
        "mode": getattr(token_topology_config, "token_source_policy", None),
        "name": getattr(token_topology_config, "name", None),
        "remainder_policy": getattr(token_topology_config,
                                     "remainder_policy", None),
        "order": getattr(token_topology_config, "order", None),
        "ep_size": int(ep_size),
    }

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
    valid_token_mask: torch.Tensor | None,
    token_counts: torch.Tensor,
    actual_token_counts: torch.Tensor,
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
    if valid_token_mask is None:
        valid_token_mask = torch.ones(
            int(router_logits.shape[0]), dtype=torch.bool, device=router_logits.device)
    else:
        valid_token_mask = valid_token_mask.to(device=router_logits.device,
                                               dtype=torch.bool)

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
        "valid_token_mask": valid_token_mask.detach().cpu().contiguous(),
        "token_source_ranks": token_source_ranks.detach().cpu().to(torch.int32).contiguous(),
        "num_tokens_across_ranks": token_counts.detach().cpu().to(torch.int32).contiguous(),
        "actual_token_counts": actual_token_counts.detach().cpu().to(torch.int32).contiguous(),
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
