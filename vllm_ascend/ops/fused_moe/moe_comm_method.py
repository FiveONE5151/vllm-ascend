# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
from __future__ import annotations

import csv
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize, PrepareAndFinalizeWithAll2All,
    PrepareAndFinalizeWithAllGather, PrepareAndFinalizeWithMC2, QuantType)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    MoETokenDispatcher, TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAllGather, TokenDispatcherWithMC2)
from vllm.logger import logger
from vllm.distributed import get_ep_group

_MoECommMethods: Dict[Optional[MoECommType], MoECommMethod] = {}
TOKEN_DROP_STATS_CSV_PATH = "/mnt/jiayihuang_fs/yiwu/research/eplb_superpod/log/tokendrop/token_drop_stats.csv"


def _is_global_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _maybe_record_token_drop_stats(
    layer_idx: Optional[int],
    num_global_experts: int,
    ids_before_drop: torch.Tensor,
    scores_after_drop: torch.Tensor,
) -> None:
    if not envs_ascend.VLLM_TOKEN_DROP_LOGGING or num_global_experts <= 0:
        return
    if not _is_global_rank0():
        return

    before_counts = torch.bincount(ids_before_drop.to(torch.int64),
                                   minlength=num_global_experts)
    kept_mask = scores_after_drop > 0
    if kept_mask.any():
        after_counts = torch.bincount(ids_before_drop[kept_mask].to(
            torch.int64),
                                      minlength=num_global_experts)
    else:
        after_counts = torch.zeros(num_global_experts,
                                   dtype=torch.int64,
                                   device=before_counts.device)

    baseline_avg = (before_counts.sum().item() / num_global_experts
                    if num_global_experts > 0 else 0.0)
    if baseline_avg > 0:
        pre_ratios = before_counts.to(torch.float32) / baseline_avg
        post_ratios = after_counts.to(torch.float32) / baseline_avg
    else:
        pre_ratios = torch.zeros_like(before_counts, dtype=torch.float32)
        post_ratios = torch.zeros_like(after_counts, dtype=torch.float32)

    os.makedirs(os.path.dirname(TOKEN_DROP_STATS_CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(TOKEN_DROP_STATS_CSV_PATH)
    write_header = (not file_exists) or os.path.getsize(
        TOKEN_DROP_STATS_CSV_PATH) == 0

    rows = []
    layer_val = int(layer_idx) if layer_idx is not None else -1
    before_list = before_counts.tolist()
    after_list = after_counts.tolist()
    pre_ratio_list = pre_ratios.tolist()
    post_ratio_list = post_ratios.tolist()
    for expert_idx in range(num_global_experts):
        rows.append({
            "layer_idx": layer_val,
            "expert_idx": expert_idx,
            "pre_drop_load": before_list[expert_idx],
            "post_drop_load": after_list[expert_idx],
            "baseline_avg_load_pre_drop": baseline_avg,
            "pre_ratio_vs_pre_drop_avg": pre_ratio_list[expert_idx],
            "post_ratio_vs_pre_drop_avg": post_ratio_list[expert_idx],
        })

    fieldnames = [
        "layer_idx",
        "expert_idx",
        "pre_drop_load",
        "post_drop_load",
        "baseline_avg_load_pre_drop",
        "pre_ratio_vs_pre_drop_avg",
        "post_ratio_vs_pre_drop_avg",
    ]
    with open(TOKEN_DROP_STATS_CSV_PATH, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _global_capacity_token_drop(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    router_logits: Optional[torch.Tensor],
    num_global_experts: int,
    load_factor: float,
    group,
    layer_idx: Optional[int] = None,
) -> torch.Tensor:

    """do token drop

    Returns:
        topk_weights: topk_weights after drop, dropped tokens' weight will be set to 0
    """
    if topk_weights.numel() == 0 or num_global_experts <= 0:
        return topk_weights

    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    flat_weights = topk_weights.reshape(-1)

    # [yiwu] local token数量， 经过topk扩展
    local_count = int(flat_ids.numel())
    if local_count == 0:
        return topk_weights

    topk = int(topk_weights.size(-1))

    def _apply_capacity(ids: torch.Tensor, scores: torch.Tensor,
                        pair_count: int) -> torch.Tensor:
        if pair_count == 0:
            return scores
        if topk <= 0 or pair_count % topk != 0:
            return scores

        num_tokens = pair_count // topk
        ids_2d = ids.view(num_tokens, topk).to(torch.int64)
        scores_2d = scores.view(num_tokens, topk)

        logits_dtype = router_logits.dtype if router_logits is not None else scores_2d.dtype
        router_logits_ref = torch.zeros((num_tokens, num_global_experts),
                                        device=scores_2d.device,
                                        dtype=logits_dtype)

        topk_masked_scores = torch.zeros_like(router_logits_ref).scatter(
            1, ids_2d, scores_2d)
        topk_mask = torch.zeros_like(router_logits_ref,
                                     dtype=torch.int32).scatter(
                                         1, ids_2d, 1).bool()

        expert_capacity = math.ceil((num_tokens * topk /
                                     float(num_global_experts)) * load_factor)
        expert_capacity = min(max(expert_capacity, 0), num_tokens)
        if expert_capacity == 0:
            return torch.zeros_like(scores)

        _, capacity_indices = torch.topk(topk_masked_scores,
                                         k=expert_capacity,
                                         dim=0,
                                         sorted=False)
        capacity_mask = torch.zeros_like(router_logits_ref,
                                         dtype=torch.bool).scatter(
                                             0, capacity_indices, True)
        final_mask = topk_mask & capacity_mask

        final_scores = scores_2d.masked_fill(~final_mask.gather(1, ids_2d),
                                             0.0)
        return final_scores.reshape(-1)

    if (not dist.is_available()) or (not dist.is_initialized()):
        dropped_scores = _apply_capacity(flat_ids, flat_weights, local_count)
        _maybe_record_token_drop_stats(layer_idx, num_global_experts, flat_ids,
                                       dropped_scores)
        return dropped_scores.reshape_as(topk_weights)

    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        dropped_scores = _apply_capacity(flat_ids, flat_weights, local_count)
        _maybe_record_token_drop_stats(layer_idx, num_global_experts, flat_ids,
                                       dropped_scores)
        return dropped_scores.reshape_as(topk_weights)
    rank = dist.get_rank(group=group)

    counts = [local_count for _ in range(world_size)]

    gathered_ids = [torch.empty_like(flat_ids) for _ in range(world_size)]
    gathered_weights = [
        torch.empty_like(flat_weights) for _ in range(world_size)
    ]
    global_ids = get_ep_group().all_gather(flat_ids, dim=0)
    global_weights = get_ep_group().all_gather(flat_weights, dim=0)
    # dist.all_gather(gathered_ids, flat_ids, group=group)
    # dist.all_gather(gathered_weights, flat_weights, group=group)

    # global_ids = torch.cat(gathered_ids, dim=0)
    # global_weights = torch.cat(gathered_weights, dim=0)

    global_pairs = int(global_ids.numel())
    if global_pairs == 0:
        return topk_weights

    dropped_global_weights = _apply_capacity(global_ids, global_weights,
                                             global_pairs)
    _maybe_record_token_drop_stats(layer_idx, num_global_experts, global_ids,
                                   dropped_global_weights)
    keep_mask_global = dropped_global_weights > 0

    local_offset = sum(counts[:rank])
    local_keep = keep_mask_global[local_offset:local_offset + local_count]
    local_keep = local_keep.reshape_as(topk_weights)

    return topk_weights.masked_fill(~local_keep, 0.0)


def get_moe_comm_method(
        moe_comm_type: Optional[MoECommType]) -> Optional[MoECommMethod]:
    return _MoECommMethods.get(moe_comm_type, None)


def setup_moe_comm_method(moe_config):
    _MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
    _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)
    _MoECommMethods[MoECommType.MC2] = MC2CommImpl(moe_config)
    _MoECommMethods[MoECommType.FUSED_MC2] = FusedMC2CommImpl(moe_config)


@dataclass
class FusedExpertsResult:
    routed_out: torch.Tensor
    # This field is for shared experts and should be set by the MoE
    # communication method that supports shared experts in parallel with routed
    # experts.
    before_dispatch_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    # For dynamic_eplb
    group_list_type: int | None = None
    expert_tokens: torch.Tensor | None = None


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        hidden_states, router_logits, mc2_mask, context_metadata = self.prepare_finalize.prepare(
            hidden_states, router_logits, enable_shared_expert_dp,
            replace_allreduce, quant_type)
        return hidden_states, router_logits, mc2_mask, context_metadata

    def finalize(self,
                 hidden_states: torch.Tensor,
                 reduce_results: bool,
                 context_metadata: Optional[dict] = None) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(hidden_states,
                                                       reduce_results,
                                                       context_metadata)
        return hidden_states

    def fused_experts(
            self,
            hidden_states: torch.Tensor,
            w1: torch.Tensor | list[torch.Tensor],
            w2: torch.Tensor | list[torch.Tensor],
            router_logits: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            activation: str = "silu",
            apply_router_weight_on_input: bool = False,
            use_int8_w8a8: bool = False,
            use_int4_w4a8: bool = False,
            use_int4_w4a16: bool = False,
            expert_map: Optional[torch.Tensor] = None,
            w1_scale: Optional[list[torch.Tensor]] = None,
            w2_scale: Optional[list[torch.Tensor]] = None,
            w1_scale_bias: torch.Tensor = None,
            w2_scale_bias: torch.Tensor = None,
            w1_offset: Optional[torch.Tensor] = None,
            w2_offset: Optional[torch.Tensor] = None,
            # For load balance
            log2phy: torch.Tensor = None,
            need_trans: bool = False,
            dynamic_eplb: bool = False,
            mc2_mask: torch.Tensor = None,
            pertoken_scale: Optional[torch.Tensor] = None):
        # Check constraints
        assert hidden_states.dtype in [
            torch.float32, torch.float16, torch.bfloat16, torch.int8
        ]

        moe_comm_method = get_forward_context().moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        # Apply log2phy if needed
        if log2phy is not None:
            topk_ids = log2phy[topk_ids]

        ### [yiwu] token drop begin
        forward_context = get_forward_context()
        if envs_ascend.VLLM_ENABLE_TOKEN_DROP and forward_context.moe_comm_type in {
                MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2
        }:
            topk_weights = _global_capacity_token_drop(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
                num_global_experts=self.moe_config.num_experts,
                load_factor=envs_ascend.VLLM_TOKEN_DROP_LOAD_FACTOR,
                group=get_ep_group().device_group,
                layer_idx=getattr(forward_context, "layer_idx", None),
            )

        ### [yiwu] token drop end

        before_dispatch_evt = torch.npu.current_stream().record_event()

        # all2all and mc2 prepare will pad tokens and split it, take only part of global tokens
        # logger.info(f"[yiwu][DEBUG][Rank {get_ep_group().rank}] num of tokens before dispatch: {hidden_states.size(0)}")

        dispatch_results = self.token_dispatcher.token_dispatch(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            global_redundant_expert_num=self.moe_config.
            global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            with_quant=use_int8_w8a8 or use_int4_w4a8,
            dynamic_eplb=dynamic_eplb,
            pertoken_scale=pertoken_scale)

        # [yiwu] the hiddenstates are permuted and only contain the tokens for local experts, 
        # so the num of tokens may change after dispatch, log it here for debugging and analysis.
        # logger.info(f"[yiwu][DEBUG][Rank {get_ep_group().rank}] using comm method: {moe_comm_method.__class__.__name__}")
        # logger.info(f"[yiwu][DEBUG][Rank {get_ep_group().rank}] num of tokens after dispatch: {dispatch_results.hidden_states.size(0)}")

        mlp_output = unified_apply_mlp(
            hidden_states=dispatch_results.hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            group_list=dispatch_results.group_list,
            dynamic_scale=dispatch_results.dynamic_scale,
            group_list_type=dispatch_results.group_list_type,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
            topk_scales=dispatch_results.topk_scales,
            with_quant=use_int8_w8a8 or use_int4_w4a8 or use_int4_w4a16,
            fusion=use_int8_w8a8,
            need_trans=need_trans,
            dynamic_eplb=dynamic_eplb)

        before_combine_evt = torch.npu.current_stream().record_event()
        combine_results = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            context_metadata=dispatch_results.context_metadata)

        return FusedExpertsResult(
            routed_out=combine_results.routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=dispatch_results.group_list_type,
            expert_tokens=dispatch_results.group_list)

    @abstractmethod
    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        raise NotImplementedError(
            "_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        raise NotImplementedError(
            "_get_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts)

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.
    
    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_grouped_matmul` is available.

    This implementation uses all-to-all communication to exchange tokens
    between data parallel ranks before and after the MLP computation. It should
    have better performance than AllGatherCommImpl when DP size > 1.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts)

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)


class FusedMC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.
    
    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def fused_experts(
            self,
            hidden_states: torch.Tensor,
            w1: torch.Tensor | list[torch.Tensor],
            w2: torch.Tensor | list[torch.Tensor],
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            router_logits: Optional[torch.Tensor] = None,
            activation: str = "silu",
            apply_router_weight_on_input: bool = False,
            use_int8_w8a8: bool = False,
            use_int4_w4a8: bool = False,
            use_int4_w4a16: bool = False,
            expert_map: Optional[torch.Tensor] = None,
            w1_scale: Optional[list[torch.Tensor]] = None,
            w2_scale: Optional[list[torch.Tensor]] = None,
            w1_scale_bias: torch.Tensor = None,
            w2_scale_bias: torch.Tensor = None,
            w1_offset: Optional[torch.Tensor] = None,
            w2_offset: Optional[torch.Tensor] = None,
            # For load balance
            log2phy: torch.Tensor = None,
            need_trans: bool = False,
            dynamic_eplb: bool = False,
            mc2_mask: torch.Tensor = None,
            pertoken_scale: Optional[torch.Tensor] = None):
        assert not (
            w1_scale is None or w2_scale is None
        ), "w1_scale and w2_scale cannot be None for FusedMC2CommImpl."

        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), \
            "token_dispatcher must be an instance of TokenDispatcherWithMC2."

        # Apply log2phy if needed
        if log2phy is not None:
            topk_ids = log2phy[topk_ids]

        ### [yiwu] token drop begin
        forward_context = get_forward_context()
        if envs_ascend.VLLM_ENABLE_TOKEN_DROP and forward_context.moe_comm_type in {
                MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2
        }:
            topk_weights = _global_capacity_token_drop(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
                num_global_experts=self.moe_config.num_experts,
                load_factor=envs_ascend.VLLM_TOKEN_DROP_LOAD_FACTOR,
                group=get_ep_group().device_group,
                layer_idx=getattr(forward_context, "layer_idx", None),
            )
        ### [yiwu] token drop end

        group_list_type = None
        expert_tokens = None
        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
            out = torch.empty_like(hidden_states)
            torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                x=hidden_states,
                weight1=w1,
                weight2=w2,
                expert_idx=topk_ids,
                scale1=w1_scale,
                scale2=w2_scale,
                probs=topk_weights.to(torch.float32),
                group=self.token_dispatcher.moe_all_to_all_group_name,
                max_output_size=65536,
                out=out,
            )
        elif envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2:
            assert expert_map is not None, "expert_map cannot be None."
            group_list_type = 1
            out, expert_tokens = torch.ops._C_ascend.dispatch_gmm_combine_decode(  # type: ignore
                x=hidden_states,
                expert_ids=topk_ids,
                gmm1_permuted_weight=w1,
                gmm1_permuted_weight_scale=w1_scale,
                gmm2_weight=w2,
                gmm2_weight_scale=w2_scale,
                expert_smooth_scales=None,
                expert_scales=topk_weights.to(torch.float32),
                group_ep=self.token_dispatcher.moe_all_to_all_group_name,
                ep_rank_size=self.token_dispatcher.ep_world_size,
                ep_rank_id=self.token_dispatcher.ep_rank_id,
                moe_expert_num=self.moe_config.num_experts,
                global_bs=self.token_dispatcher.global_bs)
        else:
            raise ValueError(
                f"Wrong value of {envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2=}")
        return FusedExpertsResult(routed_out=out,
                                  group_list_type=group_list_type,
                                  expert_tokens=expert_tokens)
