# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass, field
import math
from typing import Optional
from sympy import O
import torch_npu.distributed
from typing_extensions import override

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.comm_utils import (
    async_all_to_all, gather_from_sequence_parallel_region)
from vllm_ascend.ops.fused_moe.token_drop_utils import \
    compute_num_global_tokens_per_expert_after_drop
from vllm_ascend.utils import (AscendDeviceType, get_ascend_device_type,
                               is_hierarchical_communication_enabled)

from vllm.forward_context import get_forward_context
import os
from vllm.logger import logger
@dataclass
class TokenDispatchResult:
    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    dynamic_scale: torch.Tensor | None = field(default=None)
    topk_scales: torch.Tensor | None = field(default=None)
    context_metadata: dict = field(default_factory=dict)


@dataclass
class TokenCombineResult:
    routed_out: torch.Tensor


class MoETokenDispatcher(ABC):

    def __init__(self, **kwargs) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.top_k = kwargs.get("top_k", 0)

        # num of global experts
        self.num_experts = kwargs.get("num_experts", 0)

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return get_ep_group().device_group

    @property
    def ep_rank(self):
        return get_ep_group().rank_in_group

    @property
    def ep_size(self):
        return get_ep_group().world_size

    @abstractmethod
    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: Optional[torch.Tensor] = None,
        global_redundant_expert_num: int = 0,
        mc2_mask: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        with_quant: bool = False,
        dynamic_eplb: bool = False,
        pertoken_scale: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
    ) -> TokenDispatchResult:
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_combine(self,
                      hidden_states: torch.Tensor,
                      context_metadata: dict,
                      bias: torch.Tensor | None = None) -> TokenCombineResult:
        raise NotImplementedError("Combine function not implemented.")


class TokenDispatcherWithMC2(MoETokenDispatcher):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device_group = get_mc2_group().device_group
        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        self.ep_rank_id = get_mc2_group().rank_in_group
        self.ep_world_size = get_mc2_group().world_size
        self.enable_dispatch_v2 = hasattr(torch_npu,
                                          "npu_moe_distribute_dispatch_v2")
        self.need_extra_args = (
            get_ascend_device_type() == AscendDeviceType.A3)

        # NOTE: When in A2, setting the environment variables HCCL_INTRA_PCIE_ENABLE=1 and
        # HCCL_INTRA_ROCE_ENABLE=0 can reduce cross-machine communication traffic and significantly
        # improve communication performance.
        self.need_expert_scale = is_hierarchical_communication_enabled()
        self.with_quant = False

        # Here we need to calculate the global_bs = max_bs_per_rank * ep_world_size to execute
        # dispatch & combine operators with different input num_tokens per rank.
        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config
        compilation_config = vllm_config.compilation_config
        speculative_config = vllm_config.speculative_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        uniform_decode_query_len = 1 if not speculative_config else \
            1 + speculative_config.num_speculative_tokens
        decode_max_num_seqs = getattr(scheduler_config, 'decode_max_num_seqs',
                                      0)
        max_num_reqs = max(scheduler_config.max_num_seqs, decode_max_num_seqs)
        if compilation_config.cudagraph_capture_sizes:
            max_num_tokens = compilation_config.max_cudagraph_capture_size
        else:
            max_num_tokens = min(max_num_reqs * uniform_decode_query_len, 512)
        num_tokens_per_tp_rank = (max_num_tokens + tp_size - 1) // tp_size
        self.global_bs = num_tokens_per_tp_rank * self.ep_world_size

    def get_dispatch_mc2_kwargs(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor,
        mc2_mask: torch.Tensor,
        global_redundant_expert_num: int = 0,
    ):
        quant_mode = 2 if self.with_quant else 0
        self.moe_expert_num = len(expert_map) + global_redundant_expert_num
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
            "expert_token_nums_type": 0,
        }

        stage1_kwargs = {
            "scales": None,
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }
        if self.need_extra_args:
            stage1_kwargs.update({
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        if self.need_expert_scale:
            stage1_kwargs.update({
                "expert_scales":
                topk_weights.to(torch.float32),
            })

        kwargs_mc2.update(stage1_kwargs)
        return kwargs_mc2

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False,
                       dynamic_eplb: bool = False,
                       pertoken_scale: Optional[torch.Tensor] = None,
                       router_logits: Optional[torch.Tensor] = None):
        self.with_quant = with_quant

        kwargs_mc2 = self.get_dispatch_mc2_kwargs(hidden_states, topk_weights,
                                                  topk_ids, expert_map,
                                                  mc2_mask,
                                                  global_redundant_expert_num)
        output = torch_npu.npu_moe_distribute_dispatch_v2(
            **kwargs_mc2
        ) if self.enable_dispatch_v2 else torch_npu.npu_moe_distribute_dispatch(
            **kwargs_mc2)
        # comm_stream.wait_stream(torch.npu.current_stream())
        expand_x, dynamic_scale, assist_info_for_combine, expert_token_nums, \
            ep_recv_counts, tp_recv_counts, expand_scales = output[0:7]

        context_metadata = {
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "expert_map": expert_map,
            "ep_recv_counts": ep_recv_counts,
            "tp_recv_counts": tp_recv_counts,
            "assist_info_for_combine": assist_info_for_combine,
            "expand_scales": expand_scales
        }

        group_list_type = 0
        return TokenDispatchResult(hidden_states=expand_x,
                                   dynamic_scale=dynamic_scale,
                                   group_list=expert_token_nums,
                                   group_list_type=group_list_type,
                                   context_metadata=context_metadata)

    def get_combine_mc_kwargs(self, hidden_states: torch.Tensor,
                              context_metadata: dict):
        expert_map = context_metadata["expert_map"]
        topk_ids = context_metadata["topk_ids"]
        topk_weights = context_metadata["topk_weights"]
        ep_recv_counts = context_metadata["ep_recv_counts"]
        tp_recv_counts = context_metadata["tp_recv_counts"]
        assist_info_for_combine = context_metadata["assist_info_for_combine"]
        expand_scales = context_metadata["expand_scales"]

        assert expert_map is not None

        kwargs_mc2 = {
            "expand_x": hidden_states,
            "expert_ids": topk_ids,
            "expert_scales": topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
        }

        if self.with_quant:
            tp_recv_counts = torch.empty(1,
                                         dtype=torch.int32,
                                         device=hidden_states.device)

        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
            "expand_scales": expand_scales,
        }

        if self.enable_dispatch_v2:
            stage3_kwargs["assist_info_for_combine"] = assist_info_for_combine
        else:
            stage3_kwargs["expand_idx"] = assist_info_for_combine

        if self.need_extra_args:
            stage3_kwargs.update({
                "tp_send_counts": tp_recv_counts,
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })

        kwargs_mc2.update(stage3_kwargs)
        return kwargs_mc2

    def token_combine(self, hidden_states, context_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        kwargs_mc2 = self.get_combine_mc_kwargs(hidden_states,
                                                context_metadata)
        combined_output = torch_npu.npu_moe_distribute_combine_v2(**kwargs_mc2) \
            if self.enable_dispatch_v2 else torch_npu.npu_moe_distribute_combine(**kwargs_mc2)

        return TokenCombineResult(routed_out=combined_output, )


class TokenDispatcherWithAllGather(MoETokenDispatcher):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_router_weight_on_input = False
        self.max_num_tokens = kwargs.get("max_num_tokens")
        num_experts_local = kwargs.get("num_local_experts", 0)
        self.num_experts_local = num_experts_local.item() if torch.is_tensor(
            num_experts_local) else int(num_experts_local)
        self.original_shape = None
        self.with_quant = False

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False,
                       dynamic_eplb: bool = False,
                       pertoken_scale: Optional[torch.Tensor] = None,
                       router_logits: Optional[torch.Tensor] = None):
        self.with_quant = with_quant
        self.original_shape = hidden_states.shape

        num_tokens = hidden_states.shape[:-1].numel()
        self.apply_router_weight_on_input = apply_router_weight_on_input
        if self.apply_router_weight_on_input:
            assert (topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * \
                topk_weights.to(hidden_states.dtype)
        if expert_map is not None:
            global_num_experts = len(expert_map) + global_redundant_expert_num
            mask = (expert_map[topk_ids] != -1)
            topk_weights = topk_weights * mask
            first_expert_idx = get_ep_group(
            ).rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local
            global_num_experts = self.num_experts_local

        sorted_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
            torch.ops._C_ascend.npu_moe_init_routing_custom(
                hidden_states,
                topk_ids,
                scale=pertoken_scale,
                active_num=num_tokens * self.top_k,
                expert_num=global_num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[first_expert_idx, last_expert_idx],
                quant_mode=1
                if self.with_quant and pertoken_scale is None else -1,
            ))
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1  # `count` mode
        context_metadata = {
            "topk_weights": topk_weights,
            "expanded_row_idx": expanded_row_idx
        }

        return TokenDispatchResult(
            hidden_states=sorted_hidden_states,
            dynamic_scale=pertoken_scale if self.with_quant else None,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            context_metadata=context_metadata,
        )

    def token_combine(self, hidden_states, context_metadata, bias=None):
        assert self.original_shape is not None
        final_hidden_states = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=torch.abs(context_metadata["expanded_row_idx"]),
            probs=context_metadata["topk_weights"])
        if len(self.original_shape) == 3:
            final_hidden_states = final_hidden_states.view(self.original_shape)

        # these values are no longer used, so they need to be set to None for memory release.
        return TokenCombineResult(routed_out=final_hidden_states)


class TokenDispatcherWithAll2AllV(MoETokenDispatcher):
    """
    The implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.with_quant = False
        self.num_local_experts = kwargs.get("num_local_experts", 0)

        self.hidden_shape = None
        self.hidden_shape_before_permute = None

        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int64,
                device=torch.npu.current_device(),
            )

        local_expert_indices_offset = (self.ep_rank * self.num_local_experts)

        self.local_expert_indices = [
            local_expert_indices_offset + i
            for i in range(self.num_local_experts)
        ]
        assert (len(self.local_expert_indices) == self.num_local_experts
                ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (self.local_expert_indices[i] ==
                    self.local_expert_indices[i + 1] -
                    1), "local_expert_indices must be continuous"

        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=self.ep_group)
        backend = self.ep_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False,
                       dynamic_eplb: bool = False,
                       pertoken_scale: Optional[torch.Tensor] = None,
                       router_logits: Optional[torch.Tensor] = None):

        """token dispatch for all2all

        Returns:
            1) 对外返回：TokenDispatchResult
                hidden_states：global_input_tokens，已经完成本地 permute + 跨 EP all2all + 本地按 expert 重排后的张量；形状通常是 [本 rank 收到并分配给本地 experts 的 token 总数, hidden]。
                dynamic_scale：dynamic_scale_final。仅量化开启(with_quant=True)时有效；否则是 None。
                group_list：tokens_per_expert，表示本 rank 上每个 local expert 的 token 数（count 列表）。
                group_list_type：固定为 1，表示 group_list 是 “count mode”（不是前缀和/offset 模式）。
                context_metadata：给 token_combine 用的上下文（见第 2 点）。
                
            2) context_metadata 字段含义
                input_splits：本 rank 在 all2all 中发往各对端 rank 的 token 数（send split）。
                output_splits：本 rank 在 all2all 中从各对端 rank 接收的 token 数（recv split）。
                topk_weights：router 给每个 token 的 top-k 权重，combine 时用于加权还原。
                reversed_local_input_permutation_mapping：第一次本地 npu_moe_token_permute 的逆映射（最终恢复原 token 顺序用）。
                reversed_global_input_permutation_mapping：第二次“按本地 expert 分组”重排的逆映射；仅 num_local_experts > 1 时有值，否则为 None。
        """
        self.with_quant = with_quant
        self.hidden_shape = hidden_states.shape

        # [yiwu] get token stats, and permuted tokens
        (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
        ) = self._dispatch_preprocess(hidden_states, topk_ids)

        dynamic_scale_after_all2all = None
        if self.with_quant:
            permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_local_input_tokens)
            _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
                dynamic_scale, output_splits, input_splits, self.ep_group)
            permute2_ep_all_to_all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        # [yiwu] perform all2all communication
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens, output_splits, input_splits,
            self.ep_group)
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # Postprocess
        # [yiwu] do permutation to place tokens by local experts order
        global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = self._dispatch_postprocess(
            global_input_tokens, dynamic_scale_after_all2all,
            global_input_tokens_local_experts_indices)

        context_metadata = {
            "input_splits":
            input_splits,
            "output_splits":
            output_splits,
            "topk_weights":
            topk_weights,
            "reversed_local_input_permutation_mapping":
            reversed_local_input_permutation_mapping,
            "reversed_global_input_permutation_mapping":
            reversed_global_input_permutation_mapping
        }

        return TokenDispatchResult(
            hidden_states=global_input_tokens,
            dynamic_scale=dynamic_scale_final,
            group_list=tokens_per_expert,
            group_list_type=1,
            context_metadata=context_metadata,
        )

    def token_combine(self, hidden_states, context_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        # 1. Preprocess using metadata
        hidden_states = self._combine_preprocess(hidden_states,
                                                 context_metadata)

        # 2. AllToAll
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            context_metadata["input_splits"],
            context_metadata["output_splits"],
            self.ep_group,
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        # 3. Postprocess using metadata
        output = self._combine_postprocess(permutated_local_input_tokens,
                                           context_metadata)

        return TokenCombineResult(routed_out=output)

    def _dispatch_preprocess(self, hidden_states, topk_ids):
        assert self.hidden_shape is not None
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        (
            tokens_per_expert,
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
        ) = self._preprocess(topk_ids)

        self.hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=self.num_out_tokens,
        )

        return (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
        )

    def _preprocess(self, topk_ids: torch.Tensor):
        """
        - `input_splits`: 本rank向其他rank发送的token数量, 形状为 `[ep_size]`
        - `output_splits`: 本rank接收其他rank的token数量, 形状为 `[ep_size]`
        - `num_global_tokens_per_local_expert`: 本地专家接接收其他rank的token数量, 形状为 `[ep_size, num_local_Experts]`
        - `num_tokens_per_local_expert`: [num_local_experts,], 本地专家接收的全局token数量
        - `global_input_tokens_local_experts_indices`: 根据每个local expert的global token数量，生成每个global token对应的local expert id; [num_local_processed_tokens,]
        """
        num_local_tokens_per_expert = torch.histc(topk_ids,
                                                  bins=self.num_experts,
                                                  min=0,
                                                  max=self.num_experts)

        ep_size = self.ep_size
        self.num_out_tokens = topk_ids.numel()

        input_splits = (num_local_tokens_per_expert.reshape(
            ep_size,
            self.num_local_experts).sum(axis=1).to(torch.device("cpu"),
                                                   non_blocking=True).numpy())

        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert,
            group=self.ep_group).reshape(ep_size, self.num_experts)
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices[
            0]:self.local_expert_indices[-1] + 1]
        if num_global_tokens_per_local_expert is None:
            raise ValueError(
                "num_global_tokens_per_local_expert must be set before sum.")

        output_splits = (num_global_tokens_per_local_expert.sum(axis=-1).to(
            torch.device("cpu"), non_blocking=True).numpy())
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(
            axis=0)

        global_input_tokens_local_experts_indices = None
        if self.num_local_experts > 1:
            if num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                num_global_tokens_per_local_expert.ravel())
        else:
            torch.npu.synchronize()

        return (
            num_tokens_per_local_expert,
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
        )

    def _dispatch_postprocess(self, global_input_tokens,
                              dynamic_scale_after_all2all,
                              global_input_tokens_local_experts_indices):
        # Early return if no local experts or no tokens
        if self.num_local_experts <= 1:
            return global_input_tokens, dynamic_scale_after_all2all, None

        # Handle quantized case
        if self.with_quant:
            assert global_input_tokens_local_experts_indices is not None, \
                "global_input_tokens_local_experts_indices must be provided"
            dynamic_scale_after_all2all, _ = torch_npu.npu_moe_token_permute(
                dynamic_scale_after_all2all.unsqueeze(-1),
                global_input_tokens_local_experts_indices)
            dynamic_scale_after_all2all = dynamic_scale_after_all2all.squeeze(
                -1)

        # Non-quantized case
        global_input_tokens, reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens, global_input_tokens_local_experts_indices)
        return global_input_tokens, dynamic_scale_after_all2all, reversed_global_input_permutation_mapping

    def _combine_preprocess(self, hidden_states: torch.Tensor,
                            context_metadata: dict) -> torch.Tensor:
        # Unpermutation 2: expert output to AlltoAll input
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1:
            rev_global = context_metadata[
                "reversed_global_input_permutation_mapping"]
            hidden_states = torch_npu.npu_moe_token_unpermute(
                hidden_states, rev_global)
        return hidden_states

    def _combine_postprocess(self, permutated_local_input_tokens: torch.Tensor,
                             context_metadata: dict) -> torch.Tensor:
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=context_metadata[
                "reversed_local_input_permutation_mapping"].to(torch.int32),
            probs=context_metadata["topk_weights"],
            restore_shape=self.hidden_shape_before_permute,
        )
        output = output.view(self.hidden_shape)
        return output


class TokenDispatcherWithAll2AllvUnified(TokenDispatcherWithAll2AllV):
    """
    Unified dispatcher that handles both regular and token drop cases.

    This dispatcher ONLY handles communication (permute + all2all + unpermute).
    Token drop logic is applied in experts_selector via TokenDropStrategy,
    so the dispatcher receives pre-processed topk_ids and topk_weights.

    Key features:
    - Handles both [T, topk] (non-expanded) and [T, topk+N] (expanded) shapes
    - Dropped tokens have expert index set to num_experts (sentinel)
    - Computes splits based on valid tokens (non-sentinel)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_expanded = kwargs.get("is_expanded", False)
        self.expanded_topk = (self.top_k + self.num_local_experts
                              if self.is_expanded else self.top_k)
        self.num_out_tokens_before_drop = None

        logger.info(
            f"[UnifiedDispatcher] Initialized with is_expanded={self.is_expanded}, "
            f"top_k={self.top_k}, expanded_topk={self.expanded_topk}, "
            f"num_local_experts={self.num_local_experts}"
        )

    def token_dispatch(
            self,
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            expert_map: Optional[torch.Tensor] = None,
            global_redundant_expert_num: int = 0,
            mc2_mask: Optional[torch.Tensor] = None,
            apply_router_weight_on_input: bool = False,
            with_quant: bool = False,
            dynamic_eplb: bool = False,
            pertoken_scale: Optional[torch.Tensor] = None,
            router_logits: Optional[torch.Tensor] = None):
        """
        Token dispatch for unified dispatcher.

        topk_ids already processed by experts_selector (with token drop applied if enabled).
        - Non-expanded: [T, topk] shape
        - Expanded: [T, topk + num_local_experts] shape
        Dropped tokens have expert index set to num_experts (sentinel).

        Pipeline:
        1. Compute splits from topk_ids (already dropped)
        2. Permute (all token-expert pairs including sentinels)
        3. Truncate kept tokens (sentinel tokens grouped at end in permuted buffer)
        4. All2all
        5. Postprocess (local expert grouping)
        """
        self.with_quant = with_quant
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        self.hidden_shape_before_permute = hidden_states.shape

        # Compute preprocessing based on pre-dropped topk_ids
        (tokens_per_expert,
         input_splits,
         output_splits,
         num_global_tokens_per_local_expert,
         global_input_tokens_local_experts_indices,
         num_out_tokens_before_drop,
         num_out_tokens_after_drop) = self._preprocess_with_dropped_topk(topk_ids)

        self.num_out_tokens_before_drop = num_out_tokens_before_drop
        self.num_out_tokens = num_out_tokens_after_drop

        # Permute - includes all token-expert pairs (sentinel pairs grouped at end)
        permutated_local_input_tokens, reversed_local_input_permutation_mapping = \
            torch_npu.npu_moe_token_permute(
                tokens=hidden_states,
                indices=topk_ids,
                num_out_tokens=num_out_tokens_before_drop,
            )

        # Truncate to kept tokens (sentinel pairs are at the end of permuted buffer)
        permutated_local_input_tokens = permutated_local_input_tokens[:self.num_out_tokens]

        # Store full reversed mapping for combine
        full_reversed_local_input_permutation_mapping = reversed_local_input_permutation_mapping

        dynamic_scale_after_all2all = None
        if self.with_quant:
            permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_local_input_tokens)
            _, dynamic_scale_after_all2all, quant_all2all_handle = async_all_to_all(
                dynamic_scale, output_splits, input_splits, self.ep_group)
            quant_all2all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        # All2all communication
        _, global_input_tokens, all2all_handle = async_all_to_all(
            permutated_local_input_tokens, output_splits, input_splits, self.ep_group)
        all2all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # Postprocess - local expert grouping
        global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = \
            self._dispatch_postprocess(
                global_input_tokens,
                dynamic_scale_after_all2all,
                global_input_tokens_local_experts_indices,
            )

        # Build context metadata
        context_metadata = {
            "input_splits": input_splits,
            "output_splits": output_splits,
            "topk_weights": topk_weights,
            "full_reversed_local_input_permutation_mapping": full_reversed_local_input_permutation_mapping,
            "reversed_global_input_permutation_mapping": reversed_global_input_permutation_mapping,
            "num_out_tokens_before_drop": num_out_tokens_before_drop,
            "num_out_tokens_after_drop": num_out_tokens_after_drop,
        }

        return TokenDispatchResult(
            hidden_states=global_input_tokens,
            dynamic_scale=dynamic_scale_final,
            group_list=tokens_per_expert,
            group_list_type=1,
            context_metadata=context_metadata,
        )

    def _preprocess_with_dropped_topk(self, topk_ids: torch.Tensor):
        """
        Preprocess with already-dropped topk_ids.

        topk_ids may contain sentinel values (num_experts) for dropped tokens.
        Compute splits based on valid (non-sentinel) tokens.

        Returns:
            num_tokens_per_local_expert: [num_local_experts]
            input_splits: numpy array [ep_size]
            output_splits: numpy array [ep_size]
            num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            global_input_tokens_local_experts_indices: [num_received_tokens] or None
            num_out_tokens_before_drop: T_local * expanded_topk (or topk)
            num_out_tokens_after_drop: actual kept token count
        """
        T_local = topk_ids.shape[0]
        actual_topk = topk_ids.shape[1]  # Could be topk or topk + num_local_experts

        # Keep the same global-stat flow as token-drop/expanded-drop paths.
        try:
            num_tokens_across_dp = get_forward_context(
            ).dp_metadata.num_tokens_across_dp_cpu
        except AssertionError:
            num_tokens_across_dp = torch.full(
                (self.ep_size, ),
                T_local,
                dtype=torch.int64,
                device=torch.device("cpu"),
            )

        T_global = int(num_tokens_across_dp.sum().item())
        global_topk_ids_after_drop = torch.zeros(
            (T_global, actual_topk),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )

        # all_gather_into_tensor_uneven requires contiguous input/output tensors.
        topk_ids_contiguous = topk_ids.contiguous()
        torch_npu.distributed.all_gather_into_tensor_uneven(
            global_topk_ids_after_drop,
            topk_ids_contiguous,
            num_tokens_across_dp.numpy(),
            group=self.ep_group,
        )

        num_global_tokens_per_expert = compute_num_global_tokens_per_expert_after_drop(
            global_topk_ids_after_drop,
            num_tokens_across_dp,
            self.num_experts,
            self.ep_size,
        )

        # Extract local expert columns
        local_expert_start = self.local_expert_indices[0]
        local_expert_end = self.local_expert_indices[-1] + 1
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, local_expert_start:local_expert_end]

        # Compute splits
        # output_splits: tokens received per rank (sent to local experts)
        output_splits = num_global_tokens_per_local_expert.sum(dim=-1).to(
            torch.device("cpu"), non_blocking=True)

        # input_splits: tokens sent per rank
        input_splits = num_global_tokens_per_expert[self.ep_rank].reshape(
            self.ep_size, self.num_local_experts).sum(dim=1).to(
                torch.device("cpu"), non_blocking=True)

        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)

        # Compute global_input_tokens_local_experts_indices
        global_input_tokens_local_experts_indices = None
        if self.num_local_experts > 1:
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                num_global_tokens_per_local_expert.flatten()
            )

        # Compute token counts
        num_out_tokens_before_drop = T_local * actual_topk
        num_out_tokens_after_drop = int(input_splits.sum().item())

        return (
            num_tokens_per_local_expert,
            input_splits.numpy(),
            output_splits.numpy(),
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
            num_out_tokens_before_drop,
            num_out_tokens_after_drop,
        )

    def token_combine(self, hidden_states, context_metadata, bias=None):
        """Token combine for unified dispatcher."""
        assert bias is None

        # 1. Preprocess - undo local expert grouping
        hidden_states = self._combine_preprocess(hidden_states, context_metadata)

        # 2. All2all
        _, permutated_local_input_tokens, all2all_handle = async_all_to_all(
            hidden_states,
            context_metadata["input_splits"],
            context_metadata["output_splits"],
            self.ep_group,
        )
        all2all_handle.wait()
        hidden_states.untyped_storage().resize_(0)

        # 3. Reconstruct full permuted tensor
        num_out_tokens_before_drop = context_metadata["num_out_tokens_before_drop"]

        restored_local_tokens = torch.zeros(
            (num_out_tokens_before_drop, permutated_local_input_tokens.shape[-1]),
            dtype=permutated_local_input_tokens.dtype,
            device=permutated_local_input_tokens.device,
        )

        restored_local_tokens[:self.num_out_tokens] = permutated_local_input_tokens
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # 4. Unpermute with full reversed mapping
        full_reversed_mapping = context_metadata["full_reversed_local_input_permutation_mapping"]

        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=restored_local_tokens,
            sorted_indices=full_reversed_mapping.to(torch.int32),
            probs=context_metadata["topk_weights"],
            restore_shape=self.hidden_shape_before_permute,
        )

        output = output.view(self.hidden_shape)
        return TokenCombineResult(routed_out=output)


class TokenDispatcherWithAll2AllvTokenDrop(TokenDispatcherWithAll2AllV):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token_drop_load_factor = kwargs.get("token_drop_load_factor", 1.0)
        self.token_drop_local_only = kwargs.get("token_drop_local_only",
                                                False)
        self.drop_by_expert = os.getenv("VLLM_TOKEN_DROP_BY_EXPERT", "0") == "1"
        self.drop_strategy = os.getenv("VLLM_TOKEN_DROP_STRATEGY", "expert_drop") # expert_drop, device_drop, expert_expanded_drop 
        self.token_drop_logging = os.getenv("VLLM_TOKEN_DROP_LOGGING", "0") == "1"
        self.token_drop_csv_dir = os.getenv("VLLM_TOKEN_DROP_CSV_DIR", "")
        self.token_drop_step = 0

        logger.info(f"[TokenDrop]Initialized TokenDispatcherWithAll2AllvTokenDrop with token_drop_load_factor={self.token_drop_load_factor}, ")
        logger.info(
            f"[TokenDrop]drop_by_expert={self.drop_by_expert}, token_drop_logging={self.token_drop_logging}, token_drop_local_only={self.token_drop_local_only}"
        )

    def _log_token_drop_statistics(
            self,
            num_global_tokens_per_expert_before_drop: torch.Tensor,
            num_global_tokens_per_expert_after_drop: torch.Tensor,
            expert_capacity: int,
            device_capacity: int,
            step: int) -> None:
        if self.ep_rank != 0:
            return
        if get_forward_context().in_profile_run or get_forward_context().capturing or get_forward_context().is_graph_warmup:
            return

        expert_load_before = num_global_tokens_per_expert_before_drop.sum(
            dim=0).to(torch.int64)
        expert_load_after = num_global_tokens_per_expert_after_drop.sum(
            dim=0).to(torch.int64)

        rank_load_before = num_global_tokens_per_expert_before_drop.reshape(
            self.ep_size, self.ep_size, self.num_local_experts).sum(
                dim=(0, 2)).to(torch.int64)
        rank_load_after = num_global_tokens_per_expert_after_drop.reshape(
            self.ep_size, self.ep_size, self.num_local_experts).sum(
                dim=(0, 2)).to(torch.int64)

        max_expert_before, max_expert_before_idx = torch.max(expert_load_before,
                                                              dim=0)
        max_expert_after, max_expert_after_idx = torch.max(expert_load_after,
                                                            dim=0)
        max_rank_before, max_rank_before_idx = torch.max(rank_load_before,
                                                          dim=0)
        max_rank_after, max_rank_after_idx = torch.max(rank_load_after, dim=0)

        logger.info(
            "[TokenDrop][step=%d] Max expert load before=%d (expert=%d), after=%d (expert=%d), expert_capacity=%d",
            step,
            int(max_expert_before.item()),
            int(max_expert_before_idx.item()),
            int(max_expert_after.item()),
            int(max_expert_after_idx.item()),
            int(expert_capacity),
        )
        logger.info(
            "[TokenDrop][step=%d] Max rank load before=%d (rank=%d), after=%d (rank=%d), device_capacity=%d",
            step,
            int(max_rank_before.item()),
            int(max_rank_before_idx.item()),
            int(max_rank_after.item()),
            int(max_rank_after_idx.item()),
            int(device_capacity),
        )

        if not self.token_drop_csv_dir:
            logger.warning(
                "[TokenDrop][step=%d] CSV not written because VLLM_TOKEN_DROP_CSV_DIR is not set.",
                step)
            return

        try:
            os.makedirs(self.token_drop_csv_dir, exist_ok=True)
            csv_path = os.path.join(
                self.token_drop_csv_dir,
                "token_drop_stats_rank0.csv")
            should_write_header = (not os.path.exists(csv_path)
                                   or os.path.getsize(csv_path) == 0)

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if should_write_header:
                    writer.writerow([
                        "step",
                        "expert_id",
                        "before_tokens",
                        "after_tokens",
                        "dropped_tokens",
                        "capacity",
                    ])

                for expert_idx in range(self.num_experts):
                    before_val = int(expert_load_before[expert_idx].item())
                    after_val = int(expert_load_after[expert_idx].item())
                    writer.writerow([
                        step,
                        expert_idx,
                        before_val,
                        after_val,
                        before_val - after_val,
                        int(expert_capacity),
                    ])

            logger.info("[TokenDrop][step=%d] CSV saved to: %s", step,
                        csv_path)
        except Exception as e:
            logger.warning("[TokenDrop][step=%d] Failed to write CSV: %s",
                           step, str(e))

    def _rank_token_ranges(self, num_tokens_across_dp: torch.Tensor
                           ) -> list[tuple[int, int]]:
        token_prefix = torch.cumsum(num_tokens_across_dp.to(torch.int64), dim=0)
        token_starts = token_prefix - num_tokens_across_dp.to(torch.int64)
        token_ranges: list[tuple[int, int]] = []
        for rank in range(self.ep_size):
            start = int(token_starts[rank].item())
            end = int(token_prefix[rank].item())
            token_ranges.append((start, end))
        return token_ranges

    def _compute_num_global_tokens_per_expert_after_drop(
            self, global_topk_ids_after_drop: torch.Tensor,
            num_tokens_across_dp: torch.Tensor) -> torch.Tensor:
        num_global_tokens_per_expert_after_drop = torch.zeros(
            (self.ep_size, self.num_experts),
            dtype=torch.int64,
            device=global_topk_ids_after_drop.device,
        )
        token_ranges = self._rank_token_ranges(num_tokens_across_dp)
        for rank, (start, end) in enumerate(token_ranges):
            current_topk_ids_after_drop = global_topk_ids_after_drop[start:end, :]

            flat_expert_ids = current_topk_ids_after_drop.reshape(-1).to(
                torch.int64)
            valid_mask = (flat_expert_ids < self.num_experts).to(torch.int64)
            safe_expert_ids = torch.clamp(flat_expert_ids,
                                          min=0,
                                          max=self.num_experts - 1)

            per_rank_counts = torch.zeros(
                self.num_experts,
                dtype=torch.int64,
                device=global_topk_ids_after_drop.device,
            )
            per_rank_counts.scatter_add_(0, safe_expert_ids, valid_mask)
            num_global_tokens_per_expert_after_drop[rank] = per_rank_counts
        return num_global_tokens_per_expert_after_drop

    def _compute_output_splits_after_drop(
            self,
            num_global_tokens_per_expert: torch.Tensor,
            global_topk_ids_after_drop: torch.Tensor,
            ) -> torch.Tensor:

        """get tokens for each global experts after drop

        Args:
            num_global_tokens_per_expert (torch.Tensor): shape [ep_size, num_experts], 
                num of initial input local tokens of each rank processed by each global expert
            global_topk_ids_after_drop (torch.Tensor): shape [num_global_tokens, topk]
                dropped tokens will have their expert index set to num_experts (sentinel), 
                which is not a valid expert index and can be easily masked out in later steps
            expert_capacity (int): max num of tokens each global expert can process after drop

        Returns:
            torch.Tensor: output_splits: shape [ep_size], num of tokens received from each rank after drop
        """
        num_tokens_across_dp = get_forward_context().dp_metadata.num_tokens_across_dp_cpu
        token_ranges = self._rank_token_ranges(num_tokens_across_dp)
        output_splits = torch.zeros(self.ep_size, dtype=torch.int64, device=num_global_tokens_per_expert.device)
        for rank, (start, end) in enumerate(token_ranges):

            # [num_tokens_this_rank, topk]
            currrank_topk_ids_after_drop = global_topk_ids_after_drop[start:end, :]

            # for this rank's topk_ids, the dropped tokens of this rank will have their expert index set to num_experts, 
            # which is not a valid expert index and can be easily masked out in later steps
            mask_dropped_tokens = (currrank_topk_ids_after_drop == self.num_experts)
            mask_sent_to_local_tokens = (
                (currrank_topk_ids_after_drop >= self.local_expert_indices[0])
                & (currrank_topk_ids_after_drop <= self.local_expert_indices[-1]))
            mask_valid_tokens = ~mask_dropped_tokens & mask_sent_to_local_tokens
            output_splits[rank] = mask_valid_tokens.sum()
        
        return output_splits
            
    def _compute_input_splits_after_drop(
            self,
            num_local_tokens_per_expert: torch.Tensor,
            local_topk_ids_before_drop: torch.Tensor) -> torch.Tensor:
        """get tokens for each local experts after drop
        Args:
            num_local_tokens_per_expert (torch.Tensor): shape [num_experts], 
                num of local input tokens sent to each global experts
            global_topk_ids_after_drop (torch.Tensor): shape [num_global_tokens, topk]
                dropped tokens will have their expert index set to num_experts (sentinel), 
                which is not a valid expert index and can be easily masked out in later steps
            expert_capacity (int): max num of tokens each global expert can process after drop
        Returns:
            torch.Tensor: input_splits: shape [ep_size], num of tokens sent to each rank after drop
        """ 

        input_splits = torch.zeros(self.ep_size, dtype=torch.int64, device=num_local_tokens_per_expert.device)

        # [num_local_tokens, topk]
        this_rank_topk_ids_after_drop = local_topk_ids_before_drop
        
        # TODO: IMPLEMENT INPUT_SPLITS AFTER DROP
        for rank in range(self.ep_size):
            rank_experts_start_idx = rank * self.num_local_experts
            rank_experts_end_idx = rank_experts_start_idx + self.num_local_experts - 1
            mask_sent_to_rank = (this_rank_topk_ids_after_drop >= rank_experts_start_idx) & \
                                (this_rank_topk_ids_after_drop <= rank_experts_end_idx)
            mask_dropped = this_rank_topk_ids_after_drop == self.num_experts
            mask_valid = mask_sent_to_rank & ~mask_dropped
            input_splits[rank] = mask_valid.sum()
        
        return input_splits

    def _get_local_permute_keep_indices_from_topk(
            self,
            local_topk_ids_before_drop: torch.Tensor,
            local_topk_ids_after_drop: torch.Tensor,
            num_local_tokens_per_expert: torch.Tensor,
            device: torch.device) -> torch.Tensor:
        flat_ids_before_drop = local_topk_ids_before_drop.reshape(-1).to(
            torch.int64)
        flat_ids_after_drop = local_topk_ids_after_drop.reshape(-1).to(
            torch.int64)
        kept_mask = flat_ids_after_drop < self.num_experts

        # [num_experts]
        starts = torch.cumsum(num_local_tokens_per_expert.to(torch.int64), dim=0)

        # [num_experts]
        # the offset of each expert in the local permuted token sequence
        starts = torch.cat(
            [torch.tensor([0], dtype=torch.int64, device=device), starts[:-1]])

        kept_indices: list[torch.Tensor] = []
        for expert_idx in range(self.num_experts):

            # [num_local_tokens*topk,], mask of tokens that are dispatched to current expert before drop
            expert_mask = (flat_ids_before_drop == expert_idx)
            keep_mask_expert = expert_mask & kept_mask

            # [num_local_tokens*topk,]
            # 即当前expert被分配的token在后续permuted tokens中该专家buffer内的偏移量
            expert_occurrence_rank = torch.cumsum(expert_mask.to(torch.int64),
                                                  dim=0) - 1
            
            # [num_kept_tokens_this_expert,]
            # 仅取出被保留的token在当前expert中的occurrence rank
            local_rank_in_expert = expert_occurrence_rank[keep_mask_expert]
            kept_indices.append(starts[expert_idx] + local_rank_in_expert)

        if not kept_indices:
            return torch.empty(0, dtype=torch.int64, device=device)

        return torch.cat(kept_indices, dim=0).to(torch.int64)
            

    def _get_local_permute_keep_indices(
            self,
            num_local_tokens_per_expert: torch.Tensor,
            num_local_tokens_per_expert_kept: torch.Tensor,
            device: torch.device) -> torch.Tensor:
        """_summary_

        Args:
            num_local_tokens_per_expert (torch.Tensor): shape [num_experts], num of initial local tokens processed by each global expert
            num_local_tokens_per_expert_kept (torch.Tensor): shape [num_experts], num of local tokens kept for each global expert after drop
            device (torch.device): 

        Returns:
            torch.Tensor: kept_indices: the indices of local tokens to keep after drop in the local permuted token sequence; shape [num_local_tokens_kept]
        """

        # prefix sum of local tokens processed by each global experts
        # use to represent offset for each global expert in the local permuted token sequence
        starts = torch.cumsum(num_local_tokens_per_expert.to(torch.int64), dim=0)
        starts = torch.cat(
            [torch.tensor([0], dtype=torch.int64, device=starts.device), starts[:-1]])

        kept_indices: list[torch.Tensor] = []
        for expert_idx in range(self.num_experts):
            keep_count = int(num_local_tokens_per_expert_kept[expert_idx].item())
            if keep_count <= 0:
                continue
            start = int(starts[expert_idx].item())
            kept_indices.append(
                torch.arange(start,
                             start + keep_count,
                             device=device,
                             dtype=torch.int64))

        if not kept_indices:
            return torch.empty(0, dtype=torch.int64, device=device)
        return torch.cat(kept_indices, dim=0)
    
    def _compute_indices(self, scores_sub, mask_sub, expert_capacity):
        """
        Args:
            scores_sub (tensor): router logits的一个子集, [T, num_of_subset_experts]
            mask_sub (tensor): router logits的子集中被topk选中的位置的mask, [T, num_of_subset_experts]
            expert_capacity (int): 专家容量

        Returns:
            capacity_indices: 在scores_sub中被选中且不超过容量限制的位置的indices, [expert_capacity, num_of_subset_experts]
        """
        
        # 把mask无效的位置的score设置为-inf，这样在后续的topk中就不会被选中
        masked_scores = scores_sub.masked_fill(~mask_sub, float('-inf'))  # ascending 排序，mask无效
        _, capacity_indices = torch.topk(
                                    masked_scores, 
                                    k=expert_capacity, 
                                    dim=0, 
                                    sorted=False
                                    )
        # 返回 overloaded experts中的topk token indices
        return capacity_indices

    def _compute_kept_tokens_per_rank_expert(
            self, num_global_tokens_per_expert: torch.Tensor,
            expert_capacity: int) -> torch.Tensor:
        if expert_capacity <= 0:
            return torch.zeros_like(num_global_tokens_per_expert,
                                    dtype=torch.int64)
        counts = num_global_tokens_per_expert.to(torch.int64)
        kept = torch.zeros_like(counts, dtype=torch.int64)
        for expert_idx in range(counts.shape[1]):
            remaining = expert_capacity
            for rank in range(counts.shape[0]):
                if remaining <= 0:
                    break
                keep = min(int(counts[rank, expert_idx].item()), remaining)
                kept[rank, expert_idx] = keep
                remaining -= keep
        return kept

    def _get_topk_ids_weights_after_drop_by_expert(self, global_topk_ids: torch.Tensor, global_topk_weights: torch.Tensor, expert_capacity: int) -> torch.Tensor:
        mask_buffer = torch.zeros((global_topk_ids.shape[0], self.num_experts),
                                  dtype=torch.bool,
                                  device=global_topk_ids.device)
        scores_buffer = torch.zeros((global_topk_ids.shape[0], self.num_experts),
                                  dtype=global_topk_weights.dtype,
                                  device=global_topk_ids.device)
        # 当前router logits中，被gating topk选中的tokens和expert位置被标记为True
        mask_buffer.scatter_(-1, global_topk_ids, True)
        scores_buffer.scatter_(-1, global_topk_ids, global_topk_weights)

        capacity = min(expert_capacity, scores_buffer.shape[0])

        # 对于没被topk选中的token-expert位置, 填上inf用于后续capacity topk
        masked_scores = scores_buffer.masked_fill(~mask_buffer,
                                                    float('-inf'))
        _, capacity_indices = torch.topk(masked_scores,
                                            k=capacity,
                                            dim=0,
                                            sorted=False)

        # [num_global_tokens, num_global_experts]，
        # 被gating选中且不超过容量限制的位置被标记为True
        kept_mask = torch.zeros_like(mask_buffer).scatter(
            0, capacity_indices, True) & mask_buffer

        # [T, topk],
        # kept_mask是[T, num_global_experts]，gather出被gating选中且不超过容量限制的位置的mask回topk
        top_mask = kept_mask.gather(-1, global_topk_ids)

        # 把最终被选中且不超过容量限制的位置的权重保留，其他位置的权重设置为0
        # [T, topk]
        global_topk_weights = global_topk_weights * top_mask.to(
            global_topk_weights.dtype)

        # 把最终被选中但超过容量限制的位置的expert index设置为num_experts (sentinel)，表示这些token将被丢弃
        global_topk_ids = global_topk_ids.masked_fill(~top_mask, self.num_experts)

        return global_topk_weights, global_topk_ids

    def _get_topk_ids_weights_after_drop_by_device(self, global_topk_ids: torch.Tensor, global_topk_weights: torch.Tensor, device_capacity: int) -> torch.Tensor:
        if global_topk_ids.numel() == 0:
            return global_topk_weights, global_topk_ids

        if device_capacity <= 0:
            return (torch.zeros_like(global_topk_weights),
                    torch.full_like(global_topk_ids, self.num_experts))

        flat_topk_ids = global_topk_ids.reshape(-1).to(torch.int64)
        flat_topk_weights = global_topk_weights.reshape(-1)

        device_ids_all = torch.div(flat_topk_ids,
                                   self.num_local_experts,
                                   rounding_mode='floor')

        # 1) sort by score(desc), 2) stable sort by device id(asc).
        # After these two sorts, assignments in each device are already ordered
        # by score(desc), so we can keep the first `device_capacity` per device.

        # [num_global_tokens*topk,], the indices that would sort the scores in descending order
        score_order = torch.argsort(flat_topk_weights, descending=True)

        # [num_global_tokens*topk,], flattened assignment indices sorted by score(desc)
        sorted_scores_indices = score_order

        # [num_global_tokens*topk,], the device ids of the flat topk ids, sorted by score(desc)
        # TODO: can be optimized by avoiding the sort and directly get the topk indices for each device?
        # cast to fp32 to run argsort on aicore
        sorted_device_ids = device_ids_all.index_select(0, score_order).to(dtype=torch.float32)

        # [num_global_tokens*topk,], the indices that would sort the device ids in ascending order
        # since itis stable, the relative order of tokens with the same device id (sorted by score desc) will be kept
        device_order = torch.argsort(sorted_device_ids, stable=True)

        # [ num_global_tokens*topk,], the flat topk ids sorted by score(desc) and then stable sorted by device id(asc)
        sorted_indices = sorted_scores_indices.index_select(0, device_order)
        sorted_device_ids = sorted_device_ids.index_select(0, device_order)

        sorted_len = sorted_device_ids.shape[0]

        # [num_global_tokens*topk,]
        # [0,1,2,..., num_global_tokens*topk-1]
        arange_sorted = torch.arange(sorted_len,
                                     device=sorted_device_ids.device,
                                     dtype=torch.int64)

        group_start_flags = torch.ones_like(sorted_device_ids, dtype=torch.bool)

        # indicates the start offset in T*K of each device
        # [1,0,0,0,1,0,0,1,0,0] -> device0: [0,1,2,3], device1: [4,5,6], device2: [7,8,9]
        group_start_flags[1:] = sorted_device_ids[1:] != sorted_device_ids[:-1]

        # [ep_size,], the start offset of each device's tokens in the sorted token sequence
        # [0,4,7]
        group_starts = arange_sorted[group_start_flags]

        # [num_global_tokens*topk,], the device id of each token in the sorted token sequence
        # [0,0,0,0,1,1,1,2,2,2]
        group_ids = torch.cumsum(group_start_flags.to(torch.int64), dim=0) - 1

        # [num_global_tokens*topk,]
        # the rank of each token within its device in the sorted token sequence
        # [0,1,2,3,0,1,2,0,1,2]
        rank_in_device = arange_sorted - group_starts.index_select(0, group_ids)

        # [num_global_tokens*topk,], 
        # whether each token is within the capacity limit of its device
        keep_in_sorted = rank_in_device < device_capacity

        # [num_tokens_kept], 
        # the indices of tokens that are within the capacity limit of their devices in the sorted token sequence
        kept_indices = sorted_indices.index_select(
            0, torch.nonzero(keep_in_sorted, as_tuple=False).squeeze(-1))
        keep_mask_flat = torch.zeros_like(flat_topk_ids,
                                          dtype=torch.bool,
                                          device=flat_topk_ids.device)
        # [num_tokens_kept,]，把被保留的token位置标记为True
        keep_mask_flat.scatter_(0, kept_indices, True)

        keep_mask = keep_mask_flat.view_as(global_topk_ids)
        global_topk_weights = global_topk_weights * keep_mask.to(
            global_topk_weights.dtype)
        global_topk_ids = global_topk_ids.masked_fill(~keep_mask,
                                                       self.num_experts)
        return global_topk_weights, global_topk_ids

    def _preprocess_with_token_drop(self,
                                    topk_ids: torch.Tensor,
                                    topk_weights: Optional[torch.Tensor] = None,
                                    router_logits: Optional[torch.Tensor] = None,
                                    step: int = 0):
        if self.token_drop_local_only:
            return self._preprocess_with_token_drop_local(topk_ids,
                                                          topk_weights,
                                                          step)

        if topk_weights is None:
            topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        num_local_tokens_per_expert = torch.histc(topk_ids,
                                                  bins=self.num_experts,
                                                  min=0,
                                                  max=self.num_experts)
        
        # shape: [num_experts], num of initial local tokens processed by each global expert
        num_local_tokens_per_expert = num_local_tokens_per_expert.to(torch.int64)

        ep_size = self.ep_size
        self.num_out_tokens = topk_ids.numel()

        # shape: [ep_size, num_experts], num of initial local tokens of each rank processed by each global expert
        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert,
            group=self.ep_group).reshape(ep_size, self.num_experts).to(torch.int64)
        try:
            num_tokens_across_dp = get_forward_context(
            ).dp_metadata.num_tokens_across_dp_cpu
        except AssertionError:
            num_tokens_across_dp = torch.full((ep_size, ),
                                              topk_ids.shape[0],
                                              dtype=torch.int64,
                                              device=torch.device("cpu"))

        global_topk_ids = torch.zeros((num_tokens_across_dp.sum(), topk_ids.shape[1]),
                                     dtype=topk_ids.dtype,
                                     device=topk_ids.device)
        global_topk_weights = torch.zeros((num_tokens_across_dp.sum(), topk_weights.shape[1]),
                                         dtype=topk_weights.dtype,
                                         device=topk_weights.device)
        global_router_logits = torch.zeros((num_tokens_across_dp.sum(), router_logits.shape[1]),
                                         dtype=router_logits.dtype,
                                         device=router_logits.device)
        # gather global topk_ids
        torch_npu.distributed.all_gather_into_tensor_uneven(
            global_topk_ids, topk_ids, num_tokens_across_dp.numpy(), group=self.ep_group
        )
        torch_npu.distributed.all_gather_into_tensor_uneven(
            global_topk_weights, topk_weights, num_tokens_across_dp.numpy(), group=self.ep_group
        )
        torch_npu.distributed.all_gather_into_tensor_uneven(
            global_router_logits, router_logits, num_tokens_across_dp.numpy(), group=self.ep_group
        )


        if not torch.is_tensor(global_topk_ids):
            global_topk_ids = torch.cat([topk_ids] * ep_size, dim=0)
        if not torch.is_tensor(global_topk_weights):
            global_topk_weights = torch.cat([topk_weights] * ep_size, dim=0)
        if not torch.is_tensor(global_router_logits):
            global_router_logits = torch.cat([router_logits] * ep_size, dim=0)


        global_avg_tokens_per_expert = (
            num_global_tokens_per_expert.sum().to(torch.float32) /
            float(self.num_experts))
        expert_capacity = math.ceil(global_topk_ids.shape[0] * self.top_k * self.token_drop_load_factor / self.num_experts)

        device_capacity = math.ceil(expert_capacity * self.num_local_experts)

        ###
        # drop tokens according to topk weights, get the topk_ids after drop
        if self.drop_strategy == "expert_drop":
            global_topk_weights_after_drop, global_topk_ids_after_drop = \
                self._get_topk_ids_weights_after_drop_by_expert(global_topk_ids, 
                                                    global_topk_weights, expert_capacity)
        elif self.drop_strategy == "device_drop":
             # TODO: implement token drop by device, 
             # which directly drop tokens on each device according to the device capacity
             global_topk_weights_after_drop, global_topk_ids_after_drop = \
                self._get_topk_ids_weights_after_drop_by_device(global_topk_ids, 
                                                    global_topk_weights, device_capacity)
        elif self.drop_strategy == "expert_expanded_drop":
            # TODO: implement expert-expanded token drop
            global_topk_weights_after_drop, global_topk_ids_after_drop = \
                self._get_topk_ids_weights_after_expanded_drop_by_expert(global_topk_ids, 
                                                    global_topk_weights, global_router_logits, expert_capacity)
        else:
            raise ValueError(f"Invalid drop strategy: {self.drop_strategy}")
        ###

        num_global_tokens_per_expert_after_drop = self._compute_num_global_tokens_per_expert_after_drop(
            global_topk_ids_after_drop,
            num_tokens_across_dp,
        )
        num_global_tokens_per_local_expert = num_global_tokens_per_expert_after_drop[:, self.local_expert_indices[
            0]:self.local_expert_indices[-1] + 1]
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
        output_splits = num_global_tokens_per_local_expert.sum(axis=-1)

        input_splits = num_global_tokens_per_expert_after_drop[
            self.ep_rank].reshape(ep_size, self.num_local_experts).sum(axis=1)

        # shape: [ep_size,]
        # num of received tokens from each rank after drop
        output_splits = output_splits.to(device=torch.device("cpu"), non_blocking=True)

        # shape: [ep_size,]
        # num of sent tokens to each rank after drop
        input_splits = input_splits.to(device=torch.device("cpu"), non_blocking=True)

        global_input_tokens_local_experts_indices = None
        if self.num_local_experts > 1:
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                num_global_tokens_per_local_expert.ravel())

        

        rank_token_ranges = self._rank_token_ranges(num_tokens_across_dp)
        this_rank_start, this_rank_end = rank_token_ranges[self.ep_rank]
        this_rank_topk_ids_after_drop = global_topk_ids_after_drop[this_rank_start:this_rank_end,
                                                                    :]
        this_rank_topk_weights_after_drop = global_topk_weights_after_drop[this_rank_start:this_rank_end,
                                                                    :]

        self.num_out_tokens_after_drop = int(input_splits.sum().item())

        if self.token_drop_logging:
            self._log_token_drop_statistics(
                num_global_tokens_per_expert_before_drop=num_global_tokens_per_expert,
                num_global_tokens_per_expert_after_drop=
                num_global_tokens_per_expert_after_drop,
                expert_capacity=expert_capacity,
                device_capacity=device_capacity,
                step=step,
            )


        return (
            num_tokens_per_local_expert,
            input_splits.numpy(),
            output_splits.numpy(),
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
            expert_capacity,
            global_avg_tokens_per_expert,
            num_global_tokens_per_expert,
            this_rank_topk_ids_after_drop,
            this_rank_topk_weights_after_drop,
        )

    def _preprocess_with_token_drop_local(
            self,
            topk_ids: torch.Tensor,
            topk_weights: Optional[torch.Tensor] = None,
            step: int = 0):

        num_local_tokens_per_expert = torch.histc(topk_ids,
                                                  bins=self.num_experts,
                                                  min=0,
                                                  max=self.num_experts).to(
                                                      torch.int64)
        ep_size = self.ep_size
        self.num_out_tokens = topk_ids.numel()


        num_tokens_across_dp = get_forward_context(
        ).dp_metadata.num_tokens_across_dp_cpu


        total_tokens = int(num_tokens_across_dp.sum().item())
        expert_capacity = math.ceil(total_tokens * self.top_k *
                                    self.token_drop_load_factor /
                                    self.num_experts)
        device_capacity = math.ceil(expert_capacity * self.num_local_experts)

        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert,
            group=self.ep_group).reshape(ep_size, self.num_experts).to(
                torch.int64)

        # Global average before drop only depends on total assignments.
        global_avg_tokens_per_expert = torch.tensor(
            float(total_tokens * self.top_k) / float(self.num_experts),
            dtype=torch.float32,
            device=topk_ids.device,
        )

        if not self.drop_by_expert:
            logger.warning(
                "[TokenDrop] local-only mode currently supports expert-based drop only; forcing expert strategy."
            )

        local_topk_weights_after_drop, local_topk_ids_after_drop = \
            self._get_topk_ids_weights_after_drop_by_expert(
                topk_ids,
                topk_weights,
                expert_capacity,
            )

        this_rank_kept_expert_ids = local_topk_ids_after_drop[
            local_topk_ids_after_drop < self.num_experts]
        num_local_tokens_per_expert_after_drop = torch.bincount(
            this_rank_kept_expert_ids.reshape(-1),
            minlength=self.num_experts,
        ).to(torch.int64)

        # Build drop-after global matrix from per-rank count vectors.
        num_global_tokens_per_expert_after_drop = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert_after_drop,
            group=self.ep_group).reshape(ep_size, self.num_experts).to(
                torch.int64)

        num_global_tokens_per_local_expert = num_global_tokens_per_expert_after_drop[:, self.local_expert_indices[
            0]:self.local_expert_indices[-1] + 1]
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(
            axis=0)

        # Derive both splits from the same global after-drop matrix.
        output_splits = num_global_tokens_per_local_expert.sum(axis=-1)
        input_splits = num_global_tokens_per_expert_after_drop[
            self.ep_rank].reshape(ep_size, self.num_local_experts).sum(axis=1)

        output_splits = output_splits.to(device=torch.device("cpu"),
                                         non_blocking=True)
        input_splits = input_splits.to(device=torch.device("cpu"),
                                       non_blocking=True)

        global_input_tokens_local_experts_indices = None
        if self.num_local_experts > 1:
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                num_global_tokens_per_local_expert.ravel())

        self.num_out_tokens_after_drop = int(input_splits.sum().item())

        if self.token_drop_logging:
            self._log_token_drop_statistics(
                num_global_tokens_per_expert_before_drop=
                num_global_tokens_per_expert,
                num_global_tokens_per_expert_after_drop=
                num_global_tokens_per_expert_after_drop,
                expert_capacity=expert_capacity,
                device_capacity=device_capacity,
                step=step,
            )

        return (
            num_tokens_per_local_expert,
            input_splits.numpy(),
            output_splits.numpy(),
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
            expert_capacity,
            global_avg_tokens_per_expert,
            num_global_tokens_per_expert,
            local_topk_ids_after_drop,
            local_topk_weights_after_drop,
        )

    @override
    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False,
                       dynamic_eplb: bool = False,
                       pertoken_scale: Optional[torch.Tensor] = None,
                       router_logits: Optional[torch.Tensor] = None):

        """token dispatch for all2all

        Returns:
            1) 对外返回：TokenDispatchResult
                hidden_states：global_input_tokens，已经完成本地 permute + 跨 EP all2all + 本地按 expert 重排后的张量；形状通常是 [本 rank 收到并分配给本地 experts 的 token 总数, hidden]。
                dynamic_scale：dynamic_scale_final。仅量化开启(with_quant=True)时有效；否则是 None。
                group_list：tokens_per_expert，表示本 rank 上每个 local expert 的 token 数（count 列表）。
                group_list_type：固定为 1，表示 group_list 是 “count mode”（不是前缀和/offset 模式）。
                context_metadata：给 token_combine 用的上下文（见第 2 点）。
                
            2) context_metadata 字段含义
                input_splits：本 rank 在 all2all 中发往各对端 rank 的 token 数（send split）。
                output_splits：本 rank 在 all2all 中从各对端 rank 接收的 token 数（recv split）。
                topk_weights：router 给每个 token 的 top-k 权重，combine 时用于加权还原。
                reversed_local_input_permutation_mapping：第一次本地 npu_moe_token_permute 的逆映射（最终恢复原 token 顺序用）。
                    每个token复制topk个，根据token序列排序；第i个元素表示按照token序列的第i个token在第一次本地permute (专家序列)后的序列中的位置；
                reversed_global_input_permutation_mapping：第二次“按本地 expert 分组”重排的逆映射；仅 num_local_experts > 1 时有值，否则为 None。
        """
        self.with_quant = with_quant
        self.hidden_shape = hidden_states.shape
        ctx = get_forward_context()

        step = -1
        if not (ctx.is_graph_warmup or ctx.capturing or ctx.in_profile_run):
            self.token_drop_step += 1
            step = self.token_drop_step

        assert self.hidden_shape is not None
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        self.hidden_shape_before_permute = hidden_states.shape

        (
            tokens_per_expert,
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
            expert_capacity,
            global_avg_tokens_per_expert,
            num_global_tokens_per_expert_before_drop,
            topk_ids_after_drop,
            topk_weights_after_drop
        ) = self._preprocess_with_token_drop(topk_ids, topk_weights, router_logits, step)

        self.num_out_tokens_before_drop = self.num_out_tokens
        self.num_out_tokens = self.num_out_tokens_after_drop

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids_after_drop,
            num_out_tokens=self.num_out_tokens_before_drop,
        )

        # select kept tokens after drop in the local permuted token sequence
        permutated_local_input_tokens = permutated_local_input_tokens[:self.num_out_tokens]

        dynamic_scale_after_all2all = None
        if self.with_quant:
            permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_local_input_tokens)
            _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
                dynamic_scale, output_splits, input_splits, self.ep_group)
            permute2_ep_all_to_all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        # [yiwu] perform all2all communication
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens, output_splits, input_splits,
            self.ep_group)
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # Postprocess
        # [yiwu] do permutation to place tokens by local experts order
        global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = self._dispatch_postprocess(
            global_input_tokens, dynamic_scale_after_all2all,
            global_input_tokens_local_experts_indices)

        context_metadata = {
            "input_splits":
            input_splits,
            "output_splits":
            output_splits,
            "topk_weights":
            topk_weights,
            "reversed_local_input_permutation_mapping":
            reversed_local_input_permutation_mapping,
            "reversed_global_input_permutation_mapping":
            reversed_global_input_permutation_mapping,
            "num_out_tokens_before_drop":
            self.num_out_tokens_before_drop,
            "expert_capacity":
            expert_capacity,
            "global_avg_tokens_per_expert_before_drop":
            global_avg_tokens_per_expert,
            "num_global_tokens_per_expert_before_drop":
            num_global_tokens_per_expert_before_drop,
            "num_global_tokens_per_local_expert_after_drop":
            num_global_tokens_per_local_expert,
            "topk_ids_after_drop":
            topk_ids_after_drop,
            "topk_weights_after_drop":
            topk_weights_after_drop,
        }

        return TokenDispatchResult(
            hidden_states=global_input_tokens,
            dynamic_scale=dynamic_scale_final,
            group_list=tokens_per_expert,
            group_list_type=1,
            context_metadata=context_metadata,
        )

    @override
    def token_combine(self, hidden_states, context_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        # 1. Preprocess using metadata
        hidden_states = self._combine_preprocess(hidden_states,
                                                 context_metadata)

        # 2. AllToAll
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            context_metadata["input_splits"],
            context_metadata["output_splits"],
            self.ep_group,
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        restored_local_tokens = torch.zeros(
            (context_metadata["num_out_tokens_before_drop"],
             permutated_local_input_tokens.shape[-1]),
            dtype=permutated_local_input_tokens.dtype,
            device=permutated_local_input_tokens.device,
        )
        restored_local_tokens[:self.num_out_tokens] = permutated_local_input_tokens
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # 3. Postprocess using metadata
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=restored_local_tokens,
            sorted_indices=context_metadata[
                "reversed_local_input_permutation_mapping"].to(torch.int32),
            probs=context_metadata["topk_weights_after_drop"],
            restore_shape=self.hidden_shape_before_permute,
        )
        output = output.view(self.hidden_shape)

        return TokenCombineResult(routed_out=output)


class TokenDispatcherWithAll2AllvExpandedDrop(TokenDispatcherWithAll2AllV):
    """
    Token dispatcher that implements expanded drop strategy.

    Expanded drop strategy:
    - Extends each token's candidate expert set by adding all local experts of the rank where the token resides
    - Performs capacity-based token drop on the expanded candidate set
    - Goal: allow local experts to process more tokens, improving local compute efficiency and reducing cross-rank communication

    Key differences from standard token drop:
    - topk_ids shape: [T, topk + num_local_experts] instead of [T, topk]
    - Duplicate handling: expanded local experts may overlap with original top-k experts
    - Permute strategy: unified permute + truncate valid tokens (sentinel tokens are placed at the end)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Expanded drop specific parameters
        self.token_drop_load_factor = kwargs.get("token_drop_load_factor", 1.0)
        self.expanded_topk = self.top_k + self.num_local_experts  # expanded candidate count
        self.drop_strategy = os.getenv("VLLM_TOKEN_DROP_STRATEGY",
                                       "expert_expanded_drop")

        # Logging and statistics configuration
        self.token_drop_logging = os.getenv("VLLM_TOKEN_DROP_LOGGING", "0") == "1"
        self.token_drop_csv_dir = os.getenv("VLLM_TOKEN_DROP_CSV_DIR", "")
        self.token_drop_step = 0

        logger.info(
            f"[ExpandedDrop] Initialized TokenDispatcherWithAll2AllvExpandedDrop with "
            f"token_drop_load_factor={self.token_drop_load_factor}, "
            f"expanded_topk={self.expanded_topk} (original_topk={self.top_k}, num_local_experts={self.num_local_experts}), "
            f"strategy={self.drop_strategy}"
        )

    def _log_token_drop_statistics(
                self,
                num_global_tokens_per_expert_before_drop: torch.Tensor,
                num_global_tokens_per_expert_after_drop: torch.Tensor,
                expert_capacity: int,
                device_capacity: int,
                step: int) -> None:
            if self.ep_rank != 0:
                return
            if get_forward_context().in_profile_run or get_forward_context().capturing or get_forward_context().is_graph_warmup:
                return

            expert_load_before = num_global_tokens_per_expert_before_drop.sum(
                dim=0).to(torch.int64)
            expert_load_after = num_global_tokens_per_expert_after_drop.sum(
                dim=0).to(torch.int64)

            rank_load_before = num_global_tokens_per_expert_before_drop.reshape(
                self.ep_size, self.ep_size, self.num_local_experts).sum(
                    dim=(0, 2)).to(torch.int64)
            rank_load_after = num_global_tokens_per_expert_after_drop.reshape(
                self.ep_size, self.ep_size, self.num_local_experts).sum(
                    dim=(0, 2)).to(torch.int64)

            max_expert_before, max_expert_before_idx = torch.max(expert_load_before,
                                                                dim=0)
            max_expert_after, max_expert_after_idx = torch.max(expert_load_after,
                                                                dim=0)
            max_rank_before, max_rank_before_idx = torch.max(rank_load_before,
                                                            dim=0)
            max_rank_after, max_rank_after_idx = torch.max(rank_load_after, dim=0)

            logger.info(
                "[TokenDrop][step=%d] Max expert load before=%d (expert=%d), after=%d (expert=%d), expert_capacity=%d",
                step,
                int(max_expert_before.item()),
                int(max_expert_before_idx.item()),
                int(max_expert_after.item()),
                int(max_expert_after_idx.item()),
                int(expert_capacity),
            )
            logger.info(
                "[TokenDrop][step=%d] Max rank load before=%d (rank=%d), after=%d (rank=%d), device_capacity=%d",
                step,
                int(max_rank_before.item()),
                int(max_rank_before_idx.item()),
                int(max_rank_after.item()),
                int(max_rank_after_idx.item()),
                int(device_capacity),
            )

            if not self.token_drop_csv_dir:
                logger.warning(
                    "[TokenDrop][step=%d] CSV not written because VLLM_TOKEN_DROP_CSV_DIR is not set.",
                    step)
                return

            try:
                os.makedirs(self.token_drop_csv_dir, exist_ok=True)
                csv_path = os.path.join(
                    self.token_drop_csv_dir,
                    "token_drop_stats_rank0.csv")
                should_write_header = (not os.path.exists(csv_path)
                                    or os.path.getsize(csv_path) == 0)

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if should_write_header:
                        writer.writerow([
                            "step",
                            "expert_id",
                            "before_tokens",
                            "after_tokens",
                            "dropped_tokens",
                            "capacity",
                        ])

                    for expert_idx in range(self.num_experts):
                        before_val = int(expert_load_before[expert_idx].item())
                        after_val = int(expert_load_after[expert_idx].item())
                        writer.writerow([
                            step,
                            expert_idx,
                            before_val,
                            after_val,
                            before_val - after_val,
                            int(expert_capacity),
                        ])

                logger.info("[TokenDrop][step=%d] CSV saved to: %s", step,
                            csv_path)
            except Exception as e:
                logger.warning("[TokenDrop][step=%d] Failed to write CSV: %s",
                            step, str(e))


    def _rank_token_ranges(self, num_tokens_across_dp: torch.Tensor) -> list[tuple[int, int]]:
        """Compute token ranges for each rank in the global tensor."""
        token_prefix = torch.cumsum(num_tokens_across_dp.to(torch.int64), dim=0)
        token_starts = token_prefix - num_tokens_across_dp.to(torch.int64)
        token_ranges: list[tuple[int, int]] = []
        for rank in range(self.ep_size):
            start = int(token_starts[rank].item())
            end = int(token_prefix[rank].item())
            token_ranges.append((start, end))
        return token_ranges

    def _get_topk_ids_weights_after_expanded_drop_by_expert(
            self,
            global_topk_ids: torch.Tensor,          # [T, topk]
            global_topk_weights: torch.Tensor,      # [T, topk]
            global_router_logits: torch.Tensor,     # [T, num_experts]
            expert_capacity: int,
            num_tokens_across_dp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implement expanded drop strategy with deduplication.

        Steps:
        1. Construct expanded candidate expert set [T, topk + num_local_experts]
        2. Deduplicate: detect overlap between expanded local experts and original top-k experts
        3. Scatter to mask/scores buffer and perform capacity-based top-k per expert
        4. Gather back to expanded candidate view
        5. Force mark duplicate positions as sentinel

        Returns:
            expanded_global_topk_weights: [T, topk + num_local_experts]
            expand_global_topk_ids: [T, topk + num_local_experts] (deduplicated)
        """
        T = global_topk_ids.shape[0]
        topk = global_topk_ids.shape[1]
        device = global_topk_ids.device

        # Use the same routing semantics as native expert selection:
        # softmax on router logits, then top-k.
        router_probs = torch.softmax(global_router_logits, dim=-1)
        topk_weights_from_probs, topk_ids_from_probs = torch.topk(
            router_probs.to(torch.float32),
            k=topk,
            dim=-1,
        )
        topk_ids_from_probs = topk_ids_from_probs.to(global_topk_ids.dtype)

        # Step 1: Construct expanded local expert indices for each token
        token_ranges = self._rank_token_ranges(num_tokens_across_dp)
        rank_ids = torch.zeros(T, dtype=torch.int64, device=device)
        for rank, (start, end) in enumerate(token_ranges):
            rank_ids[start:end] = rank

        # Each token's local experts start index in global expert space
        rank_experts_start_idx = rank_ids * self.num_local_experts

        # Generate all local expert indices [num_local_experts]
        all_local_expert_ids = torch.arange(self.num_local_experts, device=device)

        # Broadcast to [T, num_local_experts]
        local_expert_indices_t = rank_experts_start_idx.unsqueeze(-1) + all_local_expert_ids.unsqueeze(0)

        # Concatenate to get expanded candidate set [T, topk + num_local_experts]
        expand_global_topk_ids = torch.cat([topk_ids_from_probs, local_expert_indices_t], dim=-1)

        # Build masks from softmax-topk pairs and expanded local candidates.
        topk_mask_buffer = torch.zeros((T, self.num_experts),
                           dtype=torch.bool,
                           device=device)
        topk_mask_buffer.scatter_(-1, topk_ids_from_probs, True)

        local_mask_buffer = torch.zeros((T, self.num_experts),
                        dtype=torch.bool,
                        device=device)
        local_mask_buffer.scatter_(-1, local_expert_indices_t, True)

        expanded_mask_buffer = topk_mask_buffer | local_mask_buffer

        # Step 4: Capacity-based top-k per expert
        capacity = min(expert_capacity, router_probs.shape[0])
        masked_scores = router_probs.masked_fill(~expanded_mask_buffer,
                             float('-inf'))
        _, capacity_indices = torch.topk(masked_scores, k=capacity, dim=0, sorted=True)

        # Construct kept mask: positions that are both in candidates and within capacity
        kept_mask = torch.zeros_like(expanded_mask_buffer).scatter(
            0, capacity_indices, True) & expanded_mask_buffer

        # Step 5: Gather back to expanded candidate view
        top_mask = kept_mask.gather(-1, expand_global_topk_ids)
        expanded_global_topk_weights = router_probs.gather(-1,
                                   expand_global_topk_ids)

        # deduplicaate
        for pos in range(topk, self.expanded_topk):
            duplicate_mask = (expand_global_topk_ids[:, pos:pos+1] == expand_global_topk_ids[:, :pos]).any(dim=-1)
            top_mask[:, pos] = top_mask[:, pos] & ~duplicate_mask

        # Apply mask to weights and ids
        expanded_global_topk_weights = expanded_global_topk_weights * top_mask.to(
            topk_weights_from_probs.dtype)
        expand_global_topk_ids = expand_global_topk_ids.masked_fill(~top_mask, self.num_experts)

        # Safe renormalize: avoid NaN when a token has all-zero kept weights.
        row_sum = expanded_global_topk_weights.sum(dim=-1, keepdim=True)
        eps = torch.finfo(expanded_global_topk_weights.dtype).eps
        nonzero_mask = row_sum > eps
        expanded_global_topk_weights = torch.where(
            nonzero_mask,
            expanded_global_topk_weights / row_sum.clamp_min(eps),
            torch.zeros_like(expanded_global_topk_weights),
        )
        return expanded_global_topk_weights, expand_global_topk_ids

    def _get_topk_ids_weights_after_expanded_drop_by_device(
            self,
            global_topk_ids: torch.Tensor,          # [T, topk]
            global_topk_weights: torch.Tensor,      # [T, topk]
            global_router_logits: torch.Tensor,     # [T, num_experts]
            device_capacity: int,
            num_tokens_across_dp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expanded candidates first, then drop by per-device capacity."""
        del global_topk_ids, global_topk_weights

        T = global_router_logits.shape[0]
        if T == 0:
            return (
                torch.zeros((0, self.expanded_topk),
                            dtype=torch.float32,
                            device=global_router_logits.device),
                torch.zeros((0, self.expanded_topk),
                            dtype=torch.int64,
                            device=global_router_logits.device),
            )

        device = global_router_logits.device
        router_probs = torch.softmax(global_router_logits, dim=-1)
        _, topk_ids_from_probs = torch.topk(
            router_probs.to(torch.float32),
            k=self.top_k,
            dim=-1,
        )
        topk_ids_from_probs = topk_ids_from_probs.to(torch.int64)

        token_ranges = self._rank_token_ranges(num_tokens_across_dp)
        rank_ids = torch.zeros(T, dtype=torch.int64, device=device)
        for rank, (start, end) in enumerate(token_ranges):
            rank_ids[start:end] = rank

        rank_experts_start_idx = rank_ids * self.num_local_experts
        all_local_expert_ids = torch.arange(self.num_local_experts,
                                            device=device,
                                            dtype=torch.int64)
        local_expert_indices_t = rank_experts_start_idx.unsqueeze(
            -1) + all_local_expert_ids.unsqueeze(0)

        expand_global_topk_ids = torch.cat(
            [topk_ids_from_probs, local_expert_indices_t], dim=-1)
        expanded_global_topk_weights = router_probs.gather(
            -1, expand_global_topk_ids)

        # Keep top-k positions as primary when expanded local experts duplicate.
        # [T, expanded_topk], True for candidates, False for non-candidates
        candidate_mask = torch.ones_like(expand_global_topk_ids,
                                         dtype=torch.bool,
                                         device=device)

        # Deduplication
        for pos in range(self.top_k, self.expanded_topk):
            duplicate_mask = (
                expand_global_topk_ids[:, pos:pos + 1]
                == expand_global_topk_ids[:, :pos]).any(dim=-1)
            candidate_mask[:, pos] = candidate_mask[:, pos] & ~duplicate_mask

        if device_capacity <= 0:
            return (
                torch.zeros_like(expanded_global_topk_weights),
                torch.full_like(expand_global_topk_ids, self.num_experts),
            )

        # [T * expanded_topk] flatten for sorting and selection
        flat_ids = expand_global_topk_ids.reshape(-1)
        # [T * expanded_topk] flatten weights
        flat_weights = expanded_global_topk_weights.reshape(-1)
        # [T * expanded_topk] flatten valid mask after deduplication
        flat_valid = candidate_mask.reshape(-1)

        # [T * expanded_topk] corresponding device ids for each token-expert pair
        device_ids_all = torch.div(flat_ids,
                                   self.num_local_experts,
                                   rounding_mode='floor')
        minus_inf = torch.full_like(flat_weights, float('-inf'))

        # fill non-candidate positions with -inf so they will be sorted to the end and dropped first
        # now candidate positions are chosen after topk and expanded, and deduplicated
        sortable_scores = torch.where(flat_valid, flat_weights, minus_inf)

        # sort by scores first
        score_order = torch.argsort(sortable_scores, descending=True)
        sorted_scores_indices = score_order

        # Got sorted in scores device ids of token-expert pair
        # [T * expanded_topk]
        sorted_device_ids = device_ids_all.index_select(0, score_order).to(
            dtype=torch.float32)

        # sort by device id to group tokens of the same device together, 
        # while keeping the score order stable within each device group (stable)
        device_order = torch.argsort(sorted_device_ids, stable=True)

        # got sorted by score and device topk indices, device ids, and valid mask
        sorted_indices = sorted_scores_indices.index_select(0, device_order)
        sorted_device_ids = sorted_device_ids.index_select(0, device_order)
        sorted_valid = flat_valid.index_select(0, sorted_indices)

        sorted_len = sorted_device_ids.shape[0]

        # [T * expanded_topk] arange for indexing
        arange_sorted = torch.arange(sorted_len,
                                     device=device,
                                     dtype=torch.int64)
        # [T * expanded_topk] find group boundaries where device id changes
        group_start_flags = torch.ones_like(sorted_device_ids, dtype=torch.bool)

        # mark the beginning of a device to be true, others are false
        group_start_flags[1:] = sorted_device_ids[1:] != sorted_device_ids[:-1]
        # [Num devices] group start indices in the sorted array
        group_starts = arange_sorted[group_start_flags]
        # [Num devices] group ids for each position in the sorted array
        group_ids = torch.cumsum(group_start_flags.to(torch.int64), dim=0) - 1
        # [Num devices] rank of each token in its device group
        rank_in_device = arange_sorted - group_starts.index_select(0, group_ids)

        keep_in_sorted = (rank_in_device < device_capacity) & sorted_valid
        kept_indices = sorted_indices.index_select(
            0, torch.nonzero(keep_in_sorted, as_tuple=False).squeeze(-1))

        keep_mask_flat = torch.zeros_like(flat_ids, dtype=torch.bool)
        keep_mask_flat.scatter_(0, kept_indices, True)
        keep_mask = keep_mask_flat.view_as(expand_global_topk_ids)

        expanded_global_topk_weights = expanded_global_topk_weights * keep_mask.to(
            expanded_global_topk_weights.dtype)
        expand_global_topk_ids = expand_global_topk_ids.masked_fill(
            ~keep_mask, self.num_experts)

        row_sum = expanded_global_topk_weights.sum(dim=-1, keepdim=True)
        eps = torch.finfo(expanded_global_topk_weights.dtype).eps
        nonzero_mask = row_sum > eps
        expanded_global_topk_weights = torch.where(
            nonzero_mask,
            expanded_global_topk_weights / row_sum.clamp_min(eps),
            torch.zeros_like(expanded_global_topk_weights),
        )
        return expanded_global_topk_weights, expand_global_topk_ids

    def _compute_num_global_tokens_per_expert_after_drop(
            self,
            global_topk_ids_after_drop: torch.Tensor,  # [T, expanded_topk]
            num_tokens_across_dp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute token count per global expert after expanded drop.

        Returns:
            num_global_tokens_per_expert_after_drop: [ep_size, num_experts]
        """
        num_global_tokens_per_expert_after_drop = torch.zeros(
            (self.ep_size, self.num_experts),
            dtype=torch.int64,
            device=global_topk_ids_after_drop.device,
        )

        token_ranges = self._rank_token_ranges(num_tokens_across_dp)
        for rank, (start, end) in enumerate(token_ranges):
            current_topk_ids = global_topk_ids_after_drop[start:end, :]

            # Flatten and filter sentinel values
            flat_expert_ids = current_topk_ids.reshape(-1).to(torch.int64)
            valid_mask = (flat_expert_ids < self.num_experts).to(torch.int64)
            safe_expert_ids = torch.clamp(flat_expert_ids, min=0, max=self.num_experts - 1)

            per_rank_counts = torch.zeros(
                self.num_experts,
                dtype=torch.int64,
                device=global_topk_ids_after_drop.device,
            )
            per_rank_counts.scatter_add_(0, safe_expert_ids, valid_mask)
            num_global_tokens_per_expert_after_drop[rank] = per_rank_counts

        return num_global_tokens_per_expert_after_drop

    def _preprocess_with_expanded_drop(
            self,
            topk_ids: torch.Tensor,                     # [T_local, topk]
            topk_weights: torch.Tensor,                 # [T_local, topk]
            router_logits: torch.Tensor,                # [T_local, num_experts]
            step: int = 0,
    ):
        """
        Preprocess for expanded drop strategy.

        Returns:
            num_tokens_per_local_expert: [num_local_experts]
            input_splits: numpy array [ep_size]
            output_splits: numpy array [ep_size]
            num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            global_input_tokens_local_experts_indices: [num_received_tokens] or None
            num_out_tokens_before_drop: T_local * expanded_topk
            num_out_tokens_after_drop: actual kept token count
            topk_ids_after_drop: [T_local, expanded_topk]
            topk_weights_after_drop: [T_local, expanded_topk]
            expert_capacity: int
            num_global_tokens_per_expert_before_drop: [ep_size, num_experts]
        """
        device = topk_ids.device

        # Step 1: Statistics before drop (based on original topk_ids)
        num_local_tokens_per_expert = torch.histc(
            topk_ids, bins=self.num_experts, min=0, max=self.num_experts
        ).to(torch.int64)

        num_global_tokens_per_expert_before_drop = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert, group=self.ep_group
        ).reshape(self.ep_size, self.num_experts).to(torch.int64)

        # Step 2: Get global token distribution
        num_tokens_across_dp = get_forward_context().dp_metadata.num_tokens_across_dp_cpu

        # Step 3: Global gather - shape adaptation for expanded_topk
        # Note: router_logits shape is [T, num_experts], no expansion needed
        T_global = int(num_tokens_across_dp.sum().item())

        global_router_logits = torch.zeros(
            (T_global, self.num_experts), dtype=router_logits.dtype, device=device
        )

        # Gather across EP ranks
        # Note: We only gather the original topk portion; expanded portion will be filled locally
        # Ensure tensors are contiguous for all_gather_into_tensor_uneven
        # Create contiguous output tensors for all_gather, then copy to global tensors
        global_topk_ids = torch.zeros(
            (T_global, self.top_k), dtype=topk_ids.dtype, device=device
        )
        global_topk_weights = torch.zeros(
            (T_global, self.top_k), dtype=topk_weights.dtype, device=device
        )
        torch_npu.distributed.all_gather_into_tensor_uneven(
            global_topk_ids, topk_ids, num_tokens_across_dp.numpy(), group=self.ep_group
        )
        torch_npu.distributed.all_gather_into_tensor_uneven(
            global_topk_weights, topk_weights, num_tokens_across_dp.numpy(), group=self.ep_group
        )
        torch_npu.distributed.all_gather_into_tensor_uneven(
            global_router_logits, router_logits, num_tokens_across_dp.numpy(), group=self.ep_group
        )

        # Step 4: Calculate expert capacity
        expert_capacity = math.ceil(
            T_global * self.top_k * self.token_drop_load_factor / self.num_experts
        )
        device_capacity = math.ceil(expert_capacity * self.num_local_experts)

        # Step 5: Execute expanded drop strategy.
        if self.drop_strategy == "expert_expanded_drop":
            global_topk_weights_after_drop, global_topk_ids_after_drop = \
                self._get_topk_ids_weights_after_expanded_drop_by_expert(
                    global_topk_ids,
                    global_topk_weights,
                    global_router_logits,
                    expert_capacity,
                    num_tokens_across_dp,
                )
        elif self.drop_strategy == "device_expanded_drop":
            global_topk_weights_after_drop, global_topk_ids_after_drop = \
                self._get_topk_ids_weights_after_expanded_drop_by_device(
                    global_topk_ids,
                    global_topk_weights,
                    global_router_logits,
                    device_capacity,
                    num_tokens_across_dp,
                )
        else:
            raise ValueError(
                f"Invalid expanded drop strategy: {self.drop_strategy}")

        # Step 6: Compute after_drop statistics
        num_global_tokens_per_expert_after_drop = self._compute_num_global_tokens_per_expert_after_drop(
            global_topk_ids_after_drop, num_tokens_across_dp
        )

        # Step 7: Calculate splits
        # Extract local expert columns
        local_expert_start = self.local_expert_indices[0]
        local_expert_end = self.local_expert_indices[-1] + 1
        num_global_tokens_per_local_expert = num_global_tokens_per_expert_after_drop[:, local_expert_start:local_expert_end]

        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)

        output_splits = num_global_tokens_per_local_expert.sum(dim=-1).to(
            torch.device("cpu"), non_blocking=True
        )

        input_splits = num_global_tokens_per_expert_after_drop[self.ep_rank].reshape(
            self.ep_size, self.num_local_experts
        ).sum(dim=1).to(torch.device("cpu"), non_blocking=True)

        # Step 8: Compute global_input_tokens_local_experts_indices
        global_input_tokens_local_experts_indices = None
        if self.num_local_experts > 1:
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                num_global_tokens_per_local_expert.flatten()
            )


        # Step 9: Extract current rank's after_drop results
        token_ranges = self._rank_token_ranges(num_tokens_across_dp)
        this_rank_start, this_rank_end = token_ranges[self.ep_rank]
        topk_ids_after_drop = global_topk_ids_after_drop[this_rank_start:this_rank_end, :]
        topk_weights_after_drop = global_topk_weights_after_drop[this_rank_start:this_rank_end, :]

        # Step 10: Calculate num_out_tokens
        T_local = topk_ids.shape[0]
        num_out_tokens_before_drop = T_local * self.expanded_topk
        num_out_tokens_after_drop = int(input_splits.sum().item())

        if self.token_drop_logging:
            self._log_token_drop_statistics(
                num_global_tokens_per_expert_before_drop=num_global_tokens_per_expert_before_drop,
                num_global_tokens_per_expert_after_drop=num_global_tokens_per_expert_after_drop,
                expert_capacity=expert_capacity,
                device_capacity=device_capacity,
                step=step,
            )

        return (
            num_tokens_per_local_expert,
            input_splits.numpy(),
            output_splits.numpy(),
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
            num_out_tokens_before_drop,
            num_out_tokens_after_drop,
            topk_ids_after_drop,
            topk_weights_after_drop,
            expert_capacity,
            num_global_tokens_per_expert_before_drop,
        )

    def token_dispatch(
            self,
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            expert_map: Optional[torch.Tensor] = None,
            global_redundant_expert_num: int = 0,
            mc2_mask: Optional[torch.Tensor] = None,
            apply_router_weight_on_input: bool = False,
            with_quant: bool = False,
            dynamic_eplb: bool = False,
            pertoken_scale: Optional[torch.Tensor] = None,
            router_logits: Optional[torch.Tensor] = None,
    ):
        """
        Token dispatch for expanded drop strategy.

        Key steps:
        1. Preprocess with expanded drop to get topk_ids_after_drop [T, expanded_topk]
        2. Permute with expanded topk_ids (sentinel tokens placed at end)
        3. Truncate to keep only valid tokens (num_out_tokens_after_drop)
        4. All2all communication
        5. Postprocess for local expert grouping

        Returns:
            TokenDispatchResult with context_metadata for combine
        """
        self.with_quant = with_quant
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        self.hidden_shape_before_permute = hidden_states.shape

        # Track step for logging
        ctx = get_forward_context()
        step = -1
        if not (ctx.is_graph_warmup or ctx.capturing or ctx.in_profile_run):
            self.token_drop_step += 1
            step = self.token_drop_step

        # Step 1: Preprocess with expanded drop
        (num_tokens_per_local_expert,
         input_splits,
         output_splits,
         num_global_tokens_per_local_expert,
         global_input_tokens_local_experts_indices,
         num_out_tokens_before_drop,
         num_out_tokens_after_drop,
         topk_ids_after_drop,
         topk_weights_after_drop,
         expert_capacity,
         num_global_tokens_per_expert_before_drop,
        ) = self._preprocess_with_expanded_drop(topk_ids, topk_weights, router_logits, step)

        self.num_out_tokens_before_drop = num_out_tokens_before_drop
        self.num_out_tokens = num_out_tokens_after_drop

        # Step 2: Permute - directly pass expanded topk_ids
        # npu_moe_token_permute expands all token-expert pairs
        # Note: Sentinel pairs are NOT at the end - they are intermixed
        permutated_local_input_tokens, reversed_local_input_permutation_mapping = \
            torch_npu.npu_moe_token_permute(
                tokens=hidden_states,
                indices=topk_ids_after_drop,
                num_out_tokens=num_out_tokens_before_drop,
            )

        # Step 3: Compute valid permuted positions
        # permutated tokens are expert major, so droppedd pairs (with sentinel num_experts) will be grouped together at the end of permutated buffer
        permutated_local_input_tokens = permutated_local_input_tokens[:self.num_out_tokens]

        # dont have to do the selection since we will append to expanded version before drop, then do the combine with the full reversed mapping.
        # reversed_local_input_permutation_mapping = reversed_local_input_permutation_mapping.index_select(
        #     0, valid_permuted_indices
        # )
        full_reversed_local_input_permutation_mapping = reversed_local_input_permutation_mapping

        # Step 4: All2all communication
        dynamic_scale_after_all2all = None
        if self.with_quant:
            permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_local_input_tokens
            )
            _, dynamic_scale_after_all2all, quant_all2all_handle = async_all_to_all(
                dynamic_scale, output_splits, input_splits, self.ep_group
            )
            quant_all2all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        _, global_input_tokens, all2all_handle = async_all_to_all(
            permutated_local_input_tokens, output_splits, input_splits, self.ep_group
        )
        all2all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # Step 5: Postprocess - local expert grouping
        global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = \
            self._dispatch_postprocess(
                global_input_tokens,
                dynamic_scale_after_all2all,
                global_input_tokens_local_experts_indices,
            )

        # Step 6: Build context_metadata
        context_metadata = {
            "input_splits": input_splits,
            "output_splits": output_splits,
            "topk_weights": topk_weights,  # Original topk_weights for reference
            "expanded_topk_weights": topk_weights_after_drop,  # Expanded weights for combine
            "full_reversed_local_input_permutation_mapping": full_reversed_local_input_permutation_mapping,  # Full version for unpermute
            "reversed_global_input_permutation_mapping": reversed_global_input_permutation_mapping,
            "num_out_tokens_before_drop": num_out_tokens_before_drop,
            "num_out_tokens_after_drop": num_out_tokens_after_drop,
            "original_topk": self.top_k,
            "expanded_topk": self.expanded_topk,
            "drop_strategy": self.drop_strategy,
            "expert_capacity": expert_capacity,
            "device_capacity": expert_capacity * self.num_local_experts,
            "num_global_tokens_per_expert_before_drop": num_global_tokens_per_expert_before_drop,
            "num_global_tokens_per_local_expert_after_drop": num_global_tokens_per_local_expert,
        }

        return TokenDispatchResult(
            hidden_states=global_input_tokens,
            dynamic_scale=dynamic_scale_final,
            group_list=num_tokens_per_local_expert,
            group_list_type=1,
            context_metadata=context_metadata,
        )

    @override
    def token_combine(
            self,
            hidden_states: torch.Tensor,
            context_metadata: dict,
            bias: Optional[torch.Tensor] = None,
    ):
        """
        Token combine for expanded drop strategy.

        Key steps:
        1. Preprocess - undo local expert grouping
        2. All2all - send tokens back to source ranks
        3. Reconstruct - place tokens back to valid permuted positions
        4. Create full tensor - for unpermute operation
        5. Unpermute - restore original token order with expanded weights

        Returns:
            TokenCombineResult with restored hidden states
        """
        assert bias is None, "Bias is not supported in expanded drop dispatcher."

        # Step 1: Preprocess - undo local expert grouping
        # unpermute to ep rank sequence, use the mapping of expanded permute
        hidden_states = self._combine_preprocess(hidden_states, context_metadata)

        # Step 2: All2all - send tokens back to source ranks
        _, permutated_local_input_tokens, all2all_handle = async_all_to_all(
            hidden_states,
            context_metadata["input_splits"],
            context_metadata["output_splits"],
            self.ep_group,
        )
        all2all_handle.wait()
        hidden_states.untyped_storage().resize_(0)

        # Step 3: Reconstruct full permuted tensor
        # We need to place tokens back to their original permuted positions
        num_out_tokens_before_drop = context_metadata["num_out_tokens_before_drop"]

        restored_local_tokens = torch.zeros(
            (num_out_tokens_before_drop, permutated_local_input_tokens.shape[-1]),
            dtype=permutated_local_input_tokens.dtype,
            device=permutated_local_input_tokens.device,
        )

        # restore tokens to expanded permuted positions before drop
        restored_local_tokens[:self.num_out_tokens] = permutated_local_input_tokens
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # Step 4: Unpermute - restore original token order
        # Use the FULL reversed_mapping saved during dispatch
        full_reversed_mapping = context_metadata["full_reversed_local_input_permutation_mapping"]

        # Use expanded topk_weights for weighted aggregation
        expanded_topk_weights = context_metadata["expanded_topk_weights"]

        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=restored_local_tokens,
            sorted_indices=full_reversed_mapping.to(torch.int32),
            probs=expanded_topk_weights,
            restore_shape=self.hidden_shape_before_permute,
        )

        output = output.view(self.hidden_shape)
        return TokenCombineResult(routed_out=output)