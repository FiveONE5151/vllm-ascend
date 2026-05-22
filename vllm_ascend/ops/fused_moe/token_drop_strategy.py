#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Token drop strategies for MoE expert selection.

This module implements the Strategy pattern for token drop logic,
separating it from the dispatcher. Strategies are applied during
expert selection (in experts_selector), before the dispatcher
handles communication.
"""

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch_npu.distributed
from vllm.distributed.parallel_state import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.ops.fused_moe.comm_utils import (
    gather_from_sequence_parallel_region,
)
from vllm_ascend.ops.fused_moe.token_drop_utils import (
    compute_expanded_drop_statistics,
    compute_num_global_tokens_per_expert_after_drop,
    log_expanded_drop_statistics,
    log_token_drop_statistics,
    rank_token_ranges,
    renormalize_topk_weights,
)


_GLOBAL_TOKEN_DROP_STRATEGY: Optional["TokenDropStrategy"] = None
_GLOBAL_TOKEN_DROP_STRATEGY_CONFIG: Optional[tuple] = None


@dataclass
class TokenDropResult:
    """Result of applying a token drop strategy.

    Attributes:
        topk_weights: Router weights after drop [T, topk] or [T, topk + num_local_experts]
        topk_ids: Expert IDs after drop [T, topk] or [T, topk + num_local_experts].
                  Dropped tokens have expert index set to num_experts (sentinel).
        context_metadata: Additional metadata needed by the dispatcher for
                          communication and combine operations.
    """
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    context_metadata: dict = field(default_factory=dict)


class TokenDropStrategy(ABC):
    """Abstract base class for token drop strategies.

    Properties:
        is_expanded: Whether this strategy expands the candidate set
                     (output has topk + num_local_experts columns)
        needs_global_gather: Whether this strategy needs global data
                             from all EP ranks before applying drop
    """

    @property
    @abstractmethod
    def is_expanded(self) -> bool:
        """Whether this strategy produces expanded topk (topk + num_local_experts)."""
        ...

    @property
    @abstractmethod
    def needs_global_gather(self) -> bool:
        """Whether this strategy requires global data from all EP ranks."""
        ...

    @abstractmethod
    def apply(
        self,
        scores: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        num_local_experts: int,
        ep_rank: int,
        ep_size: int,
        global_topk_ids: Optional[torch.Tensor] = None,
        global_topk_weights: Optional[torch.Tensor] = None,
        global_router_logits: Optional[torch.Tensor] = None,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
    ) -> TokenDropResult:
        """Apply the token drop strategy.

        Args:
            scores: Router scores [T_local, num_experts] (softmax or sigmoid output)
            topk_weights: Selected topk weights [T_local, topk]
            topk_ids: Selected topk expert IDs [T_local, topk]
            num_experts: Total number of global experts
            num_local_experts: Number of local experts per rank
            ep_rank: Current EP rank
            ep_size: Total EP world size
            global_topk_ids: Gathered topk IDs [T_global, topk] (if needs_global_gather)
            global_topk_weights: Gathered topk weights [T_global, topk] (if needs_global_gather)
            global_router_logits: Gathered router logits [T_global, num_experts] (if needs_global_gather)
            num_tokens_across_dp: Token counts per rank [ep_size]

        Returns:
            TokenDropResult with processed topk_weights, topk_ids, and metadata
        """
        ...


class ExpertDropStrategy(TokenDropStrategy):
    """Drop tokens by expert capacity.

    For each expert, keep at most `expert_capacity` tokens based on their
    routing weights. Tokens exceeding the capacity are dropped (weight set
    to 0, expert id set to num_experts sentinel).

    Output shape: [T, topk] (same as input)
    """

    def __init__(self, load_factor: float = 1.0, **kwargs):
        self.load_factor = load_factor
        self.token_drop_logging = os.getenv("VLLM_TOKEN_DROP_LOGGING", "0") == "1"
        self.token_drop_csv_dir = os.getenv("VLLM_TOKEN_DROP_CSV_DIR", "")
        self._step = 0

    @property
    def is_expanded(self) -> bool:
        return False

    @property
    def needs_global_gather(self) -> bool:
        return True

    def apply(
        self,
        scores: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        num_local_experts: int,
        ep_rank: int,
        ep_size: int,
        global_topk_ids: Optional[torch.Tensor] = None,
        global_topk_weights: Optional[torch.Tensor] = None,
        global_router_logits: Optional[torch.Tensor] = None,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
    ) -> TokenDropResult:
        assert global_topk_ids is not None
        assert global_topk_weights is not None
        assert num_tokens_across_dp is not None

        topk = topk_weights.shape[1]
        T_global = global_topk_ids.shape[0]

        # Compute capacities
        expert_capacity = math.ceil(T_global * topk * self.load_factor / num_experts)
        device_capacity = math.ceil(expert_capacity * num_local_experts)

        # Apply expert-level capacity drop
        global_topk_weights_after_drop, global_topk_ids_after_drop = \
            self._drop_by_expert(global_topk_ids, global_topk_weights,
                                 expert_capacity, num_experts)

        # Extract this rank's results
        token_ranges_list = rank_token_ranges(num_tokens_across_dp, ep_size)
        this_rank_start, this_rank_end = token_ranges_list[ep_rank]
        this_rank_topk_ids_after_drop = global_topk_ids_after_drop[this_rank_start:this_rank_end, :]
        this_rank_topk_weights_after_drop = global_topk_weights_after_drop[this_rank_start:this_rank_end, :]


        if self.token_drop_logging:
            # Logging
            step = self._maybe_increment_step()
            # Compute per-rank after-drop results
            num_global_tokens_per_expert_after_drop = compute_num_global_tokens_per_expert_after_drop(
            global_topk_ids_after_drop, num_tokens_across_dp, num_experts, ep_size)

            # Compute global before-drop stats for logging
            num_local_tokens_per_expert = torch.histc(
                topk_ids, bins=num_experts, min=0, max=num_experts).to(torch.int64)
            num_global_tokens_per_expert_before_drop = gather_from_sequence_parallel_region(
                num_local_tokens_per_expert,
                group=get_ep_group().device_group).reshape(ep_size, num_experts).to(torch.int64)

            log_token_drop_statistics(
                ep_rank=ep_rank,
                ep_size=ep_size,
                num_global_tokens_per_expert_before_drop=num_global_tokens_per_expert_before_drop,
                num_global_tokens_per_expert_after_drop=num_global_tokens_per_expert_after_drop,
                expert_capacity=expert_capacity,
                device_capacity=device_capacity,
                step=step,
                token_drop_csv_dir=self.token_drop_csv_dir,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
            )

        return this_rank_topk_weights_after_drop, this_rank_topk_ids_after_drop

    def _drop_by_expert(
        self,
        global_topk_ids: torch.Tensor,
        global_topk_weights: torch.Tensor,
        expert_capacity: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply expert-level capacity drop on global tensors."""
        mask_buffer = torch.zeros((global_topk_ids.shape[0], num_experts),
                                  dtype=torch.bool,
                                  device=global_topk_ids.device)
        scores_buffer = torch.zeros((global_topk_ids.shape[0], num_experts),
                                    dtype=global_topk_weights.dtype,
                                    device=global_topk_ids.device)

        # Mark token-expert positions selected by gating topk
        mask_buffer.scatter_(-1, global_topk_ids, True)
        scores_buffer.scatter_(-1, global_topk_ids, global_topk_weights)

        capacity = min(expert_capacity, scores_buffer.shape[0])

        # Fill non-selected positions with -inf for capacity topk
        masked_scores = scores_buffer.masked_fill(~mask_buffer, float('-inf'))
        _, capacity_indices = torch.topk(masked_scores,
                                          k=capacity,
                                          dim=0,
                                          sorted=False)

        # Positions that are both selected and within capacity
        kept_mask = torch.zeros_like(mask_buffer).scatter(
            0, capacity_indices, True) & mask_buffer

        # Gather back to topk view
        top_mask = kept_mask.gather(-1, global_topk_ids)

        # Zero out dropped weights
        global_topk_weights = global_topk_weights * top_mask.to(global_topk_weights.dtype)

        # Mark dropped tokens with sentinel
        global_topk_ids = global_topk_ids.masked_fill(~top_mask, num_experts)

        return global_topk_weights, global_topk_ids

    def _maybe_increment_step(self) -> int:
        ctx = get_forward_context()
        if not (ctx.is_graph_warmup or ctx.capturing or ctx.in_profile_run):
            self._step += 1
        return self._step


class DeviceDropStrategy(TokenDropStrategy):
    """Drop tokens by device capacity.

    Sort all token-expert pairs by score (desc), then by device id (asc),
    and keep at most `device_capacity` tokens per device.

    Output shape: [T, topk] (same as input)
    """

    def __init__(self, load_factor: float = 1.0, **kwargs):
        self.load_factor = load_factor
        self.token_drop_logging = os.getenv("VLLM_TOKEN_DROP_LOGGING", "0") == "1"
        self.token_drop_csv_dir = os.getenv("VLLM_TOKEN_DROP_CSV_DIR", "")
        self._step = 0

    @property
    def is_expanded(self) -> bool:
        return False

    @property
    def needs_global_gather(self) -> bool:
        return True

    def apply(
        self,
        scores: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        num_local_experts: int,
        ep_rank: int,
        ep_size: int,
        global_topk_ids: Optional[torch.Tensor] = None,
        global_topk_weights: Optional[torch.Tensor] = None,
        global_router_logits: Optional[torch.Tensor] = None,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
    ) -> TokenDropResult:
        assert global_topk_ids is not None
        assert global_topk_weights is not None
        assert num_tokens_across_dp is not None

        topk = topk_weights.shape[1]
        T_global = global_topk_ids.shape[0]

        # Compute capacities
        expert_capacity = math.ceil(T_global * topk * self.load_factor / num_experts)
        device_capacity = math.ceil(expert_capacity * num_local_experts)

        # Apply device-level capacity drop
        global_topk_weights_after_drop, global_topk_ids_after_drop = \
            self._drop_by_device(global_topk_ids, global_topk_weights,
                                 device_capacity, num_experts, num_local_experts)

        # Extract this rank's results
        token_ranges_list = rank_token_ranges(num_tokens_across_dp, ep_size)
        this_rank_start, this_rank_end = token_ranges_list[ep_rank]
        this_rank_topk_ids_after_drop = global_topk_ids_after_drop[this_rank_start:this_rank_end, :]
        this_rank_topk_weights_after_drop = global_topk_weights_after_drop[this_rank_start:this_rank_end, :]

        if self.token_drop_logging:
            # Logging
            step = self._maybe_increment_step()

            # Compute per-rank after-drop results
            num_global_tokens_per_expert_after_drop = compute_num_global_tokens_per_expert_after_drop(
                global_topk_ids_after_drop, num_tokens_across_dp, num_experts, ep_size)

            # Compute global before-drop stats for logging
            num_local_tokens_per_expert = torch.histc(
                topk_ids, bins=num_experts, min=0, max=num_experts).to(torch.int64)
            num_global_tokens_per_expert_before_drop = gather_from_sequence_parallel_region(
                num_local_tokens_per_expert,
                group=get_ep_group().device_group).reshape(ep_size, num_experts).to(torch.int64)

            log_token_drop_statistics(
                ep_rank=ep_rank,
                ep_size=ep_size,
                num_global_tokens_per_expert_before_drop=num_global_tokens_per_expert_before_drop,
                num_global_tokens_per_expert_after_drop=num_global_tokens_per_expert_after_drop,
                expert_capacity=expert_capacity,
                device_capacity=device_capacity,
                step=step,
                token_drop_csv_dir=self.token_drop_csv_dir,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
            )

        return this_rank_topk_weights_after_drop, this_rank_topk_ids_after_drop

    def _drop_by_device(
        self,
        global_topk_ids: torch.Tensor,
        global_topk_weights: torch.Tensor,
        device_capacity: int,
        num_experts: int,
        num_local_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply device-level capacity drop on global tensors."""
        if global_topk_ids.numel() == 0:
            return global_topk_weights, global_topk_ids

        if device_capacity <= 0:
            return (torch.zeros_like(global_topk_weights),
                    torch.full_like(global_topk_ids, num_experts))

        flat_topk_ids = global_topk_ids.reshape(-1).to(torch.int64)
        flat_topk_weights = global_topk_weights.reshape(-1)

        device_ids_all = torch.div(flat_topk_ids,
                                   num_local_experts,
                                   rounding_mode='floor')

        # 1) Sort by score (desc), 2) stable sort by device id (asc)
        score_order = torch.argsort(flat_topk_weights, descending=True)
        sorted_scores_indices = score_order

        # Cast to fp32 to run argsort on aicore
        sorted_device_ids = device_ids_all.index_select(0, score_order).to(
            dtype=torch.float32)

        device_order = torch.argsort(sorted_device_ids, stable=True)

        sorted_indices = sorted_scores_indices.index_select(0, device_order)
        sorted_device_ids = sorted_device_ids.index_select(0, device_order)

        sorted_len = sorted_device_ids.shape[0]

        arange_sorted = torch.arange(sorted_len,
                                     device=sorted_device_ids.device,
                                     dtype=torch.int64)

        group_start_flags = torch.ones_like(sorted_device_ids, dtype=torch.bool)
        group_start_flags[1:] = sorted_device_ids[1:] != sorted_device_ids[:-1]

        group_starts = arange_sorted[group_start_flags]
        group_ids = torch.cumsum(group_start_flags.to(torch.int64), dim=0) - 1
        rank_in_device = arange_sorted - group_starts.index_select(0, group_ids)

        keep_in_sorted = rank_in_device < device_capacity

        kept_indices = sorted_indices.index_select(
            0, torch.nonzero(keep_in_sorted, as_tuple=False).squeeze(-1))
        keep_mask_flat = torch.zeros_like(flat_topk_ids,
                                          dtype=torch.bool,
                                          device=flat_topk_ids.device)
        keep_mask_flat.scatter_(0, kept_indices, True)

        keep_mask = keep_mask_flat.view_as(global_topk_ids)
        global_topk_weights = global_topk_weights * keep_mask.to(global_topk_weights.dtype)
        global_topk_ids = global_topk_ids.masked_fill(~keep_mask, num_experts)

        return global_topk_weights, global_topk_ids

    def _maybe_increment_step(self) -> int:
        ctx = get_forward_context()
        if not (ctx.is_graph_warmup or ctx.capturing or ctx.in_profile_run):
            self._step += 1
        return self._step


class ExpertDropLocalStrategy(TokenDropStrategy):
    """Local-only expert drop strategy.

    Applies expert-level capacity drop using only local rank data,
    without gathering global data from other EP ranks.

    Output shape: [T, topk] (same as input)
    """

    def __init__(self, load_factor: float = 1.0, **kwargs):
        self.load_factor = load_factor
        self.token_drop_logging = os.getenv("VLLM_TOKEN_DROP_LOGGING", "0") == "1"
        self.token_drop_csv_dir = os.getenv("VLLM_TOKEN_DROP_CSV_DIR", "")
        self._step = 0

    @property
    def is_expanded(self) -> bool:
        return False

    @property
    def needs_global_gather(self) -> bool:
        return False

    def apply(
        self,
        scores: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        num_local_experts: int,
        ep_rank: int,
        ep_size: int,
        global_topk_ids: Optional[torch.Tensor] = None,
        global_topk_weights: Optional[torch.Tensor] = None,
        global_router_logits: Optional[torch.Tensor] = None,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
    ) -> TokenDropResult:
        topk = topk_weights.shape[1]

        total_tokens = topk_weights.shape[0]

        expert_capacity = math.ceil(total_tokens * topk * self.load_factor / num_experts)
        device_capacity = math.ceil(expert_capacity * num_local_experts)

        # Apply expert-level drop on local data only
        local_topk_weights_after_drop, local_topk_ids_after_drop = \
            self._drop_by_expert_local(topk_ids, topk_weights,
                                       expert_capacity, num_experts)

        # Compute local after-drop stats
        this_rank_kept_expert_ids = local_topk_ids_after_drop[
            local_topk_ids_after_drop < num_experts]
        num_local_tokens_per_expert_after_drop = torch.bincount(
            this_rank_kept_expert_ids.reshape(-1),
            minlength=num_experts,
        ).to(torch.int64)


        if self.token_drop_logging:
            # Logging
            step = self._maybe_increment_step()

            # Build global after-drop matrix for logging
            num_global_tokens_per_expert_after_drop = gather_from_sequence_parallel_region(
                num_local_tokens_per_expert_after_drop,
                group=get_ep_group().device_group).reshape(ep_size, num_experts).to(torch.int64)

            # Before-drop stats
            num_local_tokens_per_expert = torch.histc(
                topk_ids, bins=num_experts, min=0, max=num_experts).to(torch.int64)
            num_global_tokens_per_expert_before_drop = gather_from_sequence_parallel_region(
                num_local_tokens_per_expert,
                group=get_ep_group().device_group).reshape(ep_size, num_experts).to(torch.int64)

            log_token_drop_statistics(
                ep_rank=ep_rank,
                ep_size=ep_size,
                num_global_tokens_per_expert_before_drop=num_global_tokens_per_expert_before_drop,
                num_global_tokens_per_expert_after_drop=num_global_tokens_per_expert_after_drop,
                expert_capacity=expert_capacity,
                device_capacity=device_capacity,
                step=step,
                token_drop_csv_dir=self.token_drop_csv_dir,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
            )

        return local_topk_weights_after_drop, local_topk_ids_after_drop

    def _drop_by_expert_local(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_capacity: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply expert-level capacity drop on local tensors."""
        mask_buffer = torch.zeros((topk_ids.shape[0], num_experts),
                                  dtype=torch.bool,
                                  device=topk_ids.device)
        scores_buffer = torch.zeros((topk_ids.shape[0], num_experts),
                                    dtype=topk_weights.dtype,
                                    device=topk_weights.device)

        mask_buffer.scatter_(-1, topk_ids, True)
        scores_buffer.scatter_(-1, topk_ids, topk_weights)

        capacity = min(expert_capacity, scores_buffer.shape[0])

        masked_scores = scores_buffer.masked_fill(~mask_buffer, float('-inf'))
        _, capacity_indices = torch.topk(masked_scores,
                                          k=capacity,
                                          dim=0,
                                          sorted=False)

        kept_mask = torch.zeros_like(mask_buffer).scatter(
            0, capacity_indices, True) & mask_buffer

        top_mask = kept_mask.gather(-1, topk_ids)

        topk_weights_after = topk_weights * top_mask.to(topk_weights.dtype)
        topk_ids_after = topk_ids.masked_fill(~top_mask, num_experts)

        return topk_weights_after, topk_ids_after

    def _maybe_increment_step(self) -> int:
        ctx = get_forward_context()
        if not (ctx.is_graph_warmup or ctx.capturing or ctx.in_profile_run):
            self._step += 1
        return self._step


class ExpertExpandedDropStrategy(TokenDropStrategy):
    """Expanded drop strategy by expert capacity.

    Extends each token's candidate expert set by adding all local experts
    of the rank where the token resides, then performs capacity-based drop
    by expert.

    Output shape: [T, topk + num_local_experts]
    """

    def __init__(self, load_factor: float = 1.0, top_k: int = 0, **kwargs):
        self.load_factor = load_factor
        self.top_k = top_k
        self.token_drop_logging = os.getenv("VLLM_TOKEN_DROP_LOGGING", "0") == "1"
        self.token_drop_csv_dir = os.getenv("VLLM_TOKEN_DROP_CSV_DIR", "")
        self._step = 0

    @property
    def is_expanded(self) -> bool:
        return True

    @property
    def needs_global_gather(self) -> bool:
        return True

    def apply(
        self,
        scores: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        num_local_experts: int,
        ep_rank: int,
        ep_size: int,
        global_topk_ids: Optional[torch.Tensor] = None,
        global_topk_weights: Optional[torch.Tensor] = None,
        global_router_logits: Optional[torch.Tensor] = None,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
    ) -> TokenDropResult:
        assert global_router_logits is not None
        assert num_tokens_across_dp is not None

        expanded_topk = self.top_k + num_local_experts
        T = global_router_logits.shape[0]
        device = global_router_logits.device

        # Compute capacities
        expert_capacity = math.ceil(
            T * self.top_k * self.load_factor / num_experts)
        device_capacity = math.ceil(expert_capacity * num_local_experts)

        # Compute topk from router logits (softmax -> topk)
        router_probs = torch.softmax(global_router_logits, dim=-1)
        topk_weights_from_probs, topk_ids_from_probs = torch.topk(
            router_probs.to(torch.float32),
            k=self.top_k,
            dim=-1,
        )
        topk_ids_from_probs = topk_ids_from_probs.to(global_topk_ids.dtype if global_topk_ids is not None else torch.int32)

        # Construct expanded local expert indices
        token_ranges_list = rank_token_ranges(num_tokens_across_dp, ep_size)
        rank_ids = torch.zeros(T, dtype=torch.int64, device=device)
        for rank, (start, end) in enumerate(token_ranges_list):
            rank_ids[start:end] = rank

        rank_experts_start_idx = rank_ids * num_local_experts
        all_local_expert_ids = torch.arange(num_local_experts, device=device)
        local_expert_indices_t = rank_experts_start_idx.unsqueeze(-1) + all_local_expert_ids.unsqueeze(0)

        # Concatenate to get expanded candidate set [T, topk + num_local_experts]
        expand_global_topk_ids = torch.cat([topk_ids_from_probs, local_expert_indices_t], dim=-1)

        # Build masks
        topk_mask_buffer = torch.zeros((T, num_experts),
                                       dtype=torch.bool,
                                       device=device)
        topk_mask_buffer.scatter_(-1, topk_ids_from_probs, True)

        local_mask_buffer = torch.zeros((T, num_experts),
                                        dtype=torch.bool,
                                        device=device)
        local_mask_buffer.scatter_(-1, local_expert_indices_t, True)

        expanded_mask_buffer = topk_mask_buffer | local_mask_buffer

        # Capacity-based topk per expert
        capacity = min(expert_capacity, router_probs.shape[0])
        masked_scores = router_probs.masked_fill(~expanded_mask_buffer, float('-inf'))
        _, capacity_indices = torch.topk(masked_scores, k=capacity, dim=0, sorted=True)

        kept_mask = torch.zeros_like(expanded_mask_buffer).scatter(
            0, capacity_indices, True) & expanded_mask_buffer

        # Gather back to expanded candidate view
        top_mask = kept_mask.gather(-1, expand_global_topk_ids)
        expanded_global_topk_weights = router_probs.gather(-1, expand_global_topk_ids)

        # Deduplicate: for expanded positions, check overlap with original topk
        duplicate_suppression_count = 0
        for pos in range(self.top_k, expanded_topk):
            duplicate_mask = (expand_global_topk_ids[:, pos:pos+1] == expand_global_topk_ids[:, :pos]).any(dim=-1)
            top_mask[:, pos] = top_mask[:, pos] & ~duplicate_mask
            duplicate_suppression_count += duplicate_mask.sum().item()

        # Compute expanded drop statistics (using raw softmax probabilities)
        expanded_drop_stats = None
        if self.token_drop_logging:
            expanded_drop_stats = compute_expanded_drop_statistics(
                router_probs=router_probs,
                expand_global_topk_ids=expand_global_topk_ids,
                top_mask=top_mask,
                top_k=self.top_k,
                duplicate_suppression_count=duplicate_suppression_count,
            )

        # Apply mask
        expanded_global_topk_weights = expanded_global_topk_weights * top_mask.to(
            topk_weights_from_probs.dtype)
        expand_global_topk_ids = expand_global_topk_ids.masked_fill(~top_mask, num_experts)

        # Safe renormalize
        expanded_global_topk_weights = renormalize_topk_weights(expanded_global_topk_weights)

        # Extract this rank's results
        this_rank_start, this_rank_end = token_ranges_list[ep_rank]
        topk_ids_after_drop = expand_global_topk_ids[this_rank_start:this_rank_end, :]
        topk_weights_after_drop = expanded_global_topk_weights[this_rank_start:this_rank_end, :]

        if self.token_drop_logging:
            # Logging
            step = self._maybe_increment_step()

            # Compute after-drop stats
            num_global_tokens_per_expert_after_drop = compute_num_global_tokens_per_expert_after_drop(
                expand_global_topk_ids, num_tokens_across_dp, num_experts, ep_size)

            # Before-drop stats
            num_local_tokens_per_expert = torch.histc(
                topk_ids, bins=num_experts, min=0, max=num_experts).to(torch.int64)
            num_global_tokens_per_expert_before_drop = gather_from_sequence_parallel_region(
                num_local_tokens_per_expert,
                group=get_ep_group().device_group).reshape(ep_size, num_experts).to(torch.int64)

            log_token_drop_statistics(
                ep_rank=ep_rank,
                ep_size=ep_size,
                num_global_tokens_per_expert_before_drop=num_global_tokens_per_expert_before_drop,
                num_global_tokens_per_expert_after_drop=num_global_tokens_per_expert_after_drop,
                expert_capacity=expert_capacity,
                device_capacity=device_capacity,
                step=step,
                token_drop_csv_dir=self.token_drop_csv_dir,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
            )

            # Log expanded drop statistics
            if expanded_drop_stats is not None:
                log_expanded_drop_statistics(
                    ep_rank=ep_rank,
                    step=step,
                    stats=expanded_drop_stats,
                    token_drop_csv_dir=self.token_drop_csv_dir,
                )

        return topk_weights_after_drop, topk_ids_after_drop

    def _maybe_increment_step(self) -> int:
        ctx = get_forward_context()
        if not (ctx.is_graph_warmup or ctx.capturing or ctx.in_profile_run):
            self._step += 1
        return self._step


class DeviceExpandedDropStrategy(TokenDropStrategy):
    """Expanded drop strategy by device capacity.

    Extends each token's candidate expert set by adding all local experts
    of the rank where the token resides, then performs capacity-based drop
    by device.

    Output shape: [T, topk + num_local_experts]
    """

    def __init__(self, load_factor: float = 1.0, top_k: int = 0, **kwargs):
        self.load_factor = load_factor
        self.top_k = top_k
        self.token_drop_logging = os.getenv("VLLM_TOKEN_DROP_LOGGING", "0") == "1"
        self.token_drop_csv_dir = os.getenv("VLLM_TOKEN_DROP_CSV_DIR", "")
        self._step = 0

    @property
    def is_expanded(self) -> bool:
        return True

    @property
    def needs_global_gather(self) -> bool:
        return True

    def apply(
        self,
        scores: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        num_local_experts: int,
        ep_rank: int,
        ep_size: int,
        global_topk_ids: Optional[torch.Tensor] = None,
        global_topk_weights: Optional[torch.Tensor] = None,
        global_router_logits: Optional[torch.Tensor] = None,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
    ) -> TokenDropResult:
        assert global_router_logits is not None
        assert num_tokens_across_dp is not None

        expanded_topk = self.top_k + num_local_experts
        T = global_router_logits.shape[0]
        device = global_router_logits.device

        if T == 0:
            return TokenDropResult(
                topk_weights=torch.zeros((0, expanded_topk),
                                         dtype=torch.float32,
                                         device=device),
                topk_ids=torch.zeros((0, expanded_topk),
                                     dtype=torch.int64,
                                     device=device),
                context_metadata={},
            )

        # Compute capacities
        expert_capacity = math.ceil(
            T * self.top_k * self.load_factor / num_experts)
        device_capacity = math.ceil(expert_capacity * num_local_experts)

        # Compute topk from router logits
        router_probs = torch.softmax(global_router_logits, dim=-1)
        _, topk_ids_from_probs = torch.topk(
            router_probs.to(torch.float32),
            k=self.top_k,
            dim=-1,
        )
        topk_ids_from_probs = topk_ids_from_probs.to(torch.int64)

        # Construct expanded local expert indices
        token_ranges_list = rank_token_ranges(num_tokens_across_dp, ep_size)
        rank_ids = torch.zeros(T, dtype=torch.int64, device=device)
        for rank, (start, end) in enumerate(token_ranges_list):
            rank_ids[start:end] = rank

        rank_experts_start_idx = rank_ids * num_local_experts
        all_local_expert_ids = torch.arange(num_local_experts,
                                            device=device,
                                            dtype=torch.int64)
        local_expert_indices_t = rank_experts_start_idx.unsqueeze(
            -1) + all_local_expert_ids.unsqueeze(0)

        expand_global_topk_ids = torch.cat(
            [topk_ids_from_probs, local_expert_indices_t], dim=-1)
        expanded_global_topk_weights = router_probs.gather(
            -1, expand_global_topk_ids)

        # Keep topk positions as primary when expanded local experts duplicate
        candidate_mask = torch.ones_like(expand_global_topk_ids,
                                         dtype=torch.bool,
                                         device=device)

        # Deduplication
        for pos in range(self.top_k, expanded_topk):
            duplicate_mask = (
                expand_global_topk_ids[:, pos:pos + 1]
                == expand_global_topk_ids[:, :pos]).any(dim=-1)
            candidate_mask[:, pos] = candidate_mask[:, pos] & ~duplicate_mask

        if device_capacity <= 0:
            return TokenDropResult(
                topk_weights=torch.zeros_like(expanded_global_topk_weights),
                topk_ids=torch.full_like(expand_global_topk_ids, num_experts),
                context_metadata={},
            )

        # Flatten for sorting and selection
        flat_ids = expand_global_topk_ids.reshape(-1)
        flat_weights = expanded_global_topk_weights.reshape(-1)
        flat_valid = candidate_mask.reshape(-1)

        device_ids_all = torch.div(flat_ids,
                                   num_local_experts,
                                   rounding_mode='floor')
        minus_inf = torch.full_like(flat_weights, float('-inf'))

        sortable_scores = torch.where(flat_valid, flat_weights, minus_inf)

        # Sort by scores first
        score_order = torch.argsort(sortable_scores, descending=True)
        sorted_scores_indices = score_order

        sorted_device_ids = device_ids_all.index_select(0, score_order).to(
            dtype=torch.float32)

        # Sort by device id to group tokens of the same device
        device_order = torch.argsort(sorted_device_ids, stable=True)

        sorted_indices = sorted_scores_indices.index_select(0, device_order)
        sorted_device_ids = sorted_device_ids.index_select(0, device_order)
        sorted_valid = flat_valid.index_select(0, sorted_indices)

        sorted_len = sorted_device_ids.shape[0]

        arange_sorted = torch.arange(sorted_len,
                                     device=device,
                                     dtype=torch.int64)
        group_start_flags = torch.ones_like(sorted_device_ids, dtype=torch.bool)
        group_start_flags[1:] = sorted_device_ids[1:] != sorted_device_ids[:-1]
        group_starts = arange_sorted[group_start_flags]
        group_ids = torch.cumsum(group_start_flags.to(torch.int64), dim=0) - 1
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
            ~keep_mask, num_experts)

        # Safe renormalize
        expanded_global_topk_weights = renormalize_topk_weights(expanded_global_topk_weights)

        # Extract this rank's results
        this_rank_start, this_rank_end = token_ranges_list[ep_rank]
        topk_ids_after_drop = expand_global_topk_ids[this_rank_start:this_rank_end, :]
        topk_weights_after_drop = expanded_global_topk_weights[this_rank_start:this_rank_end, :]

        if self.token_drop_logging:
            # Logging
            step = self._maybe_increment_step()

            # Compute after-drop stats
            num_global_tokens_per_expert_after_drop = compute_num_global_tokens_per_expert_after_drop(
                expand_global_topk_ids, num_tokens_across_dp, num_experts, ep_size)

            # Before-drop stats
            num_local_tokens_per_expert = torch.histc(
                topk_ids, bins=num_experts, min=0, max=num_experts).to(torch.int64)
            num_global_tokens_per_expert_before_drop = gather_from_sequence_parallel_region(
                num_local_tokens_per_expert,
                group=get_ep_group().device_group).reshape(ep_size, num_experts).to(torch.int64)

            log_token_drop_statistics(
                ep_rank=ep_rank,
                ep_size=ep_size,
                num_global_tokens_per_expert_before_drop=num_global_tokens_per_expert_before_drop,
                num_global_tokens_per_expert_after_drop=num_global_tokens_per_expert_after_drop,
                expert_capacity=expert_capacity,
                device_capacity=device_capacity,
                step=step,
                token_drop_csv_dir=self.token_drop_csv_dir,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
            )

        return topk_weights_after_drop, topk_ids_after_drop

    def _maybe_increment_step(self) -> int:
        ctx = get_forward_context()
        if not (ctx.is_graph_warmup or ctx.capturing or ctx.in_profile_run):
            self._step += 1
        return self._step


def create_token_drop_strategy(
    strategy_name: str,
    load_factor: float = 1.0,
    local_only: bool = False,
    top_k: int = 0,
    **kwargs,
) -> TokenDropStrategy:
    """Factory function to create a token drop strategy.

    Args:
        strategy_name: Name of the strategy ("expert_drop", "device_drop",
                       "expert_expanded_drop", "device_expanded_drop")
        load_factor: Token drop load factor
        local_only: Whether to use local-only strategy (no global gather)
        top_k: Number of top-k experts (needed for expanded strategies)

    Returns:
        TokenDropStrategy instance
    """
    global _GLOBAL_TOKEN_DROP_STRATEGY
    global _GLOBAL_TOKEN_DROP_STRATEGY_CONFIG

    effective_strategy_name = "expert_drop_local" if local_only else strategy_name
    kwargs_signature = tuple(sorted((k, repr(v)) for k, v in kwargs.items()))
    requested_config = (
        effective_strategy_name,
        float(load_factor),
        bool(local_only),
        int(top_k),
        kwargs_signature,
    )

    if _GLOBAL_TOKEN_DROP_STRATEGY is not None:
        if _GLOBAL_TOKEN_DROP_STRATEGY_CONFIG != requested_config:
            raise ValueError(
                "TokenDropStrategy singleton already initialized with a different configuration. "
                f"existing={_GLOBAL_TOKEN_DROP_STRATEGY_CONFIG}, "
                f"requested={requested_config}")
        logger.info(
            "[TokenDrop] Reusing global strategy instance: %s",
            _GLOBAL_TOKEN_DROP_STRATEGY.__class__.__name__,
        )
        return _GLOBAL_TOKEN_DROP_STRATEGY

    if local_only:
        strategy = ExpertDropLocalStrategy(load_factor=load_factor, **kwargs)
        logger.info(
            "[TokenDrop] Creating global ExpertDropLocalStrategy with load_factor=%s",
            load_factor,
        )
        _GLOBAL_TOKEN_DROP_STRATEGY = strategy
        _GLOBAL_TOKEN_DROP_STRATEGY_CONFIG = requested_config
        return strategy

    strategies = {
        "expert_drop": ExpertDropStrategy,
        "device_drop": DeviceDropStrategy,
        "expert_expanded_drop": ExpertExpandedDropStrategy,
        "device_expanded_drop": DeviceExpandedDropStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown token drop strategy: {strategy_name}. "
            f"Available strategies: {list(strategies.keys())}")

    strategy_cls = strategies[strategy_name]
    # Expanded strategies need top_k
    if strategy_name in ("expert_expanded_drop", "device_expanded_drop"):
        strategy = strategy_cls(load_factor=load_factor, top_k=top_k, **kwargs)
    else:
        strategy = strategy_cls(load_factor=load_factor, **kwargs)

    logger.info(
        "[TokenDrop] Creating global %s with load_factor=%s, is_expanded=%s",
        strategy_cls.__name__,
        load_factor,
        strategy.is_expanded,
    )
    _GLOBAL_TOKEN_DROP_STRATEGY = strategy
    _GLOBAL_TOKEN_DROP_STRATEGY_CONFIG = requested_config
    return strategy
