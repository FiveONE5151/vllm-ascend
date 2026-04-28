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
"""Utility functions for token drop strategies."""

import csv
import os
from dataclasses import dataclass
from typing import Optional

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import logger


def log_token_drop_statistics(
    ep_rank: int,
    ep_size: int,
    num_global_tokens_per_expert_before_drop: torch.Tensor,
    num_global_tokens_per_expert_after_drop: torch.Tensor,
    expert_capacity: int,
    device_capacity: int,
    step: int,
    token_drop_csv_dir: str,
    num_experts: int,
    num_local_experts: int,
) -> None:
    """Log token drop statistics to logger and optionally CSV.

    Args:
        ep_rank: Current EP rank
        ep_size: Total EP world size
        num_global_tokens_per_expert_before_drop: Token counts before drop [ep_size, num_experts]
        num_global_tokens_per_expert_after_drop: Token counts after drop [ep_size, num_experts]
        expert_capacity: Capacity per expert
        device_capacity: Capacity per device
        step: Current step number
        token_drop_csv_dir: Directory to write CSV files
        num_experts: Total number of experts
        num_local_experts: Number of local experts per rank
    """
    if ep_rank != 0:
        return

    ctx = get_forward_context()
    if ctx.in_profile_run or ctx.capturing or ctx.is_graph_warmup:
        return

    expert_load_before = num_global_tokens_per_expert_before_drop.sum(
        dim=0).to(torch.int64)
    expert_load_after = num_global_tokens_per_expert_after_drop.sum(
        dim=0).to(torch.int64)

    rank_load_before = num_global_tokens_per_expert_before_drop.reshape(
        ep_size, ep_size, num_local_experts).sum(dim=(0, 2)).to(torch.int64)
    rank_load_after = num_global_tokens_per_expert_after_drop.reshape(
        ep_size, ep_size, num_local_experts).sum(dim=(0, 2)).to(torch.int64)

    max_expert_before, max_expert_before_idx = torch.max(expert_load_before, dim=0)
    max_expert_after, max_expert_after_idx = torch.max(expert_load_after, dim=0)
    max_rank_before, max_rank_before_idx = torch.max(rank_load_before, dim=0)
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

    if not token_drop_csv_dir:
        logger.warning(
            "[TokenDrop][step=%d] CSV not written because VLLM_TOKEN_DROP_CSV_DIR is not set.",
            step)
        return

    try:
        os.makedirs(token_drop_csv_dir, exist_ok=True)
        csv_path = os.path.join(token_drop_csv_dir, "token_drop_stats_rank0.csv")
        should_write_header = (not os.path.exists(csv_path) or
                               os.path.getsize(csv_path) == 0)

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

            for expert_idx in range(num_experts):
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

        logger.info("[TokenDrop][step=%d] CSV saved to: %s", step, csv_path)
    except Exception as e:
        logger.warning("[TokenDrop][step=%d] Failed to write CSV: %s", step, str(e))


def rank_token_ranges(num_tokens_across_dp: torch.Tensor,
                      ep_size: int) -> list[tuple[int, int]]:
    """Compute token ranges for each rank in the global tensor.

    Args:
        num_tokens_across_dp: Token counts per rank [ep_size]
        ep_size: Total EP world size

    Returns:
        List of (start, end) tuples for each rank
    """
    token_prefix = torch.cumsum(num_tokens_across_dp.to(torch.int64), dim=0)
    token_starts = token_prefix - num_tokens_across_dp.to(torch.int64)
    token_ranges: list[tuple[int, int]] = []
    for rank in range(ep_size):
        start = int(token_starts[rank].item())
        end = int(token_prefix[rank].item())
        token_ranges.append((start, end))
    return token_ranges


def compute_num_global_tokens_per_expert_after_drop(
    global_topk_ids_after_drop: torch.Tensor,
    num_tokens_across_dp: torch.Tensor,
    num_experts: int,
    ep_size: int,
) -> torch.Tensor:
    """Compute token count per global expert after drop.

    Args:
        global_topk_ids_after_drop: Topk IDs after drop [T_global, topk]
        num_tokens_across_dp: Token counts per rank [ep_size]
        num_experts: Total number of experts
        ep_size: Total EP world size

    Returns:
        Token counts per expert per rank [ep_size, num_experts]
    """
    num_global_tokens_per_expert_after_drop = torch.zeros(
        (ep_size, num_experts),
        dtype=torch.int64,
        device=global_topk_ids_after_drop.device,
    )

    token_ranges = rank_token_ranges(num_tokens_across_dp, ep_size)
    for rank, (start, end) in enumerate(token_ranges):
        current_topk_ids_after_drop = global_topk_ids_after_drop[start:end, :]

        flat_expert_ids = current_topk_ids_after_drop.reshape(-1).to(torch.int64)
        valid_mask = (flat_expert_ids < num_experts).to(torch.int64)
        safe_expert_ids = torch.clamp(flat_expert_ids, min=0, max=num_experts - 1)

        per_rank_counts = torch.zeros(
            num_experts,
            dtype=torch.int64,
            device=global_topk_ids_after_drop.device,
        )
        per_rank_counts.scatter_add_(0, safe_expert_ids, valid_mask)
        num_global_tokens_per_expert_after_drop[rank] = per_rank_counts

    return num_global_tokens_per_expert_after_drop


def renormalize_topk_weights(
    topk_weights: torch.Tensor,
    eps: Optional[float] = None,
) -> torch.Tensor:
    """Safe renormalize to avoid NaN when all weights are zero.

    Args:
        topk_weights: Weight tensor [T, topk]
        eps: Epsilon for numerical stability (default: uses dtype eps)

    Returns:
        Normalized weights [T, topk]
    """
    if eps is None:
        eps = torch.finfo(topk_weights.dtype).eps

    row_sum = topk_weights.sum(dim=-1, keepdim=True)
    nonzero_mask = row_sum > eps

    normalized = torch.where(
        nonzero_mask,
        topk_weights / row_sum.clamp_min(eps),
        torch.zeros_like(topk_weights),
    )
    return normalized


@dataclass
class ExpandedDropStatistics:
    """Statistics for expanded drop strategy.

    All weights are raw softmax probabilities (before renormalization).
    """
    # Drop severity
    dropped_weight_ratio: float  # dropped_weight_sum / total_candidate_weight
    max_dropped_weight: float  # max weight among dropped pairs
    mean_dropped_weight: float  # mean weight among dropped pairs

    # Expanded expert value
    expanded_weight_ratio: float  # expanded_kept_weight / original_topk_kept_weight
    duplicate_suppression_count: int  # number of expanded positions suppressed


def compute_expanded_drop_statistics(
    router_probs: torch.Tensor,
    expand_global_topk_ids: torch.Tensor,
    top_mask: torch.Tensor,
    top_k: int,
    duplicate_suppression_count: int,
) -> ExpandedDropStatistics:
    """Compute statistics for expanded drop strategy.

    Args:
        router_probs: Softmax probabilities [T, num_experts]
        expand_global_topk_ids: Expanded candidate expert IDs [T, topk + num_local_experts]
        top_mask: Boolean mask for kept positions [T, expanded_topk]
        top_k: Number of original topk experts
        duplicate_suppression_count: Count of suppressed expanded positions (tracked externally)

    Returns:
        ExpandedDropStatistics with computed metrics
    """
    # Raw softmax probabilities for expanded candidate set (before renormalization)
    raw_weights = router_probs.gather(-1, expand_global_topk_ids)

    # Drop severity metrics
    dropped_mask = ~top_mask
    dropped_weights = raw_weights * dropped_mask.to(raw_weights.dtype)
    dropped_weight_sum = dropped_weights.sum().item()
    total_candidate_weight = raw_weights.sum().item()
    dropped_weight_ratio = dropped_weight_sum / max(total_candidate_weight, 1e-10)

    dropped_weights_flat = dropped_weights[dropped_mask]
    if dropped_weights_flat.numel() > 0:
        max_dropped_weight = dropped_weights_flat.max().item()
        mean_dropped_weight = dropped_weights_flat.mean().item()
    else:
        max_dropped_weight = 0.0
        mean_dropped_weight = 0.0

    # Expanded expert value metrics
    expanded_topk = expand_global_topk_ids.shape[1]
    expanded_kept_mask = top_mask[:, top_k:expanded_topk]  # expanded positions that are kept
    original_kept_mask = top_mask[:, :top_k]  # original topk positions that are kept

    expanded_kept_weight = (
        raw_weights[:, top_k:expanded_topk] *
        expanded_kept_mask.to(raw_weights.dtype)).sum().item()
    original_kept_weight = (
        raw_weights[:, :top_k] * original_kept_mask.to(raw_weights.dtype)).sum().item()
    expanded_weight_ratio = expanded_kept_weight / max(original_kept_weight, 1e-10)

    return ExpandedDropStatistics(
        dropped_weight_ratio=dropped_weight_ratio,
        max_dropped_weight=max_dropped_weight,
        mean_dropped_weight=mean_dropped_weight,
        expanded_weight_ratio=expanded_weight_ratio,
        duplicate_suppression_count=duplicate_suppression_count,
    )


def log_expanded_drop_statistics(
    ep_rank: int,
    step: int,
    stats: ExpandedDropStatistics,
    token_drop_csv_dir: str,
) -> None:
    """Log expanded drop statistics to logger and optionally CSV.

    Args:
        ep_rank: Current EP rank
        step: Current step number
        stats: Computed expanded drop statistics
        token_drop_csv_dir: Directory to write CSV files
    """
    if ep_rank != 0:
        return

    ctx = get_forward_context()
    if ctx.in_profile_run or ctx.capturing or ctx.is_graph_warmup:
        return

    # Logger output
    logger.info(
        "[ExpandedDrop][step=%d] Drop: ratio=%.2f%%, max=%.4f, mean=%.4f",
        step,
        stats.dropped_weight_ratio * 100,
        stats.max_dropped_weight,
        stats.mean_dropped_weight,
    )
    logger.info(
        "[ExpandedDrop][step=%d] Expanded: weight_ratio=%.2f%%, duplicate_suppressed=%d",
        step,
        stats.expanded_weight_ratio * 100,
        stats.duplicate_suppression_count,
    )

    # CSV output
    if not token_drop_csv_dir:
        return

    try:
        os.makedirs(token_drop_csv_dir, exist_ok=True)
        csv_path = os.path.join(token_drop_csv_dir, "expanded_drop_stats.csv")
        should_write_header = (not os.path.exists(csv_path) or
                               os.path.getsize(csv_path) == 0)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if should_write_header:
                writer.writerow([
                    "step",
                    "dropped_weight_ratio",
                    "max_dropped_weight",
                    "mean_dropped_weight",
                    "expanded_weight_ratio",
                    "duplicate_suppression_count",
                ])
            writer.writerow([
                step,
                f"{stats.dropped_weight_ratio:.6f}",
                f"{stats.max_dropped_weight:.6f}",
                f"{stats.mean_dropped_weight:.6f}",
                f"{stats.expanded_weight_ratio:.6f}",
                stats.duplicate_suppression_count,
            ])

        logger.info("[ExpandedDrop][step=%d] CSV saved to: %s", step, csv_path)
    except Exception as e:
        logger.warning("[ExpandedDrop][step=%d] Failed to write CSV: %s", step,
                       str(e))
