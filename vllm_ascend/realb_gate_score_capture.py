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
#
"""ReaLB metric-3 instrumentation: dump MoE gate selections on the engine path.

The module records the ``topk_ids`` / ``topk_weights`` pair that
``vllm_ascend.ops.fused_moe.experts_selector.select_experts`` actually returns -
the very weights the combine step uses - together with the per-step request and
token layout, so that the rank level expert score distribution can be
reconstructed offline. No quantisation is injected and no score is recomputed:
only the engine's own return values are stored.

Token geometry matters. On this deployment MoE input is sharded across the TP
ranks before ``select_experts`` runs (each rank routes only its own token slice),
so every rank writes its own shard and the offline analysis concatenates the
shards back onto the step's token axis. The shard layout is derived from the
step token count, checked against the shard's own token ids when the comm method
exposes them, and recorded in ``geometry.jsonl`` / ``steps_tp*.jsonl`` so the
reconstruction is verifiable rather than assumed.

Everything is gated behind ``VLLM_ASCEND_REALB_ROUTING_DUMP``. When the flag is
off (the default) every public helper returns immediately, no device buffer is
allocated and no host/device synchronisation is added.

Layout of one capture run (``VLLM_ASCEND_REALB_ROUTING_DUMP_DIR``)::

    capture_config.json
    steps/tp{r}_{step}.npz   one rank's shard: input_ids, req_index, topk_ids,
                             topk_weights, layer_ids, shard_start, shard_size
    steps_tp{r}.jsonl        per-step request layout, shard geometry, comm method
    coverage_tp{r}.jsonl     per-step layer/token coverage
    geometry.jsonl           shard geometry diagnostics
    anomalies.jsonl          bookkeeping problems (never fatal for the engine)

``topk_ids``/``topk_weights`` have shape ``[shard_rows, num_moe_layers, top_k]``
with the layer axis sorted by transformer layer index and the expert axis in
top-k slot order. Weights are stored as float32 (not float16) because a float16
relative rounding of ~4.9e-4 is comparable to the per-rank weight differences
this study compares.
"""

from __future__ import annotations

import json
import os
import re
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count
from typing import Any

import numpy as np
import torch
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from vllm_ascend import envs

logger = init_logger(__name__)

__all__ = [
    "begin_step",
    "configured_run_dir",
    "finalize_step",
    "force_native_gating_for_current_step",
    "is_enabled",
    "record_select_experts",
    "reset_state",
    "resolve_layer_id",
]

# Requests created by the OpenAI serving frontend are named "chatcmpl-<request_id>";
# startup dummy/profile runs use different ids and must never land in a capture.
REALB_REQUEST_ID_PREFIX = "chatcmpl-"
# A request whose id tag contains this marker forces the reference (native)
# gating path for its step, which is how the smoke run cross-checks the fused
# `moe_gating_top_k` operator against `_native_select_experts`.
NATIVE_GATING_TAG = "native"
# Decode-only batches are replayed from the ACLGraph without running this module;
# they are not part of the prefill statistics anyway.
SKIPPED_ATTN_STATES = frozenset({"DecodeOnly", "SpecDecoding"})
LAYER_INDEX_PATTERN = re.compile(r"layers\.(\d+)")
STEP_FILE_TEMPLATE = "steps/tp{tp_rank}_{step_id:06d}.npz"


@dataclass
class _LayerBuffer:
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class _StepContext:
    step_id: int
    attn_state: str
    force_native_gating: bool
    num_actual_tokens: int
    req_ids: list[str]
    token_start: list[int]
    num_scheduled: list[int]
    req_index: np.ndarray
    input_ids: np.ndarray
    tracked_req_ids: list[str]
    tp_rank: int = 0
    tp_size: int = 1
    shard_start: int | None = None
    shard_size: int = 0
    shard_rule: str = ""
    recorded_layers: set[int] = field(default_factory=set)


_step_context: ContextVar[_StepContext | None] = ContextVar("realb_gate_score_capture_step", default=None)
_layer_buffers: dict[int, _LayerBuffer] = {}
_step_ids = count(1)
_last_layer_ids: list[int] | None = None
_write_lock = threading.Lock()
_initialised_run_dir: str | None = None
_geometry_logged: set[int] = set()


def is_enabled() -> bool:
    """Whether the routing dump is switched on for this process."""

    return bool(envs.VLLM_ASCEND_REALB_ROUTING_DUMP)


def configured_run_dir() -> str:
    """Output directory of the capture run; fails loudly when unset."""

    run_dir = envs.VLLM_ASCEND_REALB_ROUTING_DUMP_DIR
    if not run_dir:
        raise RuntimeError(
            "VLLM_ASCEND_REALB_ROUTING_DUMP is enabled but VLLM_ASCEND_REALB_ROUTING_DUMP_DIR is not set"
        )
    return str(run_dir)


def reset_state() -> None:
    """Drop all process state. Used by tests and by a full service restart."""

    global _last_layer_ids, _initialised_run_dir, _step_ids
    _step_context.set(None)
    _layer_buffers.clear()
    _geometry_logged.clear()
    _last_layer_ids = None
    _initialised_run_dir = None
    _step_ids = count(1)


def resolve_layer_id(layer: Any) -> int:
    """Transformer layer index of a MoE layer module."""

    layer_id = getattr(layer, "layer_id", None)
    if layer_id is not None:
        return int(layer_id)
    layer_name = getattr(layer, "layer_name", None) or getattr(layer, "prefix", "") or ""
    match = LAYER_INDEX_PATTERN.search(str(layer_name))
    if match is None:
        raise RuntimeError(f"ReaLB gate-score capture cannot resolve layer id from {layer_name!r}")
    return int(match.group(1))


def force_native_gating_for_current_step() -> bool:
    """Whether the step being executed asked for the reference gating path."""

    if not is_enabled():
        return False
    context = _step_context.get()
    return bool(context is not None and context.force_native_gating)


def begin_step(
    *,
    req_ids: list[str],
    num_scheduled_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    attn_state: Any,
    input_ids: Any,
) -> None:
    """Register one model-execution step; call from the model runner before the forward.

    ``num_scheduled_tokens`` and ``num_computed_tokens`` are the per-request values
    in batch order (only the first ``len(req_ids)`` entries are used) and
    ``input_ids`` are the step's real (unpadded) token ids in batch order.
    """

    if not is_enabled():
        _step_context.set(None)
        return

    state_name = _attn_state_name(attn_state)
    if state_name in SKIPPED_ATTN_STATES:
        _step_context.set(None)
        return

    scheduled = np.asarray(num_scheduled_tokens, dtype=np.int64).reshape(-1)[: len(req_ids)]
    if len(req_ids) == 0 or scheduled.size != len(req_ids):
        _step_context.set(None)
        return
    tracked = [str(req_id) for req_id in req_ids if str(req_id).startswith(REALB_REQUEST_ID_PREFIX)]
    if not tracked:
        _step_context.set(None)
        return
    force_native_gating = any(_request_tag(req_id) == NATIVE_GATING_TAG for req_id in tracked)

    num_actual_tokens = int(scheduled.sum())
    if num_actual_tokens <= 0:
        _step_context.set(None)
        return
    ids = np.array(input_ids, dtype=np.int32, copy=True).reshape(-1)
    if ids.size < num_actual_tokens:
        _write_anomaly(
            None,
            "input_ids_short",
            {"expected": num_actual_tokens, "available": int(ids.size)},
        )
        _step_context.set(None)
        return

    computed = np.asarray(num_computed_tokens, dtype=np.int64).reshape(-1)[: len(req_ids)]
    if computed.size != len(req_ids):
        computed = np.zeros(len(req_ids), dtype=np.int64)
    req_index = np.repeat(np.arange(len(req_ids), dtype=np.int32), scheduled)
    tp_rank, tp_size = _tp_geometry()
    _step_context.set(
        _StepContext(
            step_id=next(_step_ids),
            attn_state=state_name,
            force_native_gating=force_native_gating,
            num_actual_tokens=num_actual_tokens,
            req_ids=[str(req_id) for req_id in req_ids],
            token_start=[int(value) for value in computed],
            num_scheduled=[int(value) for value in scheduled],
            req_index=req_index,
            input_ids=ids[:num_actual_tokens],
            tracked_req_ids=tracked,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
    )


def record_select_experts(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer: Any = None,
    layer_id: int | None = None,
) -> None:
    """Copy one layer's ``select_experts`` result into the step's device buffers.

    Only the rows of this rank's token shard belong to real tokens; the tail of a
    padded fused-MoE batch is dropped. Repeated records for the same step and
    layer keep the first write, which makes the call idempotent for the
    multistream gate path.
    """

    context = _step_context.get()
    if context is None:
        return
    resolved_layer_id = int(layer_id) if layer_id is not None else resolve_layer_id(layer)
    if resolved_layer_id in context.recorded_layers:
        return
    if context.shard_start is None and not _resolve_shard(context, int(topk_ids.shape[0])):
        return
    shard_rows = context.shard_size
    if topk_ids.shape[0] < shard_rows or topk_weights.shape[0] < shard_rows:
        _write_anomaly(
            context,
            "select_experts_rows_short",
            {"layer_id": resolved_layer_id, "shard_rows": shard_rows, "topk_rows": int(topk_ids.shape[0])},
        )
        return
    buffer = _buffer_for(resolved_layer_id, shard_rows, int(topk_ids.shape[1]), topk_ids.device)
    buffer.topk_ids[:shard_rows].copy_(topk_ids[:shard_rows].to(torch.int16))
    buffer.topk_weights[:shard_rows].copy_(topk_weights[:shard_rows].to(torch.float32))
    context.recorded_layers.add(resolved_layer_id)
    if context.step_id not in _geometry_logged:
        _geometry_logged.add(context.step_id)
        _append_jsonl(
            os.path.join(_ensure_run_dir(), "geometry.jsonl"),
            {
                "step_id": context.step_id,
                "tp_rank": context.tp_rank,
                "tp_size": context.tp_size,
                "moe_comm_type": _comm_method_name(),
                "num_actual_tokens": context.num_actual_tokens,
                "shard_start": context.shard_start,
                "shard_size": shard_rows,
                "shard_rule": context.shard_rule,
                "layer_id": resolved_layer_id,
                "created_at": _utc_now(),
                "pid": os.getpid(),
            },
        )


def finalize_step() -> None:
    """Flush the current step's shard to ``steps/tp<rank>_<step>.npz``."""

    global _last_layer_ids

    context = _step_context.get()
    if context is None:
        return
    _step_context.set(None)

    if not context.recorded_layers:
        _write_anomaly(context, "no_layers_recorded", {"req_ids": context.req_ids})
        return
    if context.shard_start is None:
        _write_anomaly(context, "shard_unresolved", {"req_ids": context.req_ids})
        return

    layer_ids = sorted(context.recorded_layers)
    start, size = context.shard_start, context.shard_size
    topk_ids = (
        torch.stack([_layer_buffers[layer_id].topk_ids[:size].to("cpu", copy=True) for layer_id in layer_ids], dim=1)
        .contiguous()
        .numpy()
    )
    topk_weights = (
        torch.stack(
            [_layer_buffers[layer_id].topk_weights[:size].to("cpu", copy=True) for layer_id in layer_ids], dim=1
        )
        .contiguous()
        .numpy()
    )
    if _last_layer_ids is not None and layer_ids != _last_layer_ids:
        _write_anomaly(context, "layer_set_changed", {"previous": _last_layer_ids, "current": layer_ids})
    _last_layer_ids = layer_ids

    run_dir = _ensure_run_dir()
    relative_path = STEP_FILE_TEMPLATE.format(tp_rank=context.tp_rank, step_id=context.step_id)
    created_at = _utc_now()
    with _write_lock:
        np.savez(
            os.path.join(run_dir, relative_path),
            input_ids=context.input_ids[start : start + size],
            req_index=context.req_index[start : start + size],
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            layer_ids=np.asarray(layer_ids, dtype=np.int32),
            shard_start=np.asarray([start], dtype=np.int64),
            shard_size=np.asarray([size], dtype=np.int64),
        )
        common = {
            "step_id": context.step_id,
            "created_at": created_at,
            "pid": os.getpid(),
            "tp_rank": context.tp_rank,
            "tp_size": context.tp_size,
            "attn_state": context.attn_state,
            "force_native_gating": context.force_native_gating,
            "moe_comm_type": _comm_method_name(),
            "num_actual_tokens": context.num_actual_tokens,
            "shard_start": start,
            "shard_size": size,
            "shard_rule": context.shard_rule,
            "num_layers": len(layer_ids),
            "layer_ids": layer_ids,
            "npz": relative_path,
        }
        _append_jsonl(
            os.path.join(run_dir, f"steps_tp{context.tp_rank}.jsonl"),
            {
                **common,
                "tracked_req_ids": context.tracked_req_ids,
                "requests": [
                    {"req_id": req_id, "start": token_start, "num_scheduled_tokens": num_scheduled}
                    for req_id, token_start, num_scheduled in zip(
                        context.req_ids, context.token_start, context.num_scheduled, strict=True
                    )
                ],
            },
        )
        _append_jsonl(
            os.path.join(run_dir, f"coverage_tp{context.tp_rank}.jsonl"),
            {**common, "num_tracked_requests": len(context.tracked_req_ids)},
        )


def _resolve_shard(context: _StepContext, observed_rows: int) -> bool:
    """Locate this rank's token shard inside the step's token axis."""

    rows = context.num_actual_tokens
    shard_ids = _shard_token_ids(context)
    candidates = _shard_candidates(rows, context.tp_size, context.tp_rank)
    ordered = [candidate for candidate in candidates if candidate[2] == observed_rows]
    ordered += [candidate for candidate in candidates if candidate[2] != observed_rows]

    if shard_ids is not None and shard_ids.size == observed_rows:
        matched_rule = _find_subsequence(context.input_ids, shard_ids)
        if matched_rule is not None:
            for rule, start, size in ordered:
                if rule == matched_rule and size == observed_rows:
                    _accept_shard(context, start, size, matched_rule)
                    return True
            start, size = _search_start(context.input_ids, shard_ids)
            if start is not None:
                _accept_shard(context, start, size, "content")
                return True
    for rule, start, size in ordered:
        if size != observed_rows:
            continue
        if shard_ids is not None and shard_ids.size == observed_rows:
            if not np.array_equal(context.input_ids[start : start + size], shard_ids):
                continue
        _accept_shard(context, start, size, rule)
        return True
    _write_anomaly(
        context,
        "shard_geometry_unresolved",
        {
            "num_actual_tokens": rows,
            "observed_shard_rows": observed_rows,
            "tp_rank": context.tp_rank,
            "tp_size": context.tp_size,
            "candidates": [(rule, start, size) for rule, start, size in candidates],
            "shard_ids_available": shard_ids is not None,
            "comm": _comm_method_name(),
        },
    )
    return False


def _accept_shard(context: _StepContext, start: int, size: int, rule: str) -> None:
    context.shard_start = int(start)
    context.shard_size = int(size)
    context.shard_rule = str(rule)


def _shard_candidates(rows: int, tp_size: int, tp_rank: int) -> list[tuple[str, int, int]]:
    """Candidate (rule, start, size) layouts for this rank; unverified by design."""

    if tp_size <= 1:
        return [("whole", 0, rows)]
    candidates: list[tuple[str, int, int]] = []
    # ceil-then-tail partition (torch.tensor_split semantics)
    chunk = -(-rows // tp_size)
    for index in range(tp_size):
        start = index * chunk
        size = max(0, min(chunk, rows - start))
        if index == tp_rank:
            candidates.append(("tensor_split", start, size))
    # even-as-possible partition with the remainder spread over the first ranks
    base, remainder = divmod(rows, tp_size)
    offset = 0
    for index in range(tp_size):
        size = base + (1 if index < remainder else 0)
        if index == tp_rank:
            candidates.append(("even_split", offset, size))
        offset += size
    # floor chunks with the remainder in the last chunk (torch.split semantics)
    floor_chunk = rows // tp_size
    if floor_chunk:
        start = tp_rank * floor_chunk
        size = floor_chunk if tp_rank < tp_size - 1 else rows - start
        candidates.append(("floor_split", start, size))
    return candidates


def _shard_token_ids(context: _StepContext) -> np.ndarray | None:
    """Token ids of this rank's shard, when the comm method can produce them."""

    try:
        forward_context = get_forward_context()
        method = getattr(forward_context, "moe_comm_method", None)
        splitter = getattr(method, "pad_and_split_input_ids", None)
        input_ids = getattr(forward_context, "input_ids", None)
        if splitter is None or input_ids is None:
            return None
        shard = np.array(splitter(input_ids), dtype=np.int32, copy=True).reshape(-1)
        return shard
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("ReaLB gate-score capture could not derive shard token ids", exc_info=True)
        return None


def _find_subsequence(haystack: np.ndarray, needle: np.ndarray) -> str | None:
    """Partition rule whose slot reproduces the shard ids verbatim, if any."""

    for rule, start, size in _all_partition_layouts(int(haystack.size)):
        if size != int(needle.size):
            continue
        if np.array_equal(haystack[start : start + size], needle):
            return rule
    return None


def _all_partition_layouts(rows: int) -> list[tuple[str, int, int]]:
    """Partition layouts of every rank for every plausible TP size."""

    layouts: list[tuple[str, int, int]] = []
    for tp_size in (2, 4, 8, 16):
        for tp_rank in range(tp_size):
            layouts.extend(_shard_candidates(rows, tp_size, tp_rank))
    return layouts


def _search_start(haystack: np.ndarray, needle: np.ndarray) -> tuple[int | None, int]:
    """First position where the shard ids appear (bounded scan)."""

    if needle.size == 0 or needle.size > haystack.size:
        return None, 0
    positions = np.flatnonzero(haystack == needle[0])
    for position in positions[:512]:
        end = int(position) + int(needle.size)
        if end <= haystack.size and np.array_equal(haystack[int(position) : end], needle):
            return int(position), int(needle.size)
    return None, 0


def _tp_geometry() -> tuple[int, int]:
    try:
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        return int(group.rank_in_group), int(group.world_size)
    except Exception:  # pragma: no cover - single-process tests
        return 0, 1


def _comm_method_name() -> str:
    try:
        forward_context = get_forward_context()
        return str(getattr(forward_context, "moe_comm_type", "unknown"))
    except Exception:  # pragma: no cover
        return "unknown"


def _buffer_for(layer_id: int, rows: int, top_k: int, device: torch.device) -> _LayerBuffer:
    buffer = _layer_buffers.get(layer_id)
    if (
        buffer is None
        or buffer.topk_ids.shape[0] < rows
        or buffer.topk_ids.shape[1] != top_k
        or buffer.topk_ids.device != device
    ):
        buffer = _LayerBuffer(
            topk_ids=torch.zeros((rows, top_k), dtype=torch.int16, device=device),
            topk_weights=torch.zeros((rows, top_k), dtype=torch.float32, device=device),
        )
        _layer_buffers[layer_id] = buffer
    return buffer


def _ensure_run_dir() -> str:
    global _initialised_run_dir

    run_dir = configured_run_dir()
    if _initialised_run_dir != run_dir:
        os.makedirs(os.path.join(run_dir, "steps"), exist_ok=True)
        with open(os.path.join(run_dir, "capture_config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "module": "vllm_ascend.realb_gate_score_capture",
                    "run_dir": run_dir,
                    "pid": os.getpid(),
                    "started_at": _utc_now(),
                    "topk_ids_dtype": "int16",
                    "topk_weights_dtype": "float32",
                    "skipped_attn_states": sorted(SKIPPED_ATTN_STATES),
                    "request_id_prefix": REALB_REQUEST_ID_PREFIX,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        _initialised_run_dir = run_dir
        logger.info("ReaLB gate-score capture enabled; run_dir=%s pid=%s", run_dir, os.getpid())
    return run_dir


def _write_anomaly(context: _StepContext | None, kind: str, detail: dict[str, Any]) -> None:
    entry: dict[str, Any] = {
        "kind": kind,
        "detail": detail,
        "created_at": _utc_now(),
        "pid": os.getpid(),
    }
    if context is not None:
        entry["step_id"] = context.step_id
        entry["attn_state"] = context.attn_state
        entry["req_ids"] = context.req_ids
    logger.warning("ReaLB gate-score capture anomaly: %s %s", kind, detail)
    try:
        _append_jsonl(os.path.join(_ensure_run_dir(), "anomalies.jsonl"), entry)
    except Exception:  # pragma: no cover - bookkeeping must never break the engine
        logger.exception("ReaLB gate-score capture failed to record anomaly %s", kind)


def _append_jsonl(path: str, entry: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _attn_state_name(attn_state: Any) -> str:
    name = getattr(attn_state, "name", None)
    if isinstance(name, str):
        return name
    return str(attn_state).rsplit(".", 1)[-1]


def _request_tag(req_id: str) -> str:
    """The run tag the client put in front of ``task:split:doc``."""

    return str(req_id)[len(REALB_REQUEST_ID_PREFIX) :].split(":", 1)[0]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
