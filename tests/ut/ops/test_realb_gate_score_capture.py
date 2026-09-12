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
import json
import os

import numpy as np
import pytest
import torch

from vllm_ascend import realb_gate_score_capture as capture

TOP_K = 6
SCALING_FACTOR = 2.446


class FakeMoELayer:
    """Minimal stand-in for AscendFusedMoE layer identity."""

    def __init__(self, layer_id: int | None = None, prefix: str = ""):
        if layer_id is not None:
            self.layer_id = layer_id
        self.prefix = prefix


@pytest.fixture(autouse=True)
def capture_run(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ASCEND_REALB_ROUTING_DUMP", "1")
    monkeypatch.setenv("VLLM_ASCEND_REALB_ROUTING_DUMP_DIR", str(tmp_path))
    capture.reset_state()
    yield tmp_path
    capture.reset_state()


def begin_step(ids, num_scheduled, num_computed, attn_state="PrefillNoCache", req_ids=None):
    capture.begin_step(
        req_ids=list(req_ids)
        if req_ids is not None
        else [f"chatcmpl-mmmu_val:validation:{i}" for i in range(len(num_scheduled))],
        num_scheduled_tokens=np.asarray(num_scheduled, dtype=np.int64),
        num_computed_tokens=np.asarray(num_computed, dtype=np.int64),
        attn_state=attn_state,
        input_ids=np.asarray(ids, dtype=np.int32),
    )


def make_topk(rows, layer_offset=0, dtype=torch.float32):
    topk_ids = torch.arange(rows * TOP_K, dtype=torch.int32).reshape(rows, TOP_K) + layer_offset * 100
    topk_weights = torch.full((rows, TOP_K), SCALING_FACTOR / TOP_K, dtype=dtype)
    return topk_ids, topk_weights


def test_disabled_capture_does_not_touch_the_disk(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ASCEND_REALB_ROUTING_DUMP", "0")
    capture.reset_state()

    begin_step([1, 2, 3], [3], [0])
    topk_ids, topk_weights = make_topk(3)
    capture.record_select_experts(layer=FakeMoELayer(layer_id=1), topk_ids=topk_ids, topk_weights=topk_weights)
    capture.finalize_step()

    assert not os.path.exists(os.path.join(str(tmp_path), "steps"))
    assert not os.path.exists(os.path.join(str(tmp_path), "capture_config.json"))


def test_step_is_written_with_request_layout(tmp_path):
    req_ids = ["chatcmpl-mmmu_val:validation:0", "chatcmpl-mmmu_val:validation:1"]
    begin_step([11, 22, 33, 44, 55], [3, 2], [0, 0], req_ids=req_ids)
    for layer_id in (1, 2):
        topk_ids, topk_weights = make_topk(5, layer_offset=layer_id)
        capture.record_select_experts(
            layer=FakeMoELayer(layer_id=layer_id), topk_ids=topk_ids, topk_weights=topk_weights
        )
    capture.finalize_step()

    step_path = os.path.join(str(tmp_path), "steps", "tp0_000001.npz")
    with np.load(step_path) as data:
        assert data["input_ids"].dtype == np.int32
        assert data["topk_ids"].dtype == np.int16
        assert data["topk_weights"].dtype == np.float32
        assert data["topk_ids"].shape == (5, 2, TOP_K)
        assert data["topk_weights"].shape == (5, 2, TOP_K)
        assert data["layer_ids"].tolist() == [1, 2]
        assert data["req_index"].tolist() == [0, 0, 0, 1, 1]
        assert data["input_ids"].tolist() == [11, 22, 33, 44, 55]
        assert int(data["shard_start"][0]) == 0
        assert int(data["shard_size"][0]) == 5
        assert data["topk_ids"][0, 0, 0] == 100
        assert data["topk_ids"][0, 1, 0] == 200

    with open(os.path.join(str(tmp_path), "steps_tp0.jsonl"), encoding="utf-8") as handle:
        step_meta = json.loads(handle.readline())
    assert step_meta["step_id"] == 1
    assert step_meta["num_layers"] == 2
    assert step_meta["num_actual_tokens"] == 5
    assert step_meta["tracked_req_ids"] == req_ids
    assert step_meta["shard_rule"] == "whole"
    assert step_meta["tp_size"] == 1
    assert step_meta["requests"][0] == {"req_id": req_ids[0], "start": 0, "num_scheduled_tokens": 3}
    assert step_meta["requests"][1]["num_scheduled_tokens"] == 2

    with open(os.path.join(str(tmp_path), "coverage_tp0.jsonl"), encoding="utf-8") as handle:
        coverage = json.loads(handle.readline())
    assert coverage["num_layers"] == 2
    assert coverage["num_tracked_requests"] == 2


def test_duplicate_layer_record_keeps_the_first_write(tmp_path):
    begin_step([1, 2], [2], [0])
    first_ids, first_weights = make_topk(2)
    capture.record_select_experts(layer=FakeMoELayer(layer_id=3), topk_ids=first_ids, topk_weights=first_weights)
    capture.record_select_experts(
        layer=FakeMoELayer(layer_id=3),
        topk_ids=torch.full((2, TOP_K), 999, dtype=torch.int32),
        topk_weights=torch.zeros((2, TOP_K), dtype=torch.float32),
    )
    capture.finalize_step()

    with np.load(os.path.join(str(tmp_path), "steps", "tp0_000001.npz")) as data:
        assert data["topk_ids"][0, 0, 0] == 0
        assert data["layer_ids"].tolist() == [3]


@pytest.mark.parametrize("attn_state", ["DecodeOnly", "SpecDecoding"])
def test_non_prefill_steps_are_skipped(attn_state, tmp_path):
    begin_step([7], [1], [128], attn_state=attn_state)
    topk_ids, topk_weights = make_topk(1)
    capture.record_select_experts(layer=FakeMoELayer(layer_id=1), topk_ids=topk_ids, topk_weights=topk_weights)
    capture.finalize_step()

    assert not os.path.exists(os.path.join(str(tmp_path), "steps"))


def test_steps_without_serving_requests_are_skipped(tmp_path):
    begin_step([7, 8], [2], [0], req_ids=["0", "1"])
    capture.finalize_step()

    assert not os.path.exists(os.path.join(str(tmp_path), "steps"))


def test_missing_layers_are_reported_without_breaking_the_engine(tmp_path):
    begin_step([1, 2], [2], [0])
    capture.finalize_step()

    begin_step([3, 4], [2], [2])
    topk_ids, topk_weights = make_topk(2)
    capture.record_select_experts(layer=FakeMoELayer(layer_id=1), topk_ids=topk_ids, topk_weights=topk_weights)
    capture.finalize_step()

    begin_step([5, 6], [2], [4])
    capture.record_select_experts(layer=FakeMoELayer(layer_id=2), topk_ids=topk_ids, topk_weights=topk_weights)
    capture.finalize_step()

    with open(os.path.join(str(tmp_path), "anomalies.jsonl"), encoding="utf-8") as handle:
        anomalies = [json.loads(line) for line in handle]
    kinds = [entry["kind"] for entry in anomalies]
    assert kinds == ["no_layers_recorded", "layer_set_changed"]
    assert anomalies[1]["detail"] == {"previous": [1], "current": [2]}
    assert os.path.exists(os.path.join(str(tmp_path), "steps", "tp0_000003.npz"))


def test_weights_are_stored_verbatim(tmp_path):
    begin_step([1, 2], [2], [0])
    weights = torch.tensor(
        [[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float32
    )
    capture.record_select_experts(
        layer=FakeMoELayer(layer_id=1), topk_ids=torch.zeros((2, TOP_K), dtype=torch.int32), topk_weights=weights
    )
    capture.finalize_step()

    with np.load(os.path.join(str(tmp_path), "steps", "tp0_000001.npz")) as data:
        assert np.array_equal(data["topk_weights"][:, 0, :], weights.numpy())
        assert abs(float(data["topk_weights"][1, 0, :].sum()) - 2.1) < 1e-6


def test_layer_id_resolution(tmp_path):
    assert capture.resolve_layer_id(FakeMoELayer(layer_id=7)) == 7
    assert capture.resolve_layer_id(FakeMoELayer(prefix="model.layers.12.mlp.experts")) == 12
    with pytest.raises(RuntimeError):
        capture.resolve_layer_id(FakeMoELayer(prefix="model.mlp.experts"))


def test_capture_config_is_written_once(tmp_path):
    begin_step([1], [1], [0])
    topk_ids, topk_weights = make_topk(1)
    capture.record_select_experts(layer=FakeMoELayer(layer_id=1), topk_ids=topk_ids, topk_weights=topk_weights)
    capture.finalize_step()

    with open(os.path.join(str(tmp_path), "capture_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["topk_weights_dtype"] == "float32"
    assert config["topk_ids_dtype"] == "int16"
    assert config["request_id_prefix"] == "chatcmpl-"


def test_native_marker_forces_the_reference_gating_path(tmp_path):
    begin_step([1, 2], [2], [0], req_ids=["chatcmpl-native:mmmu_val:validation:0"])
    assert capture.force_native_gating_for_current_step() is True
    topk_ids, topk_weights = make_topk(2)
    capture.record_select_experts(layer=FakeMoELayer(layer_id=1), topk_ids=topk_ids, topk_weights=topk_weights)
    capture.finalize_step()

    with open(os.path.join(str(tmp_path), "steps_tp0.jsonl"), encoding="utf-8") as handle:
        assert json.loads(handle.readline())["force_native_gating"] is True

    begin_step([3, 4], [2], [2], req_ids=["chatcmpl-realb:mmmu_val:validation:0"])
    assert capture.force_native_gating_for_current_step() is False
    capture.record_select_experts(layer=FakeMoELayer(layer_id=1), topk_ids=topk_ids, topk_weights=topk_weights)
    capture.finalize_step()

    with open(os.path.join(str(tmp_path), "steps_tp0.jsonl"), encoding="utf-8") as handle:
        assert json.loads(handle.readlines()[1])["force_native_gating"] is False


def test_shard_layout_candidates():
    assert capture._shard_candidates(125, 1, 0) == [("whole", 0, 125)]
    rank0 = capture._shard_candidates(125, 4, 0)
    assert ("even_split", 0, 32) in rank0
    assert ("floor_split", 0, 31) in rank0
    rank3 = capture._shard_candidates(125, 4, 3)
    assert ("even_split", 94, 31) in rank3
    assert ("floor_split", 93, 32) in rank3
    assert ("tensor_split", 96, 29) in rank3


def test_content_verified_shard_is_sliced(tmp_path, monkeypatch):
    """The recorded shard must be the slice of the step that this rank handled."""

    ids = np.arange(10, dtype=np.int32)
    monkeypatch.setattr(capture, "_tp_geometry", lambda: (1, 4))
    monkeypatch.setattr(capture, "_shard_token_ids", lambda context: ids[3:6].copy())
    capture.begin_step(
        req_ids=["chatcmpl-realb:mmmu_val:validation:0"],
        num_scheduled_tokens=np.asarray([10], dtype=np.int64),
        num_computed_tokens=np.asarray([0], dtype=np.int64),
        attn_state="PrefillNoCache",
        input_ids=ids,
    )
    topk_ids = torch.arange(3 * TOP_K, dtype=torch.int32).reshape(3, TOP_K)
    topk_weights = torch.full((3, TOP_K), SCALING_FACTOR / TOP_K, dtype=torch.float32)
    capture.record_select_experts(layer=FakeMoELayer(layer_id=1), topk_ids=topk_ids, topk_weights=topk_weights)
    capture.finalize_step()

    with np.load(os.path.join(str(tmp_path), "steps", "tp1_000001.npz")) as data:
        assert data["input_ids"].tolist() == [3, 4, 5]
        assert int(data["shard_start"][0]) == 3
        assert int(data["shard_size"][0]) == 3
        assert data["topk_ids"].shape == (3, 1, TOP_K)

    with open(os.path.join(str(tmp_path), "geometry.jsonl"), encoding="utf-8") as handle:
        geometry = json.loads(handle.readline())
    assert geometry["tp_rank"] == 1
    assert geometry["shard_size"] == 3
