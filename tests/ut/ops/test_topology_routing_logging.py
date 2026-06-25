from types import SimpleNamespace

import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import topology_routing as tr
from vllm_ascend.utils import adapt_patch

adapt_patch(True)


class DummyRuntimeMode:
    def __init__(self, name: str):
        self.name = name


def make_state(token_source_ranks: torch.Tensor) -> tr.TopologyRoutingState:
    layout = tr.PreparedTokenLayout(
        max_local_tokens=int(token_source_ranks.numel()),
        max_global_tokens=int(token_source_ranks.numel()),
        token_source_ranks=token_source_ranks,
        local_start=0,
        local_end=int(token_source_ranks.numel()),
        row_ids=torch.arange(int(token_source_ranks.numel()), dtype=torch.long),
    )
    return tr.TopologyRoutingState(
        ep_rank=0,
        runtime_ep_size=2,
        virtual_ep_size=2,
        top_k=2,
        runtime_num_local_experts=2,
        virtual_num_local_experts=2,
        global_num_experts=4,
        route_method="greedy_token",
        solver_config={"alpha": 1},
        token_topology_config=SimpleNamespace(
            token_source_policy="uniform_ep_rank",
            name="uniform",
            remainder_policy="tail",
            order="rank-major",
        ),
        rank_to_node=torch.tensor([0, 1], dtype=torch.long),
        expert_physical_ranks=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        prepared_layouts={int(token_source_ranks.numel()): layout},
    )


def test_apply_topology_aware_routing_logs_online_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tr.envs, "VLLM_TOPOLOGY_AWARE_ROUTING_LOGGING", True)
    monkeypatch.setattr(tr.envs, "VLLM_ENABLE_TOKEN_DROP", False)
    monkeypatch.setattr(tr.envs, "VLLM_TOPOLOGY_AWARE_ROUTING_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(tr.envs, "VLLM_TOPOLOGY_AWARE_ROUTING_CONFIG", "")
    monkeypatch.setattr(tr.envs, "VLLM_TOPOLOGY_AWARE_ROUTING_TOKEN_CONFIG", "token.json")

    ctx = SimpleNamespace(
        uniform_decode=True,
        cudagraph_runtime_mode=DummyRuntimeMode("NONE"),
        my_capturing=False,
        capturing=False,
        tar_valid_token_mask=None,
        moe_comm_type=MoECommType.ALLGATHER,
    )
    monkeypatch.setattr(tr, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(
        tr,
        "_resolve_token_counts",
        lambda router_logits, ep_size, ep_rank, comm_type: torch.tensor([2, 2], dtype=torch.int64),
    )

    baseline_calls = []

    def fake_baseline(**kwargs):
        baseline_calls.append(kwargs["router_logits"].shape)
        return torch.full((4, 2), 0.5, dtype=torch.float32), torch.tensor(
            [[0, 1], [0, 1], [2, 3], [2, 3]], dtype=torch.int32
        )

    monkeypatch.setattr(tr, "_compute_local_baseline_topk", fake_baseline)

    def fake_route(router_logits, top_k, **kwargs):
        routed_ids = torch.tensor(
            [[2, 3], [2, 3], [0, 1], [0, 1]], dtype=torch.long
        )
        routed_weights = torch.full((4, 2), 0.5, dtype=router_logits.dtype)
        return routed_weights, routed_ids

    monkeypatch.setattr(tr, "_load_route_function", lambda: fake_route)

    hidden_states = torch.randn(4, 8, dtype=torch.float32)
    router_logits = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    state = make_state(torch.tensor([0, 0, 1, 1], dtype=torch.long))

    out_weights, out_ids = tr.apply_topology_aware_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_weights=None,
        topk_ids=None,
        top_k=2,
        use_grouped_topk=False,
        scoring_func="softmax",
        renormalize=True,
        global_num_experts=4,
        num_local_experts=2,
        ep_rank=0,
        ep_size=2,
        ep_group="unused",
        moe_instance_id=7,
        layer_name="model.layers.0.mlp",
        topology_routing_state=state,
    )

    assert baseline_calls == [(4, 4)]
    assert tuple(out_weights.shape) == (4, 2)
    assert torch.equal(out_ids, torch.tensor([[2, 3], [2, 3], [0, 1], [0, 1]], dtype=torch.int32))

    capture_files = list((tmp_path / "router_logits").glob("*.pt"))
    before_files = list((tmp_path / "topk_before").glob("*.pt"))
    after_files = list((tmp_path / "topk_after").glob("*.pt"))
    assert len(capture_files) == len(before_files) == len(after_files) == 1

    capture = torch.load(capture_files[0], map_location="cpu")
    before = torch.load(before_files[0], map_location="cpu")
    after = torch.load(after_files[0], map_location="cpu")
    assert torch.equal(capture["num_tokens_across_ranks"], torch.tensor([2, 2], dtype=torch.int32))
    assert torch.equal(capture["actual_token_counts"], torch.tensor([2, 2], dtype=torch.int32))
    assert torch.equal(before["topk_ids"], torch.tensor([[0, 1], [0, 1], [2, 3], [2, 3]], dtype=torch.int32))
    assert torch.equal(after["topk_ids"], out_ids)


def test_apply_topology_aware_routing_skips_logging_in_graph_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(tr.envs, "VLLM_TOPOLOGY_AWARE_ROUTING_LOGGING", True)
    monkeypatch.setattr(tr.envs, "VLLM_ENABLE_TOKEN_DROP", False)
    monkeypatch.setattr(tr.envs, "VLLM_TOPOLOGY_AWARE_ROUTING_LOG_DIR", str(tmp_path))

    ctx = SimpleNamespace(
        uniform_decode=True,
        cudagraph_runtime_mode=DummyRuntimeMode("FULL"),
        my_capturing=False,
        capturing=False,
        tar_valid_token_mask=torch.ones(4, dtype=torch.bool),
        moe_comm_type=MoECommType.ALLGATHER,
    )
    monkeypatch.setattr(tr, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(tr, "_compute_local_baseline_topk", lambda **kwargs: (_ for _ in ()).throw(AssertionError("baseline topk should not run in graph mode")))
    monkeypatch.setattr(
        tr,
        "_load_route_function",
        lambda: (lambda router_logits, top_k, **kwargs: (torch.ones((4, 2)), torch.zeros((4, 2), dtype=torch.long))),
    )

    warn_messages = []
    monkeypatch.setattr(tr, "_warn_skip_routing_log_once", lambda key, message: warn_messages.append((key, message)))

    state = make_state(torch.tensor([0, 0, 1, 1], dtype=torch.long))
    hidden_states = torch.randn(4, 8, dtype=torch.float32)
    router_logits = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    out_weights, out_ids = tr.apply_topology_aware_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_weights=None,
        topk_ids=None,
        top_k=2,
        use_grouped_topk=False,
        scoring_func="softmax",
        renormalize=True,
        global_num_experts=4,
        num_local_experts=2,
        ep_rank=0,
        ep_size=2,
        ep_group="unused",
        moe_instance_id=7,
        layer_name="model.layers.0.mlp",
        topology_routing_state=state,
    )

    assert warn_messages
    assert not (tmp_path / "router_logits").exists()
    assert tuple(out_weights.shape) == (4, 2)
    assert tuple(out_ids.shape) == (4, 2)


def test_apply_topology_aware_routing_skips_baseline_recompute_when_logging_disabled(monkeypatch):
    monkeypatch.setattr(tr.envs, "VLLM_TOPOLOGY_AWARE_ROUTING_LOGGING", False)
    monkeypatch.setattr(tr.envs, "VLLM_ENABLE_TOKEN_DROP", False)

    ctx = SimpleNamespace(
        uniform_decode=True,
        cudagraph_runtime_mode=DummyRuntimeMode("NONE"),
        my_capturing=False,
        capturing=False,
        tar_valid_token_mask=None,
        moe_comm_type=MoECommType.ALLGATHER,
    )
    monkeypatch.setattr(tr, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(tr, "_compute_local_baseline_topk", lambda **kwargs: (_ for _ in ()).throw(AssertionError("baseline topk should not run when logging is disabled")))
    monkeypatch.setattr(tr, "_resolve_token_counts", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("token count resolution should not run when logging is disabled")))
    monkeypatch.setattr(
        tr,
        "_load_route_function",
        lambda: (lambda router_logits, top_k, **kwargs: (torch.ones((4, 2)), torch.zeros((4, 2), dtype=torch.long))),
    )

    state = make_state(torch.tensor([0, 0, 1, 1], dtype=torch.long))
    hidden_states = torch.randn(4, 8, dtype=torch.float32)
    router_logits = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    out_weights, out_ids = tr.apply_topology_aware_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_weights=None,
        topk_ids=None,
        top_k=2,
        use_grouped_topk=False,
        scoring_func="softmax",
        renormalize=True,
        global_num_experts=4,
        num_local_experts=2,
        ep_rank=0,
        ep_size=2,
        ep_group="unused",
        moe_instance_id=7,
        layer_name="model.layers.0.mlp",
        topology_routing_state=state,
    )

    assert tuple(out_weights.shape) == (4, 2)
    assert tuple(out_ids.shape) == (4, 2)
