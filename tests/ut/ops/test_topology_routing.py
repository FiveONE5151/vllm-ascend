import builtins
import types
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.envs as envs
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import topology_routing as tar


@pytest.fixture(autouse=True)
def tar_env(monkeypatch, tmp_path):
    monkeypatch.setitem(envs.env_variables, "VLLM_ENABLE_TOPOLOGY_AWARE_ROUTING", lambda: True)
    monkeypatch.setitem(envs.env_variables, "VLLM_TOPOLOGY_AWARE_ROUTING_STRATEGY", lambda: "min_cost")
    monkeypatch.setitem(envs.env_variables, "VLLM_TOPOLOGY_AWARE_ROUTING_LOGGING", lambda: False)
    monkeypatch.setitem(envs.env_variables, "VLLM_TOPOLOGY_AWARE_ROUTING_LOG_DIR", lambda: str(tmp_path))
    monkeypatch.setitem(envs.env_variables, "VLLM_TOPOLOGY_AWARE_ROUTING_CONFIG", lambda: "")
    monkeypatch.setitem(envs.env_variables, "VLLM_TOPOLOGY_AWARE_ROUTING_TOKEN_CONFIG", lambda: "token.json")
    monkeypatch.setitem(envs.env_variables, "VLLM_TOPOLOGY_AWARE_ROUTING_SOLVER_CONFIG_JSON", lambda: "")
    yield


def _state(max_tokens: int = 4) -> tar.TopologyRoutingState:
    token_source_ranks = torch.zeros(max_tokens, dtype=torch.long)
    layout = tar.PreparedTokenLayout(
        max_local_tokens=max_tokens,
        max_global_tokens=max_tokens,
        token_source_ranks=token_source_ranks,
        local_start=0,
        local_end=max_tokens,
        row_ids=torch.arange(max_tokens, dtype=torch.long),
    )
    return tar.TopologyRoutingState(
        ep_rank=0,
        runtime_ep_size=1,
        virtual_ep_size=1,
        top_k=2,
        runtime_num_local_experts=4,
        virtual_num_local_experts=4,
        global_num_experts=4,
        route_method="min_cost",
        solver_config={},
        token_topology_config=types.SimpleNamespace(token_source_policy="uniform_ep_rank"),
        rank_to_node=None,
        expert_physical_ranks=torch.zeros(4, dtype=torch.long),
        prepared_layouts={max_tokens: layout},
    )


def _ctx(*, graph_mode: bool = False):
    return types.SimpleNamespace(
        cudagraph_runtime_mode=types.SimpleNamespace(name="FULL" if graph_mode else "NONE"),
        capturing=graph_mode,
        tar_valid_token_mask=torch.tensor([True, True, False, False]),
        moe_comm_type=MoECommType.MC2,
        uniform_decode=True,
    )


def test_missing_topology_aware_routing_dependency_does_not_patch_sys_path(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "topology_aware_routing":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ModuleNotFoundError):
        tar._load_route_function()


def test_logging_disabled_does_not_compute_baseline(monkeypatch):
    router_logits = torch.randn(4, 4)
    routed_weights = torch.ones(4, 2, dtype=torch.float32)
    routed_ids = torch.tensor([[0, 1]] * 4, dtype=torch.int64)
    route = Mock(return_value=(routed_weights, routed_ids))
    monkeypatch.setattr(tar, "_load_route_function", lambda: route)
    monkeypatch.setattr(tar, "get_forward_context", lambda: _ctx())
    baseline = Mock()
    monkeypatch.setattr(tar, "_compute_local_baseline_topk", baseline)

    topk_weights, topk_ids = tar.apply_topology_aware_routing(
        hidden_states=torch.randn(4, 8),
        router_logits=router_logits,
        topk_weights=None,
        topk_ids=None,
        top_k=2,
        use_grouped_topk=False,
        scoring_func="softmax",
        renormalize=True,
        global_num_experts=4,
        num_local_experts=4,
        ep_rank=0,
        ep_size=1,
        ep_group=None,
        topology_routing_state=_state(),
    )

    baseline.assert_not_called()
    assert topk_ids.dtype == torch.int32
    assert topk_weights.shape == (4, 2)


def test_graph_mode_skips_online_routing_log(monkeypatch):
    monkeypatch.setitem(envs.env_variables, "VLLM_TOPOLOGY_AWARE_ROUTING_LOGGING", lambda: True)
    router_logits = torch.randn(4, 4)
    route = Mock(return_value=(torch.ones(4, 2), torch.tensor([[0, 1]] * 4)))
    monkeypatch.setattr(tar, "_load_route_function", lambda: route)
    monkeypatch.setattr(tar, "get_forward_context", lambda: _ctx(graph_mode=True))
    baseline = Mock()
    save_log = Mock()
    monkeypatch.setattr(tar, "_compute_local_baseline_topk", baseline)
    monkeypatch.setattr(tar, "_save_routing_log", save_log)

    tar.apply_topology_aware_routing(
        hidden_states=torch.randn(4, 8),
        router_logits=router_logits,
        topk_weights=None,
        topk_ids=None,
        top_k=2,
        use_grouped_topk=False,
        scoring_func="softmax",
        renormalize=True,
        global_num_experts=4,
        num_local_experts=4,
        ep_rank=0,
        ep_size=1,
        ep_group=None,
        topology_routing_state=_state(),
    )

    baseline.assert_not_called()
    save_log.assert_not_called()
