#
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

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

from tests.ut.base import TestBase

from vllm_ascend.ops.fused_moe.token_dispatcher import (  # isort: skip
    AscendDeviceType, TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAll2AllvTokenDrop, TokenDispatcherWithAll2AllvExpandedDrop,
    TokenDispatcherWithAllGather, TokenDispatcherWithMC2)


class TestTokenDispatcherWithMC2(TestBase):

    def setUp(self):
        self.mc2_group = MagicMock()
        self.mc2_group.device_group.return_value._get_backend.return_value.get_hccl_comm_name.return_value = "hccl_123"
        self.mc2_group.rank_in_group = 0
        self.mc2_group.world_size = 8
        self.mc2_group_patch = patch(
            "vllm_ascend.ops.fused_moe.token_dispatcher.get_mc2_group",
            return_value=self.mc2_group)
        self.mc2_group_patch.start()

        self.rank_group_patch = patch("torch.distributed.get_rank",
                                      return_value=0)
        self.rank_group_patch.start()

        # Mock get_forward_context().mc2_mask
        self.forward_context = MagicMock()
        self.forward_context.mc2_mask = torch.tensor([1, 0, 1])
        self.forward_context_patch = patch(
            "vllm.forward_context.get_forward_context",
            return_value=self.forward_context)
        self.forward_context_patch.start()

        # Mock get_ascend_device_type()
        self.ascend_soc_version_patch = patch(
            "vllm_ascend.ops.fused_moe.token_dispatcher.get_ascend_device_type",
            return_value=AscendDeviceType.A3)
        self.ascend_soc_version_patch.start()

        kwargs = {"with_quant": False, "top_k": 8, "num_experts": 128}
        self.dispatcher = TokenDispatcherWithMC2(**kwargs)

    def tearDown(self):
        self.mc2_group_patch.stop()
        self.forward_context_patch.stop()
        self.ascend_soc_version_patch.stop()

    def test_init(self):
        self.assertEqual(self.dispatcher.ep_rank_id, 0)
        self.assertEqual(self.dispatcher.ep_world_size, 8)
        self.assertFalse(self.dispatcher.with_quant)
        self.assertTrue(self.dispatcher.enable_dispatch_v2)
        self.assertTrue(self.dispatcher.need_extra_args)

    def test_get_dispatch_mc2_kwargs_without_quant(self):
        hidden_states = torch.randn(10, 128)
        topk_ids = torch.randint(0, 8, (10, 1))
        topk_weights = torch.randn(10, 1)
        expert_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        mc2_mask = None

        kwargs = self.dispatcher.get_dispatch_mc2_kwargs(
            hidden_states, topk_weights, topk_ids, expert_map, mc2_mask)
        self.assertIn("x", kwargs)
        self.assertIn("expert_ids", kwargs)
        self.assertEqual(kwargs["moe_expert_num"], 8)

    def test_token_permutation_dispatch(self):
        hidden_states = torch.randn(10, 128)
        topk_weights = torch.randn(10, 1)
        topk_ids = torch.randint(0, 8, (10, 1))
        expert_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        with patch("torch_npu.npu_moe_distribute_dispatch_v2",
                   return_value=(torch.randn(10, 128), ) * 5 +
                   (None, None)) as mock_dispatch:
            output = self.dispatcher.token_dispatch(hidden_states,
                                                    topk_weights, topk_ids,
                                                    expert_map)
            mock_dispatch.assert_called_once()
            self.assertEqual(output.group_list_type, 0)  # group_list_type == 0

    def test_get_combine_mc_kwargs_with_quant(self):
        self.dispatcher.with_quant = True
        hidden_states = torch.randn(10, 128)
        topk_ids = torch.randint(0, 8, (10, 1))
        topk_weights = torch.randn(10, 1)
        expert_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        ep_recv_counts = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        tp_recv_counts = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        mc2_mask = None
        assist_info_for_combine = torch.arange(10)

        context_metadata = {
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "expert_map": expert_map,
            "ep_recv_counts": ep_recv_counts,
            "mc2_mask": mc2_mask,
            "assist_info_for_combine": assist_info_for_combine,
            "expand_scales": None,
            "tp_recv_counts": tp_recv_counts
        }

        self.dispatcher.need_extra_args = True
        self.dispatcher.enable_dispatch_v2 = True
        self.dispatcher.moe_expert_num = len(expert_map)
        kwargs = self.dispatcher.get_combine_mc_kwargs(hidden_states,
                                                       context_metadata)
        self.assertIn("tp_send_counts", kwargs)


class TestTokenDispatcherWithAllGather(TestBase):

    def setUp(self):
        # Mock dependencies
        kwargs = {
            "apply_router_weight_on_input": False,
            "top_k": 2,
            "max_num_tokens": 100,
            "ep_size": 2,
            "num_experts": 128,
            "with_quant": False,
        }
        self.dispatcher = TokenDispatcherWithAllGather(**kwargs)

        # Mock NPU functions
        self.patcher_npu_moe_init_routing_custom = patch(
            'torch.ops._C_ascend.npu_moe_init_routing_custom')
        self.mock_npu_moe_init_routing_custom = self.patcher_npu_moe_init_routing_custom.start(
        )
        self.mock_npu_moe_init_routing_custom.return_value = (
            torch.randn(6, 128),  # sorted_hidden_states
            torch.tensor([0, 1, 2, 3, 4, 5]),  # expanded_row_idx
            torch.tensor([0, 1, 0, 1, 0, 1]),  # expanded_expert_idx
            torch.tensor([0, 1, 0, 1, 0, 1]))
        self.patcher_npu_moe_token_unpermute = patch(
            'torch_npu.npu_moe_token_unpermute')
        self.mock_npu_moe_token_unpermute = self.patcher_npu_moe_token_unpermute.start(
        )
        self.mock_npu_moe_token_unpermute.return_value = torch.randn(6, 128)

    def tearDown(self):
        self.patcher_npu_moe_init_routing_custom.stop()
        self.patcher_npu_moe_token_unpermute.stop()

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_token_dispatch_without_expert_map(self):
        hidden_states = torch.randn(3, 128)
        topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]])

        results = self.dispatcher.token_dispatch(hidden_states, topk_weights,
                                                 topk_ids, None)

        # Verify npu_moe_init_routing is called
        self.mock_npu_moe_init_routing_custom.assert_called_once()
        args, kwargs = self.mock_npu_moe_init_routing_custom.call_args

        self.assertEqual(results.group_list_type, 1)

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_token_dispatch_with_expert_map(self):
        self.dispatcher.expert_map = torch.tensor([0, 1, 2, 3])
        hidden_states = torch.randn(3, 128)
        topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]])

        results = self.dispatcher.token_dispatch(hidden_states, topk_weights,
                                                 topk_ids, None)

        # Verify npu_moe_init_routing is called
        self.mock_npu_moe_init_routing_custom.assert_called_once()
        args, kwargs = self.mock_npu_moe_init_routing_custom.call_args

        self.assertEqual(results.group_list_type, 1)

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_token_dispatch_without_quant(self):
        kwargs = {
            "apply_router_weight_on_input": False,
            "top_k": 2,
            "max_num_tokens": 100,
            "ep_size": 2,
            "num_experts": 128,
        }
        self.dispatcher_quant = TokenDispatcherWithAllGather(**kwargs)

        hidden_states = torch.randn(3, 128)
        topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]])

        results = self.dispatcher_quant.token_dispatch(hidden_states,
                                                       topk_weights, topk_ids,
                                                       None)

        self.assertEqual(results.group_list_type, 1)

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_token_dispatch_with_quant(self):
        kwargs = {
            "apply_router_weight_on_input": False,
            "top_k": 2,
            "max_num_tokens": 100,
            "ep_size": 2,
            "num_experts": 128,
        }
        self.dispatcher_quant = TokenDispatcherWithAllGather(**kwargs)

        hidden_states = torch.randn(3, 128)
        topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]])

        results = self.dispatcher_quant.token_dispatch(hidden_states,
                                                       topk_weights,
                                                       topk_ids,
                                                       None,
                                                       with_quant=True)

        self.assertIsNotNone(results.hidden_states)
        self.assertIsNotNone(results.group_list)
        self.assertIsNotNone(results.dynamic_scale)
        self.assertEqual(results.group_list_type, 1)

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_token_combine_with_expert_map(self):
        hidden_states = torch.randn(6, 128)
        context_metadata = {
            "expanded_row_idx": torch.tensor([0, 1, 1, 1, 1, 1]),
            "topk_weights": torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        }
        self.dispatcher.original_shape = (6, 128)
        final_hidden_states = self.dispatcher.token_combine(
            hidden_states, context_metadata).routed_out
        self.assertEqual(final_hidden_states.shape, (6, 128))

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_token_combine_without_expert_map(self):
        hidden_states = torch.randn(6, 128)
        context_metadata = {
            "expanded_row_idx": torch.tensor([0, 1, 1, 1, 1, 1]),
            "topk_weights": torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        }
        self.dispatcher.original_shape = (6, 128)
        final_hidden_states = self.dispatcher.token_combine(
            hidden_states, context_metadata).routed_out
        self.mock_npu_moe_token_unpermute.assert_called_once()
        self.assertEqual(final_hidden_states.shape, (6, 128))

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_token_dispatch_with_router_weight(self):
        self.dispatcher.apply_router_weight_on_input = True
        hidden_states = torch.randn(3, 128)
        topk_weights = torch.tensor([[0.7], [0.6], [0.5]])  # topk=1
        topk_ids = torch.tensor([[0], [1], [2]])

        results = self.dispatcher.token_dispatch(hidden_states, topk_weights,
                                                 topk_ids, None)
        self.assertEqual(results.hidden_states.shape, (6, 128))


class TestTokenDispatcherWithAll2AllV(TestBase):

    def setUp(self):
        # Patch properties
        patcher1 = patch.object(TokenDispatcherWithAll2AllV,
                                'ep_group',
                                new_callable=PropertyMock,
                                return_value=MagicMock())
        patcher2 = patch.object(TokenDispatcherWithAll2AllV,
                                'ep_rank',
                                new_callable=PropertyMock,
                                return_value=0)
        patcher3 = patch.object(TokenDispatcherWithAll2AllV,
                                'ep_size',
                                new_callable=PropertyMock,
                                return_value=2)

        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.addCleanup(patcher3.stop)

        self.mock_ep_group_prop = patcher1.start()
        self.mock_ep_rank_prop = patcher2.start()
        self.mock_ep_size_prop = patcher3.start()

        # Mock torch_npu.npu_moe_token_permute
        patcher4 = patch('torch_npu.npu_moe_token_permute')
        self.mock_npu_moe_token_permute = patcher4.start()
        self.addCleanup(patcher4.stop)
        self.mock_npu_moe_token_permute.return_value = (torch.randn(16, 16),
                                                        torch.arange(16))

        # Mock torch_npu.npu_moe_token_unpermute
        patcher5 = patch('torch_npu.npu_moe_token_unpermute')
        self.mock_npu_moe_token_unpermute = patcher5.start()
        self.addCleanup(patcher5.stop)
        self.mock_npu_moe_token_unpermute.return_value = torch.randn(8, 16)

        # Mock async_all_to_all
        patcher6 = patch(
            'vllm_ascend.ops.fused_moe.comm_utils.async_all_to_all')
        self.mock_async_all_to_all = patcher6.start()
        self.addCleanup(patcher6.stop)
        self.mock_async_all_to_all.return_value = (None, torch.randn(16, 16),
                                                   MagicMock())

        # Mock gather_from_sequence_parallel_region
        patcher7 = patch(
            'vllm_ascend.ops.fused_moe.token_dispatcher.gather_from_sequence_parallel_region'
        )
        self.mock_gather_from_sequence_parallel_region = patcher7.start()
        self.addCleanup(patcher7.stop)
        self.mock_gather_from_sequence_parallel_region.return_value = torch.tensor(
            [[2, 2, 2, 2], [2, 2, 2, 2]], dtype=torch.int64)

        # Mock torch.histc
        patcher8 = patch('torch.histc')
        self.mock_histc = patcher8.start()
        self.addCleanup(patcher8.stop)
        self.mock_histc.return_value = torch.tensor([2, 2, 2, 2],
                                                    dtype=torch.int64)

        # Mock torch.npu.current_device
        patcher9 = patch('torch.npu.current_device')
        self.mock_current_device = patcher9.start()
        self.addCleanup(patcher9.stop)
        self.mock_current_device.return_value = 'cpu'

        # Mock torch_npu.npu_dynamic_quant
        patcher10 = patch('torch_npu.npu_dynamic_quant')
        self.mock_npu_dynamic_quant = patcher10.start()
        self.addCleanup(patcher10.stop)
        self.mock_npu_dynamic_quant.return_value = (torch.randn(16, 16),
                                                    torch.randn(16))

        # Mock torch.ops._C_ascend.npu_moe_init_routing_custom
        patcher11 = patch('torch.ops._C_ascend.npu_moe_init_routing_custom')
        self.mock_npu_moe_init_routing_custom = patcher11.start()
        self.addCleanup(patcher11.stop)
        self.mock_npu_moe_init_routing_custom.return_value = (torch.randn(
            16, 16), torch.arange(16), None, torch.randn(16))

        # Mock torch.repeat_interleave
        patcher12 = patch('torch.repeat_interleave')
        self.mock_repeat_interleave = patcher12.start()
        self.addCleanup(patcher12.stop)
        self.mock_repeat_interleave.return_value = torch.arange(16)

        self.dispatcher = TokenDispatcherWithAll2AllV(top_k=2,
                                                      num_experts=4,
                                                      num_local_experts=2,
                                                      with_quant=False)

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_token_dispatch(self):
        hidden_states = torch.randn(8, 16)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2)).long()
        expert_map = torch.tensor([0, 1, 2, 3])

        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [0, 1], dtype=torch.int32)
        self.dispatcher.local_expert_indices = [0, 1]

        result = self.dispatcher.token_dispatch(hidden_states=hidden_states,
                                                topk_weights=topk_weights,
                                                topk_ids=topk_ids,
                                                expert_map=expert_map)

        self.assertIsNotNone(result.hidden_states)
        self.assertIsNotNone(result.group_list)
        self.assertEqual(result.group_list_type, 1)

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_token_combine(self):
        hidden_states = torch.randn(16, 16)
        context_metadata = {
            "input_splits": [4, 4],
            "output_splits": [4, 4],
            "topk_weights": torch.rand(8, 4),
            "reversed_local_input_permutation_mapping": torch.arange(8),
            "reversed_global_input_permutation_mapping": torch.arange(16),
        }
        self.dispatcher.hidden_shape = (8, 16)
        self.dispatcher.hidden_shape_before_permute = (8, 16)
        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [0, 1], dtype=torch.int32)
        self.dispatcher.local_expert_indices = [0, 1]

        output = self.dispatcher.token_combine(hidden_states, context_metadata)
        self.assertIsNotNone(output)
        self.assertEqual(output.routed_out.shape, (8, 16))

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_token_dispatch_with_quant(self):
        self.dispatcher = TokenDispatcherWithAll2AllV(top_k=2,
                                                      num_experts=4,
                                                      num_local_experts=2)

        hidden_states = torch.randn(8, 16)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2)).long()
        expert_map = torch.tensor([0, 1, 2, 3])

        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [0, 1], dtype=torch.int32)
        self.dispatcher.local_expert_indices = [0, 1]

        result = self.dispatcher.token_dispatch(hidden_states=hidden_states,
                                                topk_weights=topk_weights,
                                                topk_ids=topk_ids,
                                                expert_map=expert_map,
                                                with_quant=True)

        self.assertIsNotNone(result.hidden_states)
        self.assertIsNotNone(result.group_list)
        self.assertIsNotNone(result.dynamic_scale)
        self.assertEqual(result.group_list_type, 1)


class TestTokenDispatcherWithAll2AllvTokenDrop(TestBase):

    def setUp(self):
        patcher1 = patch.object(TokenDispatcherWithAll2AllvTokenDrop,
                                'ep_group',
                                new_callable=PropertyMock,
                                return_value=MagicMock())
        patcher2 = patch.object(TokenDispatcherWithAll2AllvTokenDrop,
                                'ep_rank',
                                new_callable=PropertyMock,
                                return_value=0)
        patcher3 = patch.object(TokenDispatcherWithAll2AllvTokenDrop,
                                'ep_size',
                                new_callable=PropertyMock,
                                return_value=2)
        patcher4 = patch('torch.distributed.get_rank', return_value=0)
        patcher5 = patch('torch.npu.current_device', return_value='npu:0')

        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.addCleanup(patcher3.stop)
        self.addCleanup(patcher4.stop)
        self.addCleanup(patcher5.stop)

        patcher1.start()
        patcher2.start()
        patcher3.start()
        patcher4.start()
        patcher5.start()

        self.dispatcher = TokenDispatcherWithAll2AllvTokenDrop(
            top_k=6,
            num_experts=4,
            num_local_experts=2,
            token_drop_load_factor=1.2,
        )

    def test_capacity_ratio_with_high_topk(self):
        # pytest tests/ut/ops/test_token_dispatcher.py -k capacity_ratio_with_high_topk -q -s
        num_tokens = 16
        num_experts = self.dispatcher.num_experts
        load_factor = self.dispatcher.token_drop_load_factor

        topk_ids = torch.zeros((num_tokens, 6), dtype=torch.int64, device='npu:0')
        topk_ids[:, 0] = 0
        topk_ids[:, 1] = 1
        topk_ids[:, 2] = 0
        topk_ids[:, 3] = 2
        topk_ids[:, 4] = 0
        topk_ids[:, 5] = 3


        local_hist = torch.bincount(topk_ids.reshape(-1),
                        minlength=num_experts).to(dtype=torch.float32,
                                      device='npu:0')
        remote_hist = torch.tensor([4, 8, 10, 10],
                       dtype=torch.float32,
                       device='npu:0')
        global_hist = torch.stack([local_hist, remote_hist], dim=0)

        with patch(
                'vllm_ascend.ops.fused_moe.token_dispatcher.gather_from_sequence_parallel_region',
                return_value=global_hist):
            (_, _, _, _, _, _, expert_capacity,
             global_avg_tokens_per_expert,
             global_before_drop) = self.dispatcher._preprocess_with_token_drop(
                 topk_ids)

        kept_tokens = self.dispatcher._compute_kept_tokens_per_rank_expert(
            global_before_drop, expert_capacity)
        per_expert_after_drop = kept_tokens.sum(dim=0).to(torch.float32)

        avg_before_drop = float(global_avg_tokens_per_expert.item())
        assert avg_before_drop > 0
        print(f"Token distribution before drop:\n{global_before_drop}")
        print(f"Expert capacity: {expert_capacity}")
        print(f"Token distribution after drop:\n{per_expert_after_drop}")

        assert torch.all(per_expert_after_drop <= expert_capacity + 1e-6)

    def test_drop_by_score_respects_capacity_for_various_configs(self):
        cases = [
            (4, 2, 24, 6),
            (6, 3, 18, 4),
            (8, 4, 16, 3),
            (4, 1, 1, 0),
            (8, 1, 2, 0),
        ]

        for num_experts, topk, num_tokens, expert_capacity in cases:
            dispatcher = TokenDispatcherWithAll2AllvTokenDrop(
                top_k=topk,
                num_experts=num_experts,
                num_local_experts=2,
                token_drop_load_factor=1.2,
            )

            subset_size = min(num_experts, max(topk + 1, 2))
            row_offsets = torch.arange(num_tokens,
                                       dtype=torch.int64,
                                       device='npu:0').unsqueeze(-1)
            col_offsets = torch.arange(topk,
                                       dtype=torch.int64,
                                       device='npu:0').unsqueeze(0)
            topk_ids = (row_offsets + col_offsets) % subset_size
            topk_weights = torch.rand((num_tokens, topk),
                                      dtype=torch.float32,
                                      device='npu:0')

            out_weights, out_ids = dispatcher._get_topk_ids_weights_after_drop_by_expert(
                topk_ids, topk_weights, expert_capacity)

            valid_mask = out_ids < num_experts
            kept_counts = torch.bincount(out_ids[valid_mask].reshape(-1),
                                         minlength=num_experts)

            assert out_ids.shape == topk_ids.shape
            assert out_weights.shape == topk_weights.shape
            assert torch.all(kept_counts <= expert_capacity)
            assert torch.all(out_weights[~valid_mask] == 0)
            assert torch.all(out_ids[valid_mask] == topk_ids[valid_mask])

    def test_drop_by_device_respects_capacity_for_various_configs(self):
        cases = [
            # (num_experts, num_local_experts, topk, num_tokens, device_capacity)
            (4, 2, 2, 24, 10),
            (8, 2, 4, 16, 12),
            (8, 4, 2, 20, 8),
            (4, 2, 1, 6, 2),
            (128, 8, 8, 128, 56),
        ]

        for num_experts, num_local_experts, topk, num_tokens, device_capacity in cases:
            dispatcher = TokenDispatcherWithAll2AllvTokenDrop(
                top_k=topk,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                token_drop_load_factor=1.2,
            )

            subset_size = min(num_experts, max(topk + 1, 2))
            row_offsets = torch.arange(num_tokens,
                                       dtype=torch.int64,
                                       device='npu:0').unsqueeze(-1)
            col_offsets = torch.arange(topk,
                                       dtype=torch.int64,
                                       device='npu:0').unsqueeze(0)
            topk_ids = (row_offsets + col_offsets) % subset_size
            topk_weights = torch.rand((num_tokens, topk),
                                      dtype=torch.float32,
                                      device='npu:0')

            out_weights, out_ids = dispatcher._get_topk_ids_weights_after_drop_by_device(
                topk_ids, topk_weights, device_capacity)

            valid_mask = out_ids < num_experts
            num_devices = num_experts // num_local_experts
            kept_device_counts = torch.bincount(
                (out_ids[valid_mask] // num_local_experts).reshape(-1),
                minlength=num_devices)

            assert out_ids.shape == topk_ids.shape
            assert out_weights.shape == topk_weights.shape
            assert torch.all(kept_device_counts <= device_capacity)
            assert torch.all(out_weights[~valid_mask] == 0)
            assert torch.all(out_ids[valid_mask] == topk_ids[valid_mask])

    def test_drop_by_device_large_ep_config(self):
        dispatcher = TokenDispatcherWithAll2AllvTokenDrop(
            top_k=8,
            num_experts=128,
            num_local_experts=8,
            token_drop_load_factor=1.2,
        )

        num_tokens = 256
        topk = 8
        device_capacity = 96

        topk_ids = torch.randint(0,
                                 dispatcher.num_experts,
                                 (num_tokens, topk),
                                 dtype=torch.int64,
                                 device='npu:0')
        topk_weights = torch.rand((num_tokens, topk),
                                  dtype=torch.float32,
                                  device='npu:0')

        out_weights, out_ids = dispatcher._get_topk_ids_weights_after_drop_by_device(
            topk_ids, topk_weights, device_capacity)

        valid_mask = out_ids < dispatcher.num_experts
        num_devices = dispatcher.num_experts // dispatcher.num_local_experts
        kept_device_counts = torch.bincount(
            (out_ids[valid_mask] // dispatcher.num_local_experts).reshape(-1),
            minlength=num_devices)

        assert out_ids.shape == topk_ids.shape
        assert out_weights.shape == topk_weights.shape
        assert torch.all(kept_device_counts <= device_capacity)
        assert torch.all(out_weights[~valid_mask] == 0)

    def test_drop_by_device_exact_topscore_assignments_large_ep(self):
        dispatcher = TokenDispatcherWithAll2AllvTokenDrop(
            top_k=8,
            num_experts=128,
            num_local_experts=8,
            token_drop_load_factor=1.2,
        )

        num_tokens = 256
        topk = 8
        device_capacity = 5

        # Deterministic ids and strictly unique scores for exact-set validation.
        # 严格每个专家16个token
        flat_ids = torch.arange(num_tokens * topk,
                                dtype=torch.int64,
                                device='npu:0') % dispatcher.num_experts
        topk_ids = flat_ids.view(num_tokens, topk)
        topk_weights = torch.arange(num_tokens * topk,
                                    dtype=torch.float32,
                                    device='npu:0').view(num_tokens, topk)

        out_weights, out_ids = dispatcher._get_topk_ids_weights_after_drop_by_device(
            topk_ids, topk_weights, device_capacity)

        flat_device_ids = (topk_ids.reshape(-1) //
                           dispatcher.num_local_experts).to(torch.int64)
        flat_scores = topk_weights.reshape(-1)
        expected_keep_flat = torch.zeros_like(flat_scores, dtype=torch.bool)

        num_devices = dispatcher.num_experts // dispatcher.num_local_experts
        for device_id in range(num_devices):
            device_mask = (flat_device_ids == device_id)
            device_indices = torch.nonzero(device_mask,
                                           as_tuple=False).squeeze(-1)
            if device_indices.numel() == 0:
                continue
            k = min(device_capacity, int(device_indices.numel()))
            device_scores = flat_scores.index_select(0, device_indices)
            _, keep_local = torch.topk(device_scores,
                                       k=k,
                                       dim=0,
                                       sorted=False)
            keep_indices = device_indices.index_select(0, keep_local)
            expected_keep_flat.scatter_(0, keep_indices, True)

        expected_keep = expected_keep_flat.view_as(topk_ids)
        actual_keep = out_ids < dispatcher.num_experts

        assert torch.equal(actual_keep, expected_keep)
        assert torch.all(out_ids[actual_keep] == topk_ids[actual_keep])
        assert torch.all(out_weights[actual_keep] == topk_weights[actual_keep])
        assert torch.all(out_ids[~actual_keep] == dispatcher.num_experts)
        assert torch.all(out_weights[~actual_keep] == 0)

    def test_drop_by_expert_exact_topscore_assignments_large_ep(self):
        dispatcher = TokenDispatcherWithAll2AllvTokenDrop(
            top_k=8,
            num_experts=128,
            num_local_experts=8,
            token_drop_load_factor=1.2,
        )

        num_tokens = 256
        topk = 8
        expert_capacity = 3

        # Deterministic ids and strictly unique scores for exact-set validation.
        flat_ids = torch.arange(num_tokens * topk,
                                dtype=torch.int64,
                                device='npu:0') % dispatcher.num_experts
        topk_ids = flat_ids.view(num_tokens, topk)
        topk_weights = torch.arange(num_tokens * topk,
                                    dtype=torch.float32,
                                    device='npu:0').view(num_tokens, topk)

        out_weights, out_ids = dispatcher._get_topk_ids_weights_after_drop_by_expert(
            topk_ids, topk_weights, expert_capacity)

        flat_scores = topk_weights.reshape(-1)
        expected_keep_flat = torch.zeros_like(flat_scores, dtype=torch.bool)

        for expert_id in range(dispatcher.num_experts):
            expert_mask = (topk_ids.reshape(-1) == expert_id)
            expert_indices = torch.nonzero(expert_mask,
                                           as_tuple=False).squeeze(-1)
            if expert_indices.numel() == 0:
                continue
            k = min(expert_capacity, int(expert_indices.numel()))
            expert_scores = flat_scores.index_select(0, expert_indices)
            _, keep_local = torch.topk(expert_scores,
                                       k=k,
                                       dim=0,
                                       sorted=False)
            keep_indices = expert_indices.index_select(0, keep_local)
            expected_keep_flat.scatter_(0, keep_indices, True)

        expected_keep = expected_keep_flat.view_as(topk_ids)
        actual_keep = out_ids < dispatcher.num_experts

        assert torch.equal(actual_keep, expected_keep)
        assert torch.all(out_ids[actual_keep] == topk_ids[actual_keep])
        assert torch.all(out_weights[actual_keep] == topk_weights[actual_keep])
        assert torch.all(out_ids[~actual_keep] == dispatcher.num_experts)
        assert torch.all(out_weights[~actual_keep] == 0)

    def test_preprocess_shapes_and_invariants_for_various_ep_topk(self):
        cases = [
            (4, 2, 2, 12),
            (6, 3, 3, 10),
            (8, 4, 2, 9),
            (4, 4, 1, 1),
            (8, 8, 1, 2),
            (128, 16, 8, 32),
        ]

        for num_experts, ep_size, topk, num_tokens in cases:
            num_local_experts = num_experts // ep_size
            assert num_local_experts > 0

            mock_ep_group = MagicMock()
            mock_ep_group._get_backend.return_value.get_hccl_comm_name.return_value = "hccl_123"

            def _all_gatherv(x, *args, **kwargs):
                return torch.cat([x] * ep_size, dim=0)

            mock_ep_group.all_gatherv.side_effect = _all_gatherv

            with patch.object(TokenDispatcherWithAll2AllvTokenDrop,
                              'ep_group',
                              new_callable=PropertyMock,
                              return_value=mock_ep_group), \
                 patch.object(TokenDispatcherWithAll2AllvTokenDrop,
                              'ep_rank',
                              new_callable=PropertyMock,
                              return_value=0), \
                 patch.object(TokenDispatcherWithAll2AllvTokenDrop,
                              'ep_size',
                              new_callable=PropertyMock,
                              return_value=ep_size), \
                 patch('torch.distributed.get_rank', return_value=0), \
                 patch('torch.npu.current_device', return_value='npu:0'):

                dispatcher = TokenDispatcherWithAll2AllvTokenDrop(
                    top_k=topk,
                    num_experts=num_experts,
                    num_local_experts=num_local_experts,
                    token_drop_load_factor=1.1,
                )

                topk_ids = torch.arange(num_tokens * topk,
                                        device='npu:0').reshape(
                                            num_tokens,
                                            topk) % num_experts
                topk_ids = topk_ids.to(torch.int64)
                topk_weights = torch.rand((num_tokens, topk),
                                          dtype=torch.float32,
                                          device='npu:0')

                forward_context = MagicMock()
                forward_context.dp_metadata.num_tokens_across_dp_cpu = torch.full(
                    (ep_size, ), num_tokens, dtype=torch.int64)
                local_hist = torch.bincount(topk_ids.reshape(-1),
                                            minlength=num_experts).to(
                                                torch.int64)
                global_hist = torch.stack([local_hist] * ep_size,
                                          dim=0).reshape(-1)

                with patch(
                        'vllm_ascend.ops.fused_moe.token_dispatcher.get_forward_context',
                        return_value=forward_context), patch(
                            'vllm_ascend.ops.fused_moe.token_dispatcher.gather_from_sequence_parallel_region',
                            return_value=global_hist):
                    (num_tokens_per_local_expert, input_splits, output_splits,
                     num_global_tokens_per_local_expert,
                     global_input_tokens_local_experts_indices,
                     local_permute_keep_indices, expert_capacity,
                     global_avg_tokens_per_expert,
                     num_global_tokens_per_expert_before_drop
                     ) = dispatcher._preprocess_with_token_drop(
                         topk_ids, topk_weights)

            assert num_tokens_per_local_expert.shape == (num_local_experts, )
            assert input_splits.shape[0] == ep_size
            assert output_splits.shape[0] == ep_size
            assert num_global_tokens_per_local_expert.shape == (
                ep_size, num_local_experts)
            assert num_global_tokens_per_expert_before_drop.shape == (
                ep_size, num_experts)
            if num_local_experts > 1:
                assert global_input_tokens_local_experts_indices is not None
                assert global_input_tokens_local_experts_indices.numel() == \
                    int(num_global_tokens_per_local_expert.sum().item())
            assert local_permute_keep_indices.dtype == torch.int64
            assert local_permute_keep_indices.numel() <= topk_ids.numel()
            assert expert_capacity >= 0
            assert float(global_avg_tokens_per_expert.item()) >= 0
            assert int(input_splits.sum()) == int(local_permute_keep_indices.numel())

    def test_preprocess_local_mode_no_topk_allgather_and_split_consistency(self):
        dispatcher = TokenDispatcherWithAll2AllvTokenDrop(
            top_k=2,
            num_experts=4,
            num_local_experts=2,
            token_drop_load_factor=1.0,
            token_drop_local_only=True,
        )

        topk_ids = torch.tensor(
            [[0, 1], [2, 3], [1, 2], [0, 3]],
            dtype=torch.int64,
            device='npu:0')
        topk_weights = torch.tensor(
            [[0.9, 0.8], [0.7, 0.6], [0.5, 0.4], [0.3, 0.2]],
            dtype=torch.float32,
            device='npu:0')

        dropped_ids = torch.tensor(
            [[0, 4], [2, 4], [1, 2], [4, 3]],
            dtype=torch.int64,
            device='npu:0')
        dropped_weights = torch.where(dropped_ids < dispatcher.num_experts,
                                      topk_weights,
                                      torch.zeros_like(topk_weights))

        local_before = torch.bincount(topk_ids.reshape(-1),
                                      minlength=dispatcher.num_experts).to(
                                          torch.int64)
        local_after = torch.bincount(
            dropped_ids[dropped_ids < dispatcher.num_experts].reshape(-1),
            minlength=dispatcher.num_experts).to(torch.int64)
        remote_before = torch.tensor([1, 2, 3, 2],
                                     dtype=torch.int64,
                                     device='npu:0')
        remote_after = torch.tensor([2, 1, 1, 1],
                                    dtype=torch.int64,
                                    device='npu:0')
        global_before = torch.stack([local_before, remote_before], dim=0)
        global_after = torch.stack([local_after, remote_after], dim=0)

        forward_context = MagicMock()
        forward_context.dp_metadata.num_tokens_across_dp_cpu = torch.tensor(
            [topk_ids.shape[0], topk_ids.shape[0]], dtype=torch.int64)

        with patch(
                'vllm_ascend.ops.fused_moe.token_dispatcher.get_forward_context',
                return_value=forward_context), patch(
                    'vllm_ascend.ops.fused_moe.token_dispatcher.gather_from_sequence_parallel_region',
                    side_effect=[global_before.reshape(-1),
                                 global_after.reshape(-1)]), patch(
                                     'torch_npu.distributed.all_gather_into_tensor_uneven'
                                 ) as mock_all_gather, patch.object(
                                     TokenDispatcherWithAll2AllvTokenDrop,
                                     '_get_topk_ids_weights_after_drop_by_expert',
                                     return_value=(dropped_weights,
                                                   dropped_ids)):
            (num_tokens_per_local_expert, input_splits, output_splits,
             num_global_tokens_per_local_expert,
             global_input_tokens_local_experts_indices,
             local_permute_keep_indices, _, _,
             _) = dispatcher._preprocess_with_token_drop(
                 topk_ids, topk_weights, step=1)

        mock_all_gather.assert_not_called()

        expected_output_splits = global_after[:, :dispatcher.num_local_experts]
        expected_output_splits = expected_output_splits.sum(
            dim=-1).cpu().numpy()
        expected_input_splits = global_after[dispatcher.ep_rank].reshape(
            dispatcher.ep_size, dispatcher.num_local_experts).sum(
                dim=1).cpu().numpy()
        expected_tokens_per_local_expert = global_after[:, :dispatcher.
                                                        num_local_experts].sum(
                                                            dim=0)

        self.assertEqual(output_splits.tolist(), expected_output_splits.tolist())
        self.assertEqual(input_splits.tolist(), expected_input_splits.tolist())
        self.assertTrue(
            torch.equal(num_tokens_per_local_expert,
                        expected_tokens_per_local_expert))
        if dispatcher.num_local_experts > 1:
            self.assertIsNotNone(global_input_tokens_local_experts_indices)
        self.assertEqual(int(input_splits.sum()),
                         int(local_permute_keep_indices.numel()))
        self.assertEqual(num_global_tokens_per_local_expert.shape,
                         (dispatcher.ep_size, dispatcher.num_local_experts))

    def test_drop_by_score_extreme_all_dropped_and_single_path(self):
        dispatcher = TokenDispatcherWithAll2AllvTokenDrop(
            top_k=1,
            num_experts=4,
            num_local_experts=2,
            token_drop_load_factor=0.0,
        )

        topk_ids = torch.tensor([[0], [1], [2], [3]],
                                dtype=torch.int64,
                                device='npu:0')
        topk_weights = torch.tensor([[0.9], [0.8], [0.7], [0.6]],
                                    dtype=torch.float32,
                                    device='npu:0')

        out_weights, out_ids = dispatcher._get_topk_ids_weights_after_drop_by_expert(
            topk_ids, topk_weights, expert_capacity=0)

        assert torch.all(out_ids == dispatcher.num_experts)
        assert torch.all(out_weights == 0)

    def test_drop_by_device_extreme_all_dropped_and_single_path(self):
        dispatcher = TokenDispatcherWithAll2AllvTokenDrop(
            top_k=2,
            num_experts=4,
            num_local_experts=2,
            token_drop_load_factor=0.0,
        )

        topk_ids = torch.tensor([[0, 1], [2, 3], [1, 2], [0, 3]],
                                dtype=torch.int64,
                                device='npu:0')
        topk_weights = torch.tensor([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4],
                                     [0.3, 0.2]],
                                    dtype=torch.float32,
                                    device='npu:0')

        out_weights, out_ids = dispatcher._get_topk_ids_weights_after_drop_by_device(
            topk_ids, topk_weights, device_capacity=0)

        assert torch.all(out_ids == dispatcher.num_experts)
        assert torch.all(out_weights == 0)

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_token_dispatch_with_quant_no_active_tokens(self):
        self.dispatcher = TokenDispatcherWithAll2AllV(top_k=2,
                                                      num_experts=4,
                                                      num_local_experts=2)

        self.mock_repeat_interleave.return_value = torch.tensor(
            [], dtype=torch.long)

        hidden_states = torch.randn(8, 16)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2)).long()
        expert_map = torch.tensor([0, 1, 2, 3])

        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [0, 1], dtype=torch.int32)
        self.dispatcher.local_expert_indices = [0, 1]

        result = self.dispatcher.token_dispatch(hidden_states=hidden_states,
                                                topk_weights=topk_weights,
                                                topk_ids=topk_ids,
                                                expert_map=expert_map,
                                                with_quant=True)

        self.assertIsNotNone(result.hidden_states)
        self.assertIsNotNone(result.group_list)
        self.assertIsNotNone(result.dynamic_scale)
        self.assertEqual(result.group_list_type, 1)


class TestTokenDispatcherWithAll2AllvExpandedDrop(TestBase):
    """
    Unit tests for Expanded Drop strategy.

    Expanded Drop strategy:
    - Extends each token's candidate expert set by adding all local experts of the rank where the token resides
    - Performs capacity-based token drop on the expanded candidate set
    - Deduplicates when local experts overlap with top-k experts
    """

    def setUp(self):
        # Patch properties
        patcher1 = patch.object(TokenDispatcherWithAll2AllvExpandedDrop,
                                'ep_group',
                                new_callable=PropertyMock,
                                return_value=MagicMock())
        patcher2 = patch.object(TokenDispatcherWithAll2AllvExpandedDrop,
                                'ep_rank',
                                new_callable=PropertyMock,
                                return_value=0)
        patcher3 = patch.object(TokenDispatcherWithAll2AllvExpandedDrop,
                                'ep_size',
                                new_callable=PropertyMock,
                                return_value=4)

        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.addCleanup(patcher3.stop)

        patcher1.start()
        patcher2.start()
        patcher3.start()

        # Common test parameters
        self.num_experts = 16
        self.num_local_experts = 4
        self.top_k = 2
        self.token_drop_load_factor = 1.2
        self.expanded_topk = self.top_k + self.num_local_experts

        # Create dispatcher with patches
        with patch('torch.npu.current_device', return_value='cpu'), \
             patch('torch.distributed.get_rank', return_value=0):
            self.dispatcher = TokenDispatcherWithAll2AllvExpandedDrop(
                top_k=self.top_k,
                num_experts=self.num_experts,
                num_local_experts=self.num_local_experts,
                token_drop_load_factor=self.token_drop_load_factor,
            )

        # Set additional attributes
        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [i % self.num_local_experts for i in range(self.num_experts)],
            dtype=torch.int32
        )
        self.dispatcher.local_expert_indices = list(
            range(0, self.num_local_experts)  # Rank 0's local experts
        )

    # ========================================
    # Test 1: Expanded Candidate Set Construction
    # ========================================

    def test_expanded_candidate_set_shape(self):
        """Test that expanded candidate set has correct shape."""
        num_tokens = 16
        topk = self.top_k

        global_topk_ids = torch.randint(
            0, self.num_experts, (num_tokens, topk), dtype=torch.int64
        )
        global_topk_weights = torch.rand((num_tokens, topk), dtype=torch.float32)
        global_router_logits = torch.rand(
            (num_tokens, self.num_experts), dtype=torch.float32
        )

        # Single rank scenario
        num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int64)
        expert_capacity = 10

        expanded_weights, expanded_ids = \
            self.dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                global_topk_ids,
                global_topk_weights,
                global_router_logits,
                expert_capacity,
                num_tokens_across_dp,
            )

        expected_shape = (num_tokens, self.expanded_topk)
        assert expanded_ids.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {expanded_ids.shape}"
        assert expanded_weights.shape == expected_shape

    def test_expanded_candidate_set_local_expert_indices(self):
        """
        Test that each token's local experts are correctly expanded.

        For rank 0 tokens, local experts should be [0, 1, 2, 3].
        For rank 1 tokens, local experts should be [4, 5, 6, 7].
        """
        num_tokens_per_rank = 8
        T_global = num_tokens_per_rank * 4  # 4 ranks
        topk = self.top_k

        # Create global topk_ids
        global_topk_ids = torch.zeros((T_global, topk), dtype=torch.int64)
        global_topk_ids[:, 0] = torch.arange(T_global) % self.num_experts
        global_topk_ids[:, 1] = (torch.arange(T_global) + 1) % self.num_experts

        global_topk_weights = torch.rand((T_global, topk), dtype=torch.float32)
        global_router_logits = torch.rand(
            (T_global, self.num_experts), dtype=torch.float32
        )

        num_tokens_across_dp = torch.tensor(
            [num_tokens_per_rank] * 4, dtype=torch.int64
        )
        expert_capacity = 10

        expanded_weights, expanded_ids = \
            self.dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                global_topk_ids,
                global_topk_weights,
                global_router_logits,
                expert_capacity,
                num_tokens_across_dp,
            )

        # Verify local expert indices for each rank
        token_ranges = self.dispatcher._rank_token_ranges(num_tokens_across_dp)

        for rank, (start, end) in enumerate(token_ranges):
            local_expert_start = rank * self.num_local_experts
            expected_local_experts = list(range(
                local_expert_start,
                local_expert_start + self.num_local_experts
            ))

            # Check expanded portion (columns topk:)
            for token_idx in range(start, end):
                actual_local_experts = expanded_ids[token_idx, topk:].tolist()
                # Filter out sentinel values
                actual_local_experts = [
                    e for e in actual_local_experts if e != self.num_experts
                ]

                for expert in actual_local_experts:
                    assert expert in expected_local_experts, \
                        f"Token {token_idx} on rank {rank}: " \
                        f"expert {expert} not in expected {expected_local_experts}"

    # ========================================
    # Test 2: Deduplication Logic
    # ========================================

    def test_deduplication_local_expert_overlaps_with_topk(self):
        """
        Test that when a local expert is already in top-k,
        the expanded position is marked as sentinel.
        """
        num_tokens = 4
        topk = self.top_k

        # Token 0: top-k = [0, 5], local experts [0,1,2,3]
        # Expert 0 appears in both top-k and local experts -> should be deduplicated
        global_topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int64)
        global_topk_ids[0, 0] = 0  # Expert 0 (local to rank 0)
        global_topk_ids[0, 1] = 5  # Expert 5 (remote)
        global_topk_ids[1:, 0] = 5
        global_topk_ids[1:, 1] = 6

        global_topk_weights = torch.rand((num_tokens, topk), dtype=torch.float32)
        global_router_logits = torch.rand(
            (num_tokens, self.num_experts), dtype=torch.float32
        )

        num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int64)
        expert_capacity = 10

        expanded_weights, expanded_ids = \
            self.dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                global_topk_ids,
                global_topk_weights,
                global_router_logits,
                expert_capacity,
                num_tokens_across_dp,
            )

        # Token 0's expanded position 0 (expert 0) should be sentinel
        token0_expanded_local = expanded_ids[0, topk:].tolist()

        # Position 0 in expanded portion corresponds to local expert 0
        # It should be marked as sentinel because it duplicates with top-k[0]
        assert token0_expanded_local[0] == self.num_experts, \
            f"Expected duplicated expert 0 to be sentinel, got {token0_expanded_local[0]}"
        assert expanded_weights[0, topk] == 0.0, \
            "Duplicated position should have weight 0"

    def test_deduplication_preserves_original_topk(self):
        """
        Verify that deduplication does not affect original top-k experts.

        When expert 0 is in both top-k and local experts:
        - Original top-k position should remain valid
        - Only expanded position should be marked as sentinel
        """
        num_tokens = 4
        topk = self.top_k

        global_topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int64)
        global_topk_ids[0, 0] = 0  # Expert 0 is in top-k

        global_topk_weights = torch.rand((num_tokens, topk), dtype=torch.float32)
        global_router_logits = torch.rand(
            (num_tokens, self.num_experts), dtype=torch.float32
        )
        # Give expert 0 high score
        global_router_logits[0, 0] = 10.0

        num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int64)
        expert_capacity = 10

        expanded_weights, expanded_ids = \
            self.dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                global_topk_ids,
                global_topk_weights,
                global_router_logits,
                expert_capacity,
                num_tokens_across_dp,
            )

        # Original top-k position should still have expert 0
        assert expanded_ids[0, 0] == 0, \
            f"Original top-k should be preserved, got {expanded_ids[0, 0]}"
        assert expanded_weights[0, 0] > 0, \
            "Original top-k weight should be preserved"

    # ========================================
    # Test 3: Capacity-Based Token Drop
    # ========================================

    def test_capacity_drop_keeps_highest_scores(self):
        """
        Test that when expert is overloaded, tokens with highest scores are kept.
        """
        num_tokens = 20
        topk = self.top_k
        expert_capacity = 5

        # All tokens select expert 0
        global_topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int64)
        global_topk_weights = torch.rand((num_tokens, topk), dtype=torch.float32)

        # Create router logits where expert 0 has increasing scores
        global_router_logits = torch.zeros(
            (num_tokens, self.num_experts), dtype=torch.float32
        )
        for i in range(num_tokens):
            global_router_logits[i, 0] = i * 1.0  # Unique, increasing scores

        num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int64)

        expanded_weights, expanded_ids = \
            self.dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                global_topk_ids,
                global_topk_weights,
                global_router_logits,
                expert_capacity,
                num_tokens_across_dp,
            )

        # Count valid tokens for expert 0
        valid_mask = expanded_ids < self.num_experts
        expert_0_mask = (expanded_ids == 0) & valid_mask
        count_expert_0 = expert_0_mask.sum().item()

        assert count_expert_0 <= expert_capacity, \
            f"Expert 0 should have at most {expert_capacity} tokens, got {count_expert_0}"

    def test_capacity_drop_per_expert_limit(self):
        """Verify that each expert has at most `capacity` tokens after drop."""
        num_tokens = 32
        topk = self.top_k

        global_topk_ids = torch.randint(
            0, self.num_experts, (num_tokens, topk), dtype=torch.int64
        )
        global_topk_weights = torch.rand((num_tokens, topk), dtype=torch.float32)
        global_router_logits = torch.rand(
            (num_tokens, self.num_experts), dtype=torch.float32
        )

        num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int64)
        expert_capacity = 8

        expanded_weights, expanded_ids = \
            self.dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                global_topk_ids,
                global_topk_weights,
                global_router_logits,
                expert_capacity,
                num_tokens_across_dp,
            )

        # Count valid tokens per expert
        valid_mask = expanded_ids < self.num_experts

        for expert_id in range(self.num_experts):
            expert_mask = (expanded_ids == expert_id) & valid_mask
            count = expert_mask.sum().item()

            assert count <= expert_capacity, \
                f"Expert {expert_id} has {count} tokens, exceeds capacity {expert_capacity}"

    # ========================================
    # Test 4: Statistics Computation
    # ========================================

    def test_compute_num_global_tokens_per_expert_after_drop(self):
        """Test num_global_tokens_per_expert_after_drop computation."""
        num_tokens_per_rank = 8
        T_global = num_tokens_per_rank * 4
        expanded_topk = self.expanded_topk

        # Create after_drop ids with known distribution
        global_topk_ids_after_drop = torch.full(
            (T_global, expanded_topk), self.num_experts, dtype=torch.int64
        )

        # Rank 0 tokens: assign to experts 0, 1
        global_topk_ids_after_drop[:num_tokens_per_rank, 0] = 0
        global_topk_ids_after_drop[:num_tokens_per_rank, 1] = 1

        # Rank 1 tokens: assign to experts 4, 5
        global_topk_ids_after_drop[
            num_tokens_per_rank:2*num_tokens_per_rank, 0
        ] = 4
        global_topk_ids_after_drop[
            num_tokens_per_rank:2*num_tokens_per_rank, 1
        ] = 5

        num_tokens_across_dp = torch.tensor(
            [num_tokens_per_rank] * 4, dtype=torch.int64
        )

        result = self.dispatcher._compute_num_global_tokens_per_expert_after_drop(
            global_topk_ids_after_drop, num_tokens_across_dp
        )

        # Verify shape
        assert result.shape == (4, self.num_experts), \
            f"Expected shape (4, {self.num_experts}), got {result.shape}"

        # Verify counts
        assert result[0, 0] == num_tokens_per_rank, \
            f"Rank 0 should have {num_tokens_per_rank} tokens for expert 0"
        assert result[0, 1] == num_tokens_per_rank, \
            f"Rank 0 should have {num_tokens_per_rank} tokens for expert 1"

    def test_rank_token_ranges(self):
        """Test _rank_token_ranges correctly computes token ranges for each rank."""
        num_tokens_across_dp = torch.tensor([8, 12, 6, 10], dtype=torch.int64)

        token_ranges = self.dispatcher._rank_token_ranges(num_tokens_across_dp)

        assert len(token_ranges) == 4
        assert token_ranges[0] == (0, 8)
        assert token_ranges[1] == (8, 20)
        assert token_ranges[2] == (20, 26)
        assert token_ranges[3] == (26, 36)

    # ========================================
    # Test 5: Edge Cases
    # ========================================

    def test_empty_token_input(self):
        """Test behavior with zero tokens."""
        global_topk_ids = torch.zeros((0, self.top_k), dtype=torch.int64)
        global_topk_weights = torch.zeros((0, self.top_k), dtype=torch.float32)
        global_router_logits = torch.zeros(
            (0, self.num_experts), dtype=torch.float32
        )

        num_tokens_across_dp = torch.tensor([0] * 4, dtype=torch.int64)
        expert_capacity = 5

        expanded_weights, expanded_ids = \
            self.dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                global_topk_ids,
                global_topk_weights,
                global_router_logits,
                expert_capacity,
                num_tokens_across_dp,
            )

        assert expanded_ids.shape == (0, self.expanded_topk)
        assert expanded_weights.shape == (0, self.expanded_topk)

    def test_capacity_zero_drops_all_tokens(self):
        """When capacity = 0, all tokens should be dropped."""
        num_tokens = 10

        global_topk_ids = torch.randint(
            0, self.num_experts, (num_tokens, self.top_k), dtype=torch.int64
        )
        global_topk_weights = torch.rand(
            (num_tokens, self.top_k), dtype=torch.float32
        )
        global_router_logits = torch.rand(
            (num_tokens, self.num_experts), dtype=torch.float32
        )

        num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int64)
        expert_capacity = 0

        expanded_weights, expanded_ids = \
            self.dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                global_topk_ids,
                global_topk_weights,
                global_router_logits,
                expert_capacity,
                num_tokens_across_dp,
            )

        # All positions should be sentinel
        assert torch.all(expanded_ids == self.num_experts), \
            "All tokens should be dropped when capacity=0"
        assert torch.all(expanded_weights == 0), \
            "All weights should be 0 when capacity=0"

    def test_large_capacity_no_drop(self):
        """When capacity is large enough, no tokens should be dropped."""
        num_tokens = 10

        global_topk_ids = torch.randint(
            0, self.num_experts, (num_tokens, self.top_k), dtype=torch.int64
        )
        global_topk_weights = torch.rand(
            (num_tokens, self.top_k), dtype=torch.float32
        )
        global_router_logits = torch.rand(
            (num_tokens, self.num_experts), dtype=torch.float32
        )

        num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int64)
        expert_capacity = num_tokens * self.top_k * 2  # Large capacity

        expanded_weights, expanded_ids = \
            self.dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                global_topk_ids,
                global_topk_weights,
                global_router_logits,
                expert_capacity,
                num_tokens_across_dp,
            )

        # Original top-k positions should all be valid
        assert torch.all(expanded_ids[:, :self.top_k] == global_topk_ids), \
            "Original top-k should be preserved when no drop"

    def test_capacity_one_keeps_only_highest_score(self):
        """Test that capacity=1 keeps only the highest-scoring token."""
        # Use single expert scenario
        dispatcher = TokenDispatcherWithAll2AllvExpandedDrop.__new__(
            TokenDispatcherWithAll2AllvExpandedDrop
        )
        dispatcher.num_experts = 4
        dispatcher.num_local_experts = 4
        dispatcher.top_k = 1
        dispatcher.expanded_topk = 1 + 4
        dispatcher.local_expert_indices = [0, 1, 2, 3]

        num_tokens = 10
        topk = 1

        global_topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int64)
        global_topk_weights = torch.ones((num_tokens, topk), dtype=torch.float32)
        global_router_logits = torch.zeros((num_tokens, 4), dtype=torch.float32)

        # Token 7 has the highest score for expert 0
        global_router_logits[7, 0] = 100.0
        for i in range(num_tokens):
            if i != 7:
                global_router_logits[i, 0] = float(i)

        num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int64)
        expert_capacity = 1

        expanded_weights, expanded_ids = \
            dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                global_topk_ids,
                global_topk_weights,
                global_router_logits,
                expert_capacity,
                num_tokens_across_dp,
            )

        # Only one token should be kept for expert 0
        valid_mask = expanded_ids < 4
        expert_0_mask = (expanded_ids == 0) & valid_mask

        assert expert_0_mask.sum().item() == 1, \
            "Only one token should be kept for expert 0"

    # ========================================
    # Test 6: Integration-style Tests
    # ========================================

    def test_full_expanded_drop_invariants(self):
        """
        Test invariants for the expanded drop workflow.

        Invariants:
        1. Sum of input_splits equals num_out_tokens_after_drop
        2. expanded_topk = topk + num_local_experts
        3. All valid tokens have non-zero weights
        """
        num_tokens = 16
        topk = self.top_k

        global_topk_ids = torch.randint(
            0, self.num_experts, (num_tokens, topk), dtype=torch.int64
        )
        global_topk_weights = torch.rand((num_tokens, topk), dtype=torch.float32)
        global_router_logits = torch.rand(
            (num_tokens, self.num_experts), dtype=torch.float32
        )

        num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int64)
        expert_capacity = 10

        expanded_weights, expanded_ids = \
            self.dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                global_topk_ids,
                global_topk_weights,
                global_router_logits,
                expert_capacity,
                num_tokens_across_dp,
            )

        # Invariant 1: expanded_topk = topk + num_local_experts
        assert expanded_ids.shape[1] == self.expanded_topk

        # Invariant 2: Valid tokens have non-zero weights
        valid_mask = expanded_ids < self.num_experts
        valid_weights = expanded_weights[valid_mask]
        assert torch.all(valid_weights != 0), \
            "Valid tokens should have non-zero weights"

        # Invariant 3: Dropped tokens have zero weights
        dropped_mask = expanded_ids == self.num_experts
        dropped_weights = expanded_weights[dropped_mask]
        assert torch.all(dropped_weights == 0), \
            "Dropped tokens should have zero weights"

    def test_expanded_drop_with_various_configs(self):
        """Test expanded drop with various configurations."""
        configs = [
            # (num_experts, num_local_experts, topk, num_tokens, capacity)
            (8, 2, 2, 16, 8),
            (16, 4, 2, 32, 10),
            (32, 4, 4, 64, 20),
            (64, 8, 4, 128, 30),
        ]

        for num_experts, num_local_experts, topk, num_tokens, capacity in configs:
            # Create temporary dispatcher
            with patch.object(
                TokenDispatcherWithAll2AllvExpandedDrop,
                'ep_group', new_callable=PropertyMock, return_value=MagicMock()
            ), \
            patch.object(
                TokenDispatcherWithAll2AllvExpandedDrop,
                'ep_rank', new_callable=PropertyMock, return_value=0
            ), \
            patch.object(
                TokenDispatcherWithAll2AllvExpandedDrop,
                'ep_size', new_callable=PropertyMock, return_value=1
            ):
                dispatcher = TokenDispatcherWithAll2AllvExpandedDrop(
                    top_k=topk,
                    num_experts=num_experts,
                    num_local_experts=num_local_experts,
                    token_drop_load_factor=1.2,
                )
                dispatcher.local_expert_indices = list(range(num_local_experts))

                global_topk_ids = torch.randint(
                    0, num_experts, (num_tokens, topk), dtype=torch.int64
                )
                global_topk_weights = torch.rand(
                    (num_tokens, topk), dtype=torch.float32
                )
                global_router_logits = torch.rand(
                    (num_tokens, num_experts), dtype=torch.float32
                )

                num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int64)

                expanded_weights, expanded_ids = \
                    dispatcher._get_topk_ids_weights_after_expanded_drop_by_expert(
                        global_topk_ids,
                        global_topk_weights,
                        global_router_logits,
                        capacity,
                        num_tokens_across_dp,
                    )

                # Verify shape
                expected_expanded_topk = topk + num_local_experts
                assert expanded_ids.shape == (num_tokens, expected_expanded_topk)

                # Verify capacity constraint
                valid_mask = expanded_ids < num_experts
                for expert_id in range(num_experts):
                    expert_mask = (expanded_ids == expert_id) & valid_mask
                    count = expert_mask.sum().item()
                    assert count <= capacity, \
                        f"Config ({num_experts}, {num_local_experts}, {topk}): " \
                        f"Expert {expert_id} exceeds capacity {capacity}"
