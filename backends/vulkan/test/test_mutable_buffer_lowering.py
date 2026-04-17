# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch

from executorch import exir
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir.delegate import executorch_call_delegate


class MutableStateModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("my_state", torch.zeros(4))

    def forward(self, x):
        y = x + self.my_state
        self.my_state.add_(1)
        return y


class TestMutableBufferLowering(unittest.TestCase):
    def test_mutable_buffer_remains_runtime_input(self):
        edge = exir.to_edge(
            torch.export.export(MutableStateModule(), (torch.randn(4),), strict=True)
        )
        lowered = edge.to_backend(VulkanPartitioner({}))

        self.assertEqual(
            lowered.exported_program().graph_signature.buffers_to_mutate,
            {"getitem_1": "my_state"},
        )

        delegate_nodes = [
            node
            for node in lowered.exported_program().graph.nodes
            if node.op == "call_function" and node.target == executorch_call_delegate
        ]
        self.assertEqual(len(delegate_nodes), 1)
        delegate_args = [
            arg.name for arg in delegate_nodes[0].args[1:] if isinstance(arg, torch.fx.Node)
        ]
        self.assertEqual(delegate_args, ["x", "b_my_state"])

        lowered_node = delegate_nodes[0].args[0]
        lowered_module = getattr(lowered.exported_program().graph_module, lowered_node.name)
        input_specs = lowered_module.original_module.graph_signature.input_specs
        self.assertEqual(
            [(spec.arg.name, spec.kind.name) for spec in input_specs],
            [
                ("_lifted_tensor_constant1", "BUFFER"),
                ("x", "USER_INPUT"),
                ("b_my_state", "USER_INPUT"),
            ],
        )
