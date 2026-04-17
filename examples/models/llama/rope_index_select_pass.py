# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export import ExportedProgram
from torch.fx import GraphModule, Node


class RopeIndexTensorToIndexSelectPass(ExportPass):
    """
    Rewrite RoPE table gathers from aten.index.Tensor to aten.index_select.default.

    Qwen3.5 RoPE lookup commonly indexes 2D frequency tables with a single 1D
    position tensor. Vulkan handles the equivalent index_select path more
    reliably than the generic aten.index.Tensor lowering.
    """

    _ROPE_FREQ_TABLE_SUFFIXES = ("rope_freqs_cos", "rope_freqs_sin")

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        rewritten = False

        for node in list(graph.nodes):
            if not self._is_rope_index_tensor(node):
                continue
            self_arg, indices = node.args
            if not isinstance(indices, (list, tuple)) or len(indices) != 1:
                continue
            index_tensor = indices[0]
            with graph.inserting_before(node):
                replacement = graph.call_function(
                    self._index_select_target(node),
                    args=(self_arg, 0, index_tensor),
                )
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            rewritten = True

        if rewritten:
            graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, rewritten)

    def call_exported_program(
        self, exported_program: ExportedProgram
    ) -> ExportedProgram:
        return super().call_exported_program(exported_program)

    def _is_rope_index_tensor(self, node: Node) -> bool:
        if node.op != "call_function":
            return False
        if node.target not in (
            torch.ops.aten.index.Tensor,
            exir_ops.edge.aten.index.Tensor,
        ):
            return False
        if len(node.args) < 2:
            return False
        self_arg = node.args[0]
        if not isinstance(self_arg, Node):
            return False
        return any(
            self_arg.name.endswith(suffix) for suffix in self._ROPE_FREQ_TABLE_SUFFIXES
        )

    def _index_select_target(self, node: Node):
        if node.target == exir_ops.edge.aten.index.Tensor:
            return exir_ops.edge.aten.index_select.default
        return torch.ops.aten.index_select.default
