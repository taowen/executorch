# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops


_CAST_OPS = {
    exir_ops.edge.aten._to_copy.default,
    exir_ops.edge.aten.to.dtype,
}


def _skip_casts(node: torch.fx.Node) -> torch.fx.Node:
    while node.target in _CAST_OPS:
        arg0 = node.args[0] if node.args else None
        if not isinstance(arg0, torch.fx.Node):
            break
        node = arg0
    return node


def _extract_sigmoid_input(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    inner = _skip_casts(node)
    if inner.target != exir_ops.edge.aten.sigmoid.default:
        return None
    if len(inner.args) < 1 or not isinstance(inner.args[0], torch.fx.Node):
        return None
    return _skip_casts(inner.args[0])


def _match_silu_node(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    if node.target != exir_ops.edge.aten.mul.Tensor:
        return None
    if len(node.args) < 2:
        return None
    a, b = node.args[0], node.args[1]
    if not isinstance(a, torch.fx.Node) or not isinstance(b, torch.fx.Node):
        return None

    for sigmoid_candidate, x_candidate in ((a, b), (b, a)):
        x = _skip_casts(x_candidate)
        sig_in = _extract_sigmoid_input(sigmoid_candidate)
        if sig_in is not None and sig_in == x:
            return x
    return None


@register_pattern_detector("silu_mul")
def find_silu_mul_pattern(node: torch.fx.Node) -> Optional[PatternMatch]:
    if node.target != exir_ops.edge.aten.mul.Tensor:
        return None
    if len(node.args) < 2:
        return None
    a, b = node.args[0], node.args[1]
    if not isinstance(a, torch.fx.Node) or not isinstance(b, torch.fx.Node):
        return None

    for silu_candidate, other in ((a, b), (b, a)):
        silu_input = _match_silu_node(_skip_casts(silu_candidate))
        if silu_input is None:
            continue
        if isinstance(other, torch.fx.Node) and other == silu_input:
            continue
        return PatternMatch(
            input_nodes=[silu_input, other],
            output_nodes=[node],
            all_nodes=[node, silu_candidate],
            anchor_node=node,
        )

    return None


@register_pattern_replacement("silu_mul")
def replace_silu_mul_with_fused_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: PatternMatch,
):
    del ep
    x, other = match.input_nodes
    anchor = match.anchor_node
    if anchor is None:
        return

    with graph_module.graph.inserting_before(anchor):
        fused = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.silu_mul.default,
            args=(x, other),
        )

    fused.meta["val"] = anchor.meta["val"]
    anchor.replace_all_uses_with(fused)
