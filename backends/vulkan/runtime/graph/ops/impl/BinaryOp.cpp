/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/View.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_binary_op_args(
    ComputeGraph& graph,
    const ValueRef self,
    const ValueRef other,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(self) == graph.packed_dim_of(other));
  VK_CHECK_COND(graph.packed_dim_of(self) == graph.packed_dim_of(out));

  const std::vector<int64_t> self_sizes = graph.sizes_of(self);
  const std::vector<int64_t> other_sizes = graph.sizes_of(other);
  const std::vector<int64_t> out_sizes = graph.sizes_of(out);

  std::vector<int64_t> broadcasted_sizes =
      calculate_broadcasted_output_size(self_sizes, other_sizes);
  VK_CHECK_COND(out_sizes == broadcasted_sizes);
}

void resize_binary_op_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  const auto& input_refs = args.at(1).refs;
  if (input_refs.empty()) {
    VK_THROW("binary op resize expects at least one input tensor ref");
  }

  // Scalar binary kernels only carry a single tensor input in the read arg
  // group. The scalar does not affect output shape, so the output shape is the
  // tensor input shape.
  if (input_refs.size() == 1) {
    graph->virtual_resize(out, graph->sizes_of(input_refs.at(0)));
    return;
  }

  // TODO(T183442143): Verify tensors are broadcastable.
  const ValueRef self = input_refs.at(0);
  const ValueRef other = input_refs.at(1);

  const std::vector<int64_t> self_sizes = graph->sizes_of(self);
  const std::vector<int64_t> other_sizes = graph->sizes_of(other);
  const std::vector<int64_t> new_out_sizes =
      calculate_broadcasted_output_size(self_sizes, other_sizes);

  graph->virtual_resize(out, new_out_sizes);
}

void add_binary_op_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef alpha,
    const ValueRef out,
    const std::string& op_name) {
  auto emit_binary_scalar_node = [&](const ValueRef tensor_in,
                                     const ValueRef scalar_in,
                                     const vkapi::ScalarType tensor_dtype,
                                     float scalar_scale = 1.0f) {
    ValueRef arg = prepack_standard_like(graph, tensor_in, out, true);
    float scalar_val = graph.extract_scalar<float>(scalar_in) * scalar_scale;

    std::string kernel_name = op_name + "_scalar";
    kernel_name.reserve(kShaderNameReserve);
    add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
    add_dtype_suffix(kernel_name, tensor_dtype);

    vkapi::ParamsBindList ubos = {graph.meta_ubo(out), graph.meta_ubo(arg)};
    graph.execute_nodes().emplace_back(new DynamicDispatchNode(
        graph,
        VK_KERNEL_FROM_STR(kernel_name),
        default_pick_global_wg_size,
        default_pick_local_wg_size,
        {{out, vkapi::kWrite}, {arg, vkapi::kRead}},
        ubos,
        {{PushConstantDataInfo(&scalar_val, sizeof(float))}},
        {},
        {},
        resize_binary_op_node));
  };

  const bool lhs_is_tensor_like =
      graph.val_is_tensor(in1) || graph.val_is_tref(in1);
  const bool rhs_is_tensor_like =
      graph.val_is_tensor(in2) || graph.val_is_tref(in2);
  auto materialize_runtime_rank0_buffer_tensor = [&](const ValueRef ref) {
    if (!graph.val_is_tensor(ref) || !graph.is_buffer_storage(ref) ||
        !graph.sizes_of(ref).empty()) {
      return ref;
    }

    const ValueRef cloned = graph.add_tensor_like(
        ref, graph.storage_type_of(ref), graph.estimate_memory_layout_of(ref));
    add_view_copy_node(graph, ref, cloned, {}, nullptr);
    return cloned;
  };

  if (!rhs_is_tensor_like && lhs_is_tensor_like) {
    VK_CHECK_COND(
        op_name == "add" || op_name == "sub" || op_name == "mul" ||
        op_name == "div" || op_name == "floor_divide");
    float scalar_scale = 1.0f;
    if ((op_name == "add" || op_name == "sub") && is_valid(alpha) &&
        !graph.val_is_string(alpha)) {
      scalar_scale = graph.extract_scalar<float>(alpha);
    }
    emit_binary_scalar_node(in1, in2, graph.dtype_of(in1), scalar_scale);
    return;
  }

  if (!lhs_is_tensor_like && rhs_is_tensor_like) {
    VK_CHECK_COND(op_name == "add" || op_name == "mul");
    VK_CHECK_COND(
        !is_valid(alpha) || graph.val_is_string(alpha) ||
        graph.extract_scalar<float>(alpha) == 1.0f);
    emit_binary_scalar_node(in2, in1, graph.dtype_of(in2));
    return;
  }

  const ValueRef lhs_input = materialize_runtime_rank0_buffer_tensor(in1);
  const ValueRef rhs_input = materialize_runtime_rank0_buffer_tensor(in2);

  ValueRef arg1 = prepack_standard_like(graph, lhs_input, out, true);
  ValueRef arg2 = prepack_standard_like(graph, rhs_input, out, true);

  check_binary_op_args(graph, arg1, arg2, out);

  float alpha_val = 1.0f;
  // String is checked since floor_div passes in an unused string argument in
  // place of alpha
  if (is_valid(alpha) && !graph.val_is_string(alpha)) {
    alpha_val = graph.extract_scalar<float>(alpha);
  }

  std::string kernel_name("binary_");
  kernel_name.reserve(kShaderNameReserve);
  kernel_name += op_name;
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(in1));

  vkapi::ParamsBindList ubos = {
      graph.meta_ubo(out), graph.meta_ubo(arg1), graph.meta_ubo(arg2)};

  // Detect packed-dim broadcasting for texture path
  int32_t in_broadcast_packed = 0;
  int32_t other_broadcast_packed = 0;
  if (!graph.is_buffer_storage(out)) {
    in_broadcast_packed = is_packed_dim_broadcasted(graph, out, arg1) ? 1 : 0;
    other_broadcast_packed =
        is_packed_dim_broadcasted(graph, out, arg2) ? 1 : 0;
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{arg1, arg2}, vkapi::kRead}},
      // Shader params buffers
      ubos,
      // Push Constants
      {{PushConstantDataInfo(&alpha_val, sizeof(float))}},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(arg1),
       graph.hashed_layout_of(arg2),
       in_broadcast_packed,
       other_broadcast_packed},
      // Resize Args
      {},
      // Resizing Logic
      resize_binary_op_node));
}

#define DEFINE_BINARY_OP_WITH_ALPHA_FN(op_name)                          \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_binary_op_node(                                           \
        graph, args[0], args[1], args[2], args[3], #op_name);            \
  }

#define DEFINE_BINARY_OP_FN(op_name)                                     \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_binary_op_node(                                           \
        graph, args[0], args[1], kDummyValueRef, args[2], #op_name);     \
  }

DEFINE_BINARY_OP_WITH_ALPHA_FN(add);
DEFINE_BINARY_OP_WITH_ALPHA_FN(sub);

// Floor div does not have an alpha, but a string argument (which is unused) is
// passed in at the same location as the alpha argument in other op.
DEFINE_BINARY_OP_WITH_ALPHA_FN(floor_divide);

DEFINE_BINARY_OP_FN(mul);
DEFINE_BINARY_OP_FN(silu_mul);
DEFINE_BINARY_OP_FN(div);
DEFINE_BINARY_OP_FN(pow);
DEFINE_BINARY_OP_FN(minimum);
DEFINE_BINARY_OP_FN(eq);
DEFINE_BINARY_OP_FN(lt);
DEFINE_BINARY_OP_FN(le);
DEFINE_BINARY_OP_FN(gt);
DEFINE_BINARY_OP_FN(ge);
DEFINE_BINARY_OP_FN(bitwise_and);

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.add.Tensor, add);
  VK_REGISTER_OP(aten.sub.Tensor, sub);
  VK_REGISTER_OP(aten.mul.Tensor, mul);
  VK_REGISTER_OP(et_vk.silu_mul.default, silu_mul);
  VK_REGISTER_OP(aten.div.Tensor, div);
  VK_REGISTER_OP(aten.div.Tensor_mode, floor_divide);
  VK_REGISTER_OP(aten.pow.Tensor_Tensor, pow);
  VK_REGISTER_OP(aten.minimum.default, minimum);
  VK_REGISTER_OP(aten.eq.Tensor, eq);
  VK_REGISTER_OP(aten.lt.Tensor, lt);
  VK_REGISTER_OP(aten.le.Tensor, le);
  VK_REGISTER_OP(aten.gt.Tensor, gt);
  VK_REGISTER_OP(aten.ge.Tensor, ge);
  VK_REGISTER_OP(aten.bitwise_and.Tensor, bitwise_and);
  VK_REGISTER_OP(aten.logical_and.default, bitwise_and);
}

} // namespace vkcompute
