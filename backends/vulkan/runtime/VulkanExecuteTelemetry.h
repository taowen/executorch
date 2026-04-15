/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch {
namespace backends {
namespace vulkan {

struct VulkanExecuteTelemetry final {
  uint64_t generation = 0;
  double copy_inputs_ms = 0.0;
  double resize_ms = 0.0;
  double compute_graph_execute_ms = 0.0;
  double copy_outputs_ms = 0.0;
  double total_backend_ms = 0.0;
  double gpu_shader_total_ms = 0.0;
  uint64_t gpu_shader_dispatch_count = 0;
};

bool get_last_vulkan_execute_telemetry(VulkanExecuteTelemetry* out);
void reset_last_vulkan_execute_telemetry();

} // namespace vulkan
} // namespace backends
} // namespace executorch

