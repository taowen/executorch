/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/ResolveLayouts.h>
#include <executorch/backends/vulkan/runtime/VulkanDelegateHeader.h>
#include "VulkanExecuteTelemetry.h"
#include <executorch/backends/vulkan/serialization/schema_generated.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/api/ShaderRegistry.h>
#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#ifdef ET_EVENT_TRACER_ENABLED
#include <executorch/backends/vulkan/runtime/graph/Logging.h>
#include <executorch/runtime/core/event_tracer_hooks_delegate.h>
#endif // ET_EVENT_TRACER_ENABLED
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/profiler.h>

#include <cstdio>
#include <cstdlib> /* strtol */
#include <cstring>
#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace vulkan {

namespace {
using SteadyClock = std::chrono::steady_clock;

thread_local bool t_has_last_execute_telemetry = false;
thread_local VulkanExecuteTelemetry t_last_execute_telemetry{};
thread_local uint64_t t_last_execute_generation = 0;

const char* get_vulkan_init_trace_path() {
  static const char* kPath = std::getenv("ET_VULKAN_INIT_TRACE_PATH");
  if (kPath == nullptr || kPath[0] == '\0') {
    return nullptr;
  }
  return kPath;
}

bool vulkan_init_trace_enabled() {
  return get_vulkan_init_trace_path() != nullptr;
}

bool vulkan_execute_trace_enabled() {
  static const bool kEnabled = []() {
    const char* env = std::getenv("ET_VULKAN_EXEC_TRACE");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
  }();
  return kEnabled;
}

bool vulkan_execute_trace_values_enabled() {
  static const bool kEnabled = []() {
    const char* env = std::getenv("ET_VULKAN_EXEC_TRACE_VALUES");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
  }();
  return kEnabled;
}

template <typename T>
void append_sample_values(
    std::string& out,
    const executorch::aten::Tensor& tensor,
    const size_t sample_count) {
  const T* data = tensor.const_data_ptr<T>();
  out += " values=[";
  for (size_t i = 0; i < sample_count; ++i) {
    if (i > 0) {
      out += ",";
    }
    out += std::to_string(static_cast<double>(data[i]));
  }
  out += "]";
}

void append_half_sample_values(
    std::string& out,
    const executorch::aten::Tensor& tensor,
    const size_t sample_count) {
  const executorch::aten::Half* data =
      tensor.const_data_ptr<executorch::aten::Half>();
  out += " values=[";
  for (size_t i = 0; i < sample_count; ++i) {
    if (i > 0) {
      out += ",";
    }
    out += std::to_string(static_cast<double>(data[i]));
  }
  out += "]";
}

void trace_vulkan_execute(
    const char* stage,
    const size_t input_idx,
    const executorch::aten::Tensor* tensor = nullptr) {
  if (!vulkan_execute_trace_enabled()) {
    return;
  }
  if (tensor == nullptr) {
    std::fprintf(
        stderr,
        "[ET_VULKAN_EXEC] stage=%s input=%zu tensor=null\n",
        stage,
        input_idx);
    std::fflush(stderr);
    return;
  }

  std::fprintf(
      stderr,
      "[ET_VULKAN_EXEC] stage=%s input=%zu dim=%zu numel=%zu dtype=%d\n",
      stage,
      input_idx,
      static_cast<size_t>(tensor->dim()),
      static_cast<size_t>(tensor->numel()),
      static_cast<int>(tensor->scalar_type()));
  std::fflush(stderr);

  if (!vulkan_execute_trace_values_enabled() || tensor->numel() == 0) {
    return;
  }

  const size_t sample_count =
      std::min(static_cast<size_t>(tensor->numel()), static_cast<size_t>(8));
  std::string detail = "[ET_VULKAN_EXEC_VALUES] stage=";
  detail += stage;
  detail += " input=";
  detail += std::to_string(input_idx);
  detail += " dtype=";
  detail += std::to_string(static_cast<int>(tensor->scalar_type()));
  detail += " numel=";
  detail += std::to_string(static_cast<size_t>(tensor->numel()));

  switch (tensor->scalar_type()) {
    case executorch::aten::ScalarType::Float:
      append_sample_values<float>(detail, *tensor, sample_count);
      break;
    case executorch::aten::ScalarType::Half:
      append_half_sample_values(detail, *tensor, sample_count);
      break;
    case executorch::aten::ScalarType::Int:
      append_sample_values<int32_t>(detail, *tensor, sample_count);
      break;
    case executorch::aten::ScalarType::Long:
      append_sample_values<int64_t>(detail, *tensor, sample_count);
      break;
    case executorch::aten::ScalarType::Byte:
      append_sample_values<uint8_t>(detail, *tensor, sample_count);
      break;
    case executorch::aten::ScalarType::Char:
      append_sample_values<int8_t>(detail, *tensor, sample_count);
      break;
    case executorch::aten::ScalarType::Bool:
      append_sample_values<bool>(detail, *tensor, sample_count);
      break;
    default:
      detail += " values=<unsupported_dtype>";
      break;
  }

  std::fprintf(stderr, "%s\n", detail.c_str());
  std::fflush(stderr);
}

std::string json_escape(const char* in) {
  if (in == nullptr) {
    return "";
  }
  std::string out;
  out.reserve(std::strlen(in) + 8);
  for (const unsigned char c : std::string(in)) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
        break;
    }
  }
  return out;
}

void append_vulkan_init_trace_line(const std::string& line) {
  const char* path = get_vulkan_init_trace_path();
  if (path == nullptr) {
    return;
  }
  FILE* f = std::fopen(path, "a");
  if (f == nullptr) {
    return;
  }
  std::fwrite(line.data(), 1, line.size(), f);
  std::fwrite("\n", 1, 1, f);
  std::fclose(f);
}

void trace_vulkan_init_event(
    const char* event,
    const char* method_name,
    const char* message) {
  if (!vulkan_init_trace_enabled()) {
    return;
  }
  std::string line = "{";
  line += "\"source\":\"vulkan_runtime\"";
  line += ",\"event\":\"";
  line += json_escape(event);
  line += "\"";
  line += ",\"method\":\"";
  line += json_escape(method_name == nullptr ? "" : method_name);
  line += "\"";
  if (message != nullptr) {
    line += ",\"message\":\"";
    line += json_escape(message);
    line += "\"";
  }
  line += "}";
  append_vulkan_init_trace_line(line);
}
} // namespace

bool get_last_vulkan_execute_telemetry(VulkanExecuteTelemetry* out) {
  if (out == nullptr || !t_has_last_execute_telemetry) {
    return false;
  }
  *out = t_last_execute_telemetry;
  return true;
}

void reset_last_vulkan_execute_telemetry() {
  t_has_last_execute_telemetry = false;
  t_last_execute_telemetry = VulkanExecuteTelemetry{};
}

namespace {

using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::kTensorDimensionLimit;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Result;
using executorch::runtime::Span;

using namespace vkcompute;

// Flatbuffer types
using VkGraphPtr = const vkgraph::VkGraph*;
using OpCallPtr = const vkgraph::OperatorCall*;
using VkValuePtr = const vkgraph::VkValue*;
using VkTensorPtr = const vkgraph::VkTensor*;
using VkBytesPtr = const vkgraph::VkBytes*;

// Flatbuffer vector types
using VkValuesVector =
    const flatbuffers::Vector<flatbuffers::Offset<vkgraph::VkValue>>*;
using BytesVector =
    const flatbuffers::Vector<flatbuffers::Offset<vkgraph::VkBytes>>*;
using UIntVector = const flatbuffers::Vector<uint32_t>*;

vkapi::ScalarType get_scalar_type(const vkgraph::VkDataType& vk_datatype) {
  switch (vk_datatype) {
    case vkgraph::VkDataType::BOOL:
      return vkapi::kBool;
    case vkgraph::VkDataType::UINT8:
      return vkapi::kByte;
    case vkgraph::VkDataType::INT8:
      return vkapi::kChar;
    case vkgraph::VkDataType::INT32:
      return vkapi::kInt;
    case vkgraph::VkDataType::INT64:
      return vkapi::kLong;
    case vkgraph::VkDataType::FLOAT16:
      return vkapi::kHalf;
    case vkgraph::VkDataType::FLOAT32:
      return vkapi::kFloat;
    case vkgraph::VkDataType::FLOAT64:
      return vkapi::kDouble;
    default:
      VK_THROW("Invalid VkDataType type encountered!");
  }
}

vkapi::ScalarType equivalent_scalar_type(
    const executorch::runtime::etensor::ScalarType& et_datatype) {
  switch (et_datatype) {
    case executorch::runtime::etensor::ScalarType::Byte:
      return vkapi::kByte;
    case executorch::runtime::etensor::ScalarType::Char:
      return vkapi::kChar;
    case executorch::runtime::etensor::ScalarType::Int:
      return vkapi::kInt;
    case executorch::runtime::etensor::ScalarType::Long:
      return vkapi::kLong;
    case executorch::runtime::etensor::ScalarType::Half:
      return vkapi::kHalf;
    case executorch::runtime::etensor::ScalarType::Float:
      return vkapi::kFloat;
    case executorch::runtime::etensor::ScalarType::Double:
      return vkapi::kDouble;
    case executorch::runtime::etensor::ScalarType::Bool:
      return vkapi::kBool;
    default:
      VK_THROW("Invalid etensor::ScalarType encountered!");
  }
}

utils::StorageType get_storage_type(
    const vkgraph::VkStorageType& vk_storage_type) {
  switch (vk_storage_type) {
    case vkgraph::VkStorageType::BUFFER:
      return utils::kBuffer;
    case vkgraph::VkStorageType::TEXTURE_3D:
      return utils::kTexture3D;
    case vkgraph::VkStorageType::TEXTURE_2D:
      return utils::kTexture2D;
    default:
      break;
  }
  VK_THROW("Invalid storage type encountered!");
}

utils::GPUMemoryLayout get_memory_layout(
    const vkgraph::VkMemoryLayout& vk_memory_layout) {
  switch (vk_memory_layout) {
    case vkgraph::VkMemoryLayout::TENSOR_WIDTH_PACKED:
      return utils::kWidthPacked;
    case vkgraph::VkMemoryLayout::TENSOR_HEIGHT_PACKED:
      return utils::kHeightPacked;
    case vkgraph::VkMemoryLayout::TENSOR_CHANNELS_PACKED:
      return utils::kChannelsPacked;
    case vkgraph::VkMemoryLayout::PACKED_INT8_4W4C:
      return utils::kPackedInt8_4W4C;
    case vkgraph::VkMemoryLayout::PACKED_INT8_4H4W:
      return utils::kPackedInt8_4H4W;
    case vkgraph::VkMemoryLayout::PACKED_INT8_4W:
      return utils::kPackedInt8_4W;
    case vkgraph::VkMemoryLayout::PACKED_INT8_4C:
      return utils::kPackedInt8_4C;
    case vkgraph::VkMemoryLayout::PACKED_INT8_4C1W:
      return utils::kPackedInt8_4C1W;
    case vkgraph::VkMemoryLayout::PACKED_INT8_CONV2D:
      // Fallback for unresolved dynamic layout
      return utils::kPackedInt8_4C1W;
    default:
      break;
  }
  VK_THROW("Invalid memory layout encountered!");
}

GraphConfig get_graph_config(ArrayRef<CompileSpec>& compile_specs) {
  GraphConfig config = GraphConfig();

  for (const CompileSpec& spec : compile_specs) {
    const uint8_t* value_data = (const uint8_t*)spec.value.buffer;
    const size_t value_size = spec.value.nbytes;
    if (strcmp(spec.key, "storage_type_override") == 0) {
      ET_CHECK_MSG(value_size == sizeof(int32_t), "Unexpected value size!");
      int value_as_int = static_cast<int>(getUInt32LE(value_data));
      utils::StorageType storage_type =
          static_cast<utils::StorageType>(value_as_int);

      config.set_storage_type_override(storage_type);
    }
    if (strcmp(spec.key, "memory_layout_override") == 0) {
      ET_CHECK_MSG(value_size == sizeof(uint32_t), "Unexpected value size!");
      uint32_t value_as_int = getUInt32LE(value_data);
      utils::GPUMemoryLayout memory_layout =
          static_cast<utils::GPUMemoryLayout>(value_as_int);

      config.set_memory_layout_override(memory_layout);
    }
    if (strcmp(spec.key, "require_dynamic_shapes") == 0) {
      ET_CHECK_MSG(value_size == sizeof(uint8_t), "Unexpected value size!");
      bool value = getBool(value_data);

      if (value) {
        config.expect_dynamic_shapes = true;
      }
    }
    if (strcmp(spec.key, "warmup_execute_after_compile") == 0) {
      ET_CHECK_MSG(value_size == sizeof(uint8_t), "Unexpected value size!");
      bool value = getBool(value_data);

      config.warmup_execute_after_compile = value;
    }
    if (strcmp(spec.key, "enable_querypool") == 0) {
      ET_CHECK_MSG(value_size == sizeof(uint8_t), "Unexpected value size!");
      bool value = getBool(value_data);
      config.enable_querypool = value;
    }
  }
#ifdef ET_EVENT_TRACER_ENABLED
  config.enable_querypool = true;
#endif // ET_EVENT_TRACER_ENABLED
  return config;
}

std::string get_compile_spec_str(const CompileSpec& spec) {
  const char* value_data = reinterpret_cast<const char*>(spec.value.buffer);
  size_t value_size = spec.value.nbytes;
  while (value_size > 0 && value_data[value_size - 1] == '\0') {
    --value_size;
  }
  return std::string(value_data, value_size);
}

std::string get_shader_bundle_path(ArrayRef<CompileSpec>& compile_specs) {
  for (const CompileSpec& spec : compile_specs) {
    if (strcmp(spec.key, "shader_bundle_path") == 0) {
      return get_compile_spec_str(spec);
    }
  }
  return std::string();
}

class GraphBuilder {
  ComputeGraph* compute_graph_;
  VkGraphPtr flatbuffer_;
  const uint8_t* constant_data_;
  const NamedDataMap* named_data_map_;
  std::vector<FreeableBuffer> loaded_buffers_from_map_;

  std::vector<ValueRef> ref_mapping_;
  std::unordered_map<uint32_t, vkgraph::VkMemoryLayout>
      memory_layout_overrides_;

 public:
  explicit GraphBuilder(
      ComputeGraph* compute_graph,
      VkGraphPtr flatbuffer,
      const uint8_t* constant_data,
      const NamedDataMap* named_data_map)
      : compute_graph_(compute_graph),
        flatbuffer_(flatbuffer),
        constant_data_(constant_data),
        named_data_map_(named_data_map),
        loaded_buffers_from_map_(),
        ref_mapping_(),
        memory_layout_overrides_() {}

  void resolve_layouts() {
    resolve_memory_layouts(
        flatbuffer_, compute_graph_, memory_layout_overrides_);
  }

  void resize(uint32_t size) {
    ref_mapping_.resize(size, INT32_MAX);
  }

  bool fb_id_exists(const uint32_t fb_id) {
    return fb_id < ref_mapping_.size() && ref_mapping_[fb_id] != INT32_MAX;
  }

  ValueRef get_fb_id_valueref(const uint32_t fb_id) {
    ET_CHECK_MSG(
        fb_id_exists(fb_id),
        "Trying to extract a value that hasn't yet been added to the graph.");

    return ref_mapping_[fb_id];
  }

  utils::GPUMemoryLayout get_resolved_memory_layout(
      const uint32_t fb_id,
      VkTensorPtr tensor_fb,
      const std::vector<int64_t>& dims_vector) {
    auto it = memory_layout_overrides_.find(fb_id);
    if (it != memory_layout_overrides_.end()) {
      return get_memory_layout(it->second);
    }

    if (tensor_fb->memory_layout() == vkgraph::VkMemoryLayout::DEFAULT_LAYOUT) {
      return compute_graph_->suggested_memory_layout(dims_vector);
    }
    return get_memory_layout(tensor_fb->memory_layout());
  }

  void add_tensor_to_graph(const uint32_t fb_id, VkTensorPtr tensor_fb) {
    const vkapi::ScalarType& dtype = get_scalar_type(tensor_fb->datatype());
    utils::StorageType storage_type =
        tensor_fb->storage_type() == vkgraph::VkStorageType::DEFAULT_STORAGE
        ? compute_graph_->suggested_storage_type()
        : get_storage_type(tensor_fb->storage_type());

    UIntVector dims_fb = tensor_fb->dims();
    const std::vector<int64_t> dims_vector(dims_fb->cbegin(), dims_fb->cend());

    // Texture metadata UBOS in Vulkan runtime are fixed to 4D. If a serialized
    // graph marks a >4D tensor as texture-backed, force buffer storage to avoid
    // invalid texture metadata access during staging/setup.
    if (storage_type != utils::kBuffer && dims_vector.size() > 4) {
      std::string msg = "fb_id=" + std::to_string(fb_id) + ",dims=[";
      for (size_t i = 0; i < dims_vector.size(); ++i) {
        if (i > 0) {
          msg += ",";
        }
        msg += std::to_string(dims_vector[i]);
      }
      msg += "],fallback_storage=BUFFER";
      trace_vulkan_init_event(
          "build_graph_storage_fallback", "GraphBuilder::add_tensor_to_graph", msg.c_str());
      storage_type = utils::kBuffer;
    }

    utils::GPUMemoryLayout memory_layout =
        get_resolved_memory_layout(fb_id, tensor_fb, dims_vector);

    ValueRef ref;
    if (tensor_fb->constant_id() >= 0) {
      VkBytesPtr constant_bytes =
          flatbuffer_->constants()->Get(tensor_fb->constant_id());

      if (constant_bytes->named_key() != nullptr &&
          constant_bytes->offset() == UINT64_MAX &&
          named_data_map_ != nullptr) {
        const std::string& data_name = constant_bytes->named_key()->str();
        Result<FreeableBuffer> buffer =
            named_data_map_->get_data(data_name.c_str());

        VK_CHECK_COND(
            buffer.ok(),
            "Failed to get constant data for key %s from named_data_map. Error code: %u",
            data_name.c_str(),
            static_cast<uint32_t>(buffer.error()));
        ref = compute_graph_->add_tensorref(
            dims_vector, dtype, std::move(buffer.get()));
      } else {
        const uint8_t* tensor_data = constant_data_ + constant_bytes->offset();
        ref = compute_graph_->add_tensorref(dims_vector, dtype, tensor_data);
      }
    } else {
      ref = compute_graph_->add_tensor(
          dims_vector,
          dtype,
          storage_type,
          memory_layout,
          tensor_fb->mem_obj_id());
    }

    ref_mapping_[fb_id] = ref;
  }

  void add_none_to_graph(const uint32_t fb_id) {
    ValueRef ref = compute_graph_->add_none();
    ref_mapping_[fb_id] = ref;
  }

  template <typename T>
  typename std::enable_if<is_valid_scalar_type<T>::value, void>::type
  add_scalar_to_graph(const uint32_t fb_id, T value) {
    ValueRef ref = compute_graph_->add_scalar(value);
    ref_mapping_[fb_id] = ref;
  }

  template <typename T>
  typename std::enable_if<is_valid_scalar_type<T>::value, void>::type
  add_scalar_list_to_graph(const uint32_t fb_id, std::vector<T>&& value) {
    ValueRef ref = compute_graph_->add_scalar_list(std::move(value));
    ref_mapping_[fb_id] = ref;
  }

  void add_value_list_to_graph(
      const uint32_t fb_id,
      std::vector<ValueRef>&& value) {
    ValueRef ref = compute_graph_->add_value_list(std::move(value));
    ref_mapping_[fb_id] = ref;
  }

  void add_string_to_graph(const uint32_t fb_id, VkValuePtr value) {
    const auto fb_str = value->value_as_String()->string_val();
    std::string string(fb_str->cbegin(), fb_str->cend());
    ValueRef ref = compute_graph_->add_string(std::move(string));
    ref_mapping_[fb_id] = ref;
  }

  void add_symint_to_graph(const uint32_t fb_id, VkValuePtr value) {
    const int32_t fb_symint = value->value_as_SymInt()->value();
    ValueRef ref = compute_graph_->add_symint(fb_symint);
    ref_mapping_[fb_id] = ref;
  }

  void add_value_to_graph(const uint32_t fb_id, VkValuePtr value) {
    ET_CHECK_MSG(
        !fb_id_exists(fb_id),
        "Trying to add a value that has already been added to the graph.");

    switch (value->value_type()) {
      case vkgraph::GraphTypes::Null:
        add_none_to_graph(fb_id);
        break;
      case vkgraph::GraphTypes::Int:
        add_scalar_to_graph(fb_id, value->value_as_Int()->int_val());
        break;
      case vkgraph::GraphTypes::Double:
        add_scalar_to_graph(fb_id, value->value_as_Double()->double_val());
        break;
      case vkgraph::GraphTypes::Bool:
        add_scalar_to_graph(fb_id, value->value_as_Bool()->bool_val());
        break;
      case vkgraph::GraphTypes::VkTensor:
        add_tensor_to_graph(fb_id, value->value_as_VkTensor());
        break;
      case vkgraph::GraphTypes::IntList:
        add_scalar_list_to_graph(
            fb_id,
            std::vector<int64_t>(
                value->value_as_IntList()->items()->cbegin(),
                value->value_as_IntList()->items()->cend()));
        break;
      case vkgraph::GraphTypes::DoubleList:
        add_scalar_list_to_graph(
            fb_id,
            std::vector<double>(
                value->value_as_DoubleList()->items()->cbegin(),
                value->value_as_DoubleList()->items()->cend()));
        break;
      case vkgraph::GraphTypes::BoolList:
        add_scalar_list_to_graph(
            fb_id,
            std::vector<bool>(
                value->value_as_BoolList()->items()->cbegin(),
                value->value_as_BoolList()->items()->cend()));
        break;
      case vkgraph::GraphTypes::ValueList:
        add_value_list_to_graph(
            fb_id,
            std::vector<ValueRef>(
                value->value_as_ValueList()->items()->cbegin(),
                value->value_as_ValueList()->items()->cend()));
        break;
      case vkgraph::GraphTypes::String:
        add_string_to_graph(fb_id, value);
        break;
      case vkgraph::GraphTypes::SymInt:
        add_symint_to_graph(fb_id, value);
        break;
      default:
        ET_CHECK_MSG(false, "Unsupported value type.");
    }
  }

  vkapi::ScalarType get_staging_scalar_type_of(const uint32_t fb_id) {
    VkTensorPtr tensor_fb =
        flatbuffer_->values()->Get(fb_id)->value_as_VkTensor();
    if (tensor_fb->staging_datatype() == vkgraph::VkDataType::UNSET) {
      return get_scalar_type(tensor_fb->datatype());
    }
    return get_scalar_type(tensor_fb->staging_datatype());
  }

  void build_graph() {
    trace_vulkan_init_event("build_graph_begin", "GraphBuilder::build_graph", nullptr);

    // Resize the mapping to the number of values in the flatbuffer
    resize(flatbuffer_->values()->size());
    trace_vulkan_init_event(
        "build_graph_after_resize",
        "GraphBuilder::build_graph",
        nullptr);

    // First, add all values to the graph
    for (uint32_t fb_id = 0; fb_id < flatbuffer_->values()->size(); ++fb_id) {
      VkValuePtr value = flatbuffer_->values()->Get(fb_id);
      add_value_to_graph(fb_id, value);
    }
    trace_vulkan_init_event(
        "build_graph_after_values",
        "GraphBuilder::build_graph",
        nullptr);

    // Parse the inputs, which will be tensors most of the time but can also be
    // symints and tensorrefs (which will be the case if the original graph had)
    // mutable buffers.
    uint32_t input_pos = 0;
    for (const uint32_t fb_id : *flatbuffer_->input_ids()) {
      const ValueRef ref = get_fb_id_valueref(fb_id);
      if (compute_graph_->val_is_tensor(ref)) {
        compute_graph_->set_input_tensor(
            ref, get_staging_scalar_type_of(fb_id));
        const std::vector<int64_t> sizes = compute_graph_->sizes_of(ref);
        std::string input_msg =
            "input_pos=" + std::to_string(input_pos) + ",fb_id=" +
            std::to_string(fb_id) + ",ref=" + std::to_string(ref) +
            ",kind=tensor,sizes=[";
        for (size_t i = 0; i < sizes.size(); ++i) {
          if (i > 0) {
            input_msg += ",";
          }
          input_msg += std::to_string(sizes[i]);
        }
        input_msg += "]";
        trace_vulkan_init_event(
            "build_graph_input",
            "GraphBuilder::build_graph",
            input_msg.c_str());
      } else {
        compute_graph_->set_val_as_input(ref);
        std::string input_msg =
            "input_pos=" + std::to_string(input_pos) + ",fb_id=" +
            std::to_string(fb_id) + ",ref=" + std::to_string(ref) +
            ",kind=non_tensor";
        trace_vulkan_init_event(
            "build_graph_input",
            "GraphBuilder::build_graph",
            input_msg.c_str());
      }
      input_pos++;
    }
    trace_vulkan_init_event(
        "build_graph_after_inputs",
        "GraphBuilder::build_graph",
        nullptr);

    // Parse the operators
    uint32_t op_index = 0;
    for (OpCallPtr op_call : *(flatbuffer_->chain())) {
      std::string op_name = op_call->name()->str();
      std::string op_begin_msg =
          "op_index=" + std::to_string(op_index) + ",name=" + op_name +
          ",argc=" + std::to_string(op_call->args()->size());
      trace_vulkan_init_event(
          "build_graph_op_begin",
          "GraphBuilder::build_graph",
          op_begin_msg.c_str());
      ET_CHECK_MSG(VK_HAS_OP(op_name), "Missing operator: %s", op_name.c_str());

      std::vector<ValueRef> args;
      args.reserve(op_call->args()->size());
      std::string arg_map_msg =
          "op_index=" + std::to_string(op_index) + ",name=" + op_name +
          ",arg_map=";
      bool first_arg = true;
      for (const auto arg_fb_id : *op_call->args()) {
        const ValueRef mapped_ref =
            get_fb_id_valueref(static_cast<uint32_t>(arg_fb_id));
        args.push_back(mapped_ref);
        if (!first_arg) {
          arg_map_msg += ";";
        }
        first_arg = false;
        arg_map_msg +=
            std::to_string(static_cast<int32_t>(arg_fb_id)) + "->" +
            std::to_string(mapped_ref);
      }
      trace_vulkan_init_event(
          "build_graph_op_args_mapped",
          "GraphBuilder::build_graph",
          arg_map_msg.c_str());

#ifdef ET_EVENT_TRACER_ENABLED
      std::string operator_json =
          make_operator_json(compute_graph_, op_name, args);
      set_current_operator_json(operator_json);
      set_current_operator_node_id(op_call->node_id());
      trace_vulkan_init_event(
          "build_graph_op_operator_json",
          "GraphBuilder::build_graph",
          operator_json.c_str());
#endif // ET_EVENT_TRACER_ENABLED

      auto vkFn = VK_GET_OP_FN(op_name);
      try {
        vkFn(*compute_graph_, args);
      } catch (const std::exception& ex) {
        std::string op_err_msg =
            "op_index=" + std::to_string(op_index) + ",name=" + op_name +
            ",what=" + ex.what();
        trace_vulkan_init_event(
            "build_graph_op_exception",
            "GraphBuilder::build_graph",
            op_err_msg.c_str());
        throw;
      } catch (...) {
        std::string op_err_msg =
            "op_index=" + std::to_string(op_index) + ",name=" + op_name +
            ",what=unknown";
        trace_vulkan_init_event(
            "build_graph_op_exception",
            "GraphBuilder::build_graph",
            op_err_msg.c_str());
        throw;
      }
      trace_vulkan_init_event(
          "build_graph_op_end",
          "GraphBuilder::build_graph",
          op_begin_msg.c_str());
      op_index++;
    }
    trace_vulkan_init_event(
        "build_graph_after_ops",
        "GraphBuilder::build_graph",
        nullptr);

    // Parse the outputs, which will be mostly tensors but may contain tensorref
    // values as well if the source graph returns parameter nodes.
    for (const uint32_t fb_id : *flatbuffer_->output_ids()) {
      const ValueRef ref = get_fb_id_valueref(fb_id);
      if (compute_graph_->val_is_tensor(ref)) {
        compute_graph_->set_output_tensor(
            ref, get_staging_scalar_type_of(fb_id));
      } else {
        compute_graph_->set_output_value(ref);
      }
    }
    trace_vulkan_init_event(
        "build_graph_after_outputs",
        "GraphBuilder::build_graph",
        nullptr);

    if (compute_graph_->graphconfig().enable_querypool) {
      for (uint32_t i = 0; i < compute_graph_->prepack_nodes().size(); ++i) {
        compute_graph_->prepack_nodes()[i]->set_node_id(i);
      }
      for (uint32_t i = 0; i < compute_graph_->execute_nodes().size(); ++i) {
        compute_graph_->execute_nodes()[i]->set_node_id(i);
      }
    }
    trace_vulkan_init_event("build_graph_end", "GraphBuilder::build_graph", nullptr);
  }
};

//
// Execution tools
//

bool maybe_resize_input(
    ComputeGraph* graph,
    const size_t input_i,
    executorch::aten::Tensor& et_tensor) {
  ValueRef in_tensor_ref = graph->inputs()[input_i].value;

  const std::vector<int64_t> in_tensor_vk_sizes =
      graph->sizes_of(in_tensor_ref);

  if (et_tensor.dim() != in_tensor_vk_sizes.size()) {
    std::string msg = "input_i=" + std::to_string(input_i) + ",vk_sizes=[";
    for (size_t i = 0; i < in_tensor_vk_sizes.size(); ++i) {
      if (i > 0) {
        msg += ",";
      }
      msg += std::to_string(in_tensor_vk_sizes[i]);
    }
    msg += "],et_sizes=[";
    for (size_t i = 0; i < et_tensor.dim(); ++i) {
      if (i > 0) {
        msg += ",";
      }
      msg += std::to_string(et_tensor.sizes()[i]);
    }
    msg += "]";
    trace_vulkan_init_event(
        "execute_input_dim_mismatch",
        "VulkanBackend::maybe_resize_input",
        msg.c_str());
  }

  ET_CHECK_MSG(
      et_tensor.dim() == in_tensor_vk_sizes.size(),
      "Cannot resize input tensor: old ndim %zu does not match new ndim %zu",
      static_cast<size_t>(in_tensor_vk_sizes.size()),
      static_cast<size_t>(et_tensor.dim()));

  bool should_resize = false;
  std::vector<int64_t> new_sizes(et_tensor.dim());
  for (size_t i = 0; i < et_tensor.dim(); i++) {
    if (in_tensor_vk_sizes[i] != et_tensor.sizes()[i]) {
      should_resize = true;
    }
    new_sizes.at(i) = et_tensor.sizes()[i];
  }

  if (should_resize) {
    graph->resize_input(input_i, new_sizes);
  }

  const size_t in_tensor_vk_numel = graph->numel_of(in_tensor_ref);
  ET_CHECK_MSG(
      in_tensor_vk_numel == et_tensor.numel(),
      "Vulkan tensor numel %zu does not match ET tensor numel %zu",
      static_cast<size_t>(in_tensor_vk_numel),
      static_cast<size_t>(et_tensor.numel()));

  return should_resize;
}

bool maybe_update_scalar_tensor(
    ComputeGraph* graph,
    const ValueRef ref,
    executorch::aten::Tensor& scalar_tensor_src) {
  const int32_t cur_val = graph->read_symint(ref);
  int32_t scalar_tensor_val = 0;
  executorch::aten::ScalarType dtype = scalar_tensor_src.scalar_type();
  if (dtype == executorch::aten::ScalarType::Int) {
    scalar_tensor_val = *scalar_tensor_src.const_data_ptr<int32_t>();
  } else if (dtype == executorch::aten::ScalarType::Long) {
    scalar_tensor_val = int32_t(*scalar_tensor_src.const_data_ptr<int64_t>());
  }
  bool was_updated = false;
  if (scalar_tensor_val != cur_val) {
    graph->set_symint(ref, scalar_tensor_val);
    was_updated = true;
  }
  return was_updated;
}

void maybe_resize_output(
    ComputeGraph* graph,
    const size_t output_i,
    executorch::aten::Tensor& et_tensor) {
  ValueRef out_tensor_ref = graph->outputs()[output_i].value;

  const std::vector<int64_t> out_tensor_vk_sizes =
      graph->sizes_of(out_tensor_ref);

  executorch::aten::SizesType new_output_size[kTensorDimensionLimit];
  size_t ndim = out_tensor_vk_sizes.size();
  for (int i = 0; i < ndim; ++i) {
    new_output_size[i] = out_tensor_vk_sizes[i];
  }

  executorch::aten::ArrayRef<executorch::aten::SizesType> output_size{
      new_output_size, ndim};
  Error err = resize_tensor(et_tensor, output_size);

  ET_CHECK_MSG(err == Error::Ok, "Failed to resize output tensor.");
}

//
// VulkanBackend class
//

class VulkanBackend final : public ::executorch::runtime::BackendInterface {
 public:
  ~VulkanBackend() override = default;

  bool is_available() const override {
    // TODO(ssjia): replace with an actual Vulkan runtime availability check
    return true;
  }

  ET_NODISCARD Error compileModel(
      const void* buffer_pointer,
      ComputeGraph* compute_graph,
      const NamedDataMap* named_data_map) const {
    trace_vulkan_init_event("compile_model_begin", "VulkanBackend::compileModel", nullptr);
    try {
      Result<VulkanDelegateHeader> header =
          VulkanDelegateHeader::parse(buffer_pointer);

      const uint8_t* flatbuffer_data = nullptr;
      const uint8_t* constant_data = nullptr;

      if (header.ok()) {
        const uint8_t* buffer_start =
            reinterpret_cast<const uint8_t*>(buffer_pointer);
        flatbuffer_data = buffer_start + header->flatbuffer_offset;
        constant_data = buffer_start + header->bytes_offset;
      } else {
        trace_vulkan_init_event(
            "compile_model_header_error",
            "VulkanBackend::compileModel",
            "VulkanDelegateHeader::parse failed");
        ET_LOG(Error, "VulkanDelegateHeader may be corrupt");
        return header.error();
      }
      trace_vulkan_init_event(
          "compile_model_after_header_parse",
          "VulkanBackend::compileModel",
          nullptr);

      ET_CHECK_OR_RETURN_ERROR(
          vkgraph::VkGraphBufferHasIdentifier(flatbuffer_data),
          DelegateInvalidCompatibility,
          "Vulkan Delegate Serialization Format version identifier '%.4s' != expected '%.4s'",
          flatbuffers::GetBufferIdentifier(flatbuffer_data),
          vkgraph::VkGraphIdentifier());
      trace_vulkan_init_event(
          "compile_model_after_identifier_check",
          "VulkanBackend::compileModel",
          nullptr);

      VkGraphPtr flatbuffer_graph = vkgraph::GetVkGraph(flatbuffer_data);

      GraphBuilder builder(
          compute_graph, flatbuffer_graph, constant_data, named_data_map);

      trace_vulkan_init_event(
          "compile_model_before_resolve_layouts",
          "VulkanBackend::compileModel",
          nullptr);
      builder.resolve_layouts();
      trace_vulkan_init_event(
          "compile_model_after_resolve_layouts",
          "VulkanBackend::compileModel",
          nullptr);
      builder.build_graph();
      trace_vulkan_init_event(
          "compile_model_after_build_graph",
          "VulkanBackend::compileModel",
          nullptr);

      compute_graph->prepare();
      trace_vulkan_init_event(
          "compile_model_after_prepare",
          "VulkanBackend::compileModel",
          nullptr);
      compute_graph->prepare_pipelines();
      trace_vulkan_init_event(
          "compile_model_after_prepare_pipelines",
          "VulkanBackend::compileModel",
          nullptr);

      compute_graph->prepack();
      trace_vulkan_init_event(
          "compile_model_after_prepack",
          "VulkanBackend::compileModel",
          nullptr);

      compute_graph->optional_warmup_execute();
      trace_vulkan_init_event(
          "compile_model_after_warmup",
          "VulkanBackend::compileModel",
          nullptr);
      trace_vulkan_init_event("compile_model_end", "VulkanBackend::compileModel", nullptr);
      return Error::Ok;
    } catch (const std::exception& ex) {
      trace_vulkan_init_event(
          "compile_model_exception",
          "VulkanBackend::compileModel",
          ex.what());
      throw;
    } catch (...) {
      trace_vulkan_init_event(
          "compile_model_exception",
          "VulkanBackend::compileModel",
          "unknown exception");
      throw;
    }
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    trace_vulkan_init_event("init_begin", "VulkanBackend::init", nullptr);
    std::string shader_bundle_path = get_shader_bundle_path(compile_specs);
    if (!shader_bundle_path.empty()) {
      trace_vulkan_init_event(
          "init_skip_shader_bundle",
          "VulkanBackend::init",
          shader_bundle_path.c_str());
      ET_LOG(
          Info,
          "Ignoring shader bundle path %s in this build; using built-in shader registry",
          shader_bundle_path.c_str());
    }

    ComputeGraph* compute_graph =
        context.get_runtime_allocator()->allocateInstance<ComputeGraph>();
    if (compute_graph == nullptr) {
      trace_vulkan_init_event(
          "init_compute_graph_alloc_failed",
          "VulkanBackend::init",
          nullptr);
      return Error::MemoryAllocationFailed;
    }

    GraphConfig graph_config = get_graph_config(compile_specs);
    graph_config.external_adapter = vkapi::set_and_get_external_adapter();
    new (compute_graph) ComputeGraph(graph_config);
    trace_vulkan_init_event(
        "init_after_compute_graph_construct",
        "VulkanBackend::init",
        nullptr);

    const NamedDataMap* named_data_map = context.get_named_data_map();
    trace_vulkan_init_event(
        "init_before_compile_model",
        "VulkanBackend::init",
        nullptr);
    Error err = compileModel(processed->data(), compute_graph, named_data_map);
    trace_vulkan_init_event(
        "init_after_compile_model",
        "VulkanBackend::init",
        nullptr);

    // This backend does not need its processed data after compiling the
    // model.
    processed->Free();

    if (err != Error::Ok) {
      trace_vulkan_init_event("init_error", "VulkanBackend::init", nullptr);
      return err;
    }

    trace_vulkan_init_event("init_end", "VulkanBackend::init", nullptr);
    return compute_graph;
  }

  Error execute(
      ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    EXECUTORCH_SCOPE_PROF("VulkanBackend::execute");
    const auto t_overall_begin = SteadyClock::now();

    ComputeGraph* compute_graph = static_cast<ComputeGraph*>(handle);

    const size_t num_inputs = compute_graph->inputs().size();
    const size_t num_outputs = compute_graph->outputs().size();
    bool should_propagate_resize = false;
#ifdef ET_EVENT_TRACER_ENABLED
    runtime::EventTracer* event_tracer = context.event_tracer();
    runtime::EventTracerEntry overall_event_tracer_entry =
        event_tracer_start_profiling_delegate(
            event_tracer,
            "ETVK_EXECUTE",
            /* delegate_debug_id = */ -1);
#endif // ET_EVENT_TRACER_ENABLED
#ifdef ET_EVENT_TRACER_ENABLED
    runtime::EventTracerEntry copy_inputs_event_tracer_entry =
        event_tracer_start_profiling_delegate(
            event_tracer,
            "ETVK_COPY_INPUTS",
            /* delegate_debug_id = */ -1);
#endif // ET_EVENT_TRACER_ENABLED
    const auto t_copy_inputs_begin = SteadyClock::now();
    for (size_t i = 0; i < num_inputs; i++) {
      const ValueRef iref = compute_graph->inputs()[i].value;
      if (compute_graph->val_is_tensor(iref)) {
        VK_CHECK_COND(args[i]->isTensor());
        trace_vulkan_execute("input_tensor_begin", i, &args[i]->toTensor());
        bool was_resized =
            maybe_resize_input(compute_graph, i, args[i]->toTensor());
        trace_vulkan_execute("input_tensor_after_resize", i, &args[i]->toTensor());
        should_propagate_resize = should_propagate_resize || was_resized;
        trace_vulkan_execute("input_tensor_before_copy", i, &args[i]->toTensor());
        compute_graph->maybe_cast_and_copy_into_staging(
            compute_graph->inputs()[i].staging,
            args[i]->toTensor().const_data_ptr(),
            args[i]->toTensor().numel(),
            equivalent_scalar_type(args[i]->toTensor().scalar_type()));
        trace_vulkan_execute("input_tensor_after_copy", i, &args[i]->toTensor());
      } else if (compute_graph->val_is_symint(iref)) {
        VK_CHECK_COND(
            args[i]->isTensor(),
            "Cannot handle symint arg to graph that is not derived from a "
            "scalar tensor at the moment.");
        trace_vulkan_execute("input_symint_begin", i, &args[i]->toTensor());
        bool was_updated = maybe_update_scalar_tensor(
            compute_graph, iref, args[i]->toTensor());
        trace_vulkan_execute("input_symint_after_update", i, &args[i]->toTensor());
        // Since symint inputs may impact tensor's sizes, trigger a resize if
        // any symbolic integer shapes are updated.
        should_propagate_resize = should_propagate_resize || was_updated;
      } else {
        VK_THROW(
            "Could not handle input with type ",
            compute_graph->get_val_type(iref));
      }
    }
    const auto t_copy_inputs_end = SteadyClock::now();
#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(
        event_tracer, copy_inputs_event_tracer_entry);
#endif // ET_EVENT_TRACER_ENABLED

    const auto t_resize_begin = SteadyClock::now();
    if (should_propagate_resize || compute_graph->has_data_dependent_shapes()) {
      trace_vulkan_execute("before_propagate_resize", 0, nullptr);
#ifdef ET_EVENT_TRACER_ENABLED
      runtime::EventTracerEntry resize_event_tracer_entry =
          event_tracer_start_profiling_delegate(
              event_tracer,
              "ETVK_RESIZE",
              /* delegate_debug_id = */ -1);
#endif // ET_EVENT_TRACER_ENABLED
      compute_graph->propagate_resize();
      trace_vulkan_execute("after_propagate_resize", 0, nullptr);
#ifdef ET_EVENT_TRACER_ENABLED
      event_tracer_end_profiling_delegate(
          event_tracer, resize_event_tracer_entry);
#endif // ET_EVENT_TRACER_ENABLED
    }
    const auto t_resize_end = SteadyClock::now();

#ifdef ET_EVENT_TRACER_ENABLED
    runtime::EventTracerEntry execute_event_tracer_entry =
        event_tracer_start_profiling_delegate(
            event_tracer,
            "ETVK_COMPUTE_GRAPH_EXECUTE",
            /* delegate_debug_id = */ -1);
#endif // ET_EVENT_TRACER_ENABLED
    const auto t_compute_begin = SteadyClock::now();
    trace_vulkan_execute("before_compute_execute", 0, nullptr);
#ifdef ET_EVENT_TRACER_ENABLED
    if (event_tracer != nullptr &&
        event_tracer->has_delegate_intermediate_output_focus()) {
      compute_graph->execute_with_delegate_debug_capture(event_tracer);
    } else {
      compute_graph->execute();
    }
#else
    compute_graph->execute();
#endif
    trace_vulkan_execute("after_compute_execute", 0, nullptr);
    const auto t_compute_end = SteadyClock::now();
#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(
        event_tracer, execute_event_tracer_entry);
#endif // ET_EVENT_TRACER_ENABLED

    double gpu_shader_total_ms = 0.0;
    uint64_t gpu_shader_dispatch_count = 0;
    if (compute_graph->context()->querypool()) {
      compute_graph->context()->querypool().extract_results();
      for (const auto& r :
           compute_graph->context()->querypool().get_shader_timestamp_data()) {
        if (r.end_time_ns >= r.start_time_ns) {
          gpu_shader_total_ms +=
              static_cast<double>(r.end_time_ns - r.start_time_ns) / 1.0e6;
        }
        gpu_shader_dispatch_count += 1;
#ifdef ET_EVENT_TRACER_ENABLED
        std::string event_name = "{" + r.kernel_name +
            ", \"dispatch_id\": " + std::to_string(r.dispatch_id) + "}";
        event_tracer_log_profiling_delegate(
            event_tracer,
            event_name.c_str(),
            /* delegate_debug_id = */ -1,
            r.start_time_ns,
            r.end_time_ns);
#endif // ET_EVENT_TRACER_ENABLED
      }
    }

#ifdef ET_EVENT_TRACER_ENABLED
    runtime::EventTracerEntry copy_outputs_event_tracer_entry =
        event_tracer_start_profiling_delegate(
            event_tracer,
            "ETVK_COPY_OUTPUTS",
            /* delegate_debug_id = */ -1);
#endif // ET_EVENT_TRACER_ENABLED
    const auto t_copy_outputs_begin = SteadyClock::now();
    const size_t output_offset = args.size() - num_outputs;
    for (size_t i = 0; i < num_outputs; i++) {
      const size_t o = output_offset + i;
      const ValueRef oref = compute_graph->outputs()[i].value;
      if (compute_graph->val_is_tensor(oref)) {
        VK_CHECK_COND(args[o]->isTensor());
        maybe_resize_output(compute_graph, i, args[o]->toTensor());
        compute_graph->maybe_cast_and_copy_from_staging(
            compute_graph->outputs()[i].staging,
            args[o]->toTensor().mutable_data_ptr(),
            args[o]->toTensor().numel(),
            equivalent_scalar_type(args[o]->toTensor().scalar_type()));
      }
      // TensorRef values represent constant tensors which will not have been
      // modified by the graph execution. Therefore, if a constant tensor is
      // returned as an output, no action is required.
      else if (compute_graph->val_is_tref(oref)) {
        continue;
      } else {
        VK_THROW(
            "Could not handle output with type ",
            compute_graph->get_val_type(oref));
      }
    }
    const auto t_copy_outputs_end = SteadyClock::now();
#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(
        event_tracer, copy_outputs_event_tracer_entry);
#endif // ET_EVENT_TRACER_ENABLED

#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(
        event_tracer, overall_event_tracer_entry);
#endif // ET_EVENT_TRACER_ENABLED

    VulkanExecuteTelemetry telemetry{};
    telemetry.generation = ++t_last_execute_generation;
    telemetry.copy_inputs_ms = std::chrono::duration<double, std::milli>(
                                   t_copy_inputs_end - t_copy_inputs_begin)
                                   .count();
    telemetry.resize_ms =
        std::chrono::duration<double, std::milli>(t_resize_end - t_resize_begin)
            .count();
    telemetry.compute_graph_execute_ms = std::chrono::duration<double, std::milli>(
                                             t_compute_end - t_compute_begin)
                                             .count();
    telemetry.copy_outputs_ms = std::chrono::duration<double, std::milli>(
                                    t_copy_outputs_end - t_copy_outputs_begin)
                                    .count();
    telemetry.total_backend_ms = std::chrono::duration<double, std::milli>(
                                     t_copy_outputs_end - t_overall_begin)
                                     .count();
    telemetry.gpu_shader_total_ms = gpu_shader_total_ms;
    telemetry.gpu_shader_dispatch_count = gpu_shader_dispatch_count;
    t_last_execute_telemetry = telemetry;
    t_has_last_execute_telemetry = true;

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      ComputeGraph* compute_graph = static_cast<ComputeGraph*>(handle);
      compute_graph->context()
          ->adapter_ptr()
          ->compute_pipeline_cache()
          .save_cache();
      // ComputeGraph is not trivially destructible. Since
      // this was constructed manually in init(), we must destroy it manually
      // here.
      compute_graph->~ComputeGraph();
    }
  }
};

auto cls = VulkanBackend();
Backend backend{"VulkanBackend", &cls};
static auto success_with_compiler = register_backend(backend);

} // namespace
} // namespace vulkan
} // namespace backends
} // namespace executorch
