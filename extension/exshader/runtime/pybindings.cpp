/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cinttypes>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/backends/vulkan/runtime/VulkanExecuteTelemetry.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/tag.h>
#include <executorch/runtime/executor/program.h>

#define THROW_IF_ERROR(error, message, ...)                       \
  ({                                                              \
    if ((error) != executorch::runtime::Error::Ok) {             \
      char msg_buf[256];                                          \
      snprintf(msg_buf, sizeof(msg_buf), message, ##__VA_ARGS__); \
      throw std::runtime_error(msg_buf);                          \
    }                                                             \
  })

namespace py = pybind11;

using ::executorch::ET_RUNTIME_NAMESPACE::BackendInterface;
using ::executorch::ET_RUNTIME_NAMESPACE::get_backend_class;
using ::executorch::ET_RUNTIME_NAMESPACE::get_backend_name;
using ::executorch::ET_RUNTIME_NAMESPACE::get_num_registered_backends;
using ::executorch::backends::vulkan::VulkanExecuteTelemetry;
using ::executorch::backends::vulkan::get_last_vulkan_execute_telemetry;
using ::executorch::backends::vulkan::reset_last_vulkan_execute_telemetry;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Program;
using ::executorch::runtime::Tag;

namespace {

using SteadyClock = std::chrono::steady_clock;

inline double elapsed_ms(
    const SteadyClock::time_point& begin,
    const SteadyClock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline bool is_non_string_sequence(const py::handle& obj) {
  return py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj) &&
      !py::isinstance<py::bytes>(obj) && !py::isinstance<py::bytearray>(obj);
}

struct ParsedIntTensor {
  std::vector<int64_t> data;
  std::vector<executorch::aten::SizesType> sizes;
};

inline bool try_parse_int_tensor_sequence(
    const py::handle& obj,
    ParsedIntTensor& parsed,
    std::string& error_msg) {
  if (!is_non_string_sequence(obj)) {
    return false;
  }
  py::sequence outer = py::reinterpret_borrow<py::sequence>(obj);
  const size_t outer_len = py::len(outer);
  if (outer_len == 0) {
    parsed.data.clear();
    parsed.sizes = {0};
    return true;
  }

  py::handle first = outer[0];
  const bool is_2d = is_non_string_sequence(first);
  if (is_2d) {
    py::sequence first_row = py::reinterpret_borrow<py::sequence>(first);
    const size_t inner_len = py::len(first_row);
    parsed.data.clear();
    parsed.data.reserve(outer_len * inner_len);
    parsed.sizes = {
        static_cast<executorch::aten::SizesType>(outer_len),
        static_cast<executorch::aten::SizesType>(inner_len)};

    for (size_t i = 0; i < outer_len; ++i) {
      py::handle row_obj = outer[i];
      if (!is_non_string_sequence(row_obj)) {
        error_msg = "Mixed nested and non-nested sequences are not supported";
        return false;
      }
      py::sequence row = py::reinterpret_borrow<py::sequence>(row_obj);
      if (py::len(row) != inner_len) {
        error_msg = "Ragged nested sequences are not supported";
        return false;
      }
      for (size_t j = 0; j < inner_len; ++j) {
        py::handle item = row[j];
        if (!py::isinstance<py::int_>(item)) {
          error_msg = "Sequence tensor only supports int elements";
          return false;
        }
        parsed.data.push_back(py::cast<int64_t>(item));
      }
    }
    return true;
  }

  parsed.data.clear();
  parsed.data.reserve(outer_len);
  parsed.sizes = {static_cast<executorch::aten::SizesType>(outer_len)};
  for (size_t i = 0; i < outer_len; ++i) {
    py::handle item = outer[i];
    if (!py::isinstance<py::int_>(item)) {
      error_msg = "Sequence tensor only supports int elements";
      return false;
    }
    parsed.data.push_back(py::cast<int64_t>(item));
  }
  return true;
}

inline TensorPtr make_tensor_ptr_from_parsed_int(ParsedIntTensor parsed) {
  return ::executorch::extension::make_tensor_ptr(
      std::move(parsed.sizes),
      std::move(parsed.data),
      {},
      {},
      executorch::aten::ScalarType::Long,
      executorch::aten::TensorShapeDynamism::STATIC);
}

inline std::vector<executorch::aten::SizesType> parse_tensor_sizes(
    const py::sequence& sizes) {
  const auto n = py::len(sizes);
  if (n <= 0) {
    throw std::runtime_error("Tensor shape must have at least 1 dimension");
  }
  std::vector<executorch::aten::SizesType> out;
  out.reserve(static_cast<size_t>(n));
  for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
    py::handle v = sizes[i];
    if (!py::isinstance<py::int_>(v)) {
      throw std::runtime_error("Tensor shape values must be integers");
    }
    const int64_t dim = py::cast<int64_t>(v);
    if (dim < 0) {
      throw std::runtime_error("Tensor shape values must be >= 0");
    }
    out.push_back(static_cast<executorch::aten::SizesType>(dim));
  }
  return out;
}

inline TensorPtr make_int64_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes) {
  const auto numel = executorch::aten::compute_numel(sizes.data(), sizes.size());
  if (numel < 0) {
    throw std::runtime_error("Invalid tensor shape: negative numel");
  }
  std::vector<int64_t> data(static_cast<size_t>(numel), 0);
  return ::executorch::extension::make_tensor_ptr(
      std::move(sizes),
      std::move(data),
      {},
      {},
      executorch::aten::ScalarType::Long,
      executorch::aten::TensorShapeDynamism::STATIC);
}

template <typename T>
inline double to_double_value(T v) {
  return static_cast<double>(v);
}

template <>
inline double to_double_value<bool>(bool v) {
  return v ? 1.0 : 0.0;
}

class PyExTensor final {
 public:
  explicit PyExTensor(TensorPtr tensor_ptr) : tensor_ptr_(std::move(tensor_ptr)) {
    if (!tensor_ptr_) {
      throw std::runtime_error("PyExTensor received null TensorPtr");
    }
  }

  py::tuple sizes() const {
    const auto shape = tensor_ptr_->sizes();
    py::tuple tup(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      tup[i] = py::cast(static_cast<int64_t>(shape[i]));
    }
    return tup;
  }

  int dtype() const {
    return static_cast<int>(tensor_ptr_->scalar_type());
  }

  size_t nbytes() const {
    return tensor_ptr_->nbytes();
  }

  ssize_t numel() const {
    return tensor_ptr_->numel();
  }

  int64_t argmax() const {
    const auto& t = *tensor_ptr_;
    const size_t n = static_cast<size_t>(t.numel());
    if (n == 0) {
      throw std::runtime_error("argmax on empty tensor");
    }
    return argmax_impl(0, n);
  }

  int64_t argmax_last_dim_row0() const {
    const auto& t = *tensor_ptr_;
    const ssize_t dim = t.dim();
    size_t row_len = 0;
    if (dim == 1) {
      row_len = static_cast<size_t>(t.size(0));
    } else if (dim == 2) {
      if (t.size(0) < 1) {
        throw std::runtime_error("argmax_last_dim_row0 requires size(0) >= 1");
      }
      row_len = static_cast<size_t>(t.size(1));
    } else {
      throw std::runtime_error(
          "argmax_last_dim_row0 currently supports 1D/2D tensors only");
    }
    if (row_len == 0) {
      throw std::runtime_error("argmax_last_dim_row0 on empty row");
    }
    return argmax_impl(0, row_len);
  }

  int64_t sample_top_p_row0(
      double temperature,
      double top_p,
      std::optional<uint64_t> seed = std::nullopt) const {
    if (temperature <= 0.0 || !std::isfinite(temperature)) {
      return argmax_last_dim_row0();
    }

    const auto& t = *tensor_ptr_;
    const ssize_t dim = t.dim();
    size_t row_len = 0;
    if (dim == 1) {
      row_len = static_cast<size_t>(t.size(0));
    } else if (dim == 2) {
      if (t.size(0) < 1) {
        throw std::runtime_error("sample_top_p_row0 requires size(0) >= 1");
      }
      row_len = static_cast<size_t>(t.size(1));
    } else {
      throw std::runtime_error(
          "sample_top_p_row0 currently supports 1D/2D tensors only");
    }
    if (row_len == 0) {
      throw std::runtime_error("sample_top_p_row0 on empty row");
    }
    return sample_top_p_impl(0, row_len, temperature, top_p, seed);
  }

  py::list tolist(std::optional<size_t> max_elems = std::nullopt) const {
    const auto& t = *tensor_ptr_;
    const size_t total = static_cast<size_t>(t.numel());
    const size_t limit =
        max_elems.has_value() ? std::min(total, max_elems.value()) : total;
    py::list out(limit);
    switch (t.scalar_type()) {
      case executorch::aten::ScalarType::Float:
        fill_list<float>(out, limit);
        break;
      case executorch::aten::ScalarType::Double:
        fill_list<double>(out, limit);
        break;
      case executorch::aten::ScalarType::Long:
        fill_list<int64_t>(out, limit);
        break;
      case executorch::aten::ScalarType::Int:
        fill_list<int32_t>(out, limit);
        break;
      case executorch::aten::ScalarType::Short:
        fill_list<int16_t>(out, limit);
        break;
      case executorch::aten::ScalarType::Char:
        fill_list<int8_t>(out, limit);
        break;
      case executorch::aten::ScalarType::Byte:
        fill_list<uint8_t>(out, limit);
        break;
      case executorch::aten::ScalarType::Bool:
        fill_list<bool>(out, limit);
        break;
      default:
        throw std::runtime_error("tolist does not support this dtype");
    }
    return out;
  }

  void set_int64_scalar(int64_t value) {
    if (tensor_ptr_->scalar_type() != executorch::aten::ScalarType::Long) {
      throw std::runtime_error("set_int64_scalar requires int64 tensor");
    }
    if (tensor_ptr_->numel() != 1) {
      throw std::runtime_error("set_int64_scalar requires tensor with numel == 1");
    }
    tensor_ptr_->mutable_data_ptr<int64_t>()[0] = value;
  }

  void set_int64_flat(size_t index, int64_t value) {
    if (tensor_ptr_->scalar_type() != executorch::aten::ScalarType::Long) {
      throw std::runtime_error("set_int64_flat requires int64 tensor");
    }
    const auto total = static_cast<size_t>(tensor_ptr_->numel());
    if (index >= total) {
      throw std::runtime_error("set_int64_flat index out of range");
    }
    tensor_ptr_->mutable_data_ptr<int64_t>()[index] = value;
  }

  void set_int64_row0_prefix(const py::sequence& values) {
    if (tensor_ptr_->scalar_type() != executorch::aten::ScalarType::Long) {
      throw std::runtime_error("set_int64_row0_prefix requires int64 tensor");
    }
    const auto& t = *tensor_ptr_;
    if (t.dim() != 2 || t.size(0) < 1) {
      throw std::runtime_error(
          "set_int64_row0_prefix requires tensor with shape [>=1, N]");
    }
    const size_t cap = static_cast<size_t>(t.size(1));
    const size_t n = static_cast<size_t>(py::len(values));
    if (n > cap) {
      throw std::runtime_error("set_int64_row0_prefix values exceed tensor width");
    }
    auto* data = tensor_ptr_->mutable_data_ptr<int64_t>();
    for (size_t i = 0; i < n; ++i) {
      py::handle v = values[i];
      if (!py::isinstance<py::int_>(v)) {
        throw std::runtime_error("set_int64_row0_prefix expects int values");
      }
      data[i] = py::cast<int64_t>(v);
    }
  }

  PyExTensor row0_prefix(size_t cols) const {
    const auto& t = *tensor_ptr_;
    if (t.dim() != 2 || t.size(0) < 1) {
      throw std::runtime_error("row0_prefix requires tensor with shape [>=1, N]");
    }
    const size_t cap = static_cast<size_t>(t.size(1));
    if (cols == 0 || cols > cap) {
      throw std::runtime_error("row0_prefix cols must be in [1, N]");
    }
    std::vector<executorch::aten::SizesType> view_sizes{
        static_cast<executorch::aten::SizesType>(1),
        static_cast<executorch::aten::SizesType>(cols)};
    TensorPtr view = ::executorch::extension::make_tensor_ptr(
        tensor_ptr_, std::move(view_sizes), {}, {});
    return PyExTensor(std::move(view));
  }

  std::string repr() const {
    return "ExTensor(shape=" + py::str(sizes()).cast<std::string>() +
        ", dtype=" + std::to_string(dtype()) +
        ", nbytes=" + std::to_string(nbytes()) + ")";
  }

  TensorPtr tensor_ptr() const {
    return tensor_ptr_;
  }

 private:
  template <typename T>
  void fill_list(py::list& out, size_t limit) const {
    const auto* data = tensor_ptr_->const_data_ptr<T>();
    for (size_t i = 0; i < limit; ++i) {
      out[i] = py::cast(data[i]);
    }
  }

  template <typename T>
  int64_t argmax_t(const T* data, size_t offset, size_t len) const {
    size_t best_idx = 0;
    double best_val = to_double_value<T>(data[offset]);
    for (size_t i = 1; i < len; ++i) {
      const double v = to_double_value<T>(data[offset + i]);
      if (v > best_val) {
        best_val = v;
        best_idx = i;
      }
    }
    return static_cast<int64_t>(best_idx);
  }

  int64_t argmax_impl(size_t offset, size_t len) const {
    const auto& t = *tensor_ptr_;
    switch (t.scalar_type()) {
      case executorch::aten::ScalarType::Float:
        return argmax_t<float>(t.const_data_ptr<float>(), offset, len);
      case executorch::aten::ScalarType::Double:
        return argmax_t<double>(t.const_data_ptr<double>(), offset, len);
      case executorch::aten::ScalarType::Long:
        return argmax_t<int64_t>(t.const_data_ptr<int64_t>(), offset, len);
      case executorch::aten::ScalarType::Int:
        return argmax_t<int32_t>(t.const_data_ptr<int32_t>(), offset, len);
      case executorch::aten::ScalarType::Short:
        return argmax_t<int16_t>(t.const_data_ptr<int16_t>(), offset, len);
      case executorch::aten::ScalarType::Char:
        return argmax_t<int8_t>(t.const_data_ptr<int8_t>(), offset, len);
      case executorch::aten::ScalarType::Byte:
        return argmax_t<uint8_t>(t.const_data_ptr<uint8_t>(), offset, len);
      case executorch::aten::ScalarType::Bool:
        return argmax_t<bool>(t.const_data_ptr<bool>(), offset, len);
      default:
        throw std::runtime_error("argmax is not supported for this dtype");
    }
  }

  template <typename T>
  int64_t sample_top_p_t(
      const T* data,
      size_t offset,
      size_t len,
      double temperature,
      double top_p,
      std::optional<uint64_t> seed) const {
    const double inv_temp = 1.0 / temperature;

    std::vector<std::pair<double, int64_t>> probs;
    probs.reserve(len);
    double max_logit = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < len; ++i) {
      const double scaled = to_double_value<T>(data[offset + i]) * inv_temp;
      max_logit = std::max(max_logit, scaled);
      probs.emplace_back(scaled, static_cast<int64_t>(i));
    }
    double sum_exp = 0.0;
    for (auto& entry : probs) {
      entry.first = std::exp(entry.first - max_logit);
      sum_exp += entry.first;
    }
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
      return argmax_t<T>(data, offset, len);
    }

    std::sort(
        probs.begin(),
        probs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    const double clipped_top_p = std::clamp(top_p, 0.0, 1.0);
    size_t keep = probs.size();
    if (clipped_top_p < 1.0) {
      double cumulative = 0.0;
      keep = 0;
      for (const auto& entry : probs) {
        cumulative += entry.first / sum_exp;
        ++keep;
        if (cumulative >= clipped_top_p && keep >= 1) {
          break;
        }
      }
      keep = std::max<size_t>(1, keep);
    }

    double selected_sum = 0.0;
    for (size_t i = 0; i < keep; ++i) {
      selected_sum += probs[i].first;
    }
    if (!(selected_sum > 0.0) || !std::isfinite(selected_sum)) {
      return probs[0].second;
    }

    std::mt19937_64 rng(
        seed.has_value() ? seed.value() : std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, selected_sum);
    const double r = dist(rng);

    double acc = 0.0;
    for (size_t i = 0; i < keep; ++i) {
      acc += probs[i].first;
      if (r <= acc) {
        return probs[i].second;
      }
    }
    return probs[keep - 1].second;
  }

  int64_t sample_top_p_impl(
      size_t offset,
      size_t len,
      double temperature,
      double top_p,
      std::optional<uint64_t> seed) const {
    const auto& t = *tensor_ptr_;
    switch (t.scalar_type()) {
      case executorch::aten::ScalarType::Float:
        return sample_top_p_t<float>(
            t.const_data_ptr<float>(), offset, len, temperature, top_p, seed);
      case executorch::aten::ScalarType::Double:
        return sample_top_p_t<double>(
            t.const_data_ptr<double>(), offset, len, temperature, top_p, seed);
      case executorch::aten::ScalarType::Long:
        return sample_top_p_t<int64_t>(
            t.const_data_ptr<int64_t>(), offset, len, temperature, top_p, seed);
      case executorch::aten::ScalarType::Int:
        return sample_top_p_t<int32_t>(
            t.const_data_ptr<int32_t>(), offset, len, temperature, top_p, seed);
      case executorch::aten::ScalarType::Short:
        return sample_top_p_t<int16_t>(
            t.const_data_ptr<int16_t>(), offset, len, temperature, top_p, seed);
      case executorch::aten::ScalarType::Char:
        return sample_top_p_t<int8_t>(
            t.const_data_ptr<int8_t>(), offset, len, temperature, top_p, seed);
      case executorch::aten::ScalarType::Byte:
        return sample_top_p_t<uint8_t>(
            t.const_data_ptr<uint8_t>(), offset, len, temperature, top_p, seed);
      case executorch::aten::ScalarType::Bool:
        return sample_top_p_t<bool>(
            t.const_data_ptr<bool>(), offset, len, temperature, top_p, seed);
      default:
        throw std::runtime_error(
            "sample_top_p_row0 is not supported for this dtype");
    }
  }

  TensorPtr tensor_ptr_;
};

inline py::dict tensor_info_to_dict(
    const executorch::runtime::TensorInfo& info) {
  py::tuple shape(info.sizes().size());
  for (size_t i = 0; i < info.sizes().size(); ++i) {
    shape[i] = py::cast(static_cast<int64_t>(info.sizes()[i]));
  }
  py::dict out;
  out["sizes"] = shape;
  out["dtype"] = py::cast(static_cast<int>(info.scalar_type()));
  out["nbytes"] = py::cast(static_cast<int64_t>(info.nbytes()));
  out["is_memory_planned"] = py::cast(info.is_memory_planned());
  return out;
}

inline py::list outputs_to_py_list(
    const std::shared_ptr<std::vector<EValue>>& outputs_owner,
    bool clone_outputs) {
  py::list py_outputs(outputs_owner->size());
  for (size_t i = 0; i < outputs_owner->size(); ++i) {
    const auto& value = outputs_owner->at(i);
    switch (value.tag) {
      case Tag::None:
        py_outputs[i] = py::none();
        break;
      case Tag::Bool:
        py_outputs[i] = py::cast(value.toBool());
        break;
      case Tag::Int:
        py_outputs[i] = py::cast(value.toInt());
        break;
      case Tag::Double:
        py_outputs[i] = py::cast(value.toDouble());
        break;
      case Tag::String:
        py_outputs[i] = py::cast(std::string(value.toString()));
        break;
      case Tag::Tensor: {
        TensorPtr tensor_ptr = clone_outputs
            ? ::executorch::extension::clone_tensor_ptr(value.toTensor())
            : ::executorch::extension::make_tensor_ptr(
                  value.toTensor(), {}, {}, {}, [outputs_owner](void*) {});
        py_outputs[i] = py::cast(PyExTensor(std::move(tensor_ptr)));
        break;
      }
      default:
        throw std::runtime_error("Unsupported output EValue tag");
    }
  }
  return py_outputs;
}

struct MethodRunStats {
  double elapsed_ms = 0.0;
  double host_input_ms = 0.0;
  double module_execute_ms = 0.0;
  double output_wrap_ms = 0.0;
  std::optional<double> vk_copy_inputs_ms = std::nullopt;
  std::optional<double> vk_resize_ms = std::nullopt;
  std::optional<double> vk_compute_graph_execute_ms = std::nullopt;
  std::optional<double> vk_copy_outputs_ms = std::nullopt;
  std::optional<double> vk_total_backend_ms = std::nullopt;
  std::optional<double> vk_gpu_shader_total_ms = std::nullopt;
  std::optional<uint64_t> vk_gpu_shader_dispatch_count = std::nullopt;
  std::optional<uint64_t> vk_generation = std::nullopt;
};

inline void clear_vulkan_stats(MethodRunStats* stats) {
  if (stats == nullptr) {
    return;
  }
  stats->vk_copy_inputs_ms = std::nullopt;
  stats->vk_resize_ms = std::nullopt;
  stats->vk_compute_graph_execute_ms = std::nullopt;
  stats->vk_copy_outputs_ms = std::nullopt;
  stats->vk_total_backend_ms = std::nullopt;
  stats->vk_gpu_shader_total_ms = std::nullopt;
  stats->vk_gpu_shader_dispatch_count = std::nullopt;
  stats->vk_generation = std::nullopt;
}

inline py::dict run_stats_to_dict(const MethodRunStats& stats) {
  py::dict out;
  out["elapsed_ms"] = py::cast(stats.elapsed_ms);
  out["host_input_ms"] = py::cast(stats.host_input_ms);
  out["module_execute_ms"] = py::cast(stats.module_execute_ms);
  out["output_wrap_ms"] = py::cast(stats.output_wrap_ms);
  out["vk_copy_inputs_ms"] =
      stats.vk_copy_inputs_ms.has_value() ? py::cast(*stats.vk_copy_inputs_ms)
                                          : py::none();
  out["vk_resize_ms"] =
      stats.vk_resize_ms.has_value() ? py::cast(*stats.vk_resize_ms) : py::none();
  out["vk_compute_graph_execute_ms"] = stats.vk_compute_graph_execute_ms.has_value()
      ? py::cast(*stats.vk_compute_graph_execute_ms)
      : py::none();
  out["vk_copy_outputs_ms"] = stats.vk_copy_outputs_ms.has_value()
      ? py::cast(*stats.vk_copy_outputs_ms)
      : py::none();
  out["vk_total_backend_ms"] = stats.vk_total_backend_ms.has_value()
      ? py::cast(*stats.vk_total_backend_ms)
      : py::none();
  out["vk_gpu_shader_total_ms"] = stats.vk_gpu_shader_total_ms.has_value()
      ? py::cast(*stats.vk_gpu_shader_total_ms)
      : py::none();
  out["vk_gpu_shader_dispatch_count"] =
      stats.vk_gpu_shader_dispatch_count.has_value()
      ? py::cast(*stats.vk_gpu_shader_dispatch_count)
      : py::none();
  out["vk_generation"] =
      stats.vk_generation.has_value() ? py::cast(*stats.vk_generation) : py::none();
  return out;
}

class SessionHandle final {
 public:
  SessionHandle(
      const std::string& model_path,
      std::optional<std::string> data_map_path,
      Program::Verification program_verification)
      : model_path_(model_path), data_map_path_(std::move(data_map_path)) {
    if (data_map_path_.has_value()) {
      module_ = std::make_shared<Module>(model_path_, data_map_path_.value());
    } else {
      module_ = std::make_shared<Module>(model_path_);
    }
    THROW_IF_ERROR(
        module_->load(program_verification),
        "Failed to load program '%s' with verification=%d",
        model_path_.c_str(),
        static_cast<int>(program_verification));
  }

  static std::unique_ptr<SessionHandle> load(
      const std::string& model_path,
      std::optional<std::string> data_map_path = std::nullopt,
      Program::Verification program_verification =
          Program::Verification::Minimal) {
    return std::make_unique<SessionHandle>(
        model_path, std::move(data_map_path), program_verification);
  }

  PyExTensor alloc_int64(const py::sequence& sizes) const {
    ensure_open();
    auto parsed = parse_tensor_sizes(sizes);
    return PyExTensor(make_int64_tensor_ptr(std::move(parsed)));
  }

  void close() {
    method_contexts_.clear();
    module_.reset();
  }

  py::list method_names() const {
    ensure_open();
    auto names_result = module_->method_names();
    THROW_IF_ERROR(
        names_result.error(),
        "Failed to get method names, error=0x%" PRIx32,
        static_cast<uint32_t>(names_result.error()));
    std::vector<std::string> names(
        names_result->begin(), names_result->end());
    std::sort(names.begin(), names.end());
    py::list out;
    for (const auto& name : names) {
      out.append(name);
    }
    return out;
  }

  py::dict method_meta(const std::string& method_name) const {
    ensure_open();
    auto meta_result = module_->method_meta(method_name);
    THROW_IF_ERROR(
        meta_result.error(),
        "Failed to get method meta for '%s', error=0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(meta_result.error()));
    const auto& meta = meta_result.get();

    py::list inputs;
    for (size_t i = 0; i < meta.num_inputs(); ++i) {
      auto tag_result = meta.input_tag(i);
      THROW_IF_ERROR(
          tag_result.error(),
          "Failed to get input tag %zu for '%s', error=0x%" PRIx32,
          i,
          method_name.c_str(),
          static_cast<uint32_t>(tag_result.error()));
      if (tag_result.get() == Tag::Tensor) {
        auto tensor_meta_result = meta.input_tensor_meta(i);
        THROW_IF_ERROR(
            tensor_meta_result.error(),
            "Failed to get input tensor meta %zu for '%s', error=0x%" PRIx32,
            i,
            method_name.c_str(),
            static_cast<uint32_t>(tensor_meta_result.error()));
        inputs.append(tensor_info_to_dict(tensor_meta_result.get()));
      } else {
        inputs.append(py::none());
      }
    }

    py::list outputs;
    for (size_t i = 0; i < meta.num_outputs(); ++i) {
      auto tag_result = meta.output_tag(i);
      THROW_IF_ERROR(
          tag_result.error(),
          "Failed to get output tag %zu for '%s', error=0x%" PRIx32,
          i,
          method_name.c_str(),
          static_cast<uint32_t>(tag_result.error()));
      if (tag_result.get() == Tag::Tensor) {
        auto tensor_meta_result = meta.output_tensor_meta(i);
        THROW_IF_ERROR(
            tensor_meta_result.error(),
            "Failed to get output tensor meta %zu for '%s', error=0x%" PRIx32,
            i,
            method_name.c_str(),
            static_cast<uint32_t>(tensor_meta_result.error()));
        outputs.append(tensor_info_to_dict(tensor_meta_result.get()));
      } else {
        outputs.append(py::none());
      }
    }

    py::dict out;
    out["name"] = method_name;
    out["inputs"] = inputs;
    out["outputs"] = outputs;
    return out;
  }

  py::list run(
      const std::string& method_name,
      const py::sequence& inputs,
      bool clone_outputs = false) {
    py::dict result = run_with_stats(method_name, inputs, clone_outputs);
    return result["values"].cast<py::list>();
  }

  py::dict run_with_stats(
      const std::string& method_name,
      const py::sequence& inputs,
      bool clone_outputs = false) {
    ensure_open();
    const auto t0 = SteadyClock::now();
    MethodContext& context = method_contexts_[method_name];
    context.last_stats = MethodRunStats();
    parse_inputs(method_name, inputs, context);
    const auto t1 = SteadyClock::now();
    execute_method(method_name, context);
    const auto t2 = SteadyClock::now();
    py::list values = get_outputs_for_context(method_name, context, clone_outputs);
    const auto t3 = SteadyClock::now();

    context.last_stats.host_input_ms = elapsed_ms(t0, t1);
    context.last_stats.module_execute_ms = elapsed_ms(t1, t2);
    context.last_stats.output_wrap_ms = elapsed_ms(t2, t3);
    context.last_stats.elapsed_ms = elapsed_ms(t0, t3);

    py::dict out;
    out["values"] = values;
    out["stats"] = run_stats_to_dict(context.last_stats);
    return out;
  }

  void set_inputs(const std::string& method_name, const py::sequence& inputs) {
    ensure_open();
    const auto t0 = SteadyClock::now();
    MethodContext& context = method_contexts_[method_name];
    parse_inputs(method_name, inputs, context);
    const auto t1 = SteadyClock::now();
    context.last_stats = MethodRunStats();
    context.last_stats.host_input_ms = elapsed_ms(t0, t1);
    context.outputs_owner.reset();
  }

  void execute(const std::string& method_name) {
    ensure_open();
    auto it = method_contexts_.find(method_name);
    if (it == method_contexts_.end()) {
      throw std::runtime_error(
          "execute called before set_inputs/run for method '" + method_name +
          "'");
    }
    const auto t0 = SteadyClock::now();
    execute_method(method_name, it->second);
    const auto t1 = SteadyClock::now();
    it->second.last_stats.module_execute_ms = elapsed_ms(t0, t1);
  }

  py::list get_outputs(const std::string& method_name, bool clone_outputs = false) {
    ensure_open();
    auto it = method_contexts_.find(method_name);
    if (it == method_contexts_.end() || !it->second.outputs_owner) {
      throw std::runtime_error(
          "get_outputs called before execute/run for method '" + method_name +
          "'");
    }
    const auto t0 = SteadyClock::now();
    py::list values = get_outputs_for_context(method_name, it->second, clone_outputs);
    const auto t1 = SteadyClock::now();
    it->second.last_stats.output_wrap_ms = elapsed_ms(t0, t1);
    it->second.last_stats.elapsed_ms = it->second.last_stats.host_input_ms +
        it->second.last_stats.module_execute_ms +
        it->second.last_stats.output_wrap_ms;
    return values;
  }

  py::dict get_last_run_stats(const std::string& method_name) const {
    ensure_open();
    auto it = method_contexts_.find(method_name);
    if (it == method_contexts_.end()) {
      throw std::runtime_error(
          "No method context found for '" + method_name +
          "'. Call set_inputs/run first.");
    }
    return run_stats_to_dict(it->second.last_stats);
  }

 private:
  struct MethodContext {
    std::vector<EValue> inputs;
    std::vector<TensorPtr> tensor_owners;
    std::shared_ptr<std::vector<EValue>> outputs_owner;
    MethodRunStats last_stats;
  };

  void parse_inputs(
      const std::string& method_name,
      const py::sequence& inputs,
      MethodContext& context) {
    const auto inputs_size = py::len(inputs);
    context.inputs.clear();
    context.tensor_owners.clear();
    context.inputs.reserve(inputs_size);
    context.tensor_owners.reserve(inputs_size);

    for (size_t i = 0; i < inputs_size; ++i) {
      py::handle input = inputs[i];
      if (py::isinstance<PyExTensor>(input)) {
        const auto& ex_tensor = input.cast<const PyExTensor&>();
        context.tensor_owners.push_back(ex_tensor.tensor_ptr());
        context.inputs.push_back(EValue(context.tensor_owners.back()));
      } else if (py::isinstance<py::none>(input)) {
        context.inputs.push_back(EValue());
      } else if (py::isinstance<py::bool_>(input)) {
        context.inputs.push_back(EValue(py::cast<bool>(input)));
      } else if (py::isinstance<py::int_>(input)) {
        context.inputs.push_back(EValue(py::cast<int64_t>(input)));
      } else {
        ParsedIntTensor parsed;
        std::string parse_error;
        if (try_parse_int_tensor_sequence(input, parsed, parse_error)) {
          context.tensor_owners.push_back(
              make_tensor_ptr_from_parsed_int(std::move(parsed)));
          context.inputs.push_back(EValue(context.tensor_owners.back()));
        } else if (!parse_error.empty()) {
          throw std::runtime_error(
              "Input " + std::to_string(i) + " for method '" + method_name +
              "' is an invalid sequence tensor: " + parse_error);
        } else {
          throw std::runtime_error(
              "Unsupported input type for method '" + method_name + "'");
        }
      }
    }
  }

  void execute_method(const std::string& method_name, MethodContext& context) {
    clear_vulkan_stats(&context.last_stats);
    reset_last_vulkan_execute_telemetry();
    auto outputs_result = module_->execute(method_name, context.inputs);
    THROW_IF_ERROR(
        outputs_result.error(),
        "Failed to execute method '%s', error=0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(outputs_result.error()));
    context.outputs_owner = std::make_shared<std::vector<EValue>>(
        std::move(outputs_result.get()));
    VulkanExecuteTelemetry vk_stats{};
    if (get_last_vulkan_execute_telemetry(&vk_stats)) {
      context.last_stats.vk_generation = vk_stats.generation;
      context.last_stats.vk_copy_inputs_ms = vk_stats.copy_inputs_ms;
      context.last_stats.vk_resize_ms = vk_stats.resize_ms;
      context.last_stats.vk_compute_graph_execute_ms =
          vk_stats.compute_graph_execute_ms;
      context.last_stats.vk_copy_outputs_ms = vk_stats.copy_outputs_ms;
      context.last_stats.vk_total_backend_ms = vk_stats.total_backend_ms;
      context.last_stats.vk_gpu_shader_total_ms = vk_stats.gpu_shader_total_ms;
      context.last_stats.vk_gpu_shader_dispatch_count =
          vk_stats.gpu_shader_dispatch_count;
    }
  }

  py::list get_outputs_for_context(
      const std::string& method_name,
      MethodContext& context,
      bool clone_outputs) {
    if (!context.outputs_owner) {
      throw std::runtime_error(
          "No outputs available for method '" + method_name + "'");
    }
    return outputs_to_py_list(context.outputs_owner, clone_outputs);
  }

  void ensure_open() const {
    if (!module_) {
      throw std::runtime_error("SessionHandle is closed");
    }
  }

  std::string model_path_;
  std::optional<std::string> data_map_path_;
  std::shared_ptr<Module> module_;
  std::unordered_map<std::string, MethodContext> method_contexts_;
};

py::list get_registered_backend_names() {
  py::list out;
  size_t num_backends = get_num_registered_backends();
  for (size_t i = 0; i < num_backends; ++i) {
    auto name_result = get_backend_name(i);
    THROW_IF_ERROR(
        name_result.error(),
        "Failed to query backend name at index %zu, error=0x%" PRIx32,
        i,
        static_cast<uint32_t>(name_result.error()));
    out.append(name_result.get());
  }
  return out;
}

py::bool_ is_backend_available(const std::string& backend_name) {
  BackendInterface* backend = get_backend_class(backend_name.c_str());
  if (backend == nullptr) {
    return false;
  }
  return backend->is_available();
}

} // namespace

PYBIND11_MODULE(_exshader_runtime, m) {
  py::enum_<Program::Verification>(m, "Verification")
      .value("Minimal", Program::Verification::Minimal)
      .value("InternalConsistency", Program::Verification::InternalConsistency);

  py::class_<PyExTensor>(m, "ExTensor")
      .def("sizes", &PyExTensor::sizes)
      .def("dtype", &PyExTensor::dtype)
      .def("nbytes", &PyExTensor::nbytes)
      .def("numel", &PyExTensor::numel)
      .def("argmax", &PyExTensor::argmax)
      .def("argmax_last_dim_row0", &PyExTensor::argmax_last_dim_row0)
      .def(
          "sample_top_p_row0",
          &PyExTensor::sample_top_p_row0,
          py::arg("temperature"),
          py::arg("top_p"),
          py::arg("seed") = py::none())
      .def("set_int64_scalar", &PyExTensor::set_int64_scalar, py::arg("value"))
      .def(
          "set_int64_flat",
          &PyExTensor::set_int64_flat,
          py::arg("index"),
          py::arg("value"))
      .def(
          "set_int64_row0_prefix",
          &PyExTensor::set_int64_row0_prefix,
          py::arg("values"))
      .def("row0_prefix", &PyExTensor::row0_prefix, py::arg("cols"))
      .def("tolist", &PyExTensor::tolist, py::arg("max_elems") = py::none())
      .def("__repr__", &PyExTensor::repr);

  py::class_<SessionHandle>(m, "SessionHandle")
      .def_static(
          "load",
          &SessionHandle::load,
          py::arg("model_path"),
          py::arg("data_map_path") = std::nullopt,
          py::arg("program_verification") = Program::Verification::Minimal)
      .def("close", &SessionHandle::close)
      .def("alloc_int64", &SessionHandle::alloc_int64, py::arg("sizes"))
      .def("method_names", &SessionHandle::method_names)
      .def("method_meta", &SessionHandle::method_meta, py::arg("method_name"))
      .def(
          "set_inputs",
          &SessionHandle::set_inputs,
          py::arg("method_name"),
          py::arg("inputs"))
      .def("execute", &SessionHandle::execute, py::arg("method_name"))
      .def(
          "get_outputs",
          &SessionHandle::get_outputs,
          py::arg("method_name"),
          py::arg("clone_outputs") = false)
      .def(
          "run",
          &SessionHandle::run,
          py::arg("method_name"),
          py::arg("inputs"),
          py::arg("clone_outputs") = false)
      .def(
          "run_with_stats",
          &SessionHandle::run_with_stats,
          py::arg("method_name"),
          py::arg("inputs"),
          py::arg("clone_outputs") = false)
      .def(
          "get_last_run_stats",
          &SessionHandle::get_last_run_stats,
          py::arg("method_name"));

  m.def("_get_registered_backend_names", &get_registered_backend_names);
  m.def("_is_available", &is_backend_available, py::arg("backend_name"));
}
