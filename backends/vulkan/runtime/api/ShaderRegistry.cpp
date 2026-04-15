/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/ShaderRegistry.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <utility>

namespace vkcompute {
namespace api {

namespace {

constexpr size_t kBundleShaderFieldCount = 11;
constexpr size_t kBundleDispatchFieldCount = 4;

std::string make_error(
    const std::string& message,
    const std::string& line = std::string()) {
  if (line.empty()) {
    return message;
  }
  return message + ": " + line;
}

std::vector<std::string> split_tsv(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, '\t')) {
    fields.push_back(field);
  }
  if (!line.empty() && line.back() == '\t') {
    fields.emplace_back("");
  }
  return fields;
}

std::vector<std::string> split_csv(const std::string& text) {
  std::vector<std::string> fields;
  if (text.empty()) {
    return fields;
  }
  std::stringstream ss(text);
  std::string field;
  while (std::getline(ss, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

bool parse_bool(const std::string& text, bool* out) {
  if (text == "1" || text == "true" || text == "TRUE") {
    *out = true;
    return true;
  }
  if (text == "0" || text == "false" || text == "FALSE") {
    *out = false;
    return true;
  }
  return false;
}

bool parse_u32(const std::string& text, uint32_t* out) {
  try {
    const unsigned long value = std::stoul(text);
    *out = static_cast<uint32_t>(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_tile_size(const std::string& text, utils::uvec3* out) {
  std::vector<std::string> parts = split_csv(text);
  if (parts.empty()) {
    *out = {1u, 1u, 1u};
    return true;
  }
  if (parts.size() != 3) {
    return false;
  }
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
  if (!parse_u32(parts[0], &x) || !parse_u32(parts[1], &y) ||
      !parse_u32(parts[2], &z)) {
    return false;
  }
  *out = {x, y, z};
  return true;
}

bool parse_layout_signature(
    const std::string& text,
    vkapi::ShaderLayout::Signature* signature) {
  signature->clear();
  for (const std::string& token : split_csv(text)) {
    if (token.empty()) {
      continue;
    }
    if (token == "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE") {
      signature->push_back(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    } else if (token == "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER") {
      signature->push_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    } else if (token == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER") {
      signature->push_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    } else if (token == "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER") {
      signature->push_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    } else {
      return false;
    }
  }
  return true;
}

bool parse_dispatch_key(const std::string& text, DispatchKey* key) {
  std::string upper = text;
  std::transform(
      upper.begin(),
      upper.end(),
      upper.begin(),
      [](const unsigned char c) { return std::toupper(c); });
  if (upper == "CATCHALL") {
    *key = DispatchKey::CATCHALL;
    return true;
  }
  if (upper == "ADRENO") {
    *key = DispatchKey::ADRENO;
    return true;
  }
  if (upper == "MALI") {
    *key = DispatchKey::MALI;
    return true;
  }
  if (upper == "OVERRIDE") {
    *key = DispatchKey::OVERRIDE;
    return true;
  }
  return false;
}

std::string join_path(const std::string& base, const std::string& relative) {
  if (relative.empty()) {
    return base;
  }
  if (!relative.empty() && relative[0] == '/') {
    return relative;
  }
  if (base.empty()) {
    return relative;
  }
  if (base.back() == '/') {
    return base + relative;
  }
  return base + "/" + relative;
}

bool read_spirv_binary(
    const std::string& path,
    std::vector<uint32_t>* spirv,
    std::string* error_msg) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs.good()) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to open SPIR-V file: " + path;
    }
    return false;
  }
  const std::streamsize file_size = ifs.tellg();
  if (file_size <= 0 || (file_size % 4) != 0) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid SPIR-V file size for: " + path;
    }
    return false;
  }
  spirv->resize(static_cast<size_t>(file_size / 4));
  ifs.seekg(0, std::ios::beg);
  if (!ifs.read(reinterpret_cast<char*>(spirv->data()), file_size)) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to read SPIR-V file: " + path;
    }
    return false;
  }
  return true;
}

} // namespace

bool ShaderRegistry::has_shader(const std::string& shader_name) {
  const ShaderListing::const_iterator it = listings_.find(shader_name);
  return it != listings_.end();
}

bool ShaderRegistry::has_dispatch(const std::string& op_name) {
  const Registry::const_iterator it = registry_.find(op_name);
  return it != registry_.end();
}

void ShaderRegistry::register_shader(vkapi::ShaderInfo&& shader_info) {
  if (has_shader(shader_info.kernel_name)) {
    VK_THROW(
        "Shader with name ", shader_info.kernel_name, "already registered");
  }
  listings_.emplace(shader_info.kernel_name, std::move(shader_info));
}

void ShaderRegistry::upsert_shader(
    vkapi::ShaderInfo&& shader_info,
    std::vector<uint32_t>&& spirv_binary) {
  const std::string shader_name = shader_info.kernel_name;
  owned_shader_binaries_[shader_name] = std::move(spirv_binary);
  const std::vector<uint32_t>& owned_binary = owned_shader_binaries_.at(shader_name);
  shader_info.src_code.bin = owned_binary.data();
  shader_info.src_code.size =
      static_cast<uint32_t>(owned_binary.size() * sizeof(uint32_t));

  listings_[shader_name] = std::move(shader_info);
}

void ShaderRegistry::register_op_dispatch(
    const std::string& op_name,
    const DispatchKey key,
    const std::string& shader_name) {
  if (!has_dispatch(op_name)) {
    registry_.emplace(op_name, Dispatcher());
  }
  const Dispatcher::const_iterator it = registry_[op_name].find(key);
  if (it != registry_[op_name].end()) {
    registry_[op_name][key] = shader_name;
  } else {
    registry_[op_name].emplace(key, shader_name);
  }
}

const vkapi::ShaderInfo& ShaderRegistry::get_shader_info(
    const std::string& shader_name) {
  const ShaderListing::const_iterator it = listings_.find(shader_name);

  VK_CHECK_COND(
      it != listings_.end(),
      "Could not find ShaderInfo with name ",
      shader_name);

  return it->second;
}

bool ShaderRegistry::load_bundle(
    const std::string& bundle_dir,
    std::string* error_msg) {
  const std::string manifest_path = join_path(bundle_dir, "bundle.tsv");
  std::ifstream manifest(manifest_path);
  if (!manifest.good()) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to open manifest: " + manifest_path;
    }
    return false;
  }

  std::string line;
  if (!std::getline(manifest, line)) {
    if (error_msg != nullptr) {
      *error_msg = "Empty manifest: " + manifest_path;
    }
    return false;
  }
  if (line != "ETVK_SHADER_BUNDLE_V1") {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported shader bundle manifest version: " + line;
    }
    return false;
  }

  size_t line_no = 1;
  while (std::getline(manifest, line)) {
    ++line_no;
    if (line.empty() || line[0] == '#') {
      continue;
    }

    const std::vector<std::string> fields = split_tsv(line);
    if (fields.empty()) {
      continue;
    }

    if (fields[0] == "shader") {
      if (fields.size() != kBundleShaderFieldCount) {
        if (error_msg != nullptr) {
          *error_msg = make_error(
              "Invalid shader record at line " + std::to_string(line_no), line);
        }
        return false;
      }

      vkapi::ShaderLayout::Signature layout;
      if (!parse_layout_signature(fields[3], &layout)) {
        if (error_msg != nullptr) {
          *error_msg = make_error(
              "Invalid shader layout at line " + std::to_string(line_no), fields[3]);
        }
        return false;
      }

      utils::uvec3 tile_size{1u, 1u, 1u};
      if (!parse_tile_size(fields[4], &tile_size)) {
        if (error_msg != nullptr) {
          *error_msg = make_error(
              "Invalid shader tile size at line " + std::to_string(line_no),
              fields[4]);
        }
        return false;
      }

      bool req_i16 = false;
      bool req_16bit = false;
      bool req_8bit = false;
      bool req_dot = false;
      bool req_i64 = false;
      bool req_f64 = false;
      if (!parse_bool(fields[5], &req_i16) ||
          !parse_bool(fields[6], &req_16bit) ||
          !parse_bool(fields[7], &req_8bit) ||
          !parse_bool(fields[8], &req_dot) ||
          !parse_bool(fields[9], &req_i64) ||
          !parse_bool(fields[10], &req_f64)) {
        if (error_msg != nullptr) {
          *error_msg = make_error(
              "Invalid shader requirement flags at line " +
                  std::to_string(line_no),
              line);
        }
        return false;
      }

      std::vector<uint32_t> spirv_binary;
      std::string read_error;
      const std::string spirv_path = join_path(bundle_dir, fields[2]);
      if (!read_spirv_binary(spirv_path, &spirv_binary, &read_error)) {
        if (error_msg != nullptr) {
          *error_msg = read_error;
        }
        return false;
      }

      upsert_shader(
          vkapi::ShaderInfo(
              fields[1],
              nullptr,
              0u,
              std::move(layout),
              tile_size,
              req_i16,
              req_16bit,
              req_8bit,
              req_dot,
              req_i64,
              req_f64),
          std::move(spirv_binary));
      continue;
    }

    if (fields[0] == "dispatch") {
      if (fields.size() != kBundleDispatchFieldCount) {
        if (error_msg != nullptr) {
          *error_msg = make_error(
              "Invalid dispatch record at line " + std::to_string(line_no), line);
        }
        return false;
      }
      DispatchKey key;
      if (!parse_dispatch_key(fields[2], &key)) {
        if (error_msg != nullptr) {
          *error_msg = make_error(
              "Invalid dispatch key at line " + std::to_string(line_no), fields[2]);
        }
        return false;
      }
      register_op_dispatch(fields[1], key, fields[3]);
      continue;
    }

    if (error_msg != nullptr) {
      *error_msg = make_error(
          "Unsupported bundle record type at line " + std::to_string(line_no),
          fields[0]);
    }
    return false;
  }

  return true;
}

ShaderRegistry& shader_registry() {
  static ShaderRegistry registry;
  return registry;
}

} // namespace api
} // namespace vkcompute
