# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Any, Dict, Optional

_PROFILE_ALLOWED_KEYS = {
    "compile_options",
    "operator_blocklist",
    "operator_allowlist",
    "nn_module_blocklist",
    "nn_module_allowlist",
}

_PROFILE_LIST_KEYS = {
    "operator_blocklist",
    "operator_allowlist",
    "nn_module_blocklist",
    "nn_module_allowlist",
}

_COMPILE_OPTIONS_ALLOWED_KEYS = {
    "buffer_limit",
    "texture_limits",
    "require_dynamic_shapes",
    "skip_bool_tensors",
    "skip_tag_memory_metadata",
    "skip_memory_planning",
    "small_texture_limits",
    "downcast_64_bit",
    "force_fp16",
    "disable_fuse_patterns",
    "disable_fuse_quantized_ops",
    "shader_bundle_path",
}


# This workspace is intentionally focused on a Linux + Vulkan export/runtime path.
# Keep the public function names to avoid breaking imports, but fail fast for
# non-Vulkan backends that are out of scope for this branch.
def _unsupported_backend(name: str) -> RuntimeError:
    return RuntimeError(
        f"{name} backend is removed in this pure-Vulkan branch. "
        "Use backend.vulkan.enabled=true."
    )


def _load_vulkan_profile_from_env() -> Dict[str, Any]:
    """Load Vulkan partition/fusion profile from environment.

    ET_VULKAN_PARTITIONER_PROFILE example:
    {
      "compile_options": {
        "buffer_limit": 134217728,
        "texture_limits": [16384, 16384, 2048],
        "disable_fuse_patterns": false,
        "disable_fuse_quantized_ops": false
      },
      "operator_blocklist": ["sdpa_with_kv_cache.default"]
    }
    """
    raw = os.environ.get("ET_VULKAN_PARTITIONER_PROFILE")
    if not raw:
        return {}

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("ET_VULKAN_PARTITIONER_PROFILE must be a JSON object")
    _validate_vulkan_profile(parsed)
    return parsed


def _validate_vulkan_profile(profile: Dict[str, Any]) -> None:
    unknown_top = set(profile.keys()) - _PROFILE_ALLOWED_KEYS
    if unknown_top:
        raise ValueError(
            "ET_VULKAN_PARTITIONER_PROFILE has unsupported key(s): "
            f"{sorted(unknown_top)}. Allowed keys: {sorted(_PROFILE_ALLOWED_KEYS)}"
        )

    for key in _PROFILE_LIST_KEYS:
        value = profile.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or not all(
            isinstance(entry, str) for entry in value
        ):
            raise ValueError(f"'{key}' must be a list of strings")

    compile_options = profile.get("compile_options")
    if compile_options is None:
        return
    if not isinstance(compile_options, dict):
        raise ValueError("'compile_options' must be a JSON object")

    unknown_compile_options = (
        set(compile_options.keys()) - _COMPILE_OPTIONS_ALLOWED_KEYS
    )
    if unknown_compile_options:
        raise ValueError(
            "'compile_options' has unsupported key(s): "
            f"{sorted(unknown_compile_options)}. Allowed keys: "
            f"{sorted(_COMPILE_OPTIONS_ALLOWED_KEYS)}"
        )

    bool_keys = {
        "require_dynamic_shapes",
        "skip_bool_tensors",
        "skip_tag_memory_metadata",
        "skip_memory_planning",
        "small_texture_limits",
        "downcast_64_bit",
        "force_fp16",
        "disable_fuse_patterns",
        "disable_fuse_quantized_ops",
    }
    for key in bool_keys:
        value = compile_options.get(key)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise ValueError(f"'compile_options.{key}' must be a boolean")

    buffer_limit = compile_options.get("buffer_limit")
    if buffer_limit is not None and not isinstance(buffer_limit, int):
        raise ValueError("'compile_options.buffer_limit' must be an integer")

    texture_limits = compile_options.get("texture_limits")
    if texture_limits is not None:
        if (
            not isinstance(texture_limits, list)
            or len(texture_limits) != 3
            or not all(isinstance(v, int) for v in texture_limits)
        ):
            raise ValueError(
                "'compile_options.texture_limits' must be a list of 3 integers"
            )

    shader_bundle_path = compile_options.get("shader_bundle_path")
    if shader_bundle_path is not None and not isinstance(shader_bundle_path, str):
        raise ValueError("'compile_options.shader_bundle_path' must be a string")


def get_xnnpack_partitioner(*args, **kwargs):
    raise _unsupported_backend("XNNPACK")


def get_vulkan_partitioner(
    dtype_override: Optional[str] = None,
    enable_dynamic_shape: bool = False,
    force_fp16: bool = False,
):
    assert (
        dtype_override == "fp32" or dtype_override is None
    ), "Vulkan backend does not support non fp32 dtypes at the moment"

    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )

    profile = _load_vulkan_profile_from_env()
    compile_options: Dict[str, Any] = {
        "require_dynamic_shapes": enable_dynamic_shape,
        "force_fp16": force_fp16,
    }
    compile_options.update(profile.get("compile_options", {}))

    return VulkanPartitioner(
        compile_options=compile_options,
        operator_blocklist=profile.get("operator_blocklist"),
        operator_allowlist=profile.get("operator_allowlist"),
        nn_module_blocklist=profile.get("nn_module_blocklist"),
        nn_module_allowlist=profile.get("nn_module_allowlist"),
    )


def get_mps_partitioner(*args, **kwargs):
    raise _unsupported_backend("MPS")


def get_openvino_partitioner(*args, **kwargs):
    raise _unsupported_backend("OpenVINO")


def get_coreml_partitioner(*args, **kwargs):
    raise _unsupported_backend("CoreML")


def get_qnn_partitioner(*args, **kwargs):
    raise _unsupported_backend("QNN")


def get_tosa_partitioner(*args, **kwargs):
    raise _unsupported_backend("TOSA")


def get_ethosu_partitioner(*args, **kwargs):
    raise _unsupported_backend("Ethos-U")


def get_vgf_partitioner(*args, **kwargs):
    raise _unsupported_backend("VGF")
