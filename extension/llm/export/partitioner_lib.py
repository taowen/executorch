# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Any, Dict, List, Optional


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
      "operator_name_blocklist": ["sdpa_with_kv_cache.default"]
    }
    """
    raw = os.environ.get("ET_VULKAN_PARTITIONER_PROFILE")
    if not raw:
        return {}

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("ET_VULKAN_PARTITIONER_PROFILE must be a JSON object")
    return parsed


def get_xnnpack_partitioner(dynamic_quant_only_partitioner: bool = True):
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
        operator_name_blocklist=profile.get("operator_name_blocklist"),
        operator_name_allowlist=profile.get("operator_name_allowlist"),
        nn_module_blocklist=profile.get("nn_module_blocklist"),
        nn_module_allowlist=profile.get("nn_module_allowlist"),
    )


def get_mps_partitioner(use_kv_cache: bool = False):
    raise _unsupported_backend("MPS")


def get_openvino_partitioner(device: str):
    raise _unsupported_backend("OpenVINO")


def get_coreml_partitioner(
    ios: int = 15,
    embedding_quantize: Optional[str] = None,
    pt2e_quantize: Optional[str] = None,
    coreml_quantize: Optional[str] = None,
    coreml_compute_units: Optional[str] = None,
):
    raise _unsupported_backend("CoreML")


def get_qnn_partitioner(
    use_kv_cache: bool = False,
    pt2e_quantize: Optional[str] = None,
    num_sharding: int = 0,
    soc_model: str = "SM8650",
):
    raise _unsupported_backend("QNN")


def get_tosa_partitioner(version: str):
    raise _unsupported_backend("TOSA")


def get_ethosu_partitioner(target: str):
    raise _unsupported_backend("Ethos-U")


def get_vgf_partitioner(
    compile_spec: Optional[str], compiler_flags: Optional[List[str]]
):
    raise _unsupported_backend("VGF")
