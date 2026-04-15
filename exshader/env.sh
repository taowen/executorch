#!/usr/bin/env bash
set -euo pipefail

# Single source of truth for local artifact paths.
# Usage:
#   source exshader/env.sh
#   echo "$ET_ARTIFACTS_ROOT" "$ET_PTE_DIR" "$ET_SHADER_BUNDLE_DIR" "$ET_BUILD_DIR"

export ET_ARTIFACTS_ROOT="${ET_ARTIFACTS_ROOT:-$PWD/artifacts}"
export ET_PTE_DIR="${ET_PTE_DIR:-$ET_ARTIFACTS_ROOT/pte}"
export ET_SHADER_BUNDLE_DIR="${ET_SHADER_BUNDLE_DIR:-$ET_ARTIFACTS_ROOT/shader_bundles/vk_bundle_m3}"

# Keep one canonical cmake build tree for Linux + Vulkan.
export ET_BUILD_DIR="${ET_BUILD_DIR:-$PWD/cmake-out-linux-vulkan}"

mkdir -p "$ET_ARTIFACTS_ROOT" "$ET_PTE_DIR" "$ET_SHADER_BUNDLE_DIR"
