#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build an external Vulkan shader bundle for runtime hot-reload.

The output directory will contain:
- generated .glsl/.h files
- compiled .spv files
- bundle.tsv manifest consumed by ShaderRegistry::load_bundle()
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Iterable


def _load_gen_vulkan_spv_module(repo_root: Path):
    module_path = repo_root / "backends" / "vulkan" / "runtime" / "gen_vulkan_spv.py"
    spec = importlib.util.spec_from_file_location("gen_vulkan_spv", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_env_overrides(items: Iterable[str] | None) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --env item '{item}'. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Vulkan shader runtime bundle")
    parser.add_argument(
        "--glsl-paths",
        nargs="+",
        required=True,
        help="One or more shader source directories containing .glsl/.yaml templates",
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="Output directory for generated shaders, SPIR-V, and bundle.tsv",
    )
    parser.add_argument(
        "--glslc-path",
        default="glslc",
        help="Path to glslc compiler (default: glslc from PATH)",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Cache directory for incremental compilation",
    )
    parser.add_argument(
        "--env",
        metavar="KEY=VALUE",
        nargs="*",
        help="Extra codegen environment entries",
    )
    parser.add_argument("--force-rebuild", action="store_true", default=False)
    parser.add_argument("--replace-u16vecn", action="store_true", default=False)
    parser.add_argument(
        "--optimize", action="store_true", help="Pass -O to glslc", default=False
    )
    parser.add_argument(
        "--optimize-size",
        action="store_true",
        help="Pass -Os to glslc",
        default=False,
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=-1,
        help="Number of threads for SPIR-V compilation. -1 uses all cores.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    gen = _load_gen_vulkan_spv_module(repo_root)

    bundle_dir = Path(args.bundle_dir).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = None
    if args.tmp_dir is not None:
        tmp_dir = Path(args.tmp_dir).resolve()
    else:
        tmp_dir = bundle_dir / ".cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env: Dict[Any, Any] = dict(gen.DEFAULT_ENV)
    env.update(gen.TYPE_MAPPINGS)
    env.update(gen.UTILITY_FNS)
    env.update(_parse_env_overrides(args.env))

    glslc_flags = ""
    if args.optimize:
        glslc_flags += " -O"
    if args.optimize_size:
        glslc_flags += " -Os"

    spv_generator = gen.SPVGenerator(
        src_dir_paths=[str(Path(p).resolve()) for p in args.glsl_paths],
        env=env,
        glslc_path=args.glslc_path,
        glslc_flags=glslc_flags,
        replace_u16vecn=args.replace_u16vecn,
    )

    spv_to_glsl_map = spv_generator.generateSPV(
        output_dir=str(bundle_dir),
        cache_dir=str(tmp_dir),
        force_rebuild=args.force_rebuild,
        nthreads=args.nthreads,
    )

    manifest_path = bundle_dir / "bundle.tsv"
    shader_lines = []
    dispatch_lines = []

    # Some generator entries are metadata/includes and may not have an SPV path.
    # Filter them before sorting so Python never compares None vs str.
    valid_entries = [
        (spv_path, glsl_path)
        for spv_path, glsl_path in spv_to_glsl_map.items()
        if spv_path is not None
    ]
    for spv_path, glsl_path in sorted(valid_entries, key=lambda kv: str(kv[0])):
        if spv_path is None:
            continue
        if not str(spv_path).endswith(".spv"):
            continue

        shader_name = gen.getName(spv_path).replace("_spv", "")
        shader_info = gen.getShaderInfo(glsl_path)

        tile = shader_info.tile_size if len(shader_info.tile_size) == 3 else [1, 1, 1]
        layouts = ",".join(shader_info.layouts)
        tile_str = f"{tile[0]},{tile[1]},{tile[2]}"
        spv_relpath = os.path.relpath(spv_path, bundle_dir)

        shader_lines.append(
            "\t".join(
                [
                    "shader",
                    shader_name,
                    spv_relpath,
                    layouts,
                    tile_str,
                    "1" if shader_info.requires_shader_int16_ext else "0",
                    "1" if shader_info.requires_16bit_storage_ext else "0",
                    "1" if shader_info.requires_8bit_storage_ext else "0",
                    "1" if shader_info.requires_integer_dot_product_ext else "0",
                    "1" if shader_info.requires_shader_int64_ext else "0",
                    "1" if shader_info.requires_shader_float64_ext else "0",
                ]
            )
        )

        if shader_info.register_for is not None:
            op_name, keys = shader_info.register_for
            for key in keys:
                dispatch_lines.append(
                    "\t".join(["dispatch", op_name, key.upper(), shader_name])
                )

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("ETVK_SHADER_BUNDLE_V1\n")
        for line in shader_lines:
            f.write(line + "\n")
        for line in dispatch_lines:
            f.write(line + "\n")

    print(f"[et_shader_jit] Wrote bundle manifest: {manifest_path}")
    print(f"[et_shader_jit] Shader entries: {len(shader_lines)}")
    print(f"[et_shader_jit] Dispatch entries: {len(dispatch_lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
