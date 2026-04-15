# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export an LLM with ExecuTorch. Currently follows the following steps:
1. Instantiate our custom PyTorch transformer definition from examples/llama/models/llama_transformer.py.
2. Load weights into the model.
3. Apply source transformations/TorchAO quantization.
4. Export model to intermediate IRs.
5. Graph transformations/PT2E quantization.
6. Partition graph and delegate to backend(s).
7. Export to final ExecuTorch .pte format.

Example usage using full CLI arguments:
python -m exshader.export_llm \
    base.model_class="llama3" \
    model.use_sdpa_with_kv_cache=True \
    model.use_kv_cache=True \
    debug.verbose=True \
    backend.xnnpack.enabled=True \
    backend.xnnpack.extended_ops=True \
    quantization.qmode="8da4w"

Example usage using config file:
python -m exshader.export_llm \
    --config example_llm_config.yaml
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import hydra

from executorch.extension.llm.export.config.llm_config import LlmConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

cs = ConfigStore.instance()
cs.store(name="llm_config", node=LlmConfig)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "CMakeLists.txt").exists() and (parent / "src" / "executorch").is_dir():
            return parent
    raise RuntimeError(f"Cannot locate repo root from {here}")


def _resolve_build_dir(repo_root: Path) -> Path:
    env_build_dir = os.environ.get("ET_BUILD_DIR")
    if env_build_dir:
        return Path(env_build_dir).expanduser().resolve()
    return repo_root / "cmake-out-linux-vulkan"


def _assert_module_from_repo(module_name: str, repo_root: Path) -> None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Cannot resolve required module: {module_name}")
    origin = Path(spec.origin).resolve()
    if repo_root not in origin.parents and origin != repo_root:
        raise RuntimeError(
            f"{module_name} resolved to '{origin}', not local repo '{repo_root}'. "
            "Run: `uv pip install -e .` to use the current workspace source."
        )


def _assert_custom_ops_aot_lib_ready(repo_root: Path) -> None:
    env_path = os.environ.get("EXECUTORCH_CUSTOM_OPS_AOT_LIB")
    if env_path:
        if not Path(env_path).expanduser().resolve().exists():
            raise RuntimeError(
                f"EXECUTORCH_CUSTOM_OPS_AOT_LIB is set but file does not exist: {env_path}"
            )
        return

    build_dir = _resolve_build_dir(repo_root)
    candidates = [
        build_dir / "extension" / "llm" / "custom_ops" / "libcustom_ops_aot_lib.so",
        build_dir / "libcustom_ops_aot_lib.so",
    ]
    if any(path.exists() for path in candidates):
        return

    raise RuntimeError(
        "Missing libcustom_ops_aot_lib.so. Build it before export:\n"
        "  source exshader/env.sh\n"
        "  cmake -S . -B \"$ET_BUILD_DIR\" "
        "-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON "
        "-DEXECUTORCH_BUILD_KERNELS_LLM=ON "
        "-DEXECUTORCH_BUILD_KERNELS_LLM_AOT=ON\n"
        "  cmake --build \"$ET_BUILD_DIR\" --target custom_ops_aot_lib -j$(nproc)"
    )


def _validate_export_environment() -> None:
    repo_root = _repo_root()
    for module_name in (
        "executorch.extension.llm.export.partitioner_lib",
        "executorch.examples.models.llama.export_llama_lib",
        "executorch.backends.vulkan.partitioner.vulkan_partitioner",
    ):
        _assert_module_from_repo(module_name, repo_root)
    _assert_custom_ops_aot_lib_ready(repo_root)


def parse_config_arg() -> Tuple[str, List[Any]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", type=str, help="Path to the LlmConfig file")
    args, remaining = parser.parse_known_args()
    return args.config, remaining


def pop_config_arg() -> str:
    """
    Removes '--config' and its value from sys.argv.
    Assumes --config is specified and argparse has already validated the args.
    Returns the config file path.
    """
    idx = sys.argv.index("--config")
    value = sys.argv[idx + 1]
    del sys.argv[idx : idx + 2]
    return value


def add_hydra_config_args(config_file_path: str) -> None:
    """
    Breaks down the config file path into directory and filename,
    resolves the directory to an absolute path, and adds the
    --config_path and --config_name arguments to sys.argv.
    """
    config_dir = os.path.dirname(config_file_path)
    config_name = os.path.basename(config_file_path)

    # Resolve to absolute path
    config_dir_abs = os.path.abspath(config_dir)

    # Add the hydra config arguments to sys.argv
    sys.argv.extend(["--config-path", config_dir_abs, "--config-name", config_name])


@hydra.main(version_base=None, config_name="llm_config", config_path=None)
def hydra_main(llm_config: LlmConfig) -> None:
    _validate_export_environment()
    from executorch.examples.models.llama.export_llama_lib import export_llama

    structured = OmegaConf.structured(LlmConfig)
    merged = OmegaConf.merge(structured, llm_config)
    llm_config_obj = OmegaConf.to_object(merged)
    export_llama(llm_config_obj)


def main() -> None:
    # First parse out the arg for whether to use Hydra or the old CLI.
    config, remaining_args = parse_config_arg()
    if config:
        # Pop out --config and its value so that they are not parsed by
        # Hydra's main.
        config_file_path = pop_config_arg()

        # Add hydra config_path and config_name arguments to sys.argv.
        add_hydra_config_args(config_file_path)

    hydra_main()


if __name__ == "__main__":
    main()
