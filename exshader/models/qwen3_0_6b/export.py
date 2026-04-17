from __future__ import annotations

import argparse
from pathlib import Path

from executorch.examples.models.llama.export_llama_lib import export_llama
from executorch.extension.llm.export.config.llm_config import DtypeOverride, LlmConfig, ModelType


def build_config(args: argparse.Namespace) -> LlmConfig:
    cfg = LlmConfig()
    cfg.base.model_class = ModelType.qwen3_0_6b
    cfg.base.params = str(Path(args.params).expanduser().resolve())
    if args.checkpoint:
        cfg.base.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    cfg.model.enable_dynamic_shape = args.enable_dynamic_shape
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    cfg.model.quantize_kv_cache = False
    cfg.model.dtype_override = DtypeOverride.fp32
    cfg.backend.vulkan.enabled = True
    cfg.backend.vulkan.force_fp16 = args.vulkan_force_fp16
    cfg.quantization.qmode = "8da4w"
    cfg.quantization.embedding_quantize = "4,32"
    cfg.export.output_name = str(Path(args.output).expanduser().resolve())
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        default="examples/models/qwen3/config/0_6b_config.json",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--output",
        default="artifacts/pte/qwen3_0_6b_vulkan_pure_candidate.pte",
    )
    parser.add_argument(
        "--enable-dynamic-shape",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vulkan-force-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = export_llama(build_config(args))
    print(output)


if __name__ == "__main__":
    main()
