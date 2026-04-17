from __future__ import annotations

import argparse
from pathlib import Path

from executorch.examples.models.llama.export_llama_lib import (
    _to_edge_and_lower_llama,
    canonical_path,
    get_output_filename_from_args,
)
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
    get_quant_weight_transform,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.examples.models.qwen3 import Qwen3Model, convert_weights
from executorch.extension.llm.export.builder import DType, LLMEdgeManager

ENABLE_DYNAMIC_SHAPE = True
VULKAN_FORCE_FP16 = True


def _resolve_checkpoint(checkpoint: str | None) -> str:
    if checkpoint:
        return str(Path(checkpoint).expanduser().resolve())
    return download_and_convert_hf_checkpoint(
        "Qwen/Qwen3-0.6B",
        convert_weights,
    )


def _build_edge_manager(
    *,
    checkpoint_path: str,
    params_path: str,
    dtype_override: DType,
) -> LLMEdgeManager:
    model_wrapper = Qwen3Model(
        model_class="qwen3_0_6b",
        checkpoint=checkpoint_path,
        params=params_path,
        use_kv_cache=True,
        use_sdpa_with_kv_cache=True,
        enable_dynamic_shape=ENABLE_DYNAMIC_SHAPE,
        max_seq_length=128,
        max_context_length=128,
        dtype_override=dtype_override.value,
    )
    model = model_wrapper.get_eager_model()
    metadata = {
        "get_max_seq_len": model.max_seq_len,
        "get_max_context_len": model.max_context_len,
        "get_n_layers": model.n_layers,
        "get_vocab_size": model.vocab_size,
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": True,
        "enable_dynamic_shape": ENABLE_DYNAMIC_SHAPE,
    }
    if getattr(model, "num_kv_shared_layers", 0) > 0:
        metadata["get_num_kv_shared_layers"] = model.num_kv_shared_layers

    return LLMEdgeManager(
        model=model,
        modelname="qwen3_0_6b",
        max_seq_len=model.max_seq_len,
        dtype=dtype_override,
        use_kv_cache=True,
        generate_full_logits=False,
        example_inputs=model_wrapper.get_example_inputs(),
        example_kwarg_inputs=None,
        dynamic_shapes=None,
        enable_dynamic_shape=ENABLE_DYNAMIC_SHAPE,
        calibration_tasks=None,
        calibration_limit=None,
        calibration_seq_length=None,
        calibration_data="Once upon a time",
        tokenizer_path=None,
        save_exported_program=False,
        verbose=False,
        metadata=metadata,
    )


def _source_transforms(*, checkpoint_path: str, checkpoint_dtype: DType):
    return [
        get_quant_embedding_transform("4,32", False, quantize_with_hqq=True),
        get_quant_weight_transform(
            quantization_mode="8da4w",
            group_size=None,
            computation_dtype=DType.fp32,
            checkpoint_dtype=checkpoint_dtype,
            checkpoint_path=checkpoint_path,
            tokenizer_path=None,
            calibration_tasks=None,
            calibration_limit=None,
            calibration_seq_length=None,
            quantize_with_hqq=True,
        ),
        replace_kv_cache_with_custom_kv_cache,
        replace_sdpa_with_custom_op,
    ]


def export_qwen3_0_6b(
    *,
    params: str,
    checkpoint: str | None,
    output: str,
) -> str:
    resolved_output = str(Path(output).expanduser().resolve())
    resolved_params = str(Path(params).expanduser().resolve())
    resolved_checkpoint = _resolve_checkpoint(checkpoint)

    quantizers = []

    checkpoint_path = canonical_path(resolved_checkpoint)
    params_path = canonical_path(resolved_params)
    output_dir_path = canonical_path(".", dir=True)
    dtype_override = DType.fp32

    builder_manager = _build_edge_manager(
        checkpoint_path=checkpoint_path,
        params_path=params_path,
        dtype_override=dtype_override,
    )
    checkpoint_dtype = DType.from_torch_dtype(builder_manager.model.checkpoint_dtype)  # type: ignore[arg-type]

    builder_manager = builder_manager.set_output_dir(output_dir_path).source_transform(
        _source_transforms(
            checkpoint_path=checkpoint_path,
            checkpoint_dtype=checkpoint_dtype,
        )
    )
    builder_manager.model = builder_manager.model.to(
        dtype=dtype_override.to_torch_dtype()
    )

    builder_exported = builder_manager.export()
    builder_exported.run_canonical_optimizations()
    modelname = builder_exported.modelname

    builder = _to_edge_and_lower_llama(
        builder_exported,
        modelname,
        [],
        quantizers,
        dtype_override=dtype_override.value,
        enable_dynamic_shape=ENABLE_DYNAMIC_SHAPE,
        vulkan_force_fp16=VULKAN_FORCE_FP16,
        generate_etrecord=False,
        verbose=False,
    )

    output_file = get_output_filename_from_args(
        output_name=resolved_output,
        modelname=modelname,
        output_dir=builder.output_dir,
        dtype=builder.dtype,
    )
    builder.save_to_pte(output_file)
    return output_file


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = export_qwen3_0_6b(
        params=args.params,
        checkpoint=args.checkpoint,
        output=args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
