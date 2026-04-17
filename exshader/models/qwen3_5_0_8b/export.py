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
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.examples.models.qwen3_5 import Qwen3_5Model, convert_weights
from executorch.extension.llm.export.builder import DType, LLMEdgeManager

# qwen3.5-0.8B is validated on static-shape export. Dynamic export currently
# hits a token_dim constraint violation during torch.export.
ENABLE_DYNAMIC_SHAPE = False
VULKAN_FORCE_FP16 = True


def _resolve_checkpoint(checkpoint: str | None) -> str:
    if checkpoint:
        return str(Path(checkpoint).expanduser().resolve())
    return download_and_convert_hf_checkpoint(
        "Qwen/Qwen3.5-0.8B",
        convert_weights,
    )


def _build_edge_manager(
    *,
    checkpoint_path: str,
    params_path: str,
    dtype_override: DType,
    max_seq_length: int,
    max_context_length: int,
    use_sdpa_with_kv_cache: bool,
) -> LLMEdgeManager:
    model_wrapper = Qwen3_5Model(
        model_class="qwen3_5_0_8b",
        checkpoint=checkpoint_path,
        params=params_path,
        use_kv_cache=True,
        use_sdpa_with_kv_cache=use_sdpa_with_kv_cache,
        enable_dynamic_shape=ENABLE_DYNAMIC_SHAPE,
        max_seq_length=max_seq_length,
        max_context_length=max_context_length,
        dtype_override=dtype_override.value,
    )
    model = model_wrapper.get_eager_model()
    metadata = {
        "get_max_seq_len": model.max_seq_len,
        "get_max_context_len": model.max_context_len,
        "get_n_layers": model.n_layers,
        "get_vocab_size": model.vocab_size,
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": use_sdpa_with_kv_cache,
        "enable_dynamic_shape": ENABLE_DYNAMIC_SHAPE,
    }
    if getattr(model, "num_kv_shared_layers", 0) > 0:
        metadata["get_num_kv_shared_layers"] = model.num_kv_shared_layers

    return LLMEdgeManager(
        model=model,
        modelname="qwen3_5_0_8b",
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


def _source_transforms(*, use_sdpa_with_kv_cache: bool):
    transforms = []
    if use_sdpa_with_kv_cache:
        transforms.append(replace_kv_cache_with_custom_kv_cache)
        transforms.append(replace_sdpa_with_custom_op)
    return transforms


def export_qwen3_5_0_8b(
    *,
    params: str,
    checkpoint: str | None,
    output: str,
    max_seq_length: int,
    max_context_length: int,
    use_sdpa_with_kv_cache: bool,
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
        max_seq_length=max_seq_length,
        max_context_length=max_context_length,
        use_sdpa_with_kv_cache=use_sdpa_with_kv_cache,
    )
    builder_manager = builder_manager.set_output_dir(output_dir_path).source_transform(
        _source_transforms(use_sdpa_with_kv_cache=use_sdpa_with_kv_cache)
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
        default="examples/models/qwen3_5/config/0_8b_config.json",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--output",
        default="artifacts/pte/qwen3_5_0_8b_vulkan_fp32_statefix_main.pte",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-context-length", type=int, default=2048)
    parser.add_argument(
        "--use-sdpa-with-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = export_qwen3_5_0_8b(
        params=args.params,
        checkpoint=args.checkpoint,
        output=args.output,
        max_seq_length=args.max_seq_length,
        max_context_length=args.max_context_length,
        use_sdpa_with_kv_cache=args.use_sdpa_with_kv_cache,
    )
    print(output)


if __name__ == "__main__":
    main()
