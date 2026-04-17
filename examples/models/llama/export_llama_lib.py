# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Example script for exporting Llama2 to flatbuffer

import argparse
import copy
import json
import logging
import re
import shlex
from functools import partial
from importlib import resources as _resources
from json import JSONDecodeError
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
from executorch.devtools.backend_debug import print_delegation_info
from executorch.devtools.etrecord import generate_etrecord as generate_etrecord_func
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass
from executorch.extension.llm.export.builder import DType, LLMEdgeManager
from executorch.extension.llm.export.config.llm_config import LlmConfig
from executorch.extension.llm.export.partitioner_lib import (
    get_vulkan_partitioner,
)
from executorch.extension.llm.export.quantizer_lib import (
    get_pt2e_quantization_params,
    get_pt2e_quantizers,
    get_vulkan_quantizer,
)
from executorch.util.activation_memory_profiler import generate_memory_trace
from omegaconf import DictConfig
from torch.export import ExportedProgram

from ..model_factory import EagerModelFactory
from .source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
    replace_kv_cache_with_quantized_kv_cache,
    replace_kv_cache_with_ring_kv_cache,
)
from .source_transformation.quantize import (
    get_quant_embedding_transform,
    get_quant_weight_transform,
)
from .source_transformation.rope import materialze_broadcast_of_rope_freq_cis
from .source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
    replace_sdpa_with_quantized_sdpa,
)

IS_FBCODE = True  #  os.environ.get("FBCODE_PLATFORM", False)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

pkg_name = __name__
verbosity_setting = None


# All models that leverage the transformer architecture defined in llama_transformer.py.
EXECUTORCH_DEFINED_MODELS = [
    "stories110m",
    "llama2",
    "llama3",
    "llama3_1",
    "llama3_2",
    "static_llama",
    "gemma3_1b",
    "qwen2_5_0_5b",
    "qwen2_5_1_5b",
    "qwen2_5_coder_32b",
    "qwen3_0_6b",
    "qwen3_1_7b",
    "qwen3_4b",
    "qwen3_5_0_8b",
    "qwen3_5_2b",
    "qwen3_5_4b",
    "phi_4_mini",
    "smollm2",
    "lfm2_350m",  # hybrid
    "lfm2_700m",  # hybrid
    "lfm2_1_2b",  # hybrid
    "lfm2_5_1_2b",  # hybrid
]
TORCHTUNE_DEFINED_MODELS = ["llama3_2_vision"]
HUGGING_FACE_REPO_IDS = {
    "gemma3_1b": "google/gemma-3-1b-it",
    "qwen2_5_0_5b": "Qwen/Qwen2.5-0.5B",
    "qwen2_5_1_5b": "Qwen/Qwen2.5-1.5B",
    "qwen2_5_coder_32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "phi_4_mini": "microsoft/Phi-4-mini-instruct",
    "smollm2": "HuggingFaceTB/SmolLM-135M",
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_1_7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_5_0_8b": "Qwen/Qwen3.5-0.8B",
    "qwen3_5_2b": "Qwen/Qwen3.5-2B",
    "qwen3_5_4b": "Qwen/Qwen3.5-4B",
    "lfm2_350m": "LiquidAI/LFM2-350M",
    "lfm2_700m": "LiquidAI/LFM2-700M",
    "lfm2_1_2b": "LiquidAI/LFM2-1.2B",
    "lfm2_5_1_2b": "LiquidAI/LFM2.5-1.2B-Instruct",
}


def set_pkg_name(name: str) -> None:
    global pkg_name
    pkg_name = name


def get_resource_path(resource_name) -> str:
    return str(_resources.files(pkg_name).joinpath(resource_name))


def set_verbosity(val):
    global verbosity_setting
    verbosity_setting = val


def verbose_export():
    return verbosity_setting


def build_model(
    model: str,
    checkpoint: str,
    params: str,
    output_dir: Optional[str] = ".",
    extra_opts: Optional[str] = "",
) -> str:
    argString = f"--model {model} --checkpoint {checkpoint} --params {params} {extra_opts} --output-dir {output_dir}"
    parser = build_args_parser()
    args = parser.parse_args(shlex.split(argString))
    llm_config = LlmConfig.from_args(args)
    return export_llama(llm_config)


def parse_list_of_ints(s):
    import ast

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and all(isinstance(i, int) for i in parsed):
            print(parsed)
            return parsed
        raise argparse.ArgumentTypeError(
            "Must be a list of integers, e.g., [0, 16, 0, 16]"
        )
    except Exception:
        raise argparse.ArgumentTypeError(
            "Must be a list of integers, e.g., [0, 16, 0, 16]"
        )


def build_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", default=".", help="output directory")
    # parser.add_argument(
    #     "-q", "--quantized_ckpt", default=None, help="quantized checkpoint file"
    # )
    parser.add_argument(
        "--model",
        default="llama3",
        choices=EXECUTORCH_DEFINED_MODELS + TORCHTUNE_DEFINED_MODELS,
        help="The Llama model to export. stories110M, llama2, llama3, llama3_1, and llama3_2 use the same underlying LlamaTransformer architecture defined in ExecuTorch. All other models use TorchTune model definitions.",
    )
    parser.add_argument(
        "-E",
        "--embedding-quantize",
        default=None,
        type=str,
        help="type of embedding quantization, '<bitwidth>,<groupsize>', e.g., '8,1024'.",
    )
    parser.add_argument(
        "--use_shared_embedding",
        action="store_true",
        help="Whether the embedding/unembedding weights should be shared.  Only available with torchao kernels.",
    )
    parser.add_argument(
        "--pt2e_quantize",
        default=None,
        choices=[
            "vulkan_8w",
        ],
        help="Use PT2E quantization. Supported option in this branch: vulkan_8w.",
    )

    parser.add_argument(
        "-qmode",
        "--quantization_mode",
        type=_qmode_type,
        default=None,
        help="type of quantization",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        required=False,
        help="Path to the checkpoint .pth file. When not provided, the model will be initialized with random weights.",
    )

    parser.add_argument(
        "--adapter_checkpoint",
        required=False,
        help="Path to the adapter.pt file from torchtune. Used if the model has trained LoRA adapters. Must provide adapter_config.json",
    )

    parser.add_argument(
        "--adapter_config",
        required=False,
        help="Path to the adapter_config.json file. Used if the model has trained LoRA adapters. Must provide adapter_checkpoint.",
    )


    parser.add_argument(
        "--calibration_tasks",
        nargs="+",
        type=str,
        default=None,
        help="Tasks for GPTQ calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=None,
        help="number of samples used for calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_seq_length",
        type=int,
        default=None,
        help="Sequence length for GPTQ calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_data",
        type=str,
        default="Once upon a time",
        help="Calibration prompts from users",
    )
    parser.add_argument(
        "-t",
        "--tokenizer_path",
        default=None,
        help="tokenizer path (Note: .model not .bin)",
    )
    parser.add_argument(
        "-kv",
        "--use_kv_cache",
        default=False,
        action="store_true",
        help="Whether or not to export a model using kv cache",
    )
    parser.add_argument(
        "--quantize_kv_cache",
        default=False,
        action="store_true",
        help="Whether or not to export a model using int8 per token quantized kv cache",
    )
    parser.add_argument(
        "--use_sdpa_with_kv_cache",
        default=False,
        action="store_true",
        help="Whether to use sdpa_with_kv_cache update op when using kv cache",
    )
    parser.add_argument(
        "--disable_dynamic_shape",
        dest="enable_dynamic_shape",
        default=True,  # Enable this by default
        action="store_false",
        help="Enable dynamic shape along seq dim. Used for faster prefill",
    )
    parser.add_argument(
        "-p",
        "--params",
        required=False,
        help="Config file for model parameters. When not provided, the model will fallback on default values defined in examples/models/llama/model_args.py.",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        default=None,
        help='metadata string in json format. Example {"key": 1, "key2": "value2"}',
    )
    parser.add_argument(
        "-s",
        "--so_library",
        default=None,
        required=False,
        help="shared library for quantized operators",
    )
    parser.add_argument(
        "--profile_memory",
        required=False,
        action="store_true",
        help="Generate chrome trace of activation memory for intermediate tensors.",
    )
    parser.add_argument(
        "-prof",
        "--profile_path",
        default=None,
        help="Use cProfile to profile model export. Results saved to profile_path as a html file.",
    )
    parser.add_argument(
        "-G",
        "--group_size",
        type=int,
        default=None,
        help="group_size for weight quantization",
    )

    parser.add_argument(
        "-d",
        "--dtype-override",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        help="Provide the dtype of the model. This must match up with the supported dtypes of the backends that you are using."
        "Please be aware that only some backends support fp16 and bf16.",
    )

    parser.add_argument(
        "-n",
        "--output_name",
        default=None,
        help="Override the output filename of the saved pte model file.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="maximum length sequence to evaluate",
    )

    parser.add_argument(
        "--max_context_length",
        type=int,
        default=128,
        help="maximum length of context for model to remember",
    )

    parser.add_argument(
        "--local_global_attention",
        type=parse_list_of_ints,
        default=None,
        help="List of integers specifying local and global attention pattern, e.g., [0, 16, 0, 16] to specify that every other layer is sliding window of 16."
        " [0, 16, 32] pattern specifes 2nd and 3rd layer has sliding window of 16 and 32 respectively."
        " [16] pattern specifies all layers have sliding window of 16.",
    )

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--use-torchao-kernels",
        action="store_true",
        help="Delegate tied-embedding and quantized linear ops to torchao kernels",
    )
    parser.add_argument(
        "--use-torchao-kernels-tied-embedding",
        action="store_true",
        help="Delegate tied-embedding ops to torchao kernels",
    )
    parser.add_argument(
        "--use-torchao-kernels-linear",
        action="store_true",
        help="Delegate linear ops to torchao kernels",
    )
    parser.add_argument("-V", "--vulkan", action="store_true")
    parser.add_argument("--vulkan-force-fp16", action="store_true")

    parser.add_argument(
        "--expand_rope_table",
        default=False,
        action="store_true",
        help="[Temp workaround] Expand sin/cos table in head dim to take vectorized path in optimized kernels.",
    )

    parser.add_argument(
        "--generate_etrecord",
        action="store_true",
        required=False,
        default=False,
        help="Generate the ETRecord debug artifact.",
    )

    parser.add_argument(
        "--generate_full_logits",
        action="store_true",
        required=False,
        default=False,
        help="Generate logits for all inputs.",
    )

    parser.add_argument(
        "--soc_model",
        help="[QNN backend] SoC model of current device. e.g. 'SM8650' for Snapdragon 8 Gen 3.",
        type=str,
        required=False,
        default="SM8650",
    )

    parser.add_argument(
        "-sq",
        "--use_spin_quant",
        type=str,
        default=None,
        choices=["cuda", "native"],
        help="Use SpinQuant for better quantization performance. Only support cuda and native.",
    )

    parser.add_argument(
        "-qat",
        "--use_qat",
        default=False,
        action="store_true",
        help="Whether the checkpoint is pre-quantized with QAT or not.",
    )

    parser.add_argument(
        "-lora",
        "--use_lora",
        type=int,
        default=0,
        help="Whether the checkpoint contains LoRA adaptors or not. 0: no LoRA adaptors; "
        "otherwise, it means the rank of LoRA adaptors. Currently it only works if QAT is enabled.",
    )

    parser.add_argument(
        "--preq_mode",
        type=str,
        default=None,
        choices=["8da4w", "8da4w_output_8da8w"],
        help="Quantization mode used for pre-quantized checkpoint. Only support 8da4w and 8da4w_output_8da8w right now.",
    )

    parser.add_argument(
        "--preq_group_size",
        type=int,
        default=32,
        help="group_size for pre-quantized checkpoint weight quantization",
    )

    parser.add_argument(
        "--preq_embedding_quantize",
        default="8,0",
        type=str,
        help="type of embedding quantization for pre-quantized checkpoint, '<bitwidth>,<groupsize>', e.g., '8,1024'.",
    )

    parser.add_argument(
        "--use_attention_sink",
        default=None,
        type=str,
        help="Use attention sink to have fluent multi-round conversation. '<sink_size>,<window_size>', e.g., '4,2044'.",
    )

    parser.add_argument(
        "--output_prune_map",
        default=None,
        help="path to the output pruning token mapping file (token_map.json)",
    )

    parser.add_argument(
        "--input_prune_map",
        default=None,
        help="path to the input pruning token mapping file (token_map.json)",
    )

    parser.add_argument(
        "--export_only",
        default=False,
        action="store_true",
        help="If true, stops right after torch.export() and saves the exported model.",
    )
    return parser


def canonical_path(path: Union[str, Path], *, dir: bool = False) -> str:
    path = str(path)

    if verbose_export():
        print(f"creating canonical path for {path}")

    if not path.startswith("par:"):
        return path

    if not IS_FBCODE:
        print("not FBCODE")
        return path[4:]
    else:
        return_val = str(_resources.files(pkg_name).joinpath(path[4:]))
        if verbose_export():
            print(f"canonical name is: {return_val}")
        return return_val


def export_llama(  # noqa: C901
    export_options: Union[argparse.Namespace, LlmConfig, DictConfig],
) -> str:
    if isinstance(export_options, argparse.Namespace):
        # Legacy CLI.
        llm_config = LlmConfig.from_args(export_options)
    elif isinstance(export_options, LlmConfig) or isinstance(
        export_options, DictConfig
    ):
        # Hydra CLI.
        llm_config = export_options
    else:
        raise ValueError(
            "Input to export_llama must be either of type argparse.Namespace or LlmConfig"
        )

    # If a checkpoint isn't provided for an HF OSS model, download and convert the
    # weights first.
    model_name = llm_config.base.model_class.value
    if not llm_config.base.checkpoint and model_name in HUGGING_FACE_REPO_IDS:
        repo_id = HUGGING_FACE_REPO_IDS[model_name]
        if model_name.startswith("qwen2_5"):
            from executorch.examples.models.qwen2_5 import convert_weights
        elif model_name == "gemma3_1b":
            from executorch.examples.models.gemma3 import convert_weights
        elif model_name.startswith("qwen3_5"):
            from executorch.examples.models.qwen3_5 import convert_weights
        elif model_name.startswith("qwen3"):
            from executorch.examples.models.qwen3 import convert_weights
        elif model_name == "phi_4_mini":
            from executorch.examples.models.phi_4_mini import convert_weights
        elif model_name == "smollm2":
            from executorch.examples.models.smollm2 import convert_weights
        elif model_name.startswith("lfm2"):
            from executorch.examples.models.lfm2 import convert_weights
        else:
            raise ValueError(
                f"Converting weights to meta format for {model_name} is not yet supported"
            )
        checkpoint = download_and_convert_hf_checkpoint(repo_id, convert_weights)
        llm_config.base.checkpoint = checkpoint

    if llm_config.debug.profile_path is not None:
        try:
            from executorch.util.python_profiler import CProfilerFlameGraph

            with CProfilerFlameGraph(llm_config.debug.profile_path):
                builder = _export_llama(llm_config)
                assert (
                    filename := builder.get_saved_pte_filename()
                ) is not None, "Fail to get file name from builder"
                return filename
        except ImportError:
            print(
                "Please run `pip install snakeviz` to install required dependencies for cProfiler flamegraph."
            )
            return ""
    else:
        builder = _export_llama(llm_config)
        assert (
            filename := builder.get_saved_pte_filename()
        ) is not None, "Fail to get file name from builder"
        return filename


def _prepare_for_llama_export(llm_config: LlmConfig) -> LLMEdgeManager:
    """
    Helper function for export_llama. Loads the model from checkpoint and params,
    and sets up a LLMEdgeManager with initial transforms and dtype conversion.

    Returns a LLMEdgeManager prior to calling export_to_edge with quantizers
    """
    # load model from checkpoint and params.json
    checkpoint_path = (
        canonical_path(llm_config.base.checkpoint)
        if llm_config.base.checkpoint
        else None
    )
    params_path = (
        canonical_path(llm_config.base.params) if llm_config.base.params else None
    )
    output_dir_path = canonical_path(llm_config.export.output_dir, dir=True)

    llm_config.base.checkpoint = checkpoint_path
    llm_config.base.params = params_path
    llm_config.export.output_dir = output_dir_path

    # Convert dtype override string to actual type.
    dtype_override = DType[llm_config.model.dtype_override.value]

    edge_manager = _load_llama_model(llm_config)

    # At this point, the model is loaded in the default fp32.

    # Checkpoint dtype should be lower or equal precision to the dtype override.
    checkpoint_dtype = edge_manager.model.checkpoint_dtype
    if not (
        checkpoint_dtype == dtype_override.to_torch_dtype()
        or (
            checkpoint_dtype == torch.float16
            and dtype_override.to_torch_dtype() == torch.float32
        )
        or (
            checkpoint_dtype == torch.bfloat16
            and dtype_override.to_torch_dtype() == torch.float32
        )
    ):
        logging.warning(
            f"Checkpoint dtype {checkpoint_dtype} precision is higher than dtype override {dtype_override.to_torch_dtype()}."
        )

    # Quantize weights in checkpoint dtype for accuracy, then cast to
    # dtype_override afterward. IntxUnpackedToInt8Tensor.to() properly
    # propagates the dtype change to scale/zero_point/output dtype.
    logging.info(f"Checkpoint dtype: {edge_manager.model.checkpoint_dtype}")
    edge_manager = edge_manager.set_output_dir(output_dir_path).source_transform(
        _get_source_transforms(
            dtype_override=dtype_override,
            checkpoint=llm_config.base.checkpoint,
            checkpoint_dtype=DType.from_torch_dtype(checkpoint_dtype),  # type: ignore
            tokenizer_path=llm_config.base.tokenizer_path,
            use_spin_quant=(
                llm_config.quantization.use_spin_quant.value
                if llm_config.quantization.use_spin_quant
                else None
            ),
            embedding_quantize=llm_config.quantization.embedding_quantize,
            use_shared_embedding=llm_config.model.use_shared_embedding,
            quantization_mode=llm_config.quantization.qmode,
            group_size=llm_config.quantization.group_size,
            calibration_tasks=llm_config.quantization.calibration_tasks,
            calibration_limit=llm_config.quantization.calibration_limit,
            calibration_seq_length=llm_config.quantization.calibration_seq_length,
            expand_rope_table=llm_config.model.expand_rope_table,
            use_custom_sdpa_with_attention_mask=getattr(
                llm_config.model, "use_custom_sdpa_with_attention_mask", False
            ),
            use_sdpa_with_kv_cache=llm_config.model.use_sdpa_with_kv_cache,
            quantize_kv_cache=llm_config.model.quantize_kv_cache,
            use_kv_cache=llm_config.model.use_kv_cache,
            use_qat=llm_config.quantization.use_qat,
            use_lora=llm_config.base.use_lora,
            preq_mode=(
                llm_config.base.preq_mode.value if llm_config.base.preq_mode else None
            ),
            preq_group_size=llm_config.base.preq_group_size,
            preq_embedding_quantize=llm_config.base.preq_embedding_quantize,
            local_global_attention=llm_config.model.local_global_attention,
            use_torchao_kernels_linear=llm_config.backend.torchao.use_torchao_kernels_linear,
            use_torchao_kernels_tied_embedding=llm_config.backend.torchao.use_torchao_kernels_tied_embedding,
            quantize_with_hqq=llm_config.quantization.use_hqq,
        )
    )

    # Now cast to the dtype override after quantization, so non-quantized
    # components use the desired computation dtype.
    edge_manager.model = edge_manager.model.to(dtype=dtype_override.to_torch_dtype())

    return edge_manager


def get_quantizer_and_quant_params(llm_config):
    pt2e_quant_params = get_pt2e_quantization_params(
        (
            llm_config.quantization.pt2e_quantize.value
            if llm_config.quantization.pt2e_quantize
            else None
        ),
        llm_config.quantization.qmode,
    )
    quantizers = get_pt2e_quantizers(pt2e_quant_params, llm_config.export.so_library)
    quant_dtype = None
    if llm_config.backend.vulkan.enabled and llm_config.quantization.pt2e_quantize:
        assert (
            len(quantizers) == 0
        ), "Should not enable both vulkan and other quantizers"
        vulkan_quantizer = get_vulkan_quantizer(
            llm_config.quantization.pt2e_quantize.value
        )
        quantizers.append(vulkan_quantizer)
    logging.info(f"Applying quantizers: {quantizers}")
    return pt2e_quant_params, quantizers, quant_dtype


def _qmode_type(value):
    choices = ["int8", "8da4w", "8da4w-gptq", "4w"]
    patterns = [r"torchao:8da(\d+)w", r"torchao:fpa(\d+)w"]

    if value in choices:
        return value

    for pattern in patterns:
        matches = re.findall(pattern, value)
        if len(matches) == 1:
            return value

    raise argparse.ArgumentTypeError(
        f"Got qmode {value}, but expected one of {choices}, or one of the regex patterns {patterns}."
    )


def _validate_args(llm_config):
    if llm_config.export.max_context_length < llm_config.export.max_seq_length:
        raise ValueError(
            f"max_context_length {llm_config.export.max_context_length} must be >= max_seq_len {llm_config.export.max_seq_length}. max_context_length impacts kv cache size that is used to remember history, while max_seq_length refers to user prompt length. Please use --max_context_length to specify context length."
        )
    if not llm_config.backend.vulkan.enabled:
        raise ValueError(
            "This branch only supports Vulkan export. Set backend.vulkan.enabled=true."
        )

    non_vulkan_enabled = []
    for name, cfg in vars(llm_config.backend).items():
        if name == "vulkan":
            continue
        if hasattr(cfg, "enabled") and getattr(cfg, "enabled"):
            non_vulkan_enabled.append(name)

    if non_vulkan_enabled:
        raise ValueError(
            "Only Vulkan backend is supported in this branch. "
            f"Disable: {', '.join(non_vulkan_enabled)}"
        )

    if llm_config.model.use_shared_embedding:
        if not (
            llm_config.quantization.embedding_quantize is not None
            and llm_config.quantization.embedding_quantize.startswith("torchao:")
        ):
            raise ValueError(
                "Shared embedding is only supported with torchao quantization."
            )

    if llm_config.multimethod.enabled:
        if llm_config.base.lora_config is not None:
            raise ValueError(
                "Cannot use both base.lora_config and multimethod.methods. "
                "Use multimethod.methods for all LoRA variants."
            )
        if llm_config.quantization.pt2e_quantize is not None:
            raise ValueError(
                "PT2E quantization is not supported with multimethod export."
            )


def _to_edge_and_lower_llama(
    builder_exported,
    modelname,
    additional_passes,
    quantizers,
    dtype_override: str = "fp32",
    enable_dynamic_shape: bool = True,
    vulkan_force_fp16: bool = False,
    generate_etrecord: bool = False,
    verbose: bool = False,
):
    builder_exported_to_edge = builder_exported.pt2e_quantize(
        quantizers
    ).export_to_edge()

    partitioners = [
        get_vulkan_partitioner(
            dtype_override,
            enable_dynamic_shape,
            vulkan_force_fp16,
        )
    ]
    modelname = f"vulkan_{modelname}"

    logging.info("Lowering model using following partitioner(s): ")
    for partitioner in partitioners:
        logging.info(f"--> {partitioner.__class__.__name__}")

    if generate_etrecord:
        if not builder_exported_to_edge.edge_manager:
            raise ValueError("Unable to generate etrecord due to missing edge manager.")

        logging.info("Generating etrecord")
        # Copy the edge manager which will be serialized into etrecord. This is memory-wise expensive.
        edge_manager_copy = copy.deepcopy(builder_exported_to_edge.edge_manager)
        builder = builder_exported_to_edge.to_backend(partitioners)
        if verbose:
            print_delegation_info(builder.edge_manager.exported_program().graph_module)
        builder = builder.to_executorch(
            passes=additional_passes,
        )

        # Generate ETRecord
        if edge_manager_copy:
            generate_etrecord_func(
                et_record="etrecord.bin",
                edge_dialect_program=edge_manager_copy,
                executorch_program=builder.export_program,
            )
            logging.info("Generated etrecord.bin")
    else:
        builder = builder_exported_to_edge.to_backend(partitioners)
        if verbose:
            print_delegation_info(builder.edge_manager.exported_program().graph_module)
        builder = builder.to_executorch(passes=additional_passes)

    return builder


def _get_multimethod_partitioners(llm_config: LlmConfig) -> Optional[List[Partitioner]]:
    return [
        get_vulkan_partitioner(
            llm_config.model.dtype_override.value,
            llm_config.model.enable_dynamic_shape,
            llm_config.backend.vulkan.force_fp16,
        )
    ]


def _get_output_filename(
    llm_config: LlmConfig, modelname: str, output_dir: str, dtype: DType
) -> str:
    """Determine output filename for the .pte file."""
    if dtype == DType.fp16:
        modelname = f"{modelname}_h"

    if llm_config.export.output_name:
        output_name = llm_config.export.output_name
        if output_name.endswith(".pte"):
            return output_name
        else:
            return f"{output_dir}/{output_name}.pte"
    else:
        return f"{output_dir}/{modelname}.pte"


def _export_llama_multimethod(llm_config: LlmConfig) -> LLMEdgeManager:
    """
    Export multiple methods (base + LoRA variants) to a single .pte file.

    For each method in llm_config.multimethod.methods:
    - If LoraConfig is None: use base model
    - If LoraConfig is provided: create model with LoRA weights

    Limitations:
    - This branch lowers multimethod export with Vulkan partitioner only.
    - PT2E quantization is not supported.
    - Each method is exported separately; export time scales linearly
      with the number of methods.
    - The final .pte file deduplicates shared weights automatically.
    """
    num_methods = len(llm_config.multimethod.methods)
    logging.info(
        f"multimethod export: exporting {num_methods} method(s). "
        "Each method requires separate model instantiation and export."
    )

    additional_passes = []
    if llm_config.base.model_class.value in TORCHTUNE_DEFINED_MODELS:
        additional_passes = [InitializedMutableBufferPass(["kv_cache_pos"])]

    # Build dict of exported programs
    method_to_program: Dict[str, ExportedProgram] = {}
    first_builder = None

    for method in llm_config.multimethod.methods:
        logging.info(f"Exporting method: {method.method_name}")

        # Create a copy of config with this method's LoRA setting
        method_config = copy.deepcopy(llm_config)
        method_config.base.lora_config = method.lora_config
        # Disable multimethod to avoid infinite recursion
        method_config.multimethod.methods = []

        # Load and prepare model for this method
        builder = _prepare_for_llama_export(method_config)
        builder = builder.export()
        builder.run_canonical_optimizations()

        # Get the exported program
        exported_program = builder._export(builder.pre_autograd_graph_module)
        method_to_program[method.method_name] = exported_program

        if first_builder is None:
            first_builder = builder

    assert first_builder is not None, "No methods to export"

    # Get partitioners based on backend config
    partitioners = _get_multimethod_partitioners(llm_config)

    # Lower all methods together using multimethod API
    edge_config = first_builder._get_edge_config()
    edge_manager = to_edge_transform_and_lower(
        method_to_program,
        partitioner=partitioners,
        compile_config=edge_config,
        constant_methods=first_builder.metadata,
        generate_etrecord=llm_config.debug.generate_etrecord,
    )

    # Convert to executorch and save
    first_builder.edge_manager = edge_manager
    first_builder = first_builder.to_executorch(
        passes=additional_passes,
        share_mutable_buffers=llm_config.multimethod.share_mutable_buffers,
    )

    output_file = _get_output_filename(
        llm_config,
        first_builder.modelname,
        first_builder.output_dir,
        first_builder.dtype,
    )
    first_builder.save_to_pte(output_file)

    return first_builder


def _export_llama(llm_config: LlmConfig) -> LLMEdgeManager:  # noqa: C901
    _validate_args(llm_config)

    # Check for multimethod export
    if llm_config.multimethod.enabled:
        return _export_llama_multimethod(llm_config)

    _, quantizers, _ = get_quantizer_and_quant_params(llm_config)

    additional_passes = []
    if llm_config.base.model_class.value in TORCHTUNE_DEFINED_MODELS:
        additional_passes = [InitializedMutableBufferPass(["kv_cache_pos"])]

    # export_to_edge
    builder_manager = _prepare_for_llama_export(llm_config)
    builder_exported = builder_manager.export()
    builder_exported.run_canonical_optimizations()
    modelname = builder_exported.modelname

    if llm_config.export.export_only:
        exit()

    builder = _to_edge_and_lower_llama(
        builder_exported,
        modelname,
        additional_passes,
        quantizers,
        dtype_override=llm_config.model.dtype_override.value,
        enable_dynamic_shape=llm_config.model.enable_dynamic_shape,
        vulkan_force_fp16=llm_config.backend.vulkan.force_fp16,
        generate_etrecord=llm_config.debug.generate_etrecord,
        verbose=llm_config.debug.verbose,
    )

    if llm_config.debug.profile_memory:
        generate_memory_trace(builder.export_program, "memory_profile.json")

    output_file = _get_output_filename(
        llm_config,
        modelname,
        builder.output_dir,
        builder.dtype,
    )
    builder.save_to_pte(output_file)
    return builder


def _load_llama_model_metadata(
    use_kv_cache: bool,
    use_sdpa_with_kv_cache: bool,
    enable_dynamic_shape: bool,
    max_seq_len: int,
    max_context_len: int,
    n_layers: int,
    vocab_size: int,
    metadata_str: Optional[str] = None,
    num_kv_shared_layers: int = 0,
):
    metadata = {
        "get_max_seq_len": max_seq_len,
        "get_max_context_len": max_context_len,
        "get_n_layers": n_layers,
        "get_vocab_size": vocab_size,
        "use_kv_cache": use_kv_cache,
        "use_sdpa_with_kv_cache": use_sdpa_with_kv_cache,
        "enable_dynamic_shape": enable_dynamic_shape,
    }
    # YOCO (You Only Cache Once) KV sharing metadata
    if num_kv_shared_layers > 0:
        metadata["get_num_kv_shared_layers"] = num_kv_shared_layers
    if metadata_str:
        try:
            extra = json.loads(metadata_str)
            for k, v in extra.items():
                metadata[k] = v
        except JSONDecodeError:
            logging.error("Invalid metadata, should be a valid JSON string")
    return metadata


def _load_llama_model(llm_config: LlmConfig) -> "LLMEdgeManager":
    """
    A helper util that builds a Llama2 model. It returns a LLMEdgeManager that
    can help further lower the model to ExecuTorch.
    Returns:
        An instance of LLMEdgeManager which contains the eager mode model.
    """

    modelname = llm_config.base.model_class.value
    if modelname in EXECUTORCH_DEFINED_MODELS:
        module_name = "llama"
        model_class_name = "Llama2Model"  # TODO: Change to "LlamaModel" in examples/models/llama/model.py.
    elif modelname in TORCHTUNE_DEFINED_MODELS:
        if modelname == "llama3_2_vision":
            module_name = "llama3_2_vision"
            model_class_name = "Llama3_2Decoder"
        else:
            raise ValueError(f"{modelname} is not a valid Llama model.")
    else:
        raise ValueError(f"{modelname} is not a valid Llama model.")

    (
        model,
        example_inputs,
        example_kwarg_inputs,
        dynamic_shapes,
    ) = EagerModelFactory.create_model(
        module_name,
        model_class_name,
        llm_config=llm_config,
    )
    # Convert dtype override string to actual type.
    dtype_override = DType[llm_config.model.dtype_override.value]

    return LLMEdgeManager(
        model=model,
        modelname=modelname,
        max_seq_len=model.max_seq_len,  # type: ignore
        dtype=dtype_override,
        use_kv_cache=llm_config.model.use_kv_cache,
        generate_full_logits=llm_config.debug.generate_full_logits,
        example_inputs=example_inputs,
        example_kwarg_inputs=example_kwarg_inputs,
        dynamic_shapes=dynamic_shapes,
        enable_dynamic_shape=llm_config.model.enable_dynamic_shape,
        calibration_tasks=llm_config.quantization.calibration_tasks,
        calibration_limit=llm_config.quantization.calibration_limit,
        calibration_seq_length=llm_config.quantization.calibration_seq_length,
        calibration_data=llm_config.quantization.calibration_data,
        tokenizer_path=llm_config.base.tokenizer_path,
        save_exported_program=llm_config.export.export_only,
        verbose=llm_config.debug.verbose,
        metadata=_load_llama_model_metadata(
            llm_config.model.use_kv_cache,
            llm_config.model.use_sdpa_with_kv_cache,
            llm_config.model.enable_dynamic_shape,
            # pyre-fixme[6]: For 5th argument expected `ModelArgs` but got
            #  `Union[Tensor, Module]`.
            model.max_seq_len,
            # pyre-fixme[6]: For 6th argument expected `ModelArgs` but got
            #  `Union[Tensor, Module]`.
            model.max_context_len,
            # pyre-fixme[6]: For 7th argument expected `int` but got `Union[Tensor,
            #  Module]`.
            model.n_layers,
            # pyre-fixme[6]: For 8th argument expected `int` but got `Union[Tensor,
            #  Module]`.
            model.vocab_size,
            llm_config.base.metadata,
            # pyre-fixme[6]: For 10th argument expected `int` but got `Union[Tensor,
            #  Module]`.
            num_kv_shared_layers=getattr(model, "num_kv_shared_layers", 0),
        ),
    )


def _get_source_transforms(  # noqa
    dtype_override: DType,
    *,
    checkpoint: Optional[str] = None,
    checkpoint_dtype: Optional[DType] = None,
    tokenizer_path: Optional[str] = None,
    use_spin_quant: Optional[str] = None,
    embedding_quantize: Optional[str] = None,
    use_shared_embedding: bool = False,
    quantization_mode: Optional[str] = None,
    group_size: Optional[int] = None,
    calibration_tasks: Optional[List[str]] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    expand_rope_table: bool = False,
    use_custom_sdpa_with_attention_mask: bool = False,
    use_sdpa_with_kv_cache: bool = False,
    quantize_kv_cache: bool = False,
    use_kv_cache: bool = False,
    use_qat: bool = False,
    use_lora: int = 0,
    preq_mode: Optional[str] = None,
    preq_group_size: Optional[int] = None,
    preq_embedding_quantize: Optional[str] = None,
    local_global_attention: Optional[List[int]] = None,
    use_torchao_kernels_linear: bool = False,
    use_torchao_kernels_tied_embedding: bool = False,
    quantize_with_hqq: bool = True,
) -> List[Callable[[torch.nn.Module], torch.nn.Module]]:
    """
    Return a list of functions that transform a graph.

    Args:
        dtype_override: The dtype to use for the model.
        checkpoint: Path to the checkpoint file.
        checkpoint_dtype: The dtype of the checkpoint. At the moment, if this is specified,
            it means that you want to run quantize transformations on the weights represented
            in their original dtype, while the overall dtype of the model maybe something
            different. If not specified, defaults to dtype_override.
        tokenizer_path: Path to the tokenizer file.
        use_spin_quant: Type of spin quant to use ("cuda" or "native").
        embedding_quantize: Type of embedding quantization.
        quantization_mode: Type of quantization mode.
        expand_rope_table: Whether to expand rope table.
        use_custom_sdpa_with_attention_mask: Whether to use custom SDPA with attention mask.
        use_sdpa_with_kv_cache: Whether to use SDPA with KV cache.
        quantize_kv_cache: Whether to quantize KV cache.
        use_kv_cache: Whether to use KV cache.
        use_shared_embedding: Whether to use shared embedding.
        use_qat: Whether to use QAT.
        use_lora: LoRA rank (0 means no LoRA).
        preq_mode: Pre-quantization mode.
        preq_group_size: Pre-quantization group size.
        preq_embedding_quantize: Pre-quantization embedding quantize.

    Returns:
        A list of transformation functions.
    """

    if not checkpoint_dtype:
        checkpoint_dtype = dtype_override

    transforms = []

    if use_spin_quant:
        if use_spin_quant == "cuda":
            from .source_transformation.spin_quant import (
                inject_fast_hadamard_transform_cuda_for_spin_quant,
            )

            transforms.append(inject_fast_hadamard_transform_cuda_for_spin_quant)
        elif use_spin_quant == "native":
            from .source_transformation.spin_quant import (
                inject_fast_hadamard_transform_native_for_spin_quant,
            )

            transforms.append(inject_fast_hadamard_transform_native_for_spin_quant)

    if embedding_quantize:
        """
        When this option is selected, it finds all embedding layers and transforms
        into quantized embedding equivalent module.

        There are cases where the checkpoint is already quantized, for example
        on use_spin_quant is enabled. In that case, it will do the appropriate
        transformations based on the given checkpoint first. In those cases,
        this wil be a no-op.
        """
        transforms.append(
            get_quant_embedding_transform(
                embedding_quantize,
                use_shared_embedding,
                quantize_with_hqq=quantize_with_hqq,
            )
        )

    # quantization_mode should be applied after embedding_quantize
    # to support shared_embedding
    if quantization_mode:
        """
        When this option is selected, it finds all linear layers and transforms
        into quantized linear equivalent module.

        There are cases where the checkpoint is already quantized, for example
        on use_spin_quant is enabled. In that case, it will do the appropriate
        transformations based on the given checkpoint first. In those cases,
        if quantization_mode is enabled, it will quantize any remaining linear
        ops that is not quantized.

        There are cases where this may be a no-op, namely, if all linears are
        quantized in the checkpoint.
        """
        transforms.append(
            get_quant_weight_transform(
                quantization_mode=quantization_mode,
                group_size=group_size,
                computation_dtype=dtype_override,
                checkpoint_dtype=checkpoint_dtype,
                checkpoint_path=checkpoint,
                tokenizer_path=tokenizer_path,
                calibration_tasks=calibration_tasks,
                calibration_limit=calibration_limit,
                calibration_seq_length=calibration_seq_length,
                quantize_with_hqq=quantize_with_hqq,
            )
        )

    if expand_rope_table:
        transforms.append(materialze_broadcast_of_rope_freq_cis)

    use_attention_mask_for_custom_sdpa = use_custom_sdpa_with_attention_mask

    if use_sdpa_with_kv_cache:
        transforms.append(replace_kv_cache_with_custom_kv_cache)
        # todo: do this optionally
        # if use attention mask instead of causal attention
        # then create partial function that sets use_attention_mask=True
        if use_attention_mask_for_custom_sdpa:
            transforms.append(
                partial(replace_sdpa_with_custom_op, use_attention_mask=True)
            )
        else:
            transforms.append(replace_sdpa_with_custom_op)

    if quantize_kv_cache:
        assert use_kv_cache, "quantize_kv_cache requires use_kv_cache=True"
        transforms.append(replace_kv_cache_with_quantized_kv_cache)
        # Right now
        transforms.append(replace_sdpa_with_quantized_sdpa)

    if use_kv_cache:
        # Vulkan-only path: no additional backend-specific source transforms.
        pass

    if local_global_attention:
        transforms.append(
            partial(
                replace_kv_cache_with_ring_kv_cache,
                layer_sizes=local_global_attention,
            )
        )

    if any([use_torchao_kernels_linear, use_torchao_kernels_tied_embedding]):
        from torchao.prototype.tensor_conversion.api import _convert_model_for_aarch64

        transforms.append(
            partial(
                _convert_model_for_aarch64,
                convert_linear=use_torchao_kernels_linear,
                convert_tied_embedding=use_torchao_kernels_tied_embedding,
            )
        )

    return transforms


def get_llama_model(llm_config: LlmConfig):
    _validate_args(llm_config)
    e_mgr = _prepare_for_llama_export(llm_config)
    model = (
        e_mgr.model.eval().to(device="cuda")
        if torch.cuda.is_available()
        else e_mgr.model.eval().to(device="cpu")
    )
    return model, e_mgr.example_inputs, e_mgr.metadata
