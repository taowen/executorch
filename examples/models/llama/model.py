# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
from typing import Optional

import torch
from executorch.examples.models.checkpoint import (
    get_checkpoint_dtype,
    get_default_model_resource_dir,
)

from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs

from executorch.extension.llm.export.config.llm_config import LlmConfig
from torchao.utils import TorchAOBaseTensor

from ..model_base import EagerModelBase


class Llama2Model(EagerModelBase):
    def __init__(
        self,
        llm_config: Optional[LlmConfig] = None,
        *,
        model_class="llama3",
        checkpoint: Optional[str] = None,
        params: Optional[str] = None,
        lora_config=None,
        use_kv_cache: bool = False,
        use_sdpa_with_kv_cache: bool = False,
        generate_full_logits: bool = False,
        enable_dynamic_shape: bool = True,
        input_prune_map: Optional[str] = None,
        output_prune_map: Optional[str] = None,
        max_seq_length: int = 128,
        max_context_length: int = 128,
        verbose: bool = False,
        use_spin_quant=None,
        use_qat: bool = False,
        use_lora: int = 0,
        use_attention_sink: Optional[str] = None,
        preq_mode=None,
        preq_group_size: int = 32,
        dtype_override="fp32",
        preq_embedding_quantize: str = "8,0",
    ):
        def _enum_value(value):
            return value.value if hasattr(value, "value") else value

        if llm_config is not None:
            checkpoint_path = llm_config.base.checkpoint
            params_path = llm_config.base.params
            lora_config = llm_config.base.lora_config
            self.model_class = llm_config.base.model_class.value
            self.use_kv_cache = llm_config.model.use_kv_cache
            self.use_sdpa_with_kv_cache_op = llm_config.model.use_sdpa_with_kv_cache
            self.generate_full_logits = llm_config.debug.generate_full_logits
            self.enable_dynamic_shape = llm_config.model.enable_dynamic_shape
            self.input_prune_map_path = llm_config.model.input_prune_map
            self.output_prune_map_path = llm_config.model.output_prune_map
            self.max_seq_len = llm_config.export.max_seq_length
            self.max_context_len = llm_config.export.max_context_length
            self.verbose = llm_config.debug.verbose
            self.use_spin_quant = _enum_value(llm_config.quantization.use_spin_quant)
            self.use_qat = llm_config.quantization.use_qat
            self.use_lora_rank = llm_config.base.use_lora
            self.use_attention_sink = llm_config.model.use_attention_sink
            self.preq_mode = _enum_value(llm_config.base.preq_mode)
            self.preq_group_size = llm_config.base.preq_group_size
            self.dtype_override = _enum_value(llm_config.model.dtype_override)
            self.preq_embedding_quantize = llm_config.base.preq_embedding_quantize
        else:
            checkpoint_path = checkpoint
            params_path = params
            self.model_class = _enum_value(model_class)
            self.use_kv_cache = use_kv_cache
            self.use_sdpa_with_kv_cache_op = use_sdpa_with_kv_cache
            self.generate_full_logits = generate_full_logits
            self.enable_dynamic_shape = enable_dynamic_shape
            self.input_prune_map_path = input_prune_map
            self.output_prune_map_path = output_prune_map
            self.max_seq_len = max_seq_length
            self.max_context_len = max_context_length
            self.verbose = verbose
            self.use_spin_quant = _enum_value(use_spin_quant)
            self.use_qat = use_qat
            self.use_lora_rank = use_lora
            self.use_attention_sink = use_attention_sink
            self.preq_mode = _enum_value(preq_mode)
            self.preq_group_size = preq_group_size
            self.dtype_override = _enum_value(dtype_override)
            self.preq_embedding_quantize = preq_embedding_quantize

        assert (
            self.max_context_len >= self.max_seq_len
        ), f"max_context_len({self.max_context_len}) must be >= max_seq_len({self.max_seq_len})"

        # The example is using a dummy small model with random weights for demo purpose only.
        # Follow the instruction in https://github.com/facebookresearch/llama to download the model.
        device = "cpu"
        # flake8: noqa: TOR102
        checkpoint = {}
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)
        if "model" in checkpoint:
            # NB: some checkpoint contains a "model" field, which is the actual weights dict
            checkpoint = checkpoint["model"]

        # Get optional params.
        params = {}
        if params_path:
            with open(params_path, "r") as f:
                params = json.loads(f.read())

        # Get adapter checkpoint.
        adapter_checkpoint = {}
        if lora_config:
            # Resolve LoRA params from adapter_config JSON if not already set.
            if lora_config.adapter_config and lora_config.lora_rank == 0:
                with open(lora_config.adapter_config, "r") as f:
                    cfg = json.load(f)
                lora_config.lora_rank = cfg["r"]
                lora_config.lora_alpha = cfg["lora_alpha"]
                lora_config.target_modules = cfg["target_modules"]

            adapter_checkpoint_path = lora_config.adapter_checkpoint
            if adapter_checkpoint_path.endswith(".pt"):
                adapter_checkpoint = torch.load(
                    adapter_checkpoint_path, map_location=device, mmap=True
                )
                from torchtune.models import convert_weights

                adapter_checkpoint = convert_weights.tune_to_meta(adapter_checkpoint)
            elif adapter_checkpoint_path.endswith(".safetensors"):
                from executorch.examples.models.llama.convert_weights import (
                    load_and_convert_unsloth_to_meta,
                )

                adapter_checkpoint = load_and_convert_unsloth_to_meta(
                    adapter_checkpoint_path
                )
            else:
                raise ValueError(
                    f"Unsupported adapter checkpoint format: {adapter_checkpoint_path}"
                )
            checkpoint.update(adapter_checkpoint)

        output_prune_map = None
        if self.output_prune_map_path is not None:
            with open(self.output_prune_map_path, "r") as f:
                output_prune_map = json.load(f)
            # Change keys from string to int (json only supports string keys).
            output_prune_map = {int(k): v for (k, v) in output_prune_map.items()}
        input_prune_map = None
        if self.input_prune_map_path is not None:
            with open(self.input_prune_map_path, "r") as f:
                input_prune_map = json.load(f)
            # Change keys from string to int (json only supports string keys).
            input_prune_map = {int(k): v for (k, v) in input_prune_map.items()}

        model_args: ModelArgs = ModelArgs(
            max_seq_len=self.max_seq_len,
            max_context_len=self.max_context_len,
            max_batch_size=1,
            use_kv_cache=self.use_kv_cache,
            use_sdpa_with_kv_cache_op=self.use_sdpa_with_kv_cache_op,
            generate_full_logits=self.generate_full_logits,
            input_prune_map=input_prune_map,
            output_prune_map=output_prune_map,
            enable_dynamic_shape=self.enable_dynamic_shape,
            r=lora_config.lora_rank if lora_config else None,
            lora_alpha=lora_config.lora_alpha if lora_config else None,
            target_modules=lora_config.target_modules if lora_config else None,
            **params,
        )

        if model_args.use_scaled_rope:
            # Older models don't have use_scaled_rope configuration
            model_name = self.model_class
            assert model_name not in ["llama2", "stories110m"]

            # Llama3_2 and newer models in ExecuTorch repo should set larger scale factor
            if model_name not in ["llama3", "llama3_1"]:
                model_args.rope_scale_factor = 32

        if self.verbose:
            print("============= weights ================")
            print("{key} : {weights.numel()} : {weights.size()}")
            for key, weights in checkpoint.items():
                print(f"{key} : {weights.numel()} : {weights.size()}")
            print("============= /weights ================")

        # Within the device="meta" context, tensors that are created do not carry data.
        # They possess all other metadata a tensor carries such as size, stride, requires_grad.
        with torch.device("meta"):
            # Model itself is loaded in default dtype, fp32.
            self.model_ = construct_transformer(model_args)
            # Get checkpoint dtype.
            if checkpoint:
                self.model_.checkpoint_dtype = get_checkpoint_dtype(checkpoint)
            else:
                self.model_.checkpoint_dtype = torch.float32

        if "int8" in str(checkpoint_path):
            print("Using int8 weight-only quantization!")
            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.examples.models.source_transformation.quantize`
            from ..source_transformation.quantize import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(self.model_)
            self.model_ = simple_quantizer.convert_for_runtime()
        elif "8da4w" in str(checkpoint_path):
            print("Using int4 weight and int8 dynamic activation quantization!")
            from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

            self.model_ = Int8DynActInt4WeightQuantizer()._convert_for_runtime(
                self.model_
            )
        elif self.use_spin_quant:
            print("Using SPIN quantization.")
            self._transform_for_pre_quantization(checkpoint, model_args)

            from .source_transformation.pre_quantization import (
                sanitize_checkpoint_from_pre_quantization,
            )

            sanitize_checkpoint_from_pre_quantization(checkpoint)
        elif self.use_qat:
            print("Using QAT quantization.")
            self._transform_for_pre_quantization(checkpoint, model_args)
            if self.use_lora_rank:
                lora_rank = self.use_lora_rank
                assert model_args.lora_args["rank"] == lora_rank
                from .source_transformation.lora import (
                    transform_linear_for_lora_after_quantization,
                )

                self.model_ = transform_linear_for_lora_after_quantization(
                    self.model_,
                    checkpoint,
                    lora_rank,
                )

            from .source_transformation.pre_quantization import (
                sanitize_checkpoint_from_pre_quantization,
            )

            sanitize_checkpoint_from_pre_quantization(checkpoint)

        if self.use_attention_sink:
            from .source_transformation.attention_sink import enable_attention_sink

            attention_sink_params = self.use_attention_sink.split(",")
            assert len(attention_sink_params) == 2, (
                f"use_attention_sink expects exactly 2 comma-separated values "
                f"(sink_size,window_size), got {len(attention_sink_params)}"
            )
            sink_size = int(attention_sink_params[0])
            window_size = int(attention_sink_params[1])

            # max_context_length must be >= sink_size + window_size to have enough RoPE frequencies
            # A larger max_context_length is allowed (and recommended) to support generation beyond
            # the sliding window size.
            assert self.max_context_len >= sink_size + window_size, (
                f"max_context_length ({self.max_context_len}) must be >= "
                f"sink_size + window_size ({sink_size + window_size})"
            )

            self.model_ = enable_attention_sink(
                module=self.model_,
                params=model_args,
                sink_size=sink_size,
                window_size=window_size,
            )

        missing, unexpected = None, None
        # assign=True: load params/buffers by assignment instead of performing an in-place copy.
        # Because we are using device="meta", tensors do not have memory associated with them
        # and an in-place copy is a no-op. Use assign=True in load_state_dict for this scenario.

        # Also, the checkpoint is loaded and dtype promoted to the transformer's dtype, which is
        # by default initialized to fp32. This is fine because every other supported type
        # losslessly converts to fp32, so we don't lose precision here.
        if checkpoint:
            missing, unexpected = self.model_.load_state_dict(
                checkpoint,
                strict=False,
                assign=True,
            )  # self.model_ = Transformer(gptconf)
            for param in self.model_.parameters():
                if isinstance(param, TorchAOBaseTensor):
                    param.requires_grad = False
            if missing:
                missing_weights = [fqn for fqn in missing if fqn.endswith(".weight")]
                if missing_weights:
                    raise ValueError(
                        f"The provided checkpoint is missing the following weights that are expected by the model: {missing_weights}. Please fix the fqn's in your checkpoint to match."
                    )
            if unexpected:
                if self.verbose:
                    print(f"Unexpected keys: {unexpected}")
        else:
            print("Checkpoint not provided, using default initialization.")
            # Because we loaded onto meta device, it is annoying to now load onto cpu
            # with the standard random initialization.
            self.model_.to_empty(device="cpu")

            def weight_reset(m):
                reset_parameters = getattr(m, "reset_parameters", None)
                if callable(reset_parameters):
                    m.reset_parameters()

            self.model_.apply(weight_reset)

        # Prune the input layer if input_prune_map is provided
        if input_prune_map is not None:
            from .source_transformation.prune_vocab import prune_input_vocab

            self.model_ = prune_input_vocab(self.model_, input_prune_map)

        # Prune the output layer if output_prune_map is provided
        if output_prune_map is not None:
            from .source_transformation.prune_vocab import prune_output_vocab

            self.model_ = prune_output_vocab(self.model_, output_prune_map)

    def get_eager_model(self) -> torch.nn.Module:
        return self.model_

    def get_example_inputs(self):
        if self.use_kv_cache:
            return self.get_example_inputs_kvcache_sdpa()
        else:
            return (
                torch.tensor(
                    [[1, 2, 3]], dtype=torch.long
                ),  # tokens, with kv cache our input token length is always just 1 token.
            )

    # assumption is the custom op doesnt support dynamic shape right now. It might but its untested so lets first get static shape working
    def get_example_inputs_kvcache_sdpa(self):
        if self.enable_dynamic_shape:
            return (
                torch.tensor([[2, 3, 4]], dtype=torch.long),
                {"input_pos": torch.tensor([0], dtype=torch.long)},
            )
        else:
            return (
                torch.tensor(
                    [[1]], dtype=torch.long
                ),  # tokens, with kv cache our input token length is always just 1 token.
                {
                    "input_pos": torch.tensor(
                        [0], dtype=torch.long
                    )  # start_pos, what token of output are we on.
                },
            )

    def _transform_for_pre_quantization(self, checkpoint, model_args):
        assert self.preq_mode, "preq_mode must be specified"
        assert self.preq_mode in [
            "8da4w",
            "8da4w_output_8da8w",
        ], f"Quantization mode {self.preq_mode} is not compatible with SpinQuant."
        assert self.preq_group_size, "preq_group_size must be specified"
        assert self.dtype_override, "dtype_override must be specified"

        from .source_transformation.pre_quantization import (
            transform_linear_for_pre_quantization,
        )

        assert (
            self.preq_group_size == model_args.quantization_args["group_size"]
        )

        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

        # Transform the output layer first if needed.
        if self.preq_mode == "8da4w_output_8da8w":
            from .source_transformation.pre_quantization import (
                transform_output_linear_for_pre_quantization,
            )

            self.model_ = transform_output_linear_for_pre_quantization(
                module=self.model_,
                checkpoint=checkpoint,
                dtype=mapping[self.dtype_override],
            )

        self.model_ = transform_linear_for_pre_quantization(
            self.model_,
            checkpoint,
            self.preq_group_size,
            mapping[self.dtype_override],
        )

        embedding_bit_width, embedding_group_size = None, None
        if self.preq_embedding_quantize:
            (
                embedding_bit_width,
                embedding_group_size,
            ) = self.preq_embedding_quantize.split(",")
            from .source_transformation.pre_quantization import (
                transform_embedding_for_pre_quantization,
            )

            if (
                embedding_group_size == "none"
                or embedding_group_size == "None"
                or embedding_group_size == "0"
            ):
                embedding_group_size = None
            else:
                embedding_group_size = int(embedding_group_size)

            self.model_ = transform_embedding_for_pre_quantization(
                self.model_,
                checkpoint,
                mapping[self.dtype_override],
                int(embedding_bit_width),
                embedding_group_size,
            )
