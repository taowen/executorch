## Summary
[Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) is now validated in the
pure-Vulkan `exshader` path on Linux for `0.8B`.

Current validated properties:
- export path: Python-first `exshader.export_llm`
- backend path: Vulkan only
- shape mode: static shape
- dtype path: `fp32`
- stateful execution: validated through teacher-forced compare against eager

The main bring-up bug for `qwen3_5_0_8b` was a mutable-buffer lowering mismatch:
mutable buffers must remain runtime inputs to delegated Vulkan subgraphs and must
not be lowered as delegate-owned buffers or prepacked constants.

## Export
From the repo root:

```bash
FLATC_EXECUTABLE="$PWD/.venv/bin/flatc" \
PYTHONPATH="$PWD/src:$PWD" \
./.venv/bin/python -m exshader.export_llm \
  base.model_class=qwen3_5_0_8b \
  base.params=examples/models/qwen3_5/config/0_8b_config.json \
  model.enable_dynamic_shape=false \
  model.use_kv_cache=true \
  model.use_sdpa_with_kv_cache=false \
  model.quantize_kv_cache=false \
  backend.vulkan.enabled=true \
  backend.vulkan.force_fp16=false \
  model.dtype_override=fp32 \
  export.max_seq_length=2048 \
  export.max_context_length=2048 \
  export.output_name="$PWD/artifacts/pte/qwen3_5_0_8b_vulkan_fp32_statefix_main.pte"
```

If `+base.checkpoint` is not provided, the exporter will download and convert the
HF checkpoint automatically.

## Run
Use the Qwen chat template and remember that `--max_len` is total sequence
length, not “max new tokens”.

```bash
FLATC_EXECUTABLE="$PWD/.venv/bin/flatc" \
PYTHONPATH="$PWD/src:$PWD" \
./.venv/bin/python -m executorch.examples.models.llama.runner.native \
  --model qwen3_5_0_8b \
  -f artifacts/pte/qwen3_5_0_8b_vulkan_fp32_statefix_main.pte \
  -p examples/models/qwen3_5/config/0_8b_config.json \
  -t /home/taowen/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/tokenizer.json \
  --tokenizer_config /home/taowen/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/tokenizer_config.json \
  --prompt $'<|im_start|>user\nWhat is 1+1?\n<|im_end|>\n<|im_start|>assistant\n' \
  --temperature 0 \
  -kv \
  --max_len 128
```

## Debugging
### Teacher-Forced Compare
Use this first before blaming Vulkan numerics:

```bash
PYTHONPATH="$PWD/src:$PWD" \
./.venv/bin/python -m exshader.diag.llm_step_compare \
  --model qwen3_5_0_8b \
  --checkpoint /home/taowen/.cache/meta_checkpoints/Qwen_Qwen3.5-0.8B.pth \
  --params examples/models/qwen3_5/config/0_8b_config.json \
  --tokenizer /home/taowen/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/tokenizer.json \
  --tokenizer-config /home/taowen/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/tokenizer_config.json \
  --vulkan-pte artifacts/pte/qwen3_5_0_8b_vulkan_fp32_statefix_main.pte \
  --prompt $'<|im_start|>user\nWhat is 1+1?\n<|im_end|>\n<|im_start|>assistant\n' \
  --max-seq-len 128 \
  --max-new-tokens 2 \
  --top-k 5 \
  --output-dir outputs/qwen35_statefix_main_compare \
  --skip-free-run
```

Expected current outcome:
- `first_divergence: null`
- step 0: eager token == Vulkan token == `<think>`

### ABI Diff
When export/lowering contracts are suspicious, compare the ABI dumps instead of
reading long logs by hand:

```bash
PYTHONPATH="$PWD/src:$PWD" \
./.venv/bin/python -m exshader.diag.abi_diff \
  --baseline /path/to/baseline.jsonl \
  --candidate /path/to/candidate.jsonl
```

## Notes
- The validated path here is **pure Vulkan**, not CPU/XNNPACK.
- Static-shape export is still the validated mode for `qwen3_5_0_8b`.
- If output text is wrong, check prompt template, tokenizer pairing, and
  sequence budget before investigating the backend.
