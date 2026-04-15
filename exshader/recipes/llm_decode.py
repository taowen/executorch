#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from exshader.runtime import Session
from pytorch_tokenizers import get_tokenizer


def _next_token(logits: Any, temperature: float, top_p: float) -> int:
    if not hasattr(logits, "argmax_last_dim_row0"):
        raise RuntimeError(
            "Unsupported logits type. exshader recipe requires _exshader_runtime.ExTensor outputs."
        )
    if temperature <= 0:
        return int(logits.argmax_last_dim_row0())
    if not hasattr(logits, "sample_top_p_row0"):
        raise RuntimeError("ExTensor missing sample_top_p_row0 API")
    return int(logits.sample_top_p_row0(float(temperature), float(top_p)))


@dataclass
class ForwardAggregates:
    forward_ms: float = 0.0
    host_input_ms: float = 0.0
    module_execute_ms: float = 0.0
    output_wrap_ms: float = 0.0
    vk_copy_inputs_ms: float = 0.0
    vk_resize_ms: float = 0.0
    vk_compute_graph_execute_ms: float = 0.0
    vk_copy_outputs_ms: float = 0.0
    vk_total_backend_ms: float = 0.0
    vk_gpu_shader_total_ms: float = 0.0
    vk_gpu_shader_dispatch_count: int = 0

    def add(self, stats: Any | None) -> None:
        if stats is None:
            return
        self.forward_ms += float(stats.elapsed_ms)
        self.host_input_ms += float(stats.host_input_ms or 0.0)
        self.module_execute_ms += float(stats.module_execute_ms or 0.0)
        self.output_wrap_ms += float(stats.output_wrap_ms or 0.0)
        self.vk_copy_inputs_ms += float(stats.vk_copy_inputs_ms or 0.0)
        self.vk_resize_ms += float(stats.vk_resize_ms or 0.0)
        self.vk_compute_graph_execute_ms += float(stats.vk_compute_graph_execute_ms or 0.0)
        self.vk_copy_outputs_ms += float(stats.vk_copy_outputs_ms or 0.0)
        self.vk_total_backend_ms += float(stats.vk_total_backend_ms or 0.0)
        self.vk_gpu_shader_total_ms += float(stats.vk_gpu_shader_total_ms or 0.0)
        self.vk_gpu_shader_dispatch_count += int(stats.vk_gpu_shader_dispatch_count or 0)


@dataclass
class LoopStats:
    prompt_tokens: int
    generated_tokens: int
    prefill_forwards: int
    decode_forwards: int
    prefill_ms: float
    decode_ms: float
    prefill: ForwardAggregates
    decode: ForwardAggregates


class LLMDecodeRecipe:
    def __init__(self, pte_path: Path, tokenizer_path: Path) -> None:
        self.session = Session.load(pte_path)
        self.forward = self.session.method("forward")
        self.tokenizer = get_tokenizer(str(tokenizer_path))

        self.max_seq_len = int(self.session.method("get_max_seq_len").run([], clone_outputs=False).values[0])
        self.max_context_len = int(
            self.session.method("get_max_context_len").run([], clone_outputs=False).values[0]
        )
        self.use_kv_cache = bool(self.session.method("use_kv_cache").run([], clone_outputs=False).values[0])

        forward_meta = self.forward.meta()
        self.max_forward_tokens = int(forward_meta.inputs[0].sizes[1])
        # Reuse ExTensor buffers across all steps to reduce Python object churn.
        self.prefill_tokens_buf = self.session.alloc_int64((1, self.max_forward_tokens))
        self.decode_token_buf = self.session.alloc_int64((1, 1))
        self.input_pos_buf = self.session.alloc_int64((1,))

    def generate(  # noqa: C901
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        add_bos: bool,
        prefill_chunk_size: int,
        stream: bool,
    ) -> tuple[str, LoopStats]:
        if not self.use_kv_cache:
            raise RuntimeError("This loop expects KV-cache enabled exported model.")

        prompt_tokens: List[int] = list(self.tokenizer.encode(prompt, bos=False, eos=False))
        bos_id = getattr(self.tokenizer, "bos_id", None)
        if add_bos and bos_id is not None:
            prompt_tokens = [int(bos_id)] + prompt_tokens

        if not prompt_tokens:
            raise RuntimeError("Empty prompt after tokenization.")

        if self.max_context_len <= 1:
            raise RuntimeError(f"Invalid max_context_len: {self.max_context_len}")

        max_prompt_tokens = self.max_context_len - 1
        if len(prompt_tokens) > max_prompt_tokens:
            prompt_tokens = prompt_tokens[-max_prompt_tokens:]

        chunk = min(prefill_chunk_size, self.max_forward_tokens)
        if chunk <= 0:
            raise RuntimeError(f"Invalid prefill chunk size: {chunk}")

        generated: List[int] = []
        current_pos = 0
        prefill_forwards = 0
        decode_forwards = 0
        prefill_agg = ForwardAggregates()
        decode_agg = ForwardAggregates()

        prefill_start = time.perf_counter()
        logits: Any | None = None
        idx = 0
        while idx < len(prompt_tokens):
            step = min(chunk, len(prompt_tokens) - idx)
            self.prefill_tokens_buf.set_int64_row0_prefix(prompt_tokens[idx : idx + step])
            self.input_pos_buf.set_int64_scalar(current_pos)
            prefill_view = self.prefill_tokens_buf.row0_prefix(step)
            forward_result = self.forward.run(
                [
                    prefill_view,
                    self.input_pos_buf,
                ],
                clone_outputs=False,
            )
            logits = forward_result.values[0]
            prefill_agg.add(forward_result.stats)
            current_pos += step
            idx += step
            prefill_forwards += 1

        assert logits is not None
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

        decode_start = time.perf_counter()
        stop_tokens = set(getattr(self.tokenizer, "stop_tokens", []))
        eos_id = getattr(self.tokenizer, "eos_id", None)
        if eos_id is not None:
            stop_tokens.add(int(eos_id))

        for _ in range(max_new_tokens):
            next_id = _next_token(logits, temperature, top_p)
            generated.append(next_id)
            if stream:
                print(self.tokenizer.decode_token(next_id), end="", flush=True)

            if next_id in stop_tokens:
                break
            if current_pos >= self.max_context_len:
                break

            self.decode_token_buf.set_int64_scalar(next_id)
            self.input_pos_buf.set_int64_scalar(current_pos)
            forward_result = self.forward.run(
                [
                    self.decode_token_buf,
                    self.input_pos_buf,
                ],
                clone_outputs=False,
            )
            logits = forward_result.values[0]
            decode_agg.add(forward_result.stats)
            current_pos += 1
            decode_forwards += 1

        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        if stream:
            print()

        text = self.tokenizer.decode(generated) if generated else ""
        stats = LoopStats(
            prompt_tokens=len(prompt_tokens),
            generated_tokens=len(generated),
            prefill_forwards=prefill_forwards,
            decode_forwards=decode_forwards,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            prefill=prefill_agg,
            decode=decode_agg,
        )
        return text, stats


def _default_tokenizer_path() -> Path:
    import glob

    candidates = sorted(
        glob.glob(
            str(
                Path.home()
                / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"
            )
        )
    )
    if not candidates:
        raise RuntimeError("Cannot find Qwen3 tokenizer.json in HF cache.")
    return Path(candidates[0])


def _print_stats(stats: LoopStats) -> None:
    decode_tps = (
        (stats.generated_tokens / (stats.decode_ms / 1000.0))
        if stats.decode_ms > 0 and stats.generated_tokens > 0
        else 0.0
    )
    print(f"[py_loop] prompt_tokens={stats.prompt_tokens}")
    print(f"[py_loop] generated_tokens={stats.generated_tokens}")
    print(f"[py_loop] prefill_forwards={stats.prefill_forwards}")
    print(f"[py_loop] decode_forwards={stats.decode_forwards}")
    print(f"[py_loop] total_forwards={stats.prefill_forwards + stats.decode_forwards}")
    print(f"[py_loop] prefill_ms={stats.prefill_ms:.2f}")
    print(f"[py_loop] decode_ms={stats.decode_ms:.2f}")
    print(f"[py_loop] decode_tok_per_sec={decode_tps:.2f}")
    print(
        "[forward_stats] prefill_forward_ms="
        f"{stats.prefill.forward_ms:.2f} host_input_ms={stats.prefill.host_input_ms:.2f} "
        f"module_execute_ms={stats.prefill.module_execute_ms:.2f} "
        f"output_wrap_ms={stats.prefill.output_wrap_ms:.2f}"
    )
    print(
        "[forward_stats] decode_forward_ms="
        f"{stats.decode.forward_ms:.2f} host_input_ms={stats.decode.host_input_ms:.2f} "
        f"module_execute_ms={stats.decode.module_execute_ms:.2f} "
        f"output_wrap_ms={stats.decode.output_wrap_ms:.2f}"
    )
    print(
        "[vk_stats] prefill_copy_inputs_ms="
        f"{stats.prefill.vk_copy_inputs_ms:.2f} resize_ms={stats.prefill.vk_resize_ms:.2f} "
        f"compute_graph_execute_ms={stats.prefill.vk_compute_graph_execute_ms:.2f} "
        f"copy_outputs_ms={stats.prefill.vk_copy_outputs_ms:.2f} "
        f"total_backend_ms={stats.prefill.vk_total_backend_ms:.2f} "
        f"gpu_shader_total_ms={stats.prefill.vk_gpu_shader_total_ms:.2f} "
        f"dispatch_count={stats.prefill.vk_gpu_shader_dispatch_count}"
    )
    print(
        "[vk_stats] decode_copy_inputs_ms="
        f"{stats.decode.vk_copy_inputs_ms:.2f} resize_ms={stats.decode.vk_resize_ms:.2f} "
        f"compute_graph_execute_ms={stats.decode.vk_compute_graph_execute_ms:.2f} "
        f"copy_outputs_ms={stats.decode.vk_copy_outputs_ms:.2f} "
        f"total_backend_ms={stats.decode.vk_total_backend_ms:.2f} "
        f"gpu_shader_total_ms={stats.decode.vk_gpu_shader_total_ms:.2f} "
        f"dispatch_count={stats.decode.vk_gpu_shader_dispatch_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Python-first LLM decode recipe on ExecuTorch Runtime (pure Vulkan path)."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to .pte")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Path to tokenizer.json (default: auto-detect Qwen3 tokenizer in HF cache).",
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prefill-chunk-size", type=int, default=127)
    parser.add_argument("--add-bos", action="store_true", default=False)
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Do not stream generated text token-by-token.",
    )
    args = parser.parse_args()

    tokenizer_path = args.tokenizer if args.tokenizer is not None else _default_tokenizer_path()
    loop = LLMDecodeRecipe(args.model, tokenizer_path)
    text, stats = loop.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        add_bos=args.add_bos,
        prefill_chunk_size=args.prefill_chunk_size,
        stream=not args.no_stream,
    )
    if args.no_stream:
        print(text)
    _print_stats(stats)


if __name__ == "__main__":
    main()
