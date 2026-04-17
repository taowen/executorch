from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from exshader.runtime import Session


def _default_pte() -> Path:
    candidates = sorted(glob.glob("artifacts/pte/gemma3_1b_vulkan*.pte"))
    if not candidates:
        raise RuntimeError("Cannot find gemma3_1b Vulkan PTE under artifacts/pte")
    return Path(candidates[-1]).resolve()


def _default_tokenizer() -> Path:
    local = Path.home() / ".cache/modelscope/fireicewolf-google-gemma-3-1b-it/tokenizer.json"
    if local.is_file():
        return local.resolve()
    candidates = sorted(
        glob.glob(
            str(
                Path.home()
                / ".cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/*/tokenizer.json"
            )
        )
    )
    if not candidates:
        raise RuntimeError("Cannot find Gemma3 tokenizer.json in local cache")
    return Path(candidates[-1]).resolve()


def _load_tokenizer(tokenizer_arg: str | None):
    tokenizer_path = Path(tokenizer_arg).expanduser() if tokenizer_arg else _default_tokenizer()
    tokenizer_dir = tokenizer_path if tokenizer_path.is_dir() else tokenizer_path.parent
    return AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)


def _build_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _pick_token(logits: Any, temperature: float, top_p: float) -> int:
    if hasattr(logits, "argmax_last_dim_row0"):
        if temperature <= 0:
            return int(logits.argmax_last_dim_row0())
        if not hasattr(logits, "sample_top_p_row0"):
            raise RuntimeError(f"Unsupported logits type for sampling: {type(logits)}")
        return int(logits.sample_top_p_row0(float(temperature), float(top_p)))
    row = logits
    if isinstance(row, torch.Tensor):
        if row.dim() == 3:
            row = row[0, -1]
        elif row.dim() == 2:
            row = row[0]
    else:
        raise RuntimeError(f"Unsupported logits type: {type(logits)}")
    if temperature <= 0:
        return int(torch.argmax(row, dim=-1).item())
    probs = torch.softmax(row / float(temperature), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cutoff = torch.cumsum(sorted_probs, dim=-1) > float(top_p)
    cutoff[0] = False
    sorted_probs = sorted_probs.masked_fill(cutoff, 0)
    sorted_probs = sorted_probs / sorted_probs.sum()
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return int(sorted_idx[sampled].item())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pte")
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompt", default="What is 1+1?")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:  # noqa: C901
    args = build_parser().parse_args()
    pte_path = Path(args.pte).expanduser().resolve() if args.pte else _default_pte()
    tokenizer = _load_tokenizer(args.tokenizer)
    prompt = _build_prompt(tokenizer, args.prompt)

    module = Session.load(pte_path)
    forward = module.method("forward")
    max_context_len = int(module.method("get_max_context_len").run([], clone_outputs=False).values[0])
    forward_meta = forward.meta()
    max_forward_tokens = int(forward_meta.inputs[0].sizes[1])

    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(prompt_tokens) >= max_context_len:
        prompt_tokens = prompt_tokens[-(max_context_len - 1) :]
    if not prompt_tokens:
        raise RuntimeError("Prompt tokenization produced no tokens")

    current_pos = 0
    prefill_forwards = 0
    logits = None
    prefill_start = time.perf_counter()
    chunk_size = min(args.prefill_chunk_size, max_forward_tokens)
    idx = 0
    while idx < len(prompt_tokens):
        step = min(chunk_size, len(prompt_tokens) - idx)
        tokens = torch.tensor([prompt_tokens[idx : idx + step]], dtype=torch.int64)
        input_pos = torch.tensor([current_pos], dtype=torch.int64)
        logits = forward.run([tokens, input_pos], clone_outputs=False).values[0]
        current_pos += step
        idx += step
        prefill_forwards += 1
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

    generated_ids: list[int] = []
    eos_ids = tokenizer.eos_token_id
    stop_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])

    decode_start = time.perf_counter()
    decode_forwards = 0
    assert logits is not None
    for _ in range(args.max_new_tokens):
        token_id = _pick_token(logits, args.temperature, args.top_p)
        generated_ids.append(token_id)
        if args.stream:
            print(
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                end="",
                flush=True,
            )
        if token_id in stop_ids or current_pos >= max_context_len:
            break
        decode_token = torch.tensor([[token_id]], dtype=torch.int64)
        input_pos = torch.tensor([current_pos], dtype=torch.int64)
        logits = forward.run([decode_token, input_pos], clone_outputs=False).values[0]
        current_pos += 1
        decode_forwards += 1
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    if args.stream:
        print()

    print(f"prefill_ms={prefill_ms:.2f}")
    print(f"decode_ms={decode_ms:.2f}")
    print(f"prefill_forwards={prefill_forwards}")
    print(f"decode_forwards={decode_forwards}")
    if decode_ms > 0:
        print(f"decode_tok_per_sec={(len(generated_ids) / (decode_ms / 1000.0)):.2f}")


if __name__ == "__main__":
    main()
