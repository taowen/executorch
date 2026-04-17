#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from exshader.diag.llm_step_compare import (
    QWEN3_5_DEFAULT_METADATA,
    _assert_local_pybindings,
    _build_native_runner_args,
    _build_llm_config,
    _decode_token,
    _flatten_last_logits,
    _forward_decode,
    _forward_prefill,
    _repo_root,
    _topk,
)
from executorch.examples.models.llama.runner.eager import EagerLlamaRunner
from executorch.examples.models.llama.runner.native import NativeLlamaRunner


@dataclass(frozen=True)
class StepSummary:
    step_index: int
    eager_token: int
    eager_decoded: str
    candidate_token: int
    candidate_decoded: str
    max_abs_diff: float
    mean_abs_diff: float
    candidate_top_tokens: List[int]
    candidate_top_scores: List[float]


@dataclass(frozen=True)
class PteSweepResult:
    pte_path: str
    first_divergence: Optional[StepSummary]
    teacher_steps: List[StepSummary]


def _create_eager_runner(
    *,
    model: str,
    checkpoint: str,
    params: str,
    tokenizer: str,
    tokenizer_config: str,
    max_seq_len: int,
    metadata: str,
) -> Any:
    llm_config = _build_llm_config(
        model_name=model,
        checkpoint=checkpoint,
        params=params,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        metadata=metadata,
    )
    runner = EagerLlamaRunner(
        llm_config=llm_config,
        tokenizer_config_path=tokenizer_config,
    )
    if llm_config.model.use_kv_cache and not llm_config.model.enable_dynamic_shape:
        runner.prefill_chunk_size = 1
    return runner


def _create_native_runner(
    *,
    model: str,
    pte: str,
    params: str,
    tokenizer: str,
    tokenizer_config: str,
    max_seq_len: int,
) -> Any:
    return NativeLlamaRunner(
        _build_native_runner_args(
            model_name=model,
            pte=pte,
            params=params,
            tokenizer=tokenizer,
            tokenizer_config=tokenizer_config,
            max_len=max_seq_len,
        )
    )


def _teacher_tokens(
    eager_runner: Any,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
) -> List[int]:
    logits = _flatten_last_logits(_forward_prefill(eager_runner, prompt_tokens))
    tokens: List[int] = []
    for step_index in range(max_new_tokens):
        teacher_token = int(torch.argmax(logits).item())
        tokens.append(teacher_token)
        if step_index + 1 >= max_new_tokens:
            break
        logits = _flatten_last_logits(
            _forward_decode(eager_runner, teacher_token, len(prompt_tokens) + step_index)
        )
    return tokens


def _compare_single_pte(
    *,
    eager_runner: Any,
    model: str,
    pte_path: str,
    params: str,
    tokenizer: str,
    tokenizer_config: str,
    prompt_tokens: Sequence[int],
    teacher_tokens: Sequence[int],
    top_k: int,
    max_seq_len: int,
) -> PteSweepResult:
    candidate_runner = _create_native_runner(
        model=model,
        pte=pte_path,
        params=params,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        max_seq_len=max_seq_len,
    )
    tokenizer_obj = eager_runner.tokenizer
    eager_logits = _flatten_last_logits(_forward_prefill(eager_runner, prompt_tokens))
    cand_logits = _flatten_last_logits(_forward_prefill(candidate_runner, prompt_tokens))

    first_divergence: Optional[StepSummary] = None
    summaries: List[StepSummary] = []
    for step_index, teacher_token in enumerate(teacher_tokens):
        eager_token = int(torch.argmax(eager_logits).item())
        candidate_token = int(torch.argmax(cand_logits).item())
        diff = (cand_logits - eager_logits).abs()
        top_tokens, top_scores = _topk(cand_logits, top_k)
        summary = StepSummary(
            step_index=step_index,
            eager_token=eager_token,
            eager_decoded=_decode_token(tokenizer_obj, eager_token),
            candidate_token=candidate_token,
            candidate_decoded=_decode_token(tokenizer_obj, candidate_token),
            max_abs_diff=float(diff.max().item()),
            mean_abs_diff=float(diff.mean().item()),
            candidate_top_tokens=top_tokens,
            candidate_top_scores=top_scores,
        )
        summaries.append(summary)
        if first_divergence is None and candidate_token != eager_token:
            first_divergence = summary

        if step_index + 1 >= len(teacher_tokens):
            break
        pos = len(prompt_tokens) + step_index
        eager_logits = _flatten_last_logits(
            _forward_decode(eager_runner, teacher_token, pos)
        )
        cand_logits = _flatten_last_logits(
            _forward_decode(candidate_runner, teacher_token, pos)
        )

    return PteSweepResult(
        pte_path=pte_path,
        first_divergence=first_divergence,
        teacher_steps=summaries,
    )


def _expand_pte_args(raw_pte_args: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for item in raw_pte_args:
        path = Path(item).expanduser()
        if any(ch in item for ch in "*?[]"):
            matches = sorted(str(p.resolve()) for p in path.parent.glob(path.name))
            expanded.extend(matches)
        else:
            expanded.append(str(path.resolve()))
    deduped: List[str] = []
    seen = set()
    for item in expanded:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep multiple PTEs against the same eager teacher-forced reference "
            "and report which PTE diverges first."
        )
    )
    parser.add_argument("--model", default="qwen3_5_0_8b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--pte", action="append", required=True)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--metadata", default=QWEN3_5_DEFAULT_METADATA)
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    repo_root = _repo_root()
    _assert_local_pybindings(repo_root)

    pte_paths = _expand_pte_args(args.pte)
    if not pte_paths:
        raise RuntimeError("No PTE paths resolved.")

    torch.set_grad_enabled(False)
    eager_runner = _create_eager_runner(
        model=args.model,
        checkpoint=args.checkpoint,
        params=args.params,
        tokenizer=args.tokenizer,
        tokenizer_config=args.tokenizer_config,
        max_seq_len=args.max_seq_len,
        metadata=args.metadata,
    )
    prompt_tokens = eager_runner.tokenizer.encode(args.prompt, bos=True, eos=False)
    teacher_tokens = _teacher_tokens(
        eager_runner=eager_runner,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    results: List[PteSweepResult] = []
    for pte_path in pte_paths:
        results.append(
            _compare_single_pte(
                eager_runner=eager_runner,
                model=args.model,
                pte_path=pte_path,
                params=args.params,
                tokenizer=args.tokenizer,
                tokenizer_config=args.tokenizer_config,
                prompt_tokens=prompt_tokens,
                teacher_tokens=teacher_tokens,
                top_k=args.top_k,
                max_seq_len=args.max_seq_len,
            )
        )

    payload = {
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "teacher_tokens": teacher_tokens,
        "results": [asdict(item) for item in results],
    }

    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = []
    for item in results:
        summary.append(
            {
                "pte": item.pte_path,
                "first_divergence_step": (
                    None if item.first_divergence is None else item.first_divergence.step_index
                ),
                "first_divergence_eager_token": (
                    None if item.first_divergence is None else item.first_divergence.eager_token
                ),
                "first_divergence_candidate_token": (
                    None
                    if item.first_divergence is None
                    else item.first_divergence.candidate_token
                ),
                "first_divergence_max_abs_diff": (
                    None
                    if item.first_divergence is None
                    else item.first_divergence.max_abs_diff
                ),
            }
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
