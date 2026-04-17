# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import torch

from executorch.examples.models.llama.runner.eager import EagerLlamaRunner
from executorch.examples.models.llama.runner.native import NativeLlamaRunner
from executorch.extension.llm.export.config.llm_config import LlmConfig, ModelType


QWEN3_5_DEFAULT_METADATA = '{"get_bos_id": 248045, "get_eos_ids":[248046,248044]}'


@dataclass(frozen=True)
class RunnerStepResult:
    token: int
    decoded: str
    top_tokens: List[int]
    top_scores: List[float]
    max_abs_diff_vs_eager: Optional[float]
    mean_abs_diff_vs_eager: Optional[float]


@dataclass(frozen=True)
class TeacherForcedStep:
    step_index: int
    input_token: Optional[int]
    input_decoded: Optional[str]
    teacher_token: int
    teacher_decoded: str
    per_runner: Dict[str, RunnerStepResult]


@dataclass(frozen=True)
class FreeRunResult:
    name: str
    generated_tokens: List[int]
    generated_text: str


def _flatten_last_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.detach().float().cpu()
    if logits.ndim == 0:
        raise ValueError("Expected logits tensor with rank >= 1.")
    if logits.ndim == 1:
        return logits
    vocab_size = logits.shape[-1]
    return logits.reshape(-1, vocab_size)[-1]


def _topk(logits: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    top_values, top_indices = torch.topk(logits, k=min(k, logits.shape[-1]))
    return top_indices.tolist(), top_values.tolist()


def _decode_token(tokenizer: Any, token: int) -> str:
    return tokenizer.decode_token(token).replace("\n", "\\n")


def _build_qwen_llm_config(
    *,
    model_name: str,
    checkpoint: str,
    params: str,
    tokenizer: str,
    max_seq_len: int,
    metadata: str,
) -> LlmConfig:
    llm_config = LlmConfig()
    llm_config.base.model_class = ModelType(model_name)
    llm_config.base.checkpoint = checkpoint
    llm_config.base.params = params
    llm_config.base.tokenizer_path = tokenizer
    llm_config.base.metadata = metadata
    llm_config.model.use_kv_cache = True
    llm_config.model.use_sdpa_with_kv_cache = False
    llm_config.model.enable_dynamic_shape = False
    llm_config.export.max_seq_length = max_seq_len
    llm_config.export.max_context_length = max_seq_len
    return llm_config


def _build_native_runner_args(
    *,
    model_name: str,
    pte: str,
    params: str,
    tokenizer: str,
    tokenizer_config: str,
    max_len: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model_name,
        pte=pte,
        params=params,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        prompt="",
        temperature=0.0,
        kv_cache=True,
        max_len=max_len,
    )


def _create_runners(
    *,
    model: str,
    checkpoint: str,
    params: str,
    tokenizer: str,
    tokenizer_config: str,
    cpu_pte: str,
    vulkan_pte: str,
    max_seq_len: int,
    metadata: str,
) -> Dict[str, Any]:
    llm_config = _build_qwen_llm_config(
        model_name=model,
        checkpoint=checkpoint,
        params=params,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        metadata=metadata,
    )
    eager_runner = EagerLlamaRunner(
        llm_config=llm_config,
        tokenizer_config_path=tokenizer_config,
    )
    if llm_config.model.use_kv_cache and not llm_config.model.enable_dynamic_shape:
        eager_runner.prefill_chunk_size = 1
    cpu_runner = NativeLlamaRunner(
        _build_native_runner_args(
            model_name=model,
            pte=cpu_pte,
            params=params,
            tokenizer=tokenizer,
            tokenizer_config=tokenizer_config,
            max_len=max_seq_len,
        )
    )
    vulkan_runner = NativeLlamaRunner(
        _build_native_runner_args(
            model_name=model,
            pte=vulkan_pte,
            params=params,
            tokenizer=tokenizer,
            tokenizer_config=tokenizer_config,
            max_len=max_seq_len,
        )
    )
    return {
        "eager": eager_runner,
        "cpu_portable": cpu_runner,
        "vulkan": vulkan_runner,
    }


def _forward_prefill(runner: Any, prompt_tokens: Sequence[int]) -> torch.Tensor:
    if runner.use_kv_cache and runner.prefill_chunk_size == 1:
        logits = None
        for index, token in enumerate(prompt_tokens):
            logits = runner.forward(
                tokens=torch.tensor([[token]], dtype=torch.long, device=runner.device),
                input_pos=torch.tensor([index], dtype=torch.long, device=runner.device),
            )
        if logits is None:
            raise RuntimeError("Prefill produced no logits.")
        return logits
    return runner.forward(
        tokens=torch.tensor([prompt_tokens], dtype=torch.long, device=runner.device),
        input_pos=(
            torch.tensor([0], dtype=torch.long, device=runner.device)
            if runner.use_kv_cache
            else None
        ),
    )


def _forward_decode(runner: Any, token: int, pos: int) -> torch.Tensor:
    return runner.forward(
        tokens=torch.tensor([[token]], dtype=torch.long, device=runner.device),
        input_pos=torch.tensor([pos], dtype=torch.long, device=runner.device),
    )


def _run_teacher_forced_compare(
    *,
    runners: Dict[str, Any],
    prompt_tokens: List[int],
    max_new_tokens: int,
    top_k: int,
) -> List[TeacherForcedStep]:
    results: List[TeacherForcedStep] = []
    last_logits: Dict[str, torch.Tensor] = {}
    reference_name = "eager"
    eager_tokenizer = runners[reference_name].tokenizer

    for name, runner in runners.items():
        last_logits[name] = _flatten_last_logits(_forward_prefill(runner, prompt_tokens))

    teacher_token = int(torch.argmax(last_logits[reference_name]).item())
    for step_index in range(max_new_tokens):
        step_results: Dict[str, RunnerStepResult] = {}
        eager_logits = last_logits[reference_name]
        for name, logits in last_logits.items():
            pred_token = int(torch.argmax(logits).item())
            top_tokens, top_scores = _topk(logits, top_k)
            if name == reference_name:
                max_abs_diff = None
                mean_abs_diff = None
            else:
                abs_diff = (logits - eager_logits).abs()
                max_abs_diff = float(abs_diff.max().item())
                mean_abs_diff = float(abs_diff.mean().item())
            step_results[name] = RunnerStepResult(
                token=pred_token,
                decoded=_decode_token(eager_tokenizer, pred_token),
                top_tokens=top_tokens,
                top_scores=top_scores,
                max_abs_diff_vs_eager=max_abs_diff,
                mean_abs_diff_vs_eager=mean_abs_diff,
            )

        input_token = None if step_index == 0 else results[-1].teacher_token
        results.append(
            TeacherForcedStep(
                step_index=step_index,
                input_token=input_token,
                input_decoded=(
                    None
                    if input_token is None
                    else _decode_token(eager_tokenizer, input_token)
                ),
                teacher_token=teacher_token,
                teacher_decoded=_decode_token(eager_tokenizer, teacher_token),
                per_runner=step_results,
            )
        )
        if step_index + 1 >= max_new_tokens:
            break

        next_logits: Dict[str, torch.Tensor] = {}
        next_pos = len(prompt_tokens) + step_index
        for name, runner in runners.items():
            next_logits[name] = _flatten_last_logits(
                _forward_decode(runner, teacher_token, next_pos)
            )
        last_logits = next_logits
        teacher_token = int(torch.argmax(last_logits[reference_name]).item())

    return results


def _run_free_generation(
    *,
    name: str,
    runner: Any,
    prompt: str,
    max_seq_len: int,
) -> FreeRunResult:
    runner.max_seq_len = max_seq_len
    tokens = runner.text_completion(prompt=prompt, temperature=0.0, echo=False)
    text = runner.tokenizer.decode(tokens)
    return FreeRunResult(name=name, generated_tokens=tokens, generated_text=text)


def _find_first_divergence(steps: Sequence[TeacherForcedStep]) -> Optional[Dict[str, Any]]:
    for step in steps:
        eager_token = step.per_runner["eager"].token
        for name, result in step.per_runner.items():
            if name == "eager":
                continue
            if result.token != eager_token:
                return {
                    "step_index": step.step_index,
                    "runner": name,
                    "eager_token": eager_token,
                    "eager_decoded": step.per_runner["eager"].decoded,
                    "other_token": result.token,
                    "other_decoded": result.decoded,
                    "teacher_token": step.teacher_token,
                    "teacher_decoded": step.teacher_decoded,
                    "input_token": step.input_token,
                    "input_decoded": step.input_decoded,
                    "max_abs_diff_vs_eager": result.max_abs_diff_vs_eager,
                    "mean_abs_diff_vs_eager": result.mean_abs_diff_vs_eager,
                }
    return None


def _steps_to_markdown(steps: Sequence[TeacherForcedStep]) -> str:
    lines = [
        "# LLM Step Compare",
        "",
        "## Teacher-Forced Steps",
        "",
    ]
    for step in steps:
        lines.append(
            f"- step `{step.step_index}` input `{step.input_token}` "
            f"(`{step.input_decoded}`) teacher `{step.teacher_token}` "
            f"(`{step.teacher_decoded}`)"
        )
        for name, result in step.per_runner.items():
            diff_suffix = ""
            if name != "eager":
                diff_suffix = (
                    f" max_abs_diff `{result.max_abs_diff_vs_eager:.6g}`"
                    f" mean_abs_diff `{result.mean_abs_diff_vs_eager:.6g}`"
                )
            lines.append(
                f"  - `{name}` pred `{result.token}` (`{result.decoded}`){diff_suffix}"
            )
        lines.append("")
    return "\n".join(lines)


def _free_runs_to_markdown(free_runs: Sequence[FreeRunResult]) -> str:
    lines = ["## Free Generation", ""]
    for result in free_runs:
        lines.append(f"- `{result.name}` tokens: `{result.generated_tokens}`")
        lines.append(f'- `{result.name}` text: `{result.generated_text}`')
        lines.append("")
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: str,
    report: Dict[str, Any],
    teacher_steps: Sequence[TeacherForcedStep],
    free_runs: Sequence[FreeRunResult],
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "llm_step_compare.json"
    md_path = output_root / "llm_step_compare.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = [
        _steps_to_markdown(teacher_steps),
        _free_runs_to_markdown(free_runs),
    ]
    md_path.write_text("\n".join(markdown), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare eager, CPU portable, and Vulkan ExecuTorch LLM execution "
            "step-by-step using teacher forcing."
        )
    )
    parser.add_argument("--model", default="qwen3_5_0_8b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-config", required=True)
    parser.add_argument("--cpu-pte", required=True)
    parser.add_argument("--vulkan-pte", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--metadata", default=QWEN3_5_DEFAULT_METADATA)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--skip-free-run",
        action="store_true",
        help="Skip free-generation pass and only run teacher-forced comparison.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    runners = _create_runners(
        model=args.model,
        checkpoint=args.checkpoint,
        params=args.params,
        tokenizer=args.tokenizer,
        tokenizer_config=args.tokenizer_config,
        cpu_pte=args.cpu_pte,
        vulkan_pte=args.vulkan_pte,
        max_seq_len=args.max_seq_len,
        metadata=args.metadata,
    )

    prompt_tokens = runners["eager"].tokenizer.encode(args.prompt, bos=True, eos=False)
    teacher_steps = _run_teacher_forced_compare(
        runners=runners,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
    )
    free_runs: List[FreeRunResult] = []
    if not args.skip_free_run:
        free_run_runners = _create_runners(
            model=args.model,
            checkpoint=args.checkpoint,
            params=args.params,
            tokenizer=args.tokenizer,
            tokenizer_config=args.tokenizer_config,
            cpu_pte=args.cpu_pte,
            vulkan_pte=args.vulkan_pte,
            max_seq_len=args.max_seq_len,
            metadata=args.metadata,
        )
        free_runs = [
            _run_free_generation(
                name=name,
                runner=runner,
                prompt=args.prompt,
                max_seq_len=len(prompt_tokens) + args.max_new_tokens,
            )
            for name, runner in free_run_runners.items()
        ]

    report = {
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "first_divergence": _find_first_divergence(teacher_steps),
        "teacher_steps": [asdict(step) for step in teacher_steps],
        "free_runs": [asdict(item) for item in free_runs],
    }

    if args.output_dir:
        _write_outputs(
            output_dir=args.output_dir,
            report=report,
            teacher_steps=teacher_steps,
            free_runs=free_runs,
        )

    print(json.dumps(report["first_divergence"], indent=2))
    for item in free_runs:
        print(f"[{item.name}] text={item.generated_text!r}")


if __name__ == "__main__":
    main()
