#!/usr/bin/env python3
"""Check whether an ExecuTorch .pte is Pure Vulkan.

This script performs two levels of checks:
1) Static check (always): parse the .pte flatbuffer and verify execution plans
   use only allowed delegate backends and contain no KernelCall in non-empty plans.
2) Runtime log check (optional): execute a command and fail if forbidden patterns
   appear in stdout/stderr (e.g. XNNPACK or CPU fallback).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ALLOWED_BACKENDS = ["VulkanBackend"]
DEFAULT_FORBIDDEN_LOG_PATTERNS = [
    r"\\bXNNPACK\\b",
    r"\\bXnnpackBackend\\b",
    r"\\bCPU fallback\\b",
    r"\\bcpu fallback\\b",
]


@dataclass
class PlanReport:
    name: str
    instruction_counts: dict[str, int]
    delegate_ids: list[str]
    total_instructions: int


def _repo_root_from_script(script_path: Path) -> Path:
    # exshader/check_pure_vulkan.py -> repo root
    return script_path.resolve().parents[1]


def _default_flatc() -> str:
    return "flatc"


def _load_program_json(pte_path: Path, schema_path: Path, flatc_bin: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="et_vkcheck_") as tmp:
        cmd = [
            flatc_bin,
            "-t",
            "--strict-json",
            "--defaults-json",
            "--raw-binary",
            "-o",
            tmp,
            str(schema_path),
            "--",
            str(pte_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "flatc failed."
                f"\nCommand: {' '.join(shlex.quote(x) for x in cmd)}"
                f"\nstdout:\n{proc.stdout}"
                f"\nstderr:\n{proc.stderr}"
            )

        json_files = list(Path(tmp).glob("*.json"))
        if not json_files:
            raise RuntimeError("flatc succeeded but produced no JSON output.")

        with json_files[0].open("r", encoding="utf-8") as f:
            return json.load(f)


def _collect_plan_reports(program_json: dict) -> list[PlanReport]:
    reports: list[PlanReport] = []
    for ep in program_json.get("execution_plan", []):
        name = ep.get("name", "<unknown>")
        delegates = ep.get("delegates", [])
        delegate_ids = [d.get("id", "<missing>") for d in delegates]

        counts: dict[str, int] = {}
        total = 0
        for chain in ep.get("chains", []):
            for inst in chain.get("instructions", []):
                itype = inst.get("instr_args_type", "<missing>")
                counts[itype] = counts.get(itype, 0) + 1
                total += 1

        reports.append(
            PlanReport(
                name=name,
                instruction_counts=counts,
                delegate_ids=delegate_ids,
                total_instructions=total,
            )
        )
    return reports


def _validate_static(
    reports: Iterable[PlanReport],
    allowed_backends: set[str],
    allow_kernel_call: bool,
) -> list[str]:
    errors: list[str] = []

    any_non_empty = False
    any_delegate_call = False

    for rep in reports:
        if rep.total_instructions == 0:
            continue
        any_non_empty = True

        delegate_set = set(rep.delegate_ids)
        unknown = sorted(b for b in delegate_set if b not in allowed_backends)
        if unknown:
            errors.append(
                f"plan '{rep.name}' uses non-allowed delegate ids: {unknown}"
            )

        kernel_calls = rep.instruction_counts.get("KernelCall", 0)
        delegate_calls = rep.instruction_counts.get("DelegateCall", 0)
        any_delegate_call = any_delegate_call or delegate_calls > 0

        if not allow_kernel_call and kernel_calls > 0:
            errors.append(
                f"plan '{rep.name}' contains KernelCall={kernel_calls} (expected 0 for pure delegate execution)"
            )

        if delegate_calls == 0:
            errors.append(
                f"plan '{rep.name}' has instructions but DelegateCall=0"
            )

    if not any_non_empty:
        errors.append("no non-empty execution plans found")
    if not any_delegate_call:
        errors.append("no DelegateCall found in any non-empty execution plan")

    return errors


def _run_and_scan_logs(run_cmd: str, forbidden_patterns: list[str]) -> tuple[int, list[str], str]:
    proc = subprocess.run(
        ["bash", "-lc", run_cmd],
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    hits: list[str] = []
    for pat in forbidden_patterns:
        if re.search(pat, combined, flags=re.IGNORECASE | re.MULTILINE):
            hits.append(pat)

    return proc.returncode, hits, combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a .pte is pure Vulkan")
    parser.add_argument("--pte", required=True, type=Path, help="Path to .pte")
    parser.add_argument(
        "--flatc",
        default=_default_flatc(),
        help="Path to flatc (default: flatc from PATH)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Path to program.fbs (default: <repo>/schema/program.fbs)",
    )
    parser.add_argument(
        "--allowed-backend",
        action="append",
        default=None,
        help="Allowed delegate backend id; repeatable (default: VulkanBackend)",
    )
    parser.add_argument(
        "--allow-kernel-call",
        action="store_true",
        help="Allow KernelCall in non-empty plans (disabled by default)",
    )
    parser.add_argument(
        "--run-cmd",
        default=None,
        help="Optional runtime command to execute and scan logs",
    )
    parser.add_argument(
        "--forbid-log-pattern",
        action="append",
        default=None,
        help=(
            "Regex pattern that must not appear in runtime logs; repeatable. "
            "Defaults include XNNPACK and CPU fallback markers."
        ),
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print machine-readable summary JSON",
    )

    args = parser.parse_args()

    script_path = Path(__file__)
    repo_root = _repo_root_from_script(script_path)
    schema_path = args.schema or (repo_root / "schema" / "program.fbs")

    if not args.pte.exists():
        print(f"ERROR: pte not found: {args.pte}", file=sys.stderr)
        return 2
    if not schema_path.exists():
        print(f"ERROR: schema not found: {schema_path}", file=sys.stderr)
        return 2

    allowed = set(args.allowed_backend or DEFAULT_ALLOWED_BACKENDS)

    program_json = _load_program_json(args.pte, schema_path, args.flatc)
    reports = _collect_plan_reports(program_json)
    static_errors = _validate_static(
        reports,
        allowed_backends=allowed,
        allow_kernel_call=args.allow_kernel_call,
    )

    runtime_rc = None
    runtime_hits: list[str] = []
    runtime_log = ""

    if args.run_cmd:
        forbid_patterns = args.forbid_log_pattern or DEFAULT_FORBIDDEN_LOG_PATTERNS
        runtime_rc, runtime_hits, runtime_log = _run_and_scan_logs(
            args.run_cmd,
            forbid_patterns,
        )
        if runtime_rc != 0:
            static_errors.append(f"run command failed with exit code {runtime_rc}")
        if runtime_hits:
            static_errors.append(
                "runtime logs contain forbidden patterns: " + ", ".join(runtime_hits)
            )

    summary = {
        "pte": str(args.pte),
        "allowed_backends": sorted(allowed),
        "plans": [
            {
                "name": r.name,
                "instruction_counts": r.instruction_counts,
                "delegate_ids": r.delegate_ids,
                "total_instructions": r.total_instructions,
            }
            for r in reports
        ],
        "runtime_check": {
            "enabled": bool(args.run_cmd),
            "exit_code": runtime_rc,
            "forbidden_hits": runtime_hits,
        },
        "ok": len(static_errors) == 0,
        "errors": static_errors,
    }

    if args.print_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("== Pure Vulkan Check ==")
        print(f"pte: {args.pte}")
        print(f"allowed delegate ids: {sorted(allowed)}")
        for r in reports:
            print(
                f"- plan={r.name} instr={r.total_instructions} "
                f"counts={r.instruction_counts} delegates={sorted(set(r.delegate_ids))}"
            )
        if args.run_cmd:
            print(f"runtime exit code: {runtime_rc}")
            if runtime_hits:
                print(f"runtime forbidden hits: {runtime_hits}")

        if static_errors:
            print("\nFAILED:")
            for err in static_errors:
                print(f"- {err}")
        else:
            print("\nPASS: pure Vulkan checks passed")

    if static_errors and runtime_log:
        # Keep output bounded while still useful for debugging.
        tail = "\n".join(runtime_log.splitlines()[-120:])
        print("\n== Runtime Log Tail ==")
        print(tail)

    return 0 if not static_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
