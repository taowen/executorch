#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{lineno}: invalid json: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"{path}:{lineno}: expected JSON object")
            records.append(obj)
    return records


def _normalize_export_entry(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    compile_specs = rec.get("compile_specs", [])
    compile_keys = tuple(
        str(item.get("key", "")) for item in compile_specs if isinstance(item, dict)
    )
    return (
        rec.get("backend_id"),
        bool(rec.get("is_submodule")),
        len(rec.get("call_delegate_args", [])),
        len(rec.get("submodule_input_specs", [])),
        len(rec.get("submodule_output_specs", [])),
        compile_keys,
    )


def _normalize_runtime_init_entry(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    compile_specs = rec.get("compile_specs", [])
    compile_key_nbytes = tuple(
        (str(item.get("key", "")), int(item.get("nbytes", 0)))
        for item in compile_specs
        if isinstance(item, dict)
    )
    return (
        rec.get("event"),
        rec.get("method"),
        int(rec.get("delegate_index", -1)),
        rec.get("backend_id"),
        int(rec.get("processed_location", -1)),
        int(rec.get("processed_index", -1)),
        int(rec.get("error_code", -1)),
        compile_key_nbytes,
    )


def _normalize_runtime_delegate_call_entry(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        rec.get("method"),
        int(rec.get("chain_index", -1)),
        int(rec.get("instruction_index", -1)),
        int(rec.get("delegate_index", -1)),
        tuple(int(x) for x in rec.get("arg_indices", [])),
    )


def _summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    events = Counter(str(r.get("event", "unknown")) for r in records)
    export_entries: List[Tuple[Any, ...]] = []
    runtime_init_entries: List[Tuple[Any, ...]] = []
    runtime_delegate_call_entries: List[Tuple[Any, ...]] = []

    for rec in records:
        event = rec.get("event")
        source = rec.get("source")
        if source == "export" and event == "insert_lowered_submodule":
            export_entries.append(_normalize_export_entry(rec))
        elif source == "runtime" and event in (
            "delegate_init_begin",
            "delegate_init_end",
        ):
            runtime_init_entries.append(_normalize_runtime_init_entry(rec))
        elif source == "runtime" and event == "delegate_call_instruction":
            runtime_delegate_call_entries.append(
                _normalize_runtime_delegate_call_entry(rec)
            )

    return {
        "event_counts": events,
        "export_entries": export_entries,
        "runtime_init_entries": runtime_init_entries,
        "runtime_delegate_call_entries": runtime_delegate_call_entries,
    }


def _first_diff(a: List[Tuple[Any, ...]], b: List[Tuple[Any, ...]]) -> str | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return f"index={i}\n  baseline={a[i]}\n  candidate={b[i]}"
    if len(a) != len(b):
        return f"length mismatch baseline={len(a)} candidate={len(b)}"
    return None


def _diff_counter(a: Counter, b: Counter) -> str | None:
    if a == b:
        return None
    keys = sorted(set(a.keys()) | set(b.keys()))
    lines = []
    for k in keys:
        va = a.get(k, 0)
        vb = b.get(k, 0)
        if va != vb:
            lines.append(f"  {k}: baseline={va}, candidate={vb}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare delegate ABI traces (JSONL) between baseline and candidate."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    if not args.baseline.exists():
        raise SystemExit(f"baseline not found: {args.baseline}")
    if not args.candidate.exists():
        raise SystemExit(f"candidate not found: {args.candidate}")

    base_records = _load_jsonl(args.baseline)
    cand_records = _load_jsonl(args.candidate)
    base = _summarize(base_records)
    cand = _summarize(cand_records)

    print("== ABI Diff ==")
    print(f"baseline: {args.baseline} ({len(base_records)} records)")
    print(f"candidate: {args.candidate} ({len(cand_records)} records)")

    has_diff = False

    event_diff = _diff_counter(base["event_counts"], cand["event_counts"])
    if event_diff:
        has_diff = True
        print("\n[event_counts] DIFF")
        print(event_diff)
    else:
        print("\n[event_counts] OK")

    export_diff = _first_diff(base["export_entries"], cand["export_entries"])
    if export_diff:
        has_diff = True
        print("\n[export_entries] DIFF")
        print(export_diff)
    else:
        print("\n[export_entries] OK")

    init_diff = _first_diff(
        base["runtime_init_entries"], cand["runtime_init_entries"]
    )
    if init_diff:
        has_diff = True
        print("\n[runtime_init_entries] DIFF")
        print(init_diff)
    else:
        print("\n[runtime_init_entries] OK")

    call_diff = _first_diff(
        base["runtime_delegate_call_entries"],
        cand["runtime_delegate_call_entries"],
    )
    if call_diff:
        has_diff = True
        print("\n[runtime_delegate_call_entries] DIFF")
        print(call_diff)
    else:
        print("\n[runtime_delegate_call_entries] OK")

    if has_diff:
        print("\nRESULT: DIFFERENT")
        return 2

    print("\nRESULT: IDENTICAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
