#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, List

import torch
from torch.utils import _pytree as pytree

from executorch.devtools import Inspector
from executorch.devtools.etrecord import parse_etrecord

import _portable_lib


ET_DTYPE_LONG = 4


def _ensure_flatc() -> None:
    configured = Path(str(os.environ.get("FLATC_EXECUTABLE", ""))).expanduser()
    if str(configured) and configured.is_file():
        return

    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / ".venv" / "bin" / "flatc"
    if candidate.is_file():
        os.environ["FLATC_EXECUTABLE"] = str(candidate)
        return

    raise RuntimeError(
        "flatc is required by Inspector ETDump parser. "
        "Set FLATC_EXECUTABLE or install flatc at .venv/bin/flatc."
    )


def _build_llm_forward_inputs(
    method_meta: Any,
    token_id: int,
    input_pos: int,
) -> List[torch.Tensor]:
    inputs: List[torch.Tensor] = []
    for i in range(method_meta.num_inputs()):
        tensor_meta = method_meta.input_tensor_meta(i)
        sizes = list(tensor_meta.sizes())
        dtype = int(tensor_meta.dtype())
        if dtype != ET_DTYPE_LONG:
            raise RuntimeError(
                f"Unsupported input dtype for method input[{i}]: {dtype}. "
                f"Expected Long({ET_DTYPE_LONG}) for LLM forward."
            )

        numel = 1
        for dim in sizes:
            numel *= int(dim)

        if i == 0:
            fill_value = int(token_id)
        elif i == 1:
            fill_value = int(input_pos)
        else:
            fill_value = 0

        tensor = torch.full((numel,), fill_value=fill_value, dtype=torch.long).reshape(
            sizes
        )
        inputs.append(tensor)

    return inputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one ET method with ETDump enabled and parse with Inspector."
    )
    parser.add_argument("--pte", required=True, help="Path to .pte model file")
    parser.add_argument("--method", default="forward", help="Method name to run")
    parser.add_argument(
        "--token-id",
        type=int,
        default=5328,
        help="Token id used for forward input[0] fill",
    )
    parser.add_argument(
        "--input-pos",
        type=int,
        default=0,
        help="Input position used for forward input[1] fill",
    )
    parser.add_argument("--etdump-out", required=True, help="Output ETDump path (.etdp)")
    parser.add_argument(
        "--etdump-debug-out",
        default=None,
        help="Optional ETDump debug buffer output path",
    )
    parser.add_argument(
        "--etrecord",
        default=None,
        help="Optional ETRecord path for Inspector symbol mapping",
    )
    parser.add_argument(
        "--etrecord-patched-out",
        default=None,
        help=(
            "Optional path to save ETRecord patched with representative inputs "
            "and first-run reference outputs."
        ),
    )
    parser.add_argument(
        "--debug-buffer-size",
        type=int,
        default=1024 * 1024 * 1024,
        help=(
            "ETDump debug buffer size in bytes for runtime intermediate tensors. "
            "Set > 0 to enable Inspector tensor debug data."
        ),
    )
    parser.add_argument(
        "--numeric-gap-metrics",
        default="MSE",
        help=(
            "Comma-separated metrics for Inspector.calculate_numeric_gap. "
            "Examples: MSE,L1,SNR . Empty string disables numeric gap."
        ),
    )
    parser.add_argument(
        "--numeric-gap-json-out",
        default=None,
        help="Optional JSON output path for numeric-gap results.",
    )
    parser.add_argument(
        "--inspector-csv-out",
        default=None,
        help="Optional Inspector dataframe CSV output path",
    )
    parser.add_argument(
        "--inspector-json-out",
        default=None,
        help="Optional Inspector dataframe JSON output path",
    )
    parser.add_argument(
        "--summary-json-out",
        default=None,
        help="Optional summary JSON output path",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print Inspector tabular output to stdout",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Number of rows to print from Inspector dataframe preview",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_flatc()
    pte = Path(args.pte).expanduser().resolve()
    if not pte.is_file():
        raise FileNotFoundError(f"PTE not found: {pte}")

    etdump_out = Path(args.etdump_out).expanduser().resolve()
    etdump_out.parent.mkdir(parents=True, exist_ok=True)

    etdump_debug_out = (
        Path(args.etdump_debug_out).expanduser().resolve()
        if args.etdump_debug_out
        else None
    )
    if etdump_debug_out is not None:
        etdump_debug_out.parent.mkdir(parents=True, exist_ok=True)

    module = _portable_lib._load_for_executorch(
        str(pte),
        enable_etdump=True,
        debug_buffer_size=max(0, int(args.debug_buffer_size)),
    )
    if not module.has_etdump():
        raise RuntimeError(
            "ETDump is unavailable on this module instance. "
            "Rebuild with EXECUTORCH_ENABLE_EVENT_TRACER=ON."
        )

    method_meta = module.method_meta(args.method)
    inputs = _build_llm_forward_inputs(
        method_meta=method_meta,
        token_id=args.token_id,
        input_pos=args.input_pos,
    )

    outputs = module.run_method(args.method, tuple(inputs), True)
    first_forward_top1 = None
    if outputs and isinstance(outputs[0], torch.Tensor):
        first_forward_top1 = int(torch.argmax(outputs[0], dim=-1).reshape(-1)[0].item())
        print(f"[inspector] first_forward_top1={first_forward_top1}")

    module.write_etdump_result_to_file(
        str(etdump_out),
        str(etdump_debug_out) if etdump_debug_out is not None else None,
    )

    if not etdump_out.is_file() or etdump_out.stat().st_size == 0:
        raise RuntimeError(
            f"ETDump was not generated at {etdump_out}. "
            "Likely missing runtime event tracer build flag."
        )

    inspector_kwargs = {"etdump_path": str(etdump_out)}
    effective_etrecord_path = ""
    if args.etrecord:
        etrecord = Path(args.etrecord).expanduser().resolve()
        if etrecord.is_file():
            parsed_etrecord = parse_etrecord(str(etrecord))

            # Follow Inspector tests: store flattened representative inputs.
            flattened_inputs = pytree.tree_flatten(tuple(inputs))[0]
            parsed_etrecord.update_representative_inputs(flattened_inputs)

            # Keep one runtime output set in ETRecord for output-level comparisons.
            runtime_outputs = list(outputs) if outputs is not None else []
            parsed_etrecord.update_reference_outputs({args.method: [runtime_outputs]})

            # Avoid mandatory re-serialization because some ETRecords include ops
            # unavailable in current Python registry (e.g. custom_sdpa fake ops).
            # Inspector accepts ETRecord objects directly.
            inspector_kwargs["etrecord"] = parsed_etrecord
            effective_etrecord_path = str(etrecord)

            if args.etrecord_patched_out:
                patched_out = Path(args.etrecord_patched_out).expanduser().resolve()
                patched_out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    parsed_etrecord.save(str(patched_out))
                    effective_etrecord_path = str(patched_out)
                    print(f"[inspector] patched_etrecord={patched_out}")
                except Exception as exc:
                    print(
                        "[inspector] warning: failed to save patched ETRecord; "
                        f"fallback to in-memory ETRecord object: {type(exc).__name__}: {exc}"
                    )
        else:
            print(f"[inspector] warning: ETRecord not found, skip mapping: {etrecord}")

    if etdump_debug_out is not None and etdump_debug_out.is_file():
        inspector_kwargs["debug_buffer_path"] = str(etdump_debug_out)
    else:
        print(
            "[inspector] warning: debug buffer missing; tensor debug and numeric gap "
            "will be unavailable."
        )

    inspector = Inspector(**inspector_kwargs)

    if args.print_table:
        inspector.print_data_tabular()

    df = inspector.to_dataframe()
    print(f"[inspector] rows={len(df)} cols={list(df.columns)}")
    if len(df) > 0:
        preferred_cols = [
            "event_name",
            "avg",
            "p50",
            "delegate_backend_name",
            "is_delegated_op",
        ]
        show_cols = [c for c in preferred_cols if c in df.columns]
        preview = df[show_cols].copy() if show_cols else df.copy()
        if "event_name" in preview.columns:
            preview["event_name"] = preview["event_name"].astype(str).map(
                lambda s: s if len(s) <= 120 else (s[:117] + "...")
            )
        print(preview.head(args.max_rows).to_string(index=False))

    if args.inspector_csv_out:
        csv_out = Path(args.inspector_csv_out).expanduser().resolve()
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_out, index=False)
        print(f"[inspector] csv={csv_out}")

    if args.inspector_json_out:
        json_out = Path(args.inspector_json_out).expanduser().resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        # Always stringify object columns before JSON export.
        # Raw pandas->ujson serialization can recurse into nested tensors/containers
        # and may crash the process on large debug_data payloads.
        safe_df = df.copy()
        for col in safe_df.columns:
            if str(safe_df[col].dtype) == "object":
                safe_df[col] = safe_df[col].map(
                    lambda v: (
                        str(v)
                        if v is None
                        else (str(v) if len(str(v)) <= 2000 else str(v)[:2000] + "...")
                    )
                )
        safe_df.to_json(json_out, orient="records", indent=2)
        print(f"[inspector] json={json_out}")

    numeric_gap_data = {}
    metrics = [m.strip() for m in str(args.numeric_gap_metrics).split(",") if m.strip()]
    can_run_numeric_gap = (
        bool(metrics)
        and "etrecord" in inspector_kwargs
        and "debug_buffer_path" in inspector_kwargs
    )
    if can_run_numeric_gap:
        try:
            runtime_outs, runtime_op_names = (
                inspector._get_runtime_intermediate_outputs_and_op_names()
            )
            seq_len_mismatch = []
            for debug_handle, (runtime_out, num_outputs) in runtime_outs.items():
                if isinstance(runtime_out, (list, tuple)):
                    actual_len = len(runtime_out)
                    if int(num_outputs) > actual_len:
                        seq_len_mismatch.append(
                            {
                                "debug_handle": list(debug_handle),
                                "num_outputs": int(num_outputs),
                                "runtime_sequence_len": int(actual_len),
                                "runtime_ops": runtime_op_names.get(debug_handle, []),
                            }
                        )
            numeric_gap_data["_precheck"] = {
                "runtime_outputs_checked": int(len(runtime_outs)),
                "sequence_len_mismatch_count": int(len(seq_len_mismatch)),
                "sequence_len_mismatch_examples": seq_len_mismatch[:32],
            }
            if seq_len_mismatch:
                print(
                    "[inspector] warning: runtime output sequence length mismatch "
                    f"for {len(seq_len_mismatch)} debug handles."
                )
        except Exception as exc:
            numeric_gap_data["_precheck"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

        for metric in metrics:
            try:
                gap_df = inspector.calculate_numeric_gap(metric)
                max_gap = None
                worst_runtime_op = None
                worst_aot_op = None
                if len(gap_df) > 0 and "gap" in gap_df.columns:
                    gap_values = gap_df["gap"].tolist()
                    norm = []
                    for v in gap_values:
                        if isinstance(v, list):
                            norm.append(float(v[0]) if v else float("nan"))
                        else:
                            norm.append(float(v))
                    max_idx = max(range(len(norm)), key=lambda i: norm[i])
                    max_gap = norm[max_idx]
                    if "runtime_ops" in gap_df.columns:
                        worst_runtime_op = str(gap_df.iloc[max_idx]["runtime_ops"])
                    if "aot_ops" in gap_df.columns:
                        worst_aot_op = str(gap_df.iloc[max_idx]["aot_ops"])

                numeric_gap_data[metric] = {
                    "rows": int(len(gap_df)),
                    "max_gap": max_gap,
                    "worst_runtime_op": worst_runtime_op,
                    "worst_aot_op": worst_aot_op,
                    "records": gap_df.to_dict(orient="records"),
                }
                print(
                    f"[inspector] numeric_gap[{metric}] rows={len(gap_df)} "
                    f"max_gap={max_gap}"
                )
            except Exception as exc:
                tb = traceback.format_exc()
                numeric_gap_data[metric] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": tb,
                }
                print(
                    f"[inspector] warning: numeric_gap[{metric}] failed: "
                    f"{type(exc).__name__}: {exc}"
                )
    elif metrics:
        print(
            "[inspector] warning: numeric gap skipped "
            "(needs etrecord + debug buffer)."
        )

    if args.numeric_gap_json_out:
        gap_out = Path(args.numeric_gap_json_out).expanduser().resolve()
        gap_out.parent.mkdir(parents=True, exist_ok=True)
        with gap_out.open("w", encoding="utf-8") as f:
            json.dump(numeric_gap_data, f, ensure_ascii=True, indent=2, sort_keys=True)
        print(f"[inspector] numeric_gap_json={gap_out}")

    if args.summary_json_out:
        summary_out = Path(args.summary_json_out).expanduser().resolve()
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "pte": str(pte),
            "method": args.method,
            "token_id": int(args.token_id),
            "input_pos": int(args.input_pos),
            "first_forward_top1": first_forward_top1,
            "etdump_path": str(etdump_out),
            "etrecord_path": effective_etrecord_path,
            "rows": int(len(df)),
            "delegated_rows": int(
                df["is_delegated_op"].fillna(False).astype(bool).sum()
            )
            if "is_delegated_op" in df.columns
            else 0,
            "delegate_backend_counts": (
                df["delegate_backend_name"]
                .fillna("None")
                .astype(str)
                .value_counts()
                .to_dict()
                if "delegate_backend_name" in df.columns
                else {}
            ),
            "numeric_gap": numeric_gap_data,
        }
        with summary_out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)
        print(f"[inspector] summary={summary_out}")

    print(f"[inspector] etdump={etdump_out}")
    if etdump_debug_out is not None:
        print(f"[inspector] etdump_debug={etdump_debug_out}")


if __name__ == "__main__":
    main()
