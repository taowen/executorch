#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two Inspector CSV runs and emit markdown/json report."
    )
    p.add_argument("--baseline-csv", required=True)
    p.add_argument("--candidate-csv", required=True)
    p.add_argument("--baseline-summary-json", default=None)
    p.add_argument("--candidate-summary-json", default=None)
    p.add_argument("--report-out", required=True, help="Markdown report output path")
    p.add_argument("--json-out", default=None, help="Optional JSON summary output path")
    p.add_argument("--top-k", type=int, default=25)
    return p.parse_args()


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    lowered = series.fillna(False).astype(str).str.lower()
    return lowered.isin({"1", "true", "yes", "y"})


def _normalize_event_name(name: Any) -> str:
    s = str(name)
    if not s.startswith("{"):
        return s
    try:
        obj = json.loads(s)
    except Exception:
        return s

    kernel = obj.get("kernel_name")
    operator = obj.get("operator", {})
    op_name = operator.get("name") if isinstance(operator, dict) else None
    operator_id = obj.get("operator_id")
    dispatch_id = obj.get("dispatch_id")
    pieces = []
    if kernel:
        pieces.append(f"kernel={kernel}")
    if op_name:
        pieces.append(f"op={op_name}")
    if operator_id is not None:
        pieces.append(f"op_id={operator_id}")
    if dispatch_id is not None:
        pieces.append(f"dispatch={dispatch_id}")
    return "VK_EXEC[" + ", ".join(pieces) + "]" if pieces else s


def _load_summary(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _build_event_agg(df: pd.DataFrame) -> pd.DataFrame:
    local = df.copy()
    local["event_norm"] = local["event_name"].map(_normalize_event_name)
    local["avg_ms"] = pd.to_numeric(local["avg (ms)"], errors="coerce").fillna(0.0)
    grouped = (
        local.groupby("event_norm", dropna=False)
        .agg(
            count=("event_norm", "size"),
            total_avg_ms=("avg_ms", "sum"),
            delegated_count=("is_delegated_op", lambda s: _to_bool_series(s).sum()),
        )
        .reset_index()
    )
    return grouped


def _metrics(df: pd.DataFrame) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    m["rows"] = int(len(df))
    delegated_mask = _to_bool_series(df["is_delegated_op"])
    m["delegated_rows"] = int(delegated_mask.sum())
    m["delegated_ratio"] = (
        float(m["delegated_rows"]) / float(m["rows"]) if m["rows"] > 0 else 0.0
    )
    avg_ms = pd.to_numeric(df["avg (ms)"], errors="coerce").fillna(0.0)
    m["sum_avg_ms"] = float(avg_ms.sum())
    m["sum_avg_ms_delegated"] = float(avg_ms[delegated_mask].sum())

    if "delegate_backend_name" in df.columns:
        m["delegate_backend_counts"] = (
            df["delegate_backend_name"].fillna("None").astype(str).value_counts().to_dict()
        )
    else:
        m["delegate_backend_counts"] = {}

    etvk = df[df["event_name"] == "ETVK_COMPUTE_GRAPH_EXECUTE"]
    m["etvk_compute_graph_execute_sum_avg_ms"] = float(
        pd.to_numeric(etvk["avg (ms)"], errors="coerce").fillna(0.0).sum()
    )
    return m


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    baseline_csv = Path(args.baseline_csv).expanduser().resolve()
    candidate_csv = Path(args.candidate_csv).expanduser().resolve()
    if not baseline_csv.is_file():
        raise FileNotFoundError(f"baseline csv not found: {baseline_csv}")
    if not candidate_csv.is_file():
        raise FileNotFoundError(f"candidate csv not found: {candidate_csv}")

    df_base = pd.read_csv(baseline_csv)
    df_cand = pd.read_csv(candidate_csv)
    required_cols = {"event_name", "is_delegated_op", "avg (ms)"}
    if not required_cols.issubset(df_base.columns):
        raise RuntimeError(f"baseline csv missing required columns: {required_cols}")
    if not required_cols.issubset(df_cand.columns):
        raise RuntimeError(f"candidate csv missing required columns: {required_cols}")

    base_summary = _load_summary(args.baseline_summary_json)
    cand_summary = _load_summary(args.candidate_summary_json)

    base_metrics = _metrics(df_base)
    cand_metrics = _metrics(df_cand)

    base_agg = _build_event_agg(df_base)
    cand_agg = _build_event_agg(df_cand)
    merged = base_agg.merge(
        cand_agg,
        on="event_norm",
        how="outer",
        suffixes=("_base", "_cand"),
    ).fillna(0.0)
    merged["delta_total_avg_ms"] = merged["total_avg_ms_cand"] - merged["total_avg_ms_base"]
    merged["abs_delta_total_avg_ms"] = merged["delta_total_avg_ms"].abs()
    merged = merged.sort_values("abs_delta_total_avg_ms", ascending=False)
    top_changed = merged.head(max(1, args.top_k)).copy()

    report_lines = []
    report_lines.append("# Inspector Compare Report")
    report_lines.append("")
    report_lines.append("## Inputs")
    report_lines.append(f"- baseline_csv: `{baseline_csv}`")
    report_lines.append(f"- candidate_csv: `{candidate_csv}`")
    if args.baseline_summary_json:
        report_lines.append(f"- baseline_summary: `{args.baseline_summary_json}`")
    if args.candidate_summary_json:
        report_lines.append(f"- candidate_summary: `{args.candidate_summary_json}`")
    report_lines.append("")
    report_lines.append("## High-Level Metrics")
    report_lines.append("| metric | baseline | candidate | delta (cand-base) |")
    report_lines.append("|---|---:|---:|---:|")
    for key in [
        "rows",
        "delegated_rows",
        "delegated_ratio",
        "sum_avg_ms",
        "sum_avg_ms_delegated",
        "etvk_compute_graph_execute_sum_avg_ms",
    ]:
        b = base_metrics.get(key, 0.0)
        c = cand_metrics.get(key, 0.0)
        d = c - b
        report_lines.append(f"| {key} | {b:.6f} | {c:.6f} | {d:.6f} |")

    b_top1 = base_summary.get("first_forward_top1")
    c_top1 = cand_summary.get("first_forward_top1")
    if b_top1 is not None or c_top1 is not None:
        report_lines.append("")
        report_lines.append("## First-Forward Top1")
        report_lines.append("| baseline | candidate | same |")
        report_lines.append("|---:|---:|---|")
        report_lines.append(f"| {b_top1} | {c_top1} | {b_top1 == c_top1} |")

    report_lines.append("")
    report_lines.append(f"## Top {max(1, args.top_k)} Event Deltas (by |delta total_avg_ms|)")
    report_lines.append(
        "| event | baseline_total_avg_ms | candidate_total_avg_ms | delta | baseline_count | candidate_count |"
    )
    report_lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in top_changed.iterrows():
        event = str(row["event_norm"]).replace("\n", " ")
        if len(event) > 220:
            event = event[:217] + "..."
        report_lines.append(
            f"| {event} | "
            f"{float(row['total_avg_ms_base']):.6f} | "
            f"{float(row['total_avg_ms_cand']):.6f} | "
            f"{float(row['delta_total_avg_ms']):.6f} | "
            f"{int(row['count_base'])} | {int(row['count_cand'])} |"
        )

    report_lines.append("")
    report_lines.append("## Delegate Backend Counts")
    report_lines.append("- baseline:")
    for k, v in base_metrics["delegate_backend_counts"].items():
        report_lines.append(f"  - {k}: {v}")
    report_lines.append("- candidate:")
    for k, v in cand_metrics["delegate_backend_counts"].items():
        report_lines.append(f"  - {k}: {v}")

    report_text = "\n".join(report_lines) + "\n"
    report_out = Path(args.report_out).expanduser().resolve()
    _write_text(report_out, report_text)
    print(f"[compare] report={report_out}")

    payload = {
        "baseline_csv": str(baseline_csv),
        "candidate_csv": str(candidate_csv),
        "baseline_summary": base_summary,
        "candidate_summary": cand_summary,
        "baseline_metrics": base_metrics,
        "candidate_metrics": cand_metrics,
        "top_changed_events": top_changed.to_dict(orient="records"),
    }
    if args.json_out:
        json_out = Path(args.json_out).expanduser().resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        print(f"[compare] json={json_out}")


if __name__ == "__main__":
    main()
