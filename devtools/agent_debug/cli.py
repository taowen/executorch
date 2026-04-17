# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

from executorch.devtools.agent_debug.core import (
    append_report_notes,
    capture_runtime_target_step,
    derive_focus_handles_from_report,
    derive_scoped_delegate_focus_from_report,
    derive_refined_focus_handles,
    diagnose_target_step,
    load_scenario,
    write_report,
)
from executorch.runtime import Verification


def _load_scoped_focus_specs(raw_value: str) -> List[Dict[str, Any]]:
    source = raw_value.strip()
    if not source:
        return []

    candidate_path = Path(source)
    if candidate_path.exists():
        payload = candidate_path.read_text(encoding="utf-8")
    else:
        payload = source

    data = json.loads(payload)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(
            "--focus-delegate-events-json must resolve to a JSON object or list of objects."
        )
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"focus delegate spec at index {index} must be a JSON object."
            )
        if "instruction_id" not in item:
            raise ValueError(
                f"focus delegate spec at index {index} is missing instruction_id."
            )
    return data


def _warn_if_import_root_differs_from_cwd() -> None:
    imported_root = Path(__file__).resolve().parents[3]
    cwd = Path.cwd().resolve()
    if imported_root == cwd or imported_root in cwd.parents:
        return
    print(
        (
            "warning: agent_debug is imported from "
            f"{imported_root}, but current working directory is {cwd}. "
            "If you meant to debug the current tree, prepend "
            "PYTHONPATH=<repo>/src:<repo>."
        ),
        file=sys.stderr,
    )


def main() -> None:
    _warn_if_import_root_differs_from_cwd()
    warnings.filterwarnings(
        "ignore",
        message="Unsupported kwarg encountered: .*",
    )
    warnings.filterwarnings(
        "ignore",
        message="The given buffer is not writable.*",
    )
    parser = argparse.ArgumentParser(
        description="Agent-oriented ExecuTorch diagnosis on top of ETRecord/ETDump/Inspector."
    )
    parser.add_argument("--pte", required=True, help="Path to the target .pte file.")
    parser.add_argument("--etrecord", required=True, help="Path to the matching ETRecord.")
    parser.add_argument(
        "--scenario",
        required=True,
        help="Python scenario file defining STEPS or build_steps().",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for etdump/debug/report outputs.",
    )
    parser.add_argument(
        "--target-step-index",
        type=int,
        default=None,
        help="Which step to diagnose. Defaults to the last step.",
    )
    parser.add_argument(
        "--debug-buffer-size",
        type=int,
        default=134217728,
        help="Debug buffer size in bytes for runtime trace capture.",
    )
    parser.add_argument(
        "--reference-graph",
        default="edge_dialect_exported_program",
        help="Reference graph name passed to Inspector graph resolution.",
    )
    parser.add_argument(
        "--divergence-tolerance",
        type=float,
        default=1e-5,
        help="Absolute-difference threshold used to mark divergence.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top divergent findings to keep in the final report.",
    )
    parser.add_argument(
        "--focus-from-report",
        default=None,
        help="Optional prior agent_report.json used to derive delegate-focused recapture handles.",
    )
    parser.add_argument(
        "--focus-event-count",
        type=int,
        default=1,
        help="How many divergent events from the prior report to union into focus handles.",
    )
    parser.add_argument(
        "--focus-handles",
        default="",
        help="Optional comma-separated delegate debug handles for focused recapture.",
    )
    parser.add_argument(
        "--focus-names",
        default="",
        help="Optional comma-separated delegate string identifiers for focused recapture.",
    )
    parser.add_argument(
        "--focus-delegate-events-json",
        default="",
        help=(
            "Optional JSON string or JSON file path describing instruction-scoped "
            "delegate focus rules. Each object must include instruction_id and may "
            "optionally include debug_handles, debug_handle_ranges, and debug_names."
        ),
    )
    parser.add_argument(
        "--reference-cache-dir",
        default="outputs/agent_reference_cache",
        help="Directory used to cache focused AOT/reference replay outputs.",
    )
    parser.add_argument(
        "--delegate-abi-report",
        default="",
        help=(
            "Optional delegate_abi_report.json used to map DELEGATE_CALL outputs "
            "to exact lowered output node names."
        ),
    )
    parser.add_argument(
        "--two-stage-reference",
        dest="two_stage_reference",
        action="store_true",
        help="Run a second focused-reference pass around the first divergent event.",
    )
    parser.add_argument(
        "--no-two-stage-reference",
        dest="two_stage_reference",
        action="store_false",
        help="Disable the second focused-reference refinement pass.",
    )
    parser.add_argument(
        "--refine-event-window",
        type=int,
        default=1,
        help="How many neighboring runtime events around the first divergence to include during stage2 refinement.",
    )
    parser.add_argument(
        "--refine-top-events",
        type=int,
        default=8,
        help="How many divergent events from stage1 to union into stage2 refinement.",
    )
    parser.set_defaults(two_stage_reference=None)
    args = parser.parse_args()

    steps = load_scenario(args.scenario)
    focus_handles = []
    scoped_focus_specs: List[Dict[str, Any]] = []
    if args.focus_from_report:
        focus_handles.extend(
            derive_focus_handles_from_report(
                args.focus_from_report,
                event_count=args.focus_event_count,
            )
        )
        scoped_focus_specs.extend(
            derive_scoped_delegate_focus_from_report(
                args.focus_from_report,
                event_count=args.focus_event_count,
            )
        )
    if args.focus_handles:
        focus_handles.extend(
            int(token.strip())
            for token in args.focus_handles.split(",")
            if token.strip()
        )
    focus_handles = list(dict.fromkeys(focus_handles))
    focus_names = list(
        dict.fromkeys(
            token.strip() for token in args.focus_names.split(",") if token.strip()
        )
    )
    if args.focus_delegate_events_json:
        scoped_focus_specs.extend(
            _load_scoped_focus_specs(args.focus_delegate_events_json)
        )
    deduped_scoped_focus_specs: List[Dict[str, Any]] = []
    seen_scoped_specs = set()
    for spec in scoped_focus_specs:
        normalized_spec = json.dumps(spec, sort_keys=True, ensure_ascii=True)
        if normalized_spec in seen_scoped_specs:
            continue
        seen_scoped_specs.add(normalized_spec)
        deduped_scoped_focus_specs.append(spec)
    scoped_focus_specs = deduped_scoped_focus_specs
    runtime_trace = capture_runtime_target_step(
        pte_path=args.pte,
        steps=steps,
        output_dir=args.output_dir,
        target_step_index=args.target_step_index,
        verification=Verification.Minimal,
        debug_buffer_size=args.debug_buffer_size,
        delegate_debug_handle_focus=focus_handles or None,
        delegate_debug_name_focus=focus_names or None,
        scoped_delegate_debug_focus=scoped_focus_specs or None,
    )
    report = diagnose_target_step(
        etrecord_path=args.etrecord,
        runtime_trace=runtime_trace,
        steps=steps,
        reference_graph=args.reference_graph,
        divergence_tolerance=args.divergence_tolerance,
        top_k=args.top_k,
        reference_debug_handle_focus=focus_handles or None,
        reference_cache_dir=args.reference_cache_dir,
        delegate_abi_report_path=args.delegate_abi_report or None,
    )
    two_stage_reference = (
        args.two_stage_reference
        if args.two_stage_reference is not None
        else bool(focus_handles or focus_names or scoped_focus_specs)
    )
    if two_stage_reference:
        write_report(report, args.output_dir, file_stem="agent_report_stage1")
        refined_handles = list(focus_handles)
        refined_handles.extend(
            derive_refined_focus_handles(
                etrecord_path=args.etrecord,
                runtime_trace=runtime_trace,
                report=report,
                event_window=args.refine_event_window,
                top_event_count=args.refine_top_events,
            )
        )
        refined_handles = list(dict.fromkeys(refined_handles))
        if refined_handles and refined_handles != focus_handles:
            report = diagnose_target_step(
                etrecord_path=args.etrecord,
                runtime_trace=runtime_trace,
                steps=steps,
                reference_graph=args.reference_graph,
                divergence_tolerance=args.divergence_tolerance,
                top_k=args.top_k,
                reference_debug_handle_focus=refined_handles,
                reference_cache_dir=args.reference_cache_dir,
                delegate_abi_report_path=args.delegate_abi_report or None,
            )
            report = append_report_notes(
                report,
                [
                    "Two-stage reference refinement was enabled.",
                    "Stage1 focus handles: "
                    + (", ".join(str(handle) for handle in focus_handles) if focus_handles else "<none>"),
                    "Stage1 focus names: "
                    + (", ".join(focus_names) if focus_names else "<none>"),
                    "Stage1 scoped delegate focus specs: "
                    + (json.dumps(scoped_focus_specs) if scoped_focus_specs else "<none>"),
                    "Stage2 focus handles: "
                    + ", ".join(str(handle) for handle in refined_handles),
                ],
            )
    json_path, md_path = write_report(report, args.output_dir)
    print(report.to_markdown())
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
