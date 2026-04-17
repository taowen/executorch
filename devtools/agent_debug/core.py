# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import hashlib
import logging
import operator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch

from executorch.devtools.inspector import Inspector
from executorch.devtools.inspector._inspector import EXCLUDED_EVENTS_FOR_INTERMEDIATE_OUTPUT
from executorch.devtools.inspector._intermediate_output_capturer import (
    IntermediateOutputCapturer,
)
from executorch.devtools.inspector._inspector_utils import (
    DebugHandle,
    get_aot_debug_handle_to_op_name_mapping,
)
from executorch.runtime import Runtime, Verification
from torch.fx.interpreter import Interpreter


_LOW_CONFIDENCE_VULKAN_TRANSFER_KERNEL_PREFIXES = (
    "nchw_to_",
    "image_to_nchw_",
    "buffer_to_nchw_",
)

_MAPPING_CONFIDENCE_EXACT = "exact"
_MAPPING_CONFIDENCE_HIGH = "high"
_MAPPING_CONFIDENCE_HEURISTIC = "heuristic"

_INSPECTOR_LOGGER_NAME = "executorch.devtools.inspector._inspector"
_SUPPRESSED_INSPECTOR_WARNING_PREFIX = "No exact delegate debug mapping"


@dataclass(frozen=True)
class ExecutionStep:
    method: str
    inputs: Tuple[Any, ...]
    label: Optional[str] = None


@dataclass(frozen=True)
class RuntimeTraceArtifacts:
    etdump_path: str
    debug_buffer_path: Optional[str]
    target_step_index: int
    target_step_label: str


@dataclass(frozen=True)
class ReferenceCaptureArtifacts:
    outputs: Dict[DebugHandle, Any]
    outputs_by_name: Dict[str, Any]
    cache_path: Optional[str]
    cache_hit: bool
    focus_handles: Tuple[int, ...]


@dataclass(frozen=True)
class TensorDiffFinding:
    event_index: int
    event_name: str
    runtime_tensor_index: int
    runtime_debug_handle: Tuple[int, ...]
    runtime_instruction_id: Optional[int]
    runtime_delegate_debug_identifier: Optional[Union[int, str]]
    runtime_delegate_backend_name: Optional[str]
    runtime_delegate_kernel_name: Optional[str]
    runtime_delegate_operator_name: Optional[str]
    runtime_delegate_dispatch_id: Optional[int]
    aot_debug_handle: Tuple[int, ...]
    aot_ops: List[str]
    shape: Optional[Tuple[int, ...]]
    dtype: str
    max_abs_diff: float
    mean_abs_diff: float
    ambiguous_candidates: List[int]
    mapping_confidence: str = _MAPPING_CONFIDENCE_HEURISTIC
    mapping_source: str = "unknown"


@dataclass(frozen=True)
class AgentDiagnosisReport:
    target_step_index: int
    target_step_label: str
    total_events: int
    matched_outputs: int
    unmapped_outputs: int
    first_divergent: Optional[TensorDiffFinding]
    top_divergences: List[TensorDiffFinding]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_step_index": self.target_step_index,
            "target_step_label": self.target_step_label,
            "total_events": self.total_events,
            "matched_outputs": self.matched_outputs,
            "unmapped_outputs": self.unmapped_outputs,
            "first_divergent": (
                asdict(self.first_divergent) if self.first_divergent is not None else None
            ),
            "top_divergences": [asdict(item) for item in self.top_divergences],
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Agent Diagnosis Report",
            "",
            f"- target step: `{self.target_step_index}` (`{self.target_step_label}`)",
            f"- total runtime events: `{self.total_events}`",
            f"- matched outputs: `{self.matched_outputs}`",
            f"- unmapped outputs: `{self.unmapped_outputs}`",
        ]
        if self.first_divergent is None:
            lines.extend(["- first divergence: `none`", ""])
        else:
            lines.extend(
                [
                    "- first divergence:",
                    (
                        f"  event `{self.first_divergent.event_index}` "
                        f"`{self.first_divergent.event_name}` "
                        f"instruction `{self.first_divergent.runtime_instruction_id}` "
                        f"delegate_id `{self.first_divergent.runtime_delegate_debug_identifier}` "
                        f"kernel `{self.first_divergent.runtime_delegate_kernel_name}` "
                        f"handle `{self.first_divergent.aot_debug_handle}` "
                        f"ops `{', '.join(self.first_divergent.aot_ops)}` "
                        f"mapping `{self.first_divergent.mapping_confidence}` "
                        f"via `{self.first_divergent.mapping_source}` "
                        f"max `{self.first_divergent.max_abs_diff:.6g}` "
                        f"mean `{self.first_divergent.mean_abs_diff:.6g}`"
                    ),
                    "",
                ]
            )
        if self.top_divergences:
            lines.append("## Top Divergences")
            lines.append("")
            for finding in self.top_divergences:
                lines.append(
                    (
                        f"- event `{finding.event_index}` `{finding.event_name}` "
                        f"tensor `{finding.runtime_tensor_index}` "
                        f"instruction `{finding.runtime_instruction_id}` "
                        f"delegate_id `{finding.runtime_delegate_debug_identifier}` "
                        f"kernel `{finding.runtime_delegate_kernel_name}` "
                        f"handle `{finding.aot_debug_handle}` "
                        f"ops `{', '.join(finding.aot_ops)}` "
                        f"mapping `{finding.mapping_confidence}` "
                        f"via `{finding.mapping_source}` "
                        f"shape `{finding.shape}` "
                        f"max `{finding.max_abs_diff:.6g}` "
                        f"mean `{finding.mean_abs_diff:.6g}`"
                    )
                )
            lines.append("")
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")
        return "\n".join(lines)


class _InspectorDelegateWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not message.startswith(_SUPPRESSED_INSPECTOR_WARNING_PREFIX)


@contextlib.contextmanager
def _suppress_expected_inspector_delegate_warnings() -> Iterable[None]:
    logger = logging.getLogger(_INSPECTOR_LOGGER_NAME)
    warning_filter = _InspectorDelegateWarningFilter()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)


def load_scenario(path: str) -> List[ExecutionStep]:
    scenario_path = Path(path)
    module = _load_python_module(scenario_path)
    if hasattr(module, "build_steps"):
        raw_steps = module.build_steps()
    elif hasattr(module, "STEPS"):
        raw_steps = module.STEPS
    else:
        raise ValueError(
            f"Scenario file {scenario_path} must define STEPS or build_steps()."
        )
    steps: List[ExecutionStep] = []
    for index, item in enumerate(raw_steps):
        if isinstance(item, ExecutionStep):
            steps.append(item)
            continue
        if (
            isinstance(item, tuple)
            and len(item) in (2, 3)
            and isinstance(item[0], str)
            and isinstance(item[1], tuple)
        ):
            label = item[2] if len(item) == 3 else None
            steps.append(ExecutionStep(method=item[0], inputs=item[1], label=label))
            continue
        raise TypeError(
            f"Unsupported step at index {index}: expected ExecutionStep or "
            "(method, inputs[, label]) tuple."
        )
    if not steps:
        raise ValueError(f"Scenario file {scenario_path} produced no steps.")
    return steps


def capture_runtime_target_step(
    pte_path: str,
    steps: Sequence[ExecutionStep],
    output_dir: str,
    target_step_index: Optional[int] = None,
    verification: Verification = Verification.Minimal,
    debug_buffer_size: int = 0,
    file_stem: str = "agent_trace",
    delegate_debug_handle_focus: Optional[Sequence[int]] = None,
    delegate_debug_handle_focus_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    delegate_debug_name_focus: Optional[Sequence[str]] = None,
    scoped_delegate_debug_focus: Optional[Sequence[Dict[str, Any]]] = None,
) -> RuntimeTraceArtifacts:
    if target_step_index is None:
        target_step_index = len(steps) - 1
    _validate_target_step_index(steps, target_step_index)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    etdump_path = output_root / f"{file_stem}_step{target_step_index}.etdp"
    debug_buffer_path = output_root / f"{file_stem}_step{target_step_index}.bin"

    runtime = Runtime.get()
    program = runtime.load_program(
        pte_path,
        verification=verification,
        enable_etdump=True,
        debug_buffer_size=debug_buffer_size,
    )
    if debug_buffer_size > 0:
        program.set_etdump_debug_level("intermediate_outputs")
    else:
        program.set_etdump_debug_level("program_outputs")
    if (
        delegate_debug_handle_focus
        or delegate_debug_handle_focus_ranges
        or delegate_debug_name_focus
        or scoped_delegate_debug_focus
    ):
        program.set_delegate_debug_handle_focus(
            debug_handles=delegate_debug_handle_focus or (),
            debug_handle_ranges=delegate_debug_handle_focus_ranges or (),
            debug_names=delegate_debug_name_focus or (),
            scoped_debug_handles=scoped_delegate_debug_focus or (),
        )
    loaded_methods: Dict[str, Any] = {}
    for index, step in enumerate(steps[: target_step_index + 1]):
        if step.method not in loaded_methods:
            loaded_methods[step.method] = program.load_method(step.method)
        if index == target_step_index:
            break
        loaded_methods[step.method].execute(step.inputs)

    program.reset_etdump()
    target_step = steps[target_step_index]
    loaded_methods[target_step.method].execute(target_step.inputs)

    program.write_etdump_result_to_file(str(etdump_path), str(debug_buffer_path))
    resolved_debug_buffer_path = (
        str(debug_buffer_path) if debug_buffer_path.exists() else None
    )
    return RuntimeTraceArtifacts(
        etdump_path=str(etdump_path),
        debug_buffer_path=resolved_debug_buffer_path,
        target_step_index=target_step_index,
        target_step_label=_step_label(steps, target_step_index),
    )


def derive_focus_handles_from_report(
    report_path: str,
    event_count: int = 1,
) -> List[int]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    handles: List[int] = []

    def extend_from_finding(finding: Optional[Dict[str, Any]]) -> None:
        if not finding:
            return
        runtime_debug_handle = finding.get("runtime_debug_handle") or []
        for handle in runtime_debug_handle:
            if isinstance(handle, int) and handle not in handles:
                handles.append(handle)

    extend_from_finding(report.get("first_divergent"))
    for finding in report.get("top_divergences", [])[: max(0, event_count - 1)]:
        extend_from_finding(finding)
    return handles


def derive_scoped_delegate_focus_from_report(
    report_path: str,
    event_count: int = 1,
) -> List[Dict[str, Any]]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    scoped_specs: Dict[int, Dict[str, Any]] = {}

    def ensure_spec(instruction_id: int) -> Dict[str, Any]:
        spec = scoped_specs.get(instruction_id)
        if spec is None:
            spec = {
                "instruction_id": instruction_id,
                "debug_handles": [],
                "debug_names": [],
            }
            scoped_specs[instruction_id] = spec
        return spec

    def extend_from_finding(finding: Optional[Dict[str, Any]]) -> None:
        if not finding:
            return
        instruction_id = finding.get("runtime_instruction_id")
        delegate_debug_identifier = finding.get("runtime_delegate_debug_identifier")
        if not isinstance(instruction_id, int) or instruction_id < 0:
            return
        if isinstance(delegate_debug_identifier, int):
            spec = ensure_spec(instruction_id)
            if delegate_debug_identifier not in spec["debug_handles"]:
                spec["debug_handles"].append(delegate_debug_identifier)
        elif isinstance(delegate_debug_identifier, str) and delegate_debug_identifier:
            spec = ensure_spec(instruction_id)
            if delegate_debug_identifier not in spec["debug_names"]:
                spec["debug_names"].append(delegate_debug_identifier)

    extend_from_finding(report.get("first_divergent"))
    for finding in report.get("top_divergences", [])[: max(0, event_count - 1)]:
        extend_from_finding(finding)

    result: List[Dict[str, Any]] = []
    for instruction_id in sorted(scoped_specs):
        spec = dict(scoped_specs[instruction_id])
        if not spec["debug_handles"]:
            spec.pop("debug_handles")
        if not spec["debug_names"]:
            spec.pop("debug_names")
        result.append(spec)
    return result


def derive_refined_focus_handles(
    *,
    etrecord_path: str,
    runtime_trace: RuntimeTraceArtifacts,
    report: AgentDiagnosisReport,
    event_window: int = 1,
    top_event_count: int = 1,
) -> List[int]:
    inspector = Inspector(
        etdump_path=runtime_trace.etdump_path,
        etrecord=etrecord_path,
        debug_buffer_path=runtime_trace.debug_buffer_path,
    )
    execute_block = _get_last_execute_block(inspector)
    return _derive_refined_focus_handles_from_execute_block(
        execute_block,
        report,
        event_window=event_window,
        top_event_count=top_event_count,
    )


def diagnose_target_step(
    *,
    etrecord_path: str,
    runtime_trace: RuntimeTraceArtifacts,
    steps: Sequence[ExecutionStep],
    reference_graph: str = "edge_dialect_exported_program",
    divergence_tolerance: float = 1e-5,
    top_k: int = 10,
    reference_debug_handle_focus: Optional[Sequence[int]] = None,
    reference_cache_dir: Optional[str] = None,
    delegate_abi_report_path: Optional[str] = None,
) -> AgentDiagnosisReport:
    _validate_target_step_index(steps, runtime_trace.target_step_index)
    with _suppress_expected_inspector_delegate_warnings():
        inspector = Inspector(
            etdump_path=runtime_trace.etdump_path,
            etrecord=etrecord_path,
            debug_buffer_path=runtime_trace.debug_buffer_path,
        )
        reference_module, _ = inspector._resolve_reference_graph(reference_graph)
    reference_capture = _capture_reference_step_outputs(
        reference_module,
        steps,
        runtime_trace.target_step_index,
        etrecord_path=etrecord_path,
        reference_graph=reference_graph,
        debug_handle_focus=reference_debug_handle_focus,
        cache_dir=reference_cache_dir,
    )
    aot_outputs = reference_capture.outputs
    aot_outputs_by_name = reference_capture.outputs_by_name
    aot_op_names = get_aot_debug_handle_to_op_name_mapping(reference_module)
    node_name_metadata = _build_reference_node_metadata(reference_module)
    delegate_output_names_by_order = _load_delegate_output_names_by_order(
        delegate_abi_report_path
    )
    execute_block = _get_last_execute_block(inspector)
    vulkan_delegate_coverage = _load_vulkan_delegate_coverage(
        inspector,
        method_name=steps[runtime_trace.target_step_index].method,
    )

    findings: List[TensorDiffFinding] = []
    unmapped_outputs = 0
    delegate_call_ordinal = -1
    skipped_unmapped_vulkan_events: Dict[
        int, Dict[int, Dict[str, Optional[Union[int, str]]]]
    ] = {}
    for event_index, event in enumerate(execute_block.events):
        if event.name in EXCLUDED_EVENTS_FOR_INTERMEDIATE_OUTPUT or not event.debug_data:
            continue
        if event.debug_handles is None:
            _record_unmapped_vulkan_delegate_event(
                skipped_unmapped_vulkan_events,
                event,
            )
            continue
        if event.name == "DELEGATE_CALL":
            delegate_call_ordinal += 1
        runtime_tensor_offset, runtime_outputs = _trim_runtime_outputs_for_event(
            event.name,
            list(event.debug_data),
            getattr(event, "num_outputs", 1),
        )
        if not runtime_outputs:
            continue
        delegate_output_names = delegate_output_names_by_order.get(delegate_call_ordinal)
        if (
            event.name == "DELEGATE_CALL"
            and delegate_output_names
            and len(delegate_output_names) == len(runtime_outputs)
        ):
            event_findings, event_unmapped = _analyze_delegate_event_by_output_names(
                event_index=event_index,
                event_name=event.name,
                runtime_debug_handle=_normalize_debug_handle(event.debug_handles),
                runtime_instruction_id=getattr(event, "_instruction_id", None),
                runtime_delegate_debug_identifier=getattr(
                    event, "delegate_debug_identifier", None
                ),
                runtime_delegate_backend_name=getattr(
                    event, "delegate_backend_name", None
                ),
                runtime_delegate_kernel_name=getattr(
                    event, "delegate_kernel_name", None
                ),
                runtime_delegate_operator_name=getattr(
                    event, "delegate_operator_name", None
                ),
                runtime_delegate_dispatch_id=getattr(
                    event, "delegate_dispatch_id", None
                ),
                runtime_outputs=runtime_outputs,
                runtime_tensor_offset=runtime_tensor_offset,
                delegate_output_names=delegate_output_names,
                aot_outputs_by_name=aot_outputs_by_name,
                node_name_metadata=node_name_metadata,
            )
        else:
            event_findings, event_unmapped = _analyze_event(
            event_index=event_index,
            event_name=event.name,
            runtime_debug_handle=_normalize_debug_handle(event.debug_handles),
            runtime_instruction_id=getattr(event, "_instruction_id", None),
            runtime_delegate_debug_identifier=getattr(
                event, "delegate_debug_identifier", None
            ),
            runtime_delegate_backend_name=getattr(
                event, "delegate_backend_name", None
            ),
            runtime_delegate_kernel_name=getattr(
                event, "delegate_kernel_name", None
            ),
            runtime_delegate_operator_name=getattr(
                event, "delegate_operator_name", None
            ),
            runtime_delegate_dispatch_id=getattr(
                event, "delegate_dispatch_id", None
            ),
            runtime_outputs=runtime_outputs,
            runtime_tensor_offset=runtime_tensor_offset,
            aot_outputs=aot_outputs,
            aot_outputs_by_name=aot_outputs_by_name,
            aot_op_names=aot_op_names,
            node_name_metadata=node_name_metadata,
            )
        findings.extend(event_findings)
        unmapped_outputs += event_unmapped

    divergent_findings = [
        finding for finding in findings if finding.max_abs_diff > divergence_tolerance
    ]
    high_confidence_divergent_findings = [
        finding
        for finding in divergent_findings
        if not _is_low_confidence_vulkan_transfer_finding(finding)
    ]
    prioritized_divergent_findings = (
        high_confidence_divergent_findings
        if high_confidence_divergent_findings
        else divergent_findings
    )
    first_divergent = (
        min(
            prioritized_divergent_findings,
            key=lambda item: (
                item.event_index,
                -_mapping_confidence_rank(item.mapping_confidence),
                item.runtime_tensor_index,
            ),
        )
        if prioritized_divergent_findings
        else None
    )
    top_divergences = sorted(
        prioritized_divergent_findings,
        key=lambda item: (
            -_mapping_confidence_rank(item.mapping_confidence),
            -item.max_abs_diff,
            item.event_index,
            item.runtime_tensor_index,
        ),
    )[:top_k]
    notes = [
        "Runtime trace is compared against the same ETRecord reference graph.",
        "Target step capture replays all prefix steps on the same program instance so stateful models keep KV/state mutations.",
        "Delegate multi-output events are matched by shape/dtype first, then by minimal numeric gap among remaining candidates.",
    ]
    if delegate_output_names_by_order:
        notes.append(
            "Delegate call events use the tail num_outputs tensors from runtime debug_data and prefer delegate ABI output names when provided."
        )
    if reference_capture.focus_handles:
        notes.append(
            "Reference replay was focused to debug handles "
            + ", ".join(str(handle) for handle in reference_capture.focus_handles)
            + "."
        )
    if reference_capture.cache_path is not None:
        notes.append(
            "Reference replay cache: "
            + ("hit" if reference_capture.cache_hit else "miss")
            + f" ({reference_capture.cache_path})."
        )
    if high_confidence_divergent_findings and len(high_confidence_divergent_findings) != len(divergent_findings):
        notes.append(
            "Low-confidence Vulkan staging/copy kernels were deprioritized when selecting first/top divergences."
        )
    notes.extend(
        _summarize_unmapped_vulkan_delegate_events(
            skipped_unmapped_vulkan_events,
            vulkan_delegate_coverage=vulkan_delegate_coverage,
        )
    )
    return AgentDiagnosisReport(
        target_step_index=runtime_trace.target_step_index,
        target_step_label=runtime_trace.target_step_label,
        total_events=len(execute_block.events),
        matched_outputs=len(findings),
        unmapped_outputs=unmapped_outputs,
        first_divergent=first_divergent,
        top_divergences=top_divergences,
        notes=notes,
    )


def write_report(report: AgentDiagnosisReport, output_dir: str, file_stem: str = "agent_report") -> Tuple[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return str(json_path), str(md_path)


def append_report_notes(
    report: AgentDiagnosisReport,
    extra_notes: Sequence[str],
) -> AgentDiagnosisReport:
    merged_notes = list(report.notes)
    merged_notes.extend(note for note in extra_notes if note)
    return replace(report, notes=merged_notes)


def _load_python_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load scenario module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_low_confidence_vulkan_transfer_finding(finding: TensorDiffFinding) -> bool:
    if finding.runtime_delegate_backend_name != "VulkanBackend":
        return False
    kernel_name = finding.runtime_delegate_kernel_name or ""
    return kernel_name.startswith(_LOW_CONFIDENCE_VULKAN_TRANSFER_KERNEL_PREFIXES)


def _mapping_confidence_rank(confidence: str) -> int:
    if confidence == _MAPPING_CONFIDENCE_EXACT:
        return 3
    if confidence == _MAPPING_CONFIDENCE_HIGH:
        return 2
    return 1


def _debug_handle_mapping_confidence(
    key: DebugHandle,
    aot_op_names: Dict[DebugHandle, List[str]],
    *,
    candidate_count: int,
) -> Tuple[str, str]:
    mapped_ops = aot_op_names.get(key, [])
    if len(mapped_ops) <= 1 and candidate_count == 1:
        return (_MAPPING_CONFIDENCE_EXACT, "debug_handle")
    if len(mapped_ops) <= 1:
        return (_MAPPING_CONFIDENCE_HIGH, "debug_handle+numeric_gap")
    return (_MAPPING_CONFIDENCE_HEURISTIC, "reused_debug_handle")


def _load_vulkan_delegate_coverage(
    inspector: Inspector,
    *,
    method_name: str,
) -> Dict[int, int]:
    etrecord = getattr(inspector, "_etrecord", None)
    if etrecord is None:
        return {}

    delegate_map = getattr(etrecord, "_delegate_map", {}) or {}
    method_delegate_map = delegate_map.get(method_name) or {}
    coverage: Dict[int, int] = {}
    for instruction_id, metadata in method_delegate_map.items():
        if not isinstance(metadata, dict) or metadata.get("name") != "VulkanBackend":
            continue
        local_ids = []
        for key in (metadata.get("delegate_map") or {}).keys():
            try:
                local_ids.append(int(key))
            except (TypeError, ValueError):
                continue
        if local_ids:
            coverage[int(instruction_id)] = max(local_ids)
    return coverage


def _extract_vulkan_delegate_local_id(
    delegate_debug_identifier: Optional[Union[int, str]],
) -> Optional[int]:
    if isinstance(delegate_debug_identifier, int):
        return delegate_debug_identifier
    if isinstance(delegate_debug_identifier, str):
        stripped = delegate_debug_identifier.strip()
        if stripped.isdigit():
            return int(stripped)
        marker = '"delegate_debug_id":'
        if marker in stripped:
            remainder = stripped.split(marker, 1)[1].lstrip()
            digits = []
            for char in remainder:
                if char.isdigit() or (char == "-" and not digits):
                    digits.append(char)
                    continue
                break
            if digits:
                try:
                    return int("".join(digits))
                except ValueError:
                    return None
    return None


def _record_unmapped_vulkan_delegate_event(
    sink: Dict[int, Dict[int, Dict[str, Optional[Union[int, str]]]]],
    event: Any,
) -> None:
    if getattr(event, "delegate_backend_name", None) != "VulkanBackend":
        return
    delegate_debug_identifier = getattr(event, "delegate_debug_identifier", None)
    if delegate_debug_identifier is None:
        return
    if isinstance(delegate_debug_identifier, str) and delegate_debug_identifier.startswith(
        "ETVK_"
    ):
        return
    instruction_id = getattr(event, "_instruction_id", None)
    local_id = _extract_vulkan_delegate_local_id(delegate_debug_identifier)
    if instruction_id is None or local_id is None:
        return
    instruction_bucket = sink.setdefault(int(instruction_id), {})
    entry = instruction_bucket.setdefault(
        local_id,
        {
            "kernel_name": None,
            "operator_name": None,
            "event_name": getattr(event, "name", None),
        },
    )
    if entry.get("kernel_name") is None and getattr(event, "delegate_kernel_name", None):
        entry["kernel_name"] = getattr(event, "delegate_kernel_name", None)
    if entry.get("operator_name") is None and getattr(
        event, "delegate_operator_name", None
    ):
        entry["operator_name"] = getattr(event, "delegate_operator_name", None)


def _summarize_unmapped_vulkan_delegate_events(
    skipped_events: Dict[int, Dict[int, Dict[str, Optional[Union[int, str]]]]],
    *,
    vulkan_delegate_coverage: Dict[int, int],
) -> List[str]:
    notes: List[str] = []
    for instruction_id in sorted(skipped_events.keys()):
        local_entries = skipped_events[instruction_id]
        if not local_entries:
            continue
        local_ids = sorted(local_entries.keys())
        kernel_examples = [
            str(local_entries[local_id]["kernel_name"])
            for local_id in local_ids
            if local_entries[local_id].get("kernel_name")
        ]
        operator_examples = [
            str(local_entries[local_id]["operator_name"])
            for local_id in local_ids
            if local_entries[local_id].get("operator_name")
        ]
        coverage_note = ""
        max_mapped_local_id = vulkan_delegate_coverage.get(instruction_id)
        if max_mapped_local_id is not None:
            coverage_note = (
                f" ETRecord delegate_map for this instruction stops at local id {max_mapped_local_id}."
            )
        example_parts = []
        if kernel_examples:
            example_parts.append(
                "kernels "
                + ", ".join(dict.fromkeys(kernel_examples))[:160]
            )
        if operator_examples:
            example_parts.append(
                "ops " + ", ".join(dict.fromkeys(operator_examples))[:160]
            )
        example_note = f" Examples: {'; '.join(example_parts)}." if example_parts else ""
        notes.append(
            "Vulkan instruction "
            + str(instruction_id)
            + " emitted "
            + str(len(local_ids))
            + " debug tensors without exact ETRecord mapping for local ids "
            + ", ".join(str(local_id) for local_id in local_ids[:12])
            + ("..." if len(local_ids) > 12 else "")
            + "."
            + coverage_note
            + example_note
        )
    return notes


def _validate_target_step_index(
    steps: Sequence[ExecutionStep], target_step_index: int
) -> None:
    if target_step_index < 0 or target_step_index >= len(steps):
        raise IndexError(
            f"target_step_index {target_step_index} is out of range for {len(steps)} step(s)"
        )


def _step_label(steps: Sequence[ExecutionStep], index: int) -> str:
    step = steps[index]
    return step.label or f"{step.method}[{index}]"


def _capture_reference_step_outputs(
    reference_module: torch.fx.GraphModule,
    steps: Sequence[ExecutionStep],
    target_step_index: int,
    *,
    etrecord_path: str,
    reference_graph: str,
    debug_handle_focus: Optional[Sequence[int]] = None,
    cache_dir: Optional[str] = None,
) -> ReferenceCaptureArtifacts:
    normalized_focus = tuple(
        sorted({int(handle) for handle in (debug_handle_focus or ())})
    )
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = _build_reference_capture_cache_key(
            etrecord_path=etrecord_path,
            reference_graph=reference_graph,
            steps=steps,
            target_step_index=target_step_index,
            debug_handle_focus=normalized_focus,
        )
        cache_path = cache_root / f"{cache_key}.pt"
        if cache_path.exists():
            cached_payload = torch.load(
                cache_path, map_location="cpu", weights_only=False
            )
            if isinstance(cached_payload, dict) and "outputs" in cached_payload:
                outputs = cached_payload.get("outputs", {})
                outputs_by_name = cached_payload.get("outputs_by_name", {})
            else:
                outputs = cached_payload
                outputs_by_name = {}
            return ReferenceCaptureArtifacts(
                outputs=outputs,
                outputs_by_name=outputs_by_name,
                cache_path=str(cache_path),
                cache_hit=True,
                focus_handles=normalized_focus,
            )

    captured_outputs: Dict[DebugHandle, Any] = {}
    captured_outputs_by_name: Dict[str, Any] = {}
    allowlist = (
        {(handle,) for handle in normalized_focus}
        if normalized_focus
        else None
    )
    for index, step in enumerate(steps[: target_step_index + 1]):
        capturer = _CombinedOutputCapturer(
            reference_module,
            debug_handle_allowlist=allowlist,
        )
        step_outputs, step_outputs_by_name = _run_combined_reference_capture_step(
            capturer, step.inputs
        )
        if index == target_step_index:
            captured_outputs = step_outputs
            captured_outputs_by_name = step_outputs_by_name
    if cache_path is not None:
        torch.save(
            {
                "outputs": captured_outputs,
                "outputs_by_name": captured_outputs_by_name,
            },
            cache_path,
        )
    return ReferenceCaptureArtifacts(
        outputs=captured_outputs,
        outputs_by_name=captured_outputs_by_name,
        cache_path=str(cache_path) if cache_path is not None else None,
        cache_hit=False,
        focus_handles=normalized_focus,
    )


def _get_last_execute_block(inspector: Inspector) -> Any:
    execute_blocks = [
        block for block in inspector.event_blocks if getattr(block, "name", None) == "Execute"
    ]
    if not execute_blocks:
        raise ValueError("No Execute event block found in runtime ETDump.")
    def debug_event_count(block: Any) -> int:
        return sum(1 for event in block.events if getattr(event, "debug_data", None))

    blocks_with_debug = [block for block in execute_blocks if debug_event_count(block) > 0]
    if blocks_with_debug:
        return max(blocks_with_debug, key=debug_event_count)
    return execute_blocks[-1]


def _normalize_debug_handle(debug_handle: Any) -> Tuple[int, ...]:
    if isinstance(debug_handle, int):
        return (debug_handle,)
    return tuple(debug_handle)


def _derive_refined_focus_handles_from_execute_block(
    execute_block: Any,
    report: AgentDiagnosisReport,
    *,
    event_window: int = 1,
    top_event_count: int = 1,
) -> List[int]:
    if report.first_divergent is None:
        return []

    handles: List[int] = []
    events = list(getattr(execute_block, "events", []))

    def extend_from_event_index(event_index: int) -> None:
        if event_index < 0 or event_index >= len(events):
            return
        event = events[event_index]
        if (
            getattr(event, "name", None) in EXCLUDED_EVENTS_FOR_INTERMEDIATE_OUTPUT
            or getattr(event, "debug_handles", None) is None
            or not getattr(event, "debug_data", None)
        ):
            return
        for handle in _normalize_debug_handle(event.debug_handles):
            if handle not in handles:
                handles.append(handle)

    center = report.first_divergent.event_index
    for event_index in range(
        max(0, center - max(0, event_window)),
        min(len(events), center + max(0, event_window) + 1),
    ):
        extend_from_event_index(event_index)

    seen_top_event_indices = set()
    for finding in report.top_divergences:
        if len(seen_top_event_indices) >= max(0, top_event_count):
            break
        if finding.event_index in seen_top_event_indices:
            continue
        seen_top_event_indices.add(finding.event_index)
        extend_from_event_index(finding.event_index)

    return handles


def _run_reference_capture_step(
    capturer: IntermediateOutputCapturer,
    step_inputs: Tuple[Any, ...],
) -> Dict[DebugHandle, Any]:
    try:
        return capturer.run_and_capture(step_inputs)
    except Exception:
        pass
    fallback_capturer = IntermediateOutputCapturer(
        capturer.module,
        debug_handle_allowlist=capturer.debug_handle_allowlist,
    )
    return fallback_capturer.run_and_capture(*step_inputs)


class _NodeNameOutputCapturer(Interpreter):
    def __init__(self, module: torch.fx.GraphModule) -> None:
        super().__init__(module)

    def run_and_capture(self, *args, **kwargs) -> Dict[str, Any]:
        captured_outputs: Dict[str, Any] = {}

        def capture_run_node(node: torch.fx.Node) -> Any:
            result = super(_NodeNameOutputCapturer, self).run_node(node)
            if node.op == "call_function":
                captured_outputs[node.name] = _clone_debug_value(result)
            return result

        original_run_node = self.run_node
        self.run_node = capture_run_node
        self.run(*args, **kwargs)
        self.run_node = original_run_node
        return captured_outputs


class _CombinedOutputCapturer(Interpreter):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        debug_handle_allowlist: Optional[Set[DebugHandle]] = None,
    ) -> None:
        super().__init__(module)
        self.debug_handle_allowlist = debug_handle_allowlist

    def run_and_capture(
        self, *args, **kwargs
    ) -> Tuple[Dict[DebugHandle, Any], Dict[str, Any]]:
        captured_outputs: Dict[DebugHandle, Any] = {}
        captured_outputs_by_name: Dict[str, Any] = {}

        def capture_run_node(node: torch.fx.Node) -> Any:
            result = super(_CombinedOutputCapturer, self).run_node(node)
            if node.op != "call_function":
                return result

            captured_outputs_by_name[node.name] = _clone_debug_value(result)

            if node.target == operator.getitem or node.meta.get("debug_handle") is None:
                return result

            debug_handle = node.meta["debug_handle"]
            key = (
                (debug_handle,)
                if isinstance(debug_handle, int)
                else tuple(debug_handle)
            )
            if (
                self.debug_handle_allowlist is not None
                and key not in self.debug_handle_allowlist
            ):
                return result
            captured_outputs[key] = _clone_debug_value(result)
            return result

        original_run_node = self.run_node
        self.run_node = capture_run_node
        self.run(*args, **kwargs)
        self.run_node = original_run_node
        return captured_outputs, captured_outputs_by_name


def _clone_debug_value(result: Any) -> Any:
    if isinstance(result, torch.Tensor):
        return result.detach().clone()
    if isinstance(result, (tuple, list)):
        return [
            item.detach().clone() if isinstance(item, torch.Tensor) else item
            for item in result
        ]
    return result


def _run_reference_capture_step_by_name(
    reference_module: torch.fx.GraphModule,
    step_inputs: Tuple[Any, ...],
) -> Dict[str, Any]:
    capturer = _NodeNameOutputCapturer(reference_module)
    try:
        return capturer.run_and_capture(step_inputs)
    except Exception:
        pass
    fallback_capturer = _NodeNameOutputCapturer(reference_module)
    return fallback_capturer.run_and_capture(*step_inputs)


def _run_combined_reference_capture_step(
    capturer: _CombinedOutputCapturer,
    step_inputs: Tuple[Any, ...],
) -> Tuple[Dict[DebugHandle, Any], Dict[str, Any]]:
    try:
        return capturer.run_and_capture(step_inputs)
    except Exception:
        pass
    fallback_capturer = _CombinedOutputCapturer(
        capturer.module,
        debug_handle_allowlist=capturer.debug_handle_allowlist,
    )
    return fallback_capturer.run_and_capture(*step_inputs)


def _build_reference_capture_cache_key(
    *,
    etrecord_path: str,
    reference_graph: str,
    steps: Sequence[ExecutionStep],
    target_step_index: int,
    debug_handle_focus: Sequence[int],
) -> str:
    etrecord_stat = Path(etrecord_path).stat()
    payload = {
        "etrecord_path": str(Path(etrecord_path).resolve()),
        "etrecord_mtime_ns": etrecord_stat.st_mtime_ns,
        "etrecord_size": etrecord_stat.st_size,
        "reference_graph": reference_graph,
        "target_step_index": target_step_index,
        "focus_handles": list(debug_handle_focus),
        "steps": [
            _step_fingerprint(step)
            for step in steps[: target_step_index + 1]
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return f"reference_capture_{digest[:16]}"


def _step_fingerprint(step: ExecutionStep) -> Dict[str, Any]:
    return {
        "method": step.method,
        "label": step.label,
        "inputs": _value_fingerprint(step.inputs),
    }


def _value_fingerprint(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "digest": _tensor_digest(value),
        }
    if isinstance(value, (list, tuple)):
        return [_value_fingerprint(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _value_fingerprint(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def _tensor_digest(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    try:
        payload = cpu_tensor.numpy().tobytes()
    except Exception:
        buffer = io.BytesIO()
        torch.save(cpu_tensor, buffer)
        payload = buffer.getvalue()
    return hashlib.sha256(payload).hexdigest()


def _unique_handles_in_order(handles: Iterable[int]) -> List[int]:
    seen = set()
    ordered = []
    for handle in handles:
        if handle not in seen:
            ordered.append(handle)
            seen.add(handle)
    return ordered


def _trim_runtime_outputs_for_event(
    event_name: str,
    runtime_outputs: List[Any],
    num_outputs: Optional[int],
) -> Tuple[int, List[Any]]:
    if num_outputs is not None and num_outputs > 0 and len(runtime_outputs) > num_outputs:
        offset = len(runtime_outputs) - num_outputs
        return offset, runtime_outputs[offset:]
    return 0, runtime_outputs


def _build_reference_node_metadata(
    reference_module: torch.fx.GraphModule,
) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for node in reference_module.graph.nodes:
        if node.op != "call_function":
            continue
        debug_handle = node.meta.get("debug_handle")
        if isinstance(debug_handle, int):
            normalized_debug_handle: Tuple[int, ...] = (debug_handle,)
        elif isinstance(debug_handle, (tuple, list)):
            normalized_debug_handle = tuple(debug_handle)
        else:
            normalized_debug_handle = ()
        metadata[node.name] = {
            "debug_handle": normalized_debug_handle,
            "op_name": node.name,
            "target_name": _target_name(node.target),
        }
    return metadata


def _load_delegate_output_names_by_order(
    delegate_abi_report_path: Optional[str],
) -> Dict[int, List[str]]:
    if not delegate_abi_report_path:
        return {}
    payload = json.loads(Path(delegate_abi_report_path).read_text(encoding="utf-8"))
    delegates = payload.get("delegates") or []
    result: Dict[int, List[str]] = {}
    for delegate in delegates:
        delegate_order = delegate.get("delegate_order")
        output_specs = delegate.get("output_specs") or []
        if not isinstance(delegate_order, int):
            continue
        result[delegate_order] = [
            spec.get("name")
            for spec in output_specs
            if isinstance(spec, dict) and isinstance(spec.get("name"), str)
        ]
    return result


def _analyze_event(
    *,
    event_index: int,
    event_name: str,
    runtime_debug_handle: Tuple[int, ...],
    runtime_instruction_id: Optional[int],
    runtime_delegate_debug_identifier: Optional[Union[int, str]],
    runtime_delegate_backend_name: Optional[str],
    runtime_delegate_kernel_name: Optional[str],
    runtime_delegate_operator_name: Optional[str],
    runtime_delegate_dispatch_id: Optional[int],
    runtime_outputs: List[Any],
    runtime_tensor_offset: int,
    aot_outputs: Dict[DebugHandle, Any],
    aot_outputs_by_name: Dict[str, Any],
    aot_op_names: Dict[DebugHandle, List[str]],
    node_name_metadata: Dict[str, Dict[str, Any]],
) -> Tuple[List[TensorDiffFinding], int]:
    available_handles = _unique_handles_in_order(runtime_debug_handle)
    findings: List[TensorDiffFinding] = []
    unmapped = 0
    event_op_hint = _extract_event_op_hint(event_name)
    runtime_operator_name_matches = bool(runtime_delegate_operator_name)
    if (
        runtime_delegate_debug_identifier is not None
        and event_name != "DELEGATE_CALL"
        and len(runtime_debug_handle) == 1
        and len(runtime_outputs) > 1
    ):
        return _analyze_scoped_delegate_event(
            event_index=event_index,
            event_name=event_name,
            runtime_debug_handle=runtime_debug_handle,
            runtime_instruction_id=runtime_instruction_id,
            runtime_delegate_debug_identifier=runtime_delegate_debug_identifier,
            runtime_delegate_backend_name=runtime_delegate_backend_name,
            runtime_delegate_kernel_name=runtime_delegate_kernel_name,
            runtime_delegate_operator_name=runtime_delegate_operator_name,
            runtime_delegate_dispatch_id=runtime_delegate_dispatch_id,
            runtime_outputs=runtime_outputs,
            runtime_tensor_offset=runtime_tensor_offset,
            aot_outputs=aot_outputs,
            aot_outputs_by_name=aot_outputs_by_name,
            aot_op_names=aot_op_names,
            node_name_metadata=node_name_metadata,
        )
    for runtime_tensor_index, runtime_output in enumerate(runtime_outputs):
        named_candidates = []
        if len(runtime_debug_handle) == 1:
            handle = runtime_debug_handle[0]
            for node_name, metadata in node_name_metadata.items():
                if metadata.get("debug_handle") != (handle,):
                    continue
                aot_output = aot_outputs_by_name.get(node_name)
                if not _are_outputs_comparable(aot_output, runtime_output):
                    continue
                target_name = metadata.get("target_name")
                if event_op_hint and not _matches_event_op_hint(
                    event_op_hint, target_name
                ):
                    continue
                if runtime_delegate_operator_name and not _matches_runtime_operator_name(
                    runtime_delegate_operator_name,
                    target_name,
                ):
                    continue
                max_abs_diff, mean_abs_diff = _compute_diff_stats(
                    aot_output, runtime_output
                )
                named_candidates.append(
                    (node_name, max_abs_diff, mean_abs_diff, metadata)
                )
        if named_candidates:
            named_candidates.sort(key=lambda item: (item[1], item[2], item[0]))
            best_name, max_abs_diff, mean_abs_diff, metadata = named_candidates[0]
            findings.append(
                TensorDiffFinding(
                    event_index=event_index,
                    event_name=event_name,
                    runtime_tensor_index=runtime_tensor_offset + runtime_tensor_index,
                    runtime_debug_handle=runtime_debug_handle,
                    runtime_instruction_id=runtime_instruction_id,
                    runtime_delegate_debug_identifier=runtime_delegate_debug_identifier,
                    runtime_delegate_backend_name=runtime_delegate_backend_name,
                    runtime_delegate_kernel_name=runtime_delegate_kernel_name,
                    runtime_delegate_operator_name=runtime_delegate_operator_name,
                    runtime_delegate_dispatch_id=runtime_delegate_dispatch_id,
                    aot_debug_handle=tuple(metadata.get("debug_handle", ())),
                    aot_ops=[best_name],
                    shape=_output_shape(runtime_output),
                    dtype=_output_dtype(runtime_output),
                    max_abs_diff=max_abs_diff,
                    mean_abs_diff=mean_abs_diff,
                    ambiguous_candidates=[],
                    mapping_confidence=(
                        _MAPPING_CONFIDENCE_HIGH
                        if event_op_hint or runtime_operator_name_matches
                        else _MAPPING_CONFIDENCE_HEURISTIC
                    ),
                    mapping_source=(
                        "debug_handle+operator_name"
                        if event_op_hint or runtime_operator_name_matches
                        else "debug_handle+node_name"
                    ),
                )
            )
            continue
        candidates = []
        for handle in available_handles:
            if _runtime_operator_conflicts_with_handle_nodes(
                handle,
                runtime_delegate_operator_name,
                node_name_metadata,
            ):
                continue
            key = (handle,)
            if key not in aot_outputs:
                continue
            aot_output = aot_outputs[key]
            if not _are_outputs_comparable(aot_output, runtime_output):
                continue
            max_abs_diff, mean_abs_diff = _compute_diff_stats(aot_output, runtime_output)
            candidates.append((handle, max_abs_diff, mean_abs_diff))
        if not candidates:
            unmapped += 1
            continue
        candidates.sort(key=lambda item: (item[1], item[2], item[0]))
        best_handle, max_abs_diff, mean_abs_diff = candidates[0]
        available_handles = [handle for handle in available_handles if handle != best_handle]
        key = (best_handle,)
        mapping_confidence, mapping_source = _debug_handle_mapping_confidence(
            key,
            aot_op_names,
            candidate_count=len(candidates),
        )
        findings.append(
            TensorDiffFinding(
                event_index=event_index,
                event_name=event_name,
                runtime_tensor_index=runtime_tensor_offset + runtime_tensor_index,
                runtime_debug_handle=runtime_debug_handle,
                runtime_instruction_id=runtime_instruction_id,
                runtime_delegate_debug_identifier=runtime_delegate_debug_identifier,
                runtime_delegate_backend_name=runtime_delegate_backend_name,
                runtime_delegate_kernel_name=runtime_delegate_kernel_name,
                runtime_delegate_operator_name=runtime_delegate_operator_name,
                runtime_delegate_dispatch_id=runtime_delegate_dispatch_id,
                aot_debug_handle=key,
                aot_ops=aot_op_names.get(key, [f"debug_handle_{best_handle}"]),
                shape=_output_shape(runtime_output),
                dtype=_output_dtype(runtime_output),
                max_abs_diff=max_abs_diff,
                mean_abs_diff=mean_abs_diff,
                ambiguous_candidates=[candidate[0] for candidate in candidates[1:]],
                mapping_confidence=mapping_confidence,
                mapping_source=mapping_source,
            )
        )
    return findings, unmapped


def _analyze_scoped_delegate_event(
    *,
    event_index: int,
    event_name: str,
    runtime_debug_handle: Tuple[int, ...],
    runtime_instruction_id: Optional[int],
    runtime_delegate_debug_identifier: Optional[Union[int, str]],
    runtime_delegate_backend_name: Optional[str],
    runtime_delegate_kernel_name: Optional[str],
    runtime_delegate_operator_name: Optional[str],
    runtime_delegate_dispatch_id: Optional[int],
    runtime_outputs: List[Any],
    runtime_tensor_offset: int,
    aot_outputs: Dict[DebugHandle, Any],
    aot_outputs_by_name: Dict[str, Any],
    aot_op_names: Dict[DebugHandle, List[str]],
    node_name_metadata: Dict[str, Dict[str, Any]],
) -> Tuple[List[TensorDiffFinding], int]:
    handle = runtime_debug_handle[0]
    named_candidates = []
    runtime_operator_name_matches = bool(runtime_delegate_operator_name)
    for node_name, metadata in node_name_metadata.items():
        if metadata.get("debug_handle") != (handle,):
            continue
        aot_output = aot_outputs_by_name.get(node_name)
        if aot_output is None:
            continue
        target_name = metadata.get("target_name")
        if runtime_delegate_operator_name and not _matches_runtime_operator_name(
            runtime_delegate_operator_name,
            target_name,
        ):
            continue
        for runtime_tensor_index, runtime_output in enumerate(runtime_outputs):
            if not _are_outputs_comparable(aot_output, runtime_output):
                continue
            max_abs_diff, mean_abs_diff = _compute_diff_stats(aot_output, runtime_output)
            named_candidates.append(
                (
                    runtime_tensor_index,
                    node_name,
                    max_abs_diff,
                    mean_abs_diff,
                    metadata,
                )
            )
    if named_candidates:
        named_candidates.sort(key=lambda item: (item[2], item[3], item[0], item[1]))
        runtime_tensor_index, best_name, max_abs_diff, mean_abs_diff, metadata = (
            named_candidates[0]
        )
        return (
            [
                TensorDiffFinding(
                    event_index=event_index,
                    event_name=event_name,
                    runtime_tensor_index=runtime_tensor_offset + runtime_tensor_index,
                    runtime_debug_handle=runtime_debug_handle,
                    runtime_instruction_id=runtime_instruction_id,
                    runtime_delegate_debug_identifier=runtime_delegate_debug_identifier,
                    runtime_delegate_backend_name=runtime_delegate_backend_name,
                    runtime_delegate_kernel_name=runtime_delegate_kernel_name,
                    runtime_delegate_operator_name=runtime_delegate_operator_name,
                    runtime_delegate_dispatch_id=runtime_delegate_dispatch_id,
                    aot_debug_handle=tuple(metadata.get("debug_handle", ())),
                    aot_ops=[best_name],
                    shape=_output_shape(runtime_outputs[runtime_tensor_index]),
                    dtype=_output_dtype(runtime_outputs[runtime_tensor_index]),
                    max_abs_diff=max_abs_diff,
                    mean_abs_diff=mean_abs_diff,
                    ambiguous_candidates=[],
                    mapping_confidence=(
                        _MAPPING_CONFIDENCE_HIGH
                        if runtime_operator_name_matches
                        else _MAPPING_CONFIDENCE_HEURISTIC
                    ),
                    mapping_source=(
                        "debug_handle+operator_name"
                        if runtime_operator_name_matches
                        else "debug_handle+node_name"
                    ),
                )
            ],
            0,
        )

    key = (handle,)
    if _runtime_operator_conflicts_with_handle_nodes(
        handle,
        runtime_delegate_operator_name,
        node_name_metadata,
    ):
        return [], 1
    aot_output = aot_outputs.get(key)
    if aot_output is None:
        return [], 1

    candidates = []
    for runtime_tensor_index, runtime_output in enumerate(runtime_outputs):
        if not _are_outputs_comparable(aot_output, runtime_output):
            continue
        max_abs_diff, mean_abs_diff = _compute_diff_stats(aot_output, runtime_output)
        candidates.append((runtime_tensor_index, max_abs_diff, mean_abs_diff))
    if not candidates:
        return [], 1

    candidates.sort(key=lambda item: (item[1], item[2], item[0]))
    runtime_tensor_index, max_abs_diff, mean_abs_diff = candidates[0]
    mapping_confidence, mapping_source = _debug_handle_mapping_confidence(
        key,
        aot_op_names,
        candidate_count=len(candidates),
    )
    return (
        [
            TensorDiffFinding(
                event_index=event_index,
                event_name=event_name,
                runtime_tensor_index=runtime_tensor_offset + runtime_tensor_index,
                runtime_debug_handle=runtime_debug_handle,
                runtime_instruction_id=runtime_instruction_id,
                runtime_delegate_debug_identifier=runtime_delegate_debug_identifier,
                runtime_delegate_backend_name=runtime_delegate_backend_name,
                runtime_delegate_kernel_name=runtime_delegate_kernel_name,
                runtime_delegate_operator_name=runtime_delegate_operator_name,
                runtime_delegate_dispatch_id=runtime_delegate_dispatch_id,
                aot_debug_handle=key,
                aot_ops=aot_op_names.get(key, [f"debug_handle_{handle}"]),
                shape=_output_shape(runtime_outputs[runtime_tensor_index]),
                dtype=_output_dtype(runtime_outputs[runtime_tensor_index]),
                max_abs_diff=max_abs_diff,
                mean_abs_diff=mean_abs_diff,
                ambiguous_candidates=[],
                mapping_confidence=mapping_confidence,
                mapping_source=mapping_source,
            )
        ],
        0,
    )


def _analyze_delegate_event_by_output_names(
    *,
    event_index: int,
    event_name: str,
    runtime_debug_handle: Tuple[int, ...],
    runtime_instruction_id: Optional[int],
    runtime_delegate_debug_identifier: Optional[Union[int, str]],
    runtime_delegate_backend_name: Optional[str],
    runtime_delegate_kernel_name: Optional[str],
    runtime_delegate_operator_name: Optional[str],
    runtime_delegate_dispatch_id: Optional[int],
    runtime_outputs: List[Any],
    runtime_tensor_offset: int,
    delegate_output_names: Sequence[str],
    aot_outputs_by_name: Dict[str, Any],
    node_name_metadata: Dict[str, Dict[str, Any]],
) -> Tuple[List[TensorDiffFinding], int]:
    findings: List[TensorDiffFinding] = []
    unmapped = 0
    for runtime_tensor_index, (runtime_output, output_name) in enumerate(
        zip(runtime_outputs, delegate_output_names)
    ):
        aot_output = aot_outputs_by_name.get(output_name)
        if not _are_outputs_comparable(aot_output, runtime_output):
            unmapped += 1
            continue
        max_abs_diff, mean_abs_diff = _compute_diff_stats(aot_output, runtime_output)
        metadata = node_name_metadata.get(output_name, {})
        findings.append(
            TensorDiffFinding(
                event_index=event_index,
                event_name=event_name,
                runtime_tensor_index=runtime_tensor_offset + runtime_tensor_index,
                runtime_debug_handle=runtime_debug_handle,
                runtime_instruction_id=runtime_instruction_id,
                runtime_delegate_debug_identifier=runtime_delegate_debug_identifier,
                runtime_delegate_backend_name=runtime_delegate_backend_name,
                runtime_delegate_kernel_name=runtime_delegate_kernel_name,
                runtime_delegate_operator_name=runtime_delegate_operator_name,
                runtime_delegate_dispatch_id=runtime_delegate_dispatch_id,
                aot_debug_handle=tuple(metadata.get("debug_handle", ())),
                aot_ops=[metadata.get("op_name", output_name)],
                shape=_output_shape(runtime_output),
                dtype=_output_dtype(runtime_output),
                max_abs_diff=max_abs_diff,
                mean_abs_diff=mean_abs_diff,
                ambiguous_candidates=[],
                mapping_confidence=_MAPPING_CONFIDENCE_EXACT,
                mapping_source="delegate_output_name",
            )
        )
    return findings, unmapped


def _are_outputs_comparable(reference_output: Any, runtime_output: Any) -> bool:
    if isinstance(reference_output, torch.Tensor) and isinstance(runtime_output, torch.Tensor):
        return (
            tuple(reference_output.shape) == tuple(runtime_output.shape)
            and reference_output.dtype == runtime_output.dtype
        )
    if isinstance(reference_output, (bool, int, float)) and isinstance(
        runtime_output, (bool, int, float)
    ):
        return True
    return False


def _compute_diff_stats(reference_output: Any, runtime_output: Any) -> Tuple[float, float]:
    if isinstance(reference_output, torch.Tensor) and isinstance(runtime_output, torch.Tensor):
        ref_tensor = reference_output.detach().cpu()
        runtime_tensor = runtime_output.detach().cpu()
        if ref_tensor.dtype == torch.bool and runtime_tensor.dtype == torch.bool:
            mismatch = torch.logical_xor(ref_tensor, runtime_tensor).to(torch.float32)
            return float(mismatch.max()), float(mismatch.mean())
        diff = (ref_tensor.to(torch.float64) - runtime_tensor.to(torch.float64)).abs()
        return float(diff.max()), float(diff.mean())
    ref_value = float(reference_output)
    runtime_value = float(runtime_output)
    diff = abs(ref_value - runtime_value)
    return diff, diff


def _output_shape(output: Any) -> Optional[Tuple[int, ...]]:
    if isinstance(output, torch.Tensor):
        return tuple(output.shape)
    return None


def _output_dtype(output: Any) -> str:
    if isinstance(output, torch.Tensor):
        return str(output.dtype)
    return type(output).__name__


def _extract_event_op_hint(event_name: str) -> Optional[str]:
    prefix = "native_call_"
    if not event_name.startswith(prefix):
        return None
    suffix = event_name[len(prefix) :]
    if suffix.endswith("_out"):
        suffix = suffix[: -len("_out")]
    elif suffix.endswith(".out"):
        suffix = suffix[: -len(".out")]
    return suffix or None


def _canonical_target_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.replace("::", ".")


def _matches_event_op_hint(event_op_hint: str, target_name: Optional[str]) -> bool:
    if not event_op_hint or not target_name:
        return False
    if target_name == f"aten.{event_op_hint}":
        return True
    if target_name.startswith(f"aten.{event_op_hint}."):
        return True
    if target_name == f"aten::{event_op_hint}":
        return True
    return target_name.startswith(f"aten::{event_op_hint}.")


def _matches_runtime_operator_name(
    runtime_operator_name: Optional[str],
    target_name: Optional[str],
) -> bool:
    canonical_runtime = _canonical_target_name(runtime_operator_name)
    canonical_target = _canonical_target_name(target_name)
    if not canonical_runtime or not canonical_target:
        return False
    if canonical_runtime == canonical_target:
        return True
    if canonical_target.startswith(canonical_runtime + "."):
        return True
    if canonical_runtime.startswith(canonical_target + "."):
        return True
    return canonical_runtime.split(".")[-1] == canonical_target.split(".")[-1]


def _debug_handle_target_names(
    handle: int,
    node_name_metadata: Dict[str, Dict[str, Any]],
) -> List[str]:
    target_names: List[str] = []
    for metadata in node_name_metadata.values():
        if metadata.get("debug_handle") != (handle,):
            continue
        target_name = metadata.get("target_name")
        if isinstance(target_name, str):
            target_names.append(target_name)
    return target_names


def _runtime_operator_conflicts_with_handle_nodes(
    handle: int,
    runtime_operator_name: Optional[str],
    node_name_metadata: Dict[str, Dict[str, Any]],
) -> bool:
    if not runtime_operator_name:
        return False
    target_names = _debug_handle_target_names(handle, node_name_metadata)
    if not target_names:
        return False
    return not any(
        _matches_runtime_operator_name(runtime_operator_name, target_name)
        for target_name in target_names
    )


def _target_name(target: Any) -> Optional[str]:
    try:
        name_attr = getattr(target, "name", None)
        if callable(name_attr):
            return name_attr()
    except Exception:
        pass
    name_attr = getattr(target, "__name__", None)
    if isinstance(name_attr, str):
        return name_attr
    return None
