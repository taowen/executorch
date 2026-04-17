# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple

from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    Bool,
    BoolList,
    Double,
    DoubleList,
    Int,
    IntList,
    Null,
    String,
    SymInt,
    ValueList,
    VkGraph,
    VkTensor,
)
from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    extract_vk_flatbuffer,
    flatbuffer_to_vk_graph,
)
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.lowered_backend_module import get_lowered_submodules
from executorch.exir.schema import DelegateCall, Program
from torch.export.exported_program import ExportedProgram, OutputKind


@dataclass(frozen=True)
class DelegateABIFinding:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class DelegateOutputSpecSummary:
    kind: str
    name: str
    target: Optional[str]


@dataclass(frozen=True)
class SerializedValueSummary:
    value_id: int
    summary: str


@dataclass(frozen=True)
class DelegateABISubgraphReport:
    delegate_order: int
    lowered_module_name: str
    backend_id: str
    runtime_input_count: int
    lowered_user_input_count: int
    lowered_total_output_count: int
    lowered_user_output_count: int
    lowered_buffer_mutation_count: int
    lowered_user_input_mutation_count: int
    output_specs: List[DelegateOutputSpecSummary]
    emitted_delegate_arg_count: Optional[int]
    emitted_visible_output_count: Optional[int]
    emitted_delegate_index: Optional[int]
    emitted_backend_id: Optional[str]
    serialized_input_count: Optional[int]
    serialized_output_count: Optional[int]
    serialized_inputs: List[SerializedValueSummary]
    serialized_outputs: List[SerializedValueSummary]
    findings: List[DelegateABIFinding]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delegate_order": self.delegate_order,
            "lowered_module_name": self.lowered_module_name,
            "backend_id": self.backend_id,
            "runtime_input_count": self.runtime_input_count,
            "lowered_user_input_count": self.lowered_user_input_count,
            "lowered_total_output_count": self.lowered_total_output_count,
            "lowered_user_output_count": self.lowered_user_output_count,
            "lowered_buffer_mutation_count": self.lowered_buffer_mutation_count,
            "lowered_user_input_mutation_count": self.lowered_user_input_mutation_count,
            "output_specs": [asdict(spec) for spec in self.output_specs],
            "emitted_delegate_arg_count": self.emitted_delegate_arg_count,
            "emitted_visible_output_count": self.emitted_visible_output_count,
            "emitted_delegate_index": self.emitted_delegate_index,
            "emitted_backend_id": self.emitted_backend_id,
            "serialized_input_count": self.serialized_input_count,
            "serialized_output_count": self.serialized_output_count,
            "serialized_inputs": [asdict(item) for item in self.serialized_inputs],
            "serialized_outputs": [asdict(item) for item in self.serialized_outputs],
            "findings": [asdict(item) for item in self.findings],
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DelegateABIReport:
    method_name: str
    inspected_delegate_count: int
    error_count: int
    warning_count: int
    delegates: List[DelegateABISubgraphReport]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "inspected_delegate_count": self.inspected_delegate_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "delegates": [delegate.to_dict() for delegate in self.delegates],
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Delegate ABI Report",
            "",
            f"- method: `{self.method_name}`",
            f"- inspected delegates: `{self.inspected_delegate_count}`",
            f"- errors: `{self.error_count}`",
            f"- warnings: `{self.warning_count}`",
            "",
        ]
        for delegate in self.delegates:
            lines.extend(
                [
                    (
                        f"## delegate `{delegate.delegate_order}` "
                        f"`{delegate.lowered_module_name}` (`{delegate.backend_id}`)"
                    ),
                    "",
                    f"- runtime inputs from `executorch_call_delegate`: `{delegate.runtime_input_count}`",
                    f"- lowered user inputs: `{delegate.lowered_user_input_count}`",
                    (
                        "- lowered outputs: "
                        f"total `{delegate.lowered_total_output_count}`, "
                        f"user `{delegate.lowered_user_output_count}`, "
                        f"buffer_mutation `{delegate.lowered_buffer_mutation_count}`, "
                        f"user_input_mutation `{delegate.lowered_user_input_mutation_count}`"
                    ),
                ]
            )
            if delegate.emitted_delegate_arg_count is not None:
                lines.append(
                    (
                        "- emitted `DelegateCall`: "
                        f"delegate_index `{delegate.emitted_delegate_index}`, "
                        f"backend `{delegate.emitted_backend_id}`, "
                        f"args `{delegate.emitted_delegate_arg_count}`, "
                        f"visible outputs `{delegate.emitted_visible_output_count}`"
                    )
                )
            if delegate.serialized_input_count is not None:
                lines.append(
                    (
                        "- serialized Vulkan blob: "
                        f"inputs `{delegate.serialized_input_count}`, "
                        f"outputs `{delegate.serialized_output_count}`"
                    )
                )
            if delegate.output_specs:
                lines.append("- lowered output specs:")
                for spec in delegate.output_specs:
                    target = f", target `{spec.target}`" if spec.target else ""
                    lines.append(f"  - `{spec.kind}` `{spec.name}`{target}")
            if delegate.serialized_outputs:
                lines.append("- serialized outputs:")
                for item in delegate.serialized_outputs:
                    lines.append(f"  - `id={item.value_id}` {item.summary}")
            if delegate.findings:
                lines.append("- findings:")
                for finding in delegate.findings:
                    lines.append(
                        f"  - `{finding.severity}` `{finding.code}`: {finding.message}"
                    )
            if delegate.notes:
                lines.append("- notes:")
                for note in delegate.notes:
                    lines.append(f"  - {note}")
            lines.append("")
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")
        return "\n".join(lines)


def inspect_executorch_manager(
    manager: Any,
    method_name: str = "forward",
) -> DelegateABIReport:
    return inspect_lowered_exported_program(
        exported_program=manager.exported_program(method_name),
        program=manager.executorch_program,
        method_name=method_name,
    )


def inspect_lowered_exported_program(
    exported_program: ExportedProgram,
    program: Optional[Program] = None,
    method_name: str = "forward",
) -> DelegateABIReport:
    lowered_submodules = get_lowered_submodules(exported_program.graph_module)
    emitted_calls = _collect_delegate_calls(program, method_name) if program else []

    findings_error_count = 0
    findings_warning_count = 0
    delegate_reports: List[DelegateABISubgraphReport] = []
    report_notes: List[str] = []

    if program is None:
        report_notes.append(
            "No ExecuTorch Program supplied; emitted DelegateCall inspection is skipped."
        )
    elif len(emitted_calls) != len(lowered_submodules):
        report_notes.append(
            "Lowered submodule count and emitted DelegateCall count differ: "
            f"{len(lowered_submodules)} vs {len(emitted_calls)}. Reports are paired by encounter order."
        )

    for index, (name, lowered_module, call_node) in enumerate(lowered_submodules):
        emitted_call = emitted_calls[index] if index < len(emitted_calls) else None
        vk_graph = _decode_vulkan_graph(lowered_module.backend_id, lowered_module.processed_bytes)
        delegate_report = _build_delegate_subgraph_report(
            delegate_order=index,
            lowered_module_name=name,
            lowered_module=lowered_module,
            call_node=call_node,
            emitted_call=emitted_call,
            vk_graph=vk_graph,
        )
        findings_error_count += sum(
            1 for finding in delegate_report.findings if finding.severity == "error"
        )
        findings_warning_count += sum(
            1 for finding in delegate_report.findings if finding.severity == "warning"
        )
        delegate_reports.append(delegate_report)

    return DelegateABIReport(
        method_name=method_name,
        inspected_delegate_count=len(delegate_reports),
        error_count=findings_error_count,
        warning_count=findings_warning_count,
        delegates=delegate_reports,
        notes=report_notes,
    )


def inspect_from_source(
    source_path: str,
    *,
    factory: str = "build_target",
    method_name: str = "forward",
) -> DelegateABIReport:
    module = _load_python_module(Path(source_path))
    if not hasattr(module, factory):
        raise AttributeError(f"Module {source_path} does not define `{factory}()`.")
    target = getattr(module, factory)()
    normalized = _normalize_inspection_target(target, method_name)
    return inspect_lowered_exported_program(
        exported_program=normalized["exported_program"],
        program=normalized.get("program"),
        method_name=normalized["method_name"],
    )


def write_delegate_abi_report(
    report: DelegateABIReport,
    output_dir: str,
    file_stem: str = "delegate_abi_report",
) -> Tuple[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return str(json_path), str(md_path)


def inspect_pte_with_source(
    source_path: str,
    *,
    pte_path: str,
    factory: str = "build_target",
    method_name: str = "forward",
) -> DelegateABIReport:
    module = _load_python_module(Path(source_path))
    if not hasattr(module, factory):
        raise AttributeError(f"Module {source_path} does not define `{factory}()`.")
    target = getattr(module, factory)()
    normalized = _normalize_inspection_target(target, method_name)
    pte = deserialize_pte_binary(Path(pte_path).read_bytes())
    return inspect_lowered_exported_program(
        exported_program=normalized["exported_program"],
        program=pte.program,
        method_name=normalized["method_name"],
    )


def _build_delegate_subgraph_report(
    *,
    delegate_order: int,
    lowered_module_name: str,
    lowered_module: Any,
    call_node: Any,
    emitted_call: Optional[Dict[str, Any]],
    vk_graph: Optional[VkGraph],
) -> DelegateABISubgraphReport:
    signature = lowered_module.original_module.graph_signature
    runtime_input_count = max(0, len(call_node.args) - 1)
    lowered_user_input_count = len(signature.user_inputs)

    output_specs = [
        DelegateOutputSpecSummary(
            kind=spec.kind.name,
            name=spec.arg.name,
            target=spec.target,
        )
        for spec in signature.output_specs
    ]

    lowered_total_output_count = len(signature.output_specs)
    lowered_user_output_count = sum(
        1 for spec in signature.output_specs if spec.kind == OutputKind.USER_OUTPUT
    )
    lowered_buffer_mutation_count = sum(
        1 for spec in signature.output_specs if spec.kind == OutputKind.BUFFER_MUTATION
    )
    lowered_user_input_mutation_count = sum(
        1
        for spec in signature.output_specs
        if spec.kind == OutputKind.USER_INPUT_MUTATION
    )

    emitted_delegate_arg_count: Optional[int] = None
    emitted_visible_output_count: Optional[int] = None
    emitted_delegate_index: Optional[int] = None
    emitted_backend_id: Optional[str] = None
    if emitted_call is not None:
        emitted_delegate_arg_count = emitted_call["arg_count"]
        emitted_delegate_index = emitted_call["delegate_index"]
        emitted_backend_id = emitted_call["backend_id"]
        emitted_visible_output_count = emitted_delegate_arg_count - runtime_input_count

    serialized_input_count: Optional[int] = None
    serialized_output_count: Optional[int] = None
    serialized_inputs: List[SerializedValueSummary] = []
    serialized_outputs: List[SerializedValueSummary] = []
    if vk_graph is not None:
        serialized_input_count = len(vk_graph.input_ids)
        serialized_output_count = len(vk_graph.output_ids)
        serialized_inputs = [
            SerializedValueSummary(value_id=value_id, summary=_summarize_vk_value(vk_graph, value_id))
            for value_id in vk_graph.input_ids
        ]
        serialized_outputs = [
            SerializedValueSummary(value_id=value_id, summary=_summarize_vk_value(vk_graph, value_id))
            for value_id in vk_graph.output_ids
        ]

    findings, notes = _analyze_delegate_contract(
        backend_id=lowered_module.backend_id,
        runtime_input_count=runtime_input_count,
        lowered_user_input_count=lowered_user_input_count,
        lowered_total_output_count=lowered_total_output_count,
        lowered_user_output_count=lowered_user_output_count,
        lowered_buffer_mutation_count=lowered_buffer_mutation_count,
        lowered_user_input_mutation_count=lowered_user_input_mutation_count,
        emitted_visible_output_count=emitted_visible_output_count,
        emitted_backend_id=emitted_backend_id,
        serialized_input_count=serialized_input_count,
        serialized_output_count=serialized_output_count,
    )

    return DelegateABISubgraphReport(
        delegate_order=delegate_order,
        lowered_module_name=lowered_module_name,
        backend_id=lowered_module.backend_id,
        runtime_input_count=runtime_input_count,
        lowered_user_input_count=lowered_user_input_count,
        lowered_total_output_count=lowered_total_output_count,
        lowered_user_output_count=lowered_user_output_count,
        lowered_buffer_mutation_count=lowered_buffer_mutation_count,
        lowered_user_input_mutation_count=lowered_user_input_mutation_count,
        output_specs=output_specs,
        emitted_delegate_arg_count=emitted_delegate_arg_count,
        emitted_visible_output_count=emitted_visible_output_count,
        emitted_delegate_index=emitted_delegate_index,
        emitted_backend_id=emitted_backend_id,
        serialized_input_count=serialized_input_count,
        serialized_output_count=serialized_output_count,
        serialized_inputs=serialized_inputs,
        serialized_outputs=serialized_outputs,
        findings=findings,
        notes=notes,
    )


def _analyze_delegate_contract(
    *,
    backend_id: str,
    runtime_input_count: int,
    lowered_user_input_count: int,
    lowered_total_output_count: int,
    lowered_user_output_count: int,
    lowered_buffer_mutation_count: int,
    lowered_user_input_mutation_count: int,
    emitted_visible_output_count: Optional[int],
    emitted_backend_id: Optional[str],
    serialized_input_count: Optional[int],
    serialized_output_count: Optional[int],
) -> Tuple[List[DelegateABIFinding], List[str]]:
    findings: List[DelegateABIFinding] = []
    notes: List[str] = []

    if runtime_input_count != lowered_user_input_count:
        findings.append(
            DelegateABIFinding(
                severity="error",
                code="delegate_input_count_mismatch",
                message=(
                    "Top-level `executorch_call_delegate` passes "
                    f"{runtime_input_count} runtime inputs but the lowered submodule "
                    f"expects {lowered_user_input_count} user inputs."
                ),
            )
        )

    if emitted_visible_output_count is not None:
        if emitted_visible_output_count < 0:
            findings.append(
                DelegateABIFinding(
                    severity="error",
                    code="delegate_emitted_arg_underflow",
                    message=(
                        "Emitted DelegateCall arg count is smaller than runtime input "
                        "count, so outputs cannot be addressed correctly."
                    ),
                )
            )
        elif emitted_visible_output_count != lowered_user_output_count:
            findings.append(
                DelegateABIFinding(
                    severity="warning",
                    code="delegate_visible_output_count_mismatch",
                    message=(
                        "Emitted DelegateCall exposes "
                        f"{emitted_visible_output_count} visible outputs while the "
                        f"lowered submodule marks {lowered_user_output_count} outputs "
                        "as USER_OUTPUT."
                    ),
                )
            )

    if emitted_backend_id is not None and emitted_backend_id != backend_id:
        findings.append(
            DelegateABIFinding(
                severity="warning",
                code="delegate_backend_ordering_mismatch",
                message=(
                    f"Lowered backend is `{backend_id}` but the emitted DelegateCall "
                    f"references backend `{emitted_backend_id}`. Pairing by encounter "
                    "order may be wrong."
                ),
            )
        )

    if serialized_input_count is not None and serialized_input_count != lowered_user_input_count:
        findings.append(
            DelegateABIFinding(
                severity="warning",
                code="delegate_serialized_input_count_mismatch",
                message=(
                    f"Serialized delegate blob exposes {serialized_input_count} inputs "
                    f"but the lowered submodule expects {lowered_user_input_count} "
                    "user inputs."
                ),
            )
        )

    is_vulkan = "vulkan" in backend_id.lower()
    mutation_output_count = (
        lowered_buffer_mutation_count + lowered_user_input_mutation_count
    )
    if is_vulkan and serialized_output_count is not None:
        if serialized_output_count != lowered_total_output_count:
            findings.append(
                DelegateABIFinding(
                    severity="warning",
                    code="vulkan_serialized_output_count_mismatch",
                    message=(
                        f"Serialized Vulkan blob contains {serialized_output_count} "
                        f"outputs, but the lowered submodule graph signature contains "
                        f"{lowered_total_output_count} outputs."
                    ),
                )
            )
        if emitted_visible_output_count is not None and serialized_output_count != emitted_visible_output_count:
            severity = "error" if mutation_output_count > 0 else "warning"
            findings.append(
                DelegateABIFinding(
                    severity=severity,
                    code="vulkan_runtime_output_abi_mismatch",
                    message=(
                        "Serialized Vulkan blob exposes "
                        f"{serialized_output_count} outputs while the emitted "
                        f"DelegateCall provides only {emitted_visible_output_count} "
                        "visible runtime outputs."
                    ),
                )
            )
        if mutation_output_count > 0 and serialized_output_count != lowered_user_output_count:
            findings.append(
                DelegateABIFinding(
                    severity="error",
                    code="vulkan_mutation_output_leakage",
                    message=(
                        "Lowered submodule contains mutation outputs "
                        f"({mutation_output_count} total), USER_OUTPUT count is "
                        f"{lowered_user_output_count}, but serialized Vulkan outputs "
                        f"count is {serialized_output_count}. This means mutation "
                        "results are still present in the Vulkan blob output list and "
                        "can misalign `VulkanBackend::execute()` output copy-out."
                    ),
                )
            )
            notes.append(
                "This is a structural ABI failure. It can be detected without running "
                "a long ETDump/Inspector numeric replay."
            )

    return findings, notes


def _collect_delegate_calls(program: Program, method_name: str) -> List[Dict[str, Any]]:
    plan = _find_execution_plan(program, method_name)
    delegate_calls: List[Dict[str, Any]] = []
    for chain_index, chain in enumerate(plan.chains):
        for instruction_index, instruction in enumerate(chain.instructions):
            instr_args = instruction.instr_args
            if not isinstance(instr_args, DelegateCall):
                continue
            delegate = plan.delegates[instr_args.delegate_index]
            delegate_calls.append(
                {
                    "chain_index": chain_index,
                    "instruction_index": instruction_index,
                    "delegate_index": instr_args.delegate_index,
                    "backend_id": delegate.id,
                    "arg_count": len(instr_args.args),
                }
            )
    return delegate_calls


def _find_execution_plan(program: Program, method_name: str):
    for plan in program.execution_plan:
        if plan.name == method_name:
            return plan
    if len(program.execution_plan) == 1:
        return program.execution_plan[0]
    available = ", ".join(plan.name for plan in program.execution_plan)
    raise KeyError(
        f"Method `{method_name}` not found in Program.execution_plan. Available: {available}"
    )


def _decode_vulkan_graph(backend_id: str, processed_bytes: bytes) -> Optional[VkGraph]:
    if "vulkan" not in backend_id.lower():
        return None
    return flatbuffer_to_vk_graph(extract_vk_flatbuffer(processed_bytes))


def _summarize_vk_value(vk_graph: VkGraph, value_id: int) -> str:
    value = vk_graph.values[value_id].value
    if isinstance(value, VkTensor):
        return (
            "tensor "
            f"shape={value.dims} "
            f"dtype={value.datatype.name} "
            f"storage={value.storage_type.name} "
            f"layout={value.memory_layout.name}"
        )
    if isinstance(value, SymInt):
        return f"symint value={value.value}"
    if isinstance(value, Int):
        return f"int value={value.int_val}"
    if isinstance(value, Double):
        return f"double value={value.double_val}"
    if isinstance(value, Bool):
        return f"bool value={value.bool_val}"
    if isinstance(value, IntList):
        return f"int_list len={len(value.items)}"
    if isinstance(value, DoubleList):
        return f"double_list len={len(value.items)}"
    if isinstance(value, BoolList):
        return f"bool_list len={len(value.items)}"
    if isinstance(value, ValueList):
        return f"value_list len={len(value.items)}"
    if isinstance(value, String):
        return f"string value={value.string_val!r}"
    if isinstance(value, Null):
        return "null"
    return type(value).__name__


def _load_python_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_inspection_target(
    target: Any,
    method_name: str,
) -> Dict[str, Any]:
    if isinstance(target, ExportedProgram):
        return {"exported_program": target, "method_name": method_name}
    if isinstance(target, tuple) and len(target) == 2:
        exported_program, program = target
        if isinstance(exported_program, ExportedProgram):
            return {
                "exported_program": exported_program,
                "program": program,
                "method_name": method_name,
            }
    if isinstance(target, dict):
        exported_program = target.get("exported_program")
        if isinstance(exported_program, ExportedProgram):
            return {
                "exported_program": exported_program,
                "program": target.get("program"),
                "method_name": target.get("method_name", method_name),
            }
    if hasattr(target, "exported_program") and hasattr(target, "executorch_program"):
        return {
            "exported_program": target.exported_program(method_name),
            "program": target.executorch_program,
            "method_name": method_name,
        }
    raise TypeError(
        "Unsupported inspection target. Expected ExportedProgram, "
        "(ExportedProgram, Program), mapping with exported_program/program, "
        "or ExecutorchProgramManager-like object."
    )
