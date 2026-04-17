# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from executorch.devtools.agent_debug.core import (
    AgentDiagnosisReport,
    ExecutionStep,
    TensorDiffFinding,
    _CombinedOutputCapturer,
    _analyze_event,
    _summarize_unmapped_vulkan_delegate_events,
    _capture_reference_step_outputs,
    _derive_refined_focus_handles_from_execute_block,
    _trim_runtime_outputs_for_event,
    capture_runtime_target_step,
    derive_focus_handles_from_report,
    derive_scoped_delegate_focus_from_report,
)
from executorch.extension.pybindings.test.make_test import ModuleAdd, create_program
from executorch.runtime import Verification
from torch.fx import symbolic_trace


class AgentDebugCoreTest(unittest.TestCase):
    def test_trim_runtime_outputs_for_delegate_internal_event(self) -> None:
        raw_outputs = [torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0)]
        offset, trimmed = _trim_runtime_outputs_for_event(
            "2",
            raw_outputs,
            num_outputs=1,
        )
        self.assertEqual(offset, 2)
        self.assertEqual(len(trimmed), 1)
        self.assertEqual(float(trimmed[0]), 1.0)

    def test_derive_refined_focus_handles_from_execute_block(self) -> None:
        events = [
            SimpleNamespace(
                name="DELEGATE_CALL",
                debug_handles=(100,),
                debug_data=[torch.zeros(1)],
            ),
            SimpleNamespace(
                name="DELEGATE_CALL",
                debug_handles=(4, 13),
                debug_data=[torch.zeros(1)],
            ),
            SimpleNamespace(
                name="native_call_slice_scatter.out",
                debug_handles=(15,),
                debug_data=[torch.zeros(1)],
            ),
            SimpleNamespace(
                name="DELEGATE_CALL",
                debug_handles=(999,),
                debug_data=[],
            ),
            SimpleNamespace(
                name="DELEGATE_CALL",
                debug_handles=(38,),
                debug_data=[torch.zeros(1)],
            ),
        ]
        report = AgentDiagnosisReport(
            target_step_index=1,
            target_step_label="step1",
            total_events=len(events),
            matched_outputs=3,
            unmapped_outputs=0,
            first_divergent=TensorDiffFinding(
                event_index=1,
                event_name="DELEGATE_CALL",
                runtime_tensor_index=0,
                runtime_debug_handle=(4, 13),
                runtime_instruction_id=7,
                runtime_delegate_debug_identifier=101,
                runtime_delegate_backend_name="VulkanBackend",
                runtime_delegate_kernel_name="binary_mul_texture3d_float",
                runtime_delegate_operator_name="aten.mul.Tensor",
                runtime_delegate_dispatch_id=101,
                aot_debug_handle=(4,),
                aot_ops=["aten.mul.Tensor"],
                shape=(1,),
                dtype="torch.float32",
                max_abs_diff=1.0,
                mean_abs_diff=1.0,
                ambiguous_candidates=[],
            ),
            top_divergences=[
                TensorDiffFinding(
                    event_index=4,
                    event_name="DELEGATE_CALL",
                    runtime_tensor_index=0,
                    runtime_debug_handle=(38,),
                    runtime_instruction_id=9,
                    runtime_delegate_debug_identifier="shader://exp",
                    runtime_delegate_backend_name="VulkanBackend",
                    runtime_delegate_kernel_name="exp_texture3d_float",
                    runtime_delegate_operator_name="aten.exp.default",
                    runtime_delegate_dispatch_id=9,
                    aot_debug_handle=(38,),
                    aot_ops=["aten.exp.default"],
                    shape=(1,),
                    dtype="torch.float32",
                    max_abs_diff=2.0,
                    mean_abs_diff=2.0,
                    ambiguous_candidates=[],
                )
            ],
            notes=[],
        )
        handles = _derive_refined_focus_handles_from_execute_block(
            SimpleNamespace(events=events),
            report,
            event_window=1,
            top_event_count=1,
        )
        self.assertEqual(handles, [100, 4, 13, 15, 38])

    def test_capture_reference_step_outputs_focus_and_cache(self) -> None:
        class TwoOps(nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                added = x + y
                return added * y

        graph_module = symbolic_trace(TwoOps())
        debug_handle = 1
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                node.meta["debug_handle"] = debug_handle
                debug_handle += 1

        steps = [
            ExecutionStep(
                "forward",
                (torch.ones(2, 2), torch.full((2, 2), 2.0)),
                "step0",
            )
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            etrecord_path = os.path.join(temp_dir, "dummy_etrecord.bin")
            with open(etrecord_path, "wb") as handle:
                handle.write(b"dummy")

            first = _capture_reference_step_outputs(
                graph_module,
                steps,
                0,
                etrecord_path=etrecord_path,
                reference_graph="edge",
                debug_handle_focus=[2],
                cache_dir=temp_dir,
            )
            self.assertFalse(first.cache_hit)
            self.assertEqual(first.focus_handles, (2,))
            self.assertEqual(set(first.outputs.keys()), {(2,)})

            second = _capture_reference_step_outputs(
                graph_module,
                steps,
                0,
                etrecord_path=etrecord_path,
                reference_graph="edge",
                debug_handle_focus=[2],
                cache_dir=temp_dir,
            )
            self.assertTrue(second.cache_hit)
            self.assertEqual(set(second.outputs.keys()), {(2,)})

    def test_analyze_event_prefers_lowest_error_candidate(self) -> None:
        aot_outputs = {
            (1,): torch.zeros(2, 2),
            (2,): torch.full((2, 2), 5.0),
        }
        aot_op_names = {
            (1,): ["aten.add.Tensor"],
            (2,): ["aten.mul.Tensor"],
        }
        runtime_outputs = [torch.full((2, 2), 4.99)]
        findings, unmapped = _analyze_event(
            event_index=3,
            event_name="DELEGATE_CALL",
            runtime_debug_handle=(1, 2),
            runtime_outputs=runtime_outputs,
            runtime_tensor_offset=0,
            aot_outputs=aot_outputs,
            aot_outputs_by_name={},
            aot_op_names=aot_op_names,
            node_name_metadata={},
            runtime_instruction_id=17,
            runtime_delegate_debug_identifier=2,
            runtime_delegate_backend_name="VulkanBackend",
            runtime_delegate_kernel_name=None,
            runtime_delegate_operator_name=None,
            runtime_delegate_dispatch_id=None,
        )
        self.assertEqual(unmapped, 0)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].aot_debug_handle, (2,))
        self.assertIn(1, findings[0].ambiguous_candidates)
        self.assertEqual(findings[0].runtime_instruction_id, 17)
        self.assertEqual(findings[0].runtime_delegate_debug_identifier, 2)

    def test_capture_reference_step_outputs_runs_combined_capture_once(self) -> None:
        class AddOne(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        graph_module = symbolic_trace(AddOne())
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                node.meta["debug_handle"] = 1

        steps = [ExecutionStep("forward", (torch.tensor([0.0]),), "step0")]
        with tempfile.TemporaryDirectory() as temp_dir:
            etrecord_path = os.path.join(temp_dir, "dummy_etrecord.bin")
            with open(etrecord_path, "wb") as handle:
                handle.write(b"dummy")

            with patch.object(
                _CombinedOutputCapturer,
                "run_and_capture",
                return_value=(
                    {(1,): torch.tensor([1.0])},
                    {"add": torch.tensor([1.0])},
                ),
            ) as run_and_capture:
                captured = _capture_reference_step_outputs(
                    graph_module,
                    steps,
                    0,
                    etrecord_path=etrecord_path,
                    reference_graph="edge",
                    cache_dir=temp_dir,
                )

        run_and_capture.assert_called_once()
        self.assertEqual(float(captured.outputs[(1,)][0]), 1.0)
        self.assertEqual(float(captured.outputs_by_name["add"][0]), 1.0)

    def test_analyze_event_prefers_event_name_for_reused_handle(self) -> None:
        runtime_output = torch.full((2, 2), 3.0)
        findings, unmapped = _analyze_event(
            event_index=5,
            event_name="native_call_mul.Scalar_out",
            runtime_debug_handle=(7,),
            runtime_outputs=[runtime_output],
            runtime_tensor_offset=0,
            aot_outputs={(7,): torch.zeros((2, 2))},
            aot_outputs_by_name={
                "aten_mul_scalar": torch.full((2, 2), 3.0),
                "aten_view_copy_default": torch.zeros((2, 2)),
            },
            aot_op_names={(7,): ["aten_view_copy_default"]},
            node_name_metadata={
                "aten_mul_scalar": {
                    "debug_handle": (7,),
                    "op_name": "aten_mul_scalar",
                    "target_name": "aten::mul.Scalar",
                },
                "aten_view_copy_default": {
                    "debug_handle": (7,),
                    "op_name": "aten_view_copy_default",
                    "target_name": "aten::view_copy",
                },
            },
            runtime_instruction_id=21,
            runtime_delegate_debug_identifier=None,
            runtime_delegate_backend_name=None,
            runtime_delegate_kernel_name=None,
            runtime_delegate_operator_name=None,
            runtime_delegate_dispatch_id=None,
        )
        self.assertEqual(unmapped, 0)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].aot_ops, ["aten_mul_scalar"])
        self.assertEqual(findings[0].mapping_confidence, "high")
        self.assertEqual(findings[0].mapping_source, "debug_handle+operator_name")

    def test_analyze_event_skips_handle_only_fallback_when_runtime_operator_conflicts(self) -> None:
        findings, unmapped = _analyze_event(
            event_index=8,
            event_name="11",
            runtime_debug_handle=(21,),
            runtime_outputs=[torch.full((2, 2), 7.0)],
            runtime_tensor_offset=0,
            aot_outputs={(21,): torch.full((2, 2), 7.0)},
            aot_outputs_by_name={
                "aten_view_copy_default_7": torch.full((2, 2), 7.0),
                "aten_mm_default_3": torch.full((2, 2), 7.0),
            },
            aot_op_names={
                (21,): ["aten_view_copy_default_7", "aten_mm_default_3"],
            },
            node_name_metadata={
                "aten_view_copy_default_7": {
                    "debug_handle": (21,),
                    "op_name": "aten_view_copy_default_7",
                    "target_name": "aten::view_copy",
                },
                "aten_mm_default_3": {
                    "debug_handle": (21,),
                    "op_name": "aten_mm_default_3",
                    "target_name": "aten::mm.default",
                },
            },
            runtime_instruction_id=14,
            runtime_delegate_debug_identifier=11,
            runtime_delegate_backend_name="VulkanBackend",
            runtime_delegate_kernel_name="binary_mul_texture3d_float",
            runtime_delegate_operator_name="aten.mul.Tensor",
            runtime_delegate_dispatch_id=11,
        )
        self.assertEqual(findings, [])
        self.assertEqual(unmapped, 1)

    def test_analyze_event_for_scoped_delegate_keeps_best_runtime_tensor(self) -> None:
        findings, unmapped = _analyze_event(
            event_index=27,
            event_name="2",
            runtime_debug_handle=(14,),
            runtime_outputs=[
                torch.tensor(1.0),
                torch.tensor(0.0),
                torch.tensor(1.0),
            ],
            runtime_tensor_offset=0,
            aot_outputs={(14,): torch.tensor(1.0)},
            aot_outputs_by_name={"aten_sub_tensor": torch.tensor(1.0)},
            aot_op_names={(14,): ["aten_sub_tensor"]},
            node_name_metadata={
                "aten_sub_tensor": {
                    "debug_handle": (14,),
                    "op_name": "aten_sub_tensor",
                    "target_name": "aten::sub.Tensor",
                }
            },
            runtime_instruction_id=9,
            runtime_delegate_debug_identifier=2,
            runtime_delegate_backend_name="VulkanBackend",
            runtime_delegate_kernel_name=None,
            runtime_delegate_operator_name="aten.sub.Tensor",
            runtime_delegate_dispatch_id=None,
        )
        self.assertEqual(unmapped, 0)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].runtime_tensor_index, 0)
        self.assertEqual(findings[0].max_abs_diff, 0.0)
        self.assertEqual(findings[0].max_abs_diff, 0.0)
        self.assertEqual(findings[0].mapping_confidence, "high")

    def test_report_markdown_mentions_first_divergence(self) -> None:
        finding = TensorDiffFinding(
            event_index=7,
            event_name="native_call_add.out",
            runtime_tensor_index=0,
            runtime_debug_handle=(5,),
            runtime_instruction_id=None,
            runtime_delegate_debug_identifier=None,
            runtime_delegate_backend_name=None,
            runtime_delegate_kernel_name=None,
            runtime_delegate_operator_name=None,
            runtime_delegate_dispatch_id=None,
            aot_debug_handle=(5,),
            aot_ops=["aten.add.Tensor"],
            shape=(2, 2),
            dtype="torch.float32",
            max_abs_diff=3.0,
            mean_abs_diff=1.0,
            ambiguous_candidates=[],
        )
        report = AgentDiagnosisReport(
            target_step_index=1,
            target_step_label="forward[1]",
            total_events=12,
            matched_outputs=5,
            unmapped_outputs=1,
            first_divergent=finding,
            top_divergences=[finding],
            notes=["note"],
        )
        markdown = report.to_markdown()
        self.assertIn("first divergence", markdown)
        self.assertIn("aten.add.Tensor", markdown)
        self.assertIn("mapping", markdown)

    def test_capture_runtime_target_step_writes_artifacts(self) -> None:
        ep, inputs = create_program(ModuleAdd())
        steps = [
            ExecutionStep("forward", tuple(inputs), "step0"),
            ExecutionStep("forward", tuple(inputs), "step1"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            pte_path = os.path.join(temp_dir, "test_program.pte")
            with open(pte_path, "wb") as handle:
                handle.write(ep.buffer)
            artifacts = capture_runtime_target_step(
                pte_path=pte_path,
                steps=steps,
                output_dir=temp_dir,
                target_step_index=1,
                verification=Verification.Minimal,
                debug_buffer_size=int(1e7),
                file_stem="runtime_capture",
            )
            self.assertTrue(os.path.exists(artifacts.etdump_path))
            self.assertTrue(os.path.exists(artifacts.debug_buffer_path))
            self.assertEqual(artifacts.target_step_label, "step1")

    def test_derive_focus_handles_from_report_uses_runtime_handle_tuple(self) -> None:
        report = {
            "first_divergent": {
                "runtime_debug_handle": [4, 13, 21],
            },
            "top_divergences": [
                {"runtime_debug_handle": [13, 15]},
                {"runtime_debug_handle": [99]},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "agent_report.json")
            with open(report_path, "w", encoding="utf-8") as handle:
                import json

                json.dump(report, handle)
            handles = derive_focus_handles_from_report(report_path, event_count=2)
            self.assertEqual(handles, [4, 13, 21, 15])

    def test_derive_scoped_delegate_focus_from_report(self) -> None:
        report = {
            "first_divergent": {
                "runtime_instruction_id": 7,
                "runtime_delegate_debug_identifier": 3,
            },
            "top_divergences": [
                {
                    "runtime_instruction_id": 7,
                    "runtime_delegate_debug_identifier": "shader://rms_norm",
                },
                {
                    "runtime_instruction_id": 11,
                    "runtime_delegate_debug_identifier": 9,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "agent_report.json")
            with open(report_path, "w", encoding="utf-8") as handle:
                import json

                json.dump(report, handle)
            scoped_specs = derive_scoped_delegate_focus_from_report(
                report_path, event_count=3
            )
            self.assertEqual(
                scoped_specs,
                [
                    {
                        "instruction_id": 7,
                        "debug_handles": [3],
                        "debug_names": ["shader://rms_norm"],
                    },
                    {
                        "instruction_id": 11,
                        "debug_handles": [9],
                    },
                ],
            )

    def test_summarize_unmapped_vulkan_delegate_events_mentions_coverage_gap(self) -> None:
        notes = _summarize_unmapped_vulkan_delegate_events(
            {
                14: {
                    35: {
                        "kernel_name": "rsqrt_float_texture3d",
                        "operator_name": "aten.rsqrt.default",
                        "event_name": "35",
                    },
                    36: {
                        "kernel_name": "binary_mul_texture3d_float",
                        "operator_name": "aten.mul.Tensor",
                        "event_name": "36",
                    },
                }
            },
            vulkan_delegate_coverage={14: 34},
        )
        self.assertEqual(len(notes), 1)
        self.assertIn("instruction 14", notes[0])
        self.assertIn("local id 34", notes[0])
        self.assertIn("rsqrt_float_texture3d", notes[0])
