# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.devtools.agent_debug.delegate_abi import _analyze_delegate_contract


class DelegateABITest(unittest.TestCase):
    def test_vulkan_mutation_outputs_are_reported(self) -> None:
        findings, notes = _analyze_delegate_contract(
            backend_id="VulkanBackend",
            runtime_input_count=6,
            lowered_user_input_count=6,
            lowered_total_output_count=3,
            lowered_user_output_count=1,
            lowered_buffer_mutation_count=2,
            lowered_user_input_mutation_count=0,
            emitted_visible_output_count=1,
            emitted_backend_id="VulkanBackend",
            serialized_input_count=6,
            serialized_output_count=3,
        )
        codes = {finding.code for finding in findings}
        self.assertIn("vulkan_runtime_output_abi_mismatch", codes)
        self.assertIn("vulkan_mutation_output_leakage", codes)
        self.assertTrue(any("structural ABI failure" in note for note in notes))

    def test_clean_vulkan_delegate_has_no_findings(self) -> None:
        findings, notes = _analyze_delegate_contract(
            backend_id="VulkanBackend",
            runtime_input_count=4,
            lowered_user_input_count=4,
            lowered_total_output_count=2,
            lowered_user_output_count=2,
            lowered_buffer_mutation_count=0,
            lowered_user_input_mutation_count=0,
            emitted_visible_output_count=2,
            emitted_backend_id="VulkanBackend",
            serialized_input_count=4,
            serialized_output_count=2,
        )
        self.assertEqual(findings, [])
        self.assertEqual(notes, [])
