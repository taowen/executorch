# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse

from executorch.devtools.agent_debug.delegate_abi import (
    inspect_from_source,
    inspect_pte_with_source,
    write_delegate_abi_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect delegate ABI alignment between lowered ExportedProgram, "
            "emitted DelegateCall instructions, and serialized Vulkan blob I/O."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Python file defining a factory that returns an ExecutorchProgramManager, "
            "ExportedProgram, or (ExportedProgram, Program)."
        ),
    )
    parser.add_argument(
        "--factory",
        default="build_target",
        help="Factory function inside --source. Defaults to `build_target`.",
    )
    parser.add_argument(
        "--method",
        default="forward",
        help="Method name to inspect. Defaults to `forward`.",
    )
    parser.add_argument(
        "--pte",
        default=None,
        help=(
            "Optional .pte path. If provided, emitted DelegateCall metadata is read "
            "from the PTE instead of the in-memory Program."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write markdown/json reports.",
    )
    args = parser.parse_args()

    if args.pte:
        report = inspect_pte_with_source(
            source_path=args.source,
            pte_path=args.pte,
            factory=args.factory,
            method_name=args.method,
        )
    else:
        report = inspect_from_source(
            source_path=args.source,
            factory=args.factory,
            method_name=args.method,
        )

    json_path, md_path = write_delegate_abi_report(report, args.output_dir)
    print(report.to_markdown())
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
