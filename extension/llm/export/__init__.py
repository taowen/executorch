# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "LLMEdgeManager",
]


def __getattr__(name: str):
    if name == "LLMEdgeManager":
        from .builder import LLMEdgeManager

        return LLMEdgeManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
