from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class TensorMeta:
    sizes: Tuple[int, ...]
    dtype: int
    nbytes: int
    is_memory_planned: bool


@dataclass(frozen=True)
class MethodMetaView:
    name: str
    inputs: tuple[TensorMeta, ...]
    outputs: tuple[TensorMeta, ...]


@dataclass(frozen=True)
class RunStats:
    elapsed_ms: float
    host_input_ms: float | None = None
    module_execute_ms: float | None = None
    output_wrap_ms: float | None = None
    vk_copy_inputs_ms: float | None = None
    vk_resize_ms: float | None = None
    vk_compute_graph_execute_ms: float | None = None
    vk_copy_outputs_ms: float | None = None
    vk_total_backend_ms: float | None = None
    vk_gpu_shader_total_ms: float | None = None
    vk_gpu_shader_dispatch_count: int | None = None
    vk_generation: int | None = None
    backend: str | None = None
    delegate: str | None = None


@dataclass(frozen=True)
class RunResult:
    values: tuple[Any, ...]
    stats: RunStats | None = None


@dataclass(frozen=True)
class SessionOptions:
    enable_etdump: bool = False
    debug_buffer_size: int = 0
    program_verification: Any = "Minimal"
    pure_vulkan_required: bool = True
