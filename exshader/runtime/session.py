from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

from .types import MethodMetaView, RunResult, RunStats, SessionOptions, TensorMeta


def _to_tensor_meta(meta: Any) -> TensorMeta:
    if isinstance(meta, dict):
        return TensorMeta(
            sizes=tuple(int(v) for v in meta.get("sizes", ())),
            dtype=int(meta.get("dtype", -1)),
            nbytes=int(meta.get("nbytes", 0)),
            is_memory_planned=bool(meta.get("is_memory_planned", False)),
        )
    return TensorMeta(
        sizes=tuple(int(v) for v in meta.sizes()),
        dtype=int(meta.dtype()),
        nbytes=int(meta.nbytes()),
        is_memory_planned=bool(meta.is_memory_planned()),
    )


def _input_summary(inputs: Sequence[Any]) -> str:
    parts: list[str] = []
    for idx, item in enumerate(inputs):
        if hasattr(item, "shape") and hasattr(item, "dtype"):
            parts.append(f"#{idx}:tensor(shape={tuple(item.shape)},dtype={item.dtype})")
        else:
            parts.append(f"#{idx}:{type(item).__name__}")
    return "[" + ", ".join(parts) + "]"


def _resolve_program_verification(program_verification: Any) -> Any:
    runtime_mod = _runtime_module()
    if not isinstance(program_verification, str):
        return program_verification
    if hasattr(runtime_mod.Verification, program_verification):
        return getattr(runtime_mod.Verification, program_verification)
    raise ValueError(
        f"Unknown program_verification='{program_verification}', "
        f"expected one of: {[k for k in dir(runtime_mod.Verification) if not k.startswith('_')]}"
    )


def _to_run_stats(raw: Any, *, fallback_elapsed_ms: float) -> RunStats:
    if not isinstance(raw, dict):
        return RunStats(elapsed_ms=fallback_elapsed_ms)
    return RunStats(
        elapsed_ms=float(raw.get("elapsed_ms", fallback_elapsed_ms)),
        host_input_ms=(
            float(raw["host_input_ms"]) if raw.get("host_input_ms") is not None else None
        ),
        module_execute_ms=(
            float(raw["module_execute_ms"]) if raw.get("module_execute_ms") is not None else None
        ),
        output_wrap_ms=(
            float(raw["output_wrap_ms"]) if raw.get("output_wrap_ms") is not None else None
        ),
        vk_copy_inputs_ms=(
            float(raw["vk_copy_inputs_ms"]) if raw.get("vk_copy_inputs_ms") is not None else None
        ),
        vk_resize_ms=(
            float(raw["vk_resize_ms"]) if raw.get("vk_resize_ms") is not None else None
        ),
        vk_compute_graph_execute_ms=(
            float(raw["vk_compute_graph_execute_ms"])
            if raw.get("vk_compute_graph_execute_ms") is not None
            else None
        ),
        vk_copy_outputs_ms=(
            float(raw["vk_copy_outputs_ms"]) if raw.get("vk_copy_outputs_ms") is not None else None
        ),
        vk_total_backend_ms=(
            float(raw["vk_total_backend_ms"]) if raw.get("vk_total_backend_ms") is not None else None
        ),
        vk_gpu_shader_total_ms=(
            float(raw["vk_gpu_shader_total_ms"])
            if raw.get("vk_gpu_shader_total_ms") is not None
            else None
        ),
        vk_gpu_shader_dispatch_count=(
            int(raw["vk_gpu_shader_dispatch_count"])
            if raw.get("vk_gpu_shader_dispatch_count") is not None
            else None
        ),
        vk_generation=(
            int(raw["vk_generation"]) if raw.get("vk_generation") is not None else None
        ),
    )


def _validate_pure_vulkan_runtime(options: SessionOptions, runtime_mod: Any) -> None:
    if not options.pure_vulkan_required:
        return
    backend_names = set(runtime_mod._get_registered_backend_names())
    if "VulkanBackend" not in backend_names:
        raise RuntimeError(
            "pure_vulkan_required=True but VulkanBackend is not registered in runtime"
        )


def _runtime_module() -> Any:
    try:
        from . import _exshader_runtime

        return _exshader_runtime
    except Exception as exc:
        raise RuntimeError(
            "Failed to import exshader runtime shim (_exshader_runtime). "
            "Build it with: `bash exshader/scripts/build_vulkan.sh`."
        ) from exc


@dataclass
class MethodHandle:
    _session: "Session"
    name: str

    def meta(self) -> MethodMetaView:
        raw = self._session._session_handle.method_meta(self.name)
        if not isinstance(raw, dict):
            raise RuntimeError(f"Unexpected method_meta type for '{self.name}': {type(raw)}")
        inputs = tuple(_to_tensor_meta(meta) for meta in raw.get("inputs", []) if meta is not None)
        outputs = tuple(_to_tensor_meta(meta) for meta in raw.get("outputs", []) if meta is not None)
        return MethodMetaView(name=self.name, inputs=inputs, outputs=outputs)

    def run(self, inputs: Sequence[Any], *, clone_outputs: bool = False) -> RunResult:
        begin = time.perf_counter()
        try:
            handle = self._session._session_handle
            if hasattr(handle, "run_with_stats"):
                result = handle.run_with_stats(self.name, list(inputs), clone_outputs)
                values = result.get("values", [])
                stats = _to_run_stats(
                    result.get("stats", {}),
                    fallback_elapsed_ms=(time.perf_counter() - begin) * 1000.0,
                )
                return RunResult(values=tuple(values), stats=stats)
            values = handle.run(self.name, list(inputs), clone_outputs)
        except Exception as exc:
            raise RuntimeError(
                f"MethodHandle.run failed: method='{self.name}', "
                f"clone_outputs={clone_outputs}, inputs={_input_summary(inputs)}, error={exc}"
            ) from exc
        elapsed_ms = (time.perf_counter() - begin) * 1000.0
        return RunResult(values=tuple(values), stats=RunStats(elapsed_ms=elapsed_ms))

    def set_inputs(self, inputs: Sequence[Any]) -> None:
        try:
            self._session._session_handle.set_inputs(self.name, list(inputs))
        except Exception as exc:
            raise RuntimeError(
                f"MethodHandle.set_inputs failed: method='{self.name}', "
                f"inputs={_input_summary(inputs)}, error={exc}"
            ) from exc

    def execute(self) -> None:
        try:
            self._session._session_handle.execute(self.name)
        except Exception as exc:
            raise RuntimeError(
                f"MethodHandle.execute failed: method='{self.name}', error={exc}"
            ) from exc

    def get_outputs(self, *, clone_outputs: bool = False) -> RunResult:
        try:
            values = self._session._session_handle.get_outputs(self.name, clone_outputs)
        except Exception as exc:
            raise RuntimeError(
                f"MethodHandle.get_outputs failed: method='{self.name}', "
                f"clone_outputs={clone_outputs}, error={exc}"
            ) from exc
        return RunResult(values=tuple(values))


@dataclass
class Session:
    _session_handle: Any | None
    model_path: Path
    _options: SessionOptions
    _method_cache: dict[str, MethodHandle] | None = None

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        options: SessionOptions | None = None,
    ) -> "Session":
        opts = options or SessionOptions()
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        runtime_mod = _runtime_module()
        _validate_pure_vulkan_runtime(opts, runtime_mod)
        verification = _resolve_program_verification(opts.program_verification)
        session_handle = runtime_mod.SessionHandle.load(
            str(path),
            None,
            verification,
        )
        return cls(
            _session_handle=session_handle,
            model_path=path,
            _options=SessionOptions(
                enable_etdump=opts.enable_etdump,
                debug_buffer_size=opts.debug_buffer_size,
                program_verification=verification,
                pure_vulkan_required=opts.pure_vulkan_required,
            ),
            _method_cache={},
        )

    def method(self, name: str) -> MethodHandle:
        if self._session_handle is None:
            raise RuntimeError("Session is closed")
        if self._method_cache is None:
            self._method_cache = {}
        cached = self._method_cache.get(name)
        if cached is not None:
            return cached
        handle = MethodHandle(_session=self, name=name)
        self._method_cache[name] = handle
        return handle

    def method_names(self) -> list[str]:
        if self._session_handle is None:
            raise RuntimeError("Session is closed")
        return [str(name) for name in self._session_handle.method_names()]

    def method_meta(self, name: str) -> MethodMetaView:
        return self.method(name).meta()

    def alloc_int64(self, sizes: Sequence[int]) -> Any:
        if self._session_handle is None:
            raise RuntimeError("Session is closed")
        return self._session_handle.alloc_int64([int(v) for v in sizes])

    def reset_state(self) -> None:
        runtime_mod = _runtime_module()
        verification = _resolve_program_verification(self._options.program_verification)
        self._session_handle = runtime_mod.SessionHandle.load(
            str(self.model_path),
            None,
            verification,
        )
        self._method_cache = {}

    def close(self) -> None:
        if self._session_handle is not None:
            self._session_handle.close()
        self._session_handle = None
        self._method_cache = None
