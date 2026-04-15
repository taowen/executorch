# API Plan: exshader Runtime Interface (Pure Vulkan, Python-first)

## 1. 目标
定义一套稳定的 C++/Python 边界，使系统满足：
1. Pure Vulkan 主链路（不接受 CPU fallback 混跑）。
2. 高频迭代在 Python（融合边界、shader、loop、编排）。
3. C++ 只做低频稳定基座（method 执行原语），不做任务语义 API。
4. 降低 C++ 重编译频率，提高 agent 研发效率。

## 2. 非目标
1. 不在本阶段新增 ASR/TTS 专用 API。
2. 不在 C++ 层实现任务级 loop（如 decode 状态机、业务采样策略）。
3. 不为了兼容历史接口保留双路径主入口。

## 3. 设计原则
1. 单一主路径：`exshader.runtime.Session + MethodHandle`。
2. 显式数据流：输入/输出、内存复用、拷贝行为可见可控。
3. 错误可诊断：错误码 + 上下文，避免“黑盒失败”。
4. 默认高性能：默认 `clone_outputs=False`，鼓励 buffer 复用。
5. 任务逻辑上移：LLM 等任务语义留在 Python recipe。
6. 不污染全局：`extension/pybindings/pybindings.cpp` 保持通用，不承载 exshader 专用语义。

## 4. 分层与职责

### L0: 基座接口（主路径）
位置：`exshader/runtime/*`

职责：
1. 加载 program，管理 session 生命周期。
2. 枚举 method，读取 method meta。
3. 执行 method（run/execute），返回结构化结果。
4. 暴露最小可观测性（计时、delegate/backend 信息）。

补充说明（边界）：
1. L0 的长期目标是两层：
   - C++ 窄 shim：`_exshader_runtime`（独立 pybind 模块，面向 exshader）。
   - Python 封装：`exshader/runtime/session.py`（对外稳定 API）。
2. 当前状态：已落地 Python 封装；独立 C++ shim 尚未落地，仍暂时复用 `portable_lib`。
3. 迁移约束：在独立 C++ shim 可用前，不再继续扩展全局 `portable_lib` 绑定行为。

### L1: 任务编排层（可变）
位置：`exshader/recipes/*`

职责：
1. 用 L0 实现具体任务 loop（当前仅 LLM）。
2. 自主迭代 chunk、采样、cache 策略。
3. 不引入 C++ 任务分叉。

### L2: 历史/过渡接口（非主路径）
示例：`TextLLMRunner.prefill_tokens/decode_next_token`

策略：
1. 可用于排障和短期实验。
2. 不作为长期架构承诺。
3. 不作为功能推进的前置依赖。

## 5. Python API 详细定义（计划）

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

@dataclass(frozen=True)
class SessionOptions:
    enable_etdump: bool = False
    debug_buffer_size: int = 0
    program_verification: str = "Minimal"  # 映射到底层枚举
    pure_vulkan_required: bool = True

@dataclass(frozen=True)
class TensorMeta:
    sizes: tuple[int, ...]
    dtype: int
    nbytes: int
    is_memory_planned: bool

@dataclass(frozen=True)
class MethodMeta:
    name: str
    inputs: tuple[TensorMeta, ...]
    outputs: tuple[TensorMeta, ...]

@dataclass(frozen=True)
class RunStats:
    elapsed_ms: float
    backend: str | None
    delegate: str | None

@dataclass(frozen=True)
class RunResult:
    values: tuple[Any, ...]
    stats: RunStats | None = None

class MethodHandle:
    def meta(self) -> MethodMeta: ...
    def run(self, inputs: Sequence[Any], *, clone_outputs: bool = False) -> RunResult: ...
    def set_inputs(self, inputs: Sequence[Any]) -> None: ...
    def execute(self) -> None: ...
    def get_outputs(self, *, clone_outputs: bool = False) -> RunResult: ...

class Session:
    @classmethod
    def load(cls, model_path: str | Path, options: SessionOptions | None = None) -> "Session": ...
    def method(self, name: str) -> MethodHandle: ...
    def method_names(self) -> list[str]: ...
    def method_meta(self, name: str) -> MethodMeta: ...
    def reset_state(self) -> None: ...
    def close(self) -> None: ...
```

### 5.1 C++ 窄 shim API（新增设计，尚未实现）

目标：不依赖 torch 输入对象，提供最小 method 原语，供 `exshader/runtime` 调用。

建议模块名：
1. `_exshader_runtime`

建议接口（Python 可见）：
```python
class SessionHandle:
    @staticmethod
    def load(model_path: str, *, enable_etdump: bool = False, debug_buffer_size: int = 0,
             program_verification: str = "Minimal") -> "SessionHandle": ...
    def close(self) -> None: ...
    def method_names(self) -> list[str]: ...
    def method_meta(self, name: str) -> MethodMeta: ...
    def run(self, name: str, inputs: list[object], *, clone_outputs: bool = False) -> list[object]: ...
```

`inputs` 最小支持类型（首版）：
1. `None` / `bool` / `int`
2. `list[int]`（1D token ids）
3. `list[list[int]]`（2D token ids, batch-first）
4. 保留 `torch.Tensor` 兼容输入仅用于过渡，但不作为主路径依赖

约束：
1. 不暴露任务语义接口（不加 `prefill_tokens` / `decode_next_token` 这类 API）。
2. 不在 shim 内实现采样、解码 loop、业务状态机。
3. 输出类型保持与底层一致，必要时由 Python 封装层做统一转换。

## 6. 数据与内存模型
1. `run(inputs, clone_outputs=False)` 为默认路径。
2. recipe 热路径必须优先使用复用对象（预分配/in-place）：
   - 重复使用输入容器与标量/tensor buffer。
   - 禁止每 token 新建大对象。
3. 输出复用策略：
   - 读后即用：`clone_outputs=False`。
   - 需跨步持有：调用方自行 clone/复制。
4. 后续扩展（计划）：
   - `TensorBuffer` 抽象，用于显式管理可复用输入输出块。

## 7. 错误模型
1. 统一异常类型：`RuntimeError` + 结构化错误信息。
2. 错误信息至少包含：
   - method 名称
   - 底层 error code（十六进制）
   - 输入摘要（shape/dtype 或参数概览）
3. 不允许“模糊错误”：
   - 例如 `Prefill failed` 这类无上下文信息应逐步淘汰。

## 8. Pure Vulkan 约束落地
1. 静态检查：
   - 每次导出后输出 `KernelCall/DelegateCall` 与 delegate 列表。
2. 运行时检查：
   - 日志中出现 fallback 关键字即失败。
3. Session 选项：
   - `pure_vulkan_required=True` 时，检测到非 Vulkan 路径直接失败。

## 9. recipe 设计规范（当前只做 LLM）
1. `exshader/recipes/llm_decode.py` 仅依赖 L0。
2. 采样策略属于 recipe：
   - greedy / temperature / top-p 等在 Python 层实现。
3. kv-cache 管理策略属于 recipe：
   - 多轮对话、截断、位置推进等不进 C++ 任务 API。

## 10. 去 torch 运行时依赖计划

### 已完成
1. `_llm_runner` 支持 core-only 构建：
   - `EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER_TORCH_IO=OFF`
2. `runner/__init__.py` 移除 `import torch` 顶层硬依赖。
3. torch-io helper（`make_image_input/make_audio_input/...`）改可选符号。

### 下一步
1. 继续清理 serving 主路径对 torch 的隐式依赖。
2. 验收标准：
   - 禁 torch 环境下可导入主模块。
   - exshader 主脚本可运行（在当前模型前提下）。
3. 具体实现路径：
   - 先落地 `_exshader_runtime` 的无 torch 输入执行路径；
   - `exshader/recipes/llm_decode.py` 改为仅用 Python list/int 与该 shim 交互；
   - 再评估是否保留 torch 作为可选加速依赖（非必须）。

## 11. 迁移步骤（执行顺序）
1. 固化 L0 API（`Session/MethodHandle`）并冻结签名。
2. 将 `llm_decode` 完全收敛到 L0，移除对 L2 的依赖。
3. 将脚本入口全部切换到 recipe（已完成，继续清理细节）。
4. 收敛错误信息与观测输出，建立回归基线（性能 + 纯 Vulkan）。
5. 新增 `_exshader_runtime`（独立 C++ shim），由 `exshader/runtime/session.py` 优先调用。
6. 在同一模型上做 A/B：
   - A: `portable_lib` 路径
   - B: `_exshader_runtime` 路径
   比较功能一致性、decode 速度、纯 Vulkan 指标。
7. A/B 通过后，`exshader` 默认切换到 `_exshader_runtime`，`portable_lib` 仅保留兜底调试用途。

## 12. 验收标准
1. 功能：
   - Qwen3 在 recipe 主路径可运行。
2. 架构：
   - recipe 不依赖任务专用 C++ API。
3. 性能：
   - decode 热路径无明显重复对象构建。
4. 纯 Vulkan：
   - runtime fallback=0，静态 delegate 检查通过。
5. 依赖：
   - serving 主路径不强制依赖完整 torch。

## 13. 风险与应对
1. 某些模型在 L2 接口可跑、L0 recipe 暂时不稳：
   - 以 L0 为主线排障，不反向扩展 L2。
2. 不同模型 method 签名差异大：
   - 在 recipe 层做模型 profile/adapter，不侵入 C++ 主干。
3. 错误码可读性差：
   - 增加错误码映射表与上下文打印。

## 14. 决策记录
1. `prefill_tokens/decode_next_token` 定位为 L2（非主路径）。
2. 平台长期承诺接口是 L0（method 级原语），不是任务级 runner API。
3. 当前阶段只推进 LLM，不提前展开 ASR/TTS。
