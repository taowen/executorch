# AGENTS.md

## 最终目标
把仓库收敛为 `exshader` 驱动的 **Pure Vulkan + Python-first** 架构：
1. 运行主链路是 Pure Vulkan，不接受 CPU fallback 混跑。
2. 高频研发在 Python 完成（融合边界、shader、loop、编排）。
3. C++ 是低频稳定基座，不按任务类型复制 runner。
4. 目标是减少 C++ 重编译，提升 agent 自主迭代速度。

## 硬约束
1. 不要兼容层：同一能力只保留一条主路径。
2. 不要 fallback：主链路出现 fallback 直接失败。
3. 不计成本不计代价：优先架构一致性与演进效率。
4. `tools/` 冻结：新增能力放 `exshader/`。
5. 可交付流程禁止 runtime hack（`PYTHONPATH` 注入、临时 symlink 等）。
6. 禁止任务专用 C++ runner 扩散（`ASRRunner/TTSRunner/...` 非目标）。

## C++/Python 接口总原则（本次更正）

### 一句话
**C++ 提供通用 method 执行原语，Python 承担全部任务语义。**

### C++ 基座（允许做）
1. Program/Session 生命周期：load、reload/reset、释放。
2. Method 执行原语：method 枚举、method 元信息、execute。
3. 输入输出与内存接口：类型/shape 检查、可复用 buffer、显式错误码。
4. Vulkan runtime 能力与可观测性：性能计时、delegate/backend 信息。

### C++ 基座（禁止做）
1. 不引入任务语义 API（如 LLM/ASR/TTS 专用 loop）。
2. 不在 C++ 实现产品级“采样策略/业务流程状态机”。
3. 不通过新增任务专用 pybind 模块来解决模型差异。

### Python 层职责
1. `exshader/runtime`：唯一对上暴露的运行时封装层。
2. `exshader/recipes/*`：任务编排层（多次 forward、采样、多模块调度）。
3. 融合边界与 shader 迭代：继续走 Python 导出/partition/JIT 链路。

## 接口分层（执行标准）

### L0（基座接口，主路径）
`exshader.runtime.Session` + `MethodHandle`：
1. `Session.load(...)`
2. `session.method(name)`
3. `MethodHandle.meta()`
4. `MethodHandle.run(inputs, clone_outputs=False)`
5. `Session.reset_state()`

### L1（任务封装，可变）
`exshader.recipes.llm_decode` 等 Python recipe：
1. 只能依赖 L0。
2. 可以快速改采样、chunk、cache 策略。
3. 不要求改 C++。

### L2（历史/过渡接口，非主路径）
`TextLLMRunner.prefill_tokens/decode_next_token` 这类任务语义接口：
1. 可用于排障或实验。
2. 不作为未来架构主干。
3. 不以其成功与否决定平台方向。

## 当前状态（聚焦接口）
1. `exshader/runtime` 已存在，`Session/MethodHandle` 已可用。
2. `exshader/runtime` 已补充接口草案中的核心类型：
   - `SessionOptions`
   - `RunResult/RunStats`
   - method 级错误上下文信息（包含 method 名与输入摘要）。
3. `exshader/recipes/llm_decode.py` 已收敛回 L0（`Session/MethodHandle`），不依赖 `TextLLMRunner` 任务接口。
4. Qwen3 现状验证：
   - `run_qwen3.sh` 可跑通。
   - `check_qwen3_pure_vulkan.sh` 通过（`KernelCall=0, DelegateCall=1`）。
5. `_llm_runner` 已支持 core-only 构建：
   - `-DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER_TORCH_IO=OFF`
   - 关闭后不导出 `make_image_input/make_audio_input/make_raw_audio_input`。
6. `runner/__init__.py` 已移除 `import torch` 顶层硬依赖（torch-io helper 为可选符号）。
7. 已验证 `_llm_runner` core 产物不链接 torch 动态库。
8. `exshader/scripts/run_qwen3.sh` 与 `check_qwen3_pure_vulkan.sh` 已支持自动选择可用 Qwen3 PTE（避免默认文件名失效）。
9. `_exshader_runtime` 已支持结构化运行统计（`host_input_ms/module_execute_ms/output_wrap_ms/elapsed_ms`），并已接入 `exshader.runtime.MethodHandle.run` 与 `exshader/recipes/llm_decode.py` 输出。
10. Vulkan backend telemetry 已接入（`copy_inputs/resize/compute_graph_execute/copy_outputs/total_backend`）；`export_qwen3.sh` 新增 `ET_ENABLE_QUERYPOOL_PROFILE=1` 导出开关用于获取 `gpu_shader_total_ms/dispatch_count`。
11. 已验证 `8da4w + emb4bit + enable_querypool` 导出可同时满足 Pure Vulkan 与 shader 级统计输出（`gpu_shader_total_ms`, `dispatch_count` 非 0）。
12. `exshader/scripts/export_qwen3.sh` 默认导出参数已切到 `quantization.qmode=8da4w` + `quantization.embedding_quantize="4,32"`，避免默认导出触发 `KernelCall=1`。

## 下一阶段（只做两件事）

### M4-3（P0）：把 LLM recipe 严格收敛到 L0
1. 已完成：`exshader/recipes/llm_decode.py` 只使用 `Session/MethodHandle`。
2. 已完成：固化内存复用策略（预分配 + in-place + `clone_outputs=False`）。
3. 已达成当前验收：Qwen3 跑通、脚本不依赖任务专用 C++ API。

### M4-4（P0）：继续去 torch 运行时硬依赖
1. 已完成：`runner/__init__.py` 去掉顶层 `import torch`，改为按需加载。
2. 已完成：core-only `_llm_runner` 下缺失 `make_image_input/make_audio_input` 不再导致导入失败（降级为可选符号）。
3. 已完成：禁 torch 条件下验证 `import executorch.extension.llm.runner` 可成功。
4. 持续项：继续清理其他非主路径（文档/测试）中的 torch 假设，但不阻塞 exshader 主链路。

## 目录约定
- `backends/vulkan/runtime/`：低频 C++ 基座（原地演进，不迁目录）。
- `backends/vulkan/patterns/` + `backends/vulkan/vulkan_preprocess.py` + `backends/vulkan/partitioner/`：融合与分区。
- `exshader/export_llm.py`：唯一导出入口。
- `exshader/runtime/`：通用运行时接口（主战场）。
- `exshader/recipes/`：Python 任务编排层（当前只推进 LLM）。
- `exshader/shader_jit/`：shader bundle 构建与热替换。
- `exshader/scripts/`：固定脚本入口。
- `artifacts/`：统一产物目录。
- `tools/`：冻结，不新增功能。

## 验收口径
1. 纯 Vulkan：runtime fallback=0；静态 `KernelCall` 持续追踪并优化。
2. 迭代效率：改融合/改shader/改loop 不重编译 C++ 主体。
3. 架构一致性：任务能力优先在 Python recipe 扩展，不新增任务专用 C++ runner。
