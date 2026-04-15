# ExecuTorch Vulkan Shader Pipeline (PyTorch -> Vulkan)

本文档说明当前仓库里一条真实可运行的链路：

1. Export from PyTorch (`torch.export` / FX-style graph)
2. Optimize graph in Python (operator selection + fusion)
3. Generate Vulkan compute shaders (GLSL)
4. Compile shaders to SPIR-V
5. Execute on GPU through a Vulkan runtime

并补充：如果把一块子图重写成一个更大的 Vulkan shader，是否需要重新导出权重。

---

## 1) Export from PyTorch (`torch.export` / FX graph)

LLM 导出主线在 `extension/llm/export`：

- 入口：
  - `extension/llm/export/export_llm.py`
  - `executorch.examples.models.llama.export_llama_lib.export_llama(...)`
- 关键 builder：
  - `extension/llm/export/builder.py`

核心行为：

- 使用 `torch.export.export(...)` 得到 `ExportedProgram`（见 `builder.py` 的 `_export()`）。
- 导出后会执行 `run_decompositions({})`，把图规整到后续 pass/partitioner 可处理的形式。
- 图本质是 FX 风格（`GraphModule` + Node），后续所有“算子选择/融合/分区”都在这个层面进行。

---

## 2) Optimize graph in Python (operator selection + fusion)

### 2.1 子图选择（哪些节点可交给 Vulkan）

主要在：

- `backends/vulkan/partitioner/vulkan_partitioner.py`
- `backends/vulkan/op_registry.py`

机制：

- `VulkanSupportedOperators` 基于 `op_registry` 的 `OpFeatures` 判断节点是否支持：
  - dtype 约束
  - storage/memory-layout 可表示性
  - dynamic shape 支持
  - 是否在 allowlist/blocklist
- `CapabilityBasedPartitioner` 生成最大可委托子图，并打上 `delegation_tag`。

### 2.2 图融合（把多个节点重写成更少/更大操作）

主要在：

- `backends/vulkan/vulkan_preprocess.py`

`VulkanBackend.preprocess(...)` 中串行执行多个 pass，例如：

- `AddmmToLinearTransform`
- `FusePatternsPass`
- `FuseQuantizedOpsTransform`
- `FoldQDQPass`
- `FuseViewCopyTransform`
- `ViewCopyToSqueezeUnsqueezePass`
- `insert_prepack_nodes`

随后用 `VkGraphBuilder` 构建 Vulkan delegate graph，并序列化为 `processed_bytes` 写入 `.pte`。

结论：当前“算子融合”主要是 Python pass 层做的图改写，不是运行时临时做。

---

## 3) Generate Vulkan compute shaders (GLSL)

shader 源码在：

- `backends/vulkan/runtime/graph/ops/glsl/`

构建时由：

- `backends/vulkan/runtime/gen_vulkan_spv.py`

完成以下工作：

- 处理 GLSL 模板和宏配置（含 dtype/layout/特性条件）
- 生成最终 `.glsl`
- 解析 shader 元信息（descriptor layout、tile size、扩展需求、dispatch 注册信息）

---

## 4) Compile shaders to SPIR-V (`glslc`)

### 关键事实（当前仓库实现）

当前实现是 **构建阶段调用 `glslc`**，不是推理时启动 `glslc` 进程。

证据路径：

- `backends/vulkan/cmake/ShaderLibrary.cmake`
  - `find_program(GLSLC_PATH glslc ...)`
  - `add_custom_command(...)` 调用 `gen_vulkan_spv.py --glslc-path=...`
- `backends/vulkan/runtime/gen_vulkan_spv.py`
  - 通过 `subprocess.run([...glslc...])` 编译 `.glsl -> .spv`
  - 支持多线程并行编译（`ThreadPool`）

然后脚本会把 `.spv` 打包生成 `spv.cpp`：

- 内嵌 SPIR-V 二进制数组
- 生成 `ShaderInfo` 注册代码
- 生成 op -> shader dispatch 注册代码

`spv.cpp` 最终被编进 `vulkan_backend` 静态库。

### 运行时发生的“编译”

运行时不会再调用外部 `glslc`，但 Vulkan driver 仍会把 SPIR-V 做 pipeline/JIT 创建（`vkCreateShaderModule` / pipeline cache）。

---

## 5) Execute on GPU through Vulkan runtime

运行时主链路：

- `backends/vulkan/runtime/VulkanBackend.cpp`
  - `init(...)`：解析 delegate 数据、构建 `ComputeGraph`、prepare pipeline、prepack
  - `execute(...)`：拷入输入、必要时 resize、`compute_graph->execute()`、拷回输出

shader/dispatch 关键点：

- `backends/vulkan/runtime/api/ShaderRegistry.*`
  - 全局注册表保存 `ShaderInfo`（SPIR-V、layout、扩展需求）
- `backends/vulkan/runtime/graph/ops/DispatchNode.cpp`
  - encode descriptor set
  - 注册并提交 shader dispatch
- `backends/vulkan/runtime/vk_api/Shader.cpp`
  - 用已内嵌的 SPIR-V 创建 `VkShaderModule`

---

## 6) 推理控制流：可复用 forward + 任务定制 loop

你问的判断是对的：仓库里确实是两层结构。

- 一层是“可复用的模型方法执行”（`module_->execute(method, inputs)`）。
- 另一层是“任务特定的 host-side loop”（C++ 里控制 prefill/decode/streaming/停止条件等）。

### 6.1 LLM（Qwen3/Llama）链路

核心文件：

- `extension/llm/runner/text_llm_runner.cpp`
- `extension/llm/runner/text_prefiller.cpp`
- `extension/llm/runner/text_token_generator.h`
- `extension/llm/runner/text_decoder_runner.cpp`
- `extension/llm/runner/io_manager/io_manager.h`

行为拆解：

- `TextDecoderRunner::step(...)` 是“单步 forward 调用封装”，本身尽量保持函数式（不做外层循环状态机）。
- `TextPrefiller` 负责 prompt prefill（可并行/可分块）。
- `TextTokenGenerator::generate(...)` 是 decode while-loop（逐 token 调 `step`，采样，EOS 判断）。
- `TextLLMRunner::generate(...)` 把 prefill + decode + tokenizer decode + stats 串起来。

KV cache 管理方式：

- loop 端维护 `start_pos/pos`（host state）。
- 每步把 token + position（或 cache_position）作为输入喂给模型。
- cache 的实际更新通常在模型图/方法内部完成；runner 主要负责“位置推进 + 调度”。
- `IOManager` 默认假设 prefill/decode 方法输入是 `(token, start_pos)` 两个输入；若模型要求显式 cache tensor 输入，则需要定制 IOManager/runner。

### 6.2 ASR（Whisper）链路

核心文件：

- `extension/asr/runner/runner.cpp`
- `extension/asr/runner/runner.h`
- 示例入口：`examples/models/whisper/main.cpp`

行为拆解：

- 先执行 `encoder` 一次（音频特征 -> encoder states）。
- 再进入 `text_decoder` 自回归循环（可选 `sampler` 方法）。
- loop 内维护 `input_id`、`cache_position`、EOS 停止、token 回调。

这也是“可复用方法 + 自定义 loop”模式，只是 loop 逻辑与 LLM 文本生成不同（先 encoder，再 decoder）。

### 6.3 TTS 现状（本仓库）

截至当前仓库代码，未看到类似 `extension/asr/runner` 这种统一的 `extension/tts/runner` 基础 runner。

- 结论：若做 TTS，通常也会走同样模式：
  - 复用导出的模型方法（例如声学模型、vocoder、streaming 子模块）；
  - 由任务侧自定义 C++/Python loop 串联这些方法（分块、状态、终止条件、后处理）。

---

## 如果把子图重写成一个大的 Vulkan shader，要不要重新导出权重？

短答案：

- 需要重新导出 `.pte`
- 不需要重新训练模型权重

解释：

- 子图融合会改变 delegate 图结构、算子边界、可能的 prepack/layout 与 compile spec，属于“编译产物变化”，所以必须重导出 `.pte`。
- 原始 checkpoint 权重通常不变，只是以新的方式被打包/预处理进导出产物。
- 例外：如果只改 runtime shader 内部实现，且 op 名称、参数协议、序列化格式完全不变，旧 `.pte` 可能可复用；但只要图结构或序列化协议变化，就必须重导出。
