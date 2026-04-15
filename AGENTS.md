# AGENTS.md

## 当前状态（已完成代码收敛）
已把导出主链路收敛到 **Pure Vulkan**：
- `extension/llm/export/partitioner_lib.py`：仅保留 `get_vulkan_partitioner` 实现；其它 backend 入口保留同名函数但直接报错。
- `examples/models/llama/export_llama_lib.py`：主导出路径仅走 Vulkan；`_get_source_transforms` 中 `qnn/mps/coreml/mlx` 分支已删除；多后端分支改为禁用。
- `backends/vulkan/partitioner/vulkan_partitioner.py`：新增基于 op 名称的 allow/block list（支持前缀 `*` 通配）。
- `backends/vulkan/vulkan_preprocess.py`：支持 Python 侧开关 `disable_fuse_patterns` / `disable_fuse_quantized_ops`。
- `et_vk.silu_mul` 融合样例已打通（pattern + custom op + runtime op + shader）：
  - Python 融合：`backends/vulkan/patterns/silu_mul.py`
  - 自定义 op：`backends/vulkan/custom_ops_lib.py`
  - Runtime dispatch：`backends/vulkan/runtime/graph/ops/impl/BinaryOp.cpp`
  - Shader 变体：`backends/vulkan/runtime/graph/ops/glsl/binary_op_{buffer,texture}.yaml`

## 算子融合实现（当前真实机制）
融合不是 C++ 侧硬编码调度，而是 Python 图改写 + 分区共同决定：
1. 源变换：`examples/models/llama/source_transformation/*` 先把模型改写为更适合 delegate 的图（如 `sdpa_with_kv_cache`）。
2. Vulkan 预处理 pass：`backends/vulkan/vulkan_preprocess.py` 顺序执行 `FusePatternsPass`、`FuseQuantizedOpsTransform` 等。
3. Pattern 替换：`backends/vulkan/patterns/*` 将 ATen 子图替换为 `et_vk.*` 自定义 op（如 `rms_norm`、`apply_rotary_emb_hf`）。
4. 分区：`backends/vulkan/partitioner/vulkan_partitioner.py` 基于 op 支持矩阵与约束，把节点合并成 Vulkan 子图。
5. 结果：融合边界本质是“哪些节点先被替换 + 哪些节点被分到同一 Vulkan 子图”。

新增融合规则的开发原则：
- 优先改 `source_transformation` / `patterns`（定义新的融合语义）。
- `partitioner` 仅用于边界与约束控制（allow/blocklist、shape/limit、纯 Vulkan 门禁）。
- 不把“新增融合语义”放在 partitioner 里实现。

## 下一个大目标（Phase 6）
把“可控算子融合”从单点验证推进到**高效率研发工作流**：
1. `M3` Shader JIT / 热替换（优先级最高）
   - C++ 底座补一次能力：运行时加载外部 SPIR-V bundle。
   - Python 工具链：GLSL -> SPIR-V -> bundle 打包与下发。
   - 必须覆盖“融合新算子”场景：新增 `et_vk.*` fused op 后，可通过 shader bundle + Python 注册流程落地，不要求每次改动都重编译 C++。
   - 验收：
     - 只改 GLSL + Python 命令即可生效，不重编译 C++ 主体。
     - 新 fused op 能在导出日志中命中，并在 Vulkan runtime 正常 dispatch。
     - `check_pure_vulkan.py` 通过（无 KernelCall fallback）。
2. `M4` Python 推理编排层
   - 用 Python 管理多次 forward 与 KV cache，先覆盖 LLM decode loop。
   - 验收：主 loop 不依赖 `llama_main` 的 C++ 逻辑。
3. `M5` 扩展到多模型任务流（ASR/TTS）
   - 通过 Python recipe 组合 encoder/decoder/sampler。
   - 验收：新增模型主要改 Python，保持 pure Vulkan 约束。

## 最近实测（2026-04-15）
1. `silu_mul` 融合命中已验证
   - 非本地源码路径（.venv 旧包）导出时未命中。
   - 使用包含新 pattern 的代码后，Qwen3 导出日志出现 `et_vk.silu_mul.default`。

2. 纯 Vulkan状态
   - 未做 embedding quantize 的导出：`KernelCall=1 + DelegateCall=1`（非纯 Vulkan）。
   - `embedding_quantize="4,32"` 后：`DelegateCall=1`，`KernelCall=0`，`check_pure_vulkan.py` 通过。

3. 性能（同 prompt、`max_new_tokens=64`）
   - `qwen3_0_6b_vulkan_emb4bit.pte`（历史基线）三次均值：decode ~`161.9` tok/s。
   - `qwen3_0_6b_vulkan_silu_emb4bit_8da4w.pte` 三次均值：decode ~`170.3` tok/s。
   - 当前观测：decode 有提升，prefill 波动较大，需固定时钟与更长样本再复核。

## 自定义融合实验流程（标准）
1. 选择目标子图
   - 先在导出图中确定稳定模式（输入输出形状与 dtype 约束清楚）。
   - 优先选择 decode 热路径高频子图，避免低价值“只改边界”实验。
2. 实现融合规则（Python）
   - 在 `backends/vulkan/patterns/` 新增 pattern 文件。
   - 实现 detector/graph matcher + replacement，将子图替换为目标 fused op。
   - 在 `backends/vulkan/patterns/__init__.py` 注册导入。
3. 对齐 Vulkan 支持
   - 若 fused op 是新 op 名，更新 `backends/vulkan/op_registry.py`。
   - 仅在确有需要时再调整 `partitioner` 约束。
4. 三组对照导出
   - A: baseline（无新融合）
   - B: profile-only（仅配置改边界）
   - C: candidate（启用新融合）
5. 固定四项对比输出
   - `tools/et_tools/check_pure_vulkan.py` 结果
   - `Operators included in this Vulkan partition`
   - prefill/decode 性能指标
   - 数值一致性摘要
6. 验收
   - candidate 出现预期 fused op 且边界变化可解释。
   - 纯 Vulkan检查通过（无 CPU fallback）。
   - 与 baseline 数值一致性在阈值内。
   - decode 指标不退化，或有明确收益。
   - 有最小回归样例覆盖该融合规则。

## 当前阶段：Phase 5（Python-first，Pure Vulkan）
目标：在 Linux 上形成“低频 C++ 基座 + 高频 Python 控制层”的开发模式，推理目标态为**纯 Vulkan**（不接受 CPU fallback 作为最终状态）。

约束定义：
- 纯 Vulkan：核心算子执行必须在 Vulkan delegate 内完成。
- 融合：指调整 Vulkan shader/子图边界，不是 Vulkan+CPU 混合切分。
- Python 优先：导出、融合策略、任务 loop、实验控制优先在 Python 完成。

## 可行性评估（基于当前代码）
1. Python 多次 forward 调度：可行
   - 现有 `runtime/__init__.py` 已提供 `Runtime -> Program -> Method.execute()` 接口，足够承载 LLM/ASR/TTS 的 Python loop。
2. 融合粒度 Python 化：可行
   - Vulkan partition/registry 位于 Python 侧（`backends/vulkan/partitioner`、`backends/vulkan/op_registry.py`），可通过配置驱动融合边界。
3. Shader JIT 且“不重编译 C++”：部分可行，需一次性补底座
   - 当前 shader 主要通过 `gen_vulkan_spv.py` 生成并静态编入 C++。
   - 要做到“改 GLSL 立即生效”，需要补一个低频 C++ 能力：运行时加载外部 SPIR-V bundle（或等价注册入口）。
4. ASR/TTS 全量纯 Vulkan：中期目标
   - LLM（Qwen3-0.6B）可作为第一优先验证路径。
   - ASR/TTS 需按模型逐个推进，不承诺一步到位全模型纯 Vulkan。

结论：计划总体可行，但必须分阶段推进，并把“shader 动态加载”设为前置里程碑。

## 里程碑（修正版）
1. `M1` 建立可复现 Pure Vulkan 基线（Qwen3-0.6B）
   - 导出、构建、运行命令固定化。
   - 形成 fallback 检查脚本（日志关键字 + 覆盖率报告）。
   - 验收：同一命令可稳定复现，且核心路径无 CPU fallback。

2. `M2` 融合粒度配置化（Python）
   - 增加 `fusion_profile`：pass 开关、allow/block list、buffer/shape 阈值。
   - 输出“未下放原因”报告（按算子统计）。
   - 验收：不改 C++ 即可改变 shader 边界，并能解释每处 fallback 原因。

3. `M3` Shader JIT（一次性 C++ 补口 + Python 工作流）
   - C++：新增运行时 shader bundle 加载能力（低频修改）。
   - Python：GLSL -> SPIR-V 编译与 bundle 打包、热替换脚本。
   - 验收：仅改 GLSL + Python 命令即可生效，无需重新链接 C++ 主体。

4. `M4` Python 任务编排层（LLM -> ASR/TTS）
   - 先落地 LLM decode loop（含 kv cache 管理）到 Python。
   - 再接 ASR/TTS 最小可用 loop（encoder/decoder/sampler 组合）。
   - 验收：多次 forward 逻辑主要在 Python，C++ runner 可降为可选。

5. `M5` 扩模型与回归体系
   - 新模型接入模板化（导出配置 + loop recipe + Vulkan 覆盖报告）。
   - 性能/正确性回归自动化。
   - 验收：新增模型主要改 Python 配置与脚本。

## 目录规划
- `python/et_fusion/`：融合/分区策略与覆盖报告
- `python/et_shader_jit/`：GLSL->SPV JIT、bundle 生成与加载工具
- `python/et_loop/`：LLM/ASR/TTS Python 编排层
- `python/et_tools/`：profiling、trace、回归脚本

## 验收口径（强约束）
1. 功能口径
   - `pure_vulkan=true` 模式下，出现 CPU fallback 即判失败。
2. 观测口径
   - 每次导出/融合调整都产出：
     - Vulkan 覆盖率
     - 未下放算子列表及原因
     - 关键性能指标（prefill/decode）
3. 迭代口径
   - 高频变更集中在 Python；C++ 仅在 runtime 能力缺口时低频变更。

## 风险与应对
1. `op_registry` 覆盖不足导致 fallback
   - 应对：优先补 Vulkan op 支持或调整导出图，禁止以 CPU fallback 作为“完成”。
2. 大 tensor/embedding 超阈值无法下放
   - 应对：参数化 `buffer_limit`，并在报告中明确触发点。
3. shader 动态加载未打通前研发效率低
   - 应对：把该能力前置到 `M3`，作为后续快速迭代前提。

## Baseline（当前可运行）
### 1) 环境准备
```bash
set -euxo pipefail
uv venv .venv
.venv/bin/python ./install_executorch.py --minimal
```

### 2) 导出 Qwen3-0.6B（Pure Vulkan 候选）
```bash
set -euxo pipefail
FLATC_EXECUTABLE="$PWD/.venv/bin/flatc" \
.venv/bin/python -m extension.llm.export.export_llm \
  base.model_class=qwen3_0_6b \
  base.params=examples/models/qwen3/config/0_6b_config.json \
  model.enable_dynamic_shape=true \
  model.use_kv_cache=true \
  model.use_sdpa_with_kv_cache=true \
  model.quantize_kv_cache=false \
  backend.vulkan.enabled=true \
  backend.vulkan.force_fp16=true \
  model.dtype_override=fp32 \
  export.output_name=qwen3_0_6b_vulkan_pure_candidate.pte
```

### 3) 构建（Vulkan-only）
```bash
set -euxo pipefail
cmake -S . -B cmake-out-linux-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/cmake-out-linux-vulkan/install" \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_BUILD_COREML=OFF \
  -DEXECUTORCH_BUILD_OPENVINO=OFF \
  -DEXECUTORCH_BUILD_QNN=OFF \
  -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON

cmake --build cmake-out-linux-vulkan -j"$(nproc)" --target install --config Release

cmake -S examples/models/llama -B cmake-out-linux-vulkan/examples/models/llama \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_PREFIX_PATH="$PWD/cmake-out-linux-vulkan/install" \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF

cmake --build cmake-out-linux-vulkan/examples/models/llama -j"$(nproc)" --config Release
```

### 4) 运行验证
```bash
set -euxo pipefail
TOKENIZER_JSON=$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json | head -n1)

./cmake-out-linux-vulkan/examples/models/llama/llama_main \
  --model_path ./qwen3_0_6b_vulkan_pure_candidate.pte \
  --tokenizer_path "${TOKENIZER_JSON:?}" \
  --prompt "Write a short poem about Vulkan."
```

### 5) 纯 Vulkan 检查（必须）
```bash
set -euxo pipefail
TOKENIZER_JSON=$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json | head -n1)

.venv/bin/python tools/et_tools/check_pure_vulkan.py \
  --pte ./qwen3_0_6b_vulkan_pure_candidate.pte \
  --flatc ./.venv/bin/flatc \
  --run-cmd "./cmake-out-linux-vulkan/examples/models/llama/llama_main --model_path ./qwen3_0_6b_vulkan_pure_candidate.pte --tokenizer_path ${TOKENIZER_JSON:?} --prompt 'Write a short poem about Vulkan.'"
```

- 若检查失败：定位未下放算子或 runtime 日志中的 fallback，再回到 `M2` 修融合/支持，不计入达成。
