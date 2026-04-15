# AGENTS.md

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
