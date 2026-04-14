# AGENTS.md

## 目标
- 只保留一条主线：`PyTorch -> ExecuTorch 导出 -> Vulkan 运行验证`。
- 用 `Qwen3-0.6B` 做端到端验证。
- 最终仓库应尽量精简，只保留“导出 + Vulkan 运行”所需代码。
- 全流程使用 `uv` 管理 Python 环境。

## 执行原则
- 所有步骤默认在仓库根目录执行：`/home/taowen/projects/executorch`。
- 先删明显无关 backend，再安装依赖，避免无用依赖下载。
- 每一步都要可重复执行（幂等），失败可重试。

## Phase 0: 第一轮裁剪（安装前）
先做一次“确定无用”删除，目标是 iOS / Android Java / QNN 等与当前目标无关部分。

```bash
set -euxo pipefail

# iOS / Apple
rm -rf backends/apple extension/apple examples/apple

# Qualcomm QNN
rm -rf backends/qualcomm examples/qualcomm

# Android Java/JNI 侧（纯 C++ Vulkan runner 不需要）
rm -rf extension/android extension/android_test examples/demo-apps

# 其他明显无关厂商 backend（按需保守删）
rm -rf backends/mediatek backends/samsung backends/nxp backends/cadence
```

说明：不要删 `backends/vulkan`、`extension/llm`、`examples/models/qwen3`、`examples/models/llama`。

## Phase 1: 用 uv 准备最小环境
```bash
set -euxo pipefail
uv venv .venv
source .venv/bin/activate

# 最小安装，避免 example 依赖全集
uv run ./install_executorch.py --minimal
```

## Phase 2: 导出 Qwen3-0.6B 到 Vulkan
```bash
set -euxo pipefail
source .venv/bin/activate

uv run python -m extension.llm.export.export_llm \
  +base.model_class=qwen3_0_6b \
  +base.params=examples/models/qwen3/config/0_6b_config.json \
  +backend.vulkan.enabled=true \
  +backend.vulkan.force_fp16=true \
  +model.dtype_override=fp32 \
  +model.use_kv_cache=true \
  +model.use_sdpa_with_kv_cache=true \
  +export.output_name=qwen3_0_6b_vulkan.pte
```

输出文件：`./qwen3_0_6b_vulkan.pte`

## Phase 3: 纯 Vulkan 运行验证（本机 Linux）
```bash
set -euxo pipefail
source .venv/bin/activate

# 先确认本机 Vulkan 环境可用（至少有 Vulkan loader + 驱动 + glslc）
vulkaninfo | head -n 20 || true
command -v glslc

cmake -S . -B cmake-out-linux-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=python \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_BUILD_COREML=OFF \
  -DEXECUTORCH_BUILD_OPENVINO=OFF \
  -DEXECUTORCH_BUILD_QNN=OFF \
  -DEXECUTORCH_BUILD_TESTS=OFF

cmake --build cmake-out-linux-vulkan -j"$(nproc)" --target install --config Release

cmake -S examples/models/llama -B cmake-out-linux-vulkan/examples/models/llama \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=python \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF

cmake --build cmake-out-linux-vulkan/examples/models/llama -j"$(nproc)" --config Release
```

本机运行验证：
```bash
# tokenizer 路径按本机实际缓存调整
TOKENIZER_JSON=$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json | head -n1)

./cmake-out-linux-vulkan/examples/models/llama/llama_main \
  --model_path ./qwen3_0_6b_vulkan.pte \
  --tokenizer_path "${TOKENIZER_JSON:?}" \
  --prompt "Write a short poem about Vulkan."
```

## Phase 4: 必要代码识别与第二轮裁剪
先收集“实际被用到”的代码，再删剩余无关目录。

建议保留白名单（最小起点）：
- `backends/vulkan/`
- `extension/llm/`
- `examples/models/qwen3/`
- `examples/models/llama/`
- `runtime/`
- `kernels/`
- `schema/`
- `exir/`
- `export/`
- `third-party/`（按构建报错再细化）

第二轮删除规则：
- 任何不在白名单且不被 CMake/导出脚本引用的目录，直接删除。
- 每删一批，必须重新执行一次 Phase 2 + Phase 3 冒烟验证。

## 验收标准
- 能稳定产出 `qwen3_0_6b_vulkan.pte`。
- `llama_main` 能在本机 Linux 加载并推理该 `.pte`。
- 构建参数里仅启用 Vulkan（`EXECUTORCH_BUILD_VULKAN=ON`，`EXECUTORCH_BUILD_XNNPACK=OFF`）。
- 相比初始仓库，非目标 backend 代码显著减少，且流程仍可复现。
