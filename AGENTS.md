# AGENTS.md

## 核心原则（硬约束）
1. 不要兼容（No Compatibility Layer）
   - 不保留双入口、不保留过渡 wrapper、不保留旧路径别名。
   - 同一能力只保留一条主路径。
2. 不要 fallback（No Fallback Path）
   - 目标链路只允许主路径：Pure Vulkan + Python-first。
   - 不以 CPU fallback、旧 backend fallback、运行时兜底逻辑作为可交付方案。
3. 不计成本，不计代价（All-in for Target Architecture）
   - 优先保证架构一致性、可演进性和研发速度。
4. `tools/` 冻结（No New Additions Under tools）
   - 后续新增脚本/模块默认放 `exshader/*`，不再往 `tools/` 增加新内容。
5. 导出链路工程化（No Runtime Hacks）
   - 可交付流程禁止依赖 `PYTHONPATH`/运行时 `sys.path` 注入。
   - 可交付流程禁止依赖 symlink 指向 `.venv` 二进制库。

## 当前状态

### 已完成（精简版）
1. 纯 Vulkan 主链路收敛完成
   - 导出分区主路径为 Vulkan；非目标 backend 在当前分支为禁用/报错路径。

2. 导出入口迁移完成
   - 主入口：`python -m exshader.export_llm`
   - 旧入口 `extension/llm/export/export_llm.py` 已移除。

3. 融合控制 Python 化已验证
   - 仅修改 Python（profile / pass 配置）可改变融合边界，不需要改 C++、不需要重编译。
   - `disable_fuse_patterns` / `disable_fuse_quantized_ops` 已真实接入 Vulkan preprocess。

4. Shader bundle 基础能力已接入（M3 基础）
   - Runtime 可读取 `shader_bundle_path` 并加载外部 bundle。
   - `exshader/shader_jit/build_shader_bundle.py` 可生成 bundle。

5. 导出工程化门禁已接入
   - 路径一致性检查（确保来自当前 repo）。
   - `custom_ops_aot_lib` 存在性检查（缺失即 fail-fast）。
   - profile 严格 schema 校验（未知字段报错，不做隐式兼容）。

6. 产物目录规范已落地
   - 统一目录：`artifacts/`
   - `pte`：`artifacts/pte/`
   - shader bundle：`artifacts/shader_bundles/`
   - 构建目录收敛：仅保留 `cmake-out-linux-vulkan/`

7. 固定脚本已落地（不再依赖临时命令）
   - `exshader/env.sh`
   - `exshader/scripts/build_vulkan.sh`
   - `exshader/scripts/export_qwen3.sh`
   - `exshader/scripts/run_qwen3.sh`
   - `exshader/scripts/check_qwen3_pure_vulkan.sh`

### 待完成（下一阶段）
1. CI 化（P0）
   - 增加 smoke test：构建 + 导出 + 纯 Vulkan 检查（Qwen3-0.6B 最小配置）。
   - 验收：CI 可稳定复现，失败时输出明确日志与原因。

2. M3 完整化（P0）
   - 把“改 GLSL -> 重新打 bundle -> 运行生效”的流程固化为一键脚本与回归用例。
   - 验收：无 C++ 重编译即可完成 shader 热替换验证。

3. M4 Python 推理编排层（P1）
   - 在 `exshader/loop/` 落地 LLM decode loop（多次 forward + kv cache 管理）。
   - 验收：主 loop 不依赖 `llama_main` C++ 逻辑。

4. M5 多任务扩展（P1）
   - 在 `exshader/recipes/` 增加 ASR/TTS/LLM 的 recipe 组织方式。
   - 验收：新增模型主要改 Python，不改 C++ 基座。

5. 回归体系（P1）
   - 固化性能与纯 Vulkan 回归（prefill/decode、KernelCall/DelegateCall、分区算子列表）。
   - 验收：可比较 baseline 与 candidate，自动给出差异。

## 当前执行入口（唯一推荐）

### 快速全流程
```bash
set -euxo pipefail
bash exshader/scripts/build_vulkan.sh
bash exshader/scripts/export_qwen3.sh qwen3_0_6b_vulkan_pure_candidate.pte
bash exshader/scripts/run_qwen3.sh qwen3_0_6b_vulkan_pure_candidate.pte
bash exshader/scripts/check_qwen3_pure_vulkan.sh qwen3_0_6b_vulkan_pure_candidate.pte
```

### 环境准备
```bash
set -euxo pipefail
uv venv .venv
. .venv/bin/activate
uv pip install -e .
python ./install_executorch.py --minimal
source exshader/env.sh
```

### Shader bundle（M3）
```bash
set -euxo pipefail
source exshader/env.sh
.venv/bin/python exshader/shader_jit/build_shader_bundle.py \
  --glsl-paths backends/vulkan/runtime/graph/ops/glsl \
  --bundle-dir "$ET_SHADER_BUNDLE_DIR" \
  --glslc-path glslc \
  --optimize

ET_VULKAN_PARTITIONER_PROFILE="{\"compile_options\":{\"shader_bundle_path\":\"$ET_SHADER_BUNDLE_DIR\"}}" \
bash exshader/scripts/export_qwen3.sh qwen3_0_6b_vulkan_bundle.pte
```

## 目录约定（执行标准）
- `backends/vulkan/runtime/`：低频 C++ 基座（原地演进，不迁移目录）
- `backends/vulkan/patterns/` + `backends/vulkan/vulkan_preprocess.py` + `backends/vulkan/partitioner/`：融合与分区主实现
- `exshader/export_llm.py`：唯一导出入口
- `exshader/shader_jit/`：shader bundle 工具
- `exshader/scripts/`：固定执行脚本
- `artifacts/`：所有本地产物
- `tools/`：冻结目录，不新增功能

## 验收口径（强约束）
1. 纯 Vulkan 口径
   - `pure_vulkan=true` 目标下，出现 CPU fallback 即判失败。
2. 观测口径
   - 每次导出/融合调整都要有：
   - Vulkan 分区算子列表
   - 纯 Vulkan 检查结果（KernelCall/DelegateCall）
   - prefill/decode 性能指标
3. 迭代口径
   - 高频修改在 Python；C++ 仅在 runtime 能力缺口时低频修改。
