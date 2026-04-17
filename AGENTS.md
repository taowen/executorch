# AGENTS.md

## Goal
把仓库收敛为 `exshader` 驱动的 **Pure Vulkan + Python-first** ExecuTorch 工作流：
1. 从 PyTorch 导出 `.pte`
2. 在 Linux 上以 Vulkan backend 运行
3. 高频改动放在 Python 导出、融合、调试、shader 工作流
4. C++ runtime 保持小而稳定

## Hard Rules
1. 不要兼容层：同一能力只保留一条主路径。
2. 不要 fallback：主链路出现 CPU fallback 视为失败。
3. 不计成本不计代价：优先架构一致性和后续演进效率。
4. `tools/` 冻结：新增能力放 `exshader/` 或已有主链目录。
5. 不为具体任务扩散专用 C++ runner。
6. 禁止 `portable_lib` 路径：模型执行主路径必须走 `exshader` 自己的 runtime shim，不允许新增或依赖 `executorch.extension.pybindings.portable_lib` / `_load_for_executorch` 作为运行入口。

## Environment Bootstrap
环境必须满足“两件事同时成立”：
1. 命令从 `/home/taowen/projects/exshader` 根目录运行。
2. `.venv` 里的 `executorch` editable install 必须指向当前仓库，而不是别的 checkout。

推荐的干净重建流程：
1. `cd /home/taowen/projects/exshader`
2. `uv venv --seed --clear .venv`
3. `./.venv/bin/python install_requirements.py --example`
4. `CMAKE_ARGS='-DEXECUTORCH_BUILD_PYBIND=ON -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_XNNPACK=OFF -DEXECUTORCH_BUILD_COREML=OFF -DEXECUTORCH_BUILD_OPENVINO=OFF -DEXECUTORCH_BUILD_QNN=OFF -DEXECUTORCH_BUILD_TESTS=OFF -DEXECUTORCH_BUILD_EXTENSION_LLM=ON -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXSHADER_RUNTIME=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON -DEXECUTORCH_BUILD_KERNELS_LLM=ON -DEXECUTORCH_BUILD_KERNELS_LLM_AOT=ON' ./.venv/bin/python -m pip install --no-build-isolation -e .`
5. `exshader/scripts/build_vulkan.sh --clean`
6. `source exshader/env.sh`

必要说明：
1. 这棵树是 Vulkan-only，editable install 不能走默认 `pybind` preset 配置；默认会把 `EXECUTORCH_BUILD_XNNPACK` 打开，直接触发根目录 `CMakeLists.txt` 的 pruned-tree 校验失败。
2. `exshader` runtime shim 依赖 `EXECUTORCH_BUILD_PYBIND=ON`。只开 `EXECUTORCH_BUILD_EXSHADER_RUNTIME=ON` 不够；如果 `EXECUTORCH_BUILD_PYBIND=OFF`，顶层 CMake 不会把 `extension/exshader/runtime` 加进构建图。
3. 仅仅传 `-DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_XNNPACK=OFF` 还不够。Qwen 主链实际依赖 `build_vulkan.sh` 那套完整开关，尤其是：
   - `EXECUTORCH_BUILD_EXSHADER_RUNTIME=ON`
   - `EXECUTORCH_BUILD_EXTENSION_LLM=ON`
   - `EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON`
   - `EXECUTORCH_BUILD_EXTENSION_MODULE=ON`
   - `EXECUTORCH_BUILD_EXTENSION_TENSOR=ON`
   - `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON`
   - `EXECUTORCH_BUILD_KERNELS_LLM=ON`
   - `EXECUTORCH_BUILD_KERNELS_LLM_AOT=ON`
4. 如果 `extension/llm/tokenizers` 安装时抱怨 CMake cache 指向别的仓库，先删：
   - `extension/llm/tokenizers/build`
   - `third-party/ao/build`
   然后重跑安装。
5. `.venv` 重建以后，优先跟一次 `exshader/scripts/build_vulkan.sh --clean`，避免 Python 绑定和旧的 Vulkan 产物错配。
6. 环境修好以后，不应该再依赖 `PYTHONPATH="$PWD/src:$PWD"` 这种临时覆盖；它只适合拿来快速确认“当前 `.venv` 是否串仓库了”。

每次重建后先做自检：
1. `./.venv/bin/python - <<'PY'`
   `import importlib.util`
   `mods = [`
   `    "executorch.examples.models.llama.runner.native",`
   `    "executorch.extension.llm.export.partitioner_lib",`
   `    "executorch.backends.vulkan.partitioner.vulkan_partitioner",`
   `]`
   `for m in mods:`
   `    spec = importlib.util.find_spec(m)`
   `    print(m, "->", spec.origin if spec else None)`
   `PY`
2. `cat .venv/lib/python3.12/site-packages/executorch-*.dist-info/direct_url.json`

正确结果必须全部指向：
1. `/home/taowen/projects/exshader/...`
2. `file:///home/taowen/projects/exshader`

如果看到的是 `/home/taowen/projects/executorch/...`，说明 `.venv` 已污染，直接重建，不要继续调模型或 runtime。

## Validated State
当前已经验证的模型路径：
1. `qwen3_0_6b`
2. `qwen3_5_0_8b`

当前已经验证的关键事实：
1. `qwen3_5_0_8b` eager 可运行。
2. `qwen3_5_0_8b` pure-Vulkan 导出可生成并运行。
3. `qwen3_5_0_8b` 的 teacher-forced compare 已和 eager 对齐。
4. 当前验证通过的导出产物是：
   - `artifacts/pte/qwen3_5_0_8b_vulkan_fp32_statefix_main.pte`

## Proven Root Cause
`qwen3_5_0_8b` 之前输出异常的根因不是“Vulkan 数值误差”，而是
export/lowering 合同错误：

1. mutable buffer 被错误地当成 delegate 内部 buffer / prepack 常量。
2. 结果是 live state 没有作为 runtime input 传给 Vulkan 子图。
3. 修复原则是：
   - mutable buffer 不进入 lowered submodule 自有占位符集合
   - mutable buffer 不参与 constant duplication
   - mutable buffer 不走 Vulkan prepack 常量路径
   - mutable buffer 在 lowered ABI 中必须表现为 `USER_INPUT`

这条约束现在必须保持。

## Current Debugging Stack
优先使用已经在树内的工具，不再临时写一次性脚本：

1. `devtools/agent_debug/llm_step_compare.py`
   - 用于 teacher-forced 对齐
   - 快速判断是“第一步就错”还是“自由生成 loop / tokenizer / max_len 问题”
2. `exshader/diag/abi_diff.py`
   - 用于比对 export/runtime ABI 结构
   - 适合发现 delegate 输入输出合同漂移
3. `devtools/agent_debug/`
   - 用于 ETRecord / Inspector / delegate 级联调试

## Working Rules For Qwen-Family Validation
验证 Qwen 系列模型时，先排除配置问题，再追查 Vulkan：
1. chat template 必须正确
2. tokenizer 和 tokenizer_config 必须匹配
3. `runner.native --max_len` 是总序列长度，不是“最多新生成 token”
4. 先做 teacher-forced compare，再判断是否是 backend 问题

## Repo Shape
1. `backends/vulkan/`
   - 低频 C++ 基座
   - Vulkan partition / preprocess / runtime / shader 相关实现
2. `exshader/export_llm.py`
   - Python 导出主入口
3. `exshader/runtime/`
   - Python-first runtime 封装
4. `exshader/models/`
   - 每个模型各自独立的导出与执行代码
5. `exshader/diag/`
   - 面向 agent 的调试与诊断工具
6. `exshader/scripts/`
   - 固定脚本入口
7. `artifacts/`
   - 本地产物目录，默认不入库

## Next Work
1. 在新导出的 `qwen3_5_0_8b_vulkan_fp32_statefix_main.pte` 上补完整 free-run 验证。
2. 继续把调试链路做成“快速判定合同错误 / tensor 对齐错误”的标准流程。
3. 只在排除 prompt / runner / tokenizer / max_len 问题后，再追查 Vulkan 数值与 runtime 问题。
