# DEBUG 记录（2026-04-15）

## 本轮目标
- 回滚所有本地未提交改动，回到干净状态。
- 在“原版 C++ 链路”验证：
  - `qwen3-0.6b` 是否可跑通；
  - `qwen3.5-0.8b` 是否可跑通；
  - 若失败，定位失败层级与调用栈。

## 当前结论
- 已完成回滚，当前工作区干净。
- 原版 C++ `llama_main` 在干净代码上可跑通 `qwen3-0.6b`。
- `qwen3.5-0.8b` 在 C++ 侧稳定崩溃，错误为 `std::out_of_range`，且不是 tokenizer/loop 问题。
- `executor_runner` 直接加载同一 `qwen3.5` PTE 也崩溃，说明问题在 runtime/delegate 初始化阶段（方法加载期）。

## 已执行步骤与证据

### 1) 回滚与状态确认
```bash
git reset --hard HEAD && git clean -fd
git status --short --branch
```
结果：
- `HEAD is now at f45b708f2 exshader: unify pure-vulkan runtime telemetry and workflows`
- `git status` 仅显示 `## main...origin/main`（无本地改动）

### 2) 原版 C++ 全量重编译确认
```bash
bash exshader/scripts/build_vulkan.sh
```
结果：
- 构建成功，产物存在：`cmake-out-linux-vulkan/examples/models/llama/llama_main`
- 末尾日志：
  - `[100%] Built target llama_main`
  - `[build_vulkan] done`

### 3) `qwen3-0.6b` C++ 链路验证（成功）
命令：
```bash
cmake-out-linux-vulkan/examples/models/llama/llama_main \
  --model_path=artifacts/pte/qwen3_0_6b_vulkan_silu_emb4bit_8da4w.pte \
  --tokenizer_path=~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/.../tokenizer.json \
  --prompt="Write one short sentence about Vulkan." \
  --max_new_tokens=8 \
  --temperature=0
```
关键输出：
- 正常生成文本
- `PyTorchObserver` 指标存在且流程完整：
  - `prefill_token_per_sec: 184.211`
  - `decode_token_per_sec: 112.903`

### 4) `qwen3.5-0.8b` C++ 链路验证（失败）
命令：
```bash
cmake-out-linux-vulkan/examples/models/llama/llama_main \
  --model_path=artifacts/pte/qwen3_5_0_8b_vulkan_fp32_pure_candidate.pte \
  --tokenizer_path=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/.../tokenizer.json \
  --prompt="Write one short sentence about Vulkan." \
  --max_new_tokens=8 \
  --temperature=0
```
错误：
```text
terminate called after throwing an instance of 'std::out_of_range'
what(): vector::_M_range_check: __n (which is 18446744073709551615) >= this->size() (which is 0)
```

### 5) 排除“单个 PTE 损坏”假设
对以下 5 个 PTE 循环测试：
- `qwen3_5_0_8b_vulkan_fp32_disable_fuse.pte`
- `qwen3_5_0_8b_vulkan_fp32_initbuf_fix.pte`
- `qwen3_5_0_8b_vulkan_fp32_nomemplan.pte`
- `qwen3_5_0_8b_vulkan_fp32_nomutbuf.pte`
- `qwen3_5_0_8b_vulkan_fp32_pure_candidate.pte`

结果：
- 全部 `rc=134`，同一 `std::out_of_range` 崩溃。

### 6) 排除“llama_main/tokenizer 循环逻辑”假设
直接使用 `executor_runner`：
```bash
cmake-out-linux-vulkan/executor_runner --model_path artifacts/pte/qwen3_0_6b_vulkan_silu_emb4bit_8da4w.pte
cmake-out-linux-vulkan/executor_runner --model_path artifacts/pte/qwen3_5_0_8b_vulkan_fp32_pure_candidate.pte
```
结果：
- `0.6b`：可执行并输出 tensor。
- `0.8b`：同样 `std::out_of_range` 崩溃。

结论：
- 崩溃与 `llama_main` 的 tokenizer 或生成 loop 无关，问题位于运行时加载/后端 delegate 初始化。

### 7) gdb 调用栈
命令：
```bash
gdb -batch -ex run -ex bt --args cmake-out-linux-vulkan/examples/models/llama/llama_main ...
```
关键栈帧：
```text
#15 executorch::runtime::BackendDelegate::Init(...)
#16 executorch::runtime::Method::init(...)
#17 executorch::runtime::Method::load(...)
#18 executorch::runtime::Program::load_method(...)
#20 executorch::extension::llm::TextLLMRunner::load()
```
说明：
- 崩溃发生在方法加载期间、delegate 初始化路径中，不在 token decode 阶段。

## 当前判断
- 问题类型：跨层 ABI/契约不一致（export -> lower -> preprocess -> runtime）。
- 现有日志不足以一次性定位“第一个失配点”，需要结构化 ABI 证据工具。

## 下一步（工具化追查）
1. 增加 `exshader/diag/abi_dump.py`：导出阶段记录 delegate 输入输出契约（JSON）。
2. 增加 runtime 结构化 trace（例如 `ET_VK_ABI_TRACE_PATH`）输出 expected/actual 参数映射（JSON）。
3. 增加 `abi_diff.py`：对比 `qwen3-0.6b`（golden）与 `qwen3.5-0.8b`（candidate）。
4. 固化 `diag/repro.sh`：clean build + export + run + collect logs。
5. 只在“第一处契约破坏点”修复，修复后双模型回归。

---

## 增量记录（2026-04-15 晚）

### 现象
- `executor_runner` 与 cmake 目录下新编 `_llm_runner` 可跑通 `qwen3_5_0_8b_vulkan_fp32_placeholder_fix.pte`。
- `llama_main` 仍失败，退出码 1，且此前因 `EXECUTORCH_ENABLE_LOGGING=OFF` 缺少可见错误信息。

### 关键定位
1. 在 `examples/models/llama/main.cpp` 增加 stderr 错误打印后，失败明确为：
   - `llama_main: generate failed: Error::Internal (1)`
2. 查看 `ET_DELEGATE_ABI_TRACE_PATH`：
   - 失败发生于 `delegate_init`，并抛出
     `vector::_M_range_check ... size() (which is 0)`。
3. 根因不是新代码逻辑回退，而是**链接到旧 install 库**：
   - `examples/models/llama` 是独立 CMake 项目，通过 `find_package(executorch)` 链接 `cmake-out-linux-vulkan/install`。
   - 该 install 树是旧时间戳产物，未包含当前工作树里的 Vulkan 修复。

### 修复
1. 执行：
   - `cmake --build cmake-out-linux-vulkan --target install`
2. 重新构建并运行 `llama_main`：
   - `qwen3.5-0.8b` 跑通（不再 `Error::Internal`）。
   - `qwen3-0.6b` 回归仍可跑通。
3. 新增固定脚本，避免再次错链旧 install：
   - `exshader/scripts/build_llama_main.sh`
   - `exshader/scripts/run_qwen3_5_llama_main.sh`

## 增量记录（2026-04-16 凌晨，官方 Inspector 主流程）

### 执行目标
- 只用官方 ETRecord/Inspector 工具链定位问题：
  - ETDump + debug buffer
  - Inspector 事件表
  - `calculate_numeric_gap`

### 本轮改动（工具）
- `exshader/diag/collect_inspector_artifacts.py`
  - `_load_for_executorch(..., enable_etdump=True, debug_buffer_size=...)`
  - `Inspector(..., debug_buffer_path=...)`
  - 使用 `parse_etrecord(...); update_representative_inputs(...)` 回填输入
  - 调用 `calculate_numeric_gap(...)` 并写入 JSON
- `exshader/scripts/inspect_with_inspector.sh`
  - 新增 `DEBUG_BUFFER_SIZE / INSPECTOR_NUMERIC_GAP / NUMERIC_GAP_METRICS`
  - 默认不再回存 patched ETRecord（避免 custom op 反序列化再序列化失败）

### 运行命令
```bash
source exshader/env.sh
INSPECTOR_NUMERIC_GAP=1 \
NUMERIC_GAP_METRICS=MSE \
DEBUG_BUFFER_SIZE=1073741824 \
bash exshader/scripts/inspect_with_inspector.sh \
  qwen3_0_6b_vulkan_inspector_flow.pte forward
```

### 结果与证据
- ETDump/Inspector 主流程成功，产物：
  - `artifacts/logs/qwen3_0_6b_vulkan_inspector_flow.etdp`
  - `artifacts/logs/qwen3_0_6b_vulkan_inspector_flow.inspector.csv`
  - `artifacts/logs/qwen3_0_6b_vulkan_inspector_flow.inspector.summary.json`
  - `artifacts/logs/qwen3_0_6b_vulkan_inspector_flow.numeric_gap.json`
- 关键统计（来自 summary）：
  - `rows=1111`
  - `delegate_backend_counts={"VulkanBackend":1108,"None":3}`
  - `first_forward_top1=25`
- `numeric_gap` 失败点（官方 Inspector 报错）：
  - `AssertionError: Cannot auto-functionalize op torch.ops.llama.update_cache.default`
  - 触发位置：`custom_kv_cache.py` 的 `torch.ops.llama.update_cache(...)`

### 进一步确认
- 解析 ETRecord：
  - `exported_program is None`
  - 仅有 `edge_dialect_program`
- 因此 Inspector 自动 fallback 到 `edge_dialect_exported_program`，而该图包含 `llama.update_cache` 高阶自动函数化节点，导致 `calculate_numeric_gap` 无法执行。

### 当前结论
- 官方工具已经接通并给出明确阻塞点：
  1. ETRecord 缺少 ATen `exported_program`（无法走首选 reference graph）。
  2. edge reference graph 中 `llama.update_cache` 不可被 auto-functionalize，阻断 numeric gap。
- 这不是 Vulkan runtime 执行失败，而是官方 numeric gap 参考图可执行性问题。

## 增量记录（2026-04-16，qwen3.5 + 官方 Inspector）

### 新进展
- 已重新导出 `qwen3_5_0_8b_vulkan_fp32_diag_base.pte` 与对应 ETRecord：
  - `artifacts/pte/qwen3_5_0_8b_vulkan_fp32_diag_base.pte`
  - `artifacts/etrecord/qwen3_5_0_8b_vulkan_fp32_diag_base.etrecord.bin`
- Inspector 主流程已稳定跑通（不再因 JSON 导出崩溃）：
  - `artifacts/logs/qwen3_5_0_8b_vulkan_fp32_diag_base.etdp`
  - `artifacts/logs/qwen3_5_0_8b_vulkan_fp32_diag_base.etdp.debug.bin`
  - `artifacts/logs/qwen3_5_0_8b_vulkan_fp32_diag_base.inspector.csv`
  - `artifacts/logs/qwen3_5_0_8b_vulkan_fp32_diag_base.numeric_gap.json`

### 关键统计
- `rows=4294`
- `delegate_backend_counts={"VulkanBackend":3725,"None":569}`
- `first_forward_top1=9150`

### numeric_gap 失败形态（与 qwen3-0.6b 不同）
- 失败错误：
  - `IndexError: list index out of range`
- 调用栈：
  - `inspector._inspector_utils._map_non_sequence_aot_output()` 取 `runtime_intermediate_output[negative_index]` 越界
- 预检查（已在工具中新增）：
  - `runtime_outputs_checked=87`
  - `sequence_len_mismatch_count=6`
  - 存在 `num_outputs > runtime_sequence_len(=0)` 的 debug handle 组（含 `num_outputs=5/37` 等），直接导致 numeric gap mapping 越界。

### 结论
- 官方工具链已把问题收敛到 Inspector 的 runtime/AOT 中间张量映射阶段：
  - 不是单纯“工具不可用”，而是部分 runtime debug 输出与 ETRecord 记录的 `num_outputs` 不一致，导致当前实现越界。
- 这提供了明确修复入口：先修正/规避该映射越界，再继续算子级 numeric gap。

### 额外确认（同日）
- 已修复导出端 ETRecord 生成逻辑：在 `generate_etrecord_func(...)` 中补充 `exported_program`，并使用 `ET_ETRECORD_PATH`。
- 重新导出后校验：
  - `parse_etrecord(...).exported_program is None? -> False`
- 重新跑 Inspector 后不再出现 “falling back to edge_dialect_exported_program” 提示，但 `numeric_gap` 仍因同一类 `IndexError` 失败（即核心阻塞仍是 runtime/AOT mapping 不一致）。
- mismatch 预检查示例（来自 `numeric_gap.json`）：
  - `sequence_len_mismatch_count=6`
  - 示例：`num_outputs=5, runtime_sequence_len=0`（对应 runtime op 如 `aten.mul.Tensor` / `aten.unsqueeze_copy.default` 的 delegate debug 条目）。

## 增量记录（2026-04-16，Vulkan clone 路径 5D 崩溃修复）

### 最小复现（修复前）
命令（pybindings 直接跑单步 forward）：
```bash
source exshader/env.sh
PYTHONPATH="$ET_BUILD_DIR:$PWD/src" .venv/bin/python - <<'PY'
import torch, _portable_lib
m = _portable_lib._load_for_executorch("artifacts/pte/qwen3_5_0_8b_vulkan_allow_aten_clone_default.pte")
mm = m.method_meta("forward")
ins = []
for i in range(mm.num_inputs()):
    s = list(mm.input_tensor_meta(i).sizes())
    ins.append(torch.full(tuple(s), 5328 if i == 0 else 0, dtype=torch.long))
m.run_method("forward", tuple(ins), True)
PY
```
结果：`Segmentation fault (core dumped)`。

## 增量记录（2026-04-17，Qwen3.5 单步定位继续收敛）

### 新增工具口
- 在 `backends/vulkan/runtime/VulkanBackend.cpp` 增加：
  - `ET_VULKAN_EXEC_TRACE=1`
  - `ET_VULKAN_EXEC_TRACE_VALUES=1`
- 新能力：
  - 在每次 Vulkan delegate `execute()` 时，打印每个输入 tensor 的
    `dim / numel / dtype`
  - 当 `ET_VULKAN_EXEC_TRACE_VALUES=1` 时，额外打印前 8 个标量

### 为什么这个工具有效
- 之前我们只能看到：
  - delegate 内部某个 debug event 的输出值
- 但看不到：
  - 这个值在进入 delegate 之前是否已经错了
- 新 trace 直接回答：
  - “copy-in shader 错了”
  - 还是“copy-in 之前源 tensor 就错了”

### 关键证据 1：`instruction 14 / local id 0` 不是实际根因
- 旧的 `agent_debug` 报告给出：
  - `instruction 14`
  - `delegate_id 0`
  - `kernel = nchw_to_image_texture3d_float_float`
  - 对比到 `handle (1,)`
  - 差异很大
- 进一步比对发现：
  - 该 event 的 runtime 值约为 eager `handle 1` 的 `69x`
  - 相关系数接近 `0.995`
- 一开始看起来像：
  - 上传 shader / texture copy 出错

### 关键证据 2：值级 execute trace 证明“上传前就已经错了”
- 用单步 `forward(token=7734, pos=0)` 跑：

```bash
ET_VULKAN_EXEC_TRACE=1 \
ET_VULKAN_EXEC_TRACE_VALUES=1 \
PYTHONPATH="$ET_BUILD_DIR:$PWD/src" \
./.venv/bin/python ...
```

- 在 `artifacts/agent_debug/vulkan_exec_step0.log` 里可见：
  - 某次 delegate execute 的输入 `1024` 维张量仍是小值：

```text
[ET_VULKAN_EXEC_VALUES] ... values=[0.006989,-0.009338,-0.015137,0.019775,...]
```

  - 下一次 delegate execute 的输入 `1024` 维张量已经变成大值：

```text
[ET_VULKAN_EXEC_VALUES] ... values=[0.778810,-0.673800,-1.294793,1.924016,...]
```

- 这说明：
  - `instruction 14` 的 `nchw_to_image_*` 只是把一个已经变大的 tensor 拷进去
  - 不是它把小值放大了

### 关键证据 3：大值来源可追到前一个 delegate 链
- ETDump 里前几个 `DELEGATE_CALL` 顺序为：
  - ordinal 0 -> `instruction 7`
  - ordinal 1 -> `instruction 9`
  - ordinal 2 -> `instruction 14`
- `instruction 9` 内部：
  - `event 38`
  - `kernel = rms_norm_texture3d_float`
  - 输出前 8 个值是：

```text
[0.778810, -0.673800, -1.294793, 1.924016, ...]
```

- 这和 `instruction 14` execute 前输入的前 8 个值完全一致。

### 当前结论
- 之前的“首个坏点 = instruction 14 local 0”是**假阳性**。
- 真正的问题不是 copy-in shader 首次放大 tensor。
- 更准确地说：
  - 当前 `agent_debug / Inspector` 对 Vulkan delegate 内部 local id 的
    `debug_handle -> AOT 节点` 对齐仍然不可靠
  - 尤其是 copy-in / internal local id 的 handle 对齐，会把 event 错配到
    像 `handle 1 (aten_embedding_default)` 这样的错误参考节点

### 这轮学到的架构事实
- Vulkan delegate 内部调试现在可以稳定拿到：
  - local debug id
  - kernel name
  - runtime tensor 内容
- 但仍然缺：
  - 一个**可信的 local-id -> 原始 AOT/Edge 参考节点**映射
- 所以当前工具能可靠回答：
  - “这个 runtime tensor 在哪一个 Vulkan kernel 后面长什么样”
  - “进入当前 delegate 之前，源 tensor 是不是已经错了”
- 还不能可靠回答：
  - “这个 kernel 的输出一定对应 eager 图里的哪一个节点”

### 下一步
1. 修正或重建 Vulkan delegate local-id 到参考图节点的映射，不再把
   `instruction 14` 的 copy-in event 错配成 `handle 1`。
2. 在 `agent_debug` 里把“明显属于 staging / copy-in 的 event”降权或单独标记，
   避免假阳性成为 `first divergence`。
3. 基于新的可信映射，再继续定位真正导致文本错误的 kernel。

### 已落实的小修正
- `agent_debug` 现在会把低置信度的 Vulkan staging/copy kernel
  （如 `nchw_to_*` / `image_to_nchw_*` / `buffer_to_nchw_*`）降权，
  不优先作为 `first divergence`。
- 重新离线诊断 `step0` 后，`first divergence` 从假的
  `instruction 14 / local 0 / nchw_to_image_*`
  移到：
  - `instruction 14`
  - `delegate_id 11`
  - `kernel = binary_mul_texture3d_float`

## 增量记录（2026-04-16，文本异常定位工具）

### 新增工具
- `exshader/diag/llm_step_compare.py`

用途：
- 用同一 prompt 同时跑：
  - eager PyTorch
  - Vulkan ExecuTorch
- 分两层比较：
  - `teacher-forced`：逐步对齐输入 token，比最后一个 logits 的 top1/topk 与数值差异
  - `free-run`：比较真实生成 token 序列与文本

这个工具回答的核心问题是：
- 如果 `teacher-forced` 已经分歧：问题在模型数值 / 导出图 / runtime
- 如果 `teacher-forced` 一致、只有 `free-run` 分歧：问题在 tokenizer / eos / max_len / 生成 loop

### 本轮运行
命令：
```bash
PYTHONPATH=/home/taowen/projects/exshader/src \
LD_LIBRARY_PATH=/home/taowen/projects/exshader/cmake-out-linux-vulkan/extension/llm/custom_ops:$LD_LIBRARY_PATH \
/home/taowen/projects/exshader/.venv/bin/python -m exshader.diag.llm_step_compare \
  --model qwen3_5_0_8b \
  --checkpoint ~/.cache/meta_checkpoints/Qwen_Qwen3.5-0.8B.pth \
  --params /home/taowen/projects/exshader/examples/models/qwen3_5/config/0_8b_config.json \
  --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/.../tokenizer.json \
  --tokenizer-config ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/.../tokenizer_config.json \
  --vulkan-pte /home/taowen/projects/exshader/artifacts/pte/qwen3_5_0_8b_vulkan_fp32_rope_fix.pte \
  --prompt 'Write one short sentence about Vulkan.' \
  --max-seq-len 96 \
  --max-new-tokens 8 \
  --top-k 5 \
  --output-dir /home/taowen/projects/exshader/outputs/qwen3_5_step_compare_vulkan_rope_fix
```

### 结果
- `first_divergence.step_index = 0`
- eager 第一个 token：
  - `271` (`"\\n\\n"`)
- vulkan 第一个 token：
  - `198` (`"\\n"`)
- step0 数值差异：
  - `max_abs_diff_vs_eager = 13.371590614318848`
  - `mean_abs_diff_vs_eager = 1.8055976629257202`

free-run 对比：
- eager:
  - tokens: `[271, 248068, 271, 248069, 271, 53, 58499, 369]`
  - text: `"\n\n<think>\n\n</think>\n\nVulkan is"`
- vulkan:
  - tokens: `[198, 262, 361, 79, 308, 490, 346, 269]`
  - text: `"\n    <pamplagor"`

### 结论
- 现在已经可以排除：
  - tokenizer 解码错误
  - eos / max_len / 生成 loop 语义问题
- 因为 **teacher-forced 在 step 0 就分歧**，问题已经在第一步 forward 的 logits 上。
- 下一个定位方向应是：
  - prefill 第一次 forward 的 Vulkan 数值执行链
  - 优先检查 embedding / early blocks / final logits projection 附近的 delegate 内部输出

### 产物
- `outputs/qwen3_5_step_compare_vulkan_rope_fix/llm_step_compare.json`
- `outputs/qwen3_5_step_compare_vulkan_rope_fix/llm_step_compare.md`

## 增量记录（2026-04-16，PTE sweep 缩小到单个算子）

### 新增工具
- `exshader/diag/llm_pte_sweep.py`

用途：
- 对同一 prompt / 同一 eager teacher-forced 参考
- 批量扫描多份 `.pte`
- 输出每份 PTE 第几步开始偏离 eager

这比单独跑完整生成更适合做 delegated-op 二分定位。

### 第一轮 sweep 结果
扫描：
- `qwen3_5_0_8b_vulkan_allowlist_empty_after_fix.pte`
- `qwen3_5_0_8b_vulkan_allow_et_vk_prepack_default.pte`
- `qwen3_5_0_8b_vulkan_allow_et_vk_rms_norm_default.pte`
- `qwen3_5_0_8b_vulkan_allow_et_vk_silu_mul_default.pte`
- `qwen3_5_0_8b_vulkan_allow_aten_linear_default.pte`
- `qwen3_5_0_8b_vulkan_allow_prefix_01_after_unsqueeze_fix.pte`
- `qwen3_5_0_8b_vulkan_allow_prefix_02_after_unsqueeze_fix.pte`
- `qwen3_5_0_8b_vulkan_fp32_rope_fix.pte`

结论：
- `allowlist_empty_after_fix`: 不偏
- `allow_et_vk_prepack_default`: 不偏
- `allow_et_vk_rms_norm_default`: 不偏
- `allow_et_vk_silu_mul_default`: 不偏
- `allow_aten_linear_default`: 不偏
- `allow_prefix_01_after_unsqueeze_fix`: 不偏
- `allow_prefix_02_after_unsqueeze_fix`: `step0` 开始偏
- `full rope_fix`: `step0` 开始偏

这说明：
- 问题不是所有 Vulkan delegated op 都会触发
- 问题从 `prefix_01 -> prefix_02` 新引入的那批子图开始出现

### runtime init trace 对比
对：
- `allow_prefix_01_after_unsqueeze_fix`
- `allow_prefix_02_after_unsqueeze_fix`

分别抓 `ET_VULKAN_INIT_TRACE_PATH` 后按 `build_graph` 拆子图。

结果：
- `prefix01 subgraphs = 60`
- `prefix02 subgraphs = 296`
- 第一个子图已经不同：
  - `prefix01` 第 0 个子图：
    - `['aten.unsqueeze_copy.default', 'aten.unsqueeze_copy.default', 'aten.unsqueeze_copy.default', 'aten.unsqueeze_copy.default']`
  - `prefix02` 第 0 个子图：
    - `['aten.mul.Tensor']`

这说明从 `prefix_02` 开始，delegate 结构已经发生明显变化，不是“在 prefix_01 上只多挂一个小尾巴”。

### 第二轮 mini sweep（验证最小嫌疑）
扫描：
- `qwen3_5_0_8b_vulkan_allow_aten_mul_Tensor.pte`
- `qwen3_5_0_8b_vulkan_allow_aten_unsqueeze_copy_default.pte`
- `qwen3_5_0_8b_vulkan_allow_et_vk_prepack_default.pte`
- `qwen3_5_0_8b_vulkan_allowlist_empty_after_fix.pte`

结果：
- `allow_aten_mul_Tensor`: `step0` 开始偏
- `allow_aten_unsqueeze_copy_default`: 不偏
- `allow_et_vk_prepack_default`: 不偏
- `allowlist_empty_after_fix`: 不偏

### 当前最强结论
- 现在已经把问题缩小到：
  - **单独允许 `aten.mul.Tensor` delegate 到 Vulkan，就足以让 qwen3.5 在 step0 logits 偏离 eager**
- 所以优先怀疑：
  - Vulkan `aten.mul.Tensor` 的实现路径
  - 特别是 qwen3.5 首次 prefill 中命中的那类 `mul`
  - 可能涉及：
    - tensor-tensor vs tensor-scalar 分支
    - broadcast / shape / packed dim
    - storage alias / clone / prepack 交互

### 调试辅助改动
- `extension/pybindings/pybindings.cpp`
  - 之前的 `std::cerr` 调试输出已改成环境变量控制：
    - `ET_PYBINDINGS_TRACE=1` 时才打印

这能避免 sweep / compare 工具被大量 `run_method` 噪声污染。

### 根因
`qwen3.5 allow_aten_clone_default` 图里存在 5D KV cache 输入（如 `[1,2,4,2048,256]`）。  
这些值在 runtime 图构建时被按 `Texture3D` 创建，随后 staging/clone 路径会访问 `texture_meta_ubo/logical_limits`。  
但 Vulkan `TextureMetadata` 与相关 UBO 逻辑固定只支持 4D，导致越界/空指针路径并触发崩溃。

### 修复
1. 在 `GraphBuilder::add_tensor_to_graph` 增加保护：
   - 若序列化值是 texture 且 `ndim > 4`，强制回退为 `BUFFER` 存储。
   - 文件：`backends/vulkan/runtime/VulkanBackend.cpp`
2. 在 staging 与 clone 路径增加显式维度校验，避免静默走到非法 texture 元数据路径：
   - `backends/vulkan/runtime/graph/ops/impl/Staging.cpp`
   - `backends/vulkan/runtime/graph/ops/impl/Clone.cpp`
3. 在 `vTensor::logical_limits()` 增加 `uniform_data_` 判空保护（>4D 不可用时给出明确错误）：
   - `backends/vulkan/runtime/api/containers/Tensor.h`

### 修复后验证
1. 重编译：
```bash
bash exshader/scripts/build_vulkan.sh
```
2. 同一复现命令重跑：
   - 不再崩溃；
   - `top1=9150` 正常返回。
3. 端到端回归：
```bash
MAX_NEW_TOKENS=8 TEMPERATURE=0 \
bash exshader/scripts/run_qwen3_5_llama_main.sh \
  qwen3_5_0_8b_vulkan_fp32_diag_base.pte \
  "Write one short sentence about Vulkan."
```
结果：流程正常完成并输出 `PyTorchObserver` 统计，无崩溃。

### 额外证据
启用 `ET_VULKAN_INIT_TRACE_PATH` 后，可看到 fallback 事件：
```text
"event":"build_graph_storage_fallback","message":"fb_id=1,dims=[1,2,4,2048,256],fallback_storage=BUFFER"
"event":"build_graph_storage_fallback","message":"fb_id=2,dims=[1,2,4,2048,256],fallback_storage=BUFFER"
```

## 2026-04-16 晚间补充：当前 `exshader` 状态复核

### 结论先行
- 现在 `qwen3.5-0.8b` 的 Vulkan 链路已经能：
  - 成功加载 `.pte`
  - 成功完成 prefill / decode
  - 成功返回 token
- 但**输出仍然不正确**，还不能算“跑通”。
- 最新复核里，问题依旧出现在：
  - **teacher-forced step 0 logits 就已经偏离 eager**

### 最新端到端复核
命令：
```bash
cd /home/taowen/projects/exshader
TOK_DIR=/home/taowen/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
PYTHONPATH=/home/taowen/projects/exshader/src \
LD_LIBRARY_PATH=/home/taowen/projects/exshader/cmake-out-linux-vulkan/extension/llm/custom_ops:${LD_LIBRARY_PATH:-} \
./.venv/bin/python -m exshader.diag.llm_step_compare \
  --model qwen3_5_0_8b \
  --checkpoint ~/.cache/meta_checkpoints/Qwen_Qwen3.5-0.8B.pth \
  --params /home/taowen/projects/exshader/examples/models/qwen3_5/config/0_8b_config.json \
  --tokenizer $TOK_DIR/tokenizer.json \
  --tokenizer-config $TOK_DIR/tokenizer_config.json \
  --vulkan-pte /home/taowen/projects/exshader/artifacts/pte/qwen3_5_0_8b_vulkan_fp32_rope_fix.pte \
  --prompt 'Write one short sentence about Vulkan.' \
  --max-seq-len 96 \
  --max-new-tokens 2 \
  --top-k 5 \
  --output-dir /tmp/qwen35_step_compare_latest
```

结果摘要：
- `first_divergence.step_index = 0`
- eager `step0 top1 = 271`，decoded 为 `"\n\n"`
- vulkan `step0 top1 = 198`，decoded 为 `"\n"`
- `max_abs_diff_vs_eager = 13.371590614318848`
- `mean_abs_diff_vs_eager = 1.8055976629257202`

free-run 对照：
- eager:
  - tokens: `[271, 248068]`
  - text: `"\n\n<think>"`
- vulkan:
  - tokens: `[198, 262]`
  - text: `"\n   "`

所以当前不是：
- tokenizer decode 问题
- 生成 loop 问题
- `max_len` 配置问题

而是：
- **第一次 forward 的 logits 就已经错了**

### 新的最小复现：`aten.mul.Tensor` 不是“全面坏掉”，而是特定形态有问题
为了把 qwen 问题缩小到 op 级，额外做了三个最小 `mul` 广播复现。

测试脚本形状：
- `[(1,1,1024)] * [(1024,)]`
- `[(1,1,1024)] * [(1,1,1)]`
- `[(1,6144,4)] * [rank-0 scalar tensor]`

结果：
- `[1,1,1024] * [1024]`
  - `max_diff = 0.0`
- `[1,1,1024] * [1,1,1]`
  - `max_diff = 0.0`
- `[1,6144,4] * rank-0 tensor`
  - runtime **直接 Abort(core dumped)**

这说明：
- 普通 broadcast `mul` 并没有普遍错误
- 当前更像是：
  - **rank-0 tensor / scalar-like tensor 路径存在缺口**
  - 或者 qwen 里的 `mul` 命中了更特殊的 storage/layout 组合

### 与 qwen `allow_aten_mul_Tensor` trace 对齐后的观察
`/tmp/allow_mul_vk_init.jsonl` 前几个算子里就能看到：
- `aten.mul.Tensor` with
  - `[1,1,1024] x [1,1,1024]`
  - `[1,1,1024] x [1,1,1]`
  - `[1,6144,4] x []`
  - `[1,1,1024] x [1024]`

其中第三类正好对应最小复现里会直接 `Abort` 的 rank-0 tensor case。

### 当前最可能的问题面
- `backends/vulkan/runtime/graph/ops/impl/BinaryOp.cpp`
- 重点不是“普通乘法公式”，而是：
  - rank-0 tensor 如何被建图
  - rank-0 tensor 在 texture path 上如何被 broadcast / indexing
  - 这类 tensor 是否应该继续走 generic tensor-tensor kernel

### 当前判断
- **回答“能跑了么”：**
  - 执行链路能跑
  - 语义还没跑对
  - 因此还不能算真正跑通

## 2026-04-16 晚间补充二：修复 rank-0 tensor 输入 ABI 崩溃

### 现象
把问题继续缩小后发现，一个比 Vulkan 数值问题更靠前的 bug 一直存在：

- 任意普通 tensor `x`
- 与一个 **rank-0 tensor 输入** `y = torch.tensor(0.5)` 相乘
- 导出成 Vulkan `.pte` 后执行
- 会在 Python `run_method()` 阶段直接 `Abort`

最小复现示例：
```python
class M(torch.nn.Module):
    def forward(self, x, y):
        return x * y
```

即使 `x` 只是：
- `[4]`
- `[1, 4]`
- `[1, 1, 4]`

也都会崩。

### 定位过程
加 trace 后发现：
- 有 `[pybindings] run_method begin`
- 没有 delegate ABI trace
- 没有 method execute trace
- 没有 Vulkan execute trace

这说明崩溃发生在：
- `extension/pybindings/pybindings.cpp`
- `PyModule::run_method()` 的**输入转换阶段**
- 还没有真正进入 `module_->execute()`

进一步检查 portable 路径：
- rank-0 `at::Tensor`
- 在 pybindings 中会被转成 `torch::executor::TensorImpl`
- 但 `sizes/strides/dim_order` 都是空 vector
- 随后 `alias_etensor_to_attensor()` 会调用 `check_tensor_meta()`
- 里面硬性要求：
  - `b.sizes().data() != nullptr`
  - `b.strides().data() != nullptr`

因此：
- rank-0 tensor 虽然 `dim == 0` 合法
- 但 metadata 指针为空
- 直接触发 `ET_CHECK`

### 修复
文件：
- `extension/pybindings/pybindings.cpp`

修复方式：
- 新增 `populate_etensor_metadata_from_aten(...)`
- 对 rank-0 tensor：
  - 仍保持 `dim = 0`
  - 但给 `sizes/strides/dim_order` 提供 1 个哨兵元素
  - 这样 metadata 指针非空
  - 同时不会改变 rank 语义，因为 `dim == 0` 时这些值不会被按真实维度读取

同时把另一条输入转换路径也一起修掉：
- `PyMethod.call()` / builder 式 tensor 输入构造

### 验证
最小复现重新执行：
- 不再崩溃
- `max_diff = 0.0`

新增单测：
- `extension/pybindings/test/test_pybindings.py::test_rank0_tensor_input`
- 结果：通过

### 对 qwen3.5 的影响判断
修完后重新跑：
- `exshader.diag.llm_step_compare`
- `qwen3_5_0_8b_vulkan_fp32_rope_fix.pte`

结果：
- `first_divergence.step_index` 仍然是 `0`
- `max_abs_diff_vs_eager` 仍然是 `13.371590614318848`
- Vulkan `step0 top1` 仍然是 `198`
- eager `step0 top1` 仍然是 `271`

结论：
- **rank-0 tensor 输入 ABI 崩溃是一个真实 bug，现已修复**
- 但它**不是** `qwen3.5` 数值偏差的根因
- `qwen3.5` 的剩余问题仍应继续沿：
  - `allow_aten_mul_Tensor`
  - Vulkan `aten.mul.Tensor`
  - broadcast / internal scalar-like tensor 路径
 继续排查

## 2026-04-17 凌晨补充：修复 `agent_debug` 单步抓取失效，并确认 delegate capture 编号语义

### 问题 1：`reset_etdump()` 之后 debug buffer 丢失

为了让 `agent_debug` 能做：
- 前缀 step 重放，保留 KV/cache 状态
- 清空 ETDump
- 只抓目标 step

之前给 `PyProgram` 增加了 `reset_etdump()`。

但实际抓出来的 `.etdp` 只有 profile event，没有任何 delegate `debug_event`。

进一步查源码发现：
- `devtools/etdump/etdump_flatcc.cpp`
- `ETDumpGen::reset()` 会把 `data_sink_ = nullptr`

而 debug tensor 写入依赖 `data_sink_` 指向 debug buffer。

因此：
- reset 之后 event tracer 仍存在
- debug level 仍存在
- filter 也仍存在
- 但 tensor 再也写不进 debug buffer
- 结果就是 ETDump 里只剩 profile event

### 修复

文件：
- `extension/pybindings/pybindings.cpp`

修改：
- `PyProgram::reset_etdump()` 在调用 `event_tracer_->reset()` 后
- 如果 `debug_buffer_size_ > 0`
- 立即重新调用：

```cpp
event_tracer_->set_debug_buffer(get_etdump_debug_buffer());
```

### 验证

修复前：
- `artifacts/agent_debug/qwen3_5_diag_focus_instr9/agent_trace_step6.etdp`
- `runs = 1`
- `debug_event_entries = 0`
- `profile_event_entries = 4366`

修复后，用全范围 focus 重新抓：
- `artifacts/agent_debug/qwen3_5_debugevent_probe_all/agent_trace_step6.etdp`
- `events = 5040`
- `debug = 674`
- `profile = 4366`

说明：
- 单步抓取链路现在真的能保留状态并抓到 delegate 内部张量

### 问题 2：Vulkan profile metadata id 与 delegate capture id 不是同一套编号

之前误以为：
- profile metadata 里看到的
  - `instruction_id=9`
  - `delegate_debug_id=17..21`
- 就应该直接作为 scoped focus 传回 runtime

但实际这样抓不到任何 delegate debug event。

进一步验证后确认：
- profile event 里的 `delegate_debug_id`
  - 是 Vulkan profiling / dispatch metadata 使用的编号
- debug event 里的 `delegate_debug_id_int`
  - 是 **instruction 内局部编号**
  - 对每个 ExecuTorch instruction 都会从 `0` 重新开始计数

实际证据：
- 对 step0 做全范围抓取后
- `instruction 14` 下看到的 delegate debug ids 是：
  - `0..41`
- 而 profile metadata 日志里同一 instruction 曾显示：
  - `35..47`

结论：
- 这两套编号**不能直接互用**
- 之前 `focus instruction 9 + debug_handles [17,18,19,20,21]` 抓不到东西是预期结果

### 当前 agent_debug 状态

目前已经确认：
- `agent_debug` 的单步重放机制可用
- delegate 内部张量抓取主路径可用
- 但工具仍有一个重要缺口：
  - report / inspector 还没有把
    - profile metadata dispatch id
    - delegate debug local id
    - kernel name
    三者自动对齐

这导致 report 中虽然已经能看到：
- `instruction_id`
- `delegate_id`（本地编号）
- 数值 diff

但解释成“具体是哪一个 Vulkan shader / kernel”仍需额外对照。

### 当前最有价值的定位结果

对 `target_step_index=0` 做全范围 delegate capture 后：

- `total runtime events = 5040`
- `matched outputs = 39`
- `unmapped outputs = 572`
- 首个报告出的偏差：
  - `event 79`
  - `instruction 14`
  - `delegate_id 0`
  - `max_abs_diff = 4.42043`

Top divergence 里还反复出现：
- `instruction 25 / 27 / 46 / 69`
- `delegate_id 0 / 1 / 11 / 33 / 34`

这说明：
- 现在已经不是“完全看不到 delegate 内部”
- 而是进入了真正的数值定位阶段
- 下一步应优先补工具，把
  - instruction-local delegate id
  - 对应 kernel name
  - 对应 AOT op / handle
  自动连起来

## 2026-04-17: debugger 补强，显式区分 ETRecord 可映射与 runtime-only Vulkan local id

本轮修复的重点不是“硬把所有 Vulkan event 都映射回 AOT 节点”，而是先让工具诚实。

### 发现

用 `ETRecord._delegate_map` 直接核对后确认：

- `instruction 14`
  - ETRecord delegate map 只有 local id `0..34`
  - runtime debug event 实际还会发出 local id `35..41`
- `instruction 16`
  - ETRecord delegate map 只有 local id `0..31`
  - runtime 还会发出 local id `32..36`

这些超出上界的 local id 对应的是 runtime 可见的 Vulkan kernel，例如：

- `rsqrt_float_texture3d`
- `binary_mul_texture3d_float`
- `permute_texture3d_float`
- `select_texture3d_float`
- `view_texture_float`

所以之前的问题不是“工具不会用”，而是：

- 旧报告默认只看 `event.debug_handles != None` 的事件
- 对于没有精确 ETRecord mapping 的 Vulkan local id，要么静默跳过，要么被 inspector 逐条 warning 淹没

### 已做修改

`devtools/agent_debug/core.py`

- `TensorDiffFinding` 新增：
  - `mapping_confidence`
  - `mapping_source`
- 报告排序改为：
  - 先按 mapping confidence
  - 再按 numeric gap
- 继续保留：
  - Vulkan staging / copy kernel 降权
- 新增：
  - 对 `debug_data` 存在但 `debug_handles` 为空的 Vulkan event 做汇总
  - 按 instruction 输出：
    - 哪些 local id 超出 ETRecord 覆盖范围
    - ETRecord 覆盖上界是多少
    - 对应 kernel / operator 示例

`devtools/agent_debug/cli.py`

- 增加 import root 自检：
  - 如果当前 working tree 是 `exshader`
  - 但实际 import 到的是另一份 editable install
  - CLI 会明确 warning，要求显式加：
    - `PYTHONPATH=/home/taowen/projects/exshader/src:/home/taowen/projects/exshader`

`diagnose_target_step()`

- 增加 inspector log 降噪：
  - 抑制海量重复的
    - `No exact delegate debug mapping ...`
  - 因为这类信息现在已经以结构化 note 的形式进入最终 report

### 当前效果

离线跑 step0 诊断时，报告现在会直接给出：

- `first divergence`
- `mapping exact/high/heuristic`
- 每个 instruction 的 unmapped Vulkan local id 摘要

例如：

- `instruction 14 emitted 7 debug tensors without exact ETRecord mapping`
- `ETRecord delegate_map for this instruction stops at local id 34`
- 并列出代表性 kernel / op

这解决了两个真实问题：

- 避免 agent 被海量 warning 干扰
- 避免把“没有精确 mapping 的 Vulkan runtime kernel”误当成已经精确定位完成

### 进一步修正：runtime operator 与 AOT handle 节点全集冲突时，不再做 handle-only fallback

又补了一层约束：

- 如果 runtime Vulkan event 提供了 `delegate_operator_name`
- 且同一个 debug handle 对应的所有 AOT 节点里，没有任何一个 target 能匹配这个 operator
- 那么该 event 直接视为 `unmapped`
- 不再回退成：
  - `heuristic`
  - `reused_debug_handle`

这样修完后，step0 里原先那个误导性的 first divergence：

- `instruction 14`
- `delegate_id 11`
- `kernel binary_mul_texture3d_float`

已经不再被选中。

新的 step0 first divergence 变成：

- `instruction 25`
- `delegate_id 32`
- `kernel binary_mul_texture3d_float`
- `mapping_confidence = high`
- `mapping_source = debug_handle+operator_name`
- `ops = aten_mul_tensor_9`

这说明当前 debugger 至少已经做到：

- 不再把明显 operator 对不上的 reused handle 当成“可疑但可参考”
- first divergence 会优先落在真正有 operator-level 约束的 event 上
