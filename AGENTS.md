# AGENTS.md

## 文档定位
本文档只记录**源码层面的架构理解**与开发约束。  
调试过程、实验记录、日志结论统一写入 `DEBUG.md`。

---

## 0. 固化命令（唯一入口）

所有构建/导出/运行都走 `exshader/scripts/*`，禁止手写临时 `cmake` 命令。

1. 构建（唯一入口）  
   `bash exshader/scripts/build_vulkan.sh`  
   清理重建：`bash exshader/scripts/build_vulkan.sh --clean`
2. 导出 Qwen3 0.6B（Pure Vulkan）  
   `bash exshader/scripts/export_qwen3.sh`
3. 导出 Qwen3.5 0.8B（Pure Vulkan）  
   `bash exshader/scripts/export_qwen3_5.sh`
4. 运行 Qwen3 推理  
   `bash exshader/scripts/run_qwen3.sh`
5. 纯 Vulkan 检查  
   `bash exshader/scripts/check_qwen3_pure_vulkan.sh`
6. 构建 C++ `llama_main`（会先同步 install 库，避免旧库错链）  
   `bash exshader/scripts/build_llama_main.sh`
7. 运行 Qwen3.5 0.8B（`llama_main`）  
   `bash exshader/scripts/run_qwen3_5_llama_main.sh`
8. ETRecord + Inspector 主流程（单步 forward 采样）  
   先导出时开启 ETRecord：  
   `ET_GENERATE_ETRECORD=1 bash exshader/scripts/export_qwen3.sh`  
   或  
   `ET_GENERATE_ETRECORD=1 bash exshader/scripts/export_qwen3_5.sh`  
   再生成 ETDump 并解析：  
   `bash exshader/scripts/inspect_with_inspector.sh <pte_name>`
9. Inspector 基线/候选自动对比报告  
   `bash exshader/scripts/inspect_compare.sh <baseline_pte> <candidate_pte>`

---

## 1. 总体架构（Pure Vulkan + Python-first）

目标链路分两段：
1. **导出链路（Python）**：PyTorch 图 -> EXIR Edge -> Vulkan delegate blobs -> `.pte`
2. **执行链路（C++ Runtime + Vulkan backend）**：`.pte` -> Method init -> delegate init -> delegate execute

核心原则：
1. 融合、分区、编排优先在 Python 完成。
2. C++ 负责稳定执行语义与后端实现，不承载任务特化 loop。
3. 运行主链路是 Pure Vulkan（不设计混合 fallback 作为正常路径）。

---

## 2. 源码入口与职责分层

### 2.1 导出入口
- `exshader/export_llm.py`
  - 负责读取 Hydra 配置、环境校验、调用 `export_llama(...)`。

### 2.2 LLM 导出编排
- `examples/models/llama/export_llama_lib.py`
  - `_validate_args(...)`：约束纯 Vulkan 分支配置。
  - `_to_edge_and_lower_llama(...)`：关键主流程  
    `pt2e_quantize -> export_to_edge -> to_backend(partitioners) -> to_executorch`
  - `export_llama(...)`：汇总模型装载、变换、量化、lowering、落盘。

### 2.3 分区器构造
- `extension/llm/export/partitioner_lib.py`
  - `get_vulkan_partitioner(...)`：创建 `VulkanPartitioner`。
  - `compile_options` 与 allow/blocklist 由配置与 `ET_VULKAN_PARTITIONER_PROFILE` 驱动。

### 2.4 Backend lowering 通用层
- `exir/backend/backend_api.py`
  - `to_backend(...)`：按 backend id 调用对应 `preprocess(...)`。
  - `_partition_and_lower_*`：按 delegation tag 建 submodule 并 lowered。
  - `_insert_lowered_submodule(...)`：把 `call_module` 替换成 `executorch_call_delegate(...)`。

### 2.5 Vulkan backend preprocess（编译前端）
- `backends/vulkan/vulkan_preprocess.py`
  - `VulkanBackend.preprocess(...)`：执行 Vulkan 专属 pass pipeline。
  - 输出 `processed_bytes`（Vulkan 图 + 常量数据 + header），写入 delegate data。

### 2.6 Runtime 执行核心
- `runtime/executor/method.cpp`
  - `Method::init(...)`：
    - 解析 values
    - 初始化 delegates（`BackendDelegate::Init`）
    - 解析 chains/instructions（含 `DelegateCall` 参数索引）
  - `Method::execute(...)`：按 instruction 派发 KernelCall / DelegateCall。

### 2.7 Vulkan backend 运行时实现
- `backends/vulkan/runtime/VulkanBackend.cpp`
  - `init(...)`：读取 compile specs，构建 `ComputeGraph`，反序列化并 `compileModel(...)`。
  - `execute(...)`：输入拷贝/resize/graph execute/输出拷贝。

---

## 3. 关键数据与 ABI 契约

### 3.1 Delegate 契约（ExecutionPlan 级）
1. `execution_plan.delegates[i]` 的顺序即 `DelegateCall.delegate_index` 的索引语义。
2. Delegate 初始化在 `Method::init` 完成；若任一 delegate init 失败，方法初始化失败，不会进入执行阶段。

### 3.2 DelegateCall 参数契约
1. `DelegateCall.args` 是 values 表索引列表。
2. Runtime 在 init 阶段将索引解析为 `Span<EValue*>` 参数表，执行时直接传给 backend。
3. Vulkan backend 在 execute 中按自身 graph 的 `num_inputs/num_outputs` 消费该参数表：
   - 前段对应 inputs
   - 尾段对应 outputs
4. 因此导出侧 `call_delegate` 参数构造顺序与 runtime/backend 解释必须一致。

### 3.3 Lowering 时输入裁剪契约
1. `_insert_lowered_submodule(...)` 会从 `call_module.args` 中剔除该分区“自有常量”（参数/缓冲），仅保留 user inputs 进入 `executorch_call_delegate`。
2. 常量通过 lowered module 内状态与 serialized delegate data 传递，不应重复出现在 runtime call args。

### 3.4 CompileSpec 契约
1. Python 侧写入 `CompileSpec(key, value-bytes)`。
2. Runtime 透明传递到 backend init，不解释 key 语义。
3. Vulkan backend 在 `get_graph_config(...)`/`get_shader_bundle_path(...)` 解释具体 key（如 `require_dynamic_shapes`, `force_fp16`, `shader_bundle_path`）。

### 3.5 Processed Bytes 契约
1. `preprocess(...)` 产物是 backend 私有格式（对 runtime 不透明）。
2. Vulkan backend 通过 `VulkanDelegateHeader` + `VkGraph` identifier 校验与解析。
3. Delegate data 索引（inline/segment + index）由 execution plan 存储并在 init 取回。

---

## 4. 典型控制流（从导出到执行）

1. `exshader/export_llm.py` 调 `export_llama(...)`。
2. 模型转 Edge 后调用 `to_backend([VulkanPartitioner])`。
3. `backend_api` 根据 delegation tags 生成多个 lowered submodules。
4. 每个 submodule 调 `VulkanBackend.preprocess(...)` 生成一个 delegate blob。
5. 序列化为 `.pte`：ExecutionPlan 内含 delegates 列表、chains、DelegateCall 指令。
6. C++ 加载 `.pte`，`Method::init` 先逐个 init delegates，再解析 instructions。
7. `Method::execute` 遇到 `DelegateCall` 时，将预解析参数列表传入 Vulkan backend `execute(...)`。

---

## 5. “哪里改什么”规则

### 5.1 需要改 Python 的场景
1. 调整算子融合边界、分区策略、allow/blocklist。
2. 调整导出 pass 组合或 compile options。
3. 调整模型 loop/编排（prefill/decode 调度）逻辑。

### 5.2 需要改 C++ 的场景
1. Runtime ABI 语义、Method/delegate 生命周期语义变更。
2. Vulkan backend 执行语义、图构建、shader 执行机制变更。
3. 新 backend capability 需要 runtime/backend 协同支持。

### 5.3 不应混用的改法
1. 不用任务专用 C++ runner 去“绕过”导出/ABI问题。
2. 不用临时 fallback 掩盖 delegate 初始化或参数契约问题。
3. 不在 `AGENTS.md` 写临时调试结论。

---

## 6. 可观测性接口（架构资产）

这些是架构层长期可复用能力，不是一次性日志：
1. `ET_EXSHADER_ABI_DUMP_PATH`：导出侧 submodule/call_delegate 契约快照（JSONL）。
2. `ET_DELEGATE_ABI_TRACE_PATH`：runtime 侧 delegate init 与 DelegateCall 参数索引快照（JSONL）。
3. `exshader/diag/abi_diff.py`：baseline/candidate 结构化 ABI 对比工具。

---

## 7. 文档边界
1. `AGENTS.md`：架构、契约、规则、长期接口。
2. `DEBUG.md`：每次实验命令、结果、失败点、时间线。
