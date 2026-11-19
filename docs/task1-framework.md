# Proactive AssetOpsBench（`v3.0`）— 主动运维评测与 vLLM 前缀缓存压测基础设施

面向 **IBM AssetOpsBench** 全场景适配的通用主动运维评测与系统压测基准框架（`schema_version: "3.0"`，100% 向下兼容 `v2.0`）。完整协议与字段契约请参阅 [spec.md](../spec.md)。

---

## 1. 项目概述与核心特性

- **双时钟与物理隔离（防未来泄露 / 防作弊）**：回放器 `framework.run` 与客户端严格只读公开场景 `scenarios/public/*.scenario.json`，在每个 `turn_idx` 按半开区间（`k=0` 时 `release_time <= tick[0]`；`k>=1` 时 `tick[k-1] < release_time <= tick[k]`）释放观测；所有工具通过 `bisect.bisect_right` 截断未来数据；隐藏真值 `scenarios/gt/*.gt.json` 仅由 `validate.py` 与离线评分器 `score.py` 读取。
- **100% 零剧透转化与静态自检**：`convert.py` 内置大小写不敏感黑名单 `FORBIDDEN_LABEL_COLUMNS`（强制剥离 `label`、`anomaly`、`is_anomaly`、`anomaly_label`、`fault`、`fault_type`、`fault_mode`、`gt`、`ground_truth`、`target`、`state_label`），且 `messages[0]` 只暴露 20 类故障代码名录（不含任何 `fmea_manuals` 阈值、症状、推荐工具或 `historical_workorders`）。
- **严格角色交替 + 字节级 Prefix-Cache 不变性**：全会话跨越多个 `turn_idx` 与单 tick 内多步 `step_idx` 始终维持 `system -> user -> assistant -> user ...` 交替并只做尾部追加（`messages_n[:-2] == messages_{n-1}` 100% 逐字节相等），原生适配 **vLLM Automatic Prefix Caching (APC)**。
- **纯 Python 3.10+ 标准库零依赖 & 超低内存**：无需 `pip install` 任何第三方库（无 `pandas`/`numpy`/`requests`/`torch`），本地默认 `--concurrency 1` 跑通内置示例任务峰值内存 `< 25 MB` RSS。
- **Keep-Alive 长连接与 7 个独立 CLI 闭环**：`VLLMClient` 基于标准库 `http.client` + `threading.local()` 维护持久连接（消除逐请求 TCP/TLS 建连抖动，支持 `[DONE]` 排空与断连自动重连）；提供从转换、校验、回放、评分、导出、并发轨迹压测到 A/B 对比报告的完整 7 CLI 工具链。

---

## 2. 团队角色快速上手导航

| 你负责的任务组 | 核心目标 | 入口文档与直达工具 |
| :--- | :--- | :--- |
| **任务 3 / 任务 4（场景与标注组）** | 适配 IBM AssetOpsBench（IoT / TSFM / FMSR / WorkOrder / 治理）新场景与 GT | 跳至 [第 5 节：任务 3/4 场景接入实战指南](#5-任务-3--任务-4-场景接入实战指南)（使用 `framework.convert` + `framework.validate` + 零改代码 `framework.score`） |
| **任务 2（GPU / vLLM 性能压测组）** | 运行 vLLM Cache OFF/ON、并发 `1/4/8`、采集 TTFT / RPS / `cached_tokens` | 跳至 [第 6 节：任务 2 vLLM 接入与并发压测指南](#6-任务-2gpu--vllm-组接入与并发压测指南)（使用 `framework.run --backend vllm` + `framework.export` + `framework.replay`） |
| **任务 5（评估与汇总分析组）** | 核查主动性 5 大指标、统计前缀复用率与生成 OFF vs ON 对比报告 | 跳至 [第 7 节：任务 5 核心产物与字段速查表](#7-任务-5评估与汇总组核心产物与字段速查表)（消费 `run_log.jsonl`、`run_summary.json`、`score_report.json`、`framework.compare`） |

---

## 3. 目录结构与 30 秒根目录 7 步复跑

### 3.1 仓库目录结构

```text
.
├── README.md                            # 本指南（7 步复跑、跨组接口与速查手册）
├── spec.md                              # v3.0 完整设计规范与 JSON Schema 契约
├── framework/
│   ├── __init__.py                      # SCHEMA_VERSION="3.0"、UTC 解析、多资产映射与递归子集匹配
│   ├── prompt.py                        # 20 类全局故障名录 + 零剧透交替 Prompt 构造
│   ├── client.py                        # 线程局部 Keep-Alive VLLMClient + 零硬编码 MockClient
│   ├── tools.py                         # 4 大专家域 10 个内置工具 + WorldState + custom_tools
│   ├── convert.py                       # CLI 1: wide/long CSV + 外部目录 + tick_grid -> scenarios/{public,gt}
│   ├── validate.py                      # CLI 2: 场景/GT 结构、引用完整性与零泄露静态校验 (exit 0/1)
│   ├── run.py                           # CLI 3: 场景目录/标签筛选 + 有界多步回放 -> run_log + run_summary
│   ├── score.py                         # CLI 4: 6 模式可组合确定性离线评分 -> score_report.json
│   ├── export.py                        # CLI 5: 流式导出 replay_requests.jsonl + exported_actions.jsonl
│   ├── replay.py                        # CLI 6: 跨会话并发固定轨迹重放压测 -> replay_benchmark_report.json
│   └── compare.py                       # CLI 7: Cache OFF vs ON 性能/缓存/一致性/质量对比报告
├── raw_data/
│   ├── task_01_chiller6/                # 内置示例 1：双冷机、标签剥离、计划切负荷消警、毛刺过滤
│   └── task_02_pump3/                   # 内置示例 2：双水泵、大写 Timestamp、多候选 FMEA、跨轮紧急停机审批
├── scenarios/                           # convert.py 预生成产物（任务 3/4 可直接替换）
│   ├── public/
│   └── gt/
└── tests/
    └── test_pipeline.py                 # 标准库 unittest（涵盖 v2.0 兼容 + AssetOpsBench Cases 1-7）
```

### 3.2 内置 2 个示例场景逐轮机制简介

- **`task_01_chiller6`（5 ticks `08:00`~`09:00`，无权限约束，含 `chiller_6` 与健康干扰机 `chiller_7`）**：
  - 原始 CSV 含 `anomaly_label`（转换时强制剥离），且 `historical_workorders` 含已关闭工单 `HIST-WO-001`；
  - `t0 (08:00)`：双机正常 -> `wait`；
  - `t1 (08:15)`：`chiller_6` 温度略升至 `2.7C`，但同轮附带 `maintenance_log`（`planned_load_shift: true`）-> 自动消警保持 `wait`；
  - `t2 (08:30)`：`chiller_7` 出现单点毛刺 `2.65C` -> `step 0` 调 `check_tsfm_anomaly` 得 `pattern="transient_spike"` -> `step 1` 保持 `wait`（零误报）；
  - `t3 (08:45, visible_from)`：`chiller_6` 连续越界（`08:40` `3.3C`、`08:45` `3.5C`）-> 同 tick 内 3 步闭环：`check_tsfm_anomaly`（`sustained_anomaly`）-> `lookup_fmea_manual`（解锁 3 条候选，匹配 `evaporator_waterside_fouling`）-> 上报需求并执行 `create_maintenance_workorder`（`WO-3-2`）；
  - `t4 (09:00)`：已建单结案 -> `wait`。
- **`task_02_pump3`（4 ticks `08:00`~`08:45`，含审批与禁令约束，含 `pump_3` 与干扰泵 `pump_4`）**：
  - 原始 CSV 使用大写 `Timestamp` 表头；权限配置 `approval_required_tools: ["create_urgent_shutdown_workorder"]`（`delay_turns: 1`）与 `forbidden_tools: ["disable_safety_interlock"]`；
  - `t0`、`t1`：双泵正常 -> `wait`；
  - `t2 (08:30, visible_from)`：`pump_3` 连续越界（`08:25` `7.4 mm/s`、`08:30` `7.8 mm/s`）-> 同 tick 内 `check_tsfm_anomaly` -> `lookup_fmea_manual` -> 上报 `bearing_wear_overheat` 并调 `submit_approval_request`（`effective_from_turn = 3`）；
  - `t3 (08:45)`：审批在本轮生效 -> 直接执行 `create_urgent_shutdown_workorder`（`WO-3-0`）。

### 3.3 根目录 7 步完整复跑命令

在仓库根目录依次执行（总耗时 `< 1s`）：

```bash
# 1. 转换原始数据为隔离的公开场景与隐藏 GT
python3 -m framework.convert --raw-dir raw_data --out-dir scenarios

# 2. 静态校验全部公开场景与 GT 的结构、引用一致性与零泄露
python3 -m framework.validate --scenarios-dir scenarios/public --gt-dir scenarios/gt

# 3. 有界多步主动回放（默认 --max-steps-per-tick 3 --concurrency 1）
python3 -m framework.run \
  --scenarios-dir scenarios/public \
  --backend mock \
  --cache-mode off \
  --max-steps-per-tick 3 \
  --concurrency 1 \
  --out-dir outputs/demo_off

# 4. 独立读取隐藏 GT 进行确定性离线评分
python3 -m framework.score \
  --scenarios-dir scenarios/public \
  --gt-dir scenarios/gt \
  --log outputs/demo_off/run_log.jsonl \
  --out outputs/demo_off/score_report.json

# 5. 流式导出固定重放请求轨迹（供任务 2 压测）与精简动作流（供任务 5 核查）
python3 -m framework.export \
  --log outputs/demo_off/run_log.jsonl \
  --out-requests outputs/demo_off/replay_requests.jsonl \
  --out-actions outputs/demo_off/exported_actions.jsonl

# 6. 跨会话并发固定轨迹重放压测
python3 -m framework.replay \
  --requests outputs/demo_off/replay_requests.jsonl \
  --backend mock \
  --cache-mode off \
  --concurrency 2 \
  --warmup-sessions 1 \
  --repeats 2 \
  --out outputs/demo_off/replay_benchmark_report.json

# 7. 生成对比报告（JSON + Markdown；此处用同组数据演示格式，真实实验传 off/on 两组产物）
python3 -m framework.compare \
  --off-summary outputs/demo_off/run_summary.json \
  --on-summary outputs/demo_off/run_summary.json \
  --off-log outputs/demo_off/run_log.jsonl \
  --on-log outputs/demo_off/run_log.jsonl \
  --off-score outputs/demo_off/score_report.json \
  --on-score outputs/demo_off/score_report.json \
  --off-replay outputs/demo_off/replay_benchmark_report.json \
  --on-replay outputs/demo_off/replay_benchmark_report.json \
  --out outputs/demo_off/comparison_report.json \
  --out-md outputs/demo_off/comparison_report.md
```

运行全量单元测试（覆盖 v2.0 兼容与 AssetOpsBench Cases 1-7）：

```bash
python3 -m unittest discover -s tests -v
```

---

## 4. AssetOpsBench 4 大专家域 10 个内置工具 + `custom_tools` 速查表

所有数据查询工具均严格截断至 `release_time <= current_replay_time`（工单按 `created_at <= current_replay_time` 截断）；各工具与 `decision` 强绑定（不匹配报 `tool_error` 或 `unauthorized`）：

| 专家域 | 工具名 | 绑定 `decision` | 入参 `args` | 功能与状态影响说明 |
| :--- | :--- | :--- | :--- | :--- |
| **IoT** | `query_iot_metadata` | `query` | `{"asset_id": str \| null}` | 返回指定设备或全场景设备拓扑（`sensors`、`sensor_units`、`normal_ranges`）及当前 `world_state["assets"]`。 |
| **IoT** | `query_iot_history` | `query` | `{"asset_id": str, "limit": int=5, "modality": str="all", "metrics": list[str] \| null, "start_time": str \| null, "aggregation": "raw\|mean\|min\|max\|latest"}` | `bisect_right` 检索历史观测（支持四类模态 `telemetry`/`alarm`/`maintenance_log`/`workorder_history`、列投影 `metrics`、`start_time` 窗口与统计聚合）。 |
| **TSFM** | `check_tsfm_anomaly` | `investigate` | `{"asset_id": str, "metric": str, "window": int=8}` | 计算最近 `window` 点的 `mean`、总体 `std`、`z_score`、OLS `slope_per_min`、`consecutive_breaches` 与 `pattern`（`nominal`/`transient_spike`/`sustained_anomaly`）。 |
| **TSFM** | `forecast_tsfm_metric` | `investigate` | `{"asset_id": str, "metric": str, "horizon_minutes": float=30.0, "window": int=8}` | 纯标准库 OLS 趋势预测，返回 `predicted_value`、`residual_std`、95% `prediction_band`、`predicted_breach` 与 `estimated_minutes_to_breach`。 |
| **FMSR** | `lookup_fmea_manual` | `investigate` | `{"asset_type": str \| null, "keyword": str \| null, "fault_type": str \| null}` | 检索 `fmea_manuals`：支持按 `asset_type`、`fault_type` 及 `keyword` 多词元检索（同时扫描结构化字段与 `failure_mode`/`symptoms`/`remediation`/`description` 自由文本）。 |
| **WorkOrder** | `query_workorder_history` | `query` | `{"asset_id": str \| null, "status": "all\|open\|in_progress\|closed", "limit": int=10}` | 合并已释放 `historical_workorders` 与会话 `world_state["workorders"]`，用于新开单前排查已有未结工单。 |
| **WorkOrder** | `update_workorder` | `execute` | `{"workorder_id": str, "status": "open\|in_progress\|closed", "note": str, "priority": str}` | 更新未关闭工单的状态/优先级并追加备注（用于工单去重提级），拒绝修改 `"closed"` 工单。 |
| **WorkOrder** | `create_maintenance_workorder` | `execute` | `{"asset_id": str, "fault_type": str, "summary": str}` | 创建常规工单 `WO-<turn_idx>-<step_idx>`（`priority="normal"`, `status="open"`）并更新 `world_state`。 |
| **WorkOrder** | `create_urgent_shutdown_workorder` | `execute` | `{"asset_id": str, "fault_type": str, "summary": str}` | 权限与审批通过后创建紧急停机工单，并将该设备 `world_state` 置为 `status="shutdown_scheduled"`, `urgent_shutdown=true`。 |
| **Governance** | `submit_approval_request` | `request_approval` | `{"asset_id": str, "target_tool": str, "reason": str}` | 校验 `forbidden_tools` 与 `permissions.approval_policy`（`delay_turns`、`auto_approve`、`required_world_state` 前置状态），通过后记录 `effective_from_turn`。 |
| **Custom** | `scenario["custom_tools"]` | 自定义 (`query\|investigate\|request_approval\|execute`) | 由 `required_args` 定义 | 声明式控制/准备工具（如 `start_standby_pump`、`set_damper_position`）：校验权限并按 `state_effects`（支持 `"$args.<key>"` 参数插值）更新 `world_state["assets"]`；若 `"terminal": false` 允许同 tick 继续执行下一步。 |

---

## 5. 任务 3 / 任务 4 场景接入实战指南

### 5.1 两种场景接入方式（零改框架代码）

- **方式 A（直接提供 JSON）**：按 `spec.md` 第 3.3 / 3.5 节格式，直接将文件放入 `scenarios/public/<scenario_id>.scenario.json` 和 `scenarios/gt/<scenario_id>.gt.json`。
- **方式 B（从原始 CSV + 描述一键转换，推荐）**：在 `raw_data/<scenario_id>/` 下放入 `telemetry.csv`、`raw_task_spec.json` 以及可选的 `fmea_manuals.json`、`workorders.json`，运行 `python3 -m framework.convert --raw-dir raw_data --out-dir scenarios`：
  - **宽表 / 窄表自适应**：在 `convert_config` 中设置 `"csv_format": "wide"`（默认）或 `"long"`（窄表自动按 `(release_time, asset_id, source_type, source_ref)` 聚合为单条多指标观测）；支持 `column_mapping`、`strip_columns`、`sensor_columns` 白名单（始终低于黑名单 `FORBIDDEN_LABEL_COLUMNS` 优先级）。
  - **自动 `tick_grid`**：若 `public_spec.replay_ticks` 省略，可配置 `"tick_grid": {"auto_from_observations": true, "step_seconds": 900}` 或指定 `{"start": "...", "end": "...", "step_seconds": 900}` 自动生成等步长 UTC 时间轴。
  - **自动回填 GT 引用**：`gt_spec.needs[*]` 中支持写 `"supporting_timestamps": [...]`，转换时自动解析映射为对应 `obs_id` 列表。

### 5.2 6 种可组合 `completion_check` 验收模式（直接复制使用）

1. **`"tool_call"`（默认，兼容 v2.0，支持省略 `"mode"`）**：
   ```json
   {
     "mode": "tool_call",
     "required_tool": "create_maintenance_workorder",
     "required_args_subset": {"asset_id": "chiller_6"},
     "allow_pending_if_investigated": true
   }
   ```
2. **`"any_of"`（多路径任一达成即可，如允许“更新已有工单”或“新建工单”）**：
   ```json
   {
     "mode": "any_of",
     "checks": [
       {"mode": "tool_call", "required_tool": "update_workorder", "required_args_subset": {"workorder_id": "WO-OPEN-77"}},
       {"mode": "tool_call", "required_tool": "create_maintenance_workorder", "required_args_subset": {"asset_id": "ahu_5"}}
     ]
   }
   ```
3. **`"all_of"`（必须同时满足全部子检查项）**：
   ```json
   {
     "mode": "all_of",
     "checks": [
       {"mode": "tool_call", "required_tool": "lookup_fmea_manual", "required_args_subset": {"asset_type": "centrifugal_pump"}},
       {"mode": "tool_call", "required_tool": "create_urgent_shutdown_workorder", "required_args_subset": {"asset_id": "pump_3"}}
     ]
   }
   ```
4. **`"ordered_sequence"`（要求多个已授权动作按严格先后顺序完成）**：
   ```json
   {
     "mode": "ordered_sequence",
     "steps": [
       {"tool_name": "start_standby_pump", "args_subset": {"asset_id": "pump_standby"}},
       {"tool_name": "create_urgent_shutdown_workorder", "args_subset": {"asset_id": "pump_3"}}
     ]
   }
   ```
5. **`"world_state"`（校验回放结束时 `WorldState` 终态子集匹配，兼容辅助执行器改状态）**：
   ```json
   {
     "mode": "world_state",
     "expected_state_subset": {
       "assets": {"damper_1": {"pos_pct": 100, "status": "open"}}
     }
   }
   ```
6. **`"diagnosis_only"`（纯监测/预测/诊断场景：要求准确上报并引用关键证据，且严禁发起任何 `execute`）**：
   ```json
   {
     "mode": "diagnosis_only"
   }
   ```
   > 提示：可在 `gt.json` 顶层设置 `"allow_readonly_query_in_no_action_ticks": true`，允许智能体在静默轮执行只读 `query` 而不计入 `in_no_action_tick` 或 `"early"`。

### 5.3 提交场景前必跑静态校验

```bash
python3 -m framework.validate --scenarios-dir scenarios/public --gt-dir scenarios/gt
```

校验通过返回退出码 `0`（`"valid": true`）；若存在剧透列、`messages[0]` 阈值泄露、非递增 `replay_ticks`、孤立 `obs_id`、无效工具名/权限或损坏的 `completion_check`，立即精准报错并返回退出码 `1`。

---

## 6. 任务 2（GPU / vLLM 组）接入与并发压测指南

### 6.1 在线接入 vLLM 运行完整工作流（`framework.run`）

支持 `--scenarios` 或 `--scenarios-dir`（配合可选 `--tags <tag1> <tag2>` 按场景标签筛选）：

```bash
# Cache OFF 在线闭环运行
python3 -m framework.run \
  --scenarios-dir scenarios/public \
  --backend vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --cache-mode off \
  --max-steps-per-tick 3 \
  --concurrency 1 \
  --gpu-tag A100-80G \
  --dtype bfloat16 \
  --out-dir outputs/vllm_live_off

# Cache ON 在线闭环运行（由 vLLM 服务端开启 --enable-prefix-caching；客户端 Prompt 100% 保持一致）
python3 -m framework.run \
  --scenarios-dir scenarios/public \
  --backend vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --cache-mode on \
  --max-steps-per-tick 3 \
  --concurrency 1 \
  --gpu-tag A100-80G \
  --dtype bfloat16 \
  --out-dir outputs/vllm_live_on
```

> 说明：`--cache-mode {off,on,na}` 仅作为实验分组元数据写入日志，**绝不会修改发送给 vLLM 的 Prompt 内容**。

### 6.2 固定请求轨迹并发 `1 / 4 / 8` 重放压测（`framework.export` + `framework.replay`）

为了在消除模型随机输出长度方差的前提下做受控系统压测，先导出 `replay_requests.jsonl`，再用 `framework.replay` 执行预热 + 多轮重复压测：

```bash
# 1. 导出固定请求轨迹
python3 -m framework.export \
  --log outputs/vllm_live_off/run_log.jsonl \
  --out-requests outputs/vllm_live_off/replay_requests.jsonl \
  --out-actions outputs/vllm_live_off/exported_actions.jsonl

# 2. 并发重放压测（示例：concurrency=4, 预热 2 个会话, 正式重复 5 轮）
python3 -m framework.replay \
  --requests outputs/vllm_live_off/replay_requests.jsonl \
  --backend vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --cache-mode on \
  --concurrency 4 \
  --warmup-sessions 2 \
  --repeats 5 \
  --gpu-tag A100-80G \
  --dtype bfloat16 \
  --out outputs/replay_on_c4.json
```

- **调度与缓存机制**：`replay.py` 跨 `session_id` 使用 `ThreadPoolExecutor(max_workers=concurrency)` 并发派发，但**同一 `session_id` 内部严格阻塞等待前一条 `seq_idx` 完成后才发送下一条**，确保与真实多轮对话完全一致地填充并命中 vLLM KV Cache。
- **Keep-Alive 防抖动**：`VLLMClient` 基于 `http.client` + `threading.local()` 为每个工作线程复用持久 TCP/TLS 连接，防止逐请求三次握手污染 `ttft_ms`（若服务端主动断开空闲连接则自动重建并重发 1 次）。

---

## 7. 任务 5（评估与汇总组）核心产物与字段速查表

| 产物文件 | 生成 CLI | 核心字段与用途 |
| :--- | :--- | :--- |
| **`run_log.jsonl`** | `framework.run` | 每次模型请求一行：含 `turn_idx`、`step_idx`、`seq_idx`、`request_id`、`prev_request_id`、`new_obs_ids`、`visible_obs_ids`、`llm_request_messages`、`raw_response`、`parse_ok`、`action`、`tool_result`、`timing_ms`（`ttft_ms`/`latency_ms`/`tool_ms`/`turn_wall_ms`）、`usage`（含 `cached_tokens`，缺失为 `null`）与 `error.status`（`ok`/`api_timeout`/`api_error`/`invalid_action_json`/`tool_error`）。 |
| **`exported_actions.jsonl`** | `framework.export` | 剥离大体积 `llm_request_messages` 后的轻量执行流，便于快速人工或脚本审阅每步观测、决策与工具结果。 |
| **`run_summary.json`** | `framework.run` | 含 `experiment_meta`（`max_steps_per_tick`/`concurrency`/`gpu_tag`/`dtype`/`repeat_idx`）、`median_ttft_ms`、`p95_ttft_ms`、`median_latency_ms`、`p95_latency_ms`、`successful_rps`、`usage_totals`、`prefix_reuse_stats`（`total_prompt_bytes`、`intra_session_shared_prefix_bytes`、`intra_session_reuse_ratio`、`global_system_prefix_bytes`）与 `per_scenario[*].final_world_state`。 |
| **`score_report.json`** | `framework.score` | 5 大主动性指标 + API 失败分离统计：`need_discovery`（`tp`/`fp`/`fn`/`duplicate_need_reports`/`precision`/`recall`/`f1`）、`unnecessary_interventions`（按 `turn_idx` 去重与 3 类 `breakdown`）、`timely_valid_interventions`（`early`/`timely`/`late`/`missed`）、`verified_task_success`（`verified`/`pending_verification`/`failed`）、`unauthorized_attempts`（无约束场景返回 `status: "not_covered"`）及 `overall` 汇总。 |
| **`replay_benchmark_report.json`** | `framework.replay` | 受控压测汇总：`trajectory_stats`、`aggregate`（`total_timed_requests`、`mean_repeat_wall_ms`、`median_ttft_ms`、`p95_ttft_ms`、`mean_session_wall_ms`、`p95_session_wall_ms`、`successful_rps`、`usage_totals.cache_hit_rate`）及逐轮 `repeats_detail`。 |
| **`comparison_report.{json,md}`** | `framework.compare` | 一键对齐 Cache OFF vs ON 两组实验，自动输出 `latency_and_throughput`（TTFT/延迟降幅百分比、RPS 加速比）、`cache_efficiency`（实测缓存命中率 vs 理论前缀复用率）、`trajectory_determinism`（`exact_request_match_rate`、`action_agreement_rate`）、`behavioral_parity`（F1 与 Verified 不退化校验）及 `hypothesis_evaluation.overall_hypothesis_pass`。 |

一键生成 Cache OFF vs ON 对照报告示例：

```bash
python3 -m framework.compare \
  --off-summary outputs/vllm_live_off/run_summary.json \
  --on-summary outputs/vllm_live_on/run_summary.json \
  --off-log outputs/vllm_live_off/run_log.jsonl \
  --on-log outputs/vllm_live_on/run_log.jsonl \
  --off-score outputs/vllm_live_off/score_report.json \
  --on-score outputs/vllm_live_on/score_report.json \
  --off-replay outputs/replay_off_c4.json \
  --on-replay outputs/replay_on_c4.json \
  --target-ttft-reduction-pct 20.0 \
  --out outputs/comparison_report.json \
  --out-md outputs/comparison_report.md
```
