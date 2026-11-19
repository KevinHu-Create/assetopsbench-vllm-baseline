# 测量协议与跨组交接

## 实验边界

任务二先做原版模型性能参考，再准备后续主动场景共用的 vLLM 环境。当前选取 IBM 的原始 `scenario_1` 与官方 `direct_llm` prompt。它没有工具调用，也没有读取工单表的能力，因此不据此声称完成完整运维流程。`task2.prepare` 校验固定 Git commit 和三个源文件的已提交字节，记录 SHA256；GT 只供响应返回后的离线精确比对。

相比官方 direct_llm，测量适配器改变的是传输方式（非流式改为 SSE）、显式固定生成上限/seed，以及记录性能。未执行官方 runner 的 OTel、清理答案和持久化代码；准确率使用原始最终文本去除两端空白后的严格比较，避免宽松清理掩盖格式问题。正式配置的 `max_tokens=2048` 与该官方 backend 的默认上限对齐。

## 手动运行与 profiling

所有命令在仓库根目录。第一终端在 GPU 主机启动：

```bash
.venv/bin/python -m task2.serve \
  --config configs/baseline.json \
  --out outputs/server-off \
  --profile-dir outputs/traces-off
```

第二终端运行正式计时：

```bash
.venv/bin/python -m task2.benchmark \
  --requests workloads/original/requests.jsonl \
  --gt workloads/original/gt.json \
  --config configs/baseline.json \
  --server-manifest outputs/server-off/manifest.json \
  --out outputs/original-c1
```

Profiler 功能已配置但没有开始采集。`--profile` 会在预热完成后调用 `/start_profile`，在 `finally` 中调用 `/stop_profile`；采集结果标成 `measurement_kind: profile`。用独立的短配置限制 trace 体积：

```bash
python3 - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path('configs/baseline.json').read_text())
cfg.update(warmup_sessions=1, measured_sessions=2, concurrency=1)
Path('outputs/profile-config.json').write_text(json.dumps(cfg, indent=2))
PY
.venv/bin/python -m task2.benchmark \
  --requests workloads/original/requests.jsonl \
  --config outputs/profile-config.json \
  --server-manifest outputs/server-off/manifest.json \
  --profile --out outputs/original-profile
```

vLLM 0.34.0 使用 `--profiler-config`。较早版本的 `VLLM_TORCH_PROFILER_DIR` 不能无条件套用到新版。Trace 保存在**服务端 GPU 主机**，不是远程客户端。用 [Perfetto](https://ui.perfetto.dev/) 查看；汇总命令：

```bash
python3 -m task2.trace_summary /absolute/path/to/rank0.pt.trace.json.gz \
  --out outputs/trace-summary.json
```

CPU operator、CUDA runtime、GPU kernel 的 duration 可能相互包含或并行，不能把这些时间相加当总耗时，也不能只凭 kernel 名字断言是 Prefill 或 Decode。摘要保留事件分类和源 trace hash，细分阶段须结合原始时间线与请求标识。Profiler 有开销，profile 运行不能填入正式延迟表。

## 指标定义

| 指标 | 本仓库口径 |
| --- | --- |
| TTFT | 客户端 `perf_counter_ns()`：发出 HTTP 请求前到首个非空生成 delta（含 reasoning）；角色事件与 usage 事件不算 token |
| first_content_ms | 第一个非空可见正文 delta；与 reasoning 首 token 分开 |
| E2E latency | 请求发出前到最终 SSE 事件处理结束，包含网络与客户端解析；不等同 GPU kernel 时间 |
| TPOT | `(最后生成 delta 到达时间 - TTFT)/(completion_tokens-1)`；输出不足 2 tokens 时为空，SSE 合并会限制其精度 |
| successful RPS | 传输完整成功请求数 / 正式阶段墙钟秒数；墙钟包含客户端调度与日志写入开销 |
| output token/s | 完整成功请求的 completion_tokens 总和 / 同一正式阶段墙钟秒数 |
| total token/s | 完整成功请求的 prompt+completion tokens / 同一墙钟秒数；与 output token/s 分列 |
| p50 / p95 | 成功请求样本的中位数 / nearest-rank 95 分位；失败请求不混入成功延迟分布 |
| failure | HTTP、超时、流损坏、提前 EOF、空输出分别统计；无自动重试，避免隐藏失败 |
| 截断 / 格式 / 答案 | `finish_reason=length`、格式错误、GT 精确比对分别记录；不等同 HTTP 失败 |
| cached_tokens | 只读服务端 usage；缺失为 null。部分 usage 缺失时不计算不完整总量或缓存命中率 |

每请求明确使用新连接并把建连计入 TTFT，与任务一 Keep-Alive 客户端有差别。跨原版和主动场景对比时统一用 `task2.benchmark` 固定轨迹入口，或保持同一传输实现并注明。一个 SSE chunk 可能含多个 tokens，绝不使用 chunk 数冒充 token 数。

每次正式运行先预热 3 个会话，单独存日志，随后测 30 个会话。并发表示同时在途的**会话**上限；每个会话内请求按 seq_idx 阻塞顺序发送。相同短题的 30 次重复只能给出这个工作负载的初步性能，不能推断全 benchmark 的表现。预热会填充 APC，因此必须说明服务启动状态与 warmup 条件；原版参考默认显式 OFF。

## 与任务三、四、五对齐

1. 共用固定的 GPU 型号/数量、驱动、vLLM 版本、模型/Tokenizer SHA、dtype、TP、上下文上限与 sampling 设置。
2. 同一 GPU 正式测量时段只允许一个实验；暂停其他请求。先留存服务日志、运行配置和 GPU 型号，避免仅手写 `--gpu-tag A100`。
3. 任务一 `--cache-mode on/off` 只写元数据，真实开关由服务启动参数控制；切换 OFF/ON 必须停旧服务并用对应 JSON 配置重启。
4. `framework.run --backend vllm` 采集真实场景闭环，`framework.export` 导出请求。任务二的固定重放入口可以读取其 `replay_requests.jsonl`，但 sampling 配置必须与轨迹中的 `inference_config` 一致。原日志没有 `scenario_id` 时汇总该字段为 unknown，保留 session_id 与实际 messages 仍可追溯。
5. 闭环 OFF/ON 可能走不同分支；行为质量仍交给 `framework.score`，性能比较注明实际请求内容、数量、生成长度是否变化。任务二的重放只重发固定请求，不会重新执行工具。
6. 固定请求公平比较时，OFF/ON 输入请求文件的 hash、顺序、采样参数、预热和重复数应一致。缓存开关不能更改 prompt，不要为了提高命中率额外注入答案或未来观测。

生成 ON 配置（仅修改 server cache mode），启动前先停止 OFF 服务：

```bash
python3 - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path('configs/baseline.json').read_text())
cfg['cache_mode'] = 'on'
Path('configs/cache-on.json').write_text(json.dumps(cfg, indent=2))
PY
python3 -m task2.serve --config configs/cache-on.json --out outputs/server-on
```

`task2.benchmark` 校验 ready GPU manifest、服务端版本和模型列表，并复制服务配置进结果。manifest 是操作记录，不是硬件远程证明；不要手动编造或复用其他 GPU 的 manifest。`task2.check` 仅在三组未采集 profiling 的真实 GPU 测量、各至少 30 个成功请求、token 统计和含 CUDA kernel 的 trace 都存在时返回成功。

参考：[vLLM 服务参数](https://docs.vllm.ai/en/stable/cli/serve/)、[Profiling](https://docs.vllm.ai/en/latest/contributing/profiling/)、[APC](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)。
