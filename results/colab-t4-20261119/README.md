# Colab T4 / AWQ 实测候选基线

2026-11-19，在 Google Colab 的一张 Tesla T4 上完成真实模型推理、三个并发条件及独立 PyTorch/CUDA profiling。**这是待团队统一的候选配置试跑，不是已经确认的任务二/三/四共同正式实验。**

## 原版基线范围和答案质量

使用 IBM AssetOpsBench 固定版本 `scenario_1` 的官方 `direct_llm` prompt，选取“MAIN 站点共有多少工单”这一原题。只将官方 prompt 改成 SSE 传输以测 TTFT，未接 MCP、数据库或工具，也未把 GT 或工单表补进 prompt。

**90 个测量请求均完整返回，传输失败为 0；90 次答案都是 `45`，官方 GT 为 `2`，答案正确率为 0。** 这是原版模型性能参考，不能声称完整 Agent 任务成功，也不能把“无传输失败”等同于行为验证通过。该官方 direct_llm 模式明确要求模型在无数据时猜测，因此此处的错误率不单独证明 AWQ 量化导致质量下降。

## 未采集 profiler 的性能

每个并发条件先预热 3 个会话，确认前缀缓存重置成功，再测量 30 个相同原题会话；APC 均显式 OFF。每条件只有一次 pilot，30 个请求不是完整项目要求的 5 次独立重复。客户端在同一 Colab 主机，每请求使用一个新 HTTP 连接。

| 并发 | 完整请求 | TTFT p50 / p95（ms） | E2E p50（ms） | 输出 token/s | 成功 RPS | 答对 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 30/30 | 194.383 / 200.677 | 285.928 | 10.130 | 3.377 | 0/30 |
| 4 | 30/30 | 414.248 / 714.551 | 804.589 | 14.728 | 4.909 | 0/30 |
| 8 | 30/30 | 1407.961 / 1450.238 | 1528.305 | 15.463 | 5.154 | 0/30 |

每请求输入 245 tokens、输出 3 tokens。输出吞吐量只包含 completion tokens；输入加输出的 total token/s 另存于 JSON，不能混用。`cached_tokens` 在服务端 usage 中缺失，保留 null；本次没有做 APC ON 对照，不能据此声称前缀缓存加速。

并发 8 比并发 1 的总体吞吐量高，单请求 TTFT 也明显增加。这是该短输出、eager 配置及 Colab 运行时的结果，不能推广为模型长文本生成吞吐量或全 AssetOpsBench 表现。

## 可复用候选配置

| 项 | 实测设置 |
| --- | --- |
| GPU | 1 × Tesla T4，15360 MiB |
| 驱动 / CUDA runtime | 580.82.07 / 13.0 |
| 模型 | Qwen/Qwen2.5-7B-Instruct-AWQ |
| 模型与 tokenizer revision | b25037543e9394b818fdfca67ab2a00ecc7dd641 |
| 权重 / 计算 | AWQ 4-bit / FP16，TP=1 |
| vLLM / Torch | 0.34.0 / 2.14.0（运行时报告 2.14.0+cu130） |
| Python / Transformers / Tokenizers | 3.13.16 / 5.22.2 / 0.23.3 |
| Attention / 执行 | TRITON_ATTN / eager；量化计算日志为 MarlinLinearKernel |
| 上下文 / 调度 | max_model_len=4096，max_num_seqs=8，gpu_memory_utilization=0.9 |
| 采样 | temperature=0，seed=42，max_tokens=2048 |

完整配置见 [t4-candidate.json](t4-candidate.json)，依赖清单见 [environment.json](environment.json)。便于团队复用的同一文件也放在 [configs/t4-colab-candidate.json](../../configs/t4-colab-candidate.json)。其 `status=locked` 表示本次硬件/软件快照固定，`scope=candidate` 表示尚未确认全组采用；两者含义不同。任务三既有结果只有模型名相同，仍不能直接并入受控对比。

## Profiling 证据

正式计时结束后，另行预热并清空缓存，再采集仅 2 个请求的 trace；不把这次延迟填入性能表。GPU rank trace 共 41204 个事件，其中 CUDA kernel 事件 2794 个，包含真实 Marlin GEMM 等计算；API server trace 仅 7 个事件，不可拿它冒充 GPU 时间线。

- [GPU trace](t4-trial/traces/rank0.1795120660745096655.pt.trace.json.gz)：SHA256 `4d497b42f58fc1cfda75ab06af6210310b6b23d3199777020cff8ccc0fed29a2`。
- [GPU 事件摘要](t4-trial/trace-summary-1.json)、[原始 profiler 表](t4-trial/traces/rank0.profiler_out.txt)、[profiling 控制记录](t4-trial/profile/profile_control.json)。
- 可将 `.pt.trace.json.gz` 在 [Perfetto](https://ui.perfetto.dev/) 打开。CPU operator、CUDA runtime 与 GPU kernel 的 duration 可能相互包含或并行，不能加总成端到端墙钟时间；也不能仅凭 kernel 名推断 Prefill/Decode 占比。

## 核验与复跑

`task2.check --trial` 已核验三组真实 GPU 数据、统一环境 ID、每组至少 30 次成功请求、token usage、缓存重置确认、profiler 停止及含 CUDA kernel 的 trace，结果为 [candidate_trial complete](t4-trial/acceptance.json)。默认正式验收仍拒绝将这些候选结果作为已确认的团队基线，详见 [provenance.json](provenance.json)。

[Colab notebook](https://colab.research.google.com/github/KevinHu-Create/assetopsbench-vllm-baseline/blob/main/notebooks/task2_t4_awq_trial.ipynb) 固定已测实现；也可在匹配的 GPU 主机运行：

```bash
bash scripts/bootstrap_gpu.sh
.venv/bin/python scripts/run_gpu.py --trial \
  --team-profile configs/t4-colab-candidate.json --out outputs/new-t4-trial
.venv/bin/python -m task2.check outputs/new-t4-trial --trial
```

实测代码 commit 为 `c5767da841b5e11aed0c10e271618c423263cb5e`；候选验收入口在后续提交中补强，不改变本次实测数据。新主机若驱动、Python、Torch/CUDA 或其他记录的版本不符，会拒绝运行；不要为通过校验手改实际环境信息。优先在同一个 Colab 运行时依次运行任务二、三、四，避免平台更新驱动带来差异。

数据下载后已释放本次 Colab T4 运行时，未购买额外 Colab 算力。公开文件不含账户凭据、课程券码、邮件或账单资料。选定原题日志中的 IBM prompt 按 Apache-2.0 标注来源，见仓库 [THIRD_PARTY.md](../../THIRD_PARTY.md)。
