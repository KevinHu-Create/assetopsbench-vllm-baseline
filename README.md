# AssetOpsBench Task 2 · vLLM 性能基线

任务二的代码、复跑流程和测量协议。基于 [HaochengLu/proactive-assetops-bench](https://github.com/HaochengLu/proactive-assetops-bench) 的任务一框架扩展，保留其提交历史、`framework/` 和场景协议。

**当前状态：Colab T4 + AWQ 的真实候选试跑已完成：并发 1/4/8 各 30 个测量请求，传输失败为 0，并取得真实 CUDA trace。官方无工具 direct_llm 模式的答案正确率为 0；完整 Agent/MCP 任务成功尚未验证。** [实测结果与原始证据](results/colab-t4-20261119/README.md)。 最新交付状态见 [results/STATUS.md](results/STATUS.md)。

## 任务二做什么

在同一 GPU、模型、精度和推理设置下，选一个 IBM 原版任务，记录 TTFT、请求吞吐量、输出 token 吞吐量与失败情况，并单独采集可分析的 GPU trace。任务三、四随后复用这套服务环境；前缀缓存收益需要在**同一主动场景的 OFF/ON** 或相同固定请求上比较，不能拿原版与改写场景的差值代替缓存收益。

本仓库选择 IBM 官方 `scenario_1`：统计 MAIN 站点的全部工单。测量入口使用官方已有的 **direct_llm 模型基线**：原题与官方 prompt 保持一致，通过 AST 读取固定版本源码，再把相同 messages 发到 vLLM 流式接口以测 TTFT。**未运行官方整套 Agent/MCP 工具链；不向模型补入工单数据或 GT。** 本次 90 次测量均输出 `45`，GT 为 `2`，答案准确率为 0；准确率与性能分别报告。它适合任务二的原版性能参考；若团队要求“原版”必须是完整工具 Agent，需要另选官方 Agent runner，不能把当前结果当成完整 Agent 基线。

- 任务一固定提交：`98ee1d0a83f2ddab3449aac204da7bd4b4f64371`。
- IBM 固定提交：`0d044061ab3ffd4b213ddf6a0868c360d7e9667c`；不随 `main` 漂移。
- 模型：`Qwen/Qwen2.5-7B-Instruct-AWQ`，模型和 tokenizer revision 均固定。
- 预设引擎：`vllm==0.34.0`；AWQ 4-bit 权重、FP16 计算、TP=1、上下文 4096；TRITON_ATTN、eager 模式，详见 [configs/baseline.json](configs/baseline.json)。**已在 Colab T4 上验证该配置；完整运行参数仍是候选，待三项任务共同对齐。**

## 本地验证（不需要 GPU）

Python 3.11+，CPU 部分仅用标准库。macOS 的 Apple GPU 不作为本实验 NVIDIA/CUDA 环境。

```bash
python3 -m unittest discover -s tests -v
bash scripts/prepare_original.sh
python3 -m task2.smoke --out outputs/http-smoke
python3 -m task2.environment --out outputs/local-environment.json
python3 -m task2.serve --print-only
```

`task2.smoke` 启动明确标记的假 SSE 服务，固定返回测试内容；它只验证 HTTP、计时、日志与汇总流程，**不做模型推理**。生成的报告有 `measurement_kind: smoke`、`verified_gpu_run: false`，不能用于中期 GPU 性能表。

## GPU 上复跑

当前共同环境候选为 **1 × T4 16GB + Qwen2.5-7B-Instruct-AWQ**。任务二、三、四及 OFF/ON 必须保持 GPU 型号、显存、数量、模型 revision、量化/计算精度、软件版本和推理参数一致，不能按库存单独替换 GPU。[configs/team_environment.json](configs/team_environment.json) 尚为 `draft`，正式运行会拒绝启动；`task2.team` 会核对实际硬件和软件，所有结果带同一环境 hash。

GCP 可用 `python3 scripts/gcp_preflight.py --project YOUR_PROJECT_ID --gpu t4` 只读检查配额。若先做兼容性试跑，在 GPU 主机完成安装后捕获候选快照：

```bash
.venv/bin/python -m task2.team snapshot --environment outputs/environment.json \
  --out outputs/t4-candidate.json
.venv/bin/python scripts/run_gpu.py --trial --team-profile outputs/t4-candidate.json \
  --out outputs/t4-trial
```

本次已测配置见 [configs/t4-colab-candidate.json](configs/t4-colab-candidate.json)，驱动、Python、Torch/CUDA 等版本不匹配时会拒绝复用。快照只锁定本次实际环境，`scope=candidate` 不代表全组已同意。试跑输出标记 `gpu_compatibility_trial`，不能通过正式基线验收。正式实验需团队对齐后使用相同的 `status=locked`、`scope=team` 配置。
若使用 Colab，可打开 [T4/AWQ 兼容性试跑 notebook](https://colab.research.google.com/github/KevinHu-Create/assetopsbench-vllm-baseline/blob/main/notebooks/task2_t4_awq_trial.ipynb)。已完成真实 GPU 试跑，先选择 T4 运行时；当前 notebook 固定相同推理实现，并包含后续补强的候选验收检查。

在 GPU 主机仓库根目录：

```bash
bash scripts/bootstrap_gpu.sh
.venv/bin/python scripts/run_gpu.py --out outputs/gpu-baseline
.venv/bin/python -m task2.check outputs/gpu-baseline
```

脚本不创建云主机、不兑换优惠券。`bootstrap_gpu.sh` 安装固定 vLLM 版本与匹配的依赖，准备原题；`run_gpu.py` 启动独占服务、等待就绪，执行并发 1/4/8 的未开启采集的正式测量（各 30 个会话），再执行仅 2 个会话的独立 profiling，最后停止服务。正式计时排除模型加载、预热、metrics 抓取与 profiler flush。请在平台上另行停止/删除云资源，结束本地服务不会停止云账单。

完整复跑、指标口径和与任务三至五的交接见 [docs/measurement.md](docs/measurement.md)。

## 交付文件

| 文件 | 内容 |
| --- | --- |
| `outputs/gpu-baseline/baseline-c*/baseline.csv` / `.md` | 原版基线结果表 |
| `baseline-c*/run_config.json` | 完整配置、源码/请求 hash、GPU/驱动、依赖版本、服务启动参数 |
| `baseline-c*/run_log.jsonl` | 每次实际请求、原始输出、流式到达时间、token 使用、错误状态 |
| `baseline-c*/warmup_log.jsonl` | 单独保存、不计入正式统计的预热请求 |
| `baseline-c*/run_summary.json` | TTFT/E2E/TPOT 分位数、RPS、token/s、失败、格式与准确率 |
| `baseline-c*/metrics_*.prom` | vLLM 服务端指标快照 |
| `profile/`、`traces/`、`trace-summary-*.json` | 独立 profiling 记录、真实 CUDA trace 与事件摘要 |
| `server/`、`acceptance.json` | 服务日志与真实交付验收状态 |

真实输出默认不入 Git；核验后可将选定表格、配置、匿名化日志放入 `results/`。大型 trace 宜通过 GitHub Release 附件共享，保留 SHA256。不要上传凭据、课程邮件或额度码。

## 学 vLLM 从哪里开始

先看 [docs/vllm-learning.md](docs/vllm-learning.md)：从 Prefill/Decode、KV Cache、PagedAttention、Continuous Batching，逐步到 Automatic Prefix Caching 和 profiling。任务二让你学会把“感觉更快”变成可复现的测量。

保留的任务一说明在 [docs/task1-framework.md](docs/task1-framework.md)，原协议在 [spec.md](spec.md)。原有框架、场景和作者贡献均保留；新增代码集中在 `task2/`、`scripts/`、`configs/` 和任务二文档中。IBM 内容来源与使用范围见 [THIRD_PARTY.md](THIRD_PARTY.md)。
