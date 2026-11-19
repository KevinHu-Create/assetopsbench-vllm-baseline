# 交付状态

截至 2026-11-19：**Colab T4 + Qwen2.5-7B-Instruct-AWQ 的真实候选基线与 CUDA profiling 已完成并归档；三项任务的完整共同配置仍待对齐。**

[完整实测报告、原始日志与 GPU trace](colab-t4-20261119/README.md)；[可复用候选配置](../configs/t4-colab-candidate.json)。

| 交付项 | 状态 | 证据 |
| --- | --- | --- |
| 任务一框架继承与协议兼容 | 完成 | 原提交历史与 8 项框架测试 |
| IBM 原题与官方 direct_llm prompt | 完成 | 固定 IBM commit、源字节 hash、请求 hash |
| vLLM GPU 服务 | 已实际验证 OFF 模式 | T4、AWQ/FP16、vLLM 0.34.0；ON 启动参数已实现，ON 性能未测 |
| TTFT、吞吐量、失败记录 | GPU 候选试跑完成 | 并发 1/4/8 各 30 次，传输失败 0/90 |
| 独立 PyTorch/CUDA profiling | 完成 | GPU rank trace 共 41204 事件，其中 2794 个 CUDA kernel 事件 |
| 本地与 CI 校验 | 通过 | 共 27 项测试；含候选验收、共享环境校验、缓存重置确认 |
| 任务二/三/四完整共同配置 | 待全组对齐 | 模型与 T4 已由用户指定，其余实际设置见候选 JSON |
| 原版完整 Agent/MCP 任务成功 | 未验证 | 本次使用官方无工具 direct_llm，答案正确率为 0 |
| APC OFF/ON 加速与行为比较 | 未执行 | 属于后续同一主动场景受控实验；不能由本表推断 |

## 真实候选结果

APC OFF；同一 GPU/软件/模型配置。每条件预热 3 个会话后确认缓存清空，再测 30 个会话；各条件仅一次 pilot，不等于完整项目的每条件 5 次独立重复。

| 并发 | TTFT p50 / p95（ms） | 输出 token/s | 成功 RPS | 传输失败 | 答对 |
| --- | --- | --- | --- | --- | --- |
| 1 | 194.383 / 200.677 | 10.130 | 3.377 | 0/30 | 0/30 |
| 4 | 414.248 / 714.551 | 14.728 | 4.909 | 0/30 | 0/30 |
| 8 | 1407.961 / 1450.238 | 15.463 | 5.154 | 0/30 | 0/30 |

模型没有接数据库或工单工具。90 次测量答案均为 `45`，官方 GT 为 `2`；无传输失败不代表原版 Agent 成功。这些数据只作为已明确边界的原版模型性能参考。

[本地验证记录](local-verification.json)。实测实现 commit `c5767da841b5e11aed0c10e271618c423263cb5e` 已通过 Python 3.11 / 3.12 / 3.13 的 [CPU verification](https://github.com/KevinHu-Create/assetopsbench-vllm-baseline/actions/runs/44609280039)；后续补强候选验收检查，未改动本次实测数据。

`python -m task2.check results/colab-t4-20261119/t4-trial --trial` 已返回 `candidate_trial complete=true`，包括真实 GPU、统一环境 ID、每组 30 次成功请求、usage、缓存重置、profiler 停止和 CUDA trace。默认正式验收会因团队配置尚未确认而拒绝候选结果；不能手改 manifest 使其通过。

## 平台与资源

GCP 课程结算与全局 GPU 配额已准备，但 5 个候选可用区的 T4 创建均返回库存不足，失败创建残留已清理并复查：实例、磁盘、路由器、公网 IP 均为 0。用户确认作业平台是 Google Colab 后，已在 Colab 完成上述真实测量；下载证据后释放了本次临时 T4 运行时，未新购 Colab 算力。

公开文件不含个人账户、券码、邮件或账单信息。配额检查可用 `scripts/gcp_preflight.py` 复跑；配额充足不保证实际库存。课程额度总额与单次实验的花费上限分别处理。
