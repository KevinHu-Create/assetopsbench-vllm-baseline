# 交付状态

截至 2026-11-19：**代码与本地验证完成，GPU 实验待访问权限/算力确认。**

| 交付项 | 状态 | 证据 |
| --- | --- | --- |
| 任务一框架继承与协议兼容 | 完成 | 原提交历史与 8 项框架测试 |
| IBM 原题与官方 direct_llm prompt 准备 | 完成 | 固定 commit、逐文件 hash、请求 hash；生成命令见 README |
| vLLM 服务配置与 OFF/ON 开关 | 已实现，待 GPU 启动验证 | `configs/baseline.json`、`task2.serve` |
| SSE TTFT、吞吐量、失败测量 | 本地协议验证通过 | 新增 10 项测试；无 GPU 推理 |
| 真实原版基线表 | 待执行 | 无实测数字 |
| 真实 PyTorch/CUDA trace | 待执行 | 无真实 trace |
| 复跑与 vLLM 学习资料 | 完成 | README、测量协议、学习指南 |

## 正式结果表

| 工作负载 | GPU | dtype | APC | 并发 | TTFT p50/p95 | output token/s | RPS | 失败 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IBM scenario_1 / official direct_llm prompt | 待确认 | BF16 预设 | OFF | 1 | 未测 | 未测 | 未测 | 未测 |
| 同上 | 待确认 | 同上 | OFF | 4 | 未测 | 未测 | 未测 | 未测 |
| 同上 | 待确认 | 同上 | OFF | 8 | 未测 | 未测 | 未测 | 未测 |

本地 HTTP fixture 的延迟来自测试服务，不能填入本表。`python -m task2.check outputs/gpu-baseline` 在缺乏真实证据时会返回非零退出码，防止误报任务已经全部完成。
