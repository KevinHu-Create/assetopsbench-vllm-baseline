# 交付状态

截至 2026-11-19：**代码与本地验证完成，全局 GPU 配额已批准，T4 + AWQ 试跑受云平台库存阻塞；团队完整配置尚未确认。**

| 交付项 | 状态 | 证据 |
| --- | --- | --- |
| 任务一框架继承与协议兼容 | 完成 | 原提交历史与 8 项框架测试 |
| IBM 原题与官方 direct_llm prompt 准备 | 完成 | 固定 commit、逐文件 hash、请求 hash；生成命令见 README |
| vLLM 服务配置与 OFF/ON 开关 | 已实现，待 GPU 启动验证 | `configs/baseline.json`、`task2.serve` |
| SSE TTFT、吞吐量、失败测量 | 本地协议验证通过 | 新增 16 项测试（共 24 项）；无 GPU 推理 |
| 真实原版基线表 | 待执行 | 无实测数字 |
| 真实 PyTorch/CUDA trace | 待执行 | 无真实 trace |
| 复跑与 vLLM 学习资料 | 完成 | README、测量协议、学习指南 |

本地验证记录见 [local-verification.json](local-verification.json)。GitHub Actions 已在 Python 3.11 / 3.12 / 3.13 上通过 [CPU verification](https://github.com/KevinHu-Create/assetopsbench-vllm-baseline/actions/runs/44605275877)。

## GPU 平台检查

独立实验项目已关联课程结算。全局 GPU 配额已批准为 1，候选区域 T4 配额各为 1。先前 L4 创建因库存不足失败，无成功 GPU 测量。根据用户补充，现尝试 T4 16GB + Qwen2.5-7B-Instruct-AWQ，并记录为候选兼容性试跑；vLLM 等剩余参数待全组同步。已尝试 us-central1-a/c/f、us-east1-c、us-west1-b，均返回 T4 库存不足。us-central1-f 的失败创建一度进入回滚状态，已清理并复查：实例、磁盘、路由器、公网 IP 均为 0。仍无真实 GPU 推理或 trace。Colab notebook 已准备并检查 Python 语法，但尚未执行。配额不保证实际库存。个人账户、券码和账单信息不公开。

## 正式结果表

| 工作负载 | GPU | dtype | APC | 并发 | TTFT p50/p95 | output token/s | RPS | 失败 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IBM scenario_1 / official direct_llm prompt | T4 16GB 候选 | AWQ / FP16 候选 | OFF | 1 | 未测 | 未测 | 未测 | 未测 |
| 同上 | T4 16GB 候选 | 同上 | OFF | 4 | 未测 | 未测 | 未测 | 未测 |
| 同上 | T4 16GB 候选 | 同上 | OFF | 8 | 未测 | 未测 | 未测 | 未测 |

本地 HTTP fixture 的延迟来自测试服务，不能填入本表。`python -m task2.check outputs/gpu-baseline` 在缺乏真实证据时会返回非零退出码，防止误报任务已经全部完成。
