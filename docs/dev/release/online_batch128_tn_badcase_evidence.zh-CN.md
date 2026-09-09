# Online Batch-128 / TN Badcase Evidence

验证日期：2026-09-09

## 目标 profile

使用 native-cursor fused engine 构建：

```text
batch=128
input=128
seq=512
TensorRT=10.13.3
BF16
```

构建成功，engine 包含 85 个 I/O tensor，cursor state/output 和 `codec0` 均存在。
Triton 也成功加载该 engine，`tts_orchestrator` 为 READY。

## 128 并发结果

使用 128 个并发 Triton gRPC 请求进行真实压测。服务在 decode 阶段发生 CUDA OOM：

```text
CUDA out of memory
GPU free: 39.31 MiB
Process memory: 29.46 GiB
engine-loop: torch.OutOfMemoryError
```

因此当前结论是：**batch=128 engine 构建通过，但 128 个同时活跃会话在当前
32GB RTX 5090 上运行未通过**。这不是可以忽略的测试噪声，线上 128 仍需通过显存
预算、KV/cache、并发 admission 或更大显存机器重新验收。

## TN 困难样例

命令：

```text
PYTHONPATH=. python tools/text_commitment_replay.py \
  --extra resources/dataset/badcase/tn_streaming_cases.txt \
  --mode token --fail-on-error --json
```

结果：

```text
cases=166
consistent=166
inconsistent=0
errors=0
```

输出保存在：
`workspace/validation/tn_streaming_cases_replay.json`

该结果证明困难样例在 FULL/TOKEN streaming commitment 层的 TN 结果一致；它不替代
音频 ASR 验收，也不掩盖 batch-128 的 OOM。
