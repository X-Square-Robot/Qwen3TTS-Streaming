# Batch-128 Native Cursor A/B Evidence

验证日期：2026-09-09

## 固定条件

- GPU：RTX 5090 32GB
- TensorRT：10.13.3
- dtype：BF16；RoPE/指定稳定层按 manifest 固定 FP32
- profile：`batch=128, input=128, seq=512`
- runtime：同一 Triton 25.10 镜像、同一 engine runtime、`MAX_BATCH_SLOTS=128`
- 请求：128 个同时活跃 Triton gRPC session

## A/B 结果

### Standard engine

```text
I/O tensors: 62
idle nvidia-smi: 30446 MiB used, 1664 MiB free
128/128 requests: PASS
wall: 3032.5 ms
```

### Native-cursor engine

```text
I/O tensors: 85
idle nvidia-smi: approximately 32000 MiB used
128 concurrent requests: CUDA OOM during decode
```

失败时日志：

```text
GPU free: 39.31 MiB
Process memory: 29.46 GiB
torch allocated: 11.27 GiB
engine-loop: torch.OutOfMemoryError
```

## 定位结果

cursor head 本身约 2M 参数，不能用参数量推断 fused graph 的显存增量。两个 plan 的
engine 文件大小只相差约 3MB，主要差异来自 TRT execution context/profile，而不是
cursor 权重：

```text
                         standard       native-cursor
profile 0 device memory  7015.8 MiB      7098.3 MiB
profile 1 device memory  1164.6 MiB      1277.2 MiB
```

standard 默认使用 decode-only profile 1；cursor 当时被 runtime 的 parity gate 强制使用
profile 0，所以支付了约 7.1GiB 的 prefill+decode context。cursor 的 profile 1 不是“没有
构建”，而是尚未取得当前实现要求的递归轨迹验收，因此当时没有自动启用。

同时发现并修复了一个独立的生命周期 bug：prefill 调用虽然将 `cursor_active=0`，旧图
仍会更新 cursor 的 `conv_history`、`last_trunk_input` 和 `seen_frames`，executor 还会
把这些无意义输出写回 slot。现在 inactive cursor 是完整 state passthrough，prefill 不再
提交 cursor 状态；第一帧 active decode 才开始推进 cursor。

## 低显存路径复验

在同一 RTX 5090、同一 `128/128/512` 配置下，强制 decode-only profile 1 做了真实
executor 压测：128 个 slot 逐个 prefill 后执行一次 batch-128 decode，成功完成；decode
后显存约 `29606 MiB used / 2502 MiB free`，未发生 OOM。随后用包含最新 runtime 的
Triton 镜像做了 128 个并发 gRPC 请求，结果 `128/128 success`，耗时约 `3.24s`，服务
日志无 OOM。这个结果证明低显存路径可行，
随后补做了同 profile 1 的 graph/eager 对照：batch 1/2 各四步的 token、音频、全部
cursor state 和 codec0 均一致。之前的失败判断来自 profile 1 graph 对 profile 0 eager
的跨 profile 比较，而不是 profile 1 的递归状态错误。

## 尚未完成

1. 用修复后的 cursor ONNX/engine 重导出，确认 prefill inactive passthrough 在 TRT 图中保持；
   当前 128 并发压测使用的是已有 cursor engine，runtime 已不再提交 prefill cursor state。
2. 用 profile 1 跑 `tn_streaming_cases.txt`，服务外离线调用 ASR，最终以文本/CER/WER、
   音频连续性和 EOS 行为完成质量验收。
3. profile 0 仅保留给 prefill/shared context，不再作为 cursor decode graph 的常驻显存成本。
