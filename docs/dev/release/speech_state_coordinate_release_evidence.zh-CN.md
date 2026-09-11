# Speech State / Coordinate Release Evidence

验证日期：2026-09-09

## 结论

本轮坐标、流式 TN、cursor、Splitter、Native Gateway 和 Triton 兼容层验证通过。
本次服务级验收使用 `/data/models/tts/qwen3-tts/trained/sample-model/0818-trained/`
导出的 native-cursor 包，而不是 standard plan。

## 核心协议

统一坐标链为：

```text
raw -> TN normalized -> clause/segment -> token -> codec frame
```

Native Cursor 与 EMA 都只负责 `codec -> token` 估计；后续 token、segment、TN、raw
逆向映射使用同一套 provenance 坐标投影。音频 chunk 携带 codec 区间，不把 PCM 时间
坐标混入文本 high-water。

## 自动化测试

完整测试基线：`1333 passed, 68 skipped`。

本轮重点联测：

```text
pytest -q tests/unit/engine_core/test_text_coordinates.py \
  tests/unit/frontend/test_shared_progress_coordinates.py \
  tests/integration/test_streaming_tn_cursor_plan_contract.py
16 passed
```

覆盖：三种输入入口、`99% -> 百分之九十九`、tail rewrite、跨 segment owner、真实
Splitter 边界、Native/EMA 共享坐标投影、lookahead hold 和音频 chunk 坐标绑定。

## Triton 服务验收

使用本地 TRT 10.13.3 验证镜像和现有 model repository 启动 Triton：

```text
tts_orchestrator: READY
tts_orchestrator_http: READY
Loaded TRT engine: model.plan (85 I/O tensors)
Executor loaded (engine=TRT, effective_max_seq=512, cuda_graph=on)
```

模型 manifest 明确声明 `native_cursor.enabled=true`，并包含 cursor state 输入、
cursor 输出和 `codec0` 输出。该 engine 由 `talker_code2wav_fused.onnx` 使用 TRT
10.13.3 构建，profile 为 `max_batch=1, max_seq=512`。

统一服务测试：

```text
health: PASS
grpc-synthesize: PASS
medium-long: PASS
very-long: PASS
```

## Native/Triton 同结果集与 ASR

固定输入：`今天天气真好，我们一起出去玩吧。`

Native 与 Triton 分别生成：

```text
workspace/validation/triton_local/native_same.wav
workspace/validation/triton_local/triton-grpc/triton_grpc_synthesize.wav
```

本次 native-cursor 双入口实际产物为：

```text
workspace/validation/sample-cursor_native.wav
workspace/validation/sample-cursor_triton/triton-grpc/triton_grpc_synthesize.wav
```

Native Gateway 目标模型 `custom-1.7b` 的 `max_batch=1` profile 下，单请求、流式文本、
medium-long、very-long 均通过。`concurrent-x4` 不属于本 profile 的承诺能力，未纳入
本次发布通过项；若需要并发 4，必须用 `--max-batch-size 4` 重新构建 engine，不能由
运行时越过 profile 上限。

使用离线 ASR 服务：
`wss://infer.x2robot.com/infer/inf-dddq5qn77jrws5eu/v1/ws`

两份音频结果均为：

```text
今天天气真好，我们一起出去玩吧。
```

BF16 下不要求 codec/token 逐 token 一致；最终验收以音频 ASR 结果为准。

## 备注

本地验证镜像使用 TRT 10.13.3，与 model artifact fingerprint 匹配。生产镜像必须
保持相同 TRT plan/runtime 版本组合，不能使用 TRT 10.15.1 运行该 plan。
