[English](openai_realtime.md) | **中文**

# OpenAI Realtime TTS 协议与 Triton 边界

## 结论

新客户端应以 `/v1/realtime` 作为主入口。standalone 默认使用
`ws://<host>:50052/v1/realtime`，Triton compose 部署默认使用
`ws://<host>:50053/v1/realtime`。`/v1/ws`、旧 gRPC 和 Triton JSON action 协议进入
兼容期：当前不删除，但不再作为新功能的公共协议设计中心。

OpenAI Realtime 的传输是全双工的。gateway 在音频持续下行时仍然读取客户端事件，
所以客户端可以并行提交后续输入或发送 `response.cancel`。当前引擎在一条连接上只允许
一个活动 `response`，完成后可以复用同一连接；这是执行并发限制，不是把 WebSocket
退化成半双工。

## 两种文本输入

完整文本使用标准 Realtime 事件：

```json
{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"你好，世界。"}]}}
{"type":"response.create"}
```

OpenAI Realtime 目前没有“向已经开始的文本 item 追加内容”的客户端事件。为了保留本
项目的 token 级输入，提供一个窄的、带命名空间的扩展；响应、音频、错误和 usage 仍
使用标准事件：

```json
{"type":"response.create"}
{"type":"qwen.input_text_buffer.append","sequence":1,"text":"你好，"}
{"type":"qwen.input_text_buffer.append","sequence":2,"text":"世界。"}
{"type":"qwen.input_text_buffer.commit"}
```

`sequence` 从 1 开始严格递增。相同 sequence 和相同 text 的重试幂等；序号缺口或相同
序号不同内容会收到 `error`，但不会关闭物理连接。服务端用
`qwen.input_text_buffer.ack` 和 `qwen.input_text_buffer.committed` 确认输入。

主要下行顺序为：

```text
response.created
  -> response.output_item.added
  -> response.content_part.added
  -> response.output_audio.delta *
  -> response.output_audio.done
  -> response.content_part.done
  -> response.output_item.done
  -> response.done
```

`response.output_audio.delta.delta` 是 Base64 编码的 mono PCM16。当前支持 16 kHz 和
24 kHz，推荐 24 kHz。

## Usage 与计费

usage 在 `response.done.response.usage` 返回，同时 gateway 发射一次服务端
`billing.usage` 结构化记录。不能以客户端是否成功收到 `response.done` 作为计费事实；
生产部署应把可注入的 `usage_recorder` 接到持久化账本。

Triton sidecar 默认启用 JSONL recorder，并把宿主机文件挂载为
`workspace/realtime_usage/realtime_usage.jsonl`。它是 append-only 的计费交接账本，
不是最终出账系统；下游应按 `response_id` 摄取和去重。

- `input_token_details.text_tokens` 使用实际模型 tokenizer 计数合成文本、instructions
  和 ref_text；不是按字符估算。
- `output_token_details.audio_tokens` 按已经发出的 PCM 时长计数，每 50 ms 一个 token，
  最后不足 50 ms 向上取整。
- `cancelled` 和 `failed` response 也返回并记录已接受输入、已发送音频对应的部分 usage。
- 当前 prefix KV cache 是本地执行优化，不等同于 OpenAI 的 cached-token 计费语义，
  因此 `cached_tokens` 报 0。

## Triton 如何兼容

Triton 不应直接实现 OpenAI WebSocket 协议。推荐拓扑是：

```text
OpenAI Realtime client
        <=> WebSocket /v1/realtime
Realtime gateway sidecar
        <=> RealtimeSessionBackend
        <=> bidirectional gRPC / Triton decoupled stream
Triton tts_orchestrator -> shared engine scheduler
```

原因是 decoupled Triton model 的流式响应需要 gRPC；Triton HTTP 不能承载这条持续的双向
会话。gateway 负责 OpenAI 事件状态机、Base64 PCM、鉴权、限流、取消和计费；Triton
只负责内部 session 的 `start / append / complete / cancel` 以及流式返回 audio/event。

代码中的 `RealtimeSessionBackend` 定义了这个边界。`EngineRealtimeBackend` 调用进程内
`TTSEngine`；已经实现的 `TritonRealtimeBackend` 为每个活动 Realtime response 持有一条
Triton streaming-gRPC client stream：`init` 持有 decoupled response，
`append_text`、`text_complete`、`cancel` 在音频反向流动时仍通过同一条 gRPC stream
发送。公共 ID 不会成为 Triton execution key。

计费时，sidecar 从当前挂载的同版本模型包加载精确 tokenizer，在本地计算 input token，
无需额外发起模型请求，也不会因 Triton replica 不同而漂移；output audio token 仍按
gateway 实际发出的 PCM 样本计数。因此它保证：

1. 一个 Realtime response 映射到一个私有 engine execution ID；
2. append 与 audio 可以同时在 gRPC stream 两个方向流动；
3. cancel 能传到同一个 execution；
4. sidecar 使用模型包同版本 tokenizer 计算输入 token；
5. gateway 仍以实际发出的 PCM 样本数计算输出 audio token。

因此，Triton 的老 JSON action 可以先保留为内部兼容适配，之后再替换；不会把它暴露成
新的公共协议，也不会阻塞 `/v1/realtime` 的演进。

## 下线顺序

1. 已完成：新增 `/v1/realtime`，旧 `/v1/ws` 和 SDK 保持可用，capabilities 同时声明
   `openai-realtime-v1` 与 `tts-session-v2alpha1`。
2. 已完成：新 SDK 的自动探测优先选择 Realtime；旧 transport 作为显式或自动
   fallback，并发出 deprecation warning。
3. 已完成：sidecar 接入双向 gRPC backend，Triton JSON action 降为内部协议。
4. 稳定期：usage ledger、鉴权、配额和断线恢复通过验收后，再公布旧协议删除版本。

当前第 1–3 步已经完成。第 4 步仍是验收门槛：durable usage、鉴权、配额与 Realtime
断线恢复达到生产要求并公布删除版本之前，兼容 transport 继续保留。
