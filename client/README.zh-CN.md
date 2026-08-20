[English](README.md) | **中文**

# Qwen3-TTS Python 客户端

`qwen3-tts-client` 是一个轻量级 Python SDK，用于与 Qwen3-TTS 部署进行通信。
OpenAI Realtime 是主协议；旧的四种 transport 继续作为迁移期 fallback，并共用同一 API。

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/realtime")
result = client.synthesize_bytes("你好，欢迎使用 Qwen3-TTS。",
                                 request=SynthesisConfig(task_type="custom_voice"))
print(result.audio_format, len(result.audio_bytes))
```

> 你所需的一切都在单一的 `qwen3tts` 包内——正常使用时你无需导入任何其他内容。

## 特性

- **OpenAI Realtime 优先** —— `openai-realtime` 是主 transport；
  `engine-websocket`、`engine-grpc`、`triton-grpc`、`triton-http` 作为兼容
  fallback，全部由同一个 `TTSClient` 承载。
- **自动检测** —— `transport="auto"`（默认）会探测端点并
  在服务端声明或接受 Realtime 时优先绑定它。
- **一次性、流式（streaming）和实时（realtime）** 三种模式。
- **同步与异步** 客户端（`TTSClient` / `AsyncTTSClient`）。
- **精简依赖** —— 核心安装仅需 `requests` 和 `websocket-client`；
  gRPC / Triton / numpy 为可选附加项。

## 安装

SDK 是否兼容由 capabilities 中的**协议族和协议大版本**决定。`engine_version`
仅用于诊断；与 SDK 发布版本不同时会告警，但不会阻止连接。先查询引擎版本：

```bash
curl http://<engine-host>:<ws-port>/v1/capabilities
# → {"engine_version": "v0.1.0", ...}
```

**通道一 —— GitHub/GitLab Release。** 安装对应
[GitHub Release](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/releases)
或 GitLab Release 所附的 wheel；私有项目需要配置凭据。部署实例的
`/demo/#/sdk` 页面会给出精确命令，不在 README 中固化容易过期的版本和文件名。

这条命令下载 wheel，不会检出 Git 仓库。GitLab 还可通过项目 PyPI 索引按
capabilities 报告的版本安装同一文件。

**通道二 —— 从部署服务获取。** 每个正式镜像都嵌入已发布的 wheel，并通过公共
服务的 `GET /sdk/` 提供：

```bash
curl https://<public-service-base>/sdk/      # 查看 .whl 列表
pip install "https://<public-service-base>/sdk/<wheel-filename>"
```

从本地检出安装：`pip install ./client`（在仓库根目录执行）。协议族或协议大版本
不兼容时会在连接时立即报 `ProtocolVersionMismatchError`；同一大版本内的协议修订
兼容，engine/SDK 发布版本不同只产生 `RuntimeWarning`。设
`QWEN3TTS_SKIP_PROTOCOL_CHECK=1` 可将协议不兼容错误降级为警告。

附加项按连接对象 / 所需功能划分。可写在 Release 直链引用中
（`qwen3-tts-client[grpc] @ https://...whl`），或用于本地安装
（`pip install "./client[grpc]"`）：

| Extra | Pulls in | Use when |
|-------|----------|----------|
| `grpc` | `grpcio`, `protobuf` | engine-grpc transport |
| `triton` | `tritonclient` | triton-grpc transport |
| `audio` | `numpy` | `synthesize_array()` (ndarray output) |
| `all` | everything above | not sure / want it all |

需要 Python 3.10+。

## 快速开始

### 一次性合成

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/realtime")
result = client.synthesize_bytes(
    "你好，欢迎使用 Qwen3-TTS。",
    request=SynthesisConfig(task_type="custom_voice", speaker="serena"),
)
# result.audio_bytes is raw PCM; result.audio_format tells you encoding + rate.
print(result.transport, result.audio_format.encoding, result.audio_format.sample_rate)
print(result.details["usage"])  # 用于计费的终态 input/output token usage
```

需要 numpy 数组而不是字节（需要 `audio` 附加项）？

```python
arr = client.synthesize_array("你好。", request=SynthesisConfig(task_type="custom_voice"))
print(arr.audio_array.shape)
```

### 流式（streaming）

增量地喂入文本（例如上游 LLM 逐步产出文本），并在音频到达时逐步消费：

```python
from qwen3tts import TTSClient, SessionStartRequest, SynthesisConfig, AudioChunk, StreamEvent

client = TTSClient.connect("ws://localhost:50052/v1/realtime")
session = client.open_stream(
    SessionStartRequest(session_id="demo", config=SynthesisConfig(task_type="custom_voice"))
)
session.send_text("你好，")
session.send_text("这是流式输入。")
session.end()

for message in session.iter_messages():
    if isinstance(message, AudioChunk):
        ...  # message.pcm_bytes
    elif isinstance(message, StreamEvent):
        print("event:", message.type)

# response.done 后可读；取消或失败也保留部分 usage。
print(session.usage, session.response_id, session.response_status)
```

### 实时播放（WebRTC / 音频设备）

引擎会以不规则的节奏发出音频。`RealtimeAudioStream` 包装一个
会话，并以挂钟节拍产出固定大小的帧，插入静音以填补空隙，从而使播放
设备 / WebRTC track 永不欠载：

```python
from qwen3tts import TTSClient, RealtimeAudioStream, SessionStartRequest, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/realtime")
session = client.open_stream(
    SessionStartRequest(session_id="webrtc", config=SynthesisConfig(task_type="custom_voice"))
)
session.send_text("你好，欢迎使用实时语音合成。")
session.end()

# 20 ms frames, silence-filled — ready for WebRTC / local playback
for frame in RealtimeAudioStream(session, chunk_s=0.02, fill_silence=True):
    if frame.is_silence:
        continue
    webrtc_track.write(frame.data)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fill_silence` | `True` | Insert silence when the engine is late; `False` = passthrough |
| `chunk_s` | `0.02` | Frame size in seconds (20 ms = WebRTC Opus frame) |
| `sample_rate` | `24000` | Audio sample rate in Hz |

### 异步

`AsyncTTSClient` 以 `await` 镜像同步 API：

```python
from qwen3tts import AsyncTTSClient, SynthesisConfig

client = await AsyncTTSClient.connect("ws://localhost:50052/v1/realtime")
result = await client.synthesize_bytes("你好。", request=SynthesisConfig(task_type="custom_voice"))
# streaming: session = await client.aopen_stream(SessionStartRequest(...))
```

## 传输方式

`TTSClient.connect(endpoint, transport="auto")` 接受一个 URL 或 `host:port`，
并自动检测后端。若要显式固定，请传入 `transport=`：

| Endpoint example | Detected transport |
|------------------|--------------------|
| `ws://localhost:50052/v1/realtime` | `openai-realtime`（standalone） |
| `ws://localhost:50053/v1/realtime` | `openai-realtime`（Triton sidecar） |
| `ws://localhost:50052/v1/ws` | `engine-websocket` |
| `localhost:50051` | `engine-grpc` |
| `http://localhost:8000` | `triton-http` / `triton-grpc` |

### Realtime 迁移与 usage

完整文本的 `synthesize_bytes()` 使用标准 Realtime
`conversation.item.create` 和 `response.create` 事件。增量 `open_stream()` 使用
服务端明确声明的 `qwen.input_text_buffer.v1` append/commit 扩展，并在音频下行时继续
发送文本。未声明该扩展的服务仍可做一次性合成，但 SDK 会在 session 建立阶段拒绝
增量流式调用。

`response.done.response.usage` 在一次性调用中映射为
`result.details["usage"]`，流式调用则在终态事件后映射为 `session.usage`。若客户端在
终态前断联，已配置的服务端计费账本仍是权威数据源。

四种旧 transport 当前不会删除，但每个进程、每种 transport 会发出一次
`FutureWarning`。迁移期间可用 `QWEN3TTS_SUPPRESS_LEGACY_TRANSPORT_WARNING=1` 临时
静默。native WebSocket 通过 `stream_resume_v1`、Realtime 通过
`qwen.response_resume.v1` 声明活动流恢复；未声明对应能力时仍明确失败。

### 鉴权与 WebSocket 长连接

PaaS 网关使用 Bearer 鉴权时直接传 `key`；默认 `None` 表示不注入鉴权：

```python
client = TTSClient.connect(
    "wss://tts.example/v1/ws",
    key="your-key",
)
```

本地容器默认使用 `http://` / `ws://`，因此本地调试通常不需要 TLS 配置。若显式启用了
自签名 HTTPS/WSS，可传证书路径并保留严格校验，或只在临时联调时关闭校验：

```python
# 推荐：信任指定的自签名证书或私有 CA
client = TTSClient.connect(
    "wss://localhost:50052/v1/realtime",
    tls_verify="/path/to/cert.local.pem",
)

# 仅限本地联调，禁止用于生产
client = TTSClient.connect(
    "wss://localhost:50052/v1/realtime",
    tls_verify=False,
)
```

默认 `tls_verify=True` 使用系统信任链。该策略覆盖 HTTPS 能力探测、WSS 首连、预热与
断线重连。`verify_protocol=False` 只跳过协议兼容检查，与 TLS 无关；旧的 `verify`
参数仅作为该选项的兼容别名。

`engine-websocket` 会保留并复用已经完成会话的物理连接；并发会话使用连接池中的
不同连接。session 以 `done`/`error` 事件为边界，而不是以 socket 关闭为边界；若旧
gateway 没有长连接协议标识，SDK 会安全丢弃而不复用该 socket。engine error 的连接
同样会被丢弃；只有明确标记可复用的成功/取消 `done` 才进入池。默认每 15 秒在空闲
连接上做一次轻量保活；关闭保活时则在复用前探测
僵尸连接。
`reconnect_attempts` 只控制初始建连；活动流使用独立恢复预算。默认
`active_stream_resume=True`、`stream_resume_attempts=2`、
`stream_resume_timeout=10.0`、`stream_resume_ack_interval=8`。

当 native WebSocket 或 OpenAI Realtime gateway 声明支持恢复时，瞬时断联不会重启
逻辑 engine session。文本通过累计序号 ACK 去重，输出从已确认的 delivery/sample
游标继续，因此 SDK 不会从头重新合成，也不会把同一段音频重复放入消息队列。恢复受
服务端声明的 grace 与回放窗口约束；token 过期、重试耗尽、协议缺口、服务进程重启或
新连接被路由到另一副本都会明确失败。旧 gateway 会自动保持原先的快速失败行为。
使用完客户端后调用 `client.close()`，或使用上下文管理器。

## 示例

可运行脚本位于 [`examples/`](examples/) —— 请先启动一个端点，然后：

```bash
python examples/quickstart.py                  # one-shot     -> quickstart.wav
python examples/streaming.py                   # incremental  -> streaming.wav
python examples/realtime.py                    # wall-clock aligned frames
python examples/quickstart.py localhost:50051  # point at engine gRPC
python examples/quickstart.py wss://localhost:50052/v1/realtime \
  --tls-ca-file /path/to/cert.local.pem
```

## API 参考

完整的公开 API 就是 `qwen3tts` 导出的所有内容—— `import qwen3tts;
help(qwen3tts)` 或阅读 `qwen3tts.__all__`。主要名称：

- **客户端：** `TTSClient`、`AsyncTTSClient`
- **请求 / 配置：** `SynthesisConfig`、`SessionStartRequest`、`AudioFormat`、
  `VADPolicy`、`OutputPolicy`
- **流消息：** `AudioChunk`、`StreamEvent`、`StreamTextChunk`
- **结果：** `BytesResult`、`ArrayResult`、`Capabilities`
- **实时：** `RealtimeAudioStream`、`TimedAudio`
- **异常：** `TTSClientError` 及其子类（`TransportNotSupportedError`、
  `TransportProbeError`、`ProtocolError`、`DependencyMissingError`、
  `StreamClosedError`、`StreamRecoveryError`）

可选的延迟 / 计时诊断位于一个独立的子模块中，正常使用时并不需要：

```python
from qwen3tts.diagnostics import LatencyAnalyzer, ServerTimingReport
```

需要从第一次调用开始接入，请先阅读[5 分钟接入](../docs/user/quickstart.zh-CN.md)；需要
调整 VAD、交付策略或流式输入时，再阅读[高级配置](../docs/user/advanced_configuration.zh-CN.md)。
