[English](README.md) | **中文**

# Qwen3-TTS Python 客户端

`qwen3-tts-client` 是一个轻量级 Python SDK，用于与 Qwen3-TTS 部署进行通信。
统一的导入根、统一的 API、四种传输方式——指向某个端点即可合成语音。

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/ws")
result = client.synthesize_bytes("你好，欢迎使用 Qwen3-TTS。",
                                 request=SynthesisConfig(task_type="custom_voice"))
print(result.audio_format, len(result.audio_bytes))
```

> 你所需的一切都在单一的 `qwen3tts` 包内——正常使用时你无需导入任何其他内容。

## 特性

- **统一 API，四种传输方式** —— `engine-websocket`、`engine-grpc`、
  `triton-grpc`、`triton-http`，全部由同一个 `TTSClient` 承载。
- **自动检测** —— `transport="auto"`（默认）会探测端点并
  绑定正确的适配器，因此通常只需传入一个 URL。
- **一次性、流式（streaming）和实时（realtime）** 三种模式。
- **同步与异步** 客户端（`TTSClient` / `AsyncTTSClient`）。
- **精简依赖** —— 核心安装仅需 `requests`；gRPC / Triton /
  numpy 为可选附加项。

## 安装

```bash
pip install qwen3-tts-client          # core (WebSocket + HTTP transports)
```

附加项，按你连接的对象 / 所需功能划分：

| Extra | Install | Pulls in | Use when |
|-------|---------|----------|----------|
| `grpc` | `pip install "qwen3-tts-client[grpc]"` | `grpcio`, `protobuf` | engine-grpc transport |
| `triton` | `pip install "qwen3-tts-client[triton]"` | `tritonclient` | triton-grpc transport |
| `audio` | `pip install "qwen3-tts-client[audio]"` | `numpy` | `synthesize_array()` (ndarray output) |
| `all` | `pip install "qwen3-tts-client[all]"` | everything above | not sure / want it all |

需要 Python 3.10+。

## 快速开始

### 一次性合成

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/ws")
result = client.synthesize_bytes(
    "你好，欢迎使用 Qwen3-TTS。",
    request=SynthesisConfig(task_type="custom_voice", speaker="serena"),
)
# result.audio_bytes is raw PCM; result.audio_format tells you encoding + rate.
print(result.transport, result.audio_format.encoding, result.audio_format.sample_rate)
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

client = TTSClient.connect("ws://localhost:50052/v1/ws")
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
```

### 实时播放（WebRTC / 音频设备）

引擎会以不规则的节奏发出音频。`RealtimeAudioStream` 包装一个
会话，并以挂钟节拍产出固定大小的帧，插入静音以填补空隙，从而使播放
设备 / WebRTC track 永不欠载：

```python
from qwen3tts import TTSClient, RealtimeAudioStream, SessionStartRequest, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/ws")
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

client = await AsyncTTSClient.connect("ws://localhost:50052/v1/ws")
result = await client.synthesize_bytes("你好。", request=SynthesisConfig(task_type="custom_voice"))
# streaming: session = await client.aopen_stream(SessionStartRequest(...))
```

## 传输方式

`TTSClient.connect(endpoint, transport="auto")` 接受一个 URL 或 `host:port`，
并自动检测后端。若要显式固定，请传入 `transport=`：

| Endpoint example | Detected transport |
|------------------|--------------------|
| `ws://localhost:50052/v1/ws` | `engine-websocket` |
| `localhost:50051` | `engine-grpc` |
| `http://localhost:8000` | `triton-http` / `triton-grpc` |

## 示例

可运行脚本位于 [`examples/`](examples/) —— 请先启动一个端点，然后：

```bash
python examples/quickstart.py                  # one-shot     -> quickstart.wav
python examples/streaming.py                   # incremental  -> streaming.wav
python examples/realtime.py                    # wall-clock aligned frames
python examples/quickstart.py localhost:50051  # point at engine gRPC
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
  `StreamClosedError`）

可选的延迟 / 计时诊断位于一个独立的子模块中，正常使用时并不需要：

```python
from qwen3tts.diagnostics import LatencyAnalyzer, ServerTimingReport
```

项目手册：`docs/user/client_sdk.zh-CN.md`。

---

### 面向贡献者

本 SDK 构建于 `qwen3tts_protocol` 之上，这是一个无依赖的包，
它持有 wire-format 类型，是由客户端、引擎和 demo server 共享的单一真相源（single
source of truth）。**客户端用户无需它**——它定义的每个
类型都从 `qwen3tts` 重新导出。仅当处理协议本身或服务端组件时，
才直接接触 `qwen3tts_protocol`。
