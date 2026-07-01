[English](client_sdk.md) | **中文**

# Python Client SDK

## 目标

这个 SDK 面向外部调用方，提供统一、轻量、可直接 `pip install` 的 Python 客户端。

支持的服务入口：

- `engine-websocket`
- `engine-grpc`
- `triton-grpc`
- `triton-http`

默认行为是 `transport="auto"`，客户端会先做探测，再绑定到具体 adaptor。

## 目录与发布

SDK 作为独立子项目放在仓库的 [`client/`](../../client) 目录下：

- 打包配置：[`client/pyproject.toml`](../../client/pyproject.toml)
- 源码入口：[`client/src/qwen3tts`](../../client/src/qwen3tts)
- 共享协议层：[`client/src/qwen3tts_protocol`](../../client/src/qwen3tts_protocol)
- SDK 单测：[`client/tests`](../../client/tests)

这样做的目的，是避免把服务端重量依赖和部署逻辑一起打进客户端 wheel。

## 安装

核心包：

```bash
cd client
pip install .
```

可选 extras：

```bash
pip install .[grpc]
pip install .[triton]
pip install .[audio]
pip install .[all]
```

依赖策略：

- `core`：`requests`，以及纯 Python websocket/http 逻辑
- `grpc`：standalone gRPC 所需 runtime
- `triton`：Triton gRPC / HTTP 所需 runtime
- `audio`：`numpy`，用于 `synthesize_array`

## 快速开始

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("localhost")
caps = client.get_capabilities()
print(caps.loaded_model_type)

result = client.synthesize_bytes(
    "你好，欢迎使用 Qwen3-TTS。",
    request=SynthesisConfig(task_type="custom_voice"),
)
print(result.audio_format)
print(len(result.audio_bytes))
```

## 统一流式接口

```python
from qwen3tts import SessionStartRequest, SynthesisConfig, TTSClient

client = TTSClient.connect("ws://localhost:50052/v1/ws")
session = client.open_stream(
    SessionStartRequest(
        session_id="demo-session",
        config=SynthesisConfig(task_type="custom_voice"),
    )
)

session.send_text("你好，")
session.send_text("这是统一流式协议。")
session.end()

for message in session.iter_messages():
    print(type(message).__name__, getattr(message, "meta", {}))
```

## 自动探测规则

显式 `transport=` 时不探测，直接走指定 adaptor。

`transport="auto"` 时：

- `ws://` / `wss://`：直接判定为 `engine-websocket`
- `http://` / `https://`：先探测 standalone `GET /v1/capabilities`，失败后探测 Triton HTTP `/v2/health/live`、`/v2/health/ready`、`/v2/models/<model>/ready`
- 裸 `host:port`：优先按端口规则探测 standalone，再探测 Triton，再回落 HTTP
- 裸 `host`：自动扩展默认候选端口 `50052`、`50051`、`8001`、`8000`

探测结果会暴露在：

- `client.resolved_transport`
- `client.probe_report`
- `client.detected_transport`

## Triton HTTP 的流式语义

Triton HTTP 本身不支持真正的 decoupled streaming infer。

因此 SDK 的统一策略是：

- `synthesize_bytes` / `synthesize_array`：直接走一次 HTTP infer
- `open_stream(...)`：本地缓存 `start/text/end`
- 调用 `end()` 后，才触发一次 HTTP infer
- 返回的 session 会显式标记 `degraded_to_oneshot=True`

这意味着：

- 它不会伪装成服务端边收边合成
- 但上层调用代码仍可复用同一套 session API

## 当前状态说明

当前版本已经完成这些结构目标：

- SDK 代码被收敛到 `client/` 子项目
- 提供统一同步 / 异步 façade
- 引入共享协议层 `qwen3tts_protocol`
- 提供 auto-detect 骨架和四类 adaptor 入口
- 为 `triton-http` 提供显式的流式降级语义

当前仍建议把它视为 v1 alpha：

- transport 适配已成型，但还需要继续补更完整的集成验证
- `engine-websocket` / `engine-grpc` 路径最接近现有服务端真实合同
- Triton 相关路径后续建议继续补真实环境 smoke test
