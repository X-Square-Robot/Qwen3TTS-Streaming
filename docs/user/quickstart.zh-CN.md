[English](quickstart.md) | **中文**

# 5 分钟接入

这份指南面向已经拿到服务地址的调用方。目标只有一个：用匹配当前实例的 SDK，完成一次
语音合成并拿到可播放的 PCM 音频。

> 只想先听效果？直接打开当前服务的 `/demo/`，输入文本后点击“合成并播放”，不需要安装
> 任何东西。

## 1. 确认服务地址

部署方通常会提供类似下面的地址：

```text
https://tts.example.com
```

浏览器页面和接口使用同一个域名：

| 用途 | 地址 |
| --- | --- |
| 在线体验 | `https://tts.example.com/demo/` |
| Python / Browser SDK | `https://tts.example.com/demo/#/sdk` |
| 服务能力 | `https://tts.example.com/v1/capabilities` |
| Python SDK 原生 WebSocket | `wss://tts.example.com/v1/ws` |
| OpenAI Realtime 兼容入口 | `wss://tts.example.com/v1/realtime` |

上表是生产环境可信 TLS 的示例。容器在本机默认提供明文 HTTP/WS：Demo 为
`http://localhost:50052/demo/`，SDK 入口为
`ws://localhost:50052/v1/ws`。除非部署方显式启用 TLS，不要把本地地址改写成
`https://` / `wss://`。

先检查服务是否返回能力信息：

```bash
curl https://tts.example.com/v1/capabilities
```

如果服务需要鉴权，请向部署方索取 API Key。

## 2. 安装当前实例的 Python SDK

打开当前实例的 **SDK** 页面，复制页面生成的 `pip install` 命令。该命令指向与服务版本
匹配的 wheel，不需要检出本仓库。

```bash
pip install "qwen3-tts-client[all] @ https://tts.example.com/sdk/<wheel-filename>"
```

长期运行的应用建议把页面显示的版本和 SHA256 固定到依赖清单中。

## 3. 完成第一次合成

把地址、API Key 和说话人替换成部署方提供的值：

```python
from qwen3tts import AudioFormat, SynthesisConfig, TTSClient

with TTSClient.connect(
    "wss://tts.example.com/v1/ws",
    key="your-api-key",  # 无鉴权时删除这一行
) as client:
    result = client.synthesize_bytes(
        "你好，这是我的第一条合成语音。",
        request=SynthesisConfig(
            task_type="custom_voice",
            speaker="serena",
            audio=AudioFormat(encoding="pcm_s16le", sample_rate=24000),
        ),
    )

print(len(result.audio_bytes), result.audio_format)
print(result.details.get("usage", {}))
```

若部署方给的是自签名 HTTPS/WSS，可把其 CA/证书文件显式交给 SDK；该策略会同时用于
HTTPS capabilities 探测、WSS 首次连接和断线重连：

```python
client = TTSClient.connect(
    "wss://localhost:50052/v1/ws",
    tls_verify="/path/to/cert.local.pem",
)
```

只做本地 TLS 联调时也可以使用 `tls_verify=False`。它会关闭证书和主机名校验，不应进入
生产配置或提交到业务代码。普通本地联调直接使用默认 `ws://` 更简单。

`result.audio_bytes` 是单声道 PCM，格式以 `result.audio_format` 为准。需要直接写成 WAV
时可运行仓库中的完整示例 `client/examples/quickstart.py`，或交给业务已有的音频库处理。

## 4. 接入前确认

- 不要猜测任务、说话人、语言或采样率；以 `/v1/capabilities` 返回值为准。
- Python SDK 新接入优先使用原生 `/v1/ws`；只有对接 OpenAI Realtime 事件模型时才使用
  `/v1/realtime` 兼容入口。旧 gRPC 和 Triton 原生接口不作为新接入起点。
- 一定等待成功、取消或失败终态；`usage` 也在终态返回。
- 浏览器接入请使用 Browser SDK，它已经处理 Base64 音频、播放游标和断线恢复。

第一次合成成功后，继续阅读[高级配置](advanced_configuration.zh-CN.md)。只有在需要自己处理
底层事件时，才阅读[接口与事件](realtime_api.zh-CN.md)。
