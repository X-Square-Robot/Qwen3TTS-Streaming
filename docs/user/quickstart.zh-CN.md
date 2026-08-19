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
| Realtime WebSocket | `wss://tts.example.com/v1/realtime` |

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
    "wss://tts.example.com/v1/realtime",
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

`result.audio_bytes` 是单声道 PCM，格式以 `result.audio_format` 为准。需要直接写成 WAV
时可运行仓库中的完整示例 `client/examples/quickstart.py`，或交给业务已有的音频库处理。

## 4. 接入前确认

- 不要猜测任务、说话人、语言或采样率；以 `/v1/capabilities` 返回值为准。
- 新接入统一使用 `/v1/realtime`，不要从旧的 gRPC 或 `/v1/ws` 开始。
- 一定等待成功、取消或失败终态；`usage` 也在终态返回。
- 浏览器接入请使用 Browser SDK，它已经处理 Base64 音频、播放游标和断线恢复。

第一次合成成功后，继续阅读[高级配置](advanced_configuration.zh-CN.md)。只有在需要自己处理
底层事件时，才阅读[接口与事件](realtime_api.zh-CN.md)。
