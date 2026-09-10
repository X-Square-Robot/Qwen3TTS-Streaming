[English](README.md) | **中文**

# 示例

`qwen3-tts-client` SDK 的可运行示例。每个示例接受一个可选的
端点参数（默认：`ws://localhost:50052/v1/ws`）。

请先启动一个 Qwen3-TTS 端点（见仓库 README），然后：

```bash
# 从 https://<public-service-base>/demo/#/sdk 复制精确安装命令。
# 或在源码检出目录安装：pip install "../[all]"
python quickstart.py                  # one-shot synthesis -> quickstart.wav
python streaming.py                   # incremental text -> streaming.wav
python realtime.py                    # wall-clock aligned frames (WebRTC/playback)
```

指向不同的端点：

```bash
python quickstart.py ws://localhost:50053/v1/ws        # Triton sidecar 原生入口
python quickstart.py ws://localhost:50052/v1/realtime  # OpenAI Realtime 兼容入口
python quickstart.py localhost:50051                  # 旧 engine gRPC
python quickstart.py http://localhost:8000            # 旧 Triton HTTP
```

| Example | Shows |
|---------|-------|
| `quickstart.py` | `TTSClient.connect` + `synthesize_bytes`, save WAV |
| `streaming.py` | `open_stream` + `send_text`/`end` + `iter_messages` |
| `realtime.py` | `RealtimeAudioStream` for isochronous playback frames |

> 这些示例中的 `speaker`（`serena`）依赖于具体模型；请使用你部署的
> 模型所支持的说话人（custom_voice 变体会暴露具名说话人）。
