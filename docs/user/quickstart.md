**English** | [中文](quickstart.zh-CN.md)

# 5-minute setup

This guide is for application developers who already have a service URL. Its single goal is to
install the SDK that matches that service and complete one synthesis request.

> Just want to hear it first? Open `/demo/` on the service, enter text, and select **Synthesize and
> play**. Nothing needs to be installed.

## 1. Identify the service URL

Your operator will normally provide a base URL such as:

```text
https://tts.example.com
```

The portal and APIs share that origin:

| Purpose | URL |
| --- | --- |
| Try it | `https://tts.example.com/demo/` |
| Python / Browser SDK | `https://tts.example.com/demo/#/sdk` |
| Engineering Lab | `https://tts.example.com/demo/#/lab` (basic Realtime experiments; optional deep panels) |
| Capabilities | `https://tts.example.com/v1/capabilities` |
| Python SDK native WebSocket | `wss://tts.example.com/v1/ws` |
| OpenAI Realtime compatibility endpoint | `wss://tts.example.com/v1/realtime` |

The table shows a production deployment with trusted TLS. A local container
serves plain HTTP/WS by default: `http://localhost:50052/demo/` for the Demo and
`ws://localhost:50052/v1/ws` for the SDK. Do not change a local URL to
`https://` / `wss://` unless the operator explicitly enabled TLS.

Verify that the service returns its capabilities:

```bash
curl https://tts.example.com/v1/capabilities
```

Ask the operator for an API key if the deployment requires authentication.

## 2. Install the matching Python SDK

Open the service's **SDK** page and copy its generated `pip install` command. It points to the wheel
that matches the running service; cloning this repository is not required.

```bash
pip install "qwen3-tts-client[all] @ https://tts.example.com/sdk/<wheel-filename>"
```

For long-lived applications, pin the version and SHA256 shown on that page.

## 3. Synthesize your first audio

Replace the URL, API key, and speaker with values supplied by the operator:

```python
from qwen3tts import AudioFormat, SynthesisConfig, TTSClient

with TTSClient.connect(
    "wss://tts.example.com/v1/ws",
    key="your-api-key",  # remove this line when authentication is disabled
) as client:
    result = client.synthesize_bytes(
        "Hello, this is my first synthesized message.",
        request=SynthesisConfig(
            task_type="custom_voice",
            speaker="serena",
            audio=AudioFormat(encoding="pcm_s16le", sample_rate=24000),
        ),
    )

print(len(result.audio_bytes), result.audio_format)
print(result.details.get("usage", {}))
```

For a self-signed HTTPS/WSS deployment, give the SDK its CA/certificate file.
The same policy is used for HTTPS capability discovery, the initial WSS dial,
and reconnects:

```python
client = TTSClient.connect(
    "wss://localhost:50052/v1/ws",
    tls_verify="/path/to/cert.local.pem",
)
```

For local TLS debugging only, `tls_verify=False` disables certificate and
hostname verification. Never ship that setting. Plain `ws://` remains the
simplest local path.

`result.audio_bytes` is mono PCM described by `result.audio_format`. Run
`client/examples/quickstart.py` for a complete WAV-writing example, or pass the bytes to your
application's audio library.

## 4. Before integrating

- Do not guess tasks, speakers, languages, or sample rates; use `/v1/capabilities`.
- New Python SDK integrations should prefer native `/v1/ws`. Use the
  `/v1/realtime` compatibility endpoint only for OpenAI Realtime event-model
  integrations; do not start new integrations on the old gRPC or Triton-native APIs.
- Always wait for a completed, cancelled, or failed terminal event; terminal usage is returned there.
- Use the Browser SDK for web apps; it handles audio decoding, playback cursors, and recovery.

After the first successful request, continue with [Advanced configuration](advanced_configuration.md).
Read [Endpoints and events](realtime_api.md) only when you need to work below an SDK.
