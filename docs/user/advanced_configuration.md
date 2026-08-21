**English** | [中文](advanced_configuration.zh-CN.md)

# Advanced configuration

Complete a request with defaults first, then tune these controls for your application. Available
values always come from the running service's `/v1/capabilities`. The Demo's **SDK** page turns the
current playground settings into copyable code.

## Choose a synthesis task

| Goal | `task_type` | Also provide |
| --- | --- | --- |
| Use a built-in voice | `custom_voice` | `speaker` |
| Design a voice from text | `voice_design` | `instruct` |
| Clone a reference voice | `voice_clone` | reference audio, and sometimes reference text |

An enum in the SDK does not mean that a deployment supports it. Check `tasks`, `speakers`,
`languages`, and `reference` in capabilities. `custom_voice` is the recommended path today.

## Full text and incremental text

Use one-shot synthesis when all text is ready:

```python
result = client.synthesize_bytes(
    "This text is already complete.",
    request=SynthesisConfig(task_type="custom_voice", speaker="serena"),
)
```

Use an incremental session for text arriving from an LLM or another streaming source:

```python
from qwen3tts import AudioChunk, SessionStartRequest, SynthesisConfig

session = client.open_stream(SessionStartRequest(
    session_id="turn-42",
    config=SynthesisConfig(task_type="custom_voice", speaker="serena"),
))
session.send_text("Hello, ")
session.send_text("this input arrived incrementally.")
session.end()

for message in session.iter_messages():
    if isinstance(message, AudioChunk):
        audio_sink.write(message.pcm_bytes)
```

`session.end()` closes text input; it does not mean playback has completed. Consume through a
terminal event.

## Audio format

Start with mono 24 kHz PCM16 for browser playback, WAV wrapping, and common realtime pipelines:

```python
from qwen3tts import AudioFormat, SynthesisConfig

config = SynthesisConfig(
    task_type="custom_voice",
    speaker="serena",
    audio=AudioFormat(encoding="pcm_s16le", sample_rate=24000, channels=1),
)
```

Only formats advertised by capabilities are valid. Read `audio_format` from the result instead of
inferring it from byte length.

## Output VAD

Output VAD detects and trims leading or trailing silence. It is not microphone input VAD and does
not verify whether generated speech matches the text.

```python
from qwen3tts import OutputPolicy, VADPolicy

config.output_policy = OutputPolicy(vad=VADPolicy(
    enabled=True,
    strategy="energy",
    chunk_ms=16,
    begin_threshold=0.30,
    begin_count=5,
    end_threshold=0.20,
    end_count=31,
    start_margin_ms=20,
))
```

- VAD waits to confirm speech onset, so effective-audio TTFT may increase.
- A high begin threshold can reject an entire valid result; detector score scales are not interchangeable.
- `start_margin_ms` preserves a small lead-in so initial consonants are not clipped.
- Keep VAD disabled when leading silence is acceptable and the earliest raw packet matters most.

Tune in the Demo while comparing raw TTFT, VAD gate time, and effective TTFT.

## Delivery policy

`guarded` is recommended. It retains a limited unplayed tail, allowing the service to retract a bad
tail after abnormal generation is detected. `firehose` forwards everything immediately but cannot
retract bytes already received by the client.

```python
from qwen3tts import OutputPolicy

config.output_policy = OutputPolicy(
    chunk_ms=0,
    emit_text_events=True,
    config={"delivery": "guarded", "delivery_window_ms": 160},
)
```

Custom players using guarded delivery should report buffered and played sample cursors. The Browser
SDK player already implements this loop.

## Authentication, timeouts, and reuse

```python
client = TTSClient.connect(
    "wss://tts.example.com/v1/ws",
    key="your-api-key",
    timeout=120.0,
    connect_timeout=5.0,
)
```

- `key` is passed to the deployment gateway as a Bearer token; never embed it in static frontend code.
- Local endpoints use `ws://` by default. For self-signed WSS, set
  `tls_verify="/path/to/cert.local.pem"`; `tls_verify=False` is for temporary
  local debugging only and must not be shipped.
- `verify_protocol=False` disables only protocol compatibility checking, not TLS verification.
- Reuse one client per process instead of reconnecting for every utterance.
- Set connection, acquisition, and active-request budgets to match your SLA.
- Log the response ID, terminal error code, and server timing on failures.

## Pre-launch checks

Cover empty text, long text, mixed languages, cancellation, network recovery, unsupported speakers,
VAD rejecting all audio, rate limiting, and server errors. For latency work, separate raw server TTFT,
VAD gating, transport, and player startup instead of relying on one end-to-end number.
