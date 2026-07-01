**English** | [中文](realtime_audio.zh-CN.md)

# G7: Client-Side Realtime Audio Stream (RealtimeAudioStream)

> Branch: `refact`
> Written: 2026-06-16
> Status: Done
> Related: the handling of `tests/save.py` in G5 of [[REFACTOR_GOALS.md]]

---

## 0. Background

The current engine pushes frames as fast as possible (first frame ~800ms, then a block roughly every ~2s), so the output cadence is uneven. `tests/save.py` implements the "non-isochronous audio → isochronous audio" adaptation logic, but it exists only as a test script and has not been integrated into the production code.

In the following scenarios, the consumer needs an isochronous audio stream:

- **WebRTC server**: the user reaches the cloud over WebRTC, and the WebRTC server reaches the engine over a custom client-server protocol. WebRTC's jitter buffer needs a steady audio input cadence, otherwise underrun dropouts occur.
- **Local realtime playback**: when the client plays audio directly, silence padding is needed to mask generation gaps.
- **Testing/benchmarking**: validating TTFT, dropout rate, and buffer watermark in realtime scenarios.

The core value of `save.py` is not "playback simulation," but **adapting a non-isochronous source to isochronous consumption**. A WebRTC server is exactly such an isochronous consumer.

---

## 1. Architecture Decision

| Option | Conclusion | Reason |
|------|------|------|
| Integrate into the engine (like VAD) | ❌ | The engine should emit frames as soon as possible; `time.sleep()` and queue blocking violate the engine's design goals, and the engine does not know the playback state |
| Integrate into the client SDK | ✅ | The client owns the playback context, it is consistent with the existing `iter_messages()` consumption interface, and it is optional and non-intrusive |
| Only as a testing tool | ❌ | Testing is just one of the consumption scenarios; realtime playback and the WebRTC server need it equally |
| Optional wrapper at the gateway layer | ⚠️ Optional | If the server needs to push the stream to the browser at a realtime cadence, a lightweight wrapper can be added in the gateway, but the core logic belongs to the client |

**Target architecture**:

```
User browser/app
      │ WebRTC (audio RTP, with built-in jitter buffer + playback cadence)
      ▼
  WebRTC server (SFU/MCU)
      │ Custom WebSocket/gRPC protocol
      ▼
  TTS Client SDK (RealtimeAudioStream optionally enabled)
      │
      ▼
  TTS engine (unchanged, still emits frames as fast as possible)
```

---

## 2. Specific Changes

### 2.1 Add `client/src/qwen3tts/realtime.py`

```python
@dataclass
class TimedAudio:
    """An audio frame with timing information."""
    data: bytes          # PCM audio data
    duration_s: float    # frame duration (seconds)
    is_silence: bool = False  # whether it is padding silence

class RealtimeAudioStream:
    """Convert a non-isochronous AudioChunk stream into an isochronous audio stream.

    Used to provide input to consumers that need a realtime cadence, such as WebRTC
    servers and local players. Disabled by default; the user creates it on demand.

    Args:
        session: BaseStreamSession, the audio source
        fill_silence: whether to fill gaps with silence (default True)
        chunk_s: isochronous output granularity, default 0.02 (20ms, aligned with the WebRTC Opus frame length)
        sample_rate: sample rate (Hz), default 24000
    """

    def __init__(
        self,
        session: BaseStreamSession,
        fill_silence: bool = True,
        chunk_s: float = 0.02,
        sample_rate: int = 24000,
    ): ...

    def __iter__(self) -> Iterator[TimedAudio]: ...
```

### 2.2 `chunk_s` Defaults to 0.02 Rather Than 0.01

- The standard WebRTC audio frame length is 20ms (the Opus default frame length)
- A 10ms granularity produces too many queue operations and silence-frame slices
- Aligning to 20ms maps directly to a WebRTC audio frame, reducing slicing and re-packetization

### 2.3 Update `client/src/qwen3tts/__init__.py`

- Export `RealtimeAudioStream` and `TimedAudio`

### 2.4 Remove `tests/save.py`

- The core logic has been migrated into `realtime.py`
- Test validation switches to unit tests of `RealtimeAudioStream`

### 2.5 Add `client/tests/test_realtime.py`

- Test correctness of silence padding
- Test first-frame latency handling
- Test the late-frame scenario
- Test termination via the sentinel value (None)
- Test the `fill_silence=False` mode (pass-through, no padding)

### 2.6 Optional Wrapper at the Gateway Layer (Later, Out of Scope for This Refactor)

- If the server needs to push the stream at a realtime cadence, a pacing wrapper can be added in `engine/gateway/`
- Decide whether to apply cadence control to the output based on the session config
- The engine core stays unchanged

---

## 3. Implementation Phases

This goal is an extension of REFACTOR_GOALS.md Phase 1 (establishing the protocol layer), and should be implemented after Phase 1 is complete and before Phase 3 (tests/ cleanup):

1. **Phase 1 extension**: add `realtime.py` under `client/src/qwen3tts/`
2. **Phase 3 prerequisite**: remove `tests/save.py`, switch to `RealtimeAudioStream`
3. **Phase 5 documentation**: update `client/README.md`, adding RealtimeAudioStream usage

---

## 4. Acceptance Criteria

1. ✅ `from qwen3tts import RealtimeAudioStream, TimedAudio` works
2. ✅ `RealtimeAudioStream(session)` produces an isochronous audio stream, automatically padding gaps with silence
3. ✅ When `fill_silence=False`, the behavior is equivalent to calling `iter_messages()` directly
4. ✅ `tests/save.py` has been removed
5. ✅ All client-package unit tests pass
6. ✅ `client/README.md` contains a RealtimeAudioStream usage example
