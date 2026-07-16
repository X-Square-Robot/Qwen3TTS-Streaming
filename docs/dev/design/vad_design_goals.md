**English** | [中文](vad_design_goals.zh-CN.md)

# TTS Output VAD Design Goals

> Branch: `refact`
> Written: 2026-06-16
> Status: **Implemented** (core VAD processor + protocol layer + gateway integration + unit tests)
> Source: consensus and decision record from three rounds of discussion

---

## 0. Problem Background

### 0.1 TTS Model Hallucinations

The Qwen3-TTS model probabilistically produces hallucinations during streaming synthesis, manifesting as:

| Hallucination type | Manifestation | Handled by |
|---------|------|---------|
| Long leading silence | The first 300~500ms of audio is near-silent | VAD trimming |
| Low-energy random noise | Model outputs low-energy but audible noise | VAD trimming |
| High-energy random noise | Model outputs noise with energy but no speech content | VAD trimming |
| Repeated token pattern | Model gets stuck in a 1-2-3-1-2-3 loop | Backend detection |
| Repeated speech segment | Model repeatedly outputs a preceding speech segment | Backend detection |
| Irrelevant speech content | Model outputs speech unrelated to the input text | Resolved on the model side |

**VAD's handling goals**: trim leading silence, intercept low/high-energy random noise.
**Backend's handling goals**: detect repeated token patterns and stop flushing.
**Out of scope for VAD**: repeated speech segments, irrelevant speech content.

### 0.2 The Experience Problem of Leading Silence

At a synthesis rate of 2x realtime, 500ms of leading silence only takes 250ms to synthesize, but the user has to wait 250ms listening to silence before hearing meaningful speech. After VAD trimming, the user no longer hears this silence.

**Important clarification**: VAD cannot reduce time-to-first-word (silence still has to be synthesized before VAD can judge it), but it can improve user experience—users would rather wait in silence than listen to 500ms of silence/background noise. This has significant value in voice interaction scenarios (voice assistants).

### 0.3 Existing Implementations

| Implementation | Location | Capability | Limitation |
|------|------|------|------|
| dBFS prefix trim | `workspace/qwen3_tts_remote.py` `_trim_prefix_silence_locked()` | Removes leading silence | Only does prefix trim, no end detection; cannot intercept hallucinated noise |
| TenVAD (ASR) | `workspace/ten_vad.py` | Full ASR VAD | Multi-segment speech state machine designed for ASR, unsuitable for the TTS scenario |
| Backend pad silence | `engine/backend/engine_loop.py` `_is_pad_silence()` | Protects the backend from exceeding the max step | Stops decoding on the generation side, not gating on the output side |

---

## 1. Design Goals

### G1: Unified VAD Protocol

All VAD modes share the same parameter interface:

```
mode:           "disabled" | "energy" | "tenvad"
chunk_ms:       per-frame duration (mode config, energy=16, tenvad=16)
begin_threshold: 0.0~1.0 float (VAD maps internally to its own units)
begin_count:    N consecutive frames above begin_threshold triggers begin
end_threshold:  0.0~1.0 float
end_count:      N consecutive frames below end_threshold triggers end
start_margin_ms: milliseconds to look back after begin triggers
```

**Unit mapping** (VAD internal implementation, uniformly exposed as 0~1):

| Mode | begin/end_threshold 0~1 mapping | Typical begin | Typical end |
|------|------------------------------|-----------|---------|
| energy | Linearly mapped to a dB scale (e.g. 0→-80dB, 1→0dB) | ~0.3 (-56dB) | ~0.2 (-64dB) |
| tenvad | Used directly as a probability threshold | ~0.6 | ~0.35 |

### G2: Three VAD Modes

| Mode | Implementation | Capability | Typical scenario |
|------|------|------|---------|
| `disabled` | Pass-through, no trimming | Fully reflects the model's behavior | Debugging/baseline |
| `energy` | Pre-emphasis + Hamming window + log energy + dB-scale threshold | Removes absolute silence; more sensitive to unvoiced consonant onsets | Lightweight trimming |
| `tenvad` | TenVad ONNX inference + TTS-specific state machine | Intercepts silence + random noise; better preserves normal pauses | Recommended for production |

**Fusion of dBFS and log energy**: unified into `energy` mode, using log energy internally (pre-emphasis + windowing + log) while exposing dB-scale thresholds externally. Whether pre-emphasis is enabled is controlled by an internal parameter.

### G3: Streaming Computation

- Emit as far as computation has progressed
- VAD processes at 16ms per frame, while TTS produces ~80ms chunks
- 100ms of pending audio → 6 frames × 16ms = 96ms decidable → emit up to 96ms
- begin introduces very low latency (begin_count × chunk_ms ≈ 80ms, less than a TTS chunk)
- end accumulates consecutively; once end triggers, emission stops

### G4: VAD Is Transparent to the Client

- VAD only performs audio gating (emit/hold/discard) and does not send begin/end events to the client
- The client receives a continuous PCM stream that may be missing the trimmed pauses/noise
- Effectively, VAD skips the noise/meaningless long silence on the user's behalf

### G5: Hysteresis Design

- begin_threshold > end_threshold (e.g. 0.6 > 0.35)
- begin_count is small (~5 frames = 80ms), end_count is large (~31 frames = 500ms)
- High begin threshold + small count → quickly confirm speech onset
- Low end threshold + large count → protect normal pauses/breathing from being trimmed by mistake

### G6: Re-begin After End

The VAD state machine supports multiple begin→end cycles:

```
SILENCE → (begin) → SPEECH → (end) → SILENCE → (begin) → SPEECH → ... → flush
```

Example scenario:
```
[silence 500ms] [speech 2s] [hallucinated noise 0.5s] [silence 0.5s] [speech 1s] [silence 300ms]
→ after VAD trimming →
[speech 2s] [hallucinated noise 0.5s] [speech 1s]  (silence trimmed; part of the noise is emitted within end_count)
```

---

## 2. Architecture Design

### 2.1 The Position of VAD in the Engine

```
EngineLoop (GPU thread)
    ↓ EngineResult(AUDIO_CHUNK)
asyncio result_queue
    ↓
VAD (asyncio side, per-session stateful streaming processor)
    ↓ gated audio
OutputPipeline
    ↓
Transport (gRPC / WebSocket)
```

**Reasons for choosing the asyncio side**:
- engine_loop is a GPU thread and should not do CPU-intensive VAD computation (TenVAD is ONNX inference)
- On the asyncio side the audio is already CPU-side bytes, so VAD does not affect the GPU thread
- The asyncio side already has timing logic and can integrate with OutputPipeline metrics
- OutputPipeline is a pure data-transformation pipeline and should not become a stateful pipeline

### 2.2 The VAD State Machine

Redesigned for the TTS scenario, not reusing the ASR multi-segment speech state machine:

```
                    ┌─────────────────────────────┐
                    │                             │
                    ▼                             │
┌──────────┐  begin triggers  ┌──────────┐  end triggers  ┌──────────┐
│ SILENCE  │ ──────────→ │ SPEECH   │ ────────→ │ SILENCE  │
│ (no emit)│             │ (emit)   │           │ (no emit)│
└──────────┘             └──────────┘           └──────────┘
     ▲                                                 │
     └─────────────── begin triggers ──────────────────┘

Any state + flush signal → emit all pending audio → reset
```

**Key differences from ASR VAD**:
- No pre_roll look-back retention (leading silence in TTS is exactly what should be dropped)
- No too-short-segment discard logic
- No multi-segment start/end event publishing
- Unified flush signal handling (SESSION_DONE / ERROR / CANCEL)

### 2.3 Buffer Design

VAD needs three buffers:

#### (a) Input frame alignment buffer

The AUDIO_CHUNK produced by TTS is not necessarily an integer multiple of a 16ms frame (24kHz × 16ms = 384 samples).
The residual samples of an incomplete frame need to be held and concatenated with the next chunk.

```
chunk arrives → concat to input_buffer → split into frames → feed frames to VAD one by one → residual returned to input_buffer
```

#### (b) Start margin retention buffer

After begin triggers, we need to look back start_margin_ms (~20ms) of audio. In the SILENCE state,
the most recent start_margin_ms of audio must be held—neither emitted nor discarded—until begin is confirmed.

```
SILENCE state: each frame of audio → hold in margin_buffer (ring buffer, capacity ≥ start_margin_ms)
begin triggers: take start_margin_ms of audio from margin_buffer + the current frame → emit together
```

**Constraint**: margin_buffer capacity < TTS chunk size (80ms), so the introduced latency is negligible.

#### (c) Pending-emit buffer

In the SPEECH state, each frame judged to be speech is held here and emitted in batches to reduce system calls.
When end triggers, the un-emitted audio in the buffer is discarded.

### 2.4 Energy VAD Implementation

```python
def compute_energy_score(frame_int16: np.ndarray, *, preemphasis: float = 0.97) -> float:
    """Pre-emphasis → Hamming window → log energy → dB scale → normalize to 0~1"""
    # 1. Pre-emphasis: y[n] = x[n] - a * x[n-1]
    # 2. Hamming window
    # 3. energy = sum(y^2)
    # 4. dB = 10 * log10(energy / (N * 32768^2) + eps)  -- absolute reference, independent of frame size
    # 5. Normalize: score = (dB + 80) / 80  -- -80dB→0, 0dB→1
    ...
```

**Advantages of the dB scale**:
- The threshold value is independent of frame size (the user sets it once)
- It has an absolute reference point (0 dB = full scale)
- The effect of pre-emphasis is merely to make unvoiced consonants "look louder" on the dB scale

### 2.5 TenVAD Implementation

Reuse only the TenVAD inference core (`TenVad.process()` → probability + flags),
and rewrite the state machine logic around the TTS scenario.

**TenVAD inference parameters**:
- hop_size = 256 (16ms @ 16kHz)
- threshold = 0.5 (model-internal threshold)
- RTF ≈ 0.015 (extremely low overhead)

**TenVAD input**: 16kHz int16 PCM. The VAD is internally responsible for downsampling the 24kHz raw PCM to 16kHz.

**TenVAD's core advantage over energy mode**: it better distinguishes "low-energy normal pauses/breathing" from "low-energy noise,"
because although pauses/breathing have low energy, TenVAD may assign them a higher probability, making it less prone to false end triggers.

---

## 3. Protocol and Configuration

### 3.1 VAD Configuration

```python
@dataclass
class VADConfig:
    enabled: bool = False
    mode: str = "disabled"      # "disabled" | "energy" | "tenvad"
    chunk_ms: int = 16          # per-frame duration
    begin_threshold: float = 0.6  # 0~1
    begin_count: int = 5        # N consecutive frames above begin_threshold triggers begin
    end_threshold: float = 0.35  # 0~1
    end_count: int = 31         # N consecutive frames below end_threshold triggers end (~500ms)
    start_margin_ms: int = 20   # look back after begin
```

### 3.2 Per-Mode Default Parameters

| Parameter | energy default | tenvad default | Note |
|------|-----------|-----------|------|
| chunk_ms | 16 | 16 | Coincidentally identical, no actual correlation |
| begin_threshold | 0.3 | 0.6 | energy maps to ~-56dB |
| begin_count | 5 | 5 | ~80ms |
| end_threshold | 0.2 | 0.35 | energy maps to ~-64dB |
| end_count | 31 | 31 | ~500ms |
| start_margin_ms | 20 | 20 | look back 20ms |

### 3.3 Observability

VAD needs to inject trimming information into `done_meta`:

```python
meta["vad_mode"] = "energy"  # or "tenvad"
meta["vad_prefix_trimmed_ms"] = "520.000"   # duration of leading silence trimmed
meta["vad_tail_trimmed_ms"] = "0.000"       # duration trimmed from the tail
meta["vad_original_audio_ms"] = "3000.000"  # total duration of the original audio
meta["vad_effective_audio_ms"] = "2480.000" # effective audio duration after trimming
meta["vad_begin_count"] = "1"               # number of begin triggers
meta["vad_end_count"] = "0"                 # number of end triggers (excluding flush)
```

**Duration semantics**:
- Engine-side RTF = original audio duration / elapsed time (unchanged)
- Client-side RTF = effective audio duration / elapsed time
- The client's `audio_duration` should be based on the effective audio duration after trimming

---

## 4. Key Design Decisions

### 4.1 Division of Responsibility Between Backend and VAD

| Layer | Responsibility | Detection means | Action |
|----|------|---------|------|
| **Backend** (engine_loop) | Detect repeated token patterns | Token sequence pattern matching (1-2-3-1-2-3) | Stop flushing, send SEGMENT_END/ERROR |
| **Backend** (engine_loop) | Protect the KV budget | pad_silence detection + dynamic_silence_limit | Stop decoding |
| **VAD** (asyncio side) | Trim leading silence | begin threshold + begin count | Do not emit silence frames |
| **VAD** (asyncio side) | Intercept hallucinated noise | end threshold + end count | Stop emitting, wait for re-begin |
| **VAD** (asyncio side) | Preserve normal pauses | Sufficiently large end_count / TenVAD probability discrimination | Do not trim pauses by mistake |

**Evolution of `SchedulerConfig.pad_silence_*`**: currently used to protect the backend from exceeding the max step.
This iteration evolves it so that the backend focuses on detecting repeated token patterns, while pad_silence detection is kept as a safety fallback
(preventing the model from getting stuck in a repeated-silence-output pattern that would exceed max_seq_len).

### 4.2 Flush Signal Mechanism

When the audio stream terminates (whether normally or abnormally), VAD needs to receive a flush signal and immediately emit all pending audio:

| Signal source | Signal type | VAD behavior |
|---------|---------|---------|
| SESSION_DONE | Normal completion | Process all pending audio → flush → reset |
| SEGMENT_END | Segment end | Same as above |
| ERROR | Abnormal termination | Immediately flush all pending (no end judgment) → reset |
| CANCEL_SESSION | Manual cancellation | Discard all pending → reset |

**Residual frame handling on SESSION_DONE**:
1. Process all complete frames in the input buffer
2. When the residual frame is incomplete, zero-pad it to a full frame and process it
3. Flush all audio in the pending-emit buffers (regardless of the current VAD state)
4. Reset the VAD state

### 4.3 Difference in End Behavior Between Energy Mode and TenVAD Mode

**Energy mode**: cannot distinguish "low-energy normal pauses" from "low-energy noise." end simply accumulates whenever the energy is below threshold,
so normal breathing/pauses are also counted toward end_count. Therefore end_count must be set larger in energy mode (e.g. 1000ms),
but this in turn makes it unable to intercept long hallucinated noise. **This is an inherent limitation of energy mode.**

**TenVAD mode**: although pauses/breathing have low energy, TenVAD may assign them a higher probability (pauses/breathing have speech characteristics),
making it less prone to false end triggers. Therefore TenVAD can use a smaller end_count (e.g. 500ms) while both preserving pauses and blocking noise.
**This is TenVAD's core advantage.**

**Recommendation**: prefer TenVAD mode in production. Energy mode serves as a lightweight alternative (no ONNX dependency),
suitable for scenarios that are not sensitive to pause preservation.

### 4.4 Impact of Post-VAD Audio Duration

After VAD trimming, the total duration of audio emitted to the client becomes shorter. Impacts:

1. **Playback-side timing**: if the client uses `audio_duration / sample_rate` to decide when to request the next segment,
   the trimmed duration will cause the client to request the next segment too early. The client should base this on the effective audio duration.
2. **timing metadata**: `done_meta` must distinguish the original duration from the effective duration (see 3.3).
3. **RTF semantics**: engine-side RTF = original audio duration / elapsed time (unchanged),
   client-side RTF = effective audio duration / elapsed time.

---

## 5. Implementation Notes

### 5.1 VAD Processor Interface

```python
class TTSVADProcessor:
    """Per-session streaming VAD processor."""

    def __init__(self, config: VADConfig, sample_rate: int = 24000): ...

    def process_chunk(self, pcm_int16: np.ndarray) -> np.ndarray:
        """Take one PCM chunk as input, return the audio to be emitted (possibly empty)."""
        ...

    def flush(self) -> np.ndarray:
        """The audio stream has ended, return all pending audio."""
        ...

    def reset(self) -> None:
        """Reset state (called when the session ends)."""
        ...

    @property
    def trimmed_ms(self) -> float: ...
    @property
    def original_audio_ms(self) -> float: ...
    @property
    def effective_audio_ms(self) -> float: ...
    @property
    def begin_count(self) -> int: ...
    @property
    def end_count(self) -> int: ...
```

### 5.2 Internal Downsampling for TenVAD Mode

TenVAD requires 16kHz input, but the engine's raw PCM is 24kHz. VAD downsamples internally:

```python
# Simple 3/2 downsampling (24kHz → 16kHz)
# Every 3 samples at 24kHz → 2 samples at 16kHz
# Use linear interpolation or a simple FIR filter
```

### 5.3 Per-Mode Difference in begin_count

| Mode | Suggested begin_count | Rationale |
|------|-----------------|------|
| energy | 5 frames (80ms) | Energy judgment is stable; 1-2 frames suffice, 5 leaves headroom |
| tenvad | 5 frames (80ms) | TenVAD single-frame probability may jitter; 2-3 frames are needed to confirm, 5 leaves headroom |

### 5.4 The end_count Trade-off

end_count is the core trade-off between **responsiveness vs. effectiveness**:

- Large end_count → protects normal pauses, but more hallucinated noise may be emitted
- Small end_count → blocks more hallucinated noise, but may cut off normal pauses

**Determine the optimal value through A/B testing with real hallucination samples after implementation.** Initial suggestions:
- energy mode: end_count = 62 frames (~1000ms), because energy mode cannot distinguish pauses from noise
- tenvad mode: end_count = 31 frames (~500ms), because TenVAD better preserves pauses

---

## 6. Items to Validate

| # | Item | Validation method |
|---|------|---------|
| 1 | CPU overhead of TenVAD ONNX under TTS realtime constraints | Benchmark: P99 latency of 24kHz→16kHz downsampling + TenVAD process() |
| 2 | Optimal end_count | A/B test: measure noise interception rate and pause preservation rate for different end_count values using real hallucination samples |
| 3 | Energy-mode threshold mapping | Record real TTS output, gather the dB distribution of silence/speech/noise frames, calibrate begin/end thresholds |
| 4 | Whether start_margin is sufficient | Test the VAD detection latency for unvoiced consonant onsets (f/s/sh, etc.), confirm that a 20ms look-back does not clip audio |
| 5 | Downsampling quality | Compare the effect of 24kHz→16kHz linear interpolation vs. FIR filtering on TenVAD probability |
| 6 | Post-VAD client behavior | Verify that the client's playback/request logic based on effective audio duration works correctly |

---

## 7. Out of Scope for This Iteration

- Backend repeated token pattern detection (1-2-3-1-2-3)—a separate task; this iteration evolves the `pad_silence_*` responsibility. *(Since implemented, 2026-07: `scheduler.token_loop_abort_frames` aborts a segment after N consecutive identical codebook-0 tokens, `eos_reason=loop_abort`. Empirically hallucination loops are period-1 on codebook-0, not 1-2-3.)*
- Root-cause fix of hallucinations on the model side
- VAD training/fine-tuning
- Client-side multi-segment audio concatenation logic (VAD is transparent to the client, so this is not needed)

---

## 8. References

- `workspace/ten_vad.py`: existing ASR TenVAD implementation (reference for the inference core, state machine not reused)
- `workspace/qwen3_tts_remote.py`: existing dBFS prefix trim implementation (reference for energy computation, to be replaced)
- `engine/backend/engine_loop.py`: Backend pad silence detection (reference for the division of responsibility)
- `engine/interface/output.py`: OutputPipeline (VAD observability integration point)
- `engine/core/types.py`: VADConfig definition (protocol extension point)
- [TEN VAD GitHub](https://github.com/TEN-framework/ten-vad): reference for the TenVAD inference core
- [streaming_hallucination.md](../investigation/streaming_hallucination.md): hallucination investigation background
