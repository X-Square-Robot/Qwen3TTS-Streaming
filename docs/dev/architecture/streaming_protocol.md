**English** | [中文](streaming_protocol.zh-CN.md)

# Standalone Protocol Redesign

> Migration status: the primary new public WebSocket protocol is OpenAI
> Realtime at `/v1/realtime`. The `tts-session-v2alpha1` contract documented
> here remains supported for compatibility. See
> [OpenAI Realtime TTS Protocol and Triton Boundary](openai_realtime.md) for
> Realtime events, usage billing, and the Triton sidecar boundary.

## Goals

Unify the standalone `gateway -> interface -> dispatcher -> backend` around an explicit session protocol, so that:

- Transport timing no longer determines synthesis semantics
- Streaming and offline share the same second-layer segmentation logic
- First-layer group pre-splitting is only enabled when the declared input mode requires it
- Future `voice_design`, `custom_voice`, `voice_clone_xvec`, `voice_clone_icl`, `base`, and `instruct` support can be added without changing the session protocol

## Layer Responsibilities

### Gateway

The Gateway acts only as a protocol adapter.

- Receives `start -> text* -> stop/end/cancel`
- Validates and normalizes `SessionConfig`
- Parses the canonical `OutputPolicy`, `VADPolicy`, and `TimingContext`
- Converts output audio to the requested output format via the shared output pipeline
- Does not infer offline/streaming behavior from chunk timing
- Does not own transport-specific protocol branches, only retaining wire-compatibility adaptation

### Interface

The Interface is the transport-agnostic external session facade facing the gateway/triton adapters.

- Located under `engine/interface/*`
- Owns the canonical request/session/event contracts:
  `SessionStartRequest`, `StreamTextChunk`, `SessionEndRequest`,
  `StreamCancelRequest`, `OutputPolicy`, `VADPolicy`, `TimingContext`,
  `StreamEvent`, `AudioFrame`
- Owns the session lifecycle and callback binding
- Owns text/token ingestion and `input_mode`-based routing
- Owns ordered audio emission back to the caller
- Owns timing normalization and output metadata emission
- Does not construct backend requests directly, only delegates to the dispatcher

### Dispatcher

The Dispatcher is the backend-facing request transformer.

- Converts `SegmentAction` into a backend `EngineRequest`
- Emits `NEW_SESSION`, `SESSION_TEXT_DONE`, and cancel/control events
- Preserves segmentation semantics, such as `FLUSH_EOS` vs `FLUSH_NOP`

### Spliter

The Spliter owns the two-layer text segmentation strategy.

- Layer 1: group pre-splitting
  - Used for `LONG_SEGMENT` and `FULL_TEXT`
  - Not used for `TOKEN` / `CLAUSE`
- Layer 2: state-machine-driven segmentation
  - Always the authority for per-segment flushing and decode budget control

### Backend

The Backend owns the synthesis state.

- Prefill/decode execution
- Cache and pause/resume state
- True streaming semantics when text is temporarily unavailable

## Session Protocol

### Canonical Contracts

All external transports now map to a single canonical session contract:

- `SessionStartRequest`
- `StreamTextChunk`
- `SessionEndRequest`
- `StreamCancelRequest`
- `OutputPolicy`
- `VADPolicy`
- `TimingContext`
- `StreamEvent`
- `AudioFrame`

The interface contract version is:

- `protocol_version = tts-session-v2alpha1`

The compatibility boundary is the protocol family plus major version, so the
current compatibility key is `tts-session/v2`. Suffixes such as `alpha1` are
revisions within that major and must remain backward-compatible; changing the
family or moving to `v3` is incompatible. `engine_version` is release
diagnostic metadata and does not participate in wire compatibility.

Current transports remain backward-compatible:

- gRPC legacy `init/text_complete`
- WebSocket `start/text/stop/end/cancel/oneshot`
- Triton legacy JSON request fields
- Remote worker top-level compatibility fields

Missing new fields always default to the existing behavior.

For WebSocket, one physical connection carries one active session at a time
and may carry multiple sessions serially. `stop` and legacy `end` both finish
input and drain output; `cancel` discards queued output. A terminal `done` or
`error` event is the session boundary, so clients must not wait for socket
closure. Persistent gateways add
`event.meta.websocket_connection_reusable="true"` only to safely reusable
successful/cancelled `done` events. Engine errors close the connection; an
absent marker tells a new SDK to fall back safely to a new connection.

WebSocket capabilities also advertise `stream_resume_v1`. A client opts in by
adding a high-entropy, client-generated token to `start.resume`; this token is
independent of the public `session_id` and the private engine execution ID. For
an opted-in session, each output event has a cumulative `delivery_seq`, while
each raw PCM binary frame is preceded by an `audio_header` carrying the same
sequence plus absolute sample bounds. Text uses strictly increasing `seq_no`
values and cumulative `text_ack`; repeated equal sequence/content pairs are
idempotent, while conflicts and gaps are protocol errors. `stop` carries the
final text sequence and is also idempotent.

After an abnormal transport close, a new WebSocket sends the token and its last
complete delivery/sample cursor. The process-local registry fences the old
attachment, preserves the same engine session, and replays only newer records.
The replay log is byte-bounded and expires after the advertised grace period;
failure is explicit rather than silently dropping output or restarting
synthesis. This recovers network/proxy disconnects only. Engine process/GPU
restarts require a new synthesis, and multi-replica deployments need sticky or
token-consistent routing so reconnects reach the instance that owns the state.

### Capability Query

Before opening a synthesis session, the client can call `GetCapabilities`.

This returns the contract the standalone engine has loaded:

- `variant`
- `loaded_model_type`
- `declared_supported_task_types`
- Supported input modes / group policies / audio formats
- ref-audio availability for standalone preprocessing
- Detailed reference preprocessing availability:
  `speaker_encoder_available`, `ref_codec_available`, `icl_available`,
  `ref_audio_max_duration_sec`, `ref_c2w_warm_state_available`,
  `ref_codec_reason`
- Interface contract metadata:
  `protocol_version`,
  `supported_websocket_features`,
  `stream_resume_grace_ms`,
  `stream_resume_max_buffer_bytes`,
  `supported_output_policy_features`,
  `supported_vad_strategies`,
  `supported_timing_fields`

The current canonical capability feature flags are:

- `supported_output_policy_features`
  - `request_context`
  - `timing_context`
  - `vad_policy`
- `supported_vad_strategies`
  - `disabled`
  - `prefix_trim`
  - `tail_guard`
  - `hybrid`
- `supported_timing_fields`
  - Client-provided timestamps, such as `client_request_ts_ms`
  - Server-normalized timestamps, such as `server_first_audio_epoch_ms`
  - Derived latency metrics, such as `server_ttft_ms`

The loaded model type is selected at engine startup. Runtime requests do not switch models; they can only confirm that the client and server use the same model contract.

In standalone mode, the external loaded model type and the backend synthesis branch are related but not identical:

- `base` -> internal `voice_clone` using the x-vector path
- `icl` -> internal `voice_clone` using the ICL path
- `custom_voice` -> internal `custom_voice`
- `voice_design` -> internal `voice_design`

### Start

`StartRequest` declares a `SessionConfig`.

Key fields:

- `task_type`
- `language`
- `speaker`
- `instruct`
- `ref_audio`
- `ref_text`
- `x_vector_only`
- `input_mode`
- `group_policy`
- `audio`
- `output_policy`
- `timing`
- `protocol_version`

Canonical `OutputPolicy` fields:

- `vad_policy`
- `chunk_ms`
- `packet_format`
- `emit_text_events`
- `config`

Canonical `VADPolicy` fields:

- `enabled`
- `strategy`
- `implementation`
- `config`

Canonical `TimingContext` fields:

- `request_id`
- `turn_id`
- `client_request_ts_ms`
- `client_text_ts_ms`
- `client_end_ts_ms`
- `extra`

`task_type` is no longer a runtime model selector. The standalone engine binds the request to the model type loaded in the manifest. The client can omit `task_type`, or send the same value as an explicit handshake check.

The server also validates model-specific fields before synthesis:

- `base`: resolve the reference first; explicit `ref_audio` without `ref_text` is still x-vector-only, explicit `ref_audio + ref_text` enters ICL, and when there is no explicit reference `speaker` is a reference alias
- `icl`: resolve the full reference first; an explicit reference must contain both `ref_audio` and `ref_text`, and when there is no explicit reference `speaker` is a reference alias
- `custom_voice`: `speaker` is a built-in custom voice name; rejects `ref_audio`, `ref_text`, and `x_vector_only`
- `voice_design`: requires `instruct`

No new protocol field was introduced for the reference alias. The meaning of `speaker` depends on the loaded model contract:

- `custom_voice`: a built-in custom voice name, such as `Serena`
- `base` / `icl`: a reference alias when `ref_audio` / `ref_text` are missing

The reference resolution order for `base` / `icl` is:

1. Explicit `ref_audio + ref_text` takes priority. If `speaker` is also present, it is retained only as reference metadata and not used for lookup.
2. If there is no explicit reference and `speaker` is set, the server performs a case-insensitive lookup in `references.entries` of `engine.yaml`.
3. If `ref_audio`, `ref_text`, and `speaker` are all missing, the server uses the default reference. `references.default` is preferred; otherwise the legacy `ENGINE_DEFAULT_BASE_REF_AUDIO_PATH` / `ENGINE_DEFAULT_BASE_REF_TEXT` / `workspace/default_refs/base_ref.wav` mechanism is used.
4. A partial reference is rejected for `icl`. For `base`, `ref_audio` only is still x-vector-only, while `ref_text` only is rejected.

Optional reference library configuration:

```yaml
references:
  default: default
  entries:
    default:
      audio_path: workspace/default_refs/base_ref.wav
      ref_text: 参考音频对应文本
      language: auto
    vivian:
      audio_path: workspace/default_refs/vivian.wav
      ref_text: 这是一段与 vivian 参考音频完全一致的文本。
      language: auto

reference_cache:
  enabled: true
  max_entries: 16
```

For `base` / `icl`, the registry `language` is applied only when the request language is empty or `auto`; an explicit request language takes priority. Explicit `ref_audio + ref_text` does not load the language from the registry, even when `speaker` is also present as reference metadata.

ICL reference preprocessing is intentionally single-request and serialized around the TRT engine. It is not batched, and `spliter.max_concurrent_segments` only affects downstream text segment / EngineLoop slot concurrency. The reference audio hard limit is reported as `ref_audio_max_duration_sec`; the current TRT build defaults to 8 seconds for `speech_tokenizer_codec_fused.engine`.

For standalone ICL preprocessing in TRT mode, the runtime package must contain the TensorRT artifacts:

```text
runtime/speaker_encoder.engine
runtime/speech_tokenizer_codec_fused.engine
```

or the equivalent plan layout:

```text
runtime/speaker_encoder/model.plan
runtime/speech_tokenizer_codec_fused/model.plan
```

The standalone ICL path intentionally does not fall back to ONNX Runtime. If `speech_tokenizer_codec_fused.engine` / `model.plan` is missing, the request fails with `speech_tokenizer_codec_fused_trt_missing`.

Reference metadata is exposed through prefill events/logs:

```text
ref_source
ref_id
ref_audio_sha256
ref_text_hash
icl_cache_hit / icl_cache_miss
ref_preprocess_runtime=trt
```

### Text

`TextChunk` carries only text. Its transport arrival pattern must not change session semantics.

It may also carry optional timing metadata, such as `client_timestamp_ms`, which is recorded as context and does not change synthesis behavior.

### End

`EndRequest` indicates that no more text will arrive for this session.

It must not be used to infer whether the session is "offline" or "streaming."

It may carry an optional `client_timestamp_ms` for timing analysis.

## Input Modes

### TOKEN

- The client sends token-level text updates
- No first-layer group pre-splitting
- The dispatcher forwards the tokenized text straight to second-layer segmentation

### CLAUSE

- The client sends clause-level text updates
- No first-layer group pre-splitting
- The dispatcher forwards the clause text directly to second-layer segmentation

### LONG_SEGMENT

- The client sends long text units
- Each long text unit is first pre-split into groups
- Each resulting group is then fed into second-layer segmentation

### FULL_TEXT

- Explicit offline mode
- The full text is buffered until `end`
- First-layer pre-splitting then runs over the entire text

## Group Policies

### AUTO

- Uses first-layer pre-splitting when `input_mode` is `LONG_SEGMENT` or `FULL_TEXT`

### NONE

- Disables first-layer pre-splitting even for long input units
- Still uses second-layer segmentation

## Audio Output Contract

The Gateway accepts an `AudioFormat` request and converts engine output from the native `PCM_F32@24kHz` to the requested wire format via the shared `engine.interface.output.OutputPipeline`.

The current implementation supports:

- `PCM_F32`, mono, `24000` or `16000`
- `PCM_S16LE`, mono, `24000` or `16000`

The output pipeline is also responsible for:

- Chunk indexing
- First-chunk marking
- start/done event normalization
- Canonical timing metadata
- Shared `protocol_version` and `output_policy_json` / `timing_context_json`
- Transport-agnostic VAD policy exposure

The current timing contract:

- `timing_contract = server_monotonic_v1`

The server is the source of truth for the strong timing metrics:

- `server_request_received_epoch_ms`
- `server_first_audio_epoch_ms`
- `server_done_epoch_ms`
- `server_ttft_ms`
- `server_total_latency_ms`

Client timestamps are optional context only:

- Recorded and echoed back when possible
- May later be used for network latency estimation
- Not treated as strongly consistent metrics

## VAD Contract

This stage does not implement actual output gating in the canonical interface. Instead, it reserves a stable policy contract so that future implementations can be plugged in without changing transport semantics.

Supported semantic policy modes:

- `disabled`
- `prefix_trim`
  - For trimming leading silence only
- `tail_guard`
  - For cutting off a hallucinated non-speech tail
- `hybrid`
  - For combining leading trimming and tail guarding

Important design rule:

- `vad_policy` describes *when* output stream gating should occur
- It does not hard-code *how* detection is implemented

This keeps future implementations compatible:

- Energy-based leading trim
- mel/log-energy leading trim
- TenVad tail guard
- Hybrid combinations

In v1:

- `vad_policy.enabled=false` by default
- The standalone engine contract performs no actual gating
- Enabling a policy only changes metadata/contract fields, unless a subsequent implementation explicitly consumes it

## Transport Mapping

All external adapters should now be thin shells on top of the canonical interface:

- gRPC
  - Maps proto fields to the canonical session contract
  - Retains legacy `init/text_complete`
- WebSocket
  - Maps JSON messages to the canonical session contract
  - Retains binary audio frames for compatibility
- Triton
  - Retains `audio_chunk`, `event_type`, `event_json`, `is_final`
  - Emits canonical metadata in `event_json.meta`
- Remote worker
  - Forwards the canonical `output_policy` / `timing_context`
  - Keeps the existing query/turn compatibility fields

The transport layer should not duplicate:

- Audio conversion logic
- Timing normalization logic
- Protocol version negotiation
- Future VAD contract binding

Unsupported combinations should fail explicitly rather than silently degrade.

## Current Implementation Notes

Already implemented in the standalone engine:

- Explicit `SessionConfig` threaded through the gateway, interface, dispatcher, and backend
- Interface routing by `input_mode`
- The long-segment `push_group_tokens()` path in the `Spliter`
- Backend prefill no longer waits for `text_complete` (if the initial text is already present)
- The `FLUSH_EOS` / `FLUSH_NOP` distinction preserved through to backend requests
- Streaming pause/resume semantics in the backend, rather than unconditional pad injection
- Standalone `base` / `icl` reference resolver, TensorRT-only reference preprocessing, in-process reference feature cache, and ICL reference prefix KV cache

Still to be implemented for full transport parity:

- Full sampling parameter passing

The OpenAI Realtime Triton sidecar and its bidirectional streaming-gRPC backend
are implemented. The legacy SDK transport remains available only as a
compatibility path during migration.
