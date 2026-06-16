# Standalone Protocol Redesign

## Goal

Unify standalone `gateway -> interface -> dispatcher -> backend` around an explicit
session protocol so that:

- transport timing no longer determines synthesis semantics
- streaming and offline share the same second-layer split logic
- the first-layer group pre-split is enabled only when the declared input mode needs it
- future `voice_design`, `custom_voice`, `voice_clone_xvec`, `voice_clone_icl`,
  `base`, and `instruct` support can be added without changing the session protocol

## Layer Roles

### Gateway

The gateway is a protocol adapter only.

- Accept `start -> text* -> end/cancel`
- Validate and normalize `SessionConfig`
- Parse canonical `OutputPolicy`, `VADPolicy`, and `TimingContext`
- Convert outgoing audio to the requested output format through a shared output pipeline
- Do not infer offline/streaming behavior from chunk timing
- Do not own transport-specific protocol branches beyond wire compatibility shims

### Interface

The interface is the transport-agnostic external session facade for gateway/triton adapters.

- Lives in `engine/interface/*`
- Owns canonical request/session/event contracts:
  `SessionStartRequest`, `StreamTextChunk`, `SessionEndRequest`,
  `StreamCancelRequest`, `OutputPolicy`, `VADPolicy`, `TimingContext`,
  `StreamEvent`, `AudioFrame`
- Owns session lifecycle and callback wiring
- Owns text/token ingestion and `input_mode`-based routing
- Owns ordered audio emission back to the caller
- Owns timing normalization and output metadata emission
- Does not construct backend requests directly beyond delegating to dispatcher

### Dispatcher

The dispatcher is the backend-facing request translator.

- Converts `SegmentAction` to backend `EngineRequest`
- Emits `NEW_SESSION`, `SESSION_TEXT_DONE`, and cancel/control events
- Preserves split semantics such as `FLUSH_EOS` vs `FLUSH_NOP`

### Spliter

The Spliter owns the two-layer text segmentation policy.

- Layer 1: group pre-split
  - used for `LONG_SEGMENT` and `FULL_TEXT`
  - not used for `TOKEN` / `CLAUSE`
- Layer 2: state-machine-driven split
  - always authoritative for per-segment flushing and decode budget control

### Backend

The backend owns synthesis state.

- Prefill/decode execution
- cache and pause/resume state
- true streaming semantics when text is temporarily unavailable

## Session Protocol

### Canonical Contract

All external transports now map into a single canonical session contract:

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

Current transports remain backward compatible:

- gRPC legacy `init/text_complete`
- WebSocket `start/text/end/cancel/oneshot`
- Triton legacy JSON request fields
- remote worker top-level compatibility fields

Missing new fields always default to the existing behavior.

### Capabilities

Before opening a synthesis session, a client may call `GetCapabilities`.

This returns the standalone engine's loaded contract:

- `variant`
- `loaded_model_type`
- `declared_supported_task_types`
- supported input modes / group policies / audio formats
- ref-audio availability for standalone preprocessing
- detailed reference preprocessing availability:
  `speaker_encoder_available`, `ref_codec_available`, `icl_available`,
  `ref_audio_max_duration_sec`, `ref_c2w_warm_state_available`,
  `ref_codec_reason`
- interface contract metadata:
  `protocol_version`,
  `supported_output_policy_features`,
  `supported_vad_strategies`,
  `supported_timing_fields`

Current canonical capability feature flags are:

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
  - client-provided timestamps such as `client_request_ts_ms`
  - server-normalized timestamps such as `server_first_audio_epoch_ms`
  - derived latency metrics such as `server_ttft_ms`

The loaded model type is chosen when the engine is started. Runtime requests do
not switch models; they can only confirm that the client and server are using
the same model contract.

In standalone mode, the external loaded model type and the backend synthesis
branch are related but not identical:

- `base` -> internal `voice_clone` with x-vector path
- `icl` -> internal `voice_clone` with ICL path
- `custom_voice` -> internal `custom_voice`
- `voice_design` -> internal `voice_design`

### Start

`StartRequest` declares a `SessionConfig`.

Important fields:

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

`task_type` is no longer the runtime model selector. The standalone engine binds
requests to the already loaded model type from the manifest. Clients may omit
`task_type`, or send the same value as an explicit handshake check.

The server also validates model-specific fields before synthesis:

- `base`: resolves a reference first; explicit `ref_audio` without `ref_text`
  remains x-vector-only, explicit `ref_audio + ref_text` enters ICL, `speaker`
  is a reference alias when no explicit reference is present
- `icl`: resolves a full reference first; explicit reference must include both
  `ref_audio` and `ref_text`, and `speaker` is a reference alias when no
  explicit reference is present
- `custom_voice`: `speaker` is a built-in custom voice name; rejects
  `ref_audio`, `ref_text`, and `x_vector_only`
- `voice_design`: requires `instruct`

No new protocol field is introduced for reference aliases. The meaning of
`speaker` depends on the loaded model contract:

- `custom_voice`: built-in custom voice name such as `Serena`
- `base` / `icl`: reference alias when `ref_audio` / `ref_text` are absent

Reference resolution order for `base` / `icl` is:

1. Explicit `ref_audio + ref_text` wins. If `speaker` is also present, it is
   retained only as reference metadata and is not used for lookup.
2. If no explicit reference is present and `speaker` is set, the server looks
   it up case-insensitively in `engine.yaml` `references.entries`.
3. If `ref_audio`, `ref_text`, and `speaker` are all absent, the server uses
   the default reference. `references.default` is preferred; otherwise the
   legacy `ENGINE_DEFAULT_BASE_REF_AUDIO_PATH` /
   `ENGINE_DEFAULT_BASE_REF_TEXT` / `workspace/default_refs/base_ref.wav`
   mechanism is used.
4. Partial references are rejected for `icl`. For `base`, `ref_audio`-only
   remains x-vector-only, while `ref_text`-only is rejected.

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

For `base` / `icl`, registry `language` is applied only when the request
language is empty or `auto`; an explicit request language wins. Explicit
`ref_audio + ref_text` does not load language from the registry even when
`speaker` is also present as reference metadata.

ICL reference preprocessing is intentionally single-request and serialized
around the TRT engines. It is not batched, and `spliter.max_concurrent_segments`
only affects downstream text segment / EngineLoop slot concurrency. The
reference audio hard limit is reported as `ref_audio_max_duration_sec`; current
TRT builds default to 8 seconds for `speech_tokenizer_codec_fused.engine`.

For standalone ICL preprocessing in TRT mode, the runtime package must contain
TensorRT artifacts:

```text
runtime/speaker_encoder.engine
runtime/speech_tokenizer_codec_fused.engine
```

or the equivalent plan layout:

```text
runtime/speaker_encoder/model.plan
runtime/speech_tokenizer_codec_fused/model.plan
```

The standalone ICL path intentionally does not fall back to ONNX Runtime. If
`speech_tokenizer_codec_fused.engine` / `model.plan` is missing, the request
fails with `speech_tokenizer_codec_fused_trt_missing`.

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

`TextChunk` carries text only. Its transport arrival pattern must not change
session semantics.

It may also carry optional timing metadata such as `client_timestamp_ms`, which
is recorded as context and does not change synthesis behavior.

### End

`EndRequest` means no more text will arrive for this session.

It must not be used to infer whether a session is "offline" or "streaming".

It may carry optional `client_timestamp_ms` for timing analysis.

## Input Modes

### TOKEN

- client sends token-scale text updates
- no first-layer group pre-split
- dispatcher forwards tokenized text to layer-2 splitting immediately

### CLAUSE

- client sends clause-scale text updates
- no first-layer group pre-split
- dispatcher forwards clause text directly to layer-2 splitting

### LONG_SEGMENT

- client sends long text units
- each long text unit is pre-split into groups first
- each resulting group is then fed to layer-2 splitting

### FULL_TEXT

- explicit offline mode
- complete text is buffered until `end`
- then layer-1 pre-split runs across the full text

## Group Policy

### AUTO

- use first-layer pre-split when `input_mode` is `LONG_SEGMENT` or `FULL_TEXT`

### NONE

- disable first-layer pre-split even for long input units
- still use layer-2 splitting

## Audio Output Contract

Gateway accepts an `AudioFormat` request and converts engine output from native
`PCM_F32@24kHz` into the requested wire format through the shared
`engine.interface.output.OutputPipeline`.

Current implementation supports:

- `PCM_F32`, mono, `24000` or `16000`
- `PCM_S16LE`, mono, `24000` or `16000`

The output pipeline is also responsible for:

- chunk indexing
- first-chunk markers
- start/done event normalization
- canonical timing metadata
- shared `protocol_version` and `output_policy_json` / `timing_context_json`
- transport-independent VAD policy exposure

Current timing contract:

- `timing_contract = server_monotonic_v1`

The server is the source of truth for strong timing metrics:

- `server_request_received_epoch_ms`
- `server_first_audio_epoch_ms`
- `server_done_epoch_ms`
- `server_ttft_ms`
- `server_total_latency_ms`

Client timestamps are optional context only:

- they are recorded and echoed where possible
- they may be used later for network-delay estimation
- they are not treated as strong consistency metrics

## VAD Contract

This phase does not implement actual output gating in the canonical interface.
Instead, it reserves a stable policy contract so future implementations can be
plugged in without changing transport semantics.

Supported semantic policy modes:

- `disabled`
- `prefix_trim`
  - intended for trimming leading silence only
- `tail_guard`
  - intended for cutting off hallucinated non-speech tails
- `hybrid`
  - intended for combining prefix trim and tail guard

Important design rule:

- `vad_policy` describes *when* output-stream gating should happen
- it does not hardcode *how* the detection is implemented

That keeps future implementations compatible:

- energy-based prefix trim
- mel/log-energy prefix trim
- TenVad tail guard
- hybrid combinations

In v1:

- default is `vad_policy.enabled=false`
- no actual gating is performed by the standalone engine contract
- enabling a policy only changes metadata / contract fields unless a later
  implementation explicitly consumes it

## Transport Mapping

All external adapters should now be thin shells over the canonical interface:

- gRPC
  - maps proto fields to canonical session contract
  - preserves legacy `init/text_complete`
- WebSocket
  - maps JSON messages to canonical session contract
  - preserves binary audio frames for compatibility
- Triton
  - preserves `audio_chunk`, `event_type`, `event_json`, `is_final`
  - emits canonical metadata inside `event_json.meta`
- remote worker
  - forwards canonical `output_policy` / `timing_context`
  - keeps existing query/turn compatibility fields

The transport layer should not duplicate:

- audio conversion logic
- timing normalization logic
- protocol-version negotiation
- future VAD contract wiring

Unsupported combinations should fail explicitly instead of silently degrading.

## Current Implementation Notes

Implemented in standalone engine:

- explicit `SessionConfig` plumbed through gateway, interface, dispatcher, and backend
- interface routing by `input_mode`
- long-segment `push_group_tokens()` path in `Spliter`
- backend prefill no longer waits for `text_complete` if initial text already exists
- `FLUSH_EOS` / `FLUSH_NOP` distinction preserved into backend requests
- streaming pause/resume semantics in backend instead of unconditional pad injection
- standalone `base` / `icl` reference resolver, TensorRT-only reference
  preprocessing, in-process reference feature cache, and ICL reference prefix
  KV cache

Still pending for full parity with Triton orchestrator:

- complete sampling parameter plumbing
- Triton gateway replacement for the current hand-written gRPC layer
