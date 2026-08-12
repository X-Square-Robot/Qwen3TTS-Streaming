**English** | [中文](openai_realtime.zh-CN.md)

# OpenAI Realtime TTS Protocol and Triton Boundary

## Decision

New clients should use `/v1/realtime` as the primary endpoint. The standalone
deployment uses `ws://<host>:50052/v1/realtime`; the Triton compose deployment
uses `ws://<host>:50053/v1/realtime` by default.
`/v1/ws`, legacy gRPC, and the Triton JSON action protocol remain available
during migration, but they are no longer the design center for new public API
features.

OpenAI Realtime is a full-duplex transport. The gateway continues reading
client events while audio is flowing downstream, allowing concurrent input and
`response.cancel`. The engine currently permits one active `response` per
connection and allows serial responses on the same socket. That is an execution
concurrency limit, not half-duplex transport.

## Text Input

Complete text uses standard Realtime events:

```json
{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Hello world."}]}}
{"type":"response.create"}
```

Realtime does not currently define a client event that appends text to an item
that is already being synthesized. Token-level ingress therefore uses one
narrow namespaced extension, while response, audio, error, and usage events
remain standard:

```json
{"type":"response.create"}
{"type":"qwen.input_text_buffer.append","sequence":1,"text":"Hello "}
{"type":"qwen.input_text_buffer.append","sequence":2,"text":"world."}
{"type":"qwen.input_text_buffer.commit"}
```

Sequences start at 1 and increase strictly. Retrying the same sequence and text
is idempotent. A gap or conflicting retry returns `error` without closing the
physical connection. Input acknowledgements are
`qwen.input_text_buffer.ack` and `qwen.input_text_buffer.committed`.

The main server event order is:

```text
response.created
  -> response.output_item.added
  -> response.content_part.added
  -> response.output_audio.delta *
  -> response.output_audio.done
  -> response.content_part.done
  -> response.output_item.done
  -> response.done
```

Audio deltas contain Base64 mono PCM16. The current supported rates are 16 kHz
and 24 kHz; 24 kHz is recommended.

### Qwen text-progress extension

Alongside standard audio events, the server emits these namespaced Realtime
data-channel events:

```text
qwen.text_token
qwen.text_boundary_commit
qwen.text_progress
```

`qwen.text_progress` is intentionally coarse in the current implementation;
it is not an ASR or phoneme-level alignment. Its `meta` includes
`progress_basis=ema_frame_ratio_v1`, `progress_quality=rough`, source-audio
frame and text-token ranges, `text_progress`, and `progress_final`. The gateway
also adds `output_sample_end`, the number of PCM samples sent so far. Clients
should use this as a UI estimate and reconcile it with their local playback or
buffer clock. A future ASR or codec-text aligner can keep the transport fields
stable while changing `progress_basis` and the estimator implementation.

## Usage and Billing

Usage is returned in `response.done.response.usage` and emitted once as a
server-side `billing.usage` lifecycle record. Production deployments should
connect the injectable `usage_recorder` to a durable ledger rather than relying
on whether the client receives the terminal event.

The Triton sidecar enables the JSONL recorder by default and bind-mounts it as
`workspace/realtime_usage/realtime_usage.jsonl`. This is an append-only handoff
ledger, not an invoice system; downstream billing should ingest and deduplicate
by `response_id`.

- Input text tokens use the model tokenizer for synthesis text, instructions,
  and reference text; they are not character estimates.
- Output audio tokens are based on PCM actually emitted: one token per 50 ms,
  rounded up for a final partial interval.
- Cancelled and failed responses retain partial usage.
- Local prefix KV cache hits are execution optimizations, not OpenAI cached-token
  billing semantics, so cached tokens are currently zero.

## Triton Compatibility

Triton should not terminate the public OpenAI WebSocket. The intended topology
is:

```text
OpenAI Realtime client
        <=> WebSocket /v1/realtime
Realtime gateway sidecar
        <=> RealtimeSessionBackend
        <=> bidirectional gRPC / Triton decoupled stream
Triton tts_orchestrator -> shared engine scheduler
```

Decoupled Triton streaming requires gRPC; Triton HTTP cannot carry this
long-lived bidirectional session. The gateway owns OpenAI event state, Base64
PCM, authentication, cancellation, and billing. Triton owns internal
`start / append / complete / cancel` operations and streams audio/events back.

`RealtimeSessionBackend` is the code boundary for this split.
`EngineRealtimeBackend` delegates to the in-process `TTSEngine`, while the
implemented `TritonRealtimeBackend` owns one Triton streaming-gRPC client stream
per active Realtime response. The `init` request owns the decoupled response;
`append_text`, `text_complete`, and `cancel` travel on the same gRPC stream while
audio flows back. Public IDs never become Triton execution keys.

For billing, the sidecar loads the tokenizer from the exact mounted model
package version and counts input locally. This avoids another model request and
keeps token accounting stable across Triton replicas. Output audio usage remains
based on PCM samples actually emitted by the gateway.

The old Triton JSON actions can therefore remain an internal adapter during the
migration and later be replaced without becoming the new public protocol.

## Migration Order

1. Implemented: add `/v1/realtime`; keep `/v1/ws` and the old SDK available.
   Capabilities advertise both protocol families.
2. Implemented: make Realtime the preferred SDK auto-detection result; retain
   legacy transports as explicit fallback with deprecation warnings.
3. Implemented: add the bidirectional Triton backend adapter and make JSON
   actions internal.
4. Announce a legacy removal release only after durable usage, auth, quotas,
   and reconnect behavior pass acceptance tests.

Steps 1–3 are implemented. Step 4 remains the acceptance gate: the compatibility
transports stay available until durable usage, auth, quotas, and Realtime
reconnect behavior are production-ready and a removal release is announced.
