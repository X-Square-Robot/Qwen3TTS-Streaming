**English** | [中文](realtime_api.zh-CN.md)

# Realtime endpoints and events

Use the Python or Browser SDK for normal application work. This page is for implementing a client in
another language, inspecting gateway events, or integrating a custom player.

## Public endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/capabilities` | Discover tasks, audio formats, and extensions on this deployment |
| WebSocket | `/v1/realtime` | Realtime endpoint for all new clients |
| `GET` | `/demo/` | Playground, SDK downloads, and same-release documentation |

Use WSS when the HTTP origin uses HTTPS. When a reverse proxy adds a path prefix, resolve relative
URLs from Demo configuration or capabilities instead of rebuilding paths from the origin.

## Connect

After `/v1/realtime` opens, the server sends `session.created`. Send `session.update` to select the
model, voice, audio format, VAD, and delivery policy. Wait for `session.updated` before creating a
response.

A physical connection can be reused for sequential responses, but currently permits one active
response at a time. Use multiple pooled connections for concurrent synthesis.

## Full-text request

```json
{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Hello, world."}]}}
{"type":"response.create"}
```

The main server event sequence is:

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

`response.output_audio.delta.delta` is Base64-encoded mono PCM16. Read the sample rate from session
configuration and capabilities. A delta does not have a fixed duration.

## Incremental-text request

OpenAI Realtime has no standard event for appending to a text item after generation begins. This
service uses a namespaced extension while keeping output in standard Realtime events:

```json
{"type":"response.create"}
{"type":"qwen.input_text_buffer.append","sequence":1,"text":"Hello, "}
{"type":"qwen.input_text_buffer.append","sequence":2,"text":"world."}
{"type":"qwen.input_text_buffer.commit"}
```

`sequence` starts at 1 and increases strictly. Retrying an identical sequence and text is idempotent;
a gap or different text for the same sequence produces an `error`. Handle
`qwen.input_text_buffer.ack` and retain unacknowledged input until the response is terminal.

## Audio cursors and playback acknowledgement

Audio events carry continuous `qwen_output_sample_start` and `qwen_output_sample_end` cursors. Reject
overlaps or gaps to prevent duplicate playback. With guarded delivery, a player reports progress:

```json
{"type":"qwen.playback.ack","response_id":"resp_...","played_through_sample":"24000","buffered_through_sample":"28800"}
```

Receiving audio, placing it in a device buffer, and having the speaker consume it are different
states. Only device consumption should advance `played_through_sample`.

## Terminal states, usage, and errors

Every response ends with `response.done`; `response.status` is `completed`, `cancelled`, or `failed`.
Clients should:

1. Correlate application logs with `response_id`.
2. Receive through `response.done`; a closed WebSocket is not a successful terminal state.
3. Read token usage from `response.done.response.usage`.
4. Read `status_details.error.code` and `message` when status is `failed`.
5. Send `response.cancel` for user cancellation, then wait for the cancelled terminal state.

Protocol errors normally arrive as `error` events. A request error does not always make the physical
connection reusable; SDKs already make that decision from terminal state and capabilities. Custom
clients need an equivalent state machine.

## Recovery

An active response is recoverable only when capabilities advertise `qwen.response_resume.v1`.
Recovery needs the server token, last acknowledged delivery sequence, received audio sample cursor,
and unacknowledged text sequences. A server restart, expired token, replay-window overflow, or routing
to another replica must fail explicitly; silently restarting synthesis would duplicate audio.
