**English** | [中文](known_limitations.zh-CN.md)

# Known Limitations and Risks

This document is the risk statement for the current open-source preview release. Before release, keep it consistent with the README, WebUI, and demo API.

## Release Positioning

The current project is an **engineering preview / research preview**, not a production-stable release. It primarily demonstrates Qwen3-TTS engineering optimizations in TensorRT, model fusion, token-level streaming scheduling, prefix cache, and frontend segmentation.

Recommended v0.1 stable scope:

- Model: `custom-1.7b`
- Task: `custom_voice`
- Deployment: standalone engine / Triton TRT streaming
- WebUI: performance showcase, trace replay, live TRT/engine comparison

## Model Path Status

| Path | Status | Risk |
| --- | --- | --- |
| `custom-1.7b` / `custom_voice` | v0.1 recommended path | Still needs continued stress testing of streaming stability, long text, concurrency, and different speakers |
| `design-1.7b` / `voice_design` | Experimental | Some code paths exist, but there is no thorough end-to-end validation |
| `base` x-vector voice clone | Planned | ref audio preprocessing, speaker embedding injection, and end-to-end validation are not complete |
| `icl` voice clone | Planned | The ref audio / ref text / ref code pipeline is not fully wired up |
| `0.6b` variants | Not the v0.1 main line | Requires independent validation of export, profile, quality, and speed |

## Streaming Stability

The current streaming mode may still exhibit:

- Hallucination: generating content the user did not input. **This risk is strongly checkpoint-dependent, and this project does not ship model weights** — one checkpoint we tested ran away (never emitting EOS) on ~10-18% of sampling seeds, while another measured 0/100 on the same deterministic seed set (see `docs/dev/investigation/streaming_hallucination.md` for the methodology). Whatever weights you bring, treat streaming hallucination as a live risk until you have validated your own checkpoint; the engine's runaway defenses (512-step cap, token loop guard `scheduler.token_loop_abort_frames`, transparent reseed-rerun of hallucinated lookahead segments `scheduler.token_loop_max_retries`, VAD gating) are always on. Streaming clients can additionally opt into guarded delivery (`output_policy.config: {"delivery": "guarded"}`): the server holds synthesized-ahead audio in a playhead-relative window and discards condemned hallucination tails before they are ever sent.
- Repetition: local words, phrases, or audio segments repeating.
- Dropped reading: skipping part of the input text.
- Insertion: inserting extra words at pauses or across segments.
- Long-text degradation: stability decreases as context and KV grow.
- Segmentation boundary anomalies: text with punctuation, numbers, English, or mixed Chinese-English may trigger suboptimal splitting.

These issues mean the current release is not suitable for direct use in production broadcasting, customer service, medical, financial, legal, or content-safety scenarios that require strong consistency.

## Performance Number Limitations

The single-stream TTFT figure (measured server TTFT 14.9 ± 0.3ms; see [benchmark methodology](benchmark_methodology.md)) is not a universal guarantee. It usually requires all of the following to hold simultaneously:

- The engine is already warmed up.
- prefix/cache hit.
- Single-stream request or low contention.
- Fixed hardware, fixed TensorRT profile, fixed dtype.
- Local or low-network-overhead link.

`128-stream avg TTFT` must also carry complete test conditions, including hardware, driver, NGC image, engine profile, input text, cache mode, sampling parameters, client measurement method, and failure rate.

## WebUI Data Sources

The WebUI has two data sources:

- fixture trace: offline JSON replay, suitable for showcasing the UI and aligning metric fields; it does not represent a live service.
- live measurement: real-time calls to the standalone engine, Triton, or the official PyTorch API.

Only when the result's `source` is `live_triton`, `live_engine`, or `live_official_pytorch` does it represent a live measurement. Default fixture values need to be re-collected on real hardware before release.

## Deployment Limitations

- The TensorRT engine is tightly coupled to the TensorRT runtime version. After switching the NGC image, TensorRT version, or driver, it is recommended to rebuild the engine.
- The runtime `max_batch`/`max_seq_len` cannot exceed the `engine_profile` recorded in the manifest, otherwise startup fails immediately.
- The regular `engine-docker` image is suitable for fixed-code deployment; during development, use `compose.sh --dev` or `compose.sh watch` to avoid frequently rebuilding the image.
- The current containers do not cover production operations capabilities such as K8s, canary releases, authentication, rate limiting, or multi-tenant isolation.

## User Notices That Must Be Kept Before Release

The README, WebUI, and release notes must make clear:

- This project is an engineering preview.
- The v0.1 recommended path is `custom-1.7b`.
- base/ICL/voice design should not be advertised as stable and ready to use.
- Streaming TTS carries risks of hallucination and long-text instability. Hallucination severity is checkpoint-dependent and this project ships no weights — users must validate their own checkpoint (see Streaming Stability above).
- Benchmark numbers must be accompanied by complete conditions and must not be written as unconditional performance guarantees.
