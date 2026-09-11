**English** | [中文](CHANGELOG.zh-CN.md)

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- The v0.2 stability line substantially suppresses the historical
  runaway-hallucination failure mode on the validated retrained checkpoint.
  The recorded full-bf16 deterministic validation is 0/100; the old 0601
  checkpoint's 10–18% result remains documented as historical evidence.
- Streaming TN now maintains raw Unicode, a mutable tail, primary
  `TextCommit` records, and the `CanonicalTextJournal` before the
  tokenizer/Spliter boundary. WeText, mixed-language routing, explicit
  fallback, and raw-to-spoken provenance share one frontend contract.
- The validated `custom-1.7b` cursor-enabled TRT plan can now consume codec0
  inside the graph and publish native text progress. `qwen.text_progress.v1`
  exposes `native`, `ema`, and `disabled` routes while keeping owner-level
  raw/normalized high-water marks separate from display interpolation.
- Version tags now run wheel-first GitHub and GitLab release pipelines: the
  main repository is checked out without submodules, each forge builds one
  client wheel, GitHub publishes it to its Release, and GitLab publishes it to
  its PyPI Registry with a Release link. The SHA256-identical published file
  is embedded in that forge's engine image.
- Standalone WebSocket connections can now carry multiple logical TTS sessions
  serially. The SDK adds a concurrent connection pool, idle keepalive, stale
  connection probing, new-session reconnects, and a `stop()` alias for `end()`.
  A reusable-`done` capability marker makes mixed-version deployments safely
  fall back to reconnect-per-session; engine-error connections are discarded.
- `TTSClient.connect()` now accepts `key=None`. A non-`None` key injects
  `Authorization: Bearer <key>` for HTTP/WebSocket and lowercase
  `authorization` metadata for gRPC. Authentication remains the deployment
  platform's responsibility; the engine does not validate this header.

### Changed

- The serving performance matrix now sets the SDK WebSocket
  `max_connections` explicitly, so 64/128-stream measurements are not
  serialized by the default 32-connection client pool. The current refresh is
  stored separately from the historical dataset.
- Consolidated the former standalone `webui/` feature showcase into
  `web/packages/demo`, the single version-matched portal served at `/demo/`.
  Playback, SDK, documentation, and the Lab now share that portal;
  `/demo/#/lab` is the only Lab entry, with deep-engineering panels gated by
  the optional backend. `demo_api` remains backend-only and does not serve a
  second frontend.
- Python SDK `transport="auto"` now prefers native `/v1/ws` and retains
  `/v1/realtime` as the OpenAI Realtime compatibility endpoint. Explicit
  endpoints still honor the caller's wire-contract choice, and automatic
  Realtime fallback occurs only when native WebSocket is unavailable.
- `engine-websocket` is no longer classified as an old compatibility
  transport; deprecation warnings now apply only to the older direct
  `engine-grpc`, `triton-grpc`, and `triton-http` paths.

### Fixed

- Split the engine WebSocket handshake budget (`connect_timeout`) from the
  established request receive-idle budget (`timeout`), including auto-detect
  probes, so an unreachable endpoint cannot occupy a streaming worker for the
  full request timeout.
- Close an already-connected WebSocket when the initial streaming `start`
  message fails, preventing descriptor leaks across repeated start failures.
- Retry short WebSocket receive timeouts while waiting for capabilities instead
  of failing on the first 200 ms polling interval.
- Normalize WebSocket send timeouts and socket errors (for example,
  `ECONNRESET`) so streaming callers consistently receive `StreamClosedError`.

## [0.1.0] — Engineering Preview

> ⚠️ **v0.1 engineering preview**: streaming mode may still exhibit hallucination/repetition/dropped reading, and is not recommended for production.
> The recommended path is `custom-1.7b` / `custom_voice`; other variants are experimental. See
> [Known Limitations](docs/user/known_limitations.md) for details.

### Added

- **Streaming TTS inference system**: exports the official Qwen3-TTS PyTorch weights into an ONNX/TensorRT runtime and
  implements streaming inference around a Triton Inference Server / standalone engine.
- **Inference engine** (`engine/`): frontend segmentation, backend inference, gateway, and core scheduling,
  including prefix cache and continuous batching.
- **Lifecycle orchestration** (`scripts/bash/`): `autorun.sh` full pipeline / per-phase (setup → build →
  package → deploy), with support for cross-host builds (probe-target / make-bundle / import-artifact /
  remote-build) and per-submodule mixed-precision TRT builds.
- **Python client SDK** (`client/`, `qwen3-tts-client`): includes the shared protocol layer and transport adapters.
- **WebUI Demo**: an aiohttp backend (`demo_api/`) + a Vite/React frontend (`webui/`).
- **Protocol single source of truth** (`proto/tts.proto`): generated with `make proto`, synced with `make proto-sync`.
- **Documentation**: user documentation (deployment, SDK, benchmark, limitations) and developer documentation (architecture, design, investigation, operations).

> The v0.1 WebUI entry above records the historical standalone layout. Current
> releases use the unified `web/packages/demo` portal; see the Unreleased entry
> for the migration boundary.

### Notes

- The project's own code is released under the [MIT](LICENSE) license, copyright XSquareRobot; upstream Qwen3-TTS
  (the `third_party/` submodule) is Apache-2.0.
- Model weights are obtained via ModelScope/HF and are not distributed with the repository; TensorRT/Triton are not bundled (NVIDIA EULA).
- The reference audio under `resources/speakers/` is synthetic audio with fictional names, containing no real recordings or personal information (see [NOTICE](NOTICE)).

[Unreleased]: https://github.com/X-Square-Robot/Qwen3TTS-Streaming/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/X-Square-Robot/Qwen3TTS-Streaming/releases/tag/v0.1.0
