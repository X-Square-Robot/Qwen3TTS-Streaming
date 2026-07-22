**English** | [中文](CHANGELOG.zh-CN.md)

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

### Notes

- The project's own code is released under the [MIT](LICENSE) license, copyright XSquareRobot; upstream Qwen3-TTS
  (the `third_party/` submodule) is Apache-2.0.
- Model weights are obtained via ModelScope/HF and are not distributed with the repository; TensorRT/Triton are not bundled (NVIDIA EULA).
- The reference audio under `resources/speakers/` is synthetic audio with fictional names, containing no real recordings or personal information (see [NOTICE](NOTICE)).

[Unreleased]: https://github.com/X-Square-Robot/Qwen3TTS-Streaming/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/X-Square-Robot/Qwen3TTS-Streaming/releases/tag/v0.1.0
