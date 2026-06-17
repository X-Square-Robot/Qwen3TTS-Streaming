# Script Guide

## Where To Start

- **`scripts/bash/autorun.sh`**: primary operator entry point (setup → build → package → deploy)
- **`scripts/python/audit_tooling_surface.py`**: governance report for scripts/ + tests/ surface
- **`tests/tools/serving_endpoints.py`**: canonical serving acceptance and benchmark tool
- **`tests/README.md`**: main map for pytest suites and manual validation tools

## Directory Roles

| Location | Role | Audience |
| --- | --- | --- |
| `scripts/bash/` | lifecycle and operator workflows (14 files) | users deploying or packaging the project |
| `scripts/bash/lib/` | shared shell library code | maintainers extending shell flows |
| `scripts/export/` | model export implementation (01–09) | maintainers working on ONNX / TRT export |
| `scripts/python/qwen3tts_tools/` | shared Python helper layer | maintainers adding or refactoring CLIs |
| `scripts/python/` | build/config/manifest tools (9 scripts) | advanced developers and maintainers |
| `scripts/compose/` | container entrypoints | deployment maintainers |
| `scripts/demo/` | demo launchers | demo users |

## scripts/python/ Inventory

| Script | Purpose |
|--------|---------|
| `audit_tooling_surface.py` | Governance audit for scripts/ + tests/ surface |
| `build_talker_code2wav_fused_trt_host.py` | Build fused TRT engine on host |
| `codec_embedding_sum.py` | Codec embedding sum utility |
| `generate_triton_configs.py` | Generate Triton model repository configs |
| `raw_websocket.py` | Minimal RFC6455 WebSocket client helpers |
| `triton_manifest_io.py` | Read/write triton_manifest.json |
| `trt_fused_io_formats.py` | Fused TRT engine I/O format utilities |
| `trt_fused_talk_c2w_profiles.py` | Fused Talker + Code2Wav TRT profiles |
| `update_triton_manifest_profile.py` | Update manifest with engine profile |

## Governance Rules

- A new user-facing validation or benchmark tool should live in `tests/tools/`, not `scripts/python/`.
- Shared types and schemas belong in `qwen3_tts_protocol`; repo-internal helpers in `scripts/python/qwen3tts_tools/`.
- Keep `scripts/python/` entry points thin — parse args, call shared code, render results.
- Before adding a new tool, check if an existing unified tool (e.g. `compare_audio.py`, `verify_engine.py`) already covers the use case.

## Audit Workflow

```bash
python scripts/python/audit_tooling_surface.py
```
