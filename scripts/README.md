# Script Guide

Open-source users need a small number of clear entry points, while maintainers still need room for export, deployment, and investigation work. This directory now uses the following mental model.

## Where To Start

- `scripts/bash/autorun.sh`: the primary operator entry point for setup, build, package, and deploy flows.
- `scripts/bash/probe_endpoints.sh`: fast readiness probe wrapper after a deployment is up.
- `scripts/python/audit_tooling_surface.py`: governance report for the current `scripts/` + `tests/` surface.
- `tests/README.md`: the main map for pytest suites and manual validation tools.

## Directory Roles

| Location | Role | Canonical audience |
| --- | --- | --- |
| `scripts/bash/` | lifecycle and operator workflows | users deploying or packaging the project |
| `scripts/bash/lib/` | shared shell library code | maintainers extending shell flows |
| `scripts/export/` | model export implementation (01–09) | maintainers working on ONNX / TRT export |
| `scripts/python/qwen3tts_tools/` | shared Python helper layer for tooling | maintainers adding or refactoring CLIs |
| `scripts/python/` | build/config/manifest tools (9 scripts) | advanced developers and maintainers |
| `scripts/compose/` | container entrypoints and compose support | deployment maintainers |
| `scripts/demo/` | demo launchers | demo users |

## Current scripts/python/ Inventory

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
- A new build, export, deployment, or packaging helper belongs in `scripts/`.
- If two Python entry points need the same repo-path, endpoint, CSV parsing, or WAV-writing logic, move it into `qwen3_tts_protocol` (for types/schemas) or `scripts/python/qwen3tts_tools/` (for repo-internal helpers).
- If two shell entry points need the same behavior, add it under `scripts/bash/lib/` and source it through `scripts/bash/tools.sh`.
- Keep `scripts/python/` entry points thin. The file should mainly parse args, call shared code, and render results.
- Compatibility wrappers are acceptable when an older command path is already in docs or teammate muscle memory, but the wrapper should delegate to the canonical implementation.

## Audit Workflow

Run this before large script/test additions or before an open-source release cut:

```bash
python scripts/python/audit_tooling_surface.py
python scripts/python/audit_tooling_surface.py --json
```

The report highlights:

- how much larger `scripts/` + `tests/` have become than `engine/`
- how many standalone Python entry points now exist
- where `sys.path` bootstrap code is still duplicated
- which function/class names are duplicated enough to consider extracting
