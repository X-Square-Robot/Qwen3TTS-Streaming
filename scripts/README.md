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
| `scripts/export/` | model export implementation | maintainers working on ONNX / TRT export |
| `scripts/python/qwen3tts_tools/` | shared Python helper layer for tooling | maintainers adding or refactoring CLIs |
| `scripts/python/` | thin CLIs, analysis helpers, and compatibility wrappers | advanced developers and maintainers |
| `scripts/compose/` | container entrypoints and compose support | deployment maintainers |
| `scripts/demo/` | demo launchers | demo users |

## Governance Rules

- A new user-facing validation or benchmark tool should live in `tests/tools/`, not `scripts/python/`.
- A new build, export, deployment, or packaging helper belongs in `scripts/`.
- If two Python entry points need the same repo-path, endpoint, CSV parsing, or WAV-writing logic, move it into `scripts/python/qwen3tts_tools/`.
- If two shell entry points need the same behavior, add it under `scripts/bash/lib/` and source it through `scripts/bash/tools.sh`.
- Keep `scripts/python/` entry points thin. The file should mainly parse args, call shared code, and render results.
- Compatibility wrappers are acceptable when an older command path is already in docs or teammate muscle memory, but the wrapper should delegate to the canonical implementation.
- Generated caches such as `__pycache__/` should never be treated as source.

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

## Migration Direction

The first shared layer intentionally stays small:

- repo/workspace constants
- serving endpoint defaults
- common CLI parsing helpers
- WAV writing helpers

That keeps the governance layer useful without turning it into another monolith. Export/model-specific logic can move later once the thin-entrypoint pattern is stable.
