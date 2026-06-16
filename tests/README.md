# Test Guide

This repository keeps test entry points deliberately separated:

| Location | Purpose | How to run |
| --- | --- | --- |
| `tests/unit/` | Fast pytest unit tests. No external serving process required. | `pytest tests/unit -q` |
| `tests/integration/` | Pytest integration checks for exported artifacts, manifests, ONNX/TRT config generation, and local build outputs. | `pytest tests/integration -q` |
| `tests/e2e/` | Pytest end-to-end checks against a running standalone engine or Triton service. Tests auto-skip when the service is not reachable. | `pytest tests/e2e -v -s` |
| `tests/tools/` | Manual validation, benchmark, audio generation, and investigation tools. These are not pytest tests. | `mamba run -n qwen3-tts python tests/tools/<tool>.py --help` |
| `tests/support/` | Shared support code used by pytest suites and manual tools. Not collected as tests. | Imported by tests/tools |
| `tests/data/` | Small fixtures used by tests and tools. | Imported by tests |
| `tests/repro/` | Frozen reproduction cases for known low-level issues. | See the reproduction README |

For the cross-cutting script/test mental model, also read [scripts/README.md](../scripts/README.md) and [docs/tooling_governance.md](../docs/tooling_governance.md).

## Recommended Entry Points

Run the normal developer suite:

```bash
pytest tests/unit tests/integration -q
```

Run Triton E2E pytest checks:

```bash
bash scripts/bash/build_triton.sh run
pytest tests/e2e/test_e2e.py -v -s
```

Run standalone engine E2E pytest checks:

```bash
python -m engine.server --config engine.yaml
pytest tests/e2e/test_engine_standalone.py -v -s
```

Run the full serving acceptance tool across standalone engine and Triton endpoints:

```bash
mamba run -n qwen3-tts python tests/tools/serving_endpoints.py
```

Run only the bare standalone engine gRPC acceptance path:

```bash
mamba run -n qwen3-tts python tests/tools/serving_endpoints.py \
  --targets engine-grpc \
  --skip-long --skip-badcase
```

Run a bare-engine TTFT distribution benchmark with warmup, mean, variance, standard deviation, percentiles, and fluctuation bars:

```bash
mamba run -n qwen3-tts python tests/tools/serving_endpoints.py \
  --targets engine-grpc \
  --skip-single --skip-streaming --skip-custom-instruct \
  --skip-concurrent --skip-long --skip-badcase \
  --ttft-warmup 3 \
  --ttft-samples 30
```

## Naming Rules

- Files named `test_*.py` under `tests/unit/`, `tests/integration/`, and `tests/e2e/` are pytest tests.
- Manual scripts must live under `tests/tools/` and must not use the `test_*.py` prefix.
- Reusable helpers shared by pytest tests and manual tools should live under `tests/support/`, so tools do not need to import pytest files.
- `scripts/` may contain launchers and build/deploy helpers, but not the canonical implementation of tests.
- Compatibility wrappers may remain in `scripts/` when an older command path already exists; the wrapper should delegate to `tests/tools/`.

These rules are meant to make the test surface obvious to new contributors and predictable for CI.

## Governance Check

When adding or refactoring manual tools, run:

```bash
python scripts/python/audit_tooling_surface.py
```

If a helper is shared across multiple CLIs, prefer extracting it into `scripts/python/qwen3tts_tools/` instead of copying it into another tool.
