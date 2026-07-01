**English** | [中文](tooling_governance.zh-CN.md)

# Tooling Governance

This project is now large enough that `scripts/` and `tests/` need architecture of their own, not just ad-hoc additions. The goal of this document is to let new contributors understand the tooling surface and to keep the open-source release effort sustainable.

## Goals

- Keep the default contributor path clearly visible.
- Distinguish automated tests, manual tools, and one-off investigations.
- Prefer a shared helper layer over copy-pasting utility functions.
- Surface duplicated behavior early, so that not every script reimplements its own copy.

## Mental Model

### 1. Product code

- `engine/`: runtime code that we want the community to depend on and review as product behavior.

### 2. Lifecycle scripts

- `scripts/bash/`: setup, build, package, deploy, probing, and environment orchestration.
- `scripts/bash/lib/`: reusable shell primitives. New shell logic should go here first, to avoid duplication across entry points.

### 3. Export implementation

- `scripts/export/`: internal implementation of model export. These files are implementation modules, not something newcomers should run directly unless they are working on export internals.

### 4. Shared utility library

- `scripts/python/`: small, lightweight-dependency Python helpers invoked by bash (JSON/profile/web parsing, etc.).
- This is the preferred home for reusable repo paths, endpoints, target lists, and WAV helper logic.
- Keep this layer small and generic. It should support tooling, not become a second application runtime.

### 5. Python maintenance CLIs

- `scripts/python/`: lightweight CLIs, wrappers, analysis scripts, and maintainer tools.
- If a file here becomes a canonical validation workflow for users, it should usually move to `tools/validation/`, optionally leaving behind a compatibility wrapper.

### 6. Automated tests

- `tests/unit/`: fast logic tests.
- `tests/integration/`: export-artifact and packaging checks.
- `tests/e2e/`: automated service-level checks against a running system.

### 6a. Shared test support

- `tests/support/`: shared support code for the test suites and manual tools.
- When both pytest files and `tools/validation/` need it, put the reusable standalone engine helpers here.
- When a support module can own a behavior, `tools/validation/` should not import `tests/e2e/test_*.py` directly.

### 7. Manual validation tools

- `tools/validation/`: acceptance, benchmarking, audio inspection, and debugging tools that are deliberately not pytest tests.
- This is the canonical home for the manual validation entry points we expect open-source users to run directly.

### 8. Frozen reproductions

- `tools/repro/`: historical bug reproductions, which should be kept isolated from the normal validation surface.

## Placement Rules

When adding a new tool, use the following decision rules:

1. If the behavior should run in CI and automatically assert correctness, add a pytest test.
2. If the behavior is exploratory, benchmark-oriented, or depends on human listening/inspection, put it in `tools/validation/`.
3. If the behavior supports setup/export/build/deploy rather than validation, put it in `scripts/`.
4. If two or more files need the same Python utility, extract it into `scripts/python/` (test-only ones into `tests/support/`).
5. If two or more shell entry points need the same logic, extract it into `scripts/bash/lib/`.

## Naming Rules

- `tests/unit/`, `tests/integration/`, and `tests/e2e/` use `test_*.py` for pytest discovery.
- `tools/validation/` must not use the `test_*.py` prefix.
- Investigation scripts under `scripts/python/` should use a descriptive prefix such as `analyze_`, `compare_`, `probe_`, `replay_`, or `verify_`.
- Compatibility wrappers should be documented in the module docstring and delegate to `tools/validation/`.

## Lightweight Entry-point Pattern

A healthy Python CLI in this repo should usually look like this:

1. Minimal import-path bootstrapping
2. Argument parsing
3. Calling shared helper / module code
4. Rendering a summary / artifact

It should not repeatedly:

- Redefine the repo-root constant
- Redefine the default serving endpoints
- Redefine WAV serialization helpers
- Duplicate the same CSV target parsing

## Release Checklist

Before an open-source release or a large tooling merge:

```bash
python scripts/python/audit_tooling_surface.py
pytest tests/unit/test_tooling_helpers.py -q
```

Use the audit report to check:

- Total scripts/tests surface size
- Number of standalone Python entry points
- Duplicated symbols worth extracting
- Remaining `sys.path` bootstrapping
- Tools that unexpectedly import pytest modules
- Unexpected cache directories under `scripts/` or `tests/`

## Recent Migration Queue

The current first wave of governance work focuses on low-risk shared needs:

- serving endpoint defaults
- repo/workspace path helpers
- target parsing helpers
- WAV writing helpers
- a repeatable audit report

After that, the next valuable extractions may be:

- variant/model path resolution
- duplicated serving client utilities
- duplicated TRT/ONNX manifest loading helpers
- duplicated experiment output/report formatting

This phased approach avoids a disruptive large-scale rewrite while gradually making the project easier for new contributors to navigate.
