# Tooling Governance

This project is now large enough that `scripts/` and `tests/` need their own architecture, not just ad-hoc additions. The goal of this document is to make the tooling surface understandable to a new contributor and keep it sustainable for open-source release work.

## Goals

- Keep the default contributor path obvious.
- Distinguish automated tests from manual tools and one-off investigations.
- Prefer shared helper layers over copy-pasted utility functions.
- Make duplicate behavior visible early, before every script grows its own implementation.

## Mental Model

### 1. Product Code

- `engine/`: runtime code that we want the community to depend on and review as product behavior.

### 2. Lifecycle Scripts

- `scripts/bash/`: setup, build, package, deploy, probing, and environment orchestration.
- `scripts/bash/lib/`: reusable shell primitives. New shell logic should land here before it is repeated across entry points.

### 3. Export Implementation

- `scripts/export/`: model export internals. These files are implementation modules, not the first thing a newcomer should run directly unless they are working on export internals.

### 4. Shared Tooling Library

- `scripts/python/qwen3tts_tools/`: small, dependency-light Python helpers for shared tooling concerns.
- This is the preferred home for reusable repo-path, endpoint, target-list, and WAV helper logic.
- Keep this layer small and generic. It should support tools, not become a second application runtime.

### 5. Python Maintenance CLIs

- `scripts/python/`: thin CLIs, wrappers, analysis scripts, and maintainer utilities.
- If a file here becomes a canonical validation workflow for users, it should usually move to `tools/validation/` and optionally leave behind a compatibility wrapper.

### 6. Automated Tests

- `tests/unit/`: fast logic tests.
- `tests/integration/`: exported artifact and packaging checks.
- `tests/e2e/`: automated service-level checks against a running system.

### 6a. Shared Test Support

- `tests/support/`: shared support code for test suites and manual tools.
- Put reusable standalone-engine helpers here when both pytest files and `tools/validation/` need them.
- `tools/validation/` should not import `tests/e2e/test_*.py` directly when a support module can own that behavior.

### 7. Manual Validation Tools

- `tools/validation/`: acceptance, benchmark, audio inspection, and debug tools that are intentionally not pytest tests.
- This is the canonical place for manual verification entry points that we expect open-source users to run directly.

### 8. Frozen Reproductions

- `tools/repro/`: historical bug reproductions that should stay isolated from the normal validation surface.

## Placement Rules

Use the following decision rules when adding new tooling:

1. If the behavior should run in CI and assert correctness automatically, add a pytest test.
2. If the behavior is exploratory, benchmark-oriented, or depends on human listening/inspection, put it in `tools/validation/`.
3. If the behavior supports setup/export/build/deploy rather than validation, put it in `scripts/`.
4. If two or more files need the same Python utility, extract it into `scripts/python/qwen3tts_tools/`.
5. If two or more shell entry points need the same logic, extract it into `scripts/bash/lib/`.

## Naming Rules

- `tests/unit/`, `tests/integration/`, and `tests/e2e/` use `test_*.py` for pytest discovery.
- `tools/validation/` must not use the `test_*.py` prefix.
- Investigation scripts in `scripts/python/` should use descriptive prefixes such as `analyze_`, `compare_`, `probe_`, `replay_`, or `verify_`.
- Compatibility wrappers should say so in their module docstring and delegate to `tools/validation/`.

## Thin Entry Point Pattern

A healthy Python CLI in this repo should usually look like this:

1. minimal import-path bootstrap
2. argument parsing
3. call into shared helper/module code
4. render summary / artifacts

What it should not do repeatedly:

- redefine repo-root constants
- redefine default serving endpoints
- redefine WAV serialization helpers
- duplicate the same CSV target parsing

## Release Checklist

Before an open-source release or major tooling merge:

```bash
python scripts/python/audit_tooling_surface.py
pytest tests/unit/test_tooling_helpers.py -q
```

Use the audit report to review:

- total script/test surface size
- count of standalone Python entry points
- duplicate symbols worth extracting
- remaining `sys.path` bootstraps
- accidental tool imports of pytest modules
- accidental cache directories under `scripts/` or `tests/`

## Near-Term Migration Queue

The current first-pass governance work focuses on low-risk shared concerns:

- serving endpoint defaults
- repo/workspace path helpers
- target parsing helpers
- WAV writing helpers
- repeatable audit reporting

After that, the next profitable extractions are likely:

- variant/model-path resolution
- repeated serving client utilities
- repeated TRT/ONNX manifest loading helpers
- repeated experiment output/report formatting

This phased approach avoids a destabilizing big-bang rewrite while still making the project progressively easier for new contributors to navigate.
