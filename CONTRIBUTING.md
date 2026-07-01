**English** | [中文](CONTRIBUTING.zh-CN.md)

# Contributing Guide

Thank you for your interest in Qwen3TTS-Streaming! This project is a **v0.1 engineering preview**, and streaming quality is still being polished (see [Known Limitations](docs/user/known_limitations.md)); you are welcome to participate via issues, discussions, and PRs.

## Development Environment

```bash
# 1. Clone (with submodules)
git clone --recursive https://github.com/X-Square-Robot/Qwen3TTS-Streaming.git
cd Qwen3TTS-Streaming

# 2. Install dependencies (using the project's conventional conda environment is recommended)
pip install -e ".[dev]"
# The client protocol layer lives in client/src; running tests requires it on PYTHONPATH
export PYTHONPATH=client/src
```

> A full engine run (export / build TRT / start the service) requires an NVIDIA GPU + NGC containers; see the [README Prerequisites](README.md#prerequisites) and the [Deployment Guide](docs/user/deployment.md). Pure logic development and unit tests do not require a GPU.

## Running Tests

```bash
# Unit tests (no GPU / Docker required — this is the tier CI runs)
PYTHONPATH=client/src pytest tests/unit -m "not gpu and not docker" -q

# Integration / e2e (may require export artifacts, a GPU, or Docker)
pytest tests/integration -q
```

Test-layer markers: `unit` / `integration` / `e2e` / `gpu` / `docker` / `slow` (see `pyproject.toml`). Before submitting, please make sure the unit tests in the affected scope pass.

## Code Style

- **Python**: type annotations, docstrings, PEP 8; self-check with `ruff check` / `ruff format` and `mypy`.
- **Bash**: shellcheck-compatible; use `snake_case` function names in `scripts/bash/lib/` modules.
- **Documentation**: prose is uniformly in Chinese, while code blocks / variable names / paths / commands stay in English (see the documentation language policy in `CLAUDE.md`).

## Protocol (proto) Changes

`proto/tts.proto` is the single source of truth for the protocol; do not hand-edit the generated code:

```bash
make proto        # Regenerate *_pb2.py
make proto-sync   # Sync to the engine/ and client/ consumers
make proto-check  # Verify it is synced (CI runs this)
```

## Submitting a PR

1. Branch off `main`; use [Conventional Commits](https://www.conventionalcommits.org/) prefixes in commit messages (`feat`/`fix`/`refactor`/`docs`/`chore`, etc.).
2. Link the related issue and explain the motivation and how it was validated (paste the test output).
3. Do not introduce new `TODO`s / leftover debug code; do not commit `workspace/` runtime artifacts (already gitignored).
4. When behavior changes, update the corresponding documentation and tests.

## Reporting Issues

When filing an issue, please try to include: the model variant (e.g. `custom-1.7b`), streaming/non-streaming mode, GPU model, reproduction steps, and a minimal input. For security issues, please report privately per [SECURITY.md](SECURITY.md) (if present).
