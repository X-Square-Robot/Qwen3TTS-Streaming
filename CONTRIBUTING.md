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

# Standalone client SDK suite
PYTHONPATH=client/src pytest client/tests -q

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

## Branching Model & Releases

Branches form a one-way promotion pipeline:

| Branch | Purpose | Merges from |
|--------|---------|-------------|
| `<username>` (e.g. `rime`) | personal development | — |
| `dev` | cross-user integration | user branches |
| `beta` | beta testing (dev + QA) | `dev` **only** |
| `main` | stable releases (visible to everyone) | `beta` **only** |

**Hotfix exception** — the only path into `main` that bypasses `beta`: for an
urgent fix on a stable release, branch `hotfix/<topic>` off `main`, fix, merge
back to `main` (tag a patch release `vX.Y.Z`), then immediately sync the fix
back into `dev` (and into `beta` if a test cycle is running).

**Version tags** are the pairing key between the engine image and the client
wheel (see [docs/user/client_sdk.md](docs/user/client_sdk.md)):

- Always `v` + PEP 440: stable `vX.Y.Z` (on `main`), beta `vX.Y.Zb1` /
  `vX.Y.Zrc1` (on `beta`). Version tags live on `dev` / `beta` / `main`
  only — **never tag versions on personal branches**.
- Personal branches need no version tags: hatch-vcs derives
  `X.Y.Z.devN+g<hash>` automatically and the engine stamp (`version` in
  versioned capabilities as `engine_version`) carries `git describe`, both
  unique per commit.
- The toolchain only recognizes `v[0-9]*` tags (hatch-vcs `tag_regex` +
  `git describe --match` in `compose.sh` / `release_client_wheel.sh`).
  Personal markers must use a namespace, e.g. `rime/some-checkpoint` —
  they are invisible to version derivation.

**Release flow**: converge `dev` → merge to `beta` → tag `vX.Y.Zb1` → tests
pass → merge to `main` → tag `vX.Y.Z`. GitHub Actions and GitLab CI each check
out only the top-level repository (no submodules) and build one wheel. GitHub
publishes it to its Release; GitLab publishes it to its PyPI Package Registry
and links it from its Release. Each pipeline downloads its exact SHA256 into
the engine image before publishing to GHCR or GitLab Container Registry. Do
not build or upload a second wheel manually within either release pipeline.

## Submitting a PR

1. Branch off `dev` and target your PR at `dev` (see the branching model above; `main` only receives merges from `beta`). Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes in commit messages (`feat`/`fix`/`refactor`/`docs`/`chore`, etc.).
2. Link the related issue and explain the motivation and how it was validated (paste the test output).
3. Do not introduce new `TODO`s / leftover debug code; do not commit `workspace/` runtime artifacts (already gitignored).
4. When behavior changes, update the corresponding documentation and tests.

## Reporting Issues

When filing an issue, please try to include: the model variant (e.g. `custom-1.7b`), streaming/non-streaming mode, GPU model, reproduction steps, and a minimal input. For security issues, please report privately per [SECURITY.md](SECURITY.md) (if present).
