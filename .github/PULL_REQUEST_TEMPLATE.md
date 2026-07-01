**English** | [中文](PULL_REQUEST_TEMPLATE.zh-CN.md)

<!-- Thanks for contributing! Please read CONTRIBUTING.md before submitting. Titles are recommended to follow Conventional Commits (e.g. feat: / fix: / docs:). -->

## Description of Changes

<!-- What this PR does and why. To link an issue, write Closes #123. -->

## Type of Change

- [ ] Bug fix
- [ ] New feature / improvement
- [ ] Documentation
- [ ] Refactor / internal cleanup (no behavior change)
- [ ] Other:

## Self-Check Checklist

- [ ] Read [CONTRIBUTING.md](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/blob/main/CONTRIBUTING.md)
- [ ] Unit tests in the affected scope pass: `PYTHONPATH=client/src pytest tests/unit -m "not gpu and not docker" -q`
- [ ] Code style self-check: `ruff check` / `ruff format` (Python), shellcheck (Bash)
- [ ] If `proto/tts.proto` was changed, regenerated with `make proto` and synced with `make proto-sync` (generated code not hand-edited)
- [ ] If behavior changed, updated the relevant documentation (`docs/`)

## How to Test

<!-- How did you validate these changes? Provide the command / environment / results. -->
