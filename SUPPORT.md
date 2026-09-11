**English** | [中文](SUPPORT.zh-CN.md)

# Getting Help

Thank you for using Qwen3TTS-Streaming! Please choose the appropriate channel based on the type of question.

## Check the Docs First

- [README](README.md) — project overview and quick start
- [User Documentation](docs/user/README.md) — deployment, SDK, benchmark, known limitations
- [Deployment Guide](docs/user/deployment.md) — the full flow of build, assemble, and start the service
- [Known Limitations](docs/user/known_limitations.md) — **known phenomena such as streaming quality and variant support status**
- [CONTRIBUTING](CONTRIBUTING.md) — development environment and contribution workflow

## Choose a Channel

| Your situation | Where to go |
|----------------|-------------|
| Usage questions, deployment questions, idea exchange | [GitHub Discussions](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/discussions) |
| A reproducible bug | [File a Bug issue](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/issues/new?template=bug_report.yml) |
| Feature / improvement suggestions | [File a Feature issue](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/issues/new?template=feature_request.yml) |
| Security vulnerabilities | Report privately, see [SECURITY.md](SECURITY.md) (**do not file publicly**) |
| Issues with the model / inference quality itself | Upstream [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) |

## Tips for Asking

Including the following information when you ask lets us help you faster:

- The variant you use (e.g. `custom-1.7b`), the gateway (standalone / triton / engine), and the engine-mode;
- GPU model, driver / CUDA version, NGC container version, commit hash;
- The reproduction command and relevant logs (please remove sensitive information).

> This project is in the **v0.2 stability release** line. The validated checkpoint substantially
> reduces runaway hallucination, but residual repetition, dropped reading, and input-dependent
> quality issues can still occur. See "Known Limitations" above; please report reproducible
> regressions with the checkpoint, input, gateway, and sampling details.
