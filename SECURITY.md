**English** | [中文](SECURITY.zh-CN.md)

# Security Policy

Thank you for helping keep Qwen3TTS-Streaming and its users secure.

## Supported Versions

This project is currently in the **v0.2 stability release** line, and provides security fixes only for the latest commit on the `main` branch. The validated checkpoint substantially reduces runaway hallucination, but operators should still review the [Known Limitations](docs/user/known_limitations.md) and validate their own workloads.

| Version | Security updates provided |
|---------|---------------------------|
| `main` (latest) | ✅ |
| Others / historical snapshots | ❌ |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public issues.**

The preferred channel is GitHub private vulnerability reporting:

1. Open the repository's **Security** tab → **Report a vulnerability**
   (i.e. [Private vulnerability reporting](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/security/advisories/new));
2. Describe the vulnerability, the affected components, the reproduction steps, and your assessment of the impact scope.

If you cannot use that channel, you can contact the maintainers via the repository's **Security Advisories**. Please try to include in your report:

- The affected components and version (commit hash);
- Reproduction steps or a PoC;
- The potential impact and, if any, suggested mitigations.

## Response Expectations

As a community-maintained preview project, we will do our best to:

- Acknowledge receipt of the report within **5 business days**;
- Assess the fix plan and communicate it with you;
- Credit the reporter in the advisory after the fix is released (unless you prefer to remain anonymous).

Please keep the vulnerability details confidential until we release a fix or until a disclosure time agreed upon by both parties.

## Security Boundary Notes

The following are outside this repository's security responsibility; please report them to the corresponding upstream:

- **Upstream model and weights**: the Qwen3-TTS model is released by Alibaba's Tongyi team; report issues with the model itself upstream;
- **Third-party runtimes**: report vulnerabilities in NVIDIA TensorRT / Triton Inference Server / CUDA to NVIDIA;
- **Submodule code**: `third_party/Qwen3-TTS/` is an upstream submodule; report code issues to the upstream repository.

For deployment-related security notes (authentication, network exposure surface, etc.), see the [Deployment Guide](docs/user/deployment.md).
