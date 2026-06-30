# 安全策略

感谢你帮助保障 Qwen3TTS-Streaming 及其用户的安全。

## 支持的版本

本项目目前处于 **v0.1 工程预览**阶段，仅对 `main` 分支的最新提交提供安全修复。预览版本不建议用于生产环境（见 [已知限制](docs/user/known_limitations.md)）。

| 版本 | 是否提供安全更新 |
|------|------------------|
| `main`（最新） | ✅ |
| 其他 / 历史快照 | ❌ |

## 报告漏洞

**请不要通过公开 issue 报告安全漏洞。**

首选渠道是 GitHub 私密漏洞报告：

1. 打开仓库的 **Security** 标签页 → **Report a vulnerability**
   （即 [Private vulnerability reporting](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/security/advisories/new)）；
2. 描述漏洞、受影响组件、复现步骤，以及你评估的影响范围。

如无法使用该渠道，可通过仓库 **Security Advisories** 联系维护者。请在报告中尽量包含：

- 受影响的组件与版本（commit hash）；
- 复现步骤或 PoC；
- 潜在影响与（如有）建议的缓解措施。

## 响应预期

作为社区维护的预览项目，我们会尽力：

- 在 **5 个工作日内**确认收到报告；
- 评估并与你沟通修复计划；
- 修复发布后，在 advisory 中对报告者致谢（除非你希望匿名）。

请在我们发布修复或双方约定的披露时间之前，对漏洞细节保密。

## 安全边界说明

以下不属于本仓库的安全责任范围，请向相应上游报告：

- **上游模型与权重**：Qwen3-TTS 模型由阿里巴巴通义团队发布，模型本身的问题请向上游反馈；
- **第三方运行时**：NVIDIA TensorRT / Triton Inference Server / CUDA 的漏洞请向 NVIDIA 报告；
- **子模块代码**：`third_party/Qwen3-TTS/` 为上游子模块，其代码问题请向上游仓库报告。

部署相关的安全须知（鉴权、网络暴露面等）见 [部署指南](docs/user/deployment.md)。
