[English](SUPPORT.md) | **中文**

# 获取帮助

感谢使用 Qwen3TTS-Streaming！请根据问题类型选择合适的渠道。

## 先看文档

- [README](README.zh-CN.md) —— 项目概览与快速开始
- [用户文档](docs/user/README.zh-CN.md) —— 部署、SDK、Benchmark、已知限制
- [部署指南](docs/user/deployment.zh-CN.md) —— 编译、组装、起服务的完整流程
- [已知限制](docs/user/known_limitations.zh-CN.md) —— **流式质量、变体支持状态等已知现象**
- [CONTRIBUTING](CONTRIBUTING.zh-CN.md) —— 开发环境与贡献流程

## 选择渠道

| 你的情况 | 去哪里 |
|----------|--------|
| 使用问题、部署疑问、想法交流 | [GitHub Discussions](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/discussions) |
| 可复现的 Bug | [提交 Bug issue](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/issues/new?template=bug_report.yml) |
| 功能 / 改进建议 | [提交 Feature issue](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/issues/new?template=feature_request.yml) |
| 安全漏洞 | 私密报告，见 [SECURITY.md](SECURITY.zh-CN.md)（**勿公开提交**） |
| 模型 / 推理质量本身的问题 | 上游 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) |

## 提问小贴士

提问时附上以下信息能让我们更快帮到你：

- 使用的 variant（如 `custom-1.7b`）、gateway（standalone / triton / engine）与 engine-mode；
- GPU 型号、驱动 / CUDA 版本、NGC 容器版本、commit hash；
- 复现命令与相关日志（请删除敏感信息）。

> 本项目当前处于 **v0.2 稳定性版本**阶段。已验证 checkpoint 已大幅降低跑飞型幻觉，
> 但仍可能出现残余重复、漏读和依赖输入的质量问题。请先看上文「已知限制」；如果能复现
> 回归，请附上 checkpoint、输入、gateway 和采样参数后报告。
