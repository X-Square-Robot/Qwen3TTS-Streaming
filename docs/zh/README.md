# Qwen3-TTS Triton — 中文文档

> 面向用户的使用文档

## 快速开始

1. 克隆仓库并运行交互式入口：
   ```bash
   git clone --recursive https://github.com/user/Qwen3-TTS-Triton.git
   cd Qwen3-TTS-Triton
   bash scripts/bash/autorun.sh
   ```

2. 或使用 Python CLI：
   ```bash
   qwen3tts all -m custom-1.7b
   ```

3. 详见 [部署指南](deployment.md) 了解分阶段执行和高级参数。

## 文档索引

| 文档 | 说明 |
|------|------|
| [部署指南](deployment.md) | 三个部署方式（standalone / engine-docker / Triton）的详细参数和配置 |
| [Client SDK](client_sdk.md) | Python SDK 安装、使用、流式 session、传输选择 |
| [Benchmark 方法](benchmark_methodology.md) | 性能数字的口径定义、硬件条件、测量方式 |
| [已知限制](known_limitations.md) | 当前版本的已知问题和实验路径状态 |
| [路线图](roadmap.md) | 计划中的功能和改进方向 |

## 推荐路径

- **v0.1 稳定**：`custom-1.7b` / `custom_voice`
- **实验**：`design-1.7b`、`base-1.7b`（x-vector）、`icl`（语音克隆）
- **未主线**：`0.6b` 变体
