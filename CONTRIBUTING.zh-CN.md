[English](CONTRIBUTING.md) | **中文**

# 贡献指南

感谢你对 Qwen3TTS-Streaming 的关注！本项目是 **v0.1 工程预览**，流式质量仍在打磨（见 [已知限制](docs/user/known_limitations.zh-CN.md)），欢迎以 issue、讨论、PR 形式参与。

## 开发环境

```bash
# 1. 克隆（含子模块）
git clone --recursive https://github.com/X-Square-Robot/Qwen3TTS-Streaming.git
cd Qwen3TTS-Streaming

# 2. 安装依赖（建议用项目约定的 conda 环境）
pip install -e ".[dev]"
# 客户端协议层在 client/src，跑测试需要它在 PYTHONPATH 上
export PYTHONPATH=client/src
```

> 完整引擎运行（导出/编译 TRT/起服务）需要 NVIDIA GPU + NGC 容器，见 [README 前置要求](README.zh-CN.md#前置要求) 与 [部署指南](docs/user/deployment.zh-CN.md)。纯逻辑开发与单元测试不需要 GPU。

## 运行测试

```bash
# 单元测试（无需 GPU / Docker —— CI 跑这一档）
PYTHONPATH=client/src pytest tests/unit -m "not gpu and not docker" -q

# 集成 / e2e（可能需要导出产物、GPU 或 Docker）
pytest tests/integration -q
```

测试分层 marker：`unit` / `integration` / `e2e` / `gpu` / `docker` / `slow`（见 `pyproject.toml`）。提交前请确保受影响范围的单元测试通过。

## 代码风格

- **Python**：类型注解、docstring、PEP 8；用 `ruff check` / `ruff format`、`mypy` 自查。
- **Bash**：shellcheck 兼容；`scripts/bash/lib/` 模块用 `snake_case` 函数名。
- **文档**：正文统一中文，代码块/变量名/路径/命令保持英文（见 `CLAUDE.md` 文档语言策略）。

## 协议（proto）改动

`proto/tts.proto` 是协议唯一真相源，生成代码不要手改：

```bash
make proto        # 重新生成 *_pb2.py
make proto-sync   # 同步到 engine/ 与 client/ 消费方
make proto-check  # 校验已同步（CI 会跑）
```

## 提交 PR

1. 从 `main` 切分支；提交信息用 [Conventional Commits](https://www.conventionalcommits.org/) 前缀（`feat`/`fix`/`refactor`/`docs`/`chore` 等）。
2. 关联相关 issue，说明动机与验证方式（贴测试输出）。
3. 不引入新的 `TODO`/调试残留；不提交 `workspace/` 运行时产物（已 gitignore）。
4. 行为有变更时更新对应文档与测试。

## 报告问题

提 issue 时请尽量包含：模型变体（如 `custom-1.7b`）、流式/非流式、GPU 型号、复现步骤与最小输入。安全问题请按 [SECURITY.md](SECURITY.zh-CN.md)（若存在）私下报告。
