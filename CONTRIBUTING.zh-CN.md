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

# 独立 client SDK 测试套件
PYTHONPATH=client/src pytest client/tests -q

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

## 分支模型与版本发布

分支按单向晋升流水线组织：

| 分支 | 用途 | 合并来源 |
|------|------|----------|
| `<用户名>`（如 `rime`） | 个人开发 | — |
| `dev` | 跨用户集成同步 | 各用户分支 |
| `beta` | beta 测试（开发+测试） | **仅** `dev` |
| `main` | 稳定版本（所有人可见） | **仅** `beta` |

**Hotfix 例外** —— 唯一绕过 `beta` 进 `main` 的通道：稳定版出紧急问题时，
从 `main` 切 `hotfix/<topic>`，修复后合回 `main`（打补丁版本 tag
`vX.Y.Z`），并**立即**把修复同步回 `dev`（若 `beta` 正在测试周期中也要同步）。

**版本 tag** 是引擎镜像与 client wheel 的配对键（见
[docs/user/client_sdk.zh-CN.md](docs/user/client_sdk.zh-CN.md)）：

- 一律 `v` + PEP 440 形式：稳定 `vX.Y.Z`（打在 `main`），beta `vX.Y.Zb1` /
  `vX.Y.Zrc1`（打在 `beta`）。版本 tag 只打在 `dev` / `beta` / `main` ——
  **个人分支禁止打版本 tag**。
- 个人分支不需要版本 tag：hatch-vcs 自动推导 `X.Y.Z.devN+g<hash>`，引擎
  在版本化 capabilities 的 `engine_version` 中携带 `git describe` 输出，
  两者都精确到 commit。
- 工具链只认 `v[0-9]*` 形式的 tag（hatch-vcs `tag_regex` + `compose.sh` /
  `release_client_wheel.sh` 里的 `git describe --match`）。个人标记请用
  命名空间形式，如 `rime/some-checkpoint` —— 对版本推导完全不可见。

**发版流程**：`dev` 收敛 → 合入 `beta` → 打 `vX.Y.Zb1` → 测试通过 →
合入 `main` → 打 `vX.Y.Z`。GitHub Actions 与 GitLab CI 都只检出顶层仓库
（不拉子模块），各自构建唯一一份 wheel。GitHub 发布到 Release；GitLab 发布到
PyPI Package Registry 并挂到 Release。随后各流水线按 SHA256 将自己的同一文件
放进引擎镜像，分别推送 GHCR/GitLab Container Registry。每条发布流水线内都
禁止手工再构建或上传第二份 wheel。

Triton 镜像使用单独发布、不可变的依赖基座，release tag 流水线不再从 PyPI
下载数 GB 的 TensorRT wheel。某套运行时矩阵第一次发布前，先运行 GitLab 手动
流水线并设置 `BUILD_TRITON_RUNTIME_BASE=1`，以及/或者运行 GitHub 的
**Build Triton Runtime Base** workflow。`triton-deps` 阶段或
CUDA/TensorRT/PyTorch 矩阵变化时，先提升 `TRITON_RUNTIME_BASE_TAG`、发布新基座，
再创建 release tag。基座缺失时 release 会立即失败，这是有意的发布门禁。

## 提交 PR

1. 从 `dev` 切分支，PR 目标也是 `dev`（见上文分支模型；`main` 只接受来自 `beta` 的合并）。提交信息用 [Conventional Commits](https://www.conventionalcommits.org/) 前缀（`feat`/`fix`/`refactor`/`docs`/`chore` 等）。
2. 关联相关 issue，说明动机与验证方式（贴测试输出）。
3. 不引入新的 `TODO`/调试残留；不提交 `workspace/` 运行时产物（已 gitignore）。
4. 行为有变更时更新对应文档与测试。

## 报告问题

提 issue 时请尽量包含：模型变体（如 `custom-1.7b`）、流式/非流式、GPU 型号、复现步骤与最小输入。安全问题请按 [SECURITY.md](SECURITY.zh-CN.md)（若存在）私下报告。
