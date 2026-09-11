[English](CONTRIBUTING.md) | **中文**

# 贡献指南

感谢你对 Qwen3TTS-Streaming 的关注！本项目当前处于 **v0.2 稳定性版本**阶段。已验证
checkpoint 已大幅降低跑飞型幻觉，仍需要继续覆盖更多 checkpoint 和业务负载（见[已知限制](docs/user/known_limitations.zh-CN.md)），欢迎以 issue、讨论、PR 形式参与。

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

修改浏览器门户时请使用 `web/` npm workspace，并先阅读
[`web/packages/demo` 开发说明](web/packages/demo/README.zh-CN.md)。原独立的 `webui/`
前端已退出；真实 UI 检查应访问 runtime 的 `/demo/`，深度工程面板按需通过
公共 `/v1/realtime` 接口接入实例。

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
从 `main` 切 `hotfix/<topic>`，修复后合回 `main`，为 `vX.Y.Z` 运行补丁候选发布，
并**立即**把修复同步回 `dev`（若 `beta` 正在测试周期中也要同步）。

**版本 tag** 是引擎镜像与 client wheel 的配对键（见
[docs/user/client_sdk.zh-CN.md](docs/user/client_sdk.zh-CN.md)）：

- 一律 `v` + PEP 440 形式：稳定 `vX.Y.Z`（从 `main` 晋级），beta `vX.Y.Zb1` /
  `vX.Y.Zrc1`（从 `beta` 晋级）。版本 tag 由发布 promotion 流水线在候选验证通过后
  创建，**不要从个人分支手工推版本 tag**。
- 个人分支不需要版本 tag：hatch-vcs 自动推导 `X.Y.Z.devN+g<hash>`，引擎
  在版本化 capabilities 的 `engine_version` 中携带 `git describe` 输出，
  两者都精确到 commit。
- 工具链只认 `v[0-9]*` 形式的 tag（hatch-vcs `tag_regex` + `compose.sh` /
  `release_client_wheel.sh` 里的 `git describe --match`）。个人标记请用
  命名空间形式，如 `rime/some-checkpoint` —— 对版本推导完全不可见。

**发版流程**：`dev` 收敛 → 合入 `beta` → 手动运行发布 workflow，传入
`RELEASE_VERSION=vX.Y.Zb1` / `release_version=vX.Y.Zb1` 和目标 source ref →
测试已晋级的 beta → 合入 `main` → 用同样方式发布 `vX.Y.Z`。GitHub Actions 与
GitLab CI 先从选定 commit 构建候选产物和候选镜像（不拉子模块），最后的 promotion
job 才创建正式 tag/Release。CI、脚本或 runner 临时失败时重跑同一个候选 job，不要
新造版本号；只有源码 commit 或发布内容真的变化时才换版本。

启动方式：在 GitHub Actions → **Release** → **Run workflow** 中填写
`release_version` 和 `source_ref`；或在目标 GitLab ref 上运行 pipeline，填写
`RELEASE_VERSION`，必要时填写 `RELEASE_SOURCE_REF`。GitHub 和 GitLab 的候选流水线
相互独立，只晋级凭据和产物均已准备好的平台。

Triton 镜像使用单独发布、不可变的依赖基座，候选发布流水线不再从 PyPI
下载数 GB 的 TensorRT wheel。某套运行时矩阵第一次发布前，先运行 GitLab 手动
流水线并设置 `BUILD_TRITON_RUNTIME_BASE=1`，以及/或者运行 GitHub 的
**Build Triton Runtime Base** workflow。`triton-deps` 阶段或
CUDA/TensorRT/PyTorch 矩阵变化时，先提升 `TRITON_RUNTIME_BASE_TAG`。基座缺失时
候选发布会立即失败，且发生在正式版本 tag 创建之前，这是有意的发布门禁。

## 提交 PR

1. 从 `dev` 切分支，PR 目标也是 `dev`（见上文分支模型；`main` 只接受来自 `beta` 的合并）。提交信息用 [Conventional Commits](https://www.conventionalcommits.org/) 前缀（`feat`/`fix`/`refactor`/`docs`/`chore` 等）。
2. 关联相关 issue，说明动机与验证方式（贴测试输出）。
3. 不引入新的 `TODO`/调试残留；不提交 `workspace/` 运行时产物（已 gitignore）。
4. 行为有变更时更新对应文档与测试。

## 报告问题

提 issue 时请尽量包含：模型变体（如 `custom-1.7b`）、流式/非流式、GPU 型号、复现步骤与最小输入。安全问题请按 [SECURITY.md](SECURITY.zh-CN.md)（若存在）私下报告。
