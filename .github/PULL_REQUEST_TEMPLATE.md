<!-- 感谢贡献！提交前请阅读 CONTRIBUTING.md。标题建议遵循 Conventional Commits（如 feat: / fix: / docs:）。 -->

## 改动说明

<!-- 这个 PR 做了什么、为什么。关联 issue 请写 Closes #123。 -->

## 改动类型

- [ ] Bug 修复
- [ ] 新功能 / 改进
- [ ] 文档
- [ ] 重构 / 内部清理（无行为变化）
- [ ] 其他：

## 自查清单

- [ ] 已阅读 [CONTRIBUTING.md](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/blob/main/CONTRIBUTING.md)
- [ ] 受影响范围的单元测试通过：`PYTHONPATH=client/src pytest tests/unit -m "not gpu and not docker" -q`
- [ ] 代码风格自查：`ruff check` / `ruff format`（Python）、shellcheck（Bash）
- [ ] 如改动 `proto/tts.proto`，已 `make proto` 重新生成并 `make proto-sync` 同步（生成代码未手改）
- [ ] 如有行为变化，已更新相关文档（`docs/`）

## 测试方式

<!-- 你如何验证这些改动？给出命令 / 环境 / 结果。 -->
