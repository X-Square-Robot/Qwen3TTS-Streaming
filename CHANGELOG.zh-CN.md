[English](CHANGELOG.md) | **中文**

# Changelog

本项目的所有重要变更都会记录在此文件中。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- v0.2 稳定性版本已经在已验证的重训 checkpoint 上大幅抑制历史跑飞型幻觉问题。
  当前记录的全 bf16 确定性验证为 0/100；旧 0601 checkpoint 的 10–18% 结果仅作为
  历史证据保留。
- 流式 TN 现在在 tokenizer/Spliter 之前维护 raw Unicode、mutable tail、主
  `TextCommit` 和 `CanonicalTextJournal`；WeText、混合语言路由、显式 fallback 以及
  raw → spoken provenance 由同一套前端合同负责。
- 已验证的 `custom-1.7b` cursor-enabled TRT plan 现在可以在图内消费 codec0 并发布原生
  文本进度；`qwen.text_progress.v1` 暴露 `native`、`ema`、`disabled` 三种 progress
  route，owner-level raw/normalized high-water 与展示插值分离。
- 版本 tag 现在触发 wheel-first GitHub/GitLab 发布流水线：只检出顶层仓库且不拉
  子模块，每个平台各自只构建一份 client wheel；GitHub 发布到 Release，GitLab
  发布到 PyPI Registry 并挂 Release 链接，再将该平台已发布、SHA256 完全相同的
  文件嵌入对应引擎镜像。
- standalone WebSocket 现在支持在同一物理连接上串行运行多个逻辑 TTS 会话；SDK
  提供并发连接池、空闲保活、僵尸连接探测以及新会话建连失败重试，并新增与
  `end()` 等价的 `stop()`；可复用 `done` 能力标识使新旧版本混部时可安全退化为逐
  session 重连，engine error 连接不会进入池。
- `TTSClient.connect()` 新增 `key=None`：非 `None` 时自动为 HTTP/WebSocket
  添加 `Authorization: Bearer <key>`，并为 gRPC 添加小写 `authorization`
  metadata。鉴权仍由部署平台负责，engine 不校验 Header。

### Changed

- serving performance matrix 现在显式设置 SDK WebSocket `max_connections`，确保 64/128 路
  测量不会被默认 32 条连接的客户端池提前串行化；当前刷新与历史数据集分开保存。
- 将原独立的 `webui/` 特性展示前端并入 `web/packages/demo`，由运行时统一从
  `/demo/` 提供版本匹配的门户。试听、SDK、文档和实验页共用该门户，
  `/demo/#/lab` 是唯一实验入口，深度工程面板按可选后端能力门控；`demo_api` 只提供
  后端 API，不再提供第二套前端。
- Python SDK 的 `transport="auto"` 改为优先选择原生 `/v1/ws`；
  `/v1/realtime` 保留为 OpenAI Realtime 兼容入口。显式 endpoint 继续严格遵循调用方
  选择，只有原生入口不可用时才自动回退 Realtime。
- `engine-websocket` 不再被标记为旧兼容 transport；弃用告警只适用于
  `engine-grpc`、`triton-grpc` 和 `triton-http` 三种旧直连路径。

### Fixed

- 将引擎 WebSocket 握手预算（`connect_timeout`）与连接建立后的请求接收空闲预算
  （`timeout`）分离，并覆盖自动探测路径，避免不可达端点按完整请求超时长期占用
  流式 worker。
- 流式连接的首个 `start` 消息发送失败时关闭已经建立的 WebSocket，避免重复失败时
  泄漏文件描述符。
- 等待 capabilities 时容忍短轮询的接收超时，不再因为首个 200 ms 轮询超时而误判失败。
- 将 WebSocket 发送超时及 `ECONNRESET` 等 socket 错误统一归一化，使流式调用方稳定
  收到 `StreamClosedError`。

## [0.1.0] —— 工程预览

> ⚠️ **v0.1 工程预览**：流式模式仍可能出现幻觉/重复/漏读，不建议用于生产。
> 推荐路径为 `custom-1.7b` / `custom_voice`，其他变体为实验状态。详见
> [已知限制](docs/user/known_limitations.zh-CN.md)。

### Added

- **流式 TTS 推理系统**：将官方 Qwen3-TTS PyTorch 权重导出为 ONNX/TensorRT 运行时，
  围绕 Triton Inference Server / standalone engine 实现流式推理。
- **推理引擎**（`engine/`）：frontend 前端分词、backend 推理、gateway 网关、core 调度，
  含 prefix cache 与连续批处理。
- **生命周期编排**（`scripts/bash/`）：`autorun.sh` 全流程 / 分阶段（setup → build →
  package → deploy），支持跨机编译（probe-target / make-bundle / import-artifact /
  remote-build）与分子模块混合精度 TRT 编译。
- **Python 客户端 SDK**（`client/`，`qwen3-tts-client`）：含共享协议层与传输适配器。
- **WebUI Demo**：aiohttp 后端（`demo_api/`）+ Vite/React 前端（`webui/`）。
- **协议单一真相源**（`proto/tts.proto`）：`make proto` 生成、`make proto-sync` 同步。
- **文档**：用户文档（部署、SDK、Benchmark、限制）、开发者文档（架构、设计、调查、运维）。

> 上面的 v0.1 WebUI 条目记录的是历史上的独立前端布局。当前版本使用统一的
> `web/packages/demo` 门户；迁移边界见上方 Unreleased 条目。

### Notes

- 自有代码以 [MIT](LICENSE) 许可证发布，版权归 XSquareRobot；上游 Qwen3-TTS
  （`third_party/` 子模块）为 Apache-2.0。
- 模型权重经 ModelScope/HF 获取，不随仓库分发；TensorRT/Triton 不打包（NVIDIA EULA）。
- `resources/speakers/` 参考音频为合成音 + 假名，无真人录音或个人信息（见 [NOTICE](NOTICE)）。

[Unreleased]: https://github.com/X-Square-Robot/Qwen3TTS-Streaming/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/X-Square-Robot/Qwen3TTS-Streaming/releases/tag/v0.1.0
