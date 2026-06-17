# 已知限制与风险

本文档是当前开源预览版的风险说明。发布前请保持它和 README、WebUI、demo API 的口径一致。

## 版本定位

当前项目是 **工程预览版 / research preview**，不是生产稳定版本。它主要展示 Qwen3-TTS 在 TensorRT、模型 fuse、token 级流式调度、prefix cache 和前端分词上的工程优化。

推荐 v0.1 稳定范围：

- 模型：`custom-1.7b`
- 任务：`custom_voice`
- 部署：standalone engine / Triton TRT streaming
- WebUI：性能展示、trace 回放、live TRT/engine 对照

## 模型路径状态

| 路径 | 状态 | 风险 |
| --- | --- | --- |
| `custom-1.7b` / `custom_voice` | v0.1 推荐路径 | 仍需继续压测流式稳定性、长文本、并发、不同说话人 |
| `design-1.7b` / `voice_design` | 实验 | 部分代码路径存在，但没有充分端到端验证 |
| `base` x-vector voice clone | 计划中 | ref audio preprocessing、speaker embedding 注入、端到端验证未完成 |
| `icl` voice clone | 计划中 | ref audio/ref text/ref code 链路未完整打通 |
| `0.6b` variants | 非 v0.1 主线 | 需要独立验证导出、profile、质量和速度 |

## 流式稳定性

当前流式模式仍可能出现：

- 幻觉：生成用户没有输入的内容。
- 重复：局部词、短语或音频片段重复。
- 漏读：跳过部分输入文本。
- 插入：在停顿或跨 segment 时插入额外字词。
- 长文本退化：随着上下文和 KV 增长，稳定性下降。
- 分段边界异常：标点、数字、英文、中英混排等文本可能触发不理想切分。

这些问题意味着当前版本不适合直接用于有强一致性要求的生产播报、客服、医疗、金融、法律或内容安全场景。

## 性能数字限制

`13ms TTFT` 不是通用承诺。它通常需要同时满足：

- engine 已 warm up。
- prefix/cache 命中。
- 单路请求或低竞争。
- 固定硬件、固定 TensorRT profile、固定 dtype。
- 本地或低网络开销链路。

`128-stream avg TTFT` 也必须带上完整测试条件，包括硬件、driver、NGC 镜像、engine profile、输入文本、cache 模式、采样参数、客户端测量方法和失败率。

## WebUI 数据来源

WebUI 有两种数据来源：

- fixture trace：离线 JSON 回放，适合展示 UI 和对齐指标字段，不代表实时服务。
- live measurement：实时调用 standalone engine、Triton 或官方 PyTorch API。

只有结果里的 `source` 是 `live_triton`、`live_engine` 或 `live_official_pytorch` 时，才代表实时测量。默认 fixture 值在发布前需要用真实硬件重新采集。

## 部署限制

- TensorRT engine 与 TensorRT runtime 版本强相关。更换 NGC 镜像、TensorRT 版本或 driver 后，建议重新构建 engine。
- runtime 的 `max_batch`/`max_seq_len` 不能超过 manifest 中记录的 `engine_profile`，否则启动会直接失败。
- `engine-docker` 普通镜像适合固定代码部署；开发期请用 `compose.sh --dev` 或 `compose.sh watch` 避免频繁重建镜像。
- 当前容器没有覆盖 K8s、灰度发布、鉴权、限流、多租户隔离等生产运维能力。

## 发布前必须保留的用户提示

README、WebUI 和 release note 中必须明确：

- 本项目是工程预览版。
- v0.1 推荐路径是 `custom-1.7b`。
- base/ICL/voice design 不应宣传为已稳定可用。
- 流式 TTS 存在幻觉和长文本不稳定风险。
- benchmark 数字需要附带完整条件，不能写成无条件性能承诺。
