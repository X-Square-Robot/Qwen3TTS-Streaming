[English](README.md) | **中文**

# 开发者文档

> 面向贡献者、架构师和深度开发者

> **分支与发版规则**（用户分支 → `dev` → `beta` → `main`、hotfix 通道、
> 版本 tag 仅限 `v` 前缀）：见
> [贡献指南 — 分支模型与版本发布](../../CONTRIBUTING.zh-CN.md#分支模型与版本发布)。

## 架构

| 文档 | 说明 |
|------|------|
| [架构设计](architecture.zh-CN.md) | 完整架构文档 — 推理引擎、前后端、协议、KV Cache 全链路 |
| [解码 FSM 设计](architecture/decode_fsm.zh-CN.md) | 解码阶段有限状态机设计 |
| [引擎架构决策](architecture/engine_decisions.zh-CN.md) | 引擎核心设计决策和选择理由 |
| [流式协议设计](architecture/streaming_protocol.zh-CN.md) | Standalone 协议重设计 — gRPC/WebSocket 流式 |

## 设计目标

| 文档 | 说明 |
|------|------|
| [VAD 设计目标](design/vad_design_goals.zh-CN.md) | VAD（语音活动检测）设计目标与理由 |
| [可观测性目标](design/observability_goals.zh-CN.md) | 可观测性和指标设计目标 |
| [实时音频流](design/realtime_audio.zh-CN.md) | 实时音频流设计目标 |
| [流式文本消歧与 Soft Drain](design/incremental_text_normalization_and_soft_drain.zh-CN.md) | 增量 TN、单调提交、WAIT/HOLD、Soft Drain 和状态继承设计 |

## 调查报告

| 文档 | 说明 |
|------|------|
| [流式幻觉调查](investigation/streaming_hallucination.zh-CN.md) | 流式/采样幻觉问题的诊断与缓解 |
| [Code2Wav 状态大小](investigation/code2wav_state_size.zh-CN.md) | Code2Wav 状态大小分析与内存权衡 |
| [服务性能压测](investigation/serving_performance_benchmark.zh-CN.md) | 连接/排队/推理耗时拆解,跨协议与并发对比,含原始数据 |

## 运维

| 文档 | 说明 |
|------|------|
| [跨机构建](operations/cross_host_build.zh-CN.md) | 跨机构建工作流 — 在远程机器上编译 TRT engine |
| [E2E 测试总结](operations/e2e_test_summary.zh-CN.md) | E2E 测试入口与覆盖范围 |
| [计时指标参考](operations/timing_metrics.zh-CN.md) | 计时指标定义与测量方法 |
| [工具治理](operations/tooling_governance.zh-CN.md) | 工具治理 — 放置规则、命名约定、心智模型 |

普通用户请看 [用户文档](../user/)。
