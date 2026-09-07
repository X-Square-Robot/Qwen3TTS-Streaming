[English](README.md) | **中文**

# demo_api

Qwen3TTS-Streaming 内置门户的可选工程实验 API。

## 功能概述

demo_api 为内置 `/demo/#/lab` 提供可选的深度工程实验接口；普通试听不依赖它：

| 面板 | 说明 |
|------|------|
| **TRT live trace** | WebSocket 推送 Triton 音频与 decode 事件 |
| **LLM PK** | 流式 vs 离线双路对比，量化首包延迟与总耗时差异 |
| **Concurrency** | 并发压测，支持真实 Triton 请求或模拟模式，实时回传进度 |

## 前端与运行时边界

`demo_api` 只有后端职责。唯一的浏览器前端是
`web/packages/demo` 中的 React/Vite 门户，运行时从 `/demo/` 提供；不再有需要
启动的第二个 `webui/` 应用。工程实验统一从浏览器的
`/demo/#/lab` 进入。

门户的普通试听以及基础 LLM PK/并发实验使用当前实例的公共
`/v1/realtime`；runtime 配置 `DEMO_LAB_URL` 后，已迁移的深度工程面板（实时 TRT/Text
Player trace、服务端 LLM PK、多路并发）及后端 capabilities 才会请求 `demo_api`。它不
提供静态前端资源，也不替代产品门户。

在 runtime gateway 上将 `DEMO_LAB_URL` 设置为浏览器可访问的 API 根地址（例如
`http://localhost:7860`）。门户会探测该地址的 `/healthz`；可选后端不可达时，普通的
“体验”“SDK”“文档”和基础“实验”页面仍可使用；只有已迁移的深度工程面板会隐藏。例外是
docs-only 门户：它没有 live runtime，也不提供可交互实验。

## 依赖

```
pip install -r requirements.txt
```

核心依赖：

- `qwen3-tts-client[all]` — TTS 客户端及 Triton gRPC 通信
- `aiohttp>=3.9` — HTTP/WebSocket 服务框架
- `numpy>=1.24` — 音频数据处理

## 快速启动

**方式一：直接运行**

```bash
python -m demo_api
# 默认监听 0.0.0.0:7860
```

可选参数：`--host`、`--port`。

该命令只启动 API。要将它接入内置门户，请在启动 runtime 时设置
`DEMO_LAB_URL=http://localhost:7860`，再打开 runtime 的 `/demo/#/lab` 地址（例如
Triton gateway 使用 `http://localhost:50053/demo/#/lab`）。

**方式二：一键启动（推荐）**

```bash
bash scripts/demo/start_webui_demo.sh
```

该兼容脚本会自动拉起 Triton、demo_api，并指向 runtime 内置的唯一产品门户；不会启动第二套 Vite 应用。
支持 `--variant`、`--triton-slots`、`--no-triton`、`--simulated-concurrency` 等选项。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWEN_DEMO_TRITON_GRPC` | `localhost:8001` | Triton gRPC 地址 |
| `QWEN_DEMO_TRITON_MODEL` | `tts_orchestrator` | Triton 模型名 |
| `QWEN_DEMO_DEFAULT_SPEAKER` | `Serena` | 默认说话人 |
| `QWEN_DEMO_DEFAULT_LANGUAGE` | `auto` | 默认语言 |
| `QWEN_DEMO_DEFAULT_MS_PER_TOKEN` | `30` | LLM PK 上游模拟 token 速率 (ms/token) |
| `QWEN_DEMO_HOST` | `0.0.0.0` | API 监听地址 |
| `QWEN_DEMO_PORT` | `7860` | API 监听端口 |
| `QWEN_DEMO_CORS_ORIGIN` | `*` | CORS 允许来源 |
| `QWEN_DEMO_ENABLE_LIVE_CONCURRENCY` | 直接运行时为 `1`；Compose 默认 `0` | 是否启用真实并发请求（`0` 则为模拟模式） |
| `TRITON_MAX_BATCH_SLOTS` | `128` | Triton 最大批次槽位数 |
| `TRITON_MAX_SESSIONS` | `128` | Triton 最大会话数 |

## API 端点

以下是可选后端的 API 合同。内置门户的普通试听和基于 Realtime 的实验使用 runtime
公共 `/v1/*` 端点；只需要基本试听时不应依赖或公开此 API。

### HTTP

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/healthz` | 健康检查，返回 `{"ok": true}` |
| GET | `/api/v1/capabilities` | 获取后端能力、默认参数、运行时配置及限制声明 |
| POST | `/api/v1/llm-pk` | 执行 LLM PK 对比（流式 + 离线），返回双路结果 |
| POST | `/api/v1/concurrency` | 创建并发压测任务，返回 `job_id` |
| GET | `/api/v1/audio/{audio_id}` | 获取 LLM PK 等场景生成的音频文件 |

### WebSocket

| 路径 | 说明 |
|------|------|
| `/api/v1/trt-live` | Text Player 实时流式 TTS，发送 `{"type":"speak",...}` 开始 |
| `/api/v1/concurrency/{job_id}` | 并发压测进度推送，直至收到 `summary` 消息 |

## 与 client 包的关系

demo_api 依赖 `qwen3tts_protocol` 包中的 `schemas`（如 `TraceEvent`）和 `triton_types`（如 `TtsRequest`），
用于请求/响应的结构化定义。实际的 Triton gRPC 通信由 `triton_client` 模块封装，
demo_api 不再内嵌客户端逻辑，保持职责单一。浏览器侧协议与 UI 位于
`web/packages/demo` 和 `web/packages/browser-sdk`；已退出的顶层 `webui/` 路径不再是受支持
的开发或部署目标。
