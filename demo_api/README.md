# demo_api

Qwen3TTS-Streaming 的 WebUI 演示 API 服务。

## 功能概述

demo_api 为前端 WebUI 提供三个核心演示面板的后端接口：

| 面板 | 说明 |
|------|------|
| **Text Player** | 实时流式 TTS 播放，WebSocket 逐 token 推送音频与事件 |
| **LLM PK** | 流式 vs 离线双路对比，量化首包延迟与总耗时差异 |
| **Concurrency** | 并发压测，支持真实 Triton 请求或模拟模式，实时回传进度 |

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

**方式二：一键启动（推荐）**

```bash
bash scripts/demo/start_webui_demo.sh
```

该脚本会自动拉起 Triton、demo_api 和 Vite WebUI，开箱即用。
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
| `QWEN_DEMO_ENABLE_LIVE_CONCURRENCY` | `1` | 是否启用真实并发请求（`0` 则为模拟模式） |
| `TRITON_MAX_BATCH_SLOTS` | `128` | Triton 最大批次槽位数 |
| `TRITON_MAX_SESSIONS` | `128` | Triton 最大会话数 |

## API 端点

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
demo_api 不再内嵌客户端逻辑，保持职责单一。
