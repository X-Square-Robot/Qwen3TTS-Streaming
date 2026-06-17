# Qwen3-TTS Triton

*让我们像播放音频一样播放文本！*

## 引言

Qwen3-TTS Triton 是一个**工程预览版**项目：把官方 Qwen3-TTS PyTorch 权重导出为 ONNX/TensorRT 运行时，围绕 Triton/standalone engine 做 token 级流式 TTS、模型 fuse、前端分词、prefix cache、连续批处理和 WebUI 性能展示。项目开放一条已高度优化、可复现、可继续验证的工程链路，让社区一起打磨成可靠的开源推理系统。

**当前建议 v0.1 稳定范围为 `custom-1.7b` / `custom_voice` 路径**；`design-1.7b`、`base-1.7b` / x-vector 语音克隆、`icl` 语音克隆处于实验状态；`0.6b` 变体未作为 v0.1 主线。流式模式仍可能出现幻觉、重复、漏读等问题，请勿直接用于生产内容生成。

## 性能声明

项目里提到的低延迟数字是有条件结果，不是通用承诺：

- `13ms TTFT`：最低观测值，依赖指定硬件、warm engine、prefix/cache 命中、单路请求和本地链路。
- standalone `engine-grpc` TTFT 默认按 ready/reused gRPC channel 统计，和 WebSocket 一样不把客户端建连成本计入首包延迟；cold/lazy channel 会额外增加约 10ms。
- `180ms 128-stream avg TTFT`：并发压测口径，需明确硬件、cache、输入、profile、采样参数和客户端测量方式。
- WebUI 只在结果 source 标记为 `live_triton` 或 `live_engine_websocket` 且带 `audio` 字段时代表可回放的实时合成音频。

详细 benchmark 口径见 [docs/zh/benchmark_methodology.md](docs/zh/benchmark_methodology.md)。

## 能力状态

| 路径 | 当前状态 | 开源口径 |
| --- | --- | --- |
| `custom-1.7b` / `custom_voice` | 优先稳定 | v0.1 推荐路径，WebUI 和 demo 默认围绕它展示 |
| `design-1.7b` / `voice_design` | 实验 | 可保留代码和导出入口，需标注未充分测通 |
| `base-1.7b` / x-vector voice clone | 实验 | standalone 已接入 ref audio → speaker embedding；需 base 导出产物和真实端到端验证 |
| `icl` voice clone | 实验 | standalone 已接入 ref audio + ref text → ref codec/code 注入；需 TRT ref-audio engine 和真实端到端验证 |
| `0.6b` variants | 未作为 v0.1 主线 | 可保留导出/下载入口，发布前需单独验证 |

## 快速开始

```bash
git clone --recursive https://github.com/user/Qwen3-TTS-Triton.git
cd Qwen3-TTS-Triton

# 交互模式
bash scripts/bash/autorun.sh

# 一次性跑本机完整流程（custom-1.7b + standalone + TensorRT）
bash scripts/bash/autorun.sh all -m custom-1.7b
```

三阶段：**Phase A** `setup_env.sh`（下载模型、安装环境、导出 ONNX/weights/manifest）→ **Phase B** `build_engines.sh`（trtexec 编译 TensorRT engine）→ **Phase C** `package` + `deploy`（组装模型包/镜像、启动服务）。

也可以分阶段执行，适合排查问题或复用已导出的产物：

```bash
bash scripts/bash/autorun.sh setup   -m custom-1.7b          # Phase A
bash scripts/bash/autorun.sh build   -m custom-1.7b          # Phase B
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway standalone --engine-mode trt  # Phase C1
bash scripts/bash/autorun.sh deploy  -m custom-1.7b --gateway standalone --engine-mode trt  # Phase C2
```

## 部署方式

详细参数（统一入口控制参数、Engine Profile 计算逻辑、GPU 选择、模型版本号）见 [docs/zh/deployment.md](docs/zh/deployment.md)。

### Standalone

本机 Python 运行 `engine.server`，适合调试 engine、协议和 WebSocket/gRPC。启动前会组装 `workspace/model_repository/tts_orchestrator/<model-version>` 模型包，然后通过 `--model-package-dir` 读取 `runtime/`、`weights/`、`tokenizer/` 和 manifest。

```bash
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway standalone --engine-mode trt
```

默认端口：gRPC `50051`，WebSocket `ws://localhost:50052/v1/ws`，HTTP capabilities `http://localhost:50052/v1/capabilities`，health `http://localhost:8080/health`。

### Engine Docker

独立 engine 容器，使用相同模型包，镜像包含运行时和 `/app/engine` 代码。

```bash
# 组装产物 + 重建镜像
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway engine-docker --build
# 启动服务
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway engine-docker --engine-mode trt
```

开发期可用 bind mount 或 watch 模式，避免频繁重建镜像：

```bash
bash scripts/bash/compose.sh up --gateway engine --variant custom-1.7b --dev
bash scripts/bash/compose.sh watch --gateway engine --variant custom-1.7b
```

engine-docker 当前要求模型包为 `--engine-mode trt`，因为 `engine.server` 消费的是 `runtime/model.plan`；Triton 仍可用同一包结构跑 `trt` 或 `onnx`。

### Triton

组装 `workspace/model_repository` 并启动 Triton：

```bash
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway triton --engine-mode trt
```

高级调试时可以直接使用 compose：

```bash
bash scripts/bash/compose.sh prepare --gateway triton --variant custom-1.7b --engine-mode trt
bash scripts/bash/compose.sh up --gateway triton --variant custom-1.7b
```

### Base / ICL 实验路径

部署 `base-1.7b` / `icl` 实验路径时，需准备默认参考音频和 reference registry：

```bash
mkdir -p workspace/default_refs
# 放入一段 3-10 秒、24k 或可重采样的 wav:
# workspace/default_refs/base_ref.wav

ENGINE_DEFAULT_BASE_REF_AUDIO_PATH=workspace/default_refs/base_ref.wav \
ENGINE_DEFAULT_BASE_REF_TEXT="参考音频对应文本" \
bash scripts/bash/autorun.sh all -m base-1.7b --gateway standalone --engine-mode trt
```

也可以在 `engine.yaml` 中配置 reference library 和 reference cache，详细字段语义和 ICL 预处理要求见 [docs/zh/deployment.md](docs/zh/deployment.md)。

## Client SDK

独立 Python SDK 包，统一访问 engine 和 Triton 端点，支持 engine-websocket / engine-grpc / triton-grpc / triton-http 四种传输。默认 `transport="auto"` 自动探测端点。

```bash
pip install qwen3-tts-client           # 核心包
pip install qwen3-tts-client[grpc]     # + gRPC 传输
pip install qwen3-tts-client[triton]   # + Triton 传输
pip install qwen3-tts-client[all]      # 全部传输 + audio
```

快速使用：

```python
from qwen3_tts_client import TTSClient, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/ws")
result = client.synthesize_bytes(
    "你好，欢迎使用 Qwen3-TTS。",
    request=SynthesisConfig(task_type="custom_voice"),
)
```

流式 session：

```python
from qwen3_tts_client import TTSClient, SessionStartRequest, SynthesisConfig

client = TTSClient.connect("localhost")
session = client.open_stream(
    SessionStartRequest(session_id="demo", config=SynthesisConfig(task_type="custom_voice"))
)
session.send_text("你好，")
session.send_text("这是流式输入。")
session.end()
for message in session.iter_messages():
    print(type(message).__name__, getattr(message, "meta", {}))
```

详细文档见 [docs/zh/client_sdk.md](docs/zh/client_sdk.md) 和 [`client/`](client) 子项目。

## 测试与验收

测试入口统一在 `tests/`，详细地图见 [tests/README.md](tests/README.md)。

```bash
# 单元 + 集成测试
pytest tests/unit tests/integration -q

# Serving 验收与 benchmark 主入口
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py --targets engine-grpc
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py --targets triton-grpc,triton-http
```

验证 base/icl reference resolver 与 ICL prefix cache：

```bash
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py \
  --targets engine-grpc \
  --reference-tests \
  --reference-alias vivian \
  --ref-audio-path workspace/default_refs/vivian.wav \
  --ref-text "这是一段与 vivian 参考音频完全一致的文本。"
```

## WebUI Demo

WebUI 包含三个板块：**Text Player**（按 engine decode step 播放文本，前半段 text token，flush 后显示 PAD step，合成完成后 slider seek 实际 WAV 音频）、**LLM PK**（模拟上游 LLM 逐 token 吐字，流式 vs 非流式同时间轴对比）、**Concurrency**（多路合成 TTFT 分布与吞吐，默认请求 live Triton 并保存真实音频）。

**Text Player**

![Text Player 演示](docs/images/文本播放器.gif)

**LLM PK**

![流式非流式对比演示](docs/images/流式非流式对比.gif)

**Concurrency**

![多路合成演示](docs/images/多路合成.gif)

完整录屏：[演示视频.mp4](docs/videos/演示视频.mp4)

一键启动（WebUI dev server、Demo API 和 Triton 都由 launcher 启动/复用）：

```bash
bash scripts/demo/start_webui_demo.sh --variant custom-1.7b
```

也可手动分步启动：

```bash
python -m demo_api --host 0.0.0.0 --port 7860   # Terminal 1
cd webui && npm install && npm run dev             # Terminal 2
```

浏览器打开 `http://localhost:5173`。如果 live backend 不可用，WebUI 展示 warning；音频按钮只在捕获到真实 waveform bytes 时启用，不使用嘟声占位。

Docker Compose demo profile：

```bash
bash scripts/bash/compose.sh up --gateway triton --variant custom-1.7b
docker compose --profile demo -f infra/docker/compose.yaml up --build demo-api webui
```

## 流式协议

standalone engine 同时支持 gRPC 和 WebSocket。WebSocket 控制帧示例：

```json
{"type":"start","session_id":"demo","config":{"task_type":"custom_voice","speaker":"Serena"}}
{"type":"text","text":"你好，世界。"}
{"type":"end"}
```

服务端返回 JSON event frame（协议事件、文本 token、边界、完成）和 Binary frame（PCM audio chunk，格式由 start/event 元数据声明）。

## 项目结构

```text
Qwen3-TTS-Triton/
├── engine/                     # 推理引擎：frontend/backend/gateway/core
├── client/                     # 独立 Python SDK 包 (pip install qwen3-tts-client)
│   ├── src/qwen3_tts_client/  #   客户端实现与传输适配器
│   └── src/qwen3_tts_protocol/ #  共享协议层（单一真相源）
├── demo_api/                   # WebUI Demo API（依赖 client 包）
├── webui/                      # Vite/React WebUI
├── infra/
│   └── docker/                 # Dockerfile + compose 配置
├── model_repository/           # Triton Python BLS 模型定义
├── scripts/
│   ├── bash/                   # autorun/setup/build/deploy 生命周期
│   ├── export/                 # PyTorch → ONNX/manifest 导出
│   └── python/                 # 配置/manifest/audit 工具
├── tests/
│   ├── unit/                   # pytest 单元测试
│   ├── integration/            # pytest 集成测试
│   ├── e2e/                    # pytest 端到端测试
│   └── support/                # 测试共享代码
├── tools/
│   ├── validation/             # 手动验证与 benchmark
│   ├── repro/                  # 冻结的 bug 复现案例
│   └── data/                   # 工具数据
├── docs/
│   ├── zh/                    # 中文用户文档（索引：docs/zh/README.md）
│   └── en/                    # 英文开发者文档（索引：docs/en/README.md）
└── workspace/                  # 运行时产物（gitignored）
```

## 文档导航

- 📖 [中文用户文档索引](docs/zh/README.md) — 部署、SDK、Benchmark、已知限制
- 📖 [英文开发者文档索引](docs/en/README.md) — 架构、设计、调查、运维

## 许可证

本项目基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 进行工程化部署与优化。模型权重和上游代码的许可证请以 QwenLM 官方仓库为准。
