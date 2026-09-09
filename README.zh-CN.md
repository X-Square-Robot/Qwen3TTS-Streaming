[English](README.md) | **中文**

<div align="center">

# Qwen3TTS-Streaming

*让我们像播放音频一样播放文本！*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/actions/workflows/ci.yml/badge.svg)](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-v0.1%20engineering%20preview-orange.svg)](#能力状态)
[![GitHub stars](https://img.shields.io/github/stars/X-Square-Robot/Qwen3TTS-Streaming?style=social)](https://github.com/X-Square-Robot/Qwen3TTS-Streaming)

<img src="docs/images/文本播放器.gif" width="720" alt="合并前的 Text Player 历史截图；当前统一门户请打开 /demo/">

<small><em>合并前的历史截图；当前浏览器体验统一由 `/demo/` 门户提供。</em></small>

*文本 token 进去，音频 chunk 实时出来。为什么这很关键见 [Token 级流式](#token-级流式)，然后到[内置 Demo](#内置-demo-与统一文档)直接试听。*

</div>

## 引言

Qwen3TTS-Streaming 是一个**工程预览版**项目：把官方 Qwen3-TTS PyTorch 权重导出为 ONNX/TensorRT 运行时，围绕 Triton/standalone engine 做**token 级流式 TTS**、模型 fuse、前端分词、prefix cache、连续批处理和内置产品 Demo。项目开放一条已高度优化、可复现、可继续验证的工程链路，让社区一起打磨成可靠的开源推理系统。

> ⚠️ **状态：v0.1 工程预览，非生产就绪。** 流式模式仍可能出现**幻觉、重复、漏读**（当前 checkpoint 上约 10–18%，根因在模型+采样，见 [已知限制](docs/user/known_limitations.zh-CN.md)）。**当前建议稳定范围为 `custom-1.7b` / `custom_voice` 路径**；`design-1.7b`、`base-1.7b` / x-vector 语音克隆、`icl` 语音克隆处于实验状态；`0.6b` 变体未作为 v0.1 主线。请勿直接用于生产内容生成。

## 特色

### Token 级流式

大多数 TTS 链路要等一整句话——甚至整段 LLM 回复——生成完才开始合成。Qwen3TTS-Streaming 在文本 token 到达的同时就开始合成，音频在句子还没写完时就已经开始播放：

```text
传统（句子级）TTS
  LLM  "你好，今天过得怎么样？"  ──（等整句生成完）──▶  TTS  ──▶  🔊
                                                              先长时间等待，再一次性播放

Qwen3TTS-Streaming（token 级）
  LLM   "你好" ─ "，" ─ "今天" ─ "过得" ─ "怎么样" ─ "？" ──▶
           │       │      │        │        │        │
           ▼       ▼      ▼        ▼        ▼        ▼
         chunk   chunk  chunk    chunk    chunk    chunk   ──▶  🔊
                                                                 首个 chunk ~15ms 到达
```

服务端 TTFT **14.9 ± 0.3ms**（n=50，最低 14.4ms），比一次 60Hz 屏幕刷新（16.7ms）还快，远低于人眼一次眨眼所需的约 100–400ms——第一个音频 chunk 播放时，你甚至还来不及感知到等待。128 路同时突发时客户端均值为 242–275ms（随传输方式而异）。这两个数字都带有前提条件，具体依赖见[性能声明](#性能声明)。

### 为自回归流式定制的调度器

Triton 内置的 `dynamic_batching` 假设请求无状态、序列长度固定——它没有"这个请求正在 decode 途中、还在等更多文本 token、并持有存活 KV 状态"这个概念。Token 级 TTS 恰恰需要这个：每个 session 的 KV 增长不均匀，decode 必须能在上游 LLM 卡顿时暂停（`WAIT_TEXT`）而不丢状态。

所以 engine 层放弃了 Triton 作为调度器，自己跑一套**迭代级连续批处理**：padded KV 对齐 + mask、MLFQ 式优先级（新 session 保首包延迟，长跑 session 降级而非饿死），以及按 session 挂起/恢复的三阶段 decode 循环——`prefill → 边等边流 → flush`。Triton 仍然是受支持的服务入口（`tts_orchestrator` BLS 模型只是同一个 engine 上的一层薄协议适配）——不管走哪条路，底层跑的都是这套调度器。

完整的"约束 → 设计"推导见[引擎设计全景总览](docs/dev/architecture/engine_overview.zh-CN.md)，里面标注了哪些取舍是模型硬约束、哪些还能继续优化。

### 编译一次，到处部署

TensorRT engine 和具体的 GPU/驱动/TensorRT 版本组合强绑定——一台机器编译出的 `.plan` 换台机器未必能跑。如果每台目标机都要直接编译，就意味着每台机器都得装完整 NGC 工具链（还得有 GPU），生产/边缘/内网机器往往不具备这个条件。

Qwen3TTS-Streaming 把"在哪编译"和"在哪部署"拆开：

```bash
bash scripts/bash/autorun.sh probe-target --out target_profile.json                              # 1. 采集目标机指纹
bash scripts/bash/autorun.sh make-bundle  -m custom-1.7b --target-profile target_profile.json     # 2. 编译匹配的产物包
bash scripts/bash/autorun.sh import-artifact workspace/engine_artifact_bundle.tar.zst              # 3. 目标机导入即可，无需 trtexec

# 或者一步通过 SSH 完成指纹采集 + 编译：
bash scripts/bash/autorun.sh remote-build -m custom-1.7b --target-profile target_profile.json --remote-host user@host
```

完整跨机编译流程见[部署指南](docs/user/deployment.zh-CN.md)。

### 一个引擎，而非四个

这套链路的一个 decode step 要经过四个不同阶段：talker backbone（prefill/decode）、Code Predictor、codec-embedding 求和、code2wav vocoder。如果每个阶段各自导出成一个 ONNX/TensorRT engine，就意味着每一步都要 4 次 Python 调度 + 4 次 host↔device 往返——而且流式过程中的每一步都要这样。

```text
不融合 —— 每个 decode step 4 个 engine
  talker  ──▶  code predictor  ──▶  codec_sum  ──▶  code2wav  ──▶  🔊
    4 次 Python 调度、4 次 host↔device 往返，每一步都是

Qwen3TTS-Streaming —— 每个 decode step 1 个融合 engine
  talker + code predictor + codec_sum + code2wav  ──▶  🔊
    1 张 ONNX 图、1 个 TensorRT engine、1 次调度
```

这不是 TensorRT 自带的算子融合——TRT 不会自己跨模型边界做合并。真正做"模型手术"的是项目自己的导出代码：[`export_09_talker_code2wav_fused.py`](scripts/export/export_09_talker_code2wav_fused.py) 里的 `TalkerCode2WavFusedONNX` 把四个阶段串成一次 forward，导出成一张图，编译成单个 `talker_code2wav_fused.engine`。另一处融合 [`export_04_speech_tokenizer_codec_fused.py`](scripts/export/export_04_speech_tokenizer_codec_fused.py) 把 speech tokenizer 和 codec-embedding 求和合并，用于参考音频路径。

整条链路都在本仓库里，不在 `third_party/` 子模块中——导出代码（`scripts/export/`）、TensorRT profile/IO 格式辅助（`scripts/python/trt_fused_*.py`），以及测试（`tests/integration/test_trt_fused_io_formats.py`、`tests/unit/engine_core/test_executor_trt_engine.py`）。

## 亮点

- ⚡ **Token 级流式，而非句子级** —— 首个音频 chunk ~15ms 到达（服务端 TTFT 14.9 ± 0.3ms），128 路突发均值 242–275ms
- 🧩 **为自回归 decode 定制的调度器**，而非 Triton 的无状态 `dynamic_batching` —— 连续批处理 + `WAIT_TEXT` 暂停/恢复
- 🌐 **编译一次，到处部署** —— 采集目标机指纹、编译匹配产物包、目标机零 GPU 工具链导入
- 🧵 **每个 decode step 一个 TensorRT engine，而非四个** —— talker + Code Predictor + codec-embedding 求和 + code2wav 融合进一张导出图，导图到测试全流程都在本仓库
- 🧠 **Prefix KV cache** —— 16 条 LRU 缓存，命中即跳过 prefill，省 10–50ms（见[引擎设计全景总览](docs/dev/architecture/engine_overview.zh-CN.md)）
- 🧮 **Code predictor 展开成单张静态 TRT 图** —— 无逐步 KV，比逐步解码有更高 GPU 利用率（见[引擎设计全景总览](docs/dev/architecture/engine_overview.zh-CN.md)）
- 🖥️ **内置产品 Demo** —— 无代码试听、参数调节、SDK 下载、统一文档与可选实验页

前四点详见[特色](#特色)；后两点见[引擎设计全景总览](docs/dev/architecture/engine_overview.zh-CN.md)。

## 目录

- [特色](#特色)
  - [Token 级流式](#token-级流式)
  - [为自回归流式定制的调度器](#为自回归流式定制的调度器)
  - [编译一次，到处部署](#编译一次到处部署)
  - [一个引擎，而非四个](#一个引擎而非四个)
- [性能声明](#性能声明)
- [能力状态](#能力状态)
- [前置要求](#前置要求)
- [快速开始](#快速开始)
- [部署方式](#部署方式)
- [Client SDK](#client-sdk)
- [测试与验收](#测试与验收)
- [内置 Demo 与统一文档](#内置-demo-与统一文档)
- [流式协议](#流式协议)
- [项目结构](#项目结构)
- [文档导航](#文档导航)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 性能声明

项目里提到的低延迟数字是有条件结果，不是通用承诺：

| 场景 | TTFT | 前提条件 |
| --- | --- | --- |
| 单路请求，warm engine | 服务端 **14.9 ± 0.3ms**（min 14.4，p99 15.9，n=50）；本地复用 gRPC channel 的客户端侧 p50 ~16.1ms | RTX 5090、warm engine、prefix-cache 命中、单路请求、本地链路、全 bf16 `custom-1.7b`、batch=128 profile |
| 128 路并发（同时突发，均值） | **242–275ms** 随传输方式而异（engine-websocket 242 / engine-grpc 275;p99 341–494ms）。Triton 路径在 2026-07-07/08 引擎优化后未重测（旧引擎核心上最近实测均值 309） | 同一套栈、单服务隔离运行、128 路全部接纳并同批解码;突发到达是最坏情况——错峰到达时 TTFT 更低 |

> ⚠️ **128 路并发是压测出来的天花板，不是生产安全值。** 经三轮 decode 优化（2026-07-06:CP 展开图内 KV + CUDA graph decode 回放 + KV gather arena 化;2026-07-07:突发批量准入 + 逐 slot 状态入池 + 服务热路径瘦身;2026-07-08:复盘审计修复批 + 批量化 p3_launch）后，压测 GPU（RTX 5090，全 bf16 引擎，batch=128 profile）在 128 路并发下每 80ms 音频帧的解码耗时 42.1ms——RTF（音频时长 / 实际解码耗时）≈ 1.90，即约 47% 的实时余量（优化前为 119.8ms/帧，RTF ≈ 0.67,低于实时）。这点余量能吸收正常抖动，但持续的负载尖峰或偏重的请求仍可能把它吃掉。生产环境的并发规划仍应在 128 之下留足 buffer，不要顶格跑；64 路时解码耗时 24.5ms（RTF ≈ 3.3），余量充足。完整拆解与原始数据见[服务性能压测报告](docs/dev/investigation/serving_performance_benchmark.zh-CN.md)。

- standalone `engine-grpc` TTFT 默认按 ready/reused gRPC channel 统计，和 WebSocket 一样不把客户端建连成本计入首包延迟；cold/lazy channel 会额外增加约 13ms。
- Demo 实验页的单次浏览器指标与上表不是同一测量窗口或负载形态，不能直接对比；公开性能口径必须使用带完整条件的 benchmark 数据。
- 产品 Demo 只调用当前实例公共 `/v1/realtime`；live backend 不可用时明确失败，不回退到 fixture 或模拟音频。

详细 benchmark 口径见 [Benchmark 方法](docs/user/benchmark_methodology.zh-CN.md)。

## 能力状态

| 路径 | 当前状态 | 开源口径 |
| --- | --- | --- |
| `custom-1.7b` / `custom_voice` | 🟢 优先稳定 | v0.1 推荐路径，产品 Demo 默认围绕它展示 |
| `design-1.7b` / `voice_design` | 🟡 实验 | 可保留代码和导出入口，需标注未充分测通 |
| `base-1.7b` / x-vector voice clone | 🟡 实验 | standalone 已接入 ref audio → speaker embedding；需 base 导出产物和真实端到端验证 |
| `icl` voice clone | 🟡 实验 | standalone 已接入 ref audio + ref text → ref codec/code 注入；需 TRT ref-audio engine 和真实端到端验证 |
| `0.6b` variants | ⚪ 未作为 v0.1 主线 | 可保留导出/下载入口，发布前需单独验证 |

## 前置要求

- **GPU**：NVIDIA GPU，建议 ≥16GB 显存（1.7B + KV pool + TensorRT 运行时）；需匹配的 NVIDIA 驱动。
- **CUDA / TensorRT**：通过 NVIDIA NGC 容器提供（`nvcr.io/nvidia/tensorrt`、`nvcr.io/nvidia/tritonserver`）；版本矩阵见 `scripts/bash/ngc_matrix.conf`。**拉取 NGC 镜像即表示接受 NVIDIA EULA。**
- **Docker**：用于引擎/Triton 容器编排（含 NVIDIA Container Toolkit 以启用 `--gpus`）。
- **Python 环境**：Phase A 通过 conda 管理宿主机 Python 环境；若无可用环境，`setup_env.sh` 会下载安装 [Miniforge](https://github.com/conda-forge/miniforge)（BSD-3-Clause）并创建 `qwen3-tts` conda 环境。你也可以事先自行激活 conda 环境或 venv。
- **磁盘**：模型 + 导出/编译产物约需 20–40GB。
- **模型权重**：首次运行需从 ModelScope / Hugging Face 下载（见下方流程），本仓库不分发权重。

> 首次端到端跑通包含「下载权重 → 导出 ONNX → 编译 TensorRT」，耗时取决于 GPU；后续可复用产物或跨机导入。

## 快速开始

```bash
git clone --recursive https://github.com/X-Square-Robot/Qwen3TTS-Streaming.git
cd Qwen3TTS-Streaming

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

详细参数（统一入口控制参数、Engine Profile 计算逻辑、GPU 选择、模型版本号）见 [部署指南](docs/user/deployment.zh-CN.md)。

### Standalone

本机 Python 运行 `engine.server`，适合调试 engine、协议和 WebSocket/gRPC。启动前会组装 `workspace/model_repository/tts_orchestrator/<model-version>` 模型包，然后通过 `--model-package-dir` 读取 `runtime/`、`weights/`、`tokenizer/` 和 manifest。

```bash
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway standalone --engine-mode trt
```

默认端口：gRPC `50051`，原生 WebSocket `ws://localhost:50052/v1/ws`，OpenAI Realtime 兼容入口 `ws://localhost:50052/v1/realtime`，HTTP capabilities `http://localhost:50052/v1/capabilities`，health `http://localhost:8080/health`（进程启动即监听；模型加载期间返回 `503`，就绪后返回 `200`，探针细节见[部署文档](docs/user/deployment.zh-CN.md)）。

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

Triton 部署会同时启动 Triton 和公共 WebSocket sidecar。默认公共入口为原生 WebSocket
`ws://localhost:50053/v1/ws`、OpenAI Realtime 兼容入口
`ws://localhost:50053/v1/realtime`、capabilities
`http://localhost:50053/v1/capabilities` 和 health
`http://localhost:50053/health`；Triton 原生 HTTP/gRPC/metrics 端口仍为
`8000/8001/8002`。可用 `compose.sh --realtime-port` 修改宿主机端口。完整和部分
response 的 usage 都会在协议中返回，并追加到
`workspace/realtime_usage/realtime_usage.jsonl`，供计费系统消费。

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

也可以在 `engine.yaml` 中配置 reference library 和 reference cache，详细字段语义和 ICL 预处理要求见 [部署指南](docs/user/deployment.zh-CN.md)。

## Client SDK

Python SDK 在 `transport="auto"` 时优先选择原生 `engine-websocket`。OpenAI Realtime
作为兼容 transport 保留且不发出弃用告警；engine-grpc、triton-grpc、triton-http 三种
旧直连 transport 继续可用并发出弃用告警。

引擎与 SDK 从同一个 git tag 配对发布。先从 `GET /v1/capabilities` 读取
`engine_version`，再安装对应 GitHub 或 GitLab Release 中的 wheel：

```bash
curl http://<engine-host>:<ws-port>/v1/capabilities
# → {"engine_version": "v0.1.0", ...}

# 精确且可直接复制的安装命令见当前实例的 /demo/#/sdk 页面。
# 也可在对应版本的发布页选择 wheel：
# https://github.com/X-Square-Robot/Qwen3TTS-Streaming/releases

# 引擎分发的也是同一个已发布 wheel。
curl https://<public-service-base>/sdk/    # 先看列表，再使用页面返回的相对链接：
pip install "https://<public-service-base>/sdk/<wheel-filename>"

# 或从本地检出安装
pip install "./client[all]"
```

快速使用：

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/ws")
result = client.synthesize_bytes(
    "你好，欢迎使用 Qwen3-TTS。",
    request=SynthesisConfig(task_type="custom_voice"),
)
print(result.details["usage"])
```

流式 session：

```python
from qwen3tts import TTSClient, SessionStartRequest, SynthesisConfig

client = TTSClient.connect("localhost")
session = client.open_stream(
    SessionStartRequest(session_id="demo", config=SynthesisConfig(task_type="custom_voice"))
)
session.send_text("你好，")
session.send_text("这是流式输入。")
session.end()
for message in session.iter_messages():
    print(type(message).__name__, getattr(message, "meta", {}))
print(session.response_id, session.response_status, session.usage)
```

详细文档见 [Client SDK](docs/user/client_sdk.zh-CN.md) 和 [`client/`](client) 子项目。

## 测试与验收

测试入口统一在 `tests/`，详细地图见 [tests/README.md](tests/README.zh-CN.md)。

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

## 内置 Demo 与统一文档

每个正式运行时镜像都在 `/demo/` 内置与该版本匹配的实例门户。Demo 默认开启，
与 `/v1/ws`、`/v1/realtime`、`/sdk/` 使用同一个公共端口；启动时设置
`DEMO_ENABLED=false` 可将其关闭。
门户会发现当前实例的能力，通过 Browser SDK 合成，通过系统扬声器播放，并提供由
capabilities 门控的 VAD/交付参数、WAV 下载和本仓库 Markdown 文档；普通体验不依赖
独立 Demo API 或 Node 进程。

需要修改门户的贡献者请先阅读
[`web/packages/demo` 开发说明](web/packages/demo/README.zh-CN.md)。

原独立的 `webui/` 特性展示前端已并入 `web/packages/demo`；现在只有这一套浏览器前端和
一个门户入口。统一工程实验从 `/demo/#/lab` 进入：基础 LLM PK 与并发实验使用实例
公共 `/v1/realtime`；trace、LLM PK、多路并发和能力查询都通过统一 Gateway 完成，
不直接连接 Triton，也不需要独立实验后端。

内置“实验”页通过公共 Realtime 提供 **LLM PK**、并发和事件 trace。结果仅代表当前浏览器
到当前实例的本次请求，不展示硬编码性能数字，也不会用 fixture 或模拟音频替代 live backend。

以下媒体是原独立 WebUI 的历史截图，仅用于说明实验形态；其中的布局和数字都不是当前 UI
或 benchmark 声明。

**LLM PK 历史演示素材**

![流式非流式对比演示](docs/images/流式非流式对比.gif)

**Concurrency 历史演示素材**

![多路合成演示](docs/images/多路合成.gif)

完整录屏：[演示视频.mp4](docs/videos/演示视频.mp4)

Standalone 启动：

```bash
bash scripts/bash/compose.sh up --build --gateway engine --variant custom-1.7b
```

浏览器打开 `http://localhost:50052/demo/`。Triton 部署改用 `--gateway triton`，
然后打开 `http://localhost:50053/demo/`。服务挂在 `/infer/<instance>` 等反向代理前缀下
时，门户、SDK、WebSocket 和静态资源链接仍会保留该前缀。
该运行时路径不需要启动 Vite 开发服务器或第二套 WebUI 进程。

HTTP/WS 是默认本地协议；挂载证书不会自动启用 HTTPS。需要直接调试自签名 WSS 时，
Python SDK 可用 `tls_verify="/path/to/cert.local.pem"` 严格信任指定证书，或仅在临时
联调中使用 `tls_verify=False`。浏览器与 Python 的证书信任相互独立。

Kubernetes 只允许一个公开端口时，engine 容器设置
`PORT=8000 HEALTH_PORT=0`，Service 只映射 `8000`；此时
`/demo/`、`/sdk/`、`/health`、`/v1/ws` 和 `/v1/realtime` 全部共用该端口。完整探针和 Service
示例见[部署说明](docs/user/deployment.zh-CN.md#kubernetes-单端口部署)。

没有 Ingress 的开发机也可像 FunASR Nano 一样设置 `TLS_CERT_FILE` 和
`TLS_KEY_FILE`，或为内置本地证书设置 `TLS_AUTO_ENABLE=true`，由同一个公共端口直接
提供 HTTPS/WSS；具体证书挂载方式见
[开发机直接启用 HTTPS/WSS](docs/user/deployment.zh-CN.md#开发机直接启用-httpswss)。

内置“实验”页通过同一个公共 Realtime 入口提供 **LLM PK**、并发和事件 trace：

```bash
bash scripts/bash/compose.sh up --build --gateway engine --variant custom-1.7b
```

## 流式协议

官方 Python SDK 以原生 `/v1/ws` 为主协议，`/v1/realtime` 是 OpenAI Realtime 兼容入口。兼容入口是全双工 WebSocket，完整文本使用 `conversation.item.create` + `response.create`，token 级追加使用 `qwen.input_text_buffer.append/commit` 扩展。完整说明见 [OpenAI Realtime TTS 协议与 Triton 边界](docs/dev/architecture/openai_realtime.zh-CN.md)。

原生 `/v1/ws` 使用以下控制帧：

```json
{"type":"start","session_id":"demo","config":{"task_type":"custom_voice","speaker":"Serena"}}
{"type":"text","text":"你好，世界。"}
{"type":"stop"}
```

`stop` 表示优雅停止输入并排空音频（`end` 仍是兼容别名），`cancel` 表示中止当前
session。服务端返回 JSON event frame 和 Binary PCM frame。逻辑 session 以
`done`/`error` 事件为边界，而不是以 socket 关闭为边界；同一 WebSocket 随后可再次
接收 `start`。成功完成或取消产生的 `done` 可复用连接；engine `error` 会关闭连接，
下一 session 重新建立。

活动 WebSocket 流还支持有界的进程内断线恢复。SDK 发送带序号的文本并确认精确的
输出 delivery；网络/代理瞬断后，它从连接池租用替换 socket，并从最后完整音频样本
继续同一个 engine execution。它不会从头重启合成再猜测去重。恢复状态只存在于当前
服务进程且会过期，因此 engine 重启会明确失败；多副本部署需要 sticky routing 或按
token 的一致性路由。

## 项目结构

```text
Qwen3TTS-Streaming/
├── engine/                     # 推理引擎：frontend/backend/gateway/core
├── client/                     # 独立 Python SDK 包（qwen3-tts-client，以 wheel 发布）
│   ├── src/qwen3tts/           #   客户端实现与传输适配器
│   └── src/qwen3tts_protocol/  #   共享协议层（单一真相源）
├── web/                        # Browser SDK 与唯一 React/Vite 产品门户
│   ├── packages/browser-sdk/   # Browser SDK 包
│   └── packages/demo/          # 统一 /demo/ 门户（体验/SDK/文档/实验）
├── proto/                      # 协议定义唯一源（tts.proto + 生成代码）
├── model_repository/           # Triton Python BLS 模型定义
├── infra/
│   └── docker/                 # Dockerfile + compose 配置
├── scripts/
│   ├── bash/                   # autorun/setup/build/deploy 生命周期
│   ├── compose/                # 容器入口点脚本
│   ├── demo/                   # Demo / 工程实验启动脚本
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
│   ├── user/                   # 用户文档（部署、SDK、Benchmark、限制）
│   ├── dev/                    # 开发者文档（架构、设计、调查、运维）
│   └── process/                # 流程/历史文档（归档）
├── resources/                  # 静态资源（合成参考音频等）
├── third_party/                # git 子模块（Qwen3-TTS 上游，Apache-2.0）
└── workspace/                  # 运行时产物（gitignored）
```

## 文档导航

- 📖 [用户文档](docs/user/README.zh-CN.md) — 部署、SDK、Benchmark、已知限制
- 📖 [开发者文档](docs/dev/README.zh-CN.md) — 架构、设计、调查、运维

## 参与贡献

本项目是 **v0.1 工程预览**，流式质量仍在打磨，欢迎以 issue、讨论、PR 形式参与。

- 🤝 [贡献指南](CONTRIBUTING.zh-CN.md) — 开发环境、测试、proto 工作流、代码风格
- 💬 [求助渠道](SUPPORT.zh-CN.md) — 提问 / 报 Bug / 提建议如何分流
- 🔒 [安全策略](SECURITY.zh-CN.md) — 漏洞私密报告流程（请勿公开提交 issue）
- 📜 [行为准则](CODE_OF_CONDUCT.zh-CN.md) — Contributor Covenant 2.1
- 📝 [变更日志](CHANGELOG.zh-CN.md) — 版本变更记录

## 许可证

- **本项目自有代码**（`engine/`、`client/`、`web/`、`scripts/` 等）按 [MIT](LICENSE) 许可证发布，版权归 XSquareRobot。
- **上游 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)**（`third_party/` 子模块）为 Apache 2.0，与 MIT 兼容。
- **模型权重**由 Qwen/Alibaba 发布，许可证以其 [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) / [Hugging Face](https://huggingface.co/Qwen) 模型卡为准；本仓库不分发任何权重。
- **TensorRT / Triton Inference Server**（NVIDIA NGC 镜像）为 NVIDIA 专有软件，本仓库不打包，使用即表示接受 NVIDIA EULA。
- **[TEN VAD](https://github.com/TEN-framework/ten-vad)** 为**可选**依赖，仅实验性 `tenvad` VAD 模式使用（默认关闭，由用户自行安装、不打包）。其许可为 **Apache 2.0 + 附加条件**（非竞争、仅限自用），**非**标准宽松许可；启用该模式前请先阅读其条款。
- `resources/speakers/` 下的参考音频为**合成音频**、说话人名为**虚构**，不对应任何真实个人。

完整第三方归属见 [NOTICE](NOTICE)。
