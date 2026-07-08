[English](deployment.md) | **中文**

# 部署说明

本文档说明当前推荐的部署方式、profile 参数和开发期容器策略。

## 推荐路径

v0.1 推荐先使用：

```bash
bash scripts/bash/autorun.sh all -m custom-1.7b
```

首次部署建议分阶段执行，便于定位问题：

```bash
bash scripts/bash/autorun.sh setup -m custom-1.7b
bash scripts/bash/autorun.sh build -m custom-1.7b
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway standalone
```

## 统一入口与控制参数

生命周期统一入口是 `bash scripts/bash/autorun.sh`，它同时支持两种方式：

- 交互式：`bash scripts/bash/autorun.sh`
- 一次性调用：`bash scripts/bash/autorun.sh <command> -m <variant> [options]`

所有关键控制项都可以从 autorun 进入：导出 GPU、TensorRT 编译 GPU、engine profile、runtime 上限、部署方式和端口。

Phase C 被拆成两个显式命令：

```bash
# 只组装部署产物，不启动服务；跨机/打包机场景用这个
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway engine-docker

# 只在当前机器启动服务；当前机器不是生产服务机时不要执行这个
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway engine-docker
```

`autorun.sh all` 是"本机完整流程"，会执行 `setup → build → package → deploy`。如果你是在导图/打包机上为云端生产容器准备产物，通常应在导入云端返回的 engine artifact 后执行 `package`，然后把镜像和模型包交给生产部署系统；不要在打包机上执行 `deploy`。

配置优先级是：命令行参数 > 已导出的环境变量 > manifest/default。常用环境变量包括 `MODEL_VERSION`、`EXPORT_DEVICE`、`BUILD_GPU_DEVICE`、`RUNTIME_GPU_DEVICE`、`MAX_BATCH_SIZE`、`MAX_INPUT_LEN`、`MAX_SEQ_LEN`、`RUNTIME_MAX_BATCH_SIZE`、`RUNTIME_MAX_SEQ_LEN`；但推荐日常都从 autorun 参数进入，便于复现。

### Phase B 参数

```text
Phase B:
  --max-batch-size <N>          TensorRT profile 最大 batch
  --max-input-len <N>           prefill/input 最大 token 长度
  --max-seq-len <N>             KV cache 最大 sequence 长度
  --dtype bf16|fp16|fp32|fp8    TensorRT build precision（各子模块默认基准）
  --cp-precision <T>            Code Predictor 精度（默认跟随 --dtype，即 bf16）
  --code2wav-precision <T>      Code2Wav 精度（默认 fp16——卷积更快；bf16 用于全 bf16 构建）
  --triton-io-float-dtype <T>   TensorRT/Triton float I/O dtype，默认等于 --dtype
  --target-driver <ver>         按部署机 NVIDIA driver 选择 NGC 镜像
  --build-device <dev>          trtexec 编译 GPU
```

### Phase C 参数

```text
Phase C:
  --gateway standalone|triton|engine-docker
  --engine-mode trt|onnx
  --runtime-max-batch-size <N>  runtime scheduler batch 上限
  --runtime-max-seq-len <N>     runtime scheduler seq 上限
  --runtime-device <dev>        runtime 服务 GPU
```

### 模型版本号

默认会组装 Triton model version 目录 `1`。如果需要生成其他版本目录，可以通过 `--model-version <N>` 指定；这会把共享模型包放到 `workspace/model_repository/tts_orchestrator/<N>`，并让 standalone、engine Docker 和 Triton 都从 `/models/tts_orchestrator/<N>` 读取。

```bash
bash scripts/bash/autorun.sh deploy -m custom-1.7b \
  --gateway triton \
  --model-version 2
```

这个版本号是 Triton model repository 的版本目录，不是 Hugging Face / ModelScope 权重 revision。HTTP 客户端如果显式带版本，需要请求 `/v2/models/tts_orchestrator/versions/<N>/infer`；不显式带版本时则由 Triton 按仓库状态选择可用版本。

## GPU 选择

默认 `--device auto`：脚本会选择当前空闲显存最多的 GPU。你也可以显式指定同一张卡用于所有阶段：

```bash
bash scripts/bash/autorun.sh all -m custom-1.7b --device 1
```

也可以按阶段拆开指定：

```bash
bash scripts/bash/autorun.sh all -m custom-1.7b \
  --export-device auto \
  --build-device 1 \
  --runtime-device 1
```

参数含义：

```text
--device <dev>          同时作用于导出、编译、运行阶段；dev 可为 auto、0、1、cuda:1
--export-device <dev>   仅 Phase A 导出模型使用；额外支持 cpu
--build-device <dev>    仅 Phase B trtexec 编译 engine 使用；支持 auto、all、0、1、cuda:1
--runtime-device <dev>  仅 Phase C 服务运行使用；支持 auto、0、1、cuda:1
```

Phase B 会在 Docker 层限制构建 GPU，例如 `--build-device 1` 会使用类似 `docker run --gpus device=1 ...` 的方式运行 `trtexec`。因此 `trtexec` 日志里可能显示容器内 `Selected Device ID: 0`，但 UUID 会对应物理 GPU 1。

## Phase A: 导出

Phase A 下载模型、安装依赖、导出 ONNX/weights/manifest。

```bash
bash scripts/bash/autorun.sh setup -m custom-1.7b
```

常用参数：

```text
--source auto|hf|modelscope
--skip-models
--skip-deps
--skip-export
--target-driver <driver>
```

## Phase B: 构建 TensorRT engine

Phase B 在 NGC 容器里运行 trtexec，并把实际 profile 写入 manifest：

```bash
bash scripts/bash/autorun.sh build -m custom-1.7b \
  --max-batch-size 64 \
  --max-input-len 128 \
  --max-seq-len 512 \
  --dtype bf16
```

写入位置：

```text
workspace/exported/custom-1.7b/triton_manifest.json
```

关键字段：

```json
{
  "engine_profile": {
    "engine_mode": "trt",
    "engine_dtype": "bf16",
    "triton_io_float_dtype": "bf16",
    "max_batch_size": 64,
    "max_input_len": 128,
    "max_seq_len": 512,
    "builder_image": "nvcr.io/nvidia/tritonserver:26.02-py3"
  }
}
```

runtime 的 batch/seq 不能超过这里的 profile。需要更大 batch 或更长文本时，重新跑 Phase B。

### Engine Profile 详细计算逻辑

如果不显式传 `--max-batch-size`、`--max-input-len`、`--max-seq-len`，Phase B 会优先读取
`workspace/exported/<variant>/triton_manifest.json` 和导出的权重/engine/ONNX 文件，估算：

- 固定占用：TRT/ONNX 主模型、runtime embedding 权重、必要的 reference preprocessing engine
- 每路持久状态：Talker KV pool、Code2Wav KV、conv/transconv 双缓冲、token_counts
- 每步峰值：batched talker KV 输入、C2W KV/state 输入、TRT 输出缓存和少量采样/attention scratch

然后结合目标机器 `target_profile.json` 里的 GPU 总显存，向下取到支持的 profile 档位：

```text
16 / 32 / 64 / 128
```

因此跨机编译时 profile 应以生产机 `target_profile.json` 和导出产物估算为准，而不是打包机显存。例如 `custom-1.7b` 在 48G 目标卡上，如果固定占用、TRT 自留和 KV/cache 估算后仍满足余量，默认建议可以落到 `max_batch_size=128`。

如果导出 manifest 不存在，才回退到粗略显存档位：

```text
约 24 GB GPU:  max_batch=16   max_input_len=96   max_seq_len=384
约 32 GB GPU:  max_batch=32   max_input_len=128  max_seq_len=512
约 48 GB GPU:  max_batch=64   max_input_len=128  max_seq_len=512
约 80 GB GPU:  max_batch=128  max_input_len=128  max_seq_len=512
```

这只是默认建议，不是限制；显式参数仍然最高优先级。比如你可以在 24G 机器上为 48G 部署机尝试构建更大的 profile：

```bash
bash scripts/bash/autorun.sh build -m custom-1.7b \
  --build-device 1 \
  --max-batch-size 64 \
  --max-input-len 128 \
  --max-seq-len 512
```

但 TensorRT 编译本身也需要显存。如果构建机显存不足，`trtexec` 仍可能 OOM；这时需要换更大构建卡、释放显存，或降低 profile。

这些值会写入 `workspace/exported/<variant>/triton_manifest.json` 的 `engine_profile` 字段。runtime 启动时如果请求的 batch/seq 超过 profile，会直接报错；prefill 长度超过 `max_input_len` 时也会报出明确错误，避免 silent clamp 或运行时才暴露 TensorRT shape 问题。

示例：

```bash
# 构建较小 profile，便于低显存机器验证
bash scripts/bash/autorun.sh build -m custom-1.7b \
  --max-batch-size 16 --max-input-len 96 --max-seq-len 384

# runtime 使用不得超过 manifest 里记录的 profile
bash scripts/bash/autorun.sh deploy -m custom-1.7b \
  --gateway standalone \
  --runtime-max-batch-size 16 \
  --runtime-max-seq-len 384

# Triton gateway 也走同一套 runtime 上限和 GPU 入口
bash scripts/bash/autorun.sh deploy -m custom-1.7b \
  --gateway triton \
  --runtime-device 1 \
  --runtime-max-batch-size 16 \
  --runtime-max-seq-len 384
```

## Phase C: 启动服务

### Standalone

```bash
bash scripts/bash/autorun.sh deploy \
  --gateway standalone \
  -m custom-1.7b \
  --max-batch 32 \
  --max-seq-len 512
```

端点：

- gRPC: `localhost:50051`
- WebSocket: `ws://localhost:50052/v1/ws`
- capabilities: `http://localhost:50052/v1/capabilities`
- health: `http://localhost:8080/health`

`base` / `icl` reference preprocessing 在 standalone TRT 路径中由
`speaker_encoder.engine`、`speech_tokenizer_codec_fused.engine` 和可选
`code2wav_decoder.engine` 串行执行，目前不做 batch。`spliter.max_concurrent_segments`
只影响后续文本分段和 EngineLoop slot 并发，不控制 speech encoder。reference
音频最大时长以 capabilities 中的 `ref_audio_max_duration_sec` 为准，当前
TRT 构建默认是 8 秒。`reference_cache` 缓存 ref-audio preprocessing
features，`prefix_cache` 缓存 Talker ICL prefix KV，两者独立配置。启用
`reference_cache` 时，standalone engine 会在加载主 `model.plan` 之前预热
`references.default` 和 registry entries，并在每个 preprocessing 阶段后释放
ref TRT engine，减少 24GB 级显存上 request path 再加载 speaker/codec engine
触发 OOM 的概率。

### Engine Docker

```bash
bash scripts/bash/autorun.sh deploy \
  --gateway engine-docker \
  -m custom-1.7b
```

这个模式使用 engine 镜像作为固定应用层：镜像内包含 TensorRT/Python
运行时、`/app/engine` 引擎代码、默认 `/app/engine.yaml` 和启动脚本；
运行时只读挂载和 Triton 相同的模型包
`workspace/model_repository/tts_orchestrator/1`。普通镜像里已经包含
`/app/engine` 代码，因此代码更新后需要重新构建并发布镜像。

本机 standalone gateway 也使用同一个模型包，只是由本机 Python 承载
`engine.server`。也就是说，standalone、engine-docker 和 Triton 的模型产物
都是 `model_repository/tts_orchestrator/1`，差异只在运行时进程和容器层。
Phase C assemble 会同步仓库 `resources/` 到
`model_repository/tts_orchestrator/<version>/resources/`。Base/ICL 的
reference registry 可以直接使用模型包相对路径，例如
`resources/speakers/<alias>/ref.wav` 和 `resources/speakers/<alias>/ref.txt`
（通过 `ref_text_path` 配置）。

生产环境推荐把"应用镜像、模型产物、部署配置"拆成三层：

- 应用镜像：`qwen3-engine:<tag>`，包含依赖和引擎代码，不挂载源码目录。
- 模型产物：统一的 Triton-compatible `model_repository`。engine 和 Triton
  都从 `/models/tts_orchestrator/1` 读取 `runtime/`、`weights/`、
  `tokenizer/` 和 manifest。engine-docker 当前要求 `runtime/model.plan`
  这种 `trt` 包。
- 部署配置：端口、batch、seq_len、session、speaker 默认值等，通过
  `ENGINE_CONFIG` 指向只读挂载的 YAML，或通过 `ENGINE_*` 环境变量覆盖。

示例：

```bash
ENGINE_CONFIG=/etc/qwen3-tts/engine.yaml \
ENGINE_CONFIG_FILE=/srv/qwen3/config/engine.yaml \
MODEL_REPO_DIR=/srv/qwen3/model_repository \
bash scripts/bash/autorun.sh deploy --gateway engine-docker -m custom-1.7b
```

如果直接写 compose volume，可挂载：

```yaml
volumes:
  - /srv/qwen3/model_repository:/models:ro
  - /srv/qwen3/config/engine.yaml:/etc/qwen3-tts/engine.yaml:ro
environment:
  ENGINE_CONFIG: /etc/qwen3-tts/engine.yaml
```

开发期推荐改用：

```bash
bash scripts/bash/compose.sh up --gateway engine --variant custom-1.7b --dev
```

或：

```bash
bash scripts/bash/compose.sh watch --gateway engine --variant custom-1.7b
```

这样可以把环境层和代码层拆开，避免每次改 Python 代码都重新构建依赖镜像。

### Triton

```bash
bash scripts/bash/autorun.sh package \
  --gateway triton \
  -m custom-1.7b \
  --engine-mode trt

bash scripts/bash/autorun.sh deploy \
  --gateway triton \
  -m custom-1.7b
```

Triton 默认端口：

- HTTP: `localhost:8000`
- gRPC: `localhost:8001`
- Metrics: `localhost:8002`

## WebUI

本地运行：

```bash
python -m demo_api --host 0.0.0.0 --port 7860

cd webui
npm install
npm run dev
```

Compose 运行：

```bash
bash scripts/bash/autorun.sh deploy --gateway triton -m custom-1.7b
docker compose --profile demo up --build demo-api webui
```

## 常见问题

### runtime max_seq_len 超过 profile

错误类似：

```text
runtime max_seq_len=1024 exceeds engine profile max_seq_len=512
```

解决方式：

- 降低 `--max-seq-len` / `ENGINE_SCHEDULER_MAX_SEQ_LEN`。
- 或重新构建 engine：`bash scripts/bash/autorun.sh build -m custom-1.7b --max-seq-len 1024`。

### dtype 不匹配

如果 Triton 报 `TYPE_FP32` / `TYPE_BF16` 之类错误，确保：

- Phase B 的 `--dtype` 和 `--triton-io-float-dtype` 符合目标。
- `triton_manifest.json` 已被 Phase B 更新。
- 重新 assemble model_repository。

### TensorRT plan 无法反序列化

TensorRT plan 与 runtime 版本强绑定。更换 TensorRT/NGC image 后需要重新构建 engine。

### WebUI 显示 fixture fallback

说明 live Triton 或 live engine 当前不可达。检查：

- Triton gRPC 端口是否是 `localhost:8001`。
- demo API 的 `QWEN_DEMO_TRITON_GRPC` 是否正确。
- standalone engine WebSocket 是否是 `localhost:50052`。
