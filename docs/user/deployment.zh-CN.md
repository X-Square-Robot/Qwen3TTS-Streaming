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

不带 command 启动 `autorun.sh` 时，TUI 的“发布与打包信息”区可以直接输入模型发布版本、引擎编译版本、打包人和打包日期。相同字段也提供非交互参数：

```bash
bash scripts/bash/autorun.sh package -m custom-1.7b \
  --model-release-version 'zehan@20260818' \
  --engine-build-version 'rime@20260820_580_5090_v1' \
  --packager rime \
  --package-date 2026-08-20
```

`--model-version 2` 仍只表示 Triton 数字目录；`--model-release-version` 才是模型包中的 `MODEL_VERSION`。`all` 和 `all-1.7b` 不接受统一的 `--model-release-version`，避免把多个模型错误地重标成同一版本。

### Phase B 参数

```text
Phase B:
  --max-batch-size <N>          TensorRT profile 最大 batch
  --max-input-len <N>           prefill/input 最大 token 长度
  --max-seq-len <N>             KV cache 最大 sequence 长度
  --dtype bf16|fp16|fp32|fp8    TensorRT build precision（各子模块默认基准）
  --cp-precision <T>            Code Predictor 精度（默认跟随 --dtype，即 bf16）
  --code2wav-precision <T>      Code2Wav 精度（默认跟随 --dtype/bf16；fp16 仅低并发 opt-in）
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
  --realtime-port <N>           Triton OpenAI Realtime 宿主机端口（默认 50053）
```

### 模型版本号

默认会组装 Triton model version 目录 `1`。如果需要生成其他版本目录，可以通过 `--model-version <N>` 指定；这会把共享模型包放到 `workspace/model_repository/tts_orchestrator/<N>`，并让 standalone、engine Docker 和 Triton 都从 `/models/tts_orchestrator/<N>` 读取。

```bash
bash scripts/bash/autorun.sh deploy -m custom-1.7b \
  --gateway triton \
  --model-version 2
```

这个版本号是 Triton model repository 的版本目录，不是 Hugging Face / ModelScope 权重 revision。HTTP 客户端如果显式带版本，需要请求 `/v2/models/tts_orchestrator/versions/<N>/infer`；不显式带版本时则由 Triton 按仓库状态选择可用版本。

模型自身的发布版本使用另一套、随模型走的标识。源模型目录可以携带只读文本文件 `MODEL_VERSION`；也可以在 autorun TUI 中直接输入，由导出阶段写入只读版本文件。内容为单行版本号，例如：

```text
zehan@20260601
```

Phase A 会把它复制到 `workspace/exported/<variant>/MODEL_VERSION`；TUI 输入的显式版本优先，并写入同一位置。Phase C 再复制到模型包根目录 `tts_orchestrator/<N>/MODEL_VERSION`，两处文件都设置为 `0444`。组装、仓库校验和引擎启动都会拒绝缺失或空的文件。引擎编译版本独立管理，可在 TUI 中输入或用 `--engine-build-version` 指定；autorun 会把它写入只读 `ENGINE_BUILD_VERSION`，Phase C 将其带入模型包，运行时直接从包内读取。

模型包的打包溯源与模型发布版本分开保存。Phase C 会在同一模型包根目录生成只读的 `PACKAGE_INFO.json`：

```json
{
  "package_info_schema_version": 1,
  "packager": "rime",
  "packaged_on": "2026-08-20"
}
```

`packaged_on` 只精确到天，并严格使用 ISO `YYYY-MM-DD` 格式，不包含时分秒或时区。默认打包人取当前系统用户，默认打包日期取本机当天日期；CI 或可复现打包可分别用 `QWEN3_TTS_PACKAGER` 和 `QWEN3_TTS_PACKAGE_DATE` 显式传入。组装后文件权限为 `0444`，仓库校验会拒绝缺失、可写或格式不合法的文件。

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
- OpenAI Realtime: `ws://localhost:50052/v1/realtime`
- 兼容 WebSocket: `ws://localhost:50052/v1/ws`
- capabilities: `http://localhost:50052/v1/capabilities`
- health: `http://localhost:8080/health`

#### Health 端点与平台探针

health 端口在进程启动时立即监听（先于模型加载），探针始终能拿到 HTTP 应答而非拒连。路由：

| 路径 | 语义 |
|------|------|
| `/health` | 未就绪（模型加载、warmup、gateway 绑定完成之前）返回 `503` + `{"status": "loading", ...}`，就绪后返回 `200` + 完整 stats。统一的 liveness/readiness/startup 探针指向这里。 |
| `/readyz` | 与 `/health` 相同的就绪门控，不受 probe-mode 开关影响。 |
| `/livez` | 进程活着即返回 `200`（纯 liveness）。 |
| `/metrics` | 恒 `200`；加载状态体现在 body 里，而不是抓取错误。 |

引擎循环线程在启动后死亡时，`/health` 和 `/readyz` 会回落到 `503`（`"status": "engine_loop_dead"`），平台 liveness 探针据此重启容器——这是无状态引擎期望的自愈行为。

同样四条路由也挂在 WebSocket 端口（默认 `50052`）上、共享同一份就绪状态，供只能探服务端口的平台使用。与 health 端口相比有两点差异：WebSocket 端口在模型加载完成后才 bind（加载窗口内探针只见拒连，startup 宽限必须覆盖冷启动）；且它由 gateway 事件循环应答，顺带验证了真实服务路径的响应性。平台允许选择时，优先探独立的 health 端口。

#### Kubernetes 单端口部署

只允许暴露一个容器端口的平台使用 `PORT=8000`、`HEALTH_PORT=0`。这会关闭独立
health listener，但不会关闭健康检查；`/health`、`/readyz`、`/livez` 与 Demo、SDK、
capabilities、Realtime WebSocket 都由同一个 `8000` 端口提供。内部 gRPC 可以继续监听
默认 `50051`，无需写进 Service 或 Ingress。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen3-tts
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qwen3-tts
  template:
    metadata:
      labels:
        app: qwen3-tts
    spec:
      containers:
        - name: engine
          image: qwen3-engine:<tag>
          env:
            - name: PORT
              value: "8000"
            - name: HEALTH_PORT
              value: "0"
            - name: DEMO_ENABLED
              value: "true"
          ports:
            - name: public
              containerPort: 8000
          startupProbe:
            httpGet:
              path: /health
              port: public
            periodSeconds: 10
            failureThreshold: 30
          readinessProbe:
            httpGet:
              path: /readyz
              port: public
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /livez
              port: public
            periodSeconds: 10
          resources:
            limits:
              nvidia.com/gpu: 1
          volumeMounts:
            - name: models
              mountPath: /models
              readOnly: true
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: <model-pvc>
---
apiVersion: v1
kind: Service
metadata:
  name: qwen3-tts
spec:
  selector:
    app: qwen3-tts
  ports:
    - name: public
      port: 8000
      targetPort: public
```

因此同一个 Service 地址直接提供：

- `http://<service>:8000/demo/`
- `ws://<service>:8000/v1/realtime`
- `http://<service>:8000/sdk/`
- `http://<service>:8000/health`

#### 推荐：自定义域名、Demo、WebSocket 与 SDK 共用 HTTPS

把自定义域名解析到 Ingress/Gateway，并由受信 CA 证书在边缘终止 TLS。容器和 Service
继续只提供 HTTP/WS `8000`，无需把证书放进模型容器。下面以 ingress-nginx 和
cert-manager 为例；`ClusterIssuer` 名称需替换为集群实际配置：

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: qwen3-tts
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
spec:
  ingressClassName: nginx
  tls:
    - hosts: [tts.example.com]
      secretName: qwen3-tts-tls
  rules:
    - host: tts.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: qwen3-tts
                port:
                  number: 8000
```

Ingress 必须支持 WebSocket upgrade；主流 Kubernetes Ingress Controller 会为同一条 HTTP
路由处理 upgrade，上面的长读写超时用于避免长连接被过早关闭。此时只需要一个公网
`443` 入口：

- Demo：`https://tts.example.com/demo/`
- Python SDK：`TTSClient.connect("https://tts.example.com")`
- Realtime：`wss://tts.example.com/v1/realtime`
- SDK 下载：`https://tts.example.com/sdk/`

公有 CA 证书会被浏览器和 Python 默认信任，因此不需要 `tls_verify=False`、证书路径或
额外环境变量。不要再为 Demo 建立第二个 Service 或端口。若同时设置
`ENGINE_SERVER_WEBSOCKET_PORT` / `ENGINE_SERVER_HEALTH_PORT`，这些完整变量优先于
`PORT` / `HEALTH_PORT`。

开发机内网不需要麦克风或输出设备选择时，无需复制这套 TLS：公网仍经 Ingress 使用
`https://` / `wss://`，内网可同时直连同一个 Service 或容器的
`http://<ddns-host>:8000/demo/` 与 `ws://<ddns-host>:8000/v1/realtime`。播放器会在
HTTP 页面自动降级，不影响系统默认扬声器播放；Python SDK 可直接连接
`TTSClient.connect("http://<ddns-host>:8000")`。无证书服务不能写成 `https://`。

平台探针检查清单：

- startup 宽限必须覆盖冷启动（TRT 反序列化 + warmup 约 20–40 秒，取决于 GPU 和 batch
  档位；实测一次并留余量，例如 `period 10s × failureThreshold 30`）。宽限不足会导致
  容器在加载中途被杀、永远起不来。
- 探针地址用 `127.0.0.1:8080`，不要用 `localhost`——服务只绑定 IPv4 `0.0.0.0`。
- 平台 liveness 宽限无法配置到覆盖加载时长时，设置 `ENGINE_SERVER_HEALTH_PROBE_MODE=alive`
  （或 `engine.yaml` 里 `server.health_probe_mode: alive`）：`/health` 在端口起来后即返回
  `200`，代价是该路径失去就绪门控（`/readyz` 仍保留）。短别名
  `ENGINE_HEALTH_PROBE_MODE` 仅在 compose / engine-docker 部署下有效（由入口脚本映射）。
- Triton 原生 readiness 探针指向 `http://<host>:8000/v2/health/ready`；公共 Realtime
  服务路径探针指向 sidecar 的 `http://<host>:50053/health`。sidecar 和
  `tts_orchestrator` 任一未就绪时，该路径都返回 `503`。

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

Compose 只发布 gRPC `50051` 与公共 HTTP/WebSocket 网关 `50052`。用于冷启动
探测的 `8080` health listener 保留在容器内部，供 Docker healthcheck 使用，
不会绑定宿主机端口。对外的 `/health`、`/demo/`、`/sdk/`、`/v1/realtime`
和 `/v1/ws` 全部由 `50052` 提供。

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
- OpenAI Realtime sidecar：`ws://localhost:50053/v1/realtime`
- Realtime capabilities / health：`http://localhost:50053/v1/capabilities` 和
  `http://localhost:50053/health`

compose wrapper 会同时启动 Triton 和 Realtime sidecar；用 `--realtime-port <N>`
修改 sidecar 的宿主机端口。计费 usage 始终在 `response.done.response.usage` 返回；默认
部署还会把 completed、cancelled、failed response 追加到
`workspace/realtime_usage/realtime_usage.jsonl`。

### 容器内日志

Engine Docker 和 Triton 会在容器内部保留轮转日志，不依赖 Docker logging
driver：

- Engine Docker：`/var/log/qwen3tts/engine.log`
- Triton：`/var/log/qwen3tts/triton.log`
- Triton Realtime sidecar：`/var/log/qwen3tts/realtime-gateway.log`

默认每个日志文件达到 50 MiB 时轮转，并保留 10 份备份。可通过以下容器环境变量
调整策略：

| 变量 | 默认值 | 含义 |
|------|--------|------|
| `QWEN_LOG_DIR` | `/var/log/qwen3tts` | 日志文件目录。 |
| `QWEN_LOG_MAX_BYTES` | `52428800` | 单个日志文件的最大字节数。 |
| `QWEN_LOG_BACKUP_COUNT` | `10` | 保留的轮转备份文件数量。 |
| `QWEN_LOG_STDOUT` | `1` | 同时将合并后的服务输出镜像到容器 stdout；设为 `0` 可关闭。 |

stdout 镜像采用尽力而为策略，避免 logging driver 或 attach 客户端阻塞推理；需要
完整历史时应以以上日志文件为准。

进入容器 shell 后，可用以下命令查看或打包日志：

```bash
ls -lh "${QWEN_LOG_DIR:-/var/log/qwen3tts}"
tail -n 1000 "${QWEN_LOG_DIR:-/var/log/qwen3tts}/engine.log"
tail -n 1000 "${QWEN_LOG_DIR:-/var/log/qwen3tts}/triton.log"
tar -C "${QWEN_LOG_DIR:-/var/log/qwen3tts}" \
  -czf /tmp/qwen3tts-logs.tar.gz .
```

实际只会生成所选 gateway 对应的日志文件。容器删除后，容器内日志也会丢失；
如果日志需要跨容器替换保留，请将 `QWEN_LOG_DIR` 挂载到持久化存储。

## 内置 Demo 与统一文档

正式镜像已经包含同版本的产品 Demo、Browser SDK、Python wheel 索引和精选 Markdown
文档，并默认启用在 Realtime 的同一个公共入口，不需要独立 Demo API 或 Node 进程。
这个入口默认是 HTTP/WS；即使挂载目录中已有本地证书，也不会自动切换协议。
CI/CD 只构建一次 Browser SDK npm tarball，并将同一份产物内置到
`/demo/downloads/`；SDK 页面会生成指向当前实例的 `npm install "https://...tgz"`
命令，调用方不需要拉取源码仓库。GitLab tag 流水线还会把同一 tarball 发布到项目 npm
Registry，作为第二种安装渠道。
启动时设置 `DEMO_ENABLED=false` 可将其关闭：

```bash
bash scripts/bash/compose.sh up --build \
  --gateway engine --variant custom-1.7b
# 打开 http://localhost:50052/demo/
```

#### 开发机直接启用 HTTPS/WSS

与 FunASR Nano 一样，没有 Ingress 的独立开发机可以在公共 WebSocket 端口直接终止
TLS。证书与私钥只读挂载到容器，必须同时配置：

```bash
SAN_EXTRA_DNS=demo.example.test ./tools/generate_demo_local_cert.sh
TLS_AUTO_ENABLE=true \
  bash scripts/bash/compose.sh up --build --gateway engine --variant custom-1.7b
```

使用 CA 签发的证书时，显式挂载证书目录：

```bash
TLS_HOST_DIR=/host/path/to/certificate \
TLS_CERT_FILE=/app/tls/fullchain.pem \
TLS_KEY_FILE=/app/tls/privkey.pem \
bash scripts/bash/compose.sh up --build --gateway engine --variant custom-1.7b
```

此时同一个端口提供 `https://<host>:50052/demo/`、
`wss://<host>:50052/v1/realtime`、`https://<host>:50052/sdk/` 和
`https://<host>:50052/health`。设置 `TLS_AUTO_ENABLE=true` 后，入口脚本才会发现
`/app/tls/cert.local.pem` 与 `/app/tls/key.local.pem`，便于使用已经由测试浏览器显式
信任的开发证书。自动发现默认关闭，因此仅仅挂载证书目录不会把 `http://` / `ws://` 暗中
切换成 `https://` / `wss://`。证书必须覆盖实际访问域名；缺文件、只配置一项或证书
与私钥不匹配时，服务会在加载 GPU 模型前拒绝启动。

Python SDK 调试同一个自签名入口时，优先显式信任生成的证书：

```python
client = TTSClient.connect(
    "wss://localhost:50052/v1/realtime",
    tls_verify="workspace/tls/cert.local.pem",
)
```

临时验证 TLS 链路时可设 `tls_verify=False`，但不能用于生产。浏览器手动放行证书不会改变
Python 的信任库；Browser SDK 也不能从 JavaScript 关闭浏览器的证书校验。若测试目标
并非 TLS，本地应直接使用默认的 `http://` / `ws://`。

生产 Kubernetes 通常不设置 `TLS_CERT_FILE` / `TLS_KEY_FILE`，由 Ingress 或 Gateway
终止可信 TLS，再转发到 Pod 的 HTTP 端口。两种模式都只使用一个公共服务端口。

Triton 部署把 `--gateway engine` 改成 `--gateway triton`，然后打开
`http://localhost:50053/demo/`。门户全部使用相对 URL，因此服务部署在
`/infer/<instance>` 下时，Demo 资源、`/sdk/`、`/v1/capabilities` 和
`/v1/realtime` 都会保留该前缀。不应公开门户时设置 `DEMO_ENABLED=false`，
此时 `/demo/` 返回 404。

`demo_api` 仅作为详细 trace 的可选工程实验后端；仓库不再保留第二套 WebUI。LLM PK、
并发和 trace 均从同一个 `/demo/#/lab` 门户进入，仅在确有需要时显式启用 Compose 的
`demo` profile，并通过 `DEMO_LAB_URL` 公布后端地址。

### 公网网关安全边界

内置门户不实现第二套登录，也不在浏览器中输入或持久化长期 API Key。公网部署必须由
同一个反向代理同时保护 `/demo`、`/sdk` 与 `/v1/*`，并满足以下条件：

- 转发时保留完整实例路径前缀和 WebSocket upgrade；
- 对 WebSocket `Origin` 使用明确同源/允许列表校验，拒绝任意站点跨源调用；
- 在升级连接前完成用户/租户认证，并对租户实施并发、请求速率和用量配额；
- 限制文本、reference audio 和 WebSocket 消息大小；reference 上限不得高于
  `/v1/capabilities` 公布值；
- 设置握手、空闲、单次 response 和整条连接超时，并限制最大连接数；
- 不缓存 `config.json` 或包含租户信息的响应，不在访问日志记录文本、reference 或凭据。

若 `DEMO_LAB_URL` 指向跨源 `demo_api`，应把 `QWEN_DEMO_CORS_ORIGIN` 收紧为门户的
精确 Origin；生产环境不要使用默认 `*`。

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

### 内置门户不显示“实验”入口

门户只在 `lab_available=true` 且 `demo_api /healthz` 可达时显示入口。检查：

- Triton gRPC 端口是否是 `localhost:8001`。
- demo API 的 `QWEN_DEMO_TRITON_GRPC` 是否正确。
- runtime 是否设置了浏览器可访问的 `DEMO_LAB_URL`。
