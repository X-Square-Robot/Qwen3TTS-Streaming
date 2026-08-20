[English](client_sdk.md) | **中文**

# Python Client SDK

## 目标

这个 SDK 面向外部调用方，提供统一、轻量、可直接 `pip install` 的 Python 客户端。

支持的服务入口：

- `openai-realtime`（主协议）
- `engine-websocket`
- `engine-grpc`
- `triton-grpc`
- `triton-http`

默认行为是 `transport="auto"`：客户端优先探测 Realtime，仅在主协议不可用时绑定旧
adaptor。

## 目录与发布

SDK 作为独立子项目放在仓库的 [`client/`](../../client) 目录下：

- 打包配置：[`client/pyproject.toml`](../../client/pyproject.toml)
- 源码入口：[`client/src/qwen3tts`](../../client/src/qwen3tts)
- 共享协议层：[`client/src/qwen3tts_protocol`](../../client/src/qwen3tts_protocol)
- SDK 单测：[`client/tests`](../../client/tests)

这样做的目的，是避免把服务端重量依赖和部署逻辑一起打进客户端 wheel。

## 安装

### 版本兼容

引擎与 SDK 仍从同一个 git tag 发布：引擎镜像构建时把 tag 烤进去
（`git describe`），wheel 版本号也由同一 tag 推导（hatch-vcs）。引擎把发布版本
登记在**版本化的 capabilities** 里（字段 `engine_version`）；`/health` 是纯存活
探针、不带版本。查询运行中引擎的版本：

```bash
curl http://<engine-host>:<ws-port>/v1/capabilities
# → {"loaded_model_type": "...", "engine_version": "v0.1.0", "protocol_version": "...", ...}
```

推荐安装该版本的 SDK，以便完整复现发布环境。`connect()` 读取服务端
capabilities：协议族或协议大版本不兼容时抛出
`ProtocolVersionMismatchError`；同一大版本内的协议修订兼容。
`engine_version` 与 SDK 发布版本不同时只产生 `RuntimeWarning`，不会阻止连接，
具体可选功能以 capabilities 为准。需要刻意绕过协议检查时，设
`QWEN3TTS_SKIP_PROTOCOL_CHECK=1` 可将错误降级为警告；或给 `connect()` 传
`verify=False` 跳过连接时的 capabilities 校验。

### 通道一 —— GitHub/GitLab Release 与 GitLab Package Registry

每个版本 tag 都会在两个代码托管平台生成 Release wheel。权威、可直接复制的安装命令
由已部署实例的 `/demo/#/sdk` 动态生成；它使用精确的相对 `/sdk/` 资源地址，因此能
保留反向代理前缀。也可以从对应的
[GitHub Release](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/releases)
或 GitLab Release 的 `client-sdk` 链接选择 wheel。不要把旧发布的文件名复制进长期
维护的文档。

```bash
# 打开 https://<public-service-base>/demo/#/sdk，或列出同一批产物：
curl https://<public-service-base>/sdk/
pip install "https://<public-service-base>/sdk/<wheel-filename>"
```

项目的 GitLab PyPI Registry 也提供同一个文件：

```bash
pip install \
  --index-url "https://<gitlab-host>/api/v4/projects/<project-id>/packages/pypi/simple" \
  "qwen3-tts-client[all]==0.1.0"
```

私有项目建议使用 PyPI Registry，并把个人 Token 或只读 Deploy Token 配在
`.netrc`；不要把凭据写入会提交的 requirements 文件：

```text
machine <gitlab-host>
login <deploy-token-username>
password <deploy-token>
```

私有 Release 直链则需要按 GitLab 文档通过查询参数或 HTTP Header 提供个人
Access Token，因此通常先下载文件再从本地安装更简单。如果 GitLab 管理员关闭了
PyPI package forwarding，还需配置可信的依赖索引或预装 wheel 的第三方依赖。

可选 extras 为 `[grpc]`、`[triton]`、`[audio]` 和 `[all]`。SDK 本体只有一个
通用 wheel；`pip` 仍会从配置的包索引解析其声明的第三方依赖。

### 通道二 —— 从运行中引擎获取同一 wheel

每个正式运行时镜像都嵌入 tag 流水线已经发布的 wheel，并在公共服务的
`GET /sdk/` 提供：

```bash
curl https://<public-service-base>/sdk/      # 查看可用 wheel
pip install "https://<public-service-base>/sdk/<wheel-filename>"
```

### 发版不变量

向某个平台推送（或镜像同步）`vX.Y.Z`、`vX.Y.ZaN`、`vX.Y.ZbN` 或
`vX.Y.ZrcN` tag 后，该平台的 `.gitlab-ci.yml` 或
`.github/workflows/release.yml` 会独立执行同一套 build-once 约束：

1. 只检出主仓库（GitLab 为 `GIT_SUBMODULE_STRATEGY=none`，GitHub 为
   `submodules: false`）。
2. 用 `release_client_wheel.sh` 构建且仅构建一个 wheel，并做安装烟测。
3. GitLab 把 wheel 发布到 PyPI Package Registry；GitHub 把 wheel 上传到草稿
   Release。这个持久对象成为该流水线后续步骤的唯一标准输入。
4. 两边的镜像 job 都从各自标准发布位置下载 wheel、校验 SHA256，再把完全相同的
   字节放入 `/app/sdk/`，分别推送到 `cr.x2robot.cn/audio/qwen3tt-streaming`
   与 GHCR。
5. 镜像成功后，GitLab 幂等地创建或更新指向 Registry 对象的 Release 链接，
   GitHub 则公开已验证的草稿 Release；两者都不链接会过期的 job artifact。

引擎镜像 job 需要能运行 Docker 且有足够磁盘的 runner（NVIDIA PyTorch 运行时镜像
及构建缓存建议至少预留 50 GB）。该基础镜像已经包含版本匹配的 CUDA、PyTorch 和
TensorRT，本项目只在其上安装应用层 Python 依赖。GitLab 的 Docker-in-Docker runner 必须开启
privileged；GitHub 默认把该 job 派给 `self-hosted`，需要时可用仓库变量
`RELEASE_IMAGE_RUNNER` 指定容量足够的 runner label；自托管 runner 需要提供
Docker 与 GitHub CLI（`gh`）。若所选 NGC 基础镜像要求认证，还需配置受保护的
`NGC_API_KEY` secret。GitLab 镜像 job 声明了 3 小时超时，Runner 自身配置的
maximum timeout 也必须不小于 3 小时。GitLab 还必须配置 masked 的
`X2ROBOT_REGISTRY_USER` 和
`X2ROBOT_REGISTRY_PASSWORD` CI/CD 变量；若变量设为 protected，触发发布的
`v*` tag 也必须是 protected。GitLab 发布镜像沿用
`cr.x2robot.cn/audio/qwen3tt-streaming:trt25.10_580_cu13_<git-tag>` 命名，并使用
与其一致的 NVIDIA PyTorch 25.10 运行时（CUDA 13.0、TensorRT 10.13、Driver 580
通道）。应保护 `v*` tag 命名空间与发布
environment，确保只有发布维护者能触发带发布凭据的 job。GHCR package 默认私有；
若正式镜像要求匿名拉取，需要显式改为 public。

两套 CI 对国内 runner 默认启用可覆盖的下载入口：普通 Python 包使用 BFSU，
GitHub 单元测试的 CPU-only PyTorch 使用南京大学镜像；GitLab 的 Debian/Alpine
软件包以及 job 镜像分别使用 BFSU 与 DaoCloud，GitHub/GitLab 的引擎镜像 job
则默认从 DaoCloud 的 `nvcr.io` 代理拉取 NVIDIA PyTorch 基础镜像。可在仓库变量
（GitHub）或 CI/CD 变量（GitLab）中覆盖以下同名值：

| 变量 | 用途 |
| --- | --- |
| `PIP_INDEX_URL` | wheel 构建、烟测及镜像内普通 Python 依赖 |
| `ENGINE_BASE_IMAGE` | 已包含匹配 CUDA/PyTorch/TensorRT 的 NVIDIA PyTorch 基础镜像；优先指向公司 Harbor/ACR 中按 digest 同步的副本 |
| `TRITON_RUNTIME_BASE_TAG` | release 镜像 job 选用的、已预发布且不可变的 Triton Python 依赖基座 |
| `X2ROBOT_REGISTRY`、`X2ROBOT_IMAGE`、`X2ROBOT_IMAGE_TAG_PREFIX` | GitLab 引擎镜像的目标仓库及兼容性 tag 通道 |
| `X2ROBOT_REGISTRY_USER`、`X2ROBOT_REGISTRY_PASSWORD` | GitLab 推送引擎镜像所需的 masked CI/CD 凭据 |
| `PYTORCH_CPU_INDEX` | GitHub 单元测试使用的 CPU-only PyTorch 索引 |
| `RUNNER_*_IMAGE`、`DEBIAN_*_MIRROR`、`ALPINE_MIRROR` | GitLab job/service 镜像与系统包源 |

公共代理适合先恢复流水线；稳定发版更建议把 NGC 与 job 镜像预同步到内网 Registry，
再覆盖 `ENGINE_BASE_IMAGE` 和各 `RUNNER_*_IMAGE`。这些镜像不会代理 GitHub
Actions、Release API、GHCR 推送或 GitLab 发布流量，runner 仍需能访问对应发布平台。
两套发布流水线都会把 Docker inline cache 发布到可变的 `buildcache` 镜像标签（GitLab
按运行时前缀隔离）；不可变 release tag 仍是部署产物。首次构建仍需拉取较大的 NGC
基础镜像。Triton 的约束更严格：release job 必须拉取
`TRITON_RUNTIME_BASE_TAG` 指定的不可变基座，不能现场安装 Python/TensorRT 依赖。
创建 release tag 之前，先通过 GitLab 手动 `BUILD_TRITON_RUNTIME_BASE=1` 流水线或
GitHub 的 **Build Triton Runtime Base** workflow 构建一次基座。这样不稳定的
PyPI/NVIDIA 下载不在 tag 发布关键路径中；基座缺失会立即失败。

`client/dist/` 保持为被忽略的本地/CI 暂存目录；wheel 二进制不提交进 Git。
两个 tag CI 都不会调用会重新构建 wheel 的本地 `compose.sh` 路径。

### 本地检出（开发）

```bash
cd client
pip install .          # extras 同理：pip install ".[grpc]"
```

依赖策略：

- `core`：`requests` + `websocket-client`（均为纯 Python）
- `grpc`：standalone gRPC 所需 runtime
- `triton`：Triton gRPC / HTTP 所需 runtime
- `audio`：`numpy`，用于 `synthesize_array`

websocket 传输层迁移到 `websocket-client` 后的两点说明：

- `/sdk/` wheel 通道只提供 client 本体 wheel；`pip install` 仍会从你的包索引
  拉取 `requests` / `websocket-client`。受限网络环境请预装依赖或配置本地镜像源。
- `ws://` / `wss://` 连接现在遵循标准的 `http_proxy` / `https_proxy` /
  `no_proxy` 环境变量（与 `requests` 行为一致）。此前 websocket 路径总是直连。
  如果部署机配置了企业代理，请确认 `no_proxy` 覆盖引擎主机，否则连接会被
  代理隧道转发（且很可能被代理拒绝）。

## 快速开始

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("localhost")
caps = client.get_capabilities()
print(caps.loaded_model_type)

result = client.synthesize_bytes(
    "你好，欢迎使用 Qwen3-TTS。",
    request=SynthesisConfig(task_type="custom_voice"),
)
print(result.audio_format)
print(len(result.audio_bytes))
print(result.details["usage"])
```

standalone 部署中，`localhost` 会解析到
`ws://localhost:50052/v1/realtime`；Triton compose 则通过 sidecar 暴露同一公共协议，
默认地址为 `ws://localhost:50053/v1/realtime`。最终选择可从
`client.resolved_transport` 读取。

## OpenAI Realtime 与旧协议迁移

需要确定性选择入口时可显式固定主 transport：

```python
client = TTSClient.connect(
    "wss://tts.example/v1/realtime",
    transport="openai-realtime",
    key="your-key",
)
```

一次性合成使用标准 `conversation.item.create` 和 `response.create`。增量
`open_stream()` 使用服务端必须明确声明的 `qwen.input_text_buffer.v1`
append/commit 扩展。传输仍是全双工：音频 delta 下行时可以继续 append 文本或发送
`response.cancel`。

终态 `response.done.response.usage` 在一次性调用中位于
`result.details["usage"]`，流式调用中位于 `session.usage`；
`session.response_id` 和 `session.response_status` 用于计费关联和终态判断。若客户端未
收到终态即断联，已配置的服务端 ledger 仍是权威计费来源。

迁移期继续保留四种旧 transport，并按每进程、每 transport 发出一次
`FutureWarning`；可用 `QWEN3TTS_SUPPRESS_LEGACY_TRANSPORT_WARNING=1` 临时静默。
当 Realtime 入口声明 `qwen.response_resume.v1` 时，SDK 会从精确的
delivery/sample 游标恢复，并只重放尚未 ACK 的文本。恢复与 native WebSocket 一样受
进程内 registry、grace 和缓存上限约束；未声明扩展的旧服务仍保持明确失败。

对于 `engine-websocket` 传输，`timeout` 表示连接建立后请求的接收空闲预算。
如果网络握手失败时需要更快释放调用线程，可以单独设置 `connect_timeout`；
不传时为保持向后兼容，它默认等于 `timeout`：

```python
client = TTSClient.connect(
    "ws://localhost:50052/v1/ws",
    timeout=120.0,
    connect_timeout=5.0,
)
```

### PaaS 鉴权与连接复用

如果 PaaS 网关通过 Bearer Header 鉴权，传入 `key` 即可。`key=None`（默认值）
表示 SDK 不添加鉴权 Header；engine 本身不校验该 Header：

```python
client = TTSClient.connect(
    "wss://tts.example/v1/ws",
    key="your-key",  # Authorization: Bearer your-key
    connect_timeout=5.0,
    max_connections=32,
    max_idle_connections=8,
    max_pending_acquires=256,
    acquire_timeout=30.0,
)

# 只覆盖本次 capabilities 请求的等待时间。
caps = client.get_capabilities(timeout=5.0)

# 在开始接收流量前，把连接池填充到 4 条可复用空闲连接。
idle_connections = client.prewarm(connections=4, timeout=5.0)
```

同一 `engine-websocket` 物理连接会串行承载多个逻辑 session；并发 session 会各自
租用连接池中的连接。默认设置如下：

`SessionStartRequest.session_id` 只是客户端关联 ID，不再作为 engine registry key。
WebSocket/gRPC 每次启动都会生成新的私有执行 ID，因此不同请求即使复用了同一个
外部 ID，也不会互相替换或取消。需要跨多次合成关联一场长期通话时，可将
`call_id` 放入 `TimingContext.extra`，无需长期占用某条连接。

- `reconnect_attempts=1`：建立新物理连接或发送首个 `start` 失败时重试一次；
- `active_stream_resume=True`：入口支持时，为活动 native WebSocket 或 OpenAI
  Realtime 流请求安全的进程内断线恢复；
- `stream_resume_attempts=2` / `stream_resume_timeout=10.0`：为活动流使用独立且
  有界的重试次数与恢复总时限；
- `stream_resume_ack_interval=8`：每 8 个输出 delivery 发送累计 ACK，终态立即
  确认；
- `max_connections=32`：严格限制建连中、已租用、探活中和空闲连接的总数；
- `max_idle_connections=8`：最多保留 8 条空闲连接；
- `max_pending_acquires=256`：限制 FIFO 租约等待队列长度，队列满时立即抛出
  `PoolSaturatedError`；
- `acquire_timeout=30.0`：等待连接超过 30 秒时抛出
  `PoolAcquireTimeoutError`；
- `idle_ttl=None` / `max_lifetime=None`：默认不按连接年龄淘汰，`0` 也表示
  关闭；活跃连接即使超龄，也只会在本次合成完成归还后退出；
- `keepalive_interval=15.0`：每 15 秒在空闲连接上探活，设为 `0` 可关闭；
- `keepalive_jitter=0.2`：将每次维护间隔随机分散 ±20%，避免集中探活。

`TTSClient.prewarm(connections, timeout=...)` 中的 `connections` 表示期望的空闲连接
总数，而不是本次新增数量。它会并行建立缺少的连接，将目标限制在
`max_idle_connections` 和 `max_connections` 以内，并在成功时返回实际空闲连接数。可选的 `timeout`
限制每次 capabilities 往返；连接握手仍使用 `connect_timeout`。
`TTSClient.get_capabilities(timeout=...)` 同样提供单次调用的超时覆盖，且不会改变
客户端配置的流式接收超时。

逻辑 session 以终态 `done`/`error` 事件为边界，而不是以 WebSocket 关闭为边界。
支持长连接的 gateway 只会在可安全复用的成功/取消 `done` 中标记
`websocket_connection_reusable=true`。engine error 会关闭并重连；旧 gateway 若没有
该标识，SDK 同样会丢弃 socket 并安全退化为重新建连。

后台保活（或关闭保活时的复用前探活）如果发现连接已被网关回收，SDK 会丢弃它，
并在下一个 session 到来时自动建立新连接。这些空闲连接探活使用 `connect_timeout`，
而不是可能更长的流式接收超时。

对于支持恢复的 gateway，活动流传输断开后，服务端会在有界 grace 内保留同一个
engine execution。SDK 在连接池内替换坏连接，不会再次调用 `open_stream()`；它只
补发高于服务端累计 ACK 的文本，并从最后确认的 delivery/sample 游标之后补收输出。
一个 JSON `audio_header` 与紧随其后的裸 PCM binary 组成一条可回放 delivery；只有
完整 binary 已进入本地消息队列后，SDK 才推进游标。因此它绝不会退化成“从头重合成
再猜测去重”。token/窗口过期、服务进程重启、重连被路由到另一副本或恢复预算耗尽
时，会只产生一个明确的 `error`。未声明该能力的旧 gateway 保持快速失败行为。
客户端使用完毕后应调用 `client.close()`，推荐使用上下文管理器统一释放连接池。

## 统一流式接口

```python
from qwen3tts import SessionStartRequest, SynthesisConfig, TTSClient

client = TTSClient.connect("ws://localhost:50052/v1/realtime")
session = client.open_stream(
    SessionStartRequest(
        session_id="demo-session",
        config=SynthesisConfig(task_type="custom_voice"),
    )
)

session.send_text("你好，")
session.send_text("这是统一流式协议。")
session.stop()  # 与兼容接口 session.end() 等价：停止输入并排空音频

for message in session.iter_messages():
    print(type(message).__name__, getattr(message, "meta", {}))

print(session.response_id, session.response_status, session.usage)
```

中继服务可用 `session.iter_messages(post_send_idle_timeout=30.0)` 限制已经接收
`end()` 或 `cancel()`、但迟迟不返回终态事件的远端流。发送侧关闭前的静默不计入
预算，每条新消息都会重置空闲计时。需要立即终止本地会话时使用
`session.close(reason="worker shutdown")`：它会尽力发送 cancel，在传输支持时强制
断开连接，并立即解除消息消费者的阻塞。异步会话提供对应的 `aclose()` 和
`aiter_messages(post_send_idle_timeout=...)`。

## 自动探测规则

显式 `transport=` 时不探测，直接走指定 adaptor。

`transport="auto"` 时：

- `ws://` / `wss://`：`/v1/realtime` 按 Realtime 探测；根 URL 先尝试
  `/v1/realtime` 再尝试 `/v1/ws`；显式 `/v1/ws` 保持旧协议。
- `http://` / `https://`：先探测 `GET /v1/capabilities`，优先采用声明的
  `openai-realtime-v1`，否则才走旧 standalone 或 Triton HTTP 探测。
- 裸 `host:port`：`50052` 和 `50053` 优先探测 Realtime，旧协议和 Triton 探测作为
  fallback。
- 裸 `host`：按 `50052`（standalone Realtime）、`50053`（Triton Realtime
  sidecar）、`50051`、`8001`、`8000` 的优先级扩展。

探测结果会暴露在：

- `client.resolved_transport`
- `client.probe_report`
- `client.detected_transport`

## Triton HTTP 的流式语义

Triton HTTP 本身不支持真正的 decoupled streaming infer。

因此 SDK 的统一策略是：

- `synthesize_bytes` / `synthesize_array`：直接走一次 HTTP infer
- `open_stream(...)`：本地缓存 `start/text/end`
- 调用 `end()` 后，才触发一次 HTTP infer
- 返回的 session 会显式标记 `degraded_to_oneshot=True`

这意味着：

- 它不会伪装成服务端边收边合成
- 但上层调用代码仍可复用同一套 session API

## 当前状态说明

当前版本已经完成这些结构目标：

- SDK 代码被收敛到 `client/` 子项目
- 提供统一同步 / 异步 façade
- 引入共享协议层 `qwen3tts_protocol`
- OpenAI Realtime 成为 auto-detect 首选 adaptor
- 四类旧 adaptor 继续可用并发出 deprecation warning
- 为 `triton-http` 提供显式的流式降级语义

当前仍建议把它视为 v1 alpha：

- SDK 到 gateway 的 Realtime 集成已经覆盖，无需模型/GPU
- 真实 Triton sidecar 和 GPU smoke test 仍应在部署 CI 中执行
- durable usage、鉴权、配额与 Realtime 断线恢复验收通过后，才公布旧协议删除版本
