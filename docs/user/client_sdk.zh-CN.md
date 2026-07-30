[English](client_sdk.md) | **中文**

# Python Client SDK

## 目标

这个 SDK 面向外部调用方，提供统一、轻量、可直接 `pip install` 的 Python 客户端。

支持的服务入口：

- `engine-websocket`
- `engine-grpc`
- `triton-grpc`
- `triton-http`

默认行为是 `transport="auto"`，客户端会先做探测，再绑定到具体 adaptor。

## 目录与发布

SDK 作为独立子项目放在仓库的 [`client/`](../../client) 目录下：

- 打包配置：[`client/pyproject.toml`](../../client/pyproject.toml)
- 源码入口：[`client/src/qwen3tts`](../../client/src/qwen3tts)
- 共享协议层：[`client/src/qwen3tts_protocol`](../../client/src/qwen3tts_protocol)
- SDK 单测：[`client/tests`](../../client/tests)

这样做的目的，是避免把服务端重量依赖和部署逻辑一起打进客户端 wheel。

## 安装

### 版本配对

引擎与 SDK **从同一个 git tag 发布**：引擎镜像构建时把 tag 烤进去
（`git describe`），wheel 版本号也由同一 tag 推导（hatch-vcs）。引擎把发布版本
登记在**版本化的 capabilities** 里（字段 `engine_version`）；`/health` 是纯存活
探针、不带版本。查询运行中引擎的版本：

```bash
curl http://<engine-host>:<ws-port>/v1/capabilities
# → {"loaded_model_type": "...", "engine_version": "v0.1.0", "protocol_version": "...", ...}
```

按该版本安装 SDK。`connect()` 读取服务端 capabilities，配对错误时立即快速失败——
`ProtocolVersionMismatchError`（线协议代不匹配）或 `EngineVersionMismatchError`
（引擎/SDK 发布版本不匹配）。需要刻意跨版本实验时，设
`QWEN3TTS_SKIP_PROTOCOL_CHECK=1` 可把两者都降级为警告；或给 `connect()` 传
`verify=False` 彻底跳过连接时的 capabilities 校验。

### 通道一 —— 从 Git 安装（有仓库访问权限的开发者）

在 URL 中钉住引擎对应的 tag（包未发布 PyPI）：

```bash
pip install "qwen3-tts-client @ git+https://github.com/X-Square-Robot/Qwen3TTS-Streaming.git@v0.1.0#subdirectory=client"
```

走 SSH 时把 `https://github.com/` 换成 `ssh://git@github.com/`。可选
extras 写在方括号里（`[grpc]` / `[triton]` / `[audio]` / `[all]`）：

```bash
pip install "qwen3-tts-client[all] @ git+https://github.com/X-Square-Robot/Qwen3TTS-Streaming.git@v0.1.0#subdirectory=client"
```

### 通道二 —— 交付 wheel（无需仓库访问权限）

每个引擎部署都在 health 端口的 `GET /sdk/` 提供从自己源码构建的 wheel ——
从你连的引擎本体获取，配对不可能出错：

```bash
curl http://<engine-host>:<health-port>/sdk/      # 查看可用 wheel
pip install http://<engine-host>:<health-port>/sdk/qwen3_tts_client-0.1.0-py3-none-any.whl
```

独立交付的发版 wheel 在 tag 上构建：

```bash
git tag v0.2.0
bash scripts/bash/release_client_wheel.sh         # → client/dist/*.whl
```

脚本会拒绝脏工作区和未打 tag 的提交，因此交付出去的 wheel 版本号总能
精确对应它的源码。

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
```

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
- `active_stream_resume=True`：gateway 支持时，为活动 WebSocket 流请求安全的
  进程内断线恢复；
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

client = TTSClient.connect("ws://localhost:50052/v1/ws")
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

- `ws://` / `wss://`：直接判定为 `engine-websocket`
- `http://` / `https://`：先探测 standalone `GET /v1/capabilities`，失败后探测 Triton HTTP `/v2/health/live`、`/v2/health/ready`、`/v2/models/<model>/ready`
- 裸 `host:port`：优先按端口规则探测 standalone，再探测 Triton，再回落 HTTP
- 裸 `host`：自动扩展默认候选端口 `50052`、`50051`、`8001`、`8000`

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
- 提供 auto-detect 骨架和四类 adaptor 入口
- 为 `triton-http` 提供显式的流式降级语义

当前仍建议把它视为 v1 alpha：

- transport 适配已成型，但还需要继续补更完整的集成验证
- `engine-websocket` / `engine-grpc` 路径最接近现有服务端真实合同
- Triton 相关路径后续建议继续补真实环境 smoke test
