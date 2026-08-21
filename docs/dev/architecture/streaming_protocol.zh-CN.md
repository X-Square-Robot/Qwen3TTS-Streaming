[English](streaming_protocol.md) | **中文**

# 独立协议重新设计

> 协议状态：本页记录的 `tts-session-v2alpha1` 是官方 SDK 使用的原生主协议，入口为
> `/v1/ws`。OpenAI Realtime `/v1/realtime` 是兼容入口。
> Realtime 事件、usage 计费和 Triton sidecar 边界见
> [OpenAI Realtime TTS 协议与 Triton 边界](openai_realtime.zh-CN.md)。

## 目标

围绕显式会话协议统一独立的 `gateway -> interface -> dispatcher -> backend`，以便：

- 传输时序不再决定合成语义
- 流式和离线共享相同的第二层分割逻辑
- 第一层分组预分割仅在声明的输入模式需要时启用
- 未来 `voice_design`、`custom_voice`、`voice_clone_xvec`、`voice_clone_icl`、`base` 和 `instruct` 支持可以在不更改会话协议的情况下添加

## 层级职责

### Gateway

Gateway 仅作为协议适配器。

- 接收 `start -> text* -> stop/end/cancel`
- 验证并规范化 `SessionConfig`
- 解析规范的 `OutputPolicy`、`VADPolicy` 和 `TimingContext`
- 通过共享输出流水线将输出音频转换为请求的输出格式
- 不从分块时序推断离线/流式行为
- 不拥有传输特定协议分支，仅保留线路兼容性适配

### Interface

Interface 是面向 gateway/triton 适配器的传输无关外部会话门面。

- 位于 `engine/interface/*`
- 拥有规范的请求/会话/事件契约：
  `SessionStartRequest`、`StreamTextChunk`、`SessionEndRequest`、
  `StreamCancelRequest`、`OutputPolicy`、`VADPolicy`、`TimingContext`、
  `StreamEvent`、`AudioFrame`
- 拥有会话生命周期和回调绑定
- 拥有文本/token 摄取和基于 `input_mode` 的路由
- 拥有有序音频发射回调用方
- 拥有时序规范化和输出元数据发射
- 不直接构造后端请求，仅委托给 dispatcher

### Dispatcher

Dispatcher 是面向后端的请求转换器。

- 将 `SegmentAction` 转换为后端 `EngineRequest`
- 发射 `NEW_SESSION`、`SESSION_TEXT_DONE` 和取消/控制事件
- 保留分割语义，如 `FLUSH_EOS` vs `FLUSH_NOP`

### Spliter

Spliter 拥有两层文本分割策略。

- 第一层：分组预分割
  - 用于 `LONG_SEGMENT` 和 `FULL_TEXT`
  - 不用于 `TOKEN` / `CLAUSE`
- 第二层：状态机驱动的分割
  - 始终是每段刷新和解码预算控制的权威

### Backend

Backend 拥有合成状态。

- Prefill/解码执行
- 缓存和暂停/恢复状态
- 文本暂时不可用时的真正流式语义

## 会话协议

### 规范契约

所有外部传输现在映射到单一的规范会话契约：

- `SessionStartRequest`
- `StreamTextChunk`
- `SessionEndRequest`
- `StreamCancelRequest`
- `OutputPolicy`
- `VADPolicy`
- `TimingContext`
- `StreamEvent`
- `AudioFrame`

接口契约版本为：

- `protocol_version = tts-session-v2alpha1`

兼容边界由协议族和大版本共同确定，即当前兼容键为 `tts-session/v2`。
`alpha1` 等后缀表示该大版本内的协议修订，必须保持向后兼容；变更协议族或
升级到 `v3` 才是不兼容变更。`engine_version` 是发布诊断信息，不参与线协议
兼容性判定。

当前传输保持向后兼容：

- gRPC 遗留 `init/text_complete`
- WebSocket `start/text/stop/end/cancel/oneshot`
- Triton 遗留 JSON 请求字段
- 远程 worker 顶层兼容字段

缺失的新字段始终默认为现有行为。

WebSocket 的一条物理连接同一时刻只承载一个活动 session，但可串行承载多个
session。`stop` 与兼容接口 `end` 都表示停止输入并排空输出，`cancel` 会丢弃排队
输出。session 以终态 `done` 或 `error` 事件为边界，客户端不能再等待 socket 关闭。
支持长连接的 gateway 只会在可安全复用的成功/取消 `done` 中添加
`event.meta.websocket_connection_reusable="true"`。engine error 会关闭连接；缺少该
标识时，新 SDK 会安全退化为重新建连。

WebSocket capabilities 还会声明 `stream_resume_v1`。客户端通过在
`start.resume` 中加入自己生成的高熵 token 显式启用；该 token 与公开
`session_id`、私有 engine execution ID 相互独立。启用后，每条输出事件携带累计
`delivery_seq`；每个裸 PCM binary 前都有一个使用同一序号并记录绝对样本边界的
`audio_header`。文本使用严格递增的 `seq_no` 和累计 `text_ack`：相同序号与内容的
重复输入幂等，内容冲突或序号缺口属于协议错误。`stop` 携带最终文本序号并同样幂等。

传输异常断开后，新 WebSocket 携 token 和最后完整接收的 delivery/sample 游标请求
恢复。进程内 registry 会 fence 旧 attachment、保留同一个 engine session，并只回放
更新的记录。回放日志受字节上限约束，并在服务端声明的 grace 后过期；超限或过期会
明确失败，不会静默丢输出或从头重启合成。该机制只恢复网络/代理断联，不能跨 engine
进程或 GPU 重启；多副本部署还需要 sticky routing 或按 token 的一致性路由，使重连
回到持有状态的实例。

这套可靠生命周期也供 Triton native Session adapter 与 Realtime
`qwen.response_resume.v1` 扩展复用。它是共享的内部合同，不代表可以抹平 endpoint
capability：native WebSocket 使用 JSON header 加 binary PCM，Realtime 使用 Base64
audio event 与 namespaced ACK/resume event。客户端必须读取 `protocols` capability map
中对应入口的声明来选择行为。

### 能力查询

在打开合成会话之前，客户端可以调用 `GetCapabilities`。

这返回独立引擎已加载的契约：

- `variant`
- `loaded_model_type`
- `declared_supported_task_types`
- 支持的输入模式 / 分组策略 / 音频格式
- 独立预处理用的 ref-audio 可用性
- 详细的参考预处理可用性：
  `speaker_encoder_available`、`ref_codec_available`、`icl_available`、
  `ref_audio_max_duration_sec`、`ref_c2w_warm_state_available`、
  `ref_codec_reason`
- 接口契约元数据：
  `protocol_version`、
  `supported_websocket_features`、
  `stream_resume_grace_ms`、
  `stream_resume_max_buffer_bytes`、
  `supported_output_policy_features`、
  `supported_vad_strategies`、
  `supported_timing_fields`、
  `supported_progress_features`

其中 `supported_progress_features` 在支持时包含
`text_progress_anchor_v1`、`playback_progress_v1`、`qwen.text_progress.v1`。服务端保持旧的
`supported_websocket_features` 列表不变，
避免旧客户端把新增能力误判为未知必选功能。

当前规范能力特性标志为：

- `supported_output_policy_features`
  - `request_context`
  - `timing_context`
  - `vad_policy`
- `supported_vad_strategies`
  - `disabled`
  - `prefix_trim`
  - `tail_guard`
  - `hybrid`
- `supported_timing_fields`
  - 客户端提供的时间戳，如 `client_request_ts_ms`
  - 服务器规范化的时间戳，如 `server_first_audio_epoch_ms`
  - 派生延迟指标，如 `server_ttft_ms`

### 文本进度锚点与播放头

`text_progress` / `qwen.text_progress` 的 `meta` 在支持
`text_progress_anchor_v1` 时包含 `anchor_seq`、`output_sample_start/end`、
`output_sample_rate`、原文和规范化文本的 Unicode code-point 半开区间，以及
`text_input_final`、`alignment_final`。sample 坐标是经过 VAD、重采样和编码后的最终
线路 PCM；被 VAD 删除、guarded delivery 丢弃或 retry/abort 回收的音频不会生成锚点。
`progress_final` 只是 `alignment_final` 的兼容别名，不表示客户端已经播放到该位置。

SDK 将锚点交给本地 `PlaybackProgressTracker`。调用方在音频真正进入设备缓冲区和被
设备消费时分别更新 `buffered_through_sample` 与 `played_through_sample`；yield 给调用方
不等于已经播放。tracker 同时返回已确认游标和相邻锚点之间的插值游标，没有未来锚点时不
外推，只有 terminal 且播放头到达 `final_output_sample` 才报告 `playback_complete`。

支持播放反馈的 WebSocket 客户端可发送独立的 `playback_progress` 消息。它只用于校验和
遥测，不影响 guarded delivery；过期值幂等忽略，超前或非法顺序返回非终止错误。OpenAI
Realtime 本轮只在客户端本地维护 tracker，不发送该非标准客户端消息。

已加载的模型类型在引擎启动时选择。运行时请求不切换模型；它们只能确认客户端和服务器使用相同的模型契约。

在独立模式下，外部已加载模型类型和后端合成分支相关但不相同：

- `base` -> 内部 `voice_clone` 使用 x-vector 路径
- `icl` -> 内部 `voice_clone` 使用 ICL 路径
- `custom_voice` -> 内部 `custom_voice`
- `voice_design` -> 内部 `voice_design`

### Start

`StartRequest` 声明一个 `SessionConfig`。

重要字段：

- `task_type`
- `language`
- `speaker`
- `instruct`
- `ref_audio`
- `ref_text`
- `x_vector_only`
- `input_mode`
- `group_policy`
- `audio`
- `output_policy`
- `timing`
- `protocol_version`

规范 `OutputPolicy` 字段：

- `vad_policy`
- `chunk_ms`
- `packet_format`
- `emit_text_events`
- `config`

规范 `VADPolicy` 字段：

- `enabled`
- `strategy`
- `implementation`
- `config`

规范 `TimingContext` 字段：

- `request_id`
- `turn_id`
- `client_request_ts_ms`
- `client_text_ts_ms`
- `client_end_ts_ms`
- `extra`

`task_type` 不再是运行时模型选择器。独立引擎将请求绑定到 manifest 中已加载的模型类型。客户端可以省略 `task_type`，或发送相同值作为显式握手检查。

服务器还在合成前验证模型特定字段：

- `base`：首先解析参考；没有 `ref_text` 的显式 `ref_audio` 仍为 x-vector-only，显式 `ref_audio + ref_text` 进入 ICL，当没有显式参考时 `speaker` 是参考别名
- `icl`：首先解析完整参考；显式参考必须同时包含 `ref_audio` 和 `ref_text`，当没有显式参考时 `speaker` 是参考别名
- `custom_voice`：`speaker` 是内置自定义音色名称；拒绝 `ref_audio`、`ref_text` 和 `x_vector_only`
- `voice_design`：需要 `instruct`

没有为参考别名引入新的协议字段。`speaker` 的含义取决于已加载的模型契约：

- `custom_voice`：内置自定义音色名称，如 `Serena`
- `base` / `icl`：当 `ref_audio` / `ref_text` 缺失时的参考别名

`base` / `icl` 的参考解析顺序为：

1. 显式 `ref_audio + ref_text` 优先。如果也存在 `speaker`，它仅作为参考元数据保留，不用于查找。
2. 如果没有显式参考且设置了 `speaker`，服务器在 `engine.yaml` 的 `references.entries` 中进行不区分大小写的查找。
3. 如果 `ref_audio`、`ref_text` 和 `speaker` 都缺失，服务器使用默认参考。优先使用 `references.default`；否则使用遗留的 `ENGINE_DEFAULT_BASE_REF_AUDIO_PATH` / `ENGINE_DEFAULT_BASE_REF_TEXT` / `workspace/default_refs/base_ref.wav` 机制。
4. 部分参考对 `icl` 被拒绝。对于 `base`，仅 `ref_audio` 仍为 x-vector-only，而仅 `ref_text` 被拒绝。

可选参考库配置：

```yaml
references:
  default: default
  entries:
    default:
      audio_path: workspace/default_refs/base_ref.wav
      ref_text: 参考音频对应文本
      language: auto
    vivian:
      audio_path: workspace/default_refs/vivian.wav
      ref_text: 这是一段与 vivian 参考音频完全一致的文本。
      language: auto

reference_cache:
  enabled: true
  max_entries: 16
```

对于 `base` / `icl`，仅当请求语言为空或 `auto` 时才应用注册表 `language`；显式请求语言优先。显式 `ref_audio + ref_text` 即使同时存在 `speaker` 作为参考元数据，也不会从注册表加载语言。

ICL 参考预处理故意是单请求的，并在 TRT 引擎周围序列化。它不是批处理的，`spliter.max_concurrent_segments` 仅影响下游文本段 / EngineLoop 槽位并发。参考音频硬限制报告为 `ref_audio_max_duration_sec`；当前 TRT 构建对 `speech_tokenizer_codec_fused.engine` 默认为 8 秒。

对于 TRT 模式下的独立 ICL 预处理，运行时包必须包含 TensorRT 产物：

```text
runtime/speaker_encoder.engine
runtime/speech_tokenizer_codec_fused.engine
```

或等效的 plan 布局：

```text
runtime/speaker_encoder/model.plan
runtime/speech_tokenizer_codec_fused/model.plan
```

独立 ICL 路径故意不回退到 ONNX Runtime。如果 `speech_tokenizer_codec_fused.engine` / `model.plan` 缺失，请求将失败并返回 `speech_tokenizer_codec_fused_trt_missing`。

参考元数据通过 prefill 事件/日志暴露：

```text
ref_source
ref_id
ref_audio_sha256
ref_text_hash
icl_cache_hit / icl_cache_miss
ref_preprocess_runtime=trt
```

### Text

`TextChunk` 仅承载文本。其传输到达模式不得改变会话语义。

它也可以承载可选的时序元数据，如 `client_timestamp_ms`，这作为上下文记录，不改变合成行为。

### End

`EndRequest` 表示此会话不再有文本到达。

它不得用于推断会话是"离线"还是"流式"。

它可以承载可选的 `client_timestamp_ms` 用于时序分析。

## 输入模式

### TOKEN

- 客户端发送 token 级别的文本更新
- 无第一层分组预分割
- dispatcher 立即将 token 化文本转发到第二层分割

### CLAUSE

- 客户端发送子句级文本更新
- 无第一层分组预分割
- dispatcher 直接将子句文本转发到第二层分割

### LONG_SEGMENT

- 客户端发送长文本单元
- 每个长文本单元首先预分割为分组
- 每个结果分组然后送入第二层分割

### FULL_TEXT

- 显式离线模式
- 完整文本缓冲直到 `end`
- 然后在整个文本上运行第一层预分割

## 分组策略

### AUTO

- 当 `input_mode` 为 `LONG_SEGMENT` 或 `FULL_TEXT` 时使用第一层预分割

### NONE

- 即使对于长输入单元也禁用第一层预分割
- 仍使用第二层分割

## 音频输出契约

Gateway 接受 `AudioFormat` 请求，并通过共享的 `engine.interface.output.OutputPipeline` 将引擎输出从原生 `PCM_F32@24kHz` 转换为请求的线路格式。

当前实现支持：

- `PCM_F32`，单声道，`24000` 或 `16000`
- `PCM_S16LE`，单声道，`24000` 或 `16000`

输出流水线还负责：

- 分块索引
- 首块标记
- start/done 事件规范化
- 规范时序元数据
- 共享 `protocol_version` 和 `output_policy_json` / `timing_context_json`
- 传输无关的 VAD 策略暴露

当前时序契约：

- `timing_contract = server_monotonic_v1`

服务器是强时序指标的真相来源：

- `server_request_received_epoch_ms`
- `server_first_audio_epoch_ms`
- `server_done_epoch_ms`
- `server_ttft_ms`
- `server_total_latency_ms`

客户端时间戳仅为可选上下文：

- 尽可能记录并回显
- 稍后可能用于网络延迟估计
- 不被视为强一致性指标

## VAD 契约

此阶段不在规范接口中实现实际的输出门控。相反，它预留稳定的策略契约，以便未来实现可以在不更改传输语义的情况下插入。

支持的语义策略模式：

- `disabled`
- `prefix_trim`
  - 用于仅修剪前导静音
- `tail_guard`
  - 用于切断幻觉非语音尾部
- `hybrid`
  - 用于组合前导修剪和尾部保护

重要设计规则：

- `vad_policy` 描述*何时*应进行输出流门控
- 它不对检测*如何*实现进行硬编码

这使未来实现保持兼容：

- 基于能量的前导修剪
- mel/对数能量前导修剪
- TenVad 尾部保护
- 混合组合

在 v1 中：

- 默认为 `vad_policy.enabled=false`
- 独立引擎契约不执行实际门控
- 启用策略仅更改元数据/契约字段，除非后续实现显式消费它

## 传输映射

所有外部适配器现在应为规范接口之上的薄壳：

- gRPC
  - 将 proto 字段映射到规范会话契约
  - 保留遗留 `init/text_complete`
- WebSocket
  - 将 JSON 消息映射到规范会话契约
  - 为兼容性保留二进制音频帧
- Triton
  - 保留 `audio_chunk`、`event_type`、`event_json`、`is_final`
  - 在 `event_json.meta` 中发射规范元数据
- 远程 worker
  - 转发规范 `output_policy` / `timing_context`
  - 保持现有 query/turn 兼容字段

传输层不应重复：

- 音频转换逻辑
- 时序规范化逻辑
- 协议版本协商
- 未来 VAD 契约绑定

不支持的组合应显式失败，而非静默降级。

## 当前实现说明

在独立引擎中已实现：

- 显式 `SessionConfig` 贯穿 gateway、interface、dispatcher 和 backend
- 按 `input_mode` 进行接口路由
- `Spliter` 中的长段 `push_group_tokens()` 路径
- backend prefill 不再等待 `text_complete`（如果初始文本已存在）
- `FLUSH_EOS` / `FLUSH_NOP` 区分保留到后端请求
- backend 中的流式暂停/恢复语义，而非无条件 pad 注入
- 独立 `base` / `icl` 参考解析器、仅 TensorRT 参考预处理、进程内参考特征缓存和 ICL 参考前缀 KV cache

仍待实现以达到完整 transport 对等：

- 完整采样参数传递

OpenAI Realtime Triton sidecar 及其双向 streaming-gRPC backend 已实现；旧 SDK
transport 仅作为迁移期兼容路径继续保留。
