[English](advanced_configuration.md) | **中文**

# 高级配置

先保持默认值完成接入，再根据业务目标调整这里的参数。可用选项始终以当前实例的
`/v1/capabilities` 为准；Demo 的“SDK”页会把体验页上的当前参数转换成可复制代码。

## 选择合成任务

| 目标 | `task_type` | 还需要提供 |
| --- | --- | --- |
| 使用预置音色 | `custom_voice` | `speaker` |
| 根据文字描述设计声音 | `voice_design` | `instruct` |
| 克隆参考声音 | `voice_clone` | 参考音频，部分模型还需要参考文本 |

不要仅因为 SDK 中存在某个枚举就假设服务已经支持它。先检查 capabilities 的 `tasks`、
`speakers`、`languages` 和 `reference` 字段。当前推荐路径是 `custom_voice`。

## 完整文本与增量文本

文本已经完整时，使用一次性接口：

```python
result = client.synthesize_bytes(
    "这是一段已经准备好的完整文本。",
    request=SynthesisConfig(task_type="custom_voice", speaker="serena"),
)
```

文本来自 LLM 或其他流式上游时，使用增量 session，让音频生成与后续文本到达重叠：

```python
from qwen3tts import AudioChunk, SessionStartRequest, SynthesisConfig

session = client.open_stream(SessionStartRequest(
    session_id="turn-42",
    config=SynthesisConfig(task_type="custom_voice", speaker="serena"),
))
session.send_text("你好，")
session.send_text("这是增量输入。")
session.end()

for message in session.iter_messages():
    if isinstance(message, AudioChunk):
        audio_sink.write(message.pcm_bytes)
```

`session.end()` 表示文本输入完成，不表示音频已经播放完成。继续消费消息直到终态。

## 流式 TN 与文本进度

增量 session 会先把 transport delta 交给主 streaming TN，再送入 tokenizer 和
`Spliter`。主 TN 维护 raw Unicode、仍可能改写的 mutable tail、单调
`TextCommit` 以及 raw → normalized/spoken 的 owner 映射。`99%`、日期、URL、型号和
中英混排等开放片段可能暂时等待后续字符；这不是音频暂停错误，而是为了避免把尚未确定
的读法不可逆地送入 TTS。

对于声明 `native_cursor.progress_available=true` 的 `custom-1.7b` 部署，文本进度来自
cursor-enabled TRT 图，并通过 `qwen.text_progress` 事件发布。其他部署会使用 EMA 或
关闭文本进度。应用应先读取 `/v1/capabilities`，不要根据模型名自行假设 native route。

进度事件中的 `raw_codepoint_end` 和 `normalized_codepoint_end` 是保守的整数
high-water 边界；`display_*_position` 只用于高亮等展示。`alignment_final=true` 表示
该 segment 的文本对齐事件完成，不表示声卡已经播放完成；播放完成仍以音频 sample 游标和
`qwen.playback.ack` 为准。

## 音频格式

推荐从单声道 24 kHz PCM16 开始，它适合网页播放、WAV 封装和大多数实时音频链路：

```python
from qwen3tts import AudioFormat, SynthesisConfig

config = SynthesisConfig(
    task_type="custom_voice",
    speaker="serena",
    audio=AudioFormat(encoding="pcm_s16le", sample_rate=24000, channels=1),
)
```

只有 capabilities 中列出的格式才可用。不要根据字节数反推格式，读取结果携带的
`audio_format`。

## 输出 VAD

输出 VAD 用于识别并裁掉开头或结尾的静音。它不是输入麦克风 VAD，也不会判断生成内容
是否正确。

```python
from qwen3tts import OutputPolicy, VADPolicy

config.output_policy = OutputPolicy(vad=VADPolicy(
    enabled=True,
    strategy="energy",
    chunk_ms=16,
    begin_threshold=0.30,
    begin_count=5,
    end_threshold=0.20,
    end_count=31,
    start_margin_ms=20,
))
```

- 开启 VAD 会等待“确认开始说话”，因此有效音频 TTFT 可能升高。
- `begin_threshold` 太高可能过滤整段正常语音；不同检测器的阈值尺度不能混用。
- `start_margin_ms` 保留说话开始前的一小段音频，过小可能切掉辅音起始。
- 如果业务本身能接受前导静音，先关闭 VAD，以获得最早的原始首包。

先在 Demo 中试听并观察 `Raw TTFT / VAD gate / Effective TTFT`，再把参数带入代码。

## 交付策略

`guarded` 是推荐默认值：它保留有限的未播放尾部，使服务在检测到异常生成时仍有机会
撤回坏尾。`firehose` 直接发送全部可用音频，延迟路径更简单，但客户端可能已经收到无法
撤回的异常尾部。

```python
from qwen3tts import OutputPolicy

config.output_policy = OutputPolicy(
    chunk_ms=0,
    emit_text_events=True,
    config={"delivery": "guarded", "delivery_window_ms": 160},
)
```

使用 `guarded` 且自己实现播放器时，应持续上报已缓冲和已播放 sample；Browser SDK 的
播放器已实现这条链路。

## 鉴权、超时与复用

```python
client = TTSClient.connect(
    "wss://tts.example.com/v1/ws",
    key="your-api-key",
    timeout=120.0,
    connect_timeout=5.0,
)
```

- `key` 作为 Bearer Token 交给部署网关；不要写进前端静态文件。
- 本地默认使用 `ws://`。自签名 WSS 可设置
  `tls_verify="/path/to/cert.local.pem"`；临时联调可用 `tls_verify=False`，禁止用于生产。
- `verify_protocol=False` 仅关闭协议兼容检查，不会改变 TLS 校验。
- 为进程复用一个 client，不要为每句话重新建立连接。
- 为连接、获取连接和活动请求分别设置符合业务 SLA 的超时。
- 失败时记录 `response_id`、终态错误码和服务端 timing，便于定位。

## 上线前验证

至少覆盖：空文本、长文本、中英混排、取消、断网恢复、不支持的 speaker、整段被 VAD
过滤、限流和服务端错误。对延迟敏感时分别观察服务端原始 TTFT、VAD 门控、网络传输和
播放器启动，不要只看一个端到端数字。
