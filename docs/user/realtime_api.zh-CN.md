[English](realtime_api.md) | **中文**

# Realtime 接口与事件

本页描述 OpenAI Realtime **兼容入口**。Python 业务优先使用官方 SDK 的原生 `/v1/ws`；
Browser SDK、已有 OpenAI Realtime 客户端、其他语言兼容接入或协议排障才需要本页内容。

## 公共端点

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/v1/capabilities` | 查询当前实例支持的任务、音频格式和扩展 |
| WebSocket | `/v1/realtime` | OpenAI Realtime 兼容入口 |
| `GET` | `/demo/` | 在线体验、SDK 下载和同版本文档 |
| Hash 路由 | `/demo/#/lab` | 统一实验页：公共 Realtime 实验与可选深度工程面板 |

HTTP 站点使用 HTTPS 时，WebSocket 必须使用 WSS。反向代理存在路径前缀时，不要手工拼接
根路径；从 Demo 配置或 capabilities 提供的相对地址解析。

## 建立连接

连接 `/v1/realtime` 后，服务首先发送 `session.created`。客户端可发送 `session.update`
选择模型、音色、音频格式、VAD 和交付策略，并等待 `session.updated` 再创建 response。

一个物理连接可以顺序复用多个 response，但当前一次只允许一个活动 response。并发合成
使用连接池中的多条连接。

## 完整文本请求

```json
{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"你好，世界。"}]}}
{"type":"response.create"}
```

主要下行顺序：

```text
response.created
  -> response.output_item.added
  -> response.content_part.added
  -> response.output_audio.delta *
  -> response.output_audio.done
  -> response.content_part.done
  -> response.output_item.done
  -> response.done
```

`response.output_audio.delta.delta` 是 Base64 编码的单声道 PCM16。采样率以 session 配置和
capabilities 为准。不要假设一个 delta 对应固定时长。

## 增量文本请求

OpenAI Realtime 没有向已经开始的文本 item 追加内容的标准事件。本服务使用带 `qwen.`
命名空间的扩展，输出事件仍保持标准 Realtime 结构：

```json
{"type":"response.create"}
{"type":"qwen.input_text_buffer.append","sequence":1,"text":"你好，"}
{"type":"qwen.input_text_buffer.append","sequence":2,"text":"世界。"}
{"type":"qwen.input_text_buffer.commit"}
```

`sequence` 从 1 严格递增。相同序号与相同文本的重试是幂等的；序号缺口或同序号不同文本
会返回 `error`。客户端应处理 `qwen.input_text_buffer.ack`，并保留尚未确认的输入直到
response 进入终态。

## 音频游标与播放确认

音频事件携带连续的 `qwen_output_sample_start` 和 `qwen_output_sample_end`。客户端应拒绝
重叠或断裂的游标，避免重复播放。使用 guarded delivery 时，播放器持续发送：

```json
{"type":"qwen.playback.ack","response_id":"resp_...","played_through_sample":"24000","buffered_through_sample":"28800"}
```

“收到音频”“放入设备缓冲”和“扬声器已经消费”是三个不同状态。只有设备消费进度才应
推进 `played_through_sample`。

## 终态、Usage 与错误

每个 response 必须以 `response.done` 结束，其 `response.status` 为 `completed`、
`cancelled` 或 `failed`。调用方应：

1. 按 `response_id` 关联业务请求与日志。
2. 持续接收直到 `response.done`，不要把 WebSocket 关闭当作正常终态。
3. 从 `response.done.response.usage` 读取本次 token usage。
4. 对 `failed` 读取 `status_details.error.code` 和 `message`。
5. 用户取消时发送 `response.cancel`，并继续等待 cancelled 终态。

协议错误通常通过 `error` 事件返回。单个请求失败不一定意味着物理连接不可复用；SDK 已
根据终态和服务能力做出安全选择，自研客户端需要实现同等状态机。

## 断线恢复

只有 capabilities 声明 `qwen.response_resume.v1` 时才可恢复活动 response。恢复需要
服务端返回的 token、最后确认的 delivery sequence、已收到的音频 sample 游标以及未确认
文本序号。服务重启、token 过期、超出回放窗口或被路由到另一副本时必须明确失败，不能
静默重新合成并造成重复音频。
