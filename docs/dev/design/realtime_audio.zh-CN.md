[English](realtime_audio.md) | **中文**

# G7：客户端实时音频流（RealtimeAudioStream）

> 分支：`refact`
> 编写日期：2026-06-16
> 状态：已完成
> 关联：[[REFACTOR_GOALS.md]] G5 中 `tests/save.py` 的处理方案

---

## 0. 背景

当前引擎以最快速度推帧（首帧 ~800ms，后续每 ~2s 一块），输出节奏不均匀。`tests/save.py` 实现了"非等时音频 → 等时音频"的适配逻辑，但仅作为测试脚本存在，未被集成到正式代码中。

在以下场景中，消费者需要等时（isochronous）音频流：

- **WebRTC 服务器**：用户到云走 WebRTC，WebRTC 服务器到引擎走自定义客户端-服务端协议。WebRTC 的 jitter buffer 需要稳定的音频输入节奏，否则会出现 underrun 断音。
- **本地实时播放**：客户端直接播放音频时，需要静音填充来掩盖生成空隙。
- **测试/基准**：验证实时场景下的 TTFT、断音率、缓冲区水位。

`save.py` 的核心价值不是"播放模拟"，而是**非等时源 → 等时消费的适配**。WebRTC 服务器就是一个等时消费者。

---

## 1. 架构决策

| 选项 | 结论 | 原因 |
|------|------|------|
| 集成到引擎（类似 VAD） | ❌ | 引擎应尽快出帧，`time.sleep()` 和队列阻塞违背引擎设计目标；引擎不知道播放状态 |
| 集成到客户端 SDK | ✅ | 客户端拥有播放上下文，与现有 `iter_messages()` 消费接口一致，可选且无侵入 |
| 仅作为测试工具 | ❌ | 测试只是消费场景之一，实时播放和 WebRTC 服务器同样需要 |
| Gateway 层可选包装 | ⚠️ 可选 | 如果服务端需要按实时节奏推流给浏览器，可在 gateway 做轻量 wrapper，但核心逻辑归客户端 |

**目标架构**：

```
用户浏览器/APP
      │ WebRTC (音频 RTP，自带 jitter buffer + 播放节奏)
      ▼
  WebRTC 服务器 (SFU/MCU)
      │ 自定义 WebSocket/gRPC 协议
      ▼
  TTS Client SDK（RealtimeAudioStream 可选启用）
      │
      ▼
  TTS 引擎（不变，继续最快速度出帧）
```

---

## 2. 具体变更

### 2.1 新增 `client/src/qwen3tts/realtime.py`

```python
@dataclass
class TimedAudio:
    """带时间信息的音频帧。"""
    data: bytes          # PCM 音频数据
    duration_s: float    # 帧时长（秒）
    is_silence: bool = False  # 是否为填充静音

class RealtimeAudioStream:
    """将非等时的 AudioChunk 流转换为等时音频流。

    用于给 WebRTC 服务器、本地播放器等需要实时节奏的消费者提供输入。
    默认不启用，用户按需创建。

    Args:
        session: BaseStreamSession，音频来源
        fill_silence: 是否在空隙处填充静音（默认 True）
        chunk_s: 等时输出粒度，默认 0.02（20ms，对齐 WebRTC Opus 帧长）
        sample_rate: 采样率（Hz），默认 24000
    """

    def __init__(
        self,
        session: BaseStreamSession,
        fill_silence: bool = True,
        chunk_s: float = 0.02,
        sample_rate: int = 24000,
    ): ...

    def __iter__(self) -> Iterator[TimedAudio]: ...
```

### 2.2 `chunk_s` 默认 0.02 而非 0.01

- WebRTC 音频标准帧长 20ms（Opus 默认帧长）
- 10ms 粒度会产生过多队列操作和静音帧切片
- 对齐到 20ms 可直接映射到 WebRTC 音频帧，减少切片和重打包

### 2.3 更新 `client/src/qwen3tts/__init__.py`

- 导出 `RealtimeAudioStream`、`TimedAudio`

### 2.4 删除 `tests/save.py`

- 核心逻辑已迁入 `realtime.py`
- 测试验证改用 `RealtimeAudioStream` 的单元测试

### 2.5 新增 `client/tests/test_realtime.py`

- 测试静音填充正确性
- 测试首帧延迟处理
- 测试帧迟到场景
- 测试哨兵值（None）终止
- 测试 `fill_silence=False` 模式（直出，不填充）

### 2.6 Gateway 层可选包装（后续，不在本次重构范围）

- 如果需要服务端按实时节奏推流，可在 `engine/gateway/` 新增 pacing wrapper
- 根据 session 配置决定是否对输出做节奏控制
- 引擎核心不变

---

## 3. 实施阶段

本目标属于 REFACTOR_GOALS.md Phase 1（协议层建立）的扩展，应在 Phase 1 完成后、Phase 3（tests/ 清理）之前实施：

1. **Phase 1 扩展**：在 `client/src/qwen3tts/` 新增 `realtime.py`
2. **Phase 3 前置**：删除 `tests/save.py`，改用 `RealtimeAudioStream`
3. **Phase 5 文档**：更新 `client/README.md`，补充 RealtimeAudioStream 用法

---

## 4. 验收标准

1. ✅ `from qwen3tts import RealtimeAudioStream, TimedAudio` 可用
2. ✅ `RealtimeAudioStream(session)` 产生等时音频流，空隙处自动填充静音
3. ✅ `fill_silence=False` 时行为与直接 `iter_messages()` 等价
4. ✅ `tests/save.py` 已删除
5. ✅ client 包单测全部通过
6. ✅ `client/README.md` 包含 RealtimeAudioStream 使用示例
