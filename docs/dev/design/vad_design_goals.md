# TTS 输出 VAD 设计目标

> 分支：`refact`
> 编写日期：2026-06-16
> 状态：**已实现**（核心 VAD 处理器 + 协议层 + 网关集成 + 单元测试）
> 来源：三轮讨论的共识与决策记录

---

## 0. 问题背景

### 0.1 TTS 模型幻觉

Qwen3-TTS 模型在流式合成中概率性出现幻觉，表现形式包括：

| 幻觉类型 | 表现 | 处理归属 |
|---------|------|---------|
| 开头长静音 | 音频开头 300~500ms 近似静音 | VAD 裁剪 |
| 低能量随机噪声 | 模型输出低能量但人耳可辨的杂音 | VAD 裁剪 |
| 高能量随机噪声 | 模型输出有能量但非语音的噪声 | VAD 裁剪 |
| 重复 token 模式 | 模型陷入 1-2-3-1-2-3 循环 | Backend 检测 |
| 重复语音片段 | 模型重复输出上文的语音片段 | Backend 检测 |
| 无关语音内容 | 模型输出与输入文本无关的语音 | 模型侧解决 |

**VAD 的处理目标**：裁剪开头静音、拦截低/高能量随机噪声。
**Backend 的处理目标**：检测重复 token 模式并停止 flush。
**不在 VAD 目标内**：重复语音片段、无关语音内容。

### 0.2 开头静音的体验问题

合成速率 2x 实时的情况下，500ms 开头静音只需 250ms 就合成完，但用户需要等待 250ms 听静音才能听到有效语音。VAD 裁剪后用户不再听到这段静音。

**重要澄清**：VAD 不能减少首字延迟（静音仍需先合成出来 VAD 才能判断），但能改善用户体验——用户宁愿等待沉默也不愿听 500ms 静音/底噪。这在语音交互场景（语音助手）中有显著价值。

### 0.3 现有实现

| 实现 | 位置 | 能力 | 局限 |
|------|------|------|------|
| dBFS prefix trim | `workspace/qwen3_tts_remote.py` `_trim_prefix_silence_locked()` | 去除开头静音 | 只做 prefix trim，无 end 检测；无法拦截幻觉噪声 |
| TenVAD (ASR) | `workspace/ten_vad.py` | 完整的 ASR VAD | 为 ASR 设计的多段语音状态机，不适合 TTS 场景 |
| Backend pad silence | `engine/backend/engine_loop.py` `_is_pad_silence()` | 保护 backend 不超过最大 step | 生成侧停止解码，非输出侧门控 |

---

## 1. 设计目标

### G1：统一 VAD 协议

所有 VAD 模式共享同一套参数接口：

```
mode:           "disabled" | "energy" | "tenvad"
chunk_ms:       每帧时长（模式配置，energy=16, tenvad=16）
begin_threshold: 0.0~1.0 float（VAD 内部映射到各自量纲）
begin_count:    连续 N 帧高于 begin_threshold 触发 begin
end_threshold:  0.0~1.0 float
end_count:      连续 N 帧低于 end_threshold 触发 end
start_margin_ms: begin 触发后向前回溯的毫秒数
```

**量纲映射**（VAD 内部实现，对外统一 0~1）：

| 模式 | begin/end_threshold 0~1 映射 | 典型 begin | 典型 end |
|------|------------------------------|-----------|---------|
| energy | 线性映射到 dB 标度（如 0→-80dB, 1→0dB） | ~0.3 (-56dB) | ~0.2 (-64dB) |
| tenvad | 直接作为概率阈值 | ~0.6 | ~0.35 |

### G2：三种 VAD 模式

| 模式 | 实现 | 能力 | 典型场景 |
|------|------|------|---------|
| `disabled` | 直通，不做任何裁剪 | 完整体现模型效果 | 调试/基线 |
| `energy` | 预加重 + Hamming 窗 + 对数能量 + dB 标度门限 | 去除绝对静音；对清辅音开头更敏感 | 轻量级裁剪 |
| `tenvad` | TenVad ONNX 推理 + TTS 专用状态机 | 拦截静音 + 随机噪声；更好保留正常停顿 | 生产推荐 |

**dBFS 与对数能量融合**：统一为 `energy` 模式，内部使用对数能量（预加重+加窗+log），对外暴露 dB 标度门限。是否启用预加重通过内部参数控制。

### G3：流式计算

- 算到哪下发到哪
- VAD 以 16ms 一帧处理，TTS 以 ~80ms chunk 产出
- 100ms 待计算音频 → 6 帧 × 16ms = 96ms 可判定 → 下发到 96ms
- begin 引入延迟极低（begin_count × chunk_ms ≈ 80ms，小于 TTS chunk 大小）
- end 是连续累计，end 触发后停止下发

### G4：VAD 对客户端透明

- VAD 只做音频门控（下发/暂存/丢弃），不发 begin/end 事件给客户端
- 客户端收到的是一段连续 PCM 流，中间可能缺少被裁剪的停顿/噪声
- 相当于 VAD 帮用户跳过了噪音/无意义长静音的部分

### G5：滞回设计

- begin_threshold > end_threshold（如 0.6 > 0.35）
- begin_count 小（~5 帧 = 80ms），end_count 大（~31 帧 = 500ms）
- begin 门限高+数量少 → 快速确认语音开始
- end 门限低+数量多 → 保护正常停顿/呼吸不被误裁

### G6：end 后重新 begin

VAD 状态机支持多次 begin→end 循环：

```
SILENCE → (begin) → SPEECH → (end) → SILENCE → (begin) → SPEECH → ... → flush
```

场景示例：
```
[静音 500ms] [语音 2s] [幻觉噪声 0.5s] [静音 0.5s] [语音 1s] [静音 300ms]
→ VAD 裁剪后 →
[语音 2s] [幻觉噪声 0.5s] [语音 1s]  (静音被裁，噪声在 end_count 内会下发一部分)
```

---

## 2. 架构设计

### 2.1 VAD 在引擎中的位置

```
EngineLoop (GPU thread)
    ↓ EngineResult(AUDIO_CHUNK)
asyncio result_queue
    ↓
VAD (asyncio 端，per-session 有状态流式处理器)
    ↓ 门控后的音频
OutputPipeline
    ↓
Transport (gRPC / WebSocket)
```

**选择 asyncio 端的理由**：
- engine_loop 是 GPU 线程，不宜做 CPU 密集的 VAD 计算（TenVAD 是 ONNX 推理）
- asyncio 端音频已是 CPU 上的 bytes，VAD 不影响 GPU 线程
- asyncio 端已有 timing 逻辑，可与 OutputPipeline 度量集成
- OutputPipeline 是纯数据转换管道，不应变成有状态管道

### 2.2 VAD 状态机

为 TTS 场景重新设计，不复用 ASR 的多段语音状态机：

```
                    ┌─────────────────────────────┐
                    │                             │
                    ▼                             │
┌──────────┐  begin触发  ┌──────────┐  end触发  ┌──────────┐
│ SILENCE  │ ──────────→ │ SPEECH   │ ────────→ │ SILENCE  │
│ (不下发) │             │ (下发)   │           │ (不下发) │
└──────────┘             └──────────┘           └──────────┘
     ▲                                                 │
     └─────────────── begin触发 ───────────────────────┘

任何状态 + flush信号 → 下发所有 pending 音频 → 重置
```

**与 ASR VAD 的关键区别**：
- 无 pre_roll 回溯保留（TTS 开头静音就是应该丢掉的）
- 无 segment 过短丢弃逻辑
- 无多段语音的 start/end 事件发布
- flush 信号统一处理（SESSION_DONE / ERROR / CANCEL）

### 2.3 缓冲区设计

VAD 需要三个缓冲区：

#### (a) 输入帧对齐缓冲区

TTS 产出的 AUDIO_CHUNK 大小不一定是 16ms 帧的整数倍（24kHz × 16ms = 384 samples）。
需要暂存不满一帧的残余样本，与下一个 chunk 拼接。

```
chunk 到达 → 拼接到 input_buffer → 按帧切分 → 逐帧送入 VAD → 残余留回 input_buffer
```

#### (b) Start margin 保留缓冲区

begin 触发后需要向前回溯 start_margin_ms（~20ms）的音频。在 SILENCE 状态下，
最近 start_margin_ms 的音频需暂存不下发也不丢弃，等 begin 确认后决定。

```
SILENCE 状态：每帧音频 → 暂存到 margin_buffer（环形，容量 ≥ start_margin_ms）
begin 触发：从 margin_buffer 取出 start_margin_ms 音频 + 当前帧 → 一起下发
```

**约束**：margin_buffer 容量 < TTS chunk 大小（80ms），引入延迟微乎其微。

#### (c) 待下发缓冲区

SPEECH 状态下，每帧判定为语音的音频暂存于此，批量下发以减少系统调用。
end 触发时丢弃缓冲区中未下发的音频。

### 2.4 能量 VAD 实现

```python
def compute_energy_score(frame_int16: np.ndarray, *, preemphasis: float = 0.97) -> float:
    """预加重 → Hamming 加窗 → 对数能量 → dB 标度 → 归一化到 0~1"""
    # 1. 预加重: y[n] = x[n] - a * x[n-1]
    # 2. Hamming 加窗
    # 3. energy = sum(y^2)
    # 4. dB = 10 * log10(energy / (N * 32768^2) + eps)  -- 绝对参考，与帧大小无关
    # 5. 归一化: score = (dB + 80) / 80  -- -80dB→0, 0dB→1
    ...
```

**dB 标度的优势**：
- 门限值与帧大小无关（用户设置一次即可）
- 有绝对参考点（0 dB = 满幅）
- 预加重的影响只是让清辅音在 dB 标度下"看起来更响"

### 2.5 TenVAD 实现

只复用 TenVAD 的推理核心（`TenVad.process()` → probability + flags），
围绕 TTS 场景重写状态机逻辑。

**TenVAD 推理参数**：
- hop_size = 256（16ms @ 16kHz）
- threshold = 0.5（模型内部阈值）
- RTF ≈ 0.015（极低开销）

**TenVAD 输入**：16kHz int16 PCM。VAD 内部负责从 24kHz 原始 PCM 降采样到 16kHz。

**TenVAD 相对能量模式的核心优势**：能更好区分"低能量正常停顿/呼吸"和"低能量噪声"，
因为停顿/呼吸虽然能量低但 TenVAD 可能给较高概率，不容易误触 end。

---

## 3. 协议与配置

### 3.1 VAD 配置

```python
@dataclass
class VADConfig:
    enabled: bool = False
    mode: str = "disabled"      # "disabled" | "energy" | "tenvad"
    chunk_ms: int = 16          # 每帧时长
    begin_threshold: float = 0.6  # 0~1
    begin_count: int = 5        # 连续 N 帧高于 begin_threshold 触发 begin
    end_threshold: float = 0.35  # 0~1
    end_count: int = 31         # 连续 N 帧低于 end_threshold 触发 end (~500ms)
    start_margin_ms: int = 20   # begin 后向前回溯
```

### 3.2 模式默认参数

| 参数 | energy 默认 | tenvad 默认 | 说明 |
|------|-----------|-----------|------|
| chunk_ms | 16 | 16 | 恰好一致，无实际关联 |
| begin_threshold | 0.3 | 0.6 | energy 映射到 ~-56dB |
| begin_count | 5 | 5 | ~80ms |
| end_threshold | 0.2 | 0.35 | energy 映射到 ~-64dB |
| end_count | 31 | 31 | ~500ms |
| start_margin_ms | 20 | 20 | 向前回溯 20ms |

### 3.3 可观测性

VAD 需要在 `done_meta` 中注入裁剪信息：

```python
meta["vad_mode"] = "energy"  # 或 "tenvad"
meta["vad_prefix_trimmed_ms"] = "520.000"   # 开头裁剪的静音时长
meta["vad_tail_trimmed_ms"] = "0.000"       # 尾部裁剪的时长
meta["vad_original_audio_ms"] = "3000.000"  # 原始音频总时长
meta["vad_effective_audio_ms"] = "2480.000" # 裁剪后有效音频时长
meta["vad_begin_count"] = "1"               # begin 触发次数
meta["vad_end_count"] = "0"                 # end 触发次数（不含 flush）
```

**时长口径**：
- 引擎侧 RTF = 原始音频时长 / 耗时（不变）
- 客户端侧 RTF = 有效音频时长 / 耗时
- 客户端的 `audio_duration` 应基于裁剪后的有效音频时长

---

## 4. 关键设计决策

### 4.1 Backend 与 VAD 的职责分工

| 层 | 职责 | 检测手段 | 动作 |
|----|------|---------|------|
| **Backend** (engine_loop) | 检测重复 token 模式 | token 序列模式匹配（1-2-3-1-2-3） | 停止 flush，发送 SEGMENT_END/ERROR |
| **Backend** (engine_loop) | 保护 KV 预算 | pad_silence 检测 + dynamic_silence_limit | 停止 decode |
| **VAD** (asyncio 端) | 裁剪开头静音 | begin 门限 + begin 数量 | 不下发静音帧 |
| **VAD** (asyncio 端) | 拦截幻觉噪声 | end 门限 + end 数量 | 停止下发，等重新 begin |
| **VAD** (asyncio 端) | 保留正常停顿 | end_count 足够大 / TenVAD 概率区分 | 不误裁停顿 |

**`SchedulerConfig.pad_silence_*` 的演进**：当前用于保护 backend 不超过最大 step。
本期演进为：backend 专注于检测重复 token 模式，pad_silence 检测保留作为安全兜底
（防止模型陷入重复输出静音的模式导致超过 max_seq_len）。

### 4.2 Flush 信号机制

当音频流终止时（无论正常还是异常），VAD 需要收到 flush 信号并立即下发所有 pending 音频：

| 信号来源 | 信号类型 | VAD 行为 |
|---------|---------|---------|
| SESSION_DONE | 正常结束 | 处理完所有 pending audio → flush → 重置 |
| SEGMENT_END | 段结束 | 同上 |
| ERROR | 异常终止 | 立即 flush all pending（不做 end 判断）→ 重置 |
| CANCEL_SESSION | 手动取消 | 丢弃所有 pending → 重置 |

**SESSION_DONE 时的残余帧处理**：
1. 处理 input buffer 中的所有完整帧
2. 残余帧不满一帧时，补零到一帧大小并处理
3. flush 所有待下发缓冲区中的音频（无论 VAD 当前状态）
4. 重置 VAD 状态

### 4.3 能量模式 vs TenVAD 模式的 end 行为差异

**能量模式**：无法区分"低能量正常停顿"和"低能量噪声"。end 只看能量低于阈值就累计，
正常呼吸/停顿也会被计入 end_count。因此能量模式下 end_count 需要设得更大（如 1000ms），
但这又导致长幻觉噪声无法拦住。**这是能量模式的固有限制**。

**TenVAD 模式**：停顿/呼吸虽然能量低但 TenVAD 可能给较高概率（停顿/呼吸有语音特征），
不容易误触 end。因此 TenVAD 可以用较小的 end_count（如 500ms）同时保留停顿和拦住噪声。
**这是 TenVAD 的核心优势**。

**建议**：生产环境优先使用 TenVAD 模式。能量模式作为轻量级替代（无 ONNX 依赖），
适用于对停顿保留不敏感的场景。

### 4.4 VAD 裁剪后音频时长的影响

VAD 裁剪后下发到客户端的音频总时长变短。影响：

1. **播放端计时**：如果客户端用 `audio_duration / sample_rate` 决定何时请求下一段，
   裁剪后的时长会导致客户端过早请求下一段。客户端应基于有效音频时长。
2. **timing metadata**：`done_meta` 中需区分原始时长和有效时长（见 3.3）。
3. **RTF 口径**：引擎侧 RTF = 原始音频时长 / 耗时（不变），
   客户端侧 RTF = 有效音频时长 / 耗时。

---

## 5. 实现要点

### 5.1 VAD 处理器接口

```python
class TTSVADProcessor:
    """Per-session 流式 VAD 处理器。"""

    def __init__(self, config: VADConfig, sample_rate: int = 24000): ...

    def process_chunk(self, pcm_int16: np.ndarray) -> np.ndarray:
        """输入一个 PCM chunk，返回应下发的音频（可能为空）。"""
        ...

    def flush(self) -> np.ndarray:
        """音频流结束，返回所有 pending 音频。"""
        ...

    def reset(self) -> None:
        """重置状态（session 结束时调用）。"""
        ...

    @property
    def trimmed_ms(self) -> float: ...
    @property
    def original_audio_ms(self) -> float: ...
    @property
    def effective_audio_ms(self) -> float: ...
    @property
    def begin_count(self) -> int: ...
    @property
    def end_count(self) -> int: ...
```

### 5.2 TenVAD 模式的内部降采样

TenVAD 要求 16kHz 输入，但引擎原始 PCM 是 24kHz。VAD 内部做降采样：

```python
# 简单的 3/2 降采样（24kHz → 16kHz）
# 每 3 个 24kHz 样本 → 2 个 16kHz 样本
# 使用线性插值或简单的 FIR 滤波
```

### 5.3 begin_count 的模式差异

| 模式 | 建议 begin_count | 理由 |
|------|-----------------|------|
| energy | 5 帧 (80ms) | 能量判断稳定，1-2 帧即可，5 帧留余量 |
| tenvad | 5 帧 (80ms) | TenVAD 单帧概率可能抖动，需 2-3 帧确认，5 帧留余量 |

### 5.4 end_count 的权衡

end_count 是**响应速度 vs 效果**的核心取舍：

- end_count 大 → 保护正常停顿，但幻觉噪声可能下发更多
- end_count 小 → 拦住更多幻觉噪声，但可能切断正常停顿

**需要实现后用真实幻觉样本做 A/B 测试确定最优值**。初始建议：
- energy 模式：end_count = 62 帧 (~1000ms)，因为能量模式无法区分停顿和噪声
- tenvad 模式：end_count = 31 帧 (~500ms)，因为 TenVAD 能更好保留停顿

---

## 6. 待验证项

| # | 项目 | 验证方式 |
|---|------|---------|
| 1 | TenVAD ONNX 在 TTS 实时约束下的 CPU 开销 | Benchmark：24kHz→16kHz 降采样 + TenVAD process() 的 P99 延迟 |
| 2 | 最优 end_count | A/B 测试：用真实幻觉样本测试不同 end_count 的噪声拦截率和停顿保留率 |
| 3 | 能量模式门限映射 | 录制真实 TTS 输出，统计静音帧/语音帧/噪声帧的 dB 分布，标定 begin/end 门限 |
| 4 | start_margin 是否足够 | 测试清辅音开头（f/s/sh 等）的 VAD 检测延迟，确认 20ms 回溯不截音 |
| 5 | 降采样质量 | 对比 24kHz→16kHz 线性插值 vs FIR 滤波对 TenVAD 概率的影响 |
| 6 | VAD 裁剪后客户端行为 | 验证客户端基于有效音频时长的播放/请求逻辑是否正常 |

---

## 7. 不在本次范围

- Backend 重复 token 模式检测（1-2-3-1-2-3）——独立任务，本期演进 `pad_silence_*` 职责
- 模型侧幻觉根因修复
- VAD 训练/微调
- 客户端多段音频拼接逻辑（VAD 对客户端透明，不需要）

---

## 8. 参考资料

- `workspace/ten_vad.py`：现有 ASR TenVAD 实现（参考推理核心，不复用状态机）
- `workspace/qwen3_tts_remote.py`：现有 dBFS prefix trim 实现（参考能量计算，将被替换）
- `engine/backend/engine_loop.py`：Backend pad silence 检测（职责分工参考）
- `engine/interface/output.py`：OutputPipeline（VAD 可观测性集成点）
- `engine/core/types.py`：VADConfig 定义（协议扩展点）
- [TEN VAD GitHub](https://github.com/TEN-framework/ten-vad)：TenVAD 推理核心参考
- [streaming_hallucination.md](../investigation/streaming_hallucination.md)：幻觉调查背景
