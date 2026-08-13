# 推测式音频审查、提交与回滚设计

状态：提案

适用范围：独立流式 TTS 服务（WebSocket、gRPC、frontend、backend）

最后更新：2026-07-28

## 0. 结论先行

本设计把当前的“按墙钟持有一段音频，段结束时再裁尾”改成：

> **不按墙钟节流的推测生成、连续前缀审核、原子提交、只回滚未提交音频。**

核心水位如下：

```text
0 ------ P ------ CB ------ RB ------ SB ------ M ------ N
        已播放     播放缓冲尾   SDK接收/回放ACK尾  不可撤销提交尾  安全前缀尾  已生成尾
                          [CB,RB) 已接收但未进入播放 buffer
                                  [RB,SB) 已提交、仍在服务端/链路中
                                          [SB_source,N_source) 未提交、可撤销
```

图中表达的是逻辑音频顺序。正常情况下，映射后的顺序为 `P ≼ CB ≼ RB ≼ SB ≼ M ≼ N`；
数值不变量必须分坐标书写：

```text
0 <= P_output <= CB_output <= RB_output <= SB_output
0 <= SB_source <= M_source <= N_source
```

用户最初模型里的 `CB` 如果只指“SDK 已收到/排队”，就等同这里的 `RB`；如果要用它判断真实
播放器是否会 underrun，则必须另有真正的 `CB`。协议必须拆开这两个语义，不能让 resume ACK 冒充
播放 buffer 进度。

审查器如果一次确认 `[SB, M)` 整段连续前缀安全，服务端就直接执行
`commit_through(M)`，把 `SB` 一次跳到 `M`，而不是按墙钟或逐帧慢慢移动。
合成头 `N` 不等待网络发送，也不等待逐帧提交，只在推测区达到内存或时长上限时
暂停该会话。因此，RTF 带来的生产余量会转化成 `[SB_source, N_source)` 的可操作空间。

本设计作出以下主要选择：

| 问题 | 选择 |
|---|---|
| 首包是否等待 4 帧重复确认 | **不等待**。做同步 PCM 健康检查和前缀静音裁剪后，以 `BOOTSTRAP_BYPASS` 提交最小首包 |
| 正常音频由什么推进 `SB` | verifier 给出的连续安全水位 `M`，不是墙钟 |
| 对客户端是否发送 tentative audio | **不发送**。外部协议不需要 rollback/replace 消息 |
| 发现异常后从哪里重合成 | 从 `SB` 之后、异常起点之前最后一个“安全且可恢复”的边界恢复 |
| 是否重复喂“尚未说完”的文本 | **不重复喂**。文本 token 可能已经进入旧 KV；应恢复精确生成状态和文本 cursor |
| 常规恢复策略 | 恢复 Talker KV 边界、Code2Wav 全状态、文本/RNG/循环状态，换采样轨迹续写 |
| 部分 codec ICL | 只作精确恢复失败后的二级 fallback；必须有可靠文本边界，不能只用 2～3 帧 codec |
| codec 能否直接做 CTC ASR | 可以训练轻量对齐头，但 codec ID 不能零训练直接映射成文字 |
| ASR 的产品目标 | 不是自由转写，而是“原始文本的播放进度条 + 内容不匹配检测” |
| 当前第一版 reviewer | PCM 健康、VAD、codebook-0 周期 1 重复；语义一致性暂不宣称已覆盖 |

## 1. 背景、目标与非目标

### 1.1 背景

当前服务的合成速度可以快于实时播放。若把生成出的 PCM 立即送入网络，异常被发现时，
错误音频通常已经进入客户端缓冲，服务端无法撤回。反过来，若首包等待固定 4 帧确认，
以当前每个 codec frame 约 80ms 计算，会增加约 320ms 首响，不能接受。

要同时利用首响和快于实时的生产能力，必须把以下三个概念分开：

- 客户端正在播放到哪里；
- 服务端已经不可撤销地承诺了哪些音频；
- 模型已经推测生成到哪里。

### 1.2 目标

- 首包不增加多帧审核等待；前缀静音不作为“首响”发送。
- 首包之后，正常交付只由增量审核结论驱动。
- 未触发 speculative/资源上限时，合成不受墙钟播放节奏限制；审核通过的连续前缀可成批 burst 交付。
- 对提交前及时发现的异常，只修改 `[SB_source, N_source)`；bootstrap/underflow bypass、reviewer
  false accept 或迟到检测可能污染已提交区，此时必须报告 breach，不能伪装成可撤回。
- 支持段内回滚并尽量保留安全前缀的音色、语速和韵律。
- WebSocket 断线恢复只回放已经提交的音频，不暴露 speculative audio。
- 审查器可从 VAD/重复规则逐步扩展到轻量 codec-text 对齐器。
- 所有降级交付都有显式 safety class、事件和指标，不能静默退化。

### 1.3 非目标

- 不让普通客户端接收 tentative audio 后自行撤销；这需要完全不同的播放器协议。
- 不承诺 VAD 和重复规则能识别“语音清晰但内容与输入完全无关”。
- 不把 Talker 的 `text_idx` 直接解释为真实发音进度。
- 不在第一阶段实现跨进程/GPU 重启后的精确生成状态恢复。
- 不把整段高质量 ASR 转写作为必要条件；这里关注的是安全提交水位和文本进度。

## 2. 水位、坐标系与不变量

### 2.1 主水位

| 水位 | 坐标 | 含义 | 谁推进 |
|---|---|---|---|
| `P` | 输出 sample | 客户端播放器已经消费的尾位置 | 客户端 `PlaybackProgress`，或保守估算 |
| `CB` | 输出 sample | 已真正进入客户端播放 buffer 的尾位置 | 客户端播放器集成 API |
| `RB` | 输出 sample | SDK 已完整接收并可用于 resume ACK 的尾位置 | `OutputAck`；现有 `audio_through_sample` 就是该语义 |
| `SB` | 输出 sample + source 映射 | 已原子提交、进入 outbound/replay ledger 的不可撤销尾位置 | `CommitCoordinator` |
| `M` | source codec frame | 所有 required reviewer 一致确认的连续安全前缀尾 | `ReviewCoordinator` |
| `N` | source codec frame | 当前 attempt 已经生成的尾位置 | backend decoder |

`SB` 不是“某个 Python deque 的尾”，而是逻辑不可撤销边界。一旦音频被交给 gateway
的不可回收队列、resume ledger 或 socket 写入层，就必须算在 `SB` 之前。

`M` 也不是“某个孤立位置看起来没问题”。它表示从当前提交尾开始的整个连续前缀都安全；
中间存在审核空洞时，不能跨洞推进。

### 2.2 两套坐标不能混用

backend 和 reviewer 天然使用 source codec frame；客户端和协议使用最终输出 sample。
VAD 可能删除前缀样本，输出流水线还可能重采样或改变编码，因此不能直接计算
`M - SB`。

每次提交必须产生稳定映射：

```text
CommitRecord {
  session_execution_id
  source_spans[] { segment_id, attempt_id, source_frame_start, source_frame_end }
  edit_ops[]      # DROP_PREFIX/DROP_TAIL/SYNTHETIC_GAP 等，含 native sample 范围
  output_sample_start
  output_sample_end
  commit_seq
  safety_class
  reviewer_mask
}
```

`P/CB/RB/SB_output` 在输出 sample 轴上比较；`SB_source/M/N` 在 source frame 轴上比较；
两者只通过不可变的 `CommitRecord` 关联。文中的 `SB <= M` 是上述 source 映射后的简写，
绝不能直接拿 output sample 数和 codec frame 数做算术。

单个 segment 内可用整数 frame boundary。多 segment 并发/reorder 时，source 位置定义为：

```text
SourcePos = (group_order, segment_order, local_frame_boundary)
```

按文本交付顺序做字典序比较。每个 segment 各有自己的 `M_s/N_s`；如果后续 segment 已生成而前面
存在洞，就不存在一个可直接提交的全局连续 `N`。图中的 `N` 是单条线性 attempt 的简写，
跨 segment 的 speculative 时长必须累加 AttemptBuffer 中的实际 frame/sample，不能用两个 tuple 相减。

每个实际目标 codec/PCM 输出都获得连续 `local_source_frame`：若某条 prefill 路径直接产生目标音频，
它就是 frame 0；否则 decode0 的首个目标输出是 frame 0。`local_source_frame` 与 Code2Wav 的
`absolute_frame_idx` 是两个字段，ICL/reference warm state 可能使目标 frame 0 对应非零 absolute index。
checkpoint journal 必须显式保存这组映射。

### 2.3 安全不变量

实现必须持续满足：

1. **不可撤销不变量**：任何 rollback 都要求 `reject_from >= committed_source_tail`。
2. **连续前缀不变量**：`M` 之前不能存在 required reviewer 的 pending/reject 空洞。
3. **单调不变量**：同一个 attempt 内，reviewer 的 `safe_through` 只能前进。
4. **attempt fence**：旧 `attempt_id` 的 audio、review、GPU future 和 EOS 不能写入新 attempt。
5. **先预留后提交**：只有 outbound/replay ledger 已为完整 commit batch 预留成功后，才能推进 `SB`。
6. **只交付 committed**：gateway、输出转换和 resume replay 只看到 committed audio。
7. **EOS 不等于安全**：生成结束只说明 `N` 不再增长，必须等 reviewer finalization 才能提交尾部。
8. **错误不 flush tentative**：engine/verifier error 时丢弃 tentative，不能像当前实现一样无条件 flush。
9. **提交点可恢复**：推进活动 segment 的新 `SB` 前，必须保证该 source 边界存在可恢复状态。
   唯一可省略长期 checkpoint 的情况是该 segment 已被所有 required reviewer final-safe，之后不会再从
   该 segment 产生可疑 suffix。迁移期的首包 bypass 是另一项显式、非可恢复例外，但必须回显
   `bootstrap_restorable=false`；后续 reject 只能 terminal，且服务不能宣称支持段内替换。
10. **安全类别不伪装**：首包或欠载越权提交必须标记为 bypass，不能计入 policy-passed 指标。
11. **terminal 线性化**：cancel、timeout、engine/verifier error 与 commit publish、child attempt 的
    begin/activate 使用同一 session lifecycle fence；final cursor 只能位于旧 `SB` 或一个完整新 commit
    batch 之后，不能落在半批，也不能 terminal 后再推进或激活新 attempt。

### 2.4 首包特例对排序的影响

首包绕过审核的提交瞬间可能出现：

```text
M_source < SB_source <= N_source
```

这不是“首包已被证明安全”，而是产品明确接受的一次不可撤销风险。该 commit 必须记录为
`BOOTSTRAP_BYPASS`；完成后只推进 committed floor `SB_source`，**不推进**代表 reviewer 真实证据的
`M_source`。reviewer 仍异步消费并审核 bypass frame；只有它后来给出连续 safe prefix，`M_source`
才可能追上并越过 `SB_source`。这只补齐审核事实，不自动补出可恢复状态：当
`bootstrap_restorable=true` 且 `SB_source` 的恢复基座仍存在时，服务端才能恢复活动段的普通
policy-passed 增量 commit；当 `bootstrap_restorable=false` 时，该 segment 始终按不可恢复迁移段处理，
即使 `M` 已追上也必须继续持有 suffix，直到所有 required reviewer 给出 final-safe terminal 后一次性提交，
或遇到 reject/timeout 后 terminal。此迁移模式不得靠暂停越过 speculative cap：非 finalized segment
达到 `max_speculative_ms` 且不存在可提交的 restorable boundary 时，必须 fail-stop 为
`nonrestorable_speculative_limit`。若
后来证明异常从 `SB` 之前开始，只能记录 `safety_breach_after_commit`。默认必须 freeze + terminal：
当前 `SB` 的 Talker/C2W 状态可能已被异常上下文污染，不能假设“从 SB 继续”就是干净的。只有 start
时显式接受 `committed_breach_action=ALLOW_CLEAN_RESTART`，并且运行时找到独立证明干净的 checkpoint，
或在合法文本边界完成 fresh/ICL restart，才能以 degraded recovery 继续；否则一律 terminal。
没有 breach 且 reviewer 追上时，水位排序恢复为 `SB_source <= M_source <= N_source`；是否允许继续增量
提交仍单独受 `Restorable` 条件约束。

## 3. 当前实现到了哪里

### 3.1 当前实际路径

```text
backend AUDIO_CHUNK
  -> frontend AudioReorder
  -> DeliveryHoldWindow
  -> gateway VAD
  -> output conversion / coalescing
  -> WebSocket or gRPC
```

当前已经具备一些可复用基础，但尚未实现本设计的核心语义：

| 能力 | 当前状态 | 与目标的差距 |
|---|---|---|
| 首块立即发出 | 已实现 | 首块还没有正式 safety class；前缀静音处理位于更下游 |
| guarded hold | 已实现 | [`release_due()`](../../../engine/frontend/hold_window.py) 由墙钟和 window 推进，不由 reviewer 推进 |
| 段尾裁剪 | 已实现 | 只有 `SEGMENT_END` 时 settle，不能增量产生 `M` |
| lookahead 段换 seed 重跑 | 已实现 | 只适用于整个 attempt 仍在 reorder 的段；没有段内 rollback |
| codebook-0 重复守卫 | 已实现 | 只检测连续 4 帧同 token 的周期 1；结论只走 abort/EOS 路径 |
| PCM 健康/静音防御 | 已实现一部分 | 不是统一 reviewer contract，不能独立推进安全水位 |
| VAD | energy/TenVAD 已实现 | 位于 hold 之后，是输出 gate，不是 speculative reviewer |
| WebSocket 长连接 | 一次一个活动 session，可串行复用 | 可沿用，但需要扩展 commit/safety/playback 字段 |
| WebSocket resume | 已有 delivery/sample cursor、ACK、replay ledger、generation fence | ledger 目前没有“只允许 committed audio”的上游约束；ACK 不代表真实播放头 |
| gRPC bidi | 支持 start/text/end/cancel | 没有输出 ACK、绝对 sample cursor 或逻辑恢复 |
| Talker KV | slot pool 中按位置追加 | 可以用 `past_len` 回退，但还没有 checkpoint API |
| Code2Wav 状态 | 有滑窗 KV 和递归状态 | 每步原地更新，不能只调小长度恢复 |

### 3.2 当前 guarded delivery 的关键问题

[`DeliveryHoldWindow`](../../../engine/frontend/hold_window.py) 当前按“首发墙钟经过时间 +
固定 window”计算可放行字节数；[`frontend/interface.py`](../../../engine/frontend/interface.py)
每 100ms 调用一次 `release_due()`。因此即使没有新增安全结论，`SB` 也会自动前进。

段正常 EOS 时会 `flush()` 全部持有音频；loop/silence abort 时会裁掉估计的尾部再 flush。
它可以减少一部分已知尾部跑飞，但仍是：

> **时间驱动放行 + 段尾补救**，不是 **审核驱动提交 + 未提交回滚**。

### 3.3 当前重跑的边界

当前 `SEGMENT_RETRY` 会丢掉 reorder 中整个失败 attempt，把段重新放回 prefill，使用加盐 seed
重跑。只有前面仍有更早段在播放、该段整段还没进入交付路径时，这种办法才安全。

“是不是 lookahead 段”目前由是否存在更早 live segment 近似判断，不是真实检查 `SB`；
`EngineResult` 也没有完整的 `attempt_id/frame_start/frame_end/checkpoint_id`，所以无法可靠处理
旧 attempt 的迟到结果。

### 3.4 当前 VAD 和 ACK 的语义

- VAD 在各 gateway 的 output callback 中运行，已经晚于 frontend hold 的放行决策；它会删样本，
  也使 frontend 字节水位与 wire sample 水位不一致。
- 当前 VAD 在 speech 状态下会先发出低分帧，达到 end hysteresis 后才关闭语音段；它不是天然的
  保守尾部 reviewer，改造后必须把尚未定性的低分 run 保持 pending。
- WebSocket SDK 当前是在 `AudioChunk` 放进 SDK 本地消息队列后推进
  `audio_through_sample` 并发送 ACK。这个 ACK 对应 `RB`，可表示“SDK 已接收/排队”，但不等于
  播放器缓冲尾 `CB`，更不等于 `P`。
- gRPC 没有等价输出 ACK；断流会取消 engine session。

### 3.5 当前审核能力的真实边界

当前能发现或缓解：

- 非法 f32、NaN/Inf；
- 整段静音或较长尾部静音；
- codebook-0 连续相同 token 的周期 1 重复；
- 一部分超长 runaway。

当前不能可靠发现：

- 能量正常的随机噪声；
- 语音清晰但内容与原始文本无关；
- 非周期 1 的一般重复；
- 漏字、跳字、插入内容；
- 声码器产生但 codec 语义层未体现的所有波形异常。

## 4. 目标架构

### 4.1 数据流

```text
                         +----------------------+
text stream ------------>  Talker / Code2Wav   +----> ProducedAudioFrame
                         +----------------------+              |
                                                               v
                                                +---------------------------+
                                                | AttemptBuffer             |
                                                | codec + PCM + checkpoints |
                                                +---------------------------+
                                                   |       |       |
                                      +------------+       |       +-------------+
                                      v                    v                     v
                                  PCM health              VAD              repeat / ASR
                                      +--------------------+---------------------+
                                                           v
                                                 ReviewCoordinator
                                                 safe M / reject H
                                                           |
                                      +--------------------+--------------------+
                                      |                                         |
                                      v                                         v
                             CommitCoordinator                         RollbackCoordinator
                               SB <- C(M)                            restore R, attempt++
                                      |
                                      v
                         output transform + CommitLedger
                                      |
                                      v
                           WebSocket / gRPC committed audio
```

其中 `Restorable(r)` 表示活动 segment 可从 `r` 精确恢复，`FinalSafeTerminal(r)` 表示 `r` 是已经被
全部 required reviewer final-safe、不会再产生 suffix 的最终 `N`。定义：

```text
CommitEligible(r) = Restorable(r) OR FinalSafeTerminal(r)

C(M) =
  SB_source,                                             if M <= SB_source
  max { r | SB_source < r <= M and CommitEligible(r) }, if the set is non-empty
  SB_source,                                             otherwise
```

`K(M)` 专指上述集合中满足 `Restorable` 的最远边界；final-safe terminal 是明确例外。bootstrap 和
underflow bypass 走各自显式路径，不伪造 `M`。若所有活动边界都能从 anchor 精确重放，正常活动段
可简写为 `C=min(M,K_restorable_frontier)`；只有恢复能力能跟上 `M` 时，才能真正把 `SB` 跳到 `M`。

### 4.2 组件职责

#### AttemptBuffer

- 按 `segment_id + attempt_id + source_frame` 保存 speculative frame；
- 保存完整 codec、PCM、文本 cursor、轻量 checkpoint journal；
- 对 gateway 完全不可见；
- 达到 `max_speculative_ms` 时对该 session 施加 decode backpressure；
- rollback 或 commit 后按水位回收；完整目标 profile 始终保留当前 `SB_source` 的恢复基座，以及物化
  该基座所需的最近 C2W anchor/codec log；
- 若 StartAck 显式回显 `bootstrap_restorable=false`，则为 bootstrap segment 记录“无恢复基座”的迁移
  状态，禁止该活动 segment 再做普通增量 commit；只能等待 final-safe terminal 一次提交或 fail-stop，
  不能用一个不存在的 anchor 满足 `Restorable(SB)`。

#### ReviewCoordinator

- 接收所有 reviewer 的单调 verdict；
- 对 required reviewer 取连续安全前缀交集；
- 产生聚合 `M` 或最早 `reject_from=H`；
- 拒绝旧 attempt 和倒退的 safe watermark；`H < SB_source` 的 reject 不是非法 verdict，必须接收并
  转成 `safety_breach_after_commit -> freeze`。默认 `TERMINATE`；只有 StartAck 已接受
  `ALLOW_CLEAN_RESTART` 且确有独立干净恢复边界时才允许 degraded recovery，不能为了维护水位而丢掉。

#### CommitCoordinator

- 只提交全局文本/segment 顺序上的连续前缀；
- 正常 target 必须已审核且满足 `CommitEligible`（活动段可恢复，或 final-safe terminal）；
  bootstrap/underflow 只走各自显式 bypass 例外；
- 为整批输出原子预留 ledger，再更新 `SB`；
- 产生 source-frame 到 output-sample 的稳定映射；
- 不执行正常的墙钟 release。
- 与 TerminalCoordinator 共享 publish 序列；任何 terminal cause 先冻结新 commit，再在线性化点锁定
  `final_delivery_seq/final_committed_sample`。

#### RollbackCoordinator

- 在 GPU step boundary fence 旧 attempt；
- 选择安全且可恢复的回退边界 `R`；
- 恢复 Talker、Code2Wav、文本、采样和 reviewer 状态；
- 换采样轨迹生成新 attempt；
- 按策略切换 exact、ICL 或 fresh-segment fallback。

#### CommitLedger

- 只保存已 committed 的 logical delivery；
- 为 WebSocket resume、ACK trim、重放和 sample cursor 提供唯一事实源；
- tentative frame 永远不能获得 `delivery_seq`；
- ledger 满时先 backpressure commit，不能先推进 `SB` 再丢记录。

### 4.3 内部消息契约

建议把当前只带裸 PCM 的 `AUDIO_CHUNK` 扩展成以下不可变消息：

```text
ProducedAudioFrame {
  session_execution_id
  slot_id
  slot_allocation_epoch
  segment_id
  attempt_id
  source_frame_start
  source_frame_end
  full_codec_ids       # [16, T]，第一版 T=1
  pcm_f32
  text_cursor_before
  text_cursor_after
  checkpoint_before
  checkpoint_after
}

ReviewVerdict {
  session_execution_id
  segment_id
  attempt_id
  review_epoch_id
  reviewer
  reviewed_through_frame
  safe_through_frame
  reject_from_frame?   # 一旦设置，必须指向最早可疑起点
  confidence
  reason
  edit_plan_version?
  edit_plan?          # attempt-fenced，不能修改已提交 edit
  final
}

RollbackRequest {
  segment_id
  attempt_id
  reject_from_frame
  reviewer
  reason
}

AttemptStarted {
  segment_id
  attempt_id
  attempt_ordinal             # 1-based；不得超过 max_attempts_per_segment
  lifecycle: PREPARING | ACTIVE | FAILED
  parent_attempt_id
  fork_frame
  review_epoch_id
  review_floor
  retry_budget_owner_segment_id
  breach_id?                  # 仅 committed-breach clean restart
  base_checkpoint_attempt_id
  rollback_from_frame
  restore_checkpoint_id
  recovery_strategy
}
```

所有 retry/recovery 路径统一调用一个原子的 `begin_child_attempt(...)` 事务，不能各自先恢复、prefill 或
物化状态再补 attempt 身份：

1. 先取得 session lifecycle/publish fence，确认 `terminal_pending=false`、parent 仍是 current 且生成未被
   cancel；保持该 fence 时再取得 retry budget owner 锁，校验并递增 `attempt_ordinal`，分配不可复用的
   `attempt_id/review_epoch_id`，继承 `retry_budget_owner_segment_id`；超过上限则不创建 child。
2. 持久化 `lifecycle=PREPARING` 的 `AttemptStarted`，同时增加 terminal 使用的
   `retry_attempts_started_total`；从这一刻起即算“实际已启动”，即使 prefill/replay 随后失败也不退还
   ordinal 或统计。
3. 返回不可变 child handle。Talker restore/prefill、C2W replay、checkpoint materialization、reviewer
   恢复和所有 GPU future 都必须捕获该 handle；每次 state scatter 前重新校验 session generation 与 child
   lifecycle，`PREPARING` 状态不得生成可提交 delivery。
4. base 完整就绪后，`activate_child_attempt(handle)` 重新取得 session lifecycle/publish fence，只在
   `terminal_pending=false`、handle 仍是 current `PREPARING` child 且 generation fence 未变化时才能 CAS
   为 `ACTIVE`。若需要 warning，先校验 child、为 warning ledger 预留并在**同一临界区**发布 barrier，
   随后 activate；不能在 warning 与 activate 之间释放 fence。准备/激活失败改为 `FAILED`，旧 parent
   不能复活。

因此“预算预检”本身不算 attempt，`begin_child_attempt` 成功才计数；但成功后即使一帧 codec 都没产出也
已经消耗一次。这个定义同时用于 exact、换 seed、ICL、fresh 和 committed-breach clean restart。

`begin_child_attempt` 和 `activate_child_attempt` 返回 tagged result，禁止用 `NONE/false` 抹掉真实竞态
原因：

```text
BeginChildResult = STARTED(handle)
                 | BUDGET_EXHAUSTED
                 | TERMINATING(existing_cause)
                 | STALE_PARENT
                 | ALREADY_SUPERSEDED(active_child_id)
                 | INTERNAL_ERROR(code)

ActivateChildResult = ACTIVATED
                    | TERMINATING(existing_cause)
                    | STALE_OR_SUPERSEDED
                    | INTERNAL_ERROR(code)
```

所有 exact、换 seed、ICL、fresh 和 clean-restart 调用点使用同一处理表：

- `STARTED/ACTIVATED`：只有这两个成功结果能继续对应阶段；
- `BUDGET_EXHAUSTED`：以 `retry_exhausted` 竞争 session terminal fence；只有赢得线性化点才发布，若已有
  terminal 则 join 现有结果；
- `TERMINATING`：join/drop，绝不能用 retry 或 internal error 覆盖已存在的 cancel/timeout/error；
- `STALE_PARENT/ALREADY_SUPERSEDED/STALE_OR_SUPERSEDED`：把当前重复/迟到控制请求幂等丢弃，不能终止
  已经成功的 sibling child；
- `INTERNAL_ERROR(code)`：用自身稳定错误码竞争 terminal fence，失败则 join 已有 terminal。

事务内部持久化要么连同 ordinal/统计一起成功并返回 `STARTED`，要么不产生可见 child；若底层结果
不确定，必须返回不可恢复的 `INTERNAL_ERROR(attempt_journal_uncertain)`，不能猜成预算耗尽。

锁序固定为 `session lifecycle/publish fence -> retry budget owner lock`，任何路径都不得反向取得；多个
budget owner 需要同时读取时按稳定 segment order 排序。terminal transition 先取得 session fence，原子
设置 `terminal_pending`、推进 session generation，并把所有 `PREPARING/ACTIVE` child fence 后置为
`FAILED`，再固化 `retry_attempts_started_total/segments_retried` 和 final cursor。这样竞态结果只有两种：
begin 先线性化则计数包含该 child、随后 terminal 将其失败；terminal 先线性化则 begin 失败且不计数。
旧 GPU future 即使随后完成，也必须在 scatter 前被 generation/child identity 丢弃。

安全相关控制消息必须使用可靠、有界并可 backpressure 的队列。当前 backend result queue 满时
可能丢结果的行为不能用于 verdict、rollback、commit 或 terminal 消息。

新 attempt 是 suffix fork，而不是把整个历史改名：`[0,fork_frame)` 仍来自 parent attempt，
`[fork_frame,...)` 才属于新 attempt。新 reviewer 以 fork boundary 作为 epoch base，并恢复该边界的
上下文状态；不能把 parent 的 `safe_through` 数值直接当成新 attempt 自己产生的 verdict。

### 4.4 四个正交状态机

不要用一个大枚举同时表达输入、生成、提交和网络状态。

#### 输入状态

```text
OPEN --end/stop--> CLOSED
OPEN --cancel----> CANCELLED
```

`CLOSED/CANCELLED` 后收到新文本一律报稳定协议错误。

#### 生成状态

```text
PREFILL -> GENERATING(attempt=a) -> FINISHING -> FINISHED
                    |                    ^
                    +-> ROLLBACK_PENDING |
                          -> RETRYING ----+

任意非终态 -> FAILED / CANCELLED
```

#### 提交状态

```text
EMPTY -> TENTATIVE(SB_source < N_s)
      -> COMMITTING(SB_source < C_source)
      -> TENTATIVE(new SB_source, per-segment N_s may continue)
      -> DRAINED
```

#### transport attachment 状态

```text
ATTACHED -> DETACHED_GRACE -> ATTACHED(new generation)
                         \-> EXPIRED

terminal -> RETAINED_UNTIL_TERMINAL_ACK_OR_TTL -> CLOSED
```

transport generation fence 与 synthesis attempt fence 是两套不同的 fence，不能复用一个 ID。

### 4.5 多 segment 与 reorder

每个 segment 独立维护：

```text
attempt_id
produced_local_frame
safe_local_frame
restorable_local_frame
committed_local_frame
finalized
```

全局提交只能跨越文本顺序上的连续 segment 前缀。后续 segment 即使已经全部审核，也先留在
speculative reorder；前一 segment 一旦完成，CommitCoordinator 可以一次 burst 提交前一段尾部
和多个已审核后续段。这正是 `SB_source <- M_source` 的批量跳跃语义。

## 5. 审核驱动的提交算法

### 5.1 生成

每生成一个 frame：

```text
produce(frame):
  assert frame.attempt_id == active_attempt
  append frame and journal to AttemptBuffer
  N = frame.end
  fanout frame to reviewers
  if SpeculativeDuration(all uncommitted segment spans) >= max_speculative_ms:
      if engine is finalized and only reviewer finalization remains:
          wait for reviewers under review_timeout
      elif no restorable commit/reclaim path can advance before EOS:
          fence attempt; discard tentative; terminal(nonrestorable_speculative_limit)
      else:
          pause this session at next GPU step boundary
```

普通 pause 只作用于 verifier/checkpoint 暂时落后的会话 lane，不应阻塞其他 batch lane。对于
`bootstrap_restorable=false` 或整段仅能 `FinalSafeTerminal` 提交的 Phase 0/1 segment，pause 无法释放
buffer，因而禁止进入无限等待：frontend 应先把输入切成可装入 cap 的 bounded segment；实际生成仍超出
cap 时按上面的稳定 error 丢弃 tentative，只按序 drain 已 committed 的旧 `SB`，绝不 flush 尾巴。

### 5.2 安全推进

每个 required reviewer `i` 都维护自己的连续安全水位 `M_i`：

```text
M = min(M_pcm_health, M_vad, M_repeat, M_semantic_if_required, ...)
C = C(M)  # 按 4.1；M<=committed_source_tail 或无 eligible boundary 时为 no-op
```

若 `C > committed_source_tail`：

1. 收集 `[committed_source_tail, C)` 的 source spans 和 VAD edit plan；
2. 按最坏输出大小为整个 commit batch 在 ledger/outbound queue 预留容量；
3. 在事务性的 output-transform state 上完成裁剪、重采样、编码和 sample mapping；
4. 写入全部 immutable `CommitRecord` 和输出 bytes；
5. 原子发布新的 transform state、ledger 尾和 `SB`；
6. 唤醒 gateway 发送；
7. 活动 segment 原子替换 `SB_source` 的恢复基座；若是 final-safe terminal，则在 commit 成功后释放该段
   状态；再回收更老且不再需要的 speculative journal/anchor。

步骤 2～4 任一步失败时保持旧 `SB`，丢弃临时 transform state 并进入 backpressure/error，不能产生
半提交。若输出转换器有 resampler/filter history，就必须支持 clone/checkpoint 或延迟修改 live state；
不能出现“转换状态已经前进、ledger 却没有提交”的情况。

### 5.3 为什么不需要逐帧移动 `SB`

reviewer 可以保留不确定尾巴，同时不断审核更早 frame。例如重复阈值为 4 时：

```text
code0:  8  17  23  23  23
                      ^ 当前 N
```

`23` 的 run 尚未达到阈值时，只需让 run 起点之后保持 pending；run 之前的整个连续前缀可以
立即安全。下一帧若 token 变成 42，旧的 23 短 run 整体变安全，但 42 同时是一个新未决 run 的
第一帧，因此 `M` 只能跳到 42 的起始边界，不能越过 42 到新 `N`；若下一帧仍为 23，则从 23 的
run 起点 reject。这里没有固定“每包必须等 4 帧”的规则。

### 5.4 异常推进

reviewer 在 `N` 才观察到异常，不代表异常从 `N` 才开始。它必须返回最早可疑起点 `H`。

```text
review_reject(H):
  if H < committed_source_tail:
      report safety_breach_after_commit
      freeze generation
      if committed_breach_action == ALLOW_CLEAN_RESTART:
          clean_restart_after_committed_breach(H)
      else:
          terminal
  else:
      choose last safe+restorable R where SB <= R <= H
      optionally commit [SB, R)
      fence attempt
      discard [R, N)
      begin_result = begin_child_attempt(parent, fork=R, strategy=EXACT)
      switch begin_result using the shared tagged-result table
      if begin_result == STARTED(child):
          restore state_at(R, identity=child)
          activate_result = activate_child_attempt(child)
          switch activate_result using the shared tagged-result table
          if activate_result == ACTIVATED:
              generate child
```

`ALLOW_CLEAN_RESTART` 不是从污染的 `SB` 直接继续，也不是普通 rollback。它必须执行完整的
breach transition。真正的准备错误停在旧 `SB` 并按自身 code 竞争 terminal；重复/stale 请求或已存在的
terminal 则严格按共享 tagged-result 表 drop/join，不能覆盖赢家：

1. 在 Commit/Terminal 共用的 publish lock 下冻结新 commit 和生成，fence 旧 `attempt_id/review_epoch_id`、
   GPU future 与 C2W launch；已经 committed 的 ledger 和 `SB` 原样保留。
2. 丢弃旧 attempt 的 `[SB_source,N_source)` 及其 reviewer/edit/checkpoint tentative 状态；旧
   `M_i/H` 只作为 breach evidence 保留，不能进入新 epoch。
3. 在不启动任何 GPU/state 工作的前提下验证：StartAck 接受 `ALLOW_CLEAN_RESTART`；所选
   EXACT/ICL/FRESH 在 `allowed_recovery_strategies` 中；下一 ordinal 预算预检可用；并选择 clean base。
   只允许两种 base：一条**独立生成且已验证**、逻辑上正好位于当前 `SB_source` 的替代 checkpoint，
   且依赖图不经过 `[H,N)`；或一项 fresh/ICL prefill 计划，其 target cursor 正好是 CommitRecord 映射出的
   “首个未提交文本边界”，reference 只取 `H` 之前另行 policy-passed 的 context。边界早于该 cursor 会
   重复已播放文本，晚于它会漏字；无法精确映射时不允许继续。
4. 调用 `begin_child_attempt(parent=old, fork=SB_source, strategy, breach_id)`，原子消耗 budget 并取得
   tagged result；只有 `STARTED(child)` 才继续。child 已包含新
   `attempt_id/attempt_ordinal/review_epoch_id` 和相同的
   `retry_budget_owner_segment_id`。从此以后，替代 checkpoint 的 materialization、fresh/ICL prefill、
   C2W state 和 GPU future 全部挂在 child identity 下，不能再使用 session/slot 的裸身份。
5. 在 `PREPARING` child 下物化 clean base，并初始化新 reviewer epoch。该 epoch 以 `SB_source` 为
   `review_floor`，各 `M_i` 从该 floor 开始审核新 suffix；floor 只表示
   “旧的不可撤销区不属于本 epoch”，不是把已知 breach 重新标成安全。旧 epoch 的迟到 verdict 由 fence
   丢弃。新帧沿原 `SourcePos` 从 fork boundary 续号，CommitRecord 用 child lineage 标识；准备期间的
   任何中间 PCM 都不可提交。
6. clean base 全部就绪后，调用带 warning 参数的 `activate_child_attempt(child)`；该事务在同一个 session
   lifecycle/publish 临界区先发布 replayable
   `verification_warning{breach_id, action=continued, recovery_strategy}`，再把 child 置为 ACTIVE。warning
   位于旧音频 ledger 之后、任何新恢复音频之前，且 terminal 不可能插入 warning 与 activate 之间。
   只有返回 `ACTIVATED` 才开始 decode；其他结果按共享 tagged-result 表处理。
7. 只有 ACTIVE child 重新产生 `M_new > SB_source` 且满足 `CommitEligible` 后才能发布新音频；publish lock
   保证 warning 一定先于首个恢复 commit。恢复失败则在同一线性化机制下发送唯一 terminal，不能回到
   旧 attempt 或越过 attempt budget 再试。

这套转移承认 `[H,SB)` 已经不可修复，只隔离后续输出；外部 warning 和指标必须保留该事实，不能因为
新 suffix 后来 policy-pass 就清除 breach。

周期 1 重复在第 4 个相同 token 才命中，但 `H` 应指向这次 run 的第一个 frame，恢复点应在
run 之前。

### 5.5 EOS

```text
engine EOS:
  freeze final N
  ask every reviewer to finalize(N)
  if any reviewer rejects:
      rollback or fallback
  elif all required reviewers final-safe through N:
      commit_through(N)
      emit segment_end
  else:
      wait until review timeout policy fires
```

EOS 不能直接 `flush()` hold buffer。
final-safe `N` 后该 segment 不会再产生 suffix，因此不要求为 `N` 保留长期 rollback checkpoint；但在
commit 事务成功前仍必须保留 PCM/codec/edit state，失败时不能丢尾部或发送 `segment_end`。

### 5.6 播放缓冲和 underflow

`P/CB` 只用于流控、观测和可选的欠载策略，不作为安全证据。推荐支持三种显式 profile：

| profile | 首包 | 正常提交 | verifier 落后导致欠载 |
|---|---|---|---|
| `review_gated_after_bootstrap`（推荐） | 最小 bypass | 只提交 policy-passed | 按协商的 STALL / INSERT_SYNTHETIC_GAP / ERROR，不越权 |
| `balanced_underflow`（实验） | 最小 bypass | policy-passed | 只提交受硬上限约束的最小跨度，标记 `UNDERFLOW_BYPASS` |
| `legacy_firehose` | 立即 | 不等待审核 | 保持当前兼容行为 |

若启用 `balanced_underflow`，服务端可计算播放 deadline：

```text
trigger only if samples_to_ms(CB_output - P_output) < low_water_ms

first run normal C(M) commit and refresh current SB
D_output = P_output + target_client_buffer_samples

candidates = {
  r | SB_source < r <= N_source
      AND Restorable(r)
      AND ProjectedOutputEnd(r, current_edit_map) - SB_output <= max_bypass_once_samples
}

reaching = { r in candidates | ProjectedOutputEnd(r, current_edit_map) >= D_output }

forced_source =
  reaching 中 output end 最小的 r, if reaching is non-empty
  candidates 中 output end 最大的 r, otherwise if candidates is non-empty  # best effort
  NONE, otherwise
```

若 `D_output <= SB_output`，已有 committed in-flight 足以覆盖目标，不产生 bypass。否则只在 candidates
非空时提交离散整帧增量 `[SB_source, forced_source)`，整段标记为 bypass：优先选择第一个达到 target
的可恢复边界；如果离散 frame 在 hard cap 内无法达到 target，就选择 cap 内最远边界 best effort。
没有候选就 stall/error，不能越过 hard cap。配置校验应把非零 cap 至少量化到一个合法 commit 粒度，
否则该 profile 实际永远无法 bypass。
必须记录 source/output 的具体跨度和原因。
`ProjectedOutputEnd` 由 tentative native sample、裁剪计划和输出采样率保守计算，不能直接拿
`D_output` 与 codec frame 比较。没有可信 `P` 时只能
用首包发送时间做保守估算。默认不能因为客户端声称即将 underrun 就无条件释放大量未审核音频。

`balanced_underflow` 还必须有服务端硬限制：单次最大 bypass、整 session 累计最大 bypass、cooldown、
每分钟次数和 retry 后禁用窗口。客户端进度只能触发“最多到服务端上限”的请求，不能决定释放量；
超过上限转 stall/error，不能逐步退化成 firehose。第一版如果尚未完成这些限制，应不提供该 profile。
它还要求 `exact_rollback_v1` 和目标边界 `Restorable(r)`；没有 checkpoint 时不能把 underflow 当作新的
非可恢复例外，直接禁用该 profile。
与 bootstrap 相同，underflow bypass 只推进 `SB_source/SB_output`，不推进任何 reviewer 的 `M_i`；
reviewer 继续异步审核 bypass span。`M` 追上前不再产生普通 policy-passed commit，迟到 reject 按
`H < SB_source` breach 处理。

如果选择插入短静音来避免设备 underrun，这段静音是独立 committed output edit：使用
`SYNTHETIC_GAP` safety class、`source_spans=[]` 和明确 output sample range，不推进 `SB_source/M/N`。
恢复真实音频后 sample cursor 继续单调。若协议/客户端不能正确表达 synthetic span，第一版只允许
stall/error，不隐式插静音。

`INSERT_SYNTHETIC_GAP` 也受服务端硬 cap：单次时长、session 累计时长、累计次数和独立的
`synthetic_gap_cooldown_ms`；只在真实 low-water 触发，不能因恶意/停滞的 progress 无限生成静音。
超过任一 cap 或 cooldown 尚未结束时，转 StartAck 回显的 `synthetic_gap_exhausted_action=STALL|ERROR`，
不能继续填 gap。synthetic gap 不是 reviewer 进展，也不能重置 review timeout。它不共享
`bypass_cooldown_ms`，避免两个动作互相隐式改变限流语义。

### 5.7 首包立即发送不等于一定连续播放

首包绕过解决的是 TTFA，但 reviewer 仍会在 `N` 后保留动态尾巴。以 80ms/frame、2× 生成速度为例，
重复 reviewer 遇到正常 3-frame 同 token run 时，要等下一个不同 token 才能把整段确认；这段确认等待
可能消耗超过一个 80ms bootstrap frame。平均 RTF≥2 本身不能证明不会 underrun。

对未来帧需求为 `h`、codec frame 时长为 `F`、生成速度倍数为 `G` 的 reviewer，一个保守容量条件是：

```text
bootstrap_playable_ms + client_startup_buffer_ms
  > h * F / G + reviewer_compute_ms + checkpoint_materialize_ms + network_jitter_budget_ms
```

实际还要用离散事件仿真覆盖 run 起点和 packet 边界。可选解法只有明确取舍：

- 客户端收到首包后保留很小 startup buffer；这不延迟服务器首包，但会增加真实播放启动；
- 提升 reviewer/checkpoint 吞吐或使用更早证据缩短动态 horizon；
- 在协商上限内使用 `UNDERFLOW_BYPASS`；
- 允许设备短暂停顿/显式 synthetic gap；
- 增大 bootstrap bypass，但这会扩大未审核暴露，不应静默采用。

因此上线必须同时报告 server TTFA、client first-play latency 和连续播放 underrun rate，不能只优化其中一个。

## 6. 长连接协议设计

### 6.1 外部协议原则

- 客户端只接收 committed audio；rollback 是服务端内部事务。
- receipt/replay ACK 与真实 playback progress 分开。
- 所有 cursor 累计、单调；只有 receipt `RB` 必须落在完整 delivery 边界，`P/CB` 可落在 delivery 内。
- 一个物理 WebSocket 同时只承载一个活动 session，terminal 后才能串行复用。
- `done`/`error` 是逻辑 session 边界，客户端不等待 socket close。
- `done` 只能在 input closed、engine finished、全部 tentative 已 commit/drop、没有 pending retry，且
  committed ledger 已排在 terminal 之前时发送。
- review-gated `cancel` 停止新生成/commit 并丢弃 tentative，但已经进入 CommitLedger 的 delivery 按序
  drain 到既有 `SB_output`，随后发送 cancelled terminal；不能 purge ledger 后继续复用同一序列。
- 恢复只覆盖 committed ledger；tentative 状态由原 engine session 在 grace 内保留。

### 6.2 capability 与 start 协商

新增 capability：

```text
review_gated_delivery_v1
output_cursors_v1
playback_progress_v1
attempt_fencing_v1
exact_rollback_v1
```

`exact_rollback_v1` 还应报告支持的 loaded model/task、pooled/non-pooled 路径、C2W replay parity 版本和
最大 speculative/checkpoint 预算；不能只用一个布尔值掩盖部分路径不支持。

建议把长期协议从 `OutputPolicy.config` 的自由字符串升级为 typed policy：

```text
DeliveryReviewPolicy {
  mode: REVIEW_GATED | LEGACY_FIREHOSE
  bootstrap_policy: BYPASS_FIRST_AUDIBLE_PACKET | VERIFY_ALL
  minimum_required_reviewers: [PCM_HEALTH, VAD, CODEC_REPEAT, ...]
  requested_optional_reviewers: [ASR_ALIGNMENT, ...]
  retry_policy {
    max_attempts_per_segment
    allow_sampling_parameter_change
    allowed_recovery: [EXACT, ICL, FRESH, ERROR]
  }
  committed_breach_action: TERMINATE | ALLOW_CLEAN_RESTART
  allowed_start_profile_degrades[]
  max_speculative_ms
  review_timeout_ms
  underflow_policy: STALL | BYPASS_MINIMAL | INSERT_SYNTHETIC_GAP | ERROR
  underflow_limits {
    low_water_ms
    target_buffer_ms
    bypass_max_once_ms
    bypass_max_session_ms
    bypass_max_per_minute
    bypass_cooldown_ms
    disable_bypass_after_retry_ms
    synthetic_gap_max_once_ms
    synthetic_gap_max_session_ms
    synthetic_gap_max_count
    synthetic_gap_cooldown_ms
    synthetic_gap_exhausted_action: STALL | ERROR
  }
  require_review_gated_delivery
}
```

start response 必须回显服务端实际接受的 policy、reviewer、首包策略和 buffer 上限。
客户端设置 `require_review_gated_delivery=true` 而服务端不支持时，应在产生任何音频前失败。

StartAck 至少包含正式字段：

```text
DeliveryReviewAcceptance {
  profile_id
  profile_version
  actual_required_reviewers[]
  accepted_optional_reviewers[]
  bootstrap_policy
  bootstrap_max_ms
  bootstrap_restorable
  exact_rollback_supported
  underflow_policy
  max_speculative_ms
  max_checkpoint_bytes
  review_timeout_ms
  vad_begin_count
  vad_end_count
  vad_tail_pending_ms
  max_attempts_per_segment
  retry_sampling_change_allowed
  allowed_recovery_strategies[]
  committed_breach_action
  accepted_start_profile_degrades[]
  underflow_low_water_ms
  underflow_target_buffer_ms
  underflow_bypass_max_once_ms
  underflow_bypass_max_session_ms
  underflow_bypass_max_per_minute
  underflow_bypass_cooldown_ms
  underflow_disable_after_retry_ms
  synthetic_gap_max_once_ms
  synthetic_gap_max_session_ms
  synthetic_gap_max_count
  synthetic_gap_cooldown_ms
  synthetic_gap_exhausted_action
}
```

`bootstrap_restorable=false` 是迁移期能力，不是隐含默认；此时 StartAck 不能同时声称活动段可替换。
VAD 等阈值也可以只由 `profile_id/version` 间接固定，但 wire/debug capability 必须能解析到唯一实际值，
不能让两台 server 用同一 profile version 运行不同状态机。
所有正文依赖的“start 时明确允许”都以这份 acceptance 为准：实际 retry/degrade/recovery/sampling/breach
contract 未回显就视为不允许。服务端只能收紧客户端上限，不能扩大客户端没有允许的恢复、参数变化或
已提交安全违规后的继续权限。`committed_breach_action` 缺省值固定为 `TERMINATE`；
`ALLOW_CLEAN_RESTART` 只授权使用已经接受的 ICL/fresh（或独立 clean checkpoint）策略，不表示服务端
可以从受污染的 `SB` 继续，也不允许重放已经 committed 的文本。

`max_attempts_per_segment` 是不小于 1 的整数，**包含初次 attempt**。例如值为 3 表示初次生成加最多
2 次重生成；每个 segment 独立计数，fallback 产生的新分支也占用一次 attempt。任何会使已启动计数
超过该上限的新 attempt 均不得启动。

协议迁移期可以继续通过 `OutputPolicy.config` 传递同名字段，但不能长期依赖字符串 map 表达
安全契约。

建议首批 profile 真值表：

| profile | required reviewer | 首包 | review timeout | 有 attempt budget 时的恢复顺序；耗尽后 |
|---|---|---|---|---|
| `guarded_legacy` | 无统一 contract | 旧行为 | 墙钟继续放行 | 旧段尾策略 |
| `rules-v1` | PCM health + VAD + codec repeat | 一次 bootstrap bypass | stall/error | exact；耗尽后 error |
| `semantic-shadow-v1` | 与 `rules-v1` 相同；semantic 仅观测 | 一次 bootstrap bypass | semantic 不阻塞 | semantic 不触发 retry |
| `semantic-gated-v1` | rules + semantic | 一次 bootstrap bypass | abstain/stall/error | exact -> allowed fallback；耗尽后 error |

profile/version 由服务端决定并在 StartAck 固化，session 中途不能改变；任何降级必须是 start 时允许的
有限转换并回显新 profile，不能在 verifier crash 后自动切到 legacy。

客户端字段表达最低安全要求，不是用来关闭服务端 reviewer：

```text
actual_required = server_profile_required UNION client_minimum_required
```

任何 client minimum 不可用都必须在首个音频前 fail，不能走普通 degrade；服务端 profile 自带的 required
也不能因客户端省略而删除。optional request 是否接受可以回显，但不改变 minimum 失败语义。
若 union 改变了 required set，StartAck 必须回显对应的复合 profile ID/version，不能仍冒用原 profile 名称。

### 6.3 committed audio header

WebSocket 在现有 `audio_header + raw PCM binary` 上扩展：

- sample cursor 从 session 输出起点 0 开始，使用 `[start_sample,end_sample)` 半开区间；
- 一个 sample 表示每声道同一时刻的 sample frame，多声道交错数据也只把 cursor 增加 1；
- cursor 按 start 协商后的最终输出采样率计数，session 内采样率、编码和声道数不可改变；
- `delivery_seq` 在 session 内从 1 单调递增，覆盖所有可 replay 的 audio logical delivery 和事件；
  无音频事件不改变 sample cursor；
- `commit_seq` 在 session 内从 1 单调递增，只标识 audio commit batch；一个 batch 可含多个 delivery；
- `attempt_id` 在 segment 内单调，外部只作诊断/审计，不允许客户端据此控制生成；
- `segment_end` 必须排在该 segment 最后一个 committed audio delivery 的 commit barrier 之后。

```json
{
  "type": "audio_header",
  "delivery_seq": 42,
  "commit_seq": 8,
  "start_sample": 19200,
  "end_sample": 23040,
  "segment_id": 3,
  "attempt_id": 2,
  "source_spans": [
    {"segment_id": 3, "attempt_id": 2, "frame_start": 15, "frame_end": 17}
  ],
  "safety_class": "policy_passed",
  "verification_profile_id": "rules-v1",
  "verification_profile_version": 1,
  "reviewers": ["pcm_health", "vad", "codec_repeat"],
  "audio": {
    "sample_rate": 24000,
    "encoding": "pcm_f32",
    "channels": 1
  }
}
```

随后紧跟一帧 binary。一个 `commit_through(M)` 可以拆成多个受消息大小限制的 delivery；
它们共享 `commit_seq`。tentative frame 没有 delivery sequence。

`source_spans/attempt_id` 可作为 capability-controlled 诊断字段，客户端音频连续性只依赖 output sample
cursor；这样公共协议不强绑定未来仍为 12.5Hz codec。发生裁剪或 synthetic gap 时，内部完整 edit map
仍以 CommitLedger 为准。

允许的 `safety_class` 至少包括：

- `POLICY_PASSED`
- `BOOTSTRAP_BYPASS`
- `UNDERFLOW_BYPASS`
- `SYNTHETIC_GAP`
- `LEGACY_UNVERIFIED`

`POLICY_PASSED` 只表示通过 `verification_profile_id/version + reviewer_mask` 指定的审核策略，仍存在
false accept；它不是“绝对无幻觉”。未来启用经过校准的语义 reviewer 时，可以使用不同 profile，
或另设 `SEMANTIC_POLICY_PASSED`，不能回头改变 `rules-v1` 的含义。

terminal event 使用 typed 字段：

```text
Terminal {
  type: done | error
  code
  retryable
  final_delivery_seq
  final_committed_sample
  policy_passed_ms
  bypass_ms_by_class
  retry_attempts_started_total
  segments_retried
  recovery_summary
}
```

resumable transport 中 terminal 自身也是最后一个 replayable delivery；`final_delivery_seq` 等于 terminal
wrapper 的 `delivery_seq`，`final_committed_sample` 则等于最后一段音频的 `SB_output`，两者不要混用。
`retry_attempts_started_total` 是本 session 所有 segment 中“初次 attempt 之外、实际已经启动”的 attempt
总数；`segments_retried` 是至少启动过一次额外 attempt 的去重 segment 数。二者都是 session 累计值，
不承担每 segment 限制，per-segment 上限唯一由 `max_attempts_per_segment` 定义。“实际已经启动”精确定义为
`begin_child_attempt` 已成功持久化 `PREPARING`；即使随后 prefill/replay 失败且没有产出 codec，也计入。

attempt-local speculative/retry event 不进入外部 replay ledger。terminal/error 和 rollback 控制消息必须有
保留的控制面队列容量，不能被 PCM backlog 挤掉。

正常 exact retry 默认是内部事件，不打断客户端音频流。只有安全契约或播放体验发生变化时才发外部事件：

```text
quality_degraded {
  reason: bootstrap_bypass | underflow_bypass | fallback_recovery
  output_start_sample
  output_end_sample
  recovery_strategy?
}

verification_warning {
  breach_id?
  reason: safety_breach_after_commit | verifier_unavailable | retry_exhausted
  action: continued | truncated | terminating
  recovery_strategy?
}
```

调试模式可额外按 segment 发 `WatermarkUpdate(SB_source, M_s, K_s(M_s), N_s)`，普通播放器不依赖它。reason code
必须稳定，详细阈值/evidence 只进受控日志，避免把内部模型数据放到 wire 上。

### 6.4 两类客户端反馈

receipt/replay ACK：

```json
{
  "type": "ack",
  "through_delivery_seq": 41,
  "audio_through_sample": 19200
}
```

继续兼容现有语义：SDK 已完整接收 logical delivery，服务端可以 trim replay log。这里的
`audio_through_sample` 不能被解释为 `P`。

真实播放器反馈使用独立消息：

```json
{
  "type": "playback_progress",
  "played_through_sample": 14400,
  "buffered_through_sample": 19200,
  "observed_delivery_seq": 41,
  "client_monotonic_ms": 81234567
}
```

- `played_through_sample` 对应 `P`；
- `buffered_through_sample` 对应真实播放器 `CB`，不是 SDK reader queue；
- 客户端 SDK 应提供播放器集成 API，而不是自行猜测消费进度；
- 没有播放器反馈时，服务端只做保守估算，不影响 reviewer 安全结论。

反馈是不可信输入：只允许单调前进，必须满足 `P <= CB <= RB <= SB_output`。receipt `RB` 必须落在
完整 logical delivery 边界；播放器的 `P/CB` 可以落在 delivery 内部，但不得超出客户端声明已经观察到
的 delivery/sample。旧 ACK 可幂等忽略，超前或 receipt 边界不匹配应返回稳定协议错误。

### 6.5 WebSocket resume

沿用现有高熵 resume token、attachment generation、`delivery_seq`、绝对 sample cursor 和
process-local replay registry。

断线后：

1. fence 旧 transport attachment；
2. committed ledger 和 tentative attempt 都在 grace 内保留；
3. 如果 speculative buffer 达上限，暂停该 session 的 decode；
4. 用 `(through_delivery_seq, audio_through_sample=RB)` 校验 resume cursor，并回放所有
   `delivery_seq > through_delivery_seq` 的 logical delivery，包括 sample cursor 不变化的 warning、done、
   error/terminal；对音频而言，这等价于补齐 `[RB, SB_output)`，但不能只按 sample 区间筛 ledger；
5. `[SB_source, N_s)` 的各 segment speculative span 保留在 AttemptBuffer，继续审核或等待恢复，
   绝不进入 replay；
6. grace 过期后 cancel engine、丢弃 tentative、清理 ledger。

resume cursor 永远不能指向 speculative frame。进程或 GPU 崩溃后没有 live state，必须明确
`resume_not_found/state_lost`，不能静默从文本头重合成并拼到旧音频后。

### 6.6 gRPC 扩展

建议在 `SynthesizeRequest.oneof` 增加 typed 消息：

```proto
message OutputAck {
  uint64 through_delivery_seq = 1;
  uint64 received_through_sample = 2;
}

message PlaybackProgress {
  uint64 played_through_sample = 1;
  uint64 buffered_through_sample = 2;
  uint64 observed_delivery_seq = 3;
  uint64 client_monotonic_ms = 4;
}
```

`AudioChunk` 增加：

```proto
uint64 delivery_seq = 6;
uint64 commit_seq = 7;
uint64 start_sample = 8;
uint64 end_sample = 9;
uint32 segment_id = 10;
uint32 attempt_id = 11;
SafetyClass safety_class = 12;
```

旧 protobuf client 会忽略未知 response 字段；新 client 只在 capability 声明后发送新 request。
gRPC 第一阶段可以只提供 cursor/ACK 和 fail-fast，不必与 WebSocket 同时实现逻辑 resume；
第二阶段再定义 resumable bidi stream。

若 gRPC 也要求严格 text sequence，response oneof 需正式增加：

```proto
message TextAck {
  uint64 through_seq = 1;
  bool duplicate = 2;
}
```

第一阶段 capability 必须明确 `grpc_output_ack=true`、`grpc_resume=false`，不能因为 WebSocket 已支持
resume 就对 gRPC 作相同承诺。

现有 gRPC 还应顺手统一以下终态语义：

- text-before-start 不再静默忽略；
- `TextChunk.seq_no` 做严格递增/幂等校验并 ACK；
- error 与 done 只能出现一个 terminal；
- cancel 后不再 drain tentative 或旧的未提交队列。

### 6.7 coalescing 约束

当前 gateway backlog coalescing 会拼 PCM 并主要保留首块 meta。引入 cursor 后：

- coalescing 只能发生在分配 `delivery_seq`、生成最终 header 和写入 CommitLedger **之前**；
- 一旦 logical delivery 发布到 resumable ledger，其 header、binary、sample boundary 和 sequence 永久不变，
  gateway 不得再次跨 ledger record coalesce；
- 不能跨 `safety_class`、`commit_seq`、`segment_id`、`attempt lineage` 或任何 replayable event barrier
  合并；`attempt_id` 是 segment-local，不能只比较相同数值；
- 合并后必须更新 `end_sample` 和完整 `source_spans/edit_ops`；
- resume 模式严格保持已发布的 original logical delivery 边界；`source_spans` 只用于诊断/映射，不能
  代替单 header/单 sequence 的 ACK 边界；
- terminal/event 永远不能越过此前 committed audio；
- header 与 binary 仍是一个可重放的不可分 logical delivery。

### 6.8 为什么外部没有 rollback 消息

本设计从不发送 `[SB_source, N_source)`，所以正常 wire 上不应出现：

```text
server -> client: rollback / revoke / replace
```

只有未来实验“客户端缓存 tentative audio”时才需要 `revoke_after_sample`。那要求播放器承诺
指定 sample 尚未播放且能从设备 buffer 删除，不能与普通 PCM 客户端混用。

### 6.9 物理长连接生命周期与 keepalive

物理 attachment 与逻辑 resumable session 仍是两条正交状态轴；下面把关键联动并列展示，
`TERMINAL_RETAINED` 时物理 socket 可以已经关闭：

```text
IDLE --start--> ACTIVE_INPUT --stop/end--> DRAINING_OUTPUT --done--> TERMINAL_SENT
                         \--cancel---------------------------> TERMINAL_SENT/CLOSE
                         \--recoverable engine error---------> TERMINAL_RETAINED
                         \--fatal protocol/identity error----> HARD_CLOSE

TERMINAL_SENT --terminal_ack (resumable) / reusable non-resume terminal--> IDLE
              \--TTL expiry---------------------------------------------> CLOSE

TERMINAL_RETAINED --terminal_ack / TTL--> CLOSED
```

- `ACTIVE_INPUT/DRAINING_OUTPUT` 期间拒绝第二个 `start`；
- resumable text 继续使用严格递增 `seq_no` 和累计 `text_ack`：同序号同内容重传幂等，冲突或缺口报错；
- `stop/end` 携最终文本序号并保持幂等：同 final seq 重复不再次触发 EOS/finalize，冲突 final seq 报错；
- `cancel` 在 `ACTIVE_INPUT` 和 `DRAINING_OUTPUT` 都有效且幂等；
- review-gated cancel 的 final cursor 固定为收到 cancel 时已经提交的 `SB_output`，先 drain 这部分 committed
  backlog，再发送 `code=cancelled` terminal；断线时仍可 resume/replay 到相同 final cursor；
- cancel 与 commit publish 由同一 coordinator lock/sequence 串行化；final `SB_output` 取 cancel 的
  linearization point，不能与半完成 commit 竞态；
- review/drain timeout、engine error、verifier error 和 retry-exhausted 使用同一个 terminal
  linearization 机制；一旦 terminal_pending 被发布，不允许新的 commit batch 越过 final cursor；
- 同一机制还必须拒绝新的 `begin_child_attempt`，并在固化 terminal 统计前 fence 所有
  `PREPARING/ACTIVE` child；`activate_child_attempt` 只能在 session fence 内 CAS，因此 terminal 后不能
  恢复生成；
- `stop/end` 只关闭输入，必须等 speculative 尾被审核、替换或丢弃后才发送 `done`；
- input 关闭后，`OutputAck/PlaybackProgress` 仍有效，直到 terminal 完成；
- resumable session 发送 terminal 后仍保留 terminal delivery、最终 cursor 和 TTL；只有收到
  `terminal_ack` 后才能释放 lease 并回到 `IDLE`。TTL 过期则关闭/清理，不能直接复用同一 attachment；
- 非 resumable 的成功/取消 terminal 只有在明确 `connection_reusable=true` 时回到 `IDLE`；
- 可形成稳定 terminal 的 engine/verifier error 先把 error terminal 写入 resumable ledger 并保留到
  `terminal_ack/TTL`；物理 socket 可以随后关闭，但逻辑 session/terminal 仍可 resume；
- 破坏身份、cursor 或 framing、无法可信继续 ledger 的 fatal protocol error 才 hard-close 并作废 resume，
  避免已排队旧消息污染下一 session；
- transport ping/pong 只判断链路存活，不推进 `P/CB/SB`，也不延长 synthesis attempt 的审核超时；
- 无活动 session 的 idle timeout 只关闭物理 socket，不影响已完成 session；
- 长 session 的绝对 RPC/session deadline、input-idle、review/drain timeout、resume grace 是四类独立
  计时器和 reason code；input 已关闭后不再用 input-idle 终止正常 drain；
- text 输入、committed output、speculative bytes 和 ACK lag 分别限流，不能用一个总队列隐藏瓶颈。

断开进入 `DETACHED_GRACE` 后不再把 socket keepalive 当成 session 活性；由 resume grace 管理状态寿命。
若客户端只想长期复用连接而不需要 resume，可以在 terminal 后保持 `IDLE`，但不能保留上个 session 的
tentative 或 output ledger。

若产品需要“立刻停声并丢掉服务端尚未写 socket 的 committed backlog”，必须另定义
`abort_transport/purge_and_close`：它直接关闭 attachment、禁止 resume/reuse，并只保证已写入链路的字节
可能继续到达。它不是正常 `cancel`，不能发送一个带 delivery gap 的可复用 terminal。`guarded_legacy`
可保留旧 purge 行为，但必须通过 profile 区分。

## 7. 审查器设计

### 7.1 统一 contract

所有 reviewer 都必须：

- 以 `segment_id + attempt_id + source frame` 为身份；
- 返回连续前缀水位，不返回无上下文的单帧布尔值；
- `safe_through` 单调，且以后不能 reject 已经声明 safe 的 frame；
- 若需要未来帧判断，就把尾巴保持 pending；
- reject 时返回异常**起点**而非检测触发点；
- `safe_through` 同时冻结该前缀的 edit plan/version；后续不能修改已提交的 trim/drop 决策；
- EOS 时显式 finalize；
- 能保存/恢复自己的 attempt-local 状态。

聚合规则：

```text
M = min(M_i for i in required_reviewers)
```

optional reviewer 默认不阻塞 `M`；若产品允许它发 reject，必须明确是 hard-reject 还是仅 warning，
不能在运行时隐式变化。

reviewer 内部可以使用更细粒度时间轴，例如 VAD 的 16ms frame，但 commit/checkpoint/rollback 第一版
统一落在 80ms codec 边界：安全终点向下取整到已完整覆盖的 codec frame，异常起点向前取整到包含该
sample 的 codec frame。sample-level 前缀/尾部裁剪保存在 edit map 中，不把 Talker 回滚到半个 codec frame。

### 7.2 PCM health reviewer

这是唯一允许在首包提交前同步阻断的 reviewer，检查成本必须是常数级：

- byte alignment 和 frame 长度；
- NaN/Inf；
- 超过硬 corruption limit 的明显非法幅值；
- 可选 clipping ratio、DC offset、突发不连续性。

全零/全静音不是 PCM 格式错误。只要数值和格式健康，PCM health 自己的 `safe_through` 就照常越过该
frame，同时旁路发出不参与聚合的 `SilenceObservation(frame, energy, all_zero)`；只有 VAD 拥有静音
pending、`DROP_PREFIX/DROP_TAIL` edit、EOS 全静音 retry 和 timeout 的决策权。这样 required
`PCM_HEALTH` 不会把 VAD 已经裁剪或放行的静音 prefix 永久卡在 `M=min(M_i)` 之外。NaN/Inf 或格式损坏
永远不能 bypass。高能随机噪声不能只靠幅值阈值判断，否则会误伤正常爆破音；
需要单独分类器或后续 ASR/VAD 证据。

### 7.3 VAD reviewer

当前 [`engine/interface/vad.py`](../../../engine/interface/vad.py) 应从 gateway output gate
重构为 source-frame reviewer，并把“审核”和“编辑”分开：

- `PREFIX_SILENCE_DROP`：首个可听样本之前的静音直接从输出映射中删除；
- `SHORT_PAUSE_SAFE`：正常词间/句间短停顿经迟滞后可提交；
- `TAIL_SILENCE_PENDING`：尾部静音在 EOS/text progress 未确认前保持 pending；
- `PAD_SILENCE_REJECT`：只有 backend 明确进入异常 padding 状态，或 semantic progress 证明正文未完成时，
  长静音才返回 run 起点；仅凭“输入流尚未 end”不能区分自然停顿；
- `FINAL_SILENCE_TRIM`：文本已确认完成且 EOS，裁尾而非重合成；
- `ALL_SILENCE_REJECT`：整个 attempt 未出现语音，换轨迹重试。

VAD 判断“有没有语音”，不能证明语音内容正确，也不能可靠区分语音和所有噪声。

`rules-v1` 中 VAD 必须采用唯一、可执行的 safe-through 状态机，不能只实现上面的 reason 标签：

```text
PREFIX_SILENCE:
  低分/全零 frame -> 追加 DROP_PREFIX edit，保持 pending
  累计超过 max_prefix_silence_ms -> reject/retry
  连续 begin_count 个高分 subframe -> 冻结 prefix edit；完整覆盖的 onset codec frame 可 policy-pass，进入 SPEECH

SPEECH:
  高分 subframe -> 当前已完整覆盖的 codec frame 可 policy-pass
  首个低分 subframe -> 记录 silence_run_start，进入 MAYBE_SILENCE；从 run_start 起 pending

MAYBE_SILENCE:
  在 review_timeout 前恢复高分 -> 整个 silence run 作为正常 pause policy-pass，回到 SPEECH
  达到 tail_pending_ms/end_count -> 只标记 TAIL_CANDIDATE，仍 pending，不自行 reject/放行
  backend 明确异常 pad -> 从 run_start reject
  semantic progress 已完成且 engine EOS -> DROP_TAIL edit + final-safe
  没有 pad/progress/EOS 证据直到 review timeout -> 走 profile 的 stall/error，VAD 不猜测

EOS:
  从未出现 speech -> ALL_SILENCE reject/retry
  有 speech 且尾部 pending -> 冻结 DROP_TAIL edit 后 final-safe
  有 speech且无 pending -> final-safe through N
```

`begin_count/tail_pending_ms/review_timeout` 都在 StartAck 的 profile/version 中固定。长自然停顿只要在
timeout 前恢复语音，就整体成为正常 pause；超过 timeout 的行为由 profile 决定。这样 voiced frame
何时推进 `M_vad`、silent run 何时解锁和无证据尾部如何终止都有唯一语义。

### 7.4 codec 重复 reviewer

第一版保持当前已验证规则：codebook-0 同 token 连续 4 帧视为周期 1 循环。状态包括：

```text
run_token
run_start_frame
run_length
last_safe_frame
```

算法：

- run 未达到阈值时，只让 run 起点之后 pending；
- token 改变时，前一个短 run 变安全，水位推进到新 token 的起始边界；新 token 自己作为下一条
  未决 run 的第一帧，不能在同一步被声明 safe；
- run 达到阈值时，从 `run_start_frame` reject；
- EOS 时未达到阈值的短 run final-safe；
- rollback 时恢复到边界对应的 run 状态。

这意味着重复 reviewer 的审核尾巴是动态的，不是所有首包固定等待 4 帧。

周期 2～8、codec span 重放、PCM 自相关重复可以后续加入，但在跨 speaker、多语言、正常拖音数据
校准前不作为 required reviewer。当前“正常最长 run=3”的结论来自有限探测集，阈值不能未经扩展
验证就收紧。

### 7.5 当前 reviewer 组合能宣称什么

当 required reviewers 只有 `pcm_health + vad + codec_repeat` 时，header 中的 `POLICY_PASSED` 意味着：

> 音频通过格式/数值健康检查，没有命中已配置的静音规则和周期 1 codec 重复规则。

它**不意味着**“音频内容与原始文本一致”。协议 capability、指标和产品文案都必须保留这个边界。

## 8. 轻量 ASR / codec-text 播放进度条

### 8.1 可行性结论

可以直接从生成 codec 做轻量内容审核，这是很值得优先验证的工程方向；但 Qwen 官方尚未报告
12Hz codec 的 ASR/文本对齐结果，因此目前不能断言它一定优于 waveform ASR，更不能把离散 codec ID
零训练接到 CTC 就得到文字。最终主路径必须由两者的 shadow 指标决定。

Qwen3-TTS 官方报告说明 12Hz tokenizer 实际为 12.5Hz，即每帧约 80ms；第一层 codebook
由 WavLM 教师引导，偏语义，后 15 层补充声学和韵律，codec encoder/decoder 全因果。
这使 codebook-0 很适合做轻量对齐输入，但 WavLM 蒸馏不提供 codec ID 到文字 token 的映射，
仍需有监督或弱监督训练一个对齐头。

参考：[Qwen3-TTS Technical Report](https://arxiv.org/html/2601.15621)、
[WavLM](https://arxiv.org/abs/2110.13900)、
[CTC 原始论文](https://www.cs.toronto.edu/~graves/icml_2006.pdf)。

### 8.2 产品形态：不是自由转写

推荐把它定义为 **codec-text 单调进度验证器**：

```text
生成 codec0 帧 -> 小型因果 audio encoder -> a_t
                                             |
原始文本流 -> 规范化/BPE + text encoder -> e_u
                                             |
                 target-conditioned monotonic DP
                                             |
         stable_text_offset + safe_frame M + mismatch score
```

输出不是一句开放词表转写，而是：

- 当前最可能说到原始文本的哪个 source offset；
- 哪个音频 frame 之前的对齐已稳定；
- 音频是否更像 `other/garbage` 而非原始文本；
- 异常最早可能从哪个 frame 开始。

这正好是“文本播放进度条”，并可把确认到的词/短语对应音频尾直接作为新的 `M`。

### 8.3 推荐模型

第一版建议：

- 输入只用实际生成的 codebook-0 ID；
- codec embedding 128～256 维；
- 2～4 层 causal TCN 或 causal/chunked 小 Conformer；任何 right-context 都计入 review lag；
- 文本侧用紧凑 BPE，并保留 BPE 到原始字符 offset 的映射；
- 每帧只计算当前进度附近 32～64 个 source position；
- 状态包含 `stay/advance/blank/other`，必要时加高代价 insert/delete/substitute；
- 规模先从约 1～5M 参数开始，以真实并发 benchmark 调整；
- 推理保存上一帧 DP state，复杂度约 `O(active_text_window)`。

不要把 Talker hidden state 作为唯一音频输入。它本来就被原始文本条件化，容易让审核器看到
“模型准备说什么”而不是“实际生成了什么”。codec reviewer 只能验证 Talker 实际生成的 codec；
若同一份合法 codec 在 Code2Wav/TRT 路径被解成噪声或错误波形，codec reviewer 完全看不到。
PCM health/VAD 也抓不住所有高能随机噪声，因此生产上仍需保留 waveform-domain noise/intelligibility
classifier 或 waveform ASR 作为纵深防御，是否启用由 shadow 数据决定。

### 8.4 为什么不直接做标准开放词表 CTC

12.5Hz 时间分辨率很低。标准 CTC 对长度为 `U` 的 target，若有 `A` 对相邻相同 label，最少需要：

```text
T_min = U + A
```

因为相邻相同 label 之间必须插入 blank。若把字符、UTF-8 byte 或细粒度音素都当成 label，快速语音
可能出现 `T < T_min`，CTC 路径根本不存在。

因此应先统计每种语言：

```text
target units / codec frames
standard CTC path infeasible rate
```

更稳妥的做法是：

- 先使用更紧凑的 BPE，使绝大多数样本满足 frame budget；
- source-position-specific emission 可以区分两个相同 token 的位置，但只有放弃标准 CTC collapse
  语义、明确采用位置条件 topology 时，才不需要相同 label 间的 blank；仅重命名 trellis state 不够；
- 普通 forward-sum 仍要求每个非 epsilon 正文位置至少占一帧，不能天然解决 `U > T`；
- 更粗单位后仍不满足时，采用同一 acoustic step 可发多个 label 的 RNN-T/显式单调 transducer；
- epsilon 只用于标点、控制符等确实不发音单位，不能用它跳过正文来制造“审核通过”。

文本条件在线对齐方面，CTCAT 可作为“目标文本条件 + 流式 CTC 对齐更新”的研究依据；
One-TTS 可作为“无需帧级标注训练 forward-sum 单调对齐”的依据；CTC Segmentation 则是已知文本的
离线分段/forced-alignment 参考，不是流式实现证据。它们都不能直接移植到 12.5Hz 多语言长文本。

参考：[CTC-aligned Audio-Text Embedding](https://www.isca-archive.org/interspeech_2024/jin24d_interspeech.pdf)、
[One TTS Alignment To Rule Them All](https://arxiv.org/abs/2108.10447)、
[CTC Segmentation](https://arxiv.org/abs/2007.09127)。

### 8.5 forced alignment 不能单独充当安全审核

只给定原文做 forced alignment，DP 即使面对无关语音也会努力找一条路径。因此必须同时存在竞争路径：

- `match`：沿原始文本单调前进；
- `blank/silence`：不推进文本；
- `other/garbage`：有语音但不属于当前文本；
- 可选编辑路径：容忍正常读音变化，但施加高代价和次数上限。

每帧至少维护：

```text
best_source_position
position_posterior
reference_path_logprob
garbage_path_logprob
reference_margin
last_stable_audio_frame
last_stable_text_offset
```

安全推进条件示例：

```text
source position 与音频终点连续稳定 K 帧
AND reference_margin >= calibrated_threshold
AND other/garbage posterior 低于阈值
AND 其他 required reviewers 不 pending
=> safe_through_frame = M
```

异常条件包括：

- `other` 连续高后验；
- reference 相对 garbage/unconstrained 路径的 margin 持续过低；
- VAD 判定有语音但文本进度长时间不前进；
- 只能通过大量删除/插入才能继续；
- 已确认文本之后又出现先前文本片段，形成内容重复。

`other/garbage`、正文 insert/substitute 或低 margin 一旦在尚未提交区形成 hard-reject，必须锁存最早
可疑起点 `H`；不能因为后续音频重新匹配，就遗忘污染并让 `M` 跨过去。少量 edit 若只用于容忍 ASR
误差，经过该 edit 的路径不能产出最高语义 safety class；正常多读法应建进 pronunciation lattice，
而不是靠通用 edit 路径掩盖。

语义 reviewer 最好只在稳定 BPE/词端点推进 `M`，使水位按词或短语跳跃。

### 8.6 文本流处理

- 进入 reviewer 的文本必须使用与 TTS 一致的 canonical normalization，并保存 source offset map；
- 数字、缩写、多音字和语言混合可使用多读法 lattice，避免把正常读法当异常；
- 保存 raw-text append journal；流式 BPE 的新字符可能让旧尾部重新分词，因此只冻结 tokenizer
  stability boundary 之前的 target states，并保留一个可重分词 mutable suffix；
- 新文本到达时重建 mutable suffix，再把稳定新增部分 append 到 target graph，保留冻结前缀的 DP state；
- 标点和不发音控制符允许显式 epsilon 跳转；正文删除必须高惩罚；
- `text_idx` 只是模型 conditioning cursor，不能直接替代 ASR progress。

### 8.7 数据和训练

正样本：

- 真实干净语音 + 精确文本，离线编码成 Qwen codec；
- 经高质量 teacher 或人工过滤、确认内容正确的 Qwen-TTS 生成 codec + 原始文本；未经审查的生成结果
  不能直接当正样本，否则会训练 reviewer 接受幻觉；
- 覆盖语言、speaker、语速、数字、缩写、代码混读和长文本。

负样本是审核能力的关键：

- 同 speaker 的错配文本；
- 插入、删除、替换、重排、重复文本；
- codec0 连续重复和 codec span 重放；
- 静音、噪声、音乐；
- 真实线上 hallucination attempt；
- 同音、近音和一两个字差异的 hard negative。

数据和验证集必须显式区分两个域：

- 真实语音经 tokenizer encoder 得到的 codec；
- Talker 实际生成出的 codec。

至少保留一份独立的“实际生成域”验证集，不能只在重建真实语音上报告结果。噪声/音乐经 encoder 得到
的 codec 负样本只能覆盖“模型生成异常 codec”的情况；它不能覆盖“同一合法 codec 被 Code2Wav/TRT
解坏”的情况，后者必须在 waveform-domain 故障注入集上评价。

不需要大规模逐帧标注；forward-sum 可用 audio-text pair 训练。另做一小批高质量 word timestamp
数据，用于评价 `M/H` 定位误差和稳定延迟。

### 8.8 外部轻量 ASR 如何接入

外部 ASR 也必须位于 CommitCoordinator 之前，消费 tentative PCM：

```text
AttemptBuffer PCM
  -> causal resample / feature state
  -> per-session streaming ASR decoder
  -> partial hypothesis + token timestamps + confidence
  -> 与 canonical source 的带状单调匹配
  -> candidate_M / candidate_H / revision history
```

- 第一版作为 in-process CPU/ONNX worker pool 或同机 sidecar；队列按 session/attempt 有界；
- 每个 session 保留 resampler、feature、decoder 和 partial-stability state；
- attempt rollback 时必须恢复 reviewer checkpoint，或从 retained PCM context 重放到 `R`；
- partial hypothesis 允许改写，只把连续稳定若干 chunk 的 source prefix 作为 candidate；
- shadow timeout 只记缺失，不能影响提交；成为 required 后，timeout/abstain 必须冻结 `M` 并进入明确
  的 stall/underflow policy，不能被 optional fallback 静默跳过；
- bootstrap frame 仍进入同一个 ASR/semantic state；bypass 只推进 `SB_source`，不能伪造
  `M_source=SB_source`。后续路径必须继续审核首包，不能跳过其中的半个词。

外部 ASR 不需要先做到高质量开放转写；其 partial tokens 只需对齐已知原文。但计算 RTF 很低不等于
stable-token latency 很低，chunk/right-context、partial revision 和文本正规化误差必须单独测量。

### 8.9 引入步骤

1. Phase 0/1 即可并行接入轻量 waveform streaming ASR shadow，建立文本进度和误报基线；不参与提交。
2. 保存带置信度和 revision history 的 `candidate_M/H` 作为 pseudo-label；生产验收仍使用人工/高质量
   时间戳子集，不能把 shadow ASR 当事实 teacher。
3. 训练 `codec0 embedding + linear/TCN alignment` 下界模型。
4. 对比 codebook-0、16 codebook embedding sum、codebook-0 + 少量残差层。
5. codec reviewer 继续 shadow，校准 false accept、false rollback 和稳定延迟。
6. 先只在高置信词边界推进可观测 `M_semantic`，仍不 gate。
7. 只有 codec reviewer 在实际生成域的安全和延迟指标优于/补充 waveform ASR 时，才选择它作为
   最终主路径；也可以长期保留两者做纵深防御。
8. 达到验收阈值后成为 required reviewer；低置信时 abstain 并冻结 `M`，而不是强行给安全结论。

### 8.10 关键指标

不能只看 WER：

- unrelated speech / silence / noise / repeat / omission 的 false accept；
- 正常语音每小时 false rollback；
- stable `M` 相对人工/teacher word endpoint 的 p50/p95 滞后；
- reject 起点 `H` 的定位误差；
- abstain 覆盖率；
- progress source offset 误差和是否单调；
- 已提交音频时长上的 false-accept rate，以及首次错误 span 被提交的毫秒数；
- 单 session 状态内存、CPU/GPU 占用和并发吞吐。

## 9. 首帧特殊处理

首帧路径的目标是最短**可听**首响，不是最短发送一个静音包。

### 9.1 推荐顺序

```text
first produced PCM
  -> 同步格式/NaN/Inf 检查
  -> sample-level prefix silence trim
  -> 若整帧静音，丢弃并检查下一生成帧；超过 prefix cap 则 reject/retry
  -> 第一段非静音 remainder 立即 BOOTSTRAP_BYPASS commit
  -> reviewer 正常消费后续 frame，不等待 4 帧
```

同步路径不得等待：

- 重复模式达到 4 帧；
- VAD 多帧 end hysteresis；
- ASR stable token；
- segment EOS。

### 9.2 bypass 范围

- 只允许一个最小**可听输出** bypass，输出硬上限为 1 个完整 codec frame（当前约 80ms）；
- 在它之前被完整 drop 的静音 source frame 以零输出 `edit_ops` 一起推进 source cursor，因此一个
  bootstrap commit 的 source span 可以超过 1 frame，但未审核、真正暴露给客户端的 PCM 仍不得超过
  `bootstrap_max_ms`，且静音搜索总跨度受 `max_prefix_silence_ms` 限制；
- 必须先完整生成该 codec frame 和 `checkpoint_after`，再把裁剪后的整个 remainder 作为一个
  `BOOTSTRAP_BYPASS` commit batch；wire 可以把它切成首个 20～40ms delivery 加后续 delivery 来缩短
  packet 首发，但 batch 内全部 PCM 都已 committed，不能把半个 source frame 留作 tentative；
- NaN/Inf/格式损坏不能 bypass；
- metadata 必须标记 `BOOTSTRAP_BYPASS`；
- 每 session 只能成功使用一次；全静音 frame 被 drop、非法 attempt 被 reject 或 commit 事务失败时尚未
  消耗特权，retry 仍可获得真正的首个可听包；第一次成功 bootstrap commit 后，后续 retry 不再获得；
- 首包后来被 reviewer 判为异常，只能上报 `safety_breach_after_commit`，不能伪装成已替换。

上面的 `checkpoint_after` 是完整目标契约。Phase 0/1 若尚未具备最小 Talker+C2W checkpoint，只能把
首包作为协商过的非可恢复迁移例外：StartAck 回显 `bootstrap_restorable=false`、
`exact_rollback_v1=false`，后续任何需要回到该边界的异常都 fail-stop。它不能对外声称“首包之后可替换”；
要实现完整承诺，必须把首帧边界最小 checkpoint 前移，或完成 Phase 2。

### 9.3 前缀静音

前缀裁剪应是 sample 映射 edit，不要求等待完整 VAD 审核窗口。可采用快速能量门限 + 极短迟滞，
保留少量 pre-roll 防止切掉爆破音。若整个首帧是静音，继续找第一个有声 remainder；发送静音不会改善
用户感知首响，反而会占据不可撤销区域。

所有被删除的 prefix sample 都写入 attempt-fenced、版本化的 `edit_ops`，即使对应 output 长度为 0；
这样 source frame、checkpoint 和 output sample cursor 仍有唯一映射。累计前缀静音达到
`max_prefix_silence_ms` 后必须 reject/retry 或 terminal，不能无限寻找首响。

第一阶段需同时保留当前 energy/TenVAD 路径做 A/B，校准不同 speaker、语言和低音量语音的误切率。

## 10. 重合成与状态回滚

### 10.1 三种策略比较

| 策略 | 上下文/韵律 | 文本风险 | 实现成本 | 用途 |
|---|---|---|---|---|
| 精确 Talker + Code2Wav 状态回滚 | 最完整 | 最低 | 最高，需要 checkpoint/replay | **常规主路径** |
| 安全 codec suffix + 对齐文本做 ICL | 保留部分局部韵律 | 依赖可靠文本边界 | 中等，需要重新 prefill | 二级 fallback |
| 从剩余文本 fresh segment | 韵律最容易重置 | 边界猜错会重复/漏字 | 最简单 | 最终止损 |

### 10.2 对用户示例的回答

假设 `c4～c5` 异常，而文本“3”还没有在波形中完整说完。不能仅看波形就断言模型还没消费文本 3。
当前生成循环通常每步会消费 trailing text embedding；“条件已经进入 KV”和“声音已经说完”不是一回事。

主路径应恢复到异常之前的精确状态，例如：

```text
恢复 state_after(c3)
恢复当时的 text cursor / next_embed / last_codec_sum
更换 attempt seed
重新采样 c4'、c5' ...
```

而不是：

```text
保留旧 KV，再人为喂一次 3     # 可能双重条件化、重复发音
```

也不是优先：

```text
把 c2、c3 两个极短 frame 包装成新 ICL reference
```

短 codec 尾巴不是原自回归状态，且很可能落在音素/字的中间，无法可靠配对 reference text。

### 10.3 checkpoint 的精确定义

边界 `b` 的 checkpoint 定义为：

> 已生成 frame `b-1`，尚未生成 frame `b` 的完整状态。

这样 rollback 到 `b` 后，新 attempt 从 frame `b` 开始替换，不改变 `[0, b)`。

### 10.4 必须保存的状态

#### Talker

- `past_len`；
- 边界处精确 `next_embed`；
- 流式停顿时的 `last_codec_sum` 与 `next_embed=None`；
- `text_idx`、trailing/token queue 视图；
- `token_counts`；
- sampling RNG state，或可复现的 attempt seed/counter；
- pad 起点、连续静音计数、loop 状态；
- 精确 trailing embedding/journal，不能因恢复时 batch shape 不同而随意重新计算。

Talker KV pool 按 position 追加。只要 rollback 期间 slot 保持 pinned，就不需要为每个 checkpoint
复制整份 KV：恢复 `past_len`，忽略并覆盖其后的旧列即可。这个结论只适用于 preallocated pooled
append-only 路径；non-pooled 路径若从 tensor shape 推导 past length，恢复时必须把 `slot.talker_kv`
显式 slice 到 `restored_past_len`，或直接通过 capability 禁用 exact rollback，不能只改一个标量。

客户端已经提交的 canonical text log、后来到达的 `APPEND_TOKENS` 和输入关闭/EOF 是不可逆外部状态，
不属于生成 checkpoint，不能随模型状态倒退。可回滚的只有模型 text-consumption cursor。恢复后要把
checkpoint 之后到达的文本/EOF 重新接到该 cursor：特别是边界为 `next_embed=None`、
`last_codec_sum!=None` 时，如果新文本已经到达，应立即用保存的 `last_codec_sum + 下一条未消费的精确
text embedding` 恢复，而不是再次进入等待。

#### Code2Wav

- sliding-window KV 和 `c2w_len`；
- 17 个 conv recurrent states；
- 4 个 transconv overlap states；
- logical source frame 与 Code2Wav `absolute_frame_idx`；
- ping-pong buffer 的当前读写相位，或恢复时规范化相位；
- `c2w_len`、right-aligned KV layout、dummy-past/valid-mask 语义和精确 dtype/layout；
- 完整 16 codebook frame log。

这里是实现难点。当前 C2W KV 每步会原地左移，conv/transconv 也参与下一帧波形生成，只恢复
Talker KV 会造成波形不连续、点击或音色漂移。

#### 审核与身份

- `segment_id + attempt_id + logical_frame_idx`；
- 每个 reviewer 的 pending/run/DP 状态；
- source frame 到 native/output sample 的 edit map；
- checkpoint 与 commit record 的关联。

### 10.5 checkpoint 分层

不能每帧复制整份 Talker KV 和 C2W 重状态。推荐：

1. 每 80ms 保存轻 journal：标量、text cursor、next embedding、RNG/counter、完整 codec IDs、review state。
2. Talker KV 不复制，只保存 `past_len`。
3. Code2Wav 保存稀疏重 anchor，并通过完整 codec log 重放到目标边界。
4. anchor 必须拥有独立 storage 或 versioned arena，不能保存会在下一步被原地覆盖的 pool/arena view。
5. capture 必须在本步 Talker KV、C2W KV、17 conv 和 4 transconv 更新全部完成后才发布；发布前
   `checkpoint_after` 不是 `Restorable`。
6. checkpoint materialization 以 `(slot_allocation_epoch, attempt_id, frame)` 做 CAS；旧 attempt 的迟到
   capture/replay 不能推进 `K(M)`。
7. 可以每 4 帧建立重 anchor，但任意需要提交的 `M` 都应能由 anchor + journal 先精确物化；若第一版
   只能提交到重 anchor，就必须证明下一 anchor 的生成+审核+网络时间小于客户端剩余可播放 buffer，
   否则 4×80ms 粒度会重新制造下溢。

音频 frame 可以在 checkpoint capture 完成前进入 reviewer，但 CommitCoordinator 必须被
`Restorable(frame)=false` 挡住；当前 deferred C2W scatter 尚未完成时，绝不能因为 PCM 已产生就提交。

按当前默认结构估算，Talker 最大 KV 每 slot 是几十 MiB，不能按 checkpoint 复制；一份 C2W KV
加 recurrent state 是数 MiB 量级。确切数字必须由启动时按实际 engine shape/dtype 打印并纳入容量规划。
还要设置全局 `max_checkpoint_bytes`：例如 2s speculative、每 4 帧一个约数 MiB anchor、128 路并发时，
总量可能达到 GiB 级，不能只配置 per-session frame 间隔。

仓库现有 reference Code2Wav warmup 已具备“完整 codec -> C2W state”的基础路径，可抽出 live replay
primitive。但当前 standalone warmup 和 fused 单帧路径必须先做状态与下一帧波形 parity；若 chunk shape
不同导致不一致，应导出专用 T=1 replay engine，不能近似恢复上线。live replay 必须从 anchor 的非零
`absolute_frame_idx`、right-aligned cache 和 valid mask 继续，不能像 reference warmup 一样假设从零开始；
它还需要独立 TensorRT execution context/stream，或在 engine thread 内串行使用 context，不能并发复用
同一个 execution context。

### 10.6 精确 rollback 算法

发现最早可疑 frame `H`：

1. 立即标记 `rollback_pending` 并冻结该 lane 的下一次 launch；
2. 选择满足 `SB_source <= R <= H` 的最后安全且可恢复边界；
3. 若 `[SB_source, R)` 已审核，可使用被锁存的 parent attempt 身份原子提交并令 `SB_source=R`；
4. fence parent attempt generation，使当前 in-flight future 成为旧身份；等待该 GPU step 完成，但在
   state scatter 前过滤/覆盖其旧 attempt 输出；
5. 丢弃 parent `[R, N)` 的 PCM/codec/review 结果；调用统一的
   `begin_child_attempt(parent, fork=R, strategy=EXACT)` 并按共享 tagged-result 表处理；只有
   `STARTED(child)` 才能开始 restore/replay；
6. slot 保持 pinned，在 `PREPARING` child identity 下恢复 Talker `past_len` 和轻状态；
7. 在同一 child identity 下恢复最近 C2W anchor，重放 codec 到 `R`；所有 replay future 都捕获
   child `attempt_id/review_epoch_id/attempt_ordinal`；
8. 在 child 下恢复 reviewer 上下文；保留不可逆 canonical text/EOF log，只恢复 consumption cursor，并重新接入
   checkpoint 之后到达的精确 trailing embeddings；
9. 完整状态和 reviewer base 就绪后调用 `activate_child_attempt(child)` 并按共享 tagged-result 表处理；
10. 只有 `ACTIVATED` 才以新 seed/counter 从 `R` 生成 child suffix；新 attempt 重新经过 reviewer，外部
    客户端通常看不到 retry。

当前 engine 在 GPU 运行期间还会 drain inbox，因此 rollback 请求只能先设置 pending；必须在
`GPUFuture.wait()` 后的安全 step boundary 应用，不能在 future 仍引用旧 slot 时释放或覆盖状态。

`rollback_pending` 一旦设置就要立即冻结该 lane 的下一次 launch。每次 GPU launch 还必须捕获不可变
身份，而不是只保存可变 `SlotKVState` 引用：

```text
session_execution_id
slot_id
slot_allocation_epoch
segment_id
attempt_id
attempt_ordinal
review_epoch_id
```

future 完成时先按这组 launch identity 验证每个 lane。旧 lane 输出必须在 Talker KV scatter、C2W
scatter、`token_counts/RNG/next_embed` 更新**之前**被过滤；如果执行器无法逐 lane 提前过滤，则只能
允许旧结果写入临时 state，随后在同一个 barrier 内用 checkpoint 全量覆盖，不能只在“是否发 PCM”处
丢弃。slot allocation epoch 防止释放后同一 slot ID 被另一 session 复用时遭到旧 future 污染。

新 seed 建议至少绑定：

```text
H(base_seed, request_id, segment_id, attempt_id, rollback_frame)
```

若 `do_sample=false`，精确恢复会再次走回相同异常轨迹。此时要么允许 retry 时临时采样/参数扰动，
要么在仍有 attempt budget 时进入 ICL/fresh fallback，不能无限重复确定性输出。改变采样模式/参数必须在 start 的
retry policy 中明确协商；不能在客户端要求确定性时静默切换。

若 audible `H` 只是症状出现点，而更早 KV 已进入坏轨迹，新 seed 在同一 `R` 仍可能失败。重试策略可
逐级把 `R` 扩展到更早的 safe+restorable anchor，最早不越过 `SB_source`；整个 segment 都在 SB 后时可
退到 segment prefill。达到 expansion cap 后，只在 `attempt_ordinal < max_attempts_per_segment` 时进入
合法 fallback；attempt budget 已耗尽则 error，不能无界向前重算或借 fallback 绕过上限。

### 10.7 ICL fallback

只有满足以下条件才使用：

- loaded model/task 明确支持 ICL，capability 返回 `icl_available=true`；CustomVoice/VoiceDesign 等路径
  不能假设可在同一个已加载 engine 上切换成 ICL；
- 能把 policy-passed codec 转为该 ICL 路径要求的 codec-sum/reference state；
- exact checkpoint 缺失或 C2W parity 不可信；
- semantic reviewer/外部 ASR 给出稳定、完整的词或短语边界；
- 有一段 policy-passed codec 和精确对应的 reference text；
- 无副作用的 budget 预检表明仍有 ordinal；真正执行任何 ICL prefill/state 工作前必须调用
  `begin_child_attempt(..., strategy=ICL)`，成功后全部 prefill、GPU future 和 reviewer state 挂到返回的
  `PREPARING` handle；所有非 `STARTED` 结果按共享 tagged-result 表处理，不得启动。

推荐取最近约 0.8～1.5 秒 policy-passed codec 作为局部 reference，再从最后确认完整的文本边界之后
生成 target。具体长度需按语言、speaker 和 ICL 质量实验调整。2～3 个 codec frame（160～240ms）
通常太短，不应作为默认。

这些历史 codec 已位于 `SB_source` 左侧，不能跟 speculative journal 一起回收；需要单独维护便宜的
`committed_codec_context_ring`，保留 full codec、source/text 对齐和 safety profile。ring 只用于恢复，
不重新进入 output ledger。

ICL 是一次新 prefill，不等价于旧 Talker KV；它只能尽量继承局部音色/韵律。若没有可靠文本
边界，不能猜“从 3 开始”。

还必须检查文本边界与当前 `SB_source` 的关系：已经提交的音频可能停在一个词/字的中间，
而最后稳定完整词边界位于 `SB` 之前。此时从该完整词重启会重复已提交内容，从下一个词重启又可能
漏掉半个词，ICL/fresh 都不能保证正确。规则 reviewer 阶段只有 exact state rollback 能自然续写这种
中间边界；exact 不可用时应截断/error，而不是伪装成无缝替换。semantic reviewer 上线后，可以让
常规 commit 优先落在稳定文本端点，或保留额外 recovery guard，提升 fallback 可用率。

### 10.8 fresh-segment fallback

状态完全不可恢复时，从最后稳定文本边界重新起段。它最能隔离坏状态，但会重置语速、重音和韵律。
fresh 仍是同一 retry budget owner 下的新 attempt；能力、边界和 budget 先做无副作用预检，随后必须在
任何 fresh prefill/state 工作之前调用 `begin_child_attempt(..., strategy=FRESH)`。只有取得
`PREPARING` handle 才能继续，且全部工作挂到该 handle；不能通过换 `segment_id` 重置
`max_attempts_per_segment`。begin/activate 的非成功结果同样按共享 tagged-result 表处理，不能一律改写成
`retry_exhausted`。

fallback target 起点必须能与 `SB_source` 的 committed 文本边界对齐，或明确位于其后且没有遗漏正文；
不能为了“宁可重复”而选择映射到 `SB` 之前的词并再次播放已提交内容。找不到合法边界时直接
truncate/error，由上层决定是否整句重新播放，服务端不能在同一音频流里偷偷重复或漏字。

exact rollback 恢复 C2W overlap 能避免声码器冷启动和 overlap 丢失，并显著降低点击风险，但新 codec
轨迹仍可能在 `R` 产生能量、F0、音素或韵律跳变，不能承诺感知无缝；stitch guard 和拼接质量测试对
exact path 也要保留。ICL/fresh fallback 更无法修改已经提交的左侧波形。若要做双侧 crossfade，
CommitCoordinator 必须额外保留 stitch guard；但在 rollback/commit 只支持 80ms codec 边界的第一版，
不能先发送同一 codec frame 的前 60ms、只留下后 20ms。应让包含 10～20ms waveform guard 的整个 codec
frame 保持 speculative，使 `SB_source` 仍落在前一 frame boundary；否则只能对新右侧做单侧淡入/能量
匹配并接受有限变化。只有未来支持可恢复的 subframe output edit/checkpoint 时，才能缩小 source guard。

### 10.9 特殊情况

- 异常全部位于文本已确认完成后的 pad tail：直接裁尾并 EOS，不必重生成。
- 整个 segment 尚在 `SB` 后：沿用当前整段 re-prefill + 换 seed，成本和实现风险最低。
- `H < SB_source`：错误已提交，不能宣称替换成功；默认 freeze + terminal。只有 StartAck 接受
  `committed_breach_action=ALLOW_CLEAN_RESTART`，且存在独立干净 checkpoint 或合法文本边界 restart
  才能继续，并标记 degraded recovery。
- `SB_source` 落在未完成文本单位中且 exact restore 失败：不得猜测 ICL/fresh 起点，按策略截断/error。
- 当前 segment 已启动的 attempt 达 `max_attempts_per_segment`：丢 tentative；若尚有不新建 attempt 的
  合法终止处理则执行，否则 terminal error，不能用 ICL/fresh 绕过该计数上限。

## 11. 异常处理矩阵

| 异常 | 判定 | 服务端动作 | 客户端可见性 |
|---|---|---|---|
| 非法 PCM / NaN / Inf | tentative 内即时命中 | 从异常 frame reject；绝不 commit；重试或 error | 通常不可见 |
| 首帧前缀静音 | 首个有声 sample 前 | trim/drop，不作为安全重试 | 首响更早可听 |
| 整个 attempt 静音 | 无 speech 且未提交该段 | 有 budget 时整段丢弃、启动下一 attempt；否则 retry-exhausted terminal | 不可见或 error |
| 长静音且文本未完成 | VAD + text progress | 从静音 run 起点 rollback | 通常不可见 |
| 文本完成后的尾静音 | stable progress + EOS | trim tail，不重生成 | 不可见 |
| code0 周期 1 重复 | 连续 run 达阈值 | 从 run 起点 rollback | 通常不可见 |
| 一般周期/内容重复 | future reviewer | 从最早重复 span rollback | 通常不可见 |
| 语音与文本无关 | semantic margin/other 命中 | rollback；仅剩余 attempt budget 时可 ICL/fresh，否则 error | 通常不可见或 warning |
| 异常点已越过 SB | `H < SB_source` | 不伪装 rollback；记录 breach，默认 freeze+error；仅 StartAck 接受 `ALLOW_CLEAN_RESTART` 且有独立干净边界时可 degraded restart | warning/error |
| segment EOS | 仍有 review horizon | reviewer finalize；禁止直接 flush | 无提前尾包 |
| verifier timeout/crash | 无完整 required verdict | fail closed；或显式 bypass policy | error 或带 safety marker |
| retry 耗尽 | 已启动 attempt 数达到 `max_attempts_per_segment` | discard tentative；只能做不启动新 attempt 的合法终止处理，否则 terminal error | error |
| session engine error | live 进程仍可处理终态 | freeze 新 commit、discard tentative；已入 ledger 的 committed audio 保序发送后发唯一 error | error terminal |
| checkpoint 缺失 | 已部分提交且不能 exact restore | 仅有剩余 attempt budget 且能力/文本边界合法时可 ICL/fresh，否则 error；不能整段偷偷重放 | warning/error 视策略 |
| C2W replay parity 失败 | restore 校验失败 | 禁用 exact path；仅有剩余 attempt budget 时 fallback，否则 error | 指标/可能 warning |
| speculative buffer 满且仍有进展路径 | 累计时长达上限，存在可恢复边界或 engine 已 finalized | 暂停该 lane decode，等待 reviewer/commit/finalize | 可选 backpressure warning |
| 不可恢复 segment 撞 speculative cap | segment 未 finalized，且 EOS 前没有 restorable commit/reclaim 路径 | fence、discard tentative；保序 drain 到旧 `SB` 后发 `nonrestorable_speculative_limit` | error terminal |
| commit/replay ledger 满 | committed 未 ACK | 暂停 commit；硬上限后显式失败 | `resume_buffer_exceeded` |
| result/control queue 满 | 安全消息无法入队 | session fail；绝不静默丢 verdict | terminal error |
| 旧 attempt 迟到结果 | attempt 不匹配 | drop，不改任何水位 | 不可见 |
| client ACK 回退 | stale cursor | 幂等忽略 | 不可见 |
| client receipt ACK 超过 SB | `RB > SB_output` 或非 delivery 边界 | 协议错误；不据此放行 | error |
| client cancel | 任意 | 停生成/新 commit，discard tentative；按序 drain 到 cancel 时 `SB_output`，再发固定 final cursor 的 cancelled terminal | cancelled terminal |
| 网络断开，grace 内 | resume enabled | 保留 ledger/attempt；达 cap 后 pause | 重连后 replay |
| resume grace 过期 | detached 超时 | cancel、discard tentative、删 ledger | resume_not_found |
| engine/GPU 进程崩溃 | live state 丢失 | 不能进程内精确 resume | state_lost/error |
| 首包 bypass 后发现异常 | `H < SB_source` 且 safety class=bootstrap | breach；默认 freeze+error；仅接受 `ALLOW_CLEAN_RESTART` 且有独立干净边界时可 degraded restart | 不能替换首包 |
| 播放 underflow | `P` 接近 `CB` 且 `M` 落后 | 按 StartAck 的 STALL / INSERT_SYNTHETIC_GAP / BYPASS_MINIMAL / ERROR 及硬 cap | marker/warning 视配置 |

所有异常路径只能发送一个 terminal。`error` 后不能再发送 `done`，`cancel` 后不能继续 drain tentative。

## 12. 配置建议

迁移期示例：

```yaml
output_policy:
  delivery: review_gated
  bootstrap_policy: bypass_first_audible_packet
  bootstrap_max_ms: 80
  minimum_required_reviewers:
    - pcm_health
    - vad
    - codec_repeat
  semantic_reviewer: shadow
  underflow_policy: stall
  underflow_low_water_ms: 40
  underflow_target_buffer_ms: 160
  underflow_bypass_max_once_ms: 0
  underflow_bypass_max_session_ms: 0
  underflow_bypass_max_per_minute: 0
  underflow_bypass_cooldown_ms: 1000
  underflow_disable_after_retry_ms: 2000
  synthetic_gap_max_once_ms: 0
  synthetic_gap_max_session_ms: 0
  synthetic_gap_max_count: 0
  synthetic_gap_cooldown_ms: 1000
  synthetic_gap_exhausted_action: stall
  max_speculative_ms: 2000
  review_timeout_ms: 1000
  max_attempts_per_segment: 3
  retry_allow_sampling_parameter_change: false
  allowed_recovery_strategies: [exact, error]
  committed_breach_action: terminate
  allowed_start_profile_degrades: []
  require_review_gated_delivery: false

reviewers:
  vad:
    implementation: tenvad
    frame_ms: 16
    begin_count: 2
    end_count: 3
    prefix_preroll_ms: 20
    max_prefix_silence_ms: 1000
    tail_pending_ms: 240
  codec_repeat:
    period_1_abort_frames: 4
  codec_text_alignment:
    mode: shadow
    stable_frames: 2
    active_text_window: 64

rollback:
  exact_enabled: true
  c2w_anchor_frames: 4
  icl_fallback_enabled: true
  icl_context_ms: 1200
  fresh_segment_fallback: true
  stitch_guard_ms: 20
  stitch_source_guard_frames: 1
```

这里的数值是工程起点，不是已校准的生产阈值。尤其是 VAD、ASR stability、ICL context 和
speculative cap 必须按实际 speaker、语言、并发和客户端 buffer 分布校准。

兼容策略必须避免静默改变旧请求的延迟/失败语义：

- 现有 `delivery=guarded` 保留为 `guarded_legacy`，继续表示墙钟 hold + 段尾裁剪，不能映射成新策略；
- `delivery=firehose` 映射到 `legacy_firehose`；
- 新 `delivery=review_gated` 必须由 capability/policy 显式 opt-in，StartAck 回显实际 profile、version、
  required reviewer、bypass 和 timeout 行为；
- required reviewer 由服务端 profile 决定，客户端不能通过省略 reviewer 名称来关闭；不可用时拒绝启动，
  或按客户端明确允许的 degrade policy 回显降级；
- 老客户端未声明时保持版本期原行为，并在 capability 中明确；
- 新客户端可设置 `require_review_gated_delivery`，不支持时 fail fast。

当前 SDK 对协议版本存在严格相等检查，因此不能只把 `tts-session-v2alpha1` 直接改名为 v3 后期待旧
客户端忽略新字段。先用 capability 做向后兼容扩展；等 SDK 支持版本范围/feature negotiation 后，再
升级主版本并保留明确的 minimum-compatible-version。

## 13. 可观测性

### 13.1 水位

每 session 记录或采样：

- `P_output_sample`
- `CB_output_sample`
- `RB_output_sample`
- `SB_output_sample`
- `committed_source_cursor`
- 每个 segment 的 `M_source_frame/M_i/N_source_frame`
- `K_restorable_cursor`
- `speculative_ms`、high-water mark
- `committed_inflight_ms = samples_to_ms(SB-RB)`
- `sdk_received_not_playback_buffered_ms = samples_to_ms(RB-CB)`
- `client_buffer_ms = samples_to_ms(CB-P)`

### 13.2 审核与重试

- verifier 处理 RTF/吞吐和 p50/p95/p99 lag；
- safe watermark jump 大小与频率；
- pending 尾长度；
- reject reason、`H`、检测延迟；
- retry 次数、rollback 距离、恢复策略；
- old-attempt result drop 数；
- false accept、false rollback、abstain；
- text progress offset、置信度和稳定延迟。

### 13.3 提交和降级

- `POLICY_PASSED/BOOTSTRAP_BYPASS/UNDERFLOW_BYPASS/SYNTHETIC_GAP/LEGACY_UNVERIFIED` 音频毫秒数；
- 首个 raw frame、首个可听 commit、首个 policy-passed commit 时间；
- commit batch 大小、ledger reserve 等待；
- replay bytes、ACK lag、resume 次数；
- `safety_breach_after_commit`；
- speculative/ledger backpressure 时长。

### 13.4 回滚成本

- checkpoint 建立时长和显存；
- C2W anchor 数、replay frame 数和 replay latency；
- restore parity failure；
- retry 后到下一 committed audio 的 gap；
- 拼接点 click、能量/F0 跳变离线指标。

日志不得输出完整原始文本、codec 或 PCM；只记录不可逆 request/session 标识、offset 和 reason code。

## 14. 资源、并发与安全

- speculative cap 应按“每 session 毫秒 + 全局字节”双重限制；
- verifier lag 时暂停单 lane，不把整个 GPU batch 锁死；
- pinned slot 数必须纳入 scheduler admission control；
- C2W 重 checkpoint 采用 ring，只保留从当前 `SB_source` 恢复各活动 segment `N_s` 所必需的 anchor；
- 完整 codec IDs 很便宜，应优先保留；PCM 体积更大，可按 native frame 分块并及时回收；
- CommitLedger 只在原子 reserve 成功后接受 commit；
- client progress 不能被信任来扩大安全提交，只能缩小缓存估计或触发受限策略；
- resume token 保持高熵且与公开 session ID、内部 execution ID 分离；
- 多副本部署继续要求 sticky/token-consistent routing；
- 所有 timeout/degrade 必须返回稳定 reason code，不能因内部组件异常自动切回 firehose。

## 15. 测试与验收

### 15.1 水位与状态机属性测试

用 deterministic virtual clock 和随机事件序列验证：

- `P_output <= CB_output <= RB_output <= SB_output`；正常 policy-passed 区间
  `SB_source <= M_source <= N_source`；
- `M_i/M` 只在同一 attempt/review epoch 内单调；retry 后重置到 fork boundary；只有 `SB` 跨 attempt
  全局单调；
- 中间 reviewer 空洞不能跨越；
- ledger reserve 失败时 `SB` 不变；
- resampler/encoder transaction、partial commit 或 edit-map publish 失败时，live transform state、ledger 和
  `SB` 全部不变；
- reject 永远不改 `SB` 之前的输出；
- EOS/error/cancel 只产生一个 terminal；
- tentative frame 永远没有 delivery sequence。

### 15.2 首包

- 首个非静音 packet 不等待 4 帧 reviewer；
- NaN/Inf 仍同步阻断；
- 单帧部分静音能 sample-level trim；
- 20～40ms 首个 wire delivery 仍属于完整 80ms bootstrap commit batch，不能留下半帧 tentative；
- 被 drop 的 prefix 具有 output 长度可为 0 的不可变 edit record；
- 多帧全静音不发送假首响；
- 成功 bypass 只发生一次并带正确 safety class；全静音 drop/失败事务不消耗特权；
- 相对 firehose 基线，可听 TTFA 的 p95/p99 回退不超过预先给定预算，而不是笼统要求逐请求零回退。

### 15.3 reviewer

- 每个 reviewer 单调 contract/property test；
- 重复 run 在 token 改变时只让 `M` 跳到新 frame 的 start，不能越过新 run 第一帧；达到阈值时
  从 run 起点 reject；
- VAD 覆盖前缀静音、短停顿、长尾静音、全静音；
- EOS finalize 不盲目 flush；
- optional/required 组合的聚合交集正确；
- reviewer timeout 不会静默切 firehose。

### 15.4 精确恢复 parity

固定 seed 生成到 `M`，checkpoint 后生成到 `N`，再恢复同一 RNG state：

- full codec bitwise 相同；
- Talker KV prefix、`next_embed`、`token_counts` 相同；
- C2W KV、17 conv、4 transconv state 相同；
- 下一帧 wav bitwise 相同，或在明确的严格误差阈值内，并继续运行至少一个 C2W window 检查误差累积；
- 覆盖 eager/CUDA graph、batch=1/异构 past length、ICL warm state 和流式暂停点。
- 建立 anchor 后继续生成，确认 anchor 独立 tensor bitwise 不变；
- 覆盖 pooled 与 non-pooled rollback，或验证 capability 会拒绝不支持的后者；
- 覆盖目标 logical frame 0 来自 prefill，以及 ICL warm state 的非零 absolute frame index。

换 seed 恢复时验证：

- `M` 前 codec/PCM 不变；
- `M` 后轨迹变化；
- text cursor 不倒退、不跳过、不重复消费；
- checkpoint 后新 append 的文本和 EOF 不丢失；新 attempt lineage/fork frame 正确。

### 15.5 并发和竞态

- rollback 请求恰好在 GPU future in-flight 时到达；
- launch identity 在 scatter 前 fence 旧 lane；旧 future 完成后不能污染恢复后的 slot；
- 旧 attempt 的迟到 checkpoint capture/replay CAS 失败，不能推进新 attempt 的 `K(M)`；
- slot 不提前 release/reuse；
- 多 segment、多 session batch 只回退目标 lane；
- rollback 与 cancel/timeout/断连同时发生；
- 连续两次 rollback 和 retry cap；
- 安全控制队列满时 fail closed。

### 15.6 长连接和 resume

- wire 允许 at-least-once replay；header/binary 中途断开后，SDK 按 `delivery_seq` 去重后的 enqueue/effect
  exactly once；
- ACK 丢失导致 replay 时 SDK 去重；
- 只 replay committed，不 replay speculative；
- old attachment generation 被 fence；
- attempt retry 不改变已 committed cursor；
- terminal ACK、TTL、grace expiry 和 buffer overflow；
- coalescing 只在 ledger/sequence 分配前发生，且不跨 safety/segment/attempt-lineage/commit/event 边界。
- terminal/control queue 在 PCM backlog 下仍有保留容量；synthetic gap、event 和最终 cursor 顺序稳定。

### 15.7 音频/语义数据集

至少覆盖：

- 正常多语言、快慢语速、低音量、长停顿；
- 前缀/尾部/整段静音；
- NaN/Inf、clipping、随机噪声；
- 周期 1 与周期 2～8 重复；
- 无关语音、插字、漏字、重复词；
- 数字、缩写、多音字、代码混读；
- “一个字只说了一部分后开始异常”的人工样本；
- exact、ICL、fresh 三种恢复的听感盲测。

### 15.8 上线门槛

- 首个可听 packet 无固定多帧审核延迟；
- bootstrap 后，review-gated profile 不存在墙钟驱动的正常未审核提交；
- 正常 EOS 和 engine error 不 flush tentative；
- 未触发资源上限时不做 wall-clock 节流；触发 speculative cap 时按设计 backpressure；reviewer p99
  吞吐高于生产速度；
- speculative high-water 在容量范围内；
- exact rollback parity 全部通过后才启用段内恢复；
- semantic reviewer 达到预先约定的 false accept/false rollback/lag 门槛后才从 shadow 升为 required；
- 每一毫秒 bypass 都可按 session、reason 和 safety class 追踪。
- 用确定性仿真覆盖 80ms bootstrap、4 帧重复检测窗、约 2× 生成速度、checkpoint 物化耗时和网络抖动，
  验证下一次 policy-passed commit 在客户端 buffer 耗尽前可达；不能仅凭平均 RTF 推断不会 underrun。

## 16. 分阶段落地

### Phase 0：先修正语义和可观测性

这是当前最先能做、也最能避免继续走偏的一步：

1. 引入 source frame、attempt ID、commit ID 和水位指标；
2. 把 `SB` 定义成 ledger 原子提交边界；
3. 区分 SDK receipt ACK、真实 `CB` 和 `P`；
4. 先修 engine/error 路径：不再 flush 尚未定性的 hold；正常 EOS 在 ReviewCoordinator 落地前仍保留
   `guarded_legacy` 语义并明确标记，Phase 0 不能把它宣称为 policy-passed；
5. 保留首包快速路径并加入 `BOOTSTRAP_BYPASS`；
6. 在最小 checkpoint 尚未落地时显式回显 `bootstrap_restorable=false`，发生后续 reject 只能 terminal；
7. 现有 `guarded` 不再对外宣称 verifier-backed safety。

### Phase 1：规则 reviewer + CommitCoordinator

1. 新增 `AttemptBuffer/ReviewCoordinator/CommitCoordinator/CommitLedger`；
2. 把 PCM health、VAD、周期 1 repeat 改成统一增量 verdict；
3. 用 `commit_through(M)` 替换正常 `release_due()`，但只允许提交到已有可恢复 anchor，或提交已经
   final-safe、不会再产生 suffix 的完整 segment 尾；
4. 先支持“整个 segment 全在 SB 后”的整段换 seed 重跑；
5. WebSocket resume ledger 只接收 committed audio；
6. 实现 speculative/ledger backpressure 和异常矩阵；
7. 活动 segment 如果尚无 checkpoint，首包之后不得做普通增量 policy-passed commit；后续异常只能丢弃
   tentative 并 terminal，不能退回整段重跑。
8. frontend 必须把这类不可恢复输入切成有硬上限的 bounded segment；若实际输出在 EOS 前仍达到
   `max_speculative_ms`，不能 pause 等待一个永远到不了的 final-safe，而要以
   `nonrestorable_speculative_limit` fail-stop。该限制和 reason code 属于 Phase 1 对外契约。

因此，Phase 1 可以验证 reviewer-driven commit、跨 finalized lookahead segment 的
`SB_source -> M_source` 批量推进，
并消除首包固定 4 帧等待，但**尚未完成**用户要求的“已部分提交 segment 内任意位置替换”。若产品要求
Phase 1 就开放活动段增量交付，必须把下面 Phase 2 的最小 Talker+C2W checkpoint/rollback 一并前移；
不能一边推进 `SB`，一边假设未来还能整段 retry。

### Phase 2：精确段内 checkpoint/rollback

1. 暴露并保留完整 16-codebook frame；
2. 实现 Talker 轻 journal 和 pinned-slot rollback；
3. 实现 C2W sparse anchor + T=1 replay；
4. 建立 `Restorable(frame)`/`K(M)`，活动段提交只到不超过 `M` 的最远可恢复边界；final-safe terminal
   按 `CommitEligible` 例外处理；
5. 完成 GPU future/attempt fencing；
6. parity 通过后启用 exact rollback；
7. 再实现 ICL 和 fresh fallback。

完成这一阶段后，活动 segment 才能安全地把 `SB_source` 增量跳到任意可物化的 `M`，并在后续异常时
从新 `SB_source` 精确续写。

### Phase 3：外部 ASR shadow baseline

该工作流不依赖 exact rollback，可与 Phase 0/1 并行启动；这里的编号表示纳入统一 reviewer 接口的门槛：

1. 对 PCM 跑轻量 streaming ASR；
2. partial hypothesis 对齐回 canonical source text；
3. 只记录带置信度/revision history 的 `candidate_M/H`，不 gate 提交；
4. 建立真实 hallucination/正常数据集和 pseudo-label；另保留人工/高质量时间戳验收集。

### Phase 4：codec-text reviewer

1. 做 code0 可分性和 CTC 路径可行性实验；
2. 训练 target-conditioned monotonic aligner；
3. 加 garbage/negative path，避免 forced alignment 假安全；
4. shadow 校准；
5. 高置信词边界先做观测，再逐步成为 required reviewer；
6. 扩展语言、读法 lattice 和模型量化部署。

### Phase 5：性能和协议收口

1. 优化 per-lane pause/resume、checkpoint ring 和 replay；
2. gRPC typed ACK/cursor 与可选 resume；
3. SDK 播放器进度 API；
4. 版本范围协商，最终替换 free-form policy；
5. 删除旧墙钟 hold 的安全职责，只保留明确的 legacy compatibility path。

## 17. 已决策项与待确认项

### 17.1 本设计已决策

- `SB` 是不可撤销提交边界；`[SB_source,N_source)` 才是服务端可替换区域。
- 正常 `SB` 只由连续 reviewer 水位推进，不由墙钟推进。
- 首包不等多帧确认，使用最小、可观测的 bootstrap bypass。
- verifier 可以一次确认连续前缀，`SB` 可以直接跳到 `M`。
- 外部协议只发送 committed audio，不引入客户端 rollback。
- receipt ACK、client buffer 和播放头必须分开。
- exact full-state rollback 是常规策略；短 ICL context 不是常规 checkpoint。
- 保留旧 KV 时不重复喂“尚未听完”的文本。
- codec-text 方向需要训练；目标是单调进度和 mismatch，不是通用 ASR。

### 17.2 产品/实验后确认

- 默认 underflow 是 stall、短静音还是最小 `UNDERFLOW_BYPASS`；
- 完整 80ms bootstrap commit batch 的首个 wire delivery 取 20ms 还是 40ms，以及 packet 开销预算；
- VAD tail pending 和 prefix pre-roll 的语言/speaker 阈值；
- C2W anchor 间隔和每 slot 显存预算；
- fallback 无合法边界而 terminal 后，上层是否重新播放整句或提示用户；同一流内不静默重复/漏字；
- semantic reviewer 成为 required 的 false accept、false rollback 和 p99 lag 门槛；
- gRPC 是否需要与 WebSocket 同等级的逻辑 resume。

## 18. 与现有文档和代码的关系

- 当前协议背景：[`streaming_protocol.zh-CN.md`](../architecture/streaming_protocol.zh-CN.md)
- 当前幻觉调查和现有 guarded 行为：
  [`streaming_hallucination.zh-CN.md`](../investigation/streaming_hallucination.zh-CN.md)
- 当前墙钟 hold：[`engine/frontend/hold_window.py`](../../../engine/frontend/hold_window.py)
- 当前 frontend settle/retry：[`engine/frontend/interface.py`](../../../engine/frontend/interface.py)
- 当前 VAD：[`engine/interface/vad.py`](../../../engine/interface/vad.py)
- 当前 WebSocket gateway：[`engine/gateway/websocket_server.py`](../../../engine/gateway/websocket_server.py)
- 当前 WebSocket resume ledger：
  [`engine/gateway/websocket_resume.py`](../../../engine/gateway/websocket_resume.py)
- 当前 backend loop：[`engine/backend/engine_loop.py`](../../../engine/backend/engine_loop.py)
- 当前 slot/KV state：[`engine/backend/kv_cache_pool.py`](../../../engine/backend/kv_cache_pool.py)
- 当前 ICL prefill：[`engine/backend/prefill.py`](../../../engine/backend/prefill.py)
- 当前 gRPC proto：[`proto/tts.proto`](../../../proto/tts.proto)
