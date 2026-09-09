# Speech State 接入与验收任务

> 2026-09-07：T7/T8 的实施拆分。本文不改变现有 TRT、TN 或模型序列语义；默认 adapter 继续 disabled。

## 1. 已确认的接入风险

代码依据：`engine/backend/engine_loop.py`、`engine/backend/executor.py`、`engine/backend/speech_state.py`、`engine/core/speech_state_model.py`。

1. `_process_step_output_inner()` 内的 C2W pooled KV append 和 arena scatter 有批次末尾的延迟写回。`_handle_segment_eos()` 会先释放 slot，后续写回会跳过已释放槽。因此，不能简单在 EOS handler 内调用 capture 并宣称获得完整最新状态。
2. `_admit_picked()` 会恢复 prefix KV 并初始化 C2W；串行 `_try_prefill_one()` 同样会执行独立 prefill 或 prefix restore。把 restore 插在这些操作之前会被覆盖，插在之后又可能混入新旧状态或重复计算第一帧。
3. 前后 segment 当前允许独立 slot 并行。后段在前段 checkpoint 就绪前已经 admission，是合法的现有行为，但不能当作 predecessor state handoff。
4. 现有 `SpeechStateCapability` 和 adapter 方法存在，不代表调用链已可用。必须区分 adapter 声明、运行时集成和模型验收，不能仅凭注入 adapter 就宣布产品支持。
5. 正常 EOS decode 会提交神经状态但丢弃 EOS PCM。是否允许从该状态继续由模型训练合同决定；完整 snapshot 的迁移等价性不证明跨子句或 EOS 后续接合法。

相关上位设计：

- [增量 TN 与 Soft Drain](incremental_text_normalization_and_soft_drain.zh-CN.md)，尤其是 I5、Phase 0 边界和 context rollover 章节。
- [原生游标 TRT 决策](../architecture/native_cursor_trt_fusion_decision.zh-CN.md)，尤其是 lookahead、slot isolation 和异常 flush 约束。

## 2. 本轮边界

先建立 backend-owned 的操作上下文和句柄校验合同。不得自动在 EOS handler capture，不得仅因 `supports_segment_handoff` 为真就改变 admission 顺序。具体状态 payload 不经过 asyncio，也不进入公共 JSON。

完整状态迁移、子句续接和上下文压缩分别验收：

| 操作 | 可以证明什么 | 不能据此宣称什么 |
|---|---|---|
| PAUSE_RESUME 完整 snapshot | 同一声学序列在兼容 slot 恢复后继续执行 | 跨子句合法、缩短 KV、EOS 后续接合法 |
| SEGMENT_HANDOFF | 目标 X2 模型在明确边界上的子句续接质量 | 官方模型也支持、任意 checkpoint 通用 |
| CONTEXT_ROLLOVER | 声明方法和误差等级下的上下文切换 | 普通 KV clone 已解决容量限制 |

## 2.1 三类文本输入的统一处理语义

Gateway 不应为三类输入维护三套 TN。三类输入都进入同一个主
`IncrementalTextCommitter`、`TextCommit`、`CanonicalTextJournal` 和 Splitter；区别只在
输入是否已经结束，以及是否允许一次性建立预分段计划。

| 输入 case | TN / 分段行为 | label plan 与游标行为 | 结束条件 |
|---|---|---|---|
| 完整长文本 | 一次性 `final=true`，对已知文本做离线 TN 和预分段；仍由主 TN 生成 spoken projection 与 owner span | 对稳定提交区建立完整 plan；cursor-enabled TRT 可按普通 segment 使用原生游标 | 所有 segment drain 完成并收到 codec EOS |
| 不完整长文本 | 按增量输入处理。安全前缀产生 `TextCommit` 后才能进入 Splitter；`pending_raw` 保留可能被后续输入改写的尾巴。没有新文本时进入现有 `WAIT_TEXT`，不能用 PAD/EOS 解决语义未决 | 只为已提交前缀生成 label plan。新提交、tail rewrite 或 boundary-before 会产生新 plan revision；CPU 通过 owner id/raw span 重锚定，不能按旧 label index 或长度比例推坐标。未决尾巴没有 label，不驱动游标 | 调用方显式 `input_complete=true` 后，主 TN flush 全部尾巴，再允许最终 hard finalize；临时无输入不是结束 |
| Token 级输入 | Token 入口仍只提供原始/已约定的 token 事实，由主 TN 和 commitment adapter 形成统一 spoken projection | 增量产生的 label plan 驱动同一个 Splitter FSM；cursor-enabled TRT 每个 codec frame 只更新一次 cursor matcher | token 输入完成并显式结束，或按协议声明的安全 boundary 完成 segment |

case2 的关键是“提交前缀、挂起尾巴”，而不是“把每个 delta 当成独立长文本”。因此：

1. `pending_raw` 非空时，不得发布覆盖未决尾巴的 `TextCommit`、label plan、音频结束或
   successor context。
2. committed prefix 已经安全且达到显式 punctuation/capacity boundary 时，可以结束当前
   segment；这只是 per-segment drain，不表示 session 输入结束，也不自动触发 EOS 后继承。
3. 后续 delta 改写 TN tail 时，已公开的 raw/normalized high-water 只能单调增加；新旧
   plan 的坐标转换必须通过稳定 owner span 完成，展示层插值不能进入协议、计费或恢复。
4. 只有 `input_complete=true` 才能把剩余 pending tail 解析为最终 spoken form，并进入
   `HARD_FINALIZE`。`WAIT_TEXT` 是等待更多输入的运行态，不是失败，也不是声学 flush。

当前代码已有主 TN、`WAIT_TEXT`、owner-span reanchor 和统一 Splitter pending queue；
`tests/integration/test_streaming_tn_cursor_plan_contract.py` 已补齐一条跨层合同，覆盖
“未决中文数字/单位尾巴 -> 后续 delta 改写 -> plan revision -> token dispatch”的完整
事件顺序。它证明 pending tail 不发布音频/Label Plan，且 `UPDATE_CURSOR_PLAN` 先于同批
`START/APPEND_TOKENS`；独立规则 TN 或固定 label fixture 仍不能替代这项测试。

## 3. 可独立委派任务

### S1：Backend 安全边界合同

Owner：`engine/backend/speech_state.py` 或其同职责子模块、`tests/unit/engine_core/`。

- 使用枚举区分操作和安全阶段；显式携带 session、segment、attempt、slot allocation epoch 与 model/runtime fingerprint。
- context 只描述 engine thread 操作，不序列化 GPU payload。
- 拒绝跨 session、错误来源、错误 attempt、过期 generation 和不兼容 fingerprint；目标 slot epoch 不能被误当成源 slot epoch。
- `NullSpeechStateAdapter` 和既有 capture/restore 调用保持兼容。

验收：纯 CPU 合同测试覆盖合法迁移和每个拒绝路径；不得因此开启公共能力。

### S2：完整状态迁移基线

Owner：executor/slot state 与 backend snapshot 存储组件；不改 gateway、TN 或 splitter。

- 先实现 PAUSE_RESUME，不提前实现子句续接。
- 在所有 Talker/C2W/cursor 状态写回完成且无在途 graph 引用的 step 边界 capture。
- 清单包括有效 Talker KV、C2W pooled/非 pooled KV、conv/transconv、buffer parity、frame/text index、next embed、trailing/queued text、RNG、cursor lookahead/history/labels/revision/override。
- 明确 CP 是否存在持久跨帧状态，以实际 fused ABI 为准，不创建虚假的 CP KV。
- 快照不能继续引用会复用的 pool/arena/staging view；生命周期、容量预算和 release 必须可测。
- restore 先完整验证，再提交到干净目标 slot；失败不能留下可 admission 的半恢复状态。

验收：冻结输入，对比不中断与 snapshot/restore 后 next logits、codec、PCM 和 cursor 轨迹；覆盖 pool/arena、slot 复用、batch/eager/CUDA Graph。GPU 验证前不得声明 exact 已验收。

### S3：X2 模型续接合同

Owner：模型适配/训练与导出；与 S2 可以独立开展。

- 明确哪个 checkpoint 支持子句继承、合法边界 token/phase、是否保留 Talker KV、C2W 的连续条件。
- 明确捕获点在 EOS 之前还是之后，不能从现有 codec EOS 处理惯例反推训练语义。
- 定义目标段文本如何接入恢复状态，以及是否需要模型专用 prefill；禁止 restore 后再走会覆盖它的普通 prefix-cache prefill。
- native cursor 必须一起迁移或明确禁用；segment label 坐标变化必须通过 owner/reanchor，不能沿用旧 label index。

验收：checkpoint/fingerprint 合同加连续性、停顿、click、F0/energy、音色和 WER/CER 报告；不匹配明确 hard boundary。

### S4：Engine Handoff 调度接入

Owner：engine loop 的窄 lifecycle 组件；依赖 S1-S3。

- 前后段依赖必须同时覆盖 batch cache-hit 和 serial admission，不能只给其中一条路径加等待。
- checkpoint 只在已完成状态写回的 fence 发布；继承后 successor 从明确的 ready-to-decode 或模型专用 prefill 状态启动。
- handles 按源 attempt/generation 路由；retry、取消、超时、session 替换、EOS 和异常释放必须释放且不能重复使用旧句柄。
- restore 失败且尚无 successor PCM 时，可清理目标 slot 后独立 hard-boundary prefill；已有输出后不能静默重跑或重置声学状态。
- 仅 runtime 集成和模型验证均通过的操作进入公共 capability，adapter 声明本身不是开放条件。

验收：测试证明无提前 admission、无重复首帧、无半旧状态、无句柄泄漏；Native/Triton 相同生命周期合同。记录串行依赖对吞吐和首音延迟的影响。

## 4. 当前进度

S1 的 `SpeechStateContext`、capture/restore phase enum、opaque handle 的 session/source segment/attempt/slot epoch/fingerprint 校验已实现于 `engine/backend/speech_state.py`，并由 `tests/unit/engine_core/test_speech_state.py` 覆盖。它只拒绝不安全的 restore 请求，不会自动 capture、restore 或改变默认 disabled 路径。

2026-09-08 复查补齐 S1 的两处拒绝条件：restore 必须提供预期 source generation；model/runtime fingerprint 两侧都必须非空且匹配，不能以缺失 fingerprint 绕过校验。

S2 首个实现子集为 `engine/backend/slot_snapshot.py` 的 `StandaloneSlotSnapshot`：仅复制 live、非 pooled、非 arena slot。它复制 tensor、两侧 ping-pong buffer、cursor 神经状态、文本队列和 generator 当前状态；在复制前检查 tensor/RNG 字节预算，restore 返回独立的新 slot 对象。预算不含 Python 容器开销，且不是全局句柄存储预算。CPU 单测验证 source/snapshot/多次 restore 不共享可变数据，以及恢复后的 RNG 序列一致。

随后为 `SlotKVState` 增加了 `allocation_epoch`。每次物理 pool row 被重新分配时递增，pool release 和 arena 原语均可用它做 ABA 校验；它仍不是完整的 state-transfer handle。

当前又增加了 `SpeechStateHandleStore`：它在 engine thread 内持有 backend 私有 payload，以 UUID 形式生成 opaque handle，并在消费前执行既有上下文/指纹校验；成功消费后立即删除条目，重复 restore 必须失败。它仍不负责 CUDA fence、pool/arena capture 或 EngineLoop admission，因此不能单独宣称状态迁移可用。

随后增加了 pooled Talker 的左对齐 KV 和 pooled C2W 的右对齐有效窗口 snapshot/restore，以及 C2W A/B arena 原语。`C2WArenaSnapshot` 固定记录 source slot 与 capture epoch，但 restore 的目标 slot/epoch 独立传入，因此允许迁移到新 allocation，同时拒绝伪造 source metadata；capture 还验证 slot 的 read/write tensor view 确实属于该 slot 的 A/B arena row。数量、shape、dtype、parity 和 row 绑定均有单元测试覆盖。

Luna 的只读审计确认以下仍不属于这个 primitive：

- pooled C2W 与 Talker 的统一 payload 编排；
- session group 状态，以及 EngineSegment metadata 与 tensor payload 的 EngineLoop 组装；
- handle 存储、generation 生命周期与模型兼容性验证到真实 snapshot 的绑定；
- engine thread 的批末 fence 和 GPU stream 完成保证；当前 snapshot 原语不会自行判断批末 deferred scatter 是否完成；
- restore 对象的 pool adoption、GPU next logits/codec/PCM 数值验收。

EngineSegment 元数据的 detached `SegmentRuntimeMetadata` 也已建立 CPU 合同，复制
`pending_token_ids`、input completion、decode/text index、loop guard、retry/audio health、
cache counters 和 cursor plan revision；它不复制 slot 指针、cursor plan 对象、时间戳或
opaque handle。恢复要求 session/segment owner 相同。新增的
`project_successor_runtime_metadata()` 明确把跨 segment 的 successor session/index、待合成
token 和 `input_complete` 投影为新的 `pending_prefill` 元数据，并重置 predecessor 的
音频、loop、cache、retry、cursor revision 等 segment-local 字段；它仍只是 CPU 合同，尚未
与真实 X2 tensor payload 在 EngineLoop 的 successor admission 边界组装。现在该组装由同
segment migration 在 boundary 内完成，跨 segment 的真实 restore、prefill 顺序和模型连续性
仍未完成。

S3 先增加了纯 CPU 的 `SpeechStateModelContract`，用 typed enum 明确 checkpoint、合法
capture boundary、successor 启动方式、Talker/CP/C2W 保留项、cursor policy、training
scope 和 transfer accuracy。它支持从完整 mapping 严格解析并 round-trip，只提供模型
声明的校验与序列化，不会自动改变 `SpeechStateCapability`，也不会把
`READY_TO_DECODE` 推断成模型已训练支持。

本次复查了工作区中的方法层参考实现 `/home/rime/workspace/x2streaming`：发布的
`X2Streaming-TTS-1.7B` 继承语义不是直接搬运 Talker KV，而是完整复制 Code2Wav KV、
conv/transconv history 和 frame index，并只保留前驱最后 `H=4` 个 token-aligned Talker
hidden，经严格因果 text-acoustic bridge 注入 successor。前驱只有在
`input_complete=true`、文本全部消费、`codec_eos` 正常结束且 audio/text ratio 通过健康门时
才发布 context；失败则清空链并 hard-boundary 起步。`SpeechStateModelContract` 现已能用
`talker_carry=hidden_tail_bridge` 与 `talker_hidden_tail=4` 表达这一语义，不能再用
`retain_talker_kv=true` 代替它。

导出 manifest 现在可选地携带完整的 `speech_state.model_contract`。export builder 会用
同一 typed contract 做严格解析，并要求 contract 的 `model_fingerprint` 与外层
`speech_state.model_fingerprint` 相同；旧的仅有双 fingerprint manifest 仍兼容。该字段
只记录 checkpoint 的边界、状态保留、successor 入口、cursor policy 和训练范围。Executor
现在还会把 adapter capability 与该合同、外层 model/runtime fingerprint 和 transfer
accuracy 做 fail-closed 对齐；缺合同、身份不匹配或 transfer 不一致时，公共 capability
保持 disabled。因此合同声明和 runtime 门禁已具备，但真实模型 bundle 仍需经过后续数值
与连续性验收才能启用。

H2 的 fail-closed 规则还明确到 cursor policy：Executor 现在提供了 gated 的
cursor recurrent-state payload/restore ABI，但只有完整六组 recurrent input/output
binding 都存在时才设置 handoff flag。manifest 即使声明 `cursor_policy=migrate`，在
半套 cursor graph、缺少 bundle/fingerprint 或 release evidence 时仍不会开放
speech-state capability；不能把 acoustic C2W restore 当作 cursor state migration。六组
binding 名称已提升为 `engine.core.native_cursor` 的共享 ABI 合同，纯 CPU bundle verifier
会在运行时发现之前直接拒绝 `incomplete_cursor_recurrent_abi`，避免 manifest 先被误判为
verified；executor 的 CUDA 发现仍保留第二道防线。

H4 的公共 capability 现在在不改变 `SpeechStateCapability` typed wire contract 的前提下
附加 `speech_state.reason`：已启用时为 `enabled`，未启用时区分 adapter disabled、缺少
fingerprint、缺少/不匹配 model contract、transfer mismatch 和 cursor handoff ABI 不可用。
Native Gateway 与 Triton 兼容层都经过同一个 `build_gateway_capabilities()` 保留该字段；
这只是诊断信息，不能单独开放任何 runtime capability。

引擎侧已新增 dependency-free 的 `EngineExtensions`/continuity lifecycle hook：session
创建、successor admission gate、C2W warm restore、hidden observation、bridge apply、
健康 finalize、retry reset 和 session invalidation 都位于 engine thread。前端现在也有
`X2CommitmentAdapter`：只消费主 TN 已完成的 `TextCommit`、spoken projection、调用方
提供的 token count 和 splitter boundary level，向策略转发容量反馈；它没有 `feed_text`
入口，不会建立第二套在线 TN。默认没有注入 policy 时行为不变；extension 私有对象不
进入通用 slot snapshot，也不进入公共 JSON。这只是 X2 方法层接入面，尚未把外部
checkpoint 自动加载为默认 capability，也没有替代原生 cursor 的 TRT 状态路径。

策略故障也有明确的降级边界：若 `ready_to_admit`、`context_for`、bridge 或
`finalize_segment` 回调抛错，EngineLoop 会标记该 session 的 continuity policy disabled，
后继段不再等待失效的 policy，而是按普通 hard-boundary admission 继续；不会把一个可
降级的策略异常变成永久 `WAIT_TEXT`。这只保证调度不死锁，不代表继承结果有效；策略
失效、restore 失败和模型不匹配都必须在 metrics 中保留原因，并继续保持公共 capability
fail-closed。

2026-09-08 兼容性复查修复了 C2W continuation payload 的字段合同：引擎内部继续使用
明确的 `c2w_conv_states`/`c2w_transconv_states`，同时向 X2 方法层暴露
`c2w_conv`/`c2w_transconv` 只读别名。此前外部 X2 health gate 会因读不到 conv state
而把完整 snapshot 误判为不完整；现在已用外部方法层实际 finalize/context 流程验证别名
可见，但这仍不等于真实 TRT checkpoint 的数值验收。

前端 commitment 需要单独看待：外部 X2 的 `CausalCommitment.feed_text()` 同时包含
规则边界和特殊文本归一化；当前引擎的唯一 spoken-form 真相仍是
`IncrementalTextCommitter`、`TextCommit` 和 `CanonicalTextJournal`。因此
`EngineExtensions.commitment_factory` 现在只能通过 `X2CommitmentAdapter` 接入，不能直接把外部
`feed_text()` 接到前端，否则会形成第二套在线 TN。真实接入必须改成“主 TN 输出
`TextCommit`/owner span，X2 commitment 只消费提交事件并提供容量/边界策略”的窄适配器，
并证明 full text、增量长文本和 token 输入的最终 spoken projection 一致。该适配器已经
完成 CPU 合同、三入口 frontend projection 测试和接线；外部 X2 checkpoint/export/fingerprint、真实策略 factory、
以及方法层 commitment 与 Splitter FSM 的跨入口联测仍是未完成任务。在此之前不能宣称
X2 方法层已通过统一引擎启用。

2026-09-08 用外部 `/home/rime/workspace/x2streaming` 的方法层测试做了接线复核：继承/策略
测试 `14 passed, 1 skipped`，与当前 `EngineExtensions` 的 factory 和 lifecycle 方法名兼容；
外部 `test_upstream_extension_hooks` 通过。外部 `test_upstream_lifecycle_callbacks` 仍有一项
断言失败，因为它期待 successor token 保留原始 `25℃`，而统一引擎主 TN 已按唯一
spoken-form 真相提交 `二十五摄氏度`。这不是可以通过放开第二套 `feed_text()` 修掉的兼容问题；
外部测试必须改为断言主 TN 的 spoken projection，否则 H3/H4 不能算跨入口通过。

适配器现在还把主 TN 的特殊/非 literal `TextCommit` 映射为
`boundary_before`，把 spoken 末端的稳定边界映射为 `force_boundary`，两者通过
Splitter pending queue 的 typed 标记实现“先结束前段，再把新 special span 放入后段”。
这复现了 X2 的 `boundary_before` 语义，但没有重新运行 X2 的 raw-text rule frontend。

原生游标与 X2 声学继承已明确隔离：successor context 现在由 engine-owned
`CursorContinuationState` overlay 到外部 X2 的 C2W context，避免外部 policy 只保存
C2W 时丢失 fused cursor 状态。它只恢复六组神经递归 tensor，不携带 label plan；successor
的 label plan 仍必须由主 TN 重新生成。只有 cursor-enabled TRT graph
完整暴露 recurrent ABI 时，EngineLoop 才保留 successor 的 native progress；旧的 C2W-only
context、半套 ABI 或恢复失败都会显式降级 EMA/硬边界。无 continuity context 的普通
segment 仍可使用 cursor-enabled TRT plan。普通 segment 内主 TN 产生新的 plan
revision 时，EngineLoop 现在会在更新固定 label buffer 后，依据旧/新 plan 的稳定
`owner_id`，必要时再依据稳定 raw/normalized span，调用现有 `cursor_override_mu` 合同做
CPU 重锚定；不使用长度比例，也不复制 TN。只有 H3 完成 successor cursor recurrent
state 的迁移和 TRT 数值验收后，才允许去掉 successor 的 EMA 降级。

为此增加了 backend-owned `SpeechStateSnapshotBundle` 组装合同：它绑定 source session、
segment、slot、allocation epoch、segment metadata，以及 standalone/pool/arena payload
引用。`Executor.capture_speech_state_bundle()` 现在能在调用方已证明 quiescence 的前提下
按真实存储 owner 组装 standalone、pooled Talker/C2W、非池化 C2W 和 A/B arena payload，
并做 source owner、segment owner、有效历史和总 tensor 字节预算校验；slot-owned snapshot
不会把非 arena 的本地 ping-pong state 漏掉。该入口不执行 CUDA wait、不伪造 fence、不改变
公共 capability，也没有自动接入 EOS 或普通 admission；EngineLoop 的安全边界 migration
只对同一 segment 的 PAUSE_RESUME 开放给已声明 capability。

`EngineSegment` 现在持有可选 bundle owner，并在 slot release/retry cleanup 时同时清空
bundle 与 handle，避免释放后的 segment 把旧 payload 带入后续 admission；这仍不是 capture
或 restore 调度。

EngineLoop 现在提供 engine-thread-owned 的 `request_speech_state_migration()`：请求只携带
session/segment/attempt/epoch/预算元数据进入线程安全队列，只有上一轮 `GPUFuture.wait()`
完成、`_process_step_output_inner()` 完成 pooled KV append、arena scatter 和 ping-pong
flip 后的下一步边界才会执行。capture、目标 clean-slot restore、stream fence 和 source
slot release 在同一个 engine-thread 操作中完成；Future 只报告完成或错误，GPU payload
不会离开 backend。目标恢复失败时 source 保持原样，目标 allocation 被清理。默认 disabled
capability 立即 fail closed；该 baseline 只覆盖同一 segment 的 PAUSE_RESUME，尚未把
跨 segment successor/admission 接入 scheduler，也不会自动改变 EOS、prefill 或
continuous batching。迁移还要求 Executor 提供非空、已验证的 model/runtime fingerprint；
当前默认 TRT/官方路径没有这组 fingerprint，因此不会因队列代码存在而自动开放能力。
bundle 在迁移期间由 `SpeechStateHandleStore` 以 opaque handle 绑定 source attempt、
allocation epoch 和单调 generation；restore 校验成功前不会将它暴露给 queue caller，
restore 失败后旧 handle 也不可重用。
fingerprint 的来源约定为导出 bundle manifest 的 `speech_state.model_fingerprint` 与
`speech_state.runtime_fingerprint`；manifest builder 现在支持显式写入并拒绝半成品，
缺任一字段即保持 fail closed，不能用 cursor head fingerprint 或外部请求参数替代
acoustic runtime identity。现有 export 入口尚未为任何模型声明这两个值，因此能力仍关闭。

连续性 policy 的 admission、restore、bridge、hidden observation、retry capture 失败现在
都会进入稳定的 `extension_continuity_reason`；session factory 失败也会记录
`continuity_factory_failed`。这些 reason 会随 `PREFILL_DONE`/`SEGMENT_END` metrics 透传，
使 WebUI/Native/Triton 录制可以区分“等待前驱”“继承成功”“硬边界降级”和“policy 失效”，
而不是只看到一个无法定位的 disabled 布尔值。

EngineLoop 还提供了受 capability gating 的 `attach_speech_state_bundle()` 窄入口：只允许
匹配 source session/segment 的 bundle attach 一次；默认 disabled 或重复 attach 均拒绝。
配套的 `consume_speech_state_bundle()` 只允许同一 owner 成功消费一次；成功后立即从
segment 移除。两个入口只建立 ownership，不会改变 scheduler、prefill 或 decode 行为。
另外提供事务式 `restore_speech_state_bundle()`：backend restore callback 抛错时 bundle
保持 attached，只有 callback 成功返回后才消费，供上层执行 hard-boundary fallback 或
显式 cleanup。`discard_speech_state_bundle(reason=...)` 要求清理原因非空，重复清理也
会被拒绝。

2026-09-08 又补强了方法层 successor lifecycle：continuity 单测覆盖前段 EOS 后 gate 放行、
serial C2W restore 失败后的 hard-boundary fallback、cancel/session replacement 清理、retry
capture 重置和 cache-hit batch gate；当前该测试集为 `20 passed`。`Executor.apply_c2w_warm_state()` 现以
准备完成后一次提交的方式更新 slot，任一 conv/transconv/KV 转换失败都不会留下半份状态。
该生命周期测试还覆盖取消/会话移除后 engine-owned cursor continuation map 的清空，避免
session 对象继续持有 predecessor tensor。
另有环境门控的真实方法层联测 `tests/integration/test_x2_method_layer_contract.py`；设置
`X2STREAMING_ROOT` 后会直接加载外部 `X2StreamingPolicy`，full/long/token 和不完整长文本
case2 当前共 `4 passed`。

2026-09-09 新增并通过 `tests/integration/test_streaming_tn_cursor_plan_contract.py` 的
raw→TN→Label Plan→raw projection 合同：以 `99%` 为例，主 TN 产生
`百分之九十九`，cursor owner 保留 raw `[0,3)` 与 normalized `[0,6)`，cursor projector
在 owner 完成时发布相同的 raw/normalized high-water；同时验证 Label Plan revision 在
token request 之前进入 engine queue。该测试证明了 TN 与 cursor 的坐标协议不依赖 raw 和
spoken 文本等长，但仍是 CPU/队列合同，不替代真实 TRT 音频和离线 ASR 验收。

因此 S2 已完成 standalone slot copy、pooled Talker/C2W、arena copy、slot-owned
auxiliary state、executor capture/restore assembly、EngineLoop 同 segment boundary
migration primitives，以及真实 cursor-enabled TRT plan 的 full-state 下一步输出对照；仍
未完成 X2/目标 plan 的长轨迹数值对照、跨 segment successor migration，也未开放未经模型
验证的 state-transfer capability。full-state PAUSE_RESUME 等价不代表 X2 子句续接已实现。

## 5. 实施状态矩阵

| 阶段 | 当前状态 | 已有证据 | 下一闸门 |
|---|---|---|---|
| S1 安全合同 | 已完成 CPU 合同 | context、generation、fingerprint、opaque handle 校验测试 | 不得仅凭合同开放 capability |
| S2 tensor 基线 | primitive、executor capture assembly、engine-thread step-boundary 调度、双 CUDA stream fence、detached bundle round-trip，以及 successor 目标元数据 CPU 投影合同已完成；真实 cursor plan 的 eager/CUDA Graph full-state 下一步对照已通过；X2/公开 state-transfer runtime 仍未完成 | standalone、pooled Talker/C2W、arena、slot-owned、RNG、metadata、successor metadata reset、ABA、capture、migration 时序、restore/rollback、真实 CUDA fence、GPU pool/arena detached restore，以及真实 TRT 的 eager/graph token/PCM/codec_sum/token-count/conv/transconv/cursor 对照 | X2/目标 plan 的完整 state migration、不中断长轨迹数值对照和 successor handoff |
| S3 X2 续接 | typed contract、hidden-tail bridge 语义、manifest contract 校验、cursor migrate fail-closed gate、完整 cursor recurrent payload/restore ABI、严格 mapping round-trip 和 CPU owner/reanchor 合同已完成；真实 X2 方法层/TRT successor E2E、cursor 子图轨迹、5 步 FP32 fused graph 冻结输入对照和外部 staging bundle 的真实 fingerprint/layout/ABI 校验已通过；生产发布包仍未准入 | 合同测试、manifest identity mismatch 测试、cursor policy gate、runtime gate 测试、外部 X2 方法层参考实现、真实三段 successor A/B、cursor state detached/restore、32 帧 PyTorch/真实 TRT cursor 对照、5 步 ORT/FP32 TRT 对照和 successor prefill 单测、健康门/状态字段对照、owner-id/raw-span reanchor 测试、staging bundle verifier 与 Native/Triton projection | 精确 speech-state model/training contract、BF16 默认 plan 的 ASR/音频连续性与完整事件对照、不中断对照、发布 evidence |
| S4 Engine Handoff | 同 segment PAUSE_RESUME、可注入 continuity lifecycle，以及主 TN 驱动的 X2 commitment adapter 已完成；普通 segment 的 plan revision 已接入 CPU 重锚定；engine-owned cursor overlay 已把外部 C2W-only context 与 fused cursor state 解耦；C2W-only successor 和 cursor-state successor 的 EMA/native 路由已显式区分；真实 X2 successor E2E、cursor recurrent restore 和 Native/Triton capability projection 已通过；默认公共运行时未启用 | admission gate、C2W warm restore、EOS 不继承、hidden observation、retry/invalidation、C2W 外部字段兼容、commitment 接线、cursor/continuity 隔离、engine overlay、真实三段 E2E、cursor 下一步轨迹、双入口 capability 同结果、factory/restore/bridge/retry reason 单元覆盖 | 完整 speech-state bundle successor migration、不中断/继承全量 TRT 对照、Native/Triton 音频结果集 E2E 和发布 evidence |
| H5 发布闸门 | 已完成纯 CPU evidence gate、Executor/Triton 统一投影和发布校验 CLI；已有固定 GPU 的 BF16/FP32 graph microbenchmark 与真实 X2 A/B，但没有真实 evidence 文件时 native/state 仍自动关闭 | `ReleaseGate` 合同测试、Executor 与 Triton projection 测试、`validate_capability_evidence.py` CLI 测试、5 步 fused graph 报告、RTX 5090 decode microbenchmark、真实 X2 A/B `2 passed` | 固定语料的 BF16 音频/ASR 质量、连续性、服务性能、双入口 E2E 和身份绑定 evidence；真实报告通过后才生成可发布 evidence |

S2/S4 的当前代码仍不会改变默认 disabled capability、EOS、WAIT_TEXT、prefill 或
continuous batching 行为。任何“状态继承可用”的外部声明都必须等待真实合同 bundle
装载、S4 运行时验收和 Phase 2 的 TRT 数值对照；仅通过 CPU contract/runtime gate
测试不能宣称模型能力已开放。

S4 接线的硬边界已确认：capture 不能放在 `_handle_segment_eos()`，因为该路径可能先
释放 slot，而 `_process_step_output_inner()` 的 pooled KV append 与 arena scatter 在
批处理尾部才完成。安全候选点必须位于该批次所有 GPU wait、KV scatter、arena scatter 和
ping-pong flip 完成之后、下一次 decode launch/admission 之前；同 segment PAUSE_RESUME
migration 已接入该边界，跨 segment successor 的 admission 仍不得复用这条 baseline。

2026-09-09 H1 CPU 时序验收已完成：`test_engine_loop_drains_migration_after_completed_step_output`
在真实 `_run_inner()` 主循环中提交 migration，验证首轮 decode launch 期间不执行，只有
下一轮 `future.wait()` 返回并完成 `step.process` 后才 drain；Future 只返回完成状态。
另有环境门控的 `tests/integration/test_cuda_state_transfer_boundary.py` 在当前 CUDA
设备上验证专用计算流和 deferred-copy 流都被 fence，`Executor.synchronize_state_transfer()`
会在同一 CUDA stream 时去重。该证据只证明 stream boundary，不替代 pooled/arena GPU
restore、真实 TRT 状态迁移或不中断/迁移数值对照。新增环境门控的
`tests/integration/test_cuda_speech_state_bundle_roundtrip.py` 已在当前 CUDA 设备上验证
真实 pooled Talker/C2W row、右对齐 C2W history、双 parity arena 和 slot auxiliary/cursor
state 的 detached bundle round-trip；它同样不替代模型输出数值等价验收。
另有环境门控的 `tests/integration/test_real_trt_state_transfer_contract.py` 在真实
`workspace/exported/custom-1.7b/talker_code2wav_fused.engine` 上完成一次 zero-embedding
prefill、detached bundle capture/restore 和下一步 decode 对照；单 slot 与 source/target
同批 decode 的 token、PCM、`codec_sum`、updated token counts、17 路 C2W conv 与 4 路
transconv 均逐元素一致。该 standard plan
没有 cursor binding、X2 bundle 或 speech-state manifest，所以这项证据只证明现有
executor 状态 ABI 在该真实 TRT plan 上的下一步输出保持，不开放 speech-state，也不替代
X2 successor/cursor-state 迁移验收。
同一测试还把该 package 送入 Triton capability builder，确认 Executor/Native 与 Triton
对 `supported=false` 和 `reason=missing_speech_state` 的公开结果一致；即使真实 Executor
已加载且 EngineLoop 标记为 running，migration request 仍会在入队前因 capability disabled
被拒绝，不能绕过 manifest/release gate。

## 6. 下一批委派任务

以下任务只能在对应输入和模型 artifact 到位后开放；没有真实 checkpoint/plan 时，先完成
合同、fixture 和失败路径，不能用 CPU fake 结果替代模型验收。

### H1（已完成）：step-boundary migration 运行时证明

**Owner：** `engine/backend/engine_loop.py`、`tests/unit/engine_core/`。

**范围：** 只覆盖同 segment `PAUSE_RESUME`。补充一个可重复的 engine-thread 测试，证明
`GPUFuture.wait()`、pooled KV/arena scatter、ping-pong flip 全部结束后才 drain migration；
成功路径保留 slot state，失败路径保留 source、释放 target，Future 不暴露 tensor payload。

**禁止：** 不开放默认 capability，不接入 EOS，不把这条路径改造成 successor handoff。

**完成证据：** CPU lifecycle test、pooled/arena restore test、失败后 slot/handle/epoch
检查，以及一次真实 CUDA stream boundary 验收记录。

**验收结果（2026-09-09）：** 已完成。真实 `_run_inner()` 时序测试证明 migration 只在
`future.wait()` 和 step output 处理后 drain；真实 CUDA 测试证明 compute/deferred-copy
双 stream fence；真实 GPU pooled/arena round-trip 与 standard TRT 下一步输出对照均通过。
该任务不开放默认 capability，也不扩展为 successor handoff。

### H2：实现 X2 checkpoint/export bundle 装载

**Owner：** `scripts/export/`、`scripts/python/triton_manifest_io.py`、runtime loader。

**依赖：** S3 `SpeechStateModelContract` 和已冻结的 X2 checkpoint/export ABI。

**范围：** 从模型 bundle 自动发现并校验 `speech_state.model_fingerprint`、
`runtime_fingerprint`、model contract、Code2Wav state layout、hidden-tail bridge 和
cursor artifact；任一字段不匹配时保持 `DISABLED`，不得从请求参数或 cursor head 猜测。

**当前进展（2026-09-09）：** 已增加纯 CPU 的
`engine.core.speech_state_bundle.validate_speech_state_bundle()`，Executor 在读取
`triton_manifest.json` 后校验 bundle schema、model/runtime fingerprint、严格 model
contract、Code2Wav layout digest、hidden-tail bridge、model weights/runtime plan/source
copy hash 和 cursor artifact；验证失败通过 `speech_state.reason` 保持 fail-closed。原生
游标运行时也新增 `cursor_head_sha256` 对导出权重的逐字节校验，stale head 不再只因文件
存在就被挂载。bundle/hash/Executor gate 相关单测均通过，engine/gateway/frontend
相关单测集也保持通过；manifest export builder 会保留 bundle descriptor、自动补齐
Code2Wav layout digest，并拒绝缺少 artifact hash 的导出声明；当 contract 声明
`cursor_policy=migrate` 时，builder 也要求完整六组 cursor recurrent binding，避免生成
已知会被 runtime verifier 拒绝的半成品 manifest。`export_09` 同时改为显式维护 cursor
input/output ABI 列表，不再通过 I/O 列表位置切片推断 binding，避免主 TRT 图新增普通
输入输出后静默错配游标接口。
本轮又收紧了 verifier 的文件边界：导出 artifact 和 source copy 必须是 bundle 根目录内
的相对路径，`..`、绝对路径以及逃逸 bundle 的 symlink 均返回
`*_path_outside_bundle` 并保持 disabled，避免 manifest 借路径绑定到 bundle 外的任意文件。
bundle schema version 现在也严格要求 Python `int`，`true`、`1.0` 和字符串版本不会绕过
校验；verifier、export builder 和 Executor cursor discovery 的 `native_cursor.enabled` 同样
只接受严格布尔 `true`，malformed capability 不会触发实际 binding discovery。model/runtime
fingerprint 以及 model contract 的 identity 字段也必须是非空字符串，不再把数字或容器
隐式转换为身份文本。
Executor 的 runtime capability getter 也会把 malformed `progress_available` 归一化为
disabled，而不是依赖最外层 gateway 再修正；这样 health、Native Gateway 和 Triton 兼容层
观察到的是同一 fail-closed 状态。
仓库当前发布用的模型包仍没有带 `speech_state.model_fingerprint`、`runtime_fingerprint`、
model contract 和 `capability_evidence.json` 的完整 release bundle，因此还没有启用任何
生产 speech-state capability；CPU fixture 和仅有 native-cursor 图的 bundle 都不能替代
H2 完成证据。

**外部 staging 验收（2026-09-09）：** 在工作区外以真实的 X2 模型权重、cursor head、85-I/O
cursor-enabled TRT plan 和实际文件 SHA-256 组装了
`/home/rime/workspace/models/x2-speech-state-bundle-custom-1.7b/`。该 staging bundle
包含 model/runtime fingerprint、X2 model contract、Code2Wav layout、hidden-tail bridge
以及 source/export copy hash；`validate_speech_state_bundle()` 返回 `verified`，真实
Executor 也完成 cursor ABI 加载。它仍没有 `capability_evidence.json`，所以 Native 和
Triton compatibility layer 都一致返回 `speech_state.supported=false`、
`reason=release_evidence_missing`；这证明了 H2 的“真实包校验与 fail-closed 投影”路径，
不证明训练出的 speech-state 等价性，也不打开发布能力。该 contract 的 transfer 当前
明确标为 `reconstruction_approximate`，必须等模型/训练方提供精确状态合同和发布 evidence
后才能升级。

**外部模型输入核对（2026-09-09）：** 已确认公开发布的实际模型标识为
`x-square-robot/X2Streaming-TTS-1.7B`，对应游标头为
`zehan1/X2-NativeCursor-Qwen3TTS-12Hz`；两者不是旧记录中曾使用的占位仓库名。游标头
`qwen3_tts_12hz_la1_seed0.pt` 已在工作区外下载并按发布清单校验，SHA-256 为
`aa94527fddff97cca0c337529a3b840b141d8301c4a6082557984438a7e584bc`；checkpoint 元数据
声明 `labels_version=labels_v3@2026-08-27`、`left_context=26`、`right_context=1` 和
503-label vocabulary；导出器按实际膨胀卷积受感野生成运行时 `history_width=30`、
`left_context=29`、`right_context=1`，配套模型配置声明 codec vocabulary 为 3072。主模型目录同时包含标准 Qwen3-TTS 配置与
`speech_tokenizer/`，因此 `export_09` 已修正为“模型自带 tokenizer 优先、旧的共享
tokenizer 目录回退”，并由 `test_native_cursor_modules.py` 锁定这两个路径合同。引擎侧新增
`NativeCursorLabelizer`，从同一游标头的 vocab 构造 503-label plan；server 只有在 executor
已加载 cursor graph 且 vocab fingerprint 校验通过时才注入 `CursorLabelPlanAdapter`。这解决
了“测试可注入 labelizer、生产没有 labelizer”的 T2 缺口，但不等于真实 cursor-enabled TRT
bundle 已验收。
主模型 `x-square-robot/X2Streaming-TTS-1.7B` 已完成下载清单校验，`Qwen3TTSModel` 可以从该目录
加载；模型自带 speech tokenizer 也已被真实导出路径使用。工作区外的
`x2-exported/custom-1.7b/` 已完成带 native cursor 的 ONNX 导出，PyTorch 与 ONNX 在
base/alternate-prefill 两种输入下的 Talker、Code2Wav、cursor recurrent state 和
`codec0` 输出逐项通过；`triton_manifest.json` 的 cursor vocab SHA 为
`48cd0c9cb433257706a2ac328221ddfa6e6936ed5179748dcebf6472425434bc`，与运行时
labelizer 一致。TensorRT 真实构建也已完成，engine 反序列化后为 85 个 I/O，包含 12 个
cursor 输入、11 个 cursor 输出和 `codec0`，完整 recurrent ABI 可以被 Executor 发现。
对应的环境门控证据为
`tests/integration/test_real_native_cursor_trt_artifact_contract.py`，当前 `1 passed`。
这证明了 H2 的“模型权重 → cursor-enabled TRT 图 → manifest → runtime ABI”子链，但原始
导出 manifest 不是 speech-state release bundle：它尚无真实 model/runtime fingerprint、
X2 `SpeechStateModelContract` 或 `capability_evidence.json`。外部 staging 已补齐这些
结构字段并通过真实 verifier，但仍因没有发布 evidence 而 fail-closed。successor trajectory
已由真实 X2 E2E 和 cursor recurrent restore 测试部分覆盖；H2 的精确状态继承合同、完整
不中断/继承数值对照和 H5 发布证据仍未完成，生产 capability 继续 fail-closed。

2026-09-09 对工作区现有 `workspace/exported/custom-1.7b/` 做了真实 TensorRT
artifact 检查：`talker_code2wav_fused.engine` 可以在当前 RTX 5090/TensorRT 环境反序列化，
共有 62 个 I/O，包含标准 Talker/C2W state 输入输出，但没有任何 `cursor_*` binding；其
manifest 也没有 `native_cursor` 或 `speech_state` section。对应的环境门控测试
`tests/integration/test_real_trt_artifact_capability_contract.py` 通过；同一 plan 的
`tests/integration/test_real_trt_state_transfer_contract.py` 还验证了真实 prefill 后
detached bundle restore 到新 slot 的下一步 token、PCM、`codec_sum`、token counts、C2W
conv 和 transconv 逐项一致，并确认 Executor/Native 与 Triton 对同一 package 的
speech-state capability reason 一致。因此这套 artifact 证明了 standard plan 的真实加载、
下一步状态输出保持和 fail-closed 路由，但不是 X2 speech-state bundle，也不能作为 native
cursor/TRT 融合验收证据。

**完成证据：** 完整 bundle manifest、源权重与导出副本 hash、加载/拒绝矩阵，以及
`Executor.speech_state_capability` 与 public gateway capability 的一致性测试。

**H2a：统一 package 路径来源（已完成接线；生产 X2 release bundle 仍待输入）。**
`resolve_model_package_paths()` 返回的权威 `package_dir`、`manifest_path` 和
`runtime_artifact_path` 现在由 Server 传入 Executor，并由 Executor/Triton 共同使用；旧的
直接 `engine_dir` 调用才保留 runtime 目录推断兼容。因此平铺包、默认 runtime 和自定义
嵌套 runtime 不再依赖目录名称猜 bundle root。
Owner 为 `engine/server.py`、`engine/backend/executor.py`、
`engine/gateway/triton_realtime_server.py` 的包装载接线；复用既有 `ModelPackagePaths`，
不得另造路径解析规则。验收必须覆盖平铺包、默认 `runtime/`、自定义嵌套 runtime/manifest
和 runtime artifact 路径，验证两入口读取同一 manifest、同一 evidence、同一 bundle root；
缺失或越界 artifact 继续 fail-closed，旧的直接 `engine_dir` 调用保持兼容。

**验收结果（2026-09-09）：** Executor 单测覆盖平铺包和自定义嵌套 runtime 的实际 plan、
manifest 与 bundle root 选择；Triton 单测覆盖平铺、默认 `runtime/` 和自定义嵌套目录，并
验证包根 evidence 损坏时不会回退读取 runtime 下的另一份 evidence。全量回归通过。真实
X2 bundle 到位后仍需补充 artifact hash 与实际加载 plan 的数值/连续性验收。bundle verifier
现在还接受 runtime artifact 的实际路径并拒绝 `runtime_plan_path_mismatch`，因此 manifest
校验的 plan 与 Executor/Triton 选择的 plan 必须是同一解析文件。export builder 和 manifest
JSON Schema 也已与 verifier 对齐：artifact hash 必须是严格的 64 位十六进制字符串，所有
artifact path 必须是包内相对路径且不能含 `..`，model weights 和 cursor head 必须同时
声明 source path/source hash；不完整 bundle 在导出阶段即失败。

### H3：X2 successor admission 与连续性 E2E

**Owner：** `engine/backend/engine_loop.py`、X2 adapter、`tests/integration/`。

**依赖：** H2；必须使用真实 X2 checkpoint，不能只用 `CausalSpeechStateInheritance`
的 CPU fake policy。

**范围：** 覆盖 serial admission、batch cache-hit、前段 EOS、文本完全消费、C2W
warm restore、hidden-tail bridge、native cursor state/reanchor、retry、取消和 hard-boundary
fallback；验证 successor 不重复首帧、不提前 admission、不把 EOS PCM 带入下一段。

**当前进展（2026-09-09）：** CPU lifecycle 已覆盖前段 EOS 后 successor gate 放行、serial
restore 失败后的 hard-boundary fallback、cancel/session replacement、retry capture reset 和
cache-hit batch gate；warm-state restore 已改为事务式 slot 提交，并修复了 successor 已被
普通 prefill 提升为 pooled C2W slot 后，继承状态误写到 detached field、实际 decode 仍读取旧
pool row 的问题。真实 CPU `KVCachePool` round-trip 和 EngineLoop 顺序回归分别锁定了
decode owner row 以及 successor 的 `prefill → warm restore → adopt` happens-before。设置
`X2STREAMING_ROOT`
后，`tests/integration/test_x2_method_layer_contract.py` 会加载真实外部
`X2StreamingPolicy`，full/long/token 和不完整长文本 case2 当前共 4 项通过；另有
`test_streaming_tn_cursor_plan_contract.py` 验证主 TN 到 Label Plan 的队列时序。以上只证明方法层与调度边界，尚未
证明真实 TRT checkpoint 的 codec0/C2W/PCM 连续性，也没有打开真实 successor native cursor
state migration；当前已补齐 engine-owned gated `CursorContinuationState` capture/restore、
外部 C2W context overlay 和完整 recurrent binding 检查，但没有把它误当成模型能力。
EngineLoop 现在额外要求 immediately-predecessor 已进入 `done` 才允许消费 continuity
context；predecessor 尚未结束时仍可按既有 hard-boundary 并行 admission，但不会提前恢复
C2W/cursor 状态。
C2W/cursor payload 现在按 predecessor segment 保留，successor terminal、retry、policy
disable 或 session removal 时清理，避免长文本会话按 segment 线性累积递归 tensor。

2026-09-09 已使用真实外部 X2 方法层、真实 cursor-enabled TRT plan、模型 tokenizer 和
`robot_service_v1` 完成三段 successor E2E。新增的环境门控测试
`tests/integration/test_real_x2_successor_e2e.py` 复现命令并断言：音频字节数大于零、共
3 个 segment、所有 segment 都携带 continuity、主 TN 最终文本为
`今天温度二十五摄氏度。明天降至十八摄氏度，请注意保暖。`，后两个 successor 的
`prefill_done` 同时带有 `extension_continuity=restored`、`extension_bridge=prepared` 和
`cursor_progress=native_continuation`。本次实际运行结果为 `1 passed`，三个 segment 的
`eos_reason` 均为 `codec_eos`，没有重复首帧或 cursor labelization 失败。

2026-09-09 使用相同真实模型、TRT artifact 和 tokenizer 重跑该 E2E，连续性路径与 hard
boundary A/B 均通过，结果为 `2 passed in 16.87s`。该结果确认主 TN、Splitter、C2W bridge
和 cursor continuation 的运行时路径仍稳定；最终产品质量仍需对服务输出音频执行离线 ASR
对照，不把 token 级 codec 差异当作失败。

同日新增的 `tests/integration/test_real_native_cursor_state_transfer.py` 又在同一真实
TRT plan 上直接验证了 fused cursor recurrent state：两个 slot 先共同推进一帧，随后把
源 slot 的六组 cursor recurrent tensor detached，清空目标 slot，再经
`restore_cursor_state()` 恢复；下一次 decode 的普通输出和全部 cursor 输出逐项相同，结果
为 `1 passed`。这项证据补足了“事件显示 native continuation”与“图恢复后下一步轨迹一致”
之间的缺口，但它固定了 Talker/C2W 状态，只证明 cursor state ABI/图内递归恢复，不等于
完整 X2 speech-state successor 的不中断数值对照。

随后新增的 `tests/integration/test_real_native_cursor_full_state_transfer.py` 在同一真实
cursor-enabled fused plan 上验证完整 pooled bundle：Talker KV、C2W KV、conv/transconv、
slot auxiliary、采样 RNG 和六组 cursor recurrent state 一起 detached/restore；恢复目标
slot 的下一次 fused decode 在 token、PCM、hidden、`codec_sum`、token count、C2W 和 cursor
输出上逐项一致；eager 和 CUDA Graph 两条路径均通过，结果为 `2 passed`。这补足了 S2 的真实
full-state round-trip；它仍
是同 segment/同 checkpoint 的 PAUSE_RESUME 证据，不是 X2 跨 segment 的训练语义或 successor
音频质量证据。

2026-09-09 使用当前 `x2-exported/custom-1.7b` 和 `qwen3-tts` 虚拟环境重新执行该测试，
结果仍为 `2 passed in 4.39s`；同日重新执行
`tests/integration/test_real_native_cursor_cuda_graph_parity.py`，结果为 `2 passed in
5.05s`。这次复验确认状态恢复和 graph/eager 路由没有因后续 package path、TN 或发布
gate 改动回归，但仍不替代跨 segment successor 的 ASR/音频质量验收。

随后新增的 `tests/integration/test_real_native_cursor_trt_trajectory.py` 固定真实 graph
每步产生的 `codec0`，把同一 token、Label Plan 和 pre-step cursor state 喂给模型自带的
PyTorch `CursorStreamingStep`，连续比较 32 帧。`cursor_valid`/`candidate_label`/帧计数
逐项精确一致，bf16 浮点 recurrent 输出在显式 `rtol=0.05/atol=0.08` 预算内，真实运行
结果为 `1 passed`。这把 H3 从 ABI/状态恢复推进到了真实 TRT cursor 子图轨迹；但它仍固定
Talker/C2W 只验证 cursor 子图，尚未覆盖完整 PyTorch→ONNX→TRT fused graph、PCM/C2W
和 successor 不中断对照。

同日新增的 `tests/integration/test_real_native_cursor_cuda_graph_parity.py` 对同一真实
cursor-enabled fused plan 做了 batch 1 和 batch 2、各四步的 graph/eager 对照。此前把
profile 1 graph 与 profile 0 eager 跨 profile 比较，首步 `codec_sum` 的 BF16 分叉被误判为
profile 1 不可用；改为让 graph/eager 都使用 profile 1 后，token、codec sum、Talker hidden、
C2W KV、cursor 输出和 PCM 均通过。Executor 现在对 standard 和 cursor plan 默认使用
decode-only profile 1；prefill 仍由共享 context 使用 profile 0。该项补上了“同 profile 能
replay”与“递归状态轨迹保持一致”之间的边界，但不等于完整 Talker/C2W/PyTorch
数值验收，profile 1 的显存和吞吐报告仍属于 H5 待完成项。

同日对真实 fused ONNX 做了一次冻结 decode 输入探针：当前 artifact 的 `codec0` 与
cursor 离散有效输出和 TRT 一致，但 CPU ORT（float32）与 bf16 TRT 的 residual
`full_codec`、hidden、C2W 浮点轨迹出现明显差异（本次单步最大误差分别约为 1570、0.994、
1.476）。这些是数值诊断结果，不能直接等同于音频或语义回归；BF16 已验证为默认基线。
发布前置条件应改为固定语料的 ASR CER/WER、音频连续性/终止行为、cursor 状态和事件
合同，而不是要求 full_codec 逐 token 精确。

为验证精度假设，先在隔离的外部 staging 目录用现有 Phase-B 构建链生成了
`cp=fp32`、backbone/Code2Wav 仍为 BF16 的对照 plan（TensorRT 10.13，2 个 profile，
构建耗时约 310 秒，engine 约 3.86 GiB）。新增的
`tools/validation/fused_onnx_trt_parity.py` 随后对同一冻结 decode 输入做了 ORT/TRT
三方比较：相对 ORT，`cp=fp32` 将 `codec_sum` RMS 从 `0.0524` 降到 `0.0292`，C2W
从 `0.1889` 降到 `0.1033`，PCM 从 `0.2340` 降到 `0.0726`；但 hidden/logits
保持原差异，`full_codec` 仍有 4 个 codebook 不一致。这证明 per-submodule precision
可以改善下游，但不能单独解决 full fused parity。

随后又构建了全 FP32、小 profile 的 cursor-enabled plan（TensorRT 10.13，2 个 profile，
engine 约 7.14 GiB）。此前的 5 步报告是用 BF16 capture artifact 产生输入，再送入
FP32 TRT 的冻结图对照，不能当作 FP32 自身 prefill→decode 轨迹。改用与 capture engine
相同的 FP32 dtype、固定 profile 0，并关闭 TensorRT TF32 后，PyTorch→ONNX→TRT 的
`full_codec` 5/5 精确，hidden RMS 约为 `9.4e-6~6.2e-5`；这证明原先 FP32 CP 分叉
来自 TF32 builder 路径，但 FP32 仍只是数值定位对照。正确 dtype 的 BF16 自捕获 5 步
每步都有 residual codebook 分叉，hidden RMS 约为 `0.079~0.430`；额外构建的
`cp=fp32`、backbone/Code2Wav 为 BF16 方案仍有 `7/12/10/1/15` 个 `full_codec`
mismatch（codec0 5/5 一致）；这属于数值对照结果，不能直接解释为质量失败，也不能把
CP 单点升精度当作发布解。BF16 默认 plan 的最终判断应以固定语料 ASR CER/WER、音频
连续性/终止行为和 cursor/事件合同为准。因此 T4 仍需补齐 ASR/音频报告、lookahead/
EOS/flush 矩阵和性能代价评估。

为补齐这条证据链，`fused_onnx_trt_parity.py` 现支持显式传入
`--torch-model /path/to/source-checkpoint`，使用导出器自身的 PyTorch fused wrapper
在相同冻结输入上输出 `torch_vs_onnx` 和 `torch_vs_trt`。入口已经过语法与 CLI 检查，
并新增 `--output` 写入无 stdout 日志污染的 JSON，以及按 manifest 自动选择 cursor profile
0 的逻辑。该结果仍不能归档为 T4 通过证据：执行需要完整源模型、model-owned cursor
head、无冲突的显存资源和正式报告保存路径，当前结果还暴露了 CP 逐帧分叉，不能用已有
ONNX exporter 的两组 smoke verification 替代逐帧 prefill/decode/EOS 报告。

同一 RTX 5090、TensorRT 10.13、profile 0 decode shape（`c2w_attention_bias=1x1x1x2`）
下又做了 `trtexec --useCudaGraph --noDataTransfers` microbenchmark：BF16 median
`8.039 ms`/`124.38 qps`，全 FP32 median `11.247 ms`/`88.91 qps`，FP32 约增加
`39.9%` 延迟、降低 `28.5%` 吞吐。这只是 graph decode 的 GPU microbenchmark，尚未包含
EngineLoop batching、KV gather、CPU 编排、Gateway 或 Triton 兼容层成本，因此只能作为
精度方案选择输入，不能直接作为 H5 发布性能报告。

新增并执行 `tests/integration/test_real_x2_successor_e2e.py` 的真实 A/B 验收（`2 passed`）：
固定三包输入和同一 X2 checkpoint，启用 continuity 时产生 3 个 segment，后两个 successor
均有 `extension_bridge=prepared` 和 `cursor_progress=native_continuation`，输出时长约
6.56s；关闭 X2 时为 1 个 hard-boundary segment，输出时长约 5.20s，且两次都生成了
真实 WAV。该结果证明继承路径实际改变了声学执行路径，但不是不中断音质等价、click/F0/
energy 或发布 evidence。

H4 的真实兼容层证据也已补齐：环境门控测试
`tests/integration/test_real_cursor_capability_projection.py` 使用同一真实 cursor-enabled
TRT package，分别经过 Native Executor 和 Triton compatibility layer 的 manifest/runtime
读取与公共 projection，断言 `native_cursor`、`speech_state`、progress modes 和 disabled
reason 完全一致，结果为 `1 passed`。该测试只验证兼容层合同，不把 Triton 当成第二种 TRT
推理 backend，也不把 capability projection 当成 speech-state 已发布。

这次真实联测还修复了两处由“流式 TN 与原生游标耦合”暴露的 bug：

1. 单个数字尾巴既可能继续组成中文数字，也可能成为 keycap emoji 的组成部分；原先
   80ms semantic deadline 会把 `25` 拆成 `2` 和 `5`，导致主 TN 输出 `二5` 并使 cursor
   labelizer 降级。现在单数字 pending span 在收到后续输入或显式 final 前保持
   `WAIT_TEXT`。
2. `℃` 经兼容归一化会变成 `°C`，backend spelling 比 raw span 更长；原先详细 mapping
   会越过 raw owner span，触发 cursor plan 拒绝。现在没有可证明的字符对齐时回退到原始
   owner 区间，仍由主 TN 作为 spoken-form 真相来源。

该 E2E 是真实 TRT、主 TN、Splitter、C2W bridge 和 cursor continuation 的运行证据，但
不是 speech-state capability 的发布证据：用于这次 E2E 的原始真实 manifest 虽已补齐构建
profile `max_batch_size=8/max_input_len=128/max_seq_len=512`，仍没有完整的 speech-state
release contract/evidence。另有外部 staging bundle 已验证 package contract，但因缺少
`capability_evidence.json` 仍保持 fail-closed，不能替代发布包。除此之外，该 artifact 的
Code Predictor 使用 bf16；其中 codebook 分叉属于正常数值波动，不再要求改用 fp32 才能
判断产品质量。因此公共 native progress/state-transfer 继续保持 fail-closed；仍需完成
固定语料 ASR/音频验收、不中断/继承对照、性能报告和 Native/Triton 同结果集验收。
本轮又修复了两个 stale/partial handoff 边界：重复 `START_TOKENS` 替换旧 segment 时会删除
该 segment 已发布的 cursor payload；cursor recurrent payload 在 C2W warm restore 前先做无副作用
ABI preflight，预检失败不会留下“C2W 已恢复、cursor 未恢复”的半状态。对应的 retry、替换
隔离和 preflight 回归测试已通过。EngineLoop 的 bundle restore 事务现在也把 callback
返回 `False` 视为拒绝，只有无异常且非拒绝结果才消费 bundle；失败仍可走显式 cleanup 或
hard-boundary fallback。cursor restore capability/preflight 现在在 C2W mutation 之前完成，
不可用时不会先写入 predecessor 的 C2W 状态；`CursorContinuationState.restore_into()` 也改为先完成全部字段
ABI 检查和 detached clone，再一次性提交，晚到的字段错误不会留下前面 recurrent 字段已写入
的半状态。
C2W-only context 仍显式走 EMA；同
segment 的 tail rewrite reanchor 已由 owner-span 合同和 executor
override 输入覆盖，但它不等价于跨 segment cursor state handoff。另有单元合同明确证明：
即使 successor 命中 prefix cache，存在 continuation context 时也不能走 batch cache-hit，
必须留给 serial restore 路径，避免 prefix KV/C2W zero-state 覆盖继承状态。
本轮还收紧了 native successor 的准入：`CursorContinuationState` 只有在主 streaming TN
已经为同一 successor 交付 `CursorLabelPlan` 时才允许保持 native；只有 recurrent state
而没有 Label Plan 必须降级 EMA，不能让 fresh slot 的默认 label buffer 冒充 spoken-form
来源。该规则覆盖了流式 TN 与原生游标的关键耦合，但真实模型仍需证明 plan/state 在首个
decode frame 前的同一轨迹上生效。
如果 C2W 或 cursor restore 失败，EngineLoop 还会在进入首个 decode 前显式清空该 slot
的 label plan 并置 `active=false`；这避免“CPU 已报告 EMA、TRT 仍执行旧 native plan”的
半降级状态。该失败路径已有顺序合同测试，但仍属于模型无关的 CPU 安全闸门。

**完成证据：** 不中断与继承两条轨迹的 codec0、C2W、PCM、cursor 和高水位对照，
click/F0/energy/音色/文本覆盖指标，以及 Native Gateway/Triton 兼容层同结果集。

### H4：统一能力与 WebUI 演示闭环

**Owner：** gateway capability、统一 WebUI demo、E2E tests。

**依赖：** H3 通过；否则 UI 只能展示 `EMA/DISABLED`，不能伪装 native/state available。

**范围：** capability discovery 明确区分 `graph_enabled`、`progress_available`、
`speech_state.supported` 和 `speech_state.reason`；WebUI 演示 full/long/token 三入口、
EMA/native progress、hard-boundary/continuity fallback，并显示真实 lifecycle event。

**当前进展（2026-09-09）：** capability 合同已经由 SDK 类型和 schema 解析保留
`native_cursor`、`speech_state`；统一 WebUI 的体验页新增 Engine Route 状态条，分别展示
cursor graph 是否加载、native progress 是否准入、speech-state 是否启用及 fail-closed reason。
Realtime SDK 现在保留 `qwen.text_progress.meta`，体验页按实际事件的
`progress_basis=native_cursor_v1` 或 `ema_frame_ratio_v1` 标出本次会话的原生游标/EMA
路径；没有事件时显示等待或未开始，不从静态 manifest 猜测。该部分已通过 browser SDK
31 项测试、demo 12 项测试、TypeScript 检查和 production build。真实 GPU/TRT 的
continuity fallback、断连/resume 录制和 lifecycle event E2E 仍依赖 H2/H3 的真实模型
bundle，尚未宣称完成。case2 方法层合同测试验证了主 TN 的 pending tail 语义：
`99%` 单独输入时不进入 Splitter/声学请求，后续中文上下文到达后才提交
`百分之九十九，我目前`，显式 input complete 不重复提交。该测试通过真实外部
`X2StreamingPolicy` adapter，但仍是 CPU 方法层证据，不替代 TRT/X2 bundle 的连续性验收。
另一个 cursor-plan 跨层合同测试进一步验证了 Label Plan revision 先于同批 token request
进入 engine queue。
真实 standard TRT package 的 Native/Triton capability 对照又发现并修复了一个 public
wire 问题：Native 内部的 typed speech-state descriptor 不再原样泄漏到公共 JSON，而是与
Triton 一样统一投影为 `supported/reason`；同一真实 package 的 `native_cursor` 与
`speech_state` public projection 已逐字段一致。该证据仍只证明标准 plan 的 fail-closed
路由，不证明 X2 successor continuity。
公共 gateway projection 另外对 `native_cursor` 做了 fail-closed 归一化：只有严格的
boolean `enabled=true` 且 graph 已加载时，`progress_available=true` 才能进入公开 native
模式；不一致或字符串/非布尔 truthy 值都降为 EMA/DISABLED。`speech_state.supported`
也拒绝整数、数组等非布尔 descriptor，避免 malformed manifest 绕过 H4 的 capability
boundary。

2026-09-09 跨层验收修复了 Server 的属性/方法合同错配：Executor 的
`native_cursor_capability` 与 `speech_state_capability_reason` 是属性，Server 原先只接受
callable，导致已加载的 cursor graph 被报成 disabled，具体 bundle 拒绝原因被
`adapter_disabled` 覆盖。现在 capability discovery 和 health stats 均读取实际属性值，
同时兼容旧的 method-based executor。新增 Server/Executor 跨层回归在修复前复现 6 项失败；
真实 standard TRT 测试也已改为经过 `TTSEngine.describe_capabilities()` 再比较 Native/Triton
public projection，并通过下一步状态数值对照。它仍不代表真实 cursor-enabled 或 X2
successor E2E 完成。

新增的环境门控 `tests/integration/test_real_cursor_capability_projection.py` 已在真实
cursor-enabled TRT package 上验证 Native Executor 与 Triton compatibility layer 的公共
projection 逐字段一致：两者都报告 graph 已加载、native progress 未准入、EMA/DISABLED
可用，且 speech-state 保持 disabled。该结果确认了 H4 的统一接口边界，但仍没有把 Triton
误当作另一种 TRT 推理 backend；Native/Triton 的音频结果集、断连/resume 录制和正式发布
截图仍是待完成证据。

同日补齐了相邻的音频可靠性缺口：`AudioReorder` 现在对当前 playhead 的阻塞段单独计时，
后段音频不会因超时被越序释放；Frontend watchdog 超时后发布
`audio_reorder_stall_timeout` 告警，并通过内部取消请求让 Engine 统一释放 slot、held audio
和 session。该恢复路径已有 CPU 异步合同测试，但真实多 session/高并发下的取消延迟和
持有音频上限仍属于 H5/T10 待验收项。

**完成证据：** Native Gateway 和 Triton 兼容层截图/录制、协议快照、断连/resume 和
failure fallback E2E；不能把 transport resume 当 speech-state restore。

### H5：性能与发布闸门

**Owner：** backend/runtime、gateway、发布脚本。

**依赖：** H3/H4。

**范围：** 分离 TRT 主图、cursor 融合、EngineLoop admission、state capture/restore、
Native Gateway 和 Triton 兼容层开销；记录串行 successor admission 对 TTFT/吞吐的影响。

**当前进展（2026-09-09）：** 新增 `engine.runtime.release_gate.evaluate_release_gate()`，以
版本化的 `capability_evidence.json` 作为唯一发布证据入口。证据必须与 manifest 的
`speech_state.model_fingerprint`/`speech_state.runtime_fingerprint` 相同，且 manifest 和
evidence 两侧身份都必须非空；缺少 manifest identity 时即使 evidence 完整也必须关闭。native cursor 需要 TRT 数值、progress E2E、
单调性和性能证据，speech-state 需要 bundle、TRT transfer、successor E2E 和性能证据。
standalone Executor 和 Triton 兼容层都复用同一判定，manifest 单独声明
`progress_available=true` 不再足以对外开放 native；证据缺失时保持 EMA/DISABLED。可用
`scripts/python/validate_capability_evidence.py` 在发布流水线中检查，并用
`--require-native-cursor` 或 `--require-speech-state` 将缺证据变成发布失败。当前已有真实
X2 checkpoint、cursor-enabled TRT graph、recurrent restore 和 successor E2E，但仍缺完整
GPU 逐帧轨迹、不中断/继承对照和性能报告，因此发布报告仍不能声称完整生产验收。需要
区分两类状态：release evidence 继续决定发布报告是否通过；而已完成 graph/head/labelizer
runtime admission 的 native cursor 可以向在线 Gateway 公开，不能因为离线 ASR/性能报告
未完成而把已运行的 progress route 隐藏成不可用。H5 gate 还对
`schema_version` 做严格整数校验，`true`、`1.0` 等 Python 中与 `1` 宽松相等的值会保持
fail-closed，避免 malformed evidence 绕过发布闸门。

2026-09-09 补充了外部 X2 方法层的性能/稳定性基线：
`benchmark_bridge_cuda.py` 在 RTX 5090 上执行 1000 次 text-acoustic bridge，median
`204.929us`、p95 `216.504us`、峰值显存 `8,606,720` bytes；
`stress_x2streaming_cuda.py` 的 100,000 token commitment stress 和 2,000 segment
inheritance stress 均通过，后者耗时 `0.872s`、峰值显存 `10,713,600` bytes。这些数据只
证明 X2 方法层 bridge、状态清理和 bounded-memory 行为，不包含主 TRT decode、cursor
graph、EngineLoop batching、Native/Triton transport 或 successor 音频质量，因此不能直接
填入 `performance.verified=true`；完整固定 profile 性能报告仍待完成。

`capability_evidence.json` 的最小结构为：

```json
{
  "schema_version": 1,
  "model_fingerprint": "<manifest speech_state.model_fingerprint>",
  "runtime_fingerprint": "<manifest speech_state.runtime_fingerprint>",
  "performance": {"verified": true},
  "quality": {"offline_asr_verified": true},
  "native_cursor": {
    "trt_numeric_verified": true,
    "progress_e2e_verified": true,
    "monotonic_verified": true
  },
  "speech_state": {
    "bundle_verified": true,
    "trt_transfer_verified": true,
    "successor_e2e_verified": true
  }
}
```

这份文件只能由固定模型、输入、batch、GPU、精度和 CUDA Graph profile 的实验产出，不能
由 WebUI、请求参数或人工把 manifest 字段复制成 `true` 生成。

其中 `quality.offline_asr_verified` 是离线生成质量闸门：启动服务取得音频后，用固定语料、
固定参考文本和固定 ASR 评测配置检查识别结果，再把报告固化进 evidence。它不会让线上
服务在每次合成后调用 ASR；`release_gate.py` 只读取布尔证据，不联网也不依赖 ASR。
`full_codec` 的逐 token mismatch 只作为数值诊断字段，不替代 ASR 结果；ASR 之外仍需保留
音频连续性、EOS/终止行为、cursor 状态和事件顺序的结构性验收。

2026-09-09 又收紧了发布闸门的身份边界：manifest 与 evidence 的
`model_fingerprint`/`runtime_fingerprint` 必须是非空字符串，数字、布尔值和容器不会被
隐式转换后参与匹配；公共 speech-state 投影同样只接受真正的布尔 `supported`。Triton
兼容层在生成运行时能力前还会拒绝 malformed native-cursor progress flag，避免 Native
与 Triton 因 truthy metadata 产生不同的准入结果。

**完成证据：** 固定模型、输入、batch、GPU、精度和 CUDA Graph profile 的可复现实验报告，
并把 capability 开放条件写入发布检查，不满足时自动保持 EMA/DISABLED。

**离线 ASR 验收任务：** 另行运行已启动的服务，使用固定 full/long/token 语料和参考文本
录制输出音频，再调用项目约定的 ASR 接口生成 CER/WER 与逐条文本对照报告。该任务属于
发布前的离线验收，不得把 ASR client、网络请求或识别模型接入 EngineLoop、Gateway 或
在线请求路径；只有报告人工/脚本确认通过后，才写入
`quality.offline_asr_verified=true` 并参与 release gate。

**本轮真实样本结果（2026-09-09）：** 已用真实 X2 successor 服务生成
`workspace/validation/speech_state_current/x2-continuity.wav`，输入为三段跨 segment
文本 `今天温度25`、`℃。明天降至18`、`℃，请注意保暖。`，服务侧最终 spoken form 为
`今天温度二十五摄氏度。明天降至十八摄氏度，请注意保暖。`。使用固定 ASR 接口
`wss://infer.x2robot.com/infer/inf-dddq5qn77jrws5eu/v1/ws` 离线识别得到
`今天温度二十五摄氏度，明天降至十八摄氏度，请注意保暖。`，`stream_done.reason=client_stop`，
音频时长 `6560ms`，结果与预期 spoken form 仅有中文标点差异。本次样本通过音频语义验收；
它不等于 full/long/token 全语料、Native/Triton 双入口和正式发布 evidence 已经全部完成，
后续仍需补齐这些维度后再打开 release gate。

**历史验收实现参考：** `origin/feat/native-cursor` 的
`tools/validation/native_cursor_demo.py` 已提供真实引擎 WebSocket 无头验收：按
LLM 节奏分块发送文本，同时收集 WAV 和 `text_progress` 锚点，检查 progress basis、
raw 游标单调性、末尾游标、音频终态，并支持并发压测。当前分支没有直接合入该脚本，
原因是本分支已将 cursor 计算融合进主 TRT 图、协议字段和 TN owner 映射均已变化；后续
Native/Triton 双入口验收应沿用它的“真实服务 + 音频 + 事件轨迹”方法，但改用当前
`InputMode`/主 TN 合同，禁止退回 token parity 或旧的 CPU cursor 路径。

## 7. 当前完成标准

S1 的 CPU 合同通过只表示后续实现有安全接口。S2 的真 tensor snapshot、S3 的模型语义与 S4 的运行时状态机均未完成时，不得把 T7/T8 标为完成，也不得把传输 resume 或 WAIT_TEXT 当作替代验收。
