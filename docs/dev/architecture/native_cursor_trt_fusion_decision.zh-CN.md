# 原生游标、TRT 融合与流式 TN 坐标投影决策记录

> 日期：2026-09-04
> 状态：已决策；作为首版实现约束
> 范围：Qwen3-TTS TRT 导出、推理运行时、原生游标、流式 TN 与文本进度协议
> 关联：[流式文本消歧、增量文本规范化与 Soft Drain](../design/incremental_text_normalization_and_soft_drain.zh-CN.md)

---

## 1. 背景

远端分支 `feat/native-cursor` 提供了一个约 2M 参数的原生游标模型。模型观察
Qwen3-TTS 每个声学帧的 codebook-0 token，并在当前可见的口语标签附近执行局部
matcher，从而得到连续的 label 坐标 `mu` 和文本进度。

该分支的 `NativeCursorSession` 同时承担了以下职责：

- 文本追加与 label 重建；
- codec0 窗口维护和游标神经网络推理；
- TN tail rewrite 后的游标重锚定；
- 单调 public high-water；
- normalized/raw 坐标投影；
- 一帧 lookahead 和 final flush。

生产集成不能把上述职责整体原样搬入 TensorRT，也不能为“支持官方 TTS”引入
官方 PyTorch 模型作为另一个 backend。需要明确哪些是静态张量计算，哪些是动态
文本语义和协议状态。

## 2. 核心决策

### D1. 官方 TTS 支持指 TRT 图兼容，不新增官方 PyTorch backend

官方 Qwen3-TTS 模型继续走本项目已有的 TRT Talker、Code Predictor 和 Code2Wav
推理路径。官方模型没有随附本项目训练的 2M 游标头时，运行时必须使用不含游标头的
标准 TRT plan，并将文本进度降级到 EMA 或其他已实现的估计器。

禁止把官方 PyTorch 推理包装成所谓“官方 backend”来解决兼容问题。

### D2. 2M 游标模型的全部神经网络计算融合进 cursor-enabled TRT 图

首版为已经验证过的 `custom-1.7B` 构建独立的 cursor-enabled TRT plan。游标头的
权重作为图常量进入 ONNX/TRT artifact，不在 CPU 侧运行 PyTorch sidecar。

一次 Talker decode 当前产生一个声学帧：

```text
Talker logits
  -> sample codec_token_0
      ├─> Cursor codec trunk + text encoder + matcher
      └─> Code Predictor -> full_codec -> Code2Wav
```

当前 `FUSED_CHUNK_T=1`，因此严格保持：

> 一个 codec frame 的 codebook-0 token，对应一次 matcher 状态更新。

`full_codec` 中同一帧的其他 codebook 不会分别触发 matcher。未来若
`FUSED_CHUNK_T=T`，图内必须按帧顺序展开 `T` 次 matcher 更新，不能只对整个 chunk
计算一次。

### D3. 首版不把文本 encoder 拆成独立的运行时 backend

外围 CPU 将已经 TN、labelize 的固定长度 padded `label_ids` 和 `label_count` 传入
cursor-enabled TRT 图。图内完成：

- label embedding；
- causal text convolution；
- codec0 embedding 和膨胀卷积 trunk；
- local candidate gather、mask、softmax；
- `mu`、`delta`、`confidence` 和神经状态更新。

首版允许每个 codec frame 在同一个 fused 图内重算当前可见文本的 text encoder。这样会产生
一部分重复计算，但不会引入第二个 TRT 调用、跨执行上下文的 `text_state` 同步或异步文本
更新竞态。2M 游标头相对 1.7B Talker 很小，但是否可以忽略仍必须用目标 batch 和文本
长度实测，不能仅按参数量推断。

这里的 `label_ids` 是从主 TN 产生的游标 label，不是当前 Talker 的 BPE/prompt
`input_embeds`。当前 backend 的 `input_embeds` 在 prefill 阶段由 `text_embedding +
text_projection` 构造，decode 阶段通常是上一帧的 `codec_sum`；它既不包含完整可见文本，
也不在 2M 游标头训练的 503-label、256 维表示空间内。因此不能直接把 Talker 的 embedding
当作已发布游标头的 text embedding。正确做法是增加 `cursor_label_ids` 输入，并在同一个
TRT 图内使用游标头自己的 embedding 权重。

只有性能数据证明 text encoder 重算成为显著瓶颈时，才允许把 `text_state` 改为 slot
缓存；该优化不得改变一个 codec frame 一次 matcher 的语义，也不得回退到 PyTorch。

### D4. TN、Unicode 和 raw 坐标永远不进入 TRT

TRT 图只接收数值 label 和游标神经状态，不接收原始 Unicode，不执行 TN，也不知道
raw codepoint span。

外围 CPU 负责：

- streaming TN 和 mutable tail；
- Unicode/codepoint 边界；
- spoken label 构造；
- label owner span；
- TN plan revision；
- raw/normalized 坐标映射。

线上必须以主 TN 层产生的 `TextCommit` 和 `CanonicalTextJournal` 为唯一语义来源。
不得把 `feat/native-cursor` 中的 `IncrementalExpander` 作为第二套在线 TN 与主 TN
并行运行。游标适配层只能消费主 TN 已经决定的 spoken form。

### D5. `NativeCursorSession` 拆成神经状态与 CPU 后处理

目标职责拆分如下：

| 组件 | 运行位置 | 职责 |
|---|---|---|
| `CursorTextPlanAdapter` | CPU | 从主 TN commit 生成 `label_ids`、owner spans、revision |
| `Cursor TRT branch` | GPU/TRT | text encoder、codec trunk、matcher 和连续神经状态更新 |
| `CursorReanchor` | CPU | tail rewrite 时通过 owner span 计算新的 label 坐标并注入 `mu` |
| `CursorProjector` | CPU | label 坐标投影为 normalized/raw 坐标 |
| `CursorPublisher` | CPU | 单调 high-water、展示插值、final 事件和协议字段 |

在主 TN 的单调提交语义下，正常情况中 mutable tail 的 rewrite 应发生在 commit 之前，
因此不会影响已交给 TTS 和游标的 spoken prefix。`CursorReanchor` 仍作为防御性能力保留，
用于 label plan 在当前游标附近确实发生 revision 的情况。

### D6. `codec0` 图内张量和显式 output 是两件事

游标 TRT 分支直接消费采样点产生的内部 `codec_token_0`，不需要先将 token 拷贝到 CPU，
也不依赖显式 `codec0` output 再进入第二次推理。

新图可以额外导出命名的 `codec0: [B]` binding。它不是给融合后的 matcher 使用，消费方是：

- executor 的 codec EOS 和循环停止判断；
- prefill 首帧状态记录；
- engine metrics、debug dump 和回放；
- 不含游标图或兼容旧实现时的 CPU 适配路径。

运行时采用兼容读取：

```python
codec0 = outputs.get("codec0")
if codec0 is None:
    codec0 = outputs["full_codec"][:, 0]
```

因此，显式 `codec0` 是 ABI 解耦和未来减少 host 输出的手段，不是融合游标的前置条件。
在保留 `full_codec` output 的阶段，两者语义完全等价。

### D7. 协议 high-water 与展示插值分离

`raw_codepoint_end` 和 `normalized_codepoint_end` 是保守、整数、单调的协议坐标。TN
字段发生重排时，例如 `99%` 的 spoken form 与书写顺序不同，协议坐标可以在整个 owner
完成后进行字段级跳跃，不能伪造“某个 raw 字符已经可靠读完”的细粒度语义。

UI 可以消费额外的 display-only 浮点坐标。在一个 owner 内：

```text
fraction = (mu - owner.label_start) / owner.label_count
display_raw = owner.raw_start + fraction * (owner.raw_end - owner.raw_start)
```

该插值只能用于高亮动画和进度条，不能用于计费、断点恢复、文本确认或音频 release
安全判断。如果未来主 TN 提供更细的稳定 owner spans，协议 high-water 可以自然变细；
第一版不强求伪造这种粒度。

### D8. 第一版只保证已经验证的 `custom-1.7B`

当前发布游标头与训练时的 Talker、codec、帧率、label vocabulary、采样策略和训练语音域
绑定。第一版只对明确验证过的 `custom-1.7B` 组合启用 native cursor。

标准 plan 与 cursor-enabled plan 必须由 manifest 显式区分，至少记录：

```text
model_variant
model/checkpoint fingerprint
codec_vocab_size
num_codebooks
native_frame_rate
cursor_head_sha256
cursor_vocab_sha256
cursor_rules_sha256
cursor_left_context
cursor_right_context
voice/training scope
```

### D9. FP32 fused plan 必须关闭 TF32

TensorRT 的 FP32 builder 默认可能选择 TF32 matmul。它虽然保持 FP32 I/O 和 manifest
精度字段，却会在 Code Predictor 的近 tie 采样处改变 CP codebook。2026-09-09 在同一
`custom-1.7B`、同一 profile 和同一冻结输入上复测：普通全 FP32 plan 第 3 个递归点有
10 个 CP codebook 分叉；加入 `--noTF32` 后 PyTorch→ONNX→TRT 的 5 步 `full_codec`
全部精确，codec0、cursor 输出和 hidden 轨迹也回到约 `1e-5~6e-5` RMS 范围。

因此 fused build 的默认规则是：`engine_dtype` 或任一显式子模块精度为 FP32 时自动传入
`--noTF32`；`TRT_NO_TF32=0` 只允许用于明确的性能实验，不能把结果作为数值验收或发布
evidence。该规则只约束 TRT builder，不改变 Triton 兼容层或 runtime backend 架构。

不匹配时必须禁用 native cursor 并走标准 TRT + EMA，不能静默加载一个“不报错但坐标不可信”
的游标头。官方 0.6B、Base、VoiceDesign 等组合在各自完成训练和验收前不承诺 native cursor。

### D9. 复用现有计算 backend，只扩展 fused graph 合同

概念流程是：

```text
原始文本 -> 主 TN -> spoken text -> 现有 Talker 输入准备
                                -> cursor label plan
                                -> fused TRT (Talker + Cursor + Code2Wav)
                                      ├─> wav 后处理
                                      └─> cursor 后处理/文本进度
```

实现上不重写现有 scheduler、KV cache、continuous/padded batching、Code2Wav 或 engine
FSM。只增加：

- cursor label/state 的固定输入 buffer；
- fused graph 内的 cursor 分支；
- cursor state/output 的 slot 转发；
- CPU 侧 cursor 坐标和事件后处理。

当前 Talker 的 `input_embeds` 不能直接充当已发布游标头的 text embedding：它来自官方
Qwen BPE/prompt embedding 与 projection，decode 时通常还是上一帧 `codec_sum`，既没有
完整的可见文本，也不在游标头的 503-label、256 维空间。游标 text embedding 必须使用
自己的 `label_ids` 和自己的权重，但可以在同一个 fused TRT 图内完成。

声学侧则可以直接把 Talker 在图内采样出的 `codec_token_0` 接到游标自己的 codec
embedding。不能把 Qwen 的 `codec_embedding` 权重直接替换游标 trunk 的 embedding，除非
重新训练游标头；两者维度和训练语义不同。

### D10. 用能力协商选择 graph route 和 progress route

“智能导图”不应表现为一个包含大量运行时分支的万能图，而应是**同一导出工厂生成多个
静态 plan，manifest 声明能力，运行时按能力选择路由**。

路由分成两层：

```text
Graph route（按模型 artifact/engine 能力选择）
  standard TRT plan
  cursor-enabled TRT plan

Progress route（按请求和会话选择）
  native cursor
  EMA
  disabled
```

导出工厂根据 `model_variant`、codec 配置和 cursor head fingerprint 生成：

| 模型/头能力 | graph route | native 请求 | EMA 请求 |
|---|---|---|---|
| `custom-1.7B` + 匹配游标头 | cursor-enabled | 融合图输出 + native 后处理 | 同一图，忽略游标输出或切标准 plan |
| `custom-1.7B` 无游标头 | standard | 按策略降级或拒绝 | 标准图 + EMA |
| 官方其他模型 | standard | 按策略降级或拒绝 | 标准图 + EMA |

请求策略建议使用枚举而不是自由字符串：

```text
AUTO    -> 有匹配 cursor capability 就 native，否则 EMA
NATIVE  -> 没有匹配 capability 时返回明确能力错误
EMA     -> 强制使用 EMA
DISABLED-> 不发布文本进度
```

图 route 在 engine load/admission 时确定，不在一个 segment 的中途切换。native 运行中
若游标输出异常，可以只把 progress route 切换到 EMA；Talker、Code2Wav 和音频状态不重置。
标准 plan 与 cursor-enabled plan 的外部音频 ABI 保持一致，cursor output 通过 manifest
能力声明和可选 binding 适配。

### D11. `qwen3_tts_12hz_la1_seed0.pt` 是随模型权重发布的可选能力资产

`qwen3_tts_12hz_la1_seed0.pt` 不再放在仓库、分支资源目录或由命令行路径注入。模型发布方
将它放在模型权重目录中；导出阶段自动发现模型目录下的该文件（兼容过渡期也接受
`model/weights/qwen3_tts_12hz_la1_seed0.pt`），并将它复制到导出包的
`<variant>/weights/qwen3_tts_12hz_la1_seed0.pt`。已经存在的导出权重目录若包含该文件，
也可作为 standalone export 的输入；若源文件和导出
文件同时存在但内容不同，导出必须失败，避免陈旧游标头静默启用。

因此，`weights/qwen3_tts_12hz_la1_seed0.pt` 的存在是“该模型请求启用游标”的资产级开关；它仍必须与
`native_cursor.enabled=true` 的 cursor-enabled TRT plan、manifest fingerprint 和已验证的
模型组合同时成立。没有该文件时只生成/运行 standard TRT plan，不得通过 CLI 强行
挂载另一个模型的游标头。

## 3. cursor-enabled TRT 图接口

### 3.1 额外输入

首版建议使用固定最大长度配合有效长度，避免每次文本增长都改变 CUDA Graph signature：

```text
cursor_label_ids                 [B, M_max] int32
cursor_label_count               [B]        int32
cursor_active                    [B]        bool/int32
cursor_mu                        [B]        fp32
cursor_frames_since_advance      [B]        fp32
cursor_delta_history             [B, 8]     fp32
cursor_trunk_state_*             fixed per-slot state
cursor_override_valid            [B]        bool/int32
cursor_override_mu               [B]        fp32
```

`cursor_override_*` 只用于 CPU 完成 tail rewrite/reanchor 后注入新坐标。raw span 本身不进入图。

ONNX 的输入 binding 对会在输出中回写的状态使用 `_in` 后缀（例如
`cursor_mu_in`、`cursor_conv_history_in`），避免输入和输出 value name 冲突；manifest
同时保留输入/输出完整名称，executor 不依赖位置推断。

### 3.2 额外输出

```text
cursor_valid                     [B]
cursor_mu                        [B]
cursor_delta                     [B]
cursor_confidence                [B]
cursor_candidate_label           [B]
cursor_frames_since_advance      [B]
cursor_delta_history             [B, 8]
cursor_trunk_state_*             fixed per-slot state
codec0                           [B] optional ABI alias
```

输出不包含 raw 或 normalized codepoint。CPU 必须基于当前 TN plan 和 owner spans 解释
`cursor_mu`。

## 4. 融合图必须完成的流式改造

下面四项都是一个 cursor-enabled fused TRT 图必须完成的实现工作，不是把游标拆到
CPU 或第二个 backend 的理由。它们的存在恰恰说明游标应当和 Talker decode 使用同一条
有状态计算路径。

### 4.1 一帧 lookahead

发布头的 `right_context=1`。第 `t` 个 codec0 到达时，matcher 输出应描述第 `t-1`
帧，而不是第 `t` 帧。TRT state 必须保存 pending feature，并输出 `cursor_valid`，让首帧
不发布伪造估计。

当前 engine 本来就会执行一个产生 EOS 音频的 decode step，并在结果处理时丢弃该 EOS 音频。
这个 EOS codec0 正好可以作为最后一个语音帧的 lookahead，让融合图完成 pending matcher
计算；因此正常 EOS 路径不需要额外的 cursor backend 或额外音频步。EOS 对应的 matcher
结果本身不作为语音进度发布，segment final 仍由 CPU 后处理完成。只有异常终止、没有
codec EOS，或验收要求严格复现离线 zero-padding 轨迹时，才需要一个不提交 Talker、Code2Wav、
KV、PCM 的 cursor-only/masked flush。

### 4.2 膨胀卷积的精确流式状态

分支实现每帧重新截取 codec0 窗口；融合图应将 `CodecTrunk` 改造成真正的流式 step：
各卷积块保留所需历史激活，最后一层保留一个 pending lookahead，收到新的 codec0 后输出
上一帧的 trunk feature。直接把缺失历史填成 token id 0 并不等价于卷积前的零 padding，
因为 id 0 仍有非零 embedding。

正式实现应把各卷积块历史激活和 pending lookahead 作为显式 state I/O。固定 31 帧重算只
允许作为 bring-up 和数值对照路径，必须正确表示左侧零 padding 和有效长度，不能用普通
vocab token 冒充 padding。

### 4.3 动态文本与固定执行 shape

不同 slot 的可见 label 数量不同，而且文本会流式增长。首版通过 `[B, M_max]` padding 和
`label_count` mask 保持 binding shape 固定。CUDA Graph 捕获的是固定的 shape、binding 地址
和执行拓扑，不是固定的输入数值；因此每次文本更新只需把新的 `label_ids` 写入同一个
预分配 GPU buffer，更新 `label_count`，不需要因为 label 内容改变而重新 capture。真正需要
避免的是改变 shape、重新分配输入 tensor 或改变 profile。

若 text encoder 中的动态 gather、mask 或 `Floor` 在目标 TensorRT 版本导出失败，应在 ONNX
层改写为等价的标准 `Gather`/mask 图，而不是把 matcher 退回 CPU。

### 4.4 文本版本与音频帧的时序一致性

每个 matcher step 必须看到“该 codec frame 的 Talker 已经看到的 spoken text frontier”。
这正是把 cursor 分支放进同一 fused graph 的收益：同一个 slot、同一次 decode 调用同时
使用 Talker 的 codec0 和该 slot 的 cursor label buffer，不存在独立 CPU matcher 追赶音频
的窗口。

segment 下发文本时只需把主 TN 生成的新 label plan 写入该 slot 的固定输入 buffer，并在
对应 decode step 前建立明确的 happens-before 关系。这是 backend 必须补齐的输入准备和
slot state 工作，不是拆分计算图的理由。

## 5. raw 坐标映射合同

主 TN 到游标适配层必须提供稳定 owner 信息。每个 spoken label 至少能追溯到：

```text
owner_id
owner.label_start/end
owner.normalized_start/end
owner.raw_start/end
plan_revision
```

投影规则：

1. TRT 输出连续 label 坐标 `mu`；
2. CPU 找到当前 label owner；
3. 只有 owner 的可确认边界推进协议 high-water；
4. 通过 `CanonicalTextJournal` 得到会话级 raw 坐标；
5. 对外发布值必须与上一锚点取 `max`，永不回退；
6. owner 内连续比例只形成 display-only 坐标。

tail rewrite 时优先通过 `owner_id` 或稳定 raw span 重锚定，不能直接沿用旧 label index。

## 6. 被否决的方案

### 2026-09-09 补充：与估计器无关的公共文本坐标合同

raw -> TN spoken -> segment-local text token 的 provenance 由文本前端拥有。
音频侧合同为 segment-local codec frame interval -> token frontier；EMA/Native
只估计这一步。不同 segment 的 codec 起点可都为 0，不等于同一个会话帧号。
公共投影生成的事件与 `AttributedAudioChunk` 绑定，经 `AudioReorder` 进入交付顺序，
再由既有输出层赋予实际 PCM/sample 坐标，禁止按估计进度反推 PCM 长度。
EMA 与 native 都先产生同一 segment 的 token frontier，再经公共投影器返回
session-global normalized/raw 整数边界。Native 的 label mu 不等于 Talker BPE
index；labelizer 必须提供精确 label -> spoken codepoint span，由适配器查表转换
为已完成 token 数。不得用 label 数与 BPE 数的比例换算。

正常 Splitter 分段不要求 TN owner 完整落在同一 segment。带精确 label spans 的
plan 可以按 normalized window 切片，保留 owner 身份及完整 raw provenance；raw
确认边界仍由主 TN journal 决定，字段内部可保持不动。旧的无 label spans plan
保留保守切片行为，作为兼容路径，不再作为生产分段的默认合同。不得为此强制每个
TextCommit 单独合成、变更 Splitter 容量规则或引入第二套 TN。

本变更仅扩展 CPU provenance，不改变 TRT binding/权重/神经状态 ABI。测试必须覆盖
相同 token frontier 下 EMA/native 相同坐标、扩展字段中途不提前确认 raw、跨 segment
全局基址、revision/high-water、重试及交错输出。验证结果在实现测试通过后记录，当前
不以此说明音频质量或发布证据已完成。

实现入口：`engine/core/text_coordinates.py` 定义 `CodecTokenProgress`、
`SegmentTextCoordinates` 和 `TextProgressProjection`；`NativeCursorLabelizer`
提供精确 label offsets，`CursorLabelPlanAdapter` 把它们平移到全局 spoken 坐标。
普通分段允许 owner 跨界，raw 确认仍共用 `CanonicalTextJournal`。旧无 offsets
plan 保持保守兼容，不会用字符长度推断 label 个数。活动 segment 窗口增长会在
token 入队前更新 plan revision；未改变的 label+provenance 前缀可保留神经 mu。

本轮验证（2026-09-09）：完整 pytest `1333 passed, 68 skipped`；真实 X2/TRT
successor E2E `2 passed`。`test_shared_progress_coordinates.py` 覆盖相同 codec/token
frontier 的两种估计器同坐标、lookahead 保持、异常回退不后退、无神经观察不宣称
native、事件与 PCM 重排绑定；`test_streaming_tn_cursor_plan_contract.py` 覆盖
full/long/逐字符 token 输入同结果、TN expansion、活动窗口先于 token 更新，以及
真实 Splitter 切入 owner 后保留 native。旧“切入 owner 必须全段降级”的断言已替换。

| 方案 | 否决原因 |
|---|---|
| 把官方 PyTorch TTS 作为 backend | 偏离项目 TRT 主路径，且没有解决游标头缺失的能力兼容问题 |
| 2M 游标头长期作为 CPU PyTorch sidecar | 重复窗口计算、线程池开销、可能推迟进度事件，并增加另一套推理运行时 |
| 把动态 TN 或 Unicode 映射导入 TRT | 语义动态、规则复杂且与张量推理边界不匹配 |
| 使用游标分支自带 TN 与主 TN 并行 | 两套 spoken plan 可能不同，raw 投影必然失真 |
| 给所有官方模型硬挂同一个 2M 头 | 游标头与模型/codec/语音域绑定，输出可运行不代表坐标可信 |
| 首版强制拆成独立 text-encoder TRT 调用 | 增加状态同步和调度复杂度；应先证明重算确实构成瓶颈 |

## 7. 验收条件

实现进入生产前至少必须通过：

1. 同一组 codec0 和 labels 下，PyTorch 参考、ONNX 和 TRT 的逐帧 `mu`、`delta`、
   `confidence` 在约定误差内一致；
2. 明确验证一个 codec frame 只执行一次 matcher 状态更新；
3. 首帧、lookahead、EOS、正常 final、取消和异常终止的时序测试；
4. prefill 产生的首个 codec0 与普通 decode codec0 都进入同一游标状态机；
5. `99%`、序数、日期、单位、中英混排等 TN owner-span 映射测试；
6. tail rewrite 后 public raw/normalized high-water 不回退；
7. cursor-enabled `custom-1.7B` 与标准官方 plan 使用统一运行时接口；
8. 游标 manifest 不匹配时确定性降级 EMA；
9. 并发 batch 下 label padding/mask 和 per-slot state 不串扰；
10. 性能测试证明游标分支不会引入额外的每帧 CPU PyTorch 推理或不必要的 GPU→CPU 同步。

## 8. 后续变更规则

任何实现若要改变以下边界，必须先更新本决策记录并说明原因、数据和迁移方案：

- 将游标神经计算移出 cursor-enabled TRT 图；
- 引入第二套在线 TN；
- 将 raw/Unicode 投影放入 TRT；
- 改变一个 codec frame 一次 matcher 的语义；
- 在未验证 fingerprint 的模型上启用 native cursor；
- 将 display-only 插值提升为协议 high-water；
- 把官方 PyTorch 模型引入为兼容 backend。

## 9. 首版导图实现记录（2026-09-04）

本轮先落地“计算图合同”，不改变现有 standard plan 的输入输出：

- `scripts/export/native_cursor_modules.py`：将 released `la=1` 游标头转换为
  TensorRT 可导出的单帧 streaming step。每个 block 使用激活历史，最后一层保留
  pending lookahead；输入是固定长度 label/state，输出是下一帧的神经状态和数值估计。
- `scripts/export/export_09_talker_code2wav_fused.py`：自动发现模型/导出权重目录中的
  `qwen3_tts_12hz_la1_seed0.pt`，存在时生成 `Talker -> Cursor -> Code2Wav` 的 cursor-enabled
  ONNX；不存在时
  保持 standard 图合同。游标分支直接连接 Talker 内部采样出的 `full_codec[:, 0]`，不发生
  host round-trip；显式 `codec0` 只作为 ABI alias 输出。
- `scripts/export/export_01_embeddings.py`：把模型发布目录中的
  `qwen3_tts_12hz_la1_seed0.pt` 随其他导出
  权重复制到 `<variant>/weights/qwen3_tts_12hz_la1_seed0.pt`，使 bundle 携带完整的能力资产。
- `scripts/python/triton_manifest_io.py`、`trt_fused_io_formats.py`、
  `trt_fused_talk_c2w_profiles.py` 和 TRT host builder：声明 cursor capability、固定
  `[B, M_max]` label/profile、状态 shape 及 integer/float I/O 类型。
- `scripts/bash/build_engines.sh`：从 manifest 读取 cursor profile 参数；没有 capability
  时仍按原 standard profile 构建。

本轮已经补上 executor 的 per-slot 状态转发和 `set_cursor_text_plan`/
`set_cursor_reanchor` 合同；CPU `CursorTextPlanAdapter` 已由主 streaming TN 的
`TextCommit`/journal 接入，plan revision 更新还会通过稳定 owner/span 做重锚定，
并由 high-water projector 负责对外坐标。尚未在真实部署环境执行 TRT 逐帧数值验收，
因此 cursor-enabled plan 在生产启用前仍必须完成：真实 cursor state ABI、TRT
逐帧对照、successor handoff 和标准 plan 的能力路由。这是一项明确的下一步，不允许
用 Qwen BPE `input_embeds` 冒充 cursor label embedding。

## 10. 当前实现状态（2026-09-10）

上面的首版导图记录保留为历史决策背景；截至本次文档更新，已验证的
`custom-1.7b` cursor-enabled TRT runtime 已接通以下在线链路：

```text
主 streaming TN
  -> TextCommit / CanonicalTextJournal
  -> CursorLabelPlanAdapter
  -> cursor-enabled TRT graph
  -> graph 内 codec0 观察
  -> CPU owner-span / high-water projection
  -> qwen.text_progress.v1
```

当前打包 artifact 的能力边界是 `right_context=1`、`max_labels=512`、16 个 codec
codebook、80 ms frame，以及 `native`/`ema`/`disabled` 三种 progress route。原生游标
只对匹配且验证过的 `custom-1.7b` 组合开放；其他模型仍走 standard TRT，并根据能力
协商降级到 EMA 或关闭进度。正常 EOS 使用已生成并丢弃的 EOS 音频完成 pending
lookahead，不通过额外的 cursor-only flush 改变 Talker、Code2Wav、KV 或 PCM 状态。

本轮 focused TN/cursor contract tests 在 2026-09-10 为 `99 passed`。这证明当前
Python 合同、owner 映射、high-water 单调性、plan revision/reanchor 和公开事件形状
一致；它不等价于所有模型、所有 profile 或跨硬件 TRT 数值验收。

在扩大默认发布范围前仍需完成或持续保持以下 release gates：

1. 每个发布 bundle 的 `model_fingerprint`、cursor-head、label-vocab、TN rules 和语音
   域 fingerprint 必须非空且匹配；不匹配必须 fail closed 到 standard/EMA 路径。
2. 以冻结 codec0/label 输入持续比较 PyTorch 参考、ONNX 与 TRT 的逐帧轨迹，并覆盖
   prefill、decode、lookahead、EOS、tail rewrite、并发 slot 隔离和官方无头 fallback。
3. 重新采集与当前 cursor-enabled artifact 对应的 Triton/engine 性能和质量证据；旧
   benchmark 数字不得自动代表新图。
4. `SOFT_DRAIN`、训练态 coverage 和低接缝 state rollover 仍保持 opt-in/未发布状态，
   不得在 README 或 capabilities 中暗示已经具备。
