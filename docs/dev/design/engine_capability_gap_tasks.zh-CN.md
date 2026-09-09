# Engine 能力缺口与任务拆分

> 评审日期：2026-09-07  
> 状态：当前实现盘点与并行开发任务清单  
> 适用范围：统一 WebUI、Native Gateway、Triton 兼容层、主 TRT Engine、流式 TN、Splitter、原生游标、状态继承和文本进度

## 1. 目标架构

项目只有一套 Engine Runtime 和一套主 TRT 执行后端。Native Gateway 与 Triton 是两种接入方式；Triton 用于兼容部署、模型仓库和服务管理，不拥有另一套 TRT 推理后端。

```text
统一 WebUI
    |
统一 Engine API / 事件协议
    |
Native Gateway ------------------ Triton 兼容层
    |                                      |
    +-------------- Engine Runtime --------+
                           |
                  Text Processing
                           |
                    Splitter FSM
                           |
                    Engine slot state
                           |
             主 TRT fused graph（按 artifact 能力启用 cursor 分支）
             Talker + Native Cursor + Code Predictor + Code2Wav
                           |
              audio / progress / state / lifecycle events
```

原生游标已经被流式改造并融合到主 TRT 图中。它不是 CPU PyTorch sidecar，也不是第二个 backend。CPU 只负责主 TN、Label Plan、owner span、reanchor、坐标投影和事件发布；TRT 图负责游标的 text encoder、codec trunk、matcher 以及神经状态更新。

## 2. 当前能力盘点

### 2.1 已有并应复用的能力

| 能力 | 当前证据 | 状态判断 |
|---|---|---|
| 统一 WebUI | `web/packages/demo`，运行时 `/demo/` | 主迁移完成；已接入 native/EMA、speech-state capability 状态展示，真实模型 E2E 待 H2/H3 |
| Native Gateway 与 Triton 公共协议 | `engine/gateway/`、capability golden contract、Triton deployment tests | 基本完成 |
| 传输层 session resume | WebSocket resume、audio/sample cursor、response resume | 已有；它只恢复传输事件，不等于声学状态继承 |
| 自建 Engine Runtime | Engine loop、continuous batching、MLFQ、KV pool、WAIT_TEXT | 已有主干 |
| 完整文本输入 | `FrontendInterface._feed_full_text_locked()`、`Spliter.pre_split()` | 已有 |
| Token/增量文本输入 | `IncrementalTextCommitter`、commit ticker、`TextCommit` | Phase 0 已实现并有测试 |
| raw/canonical/spoken 映射 | `CanonicalTextJournal`、commit mapping、segment token spans | 已有基础能力 |
| 统一 splitter pending queue | `Spliter._pending`、offline/streaming 共用 `_drive_events` | 已基本完成，仍需端到端覆盖 |
| EMA 文本进度 | `EmaTextProgressEstimator` 和统一 text_progress event | 已有，作为 fallback 使用 |
| 主 TRT fused graph | Talker + Code Predictor + Code2Wav | 已有并投入主运行时 |
| cursor-enabled fused graph 导出 | `export_09_talker_code2wav_fused.py`、cursor bindings | 已有导出骨架和合同测试 |
| cursor slot state | `SlotKVState`、executor cursor inputs/outputs、`update_cursor_state()`、固定 buffer API | 神经状态承载和回写已有；Label buffer 复用/清尾/输入校验已补齐 |
| cursor manifest gating | model/head/codec/codebook 等校验、standard/cursor artifact 区分 | 已有基础能力 |
| 状态继承合同 | `SpeechStateCapability`、typed operations、fail-closed adapter、bundle verifier | 合同、准入门禁、EngineLoop 边界调度和 manifest/artifact fail-closed 校验已完成；真实模型迁移仍待 H2/H3 |

### 2.2 已设计但尚未形成生产能力的部分

| 缺口 | 现状 | 影响 |
|---|---|---|
| 主 TN 到 cursor Label Plan 的运行时桥接 | 已有 CPU `CursorLabelPlanAdapter` 和模型-owned `NativeCursorLabelizer`；已贯通 Frontend → Dispatcher → EngineRequest → EngineLoop → slot，并按完整 owner 为 segment 切 plan；package startup 只在 cursor graph/head/vocab fingerprint 均通过时注入 labelizer | owner 被 segment 边界切开、label vocab 不匹配或 labelizer 失败时安全降级 EMA；真实 cursor-enabled TRT bundle 仍待验收 |
| cursor 输出到统一进度事件 | engine loop 已把每步 `valid/mu/confidence/candidate_label/plan_revision` 作为标量证据送到前端；CPU `NativeCursorProgressProjector` 按 owner span 生成统一事件，invalid/lookahead 自动回退 EMA；真实 cursor-enabled TRT 已通过 recurrent restore 和三段 successor E2E | 完整 cursor trajectory、Native/Triton 同结果集和发布 evidence 仍未完成 |
| Label Plan 与 slot 的时序合同 | typed plan request、slot admission 应用、decode 前 queue happens-before、旧 revision 拒绝，以及已发布 Label Plan 的 committed-prefix 保护已完成 | stale output 与 progress high-water 的完整合同、重试和跨 segment 仍需继续验收 |
| owner-span reanchor | `CanonicalTextJournal`、stable owner/raw span、CPU `reanchor_cursor_mu()` 和 EngineLoop 的 reanchor 转发已具备；重建 plan 时已保护已发布前缀 | 真实 tail rewrite、跨 segment 和 TRT 神经状态重锚定仍未完成 |
| native cursor lookahead/EOS 处理 | 真实三段 E2E 的首尾帧、正常 `codec_eos` 和 successor cursor continuation 已通过；异常终止、masked flush 和完整 lookahead 轨迹矩阵仍未覆盖 | 仍需冻结 codec0/label 输入，覆盖异常终止与 EOS 对照，排除少报、错位或重复提交 |
| TRT 全链路数值验收 | cursor 子图 32 帧 PyTorch/TrT、真实 full-state eager round-trip，以及 cursor plan 的 batch 1/2 graph/eager 对照已通过；仍缺完整 fused graph 逐帧参考 | 仍无法确认 Talker/Code Predictor/Code2Wav/PCM 与 PyTorch/ONNX 全链路一致 |
| 对外能力协商 | executor 能读 manifest；公共 capability 现在区分 `graph_enabled` 与 `progress_available`，且 projection 对不一致/字符串及非布尔 truthy 的 cursor/state 标志 fail-closed，发布 evidence 的 schema version 也严格校验，桥接未完成时只暴露 EMA/disabled | native progress 和 state transfer 仍未开放，后续需随真实链路验收更新 |
| 状态继承实现 | 已有 backend-owned snapshot/handle、同 segment PAUSE_RESUME、step-boundary migration 时序、可注入 X2 continuity 生命周期、manifest model contract 校验，以及 executor 对合同/指纹/transfer 的 fail-closed 门禁；已补齐 gated cursor recurrent payload/restore ABI、manifest/runtime 共用六组 binding 合同、export builder 显式 cursor ABI、builder fail-fast、cursor preflight 原子性和 stale payload 清理，带 X2 acoustic successor context 时按是否携带 cursor state 显式选择 native/EMA；真实 full-state eager round-trip 和三段 X2 successor E2E 已通过；外部 staging bundle 已通过真实 fingerprint/layout/ABI verifier，但未附 release evidence | X2 speech-state 公共能力仍未开放；仍缺精确 checkpoint/training contract、跨 segment 全量 TRT 不中断对照和发布 evidence |
| X2 commitment 与主 TN 的适配 | `X2CommitmentAdapter` 已接入 Frontend，只消费主 TN `TextCommit`、spoken projection 和调用方 token count；不调用外部 `feed_text()`，可转发容量/segment feedback；CPU 测试已覆盖 full/long/token 三入口相同 spoken projection、special-span boundary 和 pending replan 保留 | 仍缺真实 X2 factory、目标运行时上的三入口联测，以及方法层 commitment 与 Splitter FSM 的跨入口联测 |
| segment handoff / context rollover FSM | 文档有 Phase 1/2 设计，engine 仍以 WAIT_TEXT 和普通 segment 生命周期为主 | 长文本低接缝续接、跨上下文 rollover 尚未落地 |
| Soft Drain / coverage | 当前设计文档明确为 Phase 1 训练和运行时工作 | 不能把它误报为已有能力 |
| reorder stall 保护 | `AudioReorder` 现在记录按当前 playhead 计时的阻塞段，并提供可注入时钟的 timeout verdict；Frontend watchdog 发结构化告警后通过内部 `CANCEL_SESSION` 走正常 Engine 终态，终态 metadata 携带 `audio_reorder_stall_timeout` | 已覆盖检测、计时重置和 watchdog 取消合同；仍需真实并发压测确认 timeout 默认值、取消延迟和持有音频上限 |
| native cursor 与 Triton/Native 双入口 E2E | 同一真实 cursor-enabled TRT package 已完成 Native Executor 与 Triton compatibility layer 的 capability projection 对照；Native runtime 另有三段 X2 successor E2E | capability projection 已对称，但 Native/Triton 音频结果集和完整 speech-state successor E2E 仍需验证，不能把 projection 对照推断为音频一致 |

### 2.3 本轮 Luna worker 验收

本轮验收覆盖了 executor 的 cursor buffer primitive 和 CPU 侧合同，不等同于 native cursor 生产链路验收：

- `set_cursor_text_plan()` 在同一 slot 的重复更新中复用固定 tensor 地址；
- 更新较短 plan 会清理旧 label 尾部，不残留上一次 revision；
- plan 更新不会清零已经存在的 cursor 神经状态；
- slot 的 label/state buffer 保持隔离；
- 超容量、浮点 label、非法 count、非法 active 和非有限 reanchor 会在 slot 修改前拒绝；
- 额外修复并覆盖了负 label id 和超出 manifest `vocab_size` 的 label id，避免非法 id 到达设备侧 embedding/gather；
- empty plan、owner span 连续覆盖、revision 和 `ProgressMode` 合同有单元测试；
- 没有执行目标 TRT plan，因此不能据此宣称 PyTorch→ONNX→TRT 数值一致或 GPU runtime 完成。

2026-09-09 对 `workspace/exported/custom-1.7b/talker_code2wav_fused.engine` 做了真实
TensorRT artifact smoke check：计划可反序列化并暴露 62 个 I/O，包含标准 Talker/C2W
状态，但没有 `cursor_*` binding，manifest 也没有 `native_cursor`/`speech_state`。
这只证明 standard plan 的 fail-closed 路由和实际加载，不替代 cursor-enabled TRT 或
X2 successor bundle 验收；对应测试为环境门控的
`tests/integration/test_real_trt_artifact_capability_contract.py`。同一 plan 另有
`tests/integration/test_real_trt_state_transfer_contract.py`，验证真实 prefill 后 detached
bundle restore 到新 slot 的下一步 token、PCM、`codec_sum`、token counts、C2W conv 和
transconv 逐项一致；这仍不是 X2 或 native cursor 验收。

验证命令：

```text
/home/rime/miniforge3/envs/qwen3-tts/bin/python -m pytest -q \
  tests/unit/engine_core/test_native_cursor_contract.py \
  tests/unit/engine_core/test_native_cursor_contract_interop.py \
  tests/unit/engine_core/test_speech_state.py \
  tests/unit/operators/test_native_cursor_modules.py
```

结果：基础 cursor 合同与模块测试为 `44 passed`；补充 executor buffer/输入校验回归为 `16 passed`；主 executor/engine loop 回归为 `52 passed`；前端与 progress 回归为 `72 passed`。这些是有重叠的测试组，不应相加作为唯一用例数。当前环境没有安装 `ruff`，因此未完成 ruff 检查；`py_compile` 和 `git diff --check` 已通过。

2026-09-09 又执行了环境门控的
`tests/integration/test_real_x2_successor_e2e.py`，结果为 `1 passed`。它使用真实
X2 方法层、真实 cursor-enabled TRT plan 和模型 tokenizer，验证三段输入经过主 TN 后
得到正确 spoken text，三个 segment 均正常 `codec_eos`，后两个 successor 均报告
`extension_continuity=restored`、`extension_bridge=prepared` 和
`cursor_progress=native_continuation`。这证明真实 Native runtime 链路接通，但不证明
speech-state 模型合同、逐帧数值等价或 Triton 兼容层同结果集；manifest 仍缺真实
speech-state fingerprints/evidence，公共 state transfer/native progress 继续 fail-closed。
另有 `tests/integration/test_real_native_cursor_state_transfer.py` 在同一真实 plan 上把
源 slot 的六组 fused cursor recurrent tensor 推进一帧后 detached，清空目标 slot，再恢复
并比较下一次 decode 的普通输出和全部 cursor 输出，结果为 `1 passed`。这证明 cursor
state restore 不是只改了 metrics；但 Talker/C2W 状态在该测试中保持相同，完整 speech-state
successor 数值对照仍未完成。
新增的 `tests/integration/test_real_native_cursor_full_state_transfer.py` 又对完整 pooled
bundle 做了真实 eager 和 CUDA Graph round-trip：Talker/C2W KV、conv/transconv、slot auxiliary、采样 RNG
和 cursor recurrent state 一起恢复，下一次 fused decode 的 token、PCM、hidden、codec、
token count、C2W 和 cursor 输出逐项一致，结果为 `1 passed`。这证明同 segment
PAUSE_RESUME 的真实 full-state 基线已经成立，但不等于跨 segment X2 successor 的模型训练
语义或音频质量验收。
`tests/integration/test_real_native_cursor_trt_trajectory.py` 又固定真实 graph 的
`codec0`，逐帧与模型自带 PyTorch cursor reference 对照 32 帧，离散输出精确一致、bf16
浮点输出在显式误差预算内，结果为 `1 passed`；这仍是 cursor 子图轨迹，不是完整 fused
Talker/C2W/PCM 或 successor 数值等价。
`tests/integration/test_real_cursor_capability_projection.py` 还验证同一 package 经 Native
Executor 和 Triton compatibility layer 后的公开 capability 逐字段一致，结果为 `1 passed`；
这验证的是统一接口合同，不是第二种 TRT backend，也不是音频结果等价。

已知边界：executor API 严格接受 Python `bool` 或 0/1 整数作为 `active`。NumPy `bool_` 会被拒绝；如果未来协议层允许 NumPy scalar 直接进入该 API，应先把类型转换放在协议边界，并为该边界增加测试。

验收结论：Luna worker 的固定 GPU buffer、状态保留、slot 隔离和输入校验可以合入作为底层 primitive；但该交付仍不能标记为 native cursor feature complete。当前已存在 `set_cursor_text_plan()` 的 TN → Label Plan → Engine slot 调用链、模型-owned labelizer startup gate、真实 cursor-enabled TRT ABI/状态恢复验收，以及 fused cursor scalar output → CPU projector → unified progress 的可注入链路；完整 cursor trajectory、segment-scoped release evidence 和 Native/Triton 同结果集仍未完成。因此 native progress 仍不能对外宣称可用。另已补上 AudioReorder 的永久阻塞保护，但它只负责安全终止，不会越序释放后段音频。

T1 已补充 `CursorLabelPlanAdapter` 纯 CPU 合同：labelizer 外置，适配器只消费主 TN 已提交的 commit 和 journaled spoken projection；支持 owner span、revision、保守 mapping 和 committed-prefix 保护。现已新增模型-owned `NativeCursorLabelizer` 和 package startup gate，前端默认仍只在真实 graph/head/vocab fingerprint 校验通过时创建 adapter。T2 已补充 typed plan request 和 slot revision/happens-before plumbing，并有 Frontend/EngineLoop 回归测试。T3 已补充 scalar handoff 和 native CPU projector，但尚未完成真实 TRT/多 segment 验收。T6 已补充一层 fail-closed 的公共 capability：标准 plan 返回 `graph_enabled=false`；cursor-enabled plan 只说明 `graph_enabled=true`，在 T4/T5 完成前保持 `progress_available=false`，公开 progress route 仅为 `EMA/DISABLED`。这不是 native cursor 上线声明。

## 3. 任务拆分

任务按依赖排序。每个任务都应由一个模型或一个小组独立认领，并提交代码、测试和简短验证报告。除非任务明确要求，不得重写 scheduler、KV cache、主 TRT 图或另起一套 TN。

### T0：冻结公共能力合同

**Owner：** `engine/core`、`engine/gateway`、协议测试

**目标：** 定义统一的请求能力、Label Plan、progress 和状态继承枚举，供后续任务共同使用。

**需要完成：**

- 定义 `ProgressMode`：`AUTO`、`NATIVE`、`EMA`、`DISABLED`；
- 定义 `LabelPlan`/`LabelPlanRevision` 的内部合同，至少包含 `label_ids`、`label_count`、`owner_spans`、`raw/canonical span`、`revision`、`final`；
- 定义 native progress 与 EMA 使用同一事件外壳时的 `basis`、`quality`、`confidence`、`final` 语义；
- 将 `SpeechStateCapability`、native cursor capability 纳入统一 capability 版本；
- 明确 `NATIVE` 不可用时是拒绝还是按配置降级，默认必须 fail-closed；
- 保持 Native Gateway 与 Triton 兼容层的公共 JSON/gRPC/WS 合同一致。

**验收：** capability schema、Python 类型、golden contract、Native/Triton 对称测试全部通过；不能出现自由字符串代替枚举。当前已完成 CPU 侧 `ProgressMode`/Label Plan 数据结构，以及 graph capability 与 progress route 的 fail-closed 公共字段；native progress 和 state transfer 仍未接入。

**依赖：** 无。  
**阻塞：** T1、T2、T6、T7。

### T1：实现主 TN 到 Label Plan 的适配器

**Owner：** `engine/frontend`，不得修改主 TN 规则为第二套 cursor TN。

**目标：** 从现有 `TextCommit`、mapping 和 `CanonicalTextJournal` 派生唯一的 cursor Label Plan。

**需要完成：**

- 将 spoken text 映射到 cursor label vocabulary；
- 每个 label 保留稳定 `owner_id` 以及 raw/normalized/canonical span；
- 支持 committed prefix、mutable tail、plan revision 和 final flush；
- 对完整文本、增量长文本、token 输入提供相同的 Label Plan 输出；
- 明确 plan 何时可以下发给 splitter/engine，mutable tail 不得提前进入已执行 segment；
- 明确超过 `cursor_max_labels` 时的分段或 fail-closed 行为；
- 不读取 Talker BPE/prompt ids 作为 cursor labels。

**验收：** 中英文、数字、日期、百分比、单位、URL、数学表达式、emoji 分包和 TN 展开都有 owner-span 测试；随机分包最终 plan 与 full-text oracle 一致；plan revision 单调且已提交 owner 不被回写。

**依赖：** T0。  
**阻塞：** T2、T3、T4。

### T1b：把 X2 commitment 适配到主 TN

**Owner：** `engine/frontend`、X2 adapter；不得修改 `IncrementalTextCommitter` 的
spoken-form 语义，也不得把外部规则归一化器接入在线主路径。

**目标：** 让 X2 的容量估计、边界策略和主 TN 的 `TextCommit` 共存，形成一条文本真相。

**需要完成：**

- 定义窄适配器：输入为主 TN 的 `TextCommit`、`mapping`、owner span 和 final 事件，
  输出只能是已提交 spoken projection 的边界/容量建议；不得重新消费 raw Unicode 并
  再执行一遍百分比、特殊文本或 emoji 归一化；
- 将 `observe_segment`、capacity threshold 和 Splitter FSM 的更新顺序固定下来，
  避免同一段音频同时更新 X2 estimator 和 Engine 自带 ratio controller 两次；
- full text、增量长文本、token 输入、尾部 rewrite、final flush 都必须产生同一份
  spoken/owner 结果；适配器失败时降级到 Engine 默认 Splitter/EMA，不破坏音频；
- 明确外部 X2 `CausalCommitment.feed_text()` 只能作为离线/独立方法层实验接口，
  不能直接套用到当前 Frontend。

**验收：** 主 TN `TextCommit` 是唯一 spoken-form 来源；随机分包的 plan、Splitter
边界和 raw/normalized high-water 与不启用 X2 adapter 的 oracle 一致；capacity 只更新
一次；Native Gateway/Triton 兼容层没有不同文本结果。

**依赖：** T1。  
**阻塞：** X2 commitment 生产接入、T2 的真实 capacity 联测。

### T2：把 Label Plan 接入 Engine slot 和 decode 时序

**Owner：** `engine/frontend/interface.py`、`engine/backend/engine_loop.py`、`engine/backend/executor.py`

**目标：** 在每一个 decode step 前，把主 TN 产生的 plan 写入对应 slot 的固定 GPU buffer，并调用已有 cursor executor API。

**需要完成：**

- 为 session/segment 建立 CPU 侧 plan owner；
- 在 prefill、segment admission、增量 commit、tail rewrite 时调用 `set_cursor_text_plan()`；
- 在 decode launch 前建立明确的 happens-before，保证该 step 使用正确的 plan revision；
- 通过固定 `[B, M_max]` buffer 和 `label_count` 更新内容，不因 label 内容变化而重捕 CUDA Graph；
- 处理 slot 释放、重试、取消和重新 admission 时的 cursor state 清零；
- 为每个输出记录观察到的 `plan_revision`，拒绝旧 revision 覆盖新状态；
- 需要 reanchor 时调用 `set_cursor_reanchor()`，不能直接改写 `cursor_mu`。

**验收：** 单 slot、多 slot、动态 batch、CUDA Graph/eager 两条路径的 plan revision 不错位；测试能证明 decode step 看到的 Label Plan 与对应 codec0 属于同一 frontier；仓库中出现真实生产调用链，而不是只在测试中调用 executor API。

**当前进度：** executor 的固定 cursor buffer primitive、CPU `CursorLabelPlanAdapter`、owner-safe segment plan slicing、TN → Label Plan → slot typed plumbing，以及 gated cursor recurrent payload/restore ABI 已完成。新增的 `NativeCursorLabelizer` 从模型自带 checkpoint vocab 构造，中文使用无声调拼音、英文使用 `en:<letter>`，并在 `TTSEngine.start()` 仅对已校验的 cursor-enabled package 建立 factory；缺依赖、vocab fingerprint 不匹配或不支持字符都会保持 EMA。真实 `x-square-robot/X2Streaming-TTS-1.7B` 已完成 cursor-enabled ONNX/TRT 导出，Executor 已验证 12 个 cursor 输入、11 个 cursor 输出、完整 recurrent ABI 和真实下一步状态恢复；外部 staging bundle 还通过了实际文件 fingerprint/layout/ABI 校验，但没有 release evidence，生产 capability 仍关闭。完整 committed frontier、跨实现 TRT 轨迹和 release evidence 尚未完成。公共 capability 现在会区分 cursor graph 已加载、Label Plan bridge 是否可用和 native progress 是否可用。  
**依赖：** T1。  
**阻塞：** T3、T4、T5。

### T3：实现 native cursor CPU 后处理和统一 progress 发布

**Owner：** `engine/frontend`、`engine/core/text_progress.py`、输出协议适配层

**目标：** 把融合图输出的 `cursor_mu`、`cursor_valid`、`cursor_confidence` 转换成统一的文本 progress event。

**需要完成：**

- 从 `cursor_mu` 找到当前 owner span；
- 通过 `CanonicalTextJournal` 投影到 normalized/raw 坐标；
- 协议 high-water 使用保守整数边界并严格单调；
- owner 内浮点插值仅作为 display 字段，不能用于确认、计费、release 或恢复；
- 处理首帧 lookahead 无 valid output、普通 decode、EOS decode、segment final、abort 和 retry；
- native 输出异常时只将 progress route 降级为 EMA，不重置 Talker、Code2Wav 或音频状态；
- 让 `OutputProcessor`、Native WebSocket、OpenAI Realtime 和 Triton 入口看到同一事件语义。

**验收：** native/EMA 使用同一事件 schema；raw/normalized high-water 永不回退；TN 展开、tail rewrite、跨 segment、segment final 以及异常终止都有测试；client 的非法 progress anchor 不会终止音频流。

**当前进度：** 已完成纯 CPU `NativeCursorProgressProjector`、owner-level high-water、owner 内 display-only 插值、lookahead invalid 回退 EMA，以及 engine loop 到前端的标量输出 handoff；segment plan 现在只接受完整 owner，边界切 owner 时自动停用 native。模型兼容 labelizer 已实现并接入 package startup gate；真实 cursor-enabled bundle、recurrent state restore、32 帧 cursor 子图轨迹和三段 successor E2E 已通过，但当前仍保持公共 `progress_available=false`，因为完整 fused trajectory、speech-state release evidence 和 Native/Triton 音频同结果集尚未完成。

**依赖：** T0、T1、T2。  
**阻塞：** T5、T7。

### T4：完成 native cursor fused TRT 验收

**Owner：** `scripts/export`、TRT 构建和 engine backend 测试

**目标：** 验证已有主 TRT cursor 分支，而不是重新设计或拆出第二个 backend。

**需要完成：**

- 固定 codec0/label/state 输入，逐帧比较 PyTorch 参考、ONNX 和 TRT 的 `mu`、`delta`、`confidence`；
- 覆盖 prefill、decode、lookahead、EOS、异常终止和 masked/cursor-only flush；
- 验证一个 codec frame 只触发一次 matcher 更新；
- 验证 cursor branch 直接消费图内 codec0，不通过 CPU 回拷再执行第二次神经推理；
- 验证 CUDA Graph replay 与 eager 数值和事件时序一致；
- 验证 batch slot 的 label padding、mask、state 和 output 不串扰；
- 验证 manifest/head/checkpoint/codec/codebook/rules fingerprint 不匹配时禁用 native cursor，走标准 TRT + EMA。

**验收：** 启动目标 GPU 服务取得固定语料音频，离线调用 ASR 检查 CER/WER 和参考文本，
再把结果写入 release evidence；线上合成路径不调用 ASR。结构性 cursor 状态/事件顺序和
所有失败路径都有明确错误或降级，不允许 silent enable。`full_codec` 逐 token 精确只作为
诊断，不作为 BF16 默认 plan 的产品质量硬门槛。

**2026-09-09 进展：** 新增 `tests/integration/test_real_native_cursor_cuda_graph_parity.py`。
在真实 cursor-enabled plan 上，profile 0 已通过 batch 1 和 batch 2、各四步的 graph/eager
对照，包含 token、
`codec_sum`、hidden、C2W KV、cursor 输出和 PCM；profile 1 虽可构图但首步递归输出已
分叉，故 Executor 对 cursor plan 固定选择 profile 0，standard plan 的 decode-only
profile 选择保持不变。BF16 正式 plan 已是此前验证过的默认基线；T4 仍缺固定语料的
音频/ASR 质量报告、batch slot/事件时序、异常 flush 和正式服务性能报告。全 FP32 仅
作为数值定位对照，不是默认部署建议。

真实 fused ONNX 冻结输入探针进一步确认了当前边界：`codec0` 和 cursor 离散输出一致，
但 CPU ORT float32 与 bf16 TRT 的 residual codebook/hidden/C2W 浮点轨迹本次最大误差约为
1570/0.994/1.476，不能据此开放 full fused parity。真实 X2 A/B 集成测试已通过
（`tests/integration/test_real_x2_successor_e2e.py`，`2 passed`）：continuity 路径实际
执行 successor bridge（3 segments、约 6.56s），关闭 X2 则走单个 hard-boundary segment
（约 5.20s），两条路径均生成真实 WAV；仍缺模型级不中断音质指标和训练合同。

针对数值诊断，已在隔离外部 staging 目录实际构建 `cp=fp32`、backbone/Code2Wav 为
BF16 的对照 plan，并用 `tools/validation/fused_onnx_trt_parity.py` 做了同一冻结输入
复测：`codec0` 仍 5/5 一致，但 `full_codec` 的 mismatch count 为 `7/12/10/1/15`。
这说明提高 CP 精度会改变离散采样路径，但不能把 codebook 逐 token 精确当作产品质量
判据，也不能仅凭该 mismatch 数量决定发布与否。此前全 FP32 plan 的
5 步报告使用了 BF16 capture artifact 产生的输入，不能作为 FP32 自身 prefill→decode
轨迹。改用与 capture engine 相同的 FP32 dtype、固定 cursor profile 0，并关闭 TensorRT
TF32 后，PyTorch→ONNX→TRT 的 `full_codec` 5/5 精确，hidden RMS 约
`9.4e-6~6.2e-5`；这闭环了 FP32 的数值定位项。正确 dtype 的 BF16 自捕获 5 步存在
正常的 residual codebook 波动，不能单独推导音频或语义回归。T4 仍需完成固定语料的
ASR CER/WER、音频时长/终止行为、lookahead/EOS/flush、batch slot/事件时序和性能
成本报告，形成正式 release evidence。

`fused_onnx_trt_parity.py` 现可通过 `--torch-model` 在同一冻结输入上运行导出器自身的
PyTorch fused wrapper，并通过 `--output` 保存干净 JSON；工具还会按 manifest 选择 cursor
profile 0、按 manifest dtype 分配 capture buffer，并校验模型自带 cursor head 的 hash。
FP32 `--noTF32` 报告可归档为数值定位证据，但尚未形成可发布的 T4 通过报告；BF16 自捕获
结果中的 CP codebook 分叉只作为诊断信息。报告必须继续覆盖 prefill/decode/EOS、音频
输出和 ASR 结果，不能把 exporter 的 smoke verification 或单一 token mismatch 统计
当作默认部署证据。

同一 RTX 5090/TensorRT 10.13 的 `trtexec --useCudaGraph --noDataTransfers` decode
microbenchmark（profile 0，`c2w_attention_bias=1x1x1x2`）显示 BF16 为 median
`8.039 ms`、`124.38 qps`，全 FP32 为 `11.247 ms`、`88.91 qps`。这意味着全 FP32
约有 `39.9%` 延迟代价；该结果未覆盖 batching、KV gather、EngineLoop 和 Gateway，仍不
等于正式发布性能报告。

**依赖：** T2 前可独立进行图级工作，最终与 T2/T3 联测。  
**阻塞：** T5。

### T5：完成 native cursor 端到端验收

**Owner：** Engine integration / E2E

**目标：** 把 T1-T4 连成真实产品路径。

**需要完成：**

- full text、增量长文本、token 输入三种入口；
- Native Gateway 与 Triton 兼容层各跑一套相同 case；
- `AUTO`、`NATIVE`、`EMA`、`DISABLED` 四种 progress route；
- 标准 plan 与 cursor-enabled plan 的能力路由；
- 多 session、多 slot、segment retry、取消、断连 resume；
- 比较 audio、segment lifecycle、progress 和 terminal event 的顺序。

**验收：** 两种入口的统一合同一致；native cursor 失败不会破坏音频；progress 不回退；长文本完整覆盖；并发 slot 没有跨会话污染。

**依赖：** T0-T4。  
**阻塞：** 对外宣布“精确文本游标可用”。

### T6：实现模型能力声明和统一路由

**Owner：** `engine/runtime`、`engine/server.py`、`engine/gateway/capabilities.py`

**目标：** 把“模型能力”和“请求策略”明确暴露给 WebUI/SDK，并由同一 Engine Runtime 选择路径。

**需要完成：**

- manifest 暴露 fused graph 是否包含 cursor branch；
- 暴露可用 progress modes、cursor fingerprint、最大 Label Plan 长度；
- 暴露 state transfer operations 和 accuracy class；
- `AUTO` 按实际 artifact capability 选择 native 或 EMA；
- `NATIVE` 无匹配能力时返回稳定错误；
- Triton 兼容层只转发/投影同一份 capability，不自己判断模型能力。
- 发布 evidence 必须绑定 manifest `speech_state.model_fingerprint` 和
  `speech_state.runtime_fingerprint`；缺少任一身份时，即使 evidence 字段完整也保持
  native/state fail-closed。

**验收：** 未加载模型、standard plan、cursor-enabled plan、错误 manifest 四种状态均有明确 capability；WebUI 不显示未实际支持的选项。

**依赖：** T0；cursor 字段可先与 T2-T5 并行。  
**阻塞：** WebUI 能力演示和 SDK 正确路由。

### T7：实现真实 speech state capture/restore

具体委派范围、代码接入风险和验收门槛见 [Speech State 接入与验收任务](speech_state_handoff_tasks.zh-CN.md)。按 S1 安全合同、S2 完整状态迁移基线、S3 X2 模型续接合同、S4 调度接入依次推进；不能直接在 EOS handler capture，再在普通 prefill 前 restore。

**Owner：** `engine/backend/speech_state.py`、`engine/backend/engine_loop.py`、`engine/backend/kv_cache_pool.py`

**目标：** 在不破坏现有调度器和 slot 架构的前提下，实现 X2Streaming-TTS 的子句间状态继承。

**状态清单至少包括：**

- Talker KV 和有效 past length；
- Code Predictor/KV 或其 fused 状态；
- Code2Wav KV；
- Code2Wav conv/transconv history；
- codec/frame index、采样 RNG 和 retry generation；
- native cursor neural state、Label Plan revision 和 reanchor 状态；
- 当前 segment 的输入完成度、pending tail 和恢复边界；
- 与 audio/progress high-water 相关的最小恢复元数据。

**需要完成：**

- 在 engine thread 安全边界 capture，不从 asyncio 线程拿 GPU tensor；
- opaque handle 的 owner、generation、容量和释放；
- `PAUSE_RESUME`、`SEGMENT_HANDOFF`、`CONTEXT_ROLLOVER` 分开实现和声明；
- exact、trained-approximate、reconstruction-approximate 分别验收；
- restore 失败时 hard boundary fallback，禁止静默混用半旧状态；
- 与 cursor state 一起迁移或明确禁止该组合。

**验收：** exact 路径比较 next logits、codec、Code2Wav 输出；近似路径做连续性、F0、energy、pause、click、speaker similarity 和 WER/CER 报告；handle 无泄漏、无跨 session 使用。

**依赖：** T0；可先于 native progress 实现，但和 T5 联测。  
**阻塞：** X2 的“状态继承”能力声明。

### T8：实现 segment handoff / context rollover

**Owner：** `engine/backend/engine_loop.py`、FSM、状态 adapter

**目标：** 把状态继承变成可调度的引擎状态，而不是一次 capture/restore API。

**需要完成：**

- 明确 `WAIT_TEXT`、segment end、soft drain、handoff、rollover 的状态转换；
- 处理持有 slot、held-slot lease、超时和 admission；
- 处理 audio credit 不足时的 gapped resume，并对外发布原因；
- rollover 失败时恢复到 hard boundary；
- 确保 progress high-water、audio reorder 和 state generation 一致。

**验收：** 长文本连续输入、文本停顿、断连恢复、slot 压力、状态迁移失败均有 FSM 和 E2E 测试；不因等待文本注入 PAD/EOS。

**依赖：** T7；Soft Drain 相关部分依赖 T9。  
**阻塞：** 低接缝长文本续接。

### T9：Soft Drain / coverage Phase 1

**Owner：** 模型训练、TRT 导出、engine loop、评测

**目标：** 按已有设计文档实现 `<tts_wait>`、coverage 和 soft drain，不在工程层伪造模型能力。

**需要完成：**

- 训练/导出 `<tts_wait>`、coverage lower bound、acoustic endpoint/tail gate；
- 新增 `SOFT_DRAIN`、`RECOVERY` 相关状态；
- step cap、early EOS、gate mismatch 和 fallback；
- 真实 LLM trace 的 underflow/gap/RTF/p95 评测；
- 保留现有 `WAIT_TEXT` 作为默认 fallback，直到质量门槛通过。

**验收：** coverage overrun、premature HOLD、premature codec EOS 为 0 或达到预先声明的阈值；不通过门槛不得默认启用。

**依赖：** T1、T8，以及模型训练产物。  
**阻塞：** Phase 1 soft drain 上线。

### T10：协议与音频重排可靠性收尾

**Owner：** gateway、output、reorder、client SDK

**需要完成：**

- AudioReorder 后端段永久不结束时的 stall timeout、健康检查和明确错误；
- native/EMA progress 在 guarded delivery、resume replay、terminal event 中保持单调；
- 统一 metadata schema，减少 timing accumulator、VAD 和内部 c2w 字段泄漏；
- Triton/Native/Realtime/gRPC 对同一 progress event 做 golden tests；
- client 对坏 anchor 降级 tracking，但继续收音频的行为保持稳定。

**验收：** transport disconnect、replay、duplicate terminal、late segment、invalid anchor、out-of-order audio 都有回归测试。

**依赖：** T0、T3；可与 T6 并行。

**当前进度：** 已完成重排 playhead 的 per-key stall 计时、零 timeout 的显式禁用、后段缓冲量
观测和 Frontend watchdog；超时后发出 `audio_reorder_stall_timeout` 告警，向 Engine 发送
带原因的内部取消请求，由统一 `SESSION_DONE` 路径释放 held audio、slot 和 session。尚缺
真实多 session/高并发压测，以及客户端在取消终态下的录制验收。

### T11：性能、并发和容量验收

**Owner：** runtime/performance

**需要完成：**

- cursor-enabled fused graph 与 standard plan 的单流、低并发、高并发 decode step、RTF、显存比较；
- CUDA Graph、eager、prefill、decode-only profile 分别测量；
- Label Plan 更新是否引入 GPU→CPU 同步或额外 TRT 调用；
- state handle、held slot、rollover 对吞吐和 p95/p99 的影响；
- 大文本、`cursor_max_labels` 上限、batch slot 隔离和 OOM 降级。

**验收：** 报告必须区分主 TRT 计算、CPU 编排、Triton 兼容层开销；不能把 Triton adapter 的网络/序列化成本误报为 TRT backend 成本。

**依赖：** T5、T7；图级部分可与 T4 并行。

## 4. 推荐并行批次

```text
Batch 0: T0 公共合同
          |
          +--> T1 TN -> Label Plan
          |       +--> T1b X2 commitment 主 TN 适配
          |               |
          |               +--> T2 slot/decode bridge
          |               |
          |               +--> T3 native progress
          |               +--> T5 native E2E
          |
          +--> T6 capability/routing
          |
          +--> T4 TRT parity（可与 T1/T2 并行，最终在 T5 汇合）

T0 ------> T7 speech state implementation ------> T8 rollover/handoff
T1/T8 ---> T9 Soft Drain / coverage
T0/T3 ---> T10 protocol/reorder reliability
T4/T5/T7 -> T11 performance/capacity
```

建议第一批分派给其他模型的是 T0、T1、T4、T6、T7 五个任务。T2/T3 需要等 T1 的合同稳定后实现；T5、T8、T9、T11 需要跨模块联调，不适合拆成互不知情的独立实现。

## 5. 不应重复实现的内容

- 不要新增 Triton TRT Backend；Triton 只作为兼容层。
- 不要把 native cursor 拆成 CPU PyTorch sidecar。
- 不要在 cursor 适配层复制 `IncrementalTextCommitter` 或另写一套 TN。
- 不要把 raw Unicode、owner span 或 raw coordinate 放入 TRT 图。
- 不要用 Talker BPE/prompt embedding 代替 cursor label embedding。
- 不要因为 Label Plan 内容变化而重捕 CUDA Graph；只更新固定 buffer 和 `label_count`。
- 不要在 native cursor 失败时重置 Talker、Code2Wav、KV 或音频流；只能降级 progress route。
- 不要把 transport resume 当成声学 state inheritance。
- 不要在没有 fingerprint 验证时静默启用 cursor-enabled artifact。

## 6. 总体完成标准

项目可以宣称“统一引擎能力完成”的最低条件是：

1. 三种文本输入模式都通过同一个主 TN → Label Plan → Splitter 合同；
2. native cursor 的 Label Plan、slot state、fused TRT、CPU projector 和 progress event 已连通；
3. standard plan 使用 EMA，cursor-enabled plan 使用 native progress，二者通过 manifest 能力协商选择；
4. Native Gateway 和 Triton 兼容层调用同一个 Engine Runtime，并通过同一套 E2E 合同；
5. X2 的状态继承有真实 capture/restore、generation、释放和失败回退；
6. transport resume、speech state transfer、cursor reanchor 三种状态概念分别定义并测试；
7. 长文本、EOS、异常终止、tail rewrite、并发 slot、断连、重试和 progress high-water 全部有回归覆盖；
8. 性能报告能区分 TRT 主图、Engine Runtime、Gateway 和 Triton 兼容层的开销。

2026-09-08 外部 X2 upstream 集成复查结果：`test_upstream_extension_hooks.py` 已通过，
生命周期测试的 session capacity、Splitter 边界和 successor 段启动也已通过；该测试最后
仍把第二批 token 写死为原始 `25℃。`。当前统一引擎按主 TN 合同输出
`二十五摄氏度。`，因此这个失败是外部测试契约未同步，不应通过恢复 raw-text 或增加第二套
TN 来修复。外部测试应改为断言主 TN 的 spoken projection，或显式使用 IdentityNormalizer
并声明这是非生产 TN fixture。

## 7. 关键代码和设计文档

- 主 TN 与 session frontend：[engine/frontend/interface.py](../../../engine/frontend/interface.py)
- Splitter：[engine/frontend/spliter/](../../../engine/frontend/spliter/)
- 主 TRT executor：[engine/backend/executor.py](../../../engine/backend/executor.py)
- Engine loop：[engine/backend/engine_loop.py](../../../engine/backend/engine_loop.py)
- cursor fused export：[scripts/export/export_09_talker_code2wav_fused.py](../../../scripts/export/export_09_talker_code2wav_fused.py)
- slot state：[engine/backend/kv_cache_pool.py](../../../engine/backend/kv_cache_pool.py)
- speech state 合同：[engine/core/speech_state.py](../../../engine/core/speech_state.py)
- speech state adapter：[engine/backend/speech_state.py](../../../engine/backend/speech_state.py)
- progress：[engine/core/text_progress.py](../../../engine/core/text_progress.py)
- public capability：[engine/gateway/capabilities.py](../../../engine/gateway/capabilities.py)
- cursor/TRT 决策：[../architecture/native_cursor_trt_fusion_decision.zh-CN.md](../architecture/native_cursor_trt_fusion_decision.zh-CN.md)
- TN/Soft Drain 设计：[incremental_text_normalization_and_soft_drain.zh-CN.md](incremental_text_normalization_and_soft_drain.zh-CN.md)
