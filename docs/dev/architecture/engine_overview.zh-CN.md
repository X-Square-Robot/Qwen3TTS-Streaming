[English](engine_overview.md) | **中文**

# 引擎设计全景总览（约束 → 设计 → 缺陷 → 可避免性）

> 编写日期：2026-06-30
> 状态：**索引/综述**——这是理解全引擎的入口地图，链接到各细分文档，不重复其内容
> 用途：**防止反复调研**。后续要理解"某个设计为什么这么干、缺陷在哪、能不能避免"，先读本文，再按链接深入
> 关联：[[engine_decisions]] [[decode_fsm]] [[mixed_precision_plan]] [[trt_llm_runtime_route_report]] [[frontend_segmentation_pipeline]] [[observability_goals]] [[observability_tiers]] [[realtime_audio]] [[vad_design_goals]] · 投资调查 `docs/dev/investigation/streaming_hallucination.zh-CN.md`、`code2wav_state_size.zh-CN.md`

---

## 0. 一切的源头：四个硬约束（C1–C4）

整台引擎每一个设计都是这四个约束的下游产物。所有"为什么"都回指这里。

| # | 约束 | 性质 |
|---|---|---|
| **C1** | 模型**自回归**，每段**固定 KV 预算**（512 step/slot 硬上限） | 硬件/模型，不可消除 |
| **C2** | 文本**异步到达**（接上游 LLM token 流），decode 必须能"等文本"（WAIT_TEXT） | 业务语义，不可消除 |
| **C3** | 前端**只能控文本 token**，音频 step 是下游产物，只能用 EMA 估（~3 step/中文字） | 三阶段架构所致，不可消除 |
| **C4** | 模型**不可靠吐 EOS**——某些 seed/采样组合永不收尾 → 跑满 512 → 幻觉（~10–18%） | 模型+采样，**非实现 bug** |

> 一句话：这台引擎是**在一个会溢出、会不收尾、只能间接控制、还要边等文本边干活的自回归模型上做实时流式 TTS**。复杂度几乎全部来自硬扛这四点。

---

## 1. 模型与运行时层

### 1.1 自建引擎而非 Triton
- **是什么**：抛弃 Triton 作主调度，自建 gateway + continuous batching + session 流控。
- **为什么**：Triton `dynamic_batching` 为无状态模型设计；自回归 decode 每请求 KV 长度不同，且需 C2 的 WAIT_TEXT 暂停/恢复，Triton 给不了。
- **缺陷**：运维设施（健康检查/加载/指标/优雅关闭）全自研。
- **可避免**：否，正确取舍。
- 详见 [[engine_decisions]]。

### 1.2 三阶段 decode + WAIT_TEXT（C1+C2 核心）
- **是什么**：prefill → 文本流式输入（边喂边出音频，无文本则 `WAIT_TEXT` 挂起保 KV）→ flush（注 EOS pad、drain）。`engine_loop.py:1187-1201`。
- **为什么**：流式 TTS 必须能在文本断流时保 KV 原地等，而非 pad 或重 prefill。
- **缺陷**：① pad 容忍实测**仅 1 token**，上游卡顿 pad≥2 即可感知停顿；② **无 EOS 保证**（C4），flush 靠 silence-abort 启发式硬切，阈值 cliff 式硬编码（`engine_loop.py:1280-1293`）。
- **可避免**：三阶段不可避免（C2）；pad/EOS 是模型能力，工程只能缓解。
- 详见 [[decode_fsm]]。

### 1.3 Padded continuous batching + MLFQ 调度
- **是什么**：迭代级调度，KV 用 **padding 到 batch 最大长 + mask** 对齐（**非 paged KV**）；MLFQ 风格优先级（新段保 TTFB、老段降级防饿死）。`scheduler.py`。
- **为什么**：编译好的 TRT engine `seq_len` 固定，paged/varlen 需 kernel 级改动，.plan 改不了。
- **缺陷**：长短请求混批 padding 浪费（~10%）；无抢占。
- **可避免**：能但代价高——需换 TRT-LLM PyTorch backend，评估为**中期路线非现在**（[[trt_llm_runtime_route_report]]）。

### 1.4 Code Predictor 展开、无 KV
- **是什么**：CP 15 步展开成单个静态 TRT 图，每步从头重算。`architecture.md §5`。
- **为什么**：CP 的 KV 省下来仅占 ~3% 延迟，但展开成 GEMM 比 GEMV 的 GPU 利用率高。
- **缺陷**：**FP16 坏掉**（stage argmax 敏感），只能 fp32/bf16；图大编译慢。
- **可避免**：FP16 不可用是数值本质。

### 1.5 混合精度 bf16/fp32/bf16
- **是什么**：backbone=bf16、CP=fp32、code2wav=bf16。
- **为什么**：CP bf16 下 stage2 近似平局被舍入翻转 → 级联幻觉（Finding #15）。
- **缺陷（关键认知）**：**精度不是幻觉根因**（Finding #17-18）——各精度同种子都 ~10-18%，全 fp32 反而 17.5% 更糟。CP→fp32 是数值必要、非幻觉充分解，只是"换一批坏种子"。
- **可避免**：CP fp32 必须留；别指望它治幻觉。
- 详见 [[mixed_precision_plan]]、`streaming_hallucination.md`。

### 1.6 Prefix KV cache
- **是什么**：16 条 LRU，缓存固定前缀 talker KV，命中跳过 prefill。`prefix_cache.py`。
- **为什么**：custom_voice 类请求 system prompt 相同，省 10-50ms。
- **缺陷**：仅精确 token 匹配；⚠️**子 agent 推断（未复核）**：miss 时可能跑两遍 TRT、多 token suffix 命中收益不均。
- **可避免**：exact-match 限制可用前缀树改进，属可优化项。

---

## 2. 文本切分 / 流控层（重设计进行中，见 [[frontend_segmentation_pipeline]]）

### 2.1 文本 token 唯一控制面 + EMA（C3）
- **是什么**：前端只能决定何时停喂/flush，音频开销=文本token×EMA比；EMA 在 SEGMENT_END 更新、clamp[2,10]、overflow 时 α=0.5。`spliter.py:583`。
- **缺陷**：① EMA 是估计，估错时整条阈值阶梯一起平移；② **clamp 在 10 饱和**，真实比>10 永远低估不收敛；③ **反馈延迟**，溢出后 1-2 段仍可能溢出。
- **可避免**：能大幅缓解——**KV 水位根治**（flush 时机判断换实测）；但段间装箱在 decode 前、无水位，**仍靠 EMA**（C3 残留硬核）。

### 2.2 三层阶梯 FSM + spliter 双路径
- **是什么**：driver FSM 用 L1/L2/L3+force 三层阈值反应式切；spliter 有流式 `feed_tokens` 与离线 `pre_split` 两条路径。`driver.py`、`spliter.py`。
- **缺陷**：① **碎片化**——driver 在 0.7cap 第一个 L1 就切；② **双路径各带一套驱动循环+背压队列**，是分叉非流水线，emoji/auto 横切需求无处安放。
- **可避免**：**能**，纯意外复杂度——统一 `_drive_events`+单 `_pending`+层级装箱+auto。

### 2.3 AudioReorder 分层重排
- **是什么**：`(group_idx, local_idx)` 二级坐标，并发段乱序完成时按文本序重排。`reorder.py`。
- **缺陷**：**无 stall 超时**，某段后端永不送 SEGMENT_END 则后续音频永久卡住。
- **可避免**：能，加超时/健康检查。

---

## 3. 输出层（均为 C4 善后）

VAD 输出门控（裁前导/尾部幻觉静音）、等时音频流（补静音给 WebRTC jitter buffer）、14 阶段可观测性。
- **为什么**：C4 幻觉模型层治不了，在输出端事后补救（VAD ~0.1-0.2ms，对引擎透明）。
- **共同缺陷**：**都不降 TTFT**——裁掉/填充的部分仍先合成了。改善体验（断音、噪声），非延迟。
- 详见 [[vad_design_goals]]、[[realtime_audio]]、[[observability_goals]]。

---

## 4. 协议 / 客户端层

- **是什么**：`SynthesizeOnce`（一元强制 FULL_TEXT）+ `SynthesizeStream`（双向）；InputMode/GroupPolicy 暴露给客户端；元数据用 string map；gRPC+WS 共享 transport-agnostic 核心。`proto/tts.proto`、`engine/gateway/`。
- **缺陷（抽象泄漏一串）**：① **InputMode 把内部切分粒度泄漏给用户**；② `ref_audio` 暴露 `c2w`/`ref_codec` 引擎内部条件；③ **`SynthesizeOnce` 静默覆盖** 客户端 InputMode→FULL_TEXT；④ **输入完成度 transport-scoped**，引擎区分不了"流还开着在等" vs "流关了该收尾"；⑤ VAD 参数**字段+config dict 双重表示**；⑥ timing accumulator 塞 `timing.extra` 协议字段（破坏序列化）；⑦ 元数据 schema 在注释里、无版本协商。
- **可避免**：大部分能（意外复杂度）。auto 默认+保留显式档解决①；其余协议卫生可收敛。唯④由 C2 决定（完成度必须显式告知）不可完全消除。

---

## 5. 核心结论：硬约束 vs 意外复杂度

### 🔴 不可避免（C1–C4 本质，只能缓解）
| 缺陷 | 根 | 能做的 |
|---|---|---|
| 幻觉 ~10-18% | C4 模型+采样 | VAD 善后、EOS 调温、强制截断；**治本靠训练** |
| KV 512 溢出风险 | C1 硬预算 | 装箱、水位提前切、尾巴结转（不丢） |
| 只能控文本、音频靠估 | C3 三阶段 | 水位把段内估计换实测；段间装箱仍靠 EMA |
| 完成度要显式告知 | C2 异步文本 | 协议理清信号，消不掉 |
| pad≤1 token / 无 EOS 保证 | C4 模型 | silence 启发式；治本靠训练 |

### 🟢 可避免（意外复杂度，待清理 = 我们要解决的）
| 缺陷 | 解 | 状态 |
|---|---|---|
| spliter 双路径分叉 | 统一 `_drive_events`+单队列 | [[frontend_segmentation_pipeline]] 已定 |
| 切分碎片化 | 层级装箱+抬 t1 | 已定 |
| EMA 估错致丢句 | KV 水位根治 | 已定 |
| emoji 跨包漏字 | Stage 0 有状态 filter | 已定 |
| InputMode 泄漏 | auto 默认+保留显式档 | 已定 |
| Reorder 无超时 | 加超时 | 待办 |
| 协议泄漏（c2w/timing accumulator/双 VAD/无版本） | 协议卫生收敛 | 待办 |
| EMA clamp 饱和、overflow α 猛拽全局 | 区分离群/系统漂移、放开 clamp | 待办 |

---

## 6. 一句话总结

> 这台引擎**该硬扛的地方扛得对**（自建 runtime、三阶段 WAIT_TEXT、padded batching、CP fp32、prefix cache）；痛点集中在两处**可消除的意外复杂度**：(a) 文本切分层的双路径分叉 + EMA 估计盲区，(b) 协议层的抽象泄漏。清理这两处即是当前工作主线。

## 7. 可信度备注

架构主干（C1–C4、三阶段、padded batching、CP无KV、混合精度、幻觉根因、双路径、协议泄漏）多来源交叉印证，**有把握**。标⚠️"子 agent 推断未复核"的少数项（prefix cache 双 TRT pass、engine_loop 若干竞态、FULL_TEXT 无上限 OOM）落地前建议各花十分钟实证。
