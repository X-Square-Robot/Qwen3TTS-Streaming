[English](observability_tiers.md) | **中文**

# 引擎可观测性分层设计（四层模型 + 升级排查链）

> 编写日期：2026-06-30
> 状态：**设计文档（实现前契约）**——本文锁定四层模型、控制面、各层字段与升级排查流程，
> 后续实现以此为参照，避免再引入一轮含糊命名。
> 关联：[[observability_goals]]（前置·计时/生命周期维度的目标与契约）· [[engine-design-overview]] ·
> [[frontend_segmentation_pipeline]] · [[vad_design_goals]] · [[decode_fsm]]

---

## 0. 本文与 `observability_goals.md` 的关系

[[observability_goals]] 定义了**计时/生命周期维度**的目标：稳定的事件命名、强/上下文指标、
raw/effective 音频、客户端可凭协议重建时间线。它是"**要观测什么**"的契约。

本文回答的是**正交的另一问题：分几层观测、每层装什么、怎么从一层升到下一层**。两者不重复：

- `observability_goals.md` 给出**事件命名空间**（`session.created` / `engine.prefill.completed` …）与**字段语义**；
- 本文把这些事件**按详细度切成四档**，补上它没覆盖的 **debug 决策日志**与 **dump 后验举证**，
  并定义一条**升级排查链**把四档串成排查流程。

事件名、字段名、强/上下文/原始/派生分类**一律沿用** `observability_goals.md`，本文只新增分层与 debug/dump 专属字段。

---

## 1. 问题与目标

### 1.1 现状（零件已散落，缺统一层级）

| 已有零件 | 文件 | 性质 |
|---|---|---|
| 结构化生命周期 JSON 日志 | `engine/core/lifecycle.py` (`LifecycleLogger.emit`) | 单一事件入口，但只发了规范 18 个 phase 里的 ~7 个 |
| 跨线程计时累加器 | `engine/core/timing.py` (`ServerTimingAccumulator`) | TTFT/链路分段齐全，序列化进协议 meta |
| 客户端自分析 | `client/src/qwen3tts/diagnostics.py` 等 | `ServerTimingReport.explain_latency()` 等，时间线分析强 |
| 全量 tensor dump | `engine/backend/debug_dump.py` (`EngineDebugDumper`) | `ENGINE_DUMP_*` env 控制，逐步 dump |
| 零散自由日志 | 各模块 `logger.info/debug` | engine_loop ~40 处，无统一语义 |

**结构性缺口**：日志级别硬编码 `INFO`（`engine/server.py:1060`），无运行时/配置/按 session 控制；
dump 是独立 env、debug 是各处 `isEnabledFor`，三者各管各的，**没有"从①升②再升③"的统一旋钮**。

### 1.2 目标排查流程（本文要建立的主线）

```
客户端自分析  ──无法解决──▶  日常日志(L1)  ──无法解决──▶  debug日志(L2)  ──无法解决──▶  dump日志(L3)
   L0                          常开                       开发机/定点开            疑难杂症
 高频问题秒级自助            日常反复排查的问题            偶发疑难                 模型后验/计算决策举证
```

让**高频问题快速自助**，让**困难问题有确定的升级路径**。每升一级，信息更细、成本更高、开启范围更窄。

---

## 2. 四层模型

| 层 | 名称 | 受众/场景 | 默认 | 体量 | 回答的核心问题 |
|----|------|----------|------|------|---------------|
| **L0** | 客户端自分析 | 客户端 SDK 使用者 | 始终可用（协议自带） | 0 服务端成本 | TTFT、链路分段、合成内容、batch、VAD —— **大部分日常问题** |
| **L1** | 日常日志 (DAILY) | 线上运维 | **常开** | 每请求几行 | 同 L0，但服务端视角更全；定位日常反复排查 |
| **L2** | debug 日志 (DEBUG) | 开发机/定点 | opt-in | 每请求几十行 | **子句怎么切的、为什么这么切、合成为什么错**（决策"为什么"） |
| **L3** | dump 日志 (DUMP) | 疑难杂症 | 手动单开 | 每步 tensor 文件 | 模型后验、采样/CP 计算决策的**原始举证** |

**层级是累加的**：`L3 ⊃ L2 ⊃ L1`。开 L2 自动含 L1 全部；开 L3 自动含 L2 全部决策日志 **再叠加** tensor dump
（决策日志与 tensor 用同一 `dump_id`/`frame_idx` 对齐，排查时"升到 dump"即拿到决策+原始证据的完整对照）。

**设计不变量**：

1. **同一事件命名空间，逐层加细**。层只决定*详细度*与*哪些可选事件触发*，不改事件名（沿用 `observability_goals.md`）。
2. **L1 的每条日常事实必须同时进协议 meta**——这是 L0 成立的前提。客户端答不出的，服务端 L1 用同名字段答，只是更全 → 升级链天然衔接。
3. **L2 = 解释（"为什么"，派生、人读、便宜）；L3 = 举证（原始 tensor/后验、机读、贵）。** 偶发问题用 L2，疑难杂症才升 L3。

---

## 3. 控制面（全局 + 按 session 覆盖）

### 3.1 级别旋钮

引入一等概念 `observability.level`，取值 `daily | debug | dump`（对应 L1/L2/L3；L0 永远在线，不需开关）。

**优先级（高 → 低，沿用 engine.yaml 既有约定）**：

```
单 session 覆盖  >  ENGINE_OBS_LEVEL 环境变量  >  engine.yaml observability.level  >  默认(daily)
```

`engine.yaml` 新增段：

```yaml
observability:
  level: daily                 # daily | debug | dump —— 全局默认档
  max_session_level: daily     # 允许客户端/请求把单会话升到的最高档（生产防滥用，见 §3.3）
  text_capture: preview        # disabled | preview | hashed | full（文本隐私，沿用 observability_goals §4）
  text_preview_chars: 64
  dump:                        # 仅 level=dump 或单会话升到 dump 时生效；等价现有 ENGINE_DUMP_*
    dir: workspace/engine_dump
    limit: 0                   # 0=不限
    include_wav: true
    sessions: ""               # 逗号分隔；空=全部
    input_keys: ""
    output_keys: ""
```

`ENGINE_OBS_LEVEL=debug` 等 env 覆盖全局档；现有 `ENGINE_DUMP_*` env 保留为 L3 细节旋钮的别名（向后兼容）。

### 3.2 按 session 覆盖（生产定点排查）

客户端在请求里携带 `output_policy.config["obs_level"]`（或 `timing.extra["obs_level"]`），
在 `session.config.validated` 阶段解析并钉到 `SessionConfig.observability_level` 上，
随后贯穿 `LifecycleLogger` 调用与 dumper gating。

意义：**生产里定点 debug 单个坏会话，而不必把全局调到 DEBUG 刷爆日志。** 这是排查链能在线上跑通的关键。

### 3.3 防滥用（重要安全约束）

按 session 升级**受服务端 `observability.max_session_level` 钳制**：

- 生产默认 `max_session_level: daily` → 客户端**无法**自行升到 debug/dump（防 dump IO 放大成 DoS、防文本泄漏）。
- 开发/预发环境设 `max_session_level: dump` → 允许客户端定点升级。
- 请求要求的档 > `max_session_level` 时，**钳到上限并在 `session.config.validated` 事件里记一条 `obs_level_clamped` 警告**，不静默吞掉。

### 3.4 运行时级别检查（实现约定）

- 决策/dump 的昂贵 payload 构造，一律先判级别再构造（沿用现有 `if not logger.isEnabledFor(DEBUG): return` 的 guard 风格），避免 L1 下产生 L2/L3 开销。
- `LifecycleLogger.emit` 增加可选 `level`/`min_level` 形参：低于当前生效档的事件直接 no-op。

---

## 4. L1 日常日志规范（常开、量少而精）

**形态**：每请求 = 少量规范生命周期事件（结构化 JSON，机读）+ **一条人读 session 摘要行**（grep 即懂）。
目标体量：稳态下每请求 INFO 行个位数。

### 4.1 五个日常问题 ⇄ 字段映射

| 日常问题 | 承载事件/字段 | 现状 |
|---|---|---|
| **TTFT 多少** | 摘要行 `server_engine_prefill_ms` / `server_first_text_dequeue_to_first_raw_audio_ms` / `server_session_create_to_first_raw_audio_ms` | timing.py 已算，缺人读摘要 |
| **客户端→链路花了多久** | 链路分段：session创建 / 文本接入 / 排队 / 推理 / 门控 / 传输（`observability_goals §验收标准`六段） | timing.py 已有派生，缺统一一行 |
| **合成了什么内容** | `final_synthesized_text` + 每段 `segment_text_preview`（受 `text_capture` 钳制） | accumulator/segment_end meta 有 preview，缺 session 级汇总 |
| **拼没拼 batch** | **新增** `engine.prefill.batched` / 摘要 `batch_summary`：每段 prefill 是单发还是混批、batch_size、同批 session/segment | **缺**——batching 决策现仅 DEBUG 散记 |
| **VAD 怎么处理的** | 摘要 `vad_summary`：`prefix_trimmed_ms` / `tail_trimmed_ms` / `begin_count` / `end_count` / `original_vs_effective_audio_ms` | `_inject_vad_metrics` 已有，仅进 done_meta，缺日常行 |

### 4.2 规范生命周期事件（补齐到全集）

沿用 `observability_goals.md §事件定义表` 的 18 个 phase。现已发 7 个，L1 应补齐以下（均 INFO、结构化）：

`request.accepted` · `session.registered` · `text.first_received` · `text.first_sent` ·
`engine.prefill.started` · `engine.decode.first_step` · `engine.audio.first_raw` ·
`output.audio.first_effective` · `session.completed` · 错误路径 `session.evicted` / `engine.prefill.failed` / `session.cancelled`。

### 4.3 Session 摘要行（替代含糊 `first_audio=231.6ms`）

请求完成时发一条结构化摘要，聚合 §4.1 五问 + 缓存路径。字段直接复用 `observability_goals §2 Session 汇总日志`
与 `timing.py` 的派生指标，**新增** `batch_summary`：

```json
{
  "lifecycle": true, "phase": "session.summary", "session_id": "abc123",
  "ttft": {"prefill_ms": 28.3, "dequeue_to_first_raw_ms": 12.1, "create_to_first_raw_ms": 231.6},
  "pipeline_ms": {"session_create": 5.2, "text_ingress": 8.1, "engine_queue": 3.0,
                  "inference": 190.0, "gating": 8.5, "transport": null},
  "text": {"final_synthesized_text": "…", "total_chars": 142, "total_segments": 3,
           "coalesced": false, "progress_protected": false},
  "batch_summary": {"segments": 3, "batched": 2, "solo": 1, "max_batch_size_seen": 4},
  "vad_summary": {"policy": "tenvad", "prefix_trimmed_ms": 8.5, "tail_trimmed_ms": 0.0,
                  "begin_count": 1, "end_count": 1, "original_audio_ms": 1820, "effective_audio_ms": 1811},
  "cache": {"prefix_cache_hit": true, "cache_tokens_reused": 512, "full_prefill_count": 1},
  "total_latency_ms": 1520.3
}
```

人读伴随行（一行 grep 即懂）：

```
INFO engine.lifecycle session=abc123 DONE ttft=231.6ms infer=190.0ms batch=2/3 vad_trim=8.5ms cache=HIT segs=3 "今天天气…"
```

---

## 5. L2 debug 日志规范（开发机/定点、解释"为什么"）

L2 在 L1 之上叠加**决策理由记录**（结构化、派生、人读、不含 tensor）。核心回答你列的三问：
**子句怎么切的、为什么这么切、合成为什么错。** 每类记录有稳定 `obs` 标签便于过滤。

### 5.1 切分决策（"为什么这么切"）—— 最大缺口

切分发生在 `engine/frontend/spliter/spliter.py`：阈值由 `_make_thresholds()`→`compute_thresholds(remaining_kv, ema_ratio, …)`
算出，`pre_split()` 在 L1 标点且 `n >= min_tokens_l1` 处切、`n >= force_split_at` 时回退 last_l1 或硬切；
流式路径由 driver FSM 用 L1/L2/L3 阈值反应式切。**这些"为什么"目前完全不可见。** L2 每产生一个切点记一条：

```json
{
  "obs": "split_decision", "session_id": "abc123", "segment_id": 2, "group_idx": 1,
  "path": "offline_pre_split",              // offline_pre_split | streaming_driver
  "trigger": "force_fallback_l1",           // l1_punct | force_fallback_l1 | force_hard_cut | driver_l1/l2/l3
  "remaining_kv": 468, "prefill_len": 12, "ema_ratio": 6.0,
  "thresholds": {"min_tokens_l1": 38, "force_split_at": 327,
                 "l1_cap": 327, "l2_cap": 374, "l3_cap": 421},
  "token_count_at_split": 327, "last_l1_pos": 41, "chosen_level": 1,
  "reason": "force_split_at(327) reached without L1 since pos 41; fell back to last L1 at 41",
  "text_preview": "……他说，"
}
```

排查"这句为什么被切碎/为什么没在该断的地方断"时，一条记录即给出阈值、KV 水位、EMA、触发分支与人读理由。

### 5.2 batch 组成（"为什么和它拼/没拼"）

prefill 批在 `engine_loop._try_prefill_pending/_try_prefill_one`（按 `RequestPriority` + `kv_pool.free_count` + 每会话 slot 上限）；
decode 批在 `_get_active_slots_mlfq`（MLFQ + 溢出/静音剔除）。L2 每批记一条：

```json
{
  "obs": "batch_compose", "stage": "prefill",          // prefill | decode
  "batch_size": 3, "kv_free_before": 60,
  "members": [{"session_id":"a","segment_id":0,"priority":"FIRST_SEGMENT"},
              {"session_id":"b","segment_id":1,"priority":"CONTINUATION"}],
  "skipped": [{"session_id":"c","segment_id":0,"reason":"max_slots_per_session"}]
}
```

### 5.3 VAD 状态跳变（"为什么裁了这段"）

L1 只给 VAD 汇总；L2 给每次 begin/end 跳变（来自 `vad_processor.process_chunk`）：

```json
{"obs":"vad_transition","session_id":"abc123","event":"begin","frame_ms":120.0,"prob":0.82,"threshold":0.5}
```

### 5.4 合成异常/采样汇总（"合成为什么错了"）

每段收尾记一条采样+收尾汇总（段级，非每步——每步属 L3），并对可疑段打**启发式异常标记**：

```json
{
  "obs": "segment_synthesis", "session_id": "abc123", "segment_id": 2,
  "do_sample": true, "temperature": 0.9, "repetition_penalty": 1.1, "sampling_seed": 178412,
  "audio_steps": 511, "text_tokens": 40, "audio_text_ratio": 12.8,
  "eos_reason": "kv_overflow",              // codec_eos | silence_abort | kv_overflow
  "anomaly": ["hit_kv_512", "ratio_outlier_gt_clamp"],   // 空数组=正常
  "reason": "ran to KV cap 512 without codec EOS; ratio 12.8 > clamp 10 → likely hallucination tail"
}
```

`eos_reason=kv_overflow/silence_abort` + `ratio` 离群 + 命中 512，三者组合正是 [[engine-design-overview]] 记录的
C4 幻觉/不收尾征兆。L2 把"合成为什么错"从日志考古变成一条带 `reason` 的记录。

---

## 6. L3 dump 日志规范（疑难杂症、原始举证）

L3 = **L2 全部决策日志** + **`EngineDebugDumper` 逐步 tensor/后验**，两者对齐。

复用现有 `engine/backend/debug_dump.py`：`level=dump`（或单会话升 dump）时自动置位等价 `ENGINE_DUMP_DIR`，
继承其 `.pt` payload + `timeline.jsonl`/`.tsv` 索引 + 按 session/key 过滤能力。**新增对齐**：

1. 把 §5.1 `split_decision` 决策快照写入 dump 目录（`split_decisions.jsonl`），与 tensor 同 session 目录。
2. dump 的 `timeline.jsonl` 每行补 `sampling_seed` / `eos_check`（`full_codec[:,0]` vs `codec_eos_id`），
   使**采样后验、CP 计算决策**与 L2 的 `segment_synthesis` 经 `dump_id`/`frame_idx` 可交叉引用。
3. dump 入口处发一条 `obs:"dump_enabled"` lifecycle 事件（含 dir/limit/sessions），让日志能定位 dump 产物位置。

排查"模型后验为什么这样、CP 哪步翻的"时，从 L2 的 `segment_synthesis` 异常段 → 按 `session_id` 进 dump 目录 →
逐步 logits/codec/CP stage 输出 + 同步的采样 seed，形成决策→证据闭环。

---

## 7. L0 客户端自分析规范（高频问题自助）

客户端凭返回协议**无需后台捞日志**即可回答 L1 的大部分问题。现有 `qwen3tts.diagnostics`
（`ServerTimingReport.explain_latency()` / `SegmentTimingReport` / `ErrorTimingReport` / `LatencyAnalyzer` / `TimelineReconstructor`）
已覆盖时间线；**缺口**是把 §4.1 五问串成一个"先看这个"的统一入口。

新增 `SessionDiagnostics.summary()`，对齐 §4.3 服务端摘要，回答同样五问 + 给出**升级建议**：

| 客户端能自答 | 来源协议字段 | 答不出时升级提示 |
|---|---|---|
| TTFT、链路分段哪段慢 | done_meta 派生时长 | "推理段占比高 → 看服务端 L1 `session.summary`" |
| 合成了什么 | `final_synthesized_text` / `segment_text_preview` | "段文本与预期不符 → 升 L2 看 `split_decision`" |
| 拼没拼 batch | done_meta `batch_summary`（§4 新增进协议） | —— |
| VAD 裁了多少 | done_meta `vad_summary` | "裁切异常 → 升 L2 看 `vad_transition`" |
| 失败在哪个阶段 | error 事件 `error_phase` | "阶段定位后 → 升 L2/L3" |

**关键**：`summary()` 在答不出时**直接打印下一级排查指令**（"在服务端 grep `session=<id>` 看 L1"或"用 `obs_level=debug` 重放"），
把升级链做成客户端可见的产品路径，而非口口相传。

---

## 8. 字段矩阵（事件 ⇄ 日志名 ⇄ 协议名 ⇄ 层级）

> **完整指标目录见 [[observability_metrics_catalog]]**——对全引擎三路穷举扫描得到的 ~400 原始字段，
> 折叠成 ~45 个观测点的单一真相源，是实现的逐项检查单。下表仅保留主干示例，避免与目录重复维护。

这是 `observability_goals.md §立即的下一步` 要求产出的矩阵，按本文四层补全。`现状`列：✅已有 / 🟡部分 / 🆕新增。

| 事件/字段 | 组件 | 日志字段名 | 协议字段名 | 层 | 强/上下文 | 现状 |
|---|---|---|---|---|---|---|
| 规范生命周期 7 phase | lifecycle | 各 phase | done_meta epoch 时间戳 | L1 | 强 | ✅ |
| 规范生命周期补 11 phase | lifecycle | 各 phase | — | L1 | 强 | 🟡 补齐 |
| TTFT 三度量 | timing | `server_*_first_raw_audio_ms` | 同名 | L1 | 强 | ✅ |
| 链路六分段 | timing | `pipeline_ms.*` | 各派生时长 | L1 | 强/上下文 | 🟡 汇总 |
| 合成内容 | interface | `final_synthesized_text` | 同名（受 text_capture 钳制） | L1 | 上下文 | 🟡 |
| **batch 汇总** | engine_loop | `batch_summary` | `server_batch_summary` | L1 | 强 | 🆕 |
| VAD 汇总 | gateway | `vad_summary` | done_meta 既有 trim 字段 | L1 | 强 | 🟡 提炼 |
| **切分决策** | spliter | `obs:split_decision` | —（仅服务端） | L2 | —— | 🆕 |
| **batch 组成** | engine_loop | `obs:batch_compose` | — | L2 | —— | 🆕 |
| **VAD 跳变** | gateway/vad | `obs:vad_transition` | — | L2 | —— | 🆕 |
| **段合成/异常** | engine_loop | `obs:segment_synthesis` | — | L2 | —— | 🆕 |
| 逐步 tensor/logits | debug_dump | `.pt` + `timeline.jsonl` | — | L3 | —— | ✅ |
| **dump↔决策对齐** | debug_dump | `split_decisions.jsonl` + seed/eos 字段 | — | L3 | —— | 🆕 |
| 客户端统一摘要 | client | — | 复用上列协议字段 | L0 | —— | 🆕 |

---

## 9. 与现有代码的映射（改造点）

| 关注点 | 文件 | 改造 |
|---|---|---|
| 级别旋钮 | `engine/config.py` + `engine.yaml` | 新增 `observability` 段、`ENGINE_OBS_LEVEL` 解析、优先级合并 |
| 日志初始化 | `engine/server.py:1060` | `basicConfig` 级别由 `observability.level` 驱动，不再硬编码 INFO |
| 级别 gating | `engine/core/lifecycle.py` | `emit` 增 `min_level`，低于生效档 no-op |
| 单会话档 | `engine/core/types.py` (`SessionConfig`) | 新增 `observability_level`，§3.2 解析 + §3.3 钳制 |
| L1 摘要 | `engine/frontend/interface.py` | `session.summary`/`session.completed` 聚合发射 |
| batch 观测 | `engine/backend/engine_loop.py` | `_try_prefill_*` / `_get_active_slots_mlfq` 记 L1 汇总 + L2 `batch_compose` |
| 切分决策 | `engine/frontend/spliter/spliter.py` | `pre_split` / driver 切点处记 L2 `split_decision`（阈值已在 `_make_thresholds`） |
| 段合成 | `engine/backend/engine_loop.py` | `_handle_segment_eos` 记 L2 `segment_synthesis` + 异常启发式 |
| VAD 观测 | `engine/gateway/grpc_server.py` | `_inject_vad_metrics` 提炼 L1 `vad_summary`；`process_chunk` 记 L2 `vad_transition` |
| dump 对齐 | `engine/backend/debug_dump.py` | level=dump 自动启用；timeline 补 seed/eos；写 `split_decisions.jsonl` |
| 客户端摘要 | `client/src/qwen3tts/diagnostics.py` | `SessionDiagnostics.summary()` + 升级提示 |

---

## 10. 分阶段推行

| 阶段 | 内容 | 产出 |
|---|---|---|
| **P0 控制面** | `observability` 配置段 + 级别旋钮 + 优先级 + 单会话钳制 + lifecycle `min_level` gating | 一个旋钮统管三档，生产可定点升级 |
| **P1 L1 日常** | 补齐生命周期事件 + `session.summary` 摘要 + `batch_summary` + `vad_summary` 提炼 + 同步进协议 | 五个日常问题各有一行可 grep；客户端拿到同名字段 |
| **P2 L0 客户端** | `SessionDiagnostics.summary()` + 升级提示 | 高频问题客户端自助，升级链入口可见 |
| **P3 L2 debug** | `split_decision` / `batch_compose` / `vad_transition` / `segment_synthesis` | 偶发疑难有"为什么"，三问可答 |
| **P4 L3 dump 对齐** | dump 自动启用 + 决策日志对齐 + seed/eos 字段 | 决策→tensor 证据闭环 |

P0→P1→P2 是排查链的"高频半边"，优先级最高；P3→P4 是"疑难半边"，在前两段稳定后接入。

---

## 11. 验收标准

1. **一个旋钮**（config/env/单会话）即可在 daily/debug/dump 间切换，生产默认 daily 且客户端无法越权升级。
2. **L1 常开下**，一条 `session.summary` + 一行人读摘要即回答 TTFT/链路/合成内容/batch/VAD 五问，无需跨文件对齐时间戳。
3. **L0 客户端**仅凭协议回答上述五问的大部分，并在答不出时打印明确的升级指令。
4. **L2** 下，"子句为什么这么切"由一条 `split_decision` 给出阈值+水位+理由；"合成为什么错"由 `segment_synthesis` 的 `eos_reason`+`anomaly`+`reason` 给出。
5. **L3** 下，L2 决策记录与 tensor dump 经 `session_id`/`dump_id`/`frame_idx` 可交叉引用，形成决策→证据闭环。
6. 字段名稳定，可作为协议契约一部分（沿用 `observability_goals.md` 命名）。

---

## 12. 非目标

- 不引入 OpenTelemetry / Prometheus（语义稳定后再议，沿用 `observability_goals §非目标`）。
- 不保证客户端与服务端全局时钟同步。
- L3 不追求实时——它是离线举证工具，体量大、单开。
- 不做 UI 仪表盘。
