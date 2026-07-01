[English](frontend_segmentation_pipeline.md) | **中文**

# 前端文本切分流水线重设计（auto 模式 / Spliter 统一 / KV 水位安全网）

> 分支：`main`
> 编写日期：2026-06-30
> 状态：**设计稿**（多轮讨论的共识与决策记录，待实现）
> 范围：`engine/frontend/`（interface / spliter / driver / dispatcher / reorder）+ `proto/tts.proto` 输入模式 + `engine/backend/` 的 KV 水位上报
> 关联：[[realtime_audio]] [[observability_goals]] [[mixed_precision_plan]]

---

## 0. 背景与动机

### 0.1 协议把内部实现细节泄漏给了用户

当前协议由**客户端**控制文本粒度：`proto/tts.proto:58` 的 `InputMode { TOKEN, CLAUSE, LONG_SEGMENT, FULL_TEXT }` + `GroupPolicy { NONE, AUTO }`。但绝大多数用户并不知道自己的文本是 token 级还是 long-text 级——这是一个**抽象层泄漏**：把"前端要不要做离线分段"这个引擎内部决策，强加给了没有信息也没有义务做此决策的用户。

四档 `InputMode` 实际混淆了三根**正交**的轴：

1. **输入到达形态**：客户端是"手上有完整文本"，还是"边产边喂、后面还有"（典型是接上游 LLM 的 token 流）。
2. **输出递交意图**：要低首包延迟、边合成边播，还是要一个完整音频文件。
3. **内部切分粒度/质量旋钮**：`TOKEN` / `CLAUSE` / `LONG_SEGMENT` 本质是"首包延迟 ↔ 鲁棒性"的权衡。用户被迫去选第 3 轴，但他只知道第 1、2 轴。

### 0.2 困境：两个临时需求挤不进原结构

- **emoji 过滤**（`engine/frontend/interface.py:51-56` `_normalize_tts_text` → `strip_emoji`）当初是临时加的。
- **auto 模式**（按内容自动决定要不要离线分句）现在需要，它要求整合"流式路径"和"离线路径"。

这两个横切需求塞不进原有的"双并列路径"结构，正是本次重设计的触发点。

### 0.3 根本病因（一句话）

> 设计意图是**一条流水线**（归一化 → 全局预切 → 喂 backend 的终点 driver），但**代码长成了一个分叉**：流式 `feed_tokens` 与离线 `push_group_tokens`/`set_full_text` 是两条并列入口、各带一套独立的 driver 驱动循环，只共享 driver 池与并发计数。需要会话级状态的横切关注点（emoji 跨包、auto）因此无处安放。

---

## 1. 核心洞察

### 1.1 边界裁定权（boundary authority）

两条路径的本质区别**不是**"流式 vs 离线"，而是**段边界由谁拍板**：

- 离线组：边界由 `pre_split` 上游拍板（组尾喂 `END` 逼 driver flush，`spliter.py:435`）。
- 流式：边界由 driver 用本地阈值自己拍板。

除此之外两条路径都只是在**把 `SpliterEvent` 喂进同一种 Driver FSM**。`SpliterEvent` 已有 `START / token / END`，`END` 语义恰好就是"关掉当前段"。

### 1.2 全局最优的作用域 = 一包文本

`pre_split` 的全局最优只能在它**看得见的范围**内成立。例：`你好吗？明天天气不错，有没有什么想吃的？`
- 流式 driver 无全局视野，可能切成 `你好吗？明天天气不错，` + `有没有什么想吃的？`（在 L2 逗号处溢出切，局部最优）。
- 全局最优应在 L1 处切：`你好吗？` + `明天天气不错，有没有什么想吃的？`。

推论（对设计有利）：**per-packet auto 的质量天然随客户端每包给的文本量增长**——客户端用"包的大小"隐式表达了优化作用域，无需命名一个 mode。流式下若 token 一个个来，没有任何阶段能找到跨包全局解，除非缓冲；而缓冲=人为加延迟，**明确否决**。

### 1.3 阶梯必然是文本 token 单位——TTS 三阶段架构所致

这**不是缺陷，是架构必然**。TTS 合成分三阶段：

1. **prefill**：文本 token 预填入 KV；
2. **文本流式输入阶段**：边喂文本 token 边出音频（`WAIT_TEXT` 语义）；
3. **flush 阶段**：不再有文本，pad EOS/NOP，drain 剩余音频。

三阶段里**前端唯一能控制的就是文本 token**（何时停喂 / flush）；音频 step 是下游产物，前端**只能估、不能直接控**。估算靠 EMA 的文本:音频比：中文 ~240ms/字 ÷ 80ms/chunk ≈ **3 step/字**，再受停顿、语速、分词字数影响而漂移，故用 EMA 滑动估计（`spliter.py:583 update_ratio`，初值 5.0，clamp `[2,10]`）。

由此：

- driver 的 `force_split_at` 按 `_token_count >= force_split_at` 触发（`driver.py:178`），只能是**文本 token 计数**——这是控制面本身。
- 真实约束（KV 溢出）在**音频 step** 维度，换算全靠 `ema_ratio`。
- `t1/t2/t3/force` 整条阶梯被同一个 `ema_ratio` 定位 → **ratio 一错，四根 rung 一起平移，没有任何一根兜得住**。

> 阶梯的"梯度"只防标点位置变化（逗号在哪、有没有 L1）；它**不防 ratio 估错**。ratio 误差需要把"估计"替换成"实测"——即 §5.3 的 KV 水位，但注意：水位只能改善**decode 过程中**的 flush 时机判断；Stage 1 装箱发生在 decode **之前**，无水位可查，**不可避免仍靠 EMA**。

---

## 2. 协议层变更

| 项 | 现状 | 目标 |
|---|---|---|
| **新增 auto** | — | `INPUT_MODE_UNSPECIFIED=0` 语义化为"引擎自动决定"，**作流式 RPC 默认**——小白用户不知道粒度差异，用 auto 即对 |
| `InputMode` 流式三档 | `TOKEN/CLAUSE/LONG_SEGMENT` 用户必选 | **保留为一等入口**——高级用户了解细节、能针对自己场景选合适 mode，应为他们保留显式控制（不是 override、不 deprecate） |
| `FULL_TEXT` | 一档 | **保留**——它表达唯一不可推断的轴：**封口 vs 开口**（还允不允许追加） |
| `GroupPolicy` | NONE/AUTO | 随 Stage 1 统一后语义并入各 mode 的内部行为 |

> 设计原则：**auto 是默认（覆盖大多数不知道粒度差异的用户），但不剥夺高级用户的显式选择权。** auto 把"该走哪档"的决策从"用户必须懂"变成"用户可以不懂"，而非"用户不准选"。

顶层意图其实**已经**由 RPC 表达：`SynthesizeOnce` 强制 `FULL_TEXT`，`SynthesizeStream` 默认流式（`grpc_server.py:555,591`）。本次新增的是流式内部"粒度自动决定"的能力，作默认；显式三档继续可用。

---

## 3. 三段流水线架构

```
Stage 0  有状态 filter chain（emoji 跨包 healing、空白归一化）
Stage 1  L1≫L2≫L3 层级装箱（乐观容量=EMA均值cap）→ 每段 threshold profile
Stage 2  单一 _drive_events 核心
          ├─ 阶梯触发器: 文本计数(无水位时) → KV 水位跨档(根治)
          ├─ 主动切 + 尾巴结转(overflow_token_ids), 零丢句
          ├─ 单 _pending 队列 + max_concurrent(受 KV pool 上界约束)
          └─ SegmentAction(token_text 含 pad) → 文本播放器
裁定权: 自始至终在 Stage 2，Stage 1 只提供全局边界建议
```

### 3.1 Stage 0：有状态 filter chain

**确认的 bug（后台调查）**：`strip_emoji`（`engine/text_normalization.py`）是逐字符、无跨包状态、无 lookahead 的扫描器；TOKEN/LONG_SEGMENT 每包独立归一化（`interface.py:198`）。一个多码点 emoji 的码点被拆到两包就漏网：

- keycap `1️⃣`、ZWJ 串 `👨‍👩‍👧`、肤色修饰 `👋🏻`、国旗 `🇺🇸` 全中招；
- 最难看：keycap 的 base `1` 不被识别为 emoji，`"hello1"`+`"️⃣world"` → `"hello1world"`，`1` 被当正文读出；
- `FULL_TEXT` 免疫（先 buffer 再整体归一化，`interface.py:205,248`）；
- 现有测试只测整串（`tests/unit/frontend/test_frontend_interface.py:22`），无拆包用例。

**解法**：Stage 0 是**有会话状态**的 filter（不是纯函数）。只在**包尾**扣下"可能是半个 emoji 的尾巴"（悬挂 ZWJ、孤立 regional indicator、base+VS 等），其余立即放行；下一包拼上再判，`mark_input_complete` 时 flush 残尾。**仅当包边界恰落在 emoji 中间才延迟一包、至多几个码点**——与被否决的"攒窗口测长度"是两码事。

emoji 与 auto 是**同一病根的两个症状**：本该有会话状态的阶段被写成了无状态逐包函数。

### 3.2 Stage 1：层级装箱（离线切分算法）

详见 [§4](#4-离线切分算法层级装箱)。产物 = **打包好的 L1 单元组 + 每组一份 threshold profile**（不是硬 END 事件流——见 §5.2 对早期方案的修正）。

### 3.3 Stage 2：统一驱动核心

- 用一个 `_drive_events` 取代 `feed_tokens`（`spliter.py:451`）与 `_drive_group`（`spliter.py:389`）两套重复循环。
- `_token_buffer`（流式）+ `_presplit_groups`（离线）+ `_presplit_thresholds` + `_next_group_idx` → 塌缩为**单一 `_pending` 队列**。这是含金量最高、也最危险的一步。
- **裁定权自始至终在 Stage 2**：driver 永远保留完整阶梯和"塞不塞得下"的最终决定权；Stage 1 只提供全局边界**建议**。

---

## 4. 离线切分算法：层级装箱

### 4.1 碎片化真凶：driver 会复切内部 L1

`compute_thresholds`（`driver.py:51-84`）：`cap = remaining_kv / ema_ratio`，`t1=0.70cap, t2=0.80cap, t3=0.90cap, force=cap`。driver `_meets_split_threshold`（`driver.py:160`）是"遇到首个达标点就切"，L1 在 `tc≥t1=0.7cap` 就切。

关键：`_drive_group` 给离线组喂的是普通 driver（`spliter.py:404`），它会在组内 `tc≥t1` 的**第一个 L1** FLUSH、把剩余 token 退回队列（`spliter.py:427-445`）。**所以即使 `pre_split` 打包成长 segment，driver 也会在 0.7cap 处复切回去** → 碎片化。

### 4.2 目标：词法优先级装箱，不是多维同权

> **不是** L1/L2/L3 同权的多维装箱（同权会在逗号处过切，毁掉韵律）；**是 `L1 ≫ L2 ≫ L3 ≫ 硬切` 的层级优先**。

- **主轴**：在容量内切在**最晚的 L1**（哪怕箱没装满也优先 L1，不为多塞两字去切逗号）。
- **回退**：仅当**单个 L1 单元本身超容量**时，才降级切最晚的 L2 → L3 → 硬切。
- `pre_split` 必须 **L2/L3-aware**（要知道回退切点落哪），但走优先级而非加权目标。
- `pre_split` 的层级与 driver 的 `_meets_split_threshold` 阶梯是**同一套层级的两个视角**：pre_split 全局 latest-fit，driver 本地 first-past-threshold。统一后应共用一份 tier 定义与 `_make_thresholds`，避免两个真相源。

### 4.3 装箱容量与并发的耦合（关注点 #1）

- 长文本 → L1 单元很多 → 即便按 `cap` 最大装箱，箱数仍 ≥ `max_concurrent`，并发照样吃满，**不冲突**。
- 仅中等长度（箱数 < `max_concurrent`、GPU 又空）才有取舍。
- 收敛点：**装箱目标容量 = `min(cap, cap / 目标并行度)`**。质量优先→`cap`（最长）；延迟/利用率优先→调小造更多箱喂满并发。`max_concurrent`（`spliter.py:132`）管"同时跑几个箱"（Stage 2 调度），箱容量管"每个箱多大"（Stage 1 质量）。
- **约束**：KV 上限固定 512/slot（`kv_cache_pool.py:43`），并发段共享 KV pool 的 slot → `max_concurrent` 实际被 pool 容量上界卡着；"目标并行度"须与 pool 能开多少 slot 对齐。

---

## 5. fit 裁定权与 KV 水位根治

### 5.1 为什么不能"obey 模式关掉阶梯"

若把 driver 设成纯 obey（t1/t2/t3 抬到无穷、只留 force），则 fit 的决定权从 Stage 2 分叉到 Stage 1 的**先验估计**。而 `force` 也只是 `ema_ratio` 快照算出的估计值；估计偏乐观时长段解不完、KV 耗尽、尾巴落地 → **大量丢句**。所以阶梯（自适应 fit 担保）**必须永远留在 Stage 2**。

### 5.2 正确关系：Stage 1 只"抬高 driver 的偏好切点"

> driver 永远保留完整阶梯与最终裁定权（不分叉）；Stage 1 用全局视野把这一段的偏好切点（`t1`）往后推到"容量内最后一个安全 L1"，并**保留 `t2/t3/force` 的 KV 安全含义**。

机制几乎免费：`_create_driver(thresholds)`（`spliter.py:205`）本就按段接收 thresholds。Stage 1 给每个打包段算一份定制 `SplitThresholds`。

| 情况 | 谁切、切在哪 |
|---|---|
| 估计准 | driver 跑到抬高后的 `t1`（Stage 1 选的晚 L1）干净切 → 段长、韵律连贯 |
| 估计偏乐观、要溢出 | 阶梯**自动重新接管**：在选点前撞 L2/L3/force 提前切（边界脏一点，但**0 丢句**） |

> 修正记录：早期方案说 "Stage 1 插硬 interior END、driver 纯 obey" 是错的。正确形态是 "打包组 + 每组抬高 t1 的 threshold profile，driver 自己在晚 L1 自然 flush"——更接近现状 `_drive_group`，增量更小。

### 5.3 根治：KV 水位把阶梯接到正确单位

**后台调查结论**：

| 量 | 判定 | 证据 |
|---|---|---|
| 段内 KV 水位 | **AVAILABLE-AND-CHEAP** | `slot.past_len`(已用) + `kv_pool.max_seq_len`(=512) 每 step 在手（`engine_loop.py:1020,1013`）；`ResultType` 可加 `KV_WATERMARK`，`_send_result` 通道现成（`engine_loop.py:1434`） |
| 溢出截断点 | **AVAILABLE-BUT-NEEDS-PLUMBING** | overflow 在 `engine_loop.py:1036` 触发；但 `text_tokens` 报的是**喂入量**非渲染量，`text_idx`→原始 text-token 映射未维护 |

**核心洞察：水位吞掉了截断点这块硬骨头。** 有了段内水位即可在**溢出发生前主动切**；主动切后未渲染尾巴走**正常段边界结转路径**，而 backend 已有结转机制 `group.overflow_token_ids`（`engine_loop.py:430-437,395-403`）会把未消费 token 自动 prepend 到下一段。于是不再需要精确截断点。

**阶梯触发器改造**（保留 prosody 分层，换信号源）：

```
L1→L2→L3→force 阶梯保留，触发器从"文本计数 vs EMA 阈值"换成"真实 KV 水位跨档"：
  水位过 0.80 → driver 愿意在 L2 切
  水位过 0.90 → 愿意在 L3 切
  撞 max_seq  → 硬 EOS（尾巴结转，不丢）
```

`KV_WATERMARK` 事件按**阈值跨档**发（非每 step），映射到 driver 的 tier 升级。这样 prosody 分层一点不丢，但喂给它的是**真实占用**而非估计。EMA 从"安全命脉"降级为"只决定 Stage 1 乐观装箱目标多大"。

**水位不改变控制面，只升级信号质量**（呼应 §1.3）：前端依旧只能控文本 token，水位响应仍是"在下一个文本 token 边界（标点）flush"。水位把 **flush 时机判断**从 EMA 估计换成实测，但 Stage 1 装箱在 decode 前发生、无水位可查，**仍靠 EMA**。

**flush 阶段自身要预留 KV headroom**：第三阶段（pad EOS/NOP + drain 剩余音频）也消耗 step。若等到水位 0.95 才触发 flush，flush 本身可能在 pad 过程中溢出。故硬切水位档必须**为 flush 尾巴留余量**（如硬切阈值取 0.90 而非贴着 max_seq）。

### 5.4 两个岔路的最终定论

- **装箱基准**：→ **乐观装箱（EMA 均值 cap）+ 水位兜底**。水位是实时安全网，可激进装箱拿最大质量；偏乐观时水位在溢出前触发干净主动切。悲观 `force` 底降级为"水位响应太晚"的粗 backstop。
- **要不要上水位**：→ **上，作为根治**。它 cheap 且吞掉了截断点 plumbing。

---

## 6. EMA 与熔断分层（水位根治后大幅简化）

现状脆弱点：`overflow` 只是 bool（不报丢多少）；`ema_overflow_alpha=0.5` 一次离群段猛拽全局 EMA（可能压碎后续段）；`clamp[2,10]`（`spliter.py:605`）——真实 ratio>10 时 EMA 饱和、永远低估、反复丢、不收敛。

水位根治后，熔断从"必需"降为"备份"：

1. **段内水位跨档主动切**（根治，新增）——audio-step 维度实时阶梯。
2. **未渲染尾巴结转**：水位/溢出触发的 EOS 把 `pending` 尾巴推进 `group.overflow_token_ids`（复用现成路径）——**唯一仍需的 backend 小改**，不需要精确截断点。
3. **离群 vs 系统漂移区分**：单个病态段（长数字串、模型拉长的重复字）只对该段保守重切，系统性漂移才动全局 EMA，避免一次离群打碎整 session。
4. EMA `clamp` 放开 / 饱和检测：作为质量调参慢慢优化，**不再 load-bearing**，不阻塞安全。

---

## 7. 文本播放器（关注点 #2）

需求：输出音频的同时通知客户端"这段是哪个 token（含 pad token）合成的"。

现状：`ActionResult` 带 token，`SegmentAction.token_text` 一路带到输出（`interface.py:292,322`）；pad 是 driver 的 `PAD_TEXT_EOS/NOP`（`driver.py:116-126`）。统一后 **Stage 2（`_drive_events`）是 SegmentAction 唯一生产者**，"哪个 token 合成了这段音频"天然是它的输出元数据，结构上免费。

约束两条：
- (a) 重构里别把 `token_text` 在路由时丢掉 → 进 golden 测试断言；
- (b) **obey/装箱后段更长、flush 更少 → pad/边界事件随之变少**，播放器 UI 对"一段对应多个 L1 单元"要能展示。**需与播放器现状确认**它是否按"一段=一个 L1 单元"假设绘制。

---

## 8. De-risk 结论（已验证，重构可放心）

- **reorder 路径无关**：`AudioReorder`（`reorder.py`）坐标系 `(group_idx, local_idx)+group_final`；dispatcher（`dispatcher.py:74-76`）已把流式段映射为"每段自成一组（local=0,final=True）"。`_pending` 塌缩**不用改 reorder**，还能删掉 `group_idx=-1` 哨兵 + 三元重映射。Step 2 只需守约定：pre_split 边界→新 group（local 归零）；driver 自身 FLUSH→组内 local++；耗尽该组 token 的段→group_final=True（即两路径今天各自的行为）。
- **调度优先级双信号已覆盖**：`local_idx>0`（`dispatcher.py:176`）是 `CONTINUATION` 优先级信号（非排序）；紧接着 178-181 行有专给流式的并行信号 `segment_idx-1 in _flushing`。守住上述映射，reorder 与优先级**两者零改动**。
- **EMA 自适应通道已存在**：`update_ratio`（`spliter.py:583`）+ `_make_thresholds` 每组用最新 EMA 重算（`spliter.py:402`）。Stage 1 装箱与 driver 阶梯只要都走 `_make_thresholds` 即同源、不分叉。

---

## 9. 分步落地计划（行为保持、测试先行）

- **Step 0 冻结**：golden 测试钉住当前 `SegmentAction` 序列——(a) 招牌全局最优 case `你好吗？明天天气不错，有没有什么想吃的？`（建议进 `tools/repro/`）、(b) 纯 token 流、(c) `FULL_TEXT`、(d) 并发满背压、(e) **跨包 emoji**（顺便把 §3.1 的 bug 钉在案上）。
- **Step 1**：`pre_split` 改产内部表示（事件流 / 打包组 + threshold profile），`set_full_text`/`push_group_tokens` 改成消费它，**行为不变**。
- **Step 2（风险点）**：上 `_drive_events` 统一两套驱动循环；`_token_buffer`+`_presplit_groups` 塌缩为 `_pending`；删 `-1` 哨兵。golden 把关。
- **Step 3**：引入 auto = 每包 L1-only `pre_split` → `_drive_events`；`UNSPECIFIED=0` 语义化为 auto、作流式默认。
- **Step 4**：auto 作流式默认；`TOKEN/CLAUSE/LONG_SEGMENT` **保留为一等显式入口**（高级用户场景选择，不 deprecate）；`FULL_TEXT` 独立保留。
- **Step 5（algorithm + safety）**：层级装箱替换"见 L1 即切"；引入 `KV_WATERMARK` 事件 + 阶梯水位触发器 + 尾巴结转；Stage 0 filter chain 转有状态、修跨包 emoji。

---

## 10. 待确认 / 未决项

1. **backend `KV_WATERMARK` 事件**：新增 `ResultType.KV_WATERMARK`，在 `_process_step_output()`（`engine_loop.py:1108+`，`slot.past_len += 1` 后）按阈值跨档发；前端 `_consume_results`（`interface.py:263+`）加 elif 分支消费。
2. **`overflow_token_ids` 结转扩展**：水位/溢出触发的 EOS 把未渲染 `pending` 尾巴推进 `group.overflow_token_ids`，而非丢弃。
3. **driver FSM 改造**：阶梯触发器从 token-count 阈值改为水位跨档，具体 FSM 规则（`driver.py:226 _build_fsm`）怎么改——**下一轮讨论**。
4. **文本播放器**对"长段=多个 L1 单元"的展示假设确认。
5. **per-packet auto 的 `min_tokens_l1` 重标定**：原值为整文离线调的，per-packet 流式下要不要放松（影响首段延迟），上真实流量标定。
6. **目标并行度 ↔ KV pool slot 上界**的对齐公式。

---

## 11. 关键文件索引

| 区域 | 文件 |
|---|---|
| 输入入口 / 归一化 / 结果消费 | `engine/frontend/interface.py`（`push_text_input:193`, `_normalize_tts_text:51`, `_consume_results:263`, SEGMENT_END:330） |
| 切分编排 | `engine/frontend/spliter/spliter.py`（`pre_split:274`, `set_full_text:337`, `push_group_tokens:363`, `_drive_group:389`, `feed_tokens:451`, `update_ratio:583`） |
| Driver FSM / 阈值 | `engine/frontend/spliter/driver.py`（`compute_thresholds:51`, `_meets_split_threshold:160`, `_normal_overflow:177`, `_build_fsm:226`） |
| 音频重排 | `engine/frontend/spliter/reorder.py` |
| 派发 / 优先级 | `engine/frontend/dispatcher.py`（坐标映射:74, 优先级:167） |
| emoji 过滤 | `engine/text_normalization.py` |
| backend 解码 / KV | `engine/backend/engine_loop.py`（水位:1020, overflow:1036, SEGMENT_END:1321, `_send_result:1434`）, `engine/backend/kv_cache_pool.py:43,243` |
| 协议 | `proto/tts.proto:58`（InputMode/GroupPolicy） |
