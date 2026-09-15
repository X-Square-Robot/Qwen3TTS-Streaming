# Streaming TN FSM 合同（当前实现）

> 这是一份运行时合同，不是目标架构。它描述当前代码真正会触发的事件、状态、动作和输出，
> 并把“代码没有定义”的部分明确列出来。

## 1. 当前不是一个 FSM，而是两个状态机

### 1.1 控制面 FSM：`SpanFSM`

位置：`engine/frontend/tn/fsm.py`。

它只记录生命周期，不扫描字符，也不调用 TN。生产入口是：

```text
SpanDriver.feed/poll
  → IncrementalTextCommitter.feed/poll
  → CommitDecision
  → SpanEvent(DECISION)
  → SpanFSM
```

### 1.2 数据面隐式 FSM：`IncrementalTextCommitter`

位置：`engine/frontend/text_commitment/committer.py`。

它没有一个显式的 state enum，而是由 `_pending`、JSON/Markdown 状态旗标、候选缓冲和
`_late_extension` 等字段共同表示状态。它才是真正决定：

- 哪些字符属于同一个 span；
- span 什么时候闭合；
- 什么时候调用 WeText/领域规则；
- 什么时候等待、提交或 fallback。

因此不能只画 `SpanFSM` 的状态图来解释当前 TN。

## 2. 控制面状态转移表

状态来自 `CommitmentState`：

```text
SCAN, OPEN, READY, NORMALIZE, COMMIT, FALLBACK, DONE
```

| 当前状态 | 事件 | guard | 下一状态 | 运行时来源 |
|---|---|---|---|---|
| 任意 | `RESET` | 事件类型匹配 | `SCAN` | 只有显式调用 `reset()`/事件时 |
| 任意 | `DECISION` | `event.decision.state == 目标状态` | decision 中的状态 | `SpanDriver.feed/poll` 在 committer 返回后发出 |
| `SCAN` | `INPUT` | `event.text` 非空 | `OPEN` | `SpanDriver.feed` 先发输入事件 |
| `OPEN` | `INPUT` | 无额外条件 | `OPEN` | 追加新的 transport delta |
| `OPEN` | `SPAN_READY` | 无 | `READY` | 默认生产路径不会自动发出 |
| `READY` | `NORMALIZE_START` | 无 | `NORMALIZE` | 默认生产路径不会自动发出 |
| `NORMALIZE` | `NORMALIZE_FINISHED` | 无 | `COMMIT` | 默认生产路径不会自动发出 |
| `COMMIT` | `INPUT` | `event.text` 非空 | `OPEN` | 显式生命周期使用 |
| `OPEN` | `TIMEOUT` | 无 | `FALLBACK` | 默认 driver 不直接发此事件，而是接收 `DECISION(FALLBACK)` |
| `OPEN` | `FALLBACK` | 无 | `FALLBACK` | 同上 |
| `FALLBACK` | `INPUT` | `event.text` 非空 | `OPEN` | timeout 后来的 late extension |
| `SCAN`/`COMMIT`/`FALLBACK` | `FINALIZE` | 无 | `DONE` | 显式 finalize |

`DECISION` 行是权威行：只要 committer 返回了一个状态，控制面就直接投影到该状态，
不会再经过 `READY → NORMALIZE → COMMIT` 的中间事件。

## 3. 数据面实际案例

下面的 `feed` 指 `IncrementalTextCommitter.feed`；`poll` 指超时检查。

| Case | 触发条件 | pending | TN/动作 | `CommitDecision` | 输出 |
|---|---|---|---|---|---|
| A. 普通稳定文本 | 字符不属于开放 semantic span，遇到普通边界 | 通常为空 | literal commit | `COMMIT` | `TextCommit` 立即进入 tokenizer |
| B. 开放 semantic span | 数字、URL、email、型号、公式、Markdown/JSON 尚可继续扩展 | 非空 | 不调用最终 TN | `OPEN` | 无 tokenizer 输出 |
| C. 词法闭合 | 观察到空格/标点/配对标记/平衡括号等类别边界 | 清空 pending | `_close_pending(reason="boundary")` | `COMMIT` 或 `FALLBACK` | 生成一个或多个 `TextCommit` |
| D. metadata 强制闭合 | `metadata.closed=True` 且 pending 非空 | 清空 pending | `_close_pending(reason="metadata_closed")` | `COMMIT`/`FALLBACK` | 生成 commit；该路径应由合同测试保护 |
| E. 最终 flush | `feed(..., final=True)` | 清空 pending | `_flush_pending(reason="final")` | `DONE` | 所有剩余 raw 必须 commit/drop/fallback |
| F. semantic timeout | `poll(now)` 到达 `next_deadline` | 清空 pending | `_fallback_pending(reason="timeout")` | `FALLBACK` | 产生 `text.fallback`，建立 fence |
| G. timeout 后续字符 | timeout 后再次 `feed(non_empty)` | 新 span | 首个新 span `force_literal` | 通常 `COMMIT`/`FALLBACK` | 额外产生 `text.span.late_extension` |
| H. pending 未超时 | `poll` 早于 deadline，或没有 pending | 保持原状 | 不做 TN | `OPEN` 或 `SCAN` | 无新输出 |
| I. 长结构 span | `len(pending) >= max_pending_chars` | 强制清空 | 关闭并强制标记 fallback | `FALLBACK` | `text.fallback` |
| J. TN disabled | `config.enabled=False` | 不建立 semantic pending | 所有输入直接 literal | `SCAN` 或 `DONE` | 原文直接提交 |
| K. final 后输入 | `_finalized=True` 后再次 feed | 不允许重开 | 返回 terminal decision | `DONE` | `text.input.after_final`（有输入时） |

## 4. 闭合规则不是统一 delimiter

| 类型 | 典型闭合条件 | 仍可能跨包扩展 |
|---|---|---|
| NUMBER/ORDINAL | 数字后遇到边界、百分号、单位、日期/范围完整 | `2`→`25`、`21`→`21st` |
| URL/EMAIL | 空白、右括号、结束或 URL grammar 足够完整 | path/query/entity 仍可继续 |
| MATH | 运算符/右操作数出现，括号平衡后遇边界 | `3*`→`3*2=6` |
| MARKDOWN | 配对 marker、链接、代码围栏或 final | `[` 可能是链接或 JSON 数组 |
| JSON | quote/escape/depth 闭合，或 final | 字符串和嵌套数组/对象 |
| ENGLISH_WORD | Latin run 结束；某些 `!/[/*_` 需要回看 | `foo`→`foo20%` 等嵌合形式 |

一个 transport packet 结束**不是**闭合事件。

## 5. 闭合后的路由

`_close_pending` 根据 `SpanKind` 路由：

| 类型 | 路由 |
|---|---|
| NUMBER/ORDINAL | 解析语言 → 候选 resolver → `CommitPolicy` → closed-span WeText 或 fallback |
| ID_CARD/PHONE | `DomainResolver` 确定性规则 → fallback |
| MATH/IDENTIFIER/VERSION | 领域 fallback；避免按普通数量读取 |
| MARKDOWN/JSON | `project_readable`，删除结构标记并生成 mapping |
| URL/EMAIL | 清理 HTML entity → closed-span TN → 符号 fallback/literal |
| 其他 | closed-span TN；结果无变化则 literal |

## 6. 可观察事件与真正触发点

| 事件 | 真实触发位置 | 备注 |
|---|---|---|
| `text.fallback` | timeout、max pending、commit 为 fallback | 不是独立 FSM 输入事件；附在 `CommitDecision.events` 中 |
| `text.span.late_extension` | timeout 后再次有输入 | 旧 fence 永不回改 |
| `text.input.after_final` | final 后再次 feed 非空文本 | 返回 `DONE`，不修改 raw 坐标 |
| `metadata_closed` | `_close_pending` 的 closure reason | 当前没有专门事件名 |
| `WAIT` | 没有独立事件/枚举 | 当前以 `OPEN + pending_raw` 表示 |

## 7. 当前合同缺口和可疑点

这些不是“已经证明的业务 bug”，而是当前无法仅凭 FSM 合同判断、或没有回归保护的地方：

1. **没有单一 FSM**：控制面和数据面分别由两个实现表达，状态可能不同步。
2. **默认路径跳过中间态**：`SPAN_READY`、`NORMALIZE_START`、`NORMALIZE_FINISHED` 在默认 driver 中不会产生；`READY/NORMALIZE` 不是可观测的真实生产阶段。
3. **`COMMIT` 事件没有默认转移行**：默认路径依赖 `DECISION(state=COMMIT)`，直接发 `COMMIT` 事件不会推进状态。
4. **`TIMEOUT`/`FALLBACK` 事件也不是默认输入**：实际通过 `poll()` 返回的 `DECISION(FALLBACK)` 投影。
5. **README 与类型不一致**：README 提到 `WAIT`，但 `CommitmentState` 没有 `WAIT`。
6. **`metadata.closed=True` 缺少端到端合同测试**：应验证 closure reason、TN 调用一次、pending 清空和输出进入 tokenizer。
7. **`read_as`/`content_kind` 尚未形成行为规则**：类型存在，但当前 committer 主要消费 `language_hint` 和 `closed`。
8. **闭合矩阵没有覆盖所有组合**：例如 Markdown/JSON、数字/公式和 emoji/keycap 的交叉情况目前只能从字符循环推断。

## 8. 缺陷审查的最低测试矩阵

每个 span 类型至少应覆盖：

```text
一次性完整输入
逐字符分包
在每个字符位置切包
明确闭合
final 闭合
timeout fallback
timeout 后 late extension
metadata.closed=True
normalizer/backend 失败
raw/spoken mapping
```

在这份矩阵补齐前，不能仅凭“现有测试通过”判断闭合规则没有缺陷。

