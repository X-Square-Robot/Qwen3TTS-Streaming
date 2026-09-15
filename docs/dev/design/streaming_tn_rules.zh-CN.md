# Streaming TN 当前实现规则

> 本文只描述当前代码行为，不定义未来的 Soft Drain、候选稳定前缀或新的外部协议。
> 规则的唯一实现来源是 `engine/frontend/text_commitment/committer.py`；本文用于建立
> 代码、状态和测试之间的索引。

## 1. 先看职责边界

当前有三个容易混淆的层：

| 层 | 代码 | 负责什么 |
|---|---|---|
| 输入与 raw 坐标 | `engine/frontend/interface.py`、`CanonicalTextJournal` | 接收分包、保留原文和 raw/spoken 映射 |
| 增量 TN 主控制器 | `IncrementalTextCommitter` | 逐字符扫描、维护开放 span、判断闭合、调用 TN/领域规则、产生单调 `TextCommit` |
| TN 生命周期观察器 | `engine/frontend/tn/SpanFSM`、`SpanDriver` | 把 committer 的输入和 `CommitDecision` 投影为可观察的生命周期状态 |

`SpanFSM` **不拥有词法规则，也不决定何时调用 WeText**。它观察并记录 committer 的决定。实际
规则不能只看 `engine/frontend/tn/`；必须和 `text_commitment/committer.py` 一起看。

## 2. 主流程

```text
raw delta
  → IncrementalTextCommitter._append（逐字符增量扫描）
  → plain buffer / pending semantic span
  → 观察到闭合、final、metadata.closed、超时或容量上限
  → _close_pending
  → 语言路由 + 候选/领域规则 + WeText 或 fallback
  → TextCommit（raw 区间、spoken 文本、mapping、原因）
  → tokenizer / splitter
```

没有足够证据闭合时，当前 span 留在 `_pending`，不会因为一个 transport packet 结束就提交。

## 3. `SpanFSM` 生命周期

`CommitmentState` 当前为：

```text
SCAN → OPEN → READY → NORMALIZE → COMMIT
  └──────────────→ FALLBACK
SCAN/COMMIT/FALLBACK → DONE（finalize）
```

生产路径通常是 `SpanDriver.feed/poll → committer.feed/poll → DECISION 事件 → SpanFSM`。
`READY/NORMALIZE` 主要是给异步或可观测集成预留的显式事件；默认 committer 并不会先把每个 span
逐步驱动成这两个状态。README 中提到的 `WAIT` 不是当前 `CommitmentState` 枚举成员；等待在生产中
表现为 `OPEN + pending_raw`。

## 4. 当前识别的 span 类型

`SpanKind` 包括：

```text
PLAIN, NUMBER, ENGLISH_WORD, ORDINAL, MATH,
URL, EMAIL, IDENTIFIER, VERSION, PHONE, ID_CARD,
MARKDOWN, JSON, EMOJI, LITERAL
```

分类不是一个简单的“遇到开始符号、等结束符号”规则。分类优先级大致为：

1. 身份证、电话、email、URL、型号/标识符；
2. 货币、时间、日期、范围、单位、分数等数量；
3. 数学表达式和比较运算；
4. 小数、版本号、序数、普通数字；
5. 其余 Latin run 作为英文词。

其中 URL、email、数字、单位、型号和公式通常没有成对 delimiter，必须靠右边界、语法变化或最终
输入来确认闭合。

## 5. 什么会让 pending span 闭合

当前闭合来源包括：

- 词法边界：空白、标点、换行、括号边界等；
- 语义边界：百分号、单位、日期/时间/范围、数学表达式平衡等；
- 结构闭合：Markdown 配对标记、链接、代码围栏、JSON 括号/字符串状态；
- `TextInputMetadata.closed=True`；
- `feed(final=True)` 的最终 flush；
- semantic idle/max wait 到期后的 fallback；
- `max_pending_chars` 到限后的可审计 fallback。

空白并不总是闭合信号。例如 URL、带空格的公式、Markdown/JSON 和跨包单位都可能继续扩展。
单个数字也会暂时保留，以避免 `2`、`25` 或 keycap emoji 被错误拆开。

## 6. 闭合后的动作路由

`_close_pending` 不直接对所有文本调用同一个 TN 函数：

| span | 当前动作 |
|---|---|
| NUMBER/ORDINAL | 有明确语言时先取 WeText 候选并经过 `CommitPolicy`；没有语言证据时保守 literal，final/timeout 对合格数量可走 fallback |
| ID_CARD/PHONE | `DomainResolver` 的确定性数字/电话规则，失败后 fallback |
| MATH/IDENTIFIER/VERSION | 领域 fallback；不把型号或公式误送入普通数量语法 |
| MARKDOWN/JSON | `project_readable` 投影出可读内容并保留 raw/output mapping |
| URL/EMAIL | 清理 HTML entity 后走 TN；无结果时使用符号化 fallback 或 literal |
| 普通英文/中文 | 按解析出的 `zh/en` 路由到 closed-span WeText；不需要变化时 literal |

候选解析和 `CommitPolicy` 只决定闭合后的 COMMIT/FALLBACK；它们不负责发现 span 的结束位置。

## 7. 单调提交和超时

一旦产生 `TextCommit`，raw end 和 spoken 文本只能前进，不能回写。若一个 span 先因 timeout 或
容量上限被 fallback 提交，后续字符不能重新并回旧 span，而会形成新的 late extension，并产生
`text.span.late_extension` 事件。

这条 commit fence 是流式 TTS 的核心安全约束：宁可局部 literal/fallback，也不能已经发声后再
把 `1` 改成 `10`、把型号改成数量或把 Markdown 重新投影。

## 8. 复杂度判断

“特殊字段闭合后调用 TN”适用于一个有明确协议 delimiter 的字段，例如完整 JSON 字段或一类明确的
标记块。当前输入不是这种协议：数字、日期、URL、email、型号和公式都可能没有 delimiter，并且
可以跨任意 transport 包到达。因此必须同时处理：

- 分包不变性和 mutable tail；
- Unicode grapheme/emoji/keycap；
- 数字、单位、日期、范围、公式之间的歧义；
- Markdown/JSON 与普通文本的冲突；
- 中英语言证据和候选 margin；
- raw → spoken 映射、commit fence、late extension；
- backend 缺失、超时、final flush 和容量上限。

所以复杂性是有业务原因的，但当前实现的复杂度也确实集中得过高：`committer.py` 约 2400 行，
同时承担 lexer、结构状态、路由、投影、映射和提交策略。`tn/SpanFSM` 本身反而很薄。

## 9. 维护规则

后续修改应遵守：

1. 不在 `SpanFSM` 或 cursor 适配层复制词法/TN 规则；
2. 新 span 类型先补 `SpanKind`、闭合条件、路由和 fallback，再补实现；
3. 每个规则至少有 one-shot、逐字符分包、final、timeout/late-extension 测试；
4. `TextCommit` 的 raw 区间和 mapping 与 spoken 输出一起验收；
5. 修改闭合语义时同步更新本文件和
   `incremental_text_normalization_and_soft_drain.zh-CN.md` 的对应章节。

