# Streaming TN 产品语法与读法合同

> 本文面向产品、算法和前端共同评审。它描述用户看到的“什么文本属于什么字段、何时
> 认为字段结束、何时开始转换为口语”的合同。实现可以使用 FST、DFA 或等价组件，但
> 不得再把 Python 分支顺序当作未公开的产品规则。

## 1. 产品目标

输入是一条持续增长的原始文本流。系统把它解析成有顺序的语法节点：

```text
Document
├── Prose*                 普通可直接朗读的文本
├── SpecialSpan            需要专门读法的文本字段
│   ├── Quantity            数字、百分比、金额、单位
│   ├── Calendar             日期、时间、范围
│   ├── Identifier           型号、版本号、订单号、证件号
│   ├── Contact              URL、Email、电话号码
│   ├── Formula              数学表达式
│   ├── Structured           Markdown、JSON、代码块
│   └── Grapheme             emoji、组合字符、keycap
└── Punctuation*            保留为韵律/标点，不参与字段读法
```

产品只关心四个结果：

1. `Prose` 可以立即进入 TTS；
2. `SpecialSpan` 在闭合前必须暂存；
3. 闭合后只能产生一个确定的 spoken form 或一个明确标记的 fallback；
4. 已发布 spoken form 永远不能被后续文本改写。

## 2. 产品语法树

以下是语义优先级，不是 Python `if` 的执行顺序。更具体的节点必须先于更一般的节点匹配。

```ebnf
Document        ::= Piece* EOF
Piece           ::= Prose | SpecialSpan | Punctuation | DroppedGrapheme

SpecialSpan     ::= Structured
                  | Contact
                  | Identifier
                  | Formula
                  | Calendar
                  | Quantity
                  | Grapheme

Structured     ::= JsonValue | MarkdownConstruct | CodeFence

Contact        ::= Url | Email | Phone
Url            ::= Scheme Host Path? Query? Fragment?
Email          ::= LocalPart "@" Domain
Phone          ::= CountryPrefix? PhoneGroup (Separator PhoneGroup)*

Identifier     ::= ModelCode | Version | OrderId | IdCard
ModelCode      ::= Latin+ ("-" | "_")? Digit+
Version        ::= ("v")? Digit+ ("." | "_" | "-") Digit+ (Digit | "." | "_")*
OrderId        ::= OrderContext Number
IdCard         ::= Digit{17} (Digit | "X" | "x")

Formula        ::= Operand Operator Operand (Operator Operand)*
Operand        ::= Number | Identifier | ParenthesizedFormula
Operator       ::= "+" | "-" | "*" | "/" | "=" | "^"
                  | "×" | "÷" | "≤" | "≥" | "≠"

Calendar       ::= Date | Time | Range
Date           ::= Year DateSep Month (DateSep Day)?
                  | Year "年" Month "月" Day? "日"?
Time           ::= Hour ":" Minute (":" Second)? Meridiem?
Range          ::= Number RangeSep Number

Quantity       ::= Percent | Currency | Measure | Decimal | Integer | Ordinal | Fraction
Percent        ::= Number "%"
Currency       ::= CurrencySymbol Number | Number CurrencySuffix
Measure        ::= Number Unit
Decimal        ::= Sign? Digit+ "." Digit+
Integer        ::= Sign? Digit+
Ordinal        ::= Digit+ ("st" | "nd" | "rd" | "th")
Fraction       ::= Digit* VulgarFraction

Grapheme       ::= Emoji | Keycap | ZWJSequence | CombiningSequence

Prose          ::= Character+
Punctuation    ::= Terminator+ | BoundaryPunctuation
```

语法中的 `Number` 是中间节点，不能在识别阶段直接决定“数量读法”。例如：

```text
25       → Integer → Quantity → 二十五
25%      → Percent → Quantity → 百分之二十五
2026-07  → DatePrefix，继续等待
3*2=6    → Formula → 三乘二等于六
B-0109   → ModelCode → B 负/连字符 0109 的产品读法
```

## 3. 前缀状态与闭合规则

每个 `SpecialSpan` 都有一个前缀状态：

```text
ABSENT → PREFIX → COMPLETE → SPOKEN
                    └──────→ FALLBACK
```

### 3.1 状态定义

| 状态 | 产品含义 | TTS 行为 |
|---|---|---|
| `ABSENT` | 当前字符不属于特殊字段 | 普通文本路径 |
| `PREFIX` | 当前前缀仍可能变成长字段 | 暂存，不调用最终 TN |
| `COMPLETE` | 语法已确认字段边界 | 调用该字段对应的 TN/FST |
| `SPOKEN` | 已产生稳定 spoken form | 只能追加，不能回改 |
| `FALLBACK` | 规则无法确定或资源超限 | 按产品 fallback 朗读并建立 fence |

### 3.2 各类字段的闭合条件

| 类别 | 进入 `PREFIX` | 进入 `COMPLETE` | 不能做的事 |
|---|---|---|---|
| 数字/小数 | 首个数字、符号或小数点 | 后继字符不再匹配数字语法，或遇明确边界 | 不能在 `2` 时假设最终是“二” |
| 百分比/金额/单位 | 数字后出现 `%`、货币符号或单位前缀 | 后缀完整且遇边界 | 不能把 `%` 拆成普通标点 |
| 日期/时间/范围 | 日期/时间前缀、范围分隔符 | grammar 完整且右侧闭合 | 不能把 `2026-07` 当减法 |
| 型号/版本/订单号 | Latin+数字或上下文触发 | 右边界出现且上下文无继续扩展证据 | 不能按数量读法强制转换 |
| URL/Email/电话 | scheme、`@`、电话前缀或数字组 | 空白、右括号、完整 grammar 或 final | 不能泄漏原始 markup/entity |
| 公式 | 运算符、等号、括号或比较符 | 操作数完整、括号平衡且遇边界 | 不能让 `3*` 立即提交 |
| Markdown/JSON | `[`、`{`、`` ` ``、配对 marker | 配对完成、JSON depth 回零或 final | 不能删除未确认的 `_`、`*` |
| emoji/grapheme | base code point | Unicode grapheme 完整 | 不能把 keycap 的数字单独朗读 |

### 3.3 transport 分包规则

transport packet 不是产品边界：

```text
"25"       = "2" + "5"
"10.5%"    = "10" + ".5" + "%"
"https://"  = "http" + "s://"
"3*2=6"     = 任意分包组合
```

在 `PREFIX` 状态，任何分包都只能更新候选前缀，不能触发 spoken commit。只有 grammar
闭合、显式字段结束或最终输入才能进入 `COMPLETE`。

## 4. 闭合后的产品动作

```text
COMPLETE
  → 选择语言
  → 选择字段读法 grammar/FST
  → 生成 spoken candidate
  → 校验 raw span 覆盖和 mapping
  → SPOKEN 或 FALLBACK
```

路由规则：

| 字段 | 首选组件 | 失败行为 |
|---|---|---|
| 数字、日期、时间、百分比、金额、单位 | WeText/FST 对应 grammar | 明确的 cardinal/literal fallback |
| 型号、版本、订单号、证件号 | 产品 identifier FST | 按字符/数字序列 literal fallback |
| URL、Email、电话 | contact FST | 符号化 literal fallback |
| 公式 | math FST | 操作符和操作数的确定性 fallback |
| Markdown、JSON、代码块 | 结构解析器 + readable projection | 保留可读文本，删除已确认格式标记 |
| 普通 prose | 原文 | 不进入特殊字段 TN |

## 5. FST 应该负责什么

### 5.1 适合 FST 的部分

下列问题适合用确定性 FST/DFA 表达：

- 数字、日期、时间、百分比、金额、单位的词法识别；
- URL、Email、电话和型号的合法字符序列；
- 数字/符号到 spoken form 的有限状态转导；
- 数学操作符和括号的有限上下文读法；
- Markdown 的有限 delimiter 识别。

### 5.2 不能只靠一个 FST 的部分

整个产品流程不是一个单独的静态 FST：

- 输入是无限增长的流，需要保存 `PREFIX`；
- commit fence 和 timeout 是会话状态，不是 TN grammar；
- raw/spoken mapping 和已发布 high-water 是外部坐标状态；
- JSON 任意嵌套严格来说需要栈/深度状态，除非设定最大深度；
- 候选等待、资源上限、fallback 和 late extension 是产品策略；
- 中英文语言选择需要会话/上下文证据。

正确的产品分层应是：

```text
Streaming Product Automaton
├── Unicode/Grapheme assembler
├── Lexical DFA/FST：raw → typed syntax node
├── TN FST：typed syntax node → spoken candidate
├── Commit policy：COMPLETE/SPOKEN/FALLBACK
└── Journal/fence：raw ↔ spoken 坐标和单调发布
```

## 6. 当前实现与产品语法的偏差

当前 `IncrementalTextCommitter` 把以上五层混在一个约 2400 行模块中：

1. `_classify` 的正则优先级同时承担产品语法优先级；
2. `_append` 同时承担 lexer、Markdown/JSON 状态、Unicode 特例和闭合动作；
3. `_close_pending` 同时承担语言路由、candidate policy、领域 fallback、projection 和 mapping；
4. `SpanFSM` 没有消费语法节点，只观察 committer 的最终 `CommitDecision`；
5. 规则的真实行为来自分支顺序，没有一棵代码外可审查的 syntax tree；
6. `metadata.read_as`、`content_kind` 尚未映射到产品语法；
7. 当前测试主要验证示例输出，没有验证完整 grammar 的互斥性、优先级和闭合完备性。

因此，当前实现不是“一个 FST 加一个小 FSM”，而是“一个大型隐式 lexer/committer，外面
再包一个观察型 FSM”。这正是需要重构的架构信号。

## 7. 产品验收标准

每个语法节点必须有以下证据：

```text
语法定义
→ 合法前缀集合
→ 闭合条件
→ 与兄弟节点的优先级
→ spoken grammar/FST
→ fallback
→ raw/spoken mapping
→ 任意分包等价测试
→ timeout/final/late-extension 测试
```

最小验收性质：

1. **互斥性**：同一 closed span 只能归入一个最终产品类别；
2. **优先级稳定**：更具体类别不会被通用数字/英文规则抢走；
3. **前缀安全**：`PREFIX` 不产生不可撤回 spoken 输出；
4. **分包不变**：无 timeout 时，任意合法分包得到同一最终 spoken form；
5. **闭合完备**：final 到来后所有 raw 字符都有 spoken、drop 或 fallback 结果；
6. **单调发布**：`raw_end`、commit id 和 spoken output 只增不减。

## 8. 重构顺序

1. 先冻结本产品语法树和字段优先级；
2. 为每个字段定义 typed syntax node，而不是直接返回 `SpanKind`；
3. 把可正则表达的字段迁移为独立 DFA/FST；
4. 把 WeText 调用限制在 typed node 已进入 `COMPLETE` 之后；
5. 保留一个很小的 streaming commit controller，只处理 `PREFIX/COMPLETE/SPOKEN/FALLBACK`；
6. 用语法树生成测试矩阵，再逐步删除 `committer.py` 中的特殊分支；
7. 最后再决定哪些结构（如 JSON 深度和 Markdown）需要栈式 parser，而不是强行塞进 FST。

## 9. 目标产品链路：稳定性 FST → TN → slot

产品链路应明确拆成三步：

```text
流式文本
  → 稳定性词法 FST / 词法树
  → 已闭合的 typed slot
  → TN/WFST 生成 spoken form
  → TTS slot
```

稳定性 FST 不负责把数字读成中文；它只回答：

```text
当前前缀属于哪个业务字段？
它还可能继续增长吗？
现在是否已经可以产出一个不可回改的 slot？
```

每次输入都返回结构化观察结果：

```python
SlotObservation(
    status="unstable" | "stable" | "invalid",
    kind="measure" | "date" | "identifier" | ...,
    raw_start=...,
    raw_end=...,
    raw_text=...,
    accepting=True | False,
    extendable=True | False,
    closure="unit_complete" | "right_boundary" | "final" | ...,
    grammar_id="business.energy.v1",
)
```

### 9.1 `3千瓦时` 示例

```text
输入       FST 状态                  产品结果
3          quantity.integer          unstable
3千        measure.number_unit_prefix unstable
3千瓦      measure.unit_prefix        unstable
3千瓦时    measure.complete           stable → 产生 measure slot
```

随后才调用 TN：

```text
raw slot: 3千瓦时
kind:     measure
grammar:  business.energy.v1
TN:       生成“三千瓦时”或业务指定 spoken form
TTS slot: spoken="三千瓦时", raw=[0,4)
```

这里的 `stable` 不是“FST 当前状态是 accepting”这么简单。产品合同要求：

1. 当前路径是 accepting；
2. 当前路径满足该业务 grammar 的闭合条件；
3. 同一个 slot 没有仍然开放的后缀分支，或业务明确规定该单位在此终止；
4. raw span 映射已经固定；
5. 产生 slot 后，后续字符只能从新的 slot 或普通文本开始。

例如业务如果同时支持 `3千瓦时/日`，则 `3千瓦时` 只能是 `accepting`，不能是 `stable`；
直到收到 `/日` 或明确右边界。也就是说，**accepting、extendable、stable 必须是三个独立属性**。

## 10. 业务自定义 FST

不同业务可以上传自己的字段 grammar，但上传物必须是受约束的 grammar 包，而不是任意 Python
代码或任意回调。建议合同如下：

```text
BusinessGrammarPackage
├── manifest.json
│   ├── grammar_id / version
│   ├── language / domain
│   ├── priority
│   ├── input_alphabet
│   ├── output_policy
│   ├── max_span_chars
│   ├── closure_policy
│   └── compatibility_version
├── lexer.fst / lexer.far
├── normalizer.fst / tn.far
├── symbol_table.txt
├── examples.jsonl
└── golden.jsonl
```

业务 grammar 至少要声明：

| 字段 | 含义 |
|---|---|
| `grammar_id/version` | 可审计、可回滚、可绑定会话 |
| `priority` | 与通用 grammar 冲突时的优先级 |
| `accepting_states` | 哪些状态构成合法字段 |
| `extendable_states` | 哪些 accepting 状态仍允许继续扩展 |
| `closure_policy` | `right_boundary`、`explicit_marker`、`unit_complete`、`final_only` 等 |
| `output_policy` | literal、TN、业务 spoken lexicon 或组合转导 |
| `max_span_chars` | 防止 grammar 无界吞入整段文本 |
| `golden` | 分包、前缀、冲突和 fallback 的验收样例 |

业务 grammar 的匹配顺序建议为：

```text
业务 grammar（按 priority）
  > 产品通用 grammar
  > 普通 prose
```

同一优先级内使用最长匹配；若多个 grammar 都能接受，必须返回冲突诊断，不能依赖 Python
字典顺序或 import 顺序。

## 11. 开源实现调研结论

当前仓库只有 `wetext==0.1.7`。它通过公开 `Normalizer`/`StreamNormalizer` 接口提供
规范化和候选结果，但仓库没有 Pynini、OpenFst 或业务 grammar 编译链；WeText 的内部实现也
不能作为本仓库可上传 grammar 的稳定接口。

候选方案：

| 方案 | 适合做什么 | 局限 |
|---|---|---|
| **OpenFst** | 运行时 FST/WFST 表示、组合、确定化、最短路和权重 | C++ 库；不提供业务 grammar DSL 和上传安全边界 |
| **Pynini + OpenFst** | Python 中编译正则、rewrite rule 和 WFST；适合离线构建业务包 | 原生依赖较重；不应在请求路径动态编译不可信输入 |
| **OpenGrm Thrax + OpenFst** | 面向 grammar 作者的 DSL，编译为 OpenFst/FAR | 需要独立编译工具链，服务端仍需定义 manifest/校验 |
| **WeTextProcessing** | 通用中英文 TN/ITN 作为基础 grammar | 当前公开 API 不暴露 lexical accepting/extendable/closure 状态 |

OpenFst 官方定位是构建、组合、优化和搜索 WFST；Pynini 是基于 OpenFst 的 Python grammar
编译层；Thrax 将正则和上下文 rewrite 编译为 OpenFst 格式。这三者适合做“grammar 编译和
运行时”，但都不会自动定义本项目的 slot 稳定性、超时、commit fence 或业务优先级。

因此初步建议是：

```text
离线：Pynini 或 Thrax 编译业务 lexer/normalizer grammar
产物：OpenFst-compatible FST/FAR + manifest + golden set
在线：轻量 FST runner 执行 prefix/accepting/extendable 查询
控制：独立 streaming slot controller 处理 buffer、timeout、fence、mapping
TN：业务 TN FST 优先；通用数字/日期回退到 WeText
```

不能把“把任意业务 FST 上传到服务端”实现成直接 `pickle`、Python 回调或请求时编译；必须
做格式校验、状态数/弧数/最大 span 限制、符号表校验、确定性/可达性检查、版本绑定和资源
隔离。
