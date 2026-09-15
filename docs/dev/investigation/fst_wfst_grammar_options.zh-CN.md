# Streaming TN 的 FST/WFST 方案调研

## 1. 调研结论

当前仓库没有自建 FST/WFST grammar。依赖只有 `wetext==0.1.7`；仓库通过公开的
`Normalizer` 和 `StreamNormalizer` API 调用它。`engine/frontend/tn/candidates.py` 中的
`WfstCandidateProvider` 只是未来扩展接口，不是已有 FST runtime。

因此，当前 `IncrementalTextCommitter` 不是 FST 的控制器，而是自行实现了大量 lexer、闭合
判断、业务 fallback 和提交策略。这正是规则难以审计的原因。

目标应改为：

```text
流式 raw text
  → Unicode/grapheme assembler
  → 业务/通用 lexical FST：prefix、accepting、extendable、kind
  → 稳定性判定：stable slot 或继续等待
  → TN/WFST：raw slot → spoken slot
  → TTS slot / raw-spoken mapping
```

## 2. 候选技术

### OpenFst

OpenFst 是 C++ 的 WFST 库，支持构建、组合、优化、确定化、最短路径和权重；它本身是
运行时和算法库，不是业务 grammar DSL。适合作为统一的二进制 FST/FAR 格式和底层 runner。

### Pynini

Pynini 是基于 OpenFst 的 Python grammar 编译层，支持字符串正则、FST 组合和上下文相关
rewrite rule。它适合离线构建业务 grammar，业务团队可以用 Python/grammar 代码生成
`lexer.fst` 和 `tn.fst`，但不应在请求路径编译不可信上传内容。

### OpenGrm Thrax

Thrax 是面向 grammar 的 DSL/compiler，可将正则和上下文 rewrite 编译为 OpenFst 格式的
加权转导器和 FAR archive。它比直接上传 Python 更适合作为业务规则发布格式，但仍需要
服务端定义 manifest、资源上限、版本和安全校验。

### WeTextProcessing

WeText 适合作为通用中文/英文 TN 基础 grammar。当前公开接口可以返回候选、cost、token
type 和 mapping，但没有向本仓库暴露 lexical `accepting`、`extendable` 或 closure
predicate。因此它不能单独承担 `3 → unstable`、`3千瓦时 → stable` 这类流式稳定性判断。

## 3. 推荐架构

```text
                         ┌────────────────────┐
raw stream ─────────────▶│ Streaming assembler │
                         └─────────┬──────────┘
                                   │ code points/graphemes
                         ┌─────────▼──────────┐
                         │ Lexical FST router  │
                         │ business > common  │
                         └─────────┬──────────┘
                                   │ SlotObservation
                         ┌─────────▼──────────┐
                         │ Stability controller│
                         │ prefix/accept/close │
                         │ timeout/fence/map  │
                         └─────────┬──────────┘
                                   │ closed raw slot
                         ┌─────────▼──────────┐
                         │ TN WFST dispatcher  │
                         │ business > WeText  │
                         └─────────┬──────────┘
                                   │ spoken slot
                         ┌─────────▼──────────┐
                         │ TTS slot + journal  │
                         └────────────────────┘
```

FST 的 `accepting` 不等于产品 `stable`。在线结果至少需要：

```text
kind
raw span
accepting
extendable
closure reason
grammar id/version
candidate/fallback policy
```

例：

```text
3       accepting=false, extendable=true  → unstable
3千     accepting=false, extendable=true  → unstable
3千瓦   accepting=false, extendable=true  → unstable
3千瓦时  accepting=true,  extendable=false → stable slot
```

如果业务还支持 `3千瓦时/日`，则 `3千瓦时` 仍然是 accepting 但 extendable，不能提前 stable。

## 4. 业务 grammar 包

建议业务上传的不是任意代码，而是可验证的 grammar 包：

```text
BusinessGrammarPackage/
├── manifest.json
├── lexer.fst 或 lexer.far
├── tn.fst 或 tn.far
├── symbol_table.txt
├── examples.jsonl
└── golden.jsonl
```

`manifest.json` 至少包含：

```json
{
  "grammar_id": "business.energy",
  "version": "1.0.0",
  "language": "zh",
  "priority": 100,
  "compatibility_version": 1,
  "max_span_chars": 64,
  "closure_policy": "unit_complete",
  "accepting_states": ["measure_complete"],
  "extendable_states": ["number_prefix", "unit_prefix"],
  "output_policy": "tn_wfst"
}
```

服务端验收 grammar 包时必须检查：

- FST/FAR 格式和 symbol table；
- 起始态、终态、可达性和空路径；
- 状态数、弧数、权重和最大 span 限制；
- 确定性或明确的加权冲突策略；
- grammar priority 和版本兼容性；
- golden 样例的任意分包、稳定性、mapping、fallback；
- 不允许 pickle、Python callback 或请求线程动态编译。

## 5. 与当前代码的迁移关系

| 当前代码 | 目标归属 |
|---|---|
| `_classify` 的正则集合 | 通用 lexical FST grammar |
| `_append` 中的数字/单位/URL/公式分支 | 各自独立的 lexer FST；保留一个 streaming runner |
| `_close_pending` 的 TN 路由 | TN dispatcher + grammar manifest |
| `DomainResolver` | 业务 TN/FST adapter |
| `_late_extension`、deadline、commit fence | Stability controller |
| `TextCommit`、journal、mapping | slot/journal contract |
| WeTextAdapter | 默认通用 TN backend |

迁移顺序：

1. 先定义 `SlotObservation` 和 `ClosedSlot` 合同；
2. 用现有规则生成通用 grammar 的 golden 集；
3. 选 Pynini/Thrax 之一构建离线 grammar，OpenFst 作为运行格式；
4. 先接入只读 shadow runner，对比当前 committer；
5. 证明稳定性、分包不变和 mapping 后，再替换生产闭合逻辑；
6. 最后拆除 `committer.py` 中对应的正则/分支。

## 6. 选型建议

第一阶段建议：**Pynini + OpenFst/FAR + 自有稳定性 controller**。

原因：

- Pynini 便于用 Python 编写和测试业务 grammar；
- OpenFst 提供成熟的 WFST 运行格式和算法；
- FAR 可以把一组命名 grammar 一起发布；
- 稳定性、timeout、commit fence、raw mapping 不被错误塞入 TN FST；
- WeText 继续作为通用 fallback，而不是被强行改造成业务 grammar registry。

Thrax 可以作为后续面向非 Python grammar 作者的发布入口；是否直接采用它，应在验证构建
速度、部署复杂度和业务团队维护能力后决定。

## 7. 参考资料

- [OpenFst 官方说明](https://openfst.org/twiki/bin/view/FST/)
- [OpenFst Quick Tour](https://github.com/google-research/openfst/blob/main/docs/quick_tour.md)
- [Pynini 官方文档](https://github.com/google-research/opengrm/blob/main/opengrm/pynini/docs/index.md)
- [OpenGrm Thrax Quick Tour](https://www.opengrm.org/twiki/bin/view/GRM/ThraxQuickTour)
- [WeTextProcessing 官方仓库](https://github.com/wenet-e2e/WeTextProcessing)

## 8. 默认 grammar 与 session patch

引擎不应把业务规则写进 engine code。建议引擎只提供 grammar runtime 和 snapshot 合同：

```text
Engine
├── DefaultGrammarSnapshot（进程级、不可变、预加载）
├── GrammarRegistry（版本、hash、能力、资源配额）
└── GrammarRuntime
      └── SessionGrammarSnapshot
            = default snapshot + ordered session patch
```

### 8.1 Session patch 的语义

patch 不是任意 Python 代码，也不是直接修改共享 FST 的 arc。它是一个可验证、可回滚的
grammar overlay：

```json
{
  "base_snapshot": "default-zh-v3@sha256:...",
  "patch_id": "energy-v1",
  "patch_version": "1.2.0",
  "operations": [
    {"op": "add", "rule_id": "energy.measure.kwh", "module": "..."},
    {"op": "replace", "rule_id": "common.number.percent", "module": "..."},
    {"op": "remove", "rule_id": "common.model_code.v1"}
  ]
}
```

推荐以 **grammar module/rule_id** 为 patch 单位，而不是以底层 FST state/arc 为 patch 单位。
底层 FST 编译器可以把最终模块集合重新组合、determinize、minimize 并生成新 snapshot；
session runtime 只加载已经验证完成的产物。

### 8.2 覆盖优先级

```text
session patch（高优先级）
  > tenant/domain patch
  > product default grammar
  > generic prose fallback
```

同一优先级内必须有稳定的 `priority + rule_id` 排序。冲突不能依赖注册顺序、Python import
顺序或哈希表迭代顺序。

一个 patch 可以替换三种不同能力，必须分别声明：

| 能力 | 作用 |
|---|---|
| `recognizer` | raw prefix → kind/accepting/extendable/closure |
| `normalizer` | closed raw slot → spoken candidate/mapping |
| `policy` | candidate priority、fallback 和 closure policy |

只替换 `recognizer` 而没有提供 `normalizer` 时，必须显式声明复用哪个已存在的 normalizer；
不能隐式回退到“当前 Python 分支”。

### 8.3 Session 绑定和热切换

默认规则或 patch 更新时：

1. 新 session 使用最新的 `DefaultGrammarSnapshot`；
2. 已运行 session 默认继续使用启动时绑定的 `SessionGrammarSnapshot`；
3. 如果产品确实需要中途切换，必须在明确的 slot/segment 边界切换；
4. 切换前必须 flush 或冻结旧 snapshot 的 pending span，不能让同一个 span 跨 grammar 版本；
5. 每个 `TextCommit` 记录 `grammar_id`、版本和 snapshot hash。

这样可以避免一个 session 在 `3千瓦` 到 `3千瓦时` 之间被换了 grammar，导致同一个 raw span
前后使用不同闭合语义。

### 8.4 Engine 与 grammar 的最小接口

```python
class GrammarRuntime(Protocol):
    def observe(
        self,
        snapshot: GrammarSnapshot,
        raw_delta: str,
        *,
        state: PrefixState,
    ) -> SlotObservation: ...

    def normalize(
        self,
        snapshot: GrammarSnapshot,
        slot: ClosedSlot,
    ) -> SpokenCandidateSet: ...
```

engine 只消费 `SlotObservation`、`ClosedSlot` 和 `SpokenCandidateSet`，不读取 grammar 内部
状态，不理解 `千瓦时`、订单号或业务字段名。业务规则由 grammar registry、编译器和测试包负责。

### 8.5 发布与安全边界

grammar snapshot 发布前必须通过：

- manifest/schema 校验；
- FST/FAR 格式、symbol table 和兼容版本校验；
- 状态数、弧数、最大 span、CPU/内存预算；
- 确定性、可达性、空路径和冲突检查；
- golden set 的稳定性、任意分包、fallback、mapping 测试；
- artifact hash 签名和可回滚版本登记。

session 热加载指“绑定一个已验证的 snapshot”，不指“在请求线程编译或执行不可信代码”。
