# TensorRT-LLM Runtime 技术路线评审报告

> 日期：2026-06-22
> 状态：评审稿
> 结论级别：建议作为整体路线评审输入，而非直接实施说明

## 结论摘要

当前项目面临的核心问题不是单一的“要不要换 TRT-LLM”，而是三个约束同时存在：

1. `backbone` 可以使用 `bf16`
2. `cp` 对数值更敏感，需要 `fp32` 才能避免明显幻觉
3. 流式场景存在“等待外部文本再继续 decode”的业务语义，不能按标准 LLM 的 run-to-completion 生成循环处理

基于当前仓库实现和截至 2026-06-22 的官方文档，结论如下：

- **短期产品化主线**：继续保留当前自建 runtime 语义，优先解决单引擎混合精度问题，即 `backbone=bf16`、`cp=fp32`、`code2wav=bf16`。
- **中期演进主线**：如果需要真正获得 paged KV / packed decode / 原生 continuous batching 收益，优先评估 **TRT-LLM PyTorch backend + 自定义 step model + 自定义 scheduler/request readiness**。
- **路线判断上的关键补充**：`backbone+cp` 仍然是决定 `next_embed` 的核心闭环，但 `code2wav` 也携带自己的 fixed-window KV 与 conv/transconv recurrent states；如果它完全留在 TRT-LLM 外部，系统仍会残留第二套 batch/cache runtime。
- **不推荐作为主线**：直接采用 stock `Executor` / stock `generate()` 托管整条生成循环。它适合标准自回归请求，不适合当前项目的 `WAIT_TEXT` 语义。
- **高风险长期路线**：`TensorRT backend + backbone+cp 统一引擎 + C++/Executor 深度定制` 可行，但工程复杂度、维护成本和调试成本最高，不适合当前产品化阶段作为第一选择。

一句话判断：

**TRT-LLM 值得做，但应该作为“中期 AR runtime 重构路线”，而不是现在立刻替换现有全部调度语义的银弹。**

## 1. 背景与问题定义

当前引擎已经具备自己的会话管理、segment 级流式协议、prefill、decode batch、KV slot 池和 code2wav 状态管理。实际部署暴露了两个主要问题：

1. **KV cache 管理效率不足**
   - 当前实现仍然是“自建 continuous scheduling + padded execution”的折中方案
   - 不能利用 vLLM / TRT-LLM 一类 runtime 的 paged attention / paged KV 能力

2. **`cp` 精度不足**
   - `cp` 在 `bf16` 下存在明显数值敏感性
   - 当前 fused TRT 引擎如果整体按 `bf16` 跑，容易诱发幻觉

与此同时，项目还有一个区别于标准 LLM 的约束：

3. **decode 可能需要等待外部文本**
   - 当流式文本尚未到齐时，当前 session 会暂停在“等待更多文本”的状态
   - 暂停期间需要保留 `last_codec_sum` 和相关 KV 状态
   - 文本到达后再恢复解码，而不是立刻进入 pad 或结束

这意味着项目真实需求不是“运行一个普通 Qwen3”，而是“运行一个带业务态暂停/恢复能力的自定义自回归系统”。

## 2. 当前实现现状

### 2.1 当前 runtime 的关键语义

仓内现有实现已经明确了以下行为：

- `engine/backend/engine_loop.py:963-980`
  - `next_embed` 在 loop 层以 `float32` 持有
  - 流式恢复时通过 `last_codec_sum + trailing_text_embed` 形成新的 `next_embed`

- `engine/backend/engine_loop.py:1179-1192`
  - 当本步拿到 `codec_sum`，但后续文本尚未到齐且 `input_complete=false` 时
  - session 进入“暂停等待文本”语义
  - 此时保留 `last_codec_sum`，不补 pad，不继续下一步 decode

- `engine/backend/executor.py:1078-1125`
  - decode step 由外层显式驱动，每次只跑一步
  - 每步都会将当前活跃 slot 的 KV gather 成 batch

- `engine/backend/executor.py:1246-1455`
  - 当前 fused executor 手工构造 `input_embeds`、`position_ids`、`attention_bias`
  - 手工组织 `talker_past_kv`、`c2w_past_kv`、conv/transconv states
  - 这本质上是自建 runtime 在模拟一个“多状态自回归执行器”

### 2.2 当前实现的两个结构性瓶颈

#### 瓶颈 A：padded execution

当前 decode batch 会把 `talker_past_kv` pad 到 batch 内最大 past 长度，再通过 `attention_bias` 屏蔽无效列。这是典型的 padded iteration-level batching。

这条路径能工作，但会带来：

- batch 内长短请求混跑时的冗余显存访问
- 更高的 KV gather/scatter 成本
- 难以达到 paged attention 路线的资源利用率

#### 瓶颈 B：loop 层和 engine 层的 dtype 不一致

当前代码已经意识到“内部计算 dtype”和“外部 I/O dtype”不是一回事。`engine/config.py:638-642` 已明确写出：

- `engine_dtype=fp32`
- `triton_io_float_dtype=bf16`

可以同时存在。

但当前 decode 发射时，`next_embed` 会被 cast 到 `self._config.dtype`：

- `engine/backend/executor.py:1087-1088`

这意味着即便 loop 层用 `float32` 持有 `next_embed`，只要引擎边界是 `bf16`，关键路径仍然会回落到 `bf16`。

## 3. 需求与约束

本次路线评审按以下目标判断：

### 3.1 必须满足

- 保留当前流式协议语义，尤其是 `WAIT_TEXT` / `暂停后恢复`
- 解决 `cp` 在 `bf16` 下的数值稳定性问题
- 支持 continuous batching 或等价的 iteration-level scheduling
- 能管理异构长度请求的 KV cache，不再长期依赖 pad 到最大长度
- 尽量保留当前 Python gateway / websocket / session 协议层

### 3.2 尽量满足

- `backbone` 利用 paged KV / packed decode
- `code2wav` 保持高吞吐的 TensorRT 路径
- 降低自研 KV 池、手工 gather/scatter 和状态拼装复杂度

### 3.3 当前阶段不宜牺牲

- 可调试性
- 业务语义正确性
- 流式稳定性

项目当前 `docs/roadmap.md` 明确表述的重点仍然是稳定性、部署和可维护性，因此路线选择应优先考虑产品化成本，而不是理论上的最高性能。

## 4. 外部技术事实

以下判断基于 TRT-LLM 官方文档与截至 2026-06-22 的 upstream 源码：

- `Executor API` 是高层异步执行接口，支持 in-flight batching / continuous batching 风格的请求执行
  - 参考：<https://nvidia.github.io/TensorRT-LLM/advanced/executor.html>

- `PyExecutor` 在后台连续运行，由 `Scheduler`、`KVCacheManager`、`ModelEngine` 等组件构成
  - `Scheduler` 负责判断哪些 active requests 在当前 step 是 ready for execution
  - 参考：<https://nvidia.github.io/TensorRT-LLM/architecture/overview.html>

- `KVCacheManager` 可以自定义并接入 `PyExecutor`
  - 参考：<https://nvidia.github.io/TensorRT-LLM/torch/kv_cache_manager.html>

- TRT-LLM 官方文档明确支持自定义 scheduler
  - 可以继承 `CapacityScheduler` / `MicroBatchScheduler`
  - 也可以直接继承 `RequestScheduler`
  - 参考：<https://nvidia.github.io/TensorRT-LLM/latest/torch/scheduler.html>

- TRT-LLM 支持新增自定义模型并注册到 PyTorch backend
  - 参考：<https://nvidia.github.io/TensorRT-LLM/torch/adding_new_model.html>

- TensorRT-LLM 1.0 起，PyTorch-based architecture 已成为默认体验
  - 参考：<https://nvidia.github.io/TensorRT-LLM/release-notes.html>

- 当前 upstream 的 `PyExecutor` 实现中，`SchedulerConfig.use_python_scheduler` 会切换到纯 Python scheduler 路径
  - 这意味着自定义 scheduler 不是“只能改 C++”的假设
  - 但当前仍需要在 `PyExecutor` 创建路径中接入自定义类，还不是像 vLLM `--scheduler-cls` 那样完全配置化的插件接口

- 旧的 Python runtime 仍存在，但官方文档明确标为“不推荐使用”
  - 参考：<https://nvidia.github.io/TensorRT-LLM/reference/memory.html>

- TRT-LLM 的 packed mode 相比 padded mode 更高效，文档也明确建议优先 packed 输入
  - 参考：<https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html>

## 5. 关键架构判断

### 5.1 对当前模型来说，需要区分“决策闭环”和“backend step 生命周期”

当前项目每一步的逻辑是：

`backbone step -> cp -> codec_sum -> next_embed -> 下一步 backbone`

因此真正的自回归闭环是：

`backbone + cp`

这意味着：

- 如果只把 `backbone` 放进 TRT-LLM，而 `cp` 仍留在外部 runtime
- 那么 TRT-LLM 管理的并不是完整 AR stage
- 真正的 decode loop 仍然在用户自建 runtime 手里

结论：

**从“谁决定下一步 `next_embed`”的角度看，`backbone+cp` 是核心 AR 闭环。**

但这不是完整的 runtime 边界。

当前一次 backend step 还会同步推进：

- `wav` / `full_codec` 的产出
- `code2wav` 的 sliding-window KV
- 17 个 conv state
- 4 个 transconv overlap state
- ref warm state 与 ping-pong buffer 语义

这意味着：

- 从“决策闭环”角度，`backbone+cp` 是核心
- 从“是否只维护一套 batch/cache/runtime”角度，`code2wav` 也属于同一步 request lifecycle

因此更准确的结论是：

**如果只把 `backbone+cp` 迁入 TRT-LLM，而把 `code2wav` 完全留在外部，就会残留第二套 `code2wav` 的 batch/KV/state runtime。**

### 5.2 `WAIT_TEXT` 是业务态，不是普通 EOS/Cancel 语义

当前系统存在一种标准 LLM 不具备的状态：

- 不是 finished
- 不是 cancelled
- 不是继续 pad
- 而是“先保留上下文，等外部文本到了再继续”

因此 stock `Executor` / stock `generate()` 并不直接匹配当前需求。

正确的抽象应是：

- `READY_DECODE`
- `WAIT_TEXT`
- `PAD_PHASE`
- `FINISHED`

然后由 scheduler 决定当前 step 是否把某个 request 放进 decode batch。

从当前 TRT-LLM 能力看，这件事不是无解：

- 官方文档已明确支持自定义 scheduler
- 当前 Python scheduler 路径已经存在“request 还活着，但当前 step 不 ready 就跳过”的先例

因此更合理的实现方式不是滥用容量调度里的 `pause` 语义，而是：

- 保留 request / KV / state 生命周期
- 通过自定义 scheduler 的 readiness gate 跳过 `WAIT_TEXT` request
- 外部文本到达后再恢复进入 decode batch

### 5.3 `code2wav` 不决定 `next_embed`，但在“最小系统修改”目标下不应轻易拆成独立 runtime

`code2wav` 很重要，但它不决定下一步 `next_embed`，因此它不是最核心的“决策闭环”。

但如果把它完全拆出 TRT-LLM runtime，会立刻带来另一个问题：

- 仍需单独维护 `code2wav` 的 batch 语义
- 仍需单独维护 fixed-window attention 的 KV cache
- 仍需单独维护 conv/transconv recurrent states
- 而且这些状态与 talker decode 是同一步 cadence 同步推进的

因此需要区分两种“最小化修改”目标：

- 如果要最小化 **TRT-LLM 首轮 PoC 开发量**，可以先只迁 `backbone+cp`
- 如果要最小化 **整个系统的 runtime 分裂与上层改动**，更合理的是让 `backbone/cp/code2wav` 共享同一 request lifecycle

结论：

**是否让 `code2wav` 进入 TRT-LLM，不应只看“它是不是 AR 决策核心”，还要看团队是否接受长期保留第二套 `code2wav` batching/cache runtime。**

## 6. 备选路线评估

### 路线 A：继续保留现有 runtime，只做单引擎混合精度

#### 核心思路

- 继续使用当前自建 `engine_loop + executor + kv_cache_pool`
- 不立即迁移到 TRT-LLM runtime
- 优先把单引擎精度改成：
  - `backbone=bf16`
  - `cp=fp32`
  - `code2wav=bf16`

#### 优点

- 与当前系统最兼容
- 不破坏 `WAIT_TEXT` / streaming pause-resume 语义
- 风险最低
- 能最快解决当前最现实的 `cp` 幻觉问题

#### 缺点

- 不能根本解决 paged KV / packed decode 问题
- 仍需维护自建 KV 池、gather/scatter 和 padded batch
- 只能获得“精度收益”，难以获得“runtime 架构收益”

#### 判断

**适合作为短期产品化主线。**

### 路线 B：TRT-LLM PyTorch backend，自定义 step model，自定义 scheduler/request readiness

#### 核心思路

- 使用 TRT-LLM PyTorch backend
- 使用 TRT-LLM 官方支持的自定义 scheduler 路径
- 通过 `PyExecutor`、`Scheduler`、`KVCacheManager` 管理 request readiness 与 AR 执行
- 外层 runtime 继续保留 session、segment、gateway 和 `WAIT_TEXT` 业务状态
- 优先评估让 `backbone/cp/code2wav` 共享同一 request lifecycle
- 如果首轮 PoC 需要降低实现门槛，可短期保留 `code2wav` 外挂，但应视为过渡形态，而非目标架构

#### 优点

- 能获得 TRT-LLM 在 AR 阶段的核心收益
  - paged KV
  - packed / varlen 风格的 attention 路径
  - continuous batching 的标准调度骨架
- 自定义 scheduler 已有官方文档支撑，不再只是“理论上可 fork”
- 比 C++ Executor 深定制更易开发和调试
- 更贴合 TRT-LLM 当前主推的默认后端方向
- 可以在模型内部显式控制精度边界
  - `backbone bf16`
  - `cp fp32`
- 若 backend 继续维持“单步统一输出”外观，上层 engine/gateway/streaming 协议可以基本不变

#### 难点

- 需要重新实现 step model 的模型定义和 forward
- 需要为 `WAIT_TEXT` 建立自定义 request readiness 和 scheduler 逻辑
- 需要明确 `cp` 的 KV / hidden / sampled states 哪些必须保 `fp32`
- 如果让 `code2wav` 共享同一 request lifecycle，则需要一起建模：
  - `c2w` sliding-window KV
  - 17 个 conv state
  - 4 个 transconv overlap state
- 如果不让 `code2wav` 共享同一 request lifecycle，则系统仍会保留第二套 `code2wav` batch/cache/runtime
- 与当前 fused TRT engine 的 build/export 链不同，需要新的一套模型接入流程
- 自定义 scheduler 虽然有官方路径，但当前自定义类仍需接入 `PyExecutor` 创建链，不是纯配置化插件

#### 判断

**这是中期最值得投入的 TRT-LLM 路线。**

如果未来确实要迁移到标准化 paged KV runtime，这条路线是推荐主线。
但它内部应再区分：

- **目标形态**：`backbone/cp/code2wav` 共享同一 request lifecycle
- **过渡形态**：先让 `backbone+cp` 进入 TRT-LLM，`code2wav` 临时外挂

### 路线 C：TRT-LLM TensorRT backend，统一 `backbone+cp` 引擎，深度定制 Executor/C++ runtime

#### 核心思路

- 将 `backbone+cp` 编成统一 TensorRT engine
- 使用 TRT-LLM 的 TensorRT backend / C++ runtime / Executor 体系
- 必要时做更深的 C++ 侧调度定制

#### 优点

- 理论性能上限最高
- 更贴近 TRT-LLM 原生 TensorRT 路线
- 长期可能获得最佳执行效率

#### 缺点

- 工程量最大
- 需要更深理解 TensorRT build、插件、强类型精度边界和 executor 生命周期
- `WAIT_TEXT` 这类业务态需要自己补更多 runtime 逻辑
- 调试链路最重

#### 判断

**技术上可行，但不适合作为当前产品化阶段第一选择。**

### 路线 D：stock `Executor` / stock `generate()` 直接托管整条生成循环

#### 核心思路

- 将当前模型尽量适配为普通生成模型
- 直接使用 TRT-LLM 高层 API

#### 判断

**不推荐。**

原因有三点：

1. 当前业务需要 `WAIT_TEXT`
2. 当前真实 AR stage 是 `backbone+cp`，不是单纯 token decode
3. `cp` 的高精度边界需要更细控制

这条路线看起来工程量小，但很可能在关键语义上卡死，最后既拿不到完整收益，也需要回退。

## 7. 关于“把 backbone、cp、code2wav 全融合进一张图”的判断

从图构建角度看，这件事不是完全做不到，但应明确区分两种“融合”：

### 7.1 计算图融合

可以做。

即：

- `backbone`
- `cp`
- `code2wav`

都放在统一模型图或统一工程体系里处理。

从算子表达角度看，`code2wav` 中的 `ConvTranspose1d` 不是这条路线的根本 blocker：

- 可以通过等价的 `ConvTranspose2d` 单高维包装实现
- 也可以继续沿用当前 export/build 侧的图改写思路

真正的难点仍然是 request lifecycle、scheduler 和 state 管理，而不是单一算子能否表示。

### 7.2 runtime 责任融合

这里不宜简单回答“应该”或“不应该”，而应按目标区分：

- 如果目标是 **最小化 TRT-LLM 首轮接入复杂度**，可以先不把 `code2wav` 全量并入
- 如果目标是 **最小化整个系统修改量，并避免残留第二套 runtime**，则应优先让 `code2wav` 与 `backbone/cp` 共享同一 request lifecycle

因此更推荐的判断顺序是：

- 第一阶段优先统一 request lifecycle 与 scheduler 语义
- 不强求第一天就必须是一张静态图、一个 engine
- 但要避免把“`code2wav` 永久留在外部第二套 runtime”误判成最优终态

## 8. 关于“cp 融入 TRT-LLM 后会不会又掉回 bf16”的判断

不会自动掉回 `bf16`，但前提是实现方式正确。

需要明确：

- “使用 TRT-LLM”不等于“全图统一低精度”
- TensorRT 支持强类型精度控制
- TRT-LLM PyTorch backend 允许在模型内部控制不同子模块的 dtype 边界

真正的风险不在“是否采用 TRT-LLM”，而在于：

- 是否为了省事，把 `cp` 也编进一个统一的 `bf16` 路径
- 是否在 `next_embed`、hidden state、cp state、sampling logits 等边界上错误地回落到 `bf16`

因此一旦走 TRT-LLM 路线，需要把下面几类状态作为重点验证对象：

- `cp` 的输入 hidden
- `cp` 的内部 attention / MLP / residual
- `cp` 的 KV cache
- `codec_sum -> next_embed` 的桥接张量
- 和 `code2wav` 的交界输出

## 9. 推荐路线

### 9.1 总体建议

推荐采用“两阶段主线 + 一条并行验证线”：

#### 主线阶段 1：当前 engine 的单引擎混合精度

目标：

- 优先解决 `cp` 在 `bf16` 下的幻觉问题
- 不破坏当前 `WAIT_TEXT` 和流式稳定性
- 保持当前产品可迭代节奏

建议配置：

- `backbone=bf16`
- `cp=fp32`
- `code2wav=bf16`
- `io_float=bf16`

#### 主线阶段 2：TRT-LLM PyTorch backend PoC

目标：

- 验证 TRT-LLM 自定义 scheduler 路线是否能承接当前 request lifecycle
- 验证三件事：
  - `WAIT_TEXT` 能否通过官方支持的自定义 scheduler/request readiness 保住
  - `cp fp32` 能否在 TRT-LLM 中稳定保住
  - 相比当前 padded execution，是否能在真实 workload 上拿到明显收益
  - `code2wav` 是否必须共享同一 request lifecycle 才能避免第二套 runtime 复杂度

实现边界：

- 外层 gateway / websocket / session / segment 语义继续保留
- scheduler 优先走 TRT-LLM 官方 `torch/scheduler` 路径
- 优先评估 `backbone/cp/code2wav` 共享同一 request lifecycle
- 如果首轮实现复杂度过高，可接受过渡变体：先让 `backbone+cp` 进入 TRT-LLM，`code2wav` 暂时外挂

#### 并行验证线：stock Executor 否证实验

可以做一个很小的实验验证 stock `Executor` 是否能够以低成本表达 `WAIT_TEXT`。

预期：

- 大概率证伪
- 价值在于尽快排除“也许高层 API 就够”的幻想

### 9.2 何时进入 TRT-LLM 正式迁移

建议只有在满足以下条件时，才将 TRT-LLM 路线从 PoC 升级为正式主线：

1. `cp fp32` 路径在 TRT-LLM 中验证稳定
2. `WAIT_TEXT` 语义可以通过自定义 scheduler/request readiness 实现
3. `code2wav` 的状态管理边界已经明确
4. 真实业务压测下，吞吐或显存效率收益足以覆盖迁移成本
5. 代码维护复杂度没有明显高于当前方案

## 10. 里程碑建议

### M0：现状冻结与指标基线

- 固化当前引擎在代表性 workload 下的指标
- 指标至少包括：
  - 首包延迟
  - 吞吐
  - 显存峰值
  - 幻觉率 / 重复率
  - 流式暂停恢复正确性

### M1：单引擎混合精度落地

- 完成 `backbone/cp/code2wav` 分段精度配置
- 用现有 runtime 验证 `cp=fp32` 是否稳定解决主要问题

### M2：TRT-LLM AR PoC

- 开启 TRT-LLM `use_python_scheduler`
- 自定义 scheduler 实现 `READY_DECODE / WAIT_TEXT`
- 验证 request readiness gate 是否足以表达“等待外部文本”
- 优先评估 `backbone/cp/code2wav` 共享同一 request lifecycle
- 若复杂度过高，再退回过渡变体：只迁 `backbone+cp`

### M3：收益评估

- 与当前主线对比：
  - 实际显存占用
  - batch 利用率
  - throughput
  - 尾延迟
  - 代码复杂度
  - 是否仍残留第二套 `code2wav` batching/cache runtime

### M4：是否转主线

- 若收益显著且语义稳定，则规划迁移
- 若收益不显著或语义代价过高，则维持当前 engine，并只吸收混合精度成果

## 11. 风险清单

### 风险 1：TRT-LLM 收益被高估

如果 `cp` 仍在 TRT-LLM 外部，真正的 AR loop 依然在用户 runtime 手里，那么 paged KV 和 continuous batching 的收益会被削弱。

### 风险 2：`WAIT_TEXT` 语义侵入太深

虽然 TRT-LLM 已官方支持自定义 scheduler，但如果要把 `WAIT_TEXT` 真正做成 runtime 的一等语义，scheduler、request readiness 和 KV 生命周期仍需共同设计。

### 风险 3：`cp fp32` 的边界控制不完整

如果只把部分输入保成 `fp32`，但中间 residual / KV / logits 路径仍然回落到 `bf16`，可能会出现“看起来迁了混合精度，实际问题没解决”的情况。

### 风险 4：一次性迁移范围过大

如果第一阶段同时迁移：

- `backbone`
- `cp`
- `code2wav`
- scheduler
- gateway 协议边界

那么定位问题会非常困难。

### 风险 5：`code2wav` 留在外部导致 runtime 分裂长期固化

如果 `backbone+cp` 已进入 TRT-LLM，但 `code2wav` 长期完全留在外部：

- 系统仍需维护第二套 `code2wav` batch 语义
- 仍需维护第二套 fixed-window KV/state 管理
- 可能削弱“迁移后系统明显更简单”的预期收益

## 12. 最终建议

本项目当前最合理的技术路线不是“立即全面切到 TRT-LLM”，而是：

1. **先用当前 runtime 解决混合精度问题**
2. **再用 TRT-LLM PyTorch backend 验证 `backbone+cp` 的 AR runtime 路线**
3. **重点验证 TRT-LLM 官方 custom scheduler 路线能否承接 `WAIT_TEXT`**
4. **只有当 `WAIT_TEXT + fp32 cp + paged KV` 三者能同时成立，且 `code2wav` 边界已清晰时，再考虑将 TRT-LLM 升格为正式主线**

如果只问一句“现在应该把产品主线押在哪”，我的建议是：

**短期押当前 engine 的混合精度修复，中期押 TRT-LLM PyTorch backend + custom scheduler 的 PoC；目标形态优先追求统一 request lifecycle，不建议当前阶段直接押 C++ Executor 深度定制。**

## 参考资料

### 仓内实现

- `engine/backend/engine_loop.py`
- `engine/backend/executor.py`
- `engine/backend/kv_cache_pool.py`
- `engine/config.py`
- `docs/roadmap.md`
- `docs/dev/architecture/engine_decisions.md`
- `docs/dev/design/mixed_precision_plan.md`

### 官方文档

- TensorRT-LLM Executor API  
  <https://nvidia.github.io/TensorRT-LLM/advanced/executor.html>

- TensorRT-LLM Architecture Overview  
  <https://nvidia.github.io/TensorRT-LLM/architecture/overview.html>

- TensorRT-LLM KV Cache Manager  
  <https://nvidia.github.io/TensorRT-LLM/torch/kv_cache_manager.html>

- TensorRT-LLM Scheduler  
  <https://nvidia.github.io/TensorRT-LLM/latest/torch/scheduler.html>

- TensorRT-LLM Adding a New Model in PyTorch Backend  
  <https://nvidia.github.io/TensorRT-LLM/torch/adding_new_model.html>

- TensorRT-LLM Release Notes  
  <https://nvidia.github.io/TensorRT-LLM/release-notes.html>

- TensorRT-LLM Memory Usage  
  <https://nvidia.github.io/TensorRT-LLM/reference/memory.html>

- TensorRT-LLM GPT Attention / packed vs padded  
  <https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html>
