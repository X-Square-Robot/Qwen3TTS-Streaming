# 流式 TTS 执行架构重构方案

> 编写日期：2026-09-14  
> 状态：设计稿，尚未作为默认运行时发布  
> 范围：文本入口、增量 TN、引擎调度和音频发送

## 1. 问题结论

当前接口使用了 `async`，但执行链仍存在两个同步长临界区：前端在事件循环任务中连续执行 TN、映射和 token 分发；引擎线程的 `_drain_inbox()` 无上限排空请求，活跃 segment 的追加请求还会触发 embedding。

回放中首段 raw 音频之后出现约 500ms 空洞。该空洞期间 VAD 单帧耗时低于 0.34ms，队列等待低于 0.05ms；引擎日志显示 segment 进入 `awaiting text`，随后集中处理 TN/token，下一段 raw 音频才产生。因此问题首先是调度和状态边界，TN 单次计算速度是后续优化项。

## 2. 设计原则

1. **会话内有序，会话间并行。** 同一会话的 TN、raw/spoken 映射、cursor revision 和 high-water 只能由一个有序 owner 修改；不同会话可以在线程池中并行。
2. **音频输出具有调度优先级。** 文本输入可以排队，但不能无限期阻塞已完成的 GPU 音频结果。
3. **所有队列有界且可观测。** 队列必须报告深度、入队时间、等待时间和丢弃/背压原因；音频不得静默丢弃。
4. **异步不等于并行。** 事件循环只负责收发和调度，CPU 密集工作不得在其中运行。
5. **TN 语义与实现解耦。** 可以保留 Python 参考实现，也可以使用 Rust/C 扩展加速，但 raw 坐标、owner span、commit fence 和 fallback 语义必须完全一致。

## 3. 目标架构

```text
Transport I/O (async)
        │  O(1) 入队
        ▼
Per-session Text Actor (有序)
        │
        ├── TN worker (共享线程池，可替换 Rust/C)
        │       └── TextCommitBatch + raw/spoken owner mapping
        │
        └── Planner/Dispatcher (有序 revision)
                ├── UPDATE_CURSOR_PLAN
                └── START/APPEND_TOKENS

Engine Scheduler (专用线程)
        ├── 有界 command drain
        ├── decode launch / GPU future harvest
        └── Priority Audio Egress
                ▼
Audio Egress (async) → VAD / reorder / gateway callback
```

### 3.1 Transport I/O

网关收到文本后只做协议校验、session 查找和计数，然后写入每会话有界 `TextIngressQueue`。不得在网关协程中执行 TN、tokenizer 或 cursor plan。队列满时产生明确背压指标，不能悄悄扩大无界内存。

### 3.2 Per-session Text Actor

每个会话只有一个 actor 顺序消费文本、flush、取消和超时消息。actor 保持 `CanonicalTextJournal`、`IncrementalTextCommitter`、emoji carry 和 raw high-water 的唯一写权限，避免多线程并发破坏映射。

actor 可以把纯 CPU 的 TN 工作提交到共享 worker，但必须以 session sequence number 串行收割结果。worker 返回不可变的 `TextCommitBatch`，不能直接修改 Session 或 engine queue。

### 3.3 TN worker 与 Rust/C

TN worker 的接口保持现有主 TN 契约：输入 raw delta 和 metadata，输出 `TextCommit`、spoken text、owner mapping、commit fence、closure/fallback reason。Python 实现作为参考和回退；Rust/C 实现必须通过冻结输入的逐 commit parity 测试后才能启用。

Rust/C 的收益是降低单批 TN 的 CPU 时间和 GIL 争用；它不能替代 actor、队列和调度隔离。即使 TN 速度提升，若仍在事件循环中一次性处理几百个 commit，音频仍可能被饿死。

### 3.4 Planner/Dispatcher

planner 将一个 `TextCommitBatch` 转为带单调 sequence 的 engine command。`UPDATE_CURSOR_PLAN` 与其对应的 token command 使用同一 revision，engine 只接受满足 revision 顺序的命令，防止为了音频优先级而重排语义状态。

planner 对一次 TN 返回的 `TextCommitBatch` 原子应用，不能把已经推进 fence 的 committer 决定拆半。达到调度预算时，只在 tokenizer/dispatcher 的可分片阶段让出执行权；分片复用同一 revision，并让音频结果先被发送。批次边界不能回退 raw/spoken high-water。

### 3.5 Engine Scheduler

engine loop 每轮按以下顺序运行：

1. 先收割已完成的 GPU future 并发布音频；
2. 处理有限数量的控制/文本 command；
3. 在 command 预算耗尽前启动 prefill/decode；
4. decode flight 期间只做有界 housekeeping，不无限 drain inbox；
5. 音频结果进入独立优先级 egress queue。

command 预算应同时受请求数和墙钟时间限制，例如每次最多 32 个请求或 1ms，具体值用回放校准。预算返回时保持 FIFO，不丢请求、不跨 session 重排。

### 3.6 Audio Egress

音频发送从 engine scheduler 中独立出来。engine 只负责把带 sequence、segment 和 metrics 的音频块放入 egress queue；VAD、reorder 和 gateway callback 在异步 egress task 中执行。发送端阻塞时必须记录 backlog 和最大帧间隔，不能阻塞下一轮 decode。

## 4. 迁移步骤

### Phase 0：观测闭环

保留当前 dump，补齐 `engine_raw_emit → result_queue_put → gateway_on_audio_enter → effective_audio` 四个时间点，并记录 text actor batch 的开始/结束和处理数量。

### Phase 1：引擎有界调度

给 `_drain_inbox()` 增加请求数/时间预算，先收割 ready audio，再进入 prefill/decode。该阶段不改变 TN 语义，可单独验证 500ms 空洞是否消失。

### Phase 2：前端 actor 化

把 `push_text_input()` 改为快速入队；每会话 actor 顺序执行现有 TN 和 dispatcher。先保留 Python TN，验证 raw mapping、commit 顺序和取消行为。

### Phase 3：TN worker 化

将 TN 计算移出事件循环，增加超时、取消和序列号校验。必要时以 Rust/C 扩展替换 worker 实现，保留 Python parity fallback。

### Phase 4：独立音频 egress

把结果发送、VAD 和 gateway callback 从 engine relay 的同步路径拆出，增加端到端帧间隔和 underflow 回放验收。

## 5. 验收指标

- 首个 raw 音频到首个 effective 音频的 wall time 分解完整，不能只报一个汇总值；
- 连续 token burst 下，音频帧间最大间隔不因 TN 批量处理超过预算；
- 同一会话 `TextCommit`、cursor revision、token command 严格单调；
- Python TN 与 Rust/C TN 在冻结输入上的 spoken text、owner mapping 和 closure reason 完全一致；
- 取消、EOS、tail rewrite、并发 session 和 engine queue 背压均有测试；
- 不引入官方 PyTorch 在线 backend，不改变 native cursor/TRT 的既有职责边界。
