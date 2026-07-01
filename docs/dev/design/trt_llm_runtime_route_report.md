**English** | [中文](trt_llm_runtime_route_report.zh-CN.md)

# TensorRT-LLM Runtime Technical Route Review Report

> Date: 2026-06-22
> Status: Review draft
> Conclusion level: Intended as input to an overall route review, not as a direct implementation guide

## Executive Summary

The core problem the project currently faces is not a single question of "should we switch to TRT-LLM," but rather three constraints existing simultaneously:

1. `backbone` can use `bf16`
2. `cp` is more numerically sensitive and requires `fp32` to avoid noticeable hallucination
3. The streaming scenario has the business semantics of "waiting for external text before continuing to decode," which cannot be handled by the run-to-completion generation loop of a standard LLM

Based on the current repository implementation and the official documentation as of 2026-06-22, the conclusions are as follows:

- **Short-term productization main line**: Continue to keep the current self-built runtime semantics, and prioritize solving the single-engine mixed precision problem, i.e. `backbone=bf16`, `cp=fp32`, `code2wav=bf16`.
- **Medium-term evolution main line**: If we genuinely need to gain the benefits of paged KV / packed decode / native continuous batching, prioritize evaluating the **TRT-LLM PyTorch backend + custom step model + custom scheduler/request readiness**.
- **A key addition to the route judgment**: `backbone+cp` is still the core loop that decides `next_embed`, but `code2wav` also carries its own fixed-window KV and conv/transconv recurrent states; if it stays entirely outside TRT-LLM, the system will still retain a second batch/cache runtime.
- **Not recommended as a main line**: Directly using the stock `Executor` / stock `generate()` to host the entire generation loop. It suits standard autoregressive requests but not the `WAIT_TEXT` semantics of the current project.
- **High-risk long-term route**: `TensorRT backend + unified backbone+cp engine + deep C++/Executor customization` is feasible, but it has the highest engineering complexity, maintenance cost, and debugging cost, and is not suitable as the first choice for the current productization stage.

The one-line judgment:

**TRT-LLM is worth doing, but it should be treated as a "medium-term AR runtime refactoring route," not a silver bullet for immediately replacing all existing scheduling semantics right now.**

## 1. Background and Problem Definition

The current engine already has its own session management, segment-level streaming protocol, prefill, decode batch, KV slot pool, and code2wav state management. Actual deployment has exposed two main problems:

1. **Insufficient KV cache management efficiency**
   - The current implementation is still a compromise of "self-built continuous scheduling + padded execution"
   - It cannot leverage the paged attention / paged KV capabilities of runtimes like vLLM / TRT-LLM

2. **Insufficient `cp` precision**
   - `cp` exhibits noticeable numerical sensitivity under `bf16`
   - If the current fused TRT engine runs entirely at `bf16`, it easily induces hallucination

At the same time, the project has a constraint that distinguishes it from a standard LLM:

3. **Decode may need to wait for external text**
   - When the streaming text has not yet fully arrived, the current session pauses in a "waiting for more text" state
   - During the pause, it needs to retain `last_codec_sum` and the related KV states
   - Decoding resumes after the text arrives, rather than immediately entering pad or finishing

This means the project's real requirement is not "running an ordinary Qwen3," but "running a custom autoregressive system with business-state pause/resume capability."

## 2. Current Implementation State

### 2.1 Key semantics of the current runtime

The existing implementation in the repo has already made the following behaviors explicit:

- `engine/backend/engine_loop.py:963-980`
  - `next_embed` is held in `float32` at the loop layer
  - On streaming resume, a new `next_embed` is formed via `last_codec_sum + trailing_text_embed`

- `engine/backend/engine_loop.py:1179-1192`
  - When the current step obtains `codec_sum`, but the subsequent text has not yet fully arrived and `input_complete=false`
  - The session enters the "pause and wait for text" semantics
  - At this point it retains `last_codec_sum`, does not pad, and does not continue to the next decode step

- `engine/backend/executor.py:1078-1125`
  - The decode step is driven explicitly by the outer layer, running only one step at a time
  - Each step gathers the KV of the currently active slots into a batch

- `engine/backend/executor.py:1246-1455`
  - The current fused executor manually constructs `input_embeds`, `position_ids`, `attention_bias`
  - It manually organizes `talker_past_kv`, `c2w_past_kv`, and the conv/transconv states
  - This is essentially the self-built runtime emulating a "multi-state autoregressive executor"

### 2.2 Two structural bottlenecks of the current implementation

#### Bottleneck A: padded execution

The current decode batch pads `talker_past_kv` to the maximum past length within the batch, then masks the invalid columns via `attention_bias`. This is typical padded iteration-level batching.

This path works, but it introduces:

- Redundant GPU memory access when long and short requests run mixed within a batch
- Higher KV gather/scatter cost
- Difficulty reaching the resource utilization of the paged attention route

#### Bottleneck B: dtype inconsistency between the loop layer and the engine layer

The current code is already aware that the "internal compute dtype" and the "external I/O dtype" are not the same thing. `engine/config.py:638-642` already spells out that:

- `engine_dtype=fp32`
- `triton_io_float_dtype=bf16`

can coexist.

But when the current decode emits, `next_embed` is cast to `self._config.dtype`:

- `engine/backend/executor.py:1087-1088`

This means that even if the loop layer holds `next_embed` in `float32`, as long as the engine boundary is `bf16`, the critical path still falls back to `bf16`.

## 3. Requirements and Constraints

This route review is judged against the following goals:

### 3.1 Must satisfy

- Preserve the current streaming protocol semantics, especially `WAIT_TEXT` / `pause then resume`
- Solve the `cp` numerical stability problem under `bf16`
- Support continuous batching or equivalent iteration-level scheduling
- Be able to manage KV cache for heterogeneous-length requests, no longer relying long-term on padding to the maximum length
- Preserve the current Python gateway / websocket / session protocol layer as much as possible

### 3.2 Satisfy if possible

- `backbone` leverages paged KV / packed decode
- `code2wav` keeps a high-throughput TensorRT path
- Reduce the complexity of the in-house KV pool, manual gather/scatter, and state assembly

### 3.3 Should not be sacrificed at the current stage

- Debuggability
- Business semantic correctness
- Streaming stability

The current focus explicitly stated in the project's `docs/roadmap.md` remains stability, deployment, and maintainability, so the route choice should prioritize productization cost over theoretical peak performance.

## 4. External Technical Facts

The following judgments are based on the TRT-LLM official documentation and the upstream source code as of 2026-06-22:

- The `Executor API` is a high-level asynchronous execution interface that supports in-flight batching / continuous batching style request execution
  - Reference: <https://nvidia.github.io/TensorRT-LLM/advanced/executor.html>

- `PyExecutor` runs continuously in the background, composed of components such as `Scheduler`, `KVCacheManager`, `ModelEngine`
  - The `Scheduler` is responsible for determining which active requests are ready for execution in the current step
  - Reference: <https://nvidia.github.io/TensorRT-LLM/architecture/overview.html>

- `KVCacheManager` can be customized and plugged into `PyExecutor`
  - Reference: <https://nvidia.github.io/TensorRT-LLM/torch/kv_cache_manager.html>

- The TRT-LLM official documentation explicitly supports custom schedulers
  - You can subclass `CapacityScheduler` / `MicroBatchScheduler`
  - You can also directly subclass `RequestScheduler`
  - Reference: <https://nvidia.github.io/TensorRT-LLM/latest/torch/scheduler.html>

- TRT-LLM supports adding a custom model and registering it into the PyTorch backend
  - Reference: <https://nvidia.github.io/TensorRT-LLM/torch/adding_new_model.html>

- As of TensorRT-LLM 1.0, the PyTorch-based architecture has become the default experience
  - Reference: <https://nvidia.github.io/TensorRT-LLM/release-notes.html>

- In the current upstream `PyExecutor` implementation, `SchedulerConfig.use_python_scheduler` switches to the pure Python scheduler path
  - This means a custom scheduler is not limited to the "you can only change C++" assumption
  - But it currently still requires plugging the custom class into the `PyExecutor` creation path, and is not yet a fully configuration-driven plugin interface like vLLM's `--scheduler-cls`

- The old Python runtime still exists, but the official documentation explicitly marks it as "not recommended"
  - Reference: <https://nvidia.github.io/TensorRT-LLM/reference/memory.html>

- TRT-LLM's packed mode is more efficient than padded mode, and the documentation explicitly recommends preferring packed input
  - Reference: <https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html>

## 5. Key Architectural Judgments

### 5.1 For the current model, we need to distinguish the "decision loop" from the "backend step lifecycle"

The logic of each step in the current project is:

`backbone step -> cp -> codec_sum -> next_embed -> next backbone step`

Therefore the true autoregressive loop is:

`backbone + cp`

This means:

- If only `backbone` is put into TRT-LLM, while `cp` remains in the external runtime
- Then what TRT-LLM manages is not a complete AR stage
- The true decode loop is still in the hands of the user's self-built runtime

Conclusion:

**From the perspective of "who decides the next `next_embed`," `backbone+cp` is the core AR loop.**

But this is not the complete runtime boundary.

A single backend step currently also advances in synchrony:

- The production of `wav` / `full_codec`
- The sliding-window KV of `code2wav`
- 17 conv states
- 4 transconv overlap states
- The ref warm state and ping-pong buffer semantics

This means:

- From the "decision loop" perspective, `backbone+cp` is the core
- From the "maintain only one batch/cache/runtime" perspective, `code2wav` also belongs to the same step's request lifecycle

Therefore the more accurate conclusion is:

**If only `backbone+cp` is migrated into TRT-LLM while `code2wav` stays entirely outside, a second `code2wav` batch/KV/state runtime will remain.**

### 5.2 `WAIT_TEXT` is a business state, not ordinary EOS/Cancel semantics

The current system has a state that a standard LLM does not have:

- Not finished
- Not cancelled
- Not continuing to pad
- But rather "retain the context first, and continue once the external text arrives"

Therefore the stock `Executor` / stock `generate()` does not directly match the current requirement.

The correct abstraction should be:

- `READY_DECODE`
- `WAIT_TEXT`
- `PAD_PHASE`
- `FINISHED`

Then the scheduler decides whether to put a given request into the decode batch for the current step.

From the perspective of current TRT-LLM capabilities, this is not unsolvable:

- The official documentation already explicitly supports custom schedulers
- The current Python scheduler path already has a precedent of "the request is still alive, but skip it if it is not ready for the current step"

Therefore the more reasonable implementation is not to abuse the `pause` semantics in capacity scheduling, but rather to:

- Preserve the request / KV / state lifecycle
- Skip `WAIT_TEXT` requests via the readiness gate of a custom scheduler
- Resume entering the decode batch once the external text arrives

### 5.3 `code2wav` does not decide `next_embed`, but under the "minimal system change" goal it should not be lightly split into an independent runtime

`code2wav` is very important, but it does not decide the next `next_embed`, so it is not the most core "decision loop."

However, if it is entirely split out of the TRT-LLM runtime, it immediately introduces another problem:

- The `code2wav` batch semantics must still be maintained separately
- The fixed-window attention KV cache must still be maintained separately
- The conv/transconv recurrent states must still be maintained separately
- And these states advance in synchrony with the talker decode at the same step cadence

Therefore we need to distinguish two "minimize change" goals:

- If we want to minimize the **development effort of the first TRT-LLM PoC**, we can first migrate only `backbone+cp`
- If we want to minimize the **runtime fragmentation of the whole system and upper-layer changes**, it is more reasonable to have `backbone/cp/code2wav` share the same request lifecycle

Conclusion:

**Whether to bring `code2wav` into TRT-LLM should not be decided solely by "whether it is the AR decision core," but also by whether the team accepts retaining a second `code2wav` batching/cache runtime long-term.**

## 6. Alternative Route Evaluation

### Route A: Keep the existing runtime, doing only single-engine mixed precision

#### Core idea

- Continue using the current self-built `engine_loop + executor + kv_cache_pool`
- Do not immediately migrate to the TRT-LLM runtime
- Prioritize changing the single-engine precision to:
  - `backbone=bf16`
  - `cp=fp32`
  - `code2wav=bf16`

#### Pros

- Most compatible with the current system
- Does not break the `WAIT_TEXT` / streaming pause-resume semantics
- Lowest risk
- Solves the most realistic `cp` hallucination problem the fastest

#### Cons

- Does not fundamentally solve the paged KV / packed decode problem
- Still requires maintaining the self-built KV pool, gather/scatter, and padded batch
- Only gains the "precision benefit," hard to gain the "runtime architecture benefit"

#### Judgment

**Suitable as the short-term productization main line.**

### Route B: TRT-LLM PyTorch backend, custom step model, custom scheduler/request readiness

#### Core idea

- Use the TRT-LLM PyTorch backend
- Use the officially supported TRT-LLM custom scheduler path
- Manage request readiness and AR execution via `PyExecutor`, `Scheduler`, `KVCacheManager`
- The outer runtime continues to retain the session, segment, gateway, and `WAIT_TEXT` business states
- Prioritize evaluating having `backbone/cp/code2wav` share the same request lifecycle
- If the first PoC needs a lower implementation barrier, `code2wav` can be kept externally attached in the short term, but this should be treated as a transitional form, not the target architecture

#### Pros

- Can gain the core benefits of TRT-LLM in the AR stage
  - paged KV
  - packed / varlen style attention path
  - the standard scheduling skeleton of continuous batching
- The custom scheduler already has official documentation support, no longer merely "theoretically forkable"
- Easier to develop and debug than deep C++ Executor customization
- More aligned with the default backend direction TRT-LLM is currently pushing
- The precision boundary can be explicitly controlled inside the model
  - `backbone bf16`
  - `cp fp32`
- If the backend continues to maintain a "unified single-step output" appearance, the upper-layer engine/gateway/streaming protocol can remain largely unchanged

#### Difficulties

- Need to reimplement the model definition and forward of the step model
- Need to establish custom request readiness and scheduler logic for `WAIT_TEXT`
- Need to clarify which of `cp`'s KV / hidden / sampled states must stay `fp32`
- If `code2wav` shares the same request lifecycle, then the following must be modeled together:
  - `c2w` sliding-window KV
  - 17 conv states
  - 4 transconv overlap states
- If `code2wav` does not share the same request lifecycle, the system will still retain a second `code2wav` batch/cache/runtime
- Different from the build/export chain of the current fused TRT engine, a new model onboarding flow is needed
- Although the custom scheduler has an official path, the current custom class still needs to plug into the `PyExecutor` creation chain, and is not a pure configuration-driven plugin

#### Judgment

**This is the most worthwhile TRT-LLM route to invest in for the medium term.**

If we do indeed need to migrate to a standardized paged KV runtime in the future, this route is the recommended main line.
But it should internally be further distinguished:

- **Target form**: `backbone/cp/code2wav` share the same request lifecycle
- **Transitional form**: first bring `backbone+cp` into TRT-LLM, with `code2wav` temporarily attached externally

### Route C: TRT-LLM TensorRT backend, unified `backbone+cp` engine, deep Executor/C++ runtime customization

#### Core idea

- Compile `backbone+cp` into a unified TensorRT engine
- Use TRT-LLM's TensorRT backend / C++ runtime / Executor system
- Do deeper C++-side scheduling customization when necessary

#### Pros

- Highest theoretical performance ceiling
- Closer to TRT-LLM's native TensorRT route
- May yield the best execution efficiency in the long term

#### Cons

- Largest engineering effort
- Requires deeper understanding of TensorRT build, plugins, strongly-typed precision boundaries, and the executor lifecycle
- Business states like `WAIT_TEXT` require adding more runtime logic yourself
- Heaviest debugging chain

#### Judgment

**Technically feasible, but not suitable as the first choice for the current productization stage.**

### Route D: stock `Executor` / stock `generate()` directly hosting the entire generation loop

#### Core idea

- Adapt the current model into an ordinary generation model as much as possible
- Directly use the TRT-LLM high-level API

#### Judgment

**Not recommended.**

There are three reasons:

1. The current business needs `WAIT_TEXT`
2. The current true AR stage is `backbone+cp`, not plain token decode
3. The high-precision boundary of `cp` requires finer control

This route looks like a small engineering effort, but it is very likely to get stuck on the key semantics, ultimately gaining neither the full benefit nor avoiding a rollback.

## 7. Judgment on "Fusing backbone, cp, and code2wav All Into One Graph"

From the graph-building perspective, this is not entirely impossible, but two kinds of "fusion" should be clearly distinguished:

### 7.1 Compute graph fusion

This can be done.

That is:

- `backbone`
- `cp`
- `code2wav`

are all handled within a unified model graph or unified engineering system.

From the operator-expression perspective, the `ConvTranspose1d` in `code2wav` is not a fundamental blocker for this route:

- It can be implemented via an equivalent `ConvTranspose2d` single-high-dimension wrapping
- Or the current export/build-side graph rewriting approach can continue to be used

The real difficulty is still request lifecycle, scheduler, and state management, not whether a single operator can be represented.

### 7.2 Runtime responsibility fusion

Here it is not appropriate to simply answer "should" or "should not"; instead, distinguish by goal:

- If the goal is to **minimize the complexity of the first TRT-LLM onboarding**, `code2wav` can be left out of full inclusion for now
- If the goal is to **minimize the amount of change to the whole system and avoid retaining a second runtime**, then `code2wav` should preferentially share the same request lifecycle with `backbone/cp`

Therefore the more recommended order of judgment is:

- In the first phase, prioritize unifying the request lifecycle and scheduler semantics
- Do not insist that on day one it must be a single static graph, one engine
- But avoid misjudging "keeping `code2wav` permanently in an external second runtime" as the optimal end state

## 8. Judgment on "Will cp Fall Back to bf16 After Being Fused Into TRT-LLM"

It will not automatically fall back to `bf16`, provided the implementation is done correctly.

The following must be clear:

- "Using TRT-LLM" is not equivalent to "unified low precision for the whole graph"
- TensorRT supports strongly-typed precision control
- The TRT-LLM PyTorch backend allows controlling the dtype boundaries of different submodules inside the model

The real risk is not "whether to adopt TRT-LLM," but rather:

- Whether, for convenience, `cp` is also compiled into a unified `bf16` path
- Whether the path incorrectly falls back to `bf16` at boundaries such as `next_embed`, hidden state, cp state, sampling logits

Therefore, once the TRT-LLM route is taken, the following categories of state need to be key validation targets:

- The input hidden of `cp`
- The internal attention / MLP / residual of `cp`
- The KV cache of `cp`
- The bridging tensor of `codec_sum -> next_embed`
- The boundary output with `code2wav`

## 9. Recommended Route

### 9.1 Overall recommendation

The recommendation is to adopt a "two-phase main line + one parallel validation line":

#### Main line phase 1: single-engine mixed precision of the current engine

Goals:

- Prioritize solving the `cp` hallucination problem under `bf16`
- Do not break the current `WAIT_TEXT` and streaming stability
- Keep the current product's iteration cadence

Recommended configuration:

- `backbone=bf16`
- `cp=fp32`
- `code2wav=bf16`
- `io_float=bf16`

#### Main line phase 2: TRT-LLM PyTorch backend PoC

Goals:

- Verify whether the TRT-LLM custom scheduler route can take over the current request lifecycle
- Verify three things:
  - Whether `WAIT_TEXT` can be preserved via the officially supported custom scheduler/request readiness
  - Whether `cp fp32` can be stably preserved in TRT-LLM
  - Whether, compared to the current padded execution, a noticeable benefit can be gained on real workloads
  - Whether `code2wav` must share the same request lifecycle to avoid the complexity of a second runtime

Implementation boundary:

- The outer gateway / websocket / session / segment semantics continue to be preserved
- The scheduler preferentially uses the official TRT-LLM `torch/scheduler` path
- Prioritize evaluating `backbone/cp/code2wav` sharing the same request lifecycle
- If the first implementation's complexity is too high, a transitional variant is acceptable: first bring `backbone+cp` into TRT-LLM, with `code2wav` temporarily attached externally

#### Parallel validation line: stock Executor disproof experiment

A very small experiment can be done to verify whether the stock `Executor` can express `WAIT_TEXT` at low cost.

Expectation:

- Most likely disproven
- The value lies in quickly ruling out the fantasy that "maybe the high-level API is enough"

### 9.2 When to enter formal TRT-LLM migration

The recommendation is to upgrade the TRT-LLM route from PoC to a formal main line only when the following conditions are met:

1. The `cp fp32` path is verified stable in TRT-LLM
2. The `WAIT_TEXT` semantics can be implemented via a custom scheduler/request readiness
3. The state management boundary of `code2wav` has been clarified
4. Under real business stress testing, the throughput or GPU memory efficiency benefit is sufficient to cover the migration cost
5. The code maintenance complexity is not noticeably higher than the current approach

## 10. Milestone Recommendations

### M0: Freeze the current state and establish a metrics baseline

- Solidify the metrics of the current engine under a representative workload
- The metrics should at least include:
  - First-packet latency
  - Throughput
  - Peak GPU memory
  - Hallucination rate / repetition rate
  - Streaming pause-resume correctness

### M1: Land single-engine mixed precision

- Complete the per-segment precision configuration of `backbone/cp/code2wav`
- Use the existing runtime to verify whether `cp=fp32` stably solves the main problem

### M2: TRT-LLM AR PoC

- Enable TRT-LLM `use_python_scheduler`
- Implement `READY_DECODE / WAIT_TEXT` in a custom scheduler
- Verify whether the request readiness gate is sufficient to express "waiting for external text"
- Prioritize evaluating `backbone/cp/code2wav` sharing the same request lifecycle
- If the complexity is too high, fall back to the transitional variant: migrate only `backbone+cp`

### M3: Benefit evaluation

- Compare against the current main line:
  - Actual GPU memory usage
  - Batch utilization
  - Throughput
  - Tail latency
  - Code complexity
  - Whether a second `code2wav` batching/cache runtime still remains

### M4: Whether to switch the main line

- If the benefit is significant and the semantics are stable, plan the migration
- If the benefit is insignificant or the semantic cost is too high, keep the current engine and only absorb the mixed precision results

## 11. Risk List

### Risk 1: TRT-LLM benefits are overestimated

If `cp` is still outside TRT-LLM, the true AR loop is still in the hands of the user runtime, so the benefits of paged KV and continuous batching will be weakened.

### Risk 2: `WAIT_TEXT` semantics intrude too deeply

Although TRT-LLM officially supports custom schedulers, if `WAIT_TEXT` is to be genuinely made a first-class semantic of the runtime, the scheduler, request readiness, and KV lifecycle still need to be designed together.

### Risk 3: Incomplete boundary control for `cp fp32`

If only part of the inputs is preserved as `fp32`, but the intermediate residual / KV / logits paths still fall back to `bf16`, then the situation of "it looks like mixed precision was migrated, but the problem was not actually solved" may occur.

### Risk 4: The one-time migration scope is too large

If the first phase migrates all of the following simultaneously:

- `backbone`
- `cp`
- `code2wav`
- scheduler
- gateway protocol boundary

then localizing problems will be very difficult.

### Risk 5: Keeping `code2wav` external causes long-term entrenched runtime fragmentation

If `backbone+cp` has already entered TRT-LLM, but `code2wav` stays entirely outside long-term:

- The system still needs to maintain a second `code2wav` batch semantics
- The system still needs to maintain a second fixed-window KV/state management
- It may weaken the expected benefit that "the system is noticeably simpler after migration"

## 12. Final Recommendation

The most reasonable technical route for this project at present is not "immediately switching fully to TRT-LLM," but rather:

1. **First solve the mixed precision problem with the current runtime**
2. **Then use the TRT-LLM PyTorch backend to validate the `backbone+cp` AR runtime route**
3. **Focus on validating whether the official TRT-LLM custom scheduler route can take over `WAIT_TEXT`**
4. **Only when `WAIT_TEXT + fp32 cp + paged KV` can all hold simultaneously, and the `code2wav` boundary is clear, then consider promoting TRT-LLM to a formal main line**

If the question is just "where should the product main line be bet right now," my recommendation is:

**Short term, bet on the mixed precision fix of the current engine; medium term, bet on the PoC of the TRT-LLM PyTorch backend + custom scheduler; for the target form, prioritize a unified request lifecycle, and do not recommend betting directly on deep C++ Executor customization at the current stage.**

## References

### In-repo implementation

- `engine/backend/engine_loop.py`
- `engine/backend/executor.py`
- `engine/backend/kv_cache_pool.py`
- `engine/config.py`
- `docs/roadmap.md`
- `docs/dev/architecture/engine_decisions.md`
- `docs/dev/design/mixed_precision_plan.md`

### Official documentation

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
