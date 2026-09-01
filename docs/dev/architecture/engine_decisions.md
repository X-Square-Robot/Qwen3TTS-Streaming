**English** | [中文](engine_decisions.zh-CN.md)

# Engine Architecture Decision: Leaving Triton, Building a Custom TTS Inference Engine

> Date: 2026-04-03  
> Status: Decided — adopt Route 1 (custom engine), keeping future Triton integration as an option

---

## Background

The project originally planned to use Triton Inference Server as the inference framework, orchestrating sub-models (TRT/ORT backends) via a Python BLS to perform streaming TTS inference. During actual development we found that Triton offers limited value for stateful autoregressive models, and the technical route needed to be re-evaluated.

## The Two Routes Evaluated

### Route 1: Custom Inference Engine (Adopted)

Leave Triton entirely and implement the gRPC service, GPU scheduling, and continuous batching ourselves.

| Dimension | Assessment |
|-----------|------------|
| Performance control | **Best** — directly manage the CUDA stream and the CPU-GPU pipeline, no framework middle layer |
| Scheduling granularity | **Best** — can implement first-packet priority, segment-level parallelism, long-text throttling |
| Extensibility | Medium — must build production components such as model loading, health checks, and metrics ourselves |
| Engineering effort | Medium — the core skeleton is already in place (`engine/`) |
| Reliability | Needs maturing — exception recovery, memory leaks, etc. require ongoing polishing |

### Route 2: Rebuild BLS + Triton Dynamic Batching (Rejected)

Spin up 64 BLS instances, each managing one session, and use Triton's dynamic batcher to collect decode requests for unified inference.

| Dimension | Assessment |
|-----------|------------|
| Performance control | **Poor** — the KV cache needs padding along the seq_len dimension, wasting significant compute/VRAM |
| Scheduling granularity | **Poor** — Triton's dynamic batcher aggregates by arrival time and cannot express priority |
| GIL bottleneck | **Severe** — 64 Python BLS instances share the GIL, serializing the CPU preparation work |
| BLS→TRT overhead | **Non-negligible** — the pb_utils + dlpack path takes a noticeable share of the ~4ms per-step budget |
| First-packet priority | **Impossible** — concurrent long-text synthesis preempts GPU resources from urgent first-token requests |

**Core reason for rejection:** Triton dynamic batching is designed for stateless requests (concatenation along the batch dimension) and does not apply to autoregressive decode where the KV cache lengths differ. All mainstream LLM inference engines (vLLM, TRT-LLM, SGLang) build their own continuous batching rather than using Triton's dynamic batcher.

#### Preconditions for Triton Dynamic Batching

For dynamic batching to take effect, all of the following conditions must be met simultaneously, which illustrates its applicability boundary:

1. **The model input's first dimension must be the batch dimension** — Triton concatenates requests along dim-0
2. **`max_batch_size > 0`** — setting it to 0 means the model does not support batching
3. **Explicitly enabled in `config.pbtxt`** — `dynamic_batching { preferred_batch_size: [...] }`
4. **The non-batch dimension shapes of all requests in a batch must be identical** — otherwise padding is required

For autoregressive decode, condition 4 is inherently unmet: each request's KV Cache `seq_len` dimension differs, and Triton does not auto-pad — it only concatenates along the batch dimension. This means either all requests happen to be the same length (unrealistic), or the BLS manually pads them and feeds them in under the illusion of batch=1 (bypassing the point of the dynamic batcher).

## Where Triton Server Truly Adds Value

Triton offers irreplaceable value in the following scenarios:

- **Stateless models**: image classification, embedding, reranking, recommendation CTR — dynamic batching can lift GPU utilization from 10% to 90%
- **Model zoos**: hosting dozens of heterogeneous models in the same cluster with unified operations
- **Enterprise infrastructure**: K8s integration, model version management, A/B testing, Prometheus metrics

For a **single-model, stateful, autoregressive, fine-scheduling-required** TTS scenario, Triton degenerates into a gRPC proxy + process container. In TRT-LLM's Triton backend, Triton is also just a shell, with all scheduling logic living inside the TRT-LLM Executor.

## Batching Strategy Selection

### Dynamic Batching vs Continuous Batching

The core difference between the two is **scheduling granularity**:

- **Dynamic Batching** (request-level scheduling): multiple requests are assembled into a batch and executed together, and **the entire batch must all finish before returning**. After short requests finish, they idle-wait for long ones, so GPU utilization drops later on. Suitable for stateless, single-forward-pass models (classification, embedding).

- **Continuous Batching** (iteration-level scheduling): scheduling is per decode step, and **requests can be added or removed every step**. Completed requests release resources immediately, new requests can join at the next step, and the GPU is always doing useful compute.

### How Continuous Batching Handles Different Shapes

The core mechanisms that make continuous batching work:

1. **The decode phase is naturally aligned** — each request's "current input" is a single token (shape `[1, hidden]`), so the QKV linear layers can batch directly, with no padding needed.

2. **Paged KV Cache** — the KV Cache is allocated in fixed-size pages (like OS virtual memory), and each request indexes non-contiguous physical VRAM through a page table. After a request finishes, its pages are reclaimed immediately, and new requests allocate on demand, so there is no need to reserve max_len of contiguous VRAM.

3. **Special Attention kernel** — standard Attention requires consistent KV lengths within a batch. Continuous batching uses:
   - **PagedAttention**: the kernel accepts `block_tables[batch, max_pages]` and `context_lens[batch]`, fetching KV from non-contiguous addresses via the page table
   - **FlashAttention varlen**: QKV is flattened to `[total_tokens, H, D]` and request boundaries are delimited by `cu_seqlens` (cumulative sequence lengths)

4. **Chunked Prefill** — long prefills are split into fixed-size chunks and executed mixed with decode requests, preventing long-text encoding from blocking in-flight decode requests.

### Why This Project Cannot Use Standard Continuous Batching

The mechanisms above depend on two preconditions, neither of which this project satisfies:

1. **PagedAttention / varlen kernel** — requires a custom CUDA kernel. This project's Attention is already compiled into the TRT engine (ONNX → trtexec → .plan), the compute graph is fixed, and there is no way to swap the Attention kernel at runtime or inject `cu_seqlens` parameters.

2. **Flexible KV Cache memory management** — the TRT engine's KV Cache input shape is `[batch, num_heads, seq_len, head_dim]`, requiring the `seq_len` dimension to be consistent across all requests in a batch. Page-table addressing or the flattened mode is not supported.

**Fundamental constraint**: the TRT engine is a fixed compute graph. Implementing standard continuous batching would require reworking the Attention graph and embedding a PagedAttention kernel — equivalent to rewriting from the ONNX export layer, or migrating to TRT-LLM.

### The Adopted Approach: Padded Iteration-Level Batching

The scheduling granularity is iteration-level (each decode step can add or remove requests), but the KV Cache is aligned by padding (constrained by the TRT engine):

- Requests can join or complete at any step, without waiting for the entire batch to finish
- Each step pads the active requests' KV Cache to the maximum length in the batch
- Completed requests release their KV Cache slot immediately, and new requests can join at the next step

This is a "continuous scheduling + padded execution" hybrid strategy — the optimal compromise under the fixed-graph constraint of TRT.

### Evolution Path

| Stage | Strategy | Notes |
|-------|----------|-------|
| Early validation | batch_size=1 + Multi-Instance | See the note below |
| Production V1 | Padded iteration-level batching | The Dispatcher manages a slot pool, with controllable padding overhead |
| Future optional | Custom PagedAttention kernel | Eliminates padding waste, requires CUDA development |
| Future optional | TRT-LLM migration | Built-in inflight batching, but requires reworking the model build pipeline |

#### Early Validation: The Multi-Instance Strategy

Rather than batching, run multiple model instances, each with batch_size=1 decoding independently:

- **Zero code changes** — pure configuration: under Triton set `instance_group[{count:N}]`, under the custom engine spin up multiple executor threads
- **No shape-alignment problem** — each instance manages its own KV Cache independently, without interference
- **CUDA Stream parallelism** — multiple instances can be assigned to different streams, giving some parallelism among GPU compute cores (subject to available compute headroom)

**Cost**: VRAM × N (each instance holds one engine context + KV Cache). For the ~1.7B talker model, running 2-4 instances on a single GPU (24GB+) is feasible, and is the most cost-effective initial concurrency approach.

## The Adopted Architecture

```
┌──────────────────────────────────────────────────┐
│  Gateway 层 (可插拔)                               │
│                                                   │
│  A) Standalone gRPC Server      B) Triton Backend │
│     engine/gateway/                (未来可选)      │
│     - 开发/生产均可用              - 大规模部署    │
│     - 零外部依赖                  - 模型热更新     │
└──────────────┬────────────────────┬───────────────┘
               │                    │
               ▼                    ▼
┌──────────────────────────────────────────────────┐
│  TTSEngine (核心引擎，不感知外层 serving 框架)     │
│                                                   │
│  公开 API:                                        │
│    synthesize_stream()                            │
│    feed_text() / text_complete() / cancel()       │
│    start() / stop()                               │
└──────────────────────┬───────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Dispatcher     EngineLoop      KVCachePool
   (asyncio)    (GPU thread)     (预分配 slot)
```

### Layer Responsibilities

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| Gateway | `engine/gateway/` | gRPC bidirectional streaming, protocol conversion, connection management |
| Frontend | `engine/frontend/` | Session management, text segmentation (Spliter), segment scheduling |
| Core | `engine/core/` | Pure data structures (EngineRequest/Result/Session), no GPU dependency |
| Backend | `engine/backend/` | GPU thread, KV cache pool, prefill, batch decode, executor |

### Key Design Constraints

1. **Self-contained engine lifecycle** — `start()`/`stop()` can be called by an external caller from any thread and do not assume the engine owns the process (preparing for future Triton integration)

2. **Pure data structures for request/response** — `EngineRequest`/`EngineResult` are pure dataclasses, bound neither to a gRPC stub nor to Triton `pb_utils`

3. **CPU-GPU pipeline** — while the GPU executes step N, the CPU processes step N-1 results and prepares step N+1 inputs

4. **Priority scheduling** — prefill priority `FIRST_SEGMENT > CONTINUATION > PREFETCHED`; the decode batch can be ordered by first-packet urgency

### Long-Text Offline Pre-Split Semantics

In the offline scenario, `presplit` and `driver` in the `Spliter` are not the same conceptual layer:

- **presplit = grouping (group)**: uses full-text visibility to split long text into several fairly natural, close-to-equal-length groups, aiming to increase parallelism during offline synthesis and minimize the risk of any single group hitting `max_seq_len`
- **driver = intra-group segmentation (segment)**: within each group, the driver still decides the real flush timing in coordination with the L1/L2/L3/d thresholds and backend state; the driver is the final arbiter
- **backend segment**: the execution unit actually submitted to the engine; one group can produce multiple backend segments

Therefore, the true hierarchy of the offline path is:

`full text -> presplit groups -> driver flush -> backend segments`

rather than "presplit directly determines the final segment boundaries."

### Execution Model for Very Long Presplit Queues

When long text is `presplit` into far more groups than `max_batch_size`, the system still executes correctly, because both the frontend and the backend apply layered throttling:

- **Frontend throttling**: offline groups are not all submitted at once, only launched up to `max_concurrent_segments`; the remaining groups stay queued waiting for earlier segments to complete
- **Backend throttling**: each decode iteration selects at most `max_batch_size` active segments to enter the batch this round
- **Ordering guarantee**: audio emission order is hierarchically reordered by `group_idx + local_idx`, avoiding "a later sentence of an earlier group being played ahead of it by a later group"

This means the main problem introduced by "a very long presplit" is **queuing and tail latency**, not a correctness failure or the engine being flooded all at once.

### EMA and Offline Long Text

Duration estimation and capacity planning are separate contracts:

- The two-sided duration EMA is updated only from sufficiently large segments
  that reach natural codec EOS.  It is used for text progress and guarded-tail
  estimation.
- A conservative safety ratio derives the text-token capacity.  It never falls
  within a session; overflow and confirmed loop/length failures may only
  tighten it.
- Offline Stage 1 snapshots one immutable capacity plan per planning epoch.
  Every group produced by that plan is opened with the same hard cap. Stage 2
  must not recompute a live cap for an already planned group: doing so used to
  turn groups into tiny residual segments when feedback raised the ratio,
  while feedback in the other direction could not merge boundaries.
- When safety (not duration) tightens, Stage 1 may explicitly replace the plan
  for a wholly unopened packet suffix. It repacks all remaining tokens in that
  cohort together under the smaller cap, while active/flushing groups and
  packet boundaries remain frozen. This prevents both repeated tiny tails and
  an unsafe old plan surviving after an overflow/loop signal.
- True streaming has no ahead-of-time group contract, so a newly opened
  streaming segment may use the latest monotonic safety ratio.  An already-open
  driver always keeps its frozen thresholds.

This keeps split boundaries deterministic and prevents normal short-duration
observations from expanding the safety budget of a long request.

### When to Introduce a Triton Shell

Consider it in the following scenarios:

- Multiple models coexisting (TTS + ASR + LLM on the same GPU)
- Large-scale K8s deployment requiring Triton's readiness/liveness probes
- An operations team that already has Triton cluster management experience
- A need for model version management to do A/B testing

Integration approach: wrap `TTSEngine` as a `TritonPythonModel`, with Triton only sending and receiving requests while all scheduling logic remains inside the engine. Estimated effort: a few hundred lines of adapter code.

## Production Components the Custom Engine Needs to Fill In

| Component | Effort | Priority | Remarks |
|-----------|--------|----------|---------|
| gRPC service + health check | Low | P0 | `grpc.aio` is mature |
| TRT engine loading | Low | P0 | `tensorrt` Python API |
| Prometheus metrics | Low | P1 | `prometheus_client` |
| Graceful shutdown / request drain | Low | P1 | signal handler + drain |
| Exception recovery / session timeout | Medium | P1 | watchdog + timeout reclamation |
| CUDA Graph optimization | Medium | P2 | fixed batch size scenarios |
| Hot reload / canary release | Medium | P3 | can be done later or delegated to Triton |
| Multi-GPU scheduling | High | P3 | 1.7B is bound to a single card, not needed for now |
