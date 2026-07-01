**English** | [中文](engine_overview.zh-CN.md)

# Engine Design Panorama (Constraints → Design → Flaws → Avoidability)

> Written: 2026-06-30
> Status: **Index/synthesis** — this is the entry map for understanding the whole engine; it links to the detailed documents rather than repeating their content
> Purpose: **Prevent repeated investigation**. Later, to understand "why a design works this way, where its flaws are, and whether they can be avoided," read this first, then follow the links deeper
> Related: [[engine_decisions]] [[decode_fsm]] [[mixed_precision_plan]] [[trt_llm_runtime_route_report]] [[frontend_segmentation_pipeline]] [[observability_goals]] [[observability_tiers]] [[realtime_audio]] [[vad_design_goals]] · investigation reports `docs/dev/investigation/streaming_hallucination.md`, `code2wav_state_size.md`

---

## 0. The Source of Everything: Four Hard Constraints (C1–C4)

Every design in the whole engine is a downstream product of these four constraints. Every "why" points back here.

| # | Constraint | Nature |
|---|---|---|
| **C1** | The model is **autoregressive**, with a **fixed KV budget per segment** (a hard cap of 512 steps/slot) | Hardware/model, cannot be eliminated |
| **C2** | Text **arrives asynchronously** (fed from an upstream LLM token stream); decode must be able to "wait for text" (WAIT_TEXT) | Business semantics, cannot be eliminated |
| **C3** | The frontend **can only control text tokens**; audio steps are a downstream product and can only be estimated with an EMA (~3 steps per Chinese character) | Caused by the three-stage architecture, cannot be eliminated |
| **C4** | The model **does not reliably emit EOS** — certain seed/sampling combinations never finish → run out the full 512 → hallucinate (~10–18%) | Model + sampling, **not an implementation bug** |

> In one sentence: this engine does **real-time streaming TTS on an autoregressive model that will overflow, may never finish, can only be controlled indirectly, and must work while waiting for text**. Nearly all of the complexity comes from bearing these four points head-on.

---

## 1. Model and Runtime Layer

### 1.1 Custom Engine Instead of Triton
- **What it is**: Drop Triton as the main scheduler and build a custom gateway + continuous batching + session flow control.
- **Why**: Triton's `dynamic_batching` is designed for stateless models; autoregressive decode has a different KV length per request and needs the WAIT_TEXT pause/resume of C2, which Triton cannot provide.
- **Flaw**: All operational infrastructure (health checks/loading/metrics/graceful shutdown) is built in-house.
- **Avoidable**: No, it's a correct trade-off.
- See [[engine_decisions]] for details.

### 1.2 Three-Stage Decode + WAIT_TEXT (Core of C1+C2)
- **What it is**: prefill → streaming text input (emit audio while feeding; with no text, `WAIT_TEXT` suspends and preserves KV) → flush (inject EOS pad, drain). `engine_loop.py:1187-1201`.
- **Why**: Streaming TTS must be able to hold the KV in place and wait when the text stream stalls, rather than pad or re-prefill.
- **Flaw**: ① pad tolerance is measured at **only 1 token**, so an upstream stall of pad≥2 produces a perceptible pause; ② **no EOS guarantee** (C4), so flush relies on a silence-abort heuristic hard cut, with cliff-style hard-coded thresholds (`engine_loop.py:1280-1293`).
- **Avoidable**: The three stages are unavoidable (C2); pad/EOS are model capabilities, and engineering can only mitigate.
- See [[decode_fsm]] for details.

### 1.3 Padded Continuous Batching + MLFQ Scheduling
- **What it is**: Iteration-level scheduling, with KV aligned by **padding to the batch max length + mask** (**not paged KV**); MLFQ-style priority (new segments protect TTFB, old segments are demoted to avoid starvation). `scheduler.py`.
- **Why**: A compiled TRT engine has a fixed `seq_len`; paged/varlen requires kernel-level changes, and the .plan cannot be modified.
- **Flaw**: Mixing long and short requests in a batch wastes padding (~10%); no preemption.
- **Avoidable**: Possible but costly — it would require switching to the TRT-LLM PyTorch backend, assessed as a **mid-term route, not now** ([[trt_llm_runtime_route_report]]).

### 1.4 Code Predictor Unrolled, No KV
- **What it is**: The CP's 15 steps are unrolled into a single static TRT graph, recomputing from scratch each step. `architecture.md §5`.
- **Why**: Saving the CP's KV only accounts for ~3% of latency, but unrolling into a GEMM has higher GPU utilization than a GEMV.
- **Flaw**: **FP16 is broken** (the stage argmax is sensitive), so only fp32/bf16 work; the large graph compiles slowly.
- **Avoidable**: The unavailability of FP16 is numerically intrinsic.

### 1.5 Mixed Precision bf16/fp32/bf16
- **What it is**: backbone=bf16, CP=fp32, code2wav=bf16.
- **Why**: Under CP bf16, stage2's near-ties get flipped by rounding → cascading hallucination (Finding #15).
- **Flaw (key insight)**: **Precision is not the root cause of hallucination** (Findings #17-18) — every precision at the same seed gives ~10-18%, and full fp32 is actually worse at 17.5%. CP→fp32 is a numerical necessity, not a sufficient fix for hallucination; it merely "swaps in a different batch of bad seeds."
- **Avoidable**: CP fp32 must stay; do not expect it to cure hallucination.
- See [[mixed_precision_plan]] and `streaming_hallucination.md` for details.

### 1.6 Prefix KV Cache
- **What it is**: A 16-entry LRU that caches the fixed-prefix talker KV; a hit skips prefill. `prefix_cache.py`.
- **Why**: custom_voice-type requests share the same system prompt, saving 10-50ms.
- **Flaw**: Exact token match only; ⚠️**sub-agent inference (not reviewed)**: on a miss it may run TRT twice, and a multi-token suffix hit yields uneven benefit.
- **Avoidable**: The exact-match limitation can be improved with a prefix tree; it's an optimizable item.

---

## 2. Text Segmentation / Flow-Control Layer (Redesign in Progress, see [[frontend_segmentation_pipeline]])

### 2.1 Text Tokens as the Sole Control Surface + EMA (C3)
- **What it is**: The frontend can only decide when to stop feeding/flush; audio cost = text tokens × the EMA ratio; the EMA is updated at SEGMENT_END, clamped to [2,10], and uses α=0.5 on overflow. `spliter.py:583`.
- **Flaw**: ① the EMA is an estimate, and when it's wrong the entire threshold ladder shifts together; ② **the clamp saturates at 10**, so a true ratio > 10 is always underestimated and never converges; ③ **feedback lag**, so 1-2 segments after an overflow may still overflow.
- **Avoidable**: Can be greatly mitigated — a **KV watermark root fix** (replace the flush-timing judgment with a real measurement); but inter-segment packing happens before decode with no watermark, so it **still relies on the EMA** (a residual hard core of C3).

### 2.2 Three-Tier Ladder FSM + Spliter Dual Path
- **What it is**: The driver FSM cuts reactively using the three-tier L1/L2/L3+force thresholds; the spliter has two paths, streaming `feed_tokens` and offline `pre_split`. `driver.py`, `spliter.py`.
- **Flaw**: ① **fragmentation** — the driver cuts at the first L1 at 0.7cap; ② **each of the two paths carries its own drive loop + backpressure queue**, so it's a fork rather than a pipeline, and the emoji/auto cross-cutting requirements have nowhere to live.
- **Avoidable**: **Yes**, pure accidental complexity — unify `_drive_events` + a single `_pending` + hierarchical packing + auto.

### 2.3 AudioReorder Hierarchical Reordering
- **What it is**: A two-level `(group_idx, local_idx)` coordinate; when concurrent segments finish out of order, they are reordered into text order. `reorder.py`.
- **Flaw**: **No stall timeout** — if the backend of some segment never sends SEGMENT_END, subsequent audio is stuck forever.
- **Avoidable**: Yes, add a timeout / health check.

---

## 3. Output Layer (All C4 Cleanup)

VAD output gating (trimming leading/trailing hallucinated silence), isochronous audio streaming (padding silence for the WebRTC jitter buffer), and 14-stage observability.
- **Why**: C4 hallucination cannot be cured at the model layer, so it is patched after the fact at the output end (VAD ~0.1-0.2ms, transparent to the engine).
- **Common flaw**: **None of them reduces TTFT** — the trimmed/padded parts were still synthesized first. They improve experience (clipping, noise), not latency.
- See [[vad_design_goals]], [[realtime_audio]], and [[observability_goals]] for details.

---

## 4. Protocol / Client Layer

- **What it is**: `SynthesizeOnce` (unary, forcing FULL_TEXT) + `SynthesizeStream` (bidirectional); InputMode/GroupPolicy exposed to the client; metadata uses a string map; gRPC+WS share a transport-agnostic core. `proto/tts.proto`, `engine/gateway/`.
- **Flaw (a string of abstraction leaks)**: ① **InputMode leaks the internal segmentation granularity to the user**; ② `ref_audio` exposes the engine-internal `c2w`/`ref_codec` conditions; ③ **`SynthesizeOnce` silently overrides** the client's InputMode → FULL_TEXT; ④ **input completeness is transport-scoped**, so the engine cannot distinguish "the stream is still open and waiting" vs "the stream is closed and should finish"; ⑤ VAD parameters have a **dual representation of fields + a config dict**; ⑥ the timing accumulator is stuffed into the `timing.extra` protocol field (breaking serialization); ⑦ the metadata schema is in comments, with no version negotiation.
- **Avoidable**: Mostly yes (accidental complexity). An auto default + a preserved explicit tier solves ①; the rest is protocol hygiene that can be consolidated. Only ④ is determined by C2 (completeness must be signaled explicitly) and cannot be fully eliminated.

---

## 5. Core Conclusion: Hard Constraints vs Accidental Complexity

### 🔴 Unavoidable (Intrinsic to C1–C4, Can Only Be Mitigated)
| Flaw | Root | What Can Be Done |
|---|---|---|
| Hallucination ~10-18% | C4 model + sampling | VAD cleanup, EOS temperature tuning, forced truncation; **the real fix is training** |
| KV 512 overflow risk | C1 hard budget | Packing, watermark early cut, tail carry-over (no loss) |
| Only text is controllable, audio is estimated | C3 three-stage | The watermark replaces the intra-segment estimate with a measurement; inter-segment packing still relies on the EMA |
| Completeness must be signaled explicitly | C2 async text | The protocol clarifies the signal; it cannot be removed |
| pad≤1 token / no EOS guarantee | C4 model | Silence heuristic; the real fix is training |

### 🟢 Avoidable (Accidental Complexity, Pending Cleanup = What We Are Solving)
| Flaw | Fix | Status |
|---|---|---|
| Spliter dual-path fork | Unify `_drive_events` + a single queue | Decided in [[frontend_segmentation_pipeline]] |
| Segmentation fragmentation | Hierarchical packing + raise t1 | Decided |
| EMA misestimation drops sentences | KV watermark root fix | Decided |
| Emoji dropped across packets | Stage 0 stateful filter | Decided |
| InputMode leak | auto default + preserved explicit tier | Decided |
| Reorder has no timeout | Add a timeout | To do |
| Protocol leaks (c2w/timing accumulator/dual VAD/no versioning) | Protocol hygiene consolidation | To do |
| EMA clamp saturation, overflow α yanks the global | Distinguish outlier / systematic drift, loosen the clamp | To do |

---

## 6. One-Sentence Summary

> This engine **bears the right things head-on** (custom runtime, three-stage WAIT_TEXT, padded batching, CP fp32, prefix cache); the pain points are concentrated in two areas of **eliminable accidental complexity**: (a) the dual-path fork of the text segmentation layer + the EMA estimation blind spot, and (b) the abstraction leaks of the protocol layer. Cleaning up these two is the current main line of work.

## 7. Confidence Note

The architectural backbone (C1–C4, three stages, padded batching, CP without KV, mixed precision, hallucination root cause, dual path, protocol leaks) is cross-verified from multiple sources and is **held with confidence**. For the few items marked ⚠️ "sub-agent inference, not reviewed" (the prefix cache dual TRT pass, several engine_loop race conditions, FULL_TEXT unbounded OOM), it is recommended to spend ten minutes empirically confirming each before acting on them.
