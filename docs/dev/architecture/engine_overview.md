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
| **C4** | The model **may not reliably emit EOS** — how often is **checkpoint-dependent**, and the project ships no weights, so this stays a standing constraint the engine must always defend against. Observed range: one internal checkpoint (0601) never finished on ~10–18% of seeds → ran out the full 512 → hallucinated; its retrain (0701) measures 0/100 on the same deterministic seeds (both cp=fp32 and full-bf16 engines). The 512-step cap, VAD gating, and runaway handling exist for whatever checkpoint a user brings | Model checkpoint + sampling, **not an implementation bug** |

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

### 1.4 Code Predictor Unrolled, With In-Graph KV (no-KV until 2026-07-06)
- **What it is**: The CP's 15 stages are unrolled into a single static TRT graph. Since 2026-07-06, each stage forwards only its 1 new token against K/V statically concatenated inside the unrolled graph — every stage's KV length is a compile-time constant, so the graph stays fully static and engine I/O is unchanged. Previously each stage recomputed the growing sequence from scratch (135 token-forwards vs 16). `architecture.md §5`.
- **Why the original no-KV choice (now inverted)**: "Saving the CP's KV is only ~3% of latency, and a GEMM beats a GEMV on utilization" — both true at small batch. At batch 128 the per-stage matmul is an M=128 GEMM either way, and the 9× recompute FLOPs had grown to ~34ms of a ~100ms decode step. The KV rework cut CP GPU time to ~10ms (c128 decode step 119.8→96.3ms, RTF 1.50→1.20); verified 10/10 token-identical vs recompute in fp32 and halluprobe 0/100.
- **Flaw**: **FP16 is broken for the CP** (the stage argmax is sensitive), so only fp32/bf16 work; the large graph compiles slowly.
- **Avoidable**: The unavailability of CP FP16 is numerically intrinsic.

### 1.5 Sub-module Precision (CP fp32 historical; default full-bf16; code2wav fp16 is a low-concurrency opt-in)
- **What it was**: backbone=bf16, CP=fp32, code2wav=bf16, adopted while hunting the 0601-era hallucination.
- **Why at the time**: Under CP bf16, stage near-ties get flipped by rounding (Finding #15: standalone CP bf16 TRT vs ORT mismatched 7/10 random trials; fp32 matched 0/10) — suspected of cascading into hallucination.
- **Key insight (still true)**: **Precision was never the root cause of hallucination** (Findings #17-18) — every precision on the 0601 weights gave ~10-18%, full fp32 was actually worse (17.5%); precision only "swapped in a different batch of bad seeds." The real root cause was the damaged 0601 checkpoint; the 0701 retrain fixed it (0/100 on the same seeds).
- **Current state (2026-07-06)**: With 0701 weights, a **full-bf16 engine (cp=bf16) also measures 0/100** on the same deterministic seed set — the CP bf16 numerical noise (Finding #15 is still a numerical fact) demonstrably does not translate into hallucination on healthy weights. CP fp32 is therefore no longer required; it costs real decode latency (the CP was ~45% of kernel time under fp32).
- **code2wav fp16 (opt-in, NOT default — concurrency-dependent)**: TRT 10.13 on sm120 has no tensor-core bf16 conv kernels (fp16 convs are 2.5-3.4× faster *in isolation*), so `CODE2WAV_PRECISION=fp16` pins only `/code2wav/*` to fp16 (the emitter also pins `/talker_fused/*` back to bf16 so the global `--fp16` cannot let talker/CP kernels float and shift sampling numerics; `build_engines.sh` auto-selects `--precisionConstraints=prefer` via `trt_fused_io_formats.py --emit constraints`, since the c2w Pad/Slice glue has no fp16 kernel and must fall back). It is numerically fine (c2w does not feed back into the talker; halluprobe 0/100 + audio level/spectrum checks). **But the isolated conv win does not survive batching.** The engine's float I/O — including the 17+4 c2w conv/transconv streaming states — is bound at a single `triton_io_float_dtype` (bf16), so an fp16 c2w round-trips those states through a bf16↔fp16 reformat **every decode step**, and that reformat cost scales with batch. Measured 2026-07-08 (RTX 5090, custom-1.7b, b128 profile, engine-grpc), c2w=fp16 vs the full-bf16 baseline **decode step**:

  | concurrency | 1 | 8 | 16 | 32 | 64 | 128 |
  |---|---|---|---|---|---|---|
  | Δ decode | **−20.8%** | **−12.9%** | **−7.8%** | +1.3% | **+6.2%** | **+7.8%** |

  fp16 wins below ~c16 (single-stream −21%) and loses above ~c32 (b128 +8%, RTF 0.525→0.566). **The default is therefore full-bf16** (the batch-128 serving profile is the production target); use `CODE2WAV_PRECISION=fp16` only for single-stream / low-concurrency latency. (Making just the c2w states fp16 to kill the reformat would need per-tensor I/O binding + mixed-dtype runtime state buffers — a larger, unshipped change.)
- See [[mixed_precision_plan]] and `streaming_hallucination.md` for details.

### 1.6 Prefix KV Cache
- **What it is**: A 16-entry LRU that caches the fixed-prefix talker KV; a hit skips prefill. `prefix_cache.py`.
- **Why**: custom_voice-type requests share the same system prompt, saving 10-50ms.
- **Flaw**: Exact token match only; ⚠️**sub-agent inference (not reviewed)**: on a miss it may run TRT twice, and a multi-token suffix hit yields uneven benefit.
- **Avoidable**: The exact-match limitation can be improved with a prefix tree; it's an optimizable item.

### 1.7 CUDA-Graph Decode Replay (since 2026-07-06)
- **What it is**: One CUDA graph captured per (batch bucket, past-len bucket in steps of 64) shape signature, replayed for each fused decode step. `executor.py GraphedFusedDecode`; parity harness: `tools/validation/graph_decode_parity.py`.
- **Why**: The fused decode enqueues ~2700 kernels per step; at batch 128 the CPU enqueue time (~33ms) matches GPU compute, and autoregression prevents cross-step pipelining. Replay collapses the per-step launch cost to ~0.005ms. Measured (all-bf16 b128): c128 decode step 95.7→70.3ms (RTF 1.20→**0.88**, 128-way real-time), c96 54.3ms, c64 38.9ms; +3.5GiB GPU memory.
- **Load-bearing details** (each one was empirically forced):
  - **Dedicated execution context bound to a decode-only optimization profile** (profile 1, emitted by `build_engines.sh` by default). Prefill sharing the context corrupts replays (2026-07-02 finding: audio drift max_abs≈0.26); a second full-profile context costs 7.1GiB which a 32GiB card cannot pay — the decode-only profile's scratch is 1.2GiB.
  - **Persistent flat staging buffers** viewed per-bucket (stable addresses, contiguous); the KV pool gathers straight into staging (`gather_talker_kv_into`), removing the multi-GiB transient batch-KV tensor on the graph path.
  - Bucket padding is masked via `attention_bias` per slot's real length; `codec_sum`/`full_codec` are cloned on the compute stream (a default-stream clone races the async replay).
  - **Cross-profile numerics**: profile 1's kernels are an independent compilation — same class of near-tie sampling shifts as any engine rebuild. Graph-vs-same-profile-eager is bitwise identical (b2/b64/b128); real-data behavior gated by halluprobe (0/100, duration distribution unchanged).
- **Fallbacks**: `ENGINE_CUDA_GRAPH_DECODE=0` disables; staging OOM steps down a 512→384→256→128 ladder; out-of-bucket steps and any replay exception fall back to the eager path (auto-disable after 3 failures).
- **Flaw**: staging + decode-profile scratch cost ~3.5GiB; first hit on a new bucket pays a ~200ms capture; the eager fallback path still allocates its transient batch-KV gather (pre-existing OOM risk at c128 + past>~380, see §5).

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
| Eager decode's transient batch-KV gather allocated B×L×H×past×D per step (up to 7.5GiB at c128×past512 → OOM under low headroom) | Persistent shared gather arena (reuses graph staging when graphs are on; lazily allocated otherwise; latched fallback to transient on OOM) | **Fixed 2026-07-06** — verified: b128×past500 eager step OK with 4.2GiB free |

---

## 6. One-Sentence Summary

> This engine **bears the right things head-on** (custom runtime, three-stage WAIT_TEXT, padded batching, prefix cache, CUDA-graph decode); the pain points are concentrated in two areas of **eliminable accidental complexity**: (a) the dual-path fork of the text segmentation layer + the EMA estimation blind spot, and (b) the abstraction leaks of the protocol layer. Cleaning up these two is the current main line of work.

## 7. Confidence Note

The architectural backbone (C1–C4, three stages, padded batching, CP unroll with in-graph KV, mixed precision, hallucination root cause, dual path, protocol leaks) is cross-verified from multiple sources and is **held with confidence**. For the few items marked ⚠️ "sub-agent inference, not reviewed" (the prefix cache dual TRT pass, several engine_loop race conditions, FULL_TEXT unbounded OOM), it is recommended to spend ten minutes empirically confirming each before acting on them.
