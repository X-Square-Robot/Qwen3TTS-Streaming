**English** | [中文](frontend_segmentation_pipeline.zh-CN.md)

# Frontend Text Segmentation Pipeline Redesign (auto mode / Spliter unification / KV watermark safety net)

> Branch: `main`
> Written: 2026-06-30
> Status: **Design draft** (consensus and decision record from multiple rounds of discussion, pending implementation)
> Scope: `engine/frontend/` (interface / spliter / driver / dispatcher / reorder) + `proto/tts.proto` input modes + KV watermark reporting in `engine/backend/`
> Related: [[realtime_audio]] [[observability_goals]] [[mixed_precision_plan]]

---

## 0. Background and Motivation

### 0.1 The Protocol Leaks Internal Implementation Details to the User

The current protocol lets the **client** control text granularity: `proto/tts.proto:58`'s `InputMode { TOKEN, CLAUSE, LONG_SEGMENT, FULL_TEXT }` + `GroupPolicy { NONE, AUTO }`. But the vast majority of users have no idea whether their text is token-level or long-text-level—this is an **abstraction leak**: it forces the engine-internal decision of "whether the frontend should do offline segmentation" onto users who have neither the information nor the obligation to make that decision.

The four `InputMode` levels actually conflate three **orthogonal** axes:

1. **Input arrival form**: whether the client "has the complete text in hand" or "produces and feeds it incrementally, with more to come" (typically the token stream from an upstream LLM).
2. **Output delivery intent**: whether it wants low first-packet latency and play-while-synthesizing, or a complete audio file.
3. **Internal segmentation granularity / quality knob**: `TOKEN` / `CLAUSE` / `LONG_SEGMENT` are essentially a "first-packet latency ↔ robustness" trade-off. The user is forced to choose along axis 3, but they only know axes 1 and 2.

### 0.2 The Dilemma: Two Ad-Hoc Requirements Don't Fit Into the Original Structure

- **emoji filtering** (`engine/frontend/interface.py:51-56` `_normalize_tts_text` → `strip_emoji`) was added ad-hoc at the time.
- **auto mode** (automatically deciding whether to do offline sentence splitting based on content) is now needed, and it requires integrating the "streaming path" and the "offline path."

These two cross-cutting requirements do not fit into the original "two parallel paths" structure—which is precisely the trigger for this redesign.

### 0.3 The Root Cause (in one sentence)

> The design intent was **a single pipeline** (normalization → global pre-split → the terminal driver that feeds the backend), but **the code grew into a fork**: streaming `feed_tokens` and offline `push_group_tokens`/`set_full_text` are two parallel entry points, each carrying its own independent driver loop, sharing only the driver pool and concurrency counter. Cross-cutting concerns that need session-level state (cross-packet emoji, auto) therefore have nowhere to live.

---

## 1. Core Insights

### 1.1 Boundary Authority

The essential difference between the two paths is **not** "streaming vs. offline," but **who decides the segment boundary**:

- Offline groups: the boundary is decided upstream by `pre_split` (the group tail feeds `END` to force the driver to flush, `spliter.py:435`).
- Streaming: the boundary is decided by the driver itself using local thresholds.

Aside from that, both paths merely **feed `SpliterEvent`s into the same kind of Driver FSM**. `SpliterEvent` already has `START / token / END`, and the semantics of `END` are exactly "close the current segment."

### 1.2 The Scope of Global Optimality = One Packet of Text

The global optimum of `pre_split` can only hold **within the range it can see**. Example: `你好吗？明天天气不错，有没有什么想吃的？`
- The streaming driver has no global view and may split it into `你好吗？明天天气不错，` + `有没有什么想吃的？` (an overflow split at an L2 comma, locally optimal).
- The global optimum should split at L1: `你好吗？` + `明天天气不错，有没有什么想吃的？`.

Corollary (favorable to the design): **the quality of per-packet auto naturally grows with the amount of text the client gives per packet**—the client implicitly expresses the optimization scope via "packet size," with no need to name a mode. In streaming, if tokens arrive one at a time, no stage can find a cross-packet global solution unless it buffers; and buffering = artificially added latency, which is **explicitly vetoed**.

### 1.3 The Ladder Must Be in Text-Token Units—Dictated by the Three-Stage TTS Architecture

This is **not a flaw, it is an architectural necessity**. TTS synthesis has three stages:

1. **prefill**: text tokens are pre-filled into the KV;
2. **text streaming-input stage**: audio is emitted while text tokens are fed in (`WAIT_TEXT` semantics);
3. **flush stage**: no more text, pad EOS/NOP, drain the remaining audio.

Across the three stages, **the only thing the frontend can control is the text tokens** (when to stop feeding / flush); audio steps are a downstream product that the frontend **can only estimate, not directly control**. The estimation relies on an EMA of the text:audio ratio: Chinese ~240ms/char ÷ 80ms/chunk ≈ **3 steps/char**, which then drifts with pauses, speech rate, and the token's character count, hence an EMA sliding estimate (`spliter.py:583 update_ratio`, initial value 5.0, clamp `[2,10]`).

From this:

- The driver's `force_split_at` triggers on `_token_count >= force_split_at` (`driver.py:178`), which can only be a **text-token count**—this is the control plane itself.
- The real constraint (KV overflow) lives in the **audio-step** dimension, and the conversion depends entirely on `ema_ratio`.
- The entire `t1/t2/t3/force` ladder is positioned by the same `ema_ratio` → **if the ratio is wrong, all four rungs shift together, and none of them can catch the fall**.

> The ladder's "gradient" only guards against changes in punctuation position (where the comma is, whether there's an L1); it **does not guard against a mis-estimated ratio**. A ratio error requires replacing "estimation" with "measurement"—namely the KV watermark of §5.3. But note: the watermark can only improve the flush-timing decision **during decode**; Stage 1 packing happens **before** decode, where no watermark is available, so it **inevitably still relies on EMA**.

---

## 2. Protocol-Layer Changes

| Item | Current | Target |
|---|---|---|
| **Add auto** | — | `INPUT_MODE_UNSPECIFIED=0` is given the semantics of "the engine decides automatically," **as the default for the streaming RPC**—novice users don't know the granularity differences, and auto is the right choice for them |
| `InputMode` streaming trio | `TOKEN/CLAUSE/LONG_SEGMENT` must be chosen by the user | **Kept as a first-class entry point**—advanced users understand the details and can choose the appropriate mode for their scenario, so explicit control should be preserved for them (not overridden, not deprecated) |
| `FULL_TEXT` | one level | **Kept**—it expresses the one axis that cannot be inferred: **sealed vs. open** (whether appending is still allowed) |
| `GroupPolicy` | NONE/AUTO | Its semantics fold into each mode's internal behavior once Stage 1 is unified |

> Design principle: **auto is the default (covering the majority of users who don't know the granularity differences), but does not strip advanced users of their explicit choice.** auto turns the "which level to take" decision from "the user must understand" into "the user may not need to understand," rather than "the user is not allowed to choose."

The top-level intent is in fact **already** expressed by the RPC: `SynthesizeOnce` forces `FULL_TEXT`, and `SynthesizeStream` defaults to streaming (`grpc_server.py:555,591`). What this iteration adds is the ability to "decide granularity automatically" inside the streaming path, as the default; the explicit trio remains available.

---

## 3. Three-Stage Pipeline Architecture

```
Stage 0  Stateful filter chain (cross-packet emoji healing, whitespace normalization)
Stage 1  L1≫L2≫L3 hierarchical packing (optimistic capacity = EMA-mean cap) → per-segment threshold profile
Stage 2  Single _drive_events core
          ├─ Ladder trigger: text count (when no watermark) → KV watermark tier crossing (root fix)
          ├─ Active split + tail carry-over (overflow_token_ids), zero sentence loss
          ├─ Single _pending queue + max_concurrent (bounded by KV pool ceiling)
          └─ SegmentAction (token_text incl. pad) → text player
Authority: entirely in Stage 2, Stage 1 only provides global boundary suggestions
```

### 3.1 Stage 0: Stateful Filter Chain

**Confirmed bug (background investigation)**: `strip_emoji` (`engine/text_normalization.py`) is a character-by-character scanner with no cross-packet state and no lookahead; TOKEN/LONG_SEGMENT normalize each packet independently (`interface.py:198`). A multi-codepoint emoji whose codepoints are split across two packets slips through:

- keycap `1️⃣`, ZWJ sequences `👨‍👩‍👧`, skin-tone modifiers `👋🏻`, flags `🇺🇸` are all affected;
- The ugliest case: the keycap's base `1` is not recognized as an emoji, so `"hello1"`+`"️⃣world"` → `"hello1world"`, and `1` gets read out as body text;
- `FULL_TEXT` is immune (it buffers first, then normalizes as a whole, `interface.py:205,248`);
- Existing tests only test whole strings (`tests/unit/frontend/test_frontend_interface.py:22`), with no split-packet cases.

**Solution**: Stage 0 is a **session-stateful** filter (not a pure function). It only holds back, at the **packet tail**, a "possible half-emoji tail" (a dangling ZWJ, an isolated regional indicator, a base+VS, etc.), releasing everything else immediately; the next packet concatenates it and re-judges, and `mark_input_complete` flushes the residual tail. **Only when a packet boundary happens to land in the middle of an emoji is there a one-packet delay, of at most a few codepoints**—which is a different thing entirely from the vetoed "accumulate a window and measure its length."

emoji and auto are **two symptoms of the same underlying illness**: a stage that should have session state was written as a stateless per-packet function.

### 3.2 Stage 1: Hierarchical Packing (Offline Segmentation Algorithm)

See [§4](#4-offline-segmentation-algorithm-hierarchical-packing). The product = **packed groups of L1 units + one threshold profile per group** (not a hard END event stream—see §5.2 for the correction to the earlier approach).

### 3.3 Stage 2: Unified Driving Core

- Use a single `_drive_events` to replace the two duplicated loops `feed_tokens` (`spliter.py:451`) and `_drive_group` (`spliter.py:389`).
- `_token_buffer` (streaming) + `_presplit_groups` (offline) + `_presplit_thresholds` + `_next_group_idx` → collapse into a **single `_pending` queue**. This is the highest-value and also the most dangerous step.
- **Authority stays in Stage 2 throughout**: the driver always retains the complete ladder and the final say on "does it fit"; Stage 1 only provides global boundary **suggestions**.

---

## 4. Offline Segmentation Algorithm: Hierarchical Packing

### 4.1 The True Culprit of Fragmentation: The Driver Re-Splits Interior L1s

`compute_thresholds` (`driver.py:51-84`): `cap = remaining_kv / ema_ratio`, `t1=0.70cap, t2=0.80cap, t3=0.90cap, force=cap`. The driver's `_meets_split_threshold` (`driver.py:160`) is "split at the first point that meets the threshold," so an L1 is split as soon as `tc≥t1=0.7cap`.

Key point: `_drive_group` feeds offline groups through an ordinary driver (`spliter.py:404`), which will FLUSH at the **first L1** inside the group where `tc≥t1` and return the remaining tokens to the queue (`spliter.py:427-445`). **So even if `pre_split` packs it into a long segment, the driver re-splits it back at 0.7cap** → fragmentation.

### 4.2 The Goal: Lexical-Priority Packing, Not Equal-Weight Multi-Dimensional Packing

> **Not** equal-weight multi-dimensional packing of L1/L2/L3 (equal weight would over-split at commas and ruin the prosody); **but** the hierarchical priority `L1 ≫ L2 ≫ L3 ≫ hard split`.

- **Main axis**: within the capacity, split at the **latest L1** (prefer L1 even if the box isn't full; don't split a comma just to squeeze in two more characters).
- **Fallback**: only when **a single L1 unit itself exceeds capacity** does it fall back to splitting at the latest L2 → L3 → hard split.
- `pre_split` must be **L2/L3-aware** (it needs to know where the fallback split points land), but it uses priority rather than a weighted objective.
- `pre_split`'s hierarchy and the driver's `_meets_split_threshold` ladder are **two views of the same hierarchy**: pre_split does global latest-fit, the driver does local first-past-threshold. Once unified, they should share a single tier definition and `_make_thresholds` to avoid two sources of truth.

### 4.3 The Coupling of Packing Capacity and Concurrency (Concern #1)

- Long text → many L1 units → even packing maximally by `cap`, the box count is still ≥ `max_concurrent`, so concurrency stays saturated—**no conflict**.
- Only medium lengths (box count < `max_concurrent`, and the GPU is idle) present a trade-off.
- Convergence point: **packing target capacity = `min(cap, cap / target parallelism)`**. Quality first → `cap` (longest); latency/utilization first → shrink it to make more boxes to saturate concurrency. `max_concurrent` (`spliter.py:132`) governs "how many boxes run simultaneously" (Stage 2 scheduling), while box capacity governs "how big each box is" (Stage 1 quality).
- **Constraint**: the KV ceiling is fixed at 512/slot (`kv_cache_pool.py:43`), and concurrent segments share the slots of the KV pool → `max_concurrent` is in effect capped by the pool capacity ceiling; the "target parallelism" must align with how many slots the pool can open.

---

## 5. Fit Authority and the KV-Watermark Root Fix

### 5.1 Why We Cannot "Turn Off the Ladder with an Obey Mode"

If the driver is set to pure obey (t1/t2/t3 raised to infinity, leaving only force), then the fit decision forks from Stage 2 to Stage 1's **prior estimate**. But `force` is also merely an estimate computed from an `ema_ratio` snapshot; when the estimate is too optimistic, a long segment cannot be solved, the KV runs out, and the tail is dropped → **massive sentence loss**. So the ladder (the adaptive fit guarantee) **must always stay in Stage 2**.

### 5.2 The Correct Relationship: Stage 1 Only "Raises the Driver's Preferred Split Point"

> The driver always retains the complete ladder and the final authority (no fork); Stage 1 uses its global view to push this segment's preferred split point (`t1`) back to "the last safe L1 within capacity," while **preserving the KV-safety meaning of `t2/t3/force`**.

The mechanism is nearly free: `_create_driver(thresholds)` (`spliter.py:205`) already receives thresholds per segment. Stage 1 computes a custom `SplitThresholds` for each packed segment.

| Situation | Who splits, and where |
|---|---|
| Estimate is accurate | The driver runs to the raised `t1` (the late L1 chosen by Stage 1) and splits cleanly → long segment, coherent prosody |
| Estimate too optimistic, overflow looms | The ladder **automatically takes over again**: hits L2/L3/force before the chosen point and splits early (a slightly dirtier boundary, but **zero sentence loss**) |

> Correction note: the earlier approach that said "Stage 1 inserts hard interior ENDs and the driver is pure obey" is wrong. The correct form is "packed groups + a threshold profile per group that raises t1, with the driver flushing naturally at the late L1 on its own"—this is closer to the current `_drive_group` and is a smaller increment.

### 5.3 The Root Fix: The KV Watermark Connects the Ladder to the Correct Unit

**Conclusion of background investigation**:

| Quantity | Verdict | Evidence |
|---|---|---|
| Intra-segment KV watermark | **AVAILABLE-AND-CHEAP** | `slot.past_len` (used) + `kv_pool.max_seq_len` (=512) are in hand every step (`engine_loop.py:1020,1013`); `ResultType` can add `KV_WATERMARK`, and the `_send_result` channel already exists (`engine_loop.py:1434`) |
| Overflow truncation point | **AVAILABLE-BUT-NEEDS-PLUMBING** | overflow triggers at `engine_loop.py:1036`; but `text_tokens` reports the **fed-in count**, not the rendered count, and the `text_idx`→original text-token mapping is not maintained |

**Core insight: the watermark swallows the hard-bone problem of the truncation point.** With the intra-segment watermark, we can **actively split before overflow occurs**; after the active split, the un-rendered tail goes through the **normal segment-boundary carry-over path**, and the backend already has a carry-over mechanism `group.overflow_token_ids` (`engine_loop.py:430-437,395-403`) that automatically prepends the unconsumed tokens to the next segment. So a precise truncation point is no longer needed.

**Ladder trigger rework** (keep the prosody hierarchy, swap the signal source):

```
The L1→L2→L3→force ladder is kept; the trigger changes from "text count vs. EMA threshold" to "real KV watermark crossing tiers":
  watermark crosses 0.80 → driver is willing to split at L2
  watermark crosses 0.90 → willing to split at L3
  hits max_seq → hard EOS (tail carried over, not dropped)
```

The `KV_WATERMARK` event is emitted on **threshold crossings** (not per step), mapping to a tier upgrade in the driver. This way the prosody hierarchy is preserved entirely, but what feeds it is **real occupancy** rather than an estimate. EMA is downgraded from a "safety lifeline" to "merely deciding how large Stage 1's optimistic packing target is."

**The watermark does not change the control plane, it only upgrades signal quality** (echoing §1.3): the frontend can still only control text tokens, and the watermark response is still "flush at the next text-token boundary (punctuation)." The watermark swaps the **flush-timing decision** from an EMA estimate to a measurement, but Stage 1 packing happens before decode where no watermark is available, so it **still relies on EMA**.

**The flush stage itself must reserve KV headroom**: the third stage (pad EOS/NOP + drain remaining audio) also consumes steps. If flush is not triggered until the watermark reaches 0.95, the flush itself may overflow during the padding process. Therefore the hard-split watermark tier must **leave headroom for the flush tail** (e.g. set the hard-split threshold to 0.90 rather than right up against max_seq).

### 5.4 Final Verdict on the Two Forks

- **Packing basis**: → **optimistic packing (EMA-mean cap) + watermark backstop**. The watermark is a real-time safety net, so we can pack aggressively for maximum quality; when too optimistic, the watermark triggers a clean active split before overflow. The pessimistic `force` floor is downgraded to a coarse backstop for "the watermark responded too late."
- **Whether to adopt the watermark**: → **yes, as the root fix**. It is cheap and swallows the truncation-point plumbing.

---

## 6. EMA and Circuit-Breaker Layering (Greatly Simplified After the Watermark Root Fix)

Current fragile points: `overflow` is just a bool (it doesn't report how much was dropped); `ema_overflow_alpha=0.5` lets a single outlier segment yank the global EMA hard (potentially crushing subsequent segments); `clamp[2,10]` (`spliter.py:605`)—when the real ratio > 10, the EMA saturates, permanently underestimates, drops repeatedly, and never converges.

After the watermark root fix, the circuit breaker is downgraded from "essential" to "backup":

1. **Intra-segment watermark tier-crossing active split** (root fix, new)—a real-time ladder in the audio-step dimension.
2. **Un-rendered tail carry-over**: the watermark/overflow-triggered EOS pushes the `pending` tail into `group.overflow_token_ids` (reusing the existing path)—**the only backend change still needed**, requiring no precise truncation point.
3. **Distinguishing outliers from systemic drift**: a single pathological segment (a long digit string, model-lengthened repeated characters) only triggers conservative re-splitting of that segment; only systemic drift moves the global EMA, avoiding a single outlier shattering the whole session.
4. Loosening the EMA `clamp` / saturation detection: to be optimized slowly as quality tuning, **no longer load-bearing**, not blocking safety.

---

## 7. Text Player (Concern #2)

Requirement: while emitting output audio, notify the client "which token (including pad tokens) this segment was synthesized from."

Current state: `ActionResult` carries the token, and `SegmentAction.token_text` is carried all the way to the output (`interface.py:292,322`); pads are the driver's `PAD_TEXT_EOS/NOP` (`driver.py:116-126`). After unification, **Stage 2 (`_drive_events`) is the sole producer of SegmentAction**, so "which token synthesized this audio" is naturally its output metadata—free structurally.

Two constraints:
- (a) In the refactor, don't drop `token_text` during routing → assert it in golden tests;
- (b) **After obey/packing, segments are longer and flushes fewer → pad/boundary events become correspondingly fewer**, so the player UI must be able to display "one segment corresponds to multiple L1 units." **This must be confirmed with the current player** as to whether it draws under the "one segment = one L1 unit" assumption.

---

## 8. De-risk Conclusions (Verified, the Refactor Can Proceed with Confidence)

- **The reorder path is unaffected**: `AudioReorder` (`reorder.py`) uses the coordinate system `(group_idx, local_idx)+group_final`; the dispatcher (`dispatcher.py:74-76`) already maps streaming segments as "each segment forms its own group (local=0, final=True)." The `_pending` collapse **requires no changes to reorder**, and can even delete the `group_idx=-1` sentinel + the three-way remapping. Step 2 only needs to keep the contract: pre_split boundary → new group (local reset to zero); a driver's own FLUSH → local++ within the group; a segment that exhausts the group's tokens → group_final=True (i.e. what each of the two paths does today).
- **The dual scheduling-priority signals are already covered**: `local_idx>0` (`dispatcher.py:176`) is a `CONTINUATION` priority signal (not sorting); immediately after, lines 178-181 have the streaming-specific parallelism signal `segment_idx-1 in _flushing`. Keeping the above mapping, both reorder and priority need **zero changes**.
- **The EMA adaptive channel already exists**: `update_ratio` (`spliter.py:583`) + `_make_thresholds` recomputes per group with the latest EMA (`spliter.py:402`). As long as both Stage 1 packing and the driver ladder go through `_make_thresholds`, they share the same source and do not fork.

---

## 9. Step-by-Step Rollout Plan (Behavior-Preserving, Tests First)

- **Step 0 freeze**: golden tests pin the current `SegmentAction` sequence—(a) the signature global-optimum case `你好吗？明天天气不错，有没有什么想吃的？` (suggested to go into `tools/repro/`), (b) a pure token stream, (c) `FULL_TEXT`, (d) full-concurrency backpressure, (e) **cross-packet emoji** (which also pins the §3.1 bug on record).
- **Step 1**: change `pre_split` to produce an internal representation (event stream / packed groups + threshold profile), change `set_full_text`/`push_group_tokens` to consume it, **with unchanged behavior**.
- **Step 2 (risk point)**: introduce `_drive_events` to unify the two driving loops; collapse `_token_buffer`+`_presplit_groups` into `_pending`; delete the `-1` sentinel. Golden tests stand guard.
- **Step 3**: introduce auto = per-packet L1-only `pre_split` → `_drive_events`; give `UNSPECIFIED=0` the auto semantics and make it the streaming default.
- **Step 4**: make auto the streaming default; **keep `TOKEN/CLAUSE/LONG_SEGMENT` as first-class explicit entry points** (advanced-user scenario selection, not deprecated); keep `FULL_TEXT` separately.
- **Step 5 (algorithm + safety)**: replace "split on sight of L1" with hierarchical packing; introduce the `KV_WATERMARK` event + the ladder watermark trigger + tail carry-over; make the Stage 0 filter chain stateful and fix cross-packet emoji.

---

## 10. Items to Confirm / Open Questions

1. **backend `KV_WATERMARK` event**: add `ResultType.KV_WATERMARK`, emit it on threshold crossings in `_process_step_output()` (`engine_loop.py:1108+`, after `slot.past_len += 1`); add an elif branch to consume it in the frontend `_consume_results` (`interface.py:263+`).
2. **`overflow_token_ids` carry-over extension**: the watermark/overflow-triggered EOS pushes the un-rendered `pending` tail into `group.overflow_token_ids` rather than discarding it.
3. **driver FSM rework**: the ladder trigger changes from a token-count threshold to watermark tier-crossing; exactly how the FSM rules (`driver.py:226 _build_fsm`) change is **for the next round of discussion**.
4. **text player** confirmation of the "long segment = multiple L1 units" display assumption.
5. **Recalibration of per-packet auto's `min_tokens_l1`**: the original value was tuned for whole-text offline; whether it should be relaxed under per-packet streaming (affecting first-segment latency) needs calibration against real traffic.
6. The alignment formula for **target parallelism ↔ KV pool slot ceiling**.

---

## 11. Key File Index

| Area | File |
|---|---|
| Input entry / normalization / result consumption | `engine/frontend/interface.py` (`push_text_input:193`, `_normalize_tts_text:51`, `_consume_results:263`, SEGMENT_END:330) |
| Segmentation orchestration | `engine/frontend/spliter/spliter.py` (`pre_split:274`, `set_full_text:337`, `push_group_tokens:363`, `_drive_group:389`, `feed_tokens:451`, `update_ratio:583`) |
| Driver FSM / thresholds | `engine/frontend/spliter/driver.py` (`compute_thresholds:51`, `_meets_split_threshold:160`, `_normal_overflow:177`, `_build_fsm:226`) |
| Audio reordering | `engine/frontend/spliter/reorder.py` |
| Dispatch / priority | `engine/frontend/dispatcher.py` (coordinate mapping:74, priority:167) |
| emoji filtering | `engine/text_normalization.py` |
| backend decode / KV | `engine/backend/engine_loop.py` (watermark:1020, overflow:1036, SEGMENT_END:1321, `_send_result:1434`), `engine/backend/kv_cache_pool.py:43,243` |
| Protocol | `proto/tts.proto:58` (InputMode/GroupPolicy) |
