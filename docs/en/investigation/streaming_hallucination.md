# Streaming hallucination investigation summary

## Scope

This document summarizes the investigation context for the streaming/sample hallucination issue in the dev branch.

Focus of investigation:
- Compare our local engine chain vs official local model behavior
- Determine whether the issue is caused by:
  1. official streaming+sample parameters being unstable on long segments, or
  2. our chain computing wrong rollout states / wrong distributions

## High-level findings so far

### 1. Official repo high-level streaming path is itself unstable on long segments

Using local official model (`Qwen3TTSForConditionalGeneration.generate`) with:
- `non_streaming_mode=False`
- `do_sample=True`
- `subtalker_dosample=True`
- `top_k=50`
- `top_p=1.0`
- `temperature=0.9`
- `repetition_penalty=1.05`

Observed on representative segments:
- short segment: no EOS (`eos_step = -1`)
- medium segment: no EOS
- long 4a segment: no EOS

This is evidence that the official repo's current streaming+sample path is not naturally stable / self-terminating on long segments.

Post-fix recheck on the real 4a long segment (`LONG_TEXT`) using
`scripts/python/official_vs_manual_rollout.py`:
- `max_steps=256`, `seed=1234`: `trailing_len=80`, `official_len=255`, `manual_len=256`, `first_divergence=-1`
- `max_steps=256`, `seed=2025`: `trailing_len=80`, `official_len=255`, `manual_len=256`, `first_divergence=-1`

Interpretation:
- after fixing the engine-side special-embedding bug, the manual chain now tracks the official sampled rollout on this long segment
- but the official sampled path still does not emit EOS within the tested 256-step budget, so long-segment instability is not explained by a remaining engine rollout mismatch

### 2. CP unrolled vs official cached CP is NOT the main issue on on-manifold inputs

Experiment: `scripts/python/cp_sampled_parity.py`

Compared on real prefill-derived on-manifold state:
- official `cp.generate(...)`
- our `CodePredictorUnrolled(...)`

Results on representative short segment:
- greedy: exact match
- sampled (20 trials): 20/20 full sequence exact match

This strongly suggests that **unrolled CP vs cached CP is not the primary source** of the observed instability, at least for the sampled short-segment cases tested.

### 3. Exact manual decomposition matches the official `talker.generate()` inputs

Experiments:
- `scripts/python/trace_official_streaming.py`
- `scripts/python/replay_official_talker_stepwise.py`

Using the exact kwargs that official `generate()` passes into `talker.forward()`:
- `input_ids`
- `attention_mask`
- `position_ids`
- `cache_position`
- `past_hidden`
- `trailing_text_hidden`
- `tts_pad_embed`

Observed on text `人工智能正在深刻改变我们的世界。`:
- prefill logits: exact match
- decode step0 CP `codec_ids`: exact match
- decode step1 CP `codec_ids`: exact match
- decode step2 CP `codec_ids`: exact match
- talker logits / `past_hidden`: max diff `0.0` on tested steps

This is strong evidence that:
- our decomposed `talker -> cp.generate -> codec_sum -> talker.model` logic is correct
- HF generation-loop plumbing is NOT the primary source of the observed mismatch
- the earlier parity failure must come from the inputs we feed into the loop, not from a hidden `generate()` behavior we failed to reproduce

### 4. Supported official wrapper prompt aligns prefill/trailing with the engine path

Experiment: `scripts/python/compare_prefill_paths.py`

For text `人工智能正在深刻改变我们的世界。`:
- supported assistant-wrapped `input_ids`: `[151644, 77091, 198, 104455, 96555, 101295, 101933, 103952, 99489, 1773, 151645, 198, 151644, 77091, 198]`
- engine bare text ids: `[104455, 96555, 101295, 101933, 103952, 99489, 1773]`
- official streaming trailing text ids from `input_id[:, 4:-5]`: `[96555, 101295, 101933, 103952, 99489, 1773]`
- engine streaming trailing text ids from the full text remainder: `[96555, 101295, 101933, 103952, 99489, 1773]`
- official trailing length (including EOS): `7`
- engine trailing length (including EOS): `7`
- prefill max diff ≈ `0.00195`
- trailing token diffs: all `0.0` except EOS token max diff ≈ `0.00049`

Interpretation:
- when the official model is invoked through its supported wrapper-style prompt, prefill and trailing text injection align with the engine path
- therefore the supported official path is a valid prefill/trailing baseline

### 5. Bare core `Qwen3TTSForConditionalGeneration.generate(...)` with the short prompt is an invalid baseline

Experiment: `scripts/python/compare_prefill_paths.py --prompt-mode raw`

If the raw core model is called with:
- `"<|im_start|>assistant\n{text}<|im_end|>"`

then the internal slicing:
- first text token: `input_id[:, 3:4]`
- trailing text: `input_id[:, 4:-5]`

will truncate the remainder of the text because the expected wrapper suffix
`"\n<|im_start|>assistant\n"` is missing.

This is a low-level input-contract mismatch, not the supported official path.

### 6. Greedy+punish parity against the supported official baseline still diverges at step2

Experiments:
- `scripts/python/greedy_punish_parity.py`
- `scripts/python/greedy_punish_stagewise_compare.py`
- `scripts/python/greedy_punish_mode_matrix.py`

Compared:
- official local `generate(..., do_sample=False, subtalker_dosample=False, repetition_penalty=1.05)` using the supported wrapper prompt
- our manual local chain with greedy+punish

Observed:
- step0 talker token matches
- step1 talker token matches
- divergence starts at step2

Example on text `人工智能正在深刻改变我们的世界。`:
- official talker tokens start: `[1995, 1085, 450, 832, 419, 209, 44, 1098, 1613, 358, 1744]`
- our talker tokens start: `[1995, 1085, 1714, 1301, 419, 209, 44, 1098, 1055, 1465, 1744]`
- first divergence = step2

This divergence remains real even after fixing the official prompt baseline, so it must be explained by a remaining state / rollout mismatch rather than prompt slicing.

### 7. Mode-matrix result points to exported prefill / early hidden-state mismatch, not the manual decode state machine

Experiment: `scripts/python/greedy_punish_mode_matrix.py`

Compared four rollout modes:
- `official_generate`
  - official `model.generate(...)`
- `official_stepwise`
  - official live-model prefill/trailing + `talker.forward` state machine + manual greedy+punish token selection
- `engine_stepwise`
  - engine exported prefill/trailing + `talker.forward` state machine + manual greedy+punish token selection
- `engine_manual`
  - engine exported prefill/trailing + manual `talker.model / cp.generate / talker.model` loop

Observed on text `人工智能正在深刻改变我们的世界。`:
- `official_generate`: `[1995, 1085, 450, 832, 419, 209, 44, 1098, 1613, 358, 1744]`
- `official_stepwise`: `[1995, 1085, 450, 832, 419, 209, 44, 1098, 1613, 1465, 1744, 1150]`
- `engine_stepwise`: `[1995, 1085, 1714, 1301, 419, 209, 44, 1098, 1055, 1465, 1744, 1150]`
- `engine_manual`: `[1995, 1085, 1714, 1301, 419, 209, 44, 1098, 1055, 1465, 1744, 1150]`

First divergence summary:
- `official_generate` vs `official_stepwise`: step `9`
- `official_generate` vs `engine_stepwise`: step `2`
- `official_generate` vs `engine_manual`: step `2`
- `official_stepwise` vs `engine_stepwise`: step `2`
- `engine_stepwise` vs `engine_manual`: exact match on tested prefix

Interpretation:
- the first problematic divergence at step2 is already present when using the official `talker.forward` state machine with engine exported prefill/trailing
- the manual direct decode loop matches that engine-stepwise path exactly on the tested prefix
- therefore the step2 failure is much more likely caused by exported prefill / early hidden-state mismatch than by missing decode-state plumbing in the manual loop

### 8. Root cause identified: exported `tts_bos/eos/pad` specials were the only meaningful prefill component mismatch

Experiment: `scripts/python/compare_live_vs_exported_prefill.py`

Compared live model vs exported runtime weights on the same supported full-text path.

Observed on text `人工智能正在深刻改变我们的世界。`:
- assistant role embedding diff: exact match
- full text embedding diff: exact match
- codec prefill stack diff: exact match
- raw exported `tts_bos/eos/pad` diff vs live:
  - max diff ≈ `0.000488`
  - mean diff ≈ `1.06e-05`
- runtime-recomputed `tts_bos/eos/pad` diff vs live: exact match

Impact on the full prefill path:
- pre-fix prefill tensor diff was small (`max ≈ 0.00195`)
- but prefill forward `past_hidden` diff amplified to:
  - max diff ≈ `0.28125`
  - mean diff ≈ `0.0517`
- after recomputing specials from the loaded BF16 modules, prefill tensor / prefill `past_hidden` / processed logits all matched exactly on the tested case

Interpretation:
- the earlier step2 divergence was triggered by tiny export-time special-embedding deltas
- those deltas were sufficient to move the prefill hidden state enough to flip the later CP stage2 near-tie

### 9. Runtime fix validated: recomputing specials in `EmbeddingWeights` removes the step2 greedy+punish divergence

Implemented fix:
- [prefill.py](/home/rime/workspace/Qwen3-TTS-Triton/engine/backend/prefill.py)
  - `EmbeddingWeights` now recomputes `tts_pad/bos/eos` from the loaded BF16 `text_embedding + text_projection`
- [export_01_embeddings.py](/home/rime/workspace/Qwen3-TTS-Triton/scripts/export/export_01_embeddings.py)
  - export now computes saved specials using target-dtype modules for runtime parity
- [test_prefill_builder.py](/home/rime/workspace/Qwen3-TTS-Triton/tests/unit/test_prefill_builder.py)
  - regression test added for runtime-recomputed specials

Validation:
- `scripts/python/greedy_punish_parity.py`
  - official talker == manual talker on the tested prefix
- `scripts/python/greedy_punish_stagewise_compare.py`
  - first talker divergence = `-1` on the tested prefix
- `scripts/python/greedy_punish_mode_matrix.py`
  - `engine_stepwise == engine_manual`
  - both now match `official_generate` on the tested prefix

Residual note:
- `official_stepwise` still diverges from `official_generate` later in the pad phase (step9 in the tested sample), but that is separate from the fixed step2 engine mismatch

### 10. After the special-embedding fix, sampled official vs manual rollout also matches on tested prefixes

Experiment: `scripts/python/official_vs_manual_rollout.py`

Observed after the runtime special-embedding fix:
- short text `人工智能正在深刻改变我们的世界。`
  - `official_len=31`, `manual_len=32`
  - `first_divergence=-1`
  - common prefix matches exactly
- longer text `人工智能正在深刻改变我们的世界。从语音识别到自然语言处理，AI的应用已经渗透到生活的方方面面。`
  - `official_len=63`, `manual_len=64`
  - `first_divergence=-1`
  - common prefix matches exactly

Interpretation:
- the special-embedding fix improves not only greedy parity but also the real sampled rollout path on tested texts
- the remaining length mismatch is now only an extra trailing token on the manual side, not an early token-content divergence

### 11. Post-fix long-segment sampled parity also holds on the original 4a and story-style cases

Experiment: `scripts/python/official_vs_manual_rollout.py`

Observed after the runtime special-embedding fix:
- 4a `LONG_TEXT`, `max_steps=256`, `seed=1234`
  - `trailing_len=80`
  - `official_len=255`, `manual_len=256`
  - `first_divergence=-1`
- 4a `LONG_TEXT`, `max_steps=256`, `seed=2025`
  - `trailing_len=80`
  - `official_len=255`, `manual_len=256`
  - `first_divergence=-1`
- `tests/data/story.txt`, `max_steps=512`, `seed=1234`
  - `trailing_len=1217`
  - `official_len=511`, `manual_len=512`
  - `first_divergence=-1`

Interpretation:
- the sampled official/manual parity improvement is not limited to short prefixes
- on the tested long cases, the fix removes early content divergence across the full tested common prefix
- for 4a, both official and manual still fail to terminate within the tested budget, so the remaining long-segment issue now looks official-side (or outside the local PyTorch rollout parity path)

### 12. Greedy+punish stagewise parity now extends through 4a text phase and into pad phase

Experiment: `scripts/python/greedy_punish_mode_matrix.py`

Observed on 4a `LONG_TEXT` after the runtime special-embedding fix:
- `max_steps=64`
  - `official_stepwise == engine_stepwise == engine_manual`
  - `official_generate` first diverges only at step `63`
- `max_steps=128`
  - `official_stepwise == engine_stepwise == engine_manual`
  - `official_generate` first diverges only at step `127`

Interpretation:
- after the fix, stagewise engine parity is preserved well beyond the earlier short-prefix failure
- the shared `official_stepwise / engine_stepwise / engine_manual` path stays aligned through both text consumption and pad-phase continuation on the tested 4a segment
- the remaining discrepancy is between official high-level `generate()` and the explicit stepwise replay, not between the engine rollout and official stepwise behavior

### 13. Post-fix standalone engine reruns do not show a gross long-text length blowup on 4a or story

Experiments:
- `tests/tools/run_engine_long_case.py --case 4a --speaker Serena`
- `tests/tools/run_engine_long_case.py --case story --speaker Serena`

Observed on the fixed runtime:
- 4a end-to-end result
  - session `longtext-medium-serena-postfix`
  - total audio `30.96s`
  - existing saved comparison WAV `workspace/audio_samples/engine/test4a_long_text_medium.wav` is `30.64s`
- story end-to-end result
  - session `longtext-story-postfix`
  - total audio `392.80s`
  - existing saved comparison WAV `workspace/audio_samples/engine/test4d_story.wav` is `395.52s`
- engine logs for the story case
  - session cleaned up normally with `segments=21/21`
  - all logged segments reported `overflow=False`

Interpretation:
- on these representative end-to-end reruns, the fixed branch does not reproduce an obvious runaway-length failure
- if a hallucination is still audible, the next useful reproduction should target the exact offending text / speaker / request path and capture dumps at that point

### 14. Real 4a greedy+punish dump still diverges from the ONNX reference in the CP tail

Experiments:
- `workspace/engine_dumps/4a_greedy_dump_fix_20260416_202542`
- fused ONNX replay on bad dump steps `000003`, `000004`, `000008`

Observed:
- `updated_token_counts` still matches between dump and ONNX replay
- `codec_0` can still match while CP tail already differs
- representative step `000003`
  - dump full codec:
    `[1085, 1989, 550, 206, 767, 1943, 1731, 1977, 327, 948, 294, 269, 761, 1761, 224, 412]`
  - ONNX / PyTorch reference:
    `[1085, 1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 693, 224, 179]`
- representative step `000008`
  - dump full codec:
    `[44, 1558, 1582, 1837, 624, 676, 1699, 1, 287, 223, 1017, 1062, 77, 1320, 499, 1365]`
  - ONNX / PyTorch reference:
    `[44, 581, 1999, 1280, 624, 1928, 1699, 898, 1199, 117, 682, 454, 953, 118, 1050, 551]`

Interpretation:
- after fixing the exported special-embedding bug, the remaining 4a bad dump is no longer explained by a local PyTorch rollout mismatch
- the divergence now shows up specifically in the TRT execution path, first in the CP tail rather than in the penalty bookkeeping

### 15. Standalone `code_predictor_unrolled` shows the same BF16 TRT problem, while FP32 TRT matches ORT

Experiments:
- built with TensorRT 10.15.1 (`nvcr.io/nvidia/tritonserver:26.02-py3`)
  - `code_predictor_unrolled_bf16.engine`
  - `code_predictor_unrolled_fp32.engine`
- random-trial parity:
  - `python tests/tools/verify_code_predictor_trt.py --engine ...code_predictor_unrolled_bf16.engine --trials 10`
  - `python tests/tools/verify_code_predictor_trt.py --engine ...code_predictor_unrolled_fp32.engine --trials 10`
- bad-state parity:
  - same script in dump mode on `000003`, `000004`, `000008`

Observed:
- standalone BF16 TRT vs ORT on random inputs:
  - mismatches `7 / 10`
- standalone FP32 TRT vs ORT on random inputs:
  - mismatches `0 / 10`
- on real bad dump state `000003`
  - ORT tail:
    `[1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 693, 224, 179]`
  - standalone BF16 TRT tail:
    `[1989, 550, 206, 767, 1943, 1731, 1977, 327, 948, 294, 269, 761, 1761, 224, 412]`
  - standalone FP32 TRT tail:
    `[1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 693, 224, 179]`
- on real bad dump state `000008`
  - standalone BF16 TRT still diverges from ORT (first tail divergence at stage `7`)
  - standalone BF16 TRT does **not** exactly match the fused TRT dump tail
  - standalone FP32 TRT exactly matches ORT
- on real bad dump state `000004`
  - standalone BF16 TRT still diverges from ORT, but does not exactly match the fused dump tail
  - standalone FP32 TRT matches ORT

Interpretation:
- the remaining issue is not “official cached CP vs our unrolled CP”
- the remaining issue is not “fused ONNX export semantics”
- the remaining issue is not “penalty parameters only”
- the strongest current explanation is:
  - **TensorRT BF16 execution of `code_predictor_unrolled` is itself numerically/semantically unstable**
  - the fused BF16 engine inherits that CP instability
  - FP32 TRT is a valid control: on tested standalone CP inputs it matches ORT exactly

### 16. Direct `official BF16` vs `TRT BF16` comparison still shows TRT mismatch

Important correction:
- `ORT(fp32)` is only a high-precision control, not the final arbiter for the production comparison
- the more relevant question is whether `TRT BF16` matches the official PyTorch BF16 behavior on the same CP inputs

Experiments:
- loaded official local model in `torch.bfloat16`
- compared:
  - official cached CP under BF16 autocast
  - standalone TRT `code_predictor_unrolled_bf16.engine`
- same fixed inputs for both sides:
  - random `past_hidden + codec_token_0`
  - 4a bad-state inputs reconstructed from fused dump inputs to isolate the CP branch

Observed on random CP inputs:
- tested seeds `42..49`
- official cached BF16 vs TRT BF16 mismatched on `7 / 8` trials
- representative seed `42`, `codec_token_0=[1809]`
  - official cached BF16:
    `[841, 1591, 305, 889, 943, 1212, 931, 61, 89, 266, 16, 637, 175, 242, 928]`
  - TRT BF16:
    `[841, 1591, 305, 1468, 490, 245, 559, 880, 89, 1014, 481, 1190, 1105, 831, 313]`

Observed on 4a bad-state-derived CP inputs:
- step `000003`
  - official cached BF16:
    `[1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 369, 224, 179]`
  - TRT BF16:
    `[1989, 550, 206, 767, 1943, 1731, 1977, 327, 948, 294, 269, 761, 1761, 224, 412]`
  - first divergence at stage `5`
- step `000004`
  - official cached BF16:
    `[1542, 1628, 1804, 1774, 1788, 16, 403, 113, 924, 299, 947, 1183, 815, 891, 29]`
  - TRT BF16:
    `[1542, 1628, 271, 21, 1297, 39, 403, 910, 924, 2026, 640, 481, 1007, 1229, 32]`
  - first divergence at stage `2`

Interpretation:
- even after removing `ORT(fp32)` from the role of final baseline, `TRT BF16` still fails to match official BF16 on the same CP inputs
- so the earlier conclusion survives the stricter comparison:
  - the remaining problem is still in the TRT BF16 execution of the CP branch, not just in parameter choice

### 17. On the original greedy+punish dump path, `updated_token_counts` matches official BF16 but `full_codec` does not

Tooling updates:
- fixed `scripts/export/talker_unified_modules.py` so BF16 replay uses FP32 softmax and casts back before the value matmul
- made `scripts/python/analyze_engine_dump.py` skip optional outputs like `hidden/logits` when the dump does not store them

Experiments:
- replayed the real `4a_greedy_dump_fix_20260416_202542` dumps through:
  - original TRT engine `talker_code2wav_fused.engine`
  - official fused PyTorch replay in `torch.bfloat16`
- command shape:
  - `python scripts/python/analyze_engine_dump.py --dump ... --dtype bfloat16 --model-path workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice`

Observed:
- original TRT rerun exactly reproduces the saved dump outputs on `000003`, `000004`, `000008`
- official BF16 replay does **not** reproduce `full_codec`
  - `000003`: `num_mismatch=13`
  - `000004`: `num_mismatch=26`
  - `000008`: `num_mismatch=21`
- but official BF16 replay does reproduce the talker-side bookkeeping:
  - `updated_token_counts: match=True` on all three dumps
  - `talker_new_kv` cosine stays around `0.99994 ~ 0.99995`
- representative first divergence positions in `full_codec`:
  - `000003` row0: first diff at index `6`
  - `000004` row0: first diff at index `3`
  - `000008` row0: first diff at index `1`

Interpretation:
- under greedy+punish, the **talker/token0 + repetition-penalty bookkeeping is aligned**
- the remaining mismatch is in the **CP tail tokens after token0**, not in `token_counts` / `penalty`
- therefore “both sides become silent, so this must be only a parameter issue” is **not** supported by the token evidence:
  - the actual `full_codec` sequences are still not aligned with official BF16

### 18. The `hidden/logits -> fp32` debug fused engine is not behavior-preserving

Experiments:
- built debug engine:
  - `workspace/exported/custom-1.7b/talker_code2wav_fused.hiddenlogits_fp32.engine`
- reran the same `4a_greedy_dump_fix` dumps through:
  - original fused engine
  - debug fused engine with `hidden/logits` outputs forced to `fp32`

Observed:
- the debug engine changes `full_codec` by itself:
  - `000003`: first diff vs original at index `2`
  - `000004`: first diff vs original at index `3`
  - `000008`: first diff vs original at index `0`
- so the debug engine tail is not equal to the original production engine tail, even on identical inputs

Interpretation:
- forcing `hidden/logits` output formats to `fp32` changes TRT builder/runtime numerics enough to alter token decisions
- therefore, comparisons that use the debug engine's exported `hidden` are only useful as **diagnostic probes**
- they must **not** be treated as ground truth for the original fused engine behavior

## Historical Pre-Fix Narrowing

### Under the corrected official baseline, the first remaining divergence was CP stage2 at talker step1

Experiment: `scripts/python/greedy_punish_stagewise_compare.py`

For the talker output divergence at step2:
- official talker processed top2 starts with `450 > 1714`
- manual talker processed top2 starts with `1714 > 450`
- official/manual CP entry diff after `small_to_mtp_projection`:
  - max diff ≈ `0.25`
  - mean diff ≈ `0.01417`
  - cosine similarity ≈ `0.999866`
- CP stage0 matched
- CP stage1 matched
- **CP stage2 diverged**

Observed for talker step1:
- official CP stage0 token: 1989
- our CP stage0 token: 1989
- official CP stage1 token: 550
- our CP stage1 token: 550
- official CP stage2 token: 1815
- our CP stage2 token: 206

This is still the first known point of argmax flip inside the shared common-prefix region under the corrected official baseline.

### CP stage2 divergence looked like a near-tie flip, not a catastrophic distribution collapse

For CP stage2 logits (talker step1):
- official top10: `[1815, 206, 1810, 459, 1801, 527, 1166, 350, 1376, 1440]`
- our top10: `[206, 1815, 459, 1810, 1801, 527, 1166, 350, 1376, 2008]`
- official top1-top2 margin: `0.25`
- our top1-top2 margin: `0.0`

Interpretation:
- candidate sets are almost the same
- top1/top2 order flips
- this looks much more like a **near-tie argmax sensitivity** than a completely wrong distribution

### CP entry inputs were very close after official projection

Compared official `cp.model` input vs our manually constructed CP input **after** `small_to_mtp_projection`:

For talker step1 CP entry:
- shape match: `[1, 2, 1024]`
- max diff ≈ 0.25
- mean diff ≈ 0.0142
- cosine similarity ≈ 0.999865

Therefore:
- CP entry is not obviously malformed
- token fed into CP is correct at this point
- state is very close but not bit-identical

### CP stage0 and stage1 logits were also very close

For talker step1:
- CP stage0 logits cosine similarity ≈ 0.99985
- CP stage1 logits cosine similarity ≈ 0.99968
- CP stage0 top1 matches
- CP stage1 top1 matches

This reinforces that the divergence is not happening immediately at CP entry.

## Important correction discovered during investigation

### The earlier “official trailing slicing bug” was caused by calling the raw core model with the wrong prompt contract

The supported official wrapper builds:
- `"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"`

Under that prompt, `input_id[:, 4:-5]` correctly recovers the full text remainder.

Therefore:
- the supported official path does **not** have the previously claimed trailing bug
- only the raw core path with the short prompt is invalid as a gold baseline

## What has been ruled out (partially or fully)

### Ruled out strongly
- Repetition penalty formula mismatch vs HF processor (matched exactly in isolated comparison)
- CP unrolled vs cached CP as the primary issue on tested on-manifold sampled cases
- Supported official prompt / trailing mismatch as the primary source of the greedy+punish parity failure
- HF generation-loop plumbing as the primary source of the short-segment parity failure
- Manual `talker.model / cp.generate / talker.model` decode-state plumbing as the primary source of the step2 divergence
- Immediate catastrophic mismatch at talker step0 or talker step1 when using the same official `talker.forward()` inputs
- Completely wrong CP input dimensionality after correcting for `small_to_mtp_projection`
- Exported text embedding / text projection / codec embedding weights as the source of the tested short-segment greedy mismatch

### Not ruled out / still under investigation
- Why `official_stepwise` / `engine_stepwise` still diverge from high-level `official_generate` at later or terminal tested steps
- Whether the end-to-end standalone/TRT serving path still shows hallucination after the local PyTorch rollout parity fix
- Official repo parameter instability on long segments still appears real and needs to be separated from any serving-side issue

## Current best evidence-backed statement

The current strongest evidence is:

1. Under the supported official prompt, prefill and trailing text injection align with the engine path.
2. The earlier short-segment greedy+punish parity failure was traced to tiny exported `tts_bos/eos/pad` deltas.
3. Recomputing those specials from the loaded BF16 modules fixes that engine-side prefill bug.
4. After the fix, short-prefix greedy+punish parity is restored.
5. After the same fix, 4a `LONG_TEXT` greedy+punish stagewise parity also holds through at least 128 tested steps, including pad phase, with `official_stepwise == engine_stepwise == engine_manual`.
6. After the same fix, sampled official vs manual rollout matches on tested short, medium, 4a, and story-style prefixes; no early content divergence has been reproduced in the local PyTorch manual chain.
7. On 4a, official sampled streaming still does not emit EOS within the tested 256-step budget even though manual parity is restored.
8. The remaining known mismatch is now between high-level `official_generate()` and the explicit stepwise replay at later or terminal tested steps, which is separate from the fixed engine prefill issue.
9. The fixed standalone engine reruns of representative 4a/story cases completed normally, with durations close to existing saved outputs and no logged segment overflow.

So the issue currently looks more like:
- a fixed engine-side prefill parity bug caused by export-time special embeddings
- a remaining discrepancy inside official high-level `generate()` vs explicit stepwise replay
- an official long-segment streaming+sample instability problem that is still present after engine/local parity restoration
- and, if a user-facing hallucination still exists, a likely need for an exact-case serving/runtime reproduction rather than more generic parity hunting

## Useful scripts created during investigation

- `scripts/python/cp_sampled_parity.py`
  - official cached CP vs unrolled CP parity
- `scripts/python/pytorch_streaming_baseline.py`
  - PyTorch bf16 streaming baseline
- `scripts/python/greedy_punish_parity.py`
  - local official vs manual greedy+punish parity
- `scripts/python/official_vs_manual_rollout.py`
  - official vs manual rollout comparison
- `scripts/python/replay_official_talker_stepwise.py`
  - official stepwise replay with explicit `position_ids` / `cache_position`
- `scripts/python/trace_official_streaming.py`
  - official trace with signature-preserving hooks
- `scripts/python/compare_prefill_paths.py`
  - supported-vs-raw official prompt comparison for prefill/trailing
- `scripts/python/greedy_punish_stagewise_compare.py`
  - corrected official baseline vs manual engine path, stage-by-stage under greedy+punish
- `scripts/python/greedy_punish_mode_matrix.py`
  - official generate / official stepwise / engine stepwise / engine manual mode matrix
- `scripts/python/compare_live_vs_exported_prefill.py`
  - raw exported specials vs runtime-recomputed specials vs live model parity

## Next recommended step

The next most valuable experiment is now:

- reproduce the exact still-bad user-facing case, if one remains, through the fixed standalone/TRT server
- enable dump capture for that exact session and compare its codec / state progression against the local PyTorch `official_stepwise` / `engine_stepwise` traces
- determine whether any remaining symptom comes from:
  - official high-level long-segment instability,
  - serving-layer segmentation / rollover behavior,
  - or TRT / downstream decode differences outside the already fixed prefill path

The goal is to answer whether:
1. the previously fixed prefill bug was the main engine-side contributor,
2. the remaining issue is now only reproducible on an exact bad case rather than on generic 4a/story regressions, or
3. there is still a separate serving/runtime issue after local rollout parity has been restored.
