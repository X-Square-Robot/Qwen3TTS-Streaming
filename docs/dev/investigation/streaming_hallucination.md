**English** | [中文](streaming_hallucination.zh-CN.md)

# Streaming Hallucination Investigation Summary

## Scope

This document summarizes the investigation background for the streaming/sampling hallucination issue in the development branch.

Investigation focus:
- Compare our local engine chain against the official local model behavior
- Determine whether the issue is caused by:
  1. Official streaming + sampling parameters being unstable on long segments, or
  2. Our chain computing the wrong rollout state / wrong distribution

## High-Level Findings So Far

### 1. The official repo's high-level streaming path is inherently unstable on long segments

Using the local official model (`Qwen3TTSForConditionalGeneration.generate`) with the parameters:
- `non_streaming_mode=False`
- `do_sample=True`
- `subtalker_dosample=True`
- `top_k=50`
- `top_p=1.0`
- `temperature=0.9`
- `repetition_penalty=1.05`

On representative segments we observed:
- Short segment: no EOS (`eos_step = -1`)
- Medium segment: no EOS
- Long 4a segment: no EOS

This is evidence that the official repo's current streaming + sampling path is inherently unstable / unable to terminate naturally on long segments.

After the fix, re-checked on the real 4a long segment (`LONG_TEXT`) using `scripts/python/official_vs_manual_rollout.py`:
- `max_steps=256`, `seed=1234`: `trailing_len=80`, `official_len=255`, `manual_len=256`, `first_divergence=-1`
- `max_steps=256`, `seed=2025`: `trailing_len=80`, `official_len=255`, `manual_len=256`, `first_divergence=-1`

Interpretation:
- After fixing the engine-side special-embedding bug, the manual chain now tracks the official sampling rollout on this long segment
- But the official sampling path still does not emit EOS within the tested 256-step budget, so the long-segment instability cannot be explained by residual engine rollout mismatch

### 2. CP unrolling vs official cached CP is not the primary problem on manifold inputs

Experiment: `scripts/python/cp_sampled_parity.py`

On real prefill-derived manifold states, compared:
- Official `cp.generate(...)`
- Our `CodePredictorUnrolled(...)`

Results on a representative short segment:
- greedy: exact match
- sampling (20 trials): 20/20 full sequences exact match

This strongly suggests that **unrolled CP vs cached CP is not the primary source of the observed instability**, at least for the tested sampled short-segment case.

### 3. Exact manual decomposition matches the official `talker.generate()` inputs

Experiments:
- `scripts/python/trace_official_streaming.py`
- `scripts/python/replay_official_talker_stepwise.py`

Using the exact kwargs that official `generate()` passes to `talker.forward()`:
- `input_ids`
- `attention_mask`
- `position_ids`
- `cache_position`
- `past_hidden`
- `trailing_text_hidden`
- `tts_pad_embed`

On the text `人工智能正在深刻改变我们的世界。` we observed:
- prefill logits: exact match
- decode step0 CP `codec_ids`: exact match
- decode step1 CP `codec_ids`: exact match
- decode step2 CP `codec_ids`: exact match
- talker logits / `past_hidden`: max diff `0.0` on the tested steps

This is strong evidence that:
- Our decomposed `talker -> cp.generate -> codec_sum -> talker.model` logic is correct
- The HF generation-loop plumbing is not the primary source of the observed mismatch
- Earlier parity failures must come from the inputs we feed into the loop, not from a hidden `generate()` behavior we failed to reproduce

### 4. The supported official wrapper prompt aligns prefill/trailing with the engine path

Experiment: `scripts/python/compare_prefill_paths.py`

For the text `人工智能正在深刻改变我们的世界。`:
- Supported assistant-wrapped `input_ids`: `[151644, 77091, 198, 104455, 96555, 101295, 101933, 103952, 99489, 1773, 151645, 198, 151644, 77091, 198]`
- Engine bare-text ids: `[104455, 96555, 101295, 101933, 103952, 99489, 1773]`
- Official streaming trailing text ids from `input_id[:, 4:-5]`: `[96555, 101295, 101933, 103952, 99489, 1773]`
- Engine streaming trailing text ids from the remainder of the full text: `[96555, 101295, 101933, 103952, 99489, 1773]`
- Official trailing length (including EOS): `7`
- Engine trailing length (including EOS): `7`
- prefill max diff ≈ `0.00195`
- trailing token diff: all `0.0` except EOS token max diff ≈ `0.00049`

Interpretation:
- When the official model is invoked through its supported wrapper-style prompt, prefill and trailing text injection align with the engine path
- Therefore the supported official path is a valid prefill/trailing baseline

### 5. Bare-core `Qwen3TTSForConditionalGeneration.generate(...)` with a short prompt is an invalid baseline

Experiment: `scripts/python/compare_prefill_paths.py --prompt-mode raw`

If the bare-core model is invoked with:
- `"<|im_start|>assistant\n{text}<|im_end|>"`

then the internal slicing:
- First text token: `input_id[:, 3:4]`
- Trailing text: `input_id[:, 4:-5]`

will truncate the text remainder, because the expected wrapper suffix `"\n<|im_start|>assistant\n"` is missing.

This is a low-level input-contract mismatch, not the supported official path.

### 6. Greedy+punish parity with the supported official baseline still diverges at step2

Experiments:
- `scripts/python/greedy_punish_parity.py`
- `scripts/python/greedy_punish_stagewise_compare.py`
- `scripts/python/greedy_punish_mode_matrix.py`

Compared:
- Official local `generate(..., do_sample=False, subtalker_dosample=False, repetition_penalty=1.05)` with the supported wrapper prompt
- Our manual local chain using greedy+punish

Observed:
- step0 talker token matches
- step1 talker token matches
- Divergence begins at step2

Example on the text `人工智能正在深刻改变我们的世界。`:
- Official talker token start: `[1995, 1085, 450, 832, 419, 209, 44, 1098, 1613, 358, 1744]`
- Our talker token start: `[1995, 1085, 1714, 1301, 419, 209, 44, 1098, 1055, 1465, 1744]`
- First divergence = step2

This divergence persists even after fixing the official prompt baseline, so it must be explained by residual state / rollout mismatch rather than prompt slicing.

### 7. Mode-matrix results point to an exported prefill / early hidden-state mismatch, not the manual decode state machine

Experiment: `scripts/python/greedy_punish_mode_matrix.py`

Compared four rollout modes:
- `official_generate`
  - Official `model.generate(...)`
- `official_stepwise`
  - Official realtime-model prefill/trailing + `talker.forward` state machine + manual greedy+punish token selection
- `engine_stepwise`
  - Engine-exported prefill/trailing + `talker.forward` state machine + manual greedy+punish token selection
- `engine_manual`
  - Engine-exported prefill/trailing + manual `talker.model / cp.generate / talker.model` loop

On the text `人工智能正在深刻改变我们的世界。` we observed:
- `official_generate`: `[1995, 1085, 450, 832, 419, 209, 44, 1098, 1613, 358, 1744]`
- `official_stepwise`: `[1995, 1085, 450, 832, 419, 209, 44, 1098, 1613, 1465, 1744, 1150]`
- `engine_stepwise`: `[1995, 1085, 1714, 1301, 419, 209, 44, 1098, 1055, 1465, 1744, 1150]`
- `engine_manual`: `[1995, 1085, 1714, 1301, 419, 209, 44, 1098, 1055, 1465, 1744, 1150]`

First-divergence summary:
- `official_generate` vs `official_stepwise`: step `9`
- `official_generate` vs `engine_stepwise`: step `2`
- `official_generate` vs `engine_manual`: step `2`
- `official_stepwise` vs `engine_stepwise`: step `2`
- `engine_stepwise` vs `engine_manual`: exact match on the tested prefix

Interpretation:
- The first problematic divergence at step2 already exists when using the official `talker.forward` state machine with engine-exported prefill/trailing
- The manual direct-decode loop matches that engine stepwise path exactly on the tested prefix
- Therefore the step2 failure is more likely caused by an exported prefill / early hidden-state mismatch than by a missing decode-state pipeline in the manual loop

### 8. Root cause identified: the exported `tts_bos/eos/pad` special embeddings are the only meaningful prefill-component mismatch

Experiment: `scripts/python/compare_live_vs_exported_prefill.py`

On the same supported full-text path, compared the live model against the exported runtime weights.

On the text `人工智能正在深刻改变我们的世界。` we observed:
- assistant-role embedding diff: exact match
- full-text embedding diff: exact match
- codec prefill stack diff: exact match
- exported `tts_bos/eos/pad` vs live original diff:
  - max diff ≈ `0.000488`
  - mean diff ≈ `1.06e-05`
- runtime-recomputed `tts_bos/eos/pad` vs live diff: exact match

Effect on the full prefill path:
- Before the fix the prefill-tensor diff was tiny (`max ≈ 0.00195`)
- But the prefill-forward `past_hidden` diff amplified to:
  - max diff ≈ `0.28125`
  - mean diff ≈ `0.0517`
- After recomputing the special embeddings from the loaded BF16 modules, the prefill tensors / prefill `past_hidden` / processed logits all match exactly on the test case

Interpretation:
- The early step2 divergence is triggered by the tiny export-time special-embedding delta
- These deltas are enough to shift the prefill hidden state, which flips a subsequent near-tie in CP stage2

### 9. Runtime fix verified: recomputing the special embeddings in `EmbeddingWeights` eliminates the step2 greedy+punish divergence

Implemented fix:
- [prefill.py](engine/backend/prefill.py)
  - `EmbeddingWeights` now recomputes `tts_pad/bos/eos` from the loaded BF16 `text_embedding + text_projection`
- [export_01_embeddings.py](scripts/export/export_01_embeddings.py)
  - Export now computes the saved special embeddings with the target-dtype modules for runtime parity
- [test_prefill_builder.py](tests/unit/test_prefill_builder.py)
  - Added a regression test for runtime-recomputed special embeddings

Verification:
- `scripts/python/greedy_punish_parity.py`
  - Official talker == manual talker on the tested prefix
- `scripts/python/greedy_punish_stagewise_compare.py`
  - First talker divergence = `-1` on the tested prefix
- `scripts/python/greedy_punish_mode_matrix.py`
  - `engine_stepwise == engine_manual`
  - Both now match `official_generate` on the tested prefix

Residual note:
- `official_stepwise` still diverges from `official_generate` late in the pad stage (step9 in the test sample), but this is separate from the fixed step2 engine mismatch

### 10. After the special-embedding fix, sampled official vs manual rollout also match on the tested prefix

Experiment: `scripts/python/official_vs_manual_rollout.py`

After the runtime special-embedding fix we observed:
- Short text `人工智能正在深刻改变我们的世界。`
  - `official_len=31`, `manual_len=32`
  - `first_divergence=-1`
  - Common prefix exact match
- Longer text `人工智能正在深刻改变我们的世界。从语音识别到自然语言处理，AI的应用已经渗透到生活的方方面面。`
  - `official_len=63`, `manual_len=64`
  - `first_divergence=-1`
  - Common prefix exact match

Interpretation:
- The special-embedding fix improved not only greedy parity but also the real sampling rollout path on the test texts
- The remaining length mismatch is now just extra trailing tokens on the manual side, not an early token-content divergence

### 11. After the fix, long-segment sampling parity also holds on the original 4a and story-style cases

Experiment: `scripts/python/official_vs_manual_rollout.py`

After the runtime special-embedding fix we observed:
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
- The sampled official/manual parity improvement is not limited to short prefixes
- On the tested long cases, the fix eliminates early content divergence over the full tested common prefix
- For 4a, official and manual still fail to terminate within the test budget, so the remaining long-segment problem now appears to be on the official side (or beyond the local PyTorch rollout parity path)

### 12. Greedy+punish stepwise parity now extends through the 4a text stage and into the pad stage

Experiment: `scripts/python/greedy_punish_mode_matrix.py`

After the runtime special-embedding fix, observed on 4a `LONG_TEXT`:
- `max_steps=64`
  - `official_stepwise == engine_stepwise == engine_manual`
  - `official_generate` first diverges only at step `63`
- `max_steps=128`
  - `official_stepwise == engine_stepwise == engine_manual`
  - `official_generate` first diverges only at step `127`

Interpretation:
- After the fix, stepwise engine parity holds well beyond the early short-prefix failure
- The shared `official_stepwise / engine_stepwise / engine_manual` path stays aligned through text consumption and into the pad-stage continuation
- The remaining difference is between the official high-level `generate()` and the explicit stepwise replay, not between the engine rollout and the official stepwise behavior

### 13. After the fix, standalone engine re-runs show no significant long-text length inflation on 4a or story

Experiments:
- `tests/tools/run_engine_long_case.py --case 4a --speaker Serena`
- `tests/tools/run_engine_long_case.py --case story --speaker Serena`

On the fixed runtime we observed:
- 4a end-to-end result
  - Session `longtext-medium-serena-postfix`
  - Total audio `30.96s`
  - Existing saved comparison WAV `workspace/audio_samples/engine/test4a_long_text_medium.wav` is `30.64s`
- story end-to-end result
  - Session `longtext-story-postfix`
  - Total audio `392.80s`
  - Existing saved comparison WAV `workspace/audio_samples/engine/test4d_story.wav` is `395.52s`
- Engine logs for the story case
  - Session cleaned up normally, `segments=21/21`
  - All logged segments report `overflow=False`

Interpretation:
- On these representative end-to-end re-runs, the fixed branch does not reproduce an obvious runaway length failure
- If a hallucination is still audible, the next useful reproduction should target the exact offending text/speaker/request path and capture a dump at that point

### 14. The real 4a greedy+punish dump still diverges from the ONNX reference at the CP tail

Experiments:
- `workspace/engine_dumps/4a_greedy_dump_fix_20260416_202542`
- Fused ONNX replay on the bad dump steps `000003`, `000004`, `000008`

Observed:
- `updated_token_counts` still match between the dump and the ONNX replay
- `codec_0` can still match while the CP tail already differs
- Representative step `000003`
  - Dump full codec:
    `[1085, 1989, 550, 206, 767, 1943, 1731, 1977, 327, 948, 294, 269, 761, 1761, 224, 412]`
  - ONNX / PyTorch reference:
    `[1085, 1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 693, 224, 179]`
- Representative step `000008`
  - Dump full codec:
    `[44, 1558, 1582, 1837, 624, 676, 1699, 1, 287, 223, 1017, 1062, 77, 1320, 499, 1365]`
  - ONNX / PyTorch reference:
    `[44, 581, 1999, 1280, 624, 1928, 1699, 898, 1199, 117, 682, 454, 953, 118, 1050, 551]`

Interpretation:
- After fixing the exported special-embedding bug, the remaining 4a bad dump is no longer explained by a local PyTorch rollout mismatch
- The divergence now appears specifically in the TRT execution path, first at the CP tail rather than in penalty bookkeeping

### 15. Standalone `code_predictor_unrolled` shows the same BF16 TRT problem, while FP32 TRT matches ORT

Experiments:
- Built with TensorRT 10.15.1 (`nvcr.io/nvidia/tritonserver:26.02-py3`)
  - `code_predictor_unrolled_bf16.engine`
  - `code_predictor_unrolled_fp32.engine`
- Random-trial parity:
  - `python tests/tools/verify_code_predictor_trt.py --engine ...code_predictor_unrolled_bf16.engine --trials 10`
  - `python tests/tools/verify_code_predictor_trt.py --engine ...code_predictor_unrolled_fp32.engine --trials 10`
- Bad-state parity:
  - Ran the same script in dump mode on `000003`, `000004`, `000008`

Observed:
- Standalone BF16 TRT vs ORT on random inputs:
  - Mismatch `7 / 10`
- Standalone FP32 TRT vs ORT on random inputs:
  - Mismatch `0 / 10`
- On real bad-dump state `000003`
  - ORT tail:
    `[1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 693, 224, 179]`
  - Standalone BF16 TRT tail:
    `[1989, 550, 206, 767, 1943, 1731, 1977, 327, 948, 294, 269, 761, 1761, 224, 412]`
  - Standalone FP32 TRT tail:
    `[1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 693, 224, 179]`
- On real bad-dump state `000008`
  - Standalone BF16 TRT still diverges from ORT (first tail divergence at stage `7`)
  - Standalone BF16 TRT does **not** exactly match the fused TRT dump tail
  - Standalone FP32 TRT matches ORT exactly
- On real bad-dump state `000004`
  - Standalone BF16 TRT still diverges from ORT, but does not exactly match the fused dump tail
  - Standalone FP32 TRT matches ORT

Interpretation:
- The remaining problem is not "official cached CP vs our unrolled CP"
- The remaining problem is not "fused ONNX export semantics"
- The remaining problem is not "penalty parameters alone"
- The strongest explanation currently is:
  - **The TensorRT BF16 execution of `code_predictor_unrolled` is itself numerically/semantically unstable**
  - The fused BF16 engine inherits that CP instability
  - FP32 TRT is a valid control: on the tested standalone CP inputs it matches ORT exactly

### 16. The direct `official BF16` vs `TRT BF16` comparison still shows a TRT mismatch

Important correction:
- `ORT(fp32)` is only a high-precision control, not the final arbiter for a production comparison
- The more relevant question is whether `TRT BF16` matches the official PyTorch BF16 behavior on the same CP inputs

Experiments:
- Loaded the official local model in `torch.bfloat16`
- Compared:
  - Official cached CP under BF16 autocast
  - Standalone TRT `code_predictor_unrolled_bf16.engine`
- Both sides use the same fixed inputs:
  - Random `past_hidden + codec_token_0`
  - The 4a bad-state input reconstructed from the fused dump inputs, to isolate the CP branch

On random CP inputs we observed:
- Test seeds `42..49`
- Official cached BF16 vs TRT BF16 mismatch on `7 / 8` trials
- Representative seed `42`, `codec_token_0=[1809]`
  - Official cached BF16:
    `[841, 1591, 305, 889, 943, 1212, 931, 61, 89, 266, 16, 637, 175, 242, 928]`
  - TRT BF16:
    `[841, 1591, 305, 1468, 490, 245, 559, 880, 89, 1014, 481, 1190, 1105, 831, 313]`

On the 4a bad-state-derived CP inputs we observed:
- Step `000003`
  - Official cached BF16:
    `[1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 369, 224, 179]`
  - TRT BF16:
    `[1989, 550, 206, 767, 1943, 1731, 1977, 327, 948, 294, 269, 761, 1761, 224, 412]`
  - First divergence at stage `5`
- Step `000004`
  - Official cached BF16:
    `[1542, 1628, 1804, 1774, 1788, 16, 403, 113, 924, 299, 947, 1183, 815, 891, 29]`
  - TRT BF16:
    `[1542, 1628, 271, 21, 1297, 39, 403, 910, 924, 2026, 640, 481, 1007, 1229, 32]`
  - First divergence at stage `2`

Interpretation:
- Even after removing `ORT(fp32)` from the final-baseline role, `TRT BF16` still fails to match official BF16 on the same CP inputs
- Therefore the earlier conclusion still holds under a stricter comparison:
  - The remaining problem is still in the TRT BF16 execution of the CP branch, not just parameter choices

### 17. On the original greedy+punish dump path, `updated_token_counts` match official BF16 but `full_codec` does not

Tooling updates:
- Fixed `scripts/export/talker_unified_modules.py` so that BF16 replay uses FP32 softmax and converts back before the value matmul
- Made `scripts/python/analyze_engine_dump.py` skip optional outputs like `hidden/logits` when the dump does not store them

Experiments:
- Replayed the real `4a_greedy_dump_fix_20260416_202542` dump through:
  - The original TRT engine `talker_code2wav_fused.engine`
  - The official fused PyTorch replay in `torch.bfloat16`
- Command form:
  - `python scripts/python/analyze_engine_dump.py --dump ... --dtype bfloat16 --model-path workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice`

Observed:
- The original TRT re-run exactly reproduces the saved dump outputs on `000003`, `000004`, `000008`
- The official BF16 replay does **not** reproduce `full_codec`
  - `000003`: `num_mismatch=13`
  - `000004`: `num_mismatch=26`
  - `000008`: `num_mismatch=21`
- But the official BF16 replay does reproduce the talker-side bookkeeping:
  - `updated_token_counts: match=True` on all three dumps
  - `talker_new_kv` cosine stays around `0.99994 ~ 0.99995`
- Representative first divergence position in `full_codec`:
  - `000003` row0: first diff at index `6`
  - `000004` row0: first diff at index `3`
  - `000008` row0: first diff at index `1`

Interpretation:
- Under greedy+punish, **talker/token0 + repetition-penalty bookkeeping are aligned**
- The remaining mismatch is in the **CP tail tokens after token0**, not in `token_counts` / `penalty`
- Therefore "both sides went silent, so this must just be a parameter problem" is **not** supported by the token evidence:
  - The actual `full_codec` sequence is still misaligned with official BF16

### 18. The `hidden/logits -> fp32` debug fused engine is not behavior-preserving

Experiments:
- Built a debug engine:
  - `workspace/exported/custom-1.7b/talker_code2wav_fused.hiddenlogits_fp32.engine`
- Re-ran the same `4a_greedy_dump_fix` dump through:
  - The original fused engine
  - The debug fused engine with `hidden/logits` outputs forced to `fp32`

Observed:
- The debug engine itself changes `full_codec`:
  - `000003`: first diff from original at index `2`
  - `000004`: first diff from original at index `3`
  - `000008`: first diff from original at index `0`
- Therefore the debug-engine tail does not equal the original production-engine tail, even on the same inputs

Interpretation:
- Forcing the `hidden/logits` output format to `fp32` changes the TRT builder/runtime numerics enough to alter token decisions
- Therefore comparisons using the `hidden` exported by the debug engine are only useful as a **diagnostic probe**
- They **must not** be treated as a baseline for the original fused-engine behavior

## Historical Pre-Fix Narrowing

### Under the corrected official baseline, the first remaining divergence is CP stage2 at talker step1

Experiment: `scripts/python/greedy_punish_stagewise_compare.py`

For the step2 talker output divergence:
- Official post-processed talker top2 starts with `450 > 1714`
- Manual post-processed talker top2 starts with `1714 > 450`
- Official/manual CP entry diff after `small_to_mtp_projection`:
  - max diff ≈ `0.25`
  - mean diff ≈ `0.01417`
  - cosine similarity ≈ `0.999866`
- CP stage0 matches
- CP stage1 matches
- **CP stage2 diverges**

At talker step1 we observed:
- Official CP stage0 token: 1989
- Our CP stage0 token: 1989
- Official CP stage1 token: 550
- Our CP stage1 token: 550
- Official CP stage2 token: 1815
- Our CP stage2 token: 206

This remains the known first argmax-flip point within the shared common-prefix region under the corrected official baseline.

### The CP stage2 divergence looks like a near-tie flip, not a catastrophic distribution collapse

For the CP stage2 logits (talker step1):
- Official top10: `[1815, 206, 1810, 459, 1801, 527, 1166, 350, 1376, 1440]`
- Our top10: `[206, 1815, 459, 1810, 1801, 527, 1166, 350, 1376, 2008]`
- Official top1-top2 margin: `0.25`
- Our top1-top2 margin: `0.0`

Interpretation:
- The candidate set is nearly identical
- top1/top2 order flips
- This looks more like **near-tie argmax sensitivity** than a completely wrong distribution

### The CP entry input is very close after the official projection

Comparing the official `cp.model` input against our manually constructed CP input **after** `small_to_mtp_projection`:

For the talker step1 CP entry:
- Shape matches: `[1, 2, 1024]`
- max diff ≈ 0.25
- mean diff ≈ 0.0142
- cosine similarity ≈ 0.999865

Therefore:
- No obvious malformation at the CP entry
- The tokens fed into CP are correct at this point
- The state is very close but not bit-identical

### The CP stage0 and stage1 logits are also very close

For talker step1:
- CP stage0 logits cosine similarity ≈ 0.99985
- CP stage1 logits cosine similarity ≈ 0.99968
- CP stage0 top1 matches
- CP stage1 top1 matches

This reinforces the evidence that the divergence does not happen immediately at the CP entry.

## Important Corrections Found During the Investigation

### The early "official trailing-slice bug" was caused by invoking the bare-core model with the wrong prompt contract

The supported official wrapper builds:
- `"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"`

Under that prompt, `input_id[:, 4:-5]` correctly recovers the full text remainder.

Therefore:
- The supported official path does **not** have the previously claimed trailing bug
- Only the bare-core path with a short prompt is invalid as a golden baseline

## What Has Been Excluded (Partially or Fully)

### Strong exclusions
- Repetition-penalty formula mismatch against the HF processor (exact match in isolated comparison)
- Unrolled CP vs cached CP as the primary problem for the tested manifold sampling case
- Supported official prompt/trailing mismatch as the primary source of greedy+punish parity failures
- The HF generation-loop plumbing as the primary source of short-segment parity failures
- The manual `talker.model / cp.generate / talker.model` decode-state pipeline as the primary source of the step2 divergence
- Immediate catastrophic mismatch at talker step0 or talker step1 when using the same official `talker.forward()` inputs
- Completely wrong CP input dimension after correcting `small_to_mtp_projection`
- Exported text embedding / text projection / codec embedding weights as the source of the tested short-segment greedy mismatch

### Not excluded / still under investigation
- Why `official_stepwise` / `engine_stepwise` still diverge from the high-level `official_generate` at late or terminal test steps
- Whether the end-to-end standalone/TRT serving path still shows hallucination after the local PyTorch rollout parity fix
- The official-repo long-segment parameter instability still looks real and needs to be separated from any serving-side issue

## Currently Best-Supported Statements

The strongest current evidence is:

1. Under the supported official prompt, prefill and trailing text injection align with the engine path.
2. The early short-segment greedy+punish parity failure traces back to the tiny exported `tts_bos/eos/pad` deltas.
3. Recomputing these special embeddings from the loaded BF16 modules fixes that engine-side prefill bug.
4. After the fix, short-prefix greedy+punish parity is restored.
5. After the same fix, 4a `LONG_TEXT` greedy+punish stepwise parity also holds for at least 128 test steps, including the pad stage, with `official_stepwise == engine_stepwise == engine_manual`.
6. After the same fix, sampled official vs manual rollout match on the tested short, medium, 4a, and story-style prefixes; no early content divergence is reproduced in the local PyTorch manual chain.
7. On 4a, even with manual parity restored, the official sampling streaming still does not emit EOS within the tested 256-step budget.
8. The remaining known mismatch is now at late or terminal test steps between the high-level `official_generate()` and the explicit stepwise replay, which is separate from the fixed engine prefill problem.
9. The fixed standalone engine's representative 4a/story re-runs complete normally, with durations close to the existing saved outputs and no logged segment overflow.

Therefore the problem currently looks more like:
- A fixed engine-side prefill parity bug caused by export-time special embeddings
- A remaining difference between the official high-level `generate()` and the explicit stepwise replay
- An official long-segment streaming + sampling instability that persists even after engine/local parity is restored
- And, if user-side hallucination still exists, it may need an exact-case serving/runtime reproduction rather than more generic parity tracing

## Useful Scripts Created During the Investigation

- `scripts/python/cp_sampled_parity.py`
  - Official cached CP vs unrolled CP parity
- `scripts/python/pytorch_streaming_baseline.py`
  - PyTorch bf16 streaming baseline
- `scripts/python/greedy_punish_parity.py`
  - Local official vs manual greedy+punish parity
- `scripts/python/official_vs_manual_rollout.py`
  - Official vs manual rollout comparison
- `scripts/python/replay_official_talker_stepwise.py`
  - Official stepwise replay with explicit `position_ids` / `cache_position`
- `scripts/python/trace_official_streaming.py`
  - Official trace with signature-preserving hooks
- `scripts/python/compare_prefill_paths.py`
  - Supported-vs-bare official prompt comparison for prefill/trailing
- `scripts/python/greedy_punish_stagewise_compare.py`
  - Corrected official baseline vs manual engine path, stage-by-stage comparison under greedy+punish
- `scripts/python/greedy_punish_mode_matrix.py`
  - Official generate / official stepwise / engine stepwise / engine manual mode matrix
- `scripts/python/compare_live_vs_exported_prefill.py`
  - Original exported special embeddings vs runtime-recomputed special embeddings vs live-model parity

## Suggested Next Steps

The most valuable experiments now are:

- If there is still a bad user-side case, reproduce the exact bad case through the fixed standalone/TRT server
- Enable dump capture for that exact session and compare its codec/state evolution against the local PyTorch `official_stepwise` / `engine_stepwise` traces
- Determine whether any remaining symptom comes from:
  - Official high-level long-segment instability,
  - Serving-layer segmentation/rolling behavior,
  - Or a TRT/downstream decode difference outside the fixed prefill path

The goal is to answer:
1. Whether the previously fixed prefill bug was the primary engine-side factor,
2. Whether the remaining problem is now only reproducible on exact bad cases rather than generic 4a/story regressions, or
3. Whether there is still an independent serving/runtime problem after the local rollout parity is restored.

## 2026-06-29 Mixed-Precision (CP=fp32) Engine Re-Check

### Background

Built a mixed-precision engine: the talker backbone stays bf16, while the code_predictor (CP) switches to fp32 (`--cp-precision fp32`, corresponding to the control conclusion in historical finding #15). The expectation was that once the CP numerical instability is eliminated, the streaming hallucination should disappear, but the `short` badcase still intermittently hallucinates (a ~10s sentence inflates to 40.32s = 504 steps of `max_seq_len(512)` overflow, with no natural EOS).

### Reproduction Method (deterministic session_id → deterministic seed)

- The engine sampling seed = `blake2b(base_seed=0, session_id, segment_idx)` (`engine/backend/executor.py:_stable_sampling_seed`). Therefore **a fixed session_id means a fixed seed means a reproducible rollout**. `tests/repeat_case.py` appends a timestamp to the session_id, making it non-reproducible; switched to a deterministic id scan (`workspace/halluc_probe.py`).
- Scanning `halluprobe-0001..0040` hits a reproducible hallucination: **`halluprobe-0007` / `0008` stably produce a 504-step overflow** (fully consistent across multiple re-runs).

### Dump Capture

- Via a compose override (`workspace/compose.dump.yaml`), inject `ENGINE_DUMP_*` into the docker engine and mount `/dumps`, while **keeping the `engine.yaml` sampling config (do_sample=true, temperature=0.9, top_k=50, repetition_penalty=1.05)** — note: do not use the greedy defaults of `run_engine_dump.py`, otherwise the rollout does not reproduce.
- Driving `halluprobe-0007` captures 1 prefill + 504 decode = 505 `.pt` files. The dump inputs include `gumbel_noise` / `cp_gumbel_noise`, so the prototype-replay sampling is **seed-aligned by construction**.

### Stepwise Comparison Against the Prototype (`workspace/scan_dump_divergence.py`, revived and fixed `analyze_engine_dump.py`)

Feed each step's dump inputs into the official PyTorch fused talker (`build_talker_unified_fused_module`) and compare `full_codec`:

- **First divergence at decode step 1, CP stage 7** (stages 0–6 fully consistent).
- The engine CP now hews closer to the **fp32** prototype than to the bf16 one (e.g. step3 matches fp32 up to stage 11 but bf16 only up to stage 6; step4 fp32→stage 7, bf16→stage 5) — **showing CP→fp32 does take effect**.
- But at step 1 stage 7 the **fp32 and bf16 prototypes are themselves inconsistent with each other** (1641 vs 57), i.e. that stage is a high-entropy near-tie point.

### Root-Cause Localization (`workspace/compare_step1_hidden.py`, with a step1 dump containing hidden/logits)

| Comparison | Engine vs fp32 prototype | Engine vs bf16 prototype |
|------|------|------|
| talker `hidden` | cos=0.99990916, **max_abs=0.295** | cos=0.99989, max_abs=0.25 |
| talker `logits` | cos=0.99997, max_abs=0.21 | cos=0.99997, max_abs=0.19 |
| talker token0(stage0) | id=1995, top1−top2 margin=**3.625** (not a tie, matches) | same as left |
| `full_codec` first divergence | stage 7 | stage 7 |

Conclusion chain:

1. **The talker backbone's (bf16 TRT) hidden is not bit-exact**: relative to the fp32 prototype, max_abs≈0.3 (cos 0.9999); relative to torch bf16 it also has max_abs≈0.25, showing that **TRT bf16 talker ≠ torch bf16 talker** (kernel/accumulation differences); the engine hidden is its own thing, different from both prototypes.
2. The talker token0 itself is robust (margin 3.6) → stage 0 is consistent.
3. That ~0.3 hidden perturbation feeds into CP; even though CP is already fp32 (correct in isolated testing, see #15), its **input is a hidden perturbed by the bf16 talker**, and stacked on the CP tail's already-near-tie stage 7 (temp=0.9 high entropy, both fp32/bf16 prototypes flip here), causes the **step 1 stage 7 argmax to flip**.
4. A single flipped sub-code cascades: from step 2 the talker token0 also starts to diverge, and the rollout enters a region that never emits EOS → 504-step overflow → 40.32s hallucination.

### Decisive Control: the prototype does not run away, only the bf16 engine does (`workspace/prototype_rollout.py`)

Using a standalone **fp32** PyTorch rollout driving the official fused talker, **sharing the same prefix state and the same random stream as the engine** (each step's gumbel comes from the engine dump, trajectory-independent; the trajectory-dependent feedback input_embeds/token_counts/past_kv uses the prototype's own outputs, and the text embedding `text_embed[k]=engine_input_embeds[k]−engine_codec_sum[k−1]` is recovered from the dump). The only variable is the decode arithmetic (fp32 prototype vs the bf16 engine in the dump).

Results:

- **The prototype (fp32) emits EOS naturally at step 126** (≈normal ~10s, within the healthy sample's 113–130 chunk range).
- **The engine (bf16) never emits EOS, running to the 504-step cap = 40.32s hallucination.**
- token0 first diverges at step 2 (consistent with the step1 CP stage7 flip cascade).

That is: **the prototype does not run away**. The two have identical starting points and random streams, differing only in the bf16 decode arithmetic → the conclusion is **this is an engine-side bf16 precision issue amplified into a runaway, not upstream model instability** (at least for this seed). Note the prototype here even reuses the engine's bf16 prefill KV as the common starting point and still converges normally, showing that the problem is in **bf16 decode accumulation**, not prefill.

### Ruling Out KV-cache rotation/copy/trimming bugs (`workspace/check_kv_rotation.py`)

Using only the dump to verify "whether each round's input is consistent" (no model needed). For each decode step k≥2, check whether the input equals what the previous step's output should have been:

- `talker_past_kv[k][..., :L_{k-1}, :] == talker_past_kv[k-1]` (prefix preservation)
- `talker_past_kv[k][..., L_{k-1}:, :] == talker_new_kv[k-1]` (incremental append)
- `token_counts[k] == updated_token_counts[k-1]`
- `position_ids` +1 each step

Results (all 504 steps; a correct bf16 copy should be bit-exact):

- step1 `past_kv == prefill new_kv`: max_abs=**0.0**
- Worst-case talker_past_kv copy error throughout: **0.0**; worst-case token_counts feedback: **0.0**; number of inconsistent steps: **0**.
- The third feedback input `input_embeds` (= previous step's codec_sum + text/pad embedding): with the text canceled out, `input_embeds[k]−codec_sum[k−1]` changes with the text token in the first ~32 steps (31 text tokens total, consistent with the log), stabilizes around step33, and in the pad stage (40..504) that pad embedding is **constant** (max drift 0.0039 = bf16 noise).

→ **The engine KV pool's scatter/gather/pingpong rotation is lossless, and each round's input is fully consistent.** So the divergence is not input contamination caused by "wrong copy / wrong trimming / precision loss", but rather the **per-step talker/CP bf16 arithmetic** itself producing different outputs (which are then correctly fed forward). This is also self-consistent with the deterministic rollout: given the same (correctly rotated) inputs and only switching to fp32 arithmetic, it terminates normally.

Note: this check targets the **talker KV** (the chain that drives codec/EOS, i.e. the runaway one). The c2w cache (with sliding window + pingpong) only affects wav synthesis and does not feed back into the talker, so it is unrelated to the "no-EOS runaway".

### Current Judgment

- **CP→fp32 is necessary but not sufficient.** The residual divergence enters from the CP input side via the **bf16 talker backbone's hidden error**, not from the CP arithmetic itself. The user's assumption that "the backbone should not fork from the prototype" does not hold: the bf16 (and TRT) backbone does fork from the prototype (max_abs≈0.3).
- Stage 7 is a model-intrinsic near-tie point (the fp32/bf16 prototypes themselves flip here), so it is sensitive to any tiny perturbation — this is the inherent fragility amplified by bf16, not simply an export/operator bug.
- The deterministic rollout shows that **fp32 decode terminates normally while bf16 decode runs away**, so promoting the talker decode path to fp32 (or higher precision) is expected to eliminate this hallucination.
  **(The measurements in the next section overturn this expectation — see "Full-fp32 Engine Measurement".)**

### Full-fp32 Engine Measurement: precision is not the root cause, only a "shuffle" (built and compared)

Built a **full-fp32** fused engine (`ENGINE_DTYPE=fp32`, backbone+cp+code2wav all fp32, trtexec 172s, engine 7.0G, runtime `I/O dtype consistency check passed: manifest=fp32`), and compared it against the original bf16(cp=fp32) engine on the **same set of 40 deterministic sessions** (`halluprobe-0001..0040`):

| Engine | Hallucinating sessions | Ratio |
|------|------|------|
| bf16(cp=fp32) | 0007, 0008, 0027, 0031, 0037 | **5/40** |
| full fp32 | 0002, 0006, 0008, 0010, 0029, 0033, 0038 | **7/40** |
| Intersection | **only 0008** | |

- fp32 **fixed** 4 (0007/0027/0031/0037) but **newly broke** 6 (0002/0006/0010/0029/0033/0038).
- The two sets are almost disjoint (only 0008 in common); the overall ratio did not go down (5→7, statistically indistinguishable at n=40, ~12–18%).
- Looking at 0007 alone is misleading: fp32 does fix 0007 (9.44s, matching the rollout-predicted ~126 steps), but this is **incidental to that seed**, not a population-level fix.

**Corrected conclusion**: promoting the talker to fp32 **does not eliminate the streaming hallucination**, it only changes "which seeds run away". The essence of the hallucination is **sampling/model-level instability** — under temperature=0.9, for this short text about 12–18% of random seeds enter a trajectory that never emits EOS and run to the 512 cap. Any tiny perturbation (bf16↔fp32, TRT tactic) just pushes different seeds into/out of the "bad attractor basin"; it does not change the incidence rate. 0008 runs away under both bf16 and fp32, a precision-independent intrinsic instability (echoing findings #1/#11: official streaming inherently fails to emit EOS in some cases).

**The real mitigation directions to pursue** (precision-independent):

1. EOS / length control: lower the temperature of token0's EOS decision, or force termination after a maximum audio step count (there is already an overflow-forced EOS, but it sounds bad).
2. Sampling strategy: lower the talker token0 sampling temperature / tune top_k / add stronger repetition control to reduce the probability of entering the bad attractor basin.
3. Runtime detection + resampling: on detecting a runaway (step count far exceeding the EMA expectation), switch the seed and re-run that segment.
4. Confirm with upstream the officially recommended streaming termination strategy.

CP→fp32 is still recommended to keep (finding #15: CP bf16 is itself numerically unstable in isolated testing), but it must be made clear that it is **not** the full solution to the hallucination.

Reproduction/comparison script: `workspace/halluc_probe.py` (the same set of deterministic seeds is scanned); build command `ENGINE_DTYPE=fp32 bash scripts/bash/build_engines.sh --variant custom-1.7b`.

### Decisive Conclusion: the prototype itself has the same hallucination — it is a model / sampling-parameter problem (`workspace/proto_rate.py`)

To distinguish "a prototype parameter problem" from "our export map being inconsistent with the prototype", we ran the autoregressive rollout of the official-weights fused talker under **pure fp32 PyTorch**: each session uses the engine's same seed (`_stable_sampling_seed(0, "halluprobe-NNNN", 0)`) to generate its own Gumbel stream (replicating `_build_sampling_noise`); the prefill state and text/pad embedding schedule are seed-independent and reuse the halluc_0007 dump; only the Gumbel varies with the seed. We tally whether EOS is emitted naturally within 504 steps.

The hallucination rates of the three implementations on the **same set of 40 seeds**:

| Implementation | Hallucinations | Ratio | Hallucinating seeds |
|------|------|------|------|
| bf16 engine (cp=fp32) | 5/40 | 12.5% | 0007 0008 0027 0031 0037 |
| full fp32 engine | 7/40 | 17.5% | 0002 0006 0008 0010 0029 0033 0038 |
| **fp32 PyTorch prototype** | **4/40** | **10%** | **0013 0017 0030 0034** |

- The three ratios are statistically consistent (10–18%, within noise at n=40), but the hallucinating-seed sets are **almost pairwise disjoint** (the prototype's 0013/0017/0030/0034 are in none of the engine sets; the engine's repeatedly-runaway 0008 has a normal EOS@119 under the prototype).
- Validation: the prototype emits EOS naturally at 0007 @115 (same order of magnitude as the fp32 engine's 118 steps and the dump-gumbel rollout's 126 steps; the difference comes from the offset of the single sampling draw at prefill and TRT/TF32 vs PyTorch numerics).

**Conclusion**: **The prototype itself runs away at a rate of ~10–18%** — exactly the "prototype parameter problem, we just hadn't randomly hit the prototype's bad region before" from the user's hypothesis. Therefore the hallucination is **not an inconsistency introduced by the export map / TRT**, but rather an **inherent instability of the model under this set of sampling parameters (temperature=0.9 + top_k=50)**: there are always about 1/6 ~ 1/8 of random seeds that fall into the "never-emit-EOS" attractor basin; bf16/fp32/TRT/PyTorch only decide which specific seeds fall in, not the incidence rate. This is consistent with findings #1/#11 (official streaming inherently fails to emit EOS in some cases), and quantifies it.

→ The fix must be at the **sampling/termination strategy** level (see sections 1–4 above); switching precision or auditing the export map is of no use.

Note: the pure-fp32 prototype rollout reuses the engine's bf16 prefill as the common starting point, and the Gumbel stream has a one-prefill-draw offset relative to the engine; these do not affect the ratio-level conclusion that "prototype hallucination rate ≈ engine hallucination rate".

### Ultimate Control: running "this self-trained checkpoint" with the official unmodified code still hallucinates (`workspace/official_baseline.py`)

> **Important correction**: `workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice` is a symlink to `0601_trained_model` — **our own trained checkpoint**, not the officially released weights. The engine, our concatenation prototype, and the "official-code baseline" below all use this self-trained checkpoint. So this section verifies: **does this self-trained checkpoint still hallucinate through the cleanest path (official unmodified inference code, fp32, no export/TRT/our scripts)?** The officially released weights are not compared here (per the user's request, focus on this checkpoint first).

The "prototype" of the previous section is a fused module **stitched together by our scripts** (official submodules + our topology/unroll, i.e. the export target). To distinguish "our stitching script failed to port some strategy" from "this checkpoint's weights are themselves problematic", we directly drive the **official unmodified** high-level API `Qwen3TTSModel.generate_custom_voice` (→`Qwen3TTSForConditionalGeneration.generate`): fp32, `non_streaming_mode=False` (streaming, matching our pipeline), `max_new_tokens=512` (aligned with the engine's 512 cap), speaker=`001` (= the default voice our `serena` falls back to because it is not in spk_id_map, see engine.yaml `default_speaker:"001"`), 40 seeds each a torch seed.

The official sampling defaults are **exactly the same** as our engine: do_sample=True, top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05, subtalker_dosample=True. eos_token_id=2150.

The hallucination rates of the four implementations on the same set of 40 seeds (>18s = runaway):

(All four use the same **self-trained checkpoint** `0601_trained_model`, speaker=`001`)

| Implementation | Hallucinations | Ratio | Hallucinating seeds |
|------|------|------|------|
| bf16 engine (cp=fp32) | 5/40 | 12.5% | 0007 0008 0027 0031 0037 |
| full fp32 engine | 7/40 | 17.5% | 0002 0006 0008 0010 0029 0033 0038 |
| our fp32 PyTorch concatenation prototype | 4/40 | 10% | 0013 0017 0030 0034 |
| **official unmodified code + self-trained ckpt (fp32)** | **6/40** | **15%** | **0003 0013 0014 0025 0026 0032** |

- **The four ratios are statistically consistent (10–18%)**; the specific bad-seed sets differ due to differing sampling implementations/precision, but the official code path and our concatenation prototype **share 0013** (both PyTorch fp32 paths judge 0013 to run away), mutually corroborating.
- The official code path has a max duration of 40.88s and a median of 9.72s, isomorphic to the engine phenomenon.

**Conclusion (answering the user's dichotomy)**: **This self-trained checkpoint runs away at ~15% even through the official unmodified code (fp32, no export/TRT/our scripts).** Therefore it is **not** "we failed to port some piece of official strategy, causing the export map/engine to be wrong" — our concatenation script, ONNX export, and TRT engine all **faithfully inherit** the intrinsic behavior of this weight. The problem is localized to **this self-trained checkpoint's weights + the officially recommended streaming sampling parameters (temp=0.9/top_k=50/rep_penalty=1.05)**: about 1/6~1/8 of random seeds cannot emit EOS naturally.

**Still unanswered**: whether Qwen3-TTS is just like this, or **this training run (0601) trained the weights badly** — this needs to be compared by running the same test with the **officially released 1.7B CustomVoice weights** (`local official weights`, already available locally). The user currently requests to focus on this checkpoint first, so the official weights have not been run yet.

**Actionable directions**: switching precision / auditing the export map / line-by-line comparison against the official code are all ineffective (verified); either mitigate at the **sampling/termination strategy** level (EMA early termination, token0 EOS temperature reduction, runaway detection + resampling), or if the comparison finds it is a bad training run, **retrain/switch checkpoint**.

Scripts: `workspace/official_baseline.py` (official-repo rate), `workspace/proto_rate.py` (our concatenation prototype rate), `workspace/halluc_probe.py` (engine rate).

### Suggested Next Steps

1. Also promote the talker backbone (at least the talker→CP hidden / `codec_head` logits path) to fp32, and verify whether the step1 stage7 divergence disappears (expected: aligned stepwise with the fp32 prototype).
2. If full fp32 is not possible: evaluate the effect of lowering the CP-tail sampling entropy (this near-tie point is triggered by temp=0.9) on the hallucination rate — but this will change the model behavior.
3. Still need one **pure PyTorch rollout** (same text, same seed) to confirm whether the prototype itself also runs away on that seed, in order to distinguish "engine amplification" from "upstream instability". Stepwise replay cannot answer this alone, because from step≥2 it is fed the engine's already-drifted state.

Reproduction scripts (all under `workspace/`, gitignored): `halluc_probe.py`, `compose.dump.yaml`, `scan_dump_divergence.py`, `compare_step1_hidden.py`, `analyze_engine_dump.py` (revived from git `ae77d68^` and fixed imports).

## 2026-07-02: The 0701 Retrained Checkpoint Fixes It — Verified Through the Production bf16 Engine

A researcher retrained the model (`/home/train/tts/qwen3-tts/trained/zehan/0701_trained_model`) and reported no hallucination. Verified with the same methodology on the same deterministic seeds.

> **Important**: the `workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice` symlink now points to `0701_trained_model` (previously `0601_trained_model`, the broken one). Every `0601` rate in the tables above is that self-trained 0601 checkpoint.

### Model-level check (official unmodified code, same 40 seeds)

| checkpoint | dtype | hallucination | duration |
|---|---|---|---|
| 0601 (old) | fp32 | 6/40 | median 9.72s, max 40.88s |
| 0701 (new) | fp32 | **0/40** | 9.52–10.48s, median 10.04s |
| 0701 (new) | bf16 | **0/40** | 9.52–10.64s, median 10.00s |

### Production path (bf16 TRT engine rebuilt from 0701)

Full pipeline from the 0701 symlink: Phase A re-export (`export_all.py --variant custom-1.7b` — must run in the `qwen3-tts` env with `PYTHONPATH=third_party/Qwen3-TTS`; `export_models.sh` uses base `python3` and fails with "qwen_tts not found"), Phase B build (`build_engines.sh --variant custom-1.7b`, default `--bf16 --layerPrecisions=/talker_fused/cp/*:fp32` = backbone bf16 + cp fp32 + code2wav bf16, 196s, 4.1G), then `compose.sh prepare` + `up` + `docker restart` (the container only reloads `model.plan` on restart — `up` alone sees no spec change).

| engine (deployed path) | hallucination | duration |
|---|---|---|
| 0601 bf16 engine | 5/40 | max 40.32s |
| **0701 bf16 engine** | **0/100** | 9.44–10.72s, median 10.04s |

### Conclusion

Three levels agree — the 0701 retraining eliminates the runaway, and the fix survives bf16 quantization all the way to the deployed bf16 TRT engine. On the exact seeds that drove 0601 to run away, 0701 never does; the duration distribution is tight (~10s, max 10.7s, far from the ~40s / 512-step cap), so it is a real EOS-margin fix, not luck. `0/100` → 95% CI upper bound ~3.6%, well below 0601's ~15%.

This confirms the earlier localization: the streaming hallucination was caused by the **0601 training run producing unhealthy weights**, not by Qwen3-TTS itself, our export map, or the TRT engine — all of which faithfully reproduced the checkpoint. Retraining (0701) fixes it.

**Deployment state**: the currently deployed engine is now the **0701 bf16 build** (no longer 0601); the 0601 engine artifacts were overwritten by this rebuild.

Scripts: `workspace/official_baseline.py` (parametrized by `QWEN_MODEL_DIR` / `QWEN_DTYPE` / `QWEN_N`), `workspace/halluc_probe.py` (engine probe, `--n`).

## 2026-07-06: cp=bf16 validated on 0701 — the fp32 CP constraint is lifted

The cp=fp32 mitigation (Finding #15 era) had never been re-tested after the 0701 retrain: all 0701 validations above ran on the mixed-precision engine (backbone bf16 + **cp fp32** + code2wav bf16). Since the hallucination root cause turned out to be the 0601 weights — with precision merely shuffling which seeds went bad — the question was whether CP's bf16 numerical noise (still a real fact in isolation: standalone CP bf16 TRT vs ORT mismatched 7/10 random trials, Finding #15) translates into hallucination on healthy weights at all.

Single-variable test: rebuilt the fused engine as **full bf16** (`--cp-precision bf16`, batch profile unchanged at 32), deployed on the production path, and ran the identical deterministic probe:

| engine (0701 weights) | hallucination | duration |
|---|---|---|
| bf16 + cp **fp32** (baseline above) | 0/100 | 9.44–10.72s, median 10.04s |
| **full bf16 (cp=bf16)** | **0/100** | 9.28–10.88s, median 10.08s |

A follow-up full-bf16 build at batch=128 also probed clean (0/40). The distributions are statistically indistinguishable; CP bf16's near-tie argmax flips demonstrably do not cascade into runaway on healthy weights — consistent with the earlier finding that precision only reshuffles bad seeds, and 0701 has none on this probe set.

**Consequence**: `build_engines.sh` no longer defaults `CP_PRECISION` to fp32 (it now follows `ENGINE_DTYPE`); fp32 CP cost real decode latency (~45% of kernel time was CP under fp32). `CP_PRECISION=fp32` remains available for numerical-parity debugging or reproducing the historical mixed-precision build.
