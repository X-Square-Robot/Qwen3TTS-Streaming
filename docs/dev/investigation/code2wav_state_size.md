**English** | [中文](code2wav_state_size.zh-CN.md)

# KV Cache and Convolution History Size for the Code2Wav Module (16 lanes × up to 1024 tokens)

## Sliding-window KV cache explained

### What is a sliding-window KV cache?

- **Sliding Window Attention**: when each token performs causal attention, it can **only see at most the past W positions** (including itself), rather than the full history "from the beginning up to the current step". W is the window size (72 here).
- **SlidingWindowKVCache**: because the model **already does not look** at history more than 72 steps back, the cache only needs to retain the K/V for the **most recent 72 time steps**; once written, earlier K/V will never be used by attention again and can be discarded to save GPU memory while keeping the computation consistent.

### Doesn't a Transformer need the full history?

- **A typical autoregressive LM**: usually **fully causal** (each position can see every token from 0 up to the current one), so it needs a full KV cache.
- **The Code2Wav decoder**: the official architecture is **local / sliding-window causal** (`layer_types = ["sliding_attention"] * num_hidden_layers`, using `create_sliding_window_causal_mask`). This is part of the **model design**, not something we changed during export — many vocoder-side designs use local attention to trade for efficiency and stability, and 72 is the default value of `sliding_window` in the config.

### Why is the window size 72?

- It comes from the default value `sliding_window=72` in **Qwen3TTSTokenizerV2DecoderConfig** (about 6 seconds of context under a 12 Hz codec), set officially in tokenizer v2 to serve as a "local attention mechanism, limiting attention context to improve efficiency".

### How is the case of only 4 tokens handled?

- When the current step **feeds in only 4 tokens** (for example the first chunk, or an empty cache):
  - `S_past = 0` or very small, and the current step's key/value has only 4 entries (or 4 + past).
  - The **actual length in the cache is min(72, S_past + 4)**, which will never exceed 72, so **no truncation is performed**.
- In other words: **with only 4 tokens, only 4 positions of K/V are stored**; the window of 72 merely means "retain at most 72" — if there are fewer than 72, all of them are retained, and no special branch is needed.

---

## Configuration sources

- **Decoder config** (Qwen3TTSTokenizerV2DecoderConfig): `third_party/Qwen3-TTS/.../configuration_qwen3_tts_tokenizer_v2.py`
- **State definitions**: `scripts/export/code2wav_streaming.py` (`get_initial_state_shapes`, `SlidingWindowKVCache`)

| Parameter | Value |
|------|-----|
| num_hidden_layers | 8 |
| num_key_value_heads | 16 |
| hidden_size | 1024 |
| head_dim | 1024/16 = 64 |
| latent_dim | 1024 |
| decoder_dim | 1536 |
| codebook_dim | 512 (getattr default) |
| **KV window_size** | **72** (sliding window, independent of decode length) |

---

## 1. KV Cache

- **Sliding window**: each layer retains only the K/V for the most recent **72** time steps, independent of "up to 1024 decode tokens".
- Single layer, single lane: `K` / `V` both have shape `[1, num_kv_heads, 72, head_dim]` = `[1, 16, 72, 64]`.

**Single lane, single layer (K+V, BF16):**

- Element count: `2 × 16 × 72 × 64 = 147,456`
- Size: `147,456 × 2 bytes = 294,912 bytes`

**Single lane, 8 layers (BF16):**

- Element count: `8 × 147,456 = 1,179,648`
- Size: `1,179,648 × 2 = 2,359,296 bytes ≈ 2.25 MB`

**16 lanes (B=16):**

- Size: `16 × 2,359,296 = 37,748,736 bytes ≈ 36.00 MB`

---

## 2. Convolution history (Conv States)

17 conv states; their shapes are independent of decode length and depend only on the batch and the channel/length dimensions (from `get_initial_state_shapes`):

| State | Shape (B=1) | Element count (B=1) |
|-------|-------------|----------------|
| conv_state_0 | (1, 512, 2) | 1,024 |
| conv_state_1,2,3 | (1, 1024, 6) × 3 | 18,432 |
| conv_state_4,5,6 | (1, 768, 6), (1, 768, 18), (1, 768, 54) | 59,904 |
| conv_state_7,8,9 | (1, 384, 6), (1, 384, 18), (1, 384, 54) | 29,952 |
| conv_state_10,11,12 | (1, 192, 6), (1, 192, 18), (1, 192, 54) | 14,976 |
| conv_state_13,14,15 | (1, 96, 6), (1, 96, 18), (1, 96, 54) | 7,488 |
| conv_state_16 | (1, 96, 6) | 576 |
| **Total** | | **131,752** |

**Single lane (BF16):** `131,752 × 2 = 263,504 bytes ≈ 0.257 MB`  
**16 lanes:** `16 × 263,504 = 4,216,064 bytes ≈ 4.02 MB`

---

## 3. Transconv overlap states (Overlap-Add)

4 transconv overlaps, `rp = [8, 5, 4, 3]`, `out_dim = decoder_dim // 2^(block_idx+1)`:

| State | Shape (B=1) | Element count (B=1) |
|-------|-------------|----------------|
| transconv_overlap_0 | (1, 768, 8) | 6,144 |
| transconv_overlap_1 | (1, 384, 5) | 1,920 |
| transconv_overlap_2 | (1, 192, 4) | 768 |
| transconv_overlap_3 | (1, 96, 3) | 288 |
| **Total** | | **9,120** |

**Single lane (BF16):** `9,120 × 2 = 18,240 bytes ≈ 0.018 MB`  
**16 lanes:** `16 × 18,240 = 291,840 bytes ≈ 0.28 MB`

---

## 4. Summary (16 lanes, BF16)

| Category | Single lane (MB) | 16 lanes (MB) |
|------|-----------|------------|
| KV cache (fixed 72 steps) | 2.25 | **36.00** |
| Conv history (17 states) | 0.257 | **4.02** |
| Transconv overlap (4) | 0.018 | **0.28** |
| **Total** | **≈2.53** | **≈40.3** |

---

## 5. On "up to 1024 decode tokens"

- Code2Wav's sequence dimension is the **codec time step** (each step feeds in chunk_T=4 frames).
- Using **SlidingWindowKVCache(window_size=72)**: each layer retains only the K/V for the most recent 72 time steps, so a longer decode does not increase the KV size.
- Therefore: at **16 lanes, up to 1024 decode tokens**, the total size of the KV cache plus the conv/transconv states is about **40.3 MB (BF16)**, independent of decode length.

If the states are stored in FP32, the sizes above are multiplied by 2, giving about **80.6 MB**.
