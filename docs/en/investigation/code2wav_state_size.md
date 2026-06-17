# Code2Wav 模块 16 路 × 最大 1024 token 的 KV Cache 与卷积历史大小

## 滑动窗口 KV Cache 说明

### 什么叫滑动窗口 KV cache？

- **滑动窗口注意力（Sliding Window Attention）**：每个 token 做因果注意力时，**只能看到过去最多 W 个位置**（含自己），而不是“从开头到当前”的全量历史。W 就是窗口大小（这里 72）。
- **SlidingWindowKVCache**：因为模型**本来就不看**超过 72 步以前的历史，所以 cache 里只保留**最近 72 个时间步**的 K/V 即可；更早的 K/V 写入后就不会再被 attention 用到，可以丢弃，以节省显存并保持计算一致。

### Transformer 不是需要全量历史吗？

- **一般自回归 LM**：通常是**全量因果**（每个位置能看到从 0 到当前的所有 token），所以需要全量 KV cache。
- **Code2Wav 的 Decoder**：官方架构是 **局部/滑动窗口因果**（`layer_types = ["sliding_attention"] * num_hidden_layers`，使用 `create_sliding_window_causal_mask`）。这是**模型设计**，不是我们导出时改的——vocoder 侧很多设计会用局部注意力换效率和稳定性，72 是配置里的 `sliding_window` 默认值。

### 窗口大小为什么是 72？

- 来自 **Qwen3TTSTokenizerV2DecoderConfig** 的默认值 `sliding_window=72`（12Hz codec 下约 6 秒上下文），是官方在 tokenizer v2 里定的，用于“local attention mechanism, limiting attention context to improve efficiency”。

### 如果只有 4 个 token 如何处理？

- 当前步**只输入 4 个 token** 时（例如首 chunk 或 cache 为空）：
  - `S_past = 0` 或很小，当前步的 key/value 只有 4 条（或 4 + past）。
  - Cache 里**实际长度就是 min(72, S_past + 4)**，不会超过 72，**不会做截断**。
- 也就是说：**只有 4 个 token 时，就只存 4 个位置的 K/V**；窗口 72 只是“最多保留 72”，不足 72 就全部保留，无需特殊分支。

---

## 配置来源

- **Decoder config**（Qwen3TTSTokenizerV2DecoderConfig）：`third_party/Qwen3-TTS/.../configuration_qwen3_tts_tokenizer_v2.py`
- **State 定义**：`scripts/export/code2wav_streaming.py`（`get_initial_state_shapes`、`SlidingWindowKVCache`）

| 参数 | 值 |
|------|-----|
| num_hidden_layers | 8 |
| num_key_value_heads | 16 |
| hidden_size | 1024 |
| head_dim | 1024/16 = 64 |
| latent_dim | 1024 |
| decoder_dim | 1536 |
| codebook_dim | 512（getattr 默认） |
| **KV window_size** | **72**（滑动窗口，与解码长度无关） |

---

## 1. KV Cache

- **滑动窗口**：每层只保留最近 **72** 个时间步的 K/V，与“最大解码 1024 token”无关。
- 单层单路：`K` / `V` 形状均为 `[1, num_kv_heads, 72, head_dim]` = `[1, 16, 72, 64]`。

**单路、单层（K+V，BF16）：**

- 元素数：`2 × 16 × 72 × 64 = 147,456`
- 体积：`147,456 × 2 bytes = 294,912 bytes`

**单路、8 层（BF16）：**

- 元素数：`8 × 147,456 = 1,179,648`
- 体积：`1,179,648 × 2 = 2,359,296 bytes ≈ 2.25 MB`

**16 路（B=16）：**

- 体积：`16 × 2,359,296 = 37,748,736 bytes ≈ 36.00 MB`

---

## 2. 卷积历史（Conv States）

17 个 conv state，形状与解码长度无关，仅与 batch 和通道/长度维有关（来自 `get_initial_state_shapes`）：

| State | Shape (B=1) | 元素数 (B=1) |
|-------|-------------|----------------|
| conv_state_0 | (1, 512, 2) | 1,024 |
| conv_state_1,2,3 | (1, 1024, 6) × 3 | 18,432 |
| conv_state_4,5,6 | (1, 768, 6), (1, 768, 18), (1, 768, 54) | 59,904 |
| conv_state_7,8,9 | (1, 384, 6), (1, 384, 18), (1, 384, 54) | 29,952 |
| conv_state_10,11,12 | (1, 192, 6), (1, 192, 18), (1, 192, 54) | 14,976 |
| conv_state_13,14,15 | (1, 96, 6), (1, 96, 18), (1, 96, 54) | 7,488 |
| conv_state_16 | (1, 96, 6) | 576 |
| **合计** | | **131,752** |

**单路（BF16）：** `131,752 × 2 = 263,504 bytes ≈ 0.257 MB`  
**16 路：** `16 × 263,504 = 4,216,064 bytes ≈ 4.02 MB`

---

## 3. Transconv 重叠状态（Overlap-Add）

4 个 transconv overlap，`rp = [8, 5, 4, 3]`，`out_dim = decoder_dim // 2^(block_idx+1)`：

| State | Shape (B=1) | 元素数 (B=1) |
|-------|-------------|----------------|
| transconv_overlap_0 | (1, 768, 8) | 6,144 |
| transconv_overlap_1 | (1, 384, 5) | 1,920 |
| transconv_overlap_2 | (1, 192, 4) | 768 |
| transconv_overlap_3 | (1, 96, 3) | 288 |
| **合计** | | **9,120** |

**单路（BF16）：** `9,120 × 2 = 18,240 bytes ≈ 0.018 MB`  
**16 路：** `16 × 18,240 = 291,840 bytes ≈ 0.28 MB`

---

## 4. 汇总（16 路，BF16）

| 类别 | 单路 (MB) | 16 路 (MB) |
|------|-----------|------------|
| KV cache（固定 72 步） | 2.25 | **36.00** |
| Conv 历史（17 个 state） | 0.257 | **4.02** |
| Transconv 重叠（4 个） | 0.018 | **0.28** |
| **合计** | **≈2.53** | **≈40.3** |

---

## 5. 关于“最大解码 1024 token”

- Code2Wav 的序列维度是 **codec 时间步**（每步输入 chunk_T=4 帧）。
- 使用 **SlidingWindowKVCache(window_size=72)**：每层只保留最近 72 个时间步的 K/V，再长的解码也不会增加 KV 体积。
- 因此：**16 路、最大解码 1024 token** 时，KV cache 与卷积/transconv 状态总大小约为 **40.3 MB（BF16）**，与解码长度无关。

若用 FP32 存状态，上述体积乘以 2，约 **80.6 MB**。
