# 流式幻觉调查总结

## 范围

本文档总结了开发分支中流式/采样幻觉问题的调查背景。

调查重点：
- 比较我们的本地引擎链与官方本地模型行为
- 确定问题是否由以下原因导致：
  1. 官方流式+采样参数在长段上不稳定，或
  2. 我们的链计算了错误的展开状态 / 错误的分布

## 迄今为止的高级发现

### 1. 官方仓库高级流式路径在长段上本身就不稳定

使用本地官方模型（`Qwen3TTSForConditionalGeneration.generate`），参数为：
- `non_streaming_mode=False`
- `do_sample=True`
- `subtalker_dosample=True`
- `top_k=50`
- `top_p=1.0`
- `temperature=0.9`
- `repetition_penalty=1.05`

在代表性段落上观察到：
- 短段：无 EOS（`eos_step = -1`）
- 中段：无 EOS
- 长 4a 段：无 EOS

这是证据表明官方仓库当前的流式+采样路径在长段上本身不稳定/不能自然终止。

修复后在真实 4a 长段（`LONG_TEXT`）上使用 `scripts/python/official_vs_manual_rollout.py` 重新检查：
- `max_steps=256`，`seed=1234`：`trailing_len=80`，`official_len=255`，`manual_len=256`，`first_divergence=-1`
- `max_steps=256`，`seed=2025`：`trailing_len=80`，`official_len=255`，`manual_len=256`，`first_divergence=-1`

解读：
- 修复引擎端特殊嵌入 bug 后，手动链现在在此长段上跟踪官方采样展开
- 但官方采样路径在测试的 256 步预算内仍不发射 EOS，因此长段不稳定不能通过剩余引擎展开不匹配来解释

### 2. CP 展开与官方缓存 CP 不是流形输入上的主要问题

实验：`scripts/python/cp_sampled_parity.py`

在真实 prefill 派生的流形状态上比较：
- 官方 `cp.generate(...)`
- 我们的 `CodePredictorUnrolled(...)`

在代表性短段上的结果：
- greedy：精确匹配
- 采样（20 次试验）：20/20 完整序列精确匹配

这强烈表明 **展开 CP 与缓存 CP 不是观察到的不稳定性的主要来源**，至少对于测试的采样短段情况。

### 3. 精确手动分解与官方 `talker.generate()` 输入匹配

实验：
- `scripts/python/trace_official_streaming.py`
- `scripts/python/replay_official_talker_stepwise.py`

使用官方 `generate()` 传递给 `talker.forward()` 的确切 kwargs：
- `input_ids`
- `attention_mask`
- `position_ids`
- `cache_position`
- `past_hidden`
- `trailing_text_hidden`
- `tts_pad_embed`

在文本 `人工智能正在深刻改变我们的世界。` 上观察到：
- prefill logits：精确匹配
- decode step0 CP `codec_ids`：精确匹配
- decode step1 CP `codec_ids`：精确匹配
- decode step2 CP `codec_ids`：精确匹配
- talker logits / `past_hidden`：测试步骤上最大差异 `0.0`

这是有力证据表明：
- 我们分解的 `talker -> cp.generate -> codec_sum -> talker.model` 逻辑是正确的
- HF 生成循环管道不是观察到的不匹配的主要来源
- 早期奇偶校验失败必然来自我们输入到循环的输入，而不是我们未能重现的隐藏 `generate()` 行为

### 4. 支持的官方包装器提示将 prefill/trailing 与引擎路径对齐

实验：`scripts/python/compare_prefill_paths.py`

对于文本 `人工智能正在深刻改变我们的世界。`：
- 支持的 assistant 包装 `input_ids`：`[151644, 77091, 198, 104455, 96555, 101295, 101933, 103952, 99489, 1773, 151645, 198, 151644, 77091, 198]`
- 引擎裸文本 id：`[104455, 96555, 101295, 101933, 103952, 99489, 1773]`
- 来自 `input_id[:, 4:-5]` 的官方流式尾部文本 id：`[96555, 101295, 101933, 103952, 99489, 1773]`
- 来自完整文本剩余部分的引擎流式尾部文本 id：`[96555, 101295, 101933, 103952, 99489, 1773]`
- 官方尾部长度（包含 EOS）：`7`
- 引擎尾部长度（包含 EOS）：`7`
- prefill 最大差异 ≈ `0.00195`
- 尾部 token 差异：除 EOS token 最大差异 ≈ `0.00049` 外全部为 `0.0`

解读：
- 当通过其支持的包装器样式提示调用官方模型时，prefill 和尾部文本注入与引擎路径对齐
- 因此支持的官方路径是有效的 prefill/trailing 基准

### 5. 使用短提示的裸核心 `Qwen3TTSForConditionalGeneration.generate(...)` 是无效基准

实验：`scripts/python/compare_prefill_paths.py --prompt-mode raw`

如果使用以下内容调用裸核心模型：
- `"<|im_start|>assistant\n{text}<|im_end|>"`

那么内部切片：
- 第一个文本 token：`input_id[:, 3:4]`
- 尾部文本：`input_id[:, 4:-5]`

将截断文本剩余部分，因为缺少预期的包装器后缀 `"\n<|im_start|>assistant\n"`。

这是低级输入契约不匹配，不是支持的官方路径。

### 6. Greedy+punish 与支持的官方基准的奇偶性仍在 step2 分歧

实验：
- `scripts/python/greedy_punish_parity.py`
- `scripts/python/greedy_punish_stagewise_compare.py`
- `scripts/python/greedy_punish_mode_matrix.py`

比较：
- 使用支持的包装器提示的官方本地 `generate(..., do_sample=False, subtalker_dosample=False, repetition_penalty=1.05)`
- 我们使用 greedy+punish 的手动本地链

观察到：
- step0 talker token 匹配
- step1 talker token 匹配
- 分歧从 step2 开始

在文本 `人工智能正在深刻改变我们的世界。` 上的示例：
- 官方 talker token 开始：`[1995, 1085, 450, 832, 419, 209, 44, 1098, 1613, 358, 1744]`
- 我们的 talker token 开始：`[1995, 1085, 1714, 1301, 419, 209, 44, 1098, 1055, 1465, 1744]`
- 首次分歧 = step2

即使在修复官方提示基准后，此分歧仍然存在，因此必须由剩余状态/展开不匹配而非提示切片来解释。

### 7. 模式矩阵结果指向导出的 prefill / 早期隐藏状态不匹配，而非手动解码状态机

实验：`scripts/python/greedy_punish_mode_matrix.py`

比较四种展开模式：
- `official_generate`
  - 官方 `model.generate(...)`
- `official_stepwise`
  - 官方实时模型 prefill/trailing + `talker.forward` 状态机 + 手动 greedy+punish token 选择
- `engine_stepwise`
  - 引擎导出 prefill/trailing + `talker.forward` 状态机 + 手动 greedy+punish token 选择
- `engine_manual`
  - 引擎导出 prefill/trailing + 手动 `talker.model / cp.generate / talker.model` 循环

在文本 `人工智能正在深刻改变我们的世界。` 上观察到：
- `official_generate`：`[1995, 1085, 450, 832, 419, 209, 44, 1098, 1613, 358, 1744]`
- `official_stepwise`：`[1995, 1085, 450, 832, 419, 209, 44, 1098, 1613, 1465, 1744, 1150]`
- `engine_stepwise`：`[1995, 1085, 1714, 1301, 419, 209, 44, 1098, 1055, 1465, 1744, 1150]`
- `engine_manual`：`[1995, 1085, 1714, 1301, 419, 209, 44, 1098, 1055, 1465, 1744, 1150]`

首次分歧总结：
- `official_generate` vs `official_stepwise`：step `9`
- `official_generate` vs `engine_stepwise`：step `2`
- `official_generate` vs `engine_manual`：step `2`
- `official_stepwise` vs `engine_stepwise`：step `2`
- `engine_stepwise` vs `engine_manual`：测试前缀上精确匹配

解读：
- step2 的第一个问题分歧在使用官方 `talker.forward` 状态机和引擎导出 prefill/trailing 时已经存在
- 手动直接解码循环在测试前缀上与该引擎逐步路径完全匹配
- 因此 step2 失败更可能是由导出的 prefill / 早期隐藏状态不匹配导致，而非手动循环中缺失解码状态管道

### 8. 根本原因已确定：导出的 `tts_bos/eos/pad` 特殊嵌入是唯一有意义的 prefill 组件不匹配

实验：`scripts/python/compare_live_vs_exported_prefill.py`

在相同支持的全文路径上比较实时模型与导出的运行时权重。

在文本 `人工智能正在深刻改变我们的世界。` 上观察到：
- assistant 角色嵌入差异：精确匹配
- 完整文本嵌入差异：精确匹配
- codec prefill 栈差异：精确匹配
- 导出的 `tts_bos/eos/pad` 与实时原始差异：
  - 最大差异 ≈ `0.000488`
  - 平均差异 ≈ `1.06e-05`
- 运行时重新计算的 `tts_bos/eos/pad` 与实时差异：精确匹配

对完整 prefill 路径的影响：
- 修复前 prefill 张量差异很小（`max ≈ 0.00195`）
- 但 prefill 前向 `past_hidden` 差异放大到：
  - 最大差异 ≈ `0.28125`
  - 平均差异 ≈ `0.0517`
- 从加载的 BF16 模块重新计算特殊嵌入后，测试用例上 prefill 张量 / prefill `past_hidden` / 处理后的 logits 全部精确匹配

解读：
- 早期 step2 分歧由微小的导出时特殊嵌入增量触发
- 这些增量足以移动 prefill 隐藏状态，从而翻转后续 CP stage2 的接近平局

### 9. 运行时修复已验证：在 `EmbeddingWeights` 中重新计算特殊嵌入消除了 step2 greedy+punish 分歧

已实现的修复：
- [prefill.py](/home/rime/workspace/Qwen3-TTS-Triton/engine/backend/prefill.py)
  - `EmbeddingWeights` 现在从加载的 BF16 `text_embedding + text_projection` 重新计算 `tts_pad/bos/eos`
- [export_01_embeddings.py](/home/rime/workspace/Qwen3-TTS-Triton/scripts/export/export_01_embeddings.py)
  - 导出现在使用目标数据类型模块计算保存的特殊嵌入以实现运行时奇偶性
- [test_prefill_builder.py](/home/rime/workspace/Qwen3-TTS-Triton/tests/unit/test_prefill_builder.py)
  - 添加了运行时重新计算特殊嵌入的回归测试

验证：
- `scripts/python/greedy_punish_parity.py`
  - 测试前缀上官方 talker == 手动 talker
- `scripts/python/greedy_punish_stagewise_compare.py`
  - 测试前缀上首次 talker 分歧 = `-1`
- `scripts/python/greedy_punish_mode_matrix.py`
  - `engine_stepwise == engine_manual`
  - 两者现在在测试前缀上匹配 `official_generate`

残留说明：
- `official_stepwise` 仍与 `official_generate` 在 pad 阶段后期分歧（测试样本中的 step9），但这与已修复的 step2 引擎不匹配是分开的

### 10. 特殊嵌入修复后，测试前缀上采样官方与手动展开也匹配

实验：`scripts/python/official_vs_manual_rollout.py`

运行时特殊嵌入修复后观察到：
- 短文本 `人工智能正在深刻改变我们的世界。`
  - `official_len=31`，`manual_len=32`
  - `first_divergence=-1`
  - 公共前缀精确匹配
- 更长文本 `人工智能正在深刻改变我们的世界。从语音识别到自然语言处理，AI的应用已经渗透到生活的方方面面。`
  - `official_len=63`，`manual_len=64`
  - `first_divergence=-1`
  - 公共前缀精确匹配

解读：
- 特殊嵌入修复不仅改善了 greedy 奇偶性，还改善了测试文本上真实的采样展开路径
- 剩余的长度不匹配现在只是手动侧的额外尾部 token，而非早期 token 内容分歧

### 11. 修复后长段采样奇偶性在原始 4a 和 story 样式用例上也成立

实验：`scripts/python/official_vs_manual_rollout.py`

运行时特殊嵌入修复后观察到：
- 4a `LONG_TEXT`，`max_steps=256`，`seed=1234`
  - `trailing_len=80`
  - `official_len=255`，`manual_len=256`
  - `first_divergence=-1`
- 4a `LONG_TEXT`，`max_steps=256`，`seed=2025`
  - `trailing_len=80`
  - `official_len=255`，`manual_len=256`
  - `first_divergence=-1`
- `tests/data/story.txt`，`max_steps=512`，`seed=1234`
  - `trailing_len=1217`
  - `official_len=511`，`manual_len=512`
  - `first_divergence=-1`

解读：
- 采样官方/手动奇偶性改善不限于短前缀
- 在测试的长用例上，修复消除了完整测试公共前缀上的早期内容分歧
- 对于 4a，官方和手动在测试预算内仍无法终止，因此剩余长段问题现在看起来是官方侧的（或超出本地 PyTorch 展开奇偶路径）

### 12. Greedy+punish 逐步奇偶性现在延伸通过 4a 文本阶段并进入 pad 阶段

实验：`scripts/python/greedy_punish_mode_matrix.py`

运行时特殊嵌入修复后在 4a `LONG_TEXT` 上观察到：
- `max_steps=64`
  - `official_stepwise == engine_stepwise == engine_manual`
  - `official_generate` 仅在 step `63` 首次分歧
- `max_steps=128`
  - `official_stepwise == engine_stepwise == engine_manual`
  - `official_generate` 仅在 step `127` 首次分歧

解读：
- 修复后，逐步引擎奇偶性保留到远超早期短前缀失败
- 共享的 `official_stepwise / engine_stepwise / engine_manual` 路径在文本消费和 pad 阶段延续上保持对齐
- 剩余差异在官方高级 `generate()` 和显式逐步重放之间，而非引擎展开和官方逐步行为之间

### 13. 修复后独立引擎重跑在 4a 或 story 上不显示明显的长文本长度膨胀

实验：
- `tests/tools/run_engine_long_case.py --case 4a --speaker Serena`
- `tests/tools/run_engine_long_case.py --case story --speaker Serena`

在修复的运行时上观察到：
- 4a 端到端结果
  - 会话 `longtext-medium-serena-postfix`
  - 总音频 `30.96s`
  - 现有保存的比较 WAV `workspace/audio_samples/engine/test4a_long_text_medium.wav` 为 `30.64s`
- story 端到端结果
  - 会话 `longtext-story-postfix`
  - 总音频 `392.80s`
  - 现有保存的比较 WAV `workspace/audio_samples/engine/test4d_story.wav` 为 `395.52s`
- story 用例的引擎日志
  - 会话正常清理，`segments=21/21`
  - 所有记录的段报告 `overflow=False`

解读：
- 在这些代表性端到端重跑上，修复的分支不重现明显的失控长度失败
- 如果幻觉仍然可听，下一个有用的复现应该针对确切的违规文本/说话者/请求路径并在该点捕获转储

### 14. 真实 4a greedy+punish 转储仍与 ONNX 参考在 CP 尾部分歧

实验：
- `workspace/engine_dumps/4a_greedy_dump_fix_20260416_202542`
- 在不良转储步骤 `000003`、`000004`、`000008` 上的融合 ONNX 重放

观察到：
- `updated_token_counts` 在转储和 ONNX 重放之间仍匹配
- `codec_0` 可以仍匹配而 CP 尾部已经不同
- 代表性步骤 `000003`
  - 转储完整 codec：
    `[1085, 1989, 550, 206, 767, 1943, 1731, 1977, 327, 948, 294, 269, 761, 1761, 224, 412]`
  - ONNX / PyTorch 参考：
    `[1085, 1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 693, 224, 179]`
- 代表性步骤 `000008`
  - 转储完整 codec：
    `[44, 1558, 1582, 1837, 624, 676, 1699, 1, 287, 223, 1017, 1062, 77, 1320, 499, 1365]`
  - ONNX / PyTorch 参考：
    `[44, 581, 1999, 1280, 624, 1928, 1699, 898, 1199, 117, 682, 454, 953, 118, 1050, 551]`

解读：
- 修复导出的特殊嵌入 bug 后，剩余的 4a 不良转储不再通过本地 PyTorch 展开不匹配来解释
- 分歧现在专门出现在 TRT 执行路径中，首先在 CP 尾部而非惩罚簿记

### 15. 独立 `code_predictor_unrolled` 显示相同的 BF16 TRT 问题，而 FP32 TRT 匹配 ORT

实验：
- 使用 TensorRT 10.15.1 构建（`nvcr.io/nvidia/tritonserver:26.02-py3`）
  - `code_predictor_unrolled_bf16.engine`
  - `code_predictor_unrolled_fp32.engine`
- 随机试验奇偶性：
  - `python tests/tools/verify_code_predictor_trt.py --engine ...code_predictor_unrolled_bf16.engine --trials 10`
  - `python tests/tools/verify_code_predictor_trt.py --engine ...code_predictor_unrolled_fp32.engine --trials 10`
- 不良状态奇偶性：
  - 在 `000003`、`000004`、`000008` 上以转储模式运行相同脚本

观察到：
- 随机输入上的独立 BF16 TRT vs ORT：
  - 不匹配 `7 / 10`
- 随机输入上的独立 FP32 TRT vs ORT：
  - 不匹配 `0 / 10`
- 在真实不良转储状态 `000003` 上
  - ORT 尾部：
    `[1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 693, 224, 179]`
  - 独立 BF16 TRT 尾部：
    `[1989, 550, 206, 767, 1943, 1731, 1977, 327, 948, 294, 269, 761, 1761, 224, 412]`
  - 独立 FP32 TRT 尾部：
    `[1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 693, 224, 179]`
- 在真实不良转储状态 `000008` 上
  - 独立 BF16 TRT 仍与 ORT 分歧（首次尾部分歧在 stage `7`）
  - 独立 BF16 TRT **不**精确匹配融合 TRT 转储尾部
  - 独立 FP32 TRT 精确匹配 ORT
- 在真实不良转储状态 `000004` 上
  - 独立 BF16 TRT 仍与 ORT 分歧，但不精确匹配融合转储尾部
  - 独立 FP32 TRT 匹配 ORT

解读：
- 剩余问题不是"官方缓存 CP vs 我们的展开 CP"
- 剩余问题不是"融合 ONNX 导出语义"
- 剩余问题不是"仅惩罚参数"
- 目前最强的解释是：
  - **`code_predictor_unrolled` 的 TensorRT BF16 执行本身在数值/语义上不稳定**
  - 融合 BF16 引擎继承了该 CP 不稳定性
  - FP32 TRT 是有效对照：在测试的独立 CP 输入上它与 ORT 精确匹配

### 16. 直接 `官方 BF16` vs `TRT BF16` 比较仍显示 TRT 不匹配

重要更正：
- `ORT(fp32)` 仅是高精度对照，不是生产比较的最终仲裁者
- 更相关的问题是 `TRT BF16` 在相同 CP 输入上是否匹配官方 PyTorch BF16 行为

实验：
- 在 `torch.bfloat16` 中加载官方本地模型
- 比较：
  - BF16 autocast 下的官方缓存 CP
  - 独立 TRT `code_predictor_unrolled_bf16.engine`
- 两边使用相同的固定输入：
  - 随机 `past_hidden + codec_token_0`
  - 从融合转储输入重构的 4a 不良状态输入，以隔离 CP 分支

在随机 CP 输入上观察到：
- 测试种子 `42..49`
- 官方缓存 BF16 vs TRT BF16 在 `7 / 8` 次试验上不匹配
- 代表性种子 `42`，`codec_token_0=[1809]`
  - 官方缓存 BF16：
    `[841, 1591, 305, 889, 943, 1212, 931, 61, 89, 266, 16, 637, 175, 242, 928]`
  - TRT BF16：
    `[841, 1591, 305, 1468, 490, 245, 559, 880, 89, 1014, 481, 1190, 1105, 831, 313]`

在 4a 不良状态派生的 CP 输入上观察到：
- 步骤 `000003`
  - 官方缓存 BF16：
    `[1989, 550, 206, 767, 1943, 287, 176, 433, 948, 294, 1167, 761, 369, 224, 179]`
  - TRT BF16：
    `[1989, 550, 206, 767, 1943, 1731, 1977, 327, 948, 294, 269, 761, 1761, 224, 412]`
  - 首次分歧在 stage `5`
- 步骤 `000004`
  - 官方缓存 BF16：
    `[1542, 1628, 1804, 1774, 1788, 16, 403, 113, 924, 299, 947, 1183, 815, 891, 29]`
  - TRT BF16：
    `[1542, 1628, 271, 21, 1297, 39, 403, 910, 924, 2026, 640, 481, 1007, 1229, 32]`
  - 首次分歧在 stage `2`

解读：
- 即使从最终基准角色中移除 `ORT(fp32)`，`TRT BF16` 在相同 CP 输入上仍无法匹配官方 BF16
- 因此早期结论在更严格的比较中仍然成立：
  - 剩余问题仍在 CP 分支的 TRT BF16 执行中，而非仅参数选择

### 17. 在原始 greedy+punish 转储路径上，`updated_token_counts` 匹配官方 BF16 但 `full_codec` 不匹配

工具更新：
- 修复了 `scripts/export/talker_unified_modules.py`，使 BF16 重放使用 FP32 softmax 并在值矩阵乘法前转换回来
- 使 `scripts/python/analyze_engine_dump.py` 在转储不存储可选输出如 `hidden/logits` 时跳过它们

实验：
- 通过以下重放真实 `4a_greedy_dump_fix_20260416_202542` 转储：
  - 原始 TRT 引擎 `talker_code2wav_fused.engine`
  - `torch.bfloat16` 中的官方融合 PyTorch 重放
- 命令形式：
  - `python scripts/python/analyze_engine_dump.py --dump ... --dtype bfloat16 --model-path workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice`

观察到：
- 原始 TRT 重跑在 `000003`、`000004`、`000008` 上精确重现保存的转储输出
- 官方 BF16 重放**不**重现 `full_codec`
  - `000003`：`num_mismatch=13`
  - `000004`：`num_mismatch=26`
  - `000008`：`num_mismatch=21`
- 但官方 BF16 重放确实重现 talker 侧簿记：
  - `updated_token_counts: match=True` 在所有三个转储上
  - `talker_new_kv` 余弦保持在 `0.99994 ~ 0.99995` 左右
- `full_codec` 中代表性首次分歧位置：
  - `000003` row0：首次差异在索引 `6`
  - `000004` row0：首次差异在索引 `3`
  - `000008` row0：首次差异在索引 `1`

解读：
- 在 greedy+punish 下，**talker/token0 + 重复惩罚簿记是对齐的**
- 剩余不匹配在 **token0 之后的 CP 尾部 token**，而非 `token_counts` / `penalty`
- 因此"两侧都变静音，所以这必然只是参数问题"**不**受 token 证据支持：
  - 实际 `full_codec` 序列仍与官方 BF16 不对齐

### 18. `hidden/logits -> fp32` 调试融合引擎不是行为保持的

实验：
- 构建调试引擎：
  - `workspace/exported/custom-1.7b/talker_code2wav_fused.hiddenlogits_fp32.engine`
- 通过以下重跑相同的 `4a_greedy_dump_fix` 转储：
  - 原始融合引擎
  - `hidden/logits` 输出强制为 `fp32` 的调试融合引擎

观察到：
- 调试引擎本身更改 `full_codec`：
  - `000003`：与原始首次差异在索引 `2`
  - `000004`：与原始首次差异在索引 `3`
  - `000008`：与原始首次差异在索引 `0`
- 因此调试引擎尾部不等于原始生产引擎尾部，即使在相同输入上

解读：
- 强制 `hidden/logits` 输出格式为 `fp32` 会更改 TRT 构建器/运行时数值，足以改变 token 决策
- 因此，使用调试引擎导出的 `hidden` 的比较仅作为**诊断探针**有用
- 它们**不得**被视为原始融合引擎行为的基准

## 历史修复前收窄

### 在更正的官方基准下，首次剩余分歧是 talker step1 的 CP stage2

实验：`scripts/python/greedy_punish_stagewise_compare.py`

对于 step2 的 talker 输出分歧：
- 官方 talker 处理后 top2 以 `450 > 1714` 开始
- 手动 talker 处理后 top2 以 `1714 > 450` 开始
- `small_to_mtp_projection` 后官方/手动 CP 入口差异：
  - 最大差异 ≈ `0.25`
  - 平均差异 ≈ `0.01417`
  - 余弦相似度 ≈ `0.999866`
- CP stage0 匹配
- CP stage1 匹配
- **CP stage2 分歧**

在 talker step1 观察到：
- 官方 CP stage0 token：1989
- 我们的 CP stage0 token：1989
- 官方 CP stage1 token：550
- 我们的 CP stage1 token：550
- 官方 CP stage2 token：1815
- 我们的 CP stage2 token：206

这仍是更正后官方基准下共享公共前缀区域内已知的首次 argmax 翻转点。

### CP stage2 分歧看起来像接近平局翻转，而非灾难性分布崩溃

对于 CP stage2 logits（talker step1）：
- 官方 top10：`[1815, 206, 1810, 459, 1801, 527, 1166, 350, 1376, 1440]`
- 我们的 top10：`[206, 1815, 459, 1810, 1801, 527, 1166, 350, 1376, 2008]`
- 官方 top1-top2 边距：`0.25`
- 我们的 top1-top2 边距：`0.0`

解读：
- 候选集几乎相同
- top1/top2 顺序翻转
- 这看起来更像是**接近平局 argmax 敏感性**而非完全错误的分布

### CP 入口输入在官方投影后非常接近

在 `small_to_mtp_projection` **之后**比较官方 `cp.model` 输入与我们手动构造的 CP 输入：

对于 talker step1 CP 入口：
- 形状匹配：`[1, 2, 1024]`
- 最大差异 ≈ 0.25
- 平均差异 ≈ 0.0142
- 余弦相似度 ≈ 0.999865

因此：
- CP 入口无明显畸形
- 输入 CP 的 token 此时正确
- 状态非常接近但不位相同

### CP stage0 和 stage1 logits 也非常接近

对于 talker step1：
- CP stage0 logits 余弦相似度 ≈ 0.99985
- CP stage1 logits 余弦相似度 ≈ 0.99968
- CP stage0 top1 匹配
- CP stage1 top1 匹配

这加强了分歧不是在 CP 入口立即发生的证据。

## 调查期间发现的重要更正

### 早期"官方尾部切片 bug"是由使用错误提示契约调用裸核心模型导致的

支持的官方包装器构建：
- `"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"`

在该提示下，`input_id[:, 4:-5]` 正确恢复完整文本剩余部分。

因此：
- 支持的官方路径**不**存在之前声称的尾部 bug
- 只有使用短提示的裸核心路径作为黄金基准是无效的

## 已排除的内容（部分或完全）

### 强排除
- 与 HF 处理器的重复惩罚公式不匹配（在隔离比较中精确匹配）
- 展开式 CP vs 缓存 CP 作为测试流形采样情况的主要问题
- 支持的官方提示/尾部不匹配作为 greedy+punish 奇偶失败的主要来源
- HF 生成循环管道作为短段奇偶失败的主要来源
- 手动 `talker.model / cp.generate / talker.model` 解码状态管道作为 step2 分歧的主要来源
- 使用相同官方 `talker.forward()` 输入时 talker step0 或 talker step1 的即时灾难性不匹配
- 更正 `small_to_mtp_projection` 后 CP 输入维度完全错误
- 导出的文本嵌入/文本投影/codec 嵌入权重作为测试短段 greedy 不匹配的来源

### 未排除/仍在调查中
- 为什么 `official_stepwise` / `engine_stepwise` 仍与高级 `official_generate` 在后期或终端测试步骤分歧
- 本地 PyTorch 展开奇偶修复后端到端独立/TRT 服务路径是否仍显示幻觉
- 官方仓库长段参数不稳定看起来仍然真实，需要与任何服务侧问题分开

## 当前最佳证据支持陈述

当前最强证据是：

1. 在支持的官方提示下，prefill 和尾部文本注入与引擎路径对齐。
2. 早期短段 greedy+punish 奇偶失败追溯到微小的导出 `tts_bos/eos/pad` 增量。
3. 从加载的 BF16 模块重新计算这些特殊嵌入修复了该引擎侧 prefill bug。
4. 修复后，短前缀 greedy+punish 奇偶性恢复。
5. 相同修复后，4a `LONG_TEXT` greedy+punish 逐步奇偶性也保持至少 128 个测试步骤，包括 pad 阶段，`official_stepwise == engine_stepwise == engine_manual`。
6. 相同修复后，测试的短、中、4a 和 story 样式前缀上采样官方 vs 手动展开匹配；本地 PyTorch 手动链中未重现早期内容分歧。
7. 在 4a 上，即使手动奇偶性恢复，官方采样流式在测试的 256 步预算内仍不发射 EOS。
8. 剩余已知不匹配现在在高级 `official_generate()` 和显式逐步重放之间的后期或终端测试步骤，这与已修复的引擎 prefill 问题分开。
9. 修复后的独立引擎代表性 4a/story 用例重跑正常完成，时长接近现有保存的输出，无记录的段溢出。

因此问题目前更像是：
- 一个已修复的引擎侧 prefill 奇偶 bug，由导出时特殊嵌入导致
- 官方高级 `generate()` 与显式逐步重放之间的剩余差异
- 引擎/本地奇偶恢复后仍存在的官方长段流式+采样不稳定问题
- 以及，如果用户侧幻觉仍存在，可能需要精确用例服务/运行时复现，而非更多通用奇偶追踪

## 调查期间创建的有用脚本

- `scripts/python/cp_sampled_parity.py`
  - 官方缓存 CP vs 展开 CP 奇偶性
- `scripts/python/pytorch_streaming_baseline.py`
  - PyTorch bf16 流式基准
- `scripts/python/greedy_punish_parity.py`
  - 本地官方 vs 手动 greedy+punish 奇偶性
- `scripts/python/official_vs_manual_rollout.py`
  - 官方 vs 手动展开比较
- `scripts/python/replay_official_talker_stepwise.py`
  - 带显式 `position_ids` / `cache_position` 的官方逐步重放
- `scripts/python/trace_official_streaming.py`
  - 带签名保留钩子的官方追踪
- `scripts/python/compare_prefill_paths.py`
  - prefill/trailing 的支持-vs-裸官方提示比较
- `scripts/python/greedy_punish_stagewise_compare.py`
  - 更正的官方基准 vs 手动引擎路径，greedy+punish 下逐阶段比较
- `scripts/python/greedy_punish_mode_matrix.py`
  - 官方 generate / 官方逐步 / 引擎逐步 / 引擎手动模式矩阵
- `scripts/python/compare_live_vs_exported_prefill.py`
  - 原始导出特殊嵌入 vs 运行时重新计算特殊嵌入 vs 实时模型奇偶性

## 下一步建议

现在最有价值的实验是：

- 如果仍有不良用户侧用例，通过修复的独立/TRT 服务器重现确切不良用例
- 为该确切会话启用转储捕获，并将其 codec/状态演进与本地 PyTorch `official_stepwise` / `engine_stepwise` 追踪比较
- 确定任何剩余症状是否来自：
  - 官方高级长段不稳定性，
  - 服务层分段/滚动行为，
  - 或已修复 prefill 路径之外的 TRT/下游解码差异

目标是回答：
1. 之前修复的 prefill bug 是否是主要的引擎侧因素，
2. 剩余问题现在是否仅可在确切不良用例上复现，而非通用 4a/story 回归，或
3. 本地展开奇偶恢复后是否仍有独立的服务/运行时问题。

## 2026-06-29 混合精度（CP=fp32）引擎复查

### 背景

构建了一版混合精度引擎：talker backbone 保持 bf16，code_predictor（CP）改为
fp32（`--cp-precision fp32`，对应历史发现 #15 的对照结论）。预期 CP 数值不稳定
被消除后流式幻觉应消失，但 `short` badcase 仍间歇性幻觉（~10s 句子膨胀到 40.32s
= 504 步 `max_seq_len(512)` overflow、无自然 EOS）。

### 复现方法（确定性 session_id → 确定性种子）

- 引擎采样种子 = `blake2b(base_seed=0, session_id, segment_idx)`
  （`engine/backend/executor.py:_stable_sampling_seed`）。因此**固定 session_id
  即固定种子即可复现的 rollout**。`tests/repeat_case.py` 给 session_id 追加了时间戳，
  导致不可复现；改用确定性 id 扫描（`workspace/halluc_probe.py`）。
- 扫描 `halluprobe-0001..0040` 即命中可复现幻觉：**`halluprobe-0007` / `0008`
  稳定产出 504 步 overflow**（多次重跑完全一致）。

### dump 捕获

- 通过 compose override（`workspace/compose.dump.yaml`）给 docker 引擎注入
  `ENGINE_DUMP_*` 并挂载 `/dumps`，**保留 `engine.yaml` 采样配置
  （do_sample=true, temperature=0.9, top_k=50, repetition_penalty=1.05）**——
  注意不要用 `run_engine_dump.py` 的 greedy 默认值，否则 rollout 不复现。
- 驱动 `halluprobe-0007` 捕获 1 prefill + 504 decode 共 505 个 `.pt`。dump 输入含
  `gumbel_noise` / `cp_gumbel_noise`，因此原型重放采样**按构造对齐种子**。

### 与原型逐步对比（`workspace/scan_dump_divergence.py`，复活并修复
`analyze_engine_dump.py`）

把每步 dump 的输入喂入官方 PyTorch fused talker（`build_talker_unified_fused_module`），
对比 `full_codec`：

- **首次分歧在 decode step 1，CP stage 7**（stage 0–6 完全一致）。
- 引擎 CP 现在更贴近 **fp32** 原型而非 bf16（例：step3 与 fp32 匹配到 stage 11、
  与 bf16 只到 stage 6；step4 fp32→stage 7、bf16→stage 5）——**说明 CP→fp32 确实
  生效**。
- 但 step 1 stage 7 处 **fp32 与 bf16 原型本身也互不一致**（1641 vs 57），即该
  stage 是高熵近平局点。

### 根因定位（`workspace/compare_step1_hidden.py`，含 hidden/logits 的 step1 dump）

| 对比 | 引擎 vs fp32 原型 | 引擎 vs bf16 原型 |
|------|------|------|
| talker `hidden` | cos=0.99990916，**max_abs=0.295** | cos=0.99989，max_abs=0.25 |
| talker `logits` | cos=0.99997，max_abs=0.21 | cos=0.99997，max_abs=0.19 |
| talker token0(stage0) | id=1995，top1−top2 margin=**3.625**（非平局，匹配） | 同左 |
| `full_codec` 首分歧 | stage 7 | stage 7 |

结论链：

1. **talker backbone（bf16 TRT）的 hidden 不是 bit-exact**：相对 fp32 原型
   max_abs≈0.3（cos 0.9999）；相对 torch bf16 也有 max_abs≈0.25，说明 **TRT bf16
   talker ≠ torch bf16 talker**（kernel/累加差异），引擎 hidden 自成一系，与两个
   原型都不同。
2. talker token0 本身鲁棒（margin 3.6）→ stage 0 一致。
3. 该 ~0.3 的 hidden 扰动传入 CP；CP 虽已 fp32（孤立测试正确，见 #15），但其**输入
   是被 bf16 talker 扰动过的 hidden**，叠加 CP 尾部 stage 7 本就近平局
   （temp=0.9 高熵、fp32/bf16 原型在此都翻转），导致 **step 1 stage 7 argmax 翻转**。
4. 单个翻转的子码级联：step 2 起 talker token0 也开始分歧，rollout 走入永不发 EOS
   的区域 → 504 步 overflow → 40.32s 幻觉。

### 决定性对照：原型不会跑飞，只有 bf16 引擎跑飞（`workspace/prototype_rollout.py`）

用独立的 **fp32** PyTorch rollout 驱动官方 fused talker，**与引擎共享同一前缀状态与
同一随机流**（每步 gumbel 取自引擎 dump，轨迹无关；轨迹相关的反馈
input_embeds/token_counts/past_kv 用原型自身输出，文本嵌入
`text_embed[k]=engine_input_embeds[k]−engine_codec_sum[k−1]` 从 dump 还原）。唯一变量
是 decode 算术（fp32 原型 vs dump 里的 bf16 引擎）。

结果：

- **原型（fp32）在 step 126 自然发 EOS**（≈正常 ~10s，落在健康样本 113–130 chunk 区间内）。
- **引擎（bf16）从不发 EOS，跑到 504 步 cap = 40.32s 幻觉**。
- token0 首次分歧在 step 2（与 step1 CP stage7 翻转级联一致）。

即：**原型不跑飞**。两者起点与随机流完全一致，差别只在 bf16 decode 算术 → 结论是
**这是引擎侧 bf16 精度问题放大成 runaway，而非上游模型不稳定**（至少对该 seed）。
注意原型这里还沿用了引擎的 bf16 prefill KV 作为公共起点仍能正常收敛，说明问题在
**bf16 decode 累积**，不在 prefill。

### 排除 KV-cache 轮转/拷贝/裁剪 bug（`workspace/check_kv_rotation.py`）

仅用 dump 验证"每轮输入是否一致"（不需模型）。对每个 decode step k≥2 检查输入是否
等于上一步输出该有的样子：

- `talker_past_kv[k][..., :L_{k-1}, :] == talker_past_kv[k-1]`（前缀保持）
- `talker_past_kv[k][..., L_{k-1}:, :] == talker_new_kv[k-1]`（增量追加）
- `token_counts[k] == updated_token_counts[k-1]`
- `position_ids` 每步 +1

结果（全 504 步，bf16 正确拷贝应 bit-exact）：

- step1 `past_kv == prefill new_kv`：max_abs=**0.0**
- 全程 talker_past_kv 拷贝误差最坏：**0.0**；token_counts 反馈最坏：**0.0**；
  不一致步数：**0**。
- 第三个反馈输入 `input_embeds`(= 上步 codec_sum + 文本/pad 嵌入)：文本相消
  `input_embeds[k]−codec_sum[k−1]` 在前 ~32 步随文本 token 变化（共 31 个文本 token，
  与日志一致），约 step33 起稳定，pad 阶段(40..504)该 pad 嵌入**恒定**（最大漂移
  0.0039 = bf16 噪声）。

→ **引擎 KV pool 的 scatter/gather/pingpong 轮转是无损的，每轮输入完全一致**。
所以分歧不是"错误拷贝/错误裁剪/精度丢失"导致的输入污染，而是**每步 talker/CP 的
bf16 算术**本身产出不同（再正确地前馈下去）。这也与决定性 rollout 自洽：给同样
（正确轮转的）输入、只换 fp32 算术，就能正常终止。

注：此检查针对 **talker KV**（驱动 codec/EOS、即 runaway 的那条链）。c2w 缓存（带
sliding window + pingpong）只影响 wav 合成、不回馈 talker，故与"无 EOS 跑飞"无关。

### 当前判断

- **CP→fp32 是必要但不充分**。残余分歧由 **bf16 talker backbone 的 hidden 误差**
  从 CP 输入端进入，而非 CP 算术本身。用户"backbone 不应与原型分叉"的假设不成立：
  bf16（且 TRT）backbone 确实与原型分叉（max_abs≈0.3）。
- stage 7 是模型固有的近平局点（fp32/bf16 原型自身在此翻转），因此对任何微小扰动
  都敏感——这是 bf16 放大的固有脆弱性，不是单纯的导出/算子 bug。
- 决定性 rollout 表明 **fp32 decode 能正常终止而 bf16 decode 跑飞**，因此把 talker
  decode 路径提到 fp32（或更高精度）预期可消除该幻觉。
  **（下节实测推翻了这条预期——见"全 fp32 引擎实测"。）**

### 全 fp32 引擎实测：精度不是根因，只是"洗牌"（构建并对比）

构建了一版**全 fp32** fused 引擎（`ENGINE_DTYPE=fp32`，backbone+cp+code2wav 均 fp32，
trtexec 172s，引擎 7.0G，运行时 `I/O dtype consistency check passed: manifest=fp32`），
在**同一组 40 个确定性 session**(`halluprobe-0001..0040`) 上与原 bf16(cp=fp32) 引擎对比：

| 引擎 | 幻觉 session | 比例 |
|------|------|------|
| bf16(cp=fp32) | 0007, 0008, 0027, 0031, 0037 | **5/40** |
| 全 fp32 | 0002, 0006, 0008, 0010, 0029, 0033, 0038 | **7/40** |
| 交集 | **仅 0008** | |

- fp32 **修好了** 4 个(0007/0027/0031/0037)，但**新弄坏了** 6 个
  (0002/0006/0010/0029/0033/0038)。
- 两个集合几乎不相交（只有 0008 共有）；总比例没下降（5→7，n=40 下统计上无差别，
  ~12–18%）。
- 单独看 0007 会被误导：fp32 确实修好 0007（9.44s，与 rollout 预测的 ~126 步吻合），
  但这是**该 seed 的偶然**，不是种群级修复。

**结论修正**：talker 提 fp32 **不能消除流式幻觉**，只是改变了"哪些 seed 跑飞"。
幻觉的本质是**采样/模型层面的不稳定**——temperature=0.9 下，对该短文本约 12–18% 的
随机种子会进入永不发 EOS 的轨迹、跑到 512 cap。任何微小扰动(bf16↔fp32、TRT tactic)
只是把不同 seed 推进/推出"坏吸引盆"，不改变发生率。0008 在 bf16/fp32 下都跑飞，是
与精度无关的固有不稳定（呼应发现 #1/#11：官方流式在部分情况本就不发 EOS）。

**真正该做的缓解方向**（精度无关）：

1. EOS / 长度控制：对 token0 的 EOS 决策降温、或设最大音频步数后强制收尾（已有
   overflow 强制 EOS，但听感差）。
2. 采样策略：降低 talker token0 采样温度 / 调 top_k / 加更强 repetition 控制，压低
   进入坏吸引盆的概率。
3. 运行期检测-重采样：检测到 runaway（步数远超 EMA 预期）就换种子重跑该 segment。
4. 与上游确认官方推荐的流式终止策略。

CP→fp32 仍建议保留（发现 #15：CP bf16 在孤立测试本身数值不稳定），但要明确它**不是**
幻觉的总解。

复现/对比脚本：`workspace/halluc_probe.py`（同一组确定性 seed 扫描）；构建命令
`ENGINE_DTYPE=fp32 bash scripts/bash/build_engines.sh --variant custom-1.7b`。

### 决定性结论：原型本身也有同样的幻觉，是模型/采样参数问题（`workspace/proto_rate.py`）

为区分"原型参数问题"还是"我们导图与原型不一致"，在**纯 fp32 PyTorch** 下跑官方权重
fused talker 的自回归 rollout：每个 session 用引擎相同的种子
（`_stable_sampling_seed(0, "halluprobe-NNNN", 0)`）自己生成 Gumbel 流
（复刻 `_build_sampling_noise`），prefill 状态与文本/pad 嵌入 schedule 与种子无关、复用
halluc_0007 dump，只有 Gumbel 随种子变。统计 504 步内是否自然发 EOS。

三方在**同一组 40 个 seed** 上的幻觉率：

| 实现 | 幻觉 | 比例 | 幻觉 seed |
|------|------|------|------|
| bf16 引擎(cp=fp32) | 5/40 | 12.5% | 0007 0008 0027 0031 0037 |
| 全 fp32 引擎 | 7/40 | 17.5% | 0002 0006 0008 0010 0029 0033 0038 |
| **fp32 PyTorch 原型** | **4/40** | **10%** | **0013 0017 0030 0034** |

- 三者比例统计上一致（10–18%，n=40 噪声内），但幻觉 seed 集合**几乎两两不相交**
  （原型的 0013/0017/0030/0034 不在任何引擎集合里；引擎反复跑飞的 0008 在原型下
  EOS@119 正常）。
- 验证：原型对 0007 自然 EOS@115（与 fp32 引擎 118 步、dump-gumbel rollout 126 步
  同量级；差异来自 prefill 那一次采样 draw 的偏移与 TRT/TF32 vs PyTorch 数值）。

**结论**：**原型本身就以 ~10–18% 的比例跑飞**——这正是用户假设里的"原型参数问题，
之前只是没随机到原型的坏区域"。因此幻觉**不是导图/TRT 引入的不一致**，而是
**temperature=0.9 + top_k=50 这套采样参数下模型固有的不稳定**：总有约 1/6 ~ 1/8 的
随机种子落入"永不发 EOS"的吸引盆，bf16/fp32/TRT/PyTorch 只决定具体哪些 seed 落入，
不改变发生率。这与发现 #1/#11（官方流式在部分情况本就不发 EOS）一致，并把它量化了。

→ 修复必须在**采样/终止策略**层面（见上节 1–4），换精度/查导图都无济于事。

注：纯 fp32 原型 rollout 沿用引擎 bf16 prefill 作为公共起点、且 Gumbel 流相对引擎有
一次 prefill draw 的偏移；这些不影响"原型幻觉率 ≈ 引擎幻觉率"这一比率级结论。

### 终极对照：用官方未改动代码跑"这个自训 checkpoint"，照样幻觉（`workspace/official_baseline.py`）

> **重要更正**：`workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice` 是软链到
> `/home/train/tts/qwen3-tts/trained/zehan/0601_trained_model`——**我们自己训练的
> checkpoint**，不是官方发布权重。引擎、我方串接原型、以及下面这个"官方代码 baseline"
> 用的全是这个自训权重。所以本节验证的是：**这个自训 checkpoint 经过最干净的路径
> （官方未改动推理代码、fp32、无导出/TRT/我方脚本）是否仍幻觉**。官方发布权重未在此对比
> （用户要求先聚焦这个 checkpoint）。

上节的"原型"是**我们脚本串起来的** fused module（官方子模块 + 我们的拓扑/unroll，
即导出对象）。为区分"我们串脚本漏移植了策略"还是"这个 checkpoint 的权重本身有问题"，
直接驱动**官方未改动**的高层 API
`Qwen3TTSModel.generate_custom_voice`(→`Qwen3TTSForConditionalGeneration.generate`)：
fp32、`non_streaming_mode=False`(流式，匹配我方管线)、`max_new_tokens=512`(与引擎
512 cap 对齐)、speaker=`001`(= 我方 `serena` 因不在 spk_id_map 而回退到的默认音色，
见 engine.yaml `default_speaker:"001"`)，40 个种子各 torch seed。

官方采样默认值与我方引擎**完全一致**：do_sample=True, top_k=50, top_p=1.0,
temperature=0.9, repetition_penalty=1.05, subtalker_dosample=True。eos_token_id=2150。

四方在同一组 40 seed 上的幻觉率（>18s 即 runaway）：

（四方都用同一个**自训 checkpoint** `zehan/0601_trained_model`，speaker=`001`）

| 实现 | 幻觉 | 比例 | 幻觉 seed |
|------|------|------|------|
| bf16 引擎(cp=fp32) | 5/40 | 12.5% | 0007 0008 0027 0031 0037 |
| 全 fp32 引擎 | 7/40 | 17.5% | 0002 0006 0008 0010 0029 0033 0038 |
| 我方 fp32 PyTorch 串接原型 | 4/40 | 10% | 0013 0017 0030 0034 |
| **官方未改动代码 + 自训 ckpt(fp32)** | **6/40** | **15%** | **0003 0013 0014 0025 0026 0032** |

- **四方比例统计上一致（10–18%）**；具体坏 seed 集合因采样实现/精度不同而异，但
  官方代码路径与我方串接原型**共享 0013**（两条 PyTorch fp32 路径都判 0013 跑飞），相互印证。
- 官方代码路径 max 时长 40.88s、median 9.72s，与引擎现象完全同构。

**结论（回答用户二分）**：**这个自训 checkpoint 经过官方未改动代码（fp32、无导出/TRT/
我方脚本）照样以 ~15% 跑飞**。因此**不是**"我方漏移植了某段官方策略导致导图/引擎出错"——
我方串接脚本、ONNX 导出、TRT 引擎都**忠实继承**了这个权重的固有行为。问题定位在
**这个自训 checkpoint 的权重 + 官方推荐流式采样参数(temp=0.9/top_k=50/rep_penalty=1.05)**：
约 1/6~1/8 的随机种子无法自然发 EOS。

**尚未回答**：是 Qwen3-TTS 本身就这样，还是**这次训练（0601）把权重训坏了**——需用
**官方发布的 1.7B CustomVoice 权重**（`/home/train/tts/qwen3-tts/official/...`，本地已有）
跑同一测试对比。用户当前要求先聚焦这个 checkpoint，故暂未跑官方权重。

**可行动方向**：换精度/查导图/逐行对官方代码都无效（已验证）；要么在**采样/终止策略**上
缓解（EMA 提前收尾、token0 EOS 降温、runaway 检测重采样），要么如果对比发现是训练训坏的，
**重训/换 checkpoint**。

脚本：`workspace/official_baseline.py`（官方仓库率）、`workspace/proto_rate.py`
（我方串接原型率）、`workspace/halluc_probe.py`（引擎率）。

### 建议的下一步

1. 把 talker backbone（至少 talker→CP 的 hidden / `codec_head` logits 路径）也提到
   fp32，验证 step1 stage7 分歧是否消失（预期：与 fp32 原型逐步对齐）。
2. 若无法全 fp32：评估降低 CP 尾部采样熵（该近平局点由 temp=0.9 触发）对幻觉率的
   影响——但会改变模型行为。
3. 仍需一次**纯 PyTorch rollout**（同文本同种子）确认原型自身在该 seed 上是否也跑飞，
   以区分"引擎放大"与"上游不稳定"。逐步重放因 step≥2 喂的是引擎漂移后的状态，无法
   单独回答此问题。

复现脚本（均在 `workspace/`，gitignored）：`halluc_probe.py`、`compose.dump.yaml`、
`scan_dump_divergence.py`、`compare_step1_hidden.py`、`analyze_engine_dump.py`
（从 git `ae77d68^` 复活并修复 import）。
