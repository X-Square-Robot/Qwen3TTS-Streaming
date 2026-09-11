# 0818 长文本幻觉率三臂评测

该入口与原有 leading-prefix 短文本 sweep 完全独立，不修改服务协议或生产引擎逻辑。
默认试验臂为 current HEAD standalone、冻结的 0818 Triton package，以及同 checkpoint
的官方 PyTorch 实现。所有合成均为 batch=1 串行；三个 trial seed 被编码进三臂共享的
public session ID，从而固定采样轨迹。

## 环境与冻结

ASR/scoring 使用专用虚拟环境：

```bash
scripts/bash/validation/setup_longform_asr_venv.sh \
  /tmp/xinteraction-provider-wheels/funasrnano-0.2.0a6-py3-none-any.whl
```

脚本强制 wheel SHA256 为
`da6ea124ee062b26336ffbbee0f358b9888588cd9091ebf036f11878916d1227`，不会修改
base 环境。Triton 评测镜像不从 PyPI 安装任何包：它由本机不可变的 0818 发布镜像
`4c314475...`（NGC PyTorch 25.10）和原始 Triton 25.10 镜像 `9ff4dd7a...`
组合，保留 torch 2.9.0a0、Triton 2.62.0 和 TRT 10.13.3.9。安全构建与只读启动见
`scripts/bash/validation/longform_runtime.sh`；该脚本不调用 assemble/autorun，会检查
动态库、Python backend、677-token 输入哈希，并在 GPU 非独占时拒绝启动。若设置
`QWEN3TTS_LONGFORM_MODEL_VERSION`，还会校验冻结模型包的发布身份；未设置时不会把某个
具体研究模型身份写死在运行器中。由于历史上
没有发布完整的 v0.1.2a6 Triton 镜像，这个可审计组合运行时也是报告中必须披露的限制。

设定输出目录与解释器：

```bash
OUT=workspace/validation/longform_0818
ASR_PY=workspace/validation/longform_asr_venv/bin/python
QWEN_PY=/home/rime/miniforge3/envs/qwen3-tts/bin/python
WHEEL=/tmp/xinteraction-provider-wheels/funasrnano-0.2.0a6-py3-none-any.whl
```

首次冻结文本、代码、checkpoint、TRT plan、配置、GPU/依赖和 ASR capability：

```bash
PYTHONPATH=client/src:. "$QWEN_PY" -m tools.validation.hallucination.longform init \
  --output-dir "$OUT" --funasr-wheel "$WHEEL"
```

输入按原始 UTF-8 字节读取，不执行 `strip()`。初始化会完整 hash checkpoint shard、
冻结 plan、packaged weights/tokenizer/code，并记录 `add_special_tokens=false` 时的 677
个输入 ID 及哈希；已有输出目录只能以相同文本/seed 续跑。

## 三臂串行采集

每次只启动一个运行时。Current HEAD：

```bash
scripts/bash/validation/longform_runtime.sh current "$OUT"
# 另一个终端等待 http://127.0.0.1:58080/readyz 后：
PYTHONPATH=client/src:. "$QWEN_PY" -m tools.validation.hallucination.longform collect-endpoint \
  --output-dir "$OUT" --arm current_head --endpoint 127.0.0.1:55151
```

默认情况下，只要 GPU 上已有任何计算进程，运行器都会 fail-closed。若实验负责人明确确认
某些常驻上下文不会影响生成，可用精确 PID allowlist 放行；必须同时填写原因。运行器会把
当时观察到的 PID、进程名、显存及原因写入 `runtime/*gpu_process_waiver*`，出现 allowlist
以外的新进程仍会拒绝启动：

```bash
export QWEN3TTS_LONGFORM_ALLOWED_GPU_PIDS=7710,19288
export QWEN3TTS_LONGFORM_GPU_WAIVER_REASON='remote desktop cannot stop; existing model context approved by experiment owner'
```

停止 current 后，构建并启动冻结 Triton（整个 repo 只读挂载）：

```bash
scripts/bash/validation/longform_runtime.sh build-triton-image "$OUT"
scripts/bash/validation/longform_runtime.sh triton "$OUT"
scripts/bash/validation/longform_runtime.sh verify-triton-mount "$OUT"
PYTHONPATH=client/src:. "$QWEN_PY" -m tools.validation.hallucination.longform collect-endpoint \
  --output-dir "$OUT" --arm triton_0818 --endpoint 127.0.0.1:58101 \
  --model-name tts_orchestrator --model-version 2
```

停止 Triton 后执行官方默认臂。封装脚本先检查 GPU 独占，再记录官方运行时；不要传
generation override，这保留 checkpoint 的 sampling defaults、官方
`non_streaming_mode=True` 和 `max_new_tokens=8192`；权重以与 TRT plan 一致的 BF16
加载，当前专用环境未安装 flash-attn，因此明确使用官方实现的 eager attention：

```bash
scripts/bash/validation/longform_runtime.sh official "$OUT"
```

## ASR、句级切片与盲审

ASR 固定中文、FSMN VAD、无热词、partial off、offline/no-pacing、并发 1。每个完整 WAV
及每个 delivered segment 都建立新 WebSocket，并严格要求唯一 `stream_done`；长于 30 秒
不会被跳过。

```bash
PYTHONPATH=client/src:. "$ASR_PY" -m tools.validation.hallucination.longform score \
  --output-dir "$OUT" --funasr-wheel "$WHEEL"
PYTHONPATH=client/src:. "$ASR_PY" -m tools.validation.hallucination.longform diagnostic-report \
  --output-dir "$OUT"
PYTHONPATH=client/src:. "$ASR_PY" -m tools.validation.hallucination.longform prepare-review \
  --output-dir "$OUT"
```

`diagnostic-report` 只在完整的 9-run/342-sentence grid 上生成 ASR、声学与引擎遥测的
预审诊断，并冻结评测器源码和输入哈希。它不能给出“确认严重幻觉率”，也不能评估或
触发条件根因 gate；严重标签、比例比较和 gate 结论只能来自后续盲审与裁决。

只把 `review/public/` 交给审阅人。第一轮填完后：

```bash
PYTHONPATH=client/src:. "$ASR_PY" -m tools.validation.hallucination.longform prepare-second-review \
  --output-dir "$OUT" --round1-csv /path/to/review_round1_filled.csv
PYTHONPATH=client/src:. "$ASR_PY" -m tools.validation.hallucination.longform prepare-adjudication \
  --output-dir "$OUT" --round1-csv /path/to/review_round1_filled.csv \
  --round2-csv /path/to/review_round2_filled.csv
```

第二轮覆盖全部非 `OK` 与固定随机 10% 的 `OK`；分歧清单仍不包含 arm/session/seed。
完成裁决后生成最终 JSON/CSV/Markdown：

```bash
PYTHONPATH=client/src:. "$ASR_PY" -m tools.validation.hallucination.longform finalize-labels \
  --output-dir "$OUT" --round1-csv /path/to/review_round1_filled.csv \
  --round2-csv /path/to/review_round2_filled.csv \
  --adjudication-csv /path/to/adjudication_filled.csv
```

最终严重分子只有 `SINGLE_UNIT_LOOP`、`ABNORMAL_NOISE`、`UNSUPPORTED_SPEECH`。
`UNSCORABLE`、TTS 失败和缺失结果从不按 clean 填补。ASR、声学和 guard/retry/abort
遥测均标记为 diagnostic-only。

## 条件根因分支

只有主报告同时满足绝对差至少 10pp、对称风险比至少 1.5、RD 95% 区间不跨 0、至少
2/3 seed 同方向且无效率规则通过时，才执行冻结 Triton 三次实跑一致的 commit 分组
matched replay；分组数从证据动态冻结（当前正式数据为 9 段），CLI 会在门槛未通过时
硬拒绝。sample 模式依次运行：

分组冻结优先使用能无缺口覆盖全文的真实 codepoint offsets。当前正式 Triton 证据中的
commit offsets 均为 `0/0`，因此只有在三个 seed 的连续 segment ID 和 commit 文本数组逐字
一致、且拼接严格等于原始全文时，才按文本累计回建 offsets，并写入
`provenance=derived_from_exact_commit_text`。部分有效坐标、seed 间差异、缺口或重复都会
fail-closed，不生成 matched 实验。

```bash
scripts/bash/validation/longform_runtime.sh current-matched-sample "$OUT"
PYTHONPATH=client/src:. "$QWEN_PY" -m tools.validation.hallucination.longform \
  collect-matched-endpoint --output-dir "$OUT" --mode sample --arm current_head \
  --endpoint 127.0.0.1:55151 --matched-runtime-confirmation current-sample-config

scripts/bash/validation/longform_runtime.sh triton-matched-sample "$OUT"
PYTHONPATH=client/src:. "$QWEN_PY" -m tools.validation.hallucination.longform \
  collect-matched-endpoint --output-dir "$OUT" --mode sample --arm triton_0818 \
  --endpoint 127.0.0.1:58101 --matched-runtime-confirmation triton-sample-config

scripts/bash/validation/longform_runtime.sh official-matched-sample "$OUT"
```

把上面的 `sample` 换为 `greedy` 可执行全 greedy 哈希对照，随后运行
`greedy-hashes --output-dir "$OUT"`。current 使用 torch 2.10，而冻结 Triton 组合运行时
使用发布原型的 torch 2.9；因此 sample 同参仍分歧时，逐步 Gumbel/runtime 版本是首要
定位变量，不能直接归因 gateway/BLS。若两个 TRT 臂同参一致但均差于官方，才对最小
失败段使用现有 `scripts/python/run_engine_dump.py --session ...`，比较 tokenizer/prefill、
Gumbel、talker logits、codec、CP/EOS 和 code2wav 的首个分歧层。
