**English** | [中文](README.zh-CN.md)

# TensorRT Myelin Minimal Reproduction (dim_count=4 vs stride_order=3)

This reproduction bundle reproduces the following error (TRT 10.15.1):

`MyelinCheckException: tensor.cpp:852: CHECK_EQ(dim_count(), stride_order().size()) failed. LHS: 4 RHS: 3`

and, at the same time, provides a passing control group that changes only one condition.

## Environment

- GPU: NVIDIA (verified on RTX 5090)
- Docker image: `nvcr.io/nvidia/tritonserver:26.02-py3`
- TensorRT: `10.15.1` (`/usr/src/tensorrt/bin/trtexec` inside the image)
- Python env: `conda env qwen3-tts` (used to export the ONNX)

## 1. Generate the Reproduction ONNX

```bash
python tools/repro/repro_myelin_stride_mismatch/build_repro_onnx.py \
  --out-root /tmp/myelin_repro_case
```

This outputs two ONNX files (both trimmed to the `wav` output only):

- fail: `/tmp/myelin_repro_case/dynamic_state/tokenizer/code2wav_decoder_wav_only.onnx`
- pass: `/tmp/myelin_repro_case/static_state_8/tokenizer/code2wav_decoder_wav_only.onnx`

## 2. Run fail + pass

```bash
bash tools/repro/repro_myelin_stride_mismatch/run_repro.sh both /tmp/myelin_repro_case
```

You can also run them separately:

```bash
bash tools/repro/repro_myelin_stride_mismatch/run_repro.sh fail /tmp/myelin_repro_case
bash tools/repro/repro_myelin_stride_mismatch/run_repro.sh pass /tmp/myelin_repro_case
```

Log locations:

- fail log: `/tmp/myelin_repro_case/logs/fail.log`
- pass log: `/tmp/myelin_repro_case/logs/pass.log`

## 2.5 One-command "Clean Environment" Minimal Pipeline (recommended)

```bash
bash tools/repro/repro_myelin_stride_mismatch/minimal_pipeline.sh \
  /tmp/myelin_repro_minimal both
```

This entry point first clears common interfering variables (such as `C2W_DEBUG_*`, `TRT_MYELIN_DISABLE`, `C2W_STATIC_STATE_BATCH`, etc.),
then pins the container version and runs the full fail/pass reproduction chain, so your analysis is not polluted by leftover experimental environments.

## 2.6 One-command "Frozen Environment" Pipeline (for filing bugs)

```bash
bash tools/repro/repro_myelin_stride_mismatch/frozen_pipeline.sh \
  /tmp/myelin_repro_frozen both
```

Compared with `minimal_pipeline.sh`, this entry point does three extra things:

- Starts with an allowlisted environment via `env -i` (minimal environment)
- Records the command list: `/tmp/myelin_repro_frozen/manifest/cmd.txt`
- Records an environment snapshot and file hashes: `env.txt`, `sha256.txt`

This output is suitable to attach directly to an NVIDIA issue, avoiding the "it's caused by your local environment variables" dispute.

## 3. Expected Behavior

### Fail case (dynamic state batch)

The following key lines should appear:

- `ForeignNode[/Slice_1.../Clip]`
- `MyelinCheckException: tensor.cpp:852`
- `LHS: 4`
- `RHS: 3`

and the build should ultimately fail (`Could not find any implementation for node {ForeignNode[/Slice_1.../Clip]}`).

### Pass case (static state batch=8)

The following should appear:

- `&&&& PASSED TensorRT.trtexec`

i.e. the build passes after changing only `conv_state_* / transconv_overlap_*` to a fixed batch=8.

## 4. Variable Comparison (key)

- Failing group: `conv_state_* / transconv_overlap_*` batch is dynamic (min=1, max=8).
- Passing group: the above state batch is fixed to `8`.
- All other input profiles are kept identical (`codes/cache_position/c2w_attention_bias/past_kv_*`).

## 5. Manual PyTorch Small-Graph Search (<=50 nodes)

If you want to submit a "hand-built small graph" to NVIDIA rather than a trimmed version of the production graph, you can use the following script to search automatically:

```bash
python tools/repro/repro_myelin_stride_mismatch/search_pytorch_repro.py \
  --out-root /tmp/myelin_pytorch_search \
  --max-nodes 50 \
  --max-trials 64
```

Output:

- `summary.json`: `/tmp/myelin_pytorch_search/summary.json`
- ONNX for each trial: `/tmp/myelin_pytorch_search/onnx/`
- TRT log for each trial: `/tmp/myelin_pytorch_search/logs/`

Notes:

- This script batch-constructs hand-built PyTorch modules, exports ONNX, runs `onnxsim`, and only runs `trtexec` on candidates with `<=50` nodes.
- Once it hits `MyelinCheckException: tensor.cpp:852` it stops early and prints the matching trial and path to the terminal.
