# Manual Tools

`tests/tools/` contains scripts that are useful for validation, benchmarking, audio inspection, and debugging, but are not pytest tests.

Shared cross-tool helper code should live in `scripts/python/qwen3tts_tools/` once more than one manual tool needs it. Keep files in `tests/tools/` focused on test intent and CLI behavior.

When a manual tool and a pytest suite need the same standalone-engine helper logic, put that shared layer in `tests/support/` instead of importing a `tests/e2e/test_*.py` module.

## Primary Tool

`serving_endpoints.py` is the canonical full serving acceptance and benchmark entry point.

```bash
mamba run -n qwen3-tts python tests/tools/serving_endpoints.py --help
mamba run -n qwen3-tts python tests/tools/serving_endpoints.py --targets engine-grpc
mamba run -n qwen3-tts python tests/tools/serving_endpoints.py --targets triton-grpc,triton-http
```

For bare-engine TTFT distribution data:

```bash
mamba run -n qwen3-tts python tests/tools/serving_endpoints.py \
  --targets engine-grpc \
  --skip-single --skip-streaming --skip-custom-instruct \
  --skip-concurrent --skip-long --skip-badcase \
  --ttft-warmup 3 \
  --ttft-samples 30
```

The TTFT report includes:

- sample count and warmup count
- mean, sample variance, population variance, standard deviation, coefficient of variation
- min, p50, p90, p95, max, range
- one fluctuation bar per measured sample, centered on the mean
- JSON details when `--json` is used

## Tool Groups

Serving and endpoint checks:

- `serving_endpoints.py`
- `engine_standalone_benchmark.py`
- `triton_tts_client.py`
- `triton_concurrent_tts.py`

Audio comparison and listening:

- `full_chain_audio_listen.py`
- `compare_official_vs_triton_audio.py`
- `compare_official_vs_fused_onnx.py`
- `generate_audio_compare.py`
- `gen_reference_audio.py`
- `gen_engine_audio.py`
- `long_streaming_listen_ab.py`
- `run_engine_long_case.py`

Export, ONNX, and TensorRT verification:

- `fused_onnx_audio.py`
- `trt_direct.py`
- `verify_fused_triton_backend.py`
- `verify_code2wav_streaming.py`
- `verify_code_predictor_trt.py`
- `verify_e2e.py`
- `verify_e2e_trt.py`
- `verify_e2e_trt_ref.py`
- `verify_multi_variant.py`
- `verify_precision_ort.py`
- `verify_prototype_parity.py`
- `verify_speech_tokenizer_encoder.py`
- `verify_trt_talker.py`
- `assemble_model_repo_check.sh`
- `dockerfile_triton_check.sh`
- `pad_tolerance_experiment.py`
- `greedy_baseline.py`

## Maintenance

Before adding a new tool here, check whether a close neighbor already exists and whether the shared logic belongs in `scripts/python/qwen3tts_tools/`. For a project-wide surface report, run:

```bash
python scripts/python/audit_tooling_surface.py
```
