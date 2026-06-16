# Manual Tools

`tests/tools/` contains scripts that are useful for validation, benchmarking, audio inspection, and debugging, but are not pytest tests.

Shared cross-tool helper code should live in `qwen3_tts_protocol` (types/schemas) or
`scripts/python/qwen3tts_tools/` (repo-internal helpers like path constants).
When a manual tool and a pytest suite need the same standalone-engine helper logic,
put that shared layer in `tests/support/` instead of importing a `tests/e2e/test_*.py` module.

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
- `gen_audio.py`
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
- `verify_onnx_autoregressive.py`
- `assemble_model_repo_check.sh`
- `dockerfile_triton_check.sh`
- `pad_tolerance_experiment.py`
- `greedy_baseline.py`

Prefill and rollout comparison (migrated from scripts/python/):

- `compare_live_vs_exported_prefill.py`
- `compare_prefill_paths.py`
- `official_prefill.py`
- `official_vs_manual_rollout.py`
- `pytorch_streaming_baseline.py`

Code predictor and parity tools (migrated from scripts/python/):

- `cp_sampled_parity.py`
- `greedy_punish_mode_matrix.py`
- `greedy_punish_parity.py`
- `greedy_punish_stagewise_compare.py`
- `suggest_engine_profile.py`

## Maintenance

Before adding a new tool here, check whether a close neighbor already exists and whether the shared logic belongs in `qwen3_tts_protocol` or `scripts/python/qwen3tts_tools/`. For a project-wide surface report, run:

```bash
python scripts/python/audit_tooling_surface.py
```
