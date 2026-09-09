#!/usr/bin/env python3
"""Compare one frozen fused decode input across ORT and TRT artifacts.

The input is captured after a real fused Executor prefill.  The same tensors
are then sent to ONNX Runtime and to one or more TensorRT plans, so the report
does not mix sampling, KV construction, or cursor label generation effects
with graph numerical differences.

Example::

    ENGINE_CUDA_GRAPH_DECODE=0 python tools/validation/fused_onnx_trt_parity.py \
        --capture-artifact /models/x2-exported/custom-1.7b \
        --onnx-artifact /models/x2-exported/custom-1.7b \
        --trt-artifact /models/x2-exported/custom-1.7b \
        --trt-artifact /tmp/cp-fp32/workspace/exported/custom-1.7b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.backend.executor import Executor, TRTEngine
from engine.backend.kv_cache_pool import ModelConfig


def _artifact_dtype(root: Path) -> torch.dtype:
    """Read the graph compute dtype used by the capture artifact."""
    manifest_path = root / "triton_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"capture artifact has no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = str(
        manifest.get("engine_dtype")
        or manifest.get("engine_profile", {}).get("engine_dtype")
        or ""
    ).lower()
    values = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return values[value]
    except KeyError as exc:
        raise ValueError(f"unsupported capture artifact engine_dtype={value!r}") from exc


def _validate_cursor_capture_package(root: Path) -> None:
    """Reject a cursor engine whose model-owned head was not packaged with it."""
    manifest = json.loads(
        (root / "triton_manifest.json").read_text(encoding="utf-8")
    )
    native = manifest.get("native_cursor")
    if not isinstance(native, dict) or native.get("enabled") is not True:
        raise RuntimeError(
            f"capture artifact is not cursor-enabled: {root / 'triton_manifest.json'}"
        )
    head = root / "weights" / "qwen3_tts_12hz_la1_seed0.pt"
    if not head.is_file():
        raise RuntimeError(
            "cursor capture artifact is missing model-owned head: "
            f"{head}"
        )
    expected = native.get("head_sha256") or native.get("cursor_head_sha256")
    if isinstance(expected, str) and expected:
        actual = hashlib.sha256(head.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                "cursor capture head hash does not match manifest: "
                f"expected={expected}, actual={actual}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-artifact", type=Path, required=True)
    parser.add_argument("--onnx-artifact", type=Path, required=True)
    parser.add_argument(
        "--torch-model",
        type=Path,
        help="Optional source checkpoint directory for a PyTorch fused reference",
    )
    parser.add_argument(
        "--trt-artifact", type=Path, action="append", required=True,
        help="TRT artifact directory; repeat to compare multiple plans",
    )
    parser.add_argument("--label-count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--random-prefill", action="store_true",
        help="Use a deterministic random prefill to avoid an immediate EOS probe",
    )
    parser.add_argument(
        "--include-prefill", action="store_true",
        help="Also compare the real prefill graph input before decode steps",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a clean JSON report to this path instead of stdout",
    )
    parser.add_argument(
        "--trt-profile",
        type=int,
        default=-1,
        help="Explicit TRT profile; -1 selects 0 for cursor plans and 1 otherwise",
    )
    return parser.parse_args()


def _commit_decode_state(executor: Executor, slot: Any, output: Any) -> None:
    pool = executor.kv_pool
    pool.scatter_talker_kv_delta(
        [slot.slot_id], output.batch_talker_kv, output.original_past_lens
    )
    if output.batch_c2w_kv is not None and not output.eos_flags[0]:
        if slot.c2w_pooled:
            pool.append_c2w_frames([slot.slot_id], output.batch_c2w_kv)
            slot.c2w_len = min(
                slot.c2w_len + 1, executor._config.c2w_sliding_window - 1
            )
        else:
            slot.c2w_kv = output.batch_c2w_kv[:1].clone()
    conv = [tensor[:1] for tensor in (output.batch_c2w_conv or [])]
    transconv = [tensor[:1] for tensor in (output.batch_c2w_transconv or [])]
    if not output.eos_flags[0] and conv and slot.pingpong_ready:
        slot.copy_c2w_and_flip(conv, transconv)
    elif not output.eos_flags[0]:
        slot.c2w_conv_states = [tensor.clone() for tensor in conv]
        slot.c2w_transconv_states = [tensor.clone() for tensor in transconv]
    if output.updated_tc is not None:
        slot.token_counts = output.updated_tc[:1].clone()
    slot.past_len += 1
    if not output.eos_flags[0]:
        slot.frame_idx += 1
    executor.update_cursor_state(slot, output.cursor_outputs)
    if output.codec_sum is not None and not output.eos_flags[0]:
        slot.next_embed = output.codec_sum[:1].clone()
        slot.last_codec_sum = None


def _capture_inputs(
    root: Path,
    label_count: int,
    steps: int,
    random_prefill: bool,
    include_prefill: bool,
) -> list[dict[str, torch.Tensor]]:
    if steps < 1:
        raise ValueError("steps must be positive")
    os.environ["ENGINE_CUDA_GRAPH_DECODE"] = "0"
    _validate_cursor_capture_package(root)
    capture_dtype = _artifact_dtype(root)
    executor = Executor(
        engine_dir=str(root),
        weights_dir=str(root / "weights"),
        max_batch_size=1,
        max_seq_len=512,
        model_config=ModelConfig(dtype=capture_dtype),
        do_sample=False,
    )
    executor.load()
    slot = executor.kv_pool.allocate("fused-parity-probe")
    if slot is None:
        raise RuntimeError("could not allocate a probe slot")
    try:
        labels = torch.arange(1, label_count + 1, device="cpu", dtype=torch.int64)
        executor.set_cursor_text_plan(slot, labels, active=True)
        if random_prefill:
            torch.manual_seed(17)
            embeds = torch.randn(
                (1, 1, executor._config.hidden_size),
                device="cuda",
                dtype=executor._config.dtype,
            ) * 0.01
        else:
            embeds = torch.zeros(
                (1, 1, executor._config.hidden_size),
                device="cuda",
                dtype=executor._config.dtype,
            )
        captured_steps: list[dict[str, torch.Tensor]] = []
        original_infer = executor._fused_engine.infer

        def capture(inputs: dict[str, torch.Tensor], *args: Any, **kwargs: Any):
            torch.cuda.synchronize()
            captured_steps.append(
                {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in inputs.items()
                }
            )
            return original_infer(inputs, *args, **kwargs)

        executor._fused_engine.infer = capture  # type: ignore[method-assign]
        executor.prefill(slot, embeds)
        if not include_prefill:
            captured_steps.clear()
        for _ in range(steps):
            output = executor.launch_decode_step([slot]).wait()
            if output.eos_flags[0]:
                break
            _commit_decode_state(executor, slot, output)
        if not captured_steps:
            raise RuntimeError("decode did not expose fused inputs")
        return captured_steps
    finally:
        if not slot.is_free:
            executor.kv_pool.release(
                slot.slot_id, expected_allocation_epoch=slot.allocation_epoch
            )
        torch.cuda.synchronize()


def _run_trt(
    root: Path, inputs_steps: list[dict[str, torch.Tensor]], profile_idx: int
) -> list[dict[str, torch.Tensor]]:
    engine = TRTEngine(str(root / "talker_code2wav_fused.engine"), torch.device("cuda"))
    engine.load()
    input_names, output_names = engine.get_io_names()
    stream = torch.cuda.current_stream()
    if int(getattr(engine._engine, "num_optimization_profiles", 1)) <= 1:
        profile_idx = 0
    engine.select_optimization_profile(profile_idx, stream)
    result = []
    for inputs in inputs_steps:
        missing = sorted(set(input_names) - set(inputs))
        if missing:
            raise RuntimeError(f"{root}: captured inputs missing {missing}")
        gpu_inputs = {
            name: inputs[name].to(device="cuda", dtype=engine.get_tensor_dtype(name))
            for name in input_names
        }
        raw = engine.infer(gpu_inputs, output_names, stream)
        stream.synchronize()
        result.append({name: value.detach().cpu().clone() for name, value in raw.items()})
    return result


def _run_onnx(
    root: Path, inputs_steps: list[dict[str, torch.Tensor]]
) -> list[dict[str, torch.Tensor]]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(root / "talker_code2wav_fused.onnx"),
        providers=["CPUExecutionProvider"],
    )
    result = []
    for inputs in inputs_steps:
        feed: dict[str, np.ndarray] = {}
        for item in session.get_inputs():
            value = inputs[item.name]
            if "float" in item.type:
                value = value.float()
            feed[item.name] = value.numpy()
        result.append(
            {
                item.name: torch.from_numpy(np.asarray(value))
                for item, value in zip(session.get_outputs(), session.run(None, feed))
            }
        )
    return result


def _run_torch(
    model_root: Path,
    onnx_root: Path,
    inputs_steps: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Run the exported fused contract directly from the source PyTorch model.

    This deliberately reuses the export wrappers.  It is therefore a frozen
    graph-input comparison, not a second implementation of the serving loop.
    """
    export_dir = REPO_ROOT / "scripts" / "export"
    if str(export_dir) not in sys.path:
        sys.path.insert(0, str(export_dir))
    from code2wav_streaming import (
        Code2WavStreamingWrapper,
        num_code2wav_hidden_layers,
    )
    from export_09_talker_code2wav_fused import (
        TalkerCode2WavCursorFusedONNX,
        TalkerCode2WavFusedONNX,
        resolve_fused_tokenizer_path,
    )
    from native_cursor_modules import CursorStreamingStep, build_cursor_head_from_checkpoint
    from talker_unified_modules import build_talker_unified_fused_module
    from utils import (
        load_speech_tokenizer,
        load_tts_model,
        patch_decoder_transconv_for_trt,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_root / "talker_code2wav_fused.onnx"),
        providers=["CPUExecutionProvider"],
    )
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    device = "cuda"
    # Load on CPU first so this diagnostic does not require a second model
    # copy during checkpoint deserialization.
    model = load_tts_model(model_root, device="cpu", dtype=torch.float32)
    tokenizer = load_speech_tokenizer(
        model_root / "speech_tokenizer", device="cpu", dtype=torch.float32
    )
    decoder = tokenizer.decoder.to(device).eval()
    patch_decoder_transconv_for_trt(decoder)
    code2wav = Code2WavStreamingWrapper(decoder, emit_delta_kv=True).to(device).eval()
    talker_fused, _, _, _, _ = build_talker_unified_fused_module(model, device=device)

    cursor_head_path = model_root / "qwen3_tts_12hz_la1_seed0.pt"
    if "cursor_label_ids" in input_names:
        if not cursor_head_path.is_file():
            raise FileNotFoundError(f"cursor-enabled inputs require {cursor_head_path}")
        cursor_head, _ = build_cursor_head_from_checkpoint(str(cursor_head_path))
        fused = TalkerCode2WavCursorFusedONNX(
            talker_fused,
            code2wav,
            num_code2wav_hidden_layers(decoder),
            CursorStreamingStep(cursor_head.to(device).eval()),
            int(inputs_steps[0]["cursor_label_ids"].shape[1]),
        ).to(device).eval()
    else:
        fused = TalkerCode2WavFusedONNX(
            talker_fused,
            code2wav,
            num_code2wav_hidden_layers(decoder),
        ).to(device).eval()

    result: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for inputs in inputs_steps:
            args = []
            for name in input_names:
                value = inputs[name].to(device)
                if value.dtype.is_floating_point:
                    value = value.float()
                args.append(value)
            outputs = fused(*args)
            result.append(
                {
                    name: value.detach().float().cpu().clone()
                    if value.dtype.is_floating_point
                    else value.detach().cpu().clone()
                    for name, value in zip(output_names, outputs)
                }
            )
    del fused, code2wav, decoder, tokenizer, model
    torch.cuda.empty_cache()
    return result


def _stats(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    if reference.dtype.is_floating_point and actual.dtype.is_floating_point:
        delta = actual.float() - reference.float()
        return {
            "exact": bool(torch.equal(reference, actual)),
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "max_abs": float(delta.abs().max().item()),
            "rms": float(torch.sqrt(torch.mean(delta.square())).item()),
        }
    if not reference.dtype.is_floating_point and not actual.dtype.is_floating_point:
        delta = actual.to(torch.int64) - reference.to(torch.int64)
        return {
            "exact": bool(torch.equal(reference, actual)),
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "mismatch_count": int(torch.count_nonzero(delta).item()),
            "max_abs": int(delta.abs().max().item()),
        }
    delta = actual.float() - reference.float()
    return {
        "exact": False,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_abs": float(delta.abs().max().item()),
        "rms": float(torch.sqrt(torch.mean(delta.square())).item()),
    }


def _compare(
    reference: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> dict[str, Any]:
    names = sorted(set(reference) & set(actual))
    return {name: _stats(reference[name], actual[name]) for name in names}


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    inputs = _capture_inputs(
        args.capture_artifact.resolve(),
        args.label_count,
        args.steps,
        args.random_prefill,
        args.include_prefill,
    )
    onnx = _run_onnx(args.onnx_artifact.resolve(), inputs)
    reports: dict[str, Any] = {
        "torch_vs_onnx": [],
        "torch_vs_trt": {},
        "onnx_vs_trt": {},
        "trt_vs_trt": {},
    }
    torch_results: list[dict[str, torch.Tensor]] | None = None
    if args.torch_model is not None:
        torch_results = _run_torch(
            args.torch_model.resolve(), args.onnx_artifact.resolve(), inputs
        )
        reports["torch_vs_onnx"] = [
            _compare(torch_step, ort_step)
            for torch_step, ort_step in zip(torch_results, onnx)
        ]
    trt_results: dict[str, list[dict[str, torch.Tensor]]] = {}
    for root in args.trt_artifact:
        resolved = root.resolve()
        if args.trt_profile >= 0:
            profile_idx = args.trt_profile
        else:
            manifest = json.loads(
                (resolved / "triton_manifest.json").read_text(encoding="utf-8")
            )
            cursor_enabled = bool(manifest.get("native_cursor", {}).get("enabled"))
            profile_idx = 0 if cursor_enabled else 1
        trt = _run_trt(resolved, inputs, profile_idx)
        trt_results[str(resolved)] = trt
        reports["onnx_vs_trt"][str(resolved)] = [
            _compare(ort_step, trt_step) for ort_step, trt_step in zip(onnx, trt)
        ]
        if torch_results is not None:
            reports["torch_vs_trt"][str(resolved)] = [
                _compare(torch_step, trt_step)
                for torch_step, trt_step in zip(torch_results, trt)
            ]
    roots = list(trt_results)
    if len(roots) >= 2:
        baseline = trt_results[roots[0]]
        for root in roots[1:]:
            reports["trt_vs_trt"][root] = [
                _compare(base_step, other_step)
                for base_step, other_step in zip(baseline, trt_results[root])
            ]
    report = json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote parity report: {args.output}", file=sys.stderr)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
