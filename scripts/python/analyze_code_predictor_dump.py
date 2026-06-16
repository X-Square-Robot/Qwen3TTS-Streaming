#!/usr/bin/env python3
"""Analyze Code Predictor parity for one dumped engine step.

This script focuses only on the CP branch:
  hidden[:, -1:, :] + codec_token_0 -> codec tail tokens

For one dump file, it compares CP results under:
  - fused TRT output tail (`full_codec[:, 1:]`)
  - standalone PyTorch unrolled CP in float32
  - standalone PyTorch unrolled CP under bf16 autocast
  - official-style cached CP in float32
  - official-style cached CP under bf16 autocast
  - standalone ONNX Runtime CP

It is designed for bad-step debugging, where generic/random CP parity tests
are not sufficient because the real hidden states can land near decision
boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from qwen3tts_tools.common import bootstrap_project_imports

bootstrap_project_imports("repo", "scripts_export", "third_party_qwen")

from engine.backend.executor import TRTEngine
from utils import CodePredictorUnrolled, load_tts_model, resolve_model_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", required=True, help="Path to one .pt dump payload")
    p.add_argument("--model-path", default=None, help="HF model path; inferred from dump/export variant when omitted")
    p.add_argument("--engine-path", default=None, help="Local fused TRT engine path; inferred when omitted")
    p.add_argument("--onnx-path", default=None, help="Standalone code_predictor_unrolled.onnx path; inferred when omitted")
    p.add_argument("--device", default="cuda", help="Device for PyTorch/TRT replay")
    p.add_argument("--report-json", default=None, help="Optional JSON output path")
    return p.parse_args()


def _load_dump(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected dump payload type: {type(payload)}")
    for key in ("metadata", "inputs", "outputs"):
        if key not in payload:
            raise KeyError(f"Dump missing top-level key: {key}")
    return payload


def _infer_variant(meta: Dict[str, Any]) -> Optional[str]:
    for key in ("engine_dir", "weights_dir"):
        raw = str(meta.get(key, "") or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if path.name == "weights":
            return path.parent.name
        if (path / "triton_manifest.json").is_file():
            return path.name
    return None


def _infer_engine_path(meta: Dict[str, Any], arg_value: Optional[str]) -> Path:
    if arg_value:
        return Path(arg_value).expanduser().resolve()

    engine_dir = Path(str(meta.get("engine_dir", "") or "")).expanduser()
    for cand in (
        engine_dir / "talker_code2wav_fused.engine",
        engine_dir / "model.plan",
    ):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError("Could not infer fused TRT engine path from dump metadata; pass --engine-path")


def _infer_onnx_path(meta: Dict[str, Any], arg_value: Optional[str]) -> Path:
    if arg_value:
        return Path(arg_value).expanduser().resolve()

    engine_dir = Path(str(meta.get("engine_dir", "") or "")).expanduser()
    cand = engine_dir / "code_predictor_unrolled.onnx"
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError("Could not infer code_predictor_unrolled.onnx from dump metadata; pass --onnx-path")


def _infer_model_path(meta: Dict[str, Any], arg_value: Optional[str]) -> Path:
    if arg_value:
        return Path(arg_value).expanduser().resolve()

    variant = _infer_variant(meta)
    if not variant:
        raise FileNotFoundError("Could not infer model variant from dump metadata; pass --model-path")
    return Path(resolve_model_path(variant)).expanduser().resolve()


def _build_full_output_names(meta: Dict[str, Any]) -> List[str]:
    names = [
        "wav",
        "codec_sum",
        "full_codec",
        "hidden",
        "logits",
        "updated_token_counts",
        "talker_new_kv",
        "c2w_new_kv",
    ]
    names.extend(meta.get("c2w_conv_output_names", []) or [])
    names.extend(meta.get("c2w_transconv_output_names", []) or [])
    return names


def _rerun_trt(
    dump_payload: Dict[str, Any],
    engine_path: Path,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    engine = TRTEngine(str(engine_path), device)
    engine.load()
    stream = torch.cuda.Stream(device=device)
    inputs = {
        key: value.to(device=device)
        for key, value in dump_payload["inputs"].items()
        if isinstance(value, torch.Tensor)
    }
    out_names = _build_full_output_names(dump_payload["metadata"])
    with torch.cuda.stream(stream):
        out = engine.infer(inputs, out_names, stream)
    stream.synchronize()
    return out


def _top2(logits: torch.Tensor) -> Dict[str, Any]:
    vals, idx = logits.float().topk(2, dim=-1)
    return {
        "top1_id": int(idx[0, 0].item()),
        "top2_id": int(idx[0, 1].item()),
        "top1_val": float(vals[0, 0].item()),
        "top2_val": float(vals[0, 1].item()),
        "gap": float((vals[0, 0] - vals[0, 1]).item()),
    }


def _trace_cp_pytorch(
    cp: CodePredictorUnrolled,
    past_hidden: torch.Tensor,
    codec_token_0: torch.Tensor,
    *,
    use_autocast_bf16: bool,
) -> Dict[str, Any]:
    traces: List[Dict[str, Any]] = []
    output_tokens: List[int] = []

    hidden_in = past_hidden.to(dtype=torch.bfloat16 if use_autocast_bf16 else torch.float32)
    token_0 = codec_token_0.to(device=past_hidden.device)

    ctx = torch.autocast(
        device_type=past_hidden.device.type,
        dtype=torch.bfloat16,
        enabled=use_autocast_bf16,
    )
    with torch.no_grad(), ctx:
        embed_0 = cp.talker_codec_embedding(token_0).unsqueeze(1)
        sequence = torch.cat([hidden_in, embed_0], dim=1)

        for stage in range(cp.num_stages):
            hidden = cp._transformer_forward(cp.projection(sequence))
            logits = cp.lm_heads[stage](hidden[:, -1:, :]).squeeze(1)
            token = logits.argmax(dim=-1)
            token_int = int(token.item())
            output_tokens.append(token_int)

            row = {
                "stage": stage + 1,
                "seq_len": int(sequence.shape[1]),
                "token": token_int,
                "sequence_dtype": str(sequence.dtype),
                "hidden_dtype": str(hidden.dtype),
                "logits_dtype": str(logits.dtype),
            }
            row.update(_top2(logits))
            traces.append(row)

            if stage < cp.num_stages - 1:
                next_embed = cp.codec_embeddings[stage](token).unsqueeze(1)
                sequence = torch.cat([sequence, next_embed], dim=1)

    return {
        "tail_tokens": output_tokens,
        "stages": traces,
    }


def _sample_cp_pytorch(
    cp: CodePredictorUnrolled,
    past_hidden: torch.Tensor,
    codec_token_0: torch.Tensor,
    cp_gumbel_noise: torch.Tensor,
    temperature: torch.Tensor,
    *,
    use_autocast_bf16: bool,
) -> List[int]:
    hidden_in = past_hidden.to(dtype=torch.bfloat16 if use_autocast_bf16 else torch.float32)
    noise_in = cp_gumbel_noise.to(device=past_hidden.device, dtype=torch.float32)
    temp_in = temperature.to(device=past_hidden.device, dtype=torch.float32)

    ctx = torch.autocast(
        device_type=past_hidden.device.type,
        dtype=torch.bfloat16,
        enabled=use_autocast_bf16,
    )
    with torch.no_grad(), ctx:
        out = cp(
            hidden_in,
            codec_token_0.to(device=past_hidden.device),
            cp_gumbel_noise=noise_in,
            temperature=temp_in,
        )
    return out.detach().cpu().reshape(-1).to(torch.int64).tolist()


def _trace_cp_cached(
    cp_model: torch.nn.Module,
    talker_codec_embedding: torch.nn.Module,
    past_hidden: torch.Tensor,
    codec_token_0: torch.Tensor,
    *,
    use_autocast_bf16: bool,
) -> Dict[str, Any]:
    traces: List[Dict[str, Any]] = []
    output_tokens: List[int] = []

    hidden_in = past_hidden.to(dtype=torch.bfloat16 if use_autocast_bf16 else torch.float32)
    token_0 = codec_token_0.to(device=past_hidden.device)

    ctx = torch.autocast(
        device_type=past_hidden.device.type,
        dtype=torch.bfloat16,
        enabled=use_autocast_bf16,
    )

    with torch.no_grad(), ctx:
        last_id_hidden = talker_codec_embedding(token_0).unsqueeze(1)
        outputs = cp_model(
            inputs_embeds=torch.cat((hidden_in, last_id_hidden), dim=1),
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        token = logits.argmax(dim=-1)
        token_int = int(token.item())
        output_tokens.append(token_int)
        row = {
            "stage": 1,
            "seq_len": 2,
            "token": token_int,
            "logits_dtype": str(logits.dtype),
        }
        row.update(_top2(logits))
        traces.append(row)

        past_key_values = outputs.past_key_values
        generation_steps = int(outputs.generation_steps)
        prev_token = token.unsqueeze(1)

        while generation_steps < len(cp_model.lm_head):
            outputs = cp_model(
                input_ids=prev_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                generation_steps=generation_steps,
            )
            logits = outputs.logits[:, -1, :]
            token = logits.argmax(dim=-1)
            token_int = int(token.item())
            output_tokens.append(token_int)
            row = {
                "stage": generation_steps + 1,
                "seq_len": generation_steps + 2,
                "token": token_int,
                "logits_dtype": str(logits.dtype),
            }
            row.update(_top2(logits))
            traces.append(row)

            prev_token = token.unsqueeze(1)
            past_key_values = outputs.past_key_values
            generation_steps = int(outputs.generation_steps)

    return {
        "tail_tokens": output_tokens,
        "stages": traces,
    }


def _run_cp_onnx(
    onnx_path: Path,
    past_hidden: torch.Tensor,
    codec_token_0: torch.Tensor,
) -> List[int]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(
        None,
        {
            "past_hidden": past_hidden.detach().cpu().float().numpy(),
            "codec_token_0": codec_token_0.detach().cpu().numpy().astype(np.int64),
        },
    )[0]
    return out.reshape(-1).astype(np.int64).tolist()


def _first_diff(lhs: List[int], rhs: List[int]) -> Optional[int]:
    for idx, (a, b) in enumerate(zip(lhs, rhs), start=1):
        if int(a) != int(b):
            return idx
    if len(lhs) != len(rhs):
        return min(len(lhs), len(rhs)) + 1
    return None


def _print_trace(
    *,
    label: str,
    fp32_trace: Dict[str, Any],
    bf16_trace: Dict[str, Any],
    cached_fp32_trace: Dict[str, Any],
    cached_bf16_trace: Dict[str, Any],
    sampled_fp32_tail: List[int],
    sampled_bf16_tail: List[int],
    onnx_tail: List[int],
    trt_tail: List[int],
) -> None:
    print(label)
    print("=" * len(label))
    print(f"Unrolled FP32 tail : {fp32_trace['tail_tokens']}")
    print(f"Unrolled BF16 tail : {bf16_trace['tail_tokens']}")
    print(f"Cached FP32 tail   : {cached_fp32_trace['tail_tokens']}")
    print(f"Cached BF16 tail   : {cached_bf16_trace['tail_tokens']}")
    print(f"Sampled FP32 tail  : {sampled_fp32_tail}")
    print(f"Sampled BF16 tail  : {sampled_bf16_tail}")
    print(f"Standalone ONNX    : {onnx_tail}")
    print(f"Fused TRT tail     : {trt_tail}")
    print()
    print(
        "first_diff_vs_cached_fp32:",
        {
            "cached_bf16": _first_diff(cached_fp32_trace["tail_tokens"], cached_bf16_trace["tail_tokens"]),
            "unrolled_fp32": _first_diff(cached_fp32_trace["tail_tokens"], fp32_trace["tail_tokens"]),
            "unrolled_bf16": _first_diff(cached_fp32_trace["tail_tokens"], bf16_trace["tail_tokens"]),
            "onnx": _first_diff(cached_fp32_trace["tail_tokens"], onnx_tail),
            "trt": _first_diff(cached_fp32_trace["tail_tokens"], trt_tail),
        },
    )
    print(
        "first_diff_vs_fused_trt:",
        {
            "sampled_fp32": _first_diff(trt_tail, sampled_fp32_tail),
            "sampled_bf16": _first_diff(trt_tail, sampled_bf16_tail),
            "onnx_argmax": _first_diff(trt_tail, onnx_tail),
            "cached_fp32_argmax": _first_diff(trt_tail, cached_fp32_trace["tail_tokens"]),
        },
    )
    print()

    print("Per-stage")
    print("---------")
    for fp32_row, bf16_row, cached_fp32_row, cached_bf16_row, onnx_token, trt_token in zip(
        fp32_trace["stages"],
        bf16_trace["stages"],
        cached_fp32_trace["stages"],
        cached_bf16_trace["stages"],
        onnx_tail,
        trt_tail,
    ):
        print(
            f"stage={cached_fp32_row['stage']:02d} "
            f"c_fp32={cached_fp32_row['token']:4d} gap={cached_fp32_row['gap']:.6f} "
            f"c_bf16={cached_bf16_row['token']:4d} gap={cached_bf16_row['gap']:.6f} "
            f"u_fp32={fp32_row['token']:4d} gap={fp32_row['gap']:.6f} "
            f"u_bf16={bf16_row['token']:4d} gap={bf16_row['gap']:.6f} "
            f"onnx={int(onnx_token):4d} "
            f"trt={int(trt_token):4d}"
        )
        if (
            cached_fp32_row["token"] != cached_bf16_row["token"]
            or cached_fp32_row["token"] != fp32_row["token"]
            or cached_fp32_row["token"] != bf16_row["token"]
            or cached_fp32_row["token"] != int(onnx_token)
            or cached_fp32_row["token"] != int(trt_token)
        ):
            print(
                f"  cached_fp32_top2=({cached_fp32_row['top1_id']},{cached_fp32_row['top2_id']}) "
                f"vals=({cached_fp32_row['top1_val']:.6f},{cached_fp32_row['top2_val']:.6f})"
            )
            print(
                f"  cached_bf16_top2=({cached_bf16_row['top1_id']},{cached_bf16_row['top2_id']}) "
                f"vals=({cached_bf16_row['top1_val']:.6f},{cached_bf16_row['top2_val']:.6f})"
            )
            print(
                f"  unrolled_fp32_top2=({fp32_row['top1_id']},{fp32_row['top2_id']}) "
                f"vals=({fp32_row['top1_val']:.6f},{fp32_row['top2_val']:.6f})"
            )
            print(
                f"  unrolled_bf16_top2=({bf16_row['top1_id']},{bf16_row['top2_id']}) "
                f"vals=({bf16_row['top1_val']:.6f},{bf16_row['top2_val']:.6f})"
            )
    print()


def main() -> None:
    args = _parse_args()
    dump_path = Path(args.dump).expanduser().resolve()
    dump_payload = _load_dump(dump_path)
    meta = dump_payload["metadata"]
    outputs_cpu = dump_payload["outputs"]

    model_path = _infer_model_path(meta, args.model_path)
    engine_path = _infer_engine_path(meta, args.engine_path)
    onnx_path = _infer_onnx_path(meta, args.onnx_path)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This script currently expects --device cuda")

    trt_out = _rerun_trt(dump_payload, engine_path, device)
    trt_hidden = trt_out["hidden"][:, -1:, :]
    trt_token_0 = trt_out["full_codec"][:, 0]
    trt_tail = trt_out["full_codec"][:, 1:].detach().cpu().numpy().astype(np.int64).reshape(-1).tolist()
    dump_tail = None
    if isinstance(outputs_cpu.get("full_codec"), torch.Tensor):
        dump_tail = outputs_cpu["full_codec"][:, 1:].detach().cpu().numpy().astype(np.int64).reshape(-1).tolist()

    model_fp32 = load_tts_model(str(model_path), device=str(device), dtype=torch.float32)
    model_bf16 = load_tts_model(str(model_path), device=str(device), dtype=torch.bfloat16)

    cp_unrolled_fp32 = CodePredictorUnrolled(
        model_fp32.talker.code_predictor,
        model_fp32.talker.model.codec_embedding,
    ).to(device).eval()
    cp_unrolled_bf16 = CodePredictorUnrolled(
        model_bf16.talker.code_predictor,
        model_bf16.talker.model.codec_embedding,
    ).to(device).eval()

    fp32_trace = _trace_cp_pytorch(cp_unrolled_fp32, trt_hidden.float(), trt_token_0, use_autocast_bf16=False)
    bf16_trace = _trace_cp_pytorch(cp_unrolled_bf16, trt_hidden.to(dtype=torch.bfloat16), trt_token_0, use_autocast_bf16=True)
    cached_fp32_trace = _trace_cp_cached(
        model_fp32.talker.code_predictor.to(device).eval(),
        model_fp32.talker.model.codec_embedding.to(device).eval(),
        trt_hidden.float(),
        trt_token_0,
        use_autocast_bf16=False,
    )
    cached_bf16_trace = _trace_cp_cached(
        model_bf16.talker.code_predictor.to(device).eval(),
        model_bf16.talker.model.codec_embedding.to(device).eval(),
        trt_hidden.to(dtype=torch.bfloat16),
        trt_token_0,
        use_autocast_bf16=True,
    )
    sampled_fp32_tail = _sample_cp_pytorch(
        cp_unrolled_fp32,
        trt_hidden.float(),
        trt_token_0,
        dump_payload["inputs"]["cp_gumbel_noise"].to(device=device),
        dump_payload["inputs"]["temperature"].to(device=device),
        use_autocast_bf16=False,
    )
    sampled_bf16_tail = _sample_cp_pytorch(
        cp_unrolled_bf16,
        trt_hidden.to(dtype=torch.bfloat16),
        trt_token_0,
        dump_payload["inputs"]["cp_gumbel_noise"].to(device=device),
        dump_payload["inputs"]["temperature"].to(device=device),
        use_autocast_bf16=True,
    )
    onnx_tail = _run_cp_onnx(onnx_path, trt_hidden, trt_token_0)

    title = f"CP Dump Analysis: {dump_path.name}"
    print(title)
    print("=" * len(title))
    print(f"dump: {dump_path}")
    print(f"model_path: {model_path}")
    print(f"engine_path: {engine_path}")
    print(f"onnx_path: {onnx_path}")
    print(f"codec_token_0: {int(trt_token_0.item())}")
    if dump_tail is not None:
        print(f"dump_tail == trt_rerun_tail: {dump_tail == trt_tail}")
    print()

    _print_trace(
        label="Parity",
        fp32_trace=fp32_trace,
        bf16_trace=bf16_trace,
        cached_fp32_trace=cached_fp32_trace,
        cached_bf16_trace=cached_bf16_trace,
        sampled_fp32_tail=sampled_fp32_tail,
        sampled_bf16_tail=sampled_bf16_tail,
        onnx_tail=onnx_tail,
        trt_tail=trt_tail,
    )

    report = {
        "dump": str(dump_path),
        "model_path": str(model_path),
        "engine_path": str(engine_path),
        "onnx_path": str(onnx_path),
        "codec_token_0": int(trt_token_0.item()),
        "dump_tail": dump_tail,
        "trt_tail": trt_tail,
        "unrolled_pytorch_fp32": fp32_trace,
        "unrolled_pytorch_bf16": bf16_trace,
        "cached_pytorch_fp32": cached_fp32_trace,
        "cached_pytorch_bf16": cached_bf16_trace,
        "sampled_pytorch_fp32": sampled_fp32_tail,
        "sampled_pytorch_bf16": sampled_bf16_tail,
        "onnx_tail": onnx_tail,
        "first_diff_vs_cached_fp32": {
            "cached_bf16": _first_diff(cached_fp32_trace["tail_tokens"], cached_bf16_trace["tail_tokens"]),
            "unrolled_fp32": _first_diff(cached_fp32_trace["tail_tokens"], fp32_trace["tail_tokens"]),
            "unrolled_bf16": _first_diff(cached_fp32_trace["tail_tokens"], bf16_trace["tail_tokens"]),
            "onnx": _first_diff(cached_fp32_trace["tail_tokens"], onnx_tail),
            "trt": _first_diff(cached_fp32_trace["tail_tokens"], trt_tail),
        },
        "first_diff_vs_fused_trt": {
            "sampled_fp32": _first_diff(trt_tail, sampled_fp32_tail),
            "sampled_bf16": _first_diff(trt_tail, sampled_bf16_tail),
            "onnx_argmax": _first_diff(trt_tail, onnx_tail),
            "cached_fp32_argmax": _first_diff(trt_tail, cached_fp32_trace["tail_tokens"]),
        },
    }
    if args.report_json:
        out_path = Path(args.report_json).expanduser().resolve()
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
