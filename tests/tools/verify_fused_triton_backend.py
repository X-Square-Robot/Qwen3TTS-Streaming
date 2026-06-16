#!/usr/bin/env python3
"""
Direct parity check: local ORT fused ONNX vs Triton TensorRT fused backend.

This bypasses tts_orchestrator BLS and talks to `talker_code2wav_fused` directly,
so it isolates TensorRT engine / Triton backend issues from BLS scheduling logic.

Example:
  mamba run -n qwen3-tts python tests/tools/verify_fused_triton_backend.py \
      --variant custom-1.7b \
      --text "你好，这是一次当前Triton阶段环境的批处理接口验证。" \
      --speaker serena \
      --language Chinese \
      --triton-url localhost:8101 \
      --steps 8
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

REPO_ROOT = bootstrap_tool_imports()

from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def top1_gap(logits: np.ndarray) -> float:
    flat = logits.astype(np.float32).reshape(-1)
    if flat.size < 2:
        return 0.0
    top2 = np.partition(flat, -2)[-2:]
    return float(top2[-1] - top2[-2])


def _normalize_triton_dtype(dtype: str) -> str:
    if dtype.startswith("TYPE_"):
        return dtype[len("TYPE_") :]
    return dtype


def _cast_for_triton(arr: np.ndarray, triton_dtype: str) -> np.ndarray:
    triton_dtype = _normalize_triton_dtype(triton_dtype)
    if triton_dtype == "FP16":
        return arr.astype(np.float16, copy=False)
    if triton_dtype in {"FP32", "BF16"}:
        return arr.astype(np.float32, copy=False)
    if triton_dtype == "INT64":
        return arr.astype(np.int64, copy=False)
    raise RuntimeError(f"Unsupported Triton dtype request: {triton_dtype}")


def _roundtrip_float(arr: np.ndarray, float_dtype: str | None) -> np.ndarray:
    if float_dtype is None or arr.dtype != np.float32:
        return arr
    dt = float_dtype.lower()
    if dt == "fp16":
        return arr.astype(np.float16).astype(np.float32)
    if dt == "bf16":
        return (
            torch.from_numpy(arr.copy())
            .to(dtype=torch.bfloat16)
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
    if dt == "fp32":
        return arr.astype(np.float32, copy=False)
    raise RuntimeError(f"Unsupported ORT float roundtrip dtype: {float_dtype}")


def _split_kv(
    out: Dict[str, np.ndarray],
    num_layers: int,
    device: torch.device,
) -> List[torch.Tensor]:
    kv: List[torch.Tensor] = []
    for i in range(num_layers):
        kv.append(torch.from_numpy(out[f"present_kv_{i}_k"].copy()).to(device, dtype=torch.float32))
        kv.append(torch.from_numpy(out[f"present_kv_{i}_v"].copy()).to(device, dtype=torch.float32))
    return kv


def _split_c2w(
    out: Dict[str, np.ndarray],
    c2w_out_names: List[str],
    device: torch.device,
    sliding_window: int,
) -> List[torch.Tensor]:
    states: List[torch.Tensor] = []
    kv_max = max(1, sliding_window - 1)
    for name in c2w_out_names:
        t = torch.from_numpy(out[name].copy()).to(device, dtype=torch.float32)
        if t.dim() >= 4 and ("past_kv" in name or "present_kv" in name) and t.shape[2] > kv_max:
            t = t[:, :, -kv_max:, :].contiguous()
        states.append(t)
    return states


def _build_c2w_attention_bias(
    batch: int,
    chunk_t: int,
    c2w_past_len: int,
    sliding_window: int,
) -> np.ndarray:
    key_total = min(c2w_past_len + chunk_t, sliding_window)
    if key_total <= 0:
        key_total = 1
    return np.zeros((batch, 1, chunk_t, key_total), dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="custom-1.7b")
    p.add_argument("--text", required=True)
    p.add_argument("--language", default="Chinese")
    p.add_argument("--speaker", default="serena")
    p.add_argument("--triton-url", default="localhost:8101")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--code2wav-sliding-window", type=int, default=72)
    p.add_argument(
        "--ort-float-feed-roundtrip",
        choices=("fp16", "bf16", "fp32"),
        default=None,
        help="Round-trip all float ORT inputs through this dtype before each inference.",
    )
    args = p.parse_args()

    if not args.variant.startswith("custom-"):
        raise ValueError("This script currently supports custom-* variants only.")

    try:
        import tritonclient.grpc as grpcclient
    except ImportError as e:
        raise RuntimeError("Install tritonclient[grpc] in qwen3-tts env.") from e

    exported_dir = REPO_ROOT / "workspace" / "exported" / args.variant
    model_dir = {
        "custom-0.6b": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "custom-1.7b": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }.get(args.variant)
    if model_dir is None:
        raise ValueError(f"Unsupported variant: {args.variant}")

    manifest = json.loads((exported_dir / "triton_manifest.json").read_text())
    onnx_path = exported_dir / "talker_code2wav_fused.onnx"
    weights_dir = exported_dir / "weights"
    tokenizer_dir = REPO_ROOT / "workspace" / "models" / model_dir

    weights = EmbeddingWeights(str(weights_dir), device_id=0)
    tokenizer = load_lightweight_tokenizer(str(tokenizer_dir))
    if tokenizer is None:
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_dir}")

    builder = PrefillBuilder(weights, tokenizer)
    plan = builder.build_plan(
        TaskType.CUSTOM_VOICE,
        text=args.text,
        language=args.language,
        speaker=args.speaker,
    )

    device = weights.device
    inputs_embeds = plan.prefill_embeds
    trailing = plan.trailing
    pad_embed = weights.tts_pad_embed
    batch, seq, _ = inputs_embeds.shape
    position_ids = (
        torch.arange(seq, device=device, dtype=torch.int64)
        .reshape(1, 1, -1, 1)
        .expand(batch, 3, seq, 1)
    )

    num_layers = int(manifest["talker"]["num_layers"])
    num_kv_heads = int(manifest["talker"]["num_kv_heads"])
    head_dim = int(manifest["talker"]["head_dim"])
    codec_eos_id = int(weights.codec_eos_id)
    c2w_in_names = list(manifest["code2wav_fused"]["c2w_state_input_names"])
    c2w_out_names = list(manifest["code2wav_fused"]["c2w_state_output_names"])
    init_shapes = [tuple(s) for s in manifest["code2wav_fused"]["initial_state_shapes"]]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]

    client = grpcclient.InferenceServerClient(url=args.triton_url)
    if not client.is_server_ready():
        raise RuntimeError(f"Triton not ready at {args.triton_url}")
    model_cfg = client.get_model_config("talker_code2wav_fused", as_json=True)["config"]
    triton_input_dtypes = {
        item["name"]: _normalize_triton_dtype(item["data_type"])
        for item in model_cfg["input"]
    }

    def init_c2w() -> List[torch.Tensor]:
        return [torch.zeros(shape, device=device, dtype=torch.float32) for shape in init_shapes]

    def build_feed(
        inp_emb: torch.Tensor,
        pos_ids: torch.Tensor,
        cache_pos: torch.Tensor,
        past_kv: List[torch.Tensor] | None,
        c2w_states: List[torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        past_len = int(past_kv[0].shape[2]) if past_kv else 0
        cur_seq = int(inp_emb.shape[1])
        feed: Dict[str, np.ndarray] = {
            "input_embeds": inp_emb.detach().cpu().float().numpy().astype(np.float32),
            "position_ids": pos_ids.detach().cpu().numpy().astype(np.int64),
            "attention_bias": np.zeros((batch, 1, cur_seq, past_len + cur_seq), dtype=np.float32),
            "cache_position": cache_pos.detach().cpu().numpy().astype(np.float32),
            "c2w_attention_bias": _build_c2w_attention_bias(
                batch=batch,
                chunk_t=int(cache_pos.shape[1]),
                c2w_past_len=int(c2w_states[0].shape[2]) if c2w_states else 0,
                sliding_window=args.code2wav_sliding_window,
            ),
        }
        if past_kv is None:
            for i in range(num_layers):
                feed[f"past_kv_{i}_k"] = np.empty((batch, num_kv_heads, 0, head_dim), dtype=np.float32)
                feed[f"past_kv_{i}_v"] = np.empty((batch, num_kv_heads, 0, head_dim), dtype=np.float32)
        else:
            for i in range(num_layers):
                feed[f"past_kv_{i}_k"] = past_kv[2 * i].detach().cpu().float().numpy().astype(np.float32)
                feed[f"past_kv_{i}_v"] = past_kv[2 * i + 1].detach().cpu().float().numpy().astype(np.float32)
        for name, t in zip(c2w_in_names, c2w_states):
            feed[name] = t.detach().cpu().float().numpy().astype(np.float32)
        return {k: v for k, v in feed.items() if k in input_names}

    def run_ort(feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        ort_feed = {
            name: _roundtrip_float(arr, args.ort_float_feed_roundtrip)
            for name, arr in feed.items()
        }
        return dict(zip(output_names, sess.run(output_names, ort_feed)))

    def run_triton(feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        inputs = []
        for name in input_names:
            arr = feed[name]
            dtype = triton_input_dtypes[name]
            arr = _cast_for_triton(arr, dtype)
            inp = grpcclient.InferInput(name, list(arr.shape), dtype)
            inp.set_data_from_numpy(arr)
            inputs.append(inp)
        outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]
        res = client.infer("talker_code2wav_fused", inputs=inputs, outputs=outputs)
        return {name: res.as_numpy(name) for name in output_names}

    c2w_ort = init_c2w()
    c2w_trt = init_c2w()
    feed0 = build_feed(
        inputs_embeds,
        position_ids,
        torch.zeros(batch, 1, device=device, dtype=torch.int64),
        None,
        c2w_ort,
    )
    ort_out = run_ort(feed0)
    trt_out = run_triton(feed0)

    print("step=0 prefill")
    print(f"  ort_argmax={int(np.argmax(ort_out['logits'][0, -1]))}")
    print(f"  trt_argmax={int(np.argmax(trt_out['logits'][0, -1]))}")
    print(f"  ort_full_codec[:4]={ort_out['full_codec'][0, :4].tolist()}")
    print(f"  trt_full_codec[:4]={trt_out['full_codec'][0, :4].tolist()}")
    print(f"  hidden_cos={cosine(ort_out['hidden'], trt_out['hidden']):.6f}")
    print(f"  logits_cos={cosine(ort_out['logits'], trt_out['logits']):.6f}")
    print(f"  codec_sum_cos={cosine(ort_out['codec_sum'], trt_out['codec_sum']):.6f}")
    print(
        f"  logit_gap_ort={top1_gap(ort_out['logits'][0, -1]):.6f} "
        f"logit_gap_trt={top1_gap(trt_out['logits'][0, -1]):.6f}"
    )
    print(f"  wav_cos={cosine(ort_out['wav'], trt_out['wav']):.6f}")

    past_ort = _split_kv(ort_out, num_layers, device)
    past_trt = _split_kv(trt_out, num_layers, device)
    c2w_ort = _split_c2w(ort_out, c2w_out_names, device, args.code2wav_sliding_window)
    c2w_trt = _split_c2w(trt_out, c2w_out_names, device, args.code2wav_sliding_window)
    next_ort = torch.from_numpy(ort_out["codec_sum"].copy()).to(device, dtype=torch.float32) + trailing[0]
    next_trt = torch.from_numpy(trt_out["codec_sum"].copy()).to(device, dtype=torch.float32) + trailing[0]

    for step in range(1, args.steps + 1):
        pos = torch.full((batch, 3, 1, 1), seq + step - 1, device=device, dtype=torch.int64)
        cache = torch.full((batch, 1), step, device=device, dtype=torch.int64)
        ort_out = run_ort(build_feed(next_ort, pos, cache, past_ort, c2w_ort))
        trt_out = run_triton(build_feed(next_trt, pos, cache, past_trt, c2w_trt))
        ort_tok = int(np.argmax(ort_out["logits"][0, -1]))
        trt_tok = int(np.argmax(trt_out["logits"][0, -1]))
        ort_full0 = int(ort_out["full_codec"][0, 0])
        trt_full0 = int(trt_out["full_codec"][0, 0])
        print(
            f"step={step} ort_tok={ort_tok} trt_tok={trt_tok} "
            f"full0_ort={ort_full0} full0_trt={trt_full0} "
            f"logits_cos={cosine(ort_out['logits'], trt_out['logits']):.6f} "
            f"hidden_cos={cosine(ort_out['hidden'], trt_out['hidden']):.6f} "
            f"codec_sum_cos={cosine(ort_out['codec_sum'], trt_out['codec_sum']):.6f} "
            f"gap_ort={top1_gap(ort_out['logits'][0, -1]):.6f} "
            f"gap_trt={top1_gap(trt_out['logits'][0, -1]):.6f}"
        )
        print(
            f"  full_codec[:4]_ort={ort_out['full_codec'][0, :4].tolist()} "
            f"full_codec[:4]_trt={trt_out['full_codec'][0, :4].tolist()}"
        )
        if ort_tok != trt_tok or ort_full0 != trt_full0:
            print(f"FIRST_DIVERGENCE step={step}")
            return 1
        if ort_tok == codec_eos_id or trt_tok == codec_eos_id:
            print(f"EOS step={step} token={ort_tok}")
            return 0

        past_ort = _split_kv(ort_out, num_layers, device)
        past_trt = _split_kv(trt_out, num_layers, device)
        c2w_ort = _split_c2w(ort_out, c2w_out_names, device, args.code2wav_sliding_window)
        c2w_trt = _split_c2w(trt_out, c2w_out_names, device, args.code2wav_sliding_window)
        text_add = trailing[step] if step < len(trailing) else pad_embed
        next_ort = torch.from_numpy(ort_out["codec_sum"].copy()).to(device, dtype=torch.float32) + text_add
        next_trt = torch.from_numpy(trt_out["codec_sum"].copy()).to(device, dtype=torch.float32) + text_add

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
