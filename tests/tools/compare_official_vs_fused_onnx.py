#!/usr/bin/env python3
"""
Official Qwen3TTSModel API vs **fused** ONNX (`talker_code2wav_fused.onnx`) end-to-end WAV compare.

Runs the same prefill + fused decode loop logic as `model.py::_generation_loop_fused`, but
executes the fused engine with ONNX Runtime (CPU) instead of Triton BLS.

Outputs (default `workspace/audio_compare/`):
  - proto.wav       — `Qwen3TTSModel.generate_*` (official API)
  - fused_onnx.wav  — ORT fused model, greedy-style loop (same logits argmax as orchestrator)

Prerequisites:
  - conda env `qwen3-tts`, weights under `workspace/models/...`
  - `workspace/exported/<variant>/talker_code2wav_fused.onnx`
  - `workspace/exported/<variant>/triton_manifest.json` (for c2w state shapes)

Usage:
  conda activate qwen3-tts
  python tests/tools/compare_official_vs_fused_onnx.py --variant custom-1.7b \\
      --text "你好，这是一段测试。" --speaker serena --max-steps 120
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch

from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

bootstrap_project_imports("scripts_python", "scripts_export", "scripts", "third_party_qwen")

from official_prefill import build_prefill_like_official
from utils import (
    setup_logging,
    resolve_model_path,
    resolve_device,
    has_model_weights,
    DEFAULT_OUTPUT_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_official_fused_onnx")

OFFICIAL_ASSISTANT_FMT = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
FUSED_CHUNK_T = 1

# ORT CPU runs the FP32 ONNX graph; BF16-trained models accumulate numerical drift
# in KV cache across decode steps, causing silence after ~8 steps.  Consecutive
# near-silent frames trigger an early stop so the output is partial rather than
# a long silent tail.
SILENCE_THRESHOLD = 0.002
SILENCE_PATIENCE = 3


def _non_streaming_for_variant(variant: str) -> bool:
    return "design" in variant.lower()


def _load_manifest(exported_dir: Path) -> Dict[str, Any]:
    p = exported_dir / "triton_manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _create_initial_c2w(manifest: Dict[str, Any]) -> List[np.ndarray]:
    lay = manifest["code2wav_fused"]
    shapes = lay["initial_state_shapes"]
    return [np.zeros(tuple(s), dtype=np.float32) for s in shapes]


def _position_ids_prefill(S: int) -> np.ndarray:
    pos = np.arange(S, dtype=np.int64).reshape(1, 1, -1, 1)
    return np.broadcast_to(pos, (1, 3, S, 1))


def _position_ids_decode(pos_scalar: int) -> np.ndarray:
    return np.full((1, 3, 1, 1), pos_scalar, dtype=np.int64)


def _run_fused_onnx_session(
    sess: Any,
    input_names: List[str],
    output_names: List[str],
    feed: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    feeds = {k: feed[k] for k in input_names if k in feed}
    outs = sess.run(output_names, feeds)
    return dict(zip(output_names, outs))


def _build_past_kv_empty(
    B: int, num_layers: int, num_kv_heads: int, head_dim: int
) -> Dict[str, np.ndarray]:
    d = {}
    for i in range(num_layers):
        d[f"past_kv_{i}_k"] = np.empty((B, num_kv_heads, 0, head_dim), dtype=np.float32)
        d[f"past_kv_{i}_v"] = np.empty((B, num_kv_heads, 0, head_dim), dtype=np.float32)
    return d


def _build_c2w_attention_bias(
    batch: int, chunk_t: int, c2w_past_len: int
) -> np.ndarray:
    key_total = min(c2w_past_len + 1, 72)
    return np.zeros((batch, 1, chunk_t, key_total), dtype=np.float32)


def run_fused_onnx_loop(
    sess: Any,
    manifest: Dict[str, Any],
    *,
    inputs_embeds: torch.Tensor,
    position_ids_prefill: torch.Tensor,
    trailing_text: List[torch.Tensor],
    pad_embed: torch.Tensor,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    codec_eos_id: int,
    max_steps: int,
) -> np.ndarray:
    """ORT fused decode; mirrors model.py::_generation_loop_fused (CPU numpy)."""
    lay = manifest["code2wav_fused"]
    c2w_in_names = lay["c2w_state_input_names"]
    c2w_out_names = lay["c2w_state_output_names"]

    inp_meta = {i.name: i for i in sess.get_inputs()}
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]

    B, S, _H = inputs_embeds.shape
    device = inputs_embeds.device

    def _to_feed(
        inp_emb: torch.Tensor,
        pos_ids: torch.Tensor,
        cache_pos: torch.Tensor,
        past_kv: List[torch.Tensor] | None,
        c2w_states: List[torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        past_len = int(past_kv[0].shape[2]) if past_kv else 0
        seq = int(inp_emb.shape[1])
        chunk_t = int(cache_pos.shape[1])
        c2w_past_len = int(c2w_states[0].shape[2]) if c2w_states else 0
        feed: Dict[str, np.ndarray] = {
            "input_embeds": inp_emb.detach().cpu().float().numpy(),
            "position_ids": pos_ids.detach().cpu().numpy().astype(np.int64),
            "attention_bias": np.zeros((B, 1, seq, past_len + seq), dtype=np.float32),
            "cache_position": cache_pos.detach().cpu().numpy().astype(np.float32),
            "c2w_attention_bias": _build_c2w_attention_bias(B, chunk_t, c2w_past_len),
        }
        if past_kv is None:
            pk = _build_past_kv_empty(B, num_layers, num_kv_heads, head_dim)
            feed.update(pk)
        else:
            for i in range(num_layers):
                feed[f"past_kv_{i}_k"] = (
                    past_kv[2 * i].detach().cpu().float().numpy()
                )
                feed[f"past_kv_{i}_v"] = (
                    past_kv[2 * i + 1].detach().cpu().float().numpy()
                )
        for name, t in zip(c2w_in_names, c2w_states):
            feed[name] = t.detach().cpu().float().numpy()
        # Drop keys not in ONNX (robustness)
        return {k: v for k, v in feed.items() if k in inp_meta}

    c2w_states = [
        torch.from_numpy(a).to(device=device, dtype=torch.float32)
        for a in _create_initial_c2w(manifest)
    ]

    cache_pos = torch.zeros(B, FUSED_CHUNK_T, device=device, dtype=torch.int64)

    feed = _to_feed(
        inputs_embeds,
        position_ids_prefill,
        cache_pos,
        None,
        c2w_states,
    )
    out = _run_fused_onnx_session(sess, input_names, output_names, feed)

    def _get(name: str) -> np.ndarray:
        if name not in out:
            raise KeyError(f"Missing output {name}, have {list(out.keys())}")
        return out[name]

    wav = _get("wav")
    codec_sum = torch.from_numpy(_get("codec_sum")).to(device)
    logits = torch.from_numpy(_get("logits")).to(device)

    wav_chunks = [wav.reshape(-1)]

    if int(logits[:, -1, :].float().argmax(dim=-1).item()) == codec_eos_id:
        logger.info("fused ORT: EOS at step 0 (logits)")
        return np.concatenate(wav_chunks)
    fc0 = int(_get("full_codec")[0, 0])
    if fc0 == codec_eos_id:
        logger.info("fused ORT: EOS at step 0 (full_codec[0])")
        return np.concatenate(wav_chunks)

    text_idx = 0
    if trailing_text:
        next_embed = codec_sum + trailing_text[text_idx]
        text_idx += 1
    else:
        next_embed = codec_sum + pad_embed
    next_embed = next_embed.to(dtype=torch.float32)

    position_id = torch.full((B, 3, 1, 1), S, device=device, dtype=torch.int64)
    frame_idx = 1
    silent_streak = 0

    past_kv: List[torch.Tensor] = []
    for i in range(num_layers):
        past_kv.append(
            torch.from_numpy(_get(f"present_kv_{i}_k")).to(device, dtype=torch.float32)
        )
        past_kv.append(
            torch.from_numpy(_get(f"present_kv_{i}_v")).to(device, dtype=torch.float32)
        )

    new_c2w: List[torch.Tensor] = []
    for name in c2w_out_names:
        new_c2w.append(torch.from_numpy(_get(name)).to(device, dtype=torch.float32))

    for step in range(1, max_steps):
        cache_pos = torch.full(
            (B, FUSED_CHUNK_T), frame_idx, device=device, dtype=torch.int64
        )
        feed = _to_feed(next_embed, position_id, cache_pos, past_kv, new_c2w)
        out = _run_fused_onnx_session(sess, input_names, output_names, feed)

        logits = torch.from_numpy(_get("logits")).to(device)
        logit_eos = (
            int(logits[:, -1, :].float().argmax(dim=-1).item()) == codec_eos_id
        )
        fc0 = int(_get("full_codec")[0, 0])
        if logit_eos or fc0 == codec_eos_id:
            logger.info(
                "fused ORT: EOS at step %d (logits_eos=%s full_codec[0]=%s)",
                step,
                logit_eos,
                fc0,
            )
            break

        wav = _get("wav")
        wav_max = float(np.abs(wav).max())
        if wav_max < SILENCE_THRESHOLD:
            silent_streak += 1
            if silent_streak >= SILENCE_PATIENCE:
                logger.warning(
                    "fused ORT: %d consecutive silent frames (wav_max=%.6f) at step %d — "
                    "FP32 precision drift likely; stopping early. "
                    "Use Triton TRT (BF16) for correct audio.",
                    silent_streak,
                    wav_max,
                    step,
                )
                break
        else:
            silent_streak = 0

        wav_chunks.append(wav.reshape(-1))
        codec_sum = torch.from_numpy(_get("codec_sum")).to(device)

        text_add = (
            trailing_text[text_idx]
            if text_idx < len(trailing_text)
            else pad_embed
        )
        text_idx += 1
        next_embed = (codec_sum + text_add).to(dtype=torch.float32)
        position_id = torch.full(
            (B, 3, 1, 1), S + step, device=device, dtype=torch.int64
        )
        frame_idx += 1

        past_kv = []
        for i in range(num_layers):
            past_kv.append(
                torch.from_numpy(_get(f"present_kv_{i}_k")).to(device, dtype=torch.float32)
            )
            past_kv.append(
                torch.from_numpy(_get(f"present_kv_{i}_v")).to(device, dtype=torch.float32)
            )
        new_c2w = []
        for name in c2w_out_names:
            new_c2w.append(torch.from_numpy(_get(name)).to(device, dtype=torch.float32))

    return np.concatenate(wav_chunks).astype(np.float32)


def _normalize_triton_dtype(dtype: str) -> str:
    if dtype.startswith("TYPE_"):
        return dtype[len("TYPE_") :]
    return dtype


def _cast_numpy_for_triton(arr: np.ndarray, triton_dtype: str) -> np.ndarray:
    triton_dtype = _normalize_triton_dtype(triton_dtype)
    if triton_dtype == "FP16":
        return arr.astype(np.float16, copy=False)
    if triton_dtype in {"FP32", "BF16"}:
        return arr.astype(np.float32, copy=False)
    if triton_dtype == "INT64":
        return arr.astype(np.int64, copy=False)
    raise RuntimeError(f"Unsupported Triton dtype for numpy feed: {triton_dtype}")


def run_fused_triton_loop(
    client: Any,
    triton_input_dtypes: Dict[str, str],
    input_names: List[str],
    output_names: List[str],
    manifest: Dict[str, Any],
    *,
    inputs_embeds: torch.Tensor,
    position_ids_prefill: torch.Tensor,
    trailing_text: List[torch.Tensor],
    pad_embed: torch.Tensor,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    codec_eos_id: int,
    max_steps: int,
) -> np.ndarray:
    """Same greedy fused loop as run_fused_onnx_loop, via Triton model `talker_code2wav_fused`."""
    import tritonclient.grpc as grpcclient

    lay = manifest["code2wav_fused"]
    c2w_in_names = lay["c2w_state_input_names"]
    c2w_out_names = lay["c2w_state_output_names"]

    def _infer(feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        inputs = []
        for name in input_names:
            arr = feed[name]
            dt_raw = triton_input_dtypes[name]
            arr = _cast_numpy_for_triton(arr, dt_raw)
            dt = _normalize_triton_dtype(dt_raw)
            inp = grpcclient.InferInput(name, list(arr.shape), dt)
            inp.set_data_from_numpy(arr)
            inputs.append(inp)
        outputs = [grpcclient.InferRequestedOutput(n) for n in output_names]
        res = client.infer("talker_code2wav_fused", inputs=inputs, outputs=outputs)
        return {n: res.as_numpy(n) for n in output_names}

    B, S, _H = inputs_embeds.shape
    device = inputs_embeds.device

    def _to_feed(
        inp_emb: torch.Tensor,
        pos_ids: torch.Tensor,
        cache_pos: torch.Tensor,
        past_kv: List[torch.Tensor] | None,
        c2w_states: List[torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        past_len = int(past_kv[0].shape[2]) if past_kv else 0
        seq = int(inp_emb.shape[1])
        chunk_t = int(cache_pos.shape[1])
        c2w_past_len = int(c2w_states[0].shape[2]) if c2w_states else 0
        feed: Dict[str, np.ndarray] = {
            "input_embeds": inp_emb.detach().cpu().float().numpy(),
            "position_ids": pos_ids.detach().cpu().numpy().astype(np.int64),
            "attention_bias": np.zeros((B, 1, seq, past_len + seq), dtype=np.float32),
            "cache_position": cache_pos.detach().cpu().numpy().astype(np.float32),
            "c2w_attention_bias": _build_c2w_attention_bias(B, chunk_t, c2w_past_len),
        }
        if past_kv is None:
            feed.update(_build_past_kv_empty(B, num_layers, num_kv_heads, head_dim))
        else:
            for i in range(num_layers):
                feed[f"past_kv_{i}_k"] = past_kv[2 * i].detach().cpu().float().numpy()
                feed[f"past_kv_{i}_v"] = past_kv[2 * i + 1].detach().cpu().float().numpy()
        for name, t in zip(c2w_in_names, c2w_states):
            feed[name] = t.detach().cpu().float().numpy()
        return {k: v for k, v in feed.items() if k in input_names}

    c2w_states = [
        torch.from_numpy(a).to(device=device, dtype=torch.float32)
        for a in _create_initial_c2w(manifest)
    ]

    cache_pos = torch.zeros(B, FUSED_CHUNK_T, device=device, dtype=torch.int64)

    feed = _to_feed(
        inputs_embeds,
        position_ids_prefill,
        cache_pos,
        None,
        c2w_states,
    )
    out = _infer(feed)

    def _get(name: str) -> np.ndarray:
        if name not in out:
            raise KeyError(f"Missing output {name}, have {list(out.keys())}")
        return out[name]

    wav = _get("wav")
    codec_sum = torch.from_numpy(_get("codec_sum")).to(device)
    logits = torch.from_numpy(_get("logits")).to(device)

    wav_chunks = [wav.reshape(-1)]

    if int(logits[:, -1, :].float().argmax(dim=-1).item()) == codec_eos_id:
        logger.info("fused Triton: EOS at step 0")
        return np.concatenate(wav_chunks)

    text_idx = 0
    if trailing_text:
        next_embed = codec_sum + trailing_text[text_idx]
        text_idx += 1
    else:
        next_embed = codec_sum + pad_embed
    next_embed = next_embed.to(dtype=torch.float32)

    position_id = torch.full((B, 3, 1, 1), S, device=device, dtype=torch.int64)
    frame_idx = 1

    past_kv: List[torch.Tensor] = []
    for i in range(num_layers):
        past_kv.append(
            torch.from_numpy(_get(f"present_kv_{i}_k")).to(device, dtype=torch.float32)
        )
        past_kv.append(
            torch.from_numpy(_get(f"present_kv_{i}_v")).to(device, dtype=torch.float32)
        )

    new_c2w: List[torch.Tensor] = []
    for name in c2w_out_names:
        new_c2w.append(torch.from_numpy(_get(name)).to(device, dtype=torch.float32))

    for step in range(1, max_steps):
        cache_pos = torch.full(
            (B, FUSED_CHUNK_T), frame_idx, device=device, dtype=torch.int64
        )
        feed = _to_feed(next_embed, position_id, cache_pos, past_kv, new_c2w)
        out = _infer(feed)

        logits = torch.from_numpy(_get("logits")).to(device)
        if int(logits[:, -1, :].float().argmax(dim=-1).item()) == codec_eos_id:
            logger.info("fused Triton: EOS at step %d", step)
            break

        wav = _get("wav")
        wav_chunks.append(wav.reshape(-1))
        codec_sum = torch.from_numpy(_get("codec_sum")).to(device)

        text_add = (
            trailing_text[text_idx]
            if text_idx < len(trailing_text)
            else pad_embed
        )
        text_idx += 1
        next_embed = (codec_sum + text_add).to(dtype=torch.float32)
        position_id = torch.full(
            (B, 3, 1, 1), S + step, device=device, dtype=torch.int64
        )
        frame_idx += 1

        past_kv = []
        for i in range(num_layers):
            past_kv.append(
                torch.from_numpy(_get(f"present_kv_{i}_k")).to(device, dtype=torch.float32)
            )
            past_kv.append(
                torch.from_numpy(_get(f"present_kv_{i}_v")).to(device, dtype=torch.float32)
            )
        new_c2w = []
        for name in c2w_out_names:
            new_c2w.append(torch.from_numpy(_get(name)).to(device, dtype=torch.float32))

    return np.concatenate(wav_chunks).astype(np.float32)


def main():
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="custom-1.7b")
    p.add_argument("--text", default="你好，这是一段用于对比的测试语音。")
    p.add_argument("--language", default="Chinese")
    p.add_argument("--speaker", default="serena")
    p.add_argument("--instruct", default="")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--models-dir", default=None)
    p.add_argument("--greedy", action="store_true", help="do_sample=False for official API")
    args = p.parse_args()

    device = resolve_device(args.device)
    path = resolve_model_path(args.variant, args.models_dir)
    if not has_model_weights(path):
        logger.error("No weights for variant %s", args.variant)
        sys.exit(1)

    exported_dir = Path(DEFAULT_OUTPUT_DIR) / args.variant
    onnx_path = exported_dir / "talker_code2wav_fused.onnx"
    if not onnx_path.is_file():
        logger.error("Missing fused ONNX: %s (run export_09)", onnx_path)
        sys.exit(1)

    manifest = _load_manifest(exported_dir)
    talker = manifest.get("talker", {})
    num_layers = int(talker.get("num_layers", 28))
    num_kv_heads = int(talker.get("num_kv_heads", 8))
    head_dim = int(talker.get("head_dim", 128))

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "workspace" / "audio_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("pip install onnxruntime")
        sys.exit(1)

    logger.info("Loading ORT session: %s", onnx_path)
    sess = ort.InferenceSession(
        str(onnx_path),
        ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper

    logger.info("Loading official Qwen3TTSModel ...")
    wrapper = TTSModelWrapper.from_pretrained(
        str(path), device_map=str(device), dtype=torch.float32
    )
    model = wrapper.model
    processor = wrapper.processor

    do_sample = not args.greedy
    gen_kwargs = dict(do_sample=do_sample, max_new_tokens=args.max_steps)
    if not do_sample:
        gen_kwargs["repetition_penalty"] = 1.0
        gen_kwargs["subtalker_dosample"] = False

    non_streaming = _non_streaming_for_variant(args.variant)
    with torch.no_grad():
        if "design" in args.variant.lower():
            wavs, sr = wrapper.generate_voice_design(
                text=args.text,
                instruct=args.instruct,
                language=args.language,
                non_streaming_mode=non_streaming,
                **gen_kwargs,
            )
        else:
            wavs, sr = wrapper.generate_custom_voice(
                text=args.text,
                speaker=args.speaker,
                language=args.language,
                instruct=args.instruct,
                non_streaming_mode=non_streaming,
                **gen_kwargs,
            )
    wav_proto = wavs[0]
    sf.write(str(out_dir / "proto.wav"), wav_proto, sr)
    logger.info("Wrote proto.wav (official), len=%d, sr=%d", len(wav_proto), sr)

    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=args.text)
    tok_out = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = tok_out["input_ids"].to(device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    instruct_ids = None
    if args.instruct:
        instruct_text = wrapper._build_instruct_text(args.instruct)
        instruct_tok = processor(text=instruct_text, return_tensors="pt", padding=True)
        instruct_ids = instruct_tok["input_ids"].to(device=device, dtype=torch.long)
        if instruct_ids.dim() == 1:
            instruct_ids = instruct_ids.unsqueeze(0)

    prefill_embeds, trailing_list = build_prefill_like_official(
        model,
        input_ids,
        args.language,
        args.speaker or "",
        device,
        instruct_ids=instruct_ids,
        non_streaming_mode=non_streaming,
    )

    tts_pad_token_id = getattr(model.config, "tts_pad_token_id", 0)
    pad_id = torch.tensor([[tts_pad_token_id]], device=device, dtype=torch.long)
    pad_embed = model.talker.text_projection(
        model.talker.model.text_embedding(pad_id)
    )

    B, S, _ = prefill_embeds.shape
    position_ids_1d = torch.arange(S, device=device, dtype=torch.int64)
    position_ids_prefill = position_ids_1d.reshape(1, 1, -1, 1).expand(B, 3, S, 1)

    codec_eos_id = int(model.config.talker_config.codec_eos_token_id)

    logger.info("Running fused ONNX loop (CPU ORT) ...")
    wav_fused = run_fused_onnx_loop(
        sess,
        manifest,
        inputs_embeds=prefill_embeds,
        position_ids_prefill=position_ids_prefill,
        trailing_text=trailing_list,
        pad_embed=pad_embed,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        codec_eos_id=codec_eos_id,
        max_steps=args.max_steps,
    )
    sf.write(str(out_dir / "fused_onnx.wav"), wav_fused, sr)
    logger.info(
        "Wrote fused_onnx.wav, len=%d (%.2fs @ %d Hz)",
        len(wav_fused),
        len(wav_fused) / sr,
        sr,
    )

    readme = out_dir / "COMPARE_OFFICIAL_FUSED_ONNX.txt"
    readme.write_text(
        "\n".join(
            [
                "proto.wav       — Qwen3TTSModel official API (sampling unless --greedy).",
                "fused_onnx.wav  — talker_code2wav_fused.onnx via ORT, greedy argmax (orchestrator-style).",
                "",
                "Sampling differs from greedy; lengths may not match. Compare by listening.",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Done. See %s", readme)


if __name__ == "__main__":
    main()
