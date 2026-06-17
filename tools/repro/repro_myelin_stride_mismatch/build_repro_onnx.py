#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import onnx
from onnx import utils


def run(cmd: list[str], env: dict[str, str], cwd: Path) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _clean_export_env(static_state_batch: int | None) -> dict[str, str]:
    env = os.environ.copy()

    # Remove known debug/experimental toggles that make reproduction non-deterministic.
    remove_exact = {
        "C2W_STATIC_STATE_BATCH",
        "C2W_DEBUG_BYPASS_QUANTIZER",
        "C2W_DEBUG_BYPASS_TRANSFORMER",
        "TRT_MYELIN_DISABLE",
        "ENGINE_DTYPE",
        "MAX_BATCH_SIZE",
        "BUILD_VERIFICATION_ENGINES",
    }
    for key in list(env.keys()):
        if key in remove_exact or key.startswith("C2W_DEBUG_"):
            env.pop(key, None)

    if static_state_batch is not None:
        env["C2W_STATIC_STATE_BATCH"] = str(static_state_batch)
    return env


def export_code2wav(repo_root: Path, out_dir: Path, static_state_batch: int | None) -> Path:
    env = _clean_export_env(static_state_batch)

    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "qwen3-tts",
        "python",
        "scripts/export/export_06_code2wav_decoder.py",
        "--device",
        "cpu",
        "--output-dir",
        str(out_dir),
    ]
    run(cmd, env=env, cwd=repo_root)
    onnx_path = out_dir / "tokenizer" / "code2wav_decoder.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX: {onnx_path}")
    return onnx_path


def extract_wav_only(src: Path, dst: Path) -> None:
    model = onnx.load(str(src))
    input_names = [i.name for i in model.graph.input]
    utils.extract_model(str(src), str(dst), input_names, ["wav"])
    m2 = onnx.load(str(dst))
    print(
        f"[extract] {dst}  nodes={len(m2.graph.node)} inputs={len(m2.graph.input)} outputs={len(m2.graph.output)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build minimal repro ONNX pair for TensorRT Myelin dim_count/stride_order mismatch."
    )
    parser.add_argument(
        "--out-root",
        default="/tmp/myelin_repro_case",
        help="Output root directory (default: /tmp/myelin_repro_case)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export_06 and only run wav-only extraction from existing ONNX files.",
    )
    parser.add_argument(
        "--static-state-batch",
        type=int,
        default=8,
        help="Static state batch for pass case (default: 8)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    out_root = Path(args.out_root).resolve()
    dyn_dir = out_root / "dynamic_state"
    sta_dir = out_root / f"static_state_{args.static_state_batch}"
    dyn_dir.mkdir(parents=True, exist_ok=True)
    sta_dir.mkdir(parents=True, exist_ok=True)

    dyn_onnx = dyn_dir / "tokenizer" / "code2wav_decoder.onnx"
    sta_onnx = sta_dir / "tokenizer" / "code2wav_decoder.onnx"

    if not args.skip_export:
        dyn_onnx = export_code2wav(repo_root, dyn_dir, static_state_batch=None)
        sta_onnx = export_code2wav(
            repo_root, sta_dir, static_state_batch=args.static_state_batch
        )
    else:
        if not dyn_onnx.exists() or not sta_onnx.exists():
            raise FileNotFoundError(
                f"--skip-export was set but missing ONNX: {dyn_onnx} or {sta_onnx}"
            )

    dyn_wav = dyn_dir / "tokenizer" / "code2wav_decoder_wav_only.onnx"
    sta_wav = sta_dir / "tokenizer" / "code2wav_decoder_wav_only.onnx"
    extract_wav_only(dyn_onnx, dyn_wav)
    extract_wav_only(sta_onnx, sta_wav)

    print("\n[done] Repro ONNX generated:")
    print(f"  fail  (dynamic state): {dyn_wav}")
    print(f"  pass  (static state ): {sta_wav}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
