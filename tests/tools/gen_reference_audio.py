#!/usr/bin/env python3
"""Generate reference audio with official Qwen3-TTS API for quality comparison.

Produces proto_*.wav using model.generate() (the known-good path).
Compare these with engine output in workspace/audio_samples/engine/.

Usage (conda activate qwen3-tts):
  python tests/tools/gen_reference_audio.py
"""

import time

import numpy as np
import soundfile as sf
import torch

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

bootstrap_project_imports("third_party_qwen")

TEXTS = {
    "test1": "你好，这是单路测试。",
    "test2": "你好，这是流式文本输入测试。",
    "test3": "你好，今天天气真好。",
    "test4": "欢迎来到人工智能语音合成的世界。",
}

MODEL_PATH = REPO_ROOT / "workspace" / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"


def main():
    out_dir = REPO_ROOT / "workspace" / "audio_samples" / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {MODEL_PATH} ...")
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    wrapper = Qwen3TTSModel.from_pretrained(
        str(MODEL_PATH), device_map="cuda:0", dtype=torch.bfloat16,
    )
    print("Model loaded.")

    for name, text in TEXTS.items():
        print(f"\nGenerating [{name}]: {text}")
        t0 = time.perf_counter()
        with torch.no_grad():
            wavs, sr = wrapper.generate_custom_voice(
                text=text,
                speaker="vivian",
                language="auto",
                do_sample=True,
                max_new_tokens=500,
            )
        elapsed = time.perf_counter() - t0
        wav = wavs[0]
        duration = len(wav) / sr
        out_path = out_dir / f"proto_{name}.wav"
        sf.write(str(out_path), wav, sr)
        print(f"  -> {out_path.name}: {len(wav)} samples, {duration:.2f}s, took {elapsed:.1f}s")

    print(f"\nAll reference audio saved to: {out_dir}")
    print("Compare with engine output in: workspace/audio_samples/engine/")

    del wrapper
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
