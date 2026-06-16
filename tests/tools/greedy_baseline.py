#!/usr/bin/env python3
"""
Greedy baseline comparison: official Qwen3-TTS API vs our Triton BLS.

Runs the SAME texts with the SAME speaker using:
  A) Official qwen_tts with do_sample=False (greedy / argmax)
  B) Our Triton orchestrator (which also uses argmax)

If both produce similar silence patterns → model-level issue under greedy.
If only Triton produces silence → BLS logic bug.

Usage:
    conda activate qwen3-tts
    python tests/tools/greedy_baseline.py \
        --model-path workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        [--triton localhost:8001] \
        [--output-dir workspace/test_greedy_baseline]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from tests.support.triton_streaming import build_request_payload, infer_stream, save_wav

SAMPLE_RATE = 24000

SPEAKER = "Serena"
LANGUAGE = "Chinese"

TEST_CASES = {
    "short": "你好，今天天气真好。",
    "medium": (
        "人工智能正在深刻改变我们的世界。"
        "从语音识别到自然语言处理，从计算机视觉到机器人技术，"
        "AI的应用已经渗透到生活的方方面面。"
        "在医疗领域，AI可以辅助诊断疾病、发现新药物。"
        "在教育领域，AI可以提供个性化的学习方案。"
        "在交通领域，自动驾驶技术正在逐步成熟。"
        "未来，人工智能将继续推动社会进步，"
        "为人类创造更多的可能性。"
    ),
    "long": (
        "在遥远的古代，人类就开始仰望星空，思考宇宙的奥秘。"
        "从古希腊的哲学家到中国的天文学家，人们不断探索着这个世界的本质。"
        "随着科学技术的发展，我们对宇宙的认知越来越深入。"
        "从伽利略的望远镜到哈勃太空望远镜，每一次技术的突破都让我们看到了更广阔的世界。"
        "如今，人工智能技术的发展为科学研究带来了前所未有的机遇。"
        "机器学习算法可以处理海量的天文数据，发现人眼无法察觉的规律。"
        "深度学习网络能够分析复杂的光谱数据，帮助我们理解恒星的演化过程。"
        "在医学领域，AI辅助诊断系统已经在某些疾病的识别上达到了专家水平。"
        "在交通出行方面，自动驾驶技术正在逐步走向成熟，有望彻底改变我们的出行方式。"
        "在教育领域，个性化学习系统能够根据每个学生的特点量身定制学习方案。"
        "在金融领域，智能风控系统可以实时监测交易异常，保护用户的资金安全。"
        "然而，技术的发展也带来了新的挑战和思考。"
        "如何确保人工智能的安全性和可控性？"
        "如何在自动化浪潮中保障就业和社会公平？"
        "这些问题需要我们全社会共同面对和解决。"
        "只有在技术进步与人文关怀之间找到平衡，"
        "才能让人工智能真正造福全人类，"
        "创造一个更加美好的未来。"
    ),
}
def analyze_silence(audio: np.ndarray, sr: int = SAMPLE_RATE,
                    threshold: float = 0.005, min_dur_ms: float = 200) -> list:
    """Detect silence segments longer than min_dur_ms."""
    abs_audio = np.abs(audio)
    is_silent = abs_audio < threshold
    min_samples = int(sr * min_dur_ms / 1000)

    segments = []
    start = None
    for i, s in enumerate(is_silent):
        if s and start is None:
            start = i
        elif not s and start is not None:
            length = i - start
            if length >= min_samples:
                segments.append((start / sr, i / sr, length / sr))
            start = None
    if start is not None:
        length = len(is_silent) - start
        if length >= min_samples:
            segments.append((start / sr, len(is_silent) / sr, length / sr))
    return segments


def run_official_greedy(model_path: str, output_dir: Path):
    """Run official Qwen3TTSModel with do_sample=False (greedy)."""
    print("\n" + "=" * 70)
    print("  Part A: Official Qwen3-TTS API — Greedy (do_sample=False)")
    print("=" * 70)

    from qwen_tts import Qwen3TTSModel

    print(f"  Loading model from {model_path} ...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    results = {}
    for name, text in TEST_CASES.items():
        print(f"\n  --- {name} ({len(text)} chars) ---")
        t0 = time.time()
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=SPEAKER,
            language=LANGUAGE,
            do_sample=False,
        )
        elapsed = time.time() - t0
        audio = wavs[0]
        duration = len(audio) / sr

        out_path = str(output_dir / f"official_greedy_{name}.wav")
        save_wav(audio, out_path, sample_rate=sr)

        silences = analyze_silence(audio, sr)
        total_silence = sum(s[2] for s in silences)

        print(f"  Duration: {duration:.2f}s  Time: {elapsed:.1f}s  "
              f"Silence segments: {len(silences)}  Total silence: {total_silence:.2f}s")
        for i, (start, end, dur) in enumerate(silences):
            print(f"    silence[{i}]: {start:.2f}s ~ {end:.2f}s ({dur:.2f}s)")
        print(f"  Saved: {out_path}")

        results[name] = {
            "duration": duration, "elapsed": elapsed,
            "silences": silences, "total_silence": total_silence,
        }

    # Also run with default sampling for reference
    print("\n" + "-" * 70)
    print("  Part A2: Official — Default sampling (do_sample=True, temp=0.9)")
    print("-" * 70)

    for name, text in TEST_CASES.items():
        print(f"\n  --- {name} ({len(text)} chars) ---")
        t0 = time.time()
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=SPEAKER,
            language=LANGUAGE,
        )
        elapsed = time.time() - t0
        audio = wavs[0]
        duration = len(audio) / sr

        out_path = str(output_dir / f"official_sampling_{name}.wav")
        save_wav(audio, out_path, sample_rate=sr)

        silences = analyze_silence(audio, sr)
        total_silence = sum(s[2] for s in silences)

        print(f"  Duration: {duration:.2f}s  Time: {elapsed:.1f}s  "
              f"Silence segments: {len(silences)}  Total silence: {total_silence:.2f}s")
        for i, (start, end, dur) in enumerate(silences):
            print(f"    silence[{i}]: {start:.2f}s ~ {end:.2f}s ({dur:.2f}s)")
        print(f"  Saved: {out_path}")

        key = f"{name}_sampling"
        results[key] = {
            "duration": duration, "elapsed": elapsed,
            "silences": silences, "total_silence": total_silence,
        }

    del model
    torch.cuda.empty_cache()
    return results


def run_triton_greedy(triton_url: str, output_dir: Path):
    """Run our Triton orchestrator with the same texts."""
    print("\n" + "=" * 70)
    print("  Part B: Triton BLS (sampling, temp=0.9)")
    print("=" * 70)

    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print("  ERROR: tritonclient[grpc] not installed, skipping Triton test")
        return {}

    results = {}
    for name, text in TEST_CASES.items():
        print(f"\n  --- {name} ({len(text)} chars) ---")

        client = grpcclient.InferenceServerClient(url=triton_url)
        req_dict = build_request_payload(
            text=text,
            task_type="custom_voice",
            speaker=SPEAKER,
            language=LANGUAGE,
            action="synthesize",
            session_id=f"greedy-baseline-{name}",
        )
        stream = infer_stream(client, grpcclient, req_dict, timeout=300)
        elapsed = stream.total_ms / 1000.0

        if stream.error:
            print(f"  ERROR: {stream.error[:200]}")
            results[name] = {"error": stream.error}
            continue
        if stream.warnings:
            for w in stream.warnings:
                print(f"  WARNING: {w}")

        audio = stream.audio if stream.audio is not None else np.array([], dtype=np.float32)
        sample_rate = int(stream.metadata.get("audio_format", {}).get("sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        duration = len(audio) / sample_rate

        out_path = str(output_dir / f"triton_argmax_{name}.wav")
        save_wav(audio, out_path, sample_rate=sample_rate)

        silences = analyze_silence(audio, sample_rate)
        total_silence = sum(s[2] for s in silences)

        print(f"  Duration: {duration:.2f}s  Time: {elapsed:.1f}s  "
              f"Silence segments: {len(silences)}  Total silence: {total_silence:.2f}s")
        for i, (start, end, dur) in enumerate(silences):
            print(f"    silence[{i}]: {start:.2f}s ~ {end:.2f}s ({dur:.2f}s)")
        print(f"  Saved: {out_path}")

        results[name] = {
            "duration": duration, "elapsed": elapsed,
            "silences": silences, "total_silence": total_silence,
        }

    return results


def print_comparison(official: dict, triton: dict):
    print("\n" + "=" * 70)
    print("  COMPARISON: Official Sampling vs Triton BLS (sampling)")
    print("=" * 70)
    print(f"  {'Case':<12} {'Official dur':>14} {'Official sil':>14} "
          f"{'Triton dur':>14} {'Triton sil':>14} {'Verdict'}")
    print("  " + "-" * 68)

    for name in TEST_CASES:
        o = official.get(name, {})
        t = triton.get(name, {})

        if "error" in t:
            print(f"  {name:<12} {'':>14} {'':>14} {'ERROR':>14} {'':>14}")
            continue

        o_dur = f"{o.get('duration', 0):.2f}s" if o else "N/A"
        o_sil = f"{o.get('total_silence', 0):.2f}s" if o else "N/A"
        t_dur = f"{t.get('duration', 0):.2f}s" if t else "N/A"
        t_sil = f"{t.get('total_silence', 0):.2f}s" if t else "N/A"

        o_sil_val = o.get("total_silence", 0) if o else 0
        t_sil_val = t.get("total_silence", 0) if t else 0

        if o_sil_val > 0.5 and t_sil_val > 0.5:
            verdict = "BOTH_SILENT (model issue)"
        elif o_sil_val <= 0.5 and t_sil_val > 0.5:
            verdict = "TRITON_ONLY (BLS bug?)"
        elif o_sil_val > 0.5 and t_sil_val <= 0.5:
            verdict = "OFFICIAL_ONLY (unexpected)"
        else:
            verdict = "BOTH_OK"

        print(f"  {name:<12} {o_dur:>14} {o_sil:>14} {t_dur:>14} {t_sil:>14} {verdict}")

    # Also show sampling baseline
    print("\n  Reference: Official with default sampling (do_sample=True)")
    for name in TEST_CASES:
        key = f"{name}_sampling"
        o = official.get(key, {})
        if o:
            print(f"  {name:<12} dur={o.get('duration', 0):.2f}s  "
                  f"silence={o.get('total_silence', 0):.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Greedy baseline comparison")
    parser.add_argument("--model-path",
                        default="workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        help="Path to official CustomVoice model")
    parser.add_argument("--triton", default="localhost:8001",
                        help="Triton gRPC address")
    parser.add_argument("--output-dir", default="workspace/test_greedy_baseline",
                        help="Output directory")
    parser.add_argument("--skip-official", action="store_true",
                        help="Skip official model test (only run Triton)")
    parser.add_argument("--skip-triton", action="store_true",
                        help="Skip Triton test (only run official)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    official_results = {}
    triton_results = {}

    if not args.skip_official:
        official_results = run_official_greedy(args.model_path, output_dir)

    if not args.skip_triton:
        triton_results = run_triton_greedy(args.triton, output_dir)

    if official_results and triton_results:
        print_comparison(official_results, triton_results)
    elif official_results:
        print("\n  (Triton skipped — only official results available)")
    elif triton_results:
        print("\n  (Official skipped — only Triton results available)")

    print(f"\n  All outputs saved to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
