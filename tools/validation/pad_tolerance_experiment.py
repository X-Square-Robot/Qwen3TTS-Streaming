"""
Pad Tolerance Experiment — A/B audio quality comparison.

Insert 0 / 1 / 2 / 3 / 5 tts_pad tokens at the midpoint of the text token
sequence and compare audio output from the original PyTorch model.

The experiment targets the **trailing_text_hidden** path: during streaming
decode, the Talker Backbone adds one trailing text embedding per step.
Inserting tts_pad tokens at the midpoint of the text simulates what happens
when a batch-padded Triton deployment feeds extra pad tokens to the model.

Usage:
    conda activate qwen3-tts
    python tools/validation/pad_tolerance_experiment.py [--model-variant custom] [--seed 42]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Ensure the validation tools directory is on sys.path for _bootstrap
_validation_dir = str(Path(__file__).resolve().parent)
if _validation_dir not in sys.path:
    sys.path.insert(0, _validation_dir)

from _bootstrap import bootstrap_tool_imports

PROJ_ROOT = bootstrap_tool_imports()
from common import bootstrap_project_imports

bootstrap_project_imports("third_party_qwen")

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
from transformers import AutoConfig, AutoModel


PAD_COUNTS = [0, 1, 2, 3, 5]

TEST_CASES = [
    {
        "text": "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
        "language": "Chinese",
        "speaker": "Vivian",
        "label": "zh_emotion",
    },
    {
        "text": "She said she would be here by noon, but nobody showed up.",
        "language": "English",
        "speaker": "Ryan",
        "label": "en_statement",
    },
]


def load_model(model_path: str, device: str = "cuda:0"):
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    return model


def inject_pad_tokens(input_ids: torch.Tensor, pad_token_id: int, n_pad: int) -> torch.Tensor:
    """Insert *n_pad* copies of pad_token_id at the midpoint of the text
    portion of input_ids.

    input_ids layout (CustomVoice, streaming=False):
        [0:3]    — role tokens  (<|im_start|> assistant \\n)
        [3:-5]   — text content
        [-5:]    — tail tokens  (<|im_end|> \\n <|im_start|> assistant \\n)

    We inject pads at the midpoint of the [3:-5] text span so both
    prefill (first text token) and trailing_text_hidden are affected.
    """
    if n_pad == 0:
        return input_ids

    ids = input_ids.squeeze(0)
    role = ids[:3]
    text = ids[3:-5]
    tail = ids[-5:]

    mid = len(text) // 2
    pad_tensor = torch.full(
        (n_pad,), pad_token_id, dtype=ids.dtype, device=ids.device
    )
    new_text = torch.cat([text[:mid], pad_tensor, text[mid:]])
    return torch.cat([role, new_text, tail]).unsqueeze(0)


def run_experiment(model, test_case: dict, pad_count: int, output_dir: Path,
                   seed: int, run_idx: int):
    label = test_case["label"]
    text = test_case["text"]
    language = test_case["language"]
    speaker = test_case["speaker"]

    formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    processor_input = model.processor(text=formatted_text, return_tensors="pt", padding=True)
    input_ids_orig = processor_input["input_ids"].to(model.device)
    if input_ids_orig.dim() == 1:
        input_ids_orig = input_ids_orig.unsqueeze(0)

    pad_token_id = model.model.config.tts_pad_token_id

    input_ids_modified = inject_pad_tokens(input_ids_orig, pad_token_id, pad_count)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    talker_codes_list, _ = model.model.generate(
        input_ids=[input_ids_modified],
        languages=[language],
        speakers=[speaker],
        non_streaming_mode=False,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    wavs, sr = model.model.speech_tokenizer.decode(
        [{"audio_codes": c} for c in talker_codes_list]
    )

    fname = f"{label}_pad{pad_count}_run{run_idx}.wav"
    out_path = output_dir / fname
    sf.write(str(out_path), wavs[0], sr)

    n_samples = len(wavs[0])
    duration = n_samples / sr

    return {
        "label": label,
        "pad_count": pad_count,
        "run_idx": run_idx,
        "file": fname,
        "duration_s": round(duration, 3),
        "gen_time_s": round(elapsed, 3),
        "n_codec_tokens": talker_codes_list[0].shape[0],
        "input_ids_len": input_ids_modified.shape[1],
        "orig_ids_len": input_ids_orig.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Pad tolerance A/B experiment")
    parser.add_argument(
        "--model-variant", default="custom",
        choices=["custom", "base"],
        help="Model variant: custom (CustomVoice) or base (Base)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=2,
                        help="Repeat each (text, pad_count) combination N times")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    models_dir = PROJ_ROOT / "workspace" / "models"
    if args.model_variant == "custom":
        model_path = str(models_dir / "Qwen3-TTS-12Hz-1.7B-CustomVoice")
    else:
        model_path = str(models_dir / "Qwen3-TTS-12Hz-1.7B-Base")

    output_dir = PROJ_ROOT / "workspace" / "pad_tolerance_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path} ...")
    model = load_model(model_path, device=args.device)
    print("Model loaded.\n")

    all_results = []
    total = len(TEST_CASES) * len(PAD_COUNTS) * args.runs
    done = 0

    for tc in TEST_CASES:
        for pad_count in PAD_COUNTS:
            for run_idx in range(args.runs):
                done += 1
                tag = f"[{done}/{total}] {tc['label']} pad={pad_count} run={run_idx}"
                print(f"{tag} ...", end=" ", flush=True)

                result = run_experiment(
                    model, tc, pad_count, output_dir,
                    seed=args.seed + run_idx,
                    run_idx=run_idx,
                )
                all_results.append(result)

                print(
                    f"OK  {result['duration_s']:.1f}s audio, "
                    f"{result['gen_time_s']:.1f}s gen, "
                    f"{result['n_codec_tokens']} tokens, "
                    f"ids {result['orig_ids_len']}→{result['input_ids_len']}"
                )

    report_path = output_dir / "results.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Label':<16} {'Pad':>4} {'Run':>4} {'Duration':>10} {'GenTime':>10} {'Tokens':>8} {'IDs':>12}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['label']:<16} {r['pad_count']:>4} {r['run_idx']:>4} "
            f"{r['duration_s']:>9.1f}s {r['gen_time_s']:>9.1f}s "
            f"{r['n_codec_tokens']:>8} "
            f"{r['orig_ids_len']:>4}→{r['input_ids_len']:>4}"
        )

    print(f"\nAudio files saved to: {output_dir}")
    print(f"Results JSON: {report_path}")
    print(f"\nA/B Comparison:")
    print("  Listen to *_pad0_* (baseline) vs *_pad1/2/3/5_* for each label.")
    print("  Same seed per run ensures differences come only from pad injection.")


if __name__ == "__main__":
    main()
