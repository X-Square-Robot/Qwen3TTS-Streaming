"""Run the badcase collection through the Qwen3-TTS client SDK and save audio.

This rewrites the old ad-hoc WebSocket script to drive synthesis through the
official Python client (``qwen3_tts_client.TTSClient``), using token-level
streaming that matches the production path.  It is meant for *manual listening
evaluation* of regressions — in particular:

  * ``resources/dataset/badcase/short.txt``     — a short utterance whose model
    occasionally hallucinates (a ~10 s sentence balloons to 30 s+).  Run it many
    times (``--repeat``) to catch the intermittent failure.
  * ``resources/dataset/badcase/long.txt``      — long paragraphs whose audio
    quality (听感) degrades.  One pass per paragraph is enough.
  * ``resources/dataset/badcase/difficult.txt`` — a whitespace-separated list of
    number / date / time items that stress text normalization.  Each item is
    synthesized independently.

Each synthesized clip is written as a WAV next to a ``_reference.txt`` and a
``_summary.json`` so the audio can be reviewed by ear afterwards.

Examples
--------
    # Everything with per-dataset defaults (short x20, long x1, difficult x1):
    python tests/repeat_case.py

    # Only the short hallucination probe, 30 repeats:
    python tests/repeat_case.py --datasets short --repeat 30

    # Long-form listening only, against a custom endpoint:
    python tests/repeat_case.py --datasets long --host 127.0.0.1 --port 50071

Requires the engine to be running and reachable (default: engine gRPC on
``localhost:50071``).  The client SDK is imported from ``client/src`` directly,
so no ``pip install`` of the client package is needed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import the client SDK straight from the in-repo source tree so this script
# works without installing the qwen3-tts-client package.
_CLIENT_SRC = REPO_ROOT / "client" / "src"
if _CLIENT_SRC.is_dir() and str(_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_CLIENT_SRC))

from qwen3_tts_client import (  # noqa: E402  (after sys.path tweak)
    AudioChunk,
    AudioFormat,
    SessionStartRequest,
    StreamEvent,
    SynthesisConfig,
    TTSClient,
)

BADCASE_DIR = REPO_ROOT / "resources" / "dataset" / "badcase"
OUTPUT_DIR = REPO_ROOT / "workspace" / "repeat_case"
SAMPLE_RATE = 24000

# Per-dataset behavior.  ``split`` selects how a file is turned into cases:
#   "line"       — one case per non-empty line
#   "whitespace" — one case per whitespace-separated token (whole file)
# ``repeat`` is the default number of repetitions when --repeat is not given.
DATASET_CONFIG: dict[str, dict] = {
    "short":     {"split": "line",       "repeat": 20, "note": "intermittent hallucination probe"},
    "long":      {"split": "line",       "repeat": 1,  "note": "long-form audio quality"},
    "difficult": {"split": "whitespace", "repeat": 1,  "note": "number/date normalization"},
}


def parse_cases(path: Path, split: str) -> list[dict]:
    """Parse a badcase file into a list of {case_no, text} entries."""
    raw = path.read_text(encoding="utf-8")
    if split == "whitespace":
        tokens = [t for t in raw.split() if t.strip()]
        return [{"case_no": i + 1, "text": t} for i, t in enumerate(tokens)]
    # default: line-based
    cases = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            cases.append({"case_no": len(cases) + 1, "text": stripped})
    return cases


def make_wav(samples_f32: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Convert float32 samples to 16-bit PCM WAV bytes."""
    pcm16 = np.clip(samples_f32 * 32767, -32768, 32767).astype(np.int16)
    n = pcm16.size
    buf = bytearray()
    buf += b"RIFF"
    buf += struct.pack("<I", 36 + n * 2)
    buf += b"WAVEfmt "
    buf += struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    buf += b"data"
    buf += struct.pack("<I", n * 2)
    buf += pcm16.tobytes()
    return bytes(buf)


def _chunk_to_array(chunk: AudioChunk) -> np.ndarray:
    """Decode a streamed PCM chunk into float32 samples."""
    encoding = (chunk.audio.encoding or "pcm_f32").lower()
    if encoding == "pcm_s16le":
        return np.frombuffer(chunk.pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    return np.frombuffer(chunk.pcm_bytes, dtype=np.float32)


def synthesize_once(
    client: TTSClient,
    text: str,
    *,
    speaker: str,
    input_mode: str,
    group_policy: str,
    session_id: str,
) -> dict:
    """Synthesize one utterance via token-streaming and collect the audio.

    Returns a dict with keys: status, samples (np.ndarray|None), sample_rate,
    duration_s, ttft_ms, total_ms, chunks, error.
    """
    cfg = SynthesisConfig(
        task_type="custom_voice",
        language="auto",
        speaker=speaker,
        input_mode=input_mode,
        group_policy=group_policy,
        audio=AudioFormat(encoding="pcm_f32", sample_rate=SAMPLE_RATE, channels=1),
    )
    start = SessionStartRequest(
        session_id=session_id,
        config=cfg,
        output_policy=cfg.output_policy,
        timing=cfg.timing_context,
    )

    parts: list[np.ndarray] = []
    sample_rate = SAMPLE_RATE
    first_ts: float | None = None
    error: str | None = None

    t0 = time.perf_counter()
    try:
        session = client.open_stream(start)
        session.send_text(text)
        session.end()
        for msg in session.iter_messages():
            if isinstance(msg, AudioChunk):
                if first_ts is None:
                    first_ts = time.perf_counter()
                if msg.audio and msg.audio.sample_rate:
                    sample_rate = int(msg.audio.sample_rate)
                arr = _chunk_to_array(msg)
                if arr.size:
                    parts.append(arr)
            elif isinstance(msg, StreamEvent):
                if msg.type == "error":
                    error = msg.message or "engine error event"
                    break
    except Exception as exc:  # noqa: BLE001 — surface any transport failure
        error = f"{type(exc).__name__}: {exc}"

    total_ms = (time.perf_counter() - t0) * 1000.0
    if not parts:
        return {
            "status": "error" if error else "no_audio", "samples": None,
            "sample_rate": sample_rate, "duration_s": 0.0, "ttft_ms": None,
            "total_ms": round(total_ms), "chunks": 0, "error": error,
        }

    samples = np.concatenate(parts)
    ttft_ms = round((first_ts - t0) * 1000.0) if first_ts else None
    return {
        "status": "ok",
        "samples": samples,
        "sample_rate": sample_rate,
        "duration_s": round(samples.size / sample_rate, 3),
        "ttft_ms": ttft_ms,
        "total_ms": round(total_ms),
        "chunks": len(parts),
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the badcase collection through the client SDK and save audio.",
    )
    parser.add_argument("--datasets", default="short,long,difficult",
                        help="Comma list of datasets under resources/dataset/badcase (default: all)")
    parser.add_argument("-n", "--repeat", type=int, default=0,
                        help="Repetitions per case (0 = per-dataset default)")
    parser.add_argument("--speaker", default="serena", help="Speaker name (default: serena)")
    parser.add_argument("--host", default="localhost", help="Engine host (default: localhost)")
    parser.add_argument("--port", type=int, default=50071, help="Engine port (default: 50071)")
    parser.add_argument("--transport", default="engine-grpc",
                        help="Client transport (engine-grpc | engine-websocket | auto)")
    parser.add_argument("--input-mode", default="token", help="Input mode (default: token)")
    parser.add_argument("--group-policy", default="auto", help="Group policy (default: auto)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-request timeout in seconds (default: 300)")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="Limit number of cases per dataset (0 = no limit)")
    parser.add_argument("--outdir", default="", help="Override output run directory")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASET_CONFIG:
            print(f"ERROR: unknown dataset {d!r}; known: {', '.join(DATASET_CONFIG)}", file=sys.stderr)
            return 2
        if not (BADCASE_DIR / f"{d}.txt").is_file():
            print(f"ERROR: missing {BADCASE_DIR / (d + '.txt')}", file=sys.stderr)
            return 2

    if args.transport == "engine-websocket":
        endpoint = f"ws://{args.host}:{args.port}/v1/ws"
    else:
        endpoint = f"{args.host}:{args.port}"

    print("=" * 72)
    print("Badcase runner (client SDK)")
    print("=" * 72)
    print(f"  endpoint:    {endpoint}  (transport={args.transport})")
    print(f"  speaker:     {args.speaker}")
    print(f"  input_mode:  {args.input_mode}   group_policy: {args.group_policy}")
    print(f"  datasets:    {', '.join(datasets)}")
    print("=" * 72)

    try:
        client = TTSClient.connect(endpoint, transport=args.transport, timeout=args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot connect to engine at {endpoint}: {exc}", file=sys.stderr)
        print("  Make sure the engine is running and the port is correct.", file=sys.stderr)
        return 1
    print(f"  connected:   transport={client.resolved_transport}\n")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.outdir) if args.outdir else OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "timestamp": timestamp,
        "endpoint": endpoint,
        "transport": client.resolved_transport,
        "speaker": args.speaker,
        "input_mode": args.input_mode,
        "group_policy": args.group_policy,
        "datasets": {},
    }
    grand_ok = grand_total = 0

    for dataset in datasets:
        conf = DATASET_CONFIG[dataset]
        repeat = args.repeat if args.repeat > 0 else conf["repeat"]
        cases = parse_cases(BADCASE_DIR / f"{dataset}.txt", conf["split"])
        if args.max_cases > 0:
            cases = cases[: args.max_cases]

        ds_dir = run_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        with open(ds_dir / "_reference.txt", "w", encoding="utf-8") as f:
            for c in cases:
                f.write(f"[case {c['case_no']:03d}] {c['text']}\n")

        print(f"\n### dataset={dataset}  ({conf['note']})  cases={len(cases)}  repeat={repeat}")
        results: list[dict] = []
        durations_by_case: dict[int, list[float]] = {}

        for c in cases:
            case_no, text = c["case_no"], c["text"]
            preview = text[:48] + ("…" if len(text) > 48 else "")
            for r in range(1, repeat + 1):
                fname = f"case{case_no:03d}_run{r:03d}.wav"
                sid = f"badcase-{dataset}-{case_no:03d}-{r:03d}-{int(time.time()*1000)}"
                res = synthesize_once(
                    client, text,
                    speaker=args.speaker, input_mode=args.input_mode,
                    group_policy=args.group_policy, session_id=sid,
                )
                rec = {
                    "file": f"{dataset}/{fname}", "case": case_no, "run": r,
                    "status": res["status"], "duration_s": res["duration_s"],
                    "ttft_ms": res["ttft_ms"], "total_ms": res["total_ms"],
                    "chunks": res["chunks"], "text_len": len(text),
                }
                if res["status"] == "ok":
                    (ds_dir / fname).write_bytes(make_wav(res["samples"], sr=res["sample_rate"]))
                    durations_by_case.setdefault(case_no, []).append(res["duration_s"])
                    print(f"  [{dataset} c{case_no:03d} r{r:03d}] OK  dur={res['duration_s']:6.2f}s "
                          f"ttft={res['ttft_ms']}ms chunks={res['chunks']}  \"{preview}\"")
                else:
                    rec["error"] = res["error"]
                    print(f"  [{dataset} c{case_no:03d} r{r:03d}] {res['status'].upper()}: {res['error']}")
                results.append(rec)
                grand_total += 1
                grand_ok += 1 if res["status"] == "ok" else 0

        # Per-case duration stats + simple hallucination flag (a run far longer
        # than the case's own median — the 10s→30s blow-up signature).
        case_stats = {}
        for case_no, durs in durations_by_case.items():
            med = statistics.median(durs)
            outliers = [round(d, 2) for d in durs if med > 0 and d > med * 1.5]
            case_stats[str(case_no)] = {
                "runs": len(durs),
                "min_s": round(min(durs), 2),
                "median_s": round(med, 2),
                "max_s": round(max(durs), 2),
                "suspected_hallucination_runs": outliers,
            }
            if outliers:
                print(f"  ⚠ case {case_no:03d}: median={med:.2f}s but runs {outliers} "
                      f"(>1.5x median) — possible hallucination")

        summary["datasets"][dataset] = {
            "note": conf["note"], "split": conf["split"], "repeat": repeat,
            "cases": len(cases), "case_stats": case_stats, "results": results,
        }

    summary["grand_total"] = grand_total
    summary["grand_ok"] = grand_ok
    (run_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Done: {grand_ok}/{grand_total} OK")
    print(f"Audio + summary: {run_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
