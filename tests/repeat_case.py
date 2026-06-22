"""Repeatedly synthesize a single case N times to check hallucination rate.

Reads the text from data.log, sends it to the local engine WebSocket service
N times via token-level streaming (matching production config), and saves
each WAV file for manual inspection.

Usage:
    python tests/repeat_case.py                          # default: 1 time
    python tests/repeat_case.py -n 20                    # 20 times
    python tests/repeat_case.py -n 20 --speaker Serena   # custom speaker
    python tests/repeat_case.py -n 20 --line 2           # only line 2 from data.log
    python tests/repeat_case.py --timeout 300            # 5 min per request

Requires:
    - Engine server running
    - websockets: pip install websockets
"""

import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_LOG = REPO_ROOT / "data.log"
OUTPUT_DIR = REPO_ROOT / "workspace" / "repeat_case"
SAMPLE_RATE = 24000

WS_HOST = "8.160.176.148"
# WS_HOST = "localhost"
WS_PORT = 1182
# WS_PORT = 50052
WS_PATH = "/v1/ws"


def parse_data_log(path: Path) -> list[dict]:
    """Parse data.log into list of {line_no, text}."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("\t", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            entries.append({"line_no": int(parts[0].strip()), "text": parts[1].strip()})
        else:
            entries.append({"line_no": len(entries) + 1, "text": stripped})
    return entries


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


async def synthesize_streaming(
    ws_uri: str,
    text: str,
    speaker: str,
    timeout: float = 600.0,
):
    """Send a token-level streaming WebSocket request and collect all audio.

    Protocol (matching production & serving_endpoints.py pattern):
      1. send "start" → wait for server "start" event before proceeding
      2. send "text"  → drain any queued audio/events (short window)
      3. send "end"   → read until "done" event

    Returns (chunks, first_chunk_sec, total_sec, error, actual_sample_rate).
    """
    import websockets

    chunks = []
    first_ts = None
    error = None
    actual_sr = SAMPLE_RATE
    encoding = "pcm_f32"
    session_started = False

    async def _recv_frame(ws, recv_deadline: float):
        """Receive one frame with timeout.

        Returns:
          - ("__audio__", raw_bytes) for binary audio frames
          - parsed dict for text (JSON) frames
          - None on timeout (no data available yet)
        Raises ConnectionError on server close.
        """
        remaining = recv_deadline - time.perf_counter()
        if remaining <= 0:
            return None
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.01, remaining))
        except asyncio.TimeoutError:
            return None
        except websockets.ConnectionClosed as cc:
            raise ConnectionError(
                f"WebSocket closed by server "
                f"(code={getattr(cc, 'code', '?')}, "
                f"reason={getattr(cc, 'reason', '')!r})"
            ) from cc
        if isinstance(raw, bytes):
            return ("__audio__", raw)
        if isinstance(raw, str):
            return json.loads(raw)
        return None

    def _process_frame(frame):
        """Process one decoded frame. Returns True if terminal (done/error)."""
        nonlocal first_ts, error, actual_sr, encoding, session_started

        if frame is None:
            return False

        # Binary audio frame
        if isinstance(frame, tuple) and frame[0] == "__audio__":
            now = time.perf_counter()
            if first_ts is None:
                first_ts = now
            raw = frame[1]
            if encoding == "pcm_s16le":
                chunks.append(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0)
            else:
                chunks.append(np.frombuffer(raw, dtype=np.float32))
            return False

        # JSON event frame
        if isinstance(frame, dict):
            msg_type = frame.get("type", "")
            if msg_type == "event":
                event = frame.get("event", {})
                event_type = event.get("type", "")

                if event_type == "start":
                    session_started = True
                    audio_info = event.get("audio", {})
                    if audio_info:
                        enc = audio_info.get("encoding")
                        if enc:
                            encoding = enc
                        sr = audio_info.get("sample_rate")
                        if sr:
                            actual_sr = int(sr)
                    return False

                if event_type == "error":
                    error = event.get("message", "engine event error")
                    return True

                if event_type in {"done", "end"}:
                    return True

            return False

        return False

    t0 = time.perf_counter()
    deadline = t0 + timeout
    connection_closed_normally = False
    try:
        async with websockets.connect(
            ws_uri,
            open_timeout=30,
            ping_interval=30,
            ping_timeout=120,
            close_timeout=10,
        ) as ws:
            # ---- Step 1: send "start", wait for server "start" event ----
            await ws.send(json.dumps({
                "type": "start",
                "session_id": f"repeat_{int(t0*1000)}",
                "config": {
                    "task_type": "custom_voice",
                    "language": "auto",
                    "speaker": speaker,
                    "input_mode": "token",
                    "group_policy": "auto",
                    "audio": {
                        "encoding": "pcm_f32",
                        "sample_rate": SAMPLE_RATE,
                        "channels": 1,
                    },
                },
            }, ensure_ascii=False))

            # Wait for server to confirm session start
            while not session_started and time.perf_counter() < deadline:
                frame = await _recv_frame(ws, min(deadline, time.perf_counter() + 5.0))
                if frame is None:
                    continue
                if _process_frame(frame):
                    break

            if not session_started and error is None:
                error = "server did not send 'start' event"
                total = time.perf_counter() - t0
                first_sec = (first_ts - t0) if first_ts else None
                return chunks, first_sec, total, error, actual_sr

            if error:
                total = time.perf_counter() - t0
                first_sec = (first_ts - t0) if first_ts else None
                return chunks, first_sec, total, error, actual_sr

            # ---- Step 2: send "text" ----
            await ws.send(json.dumps({
                "type": "text",
                "text": text,
            }, ensure_ascii=False))

            # Drain any queued frames briefly (0.1s window, matching serving_endpoints.py)
            drain_deadline = min(deadline, time.perf_counter() + 0.1)
            while time.perf_counter() < drain_deadline:
                frame = await _recv_frame(ws, drain_deadline)
                if frame is None:
                    break
                if _process_frame(frame):
                    total = time.perf_counter() - t0
                    first_sec = (first_ts - t0) if first_ts else None
                    return chunks, first_sec, total, error, actual_sr

            # ---- Step 3: send "end", read until done/error ----
            await ws.send(json.dumps({"type": "end"}))

            while time.perf_counter() < deadline:
                frame = await _recv_frame(ws, deadline)
                if frame is None:
                    error = "WebSocket recv timed out waiting for done"
                    break
                if _process_frame(frame):
                    break

    except ConnectionError as exc:
        # Server sends "done" then immediately closes the WebSocket (return ws
        # in websocket_server.py).  The close frame can arrive before we read
        # the "done" event.  If we already received audio, treat this as a
        # normal completion rather than an error.
        if chunks:
            connection_closed_normally = True
        else:
            error = str(exc)
    except Exception as exc:
        error = f"WebSocket error: {exc}"

    total = time.perf_counter() - t0
    first_sec = (first_ts - t0) if first_ts else None
    # If server closed the connection after sending all audio (done + close
    # frame race), treat it as success.
    if connection_closed_normally:
        error = None
    return chunks, first_sec, total, error, actual_sr


async def async_main():
    parser = argparse.ArgumentParser(description="Repeat TTS synthesis for hallucination check")
    parser.add_argument("-n", "--repeat", type=int, default=1, help="Number of repetitions (default: 1)")
    parser.add_argument("--speaker", type=str, default="zhitian", help="Speaker name (default: zhitian)")
    parser.add_argument("--line", type=int, default=None, help="Only synthesize a specific line from data.log")
    parser.add_argument("--host", type=str, default=WS_HOST, help="Engine WebSocket host")
    parser.add_argument("--port", type=int, default=WS_PORT, help="Engine WebSocket port")
    parser.add_argument("--path", type=str, default=WS_PATH, help="Engine WebSocket path")
    parser.add_argument("--timeout", type=float, default=600, help="Per-request timeout in seconds (default: 600)")
    args = parser.parse_args()

    try:
        import websockets
    except ImportError:
        print("ERROR: websockets not installed.")
        print("  pip install websockets")
        sys.exit(1)

    ws_uri = f"ws://{args.host}:{args.port}{args.path}"

    # Quick connectivity check
    try:
        async with websockets.connect(
            ws_uri, open_timeout=30, ping_interval=30, ping_timeout=120, close_timeout=5,
        ):
            pass
    except Exception as exc:
        print(f"ERROR: Cannot connect to engine WebSocket at {ws_uri}")
        print(f"  {exc}")
        print("  Make sure the engine server is running.")
        sys.exit(1)

    # Parse data.log
    if not DATA_LOG.exists():
        print(f"ERROR: data.log not found at {DATA_LOG}")
        sys.exit(1)

    entries = parse_data_log(DATA_LOG)
    if not entries:
        print("ERROR: data.log is empty or has no parseable entries")
        sys.exit(1)

    # Filter by line if specified
    if args.line is not None:
        entries = [e for e in entries if e["line_no"] == args.line]
        if not entries:
            print(f"ERROR: line {args.line} not found in data.log")
            sys.exit(1)

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the text being synthesized for reference
    ref_path = run_dir / "_reference.txt"
    with open(ref_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(f"[Line {e['line_no']}] {e['text']}\n\n")

    print(f"{'='*70}")
    print(f"Repeat Case - Hallucination Check")
    print(f"{'='*70}")
    print(f"  Text source:  {DATA_LOG}")
    print(f"  Lines:        {', '.join(str(e['line_no']) for e in entries)}")
    print(f"  Speaker:      {args.speaker}")
    print(f"  Input mode:   token (streaming, matching production)")
    print(f"  Repetitions:  {args.repeat}")
    print(f"  Timeout:      {args.timeout}s per request")
    print(f"  Output:       {run_dir}/")
    print(f"  Engine WS:    {ws_uri}")
    print(f"{'='*70}\n")

    results = []
    total_runs = args.repeat * len(entries)
    run_idx = 0

    for entry in entries:
        line_no = entry["line_no"]
        text = entry["text"]
        print(f"\n--- Line {line_no}: \"{text[:60]}{'...' if len(text) > 60 else ''}\" ---")

        for i in range(1, args.repeat + 1):
            run_idx += 1
            fname = f"line{line_no:02d}_run{i:03d}.wav"
            wav_path = run_dir / fname

            print(f"  [{run_idx}/{total_runs}] Run {i:3d}/{args.repeat} ... ", end="", flush=True)
            chunks, first_sec, total_sec, err, actual_sr = await synthesize_streaming(
                ws_uri, text, args.speaker, timeout=args.timeout
            )

            if err:
                print(f"ERROR: {err}")
                results.append({"file": fname, "line": line_no, "run": i, "status": "error", "error": err})
                continue

            if not chunks:
                print(f"WARNING: no audio chunks received")
                results.append({"file": fname, "line": line_no, "run": i, "status": "no_audio"})
                continue

            audio = np.concatenate(chunks)
            duration = audio.size / actual_sr
            wav_path.write_bytes(make_wav(audio, sr=actual_sr))

            ttft_ms = f"{first_sec*1000:.0f}ms" if first_sec else "N/A"
            print(f"OK  dur={duration:.2f}s  ttft={ttft_ms}  total={total_sec*1000:.0f}ms  chunks={len(chunks)}")
            results.append({
                "file": fname,
                "line": line_no,
                "run": i,
                "status": "ok",
                "duration_s": round(duration, 2),
                "ttft_ms": round(first_sec * 1000, 0) if first_sec else None,
                "total_ms": round(total_sec * 1000, 0),
                "chunks": len(chunks),
            })

    # Save summary
    summary_path = run_dir / "_summary.json"
    summary = {
        "timestamp": timestamp,
        "speaker": args.speaker,
        "input_mode": "token",
        "repeat": args.repeat,
        "lines": [e["line_no"] for e in entries],
        "text_preview": {str(e["line_no"]): e["text"][:100] for e in entries},
        "total_runs": total_runs,
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "error": sum(1 for r in results if r["status"] == "error"),
        "no_audio": sum(1 for r in results if r["status"] == "no_audio"),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"Done! {summary['ok']}/{total_runs} successful")
    if summary["error"]:
        print(f"  Errors: {summary['error']}")
    if summary["no_audio"]:
        print(f"  No audio: {summary['no_audio']}")
    print(f"\nWAV files: {run_dir}/")
    print(f"Summary:   {summary_path}")
    print(f"Reference: {ref_path}")
    print(f"{'='*70}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
