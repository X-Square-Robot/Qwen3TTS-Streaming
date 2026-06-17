"""
Quick integration test for the TTS Orchestrator on Triton.

Sends a request and collects streaming audio chunks.
Saves the result as a WAV file for playback verification.

Usage:
    python tools/validation/triton_tts_client.py [--text "..."] [--output output.wav]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the validation tools directory is on sys.path for _bootstrap
_validation_dir = str(Path(__file__).resolve().parent)
if _validation_dir not in sys.path:
    sys.path.insert(0, _validation_dir)

from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from tests.support.triton_streaming import (
    StreamResult,
    build_request_payload,
    infer_stream,
    save_wav,
)

TRITON_MODEL_VERSION = os.environ.get("TRITON_MODEL_VERSION", "1")


def _build_request(text: str, task_type: str, language: str) -> str:
    payload = build_request_payload(
        text=text,
        task_type=task_type,
        language=language,
    )
    return json.dumps(payload)


def test_http_non_streaming(url: str, text: str, output_path: str, task_type: str, language: str):
    """Test via HTTP (non-streaming, will get first response only)."""
    import requests

    req_json = _build_request(text=text, task_type=task_type, language=language)

    payload = {
        "inputs": [
            {
                "name": "request",
                "shape": [1],
                "datatype": "BYTES",
                "data": [req_json],
            }
        ],
        "outputs": [
            {"name": "audio_chunk"},
            {"name": "event_type"},
            {"name": "event_json"},
            {"name": "is_final"},
        ],
    }

    print(f"Sending request to {url} ...")
    print(f"  Text: {text}")
    t0 = time.time()

    resp = requests.post(
        f"{url}/v2/models/tts_orchestrator/versions/{TRITON_MODEL_VERSION}/infer",
        json=payload,
        timeout=120,
    )

    elapsed = time.time() - t0
    print(f"  Response status: {resp.status_code} ({elapsed:.2f}s)")

    if resp.status_code != 200:
        print(f"  Error: {resp.text}")
        return False

    result = resp.json()
    print(f"  Response keys: {list(result.keys())}")

    for out in result.get("outputs", []):
        name = out["name"]
        shape = out.get("shape", [])
        print(f"  Output '{name}': shape={shape}")

    return True


def test_grpc_streaming(
    host: str,
    port: int,
    text: str,
    output_path: str,
    task_type: str,
    language: str,
):
    """Test via gRPC streaming (decoupled model)."""
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print("tritonclient not available, installing ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "tritonclient[grpc]", "-q"])
        import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(url=f"{host}:{port}")

    if not client.is_server_ready():
        print("ERROR: Triton server not ready")
        return False

    print(f"Server ready. Sending TTS request ...")
    print(f"  Text: {text}")
    print(f"  Task type: {task_type or '<auto>'}")

    payload = build_request_payload(text=text, task_type=task_type, language=language)
    stream = infer_stream(client, grpcclient, payload, timeout=120)

    if stream.error:
        print(f"\n  Errors: {[stream.error]}")
        return False

    if stream.audio is None or stream.audio.size == 0:
        print(f"\n  No audio chunks received ({stream.total_ms/1000.0:.2f}s)")
        return False

    all_audio = stream.audio
    sample_rate = int(stream.metadata.get("audio_format", {}).get("sample_rate", 24000) or 24000)
    print(f"\n  Total audio: {len(all_audio)} samples ({len(all_audio)/sample_rate:.2f}s at {sample_rate}Hz)")
    print(f"  Latency: {stream.total_ms/1000.0:.2f}s")
    print(f"  Audio range: [{all_audio.min():.4f}, {all_audio.max():.4f}]")

    if np.all(all_audio == 0):
        print("  WARNING: Audio is all zeros!")

    save_wav(all_audio, output_path, sample_rate=sample_rate)
    print(f"  Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test TTS Orchestrator on Triton")
    parser.add_argument("--text", default="今天天气真好，我们一起出去玩吧。",
                        help="Text to synthesize")
    parser.add_argument("--output", default="workspace/test_tts_output.wav",
                        help="Output WAV file path")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--grpc-port", type=int, default=8001)
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--task-type", default="",
                        help="Optional task type. Leave empty to let the server bind to the loaded model type.")
    parser.add_argument("--language", default="auto",
                        help="Language field sent in the request payload")
    parser.add_argument("--mode", choices=["grpc", "http"], default="grpc",
                        help="Client mode")
    args = parser.parse_args()

    print("=" * 60)
    print("  TTS Orchestrator Integration Test")
    print("=" * 60)

    if args.mode == "grpc":
        ok = test_grpc_streaming(
            args.host,
            args.grpc_port,
            args.text,
            args.output,
            args.task_type,
            args.language,
        )
    else:
        ok = test_http_non_streaming(
            f"http://{args.host}:{args.http_port}",
            args.text,
            args.output,
            args.task_type,
            args.language,
        )

    print()
    if ok:
        print("  RESULT: PASS")
    else:
        print("  RESULT: FAIL")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
