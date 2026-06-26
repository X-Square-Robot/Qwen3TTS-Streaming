#!/usr/bin/env python3
"""Unified TTS benchmark tool.

Merges engine-standalone-benchmark and triton-concurrent-tts into one
entry-point selected via --target.

Targets:
  engine-standalone  E2E benchmark against the standalone gRPC engine.
  triton-concurrent  Multi-session concurrent test via Triton gRPC.

Deprecated: engine_standalone_benchmark.py, triton_concurrent_tts.py.
Use ``python -m tests.tools.benchmark --target <target>`` instead.
"""

from __future__ import annotations

import argparse, importlib, statistics, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import REPO_ROOT, bootstrap_project_imports
bootstrap_project_imports("repo", "scripts_python", "third_party_qwen")

# -- shared -----------------------------------------------------------------

def _print_deprecation():
    print("NOTE: engine_standalone_benchmark.py and triton_concurrent_tts.py are "
          "deprecated. Use  python -m tests.tools.benchmark --target <target>  instead.",
          file=sys.stderr)

def _summary(all_results: dict) -> tuple[int, int]:
    total_ok = total_fail = 0
    for name, results in all_results.items():
        ok = sum(1 for r in results if r.error is None)
        fail = sum(1 for r in results if r.error is not None)
        fc = [r.first_chunk_ms for r in results if r.first_chunk_ms is not None]
        avg = statistics.mean(fc) if fc else 0
        print(f"  {name:20s}  OK={ok}  FAIL={fail}  avg_first_chunk={avg:.0f}ms")
        total_ok += ok; total_fail += fail
    print(f"\n  TOTAL: {total_ok} OK, {total_fail} FAILED")
    return total_ok, total_fail

# -- engine-standalone ------------------------------------------------------

def _run_engine_standalone(args) -> int:
    from tests.support.engine_standalone import (
        OUTPUT_DIR, TTSResult, _check_server, _get_capabilities,
        run_badcases, run_concurrent, run_custom_voice_instruct,
        run_long_text, run_single_smoke, run_streaming_text, stress_concurrent,
    )
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    host, port = args.host, args.port

    print("=" * 60 + "\n  TTS Engine Standalone E2E Benchmark\n" + "=" * 60)
    print(f"  Engine: {host}:{port}  Concurrency: {levels}  Output: {output_dir}")
    if not _check_server(host, port):
        print(f"\nERROR: Engine not reachable at {host}:{port}"); return 1
    print("  Server: READY")
    cap = _get_capabilities(host, port)
    print(f"  Variant: {cap['variant'] or 'unknown'}  ModelType: {cap['loaded_model_type'] or 'unknown'}")

    R: dict[str, list[TTSResult]] = {}
    if not args.skip_single:       R["single"] = [run_single_smoke(host, port, output_dir)]
    if not args.skip_streaming:    R["streaming"] = [run_streaming_text(host, port, output_dir)]
    if not args.skip_custom_instruct:
        v = run_custom_voice_instruct(host, port, output_dir)
        if v: R["custom_instruct"] = v
    if not args.skip_concurrent:
        for lv in levels: R[f"concurrent_x{lv}"] = run_concurrent(host, port, lv, output_dir)
    if not args.skip_long:         R["long_text"] = run_long_text(host, port, output_dir)
    if not args.skip_badcase:      run_badcases(host, port, output_dir)
    if args.stress_concurrency > 0 and args.stress_rounds > 0:
        rounds = stress_concurrent(host, port, concurrency=args.stress_concurrency,
                                   rounds=args.stress_rounds, warmup_rounds=args.stress_warmup_rounds)
        R[f"stress_x{args.stress_concurrency}"] = [r for batch in rounds[args.stress_warmup_rounds:] for r in batch]

    print("\n" + "=" * 60 + "\n  FINAL SUMMARY\n" + "=" * 60)
    ok, fail = _summary(R)
    print(f"  Output: {output_dir.resolve()}")
    return 0 if fail == 0 else 1

# -- triton-concurrent ------------------------------------------------------

_TEST_TEXTS = [
    "你好，今天天气真好。", "欢迎来到人工智能语音合成的世界。",
    "技术创新推动着社会不断前进。", "我们正在测试多路并发的语音合成能力。",
    "这是第五路测试文本，用来验证系统的处理能力。",
    "深度学习让机器能够理解和生成自然的语音。",
    "云计算和边缘计算相结合，提供更好的用户体验。",
    "每一次迭代都让我们的系统变得更加完善。",
]
_LONG_TEXT = (
    "人工智能正在深刻改变我们的世界。从语音识别到自然语言处理，"
    "从计算机视觉到机器人技术，AI的应用已经渗透到生活的方方面面。"
    "在医疗领域，AI可以辅助诊断疾病、发现新药物。在教育领域，"
    "AI可以提供个性化的学习方案。在交通领域，自动驾驶技术正在逐步成熟。"
    "未来，人工智能将继续推动社会进步，为人类创造更多的可能性。"
)

def _get_triton_client(url):
    try: import tritonclient.grpc as g
    except ImportError: print("ERROR: tritonclient[grpc] not installed"); sys.exit(1)
    return g, g.InferenceServerClient(url=url)

def _run_triton_concurrent(args) -> int:
    from qwen3_tts_protocol import save_wav
    from tests.support.triton_streaming import StreamResult, infer_stream, infer_text_stream

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    levels = [int(x.strip()) for x in args.concurrency.split(",")]

    print("=" * 60 + "\n  TTS Concurrent Test & Benchmark\n" + "=" * 60)
    print(f"  Triton: {args.triton}  Levels: {levels}  Output: {output_dir}")
    grpc_mod, client = _get_triton_client(args.triton)
    if not client.is_server_ready():
        print("ERROR: Triton server not ready"); return 1
    if not client.is_model_ready("tts_orchestrator"):
        print("ERROR: tts_orchestrator not ready"); return 1
    print("  Server: READY")

    R: dict[str, list[StreamResult]] = {}

    # single smoke
    r = infer_stream(client, grpc_mod, {"text": "你好，这是单路测试。", "task_type": "custom_voice", "speaker": "Serena"})
    if r.audio is not None and r.audio.size > 0:
        save_wav(r.audio, str(output_dir / "test1_single_smoke.wav"))
    R["single"] = [r]

    # streaming
    if not args.skip_streaming:
        gm2, _ = _get_triton_client(args.triton)
        r = infer_text_stream(args.triton, gm2,
            {"action": "init", "session_id": uuid.uuid4().hex[:12], "task_type": "custom_voice",
             "speaker": "Serena", "text": "你好，这是流式文本输入测试。"},
            ["我们正在验证", "文本追加功能", "是否工作正常。"], chunk_delay_ms=200)
        if r.audio is not None and r.audio.size > 0:
            save_wav(r.audio, str(output_dir / "test2_streaming_text.wav"))
        R["streaming"] = [r]

    # concurrent per level
    base_mod = importlib.import_module(grpc_mod.__name__.rsplit(".", 1)[0])
    for lv in levels:
        t0 = time.perf_counter()
        results = []
        def _run(idx, _lv=lv):
            c = base_mod.InferenceServerClient(url=args.triton)
            return infer_stream(c, base_mod, {"text": _TEST_TEXTS[idx % len(_TEST_TEXTS)],
                "task_type": "custom_voice", "speaker": "Serena", "session_id": f"c{idx}"})
        with ThreadPoolExecutor(max_workers=lv) as pool:
            futs = {pool.submit(_run, i): i for i in range(lv)}
            for f in as_completed(futs):
                try: results.append(f.result())
                except Exception as e: results.append(StreamResult(session_id=f"c{futs[f]}", text="", error=str(e)))
        print(f"  x{lv} wall={time.perf_counter()-t0:.2f}s")
        for i, r in enumerate(results):
            if r.audio is not None and r.audio.size > 0:
                save_wav(r.audio, str(output_dir / f"test3_c{lv}x_{i}.wav"))
        R[f"concurrent_x{lv}"] = results

    # long text
    if not args.skip_long:
        gm3, c3 = _get_triton_client(args.triton)
        r = infer_stream(c3, gm3, {"text": _LONG_TEXT, "task_type": "custom_voice",
                          "speaker": "Serena", "session_id": "longtext"}, timeout=180)
        if r.audio is not None and r.audio.size > 0:
            save_wav(r.audio, str(output_dir / "test4_long_text.wav"))
        R["long_text"] = [r]

    # badcases
    if not args.skip_badcase:
        gm4, _ = _get_triton_client(args.triton)
        for label, req in [("empty", {"text":"", "task_type":"custom_voice", "speaker":"Serena", "session_id":"bc-e"}),
                           ("ws", {"text":"   \n\t  ", "task_type":"custom_voice", "speaker":"Serena", "session_id":"bc-w"}),
                           ("inv-task", {"text":"测试", "task_type":"nonexistent_task", "session_id":"bc-t"})]:
            c = gm4.InferenceServerClient(url=args.triton)
            r = infer_stream(c, gm4, req, timeout=15)
            print(f"  badcase {label}: error={r.error is not None}")

    print("\n" + "=" * 60 + "\n  FINAL SUMMARY\n" + "=" * 60)
    ok, fail = _summary(R)
    print(f"  Output: {output_dir.resolve()}")
    return 0 if fail == 0 else 1

# -- CLI --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified TTS benchmark tool")
    p.add_argument("--target", required=True, choices=["engine-standalone", "triton-concurrent"])
    g1 = p.add_argument_group("engine-standalone")
    g1.add_argument("--host", default="localhost"); g1.add_argument("--port", type=int, default=50051)
    g1.add_argument("--stress-concurrency", type=int, default=0)
    g1.add_argument("--stress-rounds", type=int, default=0)
    g1.add_argument("--stress-warmup-rounds", type=int, default=1)
    g1.add_argument("--skip-single", action="store_true")
    g1.add_argument("--skip-custom-instruct", action="store_true")
    g2 = p.add_argument_group("triton-concurrent")
    g2.add_argument("--triton", default="localhost:8001")
    p.add_argument("--concurrency", default="1,2,4", help="Comma-separated levels")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--skip-streaming", action="store_true")
    p.add_argument("--skip-long", action="store_true")
    p.add_argument("--skip-badcase", action="store_true")
    p.add_argument("--skip-concurrent", action="store_true")
    return p

def main(argv=None) -> int:
    _print_deprecation()
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        if args.target == "engine-standalone":
            from tests.support.engine_standalone import OUTPUT_DIR
            args.output_dir = str(OUTPUT_DIR)
        else:
            args.output_dir = "workspace/test_concurrent_output"
    return _run_engine_standalone(args) if args.target == "engine-standalone" else _run_triton_concurrent(args)

if __name__ == "__main__":
    sys.exit(main())
