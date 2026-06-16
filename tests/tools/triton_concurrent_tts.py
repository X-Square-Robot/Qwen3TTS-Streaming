#!/usr/bin/env python3
"""
Multi-session concurrent TTS test & performance benchmark.

Tests the continuous batching orchestrator with:
  1. Single-request smoke test (backward compat with action=synthesize)
  2. Streaming text input (init -> append_text -> text_complete)
  3. Multi-session concurrent requests (2, 4, 8 sessions)
  4. Performance metrics: first-chunk latency, total latency, throughput

Usage:
    conda activate qwen3-tts
    python tests/tools/triton_concurrent_tts.py [--triton localhost:8001] [--concurrency 1,2,4,8]
"""

import argparse
import sys
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from tests.support.triton_streaming import (
    StreamResult as TTSResult,
    infer_stream,
    infer_text_stream,
    save_wav,
)


def _get_client(triton_url: str):
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print("ERROR: tritonclient[grpc] not installed")
        sys.exit(1)
    return grpcclient, grpcclient.InferenceServerClient(url=triton_url)


def _send_request(client, grpcclient, req_dict: dict, timeout: float = 120.0) -> TTSResult:
    """Send a single TTS request and collect streaming audio."""
    return infer_stream(client, grpcclient, req_dict, timeout=timeout)


def _send_streaming_request(
    triton_url: str,
    grpcclient_mod,
    init_req: dict,
    text_chunks: list[str],
    chunk_delay_ms: float = 50.0,
    timeout: float = 120.0,
) -> TTSResult:
    """Send a streaming (init -> append_text* -> text_complete) request sequence."""
    return infer_text_stream(
        triton_url,
        grpcclient_mod,
        init_req,
        text_chunks,
        chunk_delay_ms=chunk_delay_ms,
        timeout=timeout,
    )


def _print_result(result: TTSResult, label: str = ""):
    prefix = f"  [{label}]" if label else "  "
    if result.error:
        print(f"{prefix} ERROR: {result.error}")
        return
    if result.warnings:
        for w in result.warnings:
            print(f"{prefix} WARNING: {w}")
    fc = f"{result.first_chunk_ms:.0f}ms" if result.first_chunk_ms is not None else "N/A"
    print(
        f"{prefix} session={result.session_id}"
        f"  first_chunk={fc}"
        f"  total={result.total_ms:.0f}ms"
        f"  chunks={result.num_chunks}"
        f"  audio={result.duration_sec:.2f}s"
        f"  RTF={result.rtf:.2f}"
    )


def _print_summary(results: list[TTSResult], label: str):
    ok = [r for r in results if r.error is None and r.first_chunk_ms is not None]
    fail = [r for r in results if r.error is not None]
    print(f"\n  ── {label} Summary ──")
    print(f"  Total: {len(results)}  OK: {len(ok)}  Failed: {len(fail)}")
    if not ok:
        return
    first_chunks = [r.first_chunk_ms for r in ok]
    totals = [r.total_ms for r in ok]
    rtfs = [r.rtf for r in ok if r.rtf > 0]
    durations = [r.duration_sec for r in ok]
    print(f"  First-chunk latency:  min={min(first_chunks):.0f}ms  "
          f"median={statistics.median(first_chunks):.0f}ms  "
          f"max={max(first_chunks):.0f}ms  "
          f"mean={statistics.mean(first_chunks):.0f}ms")
    print(f"  Total latency:        min={min(totals):.0f}ms  "
          f"median={statistics.median(totals):.0f}ms  "
          f"max={max(totals):.0f}ms  "
          f"mean={statistics.mean(totals):.0f}ms")
    if rtfs:
        print(f"  RTF (wall/audio):     min={min(rtfs):.2f}  "
              f"median={statistics.median(rtfs):.2f}  "
              f"max={max(rtfs):.2f}")
    total_audio = sum(durations)
    wall_time = max(totals) / 1000
    effective_throughput = total_audio / wall_time if wall_time > 0 else 0
    print(f"  Aggregate throughput: {total_audio:.1f}s audio / {wall_time:.1f}s wall "
          f"= {effective_throughput:.2f}x realtime")


TEST_TEXTS = [
    "你好，今天天气真好。",
    "欢迎来到人工智能语音合成的世界。",
    "技术创新推动着社会不断前进。",
    "我们正在测试多路并发的语音合成能力。",
    "这是第五路测试文本，用来验证系统的处理能力。",
    "深度学习让机器能够理解和生成自然的语音。",
    "云计算和边缘计算相结合，提供更好的用户体验。",
    "每一次迭代都让我们的系统变得更加完善。",
]

LONG_TEXT = (
    "人工智能正在深刻改变我们的世界。"
    "从语音识别到自然语言处理，从计算机视觉到机器人技术，"
    "AI的应用已经渗透到生活的方方面面。"
    "在医疗领域，AI可以辅助诊断疾病、发现新药物。"
    "在教育领域，AI可以提供个性化的学习方案。"
    "在交通领域，自动驾驶技术正在逐步成熟。"
    "未来，人工智能将继续推动社会进步，"
    "为人类创造更多的可能性。"
)

VERY_LONG_TEXT = (
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
)


def test_single_smoke(triton_url: str, output_dir: Path):
    """Test 1: Single request smoke test (backward compat)."""
    print("\n" + "=" * 60)
    print("  Test 1: Single Request Smoke Test")
    print("=" * 60)

    grpcclient_mod, client = _get_client(triton_url)

    result = _send_request(client, grpcclient_mod, {
        "text": "你好，这是单路测试。",
        "task_type": "custom_voice",
        "speaker": "Serena",
    })
    _print_result(result, "single")
    if result.audio is not None and result.audio.size > 0:
        out_path = str(output_dir / "test1_single_smoke.wav")
        save_wav(result.audio, out_path)
        print(f"  Saved: {out_path}")

    return result


def test_streaming_text(triton_url: str, output_dir: Path):
    """Test 2: Streaming text input (init -> append_text -> text_complete)."""
    print("\n" + "=" * 60)
    print("  Test 2: Streaming Text Input")
    print("=" * 60)

    grpcclient_mod, _ = _get_client(triton_url)

    import uuid
    session_id = uuid.uuid4().hex[:12]

    init_req = {
        "action": "init",
        "session_id": session_id,
        "task_type": "custom_voice",
        "speaker": "Serena",
        "text": "你好，这是流式文本输入测试。",
    }
    text_chunks = [
        "我们正在验证",
        "文本追加功能",
        "是否工作正常。",
    ]

    result = _send_streaming_request(
        triton_url, grpcclient_mod, init_req, text_chunks,
        chunk_delay_ms=200,
    )
    _print_result(result, "stream")
    if result.audio is not None and result.audio.size > 0:
        out_path = str(output_dir / "test2_streaming_text.wav")
        save_wav(result.audio, out_path)
        print(f"  Saved: {out_path}")

    return result


def test_concurrent(triton_url: str, concurrency: int, output_dir: Path):
    """Test 3: Multi-session concurrent requests."""
    print("\n" + "=" * 60)
    print(f"  Test 3: Concurrent Requests (concurrency={concurrency})")
    print("=" * 60)

    grpcclient_mod = _get_client(triton_url)[1].__class__.__module__
    import importlib
    grpcclient_mod = importlib.import_module(grpcclient_mod.rsplit('.', 1)[0])

    def _run_one(idx: int) -> TTSResult:
        text = TEST_TEXTS[idx % len(TEST_TEXTS)]
        client = grpcclient_mod.InferenceServerClient(url=triton_url)
        return _send_request(client, grpcclient_mod, {
            "text": text,
            "task_type": "custom_voice",
            "speaker": "Serena",
            "session_id": f"concurrent-{idx}",
        })

    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_run_one, i): i for i in range(concurrency)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                _print_result(result, f"c{idx}")
            except Exception as e:
                print(f"  [c{idx}] EXCEPTION: {e}")
                results.append(TTSResult(session_id=f"concurrent-{idx}", text="", error=str(e)))

    wall_time = time.perf_counter() - t0
    print(f"\n  Wall time for {concurrency} concurrent: {wall_time:.2f}s")

    _print_summary(results, f"Concurrent x{concurrency}")

    for i, r in enumerate(results):
        if r.audio is not None and r.audio.size > 0:
            out_path = str(output_dir / f"test3_concurrent_{concurrency}x_{i}.wav")
            save_wav(r.audio, out_path)

    return results


def test_long_text(triton_url: str, output_dir: Path):
    """Test 4: Long text rollover (medium + very long)."""
    print("\n" + "=" * 60)
    print("  Test 4: Long Text Rollover")
    print("=" * 60)

    grpcclient_mod, client = _get_client(triton_url)

    results = []

    # 4a: medium long text
    print("\n  --- 4a: Medium long text ---")
    r1 = _send_request(client, grpcclient_mod, {
        "text": LONG_TEXT,
        "task_type": "custom_voice",
        "speaker": "Serena",
        "session_id": "longtext-medium",
    }, timeout=180)
    _print_result(r1, "medium-long")
    if r1.audio is not None and r1.audio.size > 0:
        out_path = str(output_dir / "test4a_long_text_medium.wav")
        save_wav(r1.audio, out_path)
        print(f"  Saved: {out_path}")
    results.append(r1)

    # 4b: very long text — should trigger multi-segment rollover + dynamic split
    print("\n  --- 4b: Very long text (multi-segment) ---")
    client2 = grpcclient_mod.InferenceServerClient(url=triton_url)
    r2 = _send_request(client2, grpcclient_mod, {
        "text": VERY_LONG_TEXT,
        "task_type": "custom_voice",
        "speaker": "Serena",
        "session_id": "longtext-verylong",
    }, timeout=300)
    _print_result(r2, "very-long")
    if r2.audio is not None and r2.audio.size > 0:
        out_path = str(output_dir / "test4b_long_text_verylong.wav")
        save_wav(r2.audio, out_path)
        print(f"  Saved: {out_path}")
        print(f"  Audio duration: {r2.duration_sec:.2f}s "
              f"(~{len(VERY_LONG_TEXT)} chars)")
    results.append(r2)

    # 4c: streaming long text (init empty -> append chunks -> text_complete)
    print("\n  --- 4c: Streaming long text ---")
    import uuid
    sid = uuid.uuid4().hex[:12]
    sentences = [
        "人工智能正在深刻改变我们的世界。",
        "从语音识别到自然语言处理，从计算机视觉到机器人技术，AI的应用已经渗透到生活的方方面面。",
        "在医疗领域，AI可以辅助诊断疾病、发现新药物。在教育领域，AI可以提供个性化的学习方案。",
        "在交通领域，自动驾驶技术正在逐步成熟。未来，人工智能将继续推动社会进步，为人类创造更多的可能性。",
    ]
    r3 = _send_streaming_request(
        triton_url, grpcclient_mod,
        {"action": "init", "session_id": sid, "task_type": "custom_voice",
         "speaker": "Serena", "text": ""},
        sentences, chunk_delay_ms=300, timeout=180,
    )
    _print_result(r3, "stream-long")
    if r3.audio is not None and r3.audio.size > 0:
        out_path = str(output_dir / "test4c_streaming_long.wav")
        save_wav(r3.audio, out_path)
        print(f"  Saved: {out_path}")
    results.append(r3)

    # 4d: story — very long narrative from file (~2000 chars)
    story_path = Path(__file__).resolve().parents[2] / "tests" / "cases" / "story.txt"
    if story_path.is_file():
        print("\n  --- 4d: Story (long narrative) ---")
        story_text = story_path.read_text(encoding="utf-8").strip()
        print(f"  Story length: {len(story_text)} chars")
        client4 = grpcclient_mod.InferenceServerClient(url=triton_url)
        r4 = _send_request(client4, grpcclient_mod, {
            "text": story_text,
            "task_type": "custom_voice",
            "speaker": "Serena",
            "session_id": "longtext-story",
        }, timeout=600)
        _print_result(r4, "story")
        if r4.audio is not None and r4.audio.size > 0:
            out_path = str(output_dir / "test4d_story.wav")
            save_wav(r4.audio, out_path)
            print(f"  Saved: {out_path}")
            print(f"  Audio duration: {r4.duration_sec:.2f}s "
                  f"(~{len(story_text)} chars)")
        results.append(r4)
    else:
        print(f"\n  --- 4d: Story SKIPPED (not found: {story_path}) ---")

    return results


def test_badcases(triton_url: str, output_dir: Path):
    """Test 5: BadCase tests — boundary conditions and error handling."""
    print("\n" + "=" * 60)
    print("  Test 5: BadCase Tests")
    print("=" * 60)

    grpcclient_mod, _ = _get_client(triton_url)
    results = []
    import uuid

    # 5a: empty text (action=synthesize should fail)
    print("\n  --- 5a: Empty text (should error) ---")
    client_a = grpcclient_mod.InferenceServerClient(url=triton_url)
    r = _send_request(client_a, grpcclient_mod, {
        "text": "",
        "task_type": "custom_voice",
        "speaker": "Serena",
        "session_id": "badcase-empty",
    }, timeout=15)
    expected_error = r.error is not None
    print(f"  Got error: {expected_error} -> {'PASS' if expected_error else 'FAIL'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5a_empty_text", expected_error, r))

    # 5b: whitespace-only text (should error)
    print("\n  --- 5b: Whitespace-only text (should error) ---")
    client_b = grpcclient_mod.InferenceServerClient(url=triton_url)
    r = _send_request(client_b, grpcclient_mod, {
        "text": "   \n\t  ",
        "task_type": "custom_voice",
        "speaker": "Serena",
        "session_id": "badcase-whitespace",
    }, timeout=15)
    expected_error = r.error is not None
    print(f"  Got error: {expected_error} -> {'PASS' if expected_error else 'FAIL'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5b_whitespace_text", expected_error, r))

    # 5c: invalid task_type (should error)
    print("\n  --- 5c: Invalid task_type (should error) ---")
    client_c = grpcclient_mod.InferenceServerClient(url=triton_url)
    r = _send_request(client_c, grpcclient_mod, {
        "text": "测试",
        "task_type": "nonexistent_task",
        "session_id": "badcase-invalid-task",
    }, timeout=15)
    expected_error = r.error is not None
    print(f"  Got error: {expected_error} -> {'PASS' if expected_error else 'FAIL'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5c_invalid_task", expected_error, r))

    # 5d: streaming init with empty text, then text_complete without ever appending text
    print("\n  --- 5d: Streaming init empty -> text_complete (no text) ---")
    sid = uuid.uuid4().hex[:12]
    r = _send_streaming_request(
        triton_url, grpcclient_mod,
        {"action": "init", "session_id": sid, "task_type": "custom_voice",
         "speaker": "Serena", "text": ""},
        [],  # no text chunks
        chunk_delay_ms=100, timeout=15,
    )
    got_error_or_empty = r.error is not None or r.total_samples == 0
    print(f"  Error or empty: {got_error_or_empty} -> "
          f"{'PASS' if got_error_or_empty else 'FAIL (unexpected audio)'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5d_stream_no_text", got_error_or_empty, r))

    # 5e: append_text to nonexistent session
    print("\n  --- 5e: append_text to nonexistent session (should error) ---")
    client_e = grpcclient_mod.InferenceServerClient(url=triton_url)
    r = _send_request(client_e, grpcclient_mod, {
        "action": "append_text",
        "session_id": "nonexistent-session-999",
        "text": "hello",
    }, timeout=10)
    expected_error = r.error is not None
    print(f"  Got error: {expected_error} -> {'PASS' if expected_error else 'FAIL'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5e_append_nonexistent", expected_error, r))

    # 5f: single character text
    print("\n  --- 5f: Single character text ---")
    client_f = grpcclient_mod.InferenceServerClient(url=triton_url)
    r = _send_request(client_f, grpcclient_mod, {
        "text": "好",
        "task_type": "custom_voice",
        "speaker": "Serena",
        "session_id": "badcase-single-char",
    }, timeout=30)
    ok = r.error is None and r.total_samples > 0
    print(f"  Got audio: {ok} -> {'PASS' if ok else 'FAIL'}")
    _print_result(r, "single-char")
    if r.audio is not None and r.audio.size > 0:
        out_path = str(output_dir / "test5f_single_char.wav")
        save_wav(r.audio, out_path)
    results.append(("5f_single_char", ok, r))

    # 5g: pure punctuation
    print("\n  --- 5g: Pure punctuation text ---")
    client_g = grpcclient_mod.InferenceServerClient(url=triton_url)
    r = _send_request(client_g, grpcclient_mod, {
        "text": "。。。！！！",
        "task_type": "custom_voice",
        "speaker": "Serena",
        "session_id": "badcase-punctuation",
    }, timeout=30)
    _print_result(r, "punctuation")
    results.append(("5g_punctuation", True, r))

    # 5h: streaming with delayed text (simulate slow LLM)
    print("\n  --- 5h: Streaming with slow text input (1s delay) ---")
    sid = uuid.uuid4().hex[:12]
    r = _send_streaming_request(
        triton_url, grpcclient_mod,
        {"action": "init", "session_id": sid, "task_type": "custom_voice",
         "speaker": "Serena", "text": ""},
        ["你好，", "这是一个", "慢速输入的测试。"],
        chunk_delay_ms=1000, timeout=60,
    )
    ok = r.error is None and r.total_samples > 0
    print(f"  Got audio: {ok} -> {'PASS' if ok else 'FAIL'}")
    _print_result(r, "slow-stream")
    if r.audio is not None and r.audio.size > 0:
        out_path = str(output_dir / "test5h_slow_stream.wav")
        save_wav(r.audio, out_path)
    results.append(("5h_slow_stream", ok, r))

    # 5i: invalid speaker name (should return warning + fallback audio)
    print("\n  --- 5i: Invalid speaker name (should warn + use fallback) ---")
    client_i = grpcclient_mod.InferenceServerClient(url=triton_url)
    r = _send_request(client_i, grpcclient_mod, {
        "text": "这是一个无效说话人测试。",
        "task_type": "custom_voice",
        "speaker": "zhitian",
        "session_id": "badcase-invalid-speaker",
    }, timeout=30)
    got_warning = len(r.warnings) > 0
    got_audio = r.error is None and r.total_samples > 0
    ok = got_warning and got_audio
    print(f"  Got warning: {got_warning}, got audio: {got_audio} -> "
          f"{'PASS' if ok else 'FAIL'}")
    _print_result(r, "invalid-speaker")
    if r.audio is not None and r.audio.size > 0:
        out_path = str(output_dir / "test5i_invalid_speaker.wav")
        save_wav(r.audio, out_path)
    results.append(("5i_invalid_speaker", ok, r))

    # Summary
    print("\n  ── BadCase Summary ──")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, r in results:
        status = "PASS" if ok else "FAIL"
        err = f" error={r.error[:60]}" if r.error else ""
        samples = f" samples={r.total_samples}" if r.total_samples else ""
        print(f"  {status:4s}  {name:30s}{err}{samples}")
    print(f"\n  BadCase: {passed}/{total} passed")

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-session concurrent TTS test & benchmark")
    parser.add_argument("--triton", default="localhost:8001", help="Triton gRPC address")
    parser.add_argument("--concurrency", default="1,2,4", help="Concurrency levels (comma-separated)")
    parser.add_argument("--output-dir", default="workspace/test_concurrent_output",
                        help="Output directory for WAV files")
    parser.add_argument("--skip-streaming", action="store_true", help="Skip streaming text test")
    parser.add_argument("--skip-long", action="store_true", help="Skip long text rollover test")
    parser.add_argument("--skip-badcase", action="store_true", help="Skip badcase tests")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",")]

    print("=" * 60)
    print("  TTS Concurrent Test & Performance Benchmark")
    print("=" * 60)
    print(f"  Triton: {args.triton}")
    print(f"  Concurrency levels: {concurrency_levels}")
    print(f"  Output: {output_dir}")

    grpcclient_mod, client = _get_client(args.triton)
    if not client.is_server_ready():
        print("ERROR: Triton server not ready")
        sys.exit(1)
    if not client.is_model_ready("tts_orchestrator"):
        print("ERROR: tts_orchestrator model not ready")
        sys.exit(1)
    print("  Server: READY")

    all_results = {}

    # Test 1: Single smoke
    r = test_single_smoke(args.triton, output_dir)
    all_results["single"] = [r]

    # Test 2: Streaming text
    if not args.skip_streaming:
        r = test_streaming_text(args.triton, output_dir)
        all_results["streaming"] = [r]

    # Test 3: Concurrent at each level
    for level in concurrency_levels:
        results = test_concurrent(args.triton, level, output_dir)
        all_results[f"concurrent_x{level}"] = results

    # Test 4: Long text
    if not args.skip_long:
        long_results = test_long_text(args.triton, output_dir)
        all_results["long_text"] = long_results if isinstance(long_results, list) else [long_results]

    # Test 5: BadCases
    if not args.skip_badcase:
        test_badcases(args.triton, output_dir)

    # Final summary (functional tests — excludes badcase which has its own summary)
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY (Functional Tests)")
    print("=" * 60)
    for name, results in all_results.items():
        ok = sum(1 for r in results if r.error is None)
        fail = sum(1 for r in results if r.error is not None)
        first_chunks = [r.first_chunk_ms for r in results if r.first_chunk_ms is not None]
        avg_first = statistics.mean(first_chunks) if first_chunks else 0
        print(f"  {name:20s}  OK={ok}  FAIL={fail}  avg_first_chunk={avg_first:.0f}ms")

    total_ok = sum(1 for results in all_results.values() for r in results if r.error is None)
    total_fail = sum(1 for results in all_results.values() for r in results if r.error is not None)
    print(f"\n  TOTAL: {total_ok} OK, {total_fail} FAILED")
    print(f"  Output: {output_dir.resolve()}")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
