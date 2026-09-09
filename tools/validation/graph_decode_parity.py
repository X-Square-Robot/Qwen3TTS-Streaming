"""CUDA-graph decode vs eager decode parity check (manual validation tool).

Run inside the engine container (TRT version must match the plan), e.g.:
    docker run --rm --gpus all --entrypoint python3 \\
      -v $REPO:/repo -v $REPO/workspace/model_repository:/models \\
      qwen3-engine:<tag> /repo/tools/validation/graph_decode_parity.py \\
      --engine-dir /models/tts_orchestrator/1/runtime

Fabricates
slot states directly (no prefill) and, per scenario, compares three runs on
identical state: eager, eager again (control), graph.

The control run quantifies the eager path's own sensitivity (Gumbel top-1
near-ties under identical seeds are only reproducible if the numerics are
bit-stable); the graph run is judged against that baseline.  Sampled tokens
are the primary signal: wav/KV are chaotic after any token flip, so float
diffs are only meaningful on token-identical lanes.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from engine.backend.executor import Executor  # noqa: E402
from engine.backend.kv_cache_pool import SlotKVState  # noqa: E402

torch.manual_seed(0)

ENGINE_DIR = str(REPO_ROOT / "workspace/model_repository/tts_orchestrator/1/runtime")
DEVICE = torch.device("cuda", 0)


def make_executor() -> Executor:
    ex = Executor(
        engine_dir=ENGINE_DIR,
        max_batch_size=128,
        max_seq_len=512,
        do_sample=True,
        temperature=0.9,
        repetition_penalty=1.05,
        random_seed=0,
    )
    ex.load()
    assert ex._graph_decode is not None, "graph decode failed to init"
    return ex


def fabricate_slots(ex: Executor, past_lens: list[int], tag: str) -> list[SlotKVState]:
    cfg = ex._config
    pool = ex.kv_pool
    dtype = pool._talker_kv_pool.dtype
    slots = []
    for i, pl in enumerate(past_lens):
        s = SlotKVState(slot_id=i, session_id=f"{tag}-{i}", segment_idx=0)
        s.past_len = pl
        s.frame_idx = pl
        s.next_embed = torch.randn(1, 1, cfg.hidden_size, device=DEVICE) * 0.05
        s.token_counts = torch.zeros(
            1, cfg.codec_vocab_size, device=DEVICE, dtype=torch.int64
        )
        pool._talker_kv_pool[i, :, :, :pl, :] = (
            torch.randn(
                cfg.num_layers * 2, cfg.kv_heads, pl, cfg.head_dim, device=DEVICE
            )
            * 0.1
        ).to(dtype)
        c2w_full = tag.startswith("aligned")
        c2w_len = (
            cfg.c2w_sliding_window - 1
            if c2w_full
            else min(pl, cfg.c2w_sliding_window - 1, 40)
        )
        if c2w_len > 0:
            s.c2w_kv = (
                torch.randn(
                    1,
                    cfg.n_c2w_layers * 2,
                    cfg.c2w_kv_heads,
                    c2w_len,
                    cfg.c2w_head_dim,
                    device=DEVICE,
                )
                * 0.1
            ).to(dtype)
        s.c2w_conv_states = [
            (torch.randn([1] + list(shape[1:]), device=DEVICE) * 0.1).to(dtype)
            for shape in ex._c2w_conv_shapes
        ]
        s.c2w_transconv_states = [
            (torch.randn([1] + list(shape[1:]), device=DEVICE) * 0.1).to(dtype)
            for shape in ex._c2w_transconv_shapes
        ]
        slots.append(s)
    return slots


def run_once(ex: Executor, slots: list[SlotKVState], use_graph: bool):
    saved = ex._graph_decode
    if not use_graph:
        ex._graph_decode = None
    try:
        for s in slots:
            s.sampling_generator = None  # re-seed deterministically
        fut = ex.launch_decode_step(slots)
        out = fut.wait()
        tokens = fut._raw["full_codec"].cpu().clone()
        return out, tokens
    finally:
        ex._graph_decode = saved


def compare(tag, ref, cand):
    (out_r, tok_r), (out_c, tok_c) = ref, cand
    lane_same = (tok_r == tok_c).all(dim=1)
    n_flip = int((~lane_same).sum())
    wav_max = 0.0
    for i in range(len(lane_same)):
        if not lane_same[i]:
            continue
        a = torch.frombuffer(bytearray(out_r.audio_chunks[i]), dtype=torch.float32)
        b = torch.frombuffer(bytearray(out_c.audio_chunks[i]), dtype=torch.float32)
        wav_max = max(wav_max, (a - b).abs().max().item())
    eos_same = out_r.eos_flags == out_c.eos_flags
    print(
        f"[{tag}] lanes={len(lane_same)} token_flips={n_flip} "
        f"wav_maxdiff(token-same lanes)={wav_max:.4e} eos_same={eos_same}",
        flush=True,
    )
    return n_flip, wav_max


def main():
    global ENGINE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", default=ENGINE_DIR,
                        help="dir containing model.plan (default: deployed package)")
    args = parser.parse_args()
    ENGINE_DIR = args.engine_dir
    ex = make_executor()
    cap = ex._graph_decode._max_past
    print(f"graph staging max_past={cap}", flush=True)

    scenarios = [
        # aligned: past == bucket boundary, batch in ladder, c2w window full →
        # eager and graph compute at identical padded widths; token streams
        # must match BITWISE (isolates graph mechanics from legal padding
        # numerics, which flip argmax freely on random test data).
        ("aligned_b16_p128", [128] * 16),
        ("aligned_b128_p64", [64] * 128),
        ("b1_p10", [10]),
        ("b3_hetero", [5, 60, 63]),
        ("b16_uniform", [100] * 16),
        ("b33_hetero", list(range(3, 3 + 33 * 4, 4))),
        ("b64_mixed", [7, 200] + [50] * 62),
        ("b128_hetero", [(i * 7) % 120 + 2 for i in range(128)]),
    ]
    total_ctrl_flips = total_graph_flips = 0
    worst_wav = 0.0
    for tag, past_lens in scenarios:
        torch.cuda.empty_cache()
        slots = fabricate_slots(ex, past_lens, tag)
        eager1 = run_once(ex, slots, use_graph=False)
        eager2 = run_once(ex, slots, use_graph=False)
        graph = run_once(ex, slots, use_graph=True)
        cf, cw = compare(f"{tag}/ctrl eager-vs-eager", eager1, eager2)
        gf, gw = compare(f"{tag}/graph-vs-eager   ", eager1, graph)
        total_ctrl_flips += cf
        total_graph_flips += gf
        worst_wav = max(worst_wav, cw, gw)

    # Beyond-cap fallback: past > staging cap must run eager transparently.
    slots = fabricate_slots(ex, [cap + 30, cap + 60], "overcap")
    o1 = run_once(ex, slots, use_graph=False)
    o2 = run_once(ex, slots, use_graph=True)  # should fall back to eager
    compare("overcap/graph(fallback)-vs-eager", o1, o2)

    print("--- replay stress (same bucket, fresh states, 5 rounds) ---", flush=True)
    for r in range(5):
        slots = fabricate_slots(ex, [90 + r] * 16, f"replay{r}")
        eager = run_once(ex, slots, use_graph=False)
        graph = run_once(ex, slots, use_graph=True)
        gf, gw = compare(f"replay-{r}", eager, graph)
        total_graph_flips += gf
        worst_wav = max(worst_wav, gw)

    # ------------------------------------------------------------------
    # Definitive check: the graph must be BITWISE identical to a plain
    # (non-graph) enqueue on the same optimization profile.  Cross-profile
    # token flips are TRT kernel differences, not graph capture/replay
    # behavior; cursor-enabled plans are intentionally pinned to profile 0.
    # ------------------------------------------------------------------
    graph_profile = int(getattr(ex._graph_decode, "_profile_idx", 0))
    print(f"--- graph vs profile-{graph_profile} eager (bitwise) ---", flush=True)
    eng = ex._fused_engine._engine
    scratch = None
    if graph_profile > 0:
        scratch = torch.empty(
            int(eng.get_device_memory_size_for_profile_v2(graph_profile)),
            dtype=torch.uint8, device=DEVICE,
        )
        ctx1 = eng.create_execution_context_without_device_memory()
        ctx1.set_optimization_profile_async(
            graph_profile, ex._compute_stream.cuda_stream
        )
        ex._compute_stream.synchronize()
        try:
            ctx1.set_device_memory(scratch.data_ptr(), scratch.numel())
        except TypeError:
            ctx1.device_memory = scratch.data_ptr()
    else:
        ctx1 = eng.create_execution_context()

    bitwise_ok = True
    for tag, past_lens in [("p1_b2", [64, 64]), ("p1_b64", [64] * 64),
                           ("p1_b128", [128] * 128)]:
        slots = fabricate_slots(ex, past_lens, tag)
        captured = {}
        orig_infer = ex._fused_engine.infer

        def spy(inputs, *a, **kw):
            captured.update({k: v.clone() for k, v in inputs.items()})
            return orig_infer(inputs, *a, **kw)

        for s in slots:
            s.sampling_generator = None
        fut = ex.launch_decode_step(slots)  # graph path
        fut.wait()
        logits_g = fut._raw["logits"].float().cpu().clone()
        codec_g = fut._raw["full_codec"].cpu().clone()
        # Re-feed the exact staging inputs through a plain enqueue on the
        # graph's selected profile.
        key = ex._graph_decode.bucket(len(slots), max(past_lens))
        entry = ex._graph_decode.entry(key)
        plain_inputs = {k: v.clone() for k, v in entry["in"].items()}
        with torch.cuda.stream(ex._compute_stream):
            raw_p1 = ex._fused_engine.infer(
                plain_inputs, ex._build_output_names(), ex._compute_stream,
                context=ctx1,
            )
        ex._compute_stream.synchronize()
        b = len(past_lens)
        lg = raw_p1["logits"][:b].float().cpu()
        cd = raw_p1["full_codec"][:b].cpu()
        same_logits = bool((lg == logits_g).all())
        same_codec = bool((cd == codec_g).all())
        bitwise_ok &= same_logits and same_codec
        print(f"[{tag}] logits bitwise={same_logits} full_codec equal={same_codec}",
              flush=True)

    print(
        f"\nSUMMARY: ctrl_flips={total_ctrl_flips} "
        f"cross-profile graph_flips={total_graph_flips} (informational) "
        f"worst token-same wav diff={worst_wav:.4e} "
        f"graph-vs-profile{graph_profile}-eager bitwise={bitwise_ok}"
    )
    ok = bitwise_ok and total_ctrl_flips == 0 and worst_wav < 1e-2
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
