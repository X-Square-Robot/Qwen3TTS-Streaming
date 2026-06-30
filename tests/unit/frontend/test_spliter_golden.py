"""Golden characterization tests for the frontend Spliter / driver / emoji filter.

These pin the **current** SegmentAction behavior so the planned segmentation
pipeline refactor (see docs/dev/design/frontend_segmentation_pipeline.md) can be
done with a safety net. They are CHARACTERIZATION tests: they encode behavior as
it is today, captured empirically, not behavior as it ought to be.

Some assertions deliberately encode behavior the refactor will CHANGE. Those are
marked with ``# WILL CHANGE @ Step N`` — when that step lands, update the golden
value and document the diff in the commit. Specifically:

  * group_idx == -1 sentinel on the streaming path        -> Step 2 (explicit coords)
  * streaming L2-snap fragmentation of the signature case  -> Step 3/5 (auto + bin-packing)
  * cross-packet emoji leak                                -> Step 5 (stateful Stage 0 filter)

The signature case ``你好吗？明天天气不错，有没有什么想吃的？`` is the crux of the
whole redesign: offline pre-split finds the global-optimal L1 cut, while the
streaming driver, lacking foresight, snaps to the L2 comma. Both are frozen here.
"""
from __future__ import annotations

import pytest

from engine.frontend.spliter.spliter import Spliter
from engine.text_normalization import strip_emoji, split_pending_emoji


SIGNATURE = "你好吗？明天天气不错，有没有什么想吃的？"


def _toks(s: str):
    return [(i, ch) for i, ch in enumerate(s)]


def _summarize(actions):
    """Reduce an action stream to per-segment summaries.

    Returns a list of dicts, one per contiguous segment_idx run:
    {seg, group, local, final, text, acts} where ``text`` is the concatenation
    of token_text (captures which characters landed in which segment = the
    segmentation boundaries) and ``acts`` is the action-type sequence (captures
    PREFILL/DECODE/FLUSH structure).
    """
    segs: list[dict] = []
    for a in actions:
        if not segs or segs[-1]["seg"] != a.segment_idx:
            segs.append({
                "seg": a.segment_idx,
                "group": a.group_idx,
                "local": a.local_idx,
                "final": a.group_final,
                "text": "",
                "acts": [],
            })
        segs[-1]["acts"].append(a.action.type.name)
        segs[-1]["text"] += a.token_text
    return segs


def test_offline_set_full_text_finds_global_optimal_l1_cut():
    """Offline pre-split keeps the long second sentence whole, cutting only at L1.

    This is the global-optimal behavior the streaming driver cannot achieve.
    """
    sp = Spliter(engine_max_decode_len=100, ema_ratio=10.0)
    segs = _summarize(sp.set_full_text(_toks(SIGNATURE)))

    # NOTE: a flush-triggering punct emits FLUSH_EOS with empty token_text, so
    # seg1's terminal "？" does not appear in the reconstructed text (it triggered
    # the flush). seg0's "？" appears because that group flushes on END, not on
    # the punct. This per-path asymmetry is itself frozen here.
    assert [(s["seg"], s["group"], s["text"]) for s in segs] == [
        (0, 0, "你好吗？"),
        (1, 1, "明天天气不错，有没有什么想吃的"),
    ]
    assert segs[0]["acts"] == ["PREFILL", "DECODE", "DECODE", "DECODE", "FLUSH_EOS"]
    assert segs[1]["acts"][0] == "PREFILL"
    assert segs[1]["acts"][-1] == "FLUSH_EOS"
    assert all(s["final"] for s in segs)


def test_streaming_feed_once_snaps_to_l2_comma():
    """Streaming path fragments at the L2 comma — the local-optimum problem.

    Since Step 2, the streaming path emits explicit coordinates (group_idx ==
    segment_idx, each segment its own group) instead of the -1 sentinel.

    # WILL CHANGE @ Step 3/5: auto + bin-packing + watermark should let the
    # streaming/auto path approach the offline cut instead of snapping to L2.
    """
    sp = Spliter(engine_max_decode_len=100, ema_ratio=10.0)
    segs = _summarize(sp.feed_tokens(_toks(SIGNATURE)))

    assert [(s["seg"], s["group"], s["text"]) for s in segs] == [
        (0, 0, "你好吗？明天天气不错，"),   # WILL CHANGE @ Step 3/5 (L2 snap)
        (1, 1, "有没有什么想吃的？"),
    ]


def test_streaming_token_by_token_matches_feed_once():
    """Feeding the signature one token per call yields the same segmentation."""
    sp = Spliter(engine_max_decode_len=100, ema_ratio=10.0)
    acc = []
    for t in _toks(SIGNATURE):
        acc.extend(sp.feed_tokens([t]))
    acc.extend(sp.input_done())
    segs = _summarize(acc)

    assert [(s["seg"], s["text"]) for s in segs] == [
        (0, "你好吗？明天天气不错，"),
        (1, "有没有什么想吃的？"),
    ]


def test_push_group_tokens_monotonic_groups():
    """Long-segment groups get monotonic group_idx, one segment each here."""
    sp = Spliter(engine_max_decode_len=100, ema_ratio=2.0)
    acc = []
    acc.extend(sp.push_group_tokens([(1, "你好"), (2, "。")]))
    acc.extend(sp.push_group_tokens([(3, "世界"), (4, "。")]))
    segs = _summarize(acc)

    assert [(s["seg"], s["group"], s["text"]) for s in segs] == [
        (0, 0, "你好。"),
        (1, 1, "世界。"),
    ]


def test_concurrency_backpressure_gates_at_max_concurrent():
    """With max_concurrent=2, only 2 segments drive synchronously; rest buffer.

    Pins the backpressure semantics: buffered tokens wait for on_segment_done
    (backend feedback) to free a slot, so a 4-sentence input emits 2 segments
    synchronously and leaves both drivers flushing.
    """
    sp = Spliter(engine_max_decode_len=100, ema_ratio=10.0, max_concurrent=2)
    text = "第一句话结束了。第二句话也结束了。第三句话同样结束了。第四句话最后结束。"
    acc = sp.feed_tokens(_toks(text))
    acc.extend(sp.input_done())
    segs = _summarize(acc)

    assert [(s["seg"], s["text"]) for s in segs] == [
        (0, "第一句话结束了。"),
        (1, "第二句话也结束了。"),
    ]
    assert sorted(sp._drivers.keys()) == [0, 1]
    assert sorted(sp._flushing) == [0, 1]
    assert sp._next_segment_idx == 2


def test_auto_long_packet_engages_stage1_global_optimal():
    """Auto mode: a packet longer than one segment is pre-split with foresight,
    yielding the SAME global-optimal L1 cut as offline set_full_text."""
    sp = Spliter(engine_max_decode_len=100, ema_ratio=10.0)
    segs = _summarize(sp.feed_auto(_toks(SIGNATURE)))   # 20 tokens > force_split_at(15)

    assert [(s["seg"], s["group"], s["text"]) for s in segs] == [
        (0, 0, "你好吗？"),
        (1, 1, "明天天气不错，有没有什么想吃的"),
    ]
    # Matches the offline path exactly — Stage 1 had full foresight over the packet.
    off = _summarize(Spliter(engine_max_decode_len=100, ema_ratio=10.0).set_full_text(_toks(SIGNATURE)))
    assert [(s["seg"], s["group"], s["text"]) for s in segs] == \
           [(s["seg"], s["group"], s["text"]) for s in off]


def test_auto_token_by_token_is_transparent_streaming():
    """Auto mode: each 1-token packet is small, so Stage 1 is transparent and
    the tokens stream/coalesce — identical to plain feed_tokens (incl. L2-snap,
    which is the irreducible no-foresight cost of a true token stream)."""
    sp = Spliter(engine_max_decode_len=100, ema_ratio=10.0)
    acc = []
    for t in _toks(SIGNATURE):
        acc.extend(sp.feed_auto([t]))
    acc.extend(sp.input_done())
    segs = _summarize(acc)

    assert [(s["seg"], s["group"], s["text"]) for s in segs] == [
        (0, 0, "你好吗？明天天气不错，"),
        (1, 1, "有没有什么想吃的？"),
    ]


def test_auto_short_packet_does_not_overfragment():
    """Auto mode: a short complete packet streams transparently (driver
    coalesces) rather than flushing a tiny segment per L1."""
    sp = Spliter(engine_max_decode_len=100, ema_ratio=10.0)
    acc = sp.feed_auto(_toks("你好。"))
    acc.extend(sp.input_done())
    segs = _summarize(acc)

    assert [(s["seg"], s["text"]) for s in segs] == [(0, "你好。")]


def test_auto_mixed_streaming_then_long_packet_no_group_collision():
    """Auto mode mixing streaming + offline in one session must not collide
    group ids. A small streaming residual (its own group) followed by a long
    packet that engages Stage 1 (occupancy-aware gate) must yield distinct
    group ids — streaming and offline groups share one _next_group_idx
    namespace. Regression guard for the namespace-collision bug."""
    sp = Spliter(engine_max_decode_len=100, ema_ratio=10.0)
    a1 = sp.feed_auto(_toks("今天天气真的很不错"))           # 9 tok, no L1 → streams (group 0)
    a2 = sp.feed_auto(_toks("，我们出去玩吧。好不好呀？"))   # won't fit remaining room → Stage 1

    groups_by_seg: dict[int, int] = {}
    for sa in a1 + a2:
        groups_by_seg.setdefault(sa.segment_idx, sa.group_idx)

    assert groups_by_seg[0] == 0, groups_by_seg                     # streaming residual, own group
    assert all(g != 0 for s, g in groups_by_seg.items() if s != 0), groups_by_seg  # no collision


def test_emoji_whole_keycap_in_one_packet_is_stripped():
    """A complete keycap sequence in one packet strips cleanly (base digit too)."""
    assert strip_emoji("第1️⃣步完成✅。") == "第步完成。"


def _stage0_stream(packets):
    """Simulate the stateful Stage-0 filter: hold a partial-emoji suffix across
    packets (split_pending_emoji), strip each emitted body, flush the carry."""
    carry = ""
    out = []
    for p in packets:
        body, carry = split_pending_emoji(carry + p)
        out.append(strip_emoji(body))
    out.append(strip_emoji(carry))   # end-of-input flush
    return "".join(out)


def test_emoji_keycap_split_across_packets_healed_by_carry():
    """FIXED: a keycap split across packets no longer leaks the base digit. The
    stateful Stage-0 carry holds the trailing base until the next packet
    completes (or flushes) the sequence."""
    # base | VS+keycap
    assert _stage0_stream(["hello1", "️⃣world"]) == "helloworld"
    # base+VS | keycap  (2-char hold)
    assert _stage0_stream(["tier1️", "⃣done"]) == "tierdone"
    assert "1" not in _stage0_stream(["hello1", "️⃣world"])


def test_emoji_carry_does_not_drop_normal_trailing_digits():
    """A held digit that turns out NOT to be a keycap is emitted intact — the
    carry never loses normal numeric text, only delays it by one packet."""
    assert _stage0_stream(["price5", "6dollars"]) == "price56dollars"
    assert _stage0_stream(["count3"]) == "count3"        # flushed at end of input
