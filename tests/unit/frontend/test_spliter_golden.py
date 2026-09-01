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
whole redesign: packet-local pre-split uses hierarchical latest-fit, while the
streaming driver, lacking foresight, snaps to the L2 comma. Both are frozen here.
"""

from __future__ import annotations


from engine.frontend.spliter.spliter import Spliter
from engine.frontend.spliter.ratio import RatioOutcome
from engine.frontend.spliter.reorder import AudioReorder
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
            segs.append(
                {
                    "seg": a.segment_idx,
                    "group": a.group_idx,
                    "local": a.local_idx,
                    "final": a.group_final,
                    "text": "",
                    "acts": [],
                }
            )
        segs[-1]["acts"].append(a.action.type.name)
        segs[-1]["text"] += a.token_text
    return segs


def test_offline_set_full_text_uses_hierarchical_latest_fit():
    """Offline pre-split prefers the latest L1, then falls back to L2."""
    sp = Spliter(engine_max_decode_len=100, ema_ratio=10.0)
    segs = _summarize(sp.set_full_text(_toks(SIGNATURE)))

    # The resource-derived cap is 8. Bin-packing first cuts at the latest L1
    # ("你好吗？"), then cuts the next full buffer at its latest L2 (，),
    # leaving "有没有什么想吃的？" as a third group (pending under concurrency=2,
    # so not in this synchronous batch). The orphaned trailing "？" of the old
    # first-fit force-cut is gone.
    assert [(s["seg"], s["group"], s["text"]) for s in segs] == [
        (0, 0, "你好吗？"),
        (1, 1, "明天天气不错，"),
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
    # This fixture gives C=13, so the comma at token 11 reaches T2=11.
    sp = Spliter(engine_max_decode_len=150, ema_ratio=10.0)
    segs = _summarize(sp.feed_tokens(_toks(SIGNATURE)))

    assert [(s["seg"], s["group"], s["text"]) for s in segs] == [
        (0, 0, "你好吗？明天天气不错，"),  # WILL CHANGE @ Step 3/5 (L2 snap)
        (1, 1, "有没有什么想吃的？"),
    ]


def test_streaming_token_by_token_matches_feed_once():
    """Feeding the signature one token per call yields the same segmentation."""
    sp = Spliter(engine_max_decode_len=150, ema_ratio=10.0)
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
    # C=11 and T1=8, so each sentence-ending L1 remains an eligible cut.
    sp = Spliter(engine_max_decode_len=130, ema_ratio=10.0, max_concurrent=2)
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


def test_auto_long_packet_engages_stage1_hierarchical_latest_fit():
    """Auto mode: a packet longer than one segment is pre-split with foresight,
    yielding the same hierarchical latest-fit cuts as offline set_full_text."""
    sp = Spliter(engine_max_decode_len=100, ema_ratio=10.0)
    segs = _summarize(sp.feed_auto(_toks(SIGNATURE)))  # 20 tokens > C=8

    assert [(s["seg"], s["group"], s["text"]) for s in segs] == [
        (0, 0, "你好吗？"),
        (1, 1, "明天天气不错，"),
    ]
    # Matches the offline path exactly — Stage 1 had full foresight over the packet.
    off = _summarize(
        Spliter(engine_max_decode_len=100, ema_ratio=10.0).set_full_text(
            _toks(SIGNATURE)
        )
    )
    assert [(s["seg"], s["group"], s["text"]) for s in segs] == [
        (s["seg"], s["group"], s["text"]) for s in off
    ]


def test_auto_token_by_token_is_transparent_streaming():
    """Auto mode: each 1-token packet is small, so Stage 1 is transparent and
    the tokens stream/coalesce — identical to plain feed_tokens (incl. L2-snap,
    which is the irreducible no-foresight cost of a true token stream)."""
    sp = Spliter(engine_max_decode_len=150, ema_ratio=10.0)
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
    sp = Spliter(engine_max_decode_len=150, ema_ratio=10.0)
    acc = sp.feed_auto(_toks("你好。"))
    acc.extend(sp.input_done())
    segs = _summarize(acc)

    assert [(s["seg"], s["text"]) for s in segs] == [(0, "你好。")]


def test_auto_mixed_streaming_then_long_packet_no_group_collision():
    """Auto mode mixing streaming + offline in one session must not collide
    group ids. A small streaming residual (its own group) followed by a long
    packet that engages Stage 1 (occupancy-aware gate) must yield distinct
    group ids. Public reorder ids are assigned in FIFO open order, independently
    from offline planning keys. Regression guard for namespace collisions."""
    sp = Spliter(engine_max_decode_len=150, ema_ratio=10.0)
    a1 = sp.feed_auto(_toks("今天天气真的很不错"))  # 9 tokens fit C=13
    a2 = sp.feed_auto(
        _toks("，我们出去玩吧。好不好呀？")
    )  # won't fit remaining room → Stage 1

    groups_by_seg: dict[int, int] = {}
    for sa in a1 + a2:
        groups_by_seg.setdefault(sa.segment_idx, sa.group_idx)

    assert groups_by_seg[0] == 0, groups_by_seg  # streaming residual, own group
    assert all(g != 0 for s, g in groups_by_seg.items() if s != 0), (
        groups_by_seg
    )  # no collision


def test_auto_backpressure_keeps_stream_before_later_offline_groups():
    """Pending streaming text must not be reordered behind a later long packet.

    The first segment occupies the only slot while flushing. A short streaming
    packet queues before a long Stage-1 packet. Offline planning may allocate
    internal keys eagerly, but public AudioReorder group ids must follow actual
    FIFO open order: first segment, queued stream, then offline groups.
    """
    sp = Spliter(
        engine_max_decode_len=120,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=2.0,
        safety_ratio_initial=2.0,
        ema_overflow_alpha=1.0,
        safety_failure_multiplier=1.0,
        max_concurrent=1,
    )

    first = sp.feed_tokens([(token, "a") for token in range(50)])
    assert _summarize(first)[0]["group"] == 0
    assert sp.feed_auto([(100, "s"), (101, "s")]) == []
    assert sp.feed_auto([(200 + token, "b") for token in range(94)]) == []

    queued_stream = _summarize(sp.on_segment_done(0))
    assert [(s["seg"], s["group"], s["text"]) for s in queued_stream] == [
        (1, 1, "ss")
    ]

    # Tighten after the queued stream has already received its public id. In
    # the old shared namespace, replanning B created a tail with that same id.
    sp.observe_segment(
        5,
        2,
        outcome=RatioOutcome.KV_OVERFLOW,
        segment_idx=1,
    )

    offline_first = _summarize(sp.on_segment_done(1))
    assert offline_first[0]["group"] == 2
    assert offline_first[0]["text"] == "b" * 40

    offline_second = _summarize(sp.on_segment_done(2))
    offline_tail = _summarize(sp.on_segment_done(3))
    assert [(s["group"], len(s["text"])) for s in offline_second + offline_tail] == [
        (3, 40),
        (4, 14),
    ]

    all_segments = (
        _summarize(first) + queued_stream + offline_first + offline_second + offline_tail
    )
    assert [segment["group"] for segment in all_segments] == [0, 1, 2, 3, 4]
    assert "".join(segment["text"] for segment in all_segments) == (
        "a" * 50 + "ss" + "b" * 94
    )

    reorder = AudioReorder()
    delivered: list[bytes] = []
    for segment in reversed(all_segments):
        delivered.extend(
            reorder.push(
                segment["group"],
                segment["local"],
                segment["text"].encode(),
            )
        )
        delivered.extend(
            reorder.mark_done(
                segment["group"],
                segment["local"],
                group_final=segment["final"],
            )
        )
    assert b"".join(delivered).decode() == "a" * 50 + "ss" + "b" * 94
    assert reorder.pending_state() == {
        "next_emit": [5, 0],
        "buffered_keys": 0,
        "buffered_chunks": 0,
    }


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
    out.append(strip_emoji(carry))  # end-of-input flush
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
    assert _stage0_stream(["count3"]) == "count3"  # flushed at end of input
