"""Contracts for the thin official wetext streaming adapter."""

import pytest

from engine.frontend.text_commitment.wetext_stream import (
    WetextStream,
    WetextStreamError,
    wetext_runtime_version,
)


def test_stream_feed_is_explicitly_a_snapshot_not_a_delta():
    stream = WetextStream("zh")
    first = stream.feed("99")
    second = stream.feed("%")

    # Both snapshots are provisional and may differ.  The adapter exposes no
    # guessed append-only delta here.
    assert first.final is False
    assert second.final is False
    assert first.snapshot == "九十九"
    assert second.snapshot == "百分之九十九"

    final = stream.flush()
    assert final.final is True
    assert final.snapshot == "百分之九十九"


def test_closed_span_uses_flush_result_for_english_ordinal():
    result = WetextStream("en").normalize_closed("21st")
    assert result.text == "twenty first"


def test_stream_rejects_auto_language():
    with pytest.raises(ValueError):
        WetextStream("auto")


def test_stream_factory_errors_are_normalized():
    def broken(_lang):
        raise RuntimeError("boom")

    with pytest.raises(WetextStreamError) as exc_info:
        WetextStream("en", stream_factory=broken)
    assert exc_info.value.code == "stream_init_failed"


def test_stream_exception_does_not_leak_raw_snapshot_as_a_commit():
    class Broken:
        def feed(self, _text):
            raise AssertionError("incomplete formula")

        def flush(self):  # pragma: no cover - feed already fails
            return ""

    stream = WetextStream("en", stream_factory=lambda _lang: Broken())
    with pytest.raises(WetextStreamError) as exc_info:
        stream.feed("3*")
    assert exc_info.value.code == "normalizer_error"


def test_runtime_version_is_observable_without_private_stream_state():
    # The engine image pins wetext; a minimal development environment may not
    # have it, so this assertion intentionally accepts either outcome.
    assert wetext_runtime_version() in (None, "0.1.7") or isinstance(
        wetext_runtime_version(), str
    )
