import pytest

from engine.core.text_coordinates import (
    CodecTokenProgress, SegmentTextCoordinates, TextProgressProjection,
)
from engine.core.text_journal import CanonicalTextJournal


def coordinates(base=0):
    journal = CanonicalTextJournal(lambda text: text)
    journal.install_projection("xx99%ab", "xx百分之九十九ab", [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7])
    spans = [dict(normalized_start=i, normalized_end=i + 1,
                  raw_start=0, raw_end=0) for i in range(base, 10)]
    return SegmentTextCoordinates(spans, journal)


def test_native_and_ema_token_estimates_share_conservative_raw_projection():
    coords = coordinates(2)
    label_spans = tuple((i, i + 1) for i in range(2, 10))
    ema = TextProgressProjection(1)
    native = TextProgressProjection(1)
    for token_end in range(9):
        native_token = coords.token_end_from_labels(float(token_end), label_spans)
        expected = ema.project(CodecTokenProgress(1, token_end, token_end + 1, token_end), coords)
        actual = native.project(CodecTokenProgress(1, token_end, token_end + 1, native_token), coords)
        assert actual == expected
        assert int(actual["raw_codepoint_end"]) == (2 if token_end < 6 else token_end - 1)


def test_label_to_bpe_conversion_uses_offsets_not_ratio():
    coords = SegmentTextCoordinates([
        dict(normalized_start=10, normalized_end=15, raw_start=10, raw_end=15),
        dict(normalized_start=15, normalized_end=16, raw_start=15, raw_end=16),
    ], None)
    labels = ((10, 11), (11, 12), (13, 14), (14, 15), (15, 16))
    assert coords.token_end_from_labels(3.99, labels) == 0
    assert coords.token_end_from_labels(4, labels) == 1
    assert coords.token_end_from_labels(5, labels) == 2


def test_estimator_switch_and_revision_cannot_retract_public_high_water():
    coords = coordinates(2)
    state = TextProgressProjection(4)
    first = state.project(CodecTokenProgress(4, 0, 1, 6), coords)
    # A different estimator/revised mapping may propose a lower position.
    coords.journal.install_projection("xx99%ab", "xx百分之九十九ab", [0] * 11)
    second = state.project(CodecTokenProgress(4, 1, 2, 1), coords)
    assert second["raw_codepoint_end"] == first["raw_codepoint_end"] == "5"
    assert second["normalized_codepoint_end"] == "8"
    assert second["text_token_end"] == "6"


def test_segments_have_independent_codec_origins_and_global_text_offsets():
    coords = coordinates(8)
    first = TextProgressProjection(3).project(CodecTokenProgress(3, 0, 1, 0), coords)
    assert first["normalized_codepoint_end"] == "8"
    assert first["raw_codepoint_end"] == "5"
    assert first["source_frame_start"] == "0"
    with pytest.raises(ValueError, match="another segment"):
        TextProgressProjection(2).project(CodecTokenProgress(3, 0, 1, 1), coords)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_native_position_rejected(value):
    with pytest.raises(ValueError):
        coordinates().token_end_from_labels(value, ((0, 1),))
