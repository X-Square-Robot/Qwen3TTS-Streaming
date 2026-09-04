from engine.core.text_journal import CanonicalTextJournal
from engine.frontend.interface import _normalize_tts_text


def _normalize(text: str) -> str:
    return " ".join(text.replace("\t", " ").split(" "))


def test_journal_coordinates_are_independent_of_packetization():
    one = CanonicalTextJournal(_normalize)
    one.append("hello\tworld")

    two = CanonicalTextJournal(_normalize)
    two.append("hello")
    two.append("\tworld")

    assert one.normalized_text == two.normalized_text == "hello world"
    assert one.normalized_to_raw == two.normalized_to_raw
    assert two.raw_span(6, 11) == (6, 11)


def test_journal_trim_preserves_raw_boundaries():
    journal = CanonicalTextJournal(_normalize)
    journal.append("  hello  ")
    assert journal.trim_normalized() == "hello"
    assert journal.raw_span(0, 5) == (2, 7)


def test_deleted_trailing_codepoints_wait_for_terminal_boundary():
    def normalize(value: str) -> str:
        return value.replace("😊", "")

    journal = CanonicalTextJournal(normalize)
    journal.append("hello😊")
    assert journal.raw_span(0, 5) == (0, 5)
    journal.append(" world")
    assert journal.raw_span(0, 5) == (0, 5)
    journal.finish()
    assert journal.raw_span(0, len(journal.normalized_text)) == (0, 12)


def test_keycap_split_is_packetization_invariant():
    one = CanonicalTextJournal(_normalize_tts_text)
    one.append("第1️⃣1步")

    split = CanonicalTextJournal(_normalize_tts_text)
    split.append("第1")
    split.append("️⃣1步")

    assert split.normalized_text == one.normalized_text == "第1步"
    assert split.normalized_to_raw == one.normalized_to_raw
    assert split.raw_span(1, 2) == (1, 5)


def test_deleted_interior_characters_fold_to_next_normalized_boundary():
    journal = CanonicalTextJournal(_normalize_tts_text)
    journal.append("good😊morning")

    # The inserted canonical separator owns the removed emoji provenance.
    assert journal.raw_span(4, 5) == (4, 5)
    assert journal.raw_to_normalized[4] == 4
    assert journal.raw_to_normalized[5] == 5


def test_finish_releases_a_held_keycap_base():
    journal = CanonicalTextJournal(_normalize_tts_text)
    journal.append("第1")
    assert journal.normalized_text == "第"

    journal.finish()
    assert journal.normalized_text == "第1"
    assert journal.raw_span(1, 2) == (1, 2)


def test_session_global_trim_drops_incremental_unicode_whitespace_prefix():
    prefix = " \t\n\r\u00a0\u2003\u3000"
    journal = CanonicalTextJournal(
        _normalize_tts_text,
        strip_leading_whitespace=True,
    )

    assert journal.append(prefix) == ("", 0)
    assert journal.normalized_text == ""
    assert journal.normalized_to_raw == [len(prefix)]

    assert journal.append("你好") == ("你好", 0)
    assert journal.raw_text == prefix + "你好"
    assert journal.normalized_text == "你好"
    assert journal.raw_span(0, 2) == (len(prefix), len(prefix) + 2)
    assert journal.raw_to_normalized[: len(prefix) + 1] == [0] * (len(prefix) + 1)


def test_session_global_trim_preserves_cross_packet_word_space():
    journal = CanonicalTextJournal(
        _normalize_tts_text,
        strip_leading_whitespace=True,
    )

    assert journal.append(" hello") == ("hello", 0)
    assert journal.append(" ") == (" ", 5)
    assert journal.append("world") == ("world", 6)

    assert journal.normalized_text == "hello world"
    assert journal.raw_span(5, 6) == (6, 7)
    assert journal.raw_span(6, 11) == (7, 12)


def test_session_global_trim_removes_whitespace_exposed_by_leading_emoji():
    journal = CanonicalTextJournal(
        _normalize_tts_text,
        strip_leading_whitespace=True,
    )

    assert journal.append("\u2003😊 ") == ("", 0)
    assert journal.append("hello") == ("hello", 0)

    assert journal.normalized_text == "hello"
    assert journal.raw_span(0, 5) == (3, 8)


def test_session_global_trim_is_packetization_invariant_with_emoji_carry():
    raw = "\t\u00a0😊 hello1️⃣ world"

    whole = CanonicalTextJournal(
        _normalize_tts_text,
        strip_leading_whitespace=True,
    )
    whole.append(raw)

    split = CanonicalTextJournal(
        _normalize_tts_text,
        strip_leading_whitespace=True,
    )
    for packet in ("\t\u00a0😊 ", "hello1", "️⃣", " world"):
        split.append(packet)

    assert split.raw_text == whole.raw_text == raw
    assert split.normalized_text == whole.normalized_text == "hello world"
    assert split.normalized_to_raw == whole.normalized_to_raw
    assert split.raw_to_normalized == whole.raw_to_normalized

    whole.finish()
    split.finish()

    assert split.normalized_text == whole.normalized_text == "hello world"
    assert split.normalized_to_raw == whole.normalized_to_raw
    assert split.raw_to_normalized == whole.raw_to_normalized


def test_session_global_trim_finish_releases_held_digit_after_leading_space():
    journal = CanonicalTextJournal(
        _normalize_tts_text,
        strip_leading_whitespace=True,
    )

    assert journal.append(" 1") == ("", 0)
    assert journal.normalized_to_raw == [1]

    journal.finish()

    assert journal.normalized_text == "1"
    assert journal.raw_span(0, 1) == (1, 2)


def test_append_spoken_preserves_raw_source_for_tn_expansion():
    journal = CanonicalTextJournal(
        _normalize_tts_text,
        strip_leading_whitespace=True,
    )

    assert journal.append_spoken("是99%", 0, 1, "是") == ("是", 0)
    assert journal.append_spoken("是99%", 1, 4, "百分之九十九") == (
        "百分之九十九",
        1,
    )
    assert journal.raw_text == "是99%"
    assert journal.normalized_text == "是百分之九十九"
    assert journal.raw_span(1, len(journal.normalized_text)) == (1, 4)
    journal.finalize_projection()
    assert journal.input_final is True
