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
