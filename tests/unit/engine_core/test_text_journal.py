from engine.core.text_journal import CanonicalTextJournal


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
