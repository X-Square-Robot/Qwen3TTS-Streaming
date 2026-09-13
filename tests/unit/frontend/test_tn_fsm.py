from engine.frontend.tn import (
    CandidateRecognizer,
    ClassifierRecognizer,
    RecognizerContext,
    RecognizerRegistry,
    SpanContext,
    SpanEvent,
    SpanEventType,
    SpanFSM,
    SpanState,
    SpanTransition,
    SpanDriver,
    SpanRecognition,
)
from engine.frontend.text_commitment.types import CommitDecision, SpanKind
from engine.frontend.text_commitment.types import LanguageKind
from engine.frontend.text_commitment.committer import IncrementalTextCommitter


def test_default_span_fsm_drives_lifecycle_without_running_tn():
    fsm = SpanFSM()

    assert fsm.state is SpanState.SCAN
    assert fsm.step(SpanEvent(SpanEventType.INPUT, text="99")).state is SpanState.OPEN
    assert fsm.step(SpanEvent(SpanEventType.SPAN_READY)).state is SpanState.READY
    assert fsm.step(SpanEvent(SpanEventType.NORMALIZE_START)).state is SpanState.NORMALIZE
    assert fsm.step(SpanEvent(SpanEventType.NORMALIZE_FINISHED)).state is SpanState.COMMIT
    assert fsm.step(SpanEvent(SpanEventType.FINALIZE)).state is SpanState.DONE
    assert fsm.context.sequence == 5
    assert fsm.context.last_event is SpanEventType.FINALIZE


def test_decision_event_projects_existing_committer_state_and_payload():
    fsm = SpanFSM()
    decision = CommitDecision(
        state=SpanState.OPEN,
        pending_raw="https://example.test",
        committed_raw_end=3,
    )

    result = fsm.step(SpanEvent.from_decision(decision, text="https://example.test"))

    assert result.state is SpanState.OPEN
    assert result.context.pending_raw == "https://example.test"
    assert result.context.committed_raw_end == 3
    assert result.context.finalized is False


def test_custom_table_uses_typed_context_and_guard():
    seen: list[str] = []

    def action(context: SpanContext, event: SpanEvent) -> SpanContext:
        seen.append(event.text)
        return context

    fsm = SpanFSM(
        transitions=(
            SpanTransition(
                SpanState.SCAN,
                SpanEventType.INPUT,
                SpanState.OPEN,
                guard=lambda _ctx, event: event.text == "open",
                action=action,
                name="open_only",
            ),
        )
    )
    assert fsm.step(SpanEvent(SpanEventType.INPUT, text="ignored")).transitioned is False
    assert fsm.step(SpanEvent(SpanEventType.INPUT, text="open")).state is SpanState.OPEN
    assert seen == ["open"]


def test_reset_returns_initial_context():
    fsm = SpanFSM()
    fsm.step(SpanEvent(SpanEventType.INPUT, text="x"))
    fsm.reset()
    assert fsm.state is SpanState.SCAN
    assert fsm.context.sequence == 0
    assert fsm.context.last_event is None


def test_registry_is_ordered_and_does_not_normalize_input():
    class NumberRecognizer:
        name = "number"

        def recognize(self, text, *, context):
            if text.isascii() and text.isdigit():
                return SpanRecognition(SpanKind.NUMBER, 0, len(text), recognizer="")
            return None

    registry = RecognizerRegistry([NumberRecognizer()])
    match = registry.recognize("１２", context=RecognizerContext(raw_text="１２"))
    assert match is None  # recognizers receive source spelling unchanged


def test_span_driver_records_committer_decisions():
    class FakeCommitter:
        def feed(self, text, *, final=False, now=None, metadata=None):
            return CommitDecision(
                state=SpanState.DONE if final else SpanState.OPEN,
                pending_raw="" if final else text,
                committed_raw_end=len(text) if final else 0,
            )

        def poll(self, *, now=None):
            return CommitDecision(state=SpanState.OPEN, pending_raw="x")

    driver = SpanDriver(FakeCommitter())
    driver.feed("x")
    assert driver.fsm.state is SpanState.OPEN
    driver.feed("", final=True)
    assert driver.fsm.state is SpanState.DONE
    assert driver.last_result is not None
    assert driver.last_result.step.event.type.value == "decision"


def test_classifier_recognizer_reuses_existing_classifier_contract():
    recognizer = ClassifierRecognizer(lambda _value: SpanKind.NUMBER)
    match = recognizer.recognize("123", context=RecognizerContext(raw_offset=4))
    assert match is not None
    assert match.kind is SpanKind.NUMBER
    assert (match.raw_start, match.raw_end) == (4, 7)


def test_driver_exposes_the_committer_detector_without_copying_rules():
    driver = SpanDriver(IncrementalTextCommitter())
    match = driver.recognize("123", context=RecognizerContext(raw_offset=2))
    assert match is not None
    assert match.kind is SpanKind.NUMBER
    assert match.recognizer == "candidate"


def test_candidate_recognizer_attaches_optional_wfst_candidates():
    class Provider:
        def candidates(self, text, *, language, domain, nbest):
            return (f"{text}:{language.value}:{domain.value}:{nbest}",)

    recognizer = CandidateRecognizer(
        lambda _value: SpanKind.NUMBER,
        Provider(),
        nbest=3,
    )
    match = recognizer.recognize(
        "123",
        context=RecognizerContext(language=LanguageKind.ZH),
    )
    assert match is not None
    assert match.payload == ("123:zh:number:3",)
