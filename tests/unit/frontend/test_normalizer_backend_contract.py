"""Contract tests for the public WeText backend boundary.

These tests use tiny public-API doubles instead of loading the real WFSTs for
most cases.  The installed-runtime smoke tests live beside the stream adapter;
the important invariant here is that the commitment layer never relies on
private normalizer state or snapshot diffs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from engine.frontend.text_commitment.normalizer_backend import (
    BackendCandidate,
    BackendErrorCode,
    CandidateSource,
    MappedNormalization,
    NormalizationMapping,
    NormalizerBackendError,
    PrefixStatus,
    WetextNormalizerBackend,
)
from engine.frontend.text_commitment.types import LanguageKind, SpanKind


@dataclass
class _Candidate:
    text: str
    cost: float
    token_type: str = "cardinal"


@dataclass
class _Mapping:
    kind: str
    token_type: str
    input_start: int
    input_end: int
    output_start: int
    output_end: int
    input_text: str
    output_text: str


@dataclass
class _Result:
    input_text: str
    output_text: str
    mappings: tuple[_Mapping, ...] = ()


class _Normalizer:
    """A public-only fake matching the current WeText return shapes."""

    def __init__(self, lang: str, operator: str) -> None:
        self.lang = lang
        self.operator = operator
        self.calls: list[tuple[str, int]] = []

    def normalize_candidates(self, text: str, nbest: int = 1):
        self.calls.append(("candidates", nbest))
        if self.lang == "en" and text == "21st":
            values = [
                _Candidate("twenty first", 1.0, "ordinal"),
                _Candidate("twenty one st", 2.0, "serial"),
            ]
        else:
            values = [_Candidate(text.upper(), 1.5)]
        return values[:nbest]

    def normalize(self, text: str, nbest: int = 1):
        self.calls.append(("normalize", nbest))
        return self.normalize_candidates(text, nbest=nbest)[0].text

    def normalize_with_mapping(
        self, text: str, nbest: int = 1, include_identity: bool = False
    ):
        self.calls.append(("mapping", nbest))
        output = self.normalize_candidates(text, nbest=nbest)
        return [
            _Result(
                text,
                item.text,
                (
                    _Mapping(
                        "replace",
                        item.token_type,
                        0,
                        len(text),
                        0,
                        len(item.text),
                        text,
                        item.text,
                    ),
                ),
            )
            for item in output
        ]


class _SnapshotStream:
    """Public stream double whose snapshot changes on every suffix."""

    def __init__(self, _lang: str) -> None:
        self._value = ""

    def feed(self, text: str) -> str:
        self._value += text
        return self._value.upper()

    def flush(self) -> str:
        return self._value.upper()


def _factory(*, lang: str, operator: str):
    return _Normalizer(lang, operator)


def test_backend_requires_explicit_language_and_uses_tn_operator():
    backend = WetextNormalizerBackend(normalizer_factory=_factory)
    assert backend.available_languages == (LanguageKind.ZH, LanguageKind.EN)
    assert (
        backend.candidates("21st", language="en", domain=SpanKind.ORDINAL)[0].language
        is LanguageKind.EN
    )
    with pytest.raises(NormalizerBackendError) as error:
        backend.candidates("21", language="auto")
    assert error.value.code is BackendErrorCode.INVALID_LANGUAGE


def test_candidates_are_typed_ordered_deduplicated_and_nbest_bounded():
    backend = WetextNormalizerBackend(normalizer_factory=_factory, max_nbest=2)
    values = backend.candidates(
        "21st", language=LanguageKind.EN, domain=SpanKind.ORDINAL, nbest=99
    )
    assert values == [
        BackendCandidate(
            spoken_text="twenty first",
            class_name="ordinal",
            score=-1.0,
            source=CandidateSource.WETEXT,
            language=LanguageKind.EN,
            raw_text="21st",
            cost=1.0,
            rank=0,
        ),
        BackendCandidate(
            spoken_text="twenty one st",
            class_name="serial",
            score=-2.0,
            source=CandidateSource.WETEXT,
            language=LanguageKind.EN,
            raw_text="21st",
            cost=2.0,
            rank=1,
        ),
    ]
    assert values[0].text == values[0].spoken_text
    assert values[0].weight == values[0].score


def test_closed_normalization_and_mapping_are_dependency_free_values():
    created: list[_Normalizer] = []

    def factory(*, lang: str, operator: str):
        normalizer = _Normalizer(lang, operator)
        created.append(normalizer)
        return normalizer

    backend = WetextNormalizerBackend(normalizer_factory=factory)
    result = backend.normalize_closed("21st", language="en", domain=SpanKind.ORDINAL)
    assert isinstance(result, MappedNormalization)
    assert result.output_text == "twenty first"
    assert result.text == result.output_text
    assert result.language is LanguageKind.EN
    assert result.mappings[0] == NormalizationMapping(
        kind="replace",
        token_type="ordinal",
        input_start=0,
        input_end=4,
        output_start=0,
        output_end=12,
        input_text="21st",
        output_text="twenty first",
    )
    assert result.source_ranges == ((0, 4),)
    assert result.as_dict()["mappings"][0]["token_type"] == "ordinal"
    committed = result.to_commitment_result()
    assert committed.text == "twenty first"
    assert committed.mapping == ((0, 4),)
    # The closed result carries both text and mapping from one public graph
    # call; normalize() must not run before mapping() for the same span.
    calls = [name for normalizer in created for name, _ in normalizer.calls]
    assert calls.count("normalize") == 0
    assert calls.count("mapping") == 1


def test_prefix_oracle_never_promotes_mutable_snapshot_to_commit():
    backend = WetextNormalizerBackend(
        normalizer_factory=_factory,
        stream_factory=lambda lang: _SnapshotStream(lang),
    )
    oracle = backend.prefix_oracle("en")
    first = oracle.feed("99")
    second = oracle.feed("%")
    assert first.status is PrefixStatus.PENDING
    assert second.status is PrefixStatus.PENDING
    assert first.snapshot == "99"
    assert second.snapshot == "99%"
    assert first.stable_spoken_prefix == ""
    assert second.stable_spoken_prefix == ""
    assert not first.committable and not second.committable
    assert second.pending_raw == "99%"

    final = oracle.flush()
    assert final.status is PrefixStatus.FINAL
    assert final.final is True
    assert final.committable
    assert final.stable_spoken_prefix == "99%"
    assert final.pending_raw == ""
    controller_result = final.to_commitment_result()
    assert controller_result.closed is True
    assert controller_result.extendable is False
    assert controller_result.candidates[0].text == "99%"


def test_prefix_oracle_rejects_post_flush_delta_without_losing_raw_history():
    backend = WetextNormalizerBackend(
        normalizer_factory=_factory,
        stream_factory=lambda lang: _SnapshotStream(lang),
    )
    oracle = backend.prefix_oracle("zh")
    oracle.feed("甲")
    oracle.flush()
    late = oracle.feed("乙")
    assert late.status is PrefixStatus.ERROR
    assert late.error_code is BackendErrorCode.STREAM_CLOSED
    assert late.raw_text == "甲"


def test_mapping_is_optional_for_minimal_normalizer_and_does_not_recurse():
    class Minimal:
        def normalize(self, text: str):
            return "spoken:" + text

    backend = WetextNormalizerBackend(
        normalizer_factory=lambda lang, operator: Minimal()
    )
    mapped = backend.normalize_with_mapping("abc", language="en", nbest=1)
    assert len(mapped) == 1
    assert mapped[0].output_text == "spoken:abc"
    assert mapped[0].mappings == ()
    closed = backend.normalize_closed("abc", language="en")
    assert closed.output_text == "spoken:abc"


@pytest.mark.parametrize("mapping_failure", ["empty", "invalid", "exception"])
def test_closed_normalization_uses_ordinary_api_when_mapping_is_unavailable(mapping_failure):
    calls = []

    class OptionalMapping:
        def normalize(self, text: str):
            calls.append("normalize")
            return "spoken:" + text

        def normalize_with_mapping(self, text: str):
            calls.append("mapping")
            if mapping_failure == "exception":
                raise RuntimeError("alignment unavailable")
            if mapping_failure == "invalid":
                return [_Result(text, "")]
            return []

    backend = WetextNormalizerBackend(
        normalizer_factory=lambda lang, operator: OptionalMapping()
    )
    result = backend.normalize_closed("abc", language="en")
    assert result.output_text == "spoken:abc"
    assert result.mappings == ()
    assert calls == ["mapping", "normalize"]


def test_candidate_generator_exception_is_typed_and_preserves_cause():
    class Broken:
        def normalize_candidates(self, _text: str, nbest: int = 1):
            raise RuntimeError("graph unavailable")

    backend = WetextNormalizerBackend(
        normalizer_factory=lambda lang, operator: Broken()
    )
    with pytest.raises(NormalizerBackendError) as error:
        backend.candidates("1", language="en")
    assert error.value.code is BackendErrorCode.NORMALIZE_FAILED
    assert isinstance(error.value.cause, RuntimeError)


def test_missing_optional_runtime_is_observable_not_constructor_fatal():
    backend = WetextNormalizerBackend(
        normalizer_factory=lambda **_kwargs: (_ for _ in ()).throw(
            ImportError("missing")
        ),
    )
    assert backend.available_languages == ()
    with pytest.raises(NormalizerBackendError) as error:
        backend.normalize_closed("1", language="en")
    assert error.value.code is BackendErrorCode.INITIALIZATION_FAILED
