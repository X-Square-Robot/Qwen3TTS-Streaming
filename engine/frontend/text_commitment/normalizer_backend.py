"""Public text-normalization backend contracts.

The incremental commitment controller and a text normalizer have different
responsibilities.  A normalizer can tell us which spoken forms are allowed by
its grammar; it cannot tell us when a transport prefix is safe to release to a
TTS tokenizer.  This module makes that boundary explicit.

Only public :mod:`wetext` methods are used here.  In particular, a
``StreamNormalizer.feed`` value is treated as a mutable *snapshot*, never as
an append-only delta.  The prefix oracle therefore reports the complete raw
tail as pending until an explicit final flush.  It is conservative by design
and can be replaced by a future public, versioned prefix API without changing
the commit controller.

The dependency is optional at import time.  A deployment without wetext gets
typed ``NormalizerBackendError`` values (or an empty safe candidate result),
which lets the caller apply its configured literal/cardinal fallback and emit
an observable event instead of silently changing semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.metadata
import inspect
import logging
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    runtime_checkable,
)

from .types import (
    LanguageKind,
    NormalizationCandidate as CommitmentCandidate,
    NormalizationResult as CommitmentNormalization,
    PrefixResult as CommitmentPrefixResult,
    SpanKind,
)
from .wetext_stream import WetextStream, WetextStreamError


logger = logging.getLogger(__name__)


class BackendErrorCode(str, Enum):
    """Stable machine-readable backend failure categories."""

    INVALID_LANGUAGE = "invalid_language"
    INVALID_INPUT = "invalid_input"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    INITIALIZATION_FAILED = "initialization_failed"
    NORMALIZE_FAILED = "normalize_failed"
    INVALID_OUTPUT = "invalid_output"
    STREAM_CLOSED = "stream_closed"
    STREAM_FAILED = "stream_failed"


class PrefixStatus(str, Enum):
    """State of a prefix observation returned by :class:`WetextPrefixOracle`."""

    PENDING = "pending"
    FINAL = "final"
    ERROR = "error"


class CandidateSource(str, Enum):
    """Origin of a verbalization candidate."""

    WETEXT = "wetext"
    CUSTOM_RULE = "custom_rule"
    UPSTREAM_HINT = "upstream_hint"
    FALLBACK = "fallback"


class NormalizerBackendError(RuntimeError):
    """A typed, observable failure at the normalizer boundary."""

    def __init__(
        self,
        code: BackendErrorCode,
        message: str,
        *,
        language: LanguageKind = LanguageKind.UNKNOWN,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.language = language
        self.cause = cause


def _language(value: str | LanguageKind) -> LanguageKind:
    """Resolve only an explicit Chinese or English route.

    ``Normalizer(lang='auto')`` is intentionally not accepted here.  Auto
    detection is a whole-input batch heuristic and cannot provide a causal
    guarantee for a numeric-only open span.
    """

    if isinstance(value, LanguageKind):
        resolved = value
    else:
        raw = str(value).strip().lower().replace("_", "-")
        aliases = {
            "zh": LanguageKind.ZH,
            "zh-cn": LanguageKind.ZH,
            "zh-hans": LanguageKind.ZH,
            "chinese": LanguageKind.ZH,
            "en": LanguageKind.EN,
            "en-us": LanguageKind.EN,
            "en-gb": LanguageKind.EN,
            "english": LanguageKind.EN,
        }
        resolved = aliases.get(raw, LanguageKind.UNKNOWN)
    if resolved not in (LanguageKind.ZH, LanguageKind.EN):
        raise NormalizerBackendError(
            BackendErrorCode.INVALID_LANGUAGE,
            "normalizer backend requires an explicit 'zh' or 'en' language",
            language=LanguageKind.UNKNOWN,
        )
    return resolved


def _span_name(kind: SpanKind | str | None) -> str:
    if isinstance(kind, SpanKind):
        return kind.value
    if kind is None:
        return SpanKind.PLAIN.value
    return str(kind)


@dataclass(frozen=True, slots=True)
class NormalizationMapping:
    """Dependency-free source alignment for one replacement operation."""

    kind: str
    token_type: str
    input_start: int
    input_end: int
    output_start: int
    output_end: int
    input_text: str
    output_text: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation for logs and diagnostics."""

        return {
            "kind": self.kind,
            "token_type": self.token_type,
            "input_start": self.input_start,
            "input_end": self.input_end,
            "output_start": self.output_start,
            "output_end": self.output_end,
            "input_text": self.input_text,
            "output_text": self.output_text,
        }


@dataclass(frozen=True, slots=True)
class BackendCandidate:
    """One grammar-constrained spoken candidate.

    ``score`` is ordered high-to-low by this contract.  WeText exposes a cost
    where lower is better, so the adapter stores ``-cost`` in ``score`` and
    keeps the original value in ``cost`` for diagnostics.
    """

    spoken_text: str
    class_name: str
    score: float
    source: CandidateSource = CandidateSource.WETEXT
    language: LanguageKind = LanguageKind.UNKNOWN
    raw_text: str = ""
    cost: float | None = None
    rank: int = 0
    mapping: tuple[NormalizationMapping, ...] = ()

    # Compatibility aliases make this object convenient next to the older
    # ``types.NormalizationCandidate`` (which uses ``text``/``weight``).
    @property
    def text(self) -> str:
        return self.spoken_text

    @property
    def weight(self) -> float:
        return self.score

    def to_commitment_candidate(self) -> CommitmentCandidate:
        """Project rich backend metadata into the controller's stable type."""

        weight = self.cost if self.cost is not None else -self.score
        return CommitmentCandidate(
            text=self.spoken_text,
            language=self.language,
            mapping=tuple(
                (mapping.input_start, mapping.input_end) for mapping in self.mapping
            ),
            weight=weight,
        )


@dataclass(frozen=True, slots=True)
class MappedNormalization:
    """A closed-span verbalization plus public source mappings."""

    input_text: str
    output_text: str
    mappings: tuple[NormalizationMapping, ...] = ()
    language: LanguageKind = LanguageKind.UNKNOWN
    backend: str = "wetext"
    rank: int = 0
    cost: float | None = None

    @property
    def text(self) -> str:
        """Compatibility alias used by existing closed-span callers."""

        return self.output_text

    @property
    def source_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return raw/output-independent input ranges for old span maps."""

        return tuple(
            (mapping.input_start, mapping.input_end) for mapping in self.mappings
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "input_text": self.input_text,
            "output_text": self.output_text,
            "mappings": [mapping.as_dict() for mapping in self.mappings],
            "language": self.language.value,
            "backend": self.backend,
            "rank": self.rank,
            "cost": self.cost,
        }

    def to_commitment_result(self) -> CommitmentNormalization:
        """Project to the controller-facing closed-span result."""

        return CommitmentNormalization(
            text=self.output_text,
            language=self.language,
            mapping=self.source_ranges,
            backend=self.backend,
        )


@dataclass(frozen=True, slots=True)
class PrefixOracleResult:
    """A causal observation of a normalizer stream.

    ``snapshot`` is for shadow comparison only.  ``stable_spoken_prefix`` is
    empty for every non-final observation in the current public WeText
    adapter, so a caller cannot accidentally feed a mutable snapshot to TTS.
    """

    raw_text: str
    snapshot: str = ""
    stable_spoken_prefix: str = ""
    pending_raw: str = ""
    status: PrefixStatus = PrefixStatus.PENDING
    language: LanguageKind = LanguageKind.UNKNOWN
    final: bool = False
    reason: str = ""
    error_code: BackendErrorCode | None = None

    @property
    def committable(self) -> bool:
        """Whether ``stable_spoken_prefix`` is safe for append-only commit."""

        return self.status is PrefixStatus.FINAL and bool(self.stable_spoken_prefix)

    def to_commitment_result(self) -> CommitmentPrefixResult:
        """Project the public oracle state into the causal controller type."""

        # Import lazily to keep the backend contract independent of the
        # frontier implementation during package initialization.
        from .causal import spoken_units

        candidate = (
            (
                CommitmentCandidate(
                    text=self.stable_spoken_prefix,
                    language=self.language,
                ),
            )
            if self.committable
            else ()
        )
        return CommitmentPrefixResult(
            stable_units=spoken_units(self.stable_spoken_prefix),
            candidates=candidate,
            extendable=self.status is PrefixStatus.PENDING,
            closed=self.status is PrefixStatus.FINAL,
            pending_raw=self.pending_raw,
            error_code=self.error_code.value if self.error_code is not None else None,
        )


@runtime_checkable
class NormalizerBackend(Protocol):
    """Narrow backend interface consumed by a commitment controller."""

    def candidates(
        self,
        text: str,
        *,
        language: str | LanguageKind,
        domain: SpanKind | str | None = None,
        nbest: int = 1,
    ) -> list[BackendCandidate]:
        """Return grammar candidates ordered from best to worst."""

    def normalize_closed(
        self,
        text: str,
        *,
        language: str | LanguageKind,
        domain: SpanKind | str | None = None,
    ) -> MappedNormalization:
        """Normalize a complete, already-closed span."""

    def normalize_with_mapping(
        self,
        text: str,
        *,
        language: str | LanguageKind,
        domain: SpanKind | str | None = None,
        nbest: int = 1,
    ) -> list[MappedNormalization]:
        """Return closed-span candidates with source alignment."""


StreamFactory: TypeAlias = Callable[[str], Any]
NormalizerFactory: TypeAlias = Callable[..., Any]


def _invoke_with_nbest(method: Callable[..., Any], text: str, nbest: int) -> Any:
    """Call version-skewed WeText methods without swallowing real errors.

    Released WeText versions expose ``nbest`` as a keyword, while tiny test
    doubles and older builds sometimes accept only ``text``.  We inspect the
    public signature where possible and use one narrowly-scoped compatibility
    retry when a signature cannot be inspected.
    """

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        parameters = signature.parameters
        if "nbest" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return method(text, nbest=nbest)
        positional = [
            parameter
            for parameter in parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional) >= 2:
            return method(text, nbest)
        return method(text)
    try:
        return method(text, nbest=nbest)
    except TypeError as first_error:
        try:
            return method(text)
        except TypeError:
            raise first_error


def _items(value: Any) -> list[Any]:
    """Normalize a WeText result (string, object, or sequence) to a list."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (bytes, bytearray)):
        return [value.decode("utf-8", errors="replace")]
    if isinstance(value, Sequence):
        return list(value)
    if isinstance(value, Iterable):
        return list(value)
    # Do not iterate arbitrary objects that merely happen to define an
    # implementation-specific iterator; public result containers are covered
    # by the branches above.
    return [value]


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # NaN/inf are not useful for deterministic ranking or JSON logs.
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _mapping_from(value: Any) -> NormalizationMapping | None:
    """Convert a public mapping object/dict without importing wetext types."""

    if isinstance(value, NormalizationMapping):
        return value
    fields = {
        name: _field(value, name)
        for name in (
            "kind",
            "token_type",
            "input_start",
            "input_end",
            "output_start",
            "output_end",
            "input_text",
            "output_text",
        )
    }
    if fields["kind"] is None:
        return None
    try:
        starts_ends = (
            int(fields["input_start"]),
            int(fields["input_end"]),
            int(fields["output_start"]),
            int(fields["output_end"]),
        )
    except (TypeError, ValueError):
        return None
    if (
        starts_ends[0] < 0
        or starts_ends[1] < starts_ends[0]
        or starts_ends[2] < 0
        or starts_ends[3] < starts_ends[2]
    ):
        return None
    return NormalizationMapping(
        kind=str(fields["kind"]),
        token_type=str(fields["token_type"] or ""),
        input_start=starts_ends[0],
        input_end=starts_ends[1],
        output_start=starts_ends[2],
        output_end=starts_ends[3],
        input_text=str(fields["input_text"] or ""),
        output_text=str(fields["output_text"] or ""),
    )


def _mappings_from(value: Any) -> tuple[NormalizationMapping, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        value = [value]
    elif not isinstance(value, (list, tuple)):
        try:
            value = tuple(value)
        except TypeError:
            value = [value]
    converted: list[NormalizationMapping] = []
    for item in value:
        mapping = _mapping_from(item)
        if mapping is not None:
            converted.append(mapping)
    return tuple(converted)


def _candidate_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    candidate = _field(value, "text")
    if candidate is None:
        candidate = _field(value, "spoken_text")
    if candidate is None:
        candidate = _field(value, "output_text")
    return (
        candidate
        if isinstance(candidate, str)
        else ""
        if candidate is None
        else str(candidate)
    )


class WetextPrefixOracle:
    """Conservative wrapper around one public ``StreamNormalizer`` stream.

    The object is intentionally tiny and session-scoped.  It owns the raw
    prefix independently of wetext so diagnostics retain the source even when
    the third-party stream fails.  No private attributes are inspected.
    """

    def __init__(
        self,
        language: str | LanguageKind,
        *,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        self.language = _language(language)
        self._stream = WetextStream(
            self.language.value,
            stream_factory=stream_factory,
        )
        self._raw_text = ""
        self._final = False

    @property
    def raw_text(self) -> str:
        return self._raw_text

    @property
    def closed(self) -> bool:
        return self._final

    def feed(self, delta: str, *, final: bool = False) -> PrefixOracleResult:
        """Observe a delta, never exposing a provisional commit delta."""

        if not isinstance(delta, str):
            return PrefixOracleResult(
                raw_text=self._raw_text,
                pending_raw=self._raw_text,
                status=PrefixStatus.ERROR,
                language=self.language,
                reason="delta must be a string",
                error_code=BackendErrorCode.INVALID_INPUT,
            )
        if self._final:
            return PrefixOracleResult(
                raw_text=self._raw_text,
                pending_raw=self._raw_text,
                status=PrefixStatus.ERROR,
                language=self.language,
                reason="prefix oracle has already been flushed",
                error_code=BackendErrorCode.STREAM_CLOSED,
            )

        self._raw_text += delta
        try:
            snapshot = self._stream.feed(delta)
            if final:
                final_snapshot = self._stream.flush()
                self._final = True
                return PrefixOracleResult(
                    raw_text=self._raw_text,
                    snapshot=final_snapshot.snapshot,
                    stable_spoken_prefix=final_snapshot.snapshot,
                    pending_raw="",
                    status=PrefixStatus.FINAL,
                    language=self.language,
                    final=True,
                    reason="public_stream_flush",
                )
            return PrefixOracleResult(
                raw_text=self._raw_text,
                snapshot=snapshot.snapshot,
                # A snapshot can be rewritten by a later suffix.  Keeping this
                # empty is the safety property this class exists to enforce.
                stable_spoken_prefix="",
                pending_raw=self._raw_text,
                status=PrefixStatus.PENDING,
                language=self.language,
                reason="public_stream_snapshot_not_commit_safe",
            )
        except WetextStreamError as exc:
            return PrefixOracleResult(
                raw_text=self._raw_text,
                pending_raw=self._raw_text,
                status=PrefixStatus.ERROR,
                language=self.language,
                reason=str(exc),
                error_code=BackendErrorCode.STREAM_FAILED,
            )
        except Exception as exc:  # pragma: no cover - defensive third-party boundary
            logger.warning(
                "text.normalizer.prefix_failed",
                extra={"language": self.language.value},
                exc_info=True,
            )
            return PrefixOracleResult(
                raw_text=self._raw_text,
                pending_raw=self._raw_text,
                status=PrefixStatus.ERROR,
                language=self.language,
                reason=str(exc),
                error_code=BackendErrorCode.STREAM_FAILED,
            )

    def flush(self) -> PrefixOracleResult:
        """Finalize without adding text, returning the only safe snapshot."""

        return self.feed("", final=True)


class WetextNormalizerBackend:
    """Adapter implementing :class:`NormalizerBackend` with public WeText APIs."""

    backend_name = "wetext"

    def __init__(
        self,
        *,
        normalizer_factory: NormalizerFactory | None = None,
        stream_factory: StreamFactory | None = None,
        eager: bool = True,
        max_nbest: int = 16,
    ) -> None:
        if (
            isinstance(max_nbest, bool)
            or not isinstance(max_nbest, int)
            or max_nbest < 1
        ):
            raise ValueError("max_nbest must be a positive integer")
        self.max_nbest = max_nbest
        self._normalizer_factory = normalizer_factory
        self._stream_factory = stream_factory
        self._normalizers: dict[LanguageKind, Any] = {}
        self._init_errors: dict[LanguageKind, NormalizerBackendError] = {}
        if eager:
            for language in (LanguageKind.ZH, LanguageKind.EN):
                # Optional wetext is allowed to be absent in development and
                # in a fallback-only deployment.  Retain a typed error per
                # language instead of making backend construction fatal.
                try:
                    self._ensure_normalizer(language)
                except NormalizerBackendError:
                    continue

    @property
    def available_languages(self) -> tuple[LanguageKind, ...]:
        return tuple(
            language
            for language in (LanguageKind.ZH, LanguageKind.EN)
            if language in self._normalizers
        )

    @property
    def runtime_version(self) -> str | None:
        try:
            return importlib.metadata.version("wetext")
        except importlib.metadata.PackageNotFoundError:
            return None
        except Exception:
            logger.debug("unable to inspect wetext version", exc_info=True)
            return None

    def _factory(self) -> NormalizerFactory:
        if self._normalizer_factory is not None:
            return self._normalizer_factory
        try:
            from wetext import Normalizer  # type: ignore
        except Exception as exc:
            raise NormalizerBackendError(
                BackendErrorCode.RUNTIME_UNAVAILABLE,
                "wetext.Normalizer is not installed",
                cause=exc,
            ) from exc
        return Normalizer

    def _ensure_normalizer(self, language: LanguageKind) -> Any:
        if language in self._normalizers:
            return self._normalizers[language]
        if language in self._init_errors:
            raise self._init_errors[language]
        try:
            factory = self._factory()
            # Official WeText accepts keyword ``lang`` and ``operator``.  A
            # small compatibility fallback keeps injected test doubles easy to
            # write while still never touching private runtime state.
            try:
                normalizer = factory(lang=language.value, operator="tn")
            except TypeError as first_error:
                try:
                    normalizer = factory(language.value, "tn")
                except TypeError:
                    try:
                        normalizer = factory(language.value)
                    except TypeError:
                        raise first_error
            self._normalizers[language] = normalizer
            return normalizer
        except NormalizerBackendError:
            raise
        except Exception as exc:
            error = NormalizerBackendError(
                BackendErrorCode.INITIALIZATION_FAILED,
                f"failed to initialise wetext for lang={language.value!r}",
                language=language,
                cause=exc,
            )
            self._init_errors[language] = error
            logger.warning(
                "text.normalizer.init_failed",
                extra={"language": language.value},
                exc_info=True,
            )
            raise error from exc

    def _method(self, language: LanguageKind, name: str) -> Callable[..., Any] | None:
        normalizer = self._ensure_normalizer(language)
        method = getattr(normalizer, name, None)
        return method if callable(method) else None

    def candidates(
        self,
        text: str,
        *,
        language: str | LanguageKind,
        domain: SpanKind | str | None = None,
        nbest: int = 1,
    ) -> list[BackendCandidate]:
        resolved = _language(language)
        if not isinstance(text, str):
            raise NormalizerBackendError(
                BackendErrorCode.INVALID_INPUT,
                "normalizer input must be a string",
                language=resolved,
            )
        if isinstance(nbest, bool) or not isinstance(nbest, int) or nbest < 1:
            raise ValueError("nbest must be a positive integer")
        limit = min(nbest, self.max_nbest)
        if not text:
            return []
        try:
            method = self._method(resolved, "normalize_candidates")
            if method is None:
                method = self._method(resolved, "normalize")
            if method is None:
                raise NormalizerBackendError(
                    BackendErrorCode.NORMALIZE_FAILED,
                    "wetext normalizer exposes neither normalize_candidates nor normalize",
                    language=resolved,
                )
            raw_values = _items(_invoke_with_nbest(method, text, limit))
        except NormalizerBackendError:
            raise
        except Exception as exc:
            raise NormalizerBackendError(
                BackendErrorCode.NORMALIZE_FAILED,
                f"wetext candidate generation failed for lang={resolved.value!r}",
                language=resolved,
                cause=exc,
            ) from exc

        result: list[BackendCandidate] = []
        seen: set[str] = set()
        domain_name = _span_name(domain)
        for rank, item in enumerate(raw_values):
            spoken = _candidate_text(item)
            if not spoken or spoken in seen:
                continue
            seen.add(spoken)
            cost = _finite_float(_field(item, "cost"))
            score = -cost if cost is not None else float(-rank)
            class_name = _field(item, "class_name")
            if class_name is None:
                class_name = _field(item, "token_type")
            if class_name is None:
                class_name = _field(item, "tag")
            result.append(
                BackendCandidate(
                    spoken_text=spoken,
                    class_name=str(class_name or domain_name),
                    score=score,
                    source=CandidateSource.WETEXT,
                    language=resolved,
                    raw_text=text,
                    cost=cost,
                    rank=rank,
                    mapping=_mappings_from(
                        _field(item, "mappings", _field(item, "mapping"))
                    ),
                )
            )
            if len(result) >= limit:
                break
        return result

    def normalize_with_mapping(
        self,
        text: str,
        *,
        language: str | LanguageKind,
        domain: SpanKind | str | None = None,
        nbest: int = 1,
        include_identity: bool = False,
    ) -> list[MappedNormalization]:
        resolved = _language(language)
        if not isinstance(text, str):
            raise NormalizerBackendError(
                BackendErrorCode.INVALID_INPUT,
                "normalizer input must be a string",
                language=resolved,
            )
        if isinstance(nbest, bool) or not isinstance(nbest, int) or nbest < 1:
            raise ValueError("nbest must be a positive integer")
        limit = min(nbest, self.max_nbest)
        if not text:
            return []
        try:
            method = self._method(resolved, "normalize_with_mapping")
            if method is None:
                # Mapping is an optional WeText public API.  Use the ordinary
                # public normalizer directly and synthesize a result with no
                # detailed mappings when it is unavailable.  Calling
                # ``normalize_closed`` here would recurse because that method
                # delegates to this API when mapping support is available.
                ordinary = self._method(resolved, "normalize")
                if ordinary is None:
                    candidates = self.candidates(
                        text,
                        language=resolved,
                        domain=domain,
                        nbest=1,
                    )
                    if not candidates:
                        return []
                    return [
                        MappedNormalization(
                            input_text=text,
                            output_text=candidates[0].spoken_text,
                            language=resolved,
                            backend=self.backend_name,
                            cost=candidates[0].cost,
                        )
                    ]
                value = _invoke_with_nbest(ordinary, text, 1)
                values = _items(value)
                output = _candidate_text(values[0]) if values else ""
                if not output:
                    return []
                return [
                    MappedNormalization(
                        input_text=text,
                        output_text=output,
                        language=resolved,
                        backend=self.backend_name,
                        cost=_finite_float(_field(values[0], "cost"))
                        if values
                        else None,
                    )
                ]
            raw_values = _items(_invoke_with_nbest(method, text, limit))
        except NormalizerBackendError:
            raise
        except Exception as exc:
            raise NormalizerBackendError(
                BackendErrorCode.NORMALIZE_FAILED,
                f"wetext mapping generation failed for lang={resolved.value!r}",
                language=resolved,
                cause=exc,
            ) from exc

        results: list[MappedNormalization] = []
        for rank, item in enumerate(raw_values):
            output = _candidate_text(item)
            if not output and include_identity and rank == 0:
                output = text
            if not output:
                continue
            results.append(
                MappedNormalization(
                    input_text=str(_field(item, "input_text", text) or text),
                    output_text=output,
                    mappings=_mappings_from(
                        _field(item, "mappings", _field(item, "mapping"))
                    ),
                    language=resolved,
                    backend=self.backend_name,
                    rank=rank,
                    cost=_finite_float(_field(item, "cost")),
                )
            )
            if len(results) >= limit:
                break
        return results

    def normalize_closed(
        self,
        text: str,
        *,
        language: str | LanguageKind,
        domain: SpanKind | str | None = None,
    ) -> MappedNormalization:
        """Normalize one complete span and return the best public result."""

        resolved = _language(language)
        if not isinstance(text, str):
            raise NormalizerBackendError(
                BackendErrorCode.INVALID_INPUT,
                "normalizer input must be a string",
                language=resolved,
            )
        if not text:
            return MappedNormalization(
                input_text="",
                output_text="",
                language=resolved,
                backend=self.backend_name,
            )

        # Prefer the public mapping API when available.  It already returns
        # the spoken form and alignment in one pass; running ``normalize``
        # first and requesting mappings afterwards doubles the graph work on
        # every closed span.
        mapping_method = self._method(resolved, "normalize_with_mapping")
        if mapping_method is not None:
            try:
                mapped = self.normalize_with_mapping(
                    text,
                    language=resolved,
                    domain=domain,
                    nbest=1,
                )
                if mapped and mapped[0].output_text:
                    first = mapped[0]
                    return MappedNormalization(
                        input_text=text,
                        output_text=first.output_text,
                        mappings=first.mappings,
                        language=resolved,
                        backend=self.backend_name,
                        rank=first.rank,
                        cost=first.cost,
                    )
            except NormalizerBackendError:
                logger.info(
                    "text.normalizer.mapping_unavailable",
                    extra={"language": resolved.value, "domain": _span_name(domain)},
                )
        try:
            method = self._method(resolved, "normalize")
            if method is None:
                candidates = self.candidates(
                    text, language=resolved, domain=domain, nbest=1
                )
                if not candidates:
                    raise NormalizerBackendError(
                        BackendErrorCode.INVALID_OUTPUT,
                        "wetext returned no candidate for a closed span",
                        language=resolved,
                    )
                output = candidates[0].spoken_text
                cost = candidates[0].cost
            else:
                value = _invoke_with_nbest(method, text, 1)
                values = _items(value)
                output = _candidate_text(values[0]) if values else ""
                cost = _finite_float(_field(values[0], "cost")) if values else None
                if not output and values:
                    output = str(values[0])
            if not output:
                raise NormalizerBackendError(
                    BackendErrorCode.INVALID_OUTPUT,
                    "wetext returned an empty result for a non-empty closed span",
                    language=resolved,
                )
        except NormalizerBackendError:
            raise
        except Exception as exc:
            raise NormalizerBackendError(
                BackendErrorCode.NORMALIZE_FAILED,
                f"wetext normalization failed for lang={resolved.value!r}",
                language=resolved,
                cause=exc,
            ) from exc

        return MappedNormalization(
            input_text=text,
            output_text=output,
            language=resolved,
            backend=self.backend_name,
            rank=0,
            cost=cost,
        )

    def prefix_oracle(self, language: str | LanguageKind) -> WetextPrefixOracle:
        """Create a session-scoped public streaming oracle."""

        resolved = _language(language)
        try:
            return WetextPrefixOracle(
                resolved,
                stream_factory=self._stream_factory,
            )
        except WetextStreamError as exc:
            raise NormalizerBackendError(
                BackendErrorCode.INITIALIZATION_FAILED,
                str(exc),
                language=resolved,
                cause=exc,
            ) from exc

    # ``stream`` is a short alias matching the existing WetextAdapter API.
    def stream(self, *, language: str | LanguageKind) -> WetextPrefixOracle:
        return self.prefix_oracle(language)

    # Explicit aliases keep call sites readable while retaining the
    # ``normalize_closed`` name used by the backend protocol.
    def normalize(
        self,
        text: str,
        *,
        language: str | LanguageKind,
        domain: SpanKind | str | None = None,
    ) -> MappedNormalization:
        return self.normalize_closed(text, language=language, domain=domain)

    def observe_prefix(self, language: str | LanguageKind) -> WetextPrefixOracle:
        return self.prefix_oracle(language)


# Names used by the design document and by downstream integrations.  The
# aliases live in this module so importing the contract does not force callers
# to know the internal ``Backend*`` naming convention.
NormalizationCandidate = BackendCandidate
NormalizationResult = MappedNormalization


__all__ = [
    "BackendCandidate",
    "BackendErrorCode",
    "CandidateSource",
    "MappedNormalization",
    "NormalizationCandidate",
    "NormalizationMapping",
    "NormalizationResult",
    "NormalizerBackend",
    "NormalizerBackendError",
    "PrefixOracleResult",
    "PrefixStatus",
    "WetextNormalizerBackend",
    "WetextPrefixOracle",
]
