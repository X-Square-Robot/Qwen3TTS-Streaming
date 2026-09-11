"""Engine-owned text alias for spoken deployment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_VERSION_QUERY_TEXT = "自变量语音合成版本号"
# Keep the historical name as a public compatibility alias.  Runtime routing
# uses the configured value passed to the router/resolver.
VERSION_QUERY_TEXT = DEFAULT_VERSION_QUERY_TEXT
# Release identities are deployment metadata, not source-code defaults.  The
# neutral value is used only for source-tree/test runs without package
# metadata; production values come from ENGINE_VERSION or package sidecars.
UNKNOWN_VERSION = "unknown"
DEFAULT_ENGINE_VERSION = UNKNOWN_VERSION
DEFAULT_ENGINE_BUILD_VERSION = UNKNOWN_VERSION
DEFAULT_MODEL_VERSION = UNKNOWN_VERSION


def format_engine_model_version(
    engine_version: str,
    model_version: str,
    engine_build_version: str = DEFAULT_ENGINE_BUILD_VERSION,
) -> str:
    """Build the spoken form from tag, model release, and TRT build identity."""

    engine = str(engine_version or "").strip()
    model = str(model_version or "").strip()
    build = str(engine_build_version or "").strip()
    if not engine:
        raise ValueError("engine version must not be empty")
    if not model:
        raise ValueError("model version must not be empty")
    if not build:
        raise ValueError("engine build version must not be empty")
    return (
        f"引擎版本号：{engine}，"
        f"模型版本号：{model}，"
        f"引擎编译版本号：{build}"
    )


DEFAULT_ENGINE_MODEL_VERSION = format_engine_model_version(
    DEFAULT_ENGINE_VERSION,
    DEFAULT_MODEL_VERSION,
    DEFAULT_ENGINE_BUILD_VERSION,
)


@dataclass(frozen=True, slots=True)
class DiagnosticTextResolution:
    chunks: tuple[str, ...]
    query_matched: bool = False


def normalize_version_query_text(value: str) -> str:
    """Normalize and validate the configured spoken version trigger."""

    query_text = str(value or "").strip()
    if not query_text:
        raise ValueError("version_query_text must not be empty")
    return query_text


def resolve_diagnostic_text(
    text: str,
    version_text: str,
    version_query_text: str = VERSION_QUERY_TEXT,
) -> str:
    """Resolve an exact, complete synthesis query to its spoken payload.

    The query is deliberately exact-match only.  Applying it after full-text
    normalization prevents an ordinary sentence containing the marker from
    changing meaning and avoids rewriting text already emitted by streaming
    input modes.
    """

    if text != normalize_version_query_text(version_query_text):
        return text
    return version_text


@dataclass(slots=True)
class DiagnosticTextRouter:
    """Hold only a possible query prefix until the input is unambiguous.

    This keeps the exact query working for AUTO/TOKEN streams without buffering
    ordinary streaming input.  Once a chunk diverges, the original packet
    sequence is released unchanged.
    """

    version_query_text: str = VERSION_QUERY_TEXT
    _pending: list[str] = field(default_factory=list)
    _passthrough: bool = False

    def __post_init__(self) -> None:
        self.version_query_text = normalize_version_query_text(
            self.version_query_text
        )

    @property
    def pending_text(self) -> str:
        """Raw query-prefix text held back by the router.

        The prefix is intentionally not sent to the tokenizer until it is
        known whether the packet sequence is the exact diagnostics query.
        Exposing it read-only lets the session journal retain a complete raw
        coordinate source while preserving that release fence.
        """

        return "".join(self._pending)

    def push(self, text: str) -> tuple[str, ...]:
        if not text:
            return ()
        if self._passthrough:
            return (text,)

        candidate = "".join((*self._pending, text))
        if self.version_query_text.startswith(candidate):
            self._pending.append(text)
            return ()

        self._passthrough = True
        released = (*self._pending, text)
        self._pending.clear()
        return released

    def finish(self, version_text: str) -> DiagnosticTextResolution:
        if self._passthrough:
            return DiagnosticTextResolution(())

        pending = tuple(self._pending)
        complete_text = "".join(pending)
        self._pending.clear()
        self._passthrough = True
        if complete_text == self.version_query_text:
            return DiagnosticTextResolution((version_text,), query_matched=True)
        return DiagnosticTextResolution(pending)


__all__ = [
    "DEFAULT_ENGINE_MODEL_VERSION",
    "DEFAULT_ENGINE_BUILD_VERSION",
    "DEFAULT_ENGINE_VERSION",
    "DEFAULT_MODEL_VERSION",
    "DEFAULT_VERSION_QUERY_TEXT",
    "DiagnosticTextResolution",
    "DiagnosticTextRouter",
    "UNKNOWN_VERSION",
    "VERSION_QUERY_TEXT",
    "format_engine_model_version",
    "normalize_version_query_text",
    "resolve_diagnostic_text",
]
