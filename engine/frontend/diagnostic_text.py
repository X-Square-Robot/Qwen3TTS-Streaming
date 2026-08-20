"""Engine-owned text alias for spoken deployment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field


VERSION_QUERY_TEXT = "自变量语音合成版本号"
DEFAULT_ENGINE_VERSION = "rime@20260717_580_5090_v1"
DEFAULT_MODEL_VERSION = "zehan@20260601"


def format_engine_model_version(engine_version: str, model_version: str) -> str:
    """Build the stable spoken form from independently managed versions."""

    engine = str(engine_version or "").strip()
    model = str(model_version or "").strip()
    if not engine:
        raise ValueError("engine version must not be empty")
    if not model:
        raise ValueError("model version must not be empty")
    return f"{engine}-{model}"


DEFAULT_ENGINE_MODEL_VERSION = format_engine_model_version(
    DEFAULT_ENGINE_VERSION,
    DEFAULT_MODEL_VERSION,
)


@dataclass(frozen=True, slots=True)
class DiagnosticTextResolution:
    chunks: tuple[str, ...]
    query_matched: bool = False


def resolve_diagnostic_text(text: str, version_text: str) -> str:
    """Resolve an exact, complete synthesis query to its spoken payload.

    The query is deliberately exact-match only.  Applying it after full-text
    normalization prevents an ordinary sentence containing the marker from
    changing meaning and avoids rewriting text already emitted by streaming
    input modes.
    """

    if text != VERSION_QUERY_TEXT:
        return text
    return version_text


@dataclass(slots=True)
class DiagnosticTextRouter:
    """Hold only a possible query prefix until the input is unambiguous.

    This keeps the exact query working for AUTO/TOKEN streams without buffering
    ordinary streaming input.  Once a chunk diverges, the original packet
    sequence is released unchanged.
    """

    _pending: list[str] = field(default_factory=list)
    _passthrough: bool = False

    def push(self, text: str) -> tuple[str, ...]:
        if not text:
            return ()
        if self._passthrough:
            return (text,)

        candidate = "".join((*self._pending, text))
        if VERSION_QUERY_TEXT.startswith(candidate):
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
        if complete_text == VERSION_QUERY_TEXT:
            return DiagnosticTextResolution((version_text,), query_matched=True)
        return DiagnosticTextResolution(pending)


__all__ = [
    "DEFAULT_ENGINE_MODEL_VERSION",
    "DEFAULT_ENGINE_VERSION",
    "DEFAULT_MODEL_VERSION",
    "DiagnosticTextResolution",
    "DiagnosticTextRouter",
    "VERSION_QUERY_TEXT",
    "format_engine_model_version",
    "resolve_diagnostic_text",
]
