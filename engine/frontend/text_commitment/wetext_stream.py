"""Small, defensive adapter around wetext's public streaming API.

``wetext.StreamNormalizer`` is useful for evaluating arbitrary packetisation,
but its :meth:`feed` result is a *snapshot* of the whole stream.  A snapshot
is allowed to change when a suffix arrives (``99`` can become ``99%``), so it
must never be forwarded to the TTS tokenizer as an append-only delta.

This module deliberately does not implement any TN grammar.  It provides two
safe operations for the frontend:

* ``normalize_closed`` feeds a complete, already-bound span and calls
  ``flush``.  This is the production operation and relies only on wetext's
  public API.
* ``feed``/``flush`` expose snapshots for shadow tests and diagnostics.  They
  do not manufacture deltas from snapshots.

The package is optional at import time so the engine can still start in a
minimal environment; callers receive an explicit error and choose their
configured fallback instead of silently guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import logging
from typing import Any, Callable, Literal


logger = logging.getLogger(__name__)

Language = Literal["zh", "en"]


class WetextStreamError(RuntimeError):
    """A normalized error category for the optional wetext runtime."""

    def __init__(self, code: str, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


@dataclass(frozen=True)
class StreamSnapshot:
    """A non-committable view returned by a stream operation.

    ``snapshot`` is intentionally separate from ``text`` used by production
    commits.  Consumers must call ``normalize_closed`` (or ``flush``) before
    treating a value as final.
    """

    snapshot: str
    pending: str = ""
    final: bool = False
    backend: str = "wetext.stream"


@dataclass(frozen=True)
class ClosedNormalization:
    """Final result for one homogeneous, already-closed source span."""

    text: str
    backend: str = "wetext.stream"
    snapshot: str = ""


def _default_factory(lang: Language) -> Any:
    try:
        from wetext import StreamNormalizer  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on deployment image
        raise WetextStreamError(
            "runtime_unavailable",
            "wetext.StreamNormalizer is not installed",
            cause=exc,
        ) from exc
    try:
        return StreamNormalizer(lang=lang, operator="tn")
    except Exception as exc:
        raise WetextStreamError(
            "stream_init_failed",
            f"failed to initialise wetext stream for lang={lang!r}",
            cause=exc,
        ) from exc


class WetextStream:
    """One-utterance wrapper over ``wetext.StreamNormalizer``.

    The wrapper owns one stream instance and is not thread-safe.  A new
    instance should be created for each language island/utterance.  No private
    wetext attributes are inspected here; this is intentional because the
    public stream contract only promises snapshots and ``flush``.
    """

    def __init__(
        self,
        lang: str,
        *,
        stream_factory: Callable[[Language], Any] | None = None,
    ) -> None:
        normalized_lang = str(lang).strip().lower()
        if normalized_lang not in ("zh", "en"):
            raise ValueError("wetext stream language must be 'zh' or 'en'")
        self.lang: Language = normalized_lang  # type: ignore[assignment]
        self._factory = stream_factory or _default_factory
        try:
            self._stream = self._factory(self.lang)
        except WetextStreamError:
            raise
        except Exception as exc:
            raise WetextStreamError(
                "stream_init_failed",
                f"failed to initialise wetext stream for lang={self.lang!r}",
                cause=exc,
            ) from exc
        self._closed = False
        self._last_snapshot = ""

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def last_snapshot(self) -> str:
        """Return the last non-committable snapshot for diagnostics."""

        return self._last_snapshot

    def feed(self, text: str) -> StreamSnapshot:
        """Feed a delta and return a snapshot; never return a commit delta."""

        if self._closed:
            raise WetextStreamError("stream_closed", "wetext stream has already been flushed")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            snapshot = self._stream.feed(text)
        except Exception as exc:
            raise WetextStreamError(
                "normalizer_error",
                f"wetext stream feed failed for lang={self.lang!r}",
                cause=exc,
            ) from exc
        if not isinstance(snapshot, str):
            raise WetextStreamError(
                "invalid_output",
                "wetext StreamNormalizer.feed returned a non-string snapshot",
            )
        self._last_snapshot = snapshot
        # The value is deliberately named ``snapshot`` in the return object;
        # callers cannot accidentally mistake it for an append-only delta.
        return StreamSnapshot(snapshot=snapshot, final=False)

    def flush(self) -> StreamSnapshot:
        """Finalize the stream and return its only commit-safe value."""

        if self._closed:
            # wetext itself returns the final value on a repeated flush.  Keep
            # this operation idempotent for shutdown/finalization races.
            return StreamSnapshot(snapshot=self._last_snapshot, final=True)
        try:
            snapshot = self._stream.flush()
        except Exception as exc:
            raise WetextStreamError(
                "normalizer_error",
                f"wetext stream flush failed for lang={self.lang!r}",
                cause=exc,
            ) from exc
        if not isinstance(snapshot, str):
            raise WetextStreamError(
                "invalid_output",
                "wetext StreamNormalizer.flush returned a non-string value",
            )
        self._last_snapshot = snapshot
        self._closed = True
        return StreamSnapshot(snapshot=snapshot, final=True)

    def normalize_closed(self, text: str) -> ClosedNormalization:
        """Normalize one complete span using only public stream methods.

        A fresh wrapper is expected for each call.  Reusing a wrapper after a
        flush would make two source spans share state, so fail explicitly.
        """

        if self._closed:
            raise WetextStreamError("stream_closed", "cannot reuse a flushed stream")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            # ``feed`` is intentionally called even for identity text: this
            # exercises the same official stream path used in shadow tests.
            self.feed(text)
            final = self.flush()
        except WetextStreamError:
            raise
        return ClosedNormalization(text=final.snapshot, snapshot=final.snapshot)


def normalize_closed(text: str, *, lang: str) -> ClosedNormalization:
    """Convenience function for a single closed span."""

    return WetextStream(lang).normalize_closed(text)


def wetext_runtime_version() -> str | None:
    """Return the installed wetext version without importing its heavy graphs."""

    try:
        return importlib.metadata.version("wetext")
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        logger.debug("unable to inspect wetext package version", exc_info=True)
        return None


__all__ = [
    "ClosedNormalization",
    "StreamSnapshot",
    "WetextStream",
    "WetextStreamError",
    "normalize_closed",
    "wetext_runtime_version",
]
