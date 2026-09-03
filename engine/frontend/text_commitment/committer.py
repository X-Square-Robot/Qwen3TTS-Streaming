from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

from ...text_normalization import is_emoji_char

from .types import CommitDecision, CommitKind, SpanKind, TextCommit, TextNormalizationConfig
from .wetext_backend import WetextAdapter
from .projector import project_readable


_ASCII_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_@.$:/+\-\\]*")
_NUMERIC = re.compile(r"^[+\-]?\d+(?:[.,]\d+)?(?:%|[A-Za-z]{1,8})?$")
_ORDINAL = re.compile(r"^\d{1,6}(?:st|nd|rd|th)$", re.I)
_MATH_CHARS = set("0123456789.+-*/=^×÷()")


def _is_emoji(ch: str) -> bool:
    return "EMOJI" in unicodedata.name(ch, "") or ord(ch) in range(0x1F000, 0x1FAFF)


@dataclass
class _Pending:
    raw: str = ""
    start: int = 0
    kind: SpanKind = SpanKind.PLAIN
    first_at: float = 0.0
    last_at: float = 0.0


class IncrementalTextCommitter:
    """Append-only semantic commit fence in front of the TTS tokenizer."""

    def __init__(self, config: TextNormalizationConfig | None = None, *, adapter: WetextAdapter | None = None):
        self.config = config or TextNormalizationConfig()
        self.adapter = adapter or WetextAdapter()
        self.raw_cursor = 0
        self.committed_raw_end = 0
        self.committed_spoken_text = ""
        self.commit_fence = 0
        self._pending = _Pending()
        self._raw = ""
        self._outbox: list[TextCommit] = []
        self._late_extension = False
        self._force_literal_next = False
        self._current_lang = "zh"
        self._json_in_string = False
        self._json_escape = False
        self._plain_buffer = ""
        self._plain_start = 0

    @property
    def pending_raw(self) -> str:
        return self._pending.raw

    @property
    def pending_kind(self) -> SpanKind | None:
        return self._pending.kind if self._pending.raw else None

    @property
    def next_deadline(self) -> float | None:
        if not self._pending.raw:
            return None
        return min(
            self._pending.first_at + self.config.semantic_max_wait_ms / 1000.0,
            self._pending.last_at + self.config.semantic_idle_wait_ms / 1000.0,
        )

    def feed(self, text: str, *, final: bool = False, now: float | None = None) -> CommitDecision:
        now = time.monotonic() if now is None else now
        commits = list(self.poll(now=now).commits)
        if not self.config.enabled:
            if text:
                commits.append(self._make_commit(text, self.raw_cursor, SpanKind.LITERAL, CommitKind.LITERAL))
                self.raw_cursor += len(text)
                self.committed_raw_end = self.raw_cursor
            return CommitDecision(tuple(commits), reason="disabled")
        if text and final and self.config.projection == "readable_values":
            full_raw = self._raw + text
            projected = project_readable(full_raw)
            structured = (full_raw.lstrip().startswith(("{", "[")) or "[" in full_raw or "`" in full_raw or "**" in full_raw)
            if projected != full_raw and structured:
                self._raw = ""
                self.raw_cursor = len(text)
                self.committed_raw_end = 0
                self._pending = _Pending()
                self._outbox.append(self._make_commit(text, 0, SpanKind.LITERAL, CommitKind.LITERAL, projected))
                self.committed_spoken_text += projected
                self.commit_fence += 1
                text = ""
        late_event = self._late_extension
        if text:
            if self._late_extension:
                self._late_extension = False
                self._force_literal_next = True
            self._raw += text
            self.raw_cursor += len(text)
            self._append(text, now)
        if final:
            self._flush_pending(reason="final")
        else:
            commits.extend(self._close_stable(now))
        self._flush_plain()
        commits.extend(self._outbox)
        self._outbox.clear()
        return CommitDecision(tuple(commits), self.pending_raw, self.pending_kind, "final" if final else "feed", events=("text.span.late_extension",) if late_event else ())

    def poll(self, *, now: float | None = None) -> CommitDecision:
        now = time.monotonic() if now is None else now
        if not self._pending.raw or self.next_deadline is None or now < self.next_deadline:
            return CommitDecision(pending_raw=self.pending_raw, pending_kind=self.pending_kind)
        self._fallback_pending(reason="timeout")
        self._late_extension = True
        return CommitDecision(tuple(self._drain_outbox()), self.pending_raw, self.pending_kind, "timeout", True, ("text.fallback",))

    def _append(self, text: str, now: float) -> None:
        for ch in text:
            if _is_emoji(ch) or is_emoji_char(ch):
                # Emoji are filtered at grapheme level by the transport too;
                # keep this guard here so direct committer users cannot leak a
                # ZWJ/keycap member into an open semantic span.
                if self._pending.raw and self._pending.raw in "0123456789#*" and ord(ch) in (0xFE0F, 0x20E3):
                    self._pending = _Pending()
                continue
            if self._pending.raw and self._pending.kind == SpanKind.MARKDOWN:
                self._pending.raw += ch
                self._pending.last_at = now
                raw = self._pending.raw
                closed = (
                    (raw.startswith("**") and raw.endswith("**") and len(raw) > 4)
                    or (raw.startswith("__") and raw.endswith("__") and len(raw) > 4)
                    or (raw.startswith("```") and raw.count("```") >= 2)
                    or (raw.startswith("`") and not raw.startswith("``") and raw.count("`") >= 2)
                    or (raw.startswith("[") and ")" in raw)
                    or (raw.startswith("#") and ch in "\r\n")
                )
                if closed:
                    self._close_pending()
                continue
            if self._pending.raw and self._pending.kind == SpanKind.JSON:
                self._pending.raw += ch
                self._pending.last_at = now
                if self._json_escape:
                    self._json_escape = False
                elif ch == "\\" and self._json_in_string:
                    self._json_escape = True
                elif ch == '"':
                    self._json_in_string = not self._json_in_string
                elif not self._json_in_string and ch in "[{":
                    self._json_depth = getattr(self, "_json_depth", 0) + 1
                elif not self._json_in_string and ch in "]}":
                    self._json_depth = max(0, getattr(self, "_json_depth", 0) - 1)
                    if self._json_depth == 0:
                        self._close_pending()
                continue
            if not self._pending.raw:
                if _is_emoji(ch):
                    continue
                if ch == "{":
                    self._flush_plain()
                    self._pending = _Pending(ch, self.committed_raw_end, SpanKind.JSON, now, now)
                    self._json_depth = 1
                    self._json_in_string = False
                    self._json_escape = False
                elif ch in "*_`[#":
                    self._flush_plain()
                    self._pending = _Pending(ch, self.committed_raw_end, SpanKind.MARKDOWN, now, now)
                elif ch.isspace() or unicodedata.category(ch).startswith("P") and ch not in "%$@#":
                    self._emit_plain(ch)
                elif ord(ch) > 127 and not ch.isascii():
                    self._current_lang = "zh"
                    self._emit_plain(ch)
                else:
                    self._flush_plain()
                    self._pending = _Pending(ch, self.committed_raw_end, self._classify(ch), now, now)
                continue
            # CJK or whitespace closes an ASCII semantic span. Punctuation closes
            # all spans except symbols that are part of numbers/formulas.
            if ch.isspace() or (not ch.isascii() and not _is_emoji(ch)):
                if self._pending.kind == SpanKind.NUMBER and ch in "°²":
                    self._pending.raw += ch
                    self._pending.last_at = now
                    self._pending.kind = self._classify(self._pending.raw)
                    continue
                self._close_pending()
                if ch.isspace() or unicodedata.category(ch).startswith("P"):
                    self._emit_plain(ch)
                else:
                    self._emit_plain(ch)
            elif ch in "。！？；,，.!?;:" and self._pending.kind != SpanKind.URL:
                if self._pending.kind == SpanKind.NUMBER and ch in ".:" and self._pending.raw[-1:].isdigit():
                    self._pending.raw += ch
                    self._pending.last_at = now
                    continue
                self._close_pending()
                self._emit_plain(ch)
            else:
                self._pending.raw += ch
                self._pending.last_at = now
                self._pending.kind = self._classify(self._pending.raw)
                if len(self._pending.raw) >= self.config.max_pending_chars:
                    self._close_pending()

    def _close_stable(self, now: float) -> list[TextCommit]:
        # A completed number with a suffix is closed when a CJK boundary arrives;
        # ordinary ASCII words remain pending until whitespace/final.
        return []

    def _flush_pending(self, *, reason: str) -> list[TextCommit]:
        if not self._pending.raw:
            return []
        self._close_pending(reason=reason)
        return []

    def _close_pending(self, *, reason: str = "boundary") -> TextCommit:
        p = self._pending
        force_literal = self._force_literal_next
        self._force_literal_next = False
        lang = self._current_lang if self.config.language == "mixed_zh_en" else ("zh" if self.config.language.startswith("zh") else "en")
        if any(ord(c) > 127 for c in p.raw):
            lang = "zh"
        elif p.kind != SpanKind.NUMBER and any(c.isalpha() for c in p.raw):
            lang = "en"
        decimal_number = p.kind == SpanKind.NUMBER and re.fullmatch(r"[+\-]?\d+(?:\.\d+)?%?", p.raw)
        value = (
            project_readable(p.raw)
            if p.kind in (SpanKind.JSON, SpanKind.MARKDOWN)
            else (
                self.adapter.fallback(p.raw, lang=lang, kind=p.kind, policy=self.config.fallback)
                if decimal_number and not force_literal
                else (None if force_literal else self.adapter.normalize(p.raw, lang=lang, kind=p.kind))
            )
        )
        kind = CommitKind.NORMALIZED
        if value is None:
            value = self.adapter.fallback(p.raw, lang=lang, kind=p.kind, policy=self.config.fallback)
            kind = CommitKind.FALLBACK
        commit = self._make_commit(p.raw, p.start, p.kind, kind, value)
        self.committed_raw_end = p.start + len(p.raw)
        self.committed_spoken_text += value
        self.commit_fence += 1
        self._pending = _Pending()
        self._outbox.append(commit)
        return commit

    def _fallback_pending(self, *, reason: str) -> TextCommit:
        return self._close_pending(reason=reason)

    def _emit_plain(self, ch: str) -> None:
        if not self._plain_buffer:
            self._plain_start = self.committed_raw_end
        self._plain_buffer += ch
        self.committed_raw_end += len(ch)
        self.committed_spoken_text += ch
        self.commit_fence += 1

    def _flush_plain(self) -> None:
        if not self._plain_buffer:
            return
        self._outbox.append(
            self._make_commit(
                self._plain_buffer,
                self._plain_start,
                SpanKind.PLAIN,
                CommitKind.LITERAL,
            )
        )
        self._plain_buffer = ""

    def _drain_outbox(self) -> list[TextCommit]:
        out = self._outbox
        self._outbox = []
        return out

    def _make_commit(self, raw: str, start: int, span_kind: SpanKind, commit_kind: CommitKind, value: str | None = None) -> TextCommit:
        return TextCommit(
            start,
            start + len(raw),
            raw if value is None else value,
            span_kind,
            commit_kind,
            self.commit_fence,
            ((start, start + len(raw)),),
            raw,
        )

    @staticmethod
    def _classify(raw: str) -> SpanKind:
        if any(c in _MATH_CHARS for c in raw) and any(c in "*=×÷" for c in raw):
            return SpanKind.MATH
        if "@" in raw:
            return SpanKind.EMAIL
        if re.fullmatch(r"(?:A\$|HKD|[$€￥£])\d+(?:\.\d+)?", raw):
            return SpanKind.NUMBER
        if re.fullmatch(r"\d{4}[-/.]\d{1,2}(?:[-/.]\d{1,2})?", raw):
            return SpanKind.NUMBER
        if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?(?:m²|km/h|km|kg|ms|°C|m|mm|cm)", raw):
            return SpanKind.NUMBER
        if "://" in raw or raw.startswith(("www.", "http")):
            return SpanKind.URL
        if _ORDINAL.fullmatch(raw):
            return SpanKind.ORDINAL
        if _NUMERIC.fullmatch(raw):
            return SpanKind.NUMBER
        return SpanKind.ENGLISH_WORD
