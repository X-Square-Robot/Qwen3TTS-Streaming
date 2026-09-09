from __future__ import annotations

import html
import re
import time
import unicodedata
from dataclasses import dataclass
import logging

from ...text_normalization import is_emoji_char

from .language import LanguageResolver
from .causal import CausalFrontier
from .types import (
    CommitmentState,
    CommitmentUpdate,
    CommitDecision,
    CommitKind,
    LanguageKind,
    SpanKind,
    SemanticFamily,
    TextInputMetadata,
    TextCommit,
    TextNormalizationConfig,
)
from .candidate_resolver import CandidateResolver, family_for_kind
from .semantic_spans import SpanDetector
from .commit_policy import CommitPolicy
from .domain_resolver import DomainResolver
from .wetext_backend import WetextAdapter
from .projector import is_markdown_structured, project_readable


logger = logging.getLogger(__name__)


_ASCII_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_@.$:/+\-\\]*")
_NUMERIC = re.compile(r"^[+\-]?\d+(?:[.,]\d+)?(?:%|[A-Za-z]{1,8})?$")
_ORDINAL = re.compile(r"^\d{1,6}(?:st|nd|rd|th)$", re.I)
_VERSION = re.compile(r"^(?:v)?\d+(?:[._-]\d+)+(?:[A-Za-z]+\d*)?$", re.I)
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*@[0-9][A-Za-z0-9._-]*$")
_MODEL_CODE = re.compile(r"^[A-Za-z]+-\d+$")
# Phone numbers are recognized before the generic math/version rules.  Keep
# the grammar deliberately conservative: require a country-code marker or
# conventional grouped domestic digits, so dates such as 2029/02/02 remain
# NUMBER spans.
_PHONE_SEGMENT = (
    r"(?:\+\d{1,3}[-\s]?\d{6,}|"
    r"\(\+\d{1,3}\)\s*\d{6,}|"
    r"0\d{2,3}[-\s]\d{3,4}[-\s]\d{4}|"
    r"0\d{9,})"
)
_PHONE = re.compile(rf"^{_PHONE_SEGMENT}(?:\s*/\s*{_PHONE_SEGMENT})?$")
_HTML_ENTITY = re.compile(r"^&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);$")
_ID_CARD = re.compile(r"^\d{17}[0-9Xx]$")
# A percent suffix is a strong semantic boundary even when an upstream
# tokenizer glues it to an English word (``English20%``).  Keep this narrow:
# arbitrary alphanumeric product names remain one identifier/word span.
_EMBEDDED_NUMERIC_PERCENT = re.compile(
    r"^([A-Za-z][A-Za-z-]*?)([+\-]?\d+(?:\.\d+)?%)$"
)
_OPEN_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\([^)]*$")
_TIME = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?(?:[AaPp][Mm])?$")
_RANGE = re.compile(r"^\d+(?:\.\d+)?\s*(?:-|~)\s*\d+(?:\.\d+)?$")
# ``x``/``X`` are accepted as multiplication signs only when they occur
# between numeric operands.  They are intentionally not treated as a generic
# alphabetic operator: an isolated English ``x`` must remain an ordinary word.
_MATH_CHARS = set("0123456789.+-*/=^×÷()<>≤≥!≠xX")
_MATH_BINARY_OPERATORS = "+*/=^×÷<>≤≥≠xX"
_RIGHT_BOUNDARY_CHARS = ")]}" + "）】》」』〉»”’"
_MATH_CLOSERS = ')]}'
_QUALIFIED_NUMBER = re.compile(
    r"(?:[$€￥£¥]|A\$|HKD)|(?:\d{4}[-/.]\d{1,2}(?:[-/.]\d{1,2})?)|"
    r"(?:\d{1,2}:\d{2}(?::\d{2})?(?:[AaPp][Mm])?|"
    r"\d+(?:\.\d+)?\s*(?:-|~)\s*\d+(?:\.\d+)?|"
    r"m²|km/h|km|kg|ms|°C|℃|m|mm|cm|[μµ]g/m³)$"
)


def _is_emoji(ch: str) -> bool:
    return "EMOJI" in unicodedata.name(ch, "") or ord(ch) in range(0x1F000, 0x1FAFF)


def _is_han(ch: str) -> bool:
    """Return whether *ch* is a CJK ideograph, not merely non-ASCII."""

    name = unicodedata.name(ch, "")
    return "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name


def _is_combining(ch: str) -> bool:
    """Keep combining marks attached to the preceding grapheme/span."""

    return unicodedata.combining(ch) != 0 or unicodedata.category(ch).startswith("M")


def _at_line_start(text: str, position: int) -> bool:
    """Return whether *position* is the first non-indent code point of a line."""

    line = text[:position].rsplit("\n", 1)[-1].rsplit("\r", 1)[-1]
    return not line.strip(" \t")


def _at_inline_ordered_list_start(text: str, position: int) -> bool:
    """Return whether a numbered marker starts an inline enumeration.

    Markdown's block-list grammar requires a line start, but model prose often
    emits compact enumerations such as ``以下几类：1. ...。2. ...`` without
    newlines.  Treat only strong sentence/list separators as an inline list
    boundary; a generic space is intentionally excluded so prose like
    ``version 2. item`` remains literal.
    """

    if _at_line_start(text, position):
        return True
    prefix = text[:position].rstrip(" \t")
    return bool(prefix and prefix[-1] in "：:。！？!?；;")


def _compatibility_spelling(text: str) -> str:
    """Map Unicode compatibility characters without composing graphemes.

    Applying NFKC to the whole string would turn ``e`` + combining acute into
    a precomposed ``é``.  Mapping each source code point independently still
    handles full-width digits/operators but preserves grapheme boundaries and
    the original source alignment used by the committer.
    """

    # Superscript/subscript digits carry unit/exponent semantics (``m³``,
    # ``x²``); mapping them to ordinary digits would destroy those patterns.
    preserve = set("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉")
    return "".join(
        ch if ch in preserve else unicodedata.normalize("NFKC", ch)
        for ch in text
    )


def _canonical_math_spelling(text: str) -> str:
    """Normalize multi-codepoint comparison operators for lexical matching."""

    canonical = _compatibility_spelling(text)
    # Permit a model to place spaces around a two-character comparator
    # (``2 ! = 1``) while retaining ordinary whitespace for the lexer.
    canonical = re.sub(r"!\s*=", "≠", canonical)
    canonical = re.sub(r">\s*=", "≥", canonical)
    canonical = re.sub(r"<\s*=", "≤", canonical)
    return canonical.replace("==", "=")


def _looks_like_math(raw: str) -> bool:
    """Recognise an arithmetic-looking island without stealing signed numbers.

    The lexer may see a trailing operator (``3*``) before its right operand;
    retaining that prefix as ``MATH`` is what lets the next packet complete the
    expression.  A leading ``+``/``-`` remains a numeric sign until another
    operand/operator makes the expression unambiguous.
    """

    # Compatibility forms (full-width digits/operators, for example) are
    # semantically equivalent to their ASCII spelling.  Keep the original
    # string for source offsets, but classify against its NFKC view.
    raw = _canonical_math_spelling(raw)
    if not raw or not any(char in _MATH_CHARS for char in raw):
        return False
    # A hyphen between two numbers is deliberately *not* enough evidence for
    # arithmetic: ``3-2`` is commonly a range/date/model fragment.  Other
    # operators (including ASCII/Unicode multiplication) are unambiguous.
    binary = rf"\d\s*[{re.escape(_MATH_BINARY_OPERATORS)}]\s*[+\-]?\s*\d"
    if re.search(binary, raw):
        return True
    # An opening parenthesis may temporarily sit between the operator and its
    # right operand (``3 * (`` | ``2``).  Keep the island in MATH while the
    # grouped operand is being assembled.
    if re.search(rf"\d\s*[{re.escape(_MATH_BINARY_OPERATORS)}]\s*\(", raw):
        return True
    # Once an expression contains an explicit equality or a second arithmetic
    # operator, a hyphen in the same island is arithmetic too (``3-2=1``).
    if re.search(r"\d\s*-\s*\d", raw) and (
        "=" in raw
        or re.search(rf"\d\s*[{re.escape(_MATH_BINARY_OPERATORS)}]", raw)
    ):
        return True
    # Keep an unfinished expression open across packets.  A trailing hyphen
    # remains a possible range and is therefore handled as NUMBER by
    # ``_classify`` until its right operand or an explicit operator arrives.
    if re.search(rf"\d\s*[{re.escape(_MATH_BINARY_OPERATORS)}]\s*$", raw):
        return True
    # A signed right operand can arrive in a later packet (``3*`` | ``-2``).
    # Keep the expression in MATH while that unary sign is still open.
    if re.search(
        rf"\d\s*[{re.escape(_MATH_BINARY_OPERATORS)}]\s*[+\-]\s*$",
        raw,
    ):
        return True
    if re.search(r"\d\s*!\s*$", raw):
        return True
    if re.search(r"[A-Za-z]\s*\^\s*\d", raw):
        return True
    return False


def _tn_input(raw: str, kind: SpanKind) -> str:
    """Return the spelling passed to the TN backend.

    Unicode compatibility forms are common in model output (full-width
    digits/punctuation in particular).  Normalizing only semantic spans lets
    wetext recognize those forms while keeping ``TextCommit.raw_text`` and
    source offsets byte/codepoint-for-codepoint with the original input.
    Structured projections and ordinary prose deliberately retain their
    spelling because punctuation/format markers are meaningful to the
    outer lexer.
    """

    if kind in {
        SpanKind.NUMBER,
        SpanKind.ORDINAL,
        SpanKind.MATH,
        SpanKind.IDENTIFIER,
        SpanKind.VERSION,
        SpanKind.EMAIL,
        SpanKind.URL,
        SpanKind.PHONE,
        SpanKind.ID_CARD,
        SpanKind.ENGLISH_WORD,
    }:
        return _compatibility_spelling(raw)
    return raw


def _sanitize_url_for_tn(raw: str) -> str:
    """Decode HTML character references embedded in a URL span.

    URL punctuation remains intact, but formatting entities such as
    ``&#x20;`` must not be exposed to WeText as literal ``#x20`` digits.  A
    missing semicolon is accepted for compatibility with common HTML output.
    """

    value = html.unescape(raw)

    def decode_hex(match: re.Match[str]) -> str:
        try:
            return chr(int(match.group(1), 16))
        except (ValueError, OverflowError):
            return match.group(0)

    def decode_dec(match: re.Match[str]) -> str:
        try:
            return chr(int(match.group(1), 10))
        except (ValueError, OverflowError):
            return match.group(0)

    value = re.sub(r"&#x([0-9A-Fa-f]+);?", decode_hex, value)
    value = re.sub(r"&#([0-9]+);?", decode_dec, value)
    return value.rstrip()


@dataclass
class _Pending:
    raw: str = ""
    start: int = 0
    end: int = 0
    kind: SpanKind = SpanKind.PLAIN
    first_at: float = 0.0
    last_at: float = 0.0
    span_id: int = 0
    language_hint: str | None = None
    # A suffix which arrived after a timed-out span gets its own immutable
    # literal fence.  This is attached to the first newly opened span rather
    # than kept as a session-wide switch, otherwise an intervening ordinary
    # word/JSON/Markdown span would be forced down the late-literal path too.
    force_literal: bool = False


class IncrementalTextCommitter:
    """Append-only semantic commit fence in front of the TTS tokenizer."""

    def __init__(self, config: TextNormalizationConfig | None = None, *, adapter: WetextAdapter | None = None):
        self.config = config or TextNormalizationConfig()
        self.adapter = adapter or WetextAdapter()
        self._detector = SpanDetector(self._classify)
        self._commit_policy = CommitPolicy()
        self._domain_resolver = DomainResolver(self.adapter)
        self._candidate_resolver = CandidateResolver(
            self.adapter.backend,
            nbest=getattr(self.config, "candidate_nbest", 8),
        )
        self.raw_cursor = 0
        self.committed_raw_end = 0
        self.committed_spoken_text = ""
        self.commit_fence = 0
        self._pending = _Pending()
        self._raw = ""
        self._outbox: list[TextCommit] = []
        self._late_extension = False
        self._force_literal_next = False
        self._late_extension_anchor: int | None = None
        self._limit_fallback = False
        self._finalized = False
        self._current_lang = "unknown" if self.config.language == "mixed_zh_en" else ("zh" if self.config.language.startswith("zh") else "en")
        self._json_in_string = False
        self._json_escape = False
        self._json_depth = 0
        # ``[`` is provisionally treated as Markdown because it may start a
        # link.  Once a JSON-array-leading value is observed the span is
        # promoted to JSON and parsed with the same depth/quote state as an
        # object span.
        self._bracket_candidate = False
        self._bracket_closed_candidate = False
        self._markdown_line_marker = False
        self._ordered_marker_candidate = False
        self._leading_paren_candidate = False
        self._image_candidate = False
        self._autolink_candidate = False
        self._bang_candidate = False
        self._plain_buffer = ""
        self._plain_start = 0
        # Defer a space after a number until we see whether an operator
        # follows (e.g. ``5 > 3``).  It is emitted as plain text otherwise.
        self._pending_gap: list[tuple[str, int]] = []
        self._language_resolver = LanguageResolver(self.config.language)
        self._next_commit_id = 1
        self._next_span_id = 1
        # Even in conservative closed-span mode, keep the generic causal
        # frontier in the production path as an append-only invariant.  A
        # future prefix-oracle mode can feed n-best candidates into the same
        # object without changing the transport/fence contract.
        self._causal_frontier = CausalFrontier()
        self._frontier_source_text = ""
        self._active_metadata = TextInputMetadata()
        self._active_lang_hint: str | None = None

    @property
    def pending_raw(self) -> str:
        return self._pending.raw

    @property
    def raw_text(self) -> str:
        """Complete raw source observed by this session.

        The value includes pending spans.  It is exposed read-only so the
        frontend journal can retain original source coordinates while keeping
        only append-only spoken commits in the TTS tokenizer.
        """

        return self._raw

    @property
    def causal_frontier(self) -> CausalFrontier:
        """Read-only access to the session's monotonic spoken frontier."""

        return self._causal_frontier

    @property
    def finalized(self) -> bool:
        """Whether this input stream has received its final flush."""

        return self._finalized

    @property
    def pending_kind(self) -> SpanKind | None:
        return self._pending.kind if self._pending.raw else None

    @property
    def next_deadline(self) -> float | None:
        if not self._pending.raw:
            return None
        # A transport packet may hold the last digit temporarily while it
        # checks whether a keycap marker follows (``2`` plus U+20E3).  Do not let
        # the semantic deadline commit that digit before the next packet can
        # extend an ordinary number (``25``); the explicit final flush still
        # closes it normally.  This keeps emoji filtering from changing TN
        # semantics and prevents a split number from reaching the cursor
        # labelizer as two unrelated spans.
        if self._is_provisional_single_digit():
            return None
        return min(
            self._pending.first_at + self.config.semantic_max_wait_ms / 1000.0,
            self._pending.last_at + self.config.semantic_idle_wait_ms / 1000.0,
        )

    def _is_provisional_single_digit(self) -> bool:
        """Whether the open numeric span needs one more input boundary."""

        if self._pending.kind not in (SpanKind.NUMBER, SpanKind.ORDINAL):
            return False
        raw = _compatibility_spelling(self._pending.raw).strip()
        return len(raw) == 1 and raw.isdigit()

    def feed(
        self,
        text: str,
        *,
        final: bool = False,
        now: float | None = None,
        metadata: TextInputMetadata | None = None,
    ) -> CommitDecision:
        now = time.monotonic() if now is None else now
        if self._finalized:
            # EOS is an append fence for the committer itself as well as for
            # the surrounding Session.  A transport bug that sends a packet
            # after finalization must not reopen a span or mutate the raw
            # coordinate space.  Return a typed terminal decision so callers
            # can observe the protocol violation without an exception.
            return CommitDecision(
                reason="after_final",
                state=CommitmentState.DONE,
                events=("text.input.after_final",) if text else (),
                committed_raw_end=self.committed_raw_end,
            )
        poll_decision = self.poll(now=now)
        commits = list(poll_decision.commits)
        events = list(poll_decision.events)
        fallback = poll_decision.fallback
        if not self.config.enabled:
            if text:
                start = self.raw_cursor
                self._raw += text
                commits.append(
                    self._make_commit(
                        text,
                        start,
                        SpanKind.LITERAL,
                        CommitKind.LITERAL,
                    )
                )
                self.raw_cursor += len(text)
                self.committed_raw_end = self.raw_cursor
            if final:
                self._finalized = True
            return CommitDecision(
                commits=tuple(commits),
                reason="disabled",
                state=CommitmentState.DONE if final else CommitmentState.SCAN,
                committed_raw_end=self.committed_raw_end,
            )
        late_event = self._late_extension
        self._active_metadata = metadata or TextInputMetadata()
        # Metadata is attached to a transport delta.  Do not let a hint from a
        # previous packet leak across a code-switch when the next packet omits
        # it; session-wide language remains represented by the config/resolver.
        self._active_lang_hint = (
            self._active_metadata.language_hint.value
            if self._active_metadata.language_hint in (LanguageKind.ZH, LanguageKind.EN)
            else None
        )
        if self._pending.raw and self._active_lang_hint and not self._pending.language_hint:
            self._pending.language_hint = self._active_lang_hint
        if text:
            if self._late_extension:
                self._late_extension = False
                self._force_literal_next = True
            source_start = self.raw_cursor
            self._raw += text
            self.raw_cursor += len(text)
            self._append(text, now, source_start)
        if self._active_metadata.closed and self._pending.raw and not final:
            self._close_pending(reason="metadata_closed")
        if final:
            self._flush_pending(reason="final")
        else:
            commits.extend(self._close_stable(now))
        self._flush_plain()
        if self._limit_fallback:
            # A structural span that reaches the configured bound is closed
            # deterministically.  It must be observable even when the close
            # happened deep inside the lexer and no timer fired.
            fallback = True
            events.append("text.fallback")
            self._limit_fallback = False
        commits.extend(self._outbox)
        self._outbox.clear()
        if any(commit.commit_kind is CommitKind.FALLBACK for commit in commits):
            fallback = True
            if "text.fallback" not in events:
                events.append("text.fallback")
        if late_event:
            events.append("text.span.late_extension")
        if final:
            self._finalized = True
        return CommitDecision(
            commits=tuple(commits),
            pending_raw=self.pending_raw,
            pending_kind=self.pending_kind,
            reason="final" if final else "feed",
            fallback=fallback,
            events=tuple(dict.fromkeys(events)),
            state=(
                CommitmentState.DONE
                if final and not self.pending_raw
                else CommitmentState.OPEN
                if self.pending_raw
                else CommitmentState.FALLBACK
                if fallback
                else CommitmentState.COMMIT
            ),
            committed_raw_end=self.committed_raw_end,
        )

    def poll(self, *, now: float | None = None) -> CommitDecision:
        now = time.monotonic() if now is None else now
        if self._finalized:
            return CommitDecision(
                reason="after_final",
                state=CommitmentState.DONE,
                committed_raw_end=self.committed_raw_end,
            )
        if not self._pending.raw or self.next_deadline is None or now < self.next_deadline:
            return CommitDecision(
                pending_raw=self.pending_raw,
                pending_kind=self.pending_kind,
                state=CommitmentState.OPEN if self.pending_raw else CommitmentState.SCAN,
                committed_raw_end=self.committed_raw_end,
            )
        self._fallback_pending(reason="timeout")
        self._late_extension = True
        self._late_extension_anchor = self.committed_raw_end
        return CommitDecision(
            commits=tuple(self._drain_outbox()),
            pending_raw=self.pending_raw,
            pending_kind=self.pending_kind,
            reason="timeout",
            fallback=True,
            events=("text.fallback",),
            state=CommitmentState.FALLBACK,
            committed_raw_end=self.committed_raw_end,
        )

    def feed_update(
        self,
        text: str,
        *,
        final: bool = False,
        now: float | None = None,
        metadata: TextInputMetadata | None = None,
    ) -> CommitmentUpdate:
        """Return the typed X2-style update without changing legacy callers.

        ``CommitDecision`` remains the compatibility return type used by the
        existing frontend.  New integrations should consume this explicit
        update shape so pending data cannot be confused with committable text.
        """

        decision = self.feed(text, final=final, now=now, metadata=metadata)
        return CommitmentUpdate(
            commits=decision.commits,
            pending_raw=decision.pending_raw,
            pending_kind=decision.pending_kind,
            state=decision.state,
            reason=decision.reason,
            fallback=decision.fallback,
            events=decision.events,
            committed_raw_end=decision.committed_raw_end,
        )

    def poll_update(self, *, now: float | None = None) -> CommitmentUpdate:
        """Typed counterpart of :meth:`poll` for new engine integrations."""

        decision = self.poll(now=now)
        return CommitmentUpdate(
            commits=decision.commits,
            pending_raw=decision.pending_raw,
            pending_kind=decision.pending_kind,
            state=decision.state,
            reason=decision.reason,
            fallback=decision.fallback,
            events=decision.events,
            committed_raw_end=decision.committed_raw_end,
        )

    def _open_pending(
        self,
        raw: str,
        start: int,
        end: int,
        kind: SpanKind,
        now: float,
        language_hint: str | None = None,
        force_literal: bool | None = None,
    ) -> _Pending:
        """Create a uniquely identified unresolved span.

        The identifier is independent of commit fences: a timed-out span and
        its late suffix must remain distinguishable even though the suffix is
        processed after the old fence has been frozen.
        """

        if force_literal is None:
            force_literal = bool(
                self._force_literal_next
                and self._late_extension_anchor is not None
                and start == self._late_extension_anchor
            )
            # Consume the one-shot late-extension marker at the first new
            # semantic span.  If a separator was seen first, the later span
            # starts past the anchor and is intentionally eligible for the
            # normal TN/structured path.
            if self._force_literal_next:
                self._force_literal_next = False
                self._late_extension_anchor = None
        span = _Pending(
            raw,
            start,
            end,
            kind,
            now,
            now,
            self._next_span_id,
            language_hint or self._active_lang_hint,
            bool(force_literal),
        )
        self._next_span_id += 1
        return span

    def _append(self, text: str, now: float, source_start: int) -> None:
        for offset, ch in enumerate(text):
            source_pos = source_start + offset
            # Use a one-codepoint compatibility view for lexical decisions.
            # The original ``ch`` is always retained in pending/raw spans so
            # offsets and diagnostics stay in the caller's coordinate space.
            # NFKC can expand a character into several code points; in that
            # unusual case retaining the original is the safe choice.
            normalized_ch = unicodedata.normalize("NFKC", ch)
            lex_ch = normalized_ch if len(normalized_ch) == 1 else ch
            if _is_emoji(ch) or is_emoji_char(ch):
                # Emoji are filtered at grapheme level by the transport too;
                # keep this guard here so direct committer users cannot leak a
                # ZWJ/keycap member into an open semantic span.
                if (
                    self._pending.raw
                    and not self._pending_gap
                    and self._pending.raw[-1:] in "0123456789#*"
                    and ord(ch) in (0xFE0F, 0x20E3)
                ):
                    # A base digit followed immediately by VS-16/keycap is a
                    # single emoji grapheme, not spoken numeric material.  If
                    # the base is attached to a preceding Latin/number span
                    # (``hello1`` | ``️⃣``), remove just that base, close the
                    # preceding portion, and advance the raw frontier over the
                    # discarded sequence.  This keeps direct committer users
                    # consistent with the transport's grapheme assembler.
                    if len(self._pending.raw) > 1:
                        base_start = max(self._pending.start, self._pending.end - 1)
                        self._pending.raw = self._pending.raw[:-1]
                        self._pending.end = base_start
                        self._close_pending()
                    else:
                        self._pending = _Pending()
                    self.committed_raw_end = max(
                        self.committed_raw_end, source_pos + 1
                    )
                else:
                    self._close_pending()
                self._flush_plain()
                continue
            # Hold an exclamation mark following an ordinary Latin run for
            # one code point: ``word![alt](...)`` is an image construct, while
            # ``hello!world`` must retain the punctuation as literal text.
            # If the next character is not ``[``, close the word before the
            # marker and rescan the current character normally.
            if (
                self._pending.raw
                and self._pending.kind is SpanKind.ENGLISH_WORD
                and self._bang_candidate
            ):
                if ch == "[":
                    self._bang_candidate = False
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    continue
                prefix = self._pending.raw[:-1]
                marker_start = self._pending.end - 1
                self._pending.raw = prefix
                self._pending.end = marker_start
                self._bang_candidate = False
                self._close_pending()
                self._emit_plain("!", marker_start)
                self._append(ch, now, source_pos)
                continue
            # Formatting delimiters adjacent to an ordinary Latin run remain
            # part of that unresolved run until a matching pair/structured
            # construct is visible.  This is important for identifiers such
            # as ``foo_bar`` and prose such as ``a*b``: splitting on the first
            # punctuation character would make an unconfirmed marker
            # disappear at the commit fence.  A numeric/formula span still
            # takes the operator-aware path below.
            if (
                self._pending.raw
                and self._pending.kind not in (SpanKind.MARKDOWN, SpanKind.JSON)
                and ch in "*_~`[!"
                and self._pending.kind is not SpanKind.ENGLISH_WORD
            ):
                # Do not split an already-open Markdown span: its delimiters
                # are intentionally consumed by the branch below (otherwise
                # ``**bold**`` becomes several one-character spans).  For a
                # semantic number/formula, retain operators that happen to
                # share Markdown punctuation (``3*2``, ``2!=1`` and numeric
                # ranges such as ``3~5``).
                keep_math_delimiter = (
                    self._pending.kind in (SpanKind.NUMBER, SpanKind.MATH)
                    and (
                        (ch == "*" and _looks_like_math(self._pending.raw + ch))
                        or ch == "!"
                        or (ch == "_" and self._pending.kind is SpanKind.NUMBER)
                        or (
                            ch == "~"
                            and bool(
                                re.search(
                                    r"[+\-]?\d+(?:\.\d+)?\s*$",
                                    _compatibility_spelling(self._pending.raw),
                                )
                            )
                        )
                    )
                )
                if not keep_math_delimiter and self._pending.kind not in (
                    SpanKind.URL,
                    SpanKind.EMAIL,
                    SpanKind.IDENTIFIER,
                    SpanKind.VERSION,
                ):
                    self._close_pending()
                    self._append(ch, now, source_pos)
                    continue
            if self._pending.raw and self._pending.kind == SpanKind.MARKDOWN:
                self._pending.raw += ch
                self._pending.end = source_pos + 1
                self._pending.last_at = now
                raw = self._pending.raw

                # An ordered Markdown list (``1. item``) is ambiguous with a
                # decimal/date prefix until the character after the dot
                # arrives.  Keep the marker provisional: whitespace confirms
                # formatting, while a digit restores the NUMBER span
                # (``1.5``/``2026.07``).
                if self._ordered_marker_candidate:
                    # Locate the dot rather than assuming a one-digit marker;
                    # ``12. item`` and full-width digits must follow the same
                    # path as ``1. item``.
                    dot_index = raw.find(".")
                    after_dot = raw[dot_index + 1 : dot_index + 2] if dot_index >= 0 else ""
                    if after_dot and after_dot.isspace():
                        self._ordered_marker_candidate = False
                        self._close_pending()
                        continue
                    if after_dot and (
                        after_dot.isdigit()
                        or _compatibility_spelling(after_dot).isdigit()
                    ):
                        self._ordered_marker_candidate = False
                        self._pending.kind = self._detector.classify(raw)
                        continue
                    if len(raw) > 2:
                        self._ordered_marker_candidate = False
                        marker = raw[: dot_index + 1] if dot_index >= 0 else raw
                        self._pending.raw = marker
                        self._pending.end = self._pending.start + len(marker)
                        self._close_pending()
                        self._append(ch, now, source_pos)
                        continue

                # ``[`` is ambiguous between a Markdown link and a JSON
                # array.  Keep it provisional until the first meaningful
                # value makes JSON sufficiently likely.  This avoids routing
                # ``[label](url)`` through the JSON parser while handling
                # arrays split at any packet boundary.
                if self._bracket_candidate:
                    meaningful = raw[1:].lstrip()
                    first = meaningful[:1]

                    # A scalar-only array and a numeric Markdown link label
                    # have identical prefixes (``[1]``).  Delay the JSON
                    # decision until the closing bracket: ``](`` proves a
                    # Markdown link, while a boundary/end proves JSON.  Quote
                    # and escape state is tracked provisionally so a ``]``
                    # inside a JSON string cannot close the candidate.
                    if self._json_escape:
                        self._json_escape = False
                    elif ch == "\\" and self._json_in_string:
                        self._json_escape = True
                    elif ch == '"' and (self._json_in_string or first == '"'):
                        self._json_in_string = not self._json_in_string

                    # Nested containers are unambiguous JSON evidence and can
                    # be promoted immediately; the current opening delimiter
                    # contributes one level below the outer array.
                    if not self._json_in_string and ch in "[{":
                        self._pending.kind = SpanKind.JSON
                        self._bracket_candidate = False
                        self._bracket_closed_candidate = False
                        self._json_depth = 2
                        self._json_escape = False
                        continue

                    if not self._json_in_string and ch == "]":
                        self._bracket_candidate = False
                        self._bracket_closed_candidate = True
                        continue

                    # A non-JSON first value (typically a Markdown link label)
                    # permanently selects Markdown for this span.  JSON
                    # scalar/literal prefixes remain provisional as described
                    # above.
                    if first and first not in '"0123456789-tfn':
                        self._bracket_candidate = False

                elif self._bracket_closed_candidate:
                    if ch == "(":
                        # ``[label](`` confirms Markdown link syntax.
                        self._bracket_candidate = False
                        self._bracket_closed_candidate = False
                    else:
                        # The bracket was not followed by a link opener.  Turn
                        # the completed scalar array into JSON, close it, and
                        # scan the current character as surrounding prose.
                        prior = raw[:-1]
                        self._pending.raw = prior
                        self._pending.end = source_pos
                        self._pending.kind = SpanKind.JSON
                        self._bracket_candidate = False
                        self._bracket_closed_candidate = False
                        self._json_depth = 0
                        self._json_in_string = False
                        self._json_escape = False
                        self._close_pending()
                        self._append(ch, now, source_pos)
                        continue

                if self._image_candidate:
                    if raw == "![":
                        self._image_candidate = False
                    elif len(raw) > 1:
                        # A lone exclamation mark is ordinary punctuation;
                        # restore it before closing and rescan this character.
                        self._image_candidate = False
                        self._pending.raw = "!"
                        self._pending.end = self._pending.start + 1
                        self._close_pending()
                        self._append(ch, now, source_pos)
                        continue

                if self._autolink_candidate:
                    inner = raw[1:-1] if raw.endswith(">") else raw[1:]
                    prefixes = ("http://", "https://", "mailto:", "www.")
                    lower_inner = inner.lower()
                    if ch == ">" and any(
                        lower_inner.startswith(prefix) and len(inner) > len(prefix)
                        for prefix in prefixes
                    ):
                        self._autolink_candidate = False
                    elif any(
                        prefix.startswith(lower_inner) or lower_inner.startswith(prefix)
                        for prefix in prefixes
                    ):
                        continue
                    else:
                        # ``<tag``/``<3`` was not an autolink.  Preserve the
                        # opening angle bracket and rescan the current code
                        # point as ordinary input.
                        self._autolink_candidate = False
                        self._pending.raw = "<"
                        self._pending.end = self._pending.start + 1
                        self._close_pending()
                        self._append(ch, now, source_pos)
                        continue

                # Line-prefix markers are syntax, not spoken content.  Close
                # the marker as soon as its required separating whitespace is
                # seen, then let the body be scanned as ordinary prose.  A
                # sign immediately followed by a digit is retained as a
                # numeric negative/positive sign instead (``-2``).
                if self._markdown_line_marker:
                    marker = raw[:1]
                    if marker == "-" and re.fullmatch(r"-+", raw):
                        # Keep a run of line-start hyphens provisional until a
                        # boundary arrives.  Three or more are a Markdown
                        # horizontal rule; one/two remain literal if followed
                        # by a non-space body (``--foo``).
                        if len(raw) < 3:
                            continue
                        if ch.isspace() or ch in "\r\n" or len(raw) >= 3:
                            # Do not release the marker before seeing whether
                            # another hyphen extends the rule.  A later
                            # non-hyphen character will convert the run back
                            # to a literal span below.
                            if ch.isspace() or ch in "\r\n":
                                self._markdown_line_marker = False
                                self._close_pending()
                                continue
                            if len(raw) >= 3 and ch == "-":
                                continue
                    if marker == "-" and raw.startswith("--") and not re.fullmatch(r"-+", raw):
                        # A malformed horizontal-rule prefix is ordinary
                        # punctuation.  Keep it in the same unresolved span
                        # so it cannot be silently dropped or split into a
                        # negative number plus a word.
                        self._markdown_line_marker = False
                        self._pending.kind = SpanKind.ENGLISH_WORD
                        continue
                    if marker in "-+>" and len(raw) >= 2 and raw[1].isspace():
                        self._markdown_line_marker = False
                        self._close_pending()
                        continue
                    if marker in "-+" and len(raw) >= 2 and (
                        raw[1].isdigit() or _compatibility_spelling(raw[1]).isdigit()
                    ):
                        self._markdown_line_marker = False
                        self._pending.kind = self._detector.classify(raw)
                        continue
                    if len(raw) >= 2:
                        # ``>quote`` and malformed list markers still get a
                        # deterministic marker-only commit.  Restore the
                        # marker-only raw span before closing, then re-scan
                        # the current character so it is not swallowed by the
                        # formatting projection.
                        self._markdown_line_marker = False
                        self._pending.raw = marker
                        self._pending.end = self._pending.start + 1
                        self._close_pending()
                        self._append(ch, now, source_pos)
                        continue

                closed = (
                    # Prefer the longest delimiter first so ``***bold***``
                    # does not close after the first two trailing stars.
                    (raw.startswith("***") and raw.endswith("***") and len(raw) > 6)
                    or (raw.startswith("___") and raw.endswith("___") and len(raw) > 6)
                    or (raw.startswith("~~") and raw.endswith("~~") and len(raw) > 4)
                    or (raw.startswith("**") and not raw.startswith("***") and raw.endswith("**") and len(raw) > 4)
                    or (raw.startswith("__") and not raw.startswith("___") and raw.endswith("__") and len(raw) > 4)
                    or (raw.startswith("*") and not raw.startswith("**") and raw.endswith("*") and len(raw) > 2)
                    or (raw.startswith("_") and not raw.startswith("__") and raw.endswith("_") and len(raw) > 2)
                    or (raw.startswith("```") and raw.count("```") >= 2)
                    or (raw.startswith("`") and not raw.startswith("``") and raw.count("`") >= 2)
                    or (raw.startswith("[") and ")" in raw)
                    or (raw.startswith("![") and ")" in raw)
                    or (raw.startswith("#") and ch in "\r\n")
                )
                if closed:
                    self._close_pending()
                elif len(self._pending.raw) >= self.config.max_pending_chars:
                    # An unclosed Markdown delimiter must not pin a session
                    # forever.  Close at the configured lexical bound and let
                    # the normal fallback/event path make the degradation
                    # observable.
                    self._limit_fallback = True
                    self._close_pending(reason="max_pending")
                continue
            if self._pending.raw and self._pending.kind == SpanKind.JSON:
                self._pending.raw += ch
                self._pending.end = source_pos + 1
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
                if len(self._pending.raw) >= self.config.max_pending_chars:
                    self._limit_fallback = True
                    self._close_pending(reason="max_pending")
                continue
            # Once a link/image opener is complete, URL punctuation (``://``,
            # dots, query strings) is payload of the same unresolved
            # structure.  Keep it intact until the closing parenthesis so the
            # readable-label projector can make one atomic decision.
            if (
                self._pending.raw
                and self._pending.kind
                in (SpanKind.ENGLISH_WORD, SpanKind.URL, SpanKind.EMAIL, SpanKind.IDENTIFIER)
                and _OPEN_MARKDOWN_LINK.search(self._pending.raw)
            ):
                self._pending.raw += ch
                self._pending.end = source_pos + 1
                self._pending.last_at = now
                continue
            if self._pending.raw and self._pending.kind is SpanKind.MATH and self._leading_paren_candidate:
                # A leading ``(`` is provisional: numeric content promotes it
                # to a grouped formula, while a letter/punctuation closes the
                # opening parenthesis as ordinary prose and is rescanned.
                if lex_ch.isspace():
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    continue
                # Keep a country-code prefix provisional until its closing
                # parenthesis; otherwise ``(+86)191...`` is split before the
                # PHONE classifier sees the complete spelling.
                if ch == "+" and self._pending.raw == "(":
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    continue
                if (
                    (ch.isdigit() or lex_ch.isdigit())
                    and self._pending.raw.startswith("(+")
                    and ")" not in self._pending.raw
                ):
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    continue
                if ch.isdigit() or lex_ch.isdigit() or ch in "+-(":
                    self._leading_paren_candidate = False
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    self._pending.kind = SpanKind.MATH
                    continue
                if (
                    ch == ")"
                    and self._pending.raw.startswith("(+")
                    and self._pending.raw[2:].isdigit()
                ):
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    self._pending.kind = self._detector.classify(self._pending.raw)
                    continue
                self._leading_paren_candidate = False
                self._pending.raw = self._pending.raw[:1]
                self._pending.end = self._pending.start + 1
                self._close_pending()
                self._append(ch, now, source_pos)
                continue
            if not self._pending.raw:
                if _is_emoji(ch):
                    continue
                if lex_ch == "(":
                    self._flush_plain()
                    self._pending = self._open_pending(
                        ch, source_pos, source_pos + 1, SpanKind.MATH, now
                    )
                    self._leading_paren_candidate = True
                elif lex_ch == "{":
                    self._flush_plain()
                    self._pending = self._open_pending(
                        ch, source_pos, source_pos + 1, SpanKind.JSON, now
                    )
                    self._json_depth = 1
                    self._json_in_string = False
                    self._json_escape = False
                elif lex_ch == "[":
                    self._flush_plain()
                    self._pending = self._open_pending(
                        ch, source_pos, source_pos + 1, SpanKind.MARKDOWN, now
                    )
                    self._bracket_candidate = True
                elif lex_ch == "!":
                    self._flush_plain()
                    self._pending = self._open_pending(
                        ch, source_pos, source_pos + 1, SpanKind.MARKDOWN, now
                    )
                    self._image_candidate = True
                elif lex_ch == "<":
                    self._flush_plain()
                    self._pending = self._open_pending(
                        ch, source_pos, source_pos + 1, SpanKind.MARKDOWN, now
                    )
                    self._autolink_candidate = True
                elif lex_ch in "*_~`#":
                    self._flush_plain()
                    self._pending = self._open_pending(
                        ch, source_pos, source_pos + 1, SpanKind.MARKDOWN, now
                    )
                elif lex_ch in "-+>" and _at_line_start(self._raw, source_pos):
                    self._flush_plain()
                    self._pending = self._open_pending(
                        ch, source_pos, source_pos + 1, SpanKind.MARKDOWN, now
                    )
                    self._markdown_line_marker = True
                elif lex_ch in "$€￥£¥+-":
                    self._flush_plain()
                    self._pending = self._open_pending(
                        ch, source_pos, source_pos + 1, SpanKind.NUMBER, now
                    )
                elif ch == "&":
                    # Keep an HTML character reference together until its
                    # semicolon so ``&#x20;`` can be decoded as a space rather
                    # than sending the embedded digits through TN.
                    self._flush_plain()
                    self._pending = self._open_pending(
                        ch, source_pos, source_pos + 1, SpanKind.ENGLISH_WORD, now
                    )
                elif lex_ch.isspace() or unicodedata.category(lex_ch).startswith("P") and lex_ch not in "%$@#":
                    self._emit_plain(ch, source_pos)
                elif _is_combining(ch):
                    # A leading combining mark has no base to attach to yet;
                    # preserve it as literal text rather than manufacturing a
                    # semantic span that could be routed to the wrong graph.
                    self._emit_plain(ch, source_pos)
                elif _is_han(ch):
                    self._current_lang = "zh"
                    self._emit_plain(ch, source_pos)
                elif ch.isalpha() or ch.isdigit() or lex_ch.isalpha() or lex_ch.isdigit():
                    # Unicode Latin letters and full-width digits are still
                    # semantic word/number material.  Do not classify every
                    # non-ASCII code point as Chinese (``é`` was the notable
                    # failure mode).
                    self._flush_plain()
                    self._pending = self._open_pending(
                        ch,
                        source_pos,
                        source_pos + 1,
                        self._detector.classify(ch),
                        now,
                    )
                elif ord(ch) > 127 and not ch.isascii():
                    self._emit_plain(ch, source_pos)
                else:
                    self._flush_plain()
                    self._pending = self._open_pending(
                        ch,
                        source_pos,
                        source_pos + 1,
                        self._detector.classify(ch),
                        now,
                    )
                continue
            # Keep whitespace inside a confirmed math expression.  For a plain
            # number, defer whitespace until the next character tells us
            # whether an operator follows (for example ``5 >``).
            if lex_ch.isspace():
                if self._pending.kind == SpanKind.MATH:
                    self._pending_gap.append((ch, source_pos))
                    self._pending.last_at = now
                    continue
                if self._pending.kind == SpanKind.NUMBER:
                    self._pending_gap.append((ch, source_pos))
                    self._pending.last_at = now
                    continue
                self._close_pending()
                self._emit_plain(ch, source_pos)
                continue

            # Resolve a deferred separator *before* applying punctuation or
            # CJK boundary rules.  Otherwise an input such as ``2\n.c`` would
            # append ``.`` to the numeric span while the newline is still in
            # ``_pending_gap``; the resulting raw interval would overlap the
            # separately emitted newline.  Only operators/digits can consume
            # a separator as part of a formula.  Every other next character
            # closes the number first, then is scanned afresh.
            if self._pending_gap and self._pending.kind in (
                SpanKind.NUMBER,
                SpanKind.ORDINAL,
                SpanKind.MATH,
            ):
                # A sentence separator is not enough language evidence for a
                # numeric span at the stream head (``99%，我...``).  Retain
                # it as a deferred boundary so the following Chinese/Latin
                # context can select the correct TN graph instead of forcing
                # an irreversible literal fallback at the comma.
                if (
                    self._pending.kind in (SpanKind.NUMBER, SpanKind.ORDINAL)
                    and ch in "，,。！？!?；;：:"
                    and self._resolve_language(self._pending, None) == "unknown"
                ):
                    self._pending_gap.append((ch, source_pos))
                    self._pending.last_at = now
                    continue
                can_extend_math = (
                    self._pending.kind == SpanKind.NUMBER
                    and (
                        lex_ch in "+*/×÷=<>≤≥!xX-~"
                        or (
                            (ch.isdigit() or lex_ch.isdigit())
                            and re.search(
                                r"[+\-*/×÷=<>≤≥!xX~]\s*$",
                                self._pending.raw,
                            )
                        )
                    )
                ) or (
                    self._pending.kind == SpanKind.MATH
                    and (
                        ch.isdigit()
                        or lex_ch.isdigit()
                        or lex_ch in "+*/×÷=<>≤≥!xX-(~"
                        or (
                            lex_ch in _MATH_CLOSERS
                            and self._pending.raw.count("(")
                            > self._pending.raw.count(")")
                        )
                    )
                )
                if can_extend_math:
                    gap_text = "".join(item[0] for item in self._pending_gap)
                    self._pending.raw += gap_text + ch
                    self._pending.end = source_pos + 1
                    self._pending_gap = []
                    self._pending.last_at = now
                    # Keep a lone hyphen in NUMBER until its right operand (or
                    # an explicit second operator/equality) disambiguates a
                    # range from arithmetic.  Other operators are sufficient
                    # evidence to enter the formula state immediately.
                    if self._pending.kind == SpanKind.NUMBER:
                        self._pending.kind = self._detector.classify(self._pending.raw)
                    else:
                        self._pending.kind = SpanKind.MATH
                    continue
                # The next Latin character is only a lexical boundary, not a
                # language declaration.  Passing ``en`` here used to
                # override stronger Han evidence immediately before a numeric
                # span (``中文99% English``).  Let the resolver compare the
                # nearest local scripts instead; an explicit transport hint
                # is still preserved on the pending span.
                self._close_pending()
                self._append(ch, now, source_pos)
                continue

            # Closing punctuation terminates a semantic number/formula/word;
            # it belongs to the surrounding prose, not to the TN span.  JSON
            # and Markdown handle their own delimiters in the branches above.
            # Keeping URL/email/identifier/version punctuation untouched avoids
            # truncating transport-like tokens that legitimately contain dots.
            if (
                lex_ch in _RIGHT_BOUNDARY_CHARS
                and self._pending.kind
                not in (SpanKind.JSON, SpanKind.MARKDOWN, SpanKind.URL, SpanKind.PHONE)
            ):
                # A link/image may be embedded directly after a word.  Keep
                # its closing parenthesis in the same unresolved run so the
                # projector can validate the complete construct; otherwise
                # the generic right-boundary rule would split at ``)`` and
                # leak the label/URL as literal text.
                if (
                    self._pending.kind is SpanKind.ENGLISH_WORD
                    and "[" in self._pending.raw
                    and (
                        self._pending.raw.count("]")
                        <= self._pending.raw.count("[")
                        or self._pending.raw.count(")")
                        < self._pending.raw.count("(")
                    )
                ):
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    continue
                if (
                    self._pending.kind is SpanKind.MATH
                    and lex_ch in _MATH_CLOSERS
                    and self._pending.raw.count("(") > self._pending.raw.count(")")
                ):
                    # Parentheses are part of a formula while an opening
                    # delimiter remains unmatched (``3*(2+1)=9``).  A
                    # balanced formula reaches this branch and closes before
                    # its surrounding prose delimiter instead.
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    self._pending.kind = self._detector.classify(self._pending.raw)
                    continue
                self._close_pending()
                self._emit_plain(ch, source_pos)
                continue

            # CJK closes an ASCII semantic span.  Combining marks belong to
            # the preceding grapheme and must never be emitted as an
            # independent (usually mis-routed) Chinese character.  Other
            # alphabetic Unicode characters (for example ``é``) remain part
            # of an English/Latin word rather than being treated as CJK just
            # because they are non-ASCII.
            if not lex_ch.isascii() and not _is_emoji(ch):
                if _is_combining(ch):
                    if self._pending.raw:
                        self._pending.raw += ch
                        self._pending.end = source_pos + 1
                        self._pending.last_at = now
                    else:
                        self._emit_plain(ch, source_pos)
                    continue
                # Multiplication/division/comparison symbols are non-ASCII in
                # their common Unicode spellings.  Keep them attached to an
                # open numeric/formula span instead of closing ``4`` before
                # ``×``/``÷`` is seen.
                if self._pending.kind in (SpanKind.NUMBER, SpanKind.MATH) and lex_ch in "×÷≤≥≠<>":
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    self._pending.kind = self._detector.classify(self._pending.raw)
                    continue
                if _is_han(ch):
                    # Let the pending span's own script win.  Passing a hard
                    # Chinese hint here misroutes a Latin word immediately
                    # before CJK (``é中``/``hello中``).  Numeric-only spans can
                    # still use the Han character in ``context_after`` to
                    # select the Chinese graph.
                    self._close_pending()
                    self._current_lang = "zh"
                    self._emit_plain(ch, source_pos)
                    continue
                if ch.isalpha() or lex_ch.isalpha():
                    if self._pending.raw:
                        self._pending.raw += ch
                        self._pending.end = source_pos + 1
                        self._pending.last_at = now
                        self._pending.kind = self._detector.classify(self._pending.raw)
                    else:
                        self._flush_plain()
                        self._pending = self._open_pending(
                            ch,
                            source_pos,
                            source_pos + 1,
                            SpanKind.ENGLISH_WORD,
                            now,
                        )
                    continue
                if self._pending.kind in (SpanKind.NUMBER, SpanKind.ENGLISH_WORD) and re.match(r"^[+\-]?\d", _compatibility_spelling(self._pending.raw)) and lex_ch in "°²³℃μµ":
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    self._pending.kind = self._detector.classify(self._pending.raw)
                    continue
                self._close_pending()
                self._emit_plain(ch, source_pos)
            elif (
                lex_ch == "!"
                and self._pending.kind in (SpanKind.NUMBER, SpanKind.MATH)
            ):
                # Keep ``!`` attached while waiting to see whether the next
                # character completes ``!=``.  A lone factorial-like suffix
                # remains a deterministic literal fallback at close time.
                self._pending.raw += ch
                self._pending.end = source_pos + 1
                self._pending.last_at = now
                self._pending.kind = self._detector.classify(self._pending.raw)
            elif (
                ch == ";"
                and self._pending.raw.startswith("&")
            ):
                self._pending.raw += ch
                self._pending.end = source_pos + 1
                self._pending.last_at = now
                self._close_pending()
            elif (
                lex_ch in "，,。！？!?；;：:"
                and self._pending.kind in (SpanKind.NUMBER, SpanKind.ORDINAL)
                and self._resolve_language(self._pending, None) == "unknown"
            ):
                # Do not close a language-ambiguous number merely because a
                # separator arrived.  The next packet may provide the script
                # evidence needed to normalize it (or the deadline will
                # perform the documented fallback).
                self._pending_gap.append((ch, source_pos))
                self._pending.last_at = now
            elif lex_ch in "。！？；,，.!?;:" and self._pending.kind != SpanKind.URL:
                if lex_ch == "!" and self._pending.kind is SpanKind.ENGLISH_WORD:
                    # Defer the punctuation long enough to distinguish an
                    # adjacent image opener (``![``) from ordinary prose.
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    self._bang_candidate = True
                    continue
                if (
                    lex_ch == "."
                    and self._pending.kind is SpanKind.NUMBER
                    and _at_inline_ordered_list_start(self._raw, self._pending.start)
                    and _compatibility_spelling(self._pending.raw).isdigit()
                ):
                    # Provisional ordered-list marker (``1.``).  This also
                    # covers compact inline enumerations after strong
                    # separators (``：1. ...。2. ...``).  A following digit
                    # turns it back into a decimal/date; whitespace confirms
                    # Markdown and strips the marker.
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    self._pending.kind = SpanKind.MARKDOWN
                    self._ordered_marker_candidate = True
                    continue
                if lex_ch == "." and self._pending.kind in (SpanKind.ENGLISH_WORD, SpanKind.EMAIL, SpanKind.IDENTIFIER, SpanKind.VERSION):
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    continue
                if self._pending.kind in (SpanKind.NUMBER, SpanKind.MATH, SpanKind.ENGLISH_WORD) and lex_ch in ".:" and _compatibility_spelling(self._pending.raw)[-1:].isdigit():
                    self._pending.raw += ch
                    self._pending.end = source_pos + 1
                    self._pending.last_at = now
                    continue
                self._close_pending()
                self._emit_plain(ch, source_pos)
            else:
                self._pending.raw += ch
                self._pending.end = source_pos + 1
                self._pending.last_at = now
                self._pending.kind = self._detector.classify(self._pending.raw)
                if len(self._pending.raw) >= self.config.max_pending_chars:
                    self._limit_fallback = True
                    self._close_pending(reason="max_pending")

    def _close_stable(self, now: float) -> list[TextCommit]:
        # A completed number with a suffix is closed when a CJK boundary arrives;
        # ordinary ASCII words remain pending until whitespace/final.
        return []

    def _flush_pending(self, *, reason: str) -> list[TextCommit]:
        if not self._pending.raw:
            return []
        # A scalar array (``[1]``/``[true]``) cannot be distinguished from a
        # numeric Markdown link label until the stream ends.  With no
        # following ``(``, finalization is the JSON interpretation; promote
        # it before projection so the span kind and diagnostics agree with
        # the emitted readable value.
        if self._bracket_closed_candidate and self._pending.kind is SpanKind.MARKDOWN:
            self._pending.kind = SpanKind.JSON
            self._bracket_closed_candidate = False
        self._close_pending(reason=reason)
        return []

    def _close_pending(self, *, reason: str = "boundary", lang_hint: str | None = None) -> TextCommit | None:
        p = self._pending
        if not p.raw:
            return None
        # Some upstream tokenizers do not separate a word from a following
        # percentage (``English20%``).  The percent suffix is semantically a
        # numeric span, so split it while it is still pending; doing this
        # before either part crosses the commit fence preserves both local
        # language routing and append-only source offsets.  This is purposely
        # narrower than a general letter/digit splitter so model IDs and
        # ordinary alphanumeric words remain literal/identifier spans.
        embedded = (
            _EMBEDDED_NUMERIC_PERCENT.fullmatch(p.raw)
            if p.kind is SpanKind.ENGLISH_WORD and not self._force_literal_next
            else None
        )
        if embedded is not None:
            prefix, numeric = embedded.groups()
            split = p.start + len(prefix)
            original_gap = self._pending_gap
            self._pending_gap = []
            self._pending = self._open_pending(
                prefix,
                p.start,
                split,
                SpanKind.ENGLISH_WORD,
                p.first_at,
                p.language_hint,
                force_literal=p.force_literal,
            )
            first = self._close_pending(reason=reason, lang_hint=lang_hint)
            self._pending = self._open_pending(
                numeric,
                split,
                p.end,
                SpanKind.NUMBER,
                p.last_at,
                p.language_hint,
                force_literal=p.force_literal,
            )
            second = self._close_pending(reason=reason, lang_hint=lang_hint)
            self._pending_gap = original_gap
            return second or first
        force_literal = p.force_literal
        lang = self._resolve_language(p, lang_hint)
        route_lang = lang if lang in ("zh", "en") else ""
        backend_raw = _tn_input(p.raw, p.kind)
        if p.kind is SpanKind.URL:
            backend_raw = _sanitize_url_for_tn(backend_raw)
        mapping: tuple[tuple[int, int], ...] | None = None
        candidate_count = 0
        best_cost = None
        cost_margin = None
        decision_source = "rule"
        semantic_family = self._detector.family(p.kind)
        if route_lang and p.kind in (SpanKind.NUMBER, SpanKind.ORDINAL):
            try:
                from .types import SemioticSpan
                candidates = self._candidate_resolver.resolve(
                    SemioticSpan(
                        span_id=p.span_id,
                        raw_start=p.start,
                        raw_end=p.end,
                        raw_text=p.raw,
                        kind=p.kind,
                        closed=True,
                        family=semantic_family,
                        closure_reason=reason,
                        extendable=False,
                    ),
                    language=LanguageKind(route_lang),
                    backend_text=backend_raw,
                    domain=p.kind,
                )
                candidate_count = candidates.candidate_count
                best_cost = candidates.best_cost
                cost_margin = candidates.cost_margin
                policy_decision = self._commit_policy.decide(
                    family=semantic_family,
                    closed=True,
                    final=reason == "final",
                    candidates=candidates,
                    margin_threshold=self._margin_threshold(semantic_family),
                )
                decision_source = f"{candidates.source}:{policy_decision.reason}"
            except Exception:
                decision_source = "resolver_error"
        order_id = (
            p.kind in (SpanKind.NUMBER, SpanKind.IDENTIFIER)
            and bool(re.search(r"订单号(?:为|是)?\s*$", self._raw[: p.start]))
        )

        # The committer decides only whether a span is safe to close.  The
        # actual TN grammar is delegated to wetext's official stream API.
        # Numeric/ordinal text without language evidence remains literal; a
        # guessed zh route is precisely the ``21st -> 二十一秒t`` failure we
        # must not reintroduce.
        unresolved_sensitive = (
            not route_lang and p.kind in (SpanKind.NUMBER, SpanKind.ORDINAL)
        )
        custom_value = ""
        if p.kind in (SpanKind.MATH, SpanKind.IDENTIFIER, SpanKind.VERSION) and route_lang:
            custom_value = self._safe_fallback(
                backend_raw,
                lang=route_lang,
                kind=p.kind,
            )
        projected_markdown = is_markdown_structured(p.raw) or (
            p.kind is SpanKind.MARKDOWN
            and bool(re.fullmatch(r"[-+>]\s*|\d{1,4}[.)]\s*", p.raw))
        )
        output_span_kind = SpanKind.MARKDOWN if projected_markdown else p.kind
        if p.kind is SpanKind.ID_CARD:
            id_lang = (
                "zh"
                if any(_is_han(ch) for ch in self._raw[: p.start])
                else route_lang or "zh"
            )
            value = self._domain_resolver.identifier(backend_raw, language=id_lang).text
            if not value:
                value = self._safe_fallback(
                    backend_raw,
                    lang=id_lang,
                    kind=SpanKind.ID_CARD,
                )
            kind = CommitKind.NORMALIZED if value != p.raw else CommitKind.FALLBACK
        elif _HTML_ENTITY.fullmatch(p.raw):
            # Decode entities only after the raw span is closed.  The source
            # interval remains the original entity, while the spoken value is
            # its decoded character (``&#x20;`` -> a real space).
            value = self._domain_resolver.entity(p.raw).text
            kind = CommitKind.LITERAL
        elif order_id:
            # An order number is an identifier, not a quantity.  Separate its
            # digits before calling WeText so ``188888`` is read as
            # ``一八八八八八`` rather than ``十八万八千八百八十八``.
            value = self._domain_resolver.identifier(
                backend_raw, language=route_lang or "zh", order_id=True
            ).text
            if not value:
                value = self._safe_fallback(
                    backend_raw,
                    lang=route_lang or "zh",
                    kind=SpanKind.IDENTIFIER,
                )
            kind = CommitKind.NORMALIZED if value != p.raw else CommitKind.FALLBACK
        elif p.kind is SpanKind.PHONE:
            # WeText's generic grammar treats ``+``/``-`` as arithmetic.  The
            # phone adapter first canonicalizes each number into digit tokens,
            # then delegates their verbalization to WeText's language graph.
            value = self._domain_resolver.phone(
                backend_raw, language=route_lang or "zh"
            ).text
            if not value:
                value = self._safe_fallback(
                    backend_raw,
                    lang=route_lang or "zh",
                    kind=SpanKind.PHONE,
                )
            kind = CommitKind.NORMALIZED if value != p.raw else CommitKind.FALLBACK
        elif p.kind is SpanKind.JSON or projected_markdown:
            ordered_marker = re.fullmatch(r"(\d{1,4})[.)]\s*", p.raw)
            inline_ordered = ordered_marker is not None and not _at_line_start(
                self._raw, p.start
            )
            if inline_ordered:
                # Compact enumerations in generated prose (``：1. ...。2. ...``)
                # carry semantic numbering.  Preserve that information while
                # replacing the Markdown dot with a Chinese enumeration pause.
                # True line-start Markdown markers remain formatting-only and
                # continue to be suppressed by ``project_readable``.
                number = ordered_marker.group(1)
                marker_lang = lang if lang in ("zh", "en") else "zh"
                if marker_lang == "zh":
                    # Convert Markdown's formatting dot to the Chinese
                    # enumeration punctuation before TN.  WeText then owns
                    # the numeric verbalization (``1、`` -> ``一、``), just as
                    # it does for the surrounding numeric spans.
                    value = self._domain_resolver.ordered_marker(
                        number, language=marker_lang
                    ).text
                else:
                    value = self._domain_resolver.ordered_marker(
                        number, language=marker_lang
                    ).text
            else:
                value = self._domain_resolver.structured(p.raw).text
            # A late structured suffix is still projected so formatting and
            # transport syntax cannot leak into speech.  Mark it as fallback
            # when it arrived behind a prior fence, making the degradation
            # observable without ever rewriting the old commit.
            kind = CommitKind.FALLBACK if force_literal else CommitKind.LITERAL
        elif force_literal:
            value = p.raw
            kind = CommitKind.FALLBACK
        elif p.kind is SpanKind.MARKDOWN:
            # A lone/unclosed formatting marker is not enough evidence for a
            # projection.  Keep the source literal and surface a fallback
            # rather than asking wetext to pronounce punctuation (or silently
            # deleting an identifier's underscore/asterisk).
            value = p.raw
            kind = CommitKind.FALLBACK
        elif unresolved_sensitive:
            # A bare number/percent has no language by itself and remains
            # literal in mixed mode.  A qualified numeric form (date, money,
            # or unit) still gets a deterministic final fallback so it is not
            # silently left unreadable when no language evidence ever arrives.
            if reason in ("final", "timeout") and _QUALIFIED_NUMBER.search(backend_raw):
                value = self._safe_fallback(
                    backend_raw,
                    lang="zh",
                    kind=p.kind,
                )
            else:
                value = p.raw
            kind = CommitKind.FALLBACK
        elif p.kind in (SpanKind.MATH, SpanKind.IDENTIFIER, SpanKind.VERSION):
            # These domains are outside the generic wetext grammar.  The
            # deterministic fallback is isolated in WetextAdapter and is
            # selected only after language resolution.
            if p.kind is SpanKind.MATH and not route_lang:
                # A formula has a language-independent operator vocabulary in
                # the default policy.  Use the documented Chinese fallback
                # rather than sending an unresolved expression to a random
                # wetext graph; the record still carries UNKNOWN language.
                value = self._domain_resolver.formula(backend_raw, language="zh").text
            else:
                value = (
                    self._domain_resolver.formula(
                        backend_raw, language=route_lang or "zh"
                    ).text
                    if p.kind is SpanKind.MATH
                    else custom_value if route_lang else p.raw
                )
            kind = CommitKind.NORMALIZED if value != p.raw else CommitKind.FALLBACK
        else:
            value = (
                self.adapter.normalize_closed_stream(
                    backend_raw,
                    lang=route_lang,
                    kind=p.kind,
                )
                if route_lang
                else p.raw
            )
            unchanged = value == backend_raw
            if (value is None or unchanged) and p.kind in (
                SpanKind.NUMBER,
                SpanKind.ORDINAL,
            ) and route_lang:
                # Compatibility path for an image with an older stream graph;
                # this still uses wetext's one-shot public API before falling
                # back to deterministic cardinal spelling.
                candidate = self.adapter.normalize(
                    backend_raw,
                    lang=route_lang,
                    kind=p.kind,
                )
                value = candidate if candidate and candidate != backend_raw else None
            if value is None and p.kind in (SpanKind.NUMBER, SpanKind.ORDINAL):
                value = self._safe_fallback(
                    backend_raw,
                    lang=route_lang or "zh",
                    kind=p.kind,
                )
                kind = CommitKind.FALLBACK
            elif unchanged or value is None:
                # Words, punctuation and URLs that need no verbalization are
                # ordinary literal commits, not semantic fallback events.
                value = p.raw
                kind = CommitKind.LITERAL
            else:
                kind = CommitKind.NORMALIZED
            if kind is CommitKind.NORMALIZED and route_lang:
                mapping = self._mapping_for(
                    backend_raw,
                    p.start,
                    value,
                    lang=route_lang,
                    kind=p.kind,
                    raw_end=p.end,
                )
        if value is None:
            value = (
                self._safe_fallback(
                    backend_raw,
                    lang=route_lang or "zh",
                    kind=p.kind,
                )
                if route_lang or p.kind is SpanKind.MATH
                else p.raw
            )
            kind = CommitKind.FALLBACK
        if reason == "max_pending":
            # A hard lexical bound is a safety fallback even if a partial
            # structured projection happened to produce readable characters.
            kind = CommitKind.FALLBACK
        commit = self._make_commit(
            p.raw,
            p.start,
            output_span_kind,
            kind,
            value,
            LanguageKind.EN if lang == "en" else LanguageKind.ZH if lang == "zh" else LanguageKind.UNKNOWN,
            raw_end=p.end or p.start + len(p.raw),
            mapping=mapping,
            semantic_family=semantic_family,
            decision_source=decision_source,
            candidate_count=candidate_count,
            best_cost=best_cost,
            cost_margin=cost_margin,
            closure_reason=reason,
            fallback_reason=reason if kind is CommitKind.FALLBACK else "",
        )
        self.committed_raw_end = max(self.committed_raw_end, p.end or p.start + len(p.raw))
        self.committed_spoken_text += value
        self._pending = _Pending()
        self._bracket_candidate = False
        self._bracket_closed_candidate = False
        self._markdown_line_marker = False
        self._ordered_marker_candidate = False
        self._leading_paren_candidate = False
        self._image_candidate = False
        self._autolink_candidate = False
        self._bang_candidate = False
        self._json_depth = 0
        self._json_in_string = False
        self._json_escape = False
        self._current_lang = lang
        if lang in ("zh", "en"):
            self._language_resolver.observe(p.raw, LanguageKind(lang))
        self._outbox.append(commit)
        if self._pending_gap:
            gap = self._pending_gap
            self._pending_gap = []
            for gap_char, gap_start in gap:
                self._emit_plain(gap_char, gap_start)
        return commit

    def _resolve_language(self, pending: _Pending, lang_hint: str | None) -> str:
        """Resolve a graph from configuration and local script evidence.

        This intentionally contains no connector/dictionary list.  A numeric
        token has no language on its own; it remains unresolved until a CJK or
        Latin neighbour (or an explicit session language) supplies evidence.
        """

        hint = None
        # An explicit hint belongs to the semantic span, not merely to the
        # packet that happened to open it.  Preserve the first hint across
        # later extension packets so ``20`` (hint=en) + ``%`` (no metadata)
        # still resolves through the English graph at close time.
        effective_hint = lang_hint or pending.language_hint or self._active_lang_hint
        if effective_hint in ("zh", "en"):
            hint = LanguageKind(effective_hint)
        resolution = self._language_resolver.resolve(
            pending.raw,
            pending.kind,
            hint=hint,
            context_before=self._raw[: pending.start],
            context_after=self._raw[pending.end : self.raw_cursor],
        )
        # A language graph is selected only for strong evidence.  In
        # particular, a previous span's language is a weak prior and must not
        # route an isolated numeric span across a code-switch boundary.
        if resolution.selected in (LanguageKind.ZH, LanguageKind.EN):
            selected = next(
                (
                    candidate
                    for candidate in resolution.hypotheses
                    if candidate.language is resolution.selected
                ),
                None,
            )
            if selected is not None and selected.confidence >= 0.70:
                return resolution.selected.value
        return "unknown"

    def _fallback_pending(self, *, reason: str) -> TextCommit:
        return self._close_pending(reason=reason)

    def _margin_threshold(self, family: SemanticFamily) -> float | None:
        for name, value in getattr(self.config, "family_margin_thresholds", ()):
            if str(name) == family.value:
                return float(value)
        value = getattr(self.config, "margin_threshold", None)
        return None if value is None else float(value)

    def _safe_fallback(self, text: str, *, lang: str, kind: SpanKind) -> str:
        """Contain adapter failures at the commit boundary.

        A third-party/backend fallback is advisory; a failing fallback must
        never release an exception into the transport or reopen an old fence.
        Returning the raw span is deterministic and keeps the append-only
        contract intact.  ``feed`` marks the resulting record as fallback.
        """

        try:
            return self.adapter.fallback(
                text,
                lang=lang,
                kind=kind,
                policy=self.config.fallback,
            )
        except Exception:  # pragma: no cover - defensive plugin boundary
            logger.warning(
                "text.normalizer.fallback_failed",
                extra={"lang": lang, "kind": kind.value},
                exc_info=True,
            )
            return text

    def _mapping_for(
        self,
        raw: str,
        start: int,
        value: str,
        *,
        lang: str,
        kind: SpanKind,
        raw_end: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Convert optional public WeText mappings to session coordinates.

        Mapping is diagnostic enrichment and never gates a commit.  Different
        wetext releases return either one result object or a one-element list,
        so this adapter accepts both shapes while preserving the historical
        coarse interval fallback.
        """

        source_end = start + len(raw) if raw_end is None else int(raw_end)
        coarse = ((start, source_end),)
        if source_end < start:
            return coarse
        # ``raw`` is the backend spelling, which may be longer than the
        # original source span after compatibility expansion (for example
        # ``℃`` -> ``°C``).  Detailed backend offsets cannot be projected back
        # to Unicode source coordinates without an explicit alignment map.
        # Keep the owner span conservative in the original raw domain.
        if raw_end is not None and source_end - start != len(raw):
            return coarse
        try:
            result = self.adapter.normalize_with_mapping(raw, lang=lang, kind=kind)
        except Exception:
            return coarse
        if isinstance(result, (list, tuple)):
            result = result[0] if result else None
        output = getattr(result, "output_text", None)
        if output is None:
            output = getattr(result, "text", None)
        if output is not None and str(output) != value:
            return coarse
        mappings = getattr(result, "mappings", None)
        if mappings is None:
            mappings = getattr(result, "mapping", None)
        converted: list[tuple[int, int]] = []
        if mappings is not None:
            try:
                for item in mappings:
                    input_start = getattr(item, "input_start", None)
                    input_end = getattr(item, "input_end", None)
                    if isinstance(item, dict):
                        input_start = item.get("input_start", input_start)
                        input_end = item.get("input_end", input_end)
                    if input_start is None or input_end is None:
                        continue
                    converted.append((start + int(input_start), start + int(input_end)))
            except (TypeError, ValueError):
                converted = []
        if converted and all(
            start <= item_start <= item_end <= source_end
            for item_start, item_end in converted
        ):
            return tuple(converted)
        return coarse

    def _emit_plain(self, ch: str, source_start: int | None = None) -> None:
        if not self._plain_buffer:
            self._plain_start = self.committed_raw_end if source_start is None else source_start
        self._plain_buffer += ch
        if source_start is None:
            self.committed_raw_end += len(ch)
        else:
            self.committed_raw_end = max(
                self.committed_raw_end, source_start + len(ch)
            )
        self.committed_spoken_text += ch
        # Plain script-bearing text is useful evidence for a following
        # numeric/symbol span, but punctuation and whitespace do not establish
        # a language.  The resolver itself resets at sentence boundaries.
        self._language_resolver.observe(ch)

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

    def _make_commit(
        self,
        raw: str,
        start: int,
        span_kind: SpanKind,
        commit_kind: CommitKind,
        value: str | None = None,
        language: LanguageKind | None = None,
        raw_end: int | None = None,
        mapping: tuple[tuple[int, int], ...] | None = None,
        semantic_family: SemanticFamily = SemanticFamily.PROSE,
        decision_source: str = "",
        candidate_count: int = 0,
        best_cost: float | None = None,
        cost_margin: float | None = None,
        calibrated_confidence: float | None = None,
        closure_reason: str = "",
        fallback_reason: str = "",
    ) -> TextCommit:
        end = start + len(raw) if raw_end is None else raw_end
        # A fence identifies an append-only commit, not an internal lexer
        # transition.  Increment it exactly once here so every emitted record
        # has a strict total order even when a plain run and a semantic span
        # are closed during the same feed call.
        self.commit_fence += 1
        commit = TextCommit(
            raw_start=start,
            raw_end=end,
            tts_text=raw if value is None else value,
            span_kind=span_kind,
            commit_kind=commit_kind,
            fence=self.commit_fence,
            mapping=mapping if mapping is not None else ((start, end),),
            raw_text=raw,
            language=language or {
                "en": LanguageKind.EN,
                "zh": LanguageKind.ZH,
                "unknown": LanguageKind.UNKNOWN,
            }.get(self._current_lang, LanguageKind.UNKNOWN),
            commit_id=self._next_commit_id,
            span_id=getattr(self._pending, "span_id", 0),
            semantic_family=semantic_family,
            decision_source=decision_source,
            candidate_count=candidate_count,
            best_cost=best_cost,
            cost_margin=cost_margin,
            calibrated_confidence=calibrated_confidence,
            closure_reason=closure_reason,
            fallback_reason=fallback_reason,
        )
        # The frontier is a guard, not a source of provisional text: only a
        # fully formed commit enters it.  Keep a separate source string because
        # frontier lexical units intentionally omit a dangling trailing space.
        self._frontier_source_text += commit.tts_text
        self._causal_frontier.observe_committed(self._frontier_source_text)
        self._next_commit_id += 1
        return commit

    @staticmethod
    def _classify(raw: str) -> SpanKind:
        # Classify compatibility spellings (e.g. ``２０％``) using their
        # canonical form while preserving the original raw span in commits.
        raw = _compatibility_spelling(raw)
        if _ID_CARD.fullmatch(raw):
            return SpanKind.ID_CARD
        if _PHONE.fullmatch(raw):
            return SpanKind.PHONE
        # Transport-like identifiers must win before the generic operator
        # check: query strings contain ``=`` and were previously mislabeled as
        # mathematical expressions (which also changed their fallback).
        if "@" in raw:
            return SpanKind.IDENTIFIER if _IDENTIFIER.fullmatch(raw) else SpanKind.EMAIL
        if "://" in raw or raw.startswith(("www.", "http")):
            return SpanKind.URL
        # Product/pickup codes are identifiers, not signed numbers.  Sending
        # ``B-0109`` through a Chinese TN graph makes it sound like
        # "B negative one hundred nine"; the identifier fallback spells the
        # separator and preserves the digit sequence.
        if _MODEL_CODE.fullmatch(raw):
            return SpanKind.IDENTIFIER
        if re.fullmatch(r"(?:A\$|HKD|[$€￥£¥])\d+(?:\.\d+)?", raw):
            return SpanKind.NUMBER
        if _TIME.fullmatch(raw) or _RANGE.fullmatch(raw):
            return SpanKind.NUMBER
        if re.fullmatch(r"\d{4}[-/.]\d{1,2}(?:[-/.]\d{1,2})?", raw):
            return SpanKind.NUMBER
        # Keep a numeric range ahead of the arithmetic detector.  A partial
        # trailing hyphen is also held as NUMBER until a right operand or an
        # explicit operator/equality proves that it is a formula.
        if _RANGE.fullmatch(raw) or re.fullmatch(
            r"[+\-]?\d+(?:\.\d+)?\s*[-~]\s*\d*$", raw
        ):
            return SpanKind.NUMBER
        # Calendar forms must win over the generic subtraction detector:
        # ``2026-07-28`` is a date, not ``2026 minus 7 minus 28``.
        if _looks_like_math(raw):
            return SpanKind.MATH
        # Calendar forms win over the generic dotted/hyphenated version
        # grammar.  Otherwise ``2026-07-28`` is treated as a product ID.
        if _VERSION.fullmatch(raw):
            return SpanKind.VERSION
        if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?(?:m²|km/h|km|kg|ms|°C|℃|m|mm|cm|[μµ]g/m³)", raw):
            return SpanKind.NUMBER
        if _ORDINAL.fullmatch(raw):
            return SpanKind.ORDINAL
        if _NUMERIC.fullmatch(raw):
            return SpanKind.NUMBER
        return SpanKind.ENGLISH_WORD
