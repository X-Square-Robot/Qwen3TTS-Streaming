from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

from .normalizer_backend import (
    NormalizerBackend,
    WetextNormalizerBackend,
)
from .types import FallbackPolicy, SpanKind
from .wetext_stream import WetextStream

logger = logging.getLogger(__name__)

_MODEL_CODE = re.compile(r"^[A-Za-z]+-(\d+)$")
_TIME = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?([AaPp][Mm])?$")
_RANGE = re.compile(r"^(\d+(?:\.\d+)?)\s*([-~])\s*(\d+(?:\.\d+)?)$")


def _compatibility_spelling(text: str) -> str:
    """Canonicalize compatibility code points without composing graphemes."""

    # Whole-string NFKC would compose ``e`` + a combining acute into ``é``.
    # Per-codepoint mapping handles full-width digits/operators while retaining
    # the grapheme/source boundaries used by streaming diagnostics.
    preserve = set("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉")
    return "".join(
        ch if ch in preserve else unicodedata.normalize("NFKC", ch)
        for ch in text
    )


def _canonical_math_spelling(text: str) -> str:
    """Canonicalize compatibility and multi-character math operators."""

    canonical = _compatibility_spelling(text)
    canonical = re.sub(r"!\s*=", "≠", canonical)
    canonical = re.sub(r">\s*=", "≥", canonical)
    canonical = re.sub(r"<\s*=", "≤", canonical)
    return canonical.replace("==", "=")


class WetextAdapter:
    """Small public-API adapter for the installed wetext runtime.

    TN grammar ownership stays in wetext.  This class only normalizes return
    types, contains runtime failures, and exposes a closed-span stream call;
    session cursors and commit fences remain in the frontend.
    """

    def __init__(self, *, backend: NormalizerBackend | None = None) -> None:
        # Keep one production adapter for both the closed-span path and the
        # shadow prefix oracle.  The older ``WetextAdapter`` name remains as a
        # narrow compatibility facade for existing callers and owns only the
        # deterministic fallback functions below.
        self.backend: NormalizerBackend = backend or WetextNormalizerBackend(
            eager=True
        )

    @property
    def available_languages(self) -> tuple[str, ...]:
        values = getattr(self.backend, "available_languages", ())
        return tuple(getattr(value, "value", value) for value in values)

    def normalize(self, text: str, *, lang: str, kind: SpanKind) -> Optional[str]:
        if not text:
            return ""
        # Canonicalize compatibility spellings before handing a semantic span
        # to wetext.  The committer retains the original raw text/offsets, so
        # this does not alter source-coordinate diagnostics.
        text = _compatibility_spelling(text)
        if not any(ch.isalnum() for ch in text):
            return None
        # A hyphenated pickup/model code is not a mathematical negative
        # number.  Some wetext graphs interpret ``B-0109`` as ``B`` followed
        # by a signed number, so route it to the deterministic code fallback.
        if lang == "zh" and kind == SpanKind.ENGLISH_WORD and re.fullmatch(r"[A-Za-z]+-\d+", text):
            return None
        try:
            result = self.backend.normalize_closed(
                text,
                language=lang,
                domain=kind,
            )
            value = getattr(result, "output_text", None)
            if value is None:
                value = getattr(result, "text", result)
            # An unchanged value means the graph did not provide a useful
            # verbalization.  Returning None lets the caller take the
            # deterministic, observable fallback path (units and formulae in
            # particular are not covered by every wetext graph version).
            return value if isinstance(value, str) and value and value != text else None
        except Exception:
            logger.warning(
                "text.normalizer.failed",
                extra={"lang": lang, "kind": kind.value},
            )
            return None

    def normalize_closed_stream(
        self,
        text: str,
        *,
        lang: str,
        kind: SpanKind,
    ) -> Optional[str]:
        """Run the public closed-span normalizer on a homogeneous span.

        The backend may use WeText's one-shot ``Normalizer`` API (the current
        production adapter) while its separate prefix wrapper exposes
        ``StreamNormalizer`` snapshots for shadow/oracle use.  This method
        returns only a closed result and converts runtime failures to ``None``
        so the committer can apply its configured, observable fallback policy.
        """

        if not text:
            return ""
        text = _compatibility_spelling(text)
        try:
            result = self.backend.normalize_closed(
                text,
                language=lang,
                domain=kind,
            )
            value = getattr(result, "output_text", None)
            if value is None:
                value = getattr(result, "text", result)
        except Exception as exc:  # third-party normalizers may raise arbitrary errors
            logger.warning(
                "text.normalizer.stream_failed",
                extra={
                    "lang": lang,
                    "kind": kind.value,
                    "error_code": getattr(exc, "code", type(exc).__name__),
                },
            )
            return None
        return value if isinstance(value, str) and value else None

    def normalize_phone(self, text: str, *, lang: str) -> Optional[str]:
        """Normalize phone digits without exposing ``+``/``-`` as math.

        WeText's ordinary grammar reads ``+86-...`` as signed arithmetic.  A
        phone span is therefore canonicalized into individually separated
        digits (and a spoken country-code prefix) before calling the same
        public normalizer API.  Slash-separated numbers remain distinct with
        a short pause between them.
        """

        parts = re.split(r"\s*/\s*", text.strip())
        rendered: list[str] = []
        for part in parts:
            digits = re.findall(r"\d", part)
            if not digits:
                return None
            has_country_prefix = part.lstrip().startswith(("+", "(+"))
            prefix = "加 " if has_country_prefix and lang == "zh" else (
                "plus " if part.lstrip().startswith("+") else ""
            )
            prepared = prefix + " ".join(digits)
            value = self.normalize_closed_stream(
                prepared,
                lang=lang,
                kind=SpanKind.PHONE,
            )
            if not value:
                return None
            rendered.append("".join(value.split()))
        return "，".join(rendered)

    def normalize_digit_sequence(self, text: str, *, lang: str) -> Optional[str]:
        """Normalize an identifier as individual digits through WeText."""

        digits = re.findall(r"\d", text)
        if not digits:
            return None
        prepared = " ".join(digits)
        value = self.normalize_closed_stream(
            prepared,
            lang=lang,
            kind=SpanKind.IDENTIFIER,
        )
        if not value:
            return None
        return "".join(value.split())

    def normalize_id_card(self, text: str, *, lang: str) -> Optional[str]:
        """Normalize an 18-character ID card as a digit sequence."""

        value = str(text or "")
        if not re.fullmatch(r"\d{17}[0-9Xx]", value):
            return None
        prepared = " ".join(value[:-1]) + " " + value[-1].upper()
        normalized = self.normalize_closed_stream(
            prepared,
            lang=lang,
            kind=SpanKind.ID_CARD,
        )
        return "".join(normalized.split()) if normalized else None

    def stream(self, *, lang: str) -> WetextStream:
        """Create an isolated stream for shadow/oracle evaluation.

        The caller owns the returned object and must not share it across
        sessions.  It is exposed separately from ``normalize`` so a snapshot
        cannot accidentally enter the production commit path.
        """

        return WetextStream(lang)

    def normalize_candidates(self, text: str, *, lang: str, kind: SpanKind) -> list[str]:
        """Return n-best candidates when the installed wetext exposes them.

        Candidate generation is an oracle/audit facility; the committer still
        chooses one append-only result and never consumes StreamNormalizer
        snapshots.
        """
        try:
            values = self.backend.candidates(
                text,
                language=lang,
                domain=kind,
                nbest=16,
            )
            return [
                str(getattr(item, "spoken_text", getattr(item, "text", item)))
                for item in values
                if getattr(item, "spoken_text", getattr(item, "text", item))
            ]
        except Exception:
            logger.warning(
                "text.normalizer.candidates_failed",
                extra={"lang": lang, "kind": kind.value},
            )
            return []

    def normalize_with_mapping(self, text: str, *, lang: str, kind: SpanKind):
        try:
            values = self.backend.normalize_with_mapping(
                text,
                language=lang,
                domain=kind,
                nbest=1,
            )
            return values[0] if values else None
        except Exception:
            logger.warning(
                "text.normalizer.mapping_failed",
                extra={"lang": lang, "kind": kind.value},
            )
            return None

    def fallback(self, text: str, *, lang: str, kind: SpanKind, policy: FallbackPolicy) -> str:
        # Keep deterministic fallback behavior aligned with wetext for
        # full-width digits/operators and other Unicode compatibility forms.
        text = _compatibility_spelling(text)
        if policy == FallbackPolicy.CARDINAL_OR_LITERAL and kind == SpanKind.VERSION:
            return _version_fallback(text, lang=lang)
        if policy == FallbackPolicy.CARDINAL_OR_LITERAL and kind == SpanKind.IDENTIFIER:
            return _identifier_fallback(text, lang=lang)
        if kind == SpanKind.PHONE:
            return _phone_fallback(text, lang=lang)
        if kind == SpanKind.ID_CARD:
            return _id_card_fallback(text, lang=lang)
        if lang == "zh" and kind == SpanKind.ENGLISH_WORD:
            particulate = re.fullmatch(r"PM(\d+(?:\.\d+)?)", text, re.I)
            if particulate:
                return "PM" + _zh_cardinal(particulate.group(1))
            code = re.fullmatch(r"([A-Za-z]+)-(\d+)", text)
            if code:
                return code.group(1) + "杠" + _zh_digit_sequence(code.group(2))
        if policy == FallbackPolicy.CARDINAL_OR_LITERAL and kind in (SpanKind.NUMBER, SpanKind.ORDINAL):
            structured = _structured_number_fallback(text, lang=lang)
            if structured:
                return structured
            if text.endswith("%"):
                base = text[:-1]
                value = _zh_cardinal(base) if lang == "zh" else _en_cardinal(base)
                if value:
                    return ("百分之" + value) if lang == "zh" else (value + " percent")
            value = _zh_cardinal(text) if lang == "zh" else _en_cardinal(text)
            if value:
                return value
        if kind == SpanKind.MATH:
            value = _math_fallback(text, lang=lang)
            if value:
                return value
        return text


_ZH_DIGITS = "零一二三四五六七八九"
_EN_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _zh_digit_sequence(text: str) -> str:
    return "".join(_ZH_DIGITS[int(d)] for d in text)


def _en_digit_sequence(text: str) -> str:
    return " ".join(_EN_DIGITS[int(d)] for d in text)


def _identifier_fallback(text: str, *, lang: str) -> str:
    model_code = _MODEL_CODE.fullmatch(text)
    if model_code and lang == "en":
        # Product/pickup identifiers conventionally pronounce zero as "oh";
        # retain the visible hyphen as a pause marker without letting the
        # generic TN graph reinterpret the code as a negative number.
        prefix = text[: model_code.start(1)]
        digits = " ".join(
            "oh" if digit == "0" else _EN_DIGITS[int(digit)]
            for digit in model_code.group(1)
        )
        return prefix + digits
    if lang == "zh":
        separators = {"@": "艾特", "_": "下划线", "-": "杠", ".": "点"}
        parts: list[str] = []
        for token in re.findall(r"[A-Za-z]+|\d+|[^A-Za-z0-9]", text):
            if token.isdigit():
                parts.append(_zh_digit_sequence(token))
            elif token in separators:
                parts.append(separators[token])
            else:
                parts.append(token)
        return "".join(parts)
    separators = {"@": "at", "_": "underscore", "-": "hyphen", ".": "dot"}
    parts = []
    for token in re.findall(r"[A-Za-z]+|\d+|[^A-Za-z0-9]", text):
        if token.isdigit():
            parts.append(_en_digit_sequence(token))
        else:
            parts.append(separators.get(token, token))
    return " ".join(part for part in parts if part)


def _phone_fallback(text: str, *, lang: str) -> str:
    """Speak a recognized phone span digit-by-digit when WeText is absent."""

    parts: list[str] = []
    for part in re.split(r"\s*/\s*", text.strip()):
        digits = re.findall(r"\d", part)
        if not digits:
            continue
        if lang == "zh":
            value = "".join(_ZH_DIGITS[int(d)] for d in digits)
            if part.lstrip().startswith(("+", "(+")):
                value = "加" + value
        else:
            value = " ".join(_EN_DIGITS[int(d)] for d in digits)
            if part.lstrip().startswith(("+", "(+")):
                value = "plus " + value
        parts.append(value)
    return "，".join(parts) or text


def _id_card_fallback(text: str, *, lang: str) -> str:
    value = str(text or "")
    if lang == "zh":
        return "".join(_ZH_DIGITS[int(ch)] if ch.isdigit() else "X" for ch in value.upper())
    return " ".join((_EN_DIGITS[int(ch)] if ch.isdigit() else "X") for ch in value.upper())


def _version_fallback(text: str, *, lang: str) -> str:
    return _identifier_fallback(text, lang=lang)


def _zh_cardinal(text: str) -> str:
    m = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text)
    if not m:
        return ""
    sign = "负" if text.startswith("-") else ("正" if text.startswith("+") else "")
    body = text.lstrip("+-")
    if "." in body:
        whole, frac = body.split(".", 1)
        return sign + _zh_int(int(whole)) + "点" + "".join(_ZH_DIGITS[int(c)] for c in frac)
    return sign + _zh_int(int(body))


def _zh_int(n: int) -> str:
    if n == 0:
        return "零"
    if len(str(n)) > 5:
        return "".join(_ZH_DIGITS[int(d)] for d in str(n))
    units = ("", "十", "百", "千", "万")
    out = ""
    for i, d in enumerate(reversed(str(n))):
        d = int(d)
        if d:
            out = _ZH_DIGITS[d] + units[i] + out
        elif out and not out.startswith("零"):
            out = "零" + out
    out = out.rstrip("零").replace("一十", "十")
    return out


def _en_cardinal(text: str) -> str:
    # Keep this deterministic and dependency-free; wetext handles rich English.
    ordinal = re.fullmatch(r"(\d+)(st|nd|rd|th)", text, re.I)
    if ordinal:
        base = _en_cardinal(ordinal.group(1))
        if not base:
            return ""
        irregular = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
        words = base.split()
        last = words[-1]
        words[-1] = irregular.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
        return " ".join(words)
    sign = ""
    if text.startswith(("+", "-")):
        sign = "positive " if text[0] == "+" else "negative "
        text = text[1:]
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    if "." in text and re.fullmatch(r"\d+\.\d+", text):
        whole, frac = text.split(".", 1)
        value = _en_cardinal(whole) + " point " + " ".join(_en_cardinal(d) for d in frac)
        return sign + value + (" percent" if percent else "")
    if not text.isdigit():
        return ""
    n = int(text)
    value = _en_int(n)
    return sign + value + (" percent" if percent else "")


def _en_int(n: int) -> str:
    ones = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    teens = ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    if n < 10:
        return ones[n]
    if n < 20:
        return teens[n - 10]
    if n < 100:
        return tens[n // 10] + (" " + ones[n % 10] if n % 10 else "")
    if n < 1000:
        return ones[n // 100] + " hundred" + (" " + _en_int(n % 100) if n % 100 else "")
    for scale, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")):
        if n >= scale:
            remainder = n % scale
            return _en_int(n // scale) + " " + name + (" " + _en_int(remainder) if remainder else "")
    return str(n)


def _math_fallback(text: str, *, lang: str = "zh") -> str:
    """Verbalize a small arithmetic expression without evaluating it."""
    text = _canonical_math_spelling(text)
    operand = r"[+\-]?\d+(?:\.\d+)?"
    if lang == "en":
        operators = {
            "*": " times ",
            "×": " times ",
            "x": " times ",
            "X": " times ",
            "/": " divided by ",
            "÷": " divided by ",
            "+": " plus ",
            "-": " minus ",
            "=": " equals ",
            ">": " greater than ",
            "<": " less than ",
            "≥": " greater than or equal to ",
            "≤": " less than or equal to ",
            "^": " to the power of ",
            "≠": " not equal to ",
        }
    else:
        operators = {
            "*": "乘",
            "×": "乘",
            "x": "乘",
            "X": "乘",
            "/": "除以",
            "÷": "除以",
            "+": "加",
            "-": "减",
            "=": "等于",
            ">": "大于",
            "<": "小于",
            "≥": "大于等于",
            "≤": "小于等于",
            "^": "的幂",
            "≠": "不等于",
        }
    # Scan operands/operators instead of splitting on ``-``.  A minus can be
    # either a binary subtraction operator or a unary sign on the following
    # operand (``3*-2``), and a regular ``re.split`` cannot distinguish those
    # forms without losing the sign.  Parentheses are retained as explicit
    # spoken delimiters; they are never evaluated by this fallback.
    expression = text.strip()
    parts: list[str] = []
    position = 0
    expect_operand = True
    operator_count = 0
    paren_depth = 0
    while position < len(expression):
        while position < len(expression) and expression[position].isspace():
            position += 1
        if position >= len(expression):
            break
        char = expression[position]
        if expect_operand:
            if char == "(":
                parts.append(char)
                paren_depth += 1
                position += 1
                continue
            operand_match = re.match(operand, expression[position:])
            if operand_match is None:
                return ""
            parts.append(operand_match.group(0))
            position += operand_match.end()
            expect_operand = False
            continue
        if char == ")":
            if paren_depth <= 0:
                return ""
            parts.append(char)
            paren_depth -= 1
            position += 1
            continue
        if char not in operators:
            return ""
        parts.append(char)
        operator_count += 1
        position += 1
        expect_operand = True
    if not parts or expect_operand or paren_depth or operator_count == 0:
        return ""
    out: list[str] = []
    for part in parts:
        if part in operators:
            out.append(operators[part])
        elif part == "(":
            out.append(" left parenthesis " if lang == "en" else "左括号")
        elif part == ")":
            out.append(" right parenthesis " if lang == "en" else "右括号")
        elif re.fullmatch(operand, part):
            out.append(_zh_cardinal(part) if lang != "en" else _en_cardinal(part))
        else:
            return ""
    return ("".join(out) if lang != "en" else "".join(out).strip())


def _structured_number_fallback(text: str, *, lang: str) -> str:
    currency = re.fullmatch(r"(A\$|HKD|[$€￥£¥])([+\-]?\d+(?:\.\d+)?)", text)
    if currency:
        if lang == "zh":
            names = {"A$": "澳元", "HKD": "港币", "$": "美元", "€": "欧元", "￥": "人民币", "¥": "人民币", "£": "英镑"}
            return _zh_cardinal(currency.group(2)) + names[currency.group(1)]
        names = {"A$": "Australian dollars", "HKD": "Hong Kong dollars", "$": "dollars", "€": "euros", "￥": "Chinese yuan", "¥": "Chinese yuan", "£": "pounds"}
        value = _en_cardinal(currency.group(2))
        return value + " " + names[currency.group(1)] if value else ""
    unit = re.fullmatch(r"([+\-]?\d+(?:\.\d+)?)(m²|km/h|km|kg|ms|°C|℃|m|mm|cm|[μµ]g/m³|%)", text)
    if unit and unit.group(2) != "%":
        if lang == "zh":
            names = {"m²": "平方米", "km/h": "千米每小时", "km": "千米", "kg": "千克", "ms": "毫秒", "°C": "摄氏度", "℃": "摄氏度", "m": "米", "mm": "毫米", "cm": "厘米", "μg/m³": "微克每立方米", "µg/m³": "微克每立方米"}
            return _zh_cardinal(unit.group(1)) + names[unit.group(2)]
        names = {"m²": "square meters", "km/h": "kilometers per hour", "km": "kilometers", "kg": "kilograms", "ms": "milliseconds", "°C": "degrees Celsius", "℃": "degrees Celsius", "m": "meters", "mm": "millimeters", "cm": "centimeters", "μg/m³": "micrograms per cubic meter", "µg/m³": "micrograms per cubic meter"}
        value = _en_cardinal(unit.group(1))
        return value + " " + names[unit.group(2)] if value else ""
    date = re.fullmatch(r"(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?", text)
    if date:
        year, month, day = date.groups()
        if lang == "zh":
            result = "".join(_ZH_DIGITS[int(d)] for d in year) + "年" + _zh_cardinal(month) + "月"
            return result + (_zh_cardinal(day) + "日" if day else "")
        months = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
        result = months[int(month) - 1] + (" " + _en_cardinal(day) + "," if day else "")
        return result + " " + _en_cardinal(year)
    clock = _TIME.fullmatch(text)
    if clock:
        hour, minute, second, meridiem = clock.groups()
        if lang == "zh":
            value = _zh_cardinal(hour) + "点" + _zh_cardinal(minute) + "分"
            if second is not None:
                value += _zh_cardinal(second) + "秒"
            if meridiem:
                value = ("上午" if meridiem.lower() == "am" else "下午") + value
            return value
        value = _en_cardinal(hour) + " " + _en_cardinal(minute)
        if second is not None:
            value += " " + _en_cardinal(second)
        if meridiem:
            value += " " + meridiem.upper()
        return value
    value_range = _RANGE.fullmatch(text)
    if value_range:
        left, marker, right = value_range.groups()
        if lang == "zh":
            return _zh_cardinal(left) + ("到" if marker == "~" else "至") + _zh_cardinal(right)
        return _en_cardinal(left) + (" to " if marker == "-" else " tilde ") + _en_cardinal(right)
    return ""
