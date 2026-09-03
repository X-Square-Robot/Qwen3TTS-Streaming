from __future__ import annotations

import logging
import re
from typing import Optional

from .types import FallbackPolicy, SpanKind

logger = logging.getLogger(__name__)


class WetextAdapter:
    """Small public-API adapter.  wetext is optional and never owns commit state."""

    def __init__(self) -> None:
        self._normalizers: dict[str, object] = {}
        try:
            from wetext import Normalizer  # type: ignore

            for lang in ("zh", "en"):
                try:
                    self._normalizers[lang] = Normalizer(lang=lang, operator="tn")
                except Exception:
                    logger.exception("text.normalizer.init_failed", extra={"lang": lang})
        except Exception:
            logger.info("wetext unavailable; using literal/cardinal fallback")

    def normalize(self, text: str, *, lang: str, kind: SpanKind) -> Optional[str]:
        if not text:
            return ""
        if not any(ch.isalnum() for ch in text):
            return None
        # A hyphenated pickup/model code is not a mathematical negative
        # number.  Some wetext graphs interpret ``B-0109`` as ``B`` followed
        # by a signed number, so route it to the deterministic code fallback.
        if lang == "zh" and kind == SpanKind.ENGLISH_WORD and re.fullmatch(r"[A-Za-z]+-\d+", text):
            return None
        normalizer = self._normalizers.get(lang)
        if normalizer is None:
            return None
        try:
            value = normalizer.normalize(text)  # public API only
            # An unchanged value means the graph did not provide a useful
            # verbalization.  Returning None lets the caller take the
            # deterministic, observable fallback path (units and formulae in
            # particular are not covered by every wetext graph version).
            return value if isinstance(value, str) and value and value != text else None
        except Exception:
            logger.exception("text.normalizer.failed", extra={"lang": lang, "kind": kind.value})
            return None

    def normalize_candidates(self, text: str, *, lang: str, kind: SpanKind) -> list[str]:
        """Return n-best candidates when the installed wetext exposes them.

        Candidate generation is an oracle/audit facility; the committer still
        chooses one append-only result and never consumes StreamNormalizer
        snapshots.
        """
        normalizer = self._normalizers.get(lang)
        method = getattr(normalizer, "normalize_candidates", None)
        if not callable(method):
            value = self.normalize(text, lang=lang, kind=kind)
            return [value] if value else []
        try:
            values = method(text)
            candidates: list[str] = []
            for item in values:
                candidate = getattr(item, "text", item)
                if candidate:
                    candidates.append(str(candidate))
            return candidates
        except Exception:
            logger.exception("text.normalizer.candidates_failed", extra={"lang": lang, "kind": kind.value})
            return []

    def normalize_with_mapping(self, text: str, *, lang: str, kind: SpanKind):
        normalizer = self._normalizers.get(lang)
        method = getattr(normalizer, "normalize_with_mapping", None)
        if not callable(method):
            return None
        try:
            return method(text)
        except Exception:
            logger.exception("text.normalizer.mapping_failed", extra={"lang": lang, "kind": kind.value})
            return None

    def fallback(self, text: str, *, lang: str, kind: SpanKind, policy: FallbackPolicy) -> str:
        if policy == FallbackPolicy.CARDINAL_OR_LITERAL and kind == SpanKind.VERSION:
            return _version_fallback(text, lang=lang)
        if policy == FallbackPolicy.CARDINAL_OR_LITERAL and kind == SpanKind.IDENTIFIER:
            return _identifier_fallback(text, lang=lang)
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
            value = _math_fallback(text)
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


def _math_fallback(text: str) -> str:
    """Verbalize a small arithmetic expression without evaluating it."""
    if not re.fullmatch(r"\d+(?:\.\d+)?(?:\s*[+*/×÷=<>≤≥-]\s*\d+(?:\.\d+)?)+", text):
        return ""
    operators = {"*": "乘", "×": "乘", "/": "除以", "÷": "除以", "+": "加", "-": "减", "=": "等于", ">": "大于", "<": "小于", "≥": "大于等于", "≤": "小于等于"}
    parts = re.split(r"\s*([+*/×÷=<>≤≥-])\s*", text)
    out: list[str] = []
    for part in parts:
        if part in operators:
            out.append(operators[part])
        elif re.fullmatch(r"\d+(?:\.\d+)?", part):
            out.append(_zh_cardinal(part))
        else:
            return ""
    return "".join(out)


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
    return ""
