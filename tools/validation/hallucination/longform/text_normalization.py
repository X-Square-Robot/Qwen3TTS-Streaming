"""Transcript and numeral normalization for long-form ASR diagnostics."""

from __future__ import annotations

import importlib
import unicodedata


_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000}
_CHINESE_NUMBER_CHARS = frozenset(
    set(_CHINESE_DIGITS) | set(_SMALL_UNITS) | set(_LARGE_UNITS) | {"负"}
)


def _fallback_chinese_integer(token: str) -> str:
    """Conservative integer fallback used only when cn2an is unavailable."""

    negative = token.startswith("负")
    digits = token[1:] if negative else token
    if not digits:
        return token
    if not any(
        character in _SMALL_UNITS or character in _LARGE_UNITS
        for character in digits
    ):
        if all(character in _CHINESE_DIGITS for character in digits):
            converted = "".join(
                str(_CHINESE_DIGITS[character]) for character in digits
            )
            return f"-{converted}" if negative else converted
        return token

    total = section = number = 0
    try:
        for character in digits:
            if character in _CHINESE_DIGITS:
                number = _CHINESE_DIGITS[character]
            elif character in _SMALL_UNITS:
                unit = _SMALL_UNITS[character]
                section += (number or 1) * unit
                number = 0
            elif character in _LARGE_UNITS:
                section += number
                number = 0
                unit = _LARGE_UNITS[character]
                if unit == 10_000:
                    total += section * unit
                else:
                    total = (total + section) * unit
                section = 0
            else:
                return token
    except (KeyError, ValueError, OverflowError):
        return token
    value = total + section + number
    return str(-value if negative else value)


def _fallback_cn2an_transform(text: str) -> str:
    pieces: list[str] = []
    token: list[str] = []

    def flush() -> None:
        if token:
            pieces.append(_fallback_chinese_integer("".join(token)))
            token.clear()

    for character in text:
        if character in _CHINESE_NUMBER_CHARS:
            token.append(character)
        else:
            flush()
            pieces.append(character)
    flush()
    return "".join(pieces)


def canonicalize_numbers(text: str, *, strict: bool = False) -> str:
    """Normalize Arabic/Chinese numerals to a comparable Arabic representation.

    ``cn2an.transform(..., "cn2an")`` is the authoritative implementation in the
    evaluation environment. Import and conversion failures are non-fatal by
    default so an unavailable optional ASR environment cannot break artifact
    inspection. A small integer-only fallback covers ordinary transcripts;
    ``strict=True`` instead raises when cn2an is unavailable or rejects input.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    try:
        cn2an_module = importlib.import_module("cn2an")
    except ImportError as exc:
        if strict:
            raise RuntimeError(
                "cn2an is required for strict numeric normalization"
            ) from exc
        return _fallback_cn2an_transform(normalized)

    try:
        transformed = cn2an_module.transform(normalized, "cn2an")
    except Exception as exc:  # cn2an uses multiple exception types across releases.
        if strict:
            raise ValueError(
                "cn2an failed to canonicalize transcript numbers"
            ) from exc
        return _fallback_cn2an_transform(normalized)
    return str(transformed)


def normalize_transcript(text: str, *, canonicalize_numeric: bool = True) -> str:
    """Return lowercase Unicode alphanumerics suitable for character alignment."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    source = canonicalize_numbers(text) if canonicalize_numeric else text
    return "".join(character.casefold() for character in source if character.isalnum())


__all__ = ["canonicalize_numbers", "normalize_transcript"]
