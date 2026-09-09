"""Typed domain routes used by the streaming TN controller."""
from __future__ import annotations

import html
from dataclasses import dataclass

from .types import FallbackPolicy, SpanKind
from .wetext_backend import WetextAdapter
from .projector import project_readable


@dataclass(frozen=True)
class DomainResult:
    text: str
    normalized: bool
    source: str


class DomainResolver:
    """Own deterministic domain routes before generic WeText normalization."""

    def __init__(self, adapter: WetextAdapter):
        self.adapter = adapter

    def identifier(self, text: str, *, language: str, order_id: bool = False) -> DomainResult:
        if order_id:
            value = self.adapter.normalize_digit_sequence(text, lang=language or "zh")
            return DomainResult(value or text, bool(value and value != text), "order_id")
        value = self.adapter.normalize_id_card(text, lang=language or "zh")
        return DomainResult(value or text, bool(value and value != text), "id_card")

    def phone(self, text: str, *, language: str) -> DomainResult:
        value = self.adapter.normalize_phone(text, lang=language or "zh")
        return DomainResult(value or text, bool(value and value != text), "phone")

    @staticmethod
    def entity(text: str) -> DomainResult:
        value = html.unescape(text)
        return DomainResult(value, value != text, "html_entity")

    @staticmethod
    def structured(text: str) -> DomainResult:
        value = project_readable(text)
        return DomainResult(value, value != text, "projection")

    def ordered_marker(self, number: str, *, language: str) -> DomainResult:
        marker = f"{number}、" if language != "en" else f"{number}, "
        value = self.adapter.normalize_closed_stream(
            marker, lang=language or "zh", kind=SpanKind.NUMBER
        )
        if value and value != marker:
            return DomainResult(value, True, "ordered_marker")
        spoken = self.adapter.fallback(
            number, lang=language or "zh", kind=SpanKind.NUMBER,
            policy=FallbackPolicy.CARDINAL_OR_LITERAL,
        )
        return DomainResult(
            f"{spoken}{'、' if language != 'en' else ', '}", True, "ordered_marker_fallback"
        )

    def formula(self, text: str, *, language: str) -> DomainResult:
        value = self.adapter.fallback(
            text, lang=language or "zh", kind=SpanKind.MATH,
            policy=FallbackPolicy.CARDINAL_OR_LITERAL,
        )
        return DomainResult(value or text, bool(value and value != text), "formula")
