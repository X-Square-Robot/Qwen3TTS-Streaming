"""Model-owned spoken-text to native-cursor label conversion.

The streaming TN remains the only owner of spoken-form semantics.  This
module only converts an already committed spoken string into the label ids
declared by a native-cursor checkpoint; raw text, spans, and normalization do
not enter this API.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class NativeCursorLabelizerError(ValueError):
    """Raised when a cursor vocabulary cannot represent committed text."""


def vocab_fingerprint(vocab: Mapping[str, int]) -> str:
    ordered = sorted((str(label), int(index)) for label, index in vocab.items())
    payload = json.dumps(
        ordered,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_vocab(vocab: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(vocab, Mapping) or not vocab:
        raise NativeCursorLabelizerError("cursor checkpoint vocab must be a non-empty mapping")
    result: dict[str, int] = {}
    for raw_label, raw_index in vocab.items():
        if not isinstance(raw_label, str) or not raw_label:
            raise NativeCursorLabelizerError("cursor vocab labels must be non-empty strings")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise NativeCursorLabelizerError(
                f"cursor vocab id for {raw_label!r} must be an integer"
            )
        if raw_index <= 0:
            raise NativeCursorLabelizerError(
                f"cursor vocab id for {raw_label!r} must be positive; 0 is blank"
            )
        if raw_label in result:
            raise NativeCursorLabelizerError(f"duplicate cursor vocab label {raw_label!r}")
        result[raw_label] = raw_index
    expected = set(range(1, len(result) + 1))
    actual = set(result.values())
    if actual != expected:
        raise NativeCursorLabelizerError(
            "cursor vocab ids must be a contiguous 1..N range"
        )
    return result


def _is_cjk(ch: str) -> bool:
    codepoint = ord(ch)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


@dataclass(frozen=True, slots=True)
class NativeCursorLabelizer:
    """Convert committed spoken text using one checkpoint-owned vocabulary."""

    vocab: Mapping[str, int]

    def __post_init__(self) -> None:
        validated = _validate_vocab(self.vocab)
        object.__setattr__(self, "vocab", validated)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def vocab_sha256(self) -> str:
        return vocab_fingerprint(self.vocab)

    def __call__(self, spoken_text: str) -> tuple[int, ...]:
        if not isinstance(spoken_text, str):
            raise NativeCursorLabelizerError("spoken_text must be a string")
        if not spoken_text:
            return ()

        try:
            from pypinyin import Style, lazy_pinyin
        except ImportError as exc:
            raise NativeCursorLabelizerError(
                "pypinyin is required for the native cursor labelizer"
            ) from exc

        labels: list[int] = []

        def append_label(spoken_label: str) -> None:
            index = self.vocab.get(spoken_label)
            if index is None:
                raise NativeCursorLabelizerError(
                    f"spoken label {spoken_label!r} is missing from cursor vocabulary"
                )
            labels.append(index)

        index = 0
        while index < len(spoken_text):
            if _is_cjk(spoken_text[index]):
                start = index
                while index < len(spoken_text) and _is_cjk(spoken_text[index]):
                    index += 1
                for spoken_label in lazy_pinyin(
                    spoken_text[start:index], style=Style.NORMAL
                ):
                    append_label(spoken_label)
                continue

            ch = spoken_text[index]
            index += 1
            if ch.isascii() and ch.isalpha():
                append_label(f"en:{ch.lower()}")
            elif ch.isspace() or unicodedata.category(ch).startswith("P"):
                continue
            else:
                raise NativeCursorLabelizerError(
                    f"unsupported spoken character for cursor labels: {ch!r}"
                )
        return tuple(labels)


def _load_checkpoint(checkpoint_or_path: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(checkpoint_or_path, Mapping):
        return checkpoint_or_path
    try:
        import torch
    except ImportError as exc:
        raise NativeCursorLabelizerError(
            "torch is required to load a native cursor checkpoint"
        ) from exc
    try:
        checkpoint = torch.load(
            str(checkpoint_or_path),
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise NativeCursorLabelizerError(
            f"could not load native cursor checkpoint: {checkpoint_or_path}"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise NativeCursorLabelizerError("native cursor checkpoint must contain a mapping")
    return checkpoint


def load_native_cursor_labelizer(
    checkpoint_or_path: str | Path | Mapping[str, Any],
    *,
    expected_vocab_sha256: str | None = None,
) -> NativeCursorLabelizer:
    """Build a labelizer from the model-owned native cursor checkpoint."""

    checkpoint = _load_checkpoint(checkpoint_or_path)
    try:
        labelizer = NativeCursorLabelizer(checkpoint["vocab"])
    except KeyError as exc:
        raise NativeCursorLabelizerError(
            "native cursor checkpoint is missing vocab"
        ) from exc
    expected = str(expected_vocab_sha256 or "").strip().lower()
    if expected and labelizer.vocab_sha256 != expected:
        raise NativeCursorLabelizerError(
            "native cursor vocabulary fingerprint mismatch: "
            f"expected={expected} actual={labelizer.vocab_sha256}"
        )
    return labelizer


__all__ = (
    "NativeCursorLabelizer",
    "NativeCursorLabelizerError",
    "load_native_cursor_labelizer",
    "vocab_fingerprint",
)
