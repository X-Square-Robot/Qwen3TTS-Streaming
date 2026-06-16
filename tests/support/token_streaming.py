"""Helpers for strict token-streaming TTS tests.

The transport still carries text chunks, so a rigorous TOKEN-mode test should
send chunks that independently re-encode to the same token id stream the engine
would see for the complete text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_STREAM_TEXT = "你好，这是 token 级流式输入测试。"


class TokenChunkingUnavailable(RuntimeError):
    """Raised when the local tokenizer needed for strict token chunks is absent."""


@dataclass(frozen=True)
class TokenTextChunks:
    text: str
    chunks: list[str]
    token_ids: list[int]
    chunk_token_ids: list[list[int]]
    tokenizer_dir: str

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def max_chunk_chars(self) -> int:
        return max((len(chunk) for chunk in self.chunks), default=0)

    @property
    def max_chunk_tokens(self) -> int:
        return max((len(token_ids) for token_ids in self.chunk_token_ids), default=0)

    def to_details(self) -> dict:
        return {
            "text": self.text,
            "chunk_count": self.chunk_count,
            "token_count": self.token_count,
            "max_chunk_chars": self.max_chunk_chars,
            "max_chunk_tokens": self.max_chunk_tokens,
            "tokenizer_dir": self.tokenizer_dir,
            "chunks": list(self.chunks),
            "token_ids": list(self.token_ids),
            "chunk_token_ids": [list(token_ids) for token_ids in self.chunk_token_ids],
            "chunk_token_counts": [len(token_ids) for token_ids in self.chunk_token_ids],
        }


def _repo_path(path: str | os.PathLike[str], *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def _looks_like_tokenizer_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "tokenizer.json").is_file()
        or ((path / "vocab.json").is_file() and (path / "merges.txt").is_file())
    )


def candidate_tokenizer_dirs(*, repo_root: Path = REPO_ROOT) -> list[Path]:
    """Return likely tokenizer directories, de-duplicated in preference order."""
    candidates: list[Path] = []

    for env_name in (
        "QWEN3_TTS_TOKENIZER_DIR",
        "ENGINE_TOKENIZER_DIR",
        "TOKENIZER_DIR",
    ):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            candidates.append(_repo_path(raw, repo_root=repo_root))

    try:
        from engine.config import load_config, resolve_model_package_paths

        cfg = load_config(str(repo_root / "engine.yaml"))
        if cfg.paths.tokenizer_dir:
            candidates.append(_repo_path(cfg.paths.tokenizer_dir, repo_root=repo_root))
        if cfg.paths.model_package_dir:
            package_dir = _repo_path(cfg.paths.model_package_dir, repo_root=repo_root)
            paths = resolve_model_package_paths(str(package_dir))
            candidates.append(Path(paths.tokenizer_dir))
    except Exception:
        # Some lightweight test environments do not have PyYAML or the optional
        # tokenizer stack installed.  The fixed candidates below still give a
        # useful best effort, and callers decide whether absence is a skip/fail.
        pass

    candidates.extend(
        [
            repo_root / "workspace" / "model_repository" / "tts_orchestrator" / "1" / "tokenizer",
            repo_root / "model_repository" / "tts_orchestrator" / "1" / "tokenizer",
        ]
    )
    workspace_models = repo_root / "workspace" / "models"
    if workspace_models.is_dir():
        candidates.extend(sorted(path for path in workspace_models.iterdir() if path.is_dir()))

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_tokenizer_dir(
    tokenizer_dir: str | os.PathLike[str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
) -> Path:
    candidates = (
        [_repo_path(tokenizer_dir, repo_root=repo_root)]
        if tokenizer_dir
        else candidate_tokenizer_dirs(repo_root=repo_root)
    )
    for candidate in candidates:
        if _looks_like_tokenizer_dir(candidate):
            return candidate
    searched = ", ".join(str(path) for path in candidates) or "<none>"
    raise TokenChunkingUnavailable(
        "Could not find a Qwen3-TTS tokenizer directory for TOKEN-mode tests. "
        f"Searched: {searched}"
    )


def validate_token_text_chunks(
    text: str,
    token_ids: Iterable[int],
    chunks: Iterable[str],
    *,
    chunk_token_ids: Iterable[Iterable[int]] | None = None,
    tokenizer_dir: str | os.PathLike[str] = "",
    min_chunks: int = 2,
) -> TokenTextChunks:
    token_id_list = [int(token_id) for token_id in token_ids]
    chunk_list = [str(chunk) for chunk in chunks]
    if chunk_token_ids is None:
        chunk_token_id_list = [[token_id] for token_id in token_id_list]
        if len(token_id_list) != len(chunk_list):
            raise ValueError(
                f"token/chunk length mismatch: token_ids={len(token_id_list)} chunks={len(chunk_list)}"
            )
    else:
        chunk_token_id_list = [
            [int(token_id) for token_id in token_group]
            for token_group in chunk_token_ids
        ]
        if len(chunk_token_id_list) != len(chunk_list):
            raise ValueError(
                "chunk/chunk-token length mismatch: "
                f"chunks={len(chunk_list)} chunk_token_ids={len(chunk_token_id_list)}"
            )
        flattened = [
            token_id
            for token_group in chunk_token_id_list
            for token_id in token_group
        ]
        if flattened != token_id_list:
            raise ValueError(
                "chunk token ids do not reconstruct the original token ids: "
                f"expected={token_id_list!r} actual={flattened!r}"
            )
    if not text:
        raise ValueError("TOKEN-mode test text must not be empty")
    if len(chunk_list) < min_chunks:
        raise ValueError(
            f"tokenized text produced only {len(chunk_list)} chunks, expected at least {min_chunks}"
        )
    empty_indices = [idx for idx, chunk in enumerate(chunk_list) if chunk == ""]
    if empty_indices:
        raise ValueError(f"tokenized text produced empty text chunks at indices {empty_indices}")
    reconstructed = "".join(chunk_list)
    if reconstructed != text:
        raise ValueError(
            "token text chunks do not reconstruct the original text: "
            f"expected={text!r} actual={reconstructed!r}"
        )
    return TokenTextChunks(
        text=text,
        chunks=chunk_list,
        token_ids=token_id_list,
        chunk_token_ids=chunk_token_id_list,
        tokenizer_dir=str(tokenizer_dir),
    )


def _build_reencodable_text_chunks(
    text: str,
    token_ids: list[int],
    tokenizer,
) -> tuple[list[str], list[list[int]]]:
    """Split text into chunks that re-encode to the full tokenizer id stream.

    Byte-level BPE tokenizers can represent one visible character with several
    token ids.  Such ids cannot be sent as separate text chunks because the
    transport does not have a representation for "the empty continuation bytes"
    of a character.  This finds the smallest text chunks whose independent
    tokenization concatenates back to the exact full-tokenization ids.
    """

    text_len = len(text)
    token_count = len(token_ids)

    @lru_cache(maxsize=None)
    def encode_slice(start: int, end: int) -> tuple[int, ...]:
        return tuple(
            int(token_id)
            for token_id in tokenizer.encode_ids(
                text[start:end],
                add_special_tokens=False,
            )
        )

    @lru_cache(maxsize=None)
    def solve(char_pos: int, token_pos: int) -> tuple[tuple[str, tuple[int, ...]], ...] | None:
        if char_pos == text_len and token_pos == token_count:
            return ()
        if char_pos >= text_len or token_pos >= token_count:
            return None

        best: tuple[tuple[str, tuple[int, ...]], ...] | None = None
        for end in range(char_pos + 1, text_len + 1):
            chunk_ids = encode_slice(char_pos, end)
            if not chunk_ids:
                continue
            next_token_pos = token_pos + len(chunk_ids)
            if token_ids[token_pos:next_token_pos] != list(chunk_ids):
                continue
            rest = solve(end, next_token_pos)
            if rest is None:
                continue
            candidate = ((text[char_pos:end], chunk_ids),) + rest
            if best is None or len(candidate) > len(best):
                best = candidate
        return best

    solution = solve(0, 0)
    if solution is None:
        raise ValueError(
            "could not split text into independently re-encodable TOKEN-mode chunks"
        )

    return (
        [chunk for chunk, _ids in solution],
        [list(ids) for _chunk, ids in solution],
    )


def build_token_text_chunks(
    text: str = DEFAULT_TOKEN_STREAM_TEXT,
    *,
    tokenizer_dir: str | os.PathLike[str] | None = None,
    repo_root: Path = REPO_ROOT,
    min_chunks: int = 2,
) -> TokenTextChunks:
    """Tokenize text into strict transport chunks for INPUT_MODE_TOKEN tests."""
    resolved_dir = resolve_tokenizer_dir(tokenizer_dir, repo_root=repo_root)
    try:
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer
    except Exception as exc:
        raise TokenChunkingUnavailable(
            "Could not import the project tokenizer "
            "engine.frontend.spliter.tokenizer.LightQwen3TTSTokenizer. "
            "Install the tokenizer runtime dependencies before running strict "
            "TOKEN-mode E2E tests."
        ) from exc

    tokenizer = LightQwen3TTSTokenizer(str(resolved_dir))
    token_ids = list(tokenizer.encode_ids(text, add_special_tokens=False))
    chunks, chunk_token_ids = _build_reencodable_text_chunks(text, token_ids, tokenizer)
    tokenized = validate_token_text_chunks(
        text,
        token_ids,
        chunks,
        chunk_token_ids=chunk_token_ids,
        tokenizer_dir=resolved_dir,
        min_chunks=min_chunks,
    )

    mismatches: list[str] = []
    for idx, (expected_ids, chunk) in enumerate(zip(tokenized.chunk_token_ids, tokenized.chunks)):
        actual_ids = list(tokenizer.encode_ids(chunk, add_special_tokens=False))
        if actual_ids != expected_ids:
            mismatches.append(
                f"idx={idx} chunk={chunk!r} expected={expected_ids} actual={actual_ids}"
            )
        if len(mismatches) >= 5:
            break
    if mismatches:
        raise ValueError(
            "token text chunks are not stable TOKEN-mode chunks when re-encoded: "
            + "; ".join(mismatches)
        )

    return tokenized
