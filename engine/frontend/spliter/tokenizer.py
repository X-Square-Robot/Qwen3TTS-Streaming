import json
import logging

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class LightQwen3TTSTokenizer:
    def __init__(self, tokenizer_dir: str, logger_level: int = logging.INFO):
        self.tokenizer_dir = Path(tokenizer_dir)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logger_level)

        if not self.tokenizer_dir.is_dir():
            raise ValueError(f"Tokenizer directory {self.tokenizer_dir} does not exist")

        self.tokenizer_json_path = self.tokenizer_dir / "tokenizer.json"
        # 先看看有没有tokenizer.json文件如果有，那么直接加载就好了
        if self.tokenizer_json_path.is_file():
            try:
                self.tokenizer = Tokenizer.from_file(str(self.tokenizer_json_path))
                self.logger.info(f"Loaded tokenizer from {self.tokenizer_json_path}")
                return
            except Exception as e:
                self.logger.error(
                    f"Failed to load tokenizer from {self.tokenizer_json_path}: {e}"
                )

        # 没有tokenizer_config.json文件，那么需要从vocab.json和merges.txt文件中构建tokenizer
        vocab_path = self.tokenizer_dir / "vocab.json"
        merges_path = self.tokenizer_dir / "merges.txt"

        if not vocab_path.is_file():
            raise ValueError(f"Vocab file {vocab_path} does not exist")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        if not merges_path.is_file():
            raise ValueError(f"Merges file {merges_path} does not exist")
        merges = self._parse_merges(merges_path)

        self.tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token=None))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()

        added_tokens = self._load_added_tokens(self.tokenizer_dir)
        if added_tokens:
            self.tokenizer.add_special_tokens(added_tokens)
            self.logger.info(f"Added {len(added_tokens)} special tokens")

        self.logger.info(f"Loaded tokenizer from {self.tokenizer_dir}")

    @staticmethod
    def _display_token_text(token: str) -> str:
        """Make whitespace-bearing tokens readable in single-line debug logs."""
        return (
            str(token)
            .replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )

    @staticmethod
    def _parse_merges(merges_path: Path) -> list:
        """Parse merges.txt (GPT2/BPE format: one 'a b' per line)."""
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    merges.append((parts[0], parts[1]))
        return merges

    @staticmethod
    def _load_added_tokens(root: Path) -> List[Any]:
        """
        Parse added_tokens_decoder from tokenizer_config.json and return a list
        of tokenizers.AddedToken objects for special-token registration.
        """
        cfg_path = root / "tokenizer_config.json"
        if not cfg_path.is_file():
            return []

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return []

        decoder = cfg.get("added_tokens_decoder", {})
        if not decoder:
            return []

        tokens = []
        for _tid_str, info in sorted(decoder.items(), key=lambda x: int(x[0])):
            content = info.get("content", "")
            if not content:
                continue
            tokens.append(
                AddedToken(
                    content,
                    special=info.get("special", False),
                    normalized=info.get("normalized", False),
                    lstrip=info.get("lstrip", False),
                    rstrip=info.get("rstrip", False),
                    single_word=info.get("single_word", False),
                )
            )
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs: Any):
        """Return the raw tokenizers.Encoding object."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_ids(
        self, text: str, add_special_tokens: bool = True, **kwargs: Any
    ) -> List[int]:
        """Return token ids only."""
        return self.encode(text, add_special_tokens=add_special_tokens, **kwargs).ids

    def encode_with_offsets(
        self, text: str, add_special_tokens: bool = True, **kwargs: Any
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Return token ids and the corresponding original text offsets."""
        enc = self.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return enc.ids, enc.offsets

    def encode_with_tokens(
        self, text: str, add_special_tokens: bool = True, **kwargs: Any
    ) -> Tuple[List[int], List[str]]:
        """Return token ids and the corresponding original token."""
        enc = self.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return enc.ids, enc.tokens

    def encode_with_text(
        self, text: str, add_special_tokens: bool = True, **kwargs: Any
    ) -> Tuple[List[int], List[str]]:
        """Return token ids and stable original-text spans derived from offsets.

        Byte-level tokenizers can occasionally surface overlapping offsets for
        mixed-language text. Normalize them into monotonic, non-overlapping
        slices so joining the spans always reconstructs the original text.
        """
        enc = self.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        spans: List[str] = []
        prev_end = 0
        text_len = len(text)
        for start, end in enc.offsets:
            start = max(int(start), prev_end)
            end = max(int(end), start)
            if end > text_len:
                end = text_len
            spans.append(text[start:end])
            prev_end = end
        return enc.ids, spans

    def debug_snapshot(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return a structured tokenization snapshot for observability."""
        enc = self.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        spans: List[str] = []
        prev_end = 0
        text_len = len(text)
        for start, end in enc.offsets:
            start = max(int(start), prev_end)
            end = max(int(end), start)
            if end > text_len:
                end = text_len
            spans.append(text[start:end])
            prev_end = end

        pieces = []
        for idx, token_id in enumerate(enc.ids):
            token = enc.tokens[idx] if idx < len(enc.tokens) else ""
            offset = enc.offsets[idx] if idx < len(enc.offsets) else (0, 0)
            span = spans[idx] if idx < len(spans) else ""
            pieces.append(
                {
                    "index": idx,
                    "id": int(token_id),
                    "token": token,
                    "token_display": self._display_token_text(token),
                    "offset": [int(offset[0]), int(offset[1])],
                    "span": span,
                    "span_display": self._display_token_text(span),
                }
            )

        return {
            "text": text,
            "text_display": self._display_token_text(text),
            "add_special_tokens": bool(add_special_tokens),
            "ids": [int(token_id) for token_id in enc.ids],
            "tokens": list(enc.tokens),
            "tokens_display": [self._display_token_text(token) for token in enc.tokens],
            "offsets": [[int(start), int(end)] for start, end in enc.offsets],
            "spans": spans,
            "spans_display": [self._display_token_text(span) for span in spans],
            "pieces": pieces,
        }

    def __call__(
        self, text: str, return_tensors: str | None = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """HuggingFace-compatible __call__ for drop-in use in PrefillBuilder."""
        ids = self.encode_ids(
            text, add_special_tokens=kwargs.get("add_special_tokens", True)
        )
        if return_tensors == "np":
            import numpy as np

            arr = np.array(ids, dtype=np.int64).reshape(1, -1)
            return {"input_ids": arr}
        elif return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor([ids], dtype=torch.int64)}
        return {"input_ids": ids}

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


class _LegacyLightweightTokenizerAdapter:
    """Compatibility adapter for older tests/scripts that expect numpy payloads."""

    def __init__(self, tokenizer: LightQwen3TTSTokenizer):
        self._tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __call__(
        self, text: str, return_tensors: str | None = None, **kwargs: Any
    ) -> Dict[str, Any]:
        ids = self._tokenizer.encode_ids(
            text,
            add_special_tokens=kwargs.get("add_special_tokens", True),
        )
        if return_tensors in ("pt", "np"):
            import numpy as np

            return {"input_ids": np.array(ids, dtype=np.int64).reshape(1, -1)}
        return {"input_ids": ids}


def load_lightweight_tokenizer(tokenizer_dir: str) -> Optional[Any]:
    """Legacy loader kept for tests and verification scripts."""
    try:
        return _LegacyLightweightTokenizerAdapter(LightQwen3TTSTokenizer(tokenizer_dir))
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to load lightweight tokenizer from %s",
            tokenizer_dir,
            exc_info=True,
        )
        return None


__all__ = ("LightQwen3TTSTokenizer", "load_lightweight_tokenizer")
