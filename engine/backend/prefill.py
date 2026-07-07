"""Prefill builder for the standalone engine.

Migrated from model_repository/tts_orchestrator/1/prefill_builder.py.
Removed all Triton/pb_utils dependencies. Uses EmbeddingWeights directly.

The PrefillBuilder constructs:
  - prefill_embeds [1, S, H]:  fed to Talker's first forward pass
  - trailing list[[1,1,H]]:   one per text token, fed one-per-step during decode
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from ..text_normalization import strip_emoji

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-export from original: ResizeMLP, EmbeddingWeights, TaskType, PrefillPlan
# ---------------------------------------------------------------------------


class ResizeMLP(nn.Module):
    """text_hidden_size -> hidden_size projection."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(torch.nn.functional.silu(self.linear_fc1(x)))


class TaskType(Enum):
    VOICE_CLONE_ICL = "voice_clone_icl"
    VOICE_CLONE_XVEC = "voice_clone_xvec"
    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"


@dataclass
class PrefillPlan:
    prefill_embeds: torch.Tensor  # [1, S, H] bf16
    trailing: list  # list of [1, 1, H] bf16
    prefix_cache_key: Optional[str] = None
    cacheable_prefix_embeds: Optional[torch.Tensor] = None
    request_prefill_embeds: Optional[torch.Tensor] = None
    warnings: Optional[list] = None
    trailing_token_char_offsets: list = field(default_factory=list)


def parse_task_type(task_type_str: str, x_vector_only: bool = False) -> TaskType:
    if task_type_str == "voice_clone":
        return TaskType.VOICE_CLONE_XVEC if x_vector_only else TaskType.VOICE_CLONE_ICL
    elif task_type_str == "custom_voice":
        return TaskType.CUSTOM_VOICE
    elif task_type_str == "voice_design":
        return TaskType.VOICE_DESIGN
    else:
        raise ValueError(f"Unknown task_type: {task_type_str}")


# ---------------------------------------------------------------------------
# EmbeddingWeights — loads .pt weights, provides embed() on GPU
# ---------------------------------------------------------------------------


class EmbeddingWeights:
    """Holds text/codec/special embeddings on GPU (BF16)."""

    def __init__(
        self,
        weights_dir: str,
        device_id: int = 0,
        *,
        device: Optional[str | torch.device] = None,
        default_speaker: Optional[str] = None,
        fallback_speaker: Optional[str] = None,
    ):
        weights_dir = Path(weights_dir)
        with open(weights_dir / "config.json") as f:
            self.config = json.load(f)

        self.variant = self.config["variant"]
        self.hidden_size = self.config["talker_hidden_size"]
        self.text_hidden_size = self.config.get(
            "talker_text_hidden_size", self.hidden_size
        )
        self.vocab_size = self.config["talker_vocab_size"]
        self.codec_eos_id = self.config["codec_eos_token_id"]
        self.codec_bos_id = self.config["codec_bos_id"]
        self.codec_pad_id = self.config["codec_pad_id"]
        self.codec_nothink_id = self.config.get("codec_nothink_id", 2155)
        self.codec_think_bos_id = self.config.get("codec_think_bos_id", 2156)
        self.codec_think_eos_id = self.config.get("codec_think_eos_id", 2157)
        self.codec_think_id = self.config.get("codec_think_id", 2154)
        self.codec_language_id = self.config.get("codec_language_id", {})
        self.spk_id_map = {
            str(name).strip().lower(): int(spk_id)
            for name, spk_id in (self.config.get("spk_id", {}) or {}).items()
            if str(name).strip()
        }
        self.spk_is_dialect = {
            str(name).strip().lower(): dialect
            for name, dialect in (self.config.get("spk_is_dialect") or {}).items()
            if str(name).strip()
        }
        # CustomVoice: empty → default_speaker; unknown name → fallback_speaker.
        # Prefer engine.yaml (passed in); else weights config.json; else vivian.
        self.default_speaker = self._canonical_config_speaker(
            default_speaker
            if default_speaker is not None
            else self.config.get("default_speaker", "vivian"),
            "default_speaker",
        )
        self.fallback_speaker = self._canonical_config_speaker(
            fallback_speaker
            if fallback_speaker is not None
            else self.config.get("fallback_speaker", "vivian"),
            "fallback_speaker",
        )

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda", device_id)
        )
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype

        text_emb_sd = torch.load(
            weights_dir / "text_embedding.pt",
            map_location=self.device,
            weights_only=True,
        )
        weight = text_emb_sd["weight"]
        self.text_embedding = nn.Embedding(weight.shape[0], weight.shape[1]).to(
            device=self.device,
            dtype=dtype,
        )
        self.text_embedding.load_state_dict(
            {k: v.to(device=self.device, dtype=dtype) for k, v in text_emb_sd.items()}
        )
        self.text_embedding.eval()

        text_proj_sd = torch.load(
            weights_dir / "text_projection.pt",
            map_location=self.device,
            weights_only=True,
        )
        in_size = text_proj_sd["linear_fc1.weight"].shape[1]
        mid_size = text_proj_sd["linear_fc1.weight"].shape[0]
        out_size = text_proj_sd["linear_fc2.weight"].shape[0]
        self.text_projection = ResizeMLP(in_size, mid_size, out_size, bias=True).to(
            device=self.device,
            dtype=dtype,
        )
        self.text_projection.load_state_dict(
            {k: v.to(device=self.device, dtype=dtype) for k, v in text_proj_sd.items()}
        )
        self.text_projection.eval()

        codec_data = torch.load(
            weights_dir / "codec_embeddings.pt",
            map_location=self.device,
            weights_only=True,
        )
        talker_sd = codec_data["talker_codec_embedding"]
        c_weight = talker_sd["weight"]
        self.codec_embedding = nn.Embedding(c_weight.shape[0], c_weight.shape[1]).to(
            device=self.device,
            dtype=dtype,
        )
        self.codec_embedding.load_state_dict(
            {k: v.to(device=self.device, dtype=dtype) for k, v in talker_sd.items()}
        )
        self.codec_embedding.eval()

        special = torch.load(
            weights_dir / "special_embeddings.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.tts_pad_token_id = int(special["tts_pad_token_id"])
        self.tts_bos_token_id = int(special["tts_bos_token_id"])
        self.tts_eos_token_id = int(special["tts_eos_token_id"])
        with torch.no_grad():
            # Recompute specials with the loaded BF16 modules so runtime prefill matches
            # live-model inference instead of inheriting tiny export-time FP32 deltas.
            special_ids = torch.tensor(
                [[self.tts_pad_token_id, self.tts_bos_token_id, self.tts_eos_token_id]],
                device=self.device,
                dtype=torch.int64,
            )
            special_projected = self.text_projection(self.text_embedding(special_ids))
        self.tts_pad_embed = special_projected[:, 0:1, :].to(
            device=self.device, dtype=dtype
        )
        self.tts_bos_embed = special_projected[:, 1:2, :].to(
            device=self.device, dtype=dtype
        )
        self.tts_eos_embed = special_projected[:, 2:3, :].to(
            device=self.device, dtype=dtype
        )

        path_3d = weights_dir / "codec_embeddings_3d.pt"
        self.codec_embeddings_3d = None
        if path_3d.exists():
            self.codec_embeddings_3d = torch.load(
                path_3d,
                map_location=self.device,
                weights_only=True,
            ).to(device=self.device, dtype=dtype)

        logger.info(
            "Weights: variant=%s, hidden=%d, vocab=%d",
            self.variant,
            self.hidden_size,
            self.vocab_size,
        )

    def _canonical_config_speaker(self, speaker: Any, field_name: str) -> str:
        """Normalize configured default/fallback speakers to existing spk_id keys."""
        raw = str(speaker or "").strip().lower()
        if not self.spk_id_map:
            return raw
        if raw in self.spk_id_map:
            return raw

        first_speaker = next(iter(self.spk_id_map))
        if raw:
            logger.warning(
                "%s %r not in spk_id map; using first supported speaker %r",
                field_name,
                raw,
                first_speaker,
            )
        return first_speaker

    def text_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.text_projection(self.text_embedding(token_ids))

    def codec_embed(self, codec_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.codec_embedding(codec_ids)


# ---------------------------------------------------------------------------
# Normalize text (inlined from text_segmenter to avoid cross-dependency)
# ---------------------------------------------------------------------------


def normalize_tts_text(text: str) -> str:
    """Basic text normalization for TTS input."""
    text = strip_emoji(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    import re

    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# PrefillBuilder
# ---------------------------------------------------------------------------

OFFICIAL_ASSISTANT_FMT = (
    "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
)
OFFICIAL_REF_TEXT_FMT = "<|im_start|>assistant\n{text}<|im_end|>\n"
OFFICIAL_INSTRUCT_FMT = "<|im_start|>user\n{instruct}<|im_end|>\n"
OFFICIAL_REF_TEXT_PREFIX = "<|im_start|>assistant\n"
OFFICIAL_REF_TEXT_SUFFIX = "<|im_end|>\n"
OFFICIAL_INSTRUCT_PREFIX = "<|im_start|>user\n"
OFFICIAL_INSTRUCT_SUFFIX = "<|im_end|>\n"
PROMPT_TEMPLATE_VERSION = "qwen3-tts-official-chat-v1"

_LANGUAGE_ALIASES = {
    "zh": "chinese",
    "zh-cn": "chinese",
    "zh-hans": "chinese",
    "zh-hant": "chinese",
    "cn": "chinese",
    "en": "english",
    "en-us": "english",
    "en-gb": "english",
}


def _normalize_language_name(language: str) -> str:
    value = (language or "auto").strip().lower()
    return _LANGUAGE_ALIASES.get(value, value)


class PrefillBuilder:
    """Build prefill inputs_embeds for Talker.

    Exactly replicates the official model.generate() logic:
      - Streaming mode: first text token in prefill, rest as trailing
      - Non-streaming (VoiceDesign): all text folded into prefill
      - ICL (voice clone): reference codec + text into prefill
    """

    def __init__(
        self,
        weights: EmbeddingWeights,
        tokenizer: Any,
        *,
        output_device: Optional[str | torch.device] = None,
    ):
        self.w = weights
        self.tokenizer = tokenizer
        self.output_device = (
            torch.device(output_device) if output_device is not None else weights.device
        )
        self.output_dtype = (
            weights.dtype if self.output_device.type != "cuda" else torch.bfloat16
        )
        self._assistant_role_ids: Optional[list[int]] = None
        self._prompt_wrapper_ids: dict[tuple[str, str], list[int]] = {}
        with torch.no_grad():
            self._codec_bos_embed = weights.codec_embed(
                torch.tensor(
                    [[weights.codec_bos_id]],
                    device=weights.device,
                    dtype=torch.int64,
                ),
            )

    def _move_tensor(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        dtype = self.output_dtype if tensor.is_floating_point() else tensor.dtype
        return tensor.to(device=self.output_device, dtype=dtype)

    def _move_trailing(self, trailing: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self._move_tensor(t) for t in trailing]

    def build_plan(
        self,
        task_type: TaskType,
        text: str,
        language: str = "auto",
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        instruct_token_ids: Optional[list[int]] = None,
        spk_embedding: Optional[torch.Tensor] = None,
        ref_codes: Optional[torch.Tensor] = None,
        ref_text: Optional[str] = None,
        ref_text_token_ids: Optional[list[int]] = None,
        ref_codec_sum_vec: Optional[torch.Tensor] = None,
        ref_audio_sha256: Optional[str] = None,
        ref_feature_cache_key: Optional[str] = None,
        include_eos: bool = True,
    ) -> PrefillPlan:
        text = normalize_tts_text(text)
        token_ids = self._encode_text_ids(text)
        return self.build_plan_from_ids(
            task_type=task_type,
            token_ids=token_ids,
            language=language,
            speaker=speaker,
            instruct=instruct,
            instruct_token_ids=instruct_token_ids,
            spk_embedding=spk_embedding,
            ref_codes=ref_codes,
            ref_text=ref_text,
            ref_text_token_ids=ref_text_token_ids,
            ref_codec_sum_vec=ref_codec_sum_vec,
            ref_audio_sha256=ref_audio_sha256,
            ref_feature_cache_key=ref_feature_cache_key,
            include_eos=include_eos,
        )

    def build(
        self,
        task_type: TaskType,
        text: str,
        language: str = "auto",
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        instruct_token_ids: Optional[list[int]] = None,
        spk_embedding: Optional[torch.Tensor] = None,
        ref_codes: Optional[torch.Tensor] = None,
        ref_text: Optional[str] = None,
        ref_text_token_ids: Optional[list[int]] = None,
        ref_codec_sum_vec: Optional[torch.Tensor] = None,
        ref_audio_sha256: Optional[str] = None,
        ref_feature_cache_key: Optional[str] = None,
        include_eos: bool = True,
    ) -> tuple[torch.Tensor, list]:
        """Compatibility wrapper for older tests and verification scripts."""
        plan = self.build_plan(
            task_type=task_type,
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            instruct_token_ids=instruct_token_ids,
            spk_embedding=spk_embedding,
            ref_codes=ref_codes,
            ref_text=ref_text,
            ref_text_token_ids=ref_text_token_ids,
            ref_codec_sum_vec=ref_codec_sum_vec,
            ref_audio_sha256=ref_audio_sha256,
            ref_feature_cache_key=ref_feature_cache_key,
            include_eos=include_eos,
        )
        return plan.prefill_embeds, plan.trailing

    def build_plan_from_ids(
        self,
        task_type: TaskType,
        token_ids: list[int],
        language: str = "auto",
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        instruct_token_ids: Optional[list[int]] = None,
        spk_embedding: Optional[torch.Tensor] = None,
        ref_codes: Optional[torch.Tensor] = None,
        ref_text: Optional[str] = None,
        ref_text_token_ids: Optional[list[int]] = None,
        ref_codec_sum_vec: Optional[torch.Tensor] = None,
        ref_audio_sha256: Optional[str] = None,
        ref_feature_cache_key: Optional[str] = None,
        include_eos: bool = True,
    ) -> PrefillPlan:
        w = self.w
        device = w.device
        non_streaming_mode = task_type == TaskType.VOICE_DESIGN
        instruct_ids = self._normalize_prompt_token_ids(
            instruct_token_ids,
            instruct,
        )
        ref_ids = self._normalize_prompt_token_ids(
            ref_text_token_ids,
            ref_text,
        )
        text_ids = (
            torch.tensor(
                [token_ids],
                device=device,
                dtype=torch.int64,
            )
            if token_ids
            else torch.zeros(
                1,
                0,
                device=device,
                dtype=torch.int64,
            )
        )
        role_embed = w.text_embed(self._assistant_role_ids_tensor(device))

        speaker_embed = None
        plan_warnings: list[str] = []
        custom_speaker_id: Optional[int] = None
        if task_type == TaskType.CUSTOM_VOICE:
            resolved_speaker, custom_speaker_id, warn_msg = (
                self._resolve_custom_speaker(
                    speaker,
                    warn=True,
                )
            )
            speaker = resolved_speaker
            if warn_msg:
                plan_warnings.append(warn_msg)

        lang_lower = _normalize_language_name(language)
        language_id: Optional[int] = None
        if lang_lower != "auto":
            if lang_lower not in w.codec_language_id:
                raise NotImplementedError(f"Language {language} not implemented")
            language_id = w.codec_language_id[lang_lower]

        if lang_lower in ("chinese", "auto") and speaker:
            dialect_key = (
                w.spk_is_dialect.get(speaker.lower()) if w.spk_is_dialect else None
            )
            if dialect_key is not None and dialect_key is not False:
                dkey = str(dialect_key).lower()
                if dkey not in w.codec_language_id:
                    raise NotImplementedError(
                        f"Dialect language {dialect_key!r} not implemented"
                    )
                language_id = w.codec_language_id[dkey]

        if language_id is None:
            tag_ids = torch.tensor(
                [[w.codec_nothink_id, w.codec_think_bos_id, w.codec_think_eos_id]],
                device=device,
                dtype=torch.int64,
            )
        else:
            tag_ids = torch.tensor(
                [
                    [
                        w.codec_think_id,
                        w.codec_think_bos_id,
                        language_id,
                        w.codec_think_eos_id,
                    ]
                ],
                device=device,
                dtype=torch.int64,
            )
        codec_input_embedding_0 = w.codec_embed(tag_ids)

        codec_input_embedding_1 = w.codec_embed(
            torch.tensor(
                [[w.codec_pad_id, w.codec_bos_id]], device=device, dtype=torch.int64
            )
        )

        if task_type == TaskType.CUSTOM_VOICE:
            if custom_speaker_id is not None:
                speaker_embed = w.codec_embed(
                    torch.tensor(
                        [[custom_speaker_id]], device=device, dtype=torch.int64
                    ),
                )
            else:
                speaker = None
        elif task_type in (TaskType.VOICE_CLONE_ICL, TaskType.VOICE_CLONE_XVEC):
            if spk_embedding is not None:
                speaker_embed = spk_embedding.to(device=device, dtype=w.dtype).reshape(
                    1, 1, -1
                )

        if speaker_embed is not None:
            codec_input_embedding = torch.cat(
                [codec_input_embedding_0, speaker_embed, codec_input_embedding_1],
                dim=1,
            )
        else:
            codec_input_embedding = torch.cat(
                [codec_input_embedding_0, codec_input_embedding_1],
                dim=1,
            )

        n_codec = codec_input_embedding.shape[1]
        text_layer = torch.cat(
            [
                w.tts_pad_embed.expand(1, n_codec - 2, w.hidden_size),
                w.tts_bos_embed,
            ],
            dim=1,
        )
        dual_track = text_layer + codec_input_embedding[:, :-1]

        instruct_embed = None
        if instruct_ids and task_type in (TaskType.CUSTOM_VOICE, TaskType.VOICE_DESIGN):
            instruct_embed = w.text_embed(
                self._wrap_prompt_ids_tensor(
                    instruct_ids,
                    prefix=OFFICIAL_INSTRUCT_PREFIX,
                    suffix=OFFICIAL_INSTRUCT_SUFFIX,
                    device=device,
                )
            )

        if instruct_embed is not None:
            talker_input_embed = torch.cat(
                [instruct_embed, role_embed, dual_track], dim=1
            )
        else:
            talker_input_embed = torch.cat([role_embed, dual_track], dim=1)

        # ---- ICL path ----
        icl_handled = False
        if task_type == TaskType.VOICE_CLONE_ICL and ref_codec_sum_vec is not None:
            prefill, trailing = self._build_icl_path(
                w,
                device,
                text_ids,
                talker_input_embed,
                ref_codec_sum_vec,
                ref_ids,
                include_eos=include_eos,
            )
            char_offsets = []
            icl_handled = True

        elif (
            task_type == TaskType.VOICE_CLONE_ICL
            and ref_codes is not None
            and w.codec_embeddings_3d is not None
        ):
            T_ref, _ = ref_codes.shape
            g_idx = (
                torch.arange(16, device=device, dtype=torch.int64)
                .reshape(1, -1)
                .expand(T_ref, 16)
            )
            codec_sum_vec = (
                w.codec_embeddings_3d[g_idx, ref_codes, :].sum(dim=1).unsqueeze(0)
            )
            prefill, trailing = self._build_icl_path(
                w,
                device,
                text_ids,
                talker_input_embed,
                codec_sum_vec,
                ref_ids,
                include_eos=include_eos,
            )
            char_offsets = []
            icl_handled = True

        if not icl_handled:
            if text_ids.shape[1] > 0:
                first_text_embed = w.text_embed(text_ids[:, :1])
            else:
                first_text_embed = w.tts_pad_embed
            first_text_with_bos = first_text_embed + codec_input_embedding[:, -1:]
            talker_input_embed = torch.cat(
                [talker_input_embed, first_text_with_bos], dim=1
            )

        if not icl_handled and non_streaming_mode:
            talker_input_embed = talker_input_embed[:, :-1]
            n_text = text_ids.shape[1]
            if n_text > 0:
                text_embed = torch.cat([w.text_embed(text_ids), w.tts_eos_embed], dim=1)
            else:
                text_embed = w.tts_eos_embed
            codec_pad_ids = torch.full(
                (1, n_text + 1), w.codec_pad_id, device=device, dtype=torch.int64
            )
            codec_pad_embed = w.codec_embed(codec_pad_ids)
            prefill = torch.cat(
                [
                    talker_input_embed,
                    text_embed + codec_pad_embed,
                    w.tts_pad_embed
                    + w.codec_embed(
                        torch.tensor(
                            [[w.codec_bos_id]], device=device, dtype=torch.int64
                        )
                    ),
                ],
                dim=1,
            )
            trailing = [w.tts_pad_embed.clone()]
            char_offsets = []

        elif not icl_handled:
            # ---- Streaming mode ----
            prefill = talker_input_embed
            mid = text_ids[:, 1:] if text_ids.shape[1] > 1 else text_ids[:, :0]
            if mid.shape[1] > 0:
                mid_embed = w.text_embed(mid)
                if include_eos:
                    trailing_text = torch.cat([mid_embed, w.tts_eos_embed], dim=1)
                else:
                    trailing_text = mid_embed
            else:
                trailing_text = (
                    w.tts_eos_embed
                    if include_eos
                    else torch.zeros(
                        1, 0, w.hidden_size, device=device, dtype=torch.bfloat16
                    )
                )
            trailing = [
                trailing_text[:, i : i + 1, :].clone()
                for i in range(trailing_text.shape[1])
            ]
            char_offsets = []

        prefix_cache_key = None
        cacheable_prefix_embeds = None
        request_prefill_embeds = None
        if task_type in (
            TaskType.CUSTOM_VOICE,
            TaskType.VOICE_DESIGN,
            TaskType.VOICE_CLONE_XVEC,
        ):
            if non_streaming_mode:
                cacheable_prefix_embeds = talker_input_embed.clone().contiguous()
                request_prefill_embeds = (
                    prefill[:, cacheable_prefix_embeds.shape[1] :, :]
                    .clone()
                    .contiguous()
                )
            else:
                cacheable_prefix_embeds = prefill[:, :-1, :].clone().contiguous()
                request_prefill_embeds = prefill[:, -1:, :].clone().contiguous()
            if (
                cacheable_prefix_embeds.shape[1] > 0
                and request_prefill_embeds.shape[1] > 0
            ):
                prefix_cache_key = self._prefix_cache_key(
                    task_type,
                    language,
                    speaker,
                    instruct,
                    spk_embedding,
                    instruct_token_ids=instruct_ids,
                )
            else:
                cacheable_prefix_embeds = None
                request_prefill_embeds = None

        prefill = self._move_tensor(prefill)
        trailing = self._move_trailing(trailing)
        cacheable_prefix_embeds = self._move_tensor(cacheable_prefix_embeds)
        request_prefill_embeds = self._move_tensor(request_prefill_embeds)

        return PrefillPlan(
            prefill_embeds=prefill,
            trailing=trailing,
            prefix_cache_key=prefix_cache_key,
            cacheable_prefix_embeds=cacheable_prefix_embeds,
            request_prefill_embeds=request_prefill_embeds,
            warnings=plan_warnings or None,
            trailing_token_char_offsets=char_offsets,
        )

    def _build_icl_path(
        self,
        w,
        device,
        text_ids,
        talker_input_embed,
        ref_codec_sum_vec,
        ref_text_token_ids,
        *,
        include_eos: bool = True,
    ):
        codec_sum_vec = ref_codec_sum_vec.to(device=device, dtype=torch.bfloat16)
        if codec_sum_vec.dim() == 2:
            codec_sum_vec = codec_sum_vec.unsqueeze(0)
        if codec_sum_vec.dim() != 3:
            raise ValueError(
                f"ICL ref_codec_sum_vec must be [B, T, H], got shape {tuple(codec_sum_vec.shape)}"
            )
        codec_embed_icl = torch.cat(
            [
                w.codec_embed(
                    torch.tensor([[w.codec_bos_id]], device=device, dtype=torch.int64)
                ),
                codec_sum_vec,
            ],
            dim=1,
        )

        if ref_text_token_ids:
            ref_id = torch.tensor(
                [ref_text_token_ids],
                device=device,
                dtype=torch.int64,
            )
        else:
            ref_id = text_ids[:, :0]

        text_embed_icl = w.text_embed(torch.cat([ref_id, text_ids], dim=1))
        if include_eos:
            text_embed_icl = torch.cat([text_embed_icl, w.tts_eos_embed], dim=1)
        text_lens = text_embed_icl.shape[1]
        codec_lens = codec_embed_icl.shape[1]
        ref_text_lens = ref_id.shape[1]
        if ref_text_lens and codec_lens <= ref_text_lens:
            raise ValueError(
                "icl_ref_codec_collapsed: ICL reference codec frames do not cover ref_text tokens; "
                "speech_tokenizer_codec_fused may be an old collapsed export"
            )

        if text_lens > codec_lens:
            icl_embed = text_embed_icl[:, :codec_lens] + codec_embed_icl
            trailing = [
                text_embed_icl[:, i : i + 1] for i in range(codec_lens, text_lens)
            ]
        else:
            pad_len = codec_lens - text_lens
            if pad_len > 0:
                text_embed_icl = torch.cat(
                    [text_embed_icl] + [w.tts_pad_embed] * pad_len, dim=1
                )
            icl_embed = text_embed_icl + codec_embed_icl
            trailing = [w.tts_pad_embed] if include_eos else []

        prefill = torch.cat([talker_input_embed, icl_embed], dim=1)
        return prefill, trailing

    def build_trailing_embeds(
        self, text: str, include_eos: bool = True
    ) -> list[torch.Tensor]:
        """Build trailing embeddings for streaming text continuation."""
        text = normalize_tts_text(text)
        w = self.w
        device = w.device
        token_ids = self._encode_text_ids(text)
        text_tokens = (
            torch.tensor(
                [token_ids],
                device=device,
                dtype=torch.int64,
            )
            if token_ids
            else torch.zeros(
                1,
                0,
                device=device,
                dtype=torch.int64,
            )
        )
        if text_tokens.shape[1] > 0:
            text_embed = w.text_embed(text_tokens)
            if include_eos:
                text_embed = torch.cat([text_embed, w.tts_eos_embed], dim=1)
        else:
            if include_eos:
                text_embed = w.tts_eos_embed
            else:
                return []
        trailing = [
            text_embed[:, i : i + 1, :].clone() for i in range(text_embed.shape[1])
        ]
        return self._move_trailing(trailing)

    def _encode_text_ids(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "encode_ids"):
            return list(self.tokenizer.encode_ids(text, add_special_tokens=False))
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            return input_ids.reshape(-1).to(dtype=torch.int64).tolist()
        input_ids_np = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        return input_ids_np.tolist()

    def _assistant_role_ids_tensor(self, device: torch.device) -> torch.Tensor:
        if self._assistant_role_ids is None:
            assistant_empty_ids = self._encode_text_ids(
                OFFICIAL_ASSISTANT_FMT.format(text=""),
            )
            if len(assistant_empty_ids) < 3:
                raise ValueError(
                    "assistant prompt template produced fewer than 3 role tokens"
                )
            self._assistant_role_ids = assistant_empty_ids[:3]
        return torch.tensor(
            [self._assistant_role_ids],
            device=device,
            dtype=torch.int64,
        )

    def compute_cache_key(
        self,
        task_type: TaskType,
        language: str = "auto",
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        instruct_token_ids: Optional[list[int]] = None,
        spk_embedding: Optional[torch.Tensor] = None,
        ref_text_token_ids: Optional[list[int]] = None,
        ref_audio_sha256: Optional[str] = None,
        ref_codec_sum_vec: Optional[torch.Tensor] = None,
        ref_feature_cache_key: Optional[str] = None,
    ) -> Optional[str]:
        """Compute prefix cache key without building the full plan.

        The key depends only on (variant, task_type, language, speaker),
        NOT on the text content — so callers can check the cache before
        any tokenization or embedding work.
        """
        return self._prefix_cache_key(
            task_type,
            language,
            speaker,
            instruct,
            spk_embedding,
            instruct_token_ids=instruct_token_ids,
            ref_text_token_ids=ref_text_token_ids,
            ref_audio_sha256=ref_audio_sha256,
            ref_codec_sum_vec=ref_codec_sum_vec,
            ref_feature_cache_key=ref_feature_cache_key,
        )

    def build_suffix_from_ids(
        self,
        token_ids: list[int],
        include_eos: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Build request_prefill_embeds + trailing from raw token IDs.

        Fast path for prefix cache hits: embeds text tokens directly,
        skipping the text→template→retokenize round-trip that
        ``build_plan()`` performs.

        Args:
            token_ids: raw text token IDs (from dispatcher / spliter).
            include_eos: whether to append tts_eos_embed to trailing.

        Returns:
            (request_prefill_embeds [1,1,H], trailing list[[1,1,H]])
        """
        w = self.w
        device = w.device
        ids_tensor = torch.tensor(
            [token_ids],
            device=device,
            dtype=torch.int64,
        )
        with torch.no_grad():
            first_embed = w.text_embed(ids_tensor[:, :1])
        request_prefill_embeds = first_embed + self._codec_bos_embed

        with torch.no_grad():
            if ids_tensor.shape[1] > 1:
                mid_embed = w.text_embed(ids_tensor[:, 1:])
                if include_eos:
                    trailing_text = torch.cat(
                        [mid_embed, w.tts_eos_embed],
                        dim=1,
                    )
                else:
                    trailing_text = mid_embed
            else:
                if include_eos:
                    trailing_text = w.tts_eos_embed
                else:
                    trailing_text = torch.zeros(
                        1,
                        0,
                        w.hidden_size,
                        device=device,
                        dtype=torch.bfloat16,
                    )
        # Views into one storage: trailing tensors are read-only downstream,
        # and one clone kernel per text token dominated cache-hit admission
        # CPU cost during bursts.
        trailing = list(trailing_text.split(1, dim=1))
        return self._move_tensor(request_prefill_embeds), self._move_trailing(trailing)

    def build_suffix_batch(
        self,
        token_lists: list[list[int]],
        include_eos_flags: list[bool],
    ) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        """Batched build_suffix_from_ids for burst admission.

        Bitwise-equivalent to calling build_suffix_from_ids per session (see
        the shape note below); trailing tensors are read-only views, so the
        per-token clone kernels of the historical path are still avoided.

        Args:
            token_lists: raw text token IDs per session (each non-empty).
            include_eos_flags: whether to append tts_eos_embed per session.

        Returns:
            One (request_prefill_embeds [1,1,H], trailing list[[1,1,H]])
            tuple per session, matching build_suffix_from_ids.
        """
        w = self.w
        device = w.device

        # NOTE: text_embed includes a projection matmul, and GEMM reduction
        # order varies with input shape.  A single flat [1, sum(T)] lookup
        # produces last-bit bf16 differences vs the serial path's per-session
        # [1,1] + [1,T-1] calls — enough to flip marginal sampling seeds into
        # hallucination (see streaming_hallucination.md).  So the embed calls
        # here keep bitwise-identical shapes to build_suffix_from_ids; the
        # batching wins come from view-splitting (no per-token clones) plus
        # the caller's batched KV restore / zero-state allocation.
        results: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
        with torch.no_grad():
            for ids, include_eos in zip(token_lists, include_eos_flags):
                ids_tensor = torch.tensor([ids], device=device, dtype=torch.int64)
                first_embed = w.text_embed(ids_tensor[:, :1])
                request_embeds = first_embed + self._codec_bos_embed
                if ids_tensor.shape[1] > 1:
                    mid = w.text_embed(ids_tensor[:, 1:])
                    trailing = list(mid.split(1, dim=1))
                else:
                    trailing = []
                if include_eos:
                    trailing.append(w.tts_eos_embed)
                results.append(
                    (
                        self._move_tensor(request_embeds),
                        self._move_trailing(trailing),
                    )
                )
        return results

    def _normalize_prompt_token_ids(
        self,
        token_ids: Optional[list[int]],
        text: Optional[str],
    ) -> list[int]:
        if token_ids is not None:
            return list(token_ids)
        normalized = normalize_tts_text(text or "").strip()
        if not normalized:
            return []
        return self._encode_text_ids(normalized)

    def _wrap_prompt_ids_tensor(
        self,
        token_ids: list[int],
        *,
        prefix: str,
        suffix: str,
        device: torch.device,
    ) -> torch.Tensor:
        wrapped_ids = (
            self._static_prompt_ids(prefix)
            + list(token_ids)
            + self._static_prompt_ids(suffix)
        )
        return torch.tensor([wrapped_ids], device=device, dtype=torch.int64)

    def _static_prompt_ids(self, text: str) -> list[int]:
        key = ("static", text)
        cached = self._prompt_wrapper_ids.get(key)
        if cached is None:
            cached = self._encode_text_ids(text)
            self._prompt_wrapper_ids[key] = cached
        return cached

    def _resolve_custom_speaker(
        self,
        speaker: Optional[str],
        *,
        warn: bool = False,
    ) -> tuple[Optional[str], Optional[int], Optional[str]]:
        """Resolve a CustomVoice speaker to the actual codec embedding id."""
        spk_map = self.w.spk_id_map or {}
        if not spk_map:
            msg = (
                "CustomVoice model has no spk_id map; continuing without speaker codec"
            )
            if warn:
                logger.warning(msg)
            return None, None, msg

        raw = (speaker or "").strip()
        raw_key = raw.lower()
        if raw_key:
            spk_id = spk_map.get(raw_key)
            if spk_id is not None:
                return raw_key, int(spk_id), None

            fallback_key = (self.w.fallback_speaker or "").strip().lower()
            label = "fallback_speaker"
            if fallback_key not in spk_map:
                fallback_key = next(iter(spk_map))
                label = "first supported speaker"
            msg = f"Speaker {raw!r} not found, using {label} {fallback_key!r}"
            if warn:
                logger.warning(msg)
            return fallback_key, int(spk_map[fallback_key]), msg

        default_key = (self.w.default_speaker or "").strip().lower()
        label = "default_speaker"
        if default_key not in spk_map:
            default_key = next(iter(spk_map))
            label = "first supported speaker"
        msg = None
        if label != "default_speaker":
            msg = f"default_speaker not found, using {label} {default_key!r}"
            if warn:
                logger.warning(msg)
        return default_key, int(spk_map[default_key]), msg

    def _prefix_cache_key(
        self,
        task_type,
        language,
        speaker,
        instruct,
        spk_embedding,
        *,
        instruct_token_ids: Optional[list[int]] = None,
        ref_text_token_ids: Optional[list[int]] = None,
        ref_audio_sha256: Optional[str] = None,
        ref_codec_sum_vec: Optional[torch.Tensor] = None,
        ref_feature_cache_key: Optional[str] = None,
    ):
        instruct_ids = self._normalize_prompt_token_ids(instruct_token_ids, instruct)
        if instruct_ids:
            instruct_key = hashlib.sha1(
                np.asarray(instruct_ids, dtype=np.int64).tobytes(),
            ).hexdigest()[:16]
        else:
            instruct_key = ""
        template_key = hashlib.sha1(
            "|".join(
                (
                    PROMPT_TEMPLATE_VERSION,
                    OFFICIAL_ASSISTANT_FMT,
                    OFFICIAL_REF_TEXT_FMT,
                    OFFICIAL_INSTRUCT_FMT,
                    OFFICIAL_REF_TEXT_PREFIX,
                    OFFICIAL_REF_TEXT_SUFFIX,
                    OFFICIAL_INSTRUCT_PREFIX,
                    OFFICIAL_INSTRUCT_SUFFIX,
                )
            ).encode("utf-8"),
        ).hexdigest()[:16]
        key_parts = [
            self.w.variant,
            task_type.value,
            _normalize_language_name(language),
            instruct_key,
            template_key,
        ]
        if task_type == TaskType.CUSTOM_VOICE:
            resolved_speaker, spk_id, _ = self._resolve_custom_speaker(
                speaker,
                warn=False,
            )
            if resolved_speaker is None or spk_id is None:
                return None
            key_parts.append(f"custom_spk:{spk_id}")
        elif task_type == TaskType.VOICE_CLONE_XVEC:
            if spk_embedding is None:
                return None
            arr = spk_embedding.detach().cpu().float().contiguous().numpy()
            key_parts.append(hashlib.sha1(arr.tobytes()).hexdigest()[:16])
        elif task_type == TaskType.VOICE_CLONE_ICL:
            return None
        elif task_type != TaskType.VOICE_DESIGN:
            return None
        return "|".join(key_parts)
