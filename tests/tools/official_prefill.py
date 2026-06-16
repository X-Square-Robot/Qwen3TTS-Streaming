"""
Official-style prefill builder: replicates model.generate() prefill logic (L2086-2232)
for prototype parity and reference generation.

Supports both streaming (non_streaming_mode=False) and non-streaming
(non_streaming_mode=True) modes, matching the official code exactly.

Use when model has talker_config (e.g. loaded from HuggingFace TTS model dir).
"""

from typing import Optional

import torch


def build_prefill_like_official(
    model,
    input_id,
    language,
    speaker,
    device,
    *,
    instruct_ids: Optional[torch.Tensor] = None,
    non_streaming_mode=False,
):
    """
    Build talker prefill and trailing_text_hidden exactly like model.generate() L2086-2232.

    When non_streaming_mode=True (VoiceDesign default): all text tokens are folded into
    the prefill embedding; trailing_list contains only [tts_pad_embed].
    When non_streaming_mode=False (streaming): only the first text token is in prefill;
    remaining text tokens form trailing_list for step-wise injection during decode.

    `instruct_ids` must already be wrapped in the official user template
    (`<|im_start|>user ... <|im_end|>`), matching the high-level API path.

    Returns (talker_input_embed [1, S, H], trailing_list list[Tensor[1,1,H]]).
    """
    talker = model.talker
    cfg = model.config
    tc = getattr(cfg, "talker_config", None)
    if tc is None:
        raise RuntimeError("Model has no talker_config; cannot build official-style prefill.")

    input_id = input_id.to(device)
    speaker = speaker or ""
    if speaker == "" or speaker is None:
        speaker_embed = None
    else:
        spk_id = getattr(tc, "spk_id", {}) or {}
        if speaker.lower() not in spk_id:
            raise NotImplementedError(f"Speaker {speaker} not implemented")
        spk_id = spk_id[speaker.lower()]
        speaker_embed = talker.get_input_embeddings()(
            torch.tensor(spk_id, device=device, dtype=input_id.dtype)
        )

    language_id = None
    if language and language.lower() != "auto":
        codec_lang = getattr(tc, "codec_language_id", {}) or {}
        if language.lower() not in codec_lang:
            raise NotImplementedError(f"Language {language} not implemented")
        language_id = codec_lang[language.lower()]

    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor(
                [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                device=device,
                dtype=input_id.dtype,
            )
        )
    ).chunk(3, dim=1)

    if language_id is None:
        codec_prefill_list = [[
            tc.codec_nothink_id,
            tc.codec_think_bos_id,
            tc.codec_think_eos_id,
        ]]
    else:
        codec_prefill_list = [[
            tc.codec_think_id,
            tc.codec_think_bos_id,
            language_id,
            tc.codec_think_eos_id,
        ]]

    codec_input_emebdding_0 = talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, device=device, dtype=input_id.dtype)
    )
    codec_input_emebdding_1 = talker.get_input_embeddings()(
        torch.tensor(
            [[tc.codec_pad_id, tc.codec_bos_id]],
            device=device,
            dtype=input_id.dtype,
        )
    )
    if speaker_embed is None:
        codec_input_emebdding = torch.cat(
            [codec_input_emebdding_0, codec_input_emebdding_1], dim=1
        )
    else:
        codec_input_emebdding = torch.cat(
            [
                codec_input_emebdding_0,
                speaker_embed.view(1, 1, -1),
                codec_input_emebdding_1,
            ],
            dim=1,
        )

    # role: <|im_start|>assistant\n  (first 3 tokens)
    _talker_input_embed_role = talker.text_projection(
        talker.get_text_embeddings()(input_id[:, :3])
    )
    _talker_input_embed = torch.cat(
        (
            tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
            tts_bos_embed,
        ),
        dim=1,
    ) + codec_input_emebdding[:, :-1]
    talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)
    if instruct_ids is not None:
        instruct_ids = instruct_ids.to(device=device, dtype=input_id.dtype)
        if instruct_ids.dim() == 1:
            instruct_ids = instruct_ids.unsqueeze(0)
        instruct_embed = talker.text_projection(
            talker.get_text_embeddings()(instruct_ids)
        )
        talker_input_embed = torch.cat((instruct_embed, talker_input_embed), dim=1)

    # Append first text token + codec_bos
    talker_input_embed = torch.cat(
        [
            talker_input_embed,
            talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4]))
            + codec_input_emebdding[:, -1:],
        ],
        dim=1,
    )

    if non_streaming_mode:
        # L2203-2227: fold ALL text into prefill; trailing = pad only
        talker_input_embed = talker_input_embed[:, :-1]  # drop the first-text token we just added
        text_part = input_id[:, 3:-5]  # all text tokens (excluding role prefix and tail tokens)
        n_text = text_part.shape[1]
        text_embed = torch.cat(
            (
                talker.text_projection(talker.get_text_embeddings()(text_part)),
                tts_eos_embed,
            ),
            dim=1,
        )
        codec_pad_ids = torch.full(
            (1, n_text + 1),
            tc.codec_pad_id,
            device=device,
            dtype=input_id.dtype,
        )
        codec_pad_embed = talker.get_input_embeddings()(codec_pad_ids)
        talker_input_embed = torch.cat(
            [
                talker_input_embed,
                text_embed + codec_pad_embed,
                tts_pad_embed + talker.get_input_embeddings()(
                    torch.tensor([[tc.codec_bos_id]], device=device, dtype=input_id.dtype)
                ),
            ],
            dim=1,
        )
        trailing_list = [tts_pad_embed.clone()]
    else:
        # L2228-2232: streaming — only first text in prefill, rest as trailing
        mid = input_id[:, 4:-5] if input_id.shape[1] > 9 else input_id[:, :0]
        if mid.shape[1] > 0:
            mid_embed = talker.text_projection(talker.get_text_embeddings()(mid))
            trailing_text_hidden = torch.cat((mid_embed, tts_eos_embed), dim=1)
        else:
            trailing_text_hidden = tts_eos_embed
        n_trailing = trailing_text_hidden.shape[1]
        trailing_list = [
            trailing_text_hidden[:, i : i + 1, :].clone()
            for i in range(n_trailing)
        ]

    return talker_input_embed, trailing_list
