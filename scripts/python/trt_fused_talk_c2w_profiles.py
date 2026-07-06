#!/usr/bin/env python3
"""
Emit trtexec --minShapes / --optShapes / --maxShapes for talker_code2wav_fused.onnx.
Print three lines: MIN, OPT, MAX (comma-separated, no spaces).

Packed KV format: talker KV and C2W KV are each a single 5-D tensor instead
of 2*L individual tensors.  Conv/transconv states remain individual.
"""

from __future__ import annotations

import sys


def c2w_conv_transconv_specs(bmax: str):
    """Conv + transconv state shape specs (heterogeneous, batch-only dynamic)."""
    specs = []
    conv = [
        ("conv_state_0", "1x512x2"),
        ("conv_state_1", "1x1024x6"),
        ("conv_state_2", "1x1024x6"),
        ("conv_state_3", "1x1024x6"),
        ("conv_state_4", "1x768x6"),
        ("conv_state_5", "1x768x18"),
        ("conv_state_6", "1x768x54"),
        ("conv_state_7", "1x384x6"),
        ("conv_state_8", "1x384x18"),
        ("conv_state_9", "1x384x54"),
        ("conv_state_10", "1x192x6"),
        ("conv_state_11", "1x192x18"),
        ("conv_state_12", "1x192x54"),
        ("conv_state_13", "1x96x6"),
        ("conv_state_14", "1x96x18"),
        ("conv_state_15", "1x96x54"),
        ("conv_state_16", "1x96x6"),
    ]
    for name, s in conv:
        rest = s[2:]
        specs.append((name, s, s, f"{bmax}x{rest}"))
    tc = [
        ("transconv_overlap_0", "1x768x8"),
        ("transconv_overlap_1", "1x384x5"),
        ("transconv_overlap_2", "1x192x4"),
        ("transconv_overlap_3", "1x96x3"),
    ]
    for name, s in tc:
        rest = s[2:]
        specs.append((name, s, s, f"{bmax}x{rest}"))
    return specs


LOGITS_TOPK = 50
VOCAB_SIZE = 3072
C2W_KV_HEADS = 16
C2W_HEAD_DIM = 64
C2W_SLIDING_WINDOW = 72
CP_NUM_STAGES = 15


def compute_fused_profiles(
    H,
    KV,
    HD,
    NL,
    Bmax,
    max_in="128",
    max_seq="512",
    n_c2w=8,
    n_cp=CP_NUM_STAGES,
):
    """Return (min, opt, max) trtexec shape strings for talker_code2wav_fused.

    Single source of truth for the fused-engine optimization profile.  The
    ``main()`` CLI below and ``scripts/bash/build_engines.sh`` both call this,
    so the shape logic is never duplicated (and never drifts).
    """
    nl = int(NL)
    n_c2w = int(n_c2w)
    n_cp = int(n_cp)
    Bopt = "1"
    opt_spast = "128"
    V = VOCAB_SIZE
    K = LOGITS_TOPK

    # Packed talker KV: [B, num_layers*2, kv_heads, S_past, head_dim]
    talker_kv_dim1 = nl * 2

    parts_min = [
        f"input_embeds:1x1x{H}",
        "position_ids:1x3x1x1",
        "attention_bias:1x1x1x1",
        f"token_counts:1x{V}",
        f"gumbel_noise:1x{K}",
        f"cp_gumbel_noise:1x{n_cp}x{K}",
        "temperature:1x1",
        "penalty:1x1",
        "cache_position:1x1",
        "c2w_attention_bias:1x1x1x2",
        f"talker_past_kv:1x{talker_kv_dim1}x{KV}x0x{HD}",
        f"c2w_past_kv:1x{n_c2w * 2}x{C2W_KV_HEADS}x1x{C2W_HEAD_DIM}",
    ]
    parts_opt = [
        f"input_embeds:{Bopt}x1x{H}",
        f"position_ids:{Bopt}x3x1x1",
        f"attention_bias:{Bopt}x1x1x{int(opt_spast) + 1}",
        f"token_counts:{Bopt}x{V}",
        f"gumbel_noise:{Bopt}x{K}",
        f"cp_gumbel_noise:{Bopt}x{n_cp}x{K}",
        f"temperature:{Bopt}x1",
        f"penalty:{Bopt}x1",
        f"cache_position:{Bopt}x1",
        f"c2w_attention_bias:{Bopt}x1x1x5",
        f"talker_past_kv:{Bopt}x{talker_kv_dim1}x{KV}x{opt_spast}x{HD}",
        f"c2w_past_kv:{Bopt}x{n_c2w * 2}x{C2W_KV_HEADS}x4x{C2W_HEAD_DIM}",
    ]
    parts_max = [
        f"input_embeds:{Bmax}x{max_in}x{H}",
        f"position_ids:{Bmax}x3x{max_in}x1",
        f"attention_bias:{Bmax}x1x{max_in}x{int(max_seq) + int(max_in)}",
        f"token_counts:{Bmax}x{V}",
        f"gumbel_noise:{Bmax}x{K}",
        f"cp_gumbel_noise:{Bmax}x{n_cp}x{K}",
        f"temperature:{Bmax}x1",
        f"penalty:{Bmax}x1",
        f"cache_position:{Bmax}x1",
        f"c2w_attention_bias:{Bmax}x1x1x{C2W_SLIDING_WINDOW}",
        f"talker_past_kv:{Bmax}x{talker_kv_dim1}x{KV}x{max_seq}x{HD}",
        f"c2w_past_kv:{Bmax}x{n_c2w * 2}x{C2W_KV_HEADS}x{C2W_SLIDING_WINDOW - 1}x{C2W_HEAD_DIM}",
    ]

    for name, smin, sopt, smax in c2w_conv_transconv_specs(str(Bmax)):
        parts_min.append(f"c2w_{name}:{smin}")
        parts_opt.append(f"c2w_{name}:{sopt}")
        parts_max.append(f"c2w_{name}:{smax}")

    return ",".join(parts_min), ",".join(parts_opt), ",".join(parts_max)


def compute_fused_decode_profiles(
    H,
    KV,
    HD,
    NL,
    Bmax,
    max_seq="512",
    n_c2w=8,
    n_cp=CP_NUM_STAGES,
):
    """Return (min, opt, max) shape strings for the decode-only profile.

    Emitted as optimization profile 1 of the fused engine.  Sequence length is
    pinned to 1, so the profile's activation scratch is a fraction of profile
    0's (1.2 vs 7.6 GiB on the 128x512 1.7b build) — cheap enough for the
    CUDA-graph decode path to afford a dedicated execution context.  opt is
    the high-concurrency steady state (full batch, mid KV, warm c2w window).
    """
    nl = int(NL)
    n_c2w = int(n_c2w)
    n_cp = int(n_cp)
    V = VOCAB_SIZE
    K = LOGITS_TOPK
    talker_kv_dim1 = nl * 2
    c2w_kv_dim1 = n_c2w * 2
    c2w_warm = C2W_SLIDING_WINDOW - 1

    def parts(batch, s_past, c2w_kv_len):
        return ",".join(
            [
                f"input_embeds:{batch}x1x{H}",
                f"position_ids:{batch}x3x1x1",
                f"attention_bias:{batch}x1x1x{int(s_past) + 1}",
                f"token_counts:{batch}x{V}",
                f"gumbel_noise:{batch}x{K}",
                f"cp_gumbel_noise:{batch}x{n_cp}x{K}",
                f"temperature:{batch}x1",
                f"penalty:{batch}x1",
                f"cache_position:{batch}x1",
                f"c2w_attention_bias:{batch}x1x1x{int(c2w_kv_len) + 1}",
                f"talker_past_kv:{batch}x{talker_kv_dim1}x{KV}x{s_past}x{HD}",
                f"c2w_past_kv:{batch}x{c2w_kv_dim1}x{C2W_KV_HEADS}x{c2w_kv_len}x{C2W_HEAD_DIM}",
            ]
            + [
                f"c2w_{name}:{batch}x{spec_min[2:]}"
                for name, spec_min, _, _ in c2w_conv_transconv_specs(str(batch))
            ]
        )

    smin = parts(1, 0, 1)
    sopt = parts(Bmax, 128, c2w_warm)
    smax = parts(Bmax, max_seq, c2w_warm)
    return smin, sopt, smax


def main():
    if len(sys.argv) < 6:
        print(
            "Usage: trt_fused_talk_c2w_profiles.py H KV_HEADS HEAD_DIM NUM_LAYERS MAX_BATCH "
            "[MAX_INPUT_LEN] [MAX_SEQ_LEN] [NUM_C2W_DECODER_LAYERS] [CP_NUM_STAGES]",
            file=sys.stderr,
        )
        sys.exit(1)
    H, KV, HD, NL, Bmax = sys.argv[1:6]
    max_in = sys.argv[6] if len(sys.argv) > 6 else "128"
    max_seq = sys.argv[7] if len(sys.argv) > 7 else "512"
    n_c2w = int(sys.argv[8]) if len(sys.argv) > 8 else 8
    n_cp = int(sys.argv[9]) if len(sys.argv) > 9 else CP_NUM_STAGES

    smin, sopt, smax = compute_fused_profiles(
        H, KV, HD, NL, Bmax, max_in, max_seq, n_c2w, n_cp
    )
    print(smin)
    print(sopt)
    print(smax)
    # Lines 4-6: decode-only profile (profile 1).  Callers that only read the
    # first three lines are unaffected.
    dmin, dopt, dmax = compute_fused_decode_profiles(
        H, KV, HD, NL, Bmax, max_seq, n_c2w, n_cp
    )
    print(dmin)
    print(dopt)
    print(dmax)


if __name__ == "__main__":
    main()
