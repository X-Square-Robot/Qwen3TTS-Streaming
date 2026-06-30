"""Tests for scripts/python/trt_fused_io_formats.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PY = REPO_ROOT / "scripts" / "python"
sys.path.insert(0, str(SCRIPTS_PY))

from trt_fused_io_formats import (  # noqa: E402
    fused_input_output_io_format_strings,
    trtexec_precision_args,
)

FIXTURE_MANIFEST = REPO_ROOT / "tools" / "data" / "triton_manifest_custom_1_7b.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))


def test_trtexec_precision_args():
    assert trtexec_precision_args("bf16") == ["--bf16"]
    assert trtexec_precision_args("fp16") == ["--fp16"]
    assert trtexec_precision_args("fp8") == ["--fp8"]
    assert trtexec_precision_args("fp32") == []


def test_fused_io_format_counts_and_int_positions():
    m = _load_fixture()
    m["triton_io_float_dtype"] = "bf16"
    inp, out = fused_input_output_io_format_strings(m)
    in_parts = inp.split(",")
    out_parts = out.split(",")
    n_c2w_in = len(m["code2wav_fused"]["c2w_state_input_names"])
    n_c2w_out = len(m["code2wav_fused"]["c2w_state_output_names"])
    assert len(in_parts) == 12 + n_c2w_in
    assert len(out_parts) == 8 + n_c2w_out
    # input_embeds, position_ids, attention_bias,
    # token_counts(i64), gumbel_noise(fp32), cp_gumbel_noise(fp32),
    # temperature(fp32), penalty(fp32),
    # cache_position(fp32), c2w_attention_bias
    assert in_parts[0] == "bf16:chw"  # input_embeds
    assert in_parts[1] == "int64:chw"  # position_ids
    assert in_parts[2] == "bf16:chw"  # attention_bias
    assert in_parts[3] == "int64:chw"  # token_counts
    assert in_parts[4] == "fp32:chw"  # gumbel_noise
    assert in_parts[5] == "fp32:chw"  # cp_gumbel_noise
    assert in_parts[6] == "fp32:chw"  # temperature
    assert in_parts[7] == "fp32:chw"  # penalty
    assert in_parts[8] == "fp32:chw"  # cache_position
    assert in_parts[9] == "bf16:chw"  # c2w_attention_bias
    # outputs: wav, codec_sum, full_codec(i64), hidden, logits, updated_token_counts(i64)
    assert out_parts[0] == "bf16:chw"  # wav
    assert out_parts[1] == "bf16:chw"  # codec_sum
    assert out_parts[2] == "int64:chw"  # full_codec
    assert out_parts[3] == "bf16:chw"  # hidden
    assert out_parts[4] == "bf16:chw"  # logits
    assert out_parts[5] == "int64:chw"  # updated_token_counts


def test_fused_io_fp32_float_tokens():
    m = _load_fixture()
    m["triton_io_float_dtype"] = "fp32"
    inp, out = fused_input_output_io_format_strings(m)
    assert inp.startswith(
        "fp32:chw,int64:chw,fp32:chw,int64:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,"
    )
    assert "fp32:chw" in out.split(",")[0]


@pytest.mark.skipif(not FIXTURE_MANIFEST.is_file(), reason="fixture missing")
def test_cli_smoke():
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_PY / "trt_fused_io_formats.py"),
            str(FIXTURE_MANIFEST),
            "--emit",
            "all",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 3
    assert "bf16:chw" in lines[0]
    assert "bf16:chw" in lines[1]
    assert lines[2] == "--bf16"
