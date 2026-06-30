#!/usr/bin/env python3
# English comments only.
"""
Generate Triton config.pbtxt files from triton_manifest.json.

Replaces heredoc logic in scripts/bash/lib/triton.sh for assemble_model_repo.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "python"))

from triton_manifest_io import load_manifest

logger = logging.getLogger(__name__)

# Code2wav streaming TRT conv / transconv specs (must match former triton.sh heredoc).
_CONV_SPECS: List[Tuple[str, int, int, int]] = [
    ("conv_state_0", -1, 512, 2),
    ("conv_state_1", -1, 1024, 6),
    ("conv_state_2", -1, 1024, 6),
    ("conv_state_3", -1, 1024, 6),
    ("conv_state_4", -1, 768, 6),
    ("conv_state_5", -1, 768, 18),
    ("conv_state_6", -1, 768, 54),
    ("conv_state_7", -1, 384, 6),
    ("conv_state_8", -1, 384, 18),
    ("conv_state_9", -1, 384, 54),
    ("conv_state_10", -1, 192, 6),
    ("conv_state_11", -1, 192, 18),
    ("conv_state_12", -1, 192, 54),
    ("conv_state_13", -1, 96, 6),
    ("conv_state_14", -1, 96, 18),
    ("conv_state_15", -1, 96, 54),
    ("conv_state_16", -1, 96, 6),
]

_TRANSCONV_SPECS: List[Tuple[str, int, int, int]] = [
    ("transconv_overlap_0", -1, 768, 8),
    ("transconv_overlap_1", -1, 384, 5),
    ("transconv_overlap_2", -1, 192, 4),
    ("transconv_overlap_3", -1, 96, 3),
]


def normalize_engine_dtype(short: str) -> str:
    s = (short or "bf16").lower().strip()
    if s in ("bfloat16",):
        return "bf16"
    if s in ("float16",):
        return "fp16"
    if s in ("float32", "float"):
        return "fp32"
    if s in ("fp8", "float8"):
        return "fp8"
    return s


def to_triton_dtype(short: str) -> str:
    s = normalize_engine_dtype(short)
    mapping = {
        "float": "TYPE_FP32",
        "float32": "TYPE_FP32",
        "fp32": "TYPE_FP32",
        "float16": "TYPE_FP16",
        "fp16": "TYPE_FP16",
        "bfloat16": "TYPE_BF16",
        "bf16": "TYPE_BF16",
        "fp8": "TYPE_FP8",
        "float8": "TYPE_FP8",
        "int32": "TYPE_INT32",
        "int64": "TYPE_INT64",
        "string": "TYPE_STRING",
        "bool": "TYPE_BOOL",
    }
    return mapping.get(s, "TYPE_FP32")


def normalize_triton_io_float_dtype(short: str) -> str:
    """Normalize manifest triton_io_float_dtype / onnx_io_dtype string."""
    s = (short or "fp32").lower().strip()
    if s in ("float32", "float"):
        return "fp32"
    if s in ("bfloat16",):
        return "bf16"
    if s in ("float16",):
        return "fp16"
    if s in ("float8", "fp8"):
        return "fp8"
    if s in ("fp32", "bf16", "fp16", "fp8"):
        return s
    return "fp32"


def triton_io_float_pbtxt_from_manifest(manifest: Dict[str, Any]) -> str:
    """
    Float tensor types for Triton config must match trtexec --inputIOFormats/--outputIOFormats
    (see trt_fused_io_formats.py). Missing key defaults to fp32 for backward compatibility;
    export_09 writes triton_io_float_dtype (default bf16) into triton_manifest.json.
    """
    raw = (
        manifest.get("triton_io_float_dtype") or manifest.get("onnx_io_dtype") or "fp32"
    )
    return to_triton_dtype(normalize_triton_io_float_dtype(str(raw)))


def render_minimal_onnx(model_name: str) -> str:
    return f'''name: "{model_name}"
backend: "onnxruntime"
max_batch_size: 0

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
'''


def render_minimal_trt(model_name: str) -> str:
    return f'''name: "{model_name}"
backend: "tensorrt"
max_batch_size: 0

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
'''


def render_speaker_encoder_trt(engine_dtype: str) -> str:
    ft = to_triton_dtype(engine_dtype)
    return f"""name: "speaker_encoder"
backend: "tensorrt"
max_batch_size: 0

input [
  {{ name: "mel"  data_type: {ft}  dims: [ -1, -1, 128 ] }}
]
output [
  {{ name: "speaker_embedding"  data_type: {ft}  dims: [ -1, 1024 ] }}
]

instance_group [
  {{ count: 1  kind: KIND_GPU  gpus: [ 0 ] }}
]
"""


def render_speech_tokenizer_encoder_trt() -> str:
    return """name: "speech_tokenizer_encoder"
backend: "tensorrt"
max_batch_size: 0

input [
  { name: "waveform"  data_type: TYPE_FP32  dims: [ -1, 1, -1 ] }
]
output [
  { name: "audio_codes"  data_type: TYPE_INT64  dims: [ -1, 16, -1 ] }
]

instance_group [
  { count: 1  kind: KIND_GPU  gpus: [ 0 ] }
]
"""


def render_talker_unified_trt(talker: Dict[str, Any], engine_dtype: str) -> str:
    ft = to_triton_dtype(engine_dtype)
    H = int(talker.get("hidden_size", 2048))
    kv = int(talker.get("num_kv_heads", 8))
    hd = int(talker.get("head_dim", 128))
    nl = int(talker.get("num_layers", 28))
    v = int(talker.get("vocab_size", 3072))

    parts: List[str] = [
        'name: "talker_unified"',
        'backend: "tensorrt"',
        "max_batch_size: 0",
        "",
        "input [",
        f'  {{ name: "input_embeds"  data_type: {ft}  dims: [ -1, -1, {H} ] }}',
        "]",
        "input [",
        '  { name: "position_ids"  data_type: TYPE_INT64  dims: [ -1, 3, -1, 1 ] }',
        "]",
    ]
    for i in range(nl):
        parts.append("input [")
        parts.append(
            f'  {{ name: "past_kv_{i}_k"  data_type: {ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
        )
        parts.append("]")
        parts.append("input [")
        parts.append(
            f'  {{ name: "past_kv_{i}_v"  data_type: {ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
        )
        parts.append("]")
    parts.extend(
        [
            "output [",
            f'  {{ name: "codec_sum"  data_type: {ft}  dims: [ -1, 1, {H} ] }}',
            "]",
            "output [",
            '  { name: "full_codec"  data_type: TYPE_INT64  dims: [ -1, 16 ] }',
            "]",
            "output [",
            f'  {{ name: "hidden"  data_type: {ft}  dims: [ -1, -1, {H} ] }}',
            "]",
            "output [",
            f'  {{ name: "logits"  data_type: {ft}  dims: [ -1, -1, {v} ] }}',
            "]",
        ]
    )
    for i in range(nl):
        parts.append("output [")
        parts.append(
            f'  {{ name: "present_kv_{i}_k"  data_type: {ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
        )
        parts.append("]")
        parts.append("output [")
        parts.append(
            f'  {{ name: "present_kv_{i}_v"  data_type: {ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
        )
        parts.append("]")
    parts.extend(
        [
            "instance_group [",
            "  { count: 1  kind: KIND_GPU  gpus: [ 0 ] }",
            "]",
            "",
        ]
    )
    return "\n".join(parts)


# One codec frame at 24 kHz (matches model_repository/tts_orchestrator SAMPLES_PER_CODEC_FRAME).
_FUSED_WAV_SAMPLES_PER_FRAME = 1920


def _manifest_shape_to_triton_dims(shape: List[int]) -> str:
    """Map manifest initial_state_shapes (batch leading 1) to Triton dims (-1 batch; 0 -> dynamic)."""
    dims: List[str] = []
    for i, x in enumerate(shape):
        if i == 0:
            dims.append("-1")
        elif int(x) == 0:
            dims.append("-1")
        else:
            dims.append(str(int(x)))
    return "[ " + ", ".join(dims) + " ]"


def _c2w_state_shape_to_triton_dims(name: str, shape: List[int]) -> str:
    """
    Map manifest c2w state shapes to Triton dims.
    Keep batch dynamic; for c2w past-kv, force past length dynamic even if manifest keeps
    a concrete cold-start length (usually 1) for orchestrator initialization.
    """
    s = [int(x) for x in shape]
    if ("past_kv" in name or "present_kv" in name) and len(s) >= 3:
        s[2] = 0
    return _manifest_shape_to_triton_dims(s)


def render_talker_code2wav_fused_trt(
    manifest: Dict[str, Any], engine_dtype: str
) -> str:
    """
    Full I/O for TensorRT backend: empty input/output causes
    'failed to specify the dimensions of all input tensors or values of all input shape tensors'.

    Float tensor data_types come from manifest ``triton_io_float_dtype`` (default fp32), which must match
    the deployed ONNX/TRT engine I/O. ``engine_dtype`` / CLI is TensorRT *compute* precision and is not used here.
    """
    io_ft = triton_io_float_pbtxt_from_manifest(manifest)
    _ = engine_dtype
    talker = manifest.get("talker") or {}
    H = int(talker.get("hidden_size", 2048))
    kv = int(talker.get("num_kv_heads", 8))
    hd = int(talker.get("head_dim", 128))
    nl = int(talker.get("num_layers", 28))
    v = int(talker.get("vocab_size", 3072))

    c2w = manifest.get("code2wav_fused")
    if not isinstance(c2w, dict):
        raise ValueError(
            "triton_manifest.json must include code2wav_fused for talker_code2wav_fused TRT config"
        )
    in_names: List[str] = list(c2w.get("c2w_state_input_names") or [])
    init_shapes_raw = c2w.get("initial_state_shapes") or []
    out_names_c2w: List[str] = list(c2w.get("c2w_state_output_names") or [])
    packed_kv = bool(c2w.get("packed_kv"))
    n_c2w_layers = int(c2w.get("num_code2wav_hidden_layers", 8))
    c2w_kv_heads = int(c2w.get("c2w_kv_heads", kv))
    c2w_head_dim = int(c2w.get("c2w_head_dim", hd))
    cp_num_stages = int(
        c2w.get(
            "cp_num_stages", manifest.get("architecture", {}).get("cp_num_stages", 15)
        )
    )

    init_shapes: List[List[int]] = []
    for row in init_shapes_raw:
        if isinstance(row, (list, tuple)):
            init_shapes.append([int(x) for x in row])
        else:
            raise ValueError("initial_state_shapes entries must be lists of integers")

    if len(in_names) != len(init_shapes):
        raise ValueError(
            "code2wav_fused: len(c2w_state_input_names) != len(initial_state_shapes)"
        )
    if len(out_names_c2w) != len(in_names):
        raise ValueError(
            "code2wav_fused: len(c2w_state_output_names) must match c2w_state_input_names"
        )

    parts: List[str] = [
        'name: "talker_code2wav_fused"',
        'backend: "tensorrt"',
        "max_batch_size: 0",
        "",
        "input [",
        f'  {{ name: "input_embeds"  data_type: {io_ft}  dims: [ -1, -1, {H} ] }}',
        "]",
        "input [",
        '  { name: "position_ids"  data_type: TYPE_INT64  dims: [ -1, 3, -1, 1 ] }',
        "]",
        "input [",
        f'  {{ name: "attention_bias"  data_type: {io_ft}  dims: [ -1, 1, -1, -1 ] }}',
        "]",
        "input [",
        f'  {{ name: "token_counts"  data_type: TYPE_INT64  dims: [ -1, {v} ] }}',
        "]",
        "input [",
        '  { name: "gumbel_noise"  data_type: TYPE_FP32  dims: [ -1, 50 ] }',
        "]",
        "input [",
        f'  {{ name: "cp_gumbel_noise"  data_type: TYPE_FP32  dims: [ -1, {cp_num_stages}, 50 ] }}',
        "]",
        "input [",
        '  { name: "temperature"  data_type: TYPE_FP32  dims: [ -1, 1 ] }',
        "]",
        "input [",
        '  { name: "penalty"  data_type: TYPE_FP32  dims: [ -1, 1 ] }',
        "]",
        "input [",
        '  { name: "cache_position"  data_type: TYPE_FP32  dims: [ -1, -1 ] }',
        "]",
        "input [",
        f'  {{ name: "c2w_attention_bias"  data_type: {io_ft}  dims: [ -1, 1, -1, -1 ] }}',
        "]",
    ]
    if packed_kv:
        parts.append("input [")
        parts.append(
            f'  {{ name: "talker_past_kv"  data_type: {io_ft}  dims: [ -1, {nl * 2}, {kv}, -1, {hd} ] }}'
        )
        parts.append("]")
        parts.append("input [")
        parts.append(
            f'  {{ name: "c2w_past_kv"  data_type: {io_ft}  dims: [ -1, {n_c2w_layers * 2}, {c2w_kv_heads}, -1, {c2w_head_dim} ] }}'
        )
        parts.append("]")
    else:
        for i in range(nl):
            parts.append("input [")
            parts.append(
                f'  {{ name: "past_kv_{i}_k"  data_type: {io_ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
            )
            parts.append("]")
            parts.append("input [")
            parts.append(
                f'  {{ name: "past_kv_{i}_v"  data_type: {io_ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
            )
            parts.append("]")
    for name, shp in zip(in_names, init_shapes):
        dims = _c2w_state_shape_to_triton_dims(name, shp)
        parts.append("input [")
        parts.append(f'  {{ name: "{name}"  data_type: {io_ft}  dims: {dims} }}')
        parts.append("]")
    parts.extend(
        [
            "output [",
            f'  {{ name: "wav"  data_type: {io_ft}  dims: [ -1, {_FUSED_WAV_SAMPLES_PER_FRAME} ] }}',
            "]",
            "output [",
            f'  {{ name: "codec_sum"  data_type: {io_ft}  dims: [ -1, 1, {H} ] }}',
            "]",
            "output [",
            '  { name: "full_codec"  data_type: TYPE_INT64  dims: [ -1, 16 ] }',
            "]",
            "output [",
            f'  {{ name: "hidden"  data_type: {io_ft}  dims: [ -1, -1, {H} ] }}',
            "]",
            "output [",
            f'  {{ name: "logits"  data_type: {io_ft}  dims: [ -1, -1, {v} ] }}',
            "]",
            "output [",
            f'  {{ name: "updated_token_counts"  data_type: TYPE_INT64  dims: [ -1, {v} ] }}',
            "]",
        ]
    )
    if packed_kv:
        parts.append("output [")
        parts.append(
            f'  {{ name: "talker_new_kv"  data_type: {io_ft}  dims: [ -1, {nl * 2}, {kv}, -1, {hd} ] }}'
        )
        parts.append("]")
        parts.append("output [")
        parts.append(
            f'  {{ name: "c2w_new_kv"  data_type: {io_ft}  dims: [ -1, {n_c2w_layers * 2}, {c2w_kv_heads}, -1, {c2w_head_dim} ] }}'
        )
        parts.append("]")
    else:
        for i in range(nl):
            parts.append("output [")
            parts.append(
                f'  {{ name: "present_kv_{i}_k"  data_type: {io_ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
            )
            parts.append("]")
            parts.append("output [")
            parts.append(
                f'  {{ name: "present_kv_{i}_v"  data_type: {io_ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
            )
            parts.append("]")
    for out_name, shp in zip(out_names_c2w, init_shapes):
        dims = _c2w_state_shape_to_triton_dims(out_name, shp)
        parts.append("output [")
        parts.append(f'  {{ name: "{out_name}"  data_type: {io_ft}  dims: {dims} }}')
        parts.append("]")
    parts.extend(
        [
            "instance_group [",
            "  { count: 1  kind: KIND_GPU  gpus: [ 0 ] }",
            "]",
            "",
        ]
    )
    return "\n".join(parts)


def _dims_pbtxt(tup: Tuple[int, ...]) -> str:
    return ", ".join(str(x) for x in tup)


def render_code2wav_streaming(engine_mode: str, engine_dtype: str) -> str:
    if engine_mode == "onnx":
        return """name: "code2wav"
backend: "onnxruntime"
max_batch_size: 0

input [
  { name: "codes"  data_type: TYPE_INT64  dims: [ -1, 16, 4 ] }
]
input [
  { name: "cache_position"  data_type: TYPE_FP32  dims: [ -1, 4 ] }
]

instance_group [
  { count: 1  kind: KIND_GPU  gpus: [ 0 ] }
]
"""
    ft = to_triton_dtype(engine_dtype)
    parts: List[str] = [
        'name: "code2wav"',
        'backend: "tensorrt"',
        "max_batch_size: 0",
        "",
        "input [",
        '  { name: "codes"  data_type: TYPE_INT64  dims: [ -1, 16, 4 ] }',
        "]",
        "input [",
        '  { name: "cache_position"  data_type: TYPE_FP32  dims: [ -1, 4 ] }',
        "]",
    ]
    for i in range(8):
        parts.append("input [")
        parts.append(
            f'  {{ name: "past_kv_{i}_k"  data_type: {ft}  dims: [ -1, 16, -1, 64 ] }}'
        )
        parts.append("]")
        parts.append("input [")
        parts.append(
            f'  {{ name: "past_kv_{i}_v"  data_type: {ft}  dims: [ -1, 16, -1, 64 ] }}'
        )
        parts.append("]")
    for name, a, b, c in _CONV_SPECS:
        parts.append("input [")
        parts.append(
            f'  {{ name: "{name}"  data_type: {ft}  dims: [ {_dims_pbtxt((a, b, c))} ] }}'
        )
        parts.append("]")
    for name, a, b, c in _TRANSCONV_SPECS:
        parts.append("input [")
        parts.append(
            f'  {{ name: "{name}"  data_type: {ft}  dims: [ {_dims_pbtxt((a, b, c))} ] }}'
        )
        parts.append("]")
    parts.append("output [")
    parts.append(f'  {{ name: "wav"  data_type: {ft}  dims: [ -1, 7680 ] }}')
    parts.append("]")
    for i in range(8):
        parts.append("output [")
        parts.append(
            f'  {{ name: "present_kv_{i}_k"  data_type: {ft}  dims: [ -1, 16, -1, 64 ] }}'
        )
        parts.append("]")
        parts.append("output [")
        parts.append(
            f'  {{ name: "present_kv_{i}_v"  data_type: {ft}  dims: [ -1, 16, -1, 64 ] }}'
        )
        parts.append("]")
    for name, a, b, c in _CONV_SPECS:
        parts.append("output [")
        parts.append(
            f'  {{ name: "new_{name}"  data_type: {ft}  dims: [ {_dims_pbtxt((a, b, c))} ] }}'
        )
        parts.append("]")
    for name, a, b, c in _TRANSCONV_SPECS:
        parts.append("output [")
        parts.append(
            f'  {{ name: "new_{name}"  data_type: {ft}  dims: [ {_dims_pbtxt((a, b, c))} ] }}'
        )
        parts.append("]")
    parts.extend(
        [
            "instance_group [",
            "  { count: 1  kind: KIND_GPU  gpus: [ 0 ] }",
            "]",
            "",
        ]
    )
    return "\n".join(parts)


def _profile_int(profile: Dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(profile.get(key, 0))
    except (TypeError, ValueError):
        value = 0
    return value if value > 0 else default


def render_orchestrator(
    variant: str,
    orch: Dict[str, str],
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    tts = orch.get("tts_model_type", "unknown")
    tasks = orch.get("supported_task_types", "unknown")
    max_decode = orch.get("max_decode_steps", "4096")
    achunk = orch.get("audio_chunk_frames", "25")
    fchunk = orch.get("first_chunk_frames", "4")
    pdir = orch.get("model_package_dir", "/models/tts_orchestrator/1")
    do_sample = orch.get("do_sample", "false")
    profile = profile or {}
    max_batch_slots = _profile_int(profile, "max_batch_size", 128)
    engine_max_decode_len = _profile_int(profile, "max_seq_len", 512)
    return f'''name: "tts_orchestrator"
backend: "python"
max_batch_size: 0

model_transaction_policy {{
  decoupled: true
}}

input [
  {{
    name: "request"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "audio_chunk"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "event_type"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "event_json"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "is_final"
    data_type: TYPE_BOOL
    dims: [ 1 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

parameters: {{
  key: "model_variant"
  value: {{ string_value: "{variant}" }}
}}
parameters: {{
  key: "tts_model_type"
  value: {{ string_value: "{tts}" }}
}}
parameters: {{
  key: "supported_task_types"
  value: {{ string_value: "{tasks}" }}
}}
parameters: {{
  key: "model_package_dir"
  value: {{ string_value: "{pdir}" }}
}}
parameters: {{
  key: "max_batch_slots"
  value: {{ string_value: "{max_batch_slots}" }}
}}
parameters: {{
  key: "engine_max_decode_len"
  value: {{ string_value: "{engine_max_decode_len}" }}
}}
parameters: {{
  key: "do_sample"
  value: {{ string_value: "{do_sample}" }}
}}
parameters: {{
  key: "max_decode_steps"
  value: {{ string_value: "{max_decode}" }}
}}
parameters: {{
  key: "audio_chunk_frames"
  value: {{ string_value: "{achunk}" }}
}}
parameters: {{
  key: "first_chunk_frames"
  value: {{ string_value: "{fchunk}" }}
}}
'''


def render_orchestrator_http(variant: str, orch: Dict[str, str]) -> str:
    tts = orch.get("tts_model_type", "unknown")
    tasks = orch.get("supported_task_types", "unknown")
    return f'''name: "tts_orchestrator_http"
backend: "python"
max_batch_size: 0

input [
  {{
    name: "request"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "audio_chunk"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "event_type"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "event_json"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "is_final"
    data_type: TYPE_BOOL
    dims: [ 1 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]

parameters: {{
  key: "model_variant"
  value: {{ string_value: "{variant}" }}
}}
parameters: {{
  key: "tts_model_type"
  value: {{ string_value: "{tts}" }}
}}
parameters: {{
  key: "supported_task_types"
  value: {{ string_value: "{tasks}" }}
}}
parameters: {{
  key: "target_model"
  value: {{ string_value: "tts_orchestrator" }}
}}
parameters: {{
  key: "bls_timeout_ms"
  value: {{ string_value: "600000" }}
}}
'''


def _manifest_model_version(manifest: Dict[str, Any]) -> str:
    package = manifest.get("package")
    package_dir = ""
    if isinstance(package, dict):
        package_dir = str(package.get("model_package_dir") or "")
    if not package_dir:
        orch = manifest.get("orchestrator") or {}
        if isinstance(orch, dict):
            package_dir = str(orch.get("model_package_dir") or "")
    if not package_dir:
        package_dir = "/models/tts_orchestrator/1"

    version = Path(package_dir).name.strip()
    return version if version else "1"


def model_has_engine(model_dir: Path, model_version: str) -> Tuple[bool, bool]:
    """Returns (has_plan, has_onnx)."""
    version_dir = model_dir / model_version
    return (version_dir / "model.plan").is_file(), (
        version_dir / "model.onnx"
    ).is_file()


def generate_configs(
    manifest: Dict[str, Any],
    output_repo: Path,
    engine_mode: str,
    engine_dtype: Optional[str] = None,
) -> None:
    emode = engine_mode.lower().strip()
    ed = normalize_engine_dtype(engine_dtype or manifest.get("engine_dtype") or "bf16")
    variant = str(manifest.get("variant", "unknown"))
    talker = manifest.get("talker") or {}
    orch = manifest.get("orchestrator") or {}
    profile = manifest.get("engine_profile") or {}
    model_version = _manifest_model_version(manifest)

    models = [
        "speaker_encoder",
        "speech_tokenizer_codec_fused",
        "talker_code2wav_fused",
        "speech_tokenizer_encoder",
        "talker_unified",
        "code2wav",
    ]

    for name in models:
        mdir = output_repo / name
        if not mdir.is_dir():
            continue
        has_plan, has_onnx = model_has_engine(mdir, model_version)
        if not has_plan and not has_onnx:
            continue

        text: str
        if name == "speaker_encoder":
            if emode == "trt" and has_plan:
                text = render_speaker_encoder_trt(ed)
            else:
                text = render_minimal_onnx(name)
        elif name == "speech_tokenizer_encoder":
            if emode == "trt" and has_plan:
                text = render_speech_tokenizer_encoder_trt()
            else:
                text = render_minimal_onnx(name)
        elif name == "talker_unified":
            if emode == "trt" and has_plan:
                text = render_talker_unified_trt(talker, ed)
            else:
                text = render_minimal_onnx(name)
        elif name == "code2wav":
            text = render_code2wav_streaming(emode, ed)
        elif name == "talker_code2wav_fused":
            if emode == "trt" and has_plan:
                text = render_talker_code2wav_fused_trt(manifest, ed)
            else:
                text = render_minimal_onnx(name)
        else:
            # speech_tokenizer_codec_fused (and any future stubs)
            if emode == "trt" and has_plan:
                text = render_minimal_trt(name)
            else:
                text = render_minimal_onnx(name)

        cfg_path = mdir / "config.pbtxt"
        cfg_path.write_text(text, encoding="utf-8")
        logger.info("Wrote %s", cfg_path)

    orch_dir = output_repo / "tts_orchestrator"
    if orch_dir.is_dir():
        otxt = render_orchestrator(variant, orch, profile)
        (orch_dir / "config.pbtxt").write_text(otxt, encoding="utf-8")
        logger.info("Wrote %s", orch_dir / "config.pbtxt")

    orch_http_dir = output_repo / "tts_orchestrator_http"
    if orch_http_dir.is_dir():
        otxt = render_orchestrator_http(variant, orch)
        (orch_http_dir / "config.pbtxt").write_text(otxt, encoding="utf-8")
        logger.info("Wrote %s", orch_http_dir / "config.pbtxt")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, required=True, help="Path to triton_manifest.json"
    )
    p.add_argument(
        "--output-repo",
        type=Path,
        required=True,
        help="Assembled model_repository root",
    )
    p.add_argument(
        "--engine-mode",
        choices=["onnx", "trt"],
        required=True,
        help="Must match assembled engine files",
    )
    p.add_argument(
        "--engine-dtype",
        default=None,
        help="Override manifest engine_dtype (bf16|fp16|fp32)",
    )
    args = p.parse_args()

    manifest = load_manifest(args.manifest, output_repo=args.output_repo)
    mmode = str(manifest.get("engine_mode", args.engine_mode)).lower()
    if mmode != args.engine_mode:
        logger.warning(
            "CLI engine_mode=%s differs from manifest engine_mode=%s (using CLI)",
            args.engine_mode,
            mmode,
        )

    generate_configs(
        manifest,
        args.output_repo,
        args.engine_mode,
        engine_dtype=args.engine_dtype,
    )


if __name__ == "__main__":
    main()
