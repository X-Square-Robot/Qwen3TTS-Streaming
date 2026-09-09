# English comments only.
"""Load Triton deployment manifest (triton_manifest.json)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Version of the ONNX/export graph contract encoded by export_09.  This is
# deliberately distinct from the Triton manifest schema version.
EXPORT_PROTOCOL_VERSION = "v1"
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _valid_artifact_text(value: Any, *, path: bool = False) -> bool:
    """Validate manifest-owned artifact fields before writing a manifest."""

    if not isinstance(value, str) or not value.strip():
        return False
    if not path:
        return bool(_SHA256.fullmatch(value.strip()))
    candidate = Path(value)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _declared_binding_names(value: Any) -> set[str]:
    """Normalize malformed cursor binding declarations for fail-closed checks."""
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    return {name for name in value if isinstance(name, str)}


def load_weights_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_submodel_precisions(manifest: Dict[str, Any]) -> Dict[str, str]:
    """Resolve per-submodel compute precision from manifest.

    Returns a dict with keys 'backbone', 'cp', 'code2wav', each mapping
    to a normalized precision string (fp32, bf16, fp16, fp8).

    If the manifest does not contain per-submodel precision fields,
    all values default to engine_dtype (uniform precision).
    """
    engine_dtype = _normalize_precision(str(manifest.get("engine_dtype", "bf16")))
    return {
        "backbone": _normalize_precision(
            manifest.get("backbone_precision", ""), engine_dtype
        ),
        "cp": _normalize_precision(manifest.get("cp_precision", ""), engine_dtype),
        "code2wav": _normalize_precision(
            manifest.get("code2wav_precision", ""), engine_dtype
        ),
    }


def _normalize_precision(value: str, fallback: str = "bf16") -> str:
    """Normalize a precision string to canonical form."""
    raw = value.strip().lower() if value else ""
    aliases = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float8": "fp8",
    }
    result = aliases.get(raw, raw)
    if result in ("fp32", "bf16", "fp16", "fp8"):
        return result
    return fallback


def weights_to_talker_section(w: Dict[str, Any]) -> Dict[str, int]:
    """Map export weights/config.json keys to manifest talker section."""
    h = int(w.get("talker_hidden_size", 2048))
    n_heads = int(w.get("talker_num_heads", 16))
    kv = int(w.get("talker_num_kv_heads", 8))
    hd = w.get("talker_head_dim")
    if hd is None:
        hd = h // n_heads
    return {
        "hidden_size": h,
        "num_kv_heads": kv,
        "head_dim": int(hd),
        "num_layers": int(w.get("talker_num_layers", 28)),
        "vocab_size": int(w.get("talker_vocab_size", 3072)),
    }


def variant_orchestrator_defaults(
    variant: str,
    model_package_dir: str = "/models/tts_orchestrator/1",
) -> Dict[str, str]:
    if variant.startswith("base-"):
        tts = "base"
        tasks = "voice_clone_icl,voice_clone_xvec"
    elif variant.startswith("custom-"):
        tts = "custom_voice"
        tasks = "custom_voice"
    elif variant.startswith("design-"):
        tts = "voice_design"
        tasks = "voice_design"
    else:
        tts = "unknown"
        tasks = "unknown"
    return {
        "tts_model_type": tts,
        "supported_task_types": tasks,
        "max_decode_steps": "4096",
        "audio_chunk_frames": "25",
        "first_chunk_frames": "4",
        "model_package_dir": model_package_dir,
        "do_sample": "false",
    }


def package_defaults() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "layout": "triton_model_version",
        "model_package_dir": "/models/tts_orchestrator/1",
        "runtime_dir": "runtime",
        "weights_dir": "weights",
        "tokenizer_dir": "tokenizer",
        "manifest": "runtime/triton_manifest.json",
        "runtime_artifacts": {
            "trt": "runtime/model.plan",
            "onnx": "runtime/model.onnx",
        },
        "optional_assets": {},
    }


def load_manifest(
    manifest_path: Path,
    output_repo: Optional[Path] = None,
    model_package_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load JSON manifest.
    Optionally fill talker section from tts_orchestrator/<version>/weights/config.json in output_repo.
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest: Dict[str, Any] = json.load(f)

    variant = manifest.get("variant", "unknown")
    package_model_package_dir = model_package_dir or "/models/tts_orchestrator/1"
    package = manifest.get("package")
    if model_package_dir is None and isinstance(package, dict):
        raw_package_dir = package.get("model_package_dir")
        if raw_package_dir:
            package_model_package_dir = str(raw_package_dir)
    package_version = Path(package_model_package_dir).name or "1"

    if not manifest.get("talker") and output_repo is not None:
        wc = (
            output_repo
            / "tts_orchestrator"
            / package_version
            / "weights"
            / "config.json"
        )
        if wc.is_file():
            manifest["talker"] = weights_to_talker_section(load_weights_config(wc))
            logger.info("Filled manifest.talker from orchestrator weights/config.json")

    # Older exports predate the explicit graph-contract field.  Keep them
    # readable while making the distinction from schema_version explicit.
    manifest.setdefault("export_protocol_version", EXPORT_PROTOCOL_VERSION)

    orch = manifest.get("orchestrator")
    defaults = variant_orchestrator_defaults(variant, package_model_package_dir)
    if orch is None:
        manifest["orchestrator"] = dict(defaults)
    else:
        merged = dict(defaults)
        merged.update(orch)
        manifest["orchestrator"] = merged
    if model_package_dir is not None:
        manifest["orchestrator"]["model_package_dir"] = package_model_package_dir

    package = manifest.get("package")
    if not isinstance(package, dict):
        manifest["package"] = package_defaults()
    else:
        merged_package = package_defaults()
        for key, value in package.items():
            if isinstance(value, dict) and isinstance(merged_package.get(key), dict):
                nested = dict(merged_package[key])
                nested.update(value)
                merged_package[key] = nested
            else:
                merged_package[key] = value
        manifest["package"] = merged_package
    if model_package_dir is not None:
        manifest["package"]["model_package_dir"] = package_model_package_dir

    return manifest


def runtime_optional_assets_for_variant(
    variant: str, engine_mode: str
) -> Dict[str, str]:
    """Return production runtime support assets for a variant."""
    if not (variant.startswith("base-") or variant.startswith("icl-")):
        return {}
    suffix = "engine" if engine_mode == "trt" else "onnx"
    return {
        "speaker_encoder": f"runtime/speaker_encoder.{suffix}",
        "speech_tokenizer_codec_fused": f"runtime/speech_tokenizer_codec_fused.{suffix}",
    }


def build_manifest_for_export(
    variant: str,
    weights_config: Dict[str, Any],
    code2wav_layout: Dict[str, Any],
    engine_mode: str = "trt",
    engine_dtype: str = "bf16",
    triton_io_float_dtype: str = "bf16",
    backbone_precision: str = "",
    cp_precision: str = "",
    code2wav_precision: str = "",
    speech_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a full manifest dict after export (e.g. export_09).

    engine_dtype: TensorRT builder precision (e.g. trtexec --bf16); Phase B reads this for prec flags.
    triton_io_float_dtype: Float tensor I/O for trtexec --inputIOFormats/--outputIOFormats and Triton config.pbtxt.
        ONNX graph remains FP32 (ONNX_EXPORT_DTYPE); TRT may insert reformats at boundaries.
    backbone_precision: Compute precision for backbone (talker_unified + codec_sum) sub-graph.
        Defaults to engine_dtype if empty.
    cp_precision: Compute precision for code_predictor sub-graph.
        Defaults to engine_dtype if empty; set to fp32 to mitigate BF16 numerical sensitivity.
    code2wav_precision: Compute precision for code2wav sub-graph.
        Defaults to engine_dtype if empty.

    The ``architecture`` section provides a complete, self-contained model
    description for the standalone engine.  Priority: manifest > engine.yaml > defaults.
    """
    talker = weights_to_talker_section(weights_config)
    orch = variant_orchestrator_defaults(variant)

    architecture: Dict[str, Any] = {
        "num_layers": talker["num_layers"],
        "hidden_size": talker["hidden_size"],
        "kv_heads": talker["num_kv_heads"],
        "head_dim": talker["head_dim"],
        "codec_vocab_size": talker["vocab_size"],
        "logits_topk": code2wav_layout.get("logits_topk", 50),
        "cp_num_stages": code2wav_layout.get("cp_num_stages", 15),
        "n_c2w_layers": code2wav_layout["num_code2wav_hidden_layers"],
        "c2w_kv_heads": code2wav_layout.get("c2w_kv_heads", 16),
        "c2w_head_dim": code2wav_layout.get("c2w_head_dim", 64),
        "c2w_sliding_window": code2wav_layout.get("c2w_sliding_window", 72),
        "n_c2w_conv_states": len(
            [
                n
                for n in code2wav_layout.get("c2w_state_input_names", [])
                if "conv_state" in n
            ]
        ),
        "n_c2w_transconv_states": len(
            [
                n
                for n in code2wav_layout.get("c2w_state_input_names", [])
                if "transconv" in n
            ]
        ),
        "dtype": engine_dtype,
    }
    native_cursor = code2wav_layout.get("native_cursor")
    if native_cursor:
        # Keep this capability at the top level as well as beside the fused
        # layout so admission/routing can inspect it without understanding
        # every Code2Wav detail.
        architecture["native_cursor"] = dict(native_cursor)

    # Resolve mixed-precision defaults: unspecified fields fall back to engine_dtype
    resolved_backbone_precision = backbone_precision or engine_dtype
    resolved_cp_precision = cp_precision or engine_dtype
    resolved_code2wav_precision = code2wav_precision or engine_dtype

    declared_speech_state = speech_state
    if declared_speech_state is None:
        declared_speech_state = code2wav_layout.get("speech_state")
    if declared_speech_state is not None:
        if not isinstance(declared_speech_state, dict):
            raise ValueError("speech_state manifest section must be an object")
        raw_model_fingerprint = declared_speech_state.get("model_fingerprint", "")
        raw_runtime_fingerprint = declared_speech_state.get("runtime_fingerprint", "")
        model_fingerprint = (
            raw_model_fingerprint.strip()
            if isinstance(raw_model_fingerprint, str)
            else ""
        )
        runtime_fingerprint = (
            raw_runtime_fingerprint.strip()
            if isinstance(raw_runtime_fingerprint, str)
            else ""
        )
        if not model_fingerprint or not runtime_fingerprint:
            raise ValueError(
                "speech_state requires model_fingerprint and runtime_fingerprint"
            )
        speech_state_section = {
            "model_fingerprint": model_fingerprint,
            "runtime_fingerprint": runtime_fingerprint,
        }
        model_contract = declared_speech_state.get("model_contract")
        if model_contract is not None:
            try:
                # Keep the copied standalone config generator independent of
                # the engine package for legacy manifests.  Contract-bearing
                # exports are validated only when this optional field exists.
                from engine.core.speech_state_model import SpeechStateModelContract

                parsed_contract = SpeechStateModelContract.from_mapping(
                    model_contract
                )
            except (ImportError, TypeError, ValueError) as exc:
                raise ValueError(
                    "speech_state.model_contract is invalid"
                ) from exc
            if parsed_contract.model_fingerprint != model_fingerprint:
                raise ValueError(
                    "speech_state.model_contract model_fingerprint does not match "
                    "speech_state.model_fingerprint"
                )
            speech_state_section["model_contract"] = parsed_contract.to_dict()
        bundle = declared_speech_state.get("bundle")
        if bundle is not None:
            if not isinstance(bundle, dict):
                raise ValueError("speech_state.bundle must be an object")
            # The runtime verifier owns file existence and hash checks.  The
            # export manifest must still preserve the complete bundle
            # descriptor instead of silently dropping it.
            speech_state_section["bundle"] = dict(bundle)
    else:
        speech_state_section = None

    manifest = {
        "schema_version": 2,
        "variant": variant,
        "export_protocol_version": EXPORT_PROTOCOL_VERSION,
        "engine_mode": engine_mode,
        "engine_dtype": engine_dtype,
        "triton_io_float_dtype": triton_io_float_dtype,
        "backbone_precision": resolved_backbone_precision,
        "cp_precision": resolved_cp_precision,
        "code2wav_precision": resolved_code2wav_precision,
        "package": {
            "schema_version": 1,
            "layout": "triton_model_version",
            "model_package_dir": "/models/tts_orchestrator/1",
            "runtime_dir": "runtime",
            "weights_dir": "weights",
            "tokenizer_dir": "tokenizer",
            "manifest": "runtime/triton_manifest.json",
            "runtime_artifacts": {
                "trt": "runtime/model.plan",
                "onnx": "runtime/model.onnx",
            },
            "optional_assets": runtime_optional_assets_for_variant(
                variant, engine_mode
            ),
        },
        "engine_profile": {
            "profile_schema_version": 1,
            "engine_mode": engine_mode,
            "engine_dtype": engine_dtype,
            "triton_io_float_dtype": triton_io_float_dtype,
            "backbone_precision": resolved_backbone_precision,
            "cp_precision": resolved_cp_precision,
            "code2wav_precision": resolved_code2wav_precision,
            "builder": "trtexec",
        },
        "architecture": architecture,
        "talker": talker,
        "code2wav_fused": {
            "num_code2wav_hidden_layers": code2wav_layout["num_code2wav_hidden_layers"],
            "c2w_state_input_names": code2wav_layout["c2w_state_input_names"],
            "c2w_state_output_names": code2wav_layout["c2w_state_output_names"],
            "initial_state_shapes": code2wav_layout["initial_state_shapes"],
            "packed_kv": bool(code2wav_layout.get("packed_kv", False)),
            "c2w_kv_heads": int(code2wav_layout.get("c2w_kv_heads", 16)),
            "c2w_head_dim": int(code2wav_layout.get("c2w_head_dim", 64)),
            "c2w_sliding_window": int(code2wav_layout.get("c2w_sliding_window", 72)),
            "logits_topk": int(code2wav_layout.get("logits_topk", 50)),
            "cp_num_stages": int(code2wav_layout.get("cp_num_stages", 15)),
        },
        "native_cursor": dict(native_cursor)
        if native_cursor
        else {"enabled": False},
        "orchestrator": orch,
    }
    if speech_state_section is not None:
        bundle = speech_state_section.get("bundle")
        if bundle is not None:
            if type(bundle.get("schema_version")) is not int or bundle.get("schema_version") != 1:
                raise ValueError("speech_state.bundle schema_version must be 1")
            declared_layout_hash = str(
                bundle.get("code2wav_layout_sha256", "") or ""
            ).strip().lower()
            try:
                from engine.core.speech_state_bundle import code2wav_layout_fingerprint
            except ImportError as exc:
                raise ValueError("speech_state bundle verifier is unavailable") from exc
            expected_layout_hash = code2wav_layout_fingerprint(manifest["code2wav_fused"])
            if not declared_layout_hash:
                bundle["code2wav_layout_sha256"] = expected_layout_hash
            elif declared_layout_hash != expected_layout_hash:
                raise ValueError(
                    "speech_state.bundle code2wav layout hash does not match export layout"
                )
            artifacts = bundle.get("artifacts")
            if not isinstance(artifacts, dict):
                raise ValueError("speech_state.bundle.artifacts must be an object")
            for name in ("model_weights", "runtime_plan"):
                descriptor = artifacts.get(name)
                if (
                    not isinstance(descriptor, dict)
                    or not _valid_artifact_text(descriptor.get("path"), path=True)
                    or not _valid_artifact_text(descriptor.get("sha256"))
                ):
                    raise ValueError(f"speech_state.bundle.{name} artifact is incomplete")
                if name == "model_weights" and (
                    not _valid_artifact_text(descriptor.get("source_path"), path=True)
                    or not _valid_artifact_text(descriptor.get("source_sha256"))
                ):
                    raise ValueError(
                        "speech_state.bundle.model_weights source artifact is incomplete"
                    )
            contract = speech_state_section.get("model_contract") or {}
            if contract.get("cursor_policy") == "migrate":
                try:
                    from engine.core.native_cursor import (
                        CURSOR_RECURRENT_INPUT_BINDINGS,
                        CURSOR_RECURRENT_OUTPUT_BINDINGS,
                    )
                except ImportError as exc:
                    raise ValueError(
                        "native cursor recurrent ABI contract is unavailable"
                    ) from exc
                declared_cursor = native_cursor if isinstance(native_cursor, dict) else {}
                declared_inputs = _declared_binding_names(
                    declared_cursor.get("input_names")
                )
                declared_outputs = _declared_binding_names(
                    declared_cursor.get("output_names")
                )
                if (
                    declared_cursor.get("enabled") is not True
                    or not CURSOR_RECURRENT_INPUT_BINDINGS.issubset(declared_inputs)
                    or not CURSOR_RECURRENT_OUTPUT_BINDINGS.issubset(declared_outputs)
                ):
                    raise ValueError(
                        "speech_state.model_contract migrate requires complete "
                        "native cursor recurrent ABI"
                    )
            if contract.get("cursor_policy") != "disable":
                cursor = artifacts.get("cursor_head")
                if (
                    not isinstance(cursor, dict)
                    or not _valid_artifact_text(cursor.get("path"), path=True)
                    or not _valid_artifact_text(cursor.get("sha256"))
                    or not _valid_artifact_text(cursor.get("source_path"), path=True)
                    or not _valid_artifact_text(cursor.get("source_sha256"))
                ):
                    raise ValueError("speech_state.bundle.cursor_head artifact is incomplete")
        manifest["speech_state"] = speech_state_section
    return manifest
