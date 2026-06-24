"""Triton Inference Server deployment manager — Python replacement for ``scripts/bash/lib/triton.sh``.

Provides the :class:`TritonManager` class to assemble the Triton model
repository from exported artifacts, validate the assembled repo, resolve
NGC deployment images, and manage the Triton container lifecycle (health
check / stop).

Deprecation notice
------------------
The Bash library ``scripts/bash/lib/triton.sh`` is superseded by this module.
The shell version remains available for backward compatibility but should be
considered deprecated.  New code should import from
:mod:`qwen3tts_tools.triton` directly.

Usage from CLI::

    python -m qwen3tts_tools.triton assemble --variant custom-1.7b
    python -m qwen3tts_tools.triton validate
    python -m qwen3tts_tools.triton resolve-image
    python -m qwen3tts_tools.triton health --port 8000
    python -m qwen3tts_tools.triton stop --container qwen3-tts-triton

Usage from Python::

    from qwen3tts_tools.triton import TritonManager

    mgr = TritonManager()
    mgr.assemble_model_repo(exported_dir, "custom-1.7b", model_repo_dir, "trt", 1)
    mgr.validate_model_repo(model_repo_dir, 1)
    image = mgr.resolve_deploy_image()
    ok = mgr.health_check(port=8000, timeout=60)
    mgr.stop("qwen3-tts-triton")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from qwen3tts_tools.common import (
    REPO_ROOT,
    WORKSPACE_DIR,
    DEFAULT_TRITON_HTTP_MODEL,
    DEFAULT_TRITON_MODEL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Models that must be present in every valid Triton model repository.
_REQUIRED_MODELS = (
    "tts_orchestrator",
    "tts_orchestrator_http",
)

# Tokenizer files to copy into the orchestrator package.
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "config.json",
    "generation_config.json",
)

# Variant name -> HuggingFace model directory name mapping for tokenizer lookup.
_VARIANT_MODEL_DIRS: dict[str, str] = {
    "design-1.7b": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom-1.7b": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "base-1.7b": "Qwen3-TTS-12Hz-1.7B-Base",
    "custom-0.6b": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "base-0.6b": "Qwen3-TTS-12Hz-0.6B-Base",
}


# ---------------------------------------------------------------------------
#  Structured data
# ---------------------------------------------------------------------------

@dataclass
class ModelRepoValidation:
    """Result of validating a Triton model repository.

    Attributes:
        valid: ``True`` if all required models and assets are present.
        errors: List of error messages for missing required assets.
        warnings: List of non-fatal warning messages.
        checked_models: Names of the models that were checked.
    """

    valid: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_models: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------

def resolve_model_version(raw: Optional[str] = None) -> int:
    """Normalize the Triton model version number.

    Precedence: explicit *raw* argument > ``MODEL_VERSION`` env var >
    ``ENGINE_MODEL_VERSION`` env var > ``1``.

    Args:
        raw: Explicit version string.  When ``None``, falls back to
             environment variables then the default.

    Returns:
        Positive integer model version.

    Raises:
        ValueError: If the resolved value is not a positive integer.
    """
    value = raw
    if value is None:
        value = os.environ.get("MODEL_VERSION") or os.environ.get("ENGINE_MODEL_VERSION") or "1"
    value = value.strip()
    if value.isdigit() and int(value) > 0:
        return int(value)
    raise ValueError(f"Invalid model version: {value!r}. Use a positive integer like 1 or 2.")


def _copy(src: Path, dst: Path) -> None:
    """Copy *src* to *dst*, always using full copy (no symlinks).

    The assembled model_repository is mounted into Docker containers where
    host-absolute symlinks would be dangling, so we always copy.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy directory tree *src* to *dst* (always a full copy, no symlinks)."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)


def _resolve_model_src(base: Path, engine_mode: str) -> Optional[Path]:
    """Select the correct model artifact (``.engine`` or ``.onnx``).

    Args:
        base: Base path without extension (e.g.
              ``workspace/exported/custom-1.7b/speaker_encoder``).
        engine_mode: ``"trt"`` or ``"onnx"``.

    Returns:
        Path to the found artifact, or ``None``.
    """
    name = base.name
    if engine_mode == "trt":
        candidates = [
            base.parent / f"{name}.engine",
            base / "model.plan",
            base / f"{name}.engine",
        ]
    else:
        candidates = [
            base.parent / f"{name}.onnx",
            base / "model.onnx",
            base / f"{name}.onnx",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _has_onnx_src(base: Path) -> bool:
    """Check whether any ONNX file exists for the given base path."""
    name = base.name
    return any(
        p.is_file()
        for p in (
            base.parent / f"{name}.onnx",
            base / "model.onnx",
            base / f"{name}.onnx",
        )
    )


def _place_model(
    repo_dir: Path,
    name: str,
    src: Path,
    model_version: int,
    engine_mode: str,
) -> None:
    """Copy a model artifact into the repository with the right filename.

    TRT mode copies to ``<name>/<version>/model.plan``.
    ONNX mode copies to ``<name>/<version>/model.onnx`` and also
    copies ``.onnx.data`` external data if present.
    """
    model_dir = repo_dir / name / str(model_version)
    model_dir.mkdir(parents=True, exist_ok=True)

    if engine_mode == "trt":
        _copy(src, model_dir / "model.plan")
    else:
        _copy(src, model_dir / "model.onnx")
        # Copy ONNX external data if present
        data_src = Path(f"{src}.data")
        if data_src.is_file():
            _copy(data_src, model_dir / data_src.name)
            logger.info("    + copied external data: %s", data_src.name)


def _write_orchestrator_stub(orch_dir: Path) -> None:
    """Write a minimal ``model.py`` stub when the source tree is unavailable.

    The stub returns an error response for every request, making it obvious
    that the real orchestrator code was not copied.
    """
    orch_dir.mkdir(parents=True, exist_ok=True)
    stub_path = orch_dir / "model.py"
    if stub_path.exists():
        return
    stub_path.write_text(
        'import triton_python_backend_utils as pb_utils\n'
        'import numpy as np\n'
        'import json\n'
        '\n'
        'class TritonPythonModel:\n'
        '    def initialize(self, args):\n'
        '        self.model_config = json.loads(args["model_config"])\n'
        '        print("[TTS Orchestrator] Stub initialized")\n'
        '\n'
        '    def execute(self, requests):\n'
        '        responses = []\n'
        '        for request in requests:\n'
        '            audio = np.array([b""], dtype=object)\n'
        '            is_final = np.array([True], dtype=bool)\n'
        '            response = pb_utils.InferenceResponse(\n'
        '                output_tensors=[pb_utils.Tensor("audio_chunk", audio),\n'
        '                                pb_utils.Tensor("event_type", np.array(["error"], dtype=object)),\n'
        '                                pb_utils.Tensor("event_json", np.array([json.dumps({"type":"error","message":"stub"})], dtype=object)),\n'
        '                                pb_utils.Tensor("is_final", is_final)])\n'
        '            responses.append(response)\n'
        '        return responses\n'
        '\n'
        '    def finalize(self):\n'
        '        print("[TTS Orchestrator] Finalized")\n',
        encoding="utf-8",
    )
    logger.warning("  tts_orchestrator/model.py: stub created (no source tree)")


def _resolve_repo_manifest(repo_dir: Path, model_version: Optional[int] = None) -> Optional[Path]:
    """Find the authoritative triton_manifest.json in the assembled repo.

    Search order:
    1. ``tts_orchestrator/<version>/runtime/triton_manifest.json``
    2. ``tts_orchestrator/<version>/triton_manifest.json``
    3. ``repo_root/triton_manifest.json`` (fallback)

    Returns:
        Path to the manifest, or ``None``.
    """
    if model_version is not None:
        candidate = repo_dir / "tts_orchestrator" / str(model_version) / "runtime" / "triton_manifest.json"
        if candidate.is_file():
            return candidate
        candidate = repo_dir / "tts_orchestrator" / str(model_version) / "triton_manifest.json"
        if candidate.is_file():
            return candidate

    # Fallback: find any manifest under tts_orchestrator
    orch_dir = repo_dir / "tts_orchestrator"
    if orch_dir.is_dir():
        for candidate in sorted(orch_dir.rglob("runtime/triton_manifest.json")):
            return candidate
        for candidate in sorted(orch_dir.glob("*/triton_manifest.json")):
            return candidate

    candidate = repo_dir / "triton_manifest.json"
    if candidate.is_file():
        return candidate

    return None


def _infer_model_version_from_repo(repo_dir: Path) -> Optional[int]:
    """Infer the model version from the assembled repository manifest."""
    manifest_path = _resolve_repo_manifest(repo_dir)
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # Try package.model_package_dir first
    package = manifest.get("package") or {}
    if isinstance(package, dict):
        pkg_dir = package.get("model_package_dir", "")
        if pkg_dir:
            return int(Path(pkg_dir).name)

    # Try orchestrator.model_package_dir
    orch = manifest.get("orchestrator") or {}
    if isinstance(orch, dict):
        pkg_dir = orch.get("model_package_dir", "")
        if pkg_dir:
            return int(Path(pkg_dir).name)

    # Fall back to the manifest's parent directory name
    if manifest_path.parent.name == "runtime":
        return int(manifest_path.parent.parent.name)
    return int(manifest_path.parent.name)


# ---------------------------------------------------------------------------
#  TritonManager
# ---------------------------------------------------------------------------

class TritonManager:
    """Manage Triton Inference Server model repository and deployment.

    Responsibilities:
    * Assemble exported artifacts (ONNX / TensorRT) into a Triton
      ``model_repository`` directory layout.
    * Validate the assembled repository for completeness.
    * Resolve the correct NGC Triton image for the current GPU driver.
    * Perform HTTP health checks against a running Triton server.
    * Stop running Triton containers.

    Example::

        mgr = TritonManager()
        mgr.assemble_model_repo(
            exported_dir=Path("workspace/exported"),
            variant="custom-1.7b",
            model_repo_dir=Path("workspace/model_repository"),
            engine_mode="trt",
            model_version=1,
        )
        result = mgr.validate_model_repo(Path("workspace/model_repository"), 1)
        print(f"Valid: {result.valid}")
    """

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self._repo_root = repo_root or REPO_ROOT

    # -- Model repository assembly ------------------------------------------

    def assemble_model_repo(
        self,
        exported_dir: Path,
        variant: str,
        model_repo_dir: Path,
        engine_mode: str = "trt",
        model_version: Optional[int] = None,
    ) -> Path:
        """Assemble the Triton model repository from exported artifacts.

        Copies ONNX or TensorRT engine files from the export directory into
        the model_repository structure that Triton expects.  Also copies
        the Python BLS orchestrator code, tokenizer files, weights, and
        resources.

        Args:
            exported_dir: Path to ``workspace/exported/``.
            variant: Model variant identifier (e.g. ``"custom-1.7b"``).
            model_repo_dir: Target path for the assembled model repository.
            engine_mode: ``"trt"`` (default) or ``"onnx"``.
            model_version: Triton model version number.  Auto-resolved if
                           ``None``.

        Returns:
            Path to the assembled model repository.

        Raises:
            FileNotFoundError: If the variant directory or required manifest
                               is missing.
            RuntimeError: If a required model artifact cannot be found.
        """
        if model_version is None:
            model_version = resolve_model_version()

        variant_dir = exported_dir / variant
        tokenizer_dir = exported_dir / "tokenizer"
        orch_model_dir = model_repo_dir / "tts_orchestrator" / str(model_version)
        orch_http_model_dir = model_repo_dir / "tts_orchestrator_http" / str(model_version)
        runtime_dir = orch_model_dir / "runtime"

        # Read engine dtype from build output; default bf16
        engine_dtype = "bf16"
        dtype_file = exported_dir / ".engine_dtype"
        if dtype_file.is_file():
            try:
                engine_dtype = dtype_file.read_text(encoding="utf-8").strip() or "bf16"
            except OSError:
                engine_dtype = "bf16"

        logger.info(
            "Assembling Triton model repository "
            "(engine_mode=%s, dtype=%s, version=%d)",
            engine_mode, engine_dtype, model_version,
        )
        logger.info("  Variant:    %s", variant)
        logger.info("  Source:     %s", variant_dir)
        logger.info("  Repository: %s", model_repo_dir)

        if not variant_dir.is_dir():
            raise FileNotFoundError(
                f"Variant directory not found: {variant_dir}. "
                "Run Phase A first: autorun.sh setup or export_all.py"
            )

        manifest_src = variant_dir / "triton_manifest.json"
        if not manifest_src.is_file():
            raise FileNotFoundError(
                f"Missing {manifest_src} — run export_09 (writes manifest + fused ONNX)"
            )

        # Prepare output directory
        model_repo_dir.mkdir(parents=True, exist_ok=True)

        # Clean legacy top-level model directories
        for legacy in (
            "talker_code2wav_fused",
            "speaker_encoder",
            "speech_tokenizer_codec_fused",
            "speech_tokenizer_encoder",
            "talker_unified",
            "code2wav",
            "tts_orchestrator",
            "tts_orchestrator_http",
        ):
            legacy_path = model_repo_dir / legacy
            if legacy_path.exists():
                shutil.rmtree(legacy_path, ignore_errors=True)

        # Clean legacy files at repo root
        for legacy_file in ("triton_manifest.json", "artifact_manifest.json"):
            (model_repo_dir / legacy_file).unlink(missing_ok=True)

        runtime_dir.mkdir(parents=True, exist_ok=True)

        # -- 1. Optional voice-clone runtime assets --
        self._copy_runtime_asset(
            variant_dir / "speaker_encoder",
            runtime_dir,
            "speaker_encoder",
            engine_mode,
            required_for_trt=True,
            description="voice-clone preprocessing",
        )

        self._copy_runtime_asset(
            variant_dir / "speech_tokenizer_codec_fused",
            runtime_dir,
            "speech_tokenizer_codec_fused",
            engine_mode,
            required_for_trt=True,
            description="ICL preprocessing",
        )

        # -- 2. Talker + Code2Wav fused (required) --
        fused_src = _resolve_model_src(variant_dir / "talker_code2wav_fused", engine_mode)
        if fused_src is not None:
            if engine_mode == "trt":
                _copy(fused_src, runtime_dir / "model.plan")
                logger.info("  runtime/model.plan: OK")
            else:
                _copy(fused_src, runtime_dir / "model.onnx")
                data_src = Path(f"{fused_src}.data")
                if data_src.is_file():
                    _copy(data_src, runtime_dir / data_src.name)
                    logger.info("    + copied runtime external data: %s", data_src.name)
                logger.info("  runtime/model.onnx: OK")
        else:
            raise RuntimeError(
                f"talker_code2wav_fused: MISSING {engine_mode} file (required). "
                "Run export_09 + Phase B."
            )

        # -- 3. Verification-only models (optional, env-gated) --
        if os.environ.get("ASSEMBLE_VERIFICATION_MODELS", "0") == "1":
            self._place_verification_model(
                tokenizer_dir / "speech_tokenizer_encoder",
                model_repo_dir, "speech_tokenizer_encoder",
                model_version, engine_mode,
            )
            self._place_verification_model(
                variant_dir / "talker_unified",
                model_repo_dir, "talker_unified",
                model_version, engine_mode,
            )
            self._place_verification_model(
                tokenizer_dir / "code2wav_decoder",
                model_repo_dir, "code2wav",
                model_version, engine_mode,
            )

        # -- 5. TTS Orchestrator (Python BLS backend) --
        # Weights
        weights_src = variant_dir / "weights"
        weights_dst = orch_model_dir / "weights"
        if weights_src.is_dir():
            _copy_tree(weights_src, weights_dst)
            logger.info("  tts_orchestrator/weights: OK")
        else:
            logger.warning("  tts_orchestrator/weights: MISSING (no embedding weights found)")

        # Orchestrator Python code + engine package from source tree
        orch_py_src = self._repo_root / "model_repository" / "tts_orchestrator" / "1"
        if orch_py_src.is_dir():
            model_py_src = orch_py_src / "model.py"
            if model_py_src.is_file():
                _copy(model_py_src, orch_model_dir / "model.py")

            engine_dst = orch_model_dir / "engine"
            if engine_dst.exists():
                shutil.rmtree(engine_dst)
            engine_src = self._repo_root / "engine"
            if engine_src.is_dir():
                _copy_tree(engine_src, engine_dst)

            logger.info("  tts_orchestrator/python: OK (copied model.py + engine/ package)")
        else:
            logger.warning("  tts_orchestrator/python: source dir not found, using stub")
            _write_orchestrator_stub(orch_model_dir)

        # HTTP orchestrator
        orch_http_py_src = self._repo_root / "model_repository" / "tts_orchestrator_http" / "1"
        orch_http_model_dir.mkdir(parents=True, exist_ok=True)
        if orch_http_py_src.is_dir() and (orch_http_py_src / "model.py").is_file():
            _copy(orch_http_py_src / "model.py", orch_http_model_dir / "model.py")
            logger.info("  tts_orchestrator_http/python: OK (copied offline HTTP aggregator)")
        else:
            raise RuntimeError("tts_orchestrator_http/model.py missing in source tree")

        # Text tokenizer files for orchestrator
        model_base_dir = exported_dir.parent / "models"
        tok_dir_name = _VARIANT_MODEL_DIRS.get(variant)
        tok_dir = model_base_dir / tok_dir_name if tok_dir_name else None
        if tok_dir and tok_dir.is_dir():
            tok_dst = orch_model_dir / "tokenizer"
            tok_dst.mkdir(parents=True, exist_ok=True)
            for tf in _TOKENIZER_FILES:
                src_file = tok_dir / tf
                if src_file.is_file():
                    _copy(src_file, tok_dst / tf)

            # ICL warm state: copy code2wav_decoder.engine for base/icl variants
            if variant.startswith(("base-", "icl-")):
                if engine_mode == "trt":
                    c2w_engine = tokenizer_dir / "code2wav_decoder.engine"
                    if c2w_engine.is_file():
                        _copy(c2w_engine, tok_dst / "code2wav_decoder.engine")
                        logger.info("  tts_orchestrator/tokenizer/code2wav_decoder.engine: OK (ICL warm state)")
                elif (tokenizer_dir / "code2wav_decoder.onnx").is_file():
                    logger.warning(
                        "  tts_orchestrator/tokenizer/code2wav_decoder.engine: "
                        "SKIPPED (run Phase B to enable ICL warm state)"
                    )
            logger.info("  tts_orchestrator/tokenizer: OK")
        else:
            logger.warning("  tts_orchestrator/tokenizer: SKIPPED (model dir not found)")

        # Resources
        resources_src = self._repo_root / "resources"
        if resources_src.is_dir():
            _copy_tree(resources_src, orch_model_dir / "resources")
            logger.info("  tts_orchestrator/resources: OK")
        else:
            logger.warning("  tts_orchestrator/resources: SKIPPED (resources/ not found)")

        # Write stub if model.py is still missing
        _write_orchestrator_stub(orch_model_dir)

        # Copy triton_manifest.json into the package
        self._write_package_manifest(
            manifest_src, engine_mode, model_version,
            model_repo_dir, orch_model_dir, runtime_dir, variant,
        )

        # Copy artifact_manifest.json if present
        artifact_manifest = exported_dir / "artifact_manifest.json"
        if artifact_manifest.is_file():
            _copy(artifact_manifest, orch_model_dir / "artifact_manifest.json")
            _copy(artifact_manifest, runtime_dir / "artifact_manifest.json")
            logger.info("  artifact_manifest.json: copied (tts_orchestrator/%d + runtime)", model_version)
        else:
            logger.warning("  artifact_manifest.json: NOT FOUND in %s", exported_dir)
            logger.warning(
                "    Runtime fingerprint guard will fail-stop "
                "unless QWEN3_ALLOW_FINGERPRINT_MISMATCH=1"
            )
            logger.warning("    Run: qwen3tts build")

        # Prune unexpected files from the orchestrator directory
        _prune_orchestrator_dir(orch_model_dir)

        # Generate Triton config.pbtxt files
        self._generate_triton_configs(model_repo_dir, runtime_dir, engine_mode, engine_dtype)

        logger.info("Model repository assembled: %s", model_repo_dir)
        return model_repo_dir

    # -- Model repository validation ----------------------------------------

    def validate_model_repo(
        self,
        model_repo_dir: Path,
        model_version: Optional[int] = None,
    ) -> ModelRepoValidation:
        """Validate the assembled Triton model repository.

        Checks for required models, their config files, version directories,
        and critical orchestrator assets (``model.py``, ``engine/``,
        manifest, runtime artifact).

        Args:
            model_repo_dir: Path to the assembled model repository.
            model_version: Model version to validate.  Auto-inferred if
                           ``None``.

        Returns:
            :class:`ModelRepoValidation` with errors and warnings.
        """
        result = ModelRepoValidation()

        if model_version is None:
            model_version = _infer_model_version_from_repo(model_repo_dir)
        if model_version is None:
            model_version = resolve_model_version()

        logger.info("Validating model repository: %s (version=%d)", model_repo_dir, model_version)

        # Check each required model
        for model_name in _REQUIRED_MODELS:
            model_dir = model_repo_dir / model_name
            result.checked_models.append(model_name)

            if not (model_dir / "config.pbtxt").is_file():
                result.errors.append(f"{model_name}: no config.pbtxt (required)")
                continue

            version_dir = model_dir / str(model_version)
            if not version_dir.is_dir():
                result.errors.append(f"{model_name}: no version directory ({model_version}/)")
                continue

            logger.info("  %s: OK", model_name)

        # Deep checks for tts_orchestrator
        orch_dir = model_repo_dir / "tts_orchestrator" / str(model_version)

        # model.py
        if not (orch_dir / "model.py").is_file():
            result.errors.append(f"tts_orchestrator/{model_version}/model.py: missing")
        else:
            logger.info("  tts_orchestrator/%d/model.py: OK", model_version)

        # engine/ package
        if not (orch_dir / "engine").is_dir():
            result.errors.append(f"tts_orchestrator/{model_version}/engine/: missing")
        else:
            logger.info("  tts_orchestrator/%d/engine/: OK", model_version)

        # triton_manifest.json in package
        if not (orch_dir / "triton_manifest.json").is_file():
            result.errors.append(f"tts_orchestrator/{model_version}/triton_manifest.json: missing")
        else:
            logger.info("  tts_orchestrator/%d/triton_manifest.json: OK", model_version)

        # artifact_manifest.json (warning only)
        if not (orch_dir / "artifact_manifest.json").is_file():
            result.warnings.append(
                f"tts_orchestrator/{model_version}/artifact_manifest.json: missing. "
                "Runtime fingerprint guard will fail-stop unless "
                "QWEN3_ALLOW_FINGERPRINT_MISMATCH=1"
            )
        else:
            logger.info("  tts_orchestrator/%d/artifact_manifest.json: OK", model_version)

        # resources
        if (orch_dir / "resources").is_dir():
            logger.info("  tts_orchestrator/%d/resources/: OK", model_version)
        elif (self._repo_root / "resources").is_dir():
            result.warnings.append(
                f"tts_orchestrator/{model_version}/resources/: missing "
                "(references using resources/... may fail)"
            )

        # Runtime engine artifact
        runtime_dir = orch_dir / "runtime"
        runtime_engine = runtime_dir / "model.plan"
        if not runtime_engine.is_file():
            runtime_engine = runtime_dir / "model.onnx"
        if not runtime_engine.is_file():
            result.errors.append(f"runtime artifact: missing ({runtime_dir}/model.plan or model.onnx)")
        else:
            logger.info("  runtime engine: OK (%s)", runtime_engine.name)

        # Check for legacy top-level model (should not exist)
        if (model_repo_dir / "talker_code2wav_fused").is_dir():
            result.errors.append(
                "legacy top-level model detected: talker_code2wav_fused. "
                f"Re-run assemble so the fused engine lives under "
                f"tts_orchestrator/{model_version}/runtime/"
            )

        # HTTP orchestrator model.py
        orch_http_dir = model_repo_dir / "tts_orchestrator_http" / str(model_version)
        if not (orch_http_dir / "model.py").is_file():
            result.errors.append(f"tts_orchestrator_http/{model_version}/model.py: missing")
        else:
            logger.info("  tts_orchestrator_http/%d/model.py: OK", model_version)

        # Finalize
        if result.errors:
            logger.error("Validation failed: %d required model(s) missing", len(result.errors))
            result.valid = False
        else:
            result.valid = True
            if result.warnings:
                logger.warning(
                    "Validation passed with %d optional asset(s) missing",
                    len(result.warnings),
                )
            else:
                logger.info("Validation passed: all models present")

        return result

    # -- NGC image resolution -----------------------------------------------

    def resolve_deploy_image(self) -> Optional[str]:
        """Find the right NGC Triton image for the current GPU driver.

        Uses :mod:`qwen3tts_tools.docker` for NGC matrix lookups.  Prefers
        the custom deploy image (built with ``build-image``) and falls back
        to the raw ``-py3`` NGC base image.

        Returns:
            Docker image URI (e.g. ``"nvcr.io/nvidia/tritonserver:25.03-py3"``),
            or ``None`` if no compatible image can be resolved.
        """
        from qwen3tts_tools.docker import NgcMatrix, detect_driver_version
        from qwen3tts_tools.ngc_matrix import NGC_PY3_SUFFIX, NGC_TRITON_BASE

        driver = detect_driver_version()
        if driver is None:
            logger.error("Cannot detect NVIDIA driver version")
            return None

        matrix = NgcMatrix()
        ngc_tag = matrix.resolve_tag(driver)
        if ngc_tag is None:
            logger.error("Cannot determine NGC tag for driver %s", driver)
            return None

        # Check for a custom deploy image first
        deploy_tag = f"qwen3-tts-triton-deploy:{ngc_tag}"
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", deploy_tag],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                logger.info("Using Triton deploy image: %s", deploy_tag)
                return deploy_tag
        except (subprocess.TimeoutExpired, OSError):
            pass

        # Fall back to raw NGC base image
        base_image = f"{NGC_TRITON_BASE}:{ngc_tag}{NGC_PY3_SUFFIX}"
        logger.warning(
            "Deploy image not found (%s), falling back to: %s",
            deploy_tag, base_image,
        )
        logger.warning("Build it first: qwen3tts build")
        return base_image

    # -- Health check -------------------------------------------------------

    @staticmethod
    def health_check(port: int = 8000, timeout: float = 60.0) -> bool:
        """Check if a Triton server is healthy.

        Polls the Triton health endpoint ``/v2/health/ready`` with retries
        until the server responds or the timeout is exceeded.

        Args:
            port: HTTP port the Triton server is listening on.
            timeout: Maximum seconds to wait for the server to become ready.

        Returns:
            ``True`` if the server is healthy within the timeout.
        """
        import urllib.request
        import urllib.error

        url = f"http://localhost:{port}/v2/health/ready"
        interval = 3.0
        elapsed = 0.0

        while elapsed < timeout:
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if 200 <= resp.status < 300:
                        logger.info("Triton server ready at localhost:%d", port)
                        return True
            except urllib.error.URLError:
                pass
            except Exception:  # noqa: BLE001
                pass

            time.sleep(interval)
            elapsed += interval

        logger.error("Triton health check timed out after %.0fs", timeout)
        logger.error("Check logs: docker logs <container_name>")
        return False

    # -- Container management -----------------------------------------------

    @staticmethod
    def stop(container_name: str = "qwen3-tts-triton") -> bool:
        """Stop and remove a running Triton container.

        Args:
            container_name: Name of the Docker container to stop.

        Returns:
            ``True`` if the container was stopped or was not running.
        """
        # Check if the container exists
        try:
            proc = subprocess.run(
                ["docker", "container", "inspect", container_name],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            logger.error("Failed to inspect container: %s", container_name)
            return False

        if proc.returncode != 0:
            logger.info("Container not found: %s", container_name)
            return True

        # Remove the container (stops it if running)
        logger.info("Stopping and removing Triton container: %s", container_name)
        try:
            proc = subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                logger.info("Container removed: %s", container_name)
                return True
        except (subprocess.TimeoutExpired, OSError):
            pass

        logger.error("Failed to remove container: %s", container_name)
        return False

    # -- Internal helpers ---------------------------------------------------

    def _copy_runtime_asset(
        self,
        base_path: Path,
        runtime_dir: Path,
        asset_name: str,
        engine_mode: str,
        *,
        required_for_trt: bool = False,
        description: str = "",
    ) -> None:
        """Copy an optional runtime asset (speaker_encoder, codec, etc.)."""
        src = _resolve_model_src(base_path, engine_mode)
        if src is not None:
            if engine_mode == "trt":
                _copy(src, runtime_dir / f"{asset_name}.engine")
                logger.info("  runtime/%s.engine: OK", asset_name)
            else:
                _copy(src, runtime_dir / f"{asset_name}.onnx")
                data_src = Path(f"{src}.data")
                if data_src.is_file():
                    _copy(data_src, runtime_dir / data_src.name)
                logger.info("  runtime/%s.onnx: OK", asset_name)
        elif engine_mode == "trt" and _has_onnx_src(base_path):
            msg = f"  runtime/{asset_name}.engine: MISSING for TRT {description}. Run Phase B."
            if required_for_trt:
                raise RuntimeError(msg)
            logger.warning(msg)
        else:
            logger.warning("  runtime/%s: SKIPPED (only needed for %s)", asset_name, description)

    def _place_verification_model(
        self,
        base_path: Path,
        repo_dir: Path,
        name: str,
        model_version: int,
        engine_mode: str,
    ) -> None:
        """Copy an optional verification-only model into the repo."""
        src = _resolve_model_src(base_path, engine_mode)
        if src is not None:
            _place_model(repo_dir, name, src, model_version, engine_mode)
            logger.info("  %s: OK (verification)", name)

    def _write_package_manifest(
        self,
        manifest_src: Path,
        engine_mode: str,
        model_version: int,
        repo_dir: Path,
        orch_model_dir: Path,
        runtime_dir: Path,
        variant: str,
    ) -> None:
        """Read the source manifest and write copies into the package."""
        try:
            manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("  failed to read triton_manifest.json: %s", exc)
            return

        # Add engine_mode and package info
        manifest["engine_mode"] = engine_mode
        model_package_dir = f"/models/tts_orchestrator/{model_version}"
        manifest.setdefault("package", {})["model_package_dir"] = model_package_dir
        manifest.setdefault("orchestrator", {})["model_package_dir"] = model_package_dir

        # Adjust optional assets based on variant and engine mode
        optional_assets = manifest.setdefault("package", {}).setdefault("optional_assets", {})
        if isinstance(optional_assets, dict):
            # speech_tokenizer_encoder is verification-only
            optional_assets.pop("speech_tokenizer_encoder", None)

            if variant.startswith(("base-", "icl-")):
                if engine_mode == "trt":
                    optional_assets["speaker_encoder"] = "runtime/speaker_encoder.engine"
                    optional_assets["speech_tokenizer_codec_fused"] = "runtime/speech_tokenizer_codec_fused.engine"
                else:
                    optional_assets["speaker_encoder"] = "runtime/speaker_encoder.onnx"
                    optional_assets["speech_tokenizer_codec_fused"] = "runtime/speech_tokenizer_codec_fused.onnx"
            else:
                optional_assets.pop("speaker_encoder", None)
                optional_assets.pop("speech_tokenizer_codec_fused", None)

        # Write to both orchestrator and runtime directories
        for target in (
            orch_model_dir / "triton_manifest.json",
            runtime_dir / "triton_manifest.json",
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        logger.info("  triton_manifest.json: copied (tts_orchestrator/%d + runtime)", model_version)

    def _generate_triton_configs(
        self,
        repo_dir: Path,
        runtime_dir: Path,
        engine_mode: str,
        engine_dtype: str,
    ) -> None:
        """Run generate_triton_configs.py to create config.pbtxt files."""
        script = self._repo_root / "scripts" / "python" / "generate_triton_configs.py"
        manifest_path = runtime_dir / "triton_manifest.json"

        if not script.is_file():
            logger.warning("  generate_triton_configs.py not found — skipping config generation")
            return

        env = os.environ.copy()
        # Ensure scripts/python is on PYTHONPATH for triton_manifest_io
        scripts_python = str(self._repo_root / "scripts" / "python")
        env["PYTHONPATH"] = scripts_python + os.pathsep + env.get("PYTHONPATH", "")

        try:
            proc = subprocess.run(
                [
                    sys.executable, str(script),
                    "--manifest", str(manifest_path),
                    "--output-repo", str(repo_dir),
                    "--engine-mode", engine_mode,
                    "--engine-dtype", engine_dtype,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            if proc.returncode == 0:
                logger.info("  Triton config.pbtxt: generated from triton_manifest.json")
            else:
                logger.error("  generate_triton_configs.py failed:\n%s", proc.stderr)
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.error("  generate_triton_configs.py failed: %s", exc)


# ---------------------------------------------------------------------------
#  Internal helpers (module-level)
# ---------------------------------------------------------------------------

def _prune_orchestrator_dir(orch_dir: Path) -> None:
    """Remove unexpected files from the orchestrator directory.

    Keeps only the expected payloads: model.py, engine/, tokenizer/,
    weights/, runtime/, resources/, triton_manifest.json,
    artifact_manifest.json.
    """
    expected = {
        "model.py",
        "engine",
        "tokenizer",
        "weights",
        "runtime",
        "resources",
        "triton_manifest.json",
        "artifact_manifest.json",
    }
    for child in list(orch_dir.iterdir()):
        if child.name not in expected:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    # Remove __pycache__ directories
    for pycache in orch_dir.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)

    logger.info("  tts_orchestrator/python: pruned legacy payload")


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Minimal CLI for ad-hoc usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Triton Inference Server deployment manager for Qwen3-TTS",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # assemble
    p_assemble = sub.add_parser("assemble", help="Assemble Triton model repository")
    p_assemble.add_argument("--exported-dir", type=Path,
                            default=WORKSPACE_DIR / "exported",
                            help="Path to workspace/exported/")
    p_assemble.add_argument("--variant", default="custom-1.7b",
                            help="Model variant (default: custom-1.7b)")
    p_assemble.add_argument("--model-repo-dir", type=Path,
                            default=WORKSPACE_DIR / "model_repository",
                            help="Target model repository directory")
    p_assemble.add_argument("--engine-mode", choices=["trt", "onnx"], default="trt",
                            help="Engine mode (default: trt)")
    p_assemble.add_argument("--model-version", type=int, default=None,
                            help="Model version number (default: auto-resolve)")

    # validate
    p_validate = sub.add_parser("validate", help="Validate assembled model repository")
    p_validate.add_argument("--model-repo-dir", type=Path,
                            default=WORKSPACE_DIR / "model_repository",
                            help="Path to model repository")
    p_validate.add_argument("--model-version", type=int, default=None,
                            help="Model version number (default: auto-resolve)")

    # resolve-image
    sub.add_parser("resolve-image", help="Resolve NGC Triton image for current driver")

    # health
    p_health = sub.add_parser("health", help="Check Triton server health")
    p_health.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    p_health.add_argument("--timeout", type=float, default=60.0, help="Timeout in seconds")

    # stop
    p_stop = sub.add_parser("stop", help="Stop a Triton container")
    p_stop.add_argument("--container", default="qwen3-tts-triton",
                        help="Container name (default: qwen3-tts-triton)")

    args = parser.parse_args()
    mgr = TritonManager()

    if args.command == "assemble":
        try:
            result_dir = mgr.assemble_model_repo(
                exported_dir=args.exported_dir,
                variant=args.variant,
                model_repo_dir=args.model_repo_dir,
                engine_mode=args.engine_mode,
                model_version=args.model_version,
            )
            print(f"Model repository assembled: {result_dir}")
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "validate":
        result = mgr.validate_model_repo(
            model_repo_dir=args.model_repo_dir,
            model_version=args.model_version,
        )
        for err in result.errors:
            print(f"ERROR: {err}")
        for warn in result.warnings:
            print(f"WARNING: {warn}")
        if result.valid:
            print("Validation passed")
        else:
            print("Validation failed", file=sys.stderr)
            sys.exit(1)

    elif args.command == "resolve-image":
        image = mgr.resolve_deploy_image()
        if image:
            print(image)
        else:
            print("Cannot resolve Triton image", file=sys.stderr)
            sys.exit(1)

    elif args.command == "health":
        ok = TritonManager.health_check(port=args.port, timeout=args.timeout)
        if ok:
            print("Triton server: healthy")
        else:
            print("Triton server: NOT healthy", file=sys.stderr)
            sys.exit(1)

    elif args.command == "stop":
        ok = TritonManager.stop(container_name=args.container)
        if ok:
            print(f"Container stopped: {args.container}")
        else:
            print(f"Failed to stop container: {args.container}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
