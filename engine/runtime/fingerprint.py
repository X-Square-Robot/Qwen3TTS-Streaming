"""Runtime engine fingerprint guard.

When a TensorRT engine is compiled on one host (e.g. 4090D packaging machine,
SM 8.9) and then mistakenly deployed onto another (e.g. A800 DSW container,
SM 8.0), the only thing that will tell you is an opaque ``deserialize_cuda_engine``
failure or — worse — a segfault.

This module gives the standalone and Triton entrypoints a single, strict
check that runs **before** any TRT plan is touched.  It cross-references the
``artifact_manifest.json`` produced by the unified build pipeline against the
actual GPU / driver / TensorRT version the Python process sees, and refuses
to start the service when they disagree.

Bypass switch: set ``QWEN3_ALLOW_FINGERPRINT_MISMATCH=1`` (mirrors the bash
side ``ALLOW_FINGERPRINT_MISMATCH``).  This only makes sense for explicit
debugging — never in production rollouts.

Layered defence (see plans/unified-engine-build-pipeline):

    Build  (bash) -> artifact_manifest.json with sha256 + SM + TRT
    Deploy (bash) -> autorun.sh / deploy.sh preflight
    Runtime (this module) -> live SM / TRT check at process startup
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Public env knobs
# ---------------------------------------------------------------------------

ENV_ALLOW_MISMATCH = "QWEN3_ALLOW_FINGERPRINT_MISMATCH"
ENV_VERIFY_SHA256 = "QWEN3_VERIFY_ENGINE_SHA256"

# Standard names the build pipeline emits.  Order matters: the per-package
# copy (assembled by scripts/bash/lib/triton.sh::assemble_model_repo) takes
# precedence over the workspace copy so a stale workspace cannot mask a
# freshly assembled model_repository.
ARTIFACT_MANIFEST_CANDIDATES: tuple[str, ...] = (
    "artifact_manifest.json",
    "../artifact_manifest.json",
    "../../artifact_manifest.json",
)


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactManifest:
    """In-memory representation of artifact_manifest.json."""

    path: Path
    ngc_tag: str
    ngc_image: str
    tensorrt_version: str
    cuda_version: str
    gpu_sm: str
    gpu_name: str
    driver_version: str
    engine_dtype: str
    engines: Mapping[str, Mapping[str, Any]]
    raw: Mapping[str, Any]

    @classmethod
    def from_path(cls, path: Path) -> "ArtifactManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            path=Path(path),
            ngc_tag=str(data.get("ngc_tag") or ""),
            ngc_image=str(data.get("ngc_image") or ""),
            tensorrt_version=str(data.get("tensorrt_version") or ""),
            cuda_version=str(data.get("cuda_version") or ""),
            gpu_sm=str(data.get("gpu_sm") or ""),
            gpu_name=str(data.get("gpu_name") or ""),
            driver_version=str(data.get("driver_version") or ""),
            engine_dtype=str(data.get("engine_dtype") or ""),
            engines=dict(data.get("engines") or {}),
            raw=data,
        )


@dataclass(frozen=True)
class RuntimeEnvironment:
    """What the Python process currently sees on this host."""

    gpu_sm: str
    gpu_name: str
    tensorrt_version: str
    driver_version: str
    device_index: int

    def summary(self) -> str:
        return (
            f"sm={self.gpu_sm or '?'} "
            f"trt={self.tensorrt_version or '?'} "
            f"driver={self.driver_version or '?'} "
            f"gpu='{self.gpu_name or '?'}' "
            f"device={self.device_index}"
        )


@dataclass(frozen=True)
class FingerprintMismatch:
    """One specific field that disagrees between manifest and runtime."""

    field: str
    expected: str
    actual: str
    severity: str = "error"  # "error" | "warning"

    def render(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: expected={self.expected} actual={self.actual}"


@dataclass
class FingerprintReport:
    """Aggregated outcome of validate_engine_fingerprint."""

    manifest: ArtifactManifest | None
    env: RuntimeEnvironment
    mismatches: list[FingerprintMismatch] = field(default_factory=list)

    @property
    def errors(self) -> list[FingerprintMismatch]:
        return [m for m in self.mismatches if m.severity == "error"]

    @property
    def warnings(self) -> list[FingerprintMismatch]:
        return [m for m in self.mismatches if m.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


class FingerprintCheckError(RuntimeError):
    """Raised by enforce_engine_fingerprint() when validation fails strict mode."""


# ---------------------------------------------------------------------------
#  Manifest loading
# ---------------------------------------------------------------------------


def load_artifact_manifest(
    model_package_dir: str | os.PathLike[str],
) -> ArtifactManifest | None:
    """Locate and parse artifact_manifest.json near a model package.

    Search order (first hit wins):
        <package_dir>/artifact_manifest.json
        <package_dir>/../artifact_manifest.json
        <package_dir>/../../artifact_manifest.json

    Returns ``None`` if no candidate is found.  Callers (server.py /
    model.py) decide whether that is fatal based on the strict flag.
    """
    base = Path(model_package_dir).resolve()
    for rel in ARTIFACT_MANIFEST_CANDIDATES:
        candidate = (base / rel).resolve()
        if candidate.is_file():
            try:
                return ArtifactManifest.from_path(candidate)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to parse %s: %s", candidate, exc)
                continue
    return None


# ---------------------------------------------------------------------------
#  Runtime environment probing
# ---------------------------------------------------------------------------


def _probe_gpu_via_torch(device_index: int) -> tuple[str, str]:
    """Returns (sm_string, name) using torch.cuda.  Empty strings on failure."""
    try:
        import torch  # type: ignore
    except Exception as exc:
        logger.debug("torch unavailable for SM probing: %s", exc)
        return "", ""
    if not torch.cuda.is_available():  # type: ignore[attr-defined]
        return "", ""
    try:
        major, minor = torch.cuda.get_device_capability(device_index)  # type: ignore[attr-defined]
        name = torch.cuda.get_device_name(device_index)  # type: ignore[attr-defined]
        return f"sm_{major}{minor}", str(name or "")
    except Exception as exc:  # pragma: no cover - hardware-specific
        logger.debug("torch.cuda.get_device_capability failed: %s", exc)
        return "", ""


def _probe_gpu_via_nvidia_smi(device_index: int) -> tuple[str, str]:
    """Returns (sm_string, name) using nvidia-smi. Empty strings on failure."""
    if not shutil.which("nvidia-smi"):
        return "", ""
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={device_index}",
                    "--query-gpu=compute_cap,name",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            .strip()
            .splitlines()
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("nvidia-smi GPU probe failed: %s", exc)
        return "", ""
    if not out:
        return "", ""
    row = out[0].strip()
    if not row:
        return "", ""
    parts = [part.strip() for part in row.split(",", 1)]
    compute_cap = parts[0] if parts else ""
    name = parts[1] if len(parts) > 1 else ""
    match = re.search(r"(\d+)\s*\.\s*(\d+)", compute_cap)
    sm = f"sm_{match.group(1)}{match.group(2)}" if match else ""
    return sm, name


def _probe_tensorrt_version() -> str:
    """Returns the TensorRT version string, or empty string."""
    try:
        import tensorrt  # type: ignore
    except Exception as exc:
        logger.debug("tensorrt unavailable: %s", exc)
        return ""
    raw = getattr(tensorrt, "__version__", "") or ""
    # Normalize "10.5.0.18" → "10.5.0" (manifest uses 3-part form)
    parts = str(raw).split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return str(raw)


def _probe_driver_via_nvml() -> str:
    """Returns NVIDIA driver version via pynvml/nvidia-smi.  Empty on failure."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            raw = pynvml.nvmlSystemGetDriverVersion()
            return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception as exc:
        logger.debug("pynvml driver probe failed: %s", exc)
    # Subprocess fallback
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            .strip()
            .splitlines()
        )
        if out and re.match(r"^\d+(\.\d+){1,2}$", out[0].strip()):
            return out[0].strip()
    except Exception as exc:  # pragma: no cover
        logger.debug("nvidia-smi driver probe failed: %s", exc)
    return ""


def probe_current_environment(device_index: int = 0) -> RuntimeEnvironment:
    """Snapshot what this Python process sees on the GPU host."""
    sm, name = _probe_gpu_via_nvidia_smi(device_index)
    if not sm or not name:
        torch_sm, torch_name = _probe_gpu_via_torch(device_index)
        sm = sm or torch_sm
        name = name or torch_name
    return RuntimeEnvironment(
        gpu_sm=sm,
        gpu_name=name,
        tensorrt_version=_probe_tensorrt_version(),
        driver_version=_probe_driver_via_nvml(),
        device_index=device_index,
    )


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------


def _major_minor(s: str) -> str:
    parts = (s or "").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else s


def _normalize_gpu_name(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _driver_ge(installed: str, required: str) -> bool:
    """Return True if `installed` >= `required` on major.minor."""
    inst = re.findall(r"\d+", installed or "")
    req = re.findall(r"\d+", required or "")
    if not inst or not req:
        return False
    inst = [int(x) for x in inst]
    req = [int(x) for x in req]
    while len(inst) < 2:
        inst.append(0)
    while len(req) < 2:
        req.append(0)
    return (inst[0], inst[1]) >= (req[0], req[1])


def validate_engine_fingerprint(
    manifest: ArtifactManifest,
    env: RuntimeEnvironment,
) -> FingerprintReport:
    """Compare a manifest against the live runtime environment.

    The check is deliberately narrow: things that *will* break TRT or the
    serving stack go into ``errors``; informational drift goes into
    ``warnings``.  Schema:

      gpu_sm           hard mismatch  -> error  (engine will not load)
      gpu_name         hard mismatch  -> error  (same-SM device-model drift can deadlock)
      tensorrt_version major.minor    -> error  (plan format incompatible)
      driver_version   downgrade only -> warn   (newer driver = fine)
    """
    mismatches: list[FingerprintMismatch] = []

    if manifest.gpu_sm and env.gpu_sm and manifest.gpu_sm != env.gpu_sm:
        mismatches.append(
            FingerprintMismatch(
                field="gpu_sm",
                expected=manifest.gpu_sm,
                actual=env.gpu_sm,
                severity="error",
            )
        )

    if manifest.gpu_name and env.gpu_name:
        if _normalize_gpu_name(manifest.gpu_name) != _normalize_gpu_name(env.gpu_name):
            mismatches.append(
                FingerprintMismatch(
                    field="gpu_name",
                    expected=manifest.gpu_name,
                    actual=env.gpu_name,
                    severity="error",
                )
            )

    if manifest.tensorrt_version and env.tensorrt_version:
        if _major_minor(manifest.tensorrt_version) != _major_minor(
            env.tensorrt_version
        ):
            mismatches.append(
                FingerprintMismatch(
                    field="tensorrt_version",
                    expected=manifest.tensorrt_version,
                    actual=env.tensorrt_version,
                    severity="error",
                )
            )

    if manifest.driver_version and env.driver_version:
        # We don't fail on driver mismatch — newer driver is always OK,
        # older driver is a soft warning so ops can spot the drift.
        if not _driver_ge(env.driver_version, manifest.driver_version):
            mismatches.append(
                FingerprintMismatch(
                    field="driver_version",
                    expected=f">={manifest.driver_version}",
                    actual=env.driver_version,
                    severity="warning",
                )
            )

    return FingerprintReport(manifest=manifest, env=env, mismatches=mismatches)


def verify_engine_sha256(
    manifest: ArtifactManifest,
    runtime_dir: str | os.PathLike[str],
) -> list[FingerprintMismatch]:
    """Optional per-engine sha256 verification (slow; opt-in)."""
    runtime_root = Path(runtime_dir)
    mismatches: list[FingerprintMismatch] = []
    for rel, meta in manifest.engines.items():
        engine_path = runtime_root / rel
        if not engine_path.is_file():
            # Some engines are variant-specific; missing != failure here.
            continue
        expected = str((meta or {}).get("sha256") or "")
        if not expected:
            continue
        actual = _sha256_file(engine_path)
        if actual != expected:
            mismatches.append(
                FingerprintMismatch(
                    field=f"engines/{rel}/sha256",
                    expected=expected[:16] + "…",
                    actual=actual[:16] + "…",
                    severity="error",
                )
            )
    return mismatches


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def format_report(report: FingerprintReport, *, header: str | None = None) -> str:
    """Human-readable summary suitable for log lines and exception messages."""
    lines: list[str] = []
    if header:
        lines.append(header)
    if report.manifest is not None:
        m = report.manifest
        lines.append(
            f"manifest:  ngc={m.ngc_tag}  trt={m.tensorrt_version}  "
            f"sm={m.gpu_sm}  gpu='{m.gpu_name}'  "
            f"driver={m.driver_version}  dtype={m.engine_dtype}"
        )
        lines.append(f"           file={m.path}")
    else:
        lines.append("manifest:  <not loaded>")
    lines.append(f"runtime:   {report.env.summary()}")
    for m in report.mismatches:
        lines.append("  " + m.render())
    if not report.mismatches:
        lines.append("  no mismatches")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Public enforcement entrypoint
# ---------------------------------------------------------------------------


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def enforce_engine_fingerprint(
    model_package_dir: str | os.PathLike[str],
    *,
    device_index: int = 0,
    runtime_dir: str | os.PathLike[str] | None = None,
    strict: bool = True,
) -> FingerprintReport:
    """Run the full guard.  Raise FingerprintCheckError on strict failure.

    Designed to be called once at process startup, before any TRT plan is
    touched.  The return value is suitable for callers that want to log a
    success summary alongside their own startup banner.

    Args:
        model_package_dir: directory containing the deployed model package
                           (e.g. /models/tts_orchestrator/1).
        device_index:      which CUDA device the service will pin to.
        runtime_dir:       directory containing the per-variant .engine files
                           (for optional sha256 verification); defaults to
                           ``model_package_dir/runtime``.
        strict:            when True, missing manifest or any error-severity
                           mismatch raises FingerprintCheckError.  Override
                           with ``QWEN3_ALLOW_FINGERPRINT_MISMATCH=1``.

    Raises:
        FingerprintCheckError: strict mode + missing manifest or any error.
    """
    allow_mismatch = _truthy_env(ENV_ALLOW_MISMATCH)
    env = probe_current_environment(device_index)
    manifest = load_artifact_manifest(model_package_dir)

    if manifest is None:
        report = FingerprintReport(manifest=None, env=env)
        msg = (
            f"missing artifact_manifest.json under {model_package_dir} "
            "(searched ./, ../, ../../).  Engine origin cannot be verified — "
            "rebuild via 'autorun.sh build' or 'autorun.sh import-artifact'."
        )
        if allow_mismatch or not strict:
            logger.warning("Engine fingerprint check skipped: %s", msg)
            return report
        raise FingerprintCheckError(msg)

    report = validate_engine_fingerprint(manifest, env)

    if _truthy_env(ENV_VERIFY_SHA256):
        rt = Path(runtime_dir) if runtime_dir else Path(model_package_dir) / "runtime"
        sha_errors = verify_engine_sha256(manifest, rt)
        report.mismatches.extend(sha_errors)

    if report.warnings:
        for w in report.warnings:
            logger.warning("Engine fingerprint warning: %s", w.render())

    if not report.errors:
        logger.info(
            "Engine fingerprint OK (manifest=%s, %s)",
            manifest.path.name,
            env.summary(),
        )
        return report

    detail = format_report(report, header="Engine fingerprint mismatch:")
    if allow_mismatch or not strict:
        logger.warning(
            "%s\n(bypassed via %s=1 or strict=False)", detail, ENV_ALLOW_MISMATCH
        )
        return report

    raise FingerprintCheckError(
        f"{detail}\n\n"
        "Refusing to start.  Set "
        f"{ENV_ALLOW_MISMATCH}=1 to bypass (debugging only), or rebuild "
        "the engine package on the target hardware."
    )
