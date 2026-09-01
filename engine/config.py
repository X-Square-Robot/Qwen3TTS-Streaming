"""Engine configuration — split into engine params (YAML) and model params (manifest).

Loading priority for model architecture:
    package manifest (resolved runtime_dir) > engine.yaml model overrides > defaults

Loading priority for engine params:
    CLI args > ENGINE_* env vars > PORT aliases > engine.yaml > defaults

Usage:
    cfg = load_config("engine.yaml", cli_overrides={...})
    paths = resolve_model_package_paths(cfg.paths.model_package_dir)
    model_arch = load_model_manifest(paths.engine_dir, cfg)
    model_config = to_model_config(model_arch, cfg)
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Model architecture (from manifest, not engine.yaml)
# ---------------------------------------------------------------------------


@dataclass
class EngineProfileConfig:
    """TensorRT profile limits recorded in triton_manifest.json."""

    engine_mode: str = ""
    engine_dtype: str = ""
    triton_io_float_dtype: str = ""
    backbone_precision: str = ""
    cp_precision: str = ""
    code2wav_precision: str = ""
    max_batch_size: int = 0
    max_input_len: int = 0
    max_seq_len: int = 0
    builder: str = ""
    builder_image: str = ""
    target_driver: str = ""
    built_at_utc: str = ""


@dataclass
class ModelArchConfig:
    """Model architecture — loaded from model_manifest / triton_manifest.json.

    These values are determined at export time and travel with the model
    artifacts.  The engine reads them from the manifest file inside the
    resolved model package runtime directory instead of requiring manual
    architecture configuration.
    """

    variant: str = ""
    num_layers: int = 28
    hidden_size: int = 2048
    kv_heads: int = 8
    head_dim: int = 128
    codec_vocab_size: int = 3072
    logits_topk: int = 50
    cp_num_stages: int = 15
    n_c2w_layers: int = 8
    c2w_kv_heads: int = 16
    c2w_head_dim: int = 64
    c2w_sliding_window: int = 72
    n_c2w_conv_states: int = 17
    n_c2w_transconv_states: int = 4
    dtype: str = "bf16"
    tts_model_type: str = "unknown"
    supported_task_types: tuple[str, ...] = ()
    engine_profile: EngineProfileConfig = field(default_factory=EngineProfileConfig)


# ---------------------------------------------------------------------------
# Engine config dataclasses (engine.yaml)
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    """File paths — resolved relative to repo root or absolute."""

    model_package_dir: str = ""
    tokenizer_dir: str = ""
    weights_dir: str = ""
    engine_dir: str = ""


@dataclass(frozen=True)
class ModelPackagePaths:
    """Resolved paths inside a Triton-compatible model package."""

    package_dir: str
    engine_dir: str
    weights_dir: str
    tokenizer_dir: str
    manifest_path: str
    runtime_artifact_path: str
    engine_mode: str = "unknown"


@dataclass
class ServerConfig:
    """Internal gRPC and public HTTP/WebSocket server settings."""

    port: int = 50051
    websocket_port: int = 0
    websocket_path: str = "/v1/ws"
    tls_cert_file: str = ""
    tls_key_file: str = ""
    health_port: int = 8080
    # /health probe semantics. "ready": 503 until the engine is fully started
    # (model loaded + warmup + gateways bound), then 200 — correct when one
    # path serves unified liveness/readiness/startup probes. "alive": 200 as
    # soon as the port is up — escape hatch for platforms whose liveness
    # grace cannot be sized to cover the model load. /livez, /readyz and
    # /metrics keep fixed semantics regardless of this knob.
    health_probe_mode: str = "ready"
    max_sessions: int = 128
    request_timeout_sec: float = 120.0
    warmup_rounds: int = 3
    # Limit how far delivery may run ahead of the estimated playback head.
    # This does not delay the first chunk; RTF > 1 builds a retractable tail
    # only after synthesis gets ahead. Clients may explicitly request
    # ``output_policy.config.delivery=firehose`` for legacy pass-through.
    guarded_delivery_default: bool = True
    guarded_delivery_window_ms: int = 100
    # Speakers to prewarm the prefix KV cache for at startup: for each, run one
    # real (discarded) synthesis so the speaker/config prefix is cached and the
    # first real request hits it (~50ms cold TTFT -> ~20ms warm). Empty = off.
    prewarm_speakers: list[str] = field(default_factory=list)


@dataclass
class SchedulerConfig:
    """Decode batch scheduling parameters."""

    max_batch_size: int = 48
    max_seq_len: int = 512
    # MLFQ
    mlfq_q1_threshold: int = 50
    mlfq_q2_threshold: int = 200
    mlfq_aging_interval: int = 100
    mlfq_starvation_limit: int = 50
    # Eviction
    max_idle_sec: float = 10.0
    max_queue_size: int = 256
    # Session timeout
    session_timeout_sec: float = 300.0
    # Pad phase: minimum pad steps before silence detection activates
    min_pad_steps: int = 4
    # Pad-phase near-silence detection thresholds.
    pad_silence_peak_threshold: float = 5e-4
    pad_silence_mean_abs_threshold: float = 2e-4
    # Token loop guard, stage 1: report a suspect run when codebook-0 emits the
    # same token for this many consecutive decode steps. A suspect is telemetry
    # only; generation continues so short, naturally recovering runs are not
    # truncated (0 = disable suspect/recovery telemetry).
    token_loop_suspect_frames: int = 4
    # Token loop guard, stage 2: retry/abort only when the same codebook-0 run
    # persists to this confirmation threshold (0 = disable the whole guard).
    # The original 4-frame threshold was calibrated on one text/speaker and
    # produced a confirmed false abort on a different long-form request. True
    # loops in the original sweep persisted for 10-39 frames, hence 10 here.
    token_loop_abort_frames: int = 10
    # Require generation to have progressed this far relative to consumed text
    # before the confirmed run can retry/abort. This prevents ordinary early
    # pronunciation holds from becoming terminal verdicts. 0 disables the
    # progress gate (legacy behavior).
    token_loop_min_audio_text_ratio: float = 2.0
    # Unconditional safety cap for a persistent run while input/progress is not
    # ready (for example a stalled token stream). 0 disables this fallback.
    token_loop_emergency_abort_frames: int = 20
    # When the guard (or pad-silence abort) fires on a segment whose audio is
    # still fully buffered behind an earlier live segment, rerun it with a
    # reseeded sampling stream instead of aborting, up to this many attempts
    # (0 = always abort). Reruns of the playhead segment are never attempted:
    # its audio already streamed.
    token_loop_max_retries: int = 1
    # At max_seq_len, classify only clearly implausible audio:text ratios as
    # runaways. Genuine long-text overflow keeps its historical semantics.
    length_runaway_ratio: float = 10.0


@dataclass
class PrefixCacheConfig:
    """Cross-request prefix KV cache."""

    enabled: bool = True
    max_entries: int = 16
    max_prefix_len: int = 512


@dataclass
class ReferenceCacheConfig:
    """Cross-request reference-audio feature cache."""

    enabled: bool = True
    max_entries: int = 16


@dataclass
class SpliterConfig:
    """Text segmentation / Spliter parameters."""

    # Two independent estimators: duration EMA is learned from clean EOS and
    # drives progress attribution; safety ratio is the conservative lower
    # bound used to derive the hard text-token capacity.
    ema_ratio_initial: float = 4.5
    safety_ratio_initial: float = 5.5
    ema_alpha: float = 0.1
    # Backward-compatible name: now controls safety backoff after censored
    # overflow/failure feedback instead of contaminating the duration EMA.
    ema_overflow_alpha: float = 0.5
    ema_min_ratio: float = 2.0
    ema_max_ratio: float = 10.0
    ema_min_observation_tokens: int = 8
    safety_failure_multiplier: float = 1.25
    max_concurrent_segments: int = 2
    prefill_len: int = 12
    safety_margin: int = 8
    # Punct-tier split mins as fractions of dynamic cap (see spliter/driver.compute_thresholds)
    l1_split_cap_ratio: float = 0.70
    l2_split_cap_ratio: float = 0.80
    l3_split_cap_ratio: float = 0.90


@dataclass
class SamplingConfig:
    """Decode sampling parameters (defaults, overridable per-request)."""

    do_sample: bool = False
    temperature: float = 0.9
    repetition_penalty: float = 1.05
    top_k: int = 50
    random_seed: int = 0


@dataclass
class PrefillConfig:
    """Prefill / CustomVoice speaker defaults (see engine.yaml)."""

    default_speaker: str = "vivian"
    fallback_speaker: str = "vivian"


@dataclass
class ReferencesConfig:
    """Optional Base/ICL reference-audio library.

    ``entries`` is intentionally kept as a raw mapping so deployments can add
    reference ids without changing this dataclass.
    """

    default: str = ""
    entries: dict = field(default_factory=dict)


@dataclass
class ObservabilityConfig:
    """Tiered observability control (see docs/dev/design/observability_tiers.md).

    ``level`` is the global default tier: daily | debug | dump (L1/L2/L3).
    ``max_session_level`` caps how high a per-request override may escalate a
    single session (production防滥用; keep ``daily`` in prod).
    ``text_capture`` governs synthesized-text logging privacy.
    The ``dump_*`` fields feed the L3 dumper (equivalent to ENGINE_DUMP_*).
    """

    level: str = "daily"
    max_session_level: str = "daily"
    text_capture: str = "preview"  # disabled | preview | hashed | full
    text_preview_chars: int = 64
    health_interval_sec: float = 30.0  # engine.health periodic gauge cadence (0=off)
    dump_dir: str = ""
    dump_limit: int = 0
    dump_include_wav: bool = True
    dump_sessions: str = ""


@dataclass
class EngineConfig:
    """Top-level engine configuration (no model architecture)."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    prefix_cache: PrefixCacheConfig = field(default_factory=PrefixCacheConfig)
    reference_cache: ReferenceCacheConfig = field(default_factory=ReferenceCacheConfig)
    spliter: SpliterConfig = field(default_factory=SpliterConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    prefill: PrefillConfig = field(default_factory=PrefillConfig)
    references: ReferencesConfig = field(default_factory=ReferencesConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

_CONFIG_SECTION_NAMES = (
    "paths",
    "server",
    "scheduler",
    "prefix_cache",
    "reference_cache",
    "spliter",
    "sampling",
    "prefill",
    "references",
    "observability",
)


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _apply_env_overrides(
    raw: dict,
    environ: Mapping[str, str] | None = None,
) -> dict:
    """Apply platform port aliases and ENGINE_* environment overrides.

    E.g. ENGINE_SCHEDULER_MAX_BATCH_SIZE=32 → scheduler.max_batch_size = 32

    ``PORT`` and ``HEALTH_PORT`` are conventional container-platform aliases
    for the public WebSocket/HTTP port and the optional dedicated health port.
    The canonical ``ENGINE_SERVER_*`` variables retain higher precedence.
    """
    values = os.environ if environ is None else environ
    server = raw.setdefault("server", {})
    for alias, field_name, canonical in (
        ("PORT", "websocket_port", "ENGINE_SERVER_WEBSOCKET_PORT"),
        ("HEALTH_PORT", "health_port", "ENGINE_SERVER_HEALTH_PORT"),
    ):
        value = values.get(alias, "").strip()
        if value and canonical not in values:
            server[field_name] = _coerce_value(value)

    prefix = "ENGINE_"
    for key, val in values.items():
        if not key.startswith(prefix):
            continue
        body = key[len(prefix) :].lower()
        section = ""
        field_name = ""
        for candidate in sorted(_CONFIG_SECTION_NAMES, key=len, reverse=True):
            candidate_prefix = f"{candidate}_"
            if body.startswith(candidate_prefix):
                section = candidate
                field_name = body[len(candidate_prefix) :]
                break
        if not section:
            parts = body.split("_", 1)
            if len(parts) < 2:
                continue
            section, field_name = parts
        if not field_name:
            continue
        if section not in raw:
            raw[section] = {}
        raw[section][field_name] = _coerce_value(val)
    return raw


def _coerce_value(val: str):
    """Best-effort type coercion for env vars."""
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _dict_to_config(raw: dict) -> EngineConfig:
    """Build EngineConfig from a raw dict, ignoring unknown keys."""
    cfg = EngineConfig()
    for section_name, section_cls in [
        ("paths", PathsConfig),
        ("server", ServerConfig),
        ("scheduler", SchedulerConfig),
        ("prefix_cache", PrefixCacheConfig),
        ("reference_cache", ReferenceCacheConfig),
        ("spliter", SpliterConfig),
        ("sampling", SamplingConfig),
        ("prefill", PrefillConfig),
        ("references", ReferencesConfig),
        ("observability", ObservabilityConfig),
    ]:
        section_data = raw.get(section_name, {})
        if not isinstance(section_data, dict):
            continue
        section_obj = getattr(cfg, section_name)
        for k, v in section_data.items():
            if hasattr(section_obj, k):
                expected_type = type(getattr(section_obj, k))
                try:
                    if expected_type is bool and isinstance(v, str):
                        # bool("false") is True — parse string bools explicitly.
                        coerced = v.strip().lower() in ("1", "true", "yes", "on")
                    else:
                        coerced = expected_type(v)
                    setattr(section_obj, k, coerced)
                except (ValueError, TypeError):
                    logger.warning("Cannot set %s.%s = %r", section_name, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Public: load engine config
# ---------------------------------------------------------------------------

_AUTO_CONFIG_NAMES = ("engine.yaml", "engine.yml")


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
) -> EngineConfig:
    """Load engine config from YAML file + env vars + CLI overrides.

    Priority: CLI > env vars > YAML file > defaults.
    This only loads engine-level params.  Model architecture is loaded
    separately via ``load_model_manifest()``.

    When *config_path* is not given, auto-discovers engine.yaml / engine.yml
    in the current working directory.
    """
    raw: dict = {}

    if not config_path:
        for name in _AUTO_CONFIG_NAMES:
            candidate = Path(name)
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path:
        p = Path(config_path)
        if p.exists():
            if yaml is None:
                raise ImportError(
                    "PyYAML required for YAML config. Install: pip install pyyaml"
                )
            with open(p, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            logger.info("Loaded config from %s", p)
        else:
            logger.warning("Config file not found: %s, using defaults", p)

    _apply_env_overrides(raw)

    if cli_overrides:
        _deep_update(raw, cli_overrides)

    return _dict_to_config(raw)


def resolve_model_package_paths(
    model_package_dir: str = "",
    *,
    model_repository: str = "/models",
    model_name: str = "tts_orchestrator",
    model_version: str = "1",
) -> ModelPackagePaths:
    """Resolve the shared deployment package layout used by Triton and engine.

    The model package is the Triton model version directory:
    ``<model_repository>/<model_name>/<model_version>``.  Both runtimes load
    assets from the same subdirectories:
      - ``runtime``: TensorRT plan / runtime manifest / optional ONNX helpers
      - ``weights``: exported embedding and projection weights
      - ``tokenizer``: text tokenizer files
    """
    if model_package_dir:
        package = Path(model_package_dir)
    else:
        package = Path(model_repository) / model_name / str(model_version)

    runtime_dir = package / "runtime"
    weights_dir = package / "weights"
    tokenizer_dir = package / "tokenizer"
    manifest_path = package / "triton_manifest.json"
    if not manifest_path.is_file():
        manifest_path = runtime_dir / "triton_manifest.json"
    engine_mode = "unknown"
    runtime_artifact = runtime_dir / "model.plan"

    def _package_path(value: str, default: Path) -> Path:
        if not value:
            return default
        candidate = Path(value)
        return candidate if candidate.is_absolute() else package / candidate

    if manifest_path.is_file():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)
            engine_mode = str(data.get("engine_mode") or "unknown").lower()
            package_section = data.get("package", {})
            if isinstance(package_section, dict):
                runtime_dir = _package_path(
                    str(package_section.get("runtime_dir") or ""),
                    runtime_dir,
                )
                weights_dir = _package_path(
                    str(package_section.get("weights_dir") or ""),
                    weights_dir,
                )
                tokenizer_dir = _package_path(
                    str(package_section.get("tokenizer_dir") or ""),
                    tokenizer_dir,
                )
                artifacts = package_section.get("runtime_artifacts", {})
                if isinstance(artifacts, dict):
                    runtime_artifact = _package_path(
                        str(artifacts.get(engine_mode) or ""),
                        runtime_artifact,
                    )
                manifest_path = _package_path(
                    str(package_section.get("manifest") or ""),
                    manifest_path,
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read package manifest %s: %s", manifest_path, exc)

    if engine_mode == "onnx" and runtime_artifact == runtime_dir / "model.plan":
        runtime_artifact = runtime_dir / "model.onnx"
    elif not runtime_artifact.is_file() and (runtime_dir / "model.onnx").is_file():
        runtime_artifact = runtime_dir / "model.onnx"

    return ModelPackagePaths(
        package_dir=str(package),
        engine_dir=str(runtime_dir),
        weights_dir=str(weights_dir),
        tokenizer_dir=str(tokenizer_dir),
        manifest_path=str(manifest_path),
        runtime_artifact_path=str(runtime_artifact),
        engine_mode=engine_mode,
    )


def apply_model_package_paths(
    cfg: EngineConfig,
    paths: ModelPackagePaths,
) -> EngineConfig:
    """Apply resolved package paths to an EngineConfig in-place."""
    cfg.paths.model_package_dir = paths.package_dir
    cfg.paths.engine_dir = paths.engine_dir
    cfg.paths.weights_dir = paths.weights_dir
    cfg.paths.tokenizer_dir = paths.tokenizer_dir
    return cfg


# ---------------------------------------------------------------------------
# Public: load model manifest (architecture)
# ---------------------------------------------------------------------------

MANIFEST_FILENAMES = ("triton_manifest.json", "model_manifest.json")


def load_model_manifest(
    engine_dir: str,
    engine_config: Optional[EngineConfig] = None,
    tokenizer_dir: str = "",
) -> ModelArchConfig:
    """Load model architecture from the resolved runtime manifest directory.

    Search order for manifest:
      1. runtime_dir/triton_manifest.json
      2. package_dir/triton_manifest.json
      3. model_repository/triton_manifest.json

    If no manifest is found, falls back to auto-detection from
    tokenizer config.json, then to hardcoded defaults.

    Priority: manifest > engine.yaml "model" overrides > auto-detect > defaults.
    """
    arch = ModelArchConfig()
    manifest_data: dict = {}

    # --- Try loading manifest from engine_dir ---
    if engine_dir:
        engine_path = Path(engine_dir)
        candidate_dirs = [engine_path, engine_path.parent, engine_path.parent.parent]
        seen_dirs: set[Path] = set()
        for manifest_dir in candidate_dirs:
            if manifest_dir in seen_dirs:
                continue
            seen_dirs.add(manifest_dir)
            for fname in MANIFEST_FILENAMES:
                manifest_file = manifest_dir / fname
                if manifest_file.is_file():
                    try:
                        with open(manifest_file, encoding="utf-8") as f:
                            manifest_data = json.load(f)
                        logger.info("Loaded model manifest from %s", manifest_file)
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning("Failed to read %s: %s", manifest_file, e)
                    break
            if manifest_data:
                break

    # --- Extract architecture section ---
    arch_section = manifest_data.get("architecture", {})

    if not arch_section and manifest_data.get("talker"):
        talker = manifest_data["talker"]
        c2w = manifest_data.get("code2wav_fused", {})
        arch_section = {
            "num_layers": talker.get("num_layers"),
            "hidden_size": talker.get("hidden_size"),
            "kv_heads": talker.get("num_kv_heads"),
            "head_dim": talker.get("head_dim"),
            "codec_vocab_size": talker.get("vocab_size"),
            "n_c2w_layers": c2w.get("num_code2wav_hidden_layers"),
            "dtype": manifest_data.get("engine_dtype", "bf16"),
        }
        arch_section = {k: v for k, v in arch_section.items() if v is not None}
        logger.info("Built architecture from legacy manifest talker/code2wav sections")

    # --- Fallback: auto-detect from tokenizer config.json ---
    if not arch_section and tokenizer_dir:
        arch_section = _detect_from_tokenizer(tokenizer_dir)

    # --- Apply architecture to ModelArchConfig ---
    if manifest_data.get("variant"):
        arch.variant = manifest_data["variant"]
    orchestrator = manifest_data.get("orchestrator", {})
    if isinstance(orchestrator, dict):
        tts_model_type = orchestrator.get("tts_model_type")
        if tts_model_type:
            arch.tts_model_type = str(tts_model_type)
        supported = orchestrator.get("supported_task_types")
        if isinstance(supported, str) and supported.strip():
            arch.supported_task_types = tuple(
                s.strip() for s in supported.split(",") if s.strip()
            )
        elif isinstance(supported, list):
            arch.supported_task_types = tuple(
                str(s).strip() for s in supported if str(s).strip()
            )

    profile_section = manifest_data.get("engine_profile", {})
    if isinstance(profile_section, dict):
        for key in (
            "engine_mode",
            "engine_dtype",
            "triton_io_float_dtype",
            "backbone_precision",
            "cp_precision",
            "code2wav_precision",
            "builder",
            "builder_image",
            "target_driver",
            "built_at_utc",
        ):
            value = profile_section.get(key)
            if value is not None:
                setattr(arch.engine_profile, key, str(value))
        for key in ("max_batch_size", "max_input_len", "max_seq_len"):
            value = profile_section.get(key)
            if value is None:
                continue
            try:
                setattr(arch.engine_profile, key, int(value))
            except (ValueError, TypeError):
                logger.warning("Cannot set engine_profile.%s = %r", key, value)
    elif manifest_data:
        logger.info(
            "Manifest has no engine_profile section; runtime profile validation is disabled"
        )

    applied = []
    for k, v in arch_section.items():
        if hasattr(arch, k):
            expected_type = type(getattr(arch, k))
            try:
                setattr(arch, k, expected_type(v))
                applied.append(f"{k}={v}")
            except (ValueError, TypeError):
                pass

    if applied:
        logger.info("Model architecture: %s", ", ".join(applied))

    return arch


def _detect_from_tokenizer(tokenizer_dir: str) -> dict:
    """Fallback: extract model dimensions from HuggingFace config.json."""
    config_path = Path(tokenizer_dir) / "config.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            hf_cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", config_path, e)
        return {}

    talker = hf_cfg.get("talker_config", {})
    if not talker:
        return {}

    detected = {
        "num_layers": talker.get("num_hidden_layers"),
        "hidden_size": talker.get("hidden_size"),
        "kv_heads": talker.get("num_key_value_heads"),
        "head_dim": talker.get("head_dim"),
        "codec_vocab_size": talker.get("vocab_size"),
    }
    detected = {k: v for k, v in detected.items() if v is not None}

    if detected:
        logger.info(
            "Auto-detected from %s: %s",
            config_path.name,
            ", ".join(f"{k}={v}" for k, v in detected.items()),
        )
    return detected


# ---------------------------------------------------------------------------
# Public: convert to backend ModelConfig
# ---------------------------------------------------------------------------


def torch_dtype(dtype_str: str) -> torch.dtype:
    import torch

    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    return mapping.get(dtype_str, torch.bfloat16)


def to_model_config(arch: ModelArchConfig, cfg: EngineConfig):
    """Convert ModelArchConfig + EngineConfig → backend.kv_cache_pool.ModelConfig.

    I/O dtype uses triton_io_float_dtype (actual TRT tensor format) when set,
    falling back to arch.dtype (engine_dtype).  This matters when engine_dtype
    differs from triton_io_float_dtype, e.g. engine_dtype=fp32 with
    triton_io_float_dtype=bf16: TRT computes in fp32 but the I/O tensors are
    bf16, so the executor must pass bf16 tensors.
    """
    from .backend.kv_cache_pool import ModelConfig

    io_dtype_str = arch.engine_profile.triton_io_float_dtype or arch.dtype
    return ModelConfig(
        num_layers=arch.num_layers,
        kv_heads=arch.kv_heads,
        head_dim=arch.head_dim,
        max_seq_len=cfg.scheduler.max_seq_len,
        hidden_size=arch.hidden_size,
        dtype=torch_dtype(io_dtype_str),
        codec_vocab_size=arch.codec_vocab_size,
        logits_topk=arch.logits_topk,
        cp_num_stages=arch.cp_num_stages,
        n_c2w_layers=arch.n_c2w_layers,
        c2w_kv_heads=arch.c2w_kv_heads,
        c2w_head_dim=arch.c2w_head_dim,
        c2w_sliding_window=arch.c2w_sliding_window,
        n_c2w_conv_states=arch.n_c2w_conv_states,
        n_c2w_transconv_states=arch.n_c2w_transconv_states,
    )
