"""End-to-end orchestration for deterministic TTS regression sweeps."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.validation._bootstrap import bootstrap_tool_imports

from .asr import import_funasr_client, transcribe_wav
from .metrics import character_error_metrics, classify_trial
from .models import (
    SCREENING_LABEL,
    AsrStatus,
    ChunkPattern,
    SweepConfig,
    TextPacket,
    build_text_packets,
)
from .report import (
    json_write,
    persist_trial,
    rewrite_trial_sidecar,
    summarize_records,
)
from .synthesis import synthesize_once

REPO_ROOT = bootstrap_tool_imports()


def default_output_dir(arm: str, pattern: ChunkPattern) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    safe_arm = "".join(character if character.isalnum() else "-" for character in arm)
    return (
        REPO_ROOT
        / "workspace"
        / "hallucination_sweep"
        / f"{safe_arm or 'unspecified'}_{pattern.value}_{timestamp}"
    )


def _resolve_funasr_client_src(configured: Path | None) -> Path | None:
    if configured is not None:
        return configured
    raw_src = os.environ.get("FUNASR_CLIENT_SRC", "")
    if raw_src:
        return Path(raw_src)
    sibling = REPO_ROOT.parent / "FunAsrNano-Streaming" / "client" / "src"
    return sibling if sibling.is_dir() else None


def _run_config_payload(
    config: SweepConfig,
    packets: Sequence[TextPacket],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "arm": config.arm,
        "endpoint": config.endpoint,
        "transport": config.transport,
        "speaker": config.speaker,
        "input_mode": config.input_mode,
        "group_policy": config.group_policy,
        "pattern": config.pattern.value,
        "body_text": config.body_text,
        "body_text_repr": repr(config.body_text),
        "leading_prefix": config.leading_prefix,
        "leading_prefix_repr": repr(config.leading_prefix),
        "leading_prefix_codepoints": [
            f"U+{ord(char):04X}" for char in config.leading_prefix
        ],
        "packets": [asdict(packet) for packet in packets],
        "num_trials": config.num_trials,
        "start": config.start,
        "sid_prefix": config.sid_prefix,
        "thresholds": asdict(config.thresholds),
        "confidence": config.confidence,
        "asr": {
            "enabled": bool(config.asr_url),
            "url": config.asr_url or None,
            "language": config.asr_language,
            "hotwords": [],
            "partial_mode": "off",
            "new_connection_and_stream_per_wav": True,
            "max_duration_s": config.asr_max_duration_s,
        },
        "metadata": config.metadata,
        "screening_label": SCREENING_LABEL,
    }


def _load_optional_asr(config: SweepConfig) -> Any:
    if not config.asr_url:
        return None
    return import_funasr_client(_resolve_funasr_client_src(config.funasr_client_src))


def _connect_tts(config: SweepConfig) -> Any:
    from qwen3tts import TTSClient

    return TTSClient.connect(
        config.endpoint,
        transport=config.transport,
        timeout=config.timeout,
    )


def _synthesize_records(
    client: Any,
    config: SweepConfig,
    packets: Sequence[TextPacket],
    run_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for offset in range(config.num_trials):
        sid_index = config.start + offset
        session_id = f"{config.sid_prefix}-{sid_index:04d}"
        result = synthesize_once(
            client,
            packets,
            speaker=config.speaker,
            session_id=session_id,
            input_mode=config.input_mode,
            group_policy=config.group_policy,
            timeout=config.timeout,
        )
        record = persist_trial(
            run_dir,
            trial_index=sid_index,
            session_id=session_id,
            pattern=config.pattern,
            packets=packets,
            result=result,
        )
        records.append(record)
        print(
            f"[{offset + 1:03d}/{config.num_trials:03d}] {session_id} "
            f"{result.status.value} duration={result.duration_s:.3f}s "
            f"ttft={result.ttft_ms}ms"
        )
    return records


def _score_record(
    record: dict[str, Any],
    *,
    config: SweepConfig,
    asr_client_class: Any,
    run_dir: Path,
) -> None:
    wav_relative = record.get("artifacts", {}).get("wav")
    if not config.asr_url:
        record["asr"] = {"status": AsrStatus.DISABLED.value}
    elif not isinstance(wav_relative, str):
        record["asr"] = {
            "status": AsrStatus.SKIPPED.value,
            "reason": "no_wav",
        }
    elif float(record["duration_s"]) >= config.asr_max_duration_s:
        record["asr"] = {
            "status": AsrStatus.SKIPPED.value,
            "reason": "duration_limit",
            "limit_s": config.asr_max_duration_s,
        }
    else:
        asr = asyncio.run(
            transcribe_wav(
                asr_client_class,
                run_dir / wav_relative,
                uri=config.asr_url,
                language=config.asr_language,
                chunk_ms=config.asr_chunk_ms,
            )
        )
        if asr.get("status") == AsrStatus.OK.value:
            asr["metrics"] = character_error_metrics(
                config.body_text, str(asr.get("transcript", ""))
            )
        record["asr"] = asr
    record["screening"] = classify_trial(
        record,
        thresholds=config.thresholds,
        asr_requested=bool(config.asr_url),
    )
    rewrite_trial_sidecar(run_dir, record)


def run_sweep(config: SweepConfig) -> int:
    """Execute one complete sweep and return its CLI-compatible exit code."""

    packets = build_text_packets(
        config.body_text,
        leading_prefix=config.leading_prefix,
        pattern=config.pattern,
        split_delay_ms=config.split_delay_ms,
    )
    try:
        asr_client_class = _load_optional_asr(config)
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    run_dir = (
        config.output_dir or default_output_dir(config.arm, config.pattern)
    ).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"ERROR: output directory is not empty: {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = _run_config_payload(config, packets)
    json_write(run_dir / "run_config.json", run_config)

    print(f"output:    {run_dir}")
    print(f"endpoint:  {config.endpoint} ({config.transport})")
    print(f"arm:       {config.arm}")
    print(f"pattern:   {config.pattern.value}")
    print(f"packets:   {[repr(packet.text) for packet in packets]}")
    print(
        f"sessions:  {config.sid_prefix}-{config.start:04d} .. "
        f"({config.num_trials} total)"
    )

    try:
        client = _connect_tts(config)
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: cannot connect to TTS endpoint: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        records = _synthesize_records(client, config, packets, run_dir)
    finally:
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001 - cleanup must not hide results
            print(
                f"WARNING: failed to close TTS client: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    for record in records:
        _score_record(
            record,
            config=config,
            asr_client_class=asr_client_class,
            run_dir=run_dir,
        )

    summary = {
        **run_config,
        "completed_at": datetime.now().astimezone().isoformat(),
        "statistics": summarize_records(records, confidence=config.confidence),
        "trials": records,
    }
    json_write(run_dir / "summary.json", summary)
    stats = summary["statistics"]
    interval = stats["asr_supported_suspect_rate_wilson"]
    print(f"summary:   {run_dir / 'summary.json'}")
    if interval["total"]:
        print(
            f"{SCREENING_LABEL}: {interval['successes']}/{interval['total']} "
            f"(rate={interval['rate']}, {config.confidence:.0%} Wilson "
            f"CI=[{interval['low']}, {interval['high']}])"
        )
    else:
        print(
            f"{SCREENING_LABEL}: N/A "
            "(no ASR-scored or duration-threshold-positive trials)"
        )
    return 0 if stats["tts_failed_or_no_audio"] == 0 else 1
