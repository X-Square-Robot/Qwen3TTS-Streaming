"""Gate-protected replay of dynamically frozen Triton commit groups."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qwen3tts_protocol import OutputPolicy, VADPolicy

from .arm_types import AudioChunkRecord, CollectedRun
from .artifacts import discover_run_records, read_json, write_json
from .collection import collect_arm_runs, initialize_experiment
from .models import ArmKind, RunStatus
from .preflight import sha256_file
from .reference_groups import validate_reference_groups
from .reporting import REPORT_PROVENANCE_PATH
from .truth_gate import require_primary_truth_gate


MATCHED_GENERATION = {
    "non_streaming_mode": False,
    "do_sample": True,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_temperature": 0.9,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "max_new_tokens": 512,
}
GREEDY_GENERATION = {
    **MATCHED_GENERATION,
    "do_sample": False,
    "subtalker_dosample": False,
}


def prepare_matched_experiment(primary_root: Path, *, mode: str = "sample") -> Path:
    """Create a nested experiment only after the locked primary gate passes."""

    if mode not in {"sample", "greedy"}:
        raise ValueError("matched mode must be 'sample' or 'greedy'")
    primary_report = require_primary_truth_gate(primary_root)
    primary_manifest = read_json(primary_root / "manifest.json")
    groups_payload = read_json(primary_root / "reference_groups.json")
    source_text = str(primary_manifest["text"]["text"])
    groups = validate_reference_groups(
        list(groups_payload.get("groups") or []), source_text
    )
    if int(groups_payload.get("group_count", -1)) != len(groups):
        raise RuntimeError("frozen group_count disagrees with groups")

    matched_root = primary_root / "matched" / mode
    initialize_experiment(
        matched_root,
        text_file=Path(primary_manifest["text"]["path"]),
        seeds=primary_manifest["seeds"],
        identities={
            "parent_manifest_sha256": sha256_file(primary_root / "manifest.json"),
            "parent_reference_groups_sha256": sha256_file(
                primary_root / "reference_groups.json"
            ),
            "parent_final_labels_sha256": sha256_file(
                primary_root / "review" / "private" / "final_labels.json"
            ),
            "parent_report_sha256": sha256_file(
                primary_root / "report" / "report.json"
            ),
            "parent_report_provenance_sha256": sha256_file(
                primary_root / REPORT_PROVENANCE_PATH
            ),
            "parent_conclusion": primary_report["conclusion"],
            "mode": mode,
            "frozen_reference_groups": groups_payload,
            "sampling": MATCHED_GENERATION if mode == "sample" else GREEDY_GENERATION,
            "endpoint_input_mode": "long_segment",
            # LONG_SEGMENT/AUTO is the protocol contract for one committed
            # synthesis group per text packet.  NONE routes packets through
            # the ordinary streaming accumulator, which may merge adjacent
            # packets or split them again and therefore is not a matched
            # boundary replay.
            "endpoint_group_policy": "auto",
            "vad_enabled": False,
            "delivery": "firehose",
            "asr": dict(primary_manifest.get("identities", {}).get("asr") or {}),
        },
    )
    write_json(matched_root / "reference_groups.json", groups_payload)
    return matched_root


class MatchedEndpointArm:
    """View an endpoint adapter as a frozen-group arm for collection."""

    def __init__(self, adapter: Any, groups: Sequence[Mapping[str, Any]]) -> None:
        self.adapter = adapter
        self.arm = ArmKind(adapter.arm)
        self.groups = tuple(dict(group) for group in groups)
        self.packets = tuple(str(group["text"]) for group in self.groups)
        self.output_policy = OutputPolicy(
            vad=VADPolicy(enabled=False, strategy="disabled"),
            config={"delivery": "firehose"},
        )

    def collect(self, text: str, *, session_id: str, seed: int = 0) -> CollectedRun:
        validate_reference_groups(self.groups, text)
        run = self.adapter.collect_packets(
            self.packets,
            session_id=session_id,
            seed=seed,
            input_mode="long_segment",
            group_policy="auto",
            output_policy=self.output_policy,
        )
        committed = [
            event
            for event in run.events
            if str(event.get("type")) == "text_boundary_commit"
        ]
        committed_text = tuple(str(event.get("text") or "") for event in committed)
        committed_ids = tuple(int(event.get("segment_id", -1)) for event in committed)
        expected_ids = tuple(range(len(self.packets)))
        if committed_text != self.packets or committed_ids != expected_ids:
            return replace(
                run,
                status=RunStatus.ERROR,
                error=(
                    "matched boundary replay was not preserved by the endpoint: "
                    f"expected {len(self.packets)} frozen commits, observed "
                    f"{len(committed)}"
                ),
                eos_reason="matched_boundary_mismatch",
            )
        return run


def collect_matched_endpoint(
    primary_root: Path,
    adapter: Any,
    *,
    mode: str = "sample",
    runtime_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    matched_root = prepare_matched_experiment(primary_root, mode=mode)
    groups = read_json(matched_root / "reference_groups.json")["groups"]
    view = MatchedEndpointArm(adapter, groups)
    return collect_arm_runs(
        matched_root,
        view,
        arm_metadata={
            "matched_replay": True,
            "mode": mode,
            "runtime_must_supply_matched_sampling": True,
            **dict(runtime_metadata or {}),
        },
    )


def _combine_official_segments(
    runs: Sequence[CollectedRun],
    groups: Sequence[Mapping[str, Any]],
    *,
    trial_seed: int,
    session_id: str,
) -> CollectedRun:
    arrays: list[np.ndarray] = []
    chunks: list[AudioChunkRecord] = []
    events: list[dict[str, Any]] = []
    cursor = 0
    for segment_index, (run, group) in enumerate(zip(runs, groups, strict=True)):
        audio = np.asarray(run.samples, dtype=np.float32).reshape(-1)
        arrays.append(audio)
        for chunk in run.audio_chunks:
            chunks.append(
                replace(
                    chunk,
                    sequence_index=len(chunks),
                    sample_start=cursor + chunk.sample_start,
                    sample_end=cursor + chunk.sample_end,
                    output_sample_start=(
                        cursor + chunk.output_sample_start
                        if chunk.output_sample_start is not None
                        else None
                    ),
                    output_sample_end=(
                        cursor + chunk.output_sample_end
                        if chunk.output_sample_end is not None
                        else None
                    ),
                    meta={**chunk.meta, "segment_index": str(segment_index)},
                )
            )
        cursor += audio.size
        common_meta = {
            "raw_codepoint_start": str(group["raw_start"]),
            "raw_codepoint_end": str(group["raw_end"]),
            "sampling_seed": str(run.sampling_seed),
            "matched_official": "true",
        }
        events.extend(
            [
                {
                    "type": "text_boundary_commit",
                    "session_id": session_id,
                    "segment_id": segment_index,
                    "text": group["text"],
                    "message": "",
                    "audio": None,
                    "meta": common_meta,
                },
                {
                    "type": "segment_end",
                    "session_id": session_id,
                    "segment_id": segment_index,
                    "text": group["text"],
                    "message": "",
                    "audio": None,
                    "meta": common_meta,
                },
                {
                    "type": "text_progress",
                    "session_id": session_id,
                    "segment_id": segment_index,
                    "text": "",
                    "message": "",
                    "audio": None,
                    "meta": {
                        **common_meta,
                        "alignment_final": "true",
                        "progress_final": "true",
                        "output_sample_end": str(cursor),
                    },
                },
            ]
        )
    failures = [run.error for run in runs if run.status is not RunStatus.OK]
    samples = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
    return CollectedRun(
        arm=ArmKind.PYTORCH_0818,
        seed=trial_seed,
        session_id=session_id,
        status=RunStatus.TTS_FAILED if failures else RunStatus.OK,
        samples=samples,
        sample_rate=24_000,
        duration_s=samples.size / 24_000.0,
        events=events,
        audio_chunks=chunks,
        error="; ".join(str(value) for value in failures) if failures else None,
        total_ms=sum(run.total_ms for run in runs),
        terminal_event="error" if failures else "done",
        eos_reason="matched_segment_failure" if failures else "all_segments_complete",
    )


class MatchedOfficialArm:
    def __init__(
        self,
        adapter: Any,
        groups: Sequence[Mapping[str, Any]],
        *,
        generation_kwargs: Mapping[str, Any],
    ) -> None:
        self.adapter = adapter
        self.arm = ArmKind.PYTORCH_0818
        self.groups = tuple(dict(group) for group in groups)
        self.generation_kwargs = dict(generation_kwargs)

    def collect(self, text: str, *, session_id: str, seed: int = 0) -> CollectedRun:
        validate_reference_groups(self.groups, text)
        runs = [
            self.adapter.collect_segment(
                str(group["text"]),
                session_id,
                seed,
                segment_index,
                self.generation_kwargs,
            )
            for segment_index, group in enumerate(self.groups)
        ]
        return _combine_official_segments(
            runs, self.groups, trial_seed=seed, session_id=session_id
        )


def collect_matched_official(
    primary_root: Path,
    adapter: Any,
    *,
    mode: str = "sample",
) -> list[dict[str, Any]]:
    matched_root = prepare_matched_experiment(primary_root, mode=mode)
    groups = read_json(matched_root / "reference_groups.json")["groups"]
    generation = MATCHED_GENERATION if mode == "sample" else GREEDY_GENERATION
    view = MatchedOfficialArm(adapter, groups, generation_kwargs=generation)
    return collect_arm_runs(
        matched_root,
        view,
        arm_metadata={
            "matched_replay": True,
            "mode": mode,
            "generation_kwargs": generation,
        },
    )


def greedy_hash_comparison(primary_root: Path) -> dict[str, Any]:
    """Compare deterministic PCM identities without pretending they are truth."""

    root = primary_root / "matched" / "greedy"
    records = discover_run_records(root)
    by_seed: dict[int, dict[str, str | None]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), {})[str(record["arm"])] = record.get(
            "pcm_s16le_sha256"
        )
    comparisons = []
    for seed, hashes in sorted(by_seed.items()):
        current = hashes.get(ArmKind.CURRENT_HEAD.value)
        triton = hashes.get(ArmKind.TRITON_0818.value)
        comparisons.append(
            {
                "seed": seed,
                "current_sha256": current,
                "triton_sha256": triton,
                "current_equals_triton": bool(current and current == triton),
            }
        )
    result = {
        "comparisons": comparisons,
        "all_current_triton_equal": bool(comparisons)
        and all(item["current_equals_triton"] for item in comparisons),
        "diagnostic_only": True,
    }
    write_json(root / "greedy_hash_comparison.json", result)
    return result


__all__ = [
    "GREEDY_GENERATION",
    "MATCHED_GENERATION",
    "MatchedEndpointArm",
    "MatchedOfficialArm",
    "collect_matched_endpoint",
    "collect_matched_official",
    "greedy_hash_comparison",
    "prepare_matched_experiment",
]
