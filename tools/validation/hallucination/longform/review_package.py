"""Arm-blind review package creation and two-reviewer adjudication workflow."""

from __future__ import annotations

import csv
import hashlib
import random
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import read_json, write_json
from .models import ArmKind, ReviewLabel, parse_reference_sentences
from .preflight import sha256_file
from .review_provenance import (
    PUBLIC_CONTROL_FILES,
    finalize_review_evidence,
    verify_schema_v3_public_controls,
)
from .review_selection import (
    compute_round2_selection,
    selection_payload,
    validate_frozen_round2_selection,
    validated_review_rows,
)
from .review_ui import build_review_page


SEVERE_LABELS = frozenset(
    {
        ReviewLabel.SINGLE_UNIT_LOOP,
        ReviewLabel.ABNORMAL_NOISE,
        ReviewLabel.UNSUPPORTED_SPEECH,
    }
)
_PUBLIC_FIELDS = (
    "blind_id",
    "reference_text",
    "previous_reference",
    "next_reference",
    "audio",
    "context_audio",
    "label",
    "notes",
)
_PRIVATE_ASSET_HASH_FIELDS = (
    ("audio", "audio_sha256"),
    ("context_audio", "context_audio_sha256"),
)
def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _verified_public_asset_path(public_root: Path, relative: object) -> Path:
    value = str(relative or "").strip()
    if not value:
        raise ValueError("blind review asset path is empty")
    root = public_root.resolve()
    path = (public_root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"blind review asset escapes public directory: {value}"
        ) from exc
    if not path.is_file():
        raise ValueError(f"blind review asset is missing: {value}")
    return path


def _verify_blind_asset_hashes(output_root: Path) -> None:
    """Verify public controls and audio according to the private-key schema.

    Schema v1 had no hashes, v2 sealed audio, and v3 additionally seals the four
    original public control files.  Older frozen packages remain readable while
    every hash promised by their schema is checked fail-closed.
    """

    review_root = output_root / "review"
    key = read_json(review_root / "private" / "key.json")
    schema_version = int(key.get("schema_version", 1))
    verify_schema_v3_public_controls(
        output_root,
        key,
        require_schema_v3=False,
    )
    public_manifest = read_json(review_root / "public" / "manifest.json")
    public_rows = list(public_manifest.get("rows") or [])
    public_by_id = {str(row.get("blind_id", "")): row for row in public_rows}
    if len(public_by_id) != len(public_rows) or "" in public_by_id:
        raise ValueError("public review manifest has missing or duplicate blind_id")

    for item in key.get("rows") or []:
        blind_id = str(item.get("blind_id", "")).strip()
        expected_hashes = [
            item.get(hash_field) for _, hash_field in _PRIVATE_ASSET_HASH_FIELDS
        ]
        has_any_hash = any(value is not None for value in expected_hashes)
        if not has_any_hash and schema_version < 2:
            continue
        if not blind_id or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
            for value in expected_hashes
        ):
            raise ValueError(
                f"private review key has incomplete asset hashes: {blind_id!r}"
            )
        public = public_by_id.get(blind_id)
        if public is None:
            raise ValueError(
                f"private review key contains unknown blind_id: {blind_id}"
            )
        for public_field, hash_field in _PRIVATE_ASSET_HASH_FIELDS:
            asset = _verified_public_asset_path(
                review_root / "public", public.get(public_field)
            )
            actual = sha256_file(asset)
            expected = str(item[hash_field]).lower()
            if actual != expected:
                raise ValueError(
                    "blind review asset hash mismatch: "
                    f"blind_id={blind_id}, asset={public_field}, "
                    f"expected={expected}, actual={actual}"
                )


def _opaque_id(index: int, *, review_seed: int) -> str:
    digest = hashlib.blake2s(
        f"{review_seed}:{index}".encode("ascii"), digest_size=5
    ).hexdigest()
    return f"sample-{index + 1:04d}-{digest}"


def _review_instructions() -> str:
    return """# 0818 长文本盲审说明

本目录不包含 arm、session、seed 或原始路径。请先播放 `audio`，需要前后文时再播放
`context_audio`。参考句只用于判断参考外内容，不要用 ASR 指标代替听感。

可选标签：

- `OK`：无以下问题；
- `SINGLE_UNIT_LOOP`：参考外同一音节/字连续重复至少 3 次，或持续循环至少 0.5 秒；
- `ABNORMAL_NOISE`：异常非语言噪音持续至少 0.5 秒；
- `UNSUPPORTED_SPEECH`：确认存在参考外语音插入；
- `OMISSION`：仅遗漏，不计严重幻觉分子；
- `MISPRONUNCIATION`：仅误读，不计严重幻觉分子；
- `UNSCORABLE`：确实无法裁决，不能按 clean 处理。

推荐打开 `index.html`：页面会在浏览器本地自动保存进度，可导入之前导出的 CSV 恢复，
并在全部样本完成后导出 `review_round1_filled.csv`。请勿覆盖本目录中作为封存模板的
`review_round1.csv`；若使用表格软件，请先复制模板再填写。`notes` 可记录问题时间点。

第一位审阅人检查全部样本。第二位审阅人的清单由工具根据第一轮结果生成：全部非 `OK`
（含 `UNSCORABLE`），再加固定随机 10% 的 `OK` 样本。分歧仍在隐藏 arm 的情况下进入裁决。
"""


def _html_page(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the sealed public rows without exposing private coordinates."""

    return build_review_page(rows, public_fields=_PUBLIC_FIELDS)


def _validate_manifest_observation_grid(
    output_root: Path,
    observations: Sequence[Mapping[str, Any]],
) -> None:
    """Fail before sealing a partial blind package for a formal experiment."""

    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        # Small library-level fixtures may exercise blinding independently of
        # the formal experiment lifecycle.  The CLI always initializes first.
        return
    manifest = read_json(manifest_path)
    seeds = tuple(int(seed) for seed in manifest.get("seeds") or ())
    reference_text = str((manifest.get("text") or {}).get("text", ""))
    ordinals = tuple(
        sentence.ordinal for sentence in parse_reference_sentences(reference_text)
    )
    if len(seeds) != 3 or len(set(seeds)) != 3 or not ordinals:
        raise RuntimeError(
            "experiment manifest lacks the frozen three-seed sentence grid"
        )
    expected = {
        (arm.value, seed, ordinal)
        for arm in ArmKind
        for seed in seeds
        for ordinal in ordinals
    }
    observed: set[tuple[str, int, int]] = set()
    duplicates: list[tuple[str, int, int]] = []
    for observation in observations:
        try:
            key = (
                ArmKind(observation["arm"]).value,
                int(observation["seed"]),
                int(observation["sentence_ordinal"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "sentence observation has invalid arm/seed/ordinal"
            ) from exc
        if key in observed:
            duplicates.append(key)
        observed.add(key)
    missing = expected - observed
    extra = observed - expected
    if duplicates or missing or extra:
        raise RuntimeError(
            "refusing to seal an incomplete blind package: "
            f"expected={len(expected)}, observed={len(observed)}, "
            f"missing={len(missing)}, extra={len(extra)}, duplicates={len(duplicates)}"
        )


def _validate_formal_scoring_readiness(
    output_root: Path,
    observations: Sequence[Mapping[str, Any]],
) -> None:
    """Prevent bypassing failed ASR/TTS readiness in a formal experiment."""

    if not (output_root / "manifest.json").is_file():
        return
    summary_path = output_root / "scoring_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            "refusing to seal a formal blind package without scoring_summary.json"
        )
    summary = read_json(summary_path)
    if summary.get("ready_for_review") is not True:
        reasons = ", ".join(str(item) for item in summary.get("reasons") or [])
        raise RuntimeError(
            "refusing to seal a formal blind package while scoring is incomplete: "
            f"{reasons or 'unknown reason'}"
        )
    blocked = [
        int(observation.get("sentence_ordinal", -1))
        for observation in observations
        if observation.get("valid_for_review") is not True
    ]
    if blocked:
        raise RuntimeError(
            "refusing to seal a formal blind package with non-review-ready "
            f"observations: count={len(blocked)}"
        )
    missing_audio = []
    for observation in observations:
        for field in ("clip", "context_clip"):
            relative = observation.get(field)
            if not relative or not (output_root / str(relative)).is_file():
                missing_audio.append(
                    (int(observation.get("sentence_ordinal", -1)), field)
                )
    if missing_audio:
        raise RuntimeError(
            "refusing to seal a formal blind package with missing review audio: "
            f"count={len(missing_audio)}"
        )


def build_review_package(
    output_root: Path,
    *,
    review_seed: int = 8182026,
) -> dict[str, Any]:
    """Randomize sentence clips and create separate public/private manifests."""

    observations = list(
        read_json(output_root / "sentence_observations.json").get("observations") or []
    )
    if not observations:
        raise RuntimeError("no sentence observations are available")
    _validate_manifest_observation_grid(output_root, observations)
    _validate_formal_scoring_readiness(output_root, observations)
    review_root = output_root / "review"
    if review_root.exists() and any(review_root.iterdir()):
        raise FileExistsError(f"review package already exists: {review_root}")
    public_root = review_root / "public"
    private_root = review_root / "private"
    audio_root = public_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)

    order = list(range(len(observations)))
    random.Random(review_seed).shuffle(order)
    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for review_index, source_index in enumerate(order):
        observation = dict(observations[source_index])
        blind_id = _opaque_id(review_index, review_seed=review_seed)
        clip_relative = observation.get("clip")
        context_relative = observation.get("context_clip")
        public_audio = ""
        public_context = ""
        if clip_relative and context_relative:
            clip_target = audio_root / f"{blind_id}.wav"
            context_target = audio_root / f"{blind_id}_context.wav"
            shutil.copyfile(output_root / str(clip_relative), clip_target)
            shutil.copyfile(output_root / str(context_relative), context_target)
            public_audio = str(clip_target.relative_to(public_root))
            public_context = str(context_target.relative_to(public_root))
        public_rows.append(
            {
                "blind_id": blind_id,
                "reference_text": observation.get("reference_text", ""),
                "previous_reference": observation.get("previous_reference", ""),
                "next_reference": observation.get("next_reference", ""),
                "audio": public_audio,
                "context_audio": public_context,
                "label": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "blind_id": blind_id,
                "observation": observation,
                "audio_sha256": sha256_file(clip_target) if public_audio else None,
                "context_audio_sha256": (
                    sha256_file(context_target) if public_context else None
                ),
            }
        )

    write_json(
        public_root / "manifest.json",
        {
            "schema_version": 1,
            "sample_count": len(public_rows),
            "arm_blind": True,
            "allowed_labels": [item.value for item in ReviewLabel],
            "rows": public_rows,
        },
    )
    _write_csv(public_root / "review_round1.csv", public_rows, _PUBLIC_FIELDS)
    (public_root / "README.md").write_text(_review_instructions(), encoding="utf-8")
    (public_root / "index.html").write_text(_html_page(public_rows), encoding="utf-8")
    public_control_sha256 = {
        relative: sha256_file(public_root / relative)
        for relative in PUBLIC_CONTROL_FILES
    }
    write_json(
        private_root / "key.json",
        {
            "schema_version": 3,
            "review_seed": review_seed,
            "do_not_share_with_reviewers": True,
            "public_control_sha256": public_control_sha256,
            "rows": key_rows,
        },
    )
    return {"sample_count": len(public_rows), "public_root": str(public_root)}




def build_second_review(
    output_root: Path,
    round1_csv: Path,
    *,
    reviewer1_id: str,
    reviewer2_id: str,
    selection_seed: int = 8182027,
) -> dict[str, Any]:
    """Select all non-OK/uncertain plus a deterministic 10% of OK negatives."""

    rows = validated_review_rows(round1_csv, require_all=True)
    _verify_blind_asset_hashes(output_root)
    public_manifest = read_json(output_root / "review" / "public" / "manifest.json")
    public_rows = list(public_manifest.get("rows") or [])
    selection = compute_round2_selection(
        public_rows,
        rows,
        selection_seed=selection_seed,
    )
    by_id = {str(row["blind_id"]): row for row in public_rows}
    round2: list[dict[str, Any]] = []
    for blind_id in selection.blind_ids:
        public = dict(by_id[blind_id])
        public["label"] = ""
        public["notes"] = ""
        round2.append(public)
    target = output_root / "review" / "public" / "review_round2.csv"
    _write_csv(target, round2, _PUBLIC_FIELDS)
    write_json(
        output_root / "review" / "private" / "round2_selection.json",
        selection_payload(
            selection,
            selection_seed=selection_seed,
            reviewer1_id=reviewer1_id,
            reviewer2_id=reviewer2_id,
        ),
    )
    return {
        "selected": len(round2),
        "mandatory": len(selection.mandatory_blind_ids),
        "random_negative": len(selection.random_negative_blind_ids),
        "path": str(target),
    }


def build_adjudication(
    output_root: Path,
    round1_csv: Path,
    round2_csv: Path,
) -> dict[str, Any]:
    """Emit only reviewer disagreements, with no private arm fields."""

    _verify_blind_asset_hashes(output_root)
    round1_rows = validated_review_rows(round1_csv, require_all=True)
    round2 = validated_review_rows(round2_csv, require_all=True)
    validate_frozen_round2_selection(
        output_root,
        round1_rows=round1_rows,
        round2_rows=round2,
        require_formal=(output_root / "manifest.json").is_file(),
    )
    round1 = {row["blind_id"]: row for row in round1_rows}
    public = {
        str(row["blind_id"]): row
        for row in read_json(output_root / "review" / "public" / "manifest.json")[
            "rows"
        ]
    }
    disagreements: list[dict[str, Any]] = []
    for second in round2:
        first = round1.get(second["blind_id"])
        if first is None:
            raise ValueError(f"round2 contains unknown ID: {second['blind_id']}")
        if first["label"] == second["label"]:
            continue
        row = dict(public[second["blind_id"]])
        row.update(
            {
                "reviewer1_label": first["label"],
                "reviewer2_label": second["label"],
                "adjudicated_label": "",
                "adjudication_notes": "",
            }
        )
        disagreements.append(row)
    fields = (
        *_PUBLIC_FIELDS[:-2],
        "reviewer1_label",
        "reviewer2_label",
        "adjudicated_label",
        "adjudication_notes",
    )
    target = output_root / "review" / "public" / "adjudication.csv"
    _write_csv(target, disagreements, fields)
    return {"disagreement_count": len(disagreements), "path": str(target)}


def finalize_reviews(
    output_root: Path,
    round1_csv: Path,
    round2_csv: Path,
    adjudication_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Resolve reviews, write final labels, and seal their evidence chain."""

    _verify_blind_asset_hashes(output_root)
    return finalize_review_evidence(
        output_root,
        round1_csv,
        round2_csv,
        adjudication_csv,
        require_formal=(output_root / "manifest.json").is_file(),
    )


__all__ = [
    "SEVERE_LABELS",
    "build_adjudication",
    "build_review_package",
    "build_second_review",
    "finalize_reviews",
]
