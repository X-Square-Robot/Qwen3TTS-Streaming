from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tools.validation.hallucination.longform.arm_types import CollectedRun
from tools.validation.hallucination.longform.artifacts import read_json, write_json
from tools.validation.hallucination.longform.models import ArmKind, RunStatus
from tools.validation.hallucination.longform.rootcause_replay import (
    load_frozen_replay,
    replay_endpoint,
    score_replay_cases,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    text_file = tmp_path / "verylong.txt"
    text_file.write_bytes("第一句。\n第二句。".encode("utf-8"))
    groups_file = tmp_path / "groups.json"
    write_json(
        groups_file,
        {
            "groups": [
                {
                    "segment_id": 0,
                    "raw_start": 0,
                    "raw_end": 4,
                    "text": "第一句。",
                },
                {
                    "segment_id": 1,
                    "raw_start": 4,
                    "raw_end": 9,
                    "text": "\n第二句。",
                },
            ]
        },
    )
    return text_file, groups_file


class _Endpoint:
    arm = ArmKind.CURRENT_HEAD

    def collect_packets(
        self,
        packets: tuple[str, ...],
        session_id: str,
        seed: int,
        *,
        input_mode: str,
        group_policy: str,
        output_policy: Any,
    ) -> CollectedRun:
        assert input_mode == "long_segment"
        assert group_policy == "auto"
        assert output_policy.vad.enabled is False
        events = [
            {
                "type": "text_boundary_commit",
                "session_id": session_id,
                "segment_id": index,
                "text": text,
                "meta": {},
            }
            for index, text in enumerate(packets)
        ]
        return CollectedRun(
            arm=self.arm,
            seed=seed,
            session_id=session_id,
            status=RunStatus.OK,
            samples=np.zeros(240, dtype=np.float32),
            sample_rate=24_000,
            duration_s=0.01,
            events=events,
            terminal_event="done",
        )


class _FakeASRClient:
    connections = 0

    def __init__(
        self,
        uri: str,
        *,
        hotwords: list[str],
        language: str,
        partial_mode: str,
        hotword_config: Any,
        version_check: bool,
        vad_params: dict[str, str],
    ) -> None:
        del uri, hotwords, language, partial_mode, hotword_config
        assert version_check is True
        assert vad_params == {"type": "fsmn"}
        type(self).connections += 1

    async def __aenter__(self) -> _FakeASRClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        del exc_info

    async def transcribe_file(
        self,
        path: str,
        *,
        chunk_ms: int,
        delivery_mode: str,
        pacing: str,
    ):
        del path
        assert chunk_ms == 960
        assert delivery_mode == "offline"
        assert pacing == "none"
        yield {
            "type": "segment_final",
            "segment": {
                "text": "第一句。第二句。",
                "start_ms": 0,
                "end_ms": 10,
            },
        }
        yield {"type": "stream_done"}


def test_rootcause_replay_preserves_exact_text_including_newline(
    tmp_path: Path,
) -> None:
    text_file, groups_file = _fixture(tmp_path)

    text, groups = load_frozen_replay(text_file, groups_file)
    records = replay_endpoint(
        _Endpoint(),
        text_file=text_file,
        groups_file=groups_file,
        output_dir=tmp_path / "cases",
        seeds=(17041,),
        runtime_mode="sample",
    )

    assert text == "第一句。\n第二句。"
    assert "".join(group["text"] for group in groups) == text
    assert records[0]["frozen_boundaries_preserved"] is True
    stored = read_json(
        tmp_path / "cases" / "current_head" / "seed_17041" / "run.json"
    )
    assert stored["replay_contract"]["group_policy"] == "auto"
    assert stored["frozen_boundaries_preserved"] is True


def test_rootcause_asr_uses_one_fresh_connection_per_wav(tmp_path: Path) -> None:
    text_file, groups_file = _fixture(tmp_path)
    cases_dir = tmp_path / "cases"
    replay_endpoint(
        _Endpoint(),
        text_file=text_file,
        groups_file=groups_file,
        output_dir=cases_dir,
        seeds=(17041, 28411),
        runtime_mode="greedy",
    )
    _FakeASRClient.connections = 0

    summaries = score_replay_cases(
        cases_dir,
        text_file=text_file,
        client_class=_FakeASRClient,
        asr_url="wss://example.test/asr",
    )

    assert _FakeASRClient.connections == 2
    assert [record["cer"] for record in summaries] == [0.0, 0.0]
    assert all(Path(record["sidecar"]).is_file() for record in summaries)
