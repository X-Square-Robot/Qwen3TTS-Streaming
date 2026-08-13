"""Unified JSONL oracle contract for future text/audio alignment evaluators.

The producer can be the EMA proxy, an ASR aligner, a human timestamp export,
or a codec aligner.  Keeping the record shape independent of the producer
means both SDKs consume the same anchor coordinates during migration.

Example::

    python tools/validation/progress_oracle.py --input anchors.jsonl --check
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO


@dataclass(frozen=True)
class ProgressOracleRecord:
    case_id: str
    output_sample: int
    raw_codepoint: int
    normalized_codepoint: int
    source: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_mapping(cls, value: dict) -> "ProgressOracleRecord":
        return cls(
            case_id=str(value["case_id"]),
            output_sample=int(value["output_sample"]),
            raw_codepoint=int(value["raw_codepoint"]),
            normalized_codepoint=int(value["normalized_codepoint"]),
            source=str(value["source"]),
        )


def read_oracle_jsonl(stream: Iterable[str]) -> Iterator[ProgressOracleRecord]:
    for line_no, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record must be an object")
            yield ProgressOracleRecord.from_mapping(value)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid progress oracle record at line {line_no}") from exc


def write_oracle_jsonl(
    records: Iterable[ProgressOracleRecord], stream: TextIO
) -> None:
    previous: dict[str, ProgressOracleRecord] = {}
    for record in records:
        old = previous.get(record.case_id)
        if old is not None and (
            record.output_sample < old.output_sample
            or record.raw_codepoint < old.raw_codepoint
            or record.normalized_codepoint < old.normalized_codepoint
        ):
            raise ValueError(f"non-monotonic oracle record for {record.case_id!r}")
        if record.output_sample < 0 or record.raw_codepoint < 0 or record.normalized_codepoint < 0:
            raise ValueError("oracle coordinates must be non-negative")
        stream.write(record.to_json() + "\n")
        previous[record.case_id] = record


def validate_oracle(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        records = list(read_oracle_jsonl(stream))
    # Reuse the writer's monotonicity and coordinate checks without emitting
    # another file; this also makes the CLI useful in CI smoke gates.
    import io

    write_oracle_jsonl(records, io.StringIO())
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate records")
    args = parser.parse_args()
    count = validate_oracle(args.input)
    print(f"validated {count} progress oracle records from {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
