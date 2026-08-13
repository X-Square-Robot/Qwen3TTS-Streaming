from io import StringIO

from tools.validation.progress_oracle import (
    ProgressOracleRecord,
    read_oracle_jsonl,
    write_oracle_jsonl,
)


def test_oracle_jsonl_round_trip_has_shared_coordinate_shape():
    records = [
        ProgressOracleRecord("case", 0, 0, 0, "ema_proxy"),
        ProgressOracleRecord("case", 120, 4, 4, "ema_proxy"),
    ]
    stream = StringIO()
    write_oracle_jsonl(records, stream)
    assert list(read_oracle_jsonl(stream.getvalue().splitlines())) == records
