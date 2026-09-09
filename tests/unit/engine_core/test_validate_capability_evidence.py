import json

from scripts.python.validate_capability_evidence import main


def test_release_evidence_cli_reports_and_enforces_required_capability(tmp_path, capsys):
    manifest = tmp_path / "triton_manifest.json"
    evidence = tmp_path / "capability_evidence.json"
    manifest.write_text(
        json.dumps(
            {
                "speech_state": {
                    "model_fingerprint": "model-v1",
                    "runtime_fingerprint": "runtime-v1",
                }
            }
        ),
        encoding="utf-8",
    )
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_fingerprint": "model-v1",
                "runtime_fingerprint": "runtime-v1",
                "performance": {"verified": True},
                "quality": {"offline_asr_verified": True},
                "native_cursor": {
                    "trt_numeric_verified": True,
                    "progress_e2e_verified": True,
                    "monotonic_verified": True,
                },
                "speech_state": {
                    "bundle_verified": True,
                    "trt_transfer_verified": True,
                    "successor_e2e_verified": True,
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--manifest",
            str(manifest),
            "--evidence",
            str(evidence),
            "--require-native-cursor",
            "--require-speech-state",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["native_cursor"]["verified"] is True
    assert output["speech_state"]["verified"] is True
