import importlib.util
from pathlib import Path


REPO_ROOT = (
    Path(__file__).resolve().parents[3]
)  # tests/unit/engine_core/<file> -> repo root
# 3758ad7 moved the helper back to scripts/python (build_pipeline.sh looks
# for it there); keep this path in sync or these tests break at import.
HELPER_PATH = REPO_ROOT / "scripts" / "python" / "suggest_engine_profile.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("suggest_engine_profile", HELPER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_aware_profile_suggests_128_for_48g_1_7b(tmp_path):
    helper = _load_helper()
    exported = tmp_path / "exported"
    variant_dir = exported / "custom-1.7b"
    variant_dir.mkdir(parents=True)
    fixture = REPO_ROOT / "tools" / "data" / "triton_manifest_custom_1_7b.json"
    variant_dir.joinpath("triton_manifest.json").write_text(
        fixture.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = helper.suggest_profile(
        memory_mib=46068,
        exported_dir=exported,
        variants=["custom-1.7b"],
        engine_dtype="bf16",
        max_input_len=128,
        max_seq_len=512,
    )

    assert result["source"] == "export-manifest"
    assert result["max_batch_size"] == 128
    assert 150 <= result["per_lane_peak_mib"] <= 170
    variant = result["variants"][0]
    assert variant["talker_kv_mib"] == 56
    assert variant["persistent_mib"] > 60


def test_profile_suggestion_falls_back_without_manifest(tmp_path):
    helper = _load_helper()

    result = helper.suggest_profile(
        memory_mib=46068,
        exported_dir=tmp_path / "missing",
        variants=["custom-1.7b"],
        engine_dtype="bf16",
        max_input_len=128,
        max_seq_len=512,
    )

    assert result["source"] == "coarse-fallback"
    # 3758ad7 recalibrated the coarse tiers (>=30000 MiB -> 128); 46 GiB
    # now lands in the top tier instead of the old conservative 64.
    assert result["max_batch_size"] == 128


def test_profile_suggestion_handles_batch_cap_below_min_tier(tmp_path, monkeypatch):
    """A QWEN3_PROFILE_MAX_BATCH_CAP below the smallest tier must not IndexError."""
    helper = _load_helper()
    monkeypatch.setenv("QWEN3_PROFILE_MAX_BATCH_CAP", "8")

    result = helper.suggest_profile(
        memory_mib=46068,
        exported_dir=tmp_path / "missing",
        variants=["custom-1.7b"],
        engine_dtype="bf16",
        max_input_len=128,
        max_seq_len=512,
    )

    assert result["max_batch_size"] >= 1  # produced a value, did not crash
