from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_engine_container_mounts_runtime_tls_secret_and_maps_environment():
    compose = yaml.safe_load(_read("infra/docker/compose.yaml"))
    engine = compose["services"]["engine"]

    assert engine["environment"]["TLS_CERT_FILE"] == "${TLS_CERT_FILE:-}"
    assert engine["environment"]["TLS_KEY_FILE"] == "${TLS_KEY_FILE:-}"
    assert engine["environment"]["TLS_DIR"] == "/app/tls"
    assert any(volume.endswith(":/app/tls:ro") for volume in engine["volumes"])


def test_engine_entrypoint_auto_discovers_local_pair_and_exports_canonical_config():
    entrypoint = _read("scripts/compose/engine-entrypoint.sh")

    assert '${tls_dir}/cert.local.pem' in entrypoint
    assert '${tls_dir}/key.local.pem' in entrypoint
    assert 'export ENGINE_SERVER_TLS_CERT_FILE="$tls_cert_file"' in entrypoint
    assert 'export ENGINE_SERVER_TLS_KEY_FILE="$tls_key_file"' in entrypoint
    assert "TLS_CERT_FILE and TLS_KEY_FILE must be set together" in entrypoint
