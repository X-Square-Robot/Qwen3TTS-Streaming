import os
import subprocess
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


def test_local_demo_certificate_generator_includes_requested_dns_san(tmp_path):
    script = REPO_ROOT / "tools/generate_demo_local_cert.sh"
    env = os.environ.copy()
    env.update(
        {
            "DEMO_TLS_DIR": str(tmp_path),
            "SAN_EXTRA_DNS": "demo.example.test",
        }
    )

    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    certificate = tmp_path / "cert.local.pem"
    private_key = tmp_path / "key.local.pem"
    details = subprocess.run(
        [
            "openssl",
            "x509",
            "-in",
            str(certificate),
            "-noout",
            "-ext",
            "subjectAltName",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "DNS:localhost" in details
    assert "DNS:demo.example.test" in details
    assert private_key.stat().st_mode & 0o777 == 0o600
