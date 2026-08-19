"""Tests for optional direct HTTPS/WSS termination."""

import ssl
from unittest.mock import patch

import pytest

from engine.security import build_server_ssl_context


def test_empty_tls_configuration_keeps_plain_http():
    assert build_server_ssl_context("", "") is None


@pytest.mark.parametrize("cert,key", [("cert.pem", ""), ("", "key.pem")])
def test_partial_tls_configuration_fails_closed(cert, key):
    with pytest.raises(ValueError, match="requires both"):
        build_server_ssl_context(cert, key)


def test_missing_tls_file_fails_before_server_start(tmp_path):
    with pytest.raises(ValueError, match="certificate"):
        build_server_ssl_context(
            str(tmp_path / "missing.pem"),
            str(tmp_path / "missing.key"),
        )


def test_tls_context_loads_mounted_certificate_and_requires_tls12(tmp_path):
    certificate = tmp_path / "fullchain.pem"
    private_key = tmp_path / "privkey.pem"
    certificate.write_text("certificate", encoding="utf-8")
    private_key.write_text("private key", encoding="utf-8")

    with patch.object(ssl.SSLContext, "load_cert_chain") as load:
        context = build_server_ssl_context(str(certificate), str(private_key))

    assert isinstance(context, ssl.SSLContext)
    assert context.minimum_version == ssl.TLSVersion.TLSv1_2
    load.assert_called_once_with(
        certfile=str(certificate),
        keyfile=str(private_key),
    )
