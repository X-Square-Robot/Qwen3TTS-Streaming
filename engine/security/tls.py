"""Optional direct TLS termination for standalone public gateways."""

from __future__ import annotations

import ssl
from pathlib import Path


def build_server_ssl_context(
    cert_file: str,
    key_file: str,
) -> ssl.SSLContext | None:
    """Build a TLS 1.2+ server context, or return ``None`` for HTTP.

    Certificate configuration fails closed before GPU model loading. Private
    keys are expected to be mounted at runtime and are never image assets.
    """

    if not cert_file and not key_file:
        return None
    if not cert_file or not key_file:
        raise ValueError("TLS requires both a certificate and a private key")

    certificate = Path(cert_file)
    private_key = Path(key_file)
    if not certificate.is_file():
        raise ValueError(f"TLS certificate is not a readable file: {cert_file}")
    if not private_key.is_file():
        raise ValueError(f"TLS private key is not a readable file: {key_file}")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=str(certificate), keyfile=str(private_key))
    return context


__all__ = ("build_server_ssl_context",)
