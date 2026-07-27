from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def apply_bearer_key(
    headers: Mapping[str, str] | None,
    metadata: Any,
    key: str | None,
) -> tuple[dict[str, str] | None, tuple[tuple[str, Any], ...] | None]:
    """Return copied transport credentials with an optional bearer key.

    HTTP and WebSocket transports use ``Authorization`` while gRPC requires
    lowercase metadata keys.  ``key=None`` means that the SDK does not inject
    credentials; caller-supplied headers/metadata are still preserved.
    """

    effective_headers = None if headers is None else dict(headers)
    effective_metadata = None if metadata is None else tuple(_metadata_items(metadata))
    if key is None:
        return effective_headers, effective_metadata

    bearer = f"Bearer {key}"
    effective_headers = {
        name: value
        for name, value in (effective_headers or {}).items()
        if name.lower() != "authorization"
    }
    effective_headers["Authorization"] = bearer

    effective_metadata = tuple(
        (name, value)
        for name, value in (effective_metadata or ())
        if str(name).lower() != "authorization"
    ) + (("authorization", bearer),)
    return effective_headers, effective_metadata


def normalize_grpc_metadata(
    metadata: Any = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[tuple[str, Any], ...]:
    """Normalize SDK metadata/headers for a native gRPC call.

    Explicit gRPC metadata wins over the cross-transport ``headers`` fallback.
    Duplicate entries inside metadata are retained.  All names are lowercased
    because grpcio rejects uppercase metadata keys before issuing the RPC.
    """

    metadata_items = [
        (str(name).lower(), value) for name, value in _metadata_items(metadata)
    ]
    metadata_names = {name for name, _value in metadata_items}

    normalized_headers: dict[str, Any] = {}
    for name, value in (headers or {}).items():
        normalized_headers[str(name).lower()] = value

    header_items = [
        (name, value)
        for name, value in normalized_headers.items()
        if name not in metadata_names
    ]
    return tuple(header_items + metadata_items)


def grpc_metadata_as_headers(
    metadata: Any = None,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return metadata in the dictionary form expected by Triton's client."""

    return dict(normalize_grpc_metadata(metadata, headers))


def _metadata_items(metadata: Any) -> list[tuple[str, Any]]:
    if metadata is None:
        return []
    if isinstance(metadata, Mapping):
        return list(metadata.items())
    return list(metadata)
