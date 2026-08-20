from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import ssl
from typing import TypeAlias


TLSVerify: TypeAlias = bool | str | os.PathLike[str]


@dataclass(frozen=True)
class TLSConfig:
    """Normalized TLS verification policy shared by HTTP and WebSocket.

    The public ``tls_verify`` option follows the convention used by Requests:
    ``True`` trusts the system CA store, ``False`` disables certificate and
    hostname verification, and a filesystem path selects a CA bundle.  Keeping
    the normalized policy here prevents transport discovery, initial dials and
    reconnects from silently applying different TLS rules.
    """

    verify: bool = True
    ca_file: str | None = None

    @classmethod
    def from_value(cls, value: TLSVerify | "TLSConfig") -> "TLSConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(verify=value)
        try:
            raw_path = os.fspath(value)
        except TypeError as exc:
            raise TypeError(
                "tls_verify must be True, False, or a CA certificate path"
            ) from exc
        if isinstance(raw_path, bytes):
            raw_path = os.fsdecode(raw_path)
        if not raw_path.strip():
            raise ValueError("tls_verify CA certificate path must not be empty")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TLS CA certificate file not found: {path}")
        return cls(verify=True, ca_file=str(path))

    @property
    def is_default(self) -> bool:
        return self.verify and self.ca_file is None

    def requests_kwargs(self) -> dict[str, bool | str]:
        if self.is_default:
            return {}
        return {"verify": self.ca_file if self.ca_file is not None else self.verify}

    def websocket_sslopt(self) -> dict[str, object]:
        if not self.verify:
            return {
                "cert_reqs": ssl.CERT_NONE,
                "check_hostname": False,
            }
        options: dict[str, object] = {
            "cert_reqs": ssl.CERT_REQUIRED,
            "check_hostname": True,
        }
        if self.ca_file is not None:
            options["ca_certs"] = self.ca_file
        return options

    def forwarding_kwargs(self) -> dict[str, "TLSConfig"]:
        """Only forward non-default policy to preserve legacy test doubles."""

        if self.is_default:
            return {}
        return {"tls_verify": self}
