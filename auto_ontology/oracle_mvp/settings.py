# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator-owned Oracle connection settings; no credentials in model documents."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from urllib.parse import urlsplit


def private_text(path: Path) -> str:
    """Read a bounded regular secret file owned by the service user on POSIX."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("A private regular credential file is required")
    if os.name == "posix" and (
        before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) & 0o077
    ):
        raise ValueError("Credential ownership or permissions are unsafe")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("Credential file changed while opening")
        raw = handle.read(8193)
    if not raw or len(raw) > 8192:
        raise ValueError("Credential size is invalid")
    return raw.decode("utf-8").strip()


@dataclass(frozen=True)
class OracleSettings:
    """Connection settings supplied by an operator, never by an API caller."""

    user: str
    dsn: str
    password_file: Path
    wallet_directory: Path
    wallet_password_file: Path
    database_name: str
    dimensions: int = 384

    @classmethod
    def from_file(cls, path: Path) -> OracleSettings:
        """Load the selected Oracle configuration without a Postgres fallback."""
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("backend") != "oracle" or value.get("alwaysFreeOnly") is not True:
            raise ValueError(
                "This pilot requires an explicit Always-Free Oracle target"
            )
        if value.get("dimensions", 384) != 384:
            raise ValueError("This MVP requires 384-dimensional embeddings")
        return cls(
            user=value["user"],
            dsn=value["dsn"],
            password_file=Path(value["passwordFile"]),
            wallet_directory=Path(value["walletDirectory"]),
            wallet_password_file=Path(value["walletPasswordFile"]),
            database_name=value["databaseName"],
        )

    def connect_arguments(self) -> dict[str, object]:
        """Return driver arguments without changing any cloud configuration."""
        return {
            "user": self.user,
            "password": private_text(self.password_file),
            "dsn": self.dsn,
            "config_dir": str(self.wallet_directory),
            "wallet_location": str(self.wallet_directory),
            "wallet_password": private_text(self.wallet_password_file),
            "tcp_connect_timeout": 8,
            "retry_count": 0,
        }


def require_local_inference() -> None:
    """Fail closed rather than inherit the library's hosted inference defaults."""
    for prefix in ("DEFAULT_MODELS", "REASONING", "NON_REASONING", "EMBED", "RERANK"):
        endpoint = os.environ.get(prefix + "_ENDPOINT", "")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("127.0.0.1", "::1", "localhost")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Every inference endpoint must explicitly use loopback")
