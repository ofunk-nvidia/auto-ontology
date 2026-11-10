# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oracle JSON catalog, optimistic revisions and native VECTOR cosine search."""

from __future__ import annotations

from array import array
import hashlib
import json
import math
import re
from typing import Any

from auto_ontology.oracle_mvp.settings import OracleSettings

IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
KINDS = {"catalog", "term", "proposal"}
MAX_DOCUMENT_BYTES = 128 * 1024

DDL = """CREATE TABLE ao_mvp_documents (
    document_id VARCHAR2(128) PRIMARY KEY,
    document_kind VARCHAR2(16) NOT NULL,
    revision NUMBER(10) NOT NULL,
    document_json CLOB NOT NULL CHECK (document_json IS JSON),
    document_sha256 VARCHAR2(64) NOT NULL,
    embedding VECTOR(384, FLOAT32),
    updated_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT ao_mvp_kind CHECK (document_kind IN ('catalog','term','proposal')),
    CONSTRAINT ao_mvp_revision CHECK (revision > 0)
)"""


class RevisionConflict(ValueError):
    """The requested base revision no longer matches Oracle."""


def checked_document(
    identity: str, kind: str, document: dict[str, Any]
) -> tuple[str, str]:
    """Validate metadata before any transaction or SQL execution."""
    if not IDENTIFIER.fullmatch(identity) or kind not in KINDS:
        raise ValueError("Invalid catalog document identity or kind")
    if not isinstance(document, dict):
        raise ValueError("A JSON object is required")
    raw = json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True)
    if len(raw.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ValueError("Catalog document exceeds the reviewed size limit")
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def checked_vector(values: list[float], dimensions: int = 384) -> array:
    """Reject malformed, non-finite and zero vectors before binding them."""
    if len(values) != dimensions or any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("Invalid embedding dimensions or values")
    if not any(values):
        raise ValueError("Cosine search requires a nonzero vector")
    vector = array("f", values)
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding cannot be represented as FLOAT32")
    return vector


def _row(row: tuple[Any, ...]) -> dict[str, Any]:
    """Decode and verify a complete JSON LOB while its connection is open."""
    identity, kind, revision, payload, checksum = row[:5]
    raw = payload.read() if hasattr(payload, "read") else payload
    if hashlib.sha256(raw.encode("utf-8")).hexdigest() != checksum:
        raise ValueError("Oracle catalog document checksum mismatch")
    return {
        "id": identity,
        "kind": kind,
        "revision": int(revision),
        "document": json.loads(raw),
        "sha256": checksum,
    }


def _json_lob(cursor: Any, metadata: Any) -> Any:
    """Preserve JSON CLOB bytes for checksum verification, avoiding auto-decoding.

    python-oracledb 2+ otherwise fetches IS JSON CLOBs as Python dictionaries,
    losing the original serialized representation that the checksum protects.
    """
    if metadata.is_json:
        return cursor.var(metadata.type_code, arraysize=cursor.arraysize)
    return None


class OracleStore:
    """Only Oracle is used; application startup never creates tables implicitly."""

    def __init__(self, settings: OracleSettings, pool: Any | None = None) -> None:
        self.settings = settings
        if pool is None:
            import oracledb

            pool = oracledb.create_pool(
                **settings.connect_arguments(), min=0, max=2, increment=1
            )
        self.pool = pool

    def close(self) -> None:
        """Close this process's Oracle pool."""
        self.pool.close()

    def health(self) -> dict[str, Any]:
        """Check the real selected database and the explicit catalog schema."""
        with self.pool.acquire() as connection, connection.cursor() as cursor:
            cursor.outputtypehandler = _json_lob
            connection.call_timeout = 10000
            cursor.execute("SELECT SYS_CONTEXT('USERENV','DB_NAME') FROM dual")
            name = cursor.fetchone()[0]
            cursor.execute("SELECT count(*) FROM ao_mvp_documents")
            count = cursor.fetchone()[0]
        return {"backend": "oracle", "database": name, "documents": count}

    def list(self, kind: str) -> list[dict[str, Any]]:
        """List bounded documents of one known kind."""
        if kind not in KINDS:
            raise ValueError("Unknown catalog document kind")
        with self.pool.acquire() as connection, connection.cursor() as cursor:
            cursor.outputtypehandler = _json_lob
            connection.call_timeout = 10000
            cursor.execute(
                "SELECT document_id,document_kind,revision,document_json,document_sha256 "
                "FROM ao_mvp_documents WHERE document_kind=:kind "
                "ORDER BY document_id FETCH FIRST 100 ROWS ONLY",
                kind=kind,
            )
            return [_row(row) for row in cursor]

    def get(self, identity: str) -> dict[str, Any] | None:
        """Read one durable catalog revision; no local replacement storage."""
        if not IDENTIFIER.fullmatch(identity):
            raise ValueError("Invalid document identity")
        with self.pool.acquire() as connection, connection.cursor() as cursor:
            cursor.outputtypehandler = _json_lob
            connection.call_timeout = 10000
            cursor.execute(
                "SELECT document_id,document_kind,revision,document_json,document_sha256 "
                "FROM ao_mvp_documents WHERE document_id=:identity",
                identity=identity,
            )
            row = cursor.fetchone()
            return _row(row) if row else None

    def save(
        self,
        identity: str,
        kind: str,
        expected_revision: int,
        document: dict[str, Any],
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        """Atomically create or compare-and-swap a catalog revision in Oracle."""
        raw, checksum = checked_document(identity, kind, document)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("An explicit nonnegative base revision is required")
        vector = checked_vector(embedding) if embedding is not None else None
        with self.pool.acquire() as connection, connection.cursor() as cursor:
            cursor.outputtypehandler = _json_lob
            connection.call_timeout = 15000
            try:
                if len(raw.encode("utf-8")) > 4000:
                    import oracledb

                    cursor.setinputsizes(payload=oracledb.DB_TYPE_CLOB)
                if expected_revision == 0:
                    try:
                        cursor.execute(
                            "INSERT INTO ao_mvp_documents "
                            "(document_id,document_kind,revision,document_json,document_sha256,embedding) "
                            "VALUES (:identity,:kind,1,:payload,:checksum,:embedding)",
                            identity=identity,
                            kind=kind,
                            payload=raw,
                            checksum=checksum,
                            embedding=vector,
                        )
                    except Exception as error:
                        if getattr(error.args[0], "code", None) == 1:
                            raise RevisionConflict("Document already exists") from None
                        raise
                else:
                    cursor.execute(
                        "UPDATE ao_mvp_documents SET document_json=:payload, "
                        "document_sha256=:checksum, embedding=:embedding, "
                        "revision=revision+1,updated_at=SYSTIMESTAMP "
                        "WHERE document_id=:identity AND document_kind=:kind "
                        "AND revision=:expected",
                        payload=raw,
                        checksum=checksum,
                        embedding=vector,
                        identity=identity,
                        kind=kind,
                        expected=expected_revision,
                    )
                    if cursor.rowcount != 1:
                        raise RevisionConflict("Catalog revision changed; reload first")
                cursor.execute(
                    "SELECT document_id,document_kind,revision,document_json,document_sha256 "
                    "FROM ao_mvp_documents WHERE document_id=:identity",
                    identity=identity,
                )
                written = cursor.fetchone()
                if written is None:
                    raise RuntimeError("Saved Oracle document could not be read back")
                result = _row(written)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return result

    def search(
        self, embedding: list[float], kind: str = "term", limit: int = 5
    ) -> list[dict[str, Any]]:
        """Use exact native Oracle cosine distance; no pgvector or paid index."""
        vector = checked_vector(embedding)
        if kind not in KINDS or type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Invalid search scope or result limit")
        with self.pool.acquire() as connection, connection.cursor() as cursor:
            cursor.outputtypehandler = _json_lob
            connection.call_timeout = 10000
            cursor.execute(
                "SELECT document_id,document_kind,revision,document_json,document_sha256, "
                "VECTOR_DISTANCE(embedding,:vector,COSINE) distance "
                "FROM ao_mvp_documents WHERE document_kind=:kind AND embedding IS NOT NULL "
                "ORDER BY distance,document_id FETCH FIRST :result_limit ROWS ONLY",
                vector=vector,
                kind=kind,
                result_limit=limit,
            )
            return [dict(_row(row), distance=float(row[5])) for row in cursor]
