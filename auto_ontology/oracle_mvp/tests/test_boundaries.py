# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure cases that must remain closed before live Oracle acceptance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from auto_ontology.oracle_mvp.api import create_app
from auto_ontology.oracle_mvp.settings import (
    OracleSettings,
    private_text,
    require_local_inference,
)
from auto_ontology.oracle_mvp.store import (
    OracleStore,
    RevisionConflict,
    checked_document,
    checked_vector,
)

GATEWAY_KEY = "synthetic-test-gateway-key-32-characters"


def client_for(store: MagicMock, inference: MagicMock) -> TestClient:
    return TestClient(
        create_app(store=store, inference=inference, gateway_key=GATEWAY_KEY),
        headers={"Authorization": "Bearer " + GATEWAY_KEY},
    )


def test_missing_or_hosted_endpoint_is_never_a_fallback() -> None:
    names = ("DEFAULT_MODELS", "REASONING", "NON_REASONING", "EMBED", "RERANK")
    values = {name + "_ENDPOINT": "http://127.0.0.1:11435/v1" for name in names}
    with patch.dict(os.environ, values, clear=True):
        require_local_inference()
        for name in names:
            key = name + "_ENDPOINT"
            for invalid in (
                "",
                "https://example.com/v1",
                "http://user:pw@127.0.0.1/v1",
            ):
                with patch.dict(os.environ, {key: invalid}), pytest.raises(ValueError):
                    require_local_inference()


@pytest.mark.parametrize(
    "value",
    (
        [1.0] * 383,
        [0.0] * 384,
        [float("nan")] * 384,
        [float("inf")] * 384,
        [True] * 384,
    ),
)
def test_invalid_vectors_cannot_reach_oracle(value: list[float]) -> None:
    with pytest.raises(ValueError):
        checked_vector(value)


def test_valid_vector_and_unicode_document_are_preserved() -> None:
    values = [1.0] + [0.0] * 383
    assert checked_vector(values).tolist() == values
    raw, digest = checked_document("synthetic-1", "term", {"label": "Künstlicher Kurs"})
    assert json.loads(raw)["label"] == "Künstlicher Kurs"
    assert hashlib.sha256(raw.encode()).hexdigest() == digest


def test_invalid_identity_and_oversized_utf8_are_rejected() -> None:
    for identity, body in (("x'; DROP TABLE x", {}), ("valid", {"text": "ä" * 70000})):
        with pytest.raises(ValueError):
            checked_document(identity, "term", body)


def test_private_secret_file_is_read_but_symlink_is_rejected(tmp_path: Path) -> None:
    secret = tmp_path / "password"
    secret.write_text("synthetic-test-secret", encoding="utf-8")
    secret.chmod(0o600)
    assert private_text(secret) == "synthetic-test-secret"
    if os.name == "posix":
        secret.chmod(0o644)
        with pytest.raises(ValueError):
            private_text(secret)
        secret.chmod(0o600)
        link = tmp_path / "link"
        link.symlink_to(secret)
        with pytest.raises(ValueError):
            private_text(link)


def mocked_store() -> tuple[OracleStore, MagicMock, MagicMock]:
    settings = OracleSettings(
        "SYNTHETIC", "synthetic_low", Path("pw"), Path("wallet"), Path("wpw"), "TEST"
    )
    pool = MagicMock()
    connection = pool.acquire.return_value.__enter__.return_value
    cursor = connection.cursor.return_value.__enter__.return_value
    return OracleStore(settings, pool=pool), connection, cursor


def test_stale_revision_rolls_back_without_claiming_success() -> None:
    store, connection, cursor = mocked_store()
    cursor.rowcount = 0
    with pytest.raises(RevisionConflict):
        store.save("synthetic", "term", 3, {"label": "Test"})
    connection.rollback.assert_called_once()
    connection.commit.assert_not_called()


def test_committed_snapshot_is_read_before_releasing_write_lock() -> None:
    store, connection, cursor = mocked_store()
    document = {"label": "Synthetic"}
    raw, digest = checked_document("synthetic", "term", document)
    cursor.fetchone.return_value = ("synthetic", "term", 1, raw, digest)
    result = store.save("synthetic", "term", 0, document)
    assert result["revision"] == 1
    assert result["document"] == document
    connection.commit.assert_called_once()
    connection.rollback.assert_not_called()
    # Neither the identity nor document text is interpolated into SQL.
    (sql,) = cursor.execute.call_args_list[0].args
    assert "Synthetic" not in sql


def test_corrupt_readback_prevents_commit() -> None:
    store, connection, cursor = mocked_store()
    cursor.fetchone.return_value = ("synthetic", "term", 1, "{}", "bad-checksum")
    with pytest.raises(ValueError):
        store.save("synthetic", "term", 0, {})
    connection.rollback.assert_called_once()
    connection.commit.assert_not_called()


def test_api_rejects_caller_sql_endpoints_and_paths() -> None:
    store = MagicMock()
    inference = MagicMock()
    with client_for(store, inference) as client:
        valid = dict(
            id="synthetic",
            kind="term",
            expected_revision=0,
            label="Kurs",
            description="Übung",
        )
        for field in ("sql", "endpoint", "passwordFile", "walletDirectory"):
            assert (
                client.post(
                    "/api/catalog", json={**valid, field: "forbidden"}
                ).status_code
                == 422
            )
        store.save.assert_not_called()
        inference.embed.assert_not_called()


def test_readiness_failure_does_not_disclose_credentials_or_fall_back() -> None:
    store = MagicMock()
    store.health.side_effect = RuntimeError("password=synthetic-private-test")
    with client_for(store, MagicMock()) as client:
        result = client.get("/api/health")
        assert result.status_code == 503
        assert "synthetic-private-test" not in result.text
        assert client.get("/api/health/live").json()["postgresRequired"] is False


def test_proposal_can_never_self_approve_or_publish_rdf() -> None:
    store = MagicMock()
    inference = MagicMock()
    inference.propose.return_value = {
        "label": "Kurs",
        "description": "Übung.",
        "approved": True,
    }
    inference.embed.return_value = [1.0] + [0.0] * 383
    store.save.return_value = {"id": "synthetic", "revision": 1}
    with client_for(store, inference) as client:
        result = client.post(
            "/api/proposals", json={"label": "Kurs", "context": "Künstlicher Kurs"}
        )
        assert result.status_code == 200
        args = store.save.call_args.args
        assert args[1] == "proposal"
        assert args[3]["approved"] is False
        assert args[3]["status"] == "proposed"
        capabilities = client.get("/api/capabilities").json()
        assert capabilities["automaticRdfPublication"] is False
        assert capabilities["fullUpstreamDalCompatibility"] is False


def test_untrusted_caller_cannot_read_or_change_catalog() -> None:
    store = MagicMock()
    with TestClient(
        create_app(store=store, inference=MagicMock(), gateway_key=GATEWAY_KEY)
    ) as client:
        assert client.get("/api/catalog").status_code == 401
        assert (
            client.post(
                "/api/proposals", json={"label": "X", "context": "Y"}
            ).status_code
            == 401
        )
        assert client.get("/api/health/live").status_code == 200
        store.list.assert_not_called()
        store.save.assert_not_called()
