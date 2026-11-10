# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in acceptance against a separately provisioned Oracle test database.

No cloud resource or schema is created by this suite. Test records are synthetic
and remain in the dedicated test schema for an explicit cleanup decision.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
import uuid

import pytest
from fastapi.testclient import TestClient

from auto_ontology.oracle_mvp.api import create_app
from auto_ontology.oracle_mvp.settings import OracleSettings
from auto_ontology.oracle_mvp.store import OracleStore, RevisionConflict

_CONFIG = os.environ.get("AUTO_ONTOLOGY_ORACLE_TEST_CONFIG")
pytestmark = [
    pytest.mark.oracle_live,
    pytest.mark.skipif(
        not _CONFIG or os.environ.get("AUTO_ONTOLOGY_ORACLE_ALLOW_WRITE_TESTS") != "1",
        reason="Dedicated Oracle test config and explicit synthetic-write opt-in required",
    ),
]


@pytest.fixture
def live_store() -> Iterator[OracleStore]:
    settings = OracleSettings.from_file(Path(_CONFIG))
    assert settings.user.upper() != "ADMIN"
    store = OracleStore(settings)
    try:
        yield store
    finally:
        store.close()


def test_real_unicode_lob_cas_and_independent_pool_readback(
    live_store: OracleStore,
) -> None:
    identity = "synthetic-" + uuid.uuid4().hex
    document = {
        "label": "Künstlicher Kurs",
        "description": "Änderung prüfen. " * 600,
        "source": {"revision": 17, "ratio": 1.25},
    }
    first = live_store.save(identity, "term", 0, document)
    assert first["revision"] == 1
    independent = OracleStore(live_store.settings)
    try:
        assert independent.get(identity)["document"] == document
        second = independent.save(identity, "term", 1, {"label": "Geprüfter Test"})
        assert second["revision"] == 2
        with pytest.raises(RevisionConflict):
            live_store.save(identity, "term", 1, {"label": "Veralteter Entwurf"})
        with pytest.raises(RevisionConflict):
            live_store.save(identity, "term", 0, {"label": "Doppelte Anlage"})
        assert live_store.get(identity)["revision"] == 2
        assert live_store.get(identity)["document"]["label"] == "Geprüfter Test"
    finally:
        independent.close()


def test_real_native_oracle_cosine_distance(live_store: OracleStore) -> None:
    prefix = "synthetic-vector-" + uuid.uuid4().hex
    first = [1.0] + [0.0] * 383
    second = [0.0, 1.0] + [0.0] * 382
    live_store.save(prefix + "-a", "catalog", 0, {"label": "Synthetic A"}, first)
    live_store.save(prefix + "-b", "catalog", 0, {"label": "Synthetic B"}, second)
    hits = {
        row["id"]: row for row in live_store.search(first, kind="catalog", limit=20)
    }
    assert hits[prefix + "-a"]["distance"] == pytest.approx(0.0, abs=1e-6)
    assert hits[prefix + "-b"]["distance"] == pytest.approx(1.0, abs=1e-6)


def test_real_private_api_revisions_and_readback_after_restart(
    live_store: OracleStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Catalog writes need no inference call; every potential model endpoint
    # remains explicit loopback even in this database-only acceptance test.
    for name in ("DEFAULT_MODELS", "REASONING", "NON_REASONING", "EMBED", "RERANK"):
        monkeypatch.setenv(name + "_ENDPOINT", "http://127.0.0.1:11435/v1")
    key = "synthetic-private-api-acceptance-key-32-characters"
    identity = "synthetic-api-" + uuid.uuid4().hex
    body = dict(
        id=identity,
        kind="catalog",
        expected_revision=0,
        label="Künstlicher Katalog",
        description="Änderung prüfen. " * 200,
    )
    app = create_app(store=OracleStore(live_store.settings), gateway_key=key)
    headers = {"Authorization": "Bearer " + key}
    with TestClient(app) as client:
        assert client.post("/api/catalog", json=body).status_code == 401
        assert client.get("/api/health", headers=headers).json()["backend"] == "oracle"
        first = client.post("/api/catalog", json=body, headers=headers)
        assert first.status_code == 200
        assert first.json()["revision"] == 1
        assert first.json()["document"]["description"] == body["description"]
        assert (
            client.post("/api/catalog", json=body, headers=headers).status_code == 409
        )
        updated = {**body, "expected_revision": 1, "description": "Geprüfter Test"}
        assert (
            client.post("/api/catalog", json=updated, headers=headers).json()[
                "revision"
            ]
            == 2
        )
        assert (
            client.post(
                "/api/catalog", json={**updated, "sql": "forbidden"}, headers=headers
            ).status_code
            == 422
        )
    # A new app/pool reads only the committed Oracle state after shutdown.
    restarted = create_app(store=OracleStore(live_store.settings), gateway_key=key)
    with TestClient(restarted) as client:
        restored = client.get("/api/catalog/" + identity, headers=headers)
        assert restored.status_code == 200
        assert restored.json()["revision"] == 2
        assert restored.json()["document"]["description"] == "Geprüfter Test"
