# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Curation decisions are durable, source-bound, and never source publication."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rdflib import RDFS

from auto_ontology.oracle_mvp.store import RevisionConflict, checked_document
from auto_ontology.oracle_rdf.source import snapshot_from_rows
from auto_ontology.oracle_rdf.tests.test_source import settings, source_rows
from auto_ontology.oracle_rdf.workbench import RdfWorkbench


class MemoryStore:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(database_name="APPLICATION")
        self.rows = {}

    def health(self):
        return dict(database="APPLICATION")

    def get(self, identity):
        return deepcopy(self.rows.get(identity))

    def list(self, kind):
        return [deepcopy(row) for row in self.rows.values() if row["kind"] == kind]

    def save(self, identity, kind, revision, value):
        current = self.rows.get(identity)
        if (current["revision"] if current else 0) != revision:
            raise RevisionConflict("Changed")
        _, digest = checked_document(identity, kind, value)
        row = dict(
            id=identity,
            kind=kind,
            revision=revision + 1,
            document=deepcopy(value),
            sha256=digest,
        )
        self.rows[identity] = row
        return deepcopy(row)


def workbench():
    config = settings()
    rows = [
        *source_rows(),
        (
            "urn:example:data",
            "urn:example:item",
            "URI",
            str(RDFS.comment),
            "Synthetic source description",
            "LIT",
            None,
            "de",
        ),
    ]
    source = MagicMock(settings=config)
    source.snapshot.return_value = snapshot_from_rows(config, rows)
    inference = MagicMock()
    inference.propose.return_value = dict(description="Geprüfter künstlicher Entwurf")
    return RdfWorkbench(source, MemoryStore(), inference)


def test_report_is_reopened_from_application_store_without_repeat_validation():
    app = workbench()
    first = app.analyze()
    second = app.analyze()
    assert not first["reused"] and second["reused"]
    assert len(app.store.rows) == 1 and first["document"]["sourceWrites"] is False
    assert app.summary()["sourceDatabase"] != app.summary()["applicationDatabase"]
    app.inference.propose.assert_not_called()


def test_proposal_keeps_exact_source_fields_and_reuses_unchanged_source():
    app = workbench()
    sha = app.snapshot().sha256
    first = app.propose("urn:example:item", sha, "synthetic-actor")
    second = app.propose("urn:example:item", sha, "synthetic-actor")
    assert second["id"] == first["id"] and second["reused"]
    app.inference.propose.assert_called_once()
    before, after = first["document"]["before"], first["document"]["after"]
    assert before["graph"] == after["graph"] == "urn:example:data"
    assert before["object"]["language"] == after["object"]["language"] == "de"
    assert before["object"]["value"] == "Synthetic source description"
    assert after["object"]["value"] == "Geprüfter künstlicher Entwurf"


def test_review_persists_reason_actor_and_conflicts_with_repeat_decision():
    app = workbench()
    sha = app.snapshot().sha256
    proposal = app.propose("urn:example:item", sha, "synthetic-actor")
    accepted = app.decide(
        proposal["id"], 1, sha, "accepted-for-review", "Synthetic review", "reviewer"
    )
    assert accepted["revision"] == 2 and not accepted["document"]["sourceApplied"]
    assert app.proposals()[0]["document"]["decisions"][0]["actor"] == "reviewer"
    with pytest.raises(RevisionConflict):
        app.decide(proposal["id"], 1, sha, "rejected", "Repeated decision", "reviewer")


def test_changed_source_rejects_decision_and_proposal_before_inference():
    app = workbench()
    sha = app.snapshot().sha256
    proposal = app.propose("urn:example:item", sha, "actor")
    rows = [
        *source_rows(),
        (
            "urn:example:data",
            "urn:example:new",
            "URI",
            str(RDFS.label),
            "New",
            "LIT",
            None,
            "de",
        ),
    ]
    app.source.snapshot.return_value = snapshot_from_rows(app.source.settings, rows)
    with pytest.raises(RevisionConflict):
        app.decide(
            proposal["id"], 1, sha, "accepted-for-review", "Old source", "reviewer"
        )
    with pytest.raises(RevisionConflict):
        app.propose("urn:example:item", sha, "actor")
    app.inference.propose.assert_called_once()


def test_source_failure_remains_explainable_and_creates_no_success_record():
    app = workbench()
    app.source.snapshot.side_effect = RuntimeError("private driver details")
    with pytest.raises(RuntimeError):
        app.analyze()
    assert app.job_state["status"] == "failed" and not app.store.rows
    assert "private driver" not in str(app.job_state)
