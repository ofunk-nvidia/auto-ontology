# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model fit, operator rights, provenance and paid-call boundaries."""

from copy import deepcopy
import json
from unittest.mock import MagicMock

import pytest

from auto_ontology.oracle_mvp.model_selection import (
    LOCAL_ID,
    ModelSelection,
    ModelUnavailable,
    family_for,
    test_id as receipt_id,
)
from auto_ontology.oracle_mvp.store import RevisionConflict

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
ACTOR = "00000000-0000-4000-8000-000000000001"


class Store:
    def __init__(self):
        self.rows = {}

    def get(self, identity):
        return deepcopy(self.rows.get(identity))

    def save(self, identity, kind, revision, value):
        previous = self.rows.get(identity)
        if (previous["revision"] if previous else 0) != revision:
            raise RevisionConflict("Changed")
        self.rows[identity] = dict(
            id=identity, revision=revision + 1, document=deepcopy(value)
        )
        return self.get(identity)


class Hub:
    def __init__(self):
        self.calls = []
        self.break_source = False
        self.rows = [
            dict(id=MODEL),
            dict(id="qwen/qwen3-32b"),
            dict(id="nvidia/embedding-only"),
            dict(id="vendor/reasoning-but-undocumented"),
            dict(id="https://evil.invalid/key"),
        ]

    def request(self, path, body=None):
        self.calls.append((path, body))
        if path == "/models":
            return dict(data=self.rows)
        value = dict(
            sourceId="evil" if self.break_source else "synthetic:source:1",
            subject="synthetic:Sensor-A",
            predicate="synthetic:measures",
            object="synthetic:Temperature",
            description="Sensor-A misst die Temperatur.",
            entailedDevice=True,
            reversedMeasures=False,
            newOwnerSupported=False,
        )
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema")
        if schema and set(schema["properties"]) == {"description"}:
            value = {"description": "Sensor-A misst die Temperatur."}
        raw = json.dumps(value)
        if "tools" in body:
            message = dict(
                tool_calls=[dict(function=dict(name="ontology_result", arguments=raw))]
            )
        else:
            message = dict(content=raw)
        return dict(
            model="deployment-version-1",
            choices=[
                dict(
                    message=message,
                    finish_reason="tool_calls" if "tools" in body else "stop",
                )
            ],
            usage=dict(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        )


def manager():
    profile = dict(
        id="synthetic-profile-v1",
        keyFile="synthetic-unused",
        operatorActors=[ACTOR],
        useApproval="synthetic-approved-use",
        costMode="verified-free",
        costEvidence="synthetic-no-charge-evidence",
        freeModelIds=[MODEL],
    )
    return ModelSelection(Store(), MagicMock(), profile, Hub())


def test_atomic_profile_rotation_invalidates_catalog_and_fit_without_restart(tmp_path):
    app = manager()
    app.test(MODEL, ACTOR)
    assert app._fresh_test(MODEL)
    path = tmp_path / "profile.json"
    updated = dict(
        app.profile, id="rotated", costMode="blocked", costEvidence="", freeModelIds=[]
    )
    path.write_text(json.dumps(updated))
    path.chmod(0o600)
    app.profile_path = path
    app.refresh_profile()
    assert app.profile["id"] == "rotated"
    assert app.cached_models is None
    assert app._fresh_test(MODEL) is None
    assert not app.permitted(MODEL)
    with pytest.raises(ModelUnavailable):
        app._allowed(MODEL)


def test_live_profile_cannot_change_the_fixed_provider_endpoint(tmp_path):
    app = manager()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(dict(app.profile, baseUrl="https://example.invalid/v1")))
    path.chmod(0o600)
    app.profile_path = path
    with pytest.raises(ModelUnavailable):
        app.catalog(ACTOR)
    assert app.profile.get("baseUrl") != "https://example.invalid/v1"


def test_catalog_excludes_unknown_and_non_text_models_without_claiming_a_live_test():
    app = manager()
    result = app.catalog(ACTOR)
    assert {row["id"] for row in result["cards"]} == {MODEL, "qwen/qwen3-32b"}
    assert all(
        row["status"] == "documented-candidate" and not row["selectable"]
        for row in result["cards"]
    )
    assert all(not row["fullUpstreamValidated"] for row in result["cards"])
    assert all(path == "/models" for path, _ in app.connection.calls)
    assert "keyFile" not in json.dumps(result)


def test_missing_key_has_no_demo_candidates_and_never_contacts_an_endpoint():
    app = ModelSelection(Store(), MagicMock(), connection=Hub())
    result = app.catalog(ACTOR)
    assert result["state"] == "not-configured" and not result["cards"]
    assert not app.connection.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("useApproval", ""),
        ("costEvidence", ""),
        ("costMode", "unknown"),
        ("costMode", "paid"),
    ],
)
def test_approval_and_cost_fail_closed_before_model_calls(field, value):
    app = manager()
    app.profile[field] = value
    with pytest.raises(ModelUnavailable):
        app.test(MODEL, ACTOR)
    assert not app.connection.calls


def test_operator_required_for_tests_and_model_changes():
    app = manager()
    for action in (
        lambda: app.test(MODEL, "other"),
        lambda: app.select(LOCAL_ID, 0, "other"),
    ):
        with pytest.raises(PermissionError):
            action()
    assert not app.connection.calls


def test_untested_model_cannot_be_selected_but_passed_test_can():
    app = manager()
    with pytest.raises(ModelUnavailable):
        app.select(MODEL, 0, ACTOR)
    result = app.test(MODEL, ACTOR)
    assert result["passed"] and len(result["checks"]) == 2
    assert all(row["passed"] for row in result["checks"])
    selected = app.select(MODEL, 0, ACTOR)
    assert selected["selected"] == dict(
        provider="nvidia-inference-hub", model=MODEL, selectionRevision=1
    )
    assert next(card for card in selected["cards"] if card["id"] == MODEL)["selectable"]


def test_prompt_injection_or_changed_source_fails_the_fit_test():
    app = manager()
    app.connection.break_source = True
    assert not app.test(MODEL, ACTOR)["passed"]
    with pytest.raises(ModelUnavailable):
        app.select(MODEL, 0, ACTOR)


def test_selection_revision_and_expiry_are_enforced_without_hidden_fallback():
    app = manager()
    app.test(MODEL, ACTOR)
    app.select(MODEL, 0, ACTOR)
    with pytest.raises(RevisionConflict):
        app.select(LOCAL_ID, 0, ACTOR)
    app.store.rows[receipt_id(MODEL)]["document"]["testedAtEpoch"] = 0
    before = len(app.connection.calls)
    with pytest.raises(ModelUnavailable):
        app.propose("Sensor-A", "Synthetic context")
    assert len(app.connection.calls) == before
    app.local.propose.assert_not_called()


def test_selected_provider_model_usage_and_actual_returned_model_are_preserved():
    app = manager()
    app.test(MODEL, ACTOR)
    app.select(MODEL, 0, ACTOR)
    proposal = app.propose("Sensor-A", "Synthetic context")
    assert proposal["model"] == MODEL and proposal["provider"] == "nvidia-inference-hub"
    assert proposal["returnedModel"] == "deployment-version-1"
    assert proposal["usage"]["total_tokens"] == 70
    assert proposal["selectionRevision"] == 1 and not proposal["approved"]
    app.local.propose.assert_not_called()


def test_unknown_model_names_cannot_be_qualified_by_reasoning_substrings():
    for identity in (
        "vendor/qwen3-32b",
        "nvidia/nemotron-3.5-lightning-30b-a3b-malicious",
        "https://host/qwen/qwen3-32b",
        "vendor/reasoning-pro",
    ):
        assert family_for(identity) is None


def test_a_general_free_claim_does_not_authorize_another_model():
    app = manager()
    with pytest.raises(ModelUnavailable):
        app.test("qwen/qwen3-32b", ACTOR)
    assert not app.connection.calls


def test_generic_catalog_cannot_forge_model_qualification_or_selection():
    from fastapi.testclient import TestClient
    from auto_ontology.oracle_mvp.api import create_app

    app = manager()
    client = TestClient(
        create_app(
            store=app.store,
            inference=app,
            gateway_key="synthetic-gateway-key-at-least-32-characters",
        ),
        headers={
            "Authorization": "Bearer synthetic-gateway-key-at-least-32-characters"
        },
    )
    for identity in ("rdf-model-selection", receipt_id(MODEL)):
        response = client.post(
            "/api/catalog",
            json=dict(
                id=identity,
                kind="catalog",
                expected_revision=0,
                label="Synthetic forged fit",
                description="passed",
                source={},
            ),
        )
        assert response.status_code == 422
    assert not app.store.rows
