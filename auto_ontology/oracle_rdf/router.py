# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private-gateway endpoints for the source-bound Microsoft 365 workbench."""

from typing import Any, Literal
import uuid

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from auto_ontology.oracle_rdf.workbench import RdfWorkbench
from auto_ontology.oracle_mvp.model_selection import ModelSelection


class ProposalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resourceId: str = Field(min_length=1, max_length=8192)
    sourceSha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class DecisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)
    sourceSha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["accepted-for-review", "rejected"]
    reason: str = Field(min_length=1, max_length=2000)


class ModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_./:-]{0,199}$")


class ModelSelectionInput(ModelInput):
    revision: int = Field(ge=0)


def router_for(
    workbench: RdfWorkbench, models: ModelSelection | None = None
) -> APIRouter:
    router = APIRouter(prefix="/api/rdf")

    def actor(value: str | None) -> str:
        try:
            return str(uuid.UUID(value or ""))
        except ValueError:
            raise HTTPException(403, "Verified Microsoft 365 actor required") from None

    if models is not None:

        @router.get("/models")
        def model_catalog(
            x_m365_actor: str | None = Header(default=None),
        ) -> dict[str, Any]:
            """List documented role candidates actually available to this NVIDIA key."""
            return models.catalog(actor(x_m365_actor))

        @router.post("/models/test")
        def test_model(
            body: ModelInput, x_m365_actor: str | None = Header(default=None)
        ) -> dict[str, Any]:
            """Run bounded synthetic capability checks only under explicit operator policy."""
            return models.test(body.model, actor(x_m365_actor))

        @router.post("/models/select")
        def select_model(
            body: ModelSelectionInput, x_m365_actor: str | None = Header(default=None)
        ) -> dict[str, Any]:
            """Select a tested model with optimistic revision and an operator identity."""
            return models.select(body.model, body.revision, actor(x_m365_actor))

    @router.get("/source")
    def source() -> dict[str, Any]:
        """Show the actual distinct source/application database roles."""
        return workbench.summary()

    @router.get("/resources")
    def resources(
        query: str = Query(default="", max_length=200),
    ) -> list[dict[str, str]]:
        """List bounded real RDF resources; never substitute demonstrations."""
        return workbench.resources(query)

    @router.get("/resource")
    def resource(
        identity: str = Query(min_length=1, max_length=8192),
    ) -> dict[str, Any]:
        """Read exact source fields and named graphs for one resource."""
        try:
            return workbench.resource(identity)
        except KeyError:
            raise HTTPException(404, "RDF resource not found") from None

    @router.post("/analyze")
    def analyze() -> dict[str, Any]:
        """Validate a new source hash once and reopen persisted reports thereafter."""
        return workbench.analyze()

    @router.get("/proposals")
    def proposals() -> list[dict[str, Any]]:
        """Reopen Oracle-persisted drafts and review decisions."""
        return workbench.proposals()

    @router.post("/proposals")
    def propose(
        body: ProposalInput, x_m365_actor: str | None = Header(default=None)
    ) -> dict[str, Any]:
        """Persist a local model draft bound to the actual source content."""
        return workbench.propose(
            body.resourceId, body.sourceSha256, actor(x_m365_actor)
        )

    @router.post("/proposals/{identity}/decision")
    def decide(
        identity: str,
        body: DecisionInput,
        x_m365_actor: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Record a reasoned review decision; source-rule activation is separate."""
        try:
            return workbench.decide(
                identity,
                body.revision,
                body.sourceSha256,
                body.decision,
                body.reason,
                actor(x_m365_actor),
            )
        except KeyError:
            raise HTTPException(404, "RDF proposal not found") from None

    return router
