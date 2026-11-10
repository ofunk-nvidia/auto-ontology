# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private-gateway Oracle-only MVP; original Prisma/Next.js auth is out of scope."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import asyncio
import hmac
import os
from pathlib import Path
from typing import Any, Literal
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from auto_ontology.oracle_mvp.inference import LocalInference
from auto_ontology.oracle_mvp.settings import OracleSettings, private_text
from auto_ontology.oracle_mvp.store import OracleStore, RevisionConflict


class CatalogWrite(BaseModel):
    """A bounded operator-supplied catalog document, never connection credentials."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    kind: Literal["catalog", "term"]
    expected_revision: int = Field(ge=0)
    label: str = Field(min_length=1, max_length=200)
    description: str = Field(max_length=4000)
    source: dict[str, str | int] = Field(default_factory=dict, max_length=10)


class ProposalRequest(BaseModel):
    """Describe one selected term; arbitrary SQL and provider URLs are forbidden."""

    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=200)
    context: str = Field(min_length=1, max_length=4000)
    source: dict[str, str | int] = Field(default_factory=dict, max_length=10)


def create_app(
    *,
    store: OracleStore | None = None,
    inference: LocalInference | None = None,
    gateway_key: str | None = None,
    rdf_workbench: Any | None = None,
) -> FastAPI:
    """Create only the supported Oracle vertical slice, with no Postgres imports."""
    if store is None:
        path = os.environ.get("AUTO_ONTOLOGY_ORACLE_CONFIG")
        if not path:
            raise ValueError("An explicit operator Oracle configuration is required")
        store = OracleStore(OracleSettings.from_file(Path(path)))
    runtime = inference or LocalInference()
    if gateway_key is None:
        key_file = os.environ.get("AUTO_ONTOLOGY_ORACLE_GATEWAY_KEY_FILE")
        if not key_file:
            raise ValueError("An explicit private gateway key file is required")
        gateway_key = private_text(Path(key_file))
    if len(gateway_key) < 32:
        raise ValueError("The private gateway key is too short")

    source_config = os.environ.get("AUTO_ONTOLOGY_RDF_SOURCE_CONFIG")
    if rdf_workbench is None and source_config:
        from auto_ontology.oracle_rdf.source import OracleRdfSource, SourceSettings
        from auto_ontology.oracle_rdf.workbench import RdfWorkbench

        rdf_workbench = RdfWorkbench(
            OracleRdfSource(
                SourceSettings.from_file(
                    Path(source_config), store.settings.database_name
                )
            ),
            store,
            runtime,
        )

    async def autonomous_analysis() -> None:
        while True:
            try:
                await asyncio.to_thread(rdf_workbench.analyze)
            except Exception:
                # Request endpoints still expose failure status. Never log source
                # contents/credentials or start interactive provider sign-in.
                pass
            await asyncio.sleep(300)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        task = (
            asyncio.create_task(autonomous_analysis())
            if rdf_workbench is not None
            else None
        )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                rdf_workbench.close()
            store.close()

    app = FastAPI(title="Auto-Ontology Oracle persistence MVP", lifespan=lifespan)
    if rdf_workbench is not None:
        from auto_ontology.oracle_rdf.router import router_for

        app.include_router(router_for(rdf_workbench))

    @app.middleware("http")
    async def trusted_gateway(request: Request, call_next: Any) -> Any:
        if request.url.path not in ("/api/health/live", "/api/capabilities"):
            supplied = request.headers.get("authorization", "")
            if not hmac.compare_digest(
                supplied.encode("utf-8"), ("Bearer " + gateway_key).encode("utf-8")
            ):
                return JSONResponse(
                    status_code=401, content={"detail": "Private gateway required"}
                )
        return await call_next(request)

    @app.exception_handler(RevisionConflict)
    def conflict(_request: Any, _error: RevisionConflict) -> JSONResponse:
        return JSONResponse(
            status_code=409, content={"detail": "Revision changed; reload first"}
        )

    @app.exception_handler(ValueError)
    def invalid(_request: Any, _error: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"detail": "Invalid catalog or inference input"}
        )

    @app.get("/api/health/live")
    def live() -> dict[str, Any]:
        """Liveness does not connect to any database or inference provider."""
        return {"status": "ok", "mode": "oracle-mvp", "postgresRequired": False}

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Readiness verifies the actual Oracle schema; no fallback is attempted."""
        try:
            return {"status": "ok", **store.health(), "postgresRequired": False}
        except Exception:
            # Driver exceptions may include connection details; never expose them.
            raise HTTPException(503, "Oracle catalog unavailable") from None

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        """Make unsupported upstream surfaces explicit instead of claiming parity."""
        return {
            "mode": "oracle-mvp",
            "catalog": True,
            "optimisticRevisions": True,
            "nativeOracleVectorSearch": True,
            "localModelProposals": True,
            "originalNextJsAuthentication": False,
            "fullUpstreamDalCompatibility": False,
            "arbitrarySqlExecution": False,
            "automaticRdfPublication": False,
        }

    @app.get("/api/catalog")
    def catalog(
        kind: Literal["catalog", "term", "proposal"] = "term",
    ) -> list[dict[str, Any]]:
        """Read explicitly stored Oracle catalog documents."""
        return store.list(kind)

    @app.get("/api/catalog/{identity}")
    def document(identity: str) -> dict[str, Any]:
        """Read a catalog revision and its content checksum from Oracle."""
        result = store.get(identity)
        if result is None:
            raise HTTPException(404, "Catalog document not found")
        return result

    @app.post("/api/catalog")
    def save(body: CatalogWrite) -> dict[str, Any]:
        """Save only the MVP catalog; no source database or RDF model is updated."""
        value = body.model_dump(exclude={"id", "kind", "expected_revision"})
        vector = (
            runtime.embed(body.label + ": " + body.description)
            if body.kind == "term"
            else None
        )
        return store.save(body.id, body.kind, body.expected_revision, value, vector)

    @app.get("/api/search")
    def search(
        query: str = Query(min_length=1, max_length=1000),
        limit: int = Query(default=5, ge=1, le=20),
    ) -> list[dict[str, Any]]:
        """Search catalog terms using local embeddings and Oracle cosine distance."""
        return store.search(runtime.embed(query, query=True), limit=limit)

    @app.post("/api/proposals")
    def propose(body: ProposalRequest) -> dict[str, Any]:
        """Persist an unapproved local-AI proposal in Oracle, separately from RDF."""
        proposal = runtime.propose(body.label, body.context)
        value = {
            **proposal,
            "approved": False,
            "source": body.source,
            "status": "proposed",
        }
        vector = runtime.embed(body.label + ": " + proposal["description"])
        return store.save("proposal-" + uuid.uuid4().hex, "proposal", 0, value, vector)

    return app
