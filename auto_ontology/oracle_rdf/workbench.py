# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-bound RDF curation with durable Oracle reports and review decisions."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import threading
import time
from typing import Any
import uuid

from rdflib import BNode, Literal, RDFS, URIRef
from rdflib.namespace import SKOS

from auto_ontology.oracle_mvp.inference import LocalInference
from auto_ontology.oracle_mvp.store import OracleStore, RevisionConflict
from auto_ontology.oracle_rdf.source import OracleRdfSource, Snapshot
from auto_ontology.oracle_rdf.validation import validate_snapshot


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def term(value: Any) -> dict[str, str]:
    if isinstance(value, Literal):
        return dict(
            kind="literal",
            value=str(value),
            language=value.language or "",
            datatype=str(value.datatype or ""),
        )
    return dict(kind="blank" if isinstance(value, BNode) else "iri", value=str(value))


def resource_key(value: Any) -> str:
    return "_:" + str(value) if isinstance(value, BNode) else str(value)


class RdfWorkbench:
    """The source is read-only; curation state lives only in the other database."""

    def __init__(
        self, source: OracleRdfSource, store: OracleStore, inference: LocalInference
    ) -> None:
        if (
            source.settings.connection.database_name.upper()
            == store.settings.database_name.upper()
        ):
            raise ValueError("Source and application databases must differ")
        self.source, self.store, self.inference = source, store, inference
        self.lock = threading.RLock()
        self.cached: Snapshot | None = None
        self.cached_at = 0.0
        self.job_state: dict[str, Any] = dict(status="idle")

    def close(self) -> None:
        self.source.close()

    def snapshot(self, *, fresh: bool = False) -> Snapshot:
        with self.lock:
            if fresh or self.cached is None or time.monotonic() - self.cached_at > 15:
                self.cached = self.source.snapshot()
                self.cached_at = time.monotonic()
            return self.cached

    def summary(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        health = self.store.health()
        return dict(
            sourceId=snapshot.source_id,
            label=self.source.settings.label,
            sourceDatabase=snapshot.database_name,
            applicationDatabase=health["database"],
            sourceSha256=snapshot.sha256,
            graphs=snapshot.graph_counts,
            sourceReadOnly=True,
            postgresRequired=False,
            autonomousIntervalSeconds=300,
            sourcePublicationAvailable=False,
            analysis=self.job_state,
        )

    def resources(self, query: str = "") -> list[dict[str, str]]:
        snapshot = self.snapshot()
        values = set(snapshot.dataset.subjects())
        rows = []
        for node in values:
            labels = list(snapshot.dataset.objects(node, RDFS.label))
            label = next(
                (str(v) for v in labels if getattr(v, "language", None) == "de"),
                str(labels[0]) if labels else resource_key(node),
            )
            identity = resource_key(node)
            if query.casefold() in (label + " " + identity).casefold():
                rows.append(dict(id=identity, label=label))
        return sorted(rows, key=lambda row: (row["label"].casefold(), row["id"]))[:500]

    def resource(self, identity: str) -> dict[str, Any]:
        snapshot = self.snapshot()
        node = BNode(identity[2:]) if identity.startswith("_:") else URIRef(identity)
        rows = [
            dict(subject=term(s), predicate=str(p), object=term(o), graph=str(g))
            for s, p, o, g in snapshot.dataset.quads((node, None, None, None))
        ]
        if not rows:
            raise KeyError("Resource is absent from the current source")
        return dict(
            id=identity,
            sourceSha256=snapshot.sha256,
            statements=rows[:200],
            truncated=len(rows) > 200,
        )

    def analyze(self) -> dict[str, Any]:
        """Process changed source content once, including invalid source shapes."""
        with self.lock:
            self.job_state = dict(status="running", startedAt=utc_now())
            try:
                result = self._analyze()
                self.job_state = dict(
                    status="complete",
                    checkedAt=utc_now(),
                    sourceSha256=result["document"]["sourceSha256"],
                    reused=result["reused"],
                )
                return result
            except Exception:
                self.job_state = dict(
                    status="failed",
                    checkedAt=utc_now(),
                    message="Analyse fehlgeschlagen. Quelle und Oracle-Verbindung prüfen; es wurde kein Ersatzbestand verwendet.",
                )
                raise

    def _analyze(self) -> dict[str, Any]:
        with self.lock:
            snapshot = self.snapshot(fresh=True)
            identity = "rdf-report-" + snapshot.sha256
            existing = self.store.get(identity)
            if existing is not None:
                return dict(existing, reused=True)
            report = validate_snapshot(snapshot, self.source.settings)
            value = dict(
                type="rdf-report",
                sourceId=snapshot.source_id,
                sourceSha256=snapshot.sha256,
                createdAt=utc_now(),
                graphs=snapshot.graph_counts,
                validation=asdict(report),
                sourceWrites=False,
                modelCalls=0,
                nextAction=(
                    "Prüfbefunde an der Quelle vergleichen; bei Datentypbefunden zuerst die Oracle-RDF-Abbildung prüfen."
                    if not report.conforms
                    else "Ressourcen prüfen und begründete Verbesserungsvorschläge erstellen."
                ),
            )
            try:
                return dict(
                    self.store.save(identity, "catalog", 0, value), reused=False
                )
            except RevisionConflict:
                result = self.store.get(identity)
                if result is None:
                    raise
                return dict(result, reused=True)

    def proposals(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self.store.list("proposal")
            if row["document"].get("type") == "rdf-proposal"
            and row["document"].get("sourceId") == self.source.settings.source_id
        ]

    def propose(
        self, identity: str, expected_source: str, actor: str
    ) -> dict[str, Any]:
        with self.lock:
            return self._propose(identity, expected_source, actor)

    def _propose(
        self, identity: str, expected_source: str, actor: str
    ) -> dict[str, Any]:
        snapshot = self.snapshot(fresh=True)
        if snapshot.sha256 != expected_source:
            raise RevisionConflict(
                "Source changed; re-read before requesting a proposal"
            )
        for existing in self.proposals():
            value = existing["document"]
            if (
                value["sourceSha256"] == expected_source
                and value["resourceId"] == identity
            ):
                return dict(existing, reused=True)
        resource = self.resource(identity)
        label = next(
            (
                row["object"]["value"]
                for row in resource["statements"]
                if row["predicate"] == str(RDFS.label)
            ),
            identity,
        )
        descriptions = [
            row
            for row in resource["statements"]
            if row["predicate"] in (str(RDFS.comment), str(SKOS.definition))
            and row["object"]["kind"] == "literal"
        ]
        if len(descriptions) != 1:
            raise ValueError(
                "Select a resource with exactly one description; ambiguous source fields are not changed"
            )
        before = descriptions[0]
        suggestion = self.inference.propose(label, str(resource["statements"])[:4000])
        after = dict(
            before, object=dict(before["object"], value=suggestion["description"])
        )
        value = dict(
            type="rdf-proposal",
            sourceId=snapshot.source_id,
            sourceSha256=snapshot.sha256,
            resourceId=identity,
            label=label,
            before=before,
            after=after,
            status="proposed",
            createdAt=utc_now(),
            createdBy=actor,
            reason="Lokaler Modellvorschlag auf Grundlage der ausgewählten Quelle; fachlich zu prüfen.",
            model="local-qwen3-0.6b",
            sourceApplied=False,
            decisions=[],
        )
        return self.store.save("rdf-proposal-" + uuid.uuid4().hex, "proposal", 0, value)

    def decide(
        self,
        identity: str,
        revision: int,
        expected_source: str,
        decision: str,
        reason: str,
        actor: str,
    ) -> dict[str, Any]:
        if (
            decision not in ("accepted-for-review", "rejected")
            or not 1 <= len(reason.strip()) <= 2000
        ):
            raise ValueError(
                "A bounded reason and explicit review decision are required"
            )
        current = self.store.get(identity)
        if current is None or current["document"].get("type") != "rdf-proposal":
            raise KeyError("RDF proposal not found")
        value = current["document"]
        if (
            current["revision"] != revision
            or value["sourceId"] != self.source.settings.source_id
        ):
            raise RevisionConflict("Proposal revision changed")
        if (
            value["sourceSha256"] != expected_source
            or self.snapshot(fresh=True).sha256 != expected_source
        ):
            raise RevisionConflict("Source changed; the proposal must be recreated")
        if value["status"] != "proposed":
            raise RevisionConflict("Proposal already has a review decision")
        event = dict(
            decision=decision, reason=reason.strip(), actor=actor, decidedAt=utc_now()
        )
        return self.store.save(
            identity,
            "proposal",
            revision,
            dict(value, status=decision, decisions=[*value["decisions"], event]),
        )
