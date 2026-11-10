# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, source-bound RDF/SHACL checks before any model suggestion."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from rdflib import Literal, RDF
from rdflib.namespace import SH

from auto_ontology.oracle_rdf.source import Snapshot, SourceSettings


@dataclass(frozen=True)
class Validation:
    source_id: str
    source_sha256: str
    conforms: bool
    findings: tuple[dict[str, Any], ...]
    finding_count: int
    duration_ms: int
    truncated: bool
    phase: str = "data"


def _has_service(value: Any) -> bool:
    from pyparsing import ParseResults
    from rdflib.plugins.sparql.parserutils import CompValue

    if isinstance(value, CompValue) and value.name == "ServiceGraphPattern":
        return True
    if isinstance(value, dict):
        return any(_has_service(child) for child in value.values())
    if isinstance(value, (list, tuple, ParseResults)):
        return any(_has_service(child) for child in value)
    return False


def validate_snapshot(snapshot: Snapshot, settings: SourceSettings) -> Validation:
    """Use the declared data/shapes only; never load imports or execute JavaScript."""
    from pyshacl import validate
    from pyshacl.entrypoints import meta_validate
    from rdflib.plugins.sparql.parser import parseQuery

    if snapshot.source_id != settings.source_id:
        raise ValueError("Validation source binding mismatch")
    data = snapshot.selected(settings.data_graphs)
    shapes = snapshot.selected(settings.shape_graphs)
    if not any(shapes.triples((None, RDF.type, SH.NodeShape))) and not any(
        shapes.triples((None, RDF.type, SH.PropertyShape))
    ):
        raise ValueError("Configured shape graph contains no SHACL shapes")
    if any(shapes.triples((None, SH.js, None))):
        raise ValueError("Executable SHACL JavaScript is outside this validation mode")
    for predicate in (SH.select, SH.ask):
        for _, _, query in shapes.triples((None, predicate, None)):
            if not isinstance(query, Literal) or _has_service(parseQuery(str(query))):
                raise ValueError(
                    "External SPARQL services are forbidden in source validation"
                )
    started = time.monotonic()
    # Invalid source shapes are a real, explainable finding. Checking them first
    # avoids pySHACL's meta_shacl exception (and its unbounded logged report).
    # Never coerce Oracle literals or disable shape validation to obtain a pass.
    options = dict(
        inference="none",
        abort_on_first=False,
        allow_infos=False,
        allow_warnings=False,
        advanced=False,
        js=False,
        do_owl_imports=False,
        max_validation_depth=30,
    )
    conforms, report, _ = meta_validate(shapes, **options)
    phase = "shapes"
    if conforms:
        conforms, report, _ = validate(
            data_graph=data, shacl_graph=shapes, meta_shacl=False, **options
        )
        phase = "data"
    if not hasattr(report, "subjects"):
        raise ValueError("Validation failed to produce an RDF report")
    nodes = list(report.subjects(RDF.type, SH.ValidationResult))
    findings = []
    for result in nodes[:200]:
        value = {
            "focus": str(report.value(result, SH.focusNode) or ""),
            "path": str(report.value(result, SH.resultPath) or ""),
            "severity": str(report.value(result, SH.resultSeverity) or SH.Violation),
            "shape": str(report.value(result, SH.sourceShape) or ""),
            "constraint": str(report.value(result, SH.sourceConstraintComponent) or ""),
            "messages": [
                str(message)[:1000]
                for message in list(report.objects(result, SH.resultMessage))[:3]
            ],
        }
        offending = report.value(result, SH.value)
        if offending is not None:
            value["value"] = offending.n3()
        findings.append(value)
    return Validation(
        snapshot.source_id,
        snapshot.sha256,
        bool(conforms),
        tuple(findings),
        len(nodes),
        int((time.monotonic() - started) * 1000),
        len(nodes) > 200,
        phase,
    )
