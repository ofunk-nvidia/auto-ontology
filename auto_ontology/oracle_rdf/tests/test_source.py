# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RDF fidelity, explicit source isolation and explainable SHACL failures."""

from dataclasses import replace
from pathlib import Path

import pytest
from rdflib import BNode, Literal, RDF, RDFS, URIRef
from rdflib.namespace import SH, XSD

from auto_ontology.oracle_mvp.settings import OracleSettings
from auto_ontology.oracle_rdf.source import SourceSettings, _node, snapshot_from_rows
from auto_ontology.oracle_rdf.validation import validate_snapshot


def settings() -> SourceSettings:
    return SourceSettings(
        "synthetic",
        "Synthetic RDF source",
        OracleSettings(
            "READER",
            "synthetic_low",
            Path("password"),
            Path("wallet"),
            Path("wallet-password"),
            "SOURCE",
        ),
        "OWNER",
        "FIXED_SOURCE",
        ("urn:example:data",),
        ("urn:example:shapes",),
    )


def source_rows():
    return [
        (
            "urn:example:data",
            "urn:example:item",
            "URI",
            str(RDF.type),
            "urn:example:Widget",
            "URI",
            None,
            None,
        ),
        (
            "urn:example:data",
            "urn:example:item",
            "URI",
            str(RDFS.label),
            "Widget",
            "LIT",
            None,
            "de",
        ),
        (
            "urn:example:shapes",
            "urn:example:Shape",
            "URI",
            str(RDF.type),
            str(SH.NodeShape),
            "URI",
            None,
            None,
        ),
        (
            "urn:example:shapes",
            "urn:example:Shape",
            "URI",
            str(SH.targetClass),
            "urn:example:Widget",
            "URI",
            None,
            None,
        ),
        (
            "urn:example:shapes",
            "urn:example:Shape",
            "URI",
            str(SH.property),
            "_:property",
            "BLN",
            None,
            None,
        ),
        (
            "urn:example:shapes",
            "_:property",
            "BLN",
            str(SH.path),
            str(RDFS.label),
            "URI",
            None,
            None,
        ),
        (
            "urn:example:shapes",
            "_:property",
            "BLN",
            str(SH.minCount),
            "1",
            "LIT",
            str(XSD.integer),
            None,
        ),
    ]


def test_literal_spelling_language_and_blank_node_identity_are_preserved():
    value = _node("01", "LIT", str(XSD.integer))
    assert str(value) == "01" and value.datatype == XSD.integer
    assert _node("Bezeichnung", "LIT", str(RDF.langString), "de").language == "de"
    assert _node("_:property", "BLN") == BNode("property")
    with pytest.raises(ValueError):
        _node("Text", "LIT", str(XSD.integer), "de")


def test_snapshot_hash_is_order_independent_but_includes_datatypes():
    config = settings()
    rows = source_rows()
    original = snapshot_from_rows(config, rows)
    assert original.sha256 == snapshot_from_rows(config, reversed(rows)).sha256
    changed = [*rows[:-1], (*rows[-1][:6], str(XSD.decimal), None)]
    assert original.sha256 != snapshot_from_rows(config, changed).sha256
    assert sum(original.graph_counts.values()) == 7


def test_unconfigured_or_missing_graph_and_literal_subject_are_rejected():
    config = settings()
    rows = source_rows()
    with pytest.raises(ValueError, match="unconfigured"):
        snapshot_from_rows(config, [(*rows[0][:0], "urn:example:other", *rows[0][1:])])
    with pytest.raises(ValueError, match="absent"):
        snapshot_from_rows(config, rows[:2])
    with pytest.raises(ValueError, match="subject"):
        snapshot_from_rows(config, [(rows[0][0], "text", "LIT", *rows[0][3:])])


def test_valid_shapes_and_data_pass_with_bound_source_evidence():
    config = settings()
    snapshot = snapshot_from_rows(config, source_rows())
    report = validate_snapshot(snapshot, config)
    assert report.conforms and report.phase == "data" and not report.findings
    assert report.source_sha256 == snapshot.sha256
    with pytest.raises(ValueError, match="binding"):
        validate_snapshot(snapshot, replace(config, source_id="different"))


def test_invalid_source_shapes_return_findings_without_casting_literals():
    config = settings()
    rows = source_rows()
    rows[-1] = (*rows[-1][:6], str(XSD.decimal), None)
    snapshot = snapshot_from_rows(config, rows)
    report = validate_snapshot(snapshot, config)
    assert not report.conforms and report.phase == "shapes"
    assert report.finding_count > 0 and report.findings[0]["path"] == str(SH.minCount)
    assert (
        BNode("property"),
        SH.minCount,
        Literal("1", datatype=XSD.decimal),
    ) in snapshot.dataset


def test_missing_required_value_is_reported_as_data_failure():
    config = settings()
    rows = [
        row
        for row in source_rows()
        if not (row[0] == "urn:example:data" and row[3] == str(RDFS.label))
    ]
    report = validate_snapshot(snapshot_from_rows(config, rows), config)
    assert not report.conforms and report.phase == "data"
    assert report.findings[0]["focus"] == "urn:example:item"


@pytest.mark.parametrize(
    "predicate,value",
    [
        (SH.js, URIRef("urn:example:script")),
        (
            SH.select,
            Literal(
                "SELECT ?this WHERE { SERVICE <https://example.invalid> { ?this ?p ?o } }"
            ),
        ),
    ],
)
def test_executable_or_external_validators_are_rejected(predicate, value):
    config = settings()
    snapshot = snapshot_from_rows(config, source_rows())
    snapshot.dataset.graph(URIRef(config.shape_graphs[0])).add(
        (BNode("constraint"), predicate, value)
    )
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, config)
