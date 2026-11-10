# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a bounded, operator-owned Oracle RDF projection with read-only sessions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from rdflib import BNode, Dataset, Literal, URIRef

from auto_ontology.oracle_mvp.settings import OracleSettings

_IDENTIFIER = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_MAX_ROWS = 50000
_MAX_BYTES = 16 * 1024 * 1024


def _iri(value: str) -> URIRef:
    if (
        not value
        or len(value) > 8192
        or ":" not in value
        or re.search(r'[\s<>"{}|^`\\]', value)
    ):
        raise ValueError("The source contains an invalid absolute RDF IRI")
    return URIRef(value)


def _node(
    value: str, kind: str, datatype: str | None = None, language: str | None = None
) -> URIRef | BNode | Literal:
    """Preserve literal spelling, language and datatype instead of normalizing values."""
    if kind == "URI":
        return _iri(value)
    if kind in ("BLN", "BNODE"):
        return BNode(value.removeprefix("_:"))
    if kind.startswith("LIT"):
        if language and datatype:
            # Oracle may report rdf:langString alongside a language tag.
            if datatype != "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString":
                raise ValueError("Source literal has conflicting language and datatype")
        return Literal(
            value,
            lang=language or None,
            datatype=_iri(datatype) if datatype and not language else None,
            normalize=False,
        )
    raise ValueError("Unsupported RDF term kind in source projection")


@dataclass(frozen=True)
class SourceSettings:
    source_id: str
    label: str
    connection: OracleSettings
    owner: str
    view: str
    data_graphs: tuple[str, ...]
    shape_graphs: tuple[str, ...]

    @classmethod
    def from_file(cls, path: Path, application_database: str) -> SourceSettings:
        value = json.loads(path.read_text(encoding="utf-8"))
        connection = OracleSettings.from_file(path)
        if (
            value.get("role") != "rdf-source"
            or connection.database_name.upper() == application_database.upper()
        ):
            raise ValueError("RDF source and application database must be distinct")
        if connection.user.upper() == "ADMIN":
            raise ValueError("An application source reader cannot be ADMIN")
        owner, view = value["owner"], value["view"]
        if not _IDENTIFIER.fullmatch(owner) or not _IDENTIFIER.fullmatch(view):
            raise ValueError("Only fixed operator-owned source views are supported")
        if connection.user.upper() == owner:
            raise ValueError("The source runtime must not own its source schema")
        data = tuple(value["dataGraphs"])
        shapes = tuple(value["shapeGraphs"])
        if (
            not data
            or not shapes
            or len(data) + len(shapes) > 32
            or set(data) & set(shapes)
        ):
            raise ValueError("Distinct bounded data and shape graphs are required")
        for graph in (*data, *shapes):
            _iri(graph)
        return cls(
            value["sourceId"], value["label"], connection, owner, view, data, shapes
        )


@dataclass(frozen=True)
class Snapshot:
    source_id: str
    database_name: str
    sha256: str
    dataset: Dataset
    graph_counts: dict[str, int]

    def selected(self, graphs: tuple[str, ...]) -> Dataset:
        """A deliberate graph selection; never silently union unrelated workspaces."""
        missing = set(graphs) - set(self.graph_counts)
        if missing:
            raise ValueError("A configured source graph is absent")
        result = Dataset(default_union=True)
        for graph in graphs:
            for subject, predicate, obj in self.dataset.graph(URIRef(graph)):
                result.graph(URIRef(graph)).add((subject, predicate, obj))
        return result


def snapshot_from_rows(settings: SourceSettings, rows: Any) -> Snapshot:
    """Decode a complete source snapshot with stable RDF content evidence."""
    dataset = Dataset(default_union=True)
    size = 0
    allowed = set(settings.data_graphs) | set(settings.shape_graphs)
    for index, row in enumerate(rows):
        if index >= _MAX_ROWS:
            raise ValueError("Source snapshot exceeds the reviewed row limit")
        (
            graph,
            subject,
            subject_kind,
            predicate,
            obj,
            object_kind,
            datatype,
            language,
        ) = row
        values = [value.read() if hasattr(value, "read") else value for value in row]
        size += sum(
            len(value.encode("utf-8")) for value in values if isinstance(value, str)
        )
        if size > _MAX_BYTES:
            raise ValueError("Source snapshot exceeds the reviewed byte limit")
        (
            graph,
            subject,
            subject_kind,
            predicate,
            obj,
            object_kind,
            datatype,
            language,
        ) = values
        if graph not in allowed:
            raise ValueError("Source projection returned an unconfigured graph")
        s = _node(subject, subject_kind)
        if isinstance(s, Literal):
            raise ValueError("An RDF subject cannot be a literal")
        dataset.graph(_iri(graph)).add(
            (s, _iri(predicate), _node(obj, object_kind, datatype, language))
        )
    lines = sorted(
        f"{s.n3()} {p.n3()} {o.n3()} {g.n3()} ." for s, p, o, g in dataset.quads()
    )
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    counts = Counter(str(g) for _, _, _, g in dataset.quads())
    snapshot = Snapshot(
        settings.source_id,
        settings.connection.database_name,
        digest,
        dataset,
        dict(counts),
    )
    snapshot.selected(settings.data_graphs)
    snapshot.selected(settings.shape_graphs)
    return snapshot


class OracleRdfSource:
    """No SPARQL/SQL supplied by callers; no source write or local data fallback."""

    def __init__(self, settings: SourceSettings) -> None:
        import oracledb

        self.settings = settings
        self.pool = oracledb.create_pool(
            **settings.connection.connect_arguments(),
            min=0,
            max=1,
            increment=1,
            getmode=oracledb.POOL_GETMODE_TIMEDWAIT,
            wait_timeout=5000,
        )

    def close(self) -> None:
        self.pool.close()

    def snapshot(self) -> Snapshot:
        import oracledb

        def text_lob(cursor: Any, metadata: Any) -> Any:
            if metadata.type_code == oracledb.DB_TYPE_CLOB:
                return cursor.var(oracledb.DB_TYPE_LONG, arraysize=cursor.arraysize)
            return None

        with self.pool.acquire() as connection, connection.cursor() as cursor:
            connection.call_timeout = 15000
            cursor.arraysize = 100
            cursor.outputtypehandler = text_lob
            try:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute("SELECT SYS_CONTEXT('USERENV','DB_NAME') FROM dual")
                actual = cursor.fetchone()[0].upper()
                expected = self.settings.connection.database_name.upper()
                if actual != expected and not actual.endswith("_" + expected):
                    raise ValueError("Source connection belongs to another database")
                cursor.execute(
                    "SELECT GRAPH_VALUE,SUBJECT_VALUE,SUBJECT_KIND,PREDICATE_VALUE,"
                    "OBJECT_VALUE,OBJECT_KIND,OBJECT_DATATYPE,OBJECT_LANGUAGE "
                    f'FROM "{self.settings.owner}"."{self.settings.view}" '
                    "FETCH FIRST 50001 ROWS ONLY"
                )
                return snapshot_from_rows(self.settings, cursor)
            finally:
                connection.rollback()
