<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Oracle RDF source and Microsoft 365 application target

This extends the isolated Oracle persistence MVP with a source-bound RDF curation
API. `auto_ontology.oracle_rdf` implements source projection reads, SHACL reports,
local description drafts and optimistic review decisions. A separately deployed
SPFx consumer provides the private Microsoft 365 application surface. Deployment
and actual-host acceptance remain specific to that consumer.

This path implements the RDF curation workflow below. It does not replace every
upstream relational ingestion adapter, Prisma model or original Next.js screen.
The original application remains available to its existing PostgreSQL users.

## Two distinct database roles

The application database replaces PostgreSQL for internal catalog, revisions,
findings, proposals, job state and vector search. A different Oracle database is
the authoritative RDF source. Source profiles and connections must remain
separate; an application database must never be silently treated as its source.
Prefer a dedicated source reader with SELECT on a fixed, operator-owned RDF
projection. Do not give the runtime a source administrator credential, a generic
SPARQL UPDATE surface, or source-schema ownership.

The first use case reads a rule-authority RDF graph, validates it, explains
findings, proposes improvements, and preserves source revision/content evidence
through review. Generated suggestions must not silently activate source rules.
An unchanged source should not trigger repeated model calls.

The API periodically checks source content every five minutes and reopens an
Oracle-persisted report for unchanged content. Source-shape validation runs before
data validation. Invalid shapes produce an explicit `phase: shapes` report;
literal datatypes are never guessed or cast to force a pass. Requests cannot
select arbitrary SQL, source profiles, provider URLs or executable validators.
The source reader requires only login and SELECT on a bounded fixed view.

Description proposals preserve the original predicate, language, datatype and
graph, show exact before/after values, and are reused for the same resource and
source hash. Review requires the expected application revision and a freshly
checked source hash. `accepted-for-review` records a durable decision and does
not activate rules or update the source. Export and the source owner's existing
publication/activation process remain separate.

## Operator configuration

Enable the optional dependencies with `uv sync --extra oracle --extra rdf`.
Keep the existing explicit `AUTO_ONTOLOGY_ORACLE_CONFIG` and private gateway key.
`AUTO_ONTOLOGY_RDF_SOURCE_CONFIG` points to a separate source-reader profile:
`role: rdf-source`, a different `databaseName`, a non-owner user, fixed `owner`
and `view`, and disjoint `dataGraphs` and `shapeGraphs`. Password/wallet paths
are private operator files. Never submit them through catalog documents.

The existing Oracle factory enables `/api/rdf/source`, `/resources`, `/resource`,
`/analyze` and `/proposals` only when a source profile is supplied. Every route
requires the private trusted gateway. Proposal/review writes additionally require
a verified `X-M365-Actor`; the gateway must derive it from actual Microsoft 365
metadata, not trust a caller-supplied identity. No public API listener is required
by the private-channel deployment. The local-only inference restriction remains
in force; no hosted-model key discovery or paid inference fallback is added.

## SharePoint and Teams

Use React/SPFx and Microsoft 365 identity for the application surface. The
deployment owns its existing shared M365 theme and application-frame adapter;
private theme packages and real tenant configuration do not belong in this
public storage-provider integration. Keep public application components and
operator-only host bindings separate. Database credentials never enter SPFx.

Next.js/Prisma is not the runtime authentication store on this deployment path.
Every server-side operation needed by the chosen workflows must be available
through the authenticated API with authorization checked before data access.
Do not declare a normal, usable application from a database probe, a synthetic
screen, a package build or three new CRUD endpoints.

## Acceptance

- Open the real private Teams entry and discover the selected RDF source.
- Read graph/resources, run validation, explain source-bound findings.
- Produce and review a bounded proposal with a genuine before/after comparison.
- Persist decisions in the application Oracle database and reopen after restart.
- Reject outdated proposals when the source revision changes.
- Demonstrate autonomous processing of a new source without repeated inference
  for the same content; make unsupported model capabilities explicit.
- Keep PostgreSQL stopped, source authority protected and all resources within
  the proven free deployment.
- Verify actual host theming, keyboard access, populated/empty/error states and
  the shared application frame; a successful build alone is insufficient.

The existing upstream relational catalog, ingestion and publication contracts
remain relevant. RDF source mapping and validation are explicit integration
work; model certification is not a substitute for source publication approval.
