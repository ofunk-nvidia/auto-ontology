<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Experimental Oracle-only persistence MVP

This is a separate, explicitly limited vertical slice. It uses the existing
Auto-Ontology embedding and structured-output helpers, an Oracle JSON catalog,
native Oracle VECTOR cosine distance, and optimistic revisions. It does not
route persistence through PostgreSQL, pgvector, or SQLite.

The original application's default backend and its Next.js/Prisma authentication
database are unchanged. This MVP does **not** claim full compatibility with the
Postgres DAL, SQL-agent workflows, or the original frontend. A deployment can
instead place this private API behind an existing authenticated application
gateway. The gateway must authenticate and authorize the human caller; its
private technical credential must never be sent to the browser.

## Supported slice

| Capability | MVP |
| --- | --- |
| JSON catalog documents and term descriptions | Oracle CLOB with JSON constraint |
| Compare-and-swap updates | Revision predicate in one Oracle transaction |
| Vector storage and search | Oracle VECTOR(384, FLOAT32), exact cosine distance |
| Embeddings and proposed descriptions | Existing Auto-Ontology helpers, local endpoints |
| Proposal review status | Persisted as unapproved; cannot self-certify |
| Source RDF updates or publication | Not implemented; source stays unchanged |
| Arbitrary SQL execution | Not exposed |
| Original Next.js authentication | Not part of this MVP |
| Full original API compatibility | Not claimed; inspect `/api/capabilities` |

The first candidate deployment is an existing ARM CPU VM with a separate
Always-Free Autonomous AI Database 26ai. Free eligibility is an operator
provisioning responsibility: verify `is_free_tier=true`, fixed 20 GB storage,
disabled autoscaling, regional availability and the actual compartment quota.
The application cannot provision, upgrade, scale, or purchase cloud resources.
The `alwaysFreeOnly` setting validates operator intent; it is not a live OCI
billing attestation or a spending cap. Never describe a unit test as proof of
free capacity or a completed Oracle deployment.

## Operator setup

1. Install the optional driver using the repository's supported uv workflow:

   ```sh
   uv sync --extra oracle
   ```

2. In a **separate test database**, create a dedicated schema with a bounded
   quota. The runtime schema needs login, its own catalog table and VECTOR
   support. It needs no OCI administration signing key or source-RDF write
   privileges. Apply `auto_ontology.oracle_mvp.store.DDL` explicitly with the
   operator tooling; API startup never runs DDL or creates a database.

3. Write private local password, wallet-password and gateway-key files. On
   POSIX use owner-only permissions. Keep wallet files private too. Do not
   commit any wallet, password, live database identifier or operator config.
   The gateway key must contain at least 32 unpredictable characters.

4. Write an operator configuration outside the checkout:

   ```json
   {
     "backend": "oracle",
     "alwaysFreeOnly": true,
     "databaseName": "SYNTHETIC_FREE_DB",
     "user": "AO_MVP",
     "dsn": "synthetic_low",
     "passwordFile": "/private/oracle-password",
     "walletDirectory": "/private/wallet",
     "walletPasswordFile": "/private/wallet-password",
     "dimensions": 384
   }
   ```

5. Set `AUTO_ONTOLOGY_ORACLE_CONFIG` to that file and
   `AUTO_ONTOLOGY_ORACLE_GATEWAY_KEY_FILE` to the private technical key.
   Explicitly configure **all five** model endpoint triplets
   (`DEFAULT_MODELS`, `REASONING`, `NON_REASONING`, `EMBED`, `RERANK`) to
   loopback. Missing or external endpoints are rejected. There are no hosted
   defaults in this mode. A small local model is experimental and can produce
   inaccurate suggestions; human review remains necessary.

6. Start only on loopback, behind the trusted gateway:

   ```sh
   uv run uvicorn auto_ontology.oracle_mvp.api:create_app \
     --factory --host 127.0.0.1 --port 3011 --workers 1
   ```

   Never expose this listener directly as the user-facing application.
   `/api/health/live` and `/api/capabilities` are public diagnostic descriptions.
   Catalog, search, proposal and readiness routes require the private gateway
   bearer key. The gateway enforces user and workspace access before forwarding.

The versioned templates in `deploy/oracle-mvp/` describe this separate runtime.
They contain no Postgres settings or Postgres service dependency. Substitute
operator paths and provision the service user before installation. The systemd
example allows loopback only; add a reviewed drop-in for the selected Oracle
endpoint IP addresses before expecting readiness. Keep hosted inference blocked.
The `openai/local-ontology` model name denotes a locally created Ollama alias for
its OpenAI-compatible protocol, not an OpenAI service. Configure that alias to
an actually installed local model. The templates do not create cloud resources,
transfer secrets or install the original Next.js application.

## Acceptance

Unit tests cover malformed vectors, unsafe credentials, JSON integrity, stale
revisions, rollback, unsupported caller fields, gateway authentication and the
unapproved proposal invariant:

```sh
uv run pytest auto_ontology/oracle_mvp/tests -q
uv run ruff check auto_ontology/oracle_mvp
uv run ruff format --check auto_ontology/oracle_mvp
```

Live acceptance must additionally prove actual Oracle table creation, a native
VECTOR nearest-neighbor query, Unicode/LOB round trips, stale-revision rejection,
transaction rollback and persistence after restart. Repeat the application
checks with PostgreSQL stopped and no `POSTGRES_*` settings. Check local model
egress restrictions, memory/CPU limits and the actual OCI free status separately.
Only after these succeed may the trial Postgres runtime be stopped or removed.
Keep its data until an explicit cleanup decision.

The live check on 2026-11-09 passed three tests against an isolated Oracle
26ai Always-Free database using python-oracledb 3.4.2. This proved Unicode/LOB
round trips, independent-pool readback, stale and duplicate revision rollback,
native cosine distances, API authentication, conflict responses and readback
through a fresh application instance after shutdown. A JSON output type handler preserves the original
CLOB serialization for checksum validation; the driver's default auto-decoding
would otherwise discard it. This local result does not certify a cloud service
cutover, model quality, the original UI or a full upstream DAL port.

The original frontend also passed a build, ESLint (one existing warning), and
Prettier with native checkout line endings accepted. Original backend OpenAPI
generation and migration/schema comparison passed. A broader ARM/PG16 regression
run had 2,586 passing tests and 41 failing/erroring nodes. A fresh checkout of the
pinned, unmodified upstream reproduced exactly those same 41 nodes (2,570
passing tests). This comparison found no newly failing nodes; it is not a green
upstream CI run. The prescribed PG17/x86_64 CI, including optional Kumo wheels,
must still run for a published PR. The PG16 fixture setup omitted only the PG17
`SET transaction_timeout` statement in memory; source fixture files were not
modified.

## Upstream relevance

Oracle is a plausible storage-provider integration because it can host catalog
JSON and native vectors in the same database. This prototype intentionally
does not present a completed port of the existing Postgres schema. Work remaining
for a full backend includes dialect-specific DAL queries, JSON/array operators,
upsert semantics, migration ownership, full-text search, connection scoping and
the original authentication store.

The repository's CONTRIBUTING.md currently directs external storage integrations
to an integration request rather than an external code PR. Review the tested
prototype in the fork and submit a synthetic-only integration request with an
accurate capability matrix and live-test receipt. Do not publish credentials,
private tenancy details, or customer data.
