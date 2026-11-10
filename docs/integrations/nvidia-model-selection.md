<!-- SPDX-License-Identifier: Apache-2.0 -->
# NVIDIA model selection for the Oracle RDF workbench

This optional extension uses the OpenAI-compatible NVIDIA Inference Hub endpoint
`https://inference-api.nvidia.com/v1`. It does not change upstream model defaults,
the Oracle source, the embedding index, or the original PostgreSQL/Next.js path.

## Requirements and fit

The repository's `README.md` describes separate REASONING, NON_REASONING, EMBED
and RERANK roles. `auto_ontology/utils/llm_invoke.py` constructs structured-output
clients, including a function-calling alternative for some providers. A model's
reasoning category alone does not prove that these wire protocols work.

The RDF workbench's fit profile requires text generation, system instructions,
structured JSON **or** a valid function result, exact preservation of identifiers
and relationship direction, and a concise German description. Its synthetic
check also tests a subclass/type inference and rejects unsupported inverse and
ownership relationships. It checks JSON-schema and function-calling separately.
No real RDF source is sent during this capability test. A passing check is not a
language-quality benchmark, general prompt-injection defense, maximum-context
certification, or full upstream ingestion/text-to-SQL acceptance.

Model cards have two evidence levels:

1. **Documented candidate**: an exact model family has primary model-card or
   upstream evidence and the authenticated `/models` catalog contains its ID.
2. **Tested for the RDF workflow**: a recent synthetic endpoint receipt passed.
   Only this level, together with operator approval and model-specific verified
   free-use evidence, permits activation.

Known families are reviewed in `oracle_mvp/model_selection.py`. Unknown model IDs
are excluded rather than guessed from a name containing “reasoning”. Add further
families with primary evidence and keep actual deployment IDs unmodified.
Embedding, reranking, speech, image-only and document-parser models do not belong
in the text-model picker. A multimodal model may qualify when its text API meets
the same requirements; multimodality alone is neither required nor disqualifying.
The model card distinguishes manufacturer context information from the unverified
context limit of the concrete deployment.

## Operator configuration

Create a private operator profile outside the repository, owned by the service
user with mode 0600. Set `AUTO_ONTOLOGY_NVIDIA_PROFILE` to its path. Example:

```json
{
  "id": "synthetic-connection-revision-1",
  "baseUrl": "https://inference-api.nvidia.com/v1",
  "keyFile": "/private/operator/nvidia-key",
  "operatorActors": ["00000000-0000-4000-8000-000000000001"],
  "useApproval": "",
  "costMode": "blocked",
  "costEvidence": "",
  "freeModelIds": []
}
```

The stored key is sufficient to authenticate and read `/models`. The catalog
includes all returned model IDs; only documented text-model candidates receive
RDF model cards. Catalog listing is independent of the generation controls below.

This example intentionally cannot start external generation. Record the applicable
provider/use approval, independently verify free use for the selected model IDs,
and only then set `costMode` to `verified-free`, add `costEvidence` and the exact
`freeModelIds`. Merely seeing a model in a catalog does not establish its price.
Use a new profile ID after credential/provider authorization changes; old test
receipts must not be reused as evidence for another connection.

Allow outbound HTTPS to the reviewed host in the existing service's network
policy only as part of activation. Never broaden it to arbitrary provider URLs.
The runtime rejects redirects, never discovers keys from unrelated CLI/browser
credentials, never retries or silently selects another provider, and exposes no
credential fields to the SPFx client. The original local-only endpoint guard is
retained for local inference and embeddings.

Selection and fit receipts are optimistic-revision Oracle catalog records. The
generic catalog write endpoint cannot forge reserved `rdf-*` records. Tests expire
after 24 hours. Source-bound proposals include requested and returned model IDs,
provider, selection revision and bounded token usage. A model change does not
reuse another model's proposal. A read-only source and a review decision remain
separate from publication or activation of source rules.

The embedding model remains `all-minilm` with 384-dimensional vectors; choosing
a language model never changes an existing index. Reranking remains explicitly
unconfigured. The original upstream agent/SQL pipeline is not activated by this
picker. Llama attribution is shown on its model card and while it is selected.

## Checks

```sh
uv run pytest auto_ontology/oracle_mvp/tests/test_model_selection.py \
  auto_ontology/oracle_mvp/tests/test_boundaries.py \
  auto_ontology/oracle_rdf/tests -q
```

Unit tests use synthetic data and a fake endpoint. A production acceptance must
separately verify the real catalog, permitted model and cost policy, fit receipt,
actual model provenance, unchanged source and Microsoft 365 host behavior.
