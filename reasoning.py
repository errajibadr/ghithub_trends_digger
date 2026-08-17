docs/consuming/knowledge-agent.md
----
# Building a Knowledge Agent
s
!!! info "Audience — the self-hosted builder track"
    This guide is for engineers who **build and run their own `knowledge_agent`
    (KA) graph**: you wire your own retriever backends (a LightRAG HTTP server
    and/or a deployed `elastic_rag` gateway), assemble the graph with the
    factory, and either invoke it in-process or deploy it to a LangGraph Server
    you operate.

    If instead you only want to **call a KA graph we already host** — URL + API
    key, no infrastructure — skip to [Calling a hosted KA graph](#calling-a-hosted-ka-graph)
    at the end.

!!! danger "Not yet production — breaking changes may land without notice"
    `knowledge_agent` is actively evolving. The graph topology, state schema,
    output fields, and context keys documented here **may change between
    releases without a deprecation window**. Pin the version you tested against.

## TL;DR

A knowledge agent plans sub-queries, retrieves evidence in parallel from one or
more retrievers, compresses it into cited `Finding`s, optionally reviews
coverage and loops, and optionally synthesizes a cited answer. You build it in
three steps:

```python
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_catalog import (
    get_knowledge_agent_graph,
)

graph = get_knowledge_agent_graph(
    retrievers=[
        {"type": "lightrag",          "name": "kg",   "description": "Entity/relationship knowledge graph"},
        {"type": "elastic_rag_proxy", "name": "docs", "description": "Documentation via the elastic_rag gateway"},
    ],
    mode="answer",
)

result = await graph.ainvoke({"query": "How do I rotate RDS credentials?"})
print(result["result"].answer)
```

Everything below explains the building blocks, the two retrievers in depth, how
to configure the graph, and how to deploy it.

!!! tip "New here? Start with no backend"
    The snippet above needs retriever backends you have to stand up. To run a KA
    **with no infrastructure** (you still need an LLM key — see Install), jump to
    [Try it now — no retriever backend](#try-it-now--no-retriever-backend).

    Quick glossary: **LightRAG** = a knowledge-graph retrieval server (entities +
    relationships); **elastic_rag gateway** = a managed hybrid-search HTTP service
    you deploy ([`elastic_rag`](elastic-rag.md)); **Finding** = a compressed, cited
    unit of evidence; **RRF** = the rank-fusion step that merges keyword + vector
    hits.

!!! note "Output is an object in-process, a dict over HTTP"
    Building and calling the graph **in-process** (this guide) returns
    dataclasses — use attribute access: `result["result"].answer`. Calling a graph
    **over HTTP** (the hosted path) returns JSON — use item access:
    `result["result"]["answer"]`. The output blocks later show the *field layout*,
    not a literal dict.

## What you'll build

```
                       ┌──────────────────────────── knowledge_agent graph ─────────────────────────────┐
                       │                                                                                 │
   question ──────────►│  plan ──► retrieve ──► collect ──► [expand?] ──► compress ──► review? ──┐       │
   {"query": "..."}    │  (LLM    (parallel    (dedup/      (pull more    (chunks →   (coverage  │       │
                       │   picks   retriever    rerank)      doc context)  Findings    sufficient?)│      │
                       │   tools)  calls)                                  + Citations)│           │      │
                       │     ▲                                                         │ no        │      │
                       │     └──────────────────── loop (max_iterations) ─────────────┘           │      │
                       │                                                          yes │            │      │
                       │                          mode="evidence" ◄─────────────── ───┤            │      │
                       │                          mode="answer"   ──► synthesize ──► [review answer?]     │
                       │                                            (cited answer)                        │
                       └───────────────────────────────────────────────────┬─────────────────────────────┘
                                                                            ▼
                                                       result: Findings (evidence) | Answer (answer)
```

It's a **single compiled graph, runtime-gated**: the same topology serves every
`mode` × `search_depth` combination. You opt *up* from the cheap default
(`mode="evidence"`, `search_depth="fast"`) at call time — you do not build a
different graph per behavior.

The retrievers are the part **you supply**. KA is retriever-agnostic: it talks
to anything implementing the `BaseRetriever` protocol from `sta_agent_core`.
This guide covers the two most common backends a builder wires:

| Retriever | `type` | Backend you run | Best for |
|---|---|---|---|
| **LightRAG HTTP** | `"lightrag"` | A LightRAG HTTP server | Entity/relationship queries, graph-structured data |
| **Elastic RAG proxy** | `"elastic_rag_proxy"` | A deployed [`elastic_rag`](elastic-rag.md) LangGraph gateway | Documentation, runbooks, general text corpora |

(Two more types exist — `"elastic"` for a direct Elasticsearch client, and
`"mock"` for tests — but this guide focuses on the two above.)

## Install

```bash
uv pip install sta-agent-engine
```

`sta-agent-engine` pulls in `sta-agent-core` (the retriever adapters) as a
dependency — you do **not** install it separately. No `langgraph-sdk` is needed
to build and invoke a KA graph in-process; it's only relevant when *calling* a
remote graph over HTTP.

!!! warning "Before you start: you need an LLM provider"
    KA's planner, compressor, and synthesizer make **real LLM calls** — even with
    mock retrievers. Building a graph is not enough; running it requires a model
    the process can reach. Set these in a `.env` file in your working directory
    (the factory and retriever settings read `.env` from the current directory via
    pydantic-settings — nothing auto-loads it for you otherwise):

    ```bash
    LLM_PROVIDER=mistral            # or openai, anthropic, ...
    MISTRAL_API_KEY=sk-...          # the matching <PROVIDER>_API_KEY
    ```

    If you run a script with `uv run python my_script.py`, `.env` in the cwd is
    picked up by the settings layer. There is no implicit default provider — an
    unset `LLM_PROVIDER` (and no per-task `KA_*` model) means the graph fails at
    the first node that calls a model.

Then bring up the retriever backend(s) you intend to wire — these are external
prerequisites, not Python packages:

- **LightRAG HTTP** → a running LightRAG server (you get a `base_url`, optionally
  an API key). LightRAG is third-party software; see its
  [project README](https://github.com/HKUDS/LightRAG) for how to run a server,
  then point `RETRIEVER_LIGHTRAG_BASE_URL` at it.
- **Elastic RAG proxy** → a deployed `elastic_rag` gateway (you get a
  `gateway_url` and an API key). See [`elastic_rag`](elastic-rag.md) for what that
  gateway is and how to stand one up.

If you don't have either yet, you can still run the agent end-to-end against
**mock retrievers** — see the next section.

## Try it now — no retriever backend

Before wiring real backends, confirm your install and LLM key work end-to-end
using **mock retrievers**. The `"mock"` retriever type returns canned chunks, so
you need no LightRAG server and no gateway — but you **do** need an LLM provider
configured (see [Before you start](#install)), because the planner and the rest
of the pipeline still call a model.

```python
# save as try_ka.py, then: uv run python try_ka.py   (with .env in this directory)
import asyncio
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_catalog import (
    get_knowledge_agent_graph,
)


async def main() -> None:
    graph = get_knowledge_agent_graph(
        name="KA_Mock",
        retrievers=[
            {"type": "mock", "name": "mock_runbooks",
             "description": "Search runbook procedures for incident response and operations"},
            {"type": "mock", "name": "mock_architecture",
             "description": "Query architecture documentation and design patterns"},
        ],
        mode="evidence",          # cheapest path — raw findings, no synthesis
        search_depth="fast",
    )

    result = await graph.ainvoke({"query": "How do I restart the billing service?"})

    findings = result["result"]   # KnowledgeAgentFindings (a dataclass — use attribute access)
    print("query:", findings.query)
    for f in findings.findings:
        print("-", f.summary)


asyncio.run(main())
```

If this prints findings, your install + LLM wiring are good and you can move on
to real retrievers. If it fails at the **`plan_queries`** step with a connection
or auth error, that's your **LLM provider**, not the retriever — fix `.env`
first.

## The three-layer mental model

Building a KA is always the same three layers, bottom to top:

```
  Layer 3   GRAPH FACTORY        create_knowledge_agent(entries, config=...)
            ▲                     get_knowledge_agent_graph(retrievers=[specs])
            │
  Layer 2   RETRIEVER ENTRIES    RetrieverEntry(name, description, retriever, ...)
            ▲                     one per tool — `description` is what the planner reads
            │                     built from a typed builder OR a JSON spec dict
            │
  Layer 1   DATA / RETRIEVER     LightRAGRetriever | ElasticRagRetriever | ...
                                 (from sta_agent_core — speaks to your backend)
```

- **Layer 1** is a `BaseRetriever` that knows how to reach your backend.
- **Layer 2** wraps each retriever in a `RetrieverEntry` — a `name`, a
  `description` (**critical**: the planner LLM selects tools by description), the
  retriever instance, and optional scope/compression config. One entry = one
  tool the planner can call.
- **Layer 3** is the factory that compiles the entries into a graph.

You rarely touch Layer 1 directly — the **entry builders** (Layer 2) construct
it for you from a `base_url` / `gateway_url` and env vars.

## Choosing a factory

There are two factories. They produce the **same** kind of graph; they differ
in how you describe the retrievers and whether env model-overrides load
automatically.

| | `get_knowledge_agent_graph` | `create_knowledge_agent` |
|---|---|---|
| **Import** | `...knowledge_agent.knowledge_agent_catalog` | `...knowledge_agent` (public API) |
| **Retrievers passed as** | JSON-serializable **spec dicts** (`{"type", "name", "description", "config"}`) | typed **`RetrieverEntry`** objects (from builders) |
| **`KA_*` env model overrides** | **on by default** (`load_env_model_overrides=True`) | opt-in — call `KnowledgeAgentConfig.from_env()` yourself |
| **Build-time knobs** | flat kwargs (`mode`, `search_depth`, `compression_strategy`, …) | a `KnowledgeAgentConfig` object |
| **Best for** | server deployments, `graphs.jsonl`, JSON/UI-driven config | library code, tests, full typed control |

Both are shown side by side throughout this guide. **Rule of thumb:** reach for
`get_knowledge_agent_graph` when your retriever set comes from config/JSON or
you're deploying to a server; reach for `create_knowledge_agent` when you're
writing Python and want typed objects and explicit config.

Signatures:

```python
# Spec-driven (catalog) — knowledge_agent_catalog.py
def get_knowledge_agent_graph(
    retrievers: list[RetrieverSpec | dict] | None = None,
    *,
    name: str = "KnowledgeAgent",
    mode: str | None = None,                 # "evidence" | "answer"
    search_depth: str | None = None,         # "fast" | "deep" | "thorough"
    auto_pull_document: bool = False,
    expand_enabled: bool | None = None,
    max_expansion_rounds: int | None = None,
    max_iterations: int | None = None,
    compression_strategy: str | None = None, # "llm" | "passthrough" | "dynamic"
    model: str | BaseChatModel | None = None,
    checkpointer: Checkpointer | None = None,
    load_env_model_overrides: bool = True,
    env_overrides_prefix: str = "KA_",
    **kwargs,
) -> CompiledStateGraph: ...

# Typed (library) — knowledge_agent_graph.py
def create_knowledge_agent(
    entries: list[RetrieverEntry],
    *,
    name: str = "KnowledgeAgent",
    model: str | BaseChatModel | None = None,
    config: KnowledgeAgentConfig | None = None,
    checkpointer: Checkpointer | None = None,
    **compile_kwargs,
) -> CompiledStateGraph: ...
```

## Wiring retriever A — LightRAG HTTP

**Prerequisite:** a running LightRAG HTTP server. You need its `base_url`
(e.g. `http://localhost:9621`) and, if it's protected, an API key or
username/password.

### Spec form vs. typed form

Both forms below build the same Layer-1 `LightRAGRetriever` — pick one.

**Spec form** (with `get_knowledge_agent_graph`):

```python
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_catalog import (
    get_knowledge_agent_graph,
)

spec = {
    "type": "lightrag",
    "name": "kg",
    "description": "Query the knowledge graph for entities, relationships, and contextual chunks",
    "config": {
        "engine": "http",
        "base_url": "http://localhost:9621",
    },
}

graph = get_knowledge_agent_graph(retrievers=[spec])
```

**Typed form** (with `create_knowledge_agent`):

```python
from sta_agent_engine.agents.knowledge_agent import (
    create_lightrag_entry,
    create_knowledge_agent,
)

entry = create_lightrag_entry(
    name="kg",
    description="Query the knowledge graph for entities, relationships, and contextual chunks",
    engine="http",
    base_url="http://localhost:9621",
)

graph = create_knowledge_agent([entry])
```

### `config` keys (spec) / kwargs (typed)

| Key | Type | Default | Notes |
|---|---|---|---|
| `engine` | `str` | `"http"` | `"http"` (this section) or `"core"` (in-process LightRAG, separate setup). |
| `base_url` | `str` | — | HTTP server URL. Overrides `RETRIEVER_LIGHTRAG_BASE_URL`. **Required** (env or here) — there is no implicit default. |
| `use_twin_api` | `bool` | `false` | Use `/api/query/data` instead of `/query/data`. HTTP engine only. |
| `default_scope` | `LightRAGMetadataScope \| dict` | `None` | Build-time opaque `tag_filter`; a non-empty caller tag filter replaces it for that request. |
| `accepts_caller_scope` | `bool` | `true` | Allow caller-seeded `ka_metadata_scope.tag_filter` to constrain this entry. |
| `examples` | `list[str]` | `[]` | Optional few-shot examples injected into the planner prompt. |

`core_env_file` / `workspace` apply only to `engine="core"` and are ignored for HTTP.

!!! note "LightRAG metadata scope"
    `LightRAGRetriever` accepts `LightRAGMetadataScope`, whose `tag_filter`
    operator keys are forwarded without interpretation. It does not accept
    Elasticsearch axes such as `doc_ids` or `apcode`, and does not expose
    planner-generated metadata arguments.

### Environment variables

Read from the `RETRIEVER_LIGHTRAG_` prefix. **`auth_mode` is computed, not
configured** — set a key/password and the matching mode is selected
automatically:

| Variable | Purpose | Selects auth mode |
|---|---|---|
| `RETRIEVER_LIGHTRAG_BASE_URL` | Server URL (when not passed as `base_url`). | — |
| `RETRIEVER_LIGHTRAG_API_KEY` | Static bearer token. | **static** |
| `RETRIEVER_LIGHTRAG_AUTH_HEADER_FORMAT` | Header style for the static token (default `bearer`). | static |
| `RETRIEVER_LIGHTRAG_AUTH_USERNAME` | JWT login username. | **jwt** |
| `RETRIEVER_LIGHTRAG_AUTH_PASSWORD` | JWT login password. | **jwt** |
| `RETRIEVER_LIGHTRAG_AUTH_LOGIN_URL` | JWT login endpoint (defaults to `{base_url}/login`). | jwt |
| `RETRIEVER_LIGHTRAG_AUTH_TOKEN_TTL` | JWT token TTL (seconds). | jwt |

Resolution: `api_key` set → **static**; else `auth_password` set → **jwt**;
else → **none** (no auth header). For a local unauthenticated server, set only
`RETRIEVER_LIGHTRAG_BASE_URL`.

```
  knowledge_agent ──(lightrag entry)──► LightRAGRetriever ──HTTP──► your LightRAG server
                                          │                          (KG + chunks + entities
                                          └─ auth: none|static|jwt    + relationships)
```

### What you get back

LightRAG chunks carry knowledge-graph context (entities, relationships)
alongside text. KA wires **KG-aware compression** for LightRAG entries
automatically (`ChunkCompressor` + `KGCompressor`), so entity/relationship
structure survives into the `Finding`s. For the raw retriever response shape,
see [`LightRAGRetriever`](lightrag-http.md#output-shape).

## Wiring retriever B — Elastic RAG proxy

**Prerequisite:** a deployed [`elastic_rag`](elastic-rag.md) LangGraph gateway.
You need its `gateway_url` (HTTPS in production) and an API key. This retriever
**delegates** retrieval to that gateway — KA never holds Elasticsearch
credentials or runs embeddings; the gateway does.

### Spec form vs. typed form

Pick one — both build the same Layer-1 `ElasticRagRetriever`.

**Spec form:**

```python
spec = {
    "type": "elastic_rag_proxy",
    "name": "docs",
    "description": "Search documentation through the elastic_rag gateway",
    "config": {
        "gateway_url": "https://lgp.example.com",
        "api_key_env": "DOCS_GATEWAY_API_KEY",
        "top_k": 10,
        "expose_metadata_args": ["apcode", "app_name", "entity"],
    },
}
```

**Typed form:**

```python
from sta_agent_engine.agents.knowledge_agent import create_elastic_rag_proxy_entry

entry = create_elastic_rag_proxy_entry(
    name="docs",
    description="Search documentation through the elastic_rag gateway",
    gateway_url="https://lgp.example.com",
    api_key_env="DOCS_GATEWAY_API_KEY",
    top_k=10,
    expose_metadata_args=["apcode", "app_name", "entity"],
)
```

### `config` keys (spec) / kwargs (typed)

| Key | Type | Default | Notes |
|---|---|---|---|
| `gateway_url` | `str` | — | Gateway base URL. Overrides `ELASTIC_RAG_PROXY_RETRIEVER_GATEWAY_URL`. **URL only** — never embed credentials; HTTPS enforced in production, private/link-local hosts rejected. |
| `api_key_env` | `str` | — | **Name** of an env var holding this retriever's API key (a pointer, not the secret). Lets one process talk to multiple gateways with different keys. Falls back to `ELASTIC_RAG_PROXY_RETRIEVER_API_KEY`. |
| `top_k` | `int` | `10` | Default result count when the planner doesn't override. |
| `default_scope` | `MetadataScope` / dict | — | Build-time filter ceiling (see below). |
| `expose_metadata_args` | `list[str]` | — | Subset of `["apcode", "app_name", "entity"]` to expose as planner tool args. Boost-only. |
| `examples` | `list[str]` | `[]` | Few-shot examples for the planner. |

!!! warning "Credentials are env-only — raw secret keys are rejected"
    Passing `api_key`, `token`, or `password` directly in `config` raises
    `ValueError` at build time. Graph configs (`graphs.jsonl`) are checked in
    and rendered in admin UIs, so secrets must never live there. Provide the key
    via env, and point at it with `api_key_env` when you need a per-retriever
    name.

### Environment variables

Read from the `ELASTIC_RAG_PROXY_RETRIEVER_` prefix:

| Variable | Default | Notes |
|---|---|---|
| `ELASTIC_RAG_PROXY_RETRIEVER_GATEWAY_URL` | — | Gateway base URL when `gateway_url` not passed. Required one way or the other. |
| `ELASTIC_RAG_PROXY_RETRIEVER_API_KEY` | — | Fallback API key (when `api_key_env` not set / empty). |
| `ELASTIC_RAG_PROXY_RETRIEVER_ASSISTANT_ID` | `elastic_rag` | Remote assistant ID on the gateway. |
| `ELASTIC_RAG_PROXY_RETRIEVER_DEFAULT_TOP_K` | `10` | Default result count. |
| `ELASTIC_RAG_PROXY_RETRIEVER_TIMEOUT_S` | `30.0` | Per-attempt HTTP timeout (worst case ≈ `timeout_s × max_attempts` + backoff). |
| `ELASTIC_RAG_PROXY_RETRIEVER_DISTRIBUTED_TRACING` | `false` | Forward LangSmith trace headers to the gateway (unified trace). |

!!! tip "api_key_env holds a name, not the key — a common mistake"
    `api_key_env` holds the **name of an env var**, not the secret itself. Put
    the real key in your `.env`, then point at it by name:

    ```bash
    # .env
    DOCS_GATEWAY_API_KEY=sk-the-actual-secret
    ```
    ```python
    create_elastic_rag_proxy_entry(..., api_key_env="DOCS_GATEWAY_API_KEY")
    #                                                 ^ the env var NAME, not the value
    ```

    If you omit `api_key_env`, the adapter falls back to
    `ELASTIC_RAG_PROXY_RETRIEVER_API_KEY`. Setting `api_key="sk-..."` directly in a
    spec is **rejected** (see the warning above).

```
  knowledge_agent ──(elastic_rag_proxy entry)──► ElasticRagRetriever
                                                   │  POST /runs/wait (X-Api-Key)
                                                   ▼
                                          elastic_rag gateway ──► managed Elasticsearch
                                          (BM25 + kNN + RRF + rerank)   (chunks)
```

### Scope semantics — and the trust-boundary caveat

- **`default_scope`** is a build-time **filter ceiling** enforced *inside this
  process*: the planner and runtime axes can only widen via boosts, never
  narrow past it.
- **`expose_metadata_args`** opts specific axes (`apcode`, `app_name`,
  `entity`) into the planner's tool schema as **boost** hints — they reorder
  results, they do not admit new documents or widen the ceiling.

!!! danger "`default_scope` is not a server-side trust boundary"
    The ceiling lives in *your* KA process. A caller speaking directly to the
    gateway bypasses it, and the gateway does not verify that the asserted scope
    belongs to the caller (see [`elastic_rag` § Scope enforcement](elastic-rag.md#scope-enforcement--read-this)).
    Treat tenant isolation as a separate backend workstream — do not rely on
    `default_scope` for authorization.

!!! note "API-key forwarding is wiring-ahead"
    The proxy env names above are honored by the adapter, but end-to-end gateway
    API-key forwarding is not fully live in current hosted deployments. Keep the
    env names as reserved wiring until gateway key support lands.

## Combining both retrievers

A KA can hold any number of entries of any type. The planner picks among them
**by `description`**, so write descriptions that disambiguate.

```python
# Spec form
graph = get_knowledge_agent_graph(
    retrievers=[
        {
            "type": "lightrag",
            "name": "kg",
            "description": "Knowledge graph: entities, relationships, ownership, dependencies between services.",
            "config": {"engine": "http", "base_url": "http://localhost:9621"},
        },
        {
            "type": "elastic_rag_proxy",
            "name": "docs",
            "description": "Prose documentation and runbooks: procedures, how-tos, configuration steps.",
            "config": {"gateway_url": "https://lgp.example.com", "api_key_env": "DOCS_GATEWAY_API_KEY"},
        },
    ],
    mode="answer",
    search_depth="deep",
)
```

```python
# Typed form
from sta_agent_engine.agents.knowledge_agent import (
    create_knowledge_agent, create_lightrag_entry, create_elastic_rag_proxy_entry,
    KnowledgeAgentConfig,
)

entries = [
    create_lightrag_entry(
        name="kg",
        description="Knowledge graph: entities, relationships, ownership, dependencies between services.",
        base_url="http://localhost:9621",
    ),
    create_elastic_rag_proxy_entry(
        name="docs",
        description="Prose documentation and runbooks: procedures, how-tos, configuration steps.",
        gateway_url="https://lgp.example.com",
        api_key_env="DOCS_GATEWAY_API_KEY",
    ),
]

graph = create_knowledge_agent(entries, config=KnowledgeAgentConfig(mode="answer", search_depth="deep"))
```

On the first iteration the planner can fan out **parallel** calls — one or more
per retriever — so a multi-topic question hits both the graph and the docs at
once. Token budgets for LightRAG split automatically across parallel calls.

!!! tip "The planner needs a parallel-tool-call model"
    The default planning strategy emits native parallel tool calls. Use a model
    that supports them — **`mistral-small-2603`** is the recommended planner. If
    your planner model can't (e.g. gpt-oss), switch to the `"structured"`
    strategy (see [Planning strategy](#planning-strategy)).

## Verify a retriever in isolation

When the full graph fails, it's hard to tell *which* layer broke — the LLM, a
retriever, or the wiring. Before running the whole agent, smoke-test a retriever
on its own. The builder returns a `RetrieverEntry`; its `.retriever` is the
Layer-1 object you can call directly:

```python
import asyncio
from sta_agent_engine.agents.knowledge_agent import create_lightrag_entry

async def check() -> None:
    entry = create_lightrag_entry(name="kg", base_url="http://localhost:9621")
    resp = await entry.retriever.search(query="billing service dependencies", size=3)
    print(f"{len(resp.results)} chunks")
    for chunk in resp.results:
        print("-", chunk.content[:120], "| score:", chunk.score)
    await entry.retriever.close()   # LightRAGRetriever owns HTTP resources — close it

asyncio.run(check())
```

The same pattern works for the proxy (`create_elastic_rag_proxy_entry(...)`).
If this returns chunks, your backend + credentials are good and any remaining
failure is in the LLM or graph config — not the retriever. For the full search
API and response shape, see [`LightRAGRetriever`](lightrag-http.md) and
[`elastic_rag`](elastic-rag.md).

## Configure

### Build-time config (`KnowledgeAgentConfig`)

With `create_knowledge_agent`, pass a `KnowledgeAgentConfig`. With
`get_knowledge_agent_graph`, the common knobs are flat kwargs that the factory
folds into the config for you.

| Knob | Flat kwarg | Config field | Default | Effect |
|---|---|---|---|---|
| Output mode | `mode` | `mode` | `"evidence"` | `"evidence"` returns findings; `"answer"` synthesizes a cited answer. |
| Search depth | `search_depth` | `search_depth` | `"fast"` | Gates the review steps (see below). |
| Outer loop cap | `max_iterations` | `max_iterations` | `3` | Max plan→…→review iterations. |
| Compression | `compression_strategy` | `default_chunk_compression_strategy` | `"dynamic"` | `"llm"` (always compress), `"passthrough"` (1 chunk → 1 finding), `"dynamic"` (threshold-based). |
| Doc expansion | `auto_pull_document` | `expand.auto_pull_document` | `false` | Pull full docs on first pass (fewer rounds, higher cost). |
| Expansion loop | `expand_enabled` / `max_expansion_rounds` | `expand.*` | enabled, 1 round | Pull targeted doc/chunk windows when review finds gaps. |

Finer knobs live on the sub-configs `PlanConfig`, `CollectConfig`,
`CompressConfig`, `ExpandConfig`, `ReviewConfig`, `SynthesisConfig` — import and
pass them inside `KnowledgeAgentConfig(...)` when you need them.

### Model selection (provider-agnostic)

KA hardcodes **no** provider — you choose one in `.env`. The easiest setup is
**one default model for the whole agent**:

```bash
# .env — easiest path: one model for every KA task
LLM_PROVIDER=mistral
MISTRAL_API_KEY=sk-...                    # the matching <PROVIDER>_API_KEY
KA_DEFAULT_MODEL=mistral-small-2603       # used by every task below
```

Need a different model for one step? Override just that task with
`KA_<TASK>_<KEY>` — anything you don't set falls back to `KA_DEFAULT_MODEL`:

```bash
# A bigger model only for the final answer; everything else stays on KA_DEFAULT_MODEL
KA_SYNTHESIS_PROVIDER=mistral
KA_SYNTHESIS_MODEL=mistral-medium-3-5
KA_PLANNING_MODEL=mistral-small-2603      # planner needs parallel tool calls
KA_REVIEW_TEMPERATURE=0
```

`<KEY>` is one of `PROVIDER`, `MODEL`, `BASE_URL`, `MAX_TOKENS`, `TEMPERATURE`.
KA resolves these **five tasks** independently:

| Task (`KA_<TASK>_*`) | What it does | When it runs |
|---|---|---|
| `PLANNING` | Picks which retrievers to call and with what queries. Needs parallel tool calls — `mistral-small-2603` recommended. | Every run |
| `COMPRESSION` | Turns retrieved chunks into cited `Finding`s. | Every run |
| `REVIEW` | Judges whether coverage is sufficient. | `search_depth` ≥ `deep` |
| `SYNTHESIS` | Writes the final cited answer. | `mode="answer"` |
| `VERIFICATION` | Faithfulness-checks the answer. | `search_depth="thorough"` |

`KA_DEFAULT_*` is the shared fallback for all five. Package defaults use
`max_tokens=8192` for compression and `max_tokens=4096` for the other KA tasks;
`temperature=0.0` applies unless you override it.

**Loading the env vars:** `get_knowledge_agent_graph` reads `KA_*`
**automatically**. With `create_knowledge_agent` you opt in once:

```python
config = KnowledgeAgentConfig.from_env(prefix="KA_")   # folds KA_* env into the config
graph = create_knowledge_agent(entries, config=config)
```

`KA_<TASK>_API_KEY` is reserved but not yet wired — use the provider secret
(`<PROVIDER>_API_KEY`). Env values are *defaults*; a per-call
`context={"model_configs": {...}}` still wins.

??? note "Full model-resolution precedence (advanced)"
    KA merges config layers **per key** — the highest-priority layer that sets a
    given key wins it (so a shared slot can supply `model` while a task slot
    supplies only `max_tokens`). Highest priority first:

    1. Runtime `context["model_configs"][task]`
    2. Runtime `context["model_configs"]["all"]`
    3. Build-time `task_model_defaults[task]` (env `KA_<TASK>_*`)
    4. Runtime `context["model_configs"]["default"]`
    5. Build-time `task_model_defaults["default"]` (env `KA_DEFAULT_MODEL`)
    6. Build-time `model=` constructor instance
    7. Engine-wide `create_chat_model()` fallback (`LLM_PROVIDER` + provider env)

### Planning strategy

`PlanConfig.planning_strategy` (build-time) controls how the planner emits its
retriever calls:

| Strategy | Behavior | Use when |
|---|---|---|
| `"tool_calls"` (default) | Native parallel tool calls bound to the model. No validation round-trip; transient failures retried. | Default — needs a parallel-tool-call-capable model. |
| `"structured"` | Returns a validated structured plan, converted to tool calls. Guarantees N calls regardless of model. | The planner model can't emit concurrent tool calls (e.g. gpt-oss). |

```python
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_config import PlanConfig

config = KnowledgeAgentConfig(plan=PlanConfig(planning_strategy="structured"))
graph = create_knowledge_agent(entries, config=config)
```

## Build & run end-to-end

A complete `create_knowledge_agent` build wiring **a LightRAG retriever + an
Elastic RAG proxy retriever scoped to one business entity**, then invoking it
for a cited answer. Assumes a `.env` with your `LLM_PROVIDER`/`KA_DEFAULT_MODEL`
and `DOCS_GATEWAY_API_KEY` (see [Models via `.env`](#model-selection-provider-agnostic)).

```python
import asyncio

from sta_agent_core import MetadataScope
from sta_agent_engine.agents.knowledge_agent import (
    create_knowledge_agent,
    create_lightrag_entry,
    create_elastic_rag_proxy_entry,
    KnowledgeAgentConfig,
)


async def main() -> None:
    # 1. LightRAG entry (knowledge graph). LightRAG is NOT scope-aware, so no scope here.
    kg = create_lightrag_entry(
        name="kg",
        description="Knowledge graph: entities, relationships, and service dependencies.",
        base_url="http://localhost:9621",
    )

    # 2. Elastic-RAG-proxy entry pinned to a specific scope.
    #    default_scope is a build-time filter CEILING: retrieval can never escape it.
    docs = create_elastic_rag_proxy_entry(
        name="docs",
        description="Documentation and runbooks for the billing platform.",
        gateway_url="https://lgp.example.com",
        api_key_env="DOCS_GATEWAY_API_KEY",        # the NAME of the env var holding the key
        default_scope=MetadataScope(
            entity_filter=["<ENTITY_ID>"],          # only docs tagged with this business entity
            include_entity_childs=True,             # ...and its sub-entities
        ),
        expose_metadata_args=["apcode", "app_name"],  # planner may add boost hints WITHIN the ceiling
    )

    # 3. Build the graph. from_env folds KA_* env (e.g. KA_DEFAULT_MODEL) into the config.
    config = KnowledgeAgentConfig.from_env(
        base=KnowledgeAgentConfig(mode="answer", search_depth="deep"),
        prefix="KA_",
    )
    graph = create_knowledge_agent([kg, docs], config=config)

    # 4. Invoke and read the cited answer.
    result = await graph.ainvoke(
        {"query": "How do I rotate RDS credentials for the billing service?"},
        context={"mode": "answer", "search_depth": "deep"},
    )
    answer = result["result"]                       # KnowledgeAgentAnswer (a dataclass)
    print(answer.answer)
    for c in answer.answer_citations:
        print("-", c.title, c.url)


asyncio.run(main())
```

What the scope does here: the `docs` retriever will **only** return chunks
tagged with `<ENTITY_ID>` (or its sub-entities). The planner can add `apcode` /
`app_name` *boosts* to reorder within that set, but cannot widen past it. The
`kg` retriever is unscoped because LightRAG doesn't support metadata scope
(see [LightRAG is not metadata-scope-aware](#wiring-retriever-a--lightrag-http)).

!!! note "Spec-form equivalent"
    Prefer JSON specs / `get_knowledge_agent_graph`? The same scope is a plain
    dict — `"config": {"gateway_url": ..., "api_key_env": "DOCS_GATEWAY_API_KEY",
    "default_scope": {"entity_filter": ["<ENTITY_ID>"], "include_entity_childs": true},
    "expose_metadata_args": ["apcode", "app_name"]}` — and `KA_*` env loads
    automatically (no `from_env` call needed).

A runnable, mode-switchable version (mock / elastic / lightrag / multi) lives at
`examples/sta_agent_engine/knowledge_agent/knowledge_agent_example.py` — start
there with `USER_MODE="mock"` (no infra) to confirm the wiring before pointing
at real backends.

## Consume the result

### Input

Send **either** `messages` **or** `query` on `input`:

| Field | Type | Notes |
|---|---|---|
| `messages` | `list[{"role", "content"}]` | Query extracted from the last human message. |
| `query` | `str` | Direct string — use when you already have the question. |

!!! tip "Send a well-formed retrieval query"
    The planner splits a multi-topic question into focused per-topic searches,
    but it does **not** reformulate or semantically expand your wording — it
    won't rephrase a vague query, invent synonyms, or infer entities you didn't
    name. Retrieval quality is bounded by your input. Name the entities
    involved: *"rotate RDS credentials for the billing service"* beats *"rotate
    credentials"*.

### Runtime context (per call)

Pass on `context=` (or seed in input state). All override build-time defaults:

!!! note "Two call styles — context= in-process, configurable over HTTP"
    For an **in-process** compiled graph, pass these keys via
    `graph.ainvoke(input, context={...})` (LangGraph's context channel). When
    calling a **remote** graph with `RemoteGraph`, the same keys go under
    `config={"configurable": {...}}` instead (see [Calling a hosted KA graph](#calling-a-hosted-ka-graph)).
    Same keys, two call styles — pick the one matching how you invoke the graph.

| Field | Type | Default | Notes |
|---|---|---|---|
| `mode` | `"evidence" \| "answer"` | build config | Evidence findings vs. synthesized answer. |
| `search_depth` | `"fast" \| "deep" \| "thorough"` | build config | Review gating (below). |
| `max_iterations` | `int` | build config | Outer query-loop cap. |
| `auto_pull_document` | `bool` | build config | Pull full docs before compression. |
| `model_configs` | `dict[str, ModelConfig]` | build config | Per-task model override (highest priority). |
| `streaming_enabled` | `bool` | `true` | `false` suppresses custom stream tokens (useful when nesting KA as a tool). |

### `search_depth`

| Depth | Review evidence? | Review answer? | When |
|---|---|---|---|
| `"fast"` | no | no | Low-stakes; iterate quickly. |
| `"deep"` | yes | no | Default for production evidence/answer. |
| `"thorough"` | yes | yes | Critical answers; higher cost & latency. |

### Output — `mode="evidence"` (default)

The blocks below show the **field layout** for reference. In-process,
`result["result"]` is a `KnowledgeAgentFindings`/`KnowledgeAgentAnswer`
**dataclass** — read fields with attribute access (`result["result"].query`),
not item access. Over HTTP the same data arrives as JSON dicts.

```python
{
  "result": {                                  # KnowledgeAgentFindings (dataclass in-process)
    "query": "How do I rotate RDS credentials?",
    "findings": [
      {
        "topic": "RDS credential rotation procedure",
        "summary": "Credentials rotate via AWS Secrets Manager on a 30-day cadence...",
        "key_facts": [
          {"fact": "Rotation interval defaults to 30 days",
           "citation": {"title": "RDS runbook", "url": "https://confluence.internal/pages/PG-10294",
                        "source_type": "confluence", "snippet": "...",
                        "retriever_name": "docs", "metadata": {"pageId": "PG-10294", "chunk_index": 3}}}
        ],
        "confidence": "high",                  # "high" | "medium" | "low"
        "compression_mode": "llm"              # "llm" | "passthrough" | "kg"
      }
    ],
    "coverage": {"sufficient": true, "gaps": [], "reasoning": "...",
                 "query_suggestions": [], "fetch_targets": []},  # null when search_depth="fast"
    "retriever_names": ["docs", "kg"],
    "iteration_count": 1,
    "metadata": {}
  },
  "findings": [ /* same list, flat */ ],
  "coverage": { /* same */ },
  "collected_chunks": [ /* raw RetrievalChunk list — same shape as elastic_rag results */ ]
}
```

### Output — `mode="answer"`

```python
{
  "result": {                                  # KnowledgeAgentAnswer
    "evidence": { /* full KnowledgeAgentFindings */ },
    "answer": "To rotate RDS credentials, use AWS Secrets Manager... [1][2]\n\nSources:\n[1] [RDS runbook](https://...)",
    "answer_citations": [ /* ONLY the cites the answer used, in [1]/[2] order */ ],
    "answer_review": { "faithful": true, "explanation": "...", "unsupported_claims": [] }  # only when search_depth="thorough"
  },
  "findings": [ ... ], "coverage": { ... }, "collected_chunks": [ ... ]
}
```

The answer is **evidence-bound** — built only from retrieved findings, never the
model's own knowledge. Concrete values/identifiers/commands are preserved
verbatim; gaps and conflicts are surfaced explicitly ("the evidence does not
cover X") rather than smoothed over. Treat that as the intended signal — raise
`search_depth` or refine the query rather than expecting confidence over thin
evidence.

Inline `[N]` markers are rewritten to markdown links and a plain `Sources:`
block is appended, so a consumer reading only `result.answer` (or the last
`messages` entry in answer mode) gets clickable citations in any markdown
renderer.

## Deploy to a LangGraph Server

To serve your graph over HTTP, register it in `langgraph.json` and run the
server. Two registration shapes — pick by how the retriever set is decided.

### A. Lazy `@cache` getter (fixed retriever set)

Write a zero-arg getter that builds the graph on first call (keeps import cheap
— no network at import time, recoverable credential failures), and reference it
`module:function`:

```python
# my_pkg/ka_catalog.py
from functools import cache
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_catalog import (
    get_knowledge_agent_graph,
)

@cache
def get_my_ka_instance():
    return get_knowledge_agent_graph(
        retrievers=[
            {"type": "lightrag", "name": "kg",
             "description": "Knowledge graph: entities and relationships"},
            {"type": "elastic_rag_proxy", "name": "docs",
             "description": "Documentation via the elastic_rag gateway",
             "config": {"api_key_env": "DOCS_GATEWAY_API_KEY"}},
        ],
        name="MyKA", mode="answer",
    )
```

```json
{
  "graphs": { "my_ka": "my_pkg.ka_catalog:get_my_ka_instance" },
  "env": ".env"
}
```

!!! warning "Never re-export the factory from `__init__.py`"
    `get_knowledge_agent_graph` and the lazy getters construct retrievers
    (network). Reference them by module path in `langgraph.json` only — importing
    the package must not open connections.

Config (`base_url`, `gateway_url`, API keys, `KA_*` model overrides) comes from
the server's `.env` at first request. The `KA_*` model env loads automatically.

### B. JSON factory bridge (`graphs.jsonl`-style, UI-driven)

When a catalog/UI drives the retriever set, point a factory entry directly at
`get_knowledge_agent_graph` and pass `factory_args`:

```json
{"id": "ka_multi", "name": "KA Multi", "type": "factory",
 "module_path": "sta_agent_engine.agents.knowledge_agent.knowledge_agent_catalog",
 "factory_function": "get_knowledge_agent_graph",
 "default_context": {"mode": "answer", "search_depth": "fast"},
 "factory_args": {"mode": "answer", "name": "KA_Multi",
   "retrievers": [
     {"type": "lightrag", "name": "kg", "description": "Knowledge graph"},
     {"type": "elastic_rag_proxy", "name": "docs", "description": "Docs via gateway",
      "config": {"api_key_env": "DOCS_GATEWAY_API_KEY"}}
   ]}}
```

Because `factory_args` is checked in and may render in an admin UI, secrets
**must not** appear there — that's why the proxy takes `api_key_env` (a name),
not `api_key` (a value).

### Server-side concerns

- **Auth / headers** — configure `auth` and `http.configurable_headers` in
  `langgraph.json` to forward caller identity. See
  `.claude/skills/langgraph-agent-builder/references/server-runtime-quickref.md`.
- **Same-topology rule** — a factory must return the **same** nodes/edges for
  every caller; vary *tools and prompts* via `retrievers`, never graph
  structure.
- **Distributed tracing** — set
  `ELASTIC_RAG_PROXY_RETRIEVER_DISTRIBUTED_TRACING=true` to unify the KA trace
  with the gateway's. See
  `.claude/skills/langgraph-agent-builder/references/distributed-tracing.md`.

## Errors & known limitations

| Condition | What happens |
|---|---|
| Neither `messages` nor `query` sent | `ValidationError`. |
| `base_url` unset for LightRAG (no env, no arg) | Settings validation raises — `base_url` is required. |
| `gateway_url` unset for the proxy | Raises — set env or `gateway_url`. |
| Raw `api_key`/`token`/`password` in proxy `config` | `ValueError` at build — use `api_key_env` + env. |
| Planner model can't emit parallel tool calls | Fewer calls per turn — switch to `planning_strategy="structured"`. |
| Retrieval backend unavailable | 5xx; retry is safe but may re-run the pipeline (KA is not stateless like `elastic_rag`). |
| Answer mode, insufficient evidence | You still get a result; `coverage.sufficient=false` and the answer disclaims the gap. |
| Unknown context key | Silently ignored. |

Roadmap caveats: gateway API-key forwarding is wired but not end-to-end yet;
`default_scope` is an in-process ceiling, **not** server-enforced tenant
isolation; per-task `KA_<TASK>_API_KEY` is reserved but inert.

## Calling a hosted KA graph

If you don't build your own and instead **call a KA graph we host**, you need
only a URL and an API key — no retrievers, no factory, no infrastructure.
Published variants differ only by retrieval backend:

| Graph name | Backend |
|---|---|
| `knowledge_elastic` | Managed Elasticsearch (hybrid BM25 + kNN) |
| `knowledge_lightrag_http` | Hosted LightRAG knowledge graph |

```python
# Option A — langgraph_sdk
from langgraph_sdk import get_client

client = get_client(url="<KNOWLEDGE_AGENT_LGP_URL>", api_key="<KNOWLEDGE_AGENT_LGP_API_KEY>")
async for chunk in client.runs.stream(
    thread_id=None, assistant_id="knowledge_elastic",
    input={"messages": [{"role": "user", "content": "How do I rotate RDS credentials?"}]},
    context={"mode": "answer", "search_depth": "deep"}, stream_mode="values",
):
    if chunk.event == "values":
        print(chunk.data)

# Option B — RemoteGraph (graph-as-an-object)
from langgraph.pregel.remote import RemoteGraph

graph = RemoteGraph("knowledge_elastic", url="<KNOWLEDGE_AGENT_LGP_URL>", api_key="<KNOWLEDGE_AGENT_LGP_API_KEY>")
result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "How do I rotate RDS credentials?"}]},
    config={"configurable": {"mode": "answer", "search_depth": "deep"}},
)
print(result["result"]["answer"])
```

Install `langgraph-sdk` (HTTP) and/or `langgraph` (`RemoteGraph`). The input,
runtime context, and output schemas are identical to the build-your-own path
documented above. Remote callers do **not** register retrievers per request —
the hosted graph's retriever set is fixed server-side.

## See also

- [`elastic_rag`](elastic-rag.md) — the managed retrieval gateway the
  `elastic_rag_proxy` retriever delegates to (and a standalone way to get raw
  chunks). Stable, stateless.
- [`LightRAGRetriever`](lightrag-http.md) — the raw LightRAG retriever as a
  Python class, for building your own pipeline without KA.
- `examples/sta_agent_engine/knowledge_agent/knowledge_agent_example.py` —
  runnable, mode-switchable build example.
- `.claude/skills/langgraph-agent-builder/SKILL.md` — agent patterns, server
  runtime, auth, deployment references.

-------

docs/consuming/lightrag-http.md
----
# LightRAGRetriever (HTTP) — Library Import

## TL;DR

`LightRAGRetriever` is a **Python class** you import and instantiate in your
own code. It speaks HTTP to our hosted LightRAG server and returns
knowledge-graph–backed chunks alongside entity and relationship context.

Use this when you're building your own LangGraph agent or pipeline and want
the retriever as a building block — not a remote graph. If you want the
retriever wrapped in a ready-to-use pipeline, use
[`knowledge_lightrag_http`](knowledge-agent.md) instead.

```mermaid
flowchart LR
    you([Your Python service]) -- LightRAGRetriever --> http[<LIGHTRAG_HTTP_URL><br/>LightRAG HTTP server]
    http -- KG + chunks --> you
```

!!! note "We run the LightRAG server"
    The HTTP server at `<LIGHTRAG_HTTP_URL>` is operated by our team. You do
    **not** self-host. You receive a URL and an API key — that's it.

    The `from_core(...)` factory (in-process LightRAG) is internal-only and
    not documented here.

## Install

```bash
uv pip install sta-agent-core
```

Or add to your `pyproject.toml`:

```toml
dependencies = ["sta-agent-core"]
```

## Quickstart

### Simplest path — implicit static bearer auth

```python
from sta_agent_core.repositories.retrievers.lightrag import LightRAGRetriever

retriever = LightRAGRetriever(
    base_url="<LIGHTRAG_HTTP_URL>",
    api_key="<LIGHTRAG_API_KEY>",
)

response = await retriever.search(
    query="which services depend on the billing database?",
    size=10,
)

for chunk in response.results:
    print(chunk.content, chunk.score)

# Entity and relationship context — query-level, not per chunk
for entity in response.entities:
    print(entity["entity_name"], entity["entity_type"])

await retriever.close()
```

### Explicit auth provider

Use `from_http` when you want full control of the auth flow — e.g. JWT with
refresh, or custom bearer header formats.

```python
from sta_agent_core.adapters.auth import StaticBearerAuth
from sta_agent_core.repositories.retrievers.lightrag import LightRAGRetriever

retriever = LightRAGRetriever.from_http(
    base_url="<LIGHTRAG_HTTP_URL>",
    auth_provider=StaticBearerAuth("<LIGHTRAG_API_KEY>"),
)
```

### Twin query endpoint

The standard LightRAG query route is `/query/data`. Deployments exposing the
Twin-compatible route at `/api/query/data` can select it explicitly:

```python
retriever = LightRAGRetriever.from_http(
    base_url="<LIGHTRAG_HTTP_URL>",
    auth_provider=StaticBearerAuth("<LIGHTRAG_API_KEY>"),
    use_twin_api=True,
)
```

`use_twin_api=False` remains the default. The selected route is also preserved
when the HTTP engine retries after re-authentication.

### JWT auth (if we issue a username/password instead of a token)

```python
from sta_agent_core.adapters.auth import JWTAuth
from sta_agent_core.repositories.retrievers.lightrag import LightRAGRetriever

retriever = LightRAGRetriever.from_http(
    base_url="<LIGHTRAG_HTTP_URL>",
    auth_provider=JWTAuth(
        login_url="<LIGHTRAG_HTTP_URL>/auth/login",
        username="<LIGHTRAG_USERNAME>",
        password="<LIGHTRAG_PASSWORD>",
        token_ttl=3600,
    ),
)
```

## `.search()` parameters

```python
await retriever.search(
    query: str,
    size: int = 10,
    *,
    mode: str | None = None,
    enable_rerank: bool | None = None,
    hl_keywords: list[str] | None = None,
    ll_keywords: list[str] | None = None,
    # ...plus additional LightRAG-specific kwargs passed through to the engine
)
```

| Param | Type | Notes |
|---|---|---|
| `query` | `str` | Natural-language question. |
| `size` | `int` | Max chunks to return. |
| `mode` | `str` | LightRAG query mode: `"naive"`, `"local"`, `"global"`, `"hybrid"`, `"mix"`. Server-side default applies if omitted. |
| `enable_rerank` | `bool` | Enable server-side reranking. |
| `hl_keywords` | `list[str]` | High-level keywords to bias toward. |
| `ll_keywords` | `list[str]` | Low-level keywords to bias toward. |

## Output shape

`.search()` returns a `LightRAGSearchResponse`:

```python
{
  "results": [                               # list[LightRAGRetrievalChunk]
    {
      "content": "The billing service reads from the billing_prod Postgres...",
      "chunk_id": "lr-chunk-9f3a21",
      "score": 0.82,
      "source_url": "",                      # often empty for KG chunks
      "retriever_type": "lightrag",
      "metadata": {
        "source": "lightrag",
        /* plus whatever LightRAG returned for this chunk */
      },
      "reference_id": "kb-file-042"          # LightRAG file-level ref for entity linking
    }
  ],

  "entities": [                              # list[LightRAGEntity] — query-level context
    {
      "entity_name": "billing-svc",
      "entity_type": "service",
      "description": "Handles invoice generation and payment capture.",
      "source_id": "kb-file-042",
      "file_path": "...",
      "reference_id": "kb-file-042"
    }
  ],

  "relationships": [                         # list[LightRAGRelationship]
    {
      "src_id": "billing-svc",
      "tgt_id": "billing_prod",
      "description": "reads-from",
      /* ...additional LightRAG fields */
    }
  ],

  "references": [                            # list[LightRAGReference] — file-level sources
    { "reference_id": "kb-file-042", "file_path": "...", "title": "..." }
  ],

  "metadata": {                              # LightRAGQueryMetadata — diagnostics
    "query_mode": "hybrid",
    "hl_keywords": ["billing", "database"],
    "ll_keywords": [],
    "total_entities_found": 12,
    "entities_after_truncation": 8,
    "total_relations_found": 20,
    "relations_after_truncation": 15,
    "final_chunks_count": 10,
    "tags": {"all": ["billing"]}
  }
}
```

Access patterns:

```python
response = await retriever.search(...)

# Generic — iterate chunks like any RetrievalChunk source
for chunk in response.results:
    use(chunk.content, chunk.score, chunk.metadata)

# KG-aware — reach for query-level graph context
for entity in response.entities:
    inspect_entity(entity)
for rel in response.relationships:
    inspect_edge(rel)

# Response-level metadata for observability
print(response.metadata.query_mode, response.metadata.final_chunks_count)
```

### Chunk shape (`LightRAGRetrievalChunk`)

Inherits all fields from the base `RetrievalChunk`
(`content`, `chunk_id`, `score`, `source_url`, `retriever_type`, `metadata`)
and adds:

| Field | Type | Notes |
|---|---|---|
| `reference_id` | `str` | LightRAG's file-level reference. Ties the chunk back to the `references[]` entry. |

The inherited `score` field uses the server's `relevance_score` when present,
then falls back to `score`; it remains `None` when neither field is returned.

Consumers that only know the base protocol can still iterate `results` and
use `content` + `metadata` — `LightRAGRetrievalChunk` is Liskov-compatible.

## Lifecycle

`LightRAGRetriever` owns HTTP resources. Always close it:

```python
try:
    response = await retriever.search(...)
finally:
    await retriever.close()
```

Or use it inside a context-managed helper in your own pipeline. For
LangGraph consumers, tying `close()` to graph shutdown is the right seam.

## Errors

| Exception | Cause | Typical action |
|---|---|---|
| `RetrieverConnectionError` | Transport failure — DNS, TLS, refused. | Retry with backoff; escalate if persistent. |
| `RetrieverResponseError` | Server returned an unparseable or error response. | Inspect `.response`; most often a token / permissions issue. |
| Auth failures | Expired token, missing scope. | Static bearer: ask for a new key. JWT: the provider refreshes automatically on 401. |


## See also

- [`knowledge_lightrag_http`](knowledge-agent.md) — if you want a ready-made
  LangGraph pipeline over the same server (plan → retrieve → compress →
  synthesize) instead of the raw retriever.
- [`elastic_rag`](elastic-rag.md) — if your corpus is text documents rather
  than a knowledge graph.

-------

memory_bank/MEMORY.md
----
# Memory Bank Index

All creative-phase docs live under `creative_phases/`. Topic folders group related work; `_archive/` is the read-only history.

## Topic folders

- [creative_phases/evaluation/](creative_phases/evaluation/creative_phase_step_2_2026-05-27_implementation_log.md) — Agent evaluation skill suite: `agent-eval-designer` (framework-agnostic methodology) + `sta-eval-implementer` (in-house framework binding). **IMPLEMENTED 2026-05-27** on branch `feat/agent-evaluation` — both skills shipped (22 files, ~4500 lines, all budget-compliant; mechanical validation green: zero framework leakage in Skill A body, all cross-links resolve, Python/JSON templates valid, in-house symbol citations resolve in `evals/base/`). End-goal = use both skills to design + ship the orchestrator deep-agent eval suite (task pending).
- [creative_phases/orchestrator/](creative_phases/orchestrator/README.md) — twin_router → orchestrator deep-agent rewrite on the `deepagents` package (#13/#14). v1 design + 6-PR plan; PLANNED.
- [Orchestrator improvements master plan (6 topics)](creative_phases/orchestrator/creative_phase_2026-06-02_improvements_master_plan.md) — **PLANNED 2026-06-02** (branch `worktree-orchestrator-improvements`). 7-agent planning workflow over 6 topics: (T1) KA↔orchestrator ephemeral structured data sharing — `task` propagates via shared STATE keys only (not context), `EphemeralValue` for per-run reset; apcode is boost-only so "filter to exactly those docs" needs new KA wiring (deferred). (T2) incident decouple — seam fix is a 1-line bug (`build_incident_subagent.py:102`) fixing [ORCH-01]; package delete is consumer-visible (gated on Badr). (T3) recursion graceful-degrade — `RecursionLimitHandlerMiddleware` catching only `GraphRecursionError` at parent. (T4) reject unified factory (only navigator forwards middleware), ship tiny `_subagent_common.py` helper. (T5) prompt-guard: multimodal judge (reuse `strip_images_for_model`), ⚠️ refusal formatting, optional/mistral-small-2603 default (gated). (T6) reliable `read_picture` via layered system-reminder + bounded `tool_choice`. 11-PR sequence in 4 waves; suggested first PR = T2 seam fix. **3 blocking open Qs for Badr** (hard-filter vs boost, incident_agent external consumers, guard deploy-status). **ROUND-2 CORRECTIONS ADDENDUM appended** (post user re-challenge, 2nd 7-agent workflow): T2 **user_correct** (relocate mock → `agents/incident_agent/graph:incident_agent`, delete `twin_router/mock_incident_agent`, repoint twin_router too — collapses to 1 PR; round-1's "don't touch twin_router" inverted); T1 **user_partially_correct** (`MetadataScope.apcode_filter` hard filter + `documents` search param BOTH already exist — `metadata_scope.py:126-131`, `elastic_retriever.py:857`, `elastic_rag_proxy/context.py:40`; PR-11 re-scoped from separate workstream → small wiring; round-1 conflated tool-arg boost layer with retriever layer); T3 **user_correct** (Option B gated to middleware-accepting factories — navigator only); T4 **nuanced** (factory-in-`SubagentEntry` + `supports_middleware` flag + one `build_subagent` dispatcher — NO `_subagent_common.py`); T5 **user_partially_correct** (guard is text-only today `prompt_injection_guard.py:197-205` — AXIS-1 is NEW capability not crash-fix; AXIS-3 regression real but re-described as "screening goes silent"); T6 **nuanced** (eager-describe in `wrap_model_call` via `Overwrite`+sentinel replaces read_picture tool-nudge). Cross-topic thread: T3-B/T4/T1 converge on `SubagentEntry` as the single registration seam. NEW open Qs: elastic_rag gateway server-side `retriever_documents` enforcement (trust boundary), doc_id namespace match (`doc_keyword_field`). **ROUND-3 DECISIONS LOCKED** (with owner): T1 direct `ElasticRetriever` only — drop proxy path, gateway trust-boundary risk RETIRED (`twin_ka_entries.py:11-13`); T4 names locked **`SubagentStateBridge` base + `KnowledgeBridgeMiddleware`**, declared in `SubagentEntry.state_bridge`, shared channels = one `KnowledgeBridgeChannels` TypedDict — orchestrator/react subagents INJECT the middleware (gated by `supports_middleware`), basic graphs (KA) INHERIT the channels TypedDict statically (no runtime schema surgery); **ephemerality unverified — probe `EphemeralValue` cross-super-step survival before building** (may clear between super-steps → filter lost before `task` fires; fallback = `LastValue` + reset node); T5 multimodal judge adds image parts only when `is_multimodal(judge)`; T6 `before_agent` + synthetic `read_picture` pair (idempotency keyed to image, keep tool bound, `wrap` strip as net), drops `tool_choice` force.
- [creative_phases/twin_router/](creative_phases/twin_router/README.md) — Habilitation, habilitation-refactor, auth-evolution. Being replaced by the orchestrator rewrite (#14).
- [creative_phases/elastic_rag/](creative_phases/elastic_rag/README.md) — Eval-sweep planning + pointer to the still-at-parent `retriever_owns_expansion/` thread folder.
- [creative_phases/retriever_owns_expansion/](creative_phases/retriever_owns_expansion/README.md) — Active QMD-inspired expansion+fusion thread (#30 closed; follow-ups under #34/#35/#39).
- [creative_phases/knowledge_agent/](creative_phases/knowledge_agent/AGENTS.md) — Knowledge Agent design history (retriever layer → multi-retriever evidence-gathering agent → review loop → compression → synthesizer → presets).
- [creative_phases/lightrag/](creative_phases/lightrag/) — LightRAG local-engine + topology designs, plus the foundational retriever-protocol doc (`creative_phase_2026-02-09_retriever_and_lightrag_architecture.md`).
- [creative_phases/frontend/](creative_phases/frontend/) — Frontend creative phases (chat customization, dynamic context controls).
- [creative_phases/alfred/](creative_phases/alfred/creative_phase_2026-07-24_hitl_interrupt_contract.md) — ALFRED (AG-UI product) challenge-exploration docs; explorations live on `experiment/alfred-*` branches, creative phases land here. First doc: **HITL & interrupt interface contract** (🎨 PROPOSED 2026-07-24) — one wire schema composing custom `interrupt()` + `HumanInTheLoopMiddleware`: normalized `HITLRequest` shape for all reviewable actions (mirror the TypedDicts, don't import — not in `__all__`), `source`/`schema_version` envelope keys, frontend routing precedence, **id-map-always resume** (verified: works for middleware pauses too), `action_name` decision echo + `apply_decisions` validation closing the positional order-swap hole, streaming/detection rules (values+messages, accumulate, thread-state reconcile), JSON-DTO payload discipline (serde nulls unknowns; payloads outlive deploys). Grounded in `experiments/interrupt_contract/` probes. **AG-UI mapping RESOLVED 2026-07-27**: AG-UI's native interrupt protocol (`RunFinished` interrupt outcome, id-addressed `resume[]`, `responseSchema`, `expiresAt`) converged on the same decisions — adapter rules added (1:1 interrupt mapping incl. batches, `reason` derivation, `emitInterruptOutcome` flag; keep decisions algebra over AG-UI's poorer `approved`/`editedArgs` example).
- [creative_phases/alfred/creative_phase_2026-07-29_post_poc_decision_workshop.md](creative_phases/alfred/creative_phase_2026-07-29_post_poc_decision_workshop.md) — **IN PROGRESS, revised 2026-07-30.** Evidence-backed ALFRED post-POC workshop grounded in the `alfred-docx-comments` baseline, with an initial DOCX-plus-simple-text artifact hypothesis. Selected dual-mode fast/deep UX remains secondary evidence; ONLYOFFICE/PPTX and its concrete Quick Transform design are non-authoritative. Graph-side entitlement intersection is reopened rather than marked “retain.” Includes 24 atomic decisions, including artifact renderer matrix, side-by-side workspace, fast/deep assistance, entitlement enforcement, and effective-team resume semantics. ALF-DEC-001 remains open, the parallel product branch is comparative evidence only, and no target architecture is authoritative until its decision records are accepted. Companion: [standalone interactive decision dossier](creative_phases/alfred/alfred_architecture_decision_dossier.html) — reader-first HTML, deep links, ASCII maps, multiple anchored remarks, JSON handoff, and portable reviewed copies.
- [creative_phases/roadmap/](creative_phases/roadmap/) — Roadmap creative phases organized by pillar (p3, p4 currently populated).

## Loose creative phases

- [creative_phases/creative_phase_2026-06-06_prefix_cache_tuning.md](creative_phases/creative_phase_2026-06-06_prefix_cache_tuning.md) — **PARTIALLY IMPLEMENTED 2026-06-06.** Make the middleware stack prefix-cache friendly (vLLM APC + Anthropic prompt caching). SHIPPED: `TimeAwareMiddleware` reworked — stopped rewriting the system prompt with `datetime.now()` (the Anthropic-documented "timestamp in an early block → no cache hit" anti-pattern); now injects an **immutable `<system_reminder>` message** via `before_agent` (once per run, never mid-loop), deterministic bucket id + **scan-and-skip** so content is frozen and only ever tail-appended. New `InjectionStrategy` (INTERVAL default, `interval_hours=3` / PER_HUMAN_TURN), `role`/`tag` knobs, back-compat ctor (`section_name` alias). 18 tests green, ruff/pyright clean. NOTED FOR FUTURE WORK: full middleware audit table (worst = deprecated `mode_management`/`reflection` mode-cycle prompt+tool rewrites [CI-08]; live concern = `dynamic_tools` order stability); + backlog (document § prefix-cache hygiene in engine AGENTS.md + skill, live-model `SystemMessage`-in-messages placement check, normalizer/patch_tool_calls dedupe, explicit `cache_control` emission, cache-hit measurement). Grounded on Anthropic docs (tools→system→messages cascade) + vLLM block-level hashing.
- [creative_phases/creative_phase_2026-05-28_bundle_fixes_seed.md](creative_phases/creative_phase_2026-05-28_bundle_fixes_seed.md) — **SEED — iterating** (2026-05-28, branch `worktree-bundlefixes`). 35 captured fixes from the "Bundle Sync 27/05" notes across Global/Examples/Core/Frontend/Tests/Engine(KA)/Twin. Each has a stable ID + status; clarify → spec → implement one at a time.
- [creative_phases/creative_phase_2026-03-05_library_logging_hygiene.md](creative_phases/creative_phase_2026-03-05_library_logging_hygiene.md) — APPROVED for implementation, **NOT YET IMPLEMENTED** (verified 2026-05-07 via subagent: design helpers `register_library_logger` / `configure_library_logger` not in `packages/`, `setup_logger.py:22` still has the NullHandler the design called to remove). Remains active until picked up.
- [creative_phases/creative_phase_2026-05-17_dynamic_settings_registry.md](creative_phases/creative_phase_2026-05-17_dynamic_settings_registry.md) — **PLANNED v2.4 — READY FOR IMPLEMENTATION** (Codex-reviewed 2026-05-28; scope-trimmed 2026-05-29 → simplified 2026-05-30). ~40% shipped on `main` (`0ed2198`): dynamic `_env_prefix=` kwarg, open chat registry, convention fallback, OpenAI default-registered, `DeprecationWarning` on silent OpenAI fallback. Remaining on branch `feat/provider-tiers-v2.4` — **one PR + two follow-ups (~3 d):** (Phase 1) capacity tiers on `BaseProviderSettings` (`big`/`small`/`thinking` + `get_model(tier)` cascade) **+ collapse the three lookup structures into ONE `_LLM_REGISTRY`** — `_provider_map`+`_dynamic_registry`+`_dynamic_class_cache` fold in; convention-fallback **lazily self-populates** (no separate synthesis cache); `ProviderType` **retained as a supported input type** (NOT deprecated — backward compat); `ProviderFactory`/`get_provider_settings` become read-through shims; `ProviderSpec` carries `chat_model_factory`+`BuildContext`; built-in registrations move core→engine (vendor-freeness); (Phase 2) `LightRAGCoreSettings.for_instance_from_env` migration; (Phase 3) `_is_mistral_model` substring-dispatch fix (one-release `DeprecationWarning` bridge). **CUT/DEFERRED:** capability/multimodal axis (builder's responsibility, `MultimodalGuardMiddleware` untouched); `reasoning_effort` (vLLM probe 2026-05-29 → thinking on-by-default, passthrough a no-op, only per-family `chat_template_kwargs` moved it; streaming-reasoning gap traced to LiteLLM proxy #20246); **third-party `register_provider` + embedding/rerank registries + `BaseProviderCredentials` split (YAGNI until a real external consumer)**. 10 decisions locked. ASCII architecture map + snippets in doc. Probe: `experiments/langchain/chatopenai/reasoning_script.py`.
- [creative_phases/orchestrator/creative_phase_2026-06-20_sources_block_ownership.md](creative_phases/orchestrator/creative_phase_2026-06-20_sources_block_ownership.md) — **IMPLEMENTED 2026-06-21 (test-first, 3 commits; helpers folded into `knowledge_bridge.py`, NOT a separate util file; D2 = one audit-tagged message; decoder stop removed).** SUPERSEDES the 2026-06-18 decoder-`stop`: stop *suppressing* the planner's `Sources:` block — **ENCOURAGE** it (cite only the `ka_sources` used, as `[N] [title](url)`) and have the orchestrator **OWN** it. Terminal `after_agent`/`aafter_agent` on `KnowledgeBridgeMiddleware`: if the final answer already has a Sources header → **leave** it (D1); else **fake-stream** a canonical block via langchain_core `GenericFakeChatModel` as a separate `additional_kwargs={"sta_generated":"sources_block"}` `AIMessage` (D2). Remove `_PLANNER_STOP_SEQUENCES` + `PlannerModelResolver(stop=...)` → fixes the vLLM+Nemotron `</think>` leak (R1, user-confirmed stop-induced; LiteLLM #20246 streaming-reasoning gap is only the backup explanation). Phase-0 proof: `GenericFakeChatModel.astream` inside the exit node surfaces as `messages`-mode chunks and fires exactly once (`factory.py:1497/1638` + spike); `wrap_model_call` exposes NO token stream (handler returns a full `ModelResponse`) → the streaming-middleware idea was dropped. R4 resolved (after_agent single-fire). Builds on [[creative_phase_2026-06-18_sources_stop_enforcement]] + [[creative_phase_2026-06-16_sources_determinism_and_synthesis_context]]. **§7 post-impl refinements (2026-06-21, commits `d941c0c7`/`7dfc7e7d`/`d31c3ed2`):** header pinned to EXACTLY `**Sources**:` (literal example, stated positively — do NOT enumerate wrong headings, that primes them); block is now **comma-separated single line** `**Sources**: [1] [title](url), [2] ...` (`format_sources_block` joins with `, ` so fallback == planner shape); **KA is the source of truth**, the `<knowledge_sources>` note is only a reminder; announcer reminder rows render as `[N] [title](url)`. Delegation/caller-scope split to [[creative_phase_2026-06-21_delegation_parallelism_and_caller_scope]].
- [creative_phases/orchestrator/creative_phase_2026-06-21_delegation_parallelism_and_caller_scope.md](creative_phases/orchestrator/creative_phase_2026-06-21_delegation_parallelism_and_caller_scope.md) — **Part A + Part B IMPLEMENTED.** **A — orthogonal-asks → parallel delegation (commits `973d6068`, `8508caa8`):** the planner serialized/merged independent questions because the tasking guidance modeled only facets-of-one-topic-across-agents, never multiple independent asks (least of all to the SAME agent). Added the FACETS-vs-ORTHOGONAL split to `_SUBAGENT_TASKING_SECTION` (orthogonal asks → one scoped brief each, emitted in the SAME turn → parallel, even to the same sub-agent; agent-agnostic per `test_subagent_tasking_is_agent_agnostic`), flipped `_DEEPAGENTS_TOOL_GUIDELINES` "in sequence" → "in sequence or in parallel", and on the KA card: spawn one KA task per independent doc question + **do NOT name/hint a corpus in the brief (KA self-selects)**. **B — caller-scope hardening (commit `97287012`, test-first):** the bridge `_render_selection` steer "instruct it to query `general_doc`" existed because the caller-scope hard filter only applies on `accepts_caller_scope=True` entries (only `general_doc` has it) — without the steer the KA could search `twin_project_doc` and silently drop the filter. Fix = structural: `PlanQueriesNode._active_plan_inputs(state)` restricts the bound tool set to scope-accepting entries when **any** caller-scope axis is present (filter by the flag not the name; empty-set/no-entries guard keeps all + warns; never mutates `self._tools` — shared singleton), threaded into the bind, sanitizer, prompt tools block, and structured-output validation; then deleted the prompt steer + dead `general_corpus_name`. Anchor injection inherits the restriction. **Decisions:** trigger on ANY axis (not just `doc_ids`); **straight cut, no deprecation** (Badr confirmed no deployment relies on broad-search-under-scope). Sibling: [[creative_phase_2026-06-20_sources_block_ownership]] §7.
- [creative_phases/orchestrator/creative_phase_2026-06-18_sources_stop_enforcement.md](creative_phases/orchestrator/creative_phase_2026-06-18_sources_stop_enforcement.md) — **OFFLINE GATE IMPLEMENTED 2026-06-18 (uncommitted).** Enforcement + test plan for the "planner emits no trailing `Sources:` block" rule (the prompt already says it; the LLM violates it probabilistically). Mechanism settled by probe `experiments/orchestrator_stop_sequence/stop_sequence_probe.py`: inject decoder `stop=["\nSources:", "\nSources :"]` via **`create_chat_model(..., stop=[...])`** (lives on the model instance → survives deepagents `bind_tools`); `model.bind(stop=...)` is a **footgun** (silently dropped by `bind_tools`); middleware `model_settings["stop"]` also works but heavier; regex `strip_sources_block` is a **non-streaming-only** safety net (can't un-send a streamed block → stop is primary). Exact marker is `Sources:` (no space) — `"Sources :"` alone would never fire. Test matrix A–G (offline plumbing/marker/strip/preservation = gate; online provider-matrix/E2E-KA/streaming = opt-in `integration_online`). Invariant: NO trailing block, but PRESERVE inline `[N]` (rendered downstream from `ka_sources`) + substance. R4 (dangling `[N]`) de-risked by the 2026-06-16 determinism doc. Open Qs: stop-only vs +strip-net; is the answer streamed (makes stop mandatory); renderer live. Builds on [[creative_phase_2026-06-16_sources_determinism_and_synthesis_context]]. **§9 follow-on (IMPLEMENTED 2026-06-18, uncommitted):** killed the planner's multi-call offset arithmetic by *injecting* the canonical `ka_sources` list back as a `<knowledge_sources>` `SystemMessage` after each KA `task` (per-call, offset-numbered from `len(state["ka_sources"])`) in `KnowledgeBridgeMiddleware.wrap_tool_call` (append-only behind the untouched answer ToolMessage — pitfall #5 forbids rewriting the KA answer; `SystemMessage` not `AIMessage` to dodge Anthropic prefill firing the stop). Spike verified: deepagents `task` returns a `Command` whose `update` carries `messages`+`ka_sources`, so the merge is a one-line append. Simplified `_OUTPUT_FORMAT` (deleted the ~80-word shift rule → cite by the note) + broadened `_PLANNER_STOP_SEQUENCES` to 4 incl. `\n**Sources`/`\n## Sources`. Relocation decision: bridge MIDDLEWARE already in orchestrator; shared channel contract STAYS in KA (moving inverts knowledge_agent→orchestrator + breaks KA standalone exports). 398 orchestrator offline tests pass; ruff+pyright clean. Files: `middlewares/knowledge_bridge.py`, `orchestrator_planner_prompt.py`, `orchestrator_catalog.py`, 3 test files, orchestrator `AGENTS.md`. **§9.10 parallel-fix (IMPLEMENTED 2026-06-18, uncommitted) — SUPERSEDES the §9 wrap_tool_call mechanism:** the per-`task` `wrap_tool_call` announcer numbered colliding `[N]` under PARALLEL KA fan-out (siblings in one tool super-step share a pre-merge `request.state`, so each numbers from the same offset while the reducer concatenates them — a per-call hook can't see sibling contributions). Moved the announcer to **`before_model`/`abefore_model`**: it numbers from the POST-merge `ka_sources` + a private `ka_sources_announced` cursor (announces `ka_sources[announced:]` as `[announced+1…]`, one note per planner turn covering a whole parallel batch). Cursor is `PrivateStateAttr` on `KnowledgeBridgeState` only (never crosses `task` to KA), single-writer LastValue, reset to 0 in `before_agent`. Net simplification: bridge no longer wraps tool calls → SubagentTaskFailure is the sole/outermost `wrap_tool_call`; pitfall #5 holds trivially (announcer never touches the answer ToolMessage). Spike confirmed `before_model` sees post-merge reducer state. `_OUTPUT_FORMAT` bullet generalized ("a `<knowledge_sources>` note … may be more than one"). 399 orchestrator offline pass (41 bridge incl. an end-to-end parallel-fan-out test); ruff+pyright clean. Files: `middlewares/knowledge_bridge.py`, `orchestrator_planner_prompt.py`, `middlewares/test_knowledge_bridge.py`, `test_orchestrator_planner_guidance.py`, orchestrator `AGENTS.md`.
- [creative_phases/orchestrator/creative_phase_2026-06-16_sources_determinism_and_synthesis_context.md](creative_phases/orchestrator/creative_phase_2026-06-16_sources_determinism_and_synthesis_context.md) — **DONE (uncommitted) + 1 open thread.** Deterministic sources: planner emits NO `Sources:` block/urls, cites by the sub-agent's own number unchanged (gaps fine) with a multi-call OFFSET rule; KA `ka_sources` now 1:1 with `answer_citations` within a call (`_build_ka_sources` no second dedup) and **pure concatenation across calls** (`merge_ka_sources`, removed `_source_dedupe_key`) so the offset arithmetic is position-stable (trade-off: a doc cited by 2 KA calls shows twice). Tests green (1357 offline). Suggested commits in doc. **OPEN:** KA `SynthesizeNode` doesn't see per-page metadata/context-summary of findings → "who is contact of X Team" mis-attributes generic contact pages whose metadata is the real disambiguating signal; want a context summary per page id surfaced into synthesis. Start at `utils/findings_format.py`, `nodes/synthesize.py`, check if `Finding`/`Citation` even carry page metadata at synthesis time (may be lost in compression).
- [creative_phases/creative_phase_2026-07-13_reasoning_effort_config.md](creative_phases/creative_phase_2026-07-13_reasoning_effort_config.md) — ✅ **IMPLEMENTED 2026-07-13.** Declarative per-family reasoning control: `reasoning_effort=` on `create_chat_model` + `build_reasoning_kwargs()` / `supported_reasoning_efforts()` / `register_reasoning_family()` in `models/reasoning.py`. Plain dict table (mistral binary none/high via `model_kwargs`; nemotron-super off/low/high + nemotron-ultra off/low/medium/high via `chat_template_kwargs` — ultra low = `medium_effort` WITHOUT `force_nonempty_content`, medium/high bake it True; qwen3 = **binary off/high** per the real `Qwen/Qwen3.6-27B` card — graded `thinking_token_budget` rungs are serving-stack-dependent, shipped only as a consumer re-registration example in the doc; openai/gpt = silent native passthrough). Warn-not-raise; explicit caller kwargs beat table values on leaf conflict (deep-merge); `reasoning_family=` kwarg pins gateway-aliased names. Family matching is case/separator-insensitive (`qwen3.6`≡`qwen3-6`≡`Qwen/Qwen3.6-32B`≡`qwen3:32b`); `match` = flat AND-tuple or OR-of-AND-groups for alias dialects (lone strings coerced — missing-comma trap guarded). Revives the 2026-05-30 probe doc's "declarative per family" prescription. Consumer doc `docs/consuming/reasoning.md`. Deferred: settings-level env defaults; mistral tool-call thinking-drop converter patch (upstream `langchain_mistralai` blanks content+thinking on tool-call turns — needs online evidence).
- [creative_phases/creative_phase_2026-05-30_reasoning_compatibility.md](creative_phases/creative_phase_2026-05-30_reasoning_compatibility.md) — ✅ **REVIVED 2026-07-13** → superseded by [[creative_phase_2026-07-13_reasoning_effort_config]]; empirical findings remain the grounding evidence (thinking on-by-default; top-level `reasoning_effort` no-op through the LiteLLM gateway; only `chat_template_kwargs` moved Nemotron). Split out of the v2.4 deferral of `reasoning_effort`. Captures the empirical vLLM/LiteLLM-gateway reasoning findings (2026-05-29): thinking on-by-default for gpt-oss/Qwen3/Nemotron; top-level `reasoning_effort` a no-op (langchain-openai Responses-API rewrite doesn't fire for chat completions); reasoning parseable non-streaming from `provider_specific_fields["reasoning"]` (LiteLLM field — gateway is LiteLLM-fronted); streaming surfaces reasoning only for gpt-oss (LiteLLM #20246 / #9578); Nemotron `chat_template_kwargs.low_effort` is an effort toggle not on/off. Conclusion: no uniform lever — if revived, do it declaratively per model family; re-run the probe (`experiments/langchain/chatopenai/reasoning_script.py`) on the then-current gateway first.

## Active highlights

- [External agent capability cards for the orchestrator](creative_phases/orchestrator/creative_phase_2026-06-13_external_agent_cards.md) — **DESIGN LOCKED 2026-06-13, ready for Phase 1.** How externally-produced agents (own LangGraph deployments, run as `RemoteGraph`) self-describe scope so the planner routes to them like internal subagents. **Research:** A2A AgentCard (LF spec v1.0) is the only real interop standard (ACP merged into it; agents.json/ai-plugin.json dead; OASF niche/wraps A2A; MCP has no agent primitive) — but no standard carries source-of-truth/use-for/freshness; our `CapabilityDefinition` is richer. **Verified on installed stack:** `langgraph-api 0.9.0` already serves A2A cards (`a2a.py` `GET /a2a/{assistant_id}/.well-known/agent-card.json`) but the auto-card is FLAT — only `card.description` (+ graph key→name) is producer-controllable; `tags`/`examples`/`skills[].metadata` are hardcoded (`generate_agent_card` `a2a.py:2000-2092`); `langgraph.json` dict form `{"path","description","config"}` reads no `metadata` (`graph.py:512-563`). **Carrier proven by LIVE TEST** (langgraph dev, elastic_rag_mock, restored clean): structured-JSON-in-`description` round-trips intact through the A2A endpoint with zero server patch. **Locked v0 schema:** `{description, short_description, scope, freshness(live|periodic|static_docs), how_to_use, examples}` — `use_for` merged into `description`; `how_to_use` kept separate from `examples`; **dropped** `source_of_truth`+`dont_use_for` (over-complicate producer contract); `short_description` UX-only (not routed). **Maps to `CapabilityDefinition`** (+ NEW optional `how_to_use` field → `How to use:` section in `_compact_description`); `scope`→`use_for[0]`, `freshness`→`note` caveat. **Trust:** orchestrator assigns the catalog `key` (not the card — no `incident_agent` shadowing); profile size-capped; malformed→thin degraded path. New feature → no deprecation needed (`AGENTS.local.md`). **Phase 1** schema+adapter (`agent_card_to_capability(card,*,key)`, `extract_twin_profile`, `twin_profile_to_description` helper+JSON Schema) pure/offline/tested; **Phase 2** producer docs + extend `defining-subagent-capabilities.md`; **Phase 3 (later, needs arch decisions)** discovery/registry + `RemoteGraph` subagent build + per-deployment auth.
- [Orchestrator skills middleware — discovery + 3-PR plan](creative_phases/orchestrator/creative_phase_2026-06-12_skills_middleware.md) — **PLANNED, decisions LOCKED 2026-06-13, ready for PR-S1.** Bring agent skills to the TWIN orchestrator via deepagents' built-in `SkillsMiddleware` (0.6.7 ships it: backend+sources, progressive disclosure — name+description in system prompt, full SKILL.md read via existing `read_file`; loads once per run in `before_agent` with the SAME checkpoint-staleness bug-class `LiveMemoryMiddleware` fixes). **Decisions:** Q1 bank = packaged read-only `/skills/builtin/` FilesystemBackend in the wheel; Q2 anonymous = bank-only (no user Store route); Q3 `/<skill_name>` command leaves the user message byte-identical (append tail reminder, never rewrite); Q4 planner-only (no per-subagent skills); Q5 all roles (skills are prompt text, gated by `has_uid`+`enable_skills`, NOT `select_orchestrator_permissions`). PR-S1: two CompositeBackend routes (`/skills/builtin/` FS both shapes + `/skills/user/` `(uid,"skills")` Store auth-only) + 1 proof skill in wheel + `skills=[...]` kwarg on `create_deep_agent` + `enable_skills` full off-switch. PR-S2: `LiveSkillsMiddleware` (wrap_tool_call refresh after write/edit under /skills/) + portable `skill-creator` SKILL.md + 4-5 generic bank skills. PR-S3: `/<skill_name>` `before_agent` (after guard+skills-load) parses last HumanMessage, injects skill body as immutable `<system_reminder>` (knowledge-bridge deterministic-id pattern). Top risk: user-authored skill bodies are an injection surface the prompt guard never screens (authoring-time screening decided in PR-S2). PR-S1 pre-flight: confirm `CompositeBackend` mixes FilesystemBackend+StoreBackend routes; confirm nested `skills/<name>/SKILL.md` ships in wheel.
- [Orchestrator skills refactor/simplification (post-PR-review)](creative_phases/orchestrator/creative_phase_2026-07-22_skills_refactor_simplification.md) — **IMPLEMENTED 2026-07-22** (branch `feat/orchestrator-skills`, PR #62). 4-agent review of the shipped skills feature → 3-phase refactor + security repair: (D1) added deepagents-native `create_deep_agent(permissions=[FilesystemPermission(write-deny /skills/builtin/**)])`, but a security repair KEPT `_ReadOnlyFilesystemBackend` as the enforcement boundary (deepagents globs are `BRACE|GLOBSTAR`, no `DOTGLOB` → the deny rule can't cover dot-prefixed components like `/skills/builtin/.evil/`) and demoted the permission rule to a tool-layer UX message; (D3) `ActiveFeatures.resolve()` frozen dataclass normalizes the auth×feature matrix, `has_uid`→`has_memory` rename (public factory kwargs unchanged); (D4) Store namespaces migrate NOW `(uid, kind)` → `("users", uid, kind)` — scope-type-first, no prod stores yet so free; prepared for `("teams", team_id, …)`/`("apps", app_code, …)`; (D5) **real bug found:** `FilesystemToolGateMiddleware` keys on memory alone → skills-on/memory-off shapes mount skills but strip all fs tools (progressive disclosure + user authoring silently dead); fix = gate on `any_user_writable`; (D2/OPEN-Q1 RESOLVED) renamed `ReloadableSkillsMiddleware` → **`LoadableSkillsMiddleware`** (`loadable_skills.py`; Badr's proposal chosen over `SkillCommandsMiddleware` — both `/skill-reload` and `/<skill_name>` are user-invoked loads) + 5 internal simplifications (dead `_INJECT_MARKER`, wrong hard-coded mounts sentence, unused ctor knob, sentinel-tuple plan, shared dispatch). OPEN-Q2: landed as commits on #62. Deferred follow-up: conftest fixture promotion + shared tail-reminder helpers (predicate drift across 3 middlewares).
- [Skills distribution — tiers, frontend awareness & Context Hub](creative_phases/orchestrator/creative_phase_2026-07-24_skills_distribution.md) — **DESIGN 2026-07-24, D1–D5 locked, OPEN-Q1–Q3 gate implementation; core mechanics PROBED LIVE.** How skills are stored/shared/surfaced to frontends (Streamlit now, NestJS+BFF behind Apigee later), across multiple graphs in one deployment and standalone-repo agents. **Tiers (D1):** user→Store (unchanged) · agent-local→`skills/` folder in each agent repo (fix locating with `importlib.resources`, kill the `parent.parent` traversal) · shared→**Context Hub, piloted** (curated-tier fallback ladder Hub → shared store → wheel; Mongo rejected — no official `BaseStore`). **User-tier CORRECTION 2026-07-24 (Badr):** the self-owned external Postgres is the **user tier's target home** — user skills mutualized across ALL deployments (`AsyncPostgresStore(dsn)` per deployment behind the existing `StoreBackend`; namespace `("users", uid, "skills")` + resolvers + writable posture unchanged, only the instance changes; deployment-managed store = interim; BFF reads the one DB directly, no per-deployment fan-out; effectively resolves OPEN-Q2 → per-user-GLOBAL; must land before first prod Store writes; `/memory/*` follow-the-user flagged as a separate decision). (D2, amended) no separate package — `agents/skills` subpackage on the `agents/cards` import-light precedent (`SkillInfo` DTO + `list_skills()` + router = the vendor-neutral protocol boundary) **plus** `banks/generic/` (in-repo authoring source of the `generic-skills` Hub repo, v0 = skill-creator; supersedes the orchestrator wheel copy once landed) and a `sta skills push --skills-group-folder <dir>` command (folder = one Hub group repo, nested SKILL.md validation, `skill-group` tag, idempotent, `--dry-run`; UI-free import path — upstream `langsmith` 0.10.6 ships NO hub CLI despite the blog, re-check at pilot; git review = the review gate, Hub tags = distribution pinning); subpackage + `GET /skills/{graph_id}` are **committed v0 scope** (no longer trigger-gated); D3 also gained an explicit endpoint-placement × store-access table (deployment-hosted enumerates via `langgraph_api.store.get_store()` + local manifest; BFF-hosted aggregates via Store REST `StoreClient` + per-agent cards/endpoints, never reads wheels). (D3) frontend awareness = **producer-CLI-generated per-graph `skills_manifest.json`** → card `skills` projection (static) + `GET /skills/{graph_id}` on the custom `http.app` (live merge; store via `langgraph_api.store.get_store()`, uid explicit — `get_config()` is graph-context-only); NO code registry; clients learn skills only from each agent's self-description. (D4) shared tier = **one Hub skill repo per GROUP** (`sta-<group>-skills`, skills flat at root, `skill-group` marker tag, per-group env-tag pinning; cross-team singles get own repos, `SkillEntry`-linked). (D5) skills/examples externalize; **policies+system instructions stay in code** (trust boundary + prefix cache). (D6) federated/untrusted skills ride the EXISTING admission gate (card-poisoning class) — display is low-risk, invocation is gated; no second gate. **Live probe findings:** nested bundles push/pull byte-identical; a skill repo CAN link skill repos and pull materializes linked content; deepagents 0.6.12 ships `ContextHubBackend` natively (accepts both repo types); stock `SkillsMiddleware` discovers from the Hub E2E — **one folder-level per source**; ⚠️ `ContextHubBackend` caches the tree per instance → `/skill-reload` would serve stale (pilot item). OPEN: Q1 pilot gates (corp network, failure mode, tenancy, ergonomics) · Q2 user-skills namespace per-user-global vs per-agent (free only until first prod Store writes) · Q3 build triggers (only `importlib.resources` fix is immediate). **SEEDED 2026-07-24:** Hub repo `generic-skills` (group bundle, tags `skill-group`+`generic`) holds `skill-creator` verbatim; discovery verified live; wheel stays the orchestrator's live source pending OPEN-Q1. tmp-probe repos left in the LangSmith org for UI inspection.
- [External agent capability cards (A2A)](creative_phases/orchestrator/creative_phase_2026-06-13_external_agent_cards.md) — **Phase 1 IMPLEMENTED (uncommitted, branch `feat/external-agent-cards`), Phases 2–3 PLANNED.** How externally-produced agents (own LangGraph deploys, run as `RemoteGraph`) self-describe so the planner routes to them like internal subagents. Carrier = a JSON capability profile stringified into the graph `description` (the ONLY producer-controllable field that reaches the A2A card; `skills[].metadata` is server-built and unfillable — proven on live `langgraph-api 0.9.0`). Profile v0 = `{description, short_description, scope, freshness(FREE TEXT), how_to_use, examples}`. **Shipped:** vendor-neutral contract `agents/cards/` (`AgentCapabilityProfile` pydantic + size caps; `extract_agent_profile` description-only; `agent_profile_to_description` helper + JSON Schema; `AgentInputContract`+`extract_input_contract` parsing server-built `inputSchema` for Phase-3 invocation) — pydantic+stdlib only, so the producer CLI never imports orchestrator/graph code; orchestrator-side adapter `agents/orchestrator/sources/external_agent_card.py` (`agent_card_to_capability(card,*,key)` — key assigned by US, malformed→thin self-reported degrade) + `how_to_use` on `CapabilityDefinition` + "How to use:" render + producer CLI `sta agent-profile {validate,build,schema,example}` (Click, registered `sta` console script). 48 offline tests. Phase 2 = producer docs; Phase 3 = registry/discovery (per-`assistant_id`) + RemoteGraph wiring + admission gate.
- [External agent admission gate (mini)](creative_phases/orchestrator/creative_phase_2026-07-03_external_agent_gate.md) — **FOLLOWUP / DESIGN SKETCH — not scheduled.** Trust gate for *discovered* external-agent descriptions (untrusted, land in the planner prompt → injection/over-claim vector; structural caps are necessary-not-sufficient). Proposes a discovery-time semantic **judge guard** (cheap model, admit/quarantine/reject) keyed by `sha256(description)` verdict cache — re-judge only on change (closes TOCTOU), fail-CLOSED, runs at admission not per-request, shares one judge with `sta agent-profile --judge`. Judge fn is pure → buildable/testable offline before Phase-3 wiring. Open: verdict store (lean LangGraph Store) + quarantine surface. **Folded into the federation master doc below as Gate Layer 2.**
- [`sta agent-profile` bundling & direct-to-destination writes](creative_phases/orchestrator/creative_phase_2026-07-12_agent_profile_cli_bundling.md) — **IMPLEMENTED 2026-07-12 (uncommitted, branch `feat/external-agent-cards`)** — 35 new CLI tests (104 total), ruff+pyright clean, verified end-to-end against the real `sta` binary. Two producer-CLI enhancements on the shipped `sta agent-profile`: (1) **root include-manifest** — `agent_profile.yaml` is a bare list of card-file paths (or `{cards: [...]}`) that bundles N self-contained `{graph_key:{path,card}}` files (path lives in each ref, root adds nothing per-agent); ref paths resolve relative to the root file's dir; dup-key + flat-card-ref → errors. (2) **`--into PATH`** writes generated graphs into an existing `langgraph.json` (`graphs` block) or `Dockerfile` (`ENV LANGSERVE_GRAPHS=` line), destination inferred by filename (`--as` override). **Merge-preserve is the LOCKED default** — graphs already in the destination but not in the manifest are KEPT (both destinations share one `merge_graphs(existing,new,*,replace)` core; Dockerfile parses the JSON out of the ENV line, `'`↔`'` round-trips for free); `--replace` opt-in makes the manifest authoritative (drops extras); malformed existing value → hard error (never clobber); atomic write + `--dry-run` + added/overwritten/preserved delta report. Parent: [[external_agent_cards]] §Followup.
- [External agent federation — discovery, gating & roster mixing (Phase 3 master)](creative_phases/orchestrator/creative_phase_2026-07-12_external_agent_federation.md) — **DESIGN SKETCH 2026-07-12 — ⛔ IMPLEMENTATION BLOCKED** on the third-party discovery-API interface contract (the API we call with the user token to get a user's allowed assistants). The `{deployment_url, assistant_id, agent_card}` entry shape is an **ASSUMPTION, not confirmed** — the API may return only an id/URL (forcing a separate card fetch) and likely also answers Q1 (RemoteGraph invocation auth). Unblock = get the API contract, then revise §3/§5/Q1 before any Step 1/2 code; the rest (convergence seam, two-layer gating, caching) is contract-independent. Synthesises the two docs above (Phase-1 cards SHIPPED + admission gate) with the new **discovery-API architecture**: the orchestrator calls ONE third-party access API at init with the **user token** → `[{deployment_url, assistant_id, agent_card}]` scoped to that user's access (discovery = the remote-side habilitation; orchestrator holds NO per-remote creds). Each accessible remote becomes an ordinary `RemoteGraph`-backed `SubagentSpec` (a "federated remote subagent") merged into the same roster as local subagents — the planner can't tell them apart (convergence on `CapabilityDefinition` + `as_subagent`, already shipped). **Two-layer gating (kept LATER): Layer 1** producer CI/CD gate (`sta agent-profile --judge`, we have leverage, must be clear+safe), **Layer 2** orchestrator init-time admission gate (`sha256(desc)` verdict cache shared across users, fail-CLOSED, degrade-don't-fail). New hard problem = **per-user init-time discovery caching**: fold a discovery signature (sorted admitted `(assistant_id, desc_sha256)`) into the graph cache key so users with the same remote set share a graph (never raw token/uid). 7 open Qs (Q1 RemoteGraph invocation auth = blocking; lean user-token passthrough). Build order Step 0(DONE)→1 remote build fn→2 discovery client→3 roster merge→**4 gating**→5 defense-in-depth rendering.
- [Frontend — Streamlit aiohttp session cleanup](creative_phases/frontend/creative_phase_2026-06-02_streamlit_aiohttp_session_cleanup.md) — **PROPOSED — DEFERRED 2026-06-02**. Fix for "Unclosed client session / Unclosed connector" warnings from the Elasticsearch adapter. Root cause: the ES async client's aiohttp `ClientSession` is bound to the event loop it was created on, and nothing closes it at lifecycle end. Eval-side half SHIPPED on `perf/lazy-imports-lightrag-extras` (`876758a5`): adapter `WeakSet` + `aclose_all_adapters()` teardown helper, called in `evals/cli.py` `finally`; auto-recovery now closes the stale client; `AsyncElasticGenericRepository.aclose()` added. Frontend half (this doc) deferred — Streamlit's `graph_catalog._run_async` spins a fresh `asyncio.run` loop per interaction, so a cached in-process graph's session is orphaned each turn (accumulates → several warnings). **Option A** (recommended, ~5 LOC): call `await aclose_all_adapters()` in `_run_async`'s `finally` (+ `chat.py:383`) — kills warnings, costs one ES reconnect per turn. **Option B** (robust, larger): single persistent background loop so the session stays on a live loop — no reconnect, but real streaming-pipeline surface change. Ship A first. In-process graphs only (RemoteGraph leaks land in the server's logs).
- [Eval skills grounding audit](creative_phases/evaluation/reference_2026-06-01_eval_skills_grounding_audit.md) — **REPORT 2026-06-01, NO FIXES APPLIED.** `ground-eval-skills` workflow (24 agents) checked all 23 files of both eval skills against live `sta-eval` source. 75 findings (25 high). 2 STALE files won't import (`evaluator-wiring.md`, `evaluators_skeleton.py`: `make_tool_signal`/`_plan_size_signal` from wrong module + direct trajectory evaluators called as factories → TypeError). 5 themes: (1) those crash-level template imports; (2) `guidelines` wrongly doc'd as a `BaseReferenceOutput` base field in 3 files (it's an orchestrator-subclass field); (3) faithfulness grounding still lists `expected_facts` in 3 designer files — contradicts this session's transcript-grounding fix; (4) `dataset-buckets.md` data-gap encoding says `expected_behavior=="answer"` — contradicts this session's `acknowledge_gap`+`reference_tool_calls` semantics; (5) fabricated CLI surface (`--config` should be `-o/--override`; exit-code-2 fabricated; rubric-list "crash" now warns-and-ignores). User chose REPORT ONLY — fixes per-file later. Full per-file finding list + fixes in the doc.
- [Frontend — RemoteGraph trace/feedback linking](creative_phases/frontend/creative_phase_2026-05-26_remotegraph_trace_linking.md) — **IMPLEMENTED 2026-05-26 (Path A)**. Fix for "feedback succeeds but `get_run_url` 404s on RemoteGraphs": `RemoteGraph.astream` doesn't forward `config.run_id`, so the server-side run id (the real LangSmith trace id) drifted from the client-minted UUID. Frontend now strips `config["run_id"]` for remote graphs and, after streaming completes, reads the real id back via `remote_graph.client.runs.list(thread_id, limit=1)` and overwrites `current_turn_run_id` — feedback widget then looks up the right trace. Best-effort: list-failure falls back to the client UUID. Same PR also unwraps the `{"value": [...]}` channel envelope returned by `RemoteGraph.get_state()` to kill `MESSAGE_COERCION_FAILURE` warnings on every render. 9 new unit tests. **Path B (distributed-tracing via `RunTree` wrapper + `RemoteGraph(distributed_tracing=True)`) documented as follow-up** — architecturally cleaner (frontend becomes the parent trace, mirrors PR #43's gateway pattern) but adds parent-trace lifecycle complexity; defer until eval traces need Streamlit-side spans.
- [PR-3.5 namespace-scoped backend migration](creative_phases/orchestrator/migration_namespace_scoped_backend_2026-05-26.md) — **IMPLEMENTED 2026-05-26.** Hard cut from PR-3's callable-factory `backend=make_backend` (deprecated in deepagents 0.5.0, removal in 0.7.0 — `filesystem.py:737-748`) to the blessed pattern: backend INSTANCE with `namespace=lambda rt: ...` resolver. Per-call uid resolution lives in `resolve_memory_namespace` (reads `get_config()` since `_NamespaceRuntimeCompat` doesn't carry `.config`); per-graph anonymous-vs-authenticated dispatch lives in `build_orchestrator_backend(has_uid=...)` at catalog-build time. Same backend INSTANCE shared by `FilesystemMiddleware` (via `create_deep_agent(backend=...)`) and `LiveMemoryMiddleware` — pinned by `test_authenticated_filesystem_and_live_memory_share_same_backend_instance`. `LiveMemoryMiddleware` simplified (~30 LOC removed — both `_resolve_backend` and `_resolve_backend_for_agent` deleted; synthetic `ToolRuntime` construction gone). `test_backend_factory.py` deleted → `test_user_backend.py` (12 tests including capfd-pinned no-deprecation success signal). Verified: 145 passed, zero `warn_deprecated` in output, `-W error::DeprecationWarning:deepagents` sweep green. Implementation log appended to bottom of the doc.
- [PR-3 MemoryMiddleware design review (orchestrator deep-agent rewrite)](creative_phases/orchestrator/review_pr3_memory.md) — **IMPLEMENTED 2026-05-25.** 10-point sequential review of PR-3 (per-user memory via deepagents `MemoryMiddleware`). Shipped: `LiveMemoryMiddleware(MemoryMiddleware)` single subclass with `wrap_tool_call` post-edit sync (O(edits) backend reads, not O(turns)); `CompositeBackend` path-routes `/memory/` → `StoreBackend(namespace=(uid,"memory"))`; conditional middleware composition on `x-uid` presence (cache key gains `has_uid` bit); custom 24-line `<memory_guidelines>` for two-file model (`AGENTS.md` user-authored, `preferences.md` LLM-learned); LGP zero-config Postgres + `make_orchestrator(config, *, store=None, checkpointer=None)` optional kwargs for standalone (tests / in-process frontend). **38 PR-3 tests green** across 6 suites (12 unit live_memory + 3 prompt + 5 backend factory + 8 catalog + 7 e2e + 3 subagent isolation — exceeds plan's 22 with codex-driven add-ons). Binding revisions applied: `_get_store()` `RuntimeError` re-raised (Open Q4 REJECTED); `x-uid` allowlist-validated at factory boundary; `_backend_ref` pinned on subclass; `CompositeBackend` leading-slash preservation pinned by test; `raise ValueError` (not `assert`) guard against `memory=[...]` shortcut. Implementation log appended to the bottom of the design doc.
- [Codex adversarial review of PR-3 implementation plan](creative_phases/orchestrator/codex_adversarial_review_pr3_2026-05-25.md) — **REVIEWED + DISPOSITIONED 2026-05-25.** Codex (v0.133.0, gpt-5.5 xhigh) grounded against upstream `langchain-ai/deepagents` `main` + LangChain reference docs (local sandbox blocked `.venv` reads — re-verify locally). 5 TOP RISKS evaluated: (1) raw `x-uid` namespace validation — **ACCEPTED**, Phase B `make_backend` validates/encodes (strict allowlist or `sha256`); (2) fail-loud missing store — **ACCEPTED**, flips Open Q4, Phase A try/except must NOT swallow `_get_store()` `RuntimeError`; (3) `DeltaChannel` double-emit — **DROPPED**, wrapper is the only emission path under `wrap_tool_call` semantics, plan's `Command(update=...)` adds ToolMessage exactly once; (4) concurrent same-uid writes — **ACCEPTED (test only)**, Suite E adds test, no CAS in PR-3; (5) graph-cache `threading.Lock` — **DEFERRED**, compile is cheap (~20-80ms, lazy per `[CI-05]`), `threading.Lock` would block asyncio loop under langgraph-api 0.4.x sync constraint, revisit at PR-6 with async-factory after 0.7.x bump. SHARPENs accepted: `_backend` private (store own ref on subclass), `CompositeBackend` preserves leading slash (`/AGENTS.md`), `ToolRuntime` needs all 6 fields (prefer `request.runtime` directly), subagent state filter blocks `memory_contents` key but memory-derived text can still leak via `messages`/`structured_response`. Open Q final: Q1/Q2/Q3/Q5 agreed; Q4 REJECTED. Codex session `019e6005-e458-75d3-b3c6-68a45c6a4f50`.
- [Skills v0 remainder plan — push CLI, Hub mount, /skills endpoint](creative_phases/orchestrator/implementation_plan_2026-07-26_skills_push_hub_endpoint.md) — **PROPOSED 2026-07-26, awaiting confirmation.** Three phases on `feat/skills-v0`: (1) `sta skills push` click command (client-side delta vs Hub head, `None` tombstones so deletions propagate, `parent_commit` CAS, metadata only on create — grounded in langsmith 0.10.6 source); (2) Hub consumption mount behind `enable_hub_skills=False` — requires `ReloadableContextHubBackend` TTL wrapper (VERIFIED: stock backend snapshots tree at first read, never invalidates), source order wheel-generic→hub→builtin→user gives the Hub→wheel fallback ladder for free, `/skill-reload` invalidates; (3) `GET /skills/{graph_id}` + drill-in on `app.py` via `alist_skills_with_errors` + `StoreBackend(store=…)`, manifest half deferred. Blocking decision D-A: repo handle convention (`sta-<folder>-skills` recommended) + re-seed and delete stale `generic-skills`. OPEN-Q1 gate 1 (reachability) resolved 2026-07-26: prod already ships traces → reuse deployment `LANGSMITH_API_KEY`.
- [PR-3 implementation plan — `LiveMemoryMiddleware` + per-user backend factory](creative_phases/orchestrator/implementation_plan_pr3_memory.md) — **IMPLEMENTED 2026-05-25** (phases A-F all green; see implementation log in `review_pr3_memory.md`). Phased build plan (A→F + 22 tests + docs) executed on `feat/orchestrator-deepagent`. A: `LiveMemoryMiddleware` subclass + custom prompt landed. B: `make_backend(rt)` landed under `backends/user_backend.py` with `x-uid` allowlist regex `^[A-Za-z0-9\-_.@+:~]+$` validating BEFORE namespace construction (codex Risk 1 fix). C: `make_orchestrator` wiring + `GraphCacheKey` 4→5-tuple with `has_uid` + `store`/`checkpointer` kwargs on `create_orchestrator_factory(...)` + explicit `raise ValueError` guard against `memory=[...]`. D: `langgraph.json` already forwards `x-uid` (line 38). E: **38 tests** shipped (plan target 22; the over-delivery comes from codex add-ons — async-refresh variant, `RuntimeError` re-raise contract, invalid-uid rejection, leading-slash preservation, factory-guard sanity, concurrent same-uid). F: docs (this file + `AGENTS.md` "implemented" + Setup table + pitfalls 14–16) + review_pr3_memory.md flipped to `status: IMPLEMENTED` + Implementation log appended. Open questions resolved: Q1 `raise` (asserts stripped under `-O`); Q2 raw `x-uid` for `has_uid`; Q3 duplicate uid-read; Q4 REJECTED (fail loud on `RuntimeError`); Q5 GP deny-rule deferred.
- [Twin Router KA source-awareness](creative_phases/twin_router/creative_phase_2026-05-23_router_ka_source_awareness.md) — **APPROVED — ready to implement 2026-05-23** as a pre-merge follow-up on `feat/ka_query_planner_improvements` (PR #43, 29 commits ahead). Surface KA `RetrieverEntry` descriptions in the twin router's `lisab` `ToolDefinition` so the router routes confidently to KA in `RagMode.KNOWLEDGE`. Single source of truth: `entry.description`. Mechanical projection at graph-build time, no duplicate strings. New `list_twin_ka_sources()` factored out of `build_twin_ka_entries()` to avoid network I/O at registry build (preserves the `[CI-05]` lazy-construction invariant). Conditional rebuild of the `rag` `AgentEntry`'s `tool_definition` in `_build_graph_for_permissions` — EXTERNAL/INTERNAL paths untouched. 6 tests cover env permutations + no-construction guard + KNOWLEDGE/EXTERNAL prompt assertions. ~40 LOC + ~80 LOC tests, ~1-2h.
- [KA Contextual-Content Refactor — reranker · granularity · truncation · synthesis](creative_phases/knowledge_agent/creative_phase_2026-06-17_ka_contextual_content.md) — **IMPLEMENTED 2026-06-17** (session `ka_refacto`; all 6 phases shipped, D1-D6 resolved, adversarial review passed, NOT yet committed). Routes the structured `metadata.content` blob three ways BY CONSUMER: (a) **reranker** scores the reconstructed contextual blob (`context_summary`+body via `_rerank_document_text`) so apcode/appName/title text helps the cross-encoder — RECONSTRUCT, not a stored `raw_content` field (no leak); (b) **compressor/synthesizer** get per-chunk `contextualized_content` inside each `<chunk>`/fact + page-shared remainder once per `<document>` (`page_shared_context`/`page_shared_summary` strip the per-chunk prefix); (c) **display/citations** get clean body. Synthesis truncation fixed (`_SYNTHESIS_SOURCE_CONTEXT_CHARS=1200`; review stays at hoisted `_REVIEW_SOURCE_CONTEXT_CHARS=320`). Subagent synthesis prompt: completeness-outranks-brevity + enumerations-in-full + grounded "Next steps". **Field-gate CLEARED:** prod field is `metadata.contextualisedContent` (British 's'), confirmed in `infra/elasticsearch/ingestion/indexer.py:107`. Orchestrator's KA is wired to `fast` → next-steps are model-derived (documented in KA `AGENTS.md`). `context_summary` kept FULL (additive only) for back-compat. Tests green (engine 3750 / core 1408).
- [ExpandNode Batch Fetch + fetch_target gating](creative_phases/knowledge_agent/creative_phase_2026-05-22_expand_batch_fetch.md) — **IMPLEMENTED 2026-05-22** on branch `feat/ka_query_planner_improvements`. ExpandNode collapsed N per-item ES queries into ~1 batched query per retriever. Codex review rejected a core-side `BatchDocumentProvider`; final design = `SupportsBatchFetch` marker protocol (ES batch methods) + engine-local `FetchExecutor` adapter as the single batch-vs-loop branch point (LightRAG/externals keep the per-item `DocumentProvider` loop unchanged). Phase 0 also gated `fetch_target` IDs to `DocumentProvider`-capable retrievers. Item 2 (inner loop in `search_depth=fast`) investigated only — deferred; `deep`+`max_iterations=1` is the existing path for "expansion without outer loop".
- [ElasticRagProxy — every expansion triggers HYBRID SEARCH (bug)](creative_phases/knowledge_agent/creative_phase_2026-05-23_elasticrag_proxy_hybrid_search_bug.md) — **RESOLVED 2026-05-23**. Root cause: `langgraph dev` instance at the frontend's gateway URL was running an older branch (pre-Phase-1, no `operation` dispatch). Re-deploying the up-to-date branch restored expected behavior. Bug was NOT in branch code. Same commit also lands a defensive guard: `_dispatch_search` now short-circuits empty / whitespace-only `query` to an empty success envelope without calling the retriever, emitting a WARNING — so any future skew where an old proxy + new gateway (or any field-drop bug) produces an empty query surfaces as empty results + a single log line instead of silent BM25-empty + vector-near-origin noise. Backwards-compatible: hand-rolled empty searches continue to receive a clean "no results" envelope, no exception raised. Tests: `TestElasticRagEmptyQueryShortCircuit` (4 cases).
- [ElasticRag Batch Fetch — Phase 4 implementation plan](creative_phases/knowledge_agent/creative_phase_2026-05-23_elasticrag_batch_fetch_phase4_plan.md) — **APPROVED — ready to implement 2026-05-23** on `feat/ka_query_planner_improvements` (HEAD `128986e`, 16 commits ahead, unpushed). Phases 1-3 of the elasticrag DocumentProtocol design are DONE (operation dispatch + envelope + proxy `DocumentProvider` + cross-package contract + KA integration test + single-fetch engineering example). Phase 4 = `SupportsBatchFetch` track: gateway grows three batch ops (`get_documents`/`get_chunk_contexts`/`get_chunk_ranges`) with array-of-records wire shape for ranges (NOT string-key tuple encoding — doc-ids may carry arbitrary characters); proxy declares `supports_batch_fetch=True` + implements `SupportsBatchFetch` + correlation-guaranteed envelope parsing; `_PROXY_OPERATIONS` grows in lockstep via the existing cross-package contract test; KA integration test proves `FetchExecutor` actually drives the batch path (1 call for N targets) AND falls back to per-item loop on `batch_fetch_unavailable`. **Positions on open questions**: (a) `batch_ceiling_hit` = **truncate-and-warn** (matches direct retriever); (b) `response_cap_hit` = **proxy-only** (gateway has no wire-size view); (c) warning surface = **`logger.warning` only**, no state threading; (d) example = **split** into new file. **Hidden drift flagged**: `_BATCH_FETCH_MAX_CHUNKS` lives on backing retriever, not gateway — gateway detects truncation post-hoc; `include_entity_childs`/`include_transversal` aren't in `_SCOPE_FIELDS` (pre-existing gap, NOT fixed in this phase). 4 commits expected; branch ends 20 ahead. ~7h focused.
- [ElasticRagRetriever + DocumentProvider / SupportsBatchFetch](creative_phases/knowledge_agent/creative_phase_2026-05-22_elasticrag_documentprotocol.md) — **APPROVED — ready to implement 2026-05-22 (refined 2026-05-23 after Codex round-2 review).** Extends the `elastic_rag` gateway-proxy adapter so it satisfies `DocumentProvider` + `SupportsBatchFetch`, unblocking the KA `ExpandNode` inner expansion loop for hosted-gateway deployments (today `_resolve_provider` skips the proxy). Endpoint audit confirmed the gateway is **search-only** — the fix needs a gateway-side change: an additive `operation` discriminator field dispatched inside the existing single `retrieve` node (`operation` omitted ⇒ `search`, wire-compatible). Round-2 review hardened the design: discriminated response envelope `{operation, ok, results?, error?, warnings?}`; concrete structured-error table (7 codes → exception class mapping); gateway boundary rejects `operation`-less doc-op requests + scope fields on any non-search op; `get_chunk_ranges` wire uses array-of-records (no string-key tuple encoding — robust to doc IDs with arbitrary characters); explicit operational-limits section (`batch_ceiling_hit` / `response_cap_hit` warnings, no automatic splitting in this phase). Capability still declared unconditionally — the imprecise marker trade-off is on the record (simplicity beats a probe round-trip; mock-misconfig surfaces as `NotImplementedError` on first fetch, not silent degradation). 4 phases, TDD. `docs/consuming/` deferred (project_status #17, scope-bypass on by-ID fetch).
- [Query-Planner Review Fixes — handoff](creative_phases/knowledge_agent/creative_phase_2026-05-22_query_planner_review_fixes.md) — **HANDOFF 2026-05-22.** Fix list for a new session after the 3-reviewer pass on the uncommitted KA query-planner + `MetadataScope` work. F1 (pyright blocker at `metadata_scope.py:230`), F2 (move scope normalization to clause-build time — covers `model_copy`/`model_construct` escape hatches + kills F1), I3 (per-axis normalizer-fn policy instead of hardcoded lower/upper casing), F3 (doc contradiction reword), A4 (style). F4/F6/F7 done this session (reverted accidentally-corrupted `baseline_report.json`; re-marked the real-LLM `test_baseline_report` as `integration_online`; CLAUDE.md lesson on scoping test runs).
- [ES-KA Catalog Flatten + Naming Convention](creative_phases/knowledge_agent/creative_phase_2026-05-23_es_ka_catalog_flatten.md) — **APPROVED 2026-05-23** for branch `feat/es-backend` (PR #44, 43 commits ahead, pushed). 3 commits to land *before* the upcoming KA rebase: (1) docs-only `packages/sta_agent_engine/AGENTS.md` § "File naming convention" (forward-only, prefix bare `graph.py`/`catalog.py`/`prompts.py`/`state.py` with package short name — fixes monorepo IDE search debt) + § "Catalog factory patterns" (pre-built / `@cache` getter / `make_<name>(config[, runtime])` — version caveat: 1-arg on langgraph-api 0.4.x prod, 2-arg ≥ 0.7.x for Studio schema-read fast path) + `[CI-07]` known-issue entry for grandfathered bare names; (2) refactor `deepagent_es/catalog.py` 133 lines → < 30: new `es_ka_graph.py:get_es_knowledge_graph(...)` inlines today's `build_deepagent_es_graph` + `assemble_graph`, drops `llm_provider` kwarg, return type `CompiledStateGraph`; `catalog.py` shrinks to `@cache`'d `get_es_knowledge_graph_instance()` + TODO seat for future per-request factory. Caller updates: `langgraph.json`, `experiments/graph_configs/graphs.jsonl` (KA-conflict hotspot — single row), example, contract test, docstrings. CI-05 preserved. Package rename `deepagent_es` → `es_knowledge_agent` is out-of-scope (separate follow-up PR). Function names land in post-rename form now to avoid renaming twice.
- [KA Improvements Brainstorm — 4 threads (env model configs / trace propagation / messages-first I/O / subagent mode)](creative_phases/knowledge_agent/creative_phase_2026-05-18_ka_improvements_brainstorm.md) — **READY FOR IMPLEMENTATION 2026-05-19** on branch `ka_improvements_brainstorm`. Step 1 = three independent PRs (B trace propagation via `config.configurable` body injection mirroring `RemoteGraph(distributed_tracing=True)`; D subagent mode reusing existing concise prompt + dropping `[Fn]` citation contract; A env-driven per-task overrides + stripping hardcoded `"provider": "llmaas"` from package defaults so KA config goes provider/model-agnostic — 5-layer resolution ladder with `LLM_PROVIDER` engine-wide env as layer 5 fallback). Step 2 = Thread C (messages-first I/O) deferred — needs fresh creative phase covering dual-read shims everywhere + state versioning + checkpointer replay + frontend citation contract. Codex adversarial review + user-direction history preserved as appendices in the doc; "FINAL IMPLEMENTATION PLAN" section at the top is the source of truth.
- [ElasticRagRetriever — gateway-proxy adapter](creative_phases/knowledge_agent/creative_phase_2026-05-16_elastic_rag_retriever_adapter.md) — PROPOSED 2026-05-16, **in implementation** (worktree `ka_scope_elastic_rag_adapter`). Client-side adapter that wraps the deployed `elastic_rag` LangGraph gateway as a `BaseRetriever[RetrievalChunk]` + `SupportsMetadataScope`. Planner output + codex review (verdict REWORK) synthesized into v2 plan. Lives in `sta_agent_core` (vendor-clean — reuses `AsyncHttpAdapter`, NO `langgraph` dep). POST `/runs/wait` with flat scope-field decomposition. Engineering-only release — `docs/consuming/` deferred until project_status #17 (server-side scope enforcement). Codex non-negotiables: package boundary (core not engine), explicit kwarg allow-list (typed signature, no `**kwargs`), cross-package parity contract test. ~210 LOC + ~330 LOC tests + ~150 LOC example. Phases 1–3 in-scope; Phase 4 deferred.
- [ES mapping alignment — local vs production](creative_phases/knowledge_agent/creative_phase_2026-05-15_es_mapping_alignment.md) — **IMPLEMENTED 2026-05-15** (worktree `es_mapping_alignment`, 2 commits: alignment + review polish). Production index mapping audited 2026-05-15; 12 mismatches between local ingestion + `ElasticFieldConfig` defaults and prod resolved. `metadata.auid` (not `apcode`) is the canonical apcode field in prod (one-line `ElasticFieldConfig` default flip); `metadata.content` is a structured `\n`-separated blob (summary + Url page + Application + name + title + content body) while `metadata.contextualisedContent` carries just the summary; `metadata.entity.name` reverted to `keyword + lowercase_normalizer` to match prod; `int8_hnsw` quantization + `_source.excludes:[embedding]` added; `metadata.chunk_size{char, token_embed_model, token_llm}` replaces flat `start/end/token_count`. KA + ElasticRetriever surface unchanged — only retriever code changes are `extract_chunk_body()` helper + apcode extraction with boundary normalization in `_default_result_mapper`. 453 ES retriever + 696 KA tests green. **Live ES re-ingest deferred** (next-session: `make es/up` + `cli ingest --force --clear-cache` + smoke + probes).
- [DeepAgent with Elasticsearch as Virtual Filesystem (v3.1)](creative_phases/knowledge_agent/creative_phase_2026-05-14_deepagent_es_filesystem.md) — PROPOSED 2026-05-14, **near-term improvement opportunity** (three-round codex-validated 2026-05-14/15; design verdict "sound, ready for phase 0"). Parallel/alt agent on `langchain-ai/deepagents` 0.6.1: ES index mounted at `/knowledge/es/` via built-in `CompositeBackend`; `StateBackend` default for `/notes/` scratchpad. Built-in FS tools only (`ls`/`read_file`/`grep`/`glob`) with `custom_tool_descriptions` seam (`middleware/filesystem.py:580`) for ES semantics — no fork. `grep` grammar `[<mode>:][@{axis=val,...}] <query>` (modes `auto`/`fts`/`knn`/`h`/`hr`); runtime query scope **boost-only** via `MetadataScope.add_boosts()` — mirrors production KA scope (no path-as-scope, no regex). `read_file` accepts doc name OR pageId via single OR-query + sidecar pageId terms agg; multi-doc match returns all chunks (inline-labeled per pageId) + structured warning + `NOTE:` header — non-blocking. Build-time scope ceiling at backend ctor; user scope deferred until twin_router→entity; phase 3 hosted-graph demoted until user scope ships (`AGENTS.md:108` trust-boundary policy). New upstream additive: `ElasticRetriever.get_chunks_by_reference()` (~50 LOC). Phase 0: ~370–500 LOC + one-shot pre-spike `terms` agg gate to verify `metadata.doc.keyword` shape (path-like vs flat/Confluence vs opaque). 13 risks tracked with codex-validated severities.
- [DeepAgent-ES Phase 0 — implementation plan](creative_phases/knowledge_agent/creative_phase_2026-05-21_deepagent_es_phase0_plan.md) — **APPROVED 2026-05-21, in implementation** on branch `feat/es-backend` (cut from `worktree-deepagent-es-preflight-probe` so v3.7 doc + probe travel with it). Step-by-step plan for the deepagent-es spike, grounded in v3.7 Decision Sync + `preflight_results_2026-05-19.md`. New decision: **composable per-backend tool descriptions** — STA-authored thin `BASE_TOOL_DESCRIPTIONS` + per-backend `tool_description_fragments(prefix)` + `compose_tool_descriptions()` composer (`FilesystemMiddleware` takes one flat dict; assembly happens before middleware build). Order B→C→D→E: `MetadataScope.narrow_with` + `_EMPTY` sentinel (B3 contract test gates the ship) → `ElasticMetadataValueResolver.list_values` → example `deepagent_es_filesystem_example.py` → equivalence/addressability tests. Open items: 2-segment resolver exact-match spec (pin during D2), optional P9 `after_key` recheck (non-blocking), C1 (`title` text-only, frozen mapping — fixed constraint), upstream `STABackend` protocol (Phase 1).
- [Elastic ingestion — metadata enrichment for KA scope testing (build-time + runtime query)](creative_phases/knowledge_agent/creative_phase_2026-05-14_elastic_metadata_enrichment.md) — IMPLEMENTED (code-side) 2026-05-14 in worktree `elastic_metadata_enrichment_doc`. Sources rewritten (11 entries: 3 DU package apcodes + 2 transversal + Foundation twin for collision B + 3 Acme tenant slices). Chunker now handles `path.is_file()` for single-file sources. Local mapping reverted `entity.name` to `keyword + lowercase_normalizer` per 2026-05-15 audit (the 2026-05-14 `text` revert was based on a transcription gap — see archived [resolver aggs gap](_archive/project_resolver_aggs_entity_name_text.md)). 15 Acme markdown fixtures authored. 16 offline tests green. P1–P7 acceptance probes script at `infra/elasticsearch/probes/metadata_scope_smoke.py` — **live verification still pending** (needs `make es/up` + ingestion + probe run).
- [KA Scope Hardening — codex review follow-ups](creative_phases/knowledge_agent/creative_phase_2026-05-13_ka_scope_hardening.md) — IMPLEMENTED 2026-05-15. Groups A (#4/#5/#7/#8 — `823810c`), B (#6 — `d64e1de`), C (#3 — `be16855`), D (#1 — `SupportsMetadataScope` runtime-checkable Protocol + ES `ClassVar[Literal[True]]` marker + tool-factory build-time gate), E (#2 — composite `after_key` drain + flat-terms `sum_other_doc_count` WARN + `_DEFAULT_AGG_SIZE` 5000→65535) all landed; vocabulary rename (Layer 1/2/3 → "build-time / user / runtime query scope") landed in Group D's prep commit. **Open caveat (B1):** Group E's `_DEFAULT_AGG_SIZE=65535` × 4 aggs may breach ES `search.max_buckets` (per-response, default 65 536); verification deferred until prod cardinality numbers arrive (`infra/elasticsearch/probes/cardinality/`). Round-3 review at `.claude/PRPs/reviews/group-e-review.md`.
- [KA Scope Architecture — Three-Layer Model with Value Resolver](creative_phases/knowledge_agent/creative_phase_2026-05-11_ka_scope_three_layer.md) — IMPLEMENTED 2026-05-11 (commits `bde247a..31ce85e` + example `examples/sta_agent_engine/knowledge_agent_three_layer_scope_example.py`; L2 explicitly deferred). Originally PROPOSED v2 2026-05-11 (refined after adversarial review; v1 narrative kept inline with `[SUPERSEDED]` markers). **Supersedes** the same-day `elastic_retriever_ka_tuning` doc. First step in #45. v1 ships **L1 only** (build-time `entry.default_scope` ceiling) + **L3 boost-only** LLM args (`apcode`/`app_name`/`entity`) cleaned via new `MetadataValueResolver` Protocol with `ElasticMetadataValueResolver` impl (composite name↔id agg ∪ `entity.childs` terms agg, lazy 8am refresh with `asyncio.Lock` + jitter, 2-bucket fuzzy: codes τ=90 / names τ=80). L3 writes to `*_boost` axes only — never filter — so cross-references in other apcodes still surface via BM25 text-match (`elastic_retriever.py:497-520`). **L2 `user_scope_mode` deferred** until twin_router plumbs user→entity. Warnings → ToolMessage tail + `findings.metadata["warnings"]` + structured `logger.warning`. ~430 LOC / 6 phases. Unblocks #16 / #17 / #44.
- [ElasticRetriever → KA Integration & Tuning](creative_phases/knowledge_agent/creative_phase_2026-05-11_elastic_retriever_ka_tuning.md) — **SUPERSEDED 2026-05-11** (kept for design history). Earlier same-day doc; assembled scope at caller-context with merge tables. Replaced by the metadata-scope model above (originally framed as "three-layer" — see vocabulary banner in the design doc).
- [Twin Router KA Revamp — replace Adaptive RAG with the Knowledge Agent](creative_phases/twin_router/creative_phase_2026-05-21_twin_router_ka_revamp.md) — **PROPOSED 2026-05-21** on branch `feat/twin_router_revamp_rag_to_ka`. Swap `AdaptiveRagGraph` behind the `lisab` RAG tool for the Knowledge Agent via a new `RagMode.KNOWLEDGE` + `KnowledgeAgentRagStrategy` (sub-agent-behind-a-tool, lazy build per CI-05, answer-mode + `subagent_mode`). One KA for prod users (non-prod keep no RAG — already automatic via `POLICIES["non-prod"]=frozenset()`); KA carries two direct-`ElasticRetriever` entries — broad `twin_docs_general` (unscoped) + `twin_project_knowledge` (twin entity/apcode-scoped, `include_transversal=True`). Never the `elastic_rag_proxy`. Twin scope values anonymized via `TWIN_SCOPE_*` env arrays (`TwinScopeSettings` Pydantic BaseSettings, declared empty in `.env.example`). `Citation→RagSource` adapter keeps the twin-router contract. Per-user entity scoping deferred. Adaptive-RAG path deprecated, removed in a later PR. 4 phases.
- [Twin Router Robustness — habilitation toggle, multi-model native, layered timeout defense](creative_phases/twin_router/creative_phase_2026-05-08_twin_router_robustness.md) — Three-phase short-term hardening (#43 + #46 + #47) before the deep-agent rewrite (#14). Phase 1 IMPLEMENTED 2026-05-10; Phase 2 IMPLEMENTED 2026-05-10 (multimodal guard middleware — strip image parts for non-multimodal models, GK tool dual-call site covered, 24 tests); Phase 3 PARTIAL (L1 + L2 shipped 2026-05-10; L3 FastAPI middleware abandoned same day — wrong layer for run-level timeouts in LangGraph Platform's queue model).
- [Twin Router Robustness Phase 1 — habilitation bypass implementation log](creative_phases/twin_router/creative_phase_2026-05-10_robustness_phase1.md) — IMPLEMENTED 2026-05-10. HABILITATION_BYPASS env + DeploymentConfig toggle, BypassHabilitationProvider with role typo trap, 21 new tests, twin_router README.
- [Twin Router Robustness Phase 3 — layered timeout defense implementation log](creative_phases/twin_router/creative_phase_2026-05-10_robustness_phase3.md) — PARTIAL 2026-05-10. L1 (httpx stall fence in `create_chat_model`) + L2 (operator guide / strategic-catch recipe) shipped; L3 (FastAPI `RequestTimeoutMiddleware`) shipped then rolled back same day. Run-level wall fence pending as a separate follow-up.
- [Retriever-Owns-Expansion — active thread](creative_phases/retriever_owns_expansion/README.md) — Authoritative architectural reference (step 4 v3 amendment) + ongoing execution plan (step 8). Project_status #30 closed 2026-05-07; remaining follow-ups are #34 / #35 / #39.
- [Retriever Eval Sweep — Expansion × Fusion matrix](creative_phases/elastic_rag/creative_phase_2026-04-17_retriever_eval_sweep.md) — FUTURE. Post-F6 plan to run the existing retriever dataset across every `(expansion_hint × fusion_strategy)` combination.
- [Twin Router Habilitation](creative_phases/twin_router/creative_phase_2026-03-27_twin_router_habilitation.md) — Authorization for twin router: registry, factory closure, auto bias rules, `DeploymentConfig`.
- [Habilitation Refactor](creative_phases/twin_router/creative_phase_2026-04-03_habilitation_refactor_code_quality.md) — Fail-open fix, registry re-exports, prompt-builder unification, persona flow cleanup.
- [Auth Evolution](creative_phases/twin_router/creative_phase_2026-03-27_auth_evolution.md) — Auth layer progression: headers → FastAPI middleware → native `@auth` → `@auth.on` → policy engine.
- [Retriever & LightRAG Architecture](creative_phases/lightrag/creative_phase_2026-02-09_retriever_and_lightrag_architecture.md) — Foundational retriever protocol + LightRAG integration. Cross-referenced by knowledge_agent/, roadmap/, `.cursor/rules/sta-agent-core.mdc`.

## Archived

- [Retriever-Owns-Expansion — experiment history](creative_phases/_archive/retriever_owns_expansion/HISTORY.md) — Archived 2026-04-17. Steps 1–3 design thrash + shipped execution logs (5, 6, 7, 9).
- [Elastic RAG Workflow (Phases 1–7+9 IMPLEMENTED)](creative_phases/_archive/elastic_rag/creative_phase_2026-04-12_elastic_rag_workflow.md) — Archived 2026-05-07. RRF-first retriever + LangGraph gateway + shared factory + multi-graph LGP deploy + owned-vs-rented consumer access + `close()` cascade.
- [Next Session Brief — Phase 5 post-implementation](creative_phases/_archive/next_session_brief.md) — Archived 2026-05-07. Outdated since Phase 5 + Cycle F + step 9 closed; superseded by `project_status.md`.

## Cross-cutting / synthesis

- [evals/creative_phase_2026-03-01_evaluation_doc.md](evals/creative_phase_2026-03-01_evaluation_doc.md) — Evaluation framework synthesis (eval-driven development, three pillars, dataset/evaluator design). Continue iterating per project_status #26.

## Project

- [Corpus profile — sparse, OPS-in-tech](project_corpus_profile.md) — French IT/ops banking back-office corpus; shapes expansion tuning, eval thresholds, and DOMAIN_ENTITIES priorities.
- [External agent federation — resume playbook](project_external_agent_federation_resume.md) — ON HOLD 2026-07-15: how to restart Phase 3 (discovery → gate → roster merge) when the third-party discovery-API contract arrives; docs to load in order, first actions, code entry points, carried-over Q1–Q7.

## Reference

- [AGENTS.md](AGENTS.md) — Memory-bank conventions (lifecycle, naming, what NOT to put here)

-------

memory_bank/creative_phases/alfred/creative_phase_2026-07-24_hitl_interrupt_contract.md
----
# Creative Phase 2026-07-24 — ALFRED HITL & interrupt interface contract

**Status:** 🎨 PROPOSED — design groundwork for the ALFRED product build.
**Scope:** one coherent human-in-the-loop contract between LangGraph backends and the
ALFRED frontend (AG-UI/Next.js POC on `experiment/alfred-ag-ui`), composing **custom
`interrupt()`** and **`HumanInTheLoopMiddleware`** behind a single wire schema.
**Ground truth:** every claim marked ✅ was empirically verified against
`langgraph 1.2.9` / `langchain 1.3.14` / `langgraph-api 0.9.0` — probes in
`experiments/interrupt_contract/` (+ session probes, see § Verification).
**Related:** `experiments/alfred_ag_ui/web/lib/protocol.mjs` (current POC normalizer),
`experiments/alfred_ag_ui/web/lib/contracts.ts` (current `ResumeRequest` union).

---

## Problem

ALFRED will pause runs for humans in at least four shapes: tool-call approval,
clarification with options + free-text notes, structured form input, and "answer as
the tool". LangGraph gives us **two producers** with opposite API contracts:

| | `HumanInTheLoopMiddleware` | custom `interrupt()` in node/tool |
|---|---|---|
| Interrupts per turn | always **1** (batches all gated calls) | **N** — one per interrupting task |
| Payload schema | fixed `HITLRequest` | whatever we invent |
| Resume addressing | positional `decisions[]` | by `Interrupt.id` |
| Effect application | middleware rewrites messages | our code |
| Validation on resume | count + `allowed_decisions` ✅ | none unless we write it |

Without a contract, the frontend needs shape-sniffing per producer (the POC's
`normalizeInterrupt` already guesses from `question`/`choices` keys), every new
interrupt kind means frontend work, and positional decision matching is a live
mis-approval hazard (§ D5). This doc fixes the contract before ALFRED hardens.

## Verified ground truth (the constraints we design under)

1. **Envelope is 2 fields, ever:** `Interrupt {value, id}`. `id` = xxh3-128 of the
   task's checkpoint namespace — opaque, thread-specific, **no path/node/graph info**.
   A 3-level-deep subgraph interrupt is byte-identical to a top-level one. ✅
2. **Surfacing:** `invoke()` → values dict + aggregated `__interrupt__` list;
   `stream()` → **one chunk per interrupt, arrival order unstable** (`updates`
   chunk, or merged into `values`); `get_state()` → `snapshot.interrupts`
   (authoritative, ordered; `values` has **no** `__interrupt__` key); server →
   `thread.status == "interrupted"` + `thread.interrupts {task_id: [...]}`. ✅
3. **`messages` stream mode never carries interrupts** — a messages-only subscriber
   streams tokens then goes silent at a pause. Must multiplex with `values` or
   `updates` (SDK default is `values`). `tasks`/`debug` carry them off-shape. ✅
4. **Resume:** `Command(resume=scalar)` for exactly one pending; **id-map
   `{interrupt_id: value}` for any number** — and a dict counts as id-map only if
   *every key is 32-hex*. Scalar with >1 pending → `RuntimeError`. `None` is not a
   valid resume value. Resume = new run on the same thread, no input. ✅
5. **Id-map composes with the middleware:** `Command(resume={iid: {"decisions":[…]}})`
   resumes a `HumanInTheLoopMiddleware` pause correctly. ✅ ← this is what lets the
   frontend speak **one** resume dialect for both producers.
6. **Re-execution:** the interrupted task re-runs **from the top** on resume (tool
   body, node, or `after_model` hook; the model itself is *not* re-invoked for the
   middleware). Code before `interrupt()` runs once per round-trip → idempotency
   or move side effects after the gate. Sequential interrupts in one task share
   one `id`, matched by call order. ✅
7. **Payload = durable JSON DTO.** In-process everything survives (msgpack), but
   `langgraph_api.serde` flattens for the wire: pydantic/dataclass→dict,
   bytes→base64, set→list, **unknown class → `null` silently** (even nested).
   Payloads are checkpointed and **outlive deploys** (old-shape payloads get served
   to new frontends after redeploy). ✅
8. **Middleware validation is real but positional:** count-match + decision-type ∈
   `allowed_decisions` raise `ValueError`; an order swap between two actions with
   *identical* policies is **undetectable** and silently mis-applies. ✅

## The contract

### D1 — One wire schema: normalized HITL shape for every "reviewable action"

All interrupts that model *review of one or more actions* — tool approval, clarify
with options, forms, ask-user — use the framework's `HITLRequest` shape, **also when
emitted from our own tools/nodes**:

```jsonc
{
  "action_requests": [{ "name": "...", "args": {...}, "description": "..." }],
  "review_configs":  [{ "action_name": "...",
                        "allowed_decisions": ["approve","edit","reject","respond"],
                        "args_schema": { /* JSON Schema — drives the UI form */ } }],
  // D2 additive envelope keys (absent on genuine middleware requests — that
  // absence is itself the "framework-emitted" signal):
  "source": "billing_agent",
  "schema_version": 1
}
```

- `action_requests` = the instances (what is being asked, this turn);
  `review_configs` = the policy per action *name* (how it may be answered).
  Join on `action_name`; never sent back in the response.
- Clarify-with-options+notes fits natively: `allowed_decisions: ["edit","reject"]`
  + `args_schema` with `{"choice": {"enum": [...]}, "notes": {"type":"string"}}`. ✅
- **Mirror the TypedDicts in ALFRED code; do not import them** —
  `HITLRequest`/`ActionRequest`/`ReviewConfig`/`Decision` are *not* in
  `langchain.agents.middleware.__all__` (only the middleware + `InterruptOnConfig`
  are). Structural compatibility, zero coupling; pin a shape-parity test so
  upstream drift is caught. ✅
- Escape hatch `{"kind": "<type>", "schema_version": 1, "source": ..., "data": {...}}`
  is reserved for interactions that **break the decision algebra itself** — not
  answerable as approve/edit/reject/respond in one round-trip. A custom *look* is
  NOT a reason: special widgets (ranking, map-pick, calendar) stay inside the
  normalized renderer via additive UI hints beside `args_schema` (D2), e.g.
  `{"ui": {"order": {"widget": "rank"}}}` with the generic form as fallback.
  Wizards decompose into one normalized interrupt per round-trip (D7). Target
  population of `kind`: **zero** — it prevents schema contortion for a true
  misfit; it is not an extension mechanism, and the renderer registry behind
  D3 rule 2 ships empty.

### D2 — Envelope extensions (additive-only)

- `source` (**required in ALFRED emitters**): the envelope has no path info (§ GT-1),
  and ALFRED is multi-agent/subgraph — without it the UI cannot say *who* is asking.
- `schema_version` (int, start at 1): payloads outlive deploys (§ GT-7). Evolution
  rule: **additive-only within a version**; bump only when a rename/removal is
  unavoidable; frontend keeps a generic fallback renderer forever.
- Any future UI hints (icons, urgency) are additive keys — never repurpose existing.

### D3 — Frontend routing (formalizes the POC normalizer)

Precedence, applied to each `Interrupt.value`:

```
1. has action_requests        → tool_approval renderer (buttons from allowed_decisions,
                                 form from args_schema; enum → radio/select)
2. has kind                   → registered custom renderer for that kind
3. has question (string)      → legacy clarification renderer   [grandfathered]
4. else                       → generic fallback (pretty-print value + free-text resume)
```

Rule 4 is permanent (§ D2). The POC's option-guessing from `choices`/`options` keys
retires once emitters move to `args_schema` enums (rule 1).

### D4 — Resume protocol: the frontend always sends an id-map

One dialect, both producers, any count of pending interrupts:

```jsonc
POST /threads/{threadId}/resume
{ "resume": { "<interrupt_id>": <value> , ... } }
// middleware pause → value is {"decisions":[...]}          (✅ verified via id-map)
// custom tool/node → value is whatever that emitter accepts
```

- Kills three hazards at once: the scalar-vs-multiple `RuntimeError`, the
  "dict-payload accidentally parsed as id-map" heuristic (our values are never bare
  32-hex-keyed dicts), and cross-producer branching in the client.
- The BFF route keeps accepting the POC's `ResumeRequest` union (`{value}` |
  `{interruptId, value}` | `{decisions}`) but **normalizes to id-map** before hitting
  LangGraph, resolving ids from thread state when the client sent none.
- Client renders from accumulated stream interrupts but **re-reads
  `GET /threads/{id}` state before submitting** — authoritative set + ids (§ D6).

### D5 — Response integrity: close the positional-matching hole

`decisions[]` ↔ `action_requests[]` matching is positional; an order swap between
same-policy actions silently approves the wrong action (§ GT-8 — probe: swapped
decisions **deleted prod with no error**). ALFRED closes this in three layers:

1. **Echo:** frontend adds `action_name` to every decision:
   `{"type":"approve","action_name":"read_file"}`. Extra keys are ignored by the
   middleware ✅ — free to add today.
2. **Backend validation (own emitters):** a shared `apply_decisions(req, resp)`
   helper enforces: count match; `action_name` echo matches positionally (when
   present); `type ∈ allowed_decisions`; `edit` ⇒ `edited_action` present; reject
   unknown fields. This is middleware-parity **plus** the ordering guard neither
   producer has.
3. **Policy diversity as defense-in-depth:** dangerous actions get *distinct*
   `allowed_decisions` (e.g. destructive ⇒ `["reject"]` or `["approve","reject"]`
   with nothing else sharing that set in the same batch) — differing policies turn
   silent swaps into loud `ValueError`s even inside the unmodified middleware.

### D6 — Streaming & detection rules (frontend/BFF)

- Subscribe `stream_mode=["values","messages"]` (SDK already defaults to `values`;
  never messages-only — § GT-3).
- Treat any non-empty `__interrupt__` as "paused"; **accumulate across chunks**
  (one chunk per interrupt, unstable order — § GT-2); reconcile with
  `GET /threads/{id}` (`status == "interrupted"`, `interrupts` map) once the
  stream closes — that is the authoritative view and the reconnect/refresh path.
- Static `interrupt_before/after` pauses emit **no** stream event and no payload —
  server-visible only via thread state; resume with `input: null`, not a resume
  value. ALFRED avoids static interrupts in product flows.
- `stream_events(version="v3")` typed projections (`.interrupts`, `.interrupted`)
  are the likely future — still `@beta` in 1.2.9; revisit before GA (§ Open).

### D7 — Emitter discipline (tool/node authors)

- Payloads are JSON DTOs: plain dicts of primitives. No pydantic/BaseMessage/custom
  classes (wire-flattening + silent `null` for unknowns — § GT-7). Big artifacts go
  by reference (file/store key), never inline.
- Everything above `interrupt()` in the task body re-runs on resume (§ GT-6):
  side effects go *after* the gate or become idempotent.
- Validate the resume value in the emitter (via `apply_decisions` / kind-specific
  checks) — custom emitters get **zero** framework validation.
- Multi-question flows: prefer one `interrupt()` per round-trip (wizard = several
  cycles) over several sequential `interrupt()`s in one task — sequential ones
  share a single `id` and confuse id-keyed UI state.
- Never route secrets through an interrupt in either direction: the payload AND
  the resume value are durably checkpointed (§ GT-7) and readable via thread
  state / traces. Credential handoff goes out-of-band; the interrupt carries
  only a reference or an acknowledgement.

## ALFRED v1 interrupt catalog (product inventory, 2026-07-24)

Confirmed target population: tool HITL + four clarification shapes. **All five
are one renderer** — they differ only in `review_configs`. No `kind` in v1.

| Use case | `allowed_decisions` | `args_schema` sketch | resume decision |
|---|---|---|---|
| Tool approval | per tool policy (middleware config) | middleware-emitted (none today) | `approve` / `edit` / `reject` / `respond` |
| Question → free-text answer | `["respond"]` (+ `"reject"` if declining is allowed) | — (no form; respond is formless) | `{"type":"respond","message":"…"}` |
| Choose one (enum) | `["edit","reject"]` | `{"choice": {"enum": [...]}}`, required `[choice]` | `edit` |
| Choose one, options carry descriptions | `["edit","reject"]` | `{"choice": {"oneOf": [{"const":"a","title":"Option A","description":"…"}, …]}}` | `edit` |
| Choose one + optional note | `["edit","reject"]` | `{"choice": {"enum": [...]}, "notes": {"type":"string"}}`, required `[choice]` | `edit` |

Catalog conventions:
- **Free text = `respond`, not a one-field edit form** — keeps form machinery
  out of the simplest case and matches the middleware's own "answer as the
  tool" semantics.
- **Described options use the JSON-Schema `oneOf`/`const` idiom** (each option
  `{const, title, description}`) — the standard enum-with-labels encoding that
  schema-form renderers already understand; plain `enum` stays for bare choices.
- **`reject` on clarifications means "user declines to answer"** — whether
  declining is allowed is a per-emitter product choice; the emitter must handle
  it (proceed with a default, or end the branch gracefully).
- Foreseeable additions (plan approval, docx edit/comment review, multi-select
  via `{"type":"array","uniqueItems":true}`, batch memory-write review via
  multiple `action_requests`) all stay inside this same renderer — none is a
  `kind`.

## POC delta (what changes on `experiment/alfred-ag-ui`)

**Keep as-is (POC already implements the contract, sometimes better than assumed):**

- **Collection**: `normalizeInterrupts` deep-walks payloads (depth 8), reads both
  `__interrupt__` and `interrupts` keys, and dedupes by `id + value-fingerprint` —
  this already implements D6's "accumulate across chunks" and absorbs the
  subgraph double-emission (ns + root) for free.
- **Resume validation**: `buildResumeCommand` independently encodes two engine
  rules we verified — id-map keys must be 32-hex, and `null`/`undefined` are
  rejected as resume values (§ GT-4). Keep.
- **Streaming**: `run.ts` already subscribes `["values","updates","messages"]` +
  `stream_subgraphs: true` — a superset of D6's minimum; duplication is handled
  by the dedupe above.

**Change:**

| Area | POC today | Contract target |
|---|---|---|
| Kind detection | infers from `action_requests`/`kind`/`question` | same precedence, formalized (D3); `choices`-key guessing retired |
| Options UI | reads `value.choices/options`; `args_schema` ignored by `normalizeActions` | `args_schema` enum under rule 1 (D1) drives forms |
| Resume | union validated; `{interruptId}`/`{answers}` → id-map, **but `{decisions}` and `{value}` go scalar** → breaks with >1 pending, can't mix with tool interrupts | normalize **all** forms to id-map (D4): wrap `{decisions}` under the pending interrupt's id, resolve ids from thread state |
| Decisions | `normalizeDecision` **rebuilds decisions with only known fields — an `action_name` echo would be silently stripped** | preserve + require the echo (D5.1); the strip is the first thing to change |
| Envelope | no `source`/`schema_version` | required in ALFRED emitters (D2) |
| Validation | request-shape only; nothing checked against the *pending* interrupt (count vs `action_requests`, `allowed_decisions` policy) | shared `apply_decisions` in BFF/graph (D5.2) |

## AG-UI protocol mapping (2026-07-27 — resolves former Open Q1)

AG-UI now has **first-class interrupts** ([docs.ag-ui.com/concepts/interrupts](https://docs.ag-ui.com/concepts/interrupts)):
a run terminates with `RunFinished { outcome: { type: "interrupt", interrupts: [...] } }`
and the client resumes with `RunAgentInput.resume: [{interruptId, status:
"resolved"|"cancelled", payload}]` **covering every open interrupt**. The
protocol independently converged on this doc's decisions — validation, not rework:

| AG-UI `Interrupt` field | This doc's concept |
|---|---|
| `id` | LangGraph interrupt id (D4 id-map key) |
| `reason` (`tool_call` / `input_required` / `confirmation`) | D3 routing discriminator — promoted to envelope level |
| `message` | `action_requests[].description` / question text |
| `responseSchema` (JSON Schema; invalid payload → `RunError`) | D1 `args_schema` + D5 "declare the expected response" |
| `expiresAt` (ISO-8601 TTL) | Open Q5 stale-pause — the wire slot now exists |
| `metadata` | D2 `source` / `schema_version` + raw framework payload |
| resume `status: "cancelled"` | user **abandoned** the interrupt — NOT a denial (AG-UI convention: denials ride the payload, e.g. `{approved:false}` / a `reject` decision) |
| resume must be id-addressed, all open interrupts | **D4 id-map-always** — exact substrate match |

**Adapter rules (BFF or `@ag-ui/langgraph`):**

- **1 LangGraph interrupt → 1 AG-UI interrupt, always** — including a middleware
  batch (N `action_requests` in ONE interrupt). Do NOT fan out to N AG-UI
  interrupts: they would share one LangGraph id, forcing stateful re-aggregation
  on resume. For a batch: reason `alfred:tool_batch`, `toolCallId` unset,
  `responseSchema` = the decisions-array schema, full `HITLRequest` in `metadata`.
- `reason` derivation — AG-UI core taxonomy is `tool_call` (**requires**
  `toolCallId`), `input_required` (structured input, should carry
  `responseSchema`), `confirmation` (free-standing yes/no, boolean default);
  custom reasons are explicitly sanctioned, namespaced `<framework>:<name>`
  (`core:` reserved). Our mapping: single gated action → `tool_call` + its
  `toolCallId`; middleware batch (N>1 — no single toolCallId, so core
  `tool_call` is ineligible) → `alfred:tool_batch`; respond-only and edit-form
  clarifications → `input_required`; untethered approve/reject →
  `confirmation`. Future true `kind`s (D1 escape hatch) get their protocol
  home as `alfred:<kind>` reasons.
- Resume translation is mechanical *because of D4*: AG-UI `resume[]` →
  `Command(resume={interruptId: payload, ...})`. Per AG-UI convention denial
  arrives *inside* the payload (`{approved:false}` / `reject` decision);
  `status: "cancelled"` means the user walked away → route to the emitter's
  decline-without-answer handling, never to `reject`. Stale/expired resumes →
  `RunError`.
- `expiresAt` is populated from the stale-pause policy (Open Q5 keeps the
  policy; the wire encoding is no longer ours to design).
- `@ag-ui/langgraph` emits structured interrupt outcomes behind the opt-in
  **`emitInterruptOutcome`** flag (legacy `CustomEvent(name="on_interrupt")`
  remains the default) — adopting the adapter with that flag replaces the POC's
  hand-rolled interrupt plumbing in `run.ts`/`normalizeInterrupts` at the
  transport layer; the D1–D5 semantic layer is unchanged.

**Frontend routing by `reason` (v1 complete set)** — the field division is:
`reason` picks the card, `responseSchema` is the single source of truth for the
resume payload shape, `metadata` enriches rendering. The adapter transforms
**envelopes, never payloads** — our emitters always set `responseSchema`, so
the frontend always sends exactly what the emitter expects and the adapter
never rewrites payload bodies (AG-UI's boolean-confirmation default applies
only when `responseSchema` is absent, which for us is never).

| `reason` | card | form driven by | resume payload |
|---|---|---|---|
| `tool_call` | single-action approval | `responseSchema` + `metadata` review config for buttons | `{"decisions":[…]}` (one) |
| `alfred:tool_batch` | batch approval — one row per `metadata.action_requests[i]` | `responseSchema` (decisions array) | `{"decisions":[…]}` order-matched + `action_name` echo |
| `input_required` | clarification / form | `responseSchema` (`enum`→radio, `oneOf`→described options, string→textarea) | object per `responseSchema` |
| `confirmation` | yes/no | `responseSchema` (ours: decisions; bare `{approved}` only if schema absent) | per `responseSchema` |
| anything else | generic fallback — `message` + free-form input; **permanent** (D3 rule 4) | — | free-form |

Every card additionally offers dismissal → `status: "cancelled"` (distinct from
an in-payload denial).

Caveat: AG-UI's **recommended** tool-approval schema is `{approved: boolean,
editedArgs?: object}` — `editedArgs` is a full replacement (never merged) and
its *presence in the schema* is the edit-capability signal, their convention
mirroring our `allowed_decisions`. It is poorer than the 4-decision algebra
(`respond` has no equivalent; reject-with-reason needs an extra field), and
`responseSchema` is free-form JSON Schema — so we publish the decisions-array
schema as `responseSchema` instead of downgrading. Trade-off: a generic AG-UI
client that special-cases the recommended `{approved}` pattern won't
auto-render ours; ALFRED owns its client, and the schema stays self-describing.

## Open questions

1. **AG-UI protocol mapping** — ✅ RESOLVED 2026-07-27, see § AG-UI protocol
   mapping above. Remaining follow-up: verify `emitInterruptOutcome` behavior
   against the pinned `@ag-ui/langgraph` version when the POC adopts it.
2. **v3 `stream_events`** — adopt typed projections when out of beta? Would simplify
   D6 client code; contract above is transport-agnostic so it survives either way.
3. **First true `kind` (if ever)** — would force the custom-renderer registry
   design (D3 rule 2). Bar to clear: it must break the decision algebra, not
   the aesthetics — wizards decompose (D7), fancy widgets are UI hints beside
   `args_schema` (D1/D2). Expected to stay empty; park until reality disagrees.
4. **Shape-parity test** — where to host the mirrored-TypedDict vs upstream check
   (engine test suite vs alfred repo) once ALFRED gets its own package.
5. **Stale pauses** — no interrupt-level timeout exists in LangGraph; an
   unanswered interrupt leaves the thread `interrupted` indefinitely (server
   thread-TTL exists but expires the *whole thread*). AG-UI's `expiresAt` now
   provides the wire slot (§ AG-UI mapping) — what remains is the product
   policy (reminder, auto-reject + notify, or expiry), enforcement (nothing
   expires server-side; the emitter/BFF must act), and an "expired" UI state.

## Verification

- `experiments/interrupt_contract/interrupt_contract_probe.py` + README — envelope,
  per-channel surfacing, subgraph/parallel/sequential scenarios, HITL middleware
  4 decisions, stream-mode matrix, v1/v2 shapes.
- Session probes (2026-07-22 → 24, this doc's ✅ marks): payload serde matrix &
  redeploy durability; mid-node re-execution side-effect double-run; nested 3-level
  envelope blindness; middleware validation (count/allowed) + same-policy order-swap
  silent mis-approval; interrupt-in-tool (N interrupts, distinct ids, id-map
  mandatory); clarify tool via `edit`+`args_schema`; **id-map resume of a middleware
  pause** (the D4 keystone).
- Not yet verified: AG-UI event-layer behavior (Open-1); v3 projections under the
  server (Open-2).

-------

packages/sta_agent_core/AGENTS.md
----
# AGENTS.md — sta_agent_core

Data access layer for the STA Agent platform. Provides adapters (low-level clients),
repositories (high-level CRUD/search), and a retriever protocol for RAG — all async-first,
backend-swappable, and designed for external consumers to extend.

## Commands

```bash
# test this package (scoped)
uv run pytest tests/test_core/ -v

# test a single file
uv run pytest tests/test_core/repositories/test_elasticsearch/test_elastic_retriever.py -v

# lint one file
uv run ruff check packages/sta_agent_core/src/sta_agent_core/<path>.py --fix

# typecheck one file
uv run pyright packages/sta_agent_core/src/sta_agent_core/<path>.py

# coverage
uv run pytest tests/test_core/ --cov=packages/sta_agent_core --cov-report=term-missing
```

## Architecture

```
Consumers (sta_agent_engine, domain packages)
        ↓
repositories/retrievers/   → BaseRetriever protocol, SearchResponse[T]
repositories/              → Generic repos (Postgres CRUD, ES search, graph topology)
        ↓
adapters/                  → Raw clients (AsyncPostgresAdapter, AsyncElasticsearchAdapter,
                             AsyncHttpAdapter, GraphAdapter)
        ↓
External systems           → PostgreSQL, Elasticsearch, TigerGraph, LightRAG, NetworkX
```

### Layer Discipline

- **adapters/**: Low-level clients. Thin wrappers around external libraries. No business logic.
- **repositories/**: High-level data access. Query building, result formatting. Depends on adapters.
- **repositories/retrievers/**: RAG retrieval stack. All backends implement `BaseRetriever[T]`.
- **config/**: Provider factories and logging. No data access code.
- **models/**: Pydantic domain models. No I/O, no side effects.

Never mix layers — adapters don't import from repositories, repositories don't import from config.

## Key Patterns

**Retriever protocol** — `BaseRetriever[T]` is a `@runtime_checkable Protocol` with two
methods: `async search()` → `SearchResponse[T]` and `async close()`. All retrieval backends
implement this. Consumers depend on the protocol, never on concrete retrievers.

**SearchResponse[T_co]** — covariant, `Sequence`-like wrapper. Backends attach extra context
(e.g. LightRAG entities/relationships) without coupling consumers to specifics.

**Config resolution** — Pydantic `BaseSettings` with prefixed env vars. Search configs use
`from_context()` (TypedDict overrides from LangGraph state) and `to_search_kwargs()` (→ dict).
Existing prefixes: `POSTGRES_`, `ELASTICSEARCH_`, `RETRIEVER_LIGHTRAG_`, `TG_`, `LLMAAS_`,
`MISTRAL_`, `EMBEDDING_`, `RERANKING_`.
Factory pattern: `Adapter.from_settings(settings)` or `ProviderFactory.get_settings()`.

**Dynamic providers** — `ProviderFactory.get_provider_settings(name)` accepts
arbitrary strings, not just `ProviderType` members. Unknown names auto-derive
the env prefix as `f"{NAME.upper()}_"` and return a synthesized
`BaseProviderSettings` subclass. Use `ProviderFactory.register(name, defaults=...,
env_prefix=..., settings_class=...)` for non-env defaults or a custom prefix.
Built-in `ProviderType` members and `EVAL` fallback semantics are unchanged.

**Graph adapters** — `GraphAdapter` ABC with NetworkX and TigerGraph backends. Cypher-first
for portability. TigerGraph prefers installed queries over interpreted Cypher.

**Structured logging** — Pluggable context enrichment via `ContextEnrichedFilter` +
`register_context_extractor()`. Extractors produce `dict[str, str]` from any context source
(LangGraph ContextVar, Flask request, manual binding). The filter merges results onto each
`LogRecord`; formatters display them generically (`[k=v]` prefix for stdout, top-level keys
for JSON). No vendor imports in this package — `langchain_core` dependency is contained in
`sta_agent_engine`. See `docs/agent-core/configuration.md` § Context Enrichment Pipeline.

## Do

- Implement `BaseRetriever[T]` for new retrieval backends
- Use `dataclass` for search configs and retrieval chunks
- Use Pydantic `BaseSettings` with `env_prefix` for adapter configuration
- Use `async/await` for all adapter and repository methods
- Use `async with` for resource management (connections, sessions)
- Add custom exceptions inheriting from `RetrieverError` for retriever failures
- Export new public types from the relevant `__init__.py`
- Write unit tests alongside new adapters/repositories in `tests/test_core/`
- **Normalize to canonical keys at the boundary** — when multiple backends produce the same concept under different names, normalize in the adapter/parser so consumers see one key. Never let backend-specific naming leak upstream.

## Don't

- Don't add domain-specific business logic — that belongs in domain packages
- Don't leak adapter internals (ES client, asyncpg pool) through repository APIs
- Don't expose internal helpers or vendor-specific types in `__init__.py` exports
- Don't add new external dependencies without approval
- Don't bypass `SearchResponse` — return it from `search()`, even for simple cases
- Don't use sync I/O — everything is async
- Don't mix layers — adapters don't import repositories, repositories don't import config
- Don't let backend-specific naming leak to consumers — normalize at the boundary

## Examples

- **Reference retriever**: `repositories/retrievers/elasticsearch/` — full pattern (settings,
  config, chunk, retriever). Copy this structure for new backends.
- **Reference test**: `tests/test_core/repositories/test_elasticsearch/test_elastic_retriever.py`
- **Extension recipe**: see [Extending](#extending) below

## Safety

Allowed without asking: read files, scoped pytest, ruff/pyright on single files
Ask first: `uv add` new dependencies, modifying `__init__.py` exports, schema changes
Never: direct adapter access from agent code (use repositories), hardcode secrets

## Testing

- Tests live in `tests/test_core/` mirroring the source structure
- Unit tests mock adapters/external clients — never hit real services
- `@pytest.mark.asyncio` for async tests (auto mode configured)
- `@pytest.mark.integration_offline` for local mock data (e.g. NetworkX graphs)
- `@pytest.mark.integration` + `@pytest.mark.slow` for real-service tests (testcontainers)

## Extending

To add a new retriever backend (e.g. Pinecone):

1. Create `repositories/retrievers/pinecone/` with settings, config, chunk, and retriever
2. Implement `BaseRetriever[PineconeRetrievalChunk]`
3. Subclass `RetrievalChunk` for backend-specific fields
4. Subclass `BaseSearchConfig` with `from_context()` + `to_search_kwargs()`
5. Normalize metadata keys to match existing backends (consumers expect one name per concept)
6. Export from `__init__.py`; add tests in `tests/test_core/`

**Optional capability protocols** — declare ONLY if the backend honestly implements the contract:

- `SupportsMetadataScope` — backend honors `search(metadata_scope=...)` and
  resolves the normalized caller bundle into its own scope model. Opt in by
  adding `supports_metadata_scope: ClassVar[Literal[True]] = True` plus
  `resolve_caller_scope(...)`. Scope models implement `is_effective()` and
  `apply_caller_scope(...)`; the latter owns backend-specific combination
  semantics. Lying about support is a trust-boundary failure; back the contract
  with a test that asserts `metadata_scope` reaches the underlying query.
- `DocumentProvider` — backend implements full-document and chunk-context fetches.
  See `document_provider.py` and `ElasticRetriever.get_document(...)` for the recipe.

To add a new adapter (e.g. Redis):

1. Create `adapters/redis/` with adapter class and settings
2. Inherit from `BaseAdapter` or appropriate base (`BaseSearchAdapter`, etc.)
3. Use Pydantic Settings with `env_prefix = "REDIS_"`

## When Stuck

- Check `docs/agent-core/extending.md` for detailed patterns
- Look at `repositories/retrievers/elasticsearch/` as a reference implementation
- Ask a clarifying question or propose a short plan — don't push speculative changes

## Navigation

| Topic            | File                                                               |
| ---------------- | ------------------------------------------------------------------ |
| Overview         | `docs/agent-core/overview.md`                                      |
| Adapters         | `docs/agent-core/adapters.md`                                      |
| Repositories     | `docs/agent-core/repositories.md`                                  |
| Extending        | `docs/agent-core/extending.md`                                     |
| Logging & context| `docs/agent-core/configuration.md` (§ Context Enrichment Pipeline) |
| Design decisions | `memory_bank/creative_phase_2026-02-16_structured_logging.md`      |

-------

packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/elastic_rag_proxy/elastic_rag_retriever.py
----
"""Client-side ``BaseRetriever`` wrapper for the deployed ``elastic_rag`` gateway.

**Engineering-only release.** Caller-supplied ``metadata_scope`` is forwarded
as-is, and a bare ``search(query=...)`` falls back to the build-time
``default_scope`` baked into the retriever (so the entry's ceiling is not
silently dropped on the wire) — but the gateway does not yet enforce caller
identity against scope. Do NOT treat ``default_scope`` here (or in the consumer
KA ``RetrieverEntry``) as a security ceiling: it is client-side ergonomics, not
enforcement. Server-side scope enforcement is tracked as a follow-up workstream.

**Distributed tracing.** When ``settings.distributed_tracing`` is enabled, each
``search()`` call merges the current LangSmith run tree's trace headers
(``langsmith-trace`` and ``baggage`` from ``RunTree.to_headers()``) into the
outgoing ``/runs/wait`` request so the deployed gateway run stitches into
the caller's LangSmith trace as a child. Mirrors the propagation convention
used by ``langgraph.pregel.remote.RemoteGraph(distributed_tracing=True)``.
Off by default; deployments opt in. See
``ElasticRagRetriever._build_trace_headers``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import httpx

from sta_agent_core.adapters.http.async_http_adapter import RETRY_STATUS_CODES, AsyncHttpAdapter
from sta_agent_core.repositories.retrievers.batch_document_provider import ChunkRange
from sta_agent_core.repositories.retrievers.elasticsearch.metadata_scope import MetadataScope
from sta_agent_core.repositories.retrievers.exceptions import (
    RetrieverConnectionError,
    RetrieverResponseError,
)
from sta_agent_core.repositories.retrievers.retrieval_chunk import RetrievalChunk
from sta_agent_core.repositories.retrievers.search_response import SearchResponse

from .context import ElasticRagProxyContext
from .settings import ElasticRagProxyRetrieverSettings


logger = logging.getLogger(__name__)


_RUNS_WAIT_PATH: str = "/runs/wait"

# Statuses that ``AsyncHttpAdapter`` retries; if they exhaust the retry budget,
# the gateway is effectively unreachable per ``RetrieverConnectionError``'s
# docstring ("Server unreachable after retries (connection refused, timeout,
# 503/429)"). Other 4xx/5xx are a malformed-or-unauthorized response from a
# live server → ``RetrieverResponseError``.
_RETRY_EXHAUSTED_STATUSES: frozenset[int] = frozenset(RETRY_STATUS_CODES)


class ElasticRagRetriever:
    """Gateway-proxy adapter — ``BaseRetriever[RetrievalChunk]`` + ``SupportsMetadataScope``.

    Owns one :class:`AsyncHttpAdapter` for its lifetime. Constructors accept an
    optional ``http=`` injection point so tests can supply a ``MockTransport``
    without performing network I/O.
    """

    supports_metadata_scope: ClassVar[Literal[True]] = True
    retriever_type: ClassVar[str] = "elastic_rag_proxy"

    @staticmethod
    def resolve_caller_scope(bundle: Mapping[str, Any]) -> MetadataScope | None:
        """Resolve ``doc_ids``, ``apcode``, ``app_name``, and ``entity`` filters."""
        return MetadataScope.from_caller_scope(bundle)

    # Decision 3 — declared unconditionally. The proxy's authoritative answer
    # is only known at call time (the backing retriever may not be configured
    # for document retrieval); config-level mismatches surface as
    # ``document_provider_unavailable`` and map to ``NotImplementedError``.
    supports_document_provider: ClassVar[Literal[True]] = True

    # Same unconditional-declaration trade-off for batch fetch. The marker is
    # the gate FetchExecutor reads to take the batch path; backing-config
    # mismatches surface as ``batch_fetch_unavailable`` at call time and map
    # to ``NotImplementedError`` (so FetchExecutor's per-item fallback fires).
    supports_batch_fetch: ClassVar[Literal[True]] = True

    # Decision 3 — ``error.code`` → Python exception. Kept on the class so the
    # cross-package contract test (Phase 3) can introspect it directly and
    # assert parity with the gateway's emitter table.
    _ERROR_CODE_TO_EXCEPTION: ClassVar[dict[str, type[Exception]]] = {
        "unsupported_operation": NotImplementedError,
        "missing_required_param": ValueError,
        "document_provider_unavailable": NotImplementedError,
        "batch_fetch_unavailable": NotImplementedError,
        "invalid_range": ValueError,
        "scope_fields_not_allowed": ValueError,
        "gateway_error": RetrieverResponseError,
    }

    def __init__(
        self,
        settings: ElasticRagProxyRetrieverSettings,
        *,
        http: AsyncHttpAdapter | None = None,
        default_scope: MetadataScope | None = None,
    ) -> None:
        self._settings = settings
        self._http = http if http is not None else self._build_http(settings)
        # Build-time scope baked in by the consumer (KA ``RetrieverEntry``). A
        # bare ``search(query=...)`` falls back to it so the entry's ceiling is
        # not silently dropped on the wire. NOT a server-side trust boundary —
        # see the module docstring; the gateway forwards, it does not enforce.
        self._default_scope = default_scope
        self._closed = False

    @staticmethod
    def _build_http(settings: ElasticRagProxyRetrieverSettings) -> AsyncHttpAdapter:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if settings.api_key is not None:
            headers["X-Api-Key"] = settings.api_key.get_secret_value()
        return AsyncHttpAdapter(
            base_url=settings.gateway_url,
            timeout=settings.timeout_s,
            default_headers=headers,
        )

    async def search(
        self,
        query: str,
        size: int | None = None,
        *,
        metadata_scope: MetadataScope | None = None,
        context: ElasticRagProxyContext | None = None,
        **unknown_kwargs: Any,
    ) -> SearchResponse[RetrievalChunk]:
        """POST the gateway's ``/runs/wait`` endpoint and parse the result.

        ``query`` and ``size`` are positional-or-keyword to match the
        ``BaseRetriever`` protocol shape — a generic consumer calling
        ``await r.search("q", 5)`` works.

        ``**unknown_kwargs`` exists only to satisfy ``BaseRetriever``'s
        structural-typing contract (``Protocol.search(..., **kwargs)``). The
        adapter rejects unknown kwargs at the boundary instead of silently
        dropping them on the wire — any unexpected key raises ``TypeError``.

        Args:
            query: User query string.
            size: Number of results to ask the gateway for. Falls back to
                ``settings.default_top_k`` when ``None``. Shortcut for
                ``context.size`` — the explicit ``context.size`` wins when
                both are provided.
            metadata_scope: Optional ``MetadataScope``. Decomposed into flat
                ``input`` fields via ``model_dump(exclude_none=True)``. When
                omitted (``None``), the build-time ``default_scope`` (if any) is
                forwarded instead, so a bare call still carries the entry's
                ceiling. An explicit scope wins — the default is not merged in.
            context: Typed per-call context forwarded to the gateway. See
                :class:`ElasticRagProxyContext` for the full field list.
        """
        if unknown_kwargs:
            raise TypeError(
                f"ElasticRagRetriever.search() got unexpected keyword arguments: "
                f"{sorted(unknown_kwargs)}. Pass them through `context=ElasticRagProxyContext(...)` instead."
            )
        wire_context = context.to_wire() if context is not None else {}
        # ``size`` shortcut: only fills in if context didn't already carry one.
        resolved_size = size if size is not None else self._settings.default_top_k
        wire_context.setdefault("retriever_top_k", resolved_size)
        # Fallback semantics (strict ``is None``): an explicit scope is used
        # as-is; only a bare call falls back to the baked default. Deliberately
        # not ``narrow_with`` — that inherits boosts from ``self`` and would
        # strip the runtime boosts the KA tool injects on the explicit path.
        effective_scope = metadata_scope if metadata_scope is not None else self._default_scope
        envelope = await self._post_and_parse(
            input_payload=self._build_input_payload(query, effective_scope),
            wire_context=wire_context,
        )
        return SearchResponse(self._chunks_from_envelope(envelope))

    # --- DocumentProvider methods --------------------------------------------

    async def get_document(self, doc_id: str) -> list[RetrievalChunk]:
        """Fetch all chunks of a document, ordered by chunk_index ascending.

        See :class:`DocumentProvider` for the protocol contract. The gateway's
        backing retriever may not be configured for document retrieval — that
        surfaces here as ``NotImplementedError`` (via the
        ``document_provider_unavailable`` envelope code).
        """
        envelope = await self._post_and_parse(
            input_payload={"operation": "get_document", "doc_id": doc_id},
            wire_context={},
        )
        return self._chunks_from_envelope(envelope)

    async def get_chunk_context(self, chunk_id: str, window: int = 3) -> list[RetrievalChunk]:
        """Fetch neighbouring chunks in the range [chunk_index - window, chunk_index + window]."""
        envelope = await self._post_and_parse(
            input_payload={"operation": "get_chunk_context", "chunk_id": chunk_id, "window": window},
            wire_context={},
        )
        return self._chunks_from_envelope(envelope)

    async def get_chunk_range(self, doc_id: str, start_index: int, end_index: int) -> list[RetrievalChunk]:
        """Fetch chunks in [start_index, end_index] (inclusive), ordered ascending."""
        envelope = await self._post_and_parse(
            input_payload={
                "operation": "get_chunk_range",
                "doc_id": doc_id,
                "start_index": start_index,
                "end_index": end_index,
            },
            wire_context={},
        )
        return self._chunks_from_envelope(envelope)

    # --- SupportsBatchFetch methods ------------------------------------------

    async def get_documents(self, doc_ids: list[str]) -> dict[str, list[RetrievalChunk]]:
        """Fetch all chunks for multiple documents in one gateway call.

        Returns a mapping keyed by *every* requested doc id — an unknown
        document maps to an empty list. The proxy enforces this correlation
        guarantee even when the gateway response is light, so callers never
        have to disambiguate "absent" from "no results".
        """
        envelope = await self._post_and_parse(
            input_payload={"operation": "get_documents", "doc_ids": list(doc_ids)},
            wire_context={},
        )
        return self._documents_from_envelope(envelope, doc_ids)

    async def get_chunk_contexts(self, chunk_ids: list[str], window: int = 3) -> dict[str, list[RetrievalChunk]]:
        """Fetch neighbouring chunks around multiple anchor chunks.

        Anchor chunk ids are backing-retriever-specific (ES ``_id`` when the
        gateway is backed by ``ElasticRetriever``). The returned mapping
        carries every requested id; an anchor that cannot be resolved maps
        to an empty list.
        """
        envelope = await self._post_and_parse(
            input_payload={
                "operation": "get_chunk_contexts",
                "chunk_ids": list(chunk_ids),
                "window": window,
            },
            wire_context={},
        )
        return self._chunk_contexts_from_envelope(envelope, chunk_ids)

    async def get_chunk_ranges(self, ranges: list[ChunkRange]) -> dict[ChunkRange, list[RetrievalChunk]]:
        """Fetch multiple chunk ranges, keyed by ``(doc_id, start_index, end_index)``.

        Wire payload uses array-of-records so doc-ids with arbitrary characters
        (delimiters, whitespace, unicode) round-trip verbatim — a string-keyed
        encoding like ``"{doc_id}|{start}|{end}"`` would silently corrupt any
        doc id containing the separator.
        """
        chunk_ranges_payload = [{"doc_id": doc_id, "start_index": start, "end_index": end} for doc_id, start, end in ranges]
        envelope = await self._post_and_parse(
            input_payload={"operation": "get_chunk_ranges", "chunk_ranges": chunk_ranges_payload},
            wire_context={},
        )
        return self._chunk_ranges_from_envelope(envelope, ranges)

    # --- HTTP + envelope dispatch --------------------------------------------

    async def _post_and_parse(
        self,
        *,
        input_payload: dict[str, Any],
        wire_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a ``/runs/wait`` request and return the parsed envelope dict.

        Centralises the network/HTTP/error-translation/size-cap machinery so
        every operation (search + doc ops + future batch ops) shares the same
        wire layer. The returned dict is *always* a successful envelope —
        an ``ok=False`` envelope is translated into the documented exception
        before this returns.
        """
        body: dict[str, Any] = {
            "assistant_id": self._settings.assistant_id,
            "input": input_payload,
            "context": wire_context,
        }
        post_kwargs: dict[str, Any] = {"json": body}
        if self._settings.distributed_tracing:
            trace_headers = self._build_trace_headers()
            if trace_headers:
                post_kwargs["headers"] = trace_headers
        try:
            response = await self._http.post(_RUNS_WAIT_PATH, **post_kwargs)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            # ``from None`` suppresses the chained ``__cause__`` — otherwise
            # ``logger.exception(...)`` downstream would serialize the
            # underlying httpx exception's ``__str__()``, which for some
            # subclasses embeds the gateway URL. DEBUG-log the cause locally
            # for triage.
            logger.debug("elastic_rag gateway unreachable: %s: %s", type(exc).__name__, exc)
            raise RetrieverConnectionError(f"elastic_rag gateway unreachable ({type(exc).__name__})") from None
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            # ``httpx.HTTPStatusError.__str__()`` carries "for url '<full URL>'"
            # — ``from None`` keeps that out of the exception chain. Bodies
            # also may carry tokens, PII, or proxy interstitial HTML.
            logger.debug("elastic_rag gateway returned HTTP %d; body: %s", status, _safe_text(exc.response))
            if status in _RETRY_EXHAUSTED_STATUSES:
                raise RetrieverConnectionError(f"elastic_rag gateway unreachable after retries (HTTP {status})") from None
            raise RetrieverResponseError(f"elastic_rag gateway returned HTTP {status}") from None
        except httpx.RequestError as exc:
            # Fallback for the remaining ``httpx.RequestError`` subclasses
            # (``TooManyRedirects``, ``DecodingError``, ``UnsupportedProtocol``,
            # ``ProxyError``, ``InvalidURL``). Some of these subclasses also
            # embed a URL in ``__str__()`` (``InvalidURL`` notably) — same
            # ``from None`` + DEBUG-log treatment as above.
            logger.debug("elastic_rag gateway request failed: %s: %s", type(exc).__name__, exc)
            raise RetrieverConnectionError(f"elastic_rag gateway request failed ({type(exc).__name__})") from None
        self._enforce_response_size_cap(response, self._settings.max_response_bytes)
        envelope = self._parse_envelope(response)
        self._log_gateway_warnings(envelope)
        return envelope

    async def close(self) -> None:
        """Close the underlying HTTP client. Safe to call repeatedly.

        Only marks the retriever closed on a successful underlying close so a
        leaked ``httpx.AsyncClient`` doesn't go silently undetected. A close
        failure surfaces to the caller, but is also logged so shutdown errors
        are visible even if the caller swallows them. ``CancelledError``
        propagates unchanged so cooperative cancellation isn't smothered.
        """
        if self._closed:
            return
        try:
            await self._http.close()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("ElasticRagRetriever.close failed: %s: %s", type(exc).__name__, exc)
            raise
        else:
            self._closed = True

    # --- Internals -----------------------------------------------------------

    @staticmethod
    def _build_trace_headers() -> dict[str, str]:
        """Return LangSmith trace headers for the current run tree, or ``{}``.

        Reads the active LangSmith run tree (set by an enclosing ``@traceable``
        scope or a LangGraph node trace) and returns the headers produced by
        ``RunTree.to_headers()`` — typically ``langsmith-trace`` (trace ID +
        dotted-order parent identifier) and ``baggage`` (W3C baggage carrying
        the project name and any other propagated metadata). The LangSmith
        run-ingest layer reads these from incoming HTTP requests to stitch the
        downstream run as a child of the caller's run.

        Mirrors the propagation convention used by
        ``langgraph.pregel.remote.RemoteGraph(distributed_tracing=True)``,
        which merges the same headers into the outgoing HTTP request.

        Returns ``{}`` if ``langsmith`` is not importable, if no run tree is
        active at call time, or if ``to_headers()`` returns no string values.
        """
        try:
            from langsmith.run_helpers import get_current_run_tree
        except ImportError:
            logger.debug("distributed_tracing on but `langsmith` not importable; sending no trace headers")
            return {}
        rt = get_current_run_tree()
        if rt is None:
            logger.debug("distributed_tracing on but no active LangSmith run tree at search() call; sending no trace headers")
            return {}
        raw = rt.to_headers()
        # ``to_headers()`` may include keys with ``None`` values; httpx rejects
        # those, so filter to string-valued entries only.
        out = {k: v for k, v in raw.items() if isinstance(v, str) and v}
        if not out:
            logger.debug("distributed_tracing on, run tree present, but RunTree.to_headers() produced no string headers; sending none")
        return out

    @staticmethod
    def _build_input_payload(
        query: str,
        metadata_scope: MetadataScope | None,
    ) -> dict[str, Any]:
        """Flatten scope fields into the ``input`` body alongside ``query``.

        The gateway re-builds its own ``MetadataScope`` from these flat
        fields, so wire format stays identical to today's direct callers.
        """
        payload: dict[str, Any] = {"query": query}
        if metadata_scope is not None:
            # ``exclude_defaults=True`` drops the two bool widening flags
            # (``include_entity_childs``/``include_transversal``) when they
            # equal their default ``False`` — the gateway's
            # ``_build_metadata_scope`` re-constructs an empty scope from a
            # missing key the same way, so this is wire-shape minimization,
            # not behavior change.
            payload.update(metadata_scope.model_dump(exclude_none=True, exclude_defaults=True))
        return payload

    @staticmethod
    def _enforce_response_size_cap(response: httpx.Response, cap: int) -> None:
        """Reject responses whose body exceeds ``cap`` bytes.

        Checks ``Content-Length`` first when present; otherwise falls back to
        ``len(response.content)`` (httpx buffers chunked bodies on read, so
        this catches the chunked case once the read has completed). A rogue
        or compromised gateway returning a multi-GB body would otherwise
        buffer entirely in memory before reaching the JSON parser.
        """
        content_length_hdr = response.headers.get("Content-Length")
        if content_length_hdr:
            try:
                declared = int(content_length_hdr)
            except ValueError:
                declared = -1
            if declared > cap:
                raise RetrieverResponseError(f"elastic_rag gateway response Content-Length {declared} exceeds cap {cap}")
        actual = len(response.content)
        if actual > cap:
            raise RetrieverResponseError(f"elastic_rag gateway response body length {actual} exceeds cap {cap}")

    @classmethod
    def _parse_envelope(cls, response: httpx.Response) -> dict[str, Any]:
        """Parse the gateway response into the success-branch envelope dict.

        Accepts both the legacy search-only shape (``{"query":..., "results":[...]}``)
        and the new discriminated envelope (``{"operation","ok","results","warnings"}``)
        so the proxy keeps working against gateways that haven't been updated yet.

        On an explicit failure envelope (``ok=False``) maps ``error.code`` to the
        Python exception documented in the design and raises. Unknown codes fall
        back to ``RetrieverResponseError`` so future gateway-side additions don't
        leak through as raw envelope dicts.
        """
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug("elastic_rag gateway returned non-JSON body: %s", _safe_text(response))
            raise RetrieverResponseError("elastic_rag gateway returned non-JSON body") from exc
        if not isinstance(payload, dict):
            payload_shape = type(payload).__name__
            logger.debug("elastic_rag gateway response is not a JSON object; payload_shape=%s", payload_shape)
            raise RetrieverResponseError(f"elastic_rag gateway response must be a JSON object (got {payload_shape})")

        # Failure envelope path — translate ``code`` into the documented
        # exception and raise before the caller can mistake an empty
        # ``results`` for a successful empty fetch.
        if payload.get("ok") is False:
            error = payload.get("error")
            if not isinstance(error, dict):
                # Malformed envelope: ok=False without a structured error.
                raise RetrieverResponseError("elastic_rag gateway returned ok=false without a structured error payload")
            code = error.get("code")
            message = error.get("message") or f"gateway returned error code={code!r}"
            operation = error.get("operation") or payload.get("operation") or "<unknown>"
            exc_cls = cls._ERROR_CODE_TO_EXCEPTION.get(str(code), RetrieverResponseError)
            raise exc_cls(f"elastic_rag {operation}: {message}")

        # Success branch — legacy and new envelope shapes both must carry a
        # ``results`` channel. Reject anything else so a 200-but-malformed
        # response doesn't surface as a silent empty fetch.
        if "results" not in payload:
            # Don't include the raw payload anywhere — even at DEBUG. A
            # 200-but-malformed response (echoed prompts, leaked auth tokens)
            # would otherwise surface in any handler that picks up DEBUG
            # records. Emit only the shape so an operator can triage without
            # seeing values.
            payload_shape = list(payload.keys())
            logger.debug("elastic_rag gateway response missing 'results'; payload_shape=%s", payload_shape)
            raise RetrieverResponseError("elastic_rag gateway response missing 'results' key")
        return payload

    @staticmethod
    def _chunks_from_envelope(envelope: dict[str, Any]) -> list[RetrievalChunk]:
        """Deserialize a single-fetch ``results`` list (search + DocumentProvider).

        Batch operations have grouped shapes and use the dedicated
        ``_documents_from_envelope`` / ``_chunk_contexts_from_envelope`` /
        ``_chunk_ranges_from_envelope`` parsers below.
        """
        results_raw = envelope["results"]
        if not isinstance(results_raw, list):
            raise RetrieverResponseError(f"elastic_rag gateway 'results' must be a list (got {type(results_raw).__name__})")
        return [_chunk_from_dict(item) for item in results_raw]

    @staticmethod
    def _documents_from_envelope(envelope: dict[str, Any], requested_ids: list[str]) -> dict[str, list[RetrievalChunk]]:
        """Parse a ``get_documents`` envelope into ``dict[doc_id, list[chunk]]``.

        Enforces correlation: every ``requested_ids`` entry must be a key in
        the gateway response. A missing key indicates a gateway that violates
        the contract (or a bug in the proxy's payload assembly); raising
        ``RetrieverResponseError`` keeps the caller's invariant intact rather
        than silently dropping items.
        """
        results_raw = envelope["results"]
        if not isinstance(results_raw, dict):
            raise RetrieverResponseError(f"elastic_rag gateway 'results' must be a dict for get_documents (got {type(results_raw).__name__})")
        missing = [doc_id for doc_id in requested_ids if doc_id not in results_raw]
        if missing:
            raise RetrieverResponseError(f"elastic_rag gateway response missing {len(missing)} requested doc_id(s); correlation guarantee broken")
        out: dict[str, list[RetrievalChunk]] = {}
        for doc_id in requested_ids:
            chunks_raw = results_raw[doc_id]
            if not isinstance(chunks_raw, list):
                raise RetrieverResponseError(f"elastic_rag gateway results[{doc_id!r}] must be a list (got {type(chunks_raw).__name__})")
            out[doc_id] = [_chunk_from_dict(item) for item in chunks_raw]
        return out

    @staticmethod
    def _chunk_contexts_from_envelope(
        envelope: dict[str, Any],
        requested_ids: list[str],
    ) -> dict[str, list[RetrievalChunk]]:
        """Parse a ``get_chunk_contexts`` envelope into ``dict[chunk_id, list[chunk]]``.

        Same correlation enforcement as ``_documents_from_envelope``.
        """
        results_raw = envelope["results"]
        if not isinstance(results_raw, dict):
            raise RetrieverResponseError(f"elastic_rag gateway 'results' must be a dict for get_chunk_contexts (got {type(results_raw).__name__})")
        missing = [chunk_id for chunk_id in requested_ids if chunk_id not in results_raw]
        if missing:
            raise RetrieverResponseError(f"elastic_rag gateway response missing {len(missing)} requested chunk_id(s); correlation guarantee broken")
        out: dict[str, list[RetrievalChunk]] = {}
        for chunk_id in requested_ids:
            chunks_raw = results_raw[chunk_id]
            if not isinstance(chunks_raw, list):
                raise RetrieverResponseError(f"elastic_rag gateway results[{chunk_id!r}] must be a list (got {type(chunks_raw).__name__})")
            out[chunk_id] = [_chunk_from_dict(item) for item in chunks_raw]
        return out

    @staticmethod
    def _chunk_ranges_from_envelope(
        envelope: dict[str, Any],
        requested_ranges: list[ChunkRange],
    ) -> dict[ChunkRange, list[RetrievalChunk]]:
        """Parse a ``get_chunk_ranges`` envelope into ``dict[ChunkRange, list[chunk]]``.

        Input wire shape is array-of-records:
        ``[{doc_id, start_index, end_index, chunks: list[chunk_dict]}, ...]``.
        Re-keys by the requested tuple so doc-ids with arbitrary characters
        survive the round-trip. Every requested range must be present in the
        response; absence indicates a contract violation and raises.
        """
        results_raw = envelope["results"]
        if not isinstance(results_raw, list):
            raise RetrieverResponseError(f"elastic_rag gateway 'results' must be a list for get_chunk_ranges (got {type(results_raw).__name__})")
        response_by_tuple: dict[ChunkRange, list[Any]] = {}
        for idx, record in enumerate(results_raw):
            if not isinstance(record, dict):
                raise RetrieverResponseError(f"elastic_rag gateway results[{idx}] must be a dict (got {type(record).__name__})")
            try:
                tup: ChunkRange = (
                    str(record["doc_id"]),
                    int(record["start_index"]),
                    int(record["end_index"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RetrieverResponseError(f"elastic_rag gateway results[{idx}] missing/invalid doc_id/start_index/end_index") from exc
            chunks_raw = record.get("chunks", [])
            if not isinstance(chunks_raw, list):
                raise RetrieverResponseError(f"elastic_rag gateway results[{idx}].chunks must be a list (got {type(chunks_raw).__name__})")
            response_by_tuple[tup] = chunks_raw

        missing = [r for r in requested_ranges if r not in response_by_tuple]
        if missing:
            raise RetrieverResponseError(f"elastic_rag gateway response missing {len(missing)} requested range(s); correlation guarantee broken")
        return {r: [_chunk_from_dict(item) for item in response_by_tuple[r]] for r in requested_ranges}

    @staticmethod
    def _log_gateway_warnings(envelope: dict[str, Any]) -> None:
        """Pass gateway-emitted warnings through to the proxy logger.

        Warnings are advisory (truncation hints, capacity caps); they don't
        change the success/failure shape of the envelope. The KA consumes the
        proxy through ``FetchExecutor`` which has no warning channel, so
        ``logger.warning`` is the only surface — mirrors the direct retriever's
        own truncation behavior (``ElasticRetriever.get_documents`` logs and
        returns).
        """
        warnings_raw = envelope.get("warnings")
        if not warnings_raw:
            return
        if not isinstance(warnings_raw, list):
            logger.debug("elastic_rag gateway 'warnings' not a list; got %s", type(warnings_raw).__name__)
            return
        for entry in warnings_raw:
            if not isinstance(entry, dict):
                continue
            logger.warning(
                "elastic_rag gateway warning: code=%s message=%s details=%s",
                entry.get("code"),
                entry.get("message"),
                entry.get("details"),
            )


def _chunk_from_dict(item: Any) -> RetrievalChunk:
    if not isinstance(item, dict):
        raise RetrieverResponseError(f"chunk must be a dict, got {type(item).__name__}")
    # ``... or ""`` (not ``.get(k, "")``) is load-bearing: a present ``null``
    # value would otherwise become the string ``"None"`` via ``str(None)``.
    meta = item.get("metadata")
    if meta is not None and not isinstance(meta, dict):
        # Reject non-mapping metadata explicitly — ``dict(non_mapping)`` would
        # leak a raw ``TypeError`` past the adapter boundary.
        raise RetrieverResponseError(f"chunk 'metadata' must be a mapping or null, got {type(meta).__name__}")
    return RetrievalChunk(
        content=str(item.get("content") or ""),
        chunk_id=str(item.get("chunk_id") or ""),
        score=_safe_float(item.get("score")),
        source_url=str(item.get("source_url") or ""),
        retriever_type=str(item.get("retriever_type") or ""),
        metadata=dict(meta) if meta else {},
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_text(response: httpx.Response, limit: int = 500) -> str:
    try:
        text = response.text
    except Exception:  # noqa: BLE001 — best-effort error rendering
        return "<unreadable response body>"
    return text if len(text) <= limit else text[:limit] + "…"

-------

packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/elasticsearch/elastic_retriever.py
----
"""Elasticsearch hybrid retriever combining BM25 and kNN vector search.

This module provides a generic, configurable retriever for Elasticsearch that supports
multiple fusion strategies for combining lexical and semantic search results.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar

import httpx
from langchain_core.embeddings import Embeddings


if TYPE_CHECKING:
    # Provide identity-decorator stub so pyright sees the raw function signature
    # instead of langsmith's SupportsLangsmithExtra wrapper.
    _F = TypeVar("_F", bound=Callable[..., Any])

    def traceable(**kwargs: Any) -> Callable[[_F], _F]: ...
else:
    from langsmith import traceable

from ....adapters.elasticsearch.adapters_async import AsyncElasticsearchAdapter
from ....models.rerank_client import RerankClient, RerankResponse, RerankResult
from ..batch_document_provider import ChunkRange
from ..search_response import SearchResponse
from .elastic_retrieval_chunk import ElasticRetrievalChunk
from .elastic_search_config import (
    ElasticFieldConfig,
    ElasticSearchConfig,
    FusionStrategy,
    FusionStrategyLiteral,
)
from .fusion import (
    FusionOperator,
    PositionAwareBlend,
    RrfRerankerOperator,
    SubQuery,
    TopRankBonusRRF,
    WeightedRRF,
    resolve_fusion_operator,
)
from .metadata_scope import MetadataScope
from .query_expansion import ExpansionStrategy, QueryExpanderProtocol


logger = logging.getLogger(__name__)


class RerankUnavailableError(Exception):
    """Raised internally when a runtime reranking step fails after its own retries.

    Signals a *soft* failure: the rerank HTTP call (which already retries via
    ``RerankClient.arerank``) ultimately errored, so ``search_many`` should
    degrade to rerank-blind RRF fusion rather than return zero results. The
    original exception is chained (``from``) and the underlying ``reranker_arerank``
    LangSmith span is still recorded as failed — this wrapper only controls the
    fallback at the retriever boundary, it does not hide the failure from traces.

    This is distinct from the hard ``ValueError`` raised when a rerank-aware
    fusion operator is selected but no ``RerankClient`` is configured — that is a
    wiring bug and must surface, not silently degrade.
    """


# Anchor that terminates the contextual prefix in ``metadata.content``.
# Production folds a contextual summary plus a *variable* set of metadata fields
# into the single embedded ``content`` field and closes the prefix with a
# ``\n\nContent:`` marker (capital ``C``, blank-line separated). The parser keys
# on this anchor alone — never on a particular field name — so a changing field
# set doesn't break body recovery.
_BODY_ANCHOR = "\n\nContent:"

# Legacy anchor for the original prod-shaped template
# (see ``infra/elasticsearch/ingestion/chunker.build_structured_content`` and
# § 4 of ``creative_phase_2026-05-15_es_mapping_alignment.md``). ``rfind`` is
# load-bearing: a chunk body that happens to contain ``\ncontent: `` literals
# stays correctly anchored on the LAST occurrence.
_STRUCTURED_BODY_ANCHOR = "\ncontent: "

# Upper bound on chunks returned by a single batched fetch query
# (get_documents / get_chunk_ranges). Matches the single-document ceiling in
# get_document; a batch spanning more chunks is truncated with a warning.
_BATCH_FETCH_MAX_CHUNKS = 10_000


def parse_structured_content(structured_content: str, anchor: str = _BODY_ANCHOR) -> tuple[str, str]:
    """Split a structured ``metadata.content`` blob into ``(context_summary, body)``.

    Production folds a contextual summary plus a *variable* set of metadata
    fields into ``metadata.content``, terminating that prefix with ``anchor``
    (default ``\\n\\nContent:``). Everything after the FINAL anchor is the
    per-chunk body; everything before it is the per-page contextual prefix —
    identical across every chunk of one page. Splitting here lets the prefix be
    surfaced ONCE per page instead of repeating inside each chunk's body.

    Resolution order (last-marker ``rfind`` semantics throughout):
      1. ``anchor`` — the production ``\\n\\nContent:`` marker; one single
         leading space after the marker is dropped (a second space belongs to
         the body, e.g. indented content).
      2. Legacy ``\\ncontent: `` (lowercase, single newline, trailing space) —
         the original template; emits a ``logger.debug`` on multi-anchor clips.
      3. No anchor → ``("", structured_content)`` so un-templated / legacy raw
         indices pass through unchanged.
    """
    last = structured_content.rfind(anchor)
    if last >= 0:
        summary = structured_content[:last]
        body_start = last + len(anchor)
        if body_start < len(structured_content) and structured_content[body_start] == " ":
            body_start += 1
        return summary, structured_content[body_start:]

    last = structured_content.rfind(_STRUCTURED_BODY_ANCHOR)
    if last < 0:
        return "", structured_content
    first = structured_content.find(_STRUCTURED_BODY_ANCHOR)
    if first != last:
        logger.debug(
            "parse_structured_content: multiple '\\ncontent: ' anchors found "
            "(first=%d, last=%d); body recovery clipped on the last one — "
            "the body may have shadowed the template anchor",
            first,
            last,
        )
    return structured_content[:last], structured_content[last + len(_STRUCTURED_BODY_ANCHOR) :]


def extract_chunk_body(structured_content: str) -> str:
    """Pull the chunk body out of a structured ``metadata.content`` blob.

    Thin wrapper over :func:`parse_structured_content` that returns only the
    body. Recognizes both the production ``\\n\\nContent:`` and legacy
    ``\\ncontent: `` anchors and falls back to the full blob when neither is
    present, so legacy / un-templated indices keep working.
    """
    return parse_structured_content(structured_content)[1]


# Display-style timestamp formats observed in production for
# ``lastDocIngestion`` (Kibana-like ``"May 12, 2025 @ 10:30:00"``), with and
# without fractional seconds / full month names.
_INGESTION_TIMESTAMP_FORMATS = (
    "%b %d, %Y @ %H:%M:%S.%f",
    "%b %d, %Y @ %H:%M:%S",
    "%B %d, %Y @ %H:%M:%S.%f",
    "%B %d, %Y @ %H:%M:%S",
)


def parse_ingestion_timestamp(value: Any) -> datetime | None:
    """Parse a document-ingestion timestamp into an aware UTC datetime.

    Accepts the production display format (``"May 12, 2025 @ 10:30:00"``),
    ISO 8601 strings, and epoch milliseconds. Returns ``None`` when the value
    is missing or unparseable — callers keep the raw value in that case rather
    than dropping it, so no information is lost at the mapping boundary.
    Timezone-naive inputs are assumed UTC.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000.0, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in _INGESTION_TIMESTAMP_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _join_rerank_query(
    domain_intent: str | None,
    intent: str | None,
    query: str,
) -> str:
    """Build the intent-prepended rerank query (F5b-3).

    Canonical order per the planning doc: ``[domain_intent, intent, query]``
    joined by newlines, empty/None layers dropped. Returns just ``query``
    when neither intent layer is configured — preserves pre-F5b behavior.
    """
    return "\n".join(filter(None, [domain_intent, intent, query]))


def _process_retriever_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process inputs for LangSmith trace display."""
    return {
        "query": inputs.get("query", ""),
        "size": inputs.get("size", 10),
        "fusion_strategy": str(inputs.get("fusion_strategy", "default")),
        "documents_filter": inputs.get("documents"),
    }


def _process_search_many_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process inputs for LangSmith trace display (search_many)."""
    queries = inputs.get("queries")
    if isinstance(queries, str):
        preview: Any = queries
    elif isinstance(queries, list):
        preview = [{"type": sq.type, "query": sq.query, "weight": sq.weight} if hasattr(sq, "type") else sq for sq in queries]
    else:
        preview = None
    return {
        "queries": preview,
        "size": inputs.get("size", 10),
        "fusion": type(inputs.get("fusion")).__name__ if inputs.get("fusion") is not None else "WeightedRRF(default)",
        "documents_filter": inputs.get("documents"),
    }


def _process_embedding_outputs(outputs: Any) -> dict[str, Any]:
    """Process embedding outputs for LangSmith trace display."""
    if outputs is None:
        return {"dimensions": 0}
    if isinstance(outputs, list):
        return {"dimensions": len(outputs)}
    return {"dimensions": 0}


def _process_retriever_outputs(outputs: Any) -> dict[str, Any]:
    """Convert retrieval results to LangSmith Document format."""
    results = outputs.results if hasattr(outputs, "results") else (outputs if isinstance(outputs, list) else [])
    if not results:
        return {"documents": []}

    exclude_from_metadata = {"content", "text", "page_content"}
    docs = []
    for result in results:
        if hasattr(result, "content") and hasattr(result, "metadata"):
            metadata = {k: v for k, v in (result.metadata or {}).items() if k not in exclude_from_metadata}
            metadata["score"] = getattr(result, "score", 0.0)
            docs.append({"page_content": result.content, "type": "Document", "metadata": metadata})
        elif isinstance(result, dict):
            docs.append(
                {
                    "page_content": result.get("content", str(result)),
                    "type": "Document",
                    "metadata": {
                        "score": result.get("score", 0.0),
                        **{k: v for k, v in result.items() if k not in exclude_from_metadata | {"score"}},
                    },
                }
            )
        else:
            docs.append({"page_content": str(result), "type": "Document", "metadata": {}})
    return {"documents": docs}


def _process_get_document_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process inputs for LangSmith trace (elastic_get_document)."""
    return {"doc_id": inputs.get("doc_id", "")}


def _process_get_document_outputs(outputs: Any) -> dict[str, Any]:
    """Process outputs for LangSmith trace (elastic_get_document)."""
    chunks = outputs if isinstance(outputs, list) else []
    titles = [((getattr(c, "metadata", None) or {}).get("title") or "")[:80] for c in chunks[:5]]
    return {"num_chunks": len(chunks), "titles_preview": titles}


def _process_get_chunk_context_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process inputs for LangSmith trace (elastic_get_chunk_context)."""
    return {"chunk_id": inputs.get("chunk_id", ""), "window": inputs.get("window", 3)}


def _process_get_chunk_range_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process inputs for LangSmith trace (elastic_get_chunk_range)."""
    return {
        "doc_id": inputs.get("doc_id", ""),
        "start_index": inputs.get("start_index", 0),
        "end_index": inputs.get("end_index", 0),
    }


class ElasticRetriever:
    """Elasticsearch hybrid retriever combining BM25 text search with kNN vector search.

    Implements the BaseRetriever protocol via structural typing.
    search() returns SearchResponse[ElasticRetrievalChunk].
    search_vector_only and search_text_only remain as public methods for Elastic-specific use.

    Use ElasticFieldConfig to adapt to different index schemas (field names, boosts).

    Also satisfies ``SupportsMetadataScope`` — ``search()`` honors the
    ``metadata_scope: MetadataScope | None`` kwarg used by the KA's build-time
    + runtime-query metadata scope. Backends that don't honor it must NOT set
    this marker; the KA tool factory raises at build time when scope features
    are wired onto an unsupporting retriever.
    """

    # SupportsMetadataScope marker — see scope_capability.py and the KA
    # tool factory's build-time gate. Subclasses inherit by default; a
    # wrapper that intentionally drops scope handling must override to False.
    supports_metadata_scope: ClassVar[Literal[True]] = True

    @staticmethod
    def resolve_caller_scope(bundle: Mapping[str, Any]) -> MetadataScope | None:
        """Resolve ``doc_ids``, ``apcode``, ``app_name``, and ``entity`` filters."""
        return MetadataScope.from_caller_scope(bundle)

    # SupportsBatchFetch marker — see batch_document_provider.py. Advertises
    # the batched get_documents / get_chunk_ranges / get_chunk_contexts
    # methods. Like DocumentProvider, the methods still require page_id_field
    # and chunk_index_field to be configured (see supports_document_provider).
    supports_batch_fetch: ClassVar[Literal[True]] = True

    def __init__(
        self,
        adapter: AsyncElasticsearchAdapter,
        index: str,
        embedding_model: Embeddings,
        reranker: RerankClient | None = None,
        field_config: ElasticFieldConfig | None = None,
        search_config: ElasticSearchConfig | None = None,
        embedding_http_client: httpx.AsyncClient | None = None,
        expander: QueryExpanderProtocol | None = None,
        domain_intent: str | None = None,
        default_scope: MetadataScope | None = None,
    ) -> None:
        self.adapter = adapter
        self.index = index
        self.embedding_model = embedding_model
        self._reranker = reranker
        self._embedding_http_client = embedding_http_client
        self.field_config = field_config or ElasticFieldConfig()
        self._search_config = search_config or ElasticSearchConfig()
        # Build-time filter ceiling baked in by the consumer (KA
        # ``RetrieverEntry``). A bare call to any scope-bearing public method
        # falls back to it via ``_effective_scope`` so the ceiling is enforced
        # client-side even when the caller passes no per-call scope. The KA tool
        # always passes a non-None merged scope, so its path is unaffected.
        self._default_scope = default_scope
        # Optional expander — None keeps the retriever on the no-expansion path.
        # Runtime use lands in Cycle E (uniform search() pipeline); for now the
        # retriever only *carries* the expander so factories can inspect it.
        self._expander = expander
        # Build-time "what is this index about" — set once, threaded into every
        # expansion call as the ``domain_intent`` kwarg. Runtime per-call intent
        # lives on ``ElasticSearchConfig.intent`` and flows through
        # ``ElasticRetrieverContext.retriever_intent`` instead (Cycle F5).
        self._domain_intent = domain_intent
        # Lazy license probe cache: None = not yet probed, bool = cached decision.
        self._native_rrf_available: bool | None = None

    def _effective_scope(self, metadata_scope: MetadataScope | None) -> MetadataScope | None:
        """Resolve the per-call scope against the baked build-time default.

        Fallback semantics, strict ``is None``: an explicit ``metadata_scope``
        (non-``None``) is used as-is; only a bare call falls back to
        ``self._default_scope``. Deliberately NOT ``narrow_with`` — that
        intersects filters and inherits boosts from ``self`` only, which would
        strip the runtime boosts the KA tool injects on the explicit-scope path.

        Applied at the top of every scope-bearing public method, so no public
        entry point bypasses the ceiling. Idempotent under ``search()``'s
        delegation to ``search_many`` / ``search_text_only``: once a non-None
        scope is resolved it passes straight through downstream re-resolution.
        """
        return metadata_scope if metadata_scope is not None else self._default_scope

    # License types that grant access to Elastic's native RRF retriever query.
    # basic + missing license → Python RRF fallback.
    _RRF_LICENSED_TYPES = frozenset({"platinum", "enterprise", "trial"})

    async def _can_use_native_rrf(self) -> bool:
        """Decide whether to use Elastic's native `retriever.rrf` query.

        Modes (from ``ElasticSearchConfig.es_rrf_mode``):
            native — force native (no probe, caller owns license guarantee).
            python — force in-process RRF (skip probe entirely).
            auto   — probe ``/_license`` once on first call; cache the result.
                     Any probe error (network, auth, missing API) falls back to
                     Python RRF with a single warning.

        Returns:
            True if native RRF should be used, False for Python fallback.
        """
        mode = self._search_config.es_rrf_mode
        if mode == "native":
            return True
        if mode == "python":
            return False

        # mode == "auto" — lazy probe with instance cache
        if self._native_rrf_available is not None:
            return self._native_rrf_available

        try:
            response = await self.adapter.client.license.get()
            license_info = response.get("license", {}) if isinstance(response, dict) else {}
            license_type = str(license_info.get("type", "")).lower()
            available = license_type in self._RRF_LICENSED_TYPES
        except Exception as exc:
            logger.warning(
                "Elastic license probe failed (%s); falling back to Python RRF for this retriever instance.",
                exc.__class__.__name__,
            )
            available = False
        else:
            if not available:
                logger.warning(
                    "Elastic license type %r does not grant native RRF; using Python RRF fallback.",
                    license_type or "<unknown>",
                )

        self._native_rrf_available = available
        return available

    @property
    def search_config(self) -> ElasticSearchConfig:
        """Get the instance search configuration."""
        return self._search_config

    @property
    def reranker(self) -> RerankClient | None:
        """Get the reranker client if configured."""
        return self._reranker

    @property
    def expander(self) -> QueryExpanderProtocol | None:
        """Get the query expander if configured.

        Factories call this to validate that a non-PASS ``expansion_hint`` is
        paired with a wired expander (per v3 amendment §3.5)."""
        return self._expander

    @property
    def domain_intent(self) -> str | None:
        """Build-time "what this index is about" — threaded into every ``expand()``
        call as the ``domain_intent`` kwarg. Runtime per-call intent comes from
        ``ElasticSearchConfig.intent`` / ``ElasticRetrieverContext.retriever_intent``.
        """
        return self._domain_intent

    # ---- Cycle F3 — AUTO BM25 probe ----------------------------------------

    @staticmethod
    def _chunk_score(chunk: ElasticRetrievalChunk) -> float:
        """Safe score accessor — ``None`` treated as 0.0."""
        return chunk.score if chunk.score is not None else 0.0

    def _is_strong_signal(
        self,
        fts_results: list[ElasticRetrievalChunk],
        cfg: ElasticSearchConfig,
    ) -> bool:
        """Reuses the base FTS results — no extra ES round-trip.

        Strong signal iff the top hit clears ``auto_probe_min_score`` AND
        the gap to the runner-up clears ``auto_probe_min_gap``. Both
        conditions must hold — a high single score with a tight cluster
        below it is ambiguous, not decisive.

        With the default ``+inf`` thresholds this function returns ``False``
        for every finite score, so AUTO always resolves to MULTI until F6
        calibrates per-corpus values.
        """
        if not fts_results:
            return False
        top = self._chunk_score(fts_results[0])
        second = self._chunk_score(fts_results[1]) if len(fts_results) > 1 else 0.0
        return top >= cfg.auto_probe_min_score and (top - second) >= cfg.auto_probe_min_gap

    def _resolve_auto_hint(
        self,
        hint: ExpansionStrategy,
        base_fts_results: list[ElasticRetrievalChunk],
        cfg: ElasticSearchConfig,
    ) -> ExpansionStrategy:
        """Map AUTO → PASS (strong signal) or MULTI (weak signal).

        Non-AUTO hints pass through untouched — the retriever never
        "upgrades" an explicit PASS or a specific strategy.
        """
        if hint != ExpansionStrategy.AUTO:
            return hint
        if self._is_strong_signal(base_fts_results, cfg):
            return ExpansionStrategy.PASS
        return ExpansionStrategy.MULTI

    @staticmethod
    def _rerank_document_text(chunk: ElasticRetrievalChunk) -> str:
        """Build the text handed to the cross-encoder for one candidate.

        The reranker must score on the SAME signal BM25 matches on. BM25
        queries ``content_field`` — the full structured ``metadata.content``
        blob (contextual summary + Url/Application/apcode/appName/title block +
        body). The result mapper splits that blob into a clean ``content``
        (body) plus a ``context_summary`` (the prefix); passing only ``content``
        to ``arerank`` would blind the cross-encoder to the metadata text that
        often disambiguates which app/entity/page a generic-looking chunk
        belongs to.

        So reconstruct the full signal: prepend ``context_summary`` (when the
        mapper recovered it) to the body. Byte-exactness with the original blob
        is irrelevant to a cross-encoder — the metadata *text* being present is
        what matters. Legacy / un-templated indices (no recovered prefix) fall
        back to the body alone, exactly as before.
        """
        context_summary = chunk.metadata.get("context_summary")
        if context_summary:
            return f"{context_summary}\n\n{chunk.content}"
        return chunk.content

    @staticmethod
    def _apply_rerank_response(
        rerank_response: RerankResponse,
        candidates: list[ElasticRetrievalChunk],
    ) -> Iterator[tuple[int, RerankResult]]:
        """Yield ``(idx, result)`` for each VALID rerank result, dropping bad rows.

        Defensive contract shared by both rerank sites (`_execute_search`'s
        `RRF_RERANKER` case and `_maybe_build_rerank_scores`). Cross-encoder
        providers occasionally return out-of-range or duplicate indices; this
        helper logs and skips them so callers don't IndexError or double-score.
        """
        seen: set[int] = set()
        pool_size = len(candidates)
        for r in rerank_response.results:
            if not (0 <= r.index < pool_size):
                logger.warning(
                    "rerank: out-of-range index %d (pool size %d) — skipping",
                    r.index,
                    pool_size,
                )
                continue
            if r.index in seen:
                logger.warning(
                    "rerank: duplicate index %d — keeping first score, skipping duplicate",
                    r.index,
                )
                continue
            seen.add(r.index)
            yield r.index, r

    @traceable(
        run_type="embedding",
        name="query_embedding",
        tags=["embedding"],
        process_outputs=_process_embedding_outputs,
    )
    async def _get_query_embedding(self, query: str) -> list[float]:
        embeddings = await self.embedding_model.aembed_documents([query])
        return embeddings[0]

    def _default_result_mapper(self, hit: dict[str, Any]) -> ElasticRetrievalChunk:
        source = hit.get("_source", {})
        score = hit.get("_score", 0.0) or 0.0

        # Display content:
        #   1. If ``display_content_field`` is configured, read that directly
        #      (legacy dual-content indices with a separate raw-display field).
        #   2. Otherwise read ``content_field`` (prod = ``metadata.content``,
        #      the structured BM25 blob) and run ``extract_chunk_body()`` to
        #      recover the body for display. Falls back to the full blob when
        #      the prod template anchor is absent (legacy un-templated docs).
        context_summary = ""
        if self.field_config.display_content_field:
            content = self._get_nested_field(source, self.field_config.display_content_field, "")
        else:
            raw = self._get_nested_field(source, self.field_config.content_field, "")
            if isinstance(raw, str):
                context_summary, content = parse_structured_content(raw, self.field_config.content_body_anchor)
            else:
                content = raw

        metadata: dict[str, Any] = {}
        metadata["title"] = self._get_nested_field(source, self.field_config.title_field, "Untitled")
        metadata["doc"] = self._get_nested_field(source, self.field_config.doc_field, "")
        url = self._get_nested_field(source, self.field_config.url_field, "")
        metadata["url"] = url
        # Per-page contextual prefix recovered from the structured blob. Surfaced
        # once per page by the Knowledge Agent rather than repeated inside every
        # chunk body. Identical across all chunks of a page; absent for
        # un-templated / legacy raw content (no anchor → empty summary).
        if context_summary:
            metadata["context_summary"] = context_summary

        # Per-chunk contextual summary, surfaced under the canonical snake_case
        # key. Production stores the same summary BOTH as the leading prefix of
        # ``content_field`` (inside ``context_summary`` above) AND verbatim in a
        # dedicated field (``metadata.contextualisedContent``). Normalizing it to
        # ``contextualized_content`` lets the Knowledge Agent render the per-chunk
        # summary INSIDE each ``<chunk>`` while keeping the page-shared
        # Url/Application/apcode/title block in ``context_summary`` once per page.
        # The raw backend leaf is intentionally NOT suppressed (unlike auid →
        # apcode): ``contextualisedContent`` is already a surfaced, consumer-read
        # key via the metadata tail-merge — removing it would break callers.
        if self.field_config.contextualized_content_field:
            contextualized = self._get_nested_field(source, self.field_config.contextualized_content_field, None)
            if isinstance(contextualized, str) and contextualized:
                metadata["contextualized_content"] = contextualized

        # Extract extended metadata when field paths are configured
        if self.field_config.page_id_field:
            page_id = self._get_nested_field(source, self.field_config.page_id_field, None)
            if page_id is not None:
                metadata["pageId"] = page_id
        if self.field_config.chunk_index_field:
            chunk_idx = self._get_nested_field(source, self.field_config.chunk_index_field, None)
            if chunk_idx is not None:
                metadata["chunk_index"] = chunk_idx
        if self.field_config.app_name_field:
            app_name = self._get_nested_field(source, self.field_config.app_name_field, None)
            if app_name is not None:
                metadata["appName"] = app_name
        # Backend leaf keys to suppress from the tail-merge: when we synthesize
        # the canonical name from a configured field path, the raw backend key
        # at the corresponding ``metadata.*`` leaf must NOT also leak through —
        # per ``sta_agent_core/AGENTS.md`` "normalize to canonical keys at the
        # boundary". Specifically, ``metadata.auid`` should not appear next to
        # synthesized ``metadata["apcode"]``.
        suppressed_backend_keys: set[str] = set()
        if self.field_config.apcode_field:
            # Concept name ``apcode`` stays on the consumer surface; backend
            # path lives on ``field_config.apcode_field`` (= ``metadata.auid``
            # in prod, ``metadata.apcode`` in legacy indices).
            apcode = self._get_nested_field(source, self.field_config.apcode_field, None)
            if apcode is not None:
                metadata["apcode"] = apcode
            backend_leaf = self.field_config.apcode_field.rsplit(".", 1)[-1]
            if backend_leaf != "apcode":
                # Only suppress backend-named leaves (auid). When the legacy
                # path "metadata.apcode" is configured, the leaf is already
                # the canonical name and the existing ``if key not in metadata``
                # guard does the right thing — don't add to the suppression
                # set or we'd block our own synthesized key.
                suppressed_backend_keys.add(backend_leaf)
        for freshness_field, canonical_key in (
            (self.field_config.last_doc_update_field, "lastDocUpdate"),
            (self.field_config.last_doc_ingestion_field, "lastDocIngestion"),
        ):
            if not freshness_field:
                continue
            raw_timestamp = self._get_nested_field(source, freshness_field, None)
            if raw_timestamp is not None:
                parsed_timestamp = parse_ingestion_timestamp(raw_timestamp)
                # Canonical ISO form when parseable; raw passthrough otherwise
                # so the value is never silently dropped (downstream staleness
                # rendering skips what it cannot parse).
                metadata[canonical_key] = parsed_timestamp.isoformat() if parsed_timestamp is not None else raw_timestamp
            timestamp_leaf = freshness_field.rsplit(".", 1)[-1]
            if timestamp_leaf != canonical_key:
                suppressed_backend_keys.add(timestamp_leaf)
        # Entity extraction uses ``entity_object_field`` (the parent dict path)
        # so consumers see the full ``{name, id, childs, is_opal}`` object.
        # ``entity_field`` (the ``.name`` leaf) stays reserved for aggregations
        # and BM25 boost clauses — using it here would return just the name
        # string, breaking consumers that read ``chunk.metadata["entity"]["id"]``
        # (see ``infra/elasticsearch/probes/metadata_scope_smoke.py``).
        if self.field_config.entity_object_field:
            entity = self._get_nested_field(source, self.field_config.entity_object_field, None)
            if entity is not None:
                metadata["entity"] = entity

        # Merge remaining metadata.* fields not already captured. Backend-named
        # leaves we just normalized into canonical keys are filtered out so the
        # consumer doesn't see both names side by side.
        meta_obj = source.get("metadata", {})
        if isinstance(meta_obj, dict):
            for key, value in meta_obj.items():
                if key in metadata or key in suppressed_backend_keys:
                    continue
                metadata[key] = value

        chunk_id = str(hit.get("_id", ""))
        source_url = url if isinstance(url, str) else ""
        return ElasticRetrievalChunk(
            content=content,
            chunk_id=chunk_id,
            score=score,
            source_url=source_url,
            retriever_type="elasticsearch",
            metadata=metadata,
        )

    def _get_nested_field(self, source: dict[str, Any], field_path: str, default: Any) -> Any:
        parts = field_path.split(".")
        current = source
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def _parse_response(self, response: dict[str, Any]) -> list[ElasticRetrievalChunk]:
        hits = response.get("hits", {}).get("hits", [])
        return [self._default_result_mapper(hit) for hit in hits]

    def _build_document_filter(self, documents: list[str]) -> dict[str, Any]:
        return {"terms": {self.field_config.doc_keyword_field: documents}}

    def _metadata_field_map(self) -> dict[str, str | None]:
        """Map MetadataScope axis/filter keys → ES field paths from this retriever's field_config.

        ``doc`` is the filter-only document-id key (``MetadataScope.doc_filter``)
        — it maps to the exact-match keyword field, distinct from the analyzed
        ``doc_field`` used for BM25 title/body matching.
        """
        return {
            "entity_id": self.field_config.entity_id_field,
            "entity_name": self.field_config.entity_field,
            "entity_childs": self.field_config.entity_childs_field,
            "apcode": self.field_config.apcode_field,
            "app_name": self.field_config.app_name_field,
            "doc": self.field_config.doc_keyword_field,
        }

    def _compose_filter_query(
        self,
        documents: list[str] | None,
        metadata_scope: MetadataScope | None,
    ) -> dict[str, Any] | None:
        """Merge the ``documents`` terms filter and all ``metadata_scope`` filter
        clauses into a single bool-filter dict. Returns ``None`` when neither
        applies, so callers can skip filter wiring entirely.
        """
        filter_clauses: list[dict[str, Any]] = []
        if documents:
            filter_clauses.append(self._build_document_filter(documents))
        if metadata_scope is not None:
            filter_clauses.extend(metadata_scope.build_filter_clauses(self._metadata_field_map(), self.field_config.scope_normalizers))
        if not filter_clauses:
            return None
        if len(filter_clauses) == 1:
            return filter_clauses[0]
        return {"bool": {"filter": filter_clauses}}

    def _build_dense_vector_query(
        self,
        query_embedding: list[float],
        k: int = 10,
        num_candidates: int | None = None,
        boost: float | None = None,
        filter_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        knn_query: dict[str, Any] = {
            "field": self.field_config.embedding_field,
            "query_vector": query_embedding,
            "k": k,
            "num_candidates": num_candidates or k * 2,
        }
        if boost is not None:
            knn_query["boost"] = boost
        if filter_query is not None:
            knn_query["filter"] = filter_query
        return knn_query

    def _build_sparse_vector_query(
        self,
        query: str,
        boost: float | None = None,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        content_field = self.field_config.content_field
        title_field = self.field_config.title_field
        doc_field = self.field_config.doc_field
        title_boost_ratio = self.field_config.title_boost_ratio
        doc_boost_ratio = self.field_config.doc_boost_ratio
        content_boost = boost if boost is not None else 1.0
        title_boost = content_boost * title_boost_ratio
        doc_boost = content_boost * doc_boost_ratio
        text_clauses: list[dict[str, Any]] = [
            {"match": {content_field: {"query": query, "boost": content_boost}}},
            {"match": {title_field: {"query": query, "boost": title_boost}}},
            {"match": {doc_field: {"query": query, "boost": doc_boost}}},
        ]
        if enable_fuzzy:
            fuzzy_content_boost = content_boost * fuzzy_boost_ratio
            fuzzy_title_boost = title_boost * fuzzy_boost_ratio
            text_clauses.extend(
                [
                    {"match": {content_field: {"query": query, "fuzziness": "AUTO", "boost": fuzzy_content_boost}}},
                    {"match": {title_field: {"query": query, "fuzziness": "AUTO", "boost": fuzzy_title_boost}}},
                ]
            )
        bool_body: dict[str, Any] = {}
        if metadata_boost_clauses:
            # ES defaults minimum_should_match to 0 when a bool has a filter or must. If text
            # and boost clauses shared a single outer should, a doc matching only the boost
            # (no text match) would be admitted. Wrap text in bool.must with an inner
            # minimum_should_match=1 so metadata boosts stay score-only.
            bool_body["must"] = [{"bool": {"should": text_clauses, "minimum_should_match": 1}}]
            bool_body["should"] = list(metadata_boost_clauses)
        else:
            bool_body["should"] = text_clauses
        if filter_query is not None:
            bool_body["filter"] = filter_query
        return {"bool": bool_body}

    def _build_rrf_query(
        self,
        query: str,
        query_embedding: list[float],
        size: int = 10,
        rank_window_size: int = 50,
        rank_constant: int = 60,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        sparse_query = self._build_sparse_vector_query(
            query,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        dense_query = self._build_dense_vector_query(
            query_embedding, k=rank_window_size, num_candidates=rank_window_size * 2, filter_query=filter_query
        )
        return {
            "size": size,
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {"standard": {"query": sparse_query}},
                        {"knn": dense_query},
                    ],
                    "rank_window_size": rank_window_size,
                    "rank_constant": rank_constant,
                }
            },
        }

    def _build_boost_query(
        self,
        query: str,
        query_embedding: list[float],
        size: int = 10,
        knn_boost: float = 0.7,
        bm25_boost: float = 0.3,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        sparse_query = self._build_sparse_vector_query(
            query,
            boost=bm25_boost,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        dense_query = self._build_dense_vector_query(query_embedding, k=size, num_candidates=size * 10, boost=knn_boost, filter_query=filter_query)
        return {"size": size, "query": sparse_query, "knn": dense_query}

    def _build_dense_only_query(
        self,
        query_embedding: list[float],
        size: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dense_query = self._build_dense_vector_query(query_embedding, k=size, num_candidates=size * 2, filter_query=filter_query)
        return {"size": size, "knn": dense_query}

    def _build_sparse_only_query(
        self,
        query: str,
        size: int = 10,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        sparse_query = self._build_sparse_vector_query(
            query,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        return {"size": size, "query": sparse_query}

    async def _get_sparse_and_dense_candidates(
        self,
        query: str,
        query_embedding: list[float],
        retrieval_size: int,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> tuple[list[ElasticRetrievalChunk], list[ElasticRetrievalChunk]]:
        """Fetch BM25 + kNN candidate lists in parallel, no merging.

        Used by RRF_ONLY / RRF_RERANKER paths where the two ranked lists
        must stay separate so RRF can see their individual rank positions.
        """
        sparse_body = self._build_sparse_only_query(
            query,
            size=retrieval_size,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        dense_body = self._build_dense_only_query(query_embedding, size=retrieval_size, filter_query=filter_query)
        sparse_response, dense_response = await asyncio.gather(
            self.adapter.search(index=self.index, body=sparse_body),
            self.adapter.search(index=self.index, body=dense_body),
        )
        return self._parse_response(sparse_response), self._parse_response(dense_response)

    async def _rrf_fuse(
        self,
        query: str,
        query_embedding: list[float],
        size: int,
        rank_window_size: int,
        rank_constant: int,
        retrieval_size: int,
        filter_query: dict[str, Any] | None,
        enable_fuzzy: bool,
        fuzzy_boost_ratio: float,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> list[ElasticRetrievalChunk]:
        """Run RRF fusion — native ES query if licensed, in-process Python RRF otherwise.

        Returns top ``size`` docs ranked by RRF score.
        """
        if await self._can_use_native_rrf():
            search_body = self._build_rrf_query(
                query,
                query_embedding,
                size,
                rank_window_size,
                rank_constant,
                filter_query,
                enable_fuzzy=enable_fuzzy,
                fuzzy_boost_ratio=fuzzy_boost_ratio,
                metadata_boost_clauses=metadata_boost_clauses,
            )
            response = await self.adapter.search(index=self.index, body=search_body)
            return self._parse_response(response)

        # Python fallback: fetch both lists, fuse in-process.
        sparse_results, dense_results = await self._get_sparse_and_dense_candidates(
            query=query,
            query_embedding=query_embedding,
            retrieval_size=retrieval_size,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        return WeightedRRF(rank_constant=rank_constant).fuse(
            ranked_lists=[sparse_results, dense_results],
            weights=[1.0, 1.0],
            size=size,
        )

    async def _get_candidates_for_reranking(
        self,
        query: str,
        query_embedding: list[float],
        retrieval_size: int,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> list[ElasticRetrievalChunk]:
        sparse_body = self._build_sparse_only_query(
            query,
            size=retrieval_size,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        dense_body = self._build_dense_only_query(query_embedding, size=retrieval_size, filter_query=filter_query)
        sparse_task = self.adapter.search(index=self.index, body=sparse_body)
        dense_task = self.adapter.search(index=self.index, body=dense_body)
        sparse_response, dense_response = await asyncio.gather(sparse_task, dense_task)
        sparse_results = self._parse_response(sparse_response)
        dense_results = self._parse_response(dense_response)
        return self._merge_candidates(dense_results, sparse_results)

    def _merge_candidates(self, primary: list[ElasticRetrievalChunk], secondary: list[ElasticRetrievalChunk]) -> list[ElasticRetrievalChunk]:
        """Merge two candidate lists, deduplicating by content hash."""
        seen_hashes: set[int] = set()
        merged: list[ElasticRetrievalChunk] = []
        for item in primary:
            h = hash(item.content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                merged.append(item)
        for item in secondary:
            h = hash(item.content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                merged.append(item)
        return merged

    @traceable(
        run_type="retriever",
        name="elasticsearch_hybrid_search",
        process_inputs=_process_retriever_inputs,
        process_outputs=_process_retriever_outputs,
    )
    async def search(
        self,
        query: str,
        size: int = 10,
        *,
        documents: list[str] | None = None,
        fusion_strategy: FusionStrategy | FusionStrategyLiteral | None = None,
        rank_window_size: int | None = None,  # noqa: ARG002 — deprecated synonym for retrieval_size; kept for API parity
        rank_constant: int | None = None,
        retrieval_size: int | None = None,
        enable_fuzzy: bool | None = None,
        fuzzy_boost_ratio: float | None = None,
        metadata_scope: MetadataScope | None = None,
        rerank_top_n: int | None = None,
        expansion_hint: ExpansionStrategy | str | None = None,
        bm25_rrf_weight: float | None = None,
        knn_rrf_weight: float | None = None,
        intent: str | None = None,
        auto_probe_min_score: float | None = None,
        auto_probe_min_gap: float | None = None,
        **kwargs: Any,  # noqa: ARG002 — unknown kwargs silently dropped for caller forward-compat
    ) -> SearchResponse[ElasticRetrievalChunk]:
        """Hybrid BM25+kNN search with optional query expansion and pluggable fusion.

        Phase 5 Cycle E replaced the legacy per-strategy match dispatch with a
        single uniform pipeline:

        1. Seed two SubQueries (``lex`` + ``vec``) from the original query.
        2. If ``expansion_hint != PASS``, append the expander's variants —
           seeds always first, because ``_maybe_build_rerank_scores`` reads
           ``sub_queries[0].query`` as the reranker input (rerank ordering
           contract, v3 §3.3 / ``_maybe_build_rerank_scores:1087``).
        3. For ``WEIGHTED_RRF``, flow the config's ``bm25_rrf_weight`` /
           ``knn_rrf_weight`` into ``SubQuery.weight``. Other strategies use
           uniform 1.0 weights — weighting is a property of the sub-queries,
           not the operator (v3 lock).
        4. Resolve the ``FusionOperator`` via ``resolve_fusion_operator``.
        5. Delegate to ``search_many`` — which owns the ``_run_one`` fan-out
           and ``_maybe_build_rerank_scores`` side-channel.

        The native ES RRF fast-path was dropped — with ``asyncio.gather`` the
        two sub-queries run in parallel (wall-clock ≈ ``max(t_bm25, t_knn)``),
        so the single-RTT savings don't justify a separate code path.
        """
        metadata_scope = self._effective_scope(metadata_scope)
        resolved = self._search_config.resolve_params(
            size=size,
            fusion_strategy=fusion_strategy,
            rank_constant=rank_constant,
            retrieval_size=retrieval_size,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            rerank_top_n=rerank_top_n,
            expansion_hint=expansion_hint,
            bm25_rrf_weight=bm25_rrf_weight,
            knn_rrf_weight=knn_rrf_weight,
            intent=intent,
            auto_probe_min_score=auto_probe_min_score,
            auto_probe_min_gap=auto_probe_min_gap,
        )
        strategy = FusionStrategy(resolved["fusion_strategy"])
        hint = ExpansionStrategy(resolved["expansion_hint"])
        bm25_weight = float(resolved["bm25_rrf_weight"])
        knn_weight = float(resolved["knn_rrf_weight"])

        # Resolve AUTO before the expander is touched — the expander itself
        # raises on AUTO (it has no BM25 access). Probe is a tiny size=2
        # BM25 call whose top-2 scores drive ``_resolve_auto_hint``. Skipped
        # entirely for non-AUTO hints so the normal path stays one-RTT-cheap.
        if hint == ExpansionStrategy.AUTO:
            probe_results = await self.search_text_only(
                query,
                size=2,
                documents=documents,
                metadata_scope=metadata_scope,
                enable_fuzzy=resolved["enable_fuzzy"],
                fuzzy_boost_ratio=resolved["fuzzy_boost_ratio"],
            )
            # Use the resolved thresholds (explicit kwargs > context overrides
            # > instance config) rather than reading ``self._search_config``
            # directly, so per-call ``retriever_auto_probe_*`` overrides take
            # effect even on the generic ``search(**cfg.to_search_kwargs())``
            # dispatch path (Cycle F6c hotfix).
            probe_cfg = dataclasses.replace(
                self._search_config,
                auto_probe_min_score=float(resolved["auto_probe_min_score"]),
                auto_probe_min_gap=float(resolved["auto_probe_min_gap"]),
            )
            hint = self._resolve_auto_hint(ExpansionStrategy.AUTO, probe_results, probe_cfg)

        seed_weights = (bm25_weight, knn_weight) if strategy == FusionStrategy.WEIGHTED_RRF else (1.0, 1.0)
        sub_queries: list[SubQuery] = [
            SubQuery(type="lex", query=query, weight=seed_weights[0]),
            SubQuery(type="vec", query=query, weight=seed_weights[1]),
        ]

        if hint != ExpansionStrategy.PASS:
            if self._expander is None:
                # v3 §3.5 defense-in-depth — factory should have caught this
                # at wire-up; the runtime guard keeps the contract loud.
                raise ValueError(
                    f"expansion_hint={hint.value!r} requires an expander — "
                    f"construct ElasticRetriever with `expander=QueryExpander(...)` "
                    f"or set expansion_hint=PASS."
                )
            # Thread both intent layers through to the expander — build-time
            # ``domain_intent`` from ctor, runtime ``intent`` resolved via
            # ``resolve_params`` so an explicit ``search(intent=...)`` kwarg
            # wins over the instance config's ``intent`` (which itself came
            # from ``retriever_intent`` via ``from_context`` introspection).
            sub_queries.extend(
                await self._expander.expand(
                    query,
                    hint,
                    domain_intent=self._domain_intent,
                    intent=resolved["intent"],
                )
            )

        operator = resolve_fusion_operator(
            strategy.value,
            rank_constant=resolved["rank_constant"],
            bm25_rrf_weight=bm25_weight,
            knn_rrf_weight=knn_weight,
        )

        # Reranker query prepend (F5b-3) — makes both intent layers visible to
        # the cross-encoder. Empty layers are elided via filter(None, …) so a
        # missing intent doesn't leave a stray newline. Built here (not in
        # search_many) so search_many stays intent-agnostic for non-expansion
        # callers that want to pre-build their own SubQuery list.
        rerank_query_override = _join_rerank_query(self._domain_intent, resolved["intent"], query)

        return await self.search_many(
            queries=sub_queries,
            size=resolved["size"],
            fusion=operator,
            documents=documents,
            metadata_scope=metadata_scope,
            rerank_top_n=resolved["rerank_top_n"],
            enable_fuzzy=resolved["enable_fuzzy"],
            fuzzy_boost_ratio=resolved["fuzzy_boost_ratio"],
            rank_constant=resolved["rank_constant"],
            retrieval_size=resolved["retrieval_size"],
            rerank_query_override=rerank_query_override,
        )

    async def search_vector_only(
        self,
        query: str,
        size: int = 10,
        *,
        documents: list[str] | None = None,
        metadata_scope: MetadataScope | None = None,
    ) -> list[ElasticRetrievalChunk]:
        """Perform vector-only search (pure semantic / dense)."""
        metadata_scope = self._effective_scope(metadata_scope)
        query_embedding = await self._get_query_embedding(query)
        filter_query = self._compose_filter_query(documents, metadata_scope)
        search_body = self._build_dense_only_query(query_embedding, size, filter_query)
        response = await self.adapter.search(index=self.index, body=search_body)
        return self._parse_response(response)

    async def search_text_only(
        self,
        query: str,
        size: int = 10,
        *,
        documents: list[str] | None = None,
        metadata_scope: MetadataScope | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
    ) -> list[ElasticRetrievalChunk]:
        """Perform text-only search (pure BM25 / sparse)."""
        metadata_scope = self._effective_scope(metadata_scope)
        filter_query = self._compose_filter_query(documents, metadata_scope)
        metadata_boost_clauses = (
            metadata_scope.build_boost_clauses(self._metadata_field_map(), self.field_config.scope_normalizers)
            if metadata_scope is not None
            else None
        ) or None
        search_body = self._build_sparse_only_query(
            query,
            size,
            filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        response = await self.adapter.search(index=self.index, body=search_body)
        return self._parse_response(response)

    @traceable(
        run_type="retriever",
        name="elasticsearch_search_many",
        process_inputs=_process_search_many_inputs,
        process_outputs=_process_retriever_outputs,
    )
    async def search_many(
        self,
        queries: str | list[str] | list[SubQuery],
        size: int = 10,
        *,
        fusion: FusionOperator | None = None,
        documents: list[str] | None = None,
        metadata_scope: MetadataScope | None = None,
        rerank_top_n: int | None = None,
        enable_fuzzy: bool | None = None,
        fuzzy_boost_ratio: float | None = None,
        rank_window_size: int | None = None,  # noqa: ARG002 — deprecated synonym for retrieval_size; retained for API parity
        rank_constant: int | None = None,
        retrieval_size: int | None = None,
        rerank_query_override: str | None = None,
    ) -> SearchResponse[ElasticRetrievalChunk]:
        """Multi-variant retrieval with pluggable fusion.

        Fans out each sub-query to `search_text_only` (lex) or
        `search_vector_only` (vec/hyde) — each sub-query retrieves
        ``retrieval_size`` candidates (override via kwarg; else falls back to
        ``search_config.retrieval_size``, default 50) so fusion has a deep
        pool; the final output is truncated to ``size``. Concurrency is
        bounded by ``search_config.max_concurrent_subqueries``. Fusion uses
        the supplied ``FusionOperator`` (defaults to ``WeightedRRF``).

        Coercion rules:
          - ``str``           → [SubQuery(lex, q), SubQuery(vec, q)]
          - ``list[str]``     → each string coerced as above
          - ``list[SubQuery]``→ passthrough

        Mixed ``list[str | SubQuery]`` is rejected (``TypeError``) — the coercion
        is not defined and silent coercion would leak type confusion into fusion.

        The FIRST entry in the fused input is conventionally the "original"
        query's ranked list — strategies like ``TopRankBonusRRF`` use this to
        protect exact-match docs from expansion dilution.

        **Uniform ``documents`` filter**: when set, the same document-id filter
        is applied to every sub-query before fusion — per-variant filters are
        not supported. Put a per-variant filter on ``SubQuery`` itself in a
        future phase if the need arises.

        **Partial-failure tolerance**: individual sub-query failures are
        logged and their ranked list is dropped from fusion. If every
        sub-query fails, returns an empty response rather than raising.

        **``rank_constant`` resolution**: a bare ``WeightedRRF()``/
        ``TopRankBonusRRF()``/``PositionAwareBlend()`` (``rank_constant=None``)
        inherits the retriever's configured ``rank_constant`` from
        ``search_config``. To override, construct the operator with an
        explicit ``rank_constant``.
        """
        metadata_scope = self._effective_scope(metadata_scope)
        sub_queries = self._coerce_to_subqueries(queries)
        if not sub_queries:
            return SearchResponse(results=[])

        # Validate metadata_scope up-front — misconfig (an axis whose
        # field_map entry is None) raises ``ValueError`` from
        # ``build_filter_clauses`` / ``build_boost_clauses``. Without this
        # pre-check every sub-query would raise the same error below and
        # the ``return_exceptions=True`` gather would silently drop them
        # all, returning an empty SearchResponse — callers hit "no
        # results + a log warning" instead of the actual stack trace
        # pointing at the field_config misconfig. Cross-team-leakage
        # guard errors are non-recoverable and must propagate.
        if metadata_scope is not None and not metadata_scope.is_empty():
            field_map = self._metadata_field_map()
            normalizers = self.field_config.scope_normalizers
            metadata_scope.build_filter_clauses(field_map, normalizers)
            metadata_scope.build_boost_clauses(field_map, normalizers)

        sem = asyncio.Semaphore(self._search_config.max_concurrent_subqueries)
        per_query_size = retrieval_size if retrieval_size is not None else self._search_config.retrieval_size
        eff_enable_fuzzy = enable_fuzzy if enable_fuzzy is not None else False
        eff_fuzzy_boost = fuzzy_boost_ratio if fuzzy_boost_ratio is not None else 0.25

        async def _run_one(sq: SubQuery) -> list[ElasticRetrievalChunk]:
            async with sem:
                if sq.type == "lex":
                    return await self.search_text_only(
                        sq.query,
                        size=per_query_size,
                        documents=documents,
                        metadata_scope=metadata_scope,
                        enable_fuzzy=eff_enable_fuzzy,
                        fuzzy_boost_ratio=eff_fuzzy_boost,
                    )
                return await self.search_vector_only(
                    sq.query,
                    size=per_query_size,
                    documents=documents,
                    metadata_scope=metadata_scope,
                )

        raw_results = await asyncio.gather(*(_run_one(sq) for sq in sub_queries), return_exceptions=True)

        # Cancellation must NOT be treated as a sub-query failure — `gather(return_exceptions=True)`
        # captures `CancelledError` like any other exception, but the caller (e.g. a disconnected
        # client) needs the cancel to propagate, not a silent empty response.
        if any(isinstance(o, asyncio.CancelledError) for o in raw_results):
            raise asyncio.CancelledError

        # Drop failed sub-queries; keep their weights aligned with surviving lists.
        surviving_lists: list[list[ElasticRetrievalChunk]] = []
        surviving_weights: list[float] = []
        for sq, outcome in zip(sub_queries, raw_results, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning(
                    "search_many: sub-query (type=%s, q=%r) failed — dropping from fusion: %s",
                    sq.type,
                    sq.query[:80],
                    outcome,
                )
                continue
            surviving_lists.append(outcome)
            surviving_weights.append(sq.weight)

        if not surviving_lists:
            return SearchResponse(results=[])

        operator = self._resolve_fusion_operator(fusion, rank_constant_override=rank_constant)

        try:
            rerank_scores = await self._maybe_build_rerank_scores(
                operator=operator,
                sub_queries=sub_queries,
                surviving_lists=surviving_lists,
                surviving_weights=surviving_weights,
                rerank_top_n_override=rerank_top_n,
                size=size,
                rerank_query_override=rerank_query_override,
            )
        except RerankUnavailableError as exc:
            # Soft failure: the reranker errored after its retries. Rather than
            # return zero documents, degrade to rerank-blind RRF over the lists we
            # already retrieved. The reranker's own trace span still shows failed.
            fallback_k = rank_constant if rank_constant is not None else self._search_config.rank_constant
            logger.warning(
                "search_many: %s — falling back to RRF (rrf_only) over %d retrieved list(s).",
                exc,
                len(surviving_lists),
            )
            operator = WeightedRRF(rank_constant=fallback_k)
            rerank_scores = None

        fused = operator.fuse(
            ranked_lists=surviving_lists,
            weights=surviving_weights,
            size=size,
            rerank_scores=rerank_scores,
        )
        return SearchResponse(results=fused)

    async def _maybe_build_rerank_scores(
        self,
        *,
        operator: FusionOperator,
        sub_queries: list[SubQuery],
        surviving_lists: list[list[ElasticRetrievalChunk]],
        surviving_weights: list[float],
        rerank_top_n_override: int | None = None,
        size: int = 10,
        rerank_query_override: str | None = None,
    ) -> dict[tuple[str, str], float] | None:
        """Run reranking upstream when the fusion operator needs it.

        Only ``PositionAwareBlend`` is rerank-aware today. For rerank-blind
        operators (``WeightedRRF``, ``TopRankBonusRRF``) this returns ``None``
        without touching the reranker.

        Pipeline when rerank is needed:
          1. Pre-fuse the surviving sub-query lists with ``WeightedRRF`` to
             pick the top ``rerank_top_n`` candidates.
          2. Send those candidates (with the FIRST sub-query's ``.query`` as
             the rerank query) to the cross-encoder.
          3. Return ``{(page_id, chunk_id): relevance_score}`` — same key
             shape as ``fusion._dedup_key`` so the operator can look up.

        Raises ``ValueError`` with an actionable message if the caller picked
        a rerank-aware operator but no reranker is configured on the retriever.
        """
        if not getattr(operator, "requires_rerank_scores", False):
            return None

        if self._reranker is None:
            raise ValueError(
                f"{type(operator).__name__} requires a reranker on this ElasticRetriever — "
                "inject a RerankClient via the constructor, or choose a rerank-blind "
                "fusion operator (WeightedRRF, TopRankBonusRRF) if reranking is not wanted."
            )

        effective_rerank_top_n = rerank_top_n_override if rerank_top_n_override is not None else self._search_config.rerank_top_n
        rerank_top_n = max(effective_rerank_top_n, size)
        # Pre-rerank fuse uses the operator's resolved rank_constant so the
        # candidate pool is consistent with the rrf_rank the operator computes.
        op_k = getattr(operator, "rank_constant", None)
        pre_rrf_k = op_k if op_k is not None else self._search_config.rank_constant
        pre_rrf = WeightedRRF(rank_constant=pre_rrf_k).fuse(
            ranked_lists=surviving_lists,
            weights=surviving_weights,
            size=rerank_top_n,
        )
        if not pre_rrf:
            return {}

        # Rerank query — either the F5b-3 intent-prepended override from
        # ``search()`` or the first sub-query's raw text (convention: first
        # sub-query is the "original"; reranker never sees expansion variants).
        original_query = rerank_query_override if rerank_query_override is not None else sub_queries[0].query
        # Feed the reconstructed structured blob (contextual prefix + body), not
        # the body alone — parity with BM25, which matches the full content_field.
        docs_for_rerank = [self._rerank_document_text(c) for c in pre_rrf]
        # Clamp top_n to candidate count — some providers (Cohere et al.) 400 when
        # top_n > len(documents). Guarantees `arerank` receives a valid request even
        # when the fused pool is smaller than the configured rerank_top_n.
        #
        # The rerank call (which already retries internally via tenacity) is the
        # soft-failure boundary: on exhaustion the `reranker_arerank` LangSmith
        # span is recorded as failed, then we re-raise as RerankUnavailableError so
        # `search_many` can degrade to RRF instead of returning zero documents.
        try:
            rerank_response = await self._reranker.arerank(
                query=original_query,
                documents=docs_for_rerank,
                top_n=min(rerank_top_n, len(docs_for_rerank)),
                return_documents=False,
            )

            # Validation (out-of-range, duplicates) lives in _apply_rerank_response —
            # shared with _execute_search's RRF_RERANKER path.
            rerank_scores: dict[tuple[str, str], float] = {}
            for idx, r in self._apply_rerank_response(rerank_response, pre_rrf):
                candidate = pre_rrf[idx]
                key = (str(candidate.metadata.get("pageId") or ""), candidate.chunk_id or "")
                rerank_scores[key] = r.relevance_score
        except Exception as exc:
            raise RerankUnavailableError(f"rerank step failed for {type(operator).__name__}: {exc}") from exc
        return rerank_scores

    @classmethod
    def _coerce_to_subqueries(cls, queries: str | list[str] | list[SubQuery]) -> list[SubQuery]:
        """Apply Phase-1 coercion rules (see search_many docstring).

        Rejects mixed ``list[str | SubQuery]`` with ``TypeError`` — every
        element must be the same kind. Classmethod (not static) so subclasses
        or Phase-3 per-type-default-weight variants can override coercion
        while keeping the call-site unchanged.
        """
        if isinstance(queries, str):
            return [SubQuery(type="lex", query=queries), SubQuery(type="vec", query=queries)]
        if not queries:
            return []
        all_subq = all(isinstance(q, SubQuery) for q in queries)
        all_str = all(isinstance(q, str) for q in queries)
        if not (all_subq or all_str):
            raise TypeError(
                "search_many: queries must be list[str] OR list[SubQuery], not a mix — "
                "got types: " + ", ".join(sorted({type(q).__name__ for q in queries}))
            )
        if all_subq:
            return list(queries)  # type: ignore[arg-type]
        out: list[SubQuery] = []
        for q in queries:
            out.append(SubQuery(type="lex", query=q))  # type: ignore[arg-type]
            out.append(SubQuery(type="vec", query=q))  # type: ignore[arg-type]
        return out

    def _resolve_fusion_operator(
        self,
        fusion: FusionOperator | None,
        *,
        rank_constant_override: int | None = None,
    ) -> FusionOperator:
        """Resolve the fusion operator, injecting configured rank_constant when unset.

        Rules:
        - ``fusion is None`` → build ``WeightedRRF`` with the effective rank_constant.
        - User-supplied shipped operator with ``rank_constant=None`` → inject
          effective rank_constant via ``dataclasses.replace``.
        - User-supplied operator with an explicit ``rank_constant`` → passthrough.
        - Third-party ``FusionOperator`` without ``rank_constant`` → passthrough.

        ``rank_constant_override`` (from a per-call kwarg) takes precedence over
        ``search_config.rank_constant`` when set.
        """
        effective_k = rank_constant_override if rank_constant_override is not None else self._search_config.rank_constant
        if fusion is None:
            return WeightedRRF(rank_constant=effective_k)
        if isinstance(fusion, (WeightedRRF, TopRankBonusRRF, PositionAwareBlend, RrfRerankerOperator)) and fusion.rank_constant is None:
            return dataclasses.replace(fusion, rank_constant=effective_k)
        return fusion

    # --- DocumentProvider implementation ---

    @property
    def supports_document_provider(self) -> bool:
        """Whether this retriever can act as a ``DocumentProvider``.

        Requires both ``page_id_field`` and ``chunk_index_field`` to be
        configured in the ``ElasticFieldConfig``.
        """
        return self.field_config.page_id_field is not None and self.field_config.chunk_index_field is not None

    def _require_document_provider(self) -> None:
        """Raise if DocumentProvider prerequisites are not met."""
        if not self.supports_document_provider:
            raise NotImplementedError("DocumentProvider requires page_id_field and chunk_index_field to be set in ElasticFieldConfig.")

    @traceable(
        run_type="retriever",
        name="elastic_get_document",
        process_inputs=_process_get_document_inputs,
        process_outputs=_process_get_document_outputs,
    )
    async def get_document(self, doc_id: str) -> list[ElasticRetrievalChunk]:
        """Fetch all chunks of a document, ordered by ``chunk_index``.

        Args:
            doc_id: Document-level identifier (e.g. pageId).

        Returns:
            Ordered list of chunks for the document.
        """
        self._require_document_provider()
        body: dict[str, Any] = {
            "query": {"term": {self.field_config.page_id_field: doc_id}},
            "sort": [{self.field_config.chunk_index_field: "asc"}],
            "size": 10_000,
        }
        response = await self.adapter.search(index=self.index, body=body)
        return self._parse_response(response)

    @traceable(
        run_type="retriever",
        name="elastic_get_chunk_context",
        process_inputs=_process_get_chunk_context_inputs,
        process_outputs=_process_get_document_outputs,
    )
    async def get_chunk_context(self, chunk_id: str, window: int = 3) -> list[ElasticRetrievalChunk]:
        """Fetch neighbouring chunks around a given chunk.

        Performs a two-step lookup: (1) fetch the anchor chunk by ES ``_id``
        to resolve its ``pageId`` and ``chunk_index``, then (2) range-query
        for surrounding chunks.

        Args:
            chunk_id: The Elasticsearch ``_id`` of the anchor chunk.
            window: Number of chunks before and after to include.

        Returns:
            Chunks in the window, ordered by ``chunk_index``.
        """
        self._require_document_provider()

        # Step 1: resolve anchor chunk's document identity
        # Uses search + ids query instead of GET API so that aliases
        # pointing to multiple indices are handled correctly.
        anchor_resp = await self.adapter.search(
            index=self.index,
            body={"query": {"ids": {"values": [chunk_id]}}, "size": 1},
        )
        hits = anchor_resp.get("hits", {}).get("hits", [])
        if not hits:
            return []
        source = hits[0].get("_source", {})
        page_id = self._get_nested_field(source, self.field_config.page_id_field, None)  # type: ignore[arg-type]
        chunk_idx = self._get_nested_field(source, self.field_config.chunk_index_field, None)  # type: ignore[arg-type]

        if page_id is None or chunk_idx is None:
            return []

        # Step 2: range query for [chunk_index - window, chunk_index + window]
        return await self.get_chunk_range(
            doc_id=str(page_id),
            start_index=max(0, int(chunk_idx) - window),
            end_index=int(chunk_idx) + window,
        )

    @traceable(
        run_type="retriever",
        name="elastic_get_chunk_range",
        process_inputs=_process_get_chunk_range_inputs,
        process_outputs=_process_get_document_outputs,
    )
    async def get_chunk_range(self, doc_id: str, start_index: int, end_index: int) -> list[ElasticRetrievalChunk]:
        """Fetch a range of chunks from a document.

        Args:
            doc_id: Document-level identifier.
            start_index: First ``chunk_index`` (inclusive).
            end_index: Last ``chunk_index`` (inclusive).

        Returns:
            Chunks in the range, ordered by ``chunk_index``.
        """
        self._require_document_provider()
        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {self.field_config.page_id_field: doc_id}},
                        {
                            "range": {
                                self.field_config.chunk_index_field: {
                                    "gte": start_index,
                                    "lte": end_index,
                                }
                            }
                        },
                    ]
                }
            },
            "sort": [{self.field_config.chunk_index_field: "asc"}],
            "size": end_index - start_index + 1,
        }
        response = await self.adapter.search(index=self.index, body=body)
        return self._parse_response(response)

    # --- SupportsBatchFetch implementation ---

    @traceable(
        run_type="retriever",
        name="elastic_get_documents",
    )
    async def get_documents(self, doc_ids: list[str]) -> dict[str, list[ElasticRetrievalChunk]]:
        """Fetch all chunks for multiple documents in a single ``terms`` query.

        Args:
            doc_ids: Document-level identifiers (e.g. pageIds).

        Returns:
            Mapping ``doc_id -> chunks`` ordered by ``chunk_index``. Every input
            id is a key; an unknown document maps to an empty list.
        """
        self._require_document_provider()
        if not doc_ids:
            return {}
        unique_ids = list(dict.fromkeys(doc_ids))
        body: dict[str, Any] = {
            "query": {"terms": {self.field_config.page_id_field: unique_ids}},
            "sort": [{self.field_config.chunk_index_field: "asc"}],
            "size": _BATCH_FETCH_MAX_CHUNKS,
        }
        response = await self.adapter.search(index=self.index, body=body)
        chunks = self._parse_response(response)
        if len(chunks) >= _BATCH_FETCH_MAX_CHUNKS:
            logger.warning(
                "get_documents: hit the %d-chunk batch ceiling for %d documents — results may be truncated",
                _BATCH_FETCH_MAX_CHUNKS,
                len(unique_ids),
            )
        result: dict[str, list[ElasticRetrievalChunk]] = {doc_id: [] for doc_id in unique_ids}
        for chunk in chunks:
            page_id = chunk.metadata.get("pageId")
            if page_id is not None and str(page_id) in result:
                result[str(page_id)].append(chunk)
        return result

    @traceable(
        run_type="retriever",
        name="elastic_get_chunk_ranges",
    )
    async def get_chunk_ranges(self, ranges: list[ChunkRange]) -> dict[ChunkRange, list[ElasticRetrievalChunk]]:
        """Fetch multiple chunk ranges in a single ``bool/should`` query.

        Args:
            ranges: ``(doc_id, start_index, end_index)`` tuples, inclusive.

        Returns:
            Mapping ``range -> chunks``. Every input range is a key. A chunk
            that falls inside several (overlapping) ranges is returned under
            each of them.
        """
        self._require_document_provider()
        if not ranges:
            return {}
        should: list[dict[str, Any]] = []
        total = 0
        for doc_id, start, end in ranges:
            should.append(
                {
                    "bool": {
                        "must": [
                            {"term": {self.field_config.page_id_field: doc_id}},
                            {"range": {self.field_config.chunk_index_field: {"gte": start, "lte": end}}},
                        ]
                    }
                }
            )
            total += max(0, end - start + 1)
        if total > _BATCH_FETCH_MAX_CHUNKS:
            logger.warning(
                "get_chunk_ranges: %d requested chunks across %d ranges exceeds the %d-chunk batch ceiling — tail ranges may be silently truncated",
                total,
                len(ranges),
                _BATCH_FETCH_MAX_CHUNKS,
            )
        body: dict[str, Any] = {
            "query": {"bool": {"should": should, "minimum_should_match": 1}},
            "sort": [{self.field_config.chunk_index_field: "asc"}],
            "size": min(total, _BATCH_FETCH_MAX_CHUNKS) or 1,
        }
        response = await self.adapter.search(index=self.index, body=body)
        chunks = self._parse_response(response)
        result: dict[ChunkRange, list[ElasticRetrievalChunk]] = {r: [] for r in ranges}
        for chunk in chunks:
            page_id = chunk.metadata.get("pageId")
            chunk_idx = chunk.metadata.get("chunk_index")
            if page_id is None or chunk_idx is None:
                continue
            for r in result:
                r_doc, r_start, r_end = r
                if str(page_id) == str(r_doc) and r_start <= int(chunk_idx) <= r_end:
                    result[r].append(chunk)
        return result

    @traceable(
        run_type="retriever",
        name="elastic_get_chunk_contexts",
    )
    async def get_chunk_contexts(self, chunk_ids: list[str], window: int = 3) -> dict[str, list[ElasticRetrievalChunk]]:
        """Fetch neighbouring chunks around multiple anchor chunks.

        Resolves all anchors with one ``ids`` query, then issues a single
        batched range query for the surrounding windows.

        Args:
            chunk_ids: Elasticsearch ``_id`` values of the anchor chunks.
            window: Number of chunks before and after each anchor to include.

        Returns:
            Mapping ``chunk_id -> context chunks``. An anchor that cannot be
            resolved maps to an empty list.
        """
        self._require_document_provider()
        if not chunk_ids:
            return {}
        unique_ids = list(dict.fromkeys(chunk_ids))
        result: dict[str, list[ElasticRetrievalChunk]] = {cid: [] for cid in unique_ids}

        anchor_resp = await self.adapter.search(
            index=self.index,
            body={"query": {"ids": {"values": unique_ids}}, "size": len(unique_ids)},
        )
        anchors = self._parse_response(anchor_resp)

        # Resolve each anchor to its (page_id, chunk_index) and build a window range.
        range_for_id: dict[str, ChunkRange] = {}
        for chunk in anchors:
            page_id = chunk.metadata.get("pageId")
            chunk_idx = chunk.metadata.get("chunk_index")
            if not chunk.chunk_id or page_id is None or chunk_idx is None:
                continue
            idx = int(chunk_idx)
            range_for_id[chunk.chunk_id] = (str(page_id), max(0, idx - window), idx + window)

        if not range_for_id:
            return result

        range_results = await self.get_chunk_ranges(list(range_for_id.values()))
        for cid, r in range_for_id.items():
            result[cid] = range_results.get(r, [])
        return result

    async def close(self) -> None:
        """Cascade close across all owned clients; log failures without re-raising.

        Resources closed (best-effort): the ES adapter, the embedding httpx
        client (if injected), and the reranker. A failure in any one must not
        prevent the others from being released — using
        ``asyncio.gather(return_exceptions=True)`` keeps all coroutines
        scheduled and surfaces individual failures via warning logs.
        """
        tasks: list[tuple[str, Any]] = [("adapter", self.adapter.close())]
        if self._embedding_http_client is not None:
            tasks.append(("embedding_http_client", self._embedding_http_client.aclose()))
        if self._reranker is not None:
            tasks.append(("reranker", self._reranker.close()))

        results = await asyncio.gather(*(coro for _, coro in tasks), return_exceptions=True)
        for (label, _), result in zip(tasks, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("ElasticRetriever.close: %s failed: %r", label, result)

-------

packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/elasticsearch/metadata_scope.py
----
"""Metadata filter/boost scope for Elasticsearch retrieval (Phase 5).

`MetadataScope` is a small validated pydantic model carrying six optional
filter/boost axes plus two expansion flags. It lives in core so both the
direct-retriever callers and the `elastic_rag` graph share the same guard:
empty strings (and lists containing empty strings) are rejected at construction
time — no ES query is ever issued with a degenerate clause.

Clause builders (`build_filter_clauses`, `build_boost_clauses`) turn the model
into ES DSL fragments that the BM25 and kNN query builders compose into their
`filter` / `should` contexts. Both sub-queries receive the same scope so RRF
fusion can never operate over differently-filtered slices (cross-team leakage).
The builders also apply per-axis case-normalization (see
`DEFAULT_AXIS_NORMALIZERS`) so a value reaches ES in the casing the index
expects — regardless of how the model was constructed.

`doc_filter` is a **filter-only** field, deliberately NOT a `ScopeAxis` member.
`ScopeAxis` membership auto-exposes an axis to the LLM planner (the retriever
tool factory reads `MetadataScope.AXIS_NAMES`) and to `add_boosts` (which keys
off `BOOST_FIELDS`). `doc_filter` is a caller/state-injected hard document-id
filter: it carries NO boost and must never reach the planner, so it lives
alongside the axis filters as a plain field and is intersected by `narrow_with`
and emitted by `build_filter_clauses` without participating in either axis set.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

from ..scope_capability import MetadataScopeLike


ScopeValue = str | list[str] | None

#: Maps a single scope token (one ``str``) to its canonical, index-aligned
#: form. Applied per element, so it serves both scalar and list values.
Normalizer = Callable[[str], str]


class ScopeAxis(StrEnum):
    """The three metadata scope axes (Layer 1/2/3 of the KA scope stack).

    Single source of truth for axis names. ``str``-valued so it interoperates
    transparently with the spec/JSON config layer (``ScopeAxis.ENTITY == "entity"``).
    """

    ENTITY = "entity"
    APCODE = "apcode"
    APP_NAME = "app_name"

    @property
    def filter_key(self) -> str:
        """Field name on ``MetadataScope`` for this axis's filter (Layer 1/2)."""
        return f"{self.value}_filter"

    @property
    def boost_key(self) -> str:
        """Field name on ``MetadataScope`` for this axis's boost (Layer 3)."""
        return f"{self.value}_boost"


DEFAULT_AXIS_NORMALIZERS: Mapping[ScopeAxis, Normalizer] = {
    ScopeAxis.APP_NAME: str.lower,
    ScopeAxis.APCODE: str.upper,
}
"""Per-axis value normalizers applied at clause-build time.

``app_name`` is lowercased: the production ES mapping declares
``metadata.appName`` as a ``keyword`` with a lowercase normalizer, so indexed
values are stored lowercased and a term in any other case would silently match
nothing. ``apcode`` is uppercased: ``metadata.auid`` is a case-sensitive
``keyword`` with NO normalizer, and every apcode is ingested as a 7-character
uppercase identifier (``AP`` + 5 digits, or ``A`` + 6 digits). ``entity`` has
no entry — ``entity.id`` is case-sensitive with no single canonical case, so it
is left untouched. A consumer whose index follows a different keyword-casing
convention passes its own mapping to the clause builders.
"""


def _normalize_scope_value(value: ScopeValue, normalizer: Normalizer | None) -> ScopeValue:
    """Apply a per-token ``normalizer`` to a scope value; pass ``None`` through.

    ``normalizer=None`` (the axis carries no policy) returns the value
    unchanged. For a list, normalization can collapse mixed-case members onto
    the same token — the result is order-preservingly de-duplicated so the
    emitted ``terms`` array carries no redundant entries.
    """
    if value is None or normalizer is None:
        return value
    if isinstance(value, str):
        return normalizer(value)
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        normalized = normalizer(item)
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _reject_empty(value: ScopeValue) -> ScopeValue:
    if value is None:
        return value
    if isinstance(value, str):
        if value == "":
            raise ValueError("empty string is not a valid filter/boost value")
        return value
    if isinstance(value, list):
        if not value:
            raise ValueError("empty list is not a valid filter/boost value (ES would match nothing silently)")
        for item in value:
            if not isinstance(item, str) or item == "":
                raise ValueError("list must contain only non-empty strings")
        return value
    raise TypeError(f"unsupported scope value type: {type(value).__name__}")


class MetadataScope(BaseModel):
    """Validated filter/boost scope for a single retrieval call."""

    model_config = ConfigDict(frozen=True)

    # Single source of truth for axis name sets — derived from ``ScopeAxis``.
    # Importers (tool factory, resolver) read these instead of re-declaring
    # parallel frozensets that drift on a new axis.
    AXES: ClassVar[tuple[ScopeAxis, ...]] = tuple(ScopeAxis)
    AXIS_NAMES: ClassVar[tuple[str, ...]] = tuple(a.value for a in ScopeAxis)
    FILTER_FIELDS: ClassVar[frozenset[str]] = frozenset(a.filter_key for a in ScopeAxis)
    BOOST_FIELDS: ClassVar[frozenset[str]] = frozenset(a.boost_key for a in ScopeAxis)

    # Filter-only fields: hard filters with NO boost and NO planner exposure,
    # so they are intentionally absent from ``ScopeAxis`` (and thus from
    # ``AXIS_NAMES`` / ``BOOST_FIELDS``). ``doc_filter`` is a caller/state-
    # injected document-id filter. ``ALL_FILTER_FIELDS`` is the full set of
    # filter-bearing fields — the three axis filter keys plus the filter-only
    # ones — used by ``narrow_with`` to intersect every filter uniformly.
    FILTER_ONLY_FIELDS: ClassVar[frozenset[str]] = frozenset({"doc_filter"})
    ALL_FILTER_FIELDS: ClassVar[frozenset[str]] = FILTER_FIELDS | FILTER_ONLY_FIELDS

    entity_filter: ScopeValue = None
    entity_boost: ScopeValue = None
    apcode_filter: ScopeValue = None
    apcode_boost: ScopeValue = None
    app_name_filter: ScopeValue = None
    app_name_boost: ScopeValue = None
    doc_filter: ScopeValue = None

    include_entity_childs: bool = False
    include_transversal: bool = False

    # Sentinel marker, ``True`` only on ``MetadataScope._EMPTY``. A scope whose
    # filter axes intersected to nothing matches NO documents — distinct from a
    # default all-``None`` scope, which matches everything. Never set by callers;
    # produced solely by ``narrow_with`` when an intersection is empty. Carried
    # in ``__pydantic_private__`` so it survives ``model_copy`` and participates
    # in ``__eq__`` (``_EMPTY != MetadataScope()``).
    _match_nothing: bool = PrivateAttr(default=False)

    # Singleton "matches nothing" scope. Assigned below the class body — a class
    # cannot reference itself from within its own body. See ``narrow_with``.
    _EMPTY: ClassVar[MetadataScope]

    @field_validator(
        "entity_filter",
        "entity_boost",
        "apcode_filter",
        "apcode_boost",
        "app_name_filter",
        "app_name_boost",
        "doc_filter",
        mode="before",
    )
    @classmethod
    def _detach_list_inputs(cls, v: Any) -> Any:
        # ``frozen=True`` freezes attribute reassignment but not the
        # contents of a mutable list value. Copy any incoming list so the
        # model owns its own buffer — the caller can't later mutate it to
        # smuggle values past ``add_boosts`` validation.
        if isinstance(v, list):
            return list(v)
        return v

    @field_validator(
        "entity_filter",
        "entity_boost",
        "apcode_filter",
        "apcode_boost",
        "app_name_filter",
        "app_name_boost",
        "doc_filter",
        mode="after",
    )
    @classmethod
    def _no_empty_strings(cls, v: ScopeValue) -> ScopeValue:
        return _reject_empty(v)

    def is_empty(self) -> bool:
        """True iff no filter or boost is active.

        Expansion flags (``include_entity_childs``, ``include_transversal``)
        only widen an already-set filter/boost — without their pair, they are
        no-ops. Treating flags-only scopes as empty lets the graph skip scope
        construction and avoid paying the cost per request.
        """
        return (
            self.entity_filter is None
            and self.entity_boost is None
            and self.apcode_filter is None
            and self.apcode_boost is None
            and self.app_name_filter is None
            and self.app_name_boost is None
            and self.doc_filter is None
        )

    def is_effective(self) -> bool:
        """Return whether this scope must reach Elasticsearch.

        The match-nothing sentinel has no populated fields but is still an
        active constraint; dropping it would turn a disjoint intersection into
        an unrestricted search.
        """
        return self.matches_nothing or not self.is_empty()

    def apply_caller_scope(self, caller_scope: MetadataScopeLike) -> MetadataScope:
        """Intersect a caller scope into this build-time Elasticsearch scope."""
        if not isinstance(caller_scope, MetadataScope):
            raise TypeError(f"MetadataScope requires MetadataScope caller input, got {type(caller_scope).__name__}")
        return self.narrow_with(caller_scope)

    @classmethod
    def from_caller_scope(cls, bundle: Mapping[str, Any]) -> MetadataScope | None:
        """Resolve the Elasticsearch fields from a normalized caller bundle."""
        scope = cls(
            doc_filter=bundle.get("doc_ids"),
            apcode_filter=bundle.get("apcode"),
            app_name_filter=bundle.get("app_name"),
            entity_filter=bundle.get("entity"),
        )
        return scope if scope.is_effective() else None

    def add_boosts(self, **axes: ScopeValue) -> MetadataScope:
        """Union new boost values into existing ``_boost`` axes — boost-only.

        The runtime query scope (planner-tool args resolved via
        ``MetadataValueResolver``) writes here. Filters are never touched: the
        LLM can soft-rank docs but cannot widen or narrow the build-time
        ``default_scope`` filter ceiling.

        Args:
            **axes: Mapping ``axis_name -> str | list[str] | None``. Only
                ``entity_boost``, ``apcode_boost``, and ``app_name_boost``
                are accepted. ``None`` values are no-ops; anything else
                must satisfy the same non-empty-string validation as
                construction.

        Returns:
            A new frozen ``MetadataScope`` with axes unioned (order-preserving
            dedup) into existing values.

        Raises:
            ValueError: If an axis is not a known boost axis, or a value
                fails the empty-string validator.
        """
        unknown = set(axes) - self.BOOST_FIELDS
        if unknown:
            raise ValueError(f"add_boosts only accepts boost axes ({sorted(self.BOOST_FIELDS)}); got: {sorted(unknown)}")

        updates: dict[str, ScopeValue] = {}
        for axis, raw in axes.items():
            if raw is None:
                continue
            # Validation guard — raises on an empty string, an empty list, or
            # a list with an empty member. Axis case-normalization is applied
            # later, at clause-build time (see ``DEFAULT_AXIS_NORMALIZERS``),
            # so the union below operates on raw values.
            _reject_empty(raw)
            existing = getattr(self, axis)
            updates[axis] = _union_scope_values(existing, raw)

        return self.model_copy(update=updates) if updates else self.model_copy()

    @property
    def matches_nothing(self) -> bool:
        """True iff this scope is the never-match sentinel (``MetadataScope._EMPTY``).

        A never-match scope is produced only by ``narrow_with`` when two scopes
        constrain the same axis with no overlapping value. It is distinct from a
        default all-``None`` scope (``is_empty()`` — no filter, matches
        everything): this one matches *nothing*.
        """
        return self._match_nothing

    def narrow_with(self, other: MetadataScope) -> MetadataScope:
        """Intersect filter axes with ``other`` — never widens the scope.

        This is the trust contract for stacking a build-time filter ceiling
        with a second scope (e.g. a path-derived facet filter): the result
        admits a document only if BOTH scopes admit it. Per filter axis
        (``entity``/``apcode``/``app_name``):

        - ``self`` unconstrained on the axis → adopt ``other``'s values.
        - ``other`` unconstrained → keep ``self``'s values.
        - both constrain the axis → set intersection (order from ``self``).
        - intersection empty on ANY axis → return ``MetadataScope._EMPTY``, the
          never-match sentinel. Returning ``None`` here would silently *widen*
          the axis to "no filter" — a cross-team-leakage bug.

        Boosts are never touched (boosts only soft-rank; they cannot widen a
        filter ceiling). Widening flags (``include_entity_childs``,
        ``include_transversal``) are inherited from ``self`` only — ``other`` is
        the narrowing operand, and honoring a flag from it could widen the
        result.

        Args:
            other: The scope to intersect into ``self``.

        Returns:
            A new frozen ``MetadataScope`` no wider than ``self`` on any filter
            axis, or ``MetadataScope._EMPTY`` if some axis intersected to empty.
        """
        if self._match_nothing or other._match_nothing:
            return MetadataScope._EMPTY

        updates: dict[str, ScopeValue] = {}
        # Iterate over EVERY filter-bearing field — the three axis filter keys
        # plus the filter-only ones (``doc_filter``) — so a caller/state-
        # injected doc filter is intersected with the same AND semantics as the
        # planner-exposed axes. An empty intersection on ANY filter (incl. doc)
        # collapses the whole scope to the never-match sentinel.
        for key in self.ALL_FILTER_FIELDS:
            current = getattr(self, key)
            result = _intersect_scope_values(current, getattr(other, key))
            if result is _EMPTY_INTERSECTION:
                return MetadataScope._EMPTY
            # Record only fields that actually changed: an unchanged narrow then
            # leaves ``__pydantic_fields_set__`` untouched, so ``A.narrow_with(A)``
            # stays equal to ``A``.
            if result != current:
                updates[key] = result  # type: ignore[assignment]  # _EMPTY_INTERSECTION ruled out above
        return self.model_copy(update=updates) if updates else self.model_copy()

    def build_filter_clauses(
        self,
        field_map: dict[str, str | None],
        normalizers: Mapping[ScopeAxis, Normalizer] | None = None,
    ) -> list[dict[str, Any]]:
        """Return AND-composed filter clauses (``term`` / ``terms``).

        ``field_map`` maps axis keys (``entity_id``, ``entity_childs``,
        ``apcode``, ``app_name``) to their ES field paths.  A ``None`` mapping
        for an axis that has an active filter raises ``ValueError`` — silently
        dropping a filter would be a cross-team-leakage bug.

        ``normalizers`` is the per-axis value-normalization policy; ``None``
        uses ``DEFAULT_AXIS_NORMALIZERS`` (``app_name`` lowercased, ``apcode``
        uppercased). Normalization is applied here, at build time, so it covers
        every value path uniformly — including the ``model_copy(update=...)`` /
        ``model_construct(...)`` escape hatches that bypass field validators.

        When ``include_transversal=True`` AND ``apcode_filter`` is set, the
        apcode clause is widened from a plain ``term``/``terms`` to a
        ``bool.should`` that also admits ``appName="transversal"`` docs.
        No-op when ``apcode_filter`` is None (documented behavior).

        A never-match scope (``MetadataScope._EMPTY``, produced by
        ``narrow_with`` on an empty intersection) returns a single clause that
        matches no document — never an empty clause list, which ES would read
        as "no filter".
        """
        if self._match_nothing:
            return [{"bool": {"must_not": [{"match_all": {}}]}}]
        norm = DEFAULT_AXIS_NORMALIZERS if normalizers is None else normalizers
        entity_filter = _normalize_scope_value(self.entity_filter, norm.get(ScopeAxis.ENTITY))
        apcode_filter = _normalize_scope_value(self.apcode_filter, norm.get(ScopeAxis.APCODE))
        app_name_filter = _normalize_scope_value(self.app_name_filter, norm.get(ScopeAxis.APP_NAME))

        clauses: list[dict[str, Any]] = []
        if entity_filter is not None and self.include_entity_childs:
            clauses.append(self._build_entity_childs_clause(field_map, entity_filter, axis_label="entity_filter"))
        else:
            _append_clause(clauses, field_map, "entity_id", entity_filter, axis_label="entity_filter")
        if apcode_filter is not None and self.include_transversal:
            clauses.append(self._build_transversal_apcode_clause(field_map, apcode_filter, axis_label="apcode_filter"))
        else:
            _append_clause(clauses, field_map, "apcode", apcode_filter, axis_label="apcode_filter")
        _append_clause(clauses, field_map, "app_name", app_name_filter, axis_label="app_name_filter")
        # Filter-only ``doc_filter`` — exact keyword match (like entity), no
        # normalizer (it carries no ``DEFAULT_AXIS_NORMALIZERS`` entry) and no
        # boost counterpart. Emitted last so the doc-id hard filter ANDs with
        # the planner-exposed axes.
        _append_clause(clauses, field_map, "doc", self.doc_filter, axis_label="doc_filter")
        return clauses

    _TRANSVERSAL_APP_NAME_VALUE = "transversal"

    def _build_entity_childs_clause(
        self,
        field_map: dict[str, str | None],
        value: str | list[str],
        *,
        axis_label: str,
        include_name: bool = False,
    ) -> dict[str, Any]:
        entity_id_field = field_map.get("entity_id")
        entity_childs_field = field_map.get("entity_childs")
        if entity_id_field is None:
            raise ValueError(
                f"{axis_label} is set but field_map['entity_id'] is not configured — refusing to drop the clause (would cause cross-team leakage)."
            )
        if entity_childs_field is None:
            raise ValueError(
                f"include_entity_childs=True requires field_map['entity_childs'] to be configured (used to admit descendants into the entity {axis_label.split('_')[-1]})."
            )
        should: list[dict[str, Any]] = [
            _term_or_terms(entity_id_field, value),
            _term_or_terms(entity_childs_field, value),
        ]
        if include_name:
            entity_name_field = field_map.get("entity_name")
            if entity_name_field is not None:
                should.append(_term_or_terms(entity_name_field, value))
        return {"bool": {"should": should, "minimum_should_match": 1}}

    def _build_transversal_apcode_clause(self, field_map: dict[str, str | None], value: str | list[str], *, axis_label: str) -> dict[str, Any]:
        apcode_field = field_map.get("apcode")
        app_name_field = field_map.get("app_name")
        if apcode_field is None:
            raise ValueError(
                f"{axis_label} is set but field_map['apcode'] is not configured — refusing to drop the clause (would cause cross-team leakage)."
            )
        if app_name_field is None:
            raise ValueError(
                f"include_transversal=True requires field_map['app_name'] to be configured "
                f"(used to admit appName='transversal' docs into the apcode {axis_label.split('_')[-1]})."
            )
        return {
            "bool": {
                "should": [
                    _term_or_terms(apcode_field, value),
                    {"term": {app_name_field: self._TRANSVERSAL_APP_NAME_VALUE}},
                ],
                "minimum_should_match": 1,
            }
        }

    def build_boost_clauses(
        self,
        field_map: dict[str, str | None],
        normalizers: Mapping[ScopeAxis, Normalizer] | None = None,
    ) -> list[dict[str, Any]]:
        """Return additive boost clauses (``term`` / ``terms``) for BM25 ``should`` context.

        ``entity_boost`` widens to match both ``entity_id`` and ``entity_name``
        so passthrough deployments (no resolver) — where the planner emits a
        name while the canonical filter field is the id — still get a boost
        hit on either representation. Expansion flags widen boost symmetrically
        with filter (``include_entity_childs`` admits ``entity_childs``;
        ``include_transversal`` admits ``app_name='transversal'``). No-op when
        the paired boost value is ``None``.

        ``normalizers`` follows the same per-axis policy as
        ``build_filter_clauses`` — ``None`` uses ``DEFAULT_AXIS_NORMALIZERS``.
        """
        norm = DEFAULT_AXIS_NORMALIZERS if normalizers is None else normalizers
        entity_boost = _normalize_scope_value(self.entity_boost, norm.get(ScopeAxis.ENTITY))
        apcode_boost = _normalize_scope_value(self.apcode_boost, norm.get(ScopeAxis.APCODE))
        app_name_boost = _normalize_scope_value(self.app_name_boost, norm.get(ScopeAxis.APP_NAME))

        clauses: list[dict[str, Any]] = []
        if entity_boost is not None and self.include_entity_childs:
            clauses.append(self._build_entity_childs_clause(field_map, entity_boost, axis_label="entity_boost", include_name=True))
        elif entity_boost is not None:
            clauses.append(_build_entity_id_or_name_clause(field_map, entity_boost, axis_label="entity_boost"))
        if apcode_boost is not None and self.include_transversal:
            clauses.append(self._build_transversal_apcode_clause(field_map, apcode_boost, axis_label="apcode_boost"))
        else:
            _append_clause(clauses, field_map, "apcode", apcode_boost, axis_label="apcode_boost")
        _append_clause(clauses, field_map, "app_name", app_name_boost, axis_label="app_name_boost")
        return clauses


def _union_scope_values(existing: ScopeValue, new: str | list[str]) -> ScopeValue:
    """Order-preserving union for boost values.

    Returns a single string when the union has one element (keeps the model
    representation tight), else a list. ``existing=None`` is treated as
    "no prior value".
    """
    existing_list: list[str] = [] if existing is None else ([existing] if isinstance(existing, str) else list(existing))
    new_list: list[str] = [new] if isinstance(new, str) else list(new)

    seen: set[str] = set()
    union: list[str] = []
    for item in (*existing_list, *new_list):
        if item not in seen:
            seen.add(item)
            union.append(item)
    return union[0] if len(union) == 1 else union


# Returned by ``_intersect_scope_values`` when two axis values share no element.
# A distinct object so ``narrow_with`` can tell "empty intersection" apart from
# a legitimate ``None`` (one operand unconstrained on the axis).
_EMPTY_INTERSECTION: Any = object()


def _intersect_scope_values(a: ScopeValue, b: ScopeValue) -> ScopeValue | Any:
    """Set-intersect two filter-axis values — order preserved from ``a``.

    ``None`` means "unconstrained": if either side is ``None`` the other side
    passes through unchanged (intersecting a constraint with no-constraint
    yields the constraint). When both sides constrain the axis, returns their
    intersection — a single string when one element remains, a list otherwise,
    or ``_EMPTY_INTERSECTION`` when they are disjoint.
    """
    if a is None:
        return b
    if b is None:
        return a
    a_list: list[str] = [a] if isinstance(a, str) else list(a)
    b_set: set[str] = {b} if isinstance(b, str) else set(b)
    common = [item for item in a_list if item in b_set]
    if not common:
        return _EMPTY_INTERSECTION
    return common[0] if len(common) == 1 else common


def _term_or_terms(field: str, value: str | list[str]) -> dict[str, Any]:
    if isinstance(value, list):
        return {"terms": {field: value}}
    return {"term": {field: value}}


def _build_entity_id_or_name_clause(
    field_map: dict[str, str | None],
    value: str | list[str],
    *,
    axis_label: str,
) -> dict[str, Any]:
    """Boost on ``entity_id`` OR ``entity_name`` so passthrough boosts match
    whichever representation the planner emits.

    At least one of the two fields must be configured; if both are ``None``
    we refuse to silently drop the clause.
    """
    entity_id_field = field_map.get("entity_id")
    entity_name_field = field_map.get("entity_name")
    should: list[dict[str, Any]] = []
    if entity_id_field is not None:
        should.append(_term_or_terms(entity_id_field, value))
    if entity_name_field is not None:
        should.append(_term_or_terms(entity_name_field, value))
    if not should:
        raise ValueError(
            f"{axis_label} is set but neither field_map['entity_id'] nor field_map['entity_name'] is configured — "
            "refusing to drop the clause (would silently no-op the boost)."
        )
    if len(should) == 1:
        return should[0]
    return {"bool": {"should": should, "minimum_should_match": 1}}


def _append_clause(
    clauses: list[dict[str, Any]],
    field_map: dict[str, str | None],
    axis_key: str,
    value: ScopeValue,
    *,
    axis_label: str,
) -> None:
    if value is None:
        return
    field = field_map.get(axis_key)
    if field is None:
        raise ValueError(
            f"MetadataScope.{axis_label} is set but field_map[{axis_key!r}] is not configured "
            f"on ElasticFieldConfig — refusing to drop the filter (would cause cross-team leakage)."
        )
    clauses.append(_term_or_terms(field, value))


def _build_empty_scope() -> MetadataScope:
    """Construct the singleton never-match sentinel — see ``MetadataScope._EMPTY``.

    A default ``MetadataScope`` with the private ``_match_nothing`` marker set.
    Built once at import time; ``narrow_with`` returns this exact instance for
    every empty intersection.
    """
    scope = MetadataScope()
    scope._match_nothing = True
    return scope


MetadataScope._EMPTY = _build_empty_scope()

-------

packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/lightrag/__init__.py
----
"""LightRAG retriever, engines, and types."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .engines.http_engine import LightRAGHttpEngine
from .engines.protocol import LightRAGEngine
from .instance_registry import LightRAGInstanceRegistry
from .lightrag_core_settings import LightRAGCoreSettings
from .lightrag_metadata_scope import LightRAGMetadataScope
from .lightrag_response_parser import parse_lightrag_response
from .lightrag_retrieval_chunk import (
    LightRAGEntity,
    LightRAGReference,
    LightRAGRelationship,
    LightRAGRetrievalChunk,
)
from .lightrag_retriever import LightRAGRetriever
from .lightrag_retriever_settings import LightRAGRetrieverSettings
from .lightrag_search_config import LightRAGRetrieverContext, LightRAGSearchConfig
from .lightrag_search_response import (
    LightRAGQueryMetadata,
    LightRAGSearchResponse,
)
from .subgraph_converter import from_lightrag_response


__all__ = [
    "LightRAGEngine",
    "LightRAGEntity",
    "LightRAGHttpEngine",
    "LightRAGInstanceRegistry",
    "LightRAGMetadataScope",
    "LightRAGCoreSettings",
    "LightRAGQueryMetadata",
    "LightRAGReference",
    "LightRAGRelationship",
    "LightRAGRetrievalChunk",
    "LightRAGRetriever",
    "LightRAGRetrieverContext",
    "LightRAGRetrieverSettings",
    "LightRAGSearchConfig",
    "LightRAGSearchResponse",
    "parse_lightrag_response",
    "from_lightrag_response",
    "LightRAGCoreEngine",
]

_LAZY_EXPORTS = {
    "LightRAGCoreEngine": ".engines.core_engine",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve LightRAG exports so HTTP users do not load core deps."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return public exports for interactive discovery."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .engines.core_engine import LightRAGCoreEngine

-------

packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/lightrag/engines/http_engine.py
----
"""HTTP engine — queries a deployed LightRAG server via REST API.

Extracted from the original ``LightRAGRetriever`` to separate transport
concerns from retrieval logic.  This is the legacy/migration path;
prefer ``LightRAGCoreEngine`` for direct LightRAG core access.

Supports pluggable authentication via ``AuthProvider``:
- ``StaticBearerAuth``: static API key (default when ``api_key`` is passed)
- ``JWTAuth``: dynamic JWT with auto-refresh on 401
- ``NoAuth``: no authentication
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import httpx

from sta_agent_core.adapters.auth import AuthProvider, NoAuth, StaticBearerAuth
from sta_agent_core.adapters.http import AsyncHttpAdapter

from ...exceptions import RetrieverConnectionError, RetrieverError, RetrieverResponseError


logger = logging.getLogger(__name__)

_DOCUMENT_PROVIDER_UNSUPPORTED_MSG = (
    "LightRAG server does not support DocumentProvider operations. Upgrade the LightRAG server to include the /chunks and /documents endpoints."
)


class LightRAGHttpEngine:
    """Engine that communicates with a LightRAG HTTP server.

    Implements the ``LightRAGEngine`` protocol via structural typing.
    Handles authentication, distributed tracing header injection, and
    DocumentProvider endpoint availability detection.

    Authentication modes (mutually exclusive):

    1. Pass ``auth_provider`` directly for full control.
    2. Pass ``api_key`` for static Bearer token (backward-compatible).
    3. Pass neither for unauthenticated requests.

    On HTTP 401, the engine calls ``auth_provider.on_unauthorized()``
    and retries the request once. This enables JWTAuth to transparently
    re-authenticate.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        distributed_tracing: bool = True,
        *,
        auth_provider: AuthProvider | None = None,
        auth_header_format: str = "bearer",
        resolve_redirects: bool = True,
        use_twin_api: bool = False,
    ) -> None:
        self._auth: AuthProvider
        if auth_provider is not None:
            self._auth = auth_provider
        elif api_key:
            self._auth = StaticBearerAuth(api_key, header_format=auth_header_format)
        else:
            self._auth = NoAuth()

        self._client = AsyncHttpAdapter(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            resolve_redirects=resolve_redirects,
        )
        self._distributed_tracing = distributed_tracing
        self._document_provider_available: bool | None = None
        self._use_twin_api = use_twin_api

    # --- LightRAGEngine protocol ---------------------------------------------

    async def query(self, query: str, search_kwargs: dict[str, Any]) -> dict[str, Any]:
        """POST to the configured LightRAG query endpoint."""
        endpoint = "/api/query/data" if self._use_twin_api else "/query/data"
        payload: dict[str, Any] = {"query": query, **search_kwargs}
        headers = await self._build_headers()

        try:
            response = await self._client.post(endpoint, json=payload, headers=headers)
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                raise RetrieverResponseError("LightRAG server rejected the query payload (422)") from e
            if e.response.status_code == 401:
                try:
                    response = await self._retry_after_reauth("POST", endpoint, json=payload)
                    return response.json()
                except httpx.HTTPStatusError as retry_error:
                    raise RetrieverConnectionError(f"LightRAG server error after re-auth: {retry_error.response.status_code}") from retry_error
                except (httpx.ConnectError, httpx.TimeoutException) as retry_error:
                    raise RetrieverConnectionError(f"LightRAG server unreachable after re-auth: {retry_error}") from retry_error
                except Exception as retry_error:
                    raise RetrieverConnectionError(f"LightRAG request failed after re-auth: {retry_error}") from retry_error
            raise RetrieverConnectionError(f"LightRAG server error: {e.response.status_code}") from e
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise RetrieverConnectionError(f"LightRAG server unreachable: {e}") from e
        except Exception as e:
            raise RetrieverConnectionError(f"LightRAG request failed: {e}") from e

    async def _build_headers(self) -> dict[str, str]:
        """Merge auth headers with distributed tracing headers."""
        headers = await self._auth.get_auth_headers()
        if self._distributed_tracing:
            try:
                from langsmith.run_helpers import get_current_run_tree

                run_tree = get_current_run_tree()
                if run_tree:
                    headers.update(run_tree.to_headers())
            except ImportError:
                pass
        return headers

    async def _retry_after_reauth(self, method: str, endpoint: str, **kwargs: Any) -> httpx.Response:
        """Invalidate auth, re-authenticate, and retry the request once."""
        logger.info("LightRAGHttpEngine: 401 received, re-authenticating and retrying")
        await self._auth.on_unauthorized()
        headers = await self._build_headers()
        return await self._client.request(method, endpoint, headers=headers, **kwargs)

    async def get_document_chunks(
        self,
        doc_id: str,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any]:
        """GET /documents/{doc_id}/chunks with optional range."""
        params: dict[str, Any] = {}
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        return await self._document_provider_get(
            f"/documents/{doc_id}/chunks",
            params=params or None,
        )

    async def get_chunk_context(
        self,
        chunk_id: str,
        window: int = 3,
    ) -> dict[str, Any]:
        """GET /chunks/{chunk_id}/context?window=N."""
        return await self._document_provider_get(
            f"/chunks/{chunk_id}/context",
            params={"window": window},
        )

    async def close(self) -> None:
        await self._client.close()

    # --- DocumentProvider availability detection -----------------------------

    def _handle_document_provider_error(self, e: httpx.HTTPStatusError, endpoint: str) -> None:
        """Interpret HTTP error from a DocumentProvider endpoint; set availability and raise."""
        status = e.response.status_code
        if status == 405:
            self._document_provider_available = False
            logger.warning("LightRAG server returned 405 for %s — DocumentProvider not available", endpoint)
            raise RetrieverError(_DOCUMENT_PROVIDER_UNSUPPORTED_MSG) from e
        if status == 404:
            detail = ""
            with contextlib.suppress(Exception):
                detail = e.response.json().get("detail", "")
            is_generic_404 = not (detail and detail != "Not Found")
            if is_generic_404 and self._document_provider_available is None:
                self._document_provider_available = False
                logger.warning("LightRAG server returned 404 for %s — DocumentProvider not available", endpoint)
                raise RetrieverError(_DOCUMENT_PROVIDER_UNSUPPORTED_MSG) from e
            if self._document_provider_available is None:
                self._document_provider_available = True
                logger.info("LightRAG server supports DocumentProvider endpoints")
            msg = f"Not found: {endpoint}" + (f" ({detail})" if detail else "")
            raise RetrieverResponseError(msg) from e
        if status == 422:
            raise RetrieverResponseError(f"LightRAG server rejected request to {endpoint}: {e.response.text}") from e
        raise RetrieverConnectionError(f"LightRAG server error: {status}") from e

    async def _document_provider_get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a GET against a DocumentProvider endpoint.

        Detects unsupported servers (404/405) on the first call and
        caches the result so subsequent calls fail fast.
        """
        if self._document_provider_available is False:
            raise RetrieverError(_DOCUMENT_PROVIDER_UNSUPPORTED_MSG)

        headers = await self._build_headers()
        try:
            response = await self._client.get(endpoint, params=params, headers=headers)
            data: dict[str, Any] = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                try:
                    response = await self._retry_after_reauth("GET", endpoint, params=params)
                    data = response.json()
                except httpx.HTTPStatusError as retry_error:
                    self._handle_document_provider_error(retry_error, endpoint)
                except (httpx.ConnectError, httpx.TimeoutException) as retry_error:
                    raise RetrieverConnectionError(f"LightRAG server unreachable after re-auth: {retry_error}") from retry_error
                except Exception as retry_error:
                    raise RetrieverConnectionError(f"LightRAG request failed after re-auth: {retry_error}") from retry_error
            else:
                self._handle_document_provider_error(e, endpoint)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise RetrieverConnectionError(f"LightRAG server unreachable: {e}") from e
        except Exception as e:
            raise RetrieverConnectionError(f"LightRAG request failed: {e}") from e

        if self._document_provider_available is None:
            self._document_provider_available = True
            logger.info("LightRAG server supports DocumentProvider endpoints")
        return data

-------

packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/lightrag/lightrag_metadata_scope.py
----
"""LightRAG-specific metadata scope."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from ..scope_capability import MetadataScopeLike


class LightRAGMetadataScope(BaseModel):
    """Validated tag-filter scope forwarded to a LightRAG query.

    Operator keys are intentionally opaque because deployed LightRAG variants
    may implement different tag-filter vocabularies.
    """

    model_config = ConfigDict(frozen=True)

    tag_filter: dict[str, list[str]] | None = None

    @field_validator("tag_filter", mode="before")
    @classmethod
    def _detach_tag_filter(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {key: list(tags) if isinstance(tags, list) else tags for key, tags in value.items()}

    @field_validator("tag_filter", mode="after")
    @classmethod
    def _validate_tag_filter(cls, value: dict[str, list[str]] | None) -> dict[str, list[str]] | None:
        if value is None:
            return None
        for operator, tags in value.items():
            if not operator:
                raise ValueError("tag-filter operator keys must be non-empty strings")
            if not tags or any(not tag for tag in tags):
                raise ValueError("tag-filter values must be non-empty lists of non-empty strings")
        return value

    def is_empty(self) -> bool:
        """Return whether this scope carries no active tag filter."""
        return not self.tag_filter

    def is_effective(self) -> bool:
        """Return whether this scope must be forwarded to LightRAG."""
        return not self.is_empty()

    def apply_caller_scope(self, caller_scope: MetadataScopeLike) -> LightRAGMetadataScope:
        """Replace the build-time tag filter with a non-empty caller filter."""
        if not isinstance(caller_scope, LightRAGMetadataScope):
            raise TypeError(f"LightRAGMetadataScope requires matching caller input, got {type(caller_scope).__name__}")
        return caller_scope.model_copy(deep=True) if caller_scope.is_effective() else self.model_copy(deep=True)

    @classmethod
    def from_caller_scope(cls, bundle: Mapping[str, Any]) -> LightRAGMetadataScope | None:
        """Resolve the opaque ``tag_filter`` from a normalized caller bundle."""
        scope = cls(tag_filter=bundle.get("tag_filter"))
        return scope if scope.is_effective() else None

-------

packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/lightrag/lightrag_retriever.py
----
"""LightRAG retriever — supports HTTP and core (direct) engines.

Delegates transport/backend concerns to a ``LightRAGEngine`` implementation
(HTTP or core).  Handles search configuration, response parsing, and
``DocumentProvider`` protocol support.

Engine selection:
- ``LightRAGRetriever(base_url=...)`` — HTTP engine (backward compatible)
- ``LightRAGRetriever(engine=...)`` — any engine (new pattern)
- ``LightRAGRetriever.from_http(...)`` — explicit HTTP engine factory
- ``await LightRAGRetriever.from_core(...)`` — core engine (direct LightRAG instance)
- ``LightRAGRetriever.from_registry(...)`` — named instance from registry
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import fields as dc_fields, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, cast

from sta_agent_core.adapters.auth import AuthProvider
from sta_agent_core.repositories.retrievers.lightrag.lightrag_core_settings import (
    LightRAGCoreSettings,
)

from ..exceptions import RetrieverConnectionError, RetrieverResponseError


if TYPE_CHECKING:
    _F = TypeVar("_F", bound=Callable[..., Any])

    def traceable(**kwargs: Any) -> Callable[[_F], _F]: ...
else:
    from langsmith import traceable

from .engines.protocol import LightRAGEngine
from .lightrag_metadata_scope import LightRAGMetadataScope
from .lightrag_response_parser import _clean_entities, _clean_relationships
from .lightrag_retrieval_chunk import (
    LightRAGEntity,
    LightRAGReference,
    LightRAGRelationship,
    LightRAGRetrievalChunk,
)
from .lightrag_search_config import LightRAGSearchConfig
from .lightrag_search_response import (
    LightRAGQueryMetadata,
    LightRAGSearchResponse,
)


logger = logging.getLogger(__name__)

# Config fields eligible for pass-through via **kwargs in search().
# Excludes fields already handled as explicit named params or via `size`.
_EXTRA_CONFIG_FIELDS = frozenset(
    {f.name for f in dc_fields(LightRAGSearchConfig)} - {"top_k", "documents", "mode", "enable_rerank", "hl_keywords", "ll_keywords"}
)


_TOKEN_TRACE_FIELDS = ("max_total_tokens", "max_entity_tokens", "max_relation_tokens")


def _process_retriever_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process inputs for LangSmith trace display (compact, no large payloads)."""
    return {
        "query": inputs.get("query", ""),
        "size": inputs.get("size", 10),
        "mode": inputs.get("mode"),
        "enable_rerank": inputs.get("enable_rerank"),
        **{f: inputs[f] for f in _TOKEN_TRACE_FIELDS if f in inputs},
    }


def _doc_for_trace(r: Any) -> dict[str, Any]:
    """Build a single document dict for LangSmith trace (compact, no large content)."""
    content = getattr(r, "content", str(r))
    meta = getattr(r, "metadata", None) or {}
    exclude = {"content", "text", "page_content"}
    meta = {k: v for k, v in meta.items() if k not in exclude}
    meta["score"] = getattr(r, "score", None)
    meta["chunk_id"] = getattr(r, "chunk_id", None)
    display_content = content[:500] + "..." if len(content) > 500 else content
    return {"page_content": display_content, "type": "Document", "metadata": meta}


def _process_retriever_outputs(outputs: Any) -> dict[str, Any]:
    """Process LightRAGSearchResponse for LangSmith trace display."""
    if outputs is None:
        return {"num_results": 0}
    results = getattr(outputs, "results", None) or []
    docs = [_doc_for_trace(r) for r in results]
    metadata = getattr(outputs, "metadata", None)
    extra: dict[str, Any] = {}
    if metadata is not None:
        extra["query_mode"] = getattr(metadata, "query_mode", None)
        extra["final_chunks_count"] = getattr(metadata, "final_chunks_count", None)
        extra["total_entities_found"] = getattr(metadata, "total_entities_found", None)
        extra["entities_after_truncation"] = getattr(metadata, "entities_after_truncation", None)
        extra["total_relations_found"] = getattr(metadata, "total_relations_found", None)
        extra["relations_after_truncation"] = getattr(metadata, "relations_after_truncation", None)
        extra["hl_keywords"] = getattr(metadata, "hl_keywords", None) or []
        extra["ll_keywords"] = getattr(metadata, "ll_keywords", None) or []
    return {"num_results": len(docs), "documents": docs, **extra}


def _process_lightrag_get_document_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"doc_id": inputs.get("doc_id", "")}


def _process_lightrag_get_document_outputs(outputs: Any) -> dict[str, Any]:
    chunks = outputs if isinstance(outputs, list) else []
    return {"num_chunks": len(chunks)}


def _process_lightrag_get_chunk_context_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"chunk_id": inputs.get("chunk_id", ""), "window": inputs.get("window", 3)}


def _process_lightrag_get_chunk_range_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": inputs.get("doc_id", ""),
        "start_index": inputs.get("start_index", 0),
        "end_index": inputs.get("end_index", 0),
    }


def _chunk_score(raw: dict[str, Any]) -> float | None:
    """Prefer reranker relevance while preserving a valid zero score."""
    relevance_score = raw.get("relevance_score")
    return relevance_score if relevance_score is not None else raw.get("score")


def _parse_response(response_json: dict[str, Any], clean_response: bool) -> LightRAGSearchResponse:
    """Build LightRAGSearchResponse from engine response dict."""
    data = response_json.get("data", {})
    raw_metadata = response_json.get("metadata", {})

    chunks: list[LightRAGRetrievalChunk] = []
    for chunk in data.get("chunks", []):
        file_path = chunk.get("file_path", "")
        chunk_meta: dict[str, Any] = {"mode": raw_metadata.get("query_mode", "")}
        if file_path:
            chunk_meta["doc"] = file_path
        full_doc_id = chunk.get("full_doc_id", "")
        if full_doc_id:
            chunk_meta["pageId"] = full_doc_id
        chunk_order = chunk.get("chunk_order_index")
        if chunk_order is not None:
            chunk_meta["chunk_index"] = chunk_order
        chunks.append(
            LightRAGRetrievalChunk(
                content=chunk.get("content", ""),
                chunk_id=chunk.get("chunk_id", ""),
                score=_chunk_score(chunk),
                source_url=file_path,
                retriever_type="lightrag",
                metadata=chunk_meta,
                reference_id=chunk.get("reference_id", ""),
            )
        )

    entities = list(data.get("entities", []))
    relationships = list(data.get("relationships", []))
    if clean_response:
        entities = cast(list[LightRAGEntity], _clean_entities(entities))
        relationships = cast(list[LightRAGRelationship], _clean_relationships(relationships))

    # Higher-weight relationships first for downstream consumers (e.g. KG compression).
    relationships.sort(key=lambda r: r.get("weight", 0.0), reverse=True)

    processing = raw_metadata.get("processing_info", {})
    keywords = raw_metadata.get("keywords", {})
    tag_filter = raw_metadata.get("tag_filter", {})
    metadata = LightRAGQueryMetadata(
        query_mode=raw_metadata.get("query_mode", ""),
        hl_keywords=keywords.get("high_level", []),
        ll_keywords=keywords.get("low_level", []),
        total_entities_found=processing.get("total_entities_found", 0),
        entities_after_truncation=processing.get("entities_after_truncation", 0),
        total_relations_found=processing.get("total_relations_found", 0),
        relations_after_truncation=processing.get("relations_after_truncation", 0),
        final_chunks_count=processing.get("final_chunks_count", 0),
        tags=tag_filter,
    )

    refs: list[LightRAGReference] = list(data.get("references", []))
    return LightRAGSearchResponse(
        results=chunks,
        entities=entities,
        relationships=relationships,
        references=refs,
        metadata=metadata,
    )


def _parse_chunk_items(raw_chunks: list[dict[str, Any]]) -> list[LightRAGRetrievalChunk]:
    """Convert raw chunk dicts from engine responses into typed chunks."""
    chunks: list[LightRAGRetrievalChunk] = []
    for raw in raw_chunks:
        file_path = raw.get("file_path", "")
        full_doc_id = raw.get("full_doc_id", "")
        meta: dict[str, Any] = {
            "full_doc_id": full_doc_id,
            "pageId": full_doc_id,
            "chunk_index": raw.get("chunk_order_index", 0),
            "tokens": raw.get("tokens", 0),
        }
        if file_path:
            meta["doc"] = file_path
        chunks.append(
            LightRAGRetrievalChunk(
                content=raw.get("content", ""),
                chunk_id=raw.get("chunk_id", ""),
                score=_chunk_score(raw),
                source_url=file_path,
                retriever_type="lightrag",
                metadata=meta,
            )
        )
    return chunks


class LightRAGRetriever:
    """Retriever that queries LightRAG via an engine strategy.

    Conforms to ``BaseRetriever`` protocol via structural typing.
    Returns ``LightRAGSearchResponse`` (chunks + entities + relationships + metadata).

    Also satisfies ``DocumentProvider`` — the core engine supports it
    natively via KV store access; the HTTP engine probes for server-side
    endpoints and degrades gracefully.
    """

    supports_metadata_scope: ClassVar[Literal[True]] = True

    @staticmethod
    def resolve_caller_scope(bundle: Mapping[str, Any]) -> LightRAGMetadataScope | None:
        """Resolve only the opaque ``tag_filter`` from the caller bundle."""
        return LightRAGMetadataScope.from_caller_scope(bundle)

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        search_config: LightRAGSearchConfig | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        clean_response: bool = False,
        distributed_tracing: bool = True,
        *,
        engine: LightRAGEngine | None = None,
        auth_provider: AuthProvider | None = None,
        default_scope: LightRAGMetadataScope | None = None,
        use_twin_api: bool = False,
    ) -> None:
        """Create a retriever with either an explicit engine or HTTP params.

        For backward compatibility, passing ``base_url`` creates an HTTP
        engine automatically.  New code should use factory methods or
        pass ``engine`` directly.

        Args:
            auth_provider: Optional ``AuthProvider`` for HTTP engine auth.
                Takes precedence over ``api_key`` when both are provided.
            use_twin_api: Use ``/api/query/data`` instead of ``/query/data``
                for the HTTP engine. Ignored when an explicit engine is passed.
        """
        if engine is not None:
            self._engine = engine
        elif base_url is not None:
            from .engines.http_engine import LightRAGHttpEngine

            self._engine = LightRAGHttpEngine(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                distributed_tracing=distributed_tracing,
                auth_provider=auth_provider,
                use_twin_api=use_twin_api,
            )
        else:
            raise ValueError("Either 'engine' or 'base_url' must be provided. Use LightRAGRetriever.from_http() or LightRAGRetriever.from_core().")

        if default_scope is not None and not isinstance(default_scope, LightRAGMetadataScope):
            raise TypeError(f"LightRAGRetriever requires LightRAGMetadataScope, got {type(default_scope).__name__}")

        self._search_config = search_config or LightRAGSearchConfig()
        self._clean_response = clean_response
        self._default_scope = default_scope.model_copy(deep=True) if default_scope is not None else None

    # --- Factory methods -----------------------------------------------------

    @classmethod
    def from_http(
        cls,
        base_url: str,
        api_key: str | None = None,
        search_config: LightRAGSearchConfig | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        clean_response: bool = True,
        distributed_tracing: bool = True,
        *,
        auth_provider: AuthProvider | None = None,
        default_scope: LightRAGMetadataScope | None = None,
        use_twin_api: bool = False,
    ) -> LightRAGRetriever:
        """Create retriever with HTTP engine (legacy/migration path)."""
        from .engines.http_engine import LightRAGHttpEngine

        engine = LightRAGHttpEngine(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            distributed_tracing=distributed_tracing,
            auth_provider=auth_provider,
            use_twin_api=use_twin_api,
        )
        return cls(
            engine=engine,
            search_config=search_config,
            clean_response=clean_response,
            default_scope=default_scope,
        )

    @classmethod
    async def from_core(
        cls,
        settings: LightRAGCoreSettings,
        search_config: LightRAGSearchConfig | None = None,
        clean_response: bool = True,
        default_scope: LightRAGMetadataScope | None = None,
    ) -> LightRAGRetriever:
        """Create retriever with a core engine (direct LightRAG instance).

        Args:
            settings: A ``LightRAGCoreSettings`` instance.
            search_config: Optional search configuration overrides.
            clean_response: Whether to clean entities/relationships.
        """
        from .engines.core_engine import LightRAGCoreEngine

        engine = await LightRAGCoreEngine.create(settings)
        return cls(
            engine=engine,
            search_config=search_config,
            clean_response=clean_response,
            default_scope=default_scope,
        )

    @classmethod
    def from_registry(
        cls,
        instance_name: str,
        search_config: LightRAGSearchConfig | None = None,
        clean_response: bool = True,
        default_scope: LightRAGMetadataScope | None = None,
    ) -> LightRAGRetriever:
        """Create retriever from a registered engine instance.

        The engine must already be registered via
        ``LightRAGInstanceRegistry.register()``.

        Args:
            instance_name: Name of the registered instance (e.g. "docs").
        """
        from .instance_registry import LightRAGInstanceRegistry

        engine = LightRAGInstanceRegistry.get(instance_name)
        return cls(
            engine=engine,
            search_config=search_config,
            clean_response=clean_response,
            default_scope=default_scope,
        )

    # --- BaseRetriever -------------------------------------------------------

    @traceable(
        run_type="retriever",
        name="lightrag_search",
        process_inputs=_process_retriever_inputs,
        process_outputs=_process_retriever_outputs,
    )
    async def search(
        self,
        query: str,
        size: int = 10,
        *,
        mode: str | None = None,
        enable_rerank: bool | None = None,
        hl_keywords: list[str] | None = None,
        ll_keywords: list[str] | None = None,
        metadata_scope: LightRAGMetadataScope | None = None,
        **kwargs: Any,
    ) -> LightRAGSearchResponse:
        """Search LightRAG. Returns LightRAGSearchResponse."""
        overlay = {
            k: v
            for k, v in (
                ("mode", mode),
                ("enable_rerank", enable_rerank),
                ("hl_keywords", hl_keywords),
                ("ll_keywords", ll_keywords),
            )
            if v is not None
        }
        # Forward extra kwargs that match config fields (e.g. token budgets
        # from tool-level budget splitting: max_total_tokens, max_entity_tokens).
        # Excludes fields already handled above and top_k (set via size param).
        overlay.update((k, v) for k, v in kwargs.items() if k in _EXTRA_CONFIG_FIELDS and v is not None)
        config = replace(self._search_config, top_k=size, **overlay)
        search_kwargs = config.to_search_kwargs()

        if metadata_scope is not None and not isinstance(metadata_scope, LightRAGMetadataScope):
            raise TypeError(f"LightRAGRetriever requires LightRAGMetadataScope, got {type(metadata_scope).__name__}")
        active_scope = metadata_scope if metadata_scope is not None and not metadata_scope.is_empty() else self._default_scope
        if active_scope is not None and not active_scope.is_empty():
            search_kwargs["tag_filter"] = active_scope.tag_filter

        try:
            raw_response = await self._engine.query(query, search_kwargs)
        except (RetrieverConnectionError, RetrieverResponseError):
            raise
        except Exception as e:
            raise RetrieverConnectionError(f"LightRAG request failed: {e}") from e

        try:
            return _parse_response(raw_response, self._clean_response)
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise RetrieverResponseError(f"Malformed LightRAG response: {e}") from e

    async def close(self) -> None:
        await self._engine.close()

    async def __aenter__(self) -> LightRAGRetriever:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # --- DocumentProvider ----------------------------------------------------

    @traceable(
        run_type="retriever",
        name="lightrag_get_document",
        process_inputs=_process_lightrag_get_document_inputs,
        process_outputs=_process_lightrag_get_document_outputs,
    )
    async def get_document(self, doc_id: str) -> list[LightRAGRetrievalChunk]:
        """Fetch all chunks of a document, ordered by position."""
        data = await self._engine.get_document_chunks(doc_id)
        return _parse_chunk_items(data.get("chunks", []))

    @traceable(
        run_type="retriever",
        name="lightrag_get_chunk_context",
        process_inputs=_process_lightrag_get_chunk_context_inputs,
        process_outputs=_process_lightrag_get_document_outputs,
    )
    async def get_chunk_context(self, chunk_id: str, window: int = 3) -> list[LightRAGRetrievalChunk]:
        """Fetch neighbouring chunks around a given chunk."""
        data = await self._engine.get_chunk_context(chunk_id, window=window)
        return _parse_chunk_items(data.get("chunks", []))

    @traceable(
        run_type="retriever",
        name="lightrag_get_chunk_range",
        process_inputs=_process_lightrag_get_chunk_range_inputs,
        process_outputs=_process_lightrag_get_document_outputs,
    )
    async def get_chunk_range(self, doc_id: str, start_index: int, end_index: int) -> list[LightRAGRetrievalChunk]:
        """Fetch a positional range of chunks from a document."""
        data = await self._engine.get_document_chunks(doc_id, start=start_index, end=end_index)
        return _parse_chunk_items(data.get("chunks", []))

-------

packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/lightrag/lightrag_search_response.py
----
"""LightRAG search response and query metadata.

Extends SearchResponse with query-level KG context (entities, relationships).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..search_response import SearchResponse
from .lightrag_retrieval_chunk import (
    LightRAGEntity,
    LightRAGReference,
    LightRAGRelationship,
    LightRAGRetrievalChunk,
)


@dataclass
class LightRAGQueryMetadata:
    """Diagnostics from the /query/data response."""

    query_mode: str = ""
    hl_keywords: list[str] = field(default_factory=list)
    ll_keywords: list[str] = field(default_factory=list)
    total_entities_found: int = 0
    entities_after_truncation: int = 0
    total_relations_found: int = 0
    relations_after_truncation: int = 0
    final_chunks_count: int = 0
    tags: dict[str, list[str]] | None = None


@dataclass(frozen=True)
class LightRAGSearchResponse(SearchResponse[LightRAGRetrievalChunk]):
    """Full structured response from LightRAG /query/data.

    Iterable over chunks for generic consumers. KG-aware consumers
    use .entities, .relationships, .metadata.

    Frozen dataclass — inherits covariance and serialization from SearchResponse.
    All fields (TypedDicts, dataclass) serialize natively with LangGraph's checkpoint.

    Return types (when clean_response=False):
        entities: List of LightRAGEntity (TypedDict). Each has entity_name, entity_type,
            description, and optionally source_id (chunk id), file_path, reference_id.
            Used for lineage and citations; source_id/file_path may be <SEP>-delimited
            when the entity is derived from multiple chunks or files.
        relationships: List of LightRAGRelationship (TypedDict), ordered by weight
            descending. Each has src_id, tgt_id, description, and optionally keywords,
            weight, source_id, file_path, reference_id. source_id and file_path can
            contain <SEP>-delimited multi-values (e.g. multiple chunk ids or paths).
        references: List of LightRAGReference (reference_id -> file_path) for resolving
            reference_id on entities/relationships to document paths.
        metadata: Query diagnostics (query_mode, keyword counts, truncation stats).
    """

    entities: list[LightRAGEntity] = field(default_factory=list)
    relationships: list[LightRAGRelationship] = field(default_factory=list)
    references: list[LightRAGReference] = field(default_factory=list)
    metadata: LightRAGQueryMetadata = field(default_factory=LightRAGQueryMetadata)

    def __init__(
        self,
        results: Sequence[LightRAGRetrievalChunk] = (),
        *,
        entities: list[LightRAGEntity] | None = None,
        relationships: list[LightRAGRelationship] | None = None,
        references: list[LightRAGReference] | None = None,
        metadata: LightRAGQueryMetadata | None = None,
    ) -> None:
        super().__init__(results=results)
        # frozen=True requires object.__setattr__ in custom __init__
        object.__setattr__(self, "entities", entities or [])
        object.__setattr__(self, "relationships", relationships or [])
        object.__setattr__(self, "references", references or [])
        object.__setattr__(self, "metadata", metadata or LightRAGQueryMetadata())

-------

packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/scope_capability.py
----
"""Capability protocol for retrievers that honor ``metadata_scope=`` in search().

The Knowledge Agent's metadata scope is enforced by passing a backend-specific
scope instance to `retriever.search(metadata_scope=...)`.
The `BaseRetriever` protocol uses `**kwargs` for universality, so a backend
that doesn't explicitly handle this kwarg will silently drop the scope — the
"ceiling" advertised by `default_scope` becomes a no-op. That is a
trust-boundary failure (tenant isolation, auth scoping).

Backends opt in by declaring a class-level marker and resolving the normalized
caller bundle into their own scope model::

    class ElasticRetriever:
        supports_metadata_scope: ClassVar[Literal[True]] = True

        def resolve_caller_scope(
            self,
            bundle: Mapping[str, Any],
        ) -> MetadataScope | None:
            ...
        ...

The retriever tool factory probes the structural contract via
``isinstance(retriever, SupportsMetadataScope)`` at agent-build time and
refuses to wire scope features onto a backend that has not opted in. Mirrors
the ``DocumentProvider`` capability-protocol pattern in this same package.

Why a `Literal[True]` annotation:

- ``isinstance(...)`` against a ``@runtime_checkable Protocol`` verifies
  attribute and method presence at runtime — marker value truthiness is not
  part of the check.
  The factory enforces value-level truthiness as a second gate.
- ``Literal[True]`` is a static-typing signal: pyright/mypy will reject a
  declaration like ``supports_metadata_scope: ClassVar[bool] = False`` on a
  class that's intended to satisfy this protocol. Catches accidental misuse
  at type-check time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NotRequired, Protocol, TypedDict, runtime_checkable


class CallerMetadataScope(TypedDict, total=False):
    """Normalized per-request scope shared by retriever backends.

    Filter fields are backend-specific instructions: a retriever applies only
    the subset it recognizes. ``include_without_caller_scope`` names retriever
    entries that the planner may offer even when their backend resolves none of
    the supplied filters; their build-time scope still applies.
    """

    doc_ids: NotRequired[list[str]]
    apcode: NotRequired[list[str]]
    app_name: NotRequired[list[str]]
    entity: NotRequired[list[str]]
    tag_filter: NotRequired[dict[str, list[str]]]
    include_without_caller_scope: NotRequired[list[str]]


class MetadataScopeLike(Protocol):
    """Common behavior required from backend-specific metadata scopes."""

    def is_effective(self) -> bool:
        """Return whether the scope must be passed to the retriever."""
        ...

    def apply_caller_scope(self, caller_scope: MetadataScopeLike) -> MetadataScopeLike:
        """Combine a build-time scope with a caller-resolved scope."""
        ...


@runtime_checkable
class SupportsMetadataScope(Protocol):
    """Retriever honors metadata scope and resolves caller-scope bundles.

    Backends opt in by setting ``supports_metadata_scope`` in the class body,
    implementing ``resolve_caller_scope``, and accepting their returned scope
    type in ``search(metadata_scope=...)``. The resolver receives a normalized
    cross-backend bundle and returns only the filter subset the backend knows.
    """

    supports_metadata_scope: Literal[True]

    def resolve_caller_scope(self, bundle: Mapping[str, Any]) -> MetadataScopeLike | None:
        """Resolve recognized caller fields into this backend's scope model."""
        ...

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/AGENTS.md
----
# AGENTS.md — knowledge_agent

Multi-retriever evidence-gathering pipeline. Plans sub-queries → retrieves in parallel
→ collects/dedups → compresses to `Finding`s with `Citation`s → optionally reviews
coverage and iterates → optionally synthesizes a cited answer.

Single topology, runtime-gated. Same compiled graph serves all
`mode` × `search_depth` combinations — consumers opt UP from the cheap default.

> **Status:** breaking changes may land without notice (see consumer doc disclaimer).

## Entry points

| What | Where |
|---|---|
| Public API exports | `__init__.py` |
| Core factory (programmatic) | `knowledge_agent_graph.py:create_knowledge_agent` |
| Spec-based factory (JSONL/`langgraph.json`) | `knowledge_agent_catalog.py:get_knowledge_agent_graph` |
| `langgraph.json` graph names | `knowledge_elastic`, `knowledge_lightrag_http`, `knowledge_lightrag_core` (lazy `@cache` getters in catalog) |
| Consumer doc | `docs/consuming/knowledge-agent.md` |
| Tests | `tests/test_ai_engine/agents/knowledge_agent/` |

> Catalog wrappers (`get_knowledge_agent_graph`, lazy getters) are NOT re-exported
> from `__init__.py` (CI-05) — importing the package must not trigger network calls.

## Graph topology (single, runtime-gated)

```
START → reset_turn → plan_queries ──┬── tool_node → collect → [expand?] → compress
                                    │                                      │
                                    │                           route_after_compress
                                    │                            ↙ (fast)   ↘ (deep+)
                                    │                         exit         review_evidence
                                    │                                       ↙   │   ↘
                                    │                                   output expand plan_queries
                                    │           [answer-mode exit]
                                    │              synthesize → [thorough?] → review_answer ⇄ synthesize
                                    │                              ↓                 ↓
                                    │                            output            output
                                    └── output → END  (early exit when plan emits no tool_calls)
```

- **Build-time gate:** `expand` node only added when `agent_config.expand.enabled=True`.
- **Runtime gates** (via `KnowledgeAgentContext`): `mode` controls exit (output vs synthesize);
  `search_depth` controls whether reviews fire.
- **`reset_turn`** sits only on the `START` edge. It clears per-turn accumulators
  (`findings`, `retrieved_responses`, `answer*`, iteration/expansion counters, …) so a
  checkpointed thread does not carry one conversation turn's evidence into the next. The
  `review_evidence → plan_queries` outer loop re-enters `plan_queries` directly, bypassing
  `reset_turn`, so within-turn accumulation across review iterations is preserved.

## Planning strategy (tool_calls | structured)

`PlanQueriesNode` produces the `AIMessage.tool_calls` that the ToolNode executes.
How those calls are generated is selected by `PlanConfig.planning_strategy`:

| Strategy | How | When |
|---|---|---|
| `"tool_calls"` (**default**) | Binds retriever tools to the planner model; consumes its native `tool_calls`. Bound schema constrains tool names + args, so no validation round-trip. Transient failures retried via `.with_retry(stop_after_attempt=tool_call_retry_attempts)` (default 2). | Default. Needs a model that can emit **parallel** tool calls to fan out N retrievers per turn. |
| `"structured"` | Model returns a validated structured plan (with conversational validate-and-retry), converted to `tool_calls`. Guarantees N calls regardless of the model. | Models that can't emit concurrent tool calls (**gpt-oss**), or when you want the extra validation round-trip. |

**Model recommendation:** for the default `"tool_calls"` strategy use a
parallel-tool-call-capable model for the `planning` task — **`mistral-small-2603`**
is the recommended planner. **gpt-oss cannot emit concurrent tool calls**, so on
gpt-oss either accept fewer-calls-per-turn or set `planning_strategy="structured"`.
The strategy is build-time config (no `KA_PLANNING_STRATEGY` env knob yet).

**Planning-failure fallback.** When planning (after its own retries) yields no
usable retriever calls **and** no text, `PlanQueriesNode` would otherwise emit a
blank `AIMessage` that routes straight to `output`, leaving the consumer with an
empty last message. Instead it substitutes a non-empty `PLAN_FAILED_MESSAGE` and
sets `state["plan_failed"]=True`; `OutputNode` surfaces that as
`result.metadata["plan_failed"]` (and does **not** mislabel the fallback as a
genuine `direct_response`). A no-call turn *with* content (greeting / clarification)
is untouched. This is distinct from `metadata["no_results"]` (retriever tools ran
but returned zero evidence) and `metadata["direct_response"]` (the model answered
without retrieving).

## State / context / output

All in `knowledge_agent_state.py`.

| Schema | Purpose |
|---|---|
| `KnowledgeAgentInputState` | `messages` OR `query` (last `HumanMessage` wins if both) |
| `KnowledgeAgentState` | Full workflow state — internal tool-call thread, retrieved responses (per-retriever list), collected chunks, findings (custom reducer w/ `FindingsUpdate(replace=True)` for re-compression), KG/chunk dedup hashes, coverage, answer fields, iteration/expansion counters |
| `KnowledgeAgentOutputState` | `result` (Findings or Answer dataclass) + flat `findings`, `coverage`, `collected_chunks` |
| `KnowledgeAgentContext` | `mode`, `search_depth`, `max_iterations`, `auto_pull_document`, `model_configs`, `streaming_enabled`, collect overrides |

Output dataclasses (in `knowledge_agent_types.py`):
- **evidence mode** → `KnowledgeAgentFindings(query, findings, coverage, retriever_names, …)`
- **answer mode** → `KnowledgeAgentAnswer(evidence=…, answer, answer_citations, answer_review)` —
  `answer_citations` is the cited subset only, ordered `[1]`, `[2]`, …

## Mode × search_depth matrix

```
search_depth   mode       review_evidence  review_answer  synthesis
fast (def)     evidence*  skip             skip           skip
fast           answer     skip             skip           yes
deep           evidence   on               skip           skip
deep           answer     on               skip           yes
thorough       answer     on               on             yes
```
`*` cheapest path: `plan → retrieve → collect → compress → output`.

Resolution chain (highest wins): `runtime.context["mode"|"search_depth"]` →
`KnowledgeAgentConfig.mode|search_depth` → `"evidence"`/`"fast"`.

## Files map

```
knowledge_agent/
├── knowledge_agent_graph.py        # create_knowledge_agent() + _wire_graph()
├── knowledge_agent_catalog.py      # get_knowledge_agent_graph() + langgraph.json lazy getters
├── knowledge_agent_config.py       # KnowledgeAgentConfig + Plan/Collect/Compress/Expand/Review/Synthesis
├── knowledge_agent_state.py        # State / Input / Output / Context schemas + reducers
├── knowledge_agent_types.py        # RetrieverEntry, Citation, Finding, CoverageAssessment,
│                                   #   KnowledgeAgentFindings/Answer, FetchTarget, KnowledgeNodeTask
├── knowledge_agent_routing.py      # route_after_plan, make_route_after_review,
│                                   #   make_route_after_answer_review, _resolve_*
├── knowledge_agent_retrievers.py   # RetrieverSpec, build_entries_from_specs,
│                                   #   create_{mock,elastic,lightrag}_entry, _LazyCoreLightRAGEngine
├── knowledge_agent_prompts.py      # All node prompts (planner, compressor, reviewer, synthesizer)
├── utils/                          # Cross-cutting helpers (stateless, dependency-light):
│                                   #   citation_resolver.py — maps LLM `[N]` refs → ordered Citation list
│                                   #   findings_format.py    — render Findings → prompt-ready text
│                                   #   doc_url.py            — document URL helpers
│                                   #   trace_utils.py        — traceable_node() LangSmith span wrapper
│                                   #   env_model_configs.py  — sparse KA_<TASK>_<KEY> env overlay loader
├── nodes/                          # ResetTurnNode, PlanQueriesNode, CollectNode, CompressNode,
│                                   #   ExpandNode, ReviewEvidenceNode, ReviewAnswerNode, SynthesizeNode, OutputNode
├── tools/retriever_tool_factory.py # create_retriever_tools(entries) — one BaseTool per entry
├── compression/                    # Compressor protocol, ChunkCompressor (LLM/passthrough/dynamic),
│                                   #   KGCompressor, RetrieverEvidence, CompressResult
└── eval/                           # ka_eval_suite, ka_evaluators, ka_dataset, ka_context_builder
```

> Public API (`CitationResolver`, etc.) is re-exported from the package
> `__init__`; import helpers by submodule path (`from ..utils.<mod> import ...`).
> The `knowledge_agent_example.py` runnable lives at
> `examples/sta_agent_engine/knowledge_agent/knowledge_agent_example.py` (examples never live in
> the package source tree).

## Setup chain (3 layers)

1. **Data layer** (`sta_agent_core`): `BaseRetriever` + `BaseSearchConfig` (e.g.
   `ElasticRetriever`/`ElasticSearchConfig`, `LightRAGRetriever`/`LightRAGSearchConfig`).
2. **Agent layer**: `RetrieverEntry(name, description, retriever, search_config, compressors=…)`
   — one per tool. `description` is critical: the LLM uses it for tool selection.
   Use builders (`create_elastic_entry`, `create_lightrag_entry`, `create_mock_entry`)
   or specs (`build_entries_from_specs([{type, name, …}])`).
3. **Graph factory**: `create_knowledge_agent(entries, model=…, config=KnowledgeAgentConfig(…))`.

LightRAG token budgets (`max_total_tokens` etc.) split automatically across parallel
tool calls — each call reads sibling count from state and divides the budget (floor 1000).

## Per-task model resolution (7 layers; 1-5 merge per-key, highest wins)

Layers 1-5 are config dicts that **merge per-key** — the highest-priority layer
that defines a given key wins it. So a shared slot can supply `model` while a
task-specific slot supplies only `max_tokens`/`temperature` (or vice versa).
Layers 6-7 are pre-built-instance fallbacks, used only when 1-5 yield no
model-bearing config.

1. `runtime.context["model_configs"][task]` (e.g. `"planning"`, `"compression"`,
   `"review"`, `"synthesis"`, `"verification"` — see `KnowledgeNodeTask`)
2. `runtime.context["model_configs"]["all"]` (runtime shared slot)
3. `KnowledgeAgentConfig.task_model_defaults[task]` — env `KA_<TASK>_*` (package
   defaults carry non-provider knobs only — `max_tokens` + `temperature`; env
   overrides fold in here when built via `KnowledgeAgentConfig.from_env()`)
4. `runtime.context["model_configs"]["default"]` (runtime shared fallback)
5. `KnowledgeAgentConfig.task_model_defaults["default"]` — env `KA_DEFAULT_MODEL`
6. `default_model` arg to `create_knowledge_agent()` (pre-built instance)
7. `create_chat_model()` engine-wide env fallback — `LLM_PROVIDER` +
   `<PROVIDER>_{API_KEY,BASE_URL,MODEL}`

`KA_DEFAULT_MODEL` (layer 5) supplies the model to *every* task while each
task's package defaults contribute its knobs — they merge, so the env default
is no longer shadowed by the knobs-only per-task entries.

KA is provider/model-agnostic — the package defaults intentionally do NOT
hardcode a provider. Choose a backend at one of: layer 1/2/4 (caller), layer 3/5
via env (`KA_<TASK>_PROVIDER=...` / `KA_DEFAULT_MODEL=...`), layer 6 (build-time
`default_model`), or layer 7 (engine-wide `LLM_PROVIDER`). Env overrides remain a
*default* — runtime `context.model_configs` still wins.

Env keys per task: `KA_<TASK>_{PROVIDER,BASE_URL,MODEL,API_KEY,MAX_TOKENS,TEMPERATURE}`.
Values land as plain `create_chat_model` kwargs (not the `ModelConfig` TypedDict
shape — `temperature`/`max_tokens`, not `model_temperature`/`model_max_tokens`).

Wiring:

- **Catalog / `graphs.jsonl` deployments** (`get_knowledge_agent_graph`):
  env loading is **on by default** (`load_env_model_overrides=True`). Set
  `KA_<TASK>_PROVIDER` / `KA_<TASK>_MODEL` etc. in the process env and the
  factory picks them up automatically.
- **Direct library use** (`create_knowledge_agent`): env loading is
  **opt-in** — callers build a config with
  `KnowledgeAgentConfig.from_env(base=...)` themselves before passing it.

## Compression strategies

`ChunkCompressionStrategy` (in `knowledge_agent_config.py`):
- `LLM` — always LLM-compress chunks into `Finding`s with structured output
- `PASSTHROUGH` — deterministic 1 chunk → 1 Finding (zero LLM calls)
- `DYNAMIC` (default) — passthrough below char threshold, LLM above; threshold derives
  from `evidence_token_budget * 4` (default 100k tokens → 400k chars)

**Failure fallback (LLM path).** When LLM structured-output compression exhausts its
retries for a chunk group — validation never passes, or every attempt raises (provider
read timeouts are funnelled into `OutputValidationError` by `ainvoke_with_output_validation`)
— the group is **not** dropped. It falls back to the deterministic passthrough path, tagged
`compression_mode="passthrough_fallback"` (distinct from intentional `"passthrough"`). The
evidence is preserved with citations intact, but it is **not** LLM-synthesized, and
`OutputNode` surfaces `result.metadata["compression_degraded"] = True` for the turn.
A fallback finding is **not** re-compressed in the same pass that produced it (re-hitting a
just-failed provider is pointless), but `maybe_recompress` retries it via the LLM path on a
**later KA round** when over the dynamic threshold — so a transient outage self-heals across
iterations (bounded by `max_iterations`), falling back again each round until it succeeds.

`KGCompressionStrategy` (LightRAG only): per-entity grouping or batch. KG dedup hashes
on `(src_id, tgt_id)` so new relationships for known entities still compress.

## Iteration & expansion loops

- **Outer loop** (`max_iterations`, default 3): `plan → … → review_evidence` decides whether
  to plan more queries when `coverage.sufficient=False`.
- **Inner expansion loop** (`expand.max_expansion_rounds`, default 1): when review emits
  `fetch_targets` and expand is enabled and `auto_pull_document=False`, ExpandNode pulls
  the targeted documents/chunk windows before re-compressing.
- **`auto_pull_document=True`** short-circuits the inner loop — pulls full docs on the
  first pass instead. Reduces round-trips, increases first-pass cost.

## Metadata scope (build-time / user / runtime query)

`RetrieverEntry` carries a single optional `metadata_scope:
MetadataScopeConfig | None` field. `None` means the entry is not
scope-aware (the legacy default). When set, the three sub-fields compose
into the scope passed to `retriever.search(metadata_scope=...)`:

| Field | Scope kind | Role |
|---|---|---|
| `metadata_scope.default: MetadataScopeLike \| None` | **build-time** | Backend-specific default baked in by the consumer at agent-build time and mirrored on the retriever for bare `search()` calls. Elasticsearch treats it as a filter ceiling; LightRAG treats it as its default opaque tag filter. **It is still not a server-side trust boundary** — a caller speaking directly to a remote backend may bypass client-side enforcement. |
| `metadata_scope.exposed_axes: tuple[str, ...]` | **runtime query** | Per-axis opt-in for planner-tool args. Subset of `("apcode", "app_name", "entity")`. Adds those axes to the generated tool schema. Empty tuple (default) keeps the tool query-only. |
| `metadata_scope.value_resolver: MetadataValueResolver \| None` | **runtime query** | **Optional** vocab cleaner — routes raw LLM-emitted values through `resolve(axis, raw)` before they're unioned into `*_boost` axes via `MetadataScope.add_boosts`. When absent, raw values pass through verbatim (see below). **Boost-only by construction** — runtime-query-scope axes cannot widen or narrow the build-time filter ceiling regardless of whether a resolver is wired. |

Builders (`create_elastic_entry`, `create_elastic_rag_proxy_entry`) keep
flat `default_scope=` / `expose_metadata_args=` / `metadata_value_resolver=`
kwargs and pack them into a `MetadataScopeConfig` internally. Spec JSON
keeps the flat shape (`default_scope` / `expose_metadata_args` as top-level
dict keys); the normalizer in `knowledge_agent_retrievers.py` packs them
into a `MetadataScopeConfig` before passing to the builder.

Resolver semantics:

- **Resolver attached**: planner values go through `resolve(axis, raw)`, get
  canonicalized (fuzzy match against the index vocab), and emit
  `state["resolution_warnings"]` on misses. Best for direct retrievers where
  the client owns the index.
- **No resolver (opt-in passthrough)**: planner values are stripped, then unioned
  into the matching `*_boost` axis verbatim. Boost is additive
  (`should` clauses) — a typo or out-of-vocab value contributes nothing
  (silent precision miss), but never widens the filter ceiling. Appropriate
  for proxy/gateway retrievers where the client doesn't reach the index, or
  for any deployment that accepts the precision trade-off. **Observability
  trade-off**: hits log at `DEBUG` only; misses produce no signal at all
  (no state warning, no ToolMessage annotation, no log). A passthrough
  deployment cannot distinguish "boost matched" from "boost missed due to
  typo" without inspecting the retriever response itself.

Build-time validation:

- Wiring `metadata_scope.default` or `metadata_scope.exposed_axes` onto a
  retriever that does NOT satisfy `SupportsMetadataScope` (see
  `sta_agent_core.repositories.retrievers.scope_capability`) raises `TypeError`
  at tool-factory time. The `BaseRetriever` protocol uses `**kwargs`, so a
  backend that doesn't honor `metadata_scope=...` would silently drop the
  scope and the build-time default would become a no-op. `ElasticRetriever`,
  `ElasticRagRetriever`, and `LightRAGRetriever` opt in; `MockRetriever` does not.

Warnings (fuzzy hits, out-of-vocab drops) flow only when a resolver is attached:
1. `ToolMessage` summary tail (planner sees on the next turn).
2. `state["resolution_warnings"]` (accumulated; OutputNode mirrors into
   `KnowledgeAgentFindings.metadata["warnings"]`).
3. `logger.warning("metadata_resolution", extra={...})` for observability.

**User scope** (per-user auth scoping via `user_scope_mode`) is deferred — see
`memory_bank/creative_phases/knowledge_agent/creative_phase_2026-05-11_ka_scope_three_layer.md`
for the full design contract and the engineering example at
`examples/sta_agent_engine/knowledge_agent/knowledge_agent_three_layer_scope_example.py`.

### Caller-supplied request scope (`ka_metadata_scope`)

A fourth scope kind, distinct from the three above: a per-request bundle the
**caller** seeds in input state — not a planner tool arg, not build-time config.
Filter keys are backend-specific instructions. They do not automatically
constrain a backend that does not recognize them.

- **Channel:** `ka_metadata_scope` — run-scoped (`UntrackedValue`, never
  checkpointed), input-only (`OmitFromOutput`). Shape: the `KaMetadataScope`
  TypedDict `{doc_ids, apcode, app_name, entity, tag_filter,
  include_without_caller_scope}`. Elasticsearch recognizes the first four;
  LightRAG recognizes only `tag_filter` and forwards its operator keys without
  interpretation. A `*_boost` key is dropped and warned.
- **How it arrives:** a caller seeds it directly —
  `graph.ainvoke({"query": ..., "ka_metadata_scope": {"doc_ids": [...]}})`.
  Under the orchestrator, the deepagents `task` tool carries it across the
  bridge declared by `KnowledgeBridgeMiddleware` (whose `before_agent` also
  surfaces a `doc_ids` selection to the planner as a `<system_reminder>` so it
  delegates instead of answering from general knowledge — see the orchestrator
  `AGENTS.md` § State bridge). There is no header/producer middleware — input
  state is the contract.
- **How it applies:** each `SupportsMetadataScope` retriever resolves its own
  subset. Elasticsearch maps `doc_ids` → `doc_filter` and the other recognized
  axes to `*_filter`, then intersects with its build-time scope. LightRAG maps
  only `tag_filter`; a non-empty caller tag filter replaces its build-time tag
  filter. Tag operator names are backend-owned and opaque to the KA.
- **Per-retriever opt-in:** only entries with `accepts_caller_scope=True` read
  it (default `False`). Set it on broad entries (e.g. `general_doc`); leave
  scoped entries (e.g. `twin_project_doc`) opted out so caller filtering never
  crosses their boundary. Also requires the backend to honor `metadata_scope=`
  (`SupportsMetadataScope`) — an unsupported backend warns and skips.
- **Planner tool restriction:** no scope (or a payload that normalizes to
  empty) binds the normal full tool set. A non-empty scope binds only
  `accepts_caller_scope=True` entries whose resolver returns an effective
  backend scope. If none match, no retriever tool is bound; the KA never falls
  back to an entry that would ignore the filters. To intentionally offer an
  entry without a caller filter, add its entry name to
  `include_without_caller_scope`; its build-time scope still applies. When
  entry metadata is unavailable, the planner cannot resolve applicability and
  retains the existing warn-and-keep-all fallback.
- **Public symbols:** `KaMetadataScope`, `KA_METADATA_SCOPE_KEY` (exported from
  the package `__init__`).

## Knobs cheat sheet

| Want | Set |
|---|---|
| Cheapest path | defaults (mode=evidence, search_depth=fast) |
| Cited answer | `context={"mode": "answer"}` |
| Reliable evidence | `context={"search_depth": "deep"}` |
| Critical answer w/ faithfulness check | `context={"mode": "answer", "search_depth": "thorough"}` |
| Skip inner expansion loop, pull whole docs upfront | `context={"auto_pull_document": true}` |
| Cheaper planner | `context={"model_configs": {"planning": {"model": "gpt-4o-mini"}}}` |
| Planner can't emit parallel tool calls (e.g. gpt-oss) | `KnowledgeAgentConfig(plan=PlanConfig(planning_strategy="structured"))` |
| Suppress streaming when nested as a tool | `context={"streaming_enabled": false}` |
| Nested as a tool / sub-agent (terse answer, no custom citation events) | `context={"mode": "answer", "subagent_mode": true, "streaming_enabled": false}` |
| Cap query loop | `context={"max_iterations": 1}` |
| Tune/disable the synthesis input safety cap | `KnowledgeAgentConfig(synthesis=SynthesisConfig(max_synthesis_input_tokens=120_000))` (0 = off) |

## Commands

```bash
# all KA tests
uv run pytest tests/test_ai_engine/agents/knowledge_agent/ -v

# routing / topology
uv run pytest tests/test_ai_engine/agents/knowledge_agent/test_routing.py tests/test_ai_engine/agents/knowledge_agent/test_review_loop_unit.py -v

# compression methods
uv run pytest tests/test_ai_engine/agents/knowledge_agent/test_chunk_compression_method.py tests/test_ai_engine/agents/knowledge_agent/test_kg_compression_method.py tests/test_ai_engine/agents/knowledge_agent/test_dynamic_compression.py -v

# offline integration (graph end-to-end with FakeChatModel)
uv run pytest tests/test_ai_engine/agents/knowledge_agent/test_knowledge_agent_unit.py tests/test_ai_engine/agents/knowledge_agent/test_evidence_integration.py -v

# eval suite
uv run pytest tests/test_ai_engine/agents/knowledge_agent/test_ka_evaluators.py tests/test_ai_engine/agents/knowledge_agent/test_ka_dataset.py -v

# Studio (live) — pick a graph from langgraph.json
make langgraph/dev   # then open knowledge_elastic / knowledge_lightrag_http
```

## Pitfalls (KA-specific)

- **Synthesis input safety cap**: `SynthesizeNode` bounds its LLM call (system +
  human message) to `synthesis.max_synthesis_input_tokens * 4` chars (default
  120k tokens, `0` disables). Findings are shed lowest-confidence-first via
  `CitationResolver.prepare_context(max_chars=...)`, then a final hard clip
  guarantees the ceiling even if a single finding exceeds it — so a degraded turn
  (e.g. an oversized `passthrough_fallback` group that escaped the evidence
  budget) can't hard-fail the model on context size. The cap is deliberately
  above `evidence_token_budget` (100k) — it's a last-resort guard, not the normal
  path. When it fires, `OutputNode` sets `result.metadata["synthesis_truncated"]`.

- **Next-steps are model-derived in `fast` mode**: the synthesis prompts ask for a
  "Next steps" pointer (follow-up searches / specific documents to read) when the
  evidence leaves a gap. Concrete `<gaps>`/`<suggestions>`/`<fetch_targets>` only
  populate the `<evidence_review>` block when review ran (`search_depth=deep`/
  `thorough`). In the default `fast` path (and the orchestrator's KA delegation,
  which is wired to `fast`), `coverage` is `None` → `<status>not_run</status>`, so
  next-steps are derived by the model from the findings alone. They stay grounded
  (the prompt constrains suggestions to document identities present in the
  evidence and to THIS query's gap), but they are weaker than the review-backed
  targets — raise `search_depth` when you need review-grounded follow-ups.
  The synthesis prompt also receives a `<searches_already_run>` block (the
  queries the planner actually searched this turn, derived read-only from the
  `messages` tool-call thread by `utils/executed_queries.py`) and is told not to
  re-suggest any of them — so next-steps point only at genuinely new angles or a
  specific document to fetch.
- **Adjacency is surfaced labeled, not dropped (synthesis)**: the synthesis
  directness rule forbids presenting adjacency AS the answer (a rollback is not a
  failover), but it does NOT license dropping it. Related/adjacent material is
  developed in full, labeled "related, not the exact ask"; the no-answer contract
  leads with a TLDR only on a BIG/core gap (skipped on a minor one) and otherwise
  develops every finding. This is synthesis-scoped (`_SYNTH_DIRECTNESS_ACTION`);
  compress/review keep their stricter adjacency handling.
- **CI-05 — never re-export catalog wrappers from `__init__.py`**: `get_knowledge_agent_graph`
  and the `get_knowledge_*_instance` getters trigger retriever construction (network).
  Reference them by module path in JSONL/`langgraph.json` only.
- **CI-06 — retriever `top_k` must be set on the entry's `search_config`**: the tool factory
  passes `to_search_kwargs()` to `retriever.search()` but the LightRAG retriever ignores `size`
  there. Set `top_k` on `LightRAGSearchConfig` at entry-build time, not via tool args.
- **`messages` vs `query`**: external callers use `query` (or last `HumanMessage`). `messages`
  in state is the internal tool-call thread (AIMessage + ToolMessages) — not a chat history
  surface. TODO(phase2) splits these.
- **Query quality is the caller's responsibility**: `plan_queries` plans retrieval from the
  caller-supplied text. It decomposes a multi-topic question into one focused search per
  topic, but does not reformulate or semantically expand the wording — it will not rephrase
  a vague query, invent synonyms, or infer unnamed entities. Retrieval quality is bounded by
  the input. Callers should send a specific, self-contained query that names the entities
  involved; a terse fragment retrieves poorly. See the `KnowledgeAgentInputState` docstring
  and `docs/consuming/knowledge-agent.md` § Input schema.
- **Messages channel as conversational trace**: every node that drives the loop writes a
  short message so the channel reads as a readable trace.
  `plan_queries` emits the tool-calling `AIMessage` (existing behaviour); `tool_node`
  emits `ToolMessage`s (existing behaviour);
  `review_evidence` emits a one-line narration AIMessage of its verdict
  (sufficient / gaps / no evidence); `output` emits the final answer (answer mode) or
  the canned no-results AIMessage. Routing remains driven by structured state fields,
  not the messages.
  **`SynthesizeNode` never writes `messages`** — `OutputNode` is the single emitter of
  the answer. This matters in `search_depth="thorough"`: `synthesize` can run several
  times (a `review_answer` rejection routes back for another attempt), so emitting per
  attempt would leak drafts the graph later replaces. `OutputNode` runs once, after the
  review loop settles, so only the accepted answer reaches `messages`.
  **`messages[-1].content` guarantee scope**: the "last message is the synthesized
  answer" contract only holds for `mode="answer"` runs. In `mode="evidence"` the last
  message is whatever the final node wrote (typically the review narration), and the
  consumer is expected to read `state["result"]` directly. The deep-agent
  `task(description)` integration path is answer-mode only. Exception caveat: if a
  node raises before `OutputNode` is reached, `messages` reflects the state of the
  prior node — orchestrators that catch graph exceptions should not trust
  `messages[-1]` as a success signal.
- **Findings reducer**: defaults to append. Re-compression must wrap with
  `FindingsUpdate(findings, replace=True)` to overwrite accumulated passthrough findings.
- **Per-turn state reset**: `ResetTurnNode` (on the `START` edge) clears turn-scoped
  state so a checkpointed thread does not leak one turn's findings/answer into the next.
  When you add a new state field that should not survive into the next conversation turn,
  add it to `ResetTurnNode`'s reset dict. Fields with an accumulating reducer (`add`-style
  or custom) must be cleared with `langgraph.types.Overwrite(...)`, which bypasses the
  reducer and writes the value directly; plain last-writer-wins fields are reset by
  writing the empty value. `OutputNode` detects answer mode via `answer_attempt > 0`
  (a turn-scoped signal `SynthesizeNode` sets and `ResetTurnNode` clears) — not by the
  presence of the `answer` key, which persists across turns on a checkpointed thread.
- **ToolNode parallelism**: `plan_queries` may emit N tool_calls to the same retriever;
  `retrieved_responses` reducer keys by retriever name and appends. Token budgets auto-split.
- **`include_original_query`**: on iteration 1, one extra tool call per selected retriever
  is injected with the user's exact query (preserves intent alongside LLM-decomposed queries).
- **Same topology rule** (LangGraph Server factory invariant): `create_knowledge_agent` must
  return the same nodes/edges regardless of caller — vary tools and prompts via `entries`,
  not graph structure.
- **`subagent_mode` suppresses citation custom events**: when `context={"subagent_mode": true}`,
  `SynthesizeNode` skips the `citation_map` / `citation_order` custom stream events and the
  token-level `[Fn]→[N]` rewrite. Inline `[N]` markers and `answer_citations` are still
  populated (post-hoc `CitationResolver.resolve()` runs once). Frontends that subscribe to
  the live citation events will see nothing — switch to default mode if you need them streamed.
- **Inline citations are markdown links + a `Sources:` block — every mode**: after
  `CitationResolver.resolve()`, `SynthesizeNode` runs `linkify_citations()` to rewrite each
  inline `[N]` marker into a markdown link `[N](url)` (markers with no url / out of range
  stay bare) and insert a `, ` separator between any two directly-adjacent markers
  (`[1][2]` → `[1](u), [2](u)`), then appends a plain-text `Sources:` block (`[N] [Title](url)` per cited
  source, deduped). Both apply regardless of `subagent_mode`. This makes the answer
  self-contained — a consumer reading only `messages` / `state["answer"]` gets clickable
  citations in any markdown renderer, no custom citation stream events needed. The
  structured `answer_citations` is still populated. A KA-native frontend that styles bare
  `[N]` markers must tolerate the `[N](url)` form — `citation_renderer._CITATION_NUM_RE`
  consumes the optional trailing `(url)` so a pre-linkified marker still renders as one
  pill badge. The streamed `token` custom events still carry bare `[N]` (the
  `StreamingCitationResolver` is not linkified); the final committed answer carries the
  links, and the answer-message-id dedup means the final replaces the streamed view.
- **`OutputNode` emits the final answer as an `AIMessage` on `messages`**: every answer-mode
  run appends the synthesized answer to the `messages` channel so orchestrator callers
  consuming KA via `astream(stream_mode="messages")` (or the deep-agent
  `result["messages"][-1].content` pattern) receive the answer as the last message. The
  `messages` key is on `KnowledgeAgentOutputState`, so it survives the output-schema
  filter and reaches `graph.ainvoke()` consumers.
- **Answer message id is shared with the token stream**: `SynthesizeNode` captures the
  LLM's own response id (`chunk.id` from the streamed `AIMessageChunk`s, or `response.id`
  on the ainvoke fallback — the same id LangGraph stamps on `stream_mode="messages"`
  chunks), stores it in `answer_message_id`, and stamps it on every `token` custom stream
  event. `OutputNode` reuses it as the id of the final answer `AIMessage`. A consumer
  streaming tokens can match the final message to the stream and deduplicate. The
  `citation_map` event carries no id (emitted before the LLM call, id not yet known);
  `citation_order` carries it. In `search_depth="thorough"` the last accepted synthesis
  attempt's id wins (last-writer-wins field); earlier rejected drafts streamed under
  different ids. The id is `None` on the no-results short-circuit, or if the model
  produced no id — `add_messages` then assigns a fresh id. The answer `OutputNode` emits
  includes the `Sources:` block (see above) verbatim; KA-native frontends can still read
  `state["answer"]` / `state["result"]` directly.
- **Answer-mode result shape is stable**: in `mode="answer"` `OutputNode` always returns a
  `KnowledgeAgentAnswer` (never a bare `KnowledgeAgentFindings`), so consumers reading
  `result.answer` have a stable type. On the no-results path the canned
  `"No relevant information found for this query."` becomes the `answer` text.
  `mode="evidence"` returns `KnowledgeAgentFindings`. The exception is the
  direct-response early-exit (LLM answered with no tool calls) — `OutputNode` returns
  `KnowledgeAgentFindings` there since no synthesis ran.
- **No-results AIMessage and `metadata["no_results"]`**: when retriever tools are invoked
  but yield zero findings (and synthesis didn't produce an answer), `OutputNode` tags
  `result.metadata["no_results"] = True` and appends the canned message
  `"No relevant information found for this query."` to `messages`. Distinct from the
  direct-response path (no findings AND no retrieval — the LLM answered without calling
  tools), which preserves the original AIMessage on `messages` and exposes
  `metadata["direct_response"]`.

## Refs

- Consumer doc: `docs/consuming/knowledge-agent.md`
- Hosted retriever doc: `docs/consuming/elastic-rag.md`, `docs/consuming/lightrag-http.md`
- Design history: `memory_bank/creative_phases/knowledge_agent/` (see `MEMORY.md` index)
- Engine-wide patterns: `packages/sta_agent_engine/AGENTS.md`
- LangGraph patterns: `.claude/skills/langgraph-agent-builder/SKILL.md`

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/knowledge_agent_retrievers.py
----
"""Knowledge Agent retriever builders and MockRetriever.

Provides:
- MockRetriever: Protocol-compliant mock for testing (BaseRetriever[RetrievalChunk])
- RetrieverSpec: JSON-serializable spec for building RetrieverEntry
- create_mock_entry, create_elastic_entry, create_lightrag_entry: entry builders
  with explicit config overrides (override > env vars > defaults)
- _LazyCoreLightRAGEngine: Lazy-initializing wrapper for LightRAG core engine
  (defers async init to first use, keeps builder chain sync)
- RETRIEVER_BUILDERS: registry mapping type string to builder
- build_entries_from_specs: build list[RetrieverEntry] from list[RetrieverSpec | dict]
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from sta_agent_core.adapters.auth import AuthProvider
from sta_agent_core.repositories.retrievers import BaseSearchConfig
from sta_agent_core.repositories.retrievers.elasticsearch.metadata_scope import MetadataScope
from sta_agent_core.repositories.retrievers.lightrag.lightrag_metadata_scope import LightRAGMetadataScope
from sta_agent_core.repositories.retrievers.metadata_value_resolver import MetadataValueResolver
from sta_agent_core.repositories.retrievers.scope_capability import MetadataScopeLike
from sta_agent_engine.retrievers import MockRetriever

from .knowledge_agent_types import MetadataScopeConfig, RetrieverEntry


logger = logging.getLogger(__name__)


__all__ = [
    "MockRetriever",
    "RetrieverSpec",
    "create_mock_entry",
    "create_elastic_entry",
    "create_elastic_rag_proxy_entry",
    "create_lightrag_entry",
    "RETRIEVER_BUILDERS",
    "build_entries_from_specs",
]


# ---------------------------------------------------------------------------
# RetrieverSpec — JSON-serializable spec for catalog/JSONL
# ---------------------------------------------------------------------------


@dataclass
class RetrieverSpec:
    """JSON-serializable spec for building a RetrieverEntry.

    config keys are passed as overrides to the entry builder;
    resolution order: explicit config > env vars > defaults.
    """

    type: str  # "elastic", "lightrag", "mock"
    name: str
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# _LazyCoreLightRAGEngine — deferred async init for core engine
# ---------------------------------------------------------------------------


class _LazyCoreLightRAGEngine:
    """Lazy-initializing wrapper around ``LightRAGCoreEngine``.

    Stores ``LightRAGCoreSettings`` at build time (sync) and creates the
    real ``LightRAGCoreEngine`` on the first async method call.  This keeps
    ``create_lightrag_entry`` and ``build_entries_from_specs`` synchronous
    while supporting core engine mode from JSONL graph configs.

    Implements ``LightRAGEngine`` protocol via structural typing.
    """

    def __init__(
        self,
        core_env_file: str = ".lightrag.env",
        workspace: str | None = None,
    ) -> None:
        self._core_env_file = core_env_file
        self._workspace_override = workspace
        self._engine: Any = None

    async def _ensure_engine(self) -> Any:
        if self._engine is None:
            from sta_agent_core.repositories.retrievers.lightrag import (
                LightRAGCoreSettings,
            )
            from sta_agent_core.repositories.retrievers.lightrag.engines.core_engine import (
                LightRAGCoreEngine,
            )

            settings = LightRAGCoreSettings.for_instance(self._core_env_file)
            if self._workspace_override:
                settings = settings.model_copy(update={"workspace": self._workspace_override})
            self._engine = await LightRAGCoreEngine.create(settings)
            logger.info(
                "LightRAG core engine initialized (env_file=%s, workspace=%s)",
                self._core_env_file,
                settings.workspace,
            )
        return self._engine

    async def query(self, query: str, search_kwargs: dict[str, Any]) -> dict[str, Any]:
        engine = await self._ensure_engine()
        return await engine.query(query, search_kwargs)

    async def get_document_chunks(
        self,
        doc_id: str,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any]:
        engine = await self._ensure_engine()
        return await engine.get_document_chunks(doc_id, start=start, end=end)

    async def get_chunk_context(
        self,
        chunk_id: str,
        window: int = 3,
    ) -> dict[str, Any]:
        engine = await self._ensure_engine()
        return await engine.get_chunk_context(chunk_id, window=window)

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.close()
            self._engine = None


# ---------------------------------------------------------------------------
# Entry builders — explicit overrides override env vars
# ---------------------------------------------------------------------------


def create_mock_entry(
    name: str = "mock",
    description: str = "Mock retriever for testing",
    *,
    num_results: int = 3,
    examples: list[str] | None = None,
    **kwargs: Any,
) -> RetrieverEntry:
    """Create a RetrieverEntry for MockRetriever. Unknown kwargs ignored.

    ``MockRetriever`` does not declare ``SupportsMetadataScope``, so metadata
    scope kwargs (``default_scope`` / ``expose_metadata_args`` /
    ``metadata_value_resolver``) are explicitly rejected — a programmatic
    caller wiring scope onto this backend would otherwise silently drop into
    ``**kwargs`` (the scope never reaches ``RetrieverEntry``, so the factory
    gate cannot fire either).
    """
    _reject_scope_kwargs("create_mock_entry", kwargs)
    if kwargs:
        logger.debug("create_mock_entry ignoring unknown config keys: %s", list(kwargs))
    return RetrieverEntry(
        name=name,
        description=description,
        retriever=MockRetriever(name=name, num_results=num_results),
        examples=examples or [],
    )


def create_elastic_entry(
    name: str = "elastic_docs",
    description: str = "Search documentation using Elasticsearch hybrid (BM25 + kNN)",
    *,
    es_host: str | None = None,
    es_index: str | None = None,
    field_config: dict[str, Any] | None = None,
    search_config: BaseSearchConfig | None = None,
    expansion_hint: Any = None,
    fusion_strategy: Any = None,
    examples: list[str] | None = None,
    default_scope: MetadataScope | dict[str, Any] | None = None,
    expose_metadata_args: list[str] | None = None,
    metadata_value_resolver: MetadataValueResolver | None = None,
    accepts_caller_scope: bool = False,
    **kwargs: Any,
) -> RetrieverEntry:
    """Create a RetrieverEntry for ElasticRetriever.

    DocumentProvider (expansion) uses ``metadata.pageId`` and ``metadata.chunk_index``
    by default.  Pass ``field_config`` to override for non-standard indices.

    Step 9 Track B — construction-time strategy selection:

    - ``expansion_hint`` accepts ``ExpansionStrategy`` or its string form
      (``"pass"`` / ``"keyword"`` / ``"multi"`` / ...). When set, it is
      forwarded to ``ExpansionSettings(hint=...)`` so this deployment's
      retriever is built with the matching ``QueryExpander`` (or none for
      ``PASS``), overriding ``RETRIEVER_EXPANSION_HINT``.
    - ``fusion_strategy`` accepts ``FusionStrategy`` or its string form
      (``"rrf_only"`` / ``"rrf_reranker"`` / ...). When set, it is merged
      into ``search_config`` via ``dataclasses.replace`` so other fields
      on a pre-built config (``top_k``, ``rank_constant``, ...) are
      preserved.

    Both kwargs are construction-only — per-call overrides go through the
    gateway context (``elastic_rag``) or the core retriever context; the
    knowledge agent itself stays strategy-agnostic.
    """
    from sta_agent_core.repositories.retrievers.elasticsearch import (
        ElasticSearchConfig,
        FusionStrategy,
    )
    from sta_agent_core.repositories.retrievers.elasticsearch.query_expansion import (
        ExpansionStrategy,
    )
    from sta_agent_engine.retrievers import build_elastic_retriever_from_env
    from sta_agent_engine.retrievers.expansion_settings import ExpansionSettings

    if kwargs:
        logger.debug("create_elastic_entry ignoring unknown config keys: %s", list(kwargs))

    if isinstance(default_scope, dict):
        default_scope = MetadataScope.model_validate(default_scope)
    elif default_scope is not None and not isinstance(default_scope, MetadataScope):
        raise TypeError(f"create_elastic_entry default_scope must be MetadataScope or dict, got {type(default_scope).__name__}")

    resolved_config = cast(ElasticSearchConfig, search_config) if search_config is not None else ElasticSearchConfig()
    if fusion_strategy is not None:
        coerced_fusion = fusion_strategy if isinstance(fusion_strategy, FusionStrategy) else FusionStrategy(fusion_strategy)
        resolved_config = dataclasses.replace(resolved_config, fusion_strategy=coerced_fusion)

    if expansion_hint is not None:
        coerced_hint = expansion_hint if isinstance(expansion_hint, ExpansionStrategy) else ExpansionStrategy(expansion_hint)
        expansion_settings = ExpansionSettings(hint=coerced_hint)
    else:
        expansion_settings = ExpansionSettings()

    retriever = build_elastic_retriever_from_env(
        search_config=resolved_config,
        es_host=es_host,
        es_index=es_index,
        field_config=field_config,
        expansion_settings=expansion_settings,
        # Bake the build-time ceiling into the retriever too, so a bare
        # ``retriever.search(query=...)`` enforces it client-side — not only the
        # tool's per-call merge path (which keeps reading entry.metadata_scope).
        default_scope=default_scope,
    )
    return RetrieverEntry(
        name=name,
        description=description,
        retriever=retriever,
        search_config=resolved_config,
        examples=examples or [],
        metadata_scope=_build_scope_config(
            default_scope=default_scope,
            expose_metadata_args=expose_metadata_args,
            metadata_value_resolver=metadata_value_resolver,
        ),
        accepts_caller_scope=accepts_caller_scope,
    )


def _build_auth_provider(
    settings: Any,
) -> AuthProvider:
    """Build the appropriate AuthProvider from LightRAGRetrieverSettings."""
    from sta_agent_core.adapters.auth import JWTAuth, NoAuth, StaticBearerAuth

    mode = settings.auth_mode
    if mode == "static":
        header_format = getattr(settings, "auth_header_format", "bearer")
        return StaticBearerAuth(settings.api_key, header_format=header_format)
    if mode == "jwt":
        return JWTAuth(
            login_url=settings.resolved_login_url,
            username=settings.auth_username,
            password=settings.auth_password,
            token_ttl=settings.auth_token_ttl,
        )
    return NoAuth()


def create_lightrag_entry(
    name: str = "lightrag_kg",
    description: str = "Query knowledge graph for entities, relationships, and contextual chunks",
    *,
    engine: str = "http",
    base_url: str | None = None,
    core_env_file: str | None = None,
    workspace: str | None = None,
    default_scope: LightRAGMetadataScope | dict[str, Any] | None = None,
    accepts_caller_scope: bool = True,
    use_twin_api: bool = False,
    examples: list[str] | None = None,
    **kwargs: Any,
) -> RetrieverEntry:
    """Create a RetrieverEntry for LightRAGRetriever.

    Args:
        engine: ``"http"`` (default) for HTTP engine, ``"core"`` for
            in-process LightRAG via ``LightRAGCoreEngine`` (lazy init).
        base_url: HTTP base URL. Overrides ``RETRIEVER_LIGHTRAG_BASE_URL``
            env var.  Ignored when ``engine="core"``.
        core_env_file: Path to env file for core engine settings
            (e.g. ``".lightrag.env"``).  Ignored when ``engine="http"``.
        workspace: Override the workspace name from the env file.
            Allows switching workspaces without separate env files.
            Ignored when ``engine="http"``.
        default_scope: Optional build-time LightRAG tag filter. A non-empty
            caller ``tag_filter`` replaces it for that request.
        accepts_caller_scope: Whether to apply caller-seeded ``tag_filter``
            values. Enabled by default for LightRAG entries.
        use_twin_api: Use ``/api/query/data`` instead of ``/query/data`` for
            the HTTP engine. Ignored when ``engine="core"``.

    JSONL config examples::

        // HTTP mode (default)
        {"type": "lightrag", "name": "lightrag_kg"}

        // Core engine mode
        {"type": "lightrag", "name": "lightrag_kg",
         "config": {"engine": "core", "core_env_file": ".lightrag.env"}}

        // Core engine with workspace override
        {"type": "lightrag", "name": "lightrag_kg",
         "config": {"engine": "core", "workspace": "docs_v2"}}
    """
    _reject_scope_kwargs("create_lightrag_entry", kwargs)
    if kwargs:
        logger.debug("create_lightrag_entry ignoring unknown config keys: %s", list(kwargs))

    if isinstance(default_scope, dict):
        default_scope = LightRAGMetadataScope.model_validate(default_scope)
    elif default_scope is not None and not isinstance(default_scope, LightRAGMetadataScope):
        raise TypeError(f"create_lightrag_entry default_scope must be LightRAGMetadataScope or dict, got {type(default_scope).__name__}")

    from sta_agent_core.repositories.retrievers.lightrag import (
        LightRAGRetriever,
        LightRAGSearchConfig,
    )

    from .compression import ChunkCompressor, KGCompressor

    if engine == "core":
        lazy_engine = _LazyCoreLightRAGEngine(
            core_env_file=core_env_file or ".lightrag.env",
            workspace=workspace,
        )
        retriever = LightRAGRetriever(engine=lazy_engine, default_scope=default_scope)
    else:
        from sta_agent_core.repositories.retrievers.lightrag import LightRAGRetrieverSettings

        settings_kwargs: dict[str, Any] = {"base_url": base_url} if base_url else {}
        settings = LightRAGRetrieverSettings(**settings_kwargs)
        url = settings.base_url
        auth_provider = _build_auth_provider(settings)
        retriever = LightRAGRetriever(
            base_url=url,
            auth_provider=auth_provider,
            default_scope=default_scope,
            use_twin_api=use_twin_api,
        )

    return RetrieverEntry(
        name=name,
        description=description,
        retriever=retriever,
        search_config=LightRAGSearchConfig(),
        examples=examples or [],
        compressors=[ChunkCompressor(), KGCompressor()],
        metadata_scope=_build_scope_config(
            default_scope=default_scope,
            expose_metadata_args=None,
            metadata_value_resolver=None,
        ),
        accepts_caller_scope=accepts_caller_scope,
    )


# Raw-credential kwargs the proxy builder rejects loudly — keep this list in
# sync with any credential-VALUE field a user might be tempted to drop into a
# JSONL spec's `config`. Env-var-name pointers (e.g. ``api_key_env``) are NOT
# secrets and are accepted.
_PROXY_BUILDER_SECRET_KWARGS: frozenset[str] = frozenset({"api_key", "token", "password"})


def create_elastic_rag_proxy_entry(
    name: str = "elastic_rag_proxy",
    description: str = "Search documentation via the deployed elastic_rag gateway proxy",
    *,
    gateway_url: str | None = None,
    api_key_env: str | None = None,
    top_k: int = 10,
    examples: list[str] | None = None,
    default_scope: MetadataScope | dict[str, Any] | None = None,
    expose_metadata_args: list[str] | None = None,
    metadata_value_resolver: MetadataValueResolver | None = None,
    **kwargs: Any,
) -> RetrieverEntry:
    """Create a RetrieverEntry for the :class:`ElasticRagRetriever` gateway proxy.

    Credentials are env-only: the proxy adapter reads
    ``ELASTIC_RAG_PROXY_RETRIEVER_API_KEY`` by default. Per-spec
    ``api_key_env`` overrides the env-var NAME (the spec carries a pointer,
    not the secret value) — useful when one process talks to multiple
    gateways with different keys.

    Args:
        name: Tool name surfaced to the planner.
        description: Tool description the planner reads for selection.
        gateway_url: Optional override of the gateway base URL. When ``None``,
            ``ElasticRagProxyRetrieverSettings`` reads
            ``ELASTIC_RAG_PROXY_RETRIEVER_GATEWAY_URL``. URL only — never
            embed credentials.
        api_key_env: NAME of an env var holding the per-spec API key. When set
            and the env var resolves non-empty, that value is forwarded to
            the proxy. When ``None`` (default) or the var is unset, the
            adapter falls back to ``ELASTIC_RAG_PROXY_RETRIEVER_API_KEY``.
        top_k: Default ``size`` for ``search()`` when caller does not override.
        examples: Optional few-shot examples for prompt injection.
        default_scope: Build-time filter ceiling. Baked into both the entry (the
            tool's per-call merge path) and the retriever, so a bare
            ``retriever.search(query=...)`` forwards it too. Client-side
            ergonomics — not a server-side trust boundary (the gateway forwards
            scope, it does not enforce caller identity against it).
        expose_metadata_args: Per-axis opt-in for runtime planner-tool args
            (subset of ``["apcode", "app_name", "entity"]``). Boost-only.
        metadata_value_resolver: Optional client-side vocab canonicalizer.
            JSON specs cannot carry this (Python Protocol instance) — see
            ``_validate_spec_scope_config``.

    Raises:
        ValueError: If ``kwargs`` contains a raw-credential key (``api_key``,
            ``token``, ``password``). Set them via env instead.
    """
    leaked = _PROXY_BUILDER_SECRET_KWARGS & kwargs.keys()
    if leaked:
        raise ValueError(
            f"create_elastic_rag_proxy_entry rejects credential kwargs {sorted(leaked)} — "
            "set them via env (e.g. ELASTIC_RAG_PROXY_RETRIEVER_API_KEY) or pass an "
            "env-var NAME via `api_key_env=` instead. graphs.jsonl is checked in and "
            "the frontend renders factory_args in the sidebar."
        )
    if kwargs:
        logger.debug("create_elastic_rag_proxy_entry ignoring unknown config keys: %s", list(kwargs))

    if isinstance(default_scope, dict):
        default_scope = MetadataScope.model_validate(default_scope)
    elif default_scope is not None and not isinstance(default_scope, MetadataScope):
        raise TypeError(f"create_elastic_rag_proxy_entry default_scope must be MetadataScope or dict, got {type(default_scope).__name__}")

    from sta_agent_core.repositories.retrievers.elastic_rag_proxy import (
        ElasticRagProxyRetrieverSettings,
        ElasticRagRetriever,
    )

    settings_kwargs: dict[str, Any] = {"default_top_k": top_k}
    if gateway_url:
        settings_kwargs["gateway_url"] = gateway_url
    if api_key_env:
        api_key_value = os.getenv(api_key_env)
        if api_key_value:
            settings_kwargs["api_key"] = api_key_value
        else:
            logger.warning(
                "create_elastic_rag_proxy_entry: api_key_env=%r is set but the env var is empty/unset on entry name=%r — "
                "falling back to ELASTIC_RAG_PROXY_RETRIEVER_API_KEY (or no key).",
                api_key_env,
                name,
            )
    retriever = ElasticRagRetriever(
        settings=ElasticRagProxyRetrieverSettings(**settings_kwargs),
        # Bake the build-time ceiling into the proxy so a bare
        # ``retriever.search(query=...)`` forwards it on the wire. This is
        # client-side ergonomics, not server-side enforcement — the gateway
        # forwards scope, it does not yet enforce caller identity against it.
        default_scope=default_scope,
    )

    return RetrieverEntry(
        name=name,
        description=description,
        retriever=retriever,
        examples=examples or [],
        metadata_scope=_build_scope_config(
            default_scope=default_scope,
            expose_metadata_args=expose_metadata_args,
            metadata_value_resolver=metadata_value_resolver,
        ),
    )


# ---------------------------------------------------------------------------
# Registry and build_entries_from_specs
# ---------------------------------------------------------------------------

RETRIEVER_BUILDERS: dict[str, Callable[..., RetrieverEntry]] = {
    "elastic": create_elastic_entry,
    "lightrag": create_lightrag_entry,
    "elastic_rag_proxy": create_elastic_rag_proxy_entry,
    "mock": create_mock_entry,
}


def build_entries_from_specs(
    specs: Sequence[RetrieverSpec | dict[str, Any]],
) -> list[RetrieverEntry]:
    """Build RetrieverEntry list from RetrieverSpec or dict specs.

    Each spec must have "type" and "name"; "description" and "config" are optional.
    config is passed as kwargs to the builder (e.g. base_url, es_index).

    Scope config from JSON follows the same builder path as programmatic input:

    - ``default_scope``: a JSON spec carries this as a ``dict`` (e.g.
      ``{"apcode_filter": "BCEF"}``). Each backend builder coerces it to its
      own concrete scope model and rejects incompatible values.
    - ``metadata_value_resolver``: a JSON spec cannot carry a Python
      Protocol object; rejected with a typed ``TypeError`` referring the
      consumer to the programmatic builder. (A future resolver-registry
      pattern could make this spec-friendly — out of scope here.)
    - ``expose_metadata_args``: already a JSON-friendly ``list[str]``; no
      coercion needed.
    """
    entries: list[RetrieverEntry] = []
    for spec in specs:
        if isinstance(spec, dict):
            spec = RetrieverSpec(
                type=spec["type"],
                name=spec["name"],
                description=spec.get("description", ""),
                config=spec.get("config", {}),
                examples=spec.get("examples", []),
            )
        builder = RETRIEVER_BUILDERS.get(spec.type)
        if builder is None:
            raise ValueError(f"Unknown retriever type {spec.type!r} for entry {spec.name!r}. Known types: {sorted(RETRIEVER_BUILDERS)}")
        config = _validate_spec_scope_config(spec.config, spec.name)
        kwargs: dict[str, Any] = {
            "name": spec.name,
            "description": spec.description or "",
            "examples": spec.examples,
        }
        kwargs.update(config)
        entries.append(builder(**kwargs))
    return entries


_SCOPE_CONFIG_KEYS: tuple[str, ...] = ("default_scope", "expose_metadata_args", "metadata_value_resolver")


def _build_scope_config(
    *,
    default_scope: MetadataScopeLike | None,
    expose_metadata_args: list[str] | None,
    metadata_value_resolver: MetadataValueResolver | None,
) -> MetadataScopeConfig | None:
    """Pack the flat builder kwargs into a :class:`MetadataScopeConfig`.

    Returns ``None`` when no scope feature is set so the entry stays
    "not scope-aware" — the legacy default that lets non-scope-aware
    backends through the tool-factory gate untouched.
    """
    if default_scope is None and not expose_metadata_args and metadata_value_resolver is None:
        return None
    return MetadataScopeConfig(
        default=default_scope,
        exposed_axes=tuple(expose_metadata_args) if expose_metadata_args else (),
        value_resolver=metadata_value_resolver,
    )


def _reject_scope_kwargs(builder_name: str, kwargs: dict[str, Any]) -> None:
    """Loud-reject metadata-scope kwargs from non-scope-aware builders.

    The builder ``**kwargs`` catch-all would otherwise swallow scope keys at
    DEBUG level, producing a silent no-op at search time. This is the single
    source of truth for "is this scope wiring acceptable on this backend?" —
    spec-driven callers go through the same builder so they get the same
    rejection.
    """
    leaked = [k for k in _SCOPE_CONFIG_KEYS if k in kwargs]
    if not leaked:
        return
    raise TypeError(
        f"{builder_name}: metadata-scope kwargs {leaked!r} are not supported on this "
        "retriever type — the backend does not declare SupportsMetadataScope."
    )


def _validate_spec_scope_config(config: dict[str, Any], spec_name: str) -> dict[str, Any]:
    """Validate scope fields that cannot be represented by a JSON spec.

    Each backend builder validates and coerces its own ``default_scope`` because
    only that builder knows the concrete scope model.
    - ``metadata_value_resolver``: ``dict`` cannot carry a Python Protocol
      instance; rejected with a typed error pointing at the programmatic
      builder.

    Type-based rejection of scope keys on non-scope-aware backends is NOT
    done here — that's the builder's job via ``_reject_scope_kwargs`` (one
    source of truth, fires for both spec and programmatic callers).
    """
    if "metadata_value_resolver" in config:
        raise TypeError(
            f"spec name={spec_name!r}: 'metadata_value_resolver' cannot be set via spec config — "
            "JSON specs cannot carry Python Protocol instances. Build the entry programmatically "
            "via the relevant create_* builder, or use a future resolver-registry pattern."
        )
    return config

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/knowledge_agent_types.py
----
"""Core data types for the Knowledge Agent.

Provides:
- RetrieverEntry: Registry entry that binds a retriever with metadata for dispatch
- Citation: Source pointer attached to findings or answers
- Finding: Synthesized claim with supporting citations
- CoverageAssessment: Structured LLM output for evidence sufficiency review
- CompressedFinding: Pydantic model for LLM structured output in compression
- KnowledgeAgentFindings: Evidence-mode output bundle
- KnowledgeAgentAnswer: Answer-mode output bundle (extends findings with synthesis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from sta_agent_core.repositories import BaseRetriever, BaseSearchConfig, RetrievalChunk
from sta_agent_core.repositories.retrievers.metadata_value_resolver import MetadataValueResolver
from sta_agent_core.repositories.retrievers.scope_capability import MetadataScopeLike


if TYPE_CHECKING:
    from .compression.types import Compressor


# ---------------------------------------------------------------------------
# Dimensional type aliases — used across KA config, routing, eval, catalog
# ---------------------------------------------------------------------------

SearchDepth = Literal["fast", "deep", "thorough"]
"""Controls the review pipeline: fast (skip), deep (evidence review), thorough (+ faithfulness)."""

KAMode = Literal["evidence", "answer"]
"""Output shape: evidence (raw findings) or answer (synthesized answer)."""

CompressionStrategy = Literal["dynamic", "llm"]
"""Chunk compression strategy: dynamic (auto-decide) or llm (always LLM)."""

AutoPull = Literal["on", "off"]
"""Full-document pull upfront: on (skip expansion loop) or off (normal)."""

PASSTHROUGH_FALLBACK_MODE = "passthrough_fallback"
"""``Finding.compression_mode`` marker for evidence rescued by the deterministic
passthrough path after LLM compression exhausted its retries (e.g. provider read
timeouts). Distinct from ``"passthrough"`` so a degraded turn stays observable
(``result.metadata["compression_degraded"]``). It is **not** re-compressed in the
same pass that produced it (re-hitting a just-failed provider is pointless), but a
later KA round retries it via the LLM path when over the dynamic threshold."""

RetrieverAxis = Literal["both", "elastic", "lightrag"]
"""Which retrievers to use in evaluation."""


# ---------------------------------------------------------------------------
# RetrieverEntry — binds a retriever instance with dispatch metadata
# ---------------------------------------------------------------------------


def _default_compressors() -> list:
    """Lazy default: [ChunkCompressor()] — avoids circular import."""
    from .compression.chunk_compressor import ChunkCompressor

    return [ChunkCompressor()]


@dataclass(frozen=True)
class MetadataScopeConfig:
    """Bundle of metadata-scope plumbing for a single :class:`RetrieverEntry`.

    Groups the three scope-related knobs into one immutable record so an
    entry's scope wiring is a single attribute and the tool factory can
    snapshot it without juggling sibling fields. ``None`` on
    :attr:`RetrieverEntry.metadata_scope` means "this entry is not scope-aware"
    — the legacy default.

    Attributes:
        default: Build-time backend-specific scope. Elasticsearch treats this
            as an inviolable filter ceiling; a non-empty LightRAG caller tag
            filter replaces its opaque build-time tag filter. Empty/None means
            full access. Setting this requires ``retriever`` to satisfy
            ``SupportsMetadataScope``.
        exposed_axes: Per-axis opt-in for the runtime query scope (LLM tool
            args resolved per call). Subset of
            ``("apcode", "app_name", "entity")`` adds those axes to the
            generated tool schema. Empty tuple (default) keeps the tool
            query-only. Same ``SupportsMetadataScope`` requirement as
            ``default`` — opting in on an unsupporting retriever raises.
        value_resolver: Optional ``MetadataValueResolver`` that cleans
            planner-emitted axis values before they reach the retriever. When
            set together with ``exposed_axes``, each LLM-provided value is
            routed through ``resolver.resolve(axis, raw)`` and the canonical
            result is unioned into the entry's scope via ``add_boosts``.
    """

    default: MetadataScopeLike | None = None
    exposed_axes: tuple[str, ...] = ()
    value_resolver: MetadataValueResolver | None = None


@dataclass
class RetrieverEntry:
    """A named retriever with metadata for tool generation and dispatch.

    Each entry becomes a LangGraph tool via ``create_retriever_tool()``.
    The ``description`` field is critical — it's what the LLM uses to decide
    which retriever to call.

    Attributes:
        name: Unique identifier, used as tool name suffix (e.g. "elastic_runbooks").
        description: Rich description for LLM tool selection — explain what data
            this retriever provides and when to use it.
        retriever: The actual retriever instance implementing BaseRetriever.
        search_config: Default search configuration. Merged with LLM-provided
            overrides at call time.
        weight: Weight for optional RRF merging (Phase 2).
        exposed_params: Parameter names from search_config to expose to the LLM
            as tool arguments. None = query-only (Phase 1 default).
        tool_schema: Full custom Pydantic input schema — escape hatch for complex
            cases. Takes precedence over exposed_params when set.
        examples: Optional sample queries for few-shot prompt injection (when to call this retriever).
        compressors: Per-evidence-type compressors. Default: [ChunkCompressor()].
            LightRAG entries override with [ChunkCompressor(), KGCompressor()].
        metadata_scope: Optional :class:`MetadataScopeConfig` grouping the
            build-time filter ceiling, the per-axis opt-in for the runtime
            query scope, and the optional value resolver. ``None`` (default)
            means this entry is not scope-aware — the tool factory wires no
            scope features.
        accepts_caller_scope: Opt-in for the orchestrator-supplied per-call
            metadata scope (the ``ka_metadata_scope`` bridge channel). ``False``
            (default) means caller scope is ignored entirely for this entry —
            its retrieval is bounded only by the build-time ``metadata_scope``
            default. ``True`` lets the retriever resolve its backend-specific
            subset and combine it with that default. Elasticsearch intersects;
            LightRAG replaces its opaque tag filter. Only meaningful on a
            retriever that satisfies ``SupportsMetadataScope``.
    """

    name: str
    description: str
    retriever: BaseRetriever[RetrievalChunk]
    search_config: BaseSearchConfig | None = None
    weight: float = 1.0
    exposed_params: list[str] | None = None
    tool_schema: type[BaseModel] | None = None
    examples: list[str] = field(default_factory=list)
    compressors: list[Compressor] = field(default_factory=_default_compressors)
    metadata_scope: MetadataScopeConfig | None = None
    accepts_caller_scope: bool = False


# ---------------------------------------------------------------------------
# Citation & Finding — the output contract for compressed evidence
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """A source reference attached to a finding or answer.

    Created by compress node code from RetrievalChunk data — never by the LLM.
    The LLM produces source indices; code maps them to Citation objects.
    """

    title: str
    url: str | None = None
    source_type: str = ""
    snippet: str = ""
    retriever_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundedFact:
    """A single fact with its source citation resolved.

    Preserves the KeyFactEntry.source_index -> Citation mapping that was
    previously discarded during Finding construction. Each fact carries
    its provenance — enables per-fact attribution in the synthesizer
    and reviewer prompts.

    For ChunkCompressionMethod: source_index resolves to one Citation
    (1:1 from KeyFactEntry).

    For PassthroughCompressionMethod: each fact IS a chunk, citation is
    the chunk's citation. Fine-grain, natural 1:1.

    For KGCompressionMethod: each fact is an entity/relationship description,
    citation is the entity's source citation.
    """

    fact: str
    citation: Citation | None = None


@dataclass
class Finding:
    """A single synthesized finding from retrieved evidence.

    Created by compression methods. LLM methods produce the textual content
    (topic, summary, key_facts, confidence) and source indices; code maps
    source indices to GroundedFact objects with resolved Citations.
    Deterministic methods (Passthrough, KG) build GroundedFacts directly.

    Invariant: citations must be non-empty (faithfulness guarantee).

    Attributes:
        compression_mode: How this finding was produced. "llm" for a successful
            LLM compression, "passthrough" for the deterministic path chosen on
            purpose (PASSTHROUGH strategy or DYNAMIC below threshold), "kg" for
            KGCompressionMethod, and "passthrough_fallback" when LLM compression
            was attempted but failed all retries and the chunks were rescued via
            the passthrough path (see ``PASSTHROUGH_FALLBACK_MODE``). Used by
            CompressNode to identify re-compression targets when the global
            threshold is exceeded; "passthrough_fallback" is re-compressed on a
            later KA round, never in the pass that produced it.
    """

    topic: str
    summary: str
    key_facts: list[GroundedFact]
    confidence: str  # "high" | "medium" | "low"
    citations: list[Citation]
    retriever_sources: list[str] = field(default_factory=list)
    needs_expansion: bool = False
    compression_mode: str = "llm"


# ---------------------------------------------------------------------------
# LLM structured output models (Pydantic) — used inside nodes
# ---------------------------------------------------------------------------


class KeyFactEntry(BaseModel):
    """A single atomic fact with its source chunk attribution.

    Replaces the decoupled key_facts + source_indices pattern.
    One integer per fact is simpler for the LLM than a separate
    list of indices, eliminating the concatenation bug class.
    """

    fact: str = Field(description="A single atomic fact extracted from the evidence")
    source_index: int = Field(
        description=("Chunk ID (1-based) this fact comes from. Must be a valid chunk ID from the evidence."),
        ge=1,
    )


class CompressedFinding(BaseModel):
    """LLM structured output for a single compressed finding.

    The LLM fills this; the node code maps per-fact source_index values
    to Citation objects. Attribution is per-fact via KeyFactEntry.
    """

    topic: str = Field(description="What aspect of the query this finding covers")
    summary: str = Field(description="Compressed key information (2-4 sentences)")
    key_facts: list[KeyFactEntry] = Field(
        description="Key facts, each mapped to its source chunk",
        min_length=1,
        max_length=7,
    )
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'",
    )
    needs_expansion: bool = Field(
        default=False,
        description=(
            "True if this finding's evidence is thin or incomplete — "
            "more context from the cited source would strengthen it. "
            "Set when chunks contain meta-claims, partial procedures, "
            "or references to content not present in the current evidence."
        ),
    )


class CompressedFindings(BaseModel):
    """LLM structured output — batch of findings from a chunk group."""

    findings: list[CompressedFinding] = Field(
        description="Synthesized findings from the provided evidence. Empty if evidence is irrelevant.",
        min_length=0,
    )


class FetchTarget(BaseModel):
    """A precise fetch action for ExpandNode to auto-execute.

    Produced by ReviewEvidenceNode when it identifies specific sources
    that could fill evidence gaps. ExpandNode executes these without
    LLM planning — the target is already known.
    """

    target_id: str = Field(
        description=("Identifier for the target to fetch. For documents: a pageId or doc_id. For chunk context: a chunk _id."),
    )
    target_type: Literal["document", "chunk_context"] = Field(
        description=("Type of fetch: 'document' (all chunks from a doc), 'chunk_context' (surrounding chunks)."),
    )
    retriever_name: str = Field(
        description="Name of the retriever that produced the original evidence.",
    )
    reason: str = Field(
        description="Why this target should be fetched (for tracing/debugging).",
    )


class CoverageAssessment(BaseModel):
    """LLM structured output for evidence coverage review.

    Used by review_evidence node to decide whether to iterate or stop.
    Phase 2a adds query_suggestions for the outer query loop.
    Phase 2b adds fetch_targets for the inner expansion loop.
    """

    sufficient: bool = Field(description="True if the evidence adequately covers the query")
    gaps: list[str] = Field(
        default_factory=list,
        description="Specific aspects of the query not covered by current evidence",
    )
    reasoning: str = Field(
        description="Brief explanation of the coverage assessment",
    )
    query_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up queries to fill identified gaps. Empty when sufficient=True.",
    )
    fetch_targets: list[FetchTarget] = Field(
        default_factory=list,
        description=(
            "Precise fetch actions to pull more context from already-found sources. "
            "Use when a specific document or chunk neighborhood would fill a gap. "
            "Empty when sufficient=True."
        ),
    )


# ---------------------------------------------------------------------------
# Node task types for model resolution
# ---------------------------------------------------------------------------


class KnowledgeNodeTask(StrEnum):
    """Task types for per-node model resolution.

    Each maps to a key in context.model_configs, allowing different models
    per task (e.g. fast SLM for planning, quality LLM for compression).
    """

    DEFAULT = "default"
    PLANNING = "planning"
    COMPRESSION = "compression"
    REVIEW = "review"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"


# ---------------------------------------------------------------------------
# Output bundles
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeAgentFindings:
    """Evidence-mode output — compressed findings from all retrievers.

    This is what the Knowledge Agent returns when used as a tool by Level C.
    """

    query: str
    findings: list[Finding]
    coverage: CoverageAssessment | None = None
    retriever_names: list[str] = field(default_factory=list)
    iteration_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_sufficient(self) -> bool:
        """Whether evidence coverage was assessed as sufficient."""
        return self.coverage.sufficient if self.coverage else len(self.findings) > 0

    @property
    def all_citations(self) -> list[Citation]:
        """Flat list of all citations across findings."""
        return [c for f in self.findings for c in f.citations]


@dataclass
class AnswerReview:
    """Result of faithfulness check on a synthesized answer.

    Produced by ReviewAnswerNode. Citation coherence is guaranteed by
    CitationResolver — this checks only whether claims match cited evidence.
    """

    faithful: bool
    explanation: str
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass
class KnowledgeAgentAnswer:
    """Answer-mode output — synthesized answer plus underlying evidence.

    Uses composition: wraps a KnowledgeAgentFindings as ``.evidence``
    rather than duplicating its fields.

    ``answer_citations`` contains ONLY sources the LLM actually cited in the
    answer text — not all sources from all findings. This is the cited
    subset, ordered by ``[1]``, ``[2]``... reference number as they appear
    in the answer. The full evidence bundle (all findings, all citations)
    is available via ``.evidence.all_citations``.
    """

    evidence: KnowledgeAgentFindings
    answer: str
    answer_citations: list[Citation] = field(default_factory=list)
    answer_review: AnswerReview | None = None

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/knowledge_bridge_channels.py
----
"""Shared state-channel contract between an orchestrator and the Knowledge Agent.

A subagent invoked through the deepagents ``task`` tool exchanges data with its
parent ONLY through state keys that BOTH sides declare (``task`` copies the
parent's state into the subagent input and the subagent's result back out,
minus a fixed exclusion set). For a key to cross either way the channel name
must be byte-identical in both schemas — that shared name *is* the propagation
contract. This module is the single source of truth for those two channels so
the orchestrator middleware and the KA state schemas cannot drift apart.

Two channels, deliberately different lifetimes:

``ka_metadata_scope`` (INPUT, orchestrator → KA)
    A per-invocation bundle of backend-specific FILTER-ONLY instructions:
    Elasticsearch document/application/entity filters and LightRAG's opaque
    tag filter. Each retriever applies only the subset it recognizes. A caller
    cannot smuggle a boost across this boundary (any ``*_boost`` key is dropped
    with a warning). It is run-scoped
    — it must never bleed into the next conversation turn on a checkpointed
    thread. ``UntrackedValue`` delivers exactly that: it survives across
    super-steps within a run (so a value set at run start is still readable when
    a retriever tool fires several steps later), but is never checkpointed, so
    the next run on the same thread starts fresh with no reset node required.

``ka_sources`` (OUTPUT, KA → orchestrator)
    Grounding sources surfaced by the KA as minimal JSON-safe dicts. A single
    planner turn can delegate to the KA more than once — sequentially across
    super-steps or concurrently within one super-step — so this channel needs
    an accumulating reducer (a plain ``LastValue`` would raise on concurrent
    writes and silently overwrite on sequential ones). The reducer concatenates
    and de-duplicates; callers reset it per run with ``Overwrite(value=[])``
    (a bare ``[]`` is a no-op under an accumulate reducer).

Both values are plain JSON-serializable dicts/lists so they survive checkpoint
serialization and the ``task`` ``Command(update=...)`` round-trip without
carrying vendor objects across the boundary.
"""

from __future__ import annotations

import logging
from typing import Annotated, NotRequired

from langchain.agents.middleware.types import OmitFromInput, OmitFromOutput
from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import TypedDict

from sta_agent_core.repositories.retrievers.scope_capability import CallerMetadataScope


logger = logging.getLogger(__name__)


#: State key carrying the orchestrator-supplied metadata scope into the
#: Knowledge Agent. Byte-identical in the orchestrator middleware and KA state.
KA_METADATA_SCOPE_KEY = "ka_metadata_scope"

#: State key carrying the Knowledge Agent's grounding sources back to the
#: orchestrator. Byte-identical in the orchestrator middleware and KA state.
KA_SOURCES_KEY = "ka_sources"


class KaMetadataScope(CallerMetadataScope, total=False):
    """The orchestrator-supplied request scope bundle for the KA.

    Filter keys are backend-specific and never become soft boosts. Elasticsearch
    intersects its recognized fields with the build-time ceiling; LightRAG
    replaces its opaque build-time tag filter when a caller tag filter exists.
    ``include_without_caller_scope`` explicitly names entries that may run with
    only their build-time scope.

    All fields are ``NotRequired`` — a caller supplies only the axes it wants to
    constrain; absent axes leave that part of retrieval unscoped.
    """


def merge_ka_sources(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Accumulate KA source dicts across multiple KA calls — pure concatenation.

    Appends ``right`` (newly returned) after ``left`` (existing) with **no
    cross-call de-duplication**. Each KA call already returns its citations
    ordered and 1:1 with its own inline ``[N]`` markers (see
    ``OutputNode._build_ka_sources``); concatenating the per-call blocks keeps
    the accumulated channel a **contiguous, position-stable** list. That is what
    lets the orchestrator offset a later call's numbering by the count of
    sources already surfaced (call 2's ``[1]`` → ``[N+1]``) and have every
    ``[K]`` map to the K-th row of the displayed list.

    De-duplicating across calls would break that contiguity: a document re-cited
    by a later call would collapse back to its earlier position, so the offset
    arithmetic would land on the wrong row. The cost of concatenation is that a
    document cited by two separate KA calls appears in the panel twice — accepted
    in exchange for deterministic, offsettable numbering.

    Reset the channel per run with ``langgraph.types.Overwrite(value=[])``;
    returning a bare ``[]`` here is a no-op because this reducer only accumulates.

    Args:
        left: Sources already on the channel (``None`` on first write).
        right: Sources returned by the latest KA delegation (``None`` allowed).

    Returns:
        ``left`` followed by ``right``, a fresh list (never a shared reference).
    """
    return [*(left or []), *(right or [])]


class KnowledgeBridgeChannels(TypedDict, total=False):
    """The two shared channels, defined once for both sides to inherit/declare.

    Declaring these keys on BOTH the orchestrator (via a middleware
    ``state_schema``) and the KA graph state is what lets ``task`` carry them
    across the delegation boundary — neither key is in deepagents'
    ``_EXCLUDED_STATE_KEYS``.
    """

    # INPUT — run-scoped, never checkpointed (see module docstring). The caller
    # seeds it and it is never echoed back out, so it is ``OmitFromOutput``
    # (NOT ``OmitFromInput`` — that would drop the caller-supplied value).
    ka_metadata_scope: NotRequired[Annotated[KaMetadataScope | None, UntrackedValue, OmitFromOutput]]
    # OUTPUT — accumulate + dedupe across repeated KA delegations in one run.
    # The reducer is REQUIRED: on the orchestrator side this schema is merged in
    # by a deepagents/``create_agent`` middleware, and concurrent ``task`` calls
    # write this channel in one super-step (a plain ``LastValue`` raises "can
    # receive only one value per step").
    #
    # ORDER IS LOAD-BEARING: ``OmitFromInput`` MUST precede ``merge_ka_sources``.
    # ``create_agent`` keeps the LAST channel-defining metadata in an
    # ``Annotated``; with the reducer first (``..., merge_ka_sources,
    # OmitFromInput``) the visibility marker shadows it and the channel silently
    # degrades to ``LastValue`` — reintroducing the concurrent-write crash. With
    # ``OmitFromInput`` first the reducer wins, so the channel both accumulates
    # AND stays hidden from the input schema.
    ka_sources: NotRequired[Annotated[list[dict], OmitFromInput, merge_ka_sources]]


def normalize_doc_ids(raw: object) -> list[str]:
    """Coerce a raw ``doc_ids`` value into a clean ``list[str]``.

    Accepts a single string or an iterable of strings; trims whitespace and
    drops empty / non-string entries. Returns an empty list for anything that
    cannot yield document ids (``None``, wrong type, all-empty). Centralizing
    this keeps the page-id-vs-chunk-id namespace question in one place: the
    caller still has to ensure the ids match the retriever's keyword field, but
    a malformed value can never reach the retriever as a degenerate filter.

    Args:
        raw: The ``doc_ids`` value pulled from ``ka_metadata_scope`` state.

    Returns:
        Whitespace-trimmed, de-duplicated, order-preserving list of doc ids.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates: list[object] = [raw]
    elif isinstance(raw, (list, tuple, set)):
        candidates = list(raw)
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, str):
            continue
        trimmed = item.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        out.append(trimmed)
    return out


def normalize_apcode(raw: object) -> list[str]:
    """Coerce a raw ``apcode`` value into a clean ``list[str]`` for hard filtering.

    Mirrors :func:`normalize_doc_ids` — accepts a single apcode string or an
    iterable of them, trims and drops empties. The values flow into a
    ``MetadataScope.apcode_filter`` (a hard AND-filter), so a degenerate value
    must never reach Elasticsearch.

    Args:
        raw: The ``apcode`` value pulled from ``ka_metadata_scope`` state.

    Returns:
        Whitespace-trimmed, de-duplicated, order-preserving list of apcodes.
    """
    return normalize_doc_ids(raw)


def normalize_app_name(raw: object) -> list[str]:
    """Coerce a raw ``app_name`` value into a clean ``list[str]`` for hard filtering.

    Mirrors :func:`normalize_doc_ids` — accepts a single app-name string or an
    iterable of them, trims and drops empties. The values flow into a
    ``MetadataScope.app_name_filter`` (a hard AND-filter).

    Args:
        raw: The ``app_name`` value pulled from ``ka_metadata_scope`` state.

    Returns:
        Whitespace-trimmed, de-duplicated, order-preserving list of app names.
    """
    return normalize_doc_ids(raw)


def normalize_entity(raw: object) -> list[str]:
    """Coerce a raw ``entity`` value into a clean ``list[str]`` for hard filtering.

    Mirrors :func:`normalize_doc_ids` — accepts a single entity string or an
    iterable of them, trims and drops empties. The values flow into a
    ``MetadataScope.entity_filter`` (a hard AND-filter).

    Args:
        raw: The ``entity`` value pulled from ``ka_metadata_scope`` state.

    Returns:
        Whitespace-trimmed, de-duplicated, order-preserving list of entities.
    """
    return normalize_doc_ids(raw)


def normalize_tag_filter(raw: object) -> dict[str, list[str]] | None:
    """Normalize an opaque backend tag-filter mapping."""
    if not isinstance(raw, dict):
        return None
    normalized: dict[str, list[str]] = {}
    for operator, tags in raw.items():
        if not isinstance(operator, str) or not operator:
            continue
        clean_tags = normalize_doc_ids(tags)
        if clean_tags:
            normalized[operator] = clean_tags
    return normalized or None


#: Backend filter keys plus the explicit retriever-routing escape hatch.
_KNOWN_SCOPE_KEYS: frozenset[str] = frozenset({"doc_ids", "apcode", "app_name", "entity", "tag_filter", "include_without_caller_scope"})


def read_ka_metadata_scope(raw: object) -> KaMetadataScope:
    """Validate and normalize a raw ``ka_metadata_scope`` payload.

    Ignores unknown keys AND any ``*_boost`` key — a boost cannot cross this
    trust boundary, so a caller (or a buggy upstream) attempting to smuggle one
    is defended against here, not at the retriever. Dropped keys emit a
    ``logger.warning`` so the misuse is diagnosable. Each known axis is
    normalized into clean values (trimmed, de-duplicated, empties dropped);
    values that normalize to empty are omitted. ``tag_filter`` preserves its
    backend-specific operator keys; ``include_without_caller_scope`` contains
    retriever entry names, not filter axes.

    Args:
        raw: The value pulled off ``ka_metadata_scope`` state — expected to be a
            dict, but anything non-dict (``None``, wrong type) yields an empty
            scope so retrieval proceeds unfiltered.

    Returns:
        A :class:`KaMetadataScope` carrying only the non-empty known axes.
    """
    if not isinstance(raw, dict):
        return {}

    dropped: list[str] = []
    scope: KaMetadataScope = {}
    for key, value in raw.items():
        if not isinstance(key, str) or key not in _KNOWN_SCOPE_KEYS:
            # Unknown key or a smuggled ``*_boost`` — never a valid filter axis.
            dropped.append(str(key))
            continue
        if key == "doc_ids":
            normalized = normalize_doc_ids(value)
        elif key == "apcode":
            normalized = normalize_apcode(value)
        elif key == "app_name":
            normalized = normalize_app_name(value)
        elif key == "entity":
            normalized = normalize_entity(value)
        elif key == "tag_filter":
            normalized = normalize_tag_filter(value)
        else:  # key == "include_without_caller_scope"
            normalized = normalize_doc_ids(value)
        if normalized:
            scope[key] = normalized  # type: ignore[literal-required]

    if dropped:
        logger.warning(
            "ka_metadata_scope dropped %d unrecognized key(s): %s — only %s are recognized scope keys; "
            "boosts cannot cross the orchestrator→KA trust boundary.",
            len(dropped),
            sorted(dropped),
            sorted(_KNOWN_SCOPE_KEYS),
        )
    return scope

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/nodes/plan_queries.py
----
"""PlanQueriesNode — query planning for the Knowledge Agent.

Produces an ``AIMessage.tool_calls`` list that the downstream ToolNode executes
unchanged. Two strategies, selected by ``PlanConfig.planning_strategy``:

- ``"tool_calls"`` (default): bind the retriever tools to the model and let it
  emit native ``tool_calls`` directly. The bound schema constrains tool names
  and args, so there is no semantic validation round-trip; transient failures
  are retried via ``with_retry``. Needs a model that can emit parallel tool
  calls to fan out N retriever calls per turn.
- ``"structured"``: ask the model for a validated structured plan (with
  conversational validate-and-retry) and convert it into ``tool_calls``.
  Guarantees N calls regardless of the model's parallel-tool-call support.

Features (both strategies):
- Query resolution: prefers latest HumanMessage in ``messages``, falls back to
  ``state.query`` for orchestrator invocations without messages.
- Tool injection: injects available tool names + descriptions into the system
  prompt so smaller LLMs know exactly which tools exist.
- Deterministic query cap: hard-truncates tool_calls to ``max_queries`` after
  the LLM response, regardless of what the LLM generated.

On iteration 1: the LLM sees the user query + tool schemas.
On iteration 2+: the LLM sees compressed findings + coverage gaps + tool schemas.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any, ClassVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sta_agent_core.repositories.retrievers import SupportsMetadataScope

from ...base.nodes import NodeBase
from ...base.utils.output_validation import ModelRetry, OutputValidationError, ainvoke_with_output_validation
from ..knowledge_agent_config import KnowledgeAgentConfig, PlanConfig
from ..knowledge_agent_prompts import (
    PLAN_CALL_FORMAT_STRUCTURED,
    PLAN_CALL_FORMAT_TOOL_CALLS,
    PLAN_QUERIES_REFINEMENT_PROMPT,
    PLAN_QUERIES_SYSTEM_PROMPT,
)
from ..knowledge_agent_state import KnowledgeAgentContext, KnowledgeAgentState
from ..knowledge_agent_types import Finding, KnowledgeNodeTask, RetrieverEntry
from ..knowledge_bridge_channels import KA_METADATA_SCOPE_KEY, read_ka_metadata_scope
from ..utils.findings_format import finding_source_context_line, format_finding_block


logger = logging.getLogger(__name__)

_SUPPORTED_PLANNER_ARGS = frozenset({"query", "apcode", "app_name", "entity"})
_METADATA_PLANNER_ARGS = ("apcode", "app_name", "entity")
_PLAN_VALIDATION_RETRIES = 1

# Surfaced as the planner's message content when planning yields no usable
# retriever calls and no direct response — i.e. the model (after the strategy's
# own retries) produced neither tool calls nor any text. Without this the node
# would emit an empty AIMessage that routes straight to ``output``, leaving the
# consumer with a blank last message. Wording mirrors the input-quality guidance
# in ``KnowledgeAgentInputState``: name the specific entities to retrieve well.
PLAN_FAILED_MESSAGE = (
    "I couldn't determine how to search for this request. Please rephrase it or add "
    "more detail — naming the specific application, component, or identifier involved "
    "helps me retrieve relevant information."
)


class _PlannedRetrieverCall(BaseModel):
    """One planned retriever call emitted by the planner model."""

    tool_name: str = Field(description="Exact retriever tool name to call, e.g. search_elastic_runbooks")
    query: str = Field(description="Focused, self-contained search query")
    apcode: str | None = Field(default=None, description="Optional APCODE argument when the selected tool exposes it")
    app_name: str | None = Field(default=None, description="Optional application-name argument when the selected tool exposes it")
    entity: str | None = Field(default=None, description="Optional entity argument when the selected tool exposes it")


class _PlannedRetrieverCalls(BaseModel):
    """Structured planner output."""

    model_config = ConfigDict(extra="forbid")

    calls: list[_PlannedRetrieverCall] = Field(
        default_factory=list,
        description="Retriever calls to execute. Leave empty only when no available tool can help.",
    )
    direct_response: str | None = Field(
        default=None,
        description="Optional direct reply when the user is greeting, thanking, or asking for clarification rather than requesting retrieval.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_tool_calls(cls, value: Any) -> Any:
        """Accept legacy/native-tool-shaped ``tool_calls`` as planner calls.

        Some models continue to emit a top-level ``tool_calls`` object even
        when asked for structured output. Pydantic would otherwise ignore or
        reject that field, producing ``calls=[]`` and making routing exit to
        ``output``. Normalize that common shape into the explicit ``calls``
        schema before validation so downstream ToolNode routing still sees
        real ``AIMessage.tool_calls``.
        """
        if not isinstance(value, dict) or "tool_calls" not in value:
            return value

        data = dict(value)
        raw_tool_calls = data.pop("tool_calls")
        if "calls" not in data:
            data["calls"] = cls._planner_calls_from_tool_calls(raw_tool_calls)
        return data

    @staticmethod
    def _planner_calls_from_tool_calls(raw_tool_calls: Any) -> Any:
        if not isinstance(raw_tool_calls, list):
            return raw_tool_calls

        calls: list[Any] = []
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                calls.append(raw_call)
                continue
            raw_args = raw_call.get("args")
            args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            call: dict[str, Any] = {
                "tool_name": raw_call.get("tool_name") or raw_call.get("name"),
                "query": args.get("query") or raw_call.get("query") or "",
            }
            for axis in _METADATA_PLANNER_ARGS:
                value = args.get(axis) if axis in args else raw_call.get(axis)
                if value is not None:
                    call[axis] = value
            calls.append(call)
        return calls


class PlanQueriesNode(NodeBase[KnowledgeAgentContext]):
    """Produce ToolNode-compatible retriever calls from a structured plan.

    The node asks the model for structured ``tool_name``/``query`` records,
    validates them against the actual generated retriever tool names and
    exposed arguments, then returns an ``AIMessage`` containing ``tool_calls``
    that the ToolNode executes.

    Key behaviors:
    - Resolves query from latest HumanMessage in state.messages first, then
      falls back to state.query (for orchestrator use without messages).
    - Injects available tool names + descriptions into system prompt (XML tags).
    - Hard-truncates tool_calls to ``plan_config.max_queries`` after LLM response.

    Example:
        ```python
        plan_node = PlanQueriesNode(
            tools=retriever_tools,
            default_model=llm,
            agent_config=config,
        )
        graph.add_node("plan_queries", plan_node)
        ```
    """

    task: ClassVar[str] = KnowledgeNodeTask.PLANNING

    def __init__(
        self,
        tools: list[BaseTool],
        entries: list[RetrieverEntry] | None = None,
        default_model: BaseChatModel | None = None,
        agent_config: KnowledgeAgentConfig | None = None,
    ) -> None:
        super().__init__(default_model=default_model, node_config=agent_config)
        self._tools = tools
        self._entries = entries  # When set, used for per-retriever examples in prompt
        self._agent_config = agent_config or KnowledgeAgentConfig()
        self._tool_args_by_name = self._build_tool_args_by_name(tools)

    @property
    def plan_config(self) -> PlanConfig:
        return self._agent_config.plan

    async def __call__(
        self,
        state: KnowledgeAgentState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Invoke the planner LLM to produce retriever tool_calls.

        Dispatches on ``PlanConfig.planning_strategy`` (native ``"tool_calls"``
        binding or validated ``"structured"`` output). Resolves query from the
        latest HumanMessage in messages (falls back to state.query for
        orchestrator invocations without messages). Builds an XML-structured
        system prompt with available tools and hard-truncates tool_calls to
        max_queries.

        Args:
            state: Current workflow state.
            config: LangGraph runnable config.

        Returns:
            Dict with messages (AIMessage containing tool_calls),
            resolved query, and iteration_count increment.
        """
        # Resolve query — prefer latest HumanMessage so that new invocations
        # on the same thread (checkpointer) pick up the fresh query instead of
        # the stale `query` field written by a previous run.
        query = self._extract_query_from_messages(state.get("messages", []))
        if not query:
            query = state.get("query", "")
        if not query:
            logger.error("PlanQueriesNode: no query found in state or messages")
            raise ValueError("No query provided and no HumanMessage found in messages")

        iteration = state.get("iteration_count", 0)
        findings = state.get("findings", [])
        coverage = state.get("coverage")

        # Restrict the planner's tool set to scope-accepting retrievers when the
        # caller seeded a request scope (see _active_plan_inputs).
        active_tools, active_entries, active_args = self._active_plan_inputs(state)

        # Observability: a caller scope that reaches no scope-accepting corpus is
        # a silent no-op — the planner searches unfiltered. _active_plan_inputs
        # returns the node's own tool list (by identity) in every unrestricted
        # branch, so an identity check flags the unenforced case without
        # re-deriving the selection logic.
        scope_unenforced = bool(read_ka_metadata_scope(state.get(KA_METADATA_SCOPE_KEY))) and active_tools is self._tools

        # Build messages with XML-structured system prompt
        messages = self._build_messages(query, iteration, findings, coverage, entries=active_entries, tools=active_tools)

        if self.plan_config.planning_strategy == "tool_calls":
            response = await self._plan_via_tool_calls(messages, config, tools=active_tools, tool_args_by_name=active_args)
        else:
            response = await self._plan_via_structured_output(messages, config, tool_args_by_name=active_args)

        max_queries = self.plan_config.max_queries
        if iteration == 0 and self.plan_config.include_original_query and response.tool_calls:
            self._inject_anchor_queries(response, query)

        # Deterministic query cap — hard-truncate tool_calls (after injection)
        n_calls = len(response.tool_calls) if response.tool_calls else 0
        if response.tool_calls and n_calls > max_queries:
            logger.warning(
                "plan_queries: truncating %d tool calls to max_queries=%d",
                n_calls,
                max_queries,
            )
            response.tool_calls = response.tool_calls[:max_queries]
            n_calls = max_queries

        # Planning-failure fallback: no usable calls AND no text. The strategy's
        # own retries (with_retry / validation round-trip) are already spent by
        # here, so a blank outcome is terminal — substitute a non-empty message
        # and flag it so OutputNode surfaces it (and does not mistake it for a
        # genuine direct response). A no-call turn WITH content (greeting /
        # clarification) is left untouched.
        content_text = response.content if isinstance(response.content, str) else str(response.content or "")
        plan_failed = not response.tool_calls and not content_text.strip()
        if plan_failed:
            logger.warning("plan_queries: no usable tool calls and empty content — emitting plan-failure fallback")
            response.content = PLAN_FAILED_MESSAGE

        logger.info(
            "plan_queries (iteration %d): %d tool calls for query '%s'",
            iteration + 1,
            n_calls,
            query[:80],
        )

        update: dict[str, Any] = {
            "messages": [response],
            "query": query,
            "plan_failed": plan_failed,
            "iteration_count": 1,
            "compressed_chunk_hashes": set(),  # Reset per outer iteration (Decision 36)
            # processed_kg_hashes intentionally NOT reset: KG reformatting is
            # deterministic so re-processing produces identical findings.
            # retrieved_responses accumulates (append reducer), so without
            # persistent hashes, old relationships would create duplicate Findings.
            "expansion_rounds": 0,  # Phase 2b: reset inner loop counter
            "coverage": None,  # Phase 2b: clear stale fetch_targets (2b-D2)
        }
        if scope_unenforced and iteration == 0:
            # Once per turn (iteration 0): surface via the resolution-warnings
            # channel OutputNode mirrors into result.metadata["warnings"], so a
            # dropped caller filter is detectable downstream, not only in a log.
            update["resolution_warnings"] = [
                "caller_scope_unenforced: a caller-supplied ka_metadata_scope was present but no "
                "bound retriever accepts caller scope, so the filter did not apply — results may "
                "include out-of-selection content."
            ]
        return update

    # ------------------------------------------------------------------
    # Caller-scope tool restriction
    # ------------------------------------------------------------------

    def _active_plan_inputs(
        self,
        state: KnowledgeAgentState,
    ) -> tuple[list[BaseTool], list[RetrieverEntry] | None, dict[str, set[str]]]:
        """Restrict the planner's tool set to scope-accepting retrievers when a caller scope is present.

        A caller-supplied ``ka_metadata_scope`` is a bundle of backend-specific
        hard filters. Each scope-aware retriever resolves the subset it
        understands; the planner binds only entries with an effective resolved
        scope. ``include_without_caller_scope`` explicitly adds named entries
        whose build-time scope should apply without a caller filter.

        Returns the full ``(tools, entries, tool_args_by_name)`` unchanged when:
        no scope is set (the common path); entries are unavailable (the opt-in flag
        lives on the entry, so the node cannot tell which tools are scope-accepting);
        This method never mutates
        ``self._tools`` / ``self._entries``: the node is a shared singleton across
        parallel KA runs, so a per-call filtered copy is returned for the caller to
        thread down.
        """
        scope = read_ka_metadata_scope(state.get(KA_METADATA_SCOPE_KEY))
        if not scope:
            return self._tools, self._entries, self._tool_args_by_name
        if not self._entries:
            logger.warning(
                "plan_queries: caller scope present but no retriever entries available to check "
                "accepts_caller_scope — keeping all tools; the caller filter may not apply."
            )
            return self._tools, self._entries, self._tool_args_by_name

        explicitly_included = frozenset(scope.get("include_without_caller_scope", []))

        def accepts_scope(entry: RetrieverEntry) -> bool:
            if not entry.accepts_caller_scope:
                return False
            if entry.name in explicitly_included:
                return True
            retriever = entry.retriever
            if not isinstance(retriever, SupportsMetadataScope) or getattr(retriever, "supports_metadata_scope", None) is not True:
                return False
            resolved_scope = retriever.resolve_caller_scope(scope)
            return resolved_scope is not None and resolved_scope.is_effective()

        accepted_names = {f"search_{entry.name}" for entry in self._entries if accepts_scope(entry)}
        active_tools = [tool for tool in self._tools if tool.name in accepted_names]
        if not active_tools:
            logger.warning(
                "plan_queries: caller scope present but no retriever resolves a matching filter "
                "and none is explicitly included without caller scope — binding no retriever tools."
            )
            return [], [], {}

        active_entries = [entry for entry in self._entries if f"search_{entry.name}" in accepted_names]
        active_args = {name: args for name, args in self._tool_args_by_name.items() if name in accepted_names}
        logger.info(
            "plan_queries: caller scope present — restricting planner to %d scope-accepting retriever(s): %s",
            len(active_tools),
            sorted(accepted_names),
        )
        return active_tools, active_entries, active_args

    # ------------------------------------------------------------------
    # Planning strategies (tool_calls | structured)
    # ------------------------------------------------------------------

    async def _plan_via_tool_calls(
        self,
        messages: list[SystemMessage | HumanMessage],
        config: RunnableConfig,
        tools: list[BaseTool] | None = None,
        tool_args_by_name: dict[str, set[str]] | None = None,
    ) -> AIMessage:
        """Plan by binding retriever tools and letting the model emit native tool_calls.

        The bound schema constrains tool names and argument shape, so no
        semantic validation round-trip is needed — invalid-by-construction
        calls cannot occur. Transient model/provider failures are retried via
        ``with_retry``. Calls with an unknown name or empty ``query`` are dropped
        defensively, and args are filtered to each tool's exposed set.

        ``tools`` / ``tool_args_by_name`` default to the node's full set; a
        caller passes the scope-restricted subset (see ``_active_plan_inputs``)
        so the bound schema AND the defensive sanitizer both reject a corpus
        where the caller filter would not apply.

        Needs a model that can emit parallel tool calls to fan out N retriever
        calls per turn; models that cannot (e.g. gpt-oss) degrade to fewer calls
        — use ``planning_strategy="structured"`` there.
        """
        tools = tools if tools is not None else self._tools
        attempts = max(1, self.plan_config.tool_call_retry_attempts)
        model_with_tools = self.model.bind_tools(tools).with_retry(stop_after_attempt=attempts)
        raw = await model_with_tools.ainvoke(messages, config=config)
        response = raw if isinstance(raw, AIMessage) else AIMessage(content=str(getattr(raw, "content", "")))

        sanitized = self._sanitize_tool_calls(response.tool_calls or [], tool_args_by_name)
        response.tool_calls = sanitized  # type: ignore[assignment]
        # When calls exist the content is irrelevant to routing; on a no-call
        # turn the content is the direct response / clarification, kept as-is.
        if sanitized:
            response.content = ""
        return response

    def _sanitize_tool_calls(self, tool_calls: Sequence[Any], tool_args_by_name: dict[str, set[str]] | None = None) -> list[dict[str, Any]]:
        """Drop calls with unknown tool name or empty query; filter args to the exposed set.

        ``tool_args_by_name`` defaults to the full set; pass the scope-restricted
        subset to also drop a hallucinated call to a non-scope-accepting corpus.
        """
        tool_args_by_name = tool_args_by_name if tool_args_by_name is not None else self._tool_args_by_name
        sanitized: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name", "")
            exposed = tool_args_by_name.get(name)
            if exposed is None:
                logger.warning("plan_queries: dropping tool_call for unknown tool %r", name)
                continue
            raw_args = call.get("args")
            args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            query = (args.get("query") or "").strip()
            if not query:
                logger.warning("plan_queries: dropping tool_call for %r with empty query", name)
                continue
            clean_args: dict[str, Any] = {"query": query}
            for axis in _METADATA_PLANNER_ARGS:
                value = args.get(axis)
                if value is not None and str(value).strip() and axis in exposed:
                    clean_args[axis] = str(value).strip()
            sanitized.append(
                {
                    "id": call.get("id") or f"plan_{uuid.uuid4().hex[:8]}",
                    "name": name,
                    "args": clean_args,
                    "type": "tool_call",
                }
            )
        return sanitized

    async def _plan_via_structured_output(
        self,
        messages: list[SystemMessage | HumanMessage],
        config: RunnableConfig,
        tool_args_by_name: dict[str, set[str]] | None = None,
    ) -> AIMessage:
        """Plan via validated structured output, converted to tool_calls.

        Asks the model for a structured ``_PlannedRetrieverCalls`` plan with a
        conversational validate-and-retry, then converts the validated plan into
        an ``AIMessage.tool_calls`` list. Guarantees N calls regardless of the
        model's parallel-tool-call support, at the cost of a validation
        round-trip.

        ``tool_args_by_name`` defaults to the full set; pass the scope-restricted
        subset (see ``_active_plan_inputs``) so the validator rejects — and the
        retry feedback lists only — the scope-accepting corpora.
        """
        tool_args_by_name = tool_args_by_name if tool_args_by_name is not None else self._tool_args_by_name
        validation_ctx: dict[str, Any] = {
            "tool_args_by_name": tool_args_by_name,
        }
        try:
            plan = cast(
                _PlannedRetrieverCalls,
                await ainvoke_with_output_validation(
                    model=self.model,
                    output_type=_PlannedRetrieverCalls,
                    messages=messages,
                    output_validators=[self._validate_planned_calls],
                    validation_context=validation_ctx,
                    max_retries=_PLAN_VALIDATION_RETRIES,
                    config=config,
                ),
            )
        except OutputValidationError:
            logger.warning("PlanQueriesNode: retries exhausted, filtering invalid planned retriever calls")
            plan = self._filter_valid_planned_calls(validation_ctx.get("_last_plan"), tool_args_by_name)

        return self._plan_to_ai_message(plan, tool_args_by_name)

    def _inject_anchor_queries(self, response: AIMessage, query: str) -> None:
        """Append one tool call per selected retriever with the original user query.

        Mutates response.tool_calls in place. Called only on iteration 0 when
        include_original_query is True so the user's exact phrasing reaches retrievers.

        Skips injection for a tool if the LLM already generated a call with
        a query matching the user query (case-insensitive, stripped), avoiding
        duplicate retrieval.
        """
        normalized_query = query.strip().lower()

        existing_queries_by_tool: dict[str, set[str]] = {}
        for tc in response.tool_calls:
            tool_name = tc.get("name") or getattr(tc, "name", "")
            if not tool_name:
                continue
            args = tc.get("args") or getattr(tc, "args", {}) or {}
            q = (args.get("query") or "").strip().lower()
            existing_queries_by_tool.setdefault(tool_name, set()).add(q)

        anchor_calls = []
        for tool_name, queries in existing_queries_by_tool.items():
            if normalized_query in queries:
                logger.debug(
                    "plan_queries: skipping anchor for '%s' — LLM already generated identical query",
                    tool_name,
                )
                continue
            anchor_calls.append(
                {
                    "id": f"anchor_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "args": {"query": query},
                    "type": "tool_call",
                }
            )

        if anchor_calls:
            response.tool_calls = list(response.tool_calls) + anchor_calls  # type: ignore[assignment]
            logger.info(
                "plan_queries: injected %d anchor query tool calls (original user query)",
                len(anchor_calls),
            )

    # ------------------------------------------------------------------
    # Structured plan validation / conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tool_args_by_name(tools: list[BaseTool]) -> dict[str, set[str]]:
        """Return the LLM-visible argument names for each generated tool."""
        tool_args: dict[str, set[str]] = {}
        for tool in tools:
            args = getattr(tool, "args", None) or {}
            tool_args[tool.name] = set(args) & _SUPPORTED_PLANNER_ARGS
        return tool_args

    @staticmethod
    def _validate_planned_calls(plan: _PlannedRetrieverCalls, ctx: dict[str, Any]) -> _PlannedRetrieverCalls:
        """Validate planned calls against generated tool names and args.

        The validator is intentionally strict during retry: any invalid call
        triggers model feedback. If retries exhaust, the node hard-filters to
        valid calls so one bad call does not poison the whole plan.
        """
        if not plan.calls:
            return plan

        ctx["_last_plan"] = plan
        tool_args_by_name: dict[str, set[str]] = ctx.get("tool_args_by_name", {})
        errors: list[str] = []
        for index, call in enumerate(plan.calls, start=1):
            errors.extend(PlanQueriesNode._validation_errors_for_call(index, call, tool_args_by_name))

        if errors:
            available = ", ".join(sorted(tool_args_by_name)) or "none"
            raise ModelRetry(
                "Invalid retriever call plan:\n- " + "\n- ".join(errors) + f"\n\nUse only these tool names: {available}. "
                "Use only args exposed by that tool; the supported arg surface is query plus optional apcode, app_name, entity."
            )
        return plan

    @staticmethod
    def _validation_errors_for_call(
        index: int,
        call: _PlannedRetrieverCall,
        tool_args_by_name: dict[str, set[str]],
    ) -> list[str]:
        errors: list[str] = []
        tool_name = call.tool_name.strip()
        if tool_name not in tool_args_by_name:
            return [f"call {index}: unknown tool_name {call.tool_name!r}"]

        exposed_args = tool_args_by_name[tool_name]
        if "query" not in exposed_args:
            errors.append(f"call {index}: tool {tool_name!r} does not expose required arg 'query'")
        if not call.query.strip():
            errors.append(f"call {index}: query must be a non-empty string")

        for axis in _METADATA_PLANNER_ARGS:
            value = getattr(call, axis)
            if value is not None and value.strip() and axis not in exposed_args:
                errors.append(f"call {index}: tool {tool_name!r} does not expose arg {axis!r}")
        return errors

    def _filter_valid_planned_calls(self, plan: Any, tool_args_by_name: dict[str, set[str]] | None = None) -> _PlannedRetrieverCalls:
        """Drop invalid calls after retry exhaustion.

        ``tool_args_by_name`` defaults to the full set; pass the scope-restricted
        subset to also drop a call to a non-scope-accepting corpus.
        """
        tool_args_by_name = tool_args_by_name if tool_args_by_name is not None else self._tool_args_by_name
        if not isinstance(plan, _PlannedRetrieverCalls):
            return _PlannedRetrieverCalls(calls=[])
        valid = [
            call
            for call in plan.calls
            if not self._validation_errors_for_call(
                0,
                call,
                tool_args_by_name,
            )
        ]
        return _PlannedRetrieverCalls(calls=valid)

    @staticmethod
    def _call_args(call: _PlannedRetrieverCall, exposed_args: set[str]) -> dict[str, str]:
        args: dict[str, str] = {"query": call.query.strip()}
        for axis in _METADATA_PLANNER_ARGS:
            value = getattr(call, axis)
            if value is not None and value.strip() and axis in exposed_args:
                args[axis] = value.strip()
        return args

    def _plan_to_ai_message(self, plan: _PlannedRetrieverCalls, tool_args_by_name: dict[str, set[str]] | None = None) -> AIMessage:
        """Convert a validated structured plan into ToolNode input.

        ``tool_args_by_name`` defaults to the full set; pass the scope-restricted
        subset so arg filtering matches the corpora the planner was limited to.
        """
        tool_args_by_name = tool_args_by_name if tool_args_by_name is not None else self._tool_args_by_name
        tool_calls = []
        for call in plan.calls:
            tool_name = call.tool_name.strip()
            # Scope-enforcement boundary: every valid bound tool is a key in
            # tool_args_by_name (which is the scope-restricted subset when a caller
            # scope is present). A name absent from it is a hallucinated or
            # out-of-scope corpus — drop it rather than emit a call with empty
            # args that a non-accepting retriever would run unfiltered.
            if tool_name not in tool_args_by_name:
                continue
            exposed_args = tool_args_by_name[tool_name]
            tool_calls.append(
                {
                    "id": f"plan_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "args": self._call_args(call, exposed_args),
                    "type": "tool_call",
                }
            )
        content = "" if tool_calls else (plan.direct_response or "")
        return AIMessage(content=content, tool_calls=tool_calls)

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        query: str,
        iteration: int,
        findings: list[Finding],
        coverage: Any | None,
        entries: list[RetrieverEntry] | None = None,
        tools: list[BaseTool] | None = None,
    ) -> list[SystemMessage | HumanMessage]:
        """Build the message list for the LLM.

        System prompt includes XML-tagged available tools and max_queries constraint.
        Iteration 1: system prompt + user query.
        Iteration 2+: system prompt + refinement context + user query.

        ``entries`` / ``tools`` default to the node's full set; a caller passes the
        scope-restricted subset (see ``_active_plan_inputs``) so the prompt's tools
        block advertises only the corpora the planner is allowed to call.
        """
        # Build system prompt with injected tools and constraints
        tools_block = self._build_tools_block(entries=entries, tools=tools)
        max_queries = self.plan_config.max_queries
        call_format_constraint = PLAN_CALL_FORMAT_TOOL_CALLS if self.plan_config.planning_strategy == "tool_calls" else PLAN_CALL_FORMAT_STRUCTURED
        system_content = PLAN_QUERIES_SYSTEM_PROMPT.format(
            tools_block=tools_block,
            max_queries=max_queries,
            call_format_constraint=call_format_constraint,
        )

        messages: list[SystemMessage | HumanMessage] = [SystemMessage(content=system_content)]

        if iteration > 0 and (findings or coverage is not None):
            findings_summary = self._format_findings(findings)
            gaps = self._format_gaps(coverage)
            query_suggestions = self._format_query_suggestions(coverage)
            refinement_prompt = PLAN_QUERIES_REFINEMENT_PROMPT.format(
                findings_summary=findings_summary,
                gaps=gaps,
                query_suggestions=query_suggestions,
            )
            messages.append(SystemMessage(content=refinement_prompt))

        messages.append(HumanMessage(content=query))
        return messages

    def _build_tools_block(self, entries: list[RetrieverEntry] | None = None, tools: list[BaseTool] | None = None) -> str:
        """Format available tools as XML for the system prompt.

        When entries are provided, each tool includes name, description, and
        optional <examples> (sample queries). Otherwise name and description only.

        ``entries`` / ``tools`` default to the node's full set; a caller passes the
        scope-restricted subset (see ``_active_plan_inputs``) to advertise only the
        scope-accepting corpora.
        """
        entries = entries if entries is not None else self._entries
        tools = tools if tools is not None else self._tools
        if entries:
            parts = []
            for entry in entries:
                tool_name = f"search_{entry.name}"
                block = f'<tool name="{tool_name}">{entry.description}'
                args = self._format_tool_args(tool_name)
                if args:
                    block += f"\n  <args>{args}</args>"
                if entry.examples:
                    examples_lines = "\n".join(f'- "{q}"' for q in entry.examples[:5])
                    block += f"\n  <examples>\n  {examples_lines}\n  </examples>"
                block += "</tool>"
                parts.append(block)
            return "\n".join(parts)
        parts = []
        for tool in tools:
            args = self._format_tool_args(tool.name)
            args_block = f"\n  <args>{args}</args>" if args else ""
            parts.append(f'<tool name="{tool.name}">{tool.description}{args_block}</tool>')
        return "\n".join(parts)

    def _format_tool_args(self, tool_name: str) -> str:
        args = self._tool_args_by_name.get(tool_name, set())
        if not args:
            return ""
        ordered = ["query", *[axis for axis in _METADATA_PLANNER_ARGS if axis in args]]
        return ", ".join(arg for arg in ordered if arg in args)

    # ------------------------------------------------------------------
    # Query resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_query_from_messages(messages: Sequence[AnyMessage]) -> str:
        """Extract query from the last HumanMessage in the message list.

        Walks messages in reverse to find the most recent HumanMessage.
        Returns empty string if no HumanMessage is found.
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                content = msg.content
                return content if isinstance(content, str) else str(content)
        return ""

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_findings(findings: list[Finding]) -> str:
        """Format findings into a readable summary for the refinement prompt."""
        if not findings:
            return "No findings from previous iteration."

        parts = [
            format_finding_block(
                i,
                f.topic,
                f.summary,
                f.key_facts,
                # Surface page identity + context so the planner can re-query
                # when a previous iteration's evidence came from a generic page
                # about a different entity than the question asks about.
                source_context_line=finding_source_context_line(f.citations, max_summary_chars=160) if f.citations else None,
            )
            for i, f in enumerate(findings, 1)
        ]
        return "\n\n".join(parts)

    @staticmethod
    def _format_gaps(coverage: Any | None) -> str:
        """Format coverage gaps for the refinement prompt."""
        if coverage is None:
            return "No coverage assessment available."
        gaps = getattr(coverage, "gaps", [])
        if not gaps:
            return "No specific gaps identified."
        return "\n".join(f"- {gap}" for gap in gaps)

    @staticmethod
    def _format_query_suggestions(coverage: Any | None) -> str:
        """Format query suggestions from the review assessment."""
        if coverage is None:
            return "No query suggestions available."
        suggestions = getattr(coverage, "query_suggestions", [])
        if not suggestions:
            return "No specific query suggestions."
        return "\n".join(f"- {s}" for s in suggestions)

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/tools/retriever_tool_factory.py
----
"""Retriever tool factory — RetrieverEntry → LangGraph tool.

Creates async tools that wrap retriever.search() calls. Each tool:
- Takes ``query: str`` as its primary input (Phase 1)
- Calls the retriever's search() method with resolved kwargs
- Returns a Command that writes the full SearchResponse to state
  (``retrieved_responses`` key) while keeping ToolMessage content small
  (summary only — the LLM never sees raw chunks)

The ToolNode executes these tools in parallel when the LLM produces
multiple tool_calls in a single response.

**Token budget splitting**: When a search config emits token budget fields
(``max_total_tokens``, ``max_entity_tokens``, ``max_relation_tokens``),
each tool call automatically splits the budget by the number of sibling
calls to the same tool in the current batch. This prevents context overflow
when the LLM fires N parallel queries against the same retriever.

Phase 2 will add ``exposed_params`` support for selective parameter
exposure via dynamic Pydantic schemas (``create_model()``).
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
from typing import Any, cast

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command

from sta_agent_core.repositories.retrievers import MetadataValueResolver, SupportsMetadataScope
from sta_agent_core.repositories.retrievers.elasticsearch.metadata_scope import MetadataScope
from sta_agent_core.repositories.retrievers.scope_capability import CallerMetadataScope, MetadataScopeLike

from ..knowledge_agent_types import MetadataScopeConfig, RetrieverEntry
from ..knowledge_bridge_channels import (
    KA_METADATA_SCOPE_KEY,
    read_ka_metadata_scope,
)


logger = logging.getLogger(__name__)

_TOKEN_BUDGET_FIELDS = frozenset({"max_total_tokens", "max_entity_tokens", "max_relation_tokens"})
_MIN_TOKEN_BUDGET = 1000
# Derived from ``MetadataScope`` (single source of truth for axis names — see
# ``ScopeAxis`` in ``metadata_scope.py``). Adding a fourth axis there is a
# one-line change that flows here, into the resolver, and into ``add_boosts``
# without any synchronized edits.
#
# - ``_RUNTIME_SCOPE_AXES``: bare axis names the planner emits per call
#   (``"apcode"``, ``"app_name"``, ``"entity"``). Sister to the build-time
#   ``entry.metadata_scope.default`` filter ceiling. See knowledge_agent/AGENTS.md
#   § Metadata scope.
# - ``_BOOST_AXES``: MetadataScope field names ``add_boosts`` accepts — a
#   third-party ``MetadataValueResolver`` that returns ``target_axis_boost=
#   "apcode_filter"`` (or any non-boost target) would otherwise raise mid-tool
#   before the search try/except can catch it.
_RUNTIME_SCOPE_AXES: frozenset[str] = frozenset(MetadataScope.AXIS_NAMES)
_BOOST_AXES: frozenset[str] = MetadataScope.BOOST_FIELDS

_AXIS_DESCRIPTIONS: dict[str, str] = {
    "apcode": (
        "Optional APCODE — a 7-character application identifier, formatted as "
        "'AP' + 5 digits (e.g. AP90021) or 'A' + 6 digits (e.g. A123456). "
        "If the user's query mentions an apcode written in any case "
        "(e.g. 'AP90021', 'ap90021', 'Ap90021'), pass it here in UPPERCASE. "
        "Soft-ranks docs from that application higher (boost-only — won't "
        "filter out other docs)."
    ),
    "app_name": (
        "Optional application name to soft-rank docs from that app higher "
        "(boost-only — won't filter out other docs). If the user's query "
        "names an application, pass it here; case does not matter."
    ),
    "entity": "Optional entity name OR id to soft-rank docs scoped to that entity higher (boost-only — won't filter out other docs).",
}


def create_retriever_tool(entry: RetrieverEntry) -> BaseTool:
    """Create a LangGraph tool from a RetrieverEntry.

    The tool calls ``entry.retriever.search(query, **search_kwargs)`` and
    returns a ``Command`` that:
    - Writes the full ``SearchResponse`` to ``state.retrieved_responses[name]``
    - Appends a summary ``ToolMessage`` to the message history

    The LLM sees only the summary (e.g. "Retrieved 12 results from 'elastic_runbooks'").
    Downstream nodes (collect, compress) read from ``retrieved_responses``.

    Args:
        entry: RetrieverEntry with retriever instance, name, description, and config.

    Returns:
        A LangGraph-compatible async tool.
    """
    # Build-time trust-boundary gate: refuse to wire any metadata scope
    # feature onto a retriever that doesn't honor `metadata_scope=...` in
    # `search()`. The BaseRetriever protocol uses **kwargs for universality,
    # so unsupporting backends silently drop the kwarg — and the
    # `default_scope` ceiling becomes a no-op. Fail loud at agent-build time
    # rather than at search time. See `SupportsMetadataScope` in
    # `sta_agent_core.repositories.retrievers.scope_capability`.
    _require_metadata_scope_support(entry)

    # Snapshot mutable entry fields into closure-local immutable bindings.
    # `RetrieverEntry` is a (mutable) dataclass: without this snapshot, a
    # caller could pass a legacy entry (no scope features → gate passes),
    # then later set `entry.metadata_scope = MetadataScopeConfig(...)` and
    # the next tool call would forward `metadata_scope` to a backend that
    # drops it — silently bypassing the gate. Nothing below this block —
    # neither the build-time wiring nor the `_search` closure — may touch
    # `entry.*`; use the local snapshot only. ``MetadataScopeConfig`` is
    # ``frozen=True`` so rebinding ``scope_config`` on the entry can't
    # mutate the snapshot, and the ``default`` MetadataScope is deep-copied
    # below to sever nested list/dict mutation paths (Pydantic ``frozen=True``
    # blocks rebinding, not mutation inside containers).
    entry_name = entry.name
    entry_description = entry.description
    retriever = entry.retriever
    # Caller-scope opt-in (the ``ka_metadata_scope`` bridge channel). Only a
    # retriever that honors ``search(metadata_scope=...)`` can apply it — the
    # ``documents=`` path is no longer used for caller-injected ids. Snapshot
    # both flags into the closure so a post-build entry mutation cannot flip them.
    accepts_caller_scope = entry.accepts_caller_scope
    supports_metadata_scope = isinstance(retriever, SupportsMetadataScope) and getattr(retriever, "supports_metadata_scope", None) is True
    scope_retriever = cast(SupportsMetadataScope, retriever) if supports_metadata_scope else None
    scope_config = entry.metadata_scope or MetadataScopeConfig()
    default_scope = copy.deepcopy(scope_config.default)
    resolver = scope_config.value_resolver
    default_search_kwargs = _resolve_search_kwargs(entry)
    tool_name = f"search_{entry_name}"
    exposed_axes = _validated_runtime_scope_axes(scope_config, entry_name)

    # Passthrough-mode observability (build-time logs — server-runtime factories
    # rebuild per request, so these fire on every build until the operator wires
    # a resolver or drops the axes from metadata_scope.exposed_axes):
    #
    # - INFO when ANY axis is exposed without a resolver. Operators reading a
    #   startup log can tell at a glance "this deployment runs passthrough on
    #   tool X for axes Y, Z" — successful hits log only at DEBUG, misses are
    #   silent, so a single startup line is the cheapest way to surface the
    #   chosen mode.
    # - WARN specifically when ``entity`` is in the set. H7 widens the entity
    #   boost to id OR name so planner-emitted names DO match the name leg —
    #   but a resolver still catches typos / variants the boost can't.
    if exposed_axes and resolver is None:
        logger.info(
            "Tool %s runs in metadata-scope passthrough mode (no MetadataValueResolver attached) — "
            "exposed axes %s forward raw planner values directly into *_boost. Hits log at DEBUG; "
            "misses produce no signal. Wire a resolver to canonicalize and surface miss warnings.",
            tool_name,
            sorted(exposed_axes),
        )
        if "entity" in exposed_axes:
            logger.warning(
                "Tool %s exposes the 'entity' axis without a metadata_value_resolver. "
                "Passthrough boosts on the raw planner value — typos / casing / out-of-vocab "
                "values silently contribute no boost. Wire a MetadataValueResolver to catch "
                "those, or restrict metadata_scope.exposed_axes to 'apcode' / 'app_name'.",
                tool_name,
            )

    async def _search(query: str, runtime: ToolRuntime, **runtime_scope_args: Any) -> Command:
        """Search for information using this retriever.

        Args:
            query: Search query text.
            runtime: LangGraph ToolRuntime (auto-injected by ToolNode).
            **runtime_scope_args: Optional ``apcode``/``app_name``/``entity``
                axes exposed when the entry's ``metadata_scope.exposed_axes``
                is set. Each value is routed through the entry's
                ``metadata_scope.value_resolver`` and unioned into the entry's
                ``metadata_scope.default`` as boosts via
                ``MetadataScope.add_boosts``. These are the *runtime query
                scope* axes — boost-only, never filter (the build-time
                ``metadata_scope.default`` filter ceiling is preserved).

        Returns:
            Command that writes SearchResponse to state and summary to messages.
        """
        scope: MetadataScopeLike | None = default_scope
        warnings: list[str] = []

        if exposed_axes:
            elastic_scope = scope if isinstance(scope, MetadataScope) else MetadataScope()
            if resolver is not None:
                elastic_scope, warnings = await _resolve_runtime_scope_args(
                    resolver=resolver,
                    exposed_axes=exposed_axes,
                    raw_axes=runtime_scope_args,
                    base_scope=elastic_scope,
                    tool_name=tool_name,
                    query=query,
                )
            else:
                # No resolver attached — raw LLM values pass straight into
                # boost axes (opt-in skip of client-side vocab cleanup).
                elastic_scope = _passthrough_runtime_scope_args(
                    exposed_axes=exposed_axes,
                    raw_axes=runtime_scope_args,
                    base_scope=elastic_scope,
                )
            scope = elastic_scope

        # Orchestrator-supplied per-call metadata scope (state-driven, NOT an
        # LLM tool arg — read from ``ka_metadata_scope`` via runtime, never from
        # the planner-visible signature). Each retriever resolves only the
        # backend-specific fields it recognizes. Scope models own the asymmetric
        # combination semantics: Elasticsearch intersects; LightRAG replaces its
        # opaque tag filter.
        #
        # The scope is applied only when this entry opted in
        # (``accepts_caller_scope``). It is meaningful only on a backend that
        # honors ``metadata_scope=...``; otherwise warn (don't crash) and skip,
        # independent of whether a build-time ``default_scope`` exists — a proxy
        # entry that supports metadata_scope but has no default still gets it.
        caller_doc_filter_applied = False
        caller_doc_id_count = 0
        if accepts_caller_scope:
            caller_bundle = _read_caller_scope_bundle(runtime)
            if caller_bundle:
                if scope_retriever is not None:
                    caller_scope = scope_retriever.resolve_caller_scope(caller_bundle)
                    if caller_scope is not None and caller_scope.is_effective():
                        scope = caller_scope if scope is None else scope.apply_caller_scope(caller_scope)
                        if isinstance(caller_scope, MetadataScope):
                            caller_doc_id_count = len(_as_list(caller_scope.doc_filter))
                            caller_doc_filter_applied = caller_doc_id_count > 0
                    else:
                        logger.warning(
                            "Tool %s received a non-empty caller metadata bundle, but retriever %s "
                            "recognizes no caller filters — caller scope NOT applied; its build-time scope remains active.",
                            tool_name,
                            type(retriever).__name__,
                        )
                else:
                    logger.warning(
                        "Tool %s received a caller metadata scope but retriever %s does not honor metadata_scope — caller scope NOT applied.",
                        tool_name,
                        type(retriever).__name__,
                    )

        search_kwargs = dict(default_search_kwargs)
        if scope is not None and scope.is_effective():
            search_kwargs["metadata_scope"] = scope

        # Token-budget split runs AFTER scope merge so caller budgets
        # divide correctly (test #25 in the design doc pins this).
        search_kwargs = _apply_token_budget_split(search_kwargs, tool_name, runtime)
        try:
            response = await retriever.search(query, **search_kwargs)
            n_results = len(response)
            # A hard doc-filter that yields zero hits is the main silent-failure
            # mode (e.g. orchestrator-supplied ids in the wrong id namespace vs
            # ``doc_keyword_field``). Surface it so the mismatch is diagnosable.
            if caller_doc_filter_applied and n_results == 0:
                logger.warning(
                    "Tool %s: doc-id filter (%d ids) matched ZERO documents for query %r — "
                    "the supplied ids may not match the retriever's document keyword field "
                    "(page-id vs chunk-id namespace).",
                    tool_name,
                    caller_doc_id_count,
                    query[:80],
                )
            summary = f"Retrieved {n_results} results from '{entry_name}'"
            if warnings:
                summary = summary + "\n\nMetadata-resolution warnings:\n- " + "\n- ".join(warnings)
            logger.debug("Tool %s: %s for query '%s'", tool_name, summary, query[:80])
        except asyncio.CancelledError:
            # Cooperative cancellation must propagate — turning it into a
            # "No results" reply would let the planner march on while the
            # upstream task tree is torn down.
            raise
        except Exception:
            logger.exception("Tool %s failed for query '%s'", tool_name, query[:80])
            error_update: dict[str, Any] = {
                "messages": [
                    ToolMessage(
                        content=f"Error: retriever '{entry_name}' failed. No results.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
            # Surface any resolution warnings collected before the failure so
            # operators can still see which axes were attempted; otherwise
            # the trail vanishes on every retriever exception.
            if warnings:
                error_update["resolution_warnings"] = warnings
            return Command(update=error_update)

        update: dict[str, Any] = {
            "retrieved_responses": {entry_name: response},
            "messages": [
                ToolMessage(
                    content=summary,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
        if warnings:
            update["resolution_warnings"] = warnings
        return Command(update=update)

    # Always rewrite `_search`'s signature so langchain auto-infers a tight
    # per-entry args_schema (LLM-visible) — never the raw
    # ``**runtime_scope_args: Any`` of the underlying function. When no axes
    # are exposed the rewritten signature is just ``query`` + ``runtime``
    # (the latter is detected as an injected arg by langchain). When axes
    # ARE exposed each becomes an explicit ``Optional[str]`` field.
    #
    # Passing a custom args_schema instead would suppress the runtime
    # injection — TypeError at call time. Both ``__signature__`` (used by
    # ``inspect.signature``) and ``__annotations__`` (used by
    # ``typing.get_type_hints``, which langchain's pydantic-based schema
    # inference relies on) must be updated.
    sig = _build_search_signature(exposed_axes)
    _search.__signature__ = sig  # type: ignore[attr-defined]
    _search.__annotations__ = _annotations_from_signature(sig, return_annotation=Command)
    if exposed_axes:
        _search.__doc__ = (entry_description or "") + "\n\n" + _runtime_scope_arg_docs(exposed_axes)

    return StructuredTool.from_function(
        coroutine=_search,
        name=tool_name,
        description=entry_description,
    )


def create_retriever_tools(entries: list[RetrieverEntry]) -> list[BaseTool]:
    """Create tools for a list of RetrieverEntries.

    Args:
        entries: List of retriever entries to convert to tools.

    Returns:
        List of LangGraph-compatible async tools, one per entry.
    """
    return [create_retriever_tool(e) for e in entries]


def _retriever_supports_documents(retriever: Any) -> bool:
    """Return ``True`` when ``retriever.search`` accepts a ``documents`` filter.

    The ``documents`` hard terms-filter is a direct ``search()`` parameter on
    backends that honor it (e.g. ``ElasticRetriever``), not part of
    ``to_search_kwargs``. Other backends either silently drop unknown kwargs
    (a doc-filter would be a no-op the caller thinks is enforced) or raise
    ``TypeError`` on them — so we must detect support before forwarding.
    Detection is by signature: an explicit ``documents`` parameter. A bare
    ``**kwargs`` does NOT count as support — silently swallowing the filter is
    exactly the failure mode this gate prevents.

    Args:
        retriever: The retriever backing this tool's entry.

    Returns:
        ``True`` iff ``search`` declares an explicit ``documents`` parameter.
    """
    search = getattr(retriever, "search", None)
    if search is None:
        return False
    try:
        params = inspect.signature(search).parameters
    except (TypeError, ValueError):
        return False
    param = params.get("documents")
    return param is not None and param.kind is not inspect.Parameter.VAR_KEYWORD


def _as_list(value: Any) -> list[str]:
    """Coerce a ``ScopeValue`` (``str | list[str] | None``) into a ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _read_caller_scope_bundle(runtime: ToolRuntime) -> CallerMetadataScope | None:
    """Read the normalized cross-backend caller scope bundle from state.

    Pulls ``ka_metadata_scope`` off ``runtime.state`` (a run-scoped input set by
    the orchestrator before delegating, never an LLM tool arg), validates it
    (dropping unknown / ``*_boost`` keys with a warning). Backend-specific
    resolution belongs to ``SupportsMetadataScope.resolve_caller_scope``.

    A missing / malformed scope, or one that normalizes to nothing, yields
    ``None`` so retrieval proceeds on the build-time ceiling alone.

    Args:
        runtime: LangGraph ToolRuntime with ``.state`` access.

    Returns:
        The normalized bundle, or ``None`` when nothing is set.
    """
    state = getattr(runtime, "state", None)
    if not state:
        return None
    scope_dict = read_ka_metadata_scope(state.get(KA_METADATA_SCOPE_KEY))
    return scope_dict or None


def _resolve_search_kwargs(entry: RetrieverEntry, llm_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge search_config defaults with optional LLM-provided overrides.

    Resolution order: search_config defaults → LLM overrides (win).

    Args:
        entry: RetrieverEntry with optional search_config.
        llm_overrides: Optional overrides from LLM tool calling (Phase 2).

    Returns:
        Merged kwargs for retriever.search().
    """
    base = entry.search_config.to_search_kwargs() if entry.search_config else {}
    if llm_overrides:
        base.update(llm_overrides)
    return base


# ---------------------------------------------------------------------------
# Token budget splitting
# ---------------------------------------------------------------------------


def _count_sibling_tool_calls(tool_name: str, runtime: ToolRuntime) -> int:
    """Count how many tool_calls in the current batch target the same tool.

    Reads the last AIMessage from state to find the parallel tool_calls
    that ToolNode is executing. Returns 1 if state is unavailable or
    the message doesn't contain tool_calls (safe no-op for splitting).

    Assumes ``add_messages`` reducer preserves insertion order so the most
    recent ``AIMessage`` (the planner's latest output that triggered this
    tool call) is reached first when iterating from the tail.
    """
    state = getattr(runtime, "state", None)
    if not state:
        return 1
    messages = state.get("messages") or []
    if not messages:
        return 1
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return sum(1 for tc in msg.tool_calls if tc.get("name") == tool_name)
    return 1


def _apply_token_budget_split(
    base_kwargs: dict[str, Any],
    tool_name: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """Split token budget fields proportionally across sibling tool calls.

    If the search kwargs contain any token budget field (max_total_tokens,
    max_entity_tokens, max_relation_tokens), divides each by the number
    of parallel calls to the same tool in this batch.

    Works generically — any retriever config that emits these fields gets
    automatic budget splitting. Configs without token fields pass through
    unchanged.

    Args:
        base_kwargs: Pre-resolved search kwargs from _resolve_search_kwargs.
        tool_name: Name of this tool (e.g. "search_lightrag_arch").
        runtime: ToolRuntime with state access.

    Returns:
        kwargs with token budgets divided by sibling count, or original
        kwargs unchanged if no token budget fields are present.
    """
    budget_fields = _TOKEN_BUDGET_FIELDS & base_kwargs.keys()
    if not budget_fields:
        return base_kwargs

    sibling_count = _count_sibling_tool_calls(tool_name, runtime)
    if sibling_count <= 1:
        return base_kwargs

    split_kwargs = dict(base_kwargs)
    for field in budget_fields:
        split_kwargs[field] = max(base_kwargs[field] // sibling_count, _MIN_TOKEN_BUDGET)

    logger.info(
        "Token budget split for %s: %d sibling calls, %s",
        tool_name,
        sibling_count,
        {f: f"{base_kwargs[f]} → {split_kwargs[f]}" for f in budget_fields},
    )
    return split_kwargs


# ---------------------------------------------------------------------------
# Runtime query scope — LLM tool args routed through MetadataValueResolver
# (boost-only; the build-time `default_scope` filter ceiling is preserved)
# ---------------------------------------------------------------------------


def _require_metadata_scope_support(entry: RetrieverEntry) -> None:
    """Refuse to wire scope features onto a retriever that doesn't support them.

    Scope features = ``entry.metadata_scope.default`` (build-time filter
    ceiling) and ``entry.metadata_scope.exposed_axes`` (runtime query scope
    opt-in). Both end up routed through
    ``retriever.search(metadata_scope=...)`` — a no-op on backends that don't
    declare ``SupportsMetadataScope``.

    Raises:
        TypeError: when scope features are set on an unsupported retriever.
            Names the entry and the offending retriever class so a consumer
            with a long entry list can locate the misconfig without grepping.
    """
    scope_config = entry.metadata_scope
    if scope_config is None:
        return
    # An ineffective default is documented as "full access", and `_search`
    # forwards only scopes whose `is_effective()` is true. Match that runtime
    # behavior exactly so a no-op default does not trip the capability gate.
    has_real_default_scope = scope_config.default is not None and scope_config.default.is_effective()
    needs_scope = has_real_default_scope or bool(scope_config.exposed_axes)
    if not needs_scope:
        return
    retriever = entry.retriever
    # Two gates: structural ``isinstance(...)`` for attribute presence AND a
    # strict ``is True`` check for value. The Protocol annotation is
    # ``Literal[True]`` (see scope_capability.py) — at runtime, only the
    # literal ``True`` honors that contract. A backend that declares
    # ``supports_metadata_scope = "yes"`` or ``= 1`` (truthy non-bool) would
    # pass plain truthiness and then fail to wire `metadata_scope` correctly.
    # The strict check keeps the runtime gate aligned with the static type.
    if not isinstance(retriever, SupportsMetadataScope) or getattr(retriever, "supports_metadata_scope", None) is not True:
        raise TypeError(
            f"RetrieverEntry name={entry.name!r} configures scope features "
            f"(default_scope={scope_config.default is not None}, exposed_axes={list(scope_config.exposed_axes)!r}) "
            f"but {type(retriever).__name__} does not satisfy SupportsMetadataScope. "
            f"Either drop the scope features or declare "
            f"`supports_metadata_scope: ClassVar[Literal[True]] = True` on the retriever class "
            f"(see sta_agent_core.repositories.retrievers.scope_capability)."
        )


def _validated_runtime_scope_axes(scope_config: MetadataScopeConfig, entry_name: str) -> list[str]:
    """Return validated, ordered runtime-scope axes exposed by the entry.

    Skips silently when ``exposed_axes`` is empty so legacy entries pay zero
    cost. ``value_resolver`` is opt-in: when absent, raw LLM-emitted axis
    values flow straight into ``MetadataScope.add_boosts`` without
    client-side canonicalization. Boost clauses are ``should``-only
    additions on the retriever side — a wrong/unknown value contributes
    nothing (silent precision miss, never a scope-widening or correctness
    failure). Attach a resolver only when you want LLM typos to be caught
    and cleaned before the boost is built.
    """
    raw = scope_config.exposed_axes
    if not raw:
        return []
    unknown = set(raw) - _RUNTIME_SCOPE_AXES
    if unknown:
        raise ValueError(f"metadata_scope.exposed_axes contains unknown axes {sorted(unknown)} — must be subset of {sorted(_RUNTIME_SCOPE_AXES)}")
    if scope_config.value_resolver is None:
        logger.debug(
            "exposed_axes set without value_resolver on entry=%r — raw LLM values pass through to boost axes",
            entry_name,
        )
    # Preserve caller-specified order so the generated tool schema is stable.
    # ``unknown`` above already rejects anything outside _RUNTIME_SCOPE_AXES;
    # no second filter needed.
    return list(raw)


def _build_search_signature(axes: list[str]) -> inspect.Signature:
    """Build the signature langchain inspects to derive args_schema.

    Layout: ``query: str``, ``runtime: ToolRuntime`` (injected — excluded
    from LLM-visible schema), then one keyword-only ``Optional[str] = None``
    per opted-in axis. Anything not in the signature stays out of the
    schema the planner sees.
    """
    params: list[inspect.Parameter] = [
        inspect.Parameter("query", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        inspect.Parameter("runtime", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=ToolRuntime),
    ]
    for axis in axes:
        params.append(
            inspect.Parameter(
                axis,
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=str | None,
            )
        )
    return inspect.Signature(parameters=params)


def _runtime_scope_arg_docs(axes: list[str]) -> str:
    """Docstring tail describing the exposed axes (helps tool-description rendering)."""
    lines = ["Args:", "    query: Search query text."]
    for axis in axes:
        lines.append(f"    {axis}: {_AXIS_DESCRIPTIONS[axis]}")
    return "\n".join(lines)


def _annotations_from_signature(sig: inspect.Signature, *, return_annotation: Any) -> dict[str, Any]:
    """Convert an `inspect.Signature` back into a `__annotations__` dict.

    Langchain's pydantic-backed schema inference uses
    ``typing.get_type_hints(func)`` (which reads ``__annotations__``)
    rather than ``inspect.signature``, so a signature override alone is
    not enough — both surfaces must agree.
    """
    annotations: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if p.annotation is not inspect.Parameter.empty:
            annotations[name] = p.annotation
    annotations["return"] = return_annotation
    return annotations


async def _resolve_runtime_scope_args(
    resolver: MetadataValueResolver,
    exposed_axes: list[str],
    raw_axes: dict[str, str | None],
    base_scope: MetadataScope,
    tool_name: str,
    query: str,
) -> tuple[MetadataScope, list[str]]:
    """Route raw LLM axes through the resolver, union into boost-only scope.

    The resolver is passed in directly (rather than read off ``entry``) so a
    post-build mutation of ``entry.metadata_scope`` cannot redirect runtime
    resolution. Caller (``create_retriever_tool``) snapshots the resolver at
    build time.

    Returns the updated scope and an ordered warnings list.
    """
    boost_updates: dict[str, list[str]] = {}
    warnings: list[str] = []

    for axis in exposed_axes:
        raw = raw_axes.get(axis)
        if raw is None or not isinstance(raw, str) or not raw.strip():
            continue
        try:
            resolution = await resolver.resolve(axis, raw)
        except Exception as exc:  # noqa: BLE001 — third-party resolver, contain blast radius
            warnings.append(f"metadata resolver raised on axis={axis!r} value={raw!r} ({type(exc).__name__}); ignoring boost")
            logger.warning(
                "metadata_resolver_raised",
                extra={"tool": tool_name, "axis": axis, "raw": raw},
                exc_info=True,
            )
            continue

        if not _valid_resolution(resolution):
            warnings.append(
                f"resolver returned invalid Resolution for axis={axis!r} value={raw!r} "
                f"(matched={resolution.matched}, target_axis_boost={resolution.target_axis_boost!r}); ignoring boost"
            )
            logger.warning(
                "metadata_resolution_invalid",
                extra={
                    "tool": tool_name,
                    "axis": axis,
                    "raw": raw,
                    "matched": resolution.matched,
                    "target_axis_boost": resolution.target_axis_boost,
                },
            )
            continue

        if resolution.matched:
            # _valid_resolution above guarantees both are non-None when matched
            # and that value is either a non-empty str or a non-empty list[str].
            target_axis: str = resolution.target_axis_boost  # type: ignore[assignment]
            value = resolution.value
            bucket = boost_updates.setdefault(target_axis, [])
            # A list value happens when one raw name resolves to multiple ids
            # (duplicate display names). Flatten into ``bucket`` so add_boosts
            # unions every id; the alternative (append) would wrap the list
            # and trip the empty-string validator.
            if isinstance(value, list):
                bucket.extend(value)
            elif isinstance(value, str):
                bucket.append(value)
        if resolution.warning:
            warnings.append(resolution.warning)
            logger.warning(
                "metadata_resolution",
                extra={
                    "tool": tool_name,
                    "axis": axis,
                    "raw": raw,
                    "resolved": resolution.value,
                    "score": resolution.score,
                    "query_prefix": query[:80],
                    "matched": resolution.matched,
                },
            )

    if boost_updates:
        # add_boosts is boost-only by construction — the build-time
        # `default_scope` filter ceiling is untouched no matter what the
        # planner emits.
        base_scope = base_scope.add_boosts(**boost_updates)
    return base_scope, warnings


def _passthrough_runtime_scope_args(
    exposed_axes: list[str],
    raw_axes: dict[str, str | None],
    base_scope: MetadataScope,
) -> MetadataScope:
    """Forward raw LLM-emitted axis values straight into boost axes (no resolver).

    Used when ``expose_metadata_args`` is opted in without a
    ``MetadataValueResolver``. Each axis ``X`` maps to ``X_boost`` on
    ``MetadataScope.add_boosts`` (boost-only by construction — the build-time
    ``default_scope`` filter ceiling cannot be widened or narrowed here).
    Empty / whitespace values are skipped; strings are stripped.

    Trade-off vs. attaching a resolver:

    - **Correctness**: typos / casing / out-of-vocab values land in
      ``should`` clauses verbatim. They either match (precision lift) or
      don't (silent no-op). They never widen the filter ceiling or leak
      cross-tenant docs.
    - **Observability**: the resolver path emits warnings to
      ``state["resolution_warnings"]``, the ToolMessage tail, and
      ``logger.warning`` on a miss. The passthrough path is intentionally
      silent on the agent state (no pollution of the planner's context) —
      successful hits log at DEBUG only, misses produce no signal at all.
      A deployment running passthrough cannot tell "boost applied" from
      "boost matched nothing due to typo" without inspecting the retriever's
      response itself.
    """
    # Use ``dict[str, list[str]]`` to mirror the shape produced by the
    # resolver path; ``add_boosts`` accepts both ``str`` and ``list[str]``
    # but keeping one canonical shape avoids surprise when both paths union
    # into the same scope.
    boost_updates: dict[str, list[str]] = {}
    for axis in exposed_axes:
        raw = raw_axes.get(axis)
        if raw is None or not isinstance(raw, str) or not raw.strip():
            continue
        boost_updates[f"{axis}_boost"] = [raw.strip()]
    if boost_updates:
        logger.debug(
            "runtime_scope_passthrough applied boosts=%s (no resolver attached — values forwarded verbatim)",
            boost_updates,
        )
        return base_scope.add_boosts(**boost_updates)
    return base_scope


def _valid_resolution(resolution: Any) -> bool:
    """Sanity-check a ``Resolution`` from a third-party resolver before merging.

    An unmatched resolution is always valid (no boost merge happens).
    A matched resolution must carry both a non-empty value (str or list[str])
    AND a known boost-axis target — otherwise ``MetadataScope.add_boosts``
    raises mid-tool (before the search ``try/except``) and the whole tool
    call aborts.
    """
    if not resolution.matched:
        return True
    if resolution.target_axis_boost not in _BOOST_AXES:
        return False
    value = resolution.value
    if value is None:
        return False
    if isinstance(value, list):
        return bool(value) and all(isinstance(v, str) and v for v in value)
    return isinstance(value, str) and bool(value)

-------

tests/test_ai_engine/agents/knowledge_agent/test_caller_scope_input_seed.py
----
"""End-to-end: a caller seeds ``ka_metadata_scope`` in the graph input state and
the Knowledge Agent hard-filters retrieval to it.

This pins the *producer* contract for the caller-supplied request scope: the
scope is delivered as plain input state (no header, no planner tool arg), it
flows through the graph to the retriever tool, and it reaches the backend as a
FILTER-ONLY :class:`MetadataScope` — but only for an entry that opts in via
``accepts_caller_scope=True``. An entry that does not opt in never sees it.

The consumption mechanics (axis mapping, boost-key dropping, the
``SupportsMetadataScope`` gate) are unit-tested in
``test_retriever_tool_factory_doc_filter.py``; this test exercises the full
input → state → tool → retriever path so a regression that breaks input
seeding (e.g. marking the channel input-omitted, or the delegation tool
dropping the key) is caught.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from sta_agent_core.repositories import RetrievalChunk, SearchResponse
from sta_agent_core.repositories.retrievers import BaseSearchConfig
from sta_agent_core.repositories.retrievers.elasticsearch.metadata_scope import MetadataScope
from sta_agent_engine.agents.knowledge_agent import (
    KA_METADATA_SCOPE_KEY,
    KaMetadataScope,
    KnowledgeAgentConfig,
    RetrieverEntry,
    create_knowledge_agent,
)
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_config import ExpandConfig
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_types import (
    CompressedFinding,
    CompressedFindings,
    KeyFactEntry,
)
from sta_agent_engine.models import AgentIntegrationModel

from .conftest import make_plan_tool_calls


class _RecordingScopeRetriever:
    """Retriever double that honors ``metadata_scope=`` and records what it received.

    Declares ``supports_metadata_scope`` so the tool factory's capability gate
    forwards caller scope to it, and keeps a real ``search`` (not an AsyncMock)
    so signature-based capability detection stays intact.
    """

    supports_metadata_scope = True

    @staticmethod
    def resolve_caller_scope(bundle: Mapping[str, Any]) -> MetadataScope | None:
        return MetadataScope.from_caller_scope(bundle)

    def __init__(self, name: str, chunks: list[RetrievalChunk]) -> None:
        self.name = name
        self._chunks = chunks
        self.received_scope: MetadataScope | None = None
        self.search_called = False

    async def search(
        self,
        query: str,
        size: int = 10,
        *,
        metadata_scope: MetadataScope | None = None,
        **kwargs: Any,
    ) -> SearchResponse[RetrievalChunk]:
        self.search_called = True
        self.received_scope = metadata_scope
        return SearchResponse(results=self._chunks)

    async def close(self) -> None:
        return None


def _chunks(page_id: str, retriever_name: str) -> list[RetrievalChunk]:
    return [
        RetrievalChunk(
            content="Restart the pod with kubectl rollout restart.",
            chunk_id=f"{page_id}-chunk-0",
            score=0.9,
            source_url=f"https://docs.example.com/{page_id}",
            retriever_type=retriever_name,
            metadata={"title": f"Doc {page_id}", "doc": f"docs/{page_id}.md", "pageId": page_id, "chunk_index": 0},
        )
    ]


def _compress_response() -> CompressedFindings:
    return CompressedFindings(
        findings=[
            CompressedFinding(
                topic="Pod Restart",
                summary="Pods restart with kubectl rollout restart.",
                key_facts=[KeyFactEntry(fact="Use kubectl rollout restart", source_index=1)],
                confidence="high",
            )
        ]
    )


def _entry(retriever: _RecordingScopeRetriever, *, accepts_caller_scope: bool) -> RetrieverEntry:
    return RetrieverEntry(
        name=retriever.name,
        description="Search runbook procedures for incident response and operations",
        retriever=retriever,  # type: ignore[arg-type]
        search_config=BaseSearchConfig(top_k=10),
        accepts_caller_scope=accepts_caller_scope,
    )


def _offline_config() -> KnowledgeAgentConfig:
    # Empty task_model_defaults so every node falls through to the fake model;
    # fast depth + expand off keeps the path plan → tool → collect → compress → output.
    return KnowledgeAgentConfig(task_model_defaults={}, max_iterations=1, expand=ExpandConfig(enabled=False))


def _planner_model() -> AgentIntegrationModel:
    return AgentIntegrationModel(
        responses=[AIMessage(content="", tool_calls=make_plan_tool_calls(["scoped_runbooks"]))],
        structured_responses={CompressedFindings: _compress_response()},
    )


@pytest.mark.integration_offline
@pytest.mark.asyncio
async def test_input_seeded_scope_reaches_opted_in_retriever(no_llm_calls: None) -> None:
    """Caller-seeded ``ka_metadata_scope`` reaches an opted-in retriever as a hard filter."""
    retriever = _RecordingScopeRetriever("scoped_runbooks", _chunks("page-101", "scoped_runbooks"))
    graph = create_knowledge_agent([_entry(retriever, accepts_caller_scope=True)], model=_planner_model(), config=_offline_config())

    seed: KaMetadataScope = {"doc_ids": ["page-101"], "apcode": ["AP12345"]}
    await graph.ainvoke({"query": "How do I restart a Kubernetes pod?", KA_METADATA_SCOPE_KEY: seed})

    assert retriever.search_called
    assert retriever.received_scope is not None
    # doc_ids ride INSIDE the scope's doc_filter (not a separate documents= kwarg);
    # apcode lands on the matching hard-filter axis. Both narrow, neither boosts.
    assert retriever.received_scope.doc_filter == ["page-101"]
    assert retriever.received_scope.apcode_filter == ["AP12345"]
    assert retriever.received_scope.apcode_boost is None


@pytest.mark.integration_offline
@pytest.mark.asyncio
async def test_input_seeded_scope_excludes_non_opted_in_retriever(no_llm_calls: None) -> None:
    """A non-empty caller scope never falls back to a non-opted-in entry."""
    retriever = _RecordingScopeRetriever("scoped_runbooks", _chunks("page-101", "scoped_runbooks"))
    graph = create_knowledge_agent([_entry(retriever, accepts_caller_scope=False)], model=_planner_model(), config=_offline_config())

    seed: KaMetadataScope = {"doc_ids": ["page-101"], "apcode": ["AP12345"]}
    await graph.ainvoke({"query": "How do I restart a Kubernetes pod?", KA_METADATA_SCOPE_KEY: seed})

    assert not retriever.search_called
    assert retriever.received_scope is None


@pytest.mark.unit
def test_ka_metadata_scope_is_caller_seedable_in_input_schema() -> None:
    """``ka_metadata_scope`` is in the KA graph input schema, so a caller can seed it."""
    retriever = _RecordingScopeRetriever("scoped_runbooks", _chunks("page-101", "scoped_runbooks"))
    graph = create_knowledge_agent([_entry(retriever, accepts_caller_scope=True)], model=AgentIntegrationModel(), config=_offline_config())

    input_props = graph.get_input_jsonschema()["properties"]
    assert KA_METADATA_SCOPE_KEY in input_props

-------

tests/test_ai_engine/agents/knowledge_agent/test_output_node_resolution_warnings.py
----
"""Tests for `OutputNode` mirroring of metadata-scope warnings.

Phase 5 — when the retriever tool factory writes
``state["resolution_warnings"]`` (runtime-query-scope fuzzy hits / misses),
the final ``KnowledgeAgentFindings`` must carry the same list under
``metadata["warnings"]`` so consumers (and the answer-mode
synthesizer) can surface them.
"""

from __future__ import annotations

from typing import Any

import pytest

from sta_agent_engine.agents.knowledge_agent.knowledge_agent_types import (
    KnowledgeAgentAnswer,
    KnowledgeAgentFindings,
)
from sta_agent_engine.agents.knowledge_agent.nodes.output import OutputNode


@pytest.fixture
def output_node() -> OutputNode:
    return OutputNode()


class TestOutputNodeResolutionWarnings:
    @pytest.mark.asyncio
    async def test_warnings_mirrored_into_findings_metadata(self, output_node: OutputNode) -> None:
        state = {
            "query": "deploy invoice service",
            "findings": [],
            "retrieved_responses": {},
            "iteration_count": 1,
            "resolution_warnings": ["apcode='QQQ' not found"],
        }
        result = await output_node(state, config={})  # type: ignore[arg-type]
        out = result["result"]
        assert isinstance(out, KnowledgeAgentFindings)
        assert out.metadata["warnings"] == ["apcode='QQQ' not found"]

    @pytest.mark.asyncio
    async def test_no_warnings_no_warnings_key(self, output_node: OutputNode) -> None:
        state = {
            "query": "q",
            "findings": [],
            "retrieved_responses": {},
            "iteration_count": 1,
        }
        result = await output_node(state, config={})  # type: ignore[arg-type]
        out = result["result"]
        assert isinstance(out, KnowledgeAgentFindings)
        assert "warnings" not in out.metadata

    @pytest.mark.asyncio
    async def test_warnings_propagate_through_answer_mode(self, output_node: OutputNode) -> None:
        """In answer mode the evidence bundle is wrapped in KnowledgeAgentAnswer
        — warnings still travel via .evidence.metadata."""
        state = {
            "query": "q",
            "findings": [],
            "retrieved_responses": {},
            "iteration_count": 1,
            "answer": "Synth answer text.",
            "answer_attempt": 1,
            "answer_citations": [],
            "resolution_warnings": ["apcode='QQQ' not found", "entity='ZZZ' not found"],
        }
        result = await output_node(state, config={})  # type: ignore[arg-type]
        out = result["result"]
        assert isinstance(out, KnowledgeAgentAnswer)
        assert out.evidence.metadata["warnings"] == [
            "apcode='QQQ' not found",
            "entity='ZZZ' not found",
        ]

    @pytest.mark.asyncio
    async def test_empty_warnings_list_treated_as_no_warnings(self, output_node: OutputNode) -> None:
        state = {
            "query": "q",
            "findings": [],
            "retrieved_responses": {},
            "iteration_count": 1,
            "resolution_warnings": [],
        }
        result = await output_node(state, config={})  # type: ignore[arg-type]
        out = result["result"]
        assert isinstance(out, KnowledgeAgentFindings)
        assert "warnings" not in out.metadata


# ---------------------------------------------------------------------------
# End-to-end: warnings travel through the real graph reducer, not just OutputNode
# ---------------------------------------------------------------------------


class _StubResolverEmittingWarning:
    """Resolver double that always reports a miss with a warning."""

    def __init__(self, warning: str = "apcode='QQQ' not found") -> None:
        self._warning = warning
        self.calls: list[tuple[str, str]] = []

    async def resolve(self, axis: str, value: str) -> Any:
        from sta_agent_core.repositories.retrievers.metadata_value_resolver import Resolution

        self.calls.append((axis, value))
        return Resolution(
            matched=False,
            value=None,
            target_axis_boost=None,
            warning=self._warning,
            score=None,
            suggestions=[],
        )

    async def refresh(self) -> None:
        return None


@pytest.mark.integration_offline
class TestResolutionWarningsEndToEnd:
    """End-to-end: a resolver miss emitted by the tool-factory closure must
    travel through the real graph's ``Annotated[list[str], add]`` reducer
    and land in ``KnowledgeAgentFindings.metadata['warnings']`` without
    anyone seeding ``state['resolution_warnings']`` manually.
    """

    @pytest.fixture(autouse=True)
    def _guard(self, no_llm_calls: None) -> None:
        pass

    @pytest.mark.asyncio
    async def test_warning_emitted_by_resolver_lands_in_findings_metadata(
        self,
        integration_chunks_a: list,
    ) -> None:
        from unittest.mock import AsyncMock

        from langchain_core.messages import AIMessage

        from sta_agent_core.repositories import SearchResponse
        from sta_agent_core.repositories.retrievers.mock import MockRetriever
        from sta_agent_engine.agents.knowledge_agent import (
            CoverageAssessment,
            KnowledgeAgentConfig,
            RetrieverEntry,
            create_knowledge_agent,
        )
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_config import ExpandConfig
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_types import CompressedFindings
        from sta_agent_engine.models import AgentIntegrationModel

        # Mock retriever that satisfies SupportsMetadataScope so the tool
        # factory wires scope features through it.
        class _ScopedMock(MockRetriever):
            supports_metadata_scope = True

            @staticmethod
            def resolve_caller_scope(bundle: dict[str, Any]) -> None:
                return None

        retriever = _ScopedMock(name="elastic_runbooks", num_results=2)
        retriever.search = AsyncMock(return_value=SearchResponse(results=integration_chunks_a))  # type: ignore[method-assign]
        retriever.close = AsyncMock()  # type: ignore[method-assign]

        resolver = _StubResolverEmittingWarning(warning="apcode='QQQ' not found")
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_types import MetadataScopeConfig

        entry = RetrieverEntry(
            name="elastic_runbooks",
            description="Search runbook procedures",
            retriever=retriever,
            metadata_scope=MetadataScopeConfig(
                exposed_axes=("apcode",),
                value_resolver=resolver,  # type: ignore[arg-type]
            ),
        )

        # Planner emits a single tool call carrying the unresolvable
        # ``apcode="QQQ"`` so the resolver fires once.
        tool_calls = [
            {
                "name": "search_elastic_runbooks",
                "args": {"query": "How do I restart a Kubernetes pod?", "apcode": "QQQ"},
                "id": "call_0",
                "type": "tool_call",
            }
        ]
        model = AgentIntegrationModel(
            responses=[AIMessage(content="", tool_calls=tool_calls)],
            structured_responses={
                CompressedFindings: CompressedFindings(findings=[]),
                CoverageAssessment: CoverageAssessment(
                    sufficient=True,
                    gaps=[],
                    reasoning="Done.",
                ),
            },
        )

        graph = create_knowledge_agent(
            [entry],
            model=model,
            config=KnowledgeAgentConfig(
                task_model_defaults={},
                max_iterations=1,
                expand=ExpandConfig(enabled=False),
            ),
        )

        result = await graph.ainvoke(
            {"query": "How do I restart a Kubernetes pod?"},
            context={"search_depth": "fast"},
        )

        findings = result["result"]
        assert isinstance(findings, KnowledgeAgentFindings)
        # The resolver was called for the apcode axis.
        assert ("apcode", "QQQ") in resolver.calls
        # And its warning landed in findings.metadata via the real reducer
        # + OutputNode mirror — no manual state seeding.
        assert findings.metadata.get("warnings") == ["apcode='QQQ' not found"]

-------

tests/test_ai_engine/agents/knowledge_agent/test_plan_queries_caller_scope.py
----
"""Caller-scope hardening for query planning.

A caller-supplied ``ka_metadata_scope`` is a bundle of backend-specific filters.
When it is non-empty, ``PlanQueriesNode`` binds entries that both opt in and
resolve an effective scope, plus entries explicitly named in
``include_without_caller_scope``. With no scope, normal full-tool routing stays
unchanged.

These tests pin both the ``_active_plan_inputs`` selection logic and the
end-to-end ``__call__`` guarantee (a hallucinated call to a non-accepting corpus
is dropped when a scope is present).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from sta_agent_core.repositories.retrievers.elasticsearch.metadata_scope import MetadataScope
from sta_agent_core.repositories.retrievers.lightrag import LightRAGMetadataScope
from sta_agent_core.repositories.retrievers.mock import MockRetriever
from sta_agent_engine.agents.knowledge_agent import KnowledgeAgentConfig, KnowledgeAgentState, RetrieverEntry
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_config import PlanConfig
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_retrievers import create_mock_entry
from sta_agent_engine.agents.knowledge_agent.nodes.plan_queries import PlanQueriesNode
from sta_agent_engine.agents.knowledge_agent.tools import create_retriever_tools
from sta_agent_engine.models import AgentIntegrationModel


pytestmark = pytest.mark.unit


class _ResolvingMockRetriever(MockRetriever):
    supports_metadata_scope = True

    def __init__(self, *, name: str, backend: str) -> None:
        super().__init__(name=name)
        self.backend = backend

    def resolve_caller_scope(self, bundle: dict[str, Any]) -> MetadataScope | LightRAGMetadataScope | None:
        if self.backend == "lightrag":
            return LightRAGMetadataScope.from_caller_scope(bundle)
        return MetadataScope.from_caller_scope(bundle)


def _accepting(
    name: str,
    description: str = "Broad docs",
    *,
    backend: str = "elastic",
) -> RetrieverEntry:
    """A mock retriever entry that opts into caller scope."""
    return RetrieverEntry(
        name=name,
        description=description,
        retriever=_ResolvingMockRetriever(name=name, backend=backend),
        accepts_caller_scope=True,
    )


def _not_accepting(name: str, description: str = "Scoped docs") -> RetrieverEntry:
    """A mock retriever entry that does NOT opt into caller scope (the default)."""
    return create_mock_entry(name=name, description=description)


def _tool_calls_config(**plan_kwargs: Any) -> KnowledgeAgentConfig:
    plan_kwargs.setdefault("planning_strategy", "tool_calls")
    plan_kwargs.setdefault("include_original_query", False)
    return KnowledgeAgentConfig(plan=PlanConfig(**plan_kwargs))


def _tool_call(name: str, query: str, call_id: str = "tc") -> dict[str, Any]:
    return {"id": call_id, "name": name, "args": {"query": query}, "type": "tool_call"}


def _names(tools: list[Any]) -> list[str]:
    return sorted(tool.name for tool in tools)


class TestActivePlanInputs:
    def test_restricts_to_scope_accepting_when_doc_ids_present(self) -> None:
        entries = [_accepting("general_doc"), _not_accepting("twin_project_doc")]
        node = PlanQueriesNode(tools=create_retriever_tools(entries), entries=entries)

        tools, active_entries, args = node._active_plan_inputs({"ka_metadata_scope": {"doc_ids": ["d1"]}})

        assert _names(tools) == ["search_general_doc"]
        assert [entry.name for entry in (active_entries or [])] == ["general_doc"]
        assert set(args) == {"search_general_doc"}

    def test_no_scope_keeps_all_tools(self) -> None:
        entries = [_accepting("general_doc"), _not_accepting("twin_project_doc")]
        node = PlanQueriesNode(tools=create_retriever_tools(entries), entries=entries)

        tools, active_entries, args = node._active_plan_inputs({})

        assert _names(tools) == ["search_general_doc", "search_twin_project_doc"]
        assert [entry.name for entry in (active_entries or [])] == ["general_doc", "twin_project_doc"]
        assert set(args) == {"search_general_doc", "search_twin_project_doc"}

    def test_triggers_on_any_caller_scope_axis(self) -> None:
        # apcode-only scope leaks the same way doc_ids does — restriction must fire.
        entries = [_accepting("general_doc"), _not_accepting("twin_project_doc")]
        node = PlanQueriesNode(tools=create_retriever_tools(entries), entries=entries)

        tools, _, _ = node._active_plan_inputs({"ka_metadata_scope": {"apcode": ["AP1"]}})

        assert _names(tools) == ["search_general_doc"]

    def test_mixed_bundle_keeps_retrievers_with_relevant_axes(self) -> None:
        entries = [
            _accepting("elastic"),
            _accepting("lightrag", backend="lightrag"),
        ]
        node = PlanQueriesNode(tools=create_retriever_tools(entries), entries=entries)

        tools, active_entries, _ = node._active_plan_inputs(
            {
                "ka_metadata_scope": {
                    "doc_ids": ["d1"],
                    "tag_filter": {"private_operator": ["tag-a"]},
                }
            }
        )

        assert _names(tools) == ["search_elastic", "search_lightrag"]
        assert [entry.name for entry in (active_entries or [])] == ["elastic", "lightrag"]

    def test_tag_filter_excludes_es_only_retriever(self) -> None:
        entries = [
            _accepting("elastic"),
            _accepting("lightrag", backend="lightrag"),
        ]
        node = PlanQueriesNode(tools=create_retriever_tools(entries), entries=entries)

        tools, active_entries, _ = node._active_plan_inputs({"ka_metadata_scope": {"tag_filter": {"private_operator": ["tag-a"]}}})

        assert _names(tools) == ["search_lightrag"]
        assert [entry.name for entry in (active_entries or [])] == ["lightrag"]

    def test_explicitly_includes_retriever_without_matching_caller_filter(self) -> None:
        entries = [
            _accepting("elastic"),
            _accepting("lightrag", backend="lightrag"),
        ]
        node = PlanQueriesNode(tools=create_retriever_tools(entries), entries=entries)

        tools, active_entries, _ = node._active_plan_inputs(
            {
                "ka_metadata_scope": {
                    "doc_ids": ["d1"],
                    "include_without_caller_scope": ["lightrag"],
                }
            }
        )

        assert _names(tools) == ["search_elastic", "search_lightrag"]
        assert [entry.name for entry in (active_entries or [])] == ["elastic", "lightrag"]

    def test_matching_is_required_when_no_explicit_include_is_present(self) -> None:
        entries = [_accepting("lightrag", backend="lightrag")]
        node = PlanQueriesNode(tools=create_retriever_tools(entries), entries=entries)

        tools, active_entries, args = node._active_plan_inputs({"ka_metadata_scope": {"doc_ids": ["d1"]}})

        assert tools == []
        assert active_entries == []
        assert args == {}

    def test_binds_no_tools_when_no_entry_accepts_scope(self) -> None:
        entries = [_not_accepting("alpha"), _not_accepting("beta")]
        node = PlanQueriesNode(tools=create_retriever_tools(entries), entries=entries)

        tools, active_entries, args = node._active_plan_inputs({"ka_metadata_scope": {"doc_ids": ["d1"]}})

        assert tools == []
        assert active_entries == []
        assert args == {}

    def test_keeps_all_when_entries_unavailable(self) -> None:
        # The flag lives on the entry; with no entries the node can't filter.
        entries = [_accepting("general_doc"), _not_accepting("twin_project_doc")]
        node = PlanQueriesNode(tools=create_retriever_tools(entries))  # entries=None

        tools, _, _ = node._active_plan_inputs({"ka_metadata_scope": {"doc_ids": ["d1"]}})

        assert _names(tools) == ["search_general_doc", "search_twin_project_doc"]


class TestCallerScopeEndToEnd:
    @pytest.mark.asyncio
    async def test_call_drops_non_scope_accepting_calls_when_scope_present(self) -> None:
        entries = [_accepting("general_doc"), _not_accepting("twin_project_doc")]
        tools = create_retriever_tools(entries)
        model = AgentIntegrationModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("search_general_doc", "q1", call_id="a"),
                        _tool_call("search_twin_project_doc", "q2", call_id="b"),
                    ],
                )
            ],
        )
        node = PlanQueriesNode(tools=tools, entries=entries, default_model=model, agent_config=_tool_calls_config())
        state = KnowledgeAgentState(
            query="q",
            messages=[],
            iteration_count=0,
            findings=[],
            ka_metadata_scope={"doc_ids": ["d1"]},
        )

        with patch.object(node, "_resolve_model_for_task", return_value=model):
            result = await node(state, config={})

        assert [tc["name"] for tc in result["messages"][0].tool_calls] == ["search_general_doc"]

    @pytest.mark.asyncio
    async def test_call_keeps_all_tools_when_no_scope(self) -> None:
        entries = [_accepting("general_doc"), _not_accepting("twin_project_doc")]
        tools = create_retriever_tools(entries)
        model = AgentIntegrationModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("search_general_doc", "q1", call_id="a"),
                        _tool_call("search_twin_project_doc", "q2", call_id="b"),
                    ],
                )
            ],
        )
        node = PlanQueriesNode(tools=tools, entries=entries, default_model=model, agent_config=_tool_calls_config())
        state = KnowledgeAgentState(query="q", messages=[], iteration_count=0, findings=[])

        with patch.object(node, "_resolve_model_for_task", return_value=model):
            result = await node(state, config={})

        assert [tc["name"] for tc in result["messages"][0].tool_calls] == ["search_general_doc", "search_twin_project_doc"]

    @pytest.mark.asyncio
    async def test_call_drops_all_calls_when_no_entry_accepts_scope(self) -> None:
        """A non-empty caller scope never falls back to an unfiltered corpus."""
        entries = [_not_accepting("general_doc"), _not_accepting("twin_project_doc")]
        tools = create_retriever_tools(entries)
        model = AgentIntegrationModel(
            responses=[AIMessage(content="", tool_calls=[_tool_call("search_general_doc", "q1", call_id="a")])],
        )
        node = PlanQueriesNode(tools=tools, entries=entries, default_model=model, agent_config=_tool_calls_config())
        state = KnowledgeAgentState(query="q", messages=[], iteration_count=0, findings=[], ka_metadata_scope={"doc_ids": ["d1"]})

        with patch.object(node, "_resolve_model_for_task", return_value=model):
            result = await node(state, config={})

        assert result["messages"][0].tool_calls == []
        assert result["plan_failed"] is True
        assert not result.get("resolution_warnings")

    @pytest.mark.asyncio
    async def test_call_no_unenforced_warning_when_scope_applies(self) -> None:
        """When a scope-accepting corpus exists, the filter applies — no warning."""
        entries = [_accepting("general_doc"), _not_accepting("twin_project_doc")]
        tools = create_retriever_tools(entries)
        model = AgentIntegrationModel(
            responses=[AIMessage(content="", tool_calls=[_tool_call("search_general_doc", "q1", call_id="a")])],
        )
        node = PlanQueriesNode(tools=tools, entries=entries, default_model=model, agent_config=_tool_calls_config())
        state = KnowledgeAgentState(query="q", messages=[], iteration_count=0, findings=[], ka_metadata_scope={"doc_ids": ["d1"]})

        with patch.object(node, "_resolve_model_for_task", return_value=model):
            result = await node(state, config={})

        warnings = result.get("resolution_warnings") or []
        assert not any("caller_scope_unenforced" in w for w in warnings)

-------

tests/test_ai_engine/agents/knowledge_agent/test_plan_queries_structured_output.py
----
"""Tests for structured-output query planning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import Field

from sta_agent_core.repositories.retrievers.elasticsearch.metadata_scope import MetadataScope
from sta_agent_core.repositories.retrievers.mock import MockRetriever
from sta_agent_engine.agents.knowledge_agent import KnowledgeAgentConfig, KnowledgeAgentState, MetadataScopeConfig, RetrieverEntry
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_config import PlanConfig
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_routing import route_after_plan
from sta_agent_engine.agents.knowledge_agent.nodes.plan_queries import PLAN_FAILED_MESSAGE, PlanQueriesNode
from sta_agent_engine.agents.knowledge_agent.tools import create_retriever_tools
from sta_agent_engine.models.fake_models import AgentIntegrationModel


class _StructuredPlannerWrapper:
    def __init__(self, schema: type, batches: list[Any]) -> None:
        self._schema = schema
        self._batches = batches
        self.call_count = 0

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # noqa: A002, ARG002
        batch = self._batches[min(self.call_count, len(self._batches) - 1)]
        self.call_count += 1
        if isinstance(batch, dict):
            return self._schema.model_validate(batch)
        return self._schema(calls=batch)


class _StructuredPlannerModel(AgentIntegrationModel):
    planned_batches: list[Any] = Field(default_factory=list)
    structured_wrapper: _StructuredPlannerWrapper | None = Field(default=None)
    bind_tools_calls: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> _StructuredPlannerModel:  # noqa: ARG002
        self.bind_tools_calls += 1
        return self

    def with_structured_output(self, schema: dict[str, Any] | type, **kwargs: Any) -> _StructuredPlannerWrapper:  # type: ignore[override]
        resolved_schema = schema if isinstance(schema, type) else schema["schema"]
        wrapper = _StructuredPlannerWrapper(resolved_schema, self.planned_batches)
        self.structured_wrapper = wrapper
        return wrapper


class _ScopedMockRetriever(MockRetriever):
    supports_metadata_scope = True

    @staticmethod
    def resolve_caller_scope(bundle: Mapping[str, Any]) -> MetadataScope | None:
        return MetadataScope.from_caller_scope(bundle)


def _state(query: str = "How do I restart pods?") -> KnowledgeAgentState:
    return KnowledgeAgentState(query=query, messages=[], iteration_count=0, findings=[])


@pytest.mark.unit
class TestStructuredQueryPlanning:
    @pytest.mark.asyncio
    async def test_structured_plan_converts_to_ai_message_tool_calls(self, entries: list[RetrieverEntry]) -> None:
        tools = create_retriever_tools(entries)
        model = _StructuredPlannerModel(
            planned_batches=[
                [{"tool_name": "search_elastic_runbooks", "query": "kubectl pod restart procedure"}],
            ]
        )
        config = KnowledgeAgentConfig(plan=PlanConfig(include_original_query=False, planning_strategy="structured"))
        node = PlanQueriesNode(tools=tools, default_model=model, agent_config=config)

        with patch.object(node, "_resolve_model_for_task", return_value=model):
            result = await node(_state(), config={})

        message = result["messages"][0]
        assert isinstance(message, AIMessage)
        assert message.tool_calls == [
            {
                "name": "search_elastic_runbooks",
                "args": {"query": "kubectl pod restart procedure"},
                "id": message.tool_calls[0]["id"],
                "type": "tool_call",
            }
        ]
        assert model.bind_tools_calls == 0

    @pytest.mark.asyncio
    async def test_legacy_tool_calls_structured_output_routes_to_tool_node(self, entries: list[RetrieverEntry]) -> None:
        tools = create_retriever_tools(entries)
        model = _StructuredPlannerModel(
            planned_batches=[
                {
                    "tool_calls": [
                        {
                            "name": "search_elastic_runbooks",
                            "args": {"query": "AP12363 server hostnames dependencies"},
                        }
                    ]
                }
            ]
        )
        config = KnowledgeAgentConfig(plan=PlanConfig(include_original_query=False, planning_strategy="structured"))
        node = PlanQueriesNode(tools=tools, default_model=model, agent_config=config)

        with patch.object(node, "_resolve_model_for_task", return_value=model):
            result = await node(_state("AP12363 server hostnames dependencies"), config={})

        message = result["messages"][0]
        assert message.tool_calls == [
            {
                "name": "search_elastic_runbooks",
                "args": {"query": "AP12363 server hostnames dependencies"},
                "id": message.tool_calls[0]["id"],
                "type": "tool_call",
            }
        ]
        assert route_after_plan(KnowledgeAgentState(messages=[message])) == "tool_node"

    @pytest.mark.asyncio
    async def test_invalid_tool_name_retries_then_uses_corrected_plan(self, entries: list[RetrieverEntry]) -> None:
        tools = create_retriever_tools(entries)
        model = _StructuredPlannerModel(
            planned_batches=[
                [{"tool_name": "search_missing", "query": "first try"}],
                [{"tool_name": "search_lightrag_architecture", "query": "architecture restart dependency"}],
            ]
        )
        node = PlanQueriesNode(
            tools=tools,
            default_model=model,
            agent_config=KnowledgeAgentConfig(plan=PlanConfig(include_original_query=False, planning_strategy="structured")),
        )

        with patch.object(node, "_resolve_model_for_task", return_value=model):
            result = await node(_state(), config={})

        assert model.structured_wrapper is not None
        assert model.structured_wrapper.call_count == 2
        assert result["messages"][0].tool_calls[0]["name"] == "search_lightrag_architecture"

    @pytest.mark.asyncio
    async def test_retry_exhaustion_filters_invalid_calls(self, entries: list[RetrieverEntry]) -> None:
        tools = create_retriever_tools(entries)
        model = _StructuredPlannerModel(
            planned_batches=[
                [{"tool_name": "search_missing", "query": "bad"}],
                [
                    {"tool_name": "search_missing", "query": "still bad"},
                    {"tool_name": "search_elastic_runbooks", "query": "valid fallback"},
                ],
            ]
        )
        node = PlanQueriesNode(
            tools=tools,
            default_model=model,
            agent_config=KnowledgeAgentConfig(plan=PlanConfig(include_original_query=False, planning_strategy="structured")),
        )

        with patch.object(node, "_resolve_model_for_task", return_value=model):
            result = await node(_state(), config={})

        calls = result["messages"][0].tool_calls
        assert [call["name"] for call in calls] == ["search_elastic_runbooks"]
        assert calls[0]["args"] == {"query": "valid fallback"}

    @pytest.mark.asyncio
    async def test_no_valid_calls_emits_plan_failed_fallback(self, entries: list[RetrieverEntry]) -> None:
        # All calls invalid through retry exhaustion AND no direct_response →
        # the node must not emit a blank message (EN9). It substitutes the
        # fallback content and flags plan_failed so OutputNode can surface it.
        tools = create_retriever_tools(entries)
        model = _StructuredPlannerModel(
            planned_batches=[
                [{"tool_name": "search_missing", "query": "bad"}],
                [{"tool_name": "search_missing", "query": "still bad"}],
            ]
        )
        node = PlanQueriesNode(
            tools=tools,
            default_model=model,
            agent_config=KnowledgeAgentConfig(plan=PlanConfig(include_original_query=False, planning_strategy="structured")),
        )

        with patch.object(node, "_resolve_model_for_task", return_value=model):
            result = await node(_state(), config={})

        message = result["messages"][0]
        assert message.tool_calls == []
        assert message.content == PLAN_FAILED_MESSAGE
        assert result["plan_failed"] is True
        assert route_after_plan(KnowledgeAgentState(messages=[message])) == "output"

    @pytest.mark.asyncio
    async def test_query_cap_applies_after_anchor_injection(self, entries: list[RetrieverEntry]) -> None:
        tools = create_retriever_tools(entries)
        model = _StructuredPlannerModel(
            planned_batches=[
                [
                    {"tool_name": "search_elastic_runbooks", "query": "restart runbook"},
                    {"tool_name": "search_lightrag_architecture", "query": "restart architecture"},
                ],
            ]
        )
        config = KnowledgeAgentConfig(plan=PlanConfig(max_queries=2, include_original_query=True, planning_strategy="structured"))
        node = PlanQueriesNode(tools=tools, default_model=model, agent_config=config)

        with patch.object(node, "_resolve_model_for_task", return_value=model):
            result = await node(_state("Original restart question"), config={})

        assert len(result["messages"][0].tool_calls) == 2

    @pytest.mark.asyncio
    async def test_metadata_axes_are_preserved_for_search_general_doc_without_retry(self) -> None:
        entry = RetrieverEntry(
            name="general_doc",
            description="Search scoped documents",
            retriever=_ScopedMockRetriever(),
            metadata_scope=MetadataScopeConfig(exposed_axes=("apcode", "app_name", "entity")),
        )
        tools = create_retriever_tools([entry])
        model = _StructuredPlannerModel(
            planned_batches=[
                [
                    {
                        "tool_name": "search_general_doc",
                        "query": "AP90021 restart procedure",
                        "apcode": "AP90021",
                        "app_name": "Billing",
                        "entity": "payment-api",
                    }
                ],
            ]
        )
        node = PlanQueriesNode(
            tools=tools,
            default_model=model,
            agent_config=KnowledgeAgentConfig(plan=PlanConfig(include_original_query=False, planning_strategy="structured")),
        )

        with patch.object(node, "_resolve_model_for_task", return_value=model):
            result = await node(_state("How does AP90021 restart?"), config={})

        assert model.structured_wrapper is not None
        assert model.structured_wrapper.call_count == 1
        assert result["messages"][0].tool_calls[0]["name"] == "search_general_doc"
        assert result["messages"][0].tool_calls[0]["args"] == {
            "query": "AP90021 restart procedure",
            "apcode": "AP90021",
            "app_name": "Billing",
            "entity": "payment-api",
        }

-------

tests/test_ai_engine/agents/knowledge_agent/test_retriever_entry_scope_fields.py
----
"""Tests for the metadata-scope plumbing on :class:`RetrieverEntry`.

The build-time filter ceiling, per-axis runtime-query opt-in, and optional
vocab resolver are grouped under a single :class:`MetadataScopeConfig`
exposed via ``entry.metadata_scope``. ``None`` (the default) means the
entry is not scope-aware so existing callers don't opt in by accident.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sta_agent_core.repositories.retrievers.elasticsearch.metadata_scope import MetadataScope
from sta_agent_core.repositories.retrievers.metadata_value_resolver import (
    MetadataValueResolver,
    Resolution,
)
from sta_agent_core.repositories.retrievers.mock import MockRetriever
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_types import (
    MetadataScopeConfig,
    RetrieverEntry,
)


class _FakeResolver:
    async def resolve(self, axis: str, value: str) -> Resolution:  # noqa: ARG002
        return Resolution(matched=True, value=value, target_axis_boost=f"{axis}_boost", warning=None, score=100.0, suggestions=[])

    async def refresh(self) -> None:
        return None


class TestEntryDefaults:
    def test_metadata_scope_is_none_by_default(self) -> None:
        entry = RetrieverEntry(name="x", description="y", retriever=MockRetriever(name="x"))
        assert entry.metadata_scope is None


class TestEntryScopeFields:
    def test_default_scope_round_trips(self) -> None:
        scope = MetadataScope(apcode_filter="BCEF")
        entry = RetrieverEntry(
            name="x",
            description="y",
            retriever=MockRetriever(name="x"),
            metadata_scope=MetadataScopeConfig(default=scope),
        )
        assert entry.metadata_scope is not None
        assert entry.metadata_scope.default is scope
        assert isinstance(entry.metadata_scope.default, MetadataScope)
        assert entry.metadata_scope.default.apcode_filter == "BCEF"

    def test_exposed_axes_per_axis_subset(self) -> None:
        entry = RetrieverEntry(
            name="x",
            description="y",
            retriever=MockRetriever(name="x"),
            metadata_scope=MetadataScopeConfig(exposed_axes=("apcode",)),  # not "app_name" or "entity"
        )
        assert entry.metadata_scope is not None
        assert entry.metadata_scope.exposed_axes == ("apcode",)

    def test_resolver_attaches_and_satisfies_protocol(self) -> None:
        resolver = _FakeResolver()
        entry = RetrieverEntry(
            name="x",
            description="y",
            retriever=MockRetriever(name="x"),
            metadata_scope=MetadataScopeConfig(value_resolver=resolver),
        )
        assert entry.metadata_scope is not None
        assert isinstance(entry.metadata_scope.value_resolver, MetadataValueResolver)


class TestCreateElasticEntryWiresFields:
    """``create_elastic_entry`` keeps flat scope kwargs and packs them into the config."""

    def test_kwargs_round_trip_via_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sta_agent_engine import retrievers as engine_retrievers
        from sta_agent_engine.agents.knowledge_agent import knowledge_agent_retrievers as mod

        scope = MetadataScope(apcode_filter="BCEF")
        resolver = _FakeResolver()
        # The factory imports build_elastic_retriever_from_env lazily from
        # sta_agent_engine.retrievers — patch it at the source module so the
        # lazy import inside create_elastic_entry resolves to our stub.
        monkeypatch.setattr(engine_retrievers, "build_elastic_retriever_from_env", lambda **kw: MagicMock(name="es_retriever_stub"))

        entry = mod.create_elastic_entry(
            name="elastic_runbooks",
            description="d",
            default_scope=scope,
            expose_metadata_args=["apcode", "entity"],
            metadata_value_resolver=resolver,
        )
        assert entry.metadata_scope is not None
        assert entry.metadata_scope.default is scope
        assert entry.metadata_scope.exposed_axes == ("apcode", "entity")
        assert entry.metadata_scope.value_resolver is resolver

    def test_no_scope_kwargs_leaves_metadata_scope_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sta_agent_engine import retrievers as engine_retrievers
        from sta_agent_engine.agents.knowledge_agent import knowledge_agent_retrievers as mod

        monkeypatch.setattr(engine_retrievers, "build_elastic_retriever_from_env", lambda **kw: MagicMock(name="es_retriever_stub"))

        entry = mod.create_elastic_entry(name="elastic_runbooks", description="d")
        assert entry.metadata_scope is None

-------

tests/test_ai_engine/agents/knowledge_agent/test_retriever_tool_factory_doc_filter.py
----
"""Tests for the orchestrator-supplied per-call metadata scope in the tool factory.

The Knowledge Agent reads ``ka_metadata_scope`` (an orchestrator-set, run-scoped
state input — NOT an LLM tool arg) and threads it into retrieval as a
FILTER-ONLY :class:`MetadataScope` that NARROWS the build-time ceiling:

- ``doc_ids`` → ``MetadataScope.doc_filter`` (hard document-id terms filter),
  carried INSIDE the scope (no separate ``documents=`` kwarg for caller ids).
- ``apcode`` / ``app_name`` / ``entity`` → the matching ``*_filter`` axis (HARD
  AND-filters, never a boost). This is the key correctness pin.
- A ``*_boost`` key in the payload is dropped + warned (never reaches retrieval).
- Caller scope is applied ONLY when the entry opts in via
  ``accepts_caller_scope=True`` AND the retriever honors ``metadata_scope=...``.
- The planner-visible tool schema carries NO scope fields — state-driven.
- A doc-filter that matches zero documents logs a warning (the page-id vs
  chunk-id namespace mismatch is the main silent-failure mode).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest

from sta_agent_core.repositories.retrievers.elasticsearch.metadata_scope import MetadataScope
from sta_agent_core.repositories.retrievers.lightrag import LightRAGMetadataScope
from sta_agent_core.repositories.retrievers.mock import MockRetriever
from sta_agent_core.repositories.retrievers.scope_capability import MetadataScopeLike
from sta_agent_core.repositories.retrievers.search_response import SearchResponse
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_types import RetrieverEntry
from sta_agent_engine.agents.knowledge_agent.tools.retriever_tool_factory import (
    _read_caller_scope_bundle,
    _retriever_supports_documents,
    create_retriever_tool,
)


class _ScopeRetriever:
    """Retriever double honoring ``SupportsMetadataScope``.

    Records the ``metadata_scope`` kwarg its ``search`` receives so tests can
    assert the caller scope narrowed the build-time ceiling. The real method is
    kept (not swapped for an ``AsyncMock``) so signature-based capability
    detection stays intact.
    """

    supports_metadata_scope = True

    @staticmethod
    def resolve_caller_scope(bundle: Mapping[str, Any]) -> MetadataScopeLike | None:
        return MetadataScope.from_caller_scope(bundle)

    def __init__(self, name: str = "elastic_runbooks", num_results: int = 3) -> None:
        self.name = name
        self._num_results = num_results
        self.last_kwargs: dict[str, Any] = {}

    async def search(
        self,
        query: str,
        size: int = 10,
        **kwargs: Any,
    ) -> SearchResponse[Any]:
        # Record exactly the kwargs the factory forwarded so a test can assert
        # that ``metadata_scope`` was (or was not) passed at all.
        self.last_kwargs = dict(kwargs)
        return SearchResponse(results=[object()] * self._num_results)

    async def close(self) -> None:
        return None


class _LightRAGScopeRetriever(_ScopeRetriever):
    @staticmethod
    def resolve_caller_scope(bundle: Mapping[str, Any]) -> LightRAGMetadataScope | None:
        return LightRAGMetadataScope.from_caller_scope(bundle)


class _DocFilterRetriever:
    """Retriever double whose ``search`` declares an explicit ``documents`` param.

    Used only to pin that ``_retriever_supports_documents`` still detects a
    ``documents`` parameter — caller-injected ids no longer use that path.
    """

    supports_metadata_scope = True

    @staticmethod
    def resolve_caller_scope(bundle: Mapping[str, Any]) -> MetadataScope | None:
        return MetadataScope.from_caller_scope(bundle)

    def __init__(self, name: str = "elastic_runbooks", num_results: int = 3) -> None:
        self.name = name
        self._num_results = num_results
        self.last_kwargs: dict[str, Any] = {}

    async def search(
        self,
        query: str,
        size: int = 10,
        *,
        documents: list[str] | None = None,
        metadata_scope: MetadataScope | None = None,
        **kwargs: Any,
    ) -> SearchResponse[Any]:
        self.last_kwargs = {"documents": documents, "metadata_scope": metadata_scope, **kwargs}
        return SearchResponse(results=[object()] * self._num_results)

    async def close(self) -> None:
        return None


def _runtime(ka_metadata_scope: dict[str, Any] | None = None) -> MagicMock:
    rt = MagicMock()
    rt.tool_call_id = "tc_test"
    state: dict[str, Any] = {"messages": []}
    if ka_metadata_scope is not None:
        state["ka_metadata_scope"] = ka_metadata_scope
    rt.state = state
    return rt


def _entry(retriever: Any, *, name: str = "elastic_runbooks", accepts_caller_scope: bool = True) -> RetrieverEntry:
    return RetrieverEntry(
        name=name,
        description="Search runbooks",
        retriever=retriever,
        accepts_caller_scope=accepts_caller_scope,
    )


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


def test_documents_capability_detection() -> None:
    assert _retriever_supports_documents(_DocFilterRetriever()) is True
    # MockRetriever only has **kwargs — must NOT count as support.
    assert _retriever_supports_documents(MockRetriever(name="mock")) is False


def test_read_metadata_scope_normalizes() -> None:
    rt = _runtime({"doc_ids": ["  a ", "a", "", "b"], "apcode": "AP90021"})
    bundle = _read_caller_scope_bundle(rt)
    assert bundle == {"doc_ids": ["a", "b"], "apcode": ["AP90021"]}


def test_read_metadata_scope_absent_returns_none() -> None:
    assert _read_caller_scope_bundle(_runtime()) is None
    assert _read_caller_scope_bundle(_runtime({})) is None


def test_read_metadata_scope_all_four_axes() -> None:
    rt = _runtime({"doc_ids": ["d1"], "apcode": ["AP1"], "app_name": ["billing"], "entity": ["e1"]})
    assert _read_caller_scope_bundle(rt) == {
        "doc_ids": ["d1"],
        "apcode": ["AP1"],
        "app_name": ["billing"],
        "entity": ["e1"],
    }


def test_read_metadata_scope_resolves_only_lightrag_axis() -> None:
    bundle = _read_caller_scope_bundle(_runtime({"doc_ids": ["d1"], "tag_filter": {"private_operator": ["tag-a"]}}))
    assert bundle is not None
    scope = _LightRAGScopeRetriever.resolve_caller_scope(bundle)

    assert isinstance(scope, LightRAGMetadataScope)
    assert scope.tag_filter == {"private_operator": ["tag-a"]}


# ---------------------------------------------------------------------------
# Caller scope narrows the build-time ceiling; rides inside metadata_scope
# ---------------------------------------------------------------------------


class TestCallerScopeReachesSearch:
    @pytest.mark.asyncio
    async def test_doc_ids_ride_inside_metadata_scope(self) -> None:
        retriever = _ScopeRetriever()
        tool = create_retriever_tool(_entry(retriever))

        await tool.coroutine(query="how to deploy", runtime=_runtime({"doc_ids": ["doc-1", "doc-2"]}))  # type: ignore[arg-type]

        scope: MetadataScope = retriever.last_kwargs["metadata_scope"]
        assert scope.doc_filter == ["doc-1", "doc-2"]
        # Caller ids no longer travel via a separate ``documents=`` kwarg.
        assert "documents" not in retriever.last_kwargs

    @pytest.mark.asyncio
    async def test_no_scope_omits_metadata_scope_kwarg(self) -> None:
        retriever = _ScopeRetriever()
        tool = create_retriever_tool(_entry(retriever))

        await tool.coroutine(query="q", runtime=_runtime())  # type: ignore[arg-type]

        assert "metadata_scope" not in retriever.last_kwargs

    @pytest.mark.asyncio
    async def test_lightrag_receives_only_tag_filter_from_mixed_bundle(self) -> None:
        retriever = _LightRAGScopeRetriever(name="lightrag")
        tool = create_retriever_tool(_entry(retriever, name="lightrag"))

        await tool.coroutine(  # type: ignore[arg-type]
            query="q",
            runtime=_runtime({"doc_ids": ["d1"], "tag_filter": {"private_operator": ["tag-a"]}}),
        )

        scope = retriever.last_kwargs["metadata_scope"]
        assert isinstance(scope, LightRAGMetadataScope)
        assert scope.tag_filter == {"private_operator": ["tag-a"]}

    @pytest.mark.asyncio
    async def test_bound_retriever_warns_when_bundle_has_no_recognized_filter(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _LightRAGScopeRetriever(name="lightrag")
        tool = create_retriever_tool(_entry(retriever, name="lightrag"))

        with caplog.at_level(logging.WARNING):
            await tool.coroutine(query="q", runtime=_runtime({"doc_ids": ["d1"]}))  # type: ignore[arg-type]

        assert "metadata_scope" not in retriever.last_kwargs
        assert any("recognizes no caller filters" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_multi_axis_scope_tightens_all_four(self) -> None:
        retriever = _ScopeRetriever()
        tool = create_retriever_tool(_entry(retriever))

        await tool.coroutine(  # type: ignore[arg-type]
            query="q",
            runtime=_runtime({"doc_ids": ["d1"], "apcode": "AP90021", "app_name": "billing", "entity": "e1"}),
        )

        scope: MetadataScope = retriever.last_kwargs["metadata_scope"]
        assert scope.doc_filter == ["d1"]
        assert scope.apcode_filter == ["AP90021"]
        assert scope.app_name_filter == ["billing"]
        assert scope.entity_filter == ["e1"]
        # All hard filters — no boost axis is ever set from caller scope.
        assert scope.apcode_boost is None
        assert scope.app_name_boost is None
        assert scope.entity_boost is None

    @pytest.mark.asyncio
    async def test_boost_key_in_scope_is_dropped_and_warned(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _ScopeRetriever()
        tool = create_retriever_tool(_entry(retriever))

        with caplog.at_level(logging.WARNING):
            await tool.coroutine(  # type: ignore[arg-type]
                query="q",
                runtime=_runtime({"apcode": "AP90021", "apcode_boost": "AP90021"}),
            )

        scope: MetadataScope = retriever.last_kwargs["metadata_scope"]
        # The smuggled boost never reaches the retriever.
        assert scope.apcode_filter == ["AP90021"]
        assert scope.apcode_boost is None
        assert any("dropped" in r.message and "boost" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Per-entry opt-in
# ---------------------------------------------------------------------------


class TestAcceptsCallerScopeGate:
    @pytest.mark.asyncio
    async def test_opt_out_entry_ignores_caller_scope(self) -> None:
        retriever = _ScopeRetriever()
        tool = create_retriever_tool(_entry(retriever, accepts_caller_scope=False))

        await tool.coroutine(query="q", runtime=_runtime({"doc_ids": ["d1"], "apcode": "AP1"}))  # type: ignore[arg-type]

        # Entry did not opt in — caller scope ignored, no metadata_scope forwarded.
        assert "metadata_scope" not in retriever.last_kwargs

    @pytest.mark.asyncio
    async def test_opt_in_entry_applies_caller_scope(self) -> None:
        retriever = _ScopeRetriever()
        tool = create_retriever_tool(_entry(retriever, accepts_caller_scope=True))

        await tool.coroutine(query="q", runtime=_runtime({"apcode": "AP1"}))  # type: ignore[arg-type]

        scope: MetadataScope = retriever.last_kwargs["metadata_scope"]
        assert scope.apcode_filter == ["AP1"]

    @pytest.mark.asyncio
    async def test_unsupported_retriever_skips_scope_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        # MockRetriever does not honor metadata_scope — caller scope skipped + warned.
        retriever = MockRetriever(name="mock", num_results=2)
        tool = create_retriever_tool(_entry(retriever, name="mock", accepts_caller_scope=True))

        with caplog.at_level(logging.WARNING):
            await tool.coroutine(query="q", runtime=_runtime({"doc_ids": ["doc-1"]}))  # type: ignore[arg-type]

        assert any("does not honor metadata_scope" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# apcode → HARD apcode_filter, never apcode_boost
# ---------------------------------------------------------------------------


class TestApcodeHardFilter:
    @pytest.mark.asyncio
    async def test_apcode_reaches_apcode_filter_not_boost(self) -> None:
        retriever = _ScopeRetriever()
        tool = create_retriever_tool(_entry(retriever))

        await tool.coroutine(query="q", runtime=_runtime({"apcode": "AP90021"}))  # type: ignore[arg-type]

        scope: MetadataScope = retriever.last_kwargs["metadata_scope"]
        # HARD filter — the key correctness pin. apcode normalizes to a list.
        assert scope.apcode_filter == ["AP90021"]
        # NEVER a soft boost.
        assert scope.apcode_boost is None


# ---------------------------------------------------------------------------
# Schema is state-driven, not LLM-exposed
# ---------------------------------------------------------------------------


def test_tool_schema_has_no_scope_fields() -> None:
    tool = create_retriever_tool(_entry(_ScopeRetriever()))
    assert tool.args_schema is not None
    fields = set(tool.args_schema.model_fields)  # type: ignore[union-attr]
    assert "documents" not in fields
    assert "ka_metadata_scope" not in fields
    # LLM-visible projection is exactly the query.
    assert set(tool.args.keys()) == {"query"}


# ---------------------------------------------------------------------------
# Zero-hit warning
# ---------------------------------------------------------------------------


class TestZeroHitWarning:
    @pytest.mark.asyncio
    async def test_doc_filter_zero_hits_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _ScopeRetriever(num_results=0)  # zero hits
        tool = create_retriever_tool(_entry(retriever))

        with caplog.at_level(logging.WARNING):
            await tool.coroutine(query="q", runtime=_runtime({"doc_ids": ["doc-x"]}))  # type: ignore[arg-type]

        assert any("matched ZERO documents" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_doc_filter_with_hits_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _ScopeRetriever(num_results=1)
        tool = create_retriever_tool(_entry(retriever))

        with caplog.at_level(logging.WARNING):
            await tool.coroutine(query="q", runtime=_runtime({"doc_ids": ["doc-1"]}))  # type: ignore[arg-type]

        assert not any("matched ZERO documents" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_zero_hits_without_doc_filter_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _ScopeRetriever(num_results=0)
        tool = create_retriever_tool(_entry(retriever))

        with caplog.at_level(logging.WARNING):
            await tool.coroutine(query="q", runtime=_runtime())  # type: ignore[arg-type]

        assert not any("matched ZERO documents" in r.message for r in caplog.records)

-------

tests/test_ai_engine/agents/knowledge_agent/test_retriever_tool_factory_layer3.py
----
"""Tests for the runtime query scope inside the retriever tool factory.

Pins:
- Dynamic args_schema: query + per-axis Optional[str] when
  ``entry.metadata_scope.exposed_axes`` is set.
- LLM-emitted axes routed through ``entry.metadata_scope.value_resolver``.
- Resolver hits unioned into the entry's ``metadata_scope.default`` via
  ``add_boosts`` (boost-only — the build-time filter ceiling is never
  touched).
- Warnings appended to the ToolMessage tail AND written into state under
  ``resolution_warnings`` so OutputNode can mirror them to the result.
- Structured ``logger.warning("metadata_resolution", extra={…})`` for
  observability.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sta_agent_core.repositories.retrievers.elasticsearch.metadata_scope import MetadataScope
from sta_agent_core.repositories.retrievers.metadata_value_resolver import Resolution
from sta_agent_core.repositories.retrievers.mock import MockRetriever
from sta_agent_engine.agents.knowledge_agent.knowledge_agent_types import (
    MetadataScopeConfig,
    RetrieverEntry,
)
from sta_agent_engine.agents.knowledge_agent.tools.retriever_tool_factory import (
    create_retriever_tool,
)


def _scope_config(
    *,
    default_scope: MetadataScope | None = None,
    expose: list[str] | None = None,
    resolver: Any | None = None,
) -> MetadataScopeConfig | None:
    """Pack legacy flat kwargs into a :class:`MetadataScopeConfig` for test brevity."""
    if default_scope is None and not expose and resolver is None:
        return None
    return MetadataScopeConfig(
        default=default_scope,
        exposed_axes=tuple(expose) if expose else (),
        value_resolver=resolver,
    )


class _StubResolver:
    """Resolver double — programs canned Resolution per (axis, raw)."""

    def __init__(self, table: dict[tuple[str, str], Resolution]) -> None:
        self._table = table
        self.calls: list[tuple[str, str]] = []

    async def resolve(self, axis: str, value: str) -> Resolution:
        self.calls.append((axis, value))
        key = (axis, value)
        if key in self._table:
            return self._table[key]
        return Resolution(
            matched=False,
            value=None,
            target_axis_boost=None,
            warning=f"axis={axis} value={value!r} unknown to test stub",
            score=None,
            suggestions=[],
        )

    async def refresh(self) -> None:
        return None


def _runtime() -> MagicMock:
    rt = MagicMock()
    rt.tool_call_id = "tc_test"
    rt.state = {"messages": []}
    return rt


class _ScopedMockRetriever(MockRetriever):
    """``MockRetriever`` that opts into ``SupportsMetadataScope``.

    The plain ``MockRetriever`` ignores all kwargs, so it cannot honestly
    claim scope support — that's the trust-boundary failure mode Group D
    fixes. Tests that need the tool factory to wire scope features through
    a Mock-backed entry use this subclass; tests that exercise the gate
    itself (Group D's ``TestScopeCapabilityCheck``) use plain ``MockRetriever``.
    """

    supports_metadata_scope = True

    @staticmethod
    def resolve_caller_scope(bundle: Mapping[str, Any]) -> MetadataScope | None:
        return MetadataScope.from_caller_scope(bundle)


def _make_entry(
    *,
    name: str = "elastic_runbooks",
    expose: list[str] | None = None,
    resolver: Any | None = None,
    default_scope: MetadataScope | None = None,
) -> RetrieverEntry:
    retriever = _ScopedMockRetriever(name=name, num_results=3)
    return RetrieverEntry(
        name=name,
        description="Search runbooks",
        retriever=retriever,
        metadata_scope=_scope_config(default_scope=default_scope, expose=expose, resolver=resolver),
    )


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


class TestSchemaGeneration:
    def test_no_expose_no_args_schema(self) -> None:
        entry = _make_entry()
        tool = create_retriever_tool(entry)
        # No exposed axes → the rewritten signature is ``query`` +
        # ``runtime: ToolRuntime``. ``runtime`` stays in
        # ``args_schema.model_fields`` (langchain still introspects it) but
        # is stripped from ``tool.args`` (the LLM-visible projection).
        # ``runtime_scope_args`` (the opaque ``**kwargs`` object) MUST be
        # absent from both — otherwise the planner sees a dead surface and
        # any keys it emits are silently dropped because the resolver branch
        # in ``_search`` is gated on ``exposed_axes``.
        assert tool.args_schema is not None
        schema_fields = set(tool.args_schema.model_fields)  # type: ignore[union-attr]
        assert "runtime_scope_args" not in schema_fields, (
            f"`runtime_scope_args` leaked back into args_schema.model_fields: {schema_fields}. This re-introduces the dead-surface bug."
        )
        assert "apcode" not in schema_fields
        assert "app_name" not in schema_fields
        assert "entity" not in schema_fields
        # ``tool.args`` is the LLM-visible projection (excludes injected
        # ``runtime``). Must be exactly ``{"query"}``.
        assert set(tool.args.keys()) == {"query"}, f"Expected tool.args == {{'query'}} when no expose_metadata_args, got {set(tool.args.keys())}"

    def test_expose_apcode_only_adds_one_axis(self) -> None:
        entry = _make_entry(expose=["apcode"], resolver=_StubResolver({}))
        tool = create_retriever_tool(entry)
        assert tool.args_schema is not None
        fields = set(tool.args_schema.model_fields)  # type: ignore[union-attr]
        assert "query" in fields
        assert "apcode" in fields
        assert "app_name" not in fields
        assert "entity" not in fields

    def test_expose_all_three_axes(self) -> None:
        entry = _make_entry(expose=["apcode", "app_name", "entity"], resolver=_StubResolver({}))
        tool = create_retriever_tool(entry)
        assert tool.args_schema is not None
        fields = set(tool.args_schema.model_fields)  # type: ignore[union-attr]
        assert {"query", "apcode", "app_name", "entity"} <= fields

    def test_expose_without_resolver_builds_passthrough_tool(self) -> None:
        # metadata_value_resolver is opt-in — when absent, the tool still builds
        # and raw LLM-emitted axis values flow straight into MetadataScope.add_boosts.
        entry = _make_entry(expose=["apcode"], resolver=None)
        tool = create_retriever_tool(entry)
        assert tool.args_schema is not None
        fields = set(tool.args_schema.model_fields)  # type: ignore[union-attr]
        assert "apcode" in fields

    def test_unknown_axis_in_expose_raises(self) -> None:
        entry = _make_entry(expose=["apcode", "garbage"], resolver=_StubResolver({}))
        with pytest.raises(ValueError, match="unknown axes"):
            create_retriever_tool(entry)


# ---------------------------------------------------------------------------
# Resolution → boost merge
# ---------------------------------------------------------------------------


class TestRuntimeScopeBoostMerge:
    @pytest.mark.asyncio
    async def test_resolver_match_unioned_as_boost_into_search_kwargs(self) -> None:
        resolver = _StubResolver(
            {
                ("apcode", "BCEF"): Resolution(
                    matched=True, value="BCEF", target_axis_boost="apcode_boost", warning=None, score=100.0, suggestions=[]
                ),
            }
        )
        entry = _make_entry(expose=["apcode"], resolver=resolver)
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 3))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        await tool.coroutine(query="how to deploy", apcode="BCEF", runtime=_runtime())  # type: ignore[arg-type]
        kwargs = retriever_search.call_args.kwargs
        scope: MetadataScope = kwargs["metadata_scope"]
        assert scope.apcode_boost == "BCEF"
        # Runtime query scope must NEVER write filters
        assert scope.apcode_filter is None

    @pytest.mark.asyncio
    async def test_default_scope_filters_preserved_under_runtime_boosts(self) -> None:
        """Build-time filter ceiling stays put even when runtime query scope fires."""
        resolver = _StubResolver(
            {
                ("apcode", "BCEF"): Resolution(
                    matched=True, value="BCEF", target_axis_boost="apcode_boost", warning=None, score=100.0, suggestions=[]
                ),
            }
        )
        entry = _make_entry(
            expose=["apcode"],
            resolver=resolver,
            default_scope=MetadataScope(entity_filter="L1_ENTITY"),
        )
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        await tool.coroutine(query="q", apcode="BCEF", runtime=_runtime())  # type: ignore[arg-type]
        scope: MetadataScope = retriever_search.call_args.kwargs["metadata_scope"]
        assert scope.entity_filter == "L1_ENTITY"
        assert scope.apcode_boost == "BCEF"

    @pytest.mark.asyncio
    async def test_unmatched_resolution_does_not_touch_scope(self) -> None:
        resolver = _StubResolver(
            {
                ("apcode", "QQQ"): Resolution(
                    matched=False, value=None, target_axis_boost=None, warning="apcode='QQQ' not found", score=None, suggestions=[]
                ),
            }
        )
        entry = _make_entry(expose=["apcode"], resolver=resolver)
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        await tool.coroutine(query="q", apcode="QQQ", runtime=_runtime())  # type: ignore[arg-type]
        kwargs = retriever_search.call_args.kwargs
        # Empty scope → not passed as metadata_scope
        assert "metadata_scope" not in kwargs

    @pytest.mark.asyncio
    async def test_omitted_axis_skips_resolver_call(self) -> None:
        resolver = _StubResolver({})
        entry = _make_entry(expose=["apcode", "entity"], resolver=resolver)
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        await tool.coroutine(query="q", apcode=None, entity=None, runtime=_runtime())  # type: ignore[arg-type]
        assert resolver.calls == []  # neither axis was provided

    @pytest.mark.asyncio
    async def test_adversarial_resolver_cannot_widen_entity_filter(self) -> None:
        """A resolver returning ``target_axis_boost='entity_filter'`` (a filter
        axis, not a boost axis) MUST NOT widen the build-time filter ceiling.

        Pinning the strongest scope-widening path: even if the resolver is
        compromised or buggy, the build-time ``default_scope.entity_filter``
        must stay exactly as configured. The factory's ``_valid_resolution``
        gate rejects any non-boost ``target_axis_boost`` before
        ``MetadataScope.add_boosts`` runs.
        """
        resolver = _StubResolver(
            {
                # Adversarial shape — resolver claims this should land on a
                # FILTER axis instead of a boost. Factory must refuse.
                ("entity", "EVIL"): Resolution(
                    matched=True,
                    value="E_ACME",
                    target_axis_boost="entity_filter",  # type: ignore[arg-type]
                    warning=None,
                    score=100.0,
                    suggestions=[],
                ),
            }
        )
        entry = _make_entry(
            expose=["entity"],
            resolver=resolver,
            default_scope=MetadataScope(entity_filter="E_DU"),
        )
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        cmd = await tool.coroutine(query="q", entity="EVIL", runtime=_runtime())  # type: ignore[arg-type]
        scope: MetadataScope = retriever_search.call_args.kwargs["metadata_scope"]
        # Build-time filter is untouched — still the configured value.
        assert scope.entity_filter == "E_DU"
        # Boost axes are NOT widened either — the bad resolution was rejected.
        assert scope.entity_boost is None
        # A warning surfaces so operators can see the rejection.
        warnings = cmd.update.get("resolution_warnings", [])
        assert any("entity_filter" in w or "invalid Resolution" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# Passthrough (no resolver) → raw values flow into boost axes
# ---------------------------------------------------------------------------


class TestRuntimeScopePassthrough:
    """Resolver is opt-in. When absent, raw LLM-emitted axis values must land
    on the matching ``*_boost`` axis verbatim — never on ``*_filter``, never
    widen the build-time ``default_scope`` ceiling."""

    @pytest.mark.asyncio
    async def test_passthrough_axis_lands_on_boost_not_filter(self) -> None:
        entry = _make_entry(expose=["apcode"], resolver=None)
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        await tool.coroutine(query="q", apcode="BCEF", runtime=_runtime())  # type: ignore[arg-type]
        kwargs = retriever_search.call_args.kwargs
        scope: MetadataScope = kwargs["metadata_scope"]
        assert scope.apcode_boost == "BCEF"
        assert scope.apcode_filter is None

    @pytest.mark.asyncio
    async def test_passthrough_preserves_default_scope_filter_ceiling(self) -> None:
        entry = _make_entry(
            expose=["apcode"],
            resolver=None,
            default_scope=MetadataScope(entity_filter="L1_ENTITY"),
        )
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        await tool.coroutine(query="q", apcode="BCEF", runtime=_runtime())  # type: ignore[arg-type]
        scope: MetadataScope = retriever_search.call_args.kwargs["metadata_scope"]
        # Build-time filter still there
        assert scope.entity_filter == "L1_ENTITY"
        # Boost added
        assert scope.apcode_boost == "BCEF"
        # No filter widening
        assert scope.apcode_filter is None

    @pytest.mark.asyncio
    async def test_passthrough_skips_blank_and_whitespace_values(self) -> None:
        entry = _make_entry(expose=["apcode", "app_name", "entity"], resolver=None)
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        await tool.coroutine(query="q", apcode="", app_name="   ", entity=None, runtime=_runtime())  # type: ignore[arg-type]
        kwargs = retriever_search.call_args.kwargs
        # No axis was a usable value → no metadata_scope written
        assert "metadata_scope" not in kwargs

    @pytest.mark.asyncio
    async def test_passthrough_strips_whitespace_around_value(self) -> None:
        entry = _make_entry(expose=["apcode"], resolver=None)
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        await tool.coroutine(query="q", apcode="  BCEF  ", runtime=_runtime())  # type: ignore[arg-type]
        scope: MetadataScope = retriever_search.call_args.kwargs["metadata_scope"]
        assert scope.apcode_boost == "BCEF"

    @pytest.mark.asyncio
    async def test_passthrough_entity_axis_does_not_overwrite_entity_filter(self) -> None:
        """Dangerous-mode regression: a runtime ``entity`` arg passed via
        passthrough must NEVER land on ``entity_filter`` (the build-time
        ceiling axis) — it must land only on ``entity_boost``.

        Mirrors the tenant-isolation scenario: ``default_scope.entity_filter``
        names the allowed tenants; the planner emits a runtime ``entity``
        arg; the boost axis must take it, the filter must stay frozen.
        """
        entry = _make_entry(
            expose=["entity"],
            resolver=None,
            default_scope=MetadataScope(entity_filter=["E_DU_MAIN", "E_DU_FOUNDATION"]),
        )
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        await tool.coroutine(query="q", entity="E_ACME", runtime=_runtime())  # type: ignore[arg-type]
        scope: MetadataScope = retriever_search.call_args.kwargs["metadata_scope"]
        # Build-time filter ceiling is unchanged — ACME cannot widen it.
        assert scope.entity_filter == ["E_DU_MAIN", "E_DU_FOUNDATION"]
        # Boost landed on the BOOST axis, not the filter axis.
        assert scope.entity_boost == "E_ACME"


# ---------------------------------------------------------------------------
# Warnings + structured logging
# ---------------------------------------------------------------------------


class TestWarningsPlumbing:
    @pytest.mark.asyncio
    async def test_warning_appears_in_tool_message_tail(self) -> None:
        resolver = _StubResolver(
            {
                ("apcode", "QQQ"): Resolution(
                    matched=False, value=None, target_axis_boost=None, warning="apcode='QQQ' not found", score=None, suggestions=[]
                ),
            }
        )
        entry = _make_entry(expose=["apcode"], resolver=resolver)
        entry.retriever.search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        cmd = await tool.coroutine(query="q", apcode="QQQ", runtime=_runtime())  # type: ignore[arg-type]
        tool_msg = cmd.update["messages"][0]
        assert "apcode='QQQ' not found" in tool_msg.content

    @pytest.mark.asyncio
    async def test_warnings_written_to_state(self) -> None:
        resolver = _StubResolver(
            {
                ("apcode", "QQQ"): Resolution(matched=False, value=None, target_axis_boost=None, warning="W1", score=None, suggestions=[]),
                ("entity", "ZZZ"): Resolution(matched=False, value=None, target_axis_boost=None, warning="W2", score=None, suggestions=[]),
            }
        )
        entry = _make_entry(expose=["apcode", "entity"], resolver=resolver)
        entry.retriever.search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        cmd = await tool.coroutine(  # type: ignore[arg-type]
            query="q", apcode="QQQ", entity="ZZZ", runtime=_runtime()
        )
        assert cmd.update["resolution_warnings"] == ["W1", "W2"]

    @pytest.mark.asyncio
    async def test_no_warnings_means_no_resolution_warnings_key(self) -> None:
        resolver = _StubResolver({})
        entry = _make_entry(expose=["apcode"], resolver=resolver)
        entry.retriever.search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        cmd = await tool.coroutine(query="q", runtime=_runtime())  # type: ignore[arg-type]
        assert "resolution_warnings" not in cmd.update

    @pytest.mark.asyncio
    async def test_structured_log_emitted_on_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        resolver = _StubResolver(
            {
                ("apcode", "QQQ"): Resolution(matched=False, value=None, target_axis_boost=None, warning="W", score=42.5, suggestions=[]),
            }
        )
        entry = _make_entry(expose=["apcode"], resolver=resolver)
        entry.retriever.search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)
        with caplog.at_level("WARNING"):
            await tool.coroutine(query="q", apcode="QQQ", runtime=_runtime())  # type: ignore[arg-type]
        assert any("metadata_resolution" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Defensive contract — a buggy third-party resolver must NOT crash the tool
# ---------------------------------------------------------------------------


class _RaisingResolver:
    """Resolver double that raises from `resolve` (simulates a buggy backend)."""

    async def resolve(self, axis: str, value: str) -> Resolution:
        raise RuntimeError(f"boom on {axis}={value!r}")

    async def refresh(self) -> None:
        return None


class TestInvalidResolutionContainment:
    @pytest.mark.asyncio
    async def test_filter_axis_target_does_not_raise_and_drops_boost(self) -> None:
        """A resolver pointing at a filter axis must not crash the tool."""
        bad = _StubResolver(
            {
                ("apcode", "BCEF"): Resolution(
                    matched=True,
                    value="BCEF",
                    target_axis_boost="apcode_filter",  # NOT a boost axis
                    warning=None,
                    score=100.0,
                    suggestions=[],
                ),
            }
        )
        entry = _make_entry(expose=["apcode"], resolver=bad)
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 1))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        cmd = await tool.coroutine(query="q", apcode="BCEF", runtime=_runtime())  # type: ignore[arg-type]
        # Search ran (no abort) — invalid resolution downgraded to warning.
        retriever_search.assert_awaited_once()
        # No boost merged — filter axis is rejected.
        assert "metadata_scope" not in retriever_search.call_args.kwargs
        # Warning surfaces both in state and in the ToolMessage tail.
        assert any("invalid Resolution" in w for w in cmd.update["resolution_warnings"])
        tool_msg = cmd.update["messages"][0]
        assert "invalid Resolution" in tool_msg.content

    @pytest.mark.asyncio
    async def test_matched_true_with_value_none_does_not_raise(self) -> None:
        bad = _StubResolver(
            {
                ("apcode", "BCEF"): Resolution(
                    matched=True,
                    value=None,  # contract violation
                    target_axis_boost="apcode_boost",
                    warning=None,
                    score=100.0,
                    suggestions=[],
                ),
            }
        )
        entry = _make_entry(expose=["apcode"], resolver=bad)
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        cmd = await tool.coroutine(query="q", apcode="BCEF", runtime=_runtime())  # type: ignore[arg-type]
        retriever_search.assert_awaited_once()
        assert "metadata_scope" not in retriever_search.call_args.kwargs
        assert any("invalid Resolution" in w for w in cmd.update["resolution_warnings"])

    @pytest.mark.asyncio
    async def test_resolver_raises_contained_as_warning(self) -> None:
        entry = _make_entry(expose=["apcode"], resolver=_RaisingResolver())
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        cmd = await tool.coroutine(query="q", apcode="BCEF", runtime=_runtime())  # type: ignore[arg-type]
        # Tool call must complete — search runs anyway, exception caught.
        retriever_search.assert_awaited_once()
        warnings = cmd.update["resolution_warnings"]
        assert any("resolver raised" in w and "RuntimeError" in w for w in warnings)


# ---------------------------------------------------------------------------
# Backward compatibility — entries without expose_metadata_args
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    @pytest.mark.asyncio
    async def test_legacy_entry_has_no_extra_kwargs(self) -> None:
        """Existing entries (no runtime query scope opt-in) keep working unchanged."""
        entry = _make_entry()
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 1))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)
        cmd = await tool.coroutine(query="q", runtime=_runtime())  # type: ignore[arg-type]
        assert "resolution_warnings" not in cmd.update
        kwargs = retriever_search.call_args.kwargs
        assert "metadata_scope" not in kwargs


class TestStructuredToolAinvokeContract:
    """Pins the StructuredTool.ainvoke contract used by langgraph's ToolNode.

    Regression: Phase 4 originally passed a custom ``args_schema`` to
    ``StructuredTool.from_function``. ToolNode injects ``runtime`` into
    the tool-call args dict before validation, but pydantic v2's
    ``model_validate`` ignores extra keys against a custom schema that
    doesn't declare them — so ``runtime`` got dropped from the validated
    kwargs and ``_search()`` was called without it, raising
    ``TypeError: missing 1 required positional argument: 'runtime'``
    at every live tool call.

    The fix overrides ``__signature__`` + ``__annotations__`` on
    ``_search`` so langchain auto-builds the schema (which DOES include
    ``runtime`` as a typed field, propagated from ToolNode's injection)
    AND keeps ``runtime`` flagged as injected for the schema sent to
    the LLM (see ``BaseTool.tool_call_schema``).
    """

    @pytest.mark.asyncio
    async def test_runtime_scope_call_with_injected_runtime_succeeds(self) -> None:
        """Mirrors what ToolNode does — pre-inject ``runtime`` into the
        tool input dict before invocation. Before the fix, the validated
        kwargs dropped ``runtime`` and ``_search`` raised TypeError."""
        from langchain.tools import ToolRuntime

        resolver = _StubResolver(
            {
                ("apcode", "BCEF"): Resolution(
                    matched=True, value="BCEF", target_axis_boost="apcode_boost", warning=None, score=100.0, suggestions=[]
                ),
            }
        )
        entry = _make_entry(expose=["apcode", "app_name", "entity"], resolver=resolver)
        entry.retriever.search = AsyncMock(return_value=MagicMock(__len__=lambda self: 1))  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        # Build a real ToolRuntime (what ToolNode injects in production).
        rt = ToolRuntime(
            state={"messages": []},
            context={},
            config={},  # type: ignore[arg-type]
            stream_writer=lambda _x: None,  # type: ignore[arg-type]
            tool_call_id="tc_x",
            store=None,
        )
        result = await tool.ainvoke(
            {"query": "deploy invoice", "apcode": "BCEF", "runtime": rt},
        )
        assert hasattr(result, "update")

    def test_runtime_scope_tool_call_schema_excludes_runtime(self) -> None:
        """The schema the LLM sees (``tool_call_schema``) must hide ``runtime``
        — it's an injected param, not something the planner should fill."""
        entry = _make_entry(
            expose=["apcode", "app_name", "entity"],
            resolver=_StubResolver({}),
        )
        tool = create_retriever_tool(entry)
        llm_visible = set(tool.tool_call_schema.model_fields)  # type: ignore[union-attr]
        assert "runtime" not in llm_visible
        assert "query" in llm_visible
        assert {"apcode", "app_name", "entity"} <= llm_visible


# ---------------------------------------------------------------------------
# Group C — duplicate-name multi-boost: list[str] Resolution.value
# ---------------------------------------------------------------------------


class TestListResolutionUnionedAsBoost:
    """When the resolver returns ``Resolution.value = ["E001", "E002"]`` for a
    name shared by multiple entities, every id must land in ``entity_boost``
    so the planner's "rank docs about Strix Mobile" intent applies to every
    record carrying that display name.
    """

    @pytest.mark.asyncio
    async def test_list_value_unioned_into_boost_axis(self) -> None:
        resolver = _StubResolver(
            {
                ("entity", "Strix Mobile"): Resolution(
                    matched=True,
                    value=["E001", "E002"],
                    target_axis_boost="entity_boost",
                    warning="entity='Strix Mobile' matches 2 entities — boosting all",
                    score=100.0,
                    suggestions=[],
                ),
            }
        )
        entry = _make_entry(expose=["entity"], resolver=resolver)
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        await tool.coroutine(query="how to deploy", entity="Strix Mobile", runtime=_runtime())  # type: ignore[arg-type]
        scope: MetadataScope = retriever_search.call_args.kwargs["metadata_scope"]
        assert scope.entity_boost == ["E001", "E002"]

    @pytest.mark.asyncio
    async def test_list_value_merges_with_existing_default_boost(self) -> None:
        """A default-scope boost plus a multi-id runtime-scope resolution unions
        order-preserved without losing the L1 boost."""
        resolver = _StubResolver(
            {
                ("entity", "Strix Mobile"): Resolution(
                    matched=True,
                    value=["E001", "E002"],
                    target_axis_boost="entity_boost",
                    warning=None,
                    score=100.0,
                    suggestions=[],
                ),
            }
        )
        entry = _make_entry(
            expose=["entity"],
            resolver=resolver,
            default_scope=MetadataScope(entity_boost="E_ROOT"),
        )
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        await tool.coroutine(query="q", entity="Strix Mobile", runtime=_runtime())  # type: ignore[arg-type]
        scope: MetadataScope = retriever_search.call_args.kwargs["metadata_scope"]
        assert scope.entity_boost == ["E_ROOT", "E001", "E002"]

    @pytest.mark.asyncio
    async def test_list_warning_propagated_to_state(self) -> None:
        """The "matches N entities" warning is informational and must reach
        the planner via the same plumbing as fuzzy-hit warnings."""
        resolver = _StubResolver(
            {
                ("entity", "Strix Mobile"): Resolution(
                    matched=True,
                    value=["E001", "E002"],
                    target_axis_boost="entity_boost",
                    warning="entity='Strix Mobile' matches 2 entities — boosting all",
                    score=100.0,
                    suggestions=[],
                ),
            }
        )
        entry = _make_entry(expose=["entity"], resolver=resolver)
        entry.retriever.search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)
        cmd = await tool.coroutine(query="q", entity="Strix Mobile", runtime=_runtime())  # type: ignore[arg-type]
        assert cmd.update["resolution_warnings"] == ["entity='Strix Mobile' matches 2 entities — boosting all"]


# ---------------------------------------------------------------------------
# Group D — `SupportsMetadataScope` capability check at tool-factory build time
# ---------------------------------------------------------------------------


def _unsupported_entry(
    *,
    name: str = "elastic_runbooks",
    expose: list[str] | None = None,
    resolver: Any | None = None,
    default_scope: MetadataScope | None = None,
) -> RetrieverEntry:
    """Like ``_make_entry`` but uses plain ``MockRetriever`` (no scope marker)."""
    return RetrieverEntry(
        name=name,
        description="Search runbooks",
        retriever=MockRetriever(name=name, num_results=3),
        metadata_scope=_scope_config(default_scope=default_scope, expose=expose, resolver=resolver),
    )


class TestScopeCapabilityCheck:
    """Build-time enforcement of the ``SupportsMetadataScope`` protocol.

    A retriever that doesn't honor ``metadata_scope=...`` (e.g. LightRAG, Mock)
    must NOT be wired with ``default_scope`` or ``expose_metadata_args``: the
    factory raises TypeError so the trust-boundary failure surfaces at agent-
    build time instead of silently dropping the scope at search time.
    """

    def test_default_scope_on_unsupported_retriever_raises(self) -> None:
        # MockRetriever does not declare supports_metadata_scope. Wiring a
        # build-time scope filter onto it would silently drop the ceiling
        # (LightRAG-style **kwargs swallowing) — must fail loud.
        entry = _unsupported_entry(default_scope=MetadataScope(apcode_filter="BCEF"))
        with pytest.raises(TypeError, match="SupportsMetadataScope"):
            create_retriever_tool(entry)

    def test_expose_metadata_args_on_unsupported_retriever_raises(self) -> None:
        # Even without default_scope, opting an axis into the runtime query
        # scope means the resolver-cleaned boost will end up as
        # search(metadata_scope=…) — which an unsupported backend will drop.
        entry = _unsupported_entry(expose=["apcode"], resolver=_StubResolver({}))
        with pytest.raises(TypeError, match="SupportsMetadataScope"):
            create_retriever_tool(entry)

    def test_both_scope_features_on_unsupported_retriever_raises(self) -> None:
        # Combining the two reaches the same gate. The error message must
        # name the offending entry so a consumer with a long entry list
        # can locate it without grepping.
        entry = _unsupported_entry(
            default_scope=MetadataScope(apcode_filter="BCEF"),
            expose=["apcode"],
            resolver=_StubResolver({}),
        )
        with pytest.raises(TypeError, match=r"elastic_runbooks.*SupportsMetadataScope"):
            create_retriever_tool(entry)

    def test_legacy_entry_without_scope_features_still_builds(self) -> None:
        # Backward compat — entries that don't opt into either scope feature
        # never trigger the check, even on backends that don't declare support.
        entry = _unsupported_entry()
        tool = create_retriever_tool(entry)  # must not raise
        assert tool is not None

    def test_supported_retriever_with_default_scope_builds(self) -> None:
        # A retriever class that declares the marker passes the gate. The
        # _ScopedMockRetriever subclass declared at the top of this module
        # is the canonical opt-in fixture; we don't import ElasticRetriever
        # (avoids ES bootstrap cost).
        entry = _make_entry(default_scope=MetadataScope(apcode_filter="BCEF"))
        tool = create_retriever_tool(entry)  # must not raise
        assert tool is not None

    def test_supported_retriever_with_runtime_scope_builds(self) -> None:
        entry = _make_entry(expose=["apcode"], resolver=_StubResolver({}))
        tool = create_retriever_tool(entry)
        assert tool is not None

    def test_explicit_supports_metadata_scope_false_raises(self) -> None:
        # A backend that declares the attribute as False is treated the same
        # as one that omits it (the factory layers a positive truthiness check
        # on top of the structural ``isinstance`` probe). Pinning here so a
        # future tightening doesn't silently change behavior.
        class LiarRetriever(MockRetriever):
            supports_metadata_scope = False  # explicit opt-out

        entry = RetrieverEntry(
            name="liar",
            description="d",
            retriever=LiarRetriever(name="liar"),
            metadata_scope=MetadataScopeConfig(default=MetadataScope(apcode_filter="BCEF")),
        )
        with pytest.raises(TypeError, match="SupportsMetadataScope"):
            create_retriever_tool(entry)


# ---------------------------------------------------------------------------
# Group D — hardening tests: post-build mutation, gate-time errors, boost-only
# ---------------------------------------------------------------------------


class TestPostBuildMutationCannotBypassGate:
    """``RetrieverEntry`` is a mutable dataclass — the closure must snapshot.

    A caller could otherwise build a tool from a legacy MockRetriever (no
    scope features, gate passes), then rebind ``entry.metadata_scope`` to a
    ``MetadataScopeConfig(...)`` afterward. The next tool call would forward
    ``metadata_scope`` to a backend that drops it — silently bypassing the
    build-time gate. The fix snapshots ``retriever``, ``default_scope``,
    ``resolver``, and ``exposed_axes`` into closure-locals at build time.
    """

    @pytest.mark.asyncio
    async def test_rebinding_metadata_scope_after_build_is_ignored(self) -> None:
        # Legacy entry on plain MockRetriever — gate passes (no scope set).
        entry = _unsupported_entry(name="post_mutation")
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        # Caller mutates the entry AFTER build. If the closure read
        # `entry.metadata_scope` at call time, this would smuggle a scope
        # through to MockRetriever (which silently drops it).
        entry.metadata_scope = MetadataScopeConfig(default=MetadataScope(apcode_filter="BCEF"))

        await tool.coroutine(query="q", runtime=_runtime())  # type: ignore[arg-type]
        kwargs = retriever_search.call_args.kwargs
        # Closure captured the original `metadata_scope=None` — `metadata_scope`
        # must NOT appear in the search call.
        assert "metadata_scope" not in kwargs

    @pytest.mark.asyncio
    async def test_swapping_retriever_after_build_is_ignored(self) -> None:
        # Build with a _ScopedMockRetriever (gate passes via the marker).
        entry = _make_entry(name="post_swap", default_scope=MetadataScope(apcode_filter="BCEF"))
        original_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = original_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        # Caller swaps in a totally different retriever after build. If the
        # closure read `entry.retriever`, this swap would be honored — and a
        # malicious swap to an unsupporting retriever would silently drop
        # the build-time scope.
        evil_retriever = MockRetriever(name="evil")
        evil_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        evil_retriever.search = evil_search  # type: ignore[method-assign]
        entry.retriever = evil_retriever

        await tool.coroutine(query="q", runtime=_runtime())  # type: ignore[arg-type]
        # Closure called the original retriever, not the swapped one.
        original_search.assert_called_once()
        evil_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_mutating_default_scope_list_field_after_build_is_ignored(self) -> None:
        # MetadataScope is Pydantic frozen=True (rebinding blocked) BUT its
        # ``ScopeValue = str | list[str] | None`` fields hold lists that
        # `frozen=True` does NOT make immutable — the model just blocks
        # attribute rebinding, not list methods on existing field values.
        # Without a deep snapshot, a caller could
        # ``entry.metadata_scope.default.apcode_boost.append("EVIL")`` after
        # build and the closure would forward the widened scope on the next
        # call.
        #
        # Note: Pydantic v2 deep-copies lists AT CONSTRUCTION (so passing in
        # `boost_list = ["BCEF"]` and later `boost_list.append("EVIL")` does
        # NOT propagate). The only reachable attack is direct in-place
        # mutation of the *stored* field on the entry.
        entry = _make_entry(
            name="list_mutation",
            default_scope=MetadataScope(apcode_boost=["BCEF"]),
        )
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        # Sanity: confirm Pydantic returns a real list reference we can mutate
        # (proves the attack surface is real before the snapshot proves its
        # protection of the closure-captured copy).
        assert entry.metadata_scope is not None
        scope_via_entry = entry.metadata_scope.default
        assert isinstance(scope_via_entry, MetadataScope)
        boost_field = scope_via_entry.apcode_boost
        assert isinstance(boost_field, list)
        boost_field.append("EVIL")
        assert scope_via_entry.apcode_boost == ["BCEF", "EVIL"]

        await tool.coroutine(query="q", runtime=_runtime())  # type: ignore[arg-type]
        kwargs = retriever_search.call_args.kwargs
        forwarded_scope: MetadataScope = kwargs["metadata_scope"]
        # Closure forwarded the deep-snapshotted scope, NOT the mutated one.
        assert forwarded_scope.apcode_boost == ["BCEF"]
        assert "EVIL" not in (forwarded_scope.apcode_boost or [])

    @pytest.mark.asyncio
    async def test_swapping_resolver_after_build_is_ignored(self) -> None:
        # Build with a working resolver wired through the runtime query scope.
        original_resolver = _StubResolver(
            {("apcode", "BCEF"): Resolution(matched=True, value="BCEF", target_axis_boost="apcode_boost", warning=None, score=100.0, suggestions=[])}
        )
        entry = _make_entry(name="post_resolver", expose=["apcode"], resolver=original_resolver)
        retriever_search = AsyncMock(return_value=MagicMock(__len__=lambda self: 0))
        entry.retriever.search = retriever_search  # type: ignore[method-assign]
        tool = create_retriever_tool(entry)

        # Caller swaps the resolver after build. Closure must not honor it —
        # post-build resolver swap could redirect every runtime query to a
        # malicious resolver that boosts attacker-controlled values.
        evil_resolver = _StubResolver(
            {("apcode", "BCEF"): Resolution(matched=True, value="EVIL", target_axis_boost="apcode_boost", warning=None, score=100.0, suggestions=[])}
        )
        # Rebind the whole config to an evil one. The closure must have
        # snapshotted the original resolver; the evil one should never fire.
        entry.metadata_scope = MetadataScopeConfig(exposed_axes=("apcode",), value_resolver=evil_resolver)

        await tool.coroutine(query="q", apcode="BCEF", runtime=_runtime())  # type: ignore[arg-type]
        scope: MetadataScope = retriever_search.call_args.kwargs["metadata_scope"]
        # Original resolver fired (value="BCEF"), evil one didn't (would have been "EVIL").
        assert scope.apcode_boost == "BCEF"
        assert original_resolver.calls == [("apcode", "BCEF")]
        assert evil_resolver.calls == []


class TestStrictTrueGate:
    """`Literal[True]` contract enforcement — truthy non-bool must NOT pass."""

    def test_string_truthy_marker_rejected(self) -> None:
        # `supports_metadata_scope = "yes"` is truthy but is not literal True.
        # The Protocol's `Literal[True]` annotation says "must be True".
        class StringMarkerRetriever(MockRetriever):
            supports_metadata_scope = "yes"  # type: ignore[assignment]

        entry = RetrieverEntry(
            name="strmarker",
            description="d",
            retriever=StringMarkerRetriever(name="strmarker"),
            metadata_scope=MetadataScopeConfig(default=MetadataScope(apcode_filter="BCEF")),
        )
        with pytest.raises(TypeError, match="SupportsMetadataScope"):
            create_retriever_tool(entry)

    def test_int_truthy_marker_rejected(self) -> None:
        class IntMarkerRetriever(MockRetriever):
            supports_metadata_scope = 1  # type: ignore[assignment]

        entry = RetrieverEntry(
            name="intmarker",
            description="d",
            retriever=IntMarkerRetriever(name="intmarker"),
            metadata_scope=MetadataScopeConfig(default=MetadataScope(apcode_filter="BCEF")),
        )
        with pytest.raises(TypeError, match="SupportsMetadataScope"):
            create_retriever_tool(entry)


class TestEmptyDefaultScopeDoesNotTripGate:
    """``MetadataScope()`` (empty) is documented as "full access" and is not
    forwarded by ``_search`` (the empty check guards it). The gate must match
    that semantics — over-eager firing on Mock for a noop scope would be a
    behavior regression."""

    def test_empty_default_scope_on_mock_builds(self) -> None:
        entry = _unsupported_entry(default_scope=MetadataScope())  # empty scope
        tool = create_retriever_tool(entry)  # must not raise
        assert tool is not None

    def test_non_empty_default_scope_on_mock_still_raises(self) -> None:
        # Sanity: a real (non-empty) scope must still trip the gate.
        entry = _unsupported_entry(default_scope=MetadataScope(apcode_filter="BCEF"))
        with pytest.raises(TypeError, match="SupportsMetadataScope"):
            create_retriever_tool(entry)

    def test_boost_only_default_scope_on_mock_still_raises(self) -> None:
        # `is_empty()` returns False for boost-only scopes (verified against
        # MetadataScope's implementation), so the gate must still fire on Mock.
        # Pinning here so a future redefinition of "empty" doesn't accidentally
        # carve a hole for boost-only scopes.
        entry = _unsupported_entry(default_scope=MetadataScope(apcode_boost="X"))
        with pytest.raises(TypeError, match="SupportsMetadataScope"):
            create_retriever_tool(entry)


class TestSpecBuilderClosesScopeBypass:
    """``build_entries_from_specs`` and the non-ES builders close the scope-
    bypass surface at the earliest possible layer. ``mock`` refuses scope
    kwargs, while LightRAG accepts only its backend-specific scope model."""

    def test_create_mock_entry_does_not_accept_scope_kwargs(self) -> None:
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_retrievers import create_mock_entry

        # Scope kwargs are no longer in the signature; passing one is a hard
        # signature error so a programmatic caller can't silently wire scope
        # onto a backend that doesn't honor it.
        with pytest.raises(TypeError):
            create_mock_entry(name="m", default_scope=MetadataScope(apcode_filter="BCEF"))  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            create_mock_entry(name="m", expose_metadata_args=["apcode"])  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            create_mock_entry(name="m", metadata_value_resolver=_StubResolver({}))  # type: ignore[call-arg]

    def test_create_lightrag_entry_accepts_lightrag_default_scope(self) -> None:
        from sta_agent_core.repositories.retrievers.lightrag import LightRAGMetadataScope
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_retrievers import create_lightrag_entry

        scope = LightRAGMetadataScope(tag_filter={"private_operator": ["tag-a"]})

        entry = create_lightrag_entry(name="kg", base_url="http://lightrag", default_scope=scope)

        assert entry.metadata_scope is not None
        assert entry.metadata_scope.default == scope
        assert entry.accepts_caller_scope is True

    def test_create_lightrag_entry_forwards_twin_api_option(self) -> None:
        from sta_agent_core.repositories.retrievers.lightrag import LightRAGRetriever
        from sta_agent_core.repositories.retrievers.lightrag.engines.http_engine import LightRAGHttpEngine
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_retrievers import create_lightrag_entry

        entry = create_lightrag_entry(
            name="kg",
            base_url="http://lightrag",
            use_twin_api=True,
        )

        assert isinstance(entry.retriever, LightRAGRetriever)
        assert isinstance(entry.retriever._engine, LightRAGHttpEngine)
        assert entry.retriever._engine._use_twin_api is True

    def test_lightrag_spec_normalizes_default_scope(self) -> None:
        from sta_agent_core.repositories.retrievers.lightrag import LightRAGMetadataScope
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_retrievers import build_entries_from_specs

        entries = build_entries_from_specs(
            [
                {
                    "type": "lightrag",
                    "name": "kg",
                    "config": {
                        "base_url": "http://lightrag",
                        "default_scope": {"tag_filter": {"private_operator": ["tag-a"]}},
                    },
                }
            ]
        )

        default_scope = entries[0].metadata_scope.default if entries[0].metadata_scope else None
        assert isinstance(default_scope, LightRAGMetadataScope)
        assert default_scope.tag_filter == {"private_operator": ["tag-a"]}

    def test_build_entries_from_specs_rejects_default_scope_on_mock(self) -> None:
        # The realistic JSON-spec shape: ``default_scope`` arrives as a dict
        # for a backend that doesn't honor scope. Refused at spec layer with
        # a typed error rather than silently dropped.
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_retrievers import build_entries_from_specs

        spec = {
            "type": "mock",
            "name": "smoketest",
            "config": {"default_scope": {"apcode_filter": "BCEF"}},
        }
        with pytest.raises(TypeError, match="metadata-scope (?:keys|kwargs).*not supported"):
            build_entries_from_specs([spec])

    def test_build_entries_from_specs_rejects_metadata_scope_instance_on_mock(self) -> None:
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_retrievers import build_entries_from_specs

        scope = MetadataScope(apcode_filter="BCEF")
        spec = {"type": "mock", "name": "smoketest", "config": {"default_scope": scope}}
        with pytest.raises(TypeError, match="metadata-scope (?:keys|kwargs).*not supported"):
            build_entries_from_specs([spec])

    def test_build_entries_from_specs_rejects_expose_metadata_args_on_mock(self) -> None:
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_retrievers import build_entries_from_specs

        spec = {
            "type": "mock",
            "name": "smoketest",
            "config": {"expose_metadata_args": ["apcode"]},
        }
        with pytest.raises(TypeError, match="metadata-scope (?:keys|kwargs).*not supported"):
            build_entries_from_specs([spec])

    def test_build_entries_from_specs_rejects_resolver_on_mock(self) -> None:
        # ``metadata_value_resolver`` is a Python Protocol instance — JSON
        # specs cannot carry it. The spec-normalizer raises before the
        # builder-level ``_reject_scope_kwargs`` gate is reached, so the
        # error message points at the JSON limitation rather than the
        # backend-not-scope-aware limitation. Either is correct rejection.
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_retrievers import build_entries_from_specs

        spec = {
            "type": "mock",
            "name": "smoketest",
            "config": {"metadata_value_resolver": _StubResolver({})},
        }
        with pytest.raises(TypeError, match="metadata_value_resolver.*spec config"):
            build_entries_from_specs([spec])

    def test_build_entries_from_specs_rejects_resolver_on_elastic_spec(self) -> None:
        # MetadataValueResolver is a Python Protocol — can't be carried by
        # JSON. Loud rejection beats silent drop OR deserialization gymnastics.
        # This path runs on a scope-aware backend so it reaches the resolver
        # rejection rather than the upstream non-scope-aware rejection.
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_retrievers import build_entries_from_specs

        spec = {
            "type": "elastic",
            "name": "smoketest",
            "config": {"metadata_value_resolver": _StubResolver({})},
        }
        with pytest.raises(TypeError, match="metadata_value_resolver.*spec config"):
            build_entries_from_specs([spec])

    def test_build_entries_from_specs_rejects_invalid_default_scope_type_on_elastic(self) -> None:
        # Anything other than dict or MetadataScope is loud-rejected so an
        # AttributeError-at-gate-time failure mode can't recur. Run on an
        # elastic spec so the type validator is reached.
        from sta_agent_engine.agents.knowledge_agent.knowledge_agent_retrievers import build_entries_from_specs

        spec = {"type": "elastic", "name": "smoketest", "config": {"default_scope": "BCEF"}}
        with pytest.raises(TypeError, match="default_scope.*MetadataScope"):
            build_entries_from_specs([spec])

-------

tests/test_ai_engine/agents/orchestrator/middlewares/test_knowledge_bridge.py
----
"""Tests for :class:`KnowledgeBridgeMiddleware` and the shared bridge channels.

The middleware is pure state-schema widening: it declares the ``ka_metadata_scope``
(input) and ``ka_sources`` (output) channels so the deepagents ``task`` tool
carries them into and out of the Knowledge Agent subagent. These tests pin:

- both channels appear on the middleware's ``state_schema``;
- neither key is in deepagents' ``_EXCLUDED_STATE_KEYS`` (so ``task`` crosses
  them both ways);
- ``before_agent`` resets ``ka_sources`` via ``Overwrite`` (an accumulate
  reducer makes a bare ``[]`` a no-op);
- ``before_agent`` surfaces a caller-seeded document selection
  (``ka_metadata_scope.doc_ids``) to the planner as a deterministic-id
  ``<system_reminder>`` message — turn-anchored, idempotent on re-entry,
  with a one-shot cleared note when a previously-scoped thread loses its
  selection;
- the ``merge_ka_sources`` reducer accumulates + de-duplicates;
- the channel-name constants match the live attribute names.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command, Overwrite

from sta_agent_engine.agents.knowledge_agent.knowledge_bridge_channels import (
    KA_METADATA_SCOPE_KEY,
    KA_SOURCES_KEY,
    merge_ka_sources,
    normalize_apcode,
    normalize_app_name,
    normalize_doc_ids,
    normalize_entity,
    read_ka_metadata_scope,
)
from sta_agent_engine.agents.orchestrator.middlewares.knowledge_bridge import (
    KnowledgeBridgeMiddleware,
    KnowledgeBridgeState,
)


def test_channel_name_constants() -> None:
    assert KA_METADATA_SCOPE_KEY == "ka_metadata_scope"
    assert KA_SOURCES_KEY == "ka_sources"


def test_state_schema_declares_both_channels() -> None:
    keys = set(KnowledgeBridgeState.__annotations__)
    assert KA_METADATA_SCOPE_KEY in keys
    assert KA_SOURCES_KEY in keys
    # The base AgentState keys still come through via inheritance.
    assert "messages" in keys


def test_middleware_state_schema_is_knowledge_bridge_state() -> None:
    mw = KnowledgeBridgeMiddleware()
    assert mw.state_schema is KnowledgeBridgeState


def test_ka_sources_compiles_to_a_merge_reducer_channel_not_last_value() -> None:
    """The compiled ``ka_sources`` channel must use the ``merge_ka_sources``
    reducer, not ``LastValue``.

    The orchestrator delegates to the KA through deepagents' ``task`` tool; a
    single planner turn can fire several ``task`` calls concurrently, each writing
    ``ka_sources`` in the same super-step. A ``LastValue`` channel raises
    "can receive only one value per step" on that. ``create_agent`` only keeps a
    reducer when ``OmitFromInput`` precedes it in the ``Annotated`` — if their
    order is flipped the reducer is silently dropped and the channel degrades to
    ``LastValue``. This test pins the working order by inspecting the compiled
    graph (a schema-level annotation check would not catch the degrade).
    """
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langgraph.channels.binop import BinaryOperatorAggregate
    from langgraph.channels.last_value import LastValue

    class _BindableFake(GenericFakeChatModel):
        def bind_tools(self, tools: object, **kwargs: object) -> _BindableFake:  # noqa: ARG002
            return self

    agent = create_agent(
        _BindableFake(messages=iter([AIMessage(content="x")])),
        tools=[],
        middleware=[KnowledgeBridgeMiddleware()],
    )
    channel = agent.channels["ka_sources"]
    assert isinstance(channel, BinaryOperatorAggregate), f"ka_sources degraded to {type(channel).__name__}"
    assert not isinstance(channel, LastValue)
    # OmitFromInput is still honored — the output-only channel stays out of input.
    assert "ka_sources" not in (agent.get_input_jsonschema().get("properties") or {})


def test_keys_not_excluded_from_task_boundary() -> None:
    """Both keys must cross the deepagents ``task`` state filter, both ways."""
    from deepagents.middleware.subagents import _EXCLUDED_STATE_KEYS

    assert KA_METADATA_SCOPE_KEY not in _EXCLUDED_STATE_KEYS
    assert KA_SOURCES_KEY not in _EXCLUDED_STATE_KEYS


def test_before_agent_resets_ka_sources_with_overwrite() -> None:
    mw = KnowledgeBridgeMiddleware()
    update = mw.before_agent({}, None)  # type: ignore[arg-type]
    assert update is not None
    reset = update["ka_sources"]
    assert isinstance(reset, Overwrite)
    assert reset.value == []
    # the announce cursor resets in the same per-run update, so a fresh run
    # re-announces from [1] even on a checkpointed thread.
    assert update["ka_sources_announced"] == 0


async def test_abefore_agent_resets_ka_sources_with_overwrite() -> None:
    mw = KnowledgeBridgeMiddleware()
    update = await mw.abefore_agent({}, None)  # type: ignore[arg-type]
    assert update is not None
    reset = update["ka_sources"]
    assert isinstance(reset, Overwrite)
    assert reset.value == []
    assert update["ka_sources_announced"] == 0


# ---------------------------------------------------------------------------
# document-selection reminder (ka_metadata_scope.doc_ids → planner message)
# ---------------------------------------------------------------------------


def _reminders(update: dict) -> list:
    return update.get("messages") or []


def test_no_messages_means_reset_only() -> None:
    mw = KnowledgeBridgeMiddleware()
    update = mw.before_agent({"ka_metadata_scope": {"doc_ids": ["d1"]}}, None)  # type: ignore[arg-type]
    assert isinstance(update["ka_sources"], Overwrite)
    assert "messages" not in update


def test_reminder_injected_when_doc_ids_present() -> None:
    mw = KnowledgeBridgeMiddleware()
    state = {
        "messages": [HumanMessage("what does the runbook say?")],
        "ka_metadata_scope": {"doc_ids": ["d1", "d2"]},
    }
    update = mw.before_agent(state, None)  # type: ignore[arg-type]
    # The sources reset is still part of the same update.
    assert isinstance(update["ka_sources"], Overwrite)
    (reminder,) = _reminders(update)
    assert isinstance(reminder, SystemMessage)
    assert reminder.id is not None and reminder.id.startswith("sta-ka-scope-reminder::turn-1::")
    assert "<system_reminder>" in reminder.content
    for needle in ("d1", "d2", "knowledge_agent"):
        assert needle in reminder.content
    # The reminder must NOT name a corpus: the KA picks its own corpus, and the
    # planner tool set is structurally restricted to scope-accepting corpora
    # when a selection is present, so steering toward `general_doc` is both
    # redundant and contradicts the "don't name a corpus" tasking rule.
    assert "general_doc" not in reminder.content
    assert "corpus" not in reminder.content


async def test_async_reminder_matches_sync() -> None:
    mw = KnowledgeBridgeMiddleware()
    state = {
        "messages": [HumanMessage("q")],
        "ka_metadata_scope": {"doc_ids": ["d1"]},
    }
    sync_update = mw.before_agent(dict(state), None)  # type: ignore[arg-type]
    async_update = await mw.abefore_agent(dict(state), None)  # type: ignore[arg-type]
    assert _reminders(sync_update)[0].id == _reminders(async_update)[0].id
    assert _reminders(sync_update)[0].content == _reminders(async_update)[0].content


def test_reminder_idempotent_within_a_turn() -> None:
    """Re-entry on the same turn (interrupt resume, retry) must not duplicate."""
    mw = KnowledgeBridgeMiddleware()
    state = {
        "messages": [HumanMessage("q")],
        "ka_metadata_scope": {"doc_ids": ["d1", "d2"]},
    }
    first = _reminders(mw.before_agent(state, None))[0]  # type: ignore[arg-type]
    state["messages"] = [*state["messages"], first]
    update = mw.before_agent(state, None)  # type: ignore[arg-type]
    assert "messages" not in update
    assert isinstance(update["ka_sources"], Overwrite)


def test_reminder_id_is_order_insensitive() -> None:
    """The selection is a set — reordered ids must not mint a new reminder."""
    mw = KnowledgeBridgeMiddleware()
    state = {
        "messages": [HumanMessage("q")],
        "ka_metadata_scope": {"doc_ids": ["d1", "d2"]},
    }
    first = _reminders(mw.before_agent(state, None))[0]  # type: ignore[arg-type]
    state["messages"] = [*state["messages"], first]
    state["ka_metadata_scope"] = {"doc_ids": ["d2", "d1"]}
    assert "messages" not in mw.before_agent(state, None)  # type: ignore[arg-type]


def test_reminder_refires_on_new_turn_with_same_selection() -> None:
    mw = KnowledgeBridgeMiddleware()
    state = {
        "messages": [HumanMessage("q1")],
        "ka_metadata_scope": {"doc_ids": ["d1"]},
    }
    first = _reminders(mw.before_agent(state, None))[0]  # type: ignore[arg-type]
    state["messages"] = [*state["messages"], first, AIMessage("a1"), HumanMessage("q2")]
    (second,) = _reminders(mw.before_agent(state, None))  # type: ignore[arg-type]
    assert second.id != first.id
    assert second.id is not None and "turn-2" in second.id


def test_reminder_refires_when_selection_changes() -> None:
    mw = KnowledgeBridgeMiddleware()
    state = {
        "messages": [HumanMessage("q")],
        "ka_metadata_scope": {"doc_ids": ["d1"]},
    }
    first = _reminders(mw.before_agent(state, None))[0]  # type: ignore[arg-type]
    state["messages"] = [*state["messages"], first]
    state["ka_metadata_scope"] = {"doc_ids": ["d3"]}
    (second,) = _reminders(mw.before_agent(state, None))  # type: ignore[arg-type]
    assert second.id != first.id
    assert "d3" in second.content


def test_no_reminder_for_scope_without_doc_ids() -> None:
    """Other axes (apcode/app_name/entity) do not trigger the reminder."""
    mw = KnowledgeBridgeMiddleware()
    state = {
        "messages": [HumanMessage("q")],
        "ka_metadata_scope": {"apcode": ["AP90021"]},
    }
    assert "messages" not in mw.before_agent(state, None)  # type: ignore[arg-type]


def test_no_reminder_when_scope_absent_on_fresh_thread() -> None:
    mw = KnowledgeBridgeMiddleware()
    state = {"messages": [HumanMessage("q")]}
    assert "messages" not in mw.before_agent(state, None)  # type: ignore[arg-type]


def test_cleared_note_after_scoped_turn() -> None:
    mw = KnowledgeBridgeMiddleware()
    state = {
        "messages": [HumanMessage("q1")],
        "ka_metadata_scope": {"doc_ids": ["d1"]},
    }
    first = _reminders(mw.before_agent(state, None))[0]  # type: ignore[arg-type]
    # Next turn: the caller seeds no selection.
    state = {"messages": [HumanMessage("q1"), first, AIMessage("a1"), HumanMessage("q2")]}
    (note,) = _reminders(mw.before_agent(state, None))  # type: ignore[arg-type]
    assert note.id is not None and note.id.endswith("::cleared")
    assert "no longer applies" in note.content


def test_cleared_note_not_repeated() -> None:
    mw = KnowledgeBridgeMiddleware()
    state = {
        "messages": [HumanMessage("q1")],
        "ka_metadata_scope": {"doc_ids": ["d1"]},
    }
    first = _reminders(mw.before_agent(state, None))[0]  # type: ignore[arg-type]
    state = {"messages": [HumanMessage("q1"), first, AIMessage("a1"), HumanMessage("q2")]}
    note = _reminders(mw.before_agent(state, None))[0]  # type: ignore[arg-type]
    # A third unscoped turn: the most recent reminder is already the cleared note.
    state = {"messages": [*state["messages"], note, AIMessage("a2"), HumanMessage("q3")]}
    assert "messages" not in mw.before_agent(state, None)  # type: ignore[arg-type]


def test_selection_can_return_after_cleared_note() -> None:
    mw = KnowledgeBridgeMiddleware()
    history = [
        HumanMessage("q1"),
        AIMessage("a1"),
        HumanMessage("q2"),
    ]
    state = {"messages": history, "ka_metadata_scope": {"doc_ids": ["d1"]}}
    first = _reminders(mw.before_agent(state, None))[0]  # type: ignore[arg-type]
    state = {"messages": [*history, first, AIMessage("a2"), HumanMessage("q3")]}
    note = _reminders(mw.before_agent(state, None))[0]  # type: ignore[arg-type]
    state = {
        "messages": [*state["messages"], note, AIMessage("a3"), HumanMessage("q4")],
        "ka_metadata_scope": {"doc_ids": ["d9"]},
    }
    (again,) = _reminders(mw.before_agent(state, None))  # type: ignore[arg-type]
    assert "d9" in again.content


def test_long_doc_id_list_is_truncated() -> None:
    mw = KnowledgeBridgeMiddleware()
    doc_ids = [f"doc-{i:02d}" for i in range(25)]
    state = {
        "messages": [HumanMessage("q")],
        "ka_metadata_scope": {"doc_ids": doc_ids},
    }
    (reminder,) = _reminders(mw.before_agent(state, None))  # type: ignore[arg-type]
    assert "doc-19" in reminder.content
    assert "doc-20" not in reminder.content
    assert "+5 more" in reminder.content


# ---------------------------------------------------------------------------
# canonical-sources announcer (before_model)
# ---------------------------------------------------------------------------
#
# The announcer numbers from the POST-MERGE ``ka_sources`` list in
# ``before_model`` — the first hook after the tool super-step merges. That is
# what makes the numbering correct when the planner fans out parallel ``task``
# delegations in one turn: their source blocks are already concatenated in panel
# order on the channel, so a single note numbers them contiguously. (A per-``task``
# ``wrap_tool_call`` announcer numbered each sibling from the same pre-merge
# offset and collided — the bug this design fixes.)


def _announce(state: dict[str, Any]) -> dict[str, Any] | None:
    """Invoke the sync ``before_model`` announcer (runtime is unused)."""
    return KnowledgeBridgeMiddleware().before_model(state, None)  # type: ignore[arg-type]


def _note(update: dict[str, Any]) -> SystemMessage:
    """Pull the single injected ``<knowledge_sources>`` note from an update."""
    msgs = update["messages"]
    assert len(msgs) == 1
    note = msgs[0]
    assert isinstance(note, SystemMessage)
    assert "<knowledge_sources>" in note.content
    return note


def test_announcer_numbers_first_batch_from_one() -> None:
    sources = [{"title": "Alpha", "url": "u/a"}, {"title": "Beta", "url": "u/b"}]

    out = _announce({"ka_sources": sources})

    assert out is not None
    note = _note(out)
    assert "[1] [Alpha](u/a)" in note.content
    assert "[2] [Beta](u/b)" in note.content
    assert note.id is not None and note.id.startswith("sta-ka-sources::0::")
    # cursor advances to cover every announced row
    assert out["ka_sources_announced"] == 2


def test_parallel_batch_numbers_contiguously() -> None:
    """Two ``task`` calls fanned out in one turn: the reducer already merged both
    blocks in panel order, so a single ``before_model`` numbers them [1..5] — no
    collision (the regression this fix targets)."""
    merged = [
        {"title": "A1", "url": "a1"},
        {"title": "A2", "url": "a2"},
        {"title": "B1", "url": "b1"},
        {"title": "B2", "url": "b2"},
        {"title": "B3", "url": "b3"},
    ]

    out = _announce({"ka_sources": merged, "ka_sources_announced": 0})

    assert out is not None
    note = _note(out)
    for n, source in enumerate(merged, start=1):
        assert f"[{n}] [{source['title']}]({source['url']})" in note.content
    assert out["ka_sources_announced"] == 5


def test_announcer_applies_offset_from_the_cursor() -> None:
    """A later delegation: three rows already announced → the new row is [4]."""
    state = {
        "ka_sources": [{"title": "x"}, {"title": "y"}, {"title": "z"}, {"title": "Gamma", "url": "u/g"}],
        "ka_sources_announced": 3,
    }

    out = _announce(state)

    assert out is not None
    note = _note(out)
    assert "[4] [Gamma](u/g)" in note.content  # cursor 3 → this row becomes [4]
    assert note.id is not None and note.id.startswith("sta-ka-sources::3::")
    assert out["ka_sources_announced"] == 4


def test_no_note_when_nothing_new_merged() -> None:
    # cursor already covers every row (answer-now turn, or a retry)
    assert _announce({"ka_sources": [{"title": "A"}], "ka_sources_announced": 1}) is None


def test_no_note_on_empty_channel() -> None:
    # first planner call of the run: channel reset, nothing surfaced yet
    assert _announce({"ka_sources": []}) is None
    assert _announce({}) is None


def test_announcer_only_adds_a_message_never_touches_other_state() -> None:
    """Pitfall #5: the announcer touches no ToolMessage — it appends one
    SystemMessage and advances the cursor, nothing else."""
    out = _announce({"ka_sources": [{"title": "T", "url": "u"}]})

    assert out is not None
    assert set(out) == {"messages", "ka_sources_announced"}
    assert all(isinstance(m, SystemMessage) for m in out["messages"])


def test_announcer_id_is_deterministic_on_replay() -> None:
    """Same cursor + same sources → same id, so ``add_messages`` de-dups a replay."""
    first = _note(_announce({"ka_sources": [{"title": "A", "url": "u/a"}]}))  # type: ignore[arg-type]
    second = _note(_announce({"ka_sources": [{"title": "A", "url": "u/a"}]}))  # type: ignore[arg-type]
    assert first.id == second.id
    assert first.content == second.content


async def test_async_announcer_matches_sync() -> None:
    mw = KnowledgeBridgeMiddleware()
    sources = [{"title": "A", "url": "u/a"}]

    sync_out = mw.before_model({"ka_sources": sources}, None)  # type: ignore[arg-type]
    async_out = await mw.abefore_model({"ka_sources": sources}, None)  # type: ignore[arg-type]

    assert sync_out is not None and async_out is not None
    assert sync_out["messages"][0].id == async_out["messages"][0].id
    assert sync_out["messages"][0].content == async_out["messages"][0].content
    assert sync_out["ka_sources_announced"] == async_out["ka_sources_announced"]


def test_source_without_url_or_title_renders_safely() -> None:
    sources = [{"title": "Has Title", "url": ""}, {"title": "", "url": "only/url"}]

    out = _announce({"ka_sources": sources})

    assert out is not None
    note = _note(out).content
    assert "[1] [Has Title]" in note
    assert "[1] [Has Title](" not in note  # no dangling link parens when url is missing
    assert "[2] [Untitled source](only/url)" in note


def test_announcer_end_to_end_parallel_fanout_numbers_contiguously() -> None:
    """End-to-end through ``create_agent``: a planner fans out two delegations in
    ONE turn (A→2 sources, B→3). The ``merge_ka_sources`` reducer concatenates
    both blocks; the ``before_model`` announcer (running after the merged tool
    super-step) emits exactly ONE ``<knowledge_sources>`` note whose numbers map
    1:1 to the final ``ka_sources`` order — the parallel collision is gone.

    This exercises the real framework loop (the fix's load-bearing claim: the
    hook sees post-merge state), complementing the unit tests that assert the
    numbering given an already-merged channel.
    """
    from langchain.agents import create_agent
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool

    from sta_agent_engine.models.fake_models import AgentIntegrationModel

    @tool
    def kb(label: str, n: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Fake KA delegation returning ``n`` sources tagged ``label``."""
        srcs = [{"title": f"{label}{i}", "url": f"u/{label}{i}"} for i in range(1, n + 1)]
        return Command(update={"ka_sources": srcs, "messages": [ToolMessage(f"{label} done", tool_call_id=tool_call_id)]})

    turn1 = AIMessage(
        content="",
        tool_calls=[
            {"name": "kb", "args": {"label": "A", "n": 2}, "id": "a"},
            {"name": "kb", "args": {"label": "B", "n": 3}, "id": "b"},
        ],
    )
    turn2 = AIMessage(content="final answer")
    agent = create_agent(
        AgentIntegrationModel(responses=[turn1, turn2]),
        tools=[kb],
        middleware=[KnowledgeBridgeMiddleware()],
    )

    out = agent.invoke({"messages": [HumanMessage("go")]})

    notes = [m for m in out["messages"] if isinstance(m, SystemMessage) and "<knowledge_sources>" in (m.content or "")]
    assert len(notes) == 1, "parallel fan-out must yield ONE consolidated note, not one-per-call"
    body = notes[0].content
    # every row is numbered by its position in the FINAL merged channel
    for i, src in enumerate(out["ka_sources"], start=1):
        assert f"[{i}] [{src['title']}]" in body
    assert [s["title"] for s in out["ka_sources"]] == ["A1", "A2", "B1", "B2", "B3"]


# ---------------------------------------------------------------------------
# merge_ka_sources reducer
# ---------------------------------------------------------------------------


def test_merge_ka_sources_accumulates() -> None:
    left = [{"title": "A", "url": "u/a", "retriever_name": "r1"}]
    right = [{"title": "B", "url": "u/b", "retriever_name": "r1"}]
    merged = merge_ka_sources(left, right)
    assert merged == left + right
    # New list, not a mutated reference.
    assert merged is not left


def test_merge_ka_sources_concatenates_without_cross_call_dedup() -> None:
    # Pure concatenation keeps each KA call's block contiguous and
    # position-stable, so the orchestrator can offset a later call's numbering by
    # the count of sources already surfaced. A document re-cited by a later call
    # appears twice (accepted trade-off) — collapsing it would shift the offset
    # arithmetic onto the wrong row.
    left = [{"title": "A", "url": "u/a", "source_type": "chunk"}]  # call 1: global [1]
    right = [
        {"title": "B", "url": "u/b", "source_type": "chunk"},  # call 2 local [1] → global [2]
        {"title": "A", "url": "u/a", "source_type": "chunk"},  # re-cite of A, call 2 local [2] → global [3]
    ]
    merged = merge_ka_sources(left, right)
    assert merged == left + right  # contiguous, no dedup
    assert [s["title"] for s in merged] == ["A", "B", "A"]


def test_merge_ka_sources_handles_none() -> None:
    assert merge_ka_sources(None, None) == []
    assert merge_ka_sources(None, [{"title": "A", "url": "u", "retriever_name": "r"}]) == [{"title": "A", "url": "u", "retriever_name": "r"}]


# ---------------------------------------------------------------------------
# normalize helpers
# ---------------------------------------------------------------------------


def test_normalize_doc_ids_coerces_and_trims() -> None:
    assert normalize_doc_ids(["  a ", "b", "", "a"]) == ["a", "b"]
    assert normalize_doc_ids("single") == ["single"]
    assert normalize_doc_ids(None) == []
    assert normalize_doc_ids(123) == []
    assert normalize_doc_ids([1, "ok", None]) == ["ok"]


def test_normalize_apcode_mirrors_doc_ids() -> None:
    assert normalize_apcode(["AP90021", "AP90021", " "]) == ["AP90021"]
    assert normalize_apcode("AP00001") == ["AP00001"]


def test_normalize_app_name_mirrors_doc_ids() -> None:
    assert normalize_app_name(["billing", "billing", ""]) == ["billing"]
    assert normalize_app_name("payments") == ["payments"]
    assert normalize_app_name(None) == []


def test_normalize_entity_mirrors_doc_ids() -> None:
    assert normalize_entity(["e1", " e1 ", "e2"]) == ["e1", "e2"]
    assert normalize_entity("ent") == ["ent"]
    assert normalize_entity(42) == []


# ---------------------------------------------------------------------------
# read_ka_metadata_scope — validator / boost-guard
# ---------------------------------------------------------------------------


def test_read_ka_metadata_scope_normalizes_all_four_axes() -> None:
    scope = read_ka_metadata_scope(
        {
            "doc_ids": ["  d1 ", "d1", "d2"],
            "apcode": "AP90021",
            "app_name": ["billing"],
            "entity": ["e1", "e1"],
        }
    )
    assert scope == {
        "doc_ids": ["d1", "d2"],
        "apcode": ["AP90021"],
        "app_name": ["billing"],
        "entity": ["e1"],
    }


def test_read_ka_metadata_scope_normalizes_opaque_tag_filter() -> None:
    scope = read_ka_metadata_scope(
        {
            "tag_filter": {
                "private_operator": [" tag-a ", "tag-a", "", "tag-b"],
                "": ["ignored"],
            }
        }
    )

    assert scope == {"tag_filter": {"private_operator": ["tag-a", "tag-b"]}}


def test_read_ka_metadata_scope_normalizes_explicit_unfiltered_retriever_names() -> None:
    scope = read_ka_metadata_scope(
        {
            "doc_ids": ["doc-1"],
            "include_without_caller_scope": [" lightrag_kg ", "", "lightrag_kg"],
        }
    )

    assert scope == {
        "doc_ids": ["doc-1"],
        "include_without_caller_scope": ["lightrag_kg"],
    }


def test_read_ka_metadata_scope_drops_boost_keys_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        scope = read_ka_metadata_scope(
            {
                "apcode": "AP90021",
                "apcode_boost": "AP90021",  # smuggled boost — must be dropped
                "entity_boost": ["e1"],
                "unknown_axis": "x",
            }
        )
    # Only the valid filter axis survives — no boost leaks through.
    assert scope == {"apcode": ["AP90021"]}
    assert "apcode_boost" not in scope
    assert "entity_boost" not in scope
    assert any("dropped" in r.message and "boost" in r.message.lower() for r in caplog.records)


def test_read_ka_metadata_scope_non_dict_returns_empty() -> None:
    assert read_ka_metadata_scope(None) == {}
    assert read_ka_metadata_scope("nope") == {}
    assert read_ka_metadata_scope([1, 2]) == {}


def test_read_ka_metadata_scope_omits_empty_axes() -> None:
    # An axis that normalizes to nothing is omitted entirely.
    assert read_ka_metadata_scope({"doc_ids": ["", "  "], "apcode": "AP1"}) == {"apcode": ["AP1"]}

-------

tests/test_ai_engine/agents/orchestrator/test_twin_ka_entries.py
----
"""Tests for orchestrator-owned Knowledge Agent retriever entries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pytest

from sta_agent_core import MetadataScope
from sta_agent_core.repositories.retrievers.mock import MockRetriever
from sta_agent_engine.agents.knowledge_agent import MetadataScopeConfig, RetrieverEntry
from sta_agent_engine.agents.knowledge_agent.tools.retriever_tool_factory import create_retriever_tools
from sta_agent_engine.agents.orchestrator.sources import twin_ka_entries


pytestmark = pytest.mark.unit

_EXPOSED_METADATA_ARGS = ["apcode", "app_name", "entity"]
_EXPECTED_TOOL_ARGS = {"query", *_EXPOSED_METADATA_ARGS}


class _ScopedMockRetriever(MockRetriever):
    supports_metadata_scope: Literal[True] = True

    @staticmethod
    def resolve_caller_scope(bundle: Mapping[str, Any]) -> MetadataScope | None:
        return MetadataScope.from_caller_scope(bundle)


@pytest.fixture
def elastic_entry_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace ``create_elastic_entry`` with a network-free builder spy."""
    calls: list[dict[str, Any]] = []

    def _spy(
        name: str,
        description: str,
        *,
        default_scope: MetadataScope | None = None,
        expose_metadata_args: list[str] | None = None,
        **kwargs: Any,
    ) -> RetrieverEntry:
        calls.append(
            {
                "name": name,
                "description": description,
                "default_scope": default_scope,
                "expose_metadata_args": expose_metadata_args,
                "kwargs": kwargs,
            }
        )
        return RetrieverEntry(
            name=name,
            description=description,
            retriever=_ScopedMockRetriever(name=name),
            metadata_scope=MetadataScopeConfig(
                default=default_scope,
                exposed_axes=tuple(expose_metadata_args or ()),
            ),
        )

    monkeypatch.setattr(twin_ka_entries, "create_elastic_entry", _spy)
    return calls


def _tool_args_by_name(entries: list[RetrieverEntry]) -> dict[str, set[str]]:
    return {tool.name: set(tool.args) for tool in create_retriever_tools(entries)}


def test_general_doc_exposes_metadata_args_without_default_scope(
    elastic_entry_spy: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(twin_ka_entries, "build_twin_scope", lambda: None)

    entries = twin_ka_entries.build_twin_ka_entries()

    assert [call["name"] for call in elastic_entry_spy] == ["general_doc"]
    general = elastic_entry_spy[0]
    assert general["default_scope"] is None
    assert general["expose_metadata_args"] == _EXPOSED_METADATA_ARGS
    assert _tool_args_by_name(entries) == {"search_general_doc": _EXPECTED_TOOL_ARGS}


def test_twin_project_doc_exposes_same_metadata_args_and_keeps_filter_scope(
    elastic_entry_spy: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = MetadataScope(entity_filter=["twin-ent"], apcode_filter=["TWIN01"])
    monkeypatch.setattr(twin_ka_entries, "build_twin_scope", lambda: scope)

    entries = twin_ka_entries.build_twin_ka_entries()

    assert [call["name"] for call in elastic_entry_spy] == ["general_doc", "twin_project_doc"]
    project = elastic_entry_spy[1]
    assert project["default_scope"] is scope
    assert project["expose_metadata_args"] == _EXPOSED_METADATA_ARGS
    assert _tool_args_by_name(entries) == {
        "search_general_doc": _EXPECTED_TOOL_ARGS,
        "search_twin_project_doc": _EXPECTED_TOOL_ARGS,
    }


def test_project_knowledge_description_steers_exclusive_corpus_for_twin_program() -> None:
    """The twin corpus description steers the planner to query ONLY this corpus
    when the user names the twin program, so a twin-program question is not also
    routed to the general documentation corpus. Guards the steer against silent
    removal (it is the planner-facing surface, not enforced in code)."""
    description = twin_ka_entries._PROJECT_KNOWLEDGE_DESCRIPTION
    assert '"twin program"' in description
    assert "ONLY corpus to search" in description

-------

tests/test_core/repositories/test_elasticsearch/test_metadata_scope.py
----
"""Unit tests for MetadataScope validators (Phase 5, Cycle 1).

Empty strings and list-with-empty-string must be rejected at construction time
so graph and direct-retriever callers share the same guard — no ES call is
issued with a degenerate filter clause.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from sta_agent_core.repositories.retrievers.elasticsearch.metadata_scope import (
    MetadataScope,
    ScopeAxis,
)


EMPTY_STRING_FIELDS = [
    "entity_filter",
    "entity_boost",
    "apcode_filter",
    "apcode_boost",
    "app_name_filter",
    "app_name_boost",
]


class TestMetadataScopeEmptyStringValidation:
    @pytest.mark.parametrize("field", EMPTY_STRING_FIELDS)
    def test_rejects_empty_string(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: ""}
        with pytest.raises(ValidationError):
            MetadataScope(**kwargs)

    @pytest.mark.parametrize("field", EMPTY_STRING_FIELDS)
    def test_rejects_list_with_empty_string(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: ["valid", ""]}
        with pytest.raises(ValidationError):
            MetadataScope(**kwargs)

    @pytest.mark.parametrize("field", EMPTY_STRING_FIELDS)
    def test_rejects_empty_list(self, field: str) -> None:
        """``[]`` silently becomes ``{"terms": {field: []}}`` — ES treats that
        as match-nothing, so every result gets dropped with no signal. The
        design doc says ``[] → None`` (reject at construction)."""
        kwargs: dict[str, Any] = {field: []}
        with pytest.raises(ValidationError, match="empty"):
            MetadataScope(**kwargs)

    @pytest.mark.parametrize("field", EMPTY_STRING_FIELDS)
    def test_accepts_none(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: None}
        scope = MetadataScope(**kwargs)
        assert getattr(scope, field) is None

    @pytest.mark.parametrize("field", EMPTY_STRING_FIELDS)
    def test_accepts_single_non_empty_string(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: "valid"}
        scope = MetadataScope(**kwargs)
        # Compared case-insensitively — apcode/app_name axes case-normalize the
        # value on the way in (covered by TestAxisNormalization). This test
        # only asserts a non-empty value is accepted.
        assert getattr(scope, field).lower() == "valid"

    @pytest.mark.parametrize("field", EMPTY_STRING_FIELDS)
    def test_accepts_list_of_non_empty_strings(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: ["a", "b"]}
        scope = MetadataScope(**kwargs)
        assert [v.lower() for v in getattr(scope, field)] == ["a", "b"]

    def test_defaults_are_none_and_flags_false(self) -> None:
        scope = MetadataScope()
        for field in EMPTY_STRING_FIELDS:
            assert getattr(scope, field) is None
        assert scope.include_entity_childs is False
        assert scope.include_transversal is False


class TestCallerScopeCombination:
    def test_caller_scope_intersects_build_time_scope(self) -> None:
        build_time = MetadataScope(doc_filter=["doc-1", "doc-2"])
        caller = MetadataScope(doc_filter=["doc-2", "doc-3"])

        combined = build_time.apply_caller_scope(caller)

        assert combined.doc_filter == "doc-2"

    def test_disjoint_scope_remains_effective_match_nothing(self) -> None:
        build_time = MetadataScope(doc_filter=["doc-1"])
        caller = MetadataScope(doc_filter=["doc-2"])

        combined = build_time.apply_caller_scope(caller)

        assert combined.matches_nothing is True
        assert combined.is_effective() is True


FIELD_MAP: dict[str, str | None] = {
    "entity_id": "metadata.entity.id",
    "entity_name": "metadata.entity.name",
    "entity_childs": "metadata.entity.childs",
    "apcode": "metadata.apcode",
    "app_name": "metadata.appName",
}


class TestMetadataScopeClauseBuilders:
    def test_empty_scope_produces_no_clauses(self) -> None:
        scope = MetadataScope()
        assert scope.build_filter_clauses(FIELD_MAP) == []
        assert scope.build_boost_clauses(FIELD_MAP) == []

    def test_single_string_filter_emits_term_clause(self) -> None:
        scope = MetadataScope(entity_filter="A")
        clauses = scope.build_filter_clauses(FIELD_MAP)
        assert {"term": {"metadata.entity.id": "A"}} in clauses

    def test_list_filter_emits_terms_clause(self) -> None:
        scope = MetadataScope(entity_filter=["A", "B"])
        clauses = scope.build_filter_clauses(FIELD_MAP)
        assert {"terms": {"metadata.entity.id": ["A", "B"]}} in clauses

    def test_boost_goes_to_boost_clauses_not_filter(self) -> None:
        scope = MetadataScope(entity_boost="A")
        assert scope.build_filter_clauses(FIELD_MAP) == []
        # H7: entity boost widens to ``entity.id OR entity.name`` so a planner-
        # emitted name (passthrough deployments) still hits a boost without
        # waiting on a resolver to lift it to an id.
        expected = {
            "bool": {
                "should": [
                    {"term": {"metadata.entity.id": "A"}},
                    {"term": {"metadata.entity.name": "A"}},
                ],
                "minimum_should_match": 1,
            }
        }
        assert expected in scope.build_boost_clauses(FIELD_MAP)

    def test_entity_boost_falls_back_to_single_field_when_name_unconfigured(self) -> None:
        """When ``entity_name`` is not in field_map, the passthrough boost
        collapses to a single ``term`` on ``entity_id``. Symmetric for the
        id-only case."""
        scope = MetadataScope(entity_boost="A")
        field_map_no_name = {k: v for k, v in FIELD_MAP.items() if k != "entity_name"}
        clauses = scope.build_boost_clauses(field_map_no_name)
        assert {"term": {"metadata.entity.id": "A"}} in clauses

    def test_entity_boost_raises_when_both_id_and_name_unconfigured(self) -> None:
        scope = MetadataScope(entity_boost="A")
        field_map_neither = {k: v for k, v in FIELD_MAP.items() if k not in {"entity_id", "entity_name"}}
        with pytest.raises(ValueError, match="entity_boost"):
            scope.build_boost_clauses(field_map_neither)

    def test_filters_across_axes_all_emitted(self) -> None:
        scope = MetadataScope(entity_filter="A", apcode_filter="X", app_name_filter="svc")
        clauses = scope.build_filter_clauses(FIELD_MAP)
        assert {"term": {"metadata.entity.id": "A"}} in clauses
        assert {"term": {"metadata.apcode": "X"}} in clauses
        assert {"term": {"metadata.appName": "svc"}} in clauses

    def test_raises_when_axis_field_not_configured(self) -> None:
        scope = MetadataScope(entity_filter="A")
        field_map: dict[str, str | None] = dict(FIELD_MAP, entity_id=None)
        with pytest.raises(ValueError, match="entity_id"):
            scope.build_filter_clauses(field_map)


class TestIncludeTransversal:
    def test_flag_on_without_apcode_filter_is_noop(self) -> None:
        """When apcode_filter is None, include_transversal=True must not add any clause."""
        scope = MetadataScope(include_transversal=True)
        assert scope.build_filter_clauses(FIELD_MAP) == []

    def test_flag_off_with_apcode_filter_emits_plain_term(self) -> None:
        scope = MetadataScope(apcode_filter="X", include_transversal=False)
        clauses = scope.build_filter_clauses(FIELD_MAP)
        assert {"term": {"metadata.apcode": "X"}} in clauses

    def test_flag_on_with_apcode_filter_widens_to_bool_should(self) -> None:
        scope = MetadataScope(apcode_filter="X", include_transversal=True)
        clauses = scope.build_filter_clauses(FIELD_MAP)
        # Plain apcode term must NOT appear — replaced by widened bool.should.
        assert {"term": {"metadata.apcode": "X"}} not in clauses
        expected = {
            "bool": {
                "should": [
                    {"term": {"metadata.apcode": "X"}},
                    {"term": {"metadata.appName": "transversal"}},
                ],
                "minimum_should_match": 1,
            }
        }
        assert expected in clauses

    def test_flag_on_with_list_apcode_filter(self) -> None:
        scope = MetadataScope(apcode_filter=["X", "Y"], include_transversal=True)
        clauses = scope.build_filter_clauses(FIELD_MAP)
        expected = {
            "bool": {
                "should": [
                    {"terms": {"metadata.apcode": ["X", "Y"]}},
                    {"term": {"metadata.appName": "transversal"}},
                ],
                "minimum_should_match": 1,
            }
        }
        assert expected in clauses

    def test_flag_on_raises_when_app_name_field_unconfigured(self) -> None:
        scope = MetadataScope(apcode_filter="X", include_transversal=True)
        field_map: dict[str, str | None] = dict(FIELD_MAP, app_name=None)
        with pytest.raises(ValueError, match="app_name"):
            scope.build_filter_clauses(field_map)


class TestIncludeEntityChilds:
    def test_flag_on_without_entity_filter_is_noop(self) -> None:
        scope = MetadataScope(include_entity_childs=True)
        assert scope.build_filter_clauses(FIELD_MAP) == []

    def test_flag_off_emits_plain_entity_id_term(self) -> None:
        scope = MetadataScope(entity_filter="A", include_entity_childs=False)
        clauses = scope.build_filter_clauses(FIELD_MAP)
        assert {"term": {"metadata.entity.id": "A"}} in clauses
        # No childs clause.
        for c in clauses:
            assert "metadata.entity.childs" not in str(c)

    def test_flag_on_widens_to_bool_should_over_id_and_childs(self) -> None:
        scope = MetadataScope(entity_filter="A", include_entity_childs=True)
        clauses = scope.build_filter_clauses(FIELD_MAP)
        expected = {
            "bool": {
                "should": [
                    {"term": {"metadata.entity.id": "A"}},
                    {"term": {"metadata.entity.childs": "A"}},
                ],
                "minimum_should_match": 1,
            }
        }
        assert expected in clauses
        # Plain entity.id term must NOT remain alongside the widened clause.
        assert {"term": {"metadata.entity.id": "A"}} not in clauses

    def test_flag_on_with_list_entity_filter(self) -> None:
        scope = MetadataScope(entity_filter=["A", "B"], include_entity_childs=True)
        clauses = scope.build_filter_clauses(FIELD_MAP)
        expected = {
            "bool": {
                "should": [
                    {"terms": {"metadata.entity.id": ["A", "B"]}},
                    {"terms": {"metadata.entity.childs": ["A", "B"]}},
                ],
                "minimum_should_match": 1,
            }
        }
        assert expected in clauses

    def test_flag_on_raises_when_entity_childs_field_unconfigured(self) -> None:
        scope = MetadataScope(entity_filter="A", include_entity_childs=True)
        field_map: dict[str, str | None] = dict(FIELD_MAP, entity_childs=None)
        with pytest.raises(ValueError, match="entity_childs"):
            scope.build_filter_clauses(field_map)


class TestIsEmptyFlagsOnlyShortCircuit:
    """Phase 5 follow-up — flags without their filter are no-ops.

    ``include_entity_childs`` widens ``entity_filter``; ``include_transversal``
    widens ``apcode_filter``. Setting a flag with no corresponding filter is a
    no-op — ``build_filter_clauses`` returns ``[]``. ``is_empty()`` must treat
    these flags-only states as empty so the graph skips scope construction.
    """

    def test_include_entity_childs_alone_is_empty(self) -> None:
        scope = MetadataScope(include_entity_childs=True)
        assert scope.is_empty() is True

    def test_include_transversal_alone_is_empty(self) -> None:
        scope = MetadataScope(include_transversal=True)
        assert scope.is_empty() is True

    def test_both_flags_alone_is_empty(self) -> None:
        scope = MetadataScope(include_entity_childs=True, include_transversal=True)
        assert scope.is_empty() is True

    def test_entity_filter_with_flag_is_not_empty(self) -> None:
        scope = MetadataScope(entity_filter="A", include_entity_childs=True)
        assert scope.is_empty() is False

    def test_apcode_filter_with_flag_is_not_empty(self) -> None:
        scope = MetadataScope(apcode_filter="X", include_transversal=True)
        assert scope.is_empty() is False

    def test_all_none_is_empty(self) -> None:
        scope = MetadataScope()
        assert scope.is_empty() is True


class TestBoostMirrorsFilterWidening:
    """Phase 5 follow-up — expansion flags must widen boost too.

    Design decision (user: no asymmetry): when ``include_entity_childs=True``,
    ``entity_boost`` widens to ``entity.id OR entity.childs``; when
    ``include_transversal=True``, ``apcode_boost`` widens to
    ``apcode OR appName=transversal``. Otherwise callers mis-reading the flag
    scope would silently get boosted only by the narrow id.
    """

    def test_include_entity_childs_widens_entity_boost(self) -> None:
        scope = MetadataScope(entity_boost="A", include_entity_childs=True)
        clauses = scope.build_boost_clauses(FIELD_MAP)
        # H7: entity boost widens to id OR name; childs flag adds the childs leg.
        expected = {
            "bool": {
                "should": [
                    {"term": {"metadata.entity.id": "A"}},
                    {"term": {"metadata.entity.childs": "A"}},
                    {"term": {"metadata.entity.name": "A"}},
                ],
                "minimum_should_match": 1,
            }
        }
        assert expected in clauses
        # Plain narrow term must NOT remain alongside the widened clause.
        assert {"term": {"metadata.entity.id": "A"}} not in clauses

    def test_include_entity_childs_widens_list_entity_boost(self) -> None:
        scope = MetadataScope(entity_boost=["A", "B"], include_entity_childs=True)
        clauses = scope.build_boost_clauses(FIELD_MAP)
        expected = {
            "bool": {
                "should": [
                    {"terms": {"metadata.entity.id": ["A", "B"]}},
                    {"terms": {"metadata.entity.childs": ["A", "B"]}},
                    {"terms": {"metadata.entity.name": ["A", "B"]}},
                ],
                "minimum_should_match": 1,
            }
        }
        assert expected in clauses

    def test_include_transversal_widens_apcode_boost(self) -> None:
        scope = MetadataScope(apcode_boost="X", include_transversal=True)
        clauses = scope.build_boost_clauses(FIELD_MAP)
        expected = {
            "bool": {
                "should": [
                    {"term": {"metadata.apcode": "X"}},
                    {"term": {"metadata.appName": "transversal"}},
                ],
                "minimum_should_match": 1,
            }
        }
        assert expected in clauses
        assert {"term": {"metadata.apcode": "X"}} not in clauses

    def test_include_entity_childs_noop_when_entity_boost_none(self) -> None:
        """Flag set, but only entity_filter (not boost) → boost remains empty."""
        scope = MetadataScope(entity_filter="A", include_entity_childs=True)
        assert scope.build_boost_clauses(FIELD_MAP) == []

    def test_include_entity_childs_widens_both_filter_and_boost_independently(self) -> None:
        """Same flag applied consistently to both surfaces when both are set."""
        scope = MetadataScope(entity_filter="A", entity_boost="B", include_entity_childs=True)
        filters = scope.build_filter_clauses(FIELD_MAP)
        boosts = scope.build_boost_clauses(FIELD_MAP)
        assert {
            "bool": {
                "should": [
                    {"term": {"metadata.entity.id": "A"}},
                    {"term": {"metadata.entity.childs": "A"}},
                ],
                "minimum_should_match": 1,
            }
        } in filters
        assert {
            "bool": {
                "should": [
                    {"term": {"metadata.entity.id": "B"}},
                    {"term": {"metadata.entity.childs": "B"}},
                    {"term": {"metadata.entity.name": "B"}},
                ],
                "minimum_should_match": 1,
            }
        } in boosts

    def test_widened_boost_raises_when_entity_childs_field_unconfigured(self) -> None:
        scope = MetadataScope(entity_boost="A", include_entity_childs=True)
        field_map: dict[str, str | None] = dict(FIELD_MAP, entity_childs=None)
        with pytest.raises(ValueError, match="entity_childs"):
            scope.build_boost_clauses(field_map)

    def test_include_transversal_widens_list_apcode_boost(self) -> None:
        """Mirror of the filter-side list test for apcode_boost."""
        scope = MetadataScope(apcode_boost=["X", "Y"], include_transversal=True)
        clauses = scope.build_boost_clauses(FIELD_MAP)
        expected = {
            "bool": {
                "should": [
                    {"terms": {"metadata.apcode": ["X", "Y"]}},
                    {"term": {"metadata.appName": "transversal"}},
                ],
                "minimum_should_match": 1,
            }
        }
        assert expected in clauses

    def test_include_transversal_widens_filter_and_boost_independently(self) -> None:
        """When BOTH apcode_filter and apcode_boost are set with the
        transversal flag, each surface widens around its own value (no
        cross-contamination of value between the two clauses)."""
        scope = MetadataScope(apcode_filter="X", apcode_boost="Y", include_transversal=True)
        filters = scope.build_filter_clauses(FIELD_MAP)
        boosts = scope.build_boost_clauses(FIELD_MAP)
        assert {
            "bool": {
                "should": [
                    {"term": {"metadata.apcode": "X"}},
                    {"term": {"metadata.appName": "transversal"}},
                ],
                "minimum_should_match": 1,
            }
        } in filters
        assert {
            "bool": {
                "should": [
                    {"term": {"metadata.apcode": "Y"}},
                    {"term": {"metadata.appName": "transversal"}},
                ],
                "minimum_should_match": 1,
            }
        } in boosts

    def test_widened_apcode_boost_raises_when_app_name_field_unconfigured(self) -> None:
        """Mirror of the filter-side missing-field test for apcode_boost."""
        scope = MetadataScope(apcode_boost="X", include_transversal=True)
        field_map: dict[str, str | None] = dict(FIELD_MAP, app_name=None)
        with pytest.raises(ValueError, match="app_name"):
            scope.build_boost_clauses(field_map)


class TestBuildTimeNormalization:
    """Axis case-normalization is applied at clause-build time, not at
    construction. This covers every value path uniformly — including the
    ``model_copy(update=...)`` / ``model_construct(...)`` escape hatches that
    bypass field validators — and lets a consumer override the per-axis
    policy for an index with a different keyword-casing convention.
    """

    def test_model_construct_apcode_normalized_at_build(self) -> None:
        """``model_construct`` bypasses validators; build-time normalization
        still uppercases the apcode so the clause matches the index."""
        scope = MetadataScope.model_construct(apcode_filter="ap90021")
        clauses = scope.build_filter_clauses(FIELD_MAP)
        assert {"term": {"metadata.apcode": "AP90021"}} in clauses

    def test_model_copy_update_app_name_normalized_at_build(self) -> None:
        """``model_copy(update=...)`` skips validators; build-time
        normalization still lowercases the app_name."""
        scope = MetadataScope().model_copy(update={"app_name_filter": "BillingSvc"})
        clauses = scope.build_filter_clauses(FIELD_MAP)
        assert {"term": {"metadata.appName": "billingsvc"}} in clauses

    def test_custom_normalizer_overrides_default_in_filter(self) -> None:
        """A caller-supplied per-axis normalizer replaces the default policy."""
        scope = MetadataScope(apcode_filter="AP90021")
        clauses = scope.build_filter_clauses(FIELD_MAP, normalizers={ScopeAxis.APCODE: str.lower})
        assert {"term": {"metadata.apcode": "ap90021"}} in clauses

    def test_custom_normalizer_overrides_default_in_boost(self) -> None:
        """Same override path applies to boost clauses."""
        scope = MetadataScope(app_name_boost="billingsvc")
        clauses = scope.build_boost_clauses(FIELD_MAP, normalizers={ScopeAxis.APP_NAME: str.upper})
        assert {"term": {"metadata.appName": "BILLINGSVC"}} in clauses

    def test_empty_normalizers_mapping_is_identity(self) -> None:
        """An empty mapping disables normalization for every axis — values
        flow into the clause exactly as stored."""
        scope = MetadataScope(apcode_filter="ap90021", app_name_filter="BillingSvc")
        clauses = scope.build_filter_clauses(FIELD_MAP, normalizers={})
        assert {"term": {"metadata.apcode": "ap90021"}} in clauses
        assert {"term": {"metadata.appName": "BillingSvc"}} in clauses

    def test_normalizer_dedupes_case_collapsed_duplicates(self) -> None:
        """When normalization collapses mixed-case members to the same token,
        the emitted ``terms`` array is order-preservingly de-duplicated."""
        scope = MetadataScope.model_construct(apcode_boost=["ap1", "AP1"])
        clauses = scope.build_boost_clauses(FIELD_MAP)
        assert {"terms": {"metadata.apcode": ["AP1"]}} in clauses

-------

tests/test_core/repositories/test_lightrag/test_lightrag_metadata_scope.py
----
"""Tests for the LightRAG-specific metadata scope."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sta_agent_core.repositories.retrievers.lightrag import LightRAGMetadataScope


def test_tag_filter_keeps_backend_operator_keys_opaque() -> None:
    scope = LightRAGMetadataScope(tag_filter={"private_operator": ["tag-a", "tag-b"]})

    assert scope.tag_filter == {"private_operator": ["tag-a", "tag-b"]}
    assert not scope.is_empty()


def test_tag_filter_detaches_mutable_input() -> None:
    raw = {"private_operator": ["tag-a"]}

    scope = LightRAGMetadataScope(tag_filter=raw)
    raw["private_operator"].append("tag-b")

    assert scope.tag_filter == {"private_operator": ["tag-a"]}


def test_empty_tag_filter_is_empty_scope() -> None:
    assert LightRAGMetadataScope().is_empty()
    assert LightRAGMetadataScope(tag_filter={}).is_empty()


def test_caller_scope_replaces_build_time_scope() -> None:
    build_time = LightRAGMetadataScope(tag_filter={"private_operator": ["build-tag"]})
    caller = LightRAGMetadataScope(tag_filter={"private_operator": ["caller-tag"]})

    assert build_time.apply_caller_scope(caller) == caller


def test_effective_scope_tracks_non_empty_tag_filter() -> None:
    assert LightRAGMetadataScope().is_effective() is False
    assert LightRAGMetadataScope(tag_filter={"private_operator": ["tag-a"]}).is_effective() is True


@pytest.mark.parametrize(
    "tag_filter",
    [
        {"operator": "tag-a"},
        {"operator": [""]},
        {"": ["tag-a"]},
    ],
)
def test_invalid_tag_filter_shape_is_rejected(tag_filter: object) -> None:
    with pytest.raises(ValidationError):
        LightRAGMetadataScope(tag_filter=tag_filter)  # type: ignore[arg-type]

-------

tests/test_core/repositories/test_lightrag/test_lightrag_retriever.py
----
"""Unit tests for LightRAGRetriever (mock HTTP engine)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from sta_agent_core.repositories.retrievers import RetrieverConnectionError, RetrieverResponseError
from sta_agent_core.repositories.retrievers.lightrag import (
    LightRAGMetadataScope,
    LightRAGRetriever,
    LightRAGSearchConfig,
)


# Patch target: AsyncHttpAdapter is imported in the HTTP engine module
_ADAPTER_PATCH = "sta_agent_core.repositories.retrievers.lightrag.engines.http_engine.AsyncHttpAdapter"


def _sample_query_data_response() -> dict:
    return {
        "status": "success",
        "data": {
            "entities": [{"entity_name": "Foo", "entity_type": "Concept"}],
            "relationships": [{"src_id": "Foo", "tgt_id": "Bar", "description": "uses"}],
            "chunks": [
                {
                    "content": "Chunk one content.",
                    "chunk_id": "c1",
                    "file_path": "/docs/a.md",
                    "reference_id": "1",
                }
            ],
            "references": [{"reference_id": "1", "file_path": "/docs/a.md"}],
        },
        "metadata": {
            "query_mode": "hybrid",
            "keywords": {"high_level": ["foo"], "low_level": ["bar"]},
            "processing_info": {
                "total_entities_found": 5,
                "entities_after_truncation": 2,
                "total_relations_found": 3,
                "relations_after_truncation": 1,
                "final_chunks_count": 1,
            },
        },
    }


class TestLightRAGRetrieverSuccess:
    """Tests for successful search and parsing."""

    @pytest.mark.asyncio
    async def test_search_returns_lightrag_search_response(self) -> None:
        payload = _sample_query_data_response()
        mock_response = httpx.Response(200, json=payload)
        mock_post = AsyncMock(return_value=mock_response)
        mock_close = AsyncMock()

        with patch(_ADAPTER_PATCH) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.post = mock_post
            mock_adapter.close = mock_close
            mock_adapter_cls.return_value = mock_adapter

            retriever = LightRAGRetriever(base_url="http://lightrag:9621")
            response = await retriever.search("test query", size=5)  # type: ignore[assignment]

        assert len(response.results) == 1
        assert response.results[0].content == "Chunk one content."
        assert response.results[0].chunk_id == "c1"
        assert response.results[0].source_url == "/docs/a.md"
        assert response.results[0].retriever_type == "lightrag"
        assert response.results[0].reference_id == "1"
        assert response.results[0].metadata["doc"] == "/docs/a.md"
        assert response.metadata.query_mode == "hybrid"
        assert response.metadata.final_chunks_count == 1
        assert len(response.entities) == 1
        assert len(response.relationships) == 1
        assert len(response.references) == 1
        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == "/query/data"
        call_json = mock_post.call_args[1]["json"]
        assert call_json["query"] == "test query"
        assert call_json["chunk_top_k"] == 5

    @pytest.mark.asyncio
    async def test_search_uses_twin_query_endpoint_when_enabled(self) -> None:
        payload = _sample_query_data_response()
        mock_post = AsyncMock(return_value=httpx.Response(200, json=payload))

        with patch(_ADAPTER_PATCH) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.post = mock_post
            mock_adapter.close = AsyncMock()
            mock_adapter_cls.return_value = mock_adapter

            retriever = LightRAGRetriever(base_url="http://lightrag:9621", use_twin_api=True)
            await retriever.search("test query")

        assert mock_post.call_args.args[0] == "/api/query/data"

    @pytest.mark.asyncio
    async def test_twin_query_endpoint_is_preserved_after_reauthentication(self) -> None:
        payload = _sample_query_data_response()
        request = httpx.Request("POST", "http://lightrag:9621/api/query/data")
        unauthorized = httpx.Response(401, request=request)
        status_error = httpx.HTTPStatusError("unauthorized", request=request, response=unauthorized)
        auth_provider = MagicMock()
        auth_provider.get_auth_headers = AsyncMock(return_value={})
        auth_provider.on_unauthorized = AsyncMock()

        with patch(_ADAPTER_PATCH) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.post = AsyncMock(side_effect=status_error)
            mock_adapter.request = AsyncMock(return_value=httpx.Response(200, json=payload))
            mock_adapter.close = AsyncMock()
            mock_adapter_cls.return_value = mock_adapter

            retriever = LightRAGRetriever(
                base_url="http://lightrag:9621",
                auth_provider=auth_provider,
                use_twin_api=True,
            )
            await retriever.search("test query")

        auth_provider.on_unauthorized.assert_awaited_once()
        assert mock_adapter.request.call_args.args[:2] == ("POST", "/api/query/data")

    @pytest.mark.asyncio
    async def test_search_maps_full_doc_id_to_page_id_metadata(self) -> None:
        """When the server enriches chunks with full_doc_id, _parse_response maps it to metadata['pageId']."""
        payload = _sample_query_data_response()
        payload["data"]["chunks"][0]["full_doc_id"] = "doc-abc123"
        payload["data"]["chunks"][0]["chunk_order_index"] = 3
        mock_post = AsyncMock(return_value=httpx.Response(200, json=payload))

        with patch(_ADAPTER_PATCH) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.post = mock_post
            mock_adapter.close = AsyncMock()
            mock_adapter_cls.return_value = mock_adapter

            retriever = LightRAGRetriever(base_url="http://lightrag:9621")
            response = await retriever.search("test query")

        chunk = response.results[0]
        assert chunk.metadata["pageId"] == "doc-abc123"
        assert chunk.metadata["chunk_index"] == 3
        assert chunk.metadata["doc"] == "/docs/a.md"

    @pytest.mark.asyncio
    async def test_search_uses_relevance_score_then_score_fallback(self) -> None:
        payload = _sample_query_data_response()
        payload["data"]["chunks"] = [
            {
                "content": "Reranked chunk",
                "chunk_id": "reranked",
                "relevance_score": 0.91,
                "score": 0.42,
            },
            {
                "content": "Zero relevance chunk",
                "chunk_id": "zero-relevance",
                "relevance_score": 0.0,
                "score": 0.55,
            },
            {
                "content": "Scored chunk",
                "chunk_id": "scored",
                "score": 0.37,
            },
        ]
        mock_engine = AsyncMock()
        mock_engine.query.return_value = payload
        retriever = LightRAGRetriever(engine=mock_engine)

        response = await retriever.search("test query")

        assert [chunk.score for chunk in response.results] == [0.91, 0.0, 0.37]

    @pytest.mark.asyncio
    async def test_search_exposes_applied_tag_filter_in_response_metadata(self) -> None:
        payload = _sample_query_data_response()
        payload["metadata"]["tag_filter"] = {"private_operator": ["tag-a"]}
        mock_engine = AsyncMock()
        mock_engine.query.return_value = payload
        retriever = LightRAGRetriever(engine=mock_engine)

        response = await retriever.search("test query")

        assert response.metadata.tags == {"private_operator": ["tag-a"]}

    @pytest.mark.asyncio
    async def test_document_chunks_use_relevance_score_then_score_fallback(self) -> None:
        mock_engine = AsyncMock()
        mock_engine.get_document_chunks.return_value = {
            "chunks": [
                {"content": "Reranked", "chunk_id": "reranked", "relevance_score": 0.81, "score": 0.31},
                {"content": "Zero relevance", "chunk_id": "zero-relevance", "relevance_score": 0.0, "score": 0.51},
                {"content": "Scored", "chunk_id": "scored", "score": 0.27},
            ]
        }
        retriever = LightRAGRetriever(engine=mock_engine)

        chunks = await retriever.get_document("doc-1")

        assert [chunk.score for chunk in chunks] == [0.81, 0.0, 0.27]

    @pytest.mark.asyncio
    async def test_search_without_full_doc_id_has_no_page_id(self) -> None:
        """When full_doc_id is absent (non-enriched server), pageId is not set."""
        payload = _sample_query_data_response()
        mock_post = AsyncMock(return_value=httpx.Response(200, json=payload))

        with patch(_ADAPTER_PATCH) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.post = mock_post
            mock_adapter.close = AsyncMock()
            mock_adapter_cls.return_value = mock_adapter

            retriever = LightRAGRetriever(base_url="http://lightrag:9621")
            response = await retriever.search("test query")

        assert "pageId" not in response.results[0].metadata
        assert "chunk_order_index" not in response.results[0].metadata

    @pytest.mark.asyncio
    async def test_default_preserves_entity_and_relationship_lineage(self) -> None:
        """With clean_response=False (default), source_id, file_path, keywords, weight are preserved."""
        payload = _sample_query_data_response()
        payload["data"]["entities"][0]["source_id"] = "chunk-abc"
        payload["data"]["entities"][0]["file_path"] = "/docs/foo.md"
        payload["data"]["relationships"][0]["weight"] = 2.0
        payload["data"]["relationships"][0]["keywords"] = "a,b,c"
        payload["data"]["relationships"][0]["source_id"] = "chunk-x"
        payload["data"]["relationships"][0]["file_path"] = "/docs/bar.md"
        mock_post = AsyncMock(return_value=httpx.Response(200, json=payload))

        with patch(_ADAPTER_PATCH) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.post = mock_post
            mock_adapter.close = AsyncMock()
            mock_adapter_cls.return_value = mock_adapter

            retriever = LightRAGRetriever(base_url="http://x")
            response = await retriever.search("q")

        assert response.entities[0].get("source_id") == "chunk-abc"
        assert response.entities[0].get("file_path") == "/docs/foo.md"
        assert response.relationships[0].get("weight") == 2.0
        assert response.relationships[0].get("keywords") == "a,b,c"
        assert response.relationships[0].get("source_id") == "chunk-x"
        assert response.relationships[0].get("file_path") == "/docs/bar.md"

    @pytest.mark.asyncio
    async def test_relationships_sorted_by_weight_descending(self) -> None:
        """Relationships are returned in descending order by weight."""
        payload = _sample_query_data_response()
        payload["data"]["relationships"] = [
            {"src_id": "A", "tgt_id": "B", "description": "low", "weight": 0.5},
            {"src_id": "C", "tgt_id": "D", "description": "high", "weight": 2.0},
            {"src_id": "E", "tgt_id": "F", "description": "mid", "weight": 1.0},
        ]
        mock_post = AsyncMock(return_value=httpx.Response(200, json=payload))

        with patch(_ADAPTER_PATCH) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.post = mock_post
            mock_adapter.close = AsyncMock()
            mock_adapter_cls.return_value = mock_adapter

            retriever = LightRAGRetriever(base_url="http://x")
            response = await retriever.search("q")

        weights = [r.get("weight", 0) for r in response.relationships]
        assert weights == [2.0, 1.0, 0.5]
        assert response.relationships[0].get("description") == "high"

    @pytest.mark.asyncio
    async def test_search_respects_config_overrides(self) -> None:
        payload = _sample_query_data_response()
        mock_post = AsyncMock(return_value=httpx.Response(200, json=payload))

        with patch(_ADAPTER_PATCH) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.post = mock_post
            mock_adapter.close = AsyncMock()
            mock_adapter_cls.return_value = mock_adapter

            config = LightRAGSearchConfig(mode="local", top_k=20)
            retriever = LightRAGRetriever(base_url="http://x", search_config=config)
            await retriever.search("q", size=7, mode="global")

        call_json = mock_post.call_args[1]["json"]
        assert call_json["mode"] == "global"
        assert call_json["chunk_top_k"] == 7

    @pytest.mark.asyncio
    async def test_search_forwards_build_time_tag_filter(self) -> None:
        mock_engine = AsyncMock()
        mock_engine.query.return_value = _sample_query_data_response()
        retriever = LightRAGRetriever(
            engine=mock_engine,
            default_scope=LightRAGMetadataScope(tag_filter={"private_operator": ["build-tag"]}),
        )

        await retriever.search("q")

        search_kwargs = mock_engine.query.call_args.args[1]
        assert search_kwargs["tag_filter"] == {"private_operator": ["build-tag"]}

    @pytest.mark.asyncio
    async def test_caller_tag_filter_replaces_build_time_filter(self) -> None:
        mock_engine = AsyncMock()
        mock_engine.query.return_value = _sample_query_data_response()
        retriever = LightRAGRetriever(
            engine=mock_engine,
            default_scope=LightRAGMetadataScope(tag_filter={"private_operator": ["build-tag"]}),
        )

        await retriever.search(
            "q",
            metadata_scope=LightRAGMetadataScope(tag_filter={"private_operator": ["caller-tag"]}),
        )

        search_kwargs = mock_engine.query.call_args.args[1]
        assert search_kwargs["tag_filter"] == {"private_operator": ["caller-tag"]}

    @pytest.mark.asyncio
    async def test_empty_caller_scope_keeps_build_time_filter(self) -> None:
        mock_engine = AsyncMock()
        mock_engine.query.return_value = _sample_query_data_response()
        retriever = LightRAGRetriever(
            engine=mock_engine,
            default_scope=LightRAGMetadataScope(tag_filter={"private_operator": ["build-tag"]}),
        )

        await retriever.search("q", metadata_scope=LightRAGMetadataScope())

        search_kwargs = mock_engine.query.call_args.args[1]
        assert search_kwargs["tag_filter"] == {"private_operator": ["build-tag"]}


class TestLightRAGRetrieverErrors:
    """Tests for connection and response errors."""

    @pytest.mark.asyncio
    async def test_raises_connection_error_on_timeout(self) -> None:
        with patch(_ADAPTER_PATCH) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_adapter.close = AsyncMock()
            mock_adapter_cls.return_value = mock_adapter

            retriever = LightRAGRetriever(base_url="http://x")

            with pytest.raises(RetrieverConnectionError, match="unreachable"):
                await retriever.search("q")

    @pytest.mark.asyncio
    async def test_raises_response_error_on_malformed_json(self) -> None:
        with patch(_ADAPTER_PATCH) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.post = AsyncMock(return_value=httpx.Response(200, json={"status": "ok", "data": None}))
            mock_adapter.close = AsyncMock()
            mock_adapter_cls.return_value = mock_adapter

            retriever = LightRAGRetriever(base_url="http://x")

            with pytest.raises(RetrieverResponseError, match="Malformed"):
                await retriever.search("q")

    @pytest.mark.asyncio
    async def test_raises_sanitized_response_error_on_422(self) -> None:
        request = httpx.Request("POST", "http://x/query/data")
        response = httpx.Response(
            422,
            text='{"detail":"invalid secret-project tag"}',
            request=request,
        )
        status_error = httpx.HTTPStatusError(
            "unprocessable entity",
            request=request,
            response=response,
        )

        with patch(_ADAPTER_PATCH) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.post = AsyncMock(side_effect=status_error)
            mock_adapter.close = AsyncMock()
            mock_adapter_cls.return_value = mock_adapter

            retriever = LightRAGRetriever(base_url="http://x")

            with pytest.raises(RetrieverResponseError, match="422") as exc_info:
                await retriever.search("q")

        assert "secret-project" not in str(exc_info.value)


class TestLightRAGRetrieverConstructor:
    """Tests for constructor and factory methods."""

    def test_requires_engine_or_base_url(self) -> None:
        with pytest.raises(ValueError, match="Either 'engine' or 'base_url'"):
            LightRAGRetriever()

    def test_accepts_engine_directly(self) -> None:
        mock_engine = MagicMock()
        retriever = LightRAGRetriever(engine=mock_engine)
        assert retriever._engine is mock_engine

    def test_from_http_creates_http_engine(self) -> None:
        with patch(_ADAPTER_PATCH):
            retriever = LightRAGRetriever.from_http(base_url="http://test:9621")
        from sta_agent_core.repositories.retrievers.lightrag.engines.http_engine import LightRAGHttpEngine

        assert isinstance(retriever._engine, LightRAGHttpEngine)

    def test_from_http_forwards_twin_api_option(self) -> None:
        with patch(_ADAPTER_PATCH):
            retriever = LightRAGRetriever.from_http(
                base_url="http://test:9621",
                use_twin_api=True,
            )

        from sta_agent_core.repositories.retrievers.lightrag.engines.http_engine import LightRAGHttpEngine

        assert isinstance(retriever._engine, LightRAGHttpEngine)
        assert retriever._engine._use_twin_api is True

    @pytest.mark.asyncio
    async def test_engine_search_delegation(self) -> None:
        """Verify search() delegates to the engine's query() method."""
        mock_engine = AsyncMock()
        mock_engine.query.return_value = _sample_query_data_response()
        retriever = LightRAGRetriever(engine=mock_engine)

        response = await retriever.search("test", size=5)

        mock_engine.query.assert_called_once()
        assert len(response.results) == 1

-------

tests/test_core/repositories/test_scope_capability_protocol.py
----
"""Tests for the ``SupportsMetadataScope`` capability protocol.

``RetrieverEntry.metadata_scope`` groups the inviolable scoping primitives,
but the ``BaseRetriever`` protocol uses
``**kwargs`` for universality, so a non-supporting retriever
would silently drop ``metadata_scope=...`` and the "ceiling" becomes a no-op —
a trust-boundary failure.

The fix is an explicit ``@runtime_checkable Protocol`` (mirrors
``DocumentProvider``) requiring the literal marker and a backend-owned caller
scope resolver. The tool factory probes via ``isinstance(...)`` at build time.
These tests pin the protocol contract; build-time enforcement is
exercised in
``tests/test_ai_engine/agents/knowledge_agent/test_retriever_tool_factory_layer3.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sta_agent_core.repositories.retrievers import SupportsMetadataScope
from sta_agent_core.repositories.retrievers.elasticsearch.elastic_retriever import (
    ElasticRetriever,
)
from sta_agent_core.repositories.retrievers.lightrag.lightrag_retriever import (
    LightRAGRetriever,
)
from sta_agent_core.repositories.retrievers.mock import MockRetriever


class TestSupportsMetadataScopeProtocol:
    """Capability is class-level, structural, and resolved via isinstance()."""

    def test_protocol_is_runtime_checkable(self) -> None:
        # Without @runtime_checkable, ``isinstance(obj, Protocol)`` raises
        # TypeError. This guard fails loud if someone strips the decorator.
        # An empty placeholder object trivially fails the structural check
        # (no ``supports_metadata_scope`` attr) — the call must not raise.
        assert isinstance(object(), SupportsMetadataScope) is False

    def test_elastic_retriever_satisfies_protocol(self) -> None:
        # Bypass __init__ — the marker is a class attribute, not an instance one.
        instance = ElasticRetriever.__new__(ElasticRetriever)
        assert isinstance(instance, SupportsMetadataScope) is True

    def test_lightrag_retriever_satisfies_protocol_for_tag_filter(self) -> None:
        instance = LightRAGRetriever.__new__(LightRAGRetriever)
        assert isinstance(instance, SupportsMetadataScope) is True
        resolved = instance.resolve_caller_scope({"doc_ids": ["doc-1"], "tag_filter": {"private_operator": ["tag-a"]}})

        assert resolved is not None
        assert resolved.tag_filter == {"private_operator": ["tag-a"]}

    def test_elastic_retriever_resolves_only_elastic_fields(self) -> None:
        instance = ElasticRetriever.__new__(ElasticRetriever)

        resolved = instance.resolve_caller_scope({"doc_ids": ["doc-1"], "tag_filter": {"private_operator": ["tag-a"]}})

        assert resolved is not None
        assert resolved.doc_filter == ["doc-1"]

    def test_mock_retriever_does_not_satisfy_protocol(self) -> None:
        # MockRetriever takes no required args, so we can construct it normally.
        assert isinstance(MockRetriever(), SupportsMetadataScope) is False

    def test_marker_is_class_level_on_elastic(self) -> None:
        # Must be in the class body (not assigned in __init__) so subclasses
        # inherit the marker without re-declaration AND so structural typing
        # works on the class itself, not just instances.
        assert "supports_metadata_scope" in vars(ElasticRetriever)
        assert ElasticRetriever.supports_metadata_scope is True

    def test_third_party_class_can_opt_in(self) -> None:
        # Documents the contract for external consumers: declare the attribute
        # at class level and the protocol passes.
        class CustomRetriever:
            supports_metadata_scope = True  # type: ignore[var-annotated]

            @staticmethod
            def resolve_caller_scope(bundle: Mapping[str, Any]) -> None:
                return None

        assert isinstance(CustomRetriever(), SupportsMetadataScope) is True

    def test_third_party_class_with_falsy_marker_still_passes_isinstance(self) -> None:
        # ``@runtime_checkable Protocol`` checks attribute *presence*, not value.
        # The Literal[True] annotation is a static-typing signal for pyright /
        # mypy — the truthiness gate at runtime is enforced by callers (see
        # the build-time check in retriever_tool_factory). This test pins the
        # documented runtime semantics so a future refactor that adds a
        # value check doesn't silently change behavior.
        class LiarRetriever:
            supports_metadata_scope = False  # type: ignore[var-annotated]

            @staticmethod
            def resolve_caller_scope(bundle: Mapping[str, Any]) -> None:
                return None

        assert isinstance(LiarRetriever(), SupportsMetadataScope) is True

    def test_marker_without_resolver_does_not_satisfy_protocol(self) -> None:
        class MarkerOnlyRetriever:
            supports_metadata_scope = True

        assert isinstance(MarkerOnlyRetriever(), SupportsMetadataScope) is False

-------

