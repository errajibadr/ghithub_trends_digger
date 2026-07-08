docs/consuming/external-agent-cards.md
----
# Publishing an agent to the TWIN orchestrator

This guide is for **partner teams who build and deploy their own agent** and want
it to show up for TWIN users — either callable **directly from the UI**, or routed
to **by the orchestrator's planner** like a first-party subagent. It's the inverse
of the other pages in this section: there you *consume* our services; here you make
*your* agent consumable.

You do this by publishing a **capability card**: a small, structured description of
what your agent does and where it may be exposed. You author it once, version it
next to your graph, and the CLI compiles it into your deploy config.

!!! info "Status — card format is stable; orchestrator routing is a future feature"
    The **card format is the stable contract** — author against it now and it will
    not change out from under you. Making your agent **UI-visible works today**.
    **Orchestrator routing** (the planner auto-discovering deployments and calling
    your agent) ships in a later release. Setting `visibility.orchestrator: true`
    today *declares intent*; it does not by itself make the planner call your agent.

## Two decisions before you start

### 1. Where should your agent appear? (`visibility`)

Your agent is exposed **nowhere** until you opt in. There is no "hide" flag — you
simply choose which surfaces to turn on. The two are independent:

```yaml
visibility:
  ui: true            # end users can pick your agent directly in the UI
  orchestrator: false # the TWIN planner may route to it as a subagent (future)
```

- **UI-visible** (`ui: true`) — your agent is listed as a standalone choice in the
  UI. **If you set this, write a strong `short_description`** — that one-liner is
  the label users read to decide whether to pick your agent. A vague one gets
  ignored; a precise one ("Search & explain application logs") gets used.
- **Orchestrator-visible** (`orchestrator: true`) — the planner may delegate to
  your agent automatically as part of answering a broader question. **This is a
  future capability** (see the status note above); set it now to declare intent,
  but routing goes live in a later release.

Both default to `false`. Turning a surface on is a *request to be considered* — our
side still ANDs it with our own registry and admission checks, so it never
auto-grants exposure. The catalog key your agent is known by is assigned by us,
never read from your card, so a card can't shadow a first-party agent.

### 2. Will the orchestrator route to it *well*? (card metadata)

For the UI, `short_description` is what matters. **For orchestrator routing, the
card's metadata is the entire signal** — the planner has nothing else to go on when
deciding whether your agent fits a request. Invest in these fields:

| Field | Why the planner needs it |
|---|---|
| `description` | The primary routing signal. Write **both** what the agent does **and when to delegate to it**. |
| `scope` | The exact domain boundary, so the planner knows your agent's precise coverage and doesn't over-route. |
| `how_to_use` | Prompting guidance, so the planner phrases queries your agent handles well. |
| `examples` | 1–3 verbatim sample queries — concrete anchors the planner matches against. |

A thin card doesn't just route poorly — an oversized or malformed one **degrades**
to a "self-reported / unverified" thin entry rather than winning routing share it
didn't earn. Run `sta agent-profile validate` to catch gaps at build time.

!!! tip "Card validators are coming"
    Today `validate` applies structural checks (schema, size caps) and completeness
    heuristics. A future update adds richer **card validators** — and later an
    admission gate that screens descriptions for over-claiming and
    instruction-injection before an agent is admitted to the planner's roster.

## Quickstart — I have an agent, create its card and Docker JSON

Say your graph is importable at `./log_agent.py:graph`. Five steps, no hand-escaping.

### 1. Scaffold a card next to your graph

```bash
sta agent-profile example --yaml > log_agent.card.yaml
```

This is the **manifest** form (the default) — your card plus its deployment
identity (name + import path) in one self-contained file you keep beside
`graph.py` in source control:

```yaml
log_agent:                          # your graph key (as it appears in langgraph.json)
  path: ./log_agent.py:graph
  card:
    description: Searches application logs and answers questions about them.
    short_description: App log search
    scope: Application logs for the X business infra
    freshness: every 15 minutes
    how_to_use: Give it an application code and a time window for best results.
    examples:
      - '"errors on app X in the last hour"'
    visibility:
      orchestrator: true
      ui: false
    tags: [logs, observability, sre]
```

### 2. Fill it in

Point `path` at your graph, and write the card against the two decisions above —
`short_description` if UI-visible, rich `description` / `scope` / `how_to_use` /
`examples` if orchestrator-visible. Commit the file next to your graph.

### 3. Validate

```bash
sta agent-profile validate log_agent.card.yaml   # add --strict to fail CI on gaps
```

```
✓ log_agent: card is valid
  no completeness suggestions — looks great.
```

### 4. Generate your deploy config — pick the one that matches how you deploy

No `--name` / `--path` flags needed: the manifest already carries them.

**a) You use `langgraph.json`** (standard LangGraph deploy):

```bash
sta agent-profile build --langgraph-json log_agent.card.yaml
```

```json
{
  "log_agent": {
    "path": "./log_agent.py:graph",
    "description": "{\"description\":\"Searches application logs ...\",\"visibility\":{\"orchestrator\":true,\"ui\":false},\"tags\":[\"logs\",\"observability\",\"sre\"]}"
  }
}
```

Merge that under `"graphs"` in your `langgraph.json`. Done.

**b) You run your own Docker image** (no `langgraph build`):

```bash
sta agent-profile build --langserve-env log_agent.card.yaml
```

```dockerfile
ENV LANGSERVE_GRAPHS='{"log_agent": {"path": "./log_agent.py:graph", "description": "{\"description\":\"Searches application logs ...\",\"visibility\":{\"orchestrator\":true,\"ui\":false}}"}}'
```

Paste that `ENV` line into your Dockerfile. See
[Deploying with your own Docker image](#deploying-with-your-own-docker-image-no-langgraph-build)
for why this works.

### 5. Deploy and verify

Deploy as usual, then confirm the card the orchestrator will read:

```bash
langgraph dev
curl "localhost:2024/a2a/<assistant_id>/.well-known/agent-card.json"
```

The `description` you see there is your compiled card — that's exactly what the
planner ingests.

## The card contract

### Authoring shapes: manifest or flat card

Both are accepted and auto-detected — author whichever fits:

| Shape | Looks like | Best for |
|---|---|---|
| **Manifest** (default, recommended) | `graph key → {path, card}` | One self-contained file next to `graph.py`, versioned. `build` needs no flags. Multiple agents can share one file. |
| **Flat card** | just the profile fields | Piping a bare card, or when name/path live elsewhere. Pass `--name` / `--path` at build time. |

```bash
sta agent-profile example --yaml         # manifest (default; name/path in the file)
sta agent-profile example --flat --yaml  # flat card (name/path via CLI flags)
```

**A manifest can hold several agents** — each top-level key is one graph. `build`
merges them into a single `langgraph.json` graphs map (or one `LANGSERVE_GRAPHS`
value); `validate` reports on each in turn:

```yaml
log_agent:
  path: ./log_agent.py:graph
  card:
    description: Searches application logs and answers questions about them.
    visibility: { orchestrator: true, ui: false }
metric_agent:
  path: ./metric_agent.py:graph
  card:
    description: Answers questions about metrics and dashboards.
    visibility: { orchestrator: true, ui: false }
```

You can author in **JSON or YAML** either way. YAML is handy for the prose fields
(block scalars) and `#` comments; the CLI detects format by extension (`.yaml` /
`.yml`), or pass `--format yaml` on stdin. The **wire format is always JSON** — the
A2A card carries a JSON string; YAML is purely an authoring convenience.

### Profile fields

| Field | Type | Required | Purpose |
|---|---|---|---|
| `description` | string | **yes** | What the agent does **and when to delegate to it** — the primary routing signal. |
| `short_description` | string | no | One-liner shown in the UI when the agent is UI-visible. Not used for routing. |
| `scope` | string | no | The exact domain boundary, e.g. `"Application logs for the X business infra"`. |
| `how_to_use` | string | no | Prompting guidance so the planner queries your agent effectively. |
| `examples` | string[] | no | 1–3 verbatim sample queries your agent handles well. |
| `freshness` | string | no | Free text — how current the data is, e.g. `"real-time"`, `"every 15 minutes"`. |
| `visibility` | object | no | `{orchestrator, ui}` — which surfaces you opt into. Both default `false`. |
| `tags` | string[] | no | Descriptive labels for discovery/search only. Never access control, never shown to the planner. |

Every field is size-capped; oversized or malformed cards degrade rather than crash
routing.

### How a card reaches the orchestrator

Your agent is deployed as its own LangGraph deployment and invoked as a
`RemoteGraph`. On stock `langgraph-api`, the **only** producer-controllable field
that reaches your agent's A2A card (`/.well-known/agent-card.json`) is the graph's
`description` string. So the card travels as a **JSON object stringified into that
`description`**:

```
your langgraph.json  ──deploy──▶  A2A agent card  ──read──▶  TWIN planner roster
  graph.description                 .description                 subagent entry
  = "<card JSON>"                   = "<card JSON>"              (routing signal)
```

Because it's JSON-inside-a-JSON-string, hand-escaping is error-prone — that's the
whole reason to compile it with `sta agent-profile build` rather than write it by
hand.

## Deploying with your own Docker image (no `langgraph build`)

The A2A card is a **runtime** feature of the langgraph-api server, not a build
artifact. On startup the server reads its graphs — including each graph's
`description` — from the **`LANGSERVE_GRAPHS`** environment variable (which
`langgraph build` populates from your `langgraph.json`) and serves the card on
demand. So if you run your own image, you just set that env yourself:

```bash
sta agent-profile build --langserve-env log_agent.card.yaml
```

The emitted `ENV LANGSERVE_GRAPHS='...'` line is byte-compatible with what
`langgraph build` bakes in. Add it to your own Dockerfile (on top of the
`langchain/langgraph-api` server base, or your adaptation of the Dockerfile from
`langgraph dockerfile`). If your deployment serves several graphs, put them all in
one manifest — `build` merges them into a single `LANGSERVE_GRAPHS` map.

!!! warning "Single quotes in a field break the Dockerfile `ENV` line"
    The JSON is single-quoted, and a literal `'` in any field (e.g. `the agent's
    logs`) cannot live inside single quotes — `docker build` will mis-parse it. The
    CLI warns when it detects one. For those cases (or for docker-compose /
    Kubernetes / `.env`), set the value at **runtime** instead of baking it in:
    ```bash
    docker run -e LANGSERVE_GRAPHS="$(cat langserve_graphs.json)" your-image
    ```
    where `langserve_graphs.json` holds the JSON value (the part inside the quotes).

A production server run also needs `DATABASE_URI` (Postgres) and `REDIS_URI` —
that's the persistence/queue the platform image expects, independent of the card.
To just verify the card locally with no Docker at all: `langgraph dev`, then
`curl "localhost:2024/a2a/<assistant_id>/.well-known/agent-card.json"`.

## Inspecting the schema

```bash
sta agent-profile schema          # the JSON Schema for the card
sta agent-profile example         # a filled-in starter manifest (default)
sta agent-profile example --flat  # a filled-in bare card
```

## What we do with your card

- **Parse and validate** it defensively — a bad card degrades, it never crashes
  routing.
- **Render only the routing-relevant fields** (`description`, `scope`, `examples`,
  `how_to_use`, `freshness`) into the planner. `visibility` and `tags` drive
  exposure/discovery and are never shown to the planner.
- **Treat every field as self-reported and unverified.** Scope claims carry a
  caveat to the planner; a forthcoming admission gate screens descriptions for
  over-claiming and instruction-injection before an agent joins the roster.

-------

docs/consuming/index.md
----
# Consuming the Library

This section is for **partner teams** who want to use our retrieval and
knowledge-graph services without running the infrastructure themselves.
Everything here assumes you are calling services that our team operates
— you don't need Elasticsearch credentials, you don't self-host LightRAG.

## Pick your entry point

```mermaid
flowchart TD
    start([I want to...]) --> q1{What do I need?}
    q1 -- raw ES chunks, I'll run my own LLM --> er[elastic_rag<br/>LangGraph Platform]
    q1 -- full evidence gathering + cited answer --> ka[knowledge_agent<br/>LangGraph Platform]
    q1 -- query the LightRAG knowledge graph<br/>as a Python object --> lr[LightRAGRetriever<br/>library import]
    er --> doc1[📘 elastic-rag.md]
    ka --> doc2[📗 knowledge-agent.md]
    lr --> doc3[📙 lightrag-http.md]
```

| Entry point | Access | Use case |
|---|---|---|
| [`elastic_rag`](elastic-rag.md) | LangGraph Platform graph | Hybrid BM25 + kNN retrieval over our managed ES index. Returns ranked chunks; you bring the LLM. |
| [`knowledge_agent`](knowledge-agent.md) | Build-your-own graph (or call a hosted one), pre-production | End-to-end evidence gathering, coverage review, answer synthesis, and citations. Covers building a KA from scratch with your own LightRAG HTTP / Elastic RAG proxy retrievers, plus calling a hosted variant. |
| [`LightRAGRetriever`](lightrag-http.md) | Python library import | Direct access to our LightRAG HTTP server as a `BaseRetriever`. Use inside your own graph or pipeline. |

### Publishing your own agent

Building an agent and want the TWIN orchestrator to route to it (rather than
consuming *our* services)? See
[**Publishing an Agent (capability cards)**](external-agent-cards.md) — how to
author, validate, and deploy the capability card that lets our planner discover
and delegate to your agent.

### Coming soon

- **`elastic_rag` — tuning & eval guide**: workflow for measuring retrieval
  quality on your query set and picking `(expansion_hint, fusion_strategy, top_k)`
  for your corpus, including AUTO threshold calibration. Until this lands,
  stick with the recommended defaults documented in `elastic-rag.md`.

## Common prerequisites

- **Credentials.** Every entry point needs per-team credentials issued by us.
  Contact details are in each doc's *Getting credentials* section.
- **Python SDKs.** If calling via Python, install `langgraph-sdk` (for LGP HTTP
  and `RemoteGraph`) and / or `sta-agent-core` (for `LightRAGRetriever`).
- **Language.** The LangGraph Platform graphs expose a plain HTTP API
  (`POST /runs/stream`) — any language works. The Python SDK is provided for
  convenience, not as a requirement.

## Conventions in these docs

- **Placeholders** look like `<ELASTIC_RAG_LGP_URL>` — theme + object.
  Find-and-replace before running any snippet.
- **Two call flavors** are shown side by side for every LGP graph:
  `langgraph_sdk.get_client` (raw HTTP over the SDK) and
  `langgraph.pregel.remote.RemoteGraph` (graph-as-an-object; good when you
  want to treat the remote graph like a local one).
- **Output shapes** are copied from the real response dicts — not mockups.
  Field names, nesting, and types match what you will parse.

-------

mkdocs.yml
----
site_name: Agent Framework by STA
site_description: Internal framework for building production-ready LangGraph agents with reusable components and best practice patterns
site_author: STA Team
site_url: !ENV [DOCS_SITE_URL, 'https://docs.example.com']  # Configurable via DOCS_SITE_URL env var
site_dir: output/site  # Build to output/ directory (already in .gitignore)

repo_name: !ENV [DOCS_REPO_NAME, 'sta-agent-packages']
repo_url: !ENV [DOCS_REPO_URL, 'https://gitlab/langgraph-agent-repo']
edit_uri: !ENV [DOCS_EDIT_URI, 'edit/main/docs/']

# Copyright
copyright: Copyright &copy; 2025 STA Team

# Configuration
theme:
  name: material
  custom_dir: docs/overrides
  palette:
    # Palette toggle for light mode
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    # Palette toggle for dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

  font:
    text: Roboto
    code: Roboto Mono

  features:
    - navigation.instant         # Instant loading
    - navigation.instant.progress  # Progress indicator
    - navigation.tracking        # Anchor tracking
    - navigation.tabs            # Top-level tabs
    - navigation.tabs.sticky     # Sticky tabs
    - navigation.sections        # Navigation sections
    - navigation.expand          # Expand navigation by default
    - navigation.path            # Breadcrumbs
    - navigation.indexes         # Section index pages
    - navigation.top             # Back to top button
    - navigation.footer          # Footer navigation (prev/next)
    - search.suggest             # Search suggestions
    - search.highlight           # Highlight search results
    - search.share               # Share search results
    - content.code.copy          # Copy button on code blocks
    - content.code.annotate      # Code annotations
    - content.tabs.link          # Link content tabs
    - toc.follow                 # Follow table of contents
    # - toc.integrate              # Integrate TOC into navigation

  icon:
    repo: fontawesome/brands/github
    admonition:
      note: fontawesome/solid/note-sticky
      abstract: fontawesome/solid/book
      info: fontawesome/solid/circle-info
      tip: fontawesome/solid/bullhorn
      success: fontawesome/solid/check
      question: fontawesome/solid/circle-question
      warning: fontawesome/solid/triangle-exclamation
      failure: fontawesome/solid/bomb
      danger: fontawesome/solid/skull
      bug: fontawesome/solid/robot
      example: fontawesome/solid/flask
      quote: fontawesome/solid/quote-left

# Plugins
plugins:
  - search:
      separator: '[\s\-,:!=\[\]()"`/]+|\.(?!\d)|&[lg]t;|(?!\b)(?=[A-Z][a-z])'
  - awesome-pages
  - include-markdown

# Extensions
markdown_extensions:
  # Python Markdown
  - abbr
  - admonition
  - attr_list
  - def_list
  - footnotes
  - md_in_html
  - toc:
      permalink: true
      permalink_title: Anchor link to this section
      toc_depth: 4

  # Python Markdown Extensions
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.betterem:
      smart_enable: all
  - pymdownx.caret
  - pymdownx.details
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.keys
  - pymdownx.mark
  - pymdownx.smartsymbols
  - pymdownx.snippets:
      check_paths: true
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
      combine_header_slug: true
  - pymdownx.tasklist:
      custom_checkbox: true
  - pymdownx.tilde

# Navigation
nav:
  - Home: index.md

  - Setup:
    - setup/index.md
    - 1. DevX & UV Setup: setup/devx-uv-guide.md
    - 2. Installation: setup/installation.md
    - 3. Code Quality: setup/code-quality.md
    - Reference - UV Lock Guide: setup/uv_lock_guide.md
    - Reference - Environment Variables: setup/environment-variables.md

  - Rights & Permissions: rights.md

  - Getting Started: getting-started.md

  - Architecture: architecture.md

  - Integrations:
    - consuming/index.md
    - elastic_rag: consuming/elastic-rag.md
    - elastic_rag — Tuning & Eval: consuming/elastic-rag-tuning.md
    - knowledge_agent: consuming/knowledge-agent.md
    - LightRAG HTTP Retriever: consuming/lightrag-http.md
    - Chat Models (create_chat_model): consuming/chat-models.md
    - Publishing an Agent (capability cards): consuming/external-agent-cards.md

  - Agent Engine:
    - agent-engine/index.md
    - Overview: agent-engine/overview.md
    - Domain Context: agent-engine/domain-context.md
    - Prompting Best Practices: agent-engine/prompting-best-practices.md
    - Middlewares: agent-engine/middlewares.md
    - States: agent-engine/states.md
    - Nodes: agent-engine/nodes.md
    - Tools: agent-engine/tools.md
    - Prompts: agent-engine/prompts.md
    - Streaming: agent-engine/streaming.md
    - Evaluations:
      - agent-engine/eval/index.md
      - Philosophy: agent-engine/eval/philosophy.md
      - Testing vs Evaluation: agent-engine/eval/testing-vs-evaluation.md
      - Evaluators: agent-engine/eval/evaluators.md
      - Datasets: agent-engine/eval/datasets.md
      - Golden Dataset: agent-engine/eval/golden-dataset.md
      - RAG Evaluation: agent-engine/eval/rag-evaluation.md
      - Experiment Analysis: agent-engine/eval/experiment-analysis.md
      - Best Practices: agent-engine/eval/best-practices.md
      - Annotation Guide: agent-engine/eval/annotation-guide.md
      - Annotation Workflow: agent-engine/eval/annotation-workflow.md
      - Online Evaluation: agent-engine/eval/online-evaluation.md
      - Implementation Guide: agent-engine/eval/implementation.md
    - Testing LangGraph Graphs: agent-engine/testing-langgraph-graphs.md
    - Troubleshooting: agent-engine/troubleshooting.md
    - Examples: agent-engine/examples.md

  - Agent Core:
    - agent-core/index.md
    - Overview: agent-core/overview.md
    - Adapters: agent-core/adapters.md
    - Repositories: agent-core/repositories.md
    - Providers: agent-core/providers.md
    - Configuration: agent-core/configuration.md
    - Extending: agent-core/extending.md
    - Examples: agent-core/examples.md

  - Frontend:
    - frontend/index.md
    - Graph Configuration: frontend/graph-configuration.md
    - UI Configuration: frontend/ui-configuration.md
    - Theme Configuration: frontend/theme-configuration.md

  - NXGraph Toolkit:
    - nxgraph-toolkit/index.md
    - CLI Usage: nxgraph-toolkit/cli-usage.md

  - Roadmap:
    - roadmap/index.md
    - Data & Knowledge: roadmap/data-and-knowledge.md
    - Agent Layer: roadmap/agent-layer.md
    - Governance & Evaluation: roadmap/governance-and-evaluation.md

  - Contributing: contributing.md
  - Deprecations: deprecations.md
  # - Deployment: DEPLOYMENT.md

# Extra CSS and JavaScript
extra_css:
  - stylesheets/extra.css

extra_javascript:
  - javascripts/extra.js
  - javascripts/mermaid.min.js
  # source is from https://unpkg.com/mermaid@10/dist/mermaid.min.js to copy in docs/javascripts/mermaid.min.js

# Extra configuration
extra:
  social:
    - icon: fontawesome/brands/github
      link: !ENV [DOCS_REPO_URL, 'https://gitlab/sta/langgraph-agent-repo']
      name: Repository

  chat_agent_url: !ENV [CHAT_AGENT_URL, 'http://localhost:8501/?embed=true&embed_options=hide_toolbar&clean=true&graph=ka_evidence_fast_lightrag']

  generator: false  # Remove "Made with Material for MkDocs"

# Validation
validation:
  nav:
    omitted_files: warn
    not_found: warn
    absolute_links: warn
  links:
    not_found: warn
    absolute_links: warn
    unrecognized_links: warn

-------

packages/sta_agent_engine/pyproject.toml
----
[project]
name = "sta-agent-engine"
version = "0.3.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "errajibadr" }
]
requires-python = ">=3.12"
dependencies = [
    "sta-agent-core",
    "httpx>=0.28.1",
    "httpx-aiohttp>=0.1.9", # Boosting perfs of LLM Calls
    "langchain[openai]>=0.3.27",
    # Bounded range: custom_chat_model.py monkey-patches langchain_openai's internal
    # _convert_dict_to_message / _convert_delta_to_message_chunk to capture reasoning
    # content. That coupling is version-fragile, so cap below the next minor; the drift
    # guard in tests/.../test_custom_chat_model.py catches any internal change within the
    # range. Raise the ceiling deliberately: review the upstream diff + refresh the guard.
    "langchain-openai>=1.2,<1.4",
    "langgraph>=1.0.0",
    "pydantic-settings>=2.10.1",
    "litellm>=1.80.0",
    "langchain-mcp-adapters>=0.1.9",
    "click>=8.1.0", # CLI framework for evaluation runner
    "pyyaml>=6.0", # YAML authoring input for the `sta agent-profile` CLI
    "tqdm>=4.66.0", # Progress bars for dataset synchronization
    "tiktoken>=0.8.0", # Token counting for context monitoring
    "deepagents>=0.6.1",
    "langchain-mistralai>=1.1.1",
    "truststore>=0.10.0", # Must be injected at app.py top before any httpx import — see app.py header
]

[project.optional-dependencies]
demo = ["sta-agent-core[demo]"]
test = [
    "aiosqlite>=0.21.0",
]
lg_sqlite = [
    "langgraph-checkpoint-sqlite>=2.0.11",
    "aiosqlite>=0.21.0",
]
eval = [
    "openevals>=0.1.0", # Eval framework used in LLM evaluators
    "agentevals>=0.0.9", # Trajectory evaluation
]
all = [
    "sta-agent-engine[eval,lg_sqlite]",
]



[project.scripts]
sta = "sta_agent_engine.cli:main"
sta-agent-engine = "sta_agent_engine:main"
sta-eval = "sta_agent_engine.evals.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "scikit-learn>=1.7.2", # Only used for Vector DB in local
]
scraping = [
    "tavily-python>=0.7.11",
    "langchain-community>=0.3.29", # Scraping urls for RAG using document_loaders.sitemap import SitemapLoader
    "lxml>=6.0.1", # Scraping website for RAG
    "bs4>=0.0.2", # Scraping website for RAG
]

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/cards/__init__.py
----
"""Agent card contract — what an agent publishes about itself.

The public, vendor-neutral surface external producers and the ``sta agent-profile``
CLI code against. Two independent concerns, both parsed from an A2A ``AgentCard``:

- :class:`AgentCapabilityProfile` — producer-authored, carried JSON-stringified in
  the card ``description``. Says *what the agent does and when to delegate to it*.
- :class:`AgentInputContract` — server-built from the graph's input schema. Says
  *how to call the agent*.

**Import-weight contract:** this package depends on pydantic and stdlib only, and
must never import engine internals (models, middleware, graph stacks, or
``CapabilityDefinition``). The producer CLI imports it, so eager exports here are
safe and keep the surface discoverable. Mapping a profile onto a consumer's own
planner types belongs to that consumer — see the orchestrator's
``sources/external_agent_card.py`` adapter. Pinned by
``tests/test_ai_engine/test_import_startup.py``.
"""

from __future__ import annotations

from .capability_profile import (
    AGENT_CAPABILITY_PROFILE_SCHEMA,
    MAX_DESCRIPTION_CHARS,
    AgentCapabilityProfile,
    SurfaceVisibility,
    agent_profile_to_description,
    extract_agent_profile,
)
from .input_contract import AgentInputContract, extract_input_contract


__all__ = [
    "AGENT_CAPABILITY_PROFILE_SCHEMA",
    "MAX_DESCRIPTION_CHARS",
    "AgentCapabilityProfile",
    "AgentInputContract",
    "SurfaceVisibility",
    "agent_profile_to_description",
    "extract_agent_profile",
    "extract_input_contract",
]

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/cards/capability_profile.py
----
"""Agent capability profile — the contract an agent publishes about itself.

An agent deployed as its own LangGraph deployment self-describes through its A2A
``AgentCard``. On stock ``langgraph-api`` the only producer-controllable field
that reaches that card is the graph's ``description`` string (the graph key
becomes the card ``name``; ``tags`` / ``examples`` / ``skills[].metadata`` are
server boilerplate). So a producer publishes an :class:`AgentCapabilityProfile`
**JSON-stringified into that ``description``**.

This module is the **public producer contract**: the schema, the JSON Schema
artifact, the builder that emits the wire string, and the parser that reads it
back. It is deliberately **dependency-light — pydantic and stdlib only**.

Nothing here may import engine internals (models, middleware, graph stacks, or
``CapabilityDefinition``). External teams and the ``sta agent-profile`` CLI import
this module, so it must stay cheap and decoupled from the graph stack. Mapping a
profile onto a *consumer's* own planner types is the consumer's job, not the
contract's — see ``agents/orchestrator/sources/external_agent_card.py`` for the
orchestrator's adapter.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError


logger = logging.getLogger(__name__)

# Size caps. "Size attracts" in LLM routing — an unbounded card would win routing
# share it did not earn — so every field is bounded. A profile that exceeds a cap
# fails validation; a consumer decides how to degrade (it must never raise).
#
# The description cap is public: consumers that keep a card's raw description on a
# degraded path need the same bound to truncate against.
MAX_DESCRIPTION_CHARS = 2000
_MAX_SHORT_DESCRIPTION = 200
_MAX_SCOPE = 500
_MAX_FRESHNESS = 100
_MAX_HOW_TO_USE = 2000
_MAX_EXAMPLE = 500
_MAX_EXAMPLES = 10
_MAX_TAG = 40
_MAX_TAGS = 20

_ExampleStr = Annotated[str, StringConstraints(max_length=_MAX_EXAMPLE)]
_TagStr = Annotated[str, StringConstraints(max_length=_MAX_TAG)]


class SurfaceVisibility(BaseModel):
    """Which surfaces an agent *requests* exposure on.

    Opt-in and self-reported: both flags default ``False``, so an agent is
    exposed nowhere until it explicitly asks. Symmetric defaults keep the model
    simple (there is no "hide" — an agent just does not opt in) and least-
    exposure by default (nothing leaks from a large deployments list).

    This is a *request*, not an entitlement. A consumer **ANDs** it with its own
    registry and admission gate before actually routing to (or listing) the agent
    — a producer setting ``orchestrator=True`` does not auto-grant.

    ``extra="ignore"`` so a producer naming a future surface (e.g. a public API)
    does not break parsing on an older consumer.
    """

    model_config = ConfigDict(extra="ignore")

    orchestrator: bool = False
    """Requests to be callable by an orchestrator planner as a subagent."""

    ui: bool = False
    """Requests to be callable standalone directly from the UI."""


class AgentCapabilityProfile(BaseModel):
    """The structured contract an agent producer publishes.

    JSON-stringified into the deployment's graph ``description`` (see module
    docstring). ``extra="ignore"`` so a producer adding their own keys does not
    break parsing; every field is size-capped so an oversized profile degrades
    rather than dominating planner routing.
    """

    model_config = ConfigDict(extra="ignore")

    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_CHARS)
    """What the agent does + when to delegate to it (``use_for`` merged in)."""

    short_description: str | None = Field(default=None, max_length=_MAX_SHORT_DESCRIPTION)
    """One-liner for standalone-agent UI display. NOT used for routing."""

    scope: str | None = Field(default=None, max_length=_MAX_SCOPE)
    """Exact domain boundary, e.g. "Application logs for the X business infra"."""

    freshness: str | None = Field(default=None, max_length=_MAX_FRESHNESS)
    """Free-text data freshness, e.g. "real-time", "daily", "every 15 minutes".

    Surfaced to a planner as a note so it can weigh the answer's recency.
    """

    how_to_use: str | None = Field(default=None, max_length=_MAX_HOW_TO_USE)
    """Best-practice prompting/query guidance — distinct from ``examples``."""

    examples: list[_ExampleStr] = Field(default_factory=list, max_length=_MAX_EXAMPLES)
    """Verbatim sample user queries this agent handles well."""

    visibility: SurfaceVisibility = Field(default_factory=SurfaceVisibility)
    """Requested exposure surfaces (opt-in, both default ``False``).

    Discovery filters on ``visibility.orchestrator``; a UI catalog filters on
    ``visibility.ui``. **Not** a routing signal — consumers must not render it
    into planner text.
    """

    tags: list[_TagStr] = Field(default_factory=list, max_length=_MAX_TAGS)
    """Descriptive labels for discovery / search / categorization only.

    Never access control (that is ``visibility``) and never rendered into planner
    routing text.
    """


# Published JSON Schema artifact — producers generate/validate their profile
# against this instead of guessing the shape.
AGENT_CAPABILITY_PROFILE_SCHEMA: dict[str, Any] = AgentCapabilityProfile.model_json_schema()


def _summarize_validation_error(exc: ValidationError) -> str:
    """Summarize field locations + error types without echoing untrusted values.

    Card content is producer-controllable and, at a consumer, attacker-
    controllable — so we log *where* and *what kind* of error occurred (e.g.
    ``description: string_too_long``) but never the offending value.
    """
    return "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['type']}" for err in exc.errors())


def _try_parse_profile(raw: Any, *, card_name: Any) -> AgentCapabilityProfile | None:
    """Parse one candidate into a profile, or ``None`` if it is not one.

    A non-JSON string is treated as a plain free-text description (the expected
    shape for an agent that publishes no profile) and is rejected *silently*. A
    value that parses to a JSON object but fails validation is a producer that
    tried and got the schema wrong — logged at ``warning`` so they can fix it.
    """
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None  # plain text description — not a profile, expected.
    if not isinstance(data, Mapping):
        return None
    try:
        return AgentCapabilityProfile.model_validate(data)
    except ValidationError as exc:
        logger.warning(
            "agent card %r carries a malformed/oversized capability profile; degrading to thin capability: %s",
            card_name,
            _summarize_validation_error(exc),
        )
        return None


def extract_agent_profile(card: Mapping[str, Any]) -> AgentCapabilityProfile | None:
    """Extract an :class:`AgentCapabilityProfile` from an A2A agent card.

    The profile is carried as a JSON object inside the card ``description`` — the
    only producer-controllable field that reaches the card on stock
    ``langgraph-api`` (``skills[].metadata`` is built server-side from the input
    schema and cannot be set by a producer; see the module docstring).

    Args:
        card: A parsed A2A ``AgentCard`` (the JSON dict from
            ``/.well-known/agent-card.json``).

    Returns:
        The validated profile, or ``None`` when no valid profile is present
        (plain-text description, malformed JSON, or a profile that fails the
        size/shape constraints).
    """
    description = card.get("description")
    if not isinstance(description, str):
        return None
    return _try_parse_profile(description, card_name=card.get("name"))


def agent_profile_to_description(
    *,
    description: str,
    scope: str | None = None,
    freshness: str | None = None,
    how_to_use: str | None = None,
    examples: list[str] | None = None,
    short_description: str | None = None,
    visibility: SurfaceVisibility | Mapping[str, bool] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Build the JSON ``description`` string a producer puts in ``langgraph.json``.

    Producers hand-writing the profile face JSON-inside-JSON escaping; this helper
    validates the fields against :class:`AgentCapabilityProfile` and returns the
    canonical JSON string, so teams generate it instead of escaping by hand.
    Raises :class:`pydantic.ValidationError` on invalid input — this is the
    *producer* side (trusted, build-time), unlike a consumer adapter which must
    degrade gracefully.

    ``visibility`` accepts either a :class:`SurfaceVisibility` or a plain
    ``{"orchestrator": ..., "ui": ...}`` mapping; when omitted it defaults to both
    surfaces opt-out. The emitted JSON always carries the explicit ``visibility``
    object so an operator can see exactly what the agent opts into.

    Returns:
        A compact JSON string suitable for the graph ``description`` field.
    """
    if visibility is None:
        resolved_visibility = SurfaceVisibility()
    elif isinstance(visibility, SurfaceVisibility):
        resolved_visibility = visibility
    else:
        resolved_visibility = SurfaceVisibility.model_validate(visibility)
    profile = AgentCapabilityProfile(
        description=description,
        short_description=short_description,
        scope=scope,
        freshness=freshness,
        how_to_use=how_to_use,
        examples=examples or [],
        visibility=resolved_visibility,
        tags=tags or [],
    )
    return profile.model_dump_json(exclude_none=True)


__all__ = [
    "AGENT_CAPABILITY_PROFILE_SCHEMA",
    "MAX_DESCRIPTION_CHARS",
    "AgentCapabilityProfile",
    "SurfaceVisibility",
    "agent_profile_to_description",
    "extract_agent_profile",
]

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/cards/input_contract.py
----
"""Agent input contract — parsed from an A2A card's server-built ``inputSchema``.

Unlike the capability *profile* (which a producer hand-authors into the card
``description``), this is **server-built** by ``generate_agent_card`` from the
graph's input schema, so it is always present and trustworthy.

Dependency-light by contract: stdlib only. See ``capability_profile`` for why.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentInputContract:
    """The agent's input-field contract, from ``skills[].metadata.inputSchema``.

    Describes *how to call* the agent, not *when to route* to it:

    - ``properties`` — every accepted input field name (e.g. ``query``,
      ``doc_ids``, ``apcode_filter``). Lets a caller know which scope/filter
      channels it can bridge to this agent.
    - ``required`` — the subset that must be supplied.
    - ``supports_a2a`` — the server's ``"messages" in properties`` flag: the agent
      accepts the ``{"messages": [...]}`` shape ``task``/A2A send.
    """

    properties: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    supports_a2a: bool = False

    def accepts(self, field_name: str) -> bool:
        """Whether the agent accepts an input field of this name."""
        return field_name in self.properties


def extract_input_contract(card: Mapping[str, Any]) -> AgentInputContract | None:
    """Parse the server-built input contract from an A2A agent card.

    Reads the first ``skills[].metadata.inputSchema`` block. Defensive against
    shape drift (missing/renamed keys yield empty tuples rather than raising).

    Args:
        card: A parsed A2A ``AgentCard``.

    Returns:
        The :class:`AgentInputContract`, or ``None`` if no skill carries an
        ``inputSchema`` (e.g. a non-LangGraph card).
    """
    skills = card.get("skills")
    if not isinstance(skills, list):
        return None
    for skill in skills:
        if not isinstance(skill, Mapping):
            continue
        metadata = skill.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        input_schema = metadata.get("inputSchema")
        if not isinstance(input_schema, Mapping):
            continue
        raw_properties = input_schema.get("properties")
        raw_required = input_schema.get("required")
        return AgentInputContract(
            properties=tuple(p for p in raw_properties if isinstance(p, str)) if isinstance(raw_properties, list) else (),
            required=tuple(r for r in raw_required if isinstance(r, str)) if isinstance(raw_required, list) else (),
            supports_a2a=bool(input_schema.get("supportsA2A", False)),
        )
    return None


__all__ = ["AgentInputContract", "extract_input_contract"]

-------

packages/sta_agent_engine/src/sta_agent_engine/cli/__init__.py
----
"""``sta`` — helper CLI for third-party agent developers.

The umbrella command for building and validating agents that plug into the TWIN
orchestrator. Registered as the ``sta`` console script (see ``pyproject.toml``);
also runnable as ``python -m sta_agent_engine.cli``.

Command groups:
    sta agent-profile   build / validate an external agent's capability profile
"""

from __future__ import annotations

from sta_agent_engine.cli.app import main


__all__ = ["main"]

-------

packages/sta_agent_engine/src/sta_agent_engine/cli/__main__.py
----
"""Enable ``python -m sta_agent_engine.cli``."""

from __future__ import annotations

from sta_agent_engine.cli.app import main


if __name__ == "__main__":
    main()

-------

packages/sta_agent_engine/src/sta_agent_engine/cli/agent_profile.py
----
"""``sta agent-profile`` — build, validate, and inspect a capability profile.

An external agent self-describes to the TWIN orchestrator by publishing a
capability profile as a JSON object **stringified into its graph's
``description``** field in ``langgraph.json`` — the only producer-controllable
field that reaches the A2A agent card on stock ``langgraph-api``. Because the
description is itself a string, the profile becomes JSON-inside-JSON and
hand-escaping is error-prone. This command validates the profile and emits
exactly what to paste, escaping handled.

Two authoring shapes are accepted, auto-detected:

- **Flat card** — a bare profile (``description``, ``scope``, …) at the top
  level. The graph ``name`` / ``path`` are supplied as CLI flags at build time.
- **Manifest** — a map of graph key → ``{path, card}`` mirroring
  ``langgraph.json``'s ``graphs`` block, but with the card as a readable nested
  object instead of a stringified blob. Self-contained (name + path travel with
  the card) so a single file lives next to ``graph.py`` in source control, and
  ``build`` needs no ``--name`` / ``--path`` flags. The CLI compiles it down to
  the escaped ``langgraph.json`` entry or ``LANGSERVE_GRAPHS`` env value.

The pure functions (``load_profile``, ``build_description``,
``profile_warnings``, ``to_langgraph_entry``, ``is_manifest``, ``parse_manifest``,
``to_langgraph_graphs``) carry the logic and are unit-tested directly; the Click
commands are thin I/O wrappers.

Future improvement (not implemented): an optional ``--judge`` mode that sends the
rendered description to an LLM judge to assess whether it is complete and
unambiguous enough for reliable routing — going beyond the structural heuristics
in :func:`profile_warnings`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import click
import yaml
from pydantic import ValidationError

from sta_agent_engine.agents.cards import (
    AGENT_CAPABILITY_PROFILE_SCHEMA,
    AgentCapabilityProfile,
)


_EXAMPLE_PROFILE: dict[str, Any] = {
    "description": "Searches application logs and answers questions about them.",
    "short_description": "App log search",
    "scope": "Application logs for the X business infra",
    "freshness": "every 15 minutes",
    "how_to_use": "Give it an application code and a time window for best results.",
    "examples": ['"errors on app X in the last hour"'],
    "visibility": {"orchestrator": True, "ui": False},
    "tags": ["logs", "observability", "sre"],
}

# The manifest form: the flat card nested under a graph key alongside its import
# path — the self-contained shape a producer versions next to ``graph.py``.
_EXAMPLE_MANIFEST: dict[str, Any] = {
    "log_agent": {
        "path": "./log_agent.py:graph",
        "card": _EXAMPLE_PROFILE,
    }
}

# Below this, a description rarely carries both "what it does" and "when to
# delegate" — the two things the planner routes on.
_MIN_DESCRIPTION_CHARS = 40


def load_profile(text: str, *, fmt: str = "json") -> dict[str, Any]:
    """Parse profile text (JSON or YAML) into a dict.

    Args:
        text: The raw profile document.
        fmt: ``"json"`` (default) or ``"yaml"``.

    Raises:
        ValueError: input is not valid JSON/YAML, or is not an object/mapping.
    """
    if fmt == "yaml":
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"input is not valid YAML: {exc}") from exc
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"input is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("profile must be an object")
    return data


def build_description(data: dict[str, Any]) -> tuple[str, list[str]]:
    """Validate a profile dict and return ``(description_string, unknown_keys)``.

    Unknown keys mirror the consumer's ``extra="ignore"`` behaviour (reported,
    not fatal — a producer typo is worth surfacing). Validation goes through the
    pydantic model so a missing/oversized/wrong-type field (e.g. an omitted
    required ``description``) surfaces as a ``ValidationError`` the commands
    translate to a clean error — never a raw ``TypeError``.

    Raises:
        pydantic.ValidationError: a field is missing/oversized/wrong type.
    """
    known = set(AgentCapabilityProfile.model_fields)
    unknown = sorted(set(data) - known)
    filtered = {key: value for key, value in data.items() if key in known}
    profile = AgentCapabilityProfile.model_validate(filtered)
    return profile.model_dump_json(exclude_none=True), unknown


def _exposed_on_any_surface(visibility: Any) -> bool:
    """Whether the profile opts into at least one exposure surface."""
    if not isinstance(visibility, Mapping):
        return False
    return bool(visibility.get("orchestrator")) or bool(visibility.get("ui"))


def profile_warnings(data: dict[str, Any]) -> list[str]:
    """Heuristic completeness suggestions (improvements, not hard errors).

    Each absent-or-thin field weakens planner routing in a specific way; the
    suggestion names the consequence so a producer knows why it matters.
    """
    suggestions: list[str] = []
    description = data.get("description")
    if isinstance(description, str) and len(description.strip()) < _MIN_DESCRIPTION_CHARS:
        suggestions.append(f"description is short (<{_MIN_DESCRIPTION_CHARS} chars) — describe BOTH what the agent does and when to delegate to it.")
    if not data.get("scope"):
        suggestions.append("no scope — add the exact domain boundary so the planner knows the agent's precise coverage.")
    if not data.get("how_to_use"):
        suggestions.append("no how_to_use — add prompting guidance so the planner queries the agent effectively.")
    if not data.get("examples"):
        suggestions.append("no examples — add 1-3 verbatim sample queries to anchor routing.")
    if not data.get("freshness"):
        suggestions.append("no freshness — state how current the data is (e.g. 'real-time', 'daily', 'every 15 minutes').")
    if not _exposed_on_any_surface(data.get("visibility")):
        suggestions.append("not exposed on any surface — set visibility.orchestrator and/or visibility.ui to true, or the agent won't be discovered.")
    if not data.get("tags"):
        suggestions.append("no tags — add a few descriptive tags (e.g. 'logs', 'sre') to aid discovery and search.")
    return suggestions


def to_langgraph_entry(name: str, path: str, description: str) -> dict[str, Any]:
    """Build the ``langgraph.json`` graph entry ``{name: {path, description}}``.

    ``json.dumps`` of the returned dict escapes the embedded JSON description
    correctly, so producers never escape by hand.
    """
    return {name: {"path": path, "description": description}}


def to_langserve_graphs_value(name: str, path: str, description: str) -> str:
    """The ``LANGSERVE_GRAPHS`` env *value*: a compact one-graph JSON map.

    The A2A card is served at runtime from the ``LANGSERVE_GRAPHS`` env var (the
    server reads its graphs — and their ``description`` — from it at startup), so
    setting this env is all a custom image needs. It is byte-compatible with what
    ``langgraph build`` bakes in, so you can run your own Docker image / compose /
    Kubernetes without ``langgraph build`` and still serve the same card.
    """
    return json.dumps(to_langgraph_entry(name, path, description))


def to_dockerfile_env_line(value: str) -> str:
    """Wrap a ``LANGSERVE_GRAPHS`` value as a paste-ready Dockerfile ``ENV`` line.

    Matches the single-quoted form ``langgraph build`` emits. Single quotes keep
    the JSON's escaped inner quotes literal — but a literal ``'`` in the value
    cannot live inside them, so the caller guards against it.
    """
    return f"ENV LANGSERVE_GRAPHS='{value}'"


def to_langgraph_graphs(built: list[tuple[str, str, str]]) -> dict[str, Any]:
    """Merge ``(name, path, description)`` triples into a langgraph.json graphs map.

    Single-entry output is identical to :func:`to_langgraph_entry`; multi-entry
    merges every agent under one map — the shape both ``langgraph.json``'s
    ``graphs`` block and the ``LANGSERVE_GRAPHS`` env value expect.
    """
    graphs: dict[str, Any] = {}
    for name, path, description in built:
        graphs[name] = {"path": path, "description": description}
    return graphs


@dataclass(frozen=True)
class ManifestEntry:
    """One agent in a manifest: a graph ``name`` → import ``path`` + raw ``card``."""

    name: str
    path: str | None
    card: dict[str, Any]


def is_manifest(data: Mapping[str, Any]) -> bool:
    """Whether ``data`` is a manifest (graph key → {path, card}) vs a flat card.

    A manifest maps each graph name to a spec containing a nested ``card``. A flat
    profile carries profile fields (``description`` etc.) at the top level, so its
    values are never *all* card-bearing mappings — the two shapes are unambiguous.
    """
    return bool(data) and all(isinstance(value, Mapping) and "card" in value for value in data.values())


def parse_manifest(data: Mapping[str, Any]) -> list[ManifestEntry]:
    """Split a manifest into per-agent entries.

    Raises:
        ValueError: an agent's ``card`` is not a mapping, or ``path`` is not a string.
    """
    entries: list[ManifestEntry] = []
    for name, spec in data.items():
        card = spec.get("card")
        if not isinstance(card, Mapping):
            raise ValueError(f"agent {name!r}: 'card' must be a mapping")
        path = spec.get("path")
        if path is not None and not isinstance(path, str):
            raise ValueError(f"agent {name!r}: 'path' must be a string")
        entries.append(ManifestEntry(name=str(name), path=path, card=dict(card)))
    return entries


def _detect_format(filename: str) -> str:
    """Pick the parser by file extension; stdin / unknown defaults to JSON."""
    return "yaml" if filename.lower().endswith((".yaml", ".yml")) else "json"


def _resolve_format(choice: str, filename: str) -> str:
    """Resolve an ``--format`` choice (``auto`` → detect by extension)."""
    return _detect_format(filename) if choice == "auto" else choice


def _load_or_abort(text: str, fmt: str) -> dict[str, Any]:
    try:
        return load_profile(text, fmt=fmt)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _warn_if_single_quote(value: str) -> None:
    """Warn when a card field holds a ``'`` that would break the ENV line."""
    if "'" in value:
        click.echo(
            "warning: a card field contains a single quote (') — a single-quoted Dockerfile\n"
            "         ENV line will not parse. Set the env at runtime instead, e.g.\n"
            '         docker run -e LANGSERVE_GRAPHS="$(cat langserve_graphs.json)" your-image',
            err=True,
        )


def _manifest_entries_or_abort(data: Mapping[str, Any], name: str | None = None, path: str | None = None) -> list[ManifestEntry]:
    """Parse a manifest, applying single-agent ``--name`` / ``--path`` overrides.

    Overrides target the one-agent-per-file case; supplying them for a multi-agent
    manifest is ambiguous and rejected.
    """
    try:
        entries = parse_manifest(data)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if name or path:
        if len(entries) != 1:
            raise click.UsageError("--name/--path can only override a single-agent manifest")
        only = entries[0]
        entries = [ManifestEntry(name=name or only.name, path=path or only.path, card=only.card)]
    return entries


@click.group(name="agent-profile")
def agent_profile() -> None:
    """Build, validate, and inspect an external agent's capability profile.

    The profile is the JSON object an external agent publishes (stringified into
    its langgraph.json ``description``) so the TWIN orchestrator can route to it.
    """


@agent_profile.command()
@click.argument("file", type=click.File("r"), default="-")
@click.option("--strict", is_flag=True, help="Exit non-zero if any completeness suggestions remain.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["auto", "json", "yaml"]),
    default="auto",
    help="Input format (auto = by file extension; stdin defaults to JSON).",
)
def validate(file: click.utils.LazyFile, strict: bool, fmt: str) -> None:
    """Validate a profile and report completeness suggestions.

    FILE is a flat card or a manifest, JSON or YAML (default: stdin). Reports hard
    errors (invalid schema / oversized fields), unknown keys, and improvement
    suggestions — per agent for a manifest.
    """
    data = _load_or_abort(file.read(), _resolve_format(fmt, file.name))

    if is_manifest(data):
        _validate_manifest(data, strict=strict)
        return

    try:
        _, unknown = build_description(data)
    except ValidationError as exc:
        raise click.ClickException(f"invalid profile:\n{exc}") from exc

    click.echo("✓ profile is valid")
    for key in unknown:
        click.echo(f"  unknown field ignored: {key!r}", err=True)

    suggestions = profile_warnings(data)
    for suggestion in suggestions:
        click.echo(f"  suggestion: {suggestion}", err=True)

    if not suggestions:
        click.echo("  no completeness suggestions — looks great.")
    elif strict:
        raise SystemExit(1)


def _validate_manifest(data: Mapping[str, Any], *, strict: bool) -> None:
    """Validate each agent's card in a manifest, reporting per-agent."""
    entries = _manifest_entries_or_abort(data)
    any_suggestions = False
    for entry in entries:
        try:
            _, unknown = build_description(entry.card)
        except ValidationError as exc:
            raise click.ClickException(f"agent {entry.name!r}: invalid card:\n{exc}") from exc

        click.echo(f"✓ {entry.name}: card is valid")
        for key in unknown:
            click.echo(f"  {entry.name}: unknown field ignored: {key!r}", err=True)

        suggestions = list(profile_warnings(entry.card))
        if not entry.path:
            suggestions.insert(0, "no path — add the graph import path (e.g. ./agent.py:graph).")
        for suggestion in suggestions:
            click.echo(f"  {entry.name}: suggestion: {suggestion}", err=True)
        any_suggestions = any_suggestions or bool(suggestions)

    if not any_suggestions:
        click.echo("  no completeness suggestions — looks great.")
    elif strict:
        raise SystemExit(1)


@agent_profile.command()
@click.argument("file", type=click.File("r"), default="-")
@click.option(
    "--langgraph-json",
    "as_entry",
    is_flag=True,
    help="Emit a ready-to-paste langgraph.json graphs entry (flat card needs --name/--path; a manifest supplies them).",
)
@click.option(
    "--langserve-env",
    "as_langserve_env",
    is_flag=True,
    help="Emit a Dockerfile `ENV LANGSERVE_GRAPHS=...` line to run your own image without `langgraph build` (flat card needs --name/--path; a manifest supplies them).",
)
@click.option("--name", help="Graph key for the entry (flat card; overrides a single-agent manifest).")
@click.option("--path", "graph_path", help="Graph import path, e.g. ./agent.py:graph (flat card; overrides a single-agent manifest).")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["auto", "json", "yaml"]),
    default="auto",
    help="Input format (auto = by file extension; stdin defaults to JSON).",
)
def build(file: click.utils.LazyFile, as_entry: bool, as_langserve_env: bool, name: str | None, graph_path: str | None, fmt: str) -> None:
    """Emit the capability string to paste into your deploy config.

    Accepts either a **flat card** or a **manifest** (graph key → {path, card}),
    auto-detected.

    Flat card — three output shapes (default: the bare ``description`` string):

    - ``--langgraph-json`` — the ``langgraph.json`` graphs entry (needs ``--name``/``--path``).
    - ``--langserve-env`` — a Dockerfile ``ENV LANGSERVE_GRAPHS='...'`` line so a
      custom image serves the A2A card without ``langgraph build`` (needs ``--name``/``--path``).

    Manifest — name/path travel with the card, so no flags are needed. The default
    (and ``--langgraph-json``) emit the graphs map; ``--langserve-env`` emits the
    Dockerfile line covering every agent in the manifest.

    FILE (JSON or YAML) defaults to stdin. The emitted description is always JSON
    — the wire format the A2A card carries — regardless of the input format.
    """
    if as_entry and as_langserve_env:
        raise click.UsageError("choose one of --langgraph-json or --langserve-env, not both")

    data = _load_or_abort(file.read(), _resolve_format(fmt, file.name))

    if is_manifest(data):
        _build_manifest(data, as_langserve_env=as_langserve_env, name=name, path=graph_path)
        return

    if (as_entry or as_langserve_env) and not (name and graph_path):
        raise click.UsageError("--langgraph-json / --langserve-env requires --name and --path")

    try:
        description, unknown = build_description(data)
    except ValidationError as exc:
        raise click.ClickException(f"invalid profile:\n{exc}") from exc

    for key in unknown:
        click.echo(f"warning: ignoring unknown field {key!r}", err=True)

    if as_langserve_env:
        assert name is not None and graph_path is not None  # guarded above
        value = to_langserve_graphs_value(name, graph_path, description)
        _warn_if_single_quote(value)
        click.echo(to_dockerfile_env_line(value))
    elif as_entry:
        assert name is not None and graph_path is not None  # guarded above
        click.echo(json.dumps(to_langgraph_entry(name, graph_path, description), indent=2, ensure_ascii=False))
    else:
        click.echo(description)


def _build_manifest(data: Mapping[str, Any], *, as_langserve_env: bool, name: str | None, path: str | None) -> None:
    """Compile a manifest into a langgraph.json graphs map or a Dockerfile ENV line.

    ``name`` / ``path`` come from the file; the flags override only for a
    single-agent manifest. The default (and ``--langgraph-json``) both emit the
    graphs map, since a manifest already carries every agent's path.
    """
    entries = _manifest_entries_or_abort(data, name=name, path=path)
    built: list[tuple[str, str, str]] = []
    for entry in entries:
        if not entry.path:
            raise click.ClickException(
                f"agent {entry.name!r}: missing 'path' — add it under the agent in the manifest (or pass --path for a single-agent manifest)."
            )
        try:
            description, unknown = build_description(entry.card)
        except ValidationError as exc:
            raise click.ClickException(f"agent {entry.name!r}: invalid card:\n{exc}") from exc
        for key in unknown:
            click.echo(f"warning: agent {entry.name!r}: ignoring unknown field {key!r}", err=True)
        built.append((entry.name, entry.path, description))

    if as_langserve_env:
        value = json.dumps(to_langgraph_graphs(built))
        _warn_if_single_quote(value)
        click.echo(to_dockerfile_env_line(value))
    else:  # default and --langgraph-json both emit the graphs map
        click.echo(json.dumps(to_langgraph_graphs(built), indent=2, ensure_ascii=False))


@agent_profile.command()
def schema() -> None:
    """Print the capability-profile JSON schema."""
    click.echo(json.dumps(AGENT_CAPABILITY_PROFILE_SCHEMA, indent=2))


@agent_profile.command()
@click.option("--flat", "as_flat", is_flag=True, help="Emit a bare card instead — supply --name/--path at build time.")
@click.option("--yaml", "as_yaml", is_flag=True, help="Emit the example as YAML (add # comments once you edit it).")
def example(as_flat: bool, as_yaml: bool) -> None:
    """Print a starter example to fill in.

    Defaults to the self-contained **manifest** (graph key → path + card) — the
    recommended shape, versioned next to your graph. Use ``--flat`` for a bare card
    (name/path passed as build flags). Combine with ``--yaml`` for the readable,
    commentable authoring format.
    """
    payload = _EXAMPLE_PROFILE if as_flat else _EXAMPLE_MANIFEST
    if as_yaml:
        click.echo(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).rstrip())
    else:
        click.echo(json.dumps(payload, indent=2))


__all__ = [
    "ManifestEntry",
    "agent_profile",
    "build_description",
    "is_manifest",
    "load_profile",
    "parse_manifest",
    "profile_warnings",
    "to_dockerfile_env_line",
    "to_langgraph_entry",
    "to_langgraph_graphs",
    "to_langserve_graphs_value",
]

-------

packages/sta_agent_engine/src/sta_agent_engine/cli/app.py
----
"""Top-level ``sta`` Click group and command-group registration."""

from __future__ import annotations

import click

from sta_agent_engine import __version__
from sta_agent_engine.cli.agent_profile import agent_profile


@click.group()
@click.version_option(version=__version__, prog_name="sta")
def main() -> None:
    """STA helper CLI for building agents that plug into the TWIN orchestrator."""


main.add_command(agent_profile)


__all__ = ["main"]

-------

