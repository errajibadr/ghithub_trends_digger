# Reasoning changes bundle

Generated from the files changed by commit `504558af` (`feat(models): add
reasoning mappings for GLM DeepSeek and Qwen`), including the current Gemma 4
updates. Test files, creative phases, and unrelated working-tree changes are
excluded.

`.env.example`
----
# ==============================================================================
# .env.example - Environment Configuration Template
# ==============================================================================
# Copy this file to .env and fill in your actual values
# SECURITY: Never commit .env files with real credentials to version control
# ==============================================================================

# ==============================================================================
# SECTION: Application Core
# ==============================================================================
# Environment type (dev, stg, prod)
ENV=dev

# Default LLM provider (used if not specified in runtime)
LLM_PROVIDER=custom
# custom will pull from env var BASE_URL/API_KEY
# custom_name will pulle from env var CUSTOM_NAME_BASE_URL / CUSTOM_NAME_API_KEY

# ==============================================================================
# SECTION: Artifacts & Storage
# ==============================================================================
# Directory for storing agent-generated artifacts
# ARTIFACT_DIR=artifacts

# SSL certificate directory for API calls
# SSL_CERT_DIR=

# ==============================================================================
# SECTION: Frontend (Streamlit Configuration)
# ==============================================================================

# --- Websocket & Session Settings ---
# Keep websocket alive with 30-second pings
STREAMLIT_SERVER_WEBSOCKET_PING_INTERVAL=30
# Keep session alive for 5 minutes after disconnect (in seconds)
STREAMLIT_SERVER_DISCONNECTED_SESSION_TTL=450

# --- UI Configuration ---
STREAMLIT_UI_HIDE_TOPBAR=true

# --- Branding (Optional) ---
# Path to logo image (default: data/assets/logo.png)
# LOGO_PATH=data/assets/logo.png
# Application name (default: Conversational AI Chat)
# APP_NAME=Conversational AI Chat
# Application icon emoji (default: 🤖)
# APP_ICON=🤖
# Enable debug mode for UI development
# UI_DEBUG_MODE=false

# --- Graph Configuration ---
# Path to graph configuration storage
STA_GRAPHS_CONFIG_PATH=

# --- UI Lock Configuration (Optional) ---
# Control which UI elements are displayed in the sidebar
# BYOK (Bring Your Own Key) mode will override these when credentials are missing

# Master lock mode: Set to "strict" to enable lock mode (default: disabled)
# UI_LOCK_MODE=strict
# Hide provider selection dropdown
# DISABLE_PROVIDER_SELECTION=true
# Hide provider configuration expander
# DISABLE_PROVIDER_CONFIG=true
# Hide API key input field
# DISABLE_API_KEY_VIEW=true
# Hide model selection
# DISABLE_MODEL_SELECTION=true
# Hide LLM configuration controls
# DISABLE_LLM_CONFIG=true

# --- Streamlit Theme (Optional - Managed via env/themes/ directory) ---
# Note: Use predefined themes in env/themes/ instead of setting these manually
# STREAMLIT_SERVER_ENABLE_STATIC_SERVING=
# STREAMLIT_THEME_PRIMARY_COLOR=
# STREAMLIT_THEME_BACKGROUND_COLOR=
# (See env/themes/.env.* files for theme examples)

# ==============================================================================
# SECTION: Backend - NetworkX Graph
# ==============================================================================
# Enable demo mode (uses NetworkX instead of TigerGraph)
DEMO_MODE=true

# Data directory containing graph extracts
NX_DATA_DIR=data/extracts/mock_data

# ==============================================================================
# SECTION: Backend - Databases (Optional)
# ==============================================================================

# --- PostgreSQL (Optional - for LangGraph checkpointing & persistence) ---
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DATABASE=langgraph
# POSTGRES_USER=postgres
# POSTGRES_PASSWORD=
# POSTGRES_POOL_MIN_SIZE=10
# POSTGRES_POOL_MAX_SIZE=20

# --- Elasticsearch (Optional - for search & analytics features) ---
# General Elasticsearch Cluster
# ELASTICSEARCH_ES_HOST=http://localhost:9200
# ELASTICSEARCH_ES_PORT=9200
# ELASTICSEARCH_ES_CA_CERTS=
# ELASTICSEARCH_ES_ID=
# ELASTICSEARCH_ES_API_KEY=
# ELASTICSEARCH_ES_CLIENT_KEY=
# ELASTICSEARCH_ES_CLIENT_CERT=
# ELASTICSEARCH_ES_DEFAULT_INDEX=
# ELASTICSEARCH_ES_TIMEOUT=30
# ELASTICSEARCH_ES_VERIFY_CERTS=true
# ELASTICSEARCH_ES_MAX_RETRIES=3
# ELASTICSEARCH_ES_RETRY_ON_TIMEOUT=true

# RAG document ingestion (infra/elasticsearch — builds the docs-hybrid index)
# Base URL for per-file source URLs written to metadata.pageUrl (the file's
# repo-relative path is appended). Defaults to the GitHub blob URL when unset,
# so pageUrl is always a real clickable citation; set empty to disable.
# RAG_ELASTICSEARCH_DOCS_BASE_URL=https://github.com/errajibadr/langgraph-agent-repo/blob/main

# File Integrity Elasticsearch (Separate instance for file monitoring)
# FILE_INTEGRITY_ELASTICSEARCH_ES_NAME=elasticsearch
# FILE_INTEGRITY_ELASTICSEARCH_ES_HOST=http://localhost:9200
# FILE_INTEGRITY_ELASTICSEARCH_ES_DEFAULT_INDEX=auditbeat-test
# FILE_INTEGRITY_ELASTICSEARCH_ES_CA_CERTS=
# FILE_INTEGRITY_ELASTICSEARCH_ES_ID=
# FILE_INTEGRITY_ELASTICSEARCH_ES_API_KEY=
# FILE_INTEGRITY_ELASTICSEARCH_ES_CLIENT_KEY=
# FILE_INTEGRITY_ELASTICSEARCH_ES_CLIENT_CERT=
# FILE_INTEGRITY_ELASTICSEARCH_ES_TIMEOUT=30
# FILE_INTEGRITY_ELASTICSEARCH_ES_VERIFY=false
# FILE_INTEGRITY_ELASTICSEARCH_ES_MAX_RETRIES=3
# FILE_INTEGRITY_ELASTICSEARCH_ES_RETRY_ON_TIMEOUT=true

# Retriever Elasticsearch - Used for RAG
# RETRIEVER_ELASTICSEARCH_ES_NAME=elasticsearch
# RETRIEVER_ELASTICSEARCH_ES_HOST=http://localhost:9200
# RETRIEVER_ELASTICSEARCH_ES_DEFAULT_INDEX=auditbeat-test
# RETRIEVER_ELASTICSEARCH_ES_CA_CERTS=
# RETRIEVER_ELASTICSEARCH_ES_ID=
# RETRIEVER_ELASTICSEARCH_ES_API_KEY=
# RETRIEVER_ELASTICSEARCH_ES_CLIENT_KEY=
# RETRIEVER_ELASTICSEARCH_ES_CLIENT_CERT=
# RETRIEVER_ELASTICSEARCH_ES_TIMEOUT=30
# RETRIEVER_ELASTICSEARCH_ES_VERIFY=false
# RETRIEVER_ELASTICSEARCH_ES_MAX_RETRIES=3
# RETRIEVER_ELASTICSEARCH_ES_RETRY_ON_TIMEOUT=true

# Elastic RAG Gateway Proxy - client-side BaseRetriever talking to the deployed
# `elastic_rag` LangGraph gateway via POST /runs/wait. See
# packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/elastic_rag_proxy/.
# GATEWAY_URL is the only required field; secrets live in .env.secrets.
# ELASTIC_RAG_PROXY_RETRIEVER_GATEWAY_URL=http://localhost:2024
# ELASTIC_RAG_PROXY_RETRIEVER_ASSISTANT_ID=elastic_rag
# ELASTIC_RAG_PROXY_RETRIEVER_TIMEOUT_S=30.0
# ELASTIC_RAG_PROXY_RETRIEVER_DEFAULT_TOP_K=10
# Opt in to stitch the gateway's run under the caller's LangSmith trace.
# Injects `langsmith-trace` + `langsmith-project` (sourced from baggage) into
# the /runs/wait body's config.configurable when an enclosing @traceable
# scope is active. Requires `langsmith` to be importable; otherwise a
# graceful no-op. Baggage carries the caller's run metadata verbatim — do
# not place secrets in run metadata when this flag is on.
# ELASTIC_RAG_PROXY_RETRIEVER_DISTRIBUTED_TRACING=false

# Retriever Query Expansion - wires a QueryExpander on the ElasticRetriever.
# PASS disables expansion (no LLM client built). Any non-PASS value forces the
# factory to build a QueryExpander backed by LLMAAS_* env (set further down).
# Per-call ``expansion_hint`` context keys only work when an expander is wired.
# RETRIEVER_EXPANSION_HINT=PASS                           # PASS | AUTO | KEYWORD | MULTI | PARAPHRASE | HYDE
# RETRIEVER_EXPANSION_DOMAIN_INTENT=                      # e.g. "Internal agent platform docs — LangGraph agents, RAG, retrievers."
# RETRIEVER_EXPANSION_MULTI_TIMEOUT_S=15.0                # wall-clock budget for MULTI's KEYWORD + PARAPHRASE + HYDE gather

# Twin Router scope - anonymized entity/apcode arrays for the twin
# project-knowledge retriever (Knowledge Agent RAG path). Real values are
# tenant-identifying and MUST stay out of version control — set them only in
# the gitignored `.env`. Each *_FILTERS / *_BOOSTS var accepts a JSON array
# (["a","b"]) or a comma-separated string (a,b). Leave unset → the twin
# retriever is unscoped. *_FILTERS narrow the result set; *_BOOSTS only
# soft-rank. INCLUDE_TRANSVERSAL widens the apcode filter to also admit
# transversal docs; INCLUDE_ENTITY_CHILDS admits descendant entities.
# TWIN_SCOPE_ENTITY_FILTERS=
# TWIN_SCOPE_APCODE_FILTERS=
# TWIN_SCOPE_APP_NAME_FILTERS=
# TWIN_SCOPE_ENTITY_BOOSTS=
# TWIN_SCOPE_APCODE_BOOSTS=
# TWIN_SCOPE_APP_NAME_BOOSTS=
# TWIN_SCOPE_INCLUDE_ENTITY_CHILDS=false
# TWIN_SCOPE_INCLUDE_TRANSVERSAL=false

# --- TigerGraph (Optional - Advanced graph backend) ---
# TG_HOST=
# TG_PORT=
# TG_GRAPH_NAME=
# TG_USERNAME=admin
# TG_PASSWORD=
# TG_SECRET=

# --- JIRA API -----

# JIRA_BASE_URL=
# JIRA_BEARER_TOKEN=
# JIRA_TIMEOUT=
# JIRA_VERIFY_SSL=

# ==============================================================================
# SECTION: Integrations
# ==============================================================================

# --- LangSmith (Tracing & Debugging) ---
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=
LANGSMITH_API_KEY=
# LANGSMITH_PROJECT=my-project

# LangSmith Test Configuration (for test environment)
LANGSMITH_TEST_TRACING=false
LANGSMITH_TEST_PROJECT=

# --- RAG API Endpoints ---
RAG_API_URL=
REDHAT_RAG_API_URL=
APACHE_RAG_API_URL=
ILLUMIO_RAG_API_URL=
BIGFIX_NETBACKUP_RAG_API_URL=

# ==============================================================================
# SECTION: LLM Providers
# ==============================================================================

# --- Default Provider ---
# Base URL for the default LLM provider
BASE_URL=https://llm.provider.com/
# API key (⚠️ NEVER commit real keys)
API_KEY=sk-your-key-here
# Model name
MODEL=model-name
# Optional capacity-tier model names
# BIG_MODEL=
# SMALL_MODEL=
# THINKING_MODEL=
# Optional vision model (capability axis, not a tier) — used by create_chat_model(..., multimodal=True)
# MULTIMODAL_MODEL=
# Optional model parameters
# TEMPERATURE=0.7
# TOP_P=1.0
# MAX_TOKENS=4096
# Optional reasoning control — 'off' | 'low' | 'medium' | 'high' | 'xhigh' | 'max', translated into
# whatever the resolved model family honors (native GLM/DeepSeek/Qwen3.8 effort,
# Mistral reasoning_effort, or Nemotron/earlier-Qwen chat_template_kwargs). Leave
# unset to keep the model's server-side default. Not every
# family defines every rung (Mistral has only off/high); an unsupported value warns and
# is forwarded raw for the API to judge. REASONING_FAMILY pins the translation table when
# a gateway alias hides the real model name, and does nothing without REASONING_EFFORT.
# See docs/consuming/reasoning.md.
# REASONING_EFFORT=
# REASONING_FAMILY=

# ------ EVAL PROVIDER -----
# EVAL_BASE_URL=
# EVAL_API_KEY=
# EVAL_MODEL has no default — it MUST be set to use the eval provider (LLM-as-judge);
# create_chat_model(provider="eval") raises if it is unset.
# EVAL_MODEL=
# EVAL_BIG_MODEL=
# EVAL_SMALL_MODEL=
# EVAL_THINKING_MODEL=


# --- Custom Named Provider (Optional) ---
# Use this pattern to configure additional LLM providers
# CUSTOM_NAME_BASE_URL=
# CUSTOM_NAME_API_KEY=
# CUSTOM_NAME_MODEL=
# CUSTOM_NAME_BIG_MODEL=
# CUSTOM_NAME_SMALL_MODEL=
# CUSTOM_NAME_THINKING_MODEL=
# CUSTOM_NAME_TEMPERATURE=
# CUSTOM_NAME_TOP_P=
# CUSTOM_NAME_MAX_TOKENS=
# CUSTOM_NAME_REASONING_EFFORT=
# CUSTOM_NAME_REASONING_FAMILY=
# Every key above works for ANY provider name — the prefix is derived as <NAME>_ with no
# code change. Giving a component its own provider name (e.g. the prompt-injection judge
# below) is how you configure that component's model independently of the main agent's.

# --- Orchestrator Prompt-Injection Guard (Optional) ---
# Runs before the orchestrator planner and screens the last five human
# messages for prompt injection, phishing, data exfiltration, policy bypass,
# and tool/subagent manipulation attempts. When the judge model is multimodal
# (e.g. mistral-small-2603), image parts on the latest human turn are screened too.
# Configure PROVIDER/MODEL explicitly to pick the judge. Leaving them unset relies
# on the built-in default judge, which emits a DeprecationWarning; a future release
# will require an explicit judge model and otherwise leave screening off.
ORCHESTRATOR_PROMPT_INJECTION_GUARD_ENABLED=true
ORCHESTRATOR_PROMPT_INJECTION_GUARD_FAIL_OPEN=true
# Soft-advisory tier: a turn carrying a genuine deliverable that ALSO asks for TWIN's internal
# rules ("explain X, then list the rules you followed") passes through with a <system_reminder>
# telling the planner to answer the task and decline the rules part — instead of a hard refusal.
# Set false to fall back to block-or-pass only.
ORCHESTRATOR_PROMPT_INJECTION_GUARD_ADVISORY_ENABLED=true
ORCHESTRATOR_PROMPT_INJECTION_GUARD_PROVIDER=mistral
ORCHESTRATOR_PROMPT_INJECTION_GUARD_MODEL=mistral-small-2603
# ORCHESTRATOR_PROMPT_INJECTION_GUARD_BASE_URL=
# Response token CEILING, not a target — the verdict is a tiny JSON object and the guard is
# meant to stay concise and fast. The cap exists only so a reasoning judge (gpt-oss) that emits
# a little reasoning before the JSON does not truncate mid-object (which, under fail-open, is a
# silent pass). 512 fits a low-effort burst + the JSON. Keep the judge terse via REASONING_EFFORT.
ORCHESTRATOR_PROMPT_INJECTION_GUARD_MAX_TOKENS=512
ORCHESTRATOR_PROMPT_INJECTION_GUARD_TEMPERATURE=0.0
ORCHESTRATOR_PROMPT_INJECTION_GUARD_MAX_RETRIES=2
# Reasoning-effort knob for the judge. Set 'low' for a reasoning judge (gpt-oss) so it returns
# the verdict fast without burning reasoning tokens. Leave UNSET for the Mistral judge above —
# Mistral defines no 'low' rung, so an injected 'low' would be forwarded raw and may be rejected
# (a rejected classify call fails open = a silent pass). Only set it when the judge is gpt-oss.
# ORCHESTRATOR_PROMPT_INJECTION_GUARD_REASONING_EFFORT=low
#
# For any model knob NOT listed above, give the judge its own provider name instead of
# waiting for a matching guard setting. Unknown provider names derive their env prefix
# automatically, so the judge gets a private, complete provider settings block:
#   ORCHESTRATOR_PROMPT_INJECTION_GUARD_PROVIDER=guard_judge
#   GUARD_JUDGE_BASE_URL=... / GUARD_JUDGE_API_KEY=... / GUARD_JUDGE_TOP_P=...
# The guard's own settings above stay authoritative where they overlap.
#
# Server-owned only: request/runtime context cannot override the guard model.

# --- Orchestrator Hub Skills (Optional) ---
# Snapshot lifetime, in seconds, for Hub-served skill groups mounted under
# /skills/hub/<group>/ (active only when a deployment passes hub_skill_repos
# to create_orchestrator_factory). After the TTL the next access re-pulls the
# repo tree — how a `sta skills push` reaches a running deployment without a
# restart. <= 0 refreshes on every access; the /skill-reload chat command
# bypasses the TTL entirely.
# ORCHESTRATOR_HUB_SKILLS_TTL_SECONDS=300

# --- Orchestrator Tool Budget Guard (Optional) ---
# Unset by default: no tool-call cap. Two independent, opt-in budgets:
# 1. Global per-run cap. When reached, the next planner turn is forced to answer
#    from context with no further tool calls (the answer-now soft-landing; it keeps
#    the tools/system prefix byte-identical for vLLM prefix-cache reuse).
# ORCHESTRATOR_TOOL_BUDGET_GUARD_MAX_TOOL_CALLS=
# 2. Per-tool per-run caps (JSON map of tool name -> max calls). An exhausted
#    per-tool cap blocks only that tool with a recoverable error; it does not force
#    answer-now. Example: cap the deepagents `task` delegation tool at 5 per run.
# ORCHESTRATOR_TOOL_BUDGET_GUARD_PER_TOOL_MAX_CALLS={"task": 5}
# Server-owned only: request/runtime context cannot install or raise either budget.

# --- Orchestrator Picture Reader (Optional) ---
# Exposed as read_picture only when the planner model is not listed in the
# multimodal model registry. It lets a multimodal model inspect image content
# that text-only planner models cannot receive directly.
ORCHESTRATOR_PICTURE_READER_ENABLED=true
# Describe an image-bearing turn with the picture-reader and rewrite that turn in
# place (original text + description) instead of only stripping the image. Set to
# false to restore strip + on-demand read_picture behavior.
ORCHESTRATOR_PICTURE_READER_EAGER_DESCRIBE=true
# Keep read_picture bound as an on-demand fallback for differently-targeted second looks.
ORCHESTRATOR_PICTURE_READER_KEEP_READ_PICTURE_TOOL=false
ORCHESTRATOR_PICTURE_READER_PROVIDER=mistral
ORCHESTRATOR_PICTURE_READER_MODEL=mistral-small-2603
# ORCHESTRATOR_PICTURE_READER_BASE_URL=
ORCHESTRATOR_PICTURE_READER_MAX_TOKENS=1024
ORCHESTRATOR_PICTURE_READER_TEMPERATURE=0.0
ORCHESTRATOR_PICTURE_READER_MAX_IMAGES=12
ORCHESTRATOR_PICTURE_READER_MAX_CONTEXT_MESSAGES=12

# --- Embedding Models (Optional) ---
EMBEDDING_BASE_URL=https://embedding.provider.com/
EMBEDDING_API_KEY=sk-test
EMBEDDING_MODEL=embedding-model
EMBEDDING_DIMENSIONS=1024
# Request timeout in seconds for embedding API calls (default 30)
# EMBEDDING_TIMEOUT=30

# --- Reranking ---
RERANKING_BASE_URL=https://reranking.provider.com/
RERANKING_API_KEY=sk-rerank-key
RERANKING_MODEL=reranking-model

# --- Knowledge Agent per-task model overrides (Optional) ---
# Opt-in: consumers call KnowledgeAgentConfig.from_env() to pick these up.
# Folds into task_model_defaults (layer 3 of the resolution ladder) — runtime
# context.model_configs still wins. Provider/model are intentionally NOT
# hardcoded in the KA package; set these (or LLM_PROVIDER + <PROVIDER>_*
# at the engine-wide layer) to choose a backend per task.
#
# Tasks: default | planning | compression | review | synthesis | verification
# Keys per task: PROVIDER, TIER, BASE_URL, MODEL, MAX_TOKENS, TEMPERATURE
#   (KA_<TASK>_API_KEY lives in .env.secrets.example)
#
# KA_PLANNING_PROVIDER=
# KA_PLANNING_TIER=
# KA_PLANNING_BASE_URL=
# KA_PLANNING_MODEL=
# KA_PLANNING_MAX_TOKENS=2048
# KA_PLANNING_TEMPERATURE=0.0
#
# KA_COMPRESSION_PROVIDER=
# KA_COMPRESSION_TIER=
# KA_COMPRESSION_BASE_URL=
# KA_COMPRESSION_MODEL=
# KA_COMPRESSION_MAX_TOKENS=8192
# KA_COMPRESSION_TEMPERATURE=0.0
#
# KA_REVIEW_PROVIDER=
# KA_REVIEW_TIER=
# KA_REVIEW_BASE_URL=
# KA_REVIEW_MODEL=
# KA_REVIEW_MAX_TOKENS=4096
# KA_REVIEW_TEMPERATURE=0.0
#
# KA_SYNTHESIS_PROVIDER=
# KA_SYNTHESIS_TIER=
# KA_SYNTHESIS_BASE_URL=
# KA_SYNTHESIS_MODEL=
# KA_SYNTHESIS_MAX_TOKENS=8192
# KA_SYNTHESIS_TEMPERATURE=0.0
#
# KA_VERIFICATION_PROVIDER=
# KA_VERIFICATION_TIER=
# KA_VERIFICATION_BASE_URL=
# KA_VERIFICATION_MODEL=
# KA_VERIFICATION_MAX_TOKENS=4096
# KA_VERIFICATION_TEMPERATURE=0.0

# ==============================================================================
# SECTION: Test Configuration
# ==============================================================================
# Models used for integration tests and evaluations

# Large Model (High-capacity for complex tasks)
TEST_MODEL_LARGE=openai/gpt-oss-120b
TEST_MODEL_LARGE_PROVIDER=openai

# Medium Model (Balanced performance - good for tool usage)
TEST_MODEL_MEDIUM=mistral-small-2603
TEST_MODEL_MEDIUM_PROVIDER=mistral

# Small Model (Lightweight for simple tasks)
TEST_MODEL_SMALL=mistral-small-2603
TEST_MODEL_SMALL_PROVIDER=mistral

# Llama Model (Alternative architecture)
TEST_MODEL_LLAMA=Meta-llama33-70b-instruct
TEST_MODEL_LLAMA_PROVIDER=llmaas_dev

# ==============================================================================
# SECTION: Logging & Monitoring
# ==============================================================================
# Directory for log files
# LOG_DIR=logs

# Enable file-based logging (in addition to console)
# ENABLE_FILE_LOGGING=false

# Emit one ssl-audit log line per create_chat_model() call (diagnostic only) - set to 1 to enable
# STA_SSL_AUDIT=1

# ==============================================================================
# SECTION: Documentation (MkDocs)
# ==============================================================================
# DOCS_SITE_URL=
# DOCS_REPO_NAME=
# DOCS_REPO_URL=
# DOCS_EDIT_URI=

# Serve the built docs site from the LangGraph server at /documentation
# (requires a built site: `make docs/build`)
# DEPLOY_DOCS=true
# Override the built-site location (defaults to <repo>/output/site)
# DOCS_SITE_DIR=


# HABILITATION (ONLY TWIN for now is compatible and implements this logic)
# HABILITATION_BYPASS=1
# HABILITATION_BYPASS_ROLE=prod
# ==============================================================================
# SECTION: Habilitation Configuration
# ==============================================================================
# HABILITATION_API_BASE_URL=
# HABILITATION_API_KEY=
# HABILITATION_UIDS=
# HABILITATION_ROLE=
# HABILITATION_ROLE_TYPE=


# ==============================================================================
# End of Configuration
# ==============================================================================

-------

docs/consuming/chat-models.md
----
# Using `create_chat_model`

`create_chat_model` builds a configured chat model (a LangChain
`ChatOpenAI` or `ChatMistralAI`) from environment variables, with optional
per-call overrides. It is the single entry point for talking to any LLM
provider in this library.

```python
from sta_agent_engine.models import create_chat_model

model = create_chat_model("llmaas")
response = await model.ainvoke("What is Python in one sentence?")
```

## Selecting a provider

The first argument picks the provider. Three equivalent forms:

```python
create_chat_model("llmaas")               # built-in, by string
create_chat_model(ProviderType.MISTRAL)   # built-in, by enum (from sta_agent_core.config)
create_chat_model("openai")                  # any other name — convention provider
create_chat_model()                        # no arg → reads LLM_PROVIDER (default: custom)
```

Built-in names: `llmaas`, `llmaas_dev`, `mistral`, `custom`, `eval`, `openai`.
**Any other string is also accepted** — its env prefix is derived as
`f"{NAME.upper()}_"`, so `create_chat_model("openai")` reads `openai_*` with no
code change.

## Environment variable contract

For a provider `NAME`, set the prefixed vars (e.g. `LLMAAS_API_KEY`):

| Variable | Purpose |
|---|---|
| `{NAME}_API_KEY` | Auth key |
| `{NAME}_BASE_URL` | OpenAI-compatible endpoint |
| `{NAME}_MODEL` | Default model |
| `{NAME}_BIG_MODEL` / `{NAME}_SMALL_MODEL` / `{NAME}_THINKING_MODEL` | Capacity-tier models |
| `{NAME}_MULTIMODAL_MODEL` | Vision model |
| `{NAME}_TEMPERATURE` / `{NAME}_TOP_P` / `{NAME}_MAX_TOKENS` | Generation params |

The built-in `custom` provider uses **no prefix** (bare `API_KEY`, `BASE_URL`,
`MODEL`, …). `openai` reads the canonical `OPENAI_*` vars the official OpenAI
SDK also honors, so `create_chat_model("openai")` works with just
`OPENAI_API_KEY` set.

A ready-to-fill starter file lives at `.env.provider.example`.

## Capacity tiers — `tier=`

`tier` picks a model slot by **capacity/quality** when no explicit `model=` is
passed. All tiers answer the same request; they trade quality for latency/cost.

```python
create_chat_model("llmaas")                    # tier="default" → LLMAAS_MODEL
create_chat_model("llmaas", tier="big")        # LLMAAS_BIG_MODEL,    else MODEL
create_chat_model("llmaas", tier="small")      # LLMAAS_SMALL_MODEL,  else MODEL
create_chat_model("llmaas", tier="thinking")   # LLMAAS_THINKING_MODEL, else BIG_MODEL, else MODEL
```

Cascade: `thinking → big → model`, `big → model`, `small → model`. An unknown
tier raises `ValueError`.

## Vision models — `multimodal=True`

Multimodality is a **capability**, not a tier (you can't ask for the "big
multimodal" model). Pass `multimodal=True`:

```python
create_chat_model("llmaas", multimodal=True)
```

Resolution order:

1. `{NAME}_MULTIMODAL_MODEL` if set — used **verbatim** (authoritative; lets you
   name a vision model the built-in capability list doesn't yet recognize).
2. else `{NAME}_MODEL` **only if** it's a recognized multimodal model.
3. else **raises `ValueError`** — it never silently returns a text-only model
   that would drop images (under the guard middleware) or 400 upstream.

`multimodal=True` takes precedence over `tier=` and is ignored when an explicit
`model=` is passed. Recognized vision models are the allow-list in
`sta_agent_engine.models.capabilities` (`is_multimodal`); setting
`{NAME}_MULTIMODAL_MODEL` bypasses that check.

## Reasoning control — `reasoning_effort=`

Reasoning/thinking is controlled with a normalized `reasoning_effort`
parameter, translated per model family into the kwargs each model actually
honors (native GLM/DeepSeek/Qwen3.8 effort, Mistral's `reasoning_effort`, or
Nemotron/earlier-Qwen `chat_template_kwargs`). See the dedicated guide:
[reasoning.md](reasoning.md).

```python
create_chat_model("mistral", reasoning_effort="high")
create_chat_model("llmaas", model="nemotron-3-super-120b", reasoning_effort="off")
```

## Per-call overrides

Any keyword overrides the resolved env value:

```python
create_chat_model("llmaas", model="some-other-model", temperature=0.7, max_tokens=512)

# Bring-your-own-key (BYOK): inject credentials at call time
create_chat_model("llmaas", provider_api_key="sk-...", provider_base_url="https://...")
```

## Client dispatch (OpenAI vs Mistral)

`create_chat_model` routes to `ChatMistralAI` when the **provider name** contains
`mistral` (e.g. `mistral`, `mistral_eu`) **or the model name** contains
`mistral` / `devstral` / `magistral`; otherwise it routes to `ChatOpenAI` for
OpenAI-compatible APIs.

> **Sharp edge — the model name wins.** A Mistral-branded model on an
> OpenAI-compatible gateway (e.g. `provider="llmaas"` with
> `LLMAAS_MODEL=mistral-small-2506`) routes to the **native Mistral SDK**
> (`api.mistral.ai`), not your gateway. If your gateway serves Mistral-family
> models over an OpenAI-compatible API, pass an explicit `base_url` (and
> `api_key`) so the Mistral client targets your gateway, or use a non-Mistral
> model name.

If the OpenAI dispatch is taken but `api_key`/`base_url` did not resolve,
`ChatOpenAI` silently falls back to `OPENAI_API_KEY` + `api.openai.com`. This
fallback now emits a `DeprecationWarning` and will raise in a future release —
always set `{NAME}_API_KEY` and `{NAME}_BASE_URL` (or pass them as kwargs).

## Full example

A runnable end-to-end example is in
`examples/sta_agent_engine/chat_model_example.py`.

-------

docs/consuming/reasoning.md
----
# Reasoning control with `create_chat_model`

Reasoning ("thinking") models expose incompatible knobs: Mistral takes a
top-level `reasoning_effort` string, Nemotron and Gemma 4 use booleans nested
under `extra_body.chat_template_kwargs`, and GLM/DeepSeek combine a native
effort with a separate thinking toggle. The
`reasoning_effort` parameter gives you one vocabulary; the library translates
it into whatever the resolved model actually honors.

```python
from sta_agent_engine.models import create_chat_model

model = create_chat_model("llmaas", model="nemotron-3-super-120b", reasoning_effort="low")
# → ChatOpenAI(..., extra_body={"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}})

model = create_chat_model("mistral", reasoning_effort="high")
# → ChatMistralAI(..., model_kwargs={"reasoning_effort": "high"})
```

Omit the parameter (or pass `None`, or an empty/blank string) and **nothing is
injected** — the model keeps its server-side default. Existing code is
unaffected.

## Configuring it by environment variable

`reasoning_effort` and `reasoning_family` are ordinary provider settings, so you
don't have to thread them through your call sites. Set them under the provider's
env prefix and every `create_chat_model` call for that provider picks them up:

```bash
LLMAAS_REASONING_EFFORT=low
LLMAAS_REASONING_FAMILY=nemotron-super   # only needed for gateway aliases (below)
```

```python
create_chat_model("llmaas", model="nemotron-3-super-120b")
# → ChatOpenAI(..., extra_body={"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}})
```

An explicit kwarg still wins over the environment, so a call site that needs a
different effort can override without touching config.

Unset means unset: with no env var and no kwarg, nothing is injected. A blank
value (`LLMAAS_REASONING_EFFORT=`) is also treated as unset rather than as an
empty effort string.

This works for **any** provider name, including ones you never registered — the
prefix is derived as `<NAME>_`. That makes a dedicated provider name the way to
give one component its own reasoning settings without affecting anything else:

```bash
# a screening/classifier model that should answer fast, while the main agent thinks hard
MY_CLASSIFIER_BASE_URL=https://gateway.example/v1
MY_CLASSIFIER_API_KEY=...
MY_CLASSIFIER_MODEL=openai/gpt-oss-120b
MY_CLASSIFIER_REASONING_EFFORT=low
```

```python
create_chat_model("my_classifier")
```

Pick an effort per agent/task/thread rather than per turn — see the caching note
at the end of this page.

## Effort vocabulary and what goes on the wire

Efforts are plain strings. Built-in families support:

| effort | GLM-5.2 | DeepSeek V4 | Qwen3.8 | Earlier Qwen3.x | Gemma 4 |
|---|---|---|---|---|---|
| `"off"` | `thinking.type="disabled"` | `thinking.type="disabled"` | `enable_thinking=False` | `enable_thinking=False` | `enable_thinking=False` (default) |
| `"low"` | native `low` (evaluated as `high`) | native `high` | native `low` | — | — |
| `"medium"` | native `medium` (evaluated as `high`) | native `high` | native `medium` | — | — |
| `"high"` | native `high` | native `high` | native `xhigh` | `enable_thinking` (default) | `enable_thinking=True` |
| `"xhigh"` | native `xhigh` (evaluated as `max`) | native `max` | native `xhigh` | — | — |
| `"max"` | native `max` | native `max` | — | — | — |

GLM-5.2 also accepts the native strings `none` and `minimal`; both stop
thinking. Qwen3.8's official native levels are `low`, `medium`, and `xhigh`, so
the portable `high` rung maps to its strongest `xhigh` level. The open-weight
`Qwen3.8-2.4T-A95B` variant requires thinking and therefore intentionally has
no `off` rung; switchable variants such as `Qwen3.8-27B` do.

The built-in Qwen `off` translation targets self-hosted vLLM/SGLang and sends
`extra_body.chat_template_kwargs.enable_thinking=False`. Qwen Cloud uses a
different direct `extra_body.enable_thinking=False` shape; for that endpoint,
omit the normalized `reasoning_effort="off"` and pass its native `extra_body`
field explicitly.

The other built-in families retain these mappings:

| effort | Mistral (small / medium-3-5) | Nemotron-Super | Nemotron-Ultra |
|---|---|---|---|
| `"off"` | `reasoning_effort="none"` | `enable_thinking=False` | `enable_thinking=False` |
| `"low"` | — | `enable_thinking, low_effort` | `enable_thinking, medium_effort, force_nonempty_content=False` |
| `"medium"` | — | — | `enable_thinking, medium_effort, force_nonempty_content=True` |
| `"high"` | `reasoning_effort="high"` | `enable_thinking` (full) | `enable_thinking, force_nonempty_content=True` (full) |

Qwen3.x before 3.8 is deliberately binary: the model card (e.g. `Qwen/Qwen3.6-27B`)
documents only the `enable_thinking` switch. Graded thinking **budgets** are a
serving-stack feature (vLLM's `thinking_token_budget`, name and availability
vary by version) — if your gateway supports them, re-register the family with
budget rungs (example below).

Gemma 4 is also deliberately binary. Its official template defaults
`enable_thinking` to `False`, and exposes no documented graded effort or token
budget. `preserve_thinking` is a separate template option for retaining prior
assistant reasoning; pass it explicitly under `extra_body.chat_template_kwargs`
when needed.

`—` = not supported by that family. The table makes documented compatibility
mappings explicit (for example, DeepSeek `low` becomes native `high`). For an
unsupported value, the library warns and never raises. Families with a native
`reasoning_effort` path forward the raw value for the API to validate;
template-only families such as Nemotron, earlier Qwen3.x, and Gemma 4 inject
nothing.

Nemotron-Ultra notes: `medium`/`high` include
`force_nonempty_content=True` (required by SGLang when tool-calling with
thinking enabled); `low` is medium-effort thinking with the flag explicitly
sent as `False` (so a server-side template default can't force it back on).

The Mistral column applies to every model the library dispatches to the
Mistral client — names containing `mistral`, `devstral`, or `magistral` all
use it, so `"off"` always reaches the API as the sanctioned `"none"`.

Check what a model supports programmatically:

```python
from sta_agent_engine.models import supported_reasoning_efforts

supported_reasoning_efforts("nemotron-3-ultra-550b")   # frozenset({'off', 'low', 'medium', 'high'})
```

## Unknown models

If no family matches the model name, the value is forwarded as a top-level
`reasoning_effort` (the OpenAI-native form) with a `UserWarning`. Genuine
OpenAI reasoning models honor it; vLLM/LiteLLM-fronted gateways typically
ignore it. `gpt-*` model names take this passthrough silently — it is their
native parameter.

If your gateway serves a known family under an alias (`chat-default` actually
being Nemotron), pin the family explicitly:

```python
create_chat_model("llmaas", model="chat-default",
                  reasoning_effort="high", reasoning_family="nemotron-super")
```

`reasoning_family` only selects the translation table — passing it without
`reasoning_effort` injects nothing (the library warns and ignores it).

## Overrides always win

Explicitly-passed native kwargs are the escape hatch, and they beat the
translation on conflicting keys (with a warning). Non-conflicting keys merge:

```python
create_chat_model(
    "llmaas", model="nemotron-3-super-120b",
    reasoning_effort="high",                                  # → enable_thinking=True
    extra_body={"chat_template_kwargs": {"custom_flag": 1}},  # merged alongside
)
# wire: chat_template_kwargs == {"enable_thinking": True, "custom_flag": 1}
```

## Per-call control

`build_reasoning_kwargs()` returns the translated kwargs so you can apply
effort per call instead of per model instance:

```python
from sta_agent_engine.models import build_reasoning_kwargs

kw = build_reasoning_kwargs("nemotron-3-super-120b", "low")
model.invoke(messages, **kw)        # or model.bind(**kw)
```

This works for families that translate into `extra_body` (Nemotron, Qwen) or
a native field (OpenAI). **Exception — Mistral models are constructor-only:**
their translation rides `model_kwargs`, which `ChatMistralAI` flattens into
the request payload only when set as a constructor field; splatted at call
time it would be posted as a literal `"model_kwargs"` JSON key instead. For
Mistral, set the effort where the model is created
(`create_chat_model(..., reasoning_effort=...)`).

Prefer choosing an effort **per agent / task / thread, not per turn**: for
template-flag families, flipping `chat_template_kwargs` between turns of one
conversation re-renders the server-side prompt template and defeats
vLLM/SGLang prefix caching (latency and cost, not correctness).

## Registering your own model family

One call at application startup, no subclassing:

```python
from sta_agent_engine.models import register_reasoning_family

register_reasoning_family(
    "my-model",
    rungs={
        "off":  {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    },
    match_substrings=("my-model",),   # all must appear in the model name
)
```

Rung values are the **literal kwargs** to merge into the model constructor —
anything the underlying client accepts (`extra_body`, `model_kwargs`, native
fields) is fair game. Re-registering an existing family (including a built-in)
replaces it, so you can also override the library's defaults. Example —
extending the built-in `qwen3` family with graded thinking budgets once you've
verified your vLLM gateway supports them:

```python
register_reasoning_family(
    "qwen3",
    rungs={
        "off":    {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        "medium": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 2048}},
        "high":   {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}},
        "xhigh":  {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 8192}},
    },
    match_substrings=("qwen3",),
)
```

Verify the budget parameter name against your serving stack first — vLLM has
shipped it as `thinking_token_budget` and rejected/ignored other spellings
depending on version; a one-off probe call per rung is cheap insurance.

### Slug variants across providers

Family matching is **case- and separator-insensitive**: `-`, `_`, `.`, `:`,
`/` and spaces are stripped from both the model name and the patterns before
comparing. The same model arriving under different provider slug conventions —
`qwen3.6`, `qwen3-6`, `Qwen/Qwen3.6-32B-Instruct`, `qwen3:32b` — resolves to
the same family and produces identical wire kwargs. You only need
`reasoning_family=` when the alias shares nothing with the model's real name
(`chat-default`).

`match_substrings` supports two shapes:

- **Flat tuple = one AND-group** — all substrings must appear:
  `("nemotron-3", "ultra")`.
- **Tuple of tuples = OR of AND-groups** — the family matches when *any*
  group fully matches; use this for genuinely different naming dialects:
  `(("nemotron-3", "ultra"), ("nemo-ultra",))`.

## Known limitations

- **Mistral + tool-calling agents:** the upstream LangChain Mistral
  integration sends an assistant message's `tool_calls` *instead of* its
  content, so reasoning traces are not replayed to the API on tool-call turns.
  Plain chat turns replay them automatically. Impact is limited to extra
  re-thinking inside agent loops, not wrong answers. (Qwen3.6 addresses the
  same concern server-side with `chat_template_kwargs.preserve_thinking=True` —
  orthogonal to effort; pass it via explicit `extra_body`, it merges alongside
  the translated flags.)
- **Whether an effort actually changes behavior is ultimately decided by the
  serving stack.** Some gateways enable thinking by default and ignore
  parameters they don't recognize. When in doubt, verify with a one-off call
  per effort value and inspect the response's reasoning content.

-------

examples/sta_agent_engine/models/qwen3_8_reasoning_effort_example.py
----
"""Example: measure Qwen3.8-27B reasoning tokens across effort levels.

NOT consumer documentation — this is an engineering smoke-test of the live
``create_chat_model`` request surface. Consumer guidance lives in
``docs/consuming/reasoning.md``. Edit the USER_* constants and run:

    uv run python examples/sta_agent_engine/models/qwen3_8_reasoning_effort_example.py

REAL PAID API CALLS: six requests per repeat (default, off, low, medium, high,
and xhigh). The default repeat count is one. The configured provider must have
its ``{NAME}_API_KEY``, ``{NAME}_BASE_URL``, and model access set up.

Qwen3.8-27B natively supports ``low``, ``medium``, and ``xhigh``; its default
is ``xhigh``. This library exposes a portable ``high`` level and maps it to
Qwen's native ``xhigh``. Consequently, the high and xhigh rows intentionally
send identical reasoning configuration. ``off`` uses the self-hosted
vLLM/SGLang ``chat_template_kwargs.enable_thinking=False`` dialect.

The reported reasoning-token count prefers the gateway's authoritative usage
metadata. If the gateway omits it, the script estimates the count from the
returned reasoning trace at approximately four characters per token. A single
sample is noisy; set USER_REPEATS to 3 or more before drawing conclusions.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from sta_agent_engine.models import build_reasoning_kwargs, create_chat_model


# --- Edit these -------------------------------------------------------------
USER_PROVIDER = "llmaas"  # reads LLMAAS_* env vars; any provider name works
USER_MODEL = "Qwen/Qwen3.8-27B"  # use the exact slug exposed by your gateway
USER_PROMPT = "Find every triple of positive integers x <= y <= z satisfying 1/x + 1/y + 1/z = 1. Prove that your list is complete."
USER_MAX_TOKENS = 8192
USER_REPEATS = 1  # use >=3 for a less noisy comparison; each repeat costs six calls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffortCase:
    """One normalized effort and the Qwen-native behavior it should select."""

    label: str
    effort: str
    expected_native: str


@dataclass(frozen=True)
class Measurement:
    """Metrics captured from one live request."""

    case: EffortCase
    repeat: int
    reasoning_tokens: int
    token_source: str
    reasoning_chars: int
    output_tokens: int | None
    seconds: float
    answer_preview: str


# Empty effort deliberately suppresses a provider-level *_REASONING_EFFORT env
# value and injects nothing, allowing Qwen's actual server-side default to win.
_CASES = (
    EffortCase("default", "", "xhigh (Qwen default)"),
    EffortCase("off", "off", "thinking disabled"),
    EffortCase("low", "low", "low"),
    EffortCase("medium", "medium", "medium"),
    EffortCase("high", "high", "xhigh (portable alias)"),
    EffortCase("xhigh", "xhigh", "xhigh"),
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mapping values as-is and replace every other shape with empty."""
    return value if isinstance(value, Mapping) else {}


def _reasoning_text(response: AIMessage) -> str:
    """Extract the visible reasoning trace without double-counting aliases."""
    for key in ("reasoning_content", "reasoning"):
        value = response.additional_kwargs.get(key)
        if value:
            return str(value)

    if isinstance(response.content, list):
        parts: list[str] = []
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                parts.append(str(block.get("reasoning") or block.get("content") or ""))
        return "".join(parts)
    if isinstance(response.content, str) and "<think>" in response.content:
        _, reasoning_and_answer = response.content.split("<think>", 1)
        reasoning, separator, _ = reasoning_and_answer.partition("</think>")
        return reasoning if separator else reasoning_and_answer
    return ""


def _api_reasoning_tokens(response: AIMessage) -> int | None:
    """Read normalized LangChain usage first, then common raw API shapes."""
    usage = _as_mapping(response.usage_metadata)
    output_details = _as_mapping(usage.get("output_token_details"))
    for key in ("reasoning", "reasoning_tokens"):
        value = output_details.get(key)
        if isinstance(value, int):
            return value

    response_metadata = _as_mapping(response.response_metadata)
    raw_usage = _as_mapping(response_metadata.get("token_usage") or response_metadata.get("usage"))
    completion_details = _as_mapping(raw_usage.get("completion_tokens_details"))
    value = completion_details.get("reasoning_tokens")
    return value if isinstance(value, int) else None


def _output_tokens(response: AIMessage) -> int | None:
    """Return the provider-counted total output tokens when present."""
    usage = _as_mapping(response.usage_metadata)
    value = usage.get("output_tokens")
    if isinstance(value, int):
        return value

    response_metadata = _as_mapping(response.response_metadata)
    raw_usage = _as_mapping(response_metadata.get("token_usage") or response_metadata.get("usage"))
    value = raw_usage.get("completion_tokens")
    return value if isinstance(value, int) else None


def _count_reasoning(response: AIMessage, reasoning_text: str) -> tuple[int, str]:
    """Return the best available reasoning count and its provenance."""
    api_count = _api_reasoning_tokens(response)
    if api_count is not None:
        return api_count, "api"
    return round(len(reasoning_text) / 4), "~4 chars/token"


async def _measure(case: EffortCase, repeat: int) -> Measurement:
    """Create a model at one effort level, invoke it, and measure reasoning."""
    model = create_chat_model(
        USER_PROVIDER,
        model=USER_MODEL,
        reasoning_effort=case.effort,
        max_tokens=USER_MAX_TOKENS,
        temperature=1.0,
        top_p=0.95,
        extra_body={"top_k": 20},
    )
    started = time.perf_counter()
    response = await model.ainvoke(USER_PROMPT)
    seconds = time.perf_counter() - started
    if not isinstance(response, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(response).__name__}")

    reasoning_text = _reasoning_text(response)
    reasoning_tokens, token_source = _count_reasoning(response, reasoning_text)
    answer_preview = response.text.strip().replace("\n", " ")[:72]
    return Measurement(
        case=case,
        repeat=repeat,
        reasoning_tokens=reasoning_tokens,
        token_source=token_source,
        reasoning_chars=len(reasoning_text),
        output_tokens=_output_tokens(response),
        seconds=seconds,
        answer_preview=answer_preview,
    )


def _print_translation() -> None:
    """Show exactly what create_chat_model will merge for every case."""
    print(f"Model: {USER_MODEL}")
    print("Qwen native levels: low, medium, xhigh (default: xhigh)\n")
    print("Normalized create_chat_model translation:")
    for case in _CASES:
        translated = build_reasoning_kwargs(USER_MODEL, case.effort)
        print(f"  {case.label:<7} -> {case.expected_native:<25} {translated}")


def _print_results(measurements: list[Measurement]) -> None:
    """Print per-call results and averages per effort."""
    print("\nPer-call measurements:")
    print(f"{'effort':<8} {'run':>3} {'rsn tokens':>10} {'source':<16} {'rsn chars':>9} {'out tok':>8} {'seconds':>8}  answer")
    for item in measurements:
        output_tokens = item.output_tokens if item.output_tokens is not None else "n/a"
        print(
            f"{item.case.label:<8} {item.repeat:>3} {item.reasoning_tokens:>10} {item.token_source:<16} "
            f"{item.reasoning_chars:>9} {output_tokens:>8} {item.seconds:>8.1f}  {item.answer_preview}"
        )

    print("\nAverages:")
    print(f"{'effort':<8} {'native behavior':<25} {'avg rsn tok':>11} {'avg seconds':>11}")
    for case in _CASES:
        group = [item for item in measurements if item.case == case]
        avg_tokens = statistics.fmean(item.reasoning_tokens for item in group)
        avg_seconds = statistics.fmean(item.seconds for item in group)
        print(f"{case.label:<8} {case.expected_native:<25} {avg_tokens:>11.1f} {avg_seconds:>11.1f}")

    enabled = [item for item in measurements if item.case.label not in {"off"}]
    observable_reasoning = any(item.token_source == "api" or item.reasoning_chars > 0 for item in enabled)
    off = [item.reasoning_tokens for item in measurements if item.case.label == "off"]
    if not observable_reasoning:
        print("\nWARNING: no enabled run exposed reasoning usage or text; this gateway cannot be evaluated with this response shape.")
    elif off and any(count > 0 for count in off):
        print("\nWARNING: off returned reasoning tokens; the gateway may be ignoring chat_template_kwargs.enable_thinking=False.")
    elif off:
        print("\nPASS: off returned zero visible/provider-counted reasoning tokens.")
    print("High and xhigh are expected to be statistically similar: both send Qwen's native xhigh. Individual runs need not match exactly.")


async def main() -> None:
    """Run every effort sequentially to avoid mixing concurrent load effects."""
    if USER_REPEATS < 1:
        raise ValueError("USER_REPEATS must be at least 1")

    _print_translation()
    measurements: list[Measurement] = []
    for repeat in range(1, USER_REPEATS + 1):
        for case in _CASES:
            print(f"Running {case.label}, repeat {repeat}/{USER_REPEATS}...")
            measurements.append(await _measure(case, repeat))
    _print_results(measurements)


if __name__ == "__main__":
    asyncio.run(main())

-------

examples/sta_agent_engine/models/reasoning_effort_example.py
----
"""Example: reasoning_effort with create_chat_model — reasoning-token cost per effort.

NOT consumer documentation — this is an engineering smoke-test of the
``reasoning_effort`` surface against live Nemotron endpoints. Consumer-facing
usage lives in docs/consuming/reasoning.md. Edit the USER_* constants and run:

    uv run python examples/sta_agent_engine/models/reasoning_effort_example.py

REAL PAID API CALLS: one request per (model, prompt, effort) — 14 with the
defaults below. Requires the provider env vars ({NAME}_API_KEY +
{NAME}_BASE_URL for the provider name set below).

For each model and prompt, every supported effort is invoked once and the
reasoning cost is reported two ways: the API-counted reasoning tokens (when
the gateway returns ``completion_tokens_details.reasoning_tokens`` — the
authoritative, billed count) and an estimate from the reasoning text surfaced
in the response (~4 chars/token — a fallback for gateways that report no
token details; it undercounts terse math/symbol content).

Two prompts on purpose: an easy classic caps reasoning naturally at ~100
tokens whatever the effort (ceiling effect), so effort asymmetry only becomes
observable on the multi-step constraint puzzle. Expectation there: ``off``
≈ 0 reasoning tokens and reasoning grows with the rung. If ``off`` still
shows a sizable count, the gateway is not applying ``enable_thinking=False``
(check whether it forwards ``extra_body.chat_template_kwargs``).
"""

import asyncio
import time

from langchain_core.messages import AIMessage

from sta_agent_engine.models import create_chat_model, supported_reasoning_efforts


# --- Edit these -------------------------------------------------------------
USER_PROVIDER = "custom"  # any built-in or arbitrary name; reads {NAME}_* env vars
USER_MODELS = ("nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-ultra-550b-a55b")
USER_PROMPTS = (
    (
        "easy",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    ),
    (
        "complex",
        "Four people (Ava, Ben, Cy, Dee) each ordered a different drink (coffee, tea, juice, water) "
        "and sat in seats 1-4, left to right. Clues: (1) Ava sat immediately left of the tea drinker. "
        "(2) Ben did not order coffee and did not sit in seat 4. (3) The juice drinker sat in seat 1. "
        "(4) Cy sat exactly two seats right of Dee. (5) The water drinker sat next to Ben. "
        "Work out who sat where and who ordered what.",
    ),
)
USER_MAX_TOKENS = 4096  # cap per response so a chatty high-effort run stays bounded
# ---------------------------------------------------------------------------

_EFFORT_ORDER = ("off", "low", "medium", "high", "xhigh", "max")  # display order for whatever rungs the family defines


def _reasoning_text(response: AIMessage) -> str:
    """Collect whatever reasoning the gateway surfaces client-side.

    Gateways differ: some put it in ``additional_kwargs["reasoning_content"]``
    (the create_chat_model converters normalize to this), others emit
    ``type="reasoning"`` content blocks.
    """
    parts = [str(response.additional_kwargs.get("reasoning_content") or "")]
    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                parts.append(str(block.get("reasoning") or block.get("content") or ""))
    return "".join(parts)


async def _measure(model_name: str, effort: str, prompt: str) -> tuple[int | str, int, int | str, float, str]:
    """Invoke once at the given effort; return (api reasoning tokens, estimated
    reasoning tokens, output tokens, seconds, answer preview)."""
    model = create_chat_model(USER_PROVIDER, model=model_name, reasoning_effort=effort, max_tokens=USER_MAX_TOKENS)
    started = time.perf_counter()
    response = await model.ainvoke(prompt)
    elapsed = time.perf_counter() - started
    assert isinstance(response, AIMessage)

    usage = response.usage_metadata or {}
    api_reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    estimated_reasoning = round(len(_reasoning_text(response)) / 4)  # ~4 chars/token heuristic
    output_tokens = usage.get("output_tokens")
    answer = response.text.strip().replace("\n", " ")[:70]
    return (
        api_reasoning if api_reasoning is not None else "n/a",
        estimated_reasoning,
        output_tokens if output_tokens is not None else "n/a",
        elapsed,
        answer,
    )


async def main() -> None:
    for model_name in USER_MODELS:
        supported = supported_reasoning_efforts(model_name)
        efforts = [effort for effort in _EFFORT_ORDER if effort in supported]
        print(f"\n=== {model_name} — efforts: {', '.join(efforts)} ===")
        for label, prompt in USER_PROMPTS:
            print(f"\n[{label}] {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
            print(f"{'effort':<8} {'api rsn tok':>12} {'est rsn tok':>12} {'output tok':>11} {'seconds':>8}  answer")
            for effort in efforts:
                api_reasoning, estimated, output_tokens, elapsed, answer = await _measure(model_name, effort, prompt)
                print(f"{effort:<8} {api_reasoning:>12} {estimated:>12} {output_tokens:>11} {elapsed:>8.1f}  {answer}")


if __name__ == "__main__":
    asyncio.run(main())

-------

packages/sta_agent_engine/src/sta_agent_engine/models/custom_chat_model.py
----
import contextlib
import logging
import os
import ssl
import warnings
from collections.abc import Mapping
from typing import Any, TypedDict, cast

import certifi
import httpx
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.block_translators import get_translator, register_translator
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.chat_models import global_ssl_context as _mistral_global_ssl_context
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import base as openai_base
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

from sta_agent_core.config import BaseProviderSettings, ProviderFactory
from sta_agent_core.types import ProviderType

from ..utils.signature_utils import expose_merged_signature
from .capabilities import is_multimodal
from .reasoning import build_reasoning_kwargs, merge_reasoning_config


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSL audit helper — diagnostic hook for trust-store issues
# ---------------------------------------------------------------------------
# Silent by default. Enable with STA_SSL_AUDIT=1 in the deployment env.
#
# When enabled, each call to _create_mistral_model / _create_openai_model
# emits a "ssl-audit provider=..." log line containing:
#   - cdc_module / cdc_qualname: who owns ssl.create_default_context right
#     now (vanilla 'ssl', 'truststore._api', 'pip_system_certs.*', ...).
#     Tells you whether any trust-store patch is actually active in the
#     worker process at call time.
#   - SSL_CERT_FILE / SSL_CERT_DIR: whether the runtime env vars are set.
#   - certifi.where(): which bundle certifi would hand out by default.
#   - ca_count: number of CAs loaded into a fresh default context. Compare
#     across providers and across failing/working deploys. ~150 = healthy
#     OS store; 0 = trust store is not being read; ~1 = only corp CA (bad).
#
# If SSL ever breaks again (cert rotation, Wolfi upgrade, LangGraph Platform
# runtime change), grep 'ssl-audit' in production logs for a one-shot
# diagnosis instead of multi-session debugging.
# ---------------------------------------------------------------------------
_SSL_AUDIT_ENABLED = os.environ.get("STA_SSL_AUDIT") == "1"
_ssl_audit_log = logging.getLogger("ssl-audit")


def _describe_ctx_ca_source(ctx: ssl.SSLContext) -> str:
    """Return a short string describing where ``ctx`` gets its CAs from.

    On truststore-backed contexts (truststore.SSLContext or
    pip._vendor.truststore.SSLContext), ``get_ca_certs`` and ``cert_store_stats``
    both raise ``NotImplementedError`` by design — trust validation is delegated
    to the OS. In that case we report ``os-delegated``. Otherwise we try to get
    a numeric count and fall back to ``unknown`` if both probes fail.
    """
    ctx_cls = type(ctx)
    ctx_mod = getattr(ctx_cls, "__module__", "") or ""
    if "truststore" in ctx_mod:
        return f"os-delegated({ctx_mod})"
    try:
        stats = ctx.cert_store_stats()
        return f"cert_store_stats={stats}"
    except NotImplementedError:
        pass
    except Exception as e:  # noqa: BLE001
        return f"cert_store_stats-error={e!r}"
    try:
        return f"get_ca_certs_len={len(ctx.get_ca_certs())}"
    except Exception as e:  # noqa: BLE001
        return f"unknown({e!r})"


def _log_ssl_state(provider_label: str) -> None:
    """Emit one per-call snapshot of the process's SSL trust state.

    No-op unless STA_SSL_AUDIT=1 is set in the environment. Never raises —
    diagnostic hooks must not interfere with the caller's control flow.

    The goal of this helper is to answer, at every call site, two questions
    that are otherwise impossible to tell apart in production:

    1. Is a trust-store delegation shim active right now (truststore,
       pip-system-certs, etc.), or is it vanilla ``ssl``?
    2. What CA bundle would a *fresh* default context actually see at this
       exact moment — same as at import time, or has something changed?
    """
    if not _SSL_AUDIT_ENABLED:
        return
    try:
        cdc = ssl.create_default_context
        ctx = cdc()
        _ssl_audit_log.warning(
            "ssl-audit provider=%s cdc_module=%r cdc_qualname=%r ctx_type=%s SSL_CERT_FILE=%r SSL_CERT_DIR=%r certifi=%r ca_source=%s",
            provider_label,
            getattr(cdc, "__module__", "?"),
            getattr(cdc, "__qualname__", "?"),
            type(ctx).__module__ + "." + type(ctx).__qualname__,
            os.environ.get("SSL_CERT_FILE"),
            os.environ.get("SSL_CERT_DIR"),
            certifi.where(),
            _describe_ctx_ca_source(ctx),
        )
    except Exception as e:  # noqa: BLE001 — logging must never break a request
        _ssl_audit_log.warning("ssl-audit provider=%s inspect failed: %r", provider_label, e)


def _log_httpx_client_ssl_state(provider_label: str, async_client: Any) -> None:
    """Introspect an already-constructed httpx AsyncClient's SSL context.

    httpx freezes its ssl context inside AsyncClient.__init__, so the process-wide
    state logged by _log_ssl_state() is necessary but not sufficient — we also want
    to see what the client actually captured. httpx's internal attribute names
    change between versions, so this is best-effort and silently skips on mismatch.
    """
    if not _SSL_AUDIT_ENABLED:
        return
    try:
        transport = getattr(async_client, "_transport", None)
        pool = getattr(transport, "_pool", None) if transport is not None else None
        sc = getattr(pool, "_ssl_context", None) or getattr(transport, "_ssl_context", None) or None
        if sc is None:
            _ssl_audit_log.warning(
                "ssl-audit provider=%s:client introspection unsupported for this httpx version",
                provider_label,
            )
            return
        _ssl_audit_log.warning(
            "ssl-audit provider=%s:client ctx_type=%s ca_source=%s",
            provider_label,
            type(sc).__module__ + "." + type(sc).__qualname__,
            _describe_ctx_ca_source(sc),
        )
    except Exception as e:  # noqa: BLE001
        _ssl_audit_log.warning("ssl-audit provider=%s:client inspect failed: %r", provider_label, e)


# Reasoning/thinking keywords across providers (DeepSeek, OpenAI o-series, etc.)
_REASONING_KEYWORDS = (
    "reasoning_content",
    "reasoning",
    "reasoning_block",
    "thinking_content",
    "think",
    "thinking",
    "thinking_block",
    "think_content",
)


def _extract_reasoning(source: Mapping[str, Any], target: dict[str, Any]) -> None:
    """Extract reasoning content from a response dict into additional_kwargs.

    Checks multiple provider-specific keys and normalises them under a single
    ``reasoning_content`` key so downstream consumers have a stable interface.
    """
    for keyword in _REASONING_KEYWORDS:
        value = source.get(keyword)
        if value is not None:
            target["reasoning_content"] = value
            return


def _coerce_reasoning_to_text(value: Any) -> str:
    """Coerce a captured reasoning value of any shape into display text.

    A ``content_blocks`` reasoning block carries a string ``reasoning`` field,
    but a provider may surface reasoning as a structured object rather than a
    plain string. Rules:

    - ``str`` → returned unchanged (the common gpt-oss / vLLM case).
    - a reasoning-shaped ``dict`` (``{"type": "reasoning", "content": ...}``) →
      its ``content`` (string-coerced).
    - anything else → ``str(value)``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("type") == "reasoning" and value.get("content") is not None:
        content = value["content"]
        return content if isinstance(content, str) else str(content)
    return str(value)


# ---------------------------------------------------------------------------
# Streaming patch — _convert_delta_to_message_chunk
# ---------------------------------------------------------------------------


def _custom_convert_delta_to_message_chunk(_dict: Mapping[str, Any], default_class: type[BaseMessageChunk]) -> BaseMessageChunk:
    """
    Custom version of _convert_delta_to_message_chunk with reasoning content support.

    This function extends the original langchain_openai conversion to handle reasoning
    tokens from models that support chain-of-thought reasoning (e.g., DeepSeek, o1).

    Args:
        _dict: Dictionary containing delta message information
        default_class: Default message chunk class to use

    Returns:
        BaseMessageChunk with reasoning content in additional_kwargs if present
    """
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}

    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call

    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        with contextlib.suppress(KeyError):
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]

    _extract_reasoning(_dict, additional_kwargs)

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        )
    if role in ("system", "developer") or default_class == SystemMessageChunk:
        additional_kwargs = {"__openai_role__": "developer"} if role == "developer" else {}
        return SystemMessageChunk(content=content, id=id_, additional_kwargs=additional_kwargs)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"], id=id_)  # type: ignore[call-arg]
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    return default_class(content=content, id=id_)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Non-streaming patch — _convert_dict_to_message
# ---------------------------------------------------------------------------


def _custom_convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Custom version of _convert_dict_to_message with reasoning content support.

    The original langchain_openai implementation silently drops ``reasoning_content``
    (and similar provider-specific thinking fields) from the full API response dict.
    This patched version mirrors the original logic and additionally captures reasoning
    content into ``additional_kwargs["reasoning_content"]`` for observability.

    Args:
        _dict: Dictionary from the OpenAI API response ``choices[].message``.

    Returns:
        BaseMessage with reasoning content in additional_kwargs when present.
    """
    role = _dict.get("role")
    name = _dict.get("name")
    id_ = _dict.get("id")
    content_raw = _dict.get("content")
    content_safe = content_raw if content_raw is not None else ""

    if role == "user":
        return HumanMessage(content=content_safe, id=id_, name=name)

    if role == "assistant":
        content = content_safe
        additional_kwargs: dict = {}

        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)

        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(make_invalid_tool_call(raw_tool_call, str(e)))

        if audio := _dict.get("audio"):
            additional_kwargs["audio"] = audio

        _extract_reasoning(_dict, additional_kwargs)

        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )

    if role in ("system", "developer"):
        additional_kwargs = {"__openai_role__": role} if role == "developer" else {}
        return SystemMessage(
            content=content_safe,
            name=name,
            id=id_,
            additional_kwargs=additional_kwargs,
        )

    if role == "function":
        return FunctionMessage(content=content_safe, name=cast(str, _dict.get("name")), id=id_)

    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=content_safe,
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )

    return ChatMessage(content=content_safe, role=role or "assistant", id=id_)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Monkey patches — reasoning content support for langchain_openai
# ---------------------------------------------------------------------------
# These patches mirror upstream functions from langchain_openai.chat_models.base
# and add reasoning content extraction. They are fragile — when langchain-openai
# is upgraded, the upstream functions may change and our patches silently diverge.
#
# Drift detection: tests/test_ai_engine/models/test_custom_chat_model.py contains
# hash-based tests that fail when the upstream source changes, prompting a review.
#
# Patched against: langchain-openai==1.1.9
# Upstream functions: _convert_delta_to_message_chunk, _convert_dict_to_message
# ---------------------------------------------------------------------------

# Save originals before patching (used by drift-detection tests)
_original_convert_delta_to_message_chunk = openai_base._convert_delta_to_message_chunk
_original_convert_dict_to_message = openai_base._convert_dict_to_message

openai_base._convert_delta_to_message_chunk = _custom_convert_delta_to_message_chunk
openai_base._convert_dict_to_message = _custom_convert_dict_to_message
logger.info("Applied custom reasoning parsing logic to langchain_openai (streaming + non-streaming)")


# ---------------------------------------------------------------------------
# content_blocks reasoning — reasoning-aware OpenAI block translator
# ---------------------------------------------------------------------------
# langchain-core derives ``AIMessage.content_blocks`` via a per-provider
# translator selected by ``response_metadata["model_provider"]``. langchain-openai
# stamps ``"openai"`` on every chunk, so core routes to its OpenAI chat-completions
# translator — which builds blocks only from ``.content`` + tool calls and ignores
# ``additional_kwargs`` entirely. Core's best-effort reasoning fallback lives
# *below* that provider branch and is never reached, so the reasoning captured by
# the converter patches above (``additional_kwargs["reasoning_content"]``) is
# dropped from ``content_blocks`` — a reasoning-only stream chunk yields ``[]``.
#
# We wrap the registered OpenAI translator to re-add a reasoning block from
# ``additional_kwargs`` when one is present and the underlying translator did not
# already emit it. Purely additive: text / tool-call / multimodal blocks are
# untouched, and a responses-API message that already carries a reasoning block
# is left alone. Note: ``register_translator`` mutates a process-wide registry, so
# this affects ``content_blocks`` for every ``model_provider="openai"`` message in
# the host process — the same global-patch posture as the converter patches above.
# ---------------------------------------------------------------------------


def _reasoning_block_from_message(message: BaseMessage) -> dict[str, Any] | None:
    """Build a ``{"type": "reasoning", ...}`` block from ``additional_kwargs`` if present."""
    raw = getattr(message, "additional_kwargs", {}).get("reasoning_content")
    if raw is None:
        return None
    text = _coerce_reasoning_to_text(raw)
    return {"type": "reasoning", "reasoning": text} if text else None


def _wrap_translator_with_reasoning(translate: Any) -> Any:
    """Wrap a core block translator so it prepends a reasoning block when missing."""

    def _translate(message: Any) -> Any:
        blocks = translate(message)
        already_has_reasoning = any(isinstance(block, dict) and block.get("type") == "reasoning" for block in blocks)
        if not already_has_reasoning and (reasoning_block := _reasoning_block_from_message(message)) is not None:
            blocks.insert(0, reasoning_block)
        return blocks

    return _translate


_original_openai_translator = get_translator("openai")
if _original_openai_translator is not None:
    register_translator(
        "openai",
        _wrap_translator_with_reasoning(_original_openai_translator["translate_content"]),
        _wrap_translator_with_reasoning(_original_openai_translator["translate_content_chunk"]),
    )
    logger.info("Registered reasoning-aware content_blocks translator for provider 'openai'")


# ---------------------------------------------------------------------------
# L1 stall fence — granular httpx timeout for create_chat_model
# ---------------------------------------------------------------------------
# Phase 3 (#47) of twin_router robustness. Every chat model created here
# inherits a per-axis httpx.Timeout so a hung upstream is killed at 30s of
# byte-level silence rather than 600s of wall clock. Every axis is
# env-tunable (CHAT_MODEL_HTTPX_{CONNECT,READ,WRITE,POOL}_TIMEOUT_S) and
# per-call overridable via ``timeouts=``; the read axis carries the real
# stall budget (30s) while connect / write / pool sit at a modest 10s.
# ---------------------------------------------------------------------------
DEFAULT_HTTPX_CONNECT_TIMEOUT_S = 10.0
DEFAULT_HTTPX_WRITE_TIMEOUT_S = 30.0
DEFAULT_HTTPX_POOL_TIMEOUT_S = 10.0
DEFAULT_HTTPX_READ_TIMEOUT_S = 50.0

_HTTPX_TIMEOUT_AXES: tuple[str, ...] = ("connect", "read", "write", "pool")
_HTTPX_DEFAULTS: dict[str, float] = {
    "connect": DEFAULT_HTTPX_CONNECT_TIMEOUT_S,
    "read": DEFAULT_HTTPX_READ_TIMEOUT_S,
    "write": DEFAULT_HTTPX_WRITE_TIMEOUT_S,
    "pool": DEFAULT_HTTPX_POOL_TIMEOUT_S,
}
_HTTPX_ENV_VARS: dict[str, str] = {
    "connect": "CHAT_MODEL_HTTPX_CONNECT_TIMEOUT_S",
    "read": "CHAT_MODEL_HTTPX_READ_TIMEOUT_S",
    "write": "CHAT_MODEL_HTTPX_WRITE_TIMEOUT_S",
    "pool": "CHAT_MODEL_HTTPX_POOL_TIMEOUT_S",
}
_TIER_ONLY_CONFIG_KEYS = frozenset({"big_model", "small_model", "thinking_model", "multimodal_model", "tier"})


def _warn_legacy_read_timeout_s() -> None:
    """Emit a one-line UserWarning steering callers to the `timeouts` API."""
    warnings.warn(
        "`read_timeout_s` is removed — use `timeouts={'read': N}` instead.",
        UserWarning,
        stacklevel=3,
    )


class TimeoutOverrides(TypedDict, total=False):
    """Per-axis httpx timeout overrides for chat-model factories.

    All keys optional; omitted axes fall back to env / module defaults.
    """

    connect: float
    read: float
    write: float
    pool: float


def _resolve_httpx_timeout(
    timeouts: TimeoutOverrides | httpx.Timeout | None = None,
) -> httpx.Timeout:
    """Return the granular httpx timeout for chat-model clients.

    Precedence per axis (highest first):
        1. ``timeouts`` argument — explicit per-axis override (dict or
           ``httpx.Timeout``). The single per-call override surface.
        2. ``CHAT_MODEL_HTTPX_{AXIS}_TIMEOUT_S`` env var — operator
           rollback knob, applies process-wide.
        3. ``DEFAULT_HTTPX_{AXIS}_TIMEOUT_S`` — module default.
    """
    overrides: dict[str, float] = {}
    if isinstance(timeouts, httpx.Timeout):
        for axis in _HTTPX_TIMEOUT_AXES:
            value = getattr(timeouts, axis, None)
            if value is not None:
                overrides[axis] = float(value)
    elif timeouts is not None:
        overrides = {axis: float(timeouts[axis]) for axis in _HTTPX_TIMEOUT_AXES if axis in timeouts}

    resolved: dict[str, float] = {}
    for axis in _HTTPX_TIMEOUT_AXES:
        if axis in overrides:
            resolved[axis] = overrides[axis]
            continue
        raw = os.environ.get(_HTTPX_ENV_VARS[axis])
        resolved[axis] = float(raw) if raw else _HTTPX_DEFAULTS[axis]

    return httpx.Timeout(**resolved)


def _env_prefix_for_provider(provider: ProviderType | str | None) -> str:
    """Return the env-var prefix that ``ProviderFactory`` reads for ``provider``.

    Used to build human-readable error messages naming the exact env vars
    a caller should set. ``ProviderType.CUSTOM`` and ``None`` map to the
    empty prefix (bare ``API_KEY`` / ``BASE_URL`` / ``MODEL``).
    """
    if isinstance(provider, ProviderType):
        return "" if provider == ProviderType.CUSTOM else f"{provider.value.upper()}_"
    if isinstance(provider, str) and provider:
        return f"{provider.upper()}_"
    return ""


def _warn_silent_openai_fallback(provider: ProviderType | str | None, missing: list[str]) -> None:
    """Emit a DeprecationWarning when the OpenAI dispatch is about to fall back
    to ``OPENAI_API_KEY`` + ``api.openai.com`` from the process environment.

    The silent fallback is a long-standing footgun: callers who pass a provider
    name but forget to set its env vars get a 404 from OpenAI's default endpoint
    for whatever model name was inferred. Warn now, hard-fail in 0.10.0.

    When the caller is not already using ``provider="openai"``, append a hint
    pointing at the default-registered OpenAI provider — most accidental
    fall-throughs are actually OpenAI usage that should be made explicit.
    """
    prefix = _env_prefix_for_provider(provider)
    env_vars = [f"{prefix}{field.upper()}" for field in missing]
    hint = ""
    is_openai_provider = isinstance(provider, str) and provider.lower() == "openai"
    if not is_openai_provider:
        hint = " If you are actually using OpenAI, switch to create_chat_model('openai') — it reads OPENAI_API_KEY natively."
    warnings.warn(
        f"Provider {provider!r} has no {missing} resolvable from env vars or "
        f"kwargs. ChatOpenAI will silently fall back to OPENAI_API_KEY + "
        f"api.openai.com from the process environment. Set {env_vars} or pass "
        f"{missing} as kwargs to create_chat_model(). The silent fallback is "
        f"deprecated and will raise in 0.10.0.{hint}",
        DeprecationWarning,
        stacklevel=3,
    )


def _is_mistral_model(model_name: str | None, provider: ProviderType | str | None) -> bool:
    """Check if the configuration indicates a Mistral model.

    Args:
        model_name: The model name from configuration.
        provider: The provider type. Either a ``ProviderType`` member or
            an arbitrary string. When a string carries ``"mistral"`` as a
            substring (e.g. ``"mistral_eu"``, ``"mistral_corp"``) the
            dispatch routes to ``ChatMistralAI`` even if the model name
            doesn't carry the brand.

    Returns:
        True if this is a Mistral model/provider.
    """
    provider_name_hits = isinstance(provider, str) and "mistral" in provider.lower()
    return (
        provider == ProviderType.MISTRAL
        or provider_name_hits
        or bool(model_name and "mistral" in model_name.lower())
        or bool(model_name and "devstral" in model_name.lower())
        or bool(model_name and "magistral" in model_name.lower())
    )


def _create_mistral_model(config: dict[str, Any]) -> ChatMistralAI:
    """Create a ChatMistralAI instance with the L1 stall fence injected.

    ChatMistralAI handles Mistral's message format correctly and doesn't include
    unsupported fields like 'name' in assistant messages.

    L1 (Phase 3 / #47): pre-built httpx clients carry the granular
    ``_resolve_httpx_timeout()`` and the bearer token so that the SDK's
    relative-path requests (``self.async_client.post('/chat/completions')``)
    resolve correctly. ``max_retries=0`` is forced because Mistral SDK
    retries restart the httpx timeout and don't honour Retry-After — leaving
    the SDK default would inflate our 30s stall budget by 6×.

    The SDK's own SSL context (``langchain_mistralai.chat_models.global_ssl_context``)
    is reused so truststore delegation on Wolfi stays consistent with the
    behaviour the SDK would have produced if we hadn't injected a client.
    """
    _log_ssl_state("mistral")

    api_key = config.pop("api_key", None) or os.environ.get("MISTRAL_API_KEY")
    base_url = config.pop("base_url", None) or os.environ.get("MISTRAL_BASE_URL") or "https://api.mistral.ai/v1"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # Per-call timeout override — pass ``timeouts={"connect": 10, "read": 90, ...}``
    # or an ``httpx.Timeout`` instance. The legacy ``read_timeout_s`` kwarg
    # is removed; map any stray callers to the new API.
    if "read_timeout_s" in config:
        _warn_legacy_read_timeout_s()
        config.pop("read_timeout_s", None)
    timeouts_override = config.pop("timeouts", None)
    timeout = _resolve_httpx_timeout(timeouts=timeouts_override)

    async_client = config.pop("async_client", None) or httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=timeout,
        verify=_mistral_global_ssl_context,
    )
    sync_client = config.pop("client", None) or httpx.Client(
        base_url=base_url,
        headers=headers,
        timeout=timeout,
        verify=_mistral_global_ssl_context,
    )

    _log_httpx_client_ssl_state("mistral", async_client)

    mistral_config: dict[str, Any] = {
        "api_key": api_key,
        "endpoint": base_url,  # ChatMistralAI Field alias for base_url
        "model": config.pop("model", None),
        "temperature": config.pop("temperature", None),
        "max_tokens": config.pop("max_tokens", None),
        "top_p": config.pop("top_p", None),
        # Force max_retries=0 — Mistral SDK retries restart the httpx timeout
        # and don't honour Retry-After. Let with_retry() / agent middleware
        # own retry semantics instead.
        "max_retries": 0,
    }

    if model_kwargs := config.pop("model_kwargs", None):
        mistral_config["model_kwargs"] = model_kwargs

    # Drop the SDK's own ``timeout`` from the caller's config — the granular
    # httpx.Timeout on the injected client is the source of truth now.
    config.pop("timeout", None)
    config.pop("max_retries", None)

    # Filter out None values from the explicit fields we set above.
    mistral_config = {k: v for k, v in mistral_config.items() if v is not None}

    # Anything else the caller passed (custom Mistral kwargs) flows through.
    mistral_config = {**config, **mistral_config}

    logger.info(f"Creating ChatMistralAI with model={mistral_config.get('model')}")
    return ChatMistralAI(client=sync_client, async_client=async_client, **mistral_config)


def _create_openai_model(config: dict[str, Any]) -> ChatOpenAI:
    """Create a ChatOpenAI instance with the L1 stall fence injected.

    L1 (Phase 3 / #47): pre-built httpx clients carry the granular
    ``_resolve_httpx_timeout()`` so the openai SDK's transport inherits a
    30s read fence instead of the previous 600s blanket. ``max_retries=1``
    keeps one Retry-After-aware retry for 429 storms — OpenAI's SDK retries
    are well-behaved here, unlike Mistral's.
    """
    _log_ssl_state("openai")

    # Per-call timeout override — pass ``timeouts={"connect": 10, "read": 90, ...}``
    # or an ``httpx.Timeout`` instance. Legacy ``read_timeout_s`` is removed.
    if "read_timeout_s" in config:
        _warn_legacy_read_timeout_s()
        config.pop("read_timeout_s", None)
    timeouts_override = config.pop("timeouts", None)
    timeout = _resolve_httpx_timeout(timeouts=timeouts_override)

    async_client = config.pop("http_async_client", None) or DefaultAsyncHttpxClient(timeout=timeout)
    sync_client = config.pop("http_client", None) or DefaultHttpxClient(timeout=timeout)

    _log_httpx_client_ssl_state("openai", async_client)

    config.setdefault("max_retries", 1)

    return ChatOpenAI(
        http_async_client=async_client,
        http_client=sync_client,
        profile={"max_input_tokens": 128_000},
        **config,
    )


def _resolve_multimodal_model(settings: BaseProviderSettings, provider: str | ProviderType | None) -> str:
    """Resolve the model to use for a multimodal (vision) request.

    Resolution order (hard-fails rather than silently returning a text model):

        1. ``settings.multimodal_model`` if set — authoritative, used verbatim.
           Not re-checked against ``is_multimodal`` so an operator can declare a
           vision model the static capability list doesn't yet recognize.
        2. ``settings.model`` if ``is_multimodal(settings.model)`` — the default
           already points at a vision-capable model.
        3. Otherwise raise ``ValueError`` — ``multimodal=True`` was requested but
           the deployment has no multimodal model configured. Failing loud at
           construction beats handing back a text model that silently strips
           images (under the guard middleware) or 400s upstream.

    Args:
        settings: Resolved provider settings.
        provider: Original provider argument, for the error message.

    Returns:
        The model identifier to use for the vision request.

    Raises:
        ValueError: If no multimodal model can be resolved.
    """
    if settings.multimodal_model:
        return settings.multimodal_model
    if is_multimodal(settings.model):
        return settings.model
    env_prefix = type(settings).model_config.get("env_prefix") or ""
    provider_label = provider.value if isinstance(provider, ProviderType) else (provider or "default")
    raise ValueError(
        f"multimodal=True but no multimodal model is configured for provider {provider_label!r}: "
        f"set {env_prefix}MULTIMODAL_MODEL, or point {env_prefix}MODEL at a vision-capable model."
    )


# Tier -> the capacity-tier fields whose explicit presence means the resolved model
# did NOT fall through to the base ``model`` default. Mirrors the cascade in
# ``BaseProviderSettings.get_model``.
_TIER_MODEL_FIELDS: dict[str, tuple[str, ...]] = {
    "default": (),
    "big": ("big_model",),
    "small": ("small_model",),
    "thinking": ("thinking_model", "big_model"),
}


def _model_is_implicit_default(settings: BaseProviderSettings, tier: str | None, resolved_model: str) -> bool:
    """True when ``resolved_model`` came from the class default, not an explicit source.

    "Explicit" means a ``*_MODEL`` / ``*_<TIER>_MODEL`` env var, a registered
    default, an init value, or a ``model=`` kwarg. Detection uses the value-vs-
    default comparison rather than ``model_fields_set`` because the
    ``empty_str_to_none`` validator coerces ``*_MODEL=""`` to ``None`` while still
    marking the field "set" — so ``model_fields_set`` reports a falsely-explicit
    model. Comparing the resolved value to the field default is robust to that.

    Known benign false-positive: a consumer who explicitly sets the model to the
    exact same string as the built-in default is still treated as implicit.
    """
    tier_name = "default" if tier is None else str(tier).lower().strip()
    fields_set = settings.model_fields_set
    # An explicitly-set tier field supplied the model -> the base default was not used.
    for field in _TIER_MODEL_FIELDS.get(tier_name, ()):
        if field in fields_set and getattr(settings, field, None):
            return False
    model_field_default = type(settings).model_fields["model"].default
    return resolved_model == model_field_default


def create_chat_model(
    provider: str | ProviderType | None = None,
    *,
    tier: str = "default",
    multimodal: bool = False,
    **kwargs: Any,
) -> ChatOpenAI | ChatMistralAI:
    """
    Factory function to create a chat model with provider-specific configuration.

    Automatically selects the appropriate client based on the provider:
    - ChatMistralAI for Mistral models (handles Mistral's stricter message format)
    - ChatOpenAI for OpenAI-compatible APIs

    Built-in providers:
    - LLMaaS: Set LLM_PROVIDER=llmaas, configure with LLMAAS_* env vars
    - LLMaaS Dev: Set LLM_PROVIDER=llmaas_dev, configure with LLMAAS_DEV_* env vars
    - Mistral: Set LLM_PROVIDER=mistral, configure with MISTRAL_* env vars
    - Custom: Set LLM_PROVIDER=custom, configure with {NO_PREFIX}* env vars

    Dynamic providers:
        Any other string is accepted. The env prefix is derived as
        ``f"{NAME.upper()}_"`` and read via Pydantic Settings:

        - ``{NAME}_API_KEY``
        - ``{NAME}_BASE_URL``
        - ``{NAME}_MODEL``
        - ``{NAME}_BIG_MODEL`` / ``{NAME}_SMALL_MODEL`` / ``{NAME}_THINKING_MODEL``
        - ``{NAME}_MULTIMODAL_MODEL``
        - ``{NAME}_TEMPERATURE`` / ``{NAME}_TOP_P`` / ``{NAME}_MAX_TOKENS``

        Use ``ProviderFactory.register(name, defaults=..., env_prefix=...)``
        for non-env defaults or a custom env prefix.

    Model tiers:
        ``tier`` selects a provider model slot when no explicit ``model=`` kwarg
        is passed. Supported tiers are ``default``, ``big``, ``small``, and
        ``thinking``. ``thinking`` cascades to ``big`` and then ``model``.

    Multimodal:
        ``multimodal=True`` requests a vision-capable model (capability axis,
        not a capacity tier). It resolves ``{NAME}_MULTIMODAL_MODEL`` if set,
        else ``{NAME}_MODEL`` when that is a recognized multimodal model, else
        raises ``ValueError`` — it never silently returns a text-only model.
        ``multimodal=True`` takes precedence over ``tier`` and is ignored when
        an explicit ``model=`` kwarg is passed.

    Mistral dispatch:
        Routes to ``ChatMistralAI`` when any of these hold:

        - ``provider == ProviderType.MISTRAL``
        - The provider name (string) contains ``"mistral"``
        - The model name contains ``"mistral"`` / ``"devstral"`` / ``"magistral"``

    Credential resolution:
        ``api_key`` and ``base_url`` should resolve via env vars (under the
        provider's prefix) or kwargs. When the OpenAI dispatch path is taken
        and either is missing, ``ChatOpenAI`` silently falls back to
        ``OPENAI_API_KEY`` + ``api.openai.com`` from the process env. This
        emits a ``DeprecationWarning`` and will raise ``ValueError`` in 0.10.0.

    Reasoning control:
        ``reasoning_effort="off" | "low" | "medium" | "high" | "xhigh" | "max"`` is
        translated per model family into the kwargs the model actually honors
        (native GLM/DeepSeek/Qwen3.8 effort, Mistral's ``reasoning_effort``, or
        Nemotron/earlier-Qwen/Gemma 4 ``chat_template_kwargs`` — see
        ``models/reasoning.py``). Omitted -> nothing is injected (the
        model keeps its server-side default). Unsupported efforts warn and
        never silently substitute; explicitly-passed ``extra_body`` /
        ``model_kwargs`` win over translated values on conflicting keys.
        ``reasoning_family="nemotron-super"`` pins the family when a gateway
        alias hides the model name. Prefer one effort per agent/task/thread —
        flipping ``chat_template_kwargs`` per turn busts server-side prefix
        caches. Consumer guide: ``docs/consuming/reasoning.md``.

    Args:
        provider: ``ProviderType`` member, built-in provider string,
            or any arbitrary dynamic-provider name.
        tier: Capacity tier to resolve from provider settings when ``model``
            is not passed explicitly.
        multimodal: When True, resolve a vision-capable model (see Multimodal
            above). Takes precedence over ``tier``; ignored if ``model=`` is set.
        **kwargs: Additional configuration to override environment variables.

    Returns:
        BaseChatModel instance (ChatMistralAI or ChatOpenAI) configured for the specified provider.

    Examples:
        # Built-in provider via environment variables
        export LLM_PROVIDER=mistral
        export MISTRAL_API_KEY=your_api_key
        export MISTRAL_MODEL=mistral-small-2603
        model = create_chat_model()

        # Dynamic provider — no code change required
        export ACME_API_KEY=your_key
        export ACME_BASE_URL=https://api.acme.test/v1
        export ACME_MODEL=acme-large
        model = create_chat_model("acme")

        # Explicit Mistral provider
        model = create_chat_model("mistral")

        # Mistral-flavored dynamic provider (auto-routes to ChatMistralAI)
        model = create_chat_model("mistral_eu", api_key="...", base_url="https://eu.mistral.test", model="mistral-large")
    """
    # Lowercase string providers so dispatch matches the factory's normalization,
    # but keep the original string when it doesn't map to a known enum member.
    if isinstance(provider, str):
        try:
            provider = ProviderType(provider.lower())
        except ValueError:
            provider = provider.lower()

    provider_settings = ProviderFactory.get_provider_settings(provider)
    settings = provider_settings.model_dump()

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if "model" not in kwargs:
        if multimodal:
            settings["model"] = _resolve_multimodal_model(provider_settings, provider)
        else:
            settings["model"] = provider_settings.get_model(tier)

    # Extract credentials from context if available (BYOK mode).
    # ``pop`` (not lookup): these are aliases for ``api_key`` / ``base_url``, not
    # client constructor params. Leaving them in ``kwargs`` would merge them into
    # ``config`` below and forward them to ChatOpenAI/ChatMistralAI, which route
    # unknown kwargs into ``model_kwargs`` — leaking the raw key into the request
    # body sent over the wire and 400-ing strict OpenAI-compatible servers.
    if kwargs:
        if (provider_api_key := kwargs.pop("provider_api_key", None)) is not None:
            settings["api_key"] = provider_api_key
        if (provider_base_url := kwargs.pop("provider_base_url", None)) is not None:
            settings["base_url"] = provider_base_url

    config = {k: v for k, v in {**settings, **kwargs}.items() if v is not None}
    for key in _TIER_ONLY_CONFIG_KEYS:
        config.pop(key, None)

    # Every provider carries a default model slug, so a model always resolves and
    # construction never fails for lack of one. But relying on that *implicit*
    # default is deprecated: a consumer who never configured a model is almost
    # certainly pointing the default slug at an endpoint that does not serve it
    # (a wrong default is dangerous for third-party providers). Warn when the
    # resolved model came from the class default rather than an explicit source —
    # a *_MODEL / *_<TIER>_MODEL env var, a registered default, or model=... — and
    # keep working for now. This becomes a hard ValueError in 0.11.0. The
    # multimodal path is exempt: it already hard-fails when nothing is configured.
    if "model" not in kwargs and not multimodal and _model_is_implicit_default(provider_settings, tier, config.get("model", "")):
        prefix = type(provider_settings).model_config.get("env_prefix") or ""
        warnings.warn(
            f"create_chat_model({provider!r}) is using the built-in default model {config.get('model')!r}. "
            "Relying on an implicit default is deprecated and will raise in 0.11.0. Set "
            f"{prefix}MODEL (or {prefix}<TIER>_MODEL), register a default via "
            "ProviderFactory.register(name, defaults={'model': ...}), or pass model=... explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Determine if this is a Mistral model and use appropriate client.
    # Mistral's SDK has its own legitimate base_url fallback to api.mistral.ai —
    # do NOT warn here. The footgun only exists on the OpenAI dispatch path.
    model_name = config.get("model", "")

    # Reasoning-effort resolution (see models/reasoning.py): translate the
    # normalized effort into the per-family request kwargs. Explicit caller
    # kwargs win over translated values on leaf conflicts — raw extra_body /
    # model_kwargs are the escape hatch. ``reasoning_family`` pins the family
    # when a gateway alias hides the model name; popped unconditionally so it
    # never leaks into the client constructor.
    reasoning_family = config.pop("reasoning_family", None)
    if (reasoning_effort := config.pop("reasoning_effort", None)) is not None:
        provider_str = str(provider) if provider is not None else None
        reasoning_config = build_reasoning_kwargs(model_name, reasoning_effort, provider=provider_str, family=reasoning_family)
        config, reasoning_overrides = merge_reasoning_config(reasoning_config, config)
        if reasoning_overrides:
            warnings.warn(
                f"reasoning_effort={reasoning_effort!r}: explicitly-passed kwargs override the translated reasoning keys "
                f"{', '.join(sorted(reasoning_overrides))} — the explicit values win.",
                UserWarning,
                stacklevel=2,
            )
    elif reasoning_family is not None:
        warnings.warn(
            f"reasoning_family={reasoning_family!r} was given without reasoning_effort — it selects the translation table "
            "but injects nothing on its own, so it was ignored. Pass reasoning_effort=... alongside it.",
            UserWarning,
            stacklevel=2,
        )
    if _is_mistral_model(model_name, provider):
        logger.info(f"Detected Mistral model/provider, using ChatMistralAI for model={model_name}")
        return _create_mistral_model(config)

    # OpenAI dispatch — if api_key / base_url didn't resolve, ChatOpenAI will
    # silently pick up OPENAI_API_KEY + api.openai.com from the process env.
    # Restored to preserve main-branch behavior; the silent fallback is now
    # deprecated and will raise in 0.10.0.
    missing: list[str] = []
    if not config.get("api_key"):
        missing.append("api_key")
    if not config.get("base_url"):
        missing.append("base_url")
    if missing:
        _warn_silent_openai_fallback(provider, missing)

    return _create_openai_model(config)


# Expose merged signature for better IDE support and introspection
expose_merged_signature(create_chat_model, ChatOpenAI)

# Backward compatibility alias - can be removed after updating all imports
CustomChatModel = create_chat_model

-------

packages/sta_agent_engine/src/sta_agent_engine/models/reasoning.py
----
"""Declarative reasoning-effort configuration for chat models.

Model families expose incompatible knobs for controlling reasoning/thinking:
Mistral takes a top-level ``reasoning_effort`` string, Nemotron wants booleans
nested under ``extra_body.chat_template_kwargs``, Gemma 4 uses a binary
``enable_thinking`` template flag, and GLM/DeepSeek combine a top-level effort
with a separate thinking-mode toggle.
This module maps one normalized effort vocabulary (``off`` / ``low`` /
``medium`` / ``high`` / ``xhigh`` / ``max``) onto the literal request kwargs each family honors, so
callers write ``reasoning_effort="high"`` once and the right wire format is
emitted for whatever model resolves. Efforts are open strings — a family
registered via :func:`register_reasoning_family` may define additional rungs.

The registry is a plain data table — family name -> effort -> literal kwargs.
Adding a model family is one :func:`register_reasoning_family` call (or one
table entry); no dispatch code changes. The error posture is warn-not-raise:
requesting reasoning must never break model construction. Unknown families
degrade to forwarding ``reasoning_effort`` natively (the OpenAI form), which
genuine OpenAI APIs honor and OpenAI-compatible gateways typically ignore.

Guidance: pick an effort per agent/task/thread, not per turn. For
template-flag families, flipping ``chat_template_kwargs`` between turns of one
conversation re-renders the server-side prompt template and defeats
vLLM/SGLang prefix caching.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any


# Warnings are attributed to the first stack frame OUTSIDE this package
# directory (PEP 678-era ``skip_file_prefixes``, Python 3.12+). A fixed
# ``stacklevel`` can't be right for every entry point (direct call vs. routed
# through create_chat_model adds frames), and misattribution has a second
# cost: Python's default once-per-location filter would collapse every
# consumer call site onto one internal library line, silencing all repeats.
_WARN_SKIP_PREFIXES = (str(Path(__file__).resolve().parent),)


__all__ = [
    "build_reasoning_kwargs",
    "register_reasoning_family",
    "resolve_reasoning_family",
    "supported_reasoning_efforts",
]


# Family spec keys:
#   "match":          model-name patterns. A flat tuple of strings is ONE AND-group:
#                     all substrings must appear. A tuple of tuples is OR-of-AND-groups:
#                     the family matches if ANY group has all its substrings present —
#                     e.g. (("nemotron-3", "ultra"), ("nemo-ultra",)) covers both slug
#                     dialects. Matching is case- AND separator-insensitive: both sides
#                     are normalized by stripping `-`, `_`, `.`, `:`, `/` and spaces, so
#                     ("qwen3",) matches "qwen3.6", "qwen3-6", "Qwen/Qwen3.6-32B", and
#                     "qwen3:32b" alike — provider slug conventions don't matter.
#   "provider_match": substrings matched against the provider name (any hit wins;
#                     same normalization)
#   "native_path":    nested key path for forwarding a raw effort string when the
#                     requested effort has no rung (None -> nothing is injected).
#                     Empty "rungs" + a native_path = pure passthrough family (no warning).
#   "rungs":          effort -> literal constructor kwargs to merge
_FAMILIES: dict[str, dict[str, Any]] = {
    # GLM-5.2 exposes a flat reasoning_effort plus an independent thinking
    # toggle. Its API accepts the compatibility vocabulary below; low/medium
    # are evaluated as high, xhigh as max, and none/minimal stop thinking.
    # ``off`` uses the unambiguous toggle instead of inventing another native
    # effort alias. Keep this before any broader GLM family added later.
    "glm-5.2": {
        "match": ("glm", "5.2"),
        "provider_match": (),
        "native_path": ("reasoning_effort",),
        "rungs": {
            "off": {"extra_body": {"thinking": {"type": "disabled"}}},
            "none": {"reasoning_effort": "none"},
            "minimal": {"reasoning_effort": "minimal"},
            "low": {"reasoning_effort": "low", "extra_body": {"thinking": {"type": "enabled"}}},
            "medium": {"reasoning_effort": "medium", "extra_body": {"thinking": {"type": "enabled"}}},
            "high": {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}},
            "xhigh": {"reasoning_effort": "xhigh", "extra_body": {"thinking": {"type": "enabled"}}},
            "max": {"reasoning_effort": "max", "extra_body": {"thinking": {"type": "enabled"}}},
        },
    },
    # DeepSeek V4 supports only high/max as effective native levels. The API
    # accepts the wider compatibility vocabulary, mapping low/medium -> high
    # and xhigh -> max. Encode the effective values explicitly so the emitted
    # request is stable across official and compatible gateways.
    "deepseek-v4": {
        "match": ("deepseek", "v4"),
        "provider_match": (),
        "native_path": ("reasoning_effort",),
        "rungs": {
            "off": {"extra_body": {"thinking": {"type": "disabled"}}},
            "low": {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}},
            "medium": {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}},
            "high": {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}},
            "xhigh": {"reasoning_effort": "max", "extra_body": {"thinking": {"type": "enabled"}}},
            "max": {"reasoning_effort": "max", "extra_body": {"thinking": {"type": "enabled"}}},
        },
    },
    # Nemotron-3-Ultra: `low` sends force_nonempty_content=False (explicitly off,
    # so a server-side template default can't force it); medium/high carry
    # force_nonempty_content=True (SGLang requires it when tool-calling with
    # thinking enabled; harmless on non-tool calls).
    "nemotron-ultra": {
        "match": ("nemotron-3", "ultra"),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "low": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True, "force_nonempty_content": False}}},
            "medium": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True, "force_nonempty_content": True}}},
            "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "force_nonempty_content": True}}},
        },
    },
    "nemotron-super": {
        "match": ("nemotron-3", "super"),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "low": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}},
            "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        },
    },
    # Qwen3.8-2.4T requires thinking on every request, so it deliberately has no
    # off rung. Its native effort vocabulary is low/medium/xhigh; ``high`` is
    # the library's portable alias for the strongest supported level.
    "qwen3.8-always-thinking": {
        "match": ("qwen3.8", "2.4t"),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "low": {"reasoning_effort": "low"},
            "medium": {"reasoning_effort": "medium"},
            "high": {"reasoning_effort": "xhigh"},
            "xhigh": {"reasoning_effort": "xhigh"},
        },
    },
    # Other Qwen3.8 models expose the same graded native effort plus a hard
    # thinking switch. The built-in off rung targets self-hosted vLLM/SGLang,
    # where the switch rides in chat_template_kwargs. Qwen Cloud instead uses
    # a direct extra_body.enable_thinking field; pass that raw provider kwarg
    # without the normalized off rung when using its endpoint.
    "qwen3.8": {
        "match": ("qwen3.8",),
        "provider_match": (),
        "native_path": ("reasoning_effort",),
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "low": {"reasoning_effort": "low"},
            "medium": {"reasoning_effort": "medium"},
            "high": {"reasoning_effort": "xhigh"},
            "xhigh": {"reasoning_effort": "xhigh"},
        },
    },
    # Earlier Qwen3.x models: thinking on by default; the model card documents
    # only the binary chat_template_kwargs.enable_thinking switch
    # (Qwen/Qwen3.6-27B). Graded thinking budgets are a serving-stack feature
    # (vLLM `thinking_token_budget`, version-dependent) — deliberately NOT baked
    # in; gateways that support them can re-register this family with budget
    # rungs (see docs/consuming/reasoning.md).
    # The card also documents chat_template_kwargs.preserve_thinking=True for
    # keeping reasoning traces across agent turns — orthogonal to effort, pass it
    # via explicit extra_body (it deep-merges alongside these rungs).
    "qwen3": {
        "match": ("qwen3",),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        },
    },
    # Gemma 4 exposes only a binary chat-template thinking switch. Thinking is
    # disabled by default in the official template; there is no documented
    # graded reasoning effort or token budget. ``preserve_thinking`` controls
    # whether earlier assistant reasoning is retained and is orthogonal to
    # effort, so callers may pass it explicitly in chat_template_kwargs.
    "gemma4": {
        "match": ("gemma4",),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        },
    },
    # Mistral small/medium accept only none/high; ChatMistralAI has no native
    # reasoning_effort field, so the value rides in model_kwargs (flattened into
    # the request payload). Unsupported efforts pass through raw via native_path
    # so the Mistral API validates them itself. The match groups mirror
    # _is_mistral_model's dispatch criteria: every model that routes to
    # ChatMistralAI must use this wire dialect ("off" must become "none").
    "mistral": {
        "match": (("mistral",), ("devstral",), ("magistral",)),
        "provider_match": ("mistral",),
        "native_path": ("model_kwargs", "reasoning_effort"),
        "rungs": {
            "off": {"model_kwargs": {"reasoning_effort": "none"}},
            "high": {"model_kwargs": {"reasoning_effort": "high"}},
        },
    },
    # Real OpenAI reasoning models take reasoning_effort natively — pure silent
    # passthrough (the API validates values like "minimal"/"low"/"medium"/"high").
    "openai": {
        "match": ("gpt",),
        "provider_match": (),
        "native_path": ("reasoning_effort",),
        "rungs": {},
    },
}


_SLUG_SEPARATORS = str.maketrans("", "", "-_.:/ ")


def _normalize_slug(name: str) -> str:
    """Canonicalize a model/provider slug for matching.

    Lowercases and strips separator characters (``- _ . : /`` and spaces), so
    provider-specific slug conventions collapse to one form: ``qwen3.8``,
    ``qwen3-8``, ``Qwen/Qwen3.8-27B``, and ``qwen3:27b`` all contain the
    normalized pattern ``qwen38``/``qwen3``.
    """
    return name.lower().translate(_SLUG_SEPARATORS)


def _nested_from_path(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """Build a nested dict placing ``value`` at the given key path."""
    out: Any = value
    for key in reversed(path):
        out = {key: out}
    return out


def _match_substrings(spec_value: Any) -> tuple[str, ...]:
    """Coerce a spec's match value to a tuple of normalized patterns.

    Accepts a lone string as a single pattern — ``("qwen3")`` (missing trailing
    comma) is a string in Python, and iterating it would silently degrade to
    character-wise matching.
    """
    if isinstance(spec_value, str):
        spec_value = (spec_value,)
    return tuple(_normalize_slug(s) for s in spec_value)


def _match_groups(spec_value: Any) -> tuple[tuple[str, ...], ...]:
    """Coerce a spec's ``match`` value to OR-groups of normalized AND-substrings.

    - A lone string -> one group with one substring.
    - A flat tuple of strings -> ONE group (all substrings must match — AND).
    - A tuple containing any nested tuple/list -> every element is its own
      group (string elements become 1-substring groups); the family matches if
      ANY group fully matches (OR of ANDs).
    """
    if isinstance(spec_value, str):
        return ((_normalize_slug(spec_value),),)
    items = tuple(spec_value)
    if any(isinstance(item, (tuple, list)) for item in items):
        return tuple(_match_substrings(item) for item in items)
    return (_match_substrings(items),) if items else ()


def resolve_reasoning_family(model: str, *, provider: str | None = None, family: str | None = None) -> str | None:
    """Resolve which reasoning family applies to a model.

    Precedence: explicit ``family`` > provider substring match > model-name
    substring match (all of a family's ``match`` substrings must appear).
    Matching is case- and separator-insensitive (see :func:`_normalize_slug`),
    so the same model resolves identically across provider slug conventions —
    ``qwen3.8``, ``qwen3-8``, and ``Qwen/Qwen3.8-27B`` are one family.
    An explicit family that is not registered warns and falls back to the
    match-based resolution, so a typo degrades instead of silently no-oping.

    Args:
        model: Resolved model name (as sent to the endpoint).
        provider: Optional provider name (e.g. ``"mistral"``, ``"llmaas"``).
        family: Optional explicit family pin — use when a gateway alias hides
            the real model name (``chat-default`` actually serving Nemotron).

    Returns:
        The family name, or ``None`` if nothing matches.
    """
    if family is not None:
        if family in _FAMILIES:
            return family
        warnings.warn(
            f"Unknown reasoning_family {family!r} (registered: {sorted(_FAMILIES)}); falling back to model-name matching.",
            UserWarning,
            stacklevel=2,
            skip_file_prefixes=_WARN_SKIP_PREFIXES,
        )
    provider_normalized = _normalize_slug(provider or "")
    model_normalized = _normalize_slug(model)
    if provider_normalized:
        for name, spec in _FAMILIES.items():
            if any(sub in provider_normalized for sub in _match_substrings(spec["provider_match"])):
                return name
    for name, spec in _FAMILIES.items():
        for group in _match_groups(spec["match"]):
            if group and all(sub in model_normalized for sub in group):
                return name
    return None


def supported_reasoning_efforts(model: str, *, provider: str | None = None, family: str | None = None) -> frozenset[str]:
    """Return the effort names a model's family defines rungs for.

    An empty set means either no family matched or the family is a pure
    native-passthrough (any value is forwarded unvalidated for the API to judge).
    """
    resolved = resolve_reasoning_family(model, provider=provider, family=family)
    if resolved is None:
        return frozenset()
    return frozenset(_FAMILIES[resolved]["rungs"])


def build_reasoning_kwargs(model: str, effort: str | None, *, provider: str | None = None, family: str | None = None) -> dict[str, Any]:
    """Translate a normalized reasoning effort into model-specific kwargs.

    The returned dict is ready to splat into the model constructor. Families
    that translate into ``extra_body`` or a native field also work per call::

        kw = build_reasoning_kwargs("nemotron-3-super-120b", "low")
        model.invoke(messages, **kw)        # or model.bind(**kw)

    Exception: families whose kwargs ride ``model_kwargs`` (mistral) are
    constructor-only — ``ChatMistralAI`` flattens ``model_kwargs`` into the
    payload only from the constructor field; a call-time kwarg is posted as a
    literal ``"model_kwargs"`` JSON key. Set the effort at construction
    (``create_chat_model(..., reasoning_effort=...)``) for those.

    Behavior:
        - ``effort=None`` (or blank/whitespace) returns ``{}`` (inject nothing —
          blank means "unset", e.g. an empty env-var default).
        - A supported effort returns a fresh copy of the family's rung kwargs.
        - An unsupported effort warns (listing the supported set); families with
          a native passthrough forward the raw value for the API to validate,
          others return ``{}`` (there is no field to receive the value).
        - No family matched: warns and forwards ``{"reasoning_effort": effort}``
          (the OpenAI-native form — gateways that don't know it ignore it).

    Prefer one effort per agent/task/thread over per-turn flips: changing
    ``chat_template_kwargs`` mid-conversation busts server-side prefix caches.
    """
    if effort is None:
        return {}
    effort_name = str(effort).lower().strip()
    if not effort_name:
        return {}
    resolved = resolve_reasoning_family(model, provider=provider, family=family)
    if resolved is None:
        warnings.warn(
            f"No reasoning family matches model {model!r}; forwarding reasoning_effort={effort_name!r} as a native top-level parameter. "
            "OpenAI-style APIs honor it; vLLM/LiteLLM-fronted gateways typically ignore it. "
            "Register the model with register_reasoning_family(...) or pin reasoning_family=... to translate it properly.",
            UserWarning,
            stacklevel=2,
            skip_file_prefixes=_WARN_SKIP_PREFIXES,
        )
        return {"reasoning_effort": effort_name}
    spec = _FAMILIES[resolved]
    rungs: dict[str, dict[str, Any]] = spec["rungs"]
    if effort_name in rungs:
        return deepcopy(rungs[effort_name])
    native_path: tuple[str, ...] | None = spec["native_path"]
    if native_path is not None:
        if rungs:  # known vocabulary exists and the request is outside it
            warnings.warn(
                f"reasoning_effort={effort_name!r} is not defined for family {resolved!r} (model {model!r}; supported: {sorted(rungs)}). "
                "Forwarding the raw value for the API to validate.",
                UserWarning,
                stacklevel=2,
                skip_file_prefixes=_WARN_SKIP_PREFIXES,
            )
        return _nested_from_path(native_path, effort_name)
    warnings.warn(
        f"reasoning_effort={effort_name!r} is not supported by family {resolved!r} (model {model!r}; supported: {sorted(rungs)}). "
        "Nothing was applied — the model keeps its server-side default.",
        UserWarning,
        stacklevel=2,
        skip_file_prefixes=_WARN_SKIP_PREFIXES,
    )
    return {}


def register_reasoning_family(
    name: str,
    rungs: dict[str, dict[str, Any]],
    *,
    match_substrings: str | tuple[str | tuple[str, ...], ...] = (),
    provider_substrings: str | tuple[str, ...] = (),
    native_path: tuple[str, ...] | None = None,
) -> None:
    """Register (or replace) a reasoning family.

    Args:
        name: Family name; re-registering an existing name (including a
            built-in) replaces it, letting consumers override library defaults.
        rungs: Effort name -> literal constructor kwargs to merge (e.g.
            ``{"high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}}``).
        match_substrings: Model-name patterns for auto-detection. A flat tuple
            of strings is ONE AND-group — all substrings must appear:
            ``("nemotron-3", "ultra")``. A tuple of tuples is OR-of-AND-groups —
            the family matches when ANY group fully matches:
            ``(("nemotron-3", "ultra"), ("nemo-ultra",))`` covers two slug
            dialects. Matching is case- and separator-insensitive (``- _ . : /``
            and spaces are stripped from both sides), so ``("qwen3",)`` covers
            ``qwen3.6`` / ``qwen3-6`` / ``Qwen/Qwen3.6-32B`` regardless of the
            provider's slug convention.
        provider_substrings: Provider-name substrings that force this family
            (any hit wins; same normalization).
        native_path: Optional nested key path for forwarding a raw effort value
            when the requested effort has no rung (e.g. ``("reasoning_effort",)``).

    Note:
        Resolution scans families in registration order and the first full
        match wins — a *new* family whose patterns overlap an already-registered
        one (built-ins included) never wins for models both match. To change
        behavior for such models, re-register under the existing family's name.

    Raises:
        ValueError: If a rung value is not a dict (it must be literal kwargs),
            or if any match/provider pattern is empty after normalization
            (an empty pattern would silently match every model).
    """
    for rung_name, rung_kwargs in rungs.items():
        if not isinstance(rung_kwargs, dict):
            raise ValueError(f"Rung {rung_name!r} of family {name!r} must be a dict of constructor kwargs, got {type(rung_kwargs).__name__}")
    # A lone string is accepted as a single pattern (a missing trailing comma in
    # a 1-tuple is a string in Python — don't let it degrade to char matching).
    if isinstance(match_substrings, str):
        match_substrings = (match_substrings,)
    if isinstance(provider_substrings, str):
        provider_substrings = (provider_substrings,)
    # Reject patterns that normalize to nothing: "" is a substring of every
    # name, so an empty pattern (or empty AND-group) would hijack all models.
    for group in _match_groups(match_substrings):
        if not group or any(not sub for sub in group):
            raise ValueError(f"match_substrings for family {name!r} contains an empty pattern or group — it would match every model")
    if any(not sub for sub in _match_substrings(provider_substrings)):
        raise ValueError(f"provider_substrings for family {name!r} contains an empty pattern — it would match every provider")
    _FAMILIES[name] = {
        "match": tuple(match_substrings),
        "provider_match": tuple(provider_substrings),
        "native_path": tuple(native_path) if native_path is not None else None,
        "rungs": {str(k).lower().strip(): deepcopy(v) for k, v in rungs.items()},
    }


def merge_reasoning_config(reasoning: dict[str, Any], explicit: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Deep-merge reasoning kwargs under explicitly-passed config.

    Explicit values win on leaf conflicts — raw kwargs are the caller's escape
    hatch and must stay authoritative. Non-conflicting keys from both sides
    compose (a caller's partial ``extra_body`` me# Reasoning changes bundle

Generated from the reasoning changes in:

- `504558af` — `feat(models): add reasoning mappings for GLM DeepSeek and Qwen`
- `897f0079` — `feat(models): add Gemma 4 reasoning family`

This bundle contains the current contents of the seven non-test files changed
by those commits. Test files, creative phases, the bundle itself, and unrelated
working-tree changes are excluded.

`.env.example`
----
# ==============================================================================
# .env.example - Environment Configuration Template
# ==============================================================================
# Copy this file to .env and fill in your actual values
# SECURITY: Never commit .env files with real credentials to version control
# ==============================================================================

# ==============================================================================
# SECTION: Application Core
# ==============================================================================
# Environment type (dev, stg, prod)
ENV=dev

# Default LLM provider (used if not specified in runtime)
LLM_PROVIDER=custom
# custom will pull from env var BASE_URL/API_KEY
# custom_name will pulle from env var CUSTOM_NAME_BASE_URL / CUSTOM_NAME_API_KEY

# ==============================================================================
# SECTION: Artifacts & Storage
# ==============================================================================
# Directory for storing agent-generated artifacts
# ARTIFACT_DIR=artifacts

# SSL certificate directory for API calls
# SSL_CERT_DIR=

# ==============================================================================
# SECTION: Frontend (Streamlit Configuration)
# ==============================================================================

# --- Websocket & Session Settings ---
# Keep websocket alive with 30-second pings
STREAMLIT_SERVER_WEBSOCKET_PING_INTERVAL=30
# Keep session alive for 5 minutes after disconnect (in seconds)
STREAMLIT_SERVER_DISCONNECTED_SESSION_TTL=450

# --- UI Configuration ---
STREAMLIT_UI_HIDE_TOPBAR=true

# --- Branding (Optional) ---
# Path to logo image (default: data/assets/logo.png)
# LOGO_PATH=data/assets/logo.png
# Application name (default: Conversational AI Chat)
# APP_NAME=Conversational AI Chat
# Application icon emoji (default: 🤖)
# APP_ICON=🤖
# Enable debug mode for UI development
# UI_DEBUG_MODE=false

# --- Graph Configuration ---
# Path to graph configuration storage
STA_GRAPHS_CONFIG_PATH=

# --- UI Lock Configuration (Optional) ---
# Control which UI elements are displayed in the sidebar
# BYOK (Bring Your Own Key) mode will override these when credentials are missing

# Master lock mode: Set to "strict" to enable lock mode (default: disabled)
# UI_LOCK_MODE=strict
# Hide provider selection dropdown
# DISABLE_PROVIDER_SELECTION=true
# Hide provider configuration expander
# DISABLE_PROVIDER_CONFIG=true
# Hide API key input field
# DISABLE_API_KEY_VIEW=true
# Hide model selection
# DISABLE_MODEL_SELECTION=true
# Hide LLM configuration controls
# DISABLE_LLM_CONFIG=true

# --- Streamlit Theme (Optional - Managed via env/themes/ directory) ---
# Note: Use predefined themes in env/themes/ instead of setting these manually
# STREAMLIT_SERVER_ENABLE_STATIC_SERVING=
# STREAMLIT_THEME_PRIMARY_COLOR=
# STREAMLIT_THEME_BACKGROUND_COLOR=
# (See env/themes/.env.* files for theme examples)

# ==============================================================================
# SECTION: Backend - NetworkX Graph
# ==============================================================================
# Enable demo mode (uses NetworkX instead of TigerGraph)
DEMO_MODE=true

# Data directory containing graph extracts
NX_DATA_DIR=data/extracts/mock_data

# ==============================================================================
# SECTION: Backend - Databases (Optional)
# ==============================================================================

# --- PostgreSQL (Optional - for LangGraph checkpointing & persistence) ---
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DATABASE=langgraph
# POSTGRES_USER=postgres
# POSTGRES_PASSWORD=
# POSTGRES_POOL_MIN_SIZE=10
# POSTGRES_POOL_MAX_SIZE=20

# --- Elasticsearch (Optional - for search & analytics features) ---
# General Elasticsearch Cluster
# ELASTICSEARCH_ES_HOST=http://localhost:9200
# ELASTICSEARCH_ES_PORT=9200
# ELASTICSEARCH_ES_CA_CERTS=
# ELASTICSEARCH_ES_ID=
# ELASTICSEARCH_ES_API_KEY=
# ELASTICSEARCH_ES_CLIENT_KEY=
# ELASTICSEARCH_ES_CLIENT_CERT=
# ELASTICSEARCH_ES_DEFAULT_INDEX=
# ELASTICSEARCH_ES_TIMEOUT=30
# ELASTICSEARCH_ES_VERIFY_CERTS=true
# ELASTICSEARCH_ES_MAX_RETRIES=3
# ELASTICSEARCH_ES_RETRY_ON_TIMEOUT=true

# RAG document ingestion (infra/elasticsearch — builds the docs-hybrid index)
# Base URL for per-file source URLs written to metadata.pageUrl (the file's
# repo-relative path is appended). Defaults to the GitHub blob URL when unset,
# so pageUrl is always a real clickable citation; set empty to disable.
# RAG_ELASTICSEARCH_DOCS_BASE_URL=https://github.com/errajibadr/langgraph-agent-repo/blob/main

# File Integrity Elasticsearch (Separate instance for file monitoring)
# FILE_INTEGRITY_ELASTICSEARCH_ES_NAME=elasticsearch
# FILE_INTEGRITY_ELASTICSEARCH_ES_HOST=http://localhost:9200
# FILE_INTEGRITY_ELASTICSEARCH_ES_DEFAULT_INDEX=auditbeat-test
# FILE_INTEGRITY_ELASTICSEARCH_ES_CA_CERTS=
# FILE_INTEGRITY_ELASTICSEARCH_ES_ID=
# FILE_INTEGRITY_ELASTICSEARCH_ES_API_KEY=
# FILE_INTEGRITY_ELASTICSEARCH_ES_CLIENT_KEY=
# FILE_INTEGRITY_ELASTICSEARCH_ES_CLIENT_CERT=
# FILE_INTEGRITY_ELASTICSEARCH_ES_TIMEOUT=30
# FILE_INTEGRITY_ELASTICSEARCH_ES_VERIFY=false
# FILE_INTEGRITY_ELASTICSEARCH_ES_MAX_RETRIES=3
# FILE_INTEGRITY_ELASTICSEARCH_ES_RETRY_ON_TIMEOUT=true

# Retriever Elasticsearch - Used for RAG
# RETRIEVER_ELASTICSEARCH_ES_NAME=elasticsearch
# RETRIEVER_ELASTICSEARCH_ES_HOST=http://localhost:9200
# RETRIEVER_ELASTICSEARCH_ES_DEFAULT_INDEX=auditbeat-test
# RETRIEVER_ELASTICSEARCH_ES_CA_CERTS=
# RETRIEVER_ELASTICSEARCH_ES_ID=
# RETRIEVER_ELASTICSEARCH_ES_API_KEY=
# RETRIEVER_ELASTICSEARCH_ES_CLIENT_KEY=
# RETRIEVER_ELASTICSEARCH_ES_CLIENT_CERT=
# RETRIEVER_ELASTICSEARCH_ES_TIMEOUT=30
# RETRIEVER_ELASTICSEARCH_ES_VERIFY=false
# RETRIEVER_ELASTICSEARCH_ES_MAX_RETRIES=3
# RETRIEVER_ELASTICSEARCH_ES_RETRY_ON_TIMEOUT=true

# Elastic RAG Gateway Proxy - client-side BaseRetriever talking to the deployed
# `elastic_rag` LangGraph gateway via POST /runs/wait. See
# packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/elastic_rag_proxy/.
# GATEWAY_URL is the only required field; secrets live in .env.secrets.
# ELASTIC_RAG_PROXY_RETRIEVER_GATEWAY_URL=http://localhost:2024
# ELASTIC_RAG_PROXY_RETRIEVER_ASSISTANT_ID=elastic_rag
# ELASTIC_RAG_PROXY_RETRIEVER_TIMEOUT_S=30.0
# ELASTIC_RAG_PROXY_RETRIEVER_DEFAULT_TOP_K=10
# Opt in to stitch the gateway's run under the caller's LangSmith trace.
# Injects `langsmith-trace` + `langsmith-project` (sourced from baggage) into
# the /runs/wait body's config.configurable when an enclosing @traceable
# scope is active. Requires `langsmith` to be importable; otherwise a
# graceful no-op. Baggage carries the caller's run metadata verbatim — do
# not place secrets in run metadata when this flag is on.
# ELASTIC_RAG_PROXY_RETRIEVER_DISTRIBUTED_TRACING=false

# Retriever Query Expansion - wires a QueryExpander on the ElasticRetriever.
# PASS disables expansion (no LLM client built). Any non-PASS value forces the
# factory to build a QueryExpander backed by LLMAAS_* env (set further down).
# Per-call ``expansion_hint`` context keys only work when an expander is wired.
# RETRIEVER_EXPANSION_HINT=PASS                           # PASS | AUTO | KEYWORD | MULTI | PARAPHRASE | HYDE
# RETRIEVER_EXPANSION_DOMAIN_INTENT=                      # e.g. "Internal agent platform docs — LangGraph agents, RAG, retrievers."
# RETRIEVER_EXPANSION_MULTI_TIMEOUT_S=15.0                # wall-clock budget for MULTI's KEYWORD + PARAPHRASE + HYDE gather

# Twin Router scope - anonymized entity/apcode arrays for the twin
# project-knowledge retriever (Knowledge Agent RAG path). Real values are
# tenant-identifying and MUST stay out of version control — set them only in
# the gitignored `.env`. Each *_FILTERS / *_BOOSTS var accepts a JSON array
# (["a","b"]) or a comma-separated string (a,b). Leave unset → the twin
# retriever is unscoped. *_FILTERS narrow the result set; *_BOOSTS only
# soft-rank. INCLUDE_TRANSVERSAL widens the apcode filter to also admit
# transversal docs; INCLUDE_ENTITY_CHILDS admits descendant entities.
# TWIN_SCOPE_ENTITY_FILTERS=
# TWIN_SCOPE_APCODE_FILTERS=
# TWIN_SCOPE_APP_NAME_FILTERS=
# TWIN_SCOPE_ENTITY_BOOSTS=
# TWIN_SCOPE_APCODE_BOOSTS=
# TWIN_SCOPE_APP_NAME_BOOSTS=
# TWIN_SCOPE_INCLUDE_ENTITY_CHILDS=false
# TWIN_SCOPE_INCLUDE_TRANSVERSAL=false

# --- TigerGraph (Optional - Advanced graph backend) ---
# TG_HOST=
# TG_PORT=
# TG_GRAPH_NAME=
# TG_USERNAME=admin
# TG_PASSWORD=
# TG_SECRET=

# --- JIRA API -----

# JIRA_BASE_URL=
# JIRA_BEARER_TOKEN=
# JIRA_TIMEOUT=
# JIRA_VERIFY_SSL=

# ==============================================================================
# SECTION: Integrations
# ==============================================================================

# --- LangSmith (Tracing & Debugging) ---
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=
LANGSMITH_API_KEY=
# LANGSMITH_PROJECT=my-project

# LangSmith Test Configuration (for test environment)
LANGSMITH_TEST_TRACING=false
LANGSMITH_TEST_PROJECT=

# --- RAG API Endpoints ---
RAG_API_URL=
REDHAT_RAG_API_URL=
APACHE_RAG_API_URL=
ILLUMIO_RAG_API_URL=
BIGFIX_NETBACKUP_RAG_API_URL=

# ==============================================================================
# SECTION: LLM Providers
# ==============================================================================

# --- Default Provider ---
# Base URL for the default LLM provider
BASE_URL=https://llm.provider.com/
# API key (⚠️ NEVER commit real keys)
API_KEY=sk-your-key-here
# Model name
MODEL=model-name
# Optional capacity-tier model names
# BIG_MODEL=
# SMALL_MODEL=
# THINKING_MODEL=
# Optional vision model (capability axis, not a tier) — used by create_chat_model(..., multimodal=True)
# MULTIMODAL_MODEL=
# Optional model parameters
# TEMPERATURE=0.7
# TOP_P=1.0
# MAX_TOKENS=4096
# Optional reasoning control — 'off' | 'low' | 'medium' | 'high' | 'xhigh' | 'max', translated into
# whatever the resolved model family honors (native GLM/DeepSeek/Qwen3.8 effort,
# Mistral reasoning_effort, or Nemotron/earlier-Qwen chat_template_kwargs). Leave
# unset to keep the model's server-side default. Not every
# family defines every rung (Mistral has only off/high); an unsupported value warns and
# is forwarded raw for the API to judge. REASONING_FAMILY pins the translation table when
# a gateway alias hides the real model name, and does nothing without REASONING_EFFORT.
# See docs/consuming/reasoning.md.
# REASONING_EFFORT=
# REASONING_FAMILY=

# ------ EVAL PROVIDER -----
# EVAL_BASE_URL=
# EVAL_API_KEY=
# EVAL_MODEL has no default — it MUST be set to use the eval provider (LLM-as-judge);
# create_chat_model(provider="eval") raises if it is unset.
# EVAL_MODEL=
# EVAL_BIG_MODEL=
# EVAL_SMALL_MODEL=
# EVAL_THINKING_MODEL=


# --- Custom Named Provider (Optional) ---
# Use this pattern to configure additional LLM providers
# CUSTOM_NAME_BASE_URL=
# CUSTOM_NAME_API_KEY=
# CUSTOM_NAME_MODEL=
# CUSTOM_NAME_BIG_MODEL=
# CUSTOM_NAME_SMALL_MODEL=
# CUSTOM_NAME_THINKING_MODEL=
# CUSTOM_NAME_TEMPERATURE=
# CUSTOM_NAME_TOP_P=
# CUSTOM_NAME_MAX_TOKENS=
# CUSTOM_NAME_REASONING_EFFORT=
# CUSTOM_NAME_REASONING_FAMILY=
# Every key above works for ANY provider name — the prefix is derived as <NAME>_ with no
# code change. Giving a component its own provider name (e.g. the prompt-injection judge
# below) is how you configure that component's model independently of the main agent's.

# --- Orchestrator Prompt-Injection Guard (Optional) ---
# Runs before the orchestrator planner and screens the last five human
# messages for prompt injection, phishing, data exfiltration, policy bypass,
# and tool/subagent manipulation attempts. When the judge model is multimodal
# (e.g. mistral-small-2603), image parts on the latest human turn are screened too.
# Configure PROVIDER/MODEL explicitly to pick the judge. Leaving them unset relies
# on the built-in default judge, which emits a DeprecationWarning; a future release
# will require an explicit judge model and otherwise leave screening off.
ORCHESTRATOR_PROMPT_INJECTION_GUARD_ENABLED=true
ORCHESTRATOR_PROMPT_INJECTION_GUARD_FAIL_OPEN=true
# Soft-advisory tier: a turn carrying a genuine deliverable that ALSO asks for TWIN's internal
# rules ("explain X, then list the rules you followed") passes through with a <system_reminder>
# telling the planner to answer the task and decline the rules part — instead of a hard refusal.
# Set false to fall back to block-or-pass only.
ORCHESTRATOR_PROMPT_INJECTION_GUARD_ADVISORY_ENABLED=true
ORCHESTRATOR_PROMPT_INJECTION_GUARD_PROVIDER=mistral
ORCHESTRATOR_PROMPT_INJECTION_GUARD_MODEL=mistral-small-2603
# ORCHESTRATOR_PROMPT_INJECTION_GUARD_BASE_URL=
# Response token CEILING, not a target — the verdict is a tiny JSON object and the guard is
# meant to stay concise and fast. The cap exists only so a reasoning judge (gpt-oss) that emits
# a little reasoning before the JSON does not truncate mid-object (which, under fail-open, is a
# silent pass). 512 fits a low-effort burst + the JSON. Keep the judge terse via REASONING_EFFORT.
ORCHESTRATOR_PROMPT_INJECTION_GUARD_MAX_TOKENS=512
ORCHESTRATOR_PROMPT_INJECTION_GUARD_TEMPERATURE=0.0
ORCHESTRATOR_PROMPT_INJECTION_GUARD_MAX_RETRIES=2
# Reasoning-effort knob for the judge. Set 'low' for a reasoning judge (gpt-oss) so it returns
# the verdict fast without burning reasoning tokens. Leave UNSET for the Mistral judge above —
# Mistral defines no 'low' rung, so an injected 'low' would be forwarded raw and may be rejected
# (a rejected classify call fails open = a silent pass). Only set it when the judge is gpt-oss.
# ORCHESTRATOR_PROMPT_INJECTION_GUARD_REASONING_EFFORT=low
#
# For any model knob NOT listed above, give the judge its own provider name instead of
# waiting for a matching guard setting. Unknown provider names derive their env prefix
# automatically, so the judge gets a private, complete provider settings block:
#   ORCHESTRATOR_PROMPT_INJECTION_GUARD_PROVIDER=guard_judge
#   GUARD_JUDGE_BASE_URL=... / GUARD_JUDGE_API_KEY=... / GUARD_JUDGE_TOP_P=...
# The guard's own settings above stay authoritative where they overlap.
#
# Server-owned only: request/runtime context cannot override the guard model.

# --- Orchestrator Hub Skills (Optional) ---
# Snapshot lifetime, in seconds, for Hub-served skill groups mounted under
# /skills/hub/<group>/ (active only when a deployment passes hub_skill_repos
# to create_orchestrator_factory). After the TTL the next access re-pulls the
# repo tree — how a `sta skills push` reaches a running deployment without a
# restart. <= 0 refreshes on every access; the /skill-reload chat command
# bypasses the TTL entirely.
# ORCHESTRATOR_HUB_SKILLS_TTL_SECONDS=300

# --- Orchestrator Tool Budget Guard (Optional) ---
# Unset by default: no tool-call cap. Two independent, opt-in budgets:
# 1. Global per-run cap. When reached, the next planner turn is forced to answer
#    from context with no further tool calls (the answer-now soft-landing; it keeps
#    the tools/system prefix byte-identical for vLLM prefix-cache reuse).
# ORCHESTRATOR_TOOL_BUDGET_GUARD_MAX_TOOL_CALLS=
# 2. Per-tool per-run caps (JSON map of tool name -> max calls). An exhausted
#    per-tool cap blocks only that tool with a recoverable error; it does not force
#    answer-now. Example: cap the deepagents `task` delegation tool at 5 per run.
# ORCHESTRATOR_TOOL_BUDGET_GUARD_PER_TOOL_MAX_CALLS={"task": 5}
# Server-owned only: request/runtime context cannot install or raise either budget.

# --- Orchestrator Picture Reader (Optional) ---
# Exposed as read_picture only when the planner model is not listed in the
# multimodal model registry. It lets a multimodal model inspect image content
# that text-only planner models cannot receive directly.
ORCHESTRATOR_PICTURE_READER_ENABLED=true
# Describe an image-bearing turn with the picture-reader and rewrite that turn in
# place (original text + description) instead of only stripping the image. Set to
# false to restore strip + on-demand read_picture behavior.
ORCHESTRATOR_PICTURE_READER_EAGER_DESCRIBE=true
# Keep read_picture bound as an on-demand fallback for differently-targeted second looks.
ORCHESTRATOR_PICTURE_READER_KEEP_READ_PICTURE_TOOL=false
ORCHESTRATOR_PICTURE_READER_PROVIDER=mistral
ORCHESTRATOR_PICTURE_READER_MODEL=mistral-small-2603
# ORCHESTRATOR_PICTURE_READER_BASE_URL=
ORCHESTRATOR_PICTURE_READER_MAX_TOKENS=1024
ORCHESTRATOR_PICTURE_READER_TEMPERATURE=0.0
ORCHESTRATOR_PICTURE_READER_MAX_IMAGES=12
ORCHESTRATOR_PICTURE_READER_MAX_CONTEXT_MESSAGES=12

# --- Embedding Models (Optional) ---
EMBEDDING_BASE_URL=https://embedding.provider.com/
EMBEDDING_API_KEY=sk-test
EMBEDDING_MODEL=embedding-model
EMBEDDING_DIMENSIONS=1024
# Request timeout in seconds for embedding API calls (default 30)
# EMBEDDING_TIMEOUT=30

# --- Reranking ---
RERANKING_BASE_URL=https://reranking.provider.com/
RERANKING_API_KEY=sk-rerank-key
RERANKING_MODEL=reranking-model

# --- Knowledge Agent per-task model overrides (Optional) ---
# Opt-in: consumers call KnowledgeAgentConfig.from_env() to pick these up.
# Folds into task_model_defaults (layer 3 of the resolution ladder) — runtime
# context.model_configs still wins. Provider/model are intentionally NOT
# hardcoded in the KA package; set these (or LLM_PROVIDER + <PROVIDER>_*
# at the engine-wide layer) to choose a backend per task.
#
# Tasks: default | planning | compression | review | synthesis | verification
# Keys per task: PROVIDER, TIER, BASE_URL, MODEL, MAX_TOKENS, TEMPERATURE
#   (KA_<TASK>_API_KEY lives in .env.secrets.example)
#
# KA_PLANNING_PROVIDER=
# KA_PLANNING_TIER=
# KA_PLANNING_BASE_URL=
# KA_PLANNING_MODEL=
# KA_PLANNING_MAX_TOKENS=2048
# KA_PLANNING_TEMPERATURE=0.0
#
# KA_COMPRESSION_PROVIDER=
# KA_COMPRESSION_TIER=
# KA_COMPRESSION_BASE_URL=
# KA_COMPRESSION_MODEL=
# KA_COMPRESSION_MAX_TOKENS=8192
# KA_COMPRESSION_TEMPERATURE=0.0
#
# KA_REVIEW_PROVIDER=
# KA_REVIEW_TIER=
# KA_REVIEW_BASE_URL=
# KA_REVIEW_MODEL=
# KA_REVIEW_MAX_TOKENS=4096
# KA_REVIEW_TEMPERATURE=0.0
#
# KA_SYNTHESIS_PROVIDER=
# KA_SYNTHESIS_TIER=
# KA_SYNTHESIS_BASE_URL=
# KA_SYNTHESIS_MODEL=
# KA_SYNTHESIS_MAX_TOKENS=8192
# KA_SYNTHESIS_TEMPERATURE=0.0
#
# KA_VERIFICATION_PROVIDER=
# KA_VERIFICATION_TIER=
# KA_VERIFICATION_BASE_URL=
# KA_VERIFICATION_MODEL=
# KA_VERIFICATION_MAX_TOKENS=4096
# KA_VERIFICATION_TEMPERATURE=0.0

# ==============================================================================
# SECTION: Test Configuration
# ==============================================================================
# Models used for integration tests and evaluations

# Large Model (High-capacity for complex tasks)
TEST_MODEL_LARGE=openai/gpt-oss-120b
TEST_MODEL_LARGE_PROVIDER=openai

# Medium Model (Balanced performance - good for tool usage)
TEST_MODEL_MEDIUM=mistral-small-2603
TEST_MODEL_MEDIUM_PROVIDER=mistral

# Small Model (Lightweight for simple tasks)
TEST_MODEL_SMALL=mistral-small-2603
TEST_MODEL_SMALL_PROVIDER=mistral

# Llama Model (Alternative architecture)
TEST_MODEL_LLAMA=Meta-llama33-70b-instruct
TEST_MODEL_LLAMA_PROVIDER=llmaas_dev

# ==============================================================================
# SECTION: Logging & Monitoring
# ==============================================================================
# Directory for log files
# LOG_DIR=logs

# Enable file-based logging (in addition to console)
# ENABLE_FILE_LOGGING=false

# Emit one ssl-audit log line per create_chat_model() call (diagnostic only) - set to 1 to enable
# STA_SSL_AUDIT=1

# ==============================================================================
# SECTION: Documentation (MkDocs)
# ==============================================================================
# DOCS_SITE_URL=
# DOCS_REPO_NAME=
# DOCS_REPO_URL=
# DOCS_EDIT_URI=

# Serve the built docs site from the LangGraph server at /documentation
# (requires a built site: `make docs/build`)
# DEPLOY_DOCS=true
# Override the built-site location (defaults to <repo>/output/site)
# DOCS_SITE_DIR=


# HABILITATION (ONLY TWIN for now is compatible and implements this logic)
# HABILITATION_BYPASS=1
# HABILITATION_BYPASS_ROLE=prod
# ==============================================================================
# SECTION: Habilitation Configuration
# ==============================================================================
# HABILITATION_API_BASE_URL=
# HABILITATION_API_KEY=
# HABILITATION_UIDS=
# HABILITATION_ROLE=
# HABILITATION_ROLE_TYPE=


# ==============================================================================
# End of Configuration
# ==============================================================================

-------

docs/consuming/chat-models.md
----
# Using `create_chat_model`

`create_chat_model` builds a configured chat model (a LangChain
`ChatOpenAI` or `ChatMistralAI`) from environment variables, with optional
per-call overrides. It is the single entry point for talking to any LLM
provider in this library.

```python
from sta_agent_engine.models import create_chat_model

model = create_chat_model("llmaas")
response = await model.ainvoke("What is Python in one sentence?")
```

## Selecting a provider

The first argument picks the provider. Three equivalent forms:

```python
create_chat_model("llmaas")               # built-in, by string
create_chat_model(ProviderType.MISTRAL)   # built-in, by enum (from sta_agent_core.config)
create_chat_model("openai")                  # any other name — convention provider
create_chat_model()                        # no arg → reads LLM_PROVIDER (default: custom)
```

Built-in names: `llmaas`, `llmaas_dev`, `mistral`, `custom`, `eval`, `openai`.
**Any other string is also accepted** — its env prefix is derived as
`f"{NAME.upper()}_"`, so `create_chat_model("openai")` reads `openai_*` with no
code change.

## Environment variable contract

For a provider `NAME`, set the prefixed vars (e.g. `LLMAAS_API_KEY`):

| Variable | Purpose |
|---|---|
| `{NAME}_API_KEY` | Auth key |
| `{NAME}_BASE_URL` | OpenAI-compatible endpoint |
| `{NAME}_MODEL` | Default model |
| `{NAME}_BIG_MODEL` / `{NAME}_SMALL_MODEL` / `{NAME}_THINKING_MODEL` | Capacity-tier models |
| `{NAME}_MULTIMODAL_MODEL` | Vision model |
| `{NAME}_TEMPERATURE` / `{NAME}_TOP_P` / `{NAME}_MAX_TOKENS` | Generation params |

The built-in `custom` provider uses **no prefix** (bare `API_KEY`, `BASE_URL`,
`MODEL`, …). `openai` reads the canonical `OPENAI_*` vars the official OpenAI
SDK also honors, so `create_chat_model("openai")` works with just
`OPENAI_API_KEY` set.

A ready-to-fill starter file lives at `.env.provider.example`.

## Capacity tiers — `tier=`

`tier` picks a model slot by **capacity/quality** when no explicit `model=` is
passed. All tiers answer the same request; they trade quality for latency/cost.

```python
create_chat_model("llmaas")                    # tier="default" → LLMAAS_MODEL
create_chat_model("llmaas", tier="big")        # LLMAAS_BIG_MODEL,    else MODEL
create_chat_model("llmaas", tier="small")      # LLMAAS_SMALL_MODEL,  else MODEL
create_chat_model("llmaas", tier="thinking")   # LLMAAS_THINKING_MODEL, else BIG_MODEL, else MODEL
```

Cascade: `thinking → big → model`, `big → model`, `small → model`. An unknown
tier raises `ValueError`.

## Vision models — `multimodal=True`

Multimodality is a **capability**, not a tier (you can't ask for the "big
multimodal" model). Pass `multimodal=True`:

```python
create_chat_model("llmaas", multimodal=True)
```

Resolution order:

1. `{NAME}_MULTIMODAL_MODEL` if set — used **verbatim** (authoritative; lets you
   name a vision model the built-in capability list doesn't yet recognize).
2. else `{NAME}_MODEL` **only if** it's a recognized multimodal model.
3. else **raises `ValueError`** — it never silently returns a text-only model
   that would drop images (under the guard middleware) or 400 upstream.

`multimodal=True` takes precedence over `tier=` and is ignored when an explicit
`model=` is passed. Recognized vision models are the allow-list in
`sta_agent_engine.models.capabilities` (`is_multimodal`); setting
`{NAME}_MULTIMODAL_MODEL` bypasses that check.

## Reasoning control — `reasoning_effort=`

Reasoning/thinking is controlled with a normalized `reasoning_effort`
parameter, translated per model family into the kwargs each model actually
honors (native GLM/DeepSeek/Qwen3.8 effort, Mistral's `reasoning_effort`, or
Nemotron/earlier-Qwen `chat_template_kwargs`). See the dedicated guide:
[reasoning.md](reasoning.md).

```python
create_chat_model("mistral", reasoning_effort="high")
create_chat_model("llmaas", model="nemotron-3-super-120b", reasoning_effort="off")
```

## Per-call overrides

Any keyword overrides the resolved env value:

```python
create_chat_model("llmaas", model="some-other-model", temperature=0.7, max_tokens=512)

# Bring-your-own-key (BYOK): inject credentials at call time
create_chat_model("llmaas", provider_api_key="sk-...", provider_base_url="https://...")
```

## Client dispatch (OpenAI vs Mistral)

`create_chat_model` routes to `ChatMistralAI` when the **provider name** contains
`mistral` (e.g. `mistral`, `mistral_eu`) **or the model name** contains
`mistral` / `devstral` / `magistral`; otherwise it routes to `ChatOpenAI` for
OpenAI-compatible APIs.

> **Sharp edge — the model name wins.** A Mistral-branded model on an
> OpenAI-compatible gateway (e.g. `provider="llmaas"` with
> `LLMAAS_MODEL=mistral-small-2506`) routes to the **native Mistral SDK**
> (`api.mistral.ai`), not your gateway. If your gateway serves Mistral-family
> models over an OpenAI-compatible API, pass an explicit `base_url` (and
> `api_key`) so the Mistral client targets your gateway, or use a non-Mistral
> model name.

If the OpenAI dispatch is taken but `api_key`/`base_url` did not resolve,
`ChatOpenAI` silently falls back to `OPENAI_API_KEY` + `api.openai.com`. This
fallback now emits a `DeprecationWarning` and will raise in a future release —
always set `{NAME}_API_KEY` and `{NAME}_BASE_URL` (or pass them as kwargs).

## Full example

A runnable end-to-end example is in
`examples/sta_agent_engine/chat_model_example.py`.

-------

docs/consuming/reasoning.md
----
# Reasoning control with `create_chat_model`

Reasoning ("thinking") models expose incompatible knobs: Mistral takes a
top-level `reasoning_effort` string, Nemotron and Gemma 4 use booleans nested
under `extra_body.chat_template_kwargs`, and GLM/DeepSeek combine a native
effort with a separate thinking toggle. The
`reasoning_effort` parameter gives you one vocabulary; the library translates
it into whatever the resolved model actually honors.

```python
from sta_agent_engine.models import create_chat_model

model = create_chat_model("llmaas", model="nemotron-3-super-120b", reasoning_effort="low")
# → ChatOpenAI(..., extra_body={"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}})

model = create_chat_model("mistral", reasoning_effort="high")
# → ChatMistralAI(..., model_kwargs={"reasoning_effort": "high"})
```

Omit the parameter (or pass `None`, or an empty/blank string) and **nothing is
injected** — the model keeps its server-side default. Existing code is
unaffected.

## Configuring it by environment variable

`reasoning_effort` and `reasoning_family` are ordinary provider settings, so you
don't have to thread them through your call sites. Set them under the provider's
env prefix and every `create_chat_model` call for that provider picks them up:

```bash
LLMAAS_REASONING_EFFORT=low
LLMAAS_REASONING_FAMILY=nemotron-super   # only needed for gateway aliases (below)
```

```python
create_chat_model("llmaas", model="nemotron-3-super-120b")
# → ChatOpenAI(..., extra_body={"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}})
```

An explicit kwarg still wins over the environment, so a call site that needs a
different effort can override without touching config.

Unset means unset: with no env var and no kwarg, nothing is injected. A blank
value (`LLMAAS_REASONING_EFFORT=`) is also treated as unset rather than as an
empty effort string.

This works for **any** provider name, including ones you never registered — the
prefix is derived as `<NAME>_`. That makes a dedicated provider name the way to
give one component its own reasoning settings without affecting anything else:

```bash
# a screening/classifier model that should answer fast, while the main agent thinks hard
MY_CLASSIFIER_BASE_URL=https://gateway.example/v1
MY_CLASSIFIER_API_KEY=...
MY_CLASSIFIER_MODEL=openai/gpt-oss-120b
MY_CLASSIFIER_REASONING_EFFORT=low
```

```python
create_chat_model("my_classifier")
```

Pick an effort per agent/task/thread rather than per turn — see the caching note
at the end of this page.

## Effort vocabulary and what goes on the wire

Efforts are plain strings. Built-in families support:

| effort | GLM-5.2 | DeepSeek V4 | Qwen3.8 | Earlier Qwen3.x | Gemma 4 |
|---|---|---|---|---|---|
| `"off"` | `thinking.type="disabled"` | `thinking.type="disabled"` | `enable_thinking=False` | `enable_thinking=False` | `enable_thinking=False` (default) |
| `"low"` | native `low` (evaluated as `high`) | native `high` | native `low` | — | — |
| `"medium"` | native `medium` (evaluated as `high`) | native `high` | native `medium` | — | — |
| `"high"` | native `high` | native `high` | native `xhigh` | `enable_thinking` (default) | `enable_thinking=True` |
| `"xhigh"` | native `xhigh` (evaluated as `max`) | native `max` | native `xhigh` | — | — |
| `"max"` | native `max` | native `max` | — | — | — |

GLM-5.2 also accepts the native strings `none` and `minimal`; both stop
thinking. Qwen3.8's official native levels are `low`, `medium`, and `xhigh`, so
the portable `high` rung maps to its strongest `xhigh` level. The open-weight
`Qwen3.8-2.4T-A95B` variant requires thinking and therefore intentionally has
no `off` rung; switchable variants such as `Qwen3.8-27B` do.

The built-in Qwen `off` translation targets self-hosted vLLM/SGLang and sends
`extra_body.chat_template_kwargs.enable_thinking=False`. Qwen Cloud uses a
different direct `extra_body.enable_thinking=False` shape; for that endpoint,
omit the normalized `reasoning_effort="off"` and pass its native `extra_body`
field explicitly.

The other built-in families retain these mappings:

| effort | Mistral (small / medium-3-5) | Nemotron-Super | Nemotron-Ultra |
|---|---|---|---|
| `"off"` | `reasoning_effort="none"` | `enable_thinking=False` | `enable_thinking=False` |
| `"low"` | — | `enable_thinking, low_effort` | `enable_thinking, medium_effort, force_nonempty_content=False` |
| `"medium"` | — | — | `enable_thinking, medium_effort, force_nonempty_content=True` |
| `"high"` | `reasoning_effort="high"` | `enable_thinking` (full) | `enable_thinking, force_nonempty_content=True` (full) |

Qwen3.x before 3.8 is deliberately binary: the model card (e.g. `Qwen/Qwen3.6-27B`)
documents only the `enable_thinking` switch. Graded thinking **budgets** are a
serving-stack feature (vLLM's `thinking_token_budget`, name and availability
vary by version) — if your gateway supports them, re-register the family with
budget rungs (example below).

Gemma 4 is also deliberately binary. Its official template defaults
`enable_thinking` to `False`, and exposes no documented graded effort or token
budget. `preserve_thinking` is a separate template option for retaining prior
assistant reasoning; pass it explicitly under `extra_body.chat_template_kwargs`
when needed.

`—` = not supported by that family. The table makes documented compatibility
mappings explicit (for example, DeepSeek `low` becomes native `high`). For an
unsupported value, the library warns and never raises. Families with a native
`reasoning_effort` path forward the raw value for the API to validate;
template-only families such as Nemotron, earlier Qwen3.x, and Gemma 4 inject
nothing.

Nemotron-Ultra notes: `medium`/`high` include
`force_nonempty_content=True` (required by SGLang when tool-calling with
thinking enabled); `low` is medium-effort thinking with the flag explicitly
sent as `False` (so a server-side template default can't force it back on).

The Mistral column applies to every model the library dispatches to the
Mistral client — names containing `mistral`, `devstral`, or `magistral` all
use it, so `"off"` always reaches the API as the sanctioned `"none"`.

Check what a model supports programmatically:

```python
from sta_agent_engine.models import supported_reasoning_efforts

supported_reasoning_efforts("nemotron-3-ultra-550b")   # frozenset({'off', 'low', 'medium', 'high'})
```

## Unknown models

If no family matches the model name, the value is forwarded as a top-level
`reasoning_effort` (the OpenAI-native form) with a `UserWarning`. Genuine
OpenAI reasoning models honor it; vLLM/LiteLLM-fronted gateways typically
ignore it. `gpt-*` model names take this passthrough silently — it is their
native parameter.

If your gateway serves a known family under an alias (`chat-default` actually
being Nemotron), pin the family explicitly:

```python
create_chat_model("llmaas", model="chat-default",
                  reasoning_effort="high", reasoning_family="nemotron-super")
```

`reasoning_family` only selects the translation table — passing it without
`reasoning_effort` injects nothing (the library warns and ignores it).

## Overrides always win

Explicitly-passed native kwargs are the escape hatch, and they beat the
translation on conflicting keys (with a warning). Non-conflicting keys merge:

```python
create_chat_model(
    "llmaas", model="nemotron-3-super-120b",
    reasoning_effort="high",                                  # → enable_thinking=True
    extra_body={"chat_template_kwargs": {"custom_flag": 1}},  # merged alongside
)
# wire: chat_template_kwargs == {"enable_thinking": True, "custom_flag": 1}
```

## Per-call control

`build_reasoning_kwargs()` returns the translated kwargs so you can apply
effort per call instead of per model instance:

```python
from sta_agent_engine.models import build_reasoning_kwargs

kw = build_reasoning_kwargs("nemotron-3-super-120b", "low")
model.invoke(messages, **kw)        # or model.bind(**kw)
```

This works for families that translate into `extra_body` (Nemotron, Qwen) or
a native field (OpenAI). **Exception — Mistral models are constructor-only:**
their translation rides `model_kwargs`, which `ChatMistralAI` flattens into
the request payload only when set as a constructor field; splatted at call
time it would be posted as a literal `"model_kwargs"` JSON key instead. For
Mistral, set the effort where the model is created
(`create_chat_model(..., reasoning_effort=...)`).

Prefer choosing an effort **per agent / task / thread, not per turn**: for
template-flag families, flipping `chat_template_kwargs` between turns of one
conversation re-renders the server-side prompt template and defeats
vLLM/SGLang prefix caching (latency and cost, not correctness).

## Registering your own model family

One call at application startup, no subclassing:

```python
from sta_agent_engine.models import register_reasoning_family

register_reasoning_family(
    "my-model",
    rungs={
        "off":  {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    },
    match_substrings=("my-model",),   # all must appear in the model name
)
```

Rung values are the **literal kwargs** to merge into the model constructor —
anything the underlying client accepts (`extra_body`, `model_kwargs`, native
fields) is fair game. Re-registering an existing family (including a built-in)
replaces it, so you can also override the library's defaults. Example —
extending the built-in `qwen3` family with graded thinking budgets once you've
verified your vLLM gateway supports them:

```python
register_reasoning_family(
    "qwen3",
    rungs={
        "off":    {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        "medium": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 2048}},
        "high":   {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}},
        "xhigh":  {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 8192}},
    },
    match_substrings=("qwen3",),
)
```

Verify the budget parameter name against your serving stack first — vLLM has
shipped it as `thinking_token_budget` and rejected/ignored other spellings
depending on version; a one-off probe call per rung is cheap insurance.

### Slug variants across providers

Family matching is **case- and separator-insensitive**: `-`, `_`, `.`, `:`,
`/` and spaces are stripped from both the model name and the patterns before
comparing. The same model arriving under different provider slug conventions —
`qwen3.6`, `qwen3-6`, `Qwen/Qwen3.6-32B-Instruct`, `qwen3:32b` — resolves to
the same family and produces identical wire kwargs. You only need
`reasoning_family=` when the alias shares nothing with the model's real name
(`chat-default`).

`match_substrings` supports two shapes:

- **Flat tuple = one AND-group** — all substrings must appear:
  `("nemotron-3", "ultra")`.
- **Tuple of tuples = OR of AND-groups** — the family matches when *any*
  group fully matches; use this for genuinely different naming dialects:
  `(("nemotron-3", "ultra"), ("nemo-ultra",))`.

## Known limitations

- **Mistral + tool-calling agents:** the upstream LangChain Mistral
  integration sends an assistant message's `tool_calls` *instead of* its
  content, so reasoning traces are not replayed to the API on tool-call turns.
  Plain chat turns replay them automatically. Impact is limited to extra
  re-thinking inside agent loops, not wrong answers. (Qwen3.6 addresses the
  same concern server-side with `chat_template_kwargs.preserve_thinking=True` —
  orthogonal to effort; pass it via explicit `extra_body`, it merges alongside
  the translated flags.)
- **Whether an effort actually changes behavior is ultimately decided by the
  serving stack.** Some gateways enable thinking by default and ignore
  parameters they don't recognize. When in doubt, verify with a one-off call
  per effort value and inspect the response's reasoning content.

-------

examples/sta_agent_engine/models/qwen3_8_reasoning_effort_example.py
----
"""Example: measure Qwen3.8-27B reasoning tokens across effort levels.

NOT consumer documentation — this is an engineering smoke-test of the live
``create_chat_model`` request surface. Consumer guidance lives in
``docs/consuming/reasoning.md``. Edit the USER_* constants and run:

    uv run python examples/sta_agent_engine/models/qwen3_8_reasoning_effort_example.py

REAL PAID API CALLS: six requests per repeat (default, off, low, medium, high,
and xhigh). The default repeat count is one. The configured provider must have
its ``{NAME}_API_KEY``, ``{NAME}_BASE_URL``, and model access set up.

Qwen3.8-27B natively supports ``low``, ``medium``, and ``xhigh``; its default
is ``xhigh``. This library exposes a portable ``high`` level and maps it to
Qwen's native ``xhigh``. Consequently, the high and xhigh rows intentionally
send identical reasoning configuration. ``off`` uses the self-hosted
vLLM/SGLang ``chat_template_kwargs.enable_thinking=False`` dialect.

The reported reasoning-token count prefers the gateway's authoritative usage
metadata. If the gateway omits it, the script estimates the count from the
returned reasoning trace at approximately four characters per token. A single
sample is noisy; set USER_REPEATS to 3 or more before drawing conclusions.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from sta_agent_engine.models import build_reasoning_kwargs, create_chat_model


# --- Edit these -------------------------------------------------------------
USER_PROVIDER = "llmaas"  # reads LLMAAS_* env vars; any provider name works
USER_MODEL = "Qwen/Qwen3.8-27B"  # use the exact slug exposed by your gateway
USER_PROMPT = "Find every triple of positive integers x <= y <= z satisfying 1/x + 1/y + 1/z = 1. Prove that your list is complete."
USER_MAX_TOKENS = 8192
USER_REPEATS = 1  # use >=3 for a less noisy comparison; each repeat costs six calls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffortCase:
    """One normalized effort and the Qwen-native behavior it should select."""

    label: str
    effort: str
    expected_native: str


@dataclass(frozen=True)
class Measurement:
    """Metrics captured from one live request."""

    case: EffortCase
    repeat: int
    reasoning_tokens: int
    token_source: str
    reasoning_chars: int
    output_tokens: int | None
    seconds: float
    answer_preview: str


# Empty effort deliberately suppresses a provider-level *_REASONING_EFFORT env
# value and injects nothing, allowing Qwen's actual server-side default to win.
_CASES = (
    EffortCase("default", "", "xhigh (Qwen default)"),
    EffortCase("off", "off", "thinking disabled"),
    EffortCase("low", "low", "low"),
    EffortCase("medium", "medium", "medium"),
    EffortCase("high", "high", "xhigh (portable alias)"),
    EffortCase("xhigh", "xhigh", "xhigh"),
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mapping values as-is and replace every other shape with empty."""
    return value if isinstance(value, Mapping) else {}


def _reasoning_text(response: AIMessage) -> str:
    """Extract the visible reasoning trace without double-counting aliases."""
    for key in ("reasoning_content", "reasoning"):
        value = response.additional_kwargs.get(key)
        if value:
            return str(value)

    if isinstance(response.content, list):
        parts: list[str] = []
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                parts.append(str(block.get("reasoning") or block.get("content") or ""))
        return "".join(parts)
    if isinstance(response.content, str) and "<think>" in response.content:
        _, reasoning_and_answer = response.content.split("<think>", 1)
        reasoning, separator, _ = reasoning_and_answer.partition("</think>")
        return reasoning if separator else reasoning_and_answer
    return ""


def _api_reasoning_tokens(response: AIMessage) -> int | None:
    """Read normalized LangChain usage first, then common raw API shapes."""
    usage = _as_mapping(response.usage_metadata)
    output_details = _as_mapping(usage.get("output_token_details"))
    for key in ("reasoning", "reasoning_tokens"):
        value = output_details.get(key)
        if isinstance(value, int):
            return value

    response_metadata = _as_mapping(response.response_metadata)
    raw_usage = _as_mapping(response_metadata.get("token_usage") or response_metadata.get("usage"))
    completion_details = _as_mapping(raw_usage.get("completion_tokens_details"))
    value = completion_details.get("reasoning_tokens")
    return value if isinstance(value, int) else None


def _output_tokens(response: AIMessage) -> int | None:
    """Return the provider-counted total output tokens when present."""
    usage = _as_mapping(response.usage_metadata)
    value = usage.get("output_tokens")
    if isinstance(value, int):
        return value

    response_metadata = _as_mapping(response.response_metadata)
    raw_usage = _as_mapping(response_metadata.get("token_usage") or response_metadata.get("usage"))
    value = raw_usage.get("completion_tokens")
    return value if isinstance(value, int) else None


def _count_reasoning(response: AIMessage, reasoning_text: str) -> tuple[int, str]:
    """Return the best available reasoning count and its provenance."""
    api_count = _api_reasoning_tokens(response)
    if api_count is not None:
        return api_count, "api"
    return round(len(reasoning_text) / 4), "~4 chars/token"


async def _measure(case: EffortCase, repeat: int) -> Measurement:
    """Create a model at one effort level, invoke it, and measure reasoning."""
    model = create_chat_model(
        USER_PROVIDER,
        model=USER_MODEL,
        reasoning_effort=case.effort,
        max_tokens=USER_MAX_TOKENS,
        temperature=1.0,
        top_p=0.95,
        extra_body={"top_k": 20},
    )
    started = time.perf_counter()
    response = await model.ainvoke(USER_PROMPT)
    seconds = time.perf_counter() - started
    if not isinstance(response, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(response).__name__}")

    reasoning_text = _reasoning_text(response)
    reasoning_tokens, token_source = _count_reasoning(response, reasoning_text)
    answer_preview = response.text.strip().replace("\n", " ")[:72]
    return Measurement(
        case=case,
        repeat=repeat,
        reasoning_tokens=reasoning_tokens,
        token_source=token_source,
        reasoning_chars=len(reasoning_text),
        output_tokens=_output_tokens(response),
        seconds=seconds,
        answer_preview=answer_preview,
    )


def _print_translation() -> None:
    """Show exactly what create_chat_model will merge for every case."""
    print(f"Model: {USER_MODEL}")
    print("Qwen native levels: low, medium, xhigh (default: xhigh)\n")
    print("Normalized create_chat_model translation:")
    for case in _CASES:
        translated = build_reasoning_kwargs(USER_MODEL, case.effort)
        print(f"  {case.label:<7} -> {case.expected_native:<25} {translated}")


def _print_results(measurements: list[Measurement]) -> None:
    """Print per-call results and averages per effort."""
    print("\nPer-call measurements:")
    print(f"{'effort':<8} {'run':>3} {'rsn tokens':>10} {'source':<16} {'rsn chars':>9} {'out tok':>8} {'seconds':>8}  answer")
    for item in measurements:
        output_tokens = item.output_tokens if item.output_tokens is not None else "n/a"
        print(
            f"{item.case.label:<8} {item.repeat:>3} {item.reasoning_tokens:>10} {item.token_source:<16} "
            f"{item.reasoning_chars:>9} {output_tokens:>8} {item.seconds:>8.1f}  {item.answer_preview}"
        )

    print("\nAverages:")
    print(f"{'effort':<8} {'native behavior':<25} {'avg rsn tok':>11} {'avg seconds':>11}")
    for case in _CASES:
        group = [item for item in measurements if item.case == case]
        avg_tokens = statistics.fmean(item.reasoning_tokens for item in group)
        avg_seconds = statistics.fmean(item.seconds for item in group)
        print(f"{case.label:<8} {case.expected_native:<25} {avg_tokens:>11.1f} {avg_seconds:>11.1f}")

    enabled = [item for item in measurements if item.case.label not in {"off"}]
    observable_reasoning = any(item.token_source == "api" or item.reasoning_chars > 0 for item in enabled)
    off = [item.reasoning_tokens for item in measurements if item.case.label == "off"]
    if not observable_reasoning:
        print("\nWARNING: no enabled run exposed reasoning usage or text; this gateway cannot be evaluated with this response shape.")
    elif off and any(count > 0 for count in off):
        print("\nWARNING: off returned reasoning tokens; the gateway may be ignoring chat_template_kwargs.enable_thinking=False.")
    elif off:
        print("\nPASS: off returned zero visible/provider-counted reasoning tokens.")
    print("High and xhigh are expected to be statistically similar: both send Qwen's native xhigh. Individual runs need not match exactly.")


async def main() -> None:
    """Run every effort sequentially to avoid mixing concurrent load effects."""
    if USER_REPEATS < 1:
        raise ValueError("USER_REPEATS must be at least 1")

    _print_translation()
    measurements: list[Measurement] = []
    for repeat in range(1, USER_REPEATS + 1):
        for case in _CASES:
            print(f"Running {case.label}, repeat {repeat}/{USER_REPEATS}...")
            measurements.append(await _measure(case, repeat))
    _print_results(measurements)


if __name__ == "__main__":
    asyncio.run(main())

-------

examples/sta_agent_engine/models/reasoning_effort_example.py
----
"""Example: reasoning_effort with create_chat_model — reasoning-token cost per effort.

NOT consumer documentation — this is an engineering smoke-test of the
``reasoning_effort`` surface against live Nemotron endpoints. Consumer-facing
usage lives in docs/consuming/reasoning.md. Edit the USER_* constants and run:

    uv run python examples/sta_agent_engine/models/reasoning_effort_example.py

REAL PAID API CALLS: one request per (model, prompt, effort) — 14 with the
defaults below. Requires the provider env vars ({NAME}_API_KEY +
{NAME}_BASE_URL for the provider name set below).

For each model and prompt, every supported effort is invoked once and the
reasoning cost is reported two ways: the API-counted reasoning tokens (when
the gateway returns ``completion_tokens_details.reasoning_tokens`` — the
authoritative, billed count) and an estimate from the reasoning text surfaced
in the response (~4 chars/token — a fallback for gateways that report no
token details; it undercounts terse math/symbol content).

Two prompts on purpose: an easy classic caps reasoning naturally at ~100
tokens whatever the effort (ceiling effect), so effort asymmetry only becomes
observable on the multi-step constraint puzzle. Expectation there: ``off``
≈ 0 reasoning tokens and reasoning grows with the rung. If ``off`` still
shows a sizable count, the gateway is not applying ``enable_thinking=False``
(check whether it forwards ``extra_body.chat_template_kwargs``).
"""

import asyncio
import time

from langchain_core.messages import AIMessage

from sta_agent_engine.models import create_chat_model, supported_reasoning_efforts


# --- Edit these -------------------------------------------------------------
USER_PROVIDER = "custom"  # any built-in or arbitrary name; reads {NAME}_* env vars
USER_MODELS = ("nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-ultra-550b-a55b")
USER_PROMPTS = (
    (
        "easy",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    ),
    (
        "complex",
        "Four people (Ava, Ben, Cy, Dee) each ordered a different drink (coffee, tea, juice, water) "
        "and sat in seats 1-4, left to right. Clues: (1) Ava sat immediately left of the tea drinker. "
        "(2) Ben did not order coffee and did not sit in seat 4. (3) The juice drinker sat in seat 1. "
        "(4) Cy sat exactly two seats right of Dee. (5) The water drinker sat next to Ben. "
        "Work out who sat where and who ordered what.",
    ),
)
USER_MAX_TOKENS = 4096  # cap per response so a chatty high-effort run stays bounded
# ---------------------------------------------------------------------------

_EFFORT_ORDER = ("off", "low", "medium", "high", "xhigh", "max")  # display order for whatever rungs the family defines


def _reasoning_text(response: AIMessage) -> str:
    """Collect whatever reasoning the gateway surfaces client-side.

    Gateways differ: some put it in ``additional_kwargs["reasoning_content"]``
    (the create_chat_model converters normalize to this), others emit
    ``type="reasoning"`` content blocks.
    """
    parts = [str(response.additional_kwargs.get("reasoning_content") or "")]
    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                parts.append(str(block.get("reasoning") or block.get("content") or ""))
    return "".join(parts)


async def _measure(model_name: str, effort: str, prompt: str) -> tuple[int | str, int, int | str, float, str]:
    """Invoke once at the given effort; return (api reasoning tokens, estimated
    reasoning tokens, output tokens, seconds, answer preview)."""
    model = create_chat_model(USER_PROVIDER, model=model_name, reasoning_effort=effort, max_tokens=USER_MAX_TOKENS)
    started = time.perf_counter()
    response = await model.ainvoke(prompt)
    elapsed = time.perf_counter() - started
    assert isinstance(response, AIMessage)

    usage = response.usage_metadata or {}
    api_reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    estimated_reasoning = round(len(_reasoning_text(response)) / 4)  # ~4 chars/token heuristic
    output_tokens = usage.get("output_tokens")
    answer = response.text.strip().replace("\n", " ")[:70]
    return (
        api_reasoning if api_reasoning is not None else "n/a",
        estimated_reasoning,
        output_tokens if output_tokens is not None else "n/a",
        elapsed,
        answer,
    )


async def main() -> None:
    for model_name in USER_MODELS:
        supported = supported_reasoning_efforts(model_name)
        efforts = [effort for effort in _EFFORT_ORDER if effort in supported]
        print(f"\n=== {model_name} — efforts: {', '.join(efforts)} ===")
        for label, prompt in USER_PROMPTS:
            print(f"\n[{label}] {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
            print(f"{'effort':<8} {'api rsn tok':>12} {'est rsn tok':>12} {'output tok':>11} {'seconds':>8}  answer")
            for effort in efforts:
                api_reasoning, estimated, output_tokens, elapsed, answer = await _measure(model_name, effort, prompt)
                print(f"{effort:<8} {api_reasoning:>12} {estimated:>12} {output_tokens:>11} {elapsed:>8.1f}  {answer}")


if __name__ == "__main__":
    asyncio.run(main())

-------

packages/sta_agent_engine/src/sta_agent_engine/models/custom_chat_model.py
----
import contextlib
import logging
import os
import ssl
import warnings
from collections.abc import Mapping
from typing import Any, TypedDict, cast

import certifi
import httpx
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.block_translators import get_translator, register_translator
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.chat_models import global_ssl_context as _mistral_global_ssl_context
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import base as openai_base
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

from sta_agent_core.config import BaseProviderSettings, ProviderFactory
from sta_agent_core.types import ProviderType

from ..utils.signature_utils import expose_merged_signature
from .capabilities import is_multimodal
from .reasoning import build_reasoning_kwargs, merge_reasoning_config


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSL audit helper — diagnostic hook for trust-store issues
# ---------------------------------------------------------------------------
# Silent by default. Enable with STA_SSL_AUDIT=1 in the deployment env.
#
# When enabled, each call to _create_mistral_model / _create_openai_model
# emits a "ssl-audit provider=..." log line containing:
#   - cdc_module / cdc_qualname: who owns ssl.create_default_context right
#     now (vanilla 'ssl', 'truststore._api', 'pip_system_certs.*', ...).
#     Tells you whether any trust-store patch is actually active in the
#     worker process at call time.
#   - SSL_CERT_FILE / SSL_CERT_DIR: whether the runtime env vars are set.
#   - certifi.where(): which bundle certifi would hand out by default.
#   - ca_count: number of CAs loaded into a fresh default context. Compare
#     across providers and across failing/working deploys. ~150 = healthy
#     OS store; 0 = trust store is not being read; ~1 = only corp CA (bad).
#
# If SSL ever breaks again (cert rotation, Wolfi upgrade, LangGraph Platform
# runtime change), grep 'ssl-audit' in production logs for a one-shot
# diagnosis instead of multi-session debugging.
# ---------------------------------------------------------------------------
_SSL_AUDIT_ENABLED = os.environ.get("STA_SSL_AUDIT") == "1"
_ssl_audit_log = logging.getLogger("ssl-audit")


def _describe_ctx_ca_source(ctx: ssl.SSLContext) -> str:
    """Return a short string describing where ``ctx`` gets its CAs from.

    On truststore-backed contexts (truststore.SSLContext or
    pip._vendor.truststore.SSLContext), ``get_ca_certs`` and ``cert_store_stats``
    both raise ``NotImplementedError`` by design — trust validation is delegated
    to the OS. In that case we report ``os-delegated``. Otherwise we try to get
    a numeric count and fall back to ``unknown`` if both probes fail.
    """
    ctx_cls = type(ctx)
    ctx_mod = getattr(ctx_cls, "__module__", "") or ""
    if "truststore" in ctx_mod:
        return f"os-delegated({ctx_mod})"
    try:
        stats = ctx.cert_store_stats()
        return f"cert_store_stats={stats}"
    except NotImplementedError:
        pass
    except Exception as e:  # noqa: BLE001
        return f"cert_store_stats-error={e!r}"
    try:
        return f"get_ca_certs_len={len(ctx.get_ca_certs())}"
    except Exception as e:  # noqa: BLE001
        return f"unknown({e!r})"


def _log_ssl_state(provider_label: str) -> None:
    """Emit one per-call snapshot of the process's SSL trust state.

    No-op unless STA_SSL_AUDIT=1 is set in the environment. Never raises —
    diagnostic hooks must not interfere with the caller's control flow.

    The goal of this helper is to answer, at every call site, two questions
    that are otherwise impossible to tell apart in production:

    1. Is a trust-store delegation shim active right now (truststore,
       pip-system-certs, etc.), or is it vanilla ``ssl``?
    2. What CA bundle would a *fresh* default context actually see at this
       exact moment — same as at import time, or has something changed?
    """
    if not _SSL_AUDIT_ENABLED:
        return
    try:
        cdc = ssl.create_default_context
        ctx = cdc()
        _ssl_audit_log.warning(
            "ssl-audit provider=%s cdc_module=%r cdc_qualname=%r ctx_type=%s SSL_CERT_FILE=%r SSL_CERT_DIR=%r certifi=%r ca_source=%s",
            provider_label,
            getattr(cdc, "__module__", "?"),
            getattr(cdc, "__qualname__", "?"),
            type(ctx).__module__ + "." + type(ctx).__qualname__,
            os.environ.get("SSL_CERT_FILE"),
            os.environ.get("SSL_CERT_DIR"),
            certifi.where(),
            _describe_ctx_ca_source(ctx),
        )
    except Exception as e:  # noqa: BLE001 — logging must never break a request
        _ssl_audit_log.warning("ssl-audit provider=%s inspect failed: %r", provider_label, e)


def _log_httpx_client_ssl_state(provider_label: str, async_client: Any) -> None:
    """Introspect an already-constructed httpx AsyncClient's SSL context.

    httpx freezes its ssl context inside AsyncClient.__init__, so the process-wide
    state logged by _log_ssl_state() is necessary but not sufficient — we also want
    to see what the client actually captured. httpx's internal attribute names
    change between versions, so this is best-effort and silently skips on mismatch.
    """
    if not _SSL_AUDIT_ENABLED:
        return
    try:
        transport = getattr(async_client, "_transport", None)
        pool = getattr(transport, "_pool", None) if transport is not None else None
        sc = getattr(pool, "_ssl_context", None) or getattr(transport, "_ssl_context", None) or None
        if sc is None:
            _ssl_audit_log.warning(
                "ssl-audit provider=%s:client introspection unsupported for this httpx version",
                provider_label,
            )
            return
        _ssl_audit_log.warning(
            "ssl-audit provider=%s:client ctx_type=%s ca_source=%s",
            provider_label,
            type(sc).__module__ + "." + type(sc).__qualname__,
            _describe_ctx_ca_source(sc),
        )
    except Exception as e:  # noqa: BLE001
        _ssl_audit_log.warning("ssl-audit provider=%s:client inspect failed: %r", provider_label, e)


# Reasoning/thinking keywords across providers (DeepSeek, OpenAI o-series, etc.)
_REASONING_KEYWORDS = (
    "reasoning_content",
    "reasoning",
    "reasoning_block",
    "thinking_content",
    "think",
    "thinking",
    "thinking_block",
    "think_content",
)


def _extract_reasoning(source: Mapping[str, Any], target: dict[str, Any]) -> None:
    """Extract reasoning content from a response dict into additional_kwargs.

    Checks multiple provider-specific keys and normalises them under a single
    ``reasoning_content`` key so downstream consumers have a stable interface.
    """
    for keyword in _REASONING_KEYWORDS:
        value = source.get(keyword)
        if value is not None:
            target["reasoning_content"] = value
            return


def _coerce_reasoning_to_text(value: Any) -> str:
    """Coerce a captured reasoning value of any shape into display text.

    A ``content_blocks`` reasoning block carries a string ``reasoning`` field,
    but a provider may surface reasoning as a structured object rather than a
    plain string. Rules:

    - ``str`` → returned unchanged (the common gpt-oss / vLLM case).
    - a reasoning-shaped ``dict`` (``{"type": "reasoning", "content": ...}``) →
      its ``content`` (string-coerced).
    - anything else → ``str(value)``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("type") == "reasoning" and value.get("content") is not None:
        content = value["content"]
        return content if isinstance(content, str) else str(content)
    return str(value)


# ---------------------------------------------------------------------------
# Streaming patch — _convert_delta_to_message_chunk
# ---------------------------------------------------------------------------


def _custom_convert_delta_to_message_chunk(_dict: Mapping[str, Any], default_class: type[BaseMessageChunk]) -> BaseMessageChunk:
    """
    Custom version of _convert_delta_to_message_chunk with reasoning content support.

    This function extends the original langchain_openai conversion to handle reasoning
    tokens from models that support chain-of-thought reasoning (e.g., DeepSeek, o1).

    Args:
        _dict: Dictionary containing delta message information
        default_class: Default message chunk class to use

    Returns:
        BaseMessageChunk with reasoning content in additional_kwargs if present
    """
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}

    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call

    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        with contextlib.suppress(KeyError):
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]

    _extract_reasoning(_dict, additional_kwargs)

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        )
    if role in ("system", "developer") or default_class == SystemMessageChunk:
        additional_kwargs = {"__openai_role__": "developer"} if role == "developer" else {}
        return SystemMessageChunk(content=content, id=id_, additional_kwargs=additional_kwargs)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"], id=id_)  # type: ignore[call-arg]
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    return default_class(content=content, id=id_)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Non-streaming patch — _convert_dict_to_message
# ---------------------------------------------------------------------------


def _custom_convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Custom version of _convert_dict_to_message with reasoning content support.

    The original langchain_openai implementation silently drops ``reasoning_content``
    (and similar provider-specific thinking fields) from the full API response dict.
    This patched version mirrors the original logic and additionally captures reasoning
    content into ``additional_kwargs["reasoning_content"]`` for observability.

    Args:
        _dict: Dictionary from the OpenAI API response ``choices[].message``.

    Returns:
        BaseMessage with reasoning content in additional_kwargs when present.
    """
    role = _dict.get("role")
    name = _dict.get("name")
    id_ = _dict.get("id")
    content_raw = _dict.get("content")
    content_safe = content_raw if content_raw is not None else ""

    if role == "user":
        return HumanMessage(content=content_safe, id=id_, name=name)

    if role == "assistant":
        content = content_safe
        additional_kwargs: dict = {}

        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)

        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(make_invalid_tool_call(raw_tool_call, str(e)))

        if audio := _dict.get("audio"):
            additional_kwargs["audio"] = audio

        _extract_reasoning(_dict, additional_kwargs)

        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )

    if role in ("system", "developer"):
        additional_kwargs = {"__openai_role__": role} if role == "developer" else {}
        return SystemMessage(
            content=content_safe,
            name=name,
            id=id_,
            additional_kwargs=additional_kwargs,
        )

    if role == "function":
        return FunctionMessage(content=content_safe, name=cast(str, _dict.get("name")), id=id_)

    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=content_safe,
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )

    return ChatMessage(content=content_safe, role=role or "assistant", id=id_)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Monkey patches — reasoning content support for langchain_openai
# ---------------------------------------------------------------------------
# These patches mirror upstream functions from langchain_openai.chat_models.base
# and add reasoning content extraction. They are fragile — when langchain-openai
# is upgraded, the upstream functions may change and our patches silently diverge.
#
# Drift detection: tests/test_ai_engine/models/test_custom_chat_model.py contains
# hash-based tests that fail when the upstream source changes, prompting a review.
#
# Patched against: langchain-openai==1.1.9
# Upstream functions: _convert_delta_to_message_chunk, _convert_dict_to_message
# ---------------------------------------------------------------------------

# Save originals before patching (used by drift-detection tests)
_original_convert_delta_to_message_chunk = openai_base._convert_delta_to_message_chunk
_original_convert_dict_to_message = openai_base._convert_dict_to_message

openai_base._convert_delta_to_message_chunk = _custom_convert_delta_to_message_chunk
openai_base._convert_dict_to_message = _custom_convert_dict_to_message
logger.info("Applied custom reasoning parsing logic to langchain_openai (streaming + non-streaming)")


# ---------------------------------------------------------------------------
# content_blocks reasoning — reasoning-aware OpenAI block translator
# ---------------------------------------------------------------------------
# langchain-core derives ``AIMessage.content_blocks`` via a per-provider
# translator selected by ``response_metadata["model_provider"]``. langchain-openai
# stamps ``"openai"`` on every chunk, so core routes to its OpenAI chat-completions
# translator — which builds blocks only from ``.content`` + tool calls and ignores
# ``additional_kwargs`` entirely. Core's best-effort reasoning fallback lives
# *below* that provider branch and is never reached, so the reasoning captured by
# the converter patches above (``additional_kwargs["reasoning_content"]``) is
# dropped from ``content_blocks`` — a reasoning-only stream chunk yields ``[]``.
#
# We wrap the registered OpenAI translator to re-add a reasoning block from
# ``additional_kwargs`` when one is present and the underlying translator did not
# already emit it. Purely additive: text / tool-call / multimodal blocks are
# untouched, and a responses-API message that already carries a reasoning block
# is left alone. Note: ``register_translator`` mutates a process-wide registry, so
# this affects ``content_blocks`` for every ``model_provider="openai"`` message in
# the host process — the same global-patch posture as the converter patches above.
# ---------------------------------------------------------------------------


def _reasoning_block_from_message(message: BaseMessage) -> dict[str, Any] | None:
    """Build a ``{"type": "reasoning", ...}`` block from ``additional_kwargs`` if present."""
    raw = getattr(message, "additional_kwargs", {}).get("reasoning_content")
    if raw is None:
        return None
    text = _coerce_reasoning_to_text(raw)
    return {"type": "reasoning", "reasoning": text} if text else None


def _wrap_translator_with_reasoning(translate: Any) -> Any:
    """Wrap a core block translator so it prepends a reasoning block when missing."""

    def _translate(message: Any) -> Any:
        blocks = translate(message)
        already_has_reasoning = any(isinstance(block, dict) and block.get("type") == "reasoning" for block in blocks)
        if not already_has_reasoning and (reasoning_block := _reasoning_block_from_message(message)) is not None:
            blocks.insert(0, reasoning_block)
        return blocks

    return _translate


_original_openai_translator = get_translator("openai")
if _original_openai_translator is not None:
    register_translator(
        "openai",
        _wrap_translator_with_reasoning(_original_openai_translator["translate_content"]),
        _wrap_translator_with_reasoning(_original_openai_translator["translate_content_chunk"]),
    )
    logger.info("Registered reasoning-aware content_blocks translator for provider 'openai'")


# ---------------------------------------------------------------------------
# L1 stall fence — granular httpx timeout for create_chat_model
# ---------------------------------------------------------------------------
# Phase 3 (#47) of twin_router robustness. Every chat model created here
# inherits a per-axis httpx.Timeout so a hung upstream is killed at 30s of
# byte-level silence rather than 600s of wall clock. Every axis is
# env-tunable (CHAT_MODEL_HTTPX_{CONNECT,READ,WRITE,POOL}_TIMEOUT_S) and
# per-call overridable via ``timeouts=``; the read axis carries the real
# stall budget (30s) while connect / write / pool sit at a modest 10s.
# ---------------------------------------------------------------------------
DEFAULT_HTTPX_CONNECT_TIMEOUT_S = 10.0
DEFAULT_HTTPX_WRITE_TIMEOUT_S = 30.0
DEFAULT_HTTPX_POOL_TIMEOUT_S = 10.0
DEFAULT_HTTPX_READ_TIMEOUT_S = 50.0

_HTTPX_TIMEOUT_AXES: tuple[str, ...] = ("connect", "read", "write", "pool")
_HTTPX_DEFAULTS: dict[str, float] = {
    "connect": DEFAULT_HTTPX_CONNECT_TIMEOUT_S,
    "read": DEFAULT_HTTPX_READ_TIMEOUT_S,
    "write": DEFAULT_HTTPX_WRITE_TIMEOUT_S,
    "pool": DEFAULT_HTTPX_POOL_TIMEOUT_S,
}
_HTTPX_ENV_VARS: dict[str, str] = {
    "connect": "CHAT_MODEL_HTTPX_CONNECT_TIMEOUT_S",
    "read": "CHAT_MODEL_HTTPX_READ_TIMEOUT_S",
    "write": "CHAT_MODEL_HTTPX_WRITE_TIMEOUT_S",
    "pool": "CHAT_MODEL_HTTPX_POOL_TIMEOUT_S",
}
_TIER_ONLY_CONFIG_KEYS = frozenset({"big_model", "small_model", "thinking_model", "multimodal_model", "tier"})


def _warn_legacy_read_timeout_s() -> None:
    """Emit a one-line UserWarning steering callers to the `timeouts` API."""
    warnings.warn(
        "`read_timeout_s` is removed — use `timeouts={'read': N}` instead.",
        UserWarning,
        stacklevel=3,
    )


class TimeoutOverrides(TypedDict, total=False):
    """Per-axis httpx timeout overrides for chat-model factories.

    All keys optional; omitted axes fall back to env / module defaults.
    """

    connect: float
    read: float
    write: float
    pool: float


def _resolve_httpx_timeout(
    timeouts: TimeoutOverrides | httpx.Timeout | None = None,
) -> httpx.Timeout:
    """Return the granular httpx timeout for chat-model clients.

    Precedence per axis (highest first):
        1. ``timeouts`` argument — explicit per-axis override (dict or
           ``httpx.Timeout``). The single per-call override surface.
        2. ``CHAT_MODEL_HTTPX_{AXIS}_TIMEOUT_S`` env var — operator
           rollback knob, applies process-wide.
        3. ``DEFAULT_HTTPX_{AXIS}_TIMEOUT_S`` — module default.
    """
    overrides: dict[str, float] = {}
    if isinstance(timeouts, httpx.Timeout):
        for axis in _HTTPX_TIMEOUT_AXES:
            value = getattr(timeouts, axis, None)
            if value is not None:
                overrides[axis] = float(value)
    elif timeouts is not None:
        overrides = {axis: float(timeouts[axis]) for axis in _HTTPX_TIMEOUT_AXES if axis in timeouts}

    resolved: dict[str, float] = {}
    for axis in _HTTPX_TIMEOUT_AXES:
        if axis in overrides:
            resolved[axis] = overrides[axis]
            continue
        raw = os.environ.get(_HTTPX_ENV_VARS[axis])
        resolved[axis] = float(raw) if raw else _HTTPX_DEFAULTS[axis]

    return httpx.Timeout(**resolved)


def _env_prefix_for_provider(provider: ProviderType | str | None) -> str:
    """Return the env-var prefix that ``ProviderFactory`` reads for ``provider``.

    Used to build human-readable error messages naming the exact env vars
    a caller should set. ``ProviderType.CUSTOM`` and ``None`` map to the
    empty prefix (bare ``API_KEY`` / ``BASE_URL`` / ``MODEL``).
    """
    if isinstance(provider, ProviderType):
        return "" if provider == ProviderType.CUSTOM else f"{provider.value.upper()}_"
    if isinstance(provider, str) and provider:
        return f"{provider.upper()}_"
    return ""


def _warn_silent_openai_fallback(provider: ProviderType | str | None, missing: list[str]) -> None:
    """Emit a DeprecationWarning when the OpenAI dispatch is about to fall back
    to ``OPENAI_API_KEY`` + ``api.openai.com`` from the process environment.

    The silent fallback is a long-standing footgun: callers who pass a provider
    name but forget to set its env vars get a 404 from OpenAI's default endpoint
    for whatever model name was inferred. Warn now, hard-fail in 0.10.0.

    When the caller is not already using ``provider="openai"``, append a hint
    pointing at the default-registered OpenAI provider — most accidental
    fall-throughs are actually OpenAI usage that should be made explicit.
    """
    prefix = _env_prefix_for_provider(provider)
    env_vars = [f"{prefix}{field.upper()}" for field in missing]
    hint = ""
    is_openai_provider = isinstance(provider, str) and provider.lower() == "openai"
    if not is_openai_provider:
        hint = " If you are actually using OpenAI, switch to create_chat_model('openai') — it reads OPENAI_API_KEY natively."
    warnings.warn(
        f"Provider {provider!r} has no {missing} resolvable from env vars or "
        f"kwargs. ChatOpenAI will silently fall back to OPENAI_API_KEY + "
        f"api.openai.com from the process environment. Set {env_vars} or pass "
        f"{missing} as kwargs to create_chat_model(). The silent fallback is "
        f"deprecated and will raise in 0.10.0.{hint}",
        DeprecationWarning,
        stacklevel=3,
    )


def _is_mistral_model(model_name: str | None, provider: ProviderType | str | None) -> bool:
    """Check if the configuration indicates a Mistral model.

    Args:
        model_name: The model name from configuration.
        provider: The provider type. Either a ``ProviderType`` member or
            an arbitrary string. When a string carries ``"mistral"`` as a
            substring (e.g. ``"mistral_eu"``, ``"mistral_corp"``) the
            dispatch routes to ``ChatMistralAI`` even if the model name
            doesn't carry the brand.

    Returns:
        True if this is a Mistral model/provider.
    """
    provider_name_hits = isinstance(provider, str) and "mistral" in provider.lower()
    return (
        provider == ProviderType.MISTRAL
        or provider_name_hits
        or bool(model_name and "mistral" in model_name.lower())
        or bool(model_name and "devstral" in model_name.lower())
        or bool(model_name and "magistral" in model_name.lower())
    )


def _create_mistral_model(config: dict[str, Any]) -> ChatMistralAI:
    """Create a ChatMistralAI instance with the L1 stall fence injected.

    ChatMistralAI handles Mistral's message format correctly and doesn't include
    unsupported fields like 'name' in assistant messages.

    L1 (Phase 3 / #47): pre-built httpx clients carry the granular
    ``_resolve_httpx_timeout()`` and the bearer token so that the SDK's
    relative-path requests (``self.async_client.post('/chat/completions')``)
    resolve correctly. ``max_retries=0`` is forced because Mistral SDK
    retries restart the httpx timeout and don't honour Retry-After — leaving
    the SDK default would inflate our 30s stall budget by 6×.

    The SDK's own SSL context (``langchain_mistralai.chat_models.global_ssl_context``)
    is reused so truststore delegation on Wolfi stays consistent with the
    behaviour the SDK would have produced if we hadn't injected a client.
    """
    _log_ssl_state("mistral")

    api_key = config.pop("api_key", None) or os.environ.get("MISTRAL_API_KEY")
    base_url = config.pop("base_url", None) or os.environ.get("MISTRAL_BASE_URL") or "https://api.mistral.ai/v1"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # Per-call timeout override — pass ``timeouts={"connect": 10, "read": 90, ...}``
    # or an ``httpx.Timeout`` instance. The legacy ``read_timeout_s`` kwarg
    # is removed; map any stray callers to the new API.
    if "read_timeout_s" in config:
        _warn_legacy_read_timeout_s()
        config.pop("read_timeout_s", None)
    timeouts_override = config.pop("timeouts", None)
    timeout = _resolve_httpx_timeout(timeouts=timeouts_override)

    async_client = config.pop("async_client", None) or httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=timeout,
        verify=_mistral_global_ssl_context,
    )
    sync_client = config.pop("client", None) or httpx.Client(
        base_url=base_url,
        headers=headers,
        timeout=timeout,
        verify=_mistral_global_ssl_context,
    )

    _log_httpx_client_ssl_state("mistral", async_client)

    mistral_config: dict[str, Any] = {
        "api_key": api_key,
        "endpoint": base_url,  # ChatMistralAI Field alias for base_url
        "model": config.pop("model", None),
        "temperature": config.pop("temperature", None),
        "max_tokens": config.pop("max_tokens", None),
        "top_p": config.pop("top_p", None),
        # Force max_retries=0 — Mistral SDK retries restart the httpx timeout
        # and don't honour Retry-After. Let with_retry() / agent middleware
        # own retry semantics instead.
        "max_retries": 0,
    }

    if model_kwargs := config.pop("model_kwargs", None):
        mistral_config["model_kwargs"] = model_kwargs

    # Drop the SDK's own ``timeout`` from the caller's config — the granular
    # httpx.Timeout on the injected client is the source of truth now.
    config.pop("timeout", None)
    config.pop("max_retries", None)

    # Filter out None values from the explicit fields we set above.
    mistral_config = {k: v for k, v in mistral_config.items() if v is not None}

    # Anything else the caller passed (custom Mistral kwargs) flows through.
    mistral_config = {**config, **mistral_config}

    logger.info(f"Creating ChatMistralAI with model={mistral_config.get('model')}")
    return ChatMistralAI(client=sync_client, async_client=async_client, **mistral_config)


def _create_openai_model(config: dict[str, Any]) -> ChatOpenAI:
    """Create a ChatOpenAI instance with the L1 stall fence injected.

    L1 (Phase 3 / #47): pre-built httpx clients carry the granular
    ``_resolve_httpx_timeout()`` so the openai SDK's transport inherits a
    30s read fence instead of the previous 600s blanket. ``max_retries=1``
    keeps one Retry-After-aware retry for 429 storms — OpenAI's SDK retries
    are well-behaved here, unlike Mistral's.
    """
    _log_ssl_state("openai")

    # Per-call timeout override — pass ``timeouts={"connect": 10, "read": 90, ...}``
    # or an ``httpx.Timeout`` instance. Legacy ``read_timeout_s`` is removed.
    if "read_timeout_s" in config:
        _warn_legacy_read_timeout_s()
        config.pop("read_timeout_s", None)
    timeouts_override = config.pop("timeouts", None)
    timeout = _resolve_httpx_timeout(timeouts=timeouts_override)

    async_client = config.pop("http_async_client", None) or DefaultAsyncHttpxClient(timeout=timeout)
    sync_client = config.pop("http_client", None) or DefaultHttpxClient(timeout=timeout)

    _log_httpx_client_ssl_state("openai", async_client)

    config.setdefault("max_retries", 1)

    return ChatOpenAI(
        http_async_client=async_client,
        http_client=sync_client,
        profile={"max_input_tokens": 128_000},
        **config,
    )


def _resolve_multimodal_model(settings: BaseProviderSettings, provider: str | ProviderType | None) -> str:
    """Resolve the model to use for a multimodal (vision) request.

    Resolution order (hard-fails rather than silently returning a text model):

        1. ``settings.multimodal_model`` if set — authoritative, used verbatim.
           Not re-checked against ``is_multimodal`` so an operator can declare a
           vision model the static capability list doesn't yet recognize.
        2. ``settings.model`` if ``is_multimodal(settings.model)`` — the default
           already points at a vision-capable model.
        3. Otherwise raise ``ValueError`` — ``multimodal=True`` was requested but
           the deployment has no multimodal model configured. Failing loud at
           construction beats handing back a text model that silently strips
           images (under the guard middleware) or 400s upstream.

    Args:
        settings: Resolved provider settings.
        provider: Original provider argument, for the error message.

    Returns:
        The model identifier to use for the vision request.

    Raises:
        ValueError: If no multimodal model can be resolved.
    """
    if settings.multimodal_model:
        return settings.multimodal_model
    if is_multimodal(settings.model):
        return settings.model
    env_prefix = type(settings).model_config.get("env_prefix") or ""
    provider_label = provider.value if isinstance(provider, ProviderType) else (provider or "default")
    raise ValueError(
        f"multimodal=True but no multimodal model is configured for provider {provider_label!r}: "
        f"set {env_prefix}MULTIMODAL_MODEL, or point {env_prefix}MODEL at a vision-capable model."
    )


# Tier -> the capacity-tier fields whose explicit presence means the resolved model
# did NOT fall through to the base ``model`` default. Mirrors the cascade in
# ``BaseProviderSettings.get_model``.
_TIER_MODEL_FIELDS: dict[str, tuple[str, ...]] = {
    "default": (),
    "big": ("big_model",),
    "small": ("small_model",),
    "thinking": ("thinking_model", "big_model"),
}


def _model_is_implicit_default(settings: BaseProviderSettings, tier: str | None, resolved_model: str) -> bool:
    """True when ``resolved_model`` came from the class default, not an explicit source.

    "Explicit" means a ``*_MODEL`` / ``*_<TIER>_MODEL`` env var, a registered
    default, an init value, or a ``model=`` kwarg. Detection uses the value-vs-
    default comparison rather than ``model_fields_set`` because the
    ``empty_str_to_none`` validator coerces ``*_MODEL=""`` to ``None`` while still
    marking the field "set" — so ``model_fields_set`` reports a falsely-explicit
    model. Comparing the resolved value to the field default is robust to that.

    Known benign false-positive: a consumer who explicitly sets the model to the
    exact same string as the built-in default is still treated as implicit.
    """
    tier_name = "default" if tier is None else str(tier).lower().strip()
    fields_set = settings.model_fields_set
    # An explicitly-set tier field supplied the model -> the base default was not used.
    for field in _TIER_MODEL_FIELDS.get(tier_name, ()):
        if field in fields_set and getattr(settings, field, None):
            return False
    model_field_default = type(settings).model_fields["model"].default
    return resolved_model == model_field_default


def create_chat_model(
    provider: str | ProviderType | None = None,
    *,
    tier: str = "default",
    multimodal: bool = False,
    **kwargs: Any,
) -> ChatOpenAI | ChatMistralAI:
    """
    Factory function to create a chat model with provider-specific configuration.

    Automatically selects the appropriate client based on the provider:
    - ChatMistralAI for Mistral models (handles Mistral's stricter message format)
    - ChatOpenAI for OpenAI-compatible APIs

    Built-in providers:
    - LLMaaS: Set LLM_PROVIDER=llmaas, configure with LLMAAS_* env vars
    - LLMaaS Dev: Set LLM_PROVIDER=llmaas_dev, configure with LLMAAS_DEV_* env vars
    - Mistral: Set LLM_PROVIDER=mistral, configure with MISTRAL_* env vars
    - Custom: Set LLM_PROVIDER=custom, configure with {NO_PREFIX}* env vars

    Dynamic providers:
        Any other string is accepted. The env prefix is derived as
        ``f"{NAME.upper()}_"`` and read via Pydantic Settings:

        - ``{NAME}_API_KEY``
        - ``{NAME}_BASE_URL``
        - ``{NAME}_MODEL``
        - ``{NAME}_BIG_MODEL`` / ``{NAME}_SMALL_MODEL`` / ``{NAME}_THINKING_MODEL``
        - ``{NAME}_MULTIMODAL_MODEL``
        - ``{NAME}_TEMPERATURE`` / ``{NAME}_TOP_P`` / ``{NAME}_MAX_TOKENS``

        Use ``ProviderFactory.register(name, defaults=..., env_prefix=...)``
        for non-env defaults or a custom env prefix.

    Model tiers:
        ``tier`` selects a provider model slot when no explicit ``model=`` kwarg
        is passed. Supported tiers are ``default``, ``big``, ``small``, and
        ``thinking``. ``thinking`` cascades to ``big`` and then ``model``.

    Multimodal:
        ``multimodal=True`` requests a vision-capable model (capability axis,
        not a capacity tier). It resolves ``{NAME}_MULTIMODAL_MODEL`` if set,
        else ``{NAME}_MODEL`` when that is a recognized multimodal model, else
        raises ``ValueError`` — it never silently returns a text-only model.
        ``multimodal=True`` takes precedence over ``tier`` and is ignored when
        an explicit ``model=`` kwarg is passed.

    Mistral dispatch:
        Routes to ``ChatMistralAI`` when any of these hold:

        - ``provider == ProviderType.MISTRAL``
        - The provider name (string) contains ``"mistral"``
        - The model name contains ``"mistral"`` / ``"devstral"`` / ``"magistral"``

    Credential resolution:
        ``api_key`` and ``base_url`` should resolve via env vars (under the
        provider's prefix) or kwargs. When the OpenAI dispatch path is taken
        and either is missing, ``ChatOpenAI`` silently falls back to
        ``OPENAI_API_KEY`` + ``api.openai.com`` from the process env. This
        emits a ``DeprecationWarning`` and will raise ``ValueError`` in 0.10.0.

    Reasoning control:
        ``reasoning_effort="off" | "low" | "medium" | "high" | "xhigh" | "max"`` is
        translated per model family into the kwargs the model actually honors
        (native GLM/DeepSeek/Qwen3.8 effort, Mistral's ``reasoning_effort``, or
        Nemotron/earlier-Qwen/Gemma 4 ``chat_template_kwargs`` — see
        ``models/reasoning.py``). Omitted -> nothing is injected (the
        model keeps its server-side default). Unsupported efforts warn and
        never silently substitute; explicitly-passed ``extra_body`` /
        ``model_kwargs`` win over translated values on conflicting keys.
        ``reasoning_family="nemotron-super"`` pins the family when a gateway
        alias hides the model name. Prefer one effort per agent/task/thread —
        flipping ``chat_template_kwargs`` per turn busts server-side prefix
        caches. Consumer guide: ``docs/consuming/reasoning.md``.

    Args:
        provider: ``ProviderType`` member, built-in provider string,
            or any arbitrary dynamic-provider name.
        tier: Capacity tier to resolve from provider settings when ``model``
            is not passed explicitly.
        multimodal: When True, resolve a vision-capable model (see Multimodal
            above). Takes precedence over ``tier``; ignored if ``model=`` is set.
        **kwargs: Additional configuration to override environment variables.

    Returns:
        BaseChatModel instance (ChatMistralAI or ChatOpenAI) configured for the specified provider.

    Examples:
        # Built-in provider via environment variables
        export LLM_PROVIDER=mistral
        export MISTRAL_API_KEY=your_api_key
        export MISTRAL_MODEL=mistral-small-2603
        model = create_chat_model()

        # Dynamic provider — no code change required
        export ACME_API_KEY=your_key
        export ACME_BASE_URL=https://api.acme.test/v1
        export ACME_MODEL=acme-large
        model = create_chat_model("acme")

        # Explicit Mistral provider
        model = create_chat_model("mistral")

        # Mistral-flavored dynamic provider (auto-routes to ChatMistralAI)
        model = create_chat_model("mistral_eu", api_key="...", base_url="https://eu.mistral.test", model="mistral-large")
    """
    # Lowercase string providers so dispatch matches the factory's normalization,
    # but keep the original string when it doesn't map to a known enum member.
    if isinstance(provider, str):
        try:
            provider = ProviderType(provider.lower())
        except ValueError:
            provider = provider.lower()

    provider_settings = ProviderFactory.get_provider_settings(provider)
    settings = provider_settings.model_dump()

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if "model" not in kwargs:
        if multimodal:
            settings["model"] = _resolve_multimodal_model(provider_settings, provider)
        else:
            settings["model"] = provider_settings.get_model(tier)

    # Extract credentials from context if available (BYOK mode).
    # ``pop`` (not lookup): these are aliases for ``api_key`` / ``base_url``, not
    # client constructor params. Leaving them in ``kwargs`` would merge them into
    # ``config`` below and forward them to ChatOpenAI/ChatMistralAI, which route
    # unknown kwargs into ``model_kwargs`` — leaking the raw key into the request
    # body sent over the wire and 400-ing strict OpenAI-compatible servers.
    if kwargs:
        if (provider_api_key := kwargs.pop("provider_api_key", None)) is not None:
            settings["api_key"] = provider_api_key
        if (provider_base_url := kwargs.pop("provider_base_url", None)) is not None:
            settings["base_url"] = provider_base_url

    config = {k: v for k, v in {**settings, **kwargs}.items() if v is not None}
    for key in _TIER_ONLY_CONFIG_KEYS:
        config.pop(key, None)

    # Every provider carries a default model slug, so a model always resolves and
    # construction never fails for lack of one. But relying on that *implicit*
    # default is deprecated: a consumer who never configured a model is almost
    # certainly pointing the default slug at an endpoint that does not serve it
    # (a wrong default is dangerous for third-party providers). Warn when the
    # resolved model came from the class default rather than an explicit source —
    # a *_MODEL / *_<TIER>_MODEL env var, a registered default, or model=... — and
    # keep working for now. This becomes a hard ValueError in 0.11.0. The
    # multimodal path is exempt: it already hard-fails when nothing is configured.
    if "model" not in kwargs and not multimodal and _model_is_implicit_default(provider_settings, tier, config.get("model", "")):
        prefix = type(provider_settings).model_config.get("env_prefix") or ""
        warnings.warn(
            f"create_chat_model({provider!r}) is using the built-in default model {config.get('model')!r}. "
            "Relying on an implicit default is deprecated and will raise in 0.11.0. Set "
            f"{prefix}MODEL (or {prefix}<TIER>_MODEL), register a default via "
            "ProviderFactory.register(name, defaults={'model': ...}), or pass model=... explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Determine if this is a Mistral model and use appropriate client.
    # Mistral's SDK has its own legitimate base_url fallback to api.mistral.ai —
    # do NOT warn here. The footgun only exists on the OpenAI dispatch path.
    model_name = config.get("model", "")

    # Reasoning-effort resolution (see models/reasoning.py): translate the
    # normalized effort into the per-family request kwargs. Explicit caller
    # kwargs win over translated values on leaf conflicts — raw extra_body /
    # model_kwargs are the escape hatch. ``reasoning_family`` pins the family
    # when a gateway alias hides the model name; popped unconditionally so it
    # never leaks into the client constructor.
    reasoning_family = config.pop("reasoning_family", None)
    if (reasoning_effort := config.pop("reasoning_effort", None)) is not None:
        provider_str = str(provider) if provider is not None else None
        reasoning_config = build_reasoning_kwargs(model_name, reasoning_effort, provider=provider_str, family=reasoning_family)
        config, reasoning_overrides = merge_reasoning_config(reasoning_config, config)
        if reasoning_overrides:
            warnings.warn(
                f"reasoning_effort={reasoning_effort!r}: explicitly-passed kwargs override the translated reasoning keys "
                f"{', '.join(sorted(reasoning_overrides))} — the explicit values win.",
                UserWarning,
                stacklevel=2,
            )
    elif reasoning_family is not None:
        warnings.warn(
            f"reasoning_family={reasoning_family!r} was given without reasoning_effort — it selects the translation table "
            "but injects nothing on its own, so it was ignored. Pass reasoning_effort=... alongside it.",
            UserWarning,
            stacklevel=2,
        )
    if _is_mistral_model(model_name, provider):
        logger.info(f"Detected Mistral model/provider, using ChatMistralAI for model={model_name}")
        return _create_mistral_model(config)

    # OpenAI dispatch — if api_key / base_url didn't resolve, ChatOpenAI will
    # silently pick up OPENAI_API_KEY + api.openai.com from the process env.
    # Restored to preserve main-branch behavior; the silent fallback is now
    # deprecated and will raise in 0.10.0.
    missing: list[str] = []
    if not config.get("api_key"):
        missing.append("api_key")
    if not config.get("base_url"):
        missing.append("base_url")
    if missing:
        _warn_silent_openai_fallback(provider, missing)

    return _create_openai_model(config)


# Expose merged signature for better IDE support and introspection
expose_merged_signature(create_chat_model, ChatOpenAI)

# Backward compatibility alias - can be removed after updating all imports
CustomChatModel = create_chat_model

-------

packages/sta_agent_engine/src/sta_agent_engine/models/reasoning.py
----
"""Declarative reasoning-effort configuration for chat models.

Model families expose incompatible knobs for controlling reasoning/thinking:
Mistral takes a top-level ``reasoning_effort`` string, Nemotron wants booleans
nested under ``extra_body.chat_template_kwargs``, Gemma 4 uses a binary
``enable_thinking`` template flag, and GLM/DeepSeek combine a top-level effort
with a separate thinking-mode toggle.
This module maps one normalized effort vocabulary (``off`` / ``low`` /
``medium`` / ``high`` / ``xhigh`` / ``max``) onto the literal request kwargs each family honors, so
callers write ``reasoning_effort="high"`` once and the right wire format is
emitted for whatever model resolves. Efforts are open strings — a family
registered via :func:`register_reasoning_family` may define additional rungs.

The registry is a plain data table — family name -> effort -> literal kwargs.
Adding a model family is one :func:`register_reasoning_family` call (or one
table entry); no dispatch code changes. The error posture is warn-not-raise:
requesting reasoning must never break model construction. Unknown families
degrade to forwarding ``reasoning_effort`` natively (the OpenAI form), which
genuine OpenAI APIs honor and OpenAI-compatible gateways typically ignore.

Guidance: pick an effort per agent/task/thread, not per turn. For
template-flag families, flipping ``chat_template_kwargs`` between turns of one
conversation re-renders the server-side prompt template and defeats
vLLM/SGLang prefix caching.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any


# Warnings are attributed to the first stack frame OUTSIDE this package
# directory (PEP 678-era ``skip_file_prefixes``, Python 3.12+). A fixed
# ``stacklevel`` can't be right for every entry point (direct call vs. routed
# through create_chat_model adds frames), and misattribution has a second
# cost: Python's default once-per-location filter would collapse every
# consumer call site onto one internal library line, silencing all repeats.
_WARN_SKIP_PREFIXES = (str(Path(__file__).resolve().parent),)


__all__ = [
    "build_reasoning_kwargs",
    "register_reasoning_family",
    "resolve_reasoning_family",
    "supported_reasoning_efforts",
]


# Family spec keys:
#   "match":          model-name patterns. A flat tuple of strings is ONE AND-group:
#                     all substrings must appear. A tuple of tuples is OR-of-AND-groups:
#                     the family matches if ANY group has all its substrings present —
#                     e.g. (("nemotron-3", "ultra"), ("nemo-ultra",)) covers both slug
#                     dialects. Matching is case- AND separator-insensitive: both sides
#                     are normalized by stripping `-`, `_`, `.`, `:`, `/` and spaces, so
#                     ("qwen3",) matches "qwen3.6", "qwen3-6", "Qwen/Qwen3.6-32B", and
#                     "qwen3:32b" alike — provider slug conventions don't matter.
#   "provider_match": substrings matched against the provider name (any hit wins;
#                     same normalization)
#   "native_path":    nested key path for forwarding a raw effort string when the
#                     requested effort has no rung (None -> nothing is injected).
#                     Empty "rungs" + a native_path = pure passthrough family (no warning).
#   "rungs":          effort -> literal constructor kwargs to merge
_FAMILIES: dict[str, dict[str, Any]] = {
    # GLM-5.2 exposes a flat reasoning_effort plus an independent thinking
    # toggle. Its API accepts the compatibility vocabulary below; low/medium
    # are evaluated as high, xhigh as max, and none/minimal stop thinking.
    # ``off`` uses the unambiguous toggle instead of inventing another native
    # effort alias. Keep this before any broader GLM family added later.
    "glm-5.2": {
        "match": ("glm", "5.2"),
        "provider_match": (),
        "native_path": ("reasoning_effort",),
        "rungs": {
            "off": {"extra_body": {"thinking": {"type": "disabled"}}},
            "none": {"reasoning_effort": "none"},
            "minimal": {"reasoning_effort": "minimal"},
            "low": {"reasoning_effort": "low", "extra_body": {"thinking": {"type": "enabled"}}},
            "medium": {"reasoning_effort": "medium", "extra_body": {"thinking": {"type": "enabled"}}},
            "high": {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}},
            "xhigh": {"reasoning_effort": "xhigh", "extra_body": {"thinking": {"type": "enabled"}}},
            "max": {"reasoning_effort": "max", "extra_body": {"thinking": {"type": "enabled"}}},
        },
    },
    # DeepSeek V4 supports only high/max as effective native levels. The API
    # accepts the wider compatibility vocabulary, mapping low/medium -> high
    # and xhigh -> max. Encode the effective values explicitly so the emitted
    # request is stable across official and compatible gateways.
    "deepseek-v4": {
        "match": ("deepseek", "v4"),
        "provider_match": (),
        "native_path": ("reasoning_effort",),
        "rungs": {
            "off": {"extra_body": {"thinking": {"type": "disabled"}}},
            "low": {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}},
            "medium": {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}},
            "high": {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}},
            "xhigh": {"reasoning_effort": "max", "extra_body": {"thinking": {"type": "enabled"}}},
            "max": {"reasoning_effort": "max", "extra_body": {"thinking": {"type": "enabled"}}},
        },
    },
    # Nemotron-3-Ultra: `low` sends force_nonempty_content=False (explicitly off,
    # so a server-side template default can't force it); medium/high carry
    # force_nonempty_content=True (SGLang requires it when tool-calling with
    # thinking enabled; harmless on non-tool calls).
    "nemotron-ultra": {
        "match": ("nemotron-3", "ultra"),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "low": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True, "force_nonempty_content": False}}},
            "medium": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True, "force_nonempty_content": True}}},
            "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "force_nonempty_content": True}}},
        },
    },
    "nemotron-super": {
        "match": ("nemotron-3", "super"),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "low": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}},
            "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        },
    },
    # Qwen3.8-2.4T requires thinking on every request, so it deliberately has no
    # off rung. Its native effort vocabulary is low/medium/xhigh; ``high`` is
    # the library's portable alias for the strongest supported level.
    "qwen3.8-always-thinking": {
        "match": ("qwen3.8", "2.4t"),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "low": {"reasoning_effort": "low"},
            "medium": {"reasoning_effort": "medium"},
            "high": {"reasoning_effort": "xhigh"},
            "xhigh": {"reasoning_effort": "xhigh"},
        },
    },
    # Other Qwen3.8 models expose the same graded native effort plus a hard
    # thinking switch. The built-in off rung targets self-hosted vLLM/SGLang,
    # where the switch rides in chat_template_kwargs. Qwen Cloud instead uses
    # a direct extra_body.enable_thinking field; pass that raw provider kwarg
    # without the normalized off rung when using its endpoint.
    "qwen3.8": {
        "match": ("qwen3.8",),
        "provider_match": (),
        "native_path": ("reasoning_effort",),
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "low": {"reasoning_effort": "low"},
            "medium": {"reasoning_effort": "medium"},
            "high": {"reasoning_effort": "xhigh"},
            "xhigh": {"reasoning_effort": "xhigh"},
        },
    },
    # Earlier Qwen3.x models: thinking on by default; the model card documents
    # only the binary chat_template_kwargs.enable_thinking switch
    # (Qwen/Qwen3.6-27B). Graded thinking budgets are a serving-stack feature
    # (vLLM `thinking_token_budget`, version-dependent) — deliberately NOT baked
    # in; gateways that support them can re-register this family with budget
    # rungs (see docs/consuming/reasoning.md).
    # The card also documents chat_template_kwargs.preserve_thinking=True for
    # keeping reasoning traces across agent turns — orthogonal to effort, pass it
    # via explicit extra_body (it deep-merges alongside these rungs).
    "qwen3": {
        "match": ("qwen3",),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        },
    },
    # Gemma 4 exposes only a binary chat-template thinking switch. Thinking is
    # disabled by default in the official template; there is no documented
    # graded reasoning effort or token budget. ``preserve_thinking`` controls
    # whether earlier assistant reasoning is retained and is orthogonal to
    # effort, so callers may pass it explicitly in chat_template_kwargs.
    "gemma4": {
        "match": ("gemma4",),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        },
    },
    # Mistral small/medium accept only none/high; ChatMistralAI has no native
    # reasoning_effort field, so the value rides in model_kwargs (flattened into
    # the request payload). Unsupported efforts pass through raw via native_path
    # so the Mistral API validates them itself. The match groups mirror
    # _is_mistral_model's dispatch criteria: every model that routes to
    # ChatMistralAI must use this wire dialect ("off" must become "none").
    "mistral": {
        "match": (("mistral",), ("devstral",), ("magistral",)),
        "provider_match": ("mistral",),
        "native_path": ("model_kwargs", "reasoning_effort"),
        "rungs": {
            "off": {"model_kwargs": {"reasoning_effort": "none"}},
            "high": {"model_kwargs": {"reasoning_effort": "high"}},
        },
    },
    # Real OpenAI reasoning models take reasoning_effort natively — pure silent
    # passthrough (the API validates values like "minimal"/"low"/"medium"/"high").
    "openai": {
        "match": ("gpt",),
        "provider_match": (),
        "native_path": ("reasoning_effort",),
        "rungs": {},
    },
}


_SLUG_SEPARATORS = str.maketrans("", "", "-_.:/ ")


def _normalize_slug(name: str) -> str:
    """Canonicalize a model/provider slug for matching.

    Lowercases and strips separator characters (``- _ . : /`` and spaces), so
    provider-specific slug conventions collapse to one form: ``qwen3.8``,
    ``qwen3-8``, ``Qwen/Qwen3.8-27B``, and ``qwen3:27b`` all contain the
    normalized pattern ``qwen38``/``qwen3``.
    """
    return name.lower().translate(_SLUG_SEPARATORS)


def _nested_from_path(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """Build a nested dict placing ``value`` at the given key path."""
    out: Any = value
    for key in reversed(path):
        out = {key: out}
    return out


def _match_substrings(spec_value: Any) -> tuple[str, ...]:
    """Coerce a spec's match value to a tuple of normalized patterns.

    Accepts a lone string as a single pattern — ``("qwen3")`` (missing trailing
    comma) is a string in Python, and iterating it would silently degrade to
    character-wise matching.
    """
    if isinstance(spec_value, str):
        spec_value = (spec_value,)
    return tuple(_normalize_slug(s) for s in spec_value)


def _match_groups(spec_value: Any) -> tuple[tuple[str, ...], ...]:
    """Coerce a spec's ``match`` value to OR-groups of normalized AND-substrings.

    - A lone string -> one group with one substring.
    - A flat tuple of strings -> ONE group (all substrings must match — AND).
    - A tuple containing any nested tuple/list -> every element is its own
      group (string elements become 1-substring groups); the family matches if
      ANY group fully matches (OR of ANDs).
    """
    if isinstance(spec_value, str):
        return ((_normalize_slug(spec_value),),)
    items = tuple(spec_value)
    if any(isinstance(item, (tuple, list)) for item in items):
        return tuple(_match_substrings(item) for item in items)
    return (_match_substrings(items),) if items else ()


def resolve_reasoning_family(model: str, *, provider: str | None = None, family: str | None = None) -> str | None:
    """Resolve which reasoning family applies to a model.

    Precedence: explicit ``family`` > provider substring match > model-name
    substring match (all of a family's ``match`` substrings must appear).
    Matching is case- and separator-insensitive (see :func:`_normalize_slug`),
    so the same model resolves identically across provider slug conventions —
    ``qwen3.8``, ``qwen3-8``, and ``Qwen/Qwen3.8-27B`` are one family.
    An explicit family that is not registered warns and falls back to the
    match-based resolution, so a typo degrades instead of silently no-oping.

    Args:
        model: Resolved model name (as sent to the endpoint).
        provider: Optional provider name (e.g. ``"mistral"``, ``"llmaas"``).
        family: Optional explicit family pin — use when a gateway alias hides
            the real model name (``chat-default`` actually serving Nemotron).

    Returns:
        The family name, or ``None`` if nothing matches.
    """
    if family is not None:
        if family in _FAMILIES:
            return family
        warnings.warn(
            f"Unknown reasoning_family {family!r} (registered: {sorted(_FAMILIES)}); falling back to model-name matching.",
            UserWarning,
            stacklevel=2,
            skip_file_prefixes=_WARN_SKIP_PREFIXES,
        )
    provider_normalized = _normalize_slug(provider or "")
    model_normalized = _normalize_slug(model)
    if provider_normalized:
        for name, spec in _FAMILIES.items():
            if any(sub in provider_normalized for sub in _match_substrings(spec["provider_match"])):
                return name
    for name, spec in _FAMILIES.items():
        for group in _match_groups(spec["match"]):
            if group and all(sub in model_normalized for sub in group):
                return name
    return None


def supported_reasoning_efforts(model: str, *, provider: str | None = None, family: str | None = None) -> frozenset[str]:
    """Return the effort names a model's family defines rungs for.

    An empty set means either no family matched or the family is a pure
    native-passthrough (any value is forwarded unvalidated for the API to judge).
    """
    resolved = resolve_reasoning_family(model, provider=provider, family=family)
    if resolved is None:
        return frozenset()
    return frozenset(_FAMILIES[resolved]["rungs"])


def build_reasoning_kwargs(model: str, effort: str | None, *, provider: str | None = None, family: str | None = None) -> dict[str, Any]:
    """Translate a normalized reasoning effort into model-specific kwargs.

    The returned dict is ready to splat into the model constructor. Families
    that translate into ``extra_body`` or a native field also work per call::

        kw = build_reasoning_kwargs("nemotron-3-super-120b", "low")
        model.invoke(messages, **kw)        # or model.bind(**kw)

    Exception: families whose kwargs ride ``model_kwargs`` (mistral) are
    constructor-only — ``ChatMistralAI`` flattens ``model_kwargs`` into the
    payload only from the constructor field; a call-time kwarg is posted as a
    literal ``"model_kwargs"`` JSON key. Set the effort at construction
    (``create_chat_model(..., reasoning_effort=...)``) for those.

    Behavior:
        - ``effort=None`` (or blank/whitespace) returns ``{}`` (inject nothing —
          blank means "unset", e.g. an empty env-var default).
        - A supported effort returns a fresh copy of the family's rung kwargs.
        - An unsupported effort warns (listing the supported set); families with
          a native passthrough forward the raw value for the API to validate,
          others return ``{}`` (there is no field to receive the value).
        - No family matched: warns and forwards ``{"reasoning_effort": effort}``
          (the OpenAI-native form — gateways that don't know it ignore it).

    Prefer one effort per agent/task/thread over per-turn flips: changing
    ``chat_template_kwargs`` mid-conversation busts server-side prefix caches.
    """
    if effort is None:
        return {}
    effort_name = str(effort).lower().strip()
    if not effort_name:
        return {}
    resolved = resolve_reasoning_family(model, provider=provider, family=family)
    if resolved is None:
        warnings.warn(
            f"No reasoning family matches model {model!r}; forwarding reasoning_effort={effort_name!r} as a native top-level parameter. "
            "OpenAI-style APIs honor it; vLLM/LiteLLM-fronted gateways typically ignore it. "
            "Register the model with register_reasoning_family(...) or pin reasoning_family=... to translate it properly.",
            UserWarning,
            stacklevel=2,
            skip_file_prefixes=_WARN_SKIP_PREFIXES,
        )
        return {"reasoning_effort": effort_name}
    spec = _FAMILIES[resolved]
    rungs: dict[str, dict[str, Any]] = spec["rungs"]
    if effort_name in rungs:
        return deepcopy(rungs[effort_name])
    native_path: tuple[str, ...] | None = spec["native_path"]
    if native_path is not None:
        if rungs:  # known vocabulary exists and the request is outside it
            warnings.warn(
                f"reasoning_effort={effort_name!r} is not defined for family {resolved!r} (model {model!r}; supported: {sorted(rungs)}). "
                "Forwarding the raw value for the API to validate.",
                UserWarning,
                stacklevel=2,
                skip_file_prefixes=_WARN_SKIP_PREFIXES,
            )
        return _nested_from_path(native_path, effort_name)
    warnings.warn(
        f"reasoning_effort={effort_name!r} is not supported by family {resolved!r} (model {model!r}; supported: {sorted(rungs)}). "
        "Nothing was applied — the model keeps its server-side default.",
        UserWarning,
        stacklevel=2,
        skip_file_prefixes=_WARN_SKIP_PREFIXES,
    )
    return {}


def register_reasoning_family(
    name: str,
    rungs: dict[str, dict[str, Any]],
    *,
    match_substrings: str | tuple[str | tuple[str, ...], ...] = (),
    provider_substrings: str | tuple[str, ...] = (),
    native_path: tuple[str, ...] | None = None,
) -> None:
    """Register (or replace) a reasoning family.

    Args:
        name: Family name; re-registering an existing name (including a
            built-in) replaces it, letting consumers override library defaults.
        rungs: Effort name -> literal constructor kwargs to merge (e.g.
            ``{"high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}}``).
        match_substrings: Model-name patterns for auto-detection. A flat tuple
            of strings is ONE AND-group — all substrings must appear:
            ``("nemotron-3", "ultra")``. A tuple of tuples is OR-of-AND-groups —
            the family matches when ANY group fully matches:
            ``(("nemotron-3", "ultra"), ("nemo-ultra",))`` covers two slug
            dialects. Matching is case- and separator-insensitive (``- _ . : /``
            and spaces are stripped from both sides), so ``("qwen3",)`` covers
            ``qwen3.6`` / ``qwen3-6`` / ``Qwen/Qwen3.6-32B`` regardless of the
            provider's slug convention.
        provider_substrings: Provider-name substrings that force this family
            (any hit wins; same normalization).
        native_path: Optional nested key path for forwarding a raw effort value
            when the requested effort has no rung (e.g. ``("reasoning_effort",)``).

    Note:
        Resolution scans families in registration order and the first full
        match wins — a *new* family whose patterns overlap an already-registered
        one (built-ins included) never wins for models both match. To change
        behavior for such models, re-register under the existing family's name.

    Raises:
        ValueError: If a rung value is not a dict (it must be literal kwargs),
            or if any match/provider pattern is empty after normalization
            (an empty pattern would silently match every model).
    """
    for rung_name, rung_kwargs in rungs.items():
        if not isinstance(rung_kwargs, dict):
            raise ValueError(f"Rung {rung_name!r} of family {name!r} must be a dict of constructor kwargs, got {type(rung_kwargs).__name__}")
    # A lone string is accepted as a single pattern (a missing trailing comma in
    # a 1-tuple is a string in Python — don't let it degrade to char matching).
    if isinstance(match_substrings, str):
        match_substrings = (match_substrings,)
    if isinstance(provider_substrings, str):
        provider_substrings = (provider_substrings,)
    # Reject patterns that normalize to nothing: "" is a substring of every
    # name, so an empty pattern (or empty AND-group) would hijack all models.
    for group in _match_groups(match_substrings):
        if not group or any(not sub for sub in group):
            raise ValueError(f"match_substrings for family {name!r} contains an empty pattern or group — it would match every model")
    if any(not sub for sub in _match_substrings(provider_substrings)):
        raise ValueError(f"provider_substrings for family {name!r} contains an empty pattern — it would match every provider")
    _FAMILIES[name] = {
        "match": tuple(match_substrings),
        "provider_match": tuple(provider_substrings),
        "native_path": tuple(native_path) if native_path is not None else None,
        "rungs": {str(k).lower().strip(): deepcopy(v) for k, v in rungs.items()},
    }


def merge_reasoning_config(reasoning: dict[str, Any], explicit: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Deep-merge reasoning kwargs under explicitly-passed config.

    Explicit values win on leaf conflicts — raw kwargs are the caller's escape
    hatch and must stay authoritative. Non-conflicting keys from both sides
    compose (a caller's partial ``extra_body`` merges alongside the translated
    flags instead of clobbering them).

    Args:
        reasoning: Kwargs emitted by :func:`build_reasoning_kwargs`.
        explicit: The caller/settings config assembled by the factory.

    Returns:
        Tuple of (merged config, dotted paths where an explicit value overrode
        a differing reasoning value — for the factory to warn about).
    """
    conflicts: list[str] = []

    def _merge(base: dict[str, Any], override: dict[str, Any], path: str) -> dict[str, Any]:
        merged = dict(base)
        for key, override_value in override.items():
            key_path = f"{path}.{key}" if path else key
            if key in merged and isinstance(merged[key], dict) and isinstance(override_value, dict):
                merged[key] = _merge(merged[key], override_value, key_path)
            else:
                if key in merged and merged[key] != override_value:
                    conflicts.append(key_path)
                merged[key] = override_value
        return merged

    return _merge(deepcopy(reasoning), explicit, ""), conflicts

-------
rges alongside the translated
    flags instead of clobbering them).

    Args:
        reasoning: Kwargs emitted by :func:`build_reasoning_kwargs`.
        explicit: The caller/settings config assembled by the factory.

    Returns:
        Tuple of (merged config, dotted paths where an explicit value overrode
        a differing reasoning value — for the factory to warn about).
    """
    conflicts: list[str] = []

    def _merge(base: dict[str, Any], override: dict[str, Any], path: str) -> dict[str, Any]:
        merged = dict(base)
        for key, override_value in override.items():
            key_path = f"{path}.{key}" if path else key
            if key in merged and isinstance(merged[key], dict) and isinstance(override_value, dict):
                merged[key] = _merge(merged[key], override_value, key_path)
            else:
                if key in merged and merged[key] != override_value:
                    conflicts.append(key_path)
                merged[key] = override_value
        return merged

    return _merge(deepcopy(reasoning), explicit, ""), conflicts

-------
