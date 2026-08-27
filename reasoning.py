.env.example
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

# Alfred planner (and its shared built-in specialist model) reasoning effort.
# Explicit per-run model configuration can override this deployment default.
ALFRED_REASONING_EFFORT=low
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

# --- Lightweight specialist agents (Optional) ---
# Web Scout uses this secret through .env.secrets when public-web search is enabled.
# TAVILY_API_KEY=
# SQL Detective uses an anonymized in-memory demo when no path is configured.
# A configured path is server-owned and is never accepted from an agent tool call.
# SQL_DETECTIVE_DATABASE_PATH=

# --- Alfred remote specialists (POC only) ---
# Default (when unset): experiments/alfred_ag_ui/remote_agents.local.json.
# A missing default file disables remote specialists. Copy
# remote_agents.example.json to the local filename and restart LangGraph after
# editing it. Set this only when the catalog lives at another server-side path.
# ALFRED_REMOTE_AGENTS_CONFIG=

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

# Alfred text-artifact quick actions use the small model tier by default.
# Set these only when quick edits should use a provider or model different from
# the application's normal small-model resolution.
# ALFRED_QUICK_MODEL_PROVIDER=
# ALFRED_QUICK_MODEL=
# Opt in only for strict corporate OpenAI-compatible gateways. This folds
# mid-list system reminders into the leading system prompt and rewrites raw MCP
# tool descriptors with explicit function.parameters. Default false preserves
# Alfred's normal provider behavior.
ALFRED_STRICT_PROVIDER_COMPATIBILITY=false
# Alfred MCP Apps POC only. Export the same loopback web origin in the
# LangGraph and Next.js terminals; no remote or browser-supplied URL is accepted.
# ALFRED_MCP_APPS_BASE_URL=http://127.0.0.1:3010
# Optional vision model (capability axis, not a tier) — used by create_chat_model(..., multimodal=True)
# MULTIMODAL_MODEL=
# Optional model parameters
# TEMPERATURE=0.7
# TOP_P=1.0
# MAX_TOKENS=4096
# Alfred's presentation/DOCX composer emits a complete structured UIPlan and
# therefore needs a larger output allowance than ordinary chat responses.
ALFRED_GENERATIVE_UI_MAX_TOKENS=32000
# Retries and per-attempt timeout for the structured UIPlan composer. These
# bound failed PPTX/DOCX attempts so empty/malformed provider responses recover
# predictably instead of hanging the demo.
ALFRED_GENERATIVE_UI_MAX_RETRIES=1
ALFRED_GENERATIVE_UI_ATTEMPT_TIMEOUT_S=90

# Alfred's Next.js streaming proxy closes an upstream connection that stops
# producing bytes, bounds an individual SSE frame, and eventually releases a
# thread mutation lock even if the upstream never settles. These conservative
# defaults favor recovery during interactive demos; raise them only when a
# known gateway legitimately needs larger events or longer silent intervals.
ALFRED_SSE_IDLE_TIMEOUT_MS=120000
ALFRED_SSE_MAX_FRAME_BYTES=1048576
ALFRED_RESPONSE_LOCK_MAX_LIFETIME_MS=900000
COPILOTKIT_TELEMETRY_DISABLED=true
# Alfred also disables CopilotKit telemetry in code unless this is explicitly true.
ALFRED_COPILOTKIT_TELEMETRY_ENABLED=false

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
ORCHESTRATOR_PROMPT_INJECTION_GUARD_PROVIDER=mistral
ORCHESTRATOR_PROMPT_INJECTION_GUARD_MODEL=mistral-small-2603
# ORCHESTRATOR_PROMPT_INJECTION_GUARD_BASE_URL=
ORCHESTRATOR_PROMPT_INJECTION_GUARD_MAX_TOKENS=256
ORCHESTRATOR_PROMPT_INJECTION_GUARD_TEMPERATURE=0.0
ORCHESTRATOR_PROMPT_INJECTION_GUARD_MAX_RETRIES=2
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

experiments/alfred_ag_ui/alfred_graph.py
----
"""LangGraph entry point for the local Alfred AG-UI proof of concept.

The reusable orchestrator factory remains product-agnostic. This experimental
entry point opts into user memory, user and built-in skills, and approval gates
for filesystem mutations used by the demo UI.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.runtime import Runtime

from experiments.alfred_ag_ui.alfred_copilotkit_middleware import AlfredCopilotKitMiddleware
from experiments.alfred_ag_ui.alfred_demo_agents import ALFRED_SUBAGENT_REGISTRY
from experiments.alfred_ag_ui.alfred_habilitation import get_alfred_habilitation_provider
from experiments.alfred_ag_ui.alfred_model_compatibility import AlfredModelMessageCompatibilityMiddleware
from sta_agent_engine.agents.orchestrator.orchestrator_catalog import create_orchestrator_factory
from sta_agent_engine.models.custom_chat_model import create_chat_model


_FILE_MUTATION_APPROVALS = {
    "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
    "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]},
}


def _create_alfred_chat_model(**kwargs: Any) -> BaseChatModel:
    """Build Alfred's shared planner model with a deployment-owned effort default."""
    default_effort = os.getenv("ALFRED_REASONING_EFFORT", "low").strip() or "low"
    kwargs.setdefault("reasoning_effort", default_effort)
    return create_chat_model(**kwargs)


class AlfredArtifactTerminalMiddleware(AgentMiddleware[AgentState, Any, Any]):
    """Expose a stable terminal node for revision-only artifact checkpoints."""

    @property
    def name(self) -> str:
        """Return the stable node prefix used by the local artifact API."""
        return "alfred_artifact_terminal"

    def after_agent(self, state: AgentState, runtime: Runtime[Any]) -> None:
        """Finish the agent turn without modifying conversation state."""
        del state, runtime


make_alfred = create_orchestrator_factory(
    hab_provider=get_alfred_habilitation_provider(),
    model_factory=_create_alfred_chat_model,
    enable_memory=True,
    enable_skills=True,
    interrupt_on=_FILE_MUTATION_APPROVALS,
    deployment_middleware_factories=(
        AlfredCopilotKitMiddleware,
        AlfredModelMessageCompatibilityMiddleware,
        AlfredArtifactTerminalMiddleware,
    ),
    subagent_registry=ALFRED_SUBAGENT_REGISTRY,
)

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/base/runtime_model_config.py
----
"""Runtime model-config parsing shared by factories and middleware."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_MODEL_CONFIG_SLOTS = ("all", "default")
RUNTIME_MODEL_KEYS = frozenset(
    {
        "model",
        "tier",
        "provider",
        "model_provider",
        "base_url",
        "api_key",
        "provider_base_url",
        "provider_api_key",
        "temperature",
        "model_temperature",
        "top_p",
        "model_top_p",
        "max_tokens",
        "model_max_tokens",
        "frequency_penalty",
        "model_frequency_penalty",
        "reasoning_effort",
        "reasoning_family",
    }
)

ModelConfigCacheKey = tuple[tuple[str, str], ...] | None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return ``value`` as a mapping when it is dict-like."""
    return value if isinstance(value, Mapping) else None


def _first_str(source: Mapping[str, Any], *keys: str) -> str | None:
    """Return the first non-empty string from ``source`` for ``keys``."""
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_number(source: Mapping[str, Any], *keys: str) -> int | float | None:
    """Return the first numeric value from ``source`` for ``keys``."""
    for key in keys:
        value = source.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value
    return None


def select_model_config_source(
    payload: Mapping[str, Any],
    *,
    slots: Sequence[str] = DEFAULT_MODEL_CONFIG_SLOTS,
    include_unscoped: bool = True,
) -> Mapping[str, Any] | None:
    """Pick a model-config block from runtime context or configurable data.

    ``include_unscoped`` gates the two un-slotted sources: the singular
    ``model_config`` block and the flat root-level model keys. Set it ``False``
    to consider only the slot-keyed ``model_configs`` blocks — used by callers
    that must ignore blanket / whole-run overrides (e.g. a pinned agent).
    """
    if include_unscoped:
        model_config = _as_mapping(payload.get("model_config"))
        if model_config:
            return model_config

    model_configs = _as_mapping(payload.get("model_configs"))
    if model_configs:
        for slot in slots:
            slot_config = _as_mapping(model_configs.get(slot))
            if slot_config:
                return slot_config

    if include_unscoped and RUNTIME_MODEL_KEYS.intersection(payload):
        return payload

    return None


def normalize_model_block(source: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one model-config block into ``create_chat_model`` kwargs.

    Accepts both ``ModelConfig``-style keys (``model_temperature``,
    ``model_provider``, …) and direct ``create_chat_model`` kwarg keys
    (``temperature``, ``provider``, …). Only known scalar fields are kept;
    unknown keys are dropped. Returns a (possibly empty) plain dict so callers
    can merge several blocks per-key.
    """
    model_config: dict[str, Any] = {}
    if model := _first_str(source, "model"):
        model_config["model"] = model
    if tier := _first_str(source, "tier"):
        model_config["tier"] = tier
    if provider := _first_str(source, "provider", "model_provider"):
        model_config["provider"] = provider
    if base_url := _first_str(source, "base_url", "provider_base_url"):
        model_config["base_url"] = base_url
    if api_key := _first_str(source, "api_key", "provider_api_key"):
        model_config["api_key"] = api_key
    if reasoning_effort := _first_str(source, "reasoning_effort"):
        model_config["reasoning_effort"] = reasoning_effort
    if reasoning_family := _first_str(source, "reasoning_family"):
        model_config["reasoning_family"] = reasoning_family

    for target_key, source_keys in (
        ("temperature", ("temperature", "model_temperature")),
        ("top_p", ("top_p", "model_top_p")),
        ("frequency_penalty", ("frequency_penalty", "model_frequency_penalty")),
    ):
        value = _first_number(source, *source_keys)
        if value is not None:
            model_config[target_key] = value

    max_tokens = _first_number(source, "max_tokens", "model_max_tokens")
    if max_tokens is not None:
        model_config["max_tokens"] = int(max_tokens)

    return model_config


def merge_model_blocks(*blocks: Mapping[str, Any] | None) -> dict[str, Any]:
    """Per-key merge of model-config blocks, HIGHEST priority first.

    Unlike :func:`select_model_config_source` (which picks one whole block), this
    merges per key: a higher-priority block overrides a lower one key-by-key, and
    a lower block fills only the keys the higher ones left unset. This lets a
    shared slot supply the ``model`` while a more specific slot contributes only
    knobs (``temperature`` / ``max_tokens``). ``None`` / non-mapping blocks are
    skipped; each block is normalized via :func:`normalize_model_block`.
    """
    merged: dict[str, Any] = {}
    for block in reversed(blocks):
        if isinstance(block, Mapping):
            merged.update(normalize_model_block(block))
    return merged


def extract_runtime_model_config(
    payload: Mapping[str, Any],
    *,
    slots: Sequence[str] = DEFAULT_MODEL_CONFIG_SLOTS,
    include_unscoped: bool = True,
) -> dict[str, Any] | None:
    """Normalize runtime model config into ``create_chat_model`` kwargs.

    The runtime surface accepts both ``ModelConfig``-style keys
    (``model_temperature``) and direct ``create_chat_model`` kwarg keys
    (``temperature``). Only known scalar fields are accepted. ``include_unscoped``
    is forwarded to :func:`select_model_config_source` — pass ``False`` to skip
    the singular block and flat root keys and honor slot-keyed blocks only.
    """
    source = select_model_config_source(payload, slots=slots, include_unscoped=include_unscoped)
    if source is None:
        return None

    return normalize_model_block(source) or None


def model_config_cache_key(model_config: Mapping[str, Any] | None) -> ModelConfigCacheKey:
    """Return a cache key fragment for a runtime model config.

    API keys must partition cached graphs because model clients close over
    credentials, but the raw secret must not be stored in cache keys or logs.
    """
    if not model_config:
        return None

    parts: list[tuple[str, str]] = []
    for key in sorted(model_config):
        value = model_config[key]
        if key == "api_key":
            digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            parts.append(("api_key_sha256", digest))
        else:
            parts.append((key, str(value)))
    return tuple(parts)


def redact_model_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a log/error-safe model config copy."""
    redacted = dict(model_config)
    if "api_key" in redacted:
        redacted["api_key"] = "<set>"
    return redacted

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/orchestrator_catalog.py
----
"""Orchestrator deep-agent factory.

``make_orchestrator(config)`` is the 1-arg ``langgraph-api 0.4.x``-compatible
factory. The 2-arg ``ServerRuntime`` form lands once the deployment target
moves to ``langgraph-api >= 0.7.x``.

Per-call flow:

1. Read ``x-uid`` from ``config["configurable"]`` (header forwarded by the
   LangGraph Server).
2. Resolve the user's rights inside this factory call via the habilitation
   resolver — rights are **never** carried on :class:`OrchestratorContext`
   because that schema is caller-supplied and therefore spoofable.
3. Resolve the optional filesystem features into an :class:`ActiveFeatures` —
   the one place the ``authentication × server switches`` matrix is evaluated.
   Backend routes, skills sources, middleware selection, and the graph cache
   partition all read that object instead of recomputing the matrix.
4. Partition the resolved agent set into registry tool keys and subagent keys.
5. Resolve any planner runtime model override from ``config["configurable"]``.
6. Build tools, subagents, and the planner prompt from the filtered registry.
7. Hand off to :func:`deepagents.create_deep_agent`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import FilesystemPermission, create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from sta_agent_engine.agents.base.runtime_model_config import extract_runtime_model_config, redact_model_config


if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore

    from sta_agent_engine.agents.base.prompts.capability_definition import CapabilityDefinition

    from .habilitation.providers import HabilitationProvider

from .backends import (
    SKILLS_BUILTIN_WRITE_DENY_GLOB,
    SKILLS_GENERIC_WRITE_DENY_GLOB,
    SKILLS_HUB_WRITE_DENY_GLOB,
    build_orchestrator_backend,
    validate_uid_format,
)
from .build_context import BuildContext
from .middlewares import (
    FilesystemToolGateMiddleware,
    LiveMemoryMiddleware,
    LoadableSkillsMiddleware,
    OutputStyleMiddleware,
    PromptInjectionGuardSettings,
    ToolBudgetGuardSettings,
    compose_orchestrator_middleware,
)
from .middlewares.tool_budget_enforcement import ORCHESTRATOR_TOOL_BUDGET_GUARD_ENV_PREFIX
from .orchestrator_features import ActiveFeatures
from .orchestrator_harness_profiles import register_orchestrator_harness_profiles
from .orchestrator_resolution import (
    GraphCacheKey,
    build_graph_cache_key,
    parse_orchestrator_request,
    resolve_habilitation_provider,
    resolve_orchestrator_habilitation,
    select_orchestrator_permissions,
)
from .prompts import ORCHESTRATOR_MEMORY_SYSTEM_PROMPT, build_planner_system_prompt
from .registry import SUBAGENT_REGISTRY, TOOL_REGISTRY, SubagentSpec


# Sources passed to ``LiveMemoryMiddleware`` when ``x-uid`` is present.
# Order matters: the user-authored ``AGENTS.md`` anchors the persona; the
# agent-curated ``preferences.md`` appends learned working notes.
_MEMORY_SOURCES: list[str] = ["/memory/AGENTS.md", "/memory/preferences.md"]

#: The standard Hub opt-in: the cross-agent generic group, served from the
#: ``sta skills push`` target repo. Deployments enable the Hub tier with
#: ``create_orchestrator_factory(enable_skills=True,
#: hub_skill_repos=DEFAULT_HUB_SKILL_REPOS)`` — one import, no repo handles to
#: retype. Identifiers may carry a ref (``sta-generic-skills:<commit-hash>``)
#: to pin a group to a commit.
DEFAULT_HUB_SKILL_REPOS: Mapping[str, str] = {"generic": "sta-generic-skills"}


@dataclass(frozen=True)
class OrchestratorSkillsDeployment:
    """The optional-feature flags one served orchestrator graph is built with.

    The single source of truth shared by the module-level skills-enabled
    factory instantiation (:data:`make_orchestrator_skills`) and the
    ``/skills/{graph_id}`` endpoint registry (``server_skills.py``) — both
    consume the same frozen constant, so the endpoint structurally cannot
    disagree with the graph about which skills tiers exist. Defaults mirror
    :func:`create_orchestrator_factory`'s (everything off).
    """

    enable_memory: bool = False
    enable_skills: bool = False
    hub_skill_repos: Mapping[str, str] | None = None


#: Flags of the skills-enabled deployment entry (:data:`make_orchestrator_skills`
#: — served under the ``orchestrator_skills`` graph id in ``langgraph.json``;
#: the plain ``orchestrator`` graph is built WITHOUT skills). Consumed by that
#: instantiation below AND by the ``/skills/{graph_id}`` endpoint registry.
ORCHESTRATOR_SKILLS_DEPLOYMENT = OrchestratorSkillsDeployment(enable_memory=True, enable_skills=True)


@lru_cache(maxsize=1)
def _load_soul() -> str | None:
    """Read the static ``SOUL.md`` character file shipped beside this factory.

    The soul is the planner's fixed character + role, rendered as the first
    ``<soul>`` section of the system prompt. Returns ``None`` when the file is
    absent, empty, or unreadable, in which case the builder falls back to the
    legacy ``<identity>`` constant — so a packaging miss degrades gracefully
    instead of breaking the prompt. Read once per process (build-time, static).
    """
    try:
        text = (Path(__file__).parent / "SOUL.md").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _ensure_kwarg_absent(deep_agent_kwargs: dict[str, Any], key: str, replacement: str) -> None:
    """Reject a deepagents shortcut kwarg that would double-attach a middleware.

    ``memory=[...]`` and ``skills=[...]`` each make ``create_deep_agent``
    auto-attach its stock middleware. Both features are instead wired manually
    with an in-tree subclass, so the shortcut would stack the stock middleware
    on top of ours — double prompt injection plus a duplicate (and, for skills,
    non-refreshable) index. ``raise`` (not ``assert``) so the guard survives
    ``python -O``.

    Args:
        deep_agent_kwargs: The kwargs about to be splatted into
            ``create_deep_agent``.
        key: Shortcut kwarg name that must stay absent (or ``None``).
        replacement: Name of the middleware that owns this feature instead.

    Raises:
        ValueError: The shortcut kwarg carries a value.
    """
    if deep_agent_kwargs.get(key) is not None:
        msg = (
            f"Pass {key} via {replacement} in the middleware list; do NOT pass "
            f"{key}=[...] to create_deep_agent (would attach the stock "
            f"middleware on top of ours)."
        )
        raise ValueError(msg)


logger = logging.getLogger(__name__)


# Runtime slots that OVERRIDE the injected planner model (a caller-selected
# model). The singular ``model_config`` block and flat quick-override keys are
# also treated as overrides by ``select_model_config_source``.
_PLANNER_OVERRIDE_SLOTS: tuple[str, ...] = ("orchestrator", "planner", "all")
# ``"default"`` is a generic fallback that yields to the injected instance.
_PLANNER_FALLBACK_SLOTS: tuple[str, ...] = ("default",)


class PlannerModelResolver:
    """Resolve the planner model before compiling the deepagents graph."""

    def __init__(
        self,
        *,
        model_override: str | BaseChatModel | None,
        model_factory: Callable[..., BaseChatModel],
    ) -> None:
        # An explicitly injected instance (test seam / caller override).
        self._injected_model: str | BaseChatModel | None = model_override
        self._model_factory = model_factory
        # Lazy engine-env fallback, memoized separately so it never shadows the
        # runtime "default" slot on later calls.
        self._cached_env_model: str | BaseChatModel | None = None

    def resolve(self, configurable: Mapping[str, Any]) -> str | BaseChatModel:
        """Return the model object/string passed into ``create_deep_agent``.

        Resolution order (highest priority first):
            1. runtime override slots — ``model_config`` / ``model_configs`` in
               (``orchestrator``, ``planner``, ``all``): a caller-selected model.
            2. ``model_override`` — the build-time injected instance (test seam).
            3. runtime ``model_configs["default"]`` — a generic fallback that no
               longer overrides an injected instance.
            4. ``create_chat_model()`` — engine env fallback (lazily memoized).

        A pre-built ``model_override`` instance is returned untouched (stop /
        knobs on a real instance would have to be set at construction).
        """
        override = extract_runtime_model_config(configurable, slots=_PLANNER_OVERRIDE_SLOTS)
        if override:
            logger.info("Orchestrator planner runtime model override selected: %s", redact_model_config(override))
            return self._model_factory(**override)

        if self._injected_model is not None:
            return self._injected_model

        fallback = extract_runtime_model_config(configurable, slots=_PLANNER_FALLBACK_SLOTS)
        if fallback:
            logger.info("Orchestrator planner runtime 'default' model config selected: %s", redact_model_config(fallback))
            return self._model_factory(**fallback)

        if self._cached_env_model is None:
            self._cached_env_model = self._model_factory()
        return self._cached_env_model


def create_orchestrator_factory(
    *,
    hab_provider: HabilitationProvider | None = None,
    model_override: str | BaseChatModel | None = None,
    model_factory: Callable[..., BaseChatModel] | None = None,
    use_graph_cache: bool = True,
    graph_cache: dict[GraphCacheKey, CompiledStateGraph] | None = None,
    prompt_injection_guard_settings: PromptInjectionGuardSettings | None = None,
    tool_budget_guard_settings: ToolBudgetGuardSettings | None = None,
    store: BaseStore | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    enable_memory: bool = False,
    enable_skills: bool = False,
    hub_skill_repos: Mapping[str, str] | None = None,
    interrupt_on: Mapping[str, bool | Mapping[str, Any]] | None = None,
    deployment_middleware_factories: Sequence[Callable[[], Any]] = (),
    subagent_registry: Mapping[str, SubagentSpec] | None = None,
) -> Callable[[RunnableConfig], Any]:
    """Return a closure-scoped ``make_orchestrator`` factory.

    Args:
        hab_provider: Habilitation provider built once at factory creation.
            ``None`` resolves from environment on first factory call.
        model_override: Optional planner model override for tests.
        model_factory: Optional lazy planner-model factory. ``None`` uses the
            engine's standard ``create_chat_model`` factory. Deployments may
            wrap it to provide deployment-owned defaults while preserving
            explicit runtime model configuration.
        use_graph_cache: When ``True``, reuse compiled graphs keyed by
            bound tool/subagent slices, persona, and model config signature.
        graph_cache: Optional external cache dict for tests. When supplied,
            ``use_graph_cache`` is treated as ``True`` and this dict is used
            instead of a closure-local one.
        prompt_injection_guard_settings: Optional server-owned guardrail
            settings. Explicit settings passed here take precedence over
            environment defaults. Request/runtime context cannot override the
            guard classifier.
        tool_budget_guard_settings: Optional server-owned tool-call budget
            settings. Explicit settings passed here take precedence over
            environment defaults. Request/runtime context cannot raise the
            per-run budget.
        store: Optional ``BaseStore`` for compile-time injection. Under the
            LangGraph Platform, leave this ``None`` — the platform attaches
            its managed Postgres store at invocation. Tests and standalone
            callers (in-process Streamlit, scripts) must pass an instance
            (e.g. ``InMemoryStore()``) so memory writes have somewhere to
            land. If unset and memory is requested at runtime, the first
            backend read raises ``RuntimeError`` from
            ``langgraph.config.get_store``.
        checkpointer: Optional ``BaseCheckpointSaver`` for compile-time
            injection. Same LGP-vs-standalone contract as ``store``.
        enable_memory: Server-owned switch for the per-user memory path.
            ``False`` drops the whole memory path — no ``/memory/*`` Store
            routing, no ``LiveMemoryMiddleware`` — even when a valid ``x-uid``
            is present. Habilitation is unaffected: the uid still resolves
            roles and subagent permissions. Default ``False`` — memory is
            opt-in per deployment.
        enable_skills: Server-owned switch for the agent-skills path.
            Default ``False`` — a full off-switch: no ``SkillsMiddleware`` and
            no skills routes, so the compiled graph is byte-identical to the
            pre-skills design and an upgrade is a no-op for callers that don't
            opt in. ``True`` attaches ``SkillsMiddleware`` and mounts the
            read-only builtin skills bank for every request (anonymous
            included — skills are not habilitation-gated), plus the per-user
            ``/skills/user/*`` route for authenticated callers — which also
            keeps the filesystem toolset bound so the planner can read and
            author skills without memory being on. Skills are independent of
            ``enable_memory`` — either can be toggled alone.
        hub_skill_repos: The ONE Hub knob: map of skill-group name to Context
            Hub repo identifier (e.g. :data:`DEFAULT_HUB_SKILL_REPOS`,
            ``{"generic": "sta-generic-skills"}``). ``None``/empty (the
            default) mounts no Hub tier at all — the compiled graph is
            byte-identical to the hub-less design. A non-empty mapping mounts
            one read-only, TTL-cached route per group at
            ``/skills/hub/<group>/`` — active only while ``enable_skills`` is
            also on (the master skills switch). Identifiers may carry a
            ``:<commit-hash>`` ref to pin a group.
        interrupt_on: Optional server-owned Deep Agents HITL policy. Runtime
            config cannot install or relax this mapping.
        deployment_middleware_factories: Server-owned, zero-argument adapter
            factories appended to the reusable orchestrator middleware. Each
            compilation receives fresh middleware instances; callers cannot
            inject factories through runtime config.
        subagent_registry: Optional server-owned specialist catalog for this
            deployment. The mapping is snapshotted when the factory is
            created, so runtime callers can only narrow its entries through
            ``selected_agents`` and can never restore a specialist omitted by
            the deployment. ``None`` uses the reusable global catalog.

    Returns:
        An async factory with the 1-arg ``langgraph-api 0.4.x`` signature.
    """
    _deployment_middleware_factories = tuple(deployment_middleware_factories)
    _subagent_registry = dict(SUBAGENT_REGISTRY if subagent_registry is None else subagent_registry)
    # External caches may be shared by several deployment factories. Registry
    # keys alone are insufficient because two deployments can install
    # different builders under the same specialist name. Object identity is a
    # safe process-local signature for this in-memory cache: frozen specs are
    # snapshotted above and remain stable for the factory lifetime.
    _subagent_registry_cache_key = tuple(sorted((key, id(spec)) for key, spec in _subagent_registry.items()))
    for index, factory in enumerate(_deployment_middleware_factories):
        if not callable(factory):
            msg = f"deployment_middleware_factories[{index}] must be callable"
            raise TypeError(msg)

    _resolved_habilitation_provider: HabilitationProvider | None = hab_provider
    _hub_skill_repos: dict[str, str] = dict(hub_skill_repos or {})
    _graph_cache: dict[GraphCacheKey, CompiledStateGraph] = graph_cache if graph_cache is not None else {}
    if graph_cache is not None:
        use_graph_cache = True

    from sta_agent_engine.models.custom_chat_model import create_chat_model

    planner_model_resolver = PlannerModelResolver(
        model_override=model_override,
        model_factory=model_factory or create_chat_model,
    )
    _prompt_guard_settings = prompt_injection_guard_settings or PromptInjectionGuardSettings()
    _tool_budget_settings = tool_budget_guard_settings or ToolBudgetGuardSettings(_env_prefix=ORCHESTRATOR_TOOL_BUDGET_GUARD_ENV_PREFIX)
    _interrupt_on = dict(interrupt_on) if interrupt_on is not None else None

    async def make_orchestrator(config: RunnableConfig) -> CompiledStateGraph:
        nonlocal _resolved_habilitation_provider
        request = parse_orchestrator_request(config)

        # Read ``x-uid`` from the RAW configurable mapping (pre-normalization)
        # so the trust boundary is explicit: present-and-non-empty string is
        # authenticated, anything else is anonymous. Cannot use ``request.uid``
        # because that field normalizes missing/invalid values to a sentinel
        # string that could collide with a legitimate (but caller-chosen) uid.
        #
        # A bad-format uid (chars outside the allowlist, or oversized) is a
        # gateway/caller bug — the value still arrived over a trusted header,
        # but it cannot become a valid Store namespace component. Falling to
        # the anonymous path keeps the user's request alive (StateBackend,
        # no /memory/* routing) instead of failing late inside the Store
        # layer on first memory op. The catalog logs a warning so the gateway
        # misconfig surfaces to operators; ``resolve_memory_namespace`` keeps
        # raising ``ValueError`` as a defense-in-depth backstop for any
        # future path that bypasses this check.
        raw_uid = request.configurable.get("x-uid")
        is_authenticated = isinstance(raw_uid, str) and bool(raw_uid) and validate_uid_format(raw_uid)
        if isinstance(raw_uid, str) and raw_uid and not is_authenticated:
            # Log only the length — a malformed value could be an
            # attacker-controlled payload from a misconfigured gateway, so
            # do not emit it verbatim into logs.
            logger.warning(
                "Rejected malformed x-uid (length=%d); falling to anonymous path",
                len(raw_uid),
            )
        # Evaluate the authentication x server-switch matrix ONCE. Every later
        # decision — which routes the backend mounts, which skills sources the
        # middleware loads, which middlewares are appended, how the graph cache
        # partitions — reads this object rather than re-deriving an ``and`` of
        # its own, so those decisions cannot drift apart. Habilitation is
        # untouched by it: rights read request.uid, not these flags.
        features = ActiveFeatures.resolve(
            is_authenticated=is_authenticated,
            enable_memory=enable_memory,
            enable_skills=enable_skills,
            hub_skill_groups=tuple(_hub_skill_repos),
        )
        has_memory, has_user_skills = features.cache_bits()

        if _resolved_habilitation_provider is None:
            _resolved_habilitation_provider = resolve_habilitation_provider()

        t0 = time.perf_counter()
        habilitation = await resolve_orchestrator_habilitation(
            uid=request.uid,
            provider=_resolved_habilitation_provider,
            request_id=request.request_id,
        )
        permissions = select_orchestrator_permissions(
            habilitation.permitted_keys,
            selected_agents=request.selected_agents,
            subagent_registry=_subagent_registry,
        )
        cache_key = build_graph_cache_key(
            permissions=permissions,
            persona=request.persona,
            model_cache_key=request.model_cache_key,
            has_memory=has_memory,
            has_user_skills=has_user_skills,
            subagent_registry_cache_key=_subagent_registry_cache_key,
        )
        if use_graph_cache and not habilitation.degraded and cache_key in _graph_cache:
            logger.debug(
                "Orchestrator graph cache hit: tools=%s subagents=%s persona=%s has_memory=%s has_user_skills=%s",
                sorted(permissions.tools),
                sorted(permissions.subagents),
                request.persona,
                has_memory,
                has_user_skills,
            )
            return _graph_cache[cache_key]

        planner_model = planner_model_resolver.resolve(request.configurable)

        ctx = BuildContext(persona=request.persona)

        tools = [TOOL_REGISTRY[key].factory(ctx, TOOL_REGISTRY[key].capability) for key in sorted(permissions.tools) if key in TOOL_REGISTRY]
        tool_caps = [TOOL_REGISTRY[key].capability for key in sorted(permissions.tools) if key in TOOL_REGISTRY]

        # Each permitted spec builds its own subagent from the request context.
        # A spec's build function owns its dependencies (retriever catalogs,
        # model factories) and its soft-landing middleware, and returns both the
        # wrapped subagent and the capability the planner advertises — already
        # enriched (e.g. the KA capability carries its corpora ``sources`` so the
        # planner block self-describes, gated on the KA being permitted).
        built = {key: _subagent_registry[key].build(ctx) for key in sorted(permissions.subagents) if key in _subagent_registry}
        subagents = [b.subagent for b in built.values()] or None
        sub_caps: list[CapabilityDefinition] = [b.capability for b in built.values()]

        planner_prompt = build_planner_system_prompt(
            tools=tool_caps,
            subagents=sub_caps,
            persona=request.persona,
            auth_status=habilitation.auth_status,
            soul=_load_soul(),
        )

        middleware = compose_orchestrator_middleware(
            planner_model=planner_model,
            prompt_injection_guard_settings=_prompt_guard_settings,
            tool_budget_guard_settings=_tool_budget_settings,
        )
        output_style_middleware = next(item for item in middleware if isinstance(item, OutputStyleMiddleware))
        middleware.remove(output_style_middleware)

        # Inject the state-channel bridge of each permitted subagent. A bridge
        # widens the orchestrator graph with a subagent's exchange channels
        # (e.g. the KA's ka_metadata_scope / ka_sources) only when the caller can
        # actually reach that subagent — a non-permitted subagent never
        # declares its channels. Dedup by class so subagents sharing a bridge
        # add it once. Driven by permitted keys, so the graph cache (keyed on
        # the permitted set) stays coherent.
        #
        # Bridges are PREPENDED (before the prompt-injection guard), not
        # appended. A bridge hooks before_agent for per-run state upkeep — the
        # KA bridge resets its accumulating ka_sources output channel and, when
        # the caller seeded a document selection (ka_metadata_scope.doc_ids),
        # appends the planner-facing <system_reminder> about it. The guard also
        # hooks before_agent and returns jump_to="end" on a blocked turn, which
        # skips every later before_agent hook — so a reset sitting after the
        # guard would never run on a refused turn, leaking the previous turn's
        # accumulated sources into the blocked turn's output. Running ahead of
        # the screen is safe: a bridge reads only caller-seeded scope channels
        # (never user message content) and cannot jump, so the guard remains the
        # first hook that observes user input. Accepted trade-off: a refused
        # turn may carry the scope reminder in its checkpointed history.
        seen_bridges: set[type] = set()
        bridge_middleware: list[Any] = []
        for sub_key in sorted(permissions.subagents):
            spec = _subagent_registry.get(sub_key)
            if spec is None or spec.bridge is None or spec.bridge in seen_bridges:
                continue
            seen_bridges.add(spec.bridge)
            bridge_middleware.append(spec.bridge())
        middleware[0:0] = bridge_middleware

        # Deployment adapters are a factory-construction concern, never
        # caller-controlled runtime input. Build a fresh instance for this
        # graph compilation after the security middleware has been composed.
        middleware.extend(factory() for factory in _deployment_middleware_factories)

        # Build a backend INSTANCE per graph-cache class. The backend mounts
        # one route per enabled capability: /memory/* (authenticated, per-uid
        # Store), /skills/builtin/ + /skills/generic/ (read-only packaged
        # banks, every request), /skills/user/* (authenticated, per-uid
        # Store). With no route it is a bare StateBackend. The SAME instance
        # is shared by FilesystemMiddleware, LiveMemoryMiddleware, and
        # SkillsMiddleware so every file op resolves through one call stack
        # and one Store namespace per family.
        backend = build_orchestrator_backend(
            has_memory=features.memory,
            has_builtin_skills=features.skills_builtin,
            has_generic_skills=features.skills_builtin,
            has_user_skills=features.skills_user,
            hub_skill_repos={group: _hub_skill_repos[group] for group in features.skills_hub} or None,
        )

        # Skills sources for LoadableSkillsMiddleware — derived from the same
        # feature flags that mounted the routes above, so a mounted route always
        # has a matching source and vice versa. None when no skills route is
        # mounted → no skills middleware is attached at all.
        skills_sources = features.skills_sources()

        deep_agent_kwargs: dict[str, Any] = {
            "model": planner_model,
            "tools": tools,
            "system_prompt": planner_prompt,
            "middleware": middleware,
            "subagents": subagents,
            "store": store,
            "checkpointer": checkpointer,
            "backend": backend,
        }
        # The packaged skills banks ship inside the wheel read-only. Enforcement is
        # the mounts themselves: build_orchestrator_backend routes /skills/builtin/
        # and /skills/generic/ to mutation-refusing backends, so write_file /
        # edit_file are rejected for every path shape (including dot-prefixed
        # components a glob cannot match). This write-deny permission rule is a UX
        # layer ON TOP: for the common non-dot path it turns the refusal into a
        # clean tool-level "permission denied" message before the call reaches the
        # backend. Reads stay allowed, and /memory/** and /skills/user/** stay
        # writable — the rule names only the bank globs. Subagents inherit it. The
        # check runs on the full request path before route stripping, and only at
        # the tool boundary. Declared only when the routes are mounted so the
        # compiled graph is unchanged for cache classes without the banks.
        if features.skills_builtin:
            deny_globs = [SKILLS_BUILTIN_WRITE_DENY_GLOB, SKILLS_GENERIC_WRITE_DENY_GLOB]
            if features.skills_hub:
                deny_globs.append(SKILLS_HUB_WRITE_DENY_GLOB)
            deep_agent_kwargs["permissions"] = [FilesystemPermission(operations=["write"], paths=deny_globs, mode="deny")]
        if _interrupt_on is not None:
            deep_agent_kwargs["interrupt_on"] = _interrupt_on
        # Append after compose so both loaders run after the prompt-injection
        # guard, and their system-prompt fragments land after the stable base
        # prompt (memory then skills), preserving its reusable prefix.
        if features.memory:
            middleware.append(
                LiveMemoryMiddleware(
                    backend=backend,
                    sources=_MEMORY_SOURCES,
                    system_prompt=ORCHESTRATOR_MEMORY_SYSTEM_PROMPT,
                    add_cache_control=False,
                )
            )
        if not features.any_user_writable:
            # Nothing durable is mounted for this request: every file op would
            # land in the ephemeral per-thread StateBackend, so file
            # manipulation serves no purpose. FilesystemMiddleware itself is
            # required deepagents scaffolding (tool-result eviction, permission
            # enforcement) and cannot be excluded, so hide its tools from the
            # model instead. The gate keys on writability, not on memory: with
            # memory off but user skills mounted, the model keeps the toolset so
            # it can author and read skills.
            #
            # The read-only builtin bank alone does not open the gate. An
            # anonymous request therefore sees the skills index in its prompt
            # but cannot read a SKILL.md body — progressive disclosure needs an
            # identified caller.
            middleware.append(FilesystemToolGateMiddleware())

        # Skills are wired manually (not via create_deep_agent's skills=) so we
        # can attach the in-tree subclass in place of the stock middleware.
        # LoadableSkillsMiddleware adds the /skill-reload refresh command.
        if skills_sources:
            middleware.append(
                LoadableSkillsMiddleware(
                    backend=backend,
                    sources=skills_sources,
                )
            )

        # Output style is run-scoped rather than graph identity. Keep it as the
        # final caller middleware so its trusted block is appended after the
        # base harness, deployment, memory, and skills system sections. The
        # resulting provider request has one leading SystemMessage; custom text
        # itself remains in the latest HumanMessage and was screened by the
        # prompt-injection guard above.
        middleware.append(output_style_middleware)

        _ensure_kwarg_absent(deep_agent_kwargs, "memory", "LiveMemoryMiddleware")
        _ensure_kwarg_absent(deep_agent_kwargs, "skills", "LoadableSkillsMiddleware")

        # Disable the auto-injected general-purpose subagent only when this
        # factory is about to compile an orchestrator planner graph. The Deep
        # Agents harness-profile registry is process-global, so merely importing
        # this package or creating a factory must not affect unrelated graphs.
        register_orchestrator_harness_profiles()
        graph = create_deep_agent(**deep_agent_kwargs)

        if use_graph_cache and not habilitation.degraded:
            t_build = (time.perf_counter() - t0) * 1000
            logger.info(
                "Orchestrator graph cache miss — compiled in %.1fms for tools=%s subagents=%s persona=%s has_memory=%s has_user_skills=%s",
                t_build,
                sorted(permissions.tools),
                sorted(permissions.subagents),
                request.persona,
                has_memory,
                has_user_skills,
            )
            _graph_cache[cache_key] = graph

        return graph

    return make_orchestrator


make_orchestrator = create_orchestrator_factory()

#: Skills-enabled variant: memory (default-on) + the agent-skills path — builtin
#: bank for every request, per-user skills + `/skill-reload` / `/<skill_name>`
#: commands for authenticated callers. Separate name so deployments (and the
#: Streamlit graph registry) opt in per entry while `make_orchestrator` stays
#: byte-identical to the pre-skills graph. Built from the frozen deployment
#: constant the `/skills/{graph_id}` endpoint registry also reads, so the two
#: cannot disagree about which skills tiers exist.
make_orchestrator_skills = create_orchestrator_factory(
    enable_memory=ORCHESTRATOR_SKILLS_DEPLOYMENT.enable_memory,
    enable_skills=ORCHESTRATOR_SKILLS_DEPLOYMENT.enable_skills,
    hub_skill_repos=ORCHESTRATOR_SKILLS_DEPLOYMENT.hub_skill_repos,
)

-------

packages/sta_agent_engine/src/sta_agent_engine/models/reasoning.py
----
"""Declarative reasoning-effort configuration for chat models.

Model families expose incompatible knobs for controlling reasoning/thinking:
Mistral takes a top-level ``reasoning_effort`` string, Nemotron wants booleans
nested under ``extra_body.chat_template_kwargs``, Qwen adds a token budget.
This module maps one normalized effort vocabulary (``off`` / ``low`` /
``medium`` / ``high``) onto the literal request kwargs each family honors, so
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
    # Qwen3.8-2.4T requires thinking on every request, so it deliberately has
    # no off rung. Its strongest native level is xhigh; high remains the
    # portable library alias for that level.
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
    # Other Qwen3.8 models expose graded native effort and a separate hard
    # thinking switch for self-hosted vLLM/SGLang deployments.
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
    # Qwen3.x: thinking on by default; the model card documents only the binary
    # chat_template_kwargs.enable_thinking switch (Qwen/Qwen3.6-27B). Graded
    # thinking budgets are a serving-stack feature (vLLM `thinking_token_budget`,
    # version-dependent) — deliberately NOT baked in; gateways that support them
    # can re-register this family with budget rungs (see docs/consuming/reasoning.md).
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
    provider-specific slug conventions collapse to one form: ``qwen3.6``,
    ``qwen3-6``, ``Qwen/Qwen3.6-32B-Instruct``, and ``qwen3:32b`` all contain
    the normalized pattern ``qwen36``/``qwen3``.
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
    ``qwen3.6``, ``qwen3-6``, and ``Qwen/Qwen3.6-32B`` are one family.
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

