packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/prompts/orchestrator_planner_prompt.py
----
"""Orchestrator planner system-prompt builder."""

from __future__ import annotations

from sta_agent_engine.agents.base.prompts.capability_definition import CapabilityDefinition
from sta_agent_engine.agents.base.prompts.prompt_manager import PromptManager

from .orchestrator_grounding import (
    _CLARIFICATION_SECTION,
    _DEEPAGENTS_TOOL_GUIDELINES,
    _FUTURE_CAPABILITIES_SECTION,
    _GENERAL_KNOWLEDGE_SECTION,
    _GROUNDING_SECTION,
    _SUBAGENT_TASKING_SECTION,
    _UNCERTAINTY_SECTION,
)


# The roadmap teaser (`<future_capabilities>`: topology / coding / security
# engines) is off by default — it spends tokens on capabilities the planner
# cannot invoke and risks it offering them. Flip to ``True`` to re-advertise.
_INCLUDE_FUTURE_CAPABILITIES = False


_PLANNER_IDENTITY = """<identity>
You are TWIN, an enterprise IT operations orchestrator. You plan and
coordinate calls to specialist tools and sub-agents to answer user
questions.
</identity>"""

_PLANNER_OBJECTIVE = """<objective>
For each user message:
1. Identify the user's intent.
2. Pick the smallest tool / sub-agent that can answer it.
3. Wait for the result, then return it to the user as a complete,
   self-contained answer — or delegate further if a multi-step task requires
   it. The user sees only your reply, never the sub-agent's output.
4. Stay grounded in tool / sub-agent output — never invent facts.
</objective>"""

_OUTPUT_FORMAT = """
- Your reply is the ONLY thing the user sees: they do NOT see sub-agent
  outputs, tool results, your todos, or your intermediate steps — any general
  note that tool output is visible in real time does not apply here. So every
  reply must be COMPLETE and SELF-SUFFICIENT: restate inline every fact,
  figure, entity name, ID, and source the user needs to act on, and never
  refer to content they can't see ("as shown above", "as the sub-agent
  returned", "see the table") — there is no "above" for the user.
- Because the user only ever sees your reply, never a sub-agent's, every source
  a sub-agent cites must reach the user through you — faithfully and with its
  real url. Don't drop, paraphrase away, or re-label a sub-agent's sources.
- Reply in the user's language.
- Be concise by default; go longer only when the user asks for detail or the
  question genuinely needs it. Conciseness is about your own prose — never trim
  a sub-agent's substance (counts, rows, sources) to save space, and never
  sacrifice the completeness a self-sufficient answer requires.
- Relay sub-agent answers faithfully: preserve their substance — counts,
  figures, entity names, IDs, and codes exactly as reported. Don't recompute or
  round figures, don't relabel entities, don't add details the sub-agent didn't
  provide.
- Keep a sub-agent's formatting when it makes the answer easier to read
  (tables, lists, code blocks) rather than flattening it to prose.
- If a sub-agent reports no result, relay that plainly (see the uncertainty
  rules) — never substitute a fabricated answer.
- Citations: every ``[N]`` marker in your reply body must correspond to a row
  ``[N] [Title](url)`` in a trailing ``Sources:`` block, and every row in the
  block must be referenced by at least one ``[N]`` in the body.
- Simplest path: paste the sub-agent's reply (body + ``Sources:`` block)
  unchanged — the numbering is already consistent.
- If you rewrite or merge replies from multiple sub-agents, renumber the
  body markers contiguously (``[1]``, ``[2]``, ...) and rebuild the
  ``Sources:`` block to match. Never invent a citation; never reference a
  ``[N]`` that has no corresponding row.
- ALWAYS carry the source URL through when a sub-agent provides one. Each
  ``Sources:`` row must be ``[N] [Title](url)`` with the real url — never emit a
  bare ``[N] Title`` or ``[N] source N`` when the sub-agent returned a url. This
  applies to documentary sources; operational or data sources (live system
  records, query results) generally don't carry a url. If a sub-agent cited a
  documentary source without a url, keep its title as a plain ``[N] Title`` row
  and never invent or guess a url to fill the gap. Do not append a placeholder
  marker to the row.
"""


def build_planner_system_prompt(
    *,
    soul: str | None = None,
    tools: list[CapabilityDefinition],
    subagents: list[CapabilityDefinition],
    persona: str | None,
    auth_status: str | None,
) -> str:
    """Build the orchestrator's planner system prompt from bound capabilities.

    This builder no longer emits a ``<capabilities>`` block. The subagent
    roster reaches the model through Deep Agents' own assembled prompt — the
    auto-generated "Available subagent types" list and the ``task`` tool
    description, both built from each subagent's compact description (which now
    carries its corpora ``sources``). Keeping a parallel ``<capabilities>``
    block here only duplicated that, so this prompt covers *posture* (grounding,
    clarification, tasking discipline, output format) and leaves the roster to
    Deep Agents.

    Section ordering is deliberate for prefix caching: the static character +
    behavior block leads (a stable shared prefix across every call and user),
    and per-call dynamic content (``persona``, ``auth_status``) is appended at
    the tail so it never shifts the cached prefix.

    Args:
        tools: Capability metadata for tools actually bound on this graph. Used
            only to decide whether the optional ``<future_capabilities>`` teaser
            is worth rendering.
        subagents: Capability metadata for subagents actually registered. Gates
            the ``<subagent_tasking>`` section (delegation discipline is only
            relevant when the planner can delegate). Corpora advertisement now
            lives in each subagent's compact description, not here.
        persona: Optional per-call persona text; rendered as a trailing
            ``<persona>`` section (dynamic tail), not nested in the lead block.
        auth_status: Optional degraded-mode banner; ``None`` for normal operation.
        soul: Optional static character + role text (``SOUL.md``). When present,
            it leads the prompt as the ``<soul>`` section and subsumes
            ``<identity>``. When ``None`` (file absent/empty/unreadable), the
            builder falls back to the legacy ``<identity>`` constant.

    Returns:
        Fully-assembled system prompt string.
    """
    base = f"<soul>\n{soul.strip()}\n</soul>\n\n{_PLANNER_OBJECTIVE}" if soul else f"{_PLANNER_IDENTITY}\n\n{_PLANNER_OBJECTIVE}"
    pm = PromptManager(base)

    if _INCLUDE_FUTURE_CAPABILITIES and (tools or subagents):
        pm.add_section("future_capabilities", _FUTURE_CAPABILITIES_SECTION, mode="create")

    pm.add_section("grounding", _GROUNDING_SECTION, mode="create")
    pm.add_section("clarification", _CLARIFICATION_SECTION, mode="create")
    pm.add_section("general_knowledge", _GENERAL_KNOWLEDGE_SECTION, mode="create")
    pm.add_section("uncertainty", _UNCERTAINTY_SECTION, mode="create")
    pm.add_section("guidelines", _DEEPAGENTS_TOOL_GUIDELINES, mode="create")
    if subagents:
        # Agent-specific tasking guidance does NOT live here: each subagent's
        # routing posture (e.g. the KA's complementary-source note) rides in its
        # own compact description, which Deep Agents renders only when that
        # subagent is permitted. This section stays agent-agnostic.
        pm.add_section("subagent_tasking", _SUBAGENT_TASKING_SECTION, mode="create")
    pm.add_section("output_format", _OUTPUT_FORMAT, mode="create")

    # Dynamic tail — kept last so the static prefix above stays cache-stable.
    if persona:
        pm.add_section("persona", persona, mode="create")

    if auth_status:
        pm.add_section("auth_status", auth_status, mode="create")

    return pm.build()

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/sources/twin_ka_entries.py
----
"""Knowledge Agent retriever entries for the orchestrator.

Builds the direct-``ElasticRetriever`` entries the Knowledge Agent exposes
when the orchestrator delegates to the ``knowledge_agent`` sub-agent:

- ``general_doc`` — broad company documentation, unscoped. Always present.
- ``twin_project_doc`` — scoped to the twin team's entities/apcodes via
  the anonymized ``TWIN_SCOPE_*`` env config (see ``build_twin_scope``).
  Added **only** when a real filter ceiling is configured (see below).

Both entries use the direct ``ElasticRetriever`` (never the ``elastic_rag``
gateway proxy) so the build-time metadata scope is enforced in-process — the
proxy would bypass the client-side scope ceiling.

``build_twin_ka_entries()`` constructs ``ElasticRetriever`` instances, which can
open network connections. Call it lazily (on the first KA-permitted request),
never at module import time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sta_agent_engine.agents.knowledge_agent import create_elastic_entry

from .twin_scope import build_twin_scope


if TYPE_CHECKING:
    from sta_agent_core import MetadataScope
    from sta_agent_engine.agents.knowledge_agent import RetrieverEntry


logger = logging.getLogger(__name__)


# Planner-facing descriptions. Kept generic — they steer the Knowledge Agent's
# retriever selection and carry no tenant-identifying values (the scope itself
# is supplied out-of-band via TWIN_SCOPE_* env vars).
_GENERAL_DESCRIPTION = (
    "Search broad company documentation: procedures and processes, application and team "
    "documentation, and runbooks. Can also contain infrastructure configuration, architecture "
    "notes, and team / référent information — all as captured in written documentation."
)

_PROJECT_KNOWLEDGE_DESCRIPTION = (
    "Search the twin program's knowledge base — the right corpus for ANY question "
    "about the twin program or application: roadmaps, franchises, initiatives, and "
    "team compositions. Also covers the program's technical knowledge: the STA "
    "agent packages (agentic AI frameworks and design patterns), RAG and LightRAG "
    "internals, the agents built on them, and the internal LangGraph platform "
    "integration and DevX setup for teams onboarding onto OPS Agentic initiatives."
)

_EXPOSED_METADATA_ARGS = ("apcode", "app_name", "entity")


def _scope_has_filter(scope: MetadataScope) -> bool:
    """True when the scope sets at least one hard filter axis.

    Boost axes only soft-rank results — they never restrict the candidate set,
    so a boost-only scope is not a retrieval ceiling and must not be treated as
    one for the scoped project-knowledge entry.
    """
    return bool(scope.entity_filter or scope.apcode_filter or scope.app_name_filter)


def list_twin_ka_corpora() -> list[tuple[str, str]]:
    """Return ``[(name, description), ...]`` for the corpora the Knowledge
    Agent will actually wire on the next ``build_twin_ka_entries()`` call.

    Mirrors the filter-axis conditional in ``build_twin_ka_entries()`` so any
    consumer that needs a *listing* (e.g. the planner capability renderer, to
    advertise the KA's corpus catalog without invoking it) gets the same shape
    the KA planner will see at execution time.

    Pure config read — never constructs ``ElasticRetriever``, never opens a
    network connection. Safe to call at registry / graph build time.
    """
    corpora: list[tuple[str, str]] = [("general_doc", _GENERAL_DESCRIPTION)]
    scope = build_twin_scope()
    if scope is not None and _scope_has_filter(scope):
        corpora.append(("twin_project_doc", _PROJECT_KNOWLEDGE_DESCRIPTION))
    return corpora


# Backward-compatible alias: the twin_router shim (and its tests) import
# ``list_twin_ka_sources`` from this module by name. The orchestrator's own
# call sites use ``list_twin_ka_corpora``; keep the old name pointing at the
# same implementation so the legacy router path is unaffected.
list_twin_ka_sources = list_twin_ka_corpora


def build_twin_ka_entries() -> list[RetrieverEntry]:
    """Build the orchestrator's Knowledge Agent retriever entries.

    Always returns the broad unscoped ``general_doc`` retriever. The
    twin-scoped ``twin_project_doc`` retriever is appended **only** when
    ``build_twin_scope()`` yields a scope with at least one filter axis — a
    missing or boost-only scope would leave it unscoped (searching the whole
    index while looking scoped), so it is omitted fail-closed instead.

    Constructs ``ElasticRetriever`` instances — call lazily, never at import time.
    """
    entries = [
        create_elastic_entry(
            name="general_doc",
            description=_GENERAL_DESCRIPTION,
            default_scope=None,
            expose_metadata_args=list(_EXPOSED_METADATA_ARGS),
            accepts_caller_scope=True,
        )
    ]

    twin_scope = build_twin_scope()
    if twin_scope is not None and _scope_has_filter(twin_scope):
        entries.append(
            create_elastic_entry(
                name="twin_project_doc",
                description=_PROJECT_KNOWLEDGE_DESCRIPTION,
                default_scope=twin_scope,
                expose_metadata_args=list(_EXPOSED_METADATA_ARGS),
            )
        )
    elif twin_scope is not None:
        logger.warning(
            "TWIN_SCOPE_* defines only boost axes — a boost-only scope is not a "
            "retrieval ceiling. Omitting the twin_project_doc retriever; "
            "set at least one TWIN_SCOPE_*_FILTERS axis to enable it."
        )
    else:
        logger.info("No TWIN_SCOPE_* filter axis configured — twin_project_doc retriever omitted; the Knowledge Agent runs with general docs only.")
    return entries

-------

