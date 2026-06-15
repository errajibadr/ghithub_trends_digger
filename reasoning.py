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
- A ``Sources:`` row is a DOCUMENTARY reference a sub-agent actually cited — a
  document, article, runbook, page, or link, with its title and (when available)
  its url. The sub-agent or tool that produced an answer is that answer's
  ORIGIN, never a source — never list one as a ``Sources:`` row. When an answer
  carries no document or link (live records, lookups, computed results), emit NO
  ``Sources:`` block at all.
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
  bare ``[N] Title`` or ``[N] source N`` when the sub-agent returned a url. Only
  forward a ``Sources:`` block when a sub-agent actually shared documentary
  references or urls — those are the only things that ever appear in it.
  Operational or data results (live system records, lookups, query results) are
  NOT documentary sources — report them inline in your answer, never as a
  ``Sources:`` row. If a sub-agent cited a documentary source without a url, keep
  its title as a plain ``[N] Title`` row and never invent or guess a url to fill
  the gap. Do not append a placeholder marker to the row.
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
