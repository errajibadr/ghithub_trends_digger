packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/middlewares/knowledge_bridge.py
----
"""Knowledge Agent state bridge — declares the KA's shared channels.

The orchestrator delegates to the Knowledge Agent through the deepagents
``task`` tool, which exchanges data with a subagent only through state keys that
BOTH sides declare. The orchestrator's deep-agent state is a fixed ``AgentState``
TypedDict; the supported way to add channels is a middleware that sets
``state_schema``. This bridge does exactly that — it declares the two shared
channels so they exist on the compiled orchestrator graph and are carried into
and out of the KA subagent by ``task``:

- ``ka_metadata_scope`` (input) — an optional FILTER-ONLY metadata scope the
  orchestrator supplies before delegating; the KA hard-filters retrieval to it.
- ``ka_sources`` (output) — the grounding sources the KA surfaces back, as
  minimal JSON-safe dicts a frontend references panel can render.

The channel definitions live in
:mod:`sta_agent_engine.agents.knowledge_agent.knowledge_bridge_channels` so this
bridge and the KA state schemas reference one source of truth and cannot drift
on the channel name (the shared name *is* the propagation contract).

The bridge's ``before_agent`` hook does two things:

1. **Reset ``ka_sources`` (and the announce cursor)** — ``ka_sources`` carries
   an accumulating reducer (the planner can delegate to the KA several times in
   one run), so a checkpointed thread would otherwise grow the sources list
   across conversation turns. ``before_agent`` clears it with
   ``Overwrite(value=[])`` — a bare ``[]`` is a no-op under an accumulate
   reducer — and resets the ``ka_sources_announced`` cursor (below) to ``0`` in
   the same per-run update. ``ka_metadata_scope`` needs no reset: it is an
   ``UntrackedValue`` channel, never checkpointed, so it is fresh every run.
2. **Surface the document selection to the planner** — the scope flows to the
   KA silently through state, so without a planner-visible signal the model may
   answer from general knowledge instead of delegating. When the caller-seeded
   scope carries ``doc_ids``, the hook appends an immutable
   ``<system_reminder>`` message ("the user pre-selected these documents —
   forward the search to ``knowledge_agent``"). The injection follows the
   prefix-cache-safe pattern of ``TimeAwareMiddleware``: appended at the tail,
   frozen by a deterministic id (``sta-ka-scope-reminder::turn-{n}::{digest}``)
   so ``add_messages`` never duplicates it on an interrupt resume or retry. A
   turn whose scope no longer carries ``doc_ids`` on a previously-scoped thread
   appends a one-line cleared note instead (once), so a stale selection message
   cannot mislead later turns.

The bridge's ``before_model`` hook is the **canonical-sources announcer**. Each
KA ``task`` returns its own sources numbered ``[1..k]``; the parent's
``merge_ka_sources`` reducer concatenates every delegation's block into one
position-stable ``ka_sources`` list, so a later call's numbers need offsetting.
Computing that offset per-``task`` is *unsafe under parallel delegation*:
sibling ``task`` calls in one super-step all observe the same pre-merge state
snapshot, so each would number from the same offset and collide. Instead the
announcer runs in ``before_model`` — the first hook *after* the tool super-step
merges — and numbers from the **post-merge** list itself: it announces the
sources added since the last model call (``ka_sources[announced:]``) numbered
``[announced+1 …]``, which are exactly the rows the downstream panel renders.
A per-run ``ka_sources_announced`` cursor tracks how many rows are already
announced; a deterministic note id (``sta-ka-sources::{offset}::{digest}``)
makes a retry/resume re-entry idempotent. The note is a ``SystemMessage`` (not
an ``AIMessage``, which Anthropic would treat as a prefill that fires the
planner's ``Sources:`` decoder stop), appended at the tail so the static prompt
prefix stays cache-stable. The answer ``ToolMessage`` is never touched — the
announcer only *adds* a message (citation fidelity; see the orchestrator
``AGENTS.md`` § Pitfalls #5).

This bridge is registered on the ``knowledge_agent`` ``SubagentSpec.bridge`` and
injected into the orchestrator stack only when the caller is permitted to reach
the KA — a non-KA caller never declares these channels. The bridge sits ahead
of the prompt-injection guard (the ``ka_sources`` reset must run even on a
refused turn), which means a refused turn may also carry the scope reminder in
its checkpointed history — an accepted trade-off: the reminder reads only the
caller-seeded scope channel (never user message content) and cannot jump, so
the guard remains the first hook that observes user input.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from sta_agent_engine.agents.knowledge_agent.knowledge_bridge_channels import (
    KnowledgeBridgeChannels,
    read_ka_metadata_scope,
)

from .subagent_state_bridge import SubagentStateBridge


# additional_kwargs markers so scope reminders are identifiable in history
# regardless of role (mirrors the TimeAwareMiddleware marker convention).
_SCOPE_REMINDER_MARKER = "__sta_ka_scope_reminder__"
_SCOPE_REMINDER_BUCKET = "__sta_ka_scope_bucket__"
_REMINDER_ID_PREFIX = "sta-ka-scope-reminder"
_CLEARED_DIGEST = "cleared"

#: Cap on doc ids rendered into the reminder body — the full selection still
#: applies as a filter; the reminder only needs to convey that it exists.
_MAX_RENDERED_DOC_IDS = 20

# Markers / id prefix for the canonical-sources note. Mirrors the scope-reminder
# deterministic-id convention so a retry/resume that replays the same delegation
# re-mints the same id and ``add_messages`` never duplicates it.
_SOURCES_NOTE_MARKER = "__sta_ka_sources_note__"
_SOURCES_NOTE_ID_PREFIX = "sta-ka-sources"

#: Private per-run cursor: how many ``ka_sources`` rows the ``before_model``
#: announcer has already surfaced. ``before_agent`` resets it to 0 alongside the
#: ``ka_sources`` reset; it is single-writer (only ``before_model`` writes it,
#: in the model super-step) so a plain ``LastValue`` is safe. ``PrivateStateAttr``
#: hides it from the orchestrator's public input/output schema.
KA_SOURCES_ANNOUNCED_KEY = "ka_sources_announced"


class KnowledgeBridgeState(AgentState, KnowledgeBridgeChannels):
    """Orchestrator agent state widened with the KA bridge channels.

    Adds one orchestrator-internal channel beyond the two shared bridge
    channels: ``ka_sources_announced``, the announce cursor. It is declared on
    this state only (NOT on :class:`KnowledgeBridgeChannels`) so it never
    crosses the ``task`` boundary into the KA subagent.
    """

    ka_sources_announced: NotRequired[Annotated[int, PrivateStateAttr]]


class KnowledgeBridgeMiddleware(SubagentStateBridge):
    """Declare the Knowledge Agent bridge channels on the orchestrator graph.

    Schema-widening plus two per-run ``before_agent`` concerns: reset the
    accumulating ``ka_sources`` channel, and surface a caller-seeded document
    selection (``ka_metadata_scope.doc_ids``) to the planner as an immutable
    ``<system_reminder>`` message (see module docstring for the staleness and
    prefix-cache contract).

    It also carries one ``before_model`` concern — the **canonical-sources
    announcer**. Before each planner generation it appends an immutable
    ``<knowledge_sources>`` note for the sources added since the last model call
    (``ka_sources[announced:]``), numbered with the exact ``[N]`` the user will
    see, so the planner cites numbers it can read instead of doing multi-call
    offset arithmetic. Running in ``before_model`` (after the tool super-step
    merges) makes the numbering correct even when the planner fans out parallel
    ``task`` delegations in one turn — sibling calls share a pre-merge snapshot,
    so a per-call announcer would number them from the same offset and collide.
    No ``ToolMessage`` is ever rewritten — the announcer only adds a message
    (citation fidelity; see the orchestrator ``AGENTS.md`` § Pitfalls #5).
    """

    state_schema = KnowledgeBridgeState

    #: KA retriever entry the document filter actually applies to (the only
    #: entry with ``accepts_caller_scope=True``) — named in the reminder so the
    #: planner steers the KA toward the corpus where the selection has effect.
    general_corpus_name: str = "general_doc"

    def before_agent(self, state: KnowledgeBridgeState, runtime: Runtime[Any]) -> dict[str, Any]:  # noqa: ARG002
        """Reset ``ka_sources`` and inject the document-selection reminder (sync path)."""
        return self._compute_update(state)

    async def abefore_agent(self, state: KnowledgeBridgeState, runtime: Runtime[Any]) -> dict[str, Any]:  # noqa: ARG002
        """Async counterpart of :meth:`before_agent`."""
        return self._compute_update(state)

    # ------------------------------------------------------------------ canonical-sources announcer

    def before_model(self, state: KnowledgeBridgeState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Announce sources accumulated since the last planner call (sync path)."""
        return self._announce_sources(state)

    async def abefore_model(self, state: KnowledgeBridgeState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Async counterpart of :meth:`before_model`."""
        return self._announce_sources(state)

    def _announce_sources(self, state: KnowledgeBridgeState) -> dict[str, Any] | None:
        """Append a canonical ``<knowledge_sources>`` note for newly-merged sources.

        Reads the **post-merge** ``ka_sources`` list and the per-run
        ``ka_sources_announced`` cursor: the rows in ``ka_sources[announced:]``
        are this turn's freshly-surfaced sources (whether one ``task`` ran or
        several fanned out in parallel — the reducer has already concatenated
        them in panel order). It numbers them ``[announced+1 …]`` — exactly the
        rows the downstream references panel renders — appends an immutable note,
        and advances the cursor.

        Returns ``None`` (no update) when nothing new has merged since the last
        model call — the first planner call (empty channel), an answer-now turn,
        or a retry where the cursor already covers every row.
        """
        sources: list[dict[str, Any]] = list(state.get("ka_sources") or [])
        total = len(sources)
        announced = state.get("ka_sources_announced") or 0
        if total <= announced:
            return None
        new_sources = sources[announced:total]
        note = self._build_sources_note(new_sources, announced)
        return {"messages": [note], KA_SOURCES_ANNOUNCED_KEY: total}

    def _build_sources_note(self, new_sources: list[dict[str, Any]], offset: int) -> AnyMessage:
        """Render newly-surfaced sources as a deterministic ``<knowledge_sources>`` note.

        Each source is numbered ``[offset + i]`` (1-based from the announce
        cursor), so the numbers match the row each source occupies in the
        accumulated ``ka_sources`` list — i.e. exactly what the downstream
        references panel renders. The id is frozen by ``(offset, content)`` so a
        replayed turn re-mints the same id and is de-duplicated by
        ``add_messages``.
        """
        lines = []
        for number, source in enumerate(new_sources, start=offset + 1):
            title = (source.get("title") or "").strip() or "Untitled source"
            url = (source.get("url") or "").strip()
            lines.append(f"[{number}] {title}" + (f" — {url}" if url else ""))
        body = (
            "These sources came from the knowledge sub-agent and will be shown to the user "
            "with exactly these numbers. Cite a knowledge fact by appending the bare marker "
            "[N] shown here — do not repeat these titles or urls in your reply, do not write "
            "your own sources list."
            "\n\n" + "\n".join(lines)
        )
        bucket = f"{offset}::{self._sources_digest(new_sources, offset)}"
        return SystemMessage(
            content=f"<knowledge_sources>\n{body}\n</knowledge_sources>",
            id=f"{_SOURCES_NOTE_ID_PREFIX}::{bucket}",
            additional_kwargs={_SOURCES_NOTE_MARKER: True},
        )

    @staticmethod
    def _sources_digest(new_sources: list[dict[str, Any]], offset: int) -> str:
        """Order-sensitive content hash over the rendered sources at this offset."""
        canonical = f"{offset}\x1f" + "\x1f".join(f"{s.get('title', '')}|{s.get('url', '')}" for s in new_sources)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    # ------------------------------------------------------------------ update assembly

    def _compute_update(self, state: KnowledgeBridgeState) -> dict[str, Any]:
        """Build the per-run state update: the sources reset (channel + announce
        cursor) plus, when a caller-seeded document selection is present (or was
        just cleared), the planner-facing reminder message."""
        update: dict[str, Any] = {"ka_sources": Overwrite(value=[]), KA_SOURCES_ANNOUNCED_KEY: 0}
        reminder = self._scope_reminder(state)
        if reminder is not None:
            update["messages"] = [reminder]
        return update

    def _scope_reminder(self, state: KnowledgeBridgeState) -> AnyMessage | None:
        """Return this turn's selection reminder (or cleared note), if due.

        Deterministic-id contract: same turn + same selection → same id → the
        history scan (and ``add_messages``' replace-on-same-id semantics) make
        re-entry idempotent; a new turn or a changed selection mints a new id,
        appending a fresh reminder at the tail of the message list.
        """
        messages: Sequence[AnyMessage] = state.get("messages") or []
        if not messages:
            return None
        scope = read_ka_metadata_scope(state.get("ka_metadata_scope"))
        doc_ids = scope.get("doc_ids") or []
        turn = sum(1 for m in messages if getattr(m, "type", None) == "human")

        if not doc_ids:
            return self._cleared_note_if_due(messages, turn)

        bucket = f"turn-{turn}::{self._selection_digest(doc_ids)}"
        if self._bucket_present(messages, bucket):
            return None
        return self._build_reminder(bucket, self._render_selection(doc_ids))

    def _cleared_note_if_due(self, messages: Sequence[AnyMessage], turn: int) -> AnyMessage | None:
        """Append a one-line cleared note when a previously-scoped thread loses
        its selection — but only once (the most recent reminder must not already
        be a cleared note)."""
        last_bucket = self._last_reminder_bucket(messages)
        if last_bucket is None or last_bucket.endswith(f"::{_CLEARED_DIGEST}"):
            return None
        bucket = f"turn-{turn}::{_CLEARED_DIGEST}"
        if self._bucket_present(messages, bucket):
            return None
        body = (
            "The user's previous document selection no longer applies — the current "
            "request carries no pre-selected documents. `knowledge_agent` searches "
            "are unscoped again."
        )
        return self._build_reminder(bucket, body)

    # ------------------------------------------------------------------ rendering

    def _render_selection(self, doc_ids: list[str]) -> str:
        shown = doc_ids[:_MAX_RENDERED_DOC_IDS]
        rendered = ", ".join(shown)
        overflow = len(doc_ids) - len(shown)
        if overflow > 0:
            rendered += f", … (+{overflow} more)"
        return (
            f"The user has pre-selected documents for the current request (doc_ids: {rendered}). "
            "This selection is applied automatically as a hard retrieval filter on "
            "`knowledge_agent` searches. If the user's query may need information from these "
            "documents, forward the search to the `knowledge_agent` subagent and instruct it to "
            f"query the general documentation corpus (`{self.general_corpus_name}`), where the "
            "document filter applies — do not answer from general knowledge instead."
        )

    def _build_reminder(self, bucket: str, body: str) -> AnyMessage:
        return SystemMessage(
            content=f"<system_reminder>\n{body}\n</system_reminder>",
            id=f"{_REMINDER_ID_PREFIX}::{bucket}",
            additional_kwargs={_SCOPE_REMINDER_MARKER: True, _SCOPE_REMINDER_BUCKET: bucket},
        )

    # ------------------------------------------------------------------ history scans

    @staticmethod
    def _selection_digest(doc_ids: list[str]) -> str:
        """Order-insensitive content hash — the selection is a set, so the same
        ids in a different order must not mint a new reminder."""
        canonical = "\x1f".join(sorted(doc_ids))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _bucket_present(messages: Sequence[AnyMessage], bucket: str) -> bool:
        target_id = f"{_REMINDER_ID_PREFIX}::{bucket}"
        return any(getattr(m, "id", None) == target_id for m in messages)

    @staticmethod
    def _last_reminder_bucket(messages: Sequence[AnyMessage]) -> str | None:
        for message in reversed(messages):
            kwargs = getattr(message, "additional_kwargs", None) or {}
            if kwargs.get(_SCOPE_REMINDER_MARKER):
                bucket = kwargs.get(_SCOPE_REMINDER_BUCKET)
                return bucket if isinstance(bucket, str) else None
        return None

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
3. Partition the resolved agent set into registry tool keys and subagent keys.
4. Resolve any planner runtime model override from ``config["configurable"]``.
5. Build tools, subagents, and the planner prompt from the filtered registry.
6. Hand off to :func:`deepagents.create_deep_agent`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from sta_agent_engine.agents.base.runtime_model_config import redact_model_config


if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore

    from sta_agent_engine.agents.base.prompts.capability_definition import CapabilityDefinition

    from .habilitation.providers import HabilitationProvider

from .backends import build_orchestrator_backend, validate_uid_format
from .build_context import BuildContext
from .middlewares import (
    LiveMemoryMiddleware,
    PromptInjectionGuardSettings,
    ToolBudgetGuardSettings,
    compose_orchestrator_middleware,
)
from .middlewares.tool_budget_enforcement import ORCHESTRATOR_TOOL_BUDGET_GUARD_ENV_PREFIX
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
from .registry import SUBAGENT_REGISTRY, TOOL_REGISTRY


# Sources passed to ``LiveMemoryMiddleware`` when ``x-uid`` is present.
# Order matters: the user-authored ``AGENTS.md`` anchors the persona; the
# agent-curated ``preferences.md`` appends learned working notes.
_MEMORY_SOURCES: list[str] = ["/memory/AGENTS.md", "/memory/preferences.md"]


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


def _ensure_no_memory_kwarg(deep_agent_kwargs: dict[str, Any]) -> None:
    """Reject ``memory=[...]`` slipping into the deepagents kwargs.

    ``create_deep_agent(memory=[...])`` auto-attaches the stock
    ``MemoryMiddleware`` alongside ``LiveMemoryMiddleware``, producing
    double prompt injection plus duplicate cold-start loads. ``raise`` (not
    ``assert``) so the guard survives ``python -O``.
    """
    if deep_agent_kwargs.get("memory") is not None:
        msg = (
            "Pass memory plumbing via LiveMemoryMiddleware in the "
            "middleware list; do NOT pass memory=[...] to create_deep_agent "
            "(would attach stock MemoryMiddleware on top of ours)."
        )
        raise ValueError(msg)


logger = logging.getLogger(__name__)


# Server-owned decoder stop: the planner must never emit its own trailing
# ``Sources:`` block — the ordered source list is rendered downstream from the
# ``ka_sources`` channel, never by the planner (see the planner prompt's
# ``<output_format>``). The plain marker the planner/KA emit is ``Sources:`` (no
# space before the colon); the leading ``\n`` keeps the stop firing only at the
# block's own line, never mid-sentence. The set also covers the decorated header
# forms a model emits — italic ``*Sources*:``, bold ``**Sources**:``, and the
# ``## Sources`` heading — which a plain-substring stop misses because of the
# leading ``*`` / ``**`` / ``## `` between the newline and the word. Capped at 4
# entries for the OpenAI stop limit. Passed into ``create_chat_model``, it lands
# on the model instance (ChatOpenAI ``stop`` field / ChatMistralAI
# ``model_kwargs``) and survives the ``bind_tools`` deepagents applies.
_PLANNER_STOP_SEQUENCES: tuple[str, ...] = ("\nSources:", "\n*Sources*:", "\n**Sources**:", "\n## Sources")


class PlannerModelResolver:
    """Resolve the planner model before compiling the deepagents graph."""

    def __init__(
        self,
        *,
        model_override: str | BaseChatModel | None,
        model_factory: Callable[..., BaseChatModel],
        stop: tuple[str, ...] | None = None,
    ) -> None:
        self._default_model: str | BaseChatModel | None = model_override
        self._model_factory = model_factory
        self._stop = list(stop) if stop else None

    def resolve(self, runtime_model_config: Mapping[str, Any] | None) -> str | BaseChatModel:
        """Return the model object/string passed into ``create_deep_agent``.

        The server-owned ``stop`` (if set) is applied in BOTH factory branches —
        it is merged so the server value wins over any runtime-supplied ``stop``
        (the planner stop is an enforcement rule, not a caller-tunable knob). A
        pre-built ``model_override`` instance is returned untouched: it is a test
        seam, and stop on a real instance would have to be set at construction.
        """
        factory_kwargs: dict[str, Any] = {"stop": self._stop} if self._stop else {}
        if runtime_model_config:
            logger.info("Orchestrator planner runtime model override selected: %s", redact_model_config(runtime_model_config))
            return self._model_factory(**{**runtime_model_config, **factory_kwargs})

        if self._default_model is None:
            self._default_model = self._model_factory(**factory_kwargs)
        return self._default_model


def create_orchestrator_factory(
    *,
    hab_provider: HabilitationProvider | None = None,
    model_override: str | BaseChatModel | None = None,
    use_graph_cache: bool = True,
    graph_cache: dict[GraphCacheKey, CompiledStateGraph] | None = None,
    prompt_injection_guard_settings: PromptInjectionGuardSettings | None = None,
    tool_budget_guard_settings: ToolBudgetGuardSettings | None = None,
    store: BaseStore | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    enable_memory: bool = True,
) -> Callable[[RunnableConfig], Any]:
    """Return a closure-scoped ``make_orchestrator`` factory.

    Args:
        hab_provider: Habilitation provider built once at factory creation.
            ``None`` resolves from environment on first factory call.
        model_override: Optional planner model override for tests.
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
            ``False`` forces every request onto the anonymous backend shape
            (bare ``StateBackend``, no ``/memory/*`` Store routing, no
            ``LiveMemoryMiddleware``) even when a valid ``x-uid`` is present.
            Habilitation is unaffected — the uid still resolves roles and
            subagent permissions. Default ``True`` preserves the deployed
            behavior.

    Returns:
        An async factory with the 1-arg ``langgraph-api 0.4.x`` signature.
    """
    _resolved_habilitation_provider: HabilitationProvider | None = hab_provider
    _graph_cache: dict[GraphCacheKey, CompiledStateGraph] = graph_cache if graph_cache is not None else {}
    if graph_cache is not None:
        use_graph_cache = True

    from sta_agent_engine.models.custom_chat_model import create_chat_model

    planner_model_resolver = PlannerModelResolver(
        model_override=model_override,
        model_factory=create_chat_model,
        stop=_PLANNER_STOP_SEQUENCES,
    )
    _prompt_guard_settings = prompt_injection_guard_settings or PromptInjectionGuardSettings()
    _tool_budget_settings = tool_budget_guard_settings or ToolBudgetGuardSettings(_env_prefix=ORCHESTRATOR_TOOL_BUDGET_GUARD_ENV_PREFIX)

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
        has_uid = isinstance(raw_uid, str) and bool(raw_uid) and validate_uid_format(raw_uid)
        if isinstance(raw_uid, str) and raw_uid and not has_uid:
            # Log only the length — a malformed value could be an
            # attacker-controlled payload from a misconfigured gateway, so
            # do not emit it verbatim into logs.
            logger.warning(
                "Rejected malformed x-uid (length=%d); falling to anonymous path",
                len(raw_uid),
            )
        # Memory is structurally tied to has_uid: it selects the backend shape
        # (Store routing vs bare StateBackend), gates LiveMemoryMiddleware, and
        # partitions the graph cache. Disabling memory therefore reuses the
        # anonymous shape wholesale — habilitation above is untouched because
        # it reads request.uid, not this flag.
        has_uid = has_uid and enable_memory

        if _resolved_habilitation_provider is None:
            _resolved_habilitation_provider = resolve_habilitation_provider()

        t0 = time.perf_counter()
        habilitation = await resolve_orchestrator_habilitation(
            uid=request.uid,
            provider=_resolved_habilitation_provider,
            request_id=request.request_id,
        )
        permissions = select_orchestrator_permissions(habilitation.permitted_keys)
        cache_key = build_graph_cache_key(
            permissions=permissions,
            persona=request.persona,
            model_cache_key=request.model_cache_key,
            has_uid=has_uid,
        )
        if use_graph_cache and not habilitation.degraded and cache_key in _graph_cache:
            logger.debug(
                "Orchestrator graph cache hit: tools=%s subagents=%s persona=%s has_uid=%s",
                sorted(permissions.tools),
                sorted(permissions.subagents),
                request.persona,
                has_uid,
            )
            return _graph_cache[cache_key]

        planner_model = planner_model_resolver.resolve(request.runtime_model_config)

        ctx = BuildContext(persona=request.persona)

        tools = [TOOL_REGISTRY[key].factory(ctx, TOOL_REGISTRY[key].capability) for key in sorted(permissions.tools) if key in TOOL_REGISTRY]
        tool_caps = [TOOL_REGISTRY[key].capability for key in sorted(permissions.tools) if key in TOOL_REGISTRY]

        # Each permitted spec builds its own subagent from the request context.
        # A spec's build function owns its dependencies (retriever catalogs,
        # model factories) and its soft-landing middleware, and returns both the
        # wrapped subagent and the capability the planner advertises — already
        # enriched (e.g. the KA capability carries its corpora ``sources`` so the
        # planner block self-describes, gated on the KA being permitted).
        built = {key: SUBAGENT_REGISTRY[key].build(ctx) for key in sorted(permissions.subagents) if key in SUBAGENT_REGISTRY}
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
            spec = SUBAGENT_REGISTRY.get(sub_key)
            if spec is None or spec.bridge is None or spec.bridge in seen_bridges:
                continue
            seen_bridges.add(spec.bridge)
            bridge_middleware.append(spec.bridge())
        middleware[0:0] = bridge_middleware

        # Build a backend INSTANCE per graph-cache class (anonymous vs
        # authenticated). Anonymous → bare StateBackend (Store never
        # touched). Authenticated → CompositeBackend routing /memory/* to a
        # per-uid StoreBackend whose namespace callable reads x-uid per
        # call via get_config(). Same instance shared by FilesystemMiddleware
        # and LiveMemoryMiddleware so post-edit refresh and edit_file land
        # in the same Store namespace.
        backend = build_orchestrator_backend(has_uid=has_uid)
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
        if has_uid:
            middleware.append(
                LiveMemoryMiddleware(
                    backend=backend,
                    sources=_MEMORY_SOURCES,
                    system_prompt=ORCHESTRATOR_MEMORY_SYSTEM_PROMPT,
                    add_cache_control=False,
                )
            )

        _ensure_no_memory_kwarg(deep_agent_kwargs)

        # Disable the auto-injected general-purpose subagent only when this
        # factory is about to compile an orchestrator planner graph. The Deep
        # Agents harness-profile registry is process-global, so merely importing
        # this package or creating a factory must not affect unrelated graphs.
        register_orchestrator_harness_profiles()
        graph = create_deep_agent(**deep_agent_kwargs)

        if use_graph_cache and not habilitation.degraded:
            t_build = (time.perf_counter() - t0) * 1000
            logger.info(
                "Orchestrator graph cache miss — compiled in %.1fms for tools=%s subagents=%s persona=%s has_uid=%s",
                t_build,
                sorted(permissions.tools),
                sorted(permissions.subagents),
                request.persona,
                has_uid,
            )
            _graph_cache[cache_key] = graph

        return graph

    return make_orchestrator


make_orchestrator = create_orchestrator_factory()

-------

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
1. Identify the user's intent and the dimensions it could hold.
2. Pick the smallest SET of tools / sub-agents that together cover those
   dimensions — usually one, but every dimension the question genuinely raises
   must be covered, not just the first one that matches.
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
  figure, entity name, and ID the user needs to act on, and never refer to
  content they can't see ("as shown above", "as the sub-agent returned", "see
  the table") — there is no "above" for the user.
- Reply in the user's language.
- Be concise by default; go longer only when the user asks for detail or the
  question genuinely needs it. Conciseness is about your own prose — never trim
  a sub-agent's substance (counts, rows, and the knowledge sub-agent's citation
  markers) to save space, and never sacrifice the completeness a self-sufficient
  answer requires.
- Relay sub-agent answers faithfully: preserve their substance — counts,
  figures, entity names, IDs, and codes exactly as reported. Don't recompute or
  round figures, don't relabel entities, don't add details the sub-agent didn't
  provide.
- Keep a sub-agent's formatting when it makes the answer easier to read
  (tables, lists, code blocks) rather than flattening it to prose.
- If a sub-agent reports no result, relay that plainly (see the uncertainty
  rules) — never substitute a fabricated answer.
- Citations are knowledge-sub-agent only. Cite a knowledge fact by appending the
  bare marker ``[N]`` with the number shown for it in a ``<knowledge_sources>``
  note (these notes appear only when knowledge sources exist — there may be more
  than one across the conversation, and their numbers are already the ones the
  user will see). The bare ``[N]`` is the ONLY source reference allowed in the
  body of your reply — never a title, a url, the word "Sources", or an italicised
  source note mid-message. Never invent a number; never mark another sub-agent's
  facts (incident, topology, … carry no sources) or an operational/computed fact.
- Don't compile a sources list yourself, and ignore the delegation's own trailing
  ``Sources:`` block — the ordered list is shown to the user separately. If a
  ``Sources:`` line ever appears, it must be the very last line(s) of your reply,
  written exactly as ``Sources:`` at the line start — never bolded, a heading, or
  mid-message.
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

