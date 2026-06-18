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
# ``<output_format>``). The exact marker the planner/KA emit is ``Sources:``
# (no space before the colon); the leading ``\n`` keeps the stop firing only at
# the block's own line, never mid-sentence. Passed into ``create_chat_model``,
# it lands on the model instance (ChatOpenAI ``stop`` field / ChatMistralAI
# ``model_kwargs``) and survives the ``bind_tools`` deepagents applies.
_PLANNER_STOP_SEQUENCES: tuple[str, ...] = ("\nSources:", "\nSources :")


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
