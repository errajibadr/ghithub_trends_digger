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

        return {
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

    # ------------------------------------------------------------------
    # Caller-scope tool restriction
    # ------------------------------------------------------------------

    def _active_plan_inputs(
        self,
        state: KnowledgeAgentState,
    ) -> tuple[list[BaseTool], list[RetrieverEntry] | None, dict[str, set[str]]]:
        """Restrict the planner's tool set to scope-accepting retrievers when a caller scope is present.

        A caller-supplied ``ka_metadata_scope`` is a hard FILTER that only takes
        effect on retriever entries with ``accepts_caller_scope=True`` (see the KA
        ``AGENTS.md`` § Caller-supplied request scope). The planner picks retrievers
        from the query, not from where the filter applies — so a planner free to
        call a non-accepting corpus would search it *unfiltered*, silently ignoring
        the caller's selection and leaking out-of-scope content. When any scope axis
        is present, bind only the scope-accepting corpora so the model cannot reach
        a corpus where the filter would be dropped.

        Returns the full ``(tools, entries, tool_args_by_name)`` unchanged when:
        no scope is set (the common path); entries are unavailable (the opt-in flag
        lives on the entry, so the node cannot tell which tools are scope-accepting);
        or no entry accepts caller scope (binding zero tools would leave the KA
        unable to search — keep all and warn instead). This method never mutates
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

        accepted_names = {f"search_{entry.name}" for entry in self._entries if entry.accepts_caller_scope}
        active_tools = [tool for tool in self._tools if tool.name in accepted_names]
        if not active_tools:
            logger.warning(
                "plan_queries: caller scope present but no retriever accepts caller scope "
                "(accepts_caller_scope=True) — keeping all tools so the KA can still search; "
                "the caller filter will not apply."
            )
            return self._tools, self._entries, self._tool_args_by_name

        active_entries = [entry for entry in self._entries if entry.accepts_caller_scope]
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
            exposed_args = tool_args_by_name.get(tool_name, set())
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

packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/AGENTS.md
----
# AGENTS.md — orchestrator

Planner-driven deep-agent that replaces `twin_router`'s hand-rolled ReAct loop.
Wraps LangChain's `deepagents.create_deep_agent` and exposes three
`CompiledSubAgent` instances:

- **`knowledge_agent`** — Knowledge Agent (RAG over the source catalog).
- **`incident_agent`** — incident graph.
- **`topology_agent`** — wraps `navigator_agent` for graph/topology exploration.
  Granted to the same `prod` / `prd` roles as `knowledge_agent` and
  `incident_agent` by the orchestrator-owned policy table.

**Minimal first-party tools** — general-knowledge answers and clarifying
questions are handled natively by the planner via the `<general_knowledge>`
and `<clarification>` sections of its system prompt. The only first-party
runtime tool is `read_picture`, exposed by `MultimodalGuardMiddleware` when
the planner model is not listed in the multimodal model registry. Deepagents
auto-injected tools (`write_todos`, filesystem helpers, `task`) remain.
`TOOL_REGISTRY` is intentionally empty for policy-gated catalog tools;
future catalog tools slot in without restructuring.

> **Status:** Current branch owns habilitation under `orchestrator/habilitation/`
> and has no first-party GK/clarify tools. PR-3 memory is **implemented**:
> `LiveMemoryMiddleware` (subclass of stock `MemoryMiddleware`) with a
> `wrap_tool_call` refresh that fixes the checkpoint-staleness short-circuit;
> backend INSTANCE built per `has_uid` cache class via
> `build_orchestrator_backend(has_uid=...)` — anonymous → `StateBackend`,
> authenticated → `CompositeBackend` routing `/memory/*` to a `StoreBackend`
> whose `namespace=resolve_memory_namespace` reads `x-uid` per call. The
> same instance is shared by `FilesystemMiddleware` and
> `LiveMemoryMiddleware`. `has_uid` added to the graph cache key.
> Anonymous callers get no memory middleware. **PR-3.5 migration applied
> 2026-05-26**: replaced the original callable-factory `backend=make_backend`
> (deprecated in deepagents 0.5.0, removed in 0.7.0) with the blessed
> instance-with-`namespace=` pattern. Read
> `memory_bank/creative_phases/orchestrator/review_pr3_memory.md`,
> `implementation_plan_pr3_memory.md`, and
> `migration_namespace_scoped_backend_2026-05-26.md` before touching memory,
> backend, cache-key, or factory-signature code.

## Required reading before touching this package

Two failure modes this section is designed to prevent:

1. **Supposing instead of verifying** — the deepagents API surface looks
   intuitive but has sharp edges that bite when you guess.    Recent examples:
   `GeneralPurposeSubagentProfile` is not a `create_deep_agent` kwarg
   (it must be set on a registered `HarnessProfile` — which is exactly how
   `orchestrator_harness_profiles.py` disables GP for the `openai` / `mistral`
   planner providers); `task` strips
   `recursion_limit` from the parent config; `create_chat_model(model)`
   treats its positional arg as **provider**, not identifier.
2. **Reinventing what deepagents/langgraph already give you** — middleware
   composition, subagent registration, recursion budgets, memory backends.

Before writing or reviewing orchestrator code, consult these in priority order:

| Source | When |
|---|---|
| **Skills `/deep-agents-core`, `/deep-agents-memory`, `/deep-agents-orchestration`** | Default. Curated, version-pinned guidance on the deepagents API. Read the relevant skill *before* picking patterns. |
| **`/langgraph-agent-builder`** (in-repo skill) | STA-specific patterns — agent topology, server runtime, factory shape, auth. Auto-loaded via `CLAUDE.md`. |
| **`/langgraph-middlewares`** | Cross-check middleware ordering, hook semantics, request transformation. |
| **Deepagents source at `~/repository/deepagents/libs/deepagents/deepagents/`** | When the skill doesn't answer the question. Read `graph.py` for `create_deep_agent`, `middleware/subagents.py` for `CompiledSubAgent` + `task` config strip, `middleware/memory.py` for `MemoryMiddleware`, `profiles/harness/harness_profiles.py` for profile resolution. |
| **`memory_bank/creative_phases/orchestrator/`** | The design history. `creative_phase_2026-05-21_orchestrator_deepagent_v1.md` for the locked plan; `PR-1-fixes-2026-05-24.md` for the most recent state. |

If you're about to write `# I think this works...` — stop and read the source.
Half the findings in the PR-1 Codex review were "you supposed wrong."

## Architecture

```
caller HTTP request                                            ← LangGraph Server
   ↓ {x-uid, persona, x-request-id}
create_orchestrator_factory() → make_orchestrator(config)      ← closure-scoped 1-arg factory
   │
   ├─ resolve_habilitation_provider() (lazy, once per factory)
   ├─ resolve_orchestrator_habilitation(uid, provider)         ← per-call; rights NOT on context
   ├─ select_orchestrator_permissions(permitted)               ← registry-driven key selection
   ├─ lazy-cache KA entries (registry `_get_ka_entries`, @cache, first KA-permitted build)
   ├─ resolve planner runtime model from config["configurable"]["model_config(s)"]
   ├─ build_planner_system_prompt from each built subagent's capability
   ├─ SUBAGENT_REGISTRY[key].build(ctx) → BuiltSubagent per permitted key  ← build fn (in subagents/): raw graph → as_subagent (recursion bind + CompiledSubAgent wrap); react builds inject subagent_tool_call_guard()
   ├─ inject each permitted spec.bridge() into the middleware stack  ← permission-gated (KA → KnowledgeBridgeMiddleware)
   └─ create_deep_agent(model, tools=[], subagents, middleware, system_prompt)
       │
       └─ deepagents middleware stack (auto-injected: TodoList, Filesystem,
                                       PatchToolCalls, SubAgent, Summarization,
                                       AnthropicPromptCaching) PLUS our stack:
          1. PromptInjectionGuardMiddleware       ← pre-agent security short-circuit
          2. TimeAwareMiddleware (base/)          ← inject immutable <system_reminder> time msg (prefix-cache safe); after guard so refused turns skip it
          3. SubagentTaskFailureMiddleware         ← soft-land any subagent failure at the task tool boundary (re-raises GraphBubbleUp)
          4. MultimodalGuardMiddleware (base/)    ← strip images for text-only models + conditional read_picture
          5. MessageSequenceNormalizerMiddleware  ← repair orphan tool messages
          6. ToolBudgetEnforcementMiddleware     ← optional global per-run cap + optional per-tool caps (e.g. task<=5)
          7. GenerationRetryMiddleware            ← exception-path retry + empty-response retry (last in stack)
```

## Files map

```
orchestrator/
├── AGENTS.md                                — this file
├── CLAUDE.md                                — @AGENTS.md
├── SOUL.md                                  — static planner character + TWIN role; rendered as the lead <soul> section (build-time, shipped in the wheel)
├── __init__.py                              — public surface: OrchestratorContext + make_orchestrator
├── context.py                               — OrchestratorContext TypedDict (persona only)
├── orchestrator_catalog.py                  — create_orchestrator_factory + make_orchestrator (+ _load_soul, the cached SOUL.md reader)
├── orchestrator_harness_profiles.py         — registers GP-suppressing deepagents HarnessProfiles for the openai/mistral planner providers (idempotent; called from the factory)
├── registry.py                              — DECLARATIVE catalog: TOOL_REGISTRY + SUBAGENT_REGISTRY + SubagentSpec/ToolEntry + permission selectors (re-exports build fns/as_subagent/BuiltSubagent from subagents/). No build code.
├── orchestrator_resolution.py               — config/auth/cache-key resolution helpers (incl. has_uid)
├── build_context.py                         — BuildContext (persona-only per-request build context) + SupportsBuildContext protocol
├── backends/
│   ├── __init__.py                          — re-export build_orchestrator_backend + resolve_memory_namespace
│   └── user_backend.py                      — backend INSTANCE per has_uid + per-call namespace resolver (x-uid allowlist)
├── habilitation/                            — orchestrator-owned auth providers, resolver, policies
├── sources/
│   ├── twin_ka_entries.py                   — KA retriever catalog (orchestrator-owned)
│   └── twin_scope.py                        — TWIN_SCOPE_* scope builder
├── prompts/
│   ├── orchestrator_planner_prompt.py       — dynamic planner prompt builder (lead <soul>/<identity-fallback>; no <capabilities> block; roster comes from Deep Agents' subagent list)
│   ├── capabilities_xml.py                  — _compact_description: the subagent description Deep Agents renders (prose + use_for + examples + corpora)
│   ├── orchestrator_grounding.py            — <grounding> + <clarification> + <general_knowledge> + <uncertainty> + <guidelines>
│   └── memory_guidelines.py                 — ORCHESTRATOR_MEMORY_SYSTEM_PROMPT (two-file model + secrets-never)
├── subagents/                              — full per-subagent builds (raw graph → as_subagent wrap → BuiltSubagent); registry only references these
│   ├── __init__.py                          — re-exports build_* + packaging surface
│   ├── _packaging.py                        — SHARED leaf (imports no registry): BuiltSubagent + as_subagent + subagent_tool_call_guard + _SUBAGENT_TOOL_BUDGET
│   ├── build_knowledge_agent_subagent.py    — full KA build (build_knowledge_agent + _orchestrator_ka_config + _get_ka_entries + _KA_CAPABILITY) + budget knobs (SearchDepth, DEFAULT_KA_*, _compute_recursion_limit)
│   ├── build_incident_subagent.py           — incident raw builder (_load_incident_agent seam) + build_incident + _INCIDENT_CAPABILITY
│   └── build_topology_subagent.py           — navigator raw builder (deferred import; forwards middleware) + build_topology + _TOPOLOGY_CAPABILITY
└── middlewares/
    ├── __init__.py
    ├── orchestrator_middleware_compose.py   — minimal stack composer
    ├── live_memory.py                       — LiveMemoryMiddleware (wrap_tool_call refresh + fail-soft load)
    ├── prompt_injection_guard.py            — pre-agent classifier with jump_to="end" refusal
    ├── subagent_task_failure.py             — soft-land any subagent failure (recursion + arbitrary exceptions) at the task tool boundary; re-raises GraphBubbleUp (HITL/routing)
    ├── tool_budget_enforcement.py           — ToolBudgetEnforcementMiddleware: global per-run cap + optional per-tool caps (e.g. task<=5) + per-subagent tool-call guard
    ├── subagent_state_bridge.py             — SubagentStateBridge base (pure schema-widening, no hooks)
    └── knowledge_bridge.py                  — KnowledgeBridgeMiddleware: declares ka_metadata_scope (in) / ka_sources (out)
```

Tests mirror at `tests/test_ai_engine/agents/orchestrator/`.

## Planner prompt layering

`build_planner_system_prompt` composes the system prompt in a deliberate order
for **prefix caching**: a static character + behavior block leads (a stable
shared prefix across every call and user), and per-call dynamic content trails.

```
<soul>            ← SOUL.md (static character + TWIN role). Falls back to the
                    legacy <identity> constant when SOUL.md is absent/empty/
                    unreadable, so a packaging miss degrades, never breaks.
<objective>
<grounding> <clarification> <general_knowledge> <uncertainty> <guidelines>
[<subagent_tasking>]      ← only when subagents are permitted
<output_format>
── dynamic tail (kept last so the prefix above stays cache-stable) ──
[<persona>]               ← request.persona, now a trailing top-level section
                            (was nested in <identity>)
[<auth_status>]           ← degraded-mode banner; also bypasses the graph cache
```

**Persona precedence** (general → specific): `<soul>` (static character) →
behavior sections → `<persona>` (per-call caller flavor) → `<agent_memory>`
(per-user, injected later by `LiveMemoryMiddleware`). Memory is reference
material — on conflict with the user's current message or verified tool
evidence, the user / evidence wins (see `memory_guidelines.py`); and SOUL's
`<boundaries>` make personality never override safety, accuracy, or grounding.

`SOUL.md` is read once per process by `_load_soul()` (`@lru_cache`,
`Path(__file__).parent / "SOUL.md"`) in `orchestrator_catalog.py` and passed
into the builder as `soul=`. The builder defaults `soul=None`, so callers/tests
that omit it exercise the `<identity>` fallback path.

## Public surface

```python
from sta_agent_engine.agents.orchestrator import (
    OrchestratorContext,   # TypedDict, total=False — persona only
    make_orchestrator,     # async (config: RunnableConfig) -> CompiledStateGraph
)
```

Nothing else is exported. The declarative catalog (`SUBAGENT_REGISTRY`,
`SubagentSpec`, `ToolEntry`, permission selectors) lives in `registry.py`; the
per-subagent `build_*` functions and shared packaging (`BuiltSubagent`,
`as_subagent`, `subagent_tool_call_guard`) live under `subagents/`; and
`compose_orchestrator_middleware` under `middlewares/` — all internal API,
subject to change.

## Habilitation contract

Rights are resolved **per call** inside the factory from `x-uid` + a
habilitation provider — they are **never** carried on `OrchestratorContext`.
The context schema is caller-supplied; treating it as authoritative for
authorization would be a trust-boundary failure.

**Provider resolution order** (`resolve_habilitation_provider`):

1. `HABILITATION_BYPASS=1` → `BypassHabilitationProvider(role=HABILITATION_BYPASS_ROLE)` (default `prod`)
2. `HABILITATION_API_BASE_URL` set → `APIHabilitationProvider(api_url=..., api_key=HABILITATION_API_KEY)`
3. otherwise → `MockHabilitationProvider(fixed_role="non-prod", user_roles={"prod_user": "prod"})` for local dev

**Rights → tools / subagents** (`permission_keys` on registry entries):

| Permission key (resolver) | Orchestrator slot | Always-available? |
|---|---|---|
| `rag`, `knowledge_agent` | subagent `knowledge_agent` | No — gated |
| `incident`, `incident_agent` | subagent `incident_agent` | No — gated |
| `topology`, `topology_agent` | subagent `topology_agent` (navigator wrap) | No — gated; `prod` / `prd` today |

No policy-gated catalog tools are bound. The planner answers general-knowledge
questions and asks clarifications natively via its system prompt sections.
Anonymous UID → role `non-prod` → empty `permitted` → no subagents → no `task`
tool either. `read_picture` is separate: it is a middleware-owned fallback
tool that appears only when the planner model is not listed as multimodal.
Provider failure → fail-closed to the same anonymous slice, with a degraded
`<auth_status>` banner injected into the planner prompt.

**Renames pinned:** `lisab` → `knowledge_agent`. **No backward-compat alias.**
The string `lisab` must not appear anywhere in this package — pinned by test.

## Subagent contract — per-subagent build functions

Each subagent owns a **build function** `build_<name>(ctx: BuildContext) ->
BuiltSubagent` in its own `subagents/build_<name>_subagent.py` module — **not**
in `registry.py`. `registry.py` is purely declarative: it references each build
function from a `SubagentSpec` and does no construction itself. The build
function constructs the raw graph (capturing its own dependencies — retriever
catalogs, model factories — locally), then calls the one shared packaging helper
`as_subagent(graph, capability, *, recursion_limit)` (in
`subagents/_packaging.py`) to bind `recursion_limit` on the **inner** runnable
and wrap it in a `CompiledSubAgent`. There is no central dispatcher and no
`OrchestratorDeploy` threaded through the catalog.

The dependency chain is acyclic: `registry` → `subagents` → `subagents._packaging`
(the leaf, which imports nothing from `registry`). `_packaging` is where the
shared `BuiltSubagent`, `as_subagent`, and `subagent_tool_call_guard` live so both
the build modules and `registry` can import them without a cycle.

A subagent is declared once as a `SubagentSpec` in `SUBAGENT_REGISTRY`:

| `SubagentSpec` field | Role |
|---|---|
| `key` | Stable catalog key. |
| `capability: CapabilityDefinition` | Static capability skeleton (cheap, `sources=[]`). A build may return an enriched copy (e.g. KA fills `sources` with its corpora). |
| `permission_keys: tuple[str, ...]` | Habilitation keys that grant this subagent. |
| `build: Callable[[BuildContext], BuiltSubagent]` | Builds the subagent + the capability the planner advertises. |
| `bridge: type[AgentMiddleware] \| None` | Schema-widening middleware declaring the channels this subagent exchanges with the orchestrator. The catalog injects it **only when this subagent is permitted** (see § State bridge). |

`as_subagent` binds the limit via `.with_config(...)` on the inner graph. The
deepagents `task` tool strips `recursion_limit` from the parent config
(`~/repository/deepagents/libs/deepagents/deepagents/middleware/subagents.py:535-556`);
the inner-runnable bind survives that strip. The catalog reads
`spec.build(ctx).subagent` for the deep-agent and `.capability` for the prompt.

**KA specifics.** `build_knowledge_agent` builds the raw compiled KA graph via
`create_knowledge_agent(_get_ka_entries(), config=_orchestrator_ka_config())`,
which bakes `mode="answer"` (forced), `search_depth="deep"`, `review_cap=2` onto
a resolved `KnowledgeAgentConfig`. KA's `KnowledgeAgentInputState` accepts
`{"messages": [...]}` directly (last `HumanMessage` is the query) — exactly what
`task` sends, so no schema bridge is needed. It binds
`recursion_limit=_compute_recursion_limit("deep", 2)` and injects no
tool-call guard (KA is a fixed graph). `build_topology` injects
`subagent_tool_call_guard()` into the navigator's `post_middlewares` slot (see
§ Per-subagent tool-call guard). The KA build enriches its returned capability's
`sources` with `list_twin_ka_corpora()`. Those corpora are folded into the KA's
compact subagent description (`_compact_description`), which Deep Agents renders
in its auto-generated subagent list and `task` description — so the KA names its
corpora only when it is permitted. The planner prompt itself no longer carries a
`<capabilities>` block.

The orchestrator also bakes KA sub-agent context defaults into
`KnowledgeAgentConfig`: `subagent_mode=True`, and `streaming_enabled=True`
except at `search_depth="thorough"`. Thorough mode disables streaming because
`review_answer` can reject a draft and route back to synthesis; streaming a
rejected draft would leak an answer the KA later replaces.

The orchestrator also bakes KA sub-agent context defaults into
`KnowledgeAgentConfig`: `subagent_mode=True`, and `streaming_enabled=True`
except at `search_depth="thorough"`. Thorough mode disables streaming because
`review_answer` can reject a draft and route back to synthesis; streaming a
rejected draft would leak an answer the KA later replaces.

### Construction cost — rights-slice graph cache

`build_twin_ka_entries()` opens ES retriever connections. The
orchestrator factory has a process-local cache keyed by
`(tools, subagents, persona, planner_model_config_signature)`, where tools/subagents are registry
keys after permission resolution. The planner model signature is the
orchestrator planner model selected from runtime config; it does not get
passed as KA's default model. The prompt-injection guard is server-owned
settings and is not affected by request/runtime context. KA keeps its own
resolution chain (`KA_*` task env overrides → engine-wide env fallback)
through `KnowledgeAgentConfig.from_env(prefix="KA_")` and
`create_knowledge_agent(model=None, ...)`. Runtime model configs that
include API keys use a SHA-256 key fingerprint in the cache key, never the
raw secret.

The per-run tool budget guard is also server-owned
(`ORCHESTRATOR_TOOL_BUDGET_GUARD_MAX_TOOL_CALLS` or programmatic
`ToolBudgetGuardSettings`) and is not affected by request/runtime context.

PR-3 adds one more dimension: `has_uid`. This distinguishes anonymous graphs
from authenticated memory-enabled graphs without widening the cache to one
graph per user.

Bypass conditions (the cache must never serve a stale prompt):

- **Degraded habilitation** — the planner system prompt embeds an
  `<auth_status>` banner that varies per-request.
- **Provider error** — fail-closed path builds the anonymous slice
  with the same banner.

Cache lifetime is process-local: cleared on server restart, which is also
the only point at which env-driven KA model defaults are re-resolved.

### What bounds a KA call

No wrapper-level wall budget. Three primary controls handle it:

- **Loop control:** `KnowledgeAgentConfig.max_iterations = review_cap`
  bounds the KA's outer review-evidence loop.
- **Recursion control:** `recursion_limit = review_cap * 10 + 20` on
  the inner runnable bounds framework recursion.
- **Individual-call control:** KA's own retriever and model-client
  timeouts bound each network hop.

The end-to-end ceiling is the LangGraph Server ingress timeout.

## State bridge — KA shared channels

The orchestrator and the KA exchange data through deepagents' `task` tool,
which carries only state keys **both** sides declare. The KA's state inherits
those channels statically (`KnowledgeBridgeChannels`); the orchestrator's deep-
agent state is a fixed `AgentState`, so the channels are added by a
schema-widening middleware — `KnowledgeBridgeMiddleware` (a `SubagentStateBridge`
subclass). It is declared on the KA `SubagentSpec.bridge` and injected
into the orchestrator stack **only when `knowledge_agent` is permitted**, so a
caller who can't reach the KA never declares its channels (and the graph cache,
keyed on the permitted set, stays coherent).

Two channels:

| Channel | Direction | Shape / lifetime |
|---|---|---|
| `ka_metadata_scope` | **in** (orchestrator → KA) | `KaMetadataScope` dict `{doc_ids, apcode, app_name, entity}` — FILTER-ONLY hard filters, no boosts. `UntrackedValue` (run-scoped, never checkpointed, auto-fresh next run), `OmitFromOutput`. The KA narrows each opted-in retriever's build-time ceiling with it. |
| `ka_sources` | **out** (KA → orchestrator) | List of JSON-safe source dicts for a frontend references panel. Accumulating reducer (the planner may delegate to the KA several times per run, sequentially or in parallel); the bridge's `before_agent` resets it once per run with `Overwrite(value=[])`. Before each planner generation the bridge's `before_model` **sources-announcer** appends a `<knowledge_sources>` note numbering the sources merged since the last call (`ka_sources[announced:]`) from the post-merge channel, so the planner cites the exact `[N]` the panel will render — parallel-safe. The bridge's `after_agent` then renders a deterministic `Sources:` block from this channel when the planner's final answer omits one (§ State bridge — the Sources block). |

`ka_metadata_scope` is delivered as **caller input state** — there is no
header/producer middleware. A self-hosted caller seeds it on the orchestrator
input; `task` forwards it to the KA. **Per-retriever opt-in:** only KA retriever
entries with `accepts_caller_scope=True` read it (default `False`) — the broad
`general_doc` entry opts in; the scoped `twin_project_doc` entry does not, so
caller filtering never crosses the twin boundary. Applied via
`MetadataScope.narrow_with`, it can only tighten, never widen. The contract +
typed `KaMetadataScope` are documented in `agents/knowledge_agent/AGENTS.md`
§ Caller-supplied request scope.

**Planner visibility — the doc-selection reminder.** The scope flows to the KA
silently through state, so the bridge's `before_agent` also surfaces a
`doc_ids` selection to the planner: it appends an immutable `<system_reminder>`
message ("the user pre-selected documents d1, d2 — forward the search to
`knowledge_agent`, don't answer from general knowledge"). The reminder does
**not** name a corpus: the KA selects its own corpus, and `PlanQueriesNode`
structurally restricts its bound tool set to the `accepts_caller_scope=True`
corpora whenever a caller scope is present (see the KA `AGENTS.md`
§ Caller-supplied request scope), so steering toward `general_doc` is redundant.
Prefix-cache safe per the `TimeAwareMiddleware` pattern: tail-appended, frozen
by a deterministic id (`sta-ka-scope-reminder::turn-{n}::{digest}` — digest is
order-insensitive over the id set), so interrupt-resume/retry re-entry is
idempotent and a new turn or changed selection mints a fresh reminder. A turn
whose scope carries no `doc_ids` on a previously-scoped thread appends a
one-shot "selection no longer applies" note so stale reminders can't mislead
later turns. Only `doc_ids` triggers it — the other axes stay silent.

**Planner visibility — the canonical-sources note.** `ka_sources` numbers map to
rows by position, but each KA `task` restarts its own inline `[N]` at `[1]`, so a
later delegation's numbers need offsetting. Rather than make the planner do that
arithmetic, the bridge's **`before_model`** hook announces the sources merged
since the last planner call. It reads the *post-merge* channel and a per-run
`ka_sources_announced` cursor, and appends a `<knowledge_sources>`
`SystemMessage` listing `ka_sources[announced:]` numbered `[announced+1 …]` — the
exact `[N]` the downstream panel renders — then advances the cursor.
Deterministic id (`sta-ka-sources::{offset}::{digest}`) makes a replay/resume
idempotent.

Why `before_model` and not `wrap_tool_call`: when the planner fans out **parallel**
`task` delegations in one turn, the sibling calls all observe the same *pre-merge*
state snapshot, so a per-`task` announcer would number every block from the same
offset and collide. `before_model` is the first hook *after* the tool super-step
merges, so numbering from `ka_sources` itself is correct for one call or many. The
cursor resets to `0` in `before_agent` alongside the `ka_sources` `Overwrite([])`,
so a fresh run re-announces from `[1]`.

The planner is told (in `<output_format>`) to cite **inline** with those numbers
and to **end** a knowledge-cited reply with its own `Sources:` block of
`[N] [title](url)` links built from the note. The note is a `SystemMessage` (not
an `AIMessage`, which Anthropic would treat as a prefill); it only ever *appends* a
message — never reads or rewrites the answer `ToolMessage` (§ Pitfalls #5).

**Planner visibility — the Sources block (after_agent ownership).** The
orchestrator no longer suppresses the planner's trailing `Sources:` block with a
decoder `stop` (that stop made vLLM+Nemotron leak `</think>` each generation, and
is removed). Instead the bridge *guarantees* the block: its **`after_agent`** hook
(the exit node, fired once) checks the final answer — when it is a real generation
(`AIMessage`, non-empty text, no `tool_calls`) that used the KA (`ka_sources`
non-empty) but did not already emit a `Sources:` header, it appends a deterministic
`**Sources**:` block (`format_sources_block`) as a **separate** `AIMessage` tagged
`additional_kwargs={"sta_generated": "sources_block"}`. `aafter_agent` fake-streams
it via `GenericFakeChatModel`, whose chunks carry the committed message's
deterministic id so the frontend's `seen_message_ids` dedup renders it exactly once
(token stream, no duplicate whole-message). A model-emitted header is left
untouched (`detect_sources_header`, D1); a refused / general-knowledge / mid-loop
turn is a no-op (empty `ka_sources` or non-generation last message).

## Per-subagent tool-call guard

Distinct from the orchestrator's own per-run `ToolBudgetEnforcementMiddleware`
(stack position 5, an optional global cap), a react-agent subagent's build
function injects a **per-subagent** tool-call budget guard into that subagent's
own middleware stack (the soft-landing behavior: an over-budget run is forced to
a final answer before the recursion ceiling). The reusable helper
`subagent_tool_call_guard()` (in `subagents/_packaging.py`) returns
`[ToolBudgetEnforcementMiddleware(max_tool_calls=_SUBAGENT_TOOL_BUDGET)]`
(`_SUBAGENT_TOOL_BUDGET=15`). Only `build_topology` injects it (the navigator is
a `create_agent`-based react agent — it accepts an `AgentMiddleware`, forwarded
to its `post_middlewares` slot); the KA and incident builds are fixed graphs and
inject nothing. The navigator dedups forwarded middleware by type so a cap is
never double-installed.

## Tool contract

There is no policy-gated first-party tool surface today. General-knowledge
answers and clarifying questions live in the planner's system prompt
(`<general_knowledge>` and `<clarification>` sections in
`prompts/orchestrator_grounding.py`). The planner uses its own LLM
capability for both; no separate model is spun up for them.

`read_picture` is middleware-owned by `MultimodalGuardMiddleware`, not a
`TOOL_REGISTRY` entry. It is exposed only when the resolved planner model is
not multimodal according to `sta_agent_engine.models.capabilities.is_multimodal()`.
It forwards visible conversation context and recent image parts to the
server-configured `ORCHESTRATOR_PICTURE_READER_*` multimodal model.

`TOOL_REGISTRY` is kept as an empty dict so future policy-gated catalog tools
can be added without restructuring the catalog. When that happens, add a
`ToolEntry` with `factory(deploy, capability)` shape mirroring the
existing subagent entries.

**Clarify may return later.** The deleted `clarify_user` tool was a plain
LLM-call parity port. A future version may reintroduce a richer Clarify tool
that accepts multiple-choice options as structured args so the frontend can
display them cleanly. Until then, clarification stays in the planner prompt.

**Future subagent tasking ownership.** Current v0 task guidance is prompt-only:
the planner's `<subagent_tasking>` section shapes `task` prompts. A future
version may own/customize `SubAgentMiddleware` directly, and may propose an
upstream Deep Agents API for passing
`SubAgentMiddleware(system_prompt=..., task_description=...)` through
`create_deep_agent`.

## Middleware stack

Order is load-bearing. The list returned by `compose_orchestrator_middleware()`:

| # | Middleware | Source | Why this position |
|---|---|---|---|
| 1 | `PromptInjectionGuardMiddleware` | `orchestrator/middlewares/` | Runs once in `before_agent` before planner/model/tool work; can `jump_to="end"` with a deterministic refusal. Screens recent human turns, forwarding image parts (incl. image-only turns) to a multimodal judge. |
| 2 | `TimeAwareMiddleware` | `base/middlewares/` | `before_agent` only — appends an immutable `<system_reminder>` time message at the tail (prefix-cache safe; one per refresh window, never a system-prompt rewrite). Sits **after** the guard so a refused turn (the guard's `jump_to="end"` skips later `before_agent` hooks) does not append a reminder to a short-circuited refusal — the mirror of why state bridges sit *before* the guard. Being `before_agent`-only, it does not change the tool-call/model-call nesting of the wrappers below. |
| 3 | `SubagentTaskFailureMiddleware` | `orchestrator/middlewares/` | Outermost **exception-catching** `wrap_tool_call` wrapper. (The KA bridge no longer wraps tool calls — its sources-announcer moved to `before_model` — so this is unambiguously the outermost tool-call wrapper again.) Catches **any** exception raised by subagent code through the `task` delegation tool — an exhausted `recursion_limit` (a `GraphRecursionError`) or any other failure in a subagent we don't control — and converts it to a recoverable error `ToolMessage` so one buggy subagent cannot abort the whole run. **Re-raises `GraphBubbleUp`** (`GraphInterrupt` HITL + `ParentCommand` routing) untouched, surfaces only the exception *type* to the planner (full traceback logged server-side), and does not retry (the planner is the recovery loop; `ToolBudgetEnforcementMiddleware` caps total `task` calls). Hallucinated subagent *names* are out of scope — deepagents' `task` returns a string for an unknown name, it does not raise. Must precede `ToolBudgetEnforcementMiddleware`; sits after the guard and time-aware middleware (which only hook `before_agent`). |
| 4 | `MultimodalGuardMiddleware` | `base/middlewares/` | Strip images early so later model-facing middleware sees normalized content; expose `read_picture` only for planner models not listed as multimodal. |
| 5 | `MessageSequenceNormalizerMiddleware` | `base/middlewares/` | Pre-model — repairs orphan tool messages before the model sees the sequence. |
| 6 | `ToolBudgetEnforcementMiddleware` | `orchestrator/middlewares/` | Optional **global** per-run tool-call cap **plus** optional **per-tool** caps (e.g. `task<=5`). Subclasses LangChain's stock limiter for its count channels but owns counting in a single `after_model` pass (the `run_tool_call_count` channel is last-writer-wins, so one writer for every count key is the race-free way to combine global + per-tool + answer-now). A per-tool cap blocks only that tool with a recoverable error `ToolMessage` and never forces answer-now. Only the **global** cap forces the next planner call to answer; that answer-now turn is **prefix-cache-safe** — it keeps the `tools` and `system` blocks byte-identical (so the vLLM prefix stays cached), sets `tool_choice="none"`, and appends the budget instruction as a trailing message rather than rewriting the system prompt or dropping tools. |
| 7 | `GenerationRetryMiddleware` | `base/middlewares/` | Exception-path retry with exponential backoff and opt-in success-path retry on empty content + no tool_calls. Tool-call-only messages **pass through** — they are valid mid-loop turns. **`model_not_found`** swaps to fallback immediately (no same-model retry loop). Last in the stack — wraps the rest. |

The prompt-injection guard uses both sync and async `before_agent` hooks and
must keep `@hook_config(can_jump_to=["end"])`; without that decorator,
LangChain ignores `jump_to`. Its classifier is server-owned by default
(`ORCHESTRATOR_PROMPT_INJECTION_GUARD_*`) and fails open unless
`ORCHESTRATOR_PROMPT_INJECTION_GUARD_FAIL_OPEN=false`. Programmatic
`PromptInjectionGuardSettings` passed to the orchestrator factory take
precedence over env defaults; request/runtime context cannot override guard
settings.

**Image screening is judge-capability-gated.** Only a *multimodal* judge can
screen image content for injection; a text-only judge sees images stripped to
an omission note. So an image-bearing turn (including an image-only turn with
no text) is classified only when the resolved judge is multimodal — otherwise
the guard does not spend a classify call on content it cannot read. The default
judge (`mistral/mistral-small-2603`) is multimodal, so images are screened by
default. We accept that a deliberately-configured text-only judge cannot guard
images; `compose_orchestrator_middleware` logs a warning when eager image
description is enabled with a text-only judge so the trade-off is never silent.

**State bridges are prepended ahead of the guard.** Permitted subagent state
bridges (e.g. `KnowledgeBridgeMiddleware`) are injected at the *front* of the
assembled stack at catalog assembly, not appended. The KA bridge's
`before_agent` does per-run state upkeep: it resets an accumulating output
channel (the KA's checkpointed `ka_sources`) and appends the planner-facing
doc-selection `<system_reminder>` when the caller seeded
`ka_metadata_scope.doc_ids` (see § State bridge). The guard's `jump_to="end"`
on a blocked turn skips every later `before_agent` hook, so a reset sitting
after the guard would leak the previous turn's sources into a refused turn's
output. Running ahead of the screen is safe: the bridge reads only the
caller-seeded scope channel (never user message content) and cannot jump, so
the guard remains the first hook that observes user input. Accepted trade-off:
a refused turn may carry the scope reminder in its checkpointed history.

The same bridge also owns a `before_model` **sources-announcer**: before each
planner generation it appends a `<knowledge_sources>` note (canonical `[N]`)
numbering the sources merged since the last call (`ka_sources[announced:]`,
tracked by a per-run `ka_sources_announced` cursor), so the planner cites numbers
it can read instead of doing multi-call offset arithmetic. It runs in
`before_model` — not `wrap_tool_call` — precisely so the numbering is correct
under **parallel** `task` fan-out: sibling delegations share a pre-merge snapshot,
so a per-call announcer would number them from the same offset and collide;
`before_model` is the first hook after the tool super-step merges, so it numbers
from the final channel order. The announcer only ever *appends* a `SystemMessage`
— it never reads or rewrites the answer `ToolMessage` (§ Pitfalls #5) — and the
bridge no longer wraps tool calls at all, leaving `SubagentTaskFailureMiddleware`
the sole, outermost `wrap_tool_call` wrapper.

The same bridge also owns an **`after_agent`** hook that *guarantees* the
trailing `Sources:` block: when the final answer is a KA-grounded generation with
no `Sources:` header, it appends a deterministic `**Sources**:` block as a
separate audit-tagged `AIMessage` (`aafter_agent` fake-streams it with the
committed message's id for frontend single-render). This replaced the planner
decoder `stop` (which leaked `</think>` on vLLM+Nemotron). See § State bridge —
the Sources block.

The tool budget guard installs when **either** the global cap
(`ORCHESTRATOR_TOOL_BUDGET_GUARD_MAX_TOOL_CALLS`) **or** the per-tool caps
(`ORCHESTRATOR_TOOL_BUDGET_GUARD_PER_TOOL_MAX_CALLS`, JSON, e.g.
`'{"task": 5}'`) are set; both are opt-in and unset by default. To cap the
`task` delegation tool at 5 per run, set
`ORCHESTRATOR_TOOL_BUDGET_GUARD_PER_TOOL_MAX_CALLS='{"task": 5}'`. Runtime
`configurable` values cannot install the guard or raise a cap. The guard counts
main planner/deep-agent tool calls, including `task` delegation calls; the
prompt-injection classifier and answer-now no-tool model call are not tool calls
and are not counted. A per-tool cap blocks only the exhausted tool (a recoverable
error `ToolMessage`); only the global cap forces the prefix-cache-safe answer-now
turn.

**Deferred to later PRs:**
- `MemoryMiddleware` (PR-3) — per-user backend factory keyed on `x-uid`,
  no-op when anonymous. Verified import path:
  `from deepagents.middleware.memory import MemoryMiddleware`.
- `DynamicModel` / `DynamicTool` middlewares — slip in when orchestrator
  context carries the fields they read. Do not mix `DynamicModelMiddleware`
  with the current conditional `read_picture` wiring without redesigning it:
  `read_picture` is bound at graph construction from the planner model, while
  dynamic model switching happens at runtime inside `wrap_model_call`.

**Deepagents auto-injects** (we do NOT add them ourselves):
`TodoListMiddleware`, `FilesystemMiddleware`, `PatchToolCallsMiddleware`,
`SubAgentMiddleware` (when subagents non-empty), `create_summarization_middleware`,
`AnthropicPromptCachingMiddleware`.

## PR-3 Memory Contract (implemented)

`LiveMemoryMiddleware(MemoryMiddleware)` is the in-tree subclass that fixes
the stock middleware's checkpoint-staleness short-circuit. Stock
`MemoryMiddleware` returns early when `state["memory_contents"]` is already
present — which it always is after the first turn under a checkpointer, so
LLM edits become invisible until the thread resets. The subclass adds a
`wrap_tool_call` hook that re-reads the edited memory path through the
backend after a successful `edit_file` / `write_file` against any
configured source, and folds the refreshed content into state via a
`Command(update=...)`.

**Wiring contract (pinned by tests):**

- Pass `memory=None` to `create_deep_agent`; do not use `memory=[...]`. The
  shortcut auto-attaches stock `MemoryMiddleware` on top of ours — double
  prompt injection plus duplicate cold-start loads. The orchestrator factory
  raises `ValueError` if the kwarg slips back in (not `assert`, so the guard
  survives `python -O`).
- `LiveMemoryMiddleware` is wired manually in `middleware=[...]` only on the
  authenticated path; anonymous gets no memory middleware at all (structural
  isolation, not runtime branching).
- Backend coherence: `build_orchestrator_backend(has_uid=...)` returns one
  INSTANCE per `has_uid` cache class. The catalog passes that instance
  to both `create_deep_agent(backend=...)` (driving `FilesystemMiddleware`)
  and `LiveMemoryMiddleware(backend=...)`. Same object — post-edit refresh
  and `edit_file` provably resolve through the same call stack.
  **Callable-as-backend is deprecated in deepagents 0.5.0, removed in
  0.7.0.** Passing an instance avoids `warn_deprecated` and keeps us
  future-proof.
- Routing: `CompositeBackend(default=StateBackend(), routes={"/memory/": StoreBackend(namespace=resolve_memory_namespace)})`.
  `/memory/*` writes hit the per-user `StoreBackend` namespace
  `(uid, "memory")`; everything else stays ephemeral. The namespace
  callable reads `x-uid` per call via `langgraph.config.get_config()` —
  the same backend instance is safely shared across users because per-uid
  isolation lives in the namespace callable, not the instance.
- Sources: `/memory/AGENTS.md` (user-authored) and `/memory/preferences.md`
  (agent-curated). Order matters in the rendered prompt — AGENTS.md anchors
  the persona first.
- Cache key: `has_uid: bool` partitions anonymous and authenticated graphs
  without widening the cache to one entry per user. **Never** put raw
  `x-uid` in the cache key.
- `x-uid` validation runs at two boundaries:
  1. **Catalog (trust boundary)** — `make_orchestrator` calls
     `validate_uid_format(raw_uid)` before flipping `has_uid=True`. A
     malformed value (chars outside `^[A-Za-z0-9\-_.@+:~]+$` or > 256
     chars) falls to the anonymous path (bare `StateBackend`, no
     `LiveMemoryMiddleware`) with a `logger.warning` recording only the
     length. The user request stays alive; the gateway misconfig surfaces
     to operators in logs.
  2. **Resolver (defense in depth)** — `resolve_memory_namespace` keeps
     raising `ValueError` for the same conditions and `RuntimeError` for
     missing/empty uid. Reachable only if some future path bypasses the
     catalog check.

  The allowlist regex is matched with `re.fullmatch`, not `re.match` —
  `$` in Python regex matches before a trailing `\n`, so `re.match` would
  let `"alice\n"` slip through. Both `validate_uid_format` and
  `resolve_memory_namespace` use `fullmatch`.
- Fail-soft: backend exceptions during memory load degrade to empty memory
  + warning log. **Exception:** `RuntimeError` from the langgraph store
  lookup (missing store, deployment misconfig) is re-raised. Operators must
  see deployment errors.
- Public `make_orchestrator(config)` stays 1-arg for langgraph-api 0.4.x.
  Test/local `store` and `checkpointer` injection goes on
  `create_orchestrator_factory(..., store=..., checkpointer=...)` —
  closure-captured, not per-call.
- Server-owned off-switch: `create_orchestrator_factory(enable_memory=False)`
  forces every request onto the anonymous memory shape (bare `StateBackend`,
  no `/memory/*` Store routing, no `LiveMemoryMiddleware`) even when a valid
  `x-uid` is present — applied as `has_uid = has_uid and enable_memory`, so
  the graph cache stays coherent for free. Habilitation is untouched (it
  reads `request.uid`, not the flag): the uid still grants subagents.
  Removing `LiveMemoryMiddleware` alone is NOT a valid way to disable memory
  — the backend would still route `/memory/*` writes into the per-user Store
  via `FilesystemMiddleware`. Pinned by
  `test_enable_memory_false_forces_anonymous_shape_for_authenticated_request`.

### Setup — Store/checkpointer wiring

| Deployment | `store=` | `checkpointer=` |
|---|---|---|
| LangGraph Platform | leave `None` — platform attaches managed Postgres | leave `None` — platform attaches |
| Streamlit frontend (`LocalFactoryGraphConfig`) | auto-attached per session by `services/graph_catalog.py:_create_factory_graph` | auto-attached per session |
| Standalone (tests, scripts, ad-hoc) | pass `InMemoryStore()` (or your prod store) | pass `InMemorySaver()` (or your prod checkpointer) |

If you forget to pass a `store` in a standalone caller, the first memory
read raises `RuntimeError` from `langgraph.config.get_store` — fail loud
by design, not silent degradation.

The Streamlit frontend mirrors the LangGraph Platform contract: a single
session-scoped `InMemoryStore` lives in `st.session_state.graph_store` and
is attached to a per-session copy of factory-built graphs (parallel to the
existing checkpointer attachment). Restarting Streamlit clears the store;
restarting LangGraph Platform does not (it uses managed Postgres).
Reference standalone wiring: `examples/sta_agent_engine/orchestrator/orchestrator_memory_example.py`.

## Commands

```bash
# package tests (preferred)
uv run pytest tests/test_ai_engine/agents/orchestrator/ -v

# single file
uv run pytest tests/test_ai_engine/agents/orchestrator/test_orchestrator_catalog.py -v

# typecheck a single source file
uv run pyright packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/orchestrator_catalog.py

# format + lint a single file
uv run ruff format packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/orchestrator_catalog.py
uv run ruff check packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/orchestrator_catalog.py --fix
```

## Pitfalls (orchestrator-specific)

These are the traps a fresh agent will hit. **Read them before editing.**

1. **`task` strips `recursion_limit` from parent config.** The shared
   `as_subagent` helper binds it on the **inner** raw graph via `.with_config(...)`
   — a parent-config bind does nothing. Raw-graph builders must NOT bind it
   themselves; the build function's `as_subagent` call owns it.
   See `~/repository/deepagents/libs/deepagents/deepagents/middleware/subagents.py:535-556`.

2. **General-purpose subagent is disabled for the `openai` / `mistral` planner
   providers.** Deepagents auto-injects `general-purpose` alongside registered
   subagents; it carries no first-party orchestrator tools and only invites
   mis-routing. `orchestrator_harness_profiles.register_orchestrator_harness_profiles()`
   (called immediately before planner graph compilation, not during package
   import or outer factory creation) registers a provider-wide
   `HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False))`
   for both providers. The `task` tool survives because the KA / incident /
   topology sync subagents are still registered. Caveats: (a) keyed on the
   planner model's `ls_provider`, so a non-`openai`/`mistral` planner (llmaas,
   custom) is **not** covered — add its provider key to
   `ORCHESTRATOR_HARNESS_PROVIDERS`; (b) the deepagents harness registry is
   **process-global**, so the suppression also applies to any other
   `create_deep_agent` graph on those providers in the same process. Flip
   `DISABLE_GENERAL_PURPOSE_SUBAGENT = False` to restore GP — re-enable only if
   production telemetry shows the planner under-delegating because GP is absent.

3. **`create_chat_model(positional_arg)` treats the positional as provider,
   not model identifier.** Always use `create_chat_model(model=...)` or
   `create_chat_model(provider="...", model="...")`. Passing a model name
   positionally crashes on any non-provider string.

4. **`OrchestratorContext` is caller-supplied and must NOT carry rights.**
   The context schema has `persona` only. Rights resolve per-call from
   `x-uid` + the habilitation provider inside the factory.

5. **The KA's `Sources:` block reaches `messages[-1].content` verbatim
   because direct registration has no transform layer.** Deepagents'
   `task` reads the last AIMessage content as the `ToolMessage` content
   the planner sees. If you ever reintroduce a wrapper graph between
   `task` and the KA, preserve `result["messages"]` passthrough — any
   rewrite breaks citation fidelity.

6. **Importing `sta_agent_engine.agents.orchestrator` must not open network
   connections.** Habilitation provider construction and entries provider
   imports are local-imported inside the
   factory body. Any new dependency on a network-touching module must
   follow the same pattern.

7. **`lisab` does not exist in this package.** Rename is locked; a stray
   reference fails `test_lisab_literal_nowhere_in_orchestrator_package`.

8. **Default to direct subagent registration.** A build function constructs a
   raw compiled graph and calls `as_subagent` to bind `recursion_limit` on the
   inner runnable (for `task` config-strip survival) and wrap it in
   `CompiledSubAgent`. `CompiledSubAgent.runnable` takes any `Runnable` that
   responds to `ainvoke(state, config)`. Reach for a wrapper graph only when
   the inner can't accept `{"messages": [...]}` input or needs a
   non-orchestrator `context_schema` populated. The KA build is the reference:
   no shim, no schema bridge — knobs go on `KnowledgeAgentConfig` at build time;
   recursion-bind + wrap go through the shared `as_subagent` helper.

9. **Cache writes happen only on the normal habilitation path.** The
    rights-slice cache (keyed on `(frozenset(permitted_agents), persona)`)
    must never absorb a degraded or error-path build — those carry a
    per-request `<auth_status>` banner in the planner prompt and would
    leak that banner to a subsequent healthy request. The guard lives
    in `create_orchestrator_factory`; pinned by
    `test_orchestrator_graph_cache.py`.

10. **Topology subagent is import-time-network-sensitive.** The
    `navigator_agent` package opens a graph backend connection at
    module top level (`repo = UKGRepository(...)`). The topology builder
    defers the `get_navigator_graph` import to its function body so the
    orchestrator's `__init__` stays network-free. Any future
    refactor of `build_topology_subagent.py` must preserve the deferred
    import.

11. **The `topology` policy key is recognized and currently granted to
    `prod` / `prd`.** Treat topology as part of the prod slice unless the
    orchestrator `habilitation/policies.py` table changes.

12. **Never pass `memory=[...]` to `create_deep_agent`.** That installs stock
    `MemoryMiddleware`, which has the checkpoint-staleness bug
    `LiveMemoryMiddleware` exists to fix. The orchestrator factory
    `raise ValueError`s if the kwarg slips back in; the regression-pinning
    test sits in `test_orchestrator_catalog_memory.py`.

13. **`make_orchestrator(config)` stays one-arg.** Pinned by
    `test_factory_is_one_arg_make_graph` for langgraph-api 0.4.x. Standalone
    `store` / `checkpointer` injection lives on `create_orchestrator_factory(...)`,
    not on the public `make_orchestrator`. The 2-arg `ServerRuntime` form
    waits for langgraph-api >= 0.7.x.

14. **Fail-soft on memory load swallows `Exception` — NOT `RuntimeError`.**
    `LiveMemoryMiddleware.{a,}before_agent` degrades to empty memory + warning
    log on transient backend failures, but re-raises `RuntimeError` from the
    langgraph store lookup. Missing store is a deployment misconfig — operators
    must see it. Do not widen the except clause to also catch `RuntimeError`
    "to be safe" — that hides the misconfig in production.

15. **`x-uid` is validated at the catalog trust boundary AND inside the
    resolver.** The catalog calls `validate_uid_format(raw_uid)` before
    `has_uid=True`; a malformed value falls to the anonymous path with a
    length-only warning log (don't widen this to log the value — could be
    an attacker-controlled payload). `resolve_memory_namespace` keeps its
    `ValueError`/`RuntimeError` raises as defense-in-depth — it's only
    reachable if a future path bypasses the catalog. Use `re.fullmatch`
    everywhere; `re.match` + `$` lets `"alice\n"` slip through because
    `$` matches before a trailing newline. Pinned by `test_user_backend.py`
    + `test_orchestrator_catalog_memory.py` (`test_*_falls_to_anonymous_path`).

16. **GP subagent shares the per-user backend instance.** Auto-injected
    `general-purpose` inherits filesystem tools and could in theory write
    to or read from `/memory/*`. **Write side:** the subagent state filter
    (`_EXCLUDED_STATE_KEYS` in deepagents) blocks the resulting
    `memory_contents` from reaching the parent (verified in
    `test_subagent_memory_isolation.py`), and GP's default prompt has no
    memory awareness. **Read side is NOT blocked:** a GP `read_file`
    returns the memory body as a normal `ToolMessage` that the parent
    appends to its transcript — content leaks even though `memory_contents`
    does not. Deny-rule hardening
    (`FilesystemPermission(operations=["read", "write"], paths=["/memory/**"])`)
    on the GP subagent definition is deferred to a future PR; until then,
    treat anything in `/memory/*` as observable by any GP delegation in
    the same turn.

## Known Issues

_None currently open._

The incident subagent now lives at `agents/incident_agent/` (a single
relocated mock); `build_incident_subagent.py` binds it through the
`_load_incident_agent()` seam, so the builder's `recursion_limit` actually
lands on the inner runnable (the former `[ORCH-01]` seam-bypass).

## Refs

- Design: `memory_bank/creative_phases/orchestrator/creative_phase_2026-05-21_orchestrator_deepagent_v1.md`
- PR-3 memory review (IMPLEMENTED): `memory_bank/creative_phases/orchestrator/review_pr3_memory.md`
- PR-3.5 namespace-scoped backend migration: `memory_bank/creative_phases/orchestrator/migration_namespace_scoped_backend_2026-05-26.md`
- PR-1 synthesis: `memory_bank/creative_phases/orchestrator/PR-1-fixes-2026-05-24.md`
- Over-complexity refactor (implemented): `memory_bank/creative_phases/orchestrator/review_orchestrator.md`
- Knowledge Agent (subagent target): `agents/knowledge_agent/AGENTS.md`
- Twin router (legacy, shadowed): `agents/twin_router/`
- Habilitation policies: `agents/orchestrator/habilitation/policies.py`
- Deepagents source: `~/repository/deepagents/libs/deepagents/deepagents/`
- In-repo LangGraph patterns: `.claude/skills/langgraph-agent-builder/SKILL.md`

-------

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
an ``AIMessage``, which Anthropic would treat as a prefill), appended at the tail
so the static prompt prefix stays cache-stable. The answer ``ToolMessage`` is
never touched — the announcer only *adds* a message (citation fidelity; see the
orchestrator ``AGENTS.md`` § Pitfalls #5).

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
import re
from collections.abc import Sequence
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
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

# Audit markers + id prefix for the orchestrator-owned fallback ``Sources:``
# block (the ``after_agent`` hook appends it only when the planner omits its own).
# ``sta_generated`` flags a system-generated message in history; the deterministic
# id (content-digested) makes a replay idempotent under ``add_messages`` and lets
# the fake-stream's chunks share the committed message's id for frontend dedup.
_SOURCES_BLOCK_MARKER = "sta_generated"
_SOURCES_BLOCK_VALUE = "sources_block"
_SOURCES_BLOCK_ID_PREFIX = "sta-sources-block"
_SOURCES_BLOCK_HEADER = "**Sources**:"

# Line-anchored detection of a planner-emitted sources header. Matches at a line
# start (markdown allows ≤3 leading spaces): an optional ATX heading marker and/or
# emphasis wrapping the *plural* word "Sources", terminated by a colon OR end of
# line (the heading / bare-header forms). Requiring the colon-or-EOL terminator
# keeps prose ("These sources are great", "see the sources: below") and the
# singular "Source code:" from matching; the line anchor keeps a mid-sentence
# "sources:" or an inline ``[N]`` marker from matching.
_SOURCES_HEADER_RE = re.compile(
    r"""^[ \t]{0,3}              # up to 3 leading spaces (markdown)
        (?:\#{1,6}[ \t]*)?        # optional ATX heading marker
        (?:\*{1,3}|_{1,3})?       # optional opening emphasis
        [ \t]*
        sources                   # the literal word, plural
        (?:\*{1,3}|_{1,3})?       # optional closing emphasis
        [ \t]*
        (?::|$)                   # a colon, or end of line (heading / bare form)
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def _extract_title_url(source: dict[str, Any]) -> tuple[str, str]:
    """Pull a display ``(title, url)`` from one ``ka_sources`` row.

    Single source of truth for how a source dict renders, shared by the
    ``before_model`` announcer note and the ``after_agent`` fallback block so the
    two cannot drift: a blank/absent title falls back to ``"Untitled source"`` and
    the url is whitespace-stripped (empty when absent).
    """
    title = (source.get("title") or "").strip() or "Untitled source"
    url = (source.get("url") or "").strip()
    return title, url


def format_sources_block(ka_sources: Sequence[dict[str, Any]]) -> str:
    """Render ``ka_sources`` as a single-line markdown ``**Sources**:`` block.

    Produces ``**Sources**:`` followed by the ``[N] [title](url)`` markdown links
    **comma-separated on one line**, numbered by position (1-based) so the markers
    match the references panel. A source with no url renders as a bare ``[N] title``
    (never a dangling ``[title]()`` link). Returns ``""`` for an empty list — no
    sources means no block, never a lone header. The shape matches the canonical
    form the planner is told to emit, so this fallback and a planner-authored block
    are indistinguishable to the frontend.
    """
    sources = list(ka_sources or [])
    if not sources:
        return ""
    items = []
    for number, source in enumerate(sources, start=1):
        title, url = _extract_title_url(source)
        items.append(f"[{number}] [{title}]({url})" if url else f"[{number}] {title}")
    return f"{_SOURCES_BLOCK_HEADER} " + ", ".join(items)


def detect_sources_header(text: str) -> bool:
    """Return whether ``text`` already contains a line-anchored ``Sources:`` header.

    Case-insensitive; recognises the plain, italic (``*Sources*:``), bold
    (``**Sources**:``), and heading (``## Sources``) forms, plus a bare
    ``Sources`` header line. Deliberately does NOT match a mid-sentence
    "sources:", the singular "Source", or an inline ``[N]`` marker — so a normal
    cited answer without a real block is correctly seen as header-absent.
    """
    return bool(_SOURCES_HEADER_RE.search(text or ""))


def _extract_message_text(content: Any) -> str:
    """Flatten an ``AIMessage.content`` to plain text for regex/return.

    ``content`` may be a ``str`` or a list of content blocks (provider-dependent);
    only the textual parts are concatenated, so detection and the committed block
    work regardless of the shape (see the channels skill — content may be blocks).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _sources_block_id(block: str) -> str:
    """Deterministic id for a fallback block — content-digested so a replay
    re-mints the same id (``add_messages`` de-dups) and the fake-stream's chunks
    can carry it for frontend single-render."""
    digest = hashlib.sha256(block.encode("utf-8")).hexdigest()[:12]
    return f"{_SOURCES_BLOCK_ID_PREFIX}::{digest}"


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
            title, url = _extract_title_url(source)
            lines.append(f"[{number}] [{title}]" + (f"({url})" if url else ""))
        body = (
            "These sources came from the knowledge sub-agent. Cite a knowledge fact inline with the bare marker [N] shown here. "
            "If you cite any of them, end your reply with a block headed exactly `**Sources**:` "
            "(that precise wording, no substitute heading), listing the [N] you used as "
            "comma-separated markdown links — [N] [title](url), [N] [title](url) — copying the "
            "titles and urls below - Use [N] value FROM this mapping for inline references) "
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

    # ------------------------------------------------------------------ after_agent: Sources-block ownership

    def after_agent(self, state: KnowledgeBridgeState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Guarantee a ``Sources:`` block on a KA-grounded final answer (sync path)."""
        return self._maybe_sources_fallback(state)

    async def aafter_agent(self, state: KnowledgeBridgeState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Async counterpart: fake-stream the fallback block, then commit it.

        The streamed chunks carry the committed message's id (``GenericFakeChatModel``
        propagates it), so the frontend's ``seen_message_ids`` dedup renders the
        block exactly once — token stream, no duplicate whole-message.
        """
        update = self._maybe_sources_fallback(state)
        if update is not None:
            message = update["messages"][0]
            await self._fake_stream_block(_extract_message_text(message.content), message.id)
        return update

    def _maybe_sources_fallback(self, state: KnowledgeBridgeState) -> dict[str, Any] | None:
        """Append a deterministic ``Sources:`` block iff the final answer is a real
        KA-grounded generation that did not already emit one.

        Guards, all required: the last message is a final generation (an
        ``AIMessage`` with non-empty text and no ``tool_calls``); ``ka_sources`` is
        non-empty (the answer actually used the knowledge sub-agent); and the answer
        does not already carry a ``Sources:`` header (D1 — the model's own block is
        trusted and left untouched). Returns ``None`` when any guard fails — a
        refused turn (empty ``ka_sources``), a general-knowledge answer, a mid-loop
        tool call, an empty/non-AI last message, or an answer that already cites
        its sources. The fallback lists every ``ka_sources`` row in order: a planner
        that omitted the block likely omitted the inline ``[N]`` too, so listing all
        is the safe default (accepted numbering trade-off)."""
        messages: Sequence[AnyMessage] = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage) or getattr(last, "tool_calls", None):
            return None
        text = _extract_message_text(last.content)
        if not text.strip():
            return None
        sources = list(state.get("ka_sources") or [])
        if not sources:
            return None
        if detect_sources_header(text):
            return None
        block = format_sources_block(sources)
        message = AIMessage(
            content=block,
            id=_sources_block_id(block),
            additional_kwargs={_SOURCES_BLOCK_MARKER: _SOURCES_BLOCK_VALUE},
        )
        return {"messages": [message]}

    @staticmethod
    async def _fake_stream_block(block: str, message_id: str | None) -> None:
        """Fake-stream ``block`` token-by-token so the frontend renders it like a
        normal generation. ``GenericFakeChatModel`` propagates the input message id
        onto every streamed chunk, so the streamed chunks and the committed message
        (same id) dedupe to a single render. The chunks surface on
        ``stream_mode="messages"`` because the model call runs inside this exit
        node's execution context; the generator is consumed only for that
        side-effect — its content is discarded (the committed message carries it)."""
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        fake = GenericFakeChatModel(messages=iter([AIMessage(content=block, id=message_id)]))
        async for _ in fake.astream("go"):
            pass

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
            "documents, forward the search to the `knowledge_agent` subagent — do not answer "
            "from general knowledge instead."
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


class PlannerModelResolver:
    """Resolve the planner model before compiling the deepagents graph."""

    def __init__(
        self,
        *,
        model_override: str | BaseChatModel | None,
        model_factory: Callable[..., BaseChatModel],
    ) -> None:
        self._default_model: str | BaseChatModel | None = model_override
        self._model_factory = model_factory

    def resolve(self, runtime_model_config: Mapping[str, Any] | None) -> str | BaseChatModel:
        """Return the model object/string passed into ``create_deep_agent``.

        A pre-built ``model_override`` instance is returned untouched (a test
        seam). A runtime model config builds a fresh model from it; otherwise the
        default planner model is built once and memoized.
        """
        if runtime_model_config:
            logger.info("Orchestrator planner runtime model override selected: %s", redact_model_config(runtime_model_config))
            return self._model_factory(**runtime_model_config)

        if self._default_model is None:
            self._default_model = self._model_factory()
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
    enable_memory: bool = False,
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

packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/prompts/orchestrator_grounding.py
----
"""Grounding, clarification, general-knowledge, and deepagents auto-tool guidance.

These sections describe the planner's *behavior* — when to delegate, when to
answer directly, when to ask a follow-up — without depending on which tools
happen to be bound at runtime. The planner uses its own LLM capability for
clarification and general-knowledge answers; there is no ``clarify_user`` or
``general_knowledge`` tool to delegate to.
"""

_GROUNDING_SECTION = """
- Specialist sub-agents own their domain. When an available sub-agent's
  description matches the domain of the question, treat that sub-agent as the
  authoritative source of truth for that domain: delegate there and ground your
  answer in its output. Do not answer a domain question from your own general
  knowledge when a matching specialist exists, and do not let one specialist's
  output override another specialist on that other's home domain.
- Internal / company questions -> if a matching sub-agent is available to your `task`
  tool, delegate there first; otherwise explain that the relevant internal capability is
  unavailable in this session and ask for the missing scope if needed.
- General / public questions (programming, translation, public concepts) → answer
  directly from your own knowledge.
- Never fabricate. If you don't have the information, say so and offer to clarify
  or delegate.
"""

_CLARIFICATION_SECTION = """
Ask ONE concise clarifying question only when the user's intent is genuinely
ambiguous:
- The query could plausibly mean two different things.
- Required context (entity name, time range, scope) is missing.
- The user's preferred channel is unclear (internal docs vs. general knowledge).

Format: a single short question with 2-3 concrete options when possible, in the
user's language. Do NOT use clarification as a default opener — decide when you can.
"""

_GENERAL_KNOWLEDGE_SECTION = """
For non-company questions you can answer directly, use your own knowledge:
- Code → respond code first; explain only if non-obvious.
- Translation → translate in the user's target language.
- Text generation / reformulation → produce the requested output.
- Public technical explanations → concise, bullets over paragraphs.

Reply in the user's language. Skip preambles. Prefer bullets over paragraphs.
Label widely-accepted general knowledge with ``[GEN]`` when the user asks a
factual question and you are answering from training rather than from a
sub-agent's retrieval.
"""

_DEEPAGENTS_TOOL_GUIDELINES = """
Your `task`, `write_todos`, and filesystem tools are documented in their own
sections later in this prompt. The orchestrator-specific rules that override
that generic guidance:
- Use `write_todos` only for complex tasks that involve coordinating
  multiple specialized sub-agents or tools in sequence or in parallel — example:
  gather status from one capability, map dependencies with another, then
  synthesize a single answer. A single delegation or a direct answer never needs
  a todo list.
- The generic `task` advice ("don't use it for trivial tasks or simple
  lookups") applies to general-purpose work, NOT to internal / company
  questions: sub-agents are your only access to internal systems and documents.
  Even a one-line internal lookup (a ticket status, an application record, a
  person) goes through the matching sub-agent — never answered from your own
  knowledge because the lookup feels too small to delegate.
"""

_UNCERTAINTY_SECTION = """
When you cannot answer — the entity, document, or fact is absent from every
sub-agent result and from your own knowledge — say so explicitly and FIRST,
before any speculation. Lead with the negative result, then offer leads under a
clearly labelled hint line. Never bury a "not found" under a paragraph of
hypotheses, and never present a guess as if it were a retrieved fact.

Format:
**`<thing>`** doesn't appear to exist / could not be found in the available sources.

Hints:
- <closest related fact, adjacent entity, or where the user might look next>
"""

_SUBAGENT_TASKING_SECTION = """
Use `task(subagent_type=...)` only to delegate to a specialized sub-agent
available to your `task` tool.

Choosing a sub-agent:
- First, decompose before you delegate. Name the dimensions the question could
  hold (e.g. technical/structural, functional/business, incident or ticket
  history, procedural/how-to) and list EVERY available sub-agent relevant to any
  of them. Only then plan the delegations: one when a single agent covers every
  dimension, several — in parallel where the legs are independent — when the
  question genuinely spans dimensions. Do not stop at the first matching agent
  while a dimension the question raises is still uncovered.
- Tell apart two ways a message multiplies work. FACETS of one topic are
  different angles on the SAME subject (technical/structural vs functional/
  business) — cover each and synthesize one combined answer (the facet rule
  below). ORTHOGONAL asks are two or more independent questions in one message
  with no shared subject (e.g. "how do I install X?" alongside "what are the
  products of company Y?"). They are not facets of one thing: give EACH its own
  scoped `task` brief and emit them in the SAME turn so they run in parallel —
  even when they route to the same sub-agent, that is one parallel call per ask,
  never a single merged brief. Then synthesize the results into one reply. "A
  single agent covers it" decides a single question; it never licenses
  serializing independent asks or folding them into one brief.
- When more than one available agent could fit the same dimension, pick the one
  whose description best matches that need.
- The agent whose domain matches the question is the source of truth for that
  domain. Don't override its answer with another agent's output or your own
  general knowledge on its home domain.
- When both a documentation/knowledge agent and a live-system specialist match
  the question, the live specialist is the source of truth — query it first.
  Documentation is a written snapshot of what someone once recorded, so it can
  be stale; use it to enrich or investigate further, and flag documented
  information as possibly dated when it conflicts with or extends live results.
- A single question can raise more than one facet, each owned by a different
  agent. An open entity-identity or overview question — "what is X / tell me
  about X / describe X" about an application or system, INCLUDING a bare AP code
  ("what is APxxxxx") — has both a technical/structural facet (what it is, how
  it's built, what it connects to) and a functional/business facet (what it is
  for, what it does). Resolving an AP code to its record answers only the
  technical half. First decide how many facets the question actually raises,
  then act:
  - Two genuine facets (the default for "what is application X") → cover BOTH and
    synthesize one combined answer. Query the live/structural source of truth
    first, then the documentation source for the functional side — and flag
    documented detail as possibly stale or wrong, especially where it conflicts
    with the live result.
  - One dimension only — a question that explicitly scopes to a single aspect
    ("what does X depend on", "is X still active", "who owns X", a purely
    functional how-to) → search only that dimension; do NOT fan out. Only if it
    comes back empty should you check whether the other facet holds anything
    (the routing-signal rule below).
- If its result is weak, partial, or empty, treat that as a routing signal, not
  a verdict — don't stop at "not found". A question can be framed like one domain
  yet have its real answer in another: a "how does X work / how is it set up /
  what's the procedure" question routed to a live-system specialist often has its
  substance in internal documentation instead, and vice-versa. Before concluding
  nothing exists, proactively make ONE re-route to the better-fit complementary
  agent — pick it from the available roster by which description matches the
  actual need, not the surface framing of the question.
- When the complementary attempt still doesn't answer (or a re-route genuinely
  isn't warranted), don't end on a bare "not found": give the user a concrete,
  actionable hint — which other capability or angle would likely surface it, or
  how to rephrase or scope the question so a specific agent can find it. A useful
  next step beats a dead end.
- If a sub-agent returns no data twice for the same need, stop re-trying it:
  rephrasing the same brief a third time almost never helps. Switch to another
  agent or surface the empty result to the user.
- If repeated delegation isn't converging on a good answer, stop: tell the user
  what you tried and ask how they'd like to proceed.
- You don't have to exhaust every angle up front: when the source-of-truth
  agent's answer covers every facet the question raises, return it and, only when
  deeper digging into an answered facet might genuinely help,
  offer it as a follow-up rather than spending more delegations now. This is
  about not re-deepening a facet that is already answered — it never licenses
  skipping a second facet the question clearly raises.

When writing the `task` prompt:
- Do not forward the raw user message unless it is already a complete scoped task.
- Rewrite the task as a standalone brief for the selected subagent.
- Include only: objective, relevant entities/IDs, scope/time/env constraints,
  known context, and expected output.
- Strip unrelated conversation, routing rationale, hidden/system instructions,
  credentials, tool traces, and unrelated capabilities.
- If a term in the request is overloaded across domains (e.g. "agent" = AI
  agent vs. log agent vs. support contact; an abbreviation that could expand
  several ways), put the disambiguating context you have — the intended domain,
  system, or scope from prior turns — into the task brief so the sub-agent
  retrieves the right sense. If you genuinely cannot disambiguate, say so in the
  brief so the sub-agent can flag it or ask, rather than guessing.
- Ask one clarification instead of delegating if required context is missing.
"""

_KNOWLEDGE_AGENT_TASKING_RULE = """
The Knowledge Agent answers from internal, human-written documentation — it is
the source of truth for internal documentation, and the best-effort complement
everywhere else. That documentation is a treasure chest: broad and rich, it may
hold something on almost any internal topic — but it is a written snapshot, so
parts of it can be stale. A domain specialist queries the live system; the
Knowledge Agent retrieves what someone once wrote about it. That is why the
specialist wins on its own domain, and why documented context is an enrichment,
not a substitute.
Reach for the Knowledge Agent when the question needs internal documentation.
Ordering: when a domain specialist matches the question, always try that
source-of-truth specialist before the Knowledge Agent. Delegate to the
Knowledge Agent afterwards only to enrich the answer — when the specialist's
result leaves a real gap, or the user asked a genuinely deep or complex
question that warrants documented depth. When a domain specialist returns
nothing or only a partial answer, you may delegate to the Knowledge Agent to
fill the gap with documented context — flag it as documented (possibly dated)
information when it conflicts with or extends live results. It complements, it
never overrides a domain specialist on that specialist's own domain.
You don't have to dive deep every time: when the specialist's answer covers the
question, return it as-is. Offer a deeper dive (into internal documentation or
via another agent) as a follow-up question only when the answer is thin or the
question hints at more depth — not as a closing ritual on every reply.
"""

_FUTURE_CAPABILITIES_SECTION = (
    """In the future, Twin will evolve toward a dynamic multi-agent and multi-engine architecture with controlled access to systems and data."""
)

__all__ = [
    "_CLARIFICATION_SECTION",
    "_DEEPAGENTS_TOOL_GUIDELINES",
    "_FUTURE_CAPABILITIES_SECTION",
    "_GENERAL_KNOWLEDGE_SECTION",
    "_GROUNDING_SECTION",
    "_SUBAGENT_TASKING_SECTION",
    "_UNCERTAINTY_SECTION",
]

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
- When your reply cites knowledge sources, END it with a ``**Sources**:`` block and
  put nothing after it. Use EXACTLY this shape — the literal header ``**Sources**:``
  followed by the ``[N] [title](url)`` markdown links separated by commas on one
  line:

      **Sources**: [1] [title](url), [2] [title](url)

  The header MUST be exactly ``**Sources**:`` — keep that precise wording and styling
  for it, with no substitute heading of your own. The knowledge sub-agent is the
  source of truth for these sources; the ``<knowledge_sources>`` note(s) are only a
  reminder of the ``[N]`` → title/url mapping. Reuse each source's title and url
  under its ``[N]``; a source with no url is a bare ``[N] title``. List only the
  sources you actually used, ordered by number and separated by commas; titles and
  urls belong here and nowhere else. Put this block as the very last line(s) of your
  reply, with nothing after it. If a relayed sub-agent answer already shows
  ``[N](url)`` links or its own trailing sources list, normalise it to this form —
  bare ``[N]`` inline, one ``**Sources**:`` block at the very end.
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

packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/subagents/build_knowledge_agent_subagent.py
----
"""Knowledge Agent subagent build + recursion-budget knobs.

Owns the full ``knowledge_agent`` build: it bakes the orchestrator-owned
:class:`KnowledgeAgentConfig` knobs, constructs the raw KA graph, enriches the
advertised capability with its searchable corpora, and packages the result via
the shared :func:`as_subagent` helper. The catalog (:mod:`registry`) only
references :func:`build_knowledge_agent` from a ``SubagentSpec`` — it does not
build anything itself.

Unlike the incident and topology subagents, there is no separate raw-graph
builder: ``create_knowledge_agent`` already returns a graph that accepts
``{"messages": [...]}`` directly (the last ``HumanMessage`` is the query —
exactly what deepagents' ``task`` sends), so the build is a single function.

The search-depth / review-cap defaults and the recursion-limit derivation also
live here so the budget formula has one home.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING, Literal

from sta_agent_engine.agents.base.prompts.capability_definition import CapabilityDefinition
from sta_agent_engine.agents.knowledge_agent import CompressConfig, ExpandConfig

from ..build_context import BuildContext
from ..sources.twin_ka_entries import list_twin_ka_corpora
from ._packaging import BuiltSubagent, as_subagent


if TYPE_CHECKING:
    from sta_agent_engine.agents.knowledge_agent import KnowledgeAgentConfig, RetrieverEntry


SearchDepth = Literal["fast", "deep", "thorough"]

DEFAULT_KA_SEARCH_DEPTH: SearchDepth = "fast"
DEFAULT_KA_REVIEW_CAP: int = 2


def _compute_recursion_limit(search_depth: SearchDepth, review_cap: int) -> int:  # noqa: ARG001 - search_depth kept for call-site symmetry with the budget knobs
    """Derive a recursion_limit that covers ``review_cap`` outer iterations.

    KA outer loop is ~6 node transitions per iteration plus inner-expansion
    churn. Budget 10 transitions per review iteration + headroom — well below
    the deepagents default of 999 so we always tighten, never loosen.
    """
    return max(review_cap, 1) * 10 + 20


_KA_RECURSION_LIMIT: int = _compute_recursion_limit(DEFAULT_KA_SEARCH_DEPTH, DEFAULT_KA_REVIEW_CAP)

_KA_CAPABILITY = CapabilityDefinition(
    name="knowledge_agent",
    description=(
        "Delegate to the Knowledge Agent: it searches internal, human-written "
        "documentation and returns a synthesized, cited answer. Use it for "
        "documented procedures, runbooks, and how-tos, or to complement another "
        "specialist's answer with documented context."
    ),
    use_for=[
        "Company procedures and processes",
        "Documented application context: setup guides, configuration procedures, runbooks, how-tos",
        "The functional/business description of an application — what it is for and what it does — complementing topology's technical/structural view",
        "Complementing a specialist agent's answer with documented context",
        "Investigating documented hints when a specialist's answer lacks the detail the request needs",
    ],
    examples=[
        '"Où trouver le runbook de déploiement de l\'application X?"',
        '"Comment configurer un ecosystème Vault?"',
        '"What is the procedure to do X?"',
    ],
    note=(
        "Complementary source, not a system of record — documentation can be stale or wrong. "
        "For live incident or ticket records, current infrastructure topology, or "
        "people/org lookups, delegate to the matching domain specialist first when "
        "one is available; Only use this agent afterwards to enrich with documented context if the specialist's answer lacks the detail the request needs. "
        "When one request raises several independent documentation questions, spawn one task to this agent "
        "per question in parallel — each with its own focused brief — rather than merging unrelated topics "
        "into a single search. Describe the information need in the brief; do not name or hint which corpus "
        "to search — this agent picks the right corpus(es) itself."
    ),
)


def _orchestrator_ka_config(*, search_depth: SearchDepth = DEFAULT_KA_SEARCH_DEPTH, review_cap: int = DEFAULT_KA_REVIEW_CAP) -> KnowledgeAgentConfig:
    """Resolve a ``KnowledgeAgentConfig`` with the orchestrator-owned knobs baked in.

    Stamps ``mode="answer"`` (forced), the chosen ``search_depth`` /
    ``max_iterations``, ``subagent_mode=True``, the expansion/compression budgets,
    and ``streaming_enabled=False``. Streaming is disabled unconditionally in the
    subagent context: the KA runs behind the deepagents ``task`` boundary, so the
    orchestrator surfaces the final answer — a streamed sub-answer would leak
    intermediate (and at ``thorough`` depth, possibly-rejected) drafts the KA
    later replaces.
    """
    from sta_agent_engine.agents.knowledge_agent import KnowledgeAgentConfig

    resolved = KnowledgeAgentConfig.from_env(prefix="KA_")
    resolved.mode = "answer"
    resolved.search_depth = search_depth
    resolved.max_iterations = review_cap
    resolved.subagent_mode = True
    resolved.expand = ExpandConfig(enabled=False, structural_boundary_window=3, max_expansion_rounds=0)
    resolved.compress = CompressConfig(max_chars_per_group=280_000)
    resolved.streaming_enabled = False
    return resolved


def build_knowledge_agent(ctx: BuildContext) -> BuiltSubagent:  # noqa: ARG001 - ctx reserved for future persona-aware KA wiring
    """Build the Knowledge Agent subagent and enrich its capability with corpora."""
    from sta_agent_engine.agents.knowledge_agent import create_knowledge_agent

    graph = create_knowledge_agent(_get_ka_entries(), name="OrchestratorKnowledgeAgent", config=_orchestrator_ka_config())
    capability = replace(_KA_CAPABILITY, sources=list_twin_ka_corpora())
    # Pass the enriched ``capability`` (not the bare ``_KA_CAPABILITY`` skeleton) so the
    # corpora reach ``_compact_description`` — the compiled subagent's description is the
    # only channel the planner has for the KA's searchable corpora now that the
    # ``<capabilities>`` block is gone. ``BuiltSubagent.capability`` must match what the
    # planner is told, so both arguments use the same enriched capability.
    return BuiltSubagent(as_subagent(graph, capability, recursion_limit=_KA_RECURSION_LIMIT), capability)


@cache
def _get_ka_entries() -> list[RetrieverEntry]:
    """Build the KA retriever entries once per process.

    ``build_twin_ka_entries()`` opens ES retriever connections, so this is a
    network-touching call deferred to the function body — never at import.
    The result is cached so repeated KA builds (distinct rights slices, or a
    cache-disabled factory) reuse one set of retriever connections; the cache
    clears only on process restart, the same lifetime as the env-driven KA
    defaults baked into those entries.
    """
    from ..sources.twin_ka_entries import build_twin_ka_entries

    return build_twin_ka_entries()

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
    ' When the query states the "twin program", this is the ONLY corpus to search.'
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

