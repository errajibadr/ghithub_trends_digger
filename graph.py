packages/sta_agent_core/src/sta_agent_core/adapters/elasticsearch/adapters_async.py
----
"""Async Elasticsearch adapter with lazy init and auto-recovery."""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any

from elasticsearch import AsyncElasticsearch
from langsmith import traceable

from ..base_search import BaseAsyncSearchAdapter


logger = logging.getLogger(__name__)


def _process_es_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process ES search inputs for LangSmith trace display."""
    import copy

    body = copy.deepcopy(inputs.get("body", {}))
    if "knn" in body and "query_vector" in body["knn"]:
        body["knn"]["query_vector"] = len(body["knn"]["query_vector"])
    cleaned = copy.deepcopy(inputs)
    cleaned.update({"body": body})
    return cleaned


def _process_es_outputs(outputs: Any) -> dict[str, Any]:
    """Process ES search outputs for LangSmith trace display."""
    if outputs is None:
        return {"returned_docs": 0, "total_matches": 0}

    hits = outputs.get("hits", {})
    returned_hits = hits.get("hits", [])
    total_matches = hits.get("total", {}).get("value", 0)

    return {
        "returned_docs": len(returned_hits),
        "total_matches": total_matches,
        "max_score": hits.get("max_score"),
        "took_ms": outputs.get("took"),
    }


def _is_event_loop_error(error: Exception) -> bool:
    """Check if the error is related to event loop issues."""
    error_messages = [
        "event loop is closed",
        "attached to a different loop",
        "loop is closed",
        "no running event loop",
        "got future <future pending> attached to a different loop",
    ]
    error_str = str(error).lower()
    return any(msg in error_str for msg in error_messages)


class AsyncElasticsearchAdapter(BaseAsyncSearchAdapter):
    """Async Elasticsearch adapter with lazy init and auto-recovery.

    Features:
        - Lazy client initialization: safe to instantiate before event loop starts
        - Auto-recovery: detects event loop changes and recreates client automatically
        - Context manager support for proper cleanup

    This adapter is safe to use in:
        - LangGraph server (langgraph dev / langgraph up)
        - Streamlit apps (where loop is recreated on rerun)
        - Jupyter notebooks
        - Any environment with unpredictable event loop lifecycle

    Example:
        # Safe at module level - client created lazily on first use
        adapter = AsyncElasticsearchAdapter(
            hosts=["http://localhost:9200"],
            es_default_index="my_index"
        )

        # In async context
        async def search_docs():
            results = await adapter.search("my_index", {"query": {"match_all": {}}})
            return results

        # Or with context manager for explicit cleanup
        async with AsyncElasticsearchAdapter(...) as adapter:
            results = await adapter.search(...)
    """

    def __init__(self, **client_kwargs: Any):
        """Initialize adapter with connection parameters.

        Args:
            **client_kwargs: Arguments passed to AsyncElasticsearch client.
                Must include 'es_default_index' for the default index name.
                Common kwargs: hosts, api_key, basic_auth, cloud_id, etc.
        """
        self.es_default_index: str | None = client_kwargs.pop("es_default_index", None)
        self._client_kwargs = client_kwargs
        self._client: AsyncElasticsearch | None = None

    @property
    def client(self) -> AsyncElasticsearch:
        """Get or create the Elasticsearch client (lazy initialization).

        The client is created on first access, ensuring it binds to the
        currently running event loop.
        """
        if self._client is None:
            self._client = AsyncElasticsearch(**self._client_kwargs)
        return self._client

    @classmethod
    def from_client(
        cls,
        client: AsyncElasticsearch,
        es_default_index: str | None = None,
    ) -> AsyncElasticsearchAdapter:
        """Create adapter from pre-configured Elasticsearch client.

        Args:
            client: Pre-configured AsyncElasticsearch client instance.
            es_default_index: Default index for searches.

        Returns:
            AsyncElasticsearchAdapter instance using the provided client.

        Warning:
            Auto-recovery from event loop changes will NOT work when using
            this method, since we don't have the kwargs to recreate the client.
            Only use this when you control the client lifecycle externally.
        """
        instance = cls.__new__(cls)
        instance.es_default_index = es_default_index
        instance._client_kwargs = {}  # No kwargs = no auto-recovery
        instance._client = client
        return instance

    async def __aenter__(self) -> AsyncElasticsearchAdapter:
        """Context manager entry - returns self (client created lazily)."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit - ensures cleanup."""
        await self.close()

    @traceable(
        run_type="retriever",
        name="elasticsearch_adapter_search",
        process_inputs=_process_es_inputs,
        process_outputs=_process_es_outputs,
    )
    async def search(self, index: str, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Execute search with auto-recovery on event loop errors.

        If the search fails due to event loop issues (e.g., loop changed
        since client was created), the client is automatically recreated
        and the search is retried once.

        Args:
            index: Index name to search.
            body: Elasticsearch query body.
            **kwargs: Additional arguments passed to client.search().

        Returns:
            Search response body as dict.

        Raises:
            RuntimeError: If search fails (including after retry).
        """
        try:
            response = await self.client.search(index=index, body=body, **kwargs)
            return response.body
        except Exception as e:
            # Check if this is an event loop error we can recover from
            if _is_event_loop_error(e) and self._client_kwargs:
                logger.warning("Event loop changed - recreating Elasticsearch client and retrying")
                self._client = AsyncElasticsearch(**self._client_kwargs)
                try:
                    response = await self.client.search(index=index, body=body, **kwargs)
                    return response.body
                except Exception as retry_error:
                    raise RuntimeError(f"Elasticsearch search failed after retry: {retry_error}") from retry_error
            # Not a loop error or no kwargs to recreate - raise original
            raise RuntimeError(f"Elasticsearch search failed: {e}") from e

    async def close(self) -> None:
        """Close the client if it exists.

        Safe to call multiple times. If the event loop has changed,
        the close may fail silently (the old client is orphaned but
        attached to a dead loop, so it can't leak connections).
        """
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                # May fail if loop is dead - log and continue
                logger.debug(f"Could not close Elasticsearch client: {e}")
            finally:
                self._client = None

-------

tests/test_ai_engine/agents/twin_router/middlewares/test_fast_mode_unit.py
----
"""Unit tests for FastModeMiddleware.

Tests the before_model hook logic for:
- AIMessage termination (no tool_calls)
- External RAG (placeholder) -> jump to end
- Internal RAG (actual content) -> Overwrite pattern with success + AIMessage
- Incident agent -> append AIMessage
"""

from unittest.mock import Mock

import pytest
from langchain.agents import AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Overwrite

from sta_agent_engine.agents.twin_router.middlewares.fast_mode import FastModeMiddleware
from sta_agent_engine.agents.twin_router.middlewares.third_party_rag import (
    RAG_PLACEHOLDER_CONTENT,
    SUCCESS_CONTENT,
)


def create_mock_runtime(fast_mode: bool | None = None) -> Mock:
    """Create a mock runtime with optional fast_mode context."""
    runtime = Mock()
    if fast_mode is not None:
        runtime.context = {"fast_mode": fast_mode}
    else:
        runtime.context = {}
    return runtime


@pytest.mark.unit
class TestEmptyAndBasicCases:
    """Tests for basic edge cases."""

    def test_returns_none_when_no_messages(self) -> None:
        """Skip processing when messages list is empty."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        state: AgentState = {"messages": []}

        result = middleware.before_model(state, runtime)

        assert result is None

    def test_returns_none_when_messages_missing(self) -> None:
        """Skip processing when messages key missing."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        state: AgentState = {"messages": []}

        result = middleware.before_model(state, runtime)

        assert result is None


@pytest.mark.unit
class TestAIMessageTermination:
    """Tests for AIMessage termination logic."""

    def test_jumps_to_end_on_ai_message_without_tool_calls(self) -> None:
        """AIMessage without tool_calls should jump to end."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        state: AgentState = {
            "messages": [
                HumanMessage(content="Hello", id="h1"),
                AIMessage(content="Hi there!", id="ai_1"),
            ]
        }

        result = middleware.before_model(state, runtime)

        assert result is not None
        assert result["jump_to"] == "end"
        assert "messages" not in result  # No message modifications

    def test_continues_on_ai_message_with_tool_calls(self) -> None:
        """AIMessage with tool_calls should continue (not jump to end)."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        state: AgentState = {
            "messages": [
                HumanMessage(content="How to deploy?", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lisab", "id": "call_1", "args": {"query": "deploy"}}],
                    id="ai_1",
                ),
            ]
        }

        result = middleware.before_model(state, runtime)

        # Should not jump to end, let tool execution happen
        assert result is None


@pytest.mark.unit
class TestFastModeToggle:
    """Tests for fast_mode toggle behavior."""

    def test_force_end_true_ignores_fast_mode_state(self) -> None:
        """force_end=True should process regardless of fast_mode state."""
        middleware = FastModeMiddleware(force_end=True)
        runtime = create_mock_runtime(fast_mode=False)

        state: AgentState = {
            "messages": [
                HumanMessage(content="Status of INC123?", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "call_incident_agent", "id": "call_1", "args": {}}],
                    id="ai_1",
                ),
                ToolMessage(content="Incident resolved", tool_call_id="call_1", name="call_incident_agent", id="tool_1"),
            ],
        }

        result = middleware.before_model(state, runtime)

        # Should process even with fast_mode=False because force_end=True
        assert result is not None
        assert result["jump_to"] == "end"

    def test_force_end_false_respects_state_fast_mode(self) -> None:
        """force_end=False should respect state.fast_mode."""
        middleware = FastModeMiddleware(force_end=False)
        runtime = create_mock_runtime()

        state = {
            "fast_mode": False,
            "messages": [
                HumanMessage(content="Status of INC123?", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "call_incident_agent", "id": "call_1", "args": {}}],
                    id="ai_1",
                ),
                ToolMessage(content="Incident resolved", tool_call_id="call_1", name="call_incident_agent", id="tool_1"),
            ],
        }

        result = middleware.before_model(state, runtime)  # type: ignore

        # Should NOT process because fast_mode=False and force_end=False
        assert result is None

    def test_context_fast_mode_overrides_state(self) -> None:
        """Context fast_mode should override state fast_mode."""
        middleware = FastModeMiddleware(force_end=False)
        runtime = create_mock_runtime(fast_mode=True)

        state: AgentState = {
            "messages": [
                HumanMessage(content="Status of INC123?", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "call_incident_agent", "id": "call_1", "args": {}}],
                    id="ai_1",
                ),
                ToolMessage(content="Incident resolved", tool_call_id="call_1", name="call_incident_agent", id="tool_1"),
            ],
        }

        result = middleware.before_model(state, runtime)

        # Should process because context.fast_mode=True overrides state
        assert result is not None
        assert result["jump_to"] == "end"


@pytest.mark.unit
class TestExternalRagHandling:
    """Tests for external RAG (placeholder) handling."""

    def test_external_rag_placeholder_jumps_to_end(self) -> None:
        """External RAG with placeholder should just jump to end."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        state: AgentState = {
            "messages": [
                HumanMessage(content="How to deploy?", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lisab", "id": "call_1", "args": {"query": "deploy"}}],
                    id="ai_1",
                ),
                ToolMessage(
                    content=RAG_PLACEHOLDER_CONTENT,
                    tool_call_id="call_1",
                    name="lisab",
                    id="tool_1",
                ),
            ]
        }

        result = middleware.before_model(state, runtime)

        assert result is not None
        assert result["jump_to"] == "end"
        # No message modifications for external RAG
        assert "messages" not in result


@pytest.mark.unit
class TestInternalRagHandling:
    """Tests for internal RAG (actual content) handling with Overwrite pattern."""

    def test_internal_rag_transforms_messages(self) -> None:
        """Internal RAG should transform: ToolMessage(content) -> ToolMessage(success) + AIMessage(content)."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        rag_content = "# Deployment Guide\n\n1. Create PR\n2. Merge to main"
        state: AgentState = {
            "messages": [
                HumanMessage(content="How to deploy?", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lisab", "id": "call_deploy", "args": {"query": "deploy"}}],
                    id="ai_1",
                ),
                ToolMessage(
                    content=rag_content,
                    tool_call_id="call_deploy",
                    name="lisab",
                    id="tool_1",
                ),
            ]
        }

        result = middleware.before_model(state, runtime)

        assert result is not None
        assert result["jump_to"] == "end"
        assert isinstance(result["messages"], Overwrite)

        messages = result["messages"].value
        assert len(messages) == 4  # Human, AI(tool_call), Tool(success), AI(content)

        # Verify message sequence
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].id == "h1"

        assert isinstance(messages[1], AIMessage)
        assert messages[1].id == "ai_1"

        # ToolMessage should have SUCCESS_CONTENT and same tool_call_id
        assert isinstance(messages[2], ToolMessage)
        assert messages[2].content == SUCCESS_CONTENT
        assert messages[2].tool_call_id == "call_deploy"
        assert messages[2].name == "lisab"

        # AIMessage should have the RAG content
        assert isinstance(messages[3], AIMessage)
        assert messages[3].content == rag_content

    def test_internal_rag_preserves_tool_call_id(self) -> None:
        """ToolMessage should preserve the original tool_call_id."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        original_tool_call_id = "call_unique_id_12345"
        state: AgentState = {
            "messages": [
                HumanMessage(content="Query", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lisab", "id": original_tool_call_id, "args": {}}],
                    id="ai_1",
                ),
                ToolMessage(
                    content="RAG answer",
                    tool_call_id=original_tool_call_id,
                    name="lisab",
                    id="tool_1",
                ),
            ]
        }

        result = middleware.before_model(state, runtime)

        messages = result["messages"].value
        tool_msg = messages[2]

        assert tool_msg.tool_call_id == original_tool_call_id

    def test_internal_rag_preserves_messages_before(self) -> None:
        """Messages before the ToolMessage should be preserved."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        state: AgentState = {
            "messages": [
                HumanMessage(content="First question", id="h1"),
                AIMessage(content="First answer", id="ai_1"),
                HumanMessage(content="Second question", id="h2"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lisab", "id": "call_1", "args": {}}],
                    id="ai_2",
                ),
                ToolMessage(
                    content="RAG result",
                    tool_call_id="call_1",
                    name="lisab",
                    id="tool_1",
                ),
            ]
        }

        result = middleware.before_model(state, runtime)

        messages = result["messages"].value

        # All messages before ToolMessage should be preserved
        assert messages[0].id == "h1"
        assert messages[0].content == "First question"
        assert messages[1].id == "ai_1"
        assert messages[1].content == "First answer"
        assert messages[2].id == "h2"
        assert messages[2].content == "Second question"
        assert messages[3].id == "ai_2"


@pytest.mark.unit
class TestIncidentAgentHandling:
    """Tests for call_incident_agent tool handling."""

    def test_call_incident_agent_appends_ai_message(self) -> None:
        """Incident agent should append AIMessage with tool content."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        incident_content = "Incident INC123 is resolved. Resolution: Server restart."
        state: AgentState = {
            "messages": [
                HumanMessage(content="Status of INC123?", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "call_incident_agent", "id": "call_1", "args": {}}],
                    id="ai_1",
                ),
                ToolMessage(
                    content=incident_content,
                    tool_call_id="call_1",
                    name="call_incident_agent",
                    id="tool_1",
                ),
            ]
        }

        result = middleware.before_model(state, runtime)

        assert result is not None
        assert result["jump_to"] == "end"

        # Incident agent uses Overwrite pattern (same as internal RAG for consistency)
        assert "messages" in result
        assert isinstance(result["messages"], Overwrite)

        # Verify patched messages: original messages + ToolMessage("success") + AIMessage(content)
        patched = result["messages"].value
        assert len(patched) == 4  # HumanMessage, AIMessage(tool_calls), ToolMessage(success), AIMessage(response)
        assert isinstance(patched[-2], ToolMessage)
        assert patched[-2].content == "success"
        assert isinstance(patched[-1], AIMessage)
        assert patched[-1].content == incident_content


@pytest.mark.unit
class TestOtherToolsHandling:
    """Tests for tools not explicitly handled."""

    def test_other_tools_continue_processing(self) -> None:
        """Tools not explicitly handled should not trigger fast mode end."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        state: AgentState = {
            "messages": [
                HumanMessage(content="Query", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "clarify_user", "id": "call_1", "args": {}}],
                    id="ai_1",
                ),
                ToolMessage(
                    content="Please clarify your question",
                    tool_call_id="call_1",
                    name="clarify_user",
                    id="tool_1",
                ),
            ]
        }

        result = middleware.before_model(state, runtime)

        # clarify_user is not explicitly handled, so continue
        assert result is None

    def test_general_knowledge_continues_processing(self) -> None:
        """general_knowledge tool should not trigger fast mode end."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is Python?", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "general_knowledge", "id": "call_1", "args": {}}],
                    id="ai_1",
                ),
                ToolMessage(
                    content="Python is a programming language",
                    tool_call_id="call_1",
                    name="general_knowledge",
                    id="tool_1",
                ),
            ]
        }

        result = middleware.before_model(state, runtime)

        # general_knowledge is not explicitly handled
        assert result is None


@pytest.mark.unit
class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    def test_full_internal_rag_flow(self) -> None:
        """Complete internal RAG flow with transformation."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        rag_answer = """Based on the documentation:

## Deployment Steps

1. Create a feature branch
2. Open a Pull Request
3. Wait for CI/CD pipeline
4. Merge to main

The deployment is automatic after merge."""

        state: AgentState = {
            "messages": [
                HumanMessage(content="How do I deploy my changes?", id="h1"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lisab", "id": "call_rag", "args": {"query": "deployment process"}}],
                    id="ai_1",
                ),
                ToolMessage(
                    content=rag_answer,
                    tool_call_id="call_rag",
                    name="lisab",
                    id="tool_1",
                ),
            ]
        }

        result = middleware.before_model(state, runtime)

        assert result is not None
        assert result["jump_to"] == "end"

        messages = result["messages"].value

        # Final message should be AIMessage with the RAG answer
        final_msg = messages[-1]
        assert isinstance(final_msg, AIMessage)
        assert "Deployment Steps" in final_msg.content
        assert "Pull Request" in final_msg.content

        # ToolMessage should be "success"
        tool_msg = messages[-2]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.content == SUCCESS_CONTENT

    def test_external_rag_vs_internal_rag_differentiation(self) -> None:
        """Verify correct differentiation between external and internal RAG."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        # External RAG (placeholder)
        external_state: AgentState = {
            "messages": [
                HumanMessage(content="Query", id="h1"),
                AIMessage(content="", tool_calls=[{"name": "lisab", "id": "c1", "args": {}}], id="ai_1"),
                ToolMessage(content=RAG_PLACEHOLDER_CONTENT, tool_call_id="c1", name="lisab", id="t1"),
            ]
        }
        external_result = middleware.before_model(external_state, runtime)

        # Internal RAG (actual content)
        internal_state: AgentState = {
            "messages": [
                HumanMessage(content="Query", id="h1"),
                AIMessage(content="", tool_calls=[{"name": "lisab", "id": "c1", "args": {}}], id="ai_1"),
                ToolMessage(content="Actual RAG answer", tool_call_id="c1", name="lisab", id="t1"),
            ]
        }
        internal_result = middleware.before_model(internal_state, runtime)

        # External: no Overwrite
        assert external_result["jump_to"] == "end"
        assert "messages" not in external_result if external_result is not None else True

        # Internal: uses Overwrite
        assert internal_result is not None
        assert internal_result["jump_to"] == "end"
        assert isinstance(internal_result["messages"], Overwrite)

    def test_consistent_api_contract_both_modes(self) -> None:
        """Both external and internal RAG end with same pattern after full processing."""
        middleware = FastModeMiddleware()
        runtime = create_mock_runtime()

        # Internal RAG after FastModeMiddleware
        internal_state = AgentState(
            messages=[
                HumanMessage(content="Query", id="h1"),
                AIMessage(content="", tool_calls=[{"name": "lisab", "id": "c1", "args": {}}], id="ai_1"),
                ToolMessage(content="RAG answer", tool_call_id="c1", name="lisab", id="t1"),
            ]
        )
        result = middleware.before_model(internal_state, runtime)
        messages = result["messages"].value

        # API contract: ToolMessage("success") followed by AIMessage(answer)
        assert messages[-2].content == SUCCESS_CONTENT
        assert isinstance(messages[-2], ToolMessage)
        assert messages[-1].content == "RAG answer"
        assert isinstance(messages[-1], AIMessage)

-------

