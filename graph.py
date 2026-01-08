"""Tools module for the Twin Router.

Contains the four route tools that handle different types of user queries.

Factories:
    - `create_rag_tool()`: Configurable RAG tool (external or internal mode)
    - `create_gk_tool()`: General Knowledge tool with custom persona
    - `create_clarify_tool()`: Clarify tool with model-generated questions
    - `create_incident_tool()`: Incident agent tool with sync+async support

Default instances (backward compatible):
    - `lisab`: External RAG handoff (default behavior)
    - `general_knowledge`: GK with no custom persona
    - `clarify_user`: Clarify with default model
    - `call_incident_agent`: Incident agent with default config
"""

from .clarify import clarify_user, create_clarify_tool
from .gk import create_gk_tool, general_knowledge
from .incident import call_incident_agent, create_incident_tool
from .rag import create_rag_tool, lisab
from .rag_strategies import ExternalRagStrategy, InternalRagStrategy, RagStrategy


__all__ = [
    # RAG tools
    "lisab",
    "create_rag_tool",
    # RAG strategies
    "RagStrategy",
    "ExternalRagStrategy",
    "InternalRagStrategy",
    # Incident tools
    "call_incident_agent",
    "create_incident_tool",
    # Other tools
    "general_knowledge",
    "create_gk_tool",
    "clarify_user",
    "create_clarify_tool",
]

----------

"""Incident Agent tool for ServiceNow ticket management.

This tool invokes the mock incident agent subgraph for ticket-related queries.
Supports both sync and async execution via factory pattern.

Example:
    ```python
    # Create tool with default configuration
    incident_tool = create_incident_tool()

    # Create tool with custom description
    incident_tool = create_incident_tool(
        tool_name="servicenow_incident",
        custom_description="Custom description for incident queries",
    )
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command

from sta_agent_engine.agents.base.utils.tool_utils import clean_orphan_tool_calls

from ..mock_incident_agent import incident_agent as mock_incident_graph


if TYPE_CHECKING:
    from langchain.tools import ToolRuntime


logger = logging.getLogger(__name__)


# Tool description for LLM
INCIDENT_TOOL_DESCRIPTION = """Query ServiceNow for incident information.

Call this for:
- Incident lookups (INC numbers)
- Ticket status queries
- Priority information
- Creating or updating tickets

Args:
    query: The incident-related query

Returns:
    Response from incident agent with ticket information"""


def _prepare_messages_for_subgraph(runtime: ToolRuntime) -> list:
    """Prepare messages for subgraph invocation.

    Excludes the current tool call and cleans orphan tool calls.

    Args:
        runtime: Tool runtime with state access

    Returns:
        Cleaned list of messages for subgraph
    """
    messages = runtime.state.get("messages", [])
    messages_without_current = messages[:-1] if len(messages) > 1 else messages
    return clean_orphan_tool_calls(messages_without_current)


def _build_response_command(
    result: dict,
    tool_call_id: str,
    tool_name: str,
) -> Command:
    """Build Command response from subgraph result.

    Args:
        result: Result from incident subgraph
        tool_call_id: ID of the current tool call
        tool_name: Name of the tool for ToolMessage

    Returns:
        Command with state update
    """
    result_messages = result.get("messages", [])
    last_message = result_messages[-1] if result_messages else None

    logger.debug(f"[INCIDENT TOOL] Got response with {len(result_messages)} messages")

    return Command(
        update={
            "current_route": "INCIDENT",
            "messages": [
                ToolMessage(
                    content=last_message.content if last_message else "No response from incident agent",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                ),
            ],
        },
    )


def create_incident_tool(
    tool_name: str = "call_incident_agent",
    custom_description: str | None = None,
) -> BaseTool:
    """Factory for incident agent tool with sync+async support.

    Creates a tool that invokes the mock incident agent subgraph
    for ServiceNow ticket-related queries.

    Args:
        tool_name: Name for the tool (default "call_incident_agent")
        custom_description: Custom tool description for LLM

    Returns:
        StructuredTool with sync and async implementations

    Example:
        ```python
        # Default configuration
        incident_tool = create_incident_tool()

        # Custom name and description
        incident_tool = create_incident_tool(
            tool_name="servicenow_query",
            custom_description="Query ServiceNow for ticket information",
        )
        ```
    """
    description = custom_description or INCIDENT_TOOL_DESCRIPTION

    def incident_search_sync(query: str, runtime: ToolRuntime) -> Command:
        """Sync implementation of incident agent invocation.

        Args:
            query: The incident-related query
            runtime: Tool runtime for state and config access

        Returns:
            Command with state update containing response messages and current_route
        """
        logger.info(f"[INCIDENT TOOL] Invoking incident agent (sync) with query: {query[:100]}...")

        messages_for_subgraph = _prepare_messages_for_subgraph(runtime)

        result = mock_incident_graph.invoke(
            {"messages": messages_for_subgraph},
            checkpointer=True,
            config={"tags": ["incident_agent"]},
        )

        tool_call_id = runtime.tool_call_id or ""
        return _build_response_command(result, tool_call_id, tool_name)

    async def incident_search_async(query: str, runtime: ToolRuntime) -> Command:
        """Async implementation of incident agent invocation.

        Args:
            query: The incident-related query
            runtime: Tool runtime for state and config access

        Returns:
            Command with state update containing response messages and current_route
        """
        logger.info(f"[INCIDENT TOOL] Invoking incident agent (async) with query: {query[:100]}...")

        messages_for_subgraph = _prepare_messages_for_subgraph(runtime)

        result = await mock_incident_graph.ainvoke(
            {"messages": messages_for_subgraph},
            config={"tags": ["incident_agent"]},
        )

        tool_call_id = runtime.tool_call_id or ""
        return _build_response_command(result, tool_call_id, tool_name)

    return StructuredTool.from_function(
        name=tool_name,
        func=incident_search_sync,
        coroutine=incident_search_async,
        description=description,
    )


# Default instance for backward compatibility
call_incident_agent = create_incident_tool()
