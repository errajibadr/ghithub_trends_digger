"""State bridge for topology artifacts returned by a trusted subagent."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain.agents.middleware import AgentState
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from sta_agent_engine.agents.orchestrator.topology_artifact_channels import (
    MAX_DESCRIPTION_CHARS,
    MAX_PAYLOAD_BYTES,
    TOPOLOGY_ARTIFACT_CONTEXT_KEY,
    TOPOLOGY_ARTIFACT_SCHEMA_VERSION,
    TOPOLOGY_ARTIFACTS_KEY,
    TopologyArtifact,
    TopologyArtifactBridgeChannels,
    TopologyArtifactContext,
    validate_topology_graph,
)

from .subagent_state_bridge import SubagentStateBridge


class TopologyArtifactBridgeState(AgentState, TopologyArtifactBridgeChannels):
    """Alfred state widened with accumulated topology artifacts and context."""


logger = logging.getLogger(__name__)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]+")


class TopologyArtifactBridgeMiddleware(SubagentStateBridge):
    """Own the topology state schema and remote-output projection contract.

    The bridge is injected only when a permitted subagent declares
    ``state_bridge=topology``.  ``before_agent`` captures parent-run ownership
    once; Deep Agents carries that context into each task runnable, while the
    remote boundary continues to send only the delegated messages over the
    network.
    """

    state_schema = TopologyArtifactBridgeState

    def before_agent(
        self,
        state: TopologyArtifactBridgeState,
        runtime: Runtime[Any],
    ) -> dict[str, TopologyArtifactContext]:
        """Capture the current turn and run so returned artifacts can be anchored."""
        return {TOPOLOGY_ARTIFACT_CONTEXT_KEY: self._invocation_context(state, runtime)}

    async def abefore_agent(
        self,
        state: TopologyArtifactBridgeState,
        runtime: Runtime[Any],
    ) -> dict[str, TopologyArtifactContext]:
        """Async counterpart of :meth:`before_agent`."""
        return {TOPOLOGY_ARTIFACT_CONTEXT_KEY: self._invocation_context(state, runtime)}

    @classmethod
    def project_remote_output(
        cls,
        result: Any,
        *,
        source_agent_id: str,
        invocation_id: str,
        parent_state: Any,
    ) -> dict[str, Any]:
        """Validate ``kg_subgraph`` and wrap it as one accumulated artifact.

        A missing/``None`` graph is a valid no-artifact response. Recoverable
        optional descriptions are normalized before strict validation. An
        unsafe graph is quarantined instead of reaching LangGraph state or the
        browser, while the remote agent's final text still crosses the boundary.
        """
        if not isinstance(result, Mapping):
            raise TypeError(f"Remote agent {source_agent_id} output must be a mapping")
        raw_graph = result.get("kg_subgraph")
        if raw_graph is None:
            return {}

        repair_counts: dict[str, int] = {}
        if cls._is_within_description_repair_budget(raw_graph):
            raw_graph, repair_counts = cls._sanitize_optional_node_descriptions(raw_graph)
        if repair_counts:
            logger.warning(
                "Normalized recoverable remote topology node descriptions",
                extra={
                    "source_agent_id": source_agent_id,
                    "invocation_id": invocation_id,
                    "repair_counts": dict(sorted(repair_counts.items())),
                },
            )

        try:
            graph = validate_topology_graph(
                raw_graph,
                boundary=f"Remote agent {source_agent_id} kg_subgraph",
            )
        except TypeError:
            logger.warning(
                "Dropped invalid remote topology artifact; preserving the remote text response",
                extra={
                    "source_agent_id": source_agent_id,
                    "invocation_id": invocation_id,
                },
            )
            return {}
        artifact: TopologyArtifact = {
            "schemaVersion": TOPOLOGY_ARTIFACT_SCHEMA_VERSION,
            "artifactId": invocation_id,
            "sourceAgentId": source_agent_id,
            "graph": graph,
        }
        context = cls._read_parent_context(parent_state)
        if anchor_message_id := context.get("anchorMessageId"):
            artifact["anchorMessageId"] = anchor_message_id
        if run_id := context.get("runId"):
            artifact["runId"] = run_id
        title = graph["metadata"].get("title")
        if isinstance(title, str) and title.strip() and len(title) <= 160:
            artifact["title"] = title
        return {TOPOLOGY_ARTIFACTS_KEY: [artifact]}

    @staticmethod
    def _invocation_context(
        state: TopologyArtifactBridgeState,
        runtime: Runtime[Any],
    ) -> TopologyArtifactContext:
        context: TopologyArtifactContext = {}
        messages: Sequence[Any] = state.get("messages") or []
        for message in reversed(messages):
            if isinstance(message, HumanMessage) and message.id:
                context["anchorMessageId"] = str(message.id)
                break
        execution_info = runtime.execution_info
        if execution_info is not None and execution_info.run_id:
            context["runId"] = str(execution_info.run_id)
        return context

    @staticmethod
    def _read_parent_context(parent_state: Any) -> TopologyArtifactContext:
        if not isinstance(parent_state, Mapping):
            return {}
        value = parent_state.get(TOPOLOGY_ARTIFACT_CONTEXT_KEY)
        if not isinstance(value, Mapping):
            return {}
        context: TopologyArtifactContext = {}
        anchor_message_id = value.get("anchorMessageId")
        if isinstance(anchor_message_id, str) and anchor_message_id.strip() and len(anchor_message_id) <= 256:
            context["anchorMessageId"] = anchor_message_id
        run_id = value.get("runId")
        if isinstance(run_id, str) and run_id.strip() and len(run_id) <= 256:
            context["runId"] = run_id
        return context

    @staticmethod
    def _is_within_description_repair_budget(value: Any) -> bool:
        """Never let repair turn an oversized/raw-invalid payload into an accepted one."""
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            return False
        return len(encoded) <= MAX_PAYLOAD_BYTES

    @classmethod
    def _sanitize_optional_node_descriptions(
        cls,
        value: Any,
    ) -> tuple[Any, dict[str, int]]:
        if not isinstance(value, Mapping):
            return value, {}
        raw_nodes = value.get("nodes")
        if not isinstance(raw_nodes, list):
            return value, {}

        sanitized_graph = dict(value)
        sanitized_nodes: list[Any] = []
        mutated = False
        repair_counts: dict[str, int] = {}
        for raw_node in raw_nodes:
            if not isinstance(raw_node, Mapping) or "description" not in raw_node:
                sanitized_nodes.append(raw_node)
                continue
            sanitized_node, actions = cls._sanitize_node_description(raw_node)
            sanitized_nodes.append(sanitized_node)
            mutated = mutated or bool(actions)
            for action in actions:
                repair_counts[action] = repair_counts.get(action, 0) + 1

        if not mutated:
            return value, {}
        sanitized_graph["nodes"] = sanitized_nodes
        return sanitized_graph, repair_counts

    @staticmethod
    def _sanitize_node_description(
        raw_node: Mapping[str, Any],
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        description = raw_node.get("description")
        if not isinstance(description, str):
            sanitized = dict(raw_node)
            sanitized.pop("description", None)
            return sanitized, ("dropped_non_text",)

        if (
            description.strip()
            and len(description) <= MAX_DESCRIPTION_CHARS
            and not _CONTROL_CHARACTERS.search(description)
        ):
            return dict(raw_node), ()

        normalized = _CONTROL_CHARACTERS.sub(" ", description).strip()
        normalized = " ".join(normalized.split())
        if not normalized:
            sanitized = dict(raw_node)
            sanitized.pop("description", None)
            return sanitized, ("dropped_blank",)

        actions: list[str] = []
        if normalized != description:
            actions.append("normalized_text")
        if len(normalized) > MAX_DESCRIPTION_CHARS:
            normalized = normalized[: MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
            actions.append("truncated")

        sanitized = dict(raw_node)
        sanitized["description"] = normalized
        return sanitized, tuple(actions)


__all__ = [
    "TopologyArtifactBridgeMiddleware",
    "TopologyArtifactBridgeState",
]
