"""State bridge for topology artifacts returned by a trusted subagent."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain.agents.middleware import AgentState
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from sta_agent_engine.agents.orchestrator.topology_artifact_channels import (
    MAX_DESCRIPTION_CHARS,
    MAX_EDGE_PROPERTIES,
    MAX_EDGES,
    MAX_ID_CHARS,
    MAX_LABEL_CHARS,
    MAX_METADATA_PROPERTIES,
    MAX_NODE_PROPERTIES,
    MAX_NODES,
    MAX_PAYLOAD_BYTES,
    MAX_PROPERTY_ARRAY_ITEMS,
    MAX_PROPERTY_KEY_CHARS,
    MAX_PROPERTY_STRING_CHARS,
    MAX_TYPE_CHARS,
    MAX_WEIGHT,
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
_FORBIDDEN_PROPERTY_KEYS = frozenset({"__proto__", "constructor", "prototype"})
_TOP_LEVEL_KEYS = frozenset({"nodes", "edges", "metadata"})
_NODE_KEYS = frozenset({"id", "label", "type", "description", "properties"})
_EDGE_KEYS = frozenset({"source", "target", "label", "weight", "properties"})
_INVALID = object()


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
        representation defects are removed or normalized before strict
        validation. Repair is deliberately lossy: it may prune invalid optional
        data, but it never invents nodes, identifiers, or relationships. An
        irreparable graph is quarantined instead of reaching LangGraph state or
        the browser, while the remote agent's final text still crosses the
        boundary.
        """
        if not isinstance(result, Mapping):
            raise TypeError(f"Remote agent {source_agent_id} output must be a mapping")
        raw_graph = result.get("kg_subgraph")
        if raw_graph is None:
            return {}

        repair_counts: dict[str, int] = {}
        if cls._is_within_repair_budget(raw_graph):
            raw_graph, repair_counts = cls._sanitize_graph_lossy(raw_graph)
        if repair_counts:
            logger.warning(
                "Repaired remote topology artifact",
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
        if not graph["nodes"]:
            logger.warning(
                "Skipped empty remote topology artifact after repair" if repair_counts else "Skipped empty remote topology artifact",
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
    def _is_within_repair_budget(value: Any) -> bool:
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
    def _sanitize_graph_lossy(
        cls,
        value: Any,
    ) -> tuple[Any, dict[str, int]]:
        if not isinstance(value, Mapping):
            return value, {}

        repair_counts: dict[str, int] = {}
        unknown_top_level = len(set(value).difference(_TOP_LEVEL_KEYS))
        cls._increment(repair_counts, "dropped_unknown_field", unknown_top_level)

        raw_nodes = value.get("nodes", _INVALID)
        if not isinstance(raw_nodes, list):
            cls._increment(repair_counts, "defaulted_nodes")
            raw_nodes = []
        if len(raw_nodes) > MAX_NODES:
            cls._increment(repair_counts, "dropped_excess_node", len(raw_nodes) - MAX_NODES)

        nodes = []
        for raw_node in raw_nodes[:MAX_NODES]:
            node = cls._sanitize_node(raw_node, repair_counts)
            if node is not None:
                nodes.append(node)

        node_ids = {node["id"] for node in nodes}
        raw_edges = value.get("edges", _INVALID)
        if not isinstance(raw_edges, list):
            cls._increment(repair_counts, "defaulted_edges")
            raw_edges = []
        if len(raw_edges) > MAX_EDGES:
            cls._increment(repair_counts, "dropped_excess_edge", len(raw_edges) - MAX_EDGES)

        edges = []
        edge_fingerprints: set[tuple[str, str, str]] = set()
        for raw_edge in raw_edges[:MAX_EDGES]:
            edge = cls._sanitize_edge(
                raw_edge,
                node_ids=node_ids,
                edge_fingerprints=edge_fingerprints,
                repair_counts=repair_counts,
            )
            if edge is not None:
                edges.append(edge)

        raw_metadata = value.get("metadata", _INVALID)
        metadata = cls._sanitize_properties(
            raw_metadata,
            maximum=MAX_METADATA_PROPERTIES,
            repair_counts=repair_counts,
            default_action="defaulted_metadata",
        )
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata or {},
        }, repair_counts

    @classmethod
    def _sanitize_node(
        cls,
        value: Any,
        repair_counts: dict[str, int],
    ) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            cls._increment(repair_counts, "dropped_invalid_node")
            return None
        cls._increment(repair_counts, "dropped_unknown_field", len(set(value).difference(_NODE_KEYS)))

        node_id = value.get("id")
        if not cls._is_valid_identifier(node_id):
            cls._increment(repair_counts, "dropped_invalid_node")
            return None
        label = cls._sanitize_text(value.get("label"), maximum=MAX_LABEL_CHARS, repair_counts=repair_counts)
        if label is None:
            cls._increment(repair_counts, "dropped_invalid_node")
            return None

        node: dict[str, Any] = {"id": node_id, "label": label}
        for field, maximum in (("type", MAX_TYPE_CHARS), ("description", MAX_DESCRIPTION_CHARS)):
            if field not in value:
                continue
            text = cls._sanitize_text(value[field], maximum=maximum, repair_counts=repair_counts)
            if text is not None:
                node[field] = text

        if "properties" in value:
            properties = cls._sanitize_properties(
                value["properties"],
                maximum=MAX_NODE_PROPERTIES,
                repair_counts=repair_counts,
            )
            if properties is not None:
                node["properties"] = properties
        return node

    @classmethod
    def _sanitize_edge(
        cls,
        value: Any,
        *,
        node_ids: set[str],
        edge_fingerprints: set[tuple[str, str, str]],
        repair_counts: dict[str, int],
    ) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            cls._increment(repair_counts, "dropped_invalid_edge")
            return None
        cls._increment(repair_counts, "dropped_unknown_field", len(set(value).difference(_EDGE_KEYS)))

        source = value.get("source")
        target = value.get("target")
        if not cls._is_valid_identifier(source) or not cls._is_valid_identifier(target):
            cls._increment(repair_counts, "dropped_invalid_edge")
            return None
        if source not in node_ids or target not in node_ids:
            cls._increment(repair_counts, "dropped_dangling_edge")
            return None

        edge: dict[str, Any] = {"source": source, "target": target}
        edge_label: str | None = None
        if "label" in value:
            edge_label = cls._sanitize_text(value["label"], maximum=MAX_LABEL_CHARS, repair_counts=repair_counts)
            if edge_label is not None:
                edge["label"] = edge_label

        fingerprint = (source, target, edge_label or "")
        if fingerprint in edge_fingerprints:
            cls._increment(repair_counts, "dropped_duplicate_edge")
            return None
        edge_fingerprints.add(fingerprint)

        if "weight" in value:
            weight = value["weight"]
            if cls._is_finite_number(weight) and 0 <= weight <= MAX_WEIGHT:
                edge["weight"] = weight
            else:
                cls._increment(repair_counts, "dropped_invalid_weight")
        if "properties" in value:
            properties = cls._sanitize_properties(
                value["properties"],
                maximum=MAX_EDGE_PROPERTIES,
                repair_counts=repair_counts,
            )
            if properties is not None:
                edge["properties"] = properties
        return edge

    @classmethod
    def _sanitize_properties(
        cls,
        value: Any,
        *,
        maximum: int,
        repair_counts: dict[str, int],
        default_action: str = "dropped_invalid_properties",
    ) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            cls._increment(repair_counts, default_action)
            return None

        properties: dict[str, Any] = {}
        items = list(value.items())
        for index, (key, property_value) in enumerate(items):
            if len(properties) >= maximum:
                cls._increment(repair_counts, "dropped_excess_property", len(items) - index)
                break
            if not cls._is_valid_property_key(key):
                cls._increment(repair_counts, "dropped_invalid_property")
                continue
            normalized = cls._sanitize_property_value(property_value, repair_counts)
            if normalized is _INVALID:
                cls._increment(repair_counts, "dropped_invalid_property")
                continue
            properties[key] = normalized
        return properties

    @classmethod
    def _sanitize_property_value(
        cls,
        value: Any,
        repair_counts: dict[str, int],
    ) -> Any:
        if isinstance(value, list):
            if len(value) > MAX_PROPERTY_ARRAY_ITEMS:
                cls._increment(
                    repair_counts,
                    "dropped_excess_property_item",
                    len(value) - MAX_PROPERTY_ARRAY_ITEMS,
                )
            items = []
            for item in value[:MAX_PROPERTY_ARRAY_ITEMS]:
                normalized = cls._sanitize_scalar(item, repair_counts)
                if normalized is _INVALID:
                    cls._increment(repair_counts, "dropped_invalid_property_item")
                    continue
                items.append(normalized)
            return items
        return cls._sanitize_scalar(value, repair_counts)

    @classmethod
    def _sanitize_scalar(cls, value: Any, repair_counts: dict[str, int]) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            if len(value) <= MAX_PROPERTY_STRING_CHARS and not _CONTROL_CHARACTERS.search(value):
                return value
            normalized = " ".join(_CONTROL_CHARACTERS.sub(" ", value).split())
            cls._increment(repair_counts, "normalized_text")
            if len(normalized) > MAX_PROPERTY_STRING_CHARS:
                normalized = normalized[: MAX_PROPERTY_STRING_CHARS - 1].rstrip() + "…"
                cls._increment(repair_counts, "truncated_text")
            return normalized
        if cls._is_finite_number(value):
            return value
        return _INVALID

    @classmethod
    def _sanitize_text(
        cls,
        value: Any,
        *,
        maximum: int,
        repair_counts: dict[str, int],
    ) -> str | None:
        if not isinstance(value, str):
            cls._increment(repair_counts, "dropped_non_text")
            return None
        if value.strip() and len(value) <= maximum and not _CONTROL_CHARACTERS.search(value):
            return value

        normalized = " ".join(_CONTROL_CHARACTERS.sub(" ", value).split())
        if not normalized:
            cls._increment(repair_counts, "dropped_blank_text")
            return None
        if normalized != value:
            cls._increment(repair_counts, "normalized_text")
        if len(normalized) > maximum:
            normalized = normalized[: maximum - 1].rstrip() + "…"
            cls._increment(repair_counts, "truncated_text")
        return normalized

    @staticmethod
    def _is_valid_identifier(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip()) and len(value) <= MAX_ID_CHARS and not _CONTROL_CHARACTERS.search(value)

    @staticmethod
    def _is_valid_property_key(value: Any) -> bool:
        return (
            isinstance(value, str)
            and bool(value.strip())
            and len(value) <= MAX_PROPERTY_KEY_CHARS
            and not _CONTROL_CHARACTERS.search(value)
            and value not in _FORBIDDEN_PROPERTY_KEYS
        )

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        try:
            return math.isfinite(float(value))
        except OverflowError:
            return False

    @staticmethod
    def _increment(counts: dict[str, int], action: str, amount: int = 1) -> None:
        if amount > 0:
            counts[action] = counts.get(action, 0) + amount


__all__ = [
    "TopologyArtifactBridgeMiddleware",
    "TopologyArtifactBridgeState",
]
