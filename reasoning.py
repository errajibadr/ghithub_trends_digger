uv run python -c '
import os
from experiments.alfred_ag_ui.alfred_demo_agents import ALFRED_REMOTE_AGENT_DEFINITIONS, ALFRED_SUBAGENT_REGISTRY

agent_id = ""
definition = next(item for item in ALFRED_REMOTE_AGENT_DEFINITIONS if item.id == agent_id)
spec = ALFRED_SUBAGENT_REGISTRY[agent_id]

print("agent_id:", definition.id)
print("state_bridge:", definition.state_bridge)
print("bridge_class:", getattr(spec.bridge, "__name__", None))
print("graph_id:", definition.graph_id)
'

-----


import os

from langchain_core.messages import HumanMessage
from langgraph.pregel.remote import RemoteGraph

from experiments.alfred_ag_ui.alfred_remote_agents import _project_remote_result
from sta_agent_engine.agents.orchestrator.middlewares.topology_artifact_bridge import (
    TopologyArtifactBridgeMiddleware,
)

message = HumanMessage(
    id="topology-diagnostic",
    content=os.environ.get(
        "TOPOLOGY_TEST_PROMPT",
        "Return the dependency topology for a known application and materialize kg_subgraph.",
    ),
)

remote = RemoteGraph(
    os.environ["TOPOLOGY_GRAPH_ID"],
    url=os.environ["TOPOLOGY_URL"].rstrip("/"),
    api_key=os.environ.get("TOPOLOGY_API_KEY") or None,
    distributed_tracing=True,
)

result = remote.invoke({"messages": [message]})

if not isinstance(result, dict):
    raise TypeError(f"Unexpected output type: {type(result).__name__}")

kg = result.get("kg_subgraph")

print("remote_root_keys:", sorted(result))
print("kg_subgraph_present:", isinstance(kg, dict))

if isinstance(kg, dict):
    print("kg_keys:", sorted(kg))
    print("node_count:", len(kg.get("nodes") or []))
    print("edge_count:", len(kg.get("edges") or []))

# Teste ensuite exactement la projection utilisée par Alfred.
projected = _project_remote_result(
    result,
    parent_state={
        "messages": [message],
        "topology_artifact_context": {
            "anchorMessageId": message.id,
            "runId": "topology-diagnostic-run",
        },
    },
    name="ext_customer_topology",
    bridge=TopologyArtifactBridgeMiddleware,
    invocation_id="topology-diagnostic-invocation",
)

print("alfred_projection_keys:", sorted(projected))
print("artifact_count:", len(projected.get("topology_artifacts") or []))