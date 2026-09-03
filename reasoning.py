# Federated Orchestrator

The **Federated Orchestrator** is a standalone Deep Agents graph that routes
work to remote LangGraph assistants. An operator supplies an explicit YAML or
JSON manifest containing remote deployment URLs and assistant IDs. At startup,
the orchestrator fetches the available Agent Cards, builds its subagent roster,
and delegates through LangGraph `RemoteGraph` calls.

It does not import the TWIN orchestrator and does not provide memory, skills,
habilitation, HITL, state bridges, or automatic deployment discovery. A remote
orchestrator may delegate to its own subagents without any special local logic.

!!! important "Explicit federation"
    The manifest is the complete roster. Agent Cards enrich and authorize the
    declared entries; they never add undeclared agents. Restart the service after
    changing the manifest or a card because each factory caches its first
    successfully compiled graph.

!!! danger "Do not serve the orchestrator and its target agents together"
    Run the Federated Orchestrator in a separate process from every local agent
    it calls. Do not register both the orchestrator and a target agent in the same
    `langgraph.json` or start them with the same `langgraph dev` server. A manifest
    URL that points back to the orchestrator's own server can create recursive
    self-calls, exhaust the local worker pool, and cause request starvation or a
    deadlock. Use two servers and two ports—for example, the target agent on
    `localhost:2024` and the Federated Orchestrator on `localhost:2025`.

## Prerequisites

You need:

- Python 3.12 or later;
- `sta-agent-engine` installed in your project;
- the LangGraph CLI with its in-memory development server extra;
- an LLM endpoint, API key, and model for the orchestration planner;
- at least one deployed or separately served local LangGraph agent and its
  generated assistant UUID.

For a new UV project:

```bash
mkdir federated-orchestrator
cd federated-orchestrator
uv init --python 3.12
uv add sta-agent-engine "langgraph-cli[inmem]"
```

For an existing project, only add the missing packages:

```bash
uv add sta-agent-engine "langgraph-cli[inmem]"
```

The minimum setup uses three files in your project directory:

```text
.
├── .env
├── federated_agents.yaml
└── langgraph_orchestrator.json
```

## Option 1: import the packaged factory directly

This is the recommended setup. It requires no Python wrapper file.

### 1. Create the manifest

Create `federated_agents.yaml`:

```yaml
agents:
  - url: https://agents.example.com
    assistant_id: 01234567-89ab-5def-8123-456789abcdef
    name: incident_specialist
    override_visibility: false
```

Replace the URL and UUID with values from the target server. The UUID must be
the target's `assistant_id`, not the `graph_id` name from its
`langgraph.json`. See [Discover remote assistant IDs](#discover-remote-assistant-ids)
before starting the orchestrator.

### Choose deployed or local target agents

The manifest can mix two kinds of target:

- **Deployed agent:** use the reachable HTTPS URL of its LangGraph deployment.
- **Local agent:** start that agent with a separate `langgraph dev` process and
  use its local base URL, such as `http://127.0.0.1:2024`.

To serve a local target agent, run this in the agent's own project or directory:

```bash
uv run langgraph dev \
  --config langgraph.json \
  --port 2024 \
  --no-browser
```

Discover its generated UUID:

```bash
curl --silent --request POST \
  --url http://127.0.0.1:2024/assistants/search \
  --header 'Content-Type: application/json' \
  --data '{"limit":100,"select":["assistant_id","graph_id","name"]}' \
  | jq '.[] | {assistant_id, graph_id, name}'
```

Then add it to `federated_agents.yaml`:

```yaml
agents:
  - url: http://127.0.0.1:2024
    assistant_id: <local-agent-uuid>
    name: local_specialist
    override_visibility: false
```

Keep this target server running and start the Federated Orchestrator in a
second terminal on port `2025`. `127.0.0.1` is correct when both processes run
on the same host. From a container, use a hostname or service address that is
reachable from the orchestrator container.

### 2. Configure `.env`

Create `.env`, or add only the missing variables to your existing file:

```dotenv
FEDERATED_ORCHESTRATOR_MANIFEST_PATH=./federated_agents.yaml

LLM_PROVIDER=custom
BASE_URL=https://llm.provider.example/v1
API_KEY=<planner-api-key>
MODEL=<planner-model>
```

`custom` is the default provider. If your existing `.env` already configures a
model supported by `sta-agent-engine`, only add
`FEDERATED_ORCHESTRATOR_MANIFEST_PATH`.

Named providers use the same prefix convention. For example,
`LLM_PROVIDER=llmaas` reads `LLMAAS_BASE_URL`, `LLMAAS_API_KEY`, and
`LLMAAS_MODEL`. Keep credentials in your deployment's secret manager rather
than committing `.env`.

If a remote manifest entry declares `api_key_env`, add that named variable to
the environment as well:

```dotenv
THIRD_PARTY_AGENT_API_KEY=<remote-agent-api-key>
```

### 3. Create `langgraph_orchestrator.json`

Create this file next to `.env`:

```json
{
  "python_version": "3.12",
  "dependencies": ["."],
  "graphs": {
    "federated_orchestrator": "sta_agent_engine.agents.federated_orchestrator.federated_orchestrator_catalog:make_federated_orchestrator"
  },
  "env": ".env"
}
```

The graph entry imports the packaged async factory directly. `dependencies` is
`["."]` because the project created in the prerequisites already declares
`sta-agent-engine` in its `pyproject.toml`.

### 4. Run the server

```bash
uv run langgraph dev \
  --config langgraph_orchestrator.json \
  --port 2025 \
  --no-browser
```

The first graph build reads the manifest, resolves remote credentials, fetches
the cards, and compiles the roster. Startup fails with a configuration error if
no agent is admitted.

### 5. Invoke the orchestrator

Discover the orchestrator's generated UUID:

```bash
curl --silent --request POST \
  --url http://127.0.0.1:2025/assistants/search \
  --header 'Content-Type: application/json' \
  --data '{"graph_id":"federated_orchestrator","limit":10,"select":["assistant_id","graph_id"]}' \
  | jq -r '.[0].assistant_id'
```

Use the returned UUID in a run request:

```bash
curl --request POST \
  --url http://127.0.0.1:2025/runs/wait \
  --header 'Content-Type: application/json' \
  --data '{
    "assistant_id": "<federated-assistant-uuid>",
    "input": {
      "messages": [{
        "role": "human",
        "content": "Ask the incident specialist to investigate the checkout errors."
      }]
    }
  }'
```

## Option 2: register your own factory

Use a local wrapper when application code should supply the manifest, inject an
explicit model, or provide a checkpointer or store. Create
`federated_orchestrator.py`:

```python
from pathlib import Path

from sta_agent_engine.agents.federated_orchestrator import (
    create_federated_orchestrator_factory,
)


MANIFEST_PATH = Path(__file__).with_name("federated_agents.yaml")

make_federated_orchestrator = create_federated_orchestrator_factory(
    MANIFEST_PATH,
)
```

Then point `langgraph_orchestrator.json` at that symbol:

```json
{
  "python_version": "3.12",
  "dependencies": ["."],
  "graphs": {
    "federated_orchestrator": "./federated_orchestrator.py:make_federated_orchestrator"
  },
  "env": ".env"
}
```

In this mode, `FEDERATED_ORCHESTRATOR_MANIFEST_PATH` is optional because the
wrapper captures the path. The LLM environment variables remain required when
no explicit `model=` is passed.

The factory also accepts an in-memory dictionary or YAML/JSON content:

```python
from sta_agent_engine.agents.federated_orchestrator import (
    create_federated_orchestrator_factory,
)


MANIFEST = {
    "agents": [
        {
            "url": "https://agents.example.com",
            "assistant_id": "01234567-89ab-5def-8123-456789abcdef",
            "name": "incident_specialist",
            "override_visibility": False,
        }
    ]
}

make_federated_orchestrator = create_federated_orchestrator_factory(MANIFEST)
```

A new factory instance reloads the manifest and Agent Cards. Repeated calls to
the same factory reuse its first successfully compiled graph.

## Manifest reference

The manifest accepts a dictionary, YAML or JSON content, or a filesystem path.
Its root must contain `agents` with between 1 and 20 entries.

```yaml
agents:
  - url: https://agents.example.com
    assistant_id: 01234567-89ab-5def-8123-456789abcdef
    name: incident_specialist
    description: >-
      Investigates operational incidents and explains likely causes. Delegate
      incident diagnosis, impact analysis, and remediation questions here.
    card_url: https://agents.example.com/custom-agent-card.json
    api_key_env: THIRD_PARTY_AGENT_API_KEY
    override_visibility: false
```

| Field | Required | Purpose |
|---|---:|---|
| `url` | yes | HTTP(S) base URL of the remote LangGraph deployment. |
| `assistant_id` | yes | Generated LangGraph assistant UUID, or a numeric legacy assistant ID. A `graph_id` name is rejected. |
| `name` | no | Stable `snake_case` routing name. Falls back to the card name, then a normalized `agent_<assistant_id>`. |
| `description` | no | Routing fallback used when an override admits an unavailable or unusable card. |
| `card_url` | no | Non-standard Agent Card URL. Defaults to `{url}/a2a/{assistant_id}/.well-known/agent-card.json`. |
| `api_key_env` | no | Name of the environment variable containing the remote API key. |
| `override_visibility` | no | Explicitly admit an agent whose card is absent, invalid, or not orchestrator-visible. Defaults to `false`. |

Unknown fields, duplicate deployment identities, credential-bearing URLs, and
routing-name collisions are rejected before compilation. A declared
`api_key_env` must exist and contain a non-empty value; the manifest never
accepts a raw API key. Relative manifest paths resolve from the directory where
the server process starts, so production deployments should prefer an absolute
path.

## Discover remote assistant IDs

Query each remote LangGraph server before curating the manifest:

```bash
curl --silent --request POST \
  --url https://agents.example.com/assistants/search \
  --header 'Content-Type: application/json' \
  --header 'x-api-key: <remote-api-key>' \
  --data '{"limit":100,"select":["assistant_id","graph_id","name","description"]}' \
  | jq '.[] | {assistant_id, graph_id, name, description}'
```

You may filter one system graph directly:

```bash
curl --silent --request POST \
  --url https://agents.example.com/assistants/search \
  --header 'Content-Type: application/json' \
  --header 'x-api-key: <remote-api-key>' \
  --data '{"graph_id":"topology","limit":10,"select":["assistant_id","graph_id","name"]}'
```

For system assistants registered from `langgraph.json`, the UUID is
deterministic while the graph key remains unchanged. User-created assistants
may use different UUIDs. The A2A Agent Card endpoint requires the concrete
assistant UUID even though LangGraph run endpoints may also accept a graph-name
alias.

Verify the card with the returned UUID:

```bash
curl --fail \
  --header 'x-api-key: <remote-api-key>' \
  https://agents.example.com/a2a/<assistant-uuid>/.well-known/agent-card.json
```

Omit the API-key header for a public deployment.

## Agent Card admission

The orchestrator fetches every declared card once with a shared async client, a
three-second timeout, concurrency capped at eight, and a 64 KiB response limit.
The optional remote API key is sent as `x-api-key`.

A structured profile encoded in `card.description` may provide:

- `description`;
- `scope`;
- `how_to_use`;
- `examples`;
- `freshness`;
- `visibility.orchestrator`.

`visibility.orchestrator: true` admits the agent. A missing, invalid, or hidden
card skips it unless the trusted operator sets `override_visibility: true` in
the manifest. With an override, routing text falls back from the card
description to the manifest description and then to a generic description.

`supportsA2A: false` does not block delegation because execution uses
`RemoteGraph`, not the A2A invocation protocol.

## Runtime contract and failure behavior

Each admitted deployment must accept a state containing `messages`. Delegation
sends a `HumanMessage` describing the task. The remote result must provide
either a final `AIMessage` in `messages` or a `structured_response`.

Remote calls are fail-soft and are not retried. A remote exception becomes a
sanitized tool result so the planner can continue; URLs, credentials, and
tracebacks are never exposed to the model. Operational tracebacks remain in
server logs. A planner run may make at most ten `task` calls.

Run the Federated Orchestrator in its own service process. This prevents a
local `RemoteGraph` target from calling back into the same server and blocking
on itself, and it isolates the process-global Deep Agents harness profiles.
OpenAI and Mistral planners expose only `task` and `write_todos`; other
providers may retain their default Deep Agents harness and should be verified
before production deployment.

## Appendix: production agent roster to curate

This appendix is the operator-maintained source for the production manifest.
The agent names and intended capabilities are prefilled below. Replace every
`TBD` with values discovered from the corresponding production deployment
before publishing or running the manifest.

| Routing name | Intended capability | Production URL | Assistant UUID | Card policy |
|---|---|---|---|---|
| `topology_agent` | Application and infrastructure topology, dependencies, hosting, and communication flows. | **TBD** | **TBD** | Card must set `visibility.orchestrator: true`. |
| `es_knowledge_agent` | Elasticsearch-backed document search and evidence-grounded knowledge answers. | **TBD** | **TBD** | Card must set `visibility.orchestrator: true`. |

Ready-to-curate YAML:

```yaml
agents:
  - name: topology_agent
    url: https://<topology-production-host>
    assistant_id: <topology-assistant-uuid>
    override_visibility: false

  - name: es_knowledge_agent
    url: https://<knowledge-production-host>
    assistant_id: <knowledge-assistant-uuid>
    override_visibility: false
```

If an approved production agent cannot publish a card, add a bounded routing
`description`, set its `card_url` if the standard URL is unavailable, and use
`override_visibility: true` only as an explicit operator decision.
