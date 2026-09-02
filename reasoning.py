# LangSmith feedback and trace URLs

This document describes the backend integration required to:

- return the LangSmith `run_id` for each assistant response;
- collect thumbs up/down feedback;
- collect hallucination and completeness flags;
- return a link to the corresponding LangSmith trace.

The frontend must call the application backend. The backend owns the LangSmith
API key and calls LangSmith. Never expose `LANGSMITH_API_KEY` to the browser.

## End-to-end flow

1. The backend starts a remote graph execution.
2. LangGraph Agent Server creates the run and returns its `run_id` through
   `on_run_created`.
3. The backend returns that `run_id` to the frontend with the assistant response.
4. The frontend stores the `run_id` beside the assistant message.
5. Feedback and trace-link requests send that same `run_id` to the backend.

`thread_id` and `run_id` are different:

- `thread_id` identifies the conversation and is reused across turns.
- `run_id` identifies one graph execution and changes on every turn.

## Get the run ID from `RemoteGraph`

When using `RemoteGraph`, do not generate or pass `config["run_id"]`. Agent
Server creates the real run ID. Capture it with the synchronous
`on_run_created` callback passed to `RemoteGraph.astream`:

```python
from collections.abc import AsyncIterator
from typing import Any

from langgraph.pregel.remote import RemoteGraph


async def stream_remote_turn(
    remote_graph: RemoteGraph,
    *,
    input_state: dict[str, Any],
    thread_id: str,
) -> AsyncIterator[dict[str, Any]]:
    run_id: str | None = None
    run_id_sent = False

    def capture_run_id(created: Any) -> None:
        nonlocal run_id
        if isinstance(created, dict):
            value = created.get("run_id")
        else:
            value = getattr(created, "run_id", None)
        if value is not None:
            run_id = str(value)

    async for event in remote_graph.astream(
        input_state,
        config={"configurable": {"thread_id": thread_id}},
        on_run_created=capture_run_id,
    ):
        if run_id is not None and not run_id_sent:
            yield {"event": "run_created", "data": {"run_id": run_id}}
            run_id_sent = True

        yield {"event": "graph_event", "data": event}

    if run_id is None:
        raise RuntimeError("Agent Server did not return a run_id")
```

The frontend receives and stores an event such as:

```json
{
  "event": "run_created",
  "data": {
    "run_id": "a36092d2-4ad5-4fb4-9c0d-0dba9a2ed836"
  }
}
```

If the frontend does not need the ID before streaming completes, the backend
can include the captured value in the final response instead.

Do not search for "the latest run in the thread". Concurrent executions can
make that lookup return the wrong run.

## Application endpoints

Expose these two backend endpoints. The exact URL names may be adapted, but the
request and response contracts should remain stable.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/runs/{run_id}/feedback` | Create one feedback record |
| `GET` | `/api/runs/{run_id}/trace-url` | Return the LangSmith trace URL |

## Submit feedback

### Request

`POST /api/runs/{run_id}/feedback`

```json
{
  "key": "user_score",
  "score": 1,
  "comment": "Useful answer",
  "correction": null
}
```

Create a separate LangSmith feedback record for each selected dimension.

| Key | `0` means | `1` means | Failure value |
|---|---|---|---|
| `user_score` | Thumbs down | Thumbs up | `0` |
| `hallucination` | No hallucination | Hallucination detected | `1` |
| `completeness` | Incomplete answer | Complete answer | `0` |

> Important: `hallucination=1` is a failure flag. Its polarity is the opposite
> of `user_score` and `completeness`.

The backend must allow only these three keys and scores `0` or `1`.

Optional structured correction for a bad hallucination or completeness score:

```json
{
  "key": "hallucination",
  "score": 1,
  "comment": "The response invented the contract duration.",
  "correction": {
    "key_facts": ["The contract lasts 12 months."],
    "key_elements": ["contract duration"]
  }
}
```

### LangSmith SDK call

```python
from typing import Any

from langsmith import Client


ALLOWED_SCORES = {
    "user_score": {0, 1},
    "hallucination": {0, 1},
    "completeness": {0, 1},
}


def submit_feedback(
    client: Client,
    *,
    run_id: str,
    key: str,
    score: int,
    comment: str | None = None,
    correction: dict[str, Any] | None = None,
) -> str:
    if key not in ALLOWED_SCORES or score not in ALLOWED_SCORES[key]:
        raise ValueError("Unsupported feedback key or score")

    feedback = client.create_feedback(
        run_id=run_id,
        key=key,
        score=score,
        comment=comment,
        correction=correction,
    )
    return str(feedback.id)
```

Successful response:

```json
{
  "feedback_id": "62104630-c7f5-41dc-8ee2-0acee5c14224"
}
```

`create_feedback` creates a new record. The backend should deduplicate HTTP
retries, for example with an application idempotency key.

## Return the trace URL

### Request

`GET /api/runs/{run_id}/trace-url`

### Private URL — default

```python
from langsmith import Client


def get_private_trace_url(client: Client, run_id: str) -> str:
    run = client.read_run(run_id)
    return client.get_run_url(run=run)
```

Response:

```json
{
  "run_id": "a36092d2-4ad5-4fb4-9c0d-0dba9a2ed836",
  "url": "https://smith.langchain.com/o/.../projects/p/.../r/...?...",
  "visibility": "private"
}
```

This is a LangSmith browser URL requiring a LangSmith login. Do not construct
the URL manually; its organization and project identifiers come from the run.

The run may not be immediately readable when streaming finishes because trace
ingestion is asynchronous. The backend can return `202 Accepted` with a
`Retry-After` header so the frontend can poll for a short bounded period:

```json
{
  "status": "pending",
  "run_id": "a36092d2-4ad5-4fb4-9c0d-0dba9a2ed836"
}
```

### Public URL — explicit opt-in only

If public trace sharing is enabled by server configuration:

```python
url = client.share_run(run_id=run_id)
```

Return `"visibility": "public"`. Anyone with this URL can inspect the trace,
including captured user inputs and tool/model outputs. Public sharing must be a
server deployment policy, not a client-controlled request parameter. If public
sharing fails, fall back to the private URL.

## Configuration and workspace routing

```bash
LANGSMITH_API_KEY=ls_...
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# Required for an organization-scoped API key:
LANGSMITH_WORKSPACE_ID=<workspace-uuid>
```

For multiple workspaces, use a client scoped to the workspace authorized for
the current user/deployment:

```python
client = Client(workspace_id=workspace_id)
```

Do not trust an arbitrary workspace ID supplied by the browser. Derive or
validate it from authenticated backend context.

If the SDK cannot be used, the underlying LangSmith REST routes are:

| Method | LangSmith REST endpoint | Purpose |
|---|---|---|
| `POST` | `{LANGSMITH_ENDPOINT}/api/v1/feedback` | Create feedback; include `run_id`, `session_id`, `key`, and `score` |
| `GET` | `{LANGSMITH_ENDPOINT}/api/v1/runs/{run_id}` | Read the run |
| `PUT` | `{LANGSMITH_ENDPOINT}/api/v1/runs/{run_id}/share` | Create a public share token |

Send `X-Api-Key: <LANGSMITH_API_KEY>`. For an organization-scoped key, also
send the validated `X-Tenant-Id: <workspace_id>` header.

## Expected error handling

| Situation | Suggested response |
|---|---|
| Invalid feedback key or score | `400 Bad Request` |
| Run is still being ingested | `202 Accepted` with `Retry-After` |
| Run is absent from the authorized workspace | `404 Not Found` |
| LangSmith authentication failure | `502 Bad Gateway` |
| Temporary LangSmith failure | `503 Service Unavailable` |

## Implementation checklist

- Capture the server-generated `run_id` through `on_run_created`.
- Return the `run_id` to the frontend and store it with the assistant message.
- Never substitute `thread_id` for `run_id`.
- Never expose the LangSmith API key to the browser.
- Validate the three feedback keys and binary score values.
- Create one LangSmith feedback record per selected dimension.
- Keep trace links private unless public sharing is explicitly enabled.
- Validate workspace access on the backend.
- Treat immediate trace lookup failures as potentially retryable ingestion delay.

## Official references

- [RemoteGraph.astream](https://reference.langchain.com/python/langgraph/pregel/remote/RemoteGraph/astream)
- [LangSmith user feedback](https://docs.langchain.com/langsmith/attach-user-feedback)
- [LangSmith feedback data format](https://docs.langchain.com/langsmith/feedback-data-format)
- [LangSmith trace queries](https://docs.langchain.com/langsmith/export-traces)
- [LangSmith REST API](https://docs.langchain.com/langsmith/smith-api-ref)
