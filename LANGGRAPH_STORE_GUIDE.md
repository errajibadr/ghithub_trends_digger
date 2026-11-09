# LangGraph Store: quick guide

The **checkpointer** keeps short-term state for one thread. The **store** keeps
long-term memory across threads: a key-value store addressed by
`(namespace tuple, key)`, with optional semantic search.

## 1. In code (local / tests)

```python
from langgraph.store.memory import InMemoryStore  # prod outside the server: AsyncPostgresStore

store = InMemoryStore()  # no embeddings: put / get / search by namespace + filter
graph = builder.compile(checkpointer=checkpointer, store=store)
```

Use it from a node or a tool through the runtime:

```python
from langgraph.runtime import Runtime

async def remember(state, runtime: Runtime[Ctx]):
    ns = ("users", runtime.context.user_id, "memory")
    await runtime.store.aput(ns, "/prefs.md", {"content": "likes short answers", "kind": "preference"})
    item = await runtime.store.aget(ns, "/prefs.md")                         # item.value -> dict
    prefs = await runtime.store.asearch(ns, filter={"kind": "preference"}, limit=10)
    await runtime.store.adelete(ns, "/old.md")
```

Outside a node: `from langgraph.config import get_store; store = get_store()`.

## 2. On the deployed LangGraph platform

- **Do not pass `store=` to `compile()`.** The server injects its own
  Postgres-backed store; `runtime.store` is that store.
- Configure it in `langgraph.json` (both blocks optional):

```json
{
  "store": {
    "index": {"embed": "./src/my_pkg/embeddings.py:embeddings", "dims": 1024, "fields": ["content"]},
    "ttl": {"refresh_on_read": true, "default_ttl": 10080, "sweep_interval_minutes": 60}
  }
}
```

  `ttl` values are minutes (10080 = 7 days). Without `ttl`, items live forever.

- From outside the graph (BFF, scripts) use the SDK; namespaces are lists there:

```python
from langgraph_sdk import get_client

client = get_client(url=DEPLOYMENT_URL, api_key=API_KEY)
await client.store.put_item(["users", uid, "memory"], key="/prefs.md", value={"content": "..."})
await client.store.get_item(["users", uid, "memory"], key="/prefs.md")
await client.store.search_items(["users", uid], query="diet", limit=5)   # query needs an index
await client.store.list_namespaces(prefix=["users", uid])
await client.store.delete_item(["users", uid, "memory"], key="/prefs.md")
```

- **Lock it down with custom auth.** Without a store handler, any caller with a
  valid key can read any namespace:

```python
@auth.on.store()
async def own_namespace_only(ctx: Auth.types.AuthContext, value: dict):
    ns = tuple(value["namespace"])
    if ns[:2] != ("users", ctx.user.identity):
        raise Auth.exceptions.HTTPException(status_code=403, detail="forbidden")
```

## 3. Embeddings (optional)

Without `index`, the store still does put / get / prefix search / `filter=`.
Add `index` only when you need `search(..., query="natural language")`.

**Custom model or endpoint** (self-hosted vLLM, TEI, any OpenAI-compatible
`/v1/embeddings`). `embed` accepts `"path/to/file.py:name"` where `name` is a
LangChain `Embeddings` instance **or** an (async) function `list[str] -> list[list[float]]`.

`src/my_pkg/embeddings.py`, option A (LangChain client):

```python
import os
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model=os.environ["EMBEDDING_MODEL"],            # the served model name
    base_url=os.environ["EMBEDDING_BASE_URL"],      # e.g. https://<host>/v1
    api_key=os.environ.get("EMBEDDING_API_KEY", "none"),
    check_embedding_ctx_length=False,               # skip tiktoken: non-OpenAI model, no download
    request_timeout=30,                             # set here: an httpx client timeout is ignored
)
```

Option B (plain function, no LangChain client):

```python
import os
import httpx

async def aembed(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=30, verify=os.environ.get("SSL_CERT_FILE", True)) as c:
        r = await c.post(
            f"{os.environ['EMBEDDING_BASE_URL']}/embeddings",
            headers={"Authorization": f"Bearer {os.environ.get('EMBEDDING_API_KEY', 'none')}"},
            json={"model": os.environ["EMBEDDING_MODEL"], "input": texts},
        )
        r.raise_for_status()
        return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
```

Then `"embed": "./src/my_pkg/embeddings.py:aembed"`.

Same embeddings locally:

```python
store = InMemoryStore(index={"embed": embeddings, "dims": 1024, "fields": ["content"]})
```

Rules:

- `dims` must equal the model's output size. Changing model or dims means re-embedding.
- `fields` picks what gets embedded (`["$"]` = whole value). Per call:
  `put(..., index=False)` skips embedding, `index=["summary"]` overrides fields.
- `search(ns, query=...)` ranks by similarity; `filter=` still applies on top.

## 4. Tips

1. `search(("alice",))` is a **prefix** match: it reads every namespace below.
   Put the isolation boundary first: `("users", uid, "memory")`. Shared data gets
   its own root: `("agents", agent_id, ...)`.
2. Values must be dicts. Strings work in `InMemoryStore` until a filter runs, and the SDK rejects them.
3. `filter=` is exact match on value fields (plus `$eq $ne $gt $gte $lt $lte`); no glob, regex or `$in`.
4. Encode raw ids (emails, paths) before using them as namespace labels; the SDK rejects some chars like `.`.
5. Never log a namespace that contains a user id.
6. Deep Agents: mount memory with `CompositeBackend(routes={"/memory/": StoreBackend(namespace=...)})`;
   namespace `(uid, "memory")`, key `/AGENTS.md`, agent sees `/memory/AGENTS.md`.

---

# Reference: Store & Namespace Best Practices


Use this reference when designing LangGraph long-term memory, direct `Store`
access, Deep Agents `StoreBackend`, or a `CompositeBackend` route that mounts
memory into a filesystem-like path.

## Mental Model

LangGraph persistence has two different layers:

- **Checkpointer:** short-term, thread-scoped graph state and message history.
- **Store:** long-term, cross-thread memory for facts, preferences, files, or
  process state.

A Store row is addressed as:

```python
namespace = ("users", user_id, "memory")
key = "/preferences.md"
value = {"content": "...", "kind": "preference"}
```

The namespace is a tuple that controls isolation and prefix search. The key is
separate from the namespace. The value should be a JSON object / Python dict.

## Namespace Prefix Search

`store.search(namespace_prefix, ...)` uses a namespace prefix. It is not
necessarily an exact namespace match.

If Store contains:

```python
("alice", "memory")
("alice", "tasks")
("alice", "tasks", "archive")
```

then this searches all three:

```python
store.search(("alice",))
```

More specific prefixes narrow the result:

```python
store.search(("alice", "memory"))  # memory only
store.search(("alice", "tasks"))   # tasks and tasks/archive
```

To list namespaces rather than items:

```python
store.list_namespaces(prefix=("alice",))
```

SDK equivalents:

```python
await client.store.search_items(["alice"])
await client.store.list_namespaces(prefix=["alice"])
```

## Namespace Ordering

Order namespace segments by the broadest prefix you need to query safely.
Default rule:

```text
<hard isolation boundary>/<resource kind>/<subtype or producer>
```

Use user-first for private user memory:

```python
("users", user_id, "memory")
("users", user_id, "memory", agent_id)
```

If the store only contains per-user memory, the compact form is also fine and
matches common LangGraph examples:

```python
(user_id, "memory")
(user_id, "memory", agent_id)
```

For tenant-aware systems, tenant comes first:

```python
("tenants", tenant_id, "users", user_id, "memory")
("tenants", tenant_id, "agents", agent_id, "process_memory")
```

For private user memory plus global/shared agent process memory, use separate
namespace roots:

```python
("users", user_id, "memory")
("agents", agent_id, "process_memory")
```

This prevents one broad prefix from mixing private user memory with shared
process state.

Collection-first namespaces are valid when broad collection reads are
intentional:

```python
("memories", team_id)
("memories", team_id, agent_id)
```

But `store.search(("memories", team_id))` reads everything below that team
prefix. Do not use this shape for private memory unless team-wide sharing is
the intended access policy.

When splitting private memory by agent, prefer:

```python
("users", user_id, "memory", agent_id)
```

over:

```python
("users", user_id, agent_id, "memory")
```

when the common operations are user export/delete/search across all memory.

## Store Values And Filters

Store values should be dicts. The LangGraph SDK enforces object values, and
local `InMemoryStore` can behave inconsistently with non-dict values because
filtered search expects value fields.

Good:

```python
store.put(("users", "alice", "memory"), "/prefs.md", {"content": "...", "kind": "preference"})
```

Avoid:

```python
store.put(("users", "alice", "memory"), "/prefs.md", "plain string")
```

`filter={...}` checks fields inside each item value:

```python
store.search(("users", "alice", "memory"), filter={"kind": "preference"})
```

Filters are exact comparisons unless an explicit comparison operator is used.
They are not glob or regex patterns:

```python
store.search(("users", "alice", "memory"), filter={"path": "/notes/*"})  # exact value
```

Common comparison operators in local Store implementations:

```python
filter={"count": {"$eq": 3}}
filter={"count": {"$gt": 3}}
filter={"count": {"$gte": 3}}
filter={"count": {"$lt": 10}}
filter={"count": {"$lte": 10}}
filter={"kind": {"$ne": "temporary"}}
```

Nested dict filters match nested fields:

```python
filter={"metadata": {"kind": "task"}}
```

List filters compare list shape/value exactly. They are not `$in`.

## StoreBackend Path Model

Deep Agents `StoreBackend` maps Store rows to filesystem-like files. Keep the
namespace as the storage boundary and use keys as file paths.

For mounted memory:

```text
Store namespace:  (user_id, "memory")
Store key:        /AGENTS.md
Planner path:     /memory/AGENTS.md
```

Directory structure comes from key strings:

```text
/notes/today.md
/notes/archive/week-01.md
/scratch/draft.txt
```

Avoid duplicating the route/resource name in both namespace and key:

```text
Good:  namespace=(user_id, "memory"), key="/AGENTS.md"
Avoid: namespace=(user_id, "memory"), key="memory/AGENTS.md"
```

Agent-scoped memory keeps the same key shape:

```text
namespace=(user_id, "memory", agent_id), key="/AGENTS.md"
```

## CompositeBackend Memory Routes

Expose memory under a planner-visible path with `CompositeBackend`:

```python
CompositeBackend(
    default=StateBackend(),
    routes={
        "/memory/": StoreBackend(namespace=lambda _rt: (user_id, "memory")),
    },
)
```

The planner-visible path:

```text
/memory/AGENTS.md
```

is routed to the inner `StoreBackend` as:

```text
/AGENTS.md
```

`inner_store_backend.ls("/")` is the private root of the mounted backend, not
the global filesystem root. Planner-visible paths should come from the
composite backend:

```text
Inner StoreBackend.ls("/")        CompositeBackend.ls("/memory")
  -> /AGENTS.md                     -> /memory/AGENTS.md
  -> /preferences.md                -> /memory/preferences.md
  -> /notes/                        -> /memory/notes/
```

`CompositeBackend.ls("/")` aggregates the default backend and virtual route
directories such as `/memory/`. It should not expose mounted memory files as
`/AGENTS.md`.

In standalone scripts, `CompositeBackend.ls("/")` can fail when the default
backend is `StateBackend()` because `StateBackend` needs LangGraph runtime
state. That is a standalone-runtime limitation, not evidence that `/AGENTS.md`
is the global path.

## Deep Agents Backend Wiring

Pass a backend instance to Deep Agents middleware. Do not pass a callable
backend factory.

```python
backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/memory/": StoreBackend(namespace=resolve_memory_namespace),
    },
)
```

Keep per-call user resolution inside the `StoreBackend(namespace=...)`
callable. If the callable needs active request config and receives a plain
`Runtime`, use `langgraph.config.get_config()` rather than `rt.config`, because
only `ToolRuntime` consistently carries `.config`.

Validate externally supplied namespace components before using them. Encode raw
emails, paths, or provider-specific IDs into safe namespace labels when using
SDK-visible namespaces. The SDK can reject labels that local stores appear to
accept, such as components containing `.`.
