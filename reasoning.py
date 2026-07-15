.claude/skills/langgraph-agent-builder/references/known-pitfalls.md
----
# Known Pitfalls — LangGraph Framework

Diagnostic lookup for LangGraph framework gotchas. Read when debugging, not
before every task. For critical pitfalls, see inline warnings in SKILL.md.

Repo-specific issues belong in `CLAUDE.md` (project root).

---

## Table of Contents

1. [State Management](#1-state-management)
2. [Middleware Hooks](#2-middleware-hooks)
3. [Tools](#3-tools)
4. [Nodes](#4-nodes)
5. [Graph Composition](#5-graph-composition)
6. [Structured Output & Prompting](#6-structured-output--prompting)
7. [Runtime & Testing](#7-runtime--testing)
8. [LangSmith & Evals](#8-langsmith--evals)
9. [Server Runtime & Graph Factories](#9-server-runtime--graph-factories)
10. [Store & Memory](#10-store--memory)
11. [Reasoning / Thinking Models](#11-reasoning--thinking-models)

---

## 1. State Management

## 1.1 Orphan tool calls break LLM calls

AIMessages with `tool_calls` but no matching ToolMessage cause errors when passed
to an LLM or subgraph. **Fix**: Strip orphan tool calls before passing messages
across graph boundaries.

## 1.2 Shared state fields with multiple consumers

When multiple nodes independently filter by a shared state field (e.g. hash sets),
bypass logic in one node can be silently defeated by the other. **Fix**: Audit ALL
consumers of a shared state field when adding bypass or filter logic.

## 1.3 Don't set state fields when node is disabled

When a node self-skips (disabled), leave optional state fields as `None` rather than
setting them to a default. Downstream nodes may branch on `None` vs a value.

## 1.4 `EphemeralValue` does not survive cross-super-step

A value written into an `EphemeralValue` channel is cleared at the end of the next
super-step it isn't re-written, so it's gone before a later node or `task` reads it
(a "planner sets it, tool reads it several steps later" hand-off silently sees
`None`). **Fix**: for set-early/read-later, use `UntrackedValue` (run-scoped, never
checkpointed) or `LastValue` (+ explicit reset). `EphemeralValue` is only for
same-super-step set+read (e.g. `jump_to`). See [state-channels.md](state-channels.md).

## 1.5 Resetting a reducer channel with `[]` is a no-op

Returning `{"k": []}` for a key that has a reducer (`operator.add`, a custom
merge) does nothing — `left + [] == left` — and under a checkpointer the value
persists and keeps accumulating across runs. **Fix**: reset with
`Overwrite(value=[])` from `langgraph.types` (bypasses the reducer); for the
`messages` channel use `RemoveMessage(REMOVE_ALL_MESSAGES)`. Related: concurrent
writes to a plain `LastValue` key raise `InvalidUpdateError` — a key written by
multiple nodes/subagents in one run needs a reducer, not the default channel.

## 1.6 `PrivateStateAttr` is not a lifetime channel

`PrivateStateAttr` hides a key from public graph input/output schemas, but it
does not make the key impossible to observe internally. Middleware and tools can
still see a private key in `request.state` / `ToolRuntime.state` after the graph
writes it. Conversely, if a parent graph copies that key into another graph's
input, the child drops it when the child's schema also marks the key private.
**Fix**: use `PrivateStateAttr` for visibility, combine it with `UntrackedValue`
when the key must reset each run, and prove cross-graph/subagent concerns with a
focused repro rather than a raw dict-copy audit.

## 1.7 `create_agent` drops a reducer when a visibility marker follows it in `Annotated`

When you add a state channel through a **middleware `state_schema`** (the way to
widen a `create_agent`/deepagents graph), the order of metadata inside `Annotated`
is load-bearing. `create_agent` keeps only the **last** channel-defining metadata,
so `Annotated[list[dict], my_reducer, OmitFromInput]` silently degrades to a plain
`LastValue` channel — the reducer is shadowed by the visibility marker. The crash
shows up only later, when two writers hit the key in one super-step ("can receive
only one value per step"), e.g. concurrent `task` delegations writing a shared
output channel. **Fix**: put the reducer **last** —
`Annotated[list[dict], OmitFromInput, my_reducer]` — which keeps both the reducer
and the input-hiding. Note `UntrackedValue` (a channel *type*, not a callable
reducer) is detected regardless of order, so this bites accumulating reducers
specifically. Pin it by asserting the **compiled** channel type
(`isinstance(agent.channels["k"], BinaryOperatorAggregate)`); a schema-level
`__annotations__` check passes even when the channel has degraded.

---

## 2. Middleware Hooks

## 2.1 `wrap_model_call` has NO async fallback

Unlike node hooks (`before_model`, etc.) which fall back to sync when called via
`ainvoke()`, `wrap_model_call` **requires both sync and async implementations**.
Missing either causes a runtime failure — no fallback exists.
**Fix**: Always implement both `wrap_model_call` and `awrap_model_call`.

## 2.2 `jump_to` silently ignored without decorator

Returning `{"jump_to": "end"}` from a hook does nothing unless the hook is decorated
with `@hook_config(can_jump_to=["end"])`. No error, no warning — just silent no-op.
**Fix**: Always add `@hook_config(can_jump_to=[...])` when using `jump_to`.

## 2.3 Never mutate `request` directly

`ModelRequest` objects should be treated as immutable. Direct mutation causes
unpredictable behavior across the middleware stack.
**Fix**: Always use `request.override(tools=[...], model=..., ...)`.

## 2.4 Hook execution order is stack-based

Before hooks run top-to-bottom, after hooks run bottom-to-top, wrap hooks nest.
Placing a retry middleware before a model-routing middleware means retry wraps
routing — the fallback model may not get the routed model's config.
**Fix**: Follow the ordering convention: context injection → behavior → control flow → error handling → oversight.

## 2.5 Adding a `wrap_tool_call` middleware: ordering + stack-sequence tests

A new `wrap_tool_call` middleware must sit **before** other `wrap_tool_call`
middlewares in the composed list to be the outermost tool-call wrapper (so its
return — e.g. a recovered error `ToolMessage` — is what downstream wrappers see).
A `before_agent`-only middleware placed above it (like a pre-agent guard) does
NOT affect tool-call nesting, so it can stay first. **Fix**: insert just after
pre-agent hooks and before other tool-call wrappers; and update any
exact-stack-sequence test (`[type(m) for m in stack] == [...]`) — inserting a
middleware changes the list and silently breaks those assertions.

---

## 3. Tools

## 3.1 `return_direct=True` doesn't support complex Commands

`return_direct=True` only works with simple string returns or `Command` containing
only a `ToolMessage`. It does NOT work with `Command` returns that add AIMessage
after ToolMessage, nor with `Command(goto="__end__")`.
**Fix**: Handle termination logic at the graph level, not in the tool.

## 3.2 Tool errors must be strings, not exceptions

Raising exceptions from tools kills the agent loop. The LLM cannot recover or retry.
**Fix**: Catch exceptions inside the tool and return error messages as strings:
`return f"Error: {e}"`.

## 3.3 Large tool outputs overflow context

Tools returning raw API responses or full document contents can silently fill the
context window. **Fix**: Add truncation with a character limit and use `artifact`
on `ToolMessage` for data that needs to be preserved but not sent to the LLM.

---

## 4. Nodes

## 4.1 `__call__` + `__await__` doesn't work

LangGraph ignores `__await__` on node classes. Similarly `__call__` + `acall` is
ignored. **Fix**: For dual-mode (sync+async), implement the `Runnable` interface with
explicit `invoke()` and `ainvoke()` methods.

## 4.2 Async nodes force `ainvoke()` on the entire graph

Any `async def` node means the entire graph must be invoked with `ainvoke()`.
`invoke()` will fail. **Fix**: Use `async def` + `ainvoke()` consistently for I/O.

## 4.3 Errors should be returned in state, not raised

Raising from a node aborts the graph. **Fix**: Catch errors and return them as
state fields (e.g. `return {"error": str(e)}`). Let downstream nodes or the caller
decide how to handle the error.

---

## 5. Graph Composition

## 5.1 Zero tool_calls after tool-calling node

LLM tool-calling nodes may produce zero `tool_calls` for certain inputs.
Without a conditional edge, downstream tool-execution nodes run with nothing to do
and the full pipeline runs for nothing.
**Fix**: Always add a conditional edge after tool-calling nodes to handle the
no-tool-calls case.

## 5.2 Don't bypass the middleware stack

Calling a model directly from a node (instead of through the middleware stack)
skips all cross-cutting behavior: retry, model routing, mode management.
**Fix**: All model calls go through the middleware stack in ReAct agents.

## 5.3 Don't subclass agents for behavior variation

Subclassing creates rigid hierarchies that fight the middleware composition model.
**Fix**: Use middleware composition and domain context injection instead.

## 5.4 Subgraph state must be compatible with parent

When a subgraph returns `Command(graph=Command.PARENT, ...)`, the `update` dict
must match the parent graph's state schema and reducers. Mismatched types cause
silent corruption or runtime errors.

---

## 6. Structured Output & Prompting

## 6.1 Open-ended string fields = hallucination magnet

LLM structured output with unconstrained string fields (e.g. `source_id: str`) leads
to invented values. **Fix**: (1) Inject valid values into a prompt section,
(2) validate output against known set, (3) retry with corrective feedback,
(4) hard-filter as last resort.

## 6.2 LLMs merge list indices

When asked to produce a list of integers (e.g. `[1, 3, 5]`), LLMs may concatenate
them into `[135]`. **Fix**: Use per-item objects (e.g. `Entry(fact, source_index)`)
instead of a separate indices list.

## 6.3 Don't concatenate prompt strings

String concatenation produces fragile, hard-to-debug prompts. **Fix**: Use
compositional prompt builders with tagged sections (`<identity>`, `<constraints>`,
etc.) for composable, cacheable prompts.

## 6.4 Sub-agent output must be concise

Sub-agents should return structured, short results. The orchestrator synthesizes.
Don't include lengthy explanations in sub-agent output — it wastes parent context.

---

## 7. Runtime & Testing

## 7.1 ContextVars don't propagate across nodes

LangGraph runs each node in its own async context. A `ContextVar` set in node A is
invisible in node B. **Fix**: Pass cross-node data via state fields or
`RunnableConfig`, not manual ContextVar propagation.

## 7.2 `@traceable` intercepts `config` kwarg

LangSmith's `@traceable` consumes the `config` keyword argument by name. In LangGraph,
`config` is passed positionally so it works at runtime. But unit tests calling
`fn(state, config={})` with keyword will fail.
**Fix**: Pass `config` positionally in tests: `fn(state, {})`.

## 7.3 `load_dotenv` silently ignores `.env` when the var is already set

`load_dotenv(".env")` does not override variables already present in the
environment. If the shell (or harness) exports a falsy `LANGSMITH_TRACING`, your
graph won't trace even though `.env` says `true` — and you get no error, just
silence. **Fix**: pass `override=True` when the `.env` value must win:
`load_dotenv(".env", override=True)`.

---

## 8. LangSmith & Evals

## 8.1 Dataset schema patching requires REST

SDK `create_dataset` sets `inputs_schema` / `outputs_schema` only at creation time.
Updating schemas on an existing dataset silently does nothing via SDK.
**Fix**: Use `PATCH /api/v1/datasets/{id}` with `inputs_schema_definition` / `outputs_schema_definition`.

## 8.2 Annotation queue rubric_items only via REST

SDK's annotation queue creation has no `rubric_items` parameter. Queues created via
SDK lack rubric configuration for annotators.
**Fix**: Use `POST /api/v1/annotation-queues` with `rubric_items` in the body.
To add rubrics to existing queues, `PATCH` the queue.

## 8.3 Holistic completeness evaluator hallucinates missing facts

LLM-as-judge evaluators that assess completeness holistically (single prompt for all
facts) tend to invent missing facts that aren't actually in the reference.
**Fix**: Prefer per-fact evaluators (one LLM call per fact) or batch variants. If
using holistic, cap `missing_count` at `total_expected` as a safety net.

## 8.4 Eval extractors scan ALL output messages

Evaluators that extract tool calls or content from `outputs["messages"]` process the
entire message list, including prepended conversation history from multi-turn threads.
The `inputs` parameter is accepted in the signature but unused for filtering.
**Fix**: Use a turn-boundary filter (e.g. `get_current_turn_messages()`) to isolate
the current turn before extraction.

---

## 9. Server Runtime & Graph Factories

## 9.1 `context_schema` goes on `StateGraph()`, not `compile()`

`graph.compile(context_schema=MyCtx)` raises `TypeError`. The param belongs on the
constructor. **Fix**: `StateGraph(state_schema=S, context_schema=MyCtx)`.

## 9.2 `runtime.context` is empty in `make_graph()` factory

`ServerRuntime.context` is `{}` at factory time. Context values sent via `context={}`
in the run request ARE available — but in `config["configurable"]`, not `runtime.context`.
**Fix**: Read context from `config.get("configurable", {})` in factories.

## 9.3 Factory called for non-execution purposes

LangGraph Server calls `make_graph()` for schema reads, state history, and Studio
rendering — not just execution. Expensive operations (API calls, DB connections) run
unnecessarily. **Fix**: Guard with `if ert := runtime.execution_runtime:`.

## 9.4 Varying topology across factory calls corrupts state

Returning different node/edge structures based on user permissions causes checkpoint
corruption — state channels don't match across runs.
**Fix**: Keep topology stable. Vary tool bindings and prompt content, not graph structure.

## 9.5 Missing headers are absent, not None

Headers not sent by the caller don't appear in `config["configurable"]` at all (no key).
Using `configurable["x-uid"]` raises `KeyError`.
**Fix**: Always use `.get("x-uid", "default")` for optional headers.

## 9.6 `langsmith-trace` arrives in `configurable` but is never applied as a parent

**Verified in `langgraph-api` 0.7.70 and 0.8.7 (as of 2026-05-27). Re-check before
applying** — `grep "langsmith-trace\|tracing_context.*parent" .venv/lib/python3.12/site-packages/langgraph_api/stream.py`. If the grep hits, the fix likely landed.

`langgraph-api` forwards incoming `langsmith-trace` / `baggage` headers into
`config["configurable"]` (utils/headers.py:48) but `stream.py:astream_state`
only reads `__langsmith_project__`/`__langsmith_example_id__` — never calls
`tracing_context(parent=...)`. Result: every outer→inner call over `/runs/wait`
produces two separate LangSmith trace trees, even with
`RemoteGraph(distributed_tracing=True)`. ASGI `TracingMiddleware` doesn't help
because the worker runs in a separate async task that doesn't inherit the
request handler's ContextVars. **Fix**: prefer in-process `ainvoke()` if outer
+ inner can co-locate; otherwise monkey-patch `astream_state`. See
[distributed-tracing.md](distributed-tracing.md) for the full pattern.

## 9.7 `LANGCHAIN_TRACING_V2` must also be true, not just `LANGSMITH_TRACING`

langsmith's runtime tracing toggle reads BOTH env vars. If either resolves to
`false`, `get_current_run_tree()` returns `None` in nodes and no client-side
header propagation works. `langgraph dev` loads `.env` via dotenv semantics
(does not override existing shell vars), so `.env` setting `LANGSMITH_TRACING=true`
won't help if the shell exports `LANGCHAIN_TRACING_V2=false`.
**Fix**: set both on the command line: `LANGSMITH_TRACING=true LANGCHAIN_TRACING_V2=true uv run langgraph dev …`.

## 9.8 `langgraph dev` single-worker deadlock for same-server inner calls

Default `--n-jobs-per-worker 10` is fine; but if you pin it to 1, an outer that
calls inner via the same `langgraph dev` will deadlock — outer holds the worker,
inner queues for the same worker. **Fix**: `--n-jobs-per-worker 20` (any value > 1).

## 9.6 `@auth.authenticate` silently disabled on free self-hosted

Custom auth works in `langgraph dev` but is **silently ignored** in production
Self-Hosted Lite (free tier). No error, no warning — endpoints are just unprotected.
**Fix**: Use FastAPI middleware or Aegra for free-tier auth; or use the tier-agnostic
`extract_user_context()` pattern that falls back to headers when native auth is absent.

## 9.7 `@auth.on` has no equivalent in middleware approach

Native `@auth.on.threads.create` provides per-resource authorization (e.g., user can
only see their own threads). FastAPI middleware is all-or-nothing — valid token = full access.
**Fix**: For graph-level authorization, use the `make_graph()` factory to filter
tools/capabilities. For thread-level isolation, implement ownership checks in your
application layer or use Aegra/paid LangGraph which support `@auth.on`.

---

## 10. Store & Memory

## 10.1 `store.search()` prefix-matches namespaces

`store.search(("alice",))` searches every namespace below `("alice",)`, not just
one exact namespace. **Fix**: Use the narrowest safe prefix, e.g.
`("alice", "memory")`, and design namespace ordering around safe broad reads.

## 10.2 Collection-first namespaces can create accidental sharing

`("memories", team_id)` is a useful shared team collection, but
`store.search(("memories", team_id))` reads all memory below that prefix. **Fix**:
put hard isolation boundaries first for private data, e.g.
`("users", user_id, "memory")`, and use separate roots for shared process memory,
e.g. `("agents", agent_id, "process_memory")`.

## 10.3 Store values must be dicts

The SDK rejects non-object values, and local `InMemoryStore` can accept them until
filtered search fails because it expects `value.get(...)`. **Fix**: Always store
JSON-object values, even for file-like content: `{"content": "...", "kind": "..."}`.

## 10.4 `StoreBackend` namespace is not the file path

When `CompositeBackend` mounts `/memory/`, the route prefix is stripped before the
inner `StoreBackend` sees the key. Duplicating `memory` in both namespace and key
creates confusing paths. **Fix**: use namespace `(user_id, "memory")`, key
`"/AGENTS.md"`, and planner path `"/memory/AGENTS.md"`.

## 10.5 Backend factories are deprecated in Deep Agents

Passing `backend=lambda rt: ...` to Deep Agents middleware fires deprecation
warnings and is removed in newer versions. **Fix**: pass a backend instance and
put per-call user resolution in `StoreBackend(namespace=...)`; use
`langgraph.config.get_config()` inside the namespace callable when request config
is needed.

## 11. Reasoning / Thinking Models

## 11.1 Per-turn reasoning-effort flips bust server-side prefix caches

Changing `extra_body.chat_template_kwargs` (e.g. `enable_thinking`) between turns
of one thread re-renders the server-side prompt template — vLLM/SGLang prefix
caches miss on every turn. **Fix**: pick a reasoning effort per agent/task/thread
(`create_chat_model(reasoning_effort=...)`), not per turn; per-call
`build_reasoning_kwargs()` is for *different* tasks, not mid-thread flips.

## 11.2 ChatMistralAI drops thinking blocks on tool-call turns

`langchain_mistralai` replays assistant thinking blocks in history for plain chat
turns, but blanks `content` (thinking included) whenever the message carries
`tool_calls` — so in agent loops Mistral re-derives its plan after each tool
result (extra tokens/latency, not wrong answers). Upstream behavior
(`_convert_message_to_mistral_chat_message`). **Fix**: none needed for
correctness; if quality degrades in Mistral+tools+reasoning loops, patch the
converter to send thinking alongside `tool_calls` (repo precedent: the
langchain-openai converter monkeypatches in `models/custom_chat_model.py`).

## 11.3 Top-level `reasoning_effort` is a no-op on OpenAI-compatible gateways

langchain-openai's `reasoning_effort → reasoning={...}` rewrite is
Responses-API-only; for Chat Completions the raw top-level param is forwarded and
vLLM/LiteLLM gateways ignore it. Only `extra_body.chat_template_kwargs` moves
Nemotron/Qwen. **Fix**: use the per-family translation
(`create_chat_model(reasoning_effort=...)` / `models/reasoning.py`), never a raw
top-level `reasoning_effort` kwarg for gateway-served models.

-------

docs/consuming/chat-models.md
----
# Using `create_chat_model`

`create_chat_model` builds a configured chat model (a LangChain
`ChatOpenAI` or `ChatMistralAI`) from environment variables, with optional
per-call overrides. It is the single entry point for talking to any LLM
provider in this library.

```python
from sta_agent_engine.models import create_chat_model

model = create_chat_model("llmaas")
response = await model.ainvoke("What is Python in one sentence?")
```

## Selecting a provider

The first argument picks the provider. Three equivalent forms:

```python
create_chat_model("llmaas")               # built-in, by string
create_chat_model(ProviderType.MISTRAL)   # built-in, by enum (from sta_agent_core.config)
create_chat_model("openai")                  # any other name — convention provider
create_chat_model()                        # no arg → reads LLM_PROVIDER (default: custom)
```

Built-in names: `llmaas`, `llmaas_dev`, `mistral`, `custom`, `eval`, `openai`.
**Any other string is also accepted** — its env prefix is derived as
`f"{NAME.upper()}_"`, so `create_chat_model("openai")` reads `openai_*` with no
code change.

## Environment variable contract

For a provider `NAME`, set the prefixed vars (e.g. `LLMAAS_API_KEY`):

| Variable | Purpose |
|---|---|
| `{NAME}_API_KEY` | Auth key |
| `{NAME}_BASE_URL` | OpenAI-compatible endpoint |
| `{NAME}_MODEL` | Default model |
| `{NAME}_BIG_MODEL` / `{NAME}_SMALL_MODEL` / `{NAME}_THINKING_MODEL` | Capacity-tier models |
| `{NAME}_MULTIMODAL_MODEL` | Vision model |
| `{NAME}_TEMPERATURE` / `{NAME}_TOP_P` / `{NAME}_MAX_TOKENS` | Generation params |

The built-in `custom` provider uses **no prefix** (bare `API_KEY`, `BASE_URL`,
`MODEL`, …). `openai` reads the canonical `OPENAI_*` vars the official OpenAI
SDK also honors, so `create_chat_model("openai")` works with just
`OPENAI_API_KEY` set.

A ready-to-fill starter file lives at `.env.provider.example`.

## Capacity tiers — `tier=`

`tier` picks a model slot by **capacity/quality** when no explicit `model=` is
passed. All tiers answer the same request; they trade quality for latency/cost.

```python
create_chat_model("llmaas")                    # tier="default" → LLMAAS_MODEL
create_chat_model("llmaas", tier="big")        # LLMAAS_BIG_MODEL,    else MODEL
create_chat_model("llmaas", tier="small")      # LLMAAS_SMALL_MODEL,  else MODEL
create_chat_model("llmaas", tier="thinking")   # LLMAAS_THINKING_MODEL, else BIG_MODEL, else MODEL
```

Cascade: `thinking → big → model`, `big → model`, `small → model`. An unknown
tier raises `ValueError`.

## Vision models — `multimodal=True`

Multimodality is a **capability**, not a tier (you can't ask for the "big
multimodal" model). Pass `multimodal=True`:

```python
create_chat_model("llmaas", multimodal=True)
```

Resolution order:

1. `{NAME}_MULTIMODAL_MODEL` if set — used **verbatim** (authoritative; lets you
   name a vision model the built-in capability list doesn't yet recognize).
2. else `{NAME}_MODEL` **only if** it's a recognized multimodal model.
3. else **raises `ValueError`** — it never silently returns a text-only model
   that would drop images (under the guard middleware) or 400 upstream.

`multimodal=True` takes precedence over `tier=` and is ignored when an explicit
`model=` is passed. Recognized vision models are the allow-list in
`sta_agent_engine.models.capabilities` (`is_multimodal`); setting
`{NAME}_MULTIMODAL_MODEL` bypasses that check.

## Reasoning control — `reasoning_effort=`

Reasoning/thinking is controlled with a normalized `reasoning_effort`
parameter, translated per model family into the kwargs each model actually
honors (Mistral's `reasoning_effort`, Nemotron's and Qwen's
`chat_template_kwargs`). See the dedicated guide: [reasoning.md](reasoning.md).

```python
create_chat_model("mistral", reasoning_effort="high")
create_chat_model("llmaas", model="nemotron-3-super-120b", reasoning_effort="off")
```

## Per-call overrides

Any keyword overrides the resolved env value:

```python
create_chat_model("llmaas", model="some-other-model", temperature=0.7, max_tokens=512)

# Bring-your-own-key (BYOK): inject credentials at call time
create_chat_model("llmaas", provider_api_key="sk-...", provider_base_url="https://...")
```

## Client dispatch (OpenAI vs Mistral)

`create_chat_model` routes to `ChatMistralAI` when the **provider name** contains
`mistral` (e.g. `mistral`, `mistral_eu`) **or the model name** contains
`mistral` / `devstral` / `magistral`; otherwise it routes to `ChatOpenAI` for
OpenAI-compatible APIs.

> **Sharp edge — the model name wins.** A Mistral-branded model on an
> OpenAI-compatible gateway (e.g. `provider="llmaas"` with
> `LLMAAS_MODEL=mistral-small-2506`) routes to the **native Mistral SDK**
> (`api.mistral.ai`), not your gateway. If your gateway serves Mistral-family
> models over an OpenAI-compatible API, pass an explicit `base_url` (and
> `api_key`) so the Mistral client targets your gateway, or use a non-Mistral
> model name.

If the OpenAI dispatch is taken but `api_key`/`base_url` did not resolve,
`ChatOpenAI` silently falls back to `OPENAI_API_KEY` + `api.openai.com`. This
fallback now emits a `DeprecationWarning` and will raise in a future release —
always set `{NAME}_API_KEY` and `{NAME}_BASE_URL` (or pass them as kwargs).

## Full example

A runnable end-to-end example is in
`examples/sta_agent_engine/chat_model_example.py`.

-------

docs/consuming/reasoning.md
----
# Reasoning control with `create_chat_model`

Reasoning ("thinking") models expose incompatible knobs: Mistral takes a
top-level `reasoning_effort` string, Nemotron wants booleans nested under
`extra_body.chat_template_kwargs`, Qwen adds a token budget. The
`reasoning_effort` parameter gives you one vocabulary; the library translates
it into whatever the resolved model actually honors.

```python
from sta_agent_engine.models import create_chat_model

model = create_chat_model("llmaas", model="nemotron-3-super-120b", reasoning_effort="low")
# → ChatOpenAI(..., extra_body={"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}})

model = create_chat_model("mistral", reasoning_effort="high")
# → ChatMistralAI(..., model_kwargs={"reasoning_effort": "high"})
```

Omit the parameter (or pass `None` or an empty/blank string — handy for
env-var plumbing like `os.getenv("REASONING_EFFORT", "")`) and **nothing is
injected** — the model keeps its server-side default. Existing code is
unaffected.

## Effort vocabulary and what goes on the wire

Efforts are plain strings. Built-in families support:

| effort | Mistral (small / medium-3-5) | Nemotron-Super | Nemotron-Ultra | Qwen3.x |
|---|---|---|---|---|
| `"off"` | `reasoning_effort="none"` | `enable_thinking=False` | `enable_thinking=False` | `enable_thinking=False` |
| `"low"` | — | `enable_thinking, low_effort` | `enable_thinking, medium_effort, force_nonempty_content=False` | — |
| `"medium"` | — | — | `enable_thinking, medium_effort, force_nonempty_content=True` | — |
| `"high"` | `reasoning_effort="high"` | `enable_thinking` (full) | `enable_thinking, force_nonempty_content=True` (full) | `enable_thinking` (default) |

Qwen3.x is deliberately binary: the model card (e.g. `Qwen/Qwen3.6-27B`)
documents only the `enable_thinking` switch. Graded thinking **budgets** are a
serving-stack feature (vLLM's `thinking_token_budget`, name and availability
vary by version) — if your gateway supports them, re-register the family with
budget rungs (example below).

`—` = not supported by that family. Requesting it emits a `UserWarning` that
lists the supported set; the library never raises and **never silently
substitutes a different effort**. For families whose native knob is a
`reasoning_effort` string (Mistral, OpenAI), the raw value is still forwarded
so the API itself can validate it; for template-flag families (Nemotron, Qwen)
there is no field to receive it, so nothing is injected.

Nemotron-Ultra notes: `medium`/`high` include
`force_nonempty_content=True` (required by SGLang when tool-calling with
thinking enabled); `low` is medium-effort thinking with the flag explicitly
sent as `False` (so a server-side template default can't force it back on).

The Mistral column applies to every model the library dispatches to the
Mistral client — names containing `mistral`, `devstral`, or `magistral` all
use it, so `"off"` always reaches the API as the sanctioned `"none"`.

Check what a model supports programmatically:

```python
from sta_agent_engine.models import supported_reasoning_efforts

supported_reasoning_efforts("nemotron-3-ultra-550b")   # frozenset({'off', 'low', 'medium', 'high'})
```

## Unknown models

If no family matches the model name, the value is forwarded as a top-level
`reasoning_effort` (the OpenAI-native form) with a `UserWarning`. Genuine
OpenAI reasoning models honor it; vLLM/LiteLLM-fronted gateways typically
ignore it. `gpt-*` model names take this passthrough silently — it is their
native parameter.

If your gateway serves a known family under an alias (`chat-default` actually
being Nemotron), pin the family explicitly:

```python
create_chat_model("llmaas", model="chat-default",
                  reasoning_effort="high", reasoning_family="nemotron-super")
```

`reasoning_family` only selects the translation table — passing it without
`reasoning_effort` injects nothing (the library warns and ignores it).

## Overrides always win

Explicitly-passed native kwargs are the escape hatch, and they beat the
translation on conflicting keys (with a warning). Non-conflicting keys merge:

```python
create_chat_model(
    "llmaas", model="nemotron-3-super-120b",
    reasoning_effort="high",                                  # → enable_thinking=True
    extra_body={"chat_template_kwargs": {"custom_flag": 1}},  # merged alongside
)
# wire: chat_template_kwargs == {"enable_thinking": True, "custom_flag": 1}
```

## Per-call control

`build_reasoning_kwargs()` returns the translated kwargs so you can apply
effort per call instead of per model instance:

```python
from sta_agent_engine.models import build_reasoning_kwargs

kw = build_reasoning_kwargs("nemotron-3-super-120b", "low")
model.invoke(messages, **kw)        # or model.bind(**kw)
```

This works for families that translate into `extra_body` (Nemotron, Qwen) or
a native field (OpenAI). **Exception — Mistral models are constructor-only:**
their translation rides `model_kwargs`, which `ChatMistralAI` flattens into
the request payload only when set as a constructor field; splatted at call
time it would be posted as a literal `"model_kwargs"` JSON key instead. For
Mistral, set the effort where the model is created
(`create_chat_model(..., reasoning_effort=...)`).

Prefer choosing an effort **per agent / task / thread, not per turn**: for
template-flag families, flipping `chat_template_kwargs` between turns of one
conversation re-renders the server-side prompt template and defeats
vLLM/SGLang prefix caching (latency and cost, not correctness).

## Registering your own model family

One call at application startup, no subclassing:

```python
from sta_agent_engine.models import register_reasoning_family

register_reasoning_family(
    "my-model",
    rungs={
        "off":  {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    },
    match_substrings=("my-model",),   # all must appear in the model name
)
```

Rung values are the **literal kwargs** to merge into the model constructor —
anything the underlying client accepts (`extra_body`, `model_kwargs`, native
fields) is fair game. Re-registering an existing family (including a built-in)
replaces it, so you can also override the library's defaults. Example —
extending the built-in `qwen3` family with graded thinking budgets once you've
verified your vLLM gateway supports them:

```python
register_reasoning_family(
    "qwen3",
    rungs={
        "off":    {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        "medium": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 2048}},
        "high":   {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}},
        "xhigh":  {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 8192}},
    },
    match_substrings=("qwen3",),
)
```

Verify the budget parameter name against your serving stack first — vLLM has
shipped it as `thinking_token_budget` and rejected/ignored other spellings
depending on version; a one-off probe call per rung is cheap insurance.

### Slug variants across providers

Family matching is **case- and separator-insensitive**: `-`, `_`, `.`, `:`,
`/` and spaces are stripped from both the model name and the patterns before
comparing. The same model arriving under different provider slug conventions —
`qwen3.6`, `qwen3-6`, `Qwen/Qwen3.6-32B-Instruct`, `qwen3:32b` — resolves to
the same family and produces identical wire kwargs. You only need
`reasoning_family=` when the alias shares nothing with the model's real name
(`chat-default`).

`match_substrings` supports two shapes:

- **Flat tuple = one AND-group** — all substrings must appear:
  `("nemotron-3", "ultra")`.
- **Tuple of tuples = OR of AND-groups** — the family matches when *any*
  group fully matches; use this for genuinely different naming dialects:
  `(("nemotron-3", "ultra"), ("nemo-ultra",))`.

## Known limitations

- **Mistral + tool-calling agents:** the upstream LangChain Mistral
  integration sends an assistant message's `tool_calls` *instead of* its
  content, so reasoning traces are not replayed to the API on tool-call turns.
  Plain chat turns replay them automatically. Impact is limited to extra
  re-thinking inside agent loops, not wrong answers. (Qwen3.6 addresses the
  same concern server-side with `chat_template_kwargs.preserve_thinking=True` —
  orthogonal to effort; pass it via explicit `extra_body`, it merges alongside
  the translated flags.)
- **Whether an effort actually changes behavior is ultimately decided by the
  serving stack.** Some gateways enable thinking by default and ignore
  parameters they don't recognize. When in doubt, verify with a one-off call
  per effort value and inspect the response's reasoning content.

-------

examples/sta_agent_engine/models/reasoning_effort_example.py
----
"""Example: reasoning_effort with create_chat_model — reasoning-token cost per effort.

NOT consumer documentation — this is an engineering smoke-test of the
``reasoning_effort`` surface against live Nemotron endpoints. Consumer-facing
usage lives in docs/consuming/reasoning.md. Edit the USER_* constants and run:

    uv run python examples/sta_agent_engine/models/reasoning_effort_example.py

REAL PAID API CALLS: one request per (model, prompt, effort) — 14 with the
defaults below. Requires the provider env vars ({NAME}_API_KEY +
{NAME}_BASE_URL for the provider name set below).

For each model and prompt, every supported effort is invoked once and the
reasoning cost is reported two ways: the API-counted reasoning tokens (when
the gateway returns ``completion_tokens_details.reasoning_tokens`` — the
authoritative, billed count) and an estimate from the reasoning text surfaced
in the response (~4 chars/token — a fallback for gateways that report no
token details; it undercounts terse math/symbol content).

Two prompts on purpose: an easy classic caps reasoning naturally at ~100
tokens whatever the effort (ceiling effect), so effort asymmetry only becomes
observable on the multi-step constraint puzzle. Expectation there: ``off``
≈ 0 reasoning tokens and reasoning grows with the rung. If ``off`` still
shows a sizable count, the gateway is not applying ``enable_thinking=False``
(check whether it forwards ``extra_body.chat_template_kwargs``).
"""

import asyncio
import time

from langchain_core.messages import AIMessage

from sta_agent_engine.models import create_chat_model, supported_reasoning_efforts


# --- Edit these -------------------------------------------------------------
USER_PROVIDER = "custom"  # any built-in or arbitrary name; reads {NAME}_* env vars
USER_MODELS = ("nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-ultra-550b-a55b")
USER_PROMPTS = (
    (
        "easy",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    ),
    (
        "complex",
        "Four people (Ava, Ben, Cy, Dee) each ordered a different drink (coffee, tea, juice, water) "
        "and sat in seats 1-4, left to right. Clues: (1) Ava sat immediately left of the tea drinker. "
        "(2) Ben did not order coffee and did not sit in seat 4. (3) The juice drinker sat in seat 1. "
        "(4) Cy sat exactly two seats right of Dee. (5) The water drinker sat next to Ben. "
        "Work out who sat where and who ordered what.",
    ),
)
USER_MAX_TOKENS = 4096  # cap per response so a chatty high-effort run stays bounded
# ---------------------------------------------------------------------------

_EFFORT_ORDER = ("off", "low", "medium", "high")  # display order for whatever rungs the family defines


def _reasoning_text(response: AIMessage) -> str:
    """Collect whatever reasoning the gateway surfaces client-side.

    Gateways differ: some put it in ``additional_kwargs["reasoning_content"]``
    (the create_chat_model converters normalize to this), others emit
    ``type="reasoning"`` content blocks.
    """
    parts = [str(response.additional_kwargs.get("reasoning_content") or "")]
    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                parts.append(str(block.get("reasoning") or block.get("content") or ""))
    return "".join(parts)


async def _measure(model_name: str, effort: str, prompt: str) -> tuple[int | str, int, int | str, float, str]:
    """Invoke once at the given effort; return (api reasoning tokens, estimated
    reasoning tokens, output tokens, seconds, answer preview)."""
    model = create_chat_model(USER_PROVIDER, model=model_name, reasoning_effort=effort, max_tokens=USER_MAX_TOKENS)
    started = time.perf_counter()
    response = await model.ainvoke(prompt)
    elapsed = time.perf_counter() - started
    assert isinstance(response, AIMessage)

    usage = response.usage_metadata or {}
    api_reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    estimated_reasoning = round(len(_reasoning_text(response)) / 4)  # ~4 chars/token heuristic
    output_tokens = usage.get("output_tokens")
    answer = response.text.strip().replace("\n", " ")[:70]
    return (
        api_reasoning if api_reasoning is not None else "n/a",
        estimated_reasoning,
        output_tokens if output_tokens is not None else "n/a",
        elapsed,
        answer,
    )


async def main() -> None:
    for model_name in USER_MODELS:
        supported = supported_reasoning_efforts(model_name)
        efforts = [effort for effort in _EFFORT_ORDER if effort in supported]
        print(f"\n=== {model_name} — efforts: {', '.join(efforts)} ===")
        for label, prompt in USER_PROMPTS:
            print(f"\n[{label}] {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
            print(f"{'effort':<8} {'api rsn tok':>12} {'est rsn tok':>12} {'output tok':>11} {'seconds':>8}  answer")
            for effort in efforts:
                api_reasoning, estimated, output_tokens, elapsed, answer = await _measure(model_name, effort, prompt)
                print(f"{effort:<8} {api_reasoning:>12} {estimated:>12} {output_tokens:>11} {elapsed:>8.1f}  {answer}")


if __name__ == "__main__":
    asyncio.run(main())

-------

memory_bank/MEMORY.md
----
# Memory Bank Index

All creative-phase docs live under `creative_phases/`. Topic folders group related work; `_archive/` is the read-only history.

## Topic folders

- [creative_phases/evaluation/](creative_phases/evaluation/creative_phase_step_2_2026-05-27_implementation_log.md) — Agent evaluation skill suite: `agent-eval-designer` (framework-agnostic methodology) + `sta-eval-implementer` (in-house framework binding). **IMPLEMENTED 2026-05-27** on branch `feat/agent-evaluation` — both skills shipped (22 files, ~4500 lines, all budget-compliant; mechanical validation green: zero framework leakage in Skill A body, all cross-links resolve, Python/JSON templates valid, in-house symbol citations resolve in `evals/base/`). End-goal = use both skills to design + ship the orchestrator deep-agent eval suite (task pending).
- [creative_phases/orchestrator/](creative_phases/orchestrator/README.md) — twin_router → orchestrator deep-agent rewrite on the `deepagents` package (#13/#14). v1 design + 6-PR plan; PLANNED.
- [Orchestrator improvements master plan (6 topics)](creative_phases/orchestrator/creative_phase_2026-06-02_improvements_master_plan.md) — **PLANNED 2026-06-02** (branch `worktree-orchestrator-improvements`). 7-agent planning workflow over 6 topics: (T1) KA↔orchestrator ephemeral structured data sharing — `task` propagates via shared STATE keys only (not context), `EphemeralValue` for per-run reset; apcode is boost-only so "filter to exactly those docs" needs new KA wiring (deferred). (T2) incident decouple — seam fix is a 1-line bug (`build_incident_subagent.py:102`) fixing [ORCH-01]; package delete is consumer-visible (gated on Badr). (T3) recursion graceful-degrade — `RecursionLimitHandlerMiddleware` catching only `GraphRecursionError` at parent. (T4) reject unified factory (only navigator forwards middleware), ship tiny `_subagent_common.py` helper. (T5) prompt-guard: multimodal judge (reuse `strip_images_for_model`), ⚠️ refusal formatting, optional/mistral-small-2603 default (gated). (T6) reliable `read_picture` via layered system-reminder + bounded `tool_choice`. 11-PR sequence in 4 waves; suggested first PR = T2 seam fix. **3 blocking open Qs for Badr** (hard-filter vs boost, incident_agent external consumers, guard deploy-status). **ROUND-2 CORRECTIONS ADDENDUM appended** (post user re-challenge, 2nd 7-agent workflow): T2 **user_correct** (relocate mock → `agents/incident_agent/graph:incident_agent`, delete `twin_router/mock_incident_agent`, repoint twin_router too — collapses to 1 PR; round-1's "don't touch twin_router" inverted); T1 **user_partially_correct** (`MetadataScope.apcode_filter` hard filter + `documents` search param BOTH already exist — `metadata_scope.py:126-131`, `elastic_retriever.py:857`, `elastic_rag_proxy/context.py:40`; PR-11 re-scoped from separate workstream → small wiring; round-1 conflated tool-arg boost layer with retriever layer); T3 **user_correct** (Option B gated to middleware-accepting factories — navigator only); T4 **nuanced** (factory-in-`SubagentEntry` + `supports_middleware` flag + one `build_subagent` dispatcher — NO `_subagent_common.py`); T5 **user_partially_correct** (guard is text-only today `prompt_injection_guard.py:197-205` — AXIS-1 is NEW capability not crash-fix; AXIS-3 regression real but re-described as "screening goes silent"); T6 **nuanced** (eager-describe in `wrap_model_call` via `Overwrite`+sentinel replaces read_picture tool-nudge). Cross-topic thread: T3-B/T4/T1 converge on `SubagentEntry` as the single registration seam. NEW open Qs: elastic_rag gateway server-side `retriever_documents` enforcement (trust boundary), doc_id namespace match (`doc_keyword_field`). **ROUND-3 DECISIONS LOCKED** (with owner): T1 direct `ElasticRetriever` only — drop proxy path, gateway trust-boundary risk RETIRED (`twin_ka_entries.py:11-13`); T4 names locked **`SubagentStateBridge` base + `KnowledgeBridgeMiddleware`**, declared in `SubagentEntry.state_bridge`, shared channels = one `KnowledgeBridgeChannels` TypedDict — orchestrator/react subagents INJECT the middleware (gated by `supports_middleware`), basic graphs (KA) INHERIT the channels TypedDict statically (no runtime schema surgery); **ephemerality unverified — probe `EphemeralValue` cross-super-step survival before building** (may clear between super-steps → filter lost before `task` fires; fallback = `LastValue` + reset node); T5 multimodal judge adds image parts only when `is_multimodal(judge)`; T6 `before_agent` + synthetic `read_picture` pair (idempotency keyed to image, keep tool bound, `wrap` strip as net), drops `tool_choice` force.
- [creative_phases/twin_router/](creative_phases/twin_router/README.md) — Habilitation, habilitation-refactor, auth-evolution. Being replaced by the orchestrator rewrite (#14).
- [creative_phases/elastic_rag/](creative_phases/elastic_rag/README.md) — Eval-sweep planning + pointer to the still-at-parent `retriever_owns_expansion/` thread folder.
- [creative_phases/retriever_owns_expansion/](creative_phases/retriever_owns_expansion/README.md) — Active QMD-inspired expansion+fusion thread (#30 closed; follow-ups under #34/#35/#39).
- [creative_phases/knowledge_agent/](creative_phases/knowledge_agent/AGENTS.md) — Knowledge Agent design history (retriever layer → multi-retriever evidence-gathering agent → review loop → compression → synthesizer → presets).
- [creative_phases/lightrag/](creative_phases/lightrag/) — LightRAG local-engine + topology designs, plus the foundational retriever-protocol doc (`creative_phase_2026-02-09_retriever_and_lightrag_architecture.md`).
- [creative_phases/frontend/](creative_phases/frontend/) — Frontend creative phases (chat customization, dynamic context controls).
- [creative_phases/roadmap/](creative_phases/roadmap/) — Roadmap creative phases organized by pillar (p3, p4 currently populated).

## Loose creative phases

- [creative_phases/creative_phase_2026-06-06_prefix_cache_tuning.md](creative_phases/creative_phase_2026-06-06_prefix_cache_tuning.md) — **PARTIALLY IMPLEMENTED 2026-06-06.** Make the middleware stack prefix-cache friendly (vLLM APC + Anthropic prompt caching). SHIPPED: `TimeAwareMiddleware` reworked — stopped rewriting the system prompt with `datetime.now()` (the Anthropic-documented "timestamp in an early block → no cache hit" anti-pattern); now injects an **immutable `<system_reminder>` message** via `before_agent` (once per run, never mid-loop), deterministic bucket id + **scan-and-skip** so content is frozen and only ever tail-appended. New `InjectionStrategy` (INTERVAL default, `interval_hours=3` / PER_HUMAN_TURN), `role`/`tag` knobs, back-compat ctor (`section_name` alias). 18 tests green, ruff/pyright clean. NOTED FOR FUTURE WORK: full middleware audit table (worst = deprecated `mode_management`/`reflection` mode-cycle prompt+tool rewrites [CI-08]; live concern = `dynamic_tools` order stability); + backlog (document § prefix-cache hygiene in engine AGENTS.md + skill, live-model `SystemMessage`-in-messages placement check, normalizer/patch_tool_calls dedupe, explicit `cache_control` emission, cache-hit measurement). Grounded on Anthropic docs (tools→system→messages cascade) + vLLM block-level hashing.
- [creative_phases/creative_phase_2026-05-28_bundle_fixes_seed.md](creative_phases/creative_phase_2026-05-28_bundle_fixes_seed.md) — **SEED — iterating** (2026-05-28, branch `worktree-bundlefixes`). 35 captured fixes from the "Bundle Sync 27/05" notes across Global/Examples/Core/Frontend/Tests/Engine(KA)/Twin. Each has a stable ID + status; clarify → spec → implement one at a time.
- [creative_phases/creative_phase_2026-03-05_library_logging_hygiene.md](creative_phases/creative_phase_2026-03-05_library_logging_hygiene.md) — APPROVED for implementation, **NOT YET IMPLEMENTED** (verified 2026-05-07 via subagent: design helpers `register_library_logger` / `configure_library_logger` not in `packages/`, `setup_logger.py:22` still has the NullHandler the design called to remove). Remains active until picked up.
- [creative_phases/creative_phase_2026-05-17_dynamic_settings_registry.md](creative_phases/creative_phase_2026-05-17_dynamic_settings_registry.md) — **PLANNED v2.4 — READY FOR IMPLEMENTATION** (Codex-reviewed 2026-05-28; scope-trimmed 2026-05-29 → simplified 2026-05-30). ~40% shipped on `main` (`0ed2198`): dynamic `_env_prefix=` kwarg, open chat registry, convention fallback, OpenAI default-registered, `DeprecationWarning` on silent OpenAI fallback. Remaining on branch `feat/provider-tiers-v2.4` — **one PR + two follow-ups (~3 d):** (Phase 1) capacity tiers on `BaseProviderSettings` (`big`/`small`/`thinking` + `get_model(tier)` cascade) **+ collapse the three lookup structures into ONE `_LLM_REGISTRY`** — `_provider_map`+`_dynamic_registry`+`_dynamic_class_cache` fold in; convention-fallback **lazily self-populates** (no separate synthesis cache); `ProviderType` **retained as a supported input type** (NOT deprecated — backward compat); `ProviderFactory`/`get_provider_settings` become read-through shims; `ProviderSpec` carries `chat_model_factory`+`BuildContext`; built-in registrations move core→engine (vendor-freeness); (Phase 2) `LightRAGCoreSettings.for_instance_from_env` migration; (Phase 3) `_is_mistral_model` substring-dispatch fix (one-release `DeprecationWarning` bridge). **CUT/DEFERRED:** capability/multimodal axis (builder's responsibility, `MultimodalGuardMiddleware` untouched); `reasoning_effort` (vLLM probe 2026-05-29 → thinking on-by-default, passthrough a no-op, only per-family `chat_template_kwargs` moved it; streaming-reasoning gap traced to LiteLLM proxy #20246); **third-party `register_provider` + embedding/rerank registries + `BaseProviderCredentials` split (YAGNI until a real external consumer)**. 10 decisions locked. ASCII architecture map + snippets in doc. Probe: `experiments/langchain/chatopenai/reasoning_script.py`.
- [creative_phases/orchestrator/creative_phase_2026-06-20_sources_block_ownership.md](creative_phases/orchestrator/creative_phase_2026-06-20_sources_block_ownership.md) — **IMPLEMENTED 2026-06-21 (test-first, 3 commits; helpers folded into `knowledge_bridge.py`, NOT a separate util file; D2 = one audit-tagged message; decoder stop removed).** SUPERSEDES the 2026-06-18 decoder-`stop`: stop *suppressing* the planner's `Sources:` block — **ENCOURAGE** it (cite only the `ka_sources` used, as `[N] [title](url)`) and have the orchestrator **OWN** it. Terminal `after_agent`/`aafter_agent` on `KnowledgeBridgeMiddleware`: if the final answer already has a Sources header → **leave** it (D1); else **fake-stream** a canonical block via langchain_core `GenericFakeChatModel` as a separate `additional_kwargs={"sta_generated":"sources_block"}` `AIMessage` (D2). Remove `_PLANNER_STOP_SEQUENCES` + `PlannerModelResolver(stop=...)` → fixes the vLLM+Nemotron `</think>` leak (R1, user-confirmed stop-induced; LiteLLM #20246 streaming-reasoning gap is only the backup explanation). Phase-0 proof: `GenericFakeChatModel.astream` inside the exit node surfaces as `messages`-mode chunks and fires exactly once (`factory.py:1497/1638` + spike); `wrap_model_call` exposes NO token stream (handler returns a full `ModelResponse`) → the streaming-middleware idea was dropped. R4 resolved (after_agent single-fire). Builds on [[creative_phase_2026-06-18_sources_stop_enforcement]] + [[creative_phase_2026-06-16_sources_determinism_and_synthesis_context]]. **§7 post-impl refinements (2026-06-21, commits `d941c0c7`/`7dfc7e7d`/`d31c3ed2`):** header pinned to EXACTLY `**Sources**:` (literal example, stated positively — do NOT enumerate wrong headings, that primes them); block is now **comma-separated single line** `**Sources**: [1] [title](url), [2] ...` (`format_sources_block` joins with `, ` so fallback == planner shape); **KA is the source of truth**, the `<knowledge_sources>` note is only a reminder; announcer reminder rows render as `[N] [title](url)`. Delegation/caller-scope split to [[creative_phase_2026-06-21_delegation_parallelism_and_caller_scope]].
- [creative_phases/orchestrator/creative_phase_2026-06-21_delegation_parallelism_and_caller_scope.md](creative_phases/orchestrator/creative_phase_2026-06-21_delegation_parallelism_and_caller_scope.md) — **Part A + Part B IMPLEMENTED.** **A — orthogonal-asks → parallel delegation (commits `973d6068`, `8508caa8`):** the planner serialized/merged independent questions because the tasking guidance modeled only facets-of-one-topic-across-agents, never multiple independent asks (least of all to the SAME agent). Added the FACETS-vs-ORTHOGONAL split to `_SUBAGENT_TASKING_SECTION` (orthogonal asks → one scoped brief each, emitted in the SAME turn → parallel, even to the same sub-agent; agent-agnostic per `test_subagent_tasking_is_agent_agnostic`), flipped `_DEEPAGENTS_TOOL_GUIDELINES` "in sequence" → "in sequence or in parallel", and on the KA card: spawn one KA task per independent doc question + **do NOT name/hint a corpus in the brief (KA self-selects)**. **B — caller-scope hardening (commit `97287012`, test-first):** the bridge `_render_selection` steer "instruct it to query `general_doc`" existed because the caller-scope hard filter only applies on `accepts_caller_scope=True` entries (only `general_doc` has it) — without the steer the KA could search `twin_project_doc` and silently drop the filter. Fix = structural: `PlanQueriesNode._active_plan_inputs(state)` restricts the bound tool set to scope-accepting entries when **any** caller-scope axis is present (filter by the flag not the name; empty-set/no-entries guard keeps all + warns; never mutates `self._tools` — shared singleton), threaded into the bind, sanitizer, prompt tools block, and structured-output validation; then deleted the prompt steer + dead `general_corpus_name`. Anchor injection inherits the restriction. **Decisions:** trigger on ANY axis (not just `doc_ids`); **straight cut, no deprecation** (Badr confirmed no deployment relies on broad-search-under-scope). Sibling: [[creative_phase_2026-06-20_sources_block_ownership]] §7.
- [creative_phases/orchestrator/creative_phase_2026-06-18_sources_stop_enforcement.md](creative_phases/orchestrator/creative_phase_2026-06-18_sources_stop_enforcement.md) — **OFFLINE GATE IMPLEMENTED 2026-06-18 (uncommitted).** Enforcement + test plan for the "planner emits no trailing `Sources:` block" rule (the prompt already says it; the LLM violates it probabilistically). Mechanism settled by probe `experiments/orchestrator_stop_sequence/stop_sequence_probe.py`: inject decoder `stop=["\nSources:", "\nSources :"]` via **`create_chat_model(..., stop=[...])`** (lives on the model instance → survives deepagents `bind_tools`); `model.bind(stop=...)` is a **footgun** (silently dropped by `bind_tools`); middleware `model_settings["stop"]` also works but heavier; regex `strip_sources_block` is a **non-streaming-only** safety net (can't un-send a streamed block → stop is primary). Exact marker is `Sources:` (no space) — `"Sources :"` alone would never fire. Test matrix A–G (offline plumbing/marker/strip/preservation = gate; online provider-matrix/E2E-KA/streaming = opt-in `integration_online`). Invariant: NO trailing block, but PRESERVE inline `[N]` (rendered downstream from `ka_sources`) + substance. R4 (dangling `[N]`) de-risked by the 2026-06-16 determinism doc. Open Qs: stop-only vs +strip-net; is the answer streamed (makes stop mandatory); renderer live. Builds on [[creative_phase_2026-06-16_sources_determinism_and_synthesis_context]]. **§9 follow-on (IMPLEMENTED 2026-06-18, uncommitted):** killed the planner's multi-call offset arithmetic by *injecting* the canonical `ka_sources` list back as a `<knowledge_sources>` `SystemMessage` after each KA `task` (per-call, offset-numbered from `len(state["ka_sources"])`) in `KnowledgeBridgeMiddleware.wrap_tool_call` (append-only behind the untouched answer ToolMessage — pitfall #5 forbids rewriting the KA answer; `SystemMessage` not `AIMessage` to dodge Anthropic prefill firing the stop). Spike verified: deepagents `task` returns a `Command` whose `update` carries `messages`+`ka_sources`, so the merge is a one-line append. Simplified `_OUTPUT_FORMAT` (deleted the ~80-word shift rule → cite by the note) + broadened `_PLANNER_STOP_SEQUENCES` to 4 incl. `\n**Sources`/`\n## Sources`. Relocation decision: bridge MIDDLEWARE already in orchestrator; shared channel contract STAYS in KA (moving inverts knowledge_agent→orchestrator + breaks KA standalone exports). 398 orchestrator offline tests pass; ruff+pyright clean. Files: `middlewares/knowledge_bridge.py`, `orchestrator_planner_prompt.py`, `orchestrator_catalog.py`, 3 test files, orchestrator `AGENTS.md`. **§9.10 parallel-fix (IMPLEMENTED 2026-06-18, uncommitted) — SUPERSEDES the §9 wrap_tool_call mechanism:** the per-`task` `wrap_tool_call` announcer numbered colliding `[N]` under PARALLEL KA fan-out (siblings in one tool super-step share a pre-merge `request.state`, so each numbers from the same offset while the reducer concatenates them — a per-call hook can't see sibling contributions). Moved the announcer to **`before_model`/`abefore_model`**: it numbers from the POST-merge `ka_sources` + a private `ka_sources_announced` cursor (announces `ka_sources[announced:]` as `[announced+1…]`, one note per planner turn covering a whole parallel batch). Cursor is `PrivateStateAttr` on `KnowledgeBridgeState` only (never crosses `task` to KA), single-writer LastValue, reset to 0 in `before_agent`. Net simplification: bridge no longer wraps tool calls → SubagentTaskFailure is the sole/outermost `wrap_tool_call`; pitfall #5 holds trivially (announcer never touches the answer ToolMessage). Spike confirmed `before_model` sees post-merge reducer state. `_OUTPUT_FORMAT` bullet generalized ("a `<knowledge_sources>` note … may be more than one"). 399 orchestrator offline pass (41 bridge incl. an end-to-end parallel-fan-out test); ruff+pyright clean. Files: `middlewares/knowledge_bridge.py`, `orchestrator_planner_prompt.py`, `middlewares/test_knowledge_bridge.py`, `test_orchestrator_planner_guidance.py`, orchestrator `AGENTS.md`.
- [creative_phases/orchestrator/creative_phase_2026-06-16_sources_determinism_and_synthesis_context.md](creative_phases/orchestrator/creative_phase_2026-06-16_sources_determinism_and_synthesis_context.md) — **DONE (uncommitted) + 1 open thread.** Deterministic sources: planner emits NO `Sources:` block/urls, cites by the sub-agent's own number unchanged (gaps fine) with a multi-call OFFSET rule; KA `ka_sources` now 1:1 with `answer_citations` within a call (`_build_ka_sources` no second dedup) and **pure concatenation across calls** (`merge_ka_sources`, removed `_source_dedupe_key`) so the offset arithmetic is position-stable (trade-off: a doc cited by 2 KA calls shows twice). Tests green (1357 offline). Suggested commits in doc. **OPEN:** KA `SynthesizeNode` doesn't see per-page metadata/context-summary of findings → "who is contact of X Team" mis-attributes generic contact pages whose metadata is the real disambiguating signal; want a context summary per page id surfaced into synthesis. Start at `utils/findings_format.py`, `nodes/synthesize.py`, check if `Finding`/`Citation` even carry page metadata at synthesis time (may be lost in compression).
- [creative_phases/creative_phase_2026-07-13_reasoning_effort_config.md](creative_phases/creative_phase_2026-07-13_reasoning_effort_config.md) — ✅ **IMPLEMENTED 2026-07-13.** Declarative per-family reasoning control: `reasoning_effort=` on `create_chat_model` + `build_reasoning_kwargs()` / `supported_reasoning_efforts()` / `register_reasoning_family()` in `models/reasoning.py`. Plain dict table (mistral binary none/high via `model_kwargs`; nemotron-super off/low/high + nemotron-ultra off/low/medium/high via `chat_template_kwargs` — ultra low = `medium_effort` WITHOUT `force_nonempty_content`, medium/high bake it True; qwen3 = **binary off/high** per the real `Qwen/Qwen3.6-27B` card — graded `thinking_token_budget` rungs are serving-stack-dependent, shipped only as a consumer re-registration example in the doc; openai/gpt = silent native passthrough). Warn-not-raise; explicit caller kwargs beat table values on leaf conflict (deep-merge); `reasoning_family=` kwarg pins gateway-aliased names. Family matching is case/separator-insensitive (`qwen3.6`≡`qwen3-6`≡`Qwen/Qwen3.6-32B`≡`qwen3:32b`); `match` = flat AND-tuple or OR-of-AND-groups for alias dialects (lone strings coerced — missing-comma trap guarded). Revives the 2026-05-30 probe doc's "declarative per family" prescription. Consumer doc `docs/consuming/reasoning.md`. Deferred: settings-level env defaults; mistral tool-call thinking-drop converter patch (upstream `langchain_mistralai` blanks content+thinking on tool-call turns — needs online evidence).
- [creative_phases/creative_phase_2026-05-30_reasoning_compatibility.md](creative_phases/creative_phase_2026-05-30_reasoning_compatibility.md) — ✅ **REVIVED 2026-07-13** → superseded by [[creative_phase_2026-07-13_reasoning_effort_config]]; empirical findings remain the grounding evidence (thinking on-by-default; top-level `reasoning_effort` no-op through the LiteLLM gateway; only `chat_template_kwargs` moved Nemotron). Split out of the v2.4 deferral of `reasoning_effort`. Captures the empirical vLLM/LiteLLM-gateway reasoning findings (2026-05-29): thinking on-by-default for gpt-oss/Qwen3/Nemotron; top-level `reasoning_effort` a no-op (langchain-openai Responses-API rewrite doesn't fire for chat completions); reasoning parseable non-streaming from `provider_specific_fields["reasoning"]` (LiteLLM field — gateway is LiteLLM-fronted); streaming surfaces reasoning only for gpt-oss (LiteLLM #20246 / #9578); Nemotron `chat_template_kwargs.low_effort` is an effort toggle not on/off. Conclusion: no uniform lever — if revived, do it declaratively per model family; re-run the probe (`experiments/langchain/chatopenai/reasoning_script.py`) on the then-current gateway first.

## Active highlights

- [External agent capability cards for the orchestrator](creative_phases/orchestrator/creative_phase_2026-06-13_external_agent_cards.md) — **DESIGN LOCKED 2026-06-13, ready for Phase 1.** How externally-produced agents (own LangGraph deployments, run as `RemoteGraph`) self-describe scope so the planner routes to them like internal subagents. **Research:** A2A AgentCard (LF spec v1.0) is the only real interop standard (ACP merged into it; agents.json/ai-plugin.json dead; OASF niche/wraps A2A; MCP has no agent primitive) — but no standard carries source-of-truth/use-for/freshness; our `CapabilityDefinition` is richer. **Verified on installed stack:** `langgraph-api 0.9.0` already serves A2A cards (`a2a.py` `GET /a2a/{assistant_id}/.well-known/agent-card.json`) but the auto-card is FLAT — only `card.description` (+ graph key→name) is producer-controllable; `tags`/`examples`/`skills[].metadata` are hardcoded (`generate_agent_card` `a2a.py:2000-2092`); `langgraph.json` dict form `{"path","description","config"}` reads no `metadata` (`graph.py:512-563`). **Carrier proven by LIVE TEST** (langgraph dev, elastic_rag_mock, restored clean): structured-JSON-in-`description` round-trips intact through the A2A endpoint with zero server patch. **Locked v0 schema:** `{description, short_description, scope, freshness(live|periodic|static_docs), how_to_use, examples}` — `use_for` merged into `description`; `how_to_use` kept separate from `examples`; **dropped** `source_of_truth`+`dont_use_for` (over-complicate producer contract); `short_description` UX-only (not routed). **Maps to `CapabilityDefinition`** (+ NEW optional `how_to_use` field → `How to use:` section in `_compact_description`); `scope`→`use_for[0]`, `freshness`→`note` caveat. **Trust:** orchestrator assigns the catalog `key` (not the card — no `incident_agent` shadowing); profile size-capped; malformed→thin degraded path. New feature → no deprecation needed (`AGENTS.local.md`). **Phase 1** schema+adapter (`agent_card_to_capability(card,*,key)`, `extract_twin_profile`, `twin_profile_to_description` helper+JSON Schema) pure/offline/tested; **Phase 2** producer docs + extend `defining-subagent-capabilities.md`; **Phase 3 (later, needs arch decisions)** discovery/registry + `RemoteGraph` subagent build + per-deployment auth.
- [Orchestrator skills middleware — discovery + 3-PR plan](creative_phases/orchestrator/creative_phase_2026-06-12_skills_middleware.md) — **PLANNED, decisions LOCKED 2026-06-13, ready for PR-S1.** Bring agent skills to the TWIN orchestrator via deepagents' built-in `SkillsMiddleware` (0.6.7 ships it: backend+sources, progressive disclosure — name+description in system prompt, full SKILL.md read via existing `read_file`; loads once per run in `before_agent` with the SAME checkpoint-staleness bug-class `LiveMemoryMiddleware` fixes). **Decisions:** Q1 bank = packaged read-only `/skills/builtin/` FilesystemBackend in the wheel; Q2 anonymous = bank-only (no user Store route); Q3 `/<skill_name>` command leaves the user message byte-identical (append tail reminder, never rewrite); Q4 planner-only (no per-subagent skills); Q5 all roles (skills are prompt text, gated by `has_uid`+`enable_skills`, NOT `select_orchestrator_permissions`). PR-S1: two CompositeBackend routes (`/skills/builtin/` FS both shapes + `/skills/user/` `(uid,"skills")` Store auth-only) + 1 proof skill in wheel + `skills=[...]` kwarg on `create_deep_agent` + `enable_skills` full off-switch. PR-S2: `LiveSkillsMiddleware` (wrap_tool_call refresh after write/edit under /skills/) + portable `skill-creator` SKILL.md + 4-5 generic bank skills. PR-S3: `/<skill_name>` `before_agent` (after guard+skills-load) parses last HumanMessage, injects skill body as immutable `<system_reminder>` (knowledge-bridge deterministic-id pattern). Top risk: user-authored skill bodies are an injection surface the prompt guard never screens (authoring-time screening decided in PR-S2). PR-S1 pre-flight: confirm `CompositeBackend` mixes FilesystemBackend+StoreBackend routes; confirm nested `skills/<name>/SKILL.md` ships in wheel.
- [Frontend — Streamlit aiohttp session cleanup](creative_phases/frontend/creative_phase_2026-06-02_streamlit_aiohttp_session_cleanup.md) — **PROPOSED — DEFERRED 2026-06-02**. Fix for "Unclosed client session / Unclosed connector" warnings from the Elasticsearch adapter. Root cause: the ES async client's aiohttp `ClientSession` is bound to the event loop it was created on, and nothing closes it at lifecycle end. Eval-side half SHIPPED on `perf/lazy-imports-lightrag-extras` (`876758a5`): adapter `WeakSet` + `aclose_all_adapters()` teardown helper, called in `evals/cli.py` `finally`; auto-recovery now closes the stale client; `AsyncElasticGenericRepository.aclose()` added. Frontend half (this doc) deferred — Streamlit's `graph_catalog._run_async` spins a fresh `asyncio.run` loop per interaction, so a cached in-process graph's session is orphaned each turn (accumulates → several warnings). **Option A** (recommended, ~5 LOC): call `await aclose_all_adapters()` in `_run_async`'s `finally` (+ `chat.py:383`) — kills warnings, costs one ES reconnect per turn. **Option B** (robust, larger): single persistent background loop so the session stays on a live loop — no reconnect, but real streaming-pipeline surface change. Ship A first. In-process graphs only (RemoteGraph leaks land in the server's logs).
- [Eval skills grounding audit](creative_phases/evaluation/reference_2026-06-01_eval_skills_grounding_audit.md) — **REPORT 2026-06-01, NO FIXES APPLIED.** `ground-eval-skills` workflow (24 agents) checked all 23 files of both eval skills against live `sta-eval` source. 75 findings (25 high). 2 STALE files won't import (`evaluator-wiring.md`, `evaluators_skeleton.py`: `make_tool_signal`/`_plan_size_signal` from wrong module + direct trajectory evaluators called as factories → TypeError). 5 themes: (1) those crash-level template imports; (2) `guidelines` wrongly doc'd as a `BaseReferenceOutput` base field in 3 files (it's an orchestrator-subclass field); (3) faithfulness grounding still lists `expected_facts` in 3 designer files — contradicts this session's transcript-grounding fix; (4) `dataset-buckets.md` data-gap encoding says `expected_behavior=="answer"` — contradicts this session's `acknowledge_gap`+`reference_tool_calls` semantics; (5) fabricated CLI surface (`--config` should be `-o/--override`; exit-code-2 fabricated; rubric-list "crash" now warns-and-ignores). User chose REPORT ONLY — fixes per-file later. Full per-file finding list + fixes in the doc.
- [Frontend — RemoteGraph trace/feedback linking](creative_phases/frontend/creative_phase_2026-05-26_remotegraph_trace_linking.md) — **IMPLEMENTED 2026-05-26 (Path A)**. Fix for "feedback succeeds but `get_run_url` 404s on RemoteGraphs": `RemoteGraph.astream` doesn't forward `config.run_id`, so the server-side run id (the real LangSmith trace id) drifted from the client-minted UUID. Frontend now strips `config["run_id"]` for remote graphs and, after streaming completes, reads the real id back via `remote_graph.client.runs.list(thread_id, limit=1)` and overwrites `current_turn_run_id` — feedback widget then looks up the right trace. Best-effort: list-failure falls back to the client UUID. Same PR also unwraps the `{"value": [...]}` channel envelope returned by `RemoteGraph.get_state()` to kill `MESSAGE_COERCION_FAILURE` warnings on every render. 9 new unit tests. **Path B (distributed-tracing via `RunTree` wrapper + `RemoteGraph(distributed_tracing=True)`) documented as follow-up** — architecturally cleaner (frontend becomes the parent trace, mirrors PR #43's gateway pattern) but adds parent-trace lifecycle complexity; defer until eval traces need Streamlit-side spans.
- [PR-3.5 namespace-scoped backend migration](creative_phases/orchestrator/migration_namespace_scoped_backend_2026-05-26.md) — **IMPLEMENTED 2026-05-26.** Hard cut from PR-3's callable-factory `backend=make_backend` (deprecated in deepagents 0.5.0, removal in 0.7.0 — `filesystem.py:737-748`) to the blessed pattern: backend INSTANCE with `namespace=lambda rt: ...` resolver. Per-call uid resolution lives in `resolve_memory_namespace` (reads `get_config()` since `_NamespaceRuntimeCompat` doesn't carry `.config`); per-graph anonymous-vs-authenticated dispatch lives in `build_orchestrator_backend(has_uid=...)` at catalog-build time. Same backend INSTANCE shared by `FilesystemMiddleware` (via `create_deep_agent(backend=...)`) and `LiveMemoryMiddleware` — pinned by `test_authenticated_filesystem_and_live_memory_share_same_backend_instance`. `LiveMemoryMiddleware` simplified (~30 LOC removed — both `_resolve_backend` and `_resolve_backend_for_agent` deleted; synthetic `ToolRuntime` construction gone). `test_backend_factory.py` deleted → `test_user_backend.py` (12 tests including capfd-pinned no-deprecation success signal). Verified: 145 passed, zero `warn_deprecated` in output, `-W error::DeprecationWarning:deepagents` sweep green. Implementation log appended to bottom of the doc.
- [PR-3 MemoryMiddleware design review (orchestrator deep-agent rewrite)](creative_phases/orchestrator/review_pr3_memory.md) — **IMPLEMENTED 2026-05-25.** 10-point sequential review of PR-3 (per-user memory via deepagents `MemoryMiddleware`). Shipped: `LiveMemoryMiddleware(MemoryMiddleware)` single subclass with `wrap_tool_call` post-edit sync (O(edits) backend reads, not O(turns)); `CompositeBackend` path-routes `/memory/` → `StoreBackend(namespace=(uid,"memory"))`; conditional middleware composition on `x-uid` presence (cache key gains `has_uid` bit); custom 24-line `<memory_guidelines>` for two-file model (`AGENTS.md` user-authored, `preferences.md` LLM-learned); LGP zero-config Postgres + `make_orchestrator(config, *, store=None, checkpointer=None)` optional kwargs for standalone (tests / in-process frontend). **38 PR-3 tests green** across 6 suites (12 unit live_memory + 3 prompt + 5 backend factory + 8 catalog + 7 e2e + 3 subagent isolation — exceeds plan's 22 with codex-driven add-ons). Binding revisions applied: `_get_store()` `RuntimeError` re-raised (Open Q4 REJECTED); `x-uid` allowlist-validated at factory boundary; `_backend_ref` pinned on subclass; `CompositeBackend` leading-slash preservation pinned by test; `raise ValueError` (not `assert`) guard against `memory=[...]` shortcut. Implementation log appended to the bottom of the design doc.
- [Codex adversarial review of PR-3 implementation plan](creative_phases/orchestrator/codex_adversarial_review_pr3_2026-05-25.md) — **REVIEWED + DISPOSITIONED 2026-05-25.** Codex (v0.133.0, gpt-5.5 xhigh) grounded against upstream `langchain-ai/deepagents` `main` + LangChain reference docs (local sandbox blocked `.venv` reads — re-verify locally). 5 TOP RISKS evaluated: (1) raw `x-uid` namespace validation — **ACCEPTED**, Phase B `make_backend` validates/encodes (strict allowlist or `sha256`); (2) fail-loud missing store — **ACCEPTED**, flips Open Q4, Phase A try/except must NOT swallow `_get_store()` `RuntimeError`; (3) `DeltaChannel` double-emit — **DROPPED**, wrapper is the only emission path under `wrap_tool_call` semantics, plan's `Command(update=...)` adds ToolMessage exactly once; (4) concurrent same-uid writes — **ACCEPTED (test only)**, Suite E adds test, no CAS in PR-3; (5) graph-cache `threading.Lock` — **DEFERRED**, compile is cheap (~20-80ms, lazy per `[CI-05]`), `threading.Lock` would block asyncio loop under langgraph-api 0.4.x sync constraint, revisit at PR-6 with async-factory after 0.7.x bump. SHARPENs accepted: `_backend` private (store own ref on subclass), `CompositeBackend` preserves leading slash (`/AGENTS.md`), `ToolRuntime` needs all 6 fields (prefer `request.runtime` directly), subagent state filter blocks `memory_contents` key but memory-derived text can still leak via `messages`/`structured_response`. Open Q final: Q1/Q2/Q3/Q5 agreed; Q4 REJECTED. Codex session `019e6005-e458-75d3-b3c6-68a45c6a4f50`.
- [PR-3 implementation plan — `LiveMemoryMiddleware` + per-user backend factory](creative_phases/orchestrator/implementation_plan_pr3_memory.md) — **IMPLEMENTED 2026-05-25** (phases A-F all green; see implementation log in `review_pr3_memory.md`). Phased build plan (A→F + 22 tests + docs) executed on `feat/orchestrator-deepagent`. A: `LiveMemoryMiddleware` subclass + custom prompt landed. B: `make_backend(rt)` landed under `backends/user_backend.py` with `x-uid` allowlist regex `^[A-Za-z0-9\-_.@+:~]+$` validating BEFORE namespace construction (codex Risk 1 fix). C: `make_orchestrator` wiring + `GraphCacheKey` 4→5-tuple with `has_uid` + `store`/`checkpointer` kwargs on `create_orchestrator_factory(...)` + explicit `raise ValueError` guard against `memory=[...]`. D: `langgraph.json` already forwards `x-uid` (line 38). E: **38 tests** shipped (plan target 22; the over-delivery comes from codex add-ons — async-refresh variant, `RuntimeError` re-raise contract, invalid-uid rejection, leading-slash preservation, factory-guard sanity, concurrent same-uid). F: docs (this file + `AGENTS.md` "implemented" + Setup table + pitfalls 14–16) + review_pr3_memory.md flipped to `status: IMPLEMENTED` + Implementation log appended. Open questions resolved: Q1 `raise` (asserts stripped under `-O`); Q2 raw `x-uid` for `has_uid`; Q3 duplicate uid-read; Q4 REJECTED (fail loud on `RuntimeError`); Q5 GP deny-rule deferred.
- [Twin Router KA source-awareness](creative_phases/twin_router/creative_phase_2026-05-23_router_ka_source_awareness.md) — **APPROVED — ready to implement 2026-05-23** as a pre-merge follow-up on `feat/ka_query_planner_improvements` (PR #43, 29 commits ahead). Surface KA `RetrieverEntry` descriptions in the twin router's `lisab` `ToolDefinition` so the router routes confidently to KA in `RagMode.KNOWLEDGE`. Single source of truth: `entry.description`. Mechanical projection at graph-build time, no duplicate strings. New `list_twin_ka_sources()` factored out of `build_twin_ka_entries()` to avoid network I/O at registry build (preserves the `[CI-05]` lazy-construction invariant). Conditional rebuild of the `rag` `AgentEntry`'s `tool_definition` in `_build_graph_for_permissions` — EXTERNAL/INTERNAL paths untouched. 6 tests cover env permutations + no-construction guard + KNOWLEDGE/EXTERNAL prompt assertions. ~40 LOC + ~80 LOC tests, ~1-2h.
- [KA Contextual-Content Refactor — reranker · granularity · truncation · synthesis](creative_phases/knowledge_agent/creative_phase_2026-06-17_ka_contextual_content.md) — **IMPLEMENTED 2026-06-17** (session `ka_refacto`; all 6 phases shipped, D1-D6 resolved, adversarial review passed, NOT yet committed). Routes the structured `metadata.content` blob three ways BY CONSUMER: (a) **reranker** scores the reconstructed contextual blob (`context_summary`+body via `_rerank_document_text`) so apcode/appName/title text helps the cross-encoder — RECONSTRUCT, not a stored `raw_content` field (no leak); (b) **compressor/synthesizer** get per-chunk `contextualized_content` inside each `<chunk>`/fact + page-shared remainder once per `<document>` (`page_shared_context`/`page_shared_summary` strip the per-chunk prefix); (c) **display/citations** get clean body. Synthesis truncation fixed (`_SYNTHESIS_SOURCE_CONTEXT_CHARS=1200`; review stays at hoisted `_REVIEW_SOURCE_CONTEXT_CHARS=320`). Subagent synthesis prompt: completeness-outranks-brevity + enumerations-in-full + grounded "Next steps". **Field-gate CLEARED:** prod field is `metadata.contextualisedContent` (British 's'), confirmed in `infra/elasticsearch/ingestion/indexer.py:107`. Orchestrator's KA is wired to `fast` → next-steps are model-derived (documented in KA `AGENTS.md`). `context_summary` kept FULL (additive only) for back-compat. Tests green (engine 3750 / core 1408).
- [ExpandNode Batch Fetch + fetch_target gating](creative_phases/knowledge_agent/creative_phase_2026-05-22_expand_batch_fetch.md) — **IMPLEMENTED 2026-05-22** on branch `feat/ka_query_planner_improvements`. ExpandNode collapsed N per-item ES queries into ~1 batched query per retriever. Codex review rejected a core-side `BatchDocumentProvider`; final design = `SupportsBatchFetch` marker protocol (ES batch methods) + engine-local `FetchExecutor` adapter as the single batch-vs-loop branch point (LightRAG/externals keep the per-item `DocumentProvider` loop unchanged). Phase 0 also gated `fetch_target` IDs to `DocumentProvider`-capable retrievers. Item 2 (inner loop in `search_depth=fast`) investigated only — deferred; `deep`+`max_iterations=1` is the existing path for "expansion without outer loop".
- [ElasticRagProxy — every expansion triggers HYBRID SEARCH (bug)](creative_phases/knowledge_agent/creative_phase_2026-05-23_elasticrag_proxy_hybrid_search_bug.md) — **RESOLVED 2026-05-23**. Root cause: `langgraph dev` instance at the frontend's gateway URL was running an older branch (pre-Phase-1, no `operation` dispatch). Re-deploying the up-to-date branch restored expected behavior. Bug was NOT in branch code. Same commit also lands a defensive guard: `_dispatch_search` now short-circuits empty / whitespace-only `query` to an empty success envelope without calling the retriever, emitting a WARNING — so any future skew where an old proxy + new gateway (or any field-drop bug) produces an empty query surfaces as empty results + a single log line instead of silent BM25-empty + vector-near-origin noise. Backwards-compatible: hand-rolled empty searches continue to receive a clean "no results" envelope, no exception raised. Tests: `TestElasticRagEmptyQueryShortCircuit` (4 cases).
- [ElasticRag Batch Fetch — Phase 4 implementation plan](creative_phases/knowledge_agent/creative_phase_2026-05-23_elasticrag_batch_fetch_phase4_plan.md) — **APPROVED — ready to implement 2026-05-23** on `feat/ka_query_planner_improvements` (HEAD `128986e`, 16 commits ahead, unpushed). Phases 1-3 of the elasticrag DocumentProtocol design are DONE (operation dispatch + envelope + proxy `DocumentProvider` + cross-package contract + KA integration test + single-fetch engineering example). Phase 4 = `SupportsBatchFetch` track: gateway grows three batch ops (`get_documents`/`get_chunk_contexts`/`get_chunk_ranges`) with array-of-records wire shape for ranges (NOT string-key tuple encoding — doc-ids may carry arbitrary characters); proxy declares `supports_batch_fetch=True` + implements `SupportsBatchFetch` + correlation-guaranteed envelope parsing; `_PROXY_OPERATIONS` grows in lockstep via the existing cross-package contract test; KA integration test proves `FetchExecutor` actually drives the batch path (1 call for N targets) AND falls back to per-item loop on `batch_fetch_unavailable`. **Positions on open questions**: (a) `batch_ceiling_hit` = **truncate-and-warn** (matches direct retriever); (b) `response_cap_hit` = **proxy-only** (gateway has no wire-size view); (c) warning surface = **`logger.warning` only**, no state threading; (d) example = **split** into new file. **Hidden drift flagged**: `_BATCH_FETCH_MAX_CHUNKS` lives on backing retriever, not gateway — gateway detects truncation post-hoc; `include_entity_childs`/`include_transversal` aren't in `_SCOPE_FIELDS` (pre-existing gap, NOT fixed in this phase). 4 commits expected; branch ends 20 ahead. ~7h focused.
- [ElasticRagRetriever + DocumentProvider / SupportsBatchFetch](creative_phases/knowledge_agent/creative_phase_2026-05-22_elasticrag_documentprotocol.md) — **APPROVED — ready to implement 2026-05-22 (refined 2026-05-23 after Codex round-2 review).** Extends the `elastic_rag` gateway-proxy adapter so it satisfies `DocumentProvider` + `SupportsBatchFetch`, unblocking the KA `ExpandNode` inner expansion loop for hosted-gateway deployments (today `_resolve_provider` skips the proxy). Endpoint audit confirmed the gateway is **search-only** — the fix needs a gateway-side change: an additive `operation` discriminator field dispatched inside the existing single `retrieve` node (`operation` omitted ⇒ `search`, wire-compatible). Round-2 review hardened the design: discriminated response envelope `{operation, ok, results?, error?, warnings?}`; concrete structured-error table (7 codes → exception class mapping); gateway boundary rejects `operation`-less doc-op requests + scope fields on any non-search op; `get_chunk_ranges` wire uses array-of-records (no string-key tuple encoding — robust to doc IDs with arbitrary characters); explicit operational-limits section (`batch_ceiling_hit` / `response_cap_hit` warnings, no automatic splitting in this phase). Capability still declared unconditionally — the imprecise marker trade-off is on the record (simplicity beats a probe round-trip; mock-misconfig surfaces as `NotImplementedError` on first fetch, not silent degradation). 4 phases, TDD. `docs/consuming/` deferred (project_status #17, scope-bypass on by-ID fetch).
- [Query-Planner Review Fixes — handoff](creative_phases/knowledge_agent/creative_phase_2026-05-22_query_planner_review_fixes.md) — **HANDOFF 2026-05-22.** Fix list for a new session after the 3-reviewer pass on the uncommitted KA query-planner + `MetadataScope` work. F1 (pyright blocker at `metadata_scope.py:230`), F2 (move scope normalization to clause-build time — covers `model_copy`/`model_construct` escape hatches + kills F1), I3 (per-axis normalizer-fn policy instead of hardcoded lower/upper casing), F3 (doc contradiction reword), A4 (style). F4/F6/F7 done this session (reverted accidentally-corrupted `baseline_report.json`; re-marked the real-LLM `test_baseline_report` as `integration_online`; CLAUDE.md lesson on scoping test runs).
- [ES-KA Catalog Flatten + Naming Convention](creative_phases/knowledge_agent/creative_phase_2026-05-23_es_ka_catalog_flatten.md) — **APPROVED 2026-05-23** for branch `feat/es-backend` (PR #44, 43 commits ahead, pushed). 3 commits to land *before* the upcoming KA rebase: (1) docs-only `packages/sta_agent_engine/AGENTS.md` § "File naming convention" (forward-only, prefix bare `graph.py`/`catalog.py`/`prompts.py`/`state.py` with package short name — fixes monorepo IDE search debt) + § "Catalog factory patterns" (pre-built / `@cache` getter / `make_<name>(config[, runtime])` — version caveat: 1-arg on langgraph-api 0.4.x prod, 2-arg ≥ 0.7.x for Studio schema-read fast path) + `[CI-07]` known-issue entry for grandfathered bare names; (2) refactor `deepagent_es/catalog.py` 133 lines → < 30: new `es_ka_graph.py:get_es_knowledge_graph(...)` inlines today's `build_deepagent_es_graph` + `assemble_graph`, drops `llm_provider` kwarg, return type `CompiledStateGraph`; `catalog.py` shrinks to `@cache`'d `get_es_knowledge_graph_instance()` + TODO seat for future per-request factory. Caller updates: `langgraph.json`, `experiments/graph_configs/graphs.jsonl` (KA-conflict hotspot — single row), example, contract test, docstrings. CI-05 preserved. Package rename `deepagent_es` → `es_knowledge_agent` is out-of-scope (separate follow-up PR). Function names land in post-rename form now to avoid renaming twice.
- [KA Improvements Brainstorm — 4 threads (env model configs / trace propagation / messages-first I/O / subagent mode)](creative_phases/knowledge_agent/creative_phase_2026-05-18_ka_improvements_brainstorm.md) — **READY FOR IMPLEMENTATION 2026-05-19** on branch `ka_improvements_brainstorm`. Step 1 = three independent PRs (B trace propagation via `config.configurable` body injection mirroring `RemoteGraph(distributed_tracing=True)`; D subagent mode reusing existing concise prompt + dropping `[Fn]` citation contract; A env-driven per-task overrides + stripping hardcoded `"provider": "llmaas"` from package defaults so KA config goes provider/model-agnostic — 5-layer resolution ladder with `LLM_PROVIDER` engine-wide env as layer 5 fallback). Step 2 = Thread C (messages-first I/O) deferred — needs fresh creative phase covering dual-read shims everywhere + state versioning + checkpointer replay + frontend citation contract. Codex adversarial review + user-direction history preserved as appendices in the doc; "FINAL IMPLEMENTATION PLAN" section at the top is the source of truth.
- [ElasticRagRetriever — gateway-proxy adapter](creative_phases/knowledge_agent/creative_phase_2026-05-16_elastic_rag_retriever_adapter.md) — PROPOSED 2026-05-16, **in implementation** (worktree `ka_scope_elastic_rag_adapter`). Client-side adapter that wraps the deployed `elastic_rag` LangGraph gateway as a `BaseRetriever[RetrievalChunk]` + `SupportsMetadataScope`. Planner output + codex review (verdict REWORK) synthesized into v2 plan. Lives in `sta_agent_core` (vendor-clean — reuses `AsyncHttpAdapter`, NO `langgraph` dep). POST `/runs/wait` with flat scope-field decomposition. Engineering-only release — `docs/consuming/` deferred until project_status #17 (server-side scope enforcement). Codex non-negotiables: package boundary (core not engine), explicit kwarg allow-list (typed signature, no `**kwargs`), cross-package parity contract test. ~210 LOC + ~330 LOC tests + ~150 LOC example. Phases 1–3 in-scope; Phase 4 deferred.
- [ES mapping alignment — local vs production](creative_phases/knowledge_agent/creative_phase_2026-05-15_es_mapping_alignment.md) — **IMPLEMENTED 2026-05-15** (worktree `es_mapping_alignment`, 2 commits: alignment + review polish). Production index mapping audited 2026-05-15; 12 mismatches between local ingestion + `ElasticFieldConfig` defaults and prod resolved. `metadata.auid` (not `apcode`) is the canonical apcode field in prod (one-line `ElasticFieldConfig` default flip); `metadata.content` is a structured `\n`-separated blob (summary + Url page + Application + name + title + content body) while `metadata.contextualisedContent` carries just the summary; `metadata.entity.name` reverted to `keyword + lowercase_normalizer` to match prod; `int8_hnsw` quantization + `_source.excludes:[embedding]` added; `metadata.chunk_size{char, token_embed_model, token_llm}` replaces flat `start/end/token_count`. KA + ElasticRetriever surface unchanged — only retriever code changes are `extract_chunk_body()` helper + apcode extraction with boundary normalization in `_default_result_mapper`. 453 ES retriever + 696 KA tests green. **Live ES re-ingest deferred** (next-session: `make es/up` + `cli ingest --force --clear-cache` + smoke + probes).
- [DeepAgent with Elasticsearch as Virtual Filesystem (v3.1)](creative_phases/knowledge_agent/creative_phase_2026-05-14_deepagent_es_filesystem.md) — PROPOSED 2026-05-14, **near-term improvement opportunity** (three-round codex-validated 2026-05-14/15; design verdict "sound, ready for phase 0"). Parallel/alt agent on `langchain-ai/deepagents` 0.6.1: ES index mounted at `/knowledge/es/` via built-in `CompositeBackend`; `StateBackend` default for `/notes/` scratchpad. Built-in FS tools only (`ls`/`read_file`/`grep`/`glob`) with `custom_tool_descriptions` seam (`middleware/filesystem.py:580`) for ES semantics — no fork. `grep` grammar `[<mode>:][@{axis=val,...}] <query>` (modes `auto`/`fts`/`knn`/`h`/`hr`); runtime query scope **boost-only** via `MetadataScope.add_boosts()` — mirrors production KA scope (no path-as-scope, no regex). `read_file` accepts doc name OR pageId via single OR-query + sidecar pageId terms agg; multi-doc match returns all chunks (inline-labeled per pageId) + structured warning + `NOTE:` header — non-blocking. Build-time scope ceiling at backend ctor; user scope deferred until twin_router→entity; phase 3 hosted-graph demoted until user scope ships (`AGENTS.md:108` trust-boundary policy). New upstream additive: `ElasticRetriever.get_chunks_by_reference()` (~50 LOC). Phase 0: ~370–500 LOC + one-shot pre-spike `terms` agg gate to verify `metadata.doc.keyword` shape (path-like vs flat/Confluence vs opaque). 13 risks tracked with codex-validated severities.
- [DeepAgent-ES Phase 0 — implementation plan](creative_phases/knowledge_agent/creative_phase_2026-05-21_deepagent_es_phase0_plan.md) — **APPROVED 2026-05-21, in implementation** on branch `feat/es-backend` (cut from `worktree-deepagent-es-preflight-probe` so v3.7 doc + probe travel with it). Step-by-step plan for the deepagent-es spike, grounded in v3.7 Decision Sync + `preflight_results_2026-05-19.md`. New decision: **composable per-backend tool descriptions** — STA-authored thin `BASE_TOOL_DESCRIPTIONS` + per-backend `tool_description_fragments(prefix)` + `compose_tool_descriptions()` composer (`FilesystemMiddleware` takes one flat dict; assembly happens before middleware build). Order B→C→D→E: `MetadataScope.narrow_with` + `_EMPTY` sentinel (B3 contract test gates the ship) → `ElasticMetadataValueResolver.list_values` → example `deepagent_es_filesystem_example.py` → equivalence/addressability tests. Open items: 2-segment resolver exact-match spec (pin during D2), optional P9 `after_key` recheck (non-blocking), C1 (`title` text-only, frozen mapping — fixed constraint), upstream `STABackend` protocol (Phase 1).
- [Elastic ingestion — metadata enrichment for KA scope testing (build-time + runtime query)](creative_phases/knowledge_agent/creative_phase_2026-05-14_elastic_metadata_enrichment.md) — IMPLEMENTED (code-side) 2026-05-14 in worktree `elastic_metadata_enrichment_doc`. Sources rewritten (11 entries: 3 DU package apcodes + 2 transversal + Foundation twin for collision B + 3 Acme tenant slices). Chunker now handles `path.is_file()` for single-file sources. Local mapping reverted `entity.name` to `keyword + lowercase_normalizer` per 2026-05-15 audit (the 2026-05-14 `text` revert was based on a transcription gap — see archived [resolver aggs gap](_archive/project_resolver_aggs_entity_name_text.md)). 15 Acme markdown fixtures authored. 16 offline tests green. P1–P7 acceptance probes script at `infra/elasticsearch/probes/metadata_scope_smoke.py` — **live verification still pending** (needs `make es/up` + ingestion + probe run).
- [KA Scope Hardening — codex review follow-ups](creative_phases/knowledge_agent/creative_phase_2026-05-13_ka_scope_hardening.md) — IMPLEMENTED 2026-05-15. Groups A (#4/#5/#7/#8 — `823810c`), B (#6 — `d64e1de`), C (#3 — `be16855`), D (#1 — `SupportsMetadataScope` runtime-checkable Protocol + ES `ClassVar[Literal[True]]` marker + tool-factory build-time gate), E (#2 — composite `after_key` drain + flat-terms `sum_other_doc_count` WARN + `_DEFAULT_AGG_SIZE` 5000→65535) all landed; vocabulary rename (Layer 1/2/3 → "build-time / user / runtime query scope") landed in Group D's prep commit. **Open caveat (B1):** Group E's `_DEFAULT_AGG_SIZE=65535` × 4 aggs may breach ES `search.max_buckets` (per-response, default 65 536); verification deferred until prod cardinality numbers arrive (`infra/elasticsearch/probes/cardinality/`). Round-3 review at `.claude/PRPs/reviews/group-e-review.md`.
- [KA Scope Architecture — Three-Layer Model with Value Resolver](creative_phases/knowledge_agent/creative_phase_2026-05-11_ka_scope_three_layer.md) — IMPLEMENTED 2026-05-11 (commits `bde247a..31ce85e` + example `examples/sta_agent_engine/knowledge_agent_three_layer_scope_example.py`; L2 explicitly deferred). Originally PROPOSED v2 2026-05-11 (refined after adversarial review; v1 narrative kept inline with `[SUPERSEDED]` markers). **Supersedes** the same-day `elastic_retriever_ka_tuning` doc. First step in #45. v1 ships **L1 only** (build-time `entry.default_scope` ceiling) + **L3 boost-only** LLM args (`apcode`/`app_name`/`entity`) cleaned via new `MetadataValueResolver` Protocol with `ElasticMetadataValueResolver` impl (composite name↔id agg ∪ `entity.childs` terms agg, lazy 8am refresh with `asyncio.Lock` + jitter, 2-bucket fuzzy: codes τ=90 / names τ=80). L3 writes to `*_boost` axes only — never filter — so cross-references in other apcodes still surface via BM25 text-match (`elastic_retriever.py:497-520`). **L2 `user_scope_mode` deferred** until twin_router plumbs user→entity. Warnings → ToolMessage tail + `findings.metadata["warnings"]` + structured `logger.warning`. ~430 LOC / 6 phases. Unblocks #16 / #17 / #44.
- [ElasticRetriever → KA Integration & Tuning](creative_phases/knowledge_agent/creative_phase_2026-05-11_elastic_retriever_ka_tuning.md) — **SUPERSEDED 2026-05-11** (kept for design history). Earlier same-day doc; assembled scope at caller-context with merge tables. Replaced by the metadata-scope model above (originally framed as "three-layer" — see vocabulary banner in the design doc).
- [Twin Router KA Revamp — replace Adaptive RAG with the Knowledge Agent](creative_phases/twin_router/creative_phase_2026-05-21_twin_router_ka_revamp.md) — **PROPOSED 2026-05-21** on branch `feat/twin_router_revamp_rag_to_ka`. Swap `AdaptiveRagGraph` behind the `lisab` RAG tool for the Knowledge Agent via a new `RagMode.KNOWLEDGE` + `KnowledgeAgentRagStrategy` (sub-agent-behind-a-tool, lazy build per CI-05, answer-mode + `subagent_mode`). One KA for prod users (non-prod keep no RAG — already automatic via `POLICIES["non-prod"]=frozenset()`); KA carries two direct-`ElasticRetriever` entries — broad `twin_docs_general` (unscoped) + `twin_project_knowledge` (twin entity/apcode-scoped, `include_transversal=True`). Never the `elastic_rag_proxy`. Twin scope values anonymized via `TWIN_SCOPE_*` env arrays (`TwinScopeSettings` Pydantic BaseSettings, declared empty in `.env.example`). `Citation→RagSource` adapter keeps the twin-router contract. Per-user entity scoping deferred. Adaptive-RAG path deprecated, removed in a later PR. 4 phases.
- [Twin Router Robustness — habilitation toggle, multi-model native, layered timeout defense](creative_phases/twin_router/creative_phase_2026-05-08_twin_router_robustness.md) — Three-phase short-term hardening (#43 + #46 + #47) before the deep-agent rewrite (#14). Phase 1 IMPLEMENTED 2026-05-10; Phase 2 IMPLEMENTED 2026-05-10 (multimodal guard middleware — strip image parts for non-multimodal models, GK tool dual-call site covered, 24 tests); Phase 3 PARTIAL (L1 + L2 shipped 2026-05-10; L3 FastAPI middleware abandoned same day — wrong layer for run-level timeouts in LangGraph Platform's queue model).
- [Twin Router Robustness Phase 1 — habilitation bypass implementation log](creative_phases/twin_router/creative_phase_2026-05-10_robustness_phase1.md) — IMPLEMENTED 2026-05-10. HABILITATION_BYPASS env + DeploymentConfig toggle, BypassHabilitationProvider with role typo trap, 21 new tests, twin_router README.
- [Twin Router Robustness Phase 3 — layered timeout defense implementation log](creative_phases/twin_router/creative_phase_2026-05-10_robustness_phase3.md) — PARTIAL 2026-05-10. L1 (httpx stall fence in `create_chat_model`) + L2 (operator guide / strategic-catch recipe) shipped; L3 (FastAPI `RequestTimeoutMiddleware`) shipped then rolled back same day. Run-level wall fence pending as a separate follow-up.
- [Retriever-Owns-Expansion — active thread](creative_phases/retriever_owns_expansion/README.md) — Authoritative architectural reference (step 4 v3 amendment) + ongoing execution plan (step 8). Project_status #30 closed 2026-05-07; remaining follow-ups are #34 / #35 / #39.
- [Retriever Eval Sweep — Expansion × Fusion matrix](creative_phases/elastic_rag/creative_phase_2026-04-17_retriever_eval_sweep.md) — FUTURE. Post-F6 plan to run the existing retriever dataset across every `(expansion_hint × fusion_strategy)` combination.
- [Twin Router Habilitation](creative_phases/twin_router/creative_phase_2026-03-27_twin_router_habilitation.md) — Authorization for twin router: registry, factory closure, auto bias rules, `DeploymentConfig`.
- [Habilitation Refactor](creative_phases/twin_router/creative_phase_2026-04-03_habilitation_refactor_code_quality.md) — Fail-open fix, registry re-exports, prompt-builder unification, persona flow cleanup.
- [Auth Evolution](creative_phases/twin_router/creative_phase_2026-03-27_auth_evolution.md) — Auth layer progression: headers → FastAPI middleware → native `@auth` → `@auth.on` → policy engine.
- [Retriever & LightRAG Architecture](creative_phases/lightrag/creative_phase_2026-02-09_retriever_and_lightrag_architecture.md) — Foundational retriever protocol + LightRAG integration. Cross-referenced by knowledge_agent/, roadmap/, `.cursor/rules/sta-agent-core.mdc`.

## Archived

- [Retriever-Owns-Expansion — experiment history](creative_phases/_archive/retriever_owns_expansion/HISTORY.md) — Archived 2026-04-17. Steps 1–3 design thrash + shipped execution logs (5, 6, 7, 9).
- [Elastic RAG Workflow (Phases 1–7+9 IMPLEMENTED)](creative_phases/_archive/elastic_rag/creative_phase_2026-04-12_elastic_rag_workflow.md) — Archived 2026-05-07. RRF-first retriever + LangGraph gateway + shared factory + multi-graph LGP deploy + owned-vs-rented consumer access + `close()` cascade.
- [Next Session Brief — Phase 5 post-implementation](creative_phases/_archive/next_session_brief.md) — Archived 2026-05-07. Outdated since Phase 5 + Cycle F + step 9 closed; superseded by `project_status.md`.

## Cross-cutting / synthesis

- [evals/creative_phase_2026-03-01_evaluation_doc.md](evals/creative_phase_2026-03-01_evaluation_doc.md) — Evaluation framework synthesis (eval-driven development, three pillars, dataset/evaluator design). Continue iterating per project_status #26.

## Project

- [Corpus profile — sparse, OPS-in-tech](project_corpus_profile.md) — French IT/ops banking back-office corpus; shapes expansion tuning, eval thresholds, and DOMAIN_ENTITIES priorities.

## Reference

- [AGENTS.md](AGENTS.md) — Memory-bank conventions (lifecycle, naming, what NOT to put here)

-------

memory_bank/creative_phases/creative_phase_2026-05-30_reasoning_compatibility.md
----
# Creative Phase 2026-05-30 — Reasoning compatibility (vLLM / LiteLLM gateway)

**Status:** ✅ **REVIVED 2026-07-13** — superseded by
[[creative_phase_2026-07-13_reasoning_effort_config]], which implements the
"declarative per model family" design this doc prescribed (per-family table →
`build_reasoning_kwargs` / `reasoning_effort` on `create_chat_model`). The
empirical findings below remain the grounding evidence.
**Origin:** Split out of the v2.4 provider-settings work, which **deferred
`reasoning_effort`** after a hands-on probe (see
`creative_phase_2026-05-17_dynamic_settings_registry.md` § Decision 5).
**Evidence:** `experiments/langchain/chatopenai/reasoning_script.py` (makes real
paid calls — never run silently).

---

## Why this is parked

We considered adding a uniform `reasoning_effort` knob to `BaseProviderSettings`
so callers could dial reasoning per request. A probe of the internal LLMaaS
gateway showed there is **no uniform lever** to expose — the behavior is
per-model-family and partly an ops/proxy concern, not a clean library surface.
So we deferred it rather than ship a knob that silently does nothing. This doc
captures what we learned so the next attempt starts from facts, not assumptions.

---

## Observed behavior (internal gateway, 2026-05-29, preliminary — user-confirmed)

Probed gpt-oss, Qwen3, and Nemotron via `create_chat_model` / `ChatOpenAI`.

1. **Thinking is ON by default** for all three — they reason with no reasoning
   argument at all.
2. **Top-level `reasoning_effort` is a NO-OP** for all three. langchain-openai
   1.1.9 forwards it straight into the Chat Completions payload; its
   `reasoning_effort → reasoning={"effort": ...}` rewrite at
   `langchain_openai/chat_models/base.py:3861` is **Responses-API only** and does
   **not** fire for Chat Completions / vLLM. So the no-op is the gateway's, not a
   client bug.
3. **Non-streaming: reasoning is parseable uniformly** from all three at
   `response.choices[0].message.provider_specific_fields["reasoning"]`.
   `provider_specific_fields` is a **LiteLLM** field → the gateway is fronted by a
   **LiteLLM proxy** (litellm 1.80.0 in the venv).
4. **Streaming: only gpt-oss surfaces reasoning** in chunk deltas
   (`chunk.reasoning`). Qwen and Nemotron reason but don't emit it in the stream —
   a LiteLLM gap, [#20246](https://github.com/BerriAI/litellm/issues/20246):
   LiteLLM reconstructs `reasoning_content` from inline `<think>` tags only in its
   non-streaming path, unless `merge_reasoning_content_in_choices=True`, which has
   its own hosted_vllm bug [#9578](https://github.com/BerriAI/litellm/issues/9578).
   The same Nemotron model **did** stream reasoning chunks via OpenRouter → the
   gap is our **local LiteLLM-proxy config**, not vLLM.
5. **The only request-time knob that moved anything** was Nemotron's
   `extra_body={"chat_template_kwargs": {"low_effort": true}}` — it lowered
   reasoning to **low effort** but did **not** disable it. An **effort toggle, not
   an on/off**. No reliable off switch was found. `enable_thinking` true/false
   toggles thinking for qwen3/nemotron.
6. Control is **per-model-family**, with **no OpenAI-style uniform lever**, and is
   **not versioned by vLLM release**.

---

## What today's stack already gives the builder

`create_chat_model(provider, *, tier=..., **kwargs)` forwards arbitrary kwargs
(incl. `extra_body`) to the client, so reasoning control is reachable **today**
without a settings field:

```python
# Nemotron: drop to low effort (effort toggle, not on/off)
llm = create_chat_model("nemotron",
        extra_body={"chat_template_kwargs": {"low_effort": True}})
```

The `thinking` capacity tier (`thinking_model`) is a **model slot**, not a
reasoning switch — it points at the deployment's best reasoning model and does
not imply programmatic reasoning control.

---

## When revived — design constraints to honor

- **No uniform `reasoning_effort` field** unless a uniform lever actually exists
  on the gateway. If revived, make it **declarative per model family** (e.g. a
  mapping family → `extra_body` payload), not a single passthrough.
- A passthrough `reasoning_effort` / `OPENAI_EFFORT` is **rejected** — proven a
  no-op across all three probed models.
- The **streaming-reasoning gap is an ops/proxy issue** (LiteLLM config), not a
  library scope item. Fix or accept it at the gateway, don't paper over it in the
  client. Re-confirm against the current LiteLLM version before designing around
  it (#20246 / #9578 may be resolved upstream).
- Re-run the probe (`experiments/langchain/chatopenai/reasoning_script.py`) on the
  then-current gateway before committing to any design — these findings are
  preliminary and gateway-config-dependent.

---

## Open questions (carry forward)

1. Is there a gateway/LiteLLM setting that exposes a real off switch for reasoning
   per request, or is on-by-default fixed at deploy time?
2. Should reasoning control live on provider settings at all, or stay a per-call
   `extra_body` concern owned by the builder?
3. Will `merge_reasoning_content_in_choices=True` (once #9578 is fixed) make
   streaming reasoning uniform across families, removing the per-family special
   case?

-------

memory_bank/creative_phases/creative_phase_2026-07-13_reasoning_effort_config.md
----
# Creative Phase 2026-07-13 — Declarative reasoning-effort configuration

**Status:** ✅ APPROVED — implementing (same session).
**Origin:** Revives [[creative_phase_2026-05-30_reasoning_compatibility]] (⏸️ parked
"no uniform lever — if revived, do it declaratively per model family"). This design
is exactly that revival: a per-family declarative table, no uniform passthrough.
Also closes the `reasoning_effort` deferral from
[[creative_phase_2026-05-17_dynamic_settings_registry]] § Decision 5.
**Owner decisions:** locked 2026-07-06 → 2026-07-13 (see § Decisions).

---

## Problem

`create_chat_model` had no reasoning control. The models in use expose
**incompatible** knobs:

| Model | Mechanism | Values |
|---|---|---|
| mistral-small / mistral-medium-3-5 | top-level `reasoning_effort` (via `model_kwargs` — ChatMistralAI 1.1.4 has no native field, but `model_kwargs` flattens into the payload) | `"none"` / `"high"` (binary; Mistral docs sanction only these) |
| Nemotron-3-Super-120B | `extra_body.chat_template_kwargs` | `enable_thinking` bool + `low_effort` flag |
| Nemotron-3-Ultra-550B | `extra_body.chat_template_kwargs` | `enable_thinking` + `medium_effort` + `force_nonempty_content` (tool-calling) |
| Qwen3 (future) | `extra_body.chat_template_kwargs.enable_thinking` + top-level `extra_body.thinking_budget` (vLLM) | bool + token budget |
| Real OpenAI | native `reasoning_effort` field | `minimal/low/medium/high` |

Prior probe (2026-05-29, `experiments/langchain/chatopenai/reasoning_script.py`):
thinking is ON by default for gpt-oss/Qwen3/Nemotron on the LiteLLM-fronted
gateway; **top-level `reasoning_effort` is a no-op there** (langchain-openai's
`reasoning_effort → reasoning={...}` rewrite is Responses-API-only); the only
knob that moved anything was Nemotron's `chat_template_kwargs`.

## Design (final, post-review)

**One plain dict table + 3 pure functions** in `sta_agent_engine/models/reasoning.py`.
No Protocols, no adapter classes, no sink DSL — a family maps effort → **literal
request kwargs**:

```python
_FAMILIES = {
    "mistral":        {rungs: off→ {model_kwargs: {reasoning_effort: "none"}}, high→ {...: "high"}},
    "nemotron-super": {rungs: off/low/high → extra_body.chat_template_kwargs …},
    "nemotron-ultra": {rungs: off/low/medium/high → …},
    "qwen3":          {rungs: off/medium/high/xhigh → … + thinking_budget},
    "openai":         {rungs: {} + native passthrough (reasoning_effort=<effort>, silent)},
}
```

### Effort → wire mapping (as specified by Badr)

| effort | mistral | nemotron-super | nemotron-ultra | qwen3 |
|---|---|---|---|---|
| `off` | `reasoning_effort="none"` | `enable_thinking=False` | `enable_thinking=False` | `enable_thinking=False` |
| `low` | ⚠︎ passthrough `"low"` | `enable_thinking, low_effort` | `enable_thinking, medium_effort, force_nonempty_content=False` | ⚠︎ not applied |
| `medium` | ⚠︎ passthrough | ⚠︎ not applied | `enable_thinking, medium_effort, force_nonempty_content=True` | ⚠︎ not applied |
| `high` | `reasoning_effort="high"` | `enable_thinking` (full) | `enable_thinking, force_nonempty_content=True` (full) | `enable_thinking` |
| `xhigh` | ⚠︎ passthrough | ⚠︎ not applied | ⚠︎ not applied | ⚠︎ not applied |

**Correction (same day):** the qwen3 built-in originally shipped Badr's graded
`thinking_budget` rungs (medium/high/xhigh = 2048/4096/8192). Checked against
the real `Qwen/Qwen3.6-27B` card: it documents **only** binary
`chat_template_kwargs.enable_thinking` (thinking on by default) plus
`preserve_thinking=True` for agent-turn trace retention. Budgets are a
serving-stack feature (vLLM `thinking_token_budget`, version-dependent) — the
"knob that silently does nothing" anti-goal — so the built-in is binary and the
budget rungs live in `docs/consuming/reasoning.md` as a
`register_reasoning_family` override to apply after a gateway probe.

**Matching semantics (post slug-variance round):** patterns are case- and
separator-insensitive (`- _ . : /` + spaces stripped from both sides), so
`qwen3.6` ≡ `qwen3-6` ≡ `Qwen/Qwen3.6-32B` ≡ `qwen3:32b`. `match` accepts a
flat tuple (one AND-group) or a tuple of tuples (OR of AND-groups) for alias
dialects; lone strings are coerced to 1-tuples (missing-comma trap guarded).

⚠︎ = `UserWarning` listing the family's supported set; native-sink families
(mistral, openai) pass the raw value through so the API validates it; translated
families (nemotron, qwen) inject nothing (there is no field to receive the raw
value). Ultra's `low` = Badr's spec: medium-effort thinking **without** forcing
non-empty content on tool-call turns; `medium`/`high` carry
`force_nonempty_content=True` baked in (SGLang tool-calling requirement — no
runtime conditional needed since the flag is per-rung declarative).

### Public surface (decision 1A)

- `create_chat_model(..., reasoning_effort="high", reasoning_family=None)` —
  popped from config after model-name resolution, translated, merged. `None` =
  inject nothing = byte-identical current behavior.
- `build_reasoning_kwargs(model, effort, *, provider=None, family=None)` — pure
  helper; splat into `.bind(**kw)` / `.invoke(msgs, **kw)` for per-call control.
- `supported_reasoning_efforts(model, ...)` → frozenset of rung names.
- `register_reasoning_family(name, rungs, *, match_substrings, provider_substrings, native_path)`
  — consumer extension point; may override built-ins (last-write-wins).
- `resolve_reasoning_family(model, ...)` — debugging/verification.
- Family matching: explicit `reasoning_family` kwarg > provider substring >
  model-name substrings (all must match, case-insensitive). Explicit pin exists
  because gateways alias model names (`chat-default` → really Nemotron).

### Merge semantics (decision 2A)

Deep-merge; **explicit caller kwargs win over table-emitted values on leaf
conflicts** + `UserWarning` naming overridden paths. Rationale: raw
`extra_body`/`model_kwargs` are the escape hatch; an escape hatch that can be
overridden isn't one. Non-conflicting keys compose (partial manual `extra_body`
merges with table output).

### Error posture

Warn-not-raise everywhere (Badr's call, overriding the adversarial review's
raise-by-default): unknown family → warn + native passthrough
(`reasoning_effort=<effort>` top-level — honored by real OpenAI, known no-op on
LiteLLM/vLLM gateways per the 2026-05 probe; the warning says so). Unsupported
rung → warn + passthrough (native-sink) / not-applied (translated). Model
construction never fails because of reasoning.

## Decisions log

| # | Decision | Choice |
|---|---|---|
| — | Mechanism | Plain dict table (adversarial review's callable-adapter Protocol rejected as over-engineering after simplification round; declarative table is exactly what the 2026-05-30 doc prescribed) |
| — | Param name | `reasoning_effort` (matches Mistral + OpenAI native vocab; identity-degrades for unknown families so non-breaking) |
| — | Helper name | `build_reasoning_kwargs()` |
| 1A | Extension | `register_reasoning_family()`; table private; `reasoning_family` kwarg; settings/env fields **deferred** |
| 2A | Conflicts | Explicit caller kwargs win + warning |
| 3C | Tests | Units + wiring + request-payload-level asserts (httpx-timeout lesson) + `integration_online` smoke (manual-only) |
| 4A | Footguns | Document only: (a) per-turn effort flips bust vLLM/SGLang prefix caches (chat_template_kwargs re-renders the template); (b) upstream `langchain_mistralai` blanks assistant content (thinking included) on tool-call turns (`chat_models.py:474-478`) — degradation-only, converter patch deferred until online evidence |

## Adversarial review trail

Codex REWORK verdict (2026-07-06) — accepted: don't hijack native kwargs
semantics (resolved: identity-degrade), no silent clamping (resolved: warn +
never clamp — the earlier "positive stays positive" projection was CUT), match
on more than name-regex (resolved: family pin + provider match), sampling
auto-coupling CUT, resolver reusable per-call (resolved: `build_reasoning_kwargs`
is the shared path). Rejected: hard-raise default (Badr chose warn), discriminated
union `ReasoningBudget/ProviderReasoning` (YAGNI — budgets fit in table cells,
cf. qwen3), callable-adapter registry (dict cells cover all current cases).

## Verification items (resolved during implementation)

- ChatMistralAI 1.1.4 has **no** `reasoning_effort` field → route via
  `model_kwargs` (flattens into payload via `_default_params`, `chat_models.py:585`). ✅
- `force_nonempty_content` needs no runtime tool-detection: per-rung literal
  (Badr's spec puts it on medium/high=True, low=False). ✅
- ChatMistralAI replays thinking blocks in history automatically (parse:
  `:259-275`; serialize: `:480-486`) EXCEPT tool-call turns (`:474-478`). ✅

## Test plan

`tests/test_ai_engine/models/test_reasoning.py` (pure units + payload-level),
`test_create_chat_model_dynamic.py::TestReasoningEffort` (wiring; includes
byte-identical no-effort regression), `test_reasoning_smoke_online.py`
(`integration_online`, never auto-run). Re-run probe guidance from the 2026-05-30
doc satisfied by the smoke.

## Post-commit review round (2026-07-13, after b073a39f)

Multi-perspective review (python/general/tests/silent-failures/docs/simplify) of
the shipped commit; Badr triaged. Fixed in the follow-up commit:

- **C1** Devstral/Magistral dispatch to ChatMistralAI but didn't match the
  `mistral` family → raw `"off"` reached the wire instead of `"none"`. Fix:
  mistral match = `(("mistral",), ("devstral",), ("magistral",))`, mirroring
  `_is_mistral_model`. (`ministral` matches neither — consistent both sides.)
- **C2** Blank effort (`""`/whitespace — e.g. empty env-var default) rode the
  native lane onto the wire with no warning. Fix: blank = unset = `{}`.
- **I2** `reasoning_family` without `reasoning_effort` was a silent no-op —
  now warns (still stripped).
- **I4** Doc/wire mismatches: chat-models.md claimed a `thinking_budget`
  translation that doesn't exist; per-call splat claim was false for mistral
  (`model_kwargs` is constructor-only — call-time kwargs post a literal
  `"model_kwargs"` JSON key); Ultra `low` sends `force_nonempty_content=False`
  (explicit, not absent); `xhigh` dropped from the documented vocabulary (no
  built-in rung defines it).
- **I5** `register_reasoning_family` now rejects empty/normalize-to-empty
  match or provider patterns (`""` matches every model). Empty tuple = legal
  pin-only family.
- **I6** Test hardening: unmocked end-to-end wiring tests (real creators →
  payload/params), native top-level payload assert, cross-family
  provider-beats-name precedence, overlapping-registration insertion-order pin
  (+ docstring note: new overlapping family never wins — re-register under the
  built-in's name), provider lone-string coercion, smoke drift guard moved out
  of the `integration_online` marker so it runs in the default suite.

- **I3** (fixed after Badr's go, separate commit): warnings in reasoning.py
  now use `skip_file_prefixes` (Python 3.12) pointing past the models package
  dir, so attribution lands on the consumer's call site regardless of
  entry-point depth, and the default once-per-location filter dedupes per
  consumer line instead of collapsing all sites onto one internal line.
  Attribution pinned by tests (direct + routed through create_chat_model).

Deferred by Badr:

- **I1** Over-broad matching (`qwen3` captures `qwen:32b`≈Qwen1.5 /
  Qwen3Guard / qwen3-embedding; `("gpt",)` captures gpt-neox/gpt-j/gpt4all,
  suppressing the unknown-family warning) — only gpt-oss is in scope
  internally; revisit if the model mix changes.

## Deferred / follow-ups

- Settings-level defaults (`{NAME}_REASONING_EFFORT` / `_REASONING_FAMILY` env) — 1B, on demand.
- langchain-mistralai tool-call thinking-drop converter patch — only if online smoke shows degradation.
- Streaming-reasoning gap = ops/LiteLLM concern (#20246/#9578), out of library scope (unchanged from 2026-05-30).
- Review I1 above if deployment scope changes.

Consumer doc: `docs/consuming/reasoning.md`. Pitfall entries:
`.claude/skills/langgraph-agent-builder/references/known-pitfalls.md`.

-------

packages/sta_agent_engine/src/sta_agent_engine/models/__init__.py
----
"""Model factory functions for LLM, embedding, and rerank providers."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .capabilities import MULTIMODAL_MODELS, is_multimodal


__all__ = [
    # Capability registry
    "MULTIMODAL_MODELS",
    "is_multimodal",
    # Chat
    "create_chat_model",
    "CustomChatModel",
    # Reasoning-effort configuration
    "build_reasoning_kwargs",
    "register_reasoning_family",
    "resolve_reasoning_family",
    "supported_reasoning_efforts",
    # Fake models (for offline graph integration testing)
    "AgentIntegrationModel",
    # Fake Streaming (for custom message streaming via stream_mode="messages")
    "FakeStreamingLLM",
    # Embedding
    "create_embedding_model",
    # Rerank (HTTP-based - default)
    "create_rerank_model",
    "create_async_rerank_model",
    "RerankClient",
    "RerankResponse",
    "RerankResult",
]

_LAZY_EXPORTS = {
    # Chat
    "create_chat_model": ".custom_chat_model",
    "CustomChatModel": ".custom_chat_model",
    # Reasoning-effort configuration
    "build_reasoning_kwargs": ".reasoning",
    "register_reasoning_family": ".reasoning",
    "resolve_reasoning_family": ".reasoning",
    "supported_reasoning_efforts": ".reasoning",
    # Fake models (for offline graph integration testing)
    "AgentIntegrationModel": ".fake_models",
    # Fake Streaming (for custom message streaming via stream_mode="messages")
    "FakeStreamingLLM": ".fake_streaming_llm",
    # Embedding
    "create_embedding_model": ".embedding_model",
    # Rerank (HTTP-based - default)
    "create_rerank_model": ".rerank_model",
    "create_async_rerank_model": ".rerank_model",
    "RerankClient": ".rerank_model",
    "RerankResponse": ".rerank_model",
    "RerankResult": ".rerank_model",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve model exports by provider family."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return public exports for interactive discovery."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .custom_chat_model import CustomChatModel, create_chat_model
    from .embedding_model import create_embedding_model
    from .fake_models import AgentIntegrationModel
    from .fake_streaming_llm import FakeStreamingLLM
    from .reasoning import (
        build_reasoning_kwargs,
        register_reasoning_family,
        resolve_reasoning_family,
        supported_reasoning_efforts,
    )
    from .rerank_model import (
        RerankClient,
        RerankResponse,
        RerankResult,
        create_async_rerank_model,
        create_rerank_model,
    )

-------

packages/sta_agent_engine/src/sta_agent_engine/models/custom_chat_model.py
----
import contextlib
import logging
import os
import ssl
import warnings
from collections.abc import Mapping
from typing import Any, TypedDict, cast

import certifi
import httpx
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.block_translators import get_translator, register_translator
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.chat_models import global_ssl_context as _mistral_global_ssl_context
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import base as openai_base
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

from sta_agent_core.config import BaseProviderSettings, ProviderFactory
from sta_agent_core.types import ProviderType

from ..utils.signature_utils import expose_merged_signature
from .capabilities import is_multimodal
from .reasoning import build_reasoning_kwargs, merge_reasoning_config


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSL audit helper — diagnostic hook for trust-store issues
# ---------------------------------------------------------------------------
# Silent by default. Enable with STA_SSL_AUDIT=1 in the deployment env.
#
# When enabled, each call to _create_mistral_model / _create_openai_model
# emits a "ssl-audit provider=..." log line containing:
#   - cdc_module / cdc_qualname: who owns ssl.create_default_context right
#     now (vanilla 'ssl', 'truststore._api', 'pip_system_certs.*', ...).
#     Tells you whether any trust-store patch is actually active in the
#     worker process at call time.
#   - SSL_CERT_FILE / SSL_CERT_DIR: whether the runtime env vars are set.
#   - certifi.where(): which bundle certifi would hand out by default.
#   - ca_count: number of CAs loaded into a fresh default context. Compare
#     across providers and across failing/working deploys. ~150 = healthy
#     OS store; 0 = trust store is not being read; ~1 = only corp CA (bad).
#
# If SSL ever breaks again (cert rotation, Wolfi upgrade, LangGraph Platform
# runtime change), grep 'ssl-audit' in production logs for a one-shot
# diagnosis instead of multi-session debugging.
# ---------------------------------------------------------------------------
_SSL_AUDIT_ENABLED = os.environ.get("STA_SSL_AUDIT") == "1"
_ssl_audit_log = logging.getLogger("ssl-audit")


def _describe_ctx_ca_source(ctx: ssl.SSLContext) -> str:
    """Return a short string describing where ``ctx`` gets its CAs from.

    On truststore-backed contexts (truststore.SSLContext or
    pip._vendor.truststore.SSLContext), ``get_ca_certs`` and ``cert_store_stats``
    both raise ``NotImplementedError`` by design — trust validation is delegated
    to the OS. In that case we report ``os-delegated``. Otherwise we try to get
    a numeric count and fall back to ``unknown`` if both probes fail.
    """
    ctx_cls = type(ctx)
    ctx_mod = getattr(ctx_cls, "__module__", "") or ""
    if "truststore" in ctx_mod:
        return f"os-delegated({ctx_mod})"
    try:
        stats = ctx.cert_store_stats()
        return f"cert_store_stats={stats}"
    except NotImplementedError:
        pass
    except Exception as e:  # noqa: BLE001
        return f"cert_store_stats-error={e!r}"
    try:
        return f"get_ca_certs_len={len(ctx.get_ca_certs())}"
    except Exception as e:  # noqa: BLE001
        return f"unknown({e!r})"


def _log_ssl_state(provider_label: str) -> None:
    """Emit one per-call snapshot of the process's SSL trust state.

    No-op unless STA_SSL_AUDIT=1 is set in the environment. Never raises —
    diagnostic hooks must not interfere with the caller's control flow.

    The goal of this helper is to answer, at every call site, two questions
    that are otherwise impossible to tell apart in production:

    1. Is a trust-store delegation shim active right now (truststore,
       pip-system-certs, etc.), or is it vanilla ``ssl``?
    2. What CA bundle would a *fresh* default context actually see at this
       exact moment — same as at import time, or has something changed?
    """
    if not _SSL_AUDIT_ENABLED:
        return
    try:
        cdc = ssl.create_default_context
        ctx = cdc()
        _ssl_audit_log.warning(
            "ssl-audit provider=%s cdc_module=%r cdc_qualname=%r ctx_type=%s SSL_CERT_FILE=%r SSL_CERT_DIR=%r certifi=%r ca_source=%s",
            provider_label,
            getattr(cdc, "__module__", "?"),
            getattr(cdc, "__qualname__", "?"),
            type(ctx).__module__ + "." + type(ctx).__qualname__,
            os.environ.get("SSL_CERT_FILE"),
            os.environ.get("SSL_CERT_DIR"),
            certifi.where(),
            _describe_ctx_ca_source(ctx),
        )
    except Exception as e:  # noqa: BLE001 — logging must never break a request
        _ssl_audit_log.warning("ssl-audit provider=%s inspect failed: %r", provider_label, e)


def _log_httpx_client_ssl_state(provider_label: str, async_client: Any) -> None:
    """Introspect an already-constructed httpx AsyncClient's SSL context.

    httpx freezes its ssl context inside AsyncClient.__init__, so the process-wide
    state logged by _log_ssl_state() is necessary but not sufficient — we also want
    to see what the client actually captured. httpx's internal attribute names
    change between versions, so this is best-effort and silently skips on mismatch.
    """
    if not _SSL_AUDIT_ENABLED:
        return
    try:
        transport = getattr(async_client, "_transport", None)
        pool = getattr(transport, "_pool", None) if transport is not None else None
        sc = getattr(pool, "_ssl_context", None) or getattr(transport, "_ssl_context", None) or None
        if sc is None:
            _ssl_audit_log.warning(
                "ssl-audit provider=%s:client introspection unsupported for this httpx version",
                provider_label,
            )
            return
        _ssl_audit_log.warning(
            "ssl-audit provider=%s:client ctx_type=%s ca_source=%s",
            provider_label,
            type(sc).__module__ + "." + type(sc).__qualname__,
            _describe_ctx_ca_source(sc),
        )
    except Exception as e:  # noqa: BLE001
        _ssl_audit_log.warning("ssl-audit provider=%s:client inspect failed: %r", provider_label, e)


# Reasoning/thinking keywords across providers (DeepSeek, OpenAI o-series, etc.)
_REASONING_KEYWORDS = (
    "reasoning_content",
    "reasoning",
    "reasoning_block",
    "thinking_content",
    "think",
    "thinking",
    "thinking_block",
    "think_content",
)


def _extract_reasoning(source: Mapping[str, Any], target: dict[str, Any]) -> None:
    """Extract reasoning content from a response dict into additional_kwargs.

    Checks multiple provider-specific keys and normalises them under a single
    ``reasoning_content`` key so downstream consumers have a stable interface.
    """
    for keyword in _REASONING_KEYWORDS:
        value = source.get(keyword)
        if value is not None:
            target["reasoning_content"] = value
            return


def _coerce_reasoning_to_text(value: Any) -> str:
    """Coerce a captured reasoning value of any shape into display text.

    A ``content_blocks`` reasoning block carries a string ``reasoning`` field,
    but a provider may surface reasoning as a structured object rather than a
    plain string. Rules:

    - ``str`` → returned unchanged (the common gpt-oss / vLLM case).
    - a reasoning-shaped ``dict`` (``{"type": "reasoning", "content": ...}``) →
      its ``content`` (string-coerced).
    - anything else → ``str(value)``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("type") == "reasoning" and value.get("content") is not None:
        content = value["content"]
        return content if isinstance(content, str) else str(content)
    return str(value)


# ---------------------------------------------------------------------------
# Streaming patch — _convert_delta_to_message_chunk
# ---------------------------------------------------------------------------


def _custom_convert_delta_to_message_chunk(_dict: Mapping[str, Any], default_class: type[BaseMessageChunk]) -> BaseMessageChunk:
    """
    Custom version of _convert_delta_to_message_chunk with reasoning content support.

    This function extends the original langchain_openai conversion to handle reasoning
    tokens from models that support chain-of-thought reasoning (e.g., DeepSeek, o1).

    Args:
        _dict: Dictionary containing delta message information
        default_class: Default message chunk class to use

    Returns:
        BaseMessageChunk with reasoning content in additional_kwargs if present
    """
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}

    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call

    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        with contextlib.suppress(KeyError):
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]

    _extract_reasoning(_dict, additional_kwargs)

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        )
    if role in ("system", "developer") or default_class == SystemMessageChunk:
        additional_kwargs = {"__openai_role__": "developer"} if role == "developer" else {}
        return SystemMessageChunk(content=content, id=id_, additional_kwargs=additional_kwargs)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"], id=id_)  # type: ignore[call-arg]
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    return default_class(content=content, id=id_)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Non-streaming patch — _convert_dict_to_message
# ---------------------------------------------------------------------------


def _custom_convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Custom version of _convert_dict_to_message with reasoning content support.

    The original langchain_openai implementation silently drops ``reasoning_content``
    (and similar provider-specific thinking fields) from the full API response dict.
    This patched version mirrors the original logic and additionally captures reasoning
    content into ``additional_kwargs["reasoning_content"]`` for observability.

    Args:
        _dict: Dictionary from the OpenAI API response ``choices[].message``.

    Returns:
        BaseMessage with reasoning content in additional_kwargs when present.
    """
    role = _dict.get("role")
    name = _dict.get("name")
    id_ = _dict.get("id")
    content_raw = _dict.get("content")
    content_safe = content_raw if content_raw is not None else ""

    if role == "user":
        return HumanMessage(content=content_safe, id=id_, name=name)

    if role == "assistant":
        content = content_safe
        additional_kwargs: dict = {}

        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)

        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(make_invalid_tool_call(raw_tool_call, str(e)))

        if audio := _dict.get("audio"):
            additional_kwargs["audio"] = audio

        _extract_reasoning(_dict, additional_kwargs)

        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )

    if role in ("system", "developer"):
        additional_kwargs = {"__openai_role__": role} if role == "developer" else {}
        return SystemMessage(
            content=content_safe,
            name=name,
            id=id_,
            additional_kwargs=additional_kwargs,
        )

    if role == "function":
        return FunctionMessage(content=content_safe, name=cast(str, _dict.get("name")), id=id_)

    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=content_safe,
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )

    return ChatMessage(content=content_safe, role=role or "assistant", id=id_)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Monkey patches — reasoning content support for langchain_openai
# ---------------------------------------------------------------------------
# These patches mirror upstream functions from langchain_openai.chat_models.base
# and add reasoning content extraction. They are fragile — when langchain-openai
# is upgraded, the upstream functions may change and our patches silently diverge.
#
# Drift detection: tests/test_ai_engine/models/test_custom_chat_model.py contains
# hash-based tests that fail when the upstream source changes, prompting a review.
#
# Patched against: langchain-openai==1.1.9
# Upstream functions: _convert_delta_to_message_chunk, _convert_dict_to_message
# ---------------------------------------------------------------------------

# Save originals before patching (used by drift-detection tests)
_original_convert_delta_to_message_chunk = openai_base._convert_delta_to_message_chunk
_original_convert_dict_to_message = openai_base._convert_dict_to_message

openai_base._convert_delta_to_message_chunk = _custom_convert_delta_to_message_chunk
openai_base._convert_dict_to_message = _custom_convert_dict_to_message
logger.info("Applied custom reasoning parsing logic to langchain_openai (streaming + non-streaming)")


# ---------------------------------------------------------------------------
# content_blocks reasoning — reasoning-aware OpenAI block translator
# ---------------------------------------------------------------------------
# langchain-core derives ``AIMessage.content_blocks`` via a per-provider
# translator selected by ``response_metadata["model_provider"]``. langchain-openai
# stamps ``"openai"`` on every chunk, so core routes to its OpenAI chat-completions
# translator — which builds blocks only from ``.content`` + tool calls and ignores
# ``additional_kwargs`` entirely. Core's best-effort reasoning fallback lives
# *below* that provider branch and is never reached, so the reasoning captured by
# the converter patches above (``additional_kwargs["reasoning_content"]``) is
# dropped from ``content_blocks`` — a reasoning-only stream chunk yields ``[]``.
#
# We wrap the registered OpenAI translator to re-add a reasoning block from
# ``additional_kwargs`` when one is present and the underlying translator did not
# already emit it. Purely additive: text / tool-call / multimodal blocks are
# untouched, and a responses-API message that already carries a reasoning block
# is left alone. Note: ``register_translator`` mutates a process-wide registry, so
# this affects ``content_blocks`` for every ``model_provider="openai"`` message in
# the host process — the same global-patch posture as the converter patches above.
# ---------------------------------------------------------------------------


def _reasoning_block_from_message(message: BaseMessage) -> dict[str, Any] | None:
    """Build a ``{"type": "reasoning", ...}`` block from ``additional_kwargs`` if present."""
    raw = getattr(message, "additional_kwargs", {}).get("reasoning_content")
    if raw is None:
        return None
    text = _coerce_reasoning_to_text(raw)
    return {"type": "reasoning", "reasoning": text} if text else None


def _wrap_translator_with_reasoning(translate: Any) -> Any:
    """Wrap a core block translator so it prepends a reasoning block when missing."""

    def _translate(message: Any) -> Any:
        blocks = translate(message)
        already_has_reasoning = any(isinstance(block, dict) and block.get("type") == "reasoning" for block in blocks)
        if not already_has_reasoning and (reasoning_block := _reasoning_block_from_message(message)) is not None:
            blocks.insert(0, reasoning_block)
        return blocks

    return _translate


_original_openai_translator = get_translator("openai")
if _original_openai_translator is not None:
    register_translator(
        "openai",
        _wrap_translator_with_reasoning(_original_openai_translator["translate_content"]),
        _wrap_translator_with_reasoning(_original_openai_translator["translate_content_chunk"]),
    )
    logger.info("Registered reasoning-aware content_blocks translator for provider 'openai'")


# ---------------------------------------------------------------------------
# L1 stall fence — granular httpx timeout for create_chat_model
# ---------------------------------------------------------------------------
# Phase 3 (#47) of twin_router robustness. Every chat model created here
# inherits a per-axis httpx.Timeout so a hung upstream is killed at 30s of
# byte-level silence rather than 600s of wall clock. Every axis is
# env-tunable (CHAT_MODEL_HTTPX_{CONNECT,READ,WRITE,POOL}_TIMEOUT_S) and
# per-call overridable via ``timeouts=``; the read axis carries the real
# stall budget (30s) while connect / write / pool sit at a modest 10s.
# ---------------------------------------------------------------------------
DEFAULT_HTTPX_CONNECT_TIMEOUT_S = 10.0
DEFAULT_HTTPX_WRITE_TIMEOUT_S = 30.0
DEFAULT_HTTPX_POOL_TIMEOUT_S = 10.0
DEFAULT_HTTPX_READ_TIMEOUT_S = 50.0

_HTTPX_TIMEOUT_AXES: tuple[str, ...] = ("connect", "read", "write", "pool")
_HTTPX_DEFAULTS: dict[str, float] = {
    "connect": DEFAULT_HTTPX_CONNECT_TIMEOUT_S,
    "read": DEFAULT_HTTPX_READ_TIMEOUT_S,
    "write": DEFAULT_HTTPX_WRITE_TIMEOUT_S,
    "pool": DEFAULT_HTTPX_POOL_TIMEOUT_S,
}
_HTTPX_ENV_VARS: dict[str, str] = {
    "connect": "CHAT_MODEL_HTTPX_CONNECT_TIMEOUT_S",
    "read": "CHAT_MODEL_HTTPX_READ_TIMEOUT_S",
    "write": "CHAT_MODEL_HTTPX_WRITE_TIMEOUT_S",
    "pool": "CHAT_MODEL_HTTPX_POOL_TIMEOUT_S",
}
_TIER_ONLY_CONFIG_KEYS = frozenset({"big_model", "small_model", "thinking_model", "multimodal_model", "tier"})


def _warn_legacy_read_timeout_s() -> None:
    """Emit a one-line UserWarning steering callers to the `timeouts` API."""
    warnings.warn(
        "`read_timeout_s` is removed — use `timeouts={'read': N}` instead.",
        UserWarning,
        stacklevel=3,
    )


class TimeoutOverrides(TypedDict, total=False):
    """Per-axis httpx timeout overrides for chat-model factories.

    All keys optional; omitted axes fall back to env / module defaults.
    """

    connect: float
    read: float
    write: float
    pool: float


def _resolve_httpx_timeout(
    timeouts: TimeoutOverrides | httpx.Timeout | None = None,
) -> httpx.Timeout:
    """Return the granular httpx timeout for chat-model clients.

    Precedence per axis (highest first):
        1. ``timeouts`` argument — explicit per-axis override (dict or
           ``httpx.Timeout``). The single per-call override surface.
        2. ``CHAT_MODEL_HTTPX_{AXIS}_TIMEOUT_S`` env var — operator
           rollback knob, applies process-wide.
        3. ``DEFAULT_HTTPX_{AXIS}_TIMEOUT_S`` — module default.
    """
    overrides: dict[str, float] = {}
    if isinstance(timeouts, httpx.Timeout):
        for axis in _HTTPX_TIMEOUT_AXES:
            value = getattr(timeouts, axis, None)
            if value is not None:
                overrides[axis] = float(value)
    elif timeouts is not None:
        overrides = {axis: float(timeouts[axis]) for axis in _HTTPX_TIMEOUT_AXES if axis in timeouts}

    resolved: dict[str, float] = {}
    for axis in _HTTPX_TIMEOUT_AXES:
        if axis in overrides:
            resolved[axis] = overrides[axis]
            continue
        raw = os.environ.get(_HTTPX_ENV_VARS[axis])
        resolved[axis] = float(raw) if raw else _HTTPX_DEFAULTS[axis]

    return httpx.Timeout(**resolved)


def _env_prefix_for_provider(provider: ProviderType | str | None) -> str:
    """Return the env-var prefix that ``ProviderFactory`` reads for ``provider``.

    Used to build human-readable error messages naming the exact env vars
    a caller should set. ``ProviderType.CUSTOM`` and ``None`` map to the
    empty prefix (bare ``API_KEY`` / ``BASE_URL`` / ``MODEL``).
    """
    if isinstance(provider, ProviderType):
        return "" if provider == ProviderType.CUSTOM else f"{provider.value.upper()}_"
    if isinstance(provider, str) and provider:
        return f"{provider.upper()}_"
    return ""


def _warn_silent_openai_fallback(provider: ProviderType | str | None, missing: list[str]) -> None:
    """Emit a DeprecationWarning when the OpenAI dispatch is about to fall back
    to ``OPENAI_API_KEY`` + ``api.openai.com`` from the process environment.

    The silent fallback is a long-standing footgun: callers who pass a provider
    name but forget to set its env vars get a 404 from OpenAI's default endpoint
    for whatever model name was inferred. Warn now, hard-fail in 0.10.0.

    When the caller is not already using ``provider="openai"``, append a hint
    pointing at the default-registered OpenAI provider — most accidental
    fall-throughs are actually OpenAI usage that should be made explicit.
    """
    prefix = _env_prefix_for_provider(provider)
    env_vars = [f"{prefix}{field.upper()}" for field in missing]
    hint = ""
    is_openai_provider = isinstance(provider, str) and provider.lower() == "openai"
    if not is_openai_provider:
        hint = " If you are actually using OpenAI, switch to create_chat_model('openai') — it reads OPENAI_API_KEY natively."
    warnings.warn(
        f"Provider {provider!r} has no {missing} resolvable from env vars or "
        f"kwargs. ChatOpenAI will silently fall back to OPENAI_API_KEY + "
        f"api.openai.com from the process environment. Set {env_vars} or pass "
        f"{missing} as kwargs to create_chat_model(). The silent fallback is "
        f"deprecated and will raise in 0.10.0.{hint}",
        DeprecationWarning,
        stacklevel=3,
    )


def _is_mistral_model(model_name: str | None, provider: ProviderType | str | None) -> bool:
    """Check if the configuration indicates a Mistral model.

    Args:
        model_name: The model name from configuration.
        provider: The provider type. Either a ``ProviderType`` member or
            an arbitrary string. When a string carries ``"mistral"`` as a
            substring (e.g. ``"mistral_eu"``, ``"mistral_corp"``) the
            dispatch routes to ``ChatMistralAI`` even if the model name
            doesn't carry the brand.

    Returns:
        True if this is a Mistral model/provider.
    """
    provider_name_hits = isinstance(provider, str) and "mistral" in provider.lower()
    return (
        provider == ProviderType.MISTRAL
        or provider_name_hits
        or bool(model_name and "mistral" in model_name.lower())
        or bool(model_name and "devstral" in model_name.lower())
        or bool(model_name and "magistral" in model_name.lower())
    )


def _create_mistral_model(config: dict[str, Any]) -> ChatMistralAI:
    """Create a ChatMistralAI instance with the L1 stall fence injected.

    ChatMistralAI handles Mistral's message format correctly and doesn't include
    unsupported fields like 'name' in assistant messages.

    L1 (Phase 3 / #47): pre-built httpx clients carry the granular
    ``_resolve_httpx_timeout()`` and the bearer token so that the SDK's
    relative-path requests (``self.async_client.post('/chat/completions')``)
    resolve correctly. ``max_retries=0`` is forced because Mistral SDK
    retries restart the httpx timeout and don't honour Retry-After — leaving
    the SDK default would inflate our 30s stall budget by 6×.

    The SDK's own SSL context (``langchain_mistralai.chat_models.global_ssl_context``)
    is reused so truststore delegation on Wolfi stays consistent with the
    behaviour the SDK would have produced if we hadn't injected a client.
    """
    _log_ssl_state("mistral")

    api_key = config.pop("api_key", None) or os.environ.get("MISTRAL_API_KEY")
    base_url = config.pop("base_url", None) or os.environ.get("MISTRAL_BASE_URL") or "https://api.mistral.ai/v1"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # Per-call timeout override — pass ``timeouts={"connect": 10, "read": 90, ...}``
    # or an ``httpx.Timeout`` instance. The legacy ``read_timeout_s`` kwarg
    # is removed; map any stray callers to the new API.
    if "read_timeout_s" in config:
        _warn_legacy_read_timeout_s()
        config.pop("read_timeout_s", None)
    timeouts_override = config.pop("timeouts", None)
    timeout = _resolve_httpx_timeout(timeouts=timeouts_override)

    async_client = config.pop("async_client", None) or httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=timeout,
        verify=_mistral_global_ssl_context,
    )
    sync_client = config.pop("client", None) or httpx.Client(
        base_url=base_url,
        headers=headers,
        timeout=timeout,
        verify=_mistral_global_ssl_context,
    )

    _log_httpx_client_ssl_state("mistral", async_client)

    mistral_config: dict[str, Any] = {
        "api_key": api_key,
        "endpoint": base_url,  # ChatMistralAI Field alias for base_url
        "model": config.pop("model", None),
        "temperature": config.pop("temperature", None),
        "max_tokens": config.pop("max_tokens", None),
        "top_p": config.pop("top_p", None),
        # Force max_retries=0 — Mistral SDK retries restart the httpx timeout
        # and don't honour Retry-After. Let with_retry() / agent middleware
        # own retry semantics instead.
        "max_retries": 0,
    }

    if model_kwargs := config.pop("model_kwargs", None):
        mistral_config["model_kwargs"] = model_kwargs

    # Drop the SDK's own ``timeout`` from the caller's config — the granular
    # httpx.Timeout on the injected client is the source of truth now.
    config.pop("timeout", None)
    config.pop("max_retries", None)

    # Filter out None values from the explicit fields we set above.
    mistral_config = {k: v for k, v in mistral_config.items() if v is not None}

    # Anything else the caller passed (custom Mistral kwargs) flows through.
    mistral_config = {**config, **mistral_config}

    logger.info(f"Creating ChatMistralAI with model={mistral_config.get('model')}")
    return ChatMistralAI(client=sync_client, async_client=async_client, **mistral_config)


def _create_openai_model(config: dict[str, Any]) -> ChatOpenAI:
    """Create a ChatOpenAI instance with the L1 stall fence injected.

    L1 (Phase 3 / #47): pre-built httpx clients carry the granular
    ``_resolve_httpx_timeout()`` so the openai SDK's transport inherits a
    30s read fence instead of the previous 600s blanket. ``max_retries=1``
    keeps one Retry-After-aware retry for 429 storms — OpenAI's SDK retries
    are well-behaved here, unlike Mistral's.
    """
    _log_ssl_state("openai")

    # Per-call timeout override — pass ``timeouts={"connect": 10, "read": 90, ...}``
    # or an ``httpx.Timeout`` instance. Legacy ``read_timeout_s`` is removed.
    if "read_timeout_s" in config:
        _warn_legacy_read_timeout_s()
        config.pop("read_timeout_s", None)
    timeouts_override = config.pop("timeouts", None)
    timeout = _resolve_httpx_timeout(timeouts=timeouts_override)

    async_client = config.pop("http_async_client", None) or DefaultAsyncHttpxClient(timeout=timeout)
    sync_client = config.pop("http_client", None) or DefaultHttpxClient(timeout=timeout)

    _log_httpx_client_ssl_state("openai", async_client)

    config.setdefault("max_retries", 1)

    return ChatOpenAI(
        http_async_client=async_client,
        http_client=sync_client,
        profile={"max_input_tokens": 128_000},
        **config,
    )


def _resolve_multimodal_model(settings: BaseProviderSettings, provider: str | ProviderType | None) -> str:
    """Resolve the model to use for a multimodal (vision) request.

    Resolution order (hard-fails rather than silently returning a text model):

        1. ``settings.multimodal_model`` if set — authoritative, used verbatim.
           Not re-checked against ``is_multimodal`` so an operator can declare a
           vision model the static capability list doesn't yet recognize.
        2. ``settings.model`` if ``is_multimodal(settings.model)`` — the default
           already points at a vision-capable model.
        3. Otherwise raise ``ValueError`` — ``multimodal=True`` was requested but
           the deployment has no multimodal model configured. Failing loud at
           construction beats handing back a text model that silently strips
           images (under the guard middleware) or 400s upstream.

    Args:
        settings: Resolved provider settings.
        provider: Original provider argument, for the error message.

    Returns:
        The model identifier to use for the vision request.

    Raises:
        ValueError: If no multimodal model can be resolved.
    """
    if settings.multimodal_model:
        return settings.multimodal_model
    if is_multimodal(settings.model):
        return settings.model
    env_prefix = type(settings).model_config.get("env_prefix") or ""
    provider_label = provider.value if isinstance(provider, ProviderType) else (provider or "default")
    raise ValueError(
        f"multimodal=True but no multimodal model is configured for provider {provider_label!r}: "
        f"set {env_prefix}MULTIMODAL_MODEL, or point {env_prefix}MODEL at a vision-capable model."
    )


# Tier -> the capacity-tier fields whose explicit presence means the resolved model
# did NOT fall through to the base ``model`` default. Mirrors the cascade in
# ``BaseProviderSettings.get_model``.
_TIER_MODEL_FIELDS: dict[str, tuple[str, ...]] = {
    "default": (),
    "big": ("big_model",),
    "small": ("small_model",),
    "thinking": ("thinking_model", "big_model"),
}


def _model_is_implicit_default(settings: BaseProviderSettings, tier: str | None, resolved_model: str) -> bool:
    """True when ``resolved_model`` came from the class default, not an explicit source.

    "Explicit" means a ``*_MODEL`` / ``*_<TIER>_MODEL`` env var, a registered
    default, an init value, or a ``model=`` kwarg. Detection uses the value-vs-
    default comparison rather than ``model_fields_set`` because the
    ``empty_str_to_none`` validator coerces ``*_MODEL=""`` to ``None`` while still
    marking the field "set" — so ``model_fields_set`` reports a falsely-explicit
    model. Comparing the resolved value to the field default is robust to that.

    Known benign false-positive: a consumer who explicitly sets the model to the
    exact same string as the built-in default is still treated as implicit.
    """
    tier_name = "default" if tier is None else str(tier).lower().strip()
    fields_set = settings.model_fields_set
    # An explicitly-set tier field supplied the model -> the base default was not used.
    for field in _TIER_MODEL_FIELDS.get(tier_name, ()):
        if field in fields_set and getattr(settings, field, None):
            return False
    model_field_default = type(settings).model_fields["model"].default
    return resolved_model == model_field_default


def create_chat_model(
    provider: str | ProviderType | None = None,
    *,
    tier: str = "default",
    multimodal: bool = False,
    **kwargs: Any,
) -> ChatOpenAI | ChatMistralAI:
    """
    Factory function to create a chat model with provider-specific configuration.

    Automatically selects the appropriate client based on the provider:
    - ChatMistralAI for Mistral models (handles Mistral's stricter message format)
    - ChatOpenAI for OpenAI-compatible APIs

    Built-in providers:
    - LLMaaS: Set LLM_PROVIDER=llmaas, configure with LLMAAS_* env vars
    - LLMaaS Dev: Set LLM_PROVIDER=llmaas_dev, configure with LLMAAS_DEV_* env vars
    - Mistral: Set LLM_PROVIDER=mistral, configure with MISTRAL_* env vars
    - Custom: Set LLM_PROVIDER=custom, configure with {NO_PREFIX}* env vars

    Dynamic providers:
        Any other string is accepted. The env prefix is derived as
        ``f"{NAME.upper()}_"`` and read via Pydantic Settings:

        - ``{NAME}_API_KEY``
        - ``{NAME}_BASE_URL``
        - ``{NAME}_MODEL``
        - ``{NAME}_BIG_MODEL`` / ``{NAME}_SMALL_MODEL`` / ``{NAME}_THINKING_MODEL``
        - ``{NAME}_MULTIMODAL_MODEL``
        - ``{NAME}_TEMPERATURE`` / ``{NAME}_TOP_P`` / ``{NAME}_MAX_TOKENS``

        Use ``ProviderFactory.register(name, defaults=..., env_prefix=...)``
        for non-env defaults or a custom env prefix.

    Model tiers:
        ``tier`` selects a provider model slot when no explicit ``model=`` kwarg
        is passed. Supported tiers are ``default``, ``big``, ``small``, and
        ``thinking``. ``thinking`` cascades to ``big`` and then ``model``.

    Multimodal:
        ``multimodal=True`` requests a vision-capable model (capability axis,
        not a capacity tier). It resolves ``{NAME}_MULTIMODAL_MODEL`` if set,
        else ``{NAME}_MODEL`` when that is a recognized multimodal model, else
        raises ``ValueError`` — it never silently returns a text-only model.
        ``multimodal=True`` takes precedence over ``tier`` and is ignored when
        an explicit ``model=`` kwarg is passed.

    Mistral dispatch:
        Routes to ``ChatMistralAI`` when any of these hold:

        - ``provider == ProviderType.MISTRAL``
        - The provider name (string) contains ``"mistral"``
        - The model name contains ``"mistral"`` / ``"devstral"`` / ``"magistral"``

    Credential resolution:
        ``api_key`` and ``base_url`` should resolve via env vars (under the
        provider's prefix) or kwargs. When the OpenAI dispatch path is taken
        and either is missing, ``ChatOpenAI`` silently falls back to
        ``OPENAI_API_KEY`` + ``api.openai.com`` from the process env. This
        emits a ``DeprecationWarning`` and will raise ``ValueError`` in 0.10.0.

    Reasoning control:
        ``reasoning_effort="off" | "low" | "medium" | "high"`` is
        translated per model family into the kwargs the model actually honors
        (Mistral's ``reasoning_effort``, Nemotron/Qwen ``chat_template_kwargs``
        — see ``models/reasoning.py``). Omitted -> nothing is injected (the
        model keeps its server-side default). Unsupported efforts warn and
        never silently substitute; explicitly-passed ``extra_body`` /
        ``model_kwargs`` win over translated values on conflicting keys.
        ``reasoning_family="nemotron-super"`` pins the family when a gateway
        alias hides the model name. Prefer one effort per agent/task/thread —
        flipping ``chat_template_kwargs`` per turn busts server-side prefix
        caches. Consumer guide: ``docs/consuming/reasoning.md``.

    Args:
        provider: ``ProviderType`` member, built-in provider string,
            or any arbitrary dynamic-provider name.
        tier: Capacity tier to resolve from provider settings when ``model``
            is not passed explicitly.
        multimodal: When True, resolve a vision-capable model (see Multimodal
            above). Takes precedence over ``tier``; ignored if ``model=`` is set.
        **kwargs: Additional configuration to override environment variables.

    Returns:
        BaseChatModel instance (ChatMistralAI or ChatOpenAI) configured for the specified provider.

    Examples:
        # Built-in provider via environment variables
        export LLM_PROVIDER=mistral
        export MISTRAL_API_KEY=your_api_key
        export MISTRAL_MODEL=mistral-small-2603
        model = create_chat_model()

        # Dynamic provider — no code change required
        export ACME_API_KEY=your_key
        export ACME_BASE_URL=https://api.acme.test/v1
        export ACME_MODEL=acme-large
        model = create_chat_model("acme")

        # Explicit Mistral provider
        model = create_chat_model("mistral")

        # Mistral-flavored dynamic provider (auto-routes to ChatMistralAI)
        model = create_chat_model("mistral_eu", api_key="...", base_url="https://eu.mistral.test", model="mistral-large")
    """
    # Lowercase string providers so dispatch matches the factory's normalization,
    # but keep the original string when it doesn't map to a known enum member.
    if isinstance(provider, str):
        try:
            provider = ProviderType(provider.lower())
        except ValueError:
            provider = provider.lower()

    provider_settings = ProviderFactory.get_provider_settings(provider)
    settings = provider_settings.model_dump()

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if "model" not in kwargs:
        if multimodal:
            settings["model"] = _resolve_multimodal_model(provider_settings, provider)
        else:
            settings["model"] = provider_settings.get_model(tier)

    # Extract credentials from context if available (BYOK mode).
    # ``pop`` (not lookup): these are aliases for ``api_key`` / ``base_url``, not
    # client constructor params. Leaving them in ``kwargs`` would merge them into
    # ``config`` below and forward them to ChatOpenAI/ChatMistralAI, which route
    # unknown kwargs into ``model_kwargs`` — leaking the raw key into the request
    # body sent over the wire and 400-ing strict OpenAI-compatible servers.
    if kwargs:
        if (provider_api_key := kwargs.pop("provider_api_key", None)) is not None:
            settings["api_key"] = provider_api_key
        if (provider_base_url := kwargs.pop("provider_base_url", None)) is not None:
            settings["base_url"] = provider_base_url

    config = {k: v for k, v in {**settings, **kwargs}.items() if v is not None}
    for key in _TIER_ONLY_CONFIG_KEYS:
        config.pop(key, None)

    # Every provider carries a default model slug, so a model always resolves and
    # construction never fails for lack of one. But relying on that *implicit*
    # default is deprecated: a consumer who never configured a model is almost
    # certainly pointing the default slug at an endpoint that does not serve it
    # (a wrong default is dangerous for third-party providers). Warn when the
    # resolved model came from the class default rather than an explicit source —
    # a *_MODEL / *_<TIER>_MODEL env var, a registered default, or model=... — and
    # keep working for now. This becomes a hard ValueError in 0.11.0. The
    # multimodal path is exempt: it already hard-fails when nothing is configured.
    if "model" not in kwargs and not multimodal and _model_is_implicit_default(provider_settings, tier, config.get("model", "")):
        prefix = type(provider_settings).model_config.get("env_prefix") or ""
        warnings.warn(
            f"create_chat_model({provider!r}) is using the built-in default model {config.get('model')!r}. "
            "Relying on an implicit default is deprecated and will raise in 0.11.0. Set "
            f"{prefix}MODEL (or {prefix}<TIER>_MODEL), register a default via "
            "ProviderFactory.register(name, defaults={'model': ...}), or pass model=... explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Determine if this is a Mistral model and use appropriate client.
    # Mistral's SDK has its own legitimate base_url fallback to api.mistral.ai —
    # do NOT warn here. The footgun only exists on the OpenAI dispatch path.
    model_name = config.get("model", "")

    # Reasoning-effort resolution (see models/reasoning.py): translate the
    # normalized effort into the per-family request kwargs. Explicit caller
    # kwargs win over translated values on leaf conflicts — raw extra_body /
    # model_kwargs are the escape hatch. ``reasoning_family`` pins the family
    # when a gateway alias hides the model name; popped unconditionally so it
    # never leaks into the client constructor.
    reasoning_family = config.pop("reasoning_family", None)
    if (reasoning_effort := config.pop("reasoning_effort", None)) is not None:
        provider_str = str(provider) if provider is not None else None
        reasoning_config = build_reasoning_kwargs(model_name, reasoning_effort, provider=provider_str, family=reasoning_family)
        config, reasoning_overrides = merge_reasoning_config(reasoning_config, config)
        if reasoning_overrides:
            warnings.warn(
                f"reasoning_effort={reasoning_effort!r}: explicitly-passed kwargs override the translated reasoning keys "
                f"{', '.join(sorted(reasoning_overrides))} — the explicit values win.",
                UserWarning,
                stacklevel=2,
            )
    elif reasoning_family is not None:
        warnings.warn(
            f"reasoning_family={reasoning_family!r} was given without reasoning_effort — it selects the translation table "
            "but injects nothing on its own, so it was ignored. Pass reasoning_effort=... alongside it.",
            UserWarning,
            stacklevel=2,
        )
    if _is_mistral_model(model_name, provider):
        logger.info(f"Detected Mistral model/provider, using ChatMistralAI for model={model_name}")
        return _create_mistral_model(config)

    # OpenAI dispatch — if api_key / base_url didn't resolve, ChatOpenAI will
    # silently pick up OPENAI_API_KEY + api.openai.com from the process env.
    # Restored to preserve main-branch behavior; the silent fallback is now
    # deprecated and will raise in 0.10.0.
    missing: list[str] = []
    if not config.get("api_key"):
        missing.append("api_key")
    if not config.get("base_url"):
        missing.append("base_url")
    if missing:
        _warn_silent_openai_fallback(provider, missing)

    return _create_openai_model(config)


# Expose merged signature for better IDE support and introspection
expose_merged_signature(create_chat_model, ChatOpenAI)

# Backward compatibility alias - can be removed after updating all imports
CustomChatModel = create_chat_model

-------

packages/sta_agent_engine/src/sta_agent_engine/models/reasoning.py
----
"""Declarative reasoning-effort configuration for chat models.

Model families expose incompatible knobs for controlling reasoning/thinking:
Mistral takes a top-level ``reasoning_effort`` string, Nemotron wants booleans
nested under ``extra_body.chat_template_kwargs``, Qwen adds a token budget.
This module maps one normalized effort vocabulary (``off`` / ``low`` /
``medium`` / ``high``) onto the literal request kwargs each family honors, so
callers write ``reasoning_effort="high"`` once and the right wire format is
emitted for whatever model resolves. Efforts are open strings — a family
registered via :func:`register_reasoning_family` may define additional rungs.

The registry is a plain data table — family name -> effort -> literal kwargs.
Adding a model family is one :func:`register_reasoning_family` call (or one
table entry); no dispatch code changes. The error posture is warn-not-raise:
requesting reasoning must never break model construction. Unknown families
degrade to forwarding ``reasoning_effort`` natively (the OpenAI form), which
genuine OpenAI APIs honor and OpenAI-compatible gateways typically ignore.

Guidance: pick an effort per agent/task/thread, not per turn. For
template-flag families, flipping ``chat_template_kwargs`` between turns of one
conversation re-renders the server-side prompt template and defeats
vLLM/SGLang prefix caching.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any


# Warnings are attributed to the first stack frame OUTSIDE this package
# directory (PEP 678-era ``skip_file_prefixes``, Python 3.12+). A fixed
# ``stacklevel`` can't be right for every entry point (direct call vs. routed
# through create_chat_model adds frames), and misattribution has a second
# cost: Python's default once-per-location filter would collapse every
# consumer call site onto one internal library line, silencing all repeats.
_WARN_SKIP_PREFIXES = (str(Path(__file__).resolve().parent),)


__all__ = [
    "build_reasoning_kwargs",
    "register_reasoning_family",
    "resolve_reasoning_family",
    "supported_reasoning_efforts",
]


# Family spec keys:
#   "match":          model-name patterns. A flat tuple of strings is ONE AND-group:
#                     all substrings must appear. A tuple of tuples is OR-of-AND-groups:
#                     the family matches if ANY group has all its substrings present —
#                     e.g. (("nemotron-3", "ultra"), ("nemo-ultra",)) covers both slug
#                     dialects. Matching is case- AND separator-insensitive: both sides
#                     are normalized by stripping `-`, `_`, `.`, `:`, `/` and spaces, so
#                     ("qwen3",) matches "qwen3.6", "qwen3-6", "Qwen/Qwen3.6-32B", and
#                     "qwen3:32b" alike — provider slug conventions don't matter.
#   "provider_match": substrings matched against the provider name (any hit wins;
#                     same normalization)
#   "native_path":    nested key path for forwarding a raw effort string when the
#                     requested effort has no rung (None -> nothing is injected).
#                     Empty "rungs" + a native_path = pure passthrough family (no warning).
#   "rungs":          effort -> literal constructor kwargs to merge
_FAMILIES: dict[str, dict[str, Any]] = {
    # Nemotron-3-Ultra: `low` sends force_nonempty_content=False (explicitly off,
    # so a server-side template default can't force it); medium/high carry
    # force_nonempty_content=True (SGLang requires it when tool-calling with
    # thinking enabled; harmless on non-tool calls).
    "nemotron-ultra": {
        "match": ("nemotron-3", "ultra"),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "low": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True, "force_nonempty_content": False}}},
            "medium": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True, "force_nonempty_content": True}}},
            "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "force_nonempty_content": True}}},
        },
    },
    "nemotron-super": {
        "match": ("nemotron-3", "super"),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "low": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}},
            "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        },
    },
    # Qwen3.x: thinking on by default; the model card documents only the binary
    # chat_template_kwargs.enable_thinking switch (Qwen/Qwen3.6-27B). Graded
    # thinking budgets are a serving-stack feature (vLLM `thinking_token_budget`,
    # version-dependent) — deliberately NOT baked in; gateways that support them
    # can re-register this family with budget rungs (see docs/consuming/reasoning.md).
    # The card also documents chat_template_kwargs.preserve_thinking=True for
    # keeping reasoning traces across agent turns — orthogonal to effort, pass it
    # via explicit extra_body (it deep-merges alongside these rungs).
    "qwen3": {
        "match": ("qwen3",),
        "provider_match": (),
        "native_path": None,
        "rungs": {
            "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        },
    },
    # Mistral small/medium accept only none/high; ChatMistralAI has no native
    # reasoning_effort field, so the value rides in model_kwargs (flattened into
    # the request payload). Unsupported efforts pass through raw via native_path
    # so the Mistral API validates them itself. The match groups mirror
    # _is_mistral_model's dispatch criteria: every model that routes to
    # ChatMistralAI must use this wire dialect ("off" must become "none").
    "mistral": {
        "match": (("mistral",), ("devstral",), ("magistral",)),
        "provider_match": ("mistral",),
        "native_path": ("model_kwargs", "reasoning_effort"),
        "rungs": {
            "off": {"model_kwargs": {"reasoning_effort": "none"}},
            "high": {"model_kwargs": {"reasoning_effort": "high"}},
        },
    },
    # Real OpenAI reasoning models take reasoning_effort natively — pure silent
    # passthrough (the API validates values like "minimal"/"low"/"medium"/"high").
    "openai": {
        "match": ("gpt",),
        "provider_match": (),
        "native_path": ("reasoning_effort",),
        "rungs": {},
    },
}


_SLUG_SEPARATORS = str.maketrans("", "", "-_.:/ ")


def _normalize_slug(name: str) -> str:
    """Canonicalize a model/provider slug for matching.

    Lowercases and strips separator characters (``- _ . : /`` and spaces), so
    provider-specific slug conventions collapse to one form: ``qwen3.6``,
    ``qwen3-6``, ``Qwen/Qwen3.6-32B-Instruct``, and ``qwen3:32b`` all contain
    the normalized pattern ``qwen36``/``qwen3``.
    """
    return name.lower().translate(_SLUG_SEPARATORS)


def _nested_from_path(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """Build a nested dict placing ``value`` at the given key path."""
    out: Any = value
    for key in reversed(path):
        out = {key: out}
    return out


def _match_substrings(spec_value: Any) -> tuple[str, ...]:
    """Coerce a spec's match value to a tuple of normalized patterns.

    Accepts a lone string as a single pattern — ``("qwen3")`` (missing trailing
    comma) is a string in Python, and iterating it would silently degrade to
    character-wise matching.
    """
    if isinstance(spec_value, str):
        spec_value = (spec_value,)
    return tuple(_normalize_slug(s) for s in spec_value)


def _match_groups(spec_value: Any) -> tuple[tuple[str, ...], ...]:
    """Coerce a spec's ``match`` value to OR-groups of normalized AND-substrings.

    - A lone string -> one group with one substring.
    - A flat tuple of strings -> ONE group (all substrings must match — AND).
    - A tuple containing any nested tuple/list -> every element is its own
      group (string elements become 1-substring groups); the family matches if
      ANY group fully matches (OR of ANDs).
    """
    if isinstance(spec_value, str):
        return ((_normalize_slug(spec_value),),)
    items = tuple(spec_value)
    if any(isinstance(item, (tuple, list)) for item in items):
        return tuple(_match_substrings(item) for item in items)
    return (_match_substrings(items),) if items else ()


def resolve_reasoning_family(model: str, *, provider: str | None = None, family: str | None = None) -> str | None:
    """Resolve which reasoning family applies to a model.

    Precedence: explicit ``family`` > provider substring match > model-name
    substring match (all of a family's ``match`` substrings must appear).
    Matching is case- and separator-insensitive (see :func:`_normalize_slug`),
    so the same model resolves identically across provider slug conventions —
    ``qwen3.6``, ``qwen3-6``, and ``Qwen/Qwen3.6-32B`` are one family.
    An explicit family that is not registered warns and falls back to the
    match-based resolution, so a typo degrades instead of silently no-oping.

    Args:
        model: Resolved model name (as sent to the endpoint).
        provider: Optional provider name (e.g. ``"mistral"``, ``"llmaas"``).
        family: Optional explicit family pin — use when a gateway alias hides
            the real model name (``chat-default`` actually serving Nemotron).

    Returns:
        The family name, or ``None`` if nothing matches.
    """
    if family is not None:
        if family in _FAMILIES:
            return family
        warnings.warn(
            f"Unknown reasoning_family {family!r} (registered: {sorted(_FAMILIES)}); falling back to model-name matching.",
            UserWarning,
            stacklevel=2,
            skip_file_prefixes=_WARN_SKIP_PREFIXES,
        )
    provider_normalized = _normalize_slug(provider or "")
    model_normalized = _normalize_slug(model)
    if provider_normalized:
        for name, spec in _FAMILIES.items():
            if any(sub in provider_normalized for sub in _match_substrings(spec["provider_match"])):
                return name
    for name, spec in _FAMILIES.items():
        for group in _match_groups(spec["match"]):
            if group and all(sub in model_normalized for sub in group):
                return name
    return None


def supported_reasoning_efforts(model: str, *, provider: str | None = None, family: str | None = None) -> frozenset[str]:
    """Return the effort names a model's family defines rungs for.

    An empty set means either no family matched or the family is a pure
    native-passthrough (any value is forwarded unvalidated for the API to judge).
    """
    resolved = resolve_reasoning_family(model, provider=provider, family=family)
    if resolved is None:
        return frozenset()
    return frozenset(_FAMILIES[resolved]["rungs"])


def build_reasoning_kwargs(model: str, effort: str | None, *, provider: str | None = None, family: str | None = None) -> dict[str, Any]:
    """Translate a normalized reasoning effort into model-specific kwargs.

    The returned dict is ready to splat into the model constructor. Families
    that translate into ``extra_body`` or a native field also work per call::

        kw = build_reasoning_kwargs("nemotron-3-super-120b", "low")
        model.invoke(messages, **kw)        # or model.bind(**kw)

    Exception: families whose kwargs ride ``model_kwargs`` (mistral) are
    constructor-only — ``ChatMistralAI`` flattens ``model_kwargs`` into the
    payload only from the constructor field; a call-time kwarg is posted as a
    literal ``"model_kwargs"`` JSON key. Set the effort at construction
    (``create_chat_model(..., reasoning_effort=...)``) for those.

    Behavior:
        - ``effort=None`` (or blank/whitespace) returns ``{}`` (inject nothing —
          blank means "unset", e.g. an empty env-var default).
        - A supported effort returns a fresh copy of the family's rung kwargs.
        - An unsupported effort warns (listing the supported set); families with
          a native passthrough forward the raw value for the API to validate,
          others return ``{}`` (there is no field to receive the value).
        - No family matched: warns and forwards ``{"reasoning_effort": effort}``
          (the OpenAI-native form — gateways that don't know it ignore it).

    Prefer one effort per agent/task/thread over per-turn flips: changing
    ``chat_template_kwargs`` mid-conversation busts server-side prefix caches.
    """
    if effort is None:
        return {}
    effort_name = str(effort).lower().strip()
    if not effort_name:
        return {}
    resolved = resolve_reasoning_family(model, provider=provider, family=family)
    if resolved is None:
        warnings.warn(
            f"No reasoning family matches model {model!r}; forwarding reasoning_effort={effort_name!r} as a native top-level parameter. "
            "OpenAI-style APIs honor it; vLLM/LiteLLM-fronted gateways typically ignore it. "
            "Register the model with register_reasoning_family(...) or pin reasoning_family=... to translate it properly.",
            UserWarning,
            stacklevel=2,
            skip_file_prefixes=_WARN_SKIP_PREFIXES,
        )
        return {"reasoning_effort": effort_name}
    spec = _FAMILIES[resolved]
    rungs: dict[str, dict[str, Any]] = spec["rungs"]
    if effort_name in rungs:
        return deepcopy(rungs[effort_name])
    native_path: tuple[str, ...] | None = spec["native_path"]
    if native_path is not None:
        if rungs:  # known vocabulary exists and the request is outside it
            warnings.warn(
                f"reasoning_effort={effort_name!r} is not defined for family {resolved!r} (model {model!r}; supported: {sorted(rungs)}). "
                "Forwarding the raw value for the API to validate.",
                UserWarning,
                stacklevel=2,
                skip_file_prefixes=_WARN_SKIP_PREFIXES,
            )
        return _nested_from_path(native_path, effort_name)
    warnings.warn(
        f"reasoning_effort={effort_name!r} is not supported by family {resolved!r} (model {model!r}; supported: {sorted(rungs)}). "
        "Nothing was applied — the model keeps its server-side default.",
        UserWarning,
        stacklevel=2,
        skip_file_prefixes=_WARN_SKIP_PREFIXES,
    )
    return {}


def register_reasoning_family(
    name: str,
    rungs: dict[str, dict[str, Any]],
    *,
    match_substrings: str | tuple[str | tuple[str, ...], ...] = (),
    provider_substrings: str | tuple[str, ...] = (),
    native_path: tuple[str, ...] | None = None,
) -> None:
    """Register (or replace) a reasoning family.

    Args:
        name: Family name; re-registering an existing name (including a
            built-in) replaces it, letting consumers override library defaults.
        rungs: Effort name -> literal constructor kwargs to merge (e.g.
            ``{"high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}}``).
        match_substrings: Model-name patterns for auto-detection. A flat tuple
            of strings is ONE AND-group — all substrings must appear:
            ``("nemotron-3", "ultra")``. A tuple of tuples is OR-of-AND-groups —
            the family matches when ANY group fully matches:
            ``(("nemotron-3", "ultra"), ("nemo-ultra",))`` covers two slug
            dialects. Matching is case- and separator-insensitive (``- _ . : /``
            and spaces are stripped from both sides), so ``("qwen3",)`` covers
            ``qwen3.6`` / ``qwen3-6`` / ``Qwen/Qwen3.6-32B`` regardless of the
            provider's slug convention.
        provider_substrings: Provider-name substrings that force this family
            (any hit wins; same normalization).
        native_path: Optional nested key path for forwarding a raw effort value
            when the requested effort has no rung (e.g. ``("reasoning_effort",)``).

    Note:
        Resolution scans families in registration order and the first full
        match wins — a *new* family whose patterns overlap an already-registered
        one (built-ins included) never wins for models both match. To change
        behavior for such models, re-register under the existing family's name.

    Raises:
        ValueError: If a rung value is not a dict (it must be literal kwargs),
            or if any match/provider pattern is empty after normalization
            (an empty pattern would silently match every model).
    """
    for rung_name, rung_kwargs in rungs.items():
        if not isinstance(rung_kwargs, dict):
            raise ValueError(f"Rung {rung_name!r} of family {name!r} must be a dict of constructor kwargs, got {type(rung_kwargs).__name__}")
    # A lone string is accepted as a single pattern (a missing trailing comma in
    # a 1-tuple is a string in Python — don't let it degrade to char matching).
    if isinstance(match_substrings, str):
        match_substrings = (match_substrings,)
    if isinstance(provider_substrings, str):
        provider_substrings = (provider_substrings,)
    # Reject patterns that normalize to nothing: "" is a substring of every
    # name, so an empty pattern (or empty AND-group) would hijack all models.
    for group in _match_groups(match_substrings):
        if not group or any(not sub for sub in group):
            raise ValueError(f"match_substrings for family {name!r} contains an empty pattern or group — it would match every model")
    if any(not sub for sub in _match_substrings(provider_substrings)):
        raise ValueError(f"provider_substrings for family {name!r} contains an empty pattern — it would match every provider")
    _FAMILIES[name] = {
        "match": tuple(match_substrings),
        "provider_match": tuple(provider_substrings),
        "native_path": tuple(native_path) if native_path is not None else None,
        "rungs": {str(k).lower().strip(): deepcopy(v) for k, v in rungs.items()},
    }


def merge_reasoning_config(reasoning: dict[str, Any], explicit: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Deep-merge reasoning kwargs under explicitly-passed config.

    Explicit values win on leaf conflicts — raw kwargs are the caller's escape
    hatch and must stay authoritative. Non-conflicting keys from both sides
    compose (a caller's partial ``extra_body`` merges alongside the translated
    flags instead of clobbering them).

    Args:
        reasoning: Kwargs emitted by :func:`build_reasoning_kwargs`.
        explicit: The caller/settings config assembled by the factory.

    Returns:
        Tuple of (merged config, dotted paths where an explicit value overrode
        a differing reasoning value — for the factory to warn about).
    """
    conflicts: list[str] = []

    def _merge(base: dict[str, Any], override: dict[str, Any], path: str) -> dict[str, Any]:
        merged = dict(base)
        for key, override_value in override.items():
            key_path = f"{path}.{key}" if path else key
            if key in merged and isinstance(merged[key], dict) and isinstance(override_value, dict):
                merged[key] = _merge(merged[key], override_value, key_path)
            else:
                if key in merged and merged[key] != override_value:
                    conflicts.append(key_path)
                merged[key] = override_value
        return merged

    return _merge(deepcopy(reasoning), explicit, ""), conflicts

-------

tests/test_ai_engine/models/test_create_chat_model_dynamic.py
----
"""Tests for create_chat_model with arbitrary string providers.

Covers the engine-side wiring that lets ``create_chat_model("acme")``
flow through ``ProviderFactory.get_provider_settings("acme")`` without
the historical ``ProviderType(provider.lower())`` choke point.

Routing rules under test:

- Known ``ProviderType`` members and their lowercased string equivalents
  still resolve to the historical path (regression).
- An arbitrary string provider routes to ``ChatOpenAI`` by default.
- A provider whose **name** contains ``mistral`` (e.g. ``"mistral_eu"``)
  routes to ``ChatMistralAI`` even if the model name does not.
- A provider with arbitrary name but ``mistral``-flavored model name
  still routes to ``ChatMistralAI`` via the existing model-name fallback.

Mocks ``_create_openai_model`` and ``_create_mistral_model`` so the
dispatch is observed without instantiating real httpx clients.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sta_agent_core.types import ProviderType


_CRED_VARS: tuple[str, ...] = (
    "LLMAAS_API_KEY",
    "LLMAAS_BASE_URL",
    "LLMAAS_BIG_MODEL",
    "LLMAAS_SMALL_MODEL",
    "LLMAAS_THINKING_MODEL",
    "LLMAAS_MULTIMODAL_MODEL",
    "MISTRAL_API_KEY",
    "MISTRAL_BASE_URL",
    "MISTRAL_BIG_MODEL",
    "MISTRAL_SMALL_MODEL",
    "MISTRAL_THINKING_MODEL",
    "MISTRAL_MULTIMODAL_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BIG_MODEL",
    "OPENAI_SMALL_MODEL",
    "OPENAI_THINKING_MODEL",
    "OPENAI_MULTIMODAL_MODEL",
    # OPENAI_BASE_URL deliberately in _DELETE_VARS — OpenAISettings has a
    # non-None field default that "" would coerce away via empty_str_to_none.
    "ACME_API_KEY",
    "ACME_BASE_URL",
    "ACME_BIG_MODEL",
    "ACME_SMALL_MODEL",
    "ACME_THINKING_MODEL",
    "ACME_MULTIMODAL_MODEL",
    "API_KEY",
    "BASE_URL",
    "BIG_MODEL",
    "SMALL_MODEL",
    "THINKING_MODEL",
    "MULTIMODAL_MODEL",
    "MISTRAL_EU_API_KEY",
    "MISTRAL_EU_BASE_URL",
    "MISTRAL_EU_BIG_MODEL",
    "MISTRAL_EU_SMALL_MODEL",
    "MISTRAL_EU_THINKING_MODEL",
    "MISTRAL_EU_MULTIMODAL_MODEL",
)

_DELETE_VARS: tuple[str, ...] = (
    "LLM_PROVIDER",
    "OPENAI_BASE_URL",  # Use OpenAISettings field default ('https://api.openai.com/v1').
    "OPENAI_MODEL",
    "ACME_MODEL",
    "MISTRAL_EU_MODEL",
)


@pytest.fixture(autouse=True)
def _isolate_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same isolation strategy as ``tests/test_core/config/test_providers_dynamic.py``.

    Set credentials to ``""`` so the validator nulls them; delete vars whose
    target field is a required ``str``.
    """
    for var in _CRED_VARS:
        monkeypatch.setenv(var, "")
    for var in _DELETE_VARS:
        monkeypatch.delenv(var, raising=False)

    # Imported by name so a future rename of the helper breaks loudly.
    from sta_agent_core.config.providers import _reset_dynamic_state

    _reset_dynamic_state()


# =============================================================================
# Regression — known providers still route through the historical paths
# =============================================================================


@pytest.mark.unit
class TestKnownProviderRegression:
    """``ProviderType`` members and their string forms must keep dispatching the same way."""

    def test_llmaas_enum_routes_to_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")
        monkeypatch.setenv("LLMAAS_MODEL", "llama33-70b-instruct")

        with (
            patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock,
            patch("sta_agent_engine.models.custom_chat_model._create_mistral_model") as mistral_mock,
        ):
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model(ProviderType.LLMAAS)

            openai_mock.assert_called_once()
            mistral_mock.assert_not_called()

    def test_llmaas_string_routes_identically_to_enum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")
        monkeypatch.setenv("LLMAAS_MODEL", "llama33-70b-instruct")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("llmaas")
            openai_mock.assert_called_once()

    def test_mistral_enum_routes_to_mistral(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "k")
        monkeypatch.setenv("MISTRAL_MODEL", "mistral-small-2603")

        with (
            patch("sta_agent_engine.models.custom_chat_model._create_mistral_model") as mistral_mock,
            patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock,
        ):
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model(ProviderType.MISTRAL)

            mistral_mock.assert_called_once()
            openai_mock.assert_not_called()


# =============================================================================
# Dynamic providers — arbitrary string names route through the new path
# =============================================================================


@pytest.mark.unit
class TestDynamicProviderDispatch:
    """``create_chat_model("acme")`` must succeed and route by model / provider name."""

    def test_arbitrary_provider_routes_to_openai_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "gpt-4o-clone")

        with (
            patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock,
            patch("sta_agent_engine.models.custom_chat_model._create_mistral_model") as mistral_mock,
        ):
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme")

            openai_mock.assert_called_once()
            mistral_mock.assert_not_called()
            # Verify the merged config flowed through correctly.
            config = openai_mock.call_args.args[0]
            assert config["api_key"] == "acme-key"
            assert config["base_url"] == "https://api.acme.test/v1"
            assert config["model"] == "gpt-4o-clone"

    def test_provider_name_with_mistral_substring_routes_to_mistral(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``"mistral_eu"`` carries the brand hint → ChatMistralAI even with neutral model name."""
        monkeypatch.setenv("MISTRAL_EU_API_KEY", "eu-key")
        monkeypatch.setenv("MISTRAL_EU_BASE_URL", "https://eu.mistral.test")
        monkeypatch.setenv("MISTRAL_EU_MODEL", "neutral-model-name")

        with (
            patch("sta_agent_engine.models.custom_chat_model._create_mistral_model") as mistral_mock,
            patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock,
        ):
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("mistral_eu")

            mistral_mock.assert_called_once()
            openai_mock.assert_not_called()

    def test_dynamic_provider_with_mistral_model_name_routes_to_mistral(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Existing model-name substring fallback (``"mistral"`` in model) still wins."""
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "mistral-large-2411")

        with (
            patch("sta_agent_engine.models.custom_chat_model._create_mistral_model") as mistral_mock,
            patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock,
        ):
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme")

            mistral_mock.assert_called_once()
            openai_mock.assert_not_called()

    def test_case_insensitive_dynamic_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "k")
        monkeypatch.setenv("ACME_BASE_URL", "https://x")
        monkeypatch.setenv("ACME_MODEL", "m")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("ACME")
            openai_mock.assert_called_once()


# =============================================================================
# Kwargs override the resolved settings
# =============================================================================


@pytest.mark.unit
class TestKwargOverrides:
    """Explicit kwargs should still override env-derived settings."""

    def test_kwargs_override_dynamic_env_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "env-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://env.test")
        monkeypatch.setenv("ACME_MODEL", "env-model")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model(
                "acme",
                api_key="override-key",
                base_url="https://override.test",
                model="override-model",
            )

            config = openai_mock.call_args.args[0]
            assert config["api_key"] == "override-key"
            assert config["base_url"] == "https://override.test"
            assert config["model"] == "override-model"


# =============================================================================
# Capacity tiers — create_chat_model resolves provider tier before dispatch
# =============================================================================


@pytest.mark.unit
class TestModelTiers:
    """``tier=`` selects provider tier models without leaking tier fields to clients."""

    def test_tier_big_selects_big_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "acme-default")
        monkeypatch.setenv("ACME_BIG_MODEL", "acme-big")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme", tier="big")

            config = openai_mock.call_args.args[0]
            assert config["model"] == "acme-big"

    def test_tier_falls_back_to_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "acme-default")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme", tier="small")

            config = openai_mock.call_args.args[0]
            assert config["model"] == "acme-default"

    def test_explicit_model_kwarg_overrides_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "acme-default")
        monkeypatch.setenv("ACME_BIG_MODEL", "acme-big")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme", tier="big", model="explicit-model")

            config = openai_mock.call_args.args[0]
            assert config["model"] == "explicit-model"

    def test_provider_type_tier_compatibility(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")
        monkeypatch.setenv("LLMAAS_MODEL", "llama-default")
        monkeypatch.setenv("LLMAAS_BIG_MODEL", "llama-big")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model(ProviderType.LLMAAS, tier="big")

            config = openai_mock.call_args.args[0]
            assert config["model"] == "llama-big"

    def test_tier_fields_do_not_leak_to_openai_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "acme-default")
        monkeypatch.setenv("ACME_BIG_MODEL", "acme-big")
        monkeypatch.setenv("ACME_SMALL_MODEL", "acme-small")
        monkeypatch.setenv("ACME_THINKING_MODEL", "acme-thinking")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme", tier="thinking")

            config = openai_mock.call_args.args[0]
            assert config["model"] == "acme-thinking"
            assert "tier" not in config
            assert "big_model" not in config
            assert "small_model" not in config
            assert "thinking_model" not in config


# =============================================================================
# Multimodal — create_chat_model(..., multimodal=True) resolves a vision model
# =============================================================================


@pytest.mark.unit
class TestMultimodalModel:
    """``multimodal=True`` resolves a vision model or hard-fails — never silently text-only."""

    def test_explicit_multimodal_model_used_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An explicit slot is authoritative — used even when is_multimodal() does
        # not recognize the name (operator declaration wins).
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "acme-default")
        monkeypatch.setenv("ACME_MULTIMODAL_MODEL", "acme-vision-exotic")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme", multimodal=True)

            assert openai_mock.call_args.args[0]["model"] == "acme-vision-exotic"

    def test_falls_back_to_model_when_model_is_multimodal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No explicit slot, but MODEL is a recognized vision model (qwen3.6 prefix).
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "qwen3.6-vl")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme", multimodal=True)

            assert openai_mock.call_args.args[0]["model"] == "qwen3.6-vl"

    def test_hard_fails_when_no_multimodal_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No slot and MODEL is text-only → raise rather than return a text model.
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "acme-default")

        from sta_agent_engine.models.custom_chat_model import create_chat_model

        with pytest.raises(ValueError, match="multimodal=True"):
            create_chat_model("acme", multimodal=True)

    def test_multimodal_takes_precedence_over_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "acme-default")
        monkeypatch.setenv("ACME_BIG_MODEL", "acme-big")
        monkeypatch.setenv("ACME_MULTIMODAL_MODEL", "acme-vision")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme", tier="big", multimodal=True)

            assert openai_mock.call_args.args[0]["model"] == "acme-vision"

    def test_explicit_model_kwarg_ignores_multimodal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "acme-default")
        monkeypatch.setenv("ACME_MULTIMODAL_MODEL", "acme-vision")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme", multimodal=True, model="explicit-model")

            assert openai_mock.call_args.args[0]["model"] == "explicit-model"

    def test_multimodal_model_field_does_not_leak_to_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "acme-key")
        monkeypatch.setenv("ACME_BASE_URL", "https://api.acme.test/v1")
        monkeypatch.setenv("ACME_MODEL", "acme-default")
        monkeypatch.setenv("ACME_MULTIMODAL_MODEL", "acme-vision")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme", multimodal=True)

            assert "multimodal_model" not in openai_mock.call_args.args[0]


# =============================================================================
# DeprecationWarning on silent OpenAI fallback (slated for hard-fail in 0.10.0)
# =============================================================================


@pytest.mark.unit
class TestSilentOpenAIFallbackWarning:
    """When the OpenAI dispatch falls back to OPENAI_API_KEY + api.openai.com,
    a DeprecationWarning is emitted. Behavior is preserved for backward compat
    with the main branch; the silent fallback will raise in 0.10.0."""

    def test_unknown_provider_with_no_env_warns_and_routes_to_openai(self) -> None:
        """The reported footgun: create_chat_model('test_provider') with no setup.
        Warns instead of raising so callers can migrate before 0.10.0."""
        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with pytest.warns(DeprecationWarning, match="silent fallback"):
                create_chat_model("test_provider")
            openai_mock.assert_called_once()

    def test_warning_names_expected_env_vars(self) -> None:
        from sta_agent_engine.models.custom_chat_model import create_chat_model

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model"), pytest.warns(DeprecationWarning) as warning_records:
            create_chat_model("test_provider")
        # An implicit-default-model warning now co-occurs; select the credential one.
        msg = next(str(w.message) for w in warning_records if "silent fallback" in str(w.message))
        assert "TEST_PROVIDER_API_KEY" in msg
        assert "TEST_PROVIDER_BASE_URL" in msg
        assert "0.10.0" in msg

    def test_warning_hints_at_openai_provider_for_non_openai_callers(self) -> None:
        """Most accidental OPENAI fallbacks are actually OpenAI usage; point them at provider='openai'."""
        from sta_agent_engine.models.custom_chat_model import create_chat_model

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model"), pytest.warns(DeprecationWarning) as warning_records:
            create_chat_model("test_provider")
        msg = next(str(w.message) for w in warning_records if "silent fallback" in str(w.message))
        assert "create_chat_model('openai')" in msg

    def test_warning_omits_openai_hint_when_already_using_openai_provider(self) -> None:
        """provider='openai' with missing OPENAI_API_KEY: don't suggest switching to itself."""
        from sta_agent_engine.models.custom_chat_model import create_chat_model

        # OpenAISettings has a default base_url, so api_key is the only missing piece.
        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model"), pytest.warns(DeprecationWarning) as warning_records:
            create_chat_model("openai")
        msg = next(str(w.message) for w in warning_records if "silent fallback" in str(w.message))
        assert "switch to create_chat_model" not in msg
        assert "OPENAI_API_KEY" in msg

    def test_kwargs_alone_satisfy_credentials_no_warning(self) -> None:
        """No env vars set — passing kwargs is enough; no DeprecationWarning."""
        import warnings as _warnings

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                create_chat_model(
                    "test_provider",
                    api_key="k",
                    base_url="https://x.test",
                    model="m",
                )
            openai_mock.assert_called_once()
            silent_fallback_warnings = [w for w in caught if "silent fallback" in str(w.message)]
            assert silent_fallback_warnings == []

    def test_partial_credentials_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API key alone (no base_url) still warns about base_url fallback."""
        monkeypatch.setenv("ACME_API_KEY", "k")
        from sta_agent_engine.models.custom_chat_model import create_chat_model

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model"), pytest.warns(DeprecationWarning, match="base_url"):
            create_chat_model("acme")

    def test_helper_builds_correct_env_var_names_for_known_provider(self) -> None:
        """Direct unit test on the warning helper for a built-in ProviderType member.

        Originally went through ``create_chat_model("llmaas")`` end-to-end, but
        that path is sensitive to two pre-existing footguns that surface as test
        pollution under the full suite: (1) ``LLMaaSSettings.model`` defaults to
        ``mistral-small-2506`` which dispatches to ChatMistralAI via the
        model-name substring check; (2) ``LLMAAS_API_KEY`` / ``LLMAAS_BASE_URL``
        can be set by earlier tests via direct ``os.environ`` assignment that
        ``monkeypatch.setenv`` can't always counter. Testing the helper directly
        proves the env-var naming contract for built-in providers without those
        cross-test dependencies.
        """
        from sta_agent_core.types import ProviderType
        from sta_agent_engine.models.custom_chat_model import _warn_silent_openai_fallback

        with pytest.warns(DeprecationWarning) as warning_records:
            _warn_silent_openai_fallback(ProviderType.LLMAAS, ["api_key", "base_url"])
        msg = str(warning_records[0].message)
        assert "LLMAAS_API_KEY" in msg
        assert "LLMAAS_BASE_URL" in msg
        assert "0.10.0" in msg

    def test_mistral_dispatch_does_not_warn_about_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mistral has its own legitimate base_url fallback; no OpenAI-fallback warning."""
        import warnings as _warnings

        monkeypatch.setenv("MISTRAL_API_KEY", "k")
        # Deliberately NO MISTRAL_BASE_URL — Mistral SDK has its own default.

        with patch("sta_agent_engine.models.custom_chat_model._create_mistral_model"):
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                create_chat_model("mistral")
            silent_fallback_warnings = [w for w in caught if "silent fallback" in str(w.message)]
            assert silent_fallback_warnings == []


# =============================================================================
# Implicit-default-model deprecation — relying on a provider's built-in default
# model (no *_MODEL env var, no model= kwarg) is deprecated for ALL providers.
# =============================================================================


def _assert_no_implicit_default_warning(caught: list) -> None:
    offending = [w for w in caught if issubclass(w.category, DeprecationWarning) and "implicit default" in str(w.message)]
    assert offending == [], f"unexpected implicit-default warning: {[str(w.message) for w in offending]}"


@pytest.mark.unit
class TestModelImplicitDefaultHelper:
    """Unit tests for ``_model_is_implicit_default`` — the detection behind the warning.

    Tested directly (not only through ``create_chat_model``) because the env/.env
    plumbing makes built-in providers' model resolution non-deterministic in the
    test environment. The helper is pure: given a settings object, tier, and the
    resolved model, it answers "did this come from the class default?".
    """

    def test_value_equal_to_class_default_is_implicit(self) -> None:
        from sta_agent_core.config.providers import LLMaaSSettings
        from sta_agent_engine.models.custom_chat_model import _model_is_implicit_default

        settings = LLMaaSSettings(model="mistral-small-2506")  # == class default
        assert _model_is_implicit_default(settings, "default", "mistral-small-2506") is True

    def test_value_differing_from_default_is_explicit(self) -> None:
        from sta_agent_core.config.providers import LLMaaSSettings
        from sta_agent_engine.models.custom_chat_model import _model_is_implicit_default

        settings = LLMaaSSettings(model="some-explicit-model")
        assert _model_is_implicit_default(settings, "default", "some-explicit-model") is False

    def test_builtin_mistral_default_is_implicit(self) -> None:
        """Built-ins are not exempt (Badr's ask): relying on Mistral's default warns."""
        from sta_agent_core.config.providers import MistralSettings
        from sta_agent_engine.models.custom_chat_model import _model_is_implicit_default

        settings = MistralSettings(model="mistral-small-2603")  # == class default
        assert _model_is_implicit_default(settings, "default", "mistral-small-2603") is True

    def test_dynamic_provider_base_default_is_implicit(self) -> None:
        from sta_agent_core.config.providers import BaseProviderSettings
        from sta_agent_engine.models.custom_chat_model import _model_is_implicit_default

        settings = BaseProviderSettings(model="llama33-70b-instruct")  # base default
        assert _model_is_implicit_default(settings, "default", "llama33-70b-instruct") is True

    def test_explicit_tier_field_is_not_implicit(self) -> None:
        """An explicit *_BIG_MODEL means the big tier did not fall through to the default."""
        from sta_agent_core.config.providers import LLMaaSSettings
        from sta_agent_engine.models.custom_chat_model import _model_is_implicit_default

        settings = LLMaaSSettings(big_model="big-explicit")
        assert _model_is_implicit_default(settings, "big", "big-explicit") is False

    def test_tier_falling_through_to_default_is_implicit(self) -> None:
        from sta_agent_core.config.providers import LLMaaSSettings
        from sta_agent_engine.models.custom_chat_model import _model_is_implicit_default

        # model set to its class default, no big_model -> big tier cascades to the default.
        settings = LLMaaSSettings(model="mistral-small-2506")
        assert _model_is_implicit_default(settings, "big", settings.get_model("big")) is True


@pytest.mark.unit
class TestImplicitModelDefaultDeprecation:
    """End-to-end: relying on the implicit default model warns. Uses dynamic providers
    (and a model-equals-default built-in) to avoid the repo .env supplying a model."""

    def test_warns_for_dynamic_provider_implicit_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "k")
        monkeypatch.setenv("ACME_BASE_URL", "https://acme.test")
        # ACME_MODEL is deleted by the autouse fixture -> base default applies, no .env entry.

        with (
            patch("sta_agent_engine.models.custom_chat_model._create_openai_model"),
            patch("sta_agent_engine.models.custom_chat_model._create_mistral_model"),
        ):
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with pytest.warns(DeprecationWarning, match="implicit default"):
                create_chat_model("acme")

    def test_warns_for_builtin_when_relying_on_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Built-in providers are not exempt: setting MISTRAL_MODEL to the default still warns."""
        monkeypatch.setenv("MISTRAL_API_KEY", "k")
        monkeypatch.setenv("MISTRAL_MODEL", "mistral-small-2603")  # == built-in default

        with (
            patch("sta_agent_engine.models.custom_chat_model._create_mistral_model"),
            patch("sta_agent_engine.models.custom_chat_model._create_openai_model"),
        ):
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with pytest.warns(DeprecationWarning) as records:
                create_chat_model("mistral")
            msg = next(str(r.message) for r in records if "implicit default" in str(r.message))
            assert "MISTRAL_MODEL" in msg
            assert "0.11.0" in msg

    def test_no_warning_when_dynamic_model_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import warnings as _warnings

        monkeypatch.setenv("ACME_API_KEY", "k")
        monkeypatch.setenv("ACME_BASE_URL", "https://acme.test")
        monkeypatch.setenv("ACME_MODEL", "acme-large")

        with (
            patch("sta_agent_engine.models.custom_chat_model._create_openai_model"),
            patch("sta_agent_engine.models.custom_chat_model._create_mistral_model"),
        ):
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                create_chat_model("acme")
            _assert_no_implicit_default_warning(caught)

    def test_no_warning_when_model_kwarg_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import warnings as _warnings

        monkeypatch.setenv("ACME_API_KEY", "k")
        monkeypatch.setenv("ACME_BASE_URL", "https://acme.test")
        # ACME_MODEL deleted by fixture -> default would apply, but the kwarg wins.

        with (
            patch("sta_agent_engine.models.custom_chat_model._create_openai_model"),
            patch("sta_agent_engine.models.custom_chat_model._create_mistral_model"),
        ):
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                create_chat_model("acme", model="explicit-kwarg-model")
            _assert_no_implicit_default_warning(caught)


# =============================================================================
# BYOK aliases — provider_api_key / provider_base_url must map, never forward
# =============================================================================


@pytest.mark.unit
class TestByokAliasesDoNotLeak:
    """``provider_api_key`` / ``provider_base_url`` are aliases for ``api_key`` /
    ``base_url``, not client constructor params.

    They must be consumed into the resolved credentials and removed from the
    config handed to the client factory. Leaving them in forwards them to
    ChatOpenAI/ChatMistralAI, which route unknown kwargs into ``model_kwargs`` —
    leaking the raw key into the request body and 400-ing strict servers.
    """

    def test_openai_path_maps_and_drops_provider_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_MODEL", "acme-large")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model(
                "acme",
                provider_api_key="sk-byok",
                provider_base_url="https://byok.acme.test/v1",
            )

        config = openai_mock.call_args.args[0]
        assert config["api_key"] == "sk-byok"
        assert config["base_url"] == "https://byok.acme.test/v1"
        assert "provider_api_key" not in config
        assert "provider_base_url" not in config

    def test_mistral_path_maps_and_drops_provider_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with patch("sta_agent_engine.models.custom_chat_model._create_mistral_model") as mistral_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model(
                "mistral_eu",
                model="mistral-large",
                provider_api_key="sk-mb",
                provider_base_url="https://byok.mistral.test/v1",
            )

        config = mistral_mock.call_args.args[0]
        assert config["api_key"] == "sk-mb"
        assert config["base_url"] == "https://byok.mistral.test/v1"
        assert "provider_api_key" not in config
        assert "provider_base_url" not in config

    def test_explicit_api_key_kwarg_still_wins_without_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_MODEL", "acme-large")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("acme", api_key="sk-direct", base_url="https://direct.acme.test/v1")

        config = openai_mock.call_args.args[0]
        assert config["api_key"] == "sk-direct"
        assert config["base_url"] == "https://direct.acme.test/v1"
        assert "provider_api_key" not in config
        assert "provider_base_url" not in config


# =============================================================================
# Reasoning effort — factory wiring (translation lives in models/reasoning.py;
# these tests pin the merge into the client config)
# =============================================================================


@pytest.mark.unit
class TestReasoningEffortWiring:
    """``reasoning_effort=`` translates per family and merges into the config.

    ``None``/absent must be byte-identical to today's behavior; explicit
    caller kwargs win over translated values on leaf conflicts;
    ``reasoning_family`` pins the family and never reaches the client.
    """

    def test_absent_reasoning_effort_leaves_config_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("llmaas", model="nemotron-3-super-120b")

        config = openai_mock.call_args.args[0]
        assert "extra_body" not in config
        assert "reasoning_effort" not in config
        assert "reasoning_family" not in config

    def test_nemotron_super_low_translates_to_chat_template_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("llmaas", model="nemotron-3-super-120b", reasoning_effort="low")

        config = openai_mock.call_args.args[0]
        assert config["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}
        assert "reasoning_effort" not in config

    def test_nemotron_ultra_low_vs_medium_force_nonempty_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("llmaas", model="nemotron-3-ultra-550b", reasoning_effort="low")
            low_ctk = openai_mock.call_args.args[0]["extra_body"]["chat_template_kwargs"]
            create_chat_model("llmaas", model="nemotron-3-ultra-550b", reasoning_effort="medium")
            medium_ctk = openai_mock.call_args.args[0]["extra_body"]["chat_template_kwargs"]

        assert low_ctk == {"enable_thinking": True, "medium_effort": True, "force_nonempty_content": False}
        assert medium_ctk == {"enable_thinking": True, "medium_effort": True, "force_nonempty_content": True}

    def test_qwen_high_enables_thinking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("llmaas", model="qwen-3.6-32b", reasoning_effort="high")

        config = openai_mock.call_args.args[0]
        assert config["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}

    def test_mistral_high_routes_through_model_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "k")

        with patch("sta_agent_engine.models.custom_chat_model._create_mistral_model") as mistral_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("mistral", model="mistral-small-2603", reasoning_effort="high")

        config = mistral_mock.call_args.args[0]
        assert config["model_kwargs"] == {"reasoning_effort": "high"}
        assert "reasoning_effort" not in config

    def test_mistral_off_maps_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "k")

        with patch("sta_agent_engine.models.custom_chat_model._create_mistral_model") as mistral_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("mistral", model="mistral-small-2603", reasoning_effort="off")

        config = mistral_mock.call_args.args[0]
        assert config["model_kwargs"] == {"reasoning_effort": "none"}

    def test_devstral_gets_the_mistral_dialect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Devstral/Magistral dispatch to ChatMistralAI by name, so they must use
        # the mistral translation ("off" -> "none") — a raw passthrough would
        # send "off", which the Mistral API rejects.
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_mistral_model") as mistral_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("llmaas", model="devstral-medium-2512", reasoning_effort="off")

        config = mistral_mock.call_args.args[0]
        assert config["model_kwargs"] == {"reasoning_effort": "none"}
        assert "reasoning_effort" not in config

    def test_blank_effort_is_byte_identical_to_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # reasoning_effort="" (e.g. an empty env-var default) must behave like
        # None: nothing injected, nothing on the wire, no warning.
        import warnings as _warnings

        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                create_chat_model("llmaas", model="gpt-oss-120b", reasoning_effort="")

        assert [w for w in caught if issubclass(w.category, UserWarning) and "reasoning" in str(w.message).lower()] == []
        config = openai_mock.call_args.args[0]
        assert "reasoning_effort" not in config
        assert "extra_body" not in config

    def test_unsupported_effort_on_translated_family_warns_and_injects_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with pytest.warns(UserWarning, match="Nothing was applied"):
                create_chat_model("llmaas", model="nemotron-3-super-120b", reasoning_effort="medium")

        config = openai_mock.call_args.args[0]
        assert "extra_body" not in config
        assert "reasoning_effort" not in config

    def test_unknown_model_warns_and_forwards_top_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACME_API_KEY", "k")
        monkeypatch.setenv("ACME_BASE_URL", "https://acme.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with pytest.warns(UserWarning, match="No reasoning family matches"):
                create_chat_model("acme", model="acme-large", reasoning_effort="high")

        config = openai_mock.call_args.args[0]
        assert config["reasoning_effort"] == "high"

    def test_reasoning_warning_is_attributed_to_this_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Routed through create_chat_model the warn call is 2+ library frames
        # deep — attribution must still land on the consumer's call site (here,
        # this test file), not on a line inside the models package.
        import warnings as _warnings

        monkeypatch.setenv("ACME_API_KEY", "k")
        monkeypatch.setenv("ACME_BASE_URL", "https://acme.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model"):
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                create_chat_model("acme", model="acme-large", reasoning_effort="high")

        reasoning_warnings = [w for w in caught if "No reasoning family matches" in str(w.message)]
        assert len(reasoning_warnings) == 1
        assert reasoning_warnings[0].filename == __file__

    def test_gpt_model_passthrough_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            import warnings as _warnings

            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                create_chat_model("llmaas", model="gpt-oss-120b", reasoning_effort="low")

        assert [w for w in caught if issubclass(w.category, UserWarning) and "reasoning" in str(w.message).lower()] == []
        config = openai_mock.call_args.args[0]
        assert config["reasoning_effort"] == "low"

    def test_explicit_extra_body_wins_on_conflict_and_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with pytest.warns(UserWarning, match="explicit values win"):
                create_chat_model(
                    "llmaas",
                    model="nemotron-3-super-120b",
                    reasoning_effort="high",
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

        config = openai_mock.call_args.args[0]
        assert config["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_explicit_extra_body_composes_without_conflict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model(
                "llmaas",
                model="nemotron-3-super-120b",
                reasoning_effort="high",
                extra_body={"chat_template_kwargs": {"custom_flag": 1}},
            )

        config = openai_mock.call_args.args[0]
        assert config["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True, "custom_flag": 1}

    def test_reasoning_family_pins_gateway_aliased_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            create_chat_model("llmaas", model="chat-default", reasoning_effort="high", reasoning_family="nemotron-super")

        config = openai_mock.call_args.args[0]
        assert config["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
        assert "reasoning_family" not in config

    def test_reasoning_family_without_effort_warns_and_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A family pin alone selects a translation table but injects nothing —
        # warn so the caller learns their kwarg had no effect, and still strip
        # it so it never reaches the client constructor.
        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        with patch("sta_agent_engine.models.custom_chat_model._create_openai_model") as openai_mock:
            from sta_agent_engine.models.custom_chat_model import create_chat_model

            with pytest.warns(UserWarning, match="without reasoning_effort"):
                create_chat_model("llmaas", model="nemotron-3-super-120b", reasoning_family="nemotron-super")

        config = openai_mock.call_args.args[0]
        assert "reasoning_family" not in config

    def test_openai_path_unmocked_effort_reaches_request_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Through the REAL creator (no mock): the creators pop/filter config
        # keys, so only an end-to-end construction proves the translated kwargs
        # survive all the way into the request payload.
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        monkeypatch.setenv("LLMAAS_API_KEY", "k")
        monkeypatch.setenv("LLMAAS_BASE_URL", "https://llmaas.test")

        from sta_agent_engine.models.custom_chat_model import create_chat_model

        model = create_chat_model("llmaas", model="nemotron-3-super-120b", reasoning_effort="low")

        assert isinstance(model, ChatOpenAI)
        payload = model._get_request_payload([HumanMessage("hi")])
        assert payload["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True, "low_effort": True}

    def test_mistral_path_unmocked_effort_reaches_default_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from langchain_mistralai import ChatMistralAI

        monkeypatch.setenv("MISTRAL_API_KEY", "k")

        from sta_agent_engine.models.custom_chat_model import create_chat_model

        model = create_chat_model("mistral", model="mistral-small-2603", reasoning_effort="off")

        assert isinstance(model, ChatMistralAI)
        assert model._default_params["reasoning_effort"] == "none"

-------

tests/test_ai_engine/models/test_reasoning.py
----
"""Unit tests for the declarative reasoning-effort configuration.

Covers the pure functions in ``sta_agent_engine.models.reasoning`` — family
resolution precedence, per-family rung translation, warn-and-degrade paths,
registry extension, deep-merge conflict semantics — plus request-payload-level
assertions that the translated kwargs actually reach the client payload
(constructor-level asserts alone can lie: a kwarg can be accepted and silently
dropped before the wire).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from copy import deepcopy

import pytest

from sta_agent_engine.models import reasoning
from sta_agent_engine.models.reasoning import (
    build_reasoning_kwargs,
    merge_reasoning_config,
    register_reasoning_family,
    resolve_reasoning_family,
    supported_reasoning_efforts,
)


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    """Snapshot/restore the module-global family registry around each test."""
    snapshot = deepcopy(reasoning._FAMILIES)
    yield
    reasoning._FAMILIES.clear()
    reasoning._FAMILIES.update(snapshot)


def _assert_no_reasoning_warning(caught: list[warnings.WarningMessage]) -> None:
    reasoning_warnings = [w for w in caught if issubclass(w.category, UserWarning) and "reasoning" in str(w.message).lower()]
    assert reasoning_warnings == []


# =============================================================================
# Family resolution
# =============================================================================


@pytest.mark.unit
class TestResolveReasoningFamily:
    def test_nemotron_super_by_name(self) -> None:
        assert resolve_reasoning_family("nemotron-3-super-120b-A12B") == "nemotron-super"

    def test_nemotron_ultra_by_name(self) -> None:
        assert resolve_reasoning_family("Nemotron-3-Ultra-550B-A55B") == "nemotron-ultra"

    def test_mistral_by_name(self) -> None:
        assert resolve_reasoning_family("mistral-small-2603") == "mistral"

    def test_mistral_by_provider(self) -> None:
        # Gateway-aliased model name; the provider carries the signal.
        assert resolve_reasoning_family("chat-default", provider="mistral_eu") == "mistral"

    def test_qwen_by_name(self) -> None:
        assert resolve_reasoning_family("qwen-3.6-32b") == "qwen3"

    def test_gpt_by_name(self) -> None:
        assert resolve_reasoning_family("gpt-4o") == "openai"

    def test_explicit_family_pin_wins_over_name(self) -> None:
        assert resolve_reasoning_family("gpt-4o", family="nemotron-super") == "nemotron-super"

    def test_unknown_explicit_family_warns_and_falls_back(self) -> None:
        with pytest.warns(UserWarning, match="Unknown reasoning_family"):
            assert resolve_reasoning_family("mistral-small-2603", family="nope") == "mistral"

    def test_no_match_returns_none(self) -> None:
        assert resolve_reasoning_family("llama33-70b-instruct") is None

    @pytest.mark.parametrize("model", ["devstral-medium-2512", "magistral-medium"])
    def test_devstral_and_magistral_use_the_mistral_dialect(self, model: str) -> None:
        # These names dispatch to ChatMistralAI (_is_mistral_model), so they must
        # get the mistral wire dialect — a raw "off" passthrough would send a
        # value the Mistral API rejects (it accepts only none/high).
        assert resolve_reasoning_family(model) == "mistral"
        assert build_reasoning_kwargs(model, "off") == {"model_kwargs": {"reasoning_effort": "none"}}

    def test_ministral_is_not_mistral_family(self) -> None:
        # "ministral" contains neither "mistral" nor the other brand names —
        # mirrors _is_mistral_model, which doesn't dispatch it to ChatMistralAI.
        assert resolve_reasoning_family("ministral-8b-2410") is None

    def test_provider_match_beats_name_match_across_families(self) -> None:
        # Documented precedence: provider substring > model-name groups —
        # even when the name alone would resolve a different family.
        assert resolve_reasoning_family("nemotron-3-super-120b", provider="mistral_eu") == "mistral"

    @pytest.mark.parametrize(
        "slug",
        [
            "qwen3.6",
            "qwen3-6",
            "qwen-3.6-32b",
            "qwen3:32b",
            "Qwen/Qwen3.6-235B-A22B-Instruct",
            "qwen_3_6_fp8",
        ],
    )
    def test_qwen_slug_variants_all_resolve(self, slug: str) -> None:
        """The same model arrives under different slug conventions per provider —
        matching is separator-insensitive so they all land in one family."""
        assert resolve_reasoning_family(slug) == "qwen3"

    def test_qwen2_does_not_match_qwen3_family(self) -> None:
        assert resolve_reasoning_family("qwen2.5-72b-instruct") is None

    @pytest.mark.parametrize(
        "slug",
        [
            "nemotron-3-ultra-550b-A55B",
            "nvidia/Nemotron-3-Ultra-550B",
            "nemotron_3_ultra_550b",
            "Nemotron-3.1-Ultra-600B",
        ],
    )
    def test_nemotron_ultra_slug_variants_all_resolve(self, slug: str) -> None:
        assert resolve_reasoning_family(slug) == "nemotron-ultra"

    def test_slug_variants_translate_identically(self) -> None:
        assert build_reasoning_kwargs("qwen3-6", "high") == build_reasoning_kwargs("qwen3.6", "high")
        assert build_reasoning_kwargs("qwen3.6", "high")["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


# =============================================================================
# Supported efforts
# =============================================================================


@pytest.mark.unit
class TestSupportedReasoningEfforts:
    def test_nemotron_ultra_rungs(self) -> None:
        assert supported_reasoning_efforts("nemotron-3-ultra-550b") == frozenset({"off", "low", "medium", "high"})

    def test_nemotron_super_rungs(self) -> None:
        assert supported_reasoning_efforts("nemotron-3-super-120b") == frozenset({"off", "low", "high"})

    def test_mistral_rungs(self) -> None:
        assert supported_reasoning_efforts("mistral-small-2603") == frozenset({"off", "high"})

    def test_qwen_rungs(self) -> None:
        # Card-documented contract only (Qwen/Qwen3.6-27B): binary enable_thinking.
        # Graded thinking budgets are serving-stack-dependent — consumers re-register.
        assert supported_reasoning_efforts("qwen-3.6") == frozenset({"off", "high"})

    def test_openai_passthrough_is_empty(self) -> None:
        assert supported_reasoning_efforts("gpt-4o") == frozenset()

    def test_unknown_model_is_empty(self) -> None:
        assert supported_reasoning_efforts("llama33-70b-instruct") == frozenset()


# =============================================================================
# Translation
# =============================================================================


@pytest.mark.unit
class TestBuildReasoningKwargs:
    def test_none_effort_returns_empty(self) -> None:
        assert build_reasoning_kwargs("mistral-small-2603", None) == {}

    @pytest.mark.parametrize("blank", ["", "   "])
    @pytest.mark.parametrize("model", ["gpt-4o", "mistral-small-2603", "nemotron-3-super-120b"])
    def test_blank_effort_means_unset_and_is_silent(self, model: str, blank: str) -> None:
        # A blank effort (e.g. os.getenv("REASONING_EFFORT", "")) must behave
        # like None — nothing injected, no warning. Without this guard the
        # native-passthrough families would put "" on the wire (400 on OpenAI).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert build_reasoning_kwargs(model, blank) == {}
        _assert_no_reasoning_warning(caught)

    def test_mistral_high(self) -> None:
        assert build_reasoning_kwargs("mistral-small-2603", "high") == {"model_kwargs": {"reasoning_effort": "high"}}

    def test_mistral_off_maps_to_none(self) -> None:
        assert build_reasoning_kwargs("mistral-medium-3-5", "off") == {"model_kwargs": {"reasoning_effort": "none"}}

    def test_nemotron_super_low(self) -> None:
        assert build_reasoning_kwargs("nemotron-3-super-120b", "low") == {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}
        }

    def test_nemotron_super_high_is_full_thinking(self) -> None:
        assert build_reasoning_kwargs("nemotron-3-super-120b", "high") == {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}

    def test_nemotron_ultra_low_sends_force_nonempty_false(self) -> None:
        # low explicitly sends force_nonempty_content=False (not absent) so a
        # server-side template default can't force it back on.
        kwargs = build_reasoning_kwargs("nemotron-3-ultra-550b", "low")
        ctk = kwargs["extra_body"]["chat_template_kwargs"]
        assert ctk == {"enable_thinking": True, "medium_effort": True, "force_nonempty_content": False}

    def test_nemotron_ultra_medium_forces_content(self) -> None:
        kwargs = build_reasoning_kwargs("nemotron-3-ultra-550b", "medium")
        ctk = kwargs["extra_body"]["chat_template_kwargs"]
        assert ctk == {"enable_thinking": True, "medium_effort": True, "force_nonempty_content": True}

    def test_nemotron_ultra_high_full_thinking_forces_content(self) -> None:
        kwargs = build_reasoning_kwargs("nemotron-3-ultra-550b", "high")
        ctk = kwargs["extra_body"]["chat_template_kwargs"]
        assert ctk == {"enable_thinking": True, "force_nonempty_content": True}

    def test_qwen_off_and_high_are_the_binary_card_contract(self) -> None:
        assert build_reasoning_kwargs("qwen-3.6-32b", "off") == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        assert build_reasoning_kwargs("qwen-3.6-32b", "high") == {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}

    def test_qwen_graded_efforts_warn_until_gateway_registers_budgets(self) -> None:
        with pytest.warns(UserWarning, match="Nothing was applied"):
            assert build_reasoning_kwargs("qwen-3.6-32b", "medium") == {}

    def test_reregistered_qwen_budgets_take_over(self) -> None:
        # The documented consumer path for gateways whose vLLM supports thinking
        # budgets: re-register the family with graded rungs.
        register_reasoning_family(
            "qwen3",
            rungs={
                "off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
                "medium": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 2048}},
                "high": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}},
                "xhigh": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 8192}},
            },
            match_substrings=("qwen3",),
        )
        kwargs = build_reasoning_kwargs("qwen3.6-27b", "medium")
        assert kwargs["extra_body"]["thinking_token_budget"] == 2048

    def test_effort_is_case_insensitive(self) -> None:
        assert build_reasoning_kwargs("mistral-small-2603", "HIGH") == {"model_kwargs": {"reasoning_effort": "high"}}

    def test_returned_dict_is_isolated_from_table(self) -> None:
        first = build_reasoning_kwargs("nemotron-3-super-120b", "low")
        first["extra_body"]["chat_template_kwargs"]["enable_thinking"] = "corrupted"
        second = build_reasoning_kwargs("nemotron-3-super-120b", "low")
        assert second["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    def test_unsupported_effort_native_family_warns_and_passes_through(self) -> None:
        with pytest.warns(UserWarning, match="supported: \\['high', 'off'\\]"):
            kwargs = build_reasoning_kwargs("mistral-small-2603", "low")
        assert kwargs == {"model_kwargs": {"reasoning_effort": "low"}}

    def test_unsupported_effort_translated_family_warns_and_injects_nothing(self) -> None:
        with pytest.warns(UserWarning, match="Nothing was applied"):
            assert build_reasoning_kwargs("nemotron-3-super-120b", "medium") == {}

    def test_unknown_model_warns_and_passes_through_top_level(self) -> None:
        with pytest.warns(UserWarning, match="No reasoning family matches"):
            kwargs = build_reasoning_kwargs("llama33-70b-instruct", "high")
        assert kwargs == {"reasoning_effort": "high"}

    def test_gpt_passthrough_is_silent(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kwargs = build_reasoning_kwargs("gpt-4o", "low")
        assert kwargs == {"reasoning_effort": "low"}
        _assert_no_reasoning_warning(caught)

    def test_warnings_are_attributed_to_the_caller_not_the_library(self) -> None:
        # Attribution matters twice: the reported file:line is the caller's
        # diagnostic entry point, and Python's default once-per-location filter
        # dedupes by it — pointing at an internal line would collapse every
        # consumer call site into one silenced location.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_reasoning_kwargs("llama33-70b-instruct", "high")
            resolve_reasoning_family("mistral-small-2603", family="typo-family")
        assert len(caught) == 2
        assert all(w.filename == __file__ for w in caught)


# =============================================================================
# Registry extension
# =============================================================================


@pytest.mark.unit
class TestRegisterReasoningFamily:
    def test_registered_family_resolves_and_translates(self) -> None:
        register_reasoning_family(
            "acme-thinker",
            rungs={"off": {"extra_body": {"think": False}}, "high": {"extra_body": {"think": True}}},
            match_substrings=("acme-think",),
        )
        assert resolve_reasoning_family("acme-thinker-9b") == "acme-thinker"
        assert build_reasoning_kwargs("acme-thinker-9b", "high") == {"extra_body": {"think": True}}

    def test_reregistering_builtin_overrides_it(self) -> None:
        register_reasoning_family(
            "mistral",
            rungs={"off": {"model_kwargs": {"reasoning_effort": "disabled"}}},
            match_substrings=("mistral",),
            provider_substrings=("mistral",),
        )
        assert build_reasoning_kwargs("mistral-small-2603", "off") == {"model_kwargs": {"reasoning_effort": "disabled"}}

    def test_non_dict_rung_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            register_reasoning_family("bad", rungs={"high": "on"})  # type: ignore[dict-item]

    def test_rung_names_are_normalized(self) -> None:
        register_reasoning_family("acme", rungs={" HIGH ": {"extra_body": {"x": 1}}}, match_substrings=("acme",))
        assert build_reasoning_kwargs("acme-1b", "high") == {"extra_body": {"x": 1}}

    def test_flat_tuple_is_one_and_group(self) -> None:
        register_reasoning_family("acme", rungs={"high": {"extra_body": {"x": 1}}}, match_substrings=("acme", "turbo"))
        assert resolve_reasoning_family("acme-turbo-7b") == "acme"
        assert resolve_reasoning_family("acme-7b") is None  # missing "turbo" — AND semantics

    def test_tuple_of_tuples_is_or_of_and_groups(self) -> None:
        register_reasoning_family(
            "acme",
            rungs={"high": {"extra_body": {"x": 1}}},
            match_substrings=(("acme", "turbo"), ("acmeturbo-lite",)),
        )
        assert resolve_reasoning_family("acme-turbo-7b") == "acme"  # group 1: both present
        assert resolve_reasoning_family("AcmeTurbo-Lite-1B") == "acme"  # group 2: alias dialect
        assert resolve_reasoning_family("acme-7b") is None  # no group fully matches

    def test_lone_string_match_is_one_pattern_not_chars(self) -> None:
        # ("acme-9x") without a trailing comma is a string — must not degrade to
        # character-wise matching (which would match any name containing those chars).
        register_reasoning_family("acme", rungs={"high": {"extra_body": {"x": 1}}}, match_substrings="acme-9x")
        assert resolve_reasoning_family("acme-9x-large") == "acme"
        assert resolve_reasoning_family("xc9-acme-em") is None

    def test_lone_string_provider_is_one_pattern_not_chars(self) -> None:
        # Same coercion guard as match_substrings, on the provider side — a
        # char-wise degrade would make almost every provider name match.
        register_reasoning_family("acme", rungs={"high": {"extra_body": {"x": 1}}}, provider_substrings="acme")
        assert resolve_reasoning_family("whatever-model", provider="acme-eu") == "acme"
        assert resolve_reasoning_family("whatever-model", provider="zzz") is None

    @pytest.mark.parametrize("patterns", [("",), "", "-", ("valid", ""), (("valid",), ())])
    def test_empty_match_pattern_is_rejected(self, patterns: object) -> None:
        # "" is a substring of every name — an empty pattern (or a pattern that
        # normalizes to nothing, like "-") would silently hijack all models.
        with pytest.raises(ValueError, match="empty pattern"):
            register_reasoning_family("bad", rungs={"high": {"extra_body": {"x": 1}}}, match_substrings=patterns)  # type: ignore[arg-type]

    def test_empty_provider_pattern_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty pattern"):
            register_reasoning_family("bad", rungs={"high": {"extra_body": {"x": 1}}}, provider_substrings=("",))

    def test_no_patterns_makes_a_pin_only_family(self) -> None:
        # No match patterns at all is legal: the family is reachable only via
        # an explicit reasoning_family= pin (never by name/provider matching).
        register_reasoning_family("pin-only", rungs={"high": {"extra_body": {"x": 1}}})
        assert resolve_reasoning_family("anything-at-all") is None
        assert resolve_reasoning_family("anything-at-all", family="pin-only") == "pin-only"

    def test_new_family_overlapping_a_builtin_never_resolves(self) -> None:
        # Resolution scans in registration order: a NEW family whose patterns
        # overlap a built-in's is unreachable for doubly-matching models. The
        # supported route is re-registering under the built-in's name
        # (test_reregistered_qwen_budgets_take_over).
        register_reasoning_family("my-qwen-budgets", rungs={"high": {"extra_body": {"x": 1}}}, match_substrings=("qwen3",))
        assert resolve_reasoning_family("qwen3.6-27b") == "qwen3"


# =============================================================================
# Merge semantics — explicit caller kwargs win on leaf conflicts
# =============================================================================


@pytest.mark.unit
class TestMergeReasoningConfig:
    def test_non_conflicting_keys_compose(self) -> None:
        merged, conflicts = merge_reasoning_config(
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
            {"extra_body": {"chat_template_kwargs": {"custom_flag": 1}}, "api_key": "k"},
        )
        assert merged["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True, "custom_flag": 1}
        assert merged["api_key"] == "k"
        assert conflicts == []

    def test_explicit_leaf_wins_and_conflict_is_reported(self) -> None:
        merged, conflicts = merge_reasoning_config(
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
        assert merged["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert conflicts == ["extra_body.chat_template_kwargs.enable_thinking"]

    def test_equal_values_are_not_conflicts(self) -> None:
        merged, conflicts = merge_reasoning_config(
            {"model_kwargs": {"reasoning_effort": "high"}},
            {"model_kwargs": {"reasoning_effort": "high"}},
        )
        assert merged["model_kwargs"]["reasoning_effort"] == "high"
        assert conflicts == []

    def test_inputs_are_not_mutated(self) -> None:
        base = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
        override = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        merge_reasoning_config(base, override)
        assert base["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


# =============================================================================
# Request-payload level — the translated kwargs must reach the wire payload
# =============================================================================


@pytest.mark.unit
class TestRequestPayloadLevel:
    def test_extra_body_reaches_chatopenai_payload(self) -> None:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        kwargs = build_reasoning_kwargs("nemotron-3-super-120b", "low")
        model = ChatOpenAI(model="nemotron-3-super-120b", api_key=SecretStr("test-key"), base_url="https://llm.test/v1", **kwargs)
        payload = model._get_request_payload([HumanMessage("hi")])
        assert payload["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True, "low_effort": True}

    def test_reasoning_effort_reaches_chatmistralai_params(self) -> None:
        from langchain_mistralai import ChatMistralAI
        from pydantic import SecretStr

        kwargs = build_reasoning_kwargs("mistral-small-2603", "high")
        model = ChatMistralAI(model_name="mistral-small-2603", api_key=SecretStr("test-key"), **kwargs)
        assert model._default_params["reasoning_effort"] == "high"

    def test_native_reasoning_effort_reaches_chatopenai_payload(self) -> None:
        # The third wire shape: the silent native passthrough (gpt-*) must land
        # as a top-level reasoning_effort key in the Chat Completions payload.
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        kwargs = build_reasoning_kwargs("gpt-oss-120b", "low")
        model = ChatOpenAI(model="gpt-oss-120b", api_key=SecretStr("test-key"), base_url="https://llm.test/v1", **kwargs)
        payload = model._get_request_payload([HumanMessage("hi")])
        assert payload["reasoning_effort"] == "low"

-------

tests/test_ai_engine/models/test_reasoning_smoke_online.py
----
"""Online smoke tests for reasoning-effort translation — REAL PAID API CALLS.

Marked ``integration_online`` — the default suite deselects these; run them
manually and deliberately. Purpose: confirm the translated wire shapes are
*accepted* by the live endpoints (no 400s) and that effort values change
observable behavior. The offline suites already pin the exact kwargs; the one
thing only a live call can prove is that the serving stack honors them.

Run (per family, only when the env is configured)::

    uv run pytest tests/test_ai_engine/models/test_reasoning_smoke_online.py -m integration_online -v

Required env: the provider credentials (``LLMAAS_*`` / ``MISTRAL_*``) plus the
smoke model names below — unset models skip their tests.

    SMOKE_NEMOTRON_SUPER_MODEL=nemotron-3-super-120b
    SMOKE_NEMOTRON_ULTRA_MODEL=nemotron-3-ultra-550b
    SMOKE_MISTRAL_MODEL=mistral-small-2603
"""

from __future__ import annotations

import os

import pytest

from sta_agent_engine.models import create_chat_model, supported_reasoning_efforts


_PROMPT = "In one short sentence: what is the capital of Japan?"

# One row per smoked family: reference model name -> the efforts exercised
# online. Kept as data (not inline parametrize literals) so the offline drift
# guard below can pin it against the family tables in the default suite.
_SMOKE_EFFORT_MATRIX: dict[str, tuple[str, ...]] = {
    "nemotron-3-super-120b": ("off", "low", "high"),
    "nemotron-3-ultra-550b": ("off", "low", "medium", "high"),
    "mistral-small-2603": ("off", "high"),
}


@pytest.mark.unit
def test_smoke_matrix_covers_supported_efforts() -> None:
    """Offline drift guard — runs in the DEFAULT suite (deliberately not marked
    ``integration_online``): if a family table grows or loses rungs, this fails
    immediately instead of waiting for the next manual paid run."""
    for model_name, efforts in _SMOKE_EFFORT_MATRIX.items():
        assert frozenset(efforts) == supported_reasoning_efforts(model_name), model_name


def _smoke_model(env_var: str) -> str:
    model = os.environ.get(env_var, "").strip()
    if not model:
        pytest.skip(f"{env_var} not set — skipping online reasoning smoke")
    return model


@pytest.mark.integration_online
class TestReasoningSmokeOnline:
    @pytest.mark.parametrize("effort", _SMOKE_EFFORT_MATRIX["nemotron-3-super-120b"])
    def test_nemotron_super_accepts_all_efforts(self, effort: str) -> None:
        model_name = _smoke_model("SMOKE_NEMOTRON_SUPER_MODEL")
        model = create_chat_model("llmaas", model=model_name, reasoning_effort=effort)
        response = model.invoke(_PROMPT)
        assert response.text

    @pytest.mark.parametrize("effort", _SMOKE_EFFORT_MATRIX["nemotron-3-ultra-550b"])
    def test_nemotron_ultra_accepts_all_efforts(self, effort: str) -> None:
        model_name = _smoke_model("SMOKE_NEMOTRON_ULTRA_MODEL")
        model = create_chat_model("llmaas", model=model_name, reasoning_effort=effort)
        response = model.invoke(_PROMPT)
        assert response.text

    @pytest.mark.parametrize("effort", _SMOKE_EFFORT_MATRIX["mistral-small-2603"])
    def test_mistral_accepts_all_efforts(self, effort: str) -> None:
        model_name = _smoke_model("SMOKE_MISTRAL_MODEL")
        model = create_chat_model("mistral", model=model_name, reasoning_effort=effort)
        response = model.invoke(_PROMPT)
        assert response.text

    def test_nemotron_super_effort_changes_reasoning_output(self) -> None:
        """``high`` should surface reasoning content that ``off`` suppresses.

        Soft check: gateway stacks differ in where they surface reasoning
        (``additional_kwargs`` vs content blocks); we only assert asymmetry
        between off and high when either side exposes it at all.
        """
        model_name = _smoke_model("SMOKE_NEMOTRON_SUPER_MODEL")

        def _reasoning_len(effort: str) -> int:
            model = create_chat_model("llmaas", model=model_name, reasoning_effort=effort)
            response = model.invoke(_PROMPT)
            reasoning = response.additional_kwargs.get("reasoning_content") or ""
            blocks = (
                [b for b in response.content if isinstance(b, dict) and b.get("type") == "reasoning"] if isinstance(response.content, list) else []
            )
            return len(str(reasoning)) + sum(len(str(b)) for b in blocks)

        off_len, high_len = _reasoning_len("off"), _reasoning_len("high")
        if off_len == high_len == 0:
            pytest.skip("Gateway surfaces no reasoning content in responses — effort asymmetry not observable client-side")
        assert high_len > off_len

-------

