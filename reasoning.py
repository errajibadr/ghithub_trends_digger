# Streaming display contract — what the TWIN UI shows

This guide is for **partner teams whose agent is displayed in the TWIN UI**
(see [Publishing an Agent](external-agent-cards.md)). When a user talks to
your agent, the UI streams its execution live. This page is the contract:
**the I/O shape your graph must expose**, and **which LLM tokens show up,
where they render, and how you control both** — with two tags: `generation`
and `non_generation`.

The contract in four rules:

1. **`messages` in, `messages` out.** Your graph takes a `messages` list as
   input and exposes a `messages` state channel as output, with the final
   answer as the **last message** (`messages[-1]`).
2. **Main graph streams by default.** Every model call in your root graph
   renders token-by-token in the chat bubble.
3. **Main graph opt-out: tag `non_generation`.** Internal calls (routers,
   graders, summarizers) are suppressed when you tag them.
4. **Subgraphs are hidden by default; opt in with `generation`.** A model
   call inside a subgraph only streams if you tag it. An active subgraph
   that streams nothing shows as a *"subagent is working"* indicator.

One thing to keep in mind throughout: **"main" is relative to whichever graph
the UI runs.** Standalone, your root graph is `main`. When the TWIN
orchestrator calls your agent, the orchestrator is `main` and your whole
graph becomes a subgraph — see
[Under the TWIN orchestrator](#under-the-twin-orchestrator-your-whole-agent-becomes-a-subgraph).

## Graph I/O: `messages` in, `messages[-1]` out

Before any streaming rule applies, your graph must speak the chat shape:

```text
        input                     your graph                    output
┌───────────────────────┐   ┌───────────────────┐   ┌──────────────────────────┐
│ {"messages": [...]}   │──▶│  state with a     │──▶│ state["messages"]        │
│  last = user's turn   │   │  `messages` key   │   │  messages[-1] = answer   │
└───────────────────────┘   └───────────────────┘   └──────────────────────────┘
```

!!! info "Status — today the UI renders streamed tokens only"
    The UI does **not yet** read your final state: what the user sees is
    built entirely from **token streaming** (the rules below). Parsing
    `messages[-1]` as the final answer ships in a later release — author to
    this I/O contract now so your agent keeps working unchanged when it does.
    One consumer reads `messages[-1]` **today**: the TWIN orchestrator —
    when it delegates to your agent, your last message *is* the result it
    uses (see
    [Under the TWIN orchestrator](#under-the-twin-orchestrator-your-whole-agent-becomes-a-subgraph)).

- **Input** — the UI invokes your graph with `{"messages": [...]}`: a list of
  chat messages whose last element is the user's turn. Don't require any
  other mandatory input key; anything else your graph needs must have a
  default.
- **Output** — your graph state must expose a `messages` channel, and when
  the run ends, **`messages[-1]` must be the final assistant answer** (an
  `AIMessage`). Don't leave a tool result or an internal note as the last
  message.
- Use the `add_messages` reducer on the channel so conversation history
  accumulates correctly across turns:

```python
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict

class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

## The contract at a glance

```text
 your agent (LangGraph)                    what streams to the UI
 ══════════════════════                    ══════════════════════

┌─────────────────────────────┐
│ root graph         ns=main  │
│  ├─ model call ─────────────┼──▶  ✓ streams by default
│  │                          │
│  ├─ model call              │
│  │   tag: non_generation ───┼──▶  ✗ suppressed
│  │                          │
│  └─ subgraph node           │
│     ns=<node_name>          │
│      ├─ model call ─────────┼──▶  ✗ hidden by default
│      │                      │
│      └─ model call          │
│          tag: generation ───┼──▶  ✓ streams
└─────────────────────────────┘
```

Two independent controls:

- **Namespace** (structural — where the call runs): the root graph is
  namespace `main`; every subgraph gets a namespace built from its **node
  name**. The default is asymmetric: `main` is on, subgraphs are off.
- **Tags** (per-call — what you declare): `non_generation` opts a main-graph
  call *out*; `generation` opts a subgraph call *in*.

## What the user sees

The UI renders **one chat flow** — there is no separate sub-agent panel:

```text
┌─ TWIN chat ──────────────────────────────────────────────┐
│  🧑  why is service X degraded?                           │
│                                                           │
│  ⏳ subagent is working…                                  │ ← subgraph with
│                                                           │   no generation
│                                                           │   tag: indicator
│                                                           │   only
│  🤖  Scanning ingest errors in the last hour... ▌         │ ← subgraph call
│                                                           │   tagged
│                                                           │   generation
│  🤖  Service X is degraded because the ingest queue... ▌  │ ← ns = main
└───────────────────────────────────────────────────────────┘
```

- **`main` tokens** stream as the assistant's response. Your final answer
  must come from here.
- **Subgraph tokens** opted in with `generation` stream into the same chat
  flow as they arrive.
- **A subgraph that streams nothing** shows only a *"subagent is working"*
  indicator while it runs — the user sees activity, not content.

Because everything lands in the one chat flow, whatever you tag
`generation` is read by the user as part of the conversation — which is
exactly why answer-shaped content must never carry the tag when your agent
can run under the orchestrator (see
[below](#under-the-twin-orchestrator-your-whole-agent-becomes-a-subgraph)).

## How to tag a model call

Tags are matched as **exact strings** (case-sensitive) against the model
call's run tags. Three ways to attach them:

```python
# 1. Bind on the model — every call through this handle carries the tag
grader = model.with_config(tags=["non_generation"])

# 2. Per call — tag just this invocation
await model.ainvoke(messages, config={"tags": ["generation"]})

# 3. On a whole sub-agent invocation — tags inherit downward, so ONE tag
#    on the invocation applies to every model call inside the child graph
await child_graph.ainvoke(inputs, config={"tags": ["generation"]})
```

### Reserved tags

| Tag | Effect | Use it for |
|---|---|---|
| `generation` | Opts a **subgraph** call *into* streaming | Your **own** nested subgraphs you want visible — never your final answer call (see [Under the orchestrator](#under-the-twin-orchestrator-your-whole-agent-becomes-a-subgraph)) |
| `non_generation` | Opts a call *out* of streaming — wins whenever both tags are present | Routers, graders, guards, summarizers |
| `structured_output` | Treated as internal — suppressed | Structured/JSON extraction calls |
| `ka_synthesis` | Reserved (legacy) | Don't use in new code |

## Only streamed tokens render — your answer must stream

What the user sees is built **entirely from token streaming**. The UI does
not read anything back from your graph's final state (yet — see the status
note above). Two practical consequences:

- **Your final answer must come from a call that streams**: an untagged
  model call in the root graph, or a subgraph call tagged `generation`. If
  the answer-producing call is suppressed (`non_generation`) or sits in an
  untagged subgraph, **the user sees nothing**.
- **Keep internal outputs out of `messages` anyway.** When `messages[-1]`
  parsing lands, the `messages` channel becomes a rendered surface — write
  router/grader output to internal state keys today so nothing internal
  leaks then.

```text
              will my model call show in the UI?
              ──────────────────────────────────

  where does the call run?
     │
     ├─ ROOT GRAPH (ns = main)
     │      │
     │      ├─ tagged non_generation? ─── YES ─▶ ✗ hidden
     │      │
     │      └─ not tagged ──────────────────────▶ ✓ streams in main chat
     │
     └─ SUBGRAPH (ns = node path)
            │
            ├─ tagged generation? ─────── YES ─▶ ✓ streams in the chat flow
            │
            └─ not tagged ──────────────────────▶ ✗ hidden
```

## Under the TWIN orchestrator: your whole agent becomes a subgraph

Your agent runs in two modes, and **"main" is not the same graph in both**:

```text
 STANDALONE — your root graph IS "main"

┌─────────────────────────────────────┐
│ your graph                 ns=main  │
│  ├─ internal call                   │
│  │    tag: non_generation ──▶ ✗     │
│  └─ answer call (untagged) ──▶ ✓ ───┼──▶ streams in the main chat
└─────────────────────────────────────┘

 UNDER THE ORCHESTRATOR (V1) — your whole graph IS a subgraph, muted

┌─────────────────────────────────────┐
│ orchestrator               ns=main  │
│  ├─ planner ──▶ ✓ streams (main)    │
│  ├─ your agent     ns=<your_agent>  │
│  │    non_generation FORCED at      │
│  │    mount by us                   │
│  │   ├─ any call, any tag ──▶ ✗     │
│  │   └─ messages[-1] ──────────┐    │
│  │        (your result)        │    │
│  └─ final synthesis ◀──────────┘    │
│        ──▶ ✓ streams (main chat)    │
└─────────────────────────────────────┘
```

!!! warning "V1 — no intermediary streaming under the orchestrator"
    When the orchestrator delegates to your agent, we mount it with a
    **forced `non_generation` tag** on the whole graph. Nothing inside your
    agent streams — not even calls you tagged `generation` (`non_generation`
    wins whenever both tags are present). The user sees only the
    orchestrator's own activity and its final response. Your result still
    reaches the user because the orchestrator **reads it and restates it**.

This is deliberate: since the orchestrator always writes its own final
synthesis in the main chat, letting your calls stream too would show the
user the same content twice. V1 avoids that structurally by muting
sub-agents entirely. A later release may re-introduce opt-in progress
visibility for selected sub-agents — that flip happens in our mount
configuration, not in your code.

### What this means for you

1. **Your result must be `messages[-1]` — read today under orchestration.**
   Standalone, `messages[-1]` parsing is an upcoming feature (see the status
   note above); under the orchestrator it is **already** the mechanism: the
   delegation tool reads your last message's content as the result the
   planner works from. A tool result or internal note left as the last
   message is what the orchestrator will see — and restate.
2. **Nothing to change for standalone.** Your tags keep working exactly as
   documented; under the orchestrator they are simply inert in V1.
3. **Keep tagging internals `non_generation` anyway.** It's what hides them
   in standalone mode, and it stays correct if progress visibility is
   re-introduced later.

## What else the UI displays

Beyond tokens, two more surfaces exist today:

| Surface | What renders |
|---|---|
| Sub-agent activity | A *"subagent is working"* indicator while a subgraph runs without streaming — activity, not content |
| `messages` state channel | Not rendered today — becomes the final-answer surface (`messages[-1]`) in a later release; already consumed by the orchestrator when your agent runs as its sub-agent |

## Recipes

| I want... | Do this |
|---|---|
| My answer streaming in the main chat bubble | Call the model in your **root graph** — nothing else needed |
| To hide a router / grader / summarizer call | Tag it `non_generation` (and keep its output out of `messages`) |
| A sub-agent visibly "thinking" in the chat flow (standalone) | Tag the calls you want visible `generation` — progress-shaped content only, never the answer |
| A whole sub-agent visible, including intermediate calls | Tag the sub-agent's **invocation** `generation` — it inherits to every call inside |
| A sub-agent that works silently | Do nothing — subgraph calls are hidden by default |
| To hide a JSON-extraction call | Tag it `structured_output` (or `non_generation`) |
| My agent to work standalone **and** under the orchestrator | Answer as the **last message** of your root graph, untagged; internals `non_generation`. Under the orchestrator (V1) all your streaming is muted and `messages[-1]` is what gets used |
