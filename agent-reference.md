# Orchestrator agent — reference

Internal engineering reference for the **Orchestrator** agent (twin_router v2).

---

## 1. TL;DR

- **What it is:** a planner-driven deep-agent that wraps `deepagents.create_deep_agent`,
  replacing `twin_router`'s hand-rolled ReAct loop. The planner picks targets; specialist subagents do the work.
  v0 is wrapper with many deep agent functionnalities - we will add many layers in the futur ;)
- **Specialist subagents:** `knowledge_agent` (RAG over the source catalog),
  `incident_agent`, `topology_agent`. No first-party policy-gated tools today — general
  knowledge and clarification are answered natively from planner prompt sections and removed completely.
  Clarify could be added back in the future for some specific UX multichoice clarifications.
- **Trust model:** `x-uid` is gateway-trusted; rights, memory namespace, and subagent
  visibility resolve per call. Caller-supplied context carries `persona` only — never rights.
  - *Next steps (§ 5):* signed habilitation token replaces bare-header trust and propagates
    rights to subagents as scoped context/state.
  - *Caveat (today):* an attacker bypassing the gateway can stuff `x-uid` directly.
- **Surfaces beyond routing:** per-user persistent memory · prompt-injection guard · tool-budget enforcement · multimodal `read_picture` fallback · graph cache · planner prompt composition.

---

## 2. At a glance

| Surface | Status | Anchor |
|---|---|---|
| Planner shape (deep-agent) | IN Progress | § 4 |
| Per-call habilitation + user-scoped context | IN Progress | § 5 |
| Subagent registry (`knowledge_agent`, `incident_agent`, `topology_agent`) | IN Progress | § 5 |
| Per-user persistent memory | IN Progress | § 9 |
| Prompt injection guard (pre-agent classifier) | IN Progress | § 8 |
| Tool budget enforcement (per-run cap) | IN Progress | § 7 |
| Multimodal handling (`read_picture` fallback tool) | IN Progress | § 7 |
| Graph cache (5-tuple key partitioned on `has_uid`) | IN Progress | § 4 |
| Auto-registering RemoteGraph subagents | Roadmap | § 6 |
| Subagent `whoswho` discovery (planner-facing capability lookup) | Roadmap | § 6 |
| Per-team / per-app memory | Roadmap | § 9 |
| PII detection · streamed PII redaction | Roadmap | § 8 |
| Signed habilitation token + subagent scope propagation | Roadmap | § 5 |
| Evaluation harness (per-env datasets, complexity tiers, per-model) | Roadmap | § 15 |
| Generic Skills (deepagents primitive, unused today) | Available | § 10 |

> **What's new vs `twin_router`: fully react loop - augmented with native+custom middlewares to be a deep-agent. prompt guardails, Tool Budget guardrails, per-suer memory etc...

---

## 3. Public surface


| Symbol | Shape | Notes |
|---|---|---|
| `make_orchestrator(config)` | `async def make_orchestrator(config: RunnableConfig) -> CompiledStateGraph` | 1-arg factory pinned for `langgraph-api 0.4.x`. Importing the package opens no network — retriever / model construction is lazy on first delegation. |

> **Trust boundary (today):** rights are never on the context. `make_orchestrator`
> resolves them per call from the **gateway-trusted** `x-uid` header. A caller hitting
> the orchestrator through the gateway cannot widen access by stuffing a context
> payload — but bare-header trust is only as strong as the gateway itself: an attacker
> bypassing the gateway can stuff `x-uid` directly.
>
> **Trust boundary (roadmap, § 5):** a signed habilitation token replaces the bare
> header with cryptographic proof, removing the gateway as a single point of trust, and
> propagates rights to subagents as scoped context/state.

---

## 4. Architecture

![Architecture — request flow through the middleware stack to subagents](./orchestrator-request-flow-middleware.png)

Request → factory → resolution → deep-agent compile → middleware stack → planner LLM →
specialist subagents. All resolution is per call; the compiled graph is cached.

### 4.1 Request-to-response flow

```
HTTP request {x-uid, persona, x-request-id, model slots}
   ↓
make_orchestrator(config)                                  1-arg async factory
   │
   ├─ parse_orchestrator_request(config)                   extract headers + persona
   ├─ validate_uid_format(x-uid) → has_uid: bool           falls to anonymous on malformed
   ├─ resolve_orchestrator_habilitation(uid, provider)     permitted_keys + auth_status
   ├─ build_graph_cache_key(permitted_tools,
   │                        permitted_subagents,
   │                        persona,
   │                        model_config,
   │                        has_uid)                       5-tuple cache key
   │
   ├─ CACHE HIT  → return compiled graph
   └─ CACHE MISS → create_deep_agent(model, tools=[], subagents, middleware, system_prompt)
                   │
                   ├─ deepagents auto-injects:
                   │   TodoList · Filesystem · PatchToolCalls · SubAgent
                   │   · Summarization · AnthropicPromptCaching
                   └─ orchestrator stack (ordered):
                       1. PromptInjectionGuard
                       2. MultimodalGuard         (conditionally binds read_picture)
                       3. MessageSequenceNormalizer
                       4. ToolBudgetEnforcement
                       5. GenerationRetry         (wraps the rest)
                       6. LiveMemoryMiddleware    (authenticated path only)
                   ↓
              planner LLM ⇄ task → knowledge_agent / incident_agent / topology_agent
```

### 4.2 Graph cache

The factory holds a process-local cache. Compilation re-runs only when the cache key
changes — covers ~all requests once the factory is warm.

**Cache key — 5-tuple:**

| Slot | Why this dimension |
|---|---|
| `permitted_tools: frozenset[str]` | First-party tool surface (today: empty) |
| `permitted_subagents: frozenset[str]` | Per-role subagent slice |
| `persona: str \| None` | Persona alters the planner system prompt |
| `model_cache_key: ModelConfigCacheKey` | Runtime model overrides change the compiled graph |
| `has_uid: bool` | Partitions anonymous (`StateBackend` only) vs authenticated (`CompositeBackend` + `LiveMemoryMiddleware`) |

**Bypass conditions** (these requests are never cached — banner-bearing prompts must
not leak across users):

- **Degraded habilitation** — an `<auth_status>` banner is injected into the planner prompt.
- **Provider error** — anonymous slice built with the degraded banner.

**Concurrency.** The cache is unlocked. A cold burst can compile twice; deduplicating
under `threading.Lock` would stall the asyncio event loop under the 1-arg sync factory
contract. Compile cost is small; the lock cost would dominate the saving.

**Secret hygiene.** Runtime model configs that carry API keys appear in the cache key as a SHA-256 fingerprint, never the raw secret.

---

## 5. User-scoped context, habilitation & subagent routing

A single per-request resolution step produces a **user-scoped context** that fans out to three downstream consumers: the subagent slice (`task` tool visibility),s, and the memory namespace.

### 5.1 Configurable keys read from the request

`make_orchestrator(config)` reads these from `config["configurable"]`:

| Key | Purpose | Trusted source |
|---|---|---|
| `x-uid` | User identity → memory namespace + role lookup | Gateway header (today) |
| `x-request-id` | Correlation id for logging / tracing | Gateway header |
| `persona` | Persona text into planner system prompt | Caller — not auth-bearing |
| `orchestrator` / `planner` / `all` / `default` | Per-task runtime model overrides | Caller |

Rights derive from `x-uid` only.

### 5.3 Habilitation resolution — provider chain

`resolve_habilitation_provider()` picks one provider, lazily, once per factory boot:

| Order | Env trigger | Provider | Behaviour |
|---|---|---|---|
| 1 | `HABILITATION_BYPASS=1` | `BypassHabilitationProvider` | All requests get `HABILITATION_BYPASS_ROLE` (default `prod`) |
| 2 | `HABILITATION_API_BASE_URL` set | `APIHabilitationProvider` | Calls remote habilitation API with `HABILITATION_API_KEY` |
| 3 | (default) | `MockHabilitationProvider` | Local dev — `non-prod` default, `prod_user` → `prod` |

Provider failure → **fail-closed** to the anonymous slice (empty `permitted_keys`) +
degraded `<auth_status>` banner. No exception propagates to the caller.

### 5.4 Resolution output — the user-scoped context

`HabilitationResolution` is the orchestrator's wrapper over the upstream provider type.

| Shape | Fields | Notes |
|---|---|---|
| `HabilitationResolution` | `permitted_keys: frozenset[str]` · `degraded: bool` · `auth_status: str \| None` | Orchestrator wrapper |
| `ResolvedHabilitation` (upstream) | `user_id` · `role` · `permitted_agents` · `user` · `degraded` | Provider-side primitive |

Three `auth_status` states:

| State | Triggered when | Banner injected |
|---|---|---|
| Normal | Provider returns clean resolution | `None` (no banner) |
| Degraded | Provider returns with `degraded=True` | `"Authorization service degraded. Operating in limited mode."` |
| Unavailable | Provider raises during resolution | `"Authorization service unavailable. Operating in limited mode."` |

### 5.5 Three downstream consumers

The resolved user-scoped context fans out to:

1. **Subagent slice for `task`** — `select_orchestrator_permissions(permitted_keys)`
   intersects `permitted_keys` with each `SUBAGENT_REGISTRY` entry's `permission_keys`.
   Only matching subagents get registered. Anonymous → empty set → no `task` tool at all.
2. **Planner `<auth_status>` banner** — composed into the planner system prompt via
   `build_planner_system_prompt(..., auth_status=...)`. Degraded / unavailable states
   inject the banner; normal state injects nothing.
3. **Memory namespace** — `(uid, "memory")` for the per-user `StoreBackend`. Anonymous
   → no memory middleware at all (structural isolation — see § 9).

### 5.6 Subagent registry — who can route to what

| Subagent | Wraps | `permission_keys` | Today granted to |
|---|---|---|---|
| `knowledge_agent` | KA RAG graph over the source catalog | `("rag", "knowledge_agent")` | `prod` / `prd` |
| `whoswho` | Who is the biggest bogoss around? | `("whoswho", "whoswho_agent")` | `prod` / `prd` |
| `incident_agent` | Incident graph | `("incident", "incident_agent")` | `prod` / `prd` |
| `topology_agent` | Wraps `navigator_agent` | `("topology", "topology_agent")` | `prod` / `prd` |

`SUBAGENT_REGISTRY` is the single source of truth. Anonymous and `non-prod` roles get
the empty intersection → no subagents → no `task` tool.

### 5.7 Subagent state isolation

Deepagents' `_EXCLUDED_STATE_KEYS` filter strips these state keys both ways between
parent and subagent: `memory_contents`, `messages`, `todos`, `structured_response`,
`skills_metadata`, `skills_load_errors`. Parents see only the result; subagents see
only the task input.

> **Read-side caveat.** Deepagents' auto-injected `general-purpose` subagent has
> filesystem tools. The state filter blocks write contamination, but a GP
> `read_file("/memory/...")` returns memory content in a `ToolMessage` the parent
> appends to its transcript — treat memory as observable by any GP delegation in the
> same turn until a `FilesystemPermission` deny rule on `/memory/**` lands.

### 5.8 Trust model — roadmap

Today: `x-uid` is trusted because the gateway sets it. If an attacker bypasses the
gateway, they can stuff `x-uid` directly; the orchestrator cannot detect it.

**Next step — signed habilitation token.** Replace the bare `x-uid` header with a
short-lived signed token (`{uid, rights, expiry, issuer}`). The orchestrator verifies
signature + expiry before resolving rights. Gateway compromise becomes detectable;
bypass attacks fail signature verification.

**Step after — subagent scope propagation.** Forward the verified token (or a narrowed
delegation token) into subagents as scoped context/state. Today subagents inherit
parent permissions implicitly via the registered set — they have no identity to attach
to downstream calls (DB queries, audit logs, remote services).

**Open design questions:**

- Subagent narrowing — full-rights pass-through vs least-privilege re-issue
- Verification failure — anonymous fall-soft vs hard 401

![Today vs OBO scope propagation — subagent / backend enforcement](./orchestrator-today-vs-obo-scope.png)

---

## 6. Extending the subagent registry

The subagent surface is the main extension point. Today's registration is direct;
two roadmap items will extend it — remote subagents over the network, and dynamic
capability discovery for the planner.

### 6.1 Adding a subagent today — recipe

Four steps. ~10 lines of registration code.

1. **Add a registry entry** in `registry.py` with `permission_keys` mapping to the
   habilitation roles that should see it.
2. **Build the inner graph** — return a `CompiledStateGraph`. Any LangGraph graph
   that accepts `{"messages": [...]}` input works without a wrapper.
3. **Wrap as `CompiledSubAgent`** with `.with_config(recursion_limit=N)` bound on the
   **inner** runnable. Deepagents' `task` tool strips `recursion_limit` from parent
   config; only the inner bind survives.
4. **Update habilitation policy** in `habilitation/policies.py` if introducing a fresh
   role gate. Otherwise reuse an existing `permission_keys` slot.

Canonical reference (KA-style registration):

```python
def build_knowledge_agent_subagent(entries, review_cap: int = 2) -> CompiledSubAgent:
    config = KnowledgeAgentConfig(
        mode="answer",
        search_depth="deep",
        max_iterations=review_cap,
        subagent_mode=True,
    )
    inner = create_knowledge_agent(entries, config=config)
    return CompiledSubAgent(
        name="knowledge_agent",
        description=KNOWLEDGE_AGENT_DESCRIPTION,
        runnable=inner.with_config(recursion_limit=review_cap * 10 + 20),
    )
```

> **State-filter note.** Subagents inherit zero parent state from the
> `_EXCLUDED_STATE_KEYS` list (messages, todos, memory, skills metadata). Anything you
> want the subagent to see must travel through the `task` description string the
> planner writes.

### 6.2 Roadmap — auto-registering `RemoteGraph` subagents

Today, every subagent is local-compiled and statically registered at factory build
time. Vision: subagents deployed to a separate LangGraph Server expose a registry
endpoint; the orchestrator auto-discovers them and wraps as `RemoteGraph` instances
that satisfy the same `CompiledSubAgent.runnable` contract. Same habilitation gates
apply per role.

**Prerequisite in place.** Unified LangSmith tracing across `/runs/wait` — without it,
every outer → inner call produces orphan trace trees. The `langgraph-api` server side
now reads `langsmith-trace` from configurable and wraps the worker in a tracing
context, so remote subagents will trace as one tree.

**Open design questions:**

- Registry transport — file manifest · HTTP endpoint · service discovery (Consul / DNS-SRV)
- Refresh cadence — boot-only · TTL poll · push-on-change
- Failure mode on remote-down — fail-soft to local fallback vs hard-fail
- Per-call auth header forwarding — bare `x-uid` (today) → signed token (§ 5.8)
- Cost / latency attribution — separate metrics namespace per remote subagent
---

## 7. Middleware stack walkthrough

`compose_orchestrator_middleware(...)` returns the orchestrator's middleware list;
`make_orchestrator` appends `LiveMemoryMiddleware` to it on the authenticated path.
Order is load-bearing.

### 7.1 The 6 middlewares in order

| # | Middleware | Source | Role |
|---|---|---|---|
| 1 | `PromptInjectionGuardMiddleware` | `orchestrator/middlewares/` | Pre-agent classifier; can `jump_to="end"` with a deterministic refusal. See § 8. |
| 2 | `MultimodalGuardMiddleware` | `base/middlewares/` | Strip images early; expose `read_picture` only when the planner model is not multimodal. |
| 3 | `MessageSequenceNormalizerMiddleware` | `base/middlewares/` | Pre-model — repair orphan tool messages before the model sees them. |
| 4 | `ToolBudgetEnforcementMiddleware` | `orchestrator/middlewares/` | Optional per-run tool-call cap (§ 7.3). |
| 5 | `GenerationRetryMiddleware` | `base/middlewares/` | Exception-path retry + empty-response retry. Tool-call-only messages pass through. `model_not_found` swaps to fallback immediately. **Last — wraps the rest.** |
| 6 | `LiveMemoryMiddleware` | `orchestrator/middlewares/live_memory.py` | Authenticated path only — appended after the compose call. See § 9. |

> The prompt-injection guard uses both sync and async `before_agent` hooks and must
> keep `@hook_config(can_jump_to=["end"])`. Without that decorator, LangChain ignores
> `jump_to`.

### 7.2 Multimodal handling

`MultimodalGuardMiddleware` does two things:

- **Image stripping for text-only planner models** — drops image content parts on the
  way into the model so the chat completion API doesn't choke.
- **Conditional `read_picture` tool exposure** — binds `read_picture` into the tool
  list only when the resolved planner model is not multimodal per
  `sta_agent_engine.models.capabilities.is_multimodal()`. The tool forwards visible
  conversation context and recent image parts to the
  `ORCHESTRATOR_PICTURE_READER_*` server-configured multimodal model.

`read_picture` is **middleware-owned**, not a `TOOL_REGISTRY` entry. Habilitation
does not gate it — the planner model's multimodal capability decides.

> **Caveat for future dynamic-model wiring.** Don't combine `DynamicModelMiddleware`
> with the conditional `read_picture` binding without redesign — `read_picture` is
> bound at graph construction time, while dynamic model switching happens at runtime
> inside `wrap_model_call`.

### 7.3 Tool budget enforcement

Optional global per-run cap on tool calls. Disabled when
`ORCHESTRATOR_TOOL_BUDGET_GUARD_MAX_TOOL_CALLS` is unset. Once the cap is reached,
the next planner call is forced to `tools=[]` (answer-now mode).

**Counting rules:**

- Counts main planner / deep-agent tool calls, including `task` delegations.
- Does **not** count the prompt-injection classifier call or the answer-now no-tool
  model call.

**Trust boundary.** Settings are server-owned via env. A `ToolBudgetGuardSettings`
instance passed programmatically to `create_orchestrator_factory(...)` takes
precedence. Request `configurable` values cannot install the guard or raise the cap.

### 7.4 Deepagents auto-injected middlewares

Not in the orchestrator list — `create_deep_agent` adds these automatically:

| Auto-injected | What it does |
|---|---|
| `TodoListMiddleware` | Exposes `write_todos` tool + state. |
| `FilesystemMiddleware` | Exposes `ls` / `read_file` / `write_file` / `edit_file` over the configured `backend`. |
| `PatchToolCallsMiddleware` | Repairs malformed tool-call IDs from some providers. |
| `SubAgentMiddleware` | Builds the `task` tool from registered subagents. Added when subagents non-empty. |
| `create_summarization_middleware` | Summarizes long histories when the token limit is approached. |
| `AnthropicPromptCachingMiddleware` | Marks reusable prompt prefixes for cache hits on Claude models. |

---

## 8. Guardrails

The orchestrator owns an evolving guardrails surface. Today it's a prompt-injection
classifier; PII detection and streaming-time redaction are on the near roadmap.

### 8.1 Prompt injection guard — IN Progress

Pre-agent classifier short-circuits the request with a deterministic refusal on
detected injection.

| Aspect | Today |
|---|---|
| Position in stack | Middleware slot #1 (before any model / tool work) |
| Refusal mechanism | `jump_to="end"` (requires `@hook_config(can_jump_to=["end"])`) |
| Env config | `ORCHESTRATOR_PROMPT_INJECTION_GUARD_*` family — server-owned |
| Fail policy | Fail-open by default. Flip with `ORCHESTRATOR_PROMPT_INJECTION_GUARD_FAIL_OPEN=false` |
| Programmatic override | `PromptInjectionGuardSettings` passed to `create_orchestrator_factory(...)` beats env |
| Request override? | **No.** Caller cannot disable own filter — trust boundary |
| History handling | Flagged-injection content is redacted from the conversation history before the planner sees it |

Classifier internals evolve faster than this reference; see the middleware module
for the live implementation.

### 8.2 Roadmap — PII detection & content guardrails

**Goal.** Detect and redact PII in user input and model output before logging or
returning over the wire. Same slot can later host broader content-policy guardrails
(toxicity, jailbreak resistance beyond § 8.1).

**Off-the-shelf candidates worth evaluating:**

- **NVIDIA NeMo Guardrails** — programmable rails for input / output / topic / dialog
- **AI Guardrails** — composable guardrail pipelines
- **Microsoft Presidio** — PII detection & anonymization with named-entity recognition

> **Big caveat — infra commitment, not just a middleware drop-in.** All three
> frameworks rely on specific models under the hood (NER taggers, transformer
> classifiers, spaCy pipelines). Adopting one means downloading those models, hosting
> them somewhere with the right CPU / GPU budget, and routing every request through
> the inference layer. The middleware code is the easy part; the model lifecycle
> (versioning, A/B, latency budget) is the real cost. The build-vs-buy decision must
> include that infra cost.

**Open decisions:** framework (NeMo / AI Guardrails / Presidio / custom) · scope (input only / input+output) · action (block / redact / warn) · severity tiering · self-host vs managed inference.

### 8.3 Roadmap — streamed PII redaction

**Goal.** Redact PII inside streamed tokens before they reach the caller's wire — not
just on the final assembled response. Streaming amplifies every challenge in § 8.2:
patterns can span chunk boundaries, released tokens can't be recalled, and buffer
windows trade latency for miss-rate.

### 8.4 Roadmap shape — at a glance

```
IN Progress → Prompt injection guard (classifier + history redaction)
  ↓
Planned → PII detection + content guardrails (framework + model hosting decision)
  ↓
Planned → Streamed PII redaction (chunk-boundary safe)
```

---

## 9. Per-user persistent memory

![Memory lifecycle across two turns — load, edit, refresh](./orchestrator-live-memory-lifecycle.png)

Each authenticated user gets two memory files that survive across sessions:

- `/memory/AGENTS.md` — user-authored persona / preferences
- `/memory/preferences.md` — agent-curated notes

`LiveMemoryMiddleware` (subclass of deepagents' `MemoryMiddleware`) is wired only
on the authenticated path. **Cost model: 1 initial load + 1 read per edit; 0 reads
on idle turns.**

### 9.1 Backend shape — anonymous vs authenticated

`build_orchestrator_backend(*, has_uid: bool)` returns one INSTANCE per `has_uid`
cache class. The same instance is shared by `FilesystemMiddleware` (driving the
deepagents file tools) and `LiveMemoryMiddleware` (driving the memory sync).

| `has_uid` | Backend shape | Memory writes |
|---|---|---|
| `False` (anonymous) | bare `StateBackend` | Ephemeral — Store NEVER touched |
| `True` (authenticated) | `CompositeBackend(default=StateBackend(), routes={"/memory/": StoreBackend(namespace=resolve_memory_namespace)})` | `/memory/*` writes persist; other paths stay ephemeral |

> **Structural isolation, not runtime branching.** Anonymous callers get no memory
> middleware **and** no `StoreBackend` access path. The Store object is unreachable
> through any tool call.

### 9.2 Namespace resolution — `(uid, "memory")`

Each call resolves a namespace tuple from the gateway-trusted `x-uid` and routes the
memory backend through it. One shared backend instance serves all users — per-user
isolation lives in the namespace tuple, not in separate instances.

`x-uid` is validated at two boundaries: at the catalog (malformed → falls to anonymous
path with a length-only warning log; never log the value, attacker-controlled payload
risk) and again inside the namespace resolver as defense-in-depth if a future code
path bypasses the catalog.

> **Heads-up — namespace shape will evolve.** Today it's a 2-tuple `(uid, "memory")`.
> The per-team / per-app extension in § 9.4 will likely reshape it to something like
> `(tenant_id, "user_or_team", uid, "memory")`. Any caller code or downstream tooling
> that hard-codes the 2-tuple shape will need to migrate when that lands.

### 9.3 Fail policies

| Failure | Behaviour |
|---|---|
| Transient backend error during load | **Soft** — degrade to empty memory + warning log; user request survives |
| Deployment misconfig (no persistence backend attached) | **Loud** — fails fast so operators see it, not silently masked |
| Malformed `x-uid` | Fall to anonymous path + length-only warning log |

### 9.4 Roadmap — per-team / per-app memory

Today's namespace is a single dimension (user). The roadmap extends scope:

| Namespace shape | Meaning |
|---|---|
| `(team_id, "memory")` | Shared within a team |
| `(app_id, "memory")` | Shared within an application / persona context |
| `(uid, team_id, "memory")` | Composite per-user-per-team |

**Implementation sketches:**

- Extend `resolve_memory_namespace` to read additional scope keys from `configurable`.
- Or add `CompositeBackend` routes (`/memory/team/...`, `/memory/app/...`) alongside
  the existing `/memory/` per-user route.

**Open design questions:**

- Priority order — user → team → app, or richest scope wins?
- Conflict resolution on duplicate keys across scopes
- Write authority — who can edit team memory? (rights table extension)
- Auth boundary — gateway must trust `team_id` / `app_id` the same way it trusts `x-uid`
- Cache-key impact — per-team partition would add a 6th tuple element to § 4.2


---

## 10. Composition primitives — available but not wired

Deepagents ships extension primitives the orchestrator could opt into. Today's
configuration does not use them. Documented so the next maintainer doesn't re-invent
what already exists.

### 10.1 Skills (deepagents primitive)

`SkillsMiddleware` is a deepagents middleware that loads pluggable capability
modules from configured `sources=[...]` paths. Each skill is a directory containing
a `SKILL.md` with YAML frontmatter (name, description, allowed_tools, license).

**Status in orchestrator:** the factory calls `create_deep_agent(..., sources=None)`
(implicit). `SkillsMiddleware` is **not** auto-attached. The `skills_metadata` and
`skills_load_errors` state fields are still filtered by the subagent state filter
(defensive hygiene), but no skills are loaded today.

**If you wanted to enable it:** pass `sources=[...]` to `create_deep_agent` inside
the factory body, scoped to the resolved `permitted_keys`. Bring the same trust
boundary logic — skills loaded from per-user paths inherit the gateway-trust caveat
from § 5.8.

### 10.2 Other primitives on the shelf

| Primitive | What it does | Why not used today |
|---|---|---|
| `DynamicModelMiddleware` | Runtime model swap per request | Conflicts with the current `read_picture` static binding (§ 7.2) |
| `DynamicToolMiddleware` | Runtime tool surface mutation | First-party tool surface is empty — nothing to mutate |
| Stock `MemoryMiddleware` | Standard per-user memory | Replaced by `LiveMemoryMiddleware` to fix the checkpoint-staleness short-circuit |

---

## 11. Wiring matrix — deployment shapes

| Deployment | `store=` | `checkpointer=` | Notes |
|---|---|---|---|
| LangGraph Platform | Leave `None` — platform attaches managed Postgres | Leave `None` — platform attaches | Default production target |
| Embedded host (FastAPI mount, app process) | Pass `InMemoryStore()` per session | Pass `InMemorySaver()` per session | Restarting the host clears state |
| Standalone (tests, scripts, ad-hoc) | Pass `InMemoryStore()` or your prod store | Pass `InMemorySaver()` or your prod checkpointer | Forgetting the store fails loud on first memory read (no silent degradation) |

Reference standalone wiring lives in `examples/sta_agent_engine/`.

---

## 12. Key design decisions

| Choice | Picked | Over | Reason |
|---|---|---|---|
| Agent shape | Wrap `create_deep_agent` | Hand-rolled ReAct loop | Free middlewares (TodoList, SubAgent, Caching) |
| Rights resolution | Per-call from `x-uid` via provider | On `OrchestratorContext` | Context is caller-supplied — spoofable |
| Subagent registration | Direct `CompiledSubAgent` | Wrapper graph | KA input shape matches `{messages}`; cleaner stack trace |
| Factory signature | 1-arg `make_orchestrator(config)` | 2-arg `(config, runtime)` | `langgraph-api 0.4.x` compatibility |
| Cache key | Partitioned on `has_uid` | One entry per uid | Avoids cache explosion; per-uid isolation lives in the namespace callable |
| Memory wiring | Subclass + manual middleware list | `memory=[...]` shortcut | Shortcut double-attaches stock middleware on top |
| Malformed `x-uid` | Catalog fall-soft to anonymous + log | Hard-fail at middleware | Gateway bug shouldn't kill the request |
| Memory fail policy | Soft on transient errors, loud on deployment misconfig | All-soft or all-loud | Transient = degrade; missing-store = deploy bug, must surface |

---

## 13. Operations & observability

**What to watch:**

- **Gateway header corruption** — malformed `x-uid` falls to anonymous path and logs a length-only warning; symptom is unexpected anonymous behaviour for an otherwise-authenticated caller.
- **Memory degraded mode** — transient backend issue leaves the user request alive but with empty memory; log line surfaces the underlying error.
- **Auth degraded / unavailable banner** — visible in the planner prompt in the trace, distinguishing provider-degraded from provider-unavailable.

**Tracing:** every orchestrator run is a LangSmith trace; subagent `task` calls are nested spans. Remote subagents (§ 6.2) unify into one trace via `langsmith-trace` header propagation across `/runs/wait`.

---

## 14. Not yet in scope

Cross-index of deferred work; each item is owned by the section indicated:

- Signed habilitation token + subagent scope propagation (§ 5.8)
- Auto-registering `RemoteGraph` subagents (§ 6.2)
- PII detection · streamed PII redaction (§§ 8.2-8.3)
- Per-team / per-app memory namespaces (§ 9.4)
- `FilesystemPermission` deny rule on `/memory/**` for the GP subagent (§ 5.7)
- Multi-source memory configurable (today: 2 files)
- CAS for concurrent memory writes (today: last-write-wins)
- Evaluation harness (§ 15)

---

## 15. Evaluation — roadmap

The orchestrator does not ship a dedicated evaluation harness today. Three intertwined
needs shape the roadmap.

### 15.1 Per-environment datasets

Production, pre-prod, and dev see different data shapes. The same query against
different environments returns different evidence. **Implication:** a single dataset
can't validate the orchestrator across environments — the harness needs per-env
dataset partitioning with environment-correct expected outputs.

### 15.2 Per-model datasets

Models differ in tool-use propensity, planning depth, and error modes. A dataset that
catches regressions on a smaller model can miss them on a larger one (and vice versa).
**Implication:** dataset rows annotated with the model they were sampled against, plus
a way to slice eval runs by model.

### 15.3 Complexity tiers

Agent regressions are not uniform across query complexity. **Implication:** tier
dataset rows (e.g. *single-shot answer*, *single-subagent delegation*, *multi-subagent
planning*, *clarification loop*, *gap response*) and report per-tier pass rates
separately. A drop on the planning tier matters more than on single-shot.

### 15.4 Open design questions

- Multienv dataset partitioning
- Authority — who owns dataset curation (engineering · ops · product)
