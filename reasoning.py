docs/orchestrator/persona-presets/overview.md
----
# Persona presets: the picture in ASCII

The persona presets design at a glance. The details are in the three contracts:
[franchise](franchise.md), [TwinShield](twinshield.md) and [backend](backend.md).

## 1. Four parties, one artefact each

```text
 Franchise ──tags──▶ Context Hub ◀──reads── TwinShield ──returns──▶ Backend ──calls──▶ Orchestrator
  (author)           (store)                (entitlement)           (BFF)               (loader)
                         ▲                                                                   │
                         └──────────────────── reads ws/repo@version ────────────────────────┘

 Franchise    hands over  tagged commits (dev, staging, production)
 TwinShield   hands over  authorized_personas per user
 Backend      hands over  context.persona = {workspace_id, repo, version}
 Orchestrator loads       workspace_id/repo@version
```

## 2. Franchise: what they publish

```text
<franchise workspace>
├── incident-manager-persona      agent repo, repo tag twin-persona
│   ├── AGENTS.md                 persona instructions
│   └── skills/
│       └── triage/SKILL.md       inline skill (v1)
└── config                        agent repo, the registry
    └── config.json               { personas: [ {name, repo, description,
                                    short_description, visibility{ui:true, orchestrator:false}} ] }

 Release = tag both repos, persona first, then config:

   commit ──▶ dev          (custom tag; untagged: latest commit is used)
          ──▶ staging      (UI "promote to staging")
          ──▶ production   (UI "promote to production")
```

## 3. Environments: which tag runs where

```text
 UI environment     orchestrator called      tags listed by TwinShield
 ──────────────     ───────────────────      ─────────────────────────
 dev   (us)    ───▶ orchestrator-dev    ◀─── dev
 uat   (franch.)─┐
                 ├▶ orchestrator-qual   ◀─── dev + staging   (the stable one)
 qual  (us)   ───┘
 prod          ───▶ orchestrator prod   ◀─── production

 version in the request selects the tag; the orchestrator never refuses a tag
 because of its own environment.
```

## 4. TwinShield: discovery

```text
 discovery(user, env header)
   │
   ├─ tags for env  ──▶  dev | dev+staging | production
   │
   ├─ per authorized workspace, per tag:
   │     pull config@tag ─────────┐  (dev: no tag → latest commit)
   │                              ▼
   │     for each persona in config.json:
   │        roles allow?  ── no ──▶ skip
   │        repo has a commit at tag? ── no ──▶ drop + WARNING
   │                              │ yes
   │                              ▼
   └─ return  { assistants: [... unchanged ...],
               authorized_personas: [ {workspace_id, name, repo, version,
                                       description, short_description} ] }

 never inside assistants: an entry there is a subagent the orchestrator delegates to
```

## 5. Backend: the call

```text
 UI catalog  ◀── authorized_personas (one entry per persona and tag)
     │ user picks "incident-manager (staging)"
     ▼
 POST /threads/<tid>/runs
 {
   "assistant_id": "<orchestrator>",
   "input": {...},
   "context": { "persona": { "workspace_id": "<ws>",
                             "repo": "incident-manager-persona",
                             "version": "staging" } }
 }

 context preferred; configurable accepted; never both in one request (server 400)
 same persona + version on every run of the thread; to switch, new thread
```

## 6. Orchestrator: what happens on a run

```text
 run arrives
   │
   ├─ persona ◀── Runtime.context, else configurable
   │
   ├─ v1: trust the backend (habilitation v2 adds: persona in authorized_personas?)
   │
   ├─ same persona + version as earlier turns? ── no ──▶ turn refused, new thread
   │
   ├─ load ws/repo@version ── fails ──▶ plain orchestrator + notice, retry next turn
   │
   └─ run with <persona> section + persona skills
```

## 7. The deep agent: mounted paths

```text
/                          CompositeBackend   (built per request, routes never change)
├── (default)              StateBackend       thread scratch files
├── skills/
│   ├── generic/           read-only bank     (if added)
│   └── builtin/           read-only bank     (if added)
└── persona/               HubSkillsBackend   shared, one per (workspace, repo, tag)
    │                      or EMPTY backend   when no persona is sent
    ├── AGENTS.md          ◀── Context Hub <ws>/<repo>@<tag>
    └── skills/
        └── triage/SKILL.md

 memory/ and skills/user/ exist only where enabled (off at the client)
```

## 8. Backends and clients in the process

```text
 make_orchestrator(config)          every request, ~30 ms locally
        │ persona = {ws-A, incident-manager-persona, staging}
        ▼
 process-wide cache  (created on first use)
 ┌──────────────────────────────────────────────────────────────────────┐
 │ (ws-A, incident-manager-persona, staging) ─▶ HubSkillsBackend ─▶ snapshot (TTL 300 s)
 │ (ws-A, incident-manager-persona, dev)     ─▶ HubSkillsBackend ─▶ snapshot (TTL 300 s)
 │ (ws-B, crisis-lead-persona,      staging) ─▶ HubSkillsBackend ─▶ snapshot (TTL 300 s)
 └──────────────────────────────────────────────────────────────────────┘
        │ one langsmith Client per workspace (X-Tenant-Id)
        ▼
 one shared HTTP session ──▶ Context Hub endpoint

 why one backend per repo+tag: a ContextHubBackend is bound to one repo,
 with one snapshot cache and one lock
 why no PersonaBackend router in v1: the per-request rebuild already picks the backend
```

## 9. The build, step by step

```text
make_orchestrator(config)                       every request
  1. subject ─▶ TwinShield ─▶ subagents (roster only; personas never become subagents)
  2. persona ─▶ cache[(ws, repo, tag)]  or EMPTY backend
  3. CompositeBackend with /persona/ always mounted
  4. skills sources: generic ─▶ builtin ─▶ /persona/skills/   (last wins on a clash)
  5. middleware: [..., PersonaMiddleware, LoadableSkills(sources)]
        enable_personas on ─▶ both present on EVERY build, persona or not
        (thread reads build the graph without a persona; state schema must match)
  6. create_deep_agent(backend=composite, middleware=..., subagents=...)

PersonaMiddleware                               every run
  before_agent     refuse a persona or version switch on the thread
                   log {ws, repo, version, commit} in state (no pin in v1)
  wrap_model_call  append <persona> AGENTS.md </persona> at the prompt tail
                   or the "persona couldn't be loaded" notice

At run time
  read_file /persona/skills/triage/SKILL.md ─▶ composite ─▶ HubSkillsBackend ─▶ snapshot
```

## 10. Releases

```text
 v1 ────────────▶ v1.1 ────────────────▶ habilitation v2 ─────▶ v2
 inline skills    "persona updated"       persona checked       linked skills
 3 tags           reminder at a turn      against               + pin per thread
 commit logged    boundary                authorized_personas
```

-------

