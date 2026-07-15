docs/consuming/external-agent-cards.md
----
# Publishing an agent to the TWIN orchestrator

This guide is for **partner teams who build and deploy their own agent** and want
it to show up for TWIN users — either callable **directly from the UI**, or routed
to **by the orchestrator's planner** like a first-party subagent. It's the inverse
of the other pages in this section: there you *consume* our services; here you make
*your* agent consumable.

You do this by publishing a **capability card**: a small, structured description of
what your agent does and where it may be exposed. You author it once, version it
next to your graph, and the CLI compiles it into your deploy config.

!!! info "Status — card format is stable; orchestrator routing is a future feature"
    The **card format is the stable contract** — author against it now and it will
    not change out from under you. Making your agent **UI-visible works today**.
    **Orchestrator routing** (the planner auto-discovering deployments and calling
    your agent) ships in a later release. Setting `visibility.orchestrator: true`
    today *declares intent*; it does not by itself make the planner call your agent.

## Two decisions before you start

### 1. Where should your agent appear? (`visibility`)

Your agent is exposed **nowhere** until you opt in. There is no "hide" flag — you
simply choose which surfaces to turn on. The two are independent:

```yaml
visibility:
  ui: true            # end users can pick your agent directly in the UI
  orchestrator: false # the TWIN planner may route to it as a subagent (future)
```

- **UI-visible** (`ui: true`) — your agent is listed as a standalone choice in the
  UI. **If you set this, write a strong `short_description`** — that one-liner is
  the label users read to decide whether to pick your agent. A vague one gets
  ignored; a precise one ("Search & explain application logs") gets used.
- **Orchestrator-visible** (`orchestrator: true`) — the planner may delegate to
  your agent automatically as part of answering a broader question. **This is a
  future capability** (see the status note above); set it now to declare intent,
  but routing goes live in a later release.

Both default to `false`. Turning a surface on is a *request to be considered* — our
side still ANDs it with our own registry and admission checks, so it never
auto-grants exposure. The catalog key your agent is known by is assigned by us,
never read from your card, so a card can't shadow a first-party agent.

### 2. Will the orchestrator route to it *well*? (card metadata)

For the UI, `short_description` is what matters. **For orchestrator routing, the
card's metadata is the entire signal** — the planner has nothing else to go on when
deciding whether your agent fits a request. Invest in these fields:

| Field | Why the planner needs it |
|---|---|
| `description` | The primary routing signal. Write **both** what the agent does **and when to delegate to it**. |
| `scope` | The exact domain boundary, so the planner knows your agent's precise coverage and doesn't over-route. |
| `how_to_use` | Prompting guidance, so the planner phrases queries your agent handles well. |
| `examples` | 1–3 verbatim sample queries — concrete anchors the planner matches against. |

A thin card doesn't just route poorly — an oversized or malformed one **degrades**
to a "self-reported / unverified" thin entry rather than winning routing share it
didn't earn. Run `sta agent-profile validate` to catch gaps at build time.

!!! tip "Card validators are coming"
    Today `validate` applies structural checks (schema, size caps) and completeness
    heuristics. A future update adds richer **card validators** — and later an
    admission gate that screens descriptions for over-claiming and
    instruction-injection before an agent is admitted to the planner's roster.

## Quickstart — I have an agent, create its card and Docker JSON

Say your graph is importable at `./log_agent.py:graph`. Five steps, no hand-escaping.

### 1. Scaffold a card next to your graph

```bash
sta agent-profile example --yaml > log_agent.card.yaml
```

This is the **manifest** form (the default) — your card plus its deployment
identity (name + import path) in one self-contained file you keep beside
`graph.py` in source control. It ships annotated, with block scalars for the prose
fields (abridged here; the real output comments every field):

```yaml
# Your graph key — the same one you use in langgraph.json's `graphs` block.
log_agent:
  # Import path to your compiled graph.
  path: ./log_agent.py:graph
  card:
    # What the agent does AND when to delegate to it — the planner's primary signal.
    # `>-` folds line breaks into spaces; a blank line becomes a paragraph break.
    description: >-
      Searches application logs and answers questions about them.
      Delegate when a user asks why an application failed, or wants
      errors for a service in a given time window.

      Not for metrics or dashboards — that's a different agent's job.

    # One-liner users read in the UI when `visibility.ui` is true.
    short_description: App log search

    scope: Application logs for the X business infra
    freshness: every 15 minutes

    # `|-` keeps line breaks verbatim — use it for steps or bullets.
    how_to_use: |-
      1. Give an application code (e.g. APP123).
      2. Give a time window.
      3. Optionally pass a log level to filter on.

    examples:
      - errors on app APP123 in the last hour
      - why did the payment service fail yesterday

    # Both default false — you are exposed nowhere until you ask.
    visibility:
      orchestrator: true
      ui: false

    tags: [logs, observability, sre]
```

See [Multi-line prose](#multi-line-prose) for the block-scalar rules. It validates
clean as-is — `sta agent-profile validate` on the untouched scaffold reports no
completeness suggestions.

### 2. Fill it in

Point `path` at your graph, and write the card against the two decisions above —
`short_description` if UI-visible, rich `description` / `scope` / `how_to_use` /
`examples` if orchestrator-visible. Commit the file next to your graph.

### 3. Validate

```bash
sta agent-profile validate log_agent.card.yaml   # add --strict to fail CI on gaps
```

```
✓ log_agent: card is valid
  no completeness suggestions — looks great.
```

### 4. Generate your deploy config — pick the one that matches how you deploy

No `--name` / `--path` flags needed: the manifest already carries them.

**a) You use `langgraph.json`** (standard LangGraph deploy):

```bash
sta agent-profile build --langgraph-json log_agent.card.yaml
```

```json
{
  "log_agent": {
    "path": "./log_agent.py:graph",
    "description": "{\"description\":\"Searches application logs ...\",\"visibility\":{\"orchestrator\":true,\"ui\":false},\"tags\":[\"logs\",\"observability\",\"sre\"]}"
  }
}
```

Merge that under `"graphs"` in your `langgraph.json`. Done.

**b) You run your own Docker image** (no `langgraph build`):

```bash
sta agent-profile build --langserve-env log_agent.card.yaml
```

```dockerfile
ENV LANGSERVE_GRAPHS='{"log_agent": {"path": "./log_agent.py:graph", "description": "{\"description\":\"Searches application logs ...\",\"visibility\":{\"orchestrator\":true,\"ui\":false}}"}}'
```

Paste that `ENV` line into your Dockerfile. See
[Deploying with your own Docker image](#deploying-with-your-own-docker-image-no-langgraph-build)
for why this works.

**c) Skip the copy-paste — write straight into your config with `--into`:**

```bash
sta agent-profile build log_agent.card.yaml --into ./langgraph.json   # merges the graphs block
sta agent-profile build log_agent.card.yaml --into ./Dockerfile       # merges the ENV LANGSERVE_GRAPHS line
```

`--into` infers the destination from the filename and **merges** — graphs already
in the file that your card doesn't mention are **kept**, and re-running is
idempotent. It prints a delta (`added` / `overwritten` / `preserved`) so the write
is never a surprise. Useful flags:

- `--dry-run` — print the merged result and delta without writing.
- `--replace` — make your manifest authoritative: drop graphs in the destination
  that aren't in it (default is preserve-and-merge).
- `--create` — create the destination file if it doesn't exist yet.
- `--as langgraph-json|dockerfile` — force the kind when the filename is ambiguous.

### 5. Deploy and verify

Deploy as usual, then confirm the card the orchestrator will read:

```bash
langgraph dev
curl "localhost:2024/a2a/<assistant_id>/.well-known/agent-card.json"
```

The `description` you see there is your compiled card — that's exactly what the
planner ingests.

## The card contract

### Authoring shapes: manifest, flat card, or root manifest

All three are accepted and auto-detected — author whichever fits:

| Shape | Looks like | Best for |
|---|---|---|
| **Manifest** (default, recommended) | `graph key → {path, card}` | One self-contained file next to `graph.py`, versioned. `build` needs no flags. Multiple agents can share one file. |
| **Flat card** | just the profile fields | Piping a bare card, or when name/path live elsewhere. Pass `--name` / `--path` at build time. |
| **Root manifest** | a list of card-file paths | Bundling many agents whose cards each live next to their own `graph.py`, without inlining them into one file. |

```bash
sta agent-profile example --yaml         # manifest (default; name/path in the file)
sta agent-profile example --flat --yaml  # flat card (name/path via CLI flags)
```

**A manifest can hold several agents** — each top-level key is one graph. `build`
merges them into a single `langgraph.json` graphs map (or one `LANGSERVE_GRAPHS`
value); `validate` reports on each in turn:

```yaml
log_agent:
  path: ./log_agent.py:graph
  card:
    description: Searches application logs and answers questions about them.
    visibility: { orchestrator: true, ui: false }
metric_agent:
  path: ./metric_agent.py:graph
  card:
    description: Answers questions about metrics and dashboards.
    visibility: { orchestrator: true, ui: false }
```

**Or keep each card in its own file and bundle them with a root manifest** — a
list of paths (resolved relative to the root file). Each referenced file is a
self-contained manifest (the graph key + `path` live there, so the root file adds
nothing per-agent):

```yaml
# agent_profile.yaml  (a bare list works too)
cards:
  - ./log_agent/log_agent.card.yaml
  - ./metric_agent/metric_agent.card.yaml
```

```bash
sta agent-profile build agent_profile.yaml --into ./langgraph.json
```

`build` / `validate` treat the bundle as one set of agents (a duplicate graph key
across files is an error).

You can author in **JSON or YAML** either way. YAML is handy for the prose fields
(block scalars) and `#` comments; the CLI detects format by extension (`.yaml` /
`.yml`), or pass `--format yaml` on stdin. The **wire format is always JSON** — the
A2A card carries a JSON string; YAML is purely an authoring convenience.

### Multi-line prose

The prose fields are long. Use YAML **block scalars** rather than one giant line:

```yaml
card:
  # `>-` FOLDED: line breaks become spaces. Best for description — it reads as
  # one paragraph. A blank line becomes a real paragraph break.
  description: >-
    Searches application logs and answers questions about them.
    Delegate when the user asks why an app failed, or wants errors
    in a time window.

    Not for metrics or dashboards — that's the metric agent's job.

  # `|-` LITERAL: line breaks are kept verbatim. Best for steps or bullets.
  how_to_use: |-
    1. Give an application code (e.g. APP123).
    2. Give a time window.
    3. Optionally pass a log level.
```

Prefer the `-` chomping suffix (`>-`, `|-`) — plain `>` / `|` keep a stray trailing
newline. Content must be indented further than its key.

Nothing needs hand-escaping: newlines are encoded as `\n` in the JSON wire format,
so the Dockerfile `ENV` line stays a single physical line, and apostrophes are
handled (see the note below). Write prose naturally.

### Profile fields

| Field | Type | Required | Purpose |
|---|---|---|---|
| `description` | string | **yes** | What the agent does **and when to delegate to it** — the primary routing signal. |
| `short_description` | string | no | One-liner shown in the UI when the agent is UI-visible. Not used for routing. |
| `scope` | string | no | The exact domain boundary, e.g. `"Application logs for the X business infra"`. |
| `how_to_use` | string | no | Prompting guidance so the planner queries your agent effectively. |
| `examples` | string[] | no | 1–3 verbatim sample queries your agent handles well. |
| `freshness` | string | no | Free text — how current the data is, e.g. `"real-time"`, `"every 15 minutes"`. |
| `visibility` | object | no | `{orchestrator, ui}` — which surfaces you opt into. Both default `false`. |
| `tags` | string[] | no | Descriptive labels for discovery/search only. Never access control, never shown to the planner. |

Every field is size-capped; oversized or malformed cards degrade rather than crash
routing.

### How a card reaches the orchestrator

Your agent is deployed as its own LangGraph deployment and invoked as a
`RemoteGraph`. On stock `langgraph-api`, the **only** producer-controllable field
that reaches your agent's A2A card (`/.well-known/agent-card.json`) is the graph's
`description` string. So the card travels as a **JSON object stringified into that
`description`**:

```
your langgraph.json  ──deploy──▶  A2A agent card  ──read──▶  TWIN planner roster
  graph.description                 .description                 subagent entry
  = "<card JSON>"                   = "<card JSON>"              (routing signal)
```

Because it's JSON-inside-a-JSON-string, hand-escaping is error-prone — that's the
whole reason to compile it with `sta agent-profile build` rather than write it by
hand.

## Deploying with your own Docker image (no `langgraph build`)

The A2A card is a **runtime** feature of the langgraph-api server, not a build
artifact. On startup the server reads its graphs — including each graph's
`description` — from the **`LANGSERVE_GRAPHS`** environment variable (which
`langgraph build` populates from your `langgraph.json`) and serves the card on
demand. So if you run your own image, you just set that env yourself:

```bash
sta agent-profile build --langserve-env log_agent.card.yaml
```

The emitted `ENV LANGSERVE_GRAPHS='...'` line is byte-compatible with what
`langgraph build` bakes in. Add it to your own Dockerfile (on top of the
`langchain/langgraph-api` server base, or your adaptation of the Dockerfile from
`langgraph dockerfile`). If your deployment serves several graphs, put them all in
one manifest — `build` merges them into a single `LANGSERVE_GRAPHS` map.

Already have a Dockerfile with a `LANGSERVE_GRAPHS` line? Merge into it directly:

```bash
sta agent-profile build my_agents.yaml --langserve-env --into ./Dockerfile
```

Graphs already in that line that your manifest doesn't mention are **preserved**
(`--replace` if you instead want the manifest to be the complete set). See
[step 4c](#4-generate-your-deploy-config-pick-the-one-that-matches-how-you-deploy).

!!! note "Apostrophes are handled for you"
    The value is single-quoted, and BuildKit does no escape processing inside single
    quotes — so a literal `'` (ordinary prose: `the agent's logs`) would terminate
    the string and break `docker build`. Because the value is JSON, the CLI encodes
    any apostrophe as the `'` escape: still valid JSON, and the server decodes
    it back to `'`. Write prose naturally; nothing to work around. (`langgraph build`
    does *not* do this and emits an unbuildable Dockerfile for such a card — the one
    place this CLI's output intentionally differs.)

A production server run also needs `DATABASE_URI` (Postgres) and `REDIS_URI` —
that's the persistence/queue the platform image expects, independent of the card.
To just verify the card locally with no Docker at all: `langgraph dev`, then
`curl "localhost:2024/a2a/<assistant_id>/.well-known/agent-card.json"`.

## Inspecting the schema

```bash
sta agent-profile schema          # the JSON Schema for the card
sta agent-profile example         # a filled-in starter manifest (default)
sta agent-profile example --flat  # a filled-in bare card
```

## What we do with your card

- **Parse and validate** it defensively — a bad card degrades, it never crashes
  routing.
- **Render only the routing-relevant fields** (`description`, `scope`, `examples`,
  `how_to_use`, `freshness`) into the planner. `visibility` and `tags` drive
  exposure/discovery and are never shown to the planner.
- **Treat every field as self-reported and unverified.** Scope claims carry a
  caveat to the planner; a forthcoming admission gate screens descriptions for
  over-claiming and instruction-injection before an agent joins the roster.

-------

examples/sta_agent_engine/agent_cards/agent_profile.yaml
----
# Local test fixture — a ROOT manifest that bundles the card file(s) below.
# NOT consumer documentation. Each entry is a path to a self-contained card file
# (graph key + path + card live there); paths resolve relative to THIS file's dir.
#
# Bundle many agents without inlining them, then compile or merge in one shot:
#   uv run sta agent-profile validate examples/sta_agent_engine/agent_cards/agent_profile.yaml
#   uv run sta agent-profile build    examples/sta_agent_engine/agent_cards/agent_profile.yaml --langgraph-json
#   uv run sta agent-profile build    examples/sta_agent_engine/agent_cards/agent_profile.yaml --into ./langgraph.json --dry-run
#
# A bare list works too (drop the `cards:` key and de-indent the entries).
cards:
  - ./template_agent.card.yaml

-------

examples/sta_agent_engine/agent_cards/template_agent.card.yaml
----
# Local test fixture — a capability card for this repo's `template_agent`.
# NOT consumer documentation: it exists so you can exercise the `sta agent-profile`
# CLI (validate / build / --into) against a real in-repo graph.
#
# Shape: a self-contained manifest (graph key -> {path, card}). Try:
#   uv run sta agent-profile validate examples/sta_agent_engine/agent_cards/template_agent.card.yaml
#   uv run sta agent-profile build    examples/sta_agent_engine/agent_cards/template_agent.card.yaml --langgraph-json
#   uv run sta agent-profile build    examples/sta_agent_engine/agent_cards/template_agent.card.yaml --langserve-env

template_agent:
  # Import path to the template agent's graph factory (Engine/Role/Knowledge demo).
  # Note: get_template_graph() is a factory that takes a domain — this path is the
  # honest entry point, not a zero-config server graph. It is metadata for the card;
  # the CLI never imports it.
  path: sta_agent_engine.agents.template_agent.template_graph:get_template_graph
  card:
    # `>-` folds the paragraph; the blank line becomes a break.
    description: >-
      Reference ReAct agent (this repo's template_agent) wired to the example
      AIOps domain: a Site-Reliability assistant that searches operational
      information and checks the health and status of system components.

      Delegate when a user asks whether a service or component is healthy, or
      why one might be degraded. Demonstration agent — not a production data source.
    short_description: SRE ops assistant (template)
    scope: >-
      IT infrastructure and operations for the example AIOps domain — component
      health/status lookups and basic operational search. Reference/demo scope only.
    freshness: on-demand — queries the example tools at request time
    # `|-` keeps the steps on separate lines.
    how_to_use: |-
      1. Name the service or component you care about (e.g. "payment gateway").
      2. Ask for its status/health, or why it might be degraded.
      3. Optionally give a symptom (slow, errors, timeouts) to focus the search.
    examples:
      - is the payment service healthy?
      - why is the payment gateway slow?
      - check the status of the auth component
    visibility:
      orchestrator: true
      ui: false
    tags: [template, aiops, sre, ops, demo]

template_agent_2:
  # Import path to the template agent's graph factory (Engine/Role/Knowledge demo).
  # Note: get_template_graph() is a factory that takes a domain — this path is the
  # honest entry point, not a zero-config server graph. It is metadata for the card;
  # the CLI never imports it.
  path: sta_agent_engine.agents.template_agent.template_graph:get_template_graph
  card:
    # `>-` folds the paragraph; the blank line becomes a break.
    description: >-
      Reference ReAct agent (this repo's template_agent) wired to the example
      AIOps domain: a Site-Reliability assistant that searches operational
      information and checks the health and status of system components.

      Delegate when a user asks whether a service or component is healthy, or
      why one might be degraded. Demonstration agent — not a production data source.
    short_description: SRE ops assistant (template)
    scope: >-
      IT infrastructure and operations for the example AIOps domain — component
      health/status lookups and basic operational search. Reference/demo scope only.
    freshness: on-demand — queries the example tools at request time
    # `|-` keeps the steps on separate lines.
    how_to_use: |-
      1. Name the service or component you care about (e.g. "payment gateway").
      2. Ask for its status/health, or why it might be degraded.
      3. Optionally give a symptom (slow, errors, timeouts) to focus the search.
    examples:
      - is the payment service healthy?
      - why is the payment gateway slow?
      - check the status of the auth component
    visibility:
      orchestrator: true
      ui: false
    tags: [template, aiops, sre, ops, demo]

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/template_agent/template_agent.card.yaml
----
# Local test fixture — a capability card for this repo's `template_agent`.
# NOT consumer documentation: it exists so you can exercise the `sta agent-profile`
# CLI (validate / build / --into) against a real in-repo graph.
#
# Shape: a self-contained manifest (graph key -> {path, card}). Try:
#   uv run sta agent-profile validate examples/sta_agent_engine/agent_cards/template_agent.card.yaml
#   uv run sta agent-profile build    examples/sta_agent_engine/agent_cards/template_agent.card.yaml --langgraph-json
#   uv run sta agent-profile build    examples/sta_agent_engine/agent_cards/template_agent.card.yaml --langserve-env

template_agent:
  # Import path to the template agent's graph factory (Engine/Role/Knowledge demo).
  # Note: get_template_graph() is a factory that takes a domain — this path is the
  # honest entry point, not a zero-config server graph. It is metadata for the card;
  # the CLI never imports it.
  path: sta_agent_engine.agents.template_agent.template_graph:get_template_graph
  card:
    # `>-` folds the paragraph; the blank line becomes a break.
    description: >-
      Reference ReAct agent (this repo's template_agent) wired to the example
      AIOps domain: a Site-Reliability assistant that searches operational
      information and checks the health and status of system components.

      Delegate when a user asks whether a service or component is healthy, or
      why one might be degraded. Demonstration agent — not a production data source.
    short_description: SRE ops assistant (template)
    scope: >-
      IT infrastructure and operations for the example AIOps domain — component
      health/status lookups and basic operational search. Reference/demo scope only.
    freshness: on-demand — queries the example tools at request time
    # `|-` keeps the steps on separate lines.
    how_to_use: |-
      1. Name the service or component you care about (e.g. "payment gateway").
      2. Ask for its status/health, or why it might be degraded.
      3. Optionally give a symptom (slow, errors, timeouts) to focus the search.
    examples:
      - is the payment service healthy?
      - why is the payment gateway slow?
      - check the status of the auth component
    visibility:
      orchestrator: true
      ui: false
    tags: [template, aiops, sre, ops, demo]

-------

packages/sta_agent_engine/src/sta_agent_engine/cli/agent_profile.py
----
"""``sta agent-profile`` — build, validate, and inspect a capability profile.

An external agent self-describes to the TWIN orchestrator by publishing a
capability profile as a JSON object **stringified into its graph's
``description``** field in ``langgraph.json`` — the only producer-controllable
field that reaches the A2A agent card on stock ``langgraph-api``. Because the
description is itself a string, the profile becomes JSON-inside-JSON and
hand-escaping is error-prone. This command validates the profile and emits
exactly what to paste, escaping handled.

Three authoring shapes are accepted, auto-detected:

- **Flat card** — a bare profile (``description``, ``scope``, …) at the top
  level. The graph ``name`` / ``path`` are supplied as CLI flags at build time.
- **Manifest** — a map of graph key → ``{path, card}`` mirroring
  ``langgraph.json``'s ``graphs`` block, but with the card as a readable nested
  object instead of a stringified blob. Self-contained (name + path travel with
  the card) so a single file lives next to ``graph.py`` in source control, and
  ``build`` needs no ``--name`` / ``--path`` flags.
- **Root manifest** — a bare list of paths (or ``{cards: [paths]}``) to *card
  files*, each itself a self-contained manifest. It bundles many agents without
  hand-inlining every card into one file: each path's graph key + ``path`` come
  from the referenced file. Paths resolve relative to the root manifest's
  directory.

The CLI compiles any shape down to the escaped ``langgraph.json`` entry or the
``LANGSERVE_GRAPHS`` env value. With ``--into PATH`` it writes the result
straight into an existing ``langgraph.json`` (the ``graphs`` block) or
``Dockerfile`` (the ``ENV LANGSERVE_GRAPHS`` line), **merging** so graphs already
present that are not in the manifest are preserved (``--replace`` makes the
manifest authoritative instead).

The pure functions (``load_profile``, ``build_description``, ``profile_warnings``,
``to_langgraph_entry``, ``is_manifest``, ``parse_manifest``, ``is_root_manifest``,
``merge_graphs``, ``merge_into_langgraph_json``, ``merge_langserve_env_into_dockerfile``)
carry the logic and are unit-tested directly; the Click commands are thin I/O
wrappers.

Future improvement (not implemented): an optional ``--judge`` mode that sends the
rendered description to an LLM judge to assess whether it is complete and
unambiguous enough for reliable routing — going beyond the structural heuristics
in :func:`profile_warnings`.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import click
import yaml
from pydantic import ValidationError

from sta_agent_engine.agents.cards import (
    AGENT_CAPABILITY_PROFILE_SCHEMA,
    AgentCapabilityProfile,
)


# The starter card, authored as YAML so `example --yaml` can ship the two things a
# dict cannot round-trip through ``yaml.safe_dump``: explanatory ``#`` comments and
# block scalars for the prose fields. The YAML is the source of truth — the dicts
# (and therefore the JSON output) are parsed from it, so the two can never drift.
_EXAMPLE_CARD_YAML = """\
# What the agent does AND when to delegate to it — the planner's primary signal.
# `>-` folds line breaks into spaces; a blank line becomes a paragraph break.
description: >-
  Searches application logs and answers questions about them.
  Delegate when a user asks why an application failed, or wants
  errors for a service in a given time window.

  Not for metrics or dashboards — that's a different agent's job.

# One-liner users read in the UI. Required in spirit when `visibility.ui` is true.
short_description: App log search

# The exact domain boundary, so the planner knows your coverage and won't over-route.
scope: Application logs for the X business infra

# Free text — how current the data is (e.g. real-time, daily, every 15 minutes).
freshness: every 15 minutes

# Prompting guidance, distinct from `examples`.
# `|-` keeps line breaks verbatim — use it for steps or bullets.
how_to_use: |-
  1. Give an application code (e.g. APP123).
  2. Give a time window.
  3. Optionally pass a log level to filter on.

# 1-3 verbatim sample queries you handle well. These anchor routing.
examples:
  - errors on app APP123 in the last hour
  - why did the payment service fail yesterday

# Which surfaces you opt into. Both default false — you are exposed nowhere
# until you ask. `orchestrator` routing is a future feature; `ui` works today.
visibility:
  orchestrator: true
  ui: false

# Descriptive labels for discovery and search only. Never access control.
tags: [logs, observability, sre]
"""

# The manifest form: the same card nested under a graph key alongside its import
# path — the self-contained shape a producer versions next to ``graph.py``.
_EXAMPLE_MANIFEST_YAML = (
    "# Your graph key — the same one you use in langgraph.json's `graphs` block.\n"
    "log_agent:\n"
    "  # Import path to your compiled graph.\n"
    "  path: ./log_agent.py:graph\n"
    "  card:\n" + textwrap.indent(_EXAMPLE_CARD_YAML, "    ")
)

_EXAMPLE_PROFILE: dict[str, Any] = yaml.safe_load(_EXAMPLE_CARD_YAML)
_EXAMPLE_MANIFEST: dict[str, Any] = yaml.safe_load(_EXAMPLE_MANIFEST_YAML)

# Below this, a description rarely carries both "what it does" and "when to
# delegate" — the two things the planner routes on.
_MIN_DESCRIPTION_CHARS = 40

# BuildKit does no escape processing inside a single-quoted Dockerfile ENV value, so
# a literal apostrophe terminates the string. The value is JSON, so encode it as the
# equivalent JSON escape instead. See :func:`to_dockerfile_env_line`.
_SINGLE_QUOTE = "'"
_JSON_ESCAPED_SINGLE_QUOTE = "\\u0027"

# An ``ENV LANGSERVE_GRAPHS=<quoted JSON>`` directive (single or double quoted).
# The value may span multiple physical lines — ``langgraph build`` emits one graph
# per line inside the quotes — so match across newlines (DOTALL, non-greedy) up to
# the first closing quote at a line end. ``^[ \t]*ENV`` after MULTILINE ``^`` means a
# ``# ENV …`` comment line never matches (it starts with ``#``, not ``ENV``).
_LANGSERVE_ENV_RE = re.compile(
    r"^[ \t]*ENV[ \t]+LANGSERVE_GRAPHS[ \t]*=[ \t]*(?P<quote>['\"])(?P<value>.*?)(?P=quote)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def load_document(text: str, *, fmt: str = "json") -> Any:
    """Parse a JSON/YAML document into its Python value (dict, list, …).

    Unlike :func:`load_profile`, this does not require an object at the top level:
    a root manifest is a *list* of card-file paths (or a ``{"cards": [...]}``
    mapping).

    Raises:
        ValueError: input is not valid JSON/YAML.
    """
    if fmt == "yaml":
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"input is not valid YAML: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"input is not valid JSON: {exc}") from exc


def load_profile(text: str, *, fmt: str = "json") -> dict[str, Any]:
    """Parse profile text (JSON or YAML) into a dict.

    Args:
        text: The raw profile document.
        fmt: ``"json"`` (default) or ``"yaml"``.

    Raises:
        ValueError: input is not valid JSON/YAML, or is not an object/mapping.
    """
    data = load_document(text, fmt=fmt)
    if not isinstance(data, dict):
        raise ValueError("profile must be an object")
    return data


def build_description(data: dict[str, Any]) -> tuple[str, list[str]]:
    """Validate a profile dict and return ``(description_string, unknown_keys)``.

    Unknown keys mirror the consumer's ``extra="ignore"`` behaviour (reported,
    not fatal — a producer typo is worth surfacing). Validation goes through the
    pydantic model so a missing/oversized/wrong-type field (e.g. an omitted
    required ``description``) surfaces as a ``ValidationError`` the commands
    translate to a clean error — never a raw ``TypeError``.

    Raises:
        pydantic.ValidationError: a field is missing/oversized/wrong type.
    """
    known = set(AgentCapabilityProfile.model_fields)
    unknown = sorted(set(data) - known)
    filtered = {key: value for key, value in data.items() if key in known}
    profile = AgentCapabilityProfile.model_validate(filtered)
    return profile.model_dump_json(exclude_none=True), unknown


def _exposed_on_any_surface(visibility: Any) -> bool:
    """Whether the profile opts into at least one exposure surface."""
    if not isinstance(visibility, Mapping):
        return False
    return bool(visibility.get("orchestrator")) or bool(visibility.get("ui"))


def profile_warnings(data: dict[str, Any]) -> list[str]:
    """Heuristic completeness suggestions (improvements, not hard errors).

    Each absent-or-thin field weakens planner routing in a specific way; the
    suggestion names the consequence so a producer knows why it matters.
    """
    suggestions: list[str] = []
    description = data.get("description")
    if isinstance(description, str) and len(description.strip()) < _MIN_DESCRIPTION_CHARS:
        suggestions.append(f"description is short (<{_MIN_DESCRIPTION_CHARS} chars) — describe BOTH what the agent does and when to delegate to it.")
    if not data.get("scope"):
        suggestions.append("no scope — add the exact domain boundary so the planner knows the agent's precise coverage.")
    if not data.get("how_to_use"):
        suggestions.append("no how_to_use — add prompting guidance so the planner queries the agent effectively.")
    if not data.get("examples"):
        suggestions.append("no examples — add 1-3 verbatim sample queries to anchor routing.")
    if not data.get("freshness"):
        suggestions.append("no freshness — state how current the data is (e.g. 'real-time', 'daily', 'every 15 minutes').")
    if not _exposed_on_any_surface(data.get("visibility")):
        suggestions.append("not exposed on any surface — set visibility.orchestrator and/or visibility.ui to true, or the agent won't be discovered.")
    if not data.get("tags"):
        suggestions.append("no tags — add a few descriptive tags (e.g. 'logs', 'sre') to aid discovery and search.")
    return suggestions


def to_langgraph_entry(name: str, path: str, description: str) -> dict[str, Any]:
    """Build the ``langgraph.json`` graph entry ``{name: {path, description}}``.

    ``json.dumps`` of the returned dict escapes the embedded JSON description
    correctly, so producers never escape by hand.
    """
    return {name: {"path": path, "description": description}}


def to_langserve_graphs_value(name: str, path: str, description: str) -> str:
    """The ``LANGSERVE_GRAPHS`` env *value*: a compact one-graph JSON map.

    The A2A card is served at runtime from the ``LANGSERVE_GRAPHS`` env var (the
    server reads its graphs — and their ``description`` — from it at startup), so
    setting this env is all a custom image needs. It is byte-compatible with what
    ``langgraph build`` bakes in, so you can run your own Docker image / compose /
    Kubernetes without ``langgraph build`` and still serve the same card.
    """
    return json.dumps(to_langgraph_entry(name, path, description))


def to_dockerfile_env_line(value: str) -> str:
    """Wrap a ``LANGSERVE_GRAPHS`` JSON value as a paste-ready Dockerfile ``ENV`` line.

    Matches the single-quoted form ``langgraph build`` emits: single quotes keep the
    JSON's escaped inner double quotes literal, and BuildKit does no escape
    processing inside them.

    A literal ``'`` would terminate that quoted string and break ``docker build``
    ("unexpected end of statement") — and an ordinary apostrophe in English prose
    ("the application's logs") is enough to trigger it. Since the value is JSON, the
    apostrophe is re-encoded as the ``\\u0027`` escape: still valid JSON, no literal
    quote left to terminate the string, and it decodes back to ``'`` when the server
    parses the env. Output is byte-identical to ``langgraph build`` for any card
    without an apostrophe; for one *with* an apostrophe ``langgraph build`` emits an
    unbuildable Dockerfile and this does not. That is the sole intentional divergence.

    Args:
        value: The ``LANGSERVE_GRAPHS`` env value — a JSON graphs map. A literal
            ``'`` can only occur inside a JSON string, never in structural syntax,
            so escaping every occurrence is safe.
    """
    escaped = value.replace(_SINGLE_QUOTE, _JSON_ESCAPED_SINGLE_QUOTE)
    return f"ENV LANGSERVE_GRAPHS='{escaped}'"


def to_langgraph_graphs(built: list[tuple[str, str, str]]) -> dict[str, Any]:
    """Merge ``(name, path, description)`` triples into a langgraph.json graphs map.

    Single-entry output is identical to :func:`to_langgraph_entry`; multi-entry
    merges every agent under one map — the shape both ``langgraph.json``'s
    ``graphs`` block and the ``LANGSERVE_GRAPHS`` env value expect.
    """
    graphs: dict[str, Any] = {}
    for name, path, description in built:
        graphs[name] = {"path": path, "description": description}
    return graphs


def _format_langserve_graphs(graphs: Mapping[str, Any], *, multiline: bool) -> str:
    """Serialize a graphs map to the JSON that goes inside ``LANGSERVE_GRAPHS``.

    ``multiline`` puts one graph per line — the readable layout ``langgraph build``
    emits — and is still valid JSON (the newlines are insignificant whitespace
    between members). It collapses to a single line for zero or one graph, where
    there is nothing to break onto its own line.
    """
    if not multiline or len(graphs) <= 1:
        return json.dumps(graphs)
    entries = [f"{json.dumps(key)}: {json.dumps(value)}" for key, value in graphs.items()]
    return "{" + ",\n".join(entries) + "}"


@dataclass(frozen=True)
class ManifestEntry:
    """One agent in a manifest: a graph ``name`` → import ``path`` + raw ``card``."""

    name: str
    path: str | None
    card: dict[str, Any]


def is_manifest(data: Mapping[str, Any]) -> bool:
    """Whether ``data`` is a manifest (graph key → {path, card}) vs a flat card.

    A manifest maps each graph name to a spec containing a nested ``card``. A flat
    profile carries profile fields (``description`` etc.) at the top level, so its
    values are never *all* card-bearing mappings — the two shapes are unambiguous.
    """
    return bool(data) and all(isinstance(value, Mapping) and "card" in value for value in data.values())


def parse_manifest(data: Mapping[str, Any]) -> list[ManifestEntry]:
    """Split a manifest into per-agent entries.

    Raises:
        ValueError: an agent's ``card`` is not a mapping, or ``path`` is not a string.
    """
    entries: list[ManifestEntry] = []
    for name, spec in data.items():
        card = spec.get("card")
        if not isinstance(card, Mapping):
            raise ValueError(f"agent {name!r}: 'card' must be a mapping")
        path = spec.get("path")
        if path is not None and not isinstance(path, str):
            raise ValueError(f"agent {name!r}: 'path' must be a string")
        entries.append(ManifestEntry(name=str(name), path=path, card=dict(card)))
    return entries


def is_root_manifest(doc: Any) -> bool:
    """Whether ``doc`` is a root include-manifest (a list of card-file paths).

    Two accepted forms: a bare list of path strings, or a ``{"cards": [paths]}``
    wrapper. Distinct from a flat card (profile fields at top level) and a manifest
    (values are card-bearing mappings), so the three shapes never collide.
    """
    if isinstance(doc, list):
        return bool(doc) and all(isinstance(item, str) for item in doc)
    if isinstance(doc, Mapping):
        cards = doc.get("cards")
        return isinstance(cards, list) and bool(cards) and all(isinstance(item, str) for item in cards)
    return False


def parse_root_manifest(doc: Any) -> list[str]:
    """Return the card-file paths listed in a root manifest.

    Raises:
        ValueError: ``doc`` is not a valid root-manifest shape or is empty.
    """
    if isinstance(doc, list):
        paths = doc
    elif isinstance(doc, Mapping) and isinstance(doc.get("cards"), list):
        paths = doc["cards"]
    else:
        raise ValueError("root manifest must be a list of card paths or a {'cards': [...]} mapping")
    if not paths:
        raise ValueError("root manifest is empty — list at least one card file")
    if any(not isinstance(item, str) for item in paths):
        raise ValueError("root manifest paths must all be strings")
    return list(paths)


class GraphMergeResult(NamedTuple):
    """Outcome of merging new graphs into an existing graphs map.

    ``merged`` is the resulting map; the key lists report exactly what changed so
    the CLI can echo a delta (a write into a real config is never a black box).
    """

    merged: dict[str, Any]
    added: list[str]
    overwritten: list[str]
    preserved: list[str]
    dropped: list[str]


def merge_graphs(existing: Mapping[str, Any], new: Mapping[str, Any], *, replace: bool) -> GraphMergeResult:
    """Merge ``new`` graph entries into ``existing`` (a langgraph graphs map).

    Default (**merge-preserve**): every graph only in ``existing`` is kept and
    ``new`` overrides same-named graphs. ``replace=True`` (**authoritative**): the
    result is exactly ``new`` — graphs only in ``existing`` are dropped. Existing
    insertion order is preserved (new keys append).
    """
    existing_keys = set(existing)
    new_keys = set(new)
    added = sorted(new_keys - existing_keys)
    overwritten = sorted(new_keys & existing_keys)
    if replace:
        return GraphMergeResult(merged=dict(new), added=added, overwritten=overwritten, preserved=[], dropped=sorted(existing_keys - new_keys))
    merged = dict(existing)
    merged.update(new)
    return GraphMergeResult(merged=merged, added=added, overwritten=overwritten, preserved=sorted(existing_keys - new_keys), dropped=[])


def merge_into_langgraph_json(doc: Mapping[str, Any], new_graphs: Mapping[str, Any], *, replace: bool) -> tuple[dict[str, Any], GraphMergeResult]:
    """Merge ``new_graphs`` into a langgraph.json document's ``graphs`` block.

    Every other top-level key (``dependencies``, ``env``, …) and the document's key
    order are preserved.

    Raises:
        ValueError: ``graphs`` exists but is not an object.
    """
    graphs = doc.get("graphs", {})
    if not isinstance(graphs, Mapping):
        raise ValueError("langgraph.json 'graphs' is not an object")
    result = merge_graphs(graphs, new_graphs, replace=replace)
    updated = dict(doc)
    updated["graphs"] = result.merged
    return updated, result


def extract_langserve_graphs(dockerfile_text: str) -> tuple[dict[str, Any] | None, list[tuple[int, int]]]:
    """Parse the effective ``LANGSERVE_GRAPHS`` value and the span of every directive.

    Handles a value that spans multiple physical lines (the shape ``langgraph build``
    emits). Later ``ENV`` wins at runtime, so the *last* directive's value is
    authoritative. Returns ``(graphs, spans)`` where each span is the ``(start, end)``
    character offset of one directive — ``(None, [])`` if none is present.

    Raises:
        ValueError: the effective value is not a JSON object (never merge blindly
            over content we cannot parse).
    """
    matches = list(_LANGSERVE_ENV_RE.finditer(dockerfile_text))
    if not matches:
        return None, []
    effective = matches[-1].group("value")
    try:
        graphs = json.loads(effective)
    except json.JSONDecodeError as exc:
        raise ValueError(f"existing LANGSERVE_GRAPHS value is not valid JSON: {exc}") from exc
    if not isinstance(graphs, dict):
        raise ValueError("existing LANGSERVE_GRAPHS value is not a JSON object")
    return graphs, [match.span() for match in matches]


def merge_langserve_env_into_dockerfile(text: str, new_graphs: Mapping[str, Any], *, replace: bool) -> tuple[str, GraphMergeResult]:
    """Merge ``new_graphs`` into a Dockerfile's ``ENV LANGSERVE_GRAPHS`` directive.

    Symmetric with :func:`merge_into_langgraph_json`: graphs already present are
    preserved (merge-preserve) unless ``replace=True``. Recognises a **multi-line**
    directive (``langgraph build``'s one-graph-per-line form) and rewrites it in the
    same readable multi-line layout in the first directive's position, dropping any
    duplicate directives; appends a fresh directive only when none exists. Everything
    else is untouched.

    Raises:
        ValueError: an existing ``LANGSERVE_GRAPHS`` value is malformed.
    """
    existing_graphs, spans = extract_langserve_graphs(text)
    result = merge_graphs(existing_graphs or {}, new_graphs, replace=replace)
    env_directive = to_dockerfile_env_line(_format_langserve_graphs(result.merged, multiline=True))

    if not spans:
        body = text if text == "" or text.endswith("\n") else text + "\n"
        suffix = "\n" if text.endswith("\n") or text == "" else ""
        return body + env_directive + suffix, result

    # Splice by character offset: the merged directive replaces the first one; any
    # duplicate directives are removed (with their trailing newline) so exactly one
    # canonical directive remains. Char spans (not line lists) correctly handle a
    # directive whose value spans several physical lines.
    pieces: list[str] = []
    cursor = 0
    for index, (start, end) in enumerate(spans):
        pieces.append(text[cursor:start])
        if index == 0:
            pieces.append(env_directive)
        cursor = end
        if index != 0 and cursor < len(text) and text[cursor] == "\n":
            cursor += 1
    pieces.append(text[cursor:])
    return "".join(pieces), result


def detect_destination(path: str) -> str:
    """Infer the ``--into`` merge destination kind from a filename.

    ``*.json`` → ``"langgraph"``; ``Dockerfile`` / ``Dockerfile.*`` / ``*.dockerfile``
    → ``"dockerfile"``.

    Raises:
        ValueError: the kind can't be inferred (caller should ask for ``--as``).
    """
    name = Path(path).name.lower()
    if name.endswith(".json"):
        return "langgraph"
    if name == "dockerfile" or name.startswith("dockerfile.") or name.endswith(".dockerfile"):
        return "dockerfile"
    raise ValueError(f"cannot infer destination type from filename {Path(path).name!r} — pass --as langgraph-json|dockerfile.")


def _detect_format(filename: str) -> str:
    """Pick the parser by file extension; stdin / unknown defaults to JSON."""
    return "yaml" if filename.lower().endswith((".yaml", ".yml")) else "json"


def _resolve_format(choice: str, filename: str) -> str:
    """Resolve an ``--format`` choice (``auto`` → detect by extension)."""
    return _detect_format(filename) if choice == "auto" else choice


def _load_document_or_abort(text: str, fmt: str) -> Any:
    try:
        return load_document(text, fmt=fmt)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _manifest_base_dir(filename: str) -> Path:
    """Directory that a root manifest's card refs resolve against (stdin → CWD)."""
    if filename in ("-", "<stdin>"):
        return Path.cwd()
    return Path(filename).resolve().parent


def _manifest_entries_or_abort(data: Mapping[str, Any], name: str | None = None, path: str | None = None) -> list[ManifestEntry]:
    """Parse a manifest, applying single-agent ``--name`` / ``--path`` overrides.

    Overrides target the one-agent-per-file case; supplying them for a multi-agent
    manifest is ambiguous and rejected.
    """
    try:
        entries = parse_manifest(data)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if name or path:
        if len(entries) != 1:
            raise click.UsageError("--name/--path can only override a single-agent manifest")
        only = entries[0]
        entries = [ManifestEntry(name=name or only.name, path=path or only.path, card=only.card)]
    return entries


def _load_root_entries_or_abort(doc: Any, *, base_dir: Path) -> list[ManifestEntry]:
    """Resolve a root manifest into a flat, de-duplicated list of manifest entries.

    Each referenced file must be a self-contained manifest (its graph key + path
    live there). Raises a clean ``ClickException`` for a missing file, a flat-card
    reference (no graph path), or a duplicate graph key across files.
    """
    try:
        paths = parse_root_manifest(doc)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    entries: list[ManifestEntry] = []
    seen: dict[str, str] = {}
    for ref in paths:
        full = base_dir / ref
        if not full.is_file():
            raise click.ClickException(f"referenced card file not found: {ref} (resolved to {full})")
        try:
            sub = load_document(full.read_text(encoding="utf-8"), fmt=_detect_format(full.name))
        except ValueError as exc:
            raise click.ClickException(f"card file {ref}: {exc}") from exc
        if not (isinstance(sub, dict) and is_manifest(sub)):
            raise click.ClickException(
                f"card file {ref}: must be a self-contained manifest (graph_key: {{path, card}}); a flat card has no graph path — wrap it under a graph key with a path, or inline it."
            )
        try:
            sub_entries = parse_manifest(sub)
        except ValueError as exc:
            raise click.ClickException(f"card file {ref}: {exc}") from exc
        for entry in sub_entries:
            if entry.name in seen:
                raise click.ClickException(f"duplicate graph key {entry.name!r} — defined in both {seen[entry.name]} and {ref}.")
            seen[entry.name] = ref
            entries.append(entry)
    return entries


def _describe_or_abort(card: dict[str, Any], *, label: str | None = None) -> str:
    """Validate a card, echo unknown-key warnings, return the description string.

    ``label`` (a graph key) tags manifest-agent messages; ``None`` for a flat card.
    """
    prefix = f"agent {label!r}: " if label else ""
    kind = "card" if label else "profile"
    try:
        description, unknown = build_description(card)
    except ValidationError as exc:
        raise click.ClickException(f"{prefix}invalid {kind}:\n{exc}") from exc
    for key in unknown:
        click.echo(f"warning: {prefix}ignoring unknown field {key!r}", err=True)
    return description


def _built_from_entries_or_abort(entries: list[ManifestEntry]) -> list[tuple[str, str, str]]:
    """Compile manifest entries into ``(name, path, description)`` triples."""
    built: list[tuple[str, str, str]] = []
    for entry in entries:
        if not entry.path:
            raise click.ClickException(
                f"agent {entry.name!r}: missing 'path' — add it under the agent in the manifest (or pass --path for a single-agent manifest)."
            )
        built.append((entry.name, entry.path, _describe_or_abort(entry.card, label=entry.name)))
    return built


def _emit_built_to_stdout(built: list[tuple[str, str, str]], *, as_langserve_env: bool) -> None:
    """Print a graphs map (default / ``--langgraph-json``) or a Dockerfile ENV directive."""
    graphs = to_langgraph_graphs(built)
    if as_langserve_env:
        click.echo(to_dockerfile_env_line(_format_langserve_graphs(graphs, multiline=True)))
    else:
        click.echo(json.dumps(graphs, indent=2, ensure_ascii=False))


def _resolve_into_destination(dest_choice: str, as_entry: bool, as_langserve_env: bool, path: str) -> str:
    """Resolve the ``--into`` destination from ``--as``, shape flags, and filename.

    All present signals must agree; a conflict is a usage error rather than a silent
    guess about which file kind the producer meant.
    """
    signals: set[str] = set()
    if dest_choice != "auto":
        signals.add("langgraph" if dest_choice == "langgraph-json" else "dockerfile")
    if as_entry:
        signals.add("langgraph")
    if as_langserve_env:
        signals.add("dockerfile")
    with contextlib.suppress(ValueError):
        signals.add(detect_destination(path))
    if len(signals) > 1:
        raise click.UsageError(f"--into destination is ambiguous: {sorted(signals)} — make --as, the shape flag, and the filename agree.")
    if not signals:
        raise click.UsageError("cannot infer the --into destination from the filename; pass --as langgraph-json|dockerfile.")
    return next(iter(signals))


def _read_destination(path: str, *, create: bool) -> str | None:
    """Read a destination file; ``None`` if absent and ``--create`` was given."""
    target = Path(path)
    if target.is_file():
        return target.read_text(encoding="utf-8")
    if create:
        return None
    raise click.ClickException(f"{path} does not exist — pass --create to create it.")


def _merge_langgraph_text(existing_text: str | None, graphs: dict[str, Any], *, replace: bool) -> tuple[str, GraphMergeResult]:
    if existing_text is None or existing_text.strip() == "":
        doc: dict[str, Any] = {}
    else:
        try:
            parsed = json.loads(existing_text)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"existing langgraph.json is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise click.ClickException("existing langgraph.json is not a JSON object")
        doc = parsed
    try:
        updated, result = merge_into_langgraph_json(doc, graphs, replace=replace)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    return json.dumps(updated, indent=2, ensure_ascii=False) + "\n", result


def _merge_dockerfile_text(existing_text: str | None, graphs: dict[str, Any], *, replace: bool) -> tuple[str, GraphMergeResult]:
    try:
        return merge_langserve_env_into_dockerfile(existing_text or "", graphs, replace=replace)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _report_delta(result: GraphMergeResult) -> None:
    """Echo the merge delta to stderr so the write is auditable."""

    def fmt(keys: list[str]) -> str:
        return ", ".join(keys) if keys else "—"

    click.echo(f"  added: {fmt(result.added)}", err=True)
    click.echo(f"  overwritten: {fmt(result.overwritten)}", err=True)
    click.echo(f"  preserved: {fmt(result.preserved)}", err=True)
    if result.dropped:
        click.echo(f"  dropped (--replace): {fmt(result.dropped)}", err=True)


def _atomic_write(path: str, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + replace).

    A crash mid-write can never leave a truncated ``langgraph.json`` / ``Dockerfile``.
    """
    target = Path(path)
    directory = target.parent if str(target.parent) else Path()
    fd, tmp = tempfile.mkstemp(dir=str(directory), prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        tmp_path.replace(target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def _write_into_destination(
    path: str, graphs: dict[str, Any], *, dest_choice: str, as_entry: bool, as_langserve_env: bool, replace: bool, dry_run: bool, create: bool
) -> None:
    """Merge ``graphs`` into an existing ``langgraph.json`` / ``Dockerfile``."""
    dest = _resolve_into_destination(dest_choice, as_entry, as_langserve_env, path)
    # A dry run previews a merge even against an absent file (empty base); a real
    # write requires the file unless --create was given.
    existing_text = _read_destination(path, create=create or dry_run)
    if dest == "langgraph":
        new_text, result = _merge_langgraph_text(existing_text, graphs, replace=replace)
    else:
        new_text, result = _merge_dockerfile_text(existing_text, graphs, replace=replace)

    _report_delta(result)
    if dry_run:
        click.echo(new_text)
    else:
        _atomic_write(path, new_text)
        click.echo(f"✓ merged {len(graphs)} graph(s) into {path}", err=True)


@click.group(name="agent-profile")
def agent_profile() -> None:
    """Build, validate, and inspect an external agent's capability profile.

    The profile is the JSON object an external agent publishes (stringified into
    its langgraph.json ``description``) so the TWIN orchestrator can route to it.
    """


@agent_profile.command()
@click.argument("file", type=click.File("r"), default="-")
@click.option("--strict", is_flag=True, help="Exit non-zero if any completeness suggestions remain.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["auto", "json", "yaml"]),
    default="auto",
    help="Input format (auto = by file extension; stdin defaults to JSON).",
)
def validate(file: click.utils.LazyFile, strict: bool, fmt: str) -> None:
    """Validate a profile and report completeness suggestions.

    FILE is a flat card, a manifest, or a root manifest (a list of card paths),
    JSON or YAML (default: stdin). Reports hard errors (invalid schema / oversized
    fields), unknown keys, and improvement suggestions — per agent for a manifest.
    """
    doc = _load_document_or_abort(file.read(), _resolve_format(fmt, file.name))

    if is_root_manifest(doc):
        _validate_entries(_load_root_entries_or_abort(doc, base_dir=_manifest_base_dir(file.name)), strict=strict)
        return
    if isinstance(doc, dict) and is_manifest(doc):
        _validate_entries(_manifest_entries_or_abort(doc), strict=strict)
        return
    if not isinstance(doc, dict):
        raise click.ClickException("input must be a card, a manifest, or a root manifest (a list of card-file paths).")

    try:
        _, unknown = build_description(doc)
    except ValidationError as exc:
        raise click.ClickException(f"invalid profile:\n{exc}") from exc

    click.echo("✓ profile is valid")
    for key in unknown:
        click.echo(f"  unknown field ignored: {key!r}", err=True)

    suggestions = profile_warnings(doc)
    for suggestion in suggestions:
        click.echo(f"  suggestion: {suggestion}", err=True)

    if not suggestions:
        click.echo("  no completeness suggestions — looks great.")
    elif strict:
        raise SystemExit(1)


def _validate_entries(entries: list[ManifestEntry], *, strict: bool) -> None:
    """Validate each agent's card in a (root or inline) manifest, reporting per-agent."""
    any_suggestions = False
    for entry in entries:
        try:
            _, unknown = build_description(entry.card)
        except ValidationError as exc:
            raise click.ClickException(f"agent {entry.name!r}: invalid card:\n{exc}") from exc

        click.echo(f"✓ {entry.name}: card is valid")
        for key in unknown:
            click.echo(f"  {entry.name}: unknown field ignored: {key!r}", err=True)

        suggestions = list(profile_warnings(entry.card))
        if not entry.path:
            suggestions.insert(0, "no path — add the graph import path (e.g. ./agent.py:graph).")
        for suggestion in suggestions:
            click.echo(f"  {entry.name}: suggestion: {suggestion}", err=True)
        any_suggestions = any_suggestions or bool(suggestions)

    if not any_suggestions:
        click.echo("  no completeness suggestions — looks great.")
    elif strict:
        raise SystemExit(1)


@agent_profile.command()
@click.argument("file", type=click.File("r"), default="-")
@click.option(
    "--langgraph-json",
    "as_entry",
    is_flag=True,
    help="Emit a ready-to-paste langgraph.json graphs entry (flat card needs --name/--path; a manifest supplies them).",
)
@click.option(
    "--langserve-env",
    "as_langserve_env",
    is_flag=True,
    help="Emit a Dockerfile `ENV LANGSERVE_GRAPHS=...` line to run your own image without `langgraph build` (flat card needs --name/--path; a manifest supplies them).",
)
@click.option(
    "--into",
    "into_path",
    metavar="PATH",
    help="Merge the built graphs into an existing langgraph.json / Dockerfile in place (kind inferred from the filename).",
)
@click.option(
    "--as",
    "dest_choice",
    type=click.Choice(["auto", "langgraph-json", "dockerfile"]),
    default="auto",
    help="Force the --into destination kind when the filename is ambiguous.",
)
@click.option(
    "--replace",
    is_flag=True,
    help="With --into: make the manifest authoritative — drop graphs in the destination that are not in it (default: merge-preserve).",
)
@click.option("--dry-run", is_flag=True, help="With --into: print the merged result + delta instead of writing.")
@click.option("--create", is_flag=True, help="With --into: create the destination file if it does not exist.")
@click.option("--name", help="Graph key for the entry (flat card; overrides a single-agent manifest).")
@click.option("--path", "graph_path", help="Graph import path, e.g. ./agent.py:graph (flat card; overrides a single-agent manifest).")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["auto", "json", "yaml"]),
    default="auto",
    help="Input format (auto = by file extension; stdin defaults to JSON).",
)
def build(
    file: click.utils.LazyFile,
    as_entry: bool,
    as_langserve_env: bool,
    into_path: str | None,
    dest_choice: str,
    replace: bool,
    dry_run: bool,
    create: bool,
    name: str | None,
    graph_path: str | None,
    fmt: str,
) -> None:
    """Emit the capability string, or merge it into your deploy config.

    Accepts a **flat card**, a **manifest** (graph key → {path, card}), or a **root
    manifest** (a list of card-file paths that bundles many agents), auto-detected.

    Output to stdout (default): the bare ``description`` string for a flat card;
    the ``graphs`` map for a manifest/root manifest. ``--langgraph-json`` /
    ``--langserve-env`` select the shape (a flat card then needs ``--name``/``--path``).

    Write in place with ``--into PATH``: merges the built graphs into an existing
    ``langgraph.json`` (``graphs`` block) or ``Dockerfile`` (``ENV LANGSERVE_GRAPHS``
    line), inferred from the filename. Graphs already present but absent from the
    manifest are **preserved**; ``--replace`` makes the manifest authoritative.
    ``--dry-run`` previews, ``--create`` allows a missing target.

    FILE (JSON or YAML) defaults to stdin. The emitted description is always JSON —
    the wire format the A2A card carries — regardless of the input format.
    """
    if as_entry and as_langserve_env:
        raise click.UsageError("choose one of --langgraph-json or --langserve-env, not both")

    doc = _load_document_or_abort(file.read(), _resolve_format(fmt, file.name))

    # Resolve manifest / root-manifest inputs to built triples; None = flat card.
    if is_root_manifest(doc):
        built: list[tuple[str, str, str]] | None = _built_from_entries_or_abort(
            _load_root_entries_or_abort(doc, base_dir=_manifest_base_dir(file.name))
        )
    elif isinstance(doc, dict) and is_manifest(doc):
        built = _built_from_entries_or_abort(_manifest_entries_or_abort(doc, name=name, path=graph_path))
    elif isinstance(doc, dict):
        built = None
    else:
        raise click.ClickException("input must be a card, a manifest, or a root manifest (a list of card-file paths).")

    if into_path:
        if built is not None:
            graphs = to_langgraph_graphs(built)
        else:
            if not (name and graph_path):
                raise click.UsageError("--into requires --name and --path for a flat card")
            graphs = to_langgraph_graphs([(name, graph_path, _describe_or_abort(doc))])
        _write_into_destination(
            into_path,
            graphs,
            dest_choice=dest_choice,
            as_entry=as_entry,
            as_langserve_env=as_langserve_env,
            replace=replace,
            dry_run=dry_run,
            create=create,
        )
        return

    if built is not None:
        _emit_built_to_stdout(built, as_langserve_env=as_langserve_env)
        return

    # Flat card → stdout.
    if (as_entry or as_langserve_env) and not (name and graph_path):
        raise click.UsageError("--langgraph-json / --langserve-env requires --name and --path")

    description = _describe_or_abort(doc)
    if as_langserve_env:
        assert name is not None and graph_path is not None  # guarded above
        click.echo(to_dockerfile_env_line(to_langserve_graphs_value(name, graph_path, description)))
    elif as_entry:
        assert name is not None and graph_path is not None  # guarded above
        click.echo(json.dumps(to_langgraph_entry(name, graph_path, description), indent=2, ensure_ascii=False))
    else:
        click.echo(description)


@agent_profile.command()
def schema() -> None:
    """Print the capability-profile JSON schema."""
    click.echo(json.dumps(AGENT_CAPABILITY_PROFILE_SCHEMA, indent=2))


@agent_profile.command()
@click.option("--flat", "as_flat", is_flag=True, help="Emit a bare card instead — supply --name/--path at build time.")
@click.option("--yaml", "as_yaml", is_flag=True, help="Emit the example as commented YAML with block scalars (recommended).")
def example(as_flat: bool, as_yaml: bool) -> None:
    """Print a starter example to fill in.

    Defaults to the self-contained **manifest** (graph key → path + card) — the
    recommended shape, versioned next to your graph. Use ``--flat`` for a bare card
    (name/path passed as build flags).

    ``--yaml`` emits the annotated template: inline ``#`` comments explaining each
    field, and block scalars (``>-`` folded, ``|-`` literal) for the prose fields.
    JSON output carries the same values but, having neither comments nor block
    scalars, renders the prose with ``\\n`` escapes.
    """
    if as_yaml:
        click.echo((_EXAMPLE_CARD_YAML if as_flat else _EXAMPLE_MANIFEST_YAML).rstrip())
    else:
        click.echo(json.dumps(_EXAMPLE_PROFILE if as_flat else _EXAMPLE_MANIFEST, indent=2))


__all__ = [
    "GraphMergeResult",
    "ManifestEntry",
    "agent_profile",
    "build_description",
    "detect_destination",
    "extract_langserve_graphs",
    "is_manifest",
    "is_root_manifest",
    "load_document",
    "load_profile",
    "merge_graphs",
    "merge_into_langgraph_json",
    "merge_langserve_env_into_dockerfile",
    "parse_manifest",
    "parse_root_manifest",
    "profile_warnings",
    "to_dockerfile_env_line",
    "to_langgraph_entry",
    "to_langgraph_graphs",
    "to_langserve_graphs_value",
]

-------

tests/test_ai_engine/cli/test_agent_profile_cli.py
----
"""Tests for the ``sta agent-profile`` CLI.

Two layers: pure functions (``load_profile``, ``build_description``,
``profile_warnings``, ``to_langgraph_entry``) tested directly, and the Click
commands driven through ``CliRunner``. The strongest check is the full
producer round-trip: the string the CLI emits, parsed back through the neutral
agent-card contract, reproduces the profile.

Like the CLI itself, this suite imports no orchestrator code — only
``sta_agent_engine.agents.cards``. Mapping a profile onto the planner's
``CapabilityDefinition`` is a consumer concern, tested with that consumer.
"""

from __future__ import annotations

import json

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from sta_agent_engine.agents.cards import extract_agent_profile
from sta_agent_engine.cli.agent_profile import (
    _EXAMPLE_PROFILE,
    agent_profile,
    build_description,
    detect_destination,
    extract_langserve_graphs,
    is_manifest,
    is_root_manifest,
    load_profile,
    merge_graphs,
    merge_into_langgraph_json,
    merge_langserve_env_into_dockerfile,
    parse_manifest,
    parse_root_manifest,
    profile_warnings,
    to_dockerfile_env_line,
    to_langgraph_entry,
    to_langgraph_graphs,
    to_langserve_graphs_value,
)
from sta_agent_engine.cli.app import main


def _example() -> dict:
    return dict(_EXAMPLE_PROFILE)


def _manifest() -> dict:
    """A single-agent manifest (graph key → {path, card})."""
    return {"log_agent": {"path": "./log_agent.py:graph", "card": dict(_EXAMPLE_PROFILE)}}


def _multi_manifest() -> dict:
    """A two-agent manifest, to exercise merging into one graphs map."""
    return {
        "log_agent": {"path": "./log_agent.py:graph", "card": dict(_EXAMPLE_PROFILE)},
        "metric_agent": {
            "path": "./metric_agent.py:graph",
            "card": dict(_EXAMPLE_PROFILE) | {"description": "Answers questions about metrics and dashboards."},
        },
    }


# ── pure functions ───────────────────────────────────────────────────────────


def test_build_description_round_trips_back_through_the_card_contract() -> None:
    description, unknown = build_description(_example())

    assert unknown == []
    profile = extract_agent_profile({"name": "log_agent", "description": description})
    assert profile is not None
    assert profile.description.startswith("Searches application logs and answers questions about them.")
    assert profile.scope == "Application logs for the X business infra"
    assert profile.how_to_use is not None and profile.how_to_use.startswith("1. Give an application code")
    assert profile.freshness == "every 15 minutes"


def test_example_card_prose_uses_block_scalars_and_survives_the_wire() -> None:
    # The starter card demonstrates YAML block scalars: `>-` folds the description
    # into a paragraph (the blank line becomes one \n), `|-` keeps how_to_use steps.
    assert "\n" in _EXAMPLE_PROFILE["description"]  # folded paragraph break
    assert _EXAMPLE_PROFILE["how_to_use"].count("\n") == 2  # three literal steps

    description, _ = build_description(_example())
    payload = json.loads(description)  # newlines survive JSON encoding as \n
    assert "\n" in payload["description"]
    assert payload["how_to_use"].count("\n") == 2


def test_build_description_carries_visibility_and_tags() -> None:
    description, unknown = build_description(_example())

    assert unknown == []
    payload = json.loads(description)
    assert payload["visibility"] == {"orchestrator": True, "ui": False}
    assert payload["tags"] == ["logs", "observability", "sre"]


def test_build_description_reports_unknown_keys() -> None:
    description, unknown = build_description(_example() | {"scopee": "typo", "extra": 1})

    assert unknown == ["extra", "scopee"]
    assert json.loads(description)["scope"] == "Application logs for the X business infra"


def test_build_description_raises_on_invalid_input() -> None:
    with pytest.raises(ValidationError):
        build_description({"description": "x" * 5000})


def test_build_description_raises_validation_error_on_missing_description() -> None:
    # A missing required field must be a ValidationError (the commands translate
    # it to a clean error), never a raw TypeError from kwargs unpacking.
    with pytest.raises(ValidationError):
        build_description({"scope": "no description supplied"})


def test_load_profile_parses_yaml() -> None:
    data = load_profile("description: hello world\ntags:\n  - a\n  - b\n", fmt="yaml")

    assert data["description"] == "hello world"
    assert data["tags"] == ["a", "b"]


def test_load_profile_rejects_bad_yaml() -> None:
    with pytest.raises(ValueError, match="not valid YAML"):
        load_profile("tags: [unclosed\n", fmt="yaml")


def test_load_profile_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        load_profile("[1, 2, 3]")


def test_load_profile_rejects_bad_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        load_profile("{not json")


def test_to_langgraph_entry_escapes_embedded_json() -> None:
    description, _ = build_description(_example())
    entry = to_langgraph_entry("log_agent", "./agent.py:graph", description)

    reparsed = json.loads(json.dumps(entry))
    assert reparsed["log_agent"]["path"] == "./agent.py:graph"
    assert json.loads(reparsed["log_agent"]["description"])["scope"] == "Application logs for the X business infra"


def test_to_langserve_graphs_value_round_trips_through_adapter() -> None:
    description, _ = build_description(_example())
    value = to_langserve_graphs_value("log_agent", "m:g", description)

    graphs = json.loads(value)  # the LANGSERVE_GRAPHS env value is a JSON graphs map
    assert graphs["log_agent"]["path"] == "m:g"
    # The card the server would serve from this env round-trips back to a profile.
    profile = extract_agent_profile({"name": "log_agent", "description": graphs["log_agent"]["description"]})
    assert profile is not None
    assert profile.scope == "Application logs for the X business infra"


def test_profile_warnings_full_profile_is_clean() -> None:
    assert profile_warnings(_example()) == []


def test_profile_warnings_flags_each_missing_field() -> None:
    suggestions = profile_warnings({"description": "short"})
    joined = " ".join(suggestions)

    assert "description is short" in joined
    assert "no scope" in joined
    assert "no how_to_use" in joined
    assert "no examples" in joined
    assert "no freshness" in joined
    assert "not exposed on any surface" in joined
    assert "no tags" in joined


def test_profile_warnings_no_surface_warning_clears_when_visibility_set() -> None:
    suggestions = profile_warnings(_example() | {"visibility": {"ui": True}})

    assert not any("not exposed on any surface" in s for s in suggestions)


# ── Click commands ───────────────────────────────────────────────────────────


def test_validate_accepts_a_full_profile() -> None:
    result = CliRunner().invoke(agent_profile, ["validate"], input=json.dumps(_example()))

    assert result.exit_code == 0
    assert "✓ profile is valid" in result.output
    assert "no completeness suggestions" in result.output


def test_validate_reports_suggestions_for_thin_profile() -> None:
    result = CliRunner().invoke(agent_profile, ["validate"], input=json.dumps({"description": "short"}))

    assert result.exit_code == 0  # suggestions are not errors
    assert "suggestion: no scope" in result.output


def test_validate_strict_exits_nonzero_on_suggestions() -> None:
    result = CliRunner().invoke(agent_profile, ["validate", "--strict"], input=json.dumps({"description": "short"}))

    assert result.exit_code == 1


def test_validate_rejects_invalid_profile() -> None:
    result = CliRunner().invoke(agent_profile, ["validate"], input=json.dumps({"description": "x" * 5000}))

    assert result.exit_code != 0
    assert "invalid profile" in result.output


def test_validate_rejects_bad_json() -> None:
    result = CliRunner().invoke(agent_profile, ["validate"], input="{not json")

    assert result.exit_code != 0
    assert "not valid JSON" in result.output


def test_build_emits_bare_description_string() -> None:
    result = CliRunner().invoke(agent_profile, ["build"], input=json.dumps(_example()))

    assert result.exit_code == 0
    assert json.loads(result.output.strip())["scope"] == "Application logs for the X business infra"


def test_build_langgraph_json_mode() -> None:
    result = CliRunner().invoke(
        agent_profile,
        ["build", "--langgraph-json", "--name", "log_agent", "--path", "./a.py:g"],
        input=json.dumps(_example()),
    )

    assert result.exit_code == 0
    entry = json.loads(result.output)
    assert entry["log_agent"]["path"] == "./a.py:g"
    assert json.loads(entry["log_agent"]["description"])["freshness"] == "every 15 minutes"


def test_build_langgraph_json_requires_name_and_path() -> None:
    result = CliRunner().invoke(agent_profile, ["build", "--langgraph-json"], input=json.dumps(_example()))

    assert result.exit_code != 0
    assert "requires --name and --path" in result.output


def test_build_warns_on_unknown_key() -> None:
    result = CliRunner().invoke(agent_profile, ["build"], input=json.dumps(_example() | {"scopee": "typo"}))

    assert result.exit_code == 0
    assert "ignoring unknown field 'scopee'" in result.output


def test_schema_command_prints_json_schema() -> None:
    result = CliRunner().invoke(agent_profile, ["schema"])

    assert result.exit_code == 0
    printed = json.loads(result.output)
    assert printed["type"] == "object"
    assert "description" in printed["properties"]


def test_example_flat_prints_valid_bare_card() -> None:
    result = CliRunner().invoke(agent_profile, ["example", "--flat"])

    assert result.exit_code == 0
    example = json.loads(result.output)
    assert not is_manifest(example)
    description, unknown = build_description(example)
    assert unknown == []
    assert json.loads(description)["description"]


def test_validate_reports_missing_description_as_invalid_profile() -> None:
    result = CliRunner().invoke(agent_profile, ["validate"], input=json.dumps({"scope": "x"}))

    assert result.exit_code != 0
    assert "invalid profile" in result.output  # clean error, not a raw traceback
    assert not isinstance(result.exception, TypeError)


# ── LANGSERVE_GRAPHS env (deploy without `langgraph build`) ──────────────────


def test_build_langserve_env_emits_dockerfile_line() -> None:
    result = CliRunner().invoke(
        agent_profile,
        ["build", "--langserve-env", "--name", "log_agent", "--path", "m:g"],
        input=json.dumps(_example()),
    )

    assert result.exit_code == 0
    line = result.output.strip().splitlines()[-1]
    assert line.startswith("ENV LANGSERVE_GRAPHS='")
    assert line.endswith("'")
    inner = line[len("ENV LANGSERVE_GRAPHS='") : -1]
    graphs = json.loads(inner)  # the embedded env value is valid JSON
    assert graphs["log_agent"]["path"] == "m:g"
    assert json.loads(graphs["log_agent"]["description"])["visibility"] == {"orchestrator": True, "ui": False}


def test_build_langserve_env_requires_name_and_path() -> None:
    result = CliRunner().invoke(agent_profile, ["build", "--langserve-env"], input=json.dumps(_example()))

    assert result.exit_code != 0
    assert "requires --name and --path" in result.output


def test_build_rejects_both_output_flags() -> None:
    result = CliRunner().invoke(
        agent_profile,
        ["build", "--langgraph-json", "--langserve-env", "--name", "x", "--path", "m:g"],
        input=json.dumps(_example()),
    )

    assert result.exit_code != 0
    assert "not both" in result.output


def test_to_dockerfile_env_line_escapes_apostrophes_as_json_unicode() -> None:
    # BuildKit does no escape processing inside a single-quoted ENV value, so a
    # literal ' would terminate the string ("unexpected end of statement"). The
    # value is JSON, so the apostrophe becomes the ' escape instead.
    line = to_dockerfile_env_line('{"a": "it\'s"}')

    assert line == 'ENV LANGSERVE_GRAPHS=\'{"a": "it\\u0027s"}\''
    inner = line[len("ENV LANGSERVE_GRAPHS='") : -1]
    assert "'" not in inner  # nothing left to terminate the quoted string
    assert json.loads(inner) == {"a": "it's"}  # ...and the server decodes it back


def test_to_dockerfile_env_line_is_unchanged_without_apostrophes() -> None:
    # Byte-compatibility with `langgraph build` is preserved for ordinary cards.
    value = '{"log_agent": {"path": "m:g", "description": "plain"}}'

    assert to_dockerfile_env_line(value) == f"ENV LANGSERVE_GRAPHS='{value}'"


def test_build_langserve_env_survives_an_apostrophe_in_a_card_field() -> None:
    # An apostrophe is ordinary English prose ("the agent's own logs") — it must
    # produce a buildable ENV line, not a warning telling the user to restructure.
    profile = _example() | {"scope": "the agent's own logs", "description": "Don't use this for metrics."}
    result = CliRunner().invoke(
        agent_profile,
        ["build", "--langserve-env", "--name", "x", "--path", "m:g"],
        input=json.dumps(profile),
    )

    assert result.exit_code == 0
    assert "warning" not in result.output.lower()
    line = result.output.strip().splitlines()[-1]
    inner = line[len("ENV LANGSERVE_GRAPHS='") : -1]
    assert "'" not in inner
    # The card round-trips through the env value with apostrophes intact.
    graphs = json.loads(inner)
    profile_back = extract_agent_profile({"name": "x", "description": graphs["x"]["description"]})
    assert profile_back is not None
    assert profile_back.scope == "the agent's own logs"
    assert profile_back.description == "Don't use this for metrics."


def test_build_manifest_langserve_env_escapes_apostrophes() -> None:
    manifest = _manifest()
    manifest["log_agent"]["card"]["scope"] = "the agent's own logs"
    result = CliRunner().invoke(agent_profile, ["build", "--langserve-env"], input=json.dumps(manifest))

    assert result.exit_code == 0
    line = result.output.strip().splitlines()[-1]
    inner = line[len("ENV LANGSERVE_GRAPHS='") : -1]
    assert "'" not in inner
    graphs = json.loads(inner)
    profile_back = extract_agent_profile({"name": "log_agent", "description": graphs["log_agent"]["description"]})
    assert profile_back is not None
    assert profile_back.scope == "the agent's own logs"


# ── YAML authoring input ─────────────────────────────────────────────────────


def test_build_reads_yaml_via_format_flag() -> None:
    result = CliRunner().invoke(agent_profile, ["build", "--format", "yaml"], input=yaml.safe_dump(_example()))

    assert result.exit_code == 0
    # Wire output is always JSON regardless of input format.
    assert json.loads(result.output.strip())["freshness"] == "every 15 minutes"


def test_build_detects_yaml_by_file_extension(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "capability.yaml"
    path.write_text(yaml.safe_dump(_example()), encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", str(path)])

    assert result.exit_code == 0
    assert json.loads(result.output.strip())["scope"] == "Application logs for the X business infra"


def test_validate_accepts_yaml_input() -> None:
    result = CliRunner().invoke(agent_profile, ["validate", "--format", "yaml"], input=yaml.safe_dump(_example()))

    assert result.exit_code == 0
    assert "✓ profile is valid" in result.output


def test_example_flat_yaml_emits_round_trippable_yaml() -> None:
    result = CliRunner().invoke(agent_profile, ["example", "--flat", "--yaml"])

    assert result.exit_code == 0
    data = yaml.safe_load(result.output)
    description, unknown = build_description(data)
    assert unknown == []
    assert json.loads(description)["visibility"] == {"orchestrator": True, "ui": False}


# ── top-level `sta` group wiring ─────────────────────────────────────────────


def test_top_level_group_registers_agent_profile() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "agent-profile" in result.output


def test_top_level_agent_profile_schema_reachable() -> None:
    result = CliRunner().invoke(main, ["agent-profile", "schema"])

    assert result.exit_code == 0
    assert json.loads(result.output)["type"] == "object"


# ── manifest form (graph key → {path, card}) ─────────────────────────────────


def test_is_manifest_distinguishes_shapes() -> None:
    assert is_manifest(_manifest()) is True
    assert is_manifest(_multi_manifest()) is True
    assert is_manifest(_example()) is False  # a flat card is not a manifest
    assert is_manifest({}) is False


def test_parse_manifest_returns_entries_in_order() -> None:
    entries = parse_manifest(_multi_manifest())

    assert [e.name for e in entries] == ["log_agent", "metric_agent"]
    assert entries[0].path == "./log_agent.py:graph"
    assert entries[0].card["description"].startswith("Searches application logs and answers questions about them.")


def test_parse_manifest_rejects_non_mapping_card() -> None:
    with pytest.raises(ValueError, match="'card' must be a mapping"):
        parse_manifest({"log_agent": {"path": "m:g", "card": "oops"}})


def test_to_langgraph_graphs_merges_entries() -> None:
    graphs = to_langgraph_graphs([("a", "m:g", '{"description":"x"}'), ("b", "n:h", '{"description":"y"}')])

    assert set(graphs) == {"a", "b"}
    assert graphs["a"] == {"path": "m:g", "description": '{"description":"x"}'}


def test_build_manifest_emits_graphs_map_by_default() -> None:
    # A manifest carries name+path, so the default output is the graphs map — no flags.
    result = CliRunner().invoke(agent_profile, ["build"], input=json.dumps(_manifest()))

    assert result.exit_code == 0
    graphs = json.loads(result.output)
    assert graphs["log_agent"]["path"] == "./log_agent.py:graph"
    # The embedded description round-trips back through the card contract.
    profile = extract_agent_profile({"name": "log_agent", "description": graphs["log_agent"]["description"]})
    assert profile is not None
    assert profile.scope == "Application logs for the X business infra"


def test_build_manifest_langserve_env_covers_all_agents() -> None:
    result = CliRunner().invoke(agent_profile, ["build", "--langserve-env"], input=json.dumps(_multi_manifest()))

    assert result.exit_code == 0
    graphs, spans = extract_langserve_graphs(result.output)
    assert len(spans) == 1  # one directive...
    directive = result.output[spans[0][0] : spans[0][1]]
    assert "\n" in directive  # ...rendered multi-line (one graph per line) for readability
    assert set(graphs) == {"log_agent", "metric_agent"}
    profile = extract_agent_profile({"name": "metric_agent", "description": graphs["metric_agent"]["description"]})
    assert profile is not None
    assert profile.description == "Answers questions about metrics and dashboards."


def test_build_manifest_missing_path_errors_with_agent_name() -> None:
    manifest = {"log_agent": {"card": dict(_EXAMPLE_PROFILE)}}
    result = CliRunner().invoke(agent_profile, ["build"], input=json.dumps(manifest))

    assert result.exit_code != 0
    assert "log_agent" in result.output
    assert "missing 'path'" in result.output


def test_build_manifest_reports_invalid_card_with_agent_name() -> None:
    manifest = {"log_agent": {"path": "m:g", "card": {"scope": "no description supplied"}}}
    result = CliRunner().invoke(agent_profile, ["build"], input=json.dumps(manifest))

    assert result.exit_code != 0
    assert "log_agent" in result.output
    assert "invalid card" in result.output
    assert not isinstance(result.exception, TypeError)


def test_build_manifest_single_agent_path_override() -> None:
    result = CliRunner().invoke(agent_profile, ["build", "--path", "pkg.mod:graph"], input=json.dumps(_manifest()))

    assert result.exit_code == 0
    assert json.loads(result.output)["log_agent"]["path"] == "pkg.mod:graph"


def test_build_manifest_override_rejected_for_multi_agent() -> None:
    result = CliRunner().invoke(agent_profile, ["build", "--name", "x"], input=json.dumps(_multi_manifest()))

    assert result.exit_code != 0
    assert "single-agent manifest" in result.output


def test_validate_manifest_reports_per_agent_and_is_clean() -> None:
    result = CliRunner().invoke(agent_profile, ["validate"], input=json.dumps(_manifest()))

    assert result.exit_code == 0
    assert "✓ log_agent: card is valid" in result.output
    assert "no completeness suggestions" in result.output


def test_validate_manifest_flags_missing_path_and_thin_card() -> None:
    manifest = {"thin_agent": {"card": {"description": "short"}}}
    result = CliRunner().invoke(agent_profile, ["validate"], input=json.dumps(manifest))

    assert result.exit_code == 0  # suggestions are not errors
    assert "thin_agent: suggestion: no path" in result.output
    assert "thin_agent: suggestion: no scope" in result.output


def test_validate_manifest_strict_exits_nonzero_on_suggestions() -> None:
    manifest = {"thin_agent": {"path": "m:g", "card": {"description": "short"}}}
    result = CliRunner().invoke(agent_profile, ["validate", "--strict"], input=json.dumps(manifest))

    assert result.exit_code == 1


def test_build_manifest_detects_yaml_by_extension(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "log_agent.card.yaml"
    path.write_text(yaml.safe_dump(_manifest(), sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", str(path)])

    assert result.exit_code == 0
    assert json.loads(result.output)["log_agent"]["path"] == "./log_agent.py:graph"


def test_example_defaults_to_manifest_and_round_trips_through_build() -> None:
    result = CliRunner().invoke(agent_profile, ["example"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert is_manifest(data)  # manifest is the default shape
    built = CliRunner().invoke(agent_profile, ["build"], input=json.dumps(data))
    assert built.exit_code == 0
    assert json.loads(built.output)["log_agent"]["path"] == "./log_agent.py:graph"


def test_example_manifest_yaml_is_round_trippable() -> None:
    result = CliRunner().invoke(agent_profile, ["example", "--yaml"])

    assert result.exit_code == 0
    data = yaml.safe_load(result.output)
    assert is_manifest(data)
    entries = parse_manifest(data)
    assert entries[0].card["visibility"] == {"orchestrator": True, "ui": False}


def test_example_yaml_ships_comments_and_block_scalars() -> None:
    # `yaml.safe_dump` can round-trip neither, so the template is authored by hand;
    # this pins that the annotated form is what a producer actually receives.
    result = CliRunner().invoke(agent_profile, ["example", "--yaml"])

    assert result.exit_code == 0
    assert "description: >-" in result.output  # folded scalar for prose
    assert "how_to_use: |-" in result.output  # literal scalar for steps
    assert result.output.lstrip().startswith("#")  # leading explanatory comment
    assert "# Which surfaces you opt into" in result.output


def test_example_flat_yaml_ships_comments_and_block_scalars() -> None:
    result = CliRunner().invoke(agent_profile, ["example", "--flat", "--yaml"])

    assert result.exit_code == 0
    assert not is_manifest(yaml.safe_load(result.output))
    assert "description: >-" in result.output
    assert "how_to_use: |-" in result.output


def test_example_yaml_and_json_outputs_cannot_drift() -> None:
    # The YAML template is the source of truth; the JSON output is parsed from it.
    yaml_out = CliRunner().invoke(agent_profile, ["example", "--yaml"])
    json_out = CliRunner().invoke(agent_profile, ["example"])

    assert yaml.safe_load(yaml_out.output) == json.loads(json_out.output)

    flat_yaml = CliRunner().invoke(agent_profile, ["example", "--flat", "--yaml"])
    flat_json = CliRunner().invoke(agent_profile, ["example", "--flat"])

    assert yaml.safe_load(flat_yaml.output) == json.loads(flat_json.output)
    # ...and the manifest's nested card is exactly the flat card.
    assert yaml.safe_load(yaml_out.output)["log_agent"]["card"] == yaml.safe_load(flat_yaml.output)


# ── root include-manifest (a list of card-file paths) ────────────────────────


def _write_card_file(directory, filename: str, graph_key: str, description: str | None = None) -> None:  # noqa: ANN001
    """Write a self-contained single-agent manifest ({graph_key: {path, card}})."""
    card = dict(_EXAMPLE_PROFILE)
    if description is not None:
        card = card | {"description": description}
    manifest = {graph_key: {"path": f"./{graph_key}.py:graph", "card": card}}
    (directory / filename).write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def test_is_root_manifest_detects_shapes() -> None:
    assert is_root_manifest(["a.yaml", "b.yaml"]) is True
    assert is_root_manifest({"cards": ["a.yaml"]}) is True
    assert is_root_manifest([]) is False  # empty is not a root manifest
    assert is_root_manifest({"cards": [1, 2]}) is False  # entries must be strings
    assert is_root_manifest(_manifest()) is False  # a manifest is not a root manifest
    assert is_root_manifest(_example()) is False  # nor is a flat card


def test_parse_root_manifest_accepts_both_forms() -> None:
    assert parse_root_manifest(["a.yaml", "b.yaml"]) == ["a.yaml", "b.yaml"]
    assert parse_root_manifest({"cards": ["a.yaml"]}) == ["a.yaml"]


def test_parse_root_manifest_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_root_manifest([])


def test_build_root_manifest_bundles_multiple_card_files(tmp_path) -> None:  # noqa: ANN001
    _write_card_file(tmp_path, "log.yaml", "log_agent")
    _write_card_file(tmp_path, "metric.yaml", "metric_agent", description="Answers questions about metrics and dashboards.")
    root = tmp_path / "agent_profile.yaml"
    root.write_text(yaml.safe_dump({"cards": ["log.yaml", "metric.yaml"]}), encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", str(root)])

    assert result.exit_code == 0, result.output
    graphs = json.loads(result.output)
    assert set(graphs) == {"log_agent", "metric_agent"}
    assert graphs["log_agent"]["path"] == "./log_agent.py:graph"
    profile = extract_agent_profile({"name": "metric_agent", "description": graphs["metric_agent"]["description"]})
    assert profile is not None and profile.description == "Answers questions about metrics and dashboards."


def test_build_root_manifest_bare_list_form(tmp_path) -> None:  # noqa: ANN001
    _write_card_file(tmp_path, "log.yaml", "log_agent")
    root = tmp_path / "agent_profile.yaml"
    root.write_text(yaml.safe_dump(["log.yaml"]), encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", str(root)])

    assert result.exit_code == 0, result.output
    assert "log_agent" in json.loads(result.output)


def test_root_manifest_rejects_flat_card_reference(tmp_path) -> None:  # noqa: ANN001
    (tmp_path / "flat.yaml").write_text(yaml.safe_dump(dict(_EXAMPLE_PROFILE)), encoding="utf-8")
    root = tmp_path / "agent_profile.yaml"
    root.write_text(yaml.safe_dump(["flat.yaml"]), encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", str(root)])

    assert result.exit_code != 0
    assert "self-contained manifest" in result.output


def test_root_manifest_rejects_duplicate_graph_key(tmp_path) -> None:  # noqa: ANN001
    _write_card_file(tmp_path, "a.yaml", "log_agent")
    _write_card_file(tmp_path, "b.yaml", "log_agent")
    root = tmp_path / "agent_profile.yaml"
    root.write_text(yaml.safe_dump(["a.yaml", "b.yaml"]), encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", str(root)])

    assert result.exit_code != 0
    assert "duplicate graph key 'log_agent'" in result.output


def test_root_manifest_missing_file_errors(tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / "agent_profile.yaml"
    root.write_text(yaml.safe_dump(["nope.yaml"]), encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", str(root)])

    assert result.exit_code != 0
    assert "not found" in result.output


def test_validate_root_manifest_reports_per_agent(tmp_path) -> None:  # noqa: ANN001
    _write_card_file(tmp_path, "log.yaml", "log_agent")
    root = tmp_path / "agent_profile.yaml"
    root.write_text(yaml.safe_dump(["log.yaml"]), encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["validate", str(root)])

    assert result.exit_code == 0, result.output
    assert "✓ log_agent: card is valid" in result.output


# ── merge core ───────────────────────────────────────────────────────────────


def test_merge_graphs_preserve_is_the_default() -> None:
    existing = {"a": {"path": "a:g", "description": "x"}}
    new = {"b": {"path": "b:g", "description": "y"}}

    result = merge_graphs(existing, new, replace=False)

    assert set(result.merged) == {"a", "b"}
    assert result.added == ["b"]
    assert result.overwritten == []
    assert result.preserved == ["a"]
    assert result.dropped == []


def test_merge_graphs_overwrites_same_key() -> None:
    result = merge_graphs({"a": {"path": "old:g", "description": "old"}}, {"a": {"path": "new:g", "description": "new"}}, replace=False)

    assert result.merged["a"]["path"] == "new:g"
    assert result.overwritten == ["a"]
    assert result.added == []


def test_merge_graphs_replace_drops_extras() -> None:
    existing = {"a": {"path": "a:g", "description": "x"}, "b": {"path": "b:g", "description": "y"}}
    result = merge_graphs(existing, {"a": {"path": "a2:g", "description": "z"}}, replace=True)

    assert set(result.merged) == {"a"}
    assert result.dropped == ["b"]
    assert result.preserved == []


def test_merge_into_langgraph_json_preserves_other_keys_and_order() -> None:
    doc = {"dependencies": ["."], "graphs": {"a": {"path": "a:g", "description": "x"}}, "env": ".env"}

    updated, result = merge_into_langgraph_json(doc, {"b": {"path": "b:g", "description": "y"}}, replace=False)

    assert updated["dependencies"] == ["."]
    assert updated["env"] == ".env"
    assert set(updated["graphs"]) == {"a", "b"}
    assert list(updated) == ["dependencies", "graphs", "env"]  # key order preserved
    assert result.preserved == ["a"]


def test_merge_into_langgraph_json_creates_graphs_block_if_absent() -> None:
    updated, _ = merge_into_langgraph_json({"dependencies": ["."]}, {"a": {"path": "a:g", "description": "x"}}, replace=False)

    assert updated["graphs"]["a"]["path"] == "a:g"


def test_merge_into_langgraph_json_rejects_non_object_graphs() -> None:
    with pytest.raises(ValueError, match="not an object"):
        merge_into_langgraph_json({"graphs": [1, 2]}, {"a": {}}, replace=False)


# ── Dockerfile LANGSERVE_GRAPHS merge ────────────────────────────────────────


def test_extract_langserve_graphs_parses_value_and_span() -> None:
    text = f"FROM x\n{to_dockerfile_env_line(json.dumps({'a': {'path': 'a:g', 'description': 'x'}}))}\n"

    graphs, spans = extract_langserve_graphs(text)

    assert graphs == {"a": {"path": "a:g", "description": "x"}}
    assert len(spans) == 1


def test_extract_langserve_graphs_handles_multiline_value() -> None:
    # `langgraph build` / `langgraph dockerfile` emit one graph per line inside the
    # quotes — the value spans several physical lines. It must still be recognised.
    text = 'FROM base\nENV LANGSERVE_GRAPHS=\'{"clarify": "m:c",\n"topology": "m:t"}\'\nEXPOSE 8000\n'

    graphs, spans = extract_langserve_graphs(text)

    assert set(graphs) == {"clarify", "topology"}
    assert len(spans) == 1  # one directive, even though it spans two lines


def test_merge_dockerfile_merges_into_multiline_value_not_appends() -> None:
    # Regression: a multi-line existing directive must be MERGED into, never
    # shadowed by a second appended line (later ENV wins → silent graph loss).
    text = 'FROM base\nENV LANGSERVE_GRAPHS=\'{"clarify": "m:c",\n"topology": "m:t"}\'\nEXPOSE 8000\n'

    out, result = merge_langserve_env_into_dockerfile(text, {"template_agent": {"path": "t:g", "description": "x"}}, replace=False)

    assert out.count("ENV LANGSERVE_GRAPHS=") == 1  # merged in place, NOT duplicated
    graphs, spans = extract_langserve_graphs(out)
    assert set(graphs) == {"clarify", "topology", "template_agent"}
    assert set(result.preserved) == {"clarify", "topology"}
    assert out.startswith("FROM base\n")
    assert "EXPOSE 8000" in out
    # the merged directive stays multi-line (one graph per line), not collapsed
    directive = out[spans[0][0] : spans[0][1]]
    assert directive.count("\n") == 2  # three graphs → three lines → two internal breaks
    # re-merging the same graph is idempotent (no drift on the multi-line form)
    again, _ = merge_langserve_env_into_dockerfile(out, {"template_agent": {"path": "t:g", "description": "x"}}, replace=False)
    assert again == out


def test_extract_langserve_graphs_none_when_absent() -> None:
    assert extract_langserve_graphs("FROM x\n") == (None, [])


def test_extract_langserve_graphs_skips_commented_line() -> None:
    text = f"# {to_dockerfile_env_line(json.dumps({'a': {}}))}\nFROM x\n"

    assert extract_langserve_graphs(text) == (None, [])


def test_extract_langserve_graphs_raises_on_malformed() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        extract_langserve_graphs(f"{to_dockerfile_env_line('{not json}')}\n")


def test_merge_dockerfile_preserves_existing_graphs() -> None:
    existing = to_dockerfile_env_line(json.dumps({"graph_a": {"path": "a:g", "description": "x"}}))
    text = f"FROM base\n{existing}\n"

    out, result = merge_langserve_env_into_dockerfile(text, {"log_agent": {"path": "l:g", "description": "y"}}, replace=False)

    graphs, _ = extract_langserve_graphs(out)
    assert set(graphs) == {"graph_a", "log_agent"}
    assert result.preserved == ["graph_a"]
    assert out.startswith("FROM base\n")


def test_merge_dockerfile_appends_when_absent() -> None:
    out, _ = merge_langserve_env_into_dockerfile("FROM base\n", {"a": {"path": "a:g", "description": "x"}}, replace=False)

    assert "ENV LANGSERVE_GRAPHS='" in out
    assert out.startswith("FROM base\n")


def test_merge_dockerfile_is_idempotent() -> None:
    new = {"a": {"path": "a:g", "description": "x"}}
    once, _ = merge_langserve_env_into_dockerfile("FROM base\n", new, replace=False)
    twice, _ = merge_langserve_env_into_dockerfile(once, new, replace=False)

    assert once == twice


def test_merge_dockerfile_replace_drops_extras() -> None:
    existing = to_dockerfile_env_line(json.dumps({"graph_a": {"path": "a:g", "description": "x"}}))

    out, result = merge_langserve_env_into_dockerfile(f"{existing}\n", {"log": {"path": "l:g", "description": "y"}}, replace=True)

    graphs, _ = extract_langserve_graphs(out)
    assert set(graphs) == {"log"}
    assert result.dropped == ["graph_a"]


def test_detect_destination() -> None:
    assert detect_destination("langgraph.json") == "langgraph"
    assert detect_destination("./path/to/langgraph.json") == "langgraph"
    assert detect_destination("Dockerfile") == "dockerfile"
    assert detect_destination("Dockerfile.prod") == "dockerfile"
    assert detect_destination("prod.dockerfile") == "dockerfile"
    with pytest.raises(ValueError, match="cannot infer"):
        detect_destination("out.txt")


# ── build --into (write into an existing destination, merge-preserve) ─────────


def test_build_into_langgraph_json_merges_and_preserves(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "langgraph.json"
    target.write_text(json.dumps({"dependencies": ["."], "graphs": {"graph_a": {"path": "a:g", "description": "x"}}}), encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", "--into", str(target)], input=json.dumps(_manifest()))

    assert result.exit_code == 0, result.output
    doc = json.loads(target.read_text())
    assert set(doc["graphs"]) == {"graph_a", "log_agent"}  # existing graph_a survives
    assert doc["dependencies"] == ["."]
    assert "preserved: graph_a" in result.output


def test_build_into_langgraph_json_is_idempotent(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "langgraph.json"
    target.write_text(json.dumps({"graphs": {}}), encoding="utf-8")

    first = CliRunner().invoke(agent_profile, ["build", "--into", str(target)], input=json.dumps(_manifest()))
    assert first.exit_code == 0, first.output
    after_first = target.read_text()
    second = CliRunner().invoke(agent_profile, ["build", "--into", str(target)], input=json.dumps(_manifest()))
    assert second.exit_code == 0

    assert target.read_text() == after_first


def test_build_into_dockerfile_preserves_graphs_not_in_manifest(tmp_path) -> None:  # noqa: ANN001
    # The reviewed scenario: LANGSERVE_GRAPHS already carries graphs the manifest
    # does not mention — they must survive the merge.
    existing = to_dockerfile_env_line(json.dumps({"graph_a": {"path": "a:g", "description": "x"}}))
    target = tmp_path / "Dockerfile"
    target.write_text(f"FROM base\n{existing}\n", encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", "--langserve-env", "--into", str(target)], input=json.dumps(_manifest()))

    assert result.exit_code == 0, result.output
    graphs, _ = extract_langserve_graphs(target.read_text())
    assert set(graphs) == {"graph_a", "log_agent"}
    assert target.read_text().startswith("FROM base\n")


def test_build_into_multiline_dockerfile_merges_not_duplicates(tmp_path) -> None:  # noqa: ANN001
    # Regression for a `langgraph build`-generated Dockerfile: the existing
    # LANGSERVE_GRAPHS spans multiple lines. `--into` must merge into it, not append
    # a second directive that would shadow the originals at runtime.
    target = tmp_path / "Dockerfile.agents"
    target.write_text(
        'FROM base\nENV LANGSERVE_GRAPHS=\'{"clarify": "m:c",\n"topology": "m:t"}\'\nEXPOSE 8000\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(agent_profile, ["build", "--langserve-env", "--into", str(target)], input=json.dumps(_manifest()))

    assert result.exit_code == 0, result.output
    content = target.read_text()
    assert content.count("ENV LANGSERVE_GRAPHS=") == 1  # merged, not duplicated
    graphs, spans = extract_langserve_graphs(content)
    assert set(graphs) == {"clarify", "topology", "log_agent"}
    assert "preserved: clarify, topology" in result.output
    assert "\n" in content[spans[0][0] : spans[0][1]]  # stays multi-line for readability


def test_build_into_replace_makes_manifest_authoritative(tmp_path) -> None:  # noqa: ANN001
    existing = to_dockerfile_env_line(json.dumps({"graph_a": {"path": "a:g", "description": "x"}}))
    target = tmp_path / "Dockerfile"
    target.write_text(f"{existing}\n", encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", "--into", str(target), "--as", "dockerfile", "--replace"], input=json.dumps(_manifest()))

    assert result.exit_code == 0, result.output
    graphs, _ = extract_langserve_graphs(target.read_text())
    assert set(graphs) == {"log_agent"}  # graph_a dropped
    assert "dropped" in result.output


def test_build_into_dry_run_does_not_write(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "langgraph.json"
    original = json.dumps({"graphs": {"graph_a": {"path": "a:g", "description": "x"}}})
    target.write_text(original, encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", "--into", str(target), "--dry-run"], input=json.dumps(_manifest()))

    assert result.exit_code == 0, result.output
    assert target.read_text() == original  # unchanged
    assert "log_agent" in result.output  # merged preview printed


def test_build_into_missing_file_without_create_errors(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "langgraph.json"

    result = CliRunner().invoke(agent_profile, ["build", "--into", str(target)], input=json.dumps(_manifest()))

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_build_into_create_makes_the_file(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "langgraph.json"

    result = CliRunner().invoke(agent_profile, ["build", "--into", str(target), "--create"], input=json.dumps(_manifest()))

    assert result.exit_code == 0, result.output
    assert json.loads(target.read_text())["graphs"]["log_agent"]["path"] == "./log_agent.py:graph"


def test_build_into_malformed_destination_errors_and_never_clobbers(tmp_path) -> None:  # noqa: ANN001
    original = f"{to_dockerfile_env_line('{not json}')}\n"
    target = tmp_path / "Dockerfile"
    target.write_text(original, encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", "--into", str(target), "--as", "dockerfile"], input=json.dumps(_manifest()))

    assert result.exit_code != 0
    assert "not valid JSON" in result.output
    assert target.read_text() == original  # the malformed file is left untouched


def test_build_into_flat_card_requires_name_and_path(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "langgraph.json"
    target.write_text(json.dumps({"graphs": {}}), encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", "--into", str(target)], input=json.dumps(_example()))

    assert result.exit_code != 0
    assert "requires --name and --path" in result.output


def test_build_into_ambiguous_filename_needs_as(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "output.txt"
    target.write_text("", encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", "--into", str(target)], input=json.dumps(_manifest()))

    assert result.exit_code != 0
    assert "cannot infer" in result.output.lower()


def test_build_into_conflicting_shape_and_filename_errors(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "Dockerfile"
    target.write_text("FROM base\n", encoding="utf-8")

    result = CliRunner().invoke(agent_profile, ["build", "--langgraph-json", "--into", str(target)], input=json.dumps(_manifest()))

    assert result.exit_code != 0
    assert "ambiguous" in result.output.lower()

-------

