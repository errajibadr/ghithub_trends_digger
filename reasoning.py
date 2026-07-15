📇 New in TWIN: Agent Capability Cards

TL;DR — You can now describe your agent with a small YAML "card" and register it into TWIN in one command — no hand-editing langgraph.json or Dockerfiles, no copy-paste wiring.

What's an agent card?
A short, vendor-neutral YAML file that lives next to your agent's code and describes it: what it does, its scope, how to use it, example queries, and where it should be visible (UI, orchestrator). One card = one self-described agent, in the same format for every team.

What is it used for today?
- 🗂️ Registration — the sta agent-profile CLI validates your card and writes the wiring for you: it bundles one or many cards and merges them directly into langgraph.json or your Dockerfile's LANGSERVE_GRAPHS. Existing graphs are preserved, re-runs are idempotent — so your agent gets deployed and shows up in the TWIN UI with its description.
- 📦 Self-describing agents — the card travels with your deployment (embedded in the graph description, exposed via the standard A2A agent-card endpoint), so TWIN always has an up-to-date description of your agent straight from the source.

What's next 🔭
Cards are the foundation for orchestrator integration: in a future phase, the TWIN orchestrator will read them to advertise your agent and route requests to it like a first-party subagent.

Get started in 3 steps
1. sta agent-profile example > my_agent.card.yaml — scaffold a card next to your graph
2. Fill in description / scope / examples (sta agent-profile validate to check it)
3. sta agent-profile build my_agent.card.yaml --into langgraph.json (or --into Dockerfile)

📖 Full guide: [link to Confluence page — TBD]

Questions or want help writing your first card → reply in thread 🧵