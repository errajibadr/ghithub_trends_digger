docs/orchestrator/assets/persona-presets-journey.html
----
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Persona Presets Journey</title>
<style>
:root {
  --bg: #f8fafc; --panel: #ffffff; --fg: #0f172a; --muted: #475569; --line: #cbd5e1;
  --accent: #0f766e; --accent-fg: #ffffff; --focus: #2563eb; --code-bg: #f1f5f9; --warn-bg: #fff7ed; --warn-line: #fdba74;
  --check-bg: #ecfdf5; --check-line: #6ee7b7;
  --arrow: #94a3b8; --arrow-emphasis: #059669; --security-stroke: #e11d48; --database-stroke: #7c3aed;
  --lane-fill: rgba(248, 250, 252, 0.65); --lane-stroke: #cbd5e1; --text-dim: #94a3b8; --mask: #ffffff;
  --external-fill: rgba(148, 163, 184, 0.18); --external-stroke: #64748b; --text: #0f172a; --text-muted: #64748b;
  --cloud-fill: rgba(251, 191, 36, 0.18); --cloud-stroke: #d97706; --backend-fill: rgba(52, 211, 153, 0.18); --backend-stroke: #059669;
  --frontend-fill: rgba(34, 211, 238, 0.15); --frontend-stroke: #0891b2; --database-fill: rgba(167, 139, 250, 0.2);
  --security-fill: rgba(251, 113, 133, 0.15); --messagebus-fill: rgba(251, 146, 60, 0.15); --messagebus-stroke: #ea580c; --grid: #e2e8f0;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #020617; --panel: #0b1222; --fg: #f1f5f9; --muted: #94a3b8; --line: #334155;
    --accent: #2dd4bf; --accent-fg: #042f2e; --focus: #60a5fa; --code-bg: #111a2e; --warn-bg: #2a1708; --warn-line: #9a3412;
    --check-bg: #052e22; --check-line: #047857;
    --arrow: #64748b; --arrow-emphasis: #34d399; --security-stroke: #fb7185; --database-stroke: #a78bfa;
    --lane-fill: rgba(15, 23, 42, 0.22); --lane-stroke: #334155; --text-dim: #475569; --mask: #0f172a;
    --external-fill: rgba(30, 41, 59, 0.5); --external-stroke: #94a3b8; --text: #ffffff; --text-muted: #94a3b8;
    --cloud-fill: rgba(120, 53, 15, 0.3); --cloud-stroke: #fbbf24; --backend-fill: rgba(6, 78, 59, 0.4); --backend-stroke: #34d399;
    --frontend-fill: rgba(8, 51, 68, 0.4); --frontend-stroke: #22d3ee; --database-fill: rgba(76, 29, 149, 0.4);
    --security-fill: rgba(136, 19, 55, 0.4); --messagebus-fill: rgba(251, 146, 60, 0.3); --messagebus-stroke: #fb923c; --grid: #1e293b;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.pj-wrap { max-width: 1240px; margin: 0 auto; padding: 24px 16px 64px; }
.pj-head h1 { margin: 0 0 4px; font-size: 1.6rem; }
.pj-head p { margin: 4px 0; color: var(--muted); }
.pj-controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 16px 0 24px; }
.pj-status { color: var(--muted); font-size: 0.95rem; }
button { font: inherit; cursor: pointer; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); color: var(--fg); padding: 6px 12px; }
button.pj-primary { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
button:focus-visible, summary:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }
button[disabled] { opacity: 0.45; cursor: not-allowed; }
.pj-view { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); padding: 16px; margin: 0 0 20px; }
.pj-view h2 { margin: 0 0 4px; font-size: 1.2rem; }
.pj-view .pj-caption { margin: 0 0 12px; color: var(--muted); }
.pj-diagram { width: 100%; }
.pj-diagram svg { display: block; width: 100%; height: auto; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.pj-diagram .c-grid { stroke: var(--grid); fill: none; }
.pj-diagram .c-lane { fill: var(--lane-fill); stroke: var(--lane-stroke); stroke-dasharray: 6, 6; }
.pj-diagram .c-mask { fill: var(--mask); stroke: none; }
.pj-diagram .c-external { fill: var(--external-fill); stroke: var(--external-stroke); }
.pj-diagram .c-cloud { fill: var(--cloud-fill); stroke: var(--cloud-stroke); }
.pj-diagram .c-backend { fill: var(--backend-fill); stroke: var(--backend-stroke); }
.pj-diagram .c-frontend { fill: var(--frontend-fill); stroke: var(--frontend-stroke); }
.pj-diagram .c-database { fill: var(--database-fill); stroke: var(--database-stroke); }
.pj-diagram .c-security { fill: var(--security-fill); stroke: var(--security-stroke); }
.pj-diagram .c-messagebus { fill: var(--messagebus-fill); stroke: var(--messagebus-stroke); }
.pj-diagram .a-default { stroke: var(--arrow); fill: none; }
.pj-diagram .m-default { fill: var(--arrow); }
.pj-diagram .m-emphasis { fill: var(--arrow-emphasis); }
.pj-diagram .m-security { fill: var(--security-stroke); }
.pj-diagram .m-dashed { fill: var(--database-stroke); }
.pj-diagram .t-primary { fill: var(--text); }
.pj-diagram .t-muted { fill: var(--text-muted); }
.pj-diagram .t-dim { fill: var(--text-dim); }
.pj-diagram .semantic-sigil > * { vector-effect: non-scaling-stroke; }
.pj-diagram .semantic-sigil .sigil-fill { fill: currentColor; stroke: none; }
.pj-diagram .s-external { color: var(--external-stroke); }
.pj-diagram .s-cloud { color: var(--cloud-stroke); }
.pj-diagram .s-backend { color: var(--backend-stroke); }
.pj-diagram .s-frontend { color: var(--frontend-stroke); }
.pj-diagram .s-database { color: var(--database-stroke); }
.pj-diagram .s-security { color: var(--security-stroke); }
.pj-diagram .s-messagebus { color: var(--messagebus-stroke); }
.pj-diagram g.pj-node { cursor: pointer; }
.pj-diagram g.pj-node:focus { outline: none; }
.pj-diagram g.pj-node:focus-visible > rect:first-of-type, .pj-diagram g.pj-node:hover > rect:first-of-type { stroke: var(--focus); stroke-width: 3; }
.pj-error { color: var(--security-stroke); }
.pj-list h3 { margin: 16px 0 4px; font-size: 1rem; }
.pj-list ol { margin: 0; padding-left: 24px; }
.pj-list li { margin: 4px 0; }
.pj-list button { text-align: left; }
dialog.pj-dialog { width: min(860px, calc(100vw - 32px)); max-height: calc(100vh - 32px); border: 1px solid var(--line); border-radius: 14px; padding: 0; background: var(--panel); color: var(--fg); }
dialog.pj-dialog::backdrop { background: rgba(2, 6, 23, 0.55); }
.pj-dialog-inner { display: flex; flex-direction: column; max-height: calc(100vh - 34px); }
.pj-card-head { padding: 16px 20px 8px; border-bottom: 1px solid var(--line); }
.pj-card-head h2 { margin: 2px 0 0; font-size: 1.35rem; }
.pj-meta { color: var(--muted); font-size: 0.9rem; }
.pj-card-body { padding: 12px 20px; overflow-y: auto; }
.pj-card-body h3 { font-size: 1.02rem; margin: 18px 0 6px; }
.pj-card-foot { padding: 12px 20px; border-top: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 8px; justify-content: space-between; }
.pj-card-foot .pj-nav { display: flex; flex-wrap: wrap; gap: 8px; }
.pj-label { font-weight: 600; }
pre { background: var(--code-bg); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; overflow-x: auto; font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre; margin: 6px 0; }
.pj-table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; margin: 6px 0; }
th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: var(--code-bg); }
.pj-box { border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; margin: 10px 0; }
.pj-check { background: var(--check-bg); border-color: var(--check-line); }
.pj-warn { background: var(--warn-bg); border-color: var(--warn-line); }
.pj-badge { display: inline-block; font-size: 0.78rem; border: 1px solid var(--line); border-radius: 999px; padding: 0 8px; margin-left: 6px; color: var(--muted); }
details { border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px; margin: 10px 0; }
summary { font-weight: 600; cursor: pointer; }
.pj-snippet-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
.pj-live { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
</head>
<body>
<div class="pj-wrap" id="pj-main">
  <header class="pj-head">
    <h1 id="pj-title"></h1>
    <p id="pj-audience"></p>
    <p id="pj-outcome"></p>
  </header>
  <div class="pj-controls">
    <button type="button" class="pj-primary" id="pj-begin">Begin the guided path</button>
    <button type="button" id="pj-resume" hidden>Resume</button>
    <span class="pj-status" id="pj-status">Guided path not started. Click any node to explore its card.</span>
  </div>
  <section aria-label="Three views of the same design" id="pj-views"></section>
  <section class="pj-view pj-list" aria-labelledby="pj-list-title">
    <h2 id="pj-list-title">All steps</h2>
    <p class="pj-caption">The same cards as a list, grouped by track.</p>
    <div id="pj-list"></div>
  </section>
</div>
<dialog class="pj-dialog" id="pj-dialog" aria-labelledby="pj-card-title">
  <div class="pj-dialog-inner">
    <div class="pj-card-head">
      <div class="pj-meta" id="pj-card-meta"></div>
      <h2 id="pj-card-title"></h2>
    </div>
    <div class="pj-card-body" id="pj-card-body"></div>
    <div class="pj-card-foot">
      <div class="pj-nav" id="pj-card-nav"></div>
      <button type="button" id="pj-close">Close</button>
    </div>
  </div>
</dialog>
<div class="pj-live" aria-live="polite" id="pj-live"></div>
<script type="application/json" id="pj-data">{"content":{"title":"Persona presets for the orchestrator","audience":{"role":"twinCore engineers and team lead","prior_knowledge":"Know LangGraph, deepagents and the orchestrator code; new to Context Hub and to this design","outcome":"Can compare the three ways to hand over a persona, explain where a franchise stores a persona and who reads it, trace how one request loads it, defend the per-request rebuild, and name the gates between v1 and v2"},"detail_level":"guided","sources":[{"id":"hub-backend","kind":"inspected-document","label":"HubSkillsBackend source","location":"packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/backends/hub_backend.py"},{"id":"user-backend","kind":"inspected-document","label":"Backend composition (build_orchestrator_backend)","location":"packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/backends/user_backend.py"},{"id":"catalog","kind":"inspected-document","label":"Orchestrator factory (make_orchestrator)","location":"packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/orchestrator_catalog.py"},{"id":"probe","kind":"inspected-document","label":"Context Hub probe (read-only)","location":"probes/hub/persona_hub_probe.py"},{"id":"live-hub","kind":"tool-output","label":"Tests on LangSmith 0.18.3 with test persona repos","location":"LangSmith Context Hub"},{"id":"build-cost","kind":"tool-output","label":"Offline measurement of the orchestrator factory (static roster, fake model)","location":"orchestrator factory"},{"id":"deepagents-src","kind":"inspected-document","label":"deepagents 0.7.19 source (context_hub.py, skills.py, subagents.py)","location":"deepagents 0.7.19"},{"id":"langgraph-api-src","kind":"inspected-document","label":"langgraph-api 0.9.0 source (graph.py, route.py)","location":"langgraph-api 0.9.0"},{"id":"langsmith-src","kind":"inspected-document","label":"langsmith SDK 0.14.1 source (client.py, schemas.py)","location":"langsmith 0.14.1"}],"tracks":[{"id":"where","title":"Where it lives","display_order":["problem","options","where-personas-live","what-we-built"]},{"id":"load","title":"How it loads","display_order":["contracts","request-flow","mounted-paths"]},{"id":"decide","title":"How it runs and ships","display_order":["persona-backend","versioning","why-rebuild","risks-and-versions","roadmap"]}],"edges":[{"id":"problem-to-options","from":"problem","to":"options","role":"main"},{"id":"options-to-where","from":"options","to":"where-personas-live","role":"main"},{"id":"where-to-built","from":"where-personas-live","to":"what-we-built","role":"main"},{"id":"built-to-contracts","from":"what-we-built","to":"contracts","label":"Reuse confirmed","role":"main"},{"id":"contracts-to-request","from":"contracts","to":"request-flow","role":"main"},{"id":"request-to-paths","from":"request-flow","to":"mounted-paths","role":"main"},{"id":"paths-to-backend","from":"mounted-paths","to":"persona-backend","label":"Route traced","role":"main"},{"id":"backend-to-versioning","from":"persona-backend","to":"versioning","role":"main"},{"id":"versioning-to-rebuild","from":"versioning","to":"why-rebuild","role":"main"},{"id":"rebuild-to-risks","from":"why-rebuild","to":"risks-and-versions","role":"main"},{"id":"risks-to-roadmap","from":"risks-and-versions","to":"roadmap","role":"main"}],"nodes":[{"id":"problem","track_id":"where","detail_level":"guided","title":"The problem","objective":"State what a persona preset is and the constraints any design must meet.","summary":"twinCore owns the orchestrator, a deep agent. Each franchise owns its subagents, its own LGP workspace and a set of skills, and defines persona presets: an AGENTS.md plus skills, for example an incident manager. In the frontend a user picks either a subagent directly (exists today) or a persona, which is the orchestrator loaded with that preset.","why":"The open question is where franchises store their skills and how the orchestrator loads them without making every change depend on other teams.","prerequisites":[],"sections":[{"title":"Constraints","body":"Every later step honours them.","bullets":["Loading logic stays on the agent side: the backend sends only the persona identity.","Personas are standalone orchestrators shown in the UI; the main orchestrator does not delegate to them.","TwinShield tells which personas a user may use; the backend discovers workspaces through the platform APIs.","One LangSmith key can read every workspace.","A persona that cannot be loaded runs the plain orchestrator with a visible banner."],"collapsible":false}],"snippets":[],"checkpoint":"You can name the two things a user can pick, and the constraints any option must meet.","common_mistakes":["Treating the persona as frontend data: its files then become caller-controlled prompt content."],"source_refs":["catalog"],"assumptions":[],"links":[{"label":"Next: the three options","node_id":"options"}]},{"id":"options","track_id":"where","detail_level":"guided","title":"Three options","objective":"Compare the three ways a persona can reach the orchestrator, and know why one is chosen.","summary":"A persona can reach the orchestrator as files the backend sends in the thread state, as a persona field in the run's context or configurable that the orchestrator resolves itself, or as a dedicated orchestrator assistant whose context names the persona. The persona field is chosen; the assistant stays a compatible alternative.","why":"The choice decides who owns persona logic, where the trust boundary sits, and what has to be operated.","prerequisites":["The problem"],"sections":[{"title":"At a glance","body":"","bullets":[],"collapsible":false,"table":{"headers":["","Files in state","Persona field (chosen)","Assistant per persona"],"rows":[["Backend sends","AGENTS.md and skill files","{workspace_id, repo, version}","An assistant id"],["Content loaded by","The backend","The orchestrator, from Context Hub","The orchestrator, from Context Hub"],["Prompt written by","The caller","The franchise, through promotion","The franchise, through promotion"],["Storage","Copied into every thread","One shared snapshot per persona","One shared snapshot per persona"],["To operate","Nothing on our side","The loader only","CI/CD creating assistants on every push, recreating them on every orchestrator redeploy"],["Change persona logic in","The backend","The orchestrator","The orchestrator and the pipeline"],["Verdict","Rejected","Chosen","Alternative, compatible"]]}},{"title":"Why files in state is rejected","body":"The backend would fetch AGENTS.md and the skills and send them in the files state.","bullets":["The caller writes system instructions: a prompt-injection channel outside our trust boundary.","Persona logic (layout, links, versions, precedence) would live in teams we do not control.","The content would be checkpointed in every thread.","Files in the thread state are writable by the agent."],"collapsible":true},{"title":"Assistant per persona, in detail","body":"An assistant is a saved setup of the orchestrator graph, created once and identified by an assistant id; its context can name the persona, and the backend then only picks the assistant.","bullets":["An assistant's context reaches the running graph (Runtime.context) but never the factory's configurable, so loading still happens at call time, with the same loader.","A pipeline must create or update an assistant on every persona push, and recreate the right assistants on every orchestrator redeploy.","Assistants live in the orchestrator deployment: the pipeline needs create rights there.","Compatible with the chosen option: the loader reads Runtime.context first, so both can coexist."],"collapsible":true}],"snippets":[],"checkpoint":"For each option you can say who writes the persona's prompt and what has to be operated.","common_mistakes":["Treating the assistant option as free: it needs a pipeline that keeps assistants in sync with Context Hub and with every orchestrator deployment."],"source_refs":["catalog","langgraph-api-src"],"assumptions":[],"links":[]},{"id":"where-personas-live","track_id":"where","detail_level":"guided","title":"Where they live","objective":"Know the repository type and layout a franchise publishes, and what a linked skill is.","summary":"Each franchise workspace holds one Context Hub agent repo per persona, tagged twin-persona, and one config repo whose config.json lists the workspace's personas. In v1 a persona repo holds AGENTS.md and skills/\u003cname\u003e/SKILL.md as plain files. Linked skills are v2.","why":"A persona is an agent definition, and AGENTS.md plus skills/ is the documented agent-repo layout. Agent repos also list apart from plain skills (list_agents vs list_skills).","prerequisites":["Three options"],"sections":[{"title":"Layout (v1, inline)","body":"Inline means the files live inside the persona repo; the config repo is the workspace's registry.","preformatted":"\u003cfranchise workspace\u003e\n├── config                      (agent repo: config.json, the persona registry)\n└── incident-manager-persona    (agent repo, tag: twin-persona)\n    ├── AGENTS.md\n    └── skills/\n        ├── triage/SKILL.md\n        └── postmortem/SKILL.md","bullets":[],"collapsible":false},{"title":"Agent repo or skill repo","body":"Both types store files and read the same way: pull_agent works on a skill repo and the stock backend lists its folders (checked on an existing skill repo). The type changes meaning and discovery, not readability.","table":{"headers":["","Agent repo","Skill repo"],"rows":[["Meant for","A whole agent definition","One reusable skill"],["Documented layout","AGENTS.md, tools.json, skills/, agents/","SKILL.md plus supporting files"],["Can link to","Skills and agents","Not tested"],["Listed by","list_agents","list_skills"]]},"bullets":[],"collapsible":true},{"title":"How a link works (v2): SkillEntry","body":"An agent repo is a map of paths to entries: a FileEntry (inline text) or a link, SkillEntry or AgentEntry, carrying repo_handle, owner and an optional pin. On pull, the server returns the link and the linked repo's files inlined under its path, so readers see ordinary files. Observed on LangSmith 0.18.3 with seeded test repos:","bullets":["Links are inlined by the server: the existing backends read linked skills unchanged.","Pins on links are ignored, by commit_hash and by commit_id.","A push to a linked skill creates a new commit of every persona that links it.","A link to a repo holding several skills lands one level deeper and needs its own skills source.","A skill's folder must carry the skill's name, or a spec warning is raised."],"collapsible":true}],"snippets":[{"label":"Publish a v1 persona (Python, langsmith SDK, key scoped to the franchise workspace)","language":"python","status":"excerpt","code":"from langsmith import Client\nfrom langsmith.schemas import FileEntry\n\nClient().push_agent(\n    \"-/\u003cPERSONA_REPO\u003e\",\n    files={\n        \"AGENTS.md\": FileEntry(content=\"\u003cpersona instructions\u003e\"),\n        \"skills/triage/SKILL.md\": FileEntry(content=\"\u003cskill with name: triage\u003e\"),\n    },\n    tags=[\"twin-persona\"],\n)","expected":"pull_agent on the same repo returns AGENTS.md and skills/triage/SKILL.md as file entries. Same call shape verified on LangSmith 0.18.3."}],"checkpoint":"You can say why a persona is an agent repo even though a skill repo would read the same, and what a SkillEntry adds in v2.","common_mistakes":["Naming a skill folder differently from the skill (spec warning).","Linking a multi-skill repo under skills/: its skills sit one level too deep to be indexed."],"source_refs":["live-hub"],"assumptions":["Cross-workspace links (a franchise persona linking a twinCore skill) are untested."],"links":[]},{"id":"what-we-built","track_id":"where","detail_level":"guided","title":"What we built","objective":"Separate what already exists and ships from what the persona design adds.","summary":"HubSkillsBackend is our read-only, TTL-cached subclass of the deepagents ContextHubBackend. It is already mounted at /skills/hub/\u003cgroup\u003e/ and was pushed with the skills tier. The persona design reuses it as is: one shared instance per persona repo and tag, mounted at /persona/.","why":"The persona work is an extension of shipped code, not a new subsystem.","prerequisites":["Where they live"],"sections":[{"title":"Stock ContextHubBackend vs our HubSkillsBackend","body":"Four differences, all funnelled through the stock code paths.","table":{"headers":["","ContextHubBackend (deepagents)","HubSkillsBackend (ours)"],"rows":[["Writes","Read-write: write_file and edit_file push real Hub commits","Read-only: refused before any Hub call"],["Refresh","Pulls once, never refreshes","Re-pulls after a TTL (300 s); keeps the last good copy on a Hub outage"],["Reload","None","invalidate(), used by /skill-reload"],["Sharing","One per construction","One per repo for the whole process (get_hub_skills_backend)"]]},"bullets":[],"collapsible":false}],"snippets":[{"label":"The class (excerpt from backends/hub_backend.py)","language":"python","status":"excerpt","code":"class HubSkillsBackend(ReadOnlyBackendMixin, ContextHubBackend):\n    \"\"\"Read-only ContextHubBackend whose snapshot expires after a TTL.\"\"\"\n\n@cache\ndef get_hub_skills_backend(identifier: str) -\u003e HubSkillsBackend:\n    return HubSkillsBackend(identifier)","expected":"One shared, read-only instance per Hub repo identifier."}],"checkpoint":"You can name two behaviours of the stock ContextHubBackend that make it unsafe to mount directly.","common_mistakes":["Mounting the stock ContextHubBackend: the agent's write_file would push a real Hub commit."],"source_refs":["hub-backend","deepagents-src"],"assumptions":[],"links":[]},{"id":"contracts","track_id":"load","detail_level":"guided","title":"Contracts","objective":"Know what each stakeholder writes and reads: franchises, TwinShield, the backend and the orchestrator.","summary":"Franchises write a persona repo and a config.json registry per workspace, and tag them dev, staging or production. TwinShield reads the registry at the tags of the request's environment and returns each user's authorized personas. The backend lists those personas and calls the orchestrator with workspace, repo and version. The orchestrator loads the repo at that tag.","why":"Each contract has one writer, so each change has one owner.","prerequisites":["What we built"],"sections":[{"title":"Who writes, who reads","body":"","bullets":[],"collapsible":false,"table":{"headers":["Stakeholder","Writes","Reads"],"rows":[["Franchise","Persona agent repo (AGENTS.md, skills/) and config.json in its config repo; tags them dev, staging or production","Nothing"],["TwinShield","authorized_personas per user, next to the authorized assistants","config.json at each tag of the environment"],["Backend (BFF)","Run requests carrying the persona identity","authorized_personas, for the UI catalog"],["twinCore orchestrator","Nothing in Context Hub (read-only)","The persona repo at the requested tag (authorized_personas from habilitation v2)"]]}},{"title":"config.json: the franchise contract","body":"One registry per workspace, in an agent repo named config. The schema is owned by the client; this shape is illustrative. The registry is tagged like any repo, and each persona it lists is read at the same tag: a persona whose repo lacks the tag is dropped by TwinShield.","preformatted":"{\n  \"personas\": [\n    {\n      \"name\": \"incident-manager\",\n      \"repo\": \"incident-manager-persona\",\n      \"description\": \"...\",\n      \"short_description\": \"...\",\n      \"visibility\": {\"ui\": true, \"orchestrator\": false}\n    }\n  ]\n}","bullets":["name is the persona's stable key in the registry; repo is its agent repo, which the backend sends to the orchestrator.","No version field: the tag on the commits is the version.","visibility {ui: true, orchestrator: false}: shown in the UI, never delegated to by the main orchestrator.","Visibility is not access control: entitlement comes from TwinShield."],"collapsible":false},{"title":"TwinShield contract (proposal)","body":"Personas in a separate field of the same response, not mixed into the authorized assistants.","preformatted":"AuthorizedAssistants\n├── assistants:          [callable subagents, unchanged]\n└── authorized_personas: [{workspace_id, name, repo, version, description, short_description}, ...]","bullets":["version is the tag the persona was resolved at: dev, staging or production. Environment dev lists dev; uat and qual list dev and staging; prod lists production.","In uat and qual the same persona can be listed twice, once per tag.","An entry in assistants means a callable subagent: a persona listed there would be offered to the main orchestrator.","To agree with TwinShield: where role-based access per persona is declared, and the cache lifetime."],"collapsible":true},{"title":"Backend contract: context or configurable","body":"The backend sends the persona in the run request, in context (preferred, the long-term replacement) or in configurable. The LangGraph server rejects a single request carrying both (400); when both reach the orchestrator, context wins. Run context is also copied into configurable, so the factory sees it either way.","table":{"headers":["Persona set in","Seen by the factory (configurable)","Seen at call time (Runtime.context)"],"rows":[["Run context","Yes (copied)","Yes"],["Run configurable","Yes","Yes (copied)"],["Assistant configurable","Yes","No"],["Assistant context","No","Yes"]]},"bullets":["The backend sends {workspace_id, repo, version}, copied from the chosen authorized_personas entry.","version selects the tag the orchestrator loads; the orchestrator does not refuse a tag because of its own environment.","Persona runs from the dev UI go to the -dev orchestrator; from uat and qual to -qual, the stable one; from prod to prod.","v1 trusts the backend's choice; the check against authorized_personas arrives with orchestrator habilitation v2, without changing the request.","Same persona and version on every run of a thread; a switch on an existing thread is refused."],"collapsible":true}],"snippets":[{"label":"Run request body (LangGraph Server API, POST /threads/\u003cTHREAD_ID\u003e/runs)","language":"json","status":"placeholder","code":"{\n  \"assistant_id\": \"\u003cORCHESTRATOR_ASSISTANT_ID\u003e\",\n  \"input\": {\"messages\": [{\"role\": \"user\", \"content\": \"\u003cquestion\u003e\"}]},\n  \"context\": {\"persona\": {\"workspace_id\": \"\u003cWORKSPACE_ID\u003e\", \"repo\": \"\u003cPERSONA_REPO\u003e\", \"version\": \"staging\"}}\n}","expected":"The run starts with the persona's skills in the skills index and its AGENTS.md in a \u003cpersona\u003e prompt section. Unverified: not executed (design)."}],"checkpoint":"You can name the writer and the readers of config.json, and write the run request the backend sends.","common_mistakes":["Sending the persona in both context and configurable: the server rejects the run (400).","Listing personas among TwinShield's authorized assistants: the main orchestrator would delegate to them."],"source_refs":["langgraph-api-src","catalog"],"assumptions":["The TwinShield response shape is a proposal to agree with TwinShield.","Assistant context merging was read in the in-memory runtime; the Postgres runtime is to verify."],"links":[]},{"id":"request-flow","track_id":"load","detail_level":"guided","title":"One request","objective":"Trace one request from the BFF to a persona file.","summary":"The factory builds the graph per request as today; a PersonaMiddleware turns the persona into prompt content; the skills middleware indexes the persona's skills; read_file reaches the persona files through the composite backend.","why":"Each component has one job, so each failure has one owner.","prerequisites":["Contracts"],"sections":[{"title":"Request flow","body":"make_orchestrator already runs create_deep_agent on every request.","preformatted":" Frontend -\u003e BFF -\u003e POST /threads/{tid}/runs\n                   context.persona = {workspace_id, repo, version}  (or configurable)\n                           |\n                           v\n +------------------- make_orchestrator(config)  (every request) -------------------+\n |  subject -\u003e TwinShield roster -\u003e subagents + task tool                             |\n |  persona -\u003e shared HubSkillsBackend(workspace, repo, tag), or the empty backend   |\n |  build_orchestrator_backend() -\u003e one CompositeBackend, /persona/ always mounted   |\n |  middleware: [..., PersonaMiddleware, LoadableSkills(.., /persona/skills/)]      |\n |  create_deep_agent(backend=composite, middleware=..., subagents=...)              |\n +-----------------------------------------------------------------------------------+\n                           | run\n                           v\n  PersonaMiddleware: \u003cpersona\u003e section, commit logged in state, notice,\n                     refuse a persona switch on the thread\n  LoadableSkills:    index skills from every source, latched once per thread\n  read_file:         /persona/skills/triage/SKILL.md -\u003e composite -\u003e HubSkillsBackend","bullets":[],"collapsible":false},{"title":"PersonaMiddleware duties (v1)","body":"What a backend cannot do.","bullets":["Resolve the persona: Runtime.context first, then configurable.","Add the persona's AGENTS.md as a \u003cpersona\u003e section at the prompt tail (stable per thread, cache friendly).","Log {workspace_id, repo, version, commit} in a state channel and the run metadata.","Show a notice when the persona cannot be loaded, without latching, so the next turn retries.","Refuse the turn on a persona or version switch within a thread."],"collapsible":true}],"snippets":[],"checkpoint":"You can say which component turns the persona into prompt content and which one serves its files.","common_mistakes":["Adding the persona's AGENTS.md to LiveMemoryMiddleware sources: it bootstraps a stub for a missing source, which would write into the read-only mount."],"source_refs":["catalog"],"assumptions":["LiveMemoryMiddleware's stub bootstrap hitting the read-only mount is reasoned from the source, to confirm in the prototype."],"links":[]},{"id":"mounted-paths","track_id":"load","detail_level":"guided","title":"Mounted paths","objective":"Know which backend serves every path the agent can read.","summary":"A persona adds one mount, /persona/, present on every request with or without a persona, so the graph is identical either way.","why":"Constant routes and sources keep the graph topology the same across personas and access contexts.","prerequisites":["One request"],"sections":[{"title":"The tree","body":"One CompositeBackend per request; its routes never change. Memory and user skills appear only where they are enabled; the client's orchestrator runs with both off.","preformatted":"/                          CompositeBackend (routes never change)\n├── (default)              StateBackend        thread scratch files\n├── memory/                StoreBackend        only with memory enabled\n├── skills/\n│   ├── generic/           read-only bank      wheel package files (if added)\n│   ├── builtin/           read-only bank      wheel package files (if added)\n│   └── user/              StoreBackend        only with skills enabled\n└── persona/               HubSkillsBackend    shared, one per (workspace, repo, tag)\n    │                      or an empty read-only backend when no persona is sent\n    ├── AGENTS.md          \u003c- Context Hub: \u003cfranchise ws\u003e/\u003cpersona repo\u003e@\u003ctag\u003e\n    └── skills/\n        ├── triage/SKILL.md      \u003c- inline file in the persona repo (v1)\n        └── postmortem/SKILL.md  \u003c- SkillEntry link, inlined by the server (v2)","bullets":[],"collapsible":false},{"title":"Skills load order","body":"Last source wins on a name clash. With enable_personas on, the skills middleware is added even when enable_skills is off, over the persona source and any read-only bank.","preformatted":"/skills/generic/ -\u003e /skills/builtin/ -\u003e /persona/skills/ -\u003e /skills/user/ (when enabled)","bullets":[],"collapsible":false}],"snippets":[],"checkpoint":"You can say which backend answers read_file /persona/skills/triage/SKILL.md, and which answers /skills/user/\u003cx\u003e/SKILL.md.","common_mistakes":["Mounting /persona/ only when a persona is present: the graph then differs between requests and access contexts."],"source_refs":["user-backend"],"assumptions":[],"links":[]},{"id":"persona-backend","track_id":"decide","detail_level":"guided","title":"Loading a persona","objective":"Explain how the factory picks a shared Hub backend per persona and tag, and what stands at /persona/ when no persona is sent.","summary":"The factory already runs on every request and sees the persona, since run context is copied into configurable. It mounts at /persona/ the shared HubSkillsBackend for (workspace, repo, tag), created on first use and cached for the whole process. No persona: an empty read-only backend.","why":"No new backend class: the per-request rebuild does the routing, and the cached instances carry the Hub snapshots, so a rebuild costs no Hub call.","prerequisites":["Mounted paths"],"sections":[{"title":"What the factory does","body":"Per request, before create_deep_agent:","table":{"headers":["Step","How"],"rows":[["Read the persona","Runtime.context, else configurable (run context is copied there)"],["Pick the tag","version as sent: dev, staging or production; dev falls back to the latest commit when the repo has no dev tag (logged)"],["Get the backend","Process-wide cache keyed by (workspace, repo, tag); created on first use"],["No persona","An empty read-only backend: listing empty, reads not found, writes refused"]]},"bullets":[],"collapsible":false},{"title":"Why the mount and middlewares are always there","body":"The server also builds the graph for thread reads and updates, which carry no persona. If the persona middlewares were added only with a persona, those builds would read the thread with a different state schema than the run that wrote it. So with enable_personas on, /persona/ and the skills middleware are present on every build. An empty backend, not a StateBackend: a writable mount would let the agent write its own persona file and read it back as instructions.","bullets":[],"collapsible":false},{"title":"Clients and connections","body":"A ContextHubBackend is bound to one repo, with one snapshot cache and one lock, so there is one instance per (workspace, repo, tag). The workspace header is fixed per langsmith Client, so there is one client per workspace; all of them share one HTTP session and run without background tracing, so they are thin objects over one connection pool.","preformatted":"cache[(ws-A, incident-manager-persona, staging)] -\u003e HubSkillsBackend -\u003e snapshot (TTL)\ncache[(ws-A, incident-manager-persona, dev)]     -\u003e HubSkillsBackend -\u003e snapshot (TTL)\ncache[(ws-B, crisis-lead-persona, staging)]      -\u003e HubSkillsBackend -\u003e snapshot (TTL)\n        one Client per workspace, one shared HTTP session","bullets":[],"collapsible":true}],"snippets":[{"label":"Shape of the factory step (pseudocode, not implemented)","language":"python","status":"pseudocode","code":"persona = resolve_persona(config)  # Runtime.context, then configurable\nif persona:\n    persona_backend = hub_backend_for(persona[\"workspace_id\"], persona[\"repo\"], persona[\"version\"])\nelse:\n    persona_backend = EMPTY_PERSONA_BACKEND\nroutes[\"/persona/\"] = persona_backend","expected":"Every read under /persona/ resolves to the shared snapshot of that repo at that tag; without a persona, /persona/ is empty and read-only."}],"checkpoint":"You can say why there is one backend per (workspace, repo, tag) and why /persona/ stays mounted when no persona is sent.","common_mistakes":["Switching the identifier of one shared ContextHubBackend per call: concurrent threads overwrite each other's cache.","Adding the persona middlewares only when a persona is sent: thread reads build the graph without one."],"source_refs":["deepagents-src","hub-backend"],"assumptions":["A v2 pin per thread, or a move to compile-once, would bring back a router that resolves the persona at call time."],"links":[]},{"id":"versioning","track_id":"decide","detail_level":"guided","title":"Versioning","objective":"Know which persona version a thread sees, the gaps of the v1 default, and the options for later.","summary":"Three Context Hub tags: dev, staging, production. dev falls back to the latest commit when the repo has no dev tag; staging and production are the UI's promote targets. The tag is chosen by the request's version. The commit served is logged in the state each turn; v1.1 adds a persona-updated reminder; v2 can pin the commit per thread.","why":"Tagging becomes the release gate: a franchise decides which commit each environment runs.","prerequisites":["Loading a persona"],"sections":[{"title":"Environments","body":"The config repo and every persona repo are read at the same tag. Tags are set in the Context Hub UI: a custom name such as dev adds a tag, staging and production promote. To see what is tagged where: GET /repos/\u003cowner\u003e/\u003crepo\u003e/tags lists each tag and its commit; pull_agent(\u003crepo\u003e, version=\"staging\") returns that tag's content.","table":{"headers":["Tag","Listed in UI environment","Runs on orchestrator"],"rows":[["dev (else latest commit)","dev, uat, qual","-dev for dev; -qual for uat and qual"],["staging","uat, qual","-qual"],["production","prod","prod"]]},"bullets":[],"collapsible":false},{"title":"What is left: a new tag during an open thread","body":"","bullets":["The skills index latched on turn 1 stays as it was; skill bodies read later follow the newly tagged commit.","Rare (tagging time, open threads only); the commit logged per turn makes it visible."],"collapsible":false},{"title":"v1.1: persona-updated reminder","body":"When the logged commit changes between turns: reset the skills latch (the /skill-reload mechanism), refresh the \u003cpersona\u003e section, and append one immutable \u003csystem_reminder\u003e at the tail. The thread moves to the new version atomically at a turn boundary.","bullets":[],"collapsible":true},{"title":"v2 options for a pin","body":"A tag pins a persona commit and its linked skills: verified, the tag stays put while new commits arrive.","table":{"headers":["","No pin (v1)","Pin in state + context variable","Pin in Store"],"rows":[["Hub backends","1 per persona and tag","1 per persona commit in use","1 per persona commit in use"],["Extra code","State log","Bridge in 2 middlewares","Store read/write + expiry"],["HITL resume","Follows the tag","Pinned","Pinned"],["Replay old checkpoint","Follows the tag","That checkpoint's pin","Current pin"],["Risk","Skew at promotion","Subagent inheritance to verify","Orphan rows, first-write race"]]},"bullets":[],"collapsible":true}],"snippets":[],"checkpoint":"You can say which tag a uat user's persona runs at, on which orchestrator, and what an open thread sees after a new tag.","common_mistakes":["Configuring latest as a tag: it is not one and the pull returns 404; omit the version to mean the latest commit.","Assuming a link pin freezes a skill version: pins on links are ignored on LangSmith 0.18.3."],"source_refs":["live-hub"],"assumptions":["Custom tags set in the client's Context Hub UI (0.16.50) read the same as API-created ones; to confirm with the client probe."],"links":[]},{"id":"why-rebuild","track_id":"decide","detail_level":"guided","title":"Why rebuild","objective":"Defend rebuilding the graph per request, and say what the challenge exposed.","summary":"The platform documents the factory for loading different tools depending on the user's credentials. deepagents freezes the task tool description and the subagent dispatch map at build time, and our roster is per user, with remote subagents discovered at run time.","why":"Compile-once would move authorization from structural absence to runtime guards that must never miss.","prerequisites":["Versioning"],"sections":[{"title":"Measured cost (local warm baseline)","body":"Offline experiment, static roster, fake model, no discovery, no remote wrappers: not a production figure.","bullets":["About 30 ms per warm build, p95 at most 45 ms, under the server's 100 ms warning.","No socket from the building thread, cold or warm; about 540 B retained per build.","A persona mount adds 0 to 2.5 ms (within noise)."],"collapsible":false},{"title":"Where the team lead is right","body":"Fixes owed regardless of personas.","bullets":["A persona is content, not topology: it does not add a reason to rebuild.","The factory also runs on thread reads, history and schema reads; we never check __is_for_execution__, so a cold read can pay TwinShield discovery.","The state schema follows the grants: it can differ if entitlements change between a run and a later resume."],"collapsible":true}],"snippets":[],"checkpoint":"You can give the two build-time facts that force the rebuild, the measured cost with its scope, and one valid criticism.","common_mistakes":["Quoting 30 ms as a production number: remote rosters, model overrides and concurrency are not measured yet."],"source_refs":["build-cost","langgraph-api-src","deepagents-src"],"assumptions":[],"links":[]},{"id":"risks-and-versions","track_id":"decide","detail_level":"executable","title":"Risks \u0026 versions","objective":"Know every known limit, the version it affects, and the fix version or workaround.","summary":"Each row says where it was seen and what we do about it. The client platform runs LGP 0.15 or 0.16 (to confirm), which makes the client probe the deciding test.","why":"A limit is only manageable when its affected version and its fix or workaround are known.","prerequisites":["Why rebuild"],"sections":[{"title":"Platform and library","body":"Seen in docs, sources or live tests.","table":{"headers":["Limit","Affects","Status / fix","Workaround"],"rows":[["Context Hub availability","Self-hosted LangSmith","Introduced in v0.15 (docs)","Probe the client version; Store mirror as fallback"],["Linked-directory bugs","Self-hosted \u003c 0.17.0-rc","Fixes listed up to 0.17.0-rc","v1 uses inline skills only"],["Link pins ignored","LangSmith 0.18.3","Not fixed","Load a tag; the persona commit is the manifest"],["commit_id link cannot be pushed","langsmith SDK 0.14.1","Open (UUID not JSON serializable)","Do not pin links"],["Commit hashes repeat across repos","LangSmith 0.18.3","Observed behaviour","Always key by (workspace, repo, hash)"],["Listing returns other tenants' public repos","LangSmith 0.18.3","Observed behaviour","Filter by workspace and the twin-persona tag"],["client.info swallows /info failures","langsmith SDK 0.14.1","By design","Probe treats an empty version as a failure"],["New Client fires a background /info call","langsmith SDK 0.14.1","By design","One Client per workspace over one shared session"],["Skills middleware scans one level per source","deepagents 0.7.19","By design","Link skills individually, or one source per group"],["Factory runs on read paths","langgraph-api 0.9.0","By design","Gate discovery on __is_for_execution__"],["latest is not a tag","LangSmith 0.18.3","By design (404)","Read a tag; no version means the latest commit"],["Listings carry no tags","LangSmith 0.18.3","By design","Pull config at the tag; GET /repos/{owner}/{name}/tags per repo"],["Assistant context not in configurable","langgraph-api 0.9.0","By design","Resolve the persona at call time"],["Run with both context and configurable","langgraph-api 0.9.0","Rejected (400)","Send the persona in context or configurable, not both"]]},"bullets":[],"collapsible":false},{"title":"Design risks","body":"Handled in the design.","bullets":["v1 trusts the backend: any authenticated caller can make the orchestrator load any repo its Hub key reads. Accepted until habilitation v2 checks authorized_personas.","A failed first load must not latch an empty skills index: notice and retry next turn.","Persona text sits below \u003csoul\u003e and \u003cboundaries\u003e; user memory, where enabled, refines it.","Subagents never see persona skills: they guide the orchestrator only."],"collapsible":true}],"snippets":[{"label":"Run at the client, next to a deployment (read-only, redacted output)","language":"bash","status":"complete","code":"python persona_hub_probe.py --persona=-/\u003cPERSONA_REPO\u003e --workspace-id \u003cWORKSPACE_ID\u003e --repeat 5 --burst 8 --redact","expected":"A version line, the persona layout with linked_skills counts, full-load timings, then RESULT ok (exit 0). Verified on LangSmith 0.18.3."}],"checkpoint":"For each row you can say whether it blocks v1, v2, or neither.","common_mistakes":["Trusting link inlining on the client version without running the probe on a persona that has a link."],"source_refs":["live-hub","probe"],"assumptions":["The client version (0.15 or 0.16) is to confirm."],"links":[]},{"id":"roadmap","track_id":"decide","detail_level":"guided","title":"Roadmap","objective":"Present the release plan and its gates, ready for the team decision.","summary":"v1 ships inline personas at dev, staging and production tags with drift logged; v1.1 adds the persona-updated reminder; habilitation v2 adds the entitlement check; v2 adds linked skills and the pin, once its gates pass.","why":"Each step earns the next with evidence instead of building v2 on assumptions.","prerequisites":["Risks \u0026 versions"],"sections":[{"title":"Releases and gates","body":"","bullets":["v1: persona and config repos per workspace, inline skills, tags dev, staging and production, shared Hub backend per (workspace, repo, tag), enable_personas switch, commit logged in state.","v1.1: persona-updated system reminder when the logged commit changes.","Habilitation v2: each run's persona checked against the user's authorized_personas.","v2 gate 1: the client probe shows links inlined on its version.","v2 gate 2: governance accepts that a skill push reaches every linking persona.","v2 gate 3: pin in state (bridge) verified with subagents; cross-workspace links tested.","Alternative kept: one orchestrator assistant per persona, if the pipeline cost is accepted."],"collapsible":false},{"title":"Design decisions","body":"Persona agent repos listed in a config.json registry per workspace. Inline skills in v1. Persona {workspace_id, repo, version} in context or configurable as the only input; version selects the tag. Shared Hub backend per (workspace, repo, tag), mounted by the factory. Persona runs from uat and qual on the -qual orchestrator. No pin in v1. PersonaMiddleware for prompt, log, notice and switch refusal. Backend trusted in v1; entitlement check with habilitation v2. Per-request rebuild kept. Update reminder in v1.1.","bullets":[],"collapsible":true}],"snippets":[],"checkpoint":"You can present v1 to the team with its acknowledged gaps, and name the three gates that must pass before v2.","common_mistakes":["Starting v2 links before the client probe: link inlining may be missing on 0.15 or 0.16."],"source_refs":["catalog"],"assumptions":[],"links":[{"label":"Back to the problem","node_id":"problem"}]}]},"views":[{"id":"options","workflow":"persona-options.workflow.json","title":"The options at a glance","caption":"Three ways a persona can reach the orchestrator. The persona field is chosen; the assistant per persona stays a compatible alternative.","detailed":false,"node_map":{"opt-files-send":"options","opt-files-state":"options","opt-files-verdict":"options","opt-field-send":"contracts","opt-field-load":"persona-backend","opt-field-verdict":"where-personas-live","opt-assistant-send":"options","opt-assistant-load":"options","opt-assistant-verdict":"options"},"svg":"\u003csvg viewBox=\"0 0 900 560\" role=\"img\" lang=\"en\" aria-labelledby=\"archify-diagram-title archify-diagram-description\" data-preset=\"classic\" data-quality-profile=\"showcase\"\u003e\n        \u003ctitle id=\"archify-diagram-title\"\u003eThree ways to hand a persona to the orchestrator\u003c/title\u003e\n        \u003cdesc id=\"archify-diagram-description\"\u003eA workflow diagram generated by Archify.\u003c/desc\u003e\n        \u003c!-- Definitions --\u003e\n        \u003cdefs\u003e\n          \u003cmarker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-default\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-emphasis\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-emphasis\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-security\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-security\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-dashed\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-dashed\" /\u003e\n          \u003c/marker\u003e\n          \u003cpattern id=\"grid\" width=\"40\" height=\"40\" patternUnits=\"userSpaceOnUse\"\u003e\n            \u003cpath d=\"M 40 0 L 0 0 0 40\" class=\"c-grid\" stroke-width=\"0.5\"/\u003e\n          \u003c/pattern\u003e\n        \u003c/defs\u003e\n\n        \u003c!-- Background Grid --\u003e\n        \u003crect width=\"100%\" height=\"100%\" fill=\"url(#grid)\" /\u003e\n\n        \u003c!-- Swimlanes --\u003e\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-0\" x=\"40\" y=\"52\" width=\"832\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"74\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e01 / Files in state\u003c/text\u003e\n\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-1\" x=\"40\" y=\"176\" width=\"832\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"198\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e02 / Persona field (chosen)\u003c/text\u003e\n\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-2\" x=\"40\" y=\"300\" width=\"832\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"322\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e03 / Assistant per persona\u003c/text\u003e\n\n        \u003c!-- Phase headers --\u003e\n\n\n        \u003c!-- Workflow groups --\u003e\n\n\n        \u003c!-- Edge paths --\u003e\n        \u003cpath data-edge-from=\"opt-assistant-send\" data-edge-to=\"opt-assistant-load\" data-edge-key=\"0\" data-edge-id=\"opt-assistant-1\" data-composition-points=\"188,367;216,367\" d=\"M 188 367 L 216 367\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"opt-assistant-load\" data-edge-to=\"opt-assistant-verdict\" data-edge-key=\"1\" data-edge-id=\"opt-assistant-2\" data-composition-points=\"356,367;384,367\" d=\"M 356 367 L 384 367\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"opt-field-send\" data-edge-to=\"opt-field-load\" data-edge-key=\"2\" data-edge-id=\"opt-field-1\" data-composition-points=\"188,243;216,243\" d=\"M 188 243 L 216 243\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"opt-field-load\" data-edge-to=\"opt-field-verdict\" data-edge-key=\"3\" data-edge-id=\"opt-field-2\" data-composition-points=\"356,243;384,243\" d=\"M 356 243 L 384 243\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"opt-files-send\" data-edge-to=\"opt-files-state\" data-edge-key=\"4\" data-edge-id=\"opt-files-1\" data-composition-points=\"188,119;216,119\" d=\"M 188 119 L 216 119\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"opt-files-state\" data-edge-to=\"opt-files-verdict\" data-edge-key=\"5\" data-edge-id=\"opt-files-2\" data-composition-points=\"356,119;384,119\" d=\"M 356 119 L 384 119\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n\n        \u003c!-- Nodes --\u003e\n        \u003cg id=\"node-opt-files-send\" data-node-id=\"opt-files-send\" data-node-label=\"BFF fetches files\" tabindex=\"0\" role=\"button\" aria-label=\"Focus BFF fetches files, AGENTS.md + skills, Files in state\" aria-pressed=\"false\" data-node-kind=\"frontend\" data-node-sublabel=\"AGENTS.md + skills\" data-node-context=\"Files in state\"\u003e\n          \u003ctitle\u003eBFF fetches files · AGENTS.md + skills · Files in state\u003c/title\u003e\n          \u003crect x=\"48\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"48\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-frontend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"frontend\" class=\"semantic-sigil s-frontend\" transform=\"translate(54 99) scale(0.6875)\"\u003e\n            \u003crect x=\"2\" y=\"3\" width=\"12\" height=\"10\" rx=\"2\"/\u003e\n            \u003cpath d=\"M2 6.5h12\"/\u003e\n            \u003ccircle cx=\"4.1\" cy=\"4.8\" r=\".7\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"6.3\" cy=\"4.8\" r=\".7\" class=\"sigil-fill\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"118\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eBFF fetches files\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"118\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eAGENTS.md + skills\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-opt-files-state\" data-node-id=\"opt-files-state\" data-node-label=\"Files in state\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Files in state, copied per thread, Files in state\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"copied per thread\" data-node-context=\"Files in state\"\u003e\n          \u003ctitle\u003eFiles in state · copied per thread · Files in state\u003c/title\u003e\n          \u003crect x=\"216\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"216\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(222 99) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"286\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eFiles in state\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"286\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003ecopied per thread\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-opt-files-verdict\" data-node-id=\"opt-files-verdict\" data-node-label=\"Rejected\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Rejected, caller writes the prompt, Files in state\" aria-pressed=\"false\" data-node-kind=\"security\" data-node-sublabel=\"caller writes the prompt\" data-node-context=\"Files in state\"\u003e\n          \u003ctitle\u003eRejected · caller writes the prompt · Files in state\u003c/title\u003e\n          \u003crect x=\"384\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"384\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-security\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"security\" class=\"semantic-sigil s-security\" transform=\"translate(390 99) scale(0.6875)\"\u003e\n            \u003cpath d=\"M8 2.2 13 4v3.5c0 3.1-1.8 5.4-5 6.5-3.2-1.1-5-3.4-5-6.5V4Z\"/\u003e\n            \u003cpath d=\"m5.8 8 1.5 1.5 3-3\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"454\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eRejected\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"454\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003ecaller writes the prompt\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-opt-field-send\" data-node-id=\"opt-field-send\" data-node-label=\"BFF sends persona\" tabindex=\"0\" role=\"button\" aria-label=\"Focus BFF sends persona, context or configurable, Persona field (chosen)\" aria-pressed=\"false\" data-node-kind=\"frontend\" data-node-sublabel=\"context or configurable\" data-node-context=\"Persona field (chosen)\"\u003e\n          \u003ctitle\u003eBFF sends persona · context or configurable · Persona field (chosen)\u003c/title\u003e\n          \u003crect x=\"48\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"48\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-frontend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"frontend\" class=\"semantic-sigil s-frontend\" transform=\"translate(54 223) scale(0.6875)\"\u003e\n            \u003crect x=\"2\" y=\"3\" width=\"12\" height=\"10\" rx=\"2\"/\u003e\n            \u003cpath d=\"M2 6.5h12\"/\u003e\n            \u003ccircle cx=\"4.1\" cy=\"4.8\" r=\".7\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"6.3\" cy=\"4.8\" r=\".7\" class=\"sigil-fill\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"118\" y=\"238\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eBFF sends persona\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"118\" y=\"255\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003econtext or configurable\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-opt-field-load\" data-node-id=\"opt-field-load\" data-node-label=\"Hub backend\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Hub backend, mounted per request, Persona field (chosen)\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"mounted per request\" data-node-context=\"Persona field (chosen)\"\u003e\n          \u003ctitle\u003eHub backend · mounted per request · Persona field (chosen)\u003c/title\u003e\n          \u003crect x=\"216\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"216\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(222 223) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"286\" y=\"238\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eHub backend\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"286\" y=\"255\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003emounted per request\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-opt-field-verdict\" data-node-id=\"opt-field-verdict\" data-node-label=\"Chosen\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Chosen, Context Hub at its tag, Persona field (chosen)\" aria-pressed=\"false\" data-node-kind=\"cloud\" data-node-sublabel=\"Context Hub at its tag\" data-node-context=\"Persona field (chosen)\"\u003e\n          \u003ctitle\u003eChosen · Context Hub at its tag · Persona field (chosen)\u003c/title\u003e\n          \u003crect x=\"384\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"384\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-cloud\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"cloud\" class=\"semantic-sigil s-cloud\" transform=\"translate(390 223) scale(0.6875)\"\u003e\n            \u003cpath d=\"M4.3 12.5h7.3a2.4 2.4 0 0 0 .2-4.8 4 4 0 0 0-7.5-1.3A3.1 3.1 0 0 0 4.3 12.5Z\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"454\" y=\"238\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eChosen\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"454\" y=\"255\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eContext Hub at its tag\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-opt-assistant-send\" data-node-id=\"opt-assistant-send\" data-node-label=\"CI/CD per push\" tabindex=\"0\" role=\"button\" aria-label=\"Focus CI/CD per push, creates assistants, Assistant per persona\" aria-pressed=\"false\" data-node-kind=\"messagebus\" data-node-sublabel=\"creates assistants\" data-node-context=\"Assistant per persona\"\u003e\n          \u003ctitle\u003eCI/CD per push · creates assistants · Assistant per persona\u003c/title\u003e\n          \u003crect x=\"48\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"48\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-messagebus\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"messagebus\" class=\"semantic-sigil s-messagebus\" transform=\"translate(54 347) scale(0.6875)\"\u003e\n            \u003cpath d=\"M2.5 4.5h11M2.5 8h11M2.5 11.5h11\"/\u003e\n            \u003ccircle cx=\"5\" cy=\"4.5\" r=\"1\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"10.5\" cy=\"8\" r=\"1\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"7\" cy=\"11.5\" r=\"1\" class=\"sigil-fill\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"118\" y=\"362\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eCI/CD per push\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"118\" y=\"379\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003ecreates assistants\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-opt-assistant-load\" data-node-id=\"opt-assistant-load\" data-node-label=\"Assistant context\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Assistant context, runtime only, Assistant per persona\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"runtime only\" data-node-context=\"Assistant per persona\"\u003e\n          \u003ctitle\u003eAssistant context · runtime only · Assistant per persona\u003c/title\u003e\n          \u003crect x=\"216\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"216\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(222 347) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"286\" y=\"362\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eAssistant context\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"286\" y=\"379\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eruntime only\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-opt-assistant-verdict\" data-node-id=\"opt-assistant-verdict\" data-node-label=\"Alternative\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Alternative, pipeline + redeploy sync, Assistant per persona\" aria-pressed=\"false\" data-node-kind=\"external\" data-node-sublabel=\"pipeline + redeploy sync\" data-node-context=\"Assistant per persona\"\u003e\n          \u003ctitle\u003eAlternative · pipeline + redeploy sync · Assistant per persona\u003c/title\u003e\n          \u003crect x=\"384\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"384\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-external\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"external\" class=\"semantic-sigil s-external\" transform=\"translate(390 347) scale(0.6875)\"\u003e\n            \u003crect x=\"2.5\" y=\"5\" width=\"8.5\" height=\"8\" rx=\"1.5\"/\u003e\n            \u003cpath d=\"M8 2.5h5.5V8M13.5 2.5 7.5 8.5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"454\" y=\"362\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eAlternative\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"454\" y=\"379\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003epipeline + redeploy sync\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003c!-- Edge labels --\u003e\n\n\n\n\n\n\n\n        \u003c!-- Legend --\u003e\n        \u003cg data-legend=\"\" data-legend-bridge=\"\"\u003e\n          \u003ctext x=\"20\" y=\"428\" class=\"t-primary\" font-size=\"12\" font-weight=\"650\"\u003eLegend\u003c/text\u003e\n          \u003cg data-legend-semantic-kind=\"frontend\" data-legend-kind=\"frontend\" data-legend-label=\"User UI\" data-legend-x=\"20\" data-legend-baseline=\"448\" data-legend-width=\"74\"\u003e\n            \u003crect x=\"20\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-frontend\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"42\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eUser UI\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"backend\" data-legend-kind=\"backend\" data-legend-label=\"Agent logic\" data-legend-x=\"101\" data-legend-baseline=\"448\" data-legend-width=\"91\"\u003e\n            \u003crect x=\"101\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-backend\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"123\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eAgent logic\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"security\" data-legend-kind=\"security\" data-legend-label=\"Policy\" data-legend-x=\"199\" data-legend-baseline=\"448\" data-legend-width=\"70\"\u003e\n            \u003crect x=\"199\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-security\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"221\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003ePolicy\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"messagebus\" data-legend-kind=\"messagebus\" data-legend-label=\"Tool action\" data-legend-x=\"276\" data-legend-baseline=\"448\" data-legend-width=\"91\"\u003e\n            \u003crect x=\"276\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-messagebus\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"298\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eTool action\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"cloud\" data-legend-kind=\"cloud\" data-legend-label=\"Cloud service\" data-legend-x=\"374\" data-legend-baseline=\"448\" data-legend-width=\"100\"\u003e\n            \u003crect x=\"374\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-cloud\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"396\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eCloud service\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"external\" data-legend-kind=\"external\" data-legend-label=\"External system\" data-legend-x=\"481\" data-legend-baseline=\"448\" data-legend-width=\"109\"\u003e\n            \u003crect x=\"481\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-external\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"503\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eExternal system\u003c/text\u003e\n          \u003c/g\u003e\n        \u003c/g\u003e\n      \u003c/svg\u003e","lanes":[{"id":"files","label":"Files in state"},{"id":"field","label":"Persona field (chosen)"},{"id":"assistant","label":"Assistant per persona"}],"workflow_nodes":[{"id":"opt-files-send","label":"BFF fetches files","lane":"files"},{"id":"opt-files-state","label":"Files in state","lane":"files"},{"id":"opt-files-verdict","label":"Rejected","lane":"files"},{"id":"opt-field-send","label":"BFF sends persona","lane":"field"},{"id":"opt-field-load","label":"Hub backend","lane":"field"},{"id":"opt-field-verdict","label":"Chosen","lane":"field"},{"id":"opt-assistant-send","label":"CI/CD per push","lane":"assistant"},{"id":"opt-assistant-load","label":"Assistant context","lane":"assistant"},{"id":"opt-assistant-verdict","label":"Alternative","lane":"assistant"}]},{"id":"narrative","workflow":"persona-journey.workflow.json","title":"A. The decision, step by step","caption":"Where personas live, how one request loads them, then how it runs and ships. Every node opens its card; this is the guided path.","detailed":true,"svg":"\u003csvg viewBox=\"0 0 1180 720\" role=\"img\" lang=\"en\" aria-labelledby=\"archify-diagram-title archify-diagram-description\" data-preset=\"classic\" data-quality-profile=\"showcase\"\u003e\n        \u003ctitle id=\"archify-diagram-title\"\u003ePersona presets: from Context Hub to the orchestrator\u003c/title\u003e\n        \u003cdesc id=\"archify-diagram-description\"\u003eA workflow diagram generated by Archify.\u003c/desc\u003e\n        \u003c!-- Definitions --\u003e\n        \u003cdefs\u003e\n          \u003cmarker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-default\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-emphasis\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-emphasis\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-security\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-security\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-dashed\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-dashed\" /\u003e\n          \u003c/marker\u003e\n          \u003cpattern id=\"grid\" width=\"40\" height=\"40\" patternUnits=\"userSpaceOnUse\"\u003e\n            \u003cpath d=\"M 40 0 L 0 0 0 40\" class=\"c-grid\" stroke-width=\"0.5\"/\u003e\n          \u003c/pattern\u003e\n        \u003c/defs\u003e\n\n        \u003c!-- Background Grid --\u003e\n        \u003crect width=\"100%\" height=\"100%\" fill=\"url(#grid)\" /\u003e\n\n        \u003c!-- Swimlanes --\u003e\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-0\" x=\"40\" y=\"52\" width=\"996\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"74\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e01 / Where it lives\u003c/text\u003e\n\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-1\" x=\"40\" y=\"176\" width=\"996\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"198\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e02 / How it loads\u003c/text\u003e\n\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-2\" x=\"40\" y=\"300\" width=\"996\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"322\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e03 / How it runs and ships\u003c/text\u003e\n\n        \u003c!-- Phase headers --\u003e\n\n\n        \u003c!-- Workflow groups --\u003e\n\n\n        \u003c!-- Edge paths --\u003e\n        \u003cpath data-edge-from=\"persona-backend\" data-edge-to=\"versioning\" data-edge-key=\"0\" data-edge-id=\"backend-to-versioning\" data-composition-points=\"888,367;860,367\" d=\"M 888 367 L 860 367\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"what-we-built\" data-edge-to=\"contracts\" data-edge-label=\"Reuse confirmed\" data-edge-key=\"1\" data-edge-id=\"built-to-contracts\" data-composition-points=\"692,119;708,119;708,166;622,166;622,217\" d=\"M 692 119 L 708 119 L 708 166 L 622 166 L 622 217\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"contracts\" data-edge-to=\"request-flow\" data-edge-key=\"2\" data-edge-id=\"contracts-to-request\" data-composition-points=\"692,243;720,243\" d=\"M 692 243 L 720 243\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"options\" data-edge-to=\"where-personas-live\" data-edge-key=\"3\" data-edge-id=\"options-to-where\" data-composition-points=\"356,119;384,119\" d=\"M 356 119 L 384 119\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"mounted-paths\" data-edge-to=\"persona-backend\" data-edge-label=\"Route traced\" data-edge-key=\"4\" data-edge-id=\"paths-to-backend\" data-composition-points=\"1028,243;1091.6,243;1091.6,367;1028,367\" d=\"M 1028 243 L 1091.6 243 L 1091.6 367 L 1028 367\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"problem\" data-edge-to=\"options\" data-edge-key=\"5\" data-edge-id=\"problem-to-options\" data-composition-points=\"188,119;216,119\" d=\"M 188 119 L 216 119\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"why-rebuild\" data-edge-to=\"risks-and-versions\" data-edge-key=\"6\" data-edge-id=\"rebuild-to-risks\" data-composition-points=\"552,367;524,367\" d=\"M 552 367 L 524 367\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"request-flow\" data-edge-to=\"mounted-paths\" data-edge-key=\"7\" data-edge-id=\"request-to-paths\" data-composition-points=\"860,243;888,243\" d=\"M 860 243 L 888 243\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"risks-and-versions\" data-edge-to=\"roadmap\" data-edge-key=\"8\" data-edge-id=\"risks-to-roadmap\" data-composition-points=\"384,367;356,367\" d=\"M 384 367 L 356 367\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"versioning\" data-edge-to=\"why-rebuild\" data-edge-key=\"9\" data-edge-id=\"versioning-to-rebuild\" data-composition-points=\"720,367;692,367\" d=\"M 720 367 L 692 367\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"where-personas-live\" data-edge-to=\"what-we-built\" data-edge-key=\"10\" data-edge-id=\"where-to-built\" data-composition-points=\"524,119;552,119\" d=\"M 524 119 L 552 119\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n\n        \u003c!-- Nodes --\u003e\n        \u003cg id=\"node-problem\" data-node-id=\"problem\" data-node-label=\"The problem\" tabindex=\"0\" role=\"button\" aria-label=\"Focus The problem, personas per franchise, Where it lives\" aria-pressed=\"false\" data-node-kind=\"external\" data-node-sublabel=\"personas per franchise\" data-node-context=\"Where it lives\"\u003e\n          \u003ctitle\u003eThe problem · personas per franchise · Where it lives\u003c/title\u003e\n          \u003crect x=\"48\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"48\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-external\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"external\" class=\"semantic-sigil s-external\" transform=\"translate(54 99) scale(0.6875)\"\u003e\n            \u003crect x=\"2.5\" y=\"5\" width=\"8.5\" height=\"8\" rx=\"1.5\"/\u003e\n            \u003cpath d=\"M8 2.5h5.5V8M13.5 2.5 7.5 8.5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"118\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eThe problem\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"118\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003epersonas per franchise\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-options\" data-node-id=\"options\" data-node-label=\"Three options\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Three options, files, field, assistant, Where it lives\" aria-pressed=\"false\" data-node-kind=\"messagebus\" data-node-sublabel=\"files, field, assistant\" data-node-context=\"Where it lives\"\u003e\n          \u003ctitle\u003eThree options · files, field, assistant · Where it lives\u003c/title\u003e\n          \u003crect x=\"216\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"216\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-messagebus\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"messagebus\" class=\"semantic-sigil s-messagebus\" transform=\"translate(222 99) scale(0.6875)\"\u003e\n            \u003cpath d=\"M2.5 4.5h11M2.5 8h11M2.5 11.5h11\"/\u003e\n            \u003ccircle cx=\"5\" cy=\"4.5\" r=\"1\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"10.5\" cy=\"8\" r=\"1\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"7\" cy=\"11.5\" r=\"1\" class=\"sigil-fill\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"286\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eThree options\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"286\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003efiles, field, assistant\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-where-personas-live\" data-node-id=\"where-personas-live\" data-node-label=\"Where they live\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Where they live, persona + config repos, Where it lives\" aria-pressed=\"false\" data-node-kind=\"cloud\" data-node-sublabel=\"persona + config repos\" data-node-context=\"Where it lives\"\u003e\n          \u003ctitle\u003eWhere they live · persona + config repos · Where it lives\u003c/title\u003e\n          \u003crect x=\"384\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"384\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-cloud\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"cloud\" class=\"semantic-sigil s-cloud\" transform=\"translate(390 99) scale(0.6875)\"\u003e\n            \u003cpath d=\"M4.3 12.5h7.3a2.4 2.4 0 0 0 .2-4.8 4 4 0 0 0-7.5-1.3A3.1 3.1 0 0 0 4.3 12.5Z\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"454\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eWhere they live\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"454\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003epersona + config repos\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-what-we-built\" data-node-id=\"what-we-built\" data-node-label=\"What we built\" tabindex=\"0\" role=\"button\" aria-label=\"Focus What we built, HubSkillsBackend, Where it lives\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"HubSkillsBackend\" data-node-context=\"Where it lives\"\u003e\n          \u003ctitle\u003eWhat we built · HubSkillsBackend · Where it lives\u003c/title\u003e\n          \u003crect x=\"552\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"552\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(558 99) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"622\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eWhat we built\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"622\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eHubSkillsBackend\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-contracts\" data-node-id=\"contracts\" data-node-label=\"Contracts\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Contracts, who writes, who reads, How it loads\" aria-pressed=\"false\" data-node-kind=\"frontend\" data-node-sublabel=\"who writes, who reads\" data-node-context=\"How it loads\"\u003e\n          \u003ctitle\u003eContracts · who writes, who reads · How it loads\u003c/title\u003e\n          \u003crect x=\"552\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"552\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-frontend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"frontend\" class=\"semantic-sigil s-frontend\" transform=\"translate(558 223) scale(0.6875)\"\u003e\n            \u003crect x=\"2\" y=\"3\" width=\"12\" height=\"10\" rx=\"2\"/\u003e\n            \u003cpath d=\"M2 6.5h12\"/\u003e\n            \u003ccircle cx=\"4.1\" cy=\"4.8\" r=\".7\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"6.3\" cy=\"4.8\" r=\".7\" class=\"sigil-fill\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"622\" y=\"238\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eContracts\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"622\" y=\"255\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003ewho writes, who reads\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-request-flow\" data-node-id=\"request-flow\" data-node-label=\"One request\" tabindex=\"0\" role=\"button\" aria-label=\"Focus One request, factory to backend, How it loads\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"factory to backend\" data-node-context=\"How it loads\"\u003e\n          \u003ctitle\u003eOne request · factory to backend · How it loads\u003c/title\u003e\n          \u003crect x=\"720\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"720\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(726 223) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"790\" y=\"238\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eOne request\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"790\" y=\"255\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003efactory to backend\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-mounted-paths\" data-node-id=\"mounted-paths\" data-node-label=\"Mounted paths\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Mounted paths, /persona/ and friends, How it loads\" aria-pressed=\"false\" data-node-kind=\"database\" data-node-sublabel=\"/persona/ and friends\" data-node-context=\"How it loads\"\u003e\n          \u003ctitle\u003eMounted paths · /persona/ and friends · How it loads\u003c/title\u003e\n          \u003crect x=\"888\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"888\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-database\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"database\" class=\"semantic-sigil s-database\" transform=\"translate(894 223) scale(0.6875)\"\u003e\n            \u003cellipse cx=\"8\" cy=\"4\" rx=\"5\" ry=\"2\"/\u003e\n            \u003cpath d=\"M3 4v8c0 1.1 2.2 2 5 2s5-.9 5-2V4M3 8c0 1.1 2.2 2 5 2s5-.9 5-2\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"958\" y=\"238\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eMounted paths\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"958\" y=\"255\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003e/persona/ and friends\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-roadmap\" data-node-id=\"roadmap\" data-node-label=\"Roadmap\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Roadmap, v1, v1.1, v2, How it runs and ships\" aria-pressed=\"false\" data-node-kind=\"external\" data-node-sublabel=\"v1, v1.1, v2\" data-node-context=\"How it runs and ships\"\u003e\n          \u003ctitle\u003eRoadmap · v1, v1.1, v2 · How it runs and ships\u003c/title\u003e\n          \u003crect x=\"216\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"216\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-external\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"external\" class=\"semantic-sigil s-external\" transform=\"translate(222 347) scale(0.6875)\"\u003e\n            \u003crect x=\"2.5\" y=\"5\" width=\"8.5\" height=\"8\" rx=\"1.5\"/\u003e\n            \u003cpath d=\"M8 2.5h5.5V8M13.5 2.5 7.5 8.5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"286\" y=\"362\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eRoadmap\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"286\" y=\"379\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003ev1, v1.1, v2\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-risks-and-versions\" data-node-id=\"risks-and-versions\" data-node-label=\"Risks \u0026amp; versions\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Risks \u0026amp; versions, bugs and fix versions, How it runs and ships\" aria-pressed=\"false\" data-node-kind=\"security\" data-node-sublabel=\"bugs and fix versions\" data-node-context=\"How it runs and ships\"\u003e\n          \u003ctitle\u003eRisks \u0026amp; versions · bugs and fix versions · How it runs and ships\u003c/title\u003e\n          \u003crect x=\"384\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"384\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-security\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"security\" class=\"semantic-sigil s-security\" transform=\"translate(390 347) scale(0.6875)\"\u003e\n            \u003cpath d=\"M8 2.2 13 4v3.5c0 3.1-1.8 5.4-5 6.5-3.2-1.1-5-3.4-5-6.5V4Z\"/\u003e\n            \u003cpath d=\"m5.8 8 1.5 1.5 3-3\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"454\" y=\"362\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eRisks \u0026amp; versions\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"454\" y=\"379\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003ebugs and fix versions\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-why-rebuild\" data-node-id=\"why-rebuild\" data-node-label=\"Why rebuild\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Why rebuild, cost and authorization, How it runs and ships\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"cost and authorization\" data-node-context=\"How it runs and ships\"\u003e\n          \u003ctitle\u003eWhy rebuild · cost and authorization · How it runs and ships\u003c/title\u003e\n          \u003crect x=\"552\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"552\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(558 347) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"622\" y=\"362\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eWhy rebuild\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"622\" y=\"379\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003ecost and authorization\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-versioning\" data-node-id=\"versioning\" data-node-label=\"Versioning\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Versioning, dev, staging, production, How it runs and ships\" aria-pressed=\"false\" data-node-kind=\"messagebus\" data-node-sublabel=\"dev, staging, production\" data-node-context=\"How it runs and ships\"\u003e\n          \u003ctitle\u003eVersioning · dev, staging, production · How it runs and ships\u003c/title\u003e\n          \u003crect x=\"720\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"720\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-messagebus\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"messagebus\" class=\"semantic-sigil s-messagebus\" transform=\"translate(726 347) scale(0.6875)\"\u003e\n            \u003cpath d=\"M2.5 4.5h11M2.5 8h11M2.5 11.5h11\"/\u003e\n            \u003ccircle cx=\"5\" cy=\"4.5\" r=\"1\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"10.5\" cy=\"8\" r=\"1\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"7\" cy=\"11.5\" r=\"1\" class=\"sigil-fill\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"790\" y=\"362\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eVersioning\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"790\" y=\"379\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003edev, staging, production\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-persona-backend\" data-node-id=\"persona-backend\" data-node-label=\"Loading a persona\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Loading a persona, shared Hub backends, How it runs and ships\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"shared Hub backends\" data-node-context=\"How it runs and ships\"\u003e\n          \u003ctitle\u003eLoading a persona · shared Hub backends · How it runs and ships\u003c/title\u003e\n          \u003crect x=\"888\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"888\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(894 347) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"958\" y=\"362\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eLoading a persona\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"958\" y=\"379\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eshared Hub backends\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003c!-- Edge labels --\u003e\n\n        \u003cg data-detail=\"context\" data-edge-from=\"what-we-built\" data-edge-to=\"contracts\" data-edge-label=\"Reuse confirmed\" data-edge-key=\"1\" data-edge-id=\"built-to-contracts\"\u003e\n          \u003crect x=\"624\" y=\"146\" width=\"82\" height=\"14\" rx=\"3\" class=\"c-mask\"/\u003e\n          \u003ctext x=\"665\" y=\"156\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eReuse confirmed\u003c/text\u003e\n        \u003c/g\u003e\n\n\n        \u003cg data-detail=\"context\" data-edge-from=\"mounted-paths\" data-edge-to=\"persona-backend\" data-edge-label=\"Route traced\" data-edge-key=\"4\" data-edge-id=\"paths-to-backend\"\u003e\n          \u003crect x=\"1026\" y=\"223\" width=\"67.6\" height=\"14\" rx=\"3\" class=\"c-mask\"/\u003e\n          \u003ctext x=\"1059.8\" y=\"233\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eRoute traced\u003c/text\u003e\n        \u003c/g\u003e\n\n\n\n\n\n\n\n        \u003c!-- Legend --\u003e\n        \u003cg data-legend=\"\" data-legend-bridge=\"\"\u003e\n          \u003ctext x=\"20\" y=\"428\" class=\"t-primary\" font-size=\"12\" font-weight=\"650\"\u003eLegend\u003c/text\u003e\n          \u003cg data-legend-semantic-kind=\"frontend\" data-legend-kind=\"frontend\" data-legend-label=\"User UI\" data-legend-x=\"20\" data-legend-baseline=\"448\" data-legend-width=\"74\"\u003e\n            \u003crect x=\"20\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-frontend\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"42\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eUser UI\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"backend\" data-legend-kind=\"backend\" data-legend-label=\"Agent logic\" data-legend-x=\"101\" data-legend-baseline=\"448\" data-legend-width=\"91\"\u003e\n            \u003crect x=\"101\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-backend\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"123\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eAgent logic\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"security\" data-legend-kind=\"security\" data-legend-label=\"Policy\" data-legend-x=\"199\" data-legend-baseline=\"448\" data-legend-width=\"70\"\u003e\n            \u003crect x=\"199\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-security\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"221\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003ePolicy\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"messagebus\" data-legend-kind=\"messagebus\" data-legend-label=\"Tool action\" data-legend-x=\"276\" data-legend-baseline=\"448\" data-legend-width=\"91\"\u003e\n            \u003crect x=\"276\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-messagebus\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"298\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eTool action\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"database\" data-legend-kind=\"database\" data-legend-label=\"Context / trace\" data-legend-x=\"374\" data-legend-baseline=\"448\" data-legend-width=\"109\"\u003e\n            \u003crect x=\"374\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-database\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"396\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eContext / trace\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"cloud\" data-legend-kind=\"cloud\" data-legend-label=\"Cloud service\" data-legend-x=\"490\" data-legend-baseline=\"448\" data-legend-width=\"100\"\u003e\n            \u003crect x=\"490\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-cloud\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"512\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eCloud service\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"external\" data-legend-kind=\"external\" data-legend-label=\"External system\" data-legend-x=\"597\" data-legend-baseline=\"448\" data-legend-width=\"109\"\u003e\n            \u003crect x=\"597\" y=\"440\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-external\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"619\" y=\"448\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eExternal system\u003c/text\u003e\n          \u003c/g\u003e\n        \u003c/g\u003e\n      \u003c/svg\u003e","lanes":[{"id":"where","label":"Where it lives"},{"id":"load","label":"How it loads"},{"id":"decide","label":"How it runs and ships"}],"workflow_nodes":[{"id":"problem","label":"The problem","lane":"where"},{"id":"options","label":"Three options","lane":"where"},{"id":"where-personas-live","label":"Where they live","lane":"where"},{"id":"what-we-built","label":"What we built","lane":"where"},{"id":"contracts","label":"Contracts","lane":"load"},{"id":"request-flow","label":"One request","lane":"load"},{"id":"mounted-paths","label":"Mounted paths","lane":"load"},{"id":"persona-backend","label":"Loading a persona","lane":"decide"},{"id":"versioning","label":"Versioning","lane":"decide"},{"id":"why-rebuild","label":"Why rebuild","lane":"decide"},{"id":"risks-and-versions","label":"Risks \u0026 versions","lane":"decide"},{"id":"roadmap","label":"Roadmap","lane":"decide"}]},{"id":"rails","workflow":"persona-rails.workflow.json","title":"B. Contracts: who writes, who reads","caption":"The same design split by stakeholder. Franchises publish, TwinShield authorizes, the backend calls, the orchestrator loads.","detailed":false,"node_map":{"rails-persona":"where-personas-live","rails-config":"contracts","rails-promote":"versioning","rails-read":"contracts","rails-authorized":"contracts","rails-list":"contracts","rails-call":"contracts","rails-check":"request-flow","rails-load":"persona-backend"},"svg":"\u003csvg viewBox=\"0 0 1180 700\" role=\"img\" lang=\"en\" aria-labelledby=\"archify-diagram-title archify-diagram-description\" data-preset=\"classic\" data-quality-profile=\"showcase\"\u003e\n        \u003ctitle id=\"archify-diagram-title\"\u003eContracts: who writes, who reads\u003c/title\u003e\n        \u003cdesc id=\"archify-diagram-description\"\u003eA workflow diagram generated by Archify.\u003c/desc\u003e\n        \u003c!-- Definitions --\u003e\n        \u003cdefs\u003e\n          \u003cmarker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-default\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-emphasis\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-emphasis\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-security\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-security\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-dashed\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-dashed\" /\u003e\n          \u003c/marker\u003e\n          \u003cpattern id=\"grid\" width=\"40\" height=\"40\" patternUnits=\"userSpaceOnUse\"\u003e\n            \u003cpath d=\"M 40 0 L 0 0 0 40\" class=\"c-grid\" stroke-width=\"0.5\"/\u003e\n          \u003c/pattern\u003e\n        \u003c/defs\u003e\n\n        \u003c!-- Background Grid --\u003e\n        \u003crect width=\"100%\" height=\"100%\" fill=\"url(#grid)\" /\u003e\n\n        \u003c!-- Swimlanes --\u003e\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-0\" x=\"40\" y=\"52\" width=\"996\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"74\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e01 / Franchise\u003c/text\u003e\n\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-1\" x=\"40\" y=\"176\" width=\"996\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"198\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e02 / TwinShield\u003c/text\u003e\n\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-2\" x=\"40\" y=\"300\" width=\"996\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"322\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e03 / Backend (BFF)\u003c/text\u003e\n\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-3\" x=\"40\" y=\"424\" width=\"996\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"446\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e04 / twinCore orchestrator\u003c/text\u003e\n\n        \u003c!-- Phase headers --\u003e\n\n\n        \u003c!-- Workflow groups --\u003e\n\n\n        \u003c!-- Edge paths --\u003e\n        \u003cpath data-edge-from=\"rails-authorized\" data-edge-to=\"rails-list\" data-edge-label=\"Personas returned\" data-edge-key=\"0\" data-edge-id=\"rails-authorized-list\" data-composition-points=\"692,243;708,243;708,290;622,290;622,341\" d=\"M 692 243 L 708 243 L 708 290 L 622 290 L 622 341\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"rails-call\" data-edge-to=\"rails-check\" data-edge-label=\"Run request\" data-edge-key=\"1\" data-edge-id=\"rails-call-check\" data-composition-points=\"860,367;876,367;876,414;790,414;790,465\" d=\"M 860 367 L 876 367 L 876 414 L 790 414 L 790 465\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"rails-check\" data-edge-to=\"rails-load\" data-edge-key=\"2\" data-edge-id=\"rails-check-load\" data-composition-points=\"860,491;888,491\" d=\"M 860 491 L 888 491\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"rails-config\" data-edge-to=\"rails-promote\" data-edge-key=\"3\" data-edge-id=\"rails-config-promote\" data-composition-points=\"356,119;384,119\" d=\"M 356 119 L 384 119\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"rails-list\" data-edge-to=\"rails-call\" data-edge-key=\"4\" data-edge-id=\"rails-list-call\" data-composition-points=\"692,367;720,367\" d=\"M 692 367 L 720 367\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"rails-persona\" data-edge-to=\"rails-config\" data-edge-key=\"5\" data-edge-id=\"rails-persona-config\" data-composition-points=\"188,119;216,119\" d=\"M 188 119 L 216 119\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"rails-promote\" data-edge-to=\"rails-read\" data-edge-label=\"Published\" data-edge-key=\"6\" data-edge-id=\"rails-promote-read\" data-composition-points=\"524,119;540,119;540,166;454,166;454,217\" d=\"M 524 119 L 540 119 L 540 166 L 454 166 L 454 217\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"rails-read\" data-edge-to=\"rails-authorized\" data-edge-key=\"7\" data-edge-id=\"rails-read-authorized\" data-composition-points=\"524,243;552,243\" d=\"M 524 243 L 552 243\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n\n        \u003c!-- Nodes --\u003e\n        \u003cg id=\"node-rails-persona\" data-node-id=\"rails-persona\" data-node-label=\"Persona repo\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Persona repo, AGENTS.md + skills/, Franchise\" aria-pressed=\"false\" data-node-kind=\"cloud\" data-node-sublabel=\"AGENTS.md + skills/\" data-node-context=\"Franchise\"\u003e\n          \u003ctitle\u003ePersona repo · AGENTS.md + skills/ · Franchise\u003c/title\u003e\n          \u003crect x=\"48\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"48\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-cloud\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"cloud\" class=\"semantic-sigil s-cloud\" transform=\"translate(54 99) scale(0.6875)\"\u003e\n            \u003cpath d=\"M4.3 12.5h7.3a2.4 2.4 0 0 0 .2-4.8 4 4 0 0 0-7.5-1.3A3.1 3.1 0 0 0 4.3 12.5Z\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"118\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003ePersona repo\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"118\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eAGENTS.md + skills/\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-rails-config\" data-node-id=\"rails-config\" data-node-label=\"config.json\" tabindex=\"0\" role=\"button\" aria-label=\"Focus config.json, persona registry, Franchise\" aria-pressed=\"false\" data-node-kind=\"cloud\" data-node-sublabel=\"persona registry\" data-node-context=\"Franchise\"\u003e\n          \u003ctitle\u003econfig.json · persona registry · Franchise\u003c/title\u003e\n          \u003crect x=\"216\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"216\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-cloud\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"cloud\" class=\"semantic-sigil s-cloud\" transform=\"translate(222 99) scale(0.6875)\"\u003e\n            \u003cpath d=\"M4.3 12.5h7.3a2.4 2.4 0 0 0 .2-4.8 4 4 0 0 0-7.5-1.3A3.1 3.1 0 0 0 4.3 12.5Z\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"286\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003econfig.json\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"286\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003epersona registry\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-rails-promote\" data-node-id=\"rails-promote\" data-node-label=\"Tag\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Tag, dev, staging, production, Franchise\" aria-pressed=\"false\" data-node-kind=\"messagebus\" data-node-sublabel=\"dev, staging, production\" data-node-context=\"Franchise\"\u003e\n          \u003ctitle\u003eTag · dev, staging, production · Franchise\u003c/title\u003e\n          \u003crect x=\"384\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"384\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-messagebus\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"messagebus\" class=\"semantic-sigil s-messagebus\" transform=\"translate(390 99) scale(0.6875)\"\u003e\n            \u003cpath d=\"M2.5 4.5h11M2.5 8h11M2.5 11.5h11\"/\u003e\n            \u003ccircle cx=\"5\" cy=\"4.5\" r=\"1\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"10.5\" cy=\"8\" r=\"1\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"7\" cy=\"11.5\" r=\"1\" class=\"sigil-fill\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"454\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eTag\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"454\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003edev, staging, production\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-rails-read\" data-node-id=\"rails-read\" data-node-label=\"Reads registry\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Reads registry, config at each tag, TwinShield\" aria-pressed=\"false\" data-node-kind=\"security\" data-node-sublabel=\"config at each tag\" data-node-context=\"TwinShield\"\u003e\n          \u003ctitle\u003eReads registry · config at each tag · TwinShield\u003c/title\u003e\n          \u003crect x=\"384\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"384\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-security\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"security\" class=\"semantic-sigil s-security\" transform=\"translate(390 223) scale(0.6875)\"\u003e\n            \u003cpath d=\"M8 2.2 13 4v3.5c0 3.1-1.8 5.4-5 6.5-3.2-1.1-5-3.4-5-6.5V4Z\"/\u003e\n            \u003cpath d=\"m5.8 8 1.5 1.5 3-3\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"454\" y=\"238\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eReads registry\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"454\" y=\"255\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003econfig at each tag\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-rails-authorized\" data-node-id=\"rails-authorized\" data-node-label=\"Authorized personas\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Authorized personas, per user and role, TwinShield\" aria-pressed=\"false\" data-node-kind=\"security\" data-node-sublabel=\"per user and role\" data-node-context=\"TwinShield\"\u003e\n          \u003ctitle\u003eAuthorized personas · per user and role · TwinShield\u003c/title\u003e\n          \u003crect x=\"552\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"552\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-security\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"security\" class=\"semantic-sigil s-security\" transform=\"translate(558 223) scale(0.6875)\"\u003e\n            \u003cpath d=\"M8 2.2 13 4v3.5c0 3.1-1.8 5.4-5 6.5-3.2-1.1-5-3.4-5-6.5V4Z\"/\u003e\n            \u003cpath d=\"m5.8 8 1.5 1.5 3-3\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"622\" y=\"238\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eAuthorized personas\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"622\" y=\"255\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eper user and role\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-rails-list\" data-node-id=\"rails-list\" data-node-label=\"Lists personas\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Lists personas, UI catalog, Backend (BFF)\" aria-pressed=\"false\" data-node-kind=\"frontend\" data-node-sublabel=\"UI catalog\" data-node-context=\"Backend (BFF)\"\u003e\n          \u003ctitle\u003eLists personas · UI catalog · Backend (BFF)\u003c/title\u003e\n          \u003crect x=\"552\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"552\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-frontend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"frontend\" class=\"semantic-sigil s-frontend\" transform=\"translate(558 347) scale(0.6875)\"\u003e\n            \u003crect x=\"2\" y=\"3\" width=\"12\" height=\"10\" rx=\"2\"/\u003e\n            \u003cpath d=\"M2 6.5h12\"/\u003e\n            \u003ccircle cx=\"4.1\" cy=\"4.8\" r=\".7\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"6.3\" cy=\"4.8\" r=\".7\" class=\"sigil-fill\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"622\" y=\"362\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eLists personas\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"622\" y=\"379\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eUI catalog\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-rails-call\" data-node-id=\"rails-call\" data-node-label=\"Calls orchestrator\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Calls orchestrator, context.persona, Backend (BFF)\" aria-pressed=\"false\" data-node-kind=\"frontend\" data-node-sublabel=\"context.persona\" data-node-context=\"Backend (BFF)\"\u003e\n          \u003ctitle\u003eCalls orchestrator · context.persona · Backend (BFF)\u003c/title\u003e\n          \u003crect x=\"720\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"720\" y=\"341\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-frontend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"frontend\" class=\"semantic-sigil s-frontend\" transform=\"translate(726 347) scale(0.6875)\"\u003e\n            \u003crect x=\"2\" y=\"3\" width=\"12\" height=\"10\" rx=\"2\"/\u003e\n            \u003cpath d=\"M2 6.5h12\"/\u003e\n            \u003ccircle cx=\"4.1\" cy=\"4.8\" r=\".7\" class=\"sigil-fill\"/\u003e\n            \u003ccircle cx=\"6.3\" cy=\"4.8\" r=\".7\" class=\"sigil-fill\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"790\" y=\"362\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eCalls orchestrator\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"790\" y=\"379\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003econtext.persona\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-rails-check\" data-node-id=\"rails-check\" data-node-label=\"Trusts request\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Trusts request, check in habilitation v2, twinCore orchestrator\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"check in habilitation v2\" data-node-context=\"twinCore orchestrator\"\u003e\n          \u003ctitle\u003eTrusts request · check in habilitation v2 · twinCore orchestrator\u003c/title\u003e\n          \u003crect x=\"720\" y=\"465\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"720\" y=\"465\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(726 471) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"790\" y=\"486\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eTrusts request\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"790\" y=\"503\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003echeck in habilitation v2\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-rails-load\" data-node-id=\"rails-load\" data-node-label=\"Loads persona\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Loads persona, repo at its tag, twinCore orchestrator\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"repo at its tag\" data-node-context=\"twinCore orchestrator\"\u003e\n          \u003ctitle\u003eLoads persona · repo at its tag · twinCore orchestrator\u003c/title\u003e\n          \u003crect x=\"888\" y=\"465\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"888\" y=\"465\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(894 471) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"958\" y=\"486\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eLoads persona\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"958\" y=\"503\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003erepo at its tag\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003c!-- Edge labels --\u003e\n        \u003cg data-detail=\"context\" data-edge-from=\"rails-authorized\" data-edge-to=\"rails-list\" data-edge-label=\"Personas returned\" data-edge-key=\"0\" data-edge-id=\"rails-authorized-list\"\u003e\n          \u003crect x=\"619.2\" y=\"270\" width=\"91.6\" height=\"14\" rx=\"3\" class=\"c-mask\"/\u003e\n          \u003ctext x=\"665\" y=\"280\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003ePersonas returned\u003c/text\u003e\n        \u003c/g\u003e\n        \u003cg data-detail=\"context\" data-edge-from=\"rails-call\" data-edge-to=\"rails-check\" data-edge-label=\"Run request\" data-edge-key=\"1\" data-edge-id=\"rails-call-check\"\u003e\n          \u003crect x=\"801.6\" y=\"394\" width=\"62.8\" height=\"14\" rx=\"3\" class=\"c-mask\"/\u003e\n          \u003ctext x=\"833\" y=\"404\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eRun request\u003c/text\u003e\n        \u003c/g\u003e\n\n\n\n\n        \u003cg data-detail=\"context\" data-edge-from=\"rails-promote\" data-edge-to=\"rails-read\" data-edge-label=\"Published\" data-edge-key=\"6\" data-edge-id=\"rails-promote-read\"\u003e\n          \u003crect x=\"470.4\" y=\"146\" width=\"53.199999999999996\" height=\"14\" rx=\"3\" class=\"c-mask\"/\u003e\n          \u003ctext x=\"497\" y=\"156\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003ePublished\u003c/text\u003e\n        \u003c/g\u003e\n\n\n        \u003c!-- Legend --\u003e\n        \u003cg data-legend=\"\" data-legend-bridge=\"\"\u003e\n          \u003ctext x=\"20\" y=\"552\" class=\"t-primary\" font-size=\"12\" font-weight=\"650\"\u003eLegend\u003c/text\u003e\n          \u003cg data-legend-semantic-kind=\"frontend\" data-legend-kind=\"frontend\" data-legend-label=\"User UI\" data-legend-x=\"20\" data-legend-baseline=\"572\" data-legend-width=\"74\"\u003e\n            \u003crect x=\"20\" y=\"564\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-frontend\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"42\" y=\"572\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eUser UI\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"backend\" data-legend-kind=\"backend\" data-legend-label=\"Agent logic\" data-legend-x=\"101\" data-legend-baseline=\"572\" data-legend-width=\"91\"\u003e\n            \u003crect x=\"101\" y=\"564\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-backend\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"123\" y=\"572\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eAgent logic\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"security\" data-legend-kind=\"security\" data-legend-label=\"Policy\" data-legend-x=\"199\" data-legend-baseline=\"572\" data-legend-width=\"70\"\u003e\n            \u003crect x=\"199\" y=\"564\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-security\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"221\" y=\"572\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003ePolicy\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"messagebus\" data-legend-kind=\"messagebus\" data-legend-label=\"Tool action\" data-legend-x=\"276\" data-legend-baseline=\"572\" data-legend-width=\"91\"\u003e\n            \u003crect x=\"276\" y=\"564\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-messagebus\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"298\" y=\"572\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eTool action\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"cloud\" data-legend-kind=\"cloud\" data-legend-label=\"Cloud service\" data-legend-x=\"374\" data-legend-baseline=\"572\" data-legend-width=\"100\"\u003e\n            \u003crect x=\"374\" y=\"564\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-cloud\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"396\" y=\"572\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eCloud service\u003c/text\u003e\n          \u003c/g\u003e\n        \u003c/g\u003e\n      \u003c/svg\u003e","lanes":[{"id":"franchise","label":"Franchise"},{"id":"twinshield","label":"TwinShield"},{"id":"bff","label":"Backend (BFF)"},{"id":"twincore","label":"twinCore orchestrator"}],"workflow_nodes":[{"id":"rails-persona","label":"Persona repo","lane":"franchise"},{"id":"rails-config","label":"config.json","lane":"franchise"},{"id":"rails-promote","label":"Tag","lane":"franchise"},{"id":"rails-read","label":"Reads registry","lane":"twinshield"},{"id":"rails-authorized","label":"Authorized personas","lane":"twinshield"},{"id":"rails-list","label":"Lists personas","lane":"bff"},{"id":"rails-call","label":"Calls orchestrator","lane":"bff"},{"id":"rails-check","label":"Trusts request","lane":"twincore"},{"id":"rails-load","label":"Loads persona","lane":"twincore"}]},{"id":"ladder","workflow":"persona-ladder.workflow.json","title":"C. Release ladder: v1, v1.1, v2","caption":"Each release has an exit check that earns the next one. Nothing in v2 is built before its gates pass.","detailed":false,"node_map":{"ladder-v1":"roadmap","ladder-check1":"versioning","ladder-v11":"versioning","ladder-check2":"versioning","ladder-v2":"roadmap","ladder-check3":"risks-and-versions"},"svg":"\u003csvg viewBox=\"0 0 1180 520\" role=\"img\" lang=\"en\" aria-labelledby=\"archify-diagram-title archify-diagram-description\" data-preset=\"classic\" data-quality-profile=\"showcase\"\u003e\n        \u003ctitle id=\"archify-diagram-title\"\u003eRelease ladder: v1, v1.1, v2\u003c/title\u003e\n        \u003cdesc id=\"archify-diagram-description\"\u003eA workflow diagram generated by Archify.\u003c/desc\u003e\n        \u003c!-- Definitions --\u003e\n        \u003cdefs\u003e\n          \u003cmarker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-default\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-emphasis\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-emphasis\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-security\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-security\" /\u003e\n          \u003c/marker\u003e\n          \u003cmarker id=\"arrowhead-dashed\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"\u003e\n            \u003cpolygon points=\"0 0, 10 3.5, 0 7\" class=\"m-dashed\" /\u003e\n          \u003c/marker\u003e\n          \u003cpattern id=\"grid\" width=\"40\" height=\"40\" patternUnits=\"userSpaceOnUse\"\u003e\n            \u003cpath d=\"M 40 0 L 0 0 0 40\" class=\"c-grid\" stroke-width=\"0.5\"/\u003e\n          \u003c/pattern\u003e\n        \u003c/defs\u003e\n\n        \u003c!-- Background Grid --\u003e\n        \u003crect width=\"100%\" height=\"100%\" fill=\"url(#grid)\" /\u003e\n\n        \u003c!-- Swimlanes --\u003e\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-0\" x=\"40\" y=\"52\" width=\"756\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"74\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e01 / Release\u003c/text\u003e\n\n        \u003crect data-graph-role=\"structural-frame\" data-composition-frame-kind=\"lane\" data-composition-frame-id=\"lane-1\" x=\"40\" y=\"176\" width=\"756\" height=\"104\" rx=\"10\" class=\"c-lane\" stroke-width=\"1\"/\u003e\n        \u003ctext x=\"54\" y=\"198\" class=\"t-dim\" font-size=\"10\" font-weight=\"600\"\u003e02 / Exit check\u003c/text\u003e\n\n        \u003c!-- Phase headers --\u003e\n\n\n        \u003c!-- Workflow groups --\u003e\n\n\n        \u003c!-- Edge paths --\u003e\n        \u003cpath data-edge-from=\"ladder-check1\" data-edge-to=\"ladder-v11\" data-edge-label=\"Ship v1.1\" data-edge-key=\"0\" data-edge-id=\"ladder-check1-v11\" data-composition-points=\"238,217;238,119;288,119\" d=\"M 238 217 L 238 119 L 288 119\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"ladder-check2\" data-edge-to=\"ladder-v2\" data-edge-label=\"Ship v2\" data-edge-key=\"1\" data-edge-id=\"ladder-check2-v2\" data-composition-points=\"478,217;478,119;528,119\" d=\"M 478 217 L 478 119 L 528 119\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"ladder-v1\" data-edge-to=\"ladder-check1\" data-edge-label=\"Measure\" data-edge-key=\"2\" data-edge-id=\"ladder-v1-check1\" data-composition-points=\"118,145;118,166;238,166;238,217\" d=\"M 118 145 L 118 166 L 238 166 L 238 217\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"ladder-v11\" data-edge-to=\"ladder-check2\" data-edge-label=\"Measure\" data-edge-key=\"3\" data-edge-id=\"ladder-v11-check2\" data-composition-points=\"358,145;358,243;408,243\" d=\"M 358 145 L 358 243 L 408 243\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n        \u003cpath data-edge-from=\"ladder-v2\" data-edge-to=\"ladder-check3\" data-edge-label=\"Verify\" data-edge-key=\"4\" data-edge-id=\"ladder-v2-check3\" data-composition-points=\"598,145;598,243;648,243\" d=\"M 598 145 L 598 243 L 648 243\" class=\"a-default\" stroke-width=\"1.4\" marker-end=\"url(#arrowhead)\"/\u003e\n\n        \u003c!-- Nodes --\u003e\n        \u003cg id=\"node-ladder-v1\" data-node-id=\"ladder-v1\" data-node-label=\"v1\" tabindex=\"0\" role=\"button\" aria-label=\"Focus v1, inline skills, three tags, Release\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"inline skills, three tags\" data-node-context=\"Release\"\u003e\n          \u003ctitle\u003ev1 · inline skills, three tags · Release\u003c/title\u003e\n          \u003crect x=\"48\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"48\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(54 99) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"118\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003ev1\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"118\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003einline skills, three tags\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-ladder-v11\" data-node-id=\"ladder-v11\" data-node-label=\"v1.1\" tabindex=\"0\" role=\"button\" aria-label=\"Focus v1.1, update reminder, Release\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"update reminder\" data-node-context=\"Release\"\u003e\n          \u003ctitle\u003ev1.1 · update reminder · Release\u003c/title\u003e\n          \u003crect x=\"288\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"288\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(294 99) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"358\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003ev1.1\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"358\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eupdate reminder\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-ladder-v2\" data-node-id=\"ladder-v2\" data-node-label=\"v2\" tabindex=\"0\" role=\"button\" aria-label=\"Focus v2, linked skills + pin, Release\" aria-pressed=\"false\" data-node-kind=\"backend\" data-node-sublabel=\"linked skills + pin\" data-node-context=\"Release\"\u003e\n          \u003ctitle\u003ev2 · linked skills + pin · Release\u003c/title\u003e\n          \u003crect x=\"528\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"528\" y=\"93\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-backend\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"backend\" class=\"semantic-sigil s-backend\" transform=\"translate(534 99) scale(0.6875)\"\u003e\n            \u003cpath d=\"M6 3 3 8l3 5M10 3l3 5-3 5\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"598\" y=\"114\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003ev2\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"598\" y=\"131\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003elinked skills + pin\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-ladder-check1\" data-node-id=\"ladder-check1\" data-node-label=\"Drift visible\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Drift visible, commit logged per turn, Exit check\" aria-pressed=\"false\" data-node-kind=\"security\" data-node-sublabel=\"commit logged per turn\" data-node-context=\"Exit check\"\u003e\n          \u003ctitle\u003eDrift visible · commit logged per turn · Exit check\u003c/title\u003e\n          \u003crect x=\"168\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"168\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-security\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"security\" class=\"semantic-sigil s-security\" transform=\"translate(174 223) scale(0.6875)\"\u003e\n            \u003cpath d=\"M8 2.2 13 4v3.5c0 3.1-1.8 5.4-5 6.5-3.2-1.1-5-3.4-5-6.5V4Z\"/\u003e\n            \u003cpath d=\"m5.8 8 1.5 1.5 3-3\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"238\" y=\"238\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eDrift visible\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"238\" y=\"255\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003ecommit logged per turn\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-ladder-check2\" data-node-id=\"ladder-check2\" data-node-label=\"Clean switches\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Clean switches, reload at turn edge, Exit check\" aria-pressed=\"false\" data-node-kind=\"security\" data-node-sublabel=\"reload at turn edge\" data-node-context=\"Exit check\"\u003e\n          \u003ctitle\u003eClean switches · reload at turn edge · Exit check\u003c/title\u003e\n          \u003crect x=\"408\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"408\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-security\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"security\" class=\"semantic-sigil s-security\" transform=\"translate(414 223) scale(0.6875)\"\u003e\n            \u003cpath d=\"M8 2.2 13 4v3.5c0 3.1-1.8 5.4-5 6.5-3.2-1.1-5-3.4-5-6.5V4Z\"/\u003e\n            \u003cpath d=\"m5.8 8 1.5 1.5 3-3\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"478\" y=\"238\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eClean switches\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"478\" y=\"255\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003ereload at turn edge\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003cg id=\"node-ladder-check3\" data-node-id=\"ladder-check3\" data-node-label=\"Gates passed\" tabindex=\"0\" role=\"button\" aria-label=\"Focus Gates passed, probe + governance, Exit check\" aria-pressed=\"false\" data-node-kind=\"security\" data-node-sublabel=\"probe + governance\" data-node-context=\"Exit check\"\u003e\n          \u003ctitle\u003eGates passed · probe + governance · Exit check\u003c/title\u003e\n          \u003crect x=\"648\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-mask\"/\u003e\n          \u003crect x=\"648\" y=\"217\" width=\"140\" height=\"52\" rx=\"6\" class=\"c-security\" stroke-width=\"1.5\"/\u003e\n          \u003cg aria-hidden=\"true\" data-semantic-sigil=\"security\" class=\"semantic-sigil s-security\" transform=\"translate(654 223) scale(0.6875)\"\u003e\n            \u003cpath d=\"M8 2.2 13 4v3.5c0 3.1-1.8 5.4-5 6.5-3.2-1.1-5-3.4-5-6.5V4Z\"/\u003e\n            \u003cpath d=\"m5.8 8 1.5 1.5 3-3\"/\u003e\n          \u003c/g\u003e\n          \u003ctext data-node-label=\"\" data-detail-anchor=\"\" x=\"718\" y=\"238\" class=\"t-primary\" font-size=\"11\" font-weight=\"600\" text-anchor=\"middle\"\u003eGates passed\u003c/text\u003e\n          \u003ctext data-detail=\"context\" x=\"718\" y=\"255\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eprobe + governance\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003c!-- Edge labels --\u003e\n        \u003cg data-detail=\"context\" data-edge-from=\"ladder-check1\" data-edge-to=\"ladder-v11\" data-edge-label=\"Ship v1.1\" data-edge-key=\"0\" data-edge-id=\"ladder-check1-v11\"\u003e\n          \u003crect x=\"236.4\" y=\"99\" width=\"53.199999999999996\" height=\"14\" rx=\"3\" class=\"c-mask\"/\u003e\n          \u003ctext x=\"263\" y=\"109\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eShip v1.1\u003c/text\u003e\n        \u003c/g\u003e\n        \u003cg data-detail=\"context\" data-edge-from=\"ladder-check2\" data-edge-to=\"ladder-v2\" data-edge-label=\"Ship v2\" data-edge-key=\"1\" data-edge-id=\"ladder-check2-v2\"\u003e\n          \u003crect x=\"481.2\" y=\"99\" width=\"43.6\" height=\"14\" rx=\"3\" class=\"c-mask\"/\u003e\n          \u003ctext x=\"503\" y=\"109\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eShip v2\u003c/text\u003e\n        \u003c/g\u003e\n        \u003cg data-detail=\"context\" data-edge-from=\"ladder-v1\" data-edge-to=\"ladder-check1\" data-edge-label=\"Measure\" data-edge-key=\"2\" data-edge-id=\"ladder-v1-check1\"\u003e\n          \u003crect x=\"156.2\" y=\"146\" width=\"43.6\" height=\"14\" rx=\"3\" class=\"c-mask\"/\u003e\n          \u003ctext x=\"178\" y=\"156\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eMeasure\u003c/text\u003e\n        \u003c/g\u003e\n        \u003cg data-detail=\"context\" data-edge-from=\"ladder-v11\" data-edge-to=\"ladder-check2\" data-edge-label=\"Measure\" data-edge-key=\"3\" data-edge-id=\"ladder-v11-check2\"\u003e\n          \u003crect x=\"361.2\" y=\"223\" width=\"43.6\" height=\"14\" rx=\"3\" class=\"c-mask\"/\u003e\n          \u003ctext x=\"383\" y=\"233\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eMeasure\u003c/text\u003e\n        \u003c/g\u003e\n        \u003cg data-detail=\"context\" data-edge-from=\"ladder-v2\" data-edge-to=\"ladder-check3\" data-edge-label=\"Verify\" data-edge-key=\"4\" data-edge-id=\"ladder-v2-check3\"\u003e\n          \u003crect x=\"603.6\" y=\"223\" width=\"38.8\" height=\"14\" rx=\"3\" class=\"c-mask\"/\u003e\n          \u003ctext x=\"623\" y=\"233\" class=\"t-muted\" font-size=\"8\" text-anchor=\"middle\"\u003eVerify\u003c/text\u003e\n        \u003c/g\u003e\n\n        \u003c!-- Legend --\u003e\n        \u003cg data-legend=\"\" data-legend-bridge=\"\"\u003e\n          \u003ctext x=\"20\" y=\"304\" class=\"t-primary\" font-size=\"12\" font-weight=\"650\"\u003eLegend\u003c/text\u003e\n          \u003cg data-legend-semantic-kind=\"backend\" data-legend-kind=\"backend\" data-legend-label=\"Agent logic\" data-legend-x=\"20\" data-legend-baseline=\"324\" data-legend-width=\"91\"\u003e\n            \u003crect x=\"20\" y=\"316\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-backend\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"42\" y=\"324\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003eAgent logic\u003c/text\u003e\n          \u003c/g\u003e\n          \u003cg data-legend-semantic-kind=\"security\" data-legend-kind=\"security\" data-legend-label=\"Policy\" data-legend-x=\"118\" data-legend-baseline=\"324\" data-legend-width=\"70\"\u003e\n            \u003crect x=\"118\" y=\"316\" width=\"14\" height=\"9\" rx=\"2\" class=\"c-security\" stroke-width=\"1\"/\u003e\n            \u003ctext x=\"140\" y=\"324\" class=\"t-muted\" font-size=\"7.5\" font-weight=\"500\"\u003ePolicy\u003c/text\u003e\n          \u003c/g\u003e\n        \u003c/g\u003e\n      \u003c/svg\u003e","lanes":[{"id":"release","label":"Release"},{"id":"check","label":"Exit check"}],"workflow_nodes":[{"id":"ladder-v1","label":"v1","lane":"release"},{"id":"ladder-check1","label":"Drift visible","lane":"check"},{"id":"ladder-v11","label":"v1.1","lane":"release"},{"id":"ladder-check2","label":"Clean switches","lane":"check"},{"id":"ladder-v2","label":"v2","lane":"release"},{"id":"ladder-check3","label":"Gates passed","lane":"check"}]}]}</script>
<script>
(() => {
  'use strict';
  const DATA = JSON.parse(document.getElementById('pj-data').textContent);
  const C = DATA.content;
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const byId = new Map(C.nodes.map((n) => [n.id, n]));
  const tracks = new Map(C.tracks.map((t) => [t.id, t]));
  const sources = new Map(C.sources.map((s) => [s.id, s]));
  const progression = C.edges.filter((e) => e.role !== 'return');
  const outgoing = (id) => progression.filter((e) => e.from === id);
  const incoming = (id) => progression.filter((e) => e.to === id);
  const starts = C.nodes.filter((n) => incoming(n.id).length === 0).map((n) => n.id);

  const el = (tag, cls, text) => {
    const x = document.createElement(tag);
    if (cls) x.className = cls;
    if (text !== undefined && text !== null) x.textContent = text;
    return x;
  };
  const live = (msg) => { document.getElementById('pj-live').textContent = msg; };

  function stepInfo(id) {
    const n = byId.get(id);
    const t = tracks.get(n.track_id);
    return { track: t, pos: t.display_order.indexOf(id) + 1, total: t.display_order.length };
  }
  const titleCounts = new Map();
  C.nodes.forEach((n) => titleCounts.set(n.title, (titleCounts.get(n.title) || 0) + 1));
  function accessibleName(id) {
    const n = byId.get(id);
    if (titleCounts.get(n.title) === 1) return n.title;
    const s = stepInfo(id);
    return n.title + ', ' + s.track.title + ', step ' + s.pos + ' of ' + s.total;
  }

  // ---- SVG sanitizer: fail closed on anything outside the renderer's passive vocabulary.
  const ALLOWED_EL = new Set(['svg', 'g', 'title', 'desc', 'defs', 'marker', 'polygon', 'polyline', 'pattern', 'path', 'rect', 'text', 'tspan', 'circle', 'ellipse', 'line']);
  const ALLOWED_ATTR = new Set(['viewBox', 'role', 'lang', 'aria-labelledby', 'aria-label', 'aria-hidden', 'aria-pressed', 'id', 'class', 'markerWidth', 'markerHeight',
    'refX', 'refY', 'orient', 'points', 'width', 'height', 'patternUnits', 'd', 'stroke-width', 'fill', 'stroke', 'x', 'y', 'x1', 'y1', 'x2', 'y2', 'dx', 'dy', 'rx', 'ry',
    'cx', 'cy', 'r', 'font-size', 'font-weight', 'text-anchor', 'transform', 'marker-end', 'marker-start', 'tabindex', 'opacity', 'stroke-dasharray', 'vector-effect']);
  const URL_ATTRS = new Set(['fill', 'stroke', 'marker-end', 'marker-start']);
  const ID_REF_ATTRS = new Set(['aria-labelledby']);
  const ID_RE = /^[A-Za-z][\w.-]*$/;

  function sanitizeSvg(text, prefix) {
    const source = /^<svg\b[^>]*\sxmlns=/.test(text) ? text : text.replace(/^<svg\b/, '<svg xmlns="' + SVG_NS + '"');
    const doc = new DOMParser().parseFromString(source, 'image/svg+xml');
    if (doc.getElementsByTagName('parsererror').length) throw new Error('the SVG does not parse');
    const root = doc.documentElement;
    if (root.namespaceURI !== SVG_NS || root.localName !== 'svg') throw new Error('the root is not an SVG element');
    const strip = [];
    const walker = doc.createTreeWalker(root, NodeFilter.SHOW_COMMENT | NodeFilter.SHOW_PROCESSING_INSTRUCTION);
    while (walker.nextNode()) strip.push(walker.currentNode);
    strip.forEach((n) => n.remove());
    const all = [root, ...root.querySelectorAll('*')];
    const ids = new Map();
    for (const node of all) {
      if (node.namespaceURI !== SVG_NS || !ALLOWED_EL.has(node.localName)) throw new Error('element not allowed: ' + node.localName);
      for (const a of [...node.attributes]) {
        const name = a.name;
        const value = a.value;
        if (name === 'xmlns') continue;
        const isData = /^data-[a-z0-9-]+$/.test(name);
        if (!isData && !ALLOWED_ATTR.has(name)) throw new Error('attribute not allowed: ' + name);
        if (/javascript:|data:|expression\(/i.test(value)) throw new Error('active value in ' + name);
        if (name === 'class' && !value.split(/\s+/).filter(Boolean).every((c) => /^[A-Za-z0-9_-]+$/.test(c))) throw new Error('unexpected class token');
        if (URL_ATTRS.has(name) && value.includes('url(') && !/^url\(#[A-Za-z][\w.-]*\)$/.test(value)) throw new Error('non-local reference in ' + name);
        if (name === 'id') {
          if (!ID_RE.test(value) || ids.has(value)) throw new Error('invalid or duplicate id');
          ids.set(value, null);
        }
      }
    }
    let counter = 0;
    ids.forEach((_, key) => ids.set(key, prefix + '-' + (counter++)));
    for (const node of all) {
      if (node.hasAttribute('id')) node.setAttribute('id', ids.get(node.getAttribute('id')));
      for (const name of URL_ATTRS) {
        const v = node.getAttribute(name);
        if (v && v.startsWith('url(')) {
          const ref = v.slice(5, -1);
          if (!ids.has(ref)) throw new Error('dangling reference ' + ref);
          node.setAttribute(name, 'url(#' + ids.get(ref) + ')');
        }
      }
      for (const name of ID_REF_ATTRS) {
        const v = node.getAttribute(name);
        if (v) {
          const mapped = v.split(/\s+/).filter(Boolean).map((ref) => {
            if (!ids.has(ref)) throw new Error('dangling reference ' + ref);
            return ids.get(ref);
          });
          node.setAttribute(name, mapped.join(' '));
        }
      }
    }
    return document.importNode(root, true);
  }

  // ---- Guided state (in memory; exploration never touches it).
  const guided = { history: [], completed: false };
  let lastOpener = null;

  const dialog = document.getElementById('pj-dialog');
  const supportsModal = typeof dialog.showModal === 'function';
  function openDialog() {
    if (dialog.open) return;
    if (supportsModal) dialog.showModal();
    else { dialog.setAttribute('open', ''); document.getElementById('pj-main').inert = true; }
    document.getElementById('pj-close').focus();
  }
  function closeDialog() {
    if (supportsModal) dialog.close();
    else { dialog.removeAttribute('open'); document.getElementById('pj-main').inert = false; restoreFocus(); }
  }
  function restoreFocus() { if (lastOpener && document.contains(lastOpener)) lastOpener.focus(); }
  dialog.addEventListener('close', restoreFocus);
  document.getElementById('pj-close').addEventListener('click', closeDialog);
  document.addEventListener('keydown', (ev) => { if (!supportsModal && ev.key === 'Escape' && dialog.hasAttribute('open')) closeDialog(); });

  function updateStatus() {
    const resume = document.getElementById('pj-resume');
    const status = document.getElementById('pj-status');
    const begin = document.getElementById('pj-begin');
    if (!guided.history.length) {
      resume.hidden = true;
      begin.textContent = 'Begin the guided path';
      status.textContent = 'Guided path not started. Click any node to explore its card.';
      return;
    }
    const current = guided.history[guided.history.length - 1];
    resume.hidden = false;
    begin.textContent = 'Restart the guided path';
    status.textContent = guided.completed
      ? 'Guided path complete. Restart it, or explore any card.'
      : 'Guided path: ' + guided.history.length + ' of ' + C.nodes.length + ' steps visited, current: ' + byId.get(current).title + '.';
  }

  function addParagraph(parent, label, text) {
    if (!text) return;
    const p = el('p');
    if (label) { p.append(el('span', 'pj-label', label + ': ')); }
    p.append(document.createTextNode(text));
    parent.append(p);
  }
  function addList(parent, items, cls) {
    if (!items || !items.length) return;
    const ul = el('ul', cls);
    items.forEach((t) => ul.append(el('li', null, t)));
    parent.append(ul);
  }
  function addTable(parent, table) {
    const wrap = el('div', 'pj-table-wrap');
    const t = el('table');
    const head = el('thead');
    const hr = el('tr');
    table.headers.forEach((h) => { const th = el('th', null, h); th.setAttribute('scope', 'col'); hr.append(th); });
    head.append(hr);
    const body = el('tbody');
    table.rows.forEach((row) => {
      const tr = el('tr');
      row.forEach((cell, i) => {
        const c = el(i === 0 ? 'th' : 'td', null, cell);
        if (i === 0) c.setAttribute('scope', 'row');
        tr.append(c);
      });
      body.append(tr);
    });
    t.append(head, body);
    wrap.append(t);
    parent.append(wrap);
  }

  async function copyText(text, button) {
    let ok = false;
    try {
      if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(text); ok = true; }
    } catch (_) { ok = false; }
    if (!ok) {
      const ta = el('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.className = 'pj-live';
      document.body.append(ta);
      ta.select();
      try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
      ta.remove();
    }
    button.textContent = ok ? 'Copied' : 'Copy failed: select the text';
    live(ok ? 'Snippet copied' : 'Copy failed, select the text manually');
    setTimeout(() => { button.textContent = 'Copy'; }, 2000);
  }

  function renderCard(id, ctx) {
    const n = byId.get(id);
    const s = stepInfo(id);
    document.getElementById('pj-card-meta').textContent = s.track.title + ' · step ' + s.pos + ' of ' + s.total + ' · ' + n.detail_level
      + (ctx.mode === 'guided' ? ' · guided path' : ' · exploring');
    document.getElementById('pj-card-title').textContent = n.title;
    const body = document.getElementById('pj-card-body');
    body.replaceChildren();
    if (ctx.origin) body.append(el('p', 'pj-meta', 'Opened from ' + ctx.origin + '.'));
    addParagraph(body, 'Objective', n.objective);
    addParagraph(body, null, n.summary);
    addParagraph(body, 'Why it matters', n.why);
    if (n.prerequisites && n.prerequisites.length) { body.append(el('h3', null, 'Prerequisites')); addList(body, n.prerequisites); }
    (n.sections || []).forEach((sec) => {
      let box;
      if (sec.collapsible) { box = el('details'); box.append(el('summary', null, sec.title)); }
      else { box = el('div'); box.append(el('h3', null, sec.title)); }
      addParagraph(box, null, sec.body);
      if (sec.preformatted) box.append(el('pre', null, sec.preformatted));
      if (sec.table) addTable(box, sec.table);
      addList(box, sec.bullets);
      body.append(box);
    });
    (n.snippets || []).forEach((snip) => {
      const box = el('div', 'pj-box');
      const head = el('div', 'pj-snippet-head');
      const label = el('span', 'pj-label', snip.label);
      label.append(el('span', 'pj-badge', snip.status));
      const copy = el('button', null, 'Copy');
      copy.type = 'button';
      copy.setAttribute('aria-label', 'Copy snippet: ' + snip.label);
      copy.addEventListener('click', () => copyText(snip.code, copy));
      head.append(label, copy);
      const pre = el('pre');
      pre.append(el('code', null, snip.code));
      box.append(head, pre);
      addParagraph(box, 'Expected', snip.expected);
      body.append(box);
    });
    const check = el('div', 'pj-box pj-check');
    addParagraph(check, 'Checkpoint', n.checkpoint);
    body.append(check);
    if (n.common_mistakes && n.common_mistakes.length) {
      const warn = el('div', 'pj-box pj-warn');
      warn.append(el('span', 'pj-label', 'Likely mistake'));
      addList(warn, n.common_mistakes);
      body.append(warn);
    }
    if (n.assumptions && n.assumptions.length) {
      body.append(el('h3', null, 'Assumptions (unconfirmed)'));
      addList(body, n.assumptions);
    }
    if (n.source_refs && n.source_refs.length) {
      body.append(el('h3', null, 'Sources'));
      addList(body, n.source_refs.map((r) => { const src = sources.get(r); return src.label + ' (' + src.location + ')'; }));
    }
    if (n.links && n.links.length) {
      const row = el('p');
      n.links.forEach((l) => {
        const b = el('button', null, l.label);
        b.type = 'button';
        b.addEventListener('click', () => openCard(l.node_id, { mode: 'explore', origin: 'the card "' + n.title + '"' }));
        row.append(b);
      });
      body.append(row);
    }
    renderNav(id, ctx);
    body.scrollTop = 0;
  }

  function renderNav(id, ctx) {
    const nav = document.getElementById('pj-card-nav');
    nav.replaceChildren();
    if (ctx.mode === 'guided') {
      const back = el('button', null, 'Back');
      back.type = 'button';
      back.disabled = guided.history.length < 2;
      back.addEventListener('click', () => {
        guided.history.pop();
        guided.completed = false;
        const prev = guided.history[guided.history.length - 1];
        updateStatus();
        renderCard(prev, { mode: 'guided' });
      });
      nav.append(back);
      const next = outgoing(id);
      if (!next.length) {
        const finish = el('button', 'pj-primary', 'Finish the path');
        finish.type = 'button';
        finish.addEventListener('click', () => { guided.completed = true; updateStatus(); live('Guided path complete'); closeDialog(); });
        nav.append(finish);
      }
      next.forEach((edge) => {
        const target = byId.get(edge.to);
        const label = (edge.label ? edge.label + ': ' : '') + 'Next: ' + target.title;
        const b = el('button', 'pj-primary', label);
        b.type = 'button';
        b.addEventListener('click', () => {
          guided.history.push(edge.to);
          updateStatus();
          if (target.track_id !== byId.get(id).track_id) live('Entering track ' + tracks.get(target.track_id).title);
          renderCard(edge.to, { mode: 'guided' });
        });
        nav.append(b);
      });
    } else {
      const go = el('button', 'pj-primary', guided.history.length ? 'Resume the guided path' : 'Begin the guided path');
      go.type = 'button';
      go.addEventListener('click', () => (guided.history.length ? resume() : begin()));
      nav.append(go);
    }
  }

  function openCard(id, ctx) {
    if (!dialog.open && document.activeElement instanceof Element) lastOpener = document.activeElement;
    renderCard(id, ctx);
    openDialog();
  }
  function begin() {
    guided.history = [starts[0]];
    guided.completed = false;
    updateStatus();
    openCard(starts[0], { mode: 'guided' });
  }
  function resume() { openCard(guided.history[guided.history.length - 1], { mode: 'guided' }); }
  document.getElementById('pj-begin').addEventListener('click', begin);
  document.getElementById('pj-resume').addEventListener('click', resume);

  // ---- Page content.
  document.getElementById('pj-title').textContent = C.title;
  document.getElementById('pj-audience').textContent = 'For ' + C.audience.role + '. ' + C.audience.prior_knowledge + '.';
  document.getElementById('pj-outcome').textContent = 'After the guided path you ' + C.audience.outcome.charAt(0).toLowerCase() + C.audience.outcome.slice(1) + '.';

  const viewsRoot = document.getElementById('pj-views');
  DATA.views.forEach((view, index) => {
    const fig = el('figure', 'pj-view');
    fig.style.margin = '0 0 20px';
    const titleId = 'pj-view-title-' + index;
    const h = el('h2', null, view.title);
    h.id = titleId;
    fig.setAttribute('aria-labelledby', titleId);
    const cap = el('p', 'pj-caption', view.caption);
    const holder = el('div', 'pj-diagram');
    fig.append(h, cap, holder);
    viewsRoot.append(fig);
    const nodeLabels = new Map(view.workflow_nodes.map((n) => [n.id, n]));
    const laneLabels = new Map(view.lanes.map((l) => [l.id, l.label]));
    let svg;
    try { svg = sanitizeSvg(view.svg, 'pj-v' + index); }
    catch (err) { holder.append(el('p', 'pj-error', 'Diagram rejected by the sanitizer: ' + err.message + '. Use the step list below.')); return; }
    svg.setAttribute('aria-label', view.title);
    svg.removeAttribute('aria-labelledby');
    svg.setAttribute('role', 'group');
    svg.querySelectorAll('g[data-node-id]').forEach((g) => {
      const wid = g.getAttribute('data-node-id');
      const wnode = nodeLabels.get(wid);
      const target = view.detailed ? wid : (view.node_map || {})[wid];
      if (!wnode || !target || !byId.has(target)) { g.removeAttribute('tabindex'); g.removeAttribute('role'); return; }
      g.classList.add('pj-node');
      g.setAttribute('role', 'button');
      g.setAttribute('tabindex', '0');
      g.removeAttribute('aria-pressed');
      const lane = laneLabels.get(wnode.lane);
      g.setAttribute('aria-label', view.detailed
        ? accessibleName(target) + ', ' + lane
        : wnode.label + ', ' + lane + ' (opens the card ' + byId.get(target).title + ')');
      const activate = () => openCard(target, { mode: 'explore', origin: view.detailed ? null : view.title + ', node "' + wnode.label + '"' });
      g.addEventListener('click', activate);
      g.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); activate(); }
      });
    });
    holder.append(svg);
  });

  const list = document.getElementById('pj-list');
  C.tracks.forEach((t) => {
    const group = el('div');
    group.setAttribute('role', 'group');
    const hid = 'pj-track-' + t.id;
    const h = el('h3', null, t.title);
    h.id = hid;
    group.setAttribute('aria-labelledby', hid);
    const ol = el('ol');
    t.display_order.forEach((id) => {
      const li = el('li');
      const b = el('button', null, byId.get(id).title);
      b.type = 'button';
      b.addEventListener('click', () => openCard(id, { mode: 'explore', origin: 'the step list' }));
      li.append(b);
      ol.append(li);
    });
    group.append(h, ol);
    list.append(group);
  });
  updateStatus();
})();
</script>
</body>
</html>

-------

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

docs/orchestrator/persona-presets/franchise.md
----
# Persona presets: contract for franchises

**Audience:** franchise teams publishing persona presets.
**Status:** proposal, for review.

A persona preset is the twinCore orchestrator loaded with your instructions (`AGENTS.md`) and
your skills. You publish it in your own LangSmith workspace, in Context Hub. Users pick it in
the UI. The orchestrator never delegates to a persona: personas are standalone.

## 1. What you publish

Two kinds of Context Hub **agent** repos in your workspace.

### Persona repo (one per persona)

```text
<persona-repo>                 repo tag: twin-persona
├── AGENTS.md                  persona instructions (role, scope, tone)
└── skills/
    ├── <skill-name>/SKILL.md
    └── <skill-name>/SKILL.md
```

- Each `SKILL.md` starts with frontmatter carrying `name` and `description`.
- A skill's folder name equals its `name`.
- Skills are plain files inside the persona repo. Linked skills (`SkillEntry`) are not
  supported yet.
- No secrets, credentials or personal data in any file.

### Registry (one per workspace)

An agent repo named `config` (repo tag: `config`) holding `config.json`, the list of your
personas:

```json
{
  "personas": [
    {
      "name": "incident-manager",
      "repo": "incident-manager-persona",
      "description": "Helps run an incident from detection to postmortem.",
      "short_description": "Incident management",
      "visibility": {"ui": true, "orchestrator": false}
    }
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Stable key of the persona in your workspace (lowercase, digits, `-`). Do not rename it. |
| `repo` | yes | The persona repo's handle in your workspace. Conversations reference it: after a rename, open conversations on the old repo get the plain orchestrator with a notice, and users start a new conversation. Avoid renaming. |
| `description` | yes | What the persona does, shown in the UI. |
| `short_description` | no | Short label for lists. |
| `visibility.ui` | yes | `true` to show it in the UI. |
| `visibility.orchestrator` | yes | Always `false` for now. |

`config.json` carries no version: the version is the tag you set on the commits.

## 2. How you release

You release by tagging commits in the Context Hub UI. Three tags are read:

| Tag | How to set it | Where users can run it |
|---|---|---|
| `dev` | Add the tag `dev` to a commit. Without it, the latest commit is used (registry and persona alike). | UI environments `uat` and `qual` |
| `staging` | Promote to staging | UI environments `uat` and `qual` |
| `production` | Promote to production | UI environment `prod` |

- `uat` is your test environment. It runs your `dev` and `staging` personas side by side, on
  the stable (qual) orchestrator, so a test reflects your persona, not orchestrator changes.
- Tag **both** repos with the same tag: the persona repo first, then `config`. The registry
  at a tag is read with each persona at that same tag.
- A persona listed in `config` at `staging` or `production`, whose repo has no commit at that
  tag, is not shown to users at that tag. Check that every listed persona carries the tag
  before tagging `config`. At `dev`, a persona without the tag is shown at its latest commit.
  It is hidden only when its repo has no commit at all.
- A new tag reaches new conversations within about five minutes (cache). Conversations
  already open may pick up the new version at their next turn.

## 3. What you get

- Your persona appears in the UI for users whose roles allow it (see the [TwinShield contract](twinshield.md)).
- The orchestrator reads your repos read-only; it never writes to your workspace.
- If your persona cannot be loaded, users get the plain orchestrator with a visible notice.

## 4. Known limits

| Limit | Seen on | Consequence |
|---|---|---|
| Untagged commits are read only through the `dev` fallback | Design | No tag `dev`: `uat` runs your latest commit. |
| Pins on linked skills are ignored | LangSmith 0.18.3 | Linked skills are out of scope until this is resolved. |
| A skill folder holding several skills is not indexed | deepagents 0.7.19 | One folder per skill, directly under `skills/`. |

## 5. Open points

1. Where role-based access is declared: in `config.json` (for example a `roles` field per
   persona) or in TwinShield's rules.
2. Size limits for `AGENTS.md` and the number of skills per persona.
3. `config.yaml` instead of `config.json`, if TwinShield agrees to parse YAML.
4. A command-line helper to validate, push and tag a persona and its registry.

-------

docs/orchestrator/persona-presets/twinshield.md
----
# Persona presets: contract for TwinShield

**Audience:** TwinShield team.
**Status:** proposal, for review.

Personas are orchestrator presets published by franchises in Context Hub (see the [franchise contract](franchise.md)). TwinShield tells the UI and the orchestrator which personas a user may use, in the
same discovery call that returns the authorized assistants.

## 1. What TwinShield reads

For each workspace the user is authorized on:

1. Take the Context Hub tags listed for the request's environment:

   | Environment header | Tags |
   |---|---|
   | `dev` | `dev` |
   | `uat` | `dev`, `staging` |
   | `qual` | `dev`, `staging` |
   | `prod` | `production` |

   Tag `dev` falls back to the repo's latest commit when the repo has no `dev` tag, for the
   `config` repo and for each persona repo alike, wherever `dev` is listed. `staging` and
   `production` have no fallback. Apply the fallback first; the 404 and drop rules below apply
   only when no commit can be read at all.

2. For each tag, pull the registry: agent repo `config`, file `config.json`.
   - Python: `Client(workspace_id=<workspace>).pull_agent("-/config", version="<tag>")`.
   - HTTP: `GET /platform/hub/repos/-/config/directories?repo_type=agent&commit=<tag>`
     with the `X-Tenant-Id` header set to the workspace.
   - For `dev`, a 404 on the tag means: pull the latest commit instead.
   - A 404 that remains (no `config` repo, or `staging` / `production` not set) means the
     workspace has no registry at that tag: no personas from it at that tag.
3. Keep the personas the user's roles allow (rule to agree, see open points).
4. Drop a persona whose repo has no commit at the tag (for `dev`: whose repo has no commit at
   all), and log a WARNING naming the
   workspace, persona and tag (`GET /repos/<owner>/<repo>/tags`, one call per persona,
   cacheable). The UI never lists a persona that cannot load.

## 2. What TwinShield returns

A new field next to `assistants`, in the same response. One entry per persona **and tag**:
in `uat` and `qual` the same persona can appear twice, at `dev` and at `staging`.

```json
{
  "assistants": ["... unchanged ..."],
  "authorized_personas": [
    {
      "workspace_id": "<WORKSPACE_ID>",
      "name": "incident-manager",
      "repo": "incident-manager-persona",
      "version": "staging",
      "description": "Helps run an incident from detection to postmortem.",
      "short_description": "Incident management"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `workspace_id` | Workspace holding the persona. |
| `name` | Registry key of the persona (from `config.json`). |
| `repo` | Persona repo (from `config.json`). The backend copies it into its call, and the orchestrator loads it as sent. A later version checks it against this record. |
| `version` | The tag the persona was resolved at: `dev`, `staging` or `production`. Never a commit hash. |
| `description`, `short_description` | Copied from `config.json`, for the UI catalog. |

Rules:

- **Never put personas inside `assistants`.** An entry there means a subagent the main
  orchestrator may delegate to; personas are standalone and must not be delegated to.
- Only personas with `visibility.ui = true` are returned.
- Same cache lifetime and failure behaviour as the assistants (to confirm).

## 3. How the orchestrator uses it

In this version the orchestrator does not read `authorized_personas`: the backend uses it
to list personas and sends the chosen `{workspace_id, repo, version}`, which the orchestrator
loads as sent. A later version checks each run's persona against the user's
`authorized_personas`, from the discovery result the orchestrator already fetches, with no
additional call to TwinShield.

## 4. Known limits

| Limit | Seen on | Consequence |
|---|---|---|
| Repo listings carry no tags | LangSmith 0.18.3 | Read `config` at the tag; use the tags endpoint per repo. |
| `latest` is not a tag (404) | LangSmith 0.18.3 | Read the latest commit by pulling without a version. |

## 5. Open points

1. Where role-based access per persona is declared: a field in `config.json`, or TwinShield's
   own rules.
2. Read access: a key that can read the `config` and persona repos of every franchise
   workspace.

-------

docs/orchestrator/persona-presets/backend.md
----
# Persona presets: contract for the backend (BFF)

**Audience:** backend team calling the orchestrator.
**Status:** proposal, for review.

The backend lists the personas a user may use and calls the orchestrator with the chosen one.
It sends only the persona's identity: the orchestrator loads the persona itself.

## 1. Listing personas

Show the entries of `authorized_personas` returned by TwinShield (see the [TwinShield contract](twinshield.md)), using `description` and `short_description`. In `uat` and `qual` a persona can be listed twice,
at `dev` and at `staging`: show the tag so the user knows which one they test. The backend does
not read Context Hub.

## 2. Which orchestrator to call

| UI environment | Orchestrator deployment for a persona run |
|---|---|
| `dev` | `-dev` |
| `uat` | `-qual` (both `dev` and `staging` personas) |
| `qual` | `-qual` (both `dev` and `staging` personas) |
| `prod` | prod |

`-qual` is the stable orchestrator franchises test on: their persona changes, ours do not.

## 3. Calling the orchestrator

Send the persona in the run's `context`:

```http
POST /threads/<THREAD_ID>/runs
```

```json
{
  "assistant_id": "<ORCHESTRATOR_ASSISTANT_ID>",
  "input": {"messages": [{"role": "user", "content": "<question>"}]},
  "context": {
    "persona": {
      "workspace_id": "<WORKSPACE_ID>",
      "repo": "incident-manager-persona",
      "version": "staging"
    }
  }
}
```

| Field | Value |
|---|---|
| `workspace_id` | From the chosen `authorized_personas` entry. |
| `repo` | From the chosen entry: the persona repo the orchestrator loads. |
| `version` | From the chosen entry: `dev`, `staging` or `production`. It selects which tag the orchestrator loads. |

Rules:

- Copy the three values as TwinShield returned them; do not build or change them. The
  orchestrator loads `workspace_id/repo@version` as sent: the backend is trusted to send only
  a persona TwinShield listed for this user.
- Send the same persona, at the same `version`, on every run of a thread. To switch persona or
  tag, start a new thread.
- `context` is preferred; `config.configurable.persona` is also accepted, and `context` wins
  when both reach the orchestrator. The LangGraph server itself rejects a single run request
  carrying both `context` and `configurable` (HTTP 400): send one of them.
- Do not send files or a commit: the orchestrator resolves them.
- No persona: omit the field; the user gets the plain orchestrator.

## 4. What the orchestrator does

| Situation | Result |
|---|---|
| Valid persona | The run uses the persona's instructions and skills, at `version`. |
| Different persona or `version` on an existing thread | Turn refused; start a new thread. |
| Persona cannot be loaded (Context Hub unavailable, tag removed) | Plain orchestrator with a visible notice; retried on the next turn. |

The orchestrator does not refuse a `version` because of its own environment, and in this
version it does not check the persona against TwinShield: the backend's choice is trusted.
A later version adds that check (the persona must be in the user's `authorized_personas`)
without changing this request.

## 5. Known limits

| Limit | Seen on | Consequence |
|---|---|---|
| A run carrying both `context` and `configurable` | langgraph-api 0.9.0 | HTTP 400 from the server. |
| A persona re-tagged while a thread is open | Design | The thread may use the new commit from its next turn. |
| A persona repo renamed while a thread is open | Design | The old repo no longer loads: plain orchestrator with a notice; start a new thread. |

## 6. Open points

1. How a refused turn is reported to the UI (message text, error code).
2. Whether the UI offers "start a new conversation" when a persona was updated.

-------

