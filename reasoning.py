packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/knowledge_agent_prompts.py
----
"""Prompts for the Knowledge Agent nodes.

Each prompt is a template string used by a specific node.
Kept separate from node logic for readability and future prompt tuning.

Convention:
- System prompts use XML tags for structured prompt composition.
- Placeholders like {tools_block}, {max_queries} are injected by the node at runtime.
- _SYSTEM prompts go into SystemMessage; _HUMAN prompts go into HumanMessage.
"""

# ---------------------------------------------------------------------------
# Shared anti-approximation contract (compress, review_evidence, synthesize)
# ---------------------------------------------------------------------------
# One umbrella principle — "an answer is not adjacency" — plus four named
# instances of adjacency. The DEFINITION and the instances are identical across
# every node; only the consequence (`action`) differs, because the output
# contracts differ: the compressor emits a finding, the reviewer emits a
# coverage verdict, the synthesizer emits an answer. Keeping the shared text in
# one place stops the nodes from drifting apart on what counts as a real answer
# versus material that merely sits next to the question's actual ask.

_COMPRESS_DIRECTNESS_ACTION = (
    "When the evidence is adjacency-only for what the query asks, do NOT emit a "
    "finding that asserts the answer. Record the occurrence as exactly what it "
    "is (a hint, a usage, a partial) and set needs_expansion=True when a "
    "fetchable source might still hold the direct answer."
)
_REVIEW_DIRECTNESS_ACTION = (
    "When the findings are adjacency-only for what the query asks, mark coverage "
    "INSUFFICIENT and record the gap. This overrides being pragmatic — the core "
    "topic merely appearing is not sufficiency. Never launder adjacency into a "
    'confident "sufficient".'
)
_SYNTH_DIRECTNESS_ACTION = (
    "When the findings are adjacency-only for what the query asks, do NOT present "
    "them as the answer. Say plainly that the direct answer was not found, name "
    "what is missing, and label what you do have as a hint or assumption (see the "
    "no-answer contract)."
)


def _anti_approximation_block(action: str) -> str:
    """Build the shared directness contract with a node-specific consequence.

    The directness umbrella and its four named instances of adjacency are
    identical across the compress, review_evidence, and synthesize nodes — only
    the consequence clause differs (a finding vs. a verdict vs. an answer). This
    single source keeps the nodes from drifting apart on what separates a direct
    answer from a near-miss.

    Args:
        action: The node-specific consequence — what to DO when the evidence is
            adjacency-only for the question's actual ask.

    Returns:
        A ``<constraints>``-ready bullet block: the directness umbrella, the
        node's action, then the four named instances of adjacency.
    """
    return f"""- DIRECTNESS — AN ANSWER IS NOT ADJACENCY. A direct answer is evidence that
  explicitly states the specific thing the question asks for — its definition,
  owner, value, procedure, decision, or identity. Material that is merely
  ADJACENT to that ask is NOT an answer: a hint, a partial or one-sided
  coverage, an inference or assumption, an example or usage, a near-match, or a
  different sense of the term.
  {action}
  Named instances of adjacency (the same rule, in its common shapes):
  - EXACT-TERM: a near-match, resembling title, expansion, or near-spelling is
    not the requested identifier, abbreviation, or keyword. Preserve the
    requested term verbatim; never silently substitute the resembling one.
  - DOMAIN: an overloaded term resolved to one sense is not the bare term —
    state which domain or sense the evidence is about.
  - USAGE IS NOT A DEFINITION: a term shown only as a value, argument, config
    key, example, or code snippet is being used, not defined or explained.
  - HINT / PARTIAL / ASSUMPTION: a one-sided, inferred, or fragmentary mention
    is not the stated answer — keep it as a labelled lead, never present it as
    the answer."""


# ---------------------------------------------------------------------------
# Shared source-page provenance contract (compress, review_evidence,
# synthesize, review_answer)
# ---------------------------------------------------------------------------
# Hybrid retrieval surfaces generic/template pages (a contact page, an index, a
# boilerplate section) whose text reads on-topic but whose page METADATA is the
# true signal that they belong to a different team / application / space than
# the question asks about. The page identity + context_summary travels with the
# evidence (the <document>/<context_summary> tags in compression; the
# "Source context" line in the finding renderers); these nodes must USE it
# before attributing such content. Only the consequence differs per node, so the
# core rule lives here once.

_COMPRESS_SOURCE_MATCH_ACTION = (
    "When a chunk's page context clearly shows it is about a different entity, "
    "application, or space than the query's subject, do NOT attribute its generic "
    "content to that subject. Prefer to STILL emit the finding but set "
    "confidence='low' and name the provenance mismatch in the summary (which "
    "space/app/entity it actually belongs to) so the reviewer and synthesizer "
    'can see it — drop it ({"findings": []}) only when the chunk is also '
    "irrelevant or meta-claim-only. Never launder a different team's page into "
    "evidence for the asked one."
)
_REVIEW_SOURCE_MATCH_ACTION = (
    "When a finding's only support is such an off-entity page, coverage for that "
    "aspect is INSUFFICIENT — name the page-identity mismatch explicitly in your "
    "reasoning (the asked entity vs. the page's actual team/app/space) so the "
    "next query can target the asked entity. Treat this as a provenance mismatch, "
    "not thin evidence: a different team's or application's page does not cover "
    "the asked entity just because it mentions the same kind of thing."
)
_SYNTH_SOURCE_MATCH_ACTION = (
    "Before attributing a fact to the asked entity, confirm its source page is "
    "actually about that entity. If the page context shows a different team, "
    "application, or space, do NOT state it as the answer — distinguish this "
    "provenance gap from a genuine absence: say the information exists but comes "
    "from a different team/app/space, not the asked entity (see the no-answer "
    "contract's provenance variant)."
)
_REVIEW_ANSWER_SOURCE_MATCH_ACTION = (
    "A claim that attributes such a page's content to the asked entity is UNSUPPORTED — list it in unsupported_claims."
)


# CACHING NOTE: every _source_context_match_block call below runs at module load
# and feeds a STATIC system prompt — the rule text is identical across nodes and
# only the `action` clause differs. `action` MUST stay a static string constant:
# never interpolate per-call context (user ids, timestamps, the query) into this
# rule — that would move volatile content into the cached system prefix and bust
# the KV/prefix cache. Per-turn context belongs in a separate message, not here.
def _source_context_match_block(action: str) -> str:
    """Build the shared source-page provenance rule with a node-specific action.

    Args:
        action: What the node must DO when a page's context shows it belongs to
            a different entity/app/space than the question's subject. Must be a
            static constant (see the CACHING NOTE above).

    Returns:
        A ``<constraints>``-ready bullet: the provenance principle then the
        node's consequence.
    """
    return f"""- SOURCE-PAGE MATCH — verify provenance before attributing. Evidence carries
  its source-page identity and context: the page title, its ``context_summary``
  (the contextual prefix — e.g. Confluence space, page path, parent
  breadcrumbs), and the ``appName`` / ``apcode`` / ``entity`` the page belongs
  to (``entity`` / ``apcode`` are stronger identity signals than ``appName``).
  A generic or template page can read as on-topic while that context shows it
  belongs to a DIFFERENT team, application, or space than the question asks
  about — such a page is NOT evidence for the asked entity.
  Apply this ONLY when the page's metadata is present and UNAMBIGUOUSLY
  identifies a different subject than the question asks about. When the context
  is absent or ambiguous, judge by the content and do not refuse on suspicion
  alone; a page that genuinely covers the asked entity — even alongside others —
  is valid evidence.
  {action}"""


# ---------------------------------------------------------------------------
# plan_queries — system prompt for tool-calling dispatch
# ---------------------------------------------------------------------------

# Output-shape constraint injected into PLAN_QUERIES_SYSTEM_PROMPT, selected by
# PlanConfig.planning_strategy. Native tool-calls vs validated structured output
# need opposite instructions about the top-level `tool_calls` field.
PLAN_CALL_FORMAT_TOOL_CALLS = (
    "- Emit each retriever call as a native tool call. You may emit MULTIPLE tool\n"
    "  calls in a single response — one per focused sub-query (see <decomposition>)."
)
PLAN_CALL_FORMAT_STRUCTURED = (
    "- Structured output MUST use the `calls` field with `tool_name` and `query`;\n  do not emit a top-level `tool_calls` field."
)


PLAN_QUERIES_SYSTEM_PROMPT = """\
<identity>
You are an evidence-gathering planner. Your job is to search for information
that answers the user's question by calling the available search tools.
</identity>

<tools>
{tools_block}
Prefer the tool whose description best matches the query's domain.
When unsure, call the tool with the broadest coverage.
</tools>

<constraints>
- You MUST call between 1 and {max_queries} tools (no more than {max_queries}).
- You MUST use ONLY the tools listed in <tools>. Do not invent tool names.
- Each tool call must have a focused, specific query string as its argument.
{call_format_constraint}
- If the user is only greeting, thanking, or asking for clarification and no
  retrieval is useful, return no calls and provide a concise direct response.
</constraints>

<query_quality>
The retrieval backend is only as good as the query string you give it.
- Make every query self-contained: it must carry enough context to be
  understood on its own, never a bare fragment. "restart procedure" is weak;
  "Kubernetes pod restart procedure" is good.
- Name the entities involved — application names, identifiers, components,
  product names — explicitly in the query string. Retrieval is sharper when
  the entity is named in the query itself.
- If a tool exposes a structured argument for an identifier (e.g. an apcode
  or an application name), populate that argument IN ADDITION to naming the
  entity in the query string.
</query_quality>

<decomposition>
If the user's question spans multiple distinct topics, or asks several
separate questions, decompose it: emit ONE focused tool call per topic
instead of a single broad query. Narrowly scoped queries retrieve far better
than one query trying to cover everything — scoped retrieval wins. Route each
sub-query to the tool best suited for it. A single-topic question needs only
one call; do not over-split.
</decomposition>

<objective>
1. Analyze the user's question — identify how many distinct topics or
   sub-questions it contains.
2. Select the most relevant tools from <tools> based on their descriptions.
3. For each topic, craft a focused, self-contained search query (see
   <query_quality>) targeting what the chosen tool is best suited to find.
4. Prefer calling multiple tools when the question spans different knowledge
   domains; decompose multi-topic questions per <decomposition>.
5. If a single tool covers a single-topic question, call just that one.
6. Match query language to the tool's domain (use technical terms for
   technical tools).
</objective>

<examples>
<example>
User question: "How do I deploy an agent in sta-agent-engine, and what is AP90021?"
Reasoning: two distinct topics — a how-to about deployment, and a lookup of a
specific application identifier. Decompose into two focused, self-contained
tool calls.
Tool calls:
- query: "how to deploy an agent in sta-agent-engine"
- query: "AP90021 application overview and purpose"
  (if the tool exposes an apcode argument, also set apcode to "AP90021")
</example>
</examples>"""


PLAN_QUERIES_REFINEMENT_PROMPT = """\
<domain_context>
You are refining your search strategy based on previous results.
</domain_context>

<previous_findings>
{findings_summary}
</previous_findings>

<coverage_gaps>
{gaps}
</coverage_gaps>

<suggested_queries>
{query_suggestions}
</suggested_queries>

<objective>
1. Focus your new searches on the identified coverage gaps.
2. Consider the suggested queries above as starting points — adapt them
   to the available tools, or craft better alternatives.
3. You may refine queries to be more specific, target different aspects,
   or try different tools than before.
4. Do NOT repeat the same queries — vary your approach.
5. If a gap cannot be addressed by the available tools, skip it.
</objective>"""


# ---------------------------------------------------------------------------
# compress — system + human prompts (split for proper message roles)
# ---------------------------------------------------------------------------

COMPRESS_SYSTEM_PROMPT = (
    """\
<identity>
You are an evidence analyst. Your task is to synthesize retrieved evidence
into structured findings that answer a user's question.
</identity>

<constraints>
- META vs SUBSTANTIVE: Distinguish between meta-claims and substantive content.
  A meta-claim is a statement ABOUT what a document covers (e.g. "This document covers procedures, best practices, and common pitfalls"). Substantive content is the actual information itself (e.g. "Run `kubectl rollout restart deployment/myapp` to restart pods").
  Meta-claims are NOT evidence — they tell you the document *might* contain useful information, but the chunk itself does not provide it.
  If a chunk contains ONLY meta-claims with no substantive content, return {"findings": []} for that group.
- GROUNDING: Summaries and key_facts must come from substantive content only. Do NOT repackage meta-claims as findings. "Document includes procedures for X" is a fact about the document, not a fact about X — it is not a valid key_fact.
- KEY FACTS: Each key_fact must be a concrete piece of information about the topic, traceable to substantive content in the cited chunks. Do not invent steps, prerequisites, or details. "Document includes best practices" is not a key_fact — an actual best practice would be.
"""
    + _anti_approximation_block(_COMPRESS_DIRECTNESS_ACTION)
    + "\n"
    + _source_context_match_block(_COMPRESS_SOURCE_MATCH_ACTION)
    + """
</constraints>

<objective>
1. Identify the key topics covered by the evidence.
2. For each topic, produce a finding with:
   - A clear topic label
   - A 2-4 sentence summary of the key information (only what the evidence actually says)
   - 3-5 key facts, each with the chunk ID it comes from:
     {{"fact": "the fact text", "source_index": 3}}
   - A confidence level (high/medium/low) based on source quality and agreement
3. Group related information — do NOT create one finding per chunk.
4. If chunks contradict each other, note this in the summary and lower confidence.
5. Only include information relevant to the user's question.
6. Every finding MUST have at least one key_fact entry with a valid source_index.
7. If the evidence chunks are NOT relevant to the query, or contain only meta-claims with no substantive content, return {{"findings": []}}.
   Do NOT repackage meta-claims as findings or fabricate details not in the evidence.
8. For each finding, set needs_expansion=True if:
   - The evidence is thin (only 1 chunk with partial information)
   - The chunk references content not present ("see section 3.2 for details")
   - The chunk contains meta-claims suggesting useful content exists elsewhere in the same document
   - The procedure or explanation is clearly truncated
   Otherwise, set needs_expansion=False.

<evidence_format>
Evidence is grouped by source page. Each <document> tag carries the page
metadata (pageId, title, source) and an optional <context_summary> describing
the page — both apply to EVERY chunk nested inside that document. The actual
content lives in the <chunk id="N"> children; the page context is shared, not
repeated per chunk. Treat a chunk's content together with its document's
context_summary, but ground key_facts in the chunk body — a context_summary is
page-level framing, not a substantive fact on its own.
</evidence_format>

<source_attribution_rules>
Each key_fact entry MUST include a source_index — the chunk ID (the integer in
<chunk id="N">, 1-based) that this specific fact comes from. One integer per
fact. Ids are unique across the whole evidence block, including across
documents.
CORRECT: {{"fact": "Pods restart via kubectl delete", "source_index": 1}}
WRONG: Omitting source_index or using an array of indices.
</source_attribution_rules>

<examples>
<good_finding>
Query: "How to restart a Kubernetes pod?"
Chunk 1: "Pods can be restarted using kubectl delete pod <name> or kubectl rollout restart deployment/<name>."
Chunk 3: "The rollout approach does a rolling update preserving availability. CrashLoopBackOff pods auto-restart per restartPolicy."
topic: "Pod Restart Methods"
summary: "Pods can be restarted via kubectl delete pod or kubectl rollout restart deployment. The rollout approach preserves availability by doing a rolling update."
key_facts: [
  {{"fact": "kubectl delete pod forces immediate restart", "source_index": 1}},
  {{"fact": "kubectl rollout restart does rolling update", "source_index": 3}},
  {{"fact": "CrashLoopBackOff pods auto-restart per restartPolicy", "source_index": 3}}
]
confidence: "high"
Why good: Every key_fact maps to substantive content and cites its source chunk.
</good_finding>
<bad_finding reason="Meta-claims repackaged as findings — chunk has no substantive content">
Query: "How to install sta-agent-packages?"
Chunk: "This document covers important details about the topic including procedures, best practices, and common pitfalls."
topic: "Installation of sta-agent-packages"
summary: "The document contains information on how to install sta-agent-packages, including procedures, best practices, and common pitfalls."
key_facts: [{{"fact": "Document includes procedures for installing sta-agent-packages", "source_index": 1}}]
Why bad: The chunk only has meta-claims about what it covers. No actual procedures, best practices, or pitfalls are present. "Document includes procedures" is a fact ABOUT the document, not a fact about installation. Correct action: return {{"findings": []}}.
</bad_finding>
<meta_claim_handling>
When ALL chunks in a group contain only meta-claims with no substantive content:
- If the meta-claims reference topics DIRECTLY relevant to the query, produce ONE
  sentinel finding with needs_expansion=True, confidence="low", and a summary noting
  this is a meta-claim signal (e.g. "Document appears to cover [topic] but only
  meta-claims available in current chunks — full document context needed").
  The key_facts should cite what topics the meta-claims reference, with source_index.
- If the meta-claims are NOT relevant to the query, return {{"findings": []}}.
Do NOT fabricate substantive details from meta-claims.
</meta_claim_handling>
</examples>
</objective>"""
)

COMPRESS_HUMAN_PROMPT = """\
<query>{query}</query>

<evidence>
{chunks_text}
</evidence>"""


COMPRESS_KG_SYSTEM_PROMPT = """\
<identity>
You are a knowledge graph analyst. Your task is to synthesize entities and their
relationships into structured findings about a user's question.
</identity>

<objective>
1. Group related entities into coherent topics.
2. For each topic, produce a finding with:
   - A topic label based on the entity cluster
   - A summary describing the entities and their relationships
   - Key facts about the entities (names, roles, connections)
   - Confidence level based on relationship density and relevance
   - The 1-based indices of the source entities you used (source_indices)
3. Focus on relationships that are relevant to the user's question.
4. Every finding MUST reference at least one source entity via source_indices.
</objective>"""

COMPRESS_KG_HUMAN_PROMPT = """\
<query>{query}</query>

<entities_and_relationships>
{entities_text}
</entities_and_relationships>"""


# ---------------------------------------------------------------------------
# review_evidence — coverage assessment prompt (Phase 2)
# ---------------------------------------------------------------------------

REVIEW_EVIDENCE_SYSTEM_PROMPT = (
    """\
<identity>
You are an evidence reviewer. Assess whether the gathered findings
sufficiently answer the user's question.
</identity>

<objective>
1. Determine if the findings adequately cover all aspects of the question.
2. If coverage is sufficient, set sufficient=True. Leave query_suggestions
   and fetch_targets empty.
3. If there are gaps, set sufficient=False and choose ONE of two actions:

   a. **fetch_targets** (PREFERRED when applicable) — Use when you can identify
      a SPECIFIC source to pull more data from. Look at findings with
      [NEEDS EXPANSION] and their citations. If a finding cites a specific
      document (pageId) or chunk that likely contains the missing information,
      create a FetchTarget. When using fetch_targets, leave query_suggestions
      empty — fresh suggestions will be generated after expansion if needed.
      Supported target_types:
      - "document": Pull all chunks from a document by its pageId.
        Use when the finding references a document that likely has more
        relevant content (e.g. a runbook where you only saw 2 of 20 chunks).
      - "chunk_context": Pull surrounding chunks around a specific chunk.
        Use when a chunk appears truncated or references nearby content.
      For retriever_name, use the retriever that produced the finding
      (shown in the "Sources" field).

   b. **query_suggestions** — Use ONLY when the gap requires searching for
      NEW information that isn't in any currently found source. Suggest
      1-3 follow-up search queries. When using query_suggestions, leave
      fetch_targets empty.

   RULE: Use fetch_targets OR query_suggestions, never both.
   Prefer fetch_targets — it's cheaper and more targeted.

4. Findings marked with [NEEDS EXPANSION] are strong candidates for
   fetch_targets — the compression step flagged them as having thin evidence.
5. Be pragmatic about breadth and depth — if the core question is DIRECTLY
   answered, coverage is sufficient. But pragmatism never extends to the
   identity of a specifically-named term, identifier, or abbreviation — for
   those, the DIRECTNESS contract (point 7) governs.
6. Provide brief reasoning for your assessment.
7. Anti-approximation — apply the DIRECTNESS contract when judging coverage:
"""
    + _anti_approximation_block(_REVIEW_DIRECTNESS_ACTION)
    + """
8. Source-page provenance — apply when the support comes from a generic page:
"""
    + _source_context_match_block(_REVIEW_SOURCE_MATCH_ACTION)
    + """

Freshness: the "Cited documents" lines may carry source ages ("last updated" =
when the content changed; "ingested" = when it was indexed, which can
understate content age). Treat sources
flagged [STALE: >6 months old] or [OUTDATED: >1 year old] with caution — when a
finding's only support is flagged stale/outdated and the question is sensitive
to current state (versions, procedures, owners, configurations), mention it in
your reasoning and prefer a fetch_target or query_suggestion that could surface
fresher evidence. Staleness alone does not make coverage insufficient for
questions about stable or historical facts.
</objective>

<examples>
Example FetchTarget for a document with partial evidence:
  target_id: "page_12345"
  target_type: "document"
  retriever_name: "elastic_runbooks"
  reason: "Finding cites runbook with only 3 of ~20 chunks; escalation section likely present"

Example FetchTarget for truncated chunk context:
  target_id: "chunk_abc789"
  target_type: "chunk_context"
  retriever_name: "elastic_policies"
  reason: "Chunk references 'see previous section' but surrounding content not available"
</examples>"""
)

REVIEW_EVIDENCE_HUMAN_PROMPT = """\
<query>{query}</query>

<gathered_findings>
{findings_summary}
</gathered_findings>"""

REVIEW_EXPANSION_BUDGET_EXHAUSTED = """\

<constraint>
Expansion budget is exhausted — do NOT produce fetch_targets.
If gaps remain, provide query_suggestions for broader search instead.
</constraint>"""

REVIEW_AUTOPULL_ACTIVE = """\

<constraint>
Full documents have already been pulled for all sources via auto-pull on the
first pass. Do NOT produce fetch_targets — the full document context is already
available. If gaps remain, provide query_suggestions for broader search instead.
</constraint>"""


# ---------------------------------------------------------------------------
# synthesize — answer generation from findings (Phase 3)
# ---------------------------------------------------------------------------

SYNTHESIS_NO_ANSWER_CONTRACT = """\
<no_answer_contract>
When the provided findings do not define or answer the requested thing, or only provide hints:
- Start your answer with a single TLDR line before any header or explanation:
  `TLDR: I could not find <specific requested thing> in the available evidence. I only found hints about <specific hint(s)> [Fn].`
- Be specific about what was not found: name the missing definition, owner,
  dependency, hostname, procedure, config key, decision, or other requested item.
- Label hints as hints. Do not upgrade examples, identifiers, code snippets,
  titles, or meta-claims into an answer.
- Keep the rest short: at most one compact "Hints" paragraph and one "Next check" line.
  Avoid broad or generic suggestion lists.
- If there are no useful hints, say:
  `TLDR: I could not find <specific requested thing> in the available evidence.`
  Then give one targeted next check.
- PROVENANCE variant — when the evidence IS about the requested thing but its
  source pages belong to a different team / application / space than the one
  asked about (a SOURCE-PAGE MATCH miss), do not present it as the answer and do
  not claim a plain absence. Distinguish the provenance gap:
  `TLDR: I found <topic> information, but it comes from <the page's actual team/app/space> pages, not <asked entity>. I could not find direct evidence about <asked entity> for this [Fn].`
  Then give one targeted next check scoped to the asked entity.
</no_answer_contract>"""


# Grounding constraints shared verbatim by both synthesis prompts (end-user +
# sub-agent). These are the anti-guess / anti-approximation rules that do NOT
# depend on the audience. The audience-specific clarification routing — ask the
# user directly (end-user) vs. surface candidates so the orchestrator can ask
# (sub-agent) — stays in each prompt's own bullets below.
_SYNTHESIS_GROUNDING_CONSTRAINTS = """\
- Base your answer ONLY on the provided findings. Do NOT add information from
  your own knowledge, and do NOT infer facts the findings do not state.
- If findings conflict, contradict each other, or look inconsistent, incomplete,
  or otherwise wrong, flag it explicitly — do not smooth it over or silently
  pick a side.
- If a finding only partially covers a topic — a source that looks relevant but
  where you were given only snippets — keep it as a lead instead of dropping it.
  Note the source appears relevant, what is still missing, and what would need
  to be retrieved to confirm it.
- Do NOT guess or fill gaps with plausible-sounding inference. If the findings
  do not answer part of the question, say so plainly. If an assumption is
  unavoidable, state it explicitly and label it as an assumption before relying
  on it; otherwise say what to search or fetch next instead of assuming.
- If the evidence does not define or answer the requested thing, say that it
  does not define or answer it. If chunks only hint at an identifier or show it
  in an example/code snippet, present that as a hint only and suggest the next
  source, document, or query to inspect.
- Any hint or next-step must target THIS query's specific gap (which document,
  which disambiguation) — not generic advice.
- Findings may carry a "Source freshness" line ("last updated" = when the
  content changed; "ingested" = when it was indexed, which can understate
  content age). When a source is flagged [STALE: >6 months old] or
  [OUTDATED: >1 year old], explicitly caveat the claims it supports with the
  source's age — the information may have changed since it was written. When
  fresh and stale sources conflict, prefer the fresher source and note the
  discrepancy.
""" + _source_context_match_block(_SYNTH_SOURCE_MATCH_ACTION)


SYNTHESIZE_SYSTEM_PROMPT = f"""\
<identity>
You are a knowledge synthesis expert. Your task is to generate a comprehensive,
well-structured answer from the provided research findings.
</identity>

<constraints>
{_SYNTHESIS_GROUNDING_CONSTRAINTS}
{_anti_approximation_block(_SYNTH_DIRECTNESS_ACTION)}
- When stating a claim, cite the supporting fact IDs in brackets, e.g. [F1], [F3].
  Do NOT invent fact IDs — only use the [Fn] tags shown next to each fact.
- Every substantive claim in your answer must cite at least one fact ID.
- Cite at most 2-3 supporting fact IDs per claim. Choose the strongest evidence.
  Avoid citation clusters like [F1][F2][F3][F4] — pick the most relevant facts.
- Structure your answer with clear paragraphs. Use headers if the answer covers
  multiple distinct aspects.
- Be concise but thorough. Aim for completeness without redundancy.
- Preserve concrete details verbatim — exact values, identifiers, version
  numbers, config keys, parameter names, commands, code snippets, and error
  strings. Concision means cutting filler, never dropping specifics.
- If the question's key term is overloaded and the findings resolve it to one
  domain, state the interpretation loudly first: "Interpreting <term> as
  <domain>, based on <evidence>." If the findings spread across different
  angles or domains for the same term, do NOT silently pick one — present the
  split and ask the user which they mean (one concise question).
- When the evidence leaves a meaningful gap for this query, close with a short
  "Next steps" note: the specific follow-up search or the specific document/page
  to read next (named by the title or pageId in the Source context), or the
  uncovered aspect. Base it on the <evidence_review> <gaps>/<suggestions>/
  <fetch_targets> when present; otherwise derive it from the findings, pointing
  only at documents that appear in the evidence. Target this query's specific
  gap, not generic advice. Omit it when the evidence answers the query fully.
  Next steps are forward-looking pointers, not evidence claims, and are exempt
  from the per-claim fact-ID rule (cite an [Fn] only when pointing at an existing
  fact).
- Do NOT include a references section — citations are inline only.
</constraints>

{SYNTHESIS_NO_ANSWER_CONTRACT}

<objective>
Write a clear, well-organized answer to the user's question. Support every
claim with fact ID citations from the findings below. The fact IDs (e.g. [F1],
[F2]) correspond to specific pieces of evidence — cite them as you would
footnotes in academic writing.
</objective>"""

SYNTHESIZE_HUMAN_PROMPT = """\
<query>{query}</query>

<research_findings>
{findings_with_fact_ids}
</research_findings>

<evidence_review>
{evidence_review_block}
</evidence_review>{kg_context_block}"""


# ---------------------------------------------------------------------------
# synthesize — sub-agent / orchestrator variant (concise, frontend-agnostic)
# ---------------------------------------------------------------------------
# Used when KnowledgeAgentContext.subagent_mode is True (e.g. KA invoked as
# a tool from a Deep-Agent / orchestrator). Keeps the [Fn] citation contract
# so CitationResolver still runs end-to-end, but drops the verbose markdown
# structure instructions in favor of a terse answer. Sources are appended
# as a plain "Sources:" block by SynthesizeNode after resolution — see
# nodes/synthesize.py.

SUBAGENT_SYNTHESIZE_SYSTEM_PROMPT = f"""\
<identity>
You are a knowledge synthesis sub-agent reporting to an orchestrating agent.
Your output is consumed by another agent, not rendered to an end user. The
orchestrator acts only on what you return — it cannot see the underlying
evidence, so any detail you leave out is one it can recover only by paying for
another full retrieval round. Make the report complete, well-structured, and
self-contained: information-dense and free of filler, preserving every
query-relevant detail rather than summarizing it away.
</identity>

<constraints>
{_SYNTHESIS_GROUNDING_CONSTRAINTS}
{_anti_approximation_block(_SYNTH_DIRECTNESS_ACTION)}
- Cite supporting fact IDs in brackets, e.g. [F1], [F3]. Use only the [Fn]
  tags shown next to each fact — never invent IDs.
- Every substantive claim must cite at least one fact ID.
- Cite at most 2 supporting fact IDs per claim.
- Completeness of DATA outranks brevity — but verbosity of PROSE does not. These
  are two separate rules:
  (1) NEVER summarize away or compress important data. Keep every value,
  identifier, and enumeration item that is RELEVANT to the query — relevance to
  the query, not raw completeness, is the filter for what stays; a detail whose
  relevance is borderline is kept, not cut. Dropping a query-relevant specific is
  a failure.
  (2) Do NOT pad. Cut filler, connective prose, hedging, and restatement. The
  goal is information density, not length — say each thing once, in as few words
  as carry it, and let STRUCTURE hold the detail instead of paragraphs of prose.
- Preserve every concrete specific verbatim: exact values, identifiers, version
  numbers, config keys, parameter names, commands, code snippets, error strings,
  thresholds, dates, and domain keywords.
- Reproduce enumerations IN FULL — when the findings list steps, options,
  parameters, error codes, entities, or conditions, include EVERY item. Do NOT
  abridge to "examples include …", "such as …", or "etc."; the items you drop
  are exactly what the orchestrator may need. Preserve edge cases, qualifiers,
  and conditions ("only if …", "except when …") — they change what the answer means.
- STRUCTURE what you found so it is scannable, not a prose wall: group related
  facts, label each distinct aspect, and use tight bullet lists instead of
  narrative paragraphs. Short organizing labels or headers are fine; skip
  decorative styling (heavy bold/italic), a closing summary, and a references
  section. Inline code and code snippets are encouraged when they carry the
  actual information.
- If the question's key term is overloaded and the findings resolve it to one
  domain, state the interpretation loudly first: "Interpreting <term> as
  <domain>, based on <evidence>." If the findings spread across different
  angles or domains for the same term, do NOT silently pick one — surface the
  candidate interpretations and flag that clarification is needed so the
  orchestrator can ask the user.
- Close with a "Next steps:" line or short list ONLY when the evidence leaves a
  meaningful gap for THIS query: name the specific follow-up search, the specific
  document/page to read next (by the title or pageId shown in the Source
  context), or the uncovered aspect of the question. When the <evidence_review>
  block lists <gaps>, <suggestions>, or <fetch_targets>, base the next steps on
  those exact items — do not invent others. When evidence review did not run,
  derive them from the findings, pointing ONLY at document identities that appear
  in the evidence. Every next step must target THIS query's specific gap, never
  generic advice. Omit the section entirely when the evidence already answers the
  query fully — do not pad. Next steps are forward-looking pointers, not evidence
  claims, so they are exempt from the per-claim fact-ID rule (cite an [Fn] only
  when pointing at an existing fact).
</constraints>

{SYNTHESIS_NO_ANSWER_CONTRACT}

<objective>
Directly answer the orchestrator's question using the findings below. Preserve
every concrete value and keyword — completeness of query-relevant detail outranks
brevity. The fact IDs (e.g. [F1], [F2]) correspond to specific pieces of evidence
— cite them inline as you make each claim. If the evidence leaves a meaningful
gap for this query, close with concrete next steps (specific follow-up searches
or specific documents to read).
</objective>"""

SUBAGENT_SYNTHESIZE_HUMAN_PROMPT = """\
<query>{query}</query>

<research_findings>
{findings_with_fact_ids}
</research_findings>

<evidence_review>
{evidence_review_block}
</evidence_review>{kg_context_block}"""


# ---------------------------------------------------------------------------
# review_answer — faithfulness check on synthesized answer (Phase 3)
# ---------------------------------------------------------------------------

REVIEW_ANSWER_SYSTEM_PROMPT = (
    """\
<identity>
You are a faithfulness reviewer. Your task is to verify that every claim in
the answer is supported by the provided evidence findings.
</identity>

<constraints>
- Focus ONLY on faithfulness: does each claim in the answer match the evidence?
- Citation number correctness is NOT your concern — that is handled separately.
- A claim is "unsupported" if it states something not present in the findings,
  or contradicts the findings.
- Paraphrasing is acceptable — the claim need not be verbatim.
- If the answer qualifies a claim ("likely", "appears to"), the evidence should
  support that level of certainty.
- A trailing "Next steps" note (follow-up searches to run, or documents to read
  next) is a forward-looking suggestion, NOT a claim about the evidence. Do NOT
  treat it as unsupported — exclude it from the faithfulness check entirely.
"""
    + _source_context_match_block(_REVIEW_ANSWER_SOURCE_MATCH_ACTION)
    + """
</constraints>

<objective>
1. Read the answer and the evidence findings.
2. For each substantive claim in the answer, check if it is supported by the evidence.
3. Set faithful=True if ALL claims are supported, False if ANY claim lacks support.
4. List any unsupported claims in unsupported_claims.
5. Provide a brief explanation of your assessment.
</objective>"""
)

REVIEW_ANSWER_HUMAN_PROMPT = """\
<query>{query}</query>

<answer_to_review>
{answer}
</answer_to_review>

<evidence_findings>
{findings_summary}
</evidence_findings>"""


# ---------------------------------------------------------------------------
# Legacy aliases — kept for backward compatibility during migration.
# Remove after all nodes are updated to use the new split prompts.
# ---------------------------------------------------------------------------

COMPRESS_CHUNKS_PROMPT = COMPRESS_HUMAN_PROMPT
COMPRESS_KG_ENTITIES_PROMPT = COMPRESS_KG_HUMAN_PROMPT
REVIEW_EVIDENCE_PROMPT = REVIEW_EVIDENCE_HUMAN_PROMPT

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/prompts/orchestrator_grounding.py
----
"""Grounding, clarification, general-knowledge, and deepagents auto-tool guidance.

These sections describe the planner's *behavior* — when to delegate, when to
answer directly, when to ask a follow-up — without depending on which tools
happen to be bound at runtime. The planner uses its own LLM capability for
clarification and general-knowledge answers; there is no ``clarify_user`` or
``general_knowledge`` tool to delegate to.
"""

_GROUNDING_SECTION = """
- Specialist sub-agents own their domain. When an available sub-agent's
  description matches the domain of the question, treat that sub-agent as the
  authoritative source of truth for that domain: delegate there and ground your
  answer in its output. Do not answer a domain question from your own general
  knowledge when a matching specialist exists, and do not let one specialist's
  output override another specialist on that other's home domain.
- Internal / company questions -> if a matching sub-agent is available to your `task`
  tool, delegate there first; otherwise explain that the relevant internal capability is
  unavailable in this session and ask for the missing scope if needed.
- General / public questions (programming, translation, public concepts) → answer
  directly from your own knowledge.
- Never fabricate. If you don't have the information, say so and offer to clarify
  or delegate.
"""

_CLARIFICATION_SECTION = """
Ask ONE concise clarifying question only when the user's intent is genuinely
ambiguous:
- The query could plausibly mean two different things.
- Required context (entity name, time range, scope) is missing.
- The user's preferred channel is unclear (internal docs vs. general knowledge).

Format: a single short question with 2-3 concrete options when possible, in the
user's language. Do NOT use clarification as a default opener — decide when you can.
"""

_GENERAL_KNOWLEDGE_SECTION = """
For non-company questions you can answer directly, use your own knowledge:
- Code → respond code first; explain only if non-obvious.
- Translation → translate in the user's target language.
- Text generation / reformulation → produce the requested output.
- Public technical explanations → concise, bullets over paragraphs.

Reply in the user's language. Skip preambles. Prefer bullets over paragraphs.
Label widely-accepted general knowledge with ``[GEN]`` when the user asks a
factual question and you are answering from training rather than from a
sub-agent's retrieval.
"""

_DEEPAGENTS_TOOL_GUIDELINES = """
Your `task`, `write_todos`, and filesystem tools are documented in their own
sections later in this prompt. The orchestrator-specific rules that override
that generic guidance:
- Use `write_todos` only for complex tasks that involve coordinating
  multiple specialized sub-agents or tools in sequence — example: gather status
  from one capability, map dependencies with another, then synthesize a single
  answer. A single delegation or a direct answer never needs a todo list.
- The generic `task` advice ("don't use it for trivial tasks or simple
  lookups") applies to general-purpose work, NOT to internal / company
  questions: sub-agents are your only access to internal systems and documents.
  Even a one-line internal lookup (a ticket status, an application record, a
  person) goes through the matching sub-agent — never answered from your own
  knowledge because the lookup feels too small to delegate.
"""

_UNCERTAINTY_SECTION = """
When you cannot answer — the entity, document, or fact is absent from every
sub-agent result and from your own knowledge — say so explicitly and FIRST,
before any speculation. Lead with the negative result, then offer leads under a
clearly labelled hint line. Never bury a "not found" under a paragraph of
hypotheses, and never present a guess as if it were a retrieved fact.

Format:
**`<thing>`** doesn't appear to exist / could not be found in the available sources.

Hints:
- <closest related fact, adjacent entity, or where the user might look next>
"""

_SUBAGENT_TASKING_SECTION = """
Use `task(subagent_type=...)` only to delegate to a specialized sub-agent
available to your `task` tool.

Choosing a sub-agent:
- When more than one available agent could fit, pick the one whose description
  best matches the need.
- The agent whose domain matches the question is the source of truth for that
  domain. Don't override its answer with another agent's output or your own
  general knowledge on its home domain.
- When both a documentation/knowledge agent and a live-system specialist match
  the question, the live specialist is the source of truth — query it first.
  Documentation is a written snapshot of what someone once recorded, so it can
  be stale; use it to enrich or investigate further, and flag documented
  information as possibly dated when it conflicts with or extends live results.
- A single question can raise more than one facet, each owned by a different
  agent. An open entity-identity or overview question — "what is X / tell me
  about X / describe X" about an application or system — has both a
  technical/structural facet (what it is, how it's built, what it connects to)
  and a functional/business facet (what it is for, what it does). First decide
  how many facets the question actually raises, then act:
  - Two genuine facets → cover BOTH and synthesize one combined answer. Query
    the live/structural source of truth first, then the documentation source for
    the functional side — and flag documented detail as possibly stale or wrong,
    especially where it conflicts with the live result.
  - One dimension only ("what does X depend on", "X's identifier", a purely
    functional how-to) → search only that dimension; do NOT fan out. Only if it
    comes back empty should you check whether the other facet holds anything
    (the routing-signal rule below).
- If its result is weak, partial, or empty, treat that as a routing signal, not
  a verdict — don't stop at "not found". A question can be framed like one domain
  yet have its real answer in another: a "how does X work / how is it set up /
  what's the procedure" question routed to a live-system specialist often has its
  substance in internal documentation instead, and vice-versa. Before concluding
  nothing exists, proactively make ONE re-route to the better-fit complementary
  agent — pick it from the available roster by which description matches the
  actual need, not the surface framing of the question.
- When the complementary attempt still doesn't answer (or a re-route genuinely
  isn't warranted), don't end on a bare "not found": give the user a concrete,
  actionable hint — which other capability or angle would likely surface it, or
  how to rephrase or scope the question so a specific agent can find it. A useful
  next step beats a dead end.
- If a sub-agent returns no data twice for the same need, stop re-trying it:
  rephrasing the same brief a third time almost never helps. Switch to another
  agent or surface the empty result to the user.
- If repeated delegation isn't converging on a good answer, stop: tell the user
  what you tried and ask how they'd like to proceed.
- You don't have to exhaust every angle up front: when the source-of-truth
  agent's answer covers every facet the question raises, return it and, only when
  deeper digging into an answered facet might genuinely help,
  offer it as a follow-up rather than spending more delegations now. This is
  about not re-deepening a facet that is already answered — it never licenses
  skipping a second facet the question clearly raises.

When writing the `task` prompt:
- Do not forward the raw user message unless it is already a complete scoped task.
- Rewrite the task as a standalone brief for the selected subagent.
- Include only: objective, relevant entities/IDs, scope/time/env constraints,
  known context, and expected output.
- Strip unrelated conversation, routing rationale, hidden/system instructions,
  credentials, tool traces, and unrelated capabilities.
- If a term in the request is overloaded across domains (e.g. "agent" = AI
  agent vs. log agent vs. support contact; an abbreviation that could expand
  several ways), put the disambiguating context you have — the intended domain,
  system, or scope from prior turns — into the task brief so the sub-agent
  retrieves the right sense. If you genuinely cannot disambiguate, say so in the
  brief so the sub-agent can flag it or ask, rather than guessing.
- Ask one clarification instead of delegating if required context is missing.
"""

_KNOWLEDGE_AGENT_TASKING_RULE = """
The Knowledge Agent answers from internal, human-written documentation — it is
the source of truth for internal documentation, and the best-effort complement
everywhere else. That documentation is a treasure chest: broad and rich, it may
hold something on almost any internal topic — but it is a written snapshot, so
parts of it can be stale. A domain specialist queries the live system; the
Knowledge Agent retrieves what someone once wrote about it. That is why the
specialist wins on its own domain, and why documented context is an enrichment,
not a substitute.
Reach for the Knowledge Agent when the question needs internal documentation.
Ordering: when a domain specialist matches the question, always try that
source-of-truth specialist before the Knowledge Agent. Delegate to the
Knowledge Agent afterwards only to enrich the answer — when the specialist's
result leaves a real gap, or the user asked a genuinely deep or complex
question that warrants documented depth. When a domain specialist returns
nothing or only a partial answer, you may delegate to the Knowledge Agent to
fill the gap with documented context — flag it as documented (possibly dated)
information when it conflicts with or extends live results. It complements, it
never overrides a domain specialist on that specialist's own domain.
You don't have to dive deep every time: when the specialist's answer covers the
question, return it as-is. Offer a deeper dive (into internal documentation or
via another agent) as a follow-up question only when the answer is thin or the
question hints at more depth — not as a closing ritual on every reply.
"""

_FUTURE_CAPABILITIES_SECTION = (
    """In the future, Twin will evolve toward a dynamic multi-agent and multi-engine architecture with controlled access to systems and data."""
)

__all__ = [
    "_CLARIFICATION_SECTION",
    "_DEEPAGENTS_TOOL_GUIDELINES",
    "_FUTURE_CAPABILITIES_SECTION",
    "_GENERAL_KNOWLEDGE_SECTION",
    "_GROUNDING_SECTION",
    "_SUBAGENT_TASKING_SECTION",
    "_UNCERTAINTY_SECTION",
]

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/subagents/build_knowledge_agent_subagent.py
----
"""Knowledge Agent subagent build + recursion-budget knobs.

Owns the full ``knowledge_agent`` build: it bakes the orchestrator-owned
:class:`KnowledgeAgentConfig` knobs, constructs the raw KA graph, enriches the
advertised capability with its searchable corpora, and packages the result via
the shared :func:`as_subagent` helper. The catalog (:mod:`registry`) only
references :func:`build_knowledge_agent` from a ``SubagentSpec`` — it does not
build anything itself.

Unlike the incident and topology subagents, there is no separate raw-graph
builder: ``create_knowledge_agent`` already returns a graph that accepts
``{"messages": [...]}`` directly (the last ``HumanMessage`` is the query —
exactly what deepagents' ``task`` sends), so the build is a single function.

The search-depth / review-cap defaults and the recursion-limit derivation also
live here so the budget formula has one home.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING, Literal

from sta_agent_engine.agents.base.prompts.capability_definition import CapabilityDefinition
from sta_agent_engine.agents.knowledge_agent import CompressConfig, ExpandConfig

from ..build_context import BuildContext
from ..sources.twin_ka_entries import list_twin_ka_corpora
from ._packaging import BuiltSubagent, as_subagent


if TYPE_CHECKING:
    from sta_agent_engine.agents.knowledge_agent import KnowledgeAgentConfig, RetrieverEntry


SearchDepth = Literal["fast", "deep", "thorough"]

DEFAULT_KA_SEARCH_DEPTH: SearchDepth = "fast"
DEFAULT_KA_REVIEW_CAP: int = 2


def _compute_recursion_limit(search_depth: SearchDepth, review_cap: int) -> int:  # noqa: ARG001 - search_depth kept for call-site symmetry with the budget knobs
    """Derive a recursion_limit that covers ``review_cap`` outer iterations.

    KA outer loop is ~6 node transitions per iteration plus inner-expansion
    churn. Budget 10 transitions per review iteration + headroom — well below
    the deepagents default of 999 so we always tighten, never loosen.
    """
    return max(review_cap, 1) * 10 + 20


_KA_RECURSION_LIMIT: int = _compute_recursion_limit(DEFAULT_KA_SEARCH_DEPTH, DEFAULT_KA_REVIEW_CAP)

_KA_CAPABILITY = CapabilityDefinition(
    name="knowledge_agent",
    description=(
        "Delegate to the Knowledge Agent: it searches internal, human-written "
        "documentation and returns a synthesized, cited answer. Use it for "
        "documented procedures, runbooks, and how-tos, or to complement another "
        "specialist's answer with documented context."
    ),
    use_for=[
        "Company procedures and processes",
        "Documented application context: setup guides, configuration procedures, runbooks, how-tos",
        "The functional/business description of an application — what it is for and what it does — complementing topology's technical/structural view",
        "Complementing a specialist agent's answer with documented context",
        "Investigating documented hints when a specialist's answer lacks the detail the request needs",
    ],
    examples=[
        '"Où trouver le runbook de déploiement de l\'application X?"',
        '"Comment configurer un ecosystème Vault?"',
        '"What is the procedure to do X?"',
    ],
    note=(
        "Complementary source, not a system of record — documentation can be stale or wrong. "
        "For live incident or ticket records, current infrastructure topology, or "
        "people/org lookups, delegate to the matching domain specialist first when "
        "one is available; Only use this agent afterwards to enrich with documented context if the specialist's answer lacks the detail the request needs."
    ),
)


def _orchestrator_ka_config(*, search_depth: SearchDepth = DEFAULT_KA_SEARCH_DEPTH, review_cap: int = DEFAULT_KA_REVIEW_CAP) -> KnowledgeAgentConfig:
    """Resolve a ``KnowledgeAgentConfig`` with the orchestrator-owned knobs baked in.

    Stamps ``mode="answer"`` (forced), the chosen ``search_depth`` /
    ``max_iterations``, ``subagent_mode=True``, the expansion/compression budgets,
    and ``streaming_enabled=False``. Streaming is disabled unconditionally in the
    subagent context: the KA runs behind the deepagents ``task`` boundary, so the
    orchestrator surfaces the final answer — a streamed sub-answer would leak
    intermediate (and at ``thorough`` depth, possibly-rejected) drafts the KA
    later replaces.
    """
    from sta_agent_engine.agents.knowledge_agent import KnowledgeAgentConfig

    resolved = KnowledgeAgentConfig.from_env(prefix="KA_")
    resolved.mode = "answer"
    resolved.search_depth = search_depth
    resolved.max_iterations = review_cap
    resolved.subagent_mode = True
    resolved.expand = ExpandConfig(enabled=False, structural_boundary_window=3, max_expansion_rounds=0)
    resolved.compress = CompressConfig(max_chars_per_group=280_000)
    resolved.streaming_enabled = False
    return resolved


def build_knowledge_agent(ctx: BuildContext) -> BuiltSubagent:  # noqa: ARG001 - ctx reserved for future persona-aware KA wiring
    """Build the Knowledge Agent subagent and enrich its capability with corpora."""
    from sta_agent_engine.agents.knowledge_agent import create_knowledge_agent

    graph = create_knowledge_agent(_get_ka_entries(), name="OrchestratorKnowledgeAgent", config=_orchestrator_ka_config())
    capability = replace(_KA_CAPABILITY, sources=list_twin_ka_corpora())
    # Pass the enriched ``capability`` (not the bare ``_KA_CAPABILITY`` skeleton) so the
    # corpora reach ``_compact_description`` — the compiled subagent's description is the
    # only channel the planner has for the KA's searchable corpora now that the
    # ``<capabilities>`` block is gone. ``BuiltSubagent.capability`` must match what the
    # planner is told, so both arguments use the same enriched capability.
    return BuiltSubagent(as_subagent(graph, capability, recursion_limit=_KA_RECURSION_LIMIT), capability)


@cache
def _get_ka_entries() -> list[RetrieverEntry]:
    """Build the KA retriever entries once per process.

    ``build_twin_ka_entries()`` opens ES retriever connections, so this is a
    network-touching call deferred to the function body — never at import.
    The result is cached so repeated KA builds (distinct rights slices, or a
    cache-disabled factory) reuse one set of retriever connections; the cache
    clears only on process restart, the same lifetime as the env-driven KA
    defaults baked into those entries.
    """
    from ..sources.twin_ka_entries import build_twin_ka_entries

    return build_twin_ka_entries()

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/subagents/build_topology_subagent.py
----
"""Topology subagent build (navigator raw graph + catalog packaging).

Two functions, one file:

- :func:`build_topology_subagent` returns the **raw** compiled navigator graph.
  The navigator accepts ``{"messages": [...]}`` input (its
  ``NavigatorAgentState`` extends ``AgentState`` which carries the message
  reducer) and its ``context_schema`` is ``AgentModeContext``, a ``TypedDict,
  total=False`` with **all optional fields** — the orchestrator does not need to
  populate it. Defaults route to the agent's own env-driven model + tool
  resolution. It is the middleware-aware seam: it forwards ``extra_middleware`` /
  ``post_middleware`` into the navigator factory's ``middlewares`` /
  ``post_middlewares`` slots.
- :func:`build_topology` is the catalog build function the ``SubagentSpec``
  references: it injects :func:`subagent_tool_call_guard` into the navigator's
  ``post_middlewares`` slot, binds ``recursion_limit`` on the inner runnable, and
  wraps via :func:`as_subagent`.

The navigator import is deferred to the raw builder's function body: the
navigator package opens a graph backend connection at module top level
(``repo = UKGRepository(...)``), so importing it at module load would break the
orchestrator's import-time-network-free contract.

This subagent is inactive in production until ``habilitation/policies.py`` lists
a permission key declared by the ``topology_agent`` registry entry.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import Runnable

from sta_agent_engine.agents.base.prompts.capability_definition import CapabilityDefinition

from ..build_context import BuildContext
from ._packaging import BuiltSubagent, as_subagent, subagent_tool_call_guard


# Navigator's outer loop hits ~10 transitions per reasoning iteration
# under the default mode-switching topology. Budget review_cap * 10 + 20
# to mirror the KA recursion-limit formula and leave headroom — well
# below the deepagents default of 999 so we always tighten.
_DEFAULT_REVIEW_CAP: int = 3
DEFAULT_TOPOLOGY_RECURSION_LIMIT: int = _DEFAULT_REVIEW_CAP * 10 + 20

_TOPOLOGY_CAPABILITY = CapabilityDefinition(
    name="topology_agent",
    description=(
        "Delegate to the topology agent: it queries the knowledge graph of applications and "
        "infrastructure — application records (identified by AP code or application name), "
        "service dependencies, and component relationships. It is the source of truth for an "
        "application's technical/structural identity: that it exists, its record, and how it "
        "connects. It does NOT hold the functional/business description of what an application "
        "is for or what it does — that side lives in internal documentation."
    ),
    use_for=[
        "Application technical identity: confirming an application exists and its record, by AP code (APxxxxx) or name",
        "How applications connect and depend on each other",
        "Dependency and impact analysis",
        "Communication flows between infrastructure components",
    ],
    examples=[
        '"What is application TWIN (code APxxxxx)?" — for the technical/structural side',
        '"What does APxxxxx depend on?"',
        '"Show me the impact of removing component X."',
    ],
    note=(
        "Returns technical/structural graph data only (records, dependencies, relationships). "
        "For the functional/business side of an application — what it is for and what it does — "
        'complement with the documentation agent. An open "what is application X?" question '
        "usually wants both the technical and the functional view."
    ),
)


def build_topology_subagent(
    *,
    graph_name: str = "OrchestratorTopologyAgent",
    extra_middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None = None,
    post_middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None = None,
) -> Runnable:
    """Return the raw compiled navigator graph for ``topology_agent``.

    :func:`build_topology` binds ``recursion_limit`` and wraps the result in a
    :class:`CompiledSubAgent`.

    Args:
        graph_name: Trace name passed to the navigator factory.
        extra_middleware: Middleware injected ahead of the navigator's own
            stack (the navigator factory's ``middlewares`` slot).
        post_middleware: Middleware appended after the navigator's own stack
            (the navigator factory's ``post_middlewares`` slot) — e.g. the
            soft-landing tool-budget cap that forces a final answer before the
            recursion ceiling.

    Returns:
        The compiled navigator graph (a :class:`Runnable`), unbound and
        unwrapped.
    """
    # Local import — the navigator package opens a graph backend
    # connection at module top level. Keeping the import inside the
    # function body preserves the orchestrator's import-time-network-free
    # contract; the connection only fires when the topology subagent is
    # actually constructed for a permitted user.
    from sta_agent_engine.agents.navigator_agent import get_navigator_graph

    return get_navigator_graph(
        name=graph_name,
        middlewares=tuple(extra_middleware or ()),
        post_middlewares=tuple(post_middleware or ()),
    )


def build_topology(ctx: BuildContext) -> BuiltSubagent:  # noqa: ARG001 - ctx reserved for future persona-aware topology wiring
    """Build the topology subagent (react agent — injects the tool-call guard).

    The navigator is a ``create_agent``-based react agent, so the per-subagent
    tool-call budget guard is forwarded into its own ``post_middlewares`` slot.
    The navigator dedups forwarded middleware by type, so the guard is never
    double-installed.
    """
    graph = build_topology_subagent(post_middleware=subagent_tool_call_guard())
    return BuiltSubagent(as_subagent(graph, _TOPOLOGY_CAPABILITY, recursion_limit=DEFAULT_TOPOLOGY_RECURSION_LIMIT), _TOPOLOGY_CAPABILITY)

-------

