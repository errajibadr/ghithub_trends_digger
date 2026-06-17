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
    "them as the answer — but do NOT drop or minimize them either. Surface the "
    "related/adjacent material IN FULL, each item clearly LABELED as related and "
    "not the exact thing asked, so the caller can judge it. Name what is missing "
    "for the direct answer, then develop everything you did find. Example: for "
    '"what are the failover procedures?", a rollback is a related recovery '
    "procedure, not strictly a failover — present its details, labeled as such, "
    "rather than ignoring it. The rule forbids passing adjacency off AS the "
    "answer; it never licenses silently swallowing relevant adjacent evidence."
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
- BIG GAP → lead with the gap. When the findings do not define or answer the
  CORE of the request, open with a single TLDR line before any header naming
  exactly what is missing (the missing definition, owner, dependency, hostname,
  procedure, config key, or decision):
  `TLDR: I could not find <specific requested thing> in the available evidence.`
  When you do have related material, extend it:
  `I only found leads about <specific lead(s)> [Fn].`
- MINOR GAP → no TLDR. When you have substantial relevant evidence and only a
  secondary aspect is missing, skip the TLDR — answer directly and note the
  missing aspect inline where it belongs. Do not manufacture a "could not find"
  framing when you largely did.
- Then DEVELOP everything the evidence does contain, in full: every relevant
  value, procedure, and related/adjacent procedure you found (e.g. a rollback
  when the query asked about failover), each clearly LABELED as what it is — a
  direct answer, a related procedure, or a lead — never upgraded into the asked
  thing, and never compressed away or dropped. Do not abbreviate leads to a
  single line; give their actual content.
- Keep next-step suggestions targeted to this query's actual remaining gap, and
  do NOT re-suggest a search already listed in <searches_already_run> — propose
  only a genuinely new angle or a specific document to fetch.
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
  gap, not generic advice. Do NOT re-suggest a search already listed in
  <searches_already_run> — propose only a genuinely new angle or a specific
  document to fetch. Omit it when the evidence answers the query fully.
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

<searches_already_run>
{executed_queries_block}
</searches_already_run>

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
  generic advice. Do NOT re-suggest a search already listed in
  <searches_already_run> — propose only a genuinely new angle or a specific
  document to fetch. Omit the section entirely when the evidence already answers
  the query fully — do not pad. Next steps are forward-looking pointers, not
  evidence claims, so they are exempt from the per-claim fact-ID rule (cite an
  [Fn] only when pointing at an existing fact).
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

<searches_already_run>
{executed_queries_block}
</searches_already_run>

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

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/nodes/synthesize.py
----
"""SynthesizeNode — generate answer from findings with fact-ID citations.

Uses CitationResolver for two-phase citation workflow:
1. prepare_context: tags each GroundedFact with [Fn] IDs for the LLM prompt
2. LLM generates answer referencing [Fn] IDs (pattern-copying, not bookkeeping)
3. resolve: deterministically rewrites [Fn] → [1],[2]... by first-mention order

The LLM never sees citation numbers — only stable fact IDs. This eliminates
the unreliable citation-bookkeeping failure mode.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph.state import RunnableConfig

from ...base.nodes import NodeBase
from ..knowledge_agent_config import KnowledgeAgentConfig
from ..knowledge_agent_prompts import (
    SUBAGENT_SYNTHESIZE_HUMAN_PROMPT,
    SUBAGENT_SYNTHESIZE_SYSTEM_PROMPT,
    SYNTHESIZE_HUMAN_PROMPT,
    SYNTHESIZE_SYSTEM_PROMPT,
)
from ..knowledge_agent_state import KnowledgeAgentContext, KnowledgeAgentState
from ..knowledge_agent_types import Citation, CoverageAssessment, FetchTarget, Finding, GroundedFact, KnowledgeNodeTask
from ..utils.citation_resolver import CitationResolver, StreamingCitationResolver, linkify_citations
from ..utils.executed_queries import extract_executed_queries, format_executed_queries_block


logger = logging.getLogger(__name__)


class SynthesizeNode(NodeBase[KnowledgeAgentContext]):
    """Generate a synthesized answer from compressed findings.

    Attributes:
        task: SYNTHESIS — resolves to synthesis model config.
        _resolver: CitationResolver for fact-ID tagging and resolution.
        _synthesis_config: Controls KG context inclusion and retry budget.
    """

    task: ClassVar[str] = KnowledgeNodeTask.SYNTHESIS

    def __init__(
        self,
        *,
        default_model: Any,
        agent_config: KnowledgeAgentConfig,
    ) -> None:
        super().__init__(default_model=default_model, node_config=agent_config)
        self._resolver = CitationResolver()
        self._agent_config = agent_config
        self._synthesis_config = agent_config.synthesis

    async def __call__(
        self,
        state: KnowledgeAgentState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Generate answer from findings.

        Args:
            state: Must contain ``findings`` and ``query``.
            config: LangGraph runnable config.

        Returns:
            Dict with ``answer``, ``answer_citations``, ``answer_attempt``,
            ``answer_message_id``.
        """
        query = state.get("query", "")
        findings: list[Finding] = state.get("findings", [])
        attempt = state.get("answer_attempt", 0) + 1

        if not findings:
            logger.warning("SynthesizeNode: no findings available — returning empty answer")
            return {"answer": "", "answer_citations": [], "answer_attempt": attempt}

        subagent_mode = self._get_bool_config("subagent_mode", self._agent_config.subagent_mode)

        kg_context_block = ""
        if self._synthesis_config.include_kg_context:
            kg_context_block = self._build_kg_context(findings)

        evidence_review_block = self._build_evidence_review_block(state.get("coverage"))

        # Surface the queries already searched this turn so "Next steps" cannot
        # re-propose a search that already ran. Derived read-only from the
        # tool-call thread — no dedicated state field. See utils/executed_queries.
        executed_queries_block = format_executed_queries_block(extract_executed_queries(state.get("messages")))

        model = self._resolve_model_for_task(self.task)
        # Tag for suppression — exclude_tags={"ka_synthesis"} in
        # TokenStreamingConfig prevents raw [Fn] tokens from the "messages"
        # stream reaching the frontend.
        model = model.with_config(tags=["ka_synthesis"])

        if subagent_mode:
            system_prompt = SUBAGENT_SYNTHESIZE_SYSTEM_PROMPT
            human_prompt_template = SUBAGENT_SYNTHESIZE_HUMAN_PROMPT
        else:
            system_prompt = SYNTHESIZE_SYSTEM_PROMPT
            human_prompt_template = SYNTHESIZE_HUMAN_PROMPT

        human_content, fact_map, input_truncated = self._build_capped_human_content(
            human_template=human_prompt_template,
            system_prompt=system_prompt,
            query=query,
            findings=findings,
            evidence_review_block=evidence_review_block,
            kg_context_block=kg_context_block,
            executed_queries_block=executed_queries_block,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]

        # Acquire stream writer — gracefully degrades when called outside a
        # LangGraph streaming context (CS-D6: ainvoke/astream resilience).
        # Citation metadata (citation_map, citation_order) is ALWAYS emitted
        # so the frontend references panel works regardless of streaming mode.
        # Token streaming is gated by streaming_enabled — when False (e.g. KA
        # invoked as a tool), no visible AI message leaks into the parent stream.
        try:
            writer = get_stream_writer()
        except RuntimeError:
            writer = None

        streaming_enabled = self._get_bool_config("streaming_enabled", self._agent_config.streaming_enabled)

        # Two independent stream channels:
        #   - citation_writer: structured citation_map / citation_order events
        #     for a KA-native references panel. Suppressed in subagent_mode —
        #     sub-agent consumers don't subscribe to them; the answer still
        #     ships inline [N] markers + an appended "Sources:" block.
        #   - token_writer: answer tokens, [Fn]→[N]-resolved on the fly by
        #     StreamingCitationResolver. Gated only by streaming_enabled — it
        #     works the same under the terse subagent prompt, so streaming and
        #     subagent_mode are orthogonal.
        citation_writer = None if subagent_mode else writer
        token_writer = writer if streaming_enabled else None

        if writer:
            logger.info(
                "SynthesizeNode: stream writer acquired (subagent_mode=%s, token_streaming=%s, citation_events=%s)",
                subagent_mode,
                token_writer is not None,
                citation_writer is not None,
            )
        else:
            logger.info("SynthesizeNode: no stream writer — falling back to ainvoke path")

        if citation_writer:
            citation_map = self._build_citation_map(fact_map)
            logger.info("SynthesizeNode: emitting citation_map with %d facts", len(citation_map))
            # No ``id`` here — citation_map is emitted before the LLM call so
            # the references panel can populate ahead of token streaming; the
            # answer message id is only known once the model responds.
            citation_writer({"type": "citation_map", "node": "synthesize", "data": citation_map})
        raw_answer = ""
        message_id: str | None = None
        max_empty_retries = self._synthesis_config.max_empty_response_retries
        for empty_attempt in range(max_empty_retries + 1):
            generated, message_id = await self._generate_answer(
                model,
                messages,
                fact_map,
                config,
                token_writer,
            )
            raw_answer = generated if isinstance(generated, str) else ""
            if raw_answer.strip():
                break
            if empty_attempt < max_empty_retries:
                logger.warning(
                    "SynthesizeNode: empty model response on attempt %d/%d — retrying",
                    empty_attempt + 1,
                    max_empty_retries + 1,
                )
        else:
            fallback = self._empty_response_fallback(state.get("coverage"))
            if citation_writer:
                citation_writer({"type": "citation_order", "node": "synthesize", "id": message_id, "data": {}})
            return {
                "answer": fallback,
                "answer_citations": [],
                "answer_attempt": attempt,
                "answer_message_id": message_id,
                "synthesis_input_truncated": input_truncated,
                # The model returned empty across every retry — the answer is a
                # canned fallback, not a grounded synthesis.
                "synthesis_empty": True,
            }

        result = self._resolver.resolve(
            raw_answer,
            fact_map,
            max_refs_per_sentence=self._synthesis_config.max_refs_per_sentence,
        )

        if citation_writer:
            citation_writer({"type": "citation_order", "node": "synthesize", "id": message_id, "data": result.fact_id_to_citation_num})

        if result.orphan_refs:
            logger.warning(
                "SynthesizeNode: LLM hallucinated %d fact IDs: %s",
                len(result.orphan_refs),
                result.orphan_refs,
            )

        logger.info(
            "SynthesizeNode: attempt %d, %d citations, %d orphans, %d uncited sentences",
            attempt,
            len(result.citations),
            len(result.orphan_refs),
            len(result.uncited_sentences),
        )

        # Rewrite inline [N] markers into markdown links [N](url) so the answer
        # renders clickable citations in any plain-markdown renderer — not only
        # in a KA-native frontend that styles bare [N] markers itself.
        answer_text = linkify_citations(result.resolved_text, result.citations)
        node_output: dict[str, Any] = {
            "answer_citations": result.citations,
            "answer_attempt": attempt,
            # The LLM's own streamed message id (chunk.id / response.id) — the
            # same id LangGraph puts on the AIMessageChunks of stream_mode=
            # "messages". OutputNode reuses it on the final answer AIMessage so
            # a token-streaming consumer can deduplicate. ``None`` if the model
            # produced no id.
            "answer_message_id": message_id,
            # True when the synthesis input hit the safety ceiling and evidence
            # was shed/clipped. OutputNode mirrors it into metadata.
            "synthesis_input_truncated": input_truncated,
        }
        # Append the plain-text ``Sources:`` block in every mode. It makes the
        # answer self-contained — a consumer reading only ``messages`` (or
        # ``state["answer"]``) sees grounded references without subscribing to
        # the custom citation stream events. The inline ``[N]`` markers and the
        # structured ``answer_citations`` are still populated for KA-native
        # frontends that render a richer references panel.
        sources_block = self._format_sources_block(result.citations)
        if sources_block:
            answer_text = f"{answer_text}\n\n{sources_block}"

        # The answer is NOT committed to the ``messages`` channel here. In
        # thorough mode this node may run several times (review_answer can
        # reject a draft and route back), so emitting per attempt would leak
        # drafts the graph later replaces. OutputNode — which runs once, after
        # the review loop settles — owns the messages-channel emission.
        node_output["answer"] = answer_text
        return node_output

    def _build_capped_human_content(
        self,
        *,
        human_template: str,
        system_prompt: str,
        query: str,
        findings: list[Finding],
        evidence_review_block: str,
        kg_context_block: str,
        executed_queries_block: str,
    ) -> tuple[str, dict[str, GroundedFact], bool]:
        """Render the synthesis human message under the input safety ceiling.

        Bounds the whole LLM call (system + human) to
        ``max_synthesis_input_tokens * 4`` chars. Findings are the elastic part:
        they get ``cap - overhead`` chars and are shed lowest-confidence-first by
        ``prepare_context``. A final hard clip guarantees the ceiling even when a
        single finding exceeds the budget — the goal is to never hard-fail the
        model on context size.

        Returns ``(human_content, fact_map, truncated)``.
        """
        cap_tokens = self._synthesis_config.max_synthesis_input_tokens
        cap_chars = cap_tokens * 4 if cap_tokens > 0 else None

        findings_budget: int | None = None
        if cap_chars is not None:
            overhead = len(system_prompt) + len(
                human_template.format(
                    query=query,
                    findings_with_fact_ids="",
                    evidence_review_block=evidence_review_block,
                    kg_context_block=kg_context_block,
                    executed_queries_block=executed_queries_block,
                )
            )
            findings_budget = max(cap_chars - overhead, 0)

        formatted_findings, fact_map = self._resolver.prepare_context(findings, max_chars=findings_budget)

        human_content = human_template.format(
            query=query,
            findings_with_fact_ids=formatted_findings,
            evidence_review_block=evidence_review_block,
            kg_context_block=kg_context_block,
            executed_queries_block=executed_queries_block,
        )

        # prepare_context dropped findings if not every fact survived into the map.
        total_facts = sum(len(f.key_facts) for f in findings)
        truncated = findings_budget is not None and len(fact_map) < total_facts

        # Backstop: a single oversized finding can still blow the ceiling — clip.
        if cap_chars is not None and len(system_prompt) + len(human_content) > cap_chars:
            human_content = human_content[: max(cap_chars - len(system_prompt), 0)]
            truncated = True

        if truncated:
            logger.warning(
                "SynthesizeNode: synthesis input exceeded the %d-token safety cap — evidence was truncated",
                cap_tokens,
            )

        return human_content, fact_map, truncated

    def _get_bool_config(self, key: str, default: bool) -> bool:
        """Resolve a boolean knob while tolerating direct node unit tests."""
        try:
            return bool(self.get_config(key, default))
        except RuntimeError:
            return bool(getattr(self._agent_config, key, default))

    @staticmethod
    def _build_evidence_review_block(coverage: CoverageAssessment | None) -> str:
        """Render evidence-review context for the synthesis prompt."""
        if coverage is None:
            return "<status>not_run</status>\n<reasoning>Evidence review did not run for this search depth.</reasoning>"

        status = "sufficient" if coverage.sufficient else "insufficient"
        lines = [
            f"<status>{status}</status>",
            f"<reasoning>{coverage.reasoning}</reasoning>",
        ]
        if coverage.gaps:
            lines.append("<gaps>")
            lines.extend(f"- {gap}" for gap in coverage.gaps[:5])
            lines.append("</gaps>")
        if coverage.query_suggestions:
            lines.append("<suggestions>")
            lines.extend(f"- {suggestion}" for suggestion in coverage.query_suggestions[:5])
            lines.append("</suggestions>")
        if coverage.fetch_targets:
            lines.append("<fetch_targets>")
            lines.extend(SynthesizeNode._format_fetch_target(target) for target in coverage.fetch_targets[:5])
            lines.append("</fetch_targets>")
        return "\n".join(lines)

    @staticmethod
    def _format_fetch_target(target: FetchTarget) -> str:
        return f"- {target.target_type}:{target.target_id} via {target.retriever_name} — {target.reason}"

    @staticmethod
    def _empty_response_fallback(coverage: CoverageAssessment | None) -> str:
        base = (
            "The available evidence was retrieved, but the synthesis model returned an empty response. "
            "I cannot provide a grounded answer from these findings."
        )
        if coverage is not None and not coverage.sufficient and coverage.gaps:
            return base + " Evidence review marked the evidence insufficient: " + "; ".join(coverage.gaps[:3]) + "."
        return base

    async def _generate_answer(
        self,
        model: Any,
        messages: list,
        fact_map: dict[str, GroundedFact],
        config: RunnableConfig,
        writer: Any | None,
    ) -> tuple[str, str | None]:
        """Generate the LLM answer; return ``(text, message_id)``.

        When *writer* is available and the model supports ``astream``,
        tokens are resolved on-the-fly through :class:`StreamingCitationResolver`
        and emitted as custom stream events.  Otherwise falls back to
        ``ainvoke`` (CS-D6 resilience).

        ``message_id`` is the LLM's own response id — ``chunk.id`` from the
        streamed ``AIMessageChunk`` s (all chunks of one response share it), or
        ``response.id`` on the ainvoke fallback. This is the same id LangGraph
        stamps on the chunks of ``stream_mode="messages"``. Each emitted token
        event carries it so a consumer can associate the stream with the final
        answer AIMessage that OutputNode commits under the same id. ``None`` if
        the model produced no id.
        """
        has_astream = hasattr(model, "astream")
        logger.info(
            "SynthesizeNode._generate_answer: writer=%s, has_astream=%s → %s path",
            writer is not None,
            has_astream,
            "streaming" if (writer and has_astream) else "ainvoke fallback",
        )

        if writer and has_astream:
            streaming_resolver = StreamingCitationResolver(
                fact_map,
                max_refs_per_sentence=self._synthesis_config.max_refs_per_sentence,
            )
            accumulated: list[str] = []
            token_count = 0
            custom_emit_count = 0
            message_id: str | None = None
            async for chunk in model.astream(messages, config=config):
                if message_id is None and getattr(chunk, "id", None):
                    message_id = chunk.id
                if chunk.content is None:
                    continue
                token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                if not token:
                    continue
                token_count += 1
                accumulated.append(token)
                resolved = streaming_resolver.feed(token)
                if resolved:
                    custom_emit_count += 1
                    writer({"type": "token", "node": "synthesize", "id": message_id, "content": resolved})

            remaining = streaming_resolver.flush()
            if remaining:
                custom_emit_count += 1
                writer({"type": "token", "node": "synthesize", "id": message_id, "content": remaining})

            logger.info(
                "SynthesizeNode: streamed %d tokens, emitted %d custom events, %d citations resolved",
                token_count,
                custom_emit_count,
                len(streaming_resolver.citations),
            )
            return "".join(accumulated), message_id

        # Non-streaming fallback
        response = await model.ainvoke(messages)
        if response.content is None:
            return "", getattr(response, "id", None)
        text = response.content if isinstance(response.content, str) else str(response.content)
        return text, getattr(response, "id", None)

    @staticmethod
    def _format_sources_block(citations: list[Citation]) -> str:
        """Render ``citations`` as a plain ``Sources:`` block for subagent mode.

        Output line shape, preserving ``citations`` order (already deduped by
        :class:`CitationResolver`)::

            Sources:
            [1] [Title](url)
            [2] [Title](url)

        Falls back to title-only when ``url`` is missing, url-only when
        ``title`` is empty, and skips rows where both are empty. Returns an
        empty string when no renderable rows remain so the caller can omit
        the block entirely.
        """
        rows: list[str] = []
        for idx, citation in enumerate(citations, start=1):
            title = citation.title.strip() if citation.title else ""
            url = citation.url.strip() if citation.url else ""
            if title and url:
                rows.append(f"[{idx}] [{title}]({url})")
            elif title:
                rows.append(f"[{idx}] {title}")
            elif url:
                rows.append(f"[{idx}] {url}")
        if not rows:
            return ""
        return "Sources:\n" + "\n".join(rows)

    @staticmethod
    def _build_citation_map(fact_map: dict[str, GroundedFact]) -> dict[str, dict[str, Any]]:
        """Build ``{Fn: {url, title, snippet, chunk_id, fact_text}}`` for the frontend."""
        citation_map: dict[str, dict[str, Any]] = {}
        for fid, gf in fact_map.items():
            c = gf.citation
            citation_map[fid] = {
                "url": c.url if c else None,
                "title": c.title if c else "",
                "snippet": c.snippet if c else "",
                "chunk_id": (c.metadata or {}).get("chunk_id", "") if c else "",
                "fact_text": gf.fact,
            }
        return citation_map

    @staticmethod
    def _build_kg_context(findings: list[Finding]) -> str:
        """Extract KG entity/relationship context from KG-mode findings."""
        kg_findings = [f for f in findings if f.compression_mode == "kg"]
        if not kg_findings:
            return ""

        lines = ["\n\n<knowledge_graph_context>"]
        for finding in kg_findings:
            lines.append(f"- {finding.topic}: {finding.summary}")
        lines.append("</knowledge_graph_context>")
        return "\n".join(lines)

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/utils/executed_queries.py
----
"""Recover the retriever queries that were actually searched this turn.

The planner (``PlanQueriesNode``) records each retrieval as a tool call on an
``AIMessage`` in the ``messages`` channel; the searched text lives in
``tool_call["args"]["query"]`` and accumulates across iterations. ``SynthesizeNode``
surfaces these to the model in a ``<searches_already_run>`` block so its
"Next steps" suggestions do not re-propose a search that already ran.

There is no dedicated state field for this — the queries are derived read-only
from the tool-call thread, so no schema/reducer/reset wiring is involved.
"""

from __future__ import annotations

from typing import Any


def extract_executed_queries(messages: list[Any] | None) -> list[str]:
    """Return the distinct retriever queries run this turn, in first-seen order.

    Walks the internal tool-call thread and collects each tool call's
    ``args["query"]``. Dedup is order-preserving and case-insensitive on the
    trimmed text, so repeated or cross-iteration duplicates collapse to one
    entry while the first-seen casing is kept.

    Tolerant of both shapes a tool call can take: a plain ``dict`` (LangChain's
    ``AIMessage.tool_calls`` entries) or an object exposing ``.args``.

    Args:
        messages: The KA ``messages`` channel — planner ``AIMessage``s carrying
            ``tool_calls`` interleaved with ``ToolMessage`` results. ``None`` is
            tolerated (treated as empty).

    Returns:
        Distinct non-empty query strings in the order first searched.
    """
    seen: set[str] = set()
    queries: list[str] = []
    for message in messages or []:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for call in tool_calls:
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
            if not isinstance(args, dict):
                continue
            raw = args.get("query")
            if not isinstance(raw, str):
                continue
            query = raw.strip()
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(query)
    return queries


def format_executed_queries_block(queries: list[str]) -> str:
    """Render executed queries as a bullet list for the synthesis prompt.

    Returns a neutral ``(none recorded)`` placeholder when no queries ran so the
    ``<searches_already_run>`` section is never empty (an empty section reads as
    "unknown" to the model rather than "nothing was searched").

    Args:
        queries: Output of :func:`extract_executed_queries`.

    Returns:
        A newline-joined ``- <query>`` list, or ``(none recorded)`` when empty.
    """
    if not queries:
        return "(none recorded)"
    return "\n".join(f"- {query}" for query in queries)

-------

