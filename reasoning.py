packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/compression/helpers.py
----
"""Shared compression utilities.

Functions used by multiple Compressor implementations.
Extracted for DRY (2d-D6) — no inheritance, plain functions.

Provides:
- group_by_page_group: Group chunks by page-first identifier
- group_by_doc_group: Group chunks by document-first identifier
- group_by_source_doc: Backward-compatible alias for page-first grouping
- content_hash: SHA-256 hash of text content for dedup
- filter_new_chunks: Hash filter with full_document_expansion bypass (SC-2)
- citation_from_chunk: Build a Citation directly from chunk metadata
- split_chunks_into_groups: Split a chunk list by count AND char budget

Note: grounded_facts_to_chunks moved to ChunkCompressor (2e-D5, encapsulated).
"""

from __future__ import annotations

import hashlib
import html
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from sta_agent_core.repositories import RetrievalChunk

from ..knowledge_agent_types import Citation


# Control characters (tab/newline excluded handling below) that would break a
# single-line XML open tag if they leaked into an attribute value.
_ATTR_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# Unified set of metadata keys propagated to Citation.metadata (4-D9).
# Used by citation_from_chunk (single source of truth for both LLM
# and passthrough paths) — ensures chunk_id always reaches the frontend.
#
# ``context_summary`` (the per-page contextual prefix — Confluence space, page
# path, parent breadcrumbs) and ``apcode`` carry strong page-identity signal:
# they let a downstream reviewer/synthesizer tell whether a generic-looking
# chunk actually belongs to the asked entity or to a different team/app/space.
# Without them on the Citation, that signal — visible to the compressor in the
# ``<document>``/``<context_summary>`` tags — would be lost before synthesis.
# ``entity`` is handled separately in ``citation_from_chunk`` (it arrives as a
# dict and is normalized to its name) so its child-entity list never rides
# along on every citation.
_CITATION_METADATA_KEYS = frozenset(
    {
        "pageId",
        "page_id",
        "doc",
        "chunk_id",
        "chunk_index",
        "full_doc_id",
        "source_path",
        "appName",
        "apcode",
        "context_summary",
        "lastDocUpdate",
        "lastDocIngestion",
    }
)


def _first_metadata_value(*values: Any) -> str | None:
    """Return the first non-empty metadata value as a string."""
    for value in values:
        if value is None or value == "":
            continue
        return str(value)
    return None


def _chunk_index_sort_key(chunk: RetrievalChunk) -> tuple[int, int, int | str]:
    """Sort numeric chunk_index values numerically, missing values last."""
    raw_index = chunk.metadata.get("chunk_index")
    if raw_index is None or raw_index == "":
        return (1, 0, "")
    if isinstance(raw_index, int):
        return (0, 0, raw_index)
    if isinstance(raw_index, float) and raw_index.is_integer():
        return (0, 0, int(raw_index))
    if isinstance(raw_index, str):
        try:
            return (0, 0, int(raw_index))
        except ValueError:
            return (0, 1, raw_index)
    return (0, 1, str(raw_index))


def _page_sort_key(chunk: RetrievalChunk) -> tuple[int, str]:
    """Sort chunks by page identifier, with missing page ids last."""
    page_id = _first_metadata_value(chunk.metadata.get("pageId"), chunk.metadata.get("page_id"))
    if page_id is None:
        return (1, "")
    return (0, page_id)


def group_by_source_doc(
    chunks: list[RetrievalChunk],
) -> dict[str, list[RetrievalChunk]]:
    """Group chunks by legacy page-first source identifier.

    This function preserves the original page-first behavior for existing
    internal callers. New code should choose ``group_by_page_group`` or
    ``group_by_doc_group`` based on the desired grouping semantics.
    """
    return group_by_page_group(chunks)


def group_by_page_group(
    chunks: list[RetrievalChunk],
) -> dict[str, list[RetrievalChunk]]:
    """Group chunks by page-first identifier, sorted by chunk_index.

    Key priority:
    ``pageId/page_id → doc/full_doc_id/source_path → source_url → retriever → unknown``.

    Contract: each returned group is always sorted by ``chunk_index`` so
    downstream consumers (compression, synthesis) always see stable
    page-order context.
    """
    groups: dict[str, list[RetrievalChunk]] = defaultdict(list)
    for chunk in chunks:
        # Single source of truth for the page key — ``page_group_key`` owns the
        # ladder so the formatter's ``<document>`` boundaries can never drift
        # from this grouping.
        groups[page_group_key(chunk)].append(chunk)

    return {key: sorted(group, key=_chunk_index_sort_key) for key, group in sorted(groups.items())}


def group_by_doc_group(
    chunks: list[RetrievalChunk],
) -> dict[str, list[RetrievalChunk]]:
    """Group chunks by document-first identifier, sorted by page then chunk.

    Key priority:
    ``doc → full_doc_id → source_path → source_url → pageId/page_id → retriever → unknown``.

    Contract: documents are never mixed in one group when document metadata is
    present, and chunks within each document are ordered by
    ``(pageId/page_id, chunk_index)``.
    """
    groups: dict[str, list[RetrievalChunk]] = defaultdict(list)
    for chunk in chunks:
        meta = chunk.metadata
        group_key = (
            _first_metadata_value(
                meta.get("doc"),
                meta.get("full_doc_id"),
                meta.get("source_path"),
            )
            or chunk.source_url
            or _first_metadata_value(meta.get("pageId"), meta.get("page_id"))
            or chunk.retriever_type
            or "unknown"
        )
        groups[group_key].append(chunk)

    return {key: sorted(group, key=lambda c: (_page_sort_key(c), _chunk_index_sort_key(c))) for key, group in sorted(groups.items())}


def page_group_key(chunk: RetrievalChunk) -> str:
    """Return the page identifier used to bracket one ``<document>`` block.

    Mirrors the key priority of ``group_by_page_group`` so the prompt's page
    boundaries line up with the page grouping used everywhere else:
    ``pageId/page_id → doc/full_doc_id/source_path → source_url → retriever →
    unknown``.
    """
    meta = chunk.metadata
    return (
        _first_metadata_value(
            meta.get("pageId"),
            meta.get("page_id"),
            meta.get("doc"),
            meta.get("full_doc_id"),
            meta.get("source_path"),
        )
        or chunk.source_url
        or chunk.retriever_type
        or "unknown"
    )


def order_chunks_by_page(chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
    """Flatten ``group_by_page_group`` into a page-contiguous, sorted chunk list.

    Chunks of one page sit together and in ``chunk_index`` order. Callers that
    number chunks for the LLM (``_format_chunks_for_prompt``) and resolve those
    numbers back to chunks (``_resolve_citations_map``) must both walk this same
    order, or a ``source_index`` would point at the wrong chunk. The operation
    is idempotent — ordering an already-ordered list returns the same order.
    """
    ordered: list[RetrievalChunk] = []
    for group in group_by_page_group(chunks).values():
        ordered.extend(group)
    return ordered


def _xml_attr(value: str) -> str:
    """Escape a value for a double-quoted XML attribute, scrubbing line breaks.

    Newlines / control chars are collapsed to spaces first so a metadata value
    (e.g. a title carrying a stray newline) cannot break the single-line
    ``<document …>`` open tag onto two lines.
    """
    flattened = _ATTR_CONTROL_CHARS.sub(" ", (value or "").replace("\n", " ").replace("\r", " ")).strip()
    return html.escape(flattened, quote=True)


def _staleness_attrs(meta: dict[str, Any]) -> str:
    """Render freshness / ``staleness`` attributes from chunk metadata.

    Prefers the canonical ``lastDocUpdate`` key (source content age — the
    signal that actually matters for staleness) and falls back to
    ``lastDocIngestion`` (index age — a weak lower bound on content age, since
    re-ingestion makes old content look fresh). Both are ISO 8601 strings
    normalized by the retriever's result mapper. Returns an empty string when
    neither is present/parseable, so corpora without the fields render
    identically to before. ``staleness`` is the age in whole days at formatting
    time — it lives in a per-retrieval tool message, never in the cached system
    prefix.
    """
    for key, attr in (("lastDocUpdate", "lastUpdated"), ("lastDocIngestion", "lastIngested")):
        raw = meta.get(key)
        if not raw:
            continue
        try:
            stamped = datetime.fromisoformat(str(raw))
        except ValueError:
            continue
        if stamped.tzinfo is None:
            stamped = stamped.replace(tzinfo=UTC)
        age_days = max(0, (datetime.now(UTC) - stamped).days)
        return f' {attr}="{stamped.date().isoformat()}" staleness="{age_days}d"'
    return ""


def document_open_tag(chunk: RetrievalChunk) -> str:
    """Build the opening ``<document …>`` tag for a page group.

    Page metadata (pageId, title, doc, source, ingestion staleness) travels as
    attributes so it is rendered ONCE per page instead of repeated on every
    chunk. Title falls back ``title → doc_title → doc → source_url →
    "Untitled"`` to mirror the prior per-chunk formatting.
    """
    meta = chunk.metadata
    page_id = _first_metadata_value(meta.get("pageId"), meta.get("page_id"))
    title = meta.get("title") or meta.get("doc_title") or meta.get("doc") or chunk.source_url or "Untitled"
    doc = meta.get("doc") or ""
    source = chunk.source_url or meta.get("url") or ""

    attrs = ""
    if page_id:
        attrs += f'pageId="{_xml_attr(str(page_id))}" '
    attrs += f'title="{_xml_attr(str(title))}"'
    if doc:
        attrs += f' doc="{_xml_attr(str(doc))}"'
    if source:
        attrs += f' source="{_xml_attr(str(source))}"'
    attrs += _staleness_attrs(meta)
    return f"<document {attrs}>"


def group_context_summary(group: list[RetrievalChunk]) -> str | None:
    """First non-empty ``context_summary`` across a page group, else ``None``.

    The contextual prefix is identical for every chunk of a production page, but
    a group can mix chunks whose prefix is empty (synthetic re-compression
    chunks, expansion fetches, or chunks from another retriever sharing the page
    key) with chunks that carry it. Scanning the whole group — rather than
    trusting the first chunk — keeps the page context from being silently dropped
    when the lowest-``chunk_index`` chunk happens to lack it.
    """
    for chunk in group:
        summary = chunk.metadata.get("context_summary")
        if summary:
            return str(summary)
    return None


def group_document_chunk(group: list[RetrievalChunk]) -> RetrievalChunk:
    """Pick the chunk whose metadata best populates the ``<document>`` tag.

    Prefer the first chunk carrying a real title/doc (same group-mixing concern
    as ``group_context_summary``); fall back to the first chunk so the tag is
    always emitted.
    """
    for chunk in group:
        meta = chunk.metadata
        if meta.get("title") or meta.get("doc_title") or meta.get("doc"):
            return chunk
    return group[0]


def content_hash(text: str) -> str:
    """SHA-256 hash of text content for dedup.

    Extracted from ChunkCompressionMethod._content_hash.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def filter_new_chunks(
    chunks: list[RetrievalChunk],
    existing_hashes: set[str],
) -> list[RetrievalChunk]:
    """Filter chunks to only new (uncompressed) ones.

    Bypasses the hash filter for chunks marked with full_document_expansion
    (2b-D11): when ExpandNode fetches a full document, all chunks must
    reach the compression method for coherent cross-chunk reasoning.

    Used by ChunkCompressionMethod and PassthroughCompressionMethod (SC-2).
    """
    return [c for c in chunks if c.metadata.get("full_document_expansion") or content_hash(c.content) not in existing_hashes]


def citation_from_chunk(
    chunk: RetrievalChunk,
    retriever_name: str,
) -> Citation:
    """Build a Citation directly from chunk metadata.

    Single source of truth for Citation construction — used by both
    the LLM path (via ``_resolve_citations_map``) and the passthrough path.

    ``entity`` is normalized here rather than via the passthrough whitelist:
    the Elastic mapper emits it as a ``{name, id, childs, is_opal}`` object, and
    carrying the whole dict (with its potentially large ``childs`` list) on every
    citation would bloat state and every rendered finding line. Only the entity
    *name* carries the page-identity signal downstream rendering needs, so that
    is all that rides along — as a plain string.
    """
    metadata = {k: v for k, v in chunk.metadata.items() if k in _CITATION_METADATA_KEYS}

    raw_entity = chunk.metadata.get("entity")
    if isinstance(raw_entity, dict):
        entity_name = raw_entity.get("name")
    elif isinstance(raw_entity, str):
        entity_name = raw_entity
    else:
        entity_name = None
    if entity_name:
        metadata["entity"] = entity_name

    return Citation(
        title=chunk.metadata.get("title") or chunk.metadata.get("doc_title") or chunk.metadata.get("doc") or chunk.source_url or "Unknown",
        url=chunk.source_url or chunk.metadata.get("doc") or None,
        source_type=chunk.retriever_type or retriever_name,
        snippet=chunk.content[:300],
        retriever_name=retriever_name,
        metadata=metadata,
    )


def split_chunks_into_groups(
    chunks: list[RetrievalChunk],
    max_count: int,
    max_chars: int,
) -> list[list[RetrievalChunk]]:
    """Split a chunk list into sub-groups respecting both count and char limits.

    Iterates *chunks* in order, accumulating into the current sub-group.
    A new sub-group starts whenever adding the next chunk would exceed
    *max_count* or *max_chars* (whichever limit is hit first).

    A single oversized chunk (content > *max_chars*) is never dropped —
    it gets its own group so the caller can log a warning and let the
    LLM API handle the overflow.

    Args:
        chunks: Pre-sorted chunk list (caller is responsible for ordering).
        max_count: Maximum number of chunks per sub-group.
        max_chars: Maximum cumulative ``len(c.content)`` per sub-group.

    Returns:
        Non-empty list of non-empty sub-groups preserving input order.
        Returns ``[[]]``-free: empty *chunks* → empty list.
    """
    if max_count < 1:
        raise ValueError(f"max_count must be >= 1, got {max_count}")
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")

    if not chunks:
        return []

    groups: list[list[RetrievalChunk]] = []
    current: list[RetrievalChunk] = []
    current_chars = 0

    for chunk in chunks:
        chunk_len = len(chunk.content)
        would_exceed = len(current) >= max_count or (current and current_chars + chunk_len > max_chars)
        if would_exceed:
            groups.append(current)
            current = []
            current_chars = 0

        current.append(chunk)
        current_chars += chunk_len

    if current:
        groups.append(current)

    return groups


def grounded_facts_to_chunks(
    grounded_facts: list,
    retriever_name: str,
) -> list[RetrievalChunk]:
    """Reconstruct synthetic RetrievalChunks from GroundedFacts.

    Deprecated: prefer ChunkCompressor._grounded_facts_to_chunks (encapsulated).
    Kept as public function for backward compatibility during migration.
    """
    chunks: list[RetrievalChunk] = []
    for gf in grounded_facts:
        citation = gf.citation
        chunks.append(
            RetrievalChunk(
                content=gf.fact,
                source_url=citation.url or "" if citation else "",
                retriever_type=retriever_name,
                metadata=dict(citation.metadata) if citation else {},
                score=None,
            )
        )
    return chunks

-------

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
Your output will be consumed by another agent, not rendered to an end user —
keep it terse and information-dense.
</identity>

<constraints>
{_SYNTHESIS_GROUNDING_CONSTRAINTS}
{_anti_approximation_block(_SYNTH_DIRECTNESS_ACTION)}
- Cite supporting fact IDs in brackets, e.g. [F1], [F3]. Use only the [Fn]
  tags shown next to each fact — never invent IDs.
- Every substantive claim must cite at least one fact ID.
- Cite at most 2 supporting fact IDs per claim.
- Be concise — but concise means cutting filler and connective prose, NOT
  dropping content. Preserve every concrete specific verbatim: exact values,
  identifiers, version numbers, config keys, parameter names, commands, code
  snippets, error strings, and domain keywords. An orchestrator cannot act on
  a summary that lost the values.
- Prefer 1-3 short paragraphs OR a tight bullet list. No section headers, no
  bold/italic styling, no closing summary, no references section. Inline code
  and code snippets ARE allowed when they carry the actual information.
- If the question's key term is overloaded and the findings resolve it to one
  domain, state the interpretation loudly first: "Interpreting <term> as
  <domain>, based on <evidence>." If the findings spread across different
  angles or domains for the same term, do NOT silently pick one — surface the
  candidate interpretations and flag that clarification is needed so the
  orchestrator can ask the user.
</constraints>

{SYNTHESIS_NO_ANSWER_CONTRACT}

<objective>
Directly answer the orchestrator's question using the findings below. Be
concise, but keep every concrete value and keyword. The fact IDs (e.g. [F1],
[F2]) correspond to specific pieces of evidence — cite them inline as you make
each claim.
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

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/knowledge_bridge_channels.py
----
"""Shared state-channel contract between an orchestrator and the Knowledge Agent.

A subagent invoked through the deepagents ``task`` tool exchanges data with its
parent ONLY through state keys that BOTH sides declare (``task`` copies the
parent's state into the subagent input and the subagent's result back out,
minus a fixed exclusion set). For a key to cross either way the channel name
must be byte-identical in both schemas — that shared name *is* the propagation
contract. This module is the single source of truth for those two channels so
the orchestrator middleware and the KA state schemas cannot drift apart.

Two channels, deliberately different lifetimes:

``ka_metadata_scope`` (INPUT, orchestrator → KA)
    A per-invocation request scope: an optional set of FILTER-ONLY axes the KA
    must hard-filter retrieval to — document ids, apcode(s), application
    name(s), and entity(ies). Every axis is a hard AND-filter, never a soft
    boost: a caller cannot smuggle a boost across this trust boundary (any
    ``*_boost`` key on the payload is dropped with a warning). It is run-scoped
    — it must never bleed into the next conversation turn on a checkpointed
    thread. ``UntrackedValue`` delivers exactly that: it survives across
    super-steps within a run (so a value set at run start is still readable when
    a retriever tool fires several steps later), but is never checkpointed, so
    the next run on the same thread starts fresh with no reset node required.

``ka_sources`` (OUTPUT, KA → orchestrator)
    Grounding sources surfaced by the KA as minimal JSON-safe dicts. A single
    planner turn can delegate to the KA more than once — sequentially across
    super-steps or concurrently within one super-step — so this channel needs
    an accumulating reducer (a plain ``LastValue`` would raise on concurrent
    writes and silently overwrite on sequential ones). The reducer concatenates
    and de-duplicates; callers reset it per run with ``Overwrite(value=[])``
    (a bare ``[]`` is a no-op under an accumulate reducer).

Both values are plain JSON-serializable dicts/lists so they survive checkpoint
serialization and the ``task`` ``Command(update=...)`` round-trip without
carrying vendor objects across the boundary.
"""

from __future__ import annotations

import logging
from typing import Annotated, NotRequired

from langchain.agents.middleware.types import OmitFromInput, OmitFromOutput
from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import TypedDict


logger = logging.getLogger(__name__)


#: State key carrying the orchestrator-supplied metadata scope into the
#: Knowledge Agent. Byte-identical in the orchestrator middleware and KA state.
KA_METADATA_SCOPE_KEY = "ka_metadata_scope"

#: State key carrying the Knowledge Agent's grounding sources back to the
#: orchestrator. Byte-identical in the orchestrator middleware and KA state.
KA_SOURCES_KEY = "ka_sources"


class KaMetadataScope(TypedDict, total=False):
    """The orchestrator-supplied, FILTER-ONLY request scope for the KA.

    Every axis is a hard AND-filter — there are no boost axes here by design.
    The orchestrator seeds this scope before delegating; the KA narrows each
    retriever's build-time filter ceiling with it. A caller cannot widen the
    ceiling and cannot turn any of these into a soft boost.

    All fields are ``NotRequired`` — a caller supplies only the axes it wants to
    constrain; absent axes leave that part of retrieval unscoped.
    """

    doc_ids: NotRequired[list[str]]
    apcode: NotRequired[list[str]]
    app_name: NotRequired[list[str]]
    entity: NotRequired[list[str]]


def merge_ka_sources(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Accumulate KA source dicts across multiple KA calls — pure concatenation.

    Appends ``right`` (newly returned) after ``left`` (existing) with **no
    cross-call de-duplication**. Each KA call already returns its citations
    ordered and 1:1 with its own inline ``[N]`` markers (see
    ``OutputNode._build_ka_sources``); concatenating the per-call blocks keeps
    the accumulated channel a **contiguous, position-stable** list. That is what
    lets the orchestrator offset a later call's numbering by the count of
    sources already surfaced (call 2's ``[1]`` → ``[N+1]``) and have every
    ``[K]`` map to the K-th row of the displayed list.

    De-duplicating across calls would break that contiguity: a document re-cited
    by a later call would collapse back to its earlier position, so the offset
    arithmetic would land on the wrong row. The cost of concatenation is that a
    document cited by two separate KA calls appears in the panel twice — accepted
    in exchange for deterministic, offsettable numbering.

    Reset the channel per run with ``langgraph.types.Overwrite(value=[])``;
    returning a bare ``[]`` here is a no-op because this reducer only accumulates.

    Args:
        left: Sources already on the channel (``None`` on first write).
        right: Sources returned by the latest KA delegation (``None`` allowed).

    Returns:
        ``left`` followed by ``right``, a fresh list (never a shared reference).
    """
    return [*(left or []), *(right or [])]


class KnowledgeBridgeChannels(TypedDict, total=False):
    """The two shared channels, defined once for both sides to inherit/declare.

    Declaring these keys on BOTH the orchestrator (via a middleware
    ``state_schema``) and the KA graph state is what lets ``task`` carry them
    across the delegation boundary — neither key is in deepagents'
    ``_EXCLUDED_STATE_KEYS``.
    """

    # INPUT — run-scoped, never checkpointed (see module docstring). The caller
    # seeds it and it is never echoed back out, so it is ``OmitFromOutput``
    # (NOT ``OmitFromInput`` — that would drop the caller-supplied value).
    ka_metadata_scope: NotRequired[Annotated[KaMetadataScope | None, UntrackedValue, OmitFromOutput]]
    # OUTPUT — accumulate + dedupe across repeated KA delegations in one run.
    # The reducer is REQUIRED: on the orchestrator side this schema is merged in
    # by a deepagents/``create_agent`` middleware, and concurrent ``task`` calls
    # write this channel in one super-step (a plain ``LastValue`` raises "can
    # receive only one value per step").
    #
    # ORDER IS LOAD-BEARING: ``OmitFromInput`` MUST precede ``merge_ka_sources``.
    # ``create_agent`` keeps the LAST channel-defining metadata in an
    # ``Annotated``; with the reducer first (``..., merge_ka_sources,
    # OmitFromInput``) the visibility marker shadows it and the channel silently
    # degrades to ``LastValue`` — reintroducing the concurrent-write crash. With
    # ``OmitFromInput`` first the reducer wins, so the channel both accumulates
    # AND stays hidden from the input schema.
    ka_sources: NotRequired[Annotated[list[dict], OmitFromInput, merge_ka_sources]]


def normalize_doc_ids(raw: object) -> list[str]:
    """Coerce a raw ``doc_ids`` value into a clean ``list[str]``.

    Accepts a single string or an iterable of strings; trims whitespace and
    drops empty / non-string entries. Returns an empty list for anything that
    cannot yield document ids (``None``, wrong type, all-empty). Centralizing
    this keeps the page-id-vs-chunk-id namespace question in one place: the
    caller still has to ensure the ids match the retriever's keyword field, but
    a malformed value can never reach the retriever as a degenerate filter.

    Args:
        raw: The ``doc_ids`` value pulled from ``ka_metadata_scope`` state.

    Returns:
        Whitespace-trimmed, de-duplicated, order-preserving list of doc ids.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates: list[object] = [raw]
    elif isinstance(raw, (list, tuple, set)):
        candidates = list(raw)
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, str):
            continue
        trimmed = item.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        out.append(trimmed)
    return out


def normalize_apcode(raw: object) -> list[str]:
    """Coerce a raw ``apcode`` value into a clean ``list[str]`` for hard filtering.

    Mirrors :func:`normalize_doc_ids` — accepts a single apcode string or an
    iterable of them, trims and drops empties. The values flow into a
    ``MetadataScope.apcode_filter`` (a hard AND-filter), so a degenerate value
    must never reach Elasticsearch.

    Args:
        raw: The ``apcode`` value pulled from ``ka_metadata_scope`` state.

    Returns:
        Whitespace-trimmed, de-duplicated, order-preserving list of apcodes.
    """
    return normalize_doc_ids(raw)


def normalize_app_name(raw: object) -> list[str]:
    """Coerce a raw ``app_name`` value into a clean ``list[str]`` for hard filtering.

    Mirrors :func:`normalize_doc_ids` — accepts a single app-name string or an
    iterable of them, trims and drops empties. The values flow into a
    ``MetadataScope.app_name_filter`` (a hard AND-filter).

    Args:
        raw: The ``app_name`` value pulled from ``ka_metadata_scope`` state.

    Returns:
        Whitespace-trimmed, de-duplicated, order-preserving list of app names.
    """
    return normalize_doc_ids(raw)


def normalize_entity(raw: object) -> list[str]:
    """Coerce a raw ``entity`` value into a clean ``list[str]`` for hard filtering.

    Mirrors :func:`normalize_doc_ids` — accepts a single entity string or an
    iterable of them, trims and drops empties. The values flow into a
    ``MetadataScope.entity_filter`` (a hard AND-filter).

    Args:
        raw: The ``entity`` value pulled from ``ka_metadata_scope`` state.

    Returns:
        Whitespace-trimmed, de-duplicated, order-preserving list of entities.
    """
    return normalize_doc_ids(raw)


#: The four FILTER-ONLY axes a caller may supply on ``ka_metadata_scope``.
_KNOWN_SCOPE_AXES: frozenset[str] = frozenset({"doc_ids", "apcode", "app_name", "entity"})


def read_ka_metadata_scope(raw: object) -> KaMetadataScope:
    """Validate and normalize a raw ``ka_metadata_scope`` payload.

    Ignores unknown keys AND any ``*_boost`` key — a boost cannot cross this
    trust boundary, so a caller (or a buggy upstream) attempting to smuggle one
    is defended against here, not at the retriever. Dropped keys emit a
    ``logger.warning`` so the misuse is diagnosable. Each known axis is
    normalized into a clean ``list[str]`` (trimmed, de-duplicated, empties
    dropped); axes that normalize to empty are omitted from the result.

    Args:
        raw: The value pulled off ``ka_metadata_scope`` state — expected to be a
            dict, but anything non-dict (``None``, wrong type) yields an empty
            scope so retrieval proceeds unfiltered.

    Returns:
        A :class:`KaMetadataScope` carrying only the non-empty known axes.
    """
    if not isinstance(raw, dict):
        return {}

    dropped: list[str] = []
    scope: KaMetadataScope = {}
    for key, value in raw.items():
        if not isinstance(key, str) or key not in _KNOWN_SCOPE_AXES:
            # Unknown key or a smuggled ``*_boost`` — never a valid filter axis.
            dropped.append(str(key))
            continue
        if key == "doc_ids":
            normalized = normalize_doc_ids(value)
        elif key == "apcode":
            normalized = normalize_apcode(value)
        elif key == "app_name":
            normalized = normalize_app_name(value)
        else:  # key == "entity"
            normalized = normalize_entity(value)
        if normalized:
            scope[key] = normalized  # type: ignore[literal-required]

    if dropped:
        logger.warning(
            "ka_metadata_scope dropped %d unrecognized key(s): %s — only %s are FILTER-ONLY axes; "
            "boosts cannot cross the orchestrator→KA trust boundary.",
            len(dropped),
            sorted(dropped),
            sorted(_KNOWN_SCOPE_AXES),
        )
    return scope

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/nodes/output.py
----
"""OutputNode — package KnowledgeAgentFindings or KnowledgeAgentAnswer from state.

Terminal node that assembles the final output bundle from accumulated
state (findings, coverage, metadata). In answer mode, wraps the evidence
bundle in a KnowledgeAgentAnswer with the synthesized answer and citations.

This is the single node that commits the answer to the ``messages`` channel.
Emitting here (rather than in SynthesizeNode) means the answer lands on
``messages`` only once, *after* any ``review_answer`` retry loop has settled —
so a draft the faithfulness reviewer later rejects never reaches the consumer.

Also extracts KG subgraph data from LightRAG responses for visualization.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph.state import RunnableConfig

from sta_agent_core.repositories.retrievers.lightrag import LightRAGSearchResponse, from_lightrag_response

from ..knowledge_agent_state import KnowledgeAgentState
from ..knowledge_agent_types import (
    PASSTHROUGH_FALLBACK_MODE,
    Citation,
    Finding,
    KnowledgeAgentAnswer,
    KnowledgeAgentFindings,
)


logger = logging.getLogger(__name__)

# Canned user/agent-facing message surfaced when retrieval tools executed but
# yielded zero evidence. Kept short and neutral — orchestrators consuming KA
# via the deep-agent ``task(description)`` pattern read it as the answer.
NO_RESULTS_MESSAGE = "No relevant information found for this query."


def _serialize_source(citation: Citation) -> dict[str, Any]:
    """Minimal JSON-safe source dict for the ``ka_sources`` bridge channel.

    Exactly ``title`` / ``url`` / ``source_type`` / ``retriever_name`` — no
    snippet or large text crosses the checkpoint. A missing url normalizes to
    ``""`` so a downstream consumer never carries ``None`` across the channel.
    """
    return {
        "title": citation.title or "",
        "url": citation.url or "",
        "source_type": citation.source_type or "",
        "retriever_name": citation.retriever_name or "",
    }


class OutputNode:
    """Package state into KnowledgeAgentFindings or KnowledgeAgentAnswer.

    Answer mode (SynthesizeNode ran this turn — detected by ``answer_attempt``
    being non-zero) always yields a ``KnowledgeAgentAnswer`` so consumers
    reading ``result.answer`` have a stable shape; on the no-results path the
    canned ``NO_RESULTS_MESSAGE`` becomes the answer text. Evidence mode yields
    a ``KnowledgeAgentFindings``.

    Either way the final answer (or canned no-results text) is also committed
    to the ``messages`` channel so callers consuming KA via the deep-agent
    ``task(description)`` pattern read it as ``result["messages"][-1].content``.

    Handles the early-exit path (no tool calls from plan_queries) by
    surfacing the LLM's direct response in ``result.metadata["direct_response"]``,
    or — when planning failed to produce any usable plan (blank output) —
    flagging ``result.metadata["plan_failed"]`` instead; PlanQueriesNode has
    already substituted a non-empty fallback message on the channel.

    Example:
        ```python
        output_node = OutputNode()
        graph.add_node("output", output_node)
        ```
    """

    async def __call__(
        self,
        state: KnowledgeAgentState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Build the output bundle from state.

        Args:
            state: Current state with findings, coverage, query.
            config: LangGraph runnable config.

        Returns:
            Dict with result (KnowledgeAgentFindings or KnowledgeAgentAnswer)
            and output copies.
        """
        query = state.get("query", "")
        findings = state.get("findings", [])
        coverage = state.get("coverage")
        iteration_count = state.get("iteration_count", 1)
        retrieved = state.get("retrieved_responses", {})

        retriever_names = list(retrieved.keys())
        output_metadata: dict[str, Any] = {}

        # Answer mode is detected by ``answer_attempt`` being non-zero:
        # SynthesizeNode always increments it (to >= 1) whenever it runs, on
        # both its normal and zero-findings short-circuit paths. ResetTurnNode
        # clears it to 0 at the start of each conversation turn, so this is a
        # turn-scoped signal — a checkpointed thread whose *previous* turn ran
        # synthesis does not make the current (e.g. greeting) turn answer mode.
        synthesize_ran = state.get("answer_attempt", 0) > 0
        answer_text = state.get("answer") or ""

        # No-results: at least one retriever tool was invoked (``retrieved``
        # carries a key for it — empty SearchResponses still count as "called"),
        # no Finding survived compression, and no answer text was produced.
        # SynthesizeNode short-circuits with ``answer=""`` on zero findings, so
        # in practice ``not answer_text`` is implied by ``not findings`` in
        # answer mode — the clause guards the degenerate state where an answer
        # somehow exists without findings. Flagged in metadata for structured
        # consumers regardless of mode.
        retrieval_attempted = bool(retrieved)
        no_results = retrieval_attempted and not findings and not answer_text

        if not findings and not retrieved:
            # plan_failed: PlanQueriesNode produced no usable calls and replaced
            # its blank content with a fallback message. Flag it for structured
            # consumers and do NOT mislabel that fallback as a genuine direct
            # response. The fallback text is already the last ``messages`` entry,
            # so the channel carries a non-empty message without re-committing.
            if state.get("plan_failed"):
                output_metadata["plan_failed"] = True
            else:
                direct_response = self._extract_direct_response(state)
                if direct_response:
                    output_metadata["direct_response"] = direct_response

        if no_results:
            output_metadata["no_results"] = True

        kg_subgraph_dict = self._extract_kg_subgraph(retrieved)
        if kg_subgraph_dict:
            output_metadata["kg_subgraph"] = kg_subgraph_dict

        # Degraded compression: at least one chunk group fell back to the
        # deterministic passthrough path because LLM compression exhausted its
        # retries (e.g. provider timeouts). The evidence is preserved but not
        # LLM-synthesized — flag it so consumers can treat the turn accordingly.
        if any(f.compression_mode == PASSTHROUGH_FALLBACK_MODE for f in findings):
            output_metadata["compression_degraded"] = True

        # Synthesis input hit the safety ceiling — evidence was shed/clipped to
        # avoid a context-window hard-fail. Surface it for structured consumers.
        if state.get("synthesis_input_truncated"):
            output_metadata["synthesis_truncated"] = True

        # Synthesis returned empty across every retry — the answer is a canned
        # fallback, not a grounded synthesis. Surface it so a consumer scoring
        # "did synthesis succeed?" can distinguish it from a real answer turn.
        if state.get("synthesis_empty"):
            output_metadata["synthesis_empty"] = True

        # Mirror runtime-query-scope resolution warnings (emitted by the
        # retriever tool factory's MetadataValueResolver path) so the consumer
        # — and the answer-mode synthesizer — can surface them.
        resolution_warnings = state.get("resolution_warnings") or []
        if resolution_warnings:
            output_metadata["warnings"] = list(resolution_warnings)

        evidence = KnowledgeAgentFindings(
            query=query,
            findings=findings,
            coverage=coverage,
            retriever_names=retriever_names,
            iteration_count=iteration_count,
            metadata=output_metadata,
        )

        state_update: dict[str, Any] = {"coverage": coverage}

        if synthesize_ran:
            # Answer mode — always a KnowledgeAgentAnswer so consumers reading
            # ``result.answer`` get a stable shape. On the no-results path
            # (SynthesizeNode short-circuited with ``answer=""``) the canned
            # message becomes the answer text.
            final_answer = answer_text or NO_RESULTS_MESSAGE
            result: KnowledgeAgentFindings | KnowledgeAgentAnswer = KnowledgeAgentAnswer(
                evidence=evidence,
                answer=final_answer,
                answer_citations=state.get("answer_citations", []),
                answer_review=state.get("answer_review"),
            )
            logger.info(
                "OutputNode (answer): %d findings, %d answer_citations, %d retrievers, %d iterations, no_results=%s",
                len(findings),
                len(result.answer_citations),
                len(retriever_names),
                iteration_count,
                no_results,
            )
            # Commit the final answer to the messages channel. Reached only
            # after review_answer (if wired) has settled, so a rejected draft
            # is never surfaced. The AIMessage carries the id SynthesizeNode
            # minted and stamped on its streamed token events, so a consumer
            # that rendered the token stream can match this final message to
            # it and deduplicate. ``answer_message_id`` is ``None`` only when
            # synthesis short-circuited without streaming (no-results) — then
            # ``add_messages`` assigns a fresh id, which is fine since nothing
            # was streamed.
            state_update["messages"] = [AIMessage(content=final_answer, id=state.get("answer_message_id"))]
        else:
            result = evidence
            logger.info(
                "OutputNode (evidence): %d findings, %d retrievers, %d iterations, sufficient=%s, no_results=%s",
                len(findings),
                len(retriever_names),
                iteration_count,
                evidence.is_sufficient,
                no_results,
            )
            # Evidence mode has no synthesized answer; still surface the canned
            # message on no-results so the channel carries a meaningful last
            # message for consumers that inspect it.
            if no_results:
                state_update["messages"] = [AIMessage(content=NO_RESULTS_MESSAGE)]

        state_update["result"] = result
        if kg_subgraph_dict:
            state_update["kg_subgraph"] = kg_subgraph_dict

        # Surface grounding sources back to an orchestrator on the shared
        # ``ka_sources`` bridge channel as minimal JSON-safe dicts. In answer
        # mode the cited subset (``answer_citations``) is the right grain; in
        # evidence mode there is no answer, so collect each finding's citations.
        # Only title / url / source_type / retriever_name cross — no snippet or
        # large text, keeping the checkpoint and the task round-trip lean.
        answer_citations = state.get("answer_citations", []) if synthesize_ran else []
        ka_sources = self._build_ka_sources(answer_citations, findings)
        if ka_sources:
            state_update["ka_sources"] = ka_sources
        return state_update

    @staticmethod
    def _build_ka_sources(
        answer_citations: list[Citation],
        findings: list[Finding],
    ) -> list[dict[str, Any]]:
        """Serialize citations into minimal JSON-safe source dicts.

        In **answer mode** ``answer_citations`` is already the canonical,
        de-duplicated, first-mention-ordered list that the inline ``[N]`` markers
        and the appended ``Sources:`` block are numbered against (see
        ``CitationResolver.resolve`` and ``SynthesizeNode._format_sources_block``).
        It is serialized **1:1, in order, with NO second de-duplication**, so a
        consumer can rely on ``ka_sources[i]`` being exactly citation ``[i+1]`` —
        a reference panel that lines up with the inline markers by position. A
        second dedup here (under a different key than the resolver's) could
        collapse two entries the ``[N]`` numbering kept distinct and silently
        break that alignment.

        In **evidence mode** there are no answer citations and no ``[N]``
        numbering to preserve, so every finding's citations are flattened and
        de-duplicated by ``(url or title, source_type)`` — the same identity the
        resolver uses, so the panel shows one row per distinct source.

        Each dict carries exactly ``title``, ``url``, ``source_type``,
        ``retriever_name`` — no snippet or large text.

        Args:
            answer_citations: Cited subset from an answer-mode synthesis, already
                ordered ``[1], [2], …`` and de-duplicated by the resolver.
            findings: Evidence findings (used when there are no answer citations).

        Returns:
            Ordered list of plain source dicts — 1:1 with ``answer_citations`` in
            answer mode, de-duplicated across findings in evidence mode.
        """
        if answer_citations:
            return [_serialize_source(citation) for citation in answer_citations]

        sources: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for finding in findings:
            for citation in finding.citations:
                key = (citation.url or citation.title or "", citation.source_type or "")
                if key in seen:
                    continue
                seen.add(key)
                sources.append(_serialize_source(citation))
        return sources

    @staticmethod
    def _extract_kg_subgraph(
        retrieved: dict[str, list],
    ) -> dict[str, Any] | None:
        """Extract and merge KG subgraph data from LightRAG responses.

        Returns SubgraphData.to_dict() if any LightRAG responses found, else None.
        """
        lightrag_responses: list[LightRAGSearchResponse] = []
        for responses in retrieved.values():
            for response in responses:
                if isinstance(response, LightRAGSearchResponse):
                    lightrag_responses.append(response)

        if not lightrag_responses:
            return None

        subgraph = from_lightrag_response(lightrag_responses)
        if subgraph.is_empty:
            return None

        logger.info(
            "KG subgraph extracted: %d nodes, %d edges from %d LightRAG responses",
            subgraph.node_count,
            subgraph.edge_count,
            len(lightrag_responses),
        )
        return subgraph.to_dict()

    @staticmethod
    def _extract_direct_response(state: KnowledgeAgentState) -> str:
        """Extract text content from the last AIMessage when no tools were called."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                return msg.content if isinstance(msg.content, str) else str(msg.content)
        return ""

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/nodes/plan_queries.py
----
"""PlanQueriesNode — query planning for the Knowledge Agent.

Produces an ``AIMessage.tool_calls`` list that the downstream ToolNode executes
unchanged. Two strategies, selected by ``PlanConfig.planning_strategy``:

- ``"tool_calls"`` (default): bind the retriever tools to the model and let it
  emit native ``tool_calls`` directly. The bound schema constrains tool names
  and args, so there is no semantic validation round-trip; transient failures
  are retried via ``with_retry``. Needs a model that can emit parallel tool
  calls to fan out N retriever calls per turn.
- ``"structured"``: ask the model for a validated structured plan (with
  conversational validate-and-retry) and convert it into ``tool_calls``.
  Guarantees N calls regardless of the model's parallel-tool-call support.

Features (both strategies):
- Query resolution: prefers latest HumanMessage in ``messages``, falls back to
  ``state.query`` for orchestrator invocations without messages.
- Tool injection: injects available tool names + descriptions into the system
  prompt so smaller LLMs know exactly which tools exist.
- Deterministic query cap: hard-truncates tool_calls to ``max_queries`` after
  the LLM response, regardless of what the LLM generated.

On iteration 1: the LLM sees the user query + tool schemas.
On iteration 2+: the LLM sees compressed findings + coverage gaps + tool schemas.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any, ClassVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...base.nodes import NodeBase
from ...base.utils.output_validation import ModelRetry, OutputValidationError, ainvoke_with_output_validation
from ..knowledge_agent_config import KnowledgeAgentConfig, PlanConfig
from ..knowledge_agent_prompts import (
    PLAN_CALL_FORMAT_STRUCTURED,
    PLAN_CALL_FORMAT_TOOL_CALLS,
    PLAN_QUERIES_REFINEMENT_PROMPT,
    PLAN_QUERIES_SYSTEM_PROMPT,
)
from ..knowledge_agent_state import KnowledgeAgentContext, KnowledgeAgentState
from ..knowledge_agent_types import Finding, KnowledgeNodeTask, RetrieverEntry
from ..utils.findings_format import finding_source_context_line, format_finding_block


logger = logging.getLogger(__name__)

_SUPPORTED_PLANNER_ARGS = frozenset({"query", "apcode", "app_name", "entity"})
_METADATA_PLANNER_ARGS = ("apcode", "app_name", "entity")
_PLAN_VALIDATION_RETRIES = 1

# Surfaced as the planner's message content when planning yields no usable
# retriever calls and no direct response — i.e. the model (after the strategy's
# own retries) produced neither tool calls nor any text. Without this the node
# would emit an empty AIMessage that routes straight to ``output``, leaving the
# consumer with a blank last message. Wording mirrors the input-quality guidance
# in ``KnowledgeAgentInputState``: name the specific entities to retrieve well.
PLAN_FAILED_MESSAGE = (
    "I couldn't determine how to search for this request. Please rephrase it or add "
    "more detail — naming the specific application, component, or identifier involved "
    "helps me retrieve relevant information."
)


class _PlannedRetrieverCall(BaseModel):
    """One planned retriever call emitted by the planner model."""

    tool_name: str = Field(description="Exact retriever tool name to call, e.g. search_elastic_runbooks")
    query: str = Field(description="Focused, self-contained search query")
    apcode: str | None = Field(default=None, description="Optional APCODE argument when the selected tool exposes it")
    app_name: str | None = Field(default=None, description="Optional application-name argument when the selected tool exposes it")
    entity: str | None = Field(default=None, description="Optional entity argument when the selected tool exposes it")


class _PlannedRetrieverCalls(BaseModel):
    """Structured planner output."""

    model_config = ConfigDict(extra="forbid")

    calls: list[_PlannedRetrieverCall] = Field(
        default_factory=list,
        description="Retriever calls to execute. Leave empty only when no available tool can help.",
    )
    direct_response: str | None = Field(
        default=None,
        description="Optional direct reply when the user is greeting, thanking, or asking for clarification rather than requesting retrieval.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_tool_calls(cls, value: Any) -> Any:
        """Accept legacy/native-tool-shaped ``tool_calls`` as planner calls.

        Some models continue to emit a top-level ``tool_calls`` object even
        when asked for structured output. Pydantic would otherwise ignore or
        reject that field, producing ``calls=[]`` and making routing exit to
        ``output``. Normalize that common shape into the explicit ``calls``
        schema before validation so downstream ToolNode routing still sees
        real ``AIMessage.tool_calls``.
        """
        if not isinstance(value, dict) or "tool_calls" not in value:
            return value

        data = dict(value)
        raw_tool_calls = data.pop("tool_calls")
        if "calls" not in data:
            data["calls"] = cls._planner_calls_from_tool_calls(raw_tool_calls)
        return data

    @staticmethod
    def _planner_calls_from_tool_calls(raw_tool_calls: Any) -> Any:
        if not isinstance(raw_tool_calls, list):
            return raw_tool_calls

        calls: list[Any] = []
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                calls.append(raw_call)
                continue
            raw_args = raw_call.get("args")
            args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            call: dict[str, Any] = {
                "tool_name": raw_call.get("tool_name") or raw_call.get("name"),
                "query": args.get("query") or raw_call.get("query") or "",
            }
            for axis in _METADATA_PLANNER_ARGS:
                value = args.get(axis) if axis in args else raw_call.get(axis)
                if value is not None:
                    call[axis] = value
            calls.append(call)
        return calls


class PlanQueriesNode(NodeBase[KnowledgeAgentContext]):
    """Produce ToolNode-compatible retriever calls from a structured plan.

    The node asks the model for structured ``tool_name``/``query`` records,
    validates them against the actual generated retriever tool names and
    exposed arguments, then returns an ``AIMessage`` containing ``tool_calls``
    that the ToolNode executes.

    Key behaviors:
    - Resolves query from latest HumanMessage in state.messages first, then
      falls back to state.query (for orchestrator use without messages).
    - Injects available tool names + descriptions into system prompt (XML tags).
    - Hard-truncates tool_calls to ``plan_config.max_queries`` after LLM response.

    Example:
        ```python
        plan_node = PlanQueriesNode(
            tools=retriever_tools,
            default_model=llm,
            agent_config=config,
        )
        graph.add_node("plan_queries", plan_node)
        ```
    """

    task: ClassVar[str] = KnowledgeNodeTask.PLANNING

    def __init__(
        self,
        tools: list[BaseTool],
        entries: list[RetrieverEntry] | None = None,
        default_model: BaseChatModel | None = None,
        agent_config: KnowledgeAgentConfig | None = None,
    ) -> None:
        super().__init__(default_model=default_model, node_config=agent_config)
        self._tools = tools
        self._entries = entries  # When set, used for per-retriever examples in prompt
        self._agent_config = agent_config or KnowledgeAgentConfig()
        self._tool_args_by_name = self._build_tool_args_by_name(tools)

    @property
    def plan_config(self) -> PlanConfig:
        return self._agent_config.plan

    async def __call__(
        self,
        state: KnowledgeAgentState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Invoke the planner LLM to produce retriever tool_calls.

        Dispatches on ``PlanConfig.planning_strategy`` (native ``"tool_calls"``
        binding or validated ``"structured"`` output). Resolves query from the
        latest HumanMessage in messages (falls back to state.query for
        orchestrator invocations without messages). Builds an XML-structured
        system prompt with available tools and hard-truncates tool_calls to
        max_queries.

        Args:
            state: Current workflow state.
            config: LangGraph runnable config.

        Returns:
            Dict with messages (AIMessage containing tool_calls),
            resolved query, and iteration_count increment.
        """
        # Resolve query — prefer latest HumanMessage so that new invocations
        # on the same thread (checkpointer) pick up the fresh query instead of
        # the stale `query` field written by a previous run.
        query = self._extract_query_from_messages(state.get("messages", []))
        if not query:
            query = state.get("query", "")
        if not query:
            logger.error("PlanQueriesNode: no query found in state or messages")
            raise ValueError("No query provided and no HumanMessage found in messages")

        iteration = state.get("iteration_count", 0)
        findings = state.get("findings", [])
        coverage = state.get("coverage")

        # Build messages with XML-structured system prompt
        messages = self._build_messages(query, iteration, findings, coverage)

        if self.plan_config.planning_strategy == "tool_calls":
            response = await self._plan_via_tool_calls(messages, config)
        else:
            response = await self._plan_via_structured_output(messages, config)

        max_queries = self.plan_config.max_queries
        if iteration == 0 and self.plan_config.include_original_query and response.tool_calls:
            self._inject_anchor_queries(response, query)

        # Deterministic query cap — hard-truncate tool_calls (after injection)
        n_calls = len(response.tool_calls) if response.tool_calls else 0
        if response.tool_calls and n_calls > max_queries:
            logger.warning(
                "plan_queries: truncating %d tool calls to max_queries=%d",
                n_calls,
                max_queries,
            )
            response.tool_calls = response.tool_calls[:max_queries]
            n_calls = max_queries

        # Planning-failure fallback: no usable calls AND no text. The strategy's
        # own retries (with_retry / validation round-trip) are already spent by
        # here, so a blank outcome is terminal — substitute a non-empty message
        # and flag it so OutputNode surfaces it (and does not mistake it for a
        # genuine direct response). A no-call turn WITH content (greeting /
        # clarification) is left untouched.
        content_text = response.content if isinstance(response.content, str) else str(response.content or "")
        plan_failed = not response.tool_calls and not content_text.strip()
        if plan_failed:
            logger.warning("plan_queries: no usable tool calls and empty content — emitting plan-failure fallback")
            response.content = PLAN_FAILED_MESSAGE

        logger.info(
            "plan_queries (iteration %d): %d tool calls for query '%s'",
            iteration + 1,
            n_calls,
            query[:80],
        )

        return {
            "messages": [response],
            "query": query,
            "plan_failed": plan_failed,
            "iteration_count": 1,
            "compressed_chunk_hashes": set(),  # Reset per outer iteration (Decision 36)
            # processed_kg_hashes intentionally NOT reset: KG reformatting is
            # deterministic so re-processing produces identical findings.
            # retrieved_responses accumulates (append reducer), so without
            # persistent hashes, old relationships would create duplicate Findings.
            "expansion_rounds": 0,  # Phase 2b: reset inner loop counter
            "coverage": None,  # Phase 2b: clear stale fetch_targets (2b-D2)
        }

    # ------------------------------------------------------------------
    # Planning strategies (tool_calls | structured)
    # ------------------------------------------------------------------

    async def _plan_via_tool_calls(
        self,
        messages: list[SystemMessage | HumanMessage],
        config: RunnableConfig,
    ) -> AIMessage:
        """Plan by binding retriever tools and letting the model emit native tool_calls.

        The bound schema constrains tool names and argument shape, so no
        semantic validation round-trip is needed — invalid-by-construction
        calls cannot occur. Transient model/provider failures are retried via
        ``with_retry``. Calls with an unknown name or empty ``query`` are dropped
        defensively, and args are filtered to each tool's exposed set.

        Needs a model that can emit parallel tool calls to fan out N retriever
        calls per turn; models that cannot (e.g. gpt-oss) degrade to fewer calls
        — use ``planning_strategy="structured"`` there.
        """
        attempts = max(1, self.plan_config.tool_call_retry_attempts)
        model_with_tools = self.model.bind_tools(self._tools).with_retry(stop_after_attempt=attempts)
        raw = await model_with_tools.ainvoke(messages, config=config)
        response = raw if isinstance(raw, AIMessage) else AIMessage(content=str(getattr(raw, "content", "")))

        sanitized = self._sanitize_tool_calls(response.tool_calls or [])
        response.tool_calls = sanitized  # type: ignore[assignment]
        # When calls exist the content is irrelevant to routing; on a no-call
        # turn the content is the direct response / clarification, kept as-is.
        if sanitized:
            response.content = ""
        return response

    def _sanitize_tool_calls(self, tool_calls: Sequence[Any]) -> list[dict[str, Any]]:
        """Drop calls with unknown tool name or empty query; filter args to the exposed set."""
        sanitized: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name", "")
            exposed = self._tool_args_by_name.get(name)
            if exposed is None:
                logger.warning("plan_queries: dropping tool_call for unknown tool %r", name)
                continue
            raw_args = call.get("args")
            args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            query = (args.get("query") or "").strip()
            if not query:
                logger.warning("plan_queries: dropping tool_call for %r with empty query", name)
                continue
            clean_args: dict[str, Any] = {"query": query}
            for axis in _METADATA_PLANNER_ARGS:
                value = args.get(axis)
                if value is not None and str(value).strip() and axis in exposed:
                    clean_args[axis] = str(value).strip()
            sanitized.append(
                {
                    "id": call.get("id") or f"plan_{uuid.uuid4().hex[:8]}",
                    "name": name,
                    "args": clean_args,
                    "type": "tool_call",
                }
            )
        return sanitized

    async def _plan_via_structured_output(
        self,
        messages: list[SystemMessage | HumanMessage],
        config: RunnableConfig,
    ) -> AIMessage:
        """Plan via validated structured output, converted to tool_calls.

        Asks the model for a structured ``_PlannedRetrieverCalls`` plan with a
        conversational validate-and-retry, then converts the validated plan into
        an ``AIMessage.tool_calls`` list. Guarantees N calls regardless of the
        model's parallel-tool-call support, at the cost of a validation
        round-trip.
        """
        validation_ctx: dict[str, Any] = {
            "tool_args_by_name": self._tool_args_by_name,
        }
        try:
            plan = cast(
                _PlannedRetrieverCalls,
                await ainvoke_with_output_validation(
                    model=self.model,
                    output_type=_PlannedRetrieverCalls,
                    messages=messages,
                    output_validators=[self._validate_planned_calls],
                    validation_context=validation_ctx,
                    max_retries=_PLAN_VALIDATION_RETRIES,
                    config=config,
                ),
            )
        except OutputValidationError:
            logger.warning("PlanQueriesNode: retries exhausted, filtering invalid planned retriever calls")
            plan = self._filter_valid_planned_calls(validation_ctx.get("_last_plan"))

        return self._plan_to_ai_message(plan)

    def _inject_anchor_queries(self, response: AIMessage, query: str) -> None:
        """Append one tool call per selected retriever with the original user query.

        Mutates response.tool_calls in place. Called only on iteration 0 when
        include_original_query is True so the user's exact phrasing reaches retrievers.

        Skips injection for a tool if the LLM already generated a call with
        a query matching the user query (case-insensitive, stripped), avoiding
        duplicate retrieval.
        """
        normalized_query = query.strip().lower()

        existing_queries_by_tool: dict[str, set[str]] = {}
        for tc in response.tool_calls:
            tool_name = tc.get("name") or getattr(tc, "name", "")
            if not tool_name:
                continue
            args = tc.get("args") or getattr(tc, "args", {}) or {}
            q = (args.get("query") or "").strip().lower()
            existing_queries_by_tool.setdefault(tool_name, set()).add(q)

        anchor_calls = []
        for tool_name, queries in existing_queries_by_tool.items():
            if normalized_query in queries:
                logger.debug(
                    "plan_queries: skipping anchor for '%s' — LLM already generated identical query",
                    tool_name,
                )
                continue
            anchor_calls.append(
                {
                    "id": f"anchor_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "args": {"query": query},
                    "type": "tool_call",
                }
            )

        if anchor_calls:
            response.tool_calls = list(response.tool_calls) + anchor_calls  # type: ignore[assignment]
            logger.info(
                "plan_queries: injected %d anchor query tool calls (original user query)",
                len(anchor_calls),
            )

    # ------------------------------------------------------------------
    # Structured plan validation / conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tool_args_by_name(tools: list[BaseTool]) -> dict[str, set[str]]:
        """Return the LLM-visible argument names for each generated tool."""
        tool_args: dict[str, set[str]] = {}
        for tool in tools:
            args = getattr(tool, "args", None) or {}
            tool_args[tool.name] = set(args) & _SUPPORTED_PLANNER_ARGS
        return tool_args

    @staticmethod
    def _validate_planned_calls(plan: _PlannedRetrieverCalls, ctx: dict[str, Any]) -> _PlannedRetrieverCalls:
        """Validate planned calls against generated tool names and args.

        The validator is intentionally strict during retry: any invalid call
        triggers model feedback. If retries exhaust, the node hard-filters to
        valid calls so one bad call does not poison the whole plan.
        """
        if not plan.calls:
            return plan

        ctx["_last_plan"] = plan
        tool_args_by_name: dict[str, set[str]] = ctx.get("tool_args_by_name", {})
        errors: list[str] = []
        for index, call in enumerate(plan.calls, start=1):
            errors.extend(PlanQueriesNode._validation_errors_for_call(index, call, tool_args_by_name))

        if errors:
            available = ", ".join(sorted(tool_args_by_name)) or "none"
            raise ModelRetry(
                "Invalid retriever call plan:\n- " + "\n- ".join(errors) + f"\n\nUse only these tool names: {available}. "
                "Use only args exposed by that tool; the supported arg surface is query plus optional apcode, app_name, entity."
            )
        return plan

    @staticmethod
    def _validation_errors_for_call(
        index: int,
        call: _PlannedRetrieverCall,
        tool_args_by_name: dict[str, set[str]],
    ) -> list[str]:
        errors: list[str] = []
        tool_name = call.tool_name.strip()
        if tool_name not in tool_args_by_name:
            return [f"call {index}: unknown tool_name {call.tool_name!r}"]

        exposed_args = tool_args_by_name[tool_name]
        if "query" not in exposed_args:
            errors.append(f"call {index}: tool {tool_name!r} does not expose required arg 'query'")
        if not call.query.strip():
            errors.append(f"call {index}: query must be a non-empty string")

        for axis in _METADATA_PLANNER_ARGS:
            value = getattr(call, axis)
            if value is not None and value.strip() and axis not in exposed_args:
                errors.append(f"call {index}: tool {tool_name!r} does not expose arg {axis!r}")
        return errors

    def _filter_valid_planned_calls(self, plan: Any) -> _PlannedRetrieverCalls:
        """Drop invalid calls after retry exhaustion."""
        if not isinstance(plan, _PlannedRetrieverCalls):
            return _PlannedRetrieverCalls(calls=[])
        valid = [
            call
            for call in plan.calls
            if not self._validation_errors_for_call(
                0,
                call,
                self._tool_args_by_name,
            )
        ]
        return _PlannedRetrieverCalls(calls=valid)

    @staticmethod
    def _call_args(call: _PlannedRetrieverCall, exposed_args: set[str]) -> dict[str, str]:
        args: dict[str, str] = {"query": call.query.strip()}
        for axis in _METADATA_PLANNER_ARGS:
            value = getattr(call, axis)
            if value is not None and value.strip() and axis in exposed_args:
                args[axis] = value.strip()
        return args

    def _plan_to_ai_message(self, plan: _PlannedRetrieverCalls) -> AIMessage:
        """Convert a validated structured plan into ToolNode input."""
        tool_calls = []
        for call in plan.calls:
            tool_name = call.tool_name.strip()
            exposed_args = self._tool_args_by_name.get(tool_name, set())
            tool_calls.append(
                {
                    "id": f"plan_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "args": self._call_args(call, exposed_args),
                    "type": "tool_call",
                }
            )
        content = "" if tool_calls else (plan.direct_response or "")
        return AIMessage(content=content, tool_calls=tool_calls)

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        query: str,
        iteration: int,
        findings: list[Finding],
        coverage: Any | None,
    ) -> list[SystemMessage | HumanMessage]:
        """Build the message list for the LLM.

        System prompt includes XML-tagged available tools and max_queries constraint.
        Iteration 1: system prompt + user query.
        Iteration 2+: system prompt + refinement context + user query.
        """
        # Build system prompt with injected tools and constraints
        tools_block = self._build_tools_block()
        max_queries = self.plan_config.max_queries
        call_format_constraint = PLAN_CALL_FORMAT_TOOL_CALLS if self.plan_config.planning_strategy == "tool_calls" else PLAN_CALL_FORMAT_STRUCTURED
        system_content = PLAN_QUERIES_SYSTEM_PROMPT.format(
            tools_block=tools_block,
            max_queries=max_queries,
            call_format_constraint=call_format_constraint,
        )

        messages: list[SystemMessage | HumanMessage] = [SystemMessage(content=system_content)]

        if iteration > 0 and (findings or coverage is not None):
            findings_summary = self._format_findings(findings)
            gaps = self._format_gaps(coverage)
            query_suggestions = self._format_query_suggestions(coverage)
            refinement_prompt = PLAN_QUERIES_REFINEMENT_PROMPT.format(
                findings_summary=findings_summary,
                gaps=gaps,
                query_suggestions=query_suggestions,
            )
            messages.append(SystemMessage(content=refinement_prompt))

        messages.append(HumanMessage(content=query))
        return messages

    def _build_tools_block(self) -> str:
        """Format available tools as XML for the system prompt.

        When entries are provided, each tool includes name, description, and
        optional <examples> (sample queries). Otherwise name and description only.
        """
        if self._entries:
            parts = []
            for entry in self._entries:
                tool_name = f"search_{entry.name}"
                block = f'<tool name="{tool_name}">{entry.description}'
                args = self._format_tool_args(tool_name)
                if args:
                    block += f"\n  <args>{args}</args>"
                if entry.examples:
                    examples_lines = "\n".join(f'- "{q}"' for q in entry.examples[:5])
                    block += f"\n  <examples>\n  {examples_lines}\n  </examples>"
                block += "</tool>"
                parts.append(block)
            return "\n".join(parts)
        parts = []
        for tool in self._tools:
            args = self._format_tool_args(tool.name)
            args_block = f"\n  <args>{args}</args>" if args else ""
            parts.append(f'<tool name="{tool.name}">{tool.description}{args_block}</tool>')
        return "\n".join(parts)

    def _format_tool_args(self, tool_name: str) -> str:
        args = self._tool_args_by_name.get(tool_name, set())
        if not args:
            return ""
        ordered = ["query", *[axis for axis in _METADATA_PLANNER_ARGS if axis in args]]
        return ", ".join(arg for arg in ordered if arg in args)

    # ------------------------------------------------------------------
    # Query resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_query_from_messages(messages: Sequence[AnyMessage]) -> str:
        """Extract query from the last HumanMessage in the message list.

        Walks messages in reverse to find the most recent HumanMessage.
        Returns empty string if no HumanMessage is found.
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                content = msg.content
                return content if isinstance(content, str) else str(content)
        return ""

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_findings(findings: list[Finding]) -> str:
        """Format findings into a readable summary for the refinement prompt."""
        if not findings:
            return "No findings from previous iteration."

        parts = [
            format_finding_block(
                i,
                f.topic,
                f.summary,
                f.key_facts,
                # Surface page identity + context so the planner can re-query
                # when a previous iteration's evidence came from a generic page
                # about a different entity than the question asks about.
                source_context_line=finding_source_context_line(f.citations, max_summary_chars=160) if f.citations else None,
            )
            for i, f in enumerate(findings, 1)
        ]
        return "\n\n".join(parts)

    @staticmethod
    def _format_gaps(coverage: Any | None) -> str:
        """Format coverage gaps for the refinement prompt."""
        if coverage is None:
            return "No coverage assessment available."
        gaps = getattr(coverage, "gaps", [])
        if not gaps:
            return "No specific gaps identified."
        return "\n".join(f"- {gap}" for gap in gaps)

    @staticmethod
    def _format_query_suggestions(coverage: Any | None) -> str:
        """Format query suggestions from the review assessment."""
        if coverage is None:
            return "No query suggestions available."
        suggestions = getattr(coverage, "query_suggestions", [])
        if not suggestions:
            return "No specific query suggestions."
        return "\n".join(f"- {s}" for s in suggestions)

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/nodes/review_answer.py
----
"""ReviewAnswerNode — faithfulness check on synthesized answers.

Citation coherence is guaranteed by CitationResolver — this node checks
only whether claims in the answer match the cited evidence (faithfulness).

Uses LLM structured output → AnswerReview.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.state import RunnableConfig
from pydantic import BaseModel, Field

from ...base.nodes import NodeBase
from ..knowledge_agent_config import KnowledgeAgentConfig
from ..knowledge_agent_prompts import REVIEW_ANSWER_HUMAN_PROMPT, REVIEW_ANSWER_SYSTEM_PROMPT
from ..knowledge_agent_state import KnowledgeAgentContext, KnowledgeAgentState
from ..knowledge_agent_types import AnswerReview, Finding, KnowledgeNodeTask
from ..utils.findings_format import cited_documents_line, finding_source_context_line, format_finding_block


logger = logging.getLogger(__name__)


class _AnswerReviewOutput(BaseModel):
    """LLM structured output for answer faithfulness review."""

    faithful: bool = Field(description="True if all claims in the answer are supported by the evidence")
    explanation: str = Field(description="Brief explanation of the faithfulness assessment")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="List of specific claims from the answer that lack evidence support",
    )


class ReviewAnswerNode(NodeBase[KnowledgeAgentContext]):
    """Check faithfulness of synthesized answer against evidence.

    Attributes:
        task: VERIFICATION — resolves to verification model config.
    """

    task: ClassVar[str] = KnowledgeNodeTask.VERIFICATION

    def __init__(
        self,
        *,
        default_model: Any,
        agent_config: KnowledgeAgentConfig,
    ) -> None:
        super().__init__(default_model=default_model, node_config=agent_config)

    async def __call__(
        self,
        state: KnowledgeAgentState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Review the synthesized answer for faithfulness.

        Args:
            state: Must contain ``answer`` and ``findings``.
            config: LangGraph runnable config.

        Returns:
            Dict with ``answer_review: AnswerReview``.
        """
        answer = state.get("answer", "")
        findings: list[Finding] = state.get("findings", [])

        if not answer:
            logger.warning("ReviewAnswerNode: empty answer — marking as faithful (vacuously)")
            return {"answer_review": AnswerReview(faithful=True, explanation="Empty answer — nothing to review.")}

        findings_summary = self._format_findings_for_review(findings)

        model = self._resolve_model_for_task(self.task)
        structured_model = model.with_structured_output(_AnswerReviewOutput)

        messages = [
            SystemMessage(content=REVIEW_ANSWER_SYSTEM_PROMPT),
            HumanMessage(
                content=REVIEW_ANSWER_HUMAN_PROMPT.format(
                    query=state.get("query", ""),
                    answer=answer,
                    findings_summary=findings_summary,
                )
            ),
        ]

        review_output: _AnswerReviewOutput = cast(_AnswerReviewOutput, await structured_model.ainvoke(messages))

        review = AnswerReview(
            faithful=review_output.faithful,
            explanation=review_output.explanation,
            unsupported_claims=review_output.unsupported_claims,
        )

        logger.info(
            "ReviewAnswerNode: faithful=%s, unsupported_claims=%d",
            review.faithful,
            len(review.unsupported_claims),
        )

        return {"answer_review": review}

    @staticmethod
    def _format_findings_for_review(findings: list[Finding]) -> str:
        """Format findings for the review prompt using shared formatter."""
        blocks: list[str] = []
        for i, finding in enumerate(findings, 1):
            block = format_finding_block(
                index=i,
                topic=finding.topic,
                summary=finding.summary,
                key_facts=finding.key_facts,
                sources_line=f"Sources: {', '.join(finding.retriever_sources)}" if finding.retriever_sources else None,
                citations_line=cited_documents_line(finding.citations) if finding.citations else None,
                # Surface per-page identity + context so the faithfulness check
                # can flag a claim that attributes a generic page's content to an
                # entity the page's context shows it does not belong to. 320-char
                # cap matches synthesis so the reviewer sees the same provenance.
                source_context_line=finding_source_context_line(finding.citations, max_summary_chars=320) if finding.citations else None,
            )
            blocks.append(block)
        return "\n\n".join(blocks)

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/nodes/review_evidence.py
----
"""ReviewEvidenceNode — assess evidence coverage and decide whether to iterate.

Consumes accumulated findings and produces a CoverageAssessment via LLM
structured output. The assessment drives the routing decision:
- sufficient → output (stop)
- insufficient + iteration budget → plan_queries (outer loop)
- budget exhausted → output (stop with best-effort)

Token budget for prompt truncation comes from the unified
``KnowledgeAgentConfig.max_review_tokens`` (derived from
``evidence_token_budget``).  When findings exceed that budget, they are
sorted by confidence (high > medium > low) and truncated with a summary note.

fetch_target validation: Each FetchTarget.target_id is validated against IDs
extracted from the findings' citations (pageId, full_doc_id, chunk_id —
excluding the ``doc`` file path which is not a DocumentProvider identifier).
Uses ``ainvoke_with_output_validation`` for retry with conversational feedback.
On retry exhaustion, invalid targets are hard-filtered and query_suggestions
are backfilled from the filtered targets' reasons.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.state import RunnableConfig

from sta_agent_core.repositories.retrievers.document_provider import DocumentProvider

from ...base.nodes import NodeBase
from ...base.utils.output_validation import (
    ModelRetry,
    OutputValidationError,
    ainvoke_with_output_validation,
)
from ..knowledge_agent_config import KnowledgeAgentConfig, ReviewConfig
from ..knowledge_agent_prompts import (
    REVIEW_AUTOPULL_ACTIVE,
    REVIEW_EVIDENCE_HUMAN_PROMPT,
    REVIEW_EVIDENCE_SYSTEM_PROMPT,
    REVIEW_EXPANSION_BUDGET_EXHAUSTED,
)
from ..knowledge_agent_state import KnowledgeAgentContext, KnowledgeAgentState
from ..knowledge_agent_types import (
    CoverageAssessment,
    Finding,
    KnowledgeNodeTask,
    RetrieverEntry,
)
from ..utils.findings_format import cited_documents_line, finding_source_context_line, format_finding_block


logger = logging.getLogger(__name__)

_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}

# DocumentProvider-compatible metadata keys.
# Excludes "doc" (file path) — not a valid ID for get_document() or get_chunk_context().
_VALID_ID_KEYS = ("pageId", "page_id", "full_doc_id", "chunk_id")


# ---------------------------------------------------------------------------
# fetch_target validation helpers
# ---------------------------------------------------------------------------


def _extract_valid_ids(
    findings: list[Finding],
    non_expandable_retrievers: frozenset[str] = frozenset(),
) -> set[str]:
    """Collect all valid DocumentProvider IDs from citation metadata.

    Extracts pageId, page_id, full_doc_id, chunk_id — excludes "doc" (file
    path) which is not a valid identifier for get_document/get_chunk_context.

    A fetch_target is only actionable if its source retriever implements
    ``DocumentProvider``. IDs from a citation whose source retrievers are *all*
    known non-DocumentProvider retrievers are therefore dropped — offering them
    to the reviewer would only burn an expansion round on a retriever
    ``ExpandNode`` cannot fetch from. A citation with an empty/unknown
    ``retriever_name`` is kept (it cannot be proven non-expandable).

    ``retriever_name`` may be a comma-separated list when a citation was
    deduplicated across retrievers (see ``CitationResolver``); the ID stays
    reachable as long as at least one of those retrievers is expandable.
    """
    ids: set[str] = set()
    for finding in findings:
        for citation in finding.citations:
            names = [n.strip() for n in (citation.retriever_name or "").split(",") if n.strip()]
            if names and all(n in non_expandable_retrievers for n in names):
                continue
            meta = citation.metadata or {}
            for key in _VALID_ID_KEYS:
                val = meta.get(key)
                if val is not None and val != "":
                    ids.add(str(val))
    return ids


def _compute_non_expandable_retrievers(entries: list[RetrieverEntry]) -> frozenset[str]:
    """Return the names of entries whose retriever does NOT implement DocumentProvider.

    These retrievers cannot serve fetch_targets — ``ExpandNode`` skips them.
    """
    return frozenset(e.name for e in entries if not isinstance(e.retriever, DocumentProvider))


def _format_valid_ids(ids: set[str]) -> str:
    """Format valid IDs as a prompt constraint block."""
    if not ids:
        return "No fetch target IDs available — use query_suggestions instead."
    return "Available target IDs (use ONLY these in fetch_targets):\n  " + ", ".join(sorted(ids))


def _validate_fetch_target_ids(
    assessment: CoverageAssessment,
    ctx: dict[str, Any],
) -> CoverageAssessment:
    """Validator for ``ainvoke_with_output_validation``.

    Raises ``ModelRetry`` if any fetch_target uses a hallucinated target_id.
    Stores the assessment in ctx for hard-filter fallback on exhaustion.
    """
    valid_ids: set[str] = ctx.get("valid_ids", set())
    if not assessment.fetch_targets or not valid_ids:
        return assessment

    invalid = [t for t in assessment.fetch_targets if t.target_id not in valid_ids]
    if not invalid:
        return assessment

    # Store for _hard_filter_assessment fallback
    ctx["_last_assessment"] = assessment

    raise ModelRetry(
        f"Invalid target_ids: {', '.join(t.target_id for t in invalid)}. "
        "These IDs do NOT exist in the findings' citations.\n\n"
        f"{_format_valid_ids(valid_ids)}\n\n"
        "Use ONLY target_ids from the list above, or use query_suggestions instead."
    )


def _hard_filter_assessment(
    ctx: dict[str, Any],
    valid_ids: set[str],
) -> CoverageAssessment:
    """Fallback after retries exhausted — filter invalid, backfill suggestions."""
    last: CoverageAssessment | None = ctx.get("_last_assessment")
    if not last:
        return CoverageAssessment(
            sufficient=False,
            gaps=["Output validation failed"],
            reasoning="Structured output validation exhausted all retries.",
        )

    valid = [t for t in last.fetch_targets if t.target_id in valid_ids]
    invalid = [t for t in last.fetch_targets if t.target_id not in valid_ids]
    updates: dict[str, Any] = {"fetch_targets": valid}

    # Backfill: preserve reviewer intent when all targets are invalid
    if not valid and not last.query_suggestions and invalid:
        updates["query_suggestions"] = [t.reason for t in invalid if t.reason]

    return last.model_copy(update=updates)


# ---------------------------------------------------------------------------
# Narration helpers (module-level so the wording is greppable from tests)
# ---------------------------------------------------------------------------

# Loop-internal — emitted by ``ReviewEvidenceNode`` on every iteration that
# produces no findings. Contrast with the consumer-facing
# ``output.NO_RESULTS_MESSAGE`` which is the final answer surrogate emitted
# once when the whole run ends with no evidence.
_NO_EVIDENCE_MESSAGE = "No relevant evidence retrieved in this iteration."
_MAX_NARRATED_GAPS = 3


def _narrate_assessment(assessment: CoverageAssessment, findings: list[Finding]) -> str:
    """Render a one-line narration of the review verdict for the messages channel.

    ``coverage`` drives routing; this narration is observability on top of the
    structured field so the agent loop reads as a conversational trace. The
    gap list is truncated to the first three; when more remain the narration
    appends ``(and N more)`` so a reader can tell the list was elided.
    """
    if assessment.sufficient:
        return f"Coverage assessment: sufficient ({len(findings)} findings)."
    gaps = assessment.gaps[:_MAX_NARRATED_GAPS]
    if not gaps:
        return "Coverage gaps identified."
    elided = len(assessment.gaps) - len(gaps)
    suffix = f" (and {elided} more)" if elided > 0 else ""
    return f"Coverage gaps identified: {'; '.join(gaps)}{suffix}."


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class ReviewEvidenceNode(NodeBase[KnowledgeAgentContext]):
    """Assess evidence coverage and decide whether to iterate.

    Uses LLM structured output to produce a CoverageAssessment from
    accumulated findings. Validates fetch_targets against citation IDs
    via ``ainvoke_with_output_validation`` (same utility as CompressNode).

    Example:
        ```python
        review_node = ReviewEvidenceNode(default_model=llm, agent_config=config)
        graph.add_node("review_evidence", review_node)
        ```
    """

    task: ClassVar[str] = KnowledgeNodeTask.REVIEW

    def __init__(
        self,
        default_model: BaseChatModel | None = None,
        agent_config: KnowledgeAgentConfig | None = None,
        entries: list[RetrieverEntry] | None = None,
    ) -> None:
        super().__init__(default_model=default_model, node_config=agent_config)
        self._agent_config = agent_config or KnowledgeAgentConfig()
        # Retrievers that cannot serve fetch_targets (no DocumentProvider).
        # Used to gate the valid-ID set offered to the reviewer.
        self._non_expandable_retrievers: frozenset[str] = _compute_non_expandable_retrievers(entries or [])

    @property
    def review_config(self) -> ReviewConfig:
        return self._agent_config.review

    async def __call__(
        self,
        state: KnowledgeAgentState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        query = state.get("query", "")
        findings = state.get("findings", [])

        if not findings:
            logger.info("ReviewEvidenceNode: zero findings — returning insufficient")
            return {
                "coverage": CoverageAssessment(
                    sufficient=False,
                    gaps=["No evidence found"],
                    reasoning="No findings were produced from the retrieval and compression pipeline.",
                    query_suggestions=[query] if query else [],
                ),
                "messages": [AIMessage(content=_NO_EVIDENCE_MESSAGE)],
            }

        findings_summary = self._build_findings_summary(findings)
        valid_ids = _extract_valid_ids(findings, self._non_expandable_retrievers)

        human_content = REVIEW_EVIDENCE_HUMAN_PROMPT.format(
            query=query,
            findings_summary=findings_summary,
        )

        expansion_budget_spent = (
            self._agent_config.expand.enabled and state.get("expansion_rounds", 0) >= self._agent_config.expand.max_expansion_rounds
        )
        auto_pull_active = self._agent_config.expand.auto_pull_document

        if expansion_budget_spent:
            human_content += REVIEW_EXPANSION_BUDGET_EXHAUSTED
        if auto_pull_active:
            human_content += REVIEW_AUTOPULL_ACTIVE

        # Validation only when fetch_targets are actionable
        use_validation = bool(valid_ids) and not expansion_budget_spent and not auto_pull_active

        if use_validation:
            human_content += "\n\n<available_targets>\n" + _format_valid_ids(valid_ids) + "\n</available_targets>"

        messages = [
            SystemMessage(content=REVIEW_EVIDENCE_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        validation_ctx: dict[str, Any] = {"valid_ids": valid_ids}
        validators = [_validate_fetch_target_ids] if use_validation else []
        try:
            assessment = cast(
                CoverageAssessment,
                await ainvoke_with_output_validation(
                    model=self.model,
                    output_type=CoverageAssessment,
                    messages=messages,
                    output_validators=validators,
                    validation_context=validation_ctx,
                    config=config,
                ),
            )
        except OutputValidationError:
            if use_validation:
                logger.warning("ReviewEvidenceNode: retries exhausted, hard-filtering invalid fetch_targets")
            else:
                logger.warning("ReviewEvidenceNode: structured output retries exhausted")
            assessment = _hard_filter_assessment(validation_ctx, valid_ids)

        logger.info(
            "ReviewEvidenceNode: sufficient=%s, %d gaps, %d suggestions, %d fetch_targets",
            assessment.sufficient,
            len(assessment.gaps),
            len(assessment.query_suggestions),
            len(assessment.fetch_targets),
        )
        return {
            "coverage": assessment,
            "messages": [AIMessage(content=_narrate_assessment(assessment, findings))],
        }

    def _build_findings_summary(self, findings: list[Finding]) -> str:
        """Build findings summary with confidence-sorted, token-budget truncation.

        Sorts all findings by confidence (high first), renders each block,
        and accumulates until the approximate token budget is reached.
        Remaining findings are omitted with a summary note.

        Token budget comes from the unified ``evidence_token_budget`` via
        ``KnowledgeAgentConfig.max_review_tokens``.  Falls back to
        count-based cap (max_findings_for_review) when token budget is 0.
        """
        max_tokens = self._agent_config.max_review_tokens
        max_count = self.review_config.max_findings_for_review
        total = len(findings)

        sorted_findings = sorted(
            findings,
            key=lambda f: _CONFIDENCE_ORDER.get(f.confidence, 3),
        )

        parts: list[str] = []
        token_count = 0
        included = 0

        for i, f in enumerate(sorted_findings, 1):
            block = format_finding_block(
                i,
                f.topic,
                f.summary,
                f.key_facts,
                expansion_marker=" [NEEDS EXPANSION]" if f.needs_expansion else "",
                sources_line=f"Sources: {', '.join(f.retriever_sources) if f.retriever_sources else 'unknown'}",
                citations_line=cited_documents_line(f.citations) if f.citations else None,
                # 320-char cap: the reviewer is a provenance gate, so it must see
                # the same source context synthesis does (citation_resolver uses
                # 320). Kept in lock-step with estimate_rendered_chars so the
                # review token budget below matches what is rendered.
                source_context_line=finding_source_context_line(f.citations, max_summary_chars=320) if f.citations else None,
            )

            block_len = len(block)
            block_tokens = block_len // 4
            would_exceed_tokens = max_tokens > 0 and (token_count + block_tokens) > max_tokens and included > 0
            would_exceed_count = max_count > 0 and included >= max_count

            if would_exceed_tokens or would_exceed_count:
                break

            parts.append(block)
            token_count += block_tokens
            included += 1

        summary = "\n\n".join(parts)

        if included < total:
            omitted = total - included
            logger.warning(
                "ReviewEvidenceNode: truncated findings for review prompt — showing %d of %d (%d omitted, budget: %d tokens / %d count)",
                included,
                total,
                omitted,
                max_tokens,
                max_count,
            )
            summary += f"\n\n---\nShowing {included} of {total} findings ({omitted} lower-confidence findings omitted)."

        return summary

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/utils/citation_resolver.py
----
"""Two-phase citation workflow: prepare context with fact IDs → resolve references.

Reusable utility for any agent that needs "LLM writes with fact refs, code
resolves to numbered citations" — pure logic, no LLM calls, no state dependency.

Phase 1 (prepare_context): assigns sequential [F1]...[Fn] IDs to GroundedFacts
across all findings. Returns formatted text for the LLM prompt and the fact-ID map.

Phase 2 (resolve): parses [Fn] from LLM output, validates against the map,
deduplicates citations by (url, source_type), renumbers to [1],[2]... by
first-mention order, rewrites the text.

Phase 4 additions (4-D8):
- Layer 2: max_refs_per_sentence enforcement — caps unique [n] refs per sentence
- Layer 3: adjacent citation collapse — [1][1][1] → [1]
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass


try:
    from .doc_url import resolve_source_url as _resolve_source_url
except Exception:  # doc_url.py may not exist in older deployments
    import os

    def _resolve_source_url(url: str | None) -> str | None:  # type: ignore[misc]
        if not url:
            return None
        base = (os.getenv("DOC_BASE_URL") or "").rstrip("/")
        if base and not url.startswith(("http://", "https://")):
            return f"{base}/{url}"
        return url


from ..knowledge_agent_types import Citation, Finding, GroundedFact
from .findings_format import (
    distinct_source_pages,
    finding_freshness_line,
    finding_source_context_line,
    page_label,
)


logger = logging.getLogger(__name__)

# Zero-width characters LLMs commonly inject inside citation refs.
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad\u2060]")

# Tolerant patterns: optional zero-width chars between bracket and F, and
# between digits and closing bracket.
_FACT_REF_PATTERN = re.compile(r"[\[【［]\s*F(\d+)\s*[\]】］]")
_CITATION_NUM_PATTERN = re.compile(r"\[\d+\]")
_ADJACENT_DUP_PATTERN = re.compile(r"(\[\d+\])(\s*\1)+")

# Range refs: [F12-F15], [F12–F15], [F12‑F15], [F12-15] (optional second F).
# Handles ASCII hyphen, en-dash (U+2013), non-breaking hyphen (U+2011),
# and em-dash (U+2014).
_FACT_RANGE_PATTERN = re.compile(r"[\[【［]\s*F(\d+)\s*[-\u2011\u2013\u2014]\s*F?(\d+)\s*[\]】］]")

_MAX_RANGE_SPAN = 50

# Confidence ranking for budget-aware truncation (high kept first).
_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


def _strip_zero_width(text: str) -> str:
    """Remove zero-width Unicode characters that LLMs inject into refs."""
    return _ZERO_WIDTH_RE.sub("", text)


def _expand_fact_ranges(text: str) -> str:
    """Expand range refs like ``[F12-F15]`` into ``[F12][F13][F14][F15]``.

    Handles ASCII hyphen, en-dash, non-breaking hyphen, em-dash, and
    optional second ``F`` (e.g. ``[F12-15]``).  Reversed or excessively
    wide ranges are left as-is.
    """

    def _expand(match: re.Match) -> str:
        start, end = int(match.group(1)), int(match.group(2))
        if end < start or (end - start) > _MAX_RANGE_SPAN:
            return match.group(0)
        return "".join(f"[F{i}]" for i in range(start, end + 1))

    return _FACT_RANGE_PATTERN.sub(_expand, text)


def _default_dedup_key(citation: Citation) -> tuple[str, str]:
    """Default dedup key: (url or title, source_type)."""
    return (citation.url or citation.title, citation.source_type)


@dataclass
class CitationResult:
    """Output of deterministic citation resolution."""

    resolved_text: str
    citations: list[Citation]
    fact_id_to_citation_num: dict[str, int]
    orphan_refs: list[str]
    uncited_sentences: list[str]


class CitationResolver:
    """Two-phase citation workflow: prepare context → resolve references.

    Usage::

        resolver = CitationResolver()

        # Phase 1: tag facts for LLM prompt
        formatted_text, fact_map = resolver.prepare_context(findings)

        # ... LLM generates answer using [Fn] refs ...

        # Phase 2: deterministic resolution
        result = resolver.resolve(llm_answer, fact_map, max_refs_per_sentence=3)
        # result.resolved_text has [1], [2]... refs
        # result.citations has ordered Citation list
    """

    def prepare_context(
        self,
        findings: list[Finding],
        max_chars: int | None = None,
    ) -> tuple[str, dict[str, GroundedFact]]:
        """Build LLM context with [Fn]-tagged facts.

        Args:
            findings: List of findings from the evidence pipeline.
            max_chars: Optional char budget for the rendered findings text. When
                set, findings are sorted highest-confidence first and accumulated
                until the budget would be exceeded; the rest are dropped with a
                note (only kept facts receive [Fn] IDs). At least one finding is
                always included. ``None`` (default) renders every finding in the
                given order — the original behavior.

        Returns:
            (formatted_text, fact_map) where fact_map is {"F1": GroundedFact, ...}.
            Facts with citation=None get a tag but resolve to no citation during
            the resolve step (graceful degradation).
        """
        ordered = findings
        if max_chars is not None:
            ordered = sorted(findings, key=lambda f: _CONFIDENCE_ORDER.get(f.confidence, 3))

        fact_map: dict[str, GroundedFact] = {}
        sections: list[str] = []
        fact_counter = 0
        char_count = 0
        included = 0

        for finding in ordered:
            lines = [f"### Finding {included + 1}: {finding.topic}"]
            lines.append(f"Summary: {finding.summary}")
            # Per-finding (not per-fact) to keep token cost down — lets the
            # synthesizer weigh evidence age and caveat stale sources.
            freshness = finding_freshness_line(finding.key_facts)
            if freshness:
                lines.append(freshness)
            # Per-page identity + contextual prefix so the synthesizer can tell
            # whether a generic-looking fact is really about the asked entity or
            # belongs to a different team/app/space (the context_summary signal).
            source_context = finding_source_context_line(finding.citations, max_summary_chars=320)
            if source_context:
                lines.append(source_context)

            # When one finding draws on more than one page, the finding-level
            # "Source context" block alone cannot tie a specific fact to its
            # page. Tag each fact with its page in that case so the synthesizer
            # can attribute precisely (the default PER_DOC_TOKEN_GROUP packing
            # mixes pages in a single finding). Single-page findings stay terse.
            fact_citations = [gf.citation for gf in finding.key_facts if gf.citation]
            multi_page = distinct_source_pages(fact_citations) > 1

            section_fact_map: dict[str, GroundedFact] = {}
            if finding.key_facts:
                lines.append("Key facts:")
                for gf in finding.key_facts:
                    fact_id = f"F{fact_counter + len(section_fact_map) + 1}"
                    section_fact_map[fact_id] = gf
                    page_tag = f"  (source: {page_label(gf.citation)})" if multi_page and gf.citation else ""
                    lines.append(f"  [{fact_id}] {gf.fact}{page_tag}")

            section = "\n".join(lines)

            # Budget check — always include the first finding so the synthesizer
            # never receives an empty evidence block.
            if max_chars is not None and included > 0 and char_count + len(section) > max_chars:
                break

            sections.append(section)
            fact_map.update(section_fact_map)
            fact_counter += len(section_fact_map)
            char_count += len(section) + 2  # account for the "\n\n" join
            included += 1

        if max_chars is not None and included < len(ordered):
            omitted = len(ordered) - included
            logger.warning(
                "CitationResolver: truncated synthesis findings — showing %d of %d (%d lower-confidence omitted, budget: %d chars)",
                included,
                len(ordered),
                omitted,
                max_chars,
            )
            sections.append(f"---\nShowing {included} of {len(ordered)} findings ({omitted} lower-confidence findings omitted).")

        return "\n\n".join(sections), fact_map

    def resolve(
        self,
        text: str,
        fact_map: dict[str, GroundedFact],
        *,
        dedup_key: Callable[[Citation], tuple] | None = None,
        max_refs_per_sentence: int = 0,
    ) -> CitationResult:
        """Parse [Fn] refs, validate, dedup, renumber to [1],[2]...

        Steps:
        1. Regex-extract all [Fn] from text, preserving order of first appearance
        2. Validate each against fact_map — collect orphans
        3. Resolve Fn → GroundedFact → Citation (skip None citations)
        4. Dedup citations, merge retriever_name (comma-separated)
        5. Assign [1],[2]... by first-mention order
        6. Rewrite [Fn] → [citation_number] in text
        7. Enforce per-sentence citation density cap (Layer 2, 4-D8)
        8. Collapse adjacent identical citations (Layer 3, 4-D8)
        9. Detect uncited sentences (heuristic: split by period)

        Args:
            text: LLM-generated text containing [Fn] references.
            fact_map: Mapping from "Fn" to GroundedFact (from prepare_context).
            dedup_key: Custom dedup function. Default: (url or title, source_type).
            max_refs_per_sentence: Max unique citation numbers per sentence.
                0 = no cap (disabled).

        Returns:
            CitationResult with resolved text, ordered citations, and diagnostics.
        """
        key_fn = dedup_key or _default_dedup_key

        # Strip zero-width chars the LLM may inject inside refs.
        text = _strip_zero_width(text)

        # Expand range refs ([F12-F15] → [F12][F13][F14][F15]) before
        # individual ref extraction.
        text = _expand_fact_ranges(text)

        if re.search(r"[【［]F\d+[】］]", text):
            logger.debug("Unicode brackets detected in LLM output — normalising to ASCII")

        # 1. Extract all [Fn] refs in order of first appearance
        all_matches = _FACT_REF_PATTERN.findall(text)
        seen_fact_ids: list[str] = []
        for num_str in all_matches:
            fid = f"F{num_str}"
            if fid not in seen_fact_ids:
                seen_fact_ids.append(fid)

        # 2. Validate against fact_map, collect orphans
        orphan_refs: list[str] = []
        valid_fact_ids: list[str] = []
        for fid in seen_fact_ids:
            if fid in fact_map:
                valid_fact_ids.append(fid)
            else:
                orphan_refs.append(f"[{fid}]")

        # 3-4. Resolve to citations, dedup by key
        dedup_map: dict[tuple, int] = {}
        citations: list[Citation] = []
        fact_to_num: dict[str, int] = {}

        for fid in valid_fact_ids:
            gf = fact_map[fid]
            if gf.citation is None:
                continue

            dk = key_fn(gf.citation)
            if dk in dedup_map:
                num = dedup_map[dk]
                fact_to_num[fid] = num
                existing = citations[num - 1]
                if gf.citation.retriever_name and gf.citation.retriever_name not in existing.retriever_name:
                    existing_names = existing.retriever_name
                    merged = f"{existing_names}, {gf.citation.retriever_name}" if existing_names else gf.citation.retriever_name
                    citations[num - 1] = Citation(
                        title=existing.title,
                        url=existing.url,
                        source_type=existing.source_type,
                        snippet=existing.snippet,
                        retriever_name=merged,
                        metadata=existing.metadata,
                    )
            else:
                num = len(citations) + 1
                dedup_map[dk] = num
                fact_to_num[fid] = num
                citations.append(
                    Citation(
                        title=gf.citation.title,
                        url=_resolve_source_url(gf.citation.url),
                        source_type=gf.citation.source_type,
                        snippet=gf.citation.snippet,
                        retriever_name=gf.citation.retriever_name,
                        metadata=dict(gf.citation.metadata),
                    )
                )

        # Also map fact_ids that appear multiple times
        for num_str in all_matches:
            fid = f"F{num_str}"
            if fid in fact_map and fid not in fact_to_num:
                gf = fact_map[fid]
                if gf.citation is not None:
                    dk = key_fn(gf.citation)
                    if dk in dedup_map:
                        fact_to_num[fid] = dedup_map[dk]

        # 5-6. Rewrite [Fn] → [citation_number] in text
        def _replace_ref(match: re.Match) -> str:
            fid = f"F{match.group(1)}"
            if fid in fact_to_num:
                return f"[{fact_to_num[fid]}]"
            return match.group(0)

        resolved_text = _FACT_REF_PATTERN.sub(_replace_ref, text)

        # 7. Per-sentence citation density cap (Layer 2)
        if max_refs_per_sentence > 0:
            resolved_text = _enforce_citation_density(resolved_text, max_refs_per_sentence)

        # 8. Collapse adjacent identical citations (Layer 3)
        resolved_text = _collapse_adjacent_citations(resolved_text)

        # 9. Detect uncited sentences (heuristic)
        uncited = _find_uncited_sentences(resolved_text)

        return CitationResult(
            resolved_text=resolved_text,
            citations=citations,
            fact_id_to_citation_num=fact_to_num,
            orphan_refs=orphan_refs,
            uncited_sentences=uncited,
        )


def _enforce_citation_density(text: str, max_refs: int) -> str:
    """Cap unique citation numbers per sentence, keeping first N."""
    lines: list[str] = []
    for line in text.split("\n"):
        if not line.strip() or line.strip().startswith("#") or line.strip().startswith("-"):
            lines.append(line)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", line)
        processed: list[str] = []
        for sentence in sentences:
            refs_in_sentence = _CITATION_NUM_PATTERN.findall(sentence)
            if not refs_in_sentence:
                processed.append(sentence)
                continue

            seen: list[str] = []
            for ref in refs_in_sentence:
                if ref not in seen:
                    seen.append(ref)

            if len(seen) <= max_refs:
                processed.append(sentence)
                continue

            # Keep only the first max_refs unique refs
            allowed = set(seen[:max_refs])
            dropped = set(seen[max_refs:])

            def _make_filter(keep: set[str]) -> Callable[[re.Match], str]:
                def _filter_ref(m: re.Match) -> str:
                    return m.group(0) if m.group(0) in keep else ""

                return _filter_ref

            sentence = _CITATION_NUM_PATTERN.sub(_make_filter(allowed), sentence)
            # Clean up extra whitespace from removed refs
            sentence = re.sub(r"  +", " ", sentence).strip()
            if dropped:
                logger.debug("Citation density cap: dropped %s in sentence", dropped)
            processed.append(sentence)

        lines.append(" ".join(processed))
    return "\n".join(lines)


def _collapse_adjacent_citations(text: str) -> str:
    """Collapse adjacent identical citation numbers: [1][1][1] → [1]."""
    return _ADJACENT_DUP_PATTERN.sub(r"\1", text)


_NUM_REF_LINK_PATTERN = re.compile(r"\[(\d+)\](?!\()")


def linkify_citations(text: str, citations: list[Citation]) -> str:
    """Rewrite bare ``[N]`` citation markers into markdown links ``[N](url)``.

    Each ``[N]`` whose 1-based citation carries a non-empty url becomes a
    clickable markdown link. Markers without a url, out of range, or already
    followed by ``(`` (a link, or natural parenthetical text directly after)
    are left unchanged. Two markers written directly adjacent (``[1][2]``)
    get a ``, `` separator inserted between them so they read as a list.
    Apply to resolved text (``[N]`` form) — not raw ``[Fn]`` text.

    Args:
        text: Resolved answer text containing ``[N]`` markers.
        citations: Ordered citation list (index 0 = citation ``[1]``).

    Returns:
        Text with linkable ``[N]`` markers rewritten to ``[N](url)`` and
        directly-adjacent markers comma-separated.
    """
    if not citations:
        return text

    prev_end = -1

    def _link(match: re.Match) -> str:
        nonlocal prev_end
        # Directly adjacent markers (``[1][2]`` — no character between) get a
        # ``, `` separator so they render as a list, not a run-on.
        separator = ", " if match.start() == prev_end else ""
        prev_end = match.end()

        num = int(match.group(1))
        rendered = match.group(0)
        if 1 <= num <= len(citations):
            url = citations[num - 1].url
            if url:
                rendered = f"[{num}]({url})"
        return f"{separator}{rendered}"

    return _NUM_REF_LINK_PATTERN.sub(_link, text)


def _find_uncited_sentences(text: str) -> list[str]:
    """Find sentences with no citation references (heuristic).

    Skips short sentences (< 20 chars), headers, and sentences that are
    purely structural (lists, transitions).
    """
    uncited: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", line)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            if not _CITATION_NUM_PATTERN.search(sentence):
                uncited.append(sentence)
    return uncited


# ── Streaming citation resolver ──────────────────────────────────────────

_PARTIAL_REF = re.compile(r"[\[【［](?:F\d*(?:\s*[-\u2011\u2013\u2014]\s*F?\d*)?)?$")


class StreamingCitationResolver:
    """Buffer-based ``[Fn]`` → ``[n]`` resolver for token-by-token streaming.

    Processes tokens incrementally from ``model.astream()`` and resolves
    fact-ID references to numbered citations on the fly.  Uses the same
    dedup-by-(url, source_type) logic as :class:`CitationResolver`.

    Usage::

        sr = StreamingCitationResolver(fact_map, max_refs_per_sentence=3)
        for token in llm_stream:
            resolved = sr.feed(token)
            if resolved:
                writer({"type": "token", "content": resolved})
        remaining = sr.flush()
        if remaining:
            writer({"type": "token", "content": remaining})

    Args:
        fact_map: ``{"F1": GroundedFact, ...}`` from
            :meth:`CitationResolver.prepare_context`.
        max_refs_per_sentence: Per-sentence citation cap (0 = disabled).
            Matches Layer 2 of 4-D8 density control.
        dedup_key: Custom dedup function. Default:
            ``(url or title, source_type)``.
    """

    def __init__(
        self,
        fact_map: dict[str, GroundedFact],
        *,
        max_refs_per_sentence: int = 0,
        dedup_key: Callable[[Citation], tuple] | None = None,
    ) -> None:
        self._fact_map = fact_map
        self._max_refs = max_refs_per_sentence
        self._key_fn = dedup_key or _default_dedup_key

        self._buffer = ""
        # Trailing citation from the previous flush (for cross-boundary
        # adjacent dedup, e.g. "[1]" emitted, next flush starts with "[1]").
        self._last_emitted_ref: str = ""

        # Dedup state (mirrors CitationResolver.resolve steps 3-4)
        self._dedup_map: dict[tuple, int] = {}
        self._fact_to_num: dict[str, int] = {}
        self._citations: list[Citation] = []

        # Per-sentence density tracking
        self._sentence_ref_count: int = 0
        self._sentence_unique_refs: list[str] = []
        # True when previous flush ended with sentence-ending punctuation;
        # next flush starting with whitespace triggers a sentence reset.
        self._trailing_punct: bool = False

    # ── public API ──

    def feed(self, token: str) -> str:
        """Feed a token; return resolved text ready for emission.

        Returns an empty string when the buffer is holding a potential
        partial ``[Fn]`` reference.
        """
        self._buffer += _strip_zero_width(token)
        return self._try_flush()

    def flush(self) -> str:
        """Flush the remaining buffer at end of stream.

        Any incomplete ``[Fn]`` ref is emitted as-is (orphan).
        """
        result = self._buffer
        self._buffer = ""
        if not result:
            return ""
        result = _expand_fact_ranges(result)
        result = self._resolve_complete_refs(result)
        result = _collapse_adjacent_citations(result)
        return self._dedup_across_boundary(result)

    @property
    def citations(self) -> list[Citation]:
        """Citations discovered so far (ordered by first mention)."""
        return list(self._citations)

    @property
    def fact_id_to_citation_num(self) -> dict[str, int]:
        """Mapping of fact IDs to citation numbers built during streaming."""
        return dict(self._fact_to_num)

    # ── internals ──

    def _try_flush(self) -> str:
        """Flush resolved content up to any partial ref at the buffer tail."""
        if _PARTIAL_REF.search(self._buffer):
            m = _PARTIAL_REF.search(self._buffer)
            assert m is not None
            safe = self._buffer[: m.start()]
            self._buffer = self._buffer[m.start() :]
            if not safe:
                return ""
            self._detect_sentence_boundary(safe)
            safe = self._resolve_complete_refs(safe)
            safe = _collapse_adjacent_citations(safe)
            self._update_trailing_punct(safe)
            return self._dedup_across_boundary(safe)

        result = self._buffer
        self._buffer = ""
        if not result:
            return ""
        self._detect_sentence_boundary(result)
        result = self._resolve_complete_refs(result)
        result = _collapse_adjacent_citations(result)
        self._update_trailing_punct(result)
        return self._dedup_across_boundary(result)

    _SENTENCE_END_RE = re.compile(r"[.!?]\s")

    def _detect_sentence_boundary(self, text: str) -> None:
        """Reset per-sentence density tracking on sentence boundaries.

        Handles two cases:
        1. Cross-chunk: previous flush ended with ``.!?`` and *text*
           starts with whitespace (char-by-char streaming).
        2. Within-chunk: *text* itself contains ``[.!?]\\s``.
        """
        if self._max_refs <= 0:
            return
        if self._trailing_punct and text and text[0] in " \n\t\r":
            self._reset_sentence_tracking()
            self._trailing_punct = False
        if self._SENTENCE_END_RE.search(text):
            self._reset_sentence_tracking()

    def _update_trailing_punct(self, text: str) -> None:
        """Track whether *text* ends with sentence-ending punctuation."""
        if self._max_refs <= 0:
            return
        stripped = text.rstrip()
        self._trailing_punct = bool(stripped) and stripped[-1] in ".!?"

    def _resolve_complete_refs(self, text: str) -> str:
        """Replace all complete ``[Fn]`` in *text* with ``[n]``.

        Expands range refs first, then resolves individual refs.
        Also resets per-sentence density tracking whenever a sentence
        boundary appears in the text between citations.
        """
        text = _expand_fact_ranges(text)
        result: list[str] = []
        last_end = 0
        for match in _FACT_REF_PATTERN.finditer(text):
            prefix = text[last_end : match.start()]
            if self._max_refs > 0 and self._SENTENCE_END_RE.search(prefix):
                self._reset_sentence_tracking()
            result.append(prefix)

            fid = f"F{match.group(1)}"
            num = self._resolve_fact_id(fid)
            if num is not None:
                result.append(self._maybe_cap(f"[{num}]"))
            else:
                result.append(match.group(0))
            last_end = match.end()

        result.append(text[last_end:])
        return "".join(result)

    def _resolve_fact_id(self, fid: str) -> int | None:
        """Resolve a single fact ID to a citation number (with dedup)."""
        if fid in self._fact_to_num:
            return self._fact_to_num[fid]

        gf = self._fact_map.get(fid)
        if gf is None or gf.citation is None:
            return None

        dk = self._key_fn(gf.citation)
        if dk in self._dedup_map:
            num = self._dedup_map[dk]
            self._fact_to_num[fid] = num
            # Merge retriever_name
            existing = self._citations[num - 1]
            if gf.citation.retriever_name and gf.citation.retriever_name not in existing.retriever_name:
                merged = f"{existing.retriever_name}, {gf.citation.retriever_name}" if existing.retriever_name else gf.citation.retriever_name
                self._citations[num - 1] = Citation(
                    title=existing.title,
                    url=existing.url,
                    source_type=existing.source_type,
                    snippet=existing.snippet,
                    retriever_name=merged,
                    metadata=existing.metadata,
                )
            return num

        num = len(self._citations) + 1
        self._dedup_map[dk] = num
        self._fact_to_num[fid] = num
        self._citations.append(
            Citation(
                title=gf.citation.title,
                url=_resolve_source_url(gf.citation.url),
                source_type=gf.citation.source_type,
                snippet=gf.citation.snippet,
                retriever_name=gf.citation.retriever_name,
                metadata=dict(gf.citation.metadata),
            )
        )
        return num

    def _maybe_cap(self, ref: str) -> str:
        """Apply per-sentence density cap. Returns ref or empty string."""
        if self._max_refs <= 0:
            return ref

        if ref not in self._sentence_unique_refs:
            if len(self._sentence_unique_refs) >= self._max_refs:
                return ""
            self._sentence_unique_refs.append(ref)
        return ref

    def _dedup_across_boundary(self, text: str) -> str:
        """Remove leading citation that duplicates the previous flush's trailing citation."""
        if not text:
            return text
        if self._last_emitted_ref:
            pattern = re.compile(r"^(\s*" + re.escape(self._last_emitted_ref) + r")+")
            text = pattern.sub("", text)
        if not text:
            # Everything was stripped — the consumer still sees the previous
            # trailing ref, so keep _last_emitted_ref unchanged.
            return text
        trailing = re.search(r"(\[\d+\])\s*$", text)
        self._last_emitted_ref = trailing.group(1) if trailing else ""
        return text

    def _reset_sentence_tracking(self) -> None:
        self._sentence_unique_refs.clear()

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/utils/findings_format.py
----
"""Shared formatting and estimation for findings.

Used by CompressNode, ReviewEvidenceNode, PlanQueriesNode, and eval context builder.

Single source of truth for:
- ``format_finding_block``: the "### Finding N: topic ..." block rendered in prompts
- ``estimate_rendered_chars``: total rendered char cost of a findings list, used
  by both CompressNode (recompression decisions) and ReviewEvidenceNode (prompt
  truncation) so their token estimates stay aligned.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..knowledge_agent_types import Citation, Finding, GroundedFact


# Superset of metadata keys shown in citation display lines (includes "doc" for context).
_CITATION_DISPLAY_KEYS = ("pageId", "page_id", "doc", "full_doc_id", "chunk_id")

# Freshness signals in preference order: the source-content update date is what
# staleness actually means; the index ingestion date is only a weak fallback
# (re-ingestion makes old content look fresh), hence the distinct verb so the
# LLM knows which signal it is looking at.
_FRESHNESS_KEYS = (("lastDocUpdate", "last updated"), ("lastDocIngestion", "ingested"))


# Staleness flag thresholds (days). Past these ages the label carries an
# explicit marker so the reviewer / synthesizer treat the source with caution.
STALE_AFTER_DAYS = 180
OUTDATED_AFTER_DAYS = 365
_STALE_MARKER = "[STALE: >6 months old]"
_OUTDATED_MARKER = "[OUTDATED: >1 year old]"


def staleness_label(raw: Any, *, now: datetime | None = None) -> str | None:
    """Format a ``lastDocIngestion`` value as ``"YYYY-MM-DD (Nd ago)"``.

    Expects the canonical ISO 8601 string the retriever's result mapper emits.
    Returns ``None`` for missing or unparseable values so callers render
    nothing rather than a corrupt label. Future dates clamp to ``0d``.
    Ages past ``STALE_AFTER_DAYS`` / ``OUTDATED_AFTER_DAYS`` append an explicit
    flag — e.g. ``"2025-05-12 (412d ago) [OUTDATED: >1 year old]"`` — that the
    review and synthesis prompts instruct the model to act on.
    """
    if raw is None or raw == "":
        return None
    try:
        ingested = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if ingested.tzinfo is None:
        ingested = ingested.replace(tzinfo=UTC)
    age_days = max(0, ((now or datetime.now(UTC)) - ingested).days)
    label = f"{ingested.date().isoformat()} ({age_days}d ago)"
    if age_days > OUTDATED_AFTER_DAYS:
        return f"{label} {_OUTDATED_MARKER}"
    if age_days > STALE_AFTER_DAYS:
        return f"{label} {_STALE_MARKER}"
    return label


def freshness_from_metadata(meta: dict[str, Any] | None, *, now: datetime | None = None) -> str | None:
    """Best freshness label for one citation's metadata, with its verb.

    Prefers ``lastDocUpdate`` (content age) and falls back to
    ``lastDocIngestion`` (index age) — e.g. ``"last updated 2025-05-12
    (412d ago) [OUTDATED: >1 year old]"``. Returns ``None`` when neither key
    parses.
    """
    if not meta:
        return None
    for key, verb in _FRESHNESS_KEYS:
        label = staleness_label(meta.get(key), now=now)
        if label is not None:
            return f"{verb} {label}"
    return None


def finding_freshness_line(key_facts: list[GroundedFact], *, now: datetime | None = None) -> str | None:
    """One-line source-freshness summary across a finding's fact citations.

    Renders once per finding (not per fact) to keep token cost down. Each
    citation contributes its preferred freshness signal (``lastDocUpdate``
    first, ``lastDocIngestion`` fallback): a single distinct date yields
    ``"Source freshness: last updated 2025-05-12 (412d ago)"``; several yield
    an oldest-to-newest range. Returns ``None`` when no citation carries a
    parseable date.
    """
    dates: dict[str, tuple[datetime, str]] = {}
    for gf in key_facts:
        citation = getattr(gf, "citation", None)
        meta = getattr(citation, "metadata", None) or {}
        for key, verb in _FRESHNESS_KEYS:
            raw = meta.get(key)
            if raw is None or raw == "":
                continue
            try:
                stamped = datetime.fromisoformat(str(raw))
            except ValueError:
                continue
            if stamped.tzinfo is None:
                stamped = stamped.replace(tzinfo=UTC)
            dates[stamped.date().isoformat()] = (stamped, verb)
            break
    if not dates:
        return None
    ordered = sorted(dates.values(), key=lambda pair: pair[0])
    oldest_dt, oldest_verb = ordered[0]
    oldest = staleness_label(oldest_dt.isoformat(), now=now)
    if len(dates) == 1:
        return f"Source freshness: {oldest_verb} {oldest}"
    newest_dt, _ = ordered[-1]
    newest = staleness_label(newest_dt.isoformat(), now=now)
    return f"Source freshness: sources dated between {oldest} and {newest}"


def _format_key_fact(item: GroundedFact | str | dict) -> str:
    """Format a single key fact with optional source attribution.

    Handles three representations:
    - GroundedFact object: use .fact with [Source: title] if citation present
    - str: plain fact text (backward compat)
    - dict: serialized GroundedFact from asdict() (eval pipeline)
    """
    if isinstance(item, str):
        return f"  - {item}"

    if isinstance(item, dict):
        fact = item.get("fact", str(item))
        citation = item.get("citation")
        if citation and citation.get("title"):
            return f"  - [Source: {citation['title']}] {fact}"
        return f"  - {fact}"

    # GroundedFact object
    if item.citation and item.citation.title:
        return f"  - [Source: {item.citation.title}] {item.fact}"
    return f"  - {item.fact}"


def format_finding_block(
    index: int,
    topic: str,
    summary: str,
    key_facts: list[GroundedFact] | list[str] | list[dict],
    *,
    expansion_marker: str = "",
    sources_line: str | None = None,
    citations_line: str | None = None,
    source_context_line: str | None = None,
) -> str:
    """Format a single finding as a markdown block.

    Used by:
    - PlanQueriesNode._format_findings (refinement prompt; no expansion/sources)
    - ReviewEvidenceNode._build_findings_summary (review prompt; expansion + sources + citations)
    - ka_context_builder.serialize_findings (eval output; expansion + sources, no confidence)

    Args:
        index: 1-based finding number.
        topic: Finding topic label.
        summary: Finding summary text.
        key_facts: List of GroundedFact objects, plain strings, or serialized dicts.
        expansion_marker: Optional suffix after topic, e.g. " [NEEDS EXPANSION]".
        sources_line: Optional final line, e.g. "Sources: elastic_docs, lightrag".
        citations_line: Optional line with citation identifiers, e.g. "Cited documents: pageId=123 (elastic_docs)".
        source_context_line: Optional multi-line "Source context:" block naming
            the finding's distinct source pages and their context (see
            ``finding_source_context_line``) so the reader can tell whether the
            evidence belongs to the asked entity or a different team/app/space.

    Returns:
        Single finding block (no trailing newline).
    """
    facts = "\n".join(_format_key_fact(item) for item in key_facts)
    block = f"### Finding {index}: {topic}{expansion_marker}\n{summary}\nKey facts:\n{facts}"
    if sources_line is not None:
        block += f"\n{sources_line}"
    if citations_line is not None:
        block += f"\n{citations_line}"
    if source_context_line is not None:
        block += f"\n{source_context_line}"
    return block


# ---------------------------------------------------------------------------
# Citation display helper (shared between estimation and review prompt)
# ---------------------------------------------------------------------------


def cited_documents_line(citations: list[Citation]) -> str | None:
    """Build a compact line of citation identifiers for prompt display.

    Extracts metadata keys (pageId, doc, full_doc_id, chunk_id) from each
    citation and formats them as ``key=value (retriever)``.  Returns None
    when no displayable identifiers exist.
    """
    if not citations:
        return None
    seen: set[str] = set()
    parts: list[str] = []
    for c in citations:
        meta = c.metadata or {}
        retriever = c.retriever_name or "unknown"
        for key in _CITATION_DISPLAY_KEYS:
            val = meta.get(key)
            if val is not None and val != "":
                item = f"{key}={val} ({retriever})"
                if item not in seen:
                    seen.add(item)
                    parts.append(item)
        # Raw ISO datetimes are token-heavy and the reviewer cares about age,
        # not the timestamp — render the compact freshness label instead.
        freshness = freshness_from_metadata(meta)
        if freshness:
            item = f"{freshness} ({retriever})"
            if item not in seen:
                seen.add(item)
                parts.append(item)
        if not meta and (c.url or c.title):
            fallback = (c.url or c.title or "").strip()
            if fallback:
                item = f"doc={fallback} ({retriever})"
                if item not in seen:
                    seen.add(item)
                    parts.append(item)
    return ("Cited documents: " + ", ".join(parts)) if parts else None


# ---------------------------------------------------------------------------
# Per-page source-context line (page identity + contextual prefix)
# ---------------------------------------------------------------------------

# Metadata keys carrying per-page IDENTITY signal — distinct from freshness
# (handled above) and from the bare title. These let a reviewer/synthesizer tell
# whether a generic-looking chunk (a contact page, an index, boilerplate) is
# really about the asked entity or belongs to a different team / app / space.
_SOURCE_IDENTITY_KEYS = ("appName", "apcode", "entity")


def _first_value(*values: Any) -> str | None:
    """Return the first non-empty value as a string, else ``None``."""
    for value in values:
        if value is None or value == "":
            continue
        return str(value)
    return None


def _flatten_summary(text: Any, max_chars: int) -> str:
    """Collapse a context_summary to one whitespace-normalized, capped line."""
    flat = " ".join(str(text).split())
    if max_chars > 0 and len(flat) > max_chars:
        return flat[: max(max_chars - 1, 0)].rstrip() + "…"
    return flat


def _page_dedup_key(citation: Citation) -> str | None:
    """Identity used to collapse a finding's citations to one line per page.

    Mirrors ``page_group_key`` (compression) ladder: pageId → page_id → doc,
    falling back to the citation title so a page is still distinguishable when no
    structured id is present.
    """
    meta = citation.metadata or {}
    return _first_value(meta.get("pageId"), meta.get("page_id"), meta.get("doc")) or (citation.title or None)


def page_label(citation: Citation) -> str:
    """Short human label identifying a citation's source page.

    Used to tag individual facts with their page when a single finding draws on
    more than one page, so the synthesizer can tell which fact came from which
    page's context. Prefers a pageId, then a trimmed title, then the doc path.
    """
    meta = citation.metadata or {}
    page_id = _first_value(meta.get("pageId"), meta.get("page_id"))
    if page_id:
        return f"pageId={page_id}"
    title = (citation.title or "").strip()
    if title:
        return title if len(title) <= 60 else title[:57].rstrip() + "…"
    return _first_value(meta.get("doc")) or "unknown source"


def distinct_source_pages(citations: list[Citation]) -> int:
    """Count distinct source pages across a citation list (dedup by page key)."""
    keys: set[str] = set()
    for c in citations:
        key = _page_dedup_key(c)
        if key:
            keys.add(key)
    return len(keys)


def finding_source_context_line(citations: list[Citation], *, max_summary_chars: int = 320) -> str | None:
    """Render a finding's per-page source identity + contextual prefix.

    One line per DISTINCT source page (deduped by :func:`_page_dedup_key`),
    surfacing the page's identity (title, pageId, appName, apcode, entity name)
    and its ``context_summary`` — the contextual prefix the retriever recovered
    from the page (Confluence space, page path, parent breadcrumbs). This is the
    signal that lets a reviewer/synthesizer distinguish a page that is genuinely
    about the asked entity from a generic page whose metadata shows it belongs to
    a different team / application / space.

    A page is rendered only when it carries real identity or context beyond a
    bare title (a title alone is low signal and already implicit elsewhere).
    Returns ``None`` when no page qualifies, so callers can omit the line.

    Args:
        citations: The finding's citation list.
        max_summary_chars: Cap for each page's ``context_summary`` (0 disables
            the cap). Synthesis uses a larger cap than the review prompts.

    Returns:
        A ``"Source context:\\n- …"`` block, or ``None``.
    """
    if not citations:
        return None

    seen: set[str] = set()
    lines: list[str] = []
    for c in citations:
        key = _page_dedup_key(c)
        if key is None or key in seen:
            continue
        meta = c.metadata or {}

        identity_parts = [f"{k}={meta[k]}" for k in _SOURCE_IDENTITY_KEYS if meta.get(k) not in (None, "")]
        raw_summary = meta.get("context_summary")
        summary_text = _flatten_summary(raw_summary, max_summary_chars) if raw_summary else ""
        page_id = _first_value(meta.get("pageId"), meta.get("page_id"))
        doc = _first_value(meta.get("doc"))

        # Skip pages with nothing but a bare title — no disambiguation value.
        if not (identity_parts or summary_text or page_id or doc):
            continue
        seen.add(key)

        title = (c.title or page_id or doc or "Untitled").strip()
        label = f"{title} (pageId={page_id})" if page_id and page_id != title else title
        meta_suffix = f" [{' · '.join(identity_parts)}]" if identity_parts else ""
        summary_suffix = f": {summary_text}" if summary_text else ""
        lines.append(f"- {label}{meta_suffix}{summary_suffix}")

    if not lines:
        return None
    return "Source context:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Token cost estimation — single source of truth for CompressNode + ReviewNode
# ---------------------------------------------------------------------------


def estimate_rendered_chars(findings: list[Finding]) -> int:
    """Estimate total rendered character cost of findings as they appear in prompts.

    Renders each finding through ``format_finding_block`` with the same
    parameters ReviewEvidenceNode uses (sources_line, citations_line,
    expansion_marker), then sums the block character lengths.

    This is the single source of truth for token cost estimation.  Both
    CompressNode (recompression decisions via ``global_char_count``) and
    ReviewEvidenceNode (prompt truncation budget) use this so their
    estimates stay aligned.

    Note: inter-block separators ("\\n\\n") are excluded to match
    ReviewEvidenceNode's per-block token accounting.
    """
    total = 0
    for i, f in enumerate(findings, 1):
        block = format_finding_block(
            i,
            f.topic,
            f.summary,
            f.key_facts,
            expansion_marker=" [NEEDS EXPANSION]" if f.needs_expansion else "",
            sources_line=f"Sources: {', '.join(f.retriever_sources) if f.retriever_sources else 'unknown'}",
            citations_line=cited_documents_line(f.citations) if f.citations else None,
            # Must mirror ReviewEvidenceNode._build_findings_summary (same 320-char
            # cap) so this estimate stays aligned with what review actually renders.
            source_context_line=finding_source_context_line(f.citations, max_summary_chars=320) if f.citations else None,
        )
        total += len(block)
    return total

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/orchestrator/prompts/orchestrator_planner_prompt.py
----
"""Orchestrator planner system-prompt builder."""

from __future__ import annotations

from sta_agent_engine.agents.base.prompts.capability_definition import CapabilityDefinition
from sta_agent_engine.agents.base.prompts.prompt_manager import PromptManager

from .orchestrator_grounding import (
    _CLARIFICATION_SECTION,
    _DEEPAGENTS_TOOL_GUIDELINES,
    _FUTURE_CAPABILITIES_SECTION,
    _GENERAL_KNOWLEDGE_SECTION,
    _GROUNDING_SECTION,
    _SUBAGENT_TASKING_SECTION,
    _UNCERTAINTY_SECTION,
)


# The roadmap teaser (`<future_capabilities>`: topology / coding / security
# engines) is off by default — it spends tokens on capabilities the planner
# cannot invoke and risks it offering them. Flip to ``True`` to re-advertise.
_INCLUDE_FUTURE_CAPABILITIES = False


_PLANNER_IDENTITY = """<identity>
You are TWIN, an enterprise IT operations orchestrator. You plan and
coordinate calls to specialist tools and sub-agents to answer user
questions.
</identity>"""

_PLANNER_OBJECTIVE = """<objective>
For each user message:
1. Identify the user's intent.
2. Pick the smallest tool / sub-agent that can answer it.
3. Wait for the result, then return it to the user as a complete,
   self-contained answer — or delegate further if a multi-step task requires
   it. The user sees only your reply, never the sub-agent's output.
4. Stay grounded in tool / sub-agent output — never invent facts.
</objective>"""

_OUTPUT_FORMAT = """
- Your reply is the ONLY thing the user sees: they do NOT see sub-agent
  outputs, tool results, your todos, or your intermediate steps — any general
  note that tool output is visible in real time does not apply here. So every
  reply must be COMPLETE and SELF-SUFFICIENT: restate inline every fact,
  figure, entity name, and ID the user needs to act on, and never refer to
  content they can't see ("as shown above", "as the sub-agent returned", "see
  the table") — there is no "above" for the user.
- Reply in the user's language.
- Be concise by default; go longer only when the user asks for detail or the
  question genuinely needs it. Conciseness is about your own prose — never trim
  a sub-agent's substance (counts, rows, citation markers) to save space, and
  never sacrifice the completeness a self-sufficient answer requires.
- Relay sub-agent answers faithfully: preserve their substance — counts,
  figures, entity names, IDs, and codes exactly as reported. Don't recompute or
  round figures, don't relabel entities, don't add details the sub-agent didn't
  provide.
- Keep a sub-agent's formatting when it makes the answer easier to read
  (tables, lists, code blocks) rather than flattening it to prose.
- If a sub-agent reports no result, relay that plainly (see the uncertainty
  rules) — never substitute a fabricated answer.
- Do NOT append a ``Sources:`` block and do NOT reproduce source titles or urls.
  The ordered source list is displayed separately and deterministically after
  your reply — you never build, restate, or summarize it.
- Cite a sub-agent's grounded source by ITS number, unchanged. When a fact you
  state comes from a source the sub-agent cited as ``[N]``, mark that fact with
  the same ``[N]`` the sub-agent used. Reference only the sources you actually
  rely on and keep their original numbers — if those are the sub-agent's 1st and
  4th sources, write ``[1]`` and ``[4]`` (gaps are fine). Never renumber, never
  recompact ``[1]`` ``[4]`` to ``[1]`` ``[2]``, never invent a number, and never
  attach a number to an operational/computed fact that has no cited source.
- If you call a knowledge sub-agent more than once in a turn, each call numbers
  its own sources from ``[1]``. The displayed list concatenates them in call
  order, so OFFSET every call after the first by the total number of sources the
  earlier calls listed: if the first call listed 5 sources, the second call's
  ``[1]`` ``[2]`` become ``[6]`` ``[7]``, and so on. Count by the sources each
  sub-agent LISTED (its own ``[N]`` range), not by how many you cited — the
  displayed list shows them all.
"""


def build_planner_system_prompt(
    *,
    soul: str | None = None,
    tools: list[CapabilityDefinition],
    subagents: list[CapabilityDefinition],
    persona: str | None,
    auth_status: str | None,
) -> str:
    """Build the orchestrator's planner system prompt from bound capabilities.

    This builder no longer emits a ``<capabilities>`` block. The subagent
    roster reaches the model through Deep Agents' own assembled prompt — the
    auto-generated "Available subagent types" list and the ``task`` tool
    description, both built from each subagent's compact description (which now
    carries its corpora ``sources``). Keeping a parallel ``<capabilities>``
    block here only duplicated that, so this prompt covers *posture* (grounding,
    clarification, tasking discipline, output format) and leaves the roster to
    Deep Agents.

    Section ordering is deliberate for prefix caching: the static character +
    behavior block leads (a stable shared prefix across every call and user),
    and per-call dynamic content (``persona``, ``auth_status``) is appended at
    the tail so it never shifts the cached prefix.

    Args:
        tools: Capability metadata for tools actually bound on this graph. Used
            only to decide whether the optional ``<future_capabilities>`` teaser
            is worth rendering.
        subagents: Capability metadata for subagents actually registered. Gates
            the ``<subagent_tasking>`` section (delegation discipline is only
            relevant when the planner can delegate). Corpora advertisement now
            lives in each subagent's compact description, not here.
        persona: Optional per-call persona text; rendered as a trailing
            ``<persona>`` section (dynamic tail), not nested in the lead block.
        auth_status: Optional degraded-mode banner; ``None`` for normal operation.
        soul: Optional static character + role text (``SOUL.md``). When present,
            it leads the prompt as the ``<soul>`` section and subsumes
            ``<identity>``. When ``None`` (file absent/empty/unreadable), the
            builder falls back to the legacy ``<identity>`` constant.

    Returns:
        Fully-assembled system prompt string.
    """
    base = f"<soul>\n{soul.strip()}\n</soul>\n\n{_PLANNER_OBJECTIVE}" if soul else f"{_PLANNER_IDENTITY}\n\n{_PLANNER_OBJECTIVE}"
    pm = PromptManager(base)

    if _INCLUDE_FUTURE_CAPABILITIES and (tools or subagents):
        pm.add_section("future_capabilities", _FUTURE_CAPABILITIES_SECTION, mode="create")

    pm.add_section("grounding", _GROUNDING_SECTION, mode="create")
    pm.add_section("clarification", _CLARIFICATION_SECTION, mode="create")
    pm.add_section("general_knowledge", _GENERAL_KNOWLEDGE_SECTION, mode="create")
    pm.add_section("uncertainty", _UNCERTAINTY_SECTION, mode="create")
    pm.add_section("guidelines", _DEEPAGENTS_TOOL_GUIDELINES, mode="create")
    if subagents:
        # Agent-specific tasking guidance does NOT live here: each subagent's
        # routing posture (e.g. the KA's complementary-source note) rides in its
        # own compact description, which Deep Agents renders only when that
        # subagent is permitted. This section stays agent-agnostic.
        pm.add_section("subagent_tasking", _SUBAGENT_TASKING_SECTION, mode="create")
    pm.add_section("output_format", _OUTPUT_FORMAT, mode="create")

    # Dynamic tail — kept last so the static prefix above stays cache-stable.
    if persona:
        pm.add_section("persona", persona, mode="create")

    if auth_status:
        pm.add_section("auth_status", auth_status, mode="create")

    return pm.build()

-------

