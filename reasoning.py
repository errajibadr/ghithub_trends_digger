packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/elasticsearch/elastic_retriever.py
----
"""Elasticsearch hybrid retriever combining BM25 and kNN vector search.

This module provides a generic, configurable retriever for Elasticsearch that supports
multiple fusion strategies for combining lexical and semantic search results.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar

import httpx
from langchain_core.embeddings import Embeddings


if TYPE_CHECKING:
    # Provide identity-decorator stub so pyright sees the raw function signature
    # instead of langsmith's SupportsLangsmithExtra wrapper.
    _F = TypeVar("_F", bound=Callable[..., Any])

    def traceable(**kwargs: Any) -> Callable[[_F], _F]: ...
else:
    from langsmith import traceable

from ....adapters.elasticsearch.adapters_async import AsyncElasticsearchAdapter
from ....models.rerank_client import RerankClient, RerankResponse, RerankResult
from ..batch_document_provider import ChunkRange
from ..search_response import SearchResponse
from .elastic_retrieval_chunk import ElasticRetrievalChunk
from .elastic_search_config import (
    ElasticFieldConfig,
    ElasticSearchConfig,
    FusionStrategy,
    FusionStrategyLiteral,
)
from .fusion import (
    FusionOperator,
    PositionAwareBlend,
    RrfRerankerOperator,
    SubQuery,
    TopRankBonusRRF,
    WeightedRRF,
    resolve_fusion_operator,
)
from .metadata_scope import MetadataScope
from .query_expansion import ExpansionStrategy, QueryExpanderProtocol


logger = logging.getLogger(__name__)


class RerankUnavailableError(Exception):
    """Raised internally when a runtime reranking step fails after its own retries.

    Signals a *soft* failure: the rerank HTTP call (which already retries via
    ``RerankClient.arerank``) ultimately errored, so ``search_many`` should
    degrade to rerank-blind RRF fusion rather than return zero results. The
    original exception is chained (``from``) and the underlying ``reranker_arerank``
    LangSmith span is still recorded as failed — this wrapper only controls the
    fallback at the retriever boundary, it does not hide the failure from traces.

    This is distinct from the hard ``ValueError`` raised when a rerank-aware
    fusion operator is selected but no ``RerankClient`` is configured — that is a
    wiring bug and must surface, not silently degrade.
    """


# Anchor that terminates the contextual prefix in ``metadata.content``.
# Production folds a contextual summary plus a *variable* set of metadata fields
# into the single embedded ``content`` field and closes the prefix with a
# ``\n\nContent:`` marker (capital ``C``, blank-line separated). The parser keys
# on this anchor alone — never on a particular field name — so a changing field
# set doesn't break body recovery.
_BODY_ANCHOR = "\n\nContent:"

# Legacy anchor for the original prod-shaped template
# (see ``infra/elasticsearch/ingestion/chunker.build_structured_content`` and
# § 4 of ``creative_phase_2026-05-15_es_mapping_alignment.md``). ``rfind`` is
# load-bearing: a chunk body that happens to contain ``\ncontent: `` literals
# stays correctly anchored on the LAST occurrence.
_STRUCTURED_BODY_ANCHOR = "\ncontent: "

# Upper bound on chunks returned by a single batched fetch query
# (get_documents / get_chunk_ranges). Matches the single-document ceiling in
# get_document; a batch spanning more chunks is truncated with a warning.
_BATCH_FETCH_MAX_CHUNKS = 10_000


def parse_structured_content(structured_content: str, anchor: str = _BODY_ANCHOR) -> tuple[str, str]:
    """Split a structured ``metadata.content`` blob into ``(context_summary, body)``.

    Production folds a contextual summary plus a *variable* set of metadata
    fields into ``metadata.content``, terminating that prefix with ``anchor``
    (default ``\\n\\nContent:``). Everything after the FINAL anchor is the
    per-chunk body; everything before it is the per-page contextual prefix —
    identical across every chunk of one page. Splitting here lets the prefix be
    surfaced ONCE per page instead of repeating inside each chunk's body.

    Resolution order (last-marker ``rfind`` semantics throughout):
      1. ``anchor`` — the production ``\\n\\nContent:`` marker; one single
         leading space after the marker is dropped (a second space belongs to
         the body, e.g. indented content).
      2. Legacy ``\\ncontent: `` (lowercase, single newline, trailing space) —
         the original template; emits a ``logger.debug`` on multi-anchor clips.
      3. No anchor → ``("", structured_content)`` so un-templated / legacy raw
         indices pass through unchanged.
    """
    last = structured_content.rfind(anchor)
    if last >= 0:
        summary = structured_content[:last]
        body_start = last + len(anchor)
        if body_start < len(structured_content) and structured_content[body_start] == " ":
            body_start += 1
        return summary, structured_content[body_start:]

    last = structured_content.rfind(_STRUCTURED_BODY_ANCHOR)
    if last < 0:
        return "", structured_content
    first = structured_content.find(_STRUCTURED_BODY_ANCHOR)
    if first != last:
        logger.debug(
            "parse_structured_content: multiple '\\ncontent: ' anchors found "
            "(first=%d, last=%d); body recovery clipped on the last one — "
            "the body may have shadowed the template anchor",
            first,
            last,
        )
    return structured_content[:last], structured_content[last + len(_STRUCTURED_BODY_ANCHOR) :]


def extract_chunk_body(structured_content: str) -> str:
    """Pull the chunk body out of a structured ``metadata.content`` blob.

    Thin wrapper over :func:`parse_structured_content` that returns only the
    body. Recognizes both the production ``\\n\\nContent:`` and legacy
    ``\\ncontent: `` anchors and falls back to the full blob when neither is
    present, so legacy / un-templated indices keep working.
    """
    return parse_structured_content(structured_content)[1]


# Display-style timestamp formats observed in production for
# ``lastDocIngestion`` (Kibana-like ``"May 12, 2025 @ 10:30:00"``), with and
# without fractional seconds / full month names.
_INGESTION_TIMESTAMP_FORMATS = (
    "%b %d, %Y @ %H:%M:%S.%f",
    "%b %d, %Y @ %H:%M:%S",
    "%B %d, %Y @ %H:%M:%S.%f",
    "%B %d, %Y @ %H:%M:%S",
)


def parse_ingestion_timestamp(value: Any) -> datetime | None:
    """Parse a document-ingestion timestamp into an aware UTC datetime.

    Accepts the production display format (``"May 12, 2025 @ 10:30:00"``),
    ISO 8601 strings, and epoch milliseconds. Returns ``None`` when the value
    is missing or unparseable — callers keep the raw value in that case rather
    than dropping it, so no information is lost at the mapping boundary.
    Timezone-naive inputs are assumed UTC.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000.0, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in _INGESTION_TIMESTAMP_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _join_rerank_query(
    domain_intent: str | None,
    intent: str | None,
    query: str,
) -> str:
    """Build the intent-prepended rerank query (F5b-3).

    Canonical order per the planning doc: ``[domain_intent, intent, query]``
    joined by newlines, empty/None layers dropped. Returns just ``query``
    when neither intent layer is configured — preserves pre-F5b behavior.
    """
    return "\n".join(filter(None, [domain_intent, intent, query]))


def _process_retriever_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process inputs for LangSmith trace display."""
    return {
        "query": inputs.get("query", ""),
        "size": inputs.get("size", 10),
        "fusion_strategy": str(inputs.get("fusion_strategy", "default")),
        "documents_filter": inputs.get("documents"),
    }


def _process_search_many_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process inputs for LangSmith trace display (search_many)."""
    queries = inputs.get("queries")
    if isinstance(queries, str):
        preview: Any = queries
    elif isinstance(queries, list):
        preview = [{"type": sq.type, "query": sq.query, "weight": sq.weight} if hasattr(sq, "type") else sq for sq in queries]
    else:
        preview = None
    return {
        "queries": preview,
        "size": inputs.get("size", 10),
        "fusion": type(inputs.get("fusion")).__name__ if inputs.get("fusion") is not None else "WeightedRRF(default)",
        "documents_filter": inputs.get("documents"),
    }


def _process_embedding_outputs(outputs: Any) -> dict[str, Any]:
    """Process embedding outputs for LangSmith trace display."""
    if outputs is None:
        return {"dimensions": 0}
    if isinstance(outputs, list):
        return {"dimensions": len(outputs)}
    return {"dimensions": 0}


def _process_retriever_outputs(outputs: Any) -> dict[str, Any]:
    """Convert retrieval results to LangSmith Document format."""
    results = outputs.results if hasattr(outputs, "results") else (outputs if isinstance(outputs, list) else [])
    if not results:
        return {"documents": []}

    exclude_from_metadata = {"content", "text", "page_content"}
    docs = []
    for result in results:
        if hasattr(result, "content") and hasattr(result, "metadata"):
            metadata = {k: v for k, v in (result.metadata or {}).items() if k not in exclude_from_metadata}
            metadata["score"] = getattr(result, "score", 0.0)
            docs.append({"page_content": result.content, "type": "Document", "metadata": metadata})
        elif isinstance(result, dict):
            docs.append(
                {
                    "page_content": result.get("content", str(result)),
                    "type": "Document",
                    "metadata": {
                        "score": result.get("score", 0.0),
                        **{k: v for k, v in result.items() if k not in exclude_from_metadata | {"score"}},
                    },
                }
            )
        else:
            docs.append({"page_content": str(result), "type": "Document", "metadata": {}})
    return {"documents": docs}


def _process_get_document_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process inputs for LangSmith trace (elastic_get_document)."""
    return {"doc_id": inputs.get("doc_id", "")}


def _process_get_document_outputs(outputs: Any) -> dict[str, Any]:
    """Process outputs for LangSmith trace (elastic_get_document)."""
    chunks = outputs if isinstance(outputs, list) else []
    titles = [((getattr(c, "metadata", None) or {}).get("title") or "")[:80] for c in chunks[:5]]
    return {"num_chunks": len(chunks), "titles_preview": titles}


def _process_get_chunk_context_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process inputs for LangSmith trace (elastic_get_chunk_context)."""
    return {"chunk_id": inputs.get("chunk_id", ""), "window": inputs.get("window", 3)}


def _process_get_chunk_range_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process inputs for LangSmith trace (elastic_get_chunk_range)."""
    return {
        "doc_id": inputs.get("doc_id", ""),
        "start_index": inputs.get("start_index", 0),
        "end_index": inputs.get("end_index", 0),
    }


class ElasticRetriever:
    """Elasticsearch hybrid retriever combining BM25 text search with kNN vector search.

    Implements the BaseRetriever protocol via structural typing.
    search() returns SearchResponse[ElasticRetrievalChunk].
    search_vector_only and search_text_only remain as public methods for Elastic-specific use.

    Use ElasticFieldConfig to adapt to different index schemas (field names, boosts).

    Also satisfies ``SupportsMetadataScope`` — ``search()`` honors the
    ``metadata_scope: MetadataScope | None`` kwarg used by the KA's build-time
    + runtime-query metadata scope. Backends that don't honor it must NOT set
    this marker; the KA tool factory raises at build time when scope features
    are wired onto an unsupporting retriever.
    """

    # SupportsMetadataScope marker — see scope_capability.py and the KA
    # tool factory's build-time gate. Subclasses inherit by default; a
    # wrapper that intentionally drops scope handling must override to False.
    supports_metadata_scope: ClassVar[Literal[True]] = True

    # SupportsBatchFetch marker — see batch_document_provider.py. Advertises
    # the batched get_documents / get_chunk_ranges / get_chunk_contexts
    # methods. Like DocumentProvider, the methods still require page_id_field
    # and chunk_index_field to be configured (see supports_document_provider).
    supports_batch_fetch: ClassVar[Literal[True]] = True

    def __init__(
        self,
        adapter: AsyncElasticsearchAdapter,
        index: str,
        embedding_model: Embeddings,
        reranker: RerankClient | None = None,
        field_config: ElasticFieldConfig | None = None,
        search_config: ElasticSearchConfig | None = None,
        embedding_http_client: httpx.AsyncClient | None = None,
        expander: QueryExpanderProtocol | None = None,
        domain_intent: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.index = index
        self.embedding_model = embedding_model
        self._reranker = reranker
        self._embedding_http_client = embedding_http_client
        self.field_config = field_config or ElasticFieldConfig()
        self._search_config = search_config or ElasticSearchConfig()
        # Optional expander — None keeps the retriever on the no-expansion path.
        # Runtime use lands in Cycle E (uniform search() pipeline); for now the
        # retriever only *carries* the expander so factories can inspect it.
        self._expander = expander
        # Build-time "what is this index about" — set once, threaded into every
        # expansion call as the ``domain_intent`` kwarg. Runtime per-call intent
        # lives on ``ElasticSearchConfig.intent`` and flows through
        # ``ElasticRetrieverContext.retriever_intent`` instead (Cycle F5).
        self._domain_intent = domain_intent
        # Lazy license probe cache: None = not yet probed, bool = cached decision.
        self._native_rrf_available: bool | None = None

    # License types that grant access to Elastic's native RRF retriever query.
    # basic + missing license → Python RRF fallback.
    _RRF_LICENSED_TYPES = frozenset({"platinum", "enterprise", "trial"})

    async def _can_use_native_rrf(self) -> bool:
        """Decide whether to use Elastic's native `retriever.rrf` query.

        Modes (from ``ElasticSearchConfig.es_rrf_mode``):
            native — force native (no probe, caller owns license guarantee).
            python — force in-process RRF (skip probe entirely).
            auto   — probe ``/_license`` once on first call; cache the result.
                     Any probe error (network, auth, missing API) falls back to
                     Python RRF with a single warning.

        Returns:
            True if native RRF should be used, False for Python fallback.
        """
        mode = self._search_config.es_rrf_mode
        if mode == "native":
            return True
        if mode == "python":
            return False

        # mode == "auto" — lazy probe with instance cache
        if self._native_rrf_available is not None:
            return self._native_rrf_available

        try:
            response = await self.adapter.client.license.get()
            license_info = response.get("license", {}) if isinstance(response, dict) else {}
            license_type = str(license_info.get("type", "")).lower()
            available = license_type in self._RRF_LICENSED_TYPES
        except Exception as exc:
            logger.warning(
                "Elastic license probe failed (%s); falling back to Python RRF for this retriever instance.",
                exc.__class__.__name__,
            )
            available = False
        else:
            if not available:
                logger.warning(
                    "Elastic license type %r does not grant native RRF; using Python RRF fallback.",
                    license_type or "<unknown>",
                )

        self._native_rrf_available = available
        return available

    @property
    def search_config(self) -> ElasticSearchConfig:
        """Get the instance search configuration."""
        return self._search_config

    @property
    def reranker(self) -> RerankClient | None:
        """Get the reranker client if configured."""
        return self._reranker

    @property
    def expander(self) -> QueryExpanderProtocol | None:
        """Get the query expander if configured.

        Factories call this to validate that a non-PASS ``expansion_hint`` is
        paired with a wired expander (per v3 amendment §3.5)."""
        return self._expander

    @property
    def domain_intent(self) -> str | None:
        """Build-time "what this index is about" — threaded into every ``expand()``
        call as the ``domain_intent`` kwarg. Runtime per-call intent comes from
        ``ElasticSearchConfig.intent`` / ``ElasticRetrieverContext.retriever_intent``.
        """
        return self._domain_intent

    # ---- Cycle F3 — AUTO BM25 probe ----------------------------------------

    @staticmethod
    def _chunk_score(chunk: ElasticRetrievalChunk) -> float:
        """Safe score accessor — ``None`` treated as 0.0."""
        return chunk.score if chunk.score is not None else 0.0

    def _is_strong_signal(
        self,
        fts_results: list[ElasticRetrievalChunk],
        cfg: ElasticSearchConfig,
    ) -> bool:
        """Reuses the base FTS results — no extra ES round-trip.

        Strong signal iff the top hit clears ``auto_probe_min_score`` AND
        the gap to the runner-up clears ``auto_probe_min_gap``. Both
        conditions must hold — a high single score with a tight cluster
        below it is ambiguous, not decisive.

        With the default ``+inf`` thresholds this function returns ``False``
        for every finite score, so AUTO always resolves to MULTI until F6
        calibrates per-corpus values.
        """
        if not fts_results:
            return False
        top = self._chunk_score(fts_results[0])
        second = self._chunk_score(fts_results[1]) if len(fts_results) > 1 else 0.0
        return top >= cfg.auto_probe_min_score and (top - second) >= cfg.auto_probe_min_gap

    def _resolve_auto_hint(
        self,
        hint: ExpansionStrategy,
        base_fts_results: list[ElasticRetrievalChunk],
        cfg: ElasticSearchConfig,
    ) -> ExpansionStrategy:
        """Map AUTO → PASS (strong signal) or MULTI (weak signal).

        Non-AUTO hints pass through untouched — the retriever never
        "upgrades" an explicit PASS or a specific strategy.
        """
        if hint != ExpansionStrategy.AUTO:
            return hint
        if self._is_strong_signal(base_fts_results, cfg):
            return ExpansionStrategy.PASS
        return ExpansionStrategy.MULTI

    @staticmethod
    def _rerank_document_text(chunk: ElasticRetrievalChunk) -> str:
        """Build the text handed to the cross-encoder for one candidate.

        The reranker must score on the SAME signal BM25 matches on. BM25
        queries ``content_field`` — the full structured ``metadata.content``
        blob (contextual summary + Url/Application/apcode/appName/title block +
        body). The result mapper splits that blob into a clean ``content``
        (body) plus a ``context_summary`` (the prefix); passing only ``content``
        to ``arerank`` would blind the cross-encoder to the metadata text that
        often disambiguates which app/entity/page a generic-looking chunk
        belongs to.

        So reconstruct the full signal: prepend ``context_summary`` (when the
        mapper recovered it) to the body. Byte-exactness with the original blob
        is irrelevant to a cross-encoder — the metadata *text* being present is
        what matters. Legacy / un-templated indices (no recovered prefix) fall
        back to the body alone, exactly as before.
        """
        context_summary = chunk.metadata.get("context_summary")
        if context_summary:
            return f"{context_summary}\n\n{chunk.content}"
        return chunk.content

    @staticmethod
    def _apply_rerank_response(
        rerank_response: RerankResponse,
        candidates: list[ElasticRetrievalChunk],
    ) -> Iterator[tuple[int, RerankResult]]:
        """Yield ``(idx, result)`` for each VALID rerank result, dropping bad rows.

        Defensive contract shared by both rerank sites (`_execute_search`'s
        `RRF_RERANKER` case and `_maybe_build_rerank_scores`). Cross-encoder
        providers occasionally return out-of-range or duplicate indices; this
        helper logs and skips them so callers don't IndexError or double-score.
        """
        seen: set[int] = set()
        pool_size = len(candidates)
        for r in rerank_response.results:
            if not (0 <= r.index < pool_size):
                logger.warning(
                    "rerank: out-of-range index %d (pool size %d) — skipping",
                    r.index,
                    pool_size,
                )
                continue
            if r.index in seen:
                logger.warning(
                    "rerank: duplicate index %d — keeping first score, skipping duplicate",
                    r.index,
                )
                continue
            seen.add(r.index)
            yield r.index, r

    @traceable(
        run_type="embedding",
        name="query_embedding",
        tags=["embedding"],
        process_outputs=_process_embedding_outputs,
    )
    async def _get_query_embedding(self, query: str) -> list[float]:
        embeddings = await self.embedding_model.aembed_documents([query])
        return embeddings[0]

    def _default_result_mapper(self, hit: dict[str, Any]) -> ElasticRetrievalChunk:
        source = hit.get("_source", {})
        score = hit.get("_score", 0.0) or 0.0

        # Display content:
        #   1. If ``display_content_field`` is configured, read that directly
        #      (legacy dual-content indices with a separate raw-display field).
        #   2. Otherwise read ``content_field`` (prod = ``metadata.content``,
        #      the structured BM25 blob) and run ``extract_chunk_body()`` to
        #      recover the body for display. Falls back to the full blob when
        #      the prod template anchor is absent (legacy un-templated docs).
        context_summary = ""
        if self.field_config.display_content_field:
            content = self._get_nested_field(source, self.field_config.display_content_field, "")
        else:
            raw = self._get_nested_field(source, self.field_config.content_field, "")
            if isinstance(raw, str):
                context_summary, content = parse_structured_content(raw, self.field_config.content_body_anchor)
            else:
                content = raw

        metadata: dict[str, Any] = {}
        metadata["title"] = self._get_nested_field(source, self.field_config.title_field, "Untitled")
        metadata["doc"] = self._get_nested_field(source, self.field_config.doc_field, "")
        url = self._get_nested_field(source, self.field_config.url_field, "")
        metadata["url"] = url
        # Per-page contextual prefix recovered from the structured blob. Surfaced
        # once per page by the Knowledge Agent rather than repeated inside every
        # chunk body. Identical across all chunks of a page; absent for
        # un-templated / legacy raw content (no anchor → empty summary).
        if context_summary:
            metadata["context_summary"] = context_summary

        # Per-chunk contextual summary, surfaced under the canonical snake_case
        # key. Production stores the same summary BOTH as the leading prefix of
        # ``content_field`` (inside ``context_summary`` above) AND verbatim in a
        # dedicated field (``metadata.contextualisedContent``). Normalizing it to
        # ``contextualized_content`` lets the Knowledge Agent render the per-chunk
        # summary INSIDE each ``<chunk>`` while keeping the page-shared
        # Url/Application/apcode/title block in ``context_summary`` once per page.
        # The raw backend leaf is intentionally NOT suppressed (unlike auid →
        # apcode): ``contextualisedContent`` is already a surfaced, consumer-read
        # key via the metadata tail-merge — removing it would break callers.
        if self.field_config.contextualized_content_field:
            contextualized = self._get_nested_field(source, self.field_config.contextualized_content_field, None)
            if isinstance(contextualized, str) and contextualized:
                metadata["contextualized_content"] = contextualized

        # Extract extended metadata when field paths are configured
        if self.field_config.page_id_field:
            page_id = self._get_nested_field(source, self.field_config.page_id_field, None)
            if page_id is not None:
                metadata["pageId"] = page_id
        if self.field_config.chunk_index_field:
            chunk_idx = self._get_nested_field(source, self.field_config.chunk_index_field, None)
            if chunk_idx is not None:
                metadata["chunk_index"] = chunk_idx
        if self.field_config.app_name_field:
            app_name = self._get_nested_field(source, self.field_config.app_name_field, None)
            if app_name is not None:
                metadata["appName"] = app_name
        # Backend leaf keys to suppress from the tail-merge: when we synthesize
        # the canonical name from a configured field path, the raw backend key
        # at the corresponding ``metadata.*`` leaf must NOT also leak through —
        # per ``sta_agent_core/AGENTS.md`` "normalize to canonical keys at the
        # boundary". Specifically, ``metadata.auid`` should not appear next to
        # synthesized ``metadata["apcode"]``.
        suppressed_backend_keys: set[str] = set()
        if self.field_config.apcode_field:
            # Concept name ``apcode`` stays on the consumer surface; backend
            # path lives on ``field_config.apcode_field`` (= ``metadata.auid``
            # in prod, ``metadata.apcode`` in legacy indices).
            apcode = self._get_nested_field(source, self.field_config.apcode_field, None)
            if apcode is not None:
                metadata["apcode"] = apcode
            backend_leaf = self.field_config.apcode_field.rsplit(".", 1)[-1]
            if backend_leaf != "apcode":
                # Only suppress backend-named leaves (auid). When the legacy
                # path "metadata.apcode" is configured, the leaf is already
                # the canonical name and the existing ``if key not in metadata``
                # guard does the right thing — don't add to the suppression
                # set or we'd block our own synthesized key.
                suppressed_backend_keys.add(backend_leaf)
        for freshness_field, canonical_key in (
            (self.field_config.last_doc_update_field, "lastDocUpdate"),
            (self.field_config.last_doc_ingestion_field, "lastDocIngestion"),
        ):
            if not freshness_field:
                continue
            raw_timestamp = self._get_nested_field(source, freshness_field, None)
            if raw_timestamp is not None:
                parsed_timestamp = parse_ingestion_timestamp(raw_timestamp)
                # Canonical ISO form when parseable; raw passthrough otherwise
                # so the value is never silently dropped (downstream staleness
                # rendering skips what it cannot parse).
                metadata[canonical_key] = parsed_timestamp.isoformat() if parsed_timestamp is not None else raw_timestamp
            timestamp_leaf = freshness_field.rsplit(".", 1)[-1]
            if timestamp_leaf != canonical_key:
                suppressed_backend_keys.add(timestamp_leaf)
        # Entity extraction uses ``entity_object_field`` (the parent dict path)
        # so consumers see the full ``{name, id, childs, is_opal}`` object.
        # ``entity_field`` (the ``.name`` leaf) stays reserved for aggregations
        # and BM25 boost clauses — using it here would return just the name
        # string, breaking consumers that read ``chunk.metadata["entity"]["id"]``
        # (see ``infra/elasticsearch/probes/metadata_scope_smoke.py``).
        if self.field_config.entity_object_field:
            entity = self._get_nested_field(source, self.field_config.entity_object_field, None)
            if entity is not None:
                metadata["entity"] = entity

        # Merge remaining metadata.* fields not already captured. Backend-named
        # leaves we just normalized into canonical keys are filtered out so the
        # consumer doesn't see both names side by side.
        meta_obj = source.get("metadata", {})
        if isinstance(meta_obj, dict):
            for key, value in meta_obj.items():
                if key in metadata or key in suppressed_backend_keys:
                    continue
                metadata[key] = value

        chunk_id = str(hit.get("_id", ""))
        source_url = url if isinstance(url, str) else ""
        return ElasticRetrievalChunk(
            content=content,
            chunk_id=chunk_id,
            score=score,
            source_url=source_url,
            retriever_type="elasticsearch",
            metadata=metadata,
        )

    def _get_nested_field(self, source: dict[str, Any], field_path: str, default: Any) -> Any:
        parts = field_path.split(".")
        current = source
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def _parse_response(self, response: dict[str, Any]) -> list[ElasticRetrievalChunk]:
        hits = response.get("hits", {}).get("hits", [])
        return [self._default_result_mapper(hit) for hit in hits]

    def _build_document_filter(self, documents: list[str]) -> dict[str, Any]:
        return {"terms": {self.field_config.doc_keyword_field: documents}}

    def _metadata_field_map(self) -> dict[str, str | None]:
        """Map MetadataScope axis/filter keys → ES field paths from this retriever's field_config.

        ``doc`` is the filter-only document-id key (``MetadataScope.doc_filter``)
        — it maps to the exact-match keyword field, distinct from the analyzed
        ``doc_field`` used for BM25 title/body matching.
        """
        return {
            "entity_id": self.field_config.entity_id_field,
            "entity_name": self.field_config.entity_field,
            "entity_childs": self.field_config.entity_childs_field,
            "apcode": self.field_config.apcode_field,
            "app_name": self.field_config.app_name_field,
            "doc": self.field_config.doc_keyword_field,
        }

    def _compose_filter_query(
        self,
        documents: list[str] | None,
        metadata_scope: MetadataScope | None,
    ) -> dict[str, Any] | None:
        """Merge the ``documents`` terms filter and all ``metadata_scope`` filter
        clauses into a single bool-filter dict. Returns ``None`` when neither
        applies, so callers can skip filter wiring entirely.
        """
        filter_clauses: list[dict[str, Any]] = []
        if documents:
            filter_clauses.append(self._build_document_filter(documents))
        if metadata_scope is not None:
            filter_clauses.extend(metadata_scope.build_filter_clauses(self._metadata_field_map(), self.field_config.scope_normalizers))
        if not filter_clauses:
            return None
        if len(filter_clauses) == 1:
            return filter_clauses[0]
        return {"bool": {"filter": filter_clauses}}

    def _build_dense_vector_query(
        self,
        query_embedding: list[float],
        k: int = 10,
        num_candidates: int | None = None,
        boost: float | None = None,
        filter_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        knn_query: dict[str, Any] = {
            "field": self.field_config.embedding_field,
            "query_vector": query_embedding,
            "k": k,
            "num_candidates": num_candidates or k * 2,
        }
        if boost is not None:
            knn_query["boost"] = boost
        if filter_query is not None:
            knn_query["filter"] = filter_query
        return knn_query

    def _build_sparse_vector_query(
        self,
        query: str,
        boost: float | None = None,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        content_field = self.field_config.content_field
        title_field = self.field_config.title_field
        doc_field = self.field_config.doc_field
        title_boost_ratio = self.field_config.title_boost_ratio
        doc_boost_ratio = self.field_config.doc_boost_ratio
        content_boost = boost if boost is not None else 1.0
        title_boost = content_boost * title_boost_ratio
        doc_boost = content_boost * doc_boost_ratio
        text_clauses: list[dict[str, Any]] = [
            {"match": {content_field: {"query": query, "boost": content_boost}}},
            {"match": {title_field: {"query": query, "boost": title_boost}}},
            {"match": {doc_field: {"query": query, "boost": doc_boost}}},
        ]
        if enable_fuzzy:
            fuzzy_content_boost = content_boost * fuzzy_boost_ratio
            fuzzy_title_boost = title_boost * fuzzy_boost_ratio
            text_clauses.extend(
                [
                    {"match": {content_field: {"query": query, "fuzziness": "AUTO", "boost": fuzzy_content_boost}}},
                    {"match": {title_field: {"query": query, "fuzziness": "AUTO", "boost": fuzzy_title_boost}}},
                ]
            )
        bool_body: dict[str, Any] = {}
        if metadata_boost_clauses:
            # ES defaults minimum_should_match to 0 when a bool has a filter or must. If text
            # and boost clauses shared a single outer should, a doc matching only the boost
            # (no text match) would be admitted. Wrap text in bool.must with an inner
            # minimum_should_match=1 so metadata boosts stay score-only.
            bool_body["must"] = [{"bool": {"should": text_clauses, "minimum_should_match": 1}}]
            bool_body["should"] = list(metadata_boost_clauses)
        else:
            bool_body["should"] = text_clauses
        if filter_query is not None:
            bool_body["filter"] = filter_query
        return {"bool": bool_body}

    def _build_rrf_query(
        self,
        query: str,
        query_embedding: list[float],
        size: int = 10,
        rank_window_size: int = 50,
        rank_constant: int = 60,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        sparse_query = self._build_sparse_vector_query(
            query,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        dense_query = self._build_dense_vector_query(
            query_embedding, k=rank_window_size, num_candidates=rank_window_size * 2, filter_query=filter_query
        )
        return {
            "size": size,
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {"standard": {"query": sparse_query}},
                        {"knn": dense_query},
                    ],
                    "rank_window_size": rank_window_size,
                    "rank_constant": rank_constant,
                }
            },
        }

    def _build_boost_query(
        self,
        query: str,
        query_embedding: list[float],
        size: int = 10,
        knn_boost: float = 0.7,
        bm25_boost: float = 0.3,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        sparse_query = self._build_sparse_vector_query(
            query,
            boost=bm25_boost,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        dense_query = self._build_dense_vector_query(query_embedding, k=size, num_candidates=size * 10, boost=knn_boost, filter_query=filter_query)
        return {"size": size, "query": sparse_query, "knn": dense_query}

    def _build_dense_only_query(
        self,
        query_embedding: list[float],
        size: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dense_query = self._build_dense_vector_query(query_embedding, k=size, num_candidates=size * 2, filter_query=filter_query)
        return {"size": size, "knn": dense_query}

    def _build_sparse_only_query(
        self,
        query: str,
        size: int = 10,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        sparse_query = self._build_sparse_vector_query(
            query,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        return {"size": size, "query": sparse_query}

    async def _get_sparse_and_dense_candidates(
        self,
        query: str,
        query_embedding: list[float],
        retrieval_size: int,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> tuple[list[ElasticRetrievalChunk], list[ElasticRetrievalChunk]]:
        """Fetch BM25 + kNN candidate lists in parallel, no merging.

        Used by RRF_ONLY / RRF_RERANKER paths where the two ranked lists
        must stay separate so RRF can see their individual rank positions.
        """
        sparse_body = self._build_sparse_only_query(
            query,
            size=retrieval_size,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        dense_body = self._build_dense_only_query(query_embedding, size=retrieval_size, filter_query=filter_query)
        sparse_response, dense_response = await asyncio.gather(
            self.adapter.search(index=self.index, body=sparse_body),
            self.adapter.search(index=self.index, body=dense_body),
        )
        return self._parse_response(sparse_response), self._parse_response(dense_response)

    async def _rrf_fuse(
        self,
        query: str,
        query_embedding: list[float],
        size: int,
        rank_window_size: int,
        rank_constant: int,
        retrieval_size: int,
        filter_query: dict[str, Any] | None,
        enable_fuzzy: bool,
        fuzzy_boost_ratio: float,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> list[ElasticRetrievalChunk]:
        """Run RRF fusion — native ES query if licensed, in-process Python RRF otherwise.

        Returns top ``size`` docs ranked by RRF score.
        """
        if await self._can_use_native_rrf():
            search_body = self._build_rrf_query(
                query,
                query_embedding,
                size,
                rank_window_size,
                rank_constant,
                filter_query,
                enable_fuzzy=enable_fuzzy,
                fuzzy_boost_ratio=fuzzy_boost_ratio,
                metadata_boost_clauses=metadata_boost_clauses,
            )
            response = await self.adapter.search(index=self.index, body=search_body)
            return self._parse_response(response)

        # Python fallback: fetch both lists, fuse in-process.
        sparse_results, dense_results = await self._get_sparse_and_dense_candidates(
            query=query,
            query_embedding=query_embedding,
            retrieval_size=retrieval_size,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        return WeightedRRF(rank_constant=rank_constant).fuse(
            ranked_lists=[sparse_results, dense_results],
            weights=[1.0, 1.0],
            size=size,
        )

    async def _get_candidates_for_reranking(
        self,
        query: str,
        query_embedding: list[float],
        retrieval_size: int,
        filter_query: dict[str, Any] | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
        metadata_boost_clauses: list[dict[str, Any]] | None = None,
    ) -> list[ElasticRetrievalChunk]:
        sparse_body = self._build_sparse_only_query(
            query,
            size=retrieval_size,
            filter_query=filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        dense_body = self._build_dense_only_query(query_embedding, size=retrieval_size, filter_query=filter_query)
        sparse_task = self.adapter.search(index=self.index, body=sparse_body)
        dense_task = self.adapter.search(index=self.index, body=dense_body)
        sparse_response, dense_response = await asyncio.gather(sparse_task, dense_task)
        sparse_results = self._parse_response(sparse_response)
        dense_results = self._parse_response(dense_response)
        return self._merge_candidates(dense_results, sparse_results)

    def _merge_candidates(self, primary: list[ElasticRetrievalChunk], secondary: list[ElasticRetrievalChunk]) -> list[ElasticRetrievalChunk]:
        """Merge two candidate lists, deduplicating by content hash."""
        seen_hashes: set[int] = set()
        merged: list[ElasticRetrievalChunk] = []
        for item in primary:
            h = hash(item.content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                merged.append(item)
        for item in secondary:
            h = hash(item.content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                merged.append(item)
        return merged

    @traceable(
        run_type="retriever",
        name="elasticsearch_hybrid_search",
        process_inputs=_process_retriever_inputs,
        process_outputs=_process_retriever_outputs,
    )
    async def search(
        self,
        query: str,
        size: int = 10,
        *,
        documents: list[str] | None = None,
        fusion_strategy: FusionStrategy | FusionStrategyLiteral | None = None,
        rank_window_size: int | None = None,  # noqa: ARG002 — deprecated synonym for retrieval_size; kept for API parity
        rank_constant: int | None = None,
        retrieval_size: int | None = None,
        enable_fuzzy: bool | None = None,
        fuzzy_boost_ratio: float | None = None,
        metadata_scope: MetadataScope | None = None,
        rerank_top_n: int | None = None,
        expansion_hint: ExpansionStrategy | str | None = None,
        bm25_rrf_weight: float | None = None,
        knn_rrf_weight: float | None = None,
        intent: str | None = None,
        auto_probe_min_score: float | None = None,
        auto_probe_min_gap: float | None = None,
        **kwargs: Any,  # noqa: ARG002 — unknown kwargs silently dropped for caller forward-compat
    ) -> SearchResponse[ElasticRetrievalChunk]:
        """Hybrid BM25+kNN search with optional query expansion and pluggable fusion.

        Phase 5 Cycle E replaced the legacy per-strategy match dispatch with a
        single uniform pipeline:

        1. Seed two SubQueries (``lex`` + ``vec``) from the original query.
        2. If ``expansion_hint != PASS``, append the expander's variants —
           seeds always first, because ``_maybe_build_rerank_scores`` reads
           ``sub_queries[0].query`` as the reranker input (rerank ordering
           contract, v3 §3.3 / ``_maybe_build_rerank_scores:1087``).
        3. For ``WEIGHTED_RRF``, flow the config's ``bm25_rrf_weight`` /
           ``knn_rrf_weight`` into ``SubQuery.weight``. Other strategies use
           uniform 1.0 weights — weighting is a property of the sub-queries,
           not the operator (v3 lock).
        4. Resolve the ``FusionOperator`` via ``resolve_fusion_operator``.
        5. Delegate to ``search_many`` — which owns the ``_run_one`` fan-out
           and ``_maybe_build_rerank_scores`` side-channel.

        The native ES RRF fast-path was dropped — with ``asyncio.gather`` the
        two sub-queries run in parallel (wall-clock ≈ ``max(t_bm25, t_knn)``),
        so the single-RTT savings don't justify a separate code path.
        """
        resolved = self._search_config.resolve_params(
            size=size,
            fusion_strategy=fusion_strategy,
            rank_constant=rank_constant,
            retrieval_size=retrieval_size,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            rerank_top_n=rerank_top_n,
            expansion_hint=expansion_hint,
            bm25_rrf_weight=bm25_rrf_weight,
            knn_rrf_weight=knn_rrf_weight,
            intent=intent,
            auto_probe_min_score=auto_probe_min_score,
            auto_probe_min_gap=auto_probe_min_gap,
        )
        strategy = FusionStrategy(resolved["fusion_strategy"])
        hint = ExpansionStrategy(resolved["expansion_hint"])
        bm25_weight = float(resolved["bm25_rrf_weight"])
        knn_weight = float(resolved["knn_rrf_weight"])

        # Resolve AUTO before the expander is touched — the expander itself
        # raises on AUTO (it has no BM25 access). Probe is a tiny size=2
        # BM25 call whose top-2 scores drive ``_resolve_auto_hint``. Skipped
        # entirely for non-AUTO hints so the normal path stays one-RTT-cheap.
        if hint == ExpansionStrategy.AUTO:
            probe_results = await self.search_text_only(
                query,
                size=2,
                documents=documents,
                metadata_scope=metadata_scope,
                enable_fuzzy=resolved["enable_fuzzy"],
                fuzzy_boost_ratio=resolved["fuzzy_boost_ratio"],
            )
            # Use the resolved thresholds (explicit kwargs > context overrides
            # > instance config) rather than reading ``self._search_config``
            # directly, so per-call ``retriever_auto_probe_*`` overrides take
            # effect even on the generic ``search(**cfg.to_search_kwargs())``
            # dispatch path (Cycle F6c hotfix).
            probe_cfg = dataclasses.replace(
                self._search_config,
                auto_probe_min_score=float(resolved["auto_probe_min_score"]),
                auto_probe_min_gap=float(resolved["auto_probe_min_gap"]),
            )
            hint = self._resolve_auto_hint(ExpansionStrategy.AUTO, probe_results, probe_cfg)

        seed_weights = (bm25_weight, knn_weight) if strategy == FusionStrategy.WEIGHTED_RRF else (1.0, 1.0)
        sub_queries: list[SubQuery] = [
            SubQuery(type="lex", query=query, weight=seed_weights[0]),
            SubQuery(type="vec", query=query, weight=seed_weights[1]),
        ]

        if hint != ExpansionStrategy.PASS:
            if self._expander is None:
                # v3 §3.5 defense-in-depth — factory should have caught this
                # at wire-up; the runtime guard keeps the contract loud.
                raise ValueError(
                    f"expansion_hint={hint.value!r} requires an expander — "
                    f"construct ElasticRetriever with `expander=QueryExpander(...)` "
                    f"or set expansion_hint=PASS."
                )
            # Thread both intent layers through to the expander — build-time
            # ``domain_intent`` from ctor, runtime ``intent`` resolved via
            # ``resolve_params`` so an explicit ``search(intent=...)`` kwarg
            # wins over the instance config's ``intent`` (which itself came
            # from ``retriever_intent`` via ``from_context`` introspection).
            sub_queries.extend(
                await self._expander.expand(
                    query,
                    hint,
                    domain_intent=self._domain_intent,
                    intent=resolved["intent"],
                )
            )

        operator = resolve_fusion_operator(
            strategy.value,
            rank_constant=resolved["rank_constant"],
            bm25_rrf_weight=bm25_weight,
            knn_rrf_weight=knn_weight,
        )

        # Reranker query prepend (F5b-3) — makes both intent layers visible to
        # the cross-encoder. Empty layers are elided via filter(None, …) so a
        # missing intent doesn't leave a stray newline. Built here (not in
        # search_many) so search_many stays intent-agnostic for non-expansion
        # callers that want to pre-build their own SubQuery list.
        rerank_query_override = _join_rerank_query(self._domain_intent, resolved["intent"], query)

        return await self.search_many(
            queries=sub_queries,
            size=resolved["size"],
            fusion=operator,
            documents=documents,
            metadata_scope=metadata_scope,
            rerank_top_n=resolved["rerank_top_n"],
            enable_fuzzy=resolved["enable_fuzzy"],
            fuzzy_boost_ratio=resolved["fuzzy_boost_ratio"],
            rank_constant=resolved["rank_constant"],
            retrieval_size=resolved["retrieval_size"],
            rerank_query_override=rerank_query_override,
        )

    async def search_vector_only(
        self,
        query: str,
        size: int = 10,
        *,
        documents: list[str] | None = None,
        metadata_scope: MetadataScope | None = None,
    ) -> list[ElasticRetrievalChunk]:
        """Perform vector-only search (pure semantic / dense)."""
        query_embedding = await self._get_query_embedding(query)
        filter_query = self._compose_filter_query(documents, metadata_scope)
        search_body = self._build_dense_only_query(query_embedding, size, filter_query)
        response = await self.adapter.search(index=self.index, body=search_body)
        return self._parse_response(response)

    async def search_text_only(
        self,
        query: str,
        size: int = 10,
        *,
        documents: list[str] | None = None,
        metadata_scope: MetadataScope | None = None,
        enable_fuzzy: bool = False,
        fuzzy_boost_ratio: float = 0.25,
    ) -> list[ElasticRetrievalChunk]:
        """Perform text-only search (pure BM25 / sparse)."""
        filter_query = self._compose_filter_query(documents, metadata_scope)
        metadata_boost_clauses = (
            metadata_scope.build_boost_clauses(self._metadata_field_map(), self.field_config.scope_normalizers)
            if metadata_scope is not None
            else None
        ) or None
        search_body = self._build_sparse_only_query(
            query,
            size,
            filter_query,
            enable_fuzzy=enable_fuzzy,
            fuzzy_boost_ratio=fuzzy_boost_ratio,
            metadata_boost_clauses=metadata_boost_clauses,
        )
        response = await self.adapter.search(index=self.index, body=search_body)
        return self._parse_response(response)

    @traceable(
        run_type="retriever",
        name="elasticsearch_search_many",
        process_inputs=_process_search_many_inputs,
        process_outputs=_process_retriever_outputs,
    )
    async def search_many(
        self,
        queries: str | list[str] | list[SubQuery],
        size: int = 10,
        *,
        fusion: FusionOperator | None = None,
        documents: list[str] | None = None,
        metadata_scope: MetadataScope | None = None,
        rerank_top_n: int | None = None,
        enable_fuzzy: bool | None = None,
        fuzzy_boost_ratio: float | None = None,
        rank_window_size: int | None = None,  # noqa: ARG002 — deprecated synonym for retrieval_size; retained for API parity
        rank_constant: int | None = None,
        retrieval_size: int | None = None,
        rerank_query_override: str | None = None,
    ) -> SearchResponse[ElasticRetrievalChunk]:
        """Multi-variant retrieval with pluggable fusion.

        Fans out each sub-query to `search_text_only` (lex) or
        `search_vector_only` (vec/hyde) — each sub-query retrieves
        ``retrieval_size`` candidates (override via kwarg; else falls back to
        ``search_config.retrieval_size``, default 50) so fusion has a deep
        pool; the final output is truncated to ``size``. Concurrency is
        bounded by ``search_config.max_concurrent_subqueries``. Fusion uses
        the supplied ``FusionOperator`` (defaults to ``WeightedRRF``).

        Coercion rules:
          - ``str``           → [SubQuery(lex, q), SubQuery(vec, q)]
          - ``list[str]``     → each string coerced as above
          - ``list[SubQuery]``→ passthrough

        Mixed ``list[str | SubQuery]`` is rejected (``TypeError``) — the coercion
        is not defined and silent coercion would leak type confusion into fusion.

        The FIRST entry in the fused input is conventionally the "original"
        query's ranked list — strategies like ``TopRankBonusRRF`` use this to
        protect exact-match docs from expansion dilution.

        **Uniform ``documents`` filter**: when set, the same document-id filter
        is applied to every sub-query before fusion — per-variant filters are
        not supported. Put a per-variant filter on ``SubQuery`` itself in a
        future phase if the need arises.

        **Partial-failure tolerance**: individual sub-query failures are
        logged and their ranked list is dropped from fusion. If every
        sub-query fails, returns an empty response rather than raising.

        **``rank_constant`` resolution**: a bare ``WeightedRRF()``/
        ``TopRankBonusRRF()``/``PositionAwareBlend()`` (``rank_constant=None``)
        inherits the retriever's configured ``rank_constant`` from
        ``search_config``. To override, construct the operator with an
        explicit ``rank_constant``.
        """
        sub_queries = self._coerce_to_subqueries(queries)
        if not sub_queries:
            return SearchResponse(results=[])

        # Validate metadata_scope up-front — misconfig (an axis whose
        # field_map entry is None) raises ``ValueError`` from
        # ``build_filter_clauses`` / ``build_boost_clauses``. Without this
        # pre-check every sub-query would raise the same error below and
        # the ``return_exceptions=True`` gather would silently drop them
        # all, returning an empty SearchResponse — callers hit "no
        # results + a log warning" instead of the actual stack trace
        # pointing at the field_config misconfig. Cross-team-leakage
        # guard errors are non-recoverable and must propagate.
        if metadata_scope is not None and not metadata_scope.is_empty():
            field_map = self._metadata_field_map()
            normalizers = self.field_config.scope_normalizers
            metadata_scope.build_filter_clauses(field_map, normalizers)
            metadata_scope.build_boost_clauses(field_map, normalizers)

        sem = asyncio.Semaphore(self._search_config.max_concurrent_subqueries)
        per_query_size = retrieval_size if retrieval_size is not None else self._search_config.retrieval_size
        eff_enable_fuzzy = enable_fuzzy if enable_fuzzy is not None else False
        eff_fuzzy_boost = fuzzy_boost_ratio if fuzzy_boost_ratio is not None else 0.25

        async def _run_one(sq: SubQuery) -> list[ElasticRetrievalChunk]:
            async with sem:
                if sq.type == "lex":
                    return await self.search_text_only(
                        sq.query,
                        size=per_query_size,
                        documents=documents,
                        metadata_scope=metadata_scope,
                        enable_fuzzy=eff_enable_fuzzy,
                        fuzzy_boost_ratio=eff_fuzzy_boost,
                    )
                return await self.search_vector_only(
                    sq.query,
                    size=per_query_size,
                    documents=documents,
                    metadata_scope=metadata_scope,
                )

        raw_results = await asyncio.gather(*(_run_one(sq) for sq in sub_queries), return_exceptions=True)

        # Cancellation must NOT be treated as a sub-query failure — `gather(return_exceptions=True)`
        # captures `CancelledError` like any other exception, but the caller (e.g. a disconnected
        # client) needs the cancel to propagate, not a silent empty response.
        if any(isinstance(o, asyncio.CancelledError) for o in raw_results):
            raise asyncio.CancelledError

        # Drop failed sub-queries; keep their weights aligned with surviving lists.
        surviving_lists: list[list[ElasticRetrievalChunk]] = []
        surviving_weights: list[float] = []
        for sq, outcome in zip(sub_queries, raw_results, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning(
                    "search_many: sub-query (type=%s, q=%r) failed — dropping from fusion: %s",
                    sq.type,
                    sq.query[:80],
                    outcome,
                )
                continue
            surviving_lists.append(outcome)
            surviving_weights.append(sq.weight)

        if not surviving_lists:
            return SearchResponse(results=[])

        operator = self._resolve_fusion_operator(fusion, rank_constant_override=rank_constant)

        try:
            rerank_scores = await self._maybe_build_rerank_scores(
                operator=operator,
                sub_queries=sub_queries,
                surviving_lists=surviving_lists,
                surviving_weights=surviving_weights,
                rerank_top_n_override=rerank_top_n,
                size=size,
                rerank_query_override=rerank_query_override,
            )
        except RerankUnavailableError as exc:
            # Soft failure: the reranker errored after its retries. Rather than
            # return zero documents, degrade to rerank-blind RRF over the lists we
            # already retrieved. The reranker's own trace span still shows failed.
            fallback_k = rank_constant if rank_constant is not None else self._search_config.rank_constant
            logger.warning(
                "search_many: %s — falling back to RRF (rrf_only) over %d retrieved list(s).",
                exc,
                len(surviving_lists),
            )
            operator = WeightedRRF(rank_constant=fallback_k)
            rerank_scores = None

        fused = operator.fuse(
            ranked_lists=surviving_lists,
            weights=surviving_weights,
            size=size,
            rerank_scores=rerank_scores,
        )
        return SearchResponse(results=fused)

    async def _maybe_build_rerank_scores(
        self,
        *,
        operator: FusionOperator,
        sub_queries: list[SubQuery],
        surviving_lists: list[list[ElasticRetrievalChunk]],
        surviving_weights: list[float],
        rerank_top_n_override: int | None = None,
        size: int = 10,
        rerank_query_override: str | None = None,
    ) -> dict[tuple[str, str], float] | None:
        """Run reranking upstream when the fusion operator needs it.

        Only ``PositionAwareBlend`` is rerank-aware today. For rerank-blind
        operators (``WeightedRRF``, ``TopRankBonusRRF``) this returns ``None``
        without touching the reranker.

        Pipeline when rerank is needed:
          1. Pre-fuse the surviving sub-query lists with ``WeightedRRF`` to
             pick the top ``rerank_top_n`` candidates.
          2. Send those candidates (with the FIRST sub-query's ``.query`` as
             the rerank query) to the cross-encoder.
          3. Return ``{(page_id, chunk_id): relevance_score}`` — same key
             shape as ``fusion._dedup_key`` so the operator can look up.

        Raises ``ValueError`` with an actionable message if the caller picked
        a rerank-aware operator but no reranker is configured on the retriever.
        """
        if not getattr(operator, "requires_rerank_scores", False):
            return None

        if self._reranker is None:
            raise ValueError(
                f"{type(operator).__name__} requires a reranker on this ElasticRetriever — "
                "inject a RerankClient via the constructor, or choose a rerank-blind "
                "fusion operator (WeightedRRF, TopRankBonusRRF) if reranking is not wanted."
            )

        effective_rerank_top_n = rerank_top_n_override if rerank_top_n_override is not None else self._search_config.rerank_top_n
        rerank_top_n = max(effective_rerank_top_n, size)
        # Pre-rerank fuse uses the operator's resolved rank_constant so the
        # candidate pool is consistent with the rrf_rank the operator computes.
        op_k = getattr(operator, "rank_constant", None)
        pre_rrf_k = op_k if op_k is not None else self._search_config.rank_constant
        pre_rrf = WeightedRRF(rank_constant=pre_rrf_k).fuse(
            ranked_lists=surviving_lists,
            weights=surviving_weights,
            size=rerank_top_n,
        )
        if not pre_rrf:
            return {}

        # Rerank query — either the F5b-3 intent-prepended override from
        # ``search()`` or the first sub-query's raw text (convention: first
        # sub-query is the "original"; reranker never sees expansion variants).
        original_query = rerank_query_override if rerank_query_override is not None else sub_queries[0].query
        # Feed the reconstructed structured blob (contextual prefix + body), not
        # the body alone — parity with BM25, which matches the full content_field.
        docs_for_rerank = [self._rerank_document_text(c) for c in pre_rrf]
        # Clamp top_n to candidate count — some providers (Cohere et al.) 400 when
        # top_n > len(documents). Guarantees `arerank` receives a valid request even
        # when the fused pool is smaller than the configured rerank_top_n.
        #
        # The rerank call (which already retries internally via tenacity) is the
        # soft-failure boundary: on exhaustion the `reranker_arerank` LangSmith
        # span is recorded as failed, then we re-raise as RerankUnavailableError so
        # `search_many` can degrade to RRF instead of returning zero documents.
        try:
            rerank_response = await self._reranker.arerank(
                query=original_query,
                documents=docs_for_rerank,
                top_n=min(rerank_top_n, len(docs_for_rerank)),
                return_documents=False,
            )

            # Validation (out-of-range, duplicates) lives in _apply_rerank_response —
            # shared with _execute_search's RRF_RERANKER path.
            rerank_scores: dict[tuple[str, str], float] = {}
            for idx, r in self._apply_rerank_response(rerank_response, pre_rrf):
                candidate = pre_rrf[idx]
                key = (str(candidate.metadata.get("pageId") or ""), candidate.chunk_id or "")
                rerank_scores[key] = r.relevance_score
        except Exception as exc:
            raise RerankUnavailableError(f"rerank step failed for {type(operator).__name__}: {exc}") from exc
        return rerank_scores

    @classmethod
    def _coerce_to_subqueries(cls, queries: str | list[str] | list[SubQuery]) -> list[SubQuery]:
        """Apply Phase-1 coercion rules (see search_many docstring).

        Rejects mixed ``list[str | SubQuery]`` with ``TypeError`` — every
        element must be the same kind. Classmethod (not static) so subclasses
        or Phase-3 per-type-default-weight variants can override coercion
        while keeping the call-site unchanged.
        """
        if isinstance(queries, str):
            return [SubQuery(type="lex", query=queries), SubQuery(type="vec", query=queries)]
        if not queries:
            return []
        all_subq = all(isinstance(q, SubQuery) for q in queries)
        all_str = all(isinstance(q, str) for q in queries)
        if not (all_subq or all_str):
            raise TypeError(
                "search_many: queries must be list[str] OR list[SubQuery], not a mix — "
                "got types: " + ", ".join(sorted({type(q).__name__ for q in queries}))
            )
        if all_subq:
            return list(queries)  # type: ignore[arg-type]
        out: list[SubQuery] = []
        for q in queries:
            out.append(SubQuery(type="lex", query=q))  # type: ignore[arg-type]
            out.append(SubQuery(type="vec", query=q))  # type: ignore[arg-type]
        return out

    def _resolve_fusion_operator(
        self,
        fusion: FusionOperator | None,
        *,
        rank_constant_override: int | None = None,
    ) -> FusionOperator:
        """Resolve the fusion operator, injecting configured rank_constant when unset.

        Rules:
        - ``fusion is None`` → build ``WeightedRRF`` with the effective rank_constant.
        - User-supplied shipped operator with ``rank_constant=None`` → inject
          effective rank_constant via ``dataclasses.replace``.
        - User-supplied operator with an explicit ``rank_constant`` → passthrough.
        - Third-party ``FusionOperator`` without ``rank_constant`` → passthrough.

        ``rank_constant_override`` (from a per-call kwarg) takes precedence over
        ``search_config.rank_constant`` when set.
        """
        effective_k = rank_constant_override if rank_constant_override is not None else self._search_config.rank_constant
        if fusion is None:
            return WeightedRRF(rank_constant=effective_k)
        if isinstance(fusion, (WeightedRRF, TopRankBonusRRF, PositionAwareBlend, RrfRerankerOperator)) and fusion.rank_constant is None:
            return dataclasses.replace(fusion, rank_constant=effective_k)
        return fusion

    # --- DocumentProvider implementation ---

    @property
    def supports_document_provider(self) -> bool:
        """Whether this retriever can act as a ``DocumentProvider``.

        Requires both ``page_id_field`` and ``chunk_index_field`` to be
        configured in the ``ElasticFieldConfig``.
        """
        return self.field_config.page_id_field is not None and self.field_config.chunk_index_field is not None

    def _require_document_provider(self) -> None:
        """Raise if DocumentProvider prerequisites are not met."""
        if not self.supports_document_provider:
            raise NotImplementedError("DocumentProvider requires page_id_field and chunk_index_field to be set in ElasticFieldConfig.")

    @traceable(
        run_type="retriever",
        name="elastic_get_document",
        process_inputs=_process_get_document_inputs,
        process_outputs=_process_get_document_outputs,
    )
    async def get_document(self, doc_id: str) -> list[ElasticRetrievalChunk]:
        """Fetch all chunks of a document, ordered by ``chunk_index``.

        Args:
            doc_id: Document-level identifier (e.g. pageId).

        Returns:
            Ordered list of chunks for the document.
        """
        self._require_document_provider()
        body: dict[str, Any] = {
            "query": {"term": {self.field_config.page_id_field: doc_id}},
            "sort": [{self.field_config.chunk_index_field: "asc"}],
            "size": 10_000,
        }
        response = await self.adapter.search(index=self.index, body=body)
        return self._parse_response(response)

    @traceable(
        run_type="retriever",
        name="elastic_get_chunk_context",
        process_inputs=_process_get_chunk_context_inputs,
        process_outputs=_process_get_document_outputs,
    )
    async def get_chunk_context(self, chunk_id: str, window: int = 3) -> list[ElasticRetrievalChunk]:
        """Fetch neighbouring chunks around a given chunk.

        Performs a two-step lookup: (1) fetch the anchor chunk by ES ``_id``
        to resolve its ``pageId`` and ``chunk_index``, then (2) range-query
        for surrounding chunks.

        Args:
            chunk_id: The Elasticsearch ``_id`` of the anchor chunk.
            window: Number of chunks before and after to include.

        Returns:
            Chunks in the window, ordered by ``chunk_index``.
        """
        self._require_document_provider()

        # Step 1: resolve anchor chunk's document identity
        # Uses search + ids query instead of GET API so that aliases
        # pointing to multiple indices are handled correctly.
        anchor_resp = await self.adapter.search(
            index=self.index,
            body={"query": {"ids": {"values": [chunk_id]}}, "size": 1},
        )
        hits = anchor_resp.get("hits", {}).get("hits", [])
        if not hits:
            return []
        source = hits[0].get("_source", {})
        page_id = self._get_nested_field(source, self.field_config.page_id_field, None)  # type: ignore[arg-type]
        chunk_idx = self._get_nested_field(source, self.field_config.chunk_index_field, None)  # type: ignore[arg-type]

        if page_id is None or chunk_idx is None:
            return []

        # Step 2: range query for [chunk_index - window, chunk_index + window]
        return await self.get_chunk_range(
            doc_id=str(page_id),
            start_index=max(0, int(chunk_idx) - window),
            end_index=int(chunk_idx) + window,
        )

    @traceable(
        run_type="retriever",
        name="elastic_get_chunk_range",
        process_inputs=_process_get_chunk_range_inputs,
        process_outputs=_process_get_document_outputs,
    )
    async def get_chunk_range(self, doc_id: str, start_index: int, end_index: int) -> list[ElasticRetrievalChunk]:
        """Fetch a range of chunks from a document.

        Args:
            doc_id: Document-level identifier.
            start_index: First ``chunk_index`` (inclusive).
            end_index: Last ``chunk_index`` (inclusive).

        Returns:
            Chunks in the range, ordered by ``chunk_index``.
        """
        self._require_document_provider()
        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {self.field_config.page_id_field: doc_id}},
                        {
                            "range": {
                                self.field_config.chunk_index_field: {
                                    "gte": start_index,
                                    "lte": end_index,
                                }
                            }
                        },
                    ]
                }
            },
            "sort": [{self.field_config.chunk_index_field: "asc"}],
            "size": end_index - start_index + 1,
        }
        response = await self.adapter.search(index=self.index, body=body)
        return self._parse_response(response)

    # --- SupportsBatchFetch implementation ---

    @traceable(
        run_type="retriever",
        name="elastic_get_documents",
    )
    async def get_documents(self, doc_ids: list[str]) -> dict[str, list[ElasticRetrievalChunk]]:
        """Fetch all chunks for multiple documents in a single ``terms`` query.

        Args:
            doc_ids: Document-level identifiers (e.g. pageIds).

        Returns:
            Mapping ``doc_id -> chunks`` ordered by ``chunk_index``. Every input
            id is a key; an unknown document maps to an empty list.
        """
        self._require_document_provider()
        if not doc_ids:
            return {}
        unique_ids = list(dict.fromkeys(doc_ids))
        body: dict[str, Any] = {
            "query": {"terms": {self.field_config.page_id_field: unique_ids}},
            "sort": [{self.field_config.chunk_index_field: "asc"}],
            "size": _BATCH_FETCH_MAX_CHUNKS,
        }
        response = await self.adapter.search(index=self.index, body=body)
        chunks = self._parse_response(response)
        if len(chunks) >= _BATCH_FETCH_MAX_CHUNKS:
            logger.warning(
                "get_documents: hit the %d-chunk batch ceiling for %d documents — results may be truncated",
                _BATCH_FETCH_MAX_CHUNKS,
                len(unique_ids),
            )
        result: dict[str, list[ElasticRetrievalChunk]] = {doc_id: [] for doc_id in unique_ids}
        for chunk in chunks:
            page_id = chunk.metadata.get("pageId")
            if page_id is not None and str(page_id) in result:
                result[str(page_id)].append(chunk)
        return result

    @traceable(
        run_type="retriever",
        name="elastic_get_chunk_ranges",
    )
    async def get_chunk_ranges(self, ranges: list[ChunkRange]) -> dict[ChunkRange, list[ElasticRetrievalChunk]]:
        """Fetch multiple chunk ranges in a single ``bool/should`` query.

        Args:
            ranges: ``(doc_id, start_index, end_index)`` tuples, inclusive.

        Returns:
            Mapping ``range -> chunks``. Every input range is a key. A chunk
            that falls inside several (overlapping) ranges is returned under
            each of them.
        """
        self._require_document_provider()
        if not ranges:
            return {}
        should: list[dict[str, Any]] = []
        total = 0
        for doc_id, start, end in ranges:
            should.append(
                {
                    "bool": {
                        "must": [
                            {"term": {self.field_config.page_id_field: doc_id}},
                            {"range": {self.field_config.chunk_index_field: {"gte": start, "lte": end}}},
                        ]
                    }
                }
            )
            total += max(0, end - start + 1)
        if total > _BATCH_FETCH_MAX_CHUNKS:
            logger.warning(
                "get_chunk_ranges: %d requested chunks across %d ranges exceeds the %d-chunk batch ceiling — tail ranges may be silently truncated",
                total,
                len(ranges),
                _BATCH_FETCH_MAX_CHUNKS,
            )
        body: dict[str, Any] = {
            "query": {"bool": {"should": should, "minimum_should_match": 1}},
            "sort": [{self.field_config.chunk_index_field: "asc"}],
            "size": min(total, _BATCH_FETCH_MAX_CHUNKS) or 1,
        }
        response = await self.adapter.search(index=self.index, body=body)
        chunks = self._parse_response(response)
        result: dict[ChunkRange, list[ElasticRetrievalChunk]] = {r: [] for r in ranges}
        for chunk in chunks:
            page_id = chunk.metadata.get("pageId")
            chunk_idx = chunk.metadata.get("chunk_index")
            if page_id is None or chunk_idx is None:
                continue
            for r in result:
                r_doc, r_start, r_end = r
                if str(page_id) == str(r_doc) and r_start <= int(chunk_idx) <= r_end:
                    result[r].append(chunk)
        return result

    @traceable(
        run_type="retriever",
        name="elastic_get_chunk_contexts",
    )
    async def get_chunk_contexts(self, chunk_ids: list[str], window: int = 3) -> dict[str, list[ElasticRetrievalChunk]]:
        """Fetch neighbouring chunks around multiple anchor chunks.

        Resolves all anchors with one ``ids`` query, then issues a single
        batched range query for the surrounding windows.

        Args:
            chunk_ids: Elasticsearch ``_id`` values of the anchor chunks.
            window: Number of chunks before and after each anchor to include.

        Returns:
            Mapping ``chunk_id -> context chunks``. An anchor that cannot be
            resolved maps to an empty list.
        """
        self._require_document_provider()
        if not chunk_ids:
            return {}
        unique_ids = list(dict.fromkeys(chunk_ids))
        result: dict[str, list[ElasticRetrievalChunk]] = {cid: [] for cid in unique_ids}

        anchor_resp = await self.adapter.search(
            index=self.index,
            body={"query": {"ids": {"values": unique_ids}}, "size": len(unique_ids)},
        )
        anchors = self._parse_response(anchor_resp)

        # Resolve each anchor to its (page_id, chunk_index) and build a window range.
        range_for_id: dict[str, ChunkRange] = {}
        for chunk in anchors:
            page_id = chunk.metadata.get("pageId")
            chunk_idx = chunk.metadata.get("chunk_index")
            if not chunk.chunk_id or page_id is None or chunk_idx is None:
                continue
            idx = int(chunk_idx)
            range_for_id[chunk.chunk_id] = (str(page_id), max(0, idx - window), idx + window)

        if not range_for_id:
            return result

        range_results = await self.get_chunk_ranges(list(range_for_id.values()))
        for cid, r in range_for_id.items():
            result[cid] = range_results.get(r, [])
        return result

    async def close(self) -> None:
        """Cascade close across all owned clients; log failures without re-raising.

        Resources closed (best-effort): the ES adapter, the embedding httpx
        client (if injected), and the reranker. A failure in any one must not
        prevent the others from being released — using
        ``asyncio.gather(return_exceptions=True)`` keeps all coroutines
        scheduled and surfaces individual failures via warning logs.
        """
        tasks: list[tuple[str, Any]] = [("adapter", self.adapter.close())]
        if self._embedding_http_client is not None:
            tasks.append(("embedding_http_client", self._embedding_http_client.aclose()))
        if self._reranker is not None:
            tasks.append(("reranker", self._reranker.close()))

        results = await asyncio.gather(*(coro for _, coro in tasks), return_exceptions=True)
        for (label, _), result in zip(tasks, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("ElasticRetriever.close: %s failed: %r", label, result)

-------

packages/sta_agent_core/src/sta_agent_core/repositories/retrievers/elasticsearch/elastic_search_config.py
----
"""Elasticsearch-specific search configuration and context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, get_args

from ..base_search_config import BaseRetrieverContext, BaseSearchConfig
from .metadata_scope import Normalizer, ScopeAxis
from .query_expansion import ExpansionStrategy


# Valid fusion strategy strings — type checker rejects invalid values (e.g. "garbage").
# Keep in sync with FusionStrategy enum below: values must match FusionStrategy member values.
FusionStrategyLiteral = Literal[
    "rrf_only",
    "rrf_reranker",
    "weighted_rrf",
    "top_rank_bonus",
    "position_blend",
    "none",
    # Deprecated aliases (kept in the literal so existing string-typed callers
    # still pass type checking; runtime policy decides what each one does).
    "rrf",
    "reranker",
    "boost",
]


class FusionStrategy(StrEnum):
    """Strategy for combining BM25 and kNN results in hybrid search.

    Canonical values (Phase 5):
        RRF_ONLY        — fuse BM25+kNN via RRF (uniform weights)
        RRF_RERANKER    — RRF pool, then deterministic-score-sorted rerank
        WEIGHTED_RRF    — RRF with explicit per-list weights (rejects 1.0/1.0 default)
        TOP_RANK_BONUS  — RRF + bonus on the original-query list's top ranks
        POSITION_BLEND  — α-banded blend of RRF position + normalized rerank
        NONE            — no fusion math; concat ranked lists in input order, dedup

    Deprecated aliases (Phase 5 policy):
        RRF      — DeprecationWarning, maps to RRF_ONLY (lossless)
        BOOST    — raises ValueError (no direct equivalent — use WEIGHTED_RRF)
        RERANKER — raises ValueError (use RRF_RERANKER)
    """

    RRF_ONLY = "rrf_only"
    RRF_RERANKER = "rrf_reranker"
    WEIGHTED_RRF = "weighted_rrf"
    TOP_RANK_BONUS = "top_rank_bonus"
    POSITION_BLEND = "position_blend"
    NONE = "none"
    # Deprecated — handled by __post_init__ / resolve_fusion_operator
    RRF = "rrf"
    RERANKER = "reranker"
    BOOST = "boost"


# Deprecated aliases that warn-and-map (lossless). Hard-fail aliases live in
# _HARD_FAIL_FUSION_ALIASES below.
_DEPRECATED_FUSION_ALIASES: dict[FusionStrategy, FusionStrategy] = {
    FusionStrategy.RRF: FusionStrategy.RRF_ONLY,
}


# Aliases that raise ValueError at __post_init__ / resolve time. Maps to the
# migration message shown to the caller.
_HARD_FAIL_FUSION_ALIASES: dict[FusionStrategy, str] = {
    FusionStrategy.BOOST: (
        "FusionStrategy 'boost' is removed in Phase 5 — no direct equivalent. "
        "Use WEIGHTED_RRF with explicit bm25_rrf_weight/knn_rrf_weight, or "
        "stay on RRF_ONLY for hybrid defaults."
    ),
    FusionStrategy.RERANKER: (
        "FusionStrategy 'reranker' is removed in Phase 5 — use RRF_RERANKER instead "
        "(same fuse-then-rerank semantics with deterministic score-sorted output)."
    ),
}


RrfModeLiteral = Literal["auto", "native", "python"]
"""How to compute RRF.
    auto   — probe /_license once per retriever instance; native if licensed, else python (warn-once).
    native — force Elastic's native retriever.rrf (errors if unlicensed).
    python — always compute RRF in-process (skip license probe).
"""


# Valid expansion_hint strings — kept in sync with ``ExpansionStrategy`` below
# so the dataclass field can accept **both** the enum and a plain string
# without a ``# type: ignore``. ``ExpansionStrategy`` is already a ``StrEnum``
# (runtime-equivalent to its str value), this just teaches pyright the same
# thing. Mirrors the ``FusionStrategy | FusionStrategyLiteral`` pattern above.
ExpansionStrategyLiteral = Literal[
    "pass",
    "auto",
    "keyword",
    "paraphrase",
    "hyde",
    "multi",
]


def _assert_literal_matches_enum() -> None:
    """Ensure FusionStrategyLiteral and FusionStrategy stay in sync (run at import or in tests)."""
    literal_values = get_args(FusionStrategyLiteral)
    enum_values = [e.value for e in FusionStrategy]
    if set(literal_values) != set(enum_values):
        raise AssertionError(f"FusionStrategyLiteral and FusionStrategy out of sync: literal={literal_values!r} vs enum={enum_values!r}")


def _assert_expansion_literal_matches_enum() -> None:
    """Ensure ExpansionStrategyLiteral and ExpansionStrategy stay in sync."""
    literal_values = get_args(ExpansionStrategyLiteral)
    enum_values = [e.value for e in ExpansionStrategy]
    if set(literal_values) != set(enum_values):
        raise AssertionError(f"ExpansionStrategyLiteral and ExpansionStrategy out of sync: literal={literal_values!r} vs enum={enum_values!r}")


_assert_literal_matches_enum()
_assert_expansion_literal_matches_enum()


@dataclass
class ElasticFieldConfig:
    """Configuration for field name mapping in Elasticsearch indices.

    Core search fields map to the default ``metadata.*`` nested structure.
    Optional fields enable dual-content strategy (search vs display) and
    ``DocumentProvider`` capabilities (full-document reconstruction from chunks).

    Display content recovery (production pattern):
        Production stores a STRUCTURED BM25 blob at
        ``content_field`` (``metadata.content``) — summary + URL + app
        + title + body, ``\\n``-joined; see
        ``creative_phase_2026-05-15_es_mapping_alignment.md`` § 4. The
        retriever recovers the chunk body via
        ``elastic_retriever.extract_chunk_body()`` (rfind on
        ``\\ncontent: ``). ``display_content_field`` is therefore unset
        by default — leave it ``None`` for prod-aligned indices.

        Set ``display_content_field`` only for LEGACY indices that store
        the raw display text in a separate field (e.g. a pre-alignment
        index with ``content`` at root and the contextualised text at
        ``metadata.content``).
    """

    # --- Core search fields ---
    content_field: str = "metadata.content"
    content_body_anchor: str = "\n\nContent:"
    """Marker that terminates the contextual prefix inside ``content_field``.

    Production folds a contextual summary plus a variable set of metadata fields
    into ``metadata.content`` and closes that prefix with ``\\n\\nContent:``
    (capital ``C``, blank-line separated). ``parse_structured_content`` recovers
    the per-chunk body after this anchor and exposes the prefix as
    ``chunk.metadata["context_summary"]``. The legacy ``\\ncontent: `` anchor is
    always tried as a fallback, so overriding this only matters for indices with
    a third template convention."""
    contextualized_content_field: str | None = "metadata.contextualisedContent"
    """Native per-chunk contextual summary field. Production writes the chunk's
    contextual summary BOTH as the leading prefix of ``content_field`` AND
    verbatim into this dedicated field (``metadata.contextualisedContent`` —
    see ``infra/elasticsearch/ingestion``). When set and present on a hit, the
    result mapper surfaces it under the canonical
    ``chunk.metadata["contextualized_content"]`` key so the Knowledge Agent can
    render the per-chunk summary INSIDE each ``<chunk>`` (distinct from the
    page-shared ``context_summary`` prefix, which carries the Url/Application/
    apcode/title block once per page). Purely additive — absent field or unset
    config leaves chunk metadata unchanged. Set to ``None`` to skip extraction."""
    embedding_field: str = "embedding"
    title_field: str = "metadata.title"
    doc_field: str = "metadata.doc"
    doc_keyword_field: str = "metadata.doc.keyword"
    url_field: str = "metadata.pageUrl"
    title_boost_ratio: float = 0.5
    doc_boost_ratio: float = 0.3

    # --- Display content (legacy escape hatch) ---
    display_content_field: str | None = None
    """Legacy raw-display field for pre-alignment indices that stored display
    text separately from BM25 text. When set, the result mapper reads this
    field directly for ``chunk.content``; when ``None`` (prod-aligned default),
    the mapper recovers the body from ``content_field`` via
    ``extract_chunk_body()``. See the class docstring for the prod pattern."""

    # --- DocumentProvider support ---
    page_id_field: str | None = "metadata.pageId"
    """Document-level identifier.  Set to ``None`` to disable DocumentProvider."""

    chunk_index_field: str | None = "metadata.chunk_index"
    """Integer position of chunk within document.  Set to ``None`` to disable DocumentProvider."""

    # --- Extended metadata extraction ---
    app_name_field: str | None = "metadata.appName"
    """Application name field (e.g. ``'metadata.appName'``).  Extracted
    into chunk metadata when present."""

    last_doc_update_field: str | None = "metadata.lastDocUpdate"
    """Source-document last-update timestamp field. This is the PREFERRED
    freshness signal for downstream staleness rendering — it reflects when the
    content itself last changed. When present in ``_source``, the result mapper
    normalizes the value to an ISO 8601 string under the canonical
    ``lastDocUpdate`` metadata key. Accepts the production display format
    (``"May 12, 2025 @ 10:30:00"``), ISO 8601 strings, and epoch milliseconds;
    unparseable values pass through raw. Set to ``None`` to skip extraction."""

    last_doc_ingestion_field: str | None = "metadata.lastDocIngestion"
    """Index ingestion timestamp field — when the document was last (re)indexed,
    NOT when its content changed. Downstream staleness rendering only falls back
    to it when ``lastDocUpdate`` is absent, because a frequently re-ingested
    document looks deceptively fresh while its content may be years old.
    Same normalization/parsing behavior as ``last_doc_update_field``, canonical
    key ``lastDocIngestion``. Set to ``None`` to skip extraction (the generic
    ``metadata.*`` tail-merge still applies)."""

    entity_field: str | None = "metadata.entity.name"
    """Aggregation/filter leaf for entity NAME (e.g. ``'metadata.entity.name'``).
    Used by ``ElasticMetadataValueResolver``'s composite agg and by BM25 boost
    clauses (``MetadataScope.entity_boost`` → match on the entity name).
    Distinct from ``entity_object_field`` below which is the parent OBJECT
    path used for chunk-metadata extraction — keep them paired (the leaf is
    a child of the object)."""

    entity_object_field: str | None = "metadata.entity"
    """Object path returning the full entity dict ``{name, id, childs, is_opal}``
    for chunk-metadata extraction. Consumers read
    ``chunk.metadata["entity"]["id"]`` / ``["name"]`` etc. — see the
    ``metadata_scope_smoke`` probes. ``entity_field`` (the ``.name`` leaf) is
    for aggregations / BM25 boosts; this is for ``_source`` reads. Set to
    ``None`` to skip entity extraction in ``_default_result_mapper``."""

    # --- Metadata-scope filterable leaf fields (Phase 5) ---
    entity_id_field: str | None = "metadata.entity.id"
    """Leaf field used by ``MetadataScope.entity_filter`` / ``entity_boost``
    (e.g. ``'metadata.entity.id'``).  Distinct from ``entity_field`` (the
    object path used for metadata extraction)."""

    entity_childs_field: str | None = "metadata.entity.childs"
    """Descendant-ids array used by ``include_entity_childs=True`` expansion
    (e.g. ``'metadata.entity.childs'``)."""

    apcode_field: str | None = "metadata.auid"
    """APCODE keyword field used by ``MetadataScope.apcode_filter``/``boost``
    and by ``include_transversal`` widening. Default ``"metadata.auid"`` matches
    the audited production index (2026-05-15) — ``auid`` is the production field
    name for the apcode concept. Override to ``"metadata.apcode"`` for legacy
    indices that used the conceptual name as the field path."""

    scope_normalizers: Mapping[ScopeAxis, Normalizer] | None = None
    """Per-axis value-normalization policy for ``MetadataScope`` clause
    building. ``None`` (default) uses ``DEFAULT_AXIS_NORMALIZERS`` — ``app_name``
    lowercased, ``apcode`` uppercased — which matches the audited production
    index. Override only for an index with a different keyword-casing
    convention; the retriever forwards this to ``build_filter_clauses`` /
    ``build_boost_clauses``."""


@dataclass
class ElasticSearchConfig(BaseSearchConfig):
    """Elasticsearch-specific search configuration.

    fusion_strategy accepts FusionStrategy or a valid string literal
    ("rrf", "reranker", "boost", "none"); normalized to FusionStrategy in __post_init__.
    Invalid strings (e.g. "garbage") are rejected by the type checker.
    """

    fusion_strategy: FusionStrategy | FusionStrategyLiteral = FusionStrategy.RRF_RERANKER
    rank_window_size: int = 50
    rank_constant: int = 60
    knn_boost: float = 0.7
    bm25_boost: float = 0.3
    retrieval_size: int = 50
    enable_fuzzy: bool = False
    fuzzy_boost_ratio: float = 0.25
    es_rrf_mode: RrfModeLiteral = "auto"
    """DEPRECATED (Phase 5 Cycle E). Previously selected RRF execution mode
    (``auto`` / ``native`` / ``python``). Retired when the native ES
    ``retriever.rrf`` fast-path was dropped — fusion now always runs through
    the Python ``WeightedRRF`` pipeline because ``asyncio.gather`` on BM25+kNN
    sub-queries already gives wall-clock ≈ ``max(t_bm25, t_knn)``, so the
    single-RTT savings don't justify a separate code path. Field retained
    for construction-time API compat; functionally ignored. ``_can_use_native_rrf``
    survives as orphaned code reference (see Phase 5 log)."""
    rerank_top_n: int = 25
    """For RRF_RERANKER strategy: how many top RRF-ranked docs to send to the cross-encoder.

    Bumped 20 → 25 (2026-04-16) to give `PositionAwareBlend`'s min-max normalization a
    wider span across the reranked pool — tight clusters of rerank scores shrink to
    near-zero range otherwise, making the normed rerank term carry little signal."""

    max_concurrent_subqueries: int = 3
    """Per-call fan-out cap for ``search_many``: how many sub-queries may hit
    Elasticsearch in parallel within a single call. Does not cap across users —
    the ES client's own connection pool handles cross-user concurrency."""

    expansion_hint: ExpansionStrategy | ExpansionStrategyLiteral = ExpansionStrategy.PASS
    """Which expansion strategy the retriever should apply before fusion.

    Accepts either ``ExpansionStrategy`` enum members or a valid string
    literal (``"pass"`` / ``"auto"`` / ``"keyword"`` / …). ``__post_init__``
    normalizes strings to enum. Mirrors ``fusion_strategy``'s accept-both
    typing — ``ExpansionStrategy`` is a ``StrEnum``, runtime-equivalent to
    its value; the literal union lets pyright agree.

    ``PASS`` (default) keeps the legacy BM25+kNN single-query hybrid path.
    Any other value requires an ``expander`` wired at retriever construction
    time — the retriever raises at search time otherwise (factory-level
    validation is the louder first line of defense, per v3 §3.5)."""

    bm25_rrf_weight: float = 1.0
    """Per-list RRF weight for the BM25 sub-query. Uniform 1.0 == RRF_ONLY.

    Consumed by ``WEIGHTED_RRF`` via the retriever's SubQuery fan-out in
    Cycle E. ``resolve_fusion_operator(WEIGHTED_RRF, …)`` rejects the default
    uniform pair (1.0/1.0) to prevent silent degeneration to RRF_ONLY."""

    knn_rrf_weight: float = 1.0
    """Per-list RRF weight for the kNN sub-query. See ``bm25_rrf_weight``."""

    # --- Cycle F3 — AUTO BM25 probe thresholds ------------------------------
    # Defaults are ``+inf`` until F6 calibration. AUTO therefore never
    # declares a strong signal with defaults → always resolves to MULTI.
    # Callers that want aggressive AUTO behaviour set finite values via
    # ``ElasticSearchConfig(auto_probe_min_score=…, auto_probe_min_gap=…)``
    # or the matching ``retriever_auto_probe_*`` context overrides.
    auto_probe_min_score: float = float("inf")
    """AUTO BM25 probe — minimum top-hit score to declare strong signal.

    Deferred calibration: default is ``+inf`` so AUTO behaves identically
    to MULTI until F6 measures the real score distribution on this sparse
    low-BM25 corpus. Override per-instance once F6 lands a data-driven
    number. **Do NOT ship AUTO as PASS until calibrated.**"""

    auto_probe_min_gap: float = float("inf")
    """AUTO BM25 probe — minimum (top − second) gap to declare strong signal.

    Deferred calibration. See ``auto_probe_min_score`` for rationale."""

    intent: str | None = None
    """Runtime per-call intent surfaced via ``ElasticRetrieverContext.
    retriever_intent``. Threads into expansion prompts and the reranker
    query prepend."""

    def __post_init__(self) -> None:
        if isinstance(self.fusion_strategy, str):
            self.fusion_strategy = FusionStrategy(self.fusion_strategy)
        # Hard-fail removed aliases first — these have no auto-migration path.
        if self.fusion_strategy in _HARD_FAIL_FUSION_ALIASES:
            raise ValueError(_HARD_FAIL_FUSION_ALIASES[self.fusion_strategy])
        # Normalize lossless deprecated aliases → canonical, emit DeprecationWarning.
        if self.fusion_strategy in _DEPRECATED_FUSION_ALIASES:
            import warnings

            canonical = _DEPRECATED_FUSION_ALIASES[self.fusion_strategy]
            warnings.warn(
                f"FusionStrategy.{self.fusion_strategy.name} is deprecated; use {canonical.name} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            self.fusion_strategy = canonical
        # Mirror fusion_strategy's str → enum coercion for expansion_hint so
        # context dicts / JSON-shaped state can carry plain strings.
        if isinstance(self.expansion_hint, str):
            self.expansion_hint = ExpansionStrategy(self.expansion_hint)

    def resolve_params(
        self,
        *,
        size: int | None = None,
        fusion_strategy: str | FusionStrategy | None = None,
        rank_constant: int | None = None,
        retrieval_size: int | None = None,
        enable_fuzzy: bool | None = None,
        fuzzy_boost_ratio: float | None = None,
        rerank_top_n: int | None = None,
        expansion_hint: str | ExpansionStrategy | None = None,
        bm25_rrf_weight: float | None = None,
        knn_rrf_weight: float | None = None,
        intent: str | None = None,
        auto_probe_min_score: float | None = None,
        auto_probe_min_gap: float | None = None,
    ) -> dict[str, Any]:
        """Resolve search params: explicit overrides > instance config.

        Phase 5 Cycle E hotfix: dropped ``rank_window_size`` /
        ``knn_boost`` / ``bm25_boost`` from the override kwargs — all three
        were BOOST / native-RRF artifacts with no consumer in the uniform
        pipeline. ``rank_window_size`` callers should migrate to
        ``retrieval_size`` (they are synonyms now that native RRF is gone).

        Phase 5 Cycle F6c hotfix: ``auto_probe_min_score`` /
        ``auto_probe_min_gap`` threaded through so per-call context overrides
        (``retriever_auto_probe_*``) actually take effect. Previously the
        values were parsed into ``ElasticSearchConfig.from_context()`` but
        dropped at ``search(**cfg.to_search_kwargs())`` because the retriever's
        ``search()`` had no matching kwarg — they landed in ``**kwargs`` and
        were silently ignored.
        """
        resolved_fusion = fusion_strategy if fusion_strategy is not None else self.fusion_strategy
        if isinstance(resolved_fusion, FusionStrategy):
            resolved_fusion = resolved_fusion.value

        resolved_expansion = expansion_hint if expansion_hint is not None else self.expansion_hint
        if isinstance(resolved_expansion, ExpansionStrategy):
            resolved_expansion = resolved_expansion.value

        return {
            "size": size if size is not None else self.top_k,
            "fusion_strategy": resolved_fusion,
            "rank_constant": rank_constant if rank_constant is not None else self.rank_constant,
            "retrieval_size": retrieval_size if retrieval_size is not None else self.retrieval_size,
            "enable_fuzzy": enable_fuzzy if enable_fuzzy is not None else self.enable_fuzzy,
            "fuzzy_boost_ratio": fuzzy_boost_ratio if fuzzy_boost_ratio is not None else self.fuzzy_boost_ratio,
            "rerank_top_n": rerank_top_n if rerank_top_n is not None else self.rerank_top_n,
            "expansion_hint": resolved_expansion,
            "bm25_rrf_weight": bm25_rrf_weight if bm25_rrf_weight is not None else self.bm25_rrf_weight,
            "knn_rrf_weight": knn_rrf_weight if knn_rrf_weight is not None else self.knn_rrf_weight,
            "intent": intent if intent is not None else self.intent,
            "auto_probe_min_score": auto_probe_min_score if auto_probe_min_score is not None else self.auto_probe_min_score,
            "auto_probe_min_gap": auto_probe_min_gap if auto_probe_min_gap is not None else self.auto_probe_min_gap,
        }


class ElasticRetrieverContext(BaseRetrieverContext, total=False):
    """Elastic-specific runtime overrides in LangGraph state.

    Phase 5 Cycle E hotfix: dropped ``retriever_knn_boost`` /
    ``retriever_bm25_boost`` — no live consumer (BOOST strategy was
    hard-failed in Cycle A, pre-filter boost clauses have no equivalent
    in the uniform RRF pipeline). ``retriever_rank_window_size`` is kept
    but is now a synonym for ``retriever_retrieval_size`` (native ES RRF
    fast-path retired); callers should migrate to ``retriever_retrieval_size``.
    """

    retriever_fusion_strategy: str
    retriever_rank_window_size: int
    """DEPRECATED synonym for ``retriever_retrieval_size`` (see class docstring)."""
    retriever_rank_constant: int
    retriever_retrieval_size: int
    retriever_enable_fuzzy: bool
    retriever_fuzzy_boost_ratio: float
    retriever_expansion_hint: str
    """ExpansionStrategy value (``"pass"``/``"hyde"``/…). Coerced to enum via
    ``ElasticSearchConfig.__post_init__`` after ``from_context()``."""
    retriever_bm25_rrf_weight: float
    retriever_knn_rrf_weight: float
    retriever_intent: str
    """Runtime per-call intent surfaced into ``ElasticSearchConfig.intent``
    via ``BaseSearchConfig.from_context()`` introspection. Threads into
    expansion prompts + reranker query prepend. Build-time
    ``domain_intent`` lives on the retriever ctor, not here."""
    retriever_auto_probe_min_score: float
    """Per-call override for the AUTO BM25 probe's minimum top-hit score.
    Needs a matching ``retriever_auto_probe_min_gap`` to have an effect —
    both conditions must clear their thresholds for the probe to declare a
    strong signal. Defaults are ``+inf`` until Cycle G calibration, so this
    override is the only way to test AUTO end-to-end before then."""
    retriever_auto_probe_min_gap: float
    """Per-call override for the AUTO BM25 probe's minimum ``(top − second)``
    gap. See ``retriever_auto_probe_min_score`` for the paired-threshold
    note and default rationale."""

-------

packages/sta_agent_engine/src/sta_agent_engine/agents/knowledge_agent/compression/chunk_compressor.py
----
"""ChunkCompressor — compress text chunks into Findings via pluggable strategy.

Strategy is controlled by ``ChunkCompressionStrategy`` (LLM, PASSTHROUGH, DYNAMIC)
defined in ``knowledge_agent_config``.  Internal LLM-path grouping is controlled
by ``CompressionGroupingStrategy`` (BATCH, PER_PAGE_GROUP, PER_DOC_GROUP,
PER_DOC_TOKEN_GROUP).

Owns recompression: when the dynamic threshold is exceeded AND passthrough
findings exist, they are re-compressed via the LLM path internally.

Source-index validation uses ``ainvoke_with_output_validation`` to detect
and retry when LLMs produce concatenated indices (e.g. ``1356`` instead
of ``[1, 3, 5, 6]``).
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from typing import Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.state import RunnableConfig

from sta_agent_core.repositories import RetrievalChunk

from ...base.utils.output_validation import (
    ModelRetry,
    OutputValidationError,
    ainvoke_with_output_validation,
)
from ..knowledge_agent_config import (
    ChunkCompressionStrategy,
    CompressConfig,
    CompressionGroupingStrategy,
)
from ..knowledge_agent_prompts import COMPRESS_HUMAN_PROMPT, COMPRESS_SYSTEM_PROMPT
from ..knowledge_agent_types import (
    PASSTHROUGH_FALLBACK_MODE,
    Citation,
    CompressedFindings,
    Finding,
    GroundedFact,
)
from .helpers import (
    citation_from_chunk,
    content_hash,
    document_open_tag,
    filter_new_chunks,
    group_by_doc_group,
    group_by_page_group,
    group_document_chunk,
    group_page_shared_context,
    order_chunks_by_page,
    split_chunks_into_groups,
)
from .types import CompressResult, RetrieverEvidence


logger = logging.getLogger(__name__)


class _CompressValidationContext(TypedDict):
    """Typed context for source_indices validation."""

    num_chunks: int


class ChunkCompressor:
    """Compress text chunks into Findings using a pluggable strategy.

    Implements the Compressor protocol.

    Example:
        ```python
        compressor = ChunkCompressor(strategy=ChunkCompressionStrategy.DYNAMIC)
        result = await compressor.compress(query, evidence, model)
        # Dynamic: passthrough below threshold, LLM above.

        # After all compression, check recompression:
        replacement = await compressor.maybe_recompress(
            all_findings, global_char_count, query, model,
        )
        ```
    """

    def __init__(
        self,
        strategy: ChunkCompressionStrategy = ChunkCompressionStrategy.DYNAMIC,
        config: CompressConfig | None = None,
        dynamic_threshold: int = 400_000,
    ) -> None:
        self._strategy = strategy
        self._config = config or CompressConfig()
        self._dynamic_threshold = dynamic_threshold
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)

    @property
    def strategy(self) -> ChunkCompressionStrategy:
        return self._strategy

    @property
    def dynamic_threshold(self) -> int:
        return self._dynamic_threshold

    # ------------------------------------------------------------------
    # Compressor protocol: compress()
    # ------------------------------------------------------------------

    async def compress(
        self,
        query: str,
        evidence: RetrieverEvidence,
        model: BaseChatModel,
        config: RunnableConfig | None = None,
    ) -> CompressResult:
        """Compress chunks from evidence into Findings.

        Filters out already-compressed chunks via content hashing,
        then delegates to the resolved strategy (LLM or passthrough).

        Full-document expansion chunks bypass the hash filter (2b-D11):
        when ExpandNode fetches a full document, all chunks must reach
        compression for coherent cross-chunk reasoning.

        Args:
            query: The user's search query.
            evidence: Evidence bundle with chunks and hash sets.
            model: LLM for structured output (used only in LLM path).
            config: LangGraph runnable config.

        Returns:
            CompressResult with findings and updated compressed_hashes.
        """
        chunks = evidence.chunks
        existing_hashes = evidence.compressed_hashes

        new_chunks = filter_new_chunks(chunks, existing_hashes)
        skipped = len(chunks) - len(new_chunks)
        if skipped:
            logger.info("ChunkCompressor: skipping %d already-compressed chunks", skipped)

        if not new_chunks:
            return CompressResult(findings=[], compressed_hashes=existing_hashes)

        effective = self._resolve_strategy(evidence)
        retriever_name = evidence.retriever_name

        if effective == ChunkCompressionStrategy.LLM:
            logger.info(
                "ChunkCompressor: LLM path (%d chunks, global_char_count=%d)",
                len(new_chunks),
                evidence.global_char_count,
            )
            findings = await self._compress_llm(new_chunks, query, model, config, retriever_name)
        else:
            logger.info(
                "ChunkCompressor: passthrough path (%d chunks, global_char_count=%d)",
                len(new_chunks),
                evidence.global_char_count,
            )
            findings = self._compress_passthrough(new_chunks, retriever_name)

        new_hashes = {content_hash(c.content) for c in new_chunks}
        return CompressResult(
            findings=findings,
            compressed_hashes=existing_hashes | new_hashes,
        )

    # ------------------------------------------------------------------
    # Compressor protocol: maybe_recompress()
    # ------------------------------------------------------------------

    async def maybe_recompress(
        self,
        all_findings: list[Finding],
        global_char_count: int,
        query: str,
        model: BaseChatModel,
        config: RunnableConfig | None = None,
        current_pass_findings: list[Finding] | None = None,
    ) -> list[Finding] | None:
        """Re-compress passthrough findings if threshold exceeded.

        Only applies when strategy is DYNAMIC. Re-compresses two kinds of
        finding via the LLM path and returns the full replacement list
        (kept + recompressed). Re-compression runs one task per retriever; if a
        retriever's task fails outright, its original findings are preserved in
        the replacement list rather than dropped (the caller replaces the whole
        findings channel, so an omission is data loss):

        - ``"passthrough"`` — intentional deterministic findings (DYNAMIC below
          threshold), always eligible.
        - ``"passthrough_fallback"`` — evidence rescued after an LLM-compression
          failure (e.g. provider timeout). Eligible **only when carried over from
          a previous KA round**, never in the same pass that produced it — re-
          hitting a provider that failed moments ago is pointless. A later round
          gives it a fresh attempt; if that fails too it falls back again and is
          retried on the round after, bounded by ``max_iterations``.

        Args:
            all_findings: All findings (existing + new, all compressors).
            global_char_count: Total chars across all findings.
            query: The user's search query.
            model: LLM for structured output.
            config: LangGraph runnable config.
            current_pass_findings: Findings produced in the current compression
                pass (subset of ``all_findings``, matched by identity). Fallback
                findings in this set are skipped this round.

        Returns:
            Replacement findings list if recompression occurred, None otherwise.
        """
        if self._strategy != ChunkCompressionStrategy.DYNAMIC:
            return None
        if global_char_count <= self._dynamic_threshold:
            return None

        fresh_ids = {id(f) for f in (current_pass_findings or [])}

        def _is_recompressible(f: Finding) -> bool:
            if f.compression_mode == "passthrough":
                return True
            # Defer just-produced fallback evidence to a later round.
            return f.compression_mode == PASSTHROUGH_FALLBACK_MODE and id(f) not in fresh_ids

        recompress_ids = {id(f) for f in all_findings if _is_recompressible(f)}
        if not recompress_ids:
            return None

        kept = [f for f in all_findings if id(f) not in recompress_ids]
        targets = [f for f in all_findings if id(f) in recompress_ids]

        by_retriever: dict[str, list[Finding]] = defaultdict(list)
        for f in targets:
            retriever_name = f.retriever_sources[0] if f.retriever_sources else "unknown"
            by_retriever[retriever_name].append(f)

        tasks = []
        for retriever_name, findings in by_retriever.items():
            synthetic_chunks = []
            for f in findings:
                synthetic_chunks.extend(self._grounded_facts_to_chunks(f.key_facts, retriever_name))
            tasks.append(self._compress_llm(synthetic_chunks, query, model, config, retriever_name))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        recompressed: list[Finding] = []
        rescued_originals: list[Finding] = []
        for (retriever_name, original_findings), result in zip(by_retriever.items(), results, strict=True):
            if isinstance(result, BaseException):
                # The whole re-compression task for this retriever failed (an error
                # escaping the per-group rescue, e.g. raised while packing token
                # groups). Carry its ORIGINAL findings forward instead of dropping
                # them — the caller applies this list via FindingsUpdate(replace=True),
                # so anything omitted here vanishes from the answer. The originals
                # keep their existing mode tag, so a stale ``passthrough_fallback``
                # still surfaces ``compression_degraded`` downstream.
                logger.error(
                    "ChunkCompressor: re-compression failed for retriever %r (%s); preserving %d original finding(s)",
                    retriever_name,
                    type(result).__name__,
                    len(original_findings),
                )
                rescued_originals.extend(original_findings)
                continue
            recompressed.extend(result)

        if not recompressed:
            # Nothing recompressed successfully — keep the accumulated originals
            # untouched (reducer no-op). ``rescued_originals`` equals the full target
            # set here, so returning None loses no evidence.
            logger.warning(
                "ChunkCompressor: re-compression produced no findings from %d targets, keeping originals",
                len(targets),
            )
            return None

        # At least one retriever recompressed. Any failed retriever's evidence is
        # carried in ``rescued_originals`` so a partial failure never drops it.
        return kept + recompressed + rescued_originals

    # ------------------------------------------------------------------
    # Strategy resolution
    # ------------------------------------------------------------------

    def _resolve_strategy(self, evidence: RetrieverEvidence) -> ChunkCompressionStrategy:
        """Resolve effective strategy. Dynamic delegates based on threshold."""
        if self._strategy != ChunkCompressionStrategy.DYNAMIC:
            return self._strategy
        if evidence.global_char_count > self._dynamic_threshold:
            return ChunkCompressionStrategy.LLM
        return ChunkCompressionStrategy.PASSTHROUGH

    # ------------------------------------------------------------------
    # LLM compression path
    # ------------------------------------------------------------------

    async def _compress_llm(
        self,
        chunks: list[RetrievalChunk],
        query: str,
        model: BaseChatModel,
        config: RunnableConfig | None,
        retriever_name: str = "",
    ) -> list[Finding]:
        """Compress chunks via LLM using the configured grouping strategy."""
        if self._config.grouping_strategy == CompressionGroupingStrategy.PER_DOC_TOKEN_GROUP:
            return await self._compress_per_doc_token_group(chunks, query, model, config, retriever_name)
        if self._config.grouping_strategy == CompressionGroupingStrategy.PER_DOC_GROUP:
            return await self._compress_per_doc_group(chunks, query, model, config, retriever_name)
        if self._config.grouping_strategy == CompressionGroupingStrategy.PER_PAGE_GROUP:
            return await self._compress_per_page_group(chunks, query, model, config, retriever_name)
        return await self._compress_batch(chunks, query, model, config, retriever_name)

    async def _compress_per_doc_token_group(
        self,
        chunks: list[RetrievalChunk],
        query: str,
        model: BaseChatModel,
        config: RunnableConfig | None,
        retriever_name: str = "",
    ) -> list[Finding]:
        """Pack whole document groups into char-budgeted compression calls."""
        groups = self._pack_doc_token_groups(
            doc_groups=list(group_by_doc_group(chunks).values()),
            max_chars=self._config.max_chars_per_group,
        )
        if not groups:
            return []

        tasks = [self._compress_chunk_group(group, query, model, config, retriever_name) for group in groups]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        findings: list[Finding] = []
        for group, result in zip(groups, results, strict=True):
            if isinstance(result, BaseException):
                findings.extend(self._rescue_failed_group(group, result, retriever_name))
                continue
            findings.extend(result)

        return findings

    async def _compress_per_doc_group(
        self,
        chunks: list[RetrievalChunk],
        query: str,
        model: BaseChatModel,
        config: RunnableConfig | None,
        retriever_name: str = "",
    ) -> list[Finding]:
        """Group chunks by document, split large groups, compress in parallel."""
        groups = group_by_doc_group(chunks)
        return await self._compress_grouped_chunks(groups, query, model, config, retriever_name)

    async def _compress_per_page_group(
        self,
        chunks: list[RetrievalChunk],
        query: str,
        model: BaseChatModel,
        config: RunnableConfig | None,
        retriever_name: str = "",
    ) -> list[Finding]:
        """Group chunks by page, split large groups, compress in parallel."""
        groups = group_by_page_group(chunks)
        return await self._compress_grouped_chunks(groups, query, model, config, retriever_name)

    async def _compress_grouped_chunks(
        self,
        groups: dict[str, list[RetrievalChunk]],
        query: str,
        model: BaseChatModel,
        config: RunnableConfig | None,
        retriever_name: str = "",
    ) -> list[Finding]:
        """Split pre-grouped chunks and compress each sub-group in parallel."""
        max_count = self._config.max_chunks_per_group
        max_chars = self._config.max_chars_per_group

        tasks: list[Any] = []
        task_chunks: list[list[RetrievalChunk]] = []
        for group_chunks in groups.values():
            for sub_group in split_chunks_into_groups(group_chunks, max_count, max_chars):
                tasks.append(self._compress_chunk_group(sub_group, query, model, config, retriever_name))
                task_chunks.append(sub_group)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        findings: list[Finding] = []
        for sub_group, result in zip(task_chunks, results, strict=True):
            if isinstance(result, BaseException):
                findings.extend(self._rescue_failed_group(sub_group, result, retriever_name))
                continue
            findings.extend(result)

        return findings

    async def _compress_batch(
        self,
        chunks: list[RetrievalChunk],
        query: str,
        model: BaseChatModel,
        config: RunnableConfig | None,
        retriever_name: str = "",
    ) -> list[Finding]:
        """Compress chunks in batch mode, splitting if context budget is exceeded."""
        max_count = self._config.max_chunks_per_group
        max_chars = self._config.max_chars_per_group
        sub_groups = split_chunks_into_groups(chunks, max_count, max_chars)

        if not sub_groups:
            return []
        if len(sub_groups) == 1:
            return await self._compress_chunk_group(chunks, query, model, config, retriever_name)

        logger.info(
            "ChunkCompressor: batch mode split %d chunks into %d sub-groups (count=%d, chars=%d)",
            len(chunks),
            len(sub_groups),
            max_count,
            max_chars,
        )
        tasks = [self._compress_chunk_group(sg, query, model, config, retriever_name) for sg in sub_groups]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        findings: list[Finding] = []
        for sub_group, result in zip(sub_groups, results, strict=True):
            if isinstance(result, BaseException):
                findings.extend(self._rescue_failed_group(sub_group, result, retriever_name))
                continue
            findings.extend(result)

        return findings

    def _rescue_failed_group(
        self,
        chunks: list[RetrievalChunk],
        error: BaseException,
        retriever_name: str,
    ) -> list[Finding]:
        """Rescue a compression group whose task raised an unexpected error.

        ``_compress_chunk_group`` already funnels the *expected* failure
        (validation-retry exhaustion, provider read timeouts funnelled into
        ``OutputValidationError``) into a passthrough fallback. This handles
        anything that escapes that path: rather than silently dropping the
        group's evidence (the old ``logger.error`` + ``continue``), fall back to
        the deterministic passthrough path tagged ``PASSTHROUGH_FALLBACK_MODE`` so
        the degradation stays observable (``OutputNode`` →
        ``metadata["compression_degraded"]``). Cancellation / system-exit signals
        are not ``Exception``s and are re-raised untouched so task cancellation is
        never swallowed.
        """
        if not isinstance(error, Exception):
            raise error
        logger.warning(
            "ChunkCompressor: compression task raised %s; rescuing %d chunks via passthrough (evidence preserved)",
            type(error).__name__,
            len(chunks),
        )
        return self._compress_passthrough(chunks, retriever_name, mode=PASSTHROUGH_FALLBACK_MODE)

    @staticmethod
    def _pack_doc_token_groups(
        doc_groups: list[list[RetrievalChunk]],
        max_chars: int,
    ) -> list[list[RetrievalChunk]]:
        """Pack whole document groups into char-budgeted compression groups."""
        if max_chars < 1:
            raise ValueError(f"max_chars must be >= 1, got {max_chars}")

        groups: list[list[RetrievalChunk]] = []
        current: list[RetrievalChunk] = []
        current_chars = 0

        for doc_chunks in doc_groups:
            if not doc_chunks:
                continue

            doc_chars = sum(len(chunk.content) for chunk in doc_chunks)
            if doc_chars > max_chars:
                if current:
                    groups.append(current)
                    current = []
                    current_chars = 0
                groups.extend(ChunkCompressor._split_oversized_doc_group(doc_chunks, max_chars))
                continue

            if current and current_chars + doc_chars > max_chars:
                groups.append(current)
                current = []
                current_chars = 0

            current.extend(doc_chunks)
            current_chars += doc_chars

        if current:
            groups.append(current)

        return groups

    @staticmethod
    def _split_oversized_doc_group(
        chunks: list[RetrievalChunk],
        max_chars: int,
    ) -> list[list[RetrievalChunk]]:
        """Split one oversized document into contiguous, roughly even groups."""
        if max_chars < 1:
            raise ValueError(f"max_chars must be >= 1, got {max_chars}")
        if not chunks:
            return []

        total_chars = sum(len(chunk.content) for chunk in chunks)
        if total_chars <= max_chars:
            return [chunks]

        target_group_count = math.ceil(total_chars / max_chars)
        target_chars = math.ceil(total_chars / target_group_count)
        groups: list[list[RetrievalChunk]] = []
        current: list[RetrievalChunk] = []
        current_chars = 0

        for chunk in chunks:
            chunk_chars = len(chunk.content)
            if chunk_chars > max_chars:
                if current:
                    groups.append(current)
                    current = []
                    current_chars = 0
                groups.append([chunk])
                continue

            if current and len(groups) < target_group_count - 1 and current_chars + chunk_chars > target_chars:
                current_delta = abs(target_chars - current_chars)
                added_delta = abs(target_chars - (current_chars + chunk_chars))
                if current_chars >= target_chars or current_delta <= added_delta:
                    groups.append(current)
                    current = []
                    current_chars = 0

            current.append(chunk)
            current_chars += chunk_chars

        if current:
            groups.append(current)

        return groups

    async def _compress_chunk_group(
        self,
        chunks: list[RetrievalChunk],
        query: str,
        model: BaseChatModel,
        config: RunnableConfig | None,
        retriever_name: str = "",
    ) -> list[Finding]:
        """Compress a group of chunks into Findings via LLM structured output.

        Uses ``ainvoke_with_output_validation`` to validate source_indices.
        Falls back to empty findings if all retries are exhausted.
        """
        async with self._semaphore:
            # Page-order the group so the 1-based ids emitted by
            # _format_chunks_for_prompt line up with the chunk positions
            # _resolve_citations_map indexes for source_index → Citation. Both
            # walk this same ordering (order_chunks_by_page is idempotent).
            chunks = order_chunks_by_page(chunks)
            raw_chars = sum(len(c.content) for c in chunks)
            if raw_chars > self._config.max_chars_per_group:
                logger.warning(
                    "ChunkCompressor: group content (%d chars, ~%d tokens) exceeds "
                    "max_chars_per_group (%d) — likely a single oversized chunk; "
                    "proceeding and relying on API error handling",
                    raw_chars,
                    raw_chars // 4,
                    self._config.max_chars_per_group,
                )

            chunks_text = self._format_chunks_for_prompt(chunks)

            human_content = COMPRESS_HUMAN_PROMPT.format(
                query=query,
                chunks_text=chunks_text,
            )

            try:
                result = await ainvoke_with_output_validation(
                    model=model,
                    output_type=CompressedFindings,
                    messages=[
                        SystemMessage(content=COMPRESS_SYSTEM_PROMPT),
                        HumanMessage(content=human_content),
                    ],
                    output_validators=[self._validate_source_indices],
                    validation_context=_CompressValidationContext(num_chunks=len(chunks)),
                    max_retries=self._config.max_retries,
                    config=config,
                )
            except OutputValidationError:
                # Retries exhausted — either the model never produced a valid
                # CompressedFindings, or every attempt raised (e.g. provider read
                # timeouts, which output_validation funnels into this error). Don't
                # silently drop the group's evidence: rescue it via the
                # deterministic passthrough path, tagged so the degradation stays
                # observable (OutputNode → metadata["compression_degraded"]) and is
                # excluded from threshold re-compression.
                logger.warning(
                    "ChunkCompressor: validation-retry exhausted for group of %d chunks, falling back to passthrough (evidence preserved)",
                    len(chunks),
                )
                return self._compress_passthrough(
                    chunks,
                    retriever_name,
                    mode=PASSTHROUGH_FALLBACK_MODE,
                )

        compressed = cast(CompressedFindings, result)
        findings: list[Finding] = []
        for cf in compressed.findings:
            citation_map = self._resolve_citations_map(
                list(dict.fromkeys(entry.source_index for entry in cf.key_facts)),
                chunks,
                retriever_name,
            )
            grounded_facts = [
                GroundedFact(
                    fact=entry.fact,
                    citation=citation_map.get(entry.source_index),
                )
                for entry in cf.key_facts
            ]
            citations = list(citation_map.values())
            retriever_sources = list({c.retriever_name for c in citations if c.retriever_name})
            findings.append(
                Finding(
                    topic=cf.topic,
                    summary=cf.summary,
                    key_facts=grounded_facts,
                    confidence=cf.confidence,
                    citations=citations,
                    retriever_sources=retriever_sources,
                    needs_expansion=cf.needs_expansion,
                    compression_mode="llm",
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Passthrough compression path
    # ------------------------------------------------------------------

    def _compress_passthrough(
        self,
        chunks: list[RetrievalChunk],
        retriever_name: str,
        mode: str = "passthrough",
    ) -> list[Finding]:
        """Convert chunks to Findings without LLM calls.

        Groups chunks by source document. Each doc group becomes one Finding
        with chunk contents as GroundedFact key_facts.

        Args:
            chunks: Chunks to convert.
            retriever_name: Source retriever name for citations.
            mode: ``compression_mode`` to stamp on each Finding. Defaults to
                ``"passthrough"`` (chosen on purpose); the LLM-failure fallback
                passes ``PASSTHROUGH_FALLBACK_MODE`` so a degraded group is
                distinguishable downstream.
        """
        if self._config.grouping_strategy == CompressionGroupingStrategy.PER_PAGE_GROUP:
            doc_groups = group_by_page_group(chunks)
        else:
            doc_groups = group_by_doc_group(chunks)

        findings: list[Finding] = []
        for doc_key, group_chunks in doc_groups.items():
            finding = self._build_passthrough_finding(doc_key, group_chunks, retriever_name, mode=mode)
            findings.append(finding)

        return findings

    def _build_passthrough_finding(
        self,
        doc_key: str,
        chunks: list[RetrievalChunk],
        retriever_name: str,
        mode: str = "passthrough",
    ) -> Finding:
        """Build a Finding from a doc group without LLM synthesis."""
        topic = self._derive_topic(chunks, doc_key)
        summary = self._derive_summary(chunks)

        citations = [citation_from_chunk(c, retriever_name) for c in chunks]
        grounded_facts = [GroundedFact(fact=self._format_chunk_fact(c, retriever_name), citation=cit) for c, cit in zip(chunks, citations)]

        return Finding(
            topic=topic,
            summary=summary,
            key_facts=grounded_facts,
            confidence="medium",
            citations=citations,
            retriever_sources=[retriever_name],
            needs_expansion=False,
            compression_mode=mode,
        )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_source_indices(
        output: CompressedFindings,
        context: _CompressValidationContext,
    ) -> CompressedFindings:
        """Validate per-fact source_index against the actual number of chunks.

        Each ``KeyFactEntry.source_index`` must be within [1, num_chunks].

        Raises:
            ModelRetry: If any source index is invalid.
        """
        num_chunks = context["num_chunks"]
        errors: list[str] = []

        for i, finding in enumerate(output.findings):
            for entry in finding.key_facts:
                idx = entry.source_index
                if idx < 1 or idx > num_chunks:
                    errors.append(
                        f"Finding {i + 1} ('{finding.topic}'): fact '{entry.fact[:50]}...' has source_index {idx}, valid range is 1-{num_chunks}."
                    )

        if errors:
            raise ModelRetry("\n".join(errors))

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_chunks_for_prompt(chunks: list[RetrievalChunk]) -> str:
        """Format chunks grouped by page as XML for the LLM prompt.

        Same-page chunks share ONE ``<document>`` header carrying the page
        metadata (pageId/title/doc/source) and the page-shared
        ``<context_summary>`` (Url / Application / apcode / title block) — both
        rendered once instead of repeated inside every chunk body. Each chunk's
        OWN contextual summary travels INSIDE its ``<chunk>`` as a leading
        ``<chunk_context>`` line (from ``contextualized_content``), so the LLM
        sees per-chunk relevance signal without the page prefix drowning it.
        Chunk bodies nest as ``<chunk id="N">`` children with globally
        sequential 1-based ids over the page-ordered list, so the LLM's
        ``source_index`` maps directly to a chunk (see ``order_chunks_by_page``
        / ``_resolve_citations_map``).

        Note: ``_compress_chunk_group`` already page-orders the group before
        calling this. Re-grouping here via ``group_by_page_group`` is not
        redundant — it is idempotent and keeps the formatter a *pure function of
        its input*, so the ids it emits are always page-ordered regardless of how
        a caller hands the chunks in. That independence is the guardrail against
        a future caller introducing a silent ``source_index`` misalignment.

        The page-shared ``<context_summary>`` and the ``<document>`` metadata are
        recovered from the whole group (``group_page_shared_context`` /
        ``group_document_chunk``), not just its first chunk, so a page whose
        lowest-``chunk_index`` chunk lacks the prefix still surfaces it once.
        """
        parts: list[str] = []
        chunk_id = 0
        for group in group_by_page_group(chunks).values():
            parts.append(document_open_tag(group_document_chunk(group)))
            shared = group_page_shared_context(group)
            if shared:
                parts.append(f"<context_summary>\n{shared}\n</context_summary>")
            for chunk in group:
                chunk_id += 1
                contextual = chunk.metadata.get("contextualized_content")
                if contextual:
                    parts.append(f'<chunk id="{chunk_id}">\n<chunk_context>{contextual}</chunk_context>\n{chunk.content}\n</chunk>')
                else:
                    parts.append(f'<chunk id="{chunk_id}">\n{chunk.content}\n</chunk>')
            parts.append("</document>")
        return "\n".join(parts)

    @staticmethod
    def _resolve_citations_map(
        source_indices: list[int],
        chunks: list[RetrievalChunk],
        retriever_name: str = "",
    ) -> dict[int, Citation]:
        """Map 1-based source indices to Citation objects via ``citation_from_chunk``.

        Delegates to the shared helper so title/URL fallback chains are
        consistent across LLM and passthrough paths.  Falls back to
        ``chunk.retriever_type`` when ``retriever_name`` is empty
        (e.g. during re-compression with synthetic chunks).
        """
        citation_map: dict[int, Citation] = {}
        for idx in source_indices:
            zero_idx = idx - 1
            if zero_idx < 0 or zero_idx >= len(chunks):
                logger.warning(
                    "ChunkCompressor: source_index %d out of range (1-%d), skipping",
                    idx,
                    len(chunks),
                )
                continue
            chunk = chunks[zero_idx]
            effective_name = retriever_name or chunk.retriever_type
            citation_map[idx] = citation_from_chunk(chunk, effective_name)
        return citation_map

    @staticmethod
    def _format_chunk_fact(chunk: RetrievalChunk, retriever_name: str) -> str:
        """Format a chunk as an XML-tagged key_fact preserving provenance.

        On the passthrough path this fact text IS what the synthesizer reads
        (one GroundedFact per chunk), so the chunk's own contextual summary is
        prepended as a ``<chunk_context>`` line when present — that per-chunk
        relevance signal would otherwise be lost (the page-level
        ``context_summary`` carries page identity, not the chunk's specifics).
        """
        attrs: dict[str, str] = {"source": retriever_name}
        idx = chunk.metadata.get("chunk_index")
        if idx is not None:
            attrs["index"] = str(idx)
        page_id = chunk.metadata.get("pageId") or chunk.metadata.get("page_id")
        if page_id:
            attrs["page_id"] = str(page_id)
        attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        contextual = chunk.metadata.get("contextualized_content")
        body = f"<chunk_context>{contextual}</chunk_context>\n{chunk.content}" if contextual else chunk.content
        return f"<chunk {attr_str}>\n{body}\n</chunk>"

    @staticmethod
    def _derive_topic(chunks: list[RetrievalChunk], doc_key: str) -> str:
        """Extract topic from chunk metadata or fall back to doc key."""
        for c in chunks:
            title = c.metadata.get("title") or c.metadata.get("doc_title")
            if title:
                return str(title)
        return doc_key

    @staticmethod
    def _derive_summary(chunks: list[RetrievalChunk]) -> str:
        """Build summary from chunk summaries if available."""
        summaries = [c.metadata.get("summary", "") for c in chunks if c.metadata.get("summary")]
        if summaries:
            return " ".join(summaries)
        return f"Raw evidence from {len(chunks)} retrieved chunk(s)"

    @staticmethod
    def _grounded_facts_to_chunks(
        grounded_facts: list[GroundedFact],
        retriever_name: str,
    ) -> list[RetrievalChunk]:
        """Reconstruct synthetic RetrievalChunks from GroundedFacts for re-compression.

        Encapsulated here (2e-D5) — no external caller needs this.
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
from ..utils.findings_format import page_shared_summary


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
        "contextualized_content",
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


def page_shared_context(chunk: RetrievalChunk) -> str | None:
    """Page-shared remainder of a chunk's ``context_summary``.

    The retriever's ``context_summary`` is the full structured prefix: the
    per-chunk contextual summary FOLLOWED BY the page-shared metadata block
    (Url / Application / apcode / title — identical across a page's chunks).
    The per-chunk summary is surfaced separately INSIDE each ``<chunk>`` via
    ``contextualized_content``, so the page-level ``<context_summary>`` renders
    only this shared remainder to avoid repeating one chunk's summary at the
    page level. When no ``contextualized_content`` is present (legacy / un-
    templated indices), the full prefix is returned unchanged — exact legacy
    behavior. Returns ``None`` when nothing remains.
    """
    remainder = page_shared_summary(chunk.metadata.get("context_summary"), chunk.metadata.get("contextualized_content"))
    return remainder or None


def group_page_shared_context(group: list[RetrievalChunk]) -> str | None:
    """First non-empty page-shared context remainder across a page group.

    Mirrors ``group_context_summary``'s whole-group scan so a page whose
    lowest-``chunk_index`` chunk lacks the prefix still surfaces the shared
    block once. Each chunk of a page yields the same remainder (only the
    leading per-chunk summary differs), so the first non-empty wins.
    """
    for chunk in group:
        shared = page_shared_context(chunk)
        if shared:
            return shared
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
- Completeness of QUERY-RELEVANT detail OUTRANKS brevity. There is no fixed
  length or paragraph cap — be as long as the evidence requires to carry every
  specific the orchestrator needs to act on. "Concise" here means cutting filler
  and connective prose, NEVER dropping substance. A summary that lost the values
  is a failure, not a concise answer.
- Preserve every concrete specific verbatim: exact values, identifiers, version
  numbers, config keys, parameter names, commands, code snippets, error strings,
  thresholds, dates, and domain keywords.
- Reproduce enumerations IN FULL — when the findings list steps, options,
  parameters, error codes, entities, or conditions, include EVERY item. Do NOT
  abridge to "examples include …", "such as …", or "etc."; the items you drop
  are exactly what the orchestrator may need. Preserve edge cases, qualifiers,
  and conditions ("only if …", "except when …") — they change what the answer means.
- No section headers, no bold/italic styling, no closing summary, no references
  section. A tight bullet list is encouraged for enumerations; inline code and
  code snippets ARE allowed and preferred when they carry the actual information.
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
from ..utils.findings_format import _REVIEW_SOURCE_CONTEXT_CHARS, cited_documents_line, finding_source_context_line, format_finding_block


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
                # entity the page's context shows it does not belong to.
                source_context_line=finding_source_context_line(finding.citations, max_summary_chars=_REVIEW_SOURCE_CONTEXT_CHARS)
                if finding.citations
                else None,
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
from ..utils.findings_format import _REVIEW_SOURCE_CONTEXT_CHARS, cited_documents_line, finding_source_context_line, format_finding_block


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
                source_context_line=finding_source_context_line(f.citations, max_summary_chars=_REVIEW_SOURCE_CONTEXT_CHARS) if f.citations else None,
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

# Per-page ``context_summary`` cap at SYNTHESIS — deliberately far larger than
# the review prompts' 320 cap. The synthesizer writes the user-facing answer, so
# clipping the page's contextual prefix mid-sentence (the old 320 cap collapsed
# newlines then truncated, dropping the page-path / breadcrumb tail) cost it the
# very identity signal it needs. The whole findings block is still bounded by
# ``SynthesisConfig.max_synthesis_input_tokens``, so a generous per-page cap is
# safe. Set high enough to keep a full contextual prefix; still guards against a
# pathological multi-KB blob.
_SYNTHESIS_SOURCE_CONTEXT_CHARS = 1200


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
            source_context = finding_source_context_line(finding.citations, max_summary_chars=_SYNTHESIS_SOURCE_CONTEXT_CHARS)
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

# Per-page ``context_summary`` cap for the REVIEW + cost-estimate surfaces.
_REVIEW_SOURCE_CONTEXT_CHARS = 320


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


def page_shared_summary(context_summary: Any, contextualized_content: Any) -> str:
    """Strip the leading per-chunk contextual summary from the full structured
    ``context_summary``, leaving the page-shared metadata block (Url / Application
    / apcode / title — identical across a page's chunks).
    """
    summary = "" if context_summary is None else str(context_summary)
    if not summary:
        return ""
    ctx = "" if contextualized_content in (None, "") else str(contextualized_content)
    if ctx and summary.startswith(ctx):
        return summary[len(ctx) :].lstrip("\n").lstrip()
    return summary


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
        # Render only the page-shared remainder
        shared_summary = page_shared_summary(meta.get("context_summary"), meta.get("contextualized_content"))
        summary_text = _flatten_summary(shared_summary, max_summary_chars) if shared_summary else ""
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
            # Must mirror what the review nodes render (same _REVIEW_SOURCE_CONTEXT_CHARS
            # cap) so this estimate stays aligned with the review token budget AND
            # CompressNode's recompression math. NOT the synthesis cap (1200) — review
            # deliberately stays tighter.
            source_context_line=finding_source_context_line(f.citations, max_summary_chars=_REVIEW_SOURCE_CONTEXT_CHARS) if f.citations else None,
        )
        total += len(block)
    return total

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
  agent's answer covers the question, return it and, only when deeper digging
  might genuinely help, offer it as a follow-up rather than spending more
  delegations now.

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

