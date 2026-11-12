"""Hybrid search: vector + keyword lists fused with RRF, optionally reranked."""

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol

import numpy as np
import psycopg

from talk_to_documents.config import Settings
from talk_to_documents.config import settings as default_settings
from talk_to_documents.retrieval.fusion import FusedHit, rrf_fuse
from talk_to_documents.retrieval.rerank import TextScorer, apply_rerank
from talk_to_documents.retrieval.search import keyword_search, vector_search

Mode = Literal['keyword', 'vector', 'hybrid']
MODES = ('keyword', 'vector', 'hybrid')


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass
class SearchTimings:
    embed_ms: float = 0.0
    vector_ms: float = 0.0
    keyword_ms: float = 0.0
    fuse_ms: float = 0.0
    rerank_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.vector_ms + self.keyword_ms + self.fuse_ms + self.rerank_ms


@dataclass
class SearchResult:
    hits: list[FusedHit]
    timings: SearchTimings


def _ms(start: float) -> float:
    return (perf_counter() - start) * 1000


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    *,
    embedder: QueryEmbedder | None = None,
    mode: Mode = 'hybrid',
    top_k: int = 5,
    candidates: int | None = None,
    rerank: bool = False,
    reranker: TextScorer | None = None,
    source_type: str | None = None,
    tags: list[str] | None = None,
    settings: Settings = default_settings,
) -> SearchResult:
    """Search in one of three modes, optionally rerank, and report time spent per stage.

    keyword / vector: a single list. hybrid: both lists fused with RRF.
    With rerank=True the first `rerank_top_n` hits of the mode's ordering are rescored by the
    cross-encoder and the best `top_k` are returned; otherwise the first `top_k` are returned.
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
    candidates = candidates or settings.candidates
    timings = SearchTimings()
    rankings: dict[str, list] = {}

    if mode in ('vector', 'hybrid'):
        if embedder is None:
            from talk_to_documents.retrieval.embed import get_embedder

            embedder = get_embedder()
        started = perf_counter()
        query_vector = embedder.embed_query(query)
        timings.embed_ms = _ms(started)

        started = perf_counter()
        rankings['vector'] = vector_search(
            conn, query_vector, candidates, source_type=source_type, tags=tags
        )
        timings.vector_ms = _ms(started)

    if mode in ('keyword', 'hybrid'):
        started = perf_counter()
        rankings['keyword'] = keyword_search(
            conn, query, candidates, settings.fts_config, source_type=source_type, tags=tags
        )
        timings.keyword_ms = _ms(started)

    started = perf_counter()
    if len(rankings) == 1:
        ((name, hits),) = rankings.items()
        fused = [FusedHit(hit, hit.score, {name: rank}) for rank, hit in enumerate(hits, 1)]
    else:
        fused = rrf_fuse(rankings, k=settings.rrf_k)
    timings.fuse_ms = _ms(started)

    if rerank and fused:
        if reranker is None:
            from talk_to_documents.retrieval.rerank import get_reranker

            reranker = get_reranker()
        started = perf_counter()
        final = apply_rerank(query, fused, reranker, settings.rerank_top_n, top_k)
        timings.rerank_ms = _ms(started)
    else:
        final = fused[:top_k]

    return SearchResult(final, timings)
