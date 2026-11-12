"""Single-list retrieval queries: vector search and keyword search, with optional filters."""

from dataclasses import dataclass
from uuid import UUID

import numpy as np
import psycopg

from talk_to_documents.config import settings


@dataclass(frozen=True)
class SearchHit:
    chunk_id: UUID
    document_id: UUID
    title: str
    page: int | None
    heading_path: str | None
    content: str
    score: float  # vector: cosine similarity (higher is better); keyword: ts_rank_cd


_FILTERS = """
  (%(source_type)s::text IS NULL OR d.source_type = %(source_type)s::text)
  AND (%(tags)s::text[] IS NULL OR d.tags && %(tags)s::text[])
"""


def vector_search(
    conn: psycopg.Connection,
    query_vector: np.ndarray,
    k: int = 5,
    *,
    source_type: str | None = None,
    tags: list[str] | None = None,
) -> list[SearchHit]:
    """Nearest chunks by cosine distance. `<=>` is cosine distance (smaller is closer).

    Note: with an HNSW index and a selective filter, fewer than `k` rows can come back,
    because the filter is applied after the index scan.
    """
    rows = conn.execute(
        f"""
        SELECT c.id, c.document_id, d.title, c.page, c.heading_path, c.content,
               1 - (c.embedding <=> %(qvec)s) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {_FILTERS}
        ORDER BY c.embedding <=> %(qvec)s
        LIMIT %(k)s
        """,
        {'qvec': query_vector, 'k': k, 'source_type': source_type, 'tags': tags},
    ).fetchall()
    return [SearchHit(*row) for row in rows]


def keyword_search(
    conn: psycopg.Connection,
    query: str,
    k: int = 5,
    fts_config: str | None = None,
    *,
    source_type: str | None = None,
    tags: list[str] | None = None,
) -> list[SearchHit]:
    """Full-text search using the same config the documents were indexed with.

    websearch_to_tsquery combines the words with AND: every word must occur in the chunk.
    """
    config = fts_config or settings.fts_config
    rows = conn.execute(
        f"""
        SELECT c.id, c.document_id, d.title, c.page, c.heading_path, c.content,
               ts_rank_cd(c.tsv, q) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id,
             websearch_to_tsquery(%(cfg)s::regconfig, %(q)s) AS q
        WHERE d.fts_config = %(cfg)s::text AND c.tsv @@ q AND {_FILTERS}
        ORDER BY score DESC
        LIMIT %(k)s
        """,
        {
            'cfg': config,
            'q': query,
            'k': k,
            'source_type': source_type,
            'tags': tags,
        },
    ).fetchall()
    return [SearchHit(*row) for row in rows]
