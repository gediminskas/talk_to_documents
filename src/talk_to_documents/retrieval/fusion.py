"""Reciprocal Rank Fusion: combine several ranked lists using ranks only.

Cosine similarities (e5 compresses them into a narrow high band) and ts_rank_cd values live
on completely different scales, so raw scores are never mixed. Only ranks are.
"""

from dataclasses import dataclass, field
from uuid import UUID

from talk_to_documents.retrieval.search import SearchHit


@dataclass(frozen=True)
class FusedHit:
    hit: SearchHit
    score: float  # RRF score (or the list's own score in a single-list mode)
    ranks: dict[str, int] = field(default_factory=dict)  # e.g. {'vector': 3, 'keyword': 1}
    rerank_score: float | None = None


def rrf_fuse(
    rankings: dict[str, list[SearchHit]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[FusedHit]:
    """score(chunk) = sum over the lists containing it of weight / (k + rank), rank from 1.

    Deduplicates by chunk id (keeping the hit from the list where it ranks best). Output is
    sorted by score descending; ties break by best rank, then chunk id, so it is deterministic.
    """
    if k <= 0:
        raise ValueError('k must be positive')
    weights = weights or {}

    scores: dict[UUID, float] = {}
    ranks: dict[UUID, dict[str, int]] = {}
    best: dict[UUID, tuple[int, SearchHit]] = {}

    for name, hits in rankings.items():
        weight = weights.get(name, 1.0)
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk_id
            if name in ranks.get(chunk_id, {}):
                continue  # duplicate inside one list: the first occurrence counts
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            ranks.setdefault(chunk_id, {})[name] = rank
            if chunk_id not in best or rank < best[chunk_id][0]:
                best[chunk_id] = (rank, hit)

    fused = [FusedHit(best[cid][1], scores[cid], ranks[cid]) for cid in scores]
    fused.sort(key=lambda f: (-round(f.score, 12), min(f.ranks.values()), str(f.hit.chunk_id)))
    return fused
