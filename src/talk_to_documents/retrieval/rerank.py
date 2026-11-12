"""Cross-encoder reranking of fused search hits."""

from dataclasses import replace
from functools import lru_cache
from typing import Protocol

from talk_to_documents.config import settings
from talk_to_documents.retrieval.fusion import FusedHit
from talk_to_documents.retrieval.search import SearchHit


class TextScorer(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class Reranker:
    """Wraps a fastembed cross-encoder. Scores are raw model outputs: they are not
    probabilities and can be negative, so compare them only with each other."""

    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 16) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy: keeps --help fast

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One score per text, in the same order as `texts`."""
        if not texts:
            return []
        return [float(s) for s in self._model.rerank(query, texts, batch_size=self.batch_size)]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(settings.rerank_model, cache_dir=settings.fastembed_cache_path)


def hit_text(hit: SearchHit) -> str:
    """The text a reranker sees: heading path (if any) plus content."""
    if hit.heading_path:
        return f'{hit.heading_path}\n{hit.content}'
    return hit.content


def apply_rerank(
    query: str,
    fused: list[FusedHit],
    reranker: TextScorer,
    top_n: int,
    top_k: int,
) -> list[FusedHit]:
    """Score the first `top_n` hits, sort by reranker score (stable), return the best `top_k`."""
    head = fused[:top_n]
    if not head:
        return []
    scores = reranker.score(query, [hit_text(f.hit) for f in head])
    if len(scores) != len(head):
        raise ValueError(f'reranker returned {len(scores)} scores for {len(head)} texts')
    rescored = [replace(f, rerank_score=s) for f, s in zip(head, scores, strict=True)]
    rescored.sort(key=lambda f: f.rerank_score, reverse=True)
    return rescored[:top_k]
