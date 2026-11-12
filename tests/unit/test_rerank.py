from uuid import UUID

import pytest

from talk_to_documents.retrieval.fusion import FusedHit
from talk_to_documents.retrieval.rerank import apply_rerank, hit_text
from talk_to_documents.retrieval.search import SearchHit


def fused(i: int, content: str, heading: str | None = None) -> FusedHit:
    hit = SearchHit(UUID(int=i), UUID(int=1), f'doc{i}', None, heading, content, 0.0)
    return FusedHit(hit, score=1 / (60 + i), ranks={'vector': i})


def test_hit_text_includes_heading_path_when_present():
    assert hit_text(fused(1, 'body', 'A > B').hit) == 'A > B\nbody'
    assert hit_text(fused(1, 'body').hit) == 'body'


def test_rerank_reorders_and_sets_scores(fake_reranker):
    hits = [fused(1, 'nothing relevant'), fused(2, 'apple pie recipe'), fused(3, 'apple')]
    result = apply_rerank('apple pie recipe', hits, fake_reranker, top_n=3, top_k=3)
    assert [f.hit.title for f in result] == ['doc2', 'doc3', 'doc1']
    assert [f.rerank_score for f in result] == [3.0, 1.0, 0.0]
    assert result[0].ranks == {'vector': 2}  # original ranks are kept


def test_top_n_limits_what_is_scored_and_top_k_what_is_returned(fake_reranker):
    hits = [fused(i, f'word{i}') for i in range(1, 8)]
    result = apply_rerank('word6', hits, fake_reranker, top_n=3, top_k=2)
    assert len(fake_reranker.calls[0][1]) == 3  # only the first 3 hits were scored
    assert len(result) == 2
    assert 'doc6' not in [f.hit.title for f in result]  # rank 6 was beyond top_n


def test_empty_input(fake_reranker):
    assert apply_rerank('q', [], fake_reranker, top_n=5, top_k=5) == []
    assert fake_reranker.calls == []


def test_wrong_number_of_scores_is_an_error():
    class Broken:
        def score(self, query, texts):
            return [1.0]

    with pytest.raises(ValueError):
        apply_rerank('q', [fused(1, 'a'), fused(2, 'b')], Broken(), top_n=2, top_k=2)
