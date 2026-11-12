"""hybrid_search without a database: the two single-list searches are replaced by stubs."""

from uuid import UUID

import numpy as np
import pytest

from talk_to_documents.config import Settings
from talk_to_documents.retrieval import hybrid
from talk_to_documents.retrieval.search import SearchHit


def hit(name: str) -> SearchHit:
    return SearchHit(UUID(int=ord(name)), UUID(int=1), name, None, None, f'text {name}', 0.5)


@pytest.fixture
def stubs(monkeypatch):
    calls = {'vector': [], 'keyword': []}

    def vector_stub(conn, vector, k, **kwargs):
        calls['vector'].append((k, kwargs))
        return [hit('A'), hit('B'), hit('C')]

    def keyword_stub(conn, query, k, fts_config, **kwargs):
        calls['keyword'].append((k, fts_config, kwargs))
        return [hit('B'), hit('D')]

    monkeypatch.setattr(hybrid, 'vector_search', vector_stub)
    monkeypatch.setattr(hybrid, 'keyword_search', keyword_stub)
    return calls


class StubEmbedder:
    def embed_query(self, text):
        return np.zeros(4, dtype=np.float32)


SETTINGS = Settings(_env_file=None, candidates=7, rerank_top_n=3, fts_config='simple')


def run(**kwargs):
    return hybrid.hybrid_search(
        None, 'q words', embedder=StubEmbedder(), settings=SETTINGS, **kwargs
    )


def titles(result):
    return [f.hit.title for f in result.hits]


def test_hybrid_fuses_both_lists(stubs):
    result = run(mode='hybrid', top_k=10)
    assert titles(result) == ['B', 'A', 'D', 'C']
    assert stubs['vector'][0][0] == 7 and stubs['keyword'][0][0] == 7  # candidates passed on
    assert stubs['keyword'][0][1] == 'simple'


def test_single_list_modes(stubs):
    assert titles(run(mode='vector')) == ['A', 'B', 'C']
    assert stubs['keyword'] == []
    stubs['vector'].clear()
    result = run(mode='keyword')
    assert titles(result) == ['B', 'D'] and stubs['vector'] == []
    assert result.hits[0].ranks == {'keyword': 1}


def test_top_k_and_candidates_override(stubs):
    assert len(run(mode='hybrid', top_k=2).hits) == 2
    run(mode='hybrid', candidates=3)
    assert stubs['vector'][-1][0] == 3


def test_filters_are_passed_through(stubs):
    run(mode='hybrid', source_type='note', tags=['x'])
    assert stubs['vector'][0][1] == {'source_type': 'note', 'tags': ['x']}
    assert stubs['keyword'][0][2] == {'source_type': 'note', 'tags': ['x']}


def test_rerank_limits_to_top_n_and_reorders(stubs, fake_reranker):
    result = run(mode='hybrid', top_k=2, rerank=True, reranker=fake_reranker)
    assert len(fake_reranker.calls[0][1]) == 3  # rerank_top_n = 3
    assert len(result.hits) == 2
    assert all(f.rerank_score is not None for f in result.hits)
    assert result.timings.rerank_ms >= 0


def test_timings_only_for_stages_that_ran(stubs):
    result = run(mode='keyword')
    assert result.timings.embed_ms == 0.0 and result.timings.vector_ms == 0.0
    assert result.timings.total_ms >= 0


def test_invalid_mode(stubs):
    with pytest.raises(ValueError):
        run(mode='nonsense')
