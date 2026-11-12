from uuid import UUID

import pytest

from talk_to_documents.retrieval.fusion import rrf_fuse
from talk_to_documents.retrieval.search import SearchHit


def hit(name: str, score: float = 0.0) -> SearchHit:
    chunk_id = UUID(int=ord(name))
    return SearchHit(chunk_id, UUID(int=1), name, None, None, f'text {name}', score)


def order(fused):
    return [f.hit.title for f in fused]


def test_hand_checked_example():
    # vector [A, B, C], keyword [B, D, A], k = 60
    fused = rrf_fuse(
        {'vector': [hit('A'), hit('B'), hit('C')], 'keyword': [hit('B'), hit('D'), hit('A')]}
    )
    assert order(fused) == ['B', 'A', 'D', 'C']
    scores = {f.hit.title: f.score for f in fused}
    assert scores['A'] == pytest.approx(1 / 61 + 1 / 63)
    assert scores['B'] == pytest.approx(1 / 62 + 1 / 61)
    assert scores['C'] == pytest.approx(1 / 63)
    assert scores['D'] == pytest.approx(1 / 62)


def test_ranks_are_recorded_per_list():
    fused = rrf_fuse({'vector': [hit('A'), hit('B')], 'keyword': [hit('B')]})
    by_title = {f.hit.title: f.ranks for f in fused}
    assert by_title['A'] == {'vector': 1}
    assert by_title['B'] == {'vector': 2, 'keyword': 1}


def test_deduplicates_across_and_within_lists():
    fused = rrf_fuse({'vector': [hit('A'), hit('A')], 'keyword': [hit('A')]})
    assert len(fused) == 1
    assert fused[0].ranks == {'vector': 1, 'keyword': 1}  # the repeat at rank 2 is ignored
    assert fused[0].score == pytest.approx(2 / 61)


def test_ties_break_deterministically():
    first = rrf_fuse({'vector': [hit('A')], 'keyword': [hit('B')]})
    second = rrf_fuse({'keyword': [hit('B')], 'vector': [hit('A')]})
    assert order(first) == order(second) == ['A', 'B']  # same score and rank: by chunk id


def test_weights_change_the_order():
    rankings = {'vector': [hit('A')], 'keyword': [hit('B')]}
    assert order(rrf_fuse(rankings, weights={'keyword': 2.0})) == ['B', 'A']
    assert order(rrf_fuse(rankings, weights={'vector': 2.0})) == ['A', 'B']


def test_empty_inputs():
    assert rrf_fuse({}) == []
    assert rrf_fuse({'vector': [], 'keyword': []}) == []
    assert order(rrf_fuse({'vector': [hit('A')], 'keyword': []})) == ['A']


def test_invalid_k():
    with pytest.raises(ValueError):
        rrf_fuse({'vector': [hit('A')]}, k=0)


def test_best_rank_hit_is_kept():
    low = hit('A', score=0.1)
    high = hit('A', score=0.9)
    fused = rrf_fuse({'vector': [hit('B'), low], 'keyword': [high]})
    assert next(f for f in fused if f.hit.title == 'A').hit.score == 0.9  # rank 1 in keyword
