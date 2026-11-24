import pytest

from eval.metrics import (
    doc_hit_at_k,
    first_hit_rank,
    hit_at_k,
    normalize,
    passage_in_chunk,
    percentile,
    recall_at_k,
    reciprocal_rank,
    word_overlap,
)


def test_normalize_unifies_typography_and_case():
    assert normalize('  “Hello”   World –  ok\u00a0now ') == '"hello" world - ok now'
    assert normalize('It’s') == "it's"


def test_normalize_handles_lithuanian_decomposed_letters():
    import unicodedata

    decomposed = unicodedata.normalize('NFD', 'Šiandien ąžuolas')
    assert normalize(decomposed) == normalize('Šiandien ąžuolas') == 'šiandien ąžuolas'


def test_passage_in_chunk():
    chunk = 'The  quick brown fox\njumps over the lazy dog.'
    assert passage_in_chunk('quick brown fox jumps', chunk)
    assert passage_in_chunk('QUICK BROWN FOX', chunk)
    assert not passage_in_chunk('quick red fox', chunk)
    assert not passage_in_chunk('', chunk)  # an empty passage never matches


def test_first_hit_rank():
    chunks = ['nothing here', 'the answer is 42', 'also the answer is 42']
    assert first_hit_rank(['answer is 42'], chunks) == 2
    assert first_hit_rank(['not present'], chunks) is None
    assert first_hit_rank(['nothing here', 'answer is 42'], chunks) == 1  # any gold passage
    assert first_hit_rank(['x'], []) is None


def test_hit_at_k():
    assert hit_at_k(3, 5) and hit_at_k(5, 5)
    assert not hit_at_k(6, 5)
    assert not hit_at_k(None, 5)


def test_recall_at_k_counts_gold_passages_found_in_top_k():
    chunks = ['alpha text', 'beta text', 'gamma text', 'delta text']
    gold = ['alpha text', 'gamma text', 'missing passage']
    assert recall_at_k(gold, chunks, 2) == pytest.approx(1 / 3)
    assert recall_at_k(gold, chunks, 4) == pytest.approx(2 / 3)
    assert recall_at_k([], chunks, 4) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0
    assert reciprocal_rank(11, cutoff=10) == 0.0
    assert reciprocal_rank(10, cutoff=10) == 0.1


def test_doc_hit_at_k():
    assert doc_hit_at_k(['a.md'], ['x.md', 'a.md'], 2)
    assert not doc_hit_at_k(['a.md'], ['x.md', 'y.md', 'a.md'], 2)
    assert doc_hit_at_k(['a.md', 'b.md'], ['b.md'], 1)  # any gold document counts


def test_word_overlap():
    assert word_overlap('red fox jumps', 'the red fox jumps high') == 1.0
    assert word_overlap('red fox', 'blue bird') == 0.0
    assert word_overlap('', 'anything') == 0.0
    assert word_overlap('red car', 'a red house') == 0.5


def test_percentile():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 0) == 10
    assert percentile(values, 100) == 50
    assert percentile(values, 95) == pytest.approx(48.0)
    assert percentile([7], 95) == 7
    assert percentile([], 50) == 0.0
    assert percentile([5, 1, 3], 50) == 3  # unsorted input is fine
