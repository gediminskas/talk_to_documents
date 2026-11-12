import pytest

from talk_to_documents.ingest.pipeline import ingest_file
from talk_to_documents.retrieval.hybrid import hybrid_search
from talk_to_documents.retrieval.search import keyword_search, vector_search


def add(path_dir, name, text, conn, embedder, counter, settings, tags=None):
    path = path_dir / name
    path.write_text(text, encoding='utf-8')
    return ingest_file(
        path, tags, conn=conn, embedder=embedder, count_tokens=counter, settings=settings
    )


@pytest.fixture
def corpus(db_conn, fake_embedder, word_count, test_settings, tmp_path):
    """A: contains both query words but is dominated by filler (weak for the vector side).
    B: repeats one query word (strong for the vector side, but misses the other word)."""
    filler = ' '.join(f'filler{i}' for i in range(30))
    args = (db_conn, fake_embedder, word_count, test_settings)
    add(tmp_path, 'a.txt', f'zebra quagga {filler}', *args, tags=['x'])
    add(tmp_path, 'b.txt', 'zebra zebra zebra zebra', *args, tags=['y'])
    add(tmp_path, 'c.txt', 'completely unrelated text about bread and butter', *args)
    return db_conn


def titles(result):
    return [f.hit.title for f in result.hits]


def test_modes_see_different_documents_and_hybrid_gets_both(corpus, fake_embedder, test_settings):
    kwargs = {'embedder': fake_embedder, 'settings': test_settings, 'candidates': 1, 'top_k': 5}

    assert titles(hybrid_search(corpus, 'zebra quagga', mode='vector', **kwargs)) == ['b']
    assert titles(hybrid_search(corpus, 'zebra quagga', mode='keyword', **kwargs)) == ['a']

    both = hybrid_search(corpus, 'zebra quagga', mode='hybrid', **kwargs)
    assert set(titles(both)) == {'a', 'b'}  # each side contributed one document
    assert {f.hit.title: set(f.ranks) for f in both.hits} == {'a': {'keyword'}, 'b': {'vector'}}


def test_hybrid_ranks_a_document_found_by_both_lists_first(corpus, fake_embedder, test_settings):
    result = hybrid_search(
        corpus, 'zebra', embedder=fake_embedder, settings=test_settings, mode='hybrid', top_k=3
    )
    best = result.hits[0]
    assert set(best.ranks) == {'vector', 'keyword'}
    assert best.hit.title == 'b'  # four mentions: top of both lists


def test_hybrid_rerank_uses_candidates_and_reports_timings(
    corpus, fake_embedder, fake_reranker, test_settings
):
    result = hybrid_search(
        corpus,
        'zebra quagga',
        embedder=fake_embedder,
        reranker=fake_reranker,
        settings=test_settings,
        rerank=True,
        top_k=2,
    )
    assert titles(result)[0] == 'a'  # the fake reranker counts matching words: a has both
    assert result.hits[0].rerank_score == 2.0
    assert result.timings.embed_ms > 0 and result.timings.total_ms > 0


def test_tag_filter_applies_to_both_searches(corpus, fake_embedder):
    vector = vector_search(corpus, fake_embedder.embed_query('zebra'), k=10, tags=['x'])
    keyword = keyword_search(corpus, 'zebra', k=10, fts_config='simple', tags=['x'])
    assert {h.title for h in vector} == {'a'}
    assert {h.title for h in keyword} == {'a'}
    assert {h.title for h in keyword_search(corpus, 'zebra', 10, 'simple', tags=['nope'])} == set()


def test_source_type_filter_applies_to_both_searches(corpus, fake_embedder):
    corpus.execute("UPDATE documents SET source_type = 'note' WHERE title = 'b'")
    corpus.commit()
    vector = vector_search(corpus, fake_embedder.embed_query('zebra'), k=10, source_type='note')
    keyword = keyword_search(corpus, 'zebra', k=10, fts_config='simple', source_type='note')
    assert {h.title for h in vector} == {'b'}
    assert {h.title for h in keyword} == {'b'}
    none = vector_search(corpus, fake_embedder.embed_query('zebra'), k=10, source_type='upload')
    assert 'b' not in {h.title for h in none}


def test_unfiltered_search_still_works_with_null_filters(corpus, fake_embedder):
    assert len(vector_search(corpus, fake_embedder.embed_query('zebra'), k=10)) == 3
    assert len(keyword_search(corpus, 'zebra', k=10, fts_config='simple')) == 2
