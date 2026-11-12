from talk_to_documents.ingest.contextual import ContextCache
from talk_to_documents.ingest.pipeline import ingest_file
from talk_to_documents.retrieval.search import keyword_search
from tests.conftest import AlwaysFailingLLM, FakeLLM


def write(tmp_path, name='doc.txt', text='Alpha beta gamma delta epsilon.\n'):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return path


def ingest(path, conn, embedder, counter, settings, **kwargs):
    return ingest_file(
        path, conn=conn, embedder=embedder, count_tokens=counter, settings=settings, **kwargs
    )


def test_contexts_are_stored_searchable_and_content_stays_clean(
    db_conn, fake_embedder, word_count, test_settings, tmp_path
):
    llm = FakeLLM(reply='Quokkas live on Rottnest Island.')
    result = ingest(
        write(tmp_path),
        db_conn,
        fake_embedder,
        word_count,
        test_settings,
        contextual=True,
        llm=llm,
        context_cache=ContextCache(tmp_path / 'c.jsonl'),
    )
    assert result.status == 'ingested' and result.contexts == result.chunks

    content, prefix = db_conn.execute('SELECT content, context_prefix FROM chunks').fetchone()
    assert content == 'Alpha beta gamma delta epsilon.'  # unchanged, clean for display
    assert prefix == 'Quokkas live on Rottnest Island.'

    metadata = db_conn.execute('SELECT metadata FROM documents').fetchone()[0]
    assert metadata['contextual'] is True and metadata['prompt_version'] == 'v1'

    # a word that exists only in the generated context is findable by keyword search
    assert [h.title for h in keyword_search(db_conn, 'quokkas', 5, 'simple')] == ['doc']


def test_contextual_off_makes_no_llm_calls(
    db_conn, fake_embedder, word_count, test_settings, tmp_path
):
    llm = FakeLLM()
    result = ingest(
        write(tmp_path),
        db_conn,
        fake_embedder,
        word_count,
        test_settings,
        contextual=False,
        llm=llm,
    )
    assert llm.calls == [] and result.contexts == 0
    assert db_conn.execute('SELECT context_prefix FROM chunks').fetchone()[0] is None
    assert 'contextual' not in db_conn.execute('SELECT metadata FROM documents').fetchone()[0]


def test_failing_llm_does_not_fail_the_ingest(
    db_conn, fake_embedder, word_count, test_settings, tmp_path
):
    text = ' '.join(f'Sentence number {i} is here.' for i in range(60))  # several chunks
    result = ingest(
        write(tmp_path, text=text),
        db_conn,
        fake_embedder,
        word_count,
        test_settings,
        contextual=True,
        llm=AlwaysFailingLLM(),
        context_cache=ContextCache(None),
    )
    assert result.status == 'ingested' and result.contexts == 0
    assert (
        db_conn.execute('SELECT count(*) FROM chunks WHERE context_prefix IS NOT NULL').fetchone()[
            0
        ]
        == 0
    )
    assert db_conn.execute('SELECT metadata FROM documents').fetchone()[0]['contextual'] == 'failed'


def test_reingest_after_delete_uses_the_cache(
    db_conn, fake_embedder, word_count, test_settings, tmp_path
):
    cache = ContextCache(tmp_path / 'c.jsonl')
    path = write(tmp_path)
    first = FakeLLM(reply='Cached context.')
    ingest(
        path,
        db_conn,
        fake_embedder,
        word_count,
        test_settings,
        contextual=True,
        llm=first,
        context_cache=cache,
    )
    db_conn.execute('DELETE FROM documents')
    db_conn.commit()

    second = FakeLLM(reply='Different context.')
    ingest(
        path,
        db_conn,
        fake_embedder,
        word_count,
        test_settings,
        contextual=True,
        llm=second,
        context_cache=cache,
    )
    assert second.calls == []  # served from the cache
    assert db_conn.execute('SELECT context_prefix FROM chunks').fetchone()[0] == 'Cached context.'
