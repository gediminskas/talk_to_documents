import psycopg
import pytest

from talk_to_documents.db import delete_document, list_documents
from talk_to_documents.ingest.parser import ParseError
from talk_to_documents.ingest.pipeline import ingest_file, ingest_path
from talk_to_documents.retrieval.search import keyword_search, vector_search
from tests.conftest import WrongDimEmbedder


def ingest(path, conn, embedder, counter, settings, tags=None):
    return ingest_file(
        path, tags, conn=conn, embedder=embedder, count_tokens=counter, settings=settings
    )


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return path


GUIDE = (
    '# Guide\n\n## Install\nRun docker compose to start the database container.\n\n'
    '## Usage\nUse the ttd command to search your documents.\n'
)


def test_ingest_markdown_fills_embedding_and_tsv(
    db_conn, fake_embedder, word_count, test_settings, tmp_path
):
    result = ingest(
        write(tmp_path, 'guide.md', GUIDE),
        db_conn,
        fake_embedder,
        word_count,
        test_settings,
        ['docs'],
    )
    assert result.status == 'ingested' and result.chunks >= 2

    doc = db_conn.execute('SELECT title, embed_model, fts_config, tags FROM documents').fetchone()
    assert doc == ('Guide', 'fake-bag-of-words', 'simple', ['docs'])

    total, with_emb, with_tsv = db_conn.execute(
        'SELECT count(*), count(embedding), count(NULLIF(tsv, to_tsvector(%s))) FROM chunks', ('',)
    ).fetchone()
    assert total == with_emb == with_tsv == result.chunks

    paths = {r[0] for r in db_conn.execute('SELECT heading_path FROM chunks')}
    assert paths == {'Guide > Install', 'Guide > Usage'}


def test_duplicate_file_is_skipped(db_conn, fake_embedder, word_count, test_settings, tmp_path):
    path = write(tmp_path, 'guide.md', GUIDE)
    first = ingest(path, db_conn, fake_embedder, word_count, test_settings)
    copy = write(tmp_path, 'copy-of-guide.md', GUIDE)  # same bytes, different name
    second = ingest(copy, db_conn, fake_embedder, word_count, test_settings)

    assert first.status == 'ingested'
    assert second.status == 'duplicate' and second.document_id == first.document_id
    assert db_conn.execute('SELECT count(*) FROM documents').fetchone()[0] == 1


def test_lithuanian_document_is_found_by_vector_search(
    db_conn, fake_embedder, word_count, test_settings, tmp_path
):
    lt = write(tmp_path, 'ąžuolas.txt', 'Ąžuolas auga prie upės. Žuvys plaukia upe šiandien.\n')
    en = write(tmp_path, 'finance.txt', 'The stock market fell sharply on Monday morning.\n')
    ingest(lt, db_conn, fake_embedder, word_count, test_settings)
    ingest(en, db_conn, fake_embedder, word_count, test_settings)

    hits = vector_search(db_conn, fake_embedder.embed_query('ąžuolas upės'), k=2)
    assert hits[0].title == 'ąžuolas'
    assert hits[0].score > hits[1].score


def test_keyword_search_finds_exact_word_only_in_matching_document(
    db_conn, fake_embedder, word_count, test_settings, tmp_path
):
    ingest(
        write(tmp_path, 'a.txt', 'We deploy everything with kubernetes clusters.\n'),
        db_conn,
        fake_embedder,
        word_count,
        test_settings,
    )
    ingest(
        write(tmp_path, 'b.txt', 'Bake the bread at high temperature.\n'),
        db_conn,
        fake_embedder,
        word_count,
        test_settings,
    )

    hits = keyword_search(db_conn, 'kubernetes', k=5, fts_config='simple')
    assert [h.title for h in hits] == ['a']
    assert keyword_search(db_conn, 'nonexistentword', k=5, fts_config='simple') == []


def test_delete_cascades_and_validates_ids(
    db_conn, fake_embedder, word_count, test_settings, tmp_path
):
    result = ingest(
        write(tmp_path, 'guide.md', GUIDE), db_conn, fake_embedder, word_count, test_settings
    )
    prefix = str(result.document_id)[:8]
    assert [d.title for d in list_documents(db_conn)] == ['Guide']

    assert delete_document(db_conn, prefix) == 'Guide'
    assert db_conn.execute('SELECT count(*) FROM chunks').fetchone()[0] == 0
    with pytest.raises(LookupError):
        delete_document(db_conn, prefix)
    with pytest.raises(LookupError):
        delete_document(db_conn, "'; drop table documents; --")


def test_failed_insert_leaves_nothing_behind(db_conn, word_count, test_settings, tmp_path):
    path = write(tmp_path, 'guide.md', GUIDE)
    with pytest.raises(psycopg.Error):
        ingest(path, db_conn, WrongDimEmbedder(), word_count, test_settings)
    assert db_conn.execute('SELECT count(*) FROM documents').fetchone()[0] == 0
    assert db_conn.execute('SELECT count(*) FROM chunks').fetchone()[0] == 0


def test_ingest_path_continues_after_a_bad_file(
    db_conn, fake_embedder, word_count, test_settings, tmp_path
):
    write(tmp_path, 'empty.txt', '')
    write(tmp_path, 'good.txt', 'A perfectly fine document about retrieval.\n')
    write(tmp_path, 'ignored.docx', 'not supported, not even picked up')

    outcomes = ingest_path(
        tmp_path,
        conn=db_conn,
        embedder=fake_embedder,
        count_tokens=word_count,
        settings=test_settings,
    )
    by_name = {o.path.name: o for o in outcomes}
    assert set(by_name) == {'empty.txt', 'good.txt'}
    assert by_name['empty.txt'].error and 'empty' in by_name['empty.txt'].error
    assert by_name['good.txt'].result.status == 'ingested'


def test_single_unsupported_file_raises(
    db_conn, fake_embedder, word_count, test_settings, tmp_path
):
    with pytest.raises(ParseError):
        ingest(write(tmp_path, 'x.docx', 'hi'), db_conn, fake_embedder, word_count, test_settings)
