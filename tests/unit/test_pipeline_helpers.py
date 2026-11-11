import hashlib

import pytest

from talk_to_documents.ingest.chunker import Chunk
from talk_to_documents.ingest.pipeline import build_embedding_text, collect_files, sha256_file
from talk_to_documents.retrieval.embed import E5_PASSAGE_PREFIX, E5_QUERY_PREFIX, _prefixes_for


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / 'a.txt'
    p.write_bytes(b'hello world' * 1000)
    assert sha256_file(p) == hashlib.sha256(b'hello world' * 1000).hexdigest()


def test_embedding_text_includes_heading_path_only_when_present():
    with_heading = Chunk(0, None, 'Install > Docker', 'Run it.', 2)
    without = Chunk(1, 1, None, 'Plain text.', 2)
    assert build_embedding_text(with_heading) == 'Install > Docker\nRun it.'
    assert build_embedding_text(without) == 'Plain text.'


def test_collect_files(tmp_path):
    (tmp_path / 'b.md').write_text('x')
    (tmp_path / 'a.txt').write_text('x')
    (tmp_path / 'skip.docx').write_text('x')
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'c.pdf').write_text('x')
    assert [p.name for p in collect_files(tmp_path)] == ['a.txt', 'b.md']
    assert [p.name for p in collect_files(tmp_path, recursive=True)] == ['a.txt', 'b.md', 'c.pdf']
    assert collect_files(tmp_path / 'a.txt') == [tmp_path / 'a.txt']
    with pytest.raises(FileNotFoundError):
        collect_files(tmp_path / 'missing')


def test_e5_models_get_prefixes_other_models_do_not():
    assert _prefixes_for('intfloat/multilingual-e5-large') == (E5_QUERY_PREFIX, E5_PASSAGE_PREFIX)
    assert _prefixes_for('BAAI/bge-small-en-v1.5') == ('', '')
