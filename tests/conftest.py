"""Shared fixtures. Integration tests use a separate database, never your real one."""

import hashlib
import re

import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from talk_to_documents.config import Settings, settings
from talk_to_documents.db import init_schema

TEST_DB_NAME = 'ttd_test'
DIM = 1024


def pytest_collection_modifyitems(items):
    """Mark tests by folder so `make test` skips integration and model tests."""
    for item in items:
        parts = item.path.parts
        if 'integration' in parts:
            item.add_marker(pytest.mark.integration)
        if 'model' in parts:
            item.add_marker(pytest.mark.model)


def _url_for(dbname: str) -> str:
    info = conninfo_to_dict(settings.database_url)
    info['dbname'] = dbname
    return make_conninfo(**info)


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts that share words have higher cosine
    similarity. Lets integration tests run without downloading the real model."""

    model_name = 'fake-bag-of-words'
    dim = DIM

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(DIM, dtype=np.float32)
        for word in re.findall(r'\w+', text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = np.linalg.norm(vector)
        if norm == 0:
            vector[0] = 1.0
            return vector
        return vector / norm

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class WrongDimEmbedder(FakeEmbedder):
    """Returns vectors of the wrong size, so the database insert fails mid-transaction."""

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 8), dtype=np.float32)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def word_count():
    return lambda text: len(text.split())


@pytest.fixture(scope='session')
def test_db_url() -> str:
    assert TEST_DB_NAME == 'ttd_test'  # safety: tests must never touch the real database
    admin = psycopg.connect(_url_for('postgres'), autocommit=True)
    try:
        exists = admin.execute(
            'SELECT 1 FROM pg_database WHERE datname = %s', (TEST_DB_NAME,)
        ).fetchone()
        if not exists:
            admin.execute(f'CREATE DATABASE {TEST_DB_NAME}')
    finally:
        admin.close()
    url = _url_for(TEST_DB_NAME)
    init_schema(url=url)
    return url


@pytest.fixture
def test_settings(test_db_url) -> Settings:
    return Settings(
        _env_file=None,
        database_url=test_db_url,
        chunk_target_tokens=60,
        chunk_overlap_tokens=10,
    )


@pytest.fixture
def db_conn(test_db_url):
    conn = psycopg.connect(test_db_url)
    register_vector(conn)
    conn.execute('TRUNCATE chunks, documents')
    conn.commit()
    yield conn
    conn.rollback()
    conn.close()
