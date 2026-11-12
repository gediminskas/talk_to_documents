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
from talk_to_documents.llm import LLMError, LLMResponse

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
        if 'llm' in parts:
            item.add_marker(pytest.mark.llm)


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


class FakeLLM:
    """Records every call. `reply` is a string, or a function (system, user) -> string."""

    def __init__(self, reply='A short context.', fail_times=0):
        self.reply = reply
        self.fail_times = fail_times  # fail this many calls first, then succeed
        self.calls: list[dict] = []

    def complete(self, system, user, max_tokens=512, temperature=0.0, model=None):
        self.calls.append(
            {'system': system, 'user': user, 'max_tokens': max_tokens, 'model': model}
        )
        if self.fail_times > 0:
            self.fail_times -= 1
            raise LLMError('simulated failure')
        text = self.reply(system, user) if callable(self.reply) else self.reply
        return LLMResponse(text=text, input_tokens=None, output_tokens=None, seconds=0.0)


class AlwaysFailingLLM(FakeLLM):
    def __init__(self):
        super().__init__(fail_times=10**9)


class FakeReranker:
    """Scores a text by how many query words it contains."""

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        words = set(query.lower().split())
        return [float(len(words & set(text.lower().split()))) for text in texts]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


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
