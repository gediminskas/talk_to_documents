"""Runs the real embedding model. Slow on first run (about 2.2 GB download): make test-model."""

import numpy as np

from talk_to_documents.config import settings
from talk_to_documents.ingest.chunker import chunk_blocks, hf_token_counter
from talk_to_documents.ingest.parser import Block
from talk_to_documents.retrieval.embed import get_embedder


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_dimension_matches_schema():
    assert get_embedder().dim == 1024  # schema.sql uses vector(1024)


def test_query_and_passage_embeddings_differ():
    embedder = get_embedder()
    text = 'How do I start the database?'
    assert not np.allclose(
        embedder.embed_query(text), embedder.embed_passages([text])[0], atol=1e-4
    )


def test_translation_is_closer_than_unrelated_text():
    embedder = get_embedder()
    english, lithuanian, unrelated = embedder.embed_passages(
        [
            'The oak tree grows by the river.',
            'Ąžuolas auga prie upės.',
            'The stock market fell sharply on Monday morning.',
        ]
    )
    assert cosine(english, lithuanian) > cosine(english, unrelated)


def test_real_chunks_stay_under_model_limit():
    count = hf_token_counter(settings.embed_model)
    text = ' '.join(
        f'Sentence number {i} talks about retrieval and search quality.' for i in range(600)
    )
    chunks = chunk_blocks(
        [Block(text)], settings.chunk_target_tokens, settings.chunk_overlap_tokens, count
    )
    assert chunks
    assert max(chunk.token_count for chunk in chunks) <= 512
