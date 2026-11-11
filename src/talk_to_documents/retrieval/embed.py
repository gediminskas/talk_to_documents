"""Embedding model wrapper.

IMPORTANT: fastembed does NOT add the e5 prefixes for you. For this model family its
`query_embed` and `passage_embed` simply call `embed`. E5 models are trained with the
prefixes 'query: ' (questions) and 'passage: ' (documents), and quality drops without them,
so this wrapper adds them itself.
"""

from functools import lru_cache

import numpy as np

from talk_to_documents.config import settings

E5_QUERY_PREFIX = 'query: '
E5_PASSAGE_PREFIX = 'passage: '


def _prefixes_for(model_name: str) -> tuple[str, str]:
    if 'e5' in model_name.lower():
        return E5_QUERY_PREFIX, E5_PASSAGE_PREFIX
    return '', ''


class Embedder:
    """Loads one fastembed model and embeds queries and passages with correct prefixes."""

    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 16) -> None:
        from fastembed import TextEmbedding  # imported here so --help stays fast

        self.model_name = model_name
        self.batch_size = batch_size
        self._query_prefix, self._passage_prefix = _prefixes_for(model_name)
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    @property
    def dim(self) -> int:
        return int(self._model.embedding_size)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed document chunks. Returns an array of shape (len(texts), dim), float32."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        prefixed = [self._passage_prefix + text for text in texts]
        vectors = list(self._model.embed(prefixed, batch_size=self.batch_size))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Returns a float32 vector of shape (dim,)."""
        vectors = list(self._model.embed([self._query_prefix + text], batch_size=1))
        return np.asarray(vectors[0], dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """The shared embedder for the configured model (loaded once per process)."""
    return Embedder(settings.embed_model, cache_dir=settings.fastembed_cache_path)
