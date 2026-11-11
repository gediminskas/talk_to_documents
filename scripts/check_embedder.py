"""Task 4 checks for the embedding model. Run: uv run python scripts/check_embedder.py"""

import numpy as np
from fastembed import TextEmbedding

from talk_to_documents.config import settings
from talk_to_documents.retrieval.embed import get_embedder


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    supported = {m['model']: m for m in TextEmbedding.list_supported_models()}
    info = supported.get(settings.embed_model)
    print(f'model {settings.embed_model} supported by fastembed: {bool(info)}')
    if info:
        print(f'  dim={info["dim"]} size_gb={info["size_in_GB"]} license={info.get("license")}')

    embedder = get_embedder()
    print(
        f'loaded; embedding dimension = {embedder.dim} (schema.sql must use vector({embedder.dim}))'
    )

    english = 'The oak tree grows by the river.'
    lithuanian = 'Ąžuolas auga prie upės.'
    unrelated = 'The stock market fell sharply on Monday morning.'
    en, lt, other = embedder.embed_passages([english, lithuanian, unrelated])
    print(f'\nsimilarity English vs its Lithuanian translation: {cosine(en, lt):.3f}')
    print(f'similarity English vs unrelated sentence:          {cosine(en, other):.3f}')
    print('expected: the translation pair is clearly higher')

    query = embedder.embed_query(english)
    passage = embedder.embed_passages([english])[0]
    print(f'\nquery vs passage embedding of the same text: {cosine(query, passage):.4f}')
    print('expected: below 1.0, because the prefixes "query: " and "passage: " differ')


if __name__ == '__main__':
    main()
