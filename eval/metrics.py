"""Pure metric functions for the evaluation. No database, no models, no LLM."""

import re
import unicodedata
from collections.abc import Sequence

_PUNCTUATION = str.maketrans(
    {
        '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"',
        '\u2018': "'", '\u2019': "'", '\u201a': "'",
        '\u2013': '-', '\u2014': '-', '\u2212': '-',
        '\u00a0': ' ',
    }
)  # fmt: skip


def normalize(text: str) -> str:
    """Unicode NFC, plain quotes and dashes, collapsed whitespace, case-folded.

    Applied to both gold passages and retrieved chunks before they are compared, so that
    typographic differences never turn a correct retrieval into a miss.
    """
    text = unicodedata.normalize('NFC', text).translate(_PUNCTUATION)
    return ' '.join(text.split()).casefold()


def words(text: str) -> list[str]:
    return re.findall(r'\w+', normalize(text))


def passage_in_chunk(passage: str, chunk_text: str) -> bool:
    needle = normalize(passage)
    return bool(needle) and needle in normalize(chunk_text)


def first_hit_rank(gold_passages: Sequence[str], chunk_texts: Sequence[str]) -> int | None:
    """1-based rank of the first chunk that contains any gold passage, or None."""
    for rank, text in enumerate(chunk_texts, start=1):
        if any(passage_in_chunk(passage, text) for passage in gold_passages):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def recall_at_k(gold_passages: Sequence[str], chunk_texts: Sequence[str], k: int) -> float:
    """Fraction of gold passages found in the top k chunks. This is the metric that matters
    for multi-document questions, where one hit is not enough."""
    if not gold_passages:
        return 0.0
    top = chunk_texts[:k]
    found = sum(1 for passage in gold_passages if any(passage_in_chunk(passage, t) for t in top))
    return found / len(gold_passages)


def reciprocal_rank(rank: int | None, cutoff: int = 10) -> float:
    if rank is None or rank > cutoff:
        return 0.0
    return 1.0 / rank


def doc_hit_at_k(gold_documents: Sequence[str], ranked_documents: Sequence[str], k: int) -> bool:
    """Lenient check: did any gold document appear in the top k? A gap between this and
    hit_at_k points at chunking problems rather than ranking problems."""
    return any(document in set(ranked_documents[:k]) for document in gold_documents)


def word_overlap(question: str, passage: str) -> float:
    """Share of the question's words that also occur in the passage (0 to 1). A high value
    means the question copies the passage, which makes keyword search look better than it is."""
    question_words = set(words(question))
    if not question_words:
        return 0.0
    return len(question_words & set(words(passage))) / len(question_words)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, p from 0 to 100. Empty input gives 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)
