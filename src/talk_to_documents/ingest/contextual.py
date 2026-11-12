"""Contextual retrieval: an LLM writes a short context for each chunk at ingest time.

The context is prepended to the text that is embedded and keyword-indexed; the displayed
chunk content stays untouched. Generation is cached on disk, so re-ingesting a corpus never
pays for the same context twice, and a flaky LLM never makes an ingest fail.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from talk_to_documents.ingest.chunker import Chunk
from talk_to_documents.llm import LLM, LLMError, strip_thinking

logger = logging.getLogger(__name__)

PROMPT_VERSION = 'v1'
MAX_CONTEXT_CHARS = 400
EXCERPT_NEIGHBOURS = 2  # chunks before and after, used when the document is too long

SYSTEM_PROMPT = 'You write short search context for document chunks.'

USER_TEMPLATE = """<{tag}>
{source}
</{tag}>
Here is the chunk we want to situate within the {scope}:
<chunk>
{chunk}
</chunk>
Give a short, succinct context (one or two sentences) that situates this chunk within the \
overall document, to improve search retrieval of the chunk. Write it in the same language as \
the document. Answer only with the context and nothing else."""


@dataclass(frozen=True)
class ContextResult:
    contexts: list[str]  # exactly one per chunk; '' where no context is available
    failed: bool  # True if generation was abandoned (all contexts are then empty)
    generated: int  # LLM calls that produced a context
    cached: int  # contexts served from the cache


class ContextCache:
    """JSON-lines file of {key, context}. With path=None it only lives in memory."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._data: dict[str, str] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                try:
                    row = json.loads(line)
                    self._data[row['key']] = row['context']
                except (json.JSONDecodeError, KeyError):
                    continue  # ignore a damaged line

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, context: str) -> None:
        self._data[key] = context
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(
                    json.dumps({'key': key, 'context': context}, ensure_ascii=False) + '\n'
                )


def cache_key(model: str, source_text: str, chunk_text: str) -> str:
    payload = json.dumps([model, PROMPT_VERSION, source_text, chunk_text], ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _clean(text: str) -> str:
    text = ' '.join(strip_thinking(text).split())
    return text[:MAX_CONTEXT_CHARS].rstrip()


def _excerpt(title: str, chunks: list[Chunk], index: int) -> str:
    start = max(0, index - EXCERPT_NEIGHBOURS)
    stop = min(len(chunks), index + EXCERPT_NEIGHBOURS + 1)
    heading = chunks[index].heading_path
    parts = [f'Document title: {title}']
    if heading:
        parts.append(f'Section: {heading}')
    parts.extend(chunk.content for chunk in chunks[start:stop])
    return '\n\n'.join(parts)


def generate_contexts(
    title: str,
    document_text: str,
    chunks: list[Chunk],
    llm: LLM,
    *,
    model: str,
    max_doc_tokens: int,
    max_output_tokens: int,
    count_tokens: Callable[[str], int],
    cache: ContextCache | None = None,
    max_chunks: int | None = None,
    max_consecutive_failures: int = 5,
) -> ContextResult:
    """One context per chunk.

    If the document fits in `max_doc_tokens` the whole document is sent with every chunk;
    otherwise an excerpt (title, section, and neighbouring chunks) is sent instead. After
    `max_consecutive_failures` LLM errors in a row generation is abandoned and ALL contexts
    for the document are discarded, so a document is never indexed half with and half without.
    """
    contexts = [''] * len(chunks)
    use_whole_document = count_tokens(document_text) <= max_doc_tokens
    generated = cached = consecutive_failures = 0

    for index, chunk in enumerate(chunks):
        if max_chunks is not None and index >= max_chunks:
            logger.warning(
                '%s: only the first %d of %d chunks get a context', title, max_chunks, len(chunks)
            )
            break

        if use_whole_document:
            tag, scope, source = 'document', 'whole document', document_text
        else:
            tag, scope, source = (
                'excerpt',
                'excerpt of the document',
                _excerpt(title, chunks, index),
            )

        key = cache_key(model, source, chunk.content)
        if cache is not None and (hit := cache.get(key)) is not None:
            contexts[index] = hit
            cached += 1
            consecutive_failures = 0
            continue

        user = USER_TEMPLATE.format(tag=tag, scope=scope, source=source, chunk=chunk.content)
        try:
            response = llm.complete(
                SYSTEM_PROMPT, user, max_tokens=max_output_tokens, temperature=0.0, model=model
            )
        except LLMError as exc:
            consecutive_failures += 1
            logger.warning('%s chunk %d: context generation failed: %s', title, index, exc)
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    '%s: giving up on contexts after %d failures in a row',
                    title,
                    consecutive_failures,
                )
                return ContextResult([''] * len(chunks), True, 0, 0)
            continue

        consecutive_failures = 0
        context = _clean(response.text)
        if context:
            contexts[index] = context
            generated += 1
            if cache is not None:
                cache.put(key, context)

    return ContextResult(contexts, False, generated, cached)
