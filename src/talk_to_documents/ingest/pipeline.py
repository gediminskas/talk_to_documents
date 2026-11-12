"""Ingestion pipeline: file -> parse -> chunk -> (contexts) -> embed -> store.

Each file is written in one transaction, so a failure leaves nothing behind.
"""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import UUID

import numpy as np
import psycopg
from psycopg.types.json import Json

from talk_to_documents.config import Settings
from talk_to_documents.config import settings as default_settings
from talk_to_documents.db import get_connection
from talk_to_documents.ingest.chunker import Chunk, chunk_blocks, hf_token_counter
from talk_to_documents.ingest.contextual import (
    PROMPT_VERSION,
    ContextCache,
    generate_contexts,
)
from talk_to_documents.ingest.parser import ALLOWED_EXTENSIONS, ParseError, parse_file
from talk_to_documents.llm import LLM

logger = logging.getLogger(__name__)


class PassageEmbedder(Protocol):
    model_name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class IngestResult:
    status: str  # 'ingested' or 'duplicate'
    filename: str
    title: str
    document_id: UUID | None
    chunks: int
    seconds: float
    contexts: int = 0  # chunks that received an LLM-written context


@dataclass(frozen=True)
class FileOutcome:
    path: Path
    result: IngestResult | None = None
    error: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_embedding_text(chunk: Chunk, context: str = '') -> str:
    """Text that is embedded and keyword-indexed: context (if any), heading path (if any),
    then content. The stored `content` stays clean for display."""
    parts = [context, chunk.heading_path or '', chunk.content]
    return '\n'.join(part for part in parts if part)


def collect_files(path: Path, recursive: bool = False) -> list[Path]:
    """A single file, or the supported files inside a directory (sorted)."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = path.rglob('*') if recursive else path.glob('*')
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)


def ingest_file(
    path: str | Path,
    tags: list[str] | None = None,
    *,
    conn: psycopg.Connection | None = None,
    embedder: PassageEmbedder | None = None,
    count_tokens: Callable[[str], int] | None = None,
    contextual: bool | None = None,
    llm: LLM | None = None,
    context_cache: ContextCache | None = None,
    settings: Settings = default_settings,
) -> IngestResult:
    """Ingest one file. Raises ParseError for unreadable files. Writes nothing on failure.

    `contextual=None` means: use the CONTEXTUAL setting.
    """
    path = Path(path)
    started = perf_counter()

    if conn is None:
        with get_connection() as own_conn:
            return ingest_file(
                path,
                tags,
                conn=own_conn,
                embedder=embedder,
                count_tokens=count_tokens,
                contextual=contextual,
                llm=llm,
                context_cache=context_cache,
                settings=settings,
            )

    # Validate and parse first so bad files fail before any work is done.
    parsed = parse_file(path, settings.max_upload_mb)
    digest = sha256_file(path)

    existing = conn.execute(
        'SELECT id, title FROM documents WHERE sha256 = %s', (digest,)
    ).fetchone()
    conn.commit()  # end the read-only transaction; the write below gets its own
    if existing:
        return IngestResult(
            'duplicate', path.name, existing[1], existing[0], 0, perf_counter() - started
        )

    if embedder is None:
        from talk_to_documents.retrieval.embed import get_embedder

        embedder = get_embedder()
    if count_tokens is None:
        count_tokens = hf_token_counter(settings.embed_model)

    chunks = chunk_blocks(
        parsed.blocks, settings.chunk_target_tokens, settings.chunk_overlap_tokens, count_tokens
    )
    if not chunks:
        raise ParseError(f'{path.name}: produced no chunks')

    metadata: dict = {'blocks': len(parsed.blocks), 'chunks': len(chunks)}
    contexts = [''] * len(chunks)
    use_contextual = settings.contextual if contextual is None else contextual
    if use_contextual:
        if llm is None:
            from talk_to_documents.llm import get_llm

            llm = get_llm(settings)
        if context_cache is None:
            context_cache = ContextCache(settings.context_cache_path)
        context_model = settings.context_model or settings.llm_model or 'unknown'
        result = generate_contexts(
            parsed.title,
            '\n\n'.join(block.text for block in parsed.blocks),
            chunks,
            llm,
            model=context_model,
            max_doc_tokens=settings.context_max_doc_tokens,
            max_output_tokens=settings.context_max_output_tokens,
            count_tokens=count_tokens,
            cache=context_cache,
            max_chunks=settings.context_max_chunks_per_doc,
        )
        contexts = result.contexts
        metadata.update(
            contextual='failed' if result.failed else True,
            context_model=context_model,
            prompt_version=PROMPT_VERSION,
        )

    texts = [build_embedding_text(chunk, ctx) for chunk, ctx in zip(chunks, contexts, strict=True)]
    vectors = embedder.embed_passages(texts)  # before the transaction, so it stays short

    try:
        with conn.transaction():
            document_id = conn.execute(
                """
                INSERT INTO documents
                  (title, source_type, filename, sha256, fts_config, embed_model, tags, metadata)
                VALUES (%s, 'upload', %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    parsed.title,
                    path.name,
                    digest,
                    settings.fts_config,
                    embedder.model_name,
                    tags or [],
                    Json(metadata),
                ),
            ).fetchone()[0]

            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks
                      (document_id, ordinal, page, heading_path, content, context_prefix,
                       token_count, embedding, tsv)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_tsvector(%s::regconfig, %s))
                    """,
                    [
                        (
                            document_id,
                            chunk.ordinal,
                            chunk.page,
                            chunk.heading_path,
                            chunk.content,
                            ctx or None,
                            chunk.token_count,
                            vector,
                            settings.fts_config,
                            text,
                        )
                        for chunk, ctx, vector, text in zip(
                            chunks, contexts, vectors, texts, strict=True
                        )
                    ],
                )
    except psycopg.errors.UniqueViolation:
        # Someone ingested the same file between our check and our insert.
        conn.rollback()
        return IngestResult('duplicate', path.name, parsed.title, None, 0, perf_counter() - started)

    conn.commit()
    result = IngestResult(
        'ingested',
        path.name,
        parsed.title,
        document_id,
        len(chunks),
        perf_counter() - started,
        contexts=sum(1 for ctx in contexts if ctx),
    )
    logger.info('ingested %s: %d chunks in %.1fs', path.name, result.chunks, result.seconds)
    return result


def ingest_path(
    path: str | Path,
    tags: list[str] | None = None,
    recursive: bool = False,
    *,
    conn: psycopg.Connection | None = None,
    embedder: PassageEmbedder | None = None,
    count_tokens: Callable[[str], int] | None = None,
    contextual: bool | None = None,
    llm: LLM | None = None,
    context_cache: ContextCache | None = None,
    settings: Settings = default_settings,
) -> list[FileOutcome]:
    """Ingest a file or a directory. One failing file never stops the others."""
    files = collect_files(Path(path), recursive)

    if conn is None:
        with get_connection() as own_conn:
            return ingest_path(
                path,
                tags,
                recursive,
                conn=own_conn,
                embedder=embedder,
                count_tokens=count_tokens,
                contextual=contextual,
                llm=llm,
                context_cache=context_cache,
                settings=settings,
            )

    # Create the LLM client and the context cache once for the whole run, not per file.
    if settings.contextual if contextual is None else contextual:
        contextual = True
        if llm is None:
            from talk_to_documents.llm import get_llm

            llm = get_llm(settings)
        if context_cache is None:
            context_cache = ContextCache(settings.context_cache_path)

    outcomes: list[FileOutcome] = []
    for file in files:
        try:
            result = ingest_file(
                file,
                tags,
                conn=conn,
                embedder=embedder,
                count_tokens=count_tokens,
                contextual=contextual,
                llm=llm,
                context_cache=context_cache,
                settings=settings,
            )
            outcomes.append(FileOutcome(file, result=result))
        except Exception as exc:  # report and continue with the next file
            conn.rollback()
            logger.warning('failed to ingest %s: %s', file, exc)
            outcomes.append(FileOutcome(file, error=str(exc)))
    return outcomes
