"""Ingestion pipeline: file -> parse -> chunk -> embed -> store (one transaction per file)."""

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
from talk_to_documents.ingest.parser import ALLOWED_EXTENSIONS, ParseError, parse_file

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


def build_embedding_text(chunk: Chunk) -> str:
    """Text that is embedded and indexed for keywords: heading path (if any) plus content.
    The stored `content` stays clean for display."""
    if chunk.heading_path:
        return f'{chunk.heading_path}\n{chunk.content}'
    return chunk.content


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
    settings: Settings = default_settings,
) -> IngestResult:
    """Ingest one file. Raises ParseError for unreadable files. Writes nothing on failure."""
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

    texts = [build_embedding_text(chunk) for chunk in chunks]
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
                    Json({'blocks': len(parsed.blocks), 'chunks': len(chunks)}),
                ),
            ).fetchone()[0]

            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks
                      (document_id, ordinal, page, heading_path, content, token_count,
                       embedding, tsv)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, to_tsvector(%s::regconfig, %s))
                    """,
                    [
                        (
                            document_id,
                            chunk.ordinal,
                            chunk.page,
                            chunk.heading_path,
                            chunk.content,
                            chunk.token_count,
                            vector,
                            settings.fts_config,
                            text,
                        )
                        for chunk, vector, text in zip(chunks, vectors, texts, strict=True)
                    ],
                )
    except psycopg.errors.UniqueViolation:
        # Someone ingested the same file between our check and our insert.
        conn.rollback()
        return IngestResult('duplicate', path.name, parsed.title, None, 0, perf_counter() - started)

    conn.commit()
    result = IngestResult(
        'ingested', path.name, parsed.title, document_id, len(chunks), perf_counter() - started
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
                settings=settings,
            )

    outcomes: list[FileOutcome] = []
    for file in files:
        try:
            result = ingest_file(
                file,
                tags,
                conn=conn,
                embedder=embedder,
                count_tokens=count_tokens,
                settings=settings,
            )
            outcomes.append(FileOutcome(file, result=result))
        except Exception as exc:  # report and continue with the next file
            conn.rollback()
            logger.warning('failed to ingest %s: %s', file, exc)
            outcomes.append(FileOutcome(file, error=str(exc)))
    return outcomes
