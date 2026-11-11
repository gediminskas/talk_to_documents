"""Database helpers: connections, schema initialization, document listing and deletion."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector

from talk_to_documents.config import settings

SCHEMA_PATH = Path(__file__).parent / 'schema.sql'


def get_connection(register: bool = True, url: str | None = None) -> psycopg.Connection:
    """Open a connection. `register=False` is only for the first schema run, because
    registering the vector type needs the extension to exist already."""
    try:
        conn = psycopg.connect(url or settings.database_url)
    except psycopg.OperationalError as exc:
        raise RuntimeError(
            f'Cannot connect to the database ({exc}). Is it running? Try: make up'
        ) from exc
    if register:
        try:
            register_vector(conn)
        except psycopg.ProgrammingError as exc:
            conn.close()
            raise RuntimeError(
                'The database has no pgvector tables yet. Run: uv run ttd db init'
            ) from exc
    return conn


def init_schema(url: str | None = None) -> None:
    """Apply schema.sql. Safe to run repeatedly (every statement uses IF NOT EXISTS)."""
    sql = SCHEMA_PATH.read_text(encoding='utf-8')
    with get_connection(register=False, url=url) as conn:
        conn.execute(sql)


@dataclass(frozen=True)
class DocumentRow:
    id: UUID
    title: str
    filename: str | None
    chunks: int
    created_at: datetime


def list_documents(conn: psycopg.Connection) -> list[DocumentRow]:
    rows = conn.execute(
        """
        SELECT d.id, d.title, d.filename, count(c.id) AS chunks, d.created_at
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.id
        ORDER BY d.created_at, d.title
        """
    ).fetchall()
    return [DocumentRow(*row) for row in rows]


def delete_document(conn: psycopg.Connection, id_or_prefix: str) -> str:
    """Delete a document (its chunks go with it) by full UUID or a unique id prefix.

    Returns the title of the deleted document. Raises LookupError if the prefix matches
    nothing or more than one document.
    """
    prefix = id_or_prefix.strip().lower()
    if not prefix or not all(ch in '0123456789abcdef-' for ch in prefix):
        raise LookupError(f'{id_or_prefix!r} is not a valid document id or id prefix')

    matches = conn.execute(
        "SELECT id, title FROM documents WHERE id::text LIKE %s || '%%'", (prefix,)
    ).fetchall()
    if not matches:
        raise LookupError(f'No document matches {id_or_prefix!r}')
    if len(matches) > 1:
        raise LookupError(f'{id_or_prefix!r} matches {len(matches)} documents; use more characters')

    document_id, title = matches[0]
    conn.execute('DELETE FROM documents WHERE id = %s', (document_id,))
    conn.commit()
    return title
