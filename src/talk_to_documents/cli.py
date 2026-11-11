"""Command line interface. Commands stay thin: they call library functions."""

import functools
import logging
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from talk_to_documents.db import delete_document, get_connection, init_schema, list_documents

app = typer.Typer(
    help='talk_to_documents: hybrid RAG over your own documents.', no_args_is_help=True
)
db_app = typer.Typer(help='Database commands.', no_args_is_help=True)
app.add_typer(db_app, name='db')

console = Console()


def friendly_errors(func: Callable) -> Callable:
    """Show setup problems (database down, schema missing) as one red line, not a traceback."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as exc:
            console.print(f'[red]{exc}[/red]')
            raise typer.Exit(1) from exc

    return wrapper


class SearchMode(StrEnum):
    vector = 'vector'
    keyword = 'keyword'
    both = 'both'


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option('--verbose', '-v', help='Debug logging.')] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format='%(levelname)s %(name)s: %(message)s',
    )


@db_app.command('init')
@friendly_errors
def db_init() -> None:
    """Create the extension, tables and indexes (safe to run repeatedly)."""
    init_schema()
    console.print('schema applied')


@app.command('ingest')
@friendly_errors
def ingest(
    path: Annotated[Path, typer.Argument(help='A file, or a directory of files.')],
    tag: Annotated[
        list[str] | None, typer.Option('--tag', '-t', help='Tag to attach (repeatable).')
    ] = None,
    recursive: Annotated[
        bool, typer.Option('--recursive', '-r', help='Descend into subdirectories.')
    ] = False,
) -> None:
    """Ingest PDF, Markdown and TXT files."""
    from talk_to_documents.ingest.pipeline import ingest_path

    if not path.exists():
        console.print(f'[red]Not found:[/red] {path}')
        raise typer.Exit(1)

    outcomes = ingest_path(path, tag, recursive)

    table = Table(show_header=True, header_style='bold')
    for column in ('file', 'status', 'chunks', 'seconds'):
        table.add_column(column)
    failures = []
    for outcome in outcomes:
        if outcome.result:
            r = outcome.result
            table.add_row(outcome.path.name, r.status, str(r.chunks), f'{r.seconds:.1f}')
        else:
            table.add_row(outcome.path.name, '[red]failed[/red]', '-', '-')
            failures.append(outcome)
    console.print(table)

    for outcome in failures:
        console.print(f'[red]{outcome.path.name}:[/red] {outcome.error}')
    if not outcomes:
        console.print('No supported files found (.pdf, .md, .txt).')
    if failures:
        raise typer.Exit(1)


@app.command('list')
@friendly_errors
def list_cmd() -> None:
    """List ingested documents."""
    with get_connection() as conn:
        documents = list_documents(conn)

    table = Table(show_header=True, header_style='bold')
    for column in ('id', 'title', 'file', 'chunks', 'created'):
        table.add_column(column)
    for doc in documents:
        table.add_row(
            str(doc.id)[:8],
            doc.title,
            doc.filename or '-',
            str(doc.chunks),
            doc.created_at.strftime('%Y-%m-%d %H:%M'),
        )
    console.print(table)
    console.print(f'{len(documents)} document(s)')


@app.command('delete')
@friendly_errors
def delete_cmd(
    document_id: Annotated[str, typer.Argument(help='Full id or unique id prefix.')],
) -> None:
    """Delete a document and its chunks."""
    with get_connection() as conn:
        try:
            title = delete_document(conn, document_id)
        except LookupError as exc:
            console.print(f'[red]{exc}[/red]')
            raise typer.Exit(1) from exc
    console.print(f'deleted: {title}')


def _print_hits(heading: str, hits: list, score_label: str) -> None:
    console.print(f'\n[bold]{heading}[/bold]')
    if not hits:
        console.print('  (no results)')
        return
    table = Table(show_header=True, header_style='bold')
    for column in ('#', score_label, 'document', 'page', 'text'):
        table.add_column(column, overflow='fold')
    for rank, hit in enumerate(hits, start=1):
        snippet = ' '.join(hit.content.split())[:200]
        table.add_row(str(rank), f'{hit.score:.3f}', hit.title, str(hit.page or '-'), snippet)
    console.print(table)


@app.command('search')
@friendly_errors
def search(
    query: Annotated[str, typer.Argument(help='What to look for.')],
    mode: Annotated[SearchMode, typer.Option('--mode', '-m')] = SearchMode.both,
    top_k: Annotated[int, typer.Option('--top-k', '-k', min=1, max=50)] = 5,
) -> None:
    """Search. Day 1 prints the vector and keyword lists separately (fusion comes on Day 2)."""
    from talk_to_documents.retrieval.search import keyword_search, vector_search

    with get_connection() as conn:
        if mode in (SearchMode.vector, SearchMode.both):
            from talk_to_documents.retrieval.embed import get_embedder

            query_vector = get_embedder().embed_query(query)
            _print_hits(
                'Vector search (cosine similarity)', vector_search(conn, query_vector, top_k), 'sim'
            )
        if mode in (SearchMode.keyword, SearchMode.both):
            _print_hits('Keyword search (ts_rank_cd)', keyword_search(conn, query, top_k), 'rank')


if __name__ == '__main__':
    app()
