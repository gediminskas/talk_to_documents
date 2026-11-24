"""Validate eval/questions.jsonl against the database and the corpus files.

Run: uv run python -m eval.validate_dataset
Exit code 1 if there are errors (warnings alone do not fail).
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from eval.dataset import DatasetError, load_questions, validate
from talk_to_documents.db import get_connection
from talk_to_documents.ingest.parsers import ALLOWED_EXTENSIONS, ParseError, parse_file


def parse_corpus(corpus_dir: Path, max_upload_mb: int = 50) -> dict[str, str]:
    """Filename -> full parsed text, for the passage-in-document check."""
    texts: dict[str, str] = {}
    for path in sorted(corpus_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        try:
            parsed = parse_file(path, max_upload_mb)
        except ParseError:
            continue  # a file that cannot be parsed cannot hold a verified passage
        texts[path.name] = '\n\n'.join(block.text for block in parsed.blocks)
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--questions', default='eval/questions.jsonl')
    parser.add_argument('--corpus-dir', default='data/corpus')
    args = parser.parse_args(argv)
    console = Console()

    try:
        questions = load_questions(args.questions)
    except DatasetError as exc:
        console.print(f'[red]{exc}[/red]')
        return 1

    with get_connection() as conn:
        known = {row[0] for row in conn.execute('SELECT filename FROM documents')}
    texts = parse_corpus(Path(args.corpus_dir))

    report = validate(questions, known_documents=known, document_texts=texts)

    if report.issues:
        table = Table(show_header=True, header_style='bold')
        for column in ('level', 'question', 'message'):
            table.add_column(column, overflow='fold')
        for issue in report.issues:
            style = 'red' if issue.level == 'error' else 'yellow'
            table.add_row(f'[{style}]{issue.level}[/{style}]', issue.question_id, issue.message)
        console.print(table)

    console.print(
        f'{len(questions)} questions, {len(report.errors)} error(s), '
        f'{len(report.warnings)} warning(s)'
    )
    return 1 if report.errors else 0


if __name__ == '__main__':
    sys.exit(main())
