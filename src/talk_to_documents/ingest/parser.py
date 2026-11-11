"""File parsers: turn PDF, Markdown and TXT files into normalized text blocks.

Public API:
    parse_file(path, max_upload_mb) -> ParsedDocument
    normalize_text(text) -> str
    ParseError

Block conventions (the chunker relies on these):
    PDF       one block per page, `page` is 1-based, no heading path
    Markdown  one block per heading section, `heading_path` like 'Install > Docker'
    TXT       one block per paragraph, no page, no heading path
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

ALLOWED_EXTENSIONS = {'.pdf', '.md', '.txt'}
PDF_MAGIC = b'%PDF-'

_HYPHEN_LINE_BREAK = re.compile(r'(\w)-\n(\w)')
_HEADING = re.compile(r'^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$')
_FENCE = re.compile(r'^\s*(```|~~~)')


class ParseError(Exception):
    """Raised for unsupported, unreadable or empty files, with a readable message."""


@dataclass(frozen=True)
class Block:
    text: str
    page: int | None = None
    heading_path: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    blocks: list[Block]


def normalize_text(text: str) -> str:
    """Normalize text so that identical content always compares and searches the same.

    NFC matters for Lithuanian: letters with diacritics can be stored either as one
    code point or as a base letter plus a combining mark, which breaks keyword search.
    """
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\x00', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _dehyphenate(text: str) -> str:
    """Join words split by a hyphen at a line break, only if the next char is lowercase."""

    def join(match: re.Match[str]) -> str:
        if match.group(2).islower():
            return match.group(1) + match.group(2)
        return match.group(0)

    return _HYPHEN_LINE_BREAK.sub(join, text)


def validate_file(path: Path, max_upload_mb: int = 10) -> None:
    """Cheap checks before any parsing. Raises ParseError on failure."""
    if not path.is_file():
        raise ParseError(f'{path}: not a file')

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ', '.join(sorted(ALLOWED_EXTENSIONS))
        raise ParseError(f'{path.name}: unsupported file type {suffix!r} (allowed: {allowed})')

    size = path.stat().st_size
    if size == 0:
        raise ParseError(f'{path.name}: file is empty')
    if size > max_upload_mb * 1024 * 1024:
        raise ParseError(f'{path.name}: file is larger than {max_upload_mb} MB')

    if suffix == '.pdf':
        with path.open('rb') as handle:
            head = handle.read(1024)
        if PDF_MAGIC not in head:
            raise ParseError(f'{path.name}: does not look like a PDF (missing %PDF- header)')


def _read_text(path: Path) -> str:
    # 'utf-8-sig' reads plain UTF-8 and also strips a BOM if one is present.
    try:
        return path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ParseError(f'{path.name}: not valid UTF-8 text; re-save it as UTF-8') from exc


def _parse_pdf(path: Path) -> ParsedDocument:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ParseError(f'{path.name}: PDF is encrypted')

        blocks: list[Block] = []
        for number, page in enumerate(reader.pages, start=1):
            text = _dehyphenate(normalize_text(page.extract_text() or ''))
            if text:
                blocks.append(Block(text=text, page=number))

        meta_title = reader.metadata.title if reader.metadata else None
    except ParseError:
        raise
    except Exception as exc:  # pypdf can raise many different errors on damaged files
        raise ParseError(f'{path.name}: could not read PDF ({exc})') from exc

    if not blocks:
        raise ParseError(
            f'{path.name}: no extractable text; the file looks scanned and OCR is not supported'
        )

    title = (meta_title or '').strip() or path.stem
    return ParsedDocument(title=title, blocks=blocks)


def _strip_front_matter(text: str) -> str:
    """Remove a YAML front matter block at the very start of a Markdown file."""
    if not text.startswith('---\n'):
        return text
    end = text.find('\n---', 4)
    if end == -1:
        return text
    newline = text.find('\n', end + 1)
    return text[newline + 1 :] if newline != -1 else ''


def _parse_markdown(path: Path) -> ParsedDocument:
    text = _strip_front_matter(normalize_text(_read_text(path)))

    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []  # (heading level, heading title)
    buffer: list[str] = []
    first_h1: str | None = None
    in_fence = False

    def flush() -> None:
        body = '\n'.join(buffer).strip()
        buffer.clear()
        if body:
            heading_path = ' > '.join(title for _, title in stack) or None
            blocks.append(Block(text=body, heading_path=heading_path))

    for line in text.split('\n'):
        if _FENCE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue

        match = None if in_fence else _HEADING.match(line)
        if match is None:
            buffer.append(line)
            continue

        flush()
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        if level == 1 and first_h1 is None:
            first_h1 = title

    flush()

    if not blocks:
        raise ParseError(f'{path.name}: no text content found')
    return ParsedDocument(title=first_h1 or path.stem, blocks=blocks)


def _parse_text(path: Path) -> ParsedDocument:
    text = normalize_text(_read_text(path))
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text)]
    blocks = [Block(text=p) for p in paragraphs if p]

    if not blocks:
        raise ParseError(f'{path.name}: no text content found')
    return ParsedDocument(title=path.stem, blocks=blocks)


def parse_file(path: str | Path, max_upload_mb: int = 10) -> ParsedDocument:
    """Validate and parse a PDF, Markdown or TXT file."""
    path = Path(path)
    validate_file(path, max_upload_mb)

    suffix = path.suffix.lower()
    if suffix == '.pdf':
        return _parse_pdf(path)
    if suffix == '.md':
        return _parse_markdown(path)
    return _parse_text(path)
