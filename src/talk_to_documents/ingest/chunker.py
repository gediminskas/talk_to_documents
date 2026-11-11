"""Token-aware chunker.

Public API:
    chunk_blocks(blocks, target_tokens, overlap_tokens, count_tokens) -> list[Chunk]
    hf_token_counter(model_name) -> count function backed by the model's own tokenizer

How it works:
    1. Consecutive blocks with the same page and heading path are merged into one group,
       so many small paragraphs (typical for TXT) pack into proper chunks instead of
       producing one tiny chunk per paragraph.
    2. Each group is split into units: paragraphs, then lines, then sentences. A fenced
       code block is kept together as one unit.
    3. A unit longer than the target is hard-split on whitespace.
    4. Units are packed greedily up to `target_tokens`. The next chunk starts with the
       last few units of the previous one (about `overlap_tokens`), never across groups.
    5. A very short tail is merged into the previous chunk when it still fits under the
       hard maximum (target times 1.3).

`count_tokens` is passed in so tests can use a fake counter such as a word count.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache

from talk_to_documents.ingest.parser import Block

TokenCounter = Callable[[str], int]

MAX_TOKENS_FACTOR = 1.3  # hard maximum = target * this (except one unsplittable word)
MIN_TAIL_FRACTION = 0.2  # a tail shorter than target * this is merged into the previous chunk

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?…])\s+')
_FENCE_LINE = re.compile(r'^\s*(```|~~~)')
_WORD = re.compile(r'\S+\s*')


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    page: int | None
    heading_path: str | None
    content: str
    token_count: int


@dataclass(frozen=True)
class _Unit:
    text: str
    sep: str  # separator placed before this unit when it is not first in a chunk
    tokens: int


@lru_cache(maxsize=4)
def hf_token_counter(model_name: str) -> TokenCounter:
    """Token counter using the embedding model's own tokenizer (without special tokens)."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenizer.no_truncation()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count


def _group_blocks(blocks: Iterable[Block]) -> list[tuple[tuple[int | None, str | None], list[str]]]:
    """Merge consecutive blocks that share the same page and heading path."""
    groups: list[tuple[tuple[int | None, str | None], list[str]]] = []
    for block in blocks:
        if not block.text.strip():
            continue
        key = (block.page, block.heading_path)
        if groups and groups[-1][0] == key:
            groups[-1][1].append(block.text)
        else:
            groups.append((key, [block.text]))
    return groups


def _split_paragraphs(text: str) -> list[tuple[str, bool]]:
    """Split on blank lines. Returns (paragraph, is_code); fenced code stays in one piece."""
    paragraphs: list[tuple[str, bool]] = []
    current: list[str] = []
    in_fence = False

    def flush(is_code: bool = False) -> None:
        if current:
            paragraphs.append(('\n'.join(current), is_code))
            current.clear()

    for line in text.split('\n'):
        if _FENCE_LINE.match(line):
            if not in_fence:
                flush()
                in_fence = True
                current.append(line)
            else:
                current.append(line)
                flush(is_code=True)
                in_fence = False
            continue
        if not line.strip() and not in_fence:
            flush()
            continue
        current.append(line)

    flush(is_code=in_fence)  # an unclosed fence is still treated as code
    return paragraphs


def _hard_split(text: str, sep: str, count: TokenCounter, target: int) -> list[_Unit]:
    """Split an oversized piece on whitespace into pieces of at most `target` tokens.

    Rare path, so it simply recounts the candidate piece as it grows. A single word
    longer than the target (for example a huge URL) is emitted on its own.
    """
    units: list[_Unit] = []
    buffer: list[str] = []
    next_sep = sep

    def emit() -> None:
        nonlocal next_sep
        joined = ''.join(buffer)
        piece = joined.strip()
        if piece:
            units.append(_Unit(piece, next_sep, count(piece)))
            trailing = joined[len(joined.rstrip()) :]
            next_sep = '\n' if '\n' in trailing else ' '
        buffer.clear()

    for word in _WORD.findall(text):
        if buffer and count(''.join(buffer + [word]).strip()) > target:
            emit()
        buffer.append(word)
    emit()
    return units


def _fit_unit(text: str, sep: str, count: TokenCounter, target: int) -> list[_Unit]:
    tokens = count(text)
    if tokens <= target:
        return [_Unit(text, sep, tokens)]
    return _hard_split(text, sep, count, target)


def _make_units(text: str, count: TokenCounter, target: int) -> list[_Unit]:
    units: list[_Unit] = []
    for p_index, (paragraph, is_code) in enumerate(_split_paragraphs(text)):
        paragraph_sep = '\n\n' if p_index else ''

        if is_code:
            units.extend(_fit_unit(paragraph, paragraph_sep, count, target))
            continue

        for l_index, line in enumerate(paragraph.split('\n')):
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip()]
            for s_index, sentence in enumerate(sentences):
                if s_index:
                    sep = ' '
                elif l_index:
                    sep = '\n'
                else:
                    sep = paragraph_sep
                units.extend(_fit_unit(sentence, sep, count, target))
    return units


def _overlap_tail(units: list[_Unit], overlap: int) -> list[_Unit]:
    """Trailing units of a finished chunk worth at most `overlap` tokens."""
    if overlap <= 0:
        return []
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units[1:]):  # never reuse the whole chunk as overlap
        if total + unit.tokens > overlap:
            break
        tail.append(unit)
        total += unit.tokens
    tail.reverse()
    return tail


def _pack(
    units: list[_Unit], target: int, overlap: int, max_tokens: int, min_tail: int
) -> list[list[_Unit]]:
    chunks: list[list[_Unit]] = []
    current: list[_Unit] = []
    current_tokens = 0
    carried = 0  # leading units of `current` that are overlap from the previous chunk

    for unit in units:
        if current and current_tokens + unit.tokens > target:
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            tail_tokens = sum(u.tokens for u in tail)
            if tail_tokens + unit.tokens > target:
                tail, tail_tokens = [], 0
            current, current_tokens, carried = list(tail), tail_tokens, len(tail)
        current.append(unit)
        current_tokens += unit.tokens

    if current:
        new_units = current[carried:]
        new_tokens = sum(u.tokens for u in new_units)
        if chunks and new_tokens < min_tail:
            previous_tokens = sum(u.tokens for u in chunks[-1])
            if previous_tokens + new_tokens <= max_tokens:
                chunks[-1] = chunks[-1] + new_units
                return chunks
        chunks.append(current)
    return chunks


def _join(units: list[_Unit]) -> str:
    return ''.join((unit.sep if index else '') + unit.text for index, unit in enumerate(units))


def chunk_blocks(
    blocks: Iterable[Block],
    target_tokens: int,
    overlap_tokens: int,
    count_tokens: TokenCounter,
) -> list[Chunk]:
    if target_tokens <= 0:
        raise ValueError('target_tokens must be positive')
    if not 0 <= overlap_tokens < target_tokens:
        raise ValueError('overlap_tokens must be between 0 and target_tokens - 1')

    max_tokens = int(target_tokens * MAX_TOKENS_FACTOR)
    min_tail = int(target_tokens * MIN_TAIL_FRACTION)

    chunks: list[Chunk] = []
    for (page, heading_path), texts in _group_blocks(blocks):
        units = _make_units('\n\n'.join(texts), count_tokens, target_tokens)
        for chunk_units in _pack(units, target_tokens, overlap_tokens, max_tokens, min_tail):
            content = _join(chunk_units)
            if not content.strip():
                continue
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    page=page,
                    heading_path=heading_path,
                    content=content,
                    token_count=count_tokens(content),
                )
            )
    return chunks
