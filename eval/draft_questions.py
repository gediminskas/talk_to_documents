"""Draft candidate evaluation questions from your corpus with the LLM.

Run: uv run python -m eval.draft_questions [--count 60]

This only DRAFTS. Every candidate must be reviewed by hand before it goes into
eval/questions.jsonl (see the review checklist in the plan). Outputs:
    eval/draft.jsonl   machine-readable candidates
    eval/draft.md      the same candidates in a form that is easy to read and review
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console

from eval.metrics import passage_in_chunk, word_overlap
from talk_to_documents.db import get_connection
from talk_to_documents.llm import CachedLLM, LLMError, get_llm
from talk_to_documents.llm_json import complete_json

LITHUANIAN_LETTERS = set('ąčęėįšųūžĄČĘĖĮŠŲŪŽ')
MIN_CHUNK_WORDS = 40
MAX_QUESTION_OVERLAP = 0.6
BANNED_PHRASES = (
    'this document', 'the document', 'the text', 'the passage', 'the chunk', 'the above',
    'šiame dokumente', 'šiame tekste', 'pateiktame tekste', 'aukščiau',
)  # fmt: skip

SYSTEM = 'You write evaluation questions for a document search system.'

USER_TEMPLATE = """Here is a passage from a document:
<passage>
{chunk}
</passage>

Write ONE question that can be answered using only this passage.
Rules:
- The question must be self-contained: it must not refer to "this document", \
"the text" or "the passage".
- Paraphrase. Do not reuse long phrases from the passage in the question, so that a plain keyword \
match is not enough.
- Write the question in {question_language}.
- "quote" must be ONE sentence copied word for word from the passage (between 8 and 30 words) \
that contains the answer.
- "answer" is at most two sentences, in {question_language}.
Return ONLY a JSON object: {{"question": "...", "quote": "...", "answer": "..."}}"""


def guess_language(text: str) -> str:
    """'lt' if the text contains enough Lithuanian-specific letters, otherwise 'en'."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 'en'
    share = sum(1 for c in letters if c in LITHUANIAN_LETTERS) / len(letters)
    return 'lt' if share > 0.02 else 'en'


def select_chunks(rows: list[dict], count: int, per_document: int, seed: int) -> list[dict]:
    """Spread the sample over documents: at most `per_document` chunks from each, round robin."""
    rng = random.Random(seed)
    by_document: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if len(row['content'].split()) >= MIN_CHUNK_WORDS:
            by_document[row['document']].append(row)

    documents = sorted(by_document)
    rng.shuffle(documents)
    for document in documents:
        rng.shuffle(by_document[document])
        by_document[document] = by_document[document][:per_document]

    selected: list[dict] = []
    depth = 0
    while len(selected) < count and any(len(by_document[d]) > depth for d in documents):
        for document in documents:
            if len(selected) >= count:
                break
            if len(by_document[document]) > depth:
                selected.append(by_document[document][depth])
        depth += 1
    return selected


def check_candidate(chunk_text: str, question: str, quote: str, answer: str) -> str | None:
    """Why a candidate must be discarded, or None if it passes the automatic checks."""
    if not (question.strip() and quote.strip() and answer.strip()):
        return 'empty field'
    if not passage_in_chunk(quote, chunk_text):
        return 'quote is not verbatim in the chunk'
    quote_words = len(quote.split())
    if not 6 <= quote_words <= 40:
        return f'quote has {quote_words} words'
    if len(question.split()) < 3:
        return 'question too short'
    lowered = question.lower()
    if any(phrase in lowered for phrase in BANNED_PHRASES):
        return 'question refers to the document'
    if word_overlap(question, quote) > MAX_QUESTION_OVERLAP:
        return 'question copies the quote (too easy)'
    return None


def load_chunk_rows(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT c.id, d.filename, d.title, c.content
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.source_type = 'upload'
        ORDER BY d.filename, c.ordinal
        """
    ).fetchall()
    return [{'chunk_id': str(r[0]), 'document': r[1], 'title': r[2], 'content': r[3]} for r in rows]


def draft_candidates(
    llm, chunks: list[dict], cross_every: int = 5, console: Console | None = None
) -> tuple[list[dict], list[str]]:
    """Ask the LLM for one question per chunk. Every `cross_every`-th candidate is asked in the
    other language (a cross-language question). Returns (candidates, discard reasons)."""
    candidates: list[dict] = []
    discarded: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source_language = guess_language(chunk['content'])
        cross = index % cross_every == 0
        question_language = (
            ('en' if source_language == 'lt' else 'lt') if cross else source_language
        )
        language_name = {'en': 'English', 'lt': 'Lithuanian'}[question_language]

        try:
            reply = complete_json(
                llm,
                SYSTEM,
                USER_TEMPLATE.format(chunk=chunk['content'], question_language=language_name),
                required_keys=['question', 'quote', 'answer'],
                max_tokens=500,
            )
        except LLMError as exc:
            discarded.append(f'{chunk["document"]}: LLM failed ({exc})')
            continue

        data = reply.data
        question, quote, answer = (str(data[k]).strip() for k in ('question', 'quote', 'answer'))
        problem = check_candidate(chunk['content'], question, quote, answer)
        if problem:
            discarded.append(f'{chunk["document"]}: {problem}')
            continue

        candidates.append(
            {
                'id': f'draft-{len(candidates) + 1:03d}',
                'type': 'cross_language' if cross else 'single',
                'language': question_language,
                'split': '',
                'question': question,
                'gold': [{'document': chunk['document'], 'passage': quote}],
                'reference_answer': answer,
                'canary': '',
                'source_language': source_language,
            }
        )
        if console:
            console.print(f'  {index}/{len(chunks)} kept {len(candidates)}', end='\r')
    return candidates, discarded


def render_review_markdown(candidates: list[dict]) -> str:
    lines = [
        '# Draft questions to review',
        '',
        'For each: is the quote really the answer? Is the question self-contained and '
        'unambiguous? Does another document answer it too? Is it a paraphrase? '
        'Copy the good ones into eval/questions.jsonl and set type, split and fix wording.',
        '',
    ]
    for c in candidates:
        gold = c['gold'][0]
        lines += [
            f'## {c["id"]} ({c["type"]}, question in {c["language"]}, '
            f'source in {c["source_language"]})',
            f'- **Question:** {c["question"]}',
            f'- **Reference answer:** {c["reference_answer"]}',
            f'- **Document:** {gold["document"]}',
            f'- **Quote:** {gold["passage"]}',
            '',
        ]
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--count', type=int, default=60, help='chunks to sample')
    parser.add_argument('--per-document', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out-dir', default='eval')
    parser.add_argument('--cache', default='.cache/eval-draft.jsonl')
    args = parser.parse_args(argv)
    console = Console()

    with get_connection() as conn:
        rows = load_chunk_rows(conn)
    chunks = select_chunks(rows, args.count, args.per_document, args.seed)
    if not chunks:
        console.print('[red]No suitable chunks found. Ingest your corpus first.[/red]')
        return 1
    console.print(
        f'Sampled {len(chunks)} chunks from {len({c["document"] for c in chunks})} documents.'
    )

    llm = CachedLLM(get_llm(), args.cache)  # a restart resumes without repeating finished calls
    candidates, discarded = draft_candidates(llm, chunks, console=console)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'draft.jsonl').write_text(
        '\n'.join(json.dumps(c, ensure_ascii=False) for c in candidates) + '\n', encoding='utf-8'
    )
    (out_dir / 'draft.md').write_text(render_review_markdown(candidates), encoding='utf-8')

    console.print(f'\nKept {len(candidates)} candidates, discarded {len(discarded)}.')
    for reason in discarded[:15]:
        console.print(f'  [dim]discarded: {reason}[/dim]')
    console.print(f'Review {out_dir / "draft.md"}; LLM calls: {llm.calls}, cached: {llm.hits}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
