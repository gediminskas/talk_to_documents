"""The evaluation question set: loading and validation.

One JSON object per line in eval/questions.jsonl, with the fields:
  id, type, language, split, question, gold, reference_answer, canary
where gold is a list of {document, passage} (document = filename as stored in the database,
passage = a short verbatim quote). Unanswerable questions have an empty gold list.
"""

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from eval.metrics import normalize, passage_in_chunk, word_overlap

TYPES = ('single', 'cross_language', 'multi', 'unanswerable', 'injection')
LANGUAGES = ('en', 'lt')
SPLITS = ('dev', 'test')

# The target composition from the Day 3 plan: (type, language) -> count, and split -> count.
TARGET_BY_KIND = {
    ('single', 'en'): 12,
    ('single', 'lt'): 5,
    ('cross_language', None): 3,
    ('multi', None): 5,
    ('unanswerable', None): 3,
    ('injection', None): 2,
}
TARGET_BY_SPLIT = {'dev': 10, 'test': 20}

MAX_OVERLAP = 0.6  # a question sharing more words than this with its passage is too easy


class DatasetError(ValueError):
    """The question file cannot be read at all (bad JSON, missing fields)."""


@dataclass(frozen=True)
class Gold:
    document: str
    passage: str


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    language: str
    split: str
    question: str
    gold: tuple[Gold, ...]
    reference_answer: str
    canary: str = ''

    @property
    def answerable(self) -> bool:
        return self.type != 'unanswerable'

    @property
    def passages(self) -> list[str]:
        return [g.passage for g in self.gold]

    @property
    def documents(self) -> list[str]:
        return [g.document for g in self.gold]


@dataclass(frozen=True)
class Issue:
    level: str  # 'error' or 'warning'
    question_id: str
    message: str


REQUIRED_FIELDS = ('id', 'type', 'language', 'split', 'question', 'gold', 'reference_answer')


def parse_question(row: dict, line_number: int = 0) -> Question:
    missing = [name for name in REQUIRED_FIELDS if name not in row]
    where = f'line {line_number}' if line_number else 'record'
    if missing:
        raise DatasetError(f'{where}: missing fields {missing}')
    gold = row['gold']
    if not isinstance(gold, list) or not all(
        isinstance(g, dict) and 'document' in g and 'passage' in g for g in gold
    ):
        raise DatasetError(f'{where}: gold must be a list of {{document, passage}} objects')
    return Question(
        id=str(row['id']),
        type=str(row['type']),
        language=str(row['language']),
        split=str(row['split']),
        question=str(row['question']).strip(),
        gold=tuple(Gold(str(g['document']), str(g['passage'])) for g in gold),
        reference_answer=str(row['reference_answer']).strip(),
        canary=str(row.get('canary') or ''),
    )


def load_questions(path: str | Path) -> list[Question]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f'{path} does not exist')
    questions = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f'line {number}: invalid JSON ({exc})') from exc
        questions.append(parse_question(row, number))
    return questions


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == 'error']

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == 'warning']

    def error(self, question_id: str, message: str) -> None:
        self.issues.append(Issue('error', question_id, message))

    def warn(self, question_id: str, message: str) -> None:
        self.issues.append(Issue('warning', question_id, message))


def validate(
    questions: Iterable[Question],
    *,
    known_documents: set[str] | None = None,
    document_texts: dict[str, str] | None = None,
) -> ValidationReport:
    """Check the question set. Errors make the file unusable; warnings are things to look at.

    known_documents: filenames present in the database (None skips that check).
    document_texts: filename -> parsed text (None skips the verbatim-passage check).
    """
    questions = list(questions)
    report = ValidationReport()

    ids = Counter(q.id for q in questions)
    for question_id, count in ids.items():
        if count > 1:
            report.error(question_id, f'id used {count} times')

    for q in questions:
        if q.type not in TYPES:
            report.error(q.id, f'type must be one of {TYPES}, got {q.type!r}')
        if q.language not in LANGUAGES:
            report.error(q.id, f'language must be one of {LANGUAGES}, got {q.language!r}')
        if q.split not in SPLITS:
            report.error(q.id, f'split must be one of {SPLITS}, got {q.split!r}')
        if not q.question:
            report.error(q.id, 'empty question')
        if not q.reference_answer and q.answerable:
            report.error(q.id, 'answerable question without a reference answer')

        if q.type == 'unanswerable':
            if q.gold:
                report.error(q.id, 'unanswerable questions must have an empty gold list')
        else:
            if not q.gold:
                report.error(q.id, 'answerable question without gold passages')
            if any(not g.passage.strip() for g in q.gold):
                report.error(q.id, 'empty gold passage')
        if q.type == 'multi' and len(set(q.documents)) < 2:
            report.error(q.id, 'multi questions need gold passages from at least 2 documents')

        if q.type == 'injection' and not q.canary:
            report.error(q.id, 'injection questions need a canary')
        if q.type != 'injection' and q.canary:
            report.error(q.id, 'only injection questions may have a canary')

        for gold in q.gold:
            passage_words = len(gold.passage.split())
            if passage_words and not 5 <= passage_words <= 40:
                report.warn(
                    q.id,
                    f'passage has {passage_words} words; aim for one sentence (about 8-30 words)',
                )
            if known_documents is not None and gold.document not in known_documents:
                report.error(q.id, f'gold document {gold.document!r} is not in the database')
            if document_texts is not None:
                text = document_texts.get(gold.document)
                if text is None:
                    if known_documents is None or gold.document in known_documents:
                        report.error(q.id, f'no parsed text available for {gold.document!r}')
                elif not passage_in_chunk(gold.passage, text):
                    report.error(q.id, f'passage not found verbatim in {gold.document!r}')
            if word_overlap(q.question, gold.passage) > MAX_OVERLAP:
                report.warn(q.id, 'question shares most of its words with the passage (too easy?)')

    _check_composition(questions, report)
    return report


def _check_composition(questions: list[Question], report: ValidationReport) -> None:
    by_kind = Counter(
        (q.type, q.language if (q.type, q.language) in TARGET_BY_KIND else None) for q in questions
    )
    for kind, target in TARGET_BY_KIND.items():
        actual = by_kind.get(kind, 0)
        if actual != target:
            label = kind[0] if kind[1] is None else f'{kind[0]}/{kind[1]}'
            report.warn('-', f'{label}: {actual} questions, target {target}')
    by_split = Counter(q.split for q in questions)
    for split, target in TARGET_BY_SPLIT.items():
        if by_split.get(split, 0) != target:
            report.warn('-', f'split {split}: {by_split.get(split, 0)} questions, target {target}')
    if len(questions) != sum(TARGET_BY_KIND.values()):
        report.warn(
            '-', f'{len(questions)} questions in total, target {sum(TARGET_BY_KIND.values())}'
        )


__all__ = [
    'DatasetError',
    'Gold',
    'Issue',
    'Question',
    'ValidationReport',
    'load_questions',
    'normalize',
    'parse_question',
    'validate',
]
