import json

import pytest

from eval.dataset import (
    DatasetError,
    Gold,
    Question,
    load_questions,
    parse_question,
    validate,
)


def make(**overrides) -> Question:
    base = {
        'id': 'q1',
        'type': 'single',
        'language': 'en',
        'split': 'dev',
        'question': 'Which port does the database listen on by default?',
        'gold': (Gold('guide.md', 'The database listens on port 5433 on the host machine.'),),
        'reference_answer': 'Port 5433.',
        'canary': '',
    }
    base.update(overrides)
    return Question(**base)


def messages(report, level='error'):
    issues = report.errors if level == 'error' else report.warnings
    return [i.message for i in issues]


def test_load_questions_roundtrip(tmp_path):
    path = tmp_path / 'q.jsonl'
    row = {
        'id': 'q1', 'type': 'single', 'language': 'en', 'split': 'dev', 'question': 'Why?',
        'gold': [{'document': 'a.md', 'passage': 'Because.'}], 'reference_answer': 'Because.',
    }  # fmt: skip
    path.write_text(json.dumps(row) + '\n\n' + json.dumps({**row, 'id': 'q2'}) + '\n')
    questions = load_questions(path)
    assert [q.id for q in questions] == ['q1', 'q2']
    assert questions[0].passages == ['Because.'] and questions[0].documents == ['a.md']
    assert questions[0].canary == '' and questions[0].answerable


def test_load_errors_name_the_line(tmp_path):
    path = tmp_path / 'q.jsonl'
    path.write_text('{"id": "q1"}\n')
    with pytest.raises(DatasetError, match='line 1.*missing fields'):
        load_questions(path)
    path.write_text('not json\n')
    with pytest.raises(DatasetError, match='line 1.*invalid JSON'):
        load_questions(path)
    with pytest.raises(DatasetError, match='does not exist'):
        load_questions(tmp_path / 'missing.jsonl')


def test_bad_gold_shape_is_rejected():
    row = {
        'id': 'q', 'type': 'single', 'language': 'en', 'split': 'dev', 'question': 'x',
        'gold': ['just a string'], 'reference_answer': 'y',
    }  # fmt: skip
    with pytest.raises(DatasetError, match='gold must be'):
        parse_question(row, 3)


def test_a_good_question_has_no_errors():
    assert validate([make()]).errors == []


def test_duplicate_ids_and_bad_enums():
    report = validate([make(), make(type='weird', language='fr', split='train')])
    text = ' | '.join(messages(report))
    assert 'id used 2 times' in text
    assert 'type must be' in text and 'language must be' in text and 'split must be' in text


def test_gold_rules_by_type():
    assert any('without gold' in m for m in messages(validate([make(gold=())])))
    unanswerable_with_gold = make(type='unanswerable', reference_answer='')
    assert any('empty gold list' in m for m in messages(validate([unanswerable_with_gold])))
    ok = make(type='unanswerable', gold=(), reference_answer='')
    assert not any('gold' in m for m in messages(validate([ok])))

    one_doc = make(type='multi', gold=(Gold('a.md', 'p one here now'), Gold('a.md', 'p two here')))
    assert any('at least 2 documents' in m for m in messages(validate([one_doc])))


def test_canary_rules():
    assert any('need a canary' in m for m in messages(validate([make(type='injection')])))
    assert any('only injection' in m for m in messages(validate([make(canary='PWNED')])))
    assert not any('canary' in m for m in messages(validate([make(type='injection', canary='X1')])))


def test_known_documents_and_verbatim_passages():
    q = make()
    report = validate([q], known_documents={'other.md'})
    assert any('not in the database' in m for m in messages(report))

    ok_text = {'guide.md': 'Intro. The database listens on port 5433 on the host machine. End.'}
    assert validate([q], known_documents={'guide.md'}, document_texts=ok_text).errors == []

    wrong_text = {'guide.md': 'Something else entirely.'}
    report = validate([q], known_documents={'guide.md'}, document_texts=wrong_text)
    assert any('not found verbatim' in m for m in messages(report))


def test_passage_match_ignores_whitespace_and_typography():
    q = make(gold=(Gold('guide.md', 'It’s “quoted” text here'),))
    text = {'guide.md': 'prefix.   It\'s   "quoted"\ntext here.'}
    assert validate([q], known_documents={'guide.md'}, document_texts=text).errors == []


def test_warnings_for_copied_questions_and_odd_passage_length():
    copy = make(
        question='The database listens on port 5433 on the host machine',
    )
    assert any('too easy' in m for m in messages(validate([copy]), 'warning'))
    short = make(gold=(Gold('guide.md', 'Too short'),))
    assert any('words; aim for one sentence' in m for m in messages(validate([short]), 'warning'))


def test_composition_warnings_report_the_gap_to_the_target():
    warnings = messages(validate([make()]), 'warning')
    assert any('single/en: 1 questions, target 12' in m for m in warnings)
    assert any('multi: 0 questions, target 5' in m for m in warnings)
    assert any('1 questions in total, target 30' in m for m in warnings)
