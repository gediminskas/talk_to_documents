import json

import pytest

from eval import run_eval
from eval.aggregate import aggregate_retrieval
from eval.dataset import Gold, Question
from eval.report import render_report
from eval.run_eval import CONFIGS, retrieve_for_config, url_with_dbname
from talk_to_documents.db import get_connection as real_get_connection
from talk_to_documents.ingest.pipeline import ingest_file
from tests.conftest import FakeLLM


@pytest.fixture
def corpus(db_conn, fake_embedder, word_count, test_settings, tmp_path):
    filler = ' '.join(f'filler{i}' for i in range(30))
    files = {
        'a.txt': f'zebra quagga {filler}',
        'b.txt': 'zebra zebra zebra zebra',
        'c.txt': 'completely unrelated text about bread and butter',
    }
    for name, text in files.items():
        path = tmp_path / name
        path.write_text(text, encoding='utf-8')
        ingest_file(
            path,
            conn=db_conn,
            embedder=fake_embedder,
            count_tokens=word_count,
            settings=test_settings,
        )
    return db_conn


QUESTIONS = [
    Question(
        'q1', 'single', 'en', 'dev', 'zebra quagga', (Gold('a.txt', 'zebra quagga'),), 'a', ''
    ),
    Question('q2', 'single', 'en', 'test', 'zebra', (Gold('b.txt', 'zebra zebra'),), 'b', ''),
    Question('q3', 'unanswerable', 'en', 'test', 'bananas', (), '', ''),
]


def run(corpus, fake_embedder, fake_reranker, test_settings, name):
    return retrieve_for_config(
        corpus, QUESTIONS, CONFIGS[name],
        embedder=fake_embedder, reranker=fake_reranker, settings=test_settings,
    )  # fmt: skip


def by_id(rows):
    return {row['id']: row for row in rows}


def test_each_configuration_scores_as_designed(corpus, fake_embedder, fake_reranker, test_settings):
    keyword, _ = run(corpus, fake_embedder, fake_reranker, test_settings, 'A')
    vector, _ = run(corpus, fake_embedder, fake_reranker, test_settings, 'B')
    hybrid, _ = run(corpus, fake_embedder, fake_reranker, test_settings, 'C')
    reranked, _ = run(corpus, fake_embedder, fake_reranker, test_settings, 'D')

    # q1: keyword needs both words (only a.txt has them); vector prefers b.txt, then a.txt
    assert by_id(keyword)['q1']['rank'] == 1
    assert by_id(vector)['q1']['rank'] == 2 and by_id(vector)['q1']['hit5']
    assert not by_id(vector)['q1']['hit1']
    assert by_id(vector)['q1']['rr'] == 0.5
    assert by_id(hybrid)['q1']['rank'] == 1  # found by both lists: fused first
    assert by_id(reranked)['q1']['rank'] == 1 and by_id(reranked)['q1']['recall5'] == 1.0

    # the unanswerable question is retrieved (for generation) but never counted
    for rows in (keyword, vector, hybrid, reranked):
        q3 = by_id(rows)['q3']
        assert q3['counted'] is False and q3['rank'] is None and not q3['hit5']
        assert len(rows) == 3


def test_rows_are_self_contained_and_carry_diagnostics(
    corpus, fake_embedder, fake_reranker, test_settings
):
    rows, chunks = run(corpus, fake_embedder, fake_reranker, test_settings, 'C')
    row = by_id(rows)['q1']
    for key in ('type', 'language', 'split', 'counted', 'rank', 'hit1', 'hit5', 'hit10',
                'recall5', 'rr', 'doc_hit5', 'timings', 'top'):  # fmt: skip
        assert key in row
    assert row['doc_hit5'] is True
    assert row['timings']['total_ms'] > 0 and row['timings']['embed_ms'] > 0
    assert row['top'][0]['document'] == 'a.txt'
    assert set(chunks) == {'q1', 'q2', 'q3'}
    assert {'document', 'content', 'page'} <= set(chunks['q1'][0])


def test_retrieval_is_deterministic_apart_from_timings(
    corpus, fake_embedder, fake_reranker, test_settings
):
    def strip(rows):
        return [{k: v for k, v in row.items() if k != 'timings'} for row in rows]

    first, _ = run(corpus, fake_embedder, fake_reranker, test_settings, 'D')
    second, _ = run(corpus, fake_embedder, fake_reranker, test_settings, 'D')
    assert strip(first) == strip(second)


def test_aggregate_and_report_from_real_rows(corpus, fake_embedder, fake_reranker, test_settings):
    aggregates = {}
    for name in 'ABCD':
        rows, _ = run(corpus, fake_embedder, fake_reranker, test_settings, name)
        aggregates[name] = aggregate_retrieval(rows)
    assert aggregates['A']['counted'] == 2  # the unanswerable question is excluded
    assert aggregates['C']['hit5'] == {'hits': 2, 'n': 2}
    text = render_report({'info': {}}, aggregates, {})
    assert '| C: hybrid (RRF) | 2/2 (100%) |' in text


def test_url_with_dbname():
    url = url_with_dbname('postgresql://ttd:ttd@localhost:5433/ttd', 'ttd_ctx')
    assert 'dbname=ttd_ctx' in url and 'port=5433' in url and 'user=ttd' in url


def test_main_end_to_end_with_merge(
    corpus, test_db_url, fake_embedder, fake_reranker, monkeypatch, tmp_path
):
    questions_path = tmp_path / 'questions.jsonl'
    questions_path.write_text(
        '\n'.join(
            json.dumps(
                {
                    'id': q.id, 'type': q.type, 'language': q.language, 'split': q.split,
                    'question': q.question, 'reference_answer': q.reference_answer,
                    'gold': [{'document': g.document, 'passage': g.passage} for g in q.gold],
                }
            )
            for q in QUESTIONS
        )
        + '\n'
    )  # fmt: skip
    monkeypatch.setattr(
        run_eval, 'get_connection', lambda url=None: real_get_connection(url=test_db_url)
    )
    monkeypatch.setattr('talk_to_documents.retrieval.embed.get_embedder', lambda: fake_embedder)
    monkeypatch.setattr('talk_to_documents.retrieval.rerank.get_reranker', lambda: fake_reranker)
    monkeypatch.setattr(
        'talk_to_documents.llm.get_llm', lambda settings=None: FakeLLM(reply='zebra [1].')
    )
    judge_reply = json.dumps({'claims': [{'claim': 'c', 'supported': True}], 'verdict': 'correct'})
    monkeypatch.setattr(
        'talk_to_documents.llm.get_judge_llm', lambda settings=None: FakeLLM(reply=judge_reply)
    )

    out = tmp_path / 'results'
    common = [
        '--questions', str(questions_path), '--output', str(out),
        '--generator-cache', str(tmp_path / 'g.jsonl'), '--judge-cache', str(tmp_path / 'j.jsonl'),
    ]  # fmt: skip
    code = run_eval.main(
        ['--configs', 'A,B,C,D,E', '--generate', '--generate-configs', 'C,E', *common]
    )
    assert code == 0

    data = json.loads((out / 'results.json').read_text())
    assert set(data['retrieval']) == {'A', 'B', 'C', 'D', 'E'}
    assert set(data['generation']) == {'C', 'E'}
    assert data['aggregates']['generation']['C']['correctness']['mean'] == 1.0
    assert data['aggregates']['generation']['C']['abstention'] == {
        'hits': 0,
        'n': 1,
    }  # q3 got an answer
    assert 'Main results' in (out / 'eval.md').read_text()
    assert data['runs']['E']['documents'] == 3

    # merging one configuration keeps everything else
    code = run_eval.main(['--configs', 'A', '--no-generate', '--merge', *common])
    assert code == 0
    merged = json.loads((out / 'results.json').read_text())
    assert set(merged['retrieval']) == {'A', 'B', 'C', 'D', 'E'}
    assert set(merged['generation']) == {'C', 'E'}

    # without --merge the old results are replaced
    code = run_eval.main(['--configs', 'A', '--no-generate', *common])
    assert set(json.loads((out / 'results.json').read_text())['retrieval']) == {'A'}


def test_main_rejects_unknown_configuration_and_empty_selection(tmp_path, capsys):
    path = tmp_path / 'q.jsonl'
    row = {
        'id': 'q', 'type': 'single', 'language': 'en', 'split': 'dev', 'question': 'x',
        'gold': [{'document': 'a.txt', 'passage': 'p'}], 'reference_answer': 'r',
    }  # fmt: skip
    path.write_text(json.dumps(row) + '\n')
    assert run_eval.main(['--questions', str(path), '--configs', 'Z']) == 1
    assert run_eval.main(['--questions', str(path), '--split', 'test']) == 1  # no test questions
