import pytest

from eval.aggregate import aggregate_generation, aggregate_retrieval
from eval.report import fmt_cost, fmt_rate, render_report


def row(qid, qtype='single', language='en', split='dev', counted=True, rank=None, total=10.0, **kw):
    hit = counted and rank is not None
    base = {
        'id': qid, 'type': qtype, 'language': language, 'split': split, 'counted': counted,
        'rank': rank, 'hit1': hit and rank <= 1, 'hit5': hit and rank <= 5,
        'hit10': hit and rank <= 10, 'recall5': 1.0 if hit and rank <= 5 else 0.0,
        'rr': 1 / rank if hit else 0.0, 'doc_hit5': hit,
        'timings': {'total_ms': total},
    }  # fmt: skip
    base.update(kw)
    return base


def test_retrieval_aggregate_counts_only_answerable_questions():
    rows = [
        row('a', rank=1, total=10),
        row('b', rank=4, total=20),
        row('c', rank=None, total=30),
        row('d', qtype='unanswerable', counted=False, total=40),
    ]
    agg = aggregate_retrieval(rows)
    assert agg['questions'] == 4 and agg['counted'] == 3
    assert agg['hit1'] == {'hits': 1, 'n': 3}
    assert agg['hit5'] == {'hits': 2, 'n': 3}
    assert agg['recall5'] == pytest.approx(2 / 3)
    assert agg['mrr10'] == pytest.approx((1 + 0.25 + 0) / 3)
    assert agg['p50_ms'] == 25.0  # latency covers all four rows
    assert agg['hit5_by_type'] == {'single': {'hits': 2, 'n': 3}}


def test_breakdowns_by_language_and_split():
    rows = [
        row('a', language='en', split='dev', rank=1),
        row('b', language='lt', split='test', rank=None),
        row('c', language='lt', split='test', rank=2),
    ]
    agg = aggregate_retrieval(rows)
    assert agg['hit5_by_language'] == {'en': {'hits': 1, 'n': 1}, 'lt': {'hits': 1, 'n': 2}}
    assert agg['hit5_by_split'] == {'dev': {'hits': 1, 'n': 1}, 'test': {'hits': 1, 'n': 2}}


def test_empty_retrieval_aggregate_does_not_crash():
    agg = aggregate_retrieval([])
    assert agg['hit5']['n'] == 0 and agg['recall5'] is None and agg['p95_ms'] == 0.0


def gen_row(qtype='single', not_found=False, faith=None, corr=None, leaked=None, tin=100, tout=50):
    return {
        'id': 'x', 'type': qtype, 'language': 'en', 'split': 'dev', 'answer': '',
        'not_found': not_found, 'faithfulness': faith, 'correctness': corr,
        'canary_leaked': leaked, 'gen_input_tokens': tin, 'gen_output_tokens': tout,
        'gen_seconds': 2.0, 'judge_error': None,
    }  # fmt: skip


def test_generation_aggregate():
    rows = [
        gen_row(faith=1.0, corr=1.0),
        gen_row(faith=0.5, corr=0.5),
        gen_row(not_found=True, corr=0.0),
        gen_row(qtype='unanswerable', not_found=True),
        gen_row(qtype='unanswerable', not_found=False),
        gen_row(qtype='injection', faith=1.0, corr=1.0, leaked=False),
        gen_row(qtype='injection', faith=1.0, corr=0.0, leaked=True),
    ]
    agg = aggregate_generation(rows, price_in_per_mtok=1_000_000, price_out_per_mtok=2_000_000)
    assert agg['faithfulness'] == {'mean': pytest.approx(0.875), 'n': 4}
    assert agg['correctness']['n'] == 5 and agg['correctness']['mean'] == pytest.approx(0.5)
    assert agg['abstention'] == {'hits': 1, 'n': 2}
    assert agg['over_abstention'] == {'hits': 1, 'n': 5}
    assert agg['injection_resisted'] == {'hits': 1, 'n': 2}
    assert agg['mean_gen_seconds'] == 2.0
    assert agg['cost_per_query'] == pytest.approx(100 + 2 * 50)  # tokens x price per token


def test_generation_cost_is_zero_without_prices():
    assert aggregate_generation([gen_row()])['cost_per_query'] == 0.0


def test_formatters():
    assert fmt_rate({'hits': 18, 'n': 27}) == '18/27 (67%)'
    assert fmt_rate({'hits': 0, 'n': 0}) == '-' and fmt_rate(None) == '-'
    assert fmt_cost(0.0) == 'free' and fmt_cost(0.01234) == '0.0123' and fmt_cost(None) == '-'


def test_report_contains_the_expected_sections_and_counts():
    retrieval = {
        'A': aggregate_retrieval([row('a', rank=None), row('b', rank=2)]),
        'D': aggregate_retrieval([row('a', rank=1), row('b', rank=1)]),
    }
    generation = {'D': aggregate_generation([gen_row(faith=1.0, corr=1.0)])}
    text = render_report(
        {'info': {'Date': '2026-11-14', 'Judge model': 'judge-x'}}, retrieval, generation
    )
    for fragment in (
        '# Evaluation results', '- **Judge model:** judge-x', '## Main results',
        '| A: keyword only | 1/2 (50%) |', '| D: hybrid + rerank | 2/2 (100%) |',
        '## Diagnostics', '## Hit@5 by question type', '## Hit@5 by language', '## Limitations',
    ):  # fmt: skip
        assert fragment in text, fragment
    # configuration A was not generated: its generation columns are dashes, D has numbers
    a_line = next(line for line in text.splitlines() if line.startswith('| A: keyword only'))
    d_line = next(line for line in text.splitlines() if line.startswith('| D: hybrid + rerank'))
    assert '1.00 (n=1)' in d_line and '1.00 (n=1)' not in a_line
