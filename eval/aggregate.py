"""Turn per-question rows into per-configuration numbers.

Rows are self-contained dictionaries (they carry the question type, language and split), so
results can be merged across runs and re-aggregated without the question file.
"""

from collections import defaultdict
from collections.abc import Iterable
from statistics import mean

from eval.metrics import percentile


def _rate(rows: list[dict], key: str) -> dict:
    hits = sum(1 for r in rows if r[key])
    return {'hits': hits, 'n': len(rows)}


def aggregate_retrieval(rows: Iterable[dict]) -> dict:
    """Over answerable questions only (unanswerable ones have no gold passage)."""
    rows = list(rows)
    counted = [r for r in rows if r['counted']]
    result: dict = {
        'questions': len(rows),
        'counted': len(counted),
        'hit1': _rate(counted, 'hit1'),
        'hit5': _rate(counted, 'hit5'),
        'hit10': _rate(counted, 'hit10'),
        'doc_hit5': _rate(counted, 'doc_hit5'),
        'recall5': mean(r['recall5'] for r in counted) if counted else None,
        'mrr10': mean(r['rr'] for r in counted) if counted else None,
    }
    totals = [r['timings']['total_ms'] for r in rows]
    result['p50_ms'] = percentile(totals, 50)
    result['p95_ms'] = percentile(totals, 95)

    for dimension in ('type', 'language', 'split'):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in counted:
            groups[row[dimension]].append(row)
        result[f'hit5_by_{dimension}'] = {
            name: _rate(group, 'hit5') for name, group in groups.items()
        }
    return result


def aggregate_generation(
    rows: Iterable[dict], price_in_per_mtok: float = 0.0, price_out_per_mtok: float = 0.0
) -> dict:
    rows = list(rows)
    answerable = [r for r in rows if r['type'] != 'unanswerable']
    unanswerable = [r for r in rows if r['type'] == 'unanswerable']
    faithfulness = [r['faithfulness'] for r in rows if r['faithfulness'] is not None]
    correctness = [r['correctness'] for r in answerable if r['correctness'] is not None]
    injection = [r for r in rows if r['canary_leaked'] is not None]

    def cost(row: dict) -> float:
        tokens_in = row['gen_input_tokens'] or 0
        tokens_out = row['gen_output_tokens'] or 0
        return (tokens_in * price_in_per_mtok + tokens_out * price_out_per_mtok) / 1_000_000

    return {
        'questions': len(rows),
        'faithfulness': {
            'mean': mean(faithfulness) if faithfulness else None,
            'n': len(faithfulness),
        },
        'correctness': {'mean': mean(correctness) if correctness else None, 'n': len(correctness)},
        'abstention': {
            'hits': sum(1 for r in unanswerable if r['not_found']),
            'n': len(unanswerable),
        },
        'over_abstention': {
            'hits': sum(1 for r in answerable if r['not_found']),
            'n': len(answerable),
        },
        'injection_resisted': {
            'hits': sum(1 for r in injection if not r['canary_leaked']),
            'n': len(injection),
        },
        'judge_errors': sum(1 for r in rows if r['judge_error']),
        'mean_gen_seconds': mean(r['gen_seconds'] for r in rows) if rows else None,
        'cost_per_query': mean(cost(r) for r in rows) if rows else None,
    }
