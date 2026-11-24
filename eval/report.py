"""Render the evaluation results as Markdown."""

CONFIG_ORDER = ['A', 'B', 'C', 'D', 'E']

CONFIG_LABELS = {
    'A': 'A: keyword only',
    'B': 'B: vector only',
    'C': 'C: hybrid (RRF)',
    'D': 'D: hybrid + rerank',
    'E': 'E: hybrid + rerank + contextual',
}

LIMITATIONS = [
    'The question set is small (a few dozen questions) and was written for one corpus; '
    'differences of one or two questions are noise, not evidence.',
    'Retrieval is scored on short verbatim gold passages; a correct chunk that covers the '
    'answer in different words is counted as a miss.',
    'Faithfulness and correctness come from an LLM judge. A small judge is noisy; the judge '
    'model is listed in the setup block.',
    'Latency was measured on one machine, one query at a time, after a warm-up query.',
    'Numbers apply to this corpus and these settings, not to retrieval in general.',
]


def fmt_rate(rate: dict | None) -> str:
    if not rate or not rate['n']:
        return '-'
    return f'{rate["hits"]}/{rate["n"]} ({100 * rate["hits"] / rate["n"]:.0f}%)'


def fmt_mean(value: float | None, digits: int = 2) -> str:
    return '-' if value is None else f'{value:.{digits}f}'


def fmt_scored(entry: dict | None) -> str:
    """A mean with the number of scored items, e.g. '0.84 (n=22)'."""
    if not entry or entry['mean'] is None:
        return '-'
    return f'{entry["mean"]:.2f} (n={entry["n"]})'


def fmt_ms(value: float | None) -> str:
    return '-' if value is None else f'{value:.0f}'


def fmt_cost(value: float | None) -> str:
    if value is None:
        return '-'
    return 'free' if value == 0 else f'{value:.4f}'


def _ordered(names) -> list[str]:
    known = [c for c in CONFIG_ORDER if c in names]
    return known + sorted(n for n in names if n not in CONFIG_ORDER)


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join(lines)


def render_report(setup: dict, retrieval: dict[str, dict], generation: dict[str, dict]) -> str:
    """`retrieval` and `generation` map a configuration name to its aggregate."""
    out = ['# Evaluation results', '']

    out += ['## Setup', '']
    for key, value in setup.get('info', {}).items():
        out.append(f'- **{key}:** {value}')
    out.append('')

    out += ['## Main results', '']
    header = [
        'Configuration', 'Hit@5', 'Recall@5', 'MRR@10', 'Faithfulness', 'Correctness',
        'Abstention', 'p50 ms', 'p95 ms', 'Cost/query',
    ]  # fmt: skip
    rows = []
    for name in _ordered(retrieval):
        r = retrieval[name]
        g = generation.get(name)
        rows.append(
            [
                CONFIG_LABELS.get(name, name),
                fmt_rate(r['hit5']),
                fmt_mean(r['recall5']),
                fmt_mean(r['mrr10']),
                fmt_scored(g['faithfulness']) if g else '-',
                fmt_scored(g['correctness']) if g else '-',
                fmt_rate(g['abstention']) if g else '-',
                fmt_ms(r['p50_ms']),
                fmt_ms(r['p95_ms']),
                fmt_cost(g['cost_per_query']) if g else '-',
            ]
        )
    out += [_table(header, rows), '']
    out += [
        'Hit@5: the answer passage is in the top 5 chunks (over answerable questions). '
        'Recall@5: share of gold passages found in the top 5. MRR@10: mean reciprocal rank of '
        'the first hit, cut off at 10. Abstention: unanswerable questions that correctly '
        'returned NOT_FOUND. Dashes mean the metric was not computed for that configuration.',
        '',
    ]

    out += ['## Diagnostics', '']
    diag_rows = []
    for name in _ordered(retrieval):
        r = retrieval[name]
        g = generation.get(name)
        diag_rows.append(
            [
                CONFIG_LABELS.get(name, name),
                fmt_rate(r['hit1']),
                fmt_rate(r['hit10']),
                fmt_rate(r['doc_hit5']),
                fmt_rate(g['over_abstention']) if g else '-',
                fmt_rate(g['injection_resisted']) if g else '-',
            ]
        )
    out += [
        _table(
            ['Configuration', 'Hit@1', 'Hit@10', 'Document hit@5', 'Over-abstention',
             'Injection resisted'],
            diag_rows,
        ),
        '',
        'Document hit@5 is lenient: the right document is in the top 5, even if the answer '
        'passage is not. A large gap between it and Hit@5 points at chunking, not ranking.',
        '',
    ]  # fmt: skip

    for dimension, title in (
        ('type', 'question type'),
        ('language', 'language'),
        ('split', 'split'),
    ):
        key = f'hit5_by_{dimension}'
        names = sorted({n for r in retrieval.values() for n in r.get(key, {})})
        if not names:
            continue
        out += [f'## Hit@5 by {title}', '']
        rows = [
            [CONFIG_LABELS.get(c, c)] + [fmt_rate(retrieval[c].get(key, {}).get(n)) for n in names]
            for c in _ordered(retrieval)
        ]
        out += [_table(['Configuration'] + names, rows), '']

    out += ['## Limitations', '']
    out += [f'- {item}' for item in LIMITATIONS]
    out.append('')
    return '\n'.join(out)
