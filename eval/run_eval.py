"""Evaluation harness: run the question set against retrieval configurations A-E.

Examples:
    uv run python -m eval.run_eval --no-generate        # retrieval metrics only, no LLM
    uv run python -m eval.run_eval --generate           # plus answers and judging
    uv run python -m eval.run_eval --configs E --merge  # add one configuration to old results

Retrieval is deterministic, so a rerun gives identical numbers (latency aside).
"""

import argparse
import datetime
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from rich.console import Console

from eval.aggregate import aggregate_generation, aggregate_retrieval
from eval.answers import answer_and_judge
from eval.dataset import Question, load_questions
from eval.metrics import doc_hit_at_k, first_hit_rank, hit_at_k, recall_at_k, reciprocal_rank
from eval.report import CONFIG_ORDER, render_report
from talk_to_documents.config import Settings
from talk_to_documents.config import settings as default_settings
from talk_to_documents.db import get_connection
from talk_to_documents.retrieval.hybrid import hybrid_search

TOP_K = 10


@dataclass(frozen=True)
class EvalConfig:
    name: str
    mode: str  # keyword, vector or hybrid
    rerank: bool
    database: str  # 'main' (plain ingest) or 'ctx' (contextual retrieval ingest)


CONFIGS = {
    'A': EvalConfig('A', 'keyword', False, 'main'),
    'B': EvalConfig('B', 'vector', False, 'main'),
    'C': EvalConfig('C', 'hybrid', False, 'main'),
    'D': EvalConfig('D', 'hybrid', True, 'main'),
    'E': EvalConfig('E', 'hybrid', True, 'ctx'),
}


def url_with_dbname(url: str, dbname: str) -> str:
    info = conninfo_to_dict(url)
    info['dbname'] = dbname
    return make_conninfo(**info)


def document_names(conn: psycopg.Connection) -> dict:
    return {row[0]: row[1] for row in conn.execute('SELECT id, filename FROM documents')}


def retrieve_for_config(
    conn: psycopg.Connection,
    questions: list[Question],
    config: EvalConfig,
    *,
    embedder,
    reranker,
    settings: Settings = default_settings,
    top_k: int = TOP_K,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Run every question through one configuration.

    Returns (rows, chunks): one self-contained row per question, and the retrieved chunks per
    question id (kept in memory for answer generation, not written to the results file).
    """
    names = document_names(conn)

    def search(text: str):
        return hybrid_search(
            conn,
            text,
            embedder=embedder,
            reranker=reranker,
            mode=config.mode,
            top_k=top_k,
            rerank=config.rerank,
            settings=settings,
        )

    search('warm up')  # the first call loads models and fills caches; do not time it

    rows: list[dict] = []
    chunks_by_question: dict[str, list[dict]] = {}
    for question in questions:
        result = search(question.question)
        hits = [fused.hit for fused in result.hits]
        texts = [hit.content for hit in hits]
        documents = [names.get(hit.document_id, '?') for hit in hits]
        chunks_by_question[question.id] = [
            {'document': doc, 'content': hit.content, 'page': hit.page}
            for doc, hit in zip(documents, hits, strict=True)
        ]

        counted = question.answerable
        rank = first_hit_rank(question.passages, texts) if counted else None
        timings = result.timings
        rows.append(
            {
                'id': question.id,
                'type': question.type,
                'language': question.language,
                'split': question.split,
                'counted': counted,
                'rank': rank,
                'hit1': counted and hit_at_k(rank, 1),
                'hit5': counted and hit_at_k(rank, 5),
                'hit10': counted and hit_at_k(rank, 10),
                'recall5': recall_at_k(question.passages, texts, 5) if counted else 0.0,
                'rr': reciprocal_rank(rank, 10) if counted else 0.0,
                'doc_hit5': counted and doc_hit_at_k(question.documents, documents, 5),
                'timings': {
                    'embed_ms': timings.embed_ms,
                    'vector_ms': timings.vector_ms,
                    'keyword_ms': timings.keyword_ms,
                    'fuse_ms': timings.fuse_ms,
                    'rerank_ms': timings.rerank_ms,
                    'total_ms': timings.total_ms,
                },
                'top': [
                    {'document': doc, 'page': hit.page, 'preview': hit.content[:160]}
                    for doc, hit in list(zip(documents, hits, strict=True))[:5]
                ],
            }
        )
    return rows, chunks_by_question


def git_commit() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def corpus_size(conn: psycopg.Connection) -> tuple[int, int]:
    documents = conn.execute('SELECT count(*) FROM documents').fetchone()[0]
    chunks = conn.execute('SELECT count(*) FROM chunks').fetchone()[0]
    return documents, chunks


def build_info(settings: Settings, questions: list[Question], runs: dict) -> dict:
    sizes = ', '.join(
        f'{name}: {run["documents"]} documents / {run["chunks"]} chunks'
        for name, run in sorted(runs.items())
    )
    return {
        'Date': datetime.date.today().isoformat(),
        'Git commit': git_commit(),
        'Questions': (
            f'{len(questions)} ({sum(q.answerable for q in questions)} answerable, '
            f'{sum(not q.answerable for q in questions)} unanswerable)'
        ),
        'Embedding model': settings.embed_model,
        'Reranker': settings.rerank_model,
        'Chunking': f'{settings.chunk_target_tokens} tokens target, '
        f'{settings.chunk_overlap_tokens} overlap',
        'Candidates per list / RRF k / rerank top N': (
            f'{settings.candidates} / {settings.rrf_k} / {settings.rerank_top_n}'
        ),
        'Keyword search config': settings.fts_config,
        'Generator model': settings.llm_model or 'not configured',
        'Judge model': settings.judge_llm_model or settings.llm_model or 'not configured',
        'Corpus (per run)': sizes or '-',
    }


def load_existing(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {'runs': {}, 'retrieval': {}, 'generation': {}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--questions', default='eval/questions.jsonl')
    parser.add_argument('--configs', default=','.join(CONFIG_ORDER), help='for example A,B,C,D,E')
    parser.add_argument('--split', choices=['all', 'dev', 'test'], default='all')
    parser.add_argument('--output', default='results')
    parser.add_argument('--generate', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--generate-configs', default='C,D,E')
    parser.add_argument('--merge', action='store_true', help='keep results of other configurations')
    parser.add_argument('--ctx-database-url', default=None)
    parser.add_argument('--generator-cache', default='.cache/eval-generate.jsonl')
    parser.add_argument('--judge-cache', default='.cache/eval-judge.jsonl')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    settings = default_settings

    questions = load_questions(args.questions)
    if args.split != 'all':
        questions = [q for q in questions if q.split == args.split]
    if not questions:
        console.print('[red]No questions to run.[/red]')
        return 1

    wanted = [name.strip().upper() for name in args.configs.split(',') if name.strip()]
    unknown = [name for name in wanted if name not in CONFIGS]
    if unknown:
        console.print(f'[red]Unknown configuration(s): {unknown}[/red]')
        return 1
    generate_for = {n.strip().upper() for n in args.generate_configs.split(',') if n.strip()}

    urls = {
        'main': settings.database_url,
        'ctx': args.ctx_database_url or url_with_dbname(settings.database_url, 'ttd_ctx'),
    }

    needs_vector = any(CONFIGS[n].mode in ('vector', 'hybrid') for n in wanted)
    needs_rerank = any(CONFIGS[n].rerank for n in wanted)
    embedder = reranker = None
    if needs_vector:
        from talk_to_documents.retrieval.embed import get_embedder

        embedder = get_embedder()
    if needs_rerank:
        from talk_to_documents.retrieval.rerank import get_reranker

        reranker = get_reranker()

    generator = judge = None
    if args.generate:
        from talk_to_documents.llm import CachedLLM, get_judge_llm, get_llm

        inner_generator, inner_judge = get_llm(settings), get_judge_llm(settings)
        generator = CachedLLM(inner_generator, args.generator_cache, settings.llm_model or '')
        judge = CachedLLM(
            inner_judge, args.judge_cache, settings.judge_llm_model or settings.llm_model or ''
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / 'results.json'
    state = (
        load_existing(results_path)
        if args.merge
        else {'runs': {}, 'retrieval': {}, 'generation': {}}
    )

    for name in wanted:
        config = CONFIGS[name]
        try:
            with get_connection(url=urls[config.database]) as conn:
                documents, chunks = corpus_size(conn)
                if not documents:
                    console.print(
                        f'[yellow]{name}: database "{config.database}" is empty, skipped.[/yellow]'
                    )
                    continue
                console.print(f'[bold]{name}[/bold] retrieval over {len(questions)} questions...')
                rows, chunks_by_question = retrieve_for_config(
                    conn, questions, config, embedder=embedder, reranker=reranker, settings=settings
                )
        except RuntimeError as exc:
            console.print(f'[yellow]{name}: skipped ({exc})[/yellow]')
            continue

        state['retrieval'][name] = rows
        state['runs'][name] = {
            'database': config.database,
            'documents': documents,
            'chunks': chunks,
            'date': datetime.date.today().isoformat(),
            'commit': git_commit(),
        }

        if args.generate and name in generate_for:
            console.print(f'[bold]{name}[/bold] generating and judging answers...')
            generation_rows = []
            for number, question in enumerate(questions, start=1):
                row = answer_and_judge(
                    question,
                    chunks_by_question[question.id],
                    generator,
                    judge,
                    generator_model=settings.llm_model,
                    judge_model=settings.judge_llm_model or settings.llm_model,
                )
                generation_rows.append(row)
                console.print(f'  {number}/{len(questions)}', end='\r')
            state['generation'][name] = generation_rows
            console.print(
                f'  generator: {generator.calls} calls, {generator.hits} cached; '
                f'judge: {judge.calls} calls, {judge.hits} cached'
            )

    if not state['retrieval']:
        console.print('[red]Nothing was evaluated.[/red]')
        return 1

    retrieval_aggregates = {n: aggregate_retrieval(rows) for n, rows in state['retrieval'].items()}
    generation_aggregates = {
        n: aggregate_generation(
            rows, settings.llm_price_in_per_mtok, settings.llm_price_out_per_mtok
        )
        for n, rows in state['generation'].items()
    }
    setup = {'info': build_info(settings, questions, state['runs'])}

    results_path.write_text(
        json.dumps(
            {
                'setup': setup,
                'runs': state['runs'],
                'retrieval': state['retrieval'],
                'generation': state['generation'],
                'aggregates': {
                    'retrieval': retrieval_aggregates,
                    'generation': generation_aggregates,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )
    report = render_report(setup, retrieval_aggregates, generation_aggregates)
    (output / 'eval.md').write_text(report, encoding='utf-8')
    console.print(report)
    console.print(f'\nwritten: {output / "eval.md"} and {results_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
