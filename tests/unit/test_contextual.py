import json

from talk_to_documents.ingest.chunker import Chunk
from talk_to_documents.ingest.contextual import (
    MAX_CONTEXT_CHARS,
    ContextCache,
    cache_key,
    generate_contexts,
)
from tests.conftest import AlwaysFailingLLM, FakeLLM


def make_chunks(n: int) -> list[Chunk]:
    return [Chunk(i, None, 'Sec', f'chunk number {i} body text', 5) for i in range(n)]


def words(text: str) -> int:
    return len(text.split())


def run(llm, chunks, *, doc='short document text', cache=None, max_doc_tokens=1000, **kwargs):
    return generate_contexts(
        'My Title',
        doc,
        chunks,
        llm,
        model='m1',
        max_doc_tokens=max_doc_tokens,
        max_output_tokens=50,
        count_tokens=words,
        cache=cache,
        **kwargs,
    )


def test_one_context_per_chunk_and_whole_document_in_prompt():
    llm = FakeLLM(reply='Context text.')
    result = run(llm, make_chunks(3))
    assert result.contexts == ['Context text.'] * 3
    assert result.generated == 3 and not result.failed
    first_prompt = llm.calls[0]['user']
    assert '<document>\nshort document text\n</document>' in first_prompt
    assert 'chunk number 0 body text' in first_prompt
    assert llm.calls[0]['max_tokens'] == 50 and llm.calls[0]['model'] == 'm1'


def test_cache_prevents_repeat_calls(tmp_path):
    cache = ContextCache(tmp_path / 'c.jsonl')
    chunks = make_chunks(2)
    first = FakeLLM(reply='From the LLM.')
    run(first, chunks, cache=cache)
    assert len(first.calls) == 2

    second = FakeLLM(reply='Should not be used.')
    result = run(second, chunks, cache=ContextCache(tmp_path / 'c.jsonl'))  # reload from disk
    assert second.calls == []
    assert result.contexts == ['From the LLM.'] * 2
    assert result.cached == 2 and result.generated == 0


def test_cache_key_depends_on_model_and_text():
    base = cache_key('m1', 'doc', 'chunk')
    assert cache_key('m2', 'doc', 'chunk') != base
    assert cache_key('m1', 'other doc', 'chunk') != base
    assert cache_key('m1', 'doc', 'other chunk') != base


def test_damaged_cache_line_is_ignored(tmp_path):
    path = tmp_path / 'c.jsonl'
    path.write_text('not json\n' + json.dumps({'key': 'k', 'context': 'v'}) + '\n')
    assert ContextCache(path).get('k') == 'v'


def test_failed_calls_give_empty_context_and_processing_continues():
    llm = FakeLLM(reply='ok', fail_times=1)
    result = run(llm, make_chunks(3))
    assert result.contexts == ['', 'ok', 'ok']
    assert not result.failed


def test_five_failures_in_a_row_abandon_all_contexts():
    llm = FakeLLM(reply='ok')
    llm.fail_times = 5
    chunks = make_chunks(8)
    result = run(llm, chunks)
    assert result.failed
    assert result.contexts == [''] * 8  # never half indexed
    assert len(llm.calls) == 5  # stopped trying

    result = run(AlwaysFailingLLM(), chunks)
    assert result.failed


def test_a_success_resets_the_failure_counter():
    calls = {'n': 0}

    def reply(system, user):
        return 'ok'

    llm = FakeLLM(reply=reply)
    # fail, fail, fail, fail, succeed, fail x4 would still not reach 5 in a row
    pattern = [True] * 4 + [False] + [True] * 4 + [False]
    original = llm.complete

    def flaky(*args, **kwargs):
        should_fail = pattern[calls['n']]
        calls['n'] += 1
        llm.fail_times = 1 if should_fail else 0
        return original(*args, **kwargs)

    llm.complete = flaky
    result = run(llm, make_chunks(10))
    assert not result.failed


def test_max_chunks_cap():
    llm = FakeLLM(reply='ok')
    result = run(llm, make_chunks(5), max_chunks=2)
    assert result.contexts == ['ok', 'ok', '', '', '']
    assert len(llm.calls) == 2


def test_long_document_uses_an_excerpt_with_neighbours():
    llm = FakeLLM(reply='ok')
    chunks = make_chunks(6)
    long_doc = ' '.join(['word'] * 500)
    run(llm, chunks, doc=long_doc, max_doc_tokens=100)
    prompt = llm.calls[3]['user']  # chunk 3: neighbours are 1, 2, 4, 5
    assert '<excerpt>' in prompt and 'excerpt of the document' in prompt
    assert '<document>' not in prompt
    assert 'Document title: My Title' in prompt and 'Section: Sec' in prompt
    for i in (1, 2, 3, 4, 5):
        assert f'chunk number {i} body' in prompt
    assert 'chunk number 0 body' not in prompt
    assert long_doc not in prompt


def test_reply_is_cleaned():
    reply = '<think>hmm</think>  First line.\n\nSecond   line.  '
    assert run(FakeLLM(reply=reply), make_chunks(1)).contexts == ['First line. Second line.']

    long_reply = 'x' * 1000
    assert len(run(FakeLLM(reply=long_reply), make_chunks(1)).contexts[0]) <= MAX_CONTEXT_CHARS


def test_empty_reply_means_no_context_and_is_not_cached(tmp_path):
    cache = ContextCache(tmp_path / 'c.jsonl')
    result = run(FakeLLM(reply='<think>only thinking'), make_chunks(1), cache=cache)
    assert result.contexts == [''] and result.generated == 0
    assert not (tmp_path / 'c.jsonl').exists()
