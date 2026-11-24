import json

from eval.draft_questions import (
    check_candidate,
    draft_candidates,
    guess_language,
    render_review_markdown,
    select_chunks,
)
from tests.conftest import FakeLLM

CHUNK = (
    'The database listens on port 5433 on the host machine so that it does not collide with '
    'a local installation. ' + 'Filler sentence for length. ' * 12
)
QUOTE = 'The database listens on port 5433 on the host machine so that it does not collide'


def test_guess_language():
    assert guess_language('Ąžuolas auga prie upės, o žuvys plaukia šiandien.') == 'lt'
    assert guess_language('The oak tree grows by the river.') == 'en'
    assert guess_language('1234 !!!') == 'en'


def rows(spec: dict[str, int]) -> list[dict]:
    return [
        {'chunk_id': f'{doc}-{i}', 'document': doc, 'title': doc, 'content': 'word ' * 60}
        for doc, count in spec.items()
        for i in range(count)
    ]


def test_select_chunks_spreads_over_documents_and_is_deterministic():
    data = rows({'a.md': 10, 'b.md': 10, 'c.md': 10})
    chosen = select_chunks(data, count=7, per_document=3, seed=1)
    assert len(chosen) == 7
    per_doc = [sum(1 for c in chosen if c['document'] == d) for d in ('a.md', 'b.md', 'c.md')]
    assert max(per_doc) <= 3 and min(per_doc) >= 2  # round robin, never one document only
    assert chosen == select_chunks(data, count=7, per_document=3, seed=1)
    assert chosen != select_chunks(data, count=7, per_document=3, seed=2)


def test_select_chunks_skips_short_chunks_and_stops_when_exhausted():
    data = rows({'a.md': 2}) + [
        {'chunk_id': 's', 'document': 'a.md', 'title': 'a', 'content': 'too short'}
    ]
    assert len(select_chunks(data, count=50, per_document=5, seed=1)) == 2


def test_check_candidate_accepts_a_good_one():
    assert (
        check_candidate(
            CHUNK, 'Which host port avoids clashing with a local server?', QUOTE, 'Port 5433.'
        )
        is None
    )


def test_check_candidate_rejections():
    good_q = 'Which host port avoids clashing with a local server?'
    assert 'empty' in check_candidate(CHUNK, '', QUOTE, 'a')
    assert 'verbatim' in check_candidate(
        CHUNK, good_q, 'This sentence is not in the chunk at all really', 'a'
    )
    assert 'words' in check_candidate(CHUNK, good_q, 'The database listens', 'a')
    assert 'too short' in check_candidate(CHUNK, 'Why?', QUOTE, 'a')
    assert 'refers to the document' in check_candidate(
        CHUNK, 'What does this document say about ports?', QUOTE, 'a'
    )
    assert 'too easy' in check_candidate(
        CHUNK, 'The database listens on port 5433 on the host machine', QUOTE, 'a'
    )


def reply_for(system, user):
    return json.dumps(
        {
            'question': 'Which host port avoids clashing with a local server?',
            'quote': QUOTE,
            'answer': 'Port 5433.',
        }
    )


def test_draft_candidates_marks_every_fifth_as_cross_language():
    chunks = [
        {'chunk_id': str(i), 'document': f'd{i}.md', 'title': 't', 'content': CHUNK}
        for i in range(1, 6)
    ]
    llm = FakeLLM(reply=reply_for)
    candidates, discarded = draft_candidates(llm, chunks, cross_every=5)
    assert len(candidates) == 5 and discarded == []
    assert [c['type'] for c in candidates] == ['single'] * 4 + ['cross_language']
    assert [c['language'] for c in candidates] == ['en'] * 4 + ['lt']  # source is English
    assert 'Lithuanian' in llm.calls[4]['user'] and 'English' in llm.calls[0]['user']
    assert candidates[0]['gold'] == [{'document': 'd1.md', 'passage': QUOTE}]
    assert candidates[0]['id'] == 'draft-001' and candidates[0]['split'] == ''


def test_failed_and_invalid_candidates_are_discarded_with_a_reason():
    chunks = [{'chunk_id': '1', 'document': 'a.md', 'title': 't', 'content': CHUNK}]
    bad_quote = FakeLLM(
        reply=json.dumps(
            {
                'question': 'Which host port is used here?',
                'quote': 'not in chunk but long enough to pass',
                'answer': 'x',
            }
        )
    )
    candidates, discarded = draft_candidates(bad_quote, chunks)
    assert candidates == [] and 'verbatim' in discarded[0]

    garbage = FakeLLM(reply='not json')
    candidates, discarded = draft_candidates(garbage, chunks)
    assert candidates == [] and 'LLM failed' in discarded[0]


def test_review_markdown_lists_every_candidate():
    chunks = [{'chunk_id': '1', 'document': 'a.md', 'title': 't', 'content': CHUNK}]
    candidates, _ = draft_candidates(FakeLLM(reply=reply_for), chunks)
    text = render_review_markdown(candidates)
    assert '## draft-001' in text and 'Port 5433.' in text and 'a.md' in text
