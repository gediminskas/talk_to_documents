import pytest

from talk_to_documents.llm_json import LLMJSONError, complete_json, extract_json_object
from tests.conftest import FakeLLM


def test_plain_json():
    assert extract_json_object('{"a": 1}') == {'a': 1}


def test_json_inside_a_fence():
    assert extract_json_object('Here you go:\n```json\n{"a": 1, "b": [1, 2]}\n```\nDone.') == {
        'a': 1,
        'b': [1, 2],
    }
    assert extract_json_object('```\n{"a": 1}\n```') == {'a': 1}


def test_text_before_and_after_the_object():
    assert extract_json_object('Sure! {"verdict": "correct"} Hope that helps.') == {
        'verdict': 'correct'
    }


def test_nested_objects_and_braces_inside_strings():
    text = '{"claims": [{"claim": "uses {curly} braces", "supported": true}], "note": "a } b"}'
    parsed = extract_json_object(text)
    assert parsed['claims'][0]['claim'] == 'uses {curly} braces'
    assert parsed['note'] == 'a } b'


def test_escaped_quotes_inside_strings():
    assert extract_json_object(r'{"q": "he said \"hi\" {x}"}') == {'q': 'he said "hi" {x}'}


def test_skips_invalid_candidate_and_finds_the_next_one():
    assert extract_json_object('first {not json} then {"ok": true}') == {'ok': True}


def test_thinking_blocks_are_removed_first():
    assert extract_json_object('<think>maybe {"x": 0}</think>{"x": 1}') == {'x': 1}


def test_no_json_returns_none():
    assert extract_json_object('no braces at all') is None
    assert extract_json_object('{"unclosed": 1') is None
    assert extract_json_object('[1, 2, 3]') is None  # arrays are not objects
    assert extract_json_object('') is None


def test_complete_json_returns_data_and_usage():
    llm = FakeLLM(reply='{"verdict": "correct", "reason": "ok"}')
    reply = complete_json(llm, 'sys', 'question?', ['verdict'])
    assert reply.data['verdict'] == 'correct'
    assert reply.attempts == 1 and len(llm.calls) == 1


def test_missing_key_triggers_a_retry_with_a_repair_note():
    replies = iter(['{"other": 1}', '{"verdict": "correct"}'])
    llm = FakeLLM(reply=lambda system, user: next(replies))
    reply = complete_json(llm, None, 'question?', ['verdict'])
    assert reply.data == {'verdict': 'correct'} and reply.attempts == 2
    assert 'question?' in llm.calls[0]['user'] and 'previous reply' not in llm.calls[0]['user']
    assert 'previous reply' in llm.calls[1]['user']  # the retry is not an identical prompt
    assert llm.calls[1]['user'].startswith('question?')


def test_gives_up_after_all_attempts():
    llm = FakeLLM(reply='I refuse to answer in JSON.')
    with pytest.raises(LLMJSONError) as excinfo:
        complete_json(llm, None, 'q', ['a'], retries=2)
    assert len(llm.calls) == 3
    assert 'I refuse' in str(excinfo.value)


def test_llm_json_error_is_a_runtime_error():
    assert issubclass(LLMJSONError, RuntimeError)
