from talk_to_documents.llm import CachedLLM, LLMResponse
from tests.conftest import FakeLLM


class CountingLLM(FakeLLM):
    def complete(self, system, user, max_tokens=512, temperature=0.0, model=None):
        response = super().complete(system, user, max_tokens, temperature, model)
        return LLMResponse(response.text, 11, 7, 1.5)


def test_second_identical_call_is_a_hit(tmp_path):
    inner = CountingLLM(reply='answer')
    cached = CachedLLM(inner, tmp_path / 'c.jsonl', model_name='m')
    first = cached.complete('sys', 'user', max_tokens=100)
    second = cached.complete('sys', 'user', max_tokens=100)
    assert first.text == second.text == 'answer'
    assert (second.input_tokens, second.output_tokens, second.seconds) == (11, 7, 1.5)
    assert len(inner.calls) == 1
    assert (cached.calls, cached.hits) == (1, 1)


def test_any_change_in_the_inputs_is_a_miss(tmp_path):
    inner = CountingLLM()
    cached = CachedLLM(inner, tmp_path / 'c.jsonl', model_name='m')
    cached.complete('sys', 'user', max_tokens=100)
    cached.complete('sys2', 'user', max_tokens=100)  # system prompt
    cached.complete('sys', 'user2', max_tokens=100)  # user prompt
    cached.complete('sys', 'user', max_tokens=200)  # max tokens
    cached.complete('sys', 'user', max_tokens=100, temperature=0.5)  # temperature
    cached.complete('sys', 'user', max_tokens=100, model='other')  # model
    assert cached.calls == 6 and cached.hits == 0


def test_model_name_is_part_of_the_key(tmp_path):
    path = tmp_path / 'c.jsonl'
    CachedLLM(CountingLLM(), path, model_name='model-a').complete('s', 'u')
    other = CachedLLM(CountingLLM(), path, model_name='model-b')
    other.complete('s', 'u')
    assert other.calls == 1 and other.hits == 0


def test_cache_survives_a_reload_from_disk(tmp_path):
    path = tmp_path / 'sub' / 'c.jsonl'
    CachedLLM(CountingLLM(reply='stored'), path, model_name='m').complete('s', 'u')

    fresh_inner = CountingLLM(reply='should not be used')
    fresh = CachedLLM(fresh_inner, path, model_name='m')
    assert fresh.complete('s', 'u').text == 'stored'
    assert fresh_inner.calls == [] and fresh.hits == 1


def test_works_without_a_file_and_ignores_damaged_lines(tmp_path):
    in_memory = CachedLLM(CountingLLM(), None, model_name='m')
    in_memory.complete('s', 'u')
    in_memory.complete('s', 'u')
    assert (in_memory.calls, in_memory.hits) == (1, 1)

    path = tmp_path / 'c.jsonl'
    path.write_text('garbage line\n')
    assert CachedLLM(CountingLLM(), path, model_name='m').complete('s', 'u').text
