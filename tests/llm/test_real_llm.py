"""Calls your real LLM server (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in .env): make test-llm."""

import pytest

from talk_to_documents.config import settings
from talk_to_documents.llm import get_llm

pytestmark = pytest.mark.skipif(
    not (settings.llm_base_url and settings.llm_api_key and settings.llm_model),
    reason='LLM is not configured in .env',
)


def test_simple_completion():
    response = get_llm().complete(None, 'Reply with the single word OK.', max_tokens=50)
    assert response.text.strip()
    assert response.seconds > 0


def test_system_prompt_and_temperature_zero_are_accepted():
    response = get_llm().complete('You answer in one word.', 'Capital of Lithuania?', max_tokens=50)
    assert 'vilnius' in response.text.lower()
