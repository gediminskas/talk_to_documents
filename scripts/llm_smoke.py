"""Check that the configured LLM endpoint works and how well the model produces JSON.

Run: uv run python scripts/llm_smoke.py
Uses LLM_BASE_URL, LLM_API_KEY and LLM_MODEL from your .env (any OpenAI-compatible server).
"""

import json
import sys
import time

from openai import OpenAI

from talk_to_documents.config import settings


def main() -> int:
    if not (settings.llm_base_url and settings.llm_api_key and settings.llm_model):
        print('Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env first.')
        return 1

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    print(f'endpoint: {settings.llm_base_url}\nmodel:    {settings.llm_model}\n')

    started = time.perf_counter()
    reply = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{'role': 'user', 'content': 'Reply with the single word OK.'}],
        max_tokens=20,
    )
    elapsed = time.perf_counter() - started
    print(f'test 1 (plain reply): {reply.choices[0].message.content!r} in {elapsed:.1f}s')

    prompt = (
        'Return ONLY a JSON object with exactly two fields: '
        '"answer" (a short string) and "confident" (true or false). '
        'Question: what is the capital of Lithuania? No markdown, no commentary.'
    )
    successes = 0
    runs = 5
    for run in range(1, runs + 1):
        started = time.perf_counter()
        content = (
            client.chat.completions.create(
                model=settings.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=200,
            )
            .choices[0]
            .message.content
            or ''
        )
        elapsed = time.perf_counter() - started
        try:
            parsed = json.loads(content)  # deliberately no cleanup: measures raw reliability
            ok = isinstance(parsed, dict) and {'answer', 'confident'} <= set(parsed)
        except json.JSONDecodeError:
            ok = False
        successes += ok
        verdict = 'ok' if ok else 'NOT clean JSON'
        print(f'test 2 run {run}: {verdict} ({elapsed:.1f}s) {content[:80]!r}')

    print(f'\nclean JSON without any cleanup: {successes}/{runs}')
    print('Write this number down: it tells you how defensive the Day 5 verify step must be.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
