"""Thin LLM client for any OpenAI-compatible endpoint (commercial provider or local server)."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from talk_to_documents.config import Settings
from talk_to_documents.config import settings as default_settings


class LLMError(RuntimeError):
    """Raised when the LLM call fails or the LLM is not configured.

    A RuntimeError so the CLI shows it as a one-line message."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    seconds: float


class LLM(Protocol):
    def complete(
        self,
        system: str | None,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> LLMResponse: ...


_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by some reasoning models.

    If a <think> block is never closed (the reply was cut off while thinking), everything
    from it onwards is dropped.
    """
    text = _THINK_BLOCK.sub('', text)
    if '<think>' in text:
        text = text.split('<think>', 1)[0]
    return text.strip()


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        from openai import OpenAI  # lazy import

        self.model = model
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    def complete(
        self,
        system: str | None,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> LLMResponse:
        import openai

        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': user})

        started = perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except openai.OpenAIError as exc:
            raise LLMError(f'LLM request failed: {exc}') from exc

        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or '',
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            seconds=perf_counter() - started,
        )


def get_llm(settings: Settings = default_settings) -> OpenAICompatibleLLM:
    if not (settings.llm_base_url and settings.llm_api_key and settings.llm_model):
        raise LLMError('LLM is not configured: set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env')
    return OpenAICompatibleLLM(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        timeout=settings.llm_timeout_seconds,
    )


def get_judge_llm(settings: Settings = default_settings) -> OpenAICompatibleLLM:
    """The LLM used to judge answers. Each JUDGE_LLM_* value falls back to its LLM_* twin."""
    base_url = settings.judge_llm_base_url or settings.llm_base_url
    api_key = settings.judge_llm_api_key or settings.llm_api_key
    model = settings.judge_llm_model or settings.llm_model
    if not (base_url and api_key and model):
        raise LLMError('Judge LLM is not configured: set JUDGE_LLM_* (or LLM_*) in .env')
    return OpenAICompatibleLLM(base_url, api_key, model, timeout=settings.llm_timeout_seconds)


class CachedLLM:
    """Wraps any LLM and remembers replies in a JSON-lines file.

    The key covers the model, system prompt, user prompt, max tokens and temperature, so any
    change is a miss. A hit returns the stored text and token counts without calling the
    server, which makes evaluation reruns free and keeps results from drifting.
    """

    def __init__(self, inner: LLM, path: str | Path | None = None, model_name: str = '') -> None:
        self.inner = inner
        self.model_name = model_name or getattr(inner, 'model', '') or ''
        self.path = Path(path) if path else None
        self.calls = 0  # real calls to the wrapped LLM
        self.hits = 0  # replies served from the cache
        self._data: dict[str, LLMResponse] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                try:
                    row = json.loads(line)
                    self._data[row['key']] = LLMResponse(
                        row['text'], row['input_tokens'], row['output_tokens'], row['seconds']
                    )
                except (json.JSONDecodeError, KeyError):
                    continue  # ignore a damaged line

    def _key(
        self, system: str | None, user: str, max_tokens: int, temperature: float, model: str | None
    ) -> str:
        payload = [model or self.model_name, system, user, max_tokens, temperature]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode('utf-8')).hexdigest()

    def complete(
        self,
        system: str | None,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> LLMResponse:
        key = self._key(system, user, max_tokens, temperature, model)
        if key in self._data:
            self.hits += 1
            return self._data[key]

        response = self.inner.complete(system, user, max_tokens, temperature, model)
        self.calls += 1
        self._data[key] = response
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                'key': key,
                'text': response.text,
                'input_tokens': response.input_tokens,
                'output_tokens': response.output_tokens,
                'seconds': response.seconds,
            }
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        return response
