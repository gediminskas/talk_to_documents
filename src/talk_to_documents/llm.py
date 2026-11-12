"""Thin LLM client for any OpenAI-compatible endpoint (commercial provider or local server)."""

import re
from dataclasses import dataclass
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
