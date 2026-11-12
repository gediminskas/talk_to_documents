"""Application settings, read from environment variables and a local .env file."""

import re

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    database_url: str = 'postgresql://ttd:ttd@localhost:5433/ttd'

    embed_model: str = 'intfloat/multilingual-e5-large'
    fastembed_cache_path: str | None = './.cache/fastembed'

    fts_config: str = 'simple'

    # Retrieval (Day 2)
    candidates: int = 30  # rows fetched from each list before fusion
    rrf_k: int = 60
    rerank_model: str = 'BAAI/bge-reranker-v2-m3-int8'
    rerank_top_n: int = 20  # how many fused hits the reranker scores

    chunk_target_tokens: int = 350
    chunk_overlap_tokens: int = 50
    max_upload_mb: int = 10

    # Any OpenAI-compatible endpoint (commercial provider, Ollama, LM Studio, Unsloth Studio...)
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = 120.0

    # Contextual retrieval (Day 2): an LLM writes a short context for every chunk at ingest time
    contextual: bool = False
    context_model: str | None = None  # empty means: use llm_model
    context_max_doc_tokens: int = 6000
    context_max_output_tokens: int = 120
    context_max_chunks_per_doc: int = 300
    context_cache_path: str = './.cache/contexts.jsonl'

    @field_validator('llm_base_url', 'llm_api_key', 'llm_model', 'context_model', mode='before')
    @classmethod
    def _empty_string_is_none(cls, value: object) -> object:
        # An empty `CONTEXT_MODEL=` line in .env should mean "not set".
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator('fts_config')
    @classmethod
    def _check_fts_config(cls, value: str) -> str:
        # Used as a regconfig name in SQL (always passed as a parameter); keep it simple anyway.
        if not re.fullmatch(r'[a-z_]+', value):
            raise ValueError('FTS_CONFIG must contain only lowercase letters and underscores')
        return value


settings = Settings()
