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

    chunk_target_tokens: int = 350
    chunk_overlap_tokens: int = 50
    max_upload_mb: int = 10

    # Any OpenAI-compatible endpoint (commercial provider, Ollama, LM Studio, Unsloth Studio...)
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None

    @field_validator('fts_config')
    @classmethod
    def _check_fts_config(cls, value: str) -> str:
        # Used as a regconfig name in SQL (always passed as a parameter); keep it simple anyway.
        if not re.fullmatch(r'[a-z_]+', value):
            raise ValueError('FTS_CONFIG must contain only lowercase letters and underscores')
        return value


settings = Settings()
