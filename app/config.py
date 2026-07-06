"""Application settings, loaded from environment variables (see .env.example)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain.constants import MAX_FOLLOWUPS_DEFAULT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-5"
    llm_timeout: float = 30.0
    max_followups: int = MAX_FOLLOWUPS_DEFAULT
    log_level: str = "INFO"


settings = Settings()  # type: ignore[call-arg]  # populated from env at import time
