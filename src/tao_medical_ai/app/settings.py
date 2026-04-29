from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    env: str = "dev"
    api_prefix: str = "/v1"
    api_key: str = "local-dev-key"
    enable_auth: bool = True
    enable_rate_limit: bool = True
    rate_limit_per_minute: int = 60
    trace_dir: str = "runtime"
    log_level: str = "INFO"
    redact_logs: bool = True
    llm_provider: str = "offline"
    llm_model: str = "gemini-2.5-flash"

    model_config = SettingsConfigDict(env_prefix="TAO_", env_file=".env", extra="ignore")
