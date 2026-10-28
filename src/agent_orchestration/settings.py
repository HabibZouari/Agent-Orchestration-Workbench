from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_api_key: str = ""
    llm_base_url: str = "http://localhost:20128/v1"
    llm_model: str = "gemma-2-2b"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql://agent:agent@localhost:5432/agent_orchestration"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    approval_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
