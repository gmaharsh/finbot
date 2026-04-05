from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "finbot_chunks"

    openai_api_key: str = ""
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    jwt_secret: str = "change-me-in-production-use-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    data_dir: str = "../data"
    upload_dir: str = "../data/uploads"
    database_url: str = "sqlite:///./finbot_store.db"

    session_query_limit: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
