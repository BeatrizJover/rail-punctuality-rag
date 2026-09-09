"""
Application configuration.
Settings are loaded from environment variables and ``.env`` files using ``pydantic-settings``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration"""

    model_config = SettingsConfigDict(
        env_prefix="RAIL_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "rail_rag"
    db_user: str = "rail_rag"
    db_password: SecretStr = SecretStr("")

    #: Non-secret model parameters live in a committed YAML file; only the key is an env var.
    llm_config_path: Path = Path("config/model_config.yaml")
    llm_api_key: SecretStr = SecretStr("")

    @property
    def database_url(self) -> str:
        """SQLAlchemy connection URL."""
        return (
            f"postgresql+psycopg://{self.db_user}:"
            f"{self.db_password.get_secret_value()}@"
            f"{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide, cached ``Settings`` instance."""
    return Settings()
