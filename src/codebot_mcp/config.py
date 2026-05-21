"""Configuration for standalone codebot."""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Embeddings - Voyage API
    VOYAGE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "voyage-code-3"
    EMBEDDING_DIMENSION: int = 1024
    BATCH_SIZE: int = 128

    # Parsing
    REPO_CLONE_DIR: str = "/tmp/repos"
    MAX_REPO_SIZE_MB: int = 500

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


settings = Settings()
