"""Application configuration settings."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "遥感影像分类系统"
    app_version: str = "1.0.0"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    temp_dir: Path = Path(__file__).parent.parent.parent / "temp_classification"  # Project temp path

    # CORS
    cors_origins: list = ["*"]

    # Task settings
    max_concurrent_tasks: int = 3
    task_timeout: int = 3600  # 1 hour

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
