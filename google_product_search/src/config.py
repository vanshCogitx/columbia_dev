"""
Configuration module for Google Product Search tool.

Loads settings from environment variables and .env file using pydantic-settings.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import os
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Application configuration loaded from environment variables."""

    # --- API Keys ---
    GOOGLE_API_KEY: str = Field(..., description="Google API key.")
    GOOGLE_CX: str = Field(..., description="Google Custom Search Engine ID.")

    # --- Search Defaults ---
    SEARCH_COUNTRY: str = "in"
    SEARCH_LANGUAGE: str = "en"

    # --- Playwright ---
    PLAYWRIGHT_TIMEOUT_MS: int = 5000  # Reduced to 5 seconds for faster fallback
    PLAYWRIGHT_HEADLESS: bool = True
    PLAYWRIGHT_HEADLESS: bool = True

    # --- Limits ---
    MAX_PRODUCTS: int = 5
    MAX_QUERY_CONCURRENCY: int = 12  # boosted parallelism for queries
    MAX_PRODUCT_CONCURRENCY: int = 5
    MAX_BATCH_QUERIES: int = 10

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment variables, raising if required keys are missing."""
        missing = []
        def get(var: str) -> str:
            val = os.getenv(var)
            if val is None:
                missing.append(var)
            return val
        # Load .env file if present
        from pathlib import Path
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip('"\'')
                            os.environ.setdefault(key, val)
            except Exception as e:
                logger.warning("Failed to load .env file %s: %s", env_path, e)
        # Retrieve required vars
        api_key = get("GOOGLE_API_KEY")
        cx = get("GOOGLE_CX")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        return cls(
            GOOGLE_API_KEY=api_key,
            GOOGLE_CX=cx,
        )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a cached singleton Config instance.

    Raises:
        Exception: If required environment variables are not set.
    """
    try:
        return Config.load()
    except Exception as exc:
        logger.error("Failed to load configuration: %s", exc)
        raise

logger = logging.getLogger(__name__)



