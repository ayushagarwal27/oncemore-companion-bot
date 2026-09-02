"""Single place for every tunable. Model names move fast; keep them here."""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Some libraries (langchain's init_chat_model) read OPENAI_API_KEY straight
# from the environment instead of taking it as an argument, so we need it
# in os.environ too, not just on the Settings object below.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""

    postgres_dsn: str = "postgresql://companion:companion@localhost:5432/companion"
    redis_url: str = "redis://localhost:6379/0"

    # --- models -----------------------------------------------------------
    # Verify against the live model list before a real run; these are the
    # tiers, not sacred strings.
    chat_model: str = "gpt-5.6-sol"
    extraction_model: str = "gpt-5.6-luna"
    adjudicator_model: str = "gpt-5.6-terra"
    judge_model: str = "gpt-5.6-sol"
    oracle_model: str = "gpt-5.5-pro"

    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536

    # --- retrieval --------------------------------------------------------
    retrieval_vector_k: int = 20
    retrieval_lexical_k: int = 20
    retrieval_final_k: int = 8
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    memory_token_budget: int = 900

    # final_score = w_rrf * rrf_norm + w_importance * importance + w_strength * strength
    w_rrf: float = 0.6
    w_importance: float = 0.25
    w_strength: float = 0.15

    # --- write path -------------------------------------------------------
    adjudicator_neighbours: int = 6
    hot_path_importance_threshold: float = 0.75
    extraction_aggressiveness: str = "balanced"  # conservative | balanced | eager

    # --- caching ----------------------------------------------------------
    embedding_cache_ttl_s: int = 60 * 60 * 24 * 30
    memory_block_cache_ttl_s: int = 120
    checkpoint_ttl_minutes: int = 60 * 24 * 7

    persona_id: str = "mira"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
