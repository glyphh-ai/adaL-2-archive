"""
Ada configuration.

All configuration is read from environment variables / .env with sensible
defaults. Storage is SQLite by default; setting DATABASE_URL to a Postgres URL
switches the runtime to the Postgres backend (see resolved_storage_backend).
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Ada configuration."""

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8002, description="Server port")

    # Database — SQLite by default; a postgres:// URL enables Postgres.
    database_url: Optional[str] = Field(
        default=None,
        description="DB URL. Unset → SQLite at ~/.ada/ada.db. "
                    "postgresql://… → Postgres backend.",
    )

    # Storage engine: "sql" (fact_slots-backed; the store IS the DB, so
    # writes/forgets/amends are durable and consistent — the default) or
    # "memory" (in-process indexes; ephemeral, for benchmarks/throwaway).
    storage_mode: str = Field(
        default="sql", description="sql | memory",
        validation_alias=AliasChoices("ADA_STORAGE", "ADA_STORAGE_MODE"),
    )

    # Write-path enricher: how prose becomes universal-schema slots.
    #   "" / "local"  — the local model at ADA_LOCAL_URL when it answers
    #   "heuristic"   — regex, no model at all
    enricher: str = Field(
        default="", description="(auto) | local | heuristic",
        validation_alias=AliasChoices("ADA_ENRICHER"),
    )

    # Rotor — the memory proxy in front of Anthropic (see docs/ROTOR.md).
    rotor_enabled: bool = Field(
        default=True, description="Run the rotor on /v1/messages",
        validation_alias=AliasChoices("ADA_ROTOR"),
    )
    anthropic_upstream: str = Field(
        default="https://api.anthropic.com",
        description="Upstream for proxied /v1/* requests",
        validation_alias=AliasChoices("ADA_UPSTREAM"),
    )
    # Provider keyring — Ada is a router: the caller authenticates to Ada
    # with an Ada key, and Ada attaches the USER's own provider key
    # (stored here) when forwarding to that provider. Empty → Ada falls
    # back to forwarding whatever credential the caller sent.
    anthropic_key: str = Field(
        default="", description="User's Anthropic key, used upstream",
        validation_alias=AliasChoices("ADA_ANTHROPIC_KEY", "ANTHROPIC_API_KEY"),
    )
    openai_key: str = Field(
        default="", description="User's OpenAI key (for OpenAI routes)",
        validation_alias=AliasChoices("ADA_OPENAI_KEY", "OPENAI_API_KEY"),
    )
    embed_url: str = Field(
        default="", description="OpenAI-compatible /embeddings endpoint "
                                "(empty → ADA_LOCAL_URL)",
        validation_alias=AliasChoices("ADA_EMBED_URL"),
    )
    embed_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model (ollama pull nomic-embed-text)",
        validation_alias=AliasChoices("ADA_EMBED_MODEL"),
    )
    inject_budget: int = Field(
        default=1500, description="Max characters of injected memory",
        validation_alias=AliasChoices("ADA_INJECT_BUDGET"),
    )

    # Dev flags
    enable_docs: bool = Field(default=False, description="Enable /docs endpoint")
    cors_allow_all: bool = Field(default=True, description="Allow all CORS origins")

    # Storage — auto-resolved from DATABASE_URL (see resolved_storage_backend).
    storage_backend: str = Field(default="sqlite")

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        from pathlib import Path
        db_dir = Path.home() / ".ada"
        db_dir.mkdir(exist_ok=True)
        return f"sqlite+aiosqlite:///{db_dir}/ada.db"

    @property
    def resolved_storage_backend(self) -> str:
        # postgres when pointed at a persistent Postgres; SQLite otherwise.
        url = self.resolved_database_url.lower()
        if "postgres" in url:
            return "postgres"
        return "sqlite"

    # JWT (for token auth)
    jwt_secret_key: Optional[str] = Field(default=None)

    # Vector dimension
    max_vector_dimension: int = Field(default=65536)

    # Edge generation
    edge_generation_strategy: str = Field(default="lazy")
    edge_ttl_hours: int = Field(default=24)

    # Beam search
    default_beam_width: int = Field(default=5)
    default_max_tree_depth: int = Field(default=3)

    # Logging
    log_level: str = Field(default="INFO")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
    )
    cors_origins_production: List[str] = Field(default=[])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def validate_settings() -> None:
    """Validate settings on startup."""
    get_settings()  # triggers pydantic validation
