"""Runtime configuration, read from environment variables (prefix ``DOCFLOW_``) or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- database ---------------------------------------------------------
    # Render (and most PaaS) inject a bare DATABASE_URL, so accept both names.
    database_url: str = Field(
        default="sqlite:///./docflow.db",
        validation_alias=AliasChoices("DOCFLOW_DATABASE_URL", "DATABASE_URL"),
    )

    # --- API --------------------------------------------------------------
    api_key: str | None = None
    public_base_url: str = "http://localhost:8000"
    max_upload_mb: int = 20

    # --- LLM --------------------------------------------------------------
    llm_provider: Literal["claude", "gemini", "fake"] = "claude"
    claude_model: str = "claude-opus-5-5"
    claude_effort: Effort | None = "medium"
    claude_fallbacks: bool = True
    claude_max_tokens: int = 16000
    gemini_model: str = "gemini-2.5-flash"
    llm_timeout_seconds: float = 180.0
    llm_schema_retries: int = 1
    max_document_chars: int = 400_000
    price_overrides: dict[str, tuple[float, float]] = Field(default_factory=dict)

    # --- ingest / OCR -----------------------------------------------------
    ocr_lang: str = "eng"
    ocr_dpi: int = 300
    min_text_chars_per_page: int = 40
    max_pages: int = 50

    # --- validation -------------------------------------------------------
    review_confidence_threshold: float = 0.80
    amount_tolerance: float = 0.02

    # --- files & batch ----------------------------------------------------
    data_dir: Path = Path("data")
    watch_interval_seconds: float = 5.0
    watch_settle_seconds: float = 2.0
    max_retry_attempts: int = 3

    # --- notifications ----------------------------------------------------
    notify_webhook_url: str | None = None
    notify_secret: str | None = None
    notify_on: Literal["all", "issues", "none"] = "issues"

    # --- Google Sheets ----------------------------------------------------
    sheets_enabled: bool = False
    sheets_spreadsheet_id: str | None = None
    google_service_account_file: Path | None = None

    @field_validator(
        "api_key",
        "claude_effort",
        "notify_webhook_url",
        "notify_secret",
        "sheets_spreadsheet_id",
        "google_service_account_file",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("database_url")
    @classmethod
    def _use_psycopg_driver(cls, value: str) -> str:
        for prefix in ("postgres://", "postgresql://"):
            if value.startswith(prefix):
                return "postgresql+psycopg://" + value[len(prefix) :]
        return value

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def failed_dir(self) -> Path:
        return self.data_dir / "failed"

    @property
    def storage_dir(self) -> Path:
        """Content-addressed copies of every original, so failures can be retried."""
        return self.data_dir / "files"

    @property
    def llm_model(self) -> str:
        if self.llm_provider == "claude":
            return self.claude_model
        if self.llm_provider == "gemini":
            return self.gemini_model
        return "fake"

    def ensure_dirs(self) -> None:
        for path in (self.inbox_dir, self.processed_dir, self.failed_dir, self.storage_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
