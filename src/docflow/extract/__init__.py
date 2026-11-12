from __future__ import annotations

from docflow.config import Settings
from docflow.extract.base import (
    Completion,
    ExtractionError,
    ExtractionOutcome,
    Extractor,
    TokenUsage,
)

__all__ = [
    "Completion",
    "ExtractionError",
    "ExtractionOutcome",
    "Extractor",
    "TokenUsage",
    "build_extractor",
]


def build_extractor(settings: Settings) -> Extractor:
    if settings.llm_provider == "claude":
        from docflow.extract.claude import ClaudeExtractor

        return ClaudeExtractor(settings)
    if settings.llm_provider == "gemini":
        from docflow.extract.gemini import GeminiExtractor

        return GeminiExtractor(settings)
    from docflow.extract.fake import FakeExtractor

    return FakeExtractor(settings)
