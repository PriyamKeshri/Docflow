"""Claude extractor: Messages API with structured outputs (``output_config.format``).

Forced ``tool_choice`` is rejected by Claude Opus 5.5 / Sonnet 5.5, so the JSON
schema goes in ``output_config.format``, which guarantees the first text block is
JSON that matches it. Server-side refusal fallback is on by default.
"""

from __future__ import annotations

from typing import Any

import anthropic

from docflow.config import Settings
from docflow.extract.base import Completion, ExtractionError, Extractor, TokenUsage

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeExtractor(Extractor):
    provider = "claude"

    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        super().__init__(settings.claude_model, settings)
        # The SDK retries 408/409/429/5xx and connection errors with backoff.
        self.client = client or anthropic.Anthropic(
            timeout=settings.llm_timeout_seconds, max_retries=3
        )
        # Haiku has no server-side fallback; sending the parameter there does nothing useful.
        self.use_fallbacks = settings.claude_fallbacks and "haiku" not in self.model

    def _output_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"format": {"type": "json_schema", "schema": self.schema}}
        if self.settings.claude_effort:
            config["effort"] = self.settings.claude_effort
        return config

    def _complete(self, system: str, user: str) -> Completion:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.claude_max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": self._output_config(),
        }
        try:
            if self.use_fallbacks:
                response = self.client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                response = self.client.messages.create(**params)
        except anthropic.RateLimitError as exc:
            raise ExtractionError(f"Claude rate limit: {exc.message}", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            raise ExtractionError(
                f"Claude API error {exc.status_code}: {exc.message}",
                retryable=exc.status_code >= 500,
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ExtractionError(f"Claude API unreachable: {exc}", retryable=True) from exc

        usage = TokenUsage()
        usage.add(
            response.model, response.usage.input_tokens, response.usage.output_tokens, self.settings
        )

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise ExtractionError(
                f"Claude declined to process the document (category: {category})",
                usage=usage,
                model=response.model,
            )
        if response.stop_reason == "max_tokens":
            raise ExtractionError(
                f"output hit max_tokens ({self.settings.claude_max_tokens}); raise "
                "DOCFLOW_CLAUDE_MAX_TOKENS",
                usage=usage,
                model=response.model,
            )

        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise ExtractionError(
                "Claude response had no text block", usage=usage, model=response.model
            )
        return Completion(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
