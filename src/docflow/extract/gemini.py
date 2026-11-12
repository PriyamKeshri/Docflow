"""Gemini extractor: ``generate_content`` with a JSON response schema."""

from __future__ import annotations

from typing import Any

from docflow.config import Settings
from docflow.extract.base import Completion, ExtractionError, Extractor, TokenUsage


class GeminiExtractor(Extractor):
    provider = "gemini"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        super().__init__(settings.gemini_model, settings)
        if client is None:
            from google import genai
            from google.genai import types

            # Reads GEMINI_API_KEY (or GOOGLE_API_KEY). Timeout is in milliseconds.
            client = genai.Client(
                http_options=types.HttpOptions(timeout=int(settings.llm_timeout_seconds * 1000))
            )
        self.client = client

    def _complete(self, system: str, user: str) -> Completion:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=self.schema,
        )
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=user, config=config
            )
        except errors.APIError as exc:
            retryable = isinstance(exc, errors.ServerError) or exc.code == 429
            raise ExtractionError(
                f"Gemini API error {exc.code}: {exc.message}", retryable=retryable
            ) from exc

        meta = response.usage_metadata
        input_tokens = (meta.prompt_token_count or 0) if meta else 0
        # Thinking tokens are billed as output on the 2.5 series.
        output_tokens = (
            ((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)) if meta else 0
        )
        model = getattr(response, "model_version", None) or self.model
        usage = TokenUsage()
        usage.add(model, input_tokens, output_tokens, self.settings)

        candidate = response.candidates[0] if response.candidates else None
        finish = getattr(candidate, "finish_reason", None)
        finish_name = getattr(finish, "name", str(finish)) if finish is not None else None
        if finish_name == "MAX_TOKENS":
            raise ExtractionError(
                "Gemini output was truncated (MAX_TOKENS)", usage=usage, model=model
            )
        text = response.text
        if not text:
            raise ExtractionError(
                f"Gemini returned no text (finish_reason={finish_name}, "
                f"prompt_feedback={response.prompt_feedback})",
                usage=usage,
                model=model,
            )
        return Completion(
            text=text, model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )
