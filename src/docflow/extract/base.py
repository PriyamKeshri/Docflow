"""Provider-independent extraction loop: call the model, validate, re-ask once on failure."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import ValidationError

from docflow.config import Settings
from docflow.extract.prompts import SYSTEM_PROMPT, build_user_prompt, retry_note
from docflow.pricing import cost_usd
from docflow.schemas import SECTION_ATTR, DocType, Extraction, llm_json_schema

log = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0

    def add(self, model: str, input_tokens: int, output_tokens: int, settings: Settings) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        cost = cost_usd(model, input_tokens, output_tokens, settings.price_overrides)
        self._add_cost(cost)

    def merge(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self._add_cost(other.cost_usd)

    def _add_cost(self, cost: float | None) -> None:
        if cost is None or self.cost_usd is None:
            self.cost_usd = None  # unknown price: report unknown, not a misleading 0
        else:
            self.cost_usd = round(self.cost_usd + cost, 6)


class ExtractionError(Exception):
    """The model could not produce a valid extraction (API error, refusal, bad output)."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        usage: TokenUsage | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.usage = usage or TokenUsage()
        self.model = model


@dataclass
class Completion:
    """One raw model call."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class ExtractionOutcome:
    extraction: Extraction
    provider: str
    model: str
    usage: TokenUsage
    latency_ms: int
    attempts: int


class Extractor(ABC):
    provider = "base"

    def __init__(self, model: str, settings: Settings) -> None:
        self.model = model
        self.settings = settings
        self.schema = llm_json_schema(Extraction)

    @abstractmethod
    def _complete(self, system: str, user: str) -> Completion:
        """Make one structured-output call. Raise ExtractionError on API failure."""

    def extract(self, text: str, hint: DocType | None = None) -> ExtractionOutcome:
        if len(text) > self.settings.max_document_chars:
            # Never silently truncate: a cut-off invoice yields confidently wrong totals.
            raise ExtractionError(
                f"document text is {len(text):,} characters; the limit is "
                f"{self.settings.max_document_chars:,} (DOCFLOW_MAX_DOCUMENT_CHARS)"
            )

        user_prompt = build_user_prompt(text, hint)
        usage = TokenUsage()
        served_by = self.model
        error: str | None = None
        start = time.perf_counter()

        for attempt in range(1, self.settings.llm_schema_retries + 2):
            prompt = user_prompt if error is None else user_prompt + retry_note(error)
            try:
                completion = self._complete(SYSTEM_PROMPT, prompt)
            except ExtractionError as exc:
                exc.usage.merge(usage)  # keep the cost of earlier attempts
                raise
            served_by = completion.model
            usage.add(
                completion.model, completion.input_tokens, completion.output_tokens, self.settings
            )

            try:
                extraction = Extraction.model_validate_json(completion.text)
            except ValidationError as exc:
                error = _summarize(exc)
                log.warning("attempt %d: output failed schema validation: %s", attempt, error)
                continue

            if extraction.document_type != DocType.OTHER and extraction.section() is None:
                attr = SECTION_ATTR[extraction.document_type]
                error = f"document_type is '{extraction.document_type}' but '{attr}' is null."
                log.warning("attempt %d: %s", attempt, error)
                continue

            return ExtractionOutcome(
                extraction=extraction,
                provider=self.provider,
                model=served_by,
                usage=usage,
                latency_ms=int((time.perf_counter() - start) * 1000),
                attempts=attempt,
            )

        raise ExtractionError(
            f"model output failed validation after {self.settings.llm_schema_retries + 1} "
            f"attempts: {error}",
            usage=usage,
            model=served_by,
        )


def _summarize(exc: ValidationError, limit: int = 5) -> str:
    lines = []
    for err in exc.errors()[:limit]:
        location = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"- {location}: {err['msg']}")
    if exc.error_count() > limit:
        lines.append(f"- ... and {exc.error_count() - limit} more")
    return "\n".join(lines)
