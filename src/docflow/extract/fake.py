"""Offline extractor for tests, CI and smoke-testing the stack without an API key."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from docflow.config import Settings
from docflow.extract.base import Completion, Extractor

Scripted = str | dict[str, Any] | Callable[[str], str | dict[str, Any]]

DEFAULT_RESPONSE: dict[str, Any] = {
    "document_type": "other",
    "invoice": None,
    "bank_statement": None,
    "resume": None,
    "field_confidence": [],
    "notes": "fake provider: no model call was made",
}


class FakeExtractor(Extractor):
    """Replays scripted responses in order, then falls back to ``default``.

    A script entry may be a JSON string, a dict, or a callable that receives the
    user prompt (so tests can branch on document content).
    """

    provider = "fake"

    def __init__(
        self,
        settings: Settings,
        responses: Iterable[Scripted] = (),
        default: Scripted | None = None,
    ) -> None:
        super().__init__("fake", settings)
        self.responses = list(responses)
        self.default = default if default is not None else DEFAULT_RESPONSE
        self.prompts: list[str] = []

    def _complete(self, system: str, user: str) -> Completion:
        self.prompts.append(user)
        item = self.responses.pop(0) if self.responses else self.default
        if callable(item):
            item = item(user)
        text = item if isinstance(item, str) else json.dumps(item)
        return Completion(
            text=text,
            model="fake",
            input_tokens=(len(system) + len(user)) // 4,
            output_tokens=len(text) // 4,
        )
