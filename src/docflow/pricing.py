"""Token pricing, used to report cost per document.

Prices are USD per 1M tokens. Claude prices are Anthropic's first-party list
prices as of 2026-10; Gemini prices are Google's published 2.5-series prices
and should be checked against https://ai.google.dev/pricing before you quote
them. Override any entry with DOCFLOW_PRICE_OVERRIDES='{"model": [in, out]}'.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float


# Each model maps to (min_prompt_tokens, price) tiers, lowest tier first.
PRICES: dict[str, list[tuple[int, Price]]] = {
    "claude-fable-5-1": [(0, Price(10.00, 50.00))],
    "claude-opus-5-5": [(0, Price(4.00, 20.00))],
    "claude-opus-5": [(0, Price(5.00, 25.00))],
    "claude-opus-4-8": [(0, Price(5.00, 25.00))],
    "claude-sonnet-5-5": [(0, Price(2.00, 10.00))],
    "claude-sonnet-5": [(0, Price(2.00, 10.00))],
    "claude-sonnet-4-6": [(0, Price(3.00, 15.00))],
    "claude-haiku-5-5": [(0, Price(0.10, 0.50)), (100_001, Price(0.50, 2.50))],
    "claude-haiku-4-5": [(0, Price(1.00, 5.00))],
    "gemini-2.5-pro": [(0, Price(1.25, 10.00)), (200_001, Price(2.50, 15.00))],
    "gemini-2.5-flash": [(0, Price(0.30, 2.50))],
    "gemini-2.5-flash-lite": [(0, Price(0.10, 0.40))],
}


def _lookup(model: str) -> list[tuple[int, Price]] | None:
    if model in PRICES:
        return PRICES[model]
    # Versioned names such as "gemini-2.5-flash-001": longest matching prefix wins.
    matches = [key for key in PRICES if model.startswith(key)]
    return PRICES[max(matches, key=len)] if matches else None


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    overrides: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    """Cost of one call, or ``None`` when the model has no known price."""
    if overrides and model in overrides:
        price = Price(*overrides[model])
    else:
        tiers = _lookup(model)
        if tiers is None:
            return None
        price = next(p for threshold, p in reversed(tiers) if input_tokens >= threshold)
    total = input_tokens * price.input_per_m + output_tokens * price.output_per_m
    return round(total / 1_000_000, 6)
