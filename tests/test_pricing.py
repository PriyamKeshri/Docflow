from __future__ import annotations

import pytest

from docflow.pricing import cost_usd


def test_known_model() -> None:
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 100_000) == pytest.approx(2.0 + 1.0)


def test_tiered_pricing_above_the_threshold() -> None:
    assert cost_usd("claude-haiku-5-5", 100_000, 0) == pytest.approx(0.01)
    assert cost_usd("claude-haiku-5-5", 200_000, 0) == pytest.approx(0.10)


def test_versioned_names_match_by_prefix() -> None:
    assert cost_usd("gemini-2.5-flash-001", 1_000_000, 0) == pytest.approx(0.30)
    assert cost_usd("gemini-2.5-flash-lite", 1_000_000, 0) == pytest.approx(0.10)


def test_overrides_and_unknown_models() -> None:
    assert cost_usd("my-model", 1_000_000, 1_000_000, {"my-model": (1.0, 2.0)}) == pytest.approx(
        3.0
    )
    assert cost_usd("mystery-model", 10, 10) is None
