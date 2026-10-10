"""Deterministic checks for the cost model + bake-off report (free CI, no model)."""

from __future__ import annotations

from evals.bakeoff import build_report
from evals.pricing import PRICES, cost_usd, input_tokens


def test_cost_math() -> None:
    # 1M in + 1M out at sonnet's 3/15 = $18.
    assert abs(cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) - 18.0) < 1e-6
    # A vision call is cheaper on Gemini than Sonnet (flat 258 tok/page + low price).
    son = cost_usd(
        "claude-sonnet-4-6", input_tokens("claude-sonnet-4-6", prompt_chars=1400, pages=1), 90
    )
    gem = cost_usd(
        "gemini-2.0-flash", input_tokens("gemini-2.0-flash", prompt_chars=1400, pages=1), 90
    )
    assert gem < son / 10  # comfortably >10x cheaper on a 1-page doc


def test_page_image_tokens_scale() -> None:
    one = input_tokens("claude-sonnet-4-6", prompt_chars=1000, pages=1)
    three = input_tokens("claude-sonnet-4-6", prompt_chars=1000, pages=3)
    # Each extra page adds the per-page image token cost.
    assert three - one == 2 * PRICES["claude-sonnet-4-6"].image_tokens_per_page


def test_text_only_model_has_no_vision() -> None:
    assert PRICES["llama-3.3-70b"].image_tokens_per_page is None
    # A text-only model ignores pages (no image tokens added).
    assert input_tokens("llama-3.3-70b", prompt_chars=1000, pages=5) == input_tokens(
        "llama-3.3-70b", prompt_chars=1000, pages=0
    )


def test_report_builds_over_real_corpora() -> None:
    report = build_report()
    assert "Costed model bake-off" in report
    assert "cost cut" in report
    # text-only model shows N/A on a vision row.
    assert "—" in report
