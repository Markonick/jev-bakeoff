"""Offline checks for the configurable judge + DeepEval's deterministic JSON metric.

No model calls: constructing a judge doesn't hit the provider, and JsonCorrectness scores
by parsing, so both run key-free in the CI suite.
"""

from __future__ import annotations

import pytest
from deepeval.models import AnthropicModel, LocalModel
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

from evals.judges import local_judge


def test_default_judge_is_local_ollama() -> None:
    # Default = local Ollama via its OpenAI-compatible endpoint (LocalModel) — key-free.
    assert isinstance(local_judge(), LocalModel)


def test_judge_provider_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-eval-placeholder")  # construct only, no call
    assert isinstance(local_judge(), AnthropicModel)
    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError):
        local_judge()


def test_json_correctness_is_deterministic() -> None:
    from deepeval.metrics import JsonCorrectnessMetric

    class S(BaseModel):
        a: str | None = None

    # model is required at construction (metric wiring) but never called — scoring is a
    # pure parse against the schema. Pass the local judge; it stays untouched.
    ok = JsonCorrectnessMetric(expected_schema=S(), model=local_judge(), include_reason=False)
    assert ok.measure(LLMTestCase(input="x", actual_output='{"a": "b"}')) == 1
    bad = JsonCorrectnessMetric(expected_schema=S(), model=local_judge(), include_reason=False)
    assert bad.measure(LLMTestCase(input="x", actual_output="not json")) == 0
