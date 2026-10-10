"""Configurable DeepEval judge model — the judge is NOT Claude-only.

The LLM-graded metrics (GEval, Faithfulness, AnswerRelevancy, Toxicity, Bias,
PromptAlignment) each need a judge model to grade with. This picks it:

  EVAL_JUDGE_PROVIDER = ollama (default) | anthropic | gemini | bedrock | openai_compat
  EVAL_JUDGE_MODEL    = qwen2.5:14b (default) | claude-… | gemini-… | eu.amazon.nova-… | …

Default is **local Ollama** so the whole judge layer runs key-free (dev + CI without
credits). Point it at a frontier grader (Anthropic/Gemini) for the "official" numbers —
a 14B local judge is a weaker grader, so tighten thresholds when you switch. Deterministic
metrics (JsonCorrectness vs a schema, our BaseMetrics) need no judge at all.
"""

from __future__ import annotations

import os

from deepeval.models import (
    AmazonBedrockModel,
    AnthropicModel,
    DeepEvalBaseLLM,
    GeminiModel,
    LocalModel,
)


def local_judge() -> DeepEvalBaseLLM:
    provider = os.getenv("EVAL_JUDGE_PROVIDER", "ollama")
    model = os.getenv("EVAL_JUDGE_MODEL", "qwen2.5:14b")
    if provider == "ollama":
        # Ollama's OpenAI-compatible endpoint (/v1) via LocalModel — avoids the extra
        # `ollama` python package that deepeval's OllamaModel needs; same path our
        # OpenAICompatAdapter uses, so judge + SUT agree on how they reach the model.
        return LocalModel(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            temperature=0,
        )
    if provider == "anthropic":
        return AnthropicModel(model=model, temperature=0)
    if provider == "gemini":
        return GeminiModel(model=model, temperature=0)
    if provider == "bedrock":
        return AmazonBedrockModel(
            model=model, region=os.getenv("EVAL_JUDGE_REGION", "eu-central-1"), temperature=0
        )
    if provider == "openai_compat":
        return LocalModel(
            model=model,
            base_url=os.environ["EVAL_JUDGE_BASE_URL"],
            api_key=os.getenv("EVAL_JUDGE_API_KEY", ""),
            temperature=0,
        )
    raise ValueError(f"unknown EVAL_JUDGE_PROVIDER {provider!r}")
