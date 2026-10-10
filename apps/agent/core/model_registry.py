"""The per-surface MODEL STRATEGY — the single, app-owned source of truth.

One typed table, consumed by two sides:
- runtime: ``llm_client.resolve()`` reads each surface's ``default`` (the shipped
  production route), overridable per-run by ``LLM_<SURFACE>_PROVIDER/MODEL`` env.
- bake-off: the eval orchestrator reads each surface's ``candidates`` + ``modality`` to
  run the same eval across models and land a Langfuse dataset run per model.

This is NOT test/eval config and NOT env: the candidate matrix is application data that
lives here so there's exactly one place that answers "which model(s) does surface X use".
Env stays for overrides only. A route is ``(provider, model)`` where provider is one of
``anthropic | openai_compat | gemini | bedrock`` (see ``llm_client._PROVIDER_PREFIX``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Route = tuple[str, str]  # (provider, model)
Modality = Literal["text", "vision"]


@dataclass(frozen=True)
class SurfaceModels:
    """A surface's model strategy: the production ``default`` + the ``candidates`` the
    bake-off evaluates (the default is always first). ``modality`` tells the bake-off
    whether the surface feeds PDFs — a ``vision`` surface needs the rasterise route
    (``LLM_OPENAI_VISION=1``) for local models and picks vision-capable candidates.

    ``prompt_versions`` is the prompt A/B axis: the Langfuse prompt versions/labels the
    bake-off holds a model fixed against — the "is prompt v2 better than v1" check. It
    defaults to ``["production"]`` (the shipped label), so a surface without prompt
    management sweeps models only and is unaffected. A surface that declares >1 version
    (e.g. the portal) makes the bake-off sweep ``model × prompt_version``."""

    default: Route
    candidates: list[Route]
    modality: Modality
    prompt_versions: list[str] = field(default_factory=lambda: ["production"])


# --- routes ----------------------------------------------------------------------------
# Anthropic tiers (the accuracy reference; need a funded key — run via provider=anthropic).
_HAIKU: Route = ("anthropic", "claude-haiku-4-5-20251001")
_SONNET: Route = ("anthropic", "claude-sonnet-4-6")
_OPUS: Route = ("anthropic", "claude-opus-4-8")

# Local / open (Ollama via the OpenAI-compat endpoint — key-free).
_QWEN_7B: Route = ("openai_compat", "qwen2.5:7b")
_QWEN_14B: Route = ("openai_compat", "qwen2.5:14b")
_LLAMA_8B: Route = ("openai_compat", "llama3.1:8b")
_MISTRAL_7B: Route = ("openai_compat", "mistral:7b")
_GEMMA2_9B: Route = ("openai_compat", "gemma2:9b")
# Vision-capable local (PDF pages rasterised to images).
_QWEN_VL_7B: Route = ("openai_compat", "qwen2.5vl:7b")
_QWEN_VL_3B: Route = ("openai_compat", "qwen2.5vl:3b")
_LLAVA_7B: Route = ("openai_compat", "llava:7b")

# Other hosted clouds (API — need creds; the bake-off auto-skips a cell whose cred is
# absent). Both Gemini + Nova read PDFs natively, so they serve text AND vision surfaces.
# Gemini → LLM_GEMINI_API_KEY (Google AI Studio, has a free tier); Nova → AWS creds +
# Bedrock model access in the region (LLM_BEDROCK_REGION, default eu-central-1).
_GEMINI_FLASH: Route = ("gemini", "gemini-3.6-flash")
_NOVA_LITE: Route = ("bedrock", "eu.amazon.nova-lite-v1:0")
_NOVA_PRO: Route = ("bedrock", "eu.amazon.nova-pro-v1:0")
# Claude Haiku 4.5 via Bedrock (EU cross-region inference profile) — same model as
# _HAIKU but billed through AWS (no prepaid Anthropic credit balance to run dry) and
# resident in eu-central-1. The migration target for the vision doc surfaces once the
# bake-off shows Haiku holds accuracy vs the Sonnet reference.
_BEDROCK_HAIKU: Route = ("bedrock", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")

_TEXT_LOCAL = [_QWEN_7B, _QWEN_14B, _LLAMA_8B, _MISTRAL_7B, _GEMMA2_9B]
_VISION_LOCAL = [_QWEN_VL_7B, _QWEN_VL_3B, _LLAVA_7B]
_CLOUD_TEXT = [_GEMINI_FLASH, _NOVA_LITE]  # hosted, non-Anthropic
_CLOUD_VISION = [_GEMINI_FLASH, _NOVA_PRO]  # native-PDF readers (Pro on the doc surfaces)


# --- the strategy table ----------------------------------------------------------------
# Prod default = Sonnet on the accuracy surfaces (classifier/offer), Haiku where the eval
# gate showed Haiku ties the bigger tier (rate/validator) or on the cheap/frequent ones.
# Bake-off candidates = the Claude tiers the use-case warrants (Opus as the accuracy ceiling
# on hard/vision surfaces) + the other hosted clouds (Gemini/Nova) + the runnable local
# models for that modality. Hosted cells auto-skip when their cred is absent.
REGISTRY: dict[str, SurfaceModels] = {
    # portal has Langfuse prompt management (see eval/evals/langfuse_prompts.py) so it
    # declares a prompt A/B axis: the shipped ``production`` prompt vs a candidate ``v2``.
    "portal": SurfaceModels(
        _SONNET,
        [_SONNET, _HAIKU, _OPUS, *_CLOUD_TEXT, *_TEXT_LOCAL],
        "text",
        prompt_versions=["production", "v2"],
    ),
    # rate default = Haiku: ties Sonnet at recall 1.000 / 0 cross-bank-leaks on the 20-case
    # corpus (local qwen2.5:14b/llama3.1:8b also 1.000). Same candidate set as before, with
    # Haiku moved to lead so it satisfies the default-leads-candidates invariant.
    "rate": SurfaceModels(_HAIKU, [_HAIKU, _SONNET, *_CLOUD_TEXT, *_TEXT_LOCAL], "text"),
    # Haiku added to the bake-off (Anthropic ref + the Bedrock route) — validator, a
    # comparable read-the-document task, already ties Opus on Haiku, so classification is
    # a strong candidate to drop from Sonnet → Haiku-on-Bedrock (cheaper + no credit
    # blackout + EU). Nova Lite included as the cheapest cloud alternative. Default stays
    # Sonnet until the accuracy board confirms a winner.
    "classifier": SurfaceModels(
        _SONNET,
        [_SONNET, _HAIKU, _BEDROCK_HAIKU, _OPUS, _NOVA_LITE, *_CLOUD_VISION, *_VISION_LOCAL],
        "vision",
    ),
    "offer": SurfaceModels(_SONNET, [_SONNET, _OPUS, *_CLOUD_VISION, *_VISION_LOCAL], "vision"),
    # validator default = Haiku: 0.971 == Opus 0.971 vs Sonnet 0.990 on the 105-case corpus.
    # Same candidate set as before, Haiku moved to lead for the default-leads invariant.
    "validator": SurfaceModels(
        _HAIKU, [_HAIKU, _SONNET, _OPUS, *_CLOUD_VISION, *_VISION_LOCAL], "vision"
    ),
    # Not eval-gated (no bake-off) — cheap/frequent, default Haiku.
    "navigator": SurfaceModels(_HAIKU, [_HAIKU], "text"),
    "market_radar": SurfaceModels(_HAIKU, [_HAIKU], "text"),
    # Migrated onto the port (not eval-gated) — each keeps its shipped Sonnet default
    # (the default is what resolve() serves at runtime; the extra candidates only feed a
    # future bake-off). vision = feeds PDFs (specimen_classifier + case_intelligence read
    # the uploaded PDF), so those carry the vision-capable local candidates the registry
    # invariant requires for a vision surface.
    "progress_summary": SurfaceModels(_SONNET, [_SONNET], "text"),
    "property_analysis": SurfaceModels(_SONNET, [_SONNET], "text"),
    "firm_branding": SurfaceModels(_SONNET, [_SONNET], "text"),
    "specimen_classifier": SurfaceModels(_SONNET, [_SONNET, *_VISION_LOCAL], "vision"),
    "case_intelligence": SurfaceModels(_SONNET, [_SONNET, *_VISION_LOCAL], "vision"),
}

_FALLBACK = SurfaceModels(_SONNET, [_SONNET], "text")


def surface_models(surface: str) -> SurfaceModels:
    """The strategy for a surface (Sonnet-only fallback for an unknown surface)."""
    return REGISTRY.get(surface, _FALLBACK)
