"""Cost model for the model bake-off.

Published per-token prices + a provider-aware token estimator, so we can project the
$/call and $/1000-ops of every candidate model per surface WITHOUT hitting an API. The
accuracy half of the bake-off comes from running the gates with a key; this half is the
decision-driver (open models are ~10-60x cheaper on our doc workload — the ratio is
robust even as absolute prices drift).

The interesting lever is **image tokens per PDF page**: providers bill a document page
very differently (Anthropic ~1600 tok/page vs Gemini a flat 258 tok/page), which — on top
of the per-token price — is why Gemini/Nova are dramatically cheaper on the vision surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

# VERIFY before quoting externally — model prices drift. Ballpark as of 2026-08; the
# RATIOS matter more than the absolutes. USD per 1M tokens.
PRICES_AS_OF = "2026-08 (ballpark — verify current provider pricing)"


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    in_per_m: float  # $ / 1M input tokens
    out_per_m: float  # $ / 1M output tokens
    image_tokens_per_page: int | None  # billed tokens for one PDF page; None = no vision
    note: str = ""

    @property
    def vision(self) -> bool:
        return self.image_tokens_per_page is not None


PRICES: dict[str, ModelPrice] = {
    # Anthropic — the current baseline. Prompt caching (cached_system) can cut input on
    # repeated calls; not modelled here (conservative = it only helps Anthropic).
    "claude-sonnet-4-6": ModelPrice("anthropic", 3.00, 15.00, 1600, "current baseline"),
    "claude-haiku-4-5": ModelPrice("anthropic", 0.80, 4.00, 1600, "navigator/radar today"),
    # Google — native PDF, flat 258 tokens/page: the cheapest doc reader.
    "gemini-2.0-flash": ModelPrice("google", 0.10, 0.40, 258, "native PDF, 258 tok/page"),
    # AWS Bedrock (eu-central-1) — EU residency, IAM-native. Nova Lite/Pro read images.
    "nova-lite": ModelPrice("bedrock", 0.06, 0.24, 1300, "EU; cheap vision"),
    "nova-pro": ModelPrice("bedrock", 0.80, 3.20, 1300, "EU; stronger vision"),
    "nova-micro": ModelPrice("bedrock", 0.035, 0.14, None, "EU; text-only, tiny"),
    # Open text models (Bedrock EU pricing; DeepInfra/Together are cheaper still).
    "llama-3.3-70b": ModelPrice("bedrock", 0.72, 0.72, None, "text-only reasoning"),
}

# Greek-heavy mixed text tokenises heavier than English — ~3.5 chars/token.
_CHARS_PER_TOKEN = 3.5


def text_tokens(chars: int) -> int:
    return ceil(chars / _CHARS_PER_TOKEN)


def input_tokens(model: str, *, prompt_chars: int, pages: int = 0, page_text_chars: int = 0) -> int:
    """Estimate input tokens: the prompt text, plus either the PDF pages billed as images
    (vision surfaces) or the page text (text surfaces)."""
    price = PRICES[model]
    toks = text_tokens(prompt_chars)
    if pages and price.image_tokens_per_page is not None:
        toks += pages * price.image_tokens_per_page
    if page_text_chars:
        toks += text_tokens(page_text_chars)
    return toks


def cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICES[model]
    return in_tokens / 1e6 * p.in_per_m + out_tokens / 1e6 * p.out_per_m
