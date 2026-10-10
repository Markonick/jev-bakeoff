"""Costed model bake-off for the gated swap candidates.

For each gated surface (validator / classifier / offer / rate) this projects the $/1000
calls of every candidate model, grounded in the surface's REAL corpus (measured PDF pages
+ page text), so the comparison reflects our actual token profile — no API key needed.

The ACCURACY column is filled by running the real surface against a live model (a paid gate,
not part of this keyless slice); the gate is the quality floor that makes a swap safe.
This module is the cost half — run `make bakeoff` (or `python -m evals.bakeoff`).

Caveats printed in the report: single-page synthetic docs (real docs are multi-page →
cost scales ~linearly with pages); prompt overhead is approximate (dwarfed by image
tokens on vision surfaces); Anthropic prompt-caching is not modelled (helps the baseline).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import fitz  # PyMuPDF — count pages of the rendered corpus PDFs

from evals.classify_dataset import load_classify_cases
from evals.offer_dataset import load_offer_cases
from evals.pricing import PRICES, PRICES_AS_OF, cost_usd, input_tokens
from evals.rate_dataset import load_rate_cases

# Candidate models, baseline first. (llama/nova-micro are text-only → N/A on vision rows.)
CANDIDATES = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "gemini-2.0-flash",
    "nova-pro",
    "nova-lite",
    "llama-3.3-70b",
]
_BASELINE = "claude-sonnet-4-6"


@dataclass(frozen=True)
class SurfaceProfile:
    name: str
    modality: str  # "vision" | "text"
    prompt_chars: int  # system+user overhead (approx; dominated by image tokens on vision)
    out_tokens: int  # typical response size
    pages: float = 0.0  # mean PDF pages (vision)
    page_text_chars: int = 0  # mean page text length (text surfaces)


def _pages(pdf: bytes) -> int:
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        return doc.page_count


def _profiles() -> list[SurfaceProfile]:
    classify = load_classify_cases()
    offer = load_offer_cases()
    rate = load_rate_cases()
    return [
        # Validator ≈ classifier vision profile (own corpus not loaded here; run its gate
        # for accuracy). One-page assessment, small verdict JSON.
        SurfaceProfile("validator (#1)", "vision", prompt_chars=1500, out_tokens=220, pages=1.0),
        SurfaceProfile(
            "classifier (#2/#3)",
            "vision",
            prompt_chars=1400,
            out_tokens=90,
            pages=mean(_pages(c.pdf) for c in classify),
        ),
        SurfaceProfile(
            "offer (#6)",
            "vision",
            prompt_chars=1300,
            out_tokens=260,
            pages=mean(_pages(c.pdf) for c in offer),
        ),
        SurfaceProfile(
            "rate (#8)",
            "text",
            prompt_chars=2600,
            out_tokens=350,
            page_text_chars=int(mean(len(c.page) for c in rate)),
        ),
    ]


def _per_1000(model: str, s: SurfaceProfile) -> float | None:
    price = PRICES[model]
    if s.modality == "vision" and not price.vision:
        return None  # text-only model can't read the PDF
    in_tok = input_tokens(
        model,
        prompt_chars=s.prompt_chars,
        pages=round(s.pages) if s.modality == "vision" else 0,
        page_text_chars=s.page_text_chars if s.modality == "text" else 0,
    )
    return cost_usd(model, in_tok, s.out_tokens) * 1000


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"${v:,.2f}"


def build_report() -> str:
    profiles = _profiles()
    lines = [
        "# Costed model bake-off — gated swap candidates",
        "",
        f"Prices: {PRICES_AS_OF}. Cost is projected from each surface's REAL corpus "
        "(measured pages + text); accuracy is filled by running the gates with a key.",
        "",
        "## $ per 1,000 calls, by surface × model (vs Sonnet baseline)",
        "",
        "| Surface | " + " | ".join(CANDIDATES) + " |",
        "|" + "---|" * (len(CANDIDATES) + 1),
    ]
    for s in profiles:
        base = _per_1000(_BASELINE, s)
        cells = []
        for m in CANDIDATES:
            v = _per_1000(m, s)
            if v is None:
                cells.append("—")
            elif m == _BASELINE:
                cells.append(f"**{_fmt(v)}**")
            else:
                factor = base / v if base and v else 0
                cells.append(f"{_fmt(v)} ({factor:.0f}×)")
        lines.append(f"| {s.name} | " + " | ".join(cells) + " |")

    # Headline: per 1,000 cases, all-Sonnet vs the cheapest viable per-surface pick.
    mix = {"validator (#1)": 8, "classifier (#2/#3)": 1, "offer (#6)": 1}  # rate is scheduled
    by_name = {s.name: s for s in profiles}

    def _cost(model: str, name: str) -> float:
        v = _per_1000(model, by_name[name])
        assert v is not None, f"{model} cannot serve {name}"  # baseline is always vision-capable
        return v

    all_sonnet = sum(_cost(_BASELINE, n) * calls for n, calls in mix.items())
    cheapest = 0.0
    picks = []
    for n, calls in mix.items():
        s = by_name[n]
        opts: list[tuple[str, float]] = []
        for m in CANDIDATES:
            v = _per_1000(m, s)
            if v is not None:
                opts.append((m, v))
        best_model, best_cost = min(opts, key=lambda kv: kv[1])
        cheapest += best_cost * calls
        picks.append(f"{n} → {best_model}")
    cut = (1 - cheapest / all_sonnet) * 100 if all_sonnet else 0
    lines += [
        "",
        "## Headline — per 1,000 cases (per-case mix: 8 validations + 1 classify + 1 offer)",
        "",
        f"- **All-Sonnet (today):** ${all_sonnet:,.0f} / 1,000 cases",
        f"- **Cheapest viable per surface:** ${cheapest:,.0f} / 1,000 cases "
        f"→ **~{cut:.0f}% cost cut**",
        "- Picks: " + "; ".join(picks),
        "",
        "## Notes",
        "- Vision cost is driven by **image tokens/page** (Anthropic ~1600 vs Gemini 258) "
        "compounding with the per-token price — hence the large factors on the doc surfaces.",
        "- Synthetic corpus docs are 1 page; real docs are multi-page → per-call cost scales "
        "~linearly with pages (the ratios between models hold).",
        "- Accuracy is NOT shown — it needs the real surface run against a live model (a paid "
        "gate, not part of this keyless slice); keep only swaps that hold the gate's threshold.",
        "- text-only models (llama-3.3-70b, nova-micro) show `—` on vision rows.",
    ]
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    out = Path(__file__).resolve().parent.parent / "bakeoff.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n(written to {out})")


if __name__ == "__main__":
    main()
