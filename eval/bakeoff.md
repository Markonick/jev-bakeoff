# Costed model bake-off — gated swap candidates

Prices: 2026-08 (ballpark — verify current provider pricing). Cost is projected from each surface's REAL corpus (measured pages + text); accuracy is filled by running the gates with a key.

## $ per 1,000 calls, by surface × model (vs Sonnet baseline)

| Surface | claude-sonnet-4-6 | claude-haiku-4-5 | gemini-2.0-flash | nova-pro | nova-lite | llama-3.3-70b |
|---|---|---|---|---|---|---|
| validator (#1) | **$9.39** | $2.50 (4×) | $0.16 (60×) | $2.09 (4×) | $0.16 (60×) | — |
| classifier (#2/#3) | **$7.35** | $1.96 (4×) | $0.10 (72×) | $1.65 (4×) | $0.12 (59×) | — |
| offer (#6) | **$9.82** | $2.62 (4×) | $0.17 (59×) | $2.17 (5×) | $0.16 (60×) | — |
| rate (#8) | **$7.61** | $2.03 (4×) | $0.22 (35×) | $1.75 (4×) | $0.13 (58×) | $0.82 (9×) |

## Headline — per 1,000 cases (per-case mix: 8 validations + 1 classify + 1 offer)

- **All-Sonnet (today):** $92 / 1,000 cases
- **Cheapest viable per surface:** $2 / 1,000 cases → **~98% cost cut**
- Picks: validator (#1) → nova-lite; classifier (#2/#3) → gemini-2.0-flash; offer (#6) → nova-lite

## Notes
- Vision cost is driven by **image tokens/page** (Anthropic ~1600 vs Gemini 258) compounding with the per-token price — hence the large factors on the doc surfaces.
- Synthetic corpus docs are 1 page; real docs are multi-page → per-call cost scales ~linearly with pages (the ratios between models hold).
- Accuracy is NOT shown — it needs the real surface run against a live model (a paid gate, not part of this keyless slice); keep only swaps that hold the gate's threshold.
- text-only models (llama-3.3-70b, nova-micro) show `—` on vision rows.