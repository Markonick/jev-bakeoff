# eval — offline LLM bake-off + PII redaction

Dev-only tooling for LLM classification/extraction surfaces (Greek mortgage documents).
Standalone project pinned to Python 3.12 because the ML stack (spaCy / Presidio) has no
wheels for the agent's 3.14 runtime, and these heavy deps are kept out of the agent image.

> **This is a carved, keyless slice.** The paid LLM *accuracy* gates and their explore
> tooling (the DeepEval validator/portal/offer/rate gates, the promptfoo `board/`, the
> Langfuse experiment stack, and `evals/sut.py`, which drove the real `apps/agent` agents)
> have been removed — they require a live API key and the full agent, which are not part of
> this slice. What remains runs **fully offline, no API key**: the PII redaction layer, the
> synthetic corpus generator, and the costed model **bake-off**.

The TypeSafe **Jev** System-1 classifier adapter lives at `../apps/agent/core/jev_client.py`
(present, not yet wired into the bake-off — that's the open task).

## Setup

```bash
cd eval
uv sync                                            # deps
make model                                         # Greek spaCy NER model (redaction only)
```

## Model bake-off (`make bakeoff`) — keyless

Projects the $/1000-calls of every candidate model (Sonnet / Haiku / Gemini-Flash / Nova /
Llama) per surface, grounded in each corpus's **real token profile**
(`evals/pricing.py` + `evals/bakeoff.py` → `bakeoff.md`). No API key — this is the **cost**
half of a swap decision. The **accuracy** half requires running the real surface against a
live model (removed from this slice); a swap ships only if it holds the gate's threshold.

```bash
make bakeoff      # writes bakeoff.md
```

## Tests

```bash
make test         # deterministic, offline, free — redaction + corpus + bake-off wiring
make lint         # ruff + mypy
```

`make test` is model-free (runs the `-m "not eval"` suite). The spaCy NER path
(`use_ner=True`) is exercised manually.

## Run in a container

One image runs the whole keyless harness, identically on your machine and in CI. Build
context is the repo root (the harness imports the redaction package + `apps/agent/core`).

```bash
make docker-build     # build jev-bakeoff-eval:latest
make docker-test      # deterministic suite (no key)
make docker-bakeoff   # cost projection
```

## PII redaction (`../packages/redaction`)

Turns real Greek mortgage documents into sanitised fixtures (importable as `redaction`):

- **`recognizers.py`** — custom Greek recognizers Presidio lacks: ΑΦΜ (checksum),
  ΑΜΚΑ (date), ID card, passport, phone, plus a label-anchored name recognizer that catches
  the **ALL-CAPS names** tax forms use (which the spaCy `sm` model misses).
- **`redactor.py`** — `PiiAnalyzer` (regex + optional Greek NER) and `PdfRedactor`, which
  applies **true PyMuPDF redaction** (removes the underlying text — not a drawn-over box that
  leaves it extractable).

```bash
make redact IN=/path/to/real      # → redacted/ (gitignored)
# or directly:
uv run python -m redaction.cli --in /path/to/real --out ./redacted
```

### Safety model — read before running on real data

Automated redaction is **never 100%**. Therefore:

- **Nothing derived from real docs is committed.** `real/`, `redacted/`, `quarantine/` are
  gitignored. Human-verify a file before promoting it into any committed corpus.
- **No-text-layer / locked PDFs are quarantined**, never emitted unredacted.
- **ID cards / passports are never committed**, even redacted — a face photo and layout
  survive text redaction.

## Synthetic corpus (`corpus/`)

Seed-driven generator: `corpus/seed.yaml` (people, doc-type keywords, look-alike / unrelated
specs, cases × variants) → `corpus/generate.py` renders one PDF per `(case, variant)` with a
real Greek text layer (fpdf2 + embedded DejaVu) into the gitignored `corpus/generated/`, plus
a `manifest.jsonl` pairing each doc with the requirement to check and the **expected verdict**.

```bash
make corpus       # → corpus/generated/  (reproducible from seed; not committed)
```

Variants exercise both axes: `clean` / `wrong_year` / `wrong_name` / `wrong_type` / `empty` /
`look_alike` / `unrelated` and terminal `non_pdf` / `password`. **`look_alike`** (same layout,
identifying keywords stripped) is the headline sturdiness trap.
