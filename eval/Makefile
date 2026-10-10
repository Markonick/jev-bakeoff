# Auto-load eval/.env (gitignored) so any optional model API keys can live in a file
# instead of shell exports. Only the names defined in .env are exported to recipes; a
# var already set in your shell still wins. Copy .env.example → .env to start.
ifneq (,$(wildcard .env))
include .env
export $(shell sed -n 's/^[[:space:]]*\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*=.*/\1/p' .env)
endif

.PHONY: setup model test lint bakeoff corpus redact redact-corpus docker-build docker-test docker-bakeoff

setup:  ## install deps
	uv sync

model:  ## download the Greek spaCy NER model (redaction only)
	uv run python -m spacy download el_core_news_sm

# Deterministic, free, offline — the blocking suite. No API key required.
test:
	uv run pytest -q

lint:
	uv run ruff check . ../packages/redaction/redaction && uv run ruff format --check . && uv run mypy ../packages/redaction/redaction corpus evals

# Projected $/1000-calls of every candidate model per gated surface, from the real corpus
# token profiles (no API key). Writes bakeoff.md.
bakeoff:
	uv run python -m evals.bakeoff

# --- Containerised runs (same image local + CI). Both are key-free. ---------------------
docker-build:
	docker compose build

docker-test: ## deterministic suite in the container (no key)
	docker compose run --rm eval

docker-bakeoff:
	docker compose run --rm eval uv run python -m evals.bakeoff

# Render the synthetic corpus → corpus/generated/ (gitignored, reproducible from seed).
corpus:
	uv run python -m corpus.generate

# Redact a local folder of real PDFs → ./redacted (gitignored). Human-verify output
# before promoting anything into a committed corpus. Usage: make redact IN=/path/to/real
redact:
	uv run python -m redaction.cli --in $(IN) --out ./redacted

# Real-doc INGESTION: redacts a folder of your own real PDFs into corpus/real/ (gitignored)
# + quarantines any that need OCR. NOTHING here is committed — the redacted PDFs, the
# quarantine, and the live manifest are all gitignored, and you must HAND-VERIFY every
# redacted file (automated redaction is never 100%) before adding a row.
# Usage: make redact-corpus IN=/path/to/real   →   then fill corpus/real_manifest.yaml.
redact-corpus:
	uv run python -m redaction.cli --in $(IN) --out corpus/real --quarantine corpus/quarantine
	@echo ""
	@echo "→ Redacted PDFs in corpus/real/  (gitignored — hand-verify EACH one)."
	@echo "→ Next: cp corpus/real_manifest.example.yaml corpus/real_manifest.yaml"
	@echo "  then add one labelled row per verified doc (schema: corpus/README.md)."
