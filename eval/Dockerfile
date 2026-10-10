# syntax=docker/dockerfile:1
# Eval-harness runner — the SAME image runs locally (docker compose) and in CI.
#   no key  -> deterministic suite + bake-off (free, offline)
#   key set -> the paid LLM accuracy gates too
# Build context is the REPO ROOT (the eval imports agent source + the redaction package):
#   docker build -f eval/Dockerfile -t jev-bakeoff-eval .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Project venv OUTSIDE the source tree so a live source mount (local dev) never shadows it.
ENV UV_PROJECT_ENVIRONMENT=/opt/eval-venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    # Non-secret agent Settings stubs so config-coupled seams (rate/classifier) import even
    # outside pytest (conftest sets these too). ANTHROPIC_API_KEY is deliberately NOT set —
    # its absence makes the paid gates skip; pass it at run time to actually run them.
    DATABASE_URL=postgresql://eval:eval@localhost/eval-not-connected \
    AGENT_QUEUE_URL=https://eval.invalid/queue-not-called \
    S3_BUCKET=eval-bucket-not-called

WORKDIR /work
# The eval imports the real agent seams (sut.py puts apps/agent on sys.path) and the
# shared redaction package (pytest pythonpath = ../packages/redaction). .dockerignore
# strips every .venv/__pycache__, so these copies are source-only.
COPY packages/ /work/packages/
COPY apps/agent/ /work/apps/agent/
COPY eval/ /work/eval/

WORKDIR /work/eval
# Locked deps for reproducibility; the Greek NER model the redaction tests need.
RUN uv sync --frozen
# spaCy's downloader shells out to pip, which the uv venv omits — give it an ephemeral one.
RUN uv run --with pip python -m spacy download el_core_news_sm

# Default: the free, deterministic suite (wiring + corpus + redaction + bake-off math).
CMD ["uv", "run", "pytest", "-q", "-m", "not eval"]
