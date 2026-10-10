# jev-bakeoff

Standalone, **offline** LLM classifier bake-off + PII-redaction eval harness.

The harness projects the per-surface $/1,000-calls of candidate models over a real
Greek-mortgage-document corpus, and exercises a deterministic PII-redaction pipeline
for turning real documents into sanitised fixtures. Everything in the default workflow
runs **without an API key**.

The Jev (TypeSafe System-1) classifier adapter is present at
`apps/agent/core/jev_client.py` as a candidate backend; it is not yet wired into the
bake-off.

## Run (all keyless)

```bash
cd eval
make setup     # uv sync
make test      # deterministic suite (pytest) — redaction, corpus, bake-off, judge math
make bakeoff   # token/cost projection over the corpus → eval/bakeoff.md
```

Python 3.12 via [uv](https://docs.astral.sh/uv/). The ML stack (spaCy / Presidio /
PyMuPDF) is pinned to 3.12.

## Layout

```
jev-bakeoff/
├── eval/                       # the eval harness (own pyproject.toml, uv.lock, Makefile, Dockerfile)
│   ├── evals/                  # bake-off (bakeoff.py), datasets, pricing, judges, judge calibration
│   ├── corpus/                 # seed-driven synthetic Greek-document generator + fonts
│   └── tests/                  # deterministic, key-free test suite
├── packages/redaction/         # shared PII detection + true PDF redaction (imported as `redaction`)
└── apps/agent/core/            # core modules the eval imports (llm_client, model_registry,
                                #   jev_client, config, …) — the agent seam, core only
```

`eval/` adds `../packages/redaction` to its pytest pythonpath (imported as `redaction`)
and inserts `apps/agent` on `sys.path` to import the real agent `core.*` modules.

## Other targets

`make corpus` renders the synthetic corpus to `eval/corpus/generated/` (gitignored,
reproducible from seed). `make redact IN=/path/to/real` redacts a local folder of real
PDFs. `make docker-build` / `make docker-test` / `make docker-bakeoff` run the same
suite in a container. See `eval/README.md` for details on the redaction safety model
and the corpus.
