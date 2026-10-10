"""Batch-redact a local folder of real PDFs into sanitised fixtures.

    uv run python -m redaction.cli --in ./real --out ./redacted

Every input stays put; outputs land under ``--out`` (redacted) and ``--quarantine``
(a manifest of files that need manual/OCR handling). Both are gitignored — nothing
here is auto-committed. Human-verify a redacted file before promoting it into any
committed corpus.

Defaults to NER on (names/addresses matter for real docs); needs the spaCy Greek
model — install with ``uv run python -m spacy download el_core_news_sm``. Pass
``--no-ner`` for a regex-only pass (structured IDs only).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from redaction.redactor import PdfRedactor, PiiAnalyzer


def _redact_dir(src: Path, out: Path, quarantine: Path, *, use_ner: bool) -> int:
    pdfs = sorted(src.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {src}", file=sys.stderr)
        return 1

    analyzer = PiiAnalyzer(use_ner=use_ner)
    redactor = PdfRedactor(analyzer)
    out.mkdir(parents=True, exist_ok=True)

    totals: Counter[str] = Counter()
    quarantined: list[dict[str, str]] = []
    redacted_count = 0

    for path in pdfs:
        result = redactor.redact_bytes(path.read_bytes())
        if result.redacted and result.pdf is not None:
            (out / path.name).write_bytes(result.pdf)
            totals.update(result.entities)
            redacted_count += 1
            summary = ", ".join(f"{k}×{v}" for k, v in result.entities.items()) or "no PII found"
            print(f"  redacted  {path.name}  ({summary})")
        else:
            quarantined.append({"file": path.name, "reason": result.status})
            print(f"  QUARANTINE {path.name}  ({result.status})")

    print(f"\nRedacted {redacted_count}/{len(pdfs)} → {out}")
    if totals:
        print("Entities removed: " + ", ".join(f"{k}={v}" for k, v in totals.most_common()))
    if quarantined:
        quarantine.mkdir(parents=True, exist_ok=True)
        manifest = quarantine / "manifest.json"
        manifest.write_text(json.dumps(quarantined, indent=2, ensure_ascii=False))
        print(
            f"{len(quarantined)} file(s) quarantined (no text layer / locked) — "
            f"see {manifest}. These need OCR-based handling and must NOT be committed."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Redact PII from a folder of PDFs.")
    parser.add_argument("--in", dest="src", type=Path, required=True, help="folder of real PDFs")
    parser.add_argument("--out", type=Path, default=Path("redacted"), help="output folder")
    parser.add_argument(
        "--quarantine", type=Path, default=Path("quarantine"), help="quarantine manifest folder"
    )
    parser.add_argument(
        "--no-ner", dest="ner", action="store_false", help="skip spaCy NER (structured IDs only)"
    )
    args = parser.parse_args(argv)
    return _redact_dir(args.src, args.out, args.quarantine, use_ner=args.ner)


if __name__ == "__main__":
    raise SystemExit(main())
