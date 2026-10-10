"""Synthetic corpus for the bulk-upload **document classifier** gate.

Each case renders a Greek document to a PDF (shared ``corpus.renderers``) and pairs it
with the candidate requirements the buyer's case has open + the expected ``match_index``
(the [n] the classifier should pick, or ``null`` for no-match / misfile-avoidance). The
same ``_classify`` seam backs both the bulk-upload and capture classifiers, so this gate
covers both. Nothing derived from a real document is committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from corpus.renderers import DocSpec, render_document
from corpus.transforms import inject, multipage, scan

_CASES_YAML = Path(__file__).resolve().parent.parent / "corpus" / "classify_cases.yaml"


def _transform(pdf: bytes, name: str | None, *, seed_int: int) -> bytes:
    """Apply an optional input-diversity transform to a rendered case (ground truth
    unchanged — the document still says the same thing, it just reads harder)."""
    if name is None:
        return pdf
    if name == "scanned":
        return scan(pdf, seed=seed_int)
    if name == "multipage":
        return multipage(pdf, before=1, after=1)
    if name == "injection":
        return inject(pdf)
    raise ValueError(f"unknown transform: {name}")


@dataclass(frozen=True)
class ClassifyCase:
    id: str
    pdf: bytes
    candidates: list[dict[str, Any]]  # each: label_en, document_type, tax_year, ai_hint
    buyer_name: str | None
    expect_index: int | None  # the candidate the classifier should match, or None
    slice: str = "clean"  # per-slice score breakdown


def _spec(doc: dict[str, Any]) -> DocSpec:
    return DocSpec(
        title=doc["title"],
        subtitle=doc.get("subtitle", ""),
        fields=[(str(k), str(v)) for k, v in doc.get("fields", [])],
        body=list(doc.get("body", [])),
    )


def _candidate(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_en": c.get("label_en"),
        "document_type": c["document_type"],
        "tax_year": c.get("tax_year"),
        "ai_hint": c.get("ai_hint"),
    }


def load_classify_cases(path: Path = _CASES_YAML) -> list[ClassifyCase]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases: list[ClassifyCase] = []
    for row in data["cases"]:
        pdf = render_document(_spec(row["doc"]))
        transform = row.get("transform")
        pdf = _transform(pdf, transform, seed_int=sum(ord(c) for c in row["id"]))
        cases.append(
            ClassifyCase(
                id=row["id"],
                pdf=pdf,
                candidates=[_candidate(c) for c in row["candidates"]],
                buyer_name=row.get("buyer_name"),
                expect_index=row.get("expect_index"),
                slice=row.get("slice", "clean"),
            )
        )
    return cases


def _print_counts() -> None:
    from collections import Counter

    cases = load_classify_cases()
    by_slice = Counter(c.slice for c in cases)
    print(f"classify: {len(cases)} cases")
    print("  by slice: " + ", ".join(f"{k}={v}" for k, v in sorted(by_slice.items())))


if __name__ == "__main__":
    _print_counts()
