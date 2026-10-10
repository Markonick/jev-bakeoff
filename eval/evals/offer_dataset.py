"""Synthetic corpus for the pre-approval **offer extraction** gate.

Each case is a Greek/English bank pre-approval letter rendered to a real PDF text
layer (via the shared ``corpus.renderers``), paired with the ground-truth field values
the extractor should lift out of it. The fixtures are committed as YAML (the text +
expected fields) and rendered on the fly — nothing derived from a real customer offer
is ever committed (same rule as the validator corpus).

``expect`` holds ONLY the fields we assert for that fixture; a field left out is not
scored (keeps the paid gate robust — we assert what the letter states unambiguously).
A field explicitly set to ``null`` asserts the extractor returns ``None`` (the letter
does not state it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from corpus.renderers import DocSpec, render_document
from corpus.transforms import inject, multipage, scan

_CASES_YAML = Path(__file__).resolve().parent.parent / "corpus" / "offer_cases.yaml"


def _transform(pdf: bytes, name: str | None, *, seed_int: int) -> bytes:
    """Apply an optional input-diversity transform to a rendered offer letter (the stated
    field values are unchanged — the ground truth still holds)."""
    if name is None:
        return pdf
    if name == "scanned":
        return scan(pdf, seed=seed_int)
    if name == "multipage":
        return multipage(pdf, before=1, after=1)
    if name == "injection":
        return inject(pdf)
    raise ValueError(f"unknown transform: {name}")


# The extractor fields we know how to score (mirrors ``ExtractedOffer``). Anything a
# fixture puts in ``expect`` outside this set is a typo — fail loudly at load time.
OFFER_FIELDS = frozenset(
    {
        "bank_name",
        "loan_amount_eur",
        "term_months",
        "purpose",
        "rate_structure",
        "fixed_rate_pct",
        "fixed_period_months",
        "floating_index",
        "floating_index_pct",
        "floating_spread_pct",
        "floating_total_pct",
        "levy_pct",
        "avg_monthly_payment_eur",
    }
)


@dataclass(frozen=True)
class OfferCase:
    id: str
    pdf: bytes
    # Ground truth: field name -> expected value (str | number | None). Only these
    # fields are scored for this case.
    expect: dict[str, Any]
    slice: str = "clean"  # per-slice score breakdown


def _spec(doc: dict[str, Any]) -> DocSpec:
    return DocSpec(
        title=doc["title"],
        subtitle=doc.get("subtitle", ""),
        fields=[(str(label), str(value)) for label, value in doc.get("fields", [])],
        body=list(doc.get("body", [])),
    )


def load_offer_cases(path: Path = _CASES_YAML) -> list[OfferCase]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases: list[OfferCase] = []
    for row in data["cases"]:
        expect = dict(row.get("expect", {}))
        unknown = set(expect) - OFFER_FIELDS
        if unknown:
            raise ValueError(f"{row['id']}: unknown expect field(s) {sorted(unknown)}")
        pdf = render_document(_spec(row["doc"]))
        pdf = _transform(pdf, row.get("transform"), seed_int=sum(ord(c) for c in row["id"]))
        cases.append(
            OfferCase(
                id=row["id"],
                pdf=pdf,
                expect=expect,
                slice=row.get("slice", "clean"),
            )
        )
    return cases


def _print_counts() -> None:
    from collections import Counter

    cases = load_offer_cases()
    by_slice = Counter(c.slice for c in cases)
    print(f"offer: {len(cases)} cases")
    print("  by slice: " + ", ".join(f"{k}={v}" for k, v in sorted(by_slice.items())))


if __name__ == "__main__":
    _print_counts()
