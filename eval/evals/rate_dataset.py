"""Synthetic corpus for the bank **rate extraction** gate.

Each case is the *text* of a Greek bank rates page (committed inline — no PDF render, no
network) plus the ground-truth products the extractor should lift for ``bank_name``. The
harness drives the network-free seam ``_extract(page, bank_name)`` directly, so the fetch
layer is out of scope here (it has its own unit tests in the agent).

``expect`` lists the products that must be found (each dict asserts only the fields it
names). ``forbid`` lists product-name fragments that must NOT appear — the bank-scoping
guard for aggregator/comparison pages that also list other banks' rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CASES_YAML = Path(__file__).resolve().parent.parent / "corpus" / "rate_cases.yaml"

RATE_FIELDS = frozenset(
    {"product_name", "rate_type", "margin_pct", "fixed_rate_pct", "fixed_years", "euribor_tenor"}
)


@dataclass(frozen=True)
class RateCase:
    id: str
    bank_name: str
    page: str
    expect: list[dict[str, Any]]  # products that must be extracted
    # product-specs that must NOT appear — another bank's row leaking through the
    # bank_name scope (value-based: a leaked row shares a generic name but a telltale rate).
    forbid: list[dict[str, Any]] = field(default_factory=list)
    # per-slice score breakdown (clean / aggregator / adversarial / injection / …).
    slice: str = "clean"


def load_rate_cases(path: Path = _CASES_YAML) -> list[RateCase]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases: list[RateCase] = []
    for row in data["cases"]:
        for prod in [*row.get("expect", []), *row.get("forbid", [])]:
            unknown = set(prod) - RATE_FIELDS
            if unknown:
                raise ValueError(f"{row['id']}: unknown product field(s) {sorted(unknown)}")
        cases.append(
            RateCase(
                id=row["id"],
                bank_name=row["bank_name"],
                page=row["page"],
                expect=list(row.get("expect", [])),
                forbid=list(row.get("forbid", [])),
                slice=row.get("slice", "clean"),
            )
        )
    return cases


def _print_counts() -> None:
    from collections import Counter

    cases = load_rate_cases()
    by_slice = Counter(c.slice for c in cases)
    by_bank = Counter(c.bank_name for c in cases)
    print(f"rate: {len(cases)} cases")
    print("  by slice: " + ", ".join(f"{k}={v}" for k, v in sorted(by_slice.items())))
    print("  by bank:  " + ", ".join(f"{k}={v}" for k, v in sorted(by_bank.items())))


if __name__ == "__main__":
    _print_counts()
