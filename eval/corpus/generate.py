"""Generate the synthetic corpus: one PDF per (case, variant) + a manifest.

    uv run python -m corpus.generate            # → corpus/generated/

Each manifest row carries the *requirement* the validator will be told to check
(document_type / tax_year / buyer_name) and the expected verdict — so the DeepEval
accuracy gate can assert against ``matches`` (and terminal reason) per variant. The
PDFs are gitignored (reproducible from seed); the generator + seed are committed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from corpus.renderers import DocSpec, blank_pdf, encrypt, render_document
from corpus.transforms import inject, multipage, scan

_HERE = Path(__file__).parent
_SEED = _HERE / "seed.yaml"
_OUT = _HERE / "generated"

# Map a build variant to the slice label the gates + bake-off break scores down by. A
# slice groups variants that stress the SAME failure mode (e.g. wrong_name/wrong_type/
# unrelated all probe adversarial sturdiness). Transform passes add their own slices
# (scanned / multipage / injection) below. Every case ends up with exactly one slice.
_VARIANT_SLICE = {
    "clean": "clean",
    "wrong_year": "wrong_year",
    "wrong_name": "adversarial",
    "wrong_type": "adversarial",
    "unrelated": "adversarial",
    "look_alike": "lookalike",
    "empty": "degenerate",
    "non_pdf": "degenerate",
    "password": "degenerate",
    "apostille_missing": "clean",  # advisory-only; still an accept, so a clean-family slice
}


def slice_for(variant: str) -> str:
    return _VARIANT_SLICE.get(variant, "clean")


# A non-PDF upload (someone attaches a photo) — validate_pdf must short-circuit to
# not_pdf on the missing %PDF magic, never call the model.
_NON_PDF_BYTES = b"\xff\xd8\xff\xe0not-a-pdf-this-is-a-jpeg-header"


@dataclass
class Expect:
    matches: bool | None
    terminal: str | None = None


@dataclass
class ManifestRow:
    id: str
    file: str
    document_type: str  # what the validator is told to check for
    label: str  # catalog label_en — feeds build_expectations
    ai_hint: str | None  # catalog ai_hint — feeds build_expectations
    tax_year: int | None
    buyer_name: str
    variant: str
    slice: str  # per-slice score breakdown (clean / scanned / lookalike / injection / …)
    expect: dict[str, Any]


def _fields(name: str, afm: str, year: int | None) -> list[tuple[str, str]]:
    # No-tax-year types (e.g. a service invoice) omit the Φορολογικό Έτος line.
    fields = [("Ονοματεπώνυμο", name), ("ΑΦΜ", afm)]
    if year is not None:
        fields.append(("Φορολογικό Έτος", str(year)))
    return fields


def _spec_for_type(dtype: str, name: str, afm: str, year: int | None, seed: dict) -> DocSpec:
    dt = seed["doc_types"][dtype]
    return DocSpec(
        title=dt["title"],
        subtitle=dt.get("subtitle", ""),
        fields=_fields(name, afm, year),
        body=list(dt.get("body", [])),
    )


def _other_type(dtype: str, seed: dict) -> str:
    # Deterministic "confusable" type for the wrong_type variant.
    types = [t for t in seed["doc_types"] if t != dtype]
    return types[0]


def build_variant(
    variant: str, case: dict, seed: dict
) -> tuple[bytes, str, int | None, str, Expect]:
    """Return (pdf_bytes, document_type, tax_year, buyer_name, expect) for one variant.

    The document_type / tax_year / buyer_name are the *requirement* (what the
    validator checks against) and stay fixed; the variant perturbs the rendered
    document relative to it.
    """
    people = seed["people"]
    person = people[case["person"]]
    dtype, year, name = case["type"], case.get("year"), person["name"]

    if variant == "clean":
        spec = _spec_for_type(dtype, name, person["afm"], year, seed)
        return render_document(spec), dtype, year, name, Expect(matches=True)

    if variant == "wrong_year":
        assert year is not None, "wrong_year variant requires a tax-year type"
        spec = _spec_for_type(dtype, name, person["afm"], year - 2, seed)
        return render_document(spec), dtype, year, name, Expect(matches=False)

    if variant == "wrong_name":
        other = people[(case["person"] + 1) % len(people)]
        spec = _spec_for_type(dtype, other["name"], other["afm"], year, seed)
        return render_document(spec), dtype, year, name, Expect(matches=False)

    if variant == "wrong_type":
        other = _other_type(dtype, seed)
        spec = _spec_for_type(other, name, person["afm"], year, seed)
        return render_document(spec), dtype, year, name, Expect(matches=False)

    if variant == "look_alike":
        la = seed["look_alike"]
        spec = DocSpec(
            title=la["title"],
            subtitle=la.get("subtitle", ""),
            fields=_fields(name, person["afm"], year),
            body=list(la.get("body", [])),
        )
        return render_document(spec), dtype, year, name, Expect(matches=False)

    if variant == "unrelated":
        un = seed["unrelated"]
        spec = DocSpec(title=un["title"], subtitle=un.get("subtitle", ""), body=list(un["body"]))
        return render_document(spec), dtype, year, name, Expect(matches=False)

    if variant == "empty":
        return blank_pdf(), dtype, year, name, Expect(matches=False)

    if variant == "non_pdf":
        return _NON_PDF_BYTES, dtype, year, name, Expect(matches=None, terminal="not_pdf")

    if variant == "password":
        spec = _spec_for_type(dtype, name, person["afm"], year, seed)
        locked = encrypt(render_document(spec), password="secret")
        return locked, dtype, year, name, Expect(matches=None, terminal="password_protected")

    raise ValueError(f"unknown variant: {variant}")


# --- Foreign (Phase 3) documents --------------------------------------------
# The same canonical type rendered per reference country in that country's
# language/currency, so the corpus exercises the production country-agnostic
# validator (agents/validation.py :: system_for) rather than a Greek layout.


def _fmt(text: str, year: int | None) -> str:
    return text.format(year=year) if "{year}" in text and year is not None else text


def _apostille_block(country: dict) -> list[str]:
    # A minimal, standardised apostille block (Hague 1961) — the advisory
    # apostille check looks for the header + numbered fields, never verifies it.
    return [
        "",
        country["apostille"],
        "1. Country / Pays: ...    2. Signed by: ...    6. Date: ...",
        "9. Seal/Stamp: ...    10. Signature: ...    (No.: 2026/044812)",
    ]


def _foreign_spec(
    dtype: str, country_code: str, name: str, year: int | None, seed: dict, *, apostille: bool
) -> DocSpec:
    ft = seed["foreign_doc_types"][dtype]
    country = seed["foreign_countries"][country_code]
    flavor = ft["countries"][country_code]
    fields = [(country["name_label"], name)]
    if year is not None:
        fields.append((country["year_label"], str(year)))
    body = [_fmt(line, year) for line in flavor.get("body", [])]
    if apostille and ft.get("cert"):
        body = body + _apostille_block(country)
    return DocSpec(
        title=_fmt(flavor["title"], year),
        subtitle=flavor.get("subtitle", ""),
        fields=fields,
        body=body,
    )


def _other_foreign_type(dtype: str, seed: dict) -> str:
    types = [t for t in seed["foreign_doc_types"] if t != dtype]
    return types[0]


def build_foreign_variant(
    variant: str, case: dict, seed: dict
) -> tuple[bytes, str, int | None, str, Expect]:
    """One foreign (country, variant) document + the requirement/verdict it tests."""
    people = seed["foreign_people"]
    person = people[case["person"]]
    dtype, year, name, cc = case["type"], case.get("year"), person["name"], case["country"]

    if variant == "clean":
        spec = _foreign_spec(dtype, cc, name, year, seed, apostille=True)
        return render_document(spec), dtype, year, name, Expect(matches=True)

    if variant == "apostille_missing":
        # Advisory-only: a cert-type doc with NO apostille must still ACCEPT — the
        # apostille never drives `matches` (agents/validation.py :: _APOSTILLE_CLAUSE).
        spec = _foreign_spec(dtype, cc, name, year, seed, apostille=False)
        return render_document(spec), dtype, year, name, Expect(matches=True)

    if variant == "wrong_year":
        assert year is not None, "wrong_year variant requires a tax-year type"
        spec = _foreign_spec(dtype, cc, name, year - 2, seed, apostille=True)
        return render_document(spec), dtype, year, name, Expect(matches=False)

    if variant == "wrong_name":
        other = people[(case["person"] + 1) % len(people)]
        spec = _foreign_spec(dtype, cc, other["name"], year, seed, apostille=True)
        return render_document(spec), dtype, year, name, Expect(matches=False)

    if variant == "wrong_type":
        other = _other_foreign_type(dtype, seed)
        other_cc = next(iter(seed["foreign_doc_types"][other]["countries"]))
        spec = _foreign_spec(other, other_cc, name, year or 2024, seed, apostille=True)
        return render_document(spec), dtype, year, name, Expect(matches=False)

    if variant == "look_alike":
        la = seed["foreign_look_alike"]
        spec = DocSpec(
            title=la["title"],
            subtitle=la.get("subtitle", ""),
            fields=[(seed["foreign_countries"][cc]["name_label"], name)],
            body=list(la.get("body", [])),
        )
        return render_document(spec), dtype, year, name, Expect(matches=False)

    raise ValueError(f"unknown foreign variant: {variant}")


# --- Input-diversity transforms -------------------------------------------
# Applied to the *clean* rendering of a case. They perturb only the rendering (scan /
# extra pages / an injected instruction), never what the document says, so the verdict
# stays matches=True. Each opted-in case (seed ``transforms:`` list) fans out to one row
# per transform, on its own slice, so the gates can score robustness separately.
_TRANSFORM_SLICE = {"scanned": "scanned", "multipage": "multipage", "injection": "injection"}


def _apply_transform(name: str, pdf: bytes, *, seed_int: int) -> bytes:
    if name == "scanned":
        return scan(pdf, seed=seed_int)
    if name == "multipage":
        return multipage(pdf, before=1, after=1)
    if name == "injection":
        return inject(pdf)
    raise ValueError(f"unknown transform: {name}")


def generate(out: Path = _OUT) -> list[ManifestRow]:
    seed = yaml.safe_load(_SEED.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    for case in seed["cases"]:
        catalog = seed["doc_types"][case["type"]]
        for variant in case["variants"]:
            pdf, dtype, year, name, expect = build_variant(variant, case, seed)
            case_id = f"{case['type']}_{variant}"
            ext = "bin" if variant == "non_pdf" else "pdf"
            filename = f"{case_id}.{ext}"
            (out / filename).write_bytes(pdf)
            rows.append(
                ManifestRow(
                    id=case_id,
                    file=filename,
                    document_type=dtype,
                    label=catalog["label_en"],
                    ai_hint=catalog.get("ai_hint"),
                    tax_year=year,
                    buyer_name=name,
                    variant=variant,
                    slice=slice_for(variant),
                    expect=asdict(expect),
                )
            )

        # Transform pass: perturb the clean rendering only (ground truth unchanged).
        for tname in case.get("transforms", []):
            clean_pdf, dtype, year, name, _ = build_variant("clean", case, seed)
            seed_int = sum(ord(c) for c in case["type"]) + (year or 0)
            perturbed = _apply_transform(tname, clean_pdf, seed_int=seed_int)
            case_id = f"{case['type']}_{tname}"
            filename = f"{case_id}.pdf"
            (out / filename).write_bytes(perturbed)
            rows.append(
                ManifestRow(
                    id=case_id,
                    file=filename,
                    document_type=dtype,
                    label=catalog["label_en"],
                    ai_hint=catalog.get("ai_hint"),
                    tax_year=year,
                    buyer_name=name,
                    variant=tname,
                    slice=_TRANSFORM_SLICE[tname],
                    expect=asdict(Expect(matches=True)),
                )
            )

    for case in seed.get("foreign_cases", []):
        catalog = seed["foreign_doc_types"][case["type"]]
        for variant in case["variants"]:
            pdf, dtype, year, name, expect = build_foreign_variant(variant, case, seed)
            # country in the id keeps multi-country rows for one type distinct.
            case_id = f"{case['type']}_{case['country']}_{variant}"
            filename = f"{case_id}.pdf"
            (out / filename).write_bytes(pdf)
            rows.append(
                ManifestRow(
                    id=case_id,
                    file=filename,
                    document_type=dtype,
                    label=catalog["label_en"],
                    ai_hint=catalog.get("ai_hint"),
                    tax_year=year,
                    buyer_name=name,
                    variant=variant,
                    slice=slice_for(variant),
                    expect=asdict(expect),
                )
            )

        for tname in case.get("transforms", []):
            clean_pdf, dtype, year, name, _ = build_foreign_variant("clean", case, seed)
            seed_int = sum(ord(c) for c in case["type"]) + (year or 0)
            perturbed = _apply_transform(tname, clean_pdf, seed_int=seed_int)
            case_id = f"{case['type']}_{case['country']}_{tname}"
            filename = f"{case_id}.pdf"
            (out / filename).write_bytes(perturbed)
            rows.append(
                ManifestRow(
                    id=case_id,
                    file=filename,
                    document_type=dtype,
                    label=catalog["label_en"],
                    ai_hint=catalog.get("ai_hint"),
                    tax_year=year,
                    buyer_name=name,
                    variant=tname,
                    slice=_TRANSFORM_SLICE[tname],
                    expect=asdict(Expect(matches=True)),
                )
            )

    manifest = out / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(asdict(r), ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic eval corpus.")
    parser.add_argument("--out", type=Path, default=_OUT)
    args = parser.parse_args(argv)
    rows = generate(args.out)
    by_variant: dict[str, int] = {}
    by_slice: dict[str, int] = {}
    for r in rows:
        by_variant[r.variant] = by_variant.get(r.variant, 0) + 1
        by_slice[r.slice] = by_slice.get(r.slice, 0) + 1
    print(f"Generated {len(rows)} documents → {args.out}")
    print("  variants: " + ", ".join(f"{k}={v}" for k, v in sorted(by_variant.items())))
    print("  slices:   " + ", ".join(f"{k}={v}" for k, v in sorted(by_slice.items())))
    print(f"  manifest: {args.out / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
