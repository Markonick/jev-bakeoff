"""Corpus generator tests — render into a tmp dir and assert the fixtures carry the
traits each variant is meant to exercise (and that the manifest matches the seed).
No LLM calls here; the model-facing accuracy checks live in the DeepEval gate.
"""

import json
from pathlib import Path

import fitz
import pytest

from corpus.generate import generate


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("generated")
    generate(out)
    rows = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    return out, {r["id"]: r for r in rows}


def _text(path: Path) -> str:
    doc = fitz.open(path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def test_manifest_has_a_row_per_generated_file(corpus: tuple[Path, dict]) -> None:
    out, index = corpus
    pdfs = list(out.glob("*.pdf")) + list(out.glob("*.bin"))
    assert len(pdfs) == len(index)
    assert (out / "manifest.jsonl") in list(out.glob("*.jsonl"))


def test_clean_contains_type_keyword_and_expects_match(corpus: tuple[Path, dict]) -> None:
    out, index = corpus
    row = index["tax_assessment_clean"]
    assert row["expect"] == {"matches": True, "terminal": None}
    assert "ΕΚΚΑΘΑΡΙΣΤΙΚΟ" in _text(out / row["file"])
    assert row["buyer_name"] in _text(out / row["file"])


def test_look_alike_strips_the_identifying_keyword(corpus: tuple[Path, dict]) -> None:
    out, index = corpus
    row = index["tax_assessment_look_alike"]
    text = _text(out / row["file"])
    # Same layout, but the words that identify a tax assessment are gone.
    assert "ΕΚΚΑΘΑΡΙΣΤΙΚΟ" not in text
    assert "ΠΡΑΞΗ ΔΙΟΙΚΗΤΙΚΟΥ" not in text
    assert row["expect"]["matches"] is False


def test_wrong_year_doc_shows_a_different_year_than_the_requirement(
    corpus: tuple[Path, dict],
) -> None:
    out, index = corpus
    row = index["tax_assessment_wrong_year"]
    assert row["tax_year"] == 2024  # requirement
    text = _text(out / row["file"])
    assert "2022" in text and "2024" not in text  # document shows 2022
    assert row["expect"]["matches"] is False


def test_non_pdf_is_not_a_pdf_and_is_terminal(corpus: tuple[Path, dict]) -> None:
    out, index = corpus
    row = index["tax_assessment_non_pdf"]
    data = (out / row["file"]).read_bytes()
    assert data[:4] != b"%PDF"
    assert row["expect"] == {"matches": None, "terminal": "not_pdf"}


def test_password_variant_is_locked_and_terminal(corpus: tuple[Path, dict]) -> None:
    out, index = corpus
    row = index["tax_assessment_password"]
    doc = fitz.open(out / row["file"])
    try:
        assert doc.needs_pass
    finally:
        doc.close()
    assert row["expect"] == {"matches": None, "terminal": "password_protected"}


def test_empty_variant_has_no_text_layer(corpus: tuple[Path, dict]) -> None:
    out, index = corpus
    row = index["tax_assessment_empty"]
    assert _text(out / row["file"]).strip() == ""
    assert row["expect"]["matches"] is False


def test_every_row_carries_a_slice(corpus: tuple[Path, dict]) -> None:
    _, index = corpus
    assert all(row.get("slice") for row in index.values())
    # The known slice vocabulary — a typo (new unlabelled slice) fails loudly here.
    seen = {row["slice"] for row in index.values()}
    known = {
        "clean",
        "wrong_year",
        "adversarial",
        "lookalike",
        "degenerate",
        "scanned",
        "multipage",
        "injection",
    }
    assert seen <= known, f"unknown slice(s): {sorted(seen - known)}"


def test_scanned_transform_strips_text_but_stays_an_accept(corpus: tuple[Path, dict]) -> None:
    out, index = corpus
    row = index["tax_assessment_scanned"]
    assert row["slice"] == "scanned"
    # Rasterised → image-only PDF: no extractable text layer, but still a valid PDF that
    # ACCEPTS (ground truth unchanged — the scan is of a clean document).
    assert (out / row["file"]).read_bytes()[:4] == b"%PDF"
    assert _text(out / row["file"]).strip() == ""
    assert row["expect"]["matches"] is True


def test_multipage_transform_keeps_content_off_page_one(corpus: tuple[Path, dict]) -> None:
    out, index = corpus
    row = index["tax_assessment_multipage"]
    assert row["slice"] == "multipage"
    doc = fitz.open(out / row["file"])
    try:
        assert doc.page_count >= 3  # blank + real + blank
        assert doc[0].get_text().strip() == ""  # signal is not on page 1
    finally:
        doc.close()
    # The real document is still present + still an ACCEPT.
    assert "ΕΚΚΑΘΑΡΙΣΤΙΚΟ" in _text(out / row["file"])
    assert row["expect"]["matches"] is True


def test_injection_transform_embeds_marker_without_changing_verdict(
    corpus: tuple[Path, dict],
) -> None:
    from corpus.transforms import INJECTION_STRING

    out, index = corpus
    row = index["tax_assessment_injection"]
    assert row["slice"] == "injection"
    text = _text(out / row["file"])
    # The benign injection string is embedded, the real content survives, and the verdict
    # is unchanged — the guard must ignore an instruction found inside the document.
    assert INJECTION_STRING.split(".")[0].strip()[:20] in text
    assert "ΕΚΚΑΘΑΡΙΣΤΙΚΟ" in text
    assert row["expect"]["matches"] is True
