"""Word-level positional redaction tests — the two-column E1 leak fix.

The geometry tests build ``page.get_text("words")``-shaped tuples directly (no PDF, no
font), so they pin the column/label logic deterministically. The end-to-end test renders
a real two-column PDF with a Greek-capable builtin font (``china-s`` = Droid Sans
Fallback — plain Helvetica has no Greek glyphs, so its text layer is unextractable) and
asserts the names/address are gone from the text layer while the labels survive.
"""

import fitz
import pytest
from redaction.positional import _norm, field_value_rects
from redaction.redactor import PdfRedactor, PiiAnalyzer


# One PyMuPDF word: (x0, y0, x1, y1, text, block, line, word). Only the box + text matter.
def _w(x0: float, text: str, y: float = 100.0) -> tuple:
    return (x0, y, x0 + len(text) * 6.0, y + 10.0, text, 0, 0, 0)


def _xs(rects: list[tuple[float, float, float, float]]) -> list[float]:
    return sorted(r[0] for r in rects)


# --- normalisation --------------------------------------------------------------


def test_norm_folds_case_accents_and_punctuation() -> None:
    assert _norm("Διεύθυνση") == "ΔΙΕΥΘΥΝΣΗ"
    assert _norm("Ή") == "Η"
    assert _norm("ΤΑΥΤΟΤΗΤΑΣ:") == "ΤΑΥΤΟΤΗΤΑΣ"
    assert _norm("&") == ""


# --- two-column values ----------------------------------------------------------


def test_two_column_surname_row_redacts_both_names_not_label() -> None:
    row = [_w(72, "ΕΠΩΝΥΜΟ"), _w(220, "ΜΑΡΚΟΣ"), _w(360, "ΠΑΠΠΑ")]
    assert _xs(field_value_rects(row)) == [220.0, 360.0]


def test_lowercase_accented_label_still_anchors() -> None:
    # Same row as the form might render it in mixed case — folding must still match.
    row = [_w(72, "Επώνυμο"), _w(220, "Μάρκος"), _w(360, "Παππά")]
    assert _xs(field_value_rects(row)) == [220.0, 360.0]


def test_onoma_patera_consumes_second_label_word() -> None:
    # ΟΝΟΜΑ ΠΑΤΕΡΑ must win over bare ΟΝΟΜΑ so ΠΑΤΕΡΑ is treated as label, not value.
    row = [_w(72, "ΟΝΟΜΑ"), _w(140, "ΠΑΤΕΡΑ"), _w(240, "ΓΡΗΓΟΡΙΟΣ"), _w(360, "ΑΝΤΩΝΙΟΣ")]
    assert _xs(field_value_rects(row)) == [240.0, 360.0]


# --- addresses ------------------------------------------------------------------


def test_address_wrapped_label_and_continuation_line() -> None:
    # ΔΙΕΥΘΥΝΣΗ ΚΑΤΟΙΚΙΑΣ [values...] then a label-less ΤΚ+city continuation row.
    line1 = [
        _w(72, "ΔΙΕΥΘΥΝΣΗ", y=200),
        _w(150, "ΚΑΤΟΙΚΙΑΣ", y=200),
        _w(240, "ΒΡΑΧΝΟΥ", y=200),
        _w(320, "7,", y=200),
        _w(420, "ΛΕΣΒΟΥ", y=200),
        _w(500, "35,", y=200),
    ]
    line2 = [
        _w(72, "ΕΠΙΧΕΙΡΗΜΑΤΙΚΗΣ", y=212),  # label wrap — left column, must NOT be redacted
        _w(240, "11524", y=212),
        _w(320, "ΑΘΗΝΑ", y=212),
        _w(420, "11147", y=212),
        _w(500, "ΓΑΛΑΤΣΙ", y=212),
    ]
    rects = field_value_rects(line1 + line2)
    assert all(r[0] >= 240.0 for r in rects), "a label-column word was redacted"
    assert len(rects) == 8  # 4 value words on the label line + 4 on the continuation


def test_address_continuation_stops_at_next_label() -> None:
    line1 = [_w(72, "ΔΙΕΥΘΥΝΣΗ", y=200), _w(240, "ΒΡΑΧΝΟΥ", y=200)]
    next_label = [_w(72, "ΕΠΩΝΥΜΟ", y=212), _w(240, "ΜΑΡΚΟΣ", y=212)]
    rects = field_value_rects(line1 + next_label)
    # Both value words redacted (address value + the new labelled row's value)…
    assert _xs(rects) == [240.0, 240.0]
    # …but the continuation must not have swallowed the ΕΠΩΝΥΜΟ label itself.
    assert all(r[0] >= 240.0 for r in rects)


# --- other quasi-identifiers ----------------------------------------------------


def test_dob_value_redacted() -> None:
    row = [_w(72, "ΗΜΕΡΟΜΗΝΙΑ"), _w(160, "ΓΕΝΝΗΣΗΣ"), _w(260, "01/01/1980")]
    assert _xs(field_value_rects(row)) == [260.0]


def test_electricity_meter_number_redacted() -> None:
    row = [_w(72, "Αρ."), _w(110, "παρ."), _w(150, "ρεύματος"), _w(240, "703171787")]
    assert _xs(field_value_rects(row)) == [240.0]


# --- no false positives ---------------------------------------------------------


def test_unlabelled_row_is_left_alone() -> None:
    assert field_value_rects([_w(72, "ΠΟΣΟ"), _w(160, "ΦΟΡΟΥ"), _w(260, "1234")]) == []


def test_empty_word_list() -> None:
    assert field_value_rects([]) == []


# --- end-to-end through the redactor -------------------------------------------


def test_two_column_pdf_names_and_address_gone_from_text_layer() -> None:
    fn = "china-s"  # Droid Sans Fallback — has Greek glyphs, so the text layer extracts
    doc = fitz.open()
    page = doc.new_page()
    for y, label, v1, v2 in [
        (100, "ΕΠΩΝΥΜΟ", "ΜΑΡΚΟΣ", "ΠΑΠΠΑ"),
        (120, "ΟΝΟΜΑ", "ΝΙΚΟΛΑΟΣ", "ΕΥΣΤΑΘΙΑ"),
        (160, "ΔΙΕΥΘΥΝΣΗ", "ΒΡΑΧΝΟΥ", "ΛΕΣΒΟΥ"),
    ]:
        page.insert_text((72, y), label, fontsize=10, fontname=fn)
        page.insert_text((240, y), v1, fontsize=10, fontname=fn)
        page.insert_text((380, y), v2, fontsize=10, fontname=fn)
    page.insert_text((240, 172), "11524 ΑΘΗΝΑ", fontsize=10, fontname=fn)
    page.insert_text((380, 172), "11147 ΓΑΛΑΤΣΙ", fontsize=10, fontname=fn)
    pdf = doc.tobytes()
    doc.close()

    # use_ner=False: the positional pass runs regardless of the NER model, so CI covers it.
    result = PdfRedactor(PiiAnalyzer(use_ner=False)).redact_bytes(pdf)
    assert result.status == "redacted"
    assert result.pdf is not None
    assert result.entities["GR_FIELD"] >= 8

    out = fitz.open(stream=result.pdf, filetype="pdf")
    text = "".join(pg.get_text() for pg in out)
    out.close()

    for pii in ("ΜΑΡΚΟΣ", "ΠΑΠΠΑ", "ΝΙΚΟΛΑΟΣ", "ΕΥΣΤΑΘΙΑ", "ΒΡΑΧΝΟΥ", "ΛΕΣΒΟΥ", "ΑΘΗΝΑ", "ΓΑΛΑΤΣΙ"):
        assert pii not in text, f"leaked {pii}"
    for label in ("ΕΠΩΝΥΜΟ", "ΟΝΟΜΑ", "ΔΙΕΥΘΥΝΣΗ"):
        assert label in text, f"over-redacted the {label} label"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
