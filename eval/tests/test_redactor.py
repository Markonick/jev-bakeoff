"""Redactor tests — build tiny PDFs in-memory with PyMuPDF, redact, and assert the
PII is *actually gone from the text layer* (the whole point vs. a drawn-over box).

Regex-only analyzer (``use_ner=False``) so no spaCy model is required in CI.
"""

import fitz
import pytest
from redaction.redactor import PdfRedactor, PiiAnalyzer

_VALID_AFM = "094035012"


def _pdf_with(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def _text_of(pdf: bytes) -> str:
    doc = fitz.open(stream=pdf, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


@pytest.fixture
def redactor() -> PdfRedactor:
    return PdfRedactor(PiiAnalyzer(use_ner=False))


def test_afm_removed_from_text_layer(redactor: PdfRedactor) -> None:
    result = redactor.redact_bytes(_pdf_with(f"AFM {_VALID_AFM} end"))
    assert result.status == "redacted"
    assert result.pdf is not None
    assert result.entities["GR_AFM"] >= 1
    # True redaction: the digits are no longer extractable, not just visually covered.
    assert _VALID_AFM not in _text_of(result.pdf)


def test_email_redacted(redactor: PdfRedactor) -> None:
    result = redactor.redact_bytes(_pdf_with("contact test@example.com now"))
    assert result.status == "redacted"
    assert result.pdf is not None
    assert "test@example.com" not in _text_of(result.pdf)


def test_clean_pdf_passes_through_with_no_entities(redactor: PdfRedactor) -> None:
    result = redactor.redact_bytes(_pdf_with("just some ordinary words here"))
    assert result.status == "redacted"
    assert sum(result.entities.values()) == 0


def test_empty_pdf_quarantined_not_emitted(redactor: PdfRedactor) -> None:
    doc = fitz.open()
    doc.new_page()  # a page with no text at all
    empty = doc.tobytes()
    doc.close()

    result = redactor.redact_bytes(empty)
    assert result.status == "quarantined_no_text_layer"
    assert result.pdf is None  # never emit an unredacted image-only doc


def test_locked_pdf_quarantined(redactor: PdfRedactor) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), f"AFM {_VALID_AFM}", fontsize=11)
    locked = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()

    result = redactor.redact_bytes(locked)
    assert result.status == "quarantined_locked"
    assert result.pdf is None
