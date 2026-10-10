"""Render deterministic Greek document PDFs for the synthetic eval corpus.

fpdf2 (pure Python, no system libs — unlike weasyprint) + an embedded DejaVu font
for real Greek glyphs, so the output has an extractable Greek text layer. PyMuPDF
handles the two edge fixtures (encrypted + blank) the same way the redactor does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF — encryption + blank page
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def _line(pdf: FPDF, height: float, text: str) -> None:
    # multi_cell leaves x at the right margin by default, which zeroes the width of
    # the next one — always return to the left margin on the following line.
    pdf.multi_cell(0, height, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


_FONT_DIR = Path(__file__).parent / "fonts"
_FONT = "DejaVu"


@dataclass
class DocSpec:
    """The content of one rendered document."""

    title: str
    subtitle: str = ""
    fields: list[tuple[str, str]] = field(default_factory=list)
    body: list[str] = field(default_factory=list)


def render_document(spec: DocSpec) -> bytes:
    """Render a spec to PDF bytes with a real Greek text layer."""
    pdf = FPDF()
    pdf.add_font(_FONT, "", str(_FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font(_FONT, "B", str(_FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    if spec.subtitle:
        pdf.set_font(_FONT, "", 10)
        _line(pdf, 6, spec.subtitle)
        pdf.ln(2)

    pdf.set_font(_FONT, "B", 15)
    _line(pdf, 9, spec.title)
    pdf.ln(4)

    pdf.set_font(_FONT, "", 11)
    for label, value in spec.fields:
        _line(pdf, 8, f"{label}: {value}")
    if spec.body:
        pdf.ln(3)
        pdf.set_font(_FONT, "", 11)
        for line in spec.body:
            _line(pdf, 7, line)

    return bytes(pdf.output())


def blank_pdf() -> bytes:
    """A valid PDF with a single empty page — readable, but no text to validate."""
    doc = fitz.open()
    doc.new_page()
    data: bytes = doc.tobytes()
    doc.close()
    return data


def encrypt(pdf: bytes, *, password: str) -> bytes:
    """Password-protect a PDF (the Tiresias-report failure mode)."""
    doc = fitz.open(stream=pdf, filetype="pdf")
    data: bytes = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw=password, user_pw=password
    )
    doc.close()
    return data
