"""Reusable PII redaction for turning real Greek mortgage documents into
sanitised eval fixtures.

Presidio (detection: regex recognizers + Greek NER) + PyMuPDF (*true* redaction —
removes the underlying text, not a drawn-over box that leaves it extractable).

Neither DeepEval nor promptfoo ship this, so it lives here as reusable classes:
    - ``recognizers``: custom Greek recognizers Presidio lacks (ΑΦΜ/ΑΜΚΑ/ID/passport/phone)
    - ``redactor``:    ``PiiAnalyzer`` (detect) + ``PdfRedactor`` (redact / quarantine)
    - ``cli``:         ``python -m redaction.cli --in <dir> --out <dir>``
"""
