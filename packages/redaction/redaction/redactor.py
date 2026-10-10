"""Detect PII (Presidio) and apply *true* redaction to PDFs (PyMuPDF).

``PiiAnalyzer`` has two modes:
  - ``use_ner=False`` (default here for tests): custom Greek regex recognizers +
    Presidio's built-in email/IBAN recognizers, called directly — **no spaCy model
    needed**, so it runs in CI.
  - ``use_ner=True``: adds spaCy Greek NER (PERSON/LOCATION) for free-text names and
    addresses. Requires the ``el_core_news_*`` model (``make model`` / real runs only).

``PdfRedactor`` maps each detected span back to page rectangles and applies a
redaction annotation, which PyMuPDF resolves by **removing the underlying text** —
not merely painting a box over it. A PDF with no extractable text layer (scanned /
image-only) or a locked PDF is **quarantined**, never passed through unredacted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.predefined_recognizers import EmailRecognizer, IbanRecognizer

from redaction.positional import (
    field_value_rects,
    has_signature_stamp,
    propagated_name_rects,
    signature_stamp_rects,
)
from redaction.recognizers import foreign_recognizers, greek_recognizers

# Below this Presidio score a match is treated as noise and left alone.
_SCORE_THRESHOLD = 0.35
_SPACY_ENTITIES = ("PERSON", "LOCATION")


class PiiAnalyzer:
    """Detect PII spans in text. Regex-only by default; opt into spaCy NER for names."""

    def __init__(
        self,
        *,
        language: str = "el",
        use_ner: bool = False,
        model: str = "el_core_news_sm",
        score_threshold: float = _SCORE_THRESHOLD,
    ) -> None:
        self._language = language
        self._threshold = score_threshold
        self._recognizers: list[EntityRecognizer] = [
            *greek_recognizers(),
            *foreign_recognizers(),
            EmailRecognizer(supported_language=language),
            IbanRecognizer(supported_language=language),
        ]
        self._engine = self._build_engine(language, model) if use_ner else None

    def _build_engine(self, language: str, model: str):  # type: ignore[no-untyped-def]
        # Imported lazily so the regex-only path (and CI) never needs spaCy installed.
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_analyzer.predefined_recognizers import SpacyRecognizer

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": language, "model_name": model}],
                # Map the Greek model's NER labels onto Presidio entities explicitly —
                # the default mapping surfaces nothing for el_core_news.
                "ner_model_configuration": {
                    "model_to_presidio_entity_mapping": {
                        "PERSON": "PERSON",
                        "PER": "PERSON",
                        "LOC": "LOCATION",
                        "GPE": "LOCATION",
                        "ORG": "ORGANIZATION",
                    },
                    "low_score_entity_names": [],
                    "labels_to_ignore": ["O"],
                },
            }
        )
        registry = RecognizerRegistry()
        registry.supported_languages = [language]
        # The NLP recognizer turns spaCy's NER entities (PERSON/LOCATION) into results;
        # without it a custom registry surfaces regex hits only.
        registry.add_recognizer(SpacyRecognizer(supported_language=language))
        for recognizer in self._recognizers:
            registry.add_recognizer(recognizer)
        return AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            registry=registry,
            supported_languages=[language],
        )

    def detect(self, text: str) -> list[RecognizerResult]:
        if not text.strip():
            return []
        if self._engine is not None:
            entities = [
                *{r.supported_entities[0] for r in self._recognizers},
                *_SPACY_ENTITIES,
            ]
            results = self._engine.analyze(text=text, language=self._language, entities=entities)
        else:
            results = []
            for recognizer in self._recognizers:
                # Pattern/label recognizers accept no NLP artifacts; the base signature
                # types the arg as non-optional, hence the ignore.
                results.extend(
                    recognizer.analyze(text, recognizer.supported_entities, None)  # type: ignore[arg-type]
                )
        return [r for r in results if r.score >= self._threshold]


@dataclass
class RedactionResult:
    """Outcome for one PDF. ``pdf`` is the sanitised bytes when ``status == 'redacted'``,
    otherwise ``None`` (the file was quarantined and must be handled manually)."""

    status: str  # "redacted" | "quarantined_no_text_layer" | "quarantined_locked"
    pdf: bytes | None = None
    entities: Counter[str] = field(default_factory=Counter)

    @property
    def redacted(self) -> bool:
        return self.status == "redacted"


def _stamp_qr_rects(page: fitz.Page) -> list[fitz.Rect]:
    """QR-code image rectangles in a gov.gr signature stamp's footer band.

    The stamp's QR encodes the docs.gov.gr validation URL, which reveals the
    signer and the original un-redacted document — so it is redacted alongside
    the printed code. Scoped to the bottom band and to near-square images so the
    wide "Ψηφιακή Βεβαίωση Εγγράφου" logo band (aspect ≫ 1) is left alone.
    """
    band_top = page.rect.height * 0.7
    rects: list[fitz.Rect] = []
    for img in page.get_images(full=True):
        for rect in page.get_image_rects(img[0]):
            if rect.height <= 0 or rect.y0 <= band_top:
                continue
            if 0.7 <= rect.width / rect.height <= 1.4:
                rects.append(rect)
    return rects


class PdfRedactor:
    """Redact every PII span the analyzer finds, using true PyMuPDF redaction."""

    def __init__(self, analyzer: PiiAnalyzer, *, fill: tuple[float, float, float] = (0, 0, 0)):
        self._analyzer = analyzer
        self._fill = fill

    def redact_bytes(self, pdf: bytes) -> RedactionResult:
        doc = fitz.open(stream=pdf, filetype="pdf")
        try:
            if doc.needs_pass:
                return RedactionResult(status="quarantined_locked")

            counter: Counter[str] = Counter()
            total_text = 0
            for page in doc:
                text = page.get_text()
                total_text += len(text.strip())
                words = page.get_text("words")
                # Accumulate every rect before applying so the page-wide name sweep
                # below can see what the earlier passes already confirmed.
                marked: list[tuple[tuple[float, float, float, float], str]] = []

                for result in self._analyzer.detect(text):
                    span = text[result.start : result.end]
                    for rect in page.search_for(span):
                        marked.append(((rect.x0, rect.y0, rect.x1, rect.y1), result.entity_type))

                # Word-level positional pass: names/addresses/DOB on two-column tax
                # forms (E1/Φ2/E9) that the flat search_for can't locate across columns.
                for r in field_value_rects(words):
                    marked.append((r, "GR_FIELD"))

                # gov.gr digital-signature stamp: the signer's name/patronymic/ΑΦΜ
                # and the validation code/QR sit in a fixed footer the passes above
                # miss (name is below its label; patronymic shares a row with body
                # text). Redact the signer identity + the QR that resolves to it.
                for r in signature_stamp_rects(words):
                    marked.append((r, "GOVGR_SIGNATURE"))

                # Page-wide sweep: any name already confirmed as a labelled value is
                # blacked out wherever it re-appears in free text (e.g. the stamp's
                # "signed by <name>" sentence).
                seed = [r for r, _ in marked]
                for r in propagated_name_rects(words, seed):
                    marked.append((r, "GR_NAME"))

                if has_signature_stamp(words):
                    for rect in _stamp_qr_rects(page):
                        marked.append(((rect.x0, rect.y0, rect.x1, rect.y1), "GOVGR_QR"))

                for (x0, y0, x1, y1), entity in marked:
                    page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=self._fill)
                    counter[entity] += 1

                page.apply_redactions()

            # No extractable text → text redaction is blind to the content. Never
            # emit an "unredacted" file; flag it for OCR-based handling instead.
            if total_text == 0:
                return RedactionResult(status="quarantined_no_text_layer")

            return RedactionResult(status="redacted", pdf=doc.tobytes(), entities=counter)
        finally:
            doc.close()
