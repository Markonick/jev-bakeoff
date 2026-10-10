"""Custom Presidio recognizers for Greek identifiers Presidio does not ship.

Each is a ``PatternRecognizer`` so it works standalone (``.analyze(text, ...,
nlp_artifacts=None)``) without an NLP engine — which is what the unit tests use,
so they need no spaCy model. The ones with a real check (ΑΦΜ checksum, ΑΜΚΑ date)
override ``validate_result`` to promote a true match and drop false ones.

Redaction bias: a *missed* identifier is the leak, an over-redaction only costs a
fixture some fidelity — so patterns lean slightly permissive on purpose.
"""

from __future__ import annotations

import re

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

# Entity type names (used by the redactor + tests).
GR_AFM = "GR_AFM"
GR_AMKA = "GR_AMKA"
GR_ID_CARD = "GR_ID_CARD"
GR_PASSPORT = "GR_PASSPORT"
GR_PHONE = "GR_PHONE"
EMAIL_ADDRESS = "EMAIL_ADDRESS"  # shared with Presidio's built-in so results merge
PERSON = "PERSON"  # shared with spaCy NER so results merge

# Greek + Latin capital letters (gov docs mix visually-identical glyphs).
_GR_LATIN_UPPER = "Α-ΩA-Z"


class AfmRecognizer(PatternRecognizer):
    """Greek Tax Registration Number (ΑΦΜ) — 9 digits with a mod-11 checksum."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=GR_AFM,
            name="gr_afm",
            supported_language="el",
            patterns=[Pattern("afm-9-digits", r"\b\d{9}\b", 0.3)],
            context=["ΑΦΜ", "Α.Φ.Μ", "AFM", "ΑΦΜ:", "Αρ. Φορ. Μητρώου"],
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        if len(pattern_text) != 9 or not pattern_text.isdigit() or pattern_text == "0" * 9:
            return False
        digits = [int(c) for c in pattern_text]
        total = sum(d * (2 ** (8 - i)) for i, d in enumerate(digits[:8]))
        return (total % 11) % 10 == digits[8]


class AmkaRecognizer(PatternRecognizer):
    """Greek Social Security Number (ΑΜΚΑ) — 11 digits, first 6 are DDMMYY."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=GR_AMKA,
            name="gr_amka",
            supported_language="el",
            patterns=[Pattern("amka-11-digits", r"\b\d{11}\b", 0.3)],
            context=["ΑΜΚΑ", "AMKA", "Αρ. Μητρώου Κοιν. Ασφάλισης"],
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        if len(pattern_text) != 11 or not pattern_text.isdigit():
            return False
        day, month = int(pattern_text[0:2]), int(pattern_text[2:4])
        return 1 <= day <= 31 and 1 <= month <= 12


class GreekIdCardRecognizer(PatternRecognizer):
    """Greek ID card number — 1–2 (Greek/Latin) letters then 6 digits, e.g. ``ΑΒ123456``."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=GR_ID_CARD,
            name="gr_id_card",
            supported_language="el",
            patterns=[
                Pattern("id-card", rf"\b[{_GR_LATIN_UPPER}]{{1,2}}[\s\-]?\d{{6}}\b", 0.4),
            ],
            context=["ΑΔΤ", "Α.Δ.Τ", "Δελτίο Ταυτότητας", "ταυτότητα"],
        )


class GreekPassportRecognizer(PatternRecognizer):
    """Greek passport number — 2 letters + 7 digits, e.g. ``AN1234567``."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=GR_PASSPORT,
            name="gr_passport",
            supported_language="el",
            patterns=[Pattern("passport", rf"\b[{_GR_LATIN_UPPER}]{{2}}\d{{7}}\b", 0.5)],
            context=["διαβατήριο", "passport", "αρ. διαβατηρίου"],
        )


class GreekPhoneRecognizer(PatternRecognizer):
    """Phone numbers — Greek landline ``2XXXXXXXXX`` / mobile ``69XXXXXXXX`` (optional
    +30), plus any international ``+<cc> …`` number (e.g. a foreign counterparty's
    ``+420 734 388 794`` on an intra-community invoice)."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=GR_PHONE,
            name="gr_phone",
            supported_language="el",
            patterns=[
                Pattern("gr-phone", r"\b(\+30[\s\-]?)?(69\d{8}|2\d{9})\b", 0.4),
                # International: a leading +<country code> then grouped digits.
                Pattern("intl-phone", r"\+\d{1,3}(?:[\s\-]?\d{2,4}){2,5}\b", 0.4),
            ],
            context=["τηλ", "κινητό", "phone", "tel"],
        )


# Greek form labels that introduce a person/company name. Anchoring on these
# catches ALL-CAPS names (how tax forms render them), which the spaCy `sm` model
# misses — and works without any model, so it runs in CI.
_NAME_LABELS = (
    "Ονοματεπώνυμο",
    "Ονοματεπωνυμο",
    "Επώνυμο",
    "Επωνυμο",
    "Όνομα",
    "Ονομα",
    "Επωνυμία",
    "Επωνυμια",
    "Δικαιούχος",
    "Δικαιουχος",
    "Υπόχρεος",
    "Υποχρεος",
    "Ον/μο",
    # Invoice / receipt parties (myDATA layout): issuer + counterparty. The name
    # sits on the next line, which `\s*` already spans. Catches Latin-script names
    # (e.g. a freelancer billing a foreign client) that the Greek NER model misses.
    "ΕΚΔΟΤΗΣ",
    "Εκδότης",
    "Εκδοτης",
    "ΑΝΤΙΣΥΜΒΑΛΛΟΜΕΝΟΣ",
    "Αντισυμβαλλόμενος",
    "Αντισυμβαλλομενος",
)
_NAME_TOKEN = r"[A-Za-zΑ-Ωα-ωΆ-ώΪΫϊϋΐΰ][A-Za-zΑ-Ωα-ωΆ-ώΪΫϊϋΐΰ.'\-]+"
_LABEL_NAME_RE = re.compile(
    r"(?:" + "|".join(_NAME_LABELS) + r")\s*[:\-]?\s*"
    r"(?P<name>" + _NAME_TOKEN + r"(?:\s+" + _NAME_TOKEN + r"){0,2})"
)


class GreekNameLabelRecognizer(EntityRecognizer):
    """Redact the name that follows a Greek form label (``Ονοματεπώνυμο: …``).

    Emits ``PERSON`` for the name span only (not the label), so the form structure
    the validator reasons over survives. Complements spaCy NER, which misses the
    all-caps names typical of Greek tax documents.
    """

    def __init__(self) -> None:
        super().__init__(supported_entities=[PERSON], supported_language="el", name="gr_name_label")

    def load(self) -> None:  # no model to load
        return None

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        if PERSON not in entities:
            return []
        return [
            RecognizerResult(
                entity_type=PERSON, start=m.start("name"), end=m.end("name"), score=0.7
            )
            for m in _LABEL_NAME_RE.finditer(text)
        ]


# gov.gr / EU eIDAS digital-signature blocks. A document signed through gov.gr
# (or a qualified EU trust-service provider) carries a signature panel — often in a
# footer, margin, or a dedicated box near a timestamp/QR code — that names the
# signer, e.g. "Ψηφιακά υπογεγραμμένο από ΠΑΠΑΔΟΠΟΥΛΟΣ ΝΙΚΟΛΑΟΣ" or
# "Digitally signed by Nikolaos Papadopoulos". That name is PII and must be redacted.
#
# The name is NOT introduced by the ordinary form labels above (Ονοματεπώνυμο /
# Name), so those recognizers never fire on it, and the all-caps footer rendering is
# exactly what the spaCy sm NER model misses — hence a dedicated label anchor. We
# match both Greek and Latin script in one recognizer because the *phrase* is the
# discriminator, and gov.gr emits the Greek phrase while eIDAS PDFs emit the English.
#
# Only the signer's name span is redacted — the "gov.gr" label, the phrase itself,
# and any adjacent timestamp/serial are preserved so the block still reads as a
# signature stamp. The name may sit on the line below the phrase, which ``\s*`` spans.
_SIGNATURE_PHRASES = (
    # Greek (gov.gr) — accented and unaccented, με/χωρίς τόνους.
    r"Ψηφιακά\s+υπογεγραμμένο\s+από",
    r"Ψηφιακα\s+υπογεγραμμενο\s+απο",
    r"Υπογεγραμμένο\s+(?:ψηφιακά\s+)?από",
    r"Υπογεγραμμενο\s+(?:ψηφιακα\s+)?απο",
    r"Υπογράφων",
    r"Υπογραφων",
    r"Υπογράφουσα",
    r"Υπογραφουσα",
    # Latin (eIDAS / Adobe / DocuSign trust-service stamps).
    r"Digitally\s+signed\s+by",
    r"Signed\s+by",
    r"Signer",
    r"Signature\s+of",
)
# The signer name accepts Greek OR Latin tokens (a gov.gr stamp may transliterate).
_SIGNATURE_NAME_TOKEN = r"[A-Za-zΑ-Ωα-ωΆ-ώΪΫϊϋΐΰ][A-Za-zΑ-Ωα-ωΆ-ώΪΫϊϋΐΰ.'\-]+"
_SIGNATURE_NAME_RE = re.compile(
    r"(?:" + "|".join(_SIGNATURE_PHRASES) + r")"
    r"\s*[:\-]?\s*"
    r"(?P<name>" + _SIGNATURE_NAME_TOKEN + r"(?:\s+" + _SIGNATURE_NAME_TOKEN + r"){0,3})"
)


class SignatureBlockNameRecognizer(EntityRecognizer):
    """Redact the signer's name inside a gov.gr / eIDAS digital-signature block.

    Anchors on the signature phrasing ("Ψηφιακά υπογεγραμμένο από …",
    "Digitally signed by …", "Υπογράφων: …") and emits ``PERSON`` for the adjacent
    name span only. The phrase and the "gov.gr" label are preserved so the block
    still reads as a signature stamp — only the human name is removed.

    Exists because the signer name is not introduced by the ordinary form labels
    (Ονοματεπώνυμο / Name), so those recognizers miss it, and the all-caps footer
    rendering typical of these stamps is exactly what spaCy NER lets through.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entities=[PERSON], supported_language="el", name="signature_block_name"
        )

    def load(self) -> None:  # no model to load
        return None

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        if PERSON not in entities:
            return []
        return [
            RecognizerResult(
                entity_type=PERSON, start=m.start("name"), end=m.end("name"), score=0.7
            )
            for m in _SIGNATURE_NAME_RE.finditer(text)
        ]


# An address that follows an ``Email:`` label. Presidio's built-in EMAIL_ADDRESS
# needs a well-formed ``x@y.z``; a PDF that line-wraps the address (``test.us-``
# on one line, the rest on the next) defeats it. Anchoring on the label — and tolerating
# the punctuation/newline noise the layout injects between label and value — catches the
# fragment. The value stays email-shaped (has a dot, ``@``, or a trailing hyphen) so a bare
# following word isn't swept up.
_EMAIL_LABEL_RE = re.compile(
    r"(?:E-?mail|Ηλεκτρονικ[όο]\s+ταχυδρομείο)\s*[:\-]?[\s,;·]*"
    r"(?P<v>[A-Za-z0-9._%+\-]*(?:@[A-Za-z0-9.\-]*)?[A-Za-z0-9](?:\.[A-Za-z0-9._%+\-]+|-|@\S*))",
    re.IGNORECASE,
)

# The *head* of an address the PDF broke across lines with a hyphen: ``test.us-``
# on one line, ``er@example.com`` on the next. Presidio's built-in matches the well-formed
# tail but leaves the head; anchoring on the ``-<newline>…@domain`` continuation catches
# it. Very specific (hyphen + line break + an @-address), so it won't sweep up ordinary
# hyphenated words. Only the head span is redacted; the tail is the built-in's job.
_WRAPPED_EMAIL_RE = re.compile(
    r"(?P<v>[A-Za-z0-9._%+\-]*[A-Za-z0-9])-\n[A-Za-z0-9._%+\-]*@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)


class EmailLabelRecognizer(EntityRecognizer):
    """Redact addresses the well-formed EMAIL_ADDRESS regex misses: the value after an
    ``Email:`` label, and the head of a hyphen-wrapped address split across two lines."""

    def __init__(self) -> None:
        super().__init__(
            supported_entities=[EMAIL_ADDRESS], supported_language="el", name="email_label"
        )

    def load(self) -> None:  # no model to load
        return None

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        if EMAIL_ADDRESS not in entities:
            return []
        return [
            RecognizerResult(
                entity_type=EMAIL_ADDRESS, start=m.start("v"), end=m.end("v"), score=0.7
            )
            for pattern in (_EMAIL_LABEL_RE, _WRAPPED_EMAIL_RE)
            for m in pattern.finditer(text)
        ]


# --- Foreign (non-Greek) PII ------------------------------------------------
# The real-specimen corpus is English / German / etc. (see phase3-real-specimens.md),
# where the Greek label anchors and the Greek NER model above barely fire. These
# recognizers are language-agnostic (regex only, no model) so they run in CI too.
# All are registered as ``supported_language="el"`` to match the single-language
# engine the redactor builds — the patterns themselves are script-agnostic.

MRZ = "MRZ"  # passport machine-readable zone
FOREIGN_ID = "FOREIGN_ID"  # foreign national/tax identifier

# Latin-script equivalents of the Greek name labels (EN + DE/FR/ES/IT/NL). The label
# match is case-insensitive; the captured name must start uppercase (covers ALL-CAPS
# passport/tax-form names like ``SCHMIDT HANS`` and mixed-case ``Hans Schmidt``).
_FOREIGN_NAME_LABELS = (
    "Full name",
    "Name",
    "Surname",
    "Given names",
    "First name",
    "Last name",
    "Taxpayer name",
    "Taxpayer",
    "Account holder",
    "Accountholder",
    "Cardholder",
    "Card holder",
    "Holder",
    "Name of holder",
    "Applicant",
    "Beneficiary",
    "Payee",
    "Employee",
    "Shareholder",
    # German
    "Kontoinhaber",
    "Inhaber",
    "Steuerpflichtiger",
    "Steuerpflichtige",
    "Vorname",
    "Nachname",
    # French / Spanish / Dutch / Italian
    "Titulaire",
    "Prénom",
    "Nom",
    "Titular",
    "Nombre",
    "Naam",
    "Intestatario",
    "Nome",
)
_LATIN_NAME_TOKEN = r"[A-ZÀ-Þ][A-Za-zÀ-ÿ.'\-]+"
_FOREIGN_NAME_RE = re.compile(
    r"\b(?i:" + "|".join(re.escape(x) for x in _FOREIGN_NAME_LABELS) + r")\b"
    r"\s*[:\-]?\s*"
    r"(?P<name>" + _LATIN_NAME_TOKEN + r"(?:\s+" + _LATIN_NAME_TOKEN + r"){0,3})"
)

# Foreign tax/ID numbers introduced by a label — language-agnostic, so it covers the
# German IdNr/Steuernummer, UK UTR, Australian TFN, US SSN, and generic TIN/passport-no
# that the digit-only recognizers can't distinguish from ordinary numbers.
_ID_LABELS = (
    r"Steuer-?ID",
    r"Steueridentifikationsnummer",
    r"Identifikationsnummer",
    r"IdNr\.?",
    r"Steuernummer",
    r"Tax\s?File\s?Number",
    r"TFN",
    r"Tax\s?ID",
    r"TIN",
    r"UTR",
    r"National\s?Insurance(?:\s?(?:No|Number))?",
    r"NINO",
    r"Social\s?Security(?:\s?(?:No|Number|#))?",
    r"SSN",
    r"NIF",
    r"Passport\s?(?:No|Number)\.?",
)
_LABELED_ID_RE = re.compile(
    r"(?i:" + "|".join(_ID_LABELS) + r")\s*[:#\-]?\s*"
    r"(?P<v>[A-Za-z0-9][A-Za-z0-9\s\-]{4,18}[A-Za-z0-9])"
)


class ForeignNameLabelRecognizer(EntityRecognizer):
    """Redact the name after a Latin-script label (``Name: …`` / ``Inhaber: …``).

    The foreign counterpart of ``GreekNameLabelRecognizer`` — the real-specimen
    workflow runs ``make redact`` on English/German docs the Greek anchors miss.
    Emits ``PERSON`` for the name span only, so the form structure survives.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entities=[PERSON], supported_language="el", name="foreign_name_label"
        )

    def load(self) -> None:  # no model to load
        return None

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        if PERSON not in entities:
            return []
        return [
            RecognizerResult(
                entity_type=PERSON, start=m.start("name"), end=m.end("name"), score=0.6
            )
            for m in _FOREIGN_NAME_RE.finditer(text)
        ]


class MrzRecognizer(PatternRecognizer):
    """Passport machine-readable zone — the highest-density leak on an ID page
    (surname<<given-names + document number). The ``<<`` name filler is essentially
    never present in ordinary text, so the name-field pattern is high-precision."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=MRZ,
            name="mrz",
            supported_language="el",
            patterns=[
                # TD3 line 1: P<ISO3 + name field.
                Pattern("mrz-line1", r"P[A-Z<][A-Z]{3}[A-Z0-9<]{10,40}", 0.5),
                # The surname<<given-names field, wherever get_text() places it.
                Pattern("mrz-namefield", r"[A-Z0-9]{2,}<<[A-Z0-9<]{2,}", 0.6),
            ],
        )


class ForeignIdRecognizer(PatternRecognizer):
    """Distinctive foreign national identifiers that stand alone without a label:
    US SSN (``123-45-6789``), UK National Insurance number, US EIN (``12-3456789``)."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=FOREIGN_ID,
            name="foreign_id",
            supported_language="el",
            patterns=[
                Pattern("us-ssn", r"\b\d{3}-\d{2}-\d{4}\b", 0.5),
                Pattern(
                    "uk-nino",
                    r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s?\d{6}\s?[A-D]\b",
                    0.5,
                ),
                Pattern("us-ein", r"\b\d{2}-\d{7}\b", 0.4),
            ],
            context=["SSN", "Social Security", "National Insurance", "EIN", "Employer"],
        )


class LabeledForeignIdRecognizer(EntityRecognizer):
    """Redact the value after a foreign tax/ID label (``Steuernummer: …``, ``UTR: …``,
    ``TFN: …``, ``Passport No: …``) — language-agnostic, so one recognizer spans the
    per-country identifier zoo without a digit pattern for each."""

    def __init__(self) -> None:
        super().__init__(
            supported_entities=[FOREIGN_ID], supported_language="el", name="labeled_foreign_id"
        )

    def load(self) -> None:  # no model to load
        return None

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        if FOREIGN_ID not in entities:
            return []
        return [
            RecognizerResult(entity_type=FOREIGN_ID, start=m.start("v"), end=m.end("v"), score=0.7)
            for m in _LABELED_ID_RE.finditer(text)
        ]


def greek_recognizers() -> list[EntityRecognizer]:
    """The full set of custom Greek recognizers, ready to register or call directly."""
    return [
        AfmRecognizer(),
        AmkaRecognizer(),
        GreekIdCardRecognizer(),
        GreekPassportRecognizer(),
        GreekPhoneRecognizer(),
        GreekNameLabelRecognizer(),
        SignatureBlockNameRecognizer(),
        EmailLabelRecognizer(),
    ]


def foreign_recognizers() -> list[EntityRecognizer]:
    """Language-agnostic recognizers for the foreign real-specimen corpus (see
    ``docs/design/journey/phase3-real-specimens.md``)."""
    return [
        ForeignNameLabelRecognizer(),
        MrzRecognizer(),
        ForeignIdRecognizer(),
        LabeledForeignIdRecognizer(),
    ]
