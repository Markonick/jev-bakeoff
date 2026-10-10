"""Recognizer unit tests — all synthetic PII, no spaCy model, fully offline.

The ΑΦΜ 094035012 is checksum-valid (computed by hand: mod-11 of the first 8
digits gives check digit 2); 094035013 flips the check digit and must be rejected.
"""

from redaction.recognizers import (
    EMAIL_ADDRESS,
    FOREIGN_ID,
    GR_AFM,
    GR_AMKA,
    GR_ID_CARD,
    GR_PASSPORT,
    GR_PHONE,
    MRZ,
    PERSON,
    AfmRecognizer,
    AmkaRecognizer,
    EmailLabelRecognizer,
    ForeignIdRecognizer,
    ForeignNameLabelRecognizer,
    GreekIdCardRecognizer,
    GreekNameLabelRecognizer,
    GreekPassportRecognizer,
    GreekPhoneRecognizer,
    LabeledForeignIdRecognizer,
    MrzRecognizer,
    SignatureBlockNameRecognizer,
)

_VALID_AFM = "094035012"
_INVALID_AFM = "094035013"


def _detect(recognizer, text: str):
    return recognizer.analyze(text, recognizer.supported_entities, nlp_artifacts=None)


def _matched(recognizer, text: str) -> set[str]:
    return {text[r.start : r.end] for r in _detect(recognizer, text)}


def test_afm_valid_checksum_detected() -> None:
    hits = _detect(AfmRecognizer(), f"ΑΦΜ: {_VALID_AFM} του φορολογουμένου")
    assert any(h.entity_type == GR_AFM for h in hits)
    # A checksum-valid match is promoted to a high score.
    assert max(h.score for h in hits) > 0.5


def test_afm_bad_checksum_rejected() -> None:
    assert _matched(AfmRecognizer(), f"ΑΦΜ: {_INVALID_AFM}") == set()


def test_amka_valid_date_prefix_detected() -> None:
    # 01/01/90 → 010190 + four digits; date prefix is plausible.
    assert _matched(AmkaRecognizer(), "ΑΜΚΑ 01019012345") == {"01019012345"}


def test_amka_impossible_date_rejected() -> None:
    # 45th day / 99th month → not a date.
    assert _matched(AmkaRecognizer(), "45990012345") == set()


def test_passport_detected() -> None:
    hits = _detect(GreekPassportRecognizer(), "Διαβατήριο AN1234567")
    assert {h.entity_type for h in hits} == {GR_PASSPORT}


def test_id_card_greek_and_latin_letters() -> None:
    assert _matched(GreekIdCardRecognizer(), "ΑΔΤ ΑΒ123456") == {"ΑΒ123456"}
    assert _matched(GreekIdCardRecognizer(), "ID AB-654321") == {"AB-654321"}


def test_phone_mobile_and_landline() -> None:
    hits = _matched(GreekPhoneRecognizer(), "τηλ 6912345678 ή 2101234567")
    assert "6912345678" in hits
    assert "2101234567" in hits


def test_phone_entity_type() -> None:
    hits = _detect(GreekPhoneRecognizer(), "+30 6912345678")
    assert all(h.entity_type == GR_PHONE for h in hits)


def test_phone_international_counterparty() -> None:
    # A foreign client's number on an intra-community invoice (was leaking).
    hits = _matched(GreekPhoneRecognizer(), "Τηλ: +420 734 388 794")
    assert "+420 734 388 794" in hits


def test_name_label_catches_all_caps_name() -> None:
    # All-caps, label-prefixed — exactly what the spaCy sm model misses.
    got = _matched(GreekNameLabelRecognizer(), "Ονοματεπώνυμο: ΠΑΠΑΔΟΠΟΥΛΟΣ ΝΙΚΟΛΑΟΣ")
    assert "ΠΑΠΑΔΟΠΟΥΛΟΣ ΝΙΚΟΛΑΟΣ" in got


def test_name_label_redacts_name_only_not_the_label() -> None:
    hits = _detect(GreekNameLabelRecognizer(), "Επώνυμο: Ιωάννου")
    assert len(hits) == 1
    assert hits[0].entity_type == PERSON
    # The label word itself is preserved (span starts at the name).
    assert "Ιωάννου" == "Επώνυμο: Ιωάννου"[hits[0].start : hits[0].end]


def test_name_label_no_false_positive_without_label() -> None:
    assert _matched(GreekNameLabelRecognizer(), "Ποσό φόρου 1234 ευρώ") == set()


def test_name_label_invoice_issuer_latin_name_across_newline() -> None:
    # myDATA invoice: the issuer label sits on its own line, the (Latin-script)
    # name below it — the exact leak the Greek NER model let through.
    got = _matched(GreekNameLabelRecognizer(), "ΕΚΔΟΤΗΣ:\nPappas Georgios\n7, ΤΚ 11524")
    assert "Pappas Georgios" in got


def test_name_label_invoice_counterparty() -> None:
    got = _matched(GreekNameLabelRecognizer(), "ΑΝΤΙΣΥΜΒΑΛΛΟΜΕΝΟΣ:\nJoinIT sro")
    assert "JoinIT sro" in got


def test_email_label_catches_wellformed_address() -> None:
    got = _matched(EmailLabelRecognizer(), "Email: test.user@example.com")
    assert "test.user@example.com" in got


def test_email_label_catches_linewrapped_fragment() -> None:
    # The exact leak: a hyphen-wrapped address with comma/newline noise between the
    # label and the value, which the well-formed EMAIL_ADDRESS regex can't match.
    got = _matched(EmailLabelRecognizer(), "Email: \n, \n, test.us-\n,Τηλ:")
    assert "test.us-" in got


def test_email_wrapped_head_orphaned_from_label() -> None:
    # As `get_text()` actually linearizes it: the hyphen-wrapped head sits next to
    # other emails, far from its `Email:` label — caught by the continuation anchor.
    text = "billing@example.com, test.us-\ner@example.com,Τηλ:"
    assert "test.us" in _matched(EmailLabelRecognizer(), text)


def test_email_wrapped_ignores_plain_hyphenated_word() -> None:
    # An ordinary end-of-line hyphenated word (no @-address continuation) is left alone.
    assert _matched(EmailLabelRecognizer(), "συμ-\nβαλλόμενος") == set()


def test_email_label_entity_type() -> None:
    hits = _detect(EmailLabelRecognizer(), "E-mail: a.b@c.com")
    assert {h.entity_type for h in hits} == {EMAIL_ADDRESS}


def test_amka_and_id_do_not_collide_on_type() -> None:
    # Sanity: each recognizer only emits its own entity type.
    assert {h.entity_type for h in _detect(AmkaRecognizer(), "01019012345")} == {GR_AMKA}
    assert {h.entity_type for h in _detect(GreekIdCardRecognizer(), "ΑΒ123456")} == {GR_ID_CARD}


# --- Foreign (non-Greek) recognizers: the real-specimen corpus is EN/DE/etc. ---


def test_foreign_name_label_mixed_case() -> None:
    got = _matched(ForeignNameLabelRecognizer(), "Name: Hans Schmidt")
    assert "Hans Schmidt" in got


def test_foreign_name_label_all_caps_on_tax_form() -> None:
    # Passport / tax forms render names ALL-CAPS — the Greek NER model misses them.
    got = _matched(ForeignNameLabelRecognizer(), "Taxpayer name: SCHMIDT HANS PETER")
    assert "SCHMIDT HANS PETER" in got


def test_foreign_name_label_german_holder() -> None:
    got = _matched(ForeignNameLabelRecognizer(), "Kontoinhaber: Nikolaos Papadopoulos")
    assert "Nikolaos Papadopoulos" in got


def test_foreign_name_label_redacts_name_only_not_label() -> None:
    hits = _detect(ForeignNameLabelRecognizer(), "Holder: Jones")
    assert len(hits) == 1
    assert hits[0].entity_type == PERSON
    assert "Jones" == "Holder: Jones"[hits[0].start : hits[0].end]


def test_foreign_name_label_no_false_positive_without_label() -> None:
    assert _matched(ForeignNameLabelRecognizer(), "Total tax due 1234 EUR") == set()


def test_mrz_name_field_detected() -> None:
    # The surname<<given-names field is the tell — never appears in ordinary text.
    hits = _matched(MrzRecognizer(), "P<GBRSCHMIDT<<HANS<PETER<<<<<<<<<<<<<<<<<<<<<<")
    assert any("SCHMIDT<<HANS" in h for h in hits)


def test_mrz_entity_type() -> None:
    hits = _detect(MrzRecognizer(), "MULLER<<ANNA<<<<<<")
    assert all(h.entity_type == MRZ for h in hits)


def test_us_ssn_detected() -> None:
    hits = _detect(ForeignIdRecognizer(), "SSN 123-45-6789")
    assert {"123-45-6789"} == {"SSN 123-45-6789"[h.start : h.end] for h in hits if h.score >= 0.5}
    assert all(h.entity_type == FOREIGN_ID for h in hits)


def test_uk_nino_detected() -> None:
    assert "AB123456C" in _matched(ForeignIdRecognizer(), "National Insurance AB123456C")


def test_labeled_foreign_id_german_and_utr() -> None:
    assert "12 345 678 901" in _matched(
        LabeledForeignIdRecognizer(), "Steuernummer: 12 345 678 901"
    )
    assert "1234567890" in _matched(LabeledForeignIdRecognizer(), "UTR: 1234567890")


def test_labeled_foreign_id_australian_tfn() -> None:
    # Australia was a chosen reference country — TFN is label-anchored, 9 digits.
    assert "123 456 789" in _matched(LabeledForeignIdRecognizer(), "TFN: 123 456 789")


def test_labeled_foreign_id_passport_number() -> None:
    assert "540123456" in _matched(LabeledForeignIdRecognizer(), "Passport No: 540123456")


# --- gov.gr / eIDAS digital-signature blocks: the signer name leaks past the ---
# --- ordinary form labels AND the spaCy NER model, so it needs its own anchor. ---


def test_signature_block_greek_govgr_all_caps() -> None:
    # The exact gov.gr footer stamp: all-caps name after the Greek phrase — missed by
    # both the Ονοματεπώνυμο label anchor and the spaCy sm model.
    got = _matched(
        SignatureBlockNameRecognizer(),
        "Ψηφιακά υπογεγραμμένο από ΠΑΠΑΔΟΠΟΥΛΟΣ ΝΙΚΟΛΑΟΣ",
    )
    assert "ΠΑΠΑΔΟΠΟΥΛΟΣ ΝΙΚΟΛΑΟΣ" in got


def test_signature_block_greek_name_on_next_line_near_timestamp() -> None:
    # The name renders below the phrase, with a timestamp underneath (footer panel).
    got = _matched(
        SignatureBlockNameRecognizer(),
        "Ψηφιακά υπογεγραμμένο από\nΓεώργιος Ιωάννου\n2024-01-15 10:22:00 EET",
    )
    assert "Γεώργιος Ιωάννου" in got


def test_signature_block_greek_unaccented() -> None:
    got = _matched(
        SignatureBlockNameRecognizer(),
        "Ψηφιακα υπογεγραμμενο απο Μαρια Παπαδοπουλου",
    )
    assert "Μαρια Παπαδοπουλου" in got


def test_signature_block_latin_eidas() -> None:
    got = _matched(SignatureBlockNameRecognizer(), "Digitally signed by Nikolaos Papadopoulos")
    assert "Nikolaos Papadopoulos" in got


def test_signature_block_signer_label() -> None:
    assert "Μαρία Παπαδοπούλου" in _matched(
        SignatureBlockNameRecognizer(), "Υπογράφων: Μαρία Παπαδοπούλου"
    )
    assert "Hans Schmidt" in _matched(SignatureBlockNameRecognizer(), "Signer: Hans Schmidt")


def test_signature_block_redacts_name_only_not_the_phrase() -> None:
    text = "Digitally signed by John Smith"
    hits = _detect(SignatureBlockNameRecognizer(), text)
    assert len(hits) == 1
    assert hits[0].entity_type == PERSON
    # The "Digitally signed by" phrase (and any gov.gr label) is preserved.
    assert "John Smith" == text[hits[0].start : hits[0].end]


def test_signature_block_no_false_positive_without_phrase() -> None:
    # Ordinary text mentioning a signature but not the stamp phrasing → no name span.
    assert _matched(SignatureBlockNameRecognizer(), "Η υπογραφή είναι απαραίτητη") == set()
