"""Word-level positional redaction for two-column Greek AADE tax forms.

The flat-text ``detect(text) → page.search_for(span)`` pass in :mod:`redaction.redactor`
handles structured identifiers (ΑΦΜ / ΑΜΚΑ / ID / phone) well, but **leaks free-text
names and addresses on E1-style forms**. Two reasons, both fatal to the flat approach:

1. **Case + accents.** The form renders labels ALL-CAPS and accent-stripped
   (``ΕΠΩΝΥΜΟ``, ``ΟΝΟΜΑ``); the small Greek NER model has ~0 recall on that, and the
   flat label recognizer uses title-case anchors (``Επώνυμο``) that don't match.
2. **Two columns.** E1 is a Υπόχρεου / ΜΣΣ (filer / spouse) table, so one label anchors
   **two** values sitting in separate page columns. Even when a name is detected in the
   linearised text, ``page.search_for("ΜΑΡΚΟΣ ΠΑΠΠΑ")`` finds nothing — those two words
   are not contiguous on the page.

This pass works directly on positioned words (``page.get_text("words")``): it finds
table rows whose first cell is a known personal-data label and returns the bounding
boxes of **every value word to the right**, across both columns and the address
continuation line (``ΒΡΑΧΝΟΥ 7,`` / ``11524 ΑΘΗΝΑ``). The redactor blacks out those
boxes with true PyMuPDF redaction. Bias matches the rest of the package: a *missed*
identifier is the leak, an over-redacted label word only costs the specimen a little
fidelity — so the anchors lean permissive and the human review gate is the backstop.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

# One extracted word: (x0, y0, x1, y1, text, block_no, line_no, word_no) — the shape
# PyMuPDF's ``page.get_text("words")`` returns. Only the first five fields are used.
Word = Sequence[Any]
Rect = tuple[float, float, float, float]


def _norm(word: str) -> str:
    """Fold a word to a case/accent-insensitive, punctuation-free key.

    ``Ή`` → ``Η``, ``Διεύθυνση`` → ``ΔΙΕΥΘΥΝΣΗ``, ``ΤΑΥΤΟΤΗΤΑΣ:`` → ``ΤΑΥΤΟΤΗΤΑΣ`` —
    so a label matches however the form cased or accented it.
    """
    decomposed = unicodedata.normalize("NFD", word)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return "".join(c for c in stripped.upper() if c.isalnum())


def _labels(*rows: str) -> list[tuple[str, ...]]:
    """Normalise a list of space-separated label strings into token tuples."""
    return [tuple(_norm(t) for t in row.split()) for row in rows]


# Row labels whose value(s) are a person's name — redact the value(s), keep the label.
_NAME_LABELS = _labels(
    "Ονοματεπώνυμο υπόχρεου",
    "Επώνυμο συζύγου",
    "Όνομα συζύγου",
    "Όνομα πατέρα",
    "Όνομα πατρός",
    "Όνομα μητέρας",
    "Όνομα μητρός",
    "Επώνυμο",
    "Ονοματεπώνυμο",
    "Όνομα",
    "Πατρώνυμο",
    "Μητρώνυμο",
    "Γένος",
)
# Row labels whose value is an address or birthplace — redact the whole value line,
# including the ΤΚ + city that wrap onto the next row.
_ADDRESS_LABELS = _labels(
    "Ταχυδρομική διεύθυνση",  # ΕΝΦΙΑ / ΠΡΑΞΗ ΔΙΟΙΚΗΤΙΚΟΥ ΠΡΟΣΔΙΟΡΙΣΜΟΥ — value in the right column
    "Διεύθυνση κατοικίας",
    "Τόπος γέννησης",
    "Διεύθυνση",
    "Οδός",
    "Δήμος",
    "Πόλη",
    "Περιοχή",
    "Τοποθεσία",
)
# Row labels whose value is a date of birth (confirmed redact 2026-07-13).
_DOB_LABELS = _labels(
    "Ημερομηνία γέννησης",
    "Ημ γέννησης",
    "Ημ/νία γέννησης",
)
# Quasi-identifiers that tie a person to a property — redact the number that follows.
_QUASI_LABELS = _labels(
    "Αρ παρ ρεύματος",  # electricity supply / meter number
)

# Longest label first so ``ΟΝΟΜΑ ΠΑΤΕΡΑ`` wins over the bare ``ΟΝΟΜΑ`` (and consumes
# ΠΑΤΕΡΑ as a label word rather than redacting it as a value).
_FLAT_LABELS: list[tuple[tuple[str, ...], str]] = sorted(
    [
        *((label, "name") for label in _NAME_LABELS),
        *((label, "address") for label in _ADDRESS_LABELS),
        *((label, "dob") for label in _DOB_LABELS),
        *((label, "quasi") for label in _QUASI_LABELS),
    ],
    key=lambda item: len(item[0]),
    reverse=True,
)

# Words that continue a multi-line label after the anchor (``ΔΙΕΥΘΥΝΣΗ ΚΑΤΟΙΚΙΑΣ Ή
# ΑΣΚΗΣΗΣ ΕΠΙΧΕΙΡΗΜΑΤΙΚΗΣ ΔΡΑΣΤΗΡΙΟΤΗΤΑΣ``) — consumed as label, never redacted as value.
_CONTINUATION = {
    _norm(w)
    for w in (
        "κατοικίας",
        "ή",
        "άσκησης",
        "επιχειρηματικής",
        "δραστηριότητας",
        "και",
        "επωνυμία",  # "ΕΠΩΝΥΜΟ Ή ΕΠΩΝΥΜΙΑ" (surname or company name) is all label
    )
}
# Empty string = a pure-punctuation token (``&`` in ``ΟΔΟΣ & ΑΡΙΘΜΟΣ``) — also skipped.
_CONTINUATION.add("")


def _match(tokens: list[str]) -> tuple[str, int] | None:
    """Longest label matched as a prefix of ``tokens`` → ``(kind, n_label_tokens)``."""
    return _match_at(tokens, 0)


def _match_at(tokens: list[str], i: int) -> tuple[str, int] | None:
    """Longest label starting at position ``i`` → ``(kind, n_label_tokens)``.

    Unlike a pure prefix match this lets a label sit *anywhere* in a visual row,
    past a leading field number (``101 ΕΠΩΝΥΜΟ …``) or a preceding field group
    (``… ΜΑΡΚΟΣ 102 ΟΝΟΜΑ …``) — the layout AADE uses on VAT / Φ2 forms.
    """
    rest = len(tokens) - i
    for label, kind in _FLAT_LABELS:
        n = len(label)
        if rest >= n and tuple(tokens[i : i + n]) == label:
            return kind, n
    return None


def _has_alpha(norm_tok: str) -> bool:
    """A normalised token carrying at least one letter (a name value, not a number)."""
    return any(c.isalpha() for c in norm_tok)


def _has_digit(norm_tok: str) -> bool:
    """A normalised token carrying at least one digit (a date / meter value)."""
    return any(c.isdigit() for c in norm_tok)


def _cluster_lines(words: list[Word], y_tol: float) -> list[list[Word]]:
    """Group words into visual rows by vertical centre, each sorted left→right.

    Table cells that share a row land in one line even when PyMuPDF assigns them to
    different blocks (which it does for multi-column layouts), so a label and both of
    its column values end up together.
    """
    lines: list[tuple[float, list[Word]]] = []
    for w in words:
        yc = (w[1] + w[3]) / 2.0
        for line in lines:
            if abs(line[0] - yc) <= y_tol:
                line[1].append(w)
                break
        else:
            lines.append((yc, [w]))
    for _, group in lines:
        group.sort(key=lambda w: w[0])
    lines.sort(key=lambda line: line[0])
    return [group for _, group in lines]


def field_value_rects(words: Sequence[Word], *, max_addr_cont_lines: int = 2) -> list[Rect]:
    """Bounding boxes of every personal-data *value* on a Greek AADE form.

    ``words`` is ``page.get_text("words")``. Returns the rectangles to redact for
    each name/address/DOB/quasi-id label, handling every layout AADE uses:

    * **Values to the right** on the same visual row — E1 Υπόχρεου / ΜΣΣ columns,
      plus up to ``max_addr_cont_lines`` address wrap lines (ΤΚ + city).
    * **Multiple label→value groups per row**, each starting past a field number —
      the VAT / Φ2 ``101 ΕΠΩΝΥΜΟ ΜΑΡΚΟΣ  102 ΟΝΟΜΑ ΝΙΚΟΛΑΟΣ`` layout.
    * **Values in the row below**, column-aligned under a header label — the E9
      ``ΕΠΩΝΥΜΟ … ΟΝΟΜΑ … / ΜΑΡΚΟΣ … ΝΙΚΟΛΑΟΣ …`` table.

    Label words themselves are never returned.
    """
    real = [w for w in words if str(w[4]).strip()]
    if not real:
        return []
    heights = sorted(w[3] - w[1] for w in real if w[3] > w[1])
    med_h = heights[len(heights) // 2] if heights else 1.0
    y_tol = med_h * 0.6
    x_tol = med_h  # ~a character width of column-alignment slack

    lines = _cluster_lines(real, y_tol)
    line_tokens = [[_norm(w[4]) for w in line] for line in lines]
    rects: list[Rect] = []

    for li, line in enumerate(lines):
        tokens = line_tokens[li]
        p = 0
        while p < len(tokens):
            match = _match_at(tokens, p)
            if match is None:
                p += 1
                continue
            kind, n = match
            # Swallow label-wrap words so the printed label survives, only values go.
            value_start = p + n
            while value_start < len(tokens) and tokens[value_start] in _CONTINUATION:
                value_start += 1
            # Same-row value(s) for this label group, up to the next label group.
            values, next_pos = _row_values(line, tokens, value_start, kind)
            if values:
                rects.extend((w[0], w[1], w[2], w[3]) for w in values)
                if kind == "address":
                    rects.extend(
                        _address_continuation(
                            lines, line_tokens, li, values, x_tol, max_addr_cont_lines
                        )
                    )
                p = next_pos
                continue
            # No same-row value → the value sits in the row *below*, column-aligned
            # under this header label (E9 table). Bounded left by the label and right
            # by the next label group on the header row.
            if kind in ("name", "dob"):
                rects.extend(
                    _column_below(
                        lines, line_tokens, li, line[p][0], tokens, value_start, kind, x_tol
                    )
                )
            p = value_start
    # De-duplicate: a wrapped address value can be reached twice.
    return list(dict.fromkeys(rects))


def _row_values(
    line: list[Word], tokens: list[str], start: int, kind: str
) -> tuple[list[Word], int]:
    """Value words to the right of a label on the same row, and the index to resume at.

    Collection stops at the next label group. For a *name* it also stops at a
    numeric token (a field number such as ``102``, never part of a Greek name);
    for a *dob* / *quasi* it keeps only numeric tokens (the date / meter reading).
    """
    values: list[Word] = []
    vi = start
    while vi < len(tokens):
        if _match_at(tokens, vi) is not None:
            break  # the next label group begins
        tok = tokens[vi]
        if kind == "name":
            if not _has_alpha(tok):
                break
            values.append(line[vi])
        elif kind in ("dob", "quasi"):
            if not _has_digit(tok):
                break
            values.append(line[vi])
        else:  # address — take everything up to the next label
            values.append(line[vi])
        vi += 1
    return values, vi


def _address_continuation(
    lines: list[list[Word]],
    line_tokens: list[list[str]],
    li: int,
    values: list[Word],
    x_tol: float,
    max_lines: int,
) -> list[Rect]:
    """Redact the ΤΚ + city that wrap onto the row(s) below an address value."""
    rects: list[Rect] = []
    x_min = min(w[0] for w in values)
    consumed = 0
    j = li + 1
    while j < len(lines) and consumed < max_lines:
        if _match(line_tokens[j]) is not None:
            break  # a new labelled row — stop the address
        cont = [w for w in lines[j] if w[0] >= x_min - x_tol]
        if not cont:
            break  # nothing in the value column — address ended
        rects.extend((w[0], w[1], w[2], w[3]) for w in cont)
        consumed += 1
        j += 1
    return rects


def _column_below(
    lines: list[list[Word]],
    line_tokens: list[list[str]],
    li: int,
    label_x0: float,
    tokens: list[str],
    value_start: int,
    kind: str,
    x_tol: float,
) -> list[Rect]:
    """Redact the value(s) in the row below, column-aligned under a header label.

    The column runs from the label's left edge to the start of the next label group
    on the header row (or the page edge). Words that are themselves labels — or, for
    a name column, non-alphabetic — are left alone.
    """
    if li + 1 >= len(lines):
        return []
    right: float | None = None
    q = value_start
    while q < len(tokens):
        if _match_at(tokens, q) is not None:
            right = lines[li][q][0]
            break
        q += 1

    below = lines[li + 1]
    below_tokens = line_tokens[li + 1]
    rects: list[Rect] = []
    for k, w in enumerate(below):
        if w[0] < label_x0 - x_tol:
            continue
        if right is not None and w[0] >= right - x_tol:
            continue
        if _match_at(below_tokens, k) is not None:
            continue  # a sub-label, not a value
        if kind == "name" and not _has_alpha(below_tokens[k]):
            continue
        rects.append((w[0], w[1], w[2], w[3]))
    return rects


def propagated_name_rects(words: Sequence[Word], seed_rects: Sequence[Rect]) -> list[Rect]:
    """Extra rects for every *other* page occurrence of an already-redacted name.

    A name confirmed as a labelled field value (in ``seed_rects``) is blacked out
    wherever it re-appears in free text — most importantly the gov.gr signature
    stamp's "signed by <name>" sentence, where the name is not in a labelled cell.
    Only name-like tokens propagate (alphabetic, ≥ 4 chars) so short scaffolding
    words and numbers never trigger a page-wide sweep.
    """
    real = [w for w in words if str(w[4]).strip()]
    seed = {(round(r[0], 1), round(r[1], 1)) for r in seed_rects}
    names = {
        _norm(str(w[4]))
        for w in real
        if (round(w[0], 1), round(w[1], 1)) in seed
        and len(_norm(str(w[4]))) >= 4
        and _norm(str(w[4])).isalpha()
    }
    if not names:
        return []
    extra: list[Rect] = []
    for w in real:
        if (round(w[0], 1), round(w[1], 1)) in seed:
            continue
        if _norm(str(w[4])) in names:
            extra.append((w[0], w[1], w[2], w[3]))
    return extra


# --- gov.gr digital-signature stamp ("Ψηφιακή Βεβαίωση Εγγράφου") --------------
# Real gov.gr-signed PDFs carry a fixed footer stamp printing the *signer's*
# identity — full name (under "Υπογραφή:"), patronymic, ΑΦΜ — plus a validation
# code and QR that resolve, on docs.gov.gr, to the original un-redacted document.
# None of that survives the passes above:
#   - the signer name sits on the line *below* the "Υπογραφή:" label, not to its
#     right, so field_value_rects (same-row values only) never reaches it;
#   - "Πατρώνυμο: <name>" shares a visual row with left-column body text, so the
#     prefix-only row matcher never sees the label mid-row.
# This pass anchors on the stamp header and redacts the signer values directly.
_STAMP_ANCHOR = tuple(_norm(t) for t in "Ψηφιακή Βεβαίωση Εγγράφου".split())
_STAMP_CODE_LABEL = _labels("Κωδικός εγγράφου")[0]
_SIG_LABEL = _norm("Υπογραφή")
# Labels inside the signer column whose value identifies the signer.
_STAMP_LABELS = {
    _norm(w)
    for w in (
        "Υπογραφή",
        "Πατρώνυμο",
        "Μητρώνυμο",
        "Όνομα",
        "Επώνυμο",
        "Ονοματεπώνυμο",
        "Γένος",
        "ΑΦΜ",
    )
}
# Structural words kept visible in the stamp: the labels themselves plus the
# "Ημ. Υπογραφής" date label (the signing datetime is not personal data).
_STAMP_KEEP = _STAMP_LABELS | {_norm(w) for w in ("Ημ", "Υπογραφής")}
# A signing date / time token (22/06/2026, 13:35:37) — kept for authenticity; a
# bare ΑΦΜ (121235247) has no separator so it does not match and is redacted.
_STAMP_DATE = re.compile(r"\d[:/.]\d")


def _yc(w: Word) -> float:
    return (float(w[1]) + float(w[3])) / 2.0


def _stamp_anchor_y(real: list[Word]) -> float | None:
    ns = [_norm(str(w[4])) for w in real]
    n = len(_STAMP_ANCHOR)
    for i in range(len(real) - n + 1):
        if tuple(ns[i : i + n]) == _STAMP_ANCHOR:
            return float(real[i][1])
    return None


def has_signature_stamp(words: Sequence[Word]) -> bool:
    """True when the page carries a gov.gr digital-signature stamp footer."""
    real = [w for w in words if str(w[4]).strip()]
    return _stamp_anchor_y(real) is not None


def signature_stamp_rects(words: Sequence[Word]) -> list[Rect]:
    """Bounding boxes of the signer's identity in a gov.gr signature stamp.

    ``words`` is ``page.get_text("words")``. Returns the rectangles to redact:
    the signer name (below "Υπογραφή:"), patronymic / other name values (to the
    right of their labels), the ΑΦΜ value, and the document validation code —
    everything that de-anonymises the stamp. Label words, the signing date/time
    and the page number are left intact. Empty when no stamp is present.
    """
    real = [w for w in words if str(w[4]).strip()]
    anchor_y = _stamp_anchor_y(real)
    if anchor_y is None:
        return []
    heights = sorted(w[3] - w[1] for w in real if w[3] > w[1])
    med_h = heights[len(heights) // 2] if heights else 10.0
    y_tol = med_h * 0.6
    stamp = [w for w in real if _yc(w) > anchor_y]

    def is_pii(w: Word) -> bool:
        return _norm(str(w[4])) not in _STAMP_KEEP and not _STAMP_DATE.search(str(w[4]))

    # Left edge of the signer column, from where the labels themselves sit — the
    # name printed below "Υπογραφή:" belongs to this column, left-column body text
    # (far to the left) does not.
    label_words = [w for w in stamp if _norm(str(w[4])) in _STAMP_LABELS]
    col_x = min((w[0] for w in label_words), default=0.0) - med_h

    rects: list[Rect] = []
    for label in stamp:
        if _norm(str(label[4])) not in _STAMP_LABELS:
            continue
        lyc = _yc(label)
        # Value(s) to the right of the label on the same visual row (left-column
        # body text sits far left of the label, so it is excluded by x0 >= x1).
        for w in stamp:
            if w is label:
                continue
            if abs(_yc(w) - lyc) <= y_tol and w[0] >= label[2] - 1 and is_pii(w):
                rects.append((w[0], w[1], w[2], w[3]))
        # The signer's full name is printed on the row *below* "Υπογραφή:".
        if _norm(str(label[4])) == _SIG_LABEL:
            below = [
                w for w in stamp if lyc < _yc(w) <= lyc + med_h * 1.8 and w[0] >= col_x
            ]
            if below:
                topy = min(_yc(w) for w in below)
                rects.extend(
                    (w[0], w[1], w[2], w[3])
                    for w in below
                    if abs(_yc(w) - topy) <= y_tol and is_pii(w)
                )

    # Document validation code ("Κωδικός εγγράφου: <token>") — with the QR it
    # resolves on docs.gov.gr back to the original signer and document.
    for i in range(len(stamp) - 1):
        if (
            _norm(str(stamp[i][4])) == _STAMP_CODE_LABEL[0]
            and _norm(str(stamp[i + 1][4])) == _STAMP_CODE_LABEL[1]
        ):
            label = stamp[i + 1]
            lyc = _yc(label)
            cands = [w for w in stamp if abs(_yc(w) - lyc) <= y_tol and w[0] > label[2]]
            if cands:
                v = min(cands, key=lambda w: w[0])
                rects.append((v[0], v[1], v[2], v[3]))
    return rects
