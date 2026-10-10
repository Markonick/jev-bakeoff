"""Offline proof of the calibration MATH — no live model.

We inject two fake scorers over a synthetic fixture:
  * a GOOD judge whose scores mirror the human labels → high correlation + kappa,
  * a BAD  judge whose scores are inverted/noisy → low/negative correlation + kappa.

If the agreement math is right, the good judge must read as agreeing and the bad one as
disagreeing. Deterministic and key-free (not marked ``eval``), so it runs in CI. The real
GEval scorer is exercised only by the ``python -m evals.judge_calibration`` demo when
Ollama is up — never here.
"""

from __future__ import annotations

from evals.judge_calibration import (
    CalibrationItem,
    calibrate,
    cohens_kappa,
    load_fixture,
    pearson,
    spearman,
)

# Synthetic labelled set with a clear spread so correlation is well-defined.
_ITEMS = [
    CalibrationItem(id="a", input="i", output="o", human_score=0.95),
    CalibrationItem(id="b", input="i", output="o", human_score=0.80),
    CalibrationItem(id="c", input="i", output="o", human_score=0.60),
    CalibrationItem(id="d", input="i", output="o", human_score=0.40),
    CalibrationItem(id="e", input="i", output="o", human_score=0.20),
    CalibrationItem(id="f", input="i", output="o", human_score=0.05),
]


def test_good_judge_agrees_with_human() -> None:
    # A judge that reproduces the human label (plus a tiny constant offset) should read as
    # strong agreement: high positive correlation, high kappa, high agreement rate.
    scores = iter([0.90, 0.82, 0.58, 0.42, 0.18, 0.10])

    def good(_i: str, _o: str) -> float:
        return next(scores)

    report = calibrate(_ITEMS, good, threshold=0.5)
    assert report.n == 6
    assert report.pearson is not None and report.pearson > 0.95
    assert report.spearman is not None and report.spearman > 0.95
    # pass/fail call @0.5: human = [1,1,1,0,0,0], judge = [1,1,1,0,0,0] → perfect.
    assert report.agreement_rate == 1.0
    assert report.cohens_kappa == 1.0


def test_bad_judge_disagrees_with_human() -> None:
    # A judge whose ranking is INVERTED vs the human should read as disagreement:
    # strongly negative correlation and non-positive kappa.
    inverted = iter([0.05, 0.20, 0.40, 0.60, 0.80, 0.95])

    def bad(_i: str, _o: str) -> float:
        return next(inverted)

    report = calibrate(_ITEMS, bad, threshold=0.5)
    assert report.pearson is not None and report.pearson < -0.95
    assert report.spearman is not None and report.spearman < -0.95
    # human call [1,1,1,0,0,0] vs judge call [0,0,0,1,1,1] → every item disagrees.
    assert report.agreement_rate == 0.0
    assert report.cohens_kappa is not None and report.cohens_kappa < 0.0


def test_worst_disagreements_are_ranked_first() -> None:
    scores = iter([0.10, 0.80, 0.60, 0.40, 0.20, 0.05])  # item 'a' is the big miss (0.95 vs 0.10)

    def judge(_i: str, _o: str) -> float:
        return next(scores)

    report = calibrate(_ITEMS, judge, threshold=0.5)
    worst = report.worst(1)
    assert worst[0].item.id == "a"
    assert worst[0].disagreement > 0.8


def test_stats_helpers_handle_degenerate_input() -> None:
    # Guard the "undefined" branches so a constant/short vector never crashes the report.
    assert pearson([1.0], [1.0]) is None
    assert pearson([1.0, 1.0, 1.0], [0.2, 0.5, 0.9]) is None  # no variance in x
    assert spearman([0.5], [0.5]) is None
    assert cohens_kappa([], []) is None
    # perfect binary agreement on a constant vector → degenerate pe==1, defined as 1.0.
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_fixture_loads_and_calibrates_with_a_stub_judge() -> None:
    # The shipped fixture must load and drive the pipeline with a stub judge (no model):
    # a judge that just echoes the human label yields perfect agreement, proving the
    # fixture rows are well-formed and end-to-end wiring works offline.
    items = load_fixture()
    assert len(items) >= 10  # a non-trivial corpus
    # Stub judge = echo the human label by walking the items in order (calibrate scores
    # them in order). Perfect echo → Pearson 1.0, proving fixture rows + wiring are sound.
    echo = iter([it.human_score for it in items])
    report = calibrate(items, lambda _i, _o: next(echo), threshold=0.7)
    assert report.pearson is not None and report.pearson > 0.99
    assert report.agreement_rate == 1.0
