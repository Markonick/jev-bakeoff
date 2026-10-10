"""Judge-calibration harness — does our LLM-as-judge agree with HUMANS?

Our open-ended surfaces (portal-guide) are scored by an LLM-as-judge (a DeepEval
``GEval``), because there is no single golden answer to exact-match against. But a judge
is a *ruler of unknown accuracy* until we check it against ground truth. This harness
re-scores a set of (input, output, human_label) items with the LLM judge and reports how
well judge and human agree:

  * Pearson + Spearman correlation (do judge and human RANK/track the same way?)
  * Threshold-agreement rate + Cohen's kappa (do they make the same pass/fail CALL,
    corrected for chance?)
  * mean judge vs mean human (is the judge biased high/low?)
  * the worst per-item disagreements (where to look first)

Two input sources:
  (a) a LOCAL labelled JSONL fixture under ``evals/fixtures/`` — runs offline, is the
      demo + regression corpus, and needs no network.
  (b) an OPTIONAL Langfuse source (env-gated, ``JUDGE_CAL_SOURCE=langfuse``): pull traces
      and their HUMAN annotation scores (``source=ANNOTATION``) from the configured
      project and re-judge them.

The judge is pluggable. ``calibrate()`` takes any ``Scorer`` callable so the math is
testable with a fake scorer offline; the real judge (``geval_scorer``) reuses the shipped
portal ``GEval`` on the configurable ``local_judge`` (local Ollama by default, key-free).

Run:  ``make judge-calibration``   (or ``uv run python -m evals.judge_calibration``)
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------------------

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "portal_judge_labels.jsonl"

# A scorer re-scores one (input, output) into a judge score in the SAME scale as the human
# labels (0..1 here — GEval emits 0..1, and the fixture labels are 0..1). Injectable so the
# agreement math can be unit-tested with a fake scorer (no live model).
Scorer = Callable[[str, str], float]


@dataclass(frozen=True)
class CalibrationItem:
    """One labelled example: the judge input, the output under judgement, and the human's
    score. ``id`` is only for the disagreement report."""

    id: str
    input: str
    output: str
    human_score: float


@dataclass
class ScoredItem:
    item: CalibrationItem
    judge_score: float

    @property
    def disagreement(self) -> float:
        return abs(self.judge_score - self.item.human_score)


@dataclass
class CalibrationReport:
    scored: list[ScoredItem]
    threshold: float
    # agreement is measured on the pass/fail CALL derived at this threshold (score >= t).
    pearson: float | None
    spearman: float | None
    cohens_kappa: float | None
    agreement_rate: float
    mean_judge: float
    mean_human: float

    @property
    def n(self) -> int:
        return len(self.scored)

    def worst(self, k: int = 5) -> list[ScoredItem]:
        return sorted(self.scored, key=lambda s: s.disagreement, reverse=True)[:k]


# ---------------------------------------------------------------------------------------
# Agreement statistics (pure Python — no scipy in the eval venv; keep it dependency-free)
# ---------------------------------------------------------------------------------------


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Linear correlation. ``None`` when undefined (n < 2 or a constant vector — no
    variance to correlate). Returning None rather than 0.0 avoids a false "no agreement"."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def _rank(values: Sequence[float]) -> list[float]:
    """Fractional (average) ranks — ties share the mean of the ranks they span, which is
    what Spearman needs to stay correct under ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank across the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation = Pearson on the fractional ranks. ``None`` when undefined."""
    if len(xs) < 2:
        return None
    return pearson(_rank(xs), _rank(ys))


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Chance-corrected agreement for two binary raters. ``None`` when undefined.

    kappa = (po - pe) / (1 - pe). When raters agree perfectly *and* both are constant,
    pe == 1 → division by zero; we return 1.0 there (perfect, if degenerate) rather than
    None so a trivially-agreeing binary set still reads as agreement.
    """
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _agreement_rate(a: Sequence[int], b: Sequence[int]) -> float:
    if not a:
        return 1.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


# ---------------------------------------------------------------------------------------
# Core: re-score with the judge and compute agreement
# ---------------------------------------------------------------------------------------


def calibrate(
    items: Iterable[CalibrationItem], scorer: Scorer, *, threshold: float = 0.7
) -> CalibrationReport:
    """Re-score every item's output with ``scorer`` and measure judge↔human agreement.

    ``threshold`` defines the pass/fail CALL (score >= threshold) that the categorical
    metrics (agreement rate, Cohen's kappa) are computed on — it should match the
    threshold the surface actually gates on so the calibration reflects real decisions.
    """
    scored = [ScoredItem(item=it, judge_score=float(scorer(it.input, it.output))) for it in items]
    if not scored:
        raise ValueError("no calibration items — nothing to score")

    judge = [s.judge_score for s in scored]
    human = [s.item.human_score for s in scored]
    judge_call = [1 if v >= threshold else 0 for v in judge]
    human_call = [1 if v >= threshold else 0 for v in human]

    return CalibrationReport(
        scored=scored,
        threshold=threshold,
        pearson=pearson(judge, human),
        spearman=spearman(judge, human),
        cohens_kappa=cohens_kappa(judge_call, human_call),
        agreement_rate=_agreement_rate(judge_call, human_call),
        mean_judge=sum(judge) / len(judge),
        mean_human=sum(human) / len(human),
    )


def format_report(report: CalibrationReport) -> str:
    def _fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:+.3f}"

    lines = [
        "# Judge calibration (LLM-judge vs human)",
        "",
        f"- n items:            {report.n}",
        f"- Pearson r:          {_fmt(report.pearson)}",
        f"- Spearman rho:       {_fmt(report.spearman)}",
        f"- Cohen's kappa:      {_fmt(report.cohens_kappa)}  (pass/fail call @ "
        f"threshold {report.threshold:.2f})",
        f"- Agreement rate:     {report.agreement_rate:.0%}  (same pass/fail call)",
        f"- Mean judge score:   {report.mean_judge:.3f}",
        f"- Mean human score:   {report.mean_human:.3f}  "
        f"(judge bias {report.mean_judge - report.mean_human:+.3f})",
        "",
        "## Worst disagreements",
    ]
    for s in report.worst():
        lines.append(
            f"- `{s.item.id}`: judge {s.judge_score:.2f} vs human "
            f"{s.item.human_score:.2f}  (Δ {s.disagreement:.2f})"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------
# Source (a): local labelled JSONL fixture — offline, the demo + regression corpus
# ---------------------------------------------------------------------------------------


def load_fixture(path: Path = _FIXTURE) -> list[CalibrationItem]:
    """One JSON object per line: ``{"id", "input", "output", "human_score"}``. Blank lines
    and ``#`` comment lines are skipped so the fixture can be self-documenting."""
    items: list[CalibrationItem] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        items.append(
            CalibrationItem(
                id=str(row["id"]),
                input=str(row["input"]),
                output=str(row["output"]),
                human_score=float(row["human_score"]),
            )
        )
    if not items:
        raise ValueError(f"fixture {path} has no labelled items")
    return items


# ---------------------------------------------------------------------------------------
# Source (b): Langfuse — traces + their HUMAN annotation scores (env-gated, optional)
# ---------------------------------------------------------------------------------------


@dataclass
class LangfuseConfig:
    """Which annotation scores to treat as the human labels, and how to normalise them.

    Human annotations in Langfuse live on traces as scores with ``source=ANNOTATION``. We
    pull those, join them to their trace's input/output, and (optionally) rescale the raw
    annotation value onto the judge's 0..1 scale so the two are comparable.
    """

    score_name: str = field(default_factory=lambda: os.getenv("JUDGE_CAL_LF_SCORE", "quality"))
    limit: int = field(default_factory=lambda: int(os.getenv("JUDGE_CAL_LF_LIMIT", "50")))
    # divide the raw annotation value by this to map onto 0..1 (e.g. a 1..5 Likert → set 5).
    value_scale: float = field(default_factory=lambda: float(os.getenv("JUDGE_CAL_LF_SCALE", "1")))


def _stringify(value: Any) -> str:
    """Trace input/output can be a str, a dict, or None — normalise to the text the judge
    consumed. Dicts are dumped compactly and deterministically."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_from_langfuse(config: LangfuseConfig | None = None) -> list[CalibrationItem]:
    """Pull human annotation scores + their traces from the configured Langfuse project.

    API (Langfuse Python SDK 4.x, introspected against 4.15.1):
      * ``Langfuse().api.scores.get_many(source=ScoreSource.ANNOTATION, name=..., limit=...)``
        returns human annotation scores; each carries ``trace_id``, ``value``, ``name``.
      * ``Langfuse().api.trace.get(trace_id)`` returns the trace with ``input``/``output``.

    Needs ``LANGFUSE_HOST`` / ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` in the env
    (same as the experiment runners). Imported lazily so the fixture path stays offline.
    """
    from langfuse import Langfuse
    from langfuse.api.commons.types.score_source import ScoreSource

    cfg = config or LangfuseConfig()
    lf = Langfuse(timeout=int(os.getenv("LANGFUSE_TIMEOUT", "60")))

    scores = lf.api.scores.get_many(
        source=ScoreSource.ANNOTATION,
        name=cfg.score_name,
        limit=cfg.limit,
    )

    items: list[CalibrationItem] = []
    for score in scores.data:
        trace_id = getattr(score, "trace_id", None)
        raw_value = getattr(score, "value", None)
        if trace_id is None or raw_value is None:
            continue  # non-numeric (categorical/text) annotation, or a score with no trace
        trace = lf.api.trace.get(trace_id)
        items.append(
            CalibrationItem(
                id=str(getattr(score, "id", trace_id)),
                input=_stringify(getattr(trace, "input", None)),
                output=_stringify(getattr(trace, "output", None)),
                human_score=float(raw_value) / cfg.value_scale,
            )
        )
    if not items:
        raise ValueError(
            f"no Langfuse annotation scores named {cfg.score_name!r} with a numeric value "
            "and a joinable trace — annotate some traces first, or check JUDGE_CAL_LF_SCORE"
        )
    return items

