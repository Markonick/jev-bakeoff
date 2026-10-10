"""TypeSafe Jev via OpenRouter — the System-1 structured-classifier path.

Separate from ``core/llm_client.py`` (the LiteLLM prompt→text port). Jev is NOT an
autoregressive LLM: it answers a set of TYPED questions (noul / choice / score) in one
parallel query and returns probabilities / chosen options — a bounded-decision classifier,
not a text generator. So it gets its own thin client, not a LiteLLM route.

Access is through **OpenRouter** using the native ``typesafe_sdk``: the SDK speaks
TypeSafe's classify protocol and OpenRouter proxies it under the ``typesafe/`` namespace
(base ``https://openrouter.ai/api``, auth ``OPENROUTER_API_KEY``).
See https://openrouter.ai/docs/guides/community/typesafe-sdk and
docs/strategy/jev-integration.md.

GATED: if the SDK isn't installed or ``OPENROUTER_API_KEY`` is unset, ``jev_available()``
is False and every caller / bake-off cell skips — exactly like the credless cloud routes.

API pinned against ``typesafe-sdk==0.7.1``: ``client.system_one(state, questions, model=…)``
where each question is an SDK ``Noul``/``Choice``/``Score`` object (Choice ``criteria`` =
{option: description}, Score ``criteria`` = ordered [levels]); the response carries
``.answers`` (each a Noul/Choice/ScoreAnswer), ``.model`` and ``.usage`` (in/out tokens).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import structlog
from typesafe_sdk import AsyncTypeSafeClient
from typesafe_sdk import Choice as _SdkChoice
from typesafe_sdk import Noul as _SdkNoul
from typesafe_sdk import Score as _SdkScore

from core.llm_client import _redact_text  # reuse the port's GDPR PII scrub on `state`

logger = structlog.get_logger(__name__)

# OpenRouter is the gateway. Overridable for a private/test gateway or the direct TypeSafe
# endpoint (https://api.typesafe.ai) if we ever hold a native key.
_BASE_URL = os.getenv("TYPESAFE_BASE_URL", "https://openrouter.ai/api")
_DEFAULT_MODEL = os.getenv("JEV_MODEL", "jev-latest")


def jev_available() -> bool:
    """True when Jev is callable — the SDK is a hard dep, so this is just the OpenRouter key
    (absent ⇒ every caller / bake-off cell skips, like the credless cloud routes)."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


# --- our own question value types (callers never import the SDK) ------------------------
@dataclass(frozen=True)
class Noul:
    """A yes/no question — Jev returns a probability in [0, 1]."""

    instructions: str


@dataclass(frozen=True)
class Choice:
    """Pick one option. ``criteria`` maps option name → what that option means."""

    instructions: str
    criteria: dict[str, str]


@dataclass(frozen=True)
class Score:
    """Rate against ORDERED levels (low → high), e.g. ["low", "medium", "high"]."""

    instructions: str
    levels: list[str]


JevQuestion = Noul | Choice | Score


def _to_sdk(q: JevQuestion) -> Any:
    """Map our value type onto the SDK's Noul/Choice/Score object."""
    if isinstance(q, Noul):
        return _SdkNoul(instructions=q.instructions)
    if isinstance(q, Choice):
        return _SdkChoice(instructions=q.instructions, criteria=q.criteria)
    return _SdkScore(instructions=q.instructions, criteria=list(q.levels))


@dataclass(frozen=True)
class JevResult:
    """Normalised, SDK-agnostic result. ``answers`` is keyed by question name → a plain dict
    (``model_dump`` of the SDK answer: a ``noul`` probability, a chosen ``choice`` +
    ``probabilities``, or a ``score``). ``model`` + token ``usage`` ride along for the board."""

    answers: dict[str, Any]
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def _dump(answer: Any) -> Any:
    return answer.model_dump() if hasattr(answer, "model_dump") else answer


async def classify(
    state: str,
    questions: dict[str, JevQuestion],
    *,
    model: str | None = None,
    redact: bool = True,
) -> JevResult:
    """Answer every ``question`` about ONE ``state`` in one Jev call (questions run in
    parallel server-side). ``state`` is PII-scrubbed by default (the same regex tokeniser the
    LLM port applies to traces — ΑΦΜ/ΑΜΚΑ/IBAN/email/phone). Raises if Jev is unavailable;
    callers gate on ``jev_available()`` first (or the bake-off skips)."""
    if not jev_available():
        raise RuntimeError("Jev unavailable — set OPENROUTER_API_KEY")
    client = _new_client()
    try:
        return await _run_one(client, state, questions, model=model, redact=redact)
    finally:
        await client.aclose()


def _new_client() -> AsyncTypeSafeClient:
    return AsyncTypeSafeClient(api_key=os.environ["OPENROUTER_API_KEY"], base_url=_BASE_URL)


async def _run_one(
    client: AsyncTypeSafeClient,
    state: str,
    questions: dict[str, JevQuestion],
    *,
    model: str | None,
    redact: bool,
) -> JevResult:
    """One ``system_one`` call on a shared client (reused across a batch)."""
    payload_state = _redact_text(state) if redact else state
    sdk_questions = {k: _to_sdk(v) for k, v in questions.items()}
    res = await client.system_one(payload_state, sdk_questions, model=model or _DEFAULT_MODEL)
    usage = getattr(res, "usage", None)
    return JevResult(
        answers={k: _dump(v) for k, v in (getattr(res, "answers", {}) or {}).items()},
        model=getattr(res, "model", None),
        input_tokens=getattr(usage, "input_tokens", None) if usage is not None else None,
        output_tokens=getattr(usage, "output_tokens", None) if usage is not None else None,
    )


@dataclass(frozen=True)
class JevBatchItem:
    """One unit of work in a batch: a caller ``key`` (e.g. case id), its ``state``, and the
    ``questions`` to answer for it."""

    key: str
    state: str
    questions: dict[str, JevQuestion]


async def classify_batch(
    items: list[JevBatchItem],
    *,
    model: str | None = None,
    redact: bool = True,
    concurrency: int = 16,
) -> dict[str, JevResult | None]:
    """Classify MANY states CONCURRENTLY — the fan-out Jev is built for. The SDK has no
    multi-state batch call, so we run ``concurrency`` ``system_one`` requests at a time over
    ONE shared async client (each client's own questions still run in parallel server-side).
    100 bounded decisions finish in ~the time of the slowest one, not 100× a single call.

    Returns ``{key: JevResult}``; an item that errors (or times out) maps to ``None`` and is
    logged — one bad case never sinks the batch. Tune ``concurrency`` to OpenRouter limits."""
    if not jev_available():
        raise RuntimeError("Jev unavailable — set OPENROUTER_API_KEY")
    sem = asyncio.Semaphore(concurrency)
    client = _new_client()

    async def _guarded(item: JevBatchItem) -> tuple[str, JevResult | None]:
        async with sem:
            try:
                return item.key, await _run_one(
                    client, item.state, item.questions, model=model, redact=redact
                )
            except Exception:
                logger.warning("jev.batch_item_failed", key=item.key, exc_info=True)
                return item.key, None

    try:
        pairs = await asyncio.gather(*(_guarded(it) for it in items))
    finally:
        await client.aclose()
    return dict(pairs)
