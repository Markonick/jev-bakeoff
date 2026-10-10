"""Provider-agnostic LLM port + ONE LiteLLM adapter (Ports & Adapters / Strategy).

The agent has the Anthropic SDK baked into every surface. This module abstracts that
behind one narrow ``LLMClient`` port so a surface can be pointed at any provider via the
model-map — the enabler for cutting cost by swapping in cheaper/open models (governed by
the eval gates) and for the key-free local eval (Ollama).

**Why LiteLLM, not hand-rolled adapters.** An earlier cut shipped four bespoke adapters
(Anthropic / OpenAI-compat / Gemini / Bedrock), each owning a provider's wire format, error
taxonomy, retries and PDF encoding — i.e. we owned the whole provider zoo and had to chase
every upstream API change forever. That is exactly LiteLLM's job (100+ providers, one
OpenAI-shaped interface, maintained upstream). So there is now a SINGLE ``LiteLLMAdapter``
behind the same thin port: callers still speak our domain ``LLMRequest``/``LLMResponse`` and
never import LiteLLM directly (the seam stays swappable), while LiteLLM handles the matrix.

The model-map (``resolve`` / ``get_llm``) turns a surface + env into a LiteLLM model string
(``anthropic/…`` · ``openai/…`` for any /v1 incl. Ollama · ``gemini/…`` · ``bedrock/…``) so
the eval bake-off points a surface at a candidate model with zero code change.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import litellm
import structlog

logger = structlog.get_logger(__name__)

# Cross-provider hygiene: drop params a given provider doesn't accept (so temperature /
# max_tokens quirks don't 400), and keep LiteLLM quiet + offline (no telemetry callouts).
litellm.drop_params = True
litellm.telemetry = False
litellm.suppress_debug_info = True

# Transient-failure backoff: LiteLLM retries (exponential) on 429 / 5xx / timeouts /
# connection-drops before surfacing the error. Free-tier Gemini 503s under burst and the
# odd upstream disconnect are exactly this class — retrying turns dropped bake-off cells
# into scored ones. Tunable via LLM_NUM_RETRIES (0 disables); default 4.
_NUM_RETRIES = int(os.getenv("LLM_NUM_RETRIES", "4"))


# --- normalised request / response -----------------------------------------------------
@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class PdfPart:
    data: bytes


Part = TextPart | PdfPart


@dataclass(frozen=True)
class LLMRequest:
    system: str
    content: list[Part]
    max_tokens: int
    temperature: float = 0.0
    cacheable_system: bool = True  # optimisation hint; no-ops where unsupported


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)


# --- normalised error taxonomy (so terminal/retry logic is provider-agnostic) ----------
class LLMError(Exception):
    """Base for all adapter errors."""


class ContentError(LLMError):
    """Permanent, terminal failure (bad/unreadable PDF, context-window, auth). Never
    succeeds on retry. Maps from LiteLLM ``BadRequest``/``ContextWindow``/``Auth`` errors."""


class RateLimited(LLMError):
    """Provider throttled us — back off + retry."""


class TransientError(LLMError):
    """Transient/network/server error — retry / let SQS redeliver."""


class UnsupportedModality(LLMError):
    """The request asked for input (e.g. a PDF) this provider/model can't take."""


@dataclass(frozen=True)
class Capabilities:
    supports_pdf: bool
    supports_caching: bool


@runtime_checkable
class LLMClient(Protocol):
    model: str
    caps: Capabilities

    async def generate(self, req: LLMRequest) -> LLMResponse: ...


# --- the single LiteLLM adapter --------------------------------------------------------
def _pdf_data_url(data: bytes) -> str:
    return "data:application/pdf;base64," + base64.standard_b64encode(data).decode("ascii")


def _pdf_to_image_blocks(
    data: bytes, *, max_pages: int = 5, dpi: int = 150
) -> list[dict[str, Any]]:
    """Rasterise a PDF's pages to PNG ``image_url`` blocks — how a PDF reaches an OpenAI-
    compatible VISION model (Ollama qwen2.5vl / llava) that takes images, not documents.
    Bounded to the first ``max_pages`` so the token/latency bill stays sane."""
    import fitz  # PyMuPDF — already a dependency for PDF handling

    blocks: list[dict[str, Any]] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in list(doc)[:max_pages]:
            png = page.get_pixmap(dpi=dpi).tobytes("png")
            b64 = base64.standard_b64encode(png).decode("ascii")
            blocks.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
    return blocks


# --- GDPR redaction for traced input/output --------------------------------------------
# Prod traces of Greek mortgage docs carry PII (ΑΦΜ, ΑΜΚΑ, IBAN, email, phone, names). The
# Langfuse SDK calls a `mask` callback on every serialized input/output before export, so we
# scrub PII centrally there — fast + dependency-free (regex only, no spaCy/presidio/model
# load): a name/afm keyed field is blanked wholesale, and free-text structured PII is
# token-replaced. `mask` must be TOTAL and never raise (a throwing mask drops the trace).

# Structured-PII regexes, applied in sequence. Order matters: longer/most-specific patterns
# run first so a shorter rule can't half-eat a longer token. Digit-length rules are guarded
# by boundaries so an 11-digit ΑΜΚΑ / 10-digit phone isn't chopped by the 9-digit ΑΦΜ rule
# (and, critically, the 10-digit phone rule can't match the last 10 digits of an 11-digit
# ΑΜΚΑ — its leading boundary rejects a match that starts mid digit-run).
_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bGR\d{25}\b", re.IGNORECASE), "«IBAN»"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "«EMAIL»"),
    (re.compile(r"\b\d{11}\b"), "«AMKA»"),
    (re.compile(r"(?<!\d)(?:\+?30\s?)?\d{10}\b"), "«PHONE»"),
    (re.compile(r"\b\d{9}\b"), "«AFM»"),
]

# Keys whose VALUE is PII regardless of type — blanked wholesale (not token-scanned).
_PII_KEYS = frozenset(
    {
        "afm",
        "tax_id",
        "taxid",
        "amka",
        "name",
        "full_name",
        "first_name",
        "last_name",
        "fathersname",
        "mothersname",
        "iban",
        "account",
        "address",
        "phone",
        "telephone",
        "email",
        "dob",
        "date_of_birth",
        "birth_date",
    }
)


def _redact_text(text: str) -> str:
    """Replace structured Greek PII in free text with typed tokens (`«AFM»`, `«IBAN»`, …)."""
    for pattern, token in _REDACTIONS:
        text = pattern.sub(token, text)
    return text


def _trace_mask(*, data: Any, **_: Any) -> Any:
    """Langfuse mask callback — recursively scrub PII from a serialized input/output.

    Total + best-effort: any failure returns a sentinel rather than propagating (a mask that
    raises would drop the trace or, worse, ship the un-masked payload). Signature matches the
    SDK's ``MaskFunction`` protocol — ``data`` keyword-only (called as ``mask(data=…)``) plus
    a ``**kwargs`` sink for forward-compat.
    """
    try:
        return _mask_value(data)
    except Exception:  # noqa: BLE001
        return "«mask-error»"


def _mask_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, val in value.items():
            if isinstance(key, str) and key.lower() in _PII_KEYS:
                out[key] = "«REDACTED»"
            else:
                out[key] = _mask_value(val)
        return out
    if isinstance(value, (list, tuple)):
        return [_mask_value(item) for item in value]
    return value  # int / float / bool / None and other scalars pass through untouched


# --- prod/dev tracing (Langfuse v4 SDK, directly) --------------------------------------
# NOT via LiteLLM's langfuse callback: the classic one targets the v2 SDK
# (`langfuse.version`) and the `langfuse_otel` one runs a separate OTEL pipeline that
# doesn't export in our runtime. The v4 SDK's own observation API exports reliably
# (same path the eval harness uses). Gated on LANGFUSE_TRACING=1 so tests/local don't ship.
def _tracer() -> Any:
    """The Langfuse client for tracing, or None when off/unavailable. Never raises — tracing
    must not break a generation.

    When on, the singleton is (re)configured with a PII ``mask`` and an ``environment`` tag
    (``LANGFUSE_TRACING_ENVIRONMENT``, default ``dev``) so prod traces are GDPR-scrubbed and
    kept out of the eval project's data. host/keys are still read from the environment by the
    SDK (``LANGFUSE_HOST`` / ``_PUBLIC_KEY`` / ``_SECRET_KEY``)."""
    if os.getenv("LANGFUSE_TRACING") != "1":
        return None
    try:
        from langfuse import Langfuse, get_client

        Langfuse(
            mask=_trace_mask,
            environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "dev"),
        )
        return get_client()  # the configured singleton
    except Exception:  # noqa: BLE001
        return None


def _trace_input(req: LLMRequest) -> dict[str, Any]:
    """A trace-safe view — never log raw PDF bytes (huge); record a marker instead."""
    parts = [
        p.text if isinstance(p, TextPart) else f"[pdf {len(p.data)} bytes]" for p in req.content
    ]
    return {"system": req.system, "content": parts}


class LiteLLMAdapter:
    """One adapter over ``litellm.acompletion`` — serves every provider. ``model`` is a
    LiteLLM model string (``anthropic/…`` · ``openai/…`` · ``gemini/…`` · ``bedrock/…``);
    ``extra`` carries per-route kwargs (``api_base`` / ``api_key`` / ``aws_region_name``).

    A PDF part is delivered per route: a native-document provider gets an OpenAI ``file``
    block (LiteLLM translates to each provider's format); a rasterising vision route
    (``rasterize_pdf=True`` — OpenAI-compat + a vision model) gets per-page ``image_url``
    blocks; a text-only route (``supports_pdf=False``) raises ``UnsupportedModality`` rather
    than silently dropping the document. When tracing is on, each call is a Langfuse
    generation named by ``surface`` (so online-eval + alerts can filter per-surface)."""

    def __init__(
        self,
        model: str,
        *,
        caps: Capabilities,
        surface: str | None = None,
        rasterize_pdf: bool = False,
        **extra: Any,
    ) -> None:
        self.model = model
        self.caps = caps
        self.surface = surface
        self._rasterize_pdf = rasterize_pdf
        self._extra = extra

    def _system_message(self, req: LLMRequest) -> dict[str, Any] | None:
        # A blank system prompt means "no system message" — some surfaces legitimately
        # send none. Emitting an empty content block (esp. with cache_control) is a hard
        # 400 on Anthropic, so drop it entirely rather than forward an empty string.
        if not req.system.strip():
            return None
        # Anthropic prompt caching: mark the system block ephemeral (LiteLLM forwards
        # cache_control for caching-capable providers; ignored elsewhere).
        if req.cacheable_system and self.caps.supports_caching:
            return {
                "role": "system",
                "content": [
                    {"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}
                ],
            }
        return {"role": "system", "content": req.system}

    def _user_message(self, req: LLMRequest) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = []
        for part in req.content:
            if isinstance(part, TextPart):
                blocks.append({"type": "text", "text": part.text})
            elif not self.caps.supports_pdf:
                raise UnsupportedModality(f"{self.model} cannot read a PDF part")
            elif self._rasterize_pdf:
                blocks.extend(_pdf_to_image_blocks(part.data))
            else:
                blocks.append({"type": "file", "file": {"file_data": _pdf_data_url(part.data)}})
        return {"role": "user", "content": blocks}

    def messages(self, req: LLMRequest) -> list[dict[str, Any]]:
        sys = self._system_message(req)
        return [sys, self._user_message(req)] if sys is not None else [self._user_message(req)]

    async def _acompletion(self, req: LLMRequest) -> LLMResponse:
        try:
            resp = await litellm.acompletion(
                model=self.model,
                messages=self.messages(req),
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                num_retries=_NUM_RETRIES,
                **self._extra,
            )
        except litellm.exceptions.RateLimitError as exc:
            raise RateLimited(str(exc)) from exc
        except (
            litellm.exceptions.BadRequestError,
            litellm.exceptions.ContextWindowExceededError,
            litellm.exceptions.ContentPolicyViolationError,
            litellm.exceptions.UnprocessableEntityError,
            litellm.exceptions.AuthenticationError,
            litellm.exceptions.NotFoundError,
        ) as exc:
            raise ContentError(str(exc)) from exc
        except litellm.exceptions.APIError as exc:  # timeouts, 5xx, service unavailable
            raise TransientError(str(exc)) from exc
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            ),
        )

    async def generate(self, req: LLMRequest) -> LLMResponse:
        """The call, optionally wrapped in a Langfuse generation (prod/dev). The trace is
        best-effort — a tracing hiccup must never fail the underlying generation."""
        lf = _tracer()
        if lf is None:
            return await self._acompletion(req)
        with lf.start_as_current_observation(
            name=self.surface or "llm.generate",
            as_type="generation",
            model=self.model,
            input=_trace_input(req),
        ) as gen:
            resp = await self._acompletion(req)
            gen.update(
                output=resp.text,
                usage_details={
                    "input": resp.usage.input_tokens or 0,
                    "output": resp.usage.output_tokens or 0,
                },
            )
            return resp


# --- primary → secondary fallback -------------------------------------------------------
class FallbackLLMClient:
    """Wraps a ``primary`` and a ``secondary`` ``LLMClient`` for the "Haiku primary, local
    secondary" (dev) / cloud-secondary (prod) pattern: a RETRYABLE primary failure
    (``RateLimited`` / ``TransientError``) falls through to the secondary. Terminal failures
    (``ContentError`` / ``UnsupportedModality``) propagate — another model won't fix a bad
    PDF or an unsupported modality. Each adapter opens its OWN Langfuse observation, so a
    fallback simply yields a second one (no tracing logic lives here)."""

    def __init__(self, primary: LLMClient, secondary: LLMClient) -> None:
        self.primary = primary
        self.secondary = secondary
        # Present the primary's identity as this client's (the shipped route).
        self.model = primary.model
        self.caps = primary.caps

    async def generate(self, req: LLMRequest) -> LLMResponse:
        try:
            return await self.primary.generate(req)
        except (RateLimited, TransientError) as exc:
            logger.warning(
                "llm.fallback",
                surface=getattr(self.primary, "surface", None),
                primary=self.primary.model,
                secondary=self.secondary.model,
                error=str(exc),
            )
            return await self.secondary.generate(req)


# --- model-map factory (Strategy: pick provider+model per surface) ---------------------
# The per-surface model strategy (production defaults + bake-off candidates) is the app's
# single source of truth in core.model_registry — NOT duplicated here. Override per surface
# with env: LLM_<SURFACE>_PROVIDER + LLM_<SURFACE>_MODEL (e.g. LLM_RATE_MODEL) — how the eval
# bake-off / key-free local run points a surface at a candidate model without a code change.

# Per-provider LiteLLM prefix + capabilities. openai_compat covers any OpenAI /v1 endpoint
# (Groq / DeepInfra / vLLM / OpenRouter / Ollama) — text-only unless the route is vision.
_PROVIDER_PREFIX = {
    "anthropic": "anthropic",
    "openai_compat": "openai",
    "gemini": "gemini",
    "bedrock": "bedrock",
}
_PROVIDER_CAPS = {
    "anthropic": Capabilities(supports_pdf=True, supports_caching=True),
    "openai_compat": Capabilities(supports_pdf=False, supports_caching=False),
    "gemini": Capabilities(supports_pdf=True, supports_caching=False),
    "bedrock": Capabilities(supports_pdf=True, supports_caching=False),
}


def resolve(surface: str) -> tuple[str, str]:
    """(provider, model) for a surface — env override wins over the registry default."""
    from core.model_registry import surface_models

    provider, model = surface_models(surface).default
    key = surface.upper()
    return (
        os.getenv(f"LLM_{key}_PROVIDER", provider),
        os.getenv(f"LLM_{key}_MODEL", model),
    )


def resolve_fallback(surface: str) -> tuple[str, str] | None:
    """The OPTIONAL secondary route for a surface — ``(provider, model)`` only when BOTH
    ``LLM_<SURFACE>_FALLBACK_PROVIDER`` and ``LLM_<SURFACE>_FALLBACK_MODEL`` are set
    (non-empty). Unset ⇒ ``None`` ⇒ no fallback layer (byte-identical to a bare primary)."""
    key = surface.upper()
    provider = os.getenv(f"LLM_{key}_FALLBACK_PROVIDER")
    model = os.getenv(f"LLM_{key}_FALLBACK_MODEL")
    if provider and model:
        return (provider, model)
    return None


def _build_adapter(surface: str, provider: str, model: str) -> LiteLLMAdapter:
    """Construct the LiteLLM-backed adapter for one (provider, model) route — the SHARED
    construction used for both the primary and the secondary, so a secondary is built
    exactly like a primary would be for that provider (caps / rasterise / api-key wiring).
    Reads route creds from env: anthropic → ANTHROPIC_API_KEY; openai_compat →
    LLM_OPENAI_BASE_URL / LLM_OPENAI_API_KEY; gemini → LLM_GEMINI_API_KEY; bedrock →
    LLM_BEDROCK_REGION (IAM)."""
    if provider not in _PROVIDER_PREFIX:
        raise ValueError(f"unknown LLM provider {provider!r} for surface {surface!r}")
    litellm_model = f"{_PROVIDER_PREFIX[provider]}/{model}"
    caps = _PROVIDER_CAPS[provider]

    if provider == "anthropic":
        # Read the key straight from the env (identical value to settings.anthropic_api_key,
        # which pydantic-settings just loads from ANTHROPIC_API_KEY) — so the LLM port stays
        # decoupled from the agent's DB/queue/S3 Settings and works in any context (e.g. the
        # eval bake-off), not only inside the fully-configured agent runtime.
        return LiteLLMAdapter(
            litellm_model, caps=caps, surface=surface, api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    if provider == "openai_compat":
        # Text-only by default; LLM_OPENAI_VISION=1 declares the route is a vision model
        # (Ollama qwen2.5vl / llava), so a PDF part is rasterised to per-page images.
        vision = os.getenv("LLM_OPENAI_VISION") == "1"
        return LiteLLMAdapter(
            litellm_model,
            caps=Capabilities(supports_pdf=vision, supports_caching=False),
            rasterize_pdf=vision,
            surface=surface,
            api_base=os.environ["LLM_OPENAI_BASE_URL"],
            api_key=os.getenv("LLM_OPENAI_API_KEY") or "not-needed",
        )
    if provider == "gemini":
        return LiteLLMAdapter(
            litellm_model, caps=caps, surface=surface, api_key=os.environ["LLM_GEMINI_API_KEY"]
        )
    # bedrock — IAM creds from the environment; region explicit for EU residency.
    return LiteLLMAdapter(
        litellm_model,
        caps=caps,
        surface=surface,
        aws_region_name=os.getenv("LLM_BEDROCK_REGION", "eu-central-1"),
    )


def get_llm(surface: str) -> LLMClient:
    """Build the ``LLMClient`` for a surface. Returns the bare primary adapter unless a
    secondary route is configured (``LLM_<SURFACE>_FALLBACK_PROVIDER`` + ``_MODEL``), in
    which case it returns a ``FallbackLLMClient(primary, secondary)`` — primary served
    normally, retryable primary failures fall through to the secondary."""
    provider, model = resolve(surface)
    primary = _build_adapter(surface, provider, model)

    fallback = resolve_fallback(surface)
    if fallback is None:
        return primary
    secondary = _build_adapter(surface, fallback[0], fallback[1])
    return FallbackLLMClient(primary, secondary)
