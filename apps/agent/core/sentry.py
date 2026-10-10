"""Sentry error monitoring — shared init for this service.

No-op unless ``SENTRY_DSN`` is set, so local dev, tests, and CI stay offline by
default. We rely on Sentry's auto-enabled integrations rather than per-call
capture:

  * FastAPI/Starlette — captures unhandled errors raised in request handlers;
  * logging — captures ``logger.exception(...)`` / ERROR-level logs as events.
    Our structlog setup routes through stdlib logging, so the worker's and
    agent's SQS poll loops (which swallow-and-log their exceptions to keep
    polling) still get reported without touching the loop bodies.
"""

import sentry_sdk

from core.config import settings


def init_sentry(service: str) -> None:
    if not settings.sentry_dsn:
        return
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        release=settings.app_version,
        # Errors only for now — turn tracing on later if we want perf data.
        traces_sample_rate=0.0,
        # GDPR: we handle Greek tax/financial data. Never let Sentry attach
        # request bodies, headers, cookies, or user IP addresses by default.
        send_default_pii=False,
    )
    # Distinguishes api / worker / agent issues when they share one DSN.
    sentry_sdk.set_tag("service", service)
