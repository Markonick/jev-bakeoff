"""PostHog product analytics — shared init + capture for this service.

No-op unless ``POSTHOG_API_KEY`` is set, so local dev, tests, and CI stay
offline by default. Design decisions (mirror the Sentry posture):

  * **Explicit events only** — no autocapture, no session recording. We call
    ``capture(...)`` for named domain events (case_created, document_reviewed …).
  * **No PII** — identify people by their Clerk user id and group by broker; never
    send emails or names as properties.
  * ``capture(...)`` only enqueues onto an in-memory buffer flushed by a background
    thread, so it is safe to call from async handlers (no blocking I/O).
"""

from typing import Any

from posthog import Posthog

from core.config import settings

_client: Posthog | None = None
_service: str = "unknown"


def init_analytics(service: str) -> None:
    """Initialise the module-level PostHog client. No-op without an API key."""
    global _client, _service
    _service = service
    if not settings.posthog_api_key:
        return
    _client = Posthog(
        project_api_key=settings.posthog_api_key,
        host=settings.posthog_host,
    )


def capture(
    distinct_id: str,
    event: str,
    *,
    properties: dict[str, Any] | None = None,
    broker_id: str | None = None,
    person_profile: bool = True,
) -> None:
    """Enqueue a product-analytics event. No-op if PostHog is not configured.

    ``person_profile=False`` marks a system event (no interactive user) so PostHog
    does not create/update a person profile for ``distinct_id`` — use it for
    worker/agent events keyed on a case id rather than a Clerk user id.
    """
    if _client is None:
        return
    props: dict[str, Any] = {"service": _service, **(properties or {})}
    if not person_profile:
        props["$process_person_profile"] = False
    groups = {"broker": broker_id} if broker_id else None
    _client.capture(distinct_id=distinct_id, event=event, properties=props, groups=groups)


def shutdown_analytics() -> None:
    """Flush any buffered events on shutdown so nothing is lost."""
    if _client is not None:
        _client.flush()
