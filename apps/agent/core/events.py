"""Agent-side EventBridge publisher.

Agents are normally pure SQS consumers, but the specimen classifier needs to hand a
matched specimen off to the redaction pipeline — and it cannot reach the API to do so
(the API's ALB is internet-facing; the agent runs in a private subnet, so the NAT
hairpin times out — the same dead path the realtime push hit, fixed via NOTIFY). So
the agent publishes ``specimen.redact_requested`` itself.

The event is emitted with Source ``jev-bakeoff.api`` on purpose: it is byte-identical
to what the API would publish, so the existing ``agent_specimen_redact`` rule (which
matches that source) routes it to the redactor with no infra change. The agent's task
role already grants ``events:PutEvents`` on this bus.
"""

import json
from typing import Any

import aioboto3
from botocore.config import Config

from core.config import settings


def _make_session() -> aioboto3.Session:
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return aioboto3.Session(**kwargs)


async def publish_event(detail_type: str, detail: dict[str, Any]) -> None:
    """Publish one event to the app's custom EventBridge bus."""
    session = _make_session()
    async with session.client(
        "events",
        endpoint_url=settings.aws_endpoint_url or None,
        config=Config(retries={"max_attempts": 3}),
    ) as eb:
        await eb.put_events(
            Entries=[
                {
                    "Source": "jev-bakeoff.api",
                    "DetailType": detail_type,
                    "Detail": json.dumps(detail),
                    "EventBusName": settings.eventbridge_bus_name,
                }
            ]
        )
