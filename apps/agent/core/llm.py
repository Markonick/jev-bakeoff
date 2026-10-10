"""Shared Anthropic helpers for the agent container.

Prompt caching
--------------
``cached_system`` wraps a large, STABLE system prompt in a ``cache_control`` block so
repeat invocations that share the same prompt within the 5-minute cache TTL read the
prefix from cache (~90% cheaper on input tokens) instead of re-billing it every time.

Only apply this to a system prompt that is constant (or drawn from a small fixed set,
e.g. one-per-document-type) across invocations — that is what gets reused and hits the
cache. The per-invocation content (the uploaded PDF, page text, buyer profile) stays in
the user turn and is deliberately NOT cached: it varies every call, so caching it would
only ever pay the +25% write cost with no read to amortise it.

Safety: if the wrapped prompt is below the model's minimum cacheable size, Anthropic
silently processes the request without caching (no error), so this is safe to apply
uniformly — small prompts simply don't cache, large ones do.
"""

from anthropic.types import TextBlockParam


def cached_system(text: str) -> list[TextBlockParam]:
    """Return ``text`` as a single system content block marked for ephemeral
    (5-minute) prompt caching. Pass the result straight to ``system=`` on
    ``messages.create`` / ``messages.stream``."""
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]
