"""Fetch an article's ``og:image`` URL to hot-link on a Market-Radar news card.

The radar stores a link + summary for each surviving news item; the frontend renders a
generic category tile unless it has a real preview image. This helper GETs the article
and pulls the Open Graph ``og:image`` (or ``twitter:image``) the publisher already serves
to Facebook/messaging scrapers, returning its **URL** only.

We never download/rehost those bytes — the URL goes straight into ``news_items.image_url``
and the *browser* loads it as an ``<img src>`` (the CSP allows hot-linked https images).
Mirrors the proven extractor in ``agents.property_analysis`` (kept unchanged here to limit
blast radius). Best-effort: any failure/timeout/non-http result yields ``None`` — never
raises — so a missing image just falls back to the tile.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
import structlog

from core.net_guard import BlockedURLError, guarded_get

logger = structlog.get_logger(__name__)

# og tags live in <head>; read only enough to reach them (avoid pulling whole articles).
_MAX_BODY_BYTES = 200 * 1024


class _OgParser(HTMLParser):
    """Collect the page's preview-image signals, in preference order:
    ``<meta property="og:*"|name="twitter:*">``, ``<link rel="image_src">``, and the raw
    ``<script type="application/ld+json">`` blocks (schema.org ``image``). Gov/ESPA pages
    often omit og:image but carry a publisher logo in JSON-LD or ``image_src`` — a real
    branded image beats the generic tile."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: dict[str, str] = {}
        self.ld_blocks: list[str] = []
        self._in_ld = False
        self._ld_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v for k, v in attrs}
        if tag == "meta":
            key = a.get("property") or a.get("name") or ""
            content = a.get("content")
            if content and (key.startswith("og:") or key.startswith("twitter:")):
                self.tags.setdefault(key, content)
        elif tag == "link":
            rel = (a.get("rel") or "").lower()
            href = a.get("href")
            if href and "image_src" in rel:
                self.tags.setdefault("link:image_src", href)
        elif tag == "script" and (a.get("type") or "").lower() == "application/ld+json":
            self._in_ld = True
            self._ld_buf = []

    def handle_data(self, data: str) -> None:
        if self._in_ld:
            self._ld_buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_ld:
            self._in_ld = False
            block = "".join(self._ld_buf).strip()
            if block:
                self.ld_blocks.append(block)


def _coerce_image(img: Any) -> str | None:
    """schema.org ``image`` may be a URL string, an ``ImageObject`` ({"url": …}), or a list
    of either. Return the first URL string found."""
    if isinstance(img, str):
        return img
    if isinstance(img, dict):
        url = img.get("url") or img.get("contentUrl")
        return url if isinstance(url, str) else None
    if isinstance(img, list):
        for item in img:
            found = _coerce_image(item)
            if found:
                return found
    return None


def _ld_image(blocks: list[str]) -> str | None:
    """Pull the first ``image`` URL out of any JSON-LD block (handles a top-level list and
    the common ``@graph`` nesting). Malformed JSON is skipped."""
    for raw in blocks:
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if not isinstance(node, dict):
                continue
            graph = node.get("@graph")
            candidates = graph if isinstance(graph, list) else [node]
            for cand in candidates:
                if isinstance(cand, dict):
                    found = _coerce_image(cand.get("image"))
                    if found:
                        return found
    return None


def _resolve(raw: str | None, base_url: str) -> str | None:
    """Resolve a raw image reference to an absolute http(s) URL (protocol-relative + relative
    against the final article URL); reject non-web schemes."""
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith("//"):
        raw = "https:" + raw
    resolved = urljoin(base_url, raw)
    parsed = urlparse(resolved)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return resolved
    return None


def _pick_image_url(
    tags: dict[str, str], base_url: str, ld_blocks: list[str] | None = None
) -> str | None:
    """First usable preview image, in preference order: og:image → twitter:image →
    ``link rel=image_src`` → JSON-LD ``image``. Never fetched server-side — the browser
    hot-links it."""
    for raw in (
        tags.get("og:image"),
        tags.get("twitter:image"),
        tags.get("link:image_src"),
        _ld_image(ld_blocks or []),
    ):
        resolved = _resolve(raw, base_url)
        if resolved:
            return resolved
    return None


async def fetch_og_image(
    http: aiohttp.ClientSession, url: str, *, timeout: aiohttp.ClientTimeout | None = None
) -> str | None:
    """Return the article's ``og:image`` (or ``twitter:image``) as an absolute http(s) URL.

    SSRF-safe (``guarded_get`` validates scheme + resolved IP on every redirect hop), reads a
    bounded head slice only, and returns ``None`` on any failure/timeout/non-http result —
    never raises. Relative image URLs resolve against the response's final URL.
    """
    if not url:
        return None
    try:
        async with guarded_get(http, url) as resp:
            resp.raise_for_status()
            body = await resp.content.read(_MAX_BODY_BYTES)
            base = str(resp.url)
    except (BlockedURLError, aiohttp.ClientError, TimeoutError, UnicodeError) as exc:
        logger.info("og_image.fetch_failed", url=url[:200], error=str(exc)[:160])
        return None
    except Exception as exc:  # noqa: BLE001 — enrichment is best-effort; never sink the item
        logger.info("og_image.fetch_error", url=url[:200], error=str(exc)[:160])
        return None

    parser = _OgParser()
    parser.feed(body.decode("utf-8", "ignore"))
    return _pick_image_url(parser.tags, base or url, parser.ld_blocks)
