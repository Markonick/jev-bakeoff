"""SSRF guard for outbound fetches of user-supplied URLs.

The property-analysis agent fetches a listing URL a buyer/broker pastes, and then an
``og:image`` URL taken from that listing's (attacker-controllable) HTML — all from
inside our VPC while the task holds an IAM role. Without a guard, a pasted
``http://169.254.170.2/...`` (ECS credentials endpoint) or ``http://<internal-host>/``
is a textbook SSRF-to-credential-theft path.

``guarded_get`` enforces, on the initial URL **and every redirect hop**:
  * scheme is http/https,
  * the host resolves only to public IPs — loopback, private (RFC1918), link-local
    (incl. 169.254.0.0/16 cloud metadata), multicast, reserved and unspecified
    addresses are rejected (IPv4-mapped IPv6 is unwrapped first),
  * redirects are followed manually (``allow_redirects=False``) so each ``Location``
    is re-validated before we connect.

We use an IP denylist rather than a domain allowlist on purpose: the product lets a
buyer paste "xe.gr / spitogatos / other" listing sites, so we can't enumerate hosts.
Residual DNS-rebinding (host re-resolving to a private IP between our check and the
socket connect) is not fully closed here; the denylist blocks the direct, trivially
exploitable cases, which is the proportionate control for this fetch.
"""

import asyncio
import ipaddress
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urljoin, urlsplit

import aiohttp

_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class BlockedURLError(Exception):
    """Raised when a URL fails the SSRF guard (bad scheme, private/blocked host)."""


def _ip_blocked(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    # Unwrap IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) so the checks below apply.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


async def _validate_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise BlockedURLError(f"scheme not allowed: {parts.scheme!r}")
    host = parts.hostname
    if not host:
        raise BlockedURLError("missing host")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, parts.port or (443 if parts.scheme == "https" else 80),
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        raise BlockedURLError(f"cannot resolve host: {host}") from exc
    if not infos:
        raise BlockedURLError(f"host did not resolve: {host}")
    for info in infos:
        ip = info[4][0]
        if _ip_blocked(ip):
            raise BlockedURLError(f"host {host} resolves to blocked address {ip}")


async def validate_public_url(url: str) -> None:
    """Public SSRF check for callers that fetch a user-supplied URL *outside*
    ``guarded_get`` — e.g. a headless-browser navigation, which we can't route through
    the aiohttp guard. Validates scheme + that the host resolves only to public IPs;
    raises ``BlockedURLError`` otherwise. Note: this guards the entry URL only —
    in-browser redirects are not per-hop validated (same residual as the DNS-rebinding
    note above), which is proportionate for the gated render path.
    """
    await _validate_url(url)


@asynccontextmanager
async def guarded_get(
    session: aiohttp.ClientSession, url: str
) -> AsyncIterator[aiohttp.ClientResponse]:
    """GET ``url`` with SSRF validation on the initial URL and every redirect hop.

    Yields the final (non-redirect) response; the caller reads and closes it via the
    ``async with`` block. Raises ``BlockedURLError`` if any hop fails the guard.
    """
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        await _validate_url(current)
        resp = await session.get(current, allow_redirects=False)
        location = resp.headers.get("Location")
        if resp.status in _REDIRECT_STATUSES and location:
            resp.release()
            current = urljoin(current, location)
            continue
        try:
            yield resp
        finally:
            resp.release()
        return
    raise BlockedURLError("too many redirects")
