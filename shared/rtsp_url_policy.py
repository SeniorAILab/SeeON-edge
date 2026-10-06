from __future__ import annotations

import ipaddress
import os
import socket
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

ALLOW_PRIVATE_RTSP_ENV: Final = "ML_RTSP_ALLOW_PRIVATE_DESTINATIONS"
ALLOW_LOCAL_RTSP_ENV: Final = "ML_RTSP_ALLOW_LOCAL_DESTINATIONS"

_ALLOWED_SCHEMES: Final = frozenset({"rtsp", "rtsps"})
_SPECIAL_HOSTNAMES: Final = frozenset(
    {
        "localhost",
        "localhost.",
        "metadata",
        "metadata.google.internal",
        "metadata.google.internal.",
        "metadata.goog",
        "metadata.goog.",
        "instance-data",
        "instance-data.",
    }
)
_METADATA_NETWORKS: Final = (
    ipaddress.ip_network("169.254.169.254/32"),
    ipaddress.ip_network("fd00:ec2::254/128"),
)

HostAddressResolver = Callable[[str], Sequence[str]]


@dataclass(frozen=True, slots=True)
class ResolvedRtspEndpoint:
    original_url: str
    pinned_url: str
    hostname: str
    addresses: tuple[str, ...]
    selected_address: str


def allow_private_rtsp_from_env() -> bool:
    return os.environ.get(ALLOW_PRIVATE_RTSP_ENV, "").strip() == "1"


def allow_local_rtsp_from_env() -> bool:
    return os.environ.get(ALLOW_LOCAL_RTSP_ENV, "").strip() == "1"


def _hostname(netloc: str) -> str:
    _, _, hostport = netloc.rpartition("@")
    host = hostport
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[1:end]
        return host.strip("[]")
    return host.rsplit(":", 1)[0]


def _port_suffix(netloc: str) -> str:
    _, _, hostport = netloc.rpartition("@")
    if hostport.startswith("["):
        end = hostport.find("]")
        if end == -1:
            return ""
        rest = hostport[end + 1 :]
        return rest if rest.startswith(":") else ""
    if hostport.count(":") != 1:
        return ""
    _, _, port = hostport.partition(":")
    return f":{port}" if port else ""


def _userinfo_prefix(netloc: str) -> str:
    if "@" not in netloc:
        return ""
    userinfo, _, _ = netloc.rpartition("@")
    return f"{userinfo}@"


def _is_blocked_ip(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    allow_private: bool,
    allow_local: bool,
) -> str | None:
    for network in _METADATA_NETWORKS:
        if address in network:
            return "metadata destination is not permitted"
    if address.is_unspecified:
        return "unspecified destination is not permitted"
    if address.is_multicast:
        return "multicast destination is not permitted"
    if address.is_loopback:
        if allow_local:
            return None
        return "loopback destination is not permitted"
    if address.is_link_local:
        if allow_local:
            return None
        return "link-local destination is not permitted"
    if address.is_private or address in ipaddress.ip_network("100.64.0.0/10"):
        if allow_private or allow_local:
            return None
        return "private destination is not permitted"
    return None


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def resolve_host_a_aaaa(hostname: str) -> tuple[str, ...]:
    cleaned = hostname.strip()
    if not cleaned:
        return ()
    infos = socket.getaddrinfo(
        cleaned,
        None,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
    )
    addresses: list[str] = []
    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        raw = sockaddr[0]
        try:
            rendered = str(ipaddress.ip_address(raw))
        except ValueError:
            continue
        if rendered in seen:
            continue
        seen.add(rendered)
        addresses.append(rendered)
    return tuple(addresses)


def reject_resolved_addresses_reason(
    addresses: Sequence[str],
    *,
    allow_private: bool | None = None,
    allow_local: bool | None = None,
) -> str | None:
    if not addresses:
        return "rtsp destination could not be resolved"
    private = allow_private if allow_private is not None else allow_private_rtsp_from_env()
    local = allow_local if allow_local is not None else allow_local_rtsp_from_env()
    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            return "rtsp destination resolved to an invalid address"
        reason = _is_blocked_ip(address, allow_private=private, allow_local=local)
        if reason is not None:
            return reason
    return None


def _pin_host_in_url(url: str, address: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    ip_obj = ipaddress.ip_address(address)
    host_literal = f"[{ip_obj.compressed}]" if ip_obj.version == 6 else str(ip_obj)
    netloc = f"{_userinfo_prefix(parsed.netloc)}{host_literal}{_port_suffix(parsed.netloc)}"
    return urllib.parse.urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def reject_rtsp_url_reason(
    url: str,
    *,
    allow_private: bool | None = None,
    allow_local: bool | None = None,
) -> str | None:
    cleaned = (url or "").strip()
    if not cleaned:
        return "rtsp URL is empty"
    if any(ord(char) < 32 for char in cleaned):
        return "rtsp URL contains control characters"
    try:
        parsed = urllib.parse.urlsplit(cleaned)
    except ValueError:
        return "rtsp URL is not a valid absolute URL"
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        return "rtsp URL scheme must be rtsp or rtsps"
    if not parsed.netloc:
        return "rtsp URL must include a host"
    if parsed.fragment:
        return "rtsp URL must not include a fragment"
    host = _hostname(parsed.netloc).strip().lower().rstrip(".")
    if not host:
        return "rtsp URL must include a host"
    private = allow_private if allow_private is not None else allow_private_rtsp_from_env()
    local = allow_local if allow_local is not None else allow_local_rtsp_from_env()
    if host in _SPECIAL_HOSTNAMES or host.startswith("localhost."):
        if local and host.startswith("localhost"):
            return None
        return "special hostname destination is not permitted"
    address = _literal_ip(host)
    if address is None:
        return None
    return _is_blocked_ip(address, allow_private=private, allow_local=local)


def resolve_rtsp_endpoint(
    url: str,
    *,
    allow_private: bool | None = None,
    allow_local: bool | None = None,
    resolver: HostAddressResolver | None = None,
) -> ResolvedRtspEndpoint:
    cleaned = (url or "").strip()
    reason = reject_rtsp_url_reason(
        cleaned,
        allow_private=allow_private,
        allow_local=allow_local,
    )
    if reason is not None:
        raise ValueError(reason)

    parsed = urllib.parse.urlsplit(cleaned)
    host = _hostname(parsed.netloc).strip()
    host_key = host.lower().rstrip(".")
    private = allow_private if allow_private is not None else allow_private_rtsp_from_env()
    local = allow_local if allow_local is not None else allow_local_rtsp_from_env()

    literal = _literal_ip(host_key)
    if literal is not None:
        selected = str(literal)
        return ResolvedRtspEndpoint(
            original_url=cleaned,
            pinned_url=_pin_host_in_url(cleaned, selected),
            hostname=host_key,
            addresses=(selected,),
            selected_address=selected,
        )

    resolve = resolver if resolver is not None else resolve_host_a_aaaa
    try:
        resolved = tuple(str(ipaddress.ip_address(item)) for item in resolve(host_key))
    except (OSError, ValueError) as exc:
        raise ValueError("rtsp destination could not be resolved") from exc

    blocked = reject_resolved_addresses_reason(
        resolved,
        allow_private=private,
        allow_local=local,
    )
    if blocked is not None:
        raise ValueError(blocked)

    selected = resolved[0]
    return ResolvedRtspEndpoint(
        original_url=cleaned,
        pinned_url=_pin_host_in_url(cleaned, selected),
        hostname=host_key,
        addresses=resolved,
        selected_address=selected,
    )


def assert_rtsp_url_allowed(
    url: str,
    *,
    allow_private: bool | None = None,
    allow_local: bool | None = None,
) -> str:
    cleaned = (url or "").strip()
    reason = reject_rtsp_url_reason(cleaned, allow_private=allow_private, allow_local=allow_local)
    if reason is not None:
        raise ValueError(reason)
    return cleaned


def assert_rtsp_endpoint_allowed(
    url: str,
    *,
    allow_private: bool | None = None,
    allow_local: bool | None = None,
    resolver: HostAddressResolver | None = None,
) -> ResolvedRtspEndpoint:
    return resolve_rtsp_endpoint(
        url,
        allow_private=allow_private,
        allow_local=allow_local,
        resolver=resolver,
    )


__all__ = [
    "ALLOW_LOCAL_RTSP_ENV",
    "ALLOW_PRIVATE_RTSP_ENV",
    "HostAddressResolver",
    "ResolvedRtspEndpoint",
    "allow_local_rtsp_from_env",
    "allow_private_rtsp_from_env",
    "assert_rtsp_endpoint_allowed",
    "assert_rtsp_url_allowed",
    "reject_resolved_addresses_reason",
    "reject_rtsp_url_reason",
    "resolve_host_a_aaaa",
    "resolve_rtsp_endpoint",
]
