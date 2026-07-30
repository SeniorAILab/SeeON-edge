from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def mask_rtsp_url(url: str) -> str:
    parsed = urlsplit(url)
    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, hostport = netloc.rsplit("@", 1)
        mask = "***:***" if ":" in userinfo else "***"
        netloc = f"{mask}@{hostport}"
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query = urlencode([(key, "***") for key, _value in query_pairs], doseq=True)
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))



__all__ = ["mask_rtsp_url"]
