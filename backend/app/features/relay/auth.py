from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status


def authorize_relay(request: Request, relay_token: str | None) -> None:
    expected = getattr(request.app.state, "edge_relay_token", None)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="relay token is not configured",
        )
    if relay_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="relay token required",
        )
    if not hmac.compare_digest(relay_token.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="relay token mismatch",
        )


__all__ = ["authorize_relay"]
