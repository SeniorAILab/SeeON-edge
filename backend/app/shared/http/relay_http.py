from __future__ import annotations

import hmac
from collections.abc import Callable, Coroutine, Mapping
from typing import Any, Protocol, runtime_checkable

from fastapi import HTTPException, Request, Response, status
from fastapi.routing import APIRoute
from starlette.types import Message, Receive

RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"


@runtime_checkable
class CameraRegistry(Protocol):
    def snapshot(self) -> Mapping[str, object]: ...


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


def authorize_relay_body(
    request: Request,
    *,
    max_bytes: int,
    relay_token: str | None,
    authorization: str | None = None,
) -> None:
    reject_oversized_body(request, max_bytes=max_bytes)
    authorize_relay(request, relay_token or _bearer_token(authorization))


def bounded_body_route(
    limits: Mapping[str, int],
    *,
    before_body: Callable[[Request], object] | None = None,
) -> type[APIRoute]:
    max_bytes_by_suffix = dict(limits)

    class BoundedBodyRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            original = super().get_route_handler()
            max_bytes = next(
                (
                    limit
                    for suffix, limit in max_bytes_by_suffix.items()
                    if self.path.endswith(suffix)
                ),
                None,
            )
            if max_bytes is None:
                return original

            async def bounded_handler(request: Request) -> Response:
                if before_body is not None:
                    before_body(request)
                request._receive = _bounded_receive(request.receive, max_bytes)  # noqa: SLF001
                return await original(request)

            return bounded_handler

    return BoundedBodyRoute


def camera_binding(request: Request, camera_id: str, facility_id: str) -> dict[str, str | None]:
    store = getattr(request.app.state, "camera_registry", None)
    if not isinstance(store, CameraRegistry):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="unknown camera")
    snapshot = store.snapshot()
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="unknown camera")
    for record in cameras:
        if not isinstance(record, dict):
            continue
        local_id = record.get("id")
        backend_id = record.get("backend_camera_id")
        if camera_id in {local_id, backend_id}:
            canonical_id = backend_id or local_id
            return {
                "camera_id": str(canonical_id),
                "facility_id": facility_id,
                "resident_id": None,
                "backend_camera_id": (
                    backend_id if isinstance(backend_id, str) and backend_id.strip() else None
                ),
            }
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="unknown camera")


def _oversized_body_error(max_bytes: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        detail=f"request body exceeds maximum of {max_bytes} bytes",
    )


def _bounded_receive(receive: Receive, max_bytes: int) -> Receive:
    total = 0

    async def wrapped() -> Message:
        nonlocal total
        message = await receive()
        if message["type"] == "http.request":
            body = message.get("body", b"")
            total += len(body)
            if total > max_bytes:
                raise _oversized_body_error(max_bytes)
        return message

    return wrapped


def reject_oversized_body(request: Request, *, max_bytes: int) -> None:
    raw = request.headers.get("content-length")
    if raw is None:
        return
    try:
        length = int(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid Content-Length",
        ) from exc
    if length < 0 or length > max_bytes:
        raise _oversized_body_error(max_bytes)


def _bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    scheme, separator, token = authorization.partition(" ")
    if separator and scheme.lower() == "bearer" and token:
        return token
    return None


__all__ = [
    "RELAY_TOKEN_HEADER",
    "CameraRegistry",
    "authorize_relay",
    "authorize_relay_body",
    "bounded_body_route",
    "camera_binding",
    "reject_oversized_body",
]
