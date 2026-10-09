from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response


def is_head(request: Request) -> bool:
    return request.method.upper() == "HEAD"


def drop_body_for_head(request: Request, response: Response) -> Response:
    if is_head(request):
        response.body = b""
    return response


__all__ = ["drop_body_for_head", "is_head"]
