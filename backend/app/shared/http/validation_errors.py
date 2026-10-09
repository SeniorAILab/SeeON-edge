from __future__ import annotations

import math
from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

MAX_ESCAPED_ERROR_NODES = 10_000
OMITTED_INPUT = "omitted: too large to escape"


def _leaf(value: object) -> object:
    if isinstance(value, str):
        return value.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(value, bytes):
        return value.decode("utf-8", "backslashreplace")
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _json_safe(value: object) -> object:
    root: list[Any] = [None]
    pending: list[tuple[Any, Any, object]] = [(root, 0, value)]
    while pending:
        parent, slot, item = pending.pop()
        if isinstance(item, dict):
            mapping: dict[Any, Any] = {}
            parent[slot] = mapping
            for key, child in item.items():
                safe_key = _leaf(key)
                mapping[safe_key] = None
                pending.append((mapping, safe_key, child))
        elif isinstance(item, list | tuple):
            items: list[Any] = [None] * len(item)
            parent[slot] = items
            pending.extend((items, index, child) for index, child in enumerate(item))
        else:
            parent[slot] = _leaf(item)
    return root[0]


def _exceeds(value: object, limit: int) -> bool:
    pending: list[object] = [value]
    seen = 0
    while pending:
        seen += 1
        if seen > limit:
            return True
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list | tuple):
            pending.extend(item)
    return False


def _bounded(error: object) -> object:
    if isinstance(error, dict) and _exceeds(error, MAX_ESCAPED_ERROR_NODES):
        return {**error, "input": OMITTED_INPUT}
    return error


async def request_validation_handler(request: Request, error: Exception) -> Response:
    del request
    if not isinstance(error, RequestValidationError):
        raise error
    errors = error.errors()
    try:
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})
    except ValueError:
        bounded = [_bounded(item) for item in errors]
        return JSONResponse(
            status_code=422,
            content={"detail": _json_safe(jsonable_encoder(_json_safe(bounded)))},
        )


__all__ = ["request_validation_handler"]
