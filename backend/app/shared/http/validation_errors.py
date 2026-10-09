from __future__ import annotations

import math

from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse


def _json_safe(value: object) -> object:
    if isinstance(value, str):
        return value.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(value, bytes):
        return value.decode("utf-8", "backslashreplace")
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {_json_safe(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


async def request_validation_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    encoded = jsonable_encoder([_json_safe(item) for item in error.errors()])
    errors = [_json_safe(item) for item in encoded]
    return await request_validation_exception_handler(
        request, RequestValidationError(errors, body=error.body)
    )


__all__ = ["request_validation_handler"]
