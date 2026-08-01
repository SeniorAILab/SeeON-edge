from __future__ import annotations

import http.client
import urllib.parse
import urllib.request
from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self, TypeAlias, final

from worker.runtime.config.errors import WorkerConfigError


class HttpResponse(Protocol):
    status: int

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def read(self) -> bytes: ...


UrlOpen: TypeAlias = Callable[[urllib.request.Request, float], HttpResponse]


@final
class ManagedHttpResponse:
    def __init__(
        self,
        connection: http.client.HTTPConnection,
        response: http.client.HTTPResponse,
    ) -> None:
        self.status: int = response.status
        self._connection: http.client.HTTPConnection = connection
        self._response: http.client.HTTPResponse = response

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._response.close()
        self._connection.close()

    def read(self) -> bytes:
        return self._response.read()


def stdlib_urlopen(request: urllib.request.Request, timeout: float) -> HttpResponse:
    parsed = urllib.parse.urlsplit(request.full_url)
    hostname = parsed.hostname
    if hostname is None:
        raise WorkerConfigError("worker config URL must include a host")
    factories: dict[str, type[http.client.HTTPConnection]] = {
        "http": http.client.HTTPConnection,
        "https": http.client.HTTPSConnection,
    }
    try:
        factory = factories[parsed.scheme.lower()]
    except KeyError as error:
        raise WorkerConfigError("worker config URL must use HTTP(S)") from error
    connection = factory(hostname, parsed.port, timeout=timeout)
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    try:
        connection.request(
            request.get_method(),
            path,
            headers=dict(request.header_items()),
        )
        return ManagedHttpResponse(connection, connection.getresponse())
    except (OSError, http.client.HTTPException) as error:
        connection.close()
        raise WorkerConfigError("worker config request failed") from error


__all__ = ["HttpResponse", "ManagedHttpResponse", "UrlOpen", "stdlib_urlopen"]
