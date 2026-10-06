from __future__ import annotations

import argparse
import os
import socket
import time

_BLACKHOLED_HOSTS = ("one.one.one.one", "dns.google")
_real_getaddrinfo = socket.getaddrinfo


def _blackhole_getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
    if host in _BLACKHOLED_HOSTS:
        time.sleep(3600)
    return _real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-guard", action="store_true")
    args = parser.parse_args()

    socket.getaddrinfo = _blackhole_getaddrinfo  # type: ignore[assignment]

    if args.skip_guard:
        os.environ.pop("YOLO_OFFLINE", None)
        import ultralytics  # noqa: F401
    else:
        import worker.adapters.model.yolo_api  # noqa: F401

    print("SUBPROCESS_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
