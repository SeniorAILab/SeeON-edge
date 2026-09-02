from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_DIAGNOSTIC_ENV: Final = "SEEON_FALL_DIAGNOSTICS_ENABLED"
_MAX_BUNDLE_BYTES: Final = 2_000_000
_MAX_STORED_BUNDLES: Final = 64
_MAX_STORED_BYTES: Final = 128_000_000


@dataclass(frozen=True, slots=True)
class FallDiagnosticWriterStats:
    accepted: int
    queue_drops: int
    oversized: int
    written: int
    write_failures: int


class FallDiagnosticWriter:
    """Bounded, best-effort content-addressed writer off the alert hot path."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        persist: Callable[[bytes], None] | None = None,
        max_pending: int = 2,
        max_bundle_bytes: int = _MAX_BUNDLE_BYTES,
        max_stored_bundles: int = _MAX_STORED_BUNDLES,
        max_stored_bytes: int = _MAX_STORED_BYTES,
    ) -> None:
        if (root is None) == (persist is None):
            raise ValueError("exactly one diagnostic persistence target is required")
        if min(max_pending, max_bundle_bytes, max_stored_bundles, max_stored_bytes) <= 0:
            raise ValueError("diagnostic writer bounds must be positive")
        self._persist = (
            self._content_addressed_persist(root, max_stored_bundles, max_stored_bytes)
            if root is not None
            else persist
        )
        self._pending: queue.Queue[object] = queue.Queue(maxsize=max_pending)
        self._max_bundle_bytes = max_bundle_bytes
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._accepted = 0
        self._queue_drops = 0
        self._oversized = 0
        self._written = 0
        self._write_failures = 0

    @staticmethod
    def _content_addressed_persist(
        root: Path,
        max_stored_bundles: int,
        max_stored_bytes: int,
    ) -> Callable[[bytes], None]:
        def persist(payload: bytes) -> None:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            root.chmod(0o700)
            digest = hashlib.sha256(payload).hexdigest()
            destination = root / f"{digest}.json"
            if destination.exists():
                if destination.read_bytes() != payload:
                    raise OSError("content-addressed diagnostic collision")
                return
            stored = tuple(root.glob("*.json"))
            stored_bytes = sum(path.stat().st_size for path in stored)
            if len(stored) >= max_stored_bundles or stored_bytes + len(payload) > max_stored_bytes:
                raise OSError("diagnostic retention bound reached")
            temporary = root / f".{digest}.tmp"
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)

        return persist

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="fall-diagnostic-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, payload: bytes) -> bool:
        if len(payload) > self._max_bundle_bytes:
            self._increment("oversized")
            return False
        return self._submit(payload)

    def submit_bundle(self, payload: Mapping[str, object]) -> bool:
        return self._submit(dict(payload))

    def _submit(self, payload: object) -> bool:
        try:
            self._pending.put_nowait(payload)
        except queue.Full:
            self._increment("queue_drops")
            return False
        self._increment("accepted")
        return True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def stats(self) -> FallDiagnosticWriterStats:
        with self._lock:
            return FallDiagnosticWriterStats(
                self._accepted,
                self._queue_drops,
                self._oversized,
                self._written,
                self._write_failures,
            )

    def _increment(self, field: str) -> None:
        with self._lock:
            setattr(self, f"_{field}", getattr(self, f"_{field}") + 1)

    def _run(self) -> None:
        while not self._stop.is_set() or not self._pending.empty():
            try:
                pending = self._pending.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                payload = (
                    pending
                    if isinstance(pending, bytes)
                    else json.dumps(
                        pending,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                )
                if len(payload) > self._max_bundle_bytes:
                    self._increment("oversized")
                    continue
                assert self._persist is not None
                self._persist(payload)
            except (OSError, TypeError, ValueError):
                self._increment("write_failures")
            else:
                self._increment("written")
            finally:
                self._pending.task_done()


def build_fall_diagnostic_writer(
    env: Mapping[str, str],
    state_dir: Path,
) -> FallDiagnosticWriter | None:
    if env.get(_DIAGNOSTIC_ENV, "") != "1":
        return None
    root = state_dir / "fall-diagnostics"
    _purge_previous_boot(root)
    writer = FallDiagnosticWriter(root)
    writer.start()
    return writer


def _purge_previous_boot(root: Path) -> None:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise OSError("fall diagnostic root is not a private directory")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise OSError("fall diagnostic root contains an unsafe entry")
        path.unlink()


__all__ = [
    "FallDiagnosticWriter",
    "FallDiagnosticWriterStats",
    "build_fall_diagnostic_writer",
]
