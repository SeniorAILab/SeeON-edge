from __future__ import annotations

import json
import queue
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from edge.runtime.runtime_diagnostics import WorkerDiagnostics

_RUNTIME_STATUS_PATH = "/api/v1/relay/runtime-status"


class RuntimeStatusSender:
    """Publish worker telemetry off the capture and frame-processing threads."""

    def __init__(
        self,
        relay_url: str,
        relay_token: str,
        diagnostics: WorkerDiagnostics,
        facility_id: str | Mapping[str, str],
        *,
        publish_interval_sec: float = 5.0,
        initial_backoff_sec: float = 1.0,
        max_backoff_sec: float = 30.0,
        timeout_sec: float = 2.0,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        self._url = f"{relay_url.rstrip('/')}{_RUNTIME_STATUS_PATH}"
        self._relay_token = relay_token
        self._diagnostics = diagnostics
        self._facility_id = facility_id
        self._publish_interval_sec = max(0.0, publish_interval_sec)
        self._initial_backoff_sec = max(0.0, initial_backoff_sec)
        self._max_backoff_sec = max(self._initial_backoff_sec, max_backoff_sec)
        self._timeout_sec = timeout_sec
        self._urlopen = urllib.request.urlopen if urlopen is None else urlopen
        self._snapshots: queue.Queue[list[dict[str, object]]] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation_by_facility: dict[str, int] = {}
        self._seq_by_facility: dict[str, int] = {}
        self._generation: int | None = None
        self._state_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.publish()
        self._thread = threading.Thread(
            target=self._run,
            name="runtime-status-sender",
            daemon=True,
        )
        self._thread.start()

    def publish(self) -> bool:
        """Capture the latest diagnostic snapshot without blocking the caller."""
        snapshots = self._snapshots_for_publish()
        try:
            self._snapshots.put_nowait(snapshots)
        except queue.Full:
            try:
                self._snapshots.get_nowait()
                self._snapshots.task_done()
            except queue.Empty:
                pass
            try:
                self._snapshots.put_nowait(snapshots)
            except queue.Full:
                return False
            else:
                return True
        else:
            return True

    submit = publish
    def publish_once(self) -> bool:
        """Synchronously make one bounded publish attempt without retrying."""
        return self._post(self._snapshots_for_publish())

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def generation(self) -> int | None:
        with self._state_lock:
            return self._generation

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        pending: list[dict[str, object]] | None = None
        delay = 0.0
        failures = 0
        while not self._stop_event.wait(delay):
            self.publish()
            replacement = self._take_latest()
            if replacement is not None:
                pending = replacement
            if pending is None:
                delay = self._publish_interval_sec
                continue
            if self._post(pending):
                failures = 0
                delay = self._publish_interval_sec
            else:
                failures += 1
                delay = min(
                    self._max_backoff_sec,
                    self._initial_backoff_sec * (2 ** (failures - 1)),
                )

    def _take_latest(self) -> list[dict[str, object]] | None:
        latest: list[dict[str, object]] | None = None
        while True:
            try:
                latest = self._snapshots.get_nowait()
            except queue.Empty:
                return latest
            else:
                self._snapshots.task_done()

    def _snapshots_for_publish(self) -> list[dict[str, object]]:
        if isinstance(self._facility_id, str):
            return [self._diagnostics.to_payload(self._facility_id, None, 0)]
        return self._diagnostics.to_payloads(self._facility_id, None, 0)

    def _post(self, snapshots: list[dict[str, object]]) -> bool:
        for snapshot in snapshots:
            facility_id = str(snapshot["facility_id"])
            with self._state_lock:
                seq = self._seq_by_facility.get(facility_id, 0) + 1
                self._seq_by_facility[facility_id] = seq
                generation = self._generation_by_facility.get(facility_id)
            payload = dict(snapshot)
            payload["generation"] = generation
            payload["seq"] = seq
            request = urllib.request.Request(
                self._url,
                data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._relay_token}",
                },
                method="POST",
            )
            try:
                with self._urlopen(request, timeout=self._timeout_sec) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except (
                TimeoutError,
                OSError,
                ValueError,
                urllib.error.HTTPError,
                urllib.error.URLError,
            ):
                return False
            if body.get("accepted") is not True:
                return False
            generation_value = body.get("generation")
            if not isinstance(generation_value, int):
                return False
            with self._state_lock:
                self._generation_by_facility[facility_id] = generation_value
                self._generation = generation_value
        return True


__all__ = ["RuntimeStatusSender"]
