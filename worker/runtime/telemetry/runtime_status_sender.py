from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, final

from shared.events.delivery_queue import DeliveryQueue
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.evidence_http_transport import (
    bounded_request,
    classify_http_failure,
    encode_json,
    join_http_url,
    normalize_http_base,
    parse_json_object,
)
from shared.events.relay_failure_log import RelayFailureLog
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.wire import RelayDeliveryQueuePayload, RelayRuntimeStatusPayload

LOGGER: Final = logging.getLogger(__name__)

RUNTIME_STATUS_PATH = "/api/v1/relay/runtime-status"


@dataclass(frozen=True, slots=True)
class RuntimeStatusSenderConfig:
    publish_interval_sec: float = 5.0
    initial_backoff_sec: float = 1.0
    max_backoff_sec: float = 30.0


class RuntimeStatusTransport(Protocol):
    def send(self, payload: RelayRuntimeStatusPayload) -> int | None: ...


class RuntimeHttpRequest(Protocol):
    def __call__(
        self,
        url: str,
        method: str,
        headers: Mapping[str, str],
        data: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, str], bytes] | DeliveryFailure: ...


@final
class RelayRuntimeStatusTransport:
    def __init__(
        self,
        relay_url: str,
        relay_token: str,
        timeout_sec: float = 2.0,
        request: RuntimeHttpRequest = bounded_request,
    ) -> None:
        self._url = join_http_url(normalize_http_base(relay_url), RUNTIME_STATUS_PATH)
        self._relay_token = relay_token
        self._timeout_sec = timeout_sec
        self._request = request
        self._failure_log = RelayFailureLog(LOGGER, channel="runtime-status", method="POST")

    def send(self, payload: RelayRuntimeStatusPayload) -> int | None:
        result = self._request(
            self._url,
            "POST",
            {
                "Authorization": f"Bearer {self._relay_token}",
                "Content-Type": "application/json",
            },
            encode_json(payload),
            self._timeout_sec,
        )
        if isinstance(result, DeliveryFailure):
            self._failure_log.record_failure(result, path=RUNTIME_STATUS_PATH)
            return None
        status, headers, body = result
        if not 200 <= status < 300:
            self._failure_log.record_failure(
                classify_http_failure(status, headers), path=RUNTIME_STATUS_PATH
            )
            return None
        response = parse_json_object(body)
        generation = response.get("generation")
        if response.get("accepted") is not True or not isinstance(generation, int):
            failure = DeliveryFailure(
                DeliveryDisposition.RETRY, "MALFORMED_RESPONSE", status_code=status
            )
            self._failure_log.record_failure(failure, path=RUNTIME_STATUS_PATH)
            return None
        self._failure_log.record_success(path=RUNTIME_STATUS_PATH)
        return generation if generation >= 0 else None


@final
class RuntimeStatusSender:
    def __init__(
        self,
        diagnostics: WorkerDiagnostics,
        facility_id: str | Mapping[str, str],
        transport: RuntimeStatusTransport,
        config: RuntimeStatusSenderConfig | None = None,
        *,
        before_publish: Callable[[], None] | None = None,
        delivery_queue: DeliveryQueue | None = None,
    ) -> None:
        resolved_config = RuntimeStatusSenderConfig() if config is None else config
        self._diagnostics = diagnostics
        self._facility_id = facility_id
        self._transport = transport
        self._before_publish = before_publish
        self._delivery_queue = delivery_queue
        self._publish_interval_sec = max(0.0, resolved_config.publish_interval_sec)
        self._initial_backoff_sec = max(0.0, resolved_config.initial_backoff_sec)
        self._max_backoff_sec = max(
            self._initial_backoff_sec,
            resolved_config.max_backoff_sec,
        )
        self._snapshots: queue.Queue[list[RelayRuntimeStatusPayload]] = queue.Queue(maxsize=1)
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
        _ = self.publish()
        self._thread = threading.Thread(
            target=self._run,
            name="runtime-status-sender",
            daemon=True,
        )
        self._thread.start()

    def publish(self) -> bool:
        snapshots = self._snapshots_for_publish()
        try:
            self._snapshots.put_nowait(snapshots)
        except queue.Full:
            try:
                _ = self._snapshots.get_nowait()
                self._snapshots.task_done()
            except queue.Empty:
                return False
            try:
                self._snapshots.put_nowait(snapshots)
            except queue.Full:
                return False
        return True

    def submit(self) -> bool:
        return self.publish()

    def publish_once(self) -> bool:
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
        pending: list[RelayRuntimeStatusPayload] | None = None
        delay = 0.0
        failures = 0
        while not self._stop_event.wait(delay):
            self._log_local_snapshot()
            _ = self.publish()
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
                backoff = self._initial_backoff_sec * (2.0 ** (failures - 1))
                delay = min(
                    self._max_backoff_sec,
                    backoff,
                )

    def _log_local_snapshot(self) -> None:
        try:
            self._diagnostics.log_snapshot()
        except Exception:
            LOGGER.warning("worker diagnostics log_snapshot failed", exc_info=True)

    def _run_before_publish(self) -> None:
        if self._before_publish is None:
            return
        try:
            self._before_publish()
        except Exception:
            LOGGER.exception("runtime status before_publish failed")

    def _take_latest(self) -> list[RelayRuntimeStatusPayload] | None:
        latest: list[RelayRuntimeStatusPayload] | None = None
        while True:
            try:
                latest = self._snapshots.get_nowait()
            except queue.Empty:
                return latest
            self._snapshots.task_done()

    def _snapshots_for_publish(self) -> list[RelayRuntimeStatusPayload]:
        self._run_before_publish()
        delivery_queue = self._delivery_queue_payload()
        if isinstance(self._facility_id, str):
            snapshots = [self._diagnostics.to_payload(self._facility_id, None, 0)]
        else:
            snapshots = self._diagnostics.to_payloads(self._facility_id, None, 0)
        if delivery_queue is not None:
            for snapshot in snapshots:
                snapshot["delivery_queue"] = delivery_queue
        return snapshots

    def _delivery_queue_payload(self) -> RelayDeliveryQueuePayload | None:
        if self._delivery_queue is None:
            return None
        snapshot = self._delivery_queue.capacity_snapshot
        return RelayDeliveryQueuePayload(
            accepted_count=snapshot.accepted_count,
            accepted_bytes=snapshot.accepted_bytes,
            max_accepted_entries=snapshot.max_accepted_entries,
            max_accepted_bytes=snapshot.max_accepted_bytes,
            by_kind={kind.value: count for kind, count in snapshot.by_kind.items()},
            dead_lettered_count=snapshot.dead_lettered_count,
            dead_lettered_bytes=snapshot.dead_lettered_bytes,
            oldest_event_accepted_at=snapshot.oldest_event_accepted_at,
        )

    def _post(self, snapshots: list[RelayRuntimeStatusPayload]) -> bool:
        for snapshot in snapshots:
            facility_id = snapshot["facility_id"]
            with self._state_lock:
                seq = self._seq_by_facility.get(facility_id, 0) + 1
                self._seq_by_facility[facility_id] = seq
                generation = self._generation_by_facility.get(facility_id)
            payload = snapshot.copy()
            payload["generation"] = generation
            payload["seq"] = seq
            accepted_generation = self._transport.send(payload)
            if accepted_generation is None:
                return False
            with self._state_lock:
                self._generation_by_facility[facility_id] = accepted_generation
                self._generation = accepted_generation
        return True


__all__ = [
    "RelayRuntimeStatusTransport",
    "RuntimeHttpRequest",
    "RuntimeStatusSender",
    "RuntimeStatusSenderConfig",
    "RuntimeStatusTransport",
    "bounded_request",
]
