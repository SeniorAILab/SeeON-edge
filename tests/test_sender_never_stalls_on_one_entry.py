from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shared.events.delivery_queue import DeliveryQueue, EventEntry
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from worker.pipeline.output.evidence.evidence_sender import (
    EvidenceSender,
    SenderConfig,
)


def _entry(index: int) -> EventEntry:
    return EventEntry(
        edge_event_id=f"11111111-1111-4111-8111-{index:012d}",
        event_type="fall",
        detected_at="2026-08-22T00:00:00Z",
        camera_id="camera-1",
        facility_id="facility-1",
        decision_trace=b"{}",
        values=b'{"probability": 0.9}',
    )


class _PoisonTransport:
    def __init__(
        self,
        poisoned: str,
        disposition: DeliveryDisposition = DeliveryDisposition.PERMANENT,
    ) -> None:
        self._poisoned = poisoned
        self._disposition = disposition
        self.delivered: list[str] = []

    def send_event(self, payload: Any, edge_event_id: str) -> Any:
        if edge_event_id == self._poisoned:
            return DeliveryFailure(disposition=self._disposition, code="HTTP_500")
        self.delivered.append(edge_event_id)
        return _Receipt(edge_event_id)


class _Receipt:
    def __init__(self, edge_event_id: str) -> None:
        self.edge_event_id = edge_event_id
        self.event_id = f"backend-{edge_event_id}"


@pytest.fixture(name="queue_dir")
def _queue_dir(tmp_path: Path) -> Path:
    return tmp_path / "delivery-queue"


def test_a_permanently_failing_entry_does_not_block_the_ones_behind_it(
    queue_dir: Path,
) -> None:
    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    poisoned = entries[0].edge_event_id
    transport = _PoisonTransport(poisoned)
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )

    for _ in range(40):
        sender.run_once()

    assert poisoned not in transport.delivered, "the poisoned entry should never deliver"
    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} were delivered; a single failing entry "
        f"halted the evidence queue behind it"
    )


def test_the_exhausted_entry_is_retained_not_deleted(queue_dir: Path) -> None:
    queue = DeliveryQueue(queue_dir)
    entry = _entry(1)
    assert queue.try_admit(entry).accepted

    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=_PoisonTransport(entry.edge_event_id),
    )
    for _ in range(40):
        sender.run_once()

    assert queue.capacity_snapshot.accepted_count == 0, "the queue never drained"
    retained = sorted(queue.dead_letter_directory.iterdir())
    assert len(retained) == 1, "the exhausted entry was discarded rather than retained"


def test_a_transient_failure_is_never_dead_lettered(queue_dir: Path) -> None:
    queue = DeliveryQueue(queue_dir)
    entry = _entry(1)
    assert queue.try_admit(entry).accepted

    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=_PoisonTransport(entry.edge_event_id, disposition=DeliveryDisposition.RETRY),
    )
    for _ in range(60):
        sender.run_once()

    snapshot = queue.capacity_snapshot
    assert snapshot.accepted_count == 1, "the entry was removed during an outage"
    assert snapshot.dead_lettered_count == 0, (
        "a transient relay failure dead-lettered live evidence; an outage would "
        "discard the entire queue instead of holding it"
    )


def test_a_transiently_failing_entry_does_not_block_the_ones_behind_it(
    queue_dir: Path,
) -> None:
    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    transport = _PoisonTransport(entries[0].edge_event_id, disposition=DeliveryDisposition.RETRY)
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )

    for _ in range(40):
        sender.run_once()

    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} delivered; a transiently failing entry "
        f"starved the queue behind it, and no attempt budget can rescue that "
        f"because transient failures must retry forever"
    )
    assert queue.capacity_snapshot.dead_lettered_count == 0


def test_a_full_retention_area_does_not_stall_the_live_queue(
    queue_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shared.events import delivery_queue as module

    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    monkeypatch.setattr(module, "MAX_DEAD_LETTERED_ENTRIES", 0)

    transport = _PoisonTransport(
        entries[0].edge_event_id, disposition=DeliveryDisposition.PERMANENT
    )
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )
    for _ in range(60):
        sender.run_once()

    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} delivered; a full retention area turned the "
        f"undeliverable entry into a permanent stall of the whole queue"
    )
    assert queue.capacity_snapshot.accepted_count == 1
    assert queue.capacity_snapshot.dead_lettered_count == 0


def test_a_422_with_retention_full_does_not_stall_the_queue(
    queue_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shared.events import delivery_queue as module

    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    monkeypatch.setattr(module, "MAX_DEAD_LETTERED_ENTRIES", 0)

    class _RefusingTransport(_PoisonTransport):
        def send_event(self, payload: Any, edge_event_id: str) -> Any:
            if edge_event_id == self._poisoned:
                return DeliveryFailure(
                    disposition=DeliveryDisposition.PERMANENT,
                    code="UNPROCESSABLE",
                    status_code=422,
                )
            self.delivered.append(edge_event_id)
            return _Receipt(edge_event_id)

    transport = _RefusingTransport(entries[0].edge_event_id)
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )
    for _ in range(60):
        sender.run_once()

    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} delivered; a 422 that could not be retained "
        f"was reselected forever and starved every newer alert behind it"
    )
    assert queue.capacity_snapshot.accepted_count == 1
    assert queue.capacity_snapshot.dead_lettered_count == 0


def test_an_entry_that_raises_does_not_starve_the_queue(queue_dir: Path) -> None:
    class _RaisingTransport:
        def __init__(self, poisoned: str) -> None:
            self._poisoned = poisoned
            self.delivered: list[str] = []

        def send_event(self, payload: Any, edge_event_id: str) -> Any:
            if edge_event_id == self._poisoned:
                raise RuntimeError("entry payload is corrupt")
            self.delivered.append(edge_event_id)
            return _Receipt(edge_event_id)

    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    transport = _RaisingTransport(entries[0].edge_event_id)
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )
    for _ in range(40):
        sender.run_once()

    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} delivered; one raising entry starved the queue behind it"
    )


def test_unwritable_retention_does_not_stall_the_queue(queue_dir: Path) -> None:
    import errno
    from unittest.mock import patch

    import shared.events.delivery_queue as queue_module

    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    class _RefusingTransport(_PoisonTransport):
        def send_event(self, payload: Any, edge_event_id: str) -> Any:
            if edge_event_id == self._poisoned:
                return DeliveryFailure(
                    disposition=DeliveryDisposition.PERMANENT,
                    code="UNPROCESSABLE",
                    status_code=422,
                )
            self.delivered.append(edge_event_id)
            return _Receipt(edge_event_id)

    transport = _RefusingTransport(entries[0].edge_event_id)
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )

    swallowed: list[BaseException] = []
    with patch.object(queue_module.os, "link", side_effect=OSError(errno.ENOSPC, "no space")):
        for _ in range(40):
            try:
                sender.run_once()
            except Exception as caught:  # noqa: BLE001
                swallowed.append(caught)

    assert not swallowed, (
        f"run_once raised {swallowed[0]!r}; retention I/O failure must be handled "
        f"inside the sender, not left to the loop that cannot defer the entry"
    )

    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} delivered; an unwritable retention area "
        f"blocked every newer resident event behind the refused one"
    )
    assert queue.capacity_snapshot.accepted_count == 1


def test_a_failing_acknowledge_does_not_monopolise_the_queue(queue_dir: Path) -> None:
    import errno
    from unittest.mock import patch

    import shared.events.delivery_queue as queue_module

    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    class _AlwaysDelivers:
        def __init__(self) -> None:
            self.sent: list[str] = []

        def send_event(self, payload: Any, edge_event_id: str) -> Any:
            self.sent.append(edge_event_id)
            return _Receipt(edge_event_id)

    transport = _AlwaysDelivers()
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )

    real_unlink = queue_module.Path.unlink

    def _failing_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.parent.name == "delivery-queue":
            raise OSError(errno.EIO, "io error")
        real_unlink(self, *args, **kwargs)

    swallowed: list[BaseException] = []
    with patch.object(queue_module.Path, "unlink", _failing_unlink):
        for _ in range(30):
            try:
                sender.run_once()
            except Exception as caught:  # noqa: BLE001
                swallowed.append(caught)

    assert not swallowed, f"run_once raised {swallowed[0]!r} instead of handling it"
    assert set(transport.sent) == {entry.edge_event_id for entry in entries}, (
        f"only {sorted(set(transport.sent))} reached the backend; a failing "
        f"removal monopolised the queue and starved every newer event"
    )
