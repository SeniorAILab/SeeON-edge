"""Per-camera consume loop: ``bus.inference`` -> analytics -> decision -> output.

Pure wiring stage. Business math stays where it already lives -- extraction
and tracking in ``analytics`` (:class:`CompositeExtractor`, itself gated by
the camera's :class:`~worker.pipeline.bus.Scheduler`), domain interpretation
in ``decision`` (:class:`EventAggregator`). This module only pumps taken
packets through those two calls and forwards admitted events to the sink.
"""

from __future__ import annotations

import logging
import threading
from typing import Final, final

from worker.adapters.model.errors import FatalAcceleratorError
from worker.interfaces.bus import FrameSubscription
from worker.interfaces.output import EventSink
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.decision import EventAggregator
from worker.types import FramePacket

LOGGER: Final = logging.getLogger(__name__)
DEFAULT_POLL_TIMEOUT_SEC: Final = 0.5


@final
class CameraPipelinePump:
    """Drive one camera's inference subscription through decision to output.

    ``analytics.process`` already gates extraction by the camera's
    ``Scheduler`` (only due modules run; every packet still advances the
    tracker/scene state). This loop forwards every packet taken from
    ``subscription`` through it, then through ``decision.update``, emitting
    each admitted event to ``sink``.
    """

    def __init__(
        self,
        camera_id: str,
        subscription: FrameSubscription,
        analytics: CompositeExtractor,
        decision: EventAggregator,
        sink: EventSink,
        *,
        poll_timeout_sec: float = DEFAULT_POLL_TIMEOUT_SEC,
    ) -> None:
        self._camera_id = camera_id
        self._subscription = subscription
        self._analytics = analytics
        self._decision = decision
        self._sink = sink
        self._poll_timeout_sec = poll_timeout_sec
        self._stop_event = threading.Event()
        self.failure_count = 0

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def run(self) -> None:
        while not self._stop_event.is_set():
            packet = self._subscription.take(timeout_sec=self._poll_timeout_sec)
            if packet is None:
                continue
            try:
                self._pump_one(packet)
            except FatalAcceleratorError:
                raise
            except Exception as error:  # noqa: BLE001 - per-camera boundary
                self._record_failure(error)

    def stop(self) -> None:
        self._stop_event.set()

    def _pump_one(self, packet: FramePacket) -> None:
        result = self._analytics.process(packet)
        for event in self._decision.update(result.decision_input):
            self._sink.emit(event)

    def _record_failure(self, error: Exception) -> None:
        self.failure_count += 1
        LOGGER.warning(
            "camera pipeline pump failed processing a frame",
            extra={"camera_id": self._camera_id, "error": type(error).__name__},
        )


__all__ = ["CameraPipelinePump", "DEFAULT_POLL_TIMEOUT_SEC"]
