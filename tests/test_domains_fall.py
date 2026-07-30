from __future__ import annotations

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    DetectionLabel,
    FrameObservation,
)
from edge.domains.fall.detector import FallEventLatch
from edge.domains.fall.schema import FallEvent
from edge.perception.domain_input import DomainInput


def domain_input(observation: FrameObservation, time_sec: float) -> DomainInput:
    return DomainInput(
        observation=observation,
        frame_width=1,
        frame_height=1,
        live_track_ids=(),
        time_sec=time_sec,
        frame_index=0,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY, None),
    )


class TestFallEventLatch:
    def test_no_fall_no_event(self) -> None:
        latch = FallEventLatch()
        assert not any(latch.update_signal(False, t * 0.1) for t in range(10))
        assert latch.event_count == 0
        assert latch.first_event_sec is None

    def test_single_onset_records_time_and_counts_once(self) -> None:
        latch = FallEventLatch()
        signal = [False, False, True, True, True, False]
        onsets = [latch.update_signal(s, i * 0.5) for i, s in enumerate(signal)]
        assert onsets == [False, False, True, False, False, False]
        assert latch.event_count == 1
        assert latch.first_event_sec == 1.0

    def test_reentry_counts_new_event_keeps_first_time(self) -> None:
        latch = FallEventLatch()
        signal = [False, True, False, False, True, True]
        for i, s in enumerate(signal):
            latch.update_signal(s, float(i))
        assert latch.event_count == 2
        assert latch.first_event_sec == 1.0

    def test_fall_on_first_frame_is_an_onset(self) -> None:
        latch = FallEventLatch()
        assert latch.update_signal(True, 0.0) is True
        assert latch.first_event_sec == 0.0

    def test_update_event_returns_schema_only_on_onset(self) -> None:
        latch = FallEventLatch()
        assert latch.update_event(False, 0.0) is None
        event = latch.update_event(True, 0.5)
        assert event == FallEvent(event_count=1, onset_sec=0.5, first_event_sec=0.5)
        assert latch.update_event(True, 1.0) is None

    def test_runtime_observation_update_returns_domain_event_tuple(self) -> None:
        latch = FallEventLatch()
        observation = FrameObservation(
            detections=((), (DetectionLabel(text="FALL", confidence=0.87, is_fall=True),))
        )

        assert latch.update(domain_input(observation, 1.25)) == (
            {
                "domain": "fall",
                "event_type": "fall",
                "identity": 1,
                "probability": 0.87,
                "time_sec": 1.25,
            },
        )
        assert latch.update(domain_input(observation, 1.5)) == ()
