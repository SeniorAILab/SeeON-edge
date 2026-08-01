from __future__ import annotations

from worker.domains.fall.detector import FallEventLatch
from worker.domains.fall.schema import FallEvent

# The module-level test_runtime_observation_update_returns_domain_event_tuple
# (dict-shaped .update() output) is superseded by
# tests/test_worker_fall_decider.py:71-101
# (test_repeated_positive_frames_emit_one_typed_rising_edge), which asserts
# the same rising-edge .update() contract against the current BusinessEvent
# return type instead of the retired dict payload; not duplicated here.
#
# update_signal()/update_event() are NOT a retired surface: worker's
# FallEventLatch.update() now routes observations through
# FallWindowClassifier first (a materially different contract, covered by
# test_worker_fall_classifier.py and test_worker_fall_decider.py), but the
# low-level update_signal()/update_event() methods below it are unchanged
# verbatim (worker/domains/fall/detector.py:55-75, only the private
# `_previous_fall` field was renamed from edge's `_prev_fall`). Ported
# directly against that surface.


class TestFallEventLatch:
    def test_no_fall_no_event(self) -> None:
        latch = FallEventLatch(None, camera_id="camera-1", facility_id="facility-1")
        assert not any(latch.update_signal(False, t * 0.1) for t in range(10))
        assert latch.event_count == 0
        assert latch.first_event_sec is None

    def test_single_onset_records_time_and_counts_once(self) -> None:
        latch = FallEventLatch(None, camera_id="camera-1", facility_id="facility-1")
        signal = [False, False, True, True, True, False]
        onsets = [latch.update_signal(s, i * 0.5) for i, s in enumerate(signal)]
        assert onsets == [False, False, True, False, False, False]
        assert latch.event_count == 1
        assert latch.first_event_sec == 1.0

    def test_reentry_counts_new_event_keeps_first_time(self) -> None:
        latch = FallEventLatch(None, camera_id="camera-1", facility_id="facility-1")
        signal = [False, True, False, False, True, True]
        for i, s in enumerate(signal):
            latch.update_signal(s, float(i))
        assert latch.event_count == 2
        assert latch.first_event_sec == 1.0

    def test_fall_on_first_frame_is_an_onset(self) -> None:
        latch = FallEventLatch(None, camera_id="camera-1", facility_id="facility-1")
        assert latch.update_signal(True, 0.0) is True
        assert latch.first_event_sec == 0.0

    def test_update_event_returns_schema_only_on_onset(self) -> None:
        latch = FallEventLatch(None, camera_id="camera-1", facility_id="facility-1")
        assert latch.update_event(False, 0.0) is None
        event = latch.update_event(True, 0.5)
        assert event == FallEvent(event_count=1, onset_sec=0.5, first_event_sec=0.5)
        assert latch.update_event(True, 1.0) is None
