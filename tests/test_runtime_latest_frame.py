from __future__ import annotations

import threading
import time

from worker.pipeline.output.live_view import LatestFrameStore
from worker.types.preview import OverlaySelection


def test_latest_frame_store_is_camera_keyed_and_non_consuming() -> None:
    store = LatestFrameStore()
    store.register_camera("camera-b")
    store.publish_jpeg("camera-a", b"jpeg-1", frame_index=1)

    first = store.get_latest("camera-a")
    second = store.get_latest("camera-a")
    assert first is not None
    assert second is not None
    assert first.jpeg == b"jpeg-1"
    assert second.jpeg == b"jpeg-1"
    assert store.get_latest("camera-b") is None


def test_latest_frame_store_is_known_before_and_after_registration() -> None:
    store = LatestFrameStore()
    assert store.is_known("camera-a") is False

    store.register_camera("camera-a")
    assert store.is_known("camera-a") is True
    assert store.get_latest("camera-a") is None


def test_latest_frame_store_publish_overwrites_previous_value() -> None:
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", b"jpeg-1", frame_index=1)
    store.publish_jpeg("camera-a", b"jpeg-2", frame_index=2)

    latest = store.get_latest("camera-a")
    assert latest is not None
    assert latest.jpeg == b"jpeg-2"
    assert latest.frame_index == 2


def test_latest_frame_store_seq_defaults_to_frame_index_but_is_overridable() -> None:
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", b"jpeg-1", frame_index=7)
    assert store.get_latest("camera-a").seq == 7  # type: ignore[union-attr]

    store.publish_jpeg("camera-a", b"jpeg-2", frame_index=8, seq=99)
    latest = store.get_latest("camera-a")
    assert latest is not None
    assert latest.frame_index == 8
    assert latest.seq == 99


def test_latest_frame_store_wait_for_latest_unblocks_on_publish() -> None:
    store = LatestFrameStore()
    results: list[object] = []

    def wait_for_publish() -> None:
        results.append(store.wait_for_latest("camera-a", previous=None, timeout=1.0))

    waiter = threading.Thread(target=wait_for_publish)
    waiter.start()
    time.sleep(0.05)
    store.publish_jpeg("camera-a", b"jpeg-1", frame_index=1)
    waiter.join(timeout=1.0)

    assert len(results) == 1
    assert results[0] is not None
    assert results[0].jpeg == b"jpeg-1"  # type: ignore[union-attr]


def test_latest_frame_store_wait_for_latest_times_out_without_publish() -> None:
    store = LatestFrameStore()
    assert store.wait_for_latest("camera-a", previous=None, timeout=0.05) is None


def test_latest_frame_store_viewer_counter_tracks_open_stream_connections() -> None:
    store = LatestFrameStore()
    assert store.has_viewers("camera-a") is False

    store.mark_viewer_connected("camera-a")
    assert store.has_viewers("camera-a") is True

    store.mark_viewer_connected("camera-a")
    store.mark_viewer_disconnected("camera-a")
    assert store.has_viewers("camera-a") is True

    store.mark_viewer_disconnected("camera-a")
    assert store.has_viewers("camera-a") is False
    assert store.has_viewers("camera-b") is False


def test_latest_frame_store_viewer_counter_never_goes_negative() -> None:
    store = LatestFrameStore()
    store.mark_viewer_disconnected("camera-a")
    assert store.has_viewers("camera-a") is False

    store.mark_viewer_connected("camera-a")
    store.mark_viewer_disconnected("camera-a")
    store.mark_viewer_disconnected("camera-a")
    assert store.has_viewers("camera-a") is False


def test_latest_frame_store_snapshot_demand_is_a_one_shot_flag() -> None:
    store = LatestFrameStore()
    assert store.consume_snapshot_demand("camera-a") is False

    store.request_snapshot_refresh("camera-a")
    assert store.consume_snapshot_demand("camera-a") is True
    assert store.consume_snapshot_demand("camera-a") is False
    assert store.consume_snapshot_demand("camera-b") is False


def test_latest_frame_store_selection_defaults_enabled_and_is_per_camera() -> None:
    store = LatestFrameStore()
    assert store.get_selection("camera-a") == OverlaySelection()

    store.set_selection("camera-a", OverlaySelection(person=False, bed=True))
    assert store.get_selection("camera-a") == OverlaySelection(person=False, bed=True)
    assert store.get_selection("camera-b") == OverlaySelection()

    store.set_selection("camera-a", OverlaySelection(person=True, bed=False))
    assert store.get_selection("camera-a") == OverlaySelection(person=True, bed=False)


def test_latest_frame_store_selection_change_fences_stale_publication() -> None:
    store = LatestFrameStore()
    original = store.get_selection("camera-a")
    generation = store.selection_generation("camera-a")

    store.set_selection("camera-a", OverlaySelection(person=False, bed=True))

    assert (
        store.publish_jpeg(
            "camera-a",
            b"stale",
            frame_index=1,
            expected_selection=original,
            expected_generation=generation,
        )
        is False
    )
    assert store.get_latest("camera-a") is None


def test_latest_frame_store_listener_receives_selection() -> None:
    store = LatestFrameStore()
    calls: list[tuple[str, int, OverlaySelection, bool]] = []
    store.set_demand_listener(
        lambda camera_id, viewers, selection, snapshot_requested: calls.append(
            (camera_id, viewers, selection, snapshot_requested)
        )
    )
    selection = OverlaySelection(person=False, bed=True)

    store.set_selection("camera-a", selection)

    assert calls == [("camera-a", 0, selection, True)]
