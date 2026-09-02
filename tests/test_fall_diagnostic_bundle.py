from __future__ import annotations

import hashlib
import json
import stat
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from worker.runtime.deepstream.fall_diagnostic_replay import replay_fall_diagnostic_bundle
from worker.runtime.deepstream.fall_diagnostic_writer import (
    FallDiagnosticWriter,
    build_fall_diagnostic_writer,
)
from worker.runtime.deepstream.fall_diagnostics import (
    FallDiagnosticFrame,
    FallDiagnosticRecorder,
    FallScoreSnapshot,
)


def _tensor(value: float) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(value for _ in range(51)) for _ in range(30))


def _frame(seq: int, *, triggered: bool = False) -> FallDiagnosticFrame:
    probability = 0.9 if triggered else 0.1
    previous_state = "fall" if seq == 31 else "clear"
    current_state = "fall" if triggered else "clear"
    return FallDiagnosticFrame(
        source_pts=seq * 10,
        source_seq=seq,
        native_publish_seq=seq + 100,
        source_generation=2,
        stream_epoch=3,
        poses=(tuple((index, index + 1, 0.9) for index in range(17)),),
        boxes=((1, 2, 30, 40, 0.8),),
        track_ids=(7,),
        live_track_ids=(7,),
        score=FallScoreSnapshot(7, _tensor(probability), probability, "fresh"),
        previous_state=previous_state,
        current_state=current_state,
        triggered=triggered,
    )


def _capture_bytes() -> tuple[list[bytes], FallDiagnosticRecorder]:
    captured: list[bytes] = []
    writer = FallDiagnosticWriter(persist=captured.append)
    writer.start()
    recorder = FallDiagnosticRecorder("slot-00", writer)
    for seq in range(35):
        recorder.record(_frame(seq, triggered=seq == 30))
    writer.stop()
    return captured, recorder


def test_bundle_is_allowlisted_bounded_and_byte_deterministic() -> None:
    first, first_recorder = _capture_bytes()
    second, _ = _capture_bytes()

    assert len(first) == 1
    assert first == second
    payload = json.loads(first[0])
    assert payload.keys() == {"camera_slot", "frames", "schema_version", "threshold"}
    assert payload["camera_slot"] == "slot-00"
    assert len(payload["frames"]) == 35
    assert payload["frames"][30]["triggered"] is True
    assert first_recorder.stats().completed_bundles == 1
    rendered = first[0].decode()
    for forbidden in ("rtsp://", "camera-a", "facility", "token", "credential", "image"):
        assert forbidden not in rendered.lower()


def test_replay_recomputes_fresh_scores_and_latch_transitions() -> None:
    captured, _ = _capture_bytes()

    result = replay_fall_diagnostic_bundle(captured[0], predict=lambda tensor: tensor[0][0])

    assert result.sha256 == hashlib.sha256(captured[0]).hexdigest()
    assert result.frame_count == 35
    assert result.onset_sequences == (30,)


def test_replay_recomputes_cached_score_without_prior_bundle_state() -> None:
    captured, _ = _capture_bytes()
    payload = json.loads(captured[0])
    payload["frames"][0]["score"]["provenance"] = "cached"
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    result = replay_fall_diagnostic_bundle(raw, predict=lambda tensor: tensor[0][0])

    assert result.onset_sequences == (30,)


def test_replay_rejects_non_allowlisted_nested_fields() -> None:
    captured, _ = _capture_bytes()
    payload = json.loads(captured[0])
    payload["frames"][0]["camera_id"] = "camera-a"
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(ValueError, match="frame fields"):
        replay_fall_diagnostic_bundle(raw, predict=lambda tensor: tensor[0][0])


def test_replay_rejects_cross_epoch_bundle() -> None:
    captured, _ = _capture_bytes()
    payload = json.loads(captured[0])
    payload["frames"][30]["stream_epoch"] = 4
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(ValueError, match="continuity"):
        replay_fall_diagnostic_bundle(raw, predict=lambda tensor: tensor[0][0])


def test_early_onset_is_skipped_until_thirty_pre_onset_frames_exist() -> None:
    captured: list[bytes] = []
    writer = FallDiagnosticWriter(persist=captured.append)
    writer.start()
    recorder = FallDiagnosticRecorder("slot-00", writer)
    recorder.record(_frame(0, triggered=True))
    for seq in range(1, 31):
        recorder.record(_frame(seq))
    recorder.record(_frame(31, triggered=True))
    for seq in range(32, 36):
        recorder.record(_frame(seq))
    writer.stop()

    assert len(captured) == 1
    payload = json.loads(captured[0])
    assert payload["frames"][30]["source_seq"] == 31
    assert recorder.stats().skipped_onsets == 1


def test_fixed_person_bound_rejects_observably_without_raising() -> None:
    captured: list[bytes] = []
    writer = FallDiagnosticWriter(persist=captured.append)
    writer.start()
    recorder = FallDiagnosticRecorder("slot-00", writer)
    frame = _frame(0)
    oversized = replace(
        frame,
        poses=frame.poses * 17,
        boxes=frame.boxes * 17,
        track_ids=tuple(range(17)),
        live_track_ids=tuple(range(17)),
    )

    recorder.record(oversized)
    writer.stop()

    assert captured == []
    assert recorder.stats().rejected_frames == 1


def test_stream_discontinuity_discards_pre_onset_history() -> None:
    captured: list[bytes] = []
    writer = FallDiagnosticWriter(persist=captured.append)
    writer.start()
    recorder = FallDiagnosticRecorder("slot-00", writer)
    for seq in range(30):
        recorder.record(_frame(seq))

    restarted = replace(
        _frame(30, triggered=True),
        source_generation=3,
        stream_epoch=4,
    )
    recorder.record(restarted)
    writer.stop()

    assert captured == []
    assert recorder.stats().continuity_resets == 1
    assert recorder.stats().skipped_onsets == 1


def test_non_finite_diagnostic_data_is_counted_and_never_raises() -> None:
    captured: list[bytes] = []
    writer = FallDiagnosticWriter(persist=captured.append)
    writer.start()
    recorder = FallDiagnosticRecorder("slot-00", writer)
    frame = _frame(0)
    assert frame.score is not None

    recorder.record(replace(frame, score=replace(frame.score, probability=float("nan"))))
    writer.stop()

    assert captured == []
    assert recorder.stats().rejected_frames == 1


def test_slow_or_failed_filesystem_never_blocks_or_raises_on_submit() -> None:
    entered = threading.Event()
    release = threading.Event()

    def slow_failure(_payload: bytes) -> None:
        entered.set()
        assert release.wait(timeout=2.0)
        raise OSError("fixture write failure")

    writer = FallDiagnosticWriter(persist=slow_failure, max_pending=1)
    writer.start()
    assert writer.submit(b"{}") is True
    assert entered.wait(timeout=1.0)
    assert writer.submit(b'{"queued":1}') is True
    assert writer.submit(b'{"dropped":1}') is False
    release.set()
    writer.stop()

    stats = writer.stats()
    assert stats.write_failures == 2
    assert stats.queue_drops == 1


def test_serialization_failure_is_counted_on_writer_thread() -> None:
    writer = FallDiagnosticWriter(persist=lambda _payload: None)
    writer.start()
    assert writer.submit_bundle({"value": float("nan")}) is True
    writer.stop()

    assert writer.stats().write_failures == 1


def test_writer_uses_private_content_addressed_local_file(tmp_path: Path) -> None:
    writer = FallDiagnosticWriter(root=tmp_path)
    writer.start()
    payload = b'{"schema_version":1}'
    assert writer.submit(payload)
    writer.stop()

    expected = tmp_path / f"{hashlib.sha256(payload).hexdigest()}.json"
    assert expected.read_bytes() == payload
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(expected.stat().st_mode) == 0o600


def test_diagnostics_are_disabled_by_default_and_require_exact_opt_in(tmp_path: Path) -> None:
    assert build_fall_diagnostic_writer({}, tmp_path) is None
    assert (
        build_fall_diagnostic_writer(
            {"SEEON_FALL_DIAGNOSTICS_ENABLED": "true"},
            tmp_path,
        )
        is None
    )

    writer = build_fall_diagnostic_writer({"SEEON_FALL_DIAGNOSTICS_ENABLED": "1"}, tmp_path)
    assert writer is not None
    writer.stop()
