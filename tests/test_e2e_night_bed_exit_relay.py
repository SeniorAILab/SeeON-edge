"""Real-stack E2E: synthetic RTSP -> worker ingest/bus/perception/decision ->
real local FastAPI backend relay routes.

Per the v2 migration plan (ruling R5), this exercises the actual production
composition -- ``WorkerRuntime`` with its default (non-fake) ingest loop
factory reading a real RTSP stream published by ffmpeg through a real
MediaMTX server, and a real ``backend.app`` FastAPI instance served over
uvicorn -- with only the model-serving boundary (pose/person/bed/fall
runners) replaced by deterministic scripted doubles. Model artifacts are
intentionally absent: this is a wiring/relay proof, not an accuracy proof.

Assertions covered:
  * one night bed-exit event reaches ``POST /api/v1/relay/alerts``
  * one fall event reaches the relay despite two rising edges in the
    scripted probability sequence (``IncidentManager``'s 30s cooldown
    collapses the second detection -- proves duplicate/cooldown suppression)
  * the same bed-exit geometry produces zero alerts in the daytime window
    (``BedExitConfig.night_window`` suppression)
  * a heartbeat and a runtime-status payload reach the relay for every run
  * the night run's clip finalizes as a real H264/yuv420p/faststart MP4
  * both relayed alerts carry a real audit envelope and a bounded JPEG
    snapshot (``AlertEvidenceAttacher``, wired unconditionally into every
    camera's pump)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from e2e_worker_relay_fixtures import (
    FfmpegPublisher,
    LiveBackend,
    MediaMtxProcess,
    RecordedAlert,
    ScriptedServingClient,
    WorkerRunHandle,
    bed_exit_serving_client,
    build_worker_config,
    fall_serving_client,
    free_tcp_port,
    night_window_excluding_now,
    night_window_including_now,
    start_worker_runtime,
    stop_worker_runtime,
    wait_until,
)

RELAY_TOKEN = "e2e-relay-token"  # noqa: S105 - fixture-scoped test constant, not a real secret

NIGHT_CAMERA_ID = "camera-e2e-night"
NIGHT_FACILITY_ID = "facility-e2e-night"
FALL_CAMERA_ID = "camera-e2e-fall"
FALL_FACILITY_ID = "facility-e2e-fall"
DAY_CAMERA_ID = "camera-e2e-day"
DAY_FACILITY_ID = "facility-e2e-day"


@pytest.fixture(scope="module")
def mediamtx() -> Iterator[MediaMtxProcess]:
    process = MediaMtxProcess(rtsp_port=free_tcp_port(), path_names=("night", "fall", "day"))
    try:
        yield process
    finally:
        process.stop()


@contextmanager
def _run_scenario(
    *,
    mediamtx: MediaMtxProcess,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path_name: str,
    camera_id: str,
    facility_id: str,
    domains: dict[str, object],
    serving_client: ScriptedServingClient,
) -> Iterator[tuple[LiveBackend, WorkerRunHandle, Path]]:
    state_dir = tmp_path / "state"
    clip_store_dir = tmp_path / "clips"
    monkeypatch.setenv("ML_WORKER_STATE_DIR", str(state_dir))
    monkeypatch.setenv(
        "ML_WORKER_EVIDENCE_OUTBOX_PATH", str(state_dir / "evidence-outbox.sqlite3")
    )
    monkeypatch.setenv("ML_WORKER_EVENT_CLIP_EXPORT_ENABLED", "1")

    publisher = FfmpegPublisher(url=mediamtx.rtsp_url(path_name))
    live_backend = LiveBackend(
        relay_token=RELAY_TOKEN,
        camera_inventory={
            camera_id: {"camera_id": camera_id, "facility_id": facility_id, "resident_id": None}
        },
        state_dir=state_dir,
    )
    handle: WorkerRunHandle | None = None
    try:
        publisher.ensure_alive(settle_sec=1.0)
        config = build_worker_config(
            relay_url=live_backend.base_url,
            relay_token=RELAY_TOKEN,
            camera_id=camera_id,
            facility_id=facility_id,
            rtsp_url=mediamtx.rtsp_url(path_name),
            domains=domains,
        )
        handle = start_worker_runtime(
            config, serving_client, state_dir=state_dir, clip_store_dir=clip_store_dir
        )
        yield live_backend, handle, clip_store_dir
    finally:
        if handle is not None:
            stop_worker_runtime(handle)
        publisher.stop()
        live_backend.stop()


def test_night_bed_exit_reaches_relay_with_heartbeat_status_and_finalized_clip(
    mediamtx: MediaMtxProcess, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _run_scenario(
        mediamtx=mediamtx,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        path_name="night",
        camera_id=NIGHT_CAMERA_ID,
        facility_id=NIGHT_FACILITY_ID,
        domains={"bed_exit": {"night_window": night_window_including_now()}},
        serving_client=bed_exit_serving_client(),
    ) as (live_backend, _handle, clip_store_dir):
        wait_until(
            lambda: live_backend.ingest_client.heartbeats >= 1,
            timeout=20.0,
            what="heartbeat delivery",
        )
        wait_until(
            lambda: any(
                alert.event_type == "bed-exit"
                for alert in live_backend.ingest_client.snapshot_alerts()
            ),
            timeout=25.0,
            what="bed-exit alert relay delivery",
        )
        wait_until(
            lambda: live_backend.runtime_status_snapshot(NIGHT_FACILITY_ID) is not None,
            timeout=10.0,
            what="runtime status arrival",
        )
        alerts = [
            alert
            for alert in live_backend.ingest_client.snapshot_alerts()
            if alert.event_type == "bed-exit"
        ]
        assert len(alerts) == 1
        assert alerts[0].edge_event_id is not None
        assert alerts[0].probability == 1.0
        _assert_alert_carries_audit_and_snapshot(alerts[0])

        clip_path = _wait_for_finalized_clip(clip_store_dir, timeout=15.0)
        _assert_h264_yuv420p_faststart(clip_path)


def test_fall_reaches_relay_once_despite_two_rising_edges(
    mediamtx: MediaMtxProcess, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _run_scenario(
        mediamtx=mediamtx,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        path_name="fall",
        camera_id=FALL_CAMERA_ID,
        facility_id=FALL_FACILITY_ID,
        domains={"fall": {}},
        serving_client=fall_serving_client(),
    ) as (live_backend, _handle, _clip_store_dir):
        wait_until(
            lambda: live_backend.ingest_client.heartbeats >= 1,
            timeout=20.0,
            what="heartbeat delivery",
        )
        wait_until(
            lambda: any(
                alert.event_type == "fall"
                for alert in live_backend.ingest_client.snapshot_alerts()
            ),
            timeout=25.0,
            what="fall alert relay delivery",
        )
        wait_until(
            lambda: live_backend.runtime_status_snapshot(FALL_FACILITY_ID) is not None,
            timeout=10.0,
            what="runtime status arrival",
        )
        # Let the scripted second rising edge play out; IncidentManager's
        # cooldown key for the fall domain is (camera_id, domain, event_type)
        # -- independent of the classifier's per-onset identity -- so within
        # its 30s window the second onset must be swallowed, not relayed.
        time.sleep(3.0)
        alerts = [
            alert
            for alert in live_backend.ingest_client.snapshot_alerts()
            if alert.event_type == "fall"
        ]
        assert len(alerts) == 1
        assert alerts[0].edge_event_id is not None
        _assert_alert_carries_audit_and_snapshot(alerts[0])


def test_daytime_bed_exit_is_suppressed_before_relay(
    mediamtx: MediaMtxProcess, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _run_scenario(
        mediamtx=mediamtx,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        path_name="day",
        camera_id=DAY_CAMERA_ID,
        facility_id=DAY_FACILITY_ID,
        domains={"bed_exit": {"night_window": night_window_excluding_now()}},
        serving_client=bed_exit_serving_client(),
    ) as (live_backend, _handle, _clip_store_dir):
        wait_until(
            lambda: live_backend.ingest_client.heartbeats >= 1,
            timeout=20.0,
            what="heartbeat delivery",
        )
        # Same exit geometry as the night run; give it a generous settle
        # window to fully play out before asserting nothing was relayed.
        time.sleep(6.0)
        assert live_backend.ingest_client.snapshot_alerts() == ()


def _assert_alert_carries_audit_and_snapshot(alert: RecordedAlert) -> None:
    """AlertEvidenceAttacher (worker/pipeline/output/evidence_attacher.py) is
    unconditionally wired into every camera's pump (worker/runtime/worker.py
    ``_build_camera``), so every admitted event should carry a real audit
    envelope and a bounded JPEG snapshot rendered from the actual decoded
    frame by the time it reaches the relay.
    """
    assert alert.audit is not None
    assert alert.audit.get("detector_version") == "worker-domain-detectors-v1"
    assert "clock_source" in alert.audit
    assert alert.snapshot_bytes is not None
    assert len(alert.snapshot_bytes) > 0


def _wait_for_finalized_clip(store_dir: Path, *, timeout: float) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches = sorted(store_dir.rglob("clip.mp4"))
        if matches:
            return matches[-1]
        time.sleep(0.2)
    raise TimeoutError(f"no finalized clip.mp4 appeared under {store_dir} within {timeout}s")


def _assert_h264_yuv420p_faststart(clip_path: Path) -> None:
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None, "ffprobe must be on PATH for the clip-finalize assertion"
    result = subprocess.run(  # noqa: S603 - fixed local test binary, no shell
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt", "-of", "json", str(clip_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probed = json.loads(result.stdout)["streams"][0]
    assert probed["codec_name"] == "h264"
    assert probed["pix_fmt"] == "yuv420p"
    assert _moov_precedes_mdat(clip_path), "clip is missing a faststart (moov-before-mdat) layout"


def _moov_precedes_mdat(path: Path) -> bool:
    moov_offset: int | None = None
    mdat_offset: int | None = None
    size = path.stat().st_size
    with path.open("rb") as handle:
        offset = 0
        while offset < size:
            handle.seek(offset)
            header = handle.read(8)
            if len(header) < 8:
                break
            box_size = int.from_bytes(header[0:4], "big")
            box_type = header[4:8].decode("ascii", errors="replace")
            if box_type == "moov" and moov_offset is None:
                moov_offset = offset
            if box_type == "mdat" and mdat_offset is None:
                mdat_offset = offset
            if box_size == 1:
                handle.seek(offset + 8)
                box_size = int.from_bytes(handle.read(8), "big")
            if box_size <= 0:
                break
            offset += box_size
    return moov_offset is not None and mdat_offset is not None and moov_offset < mdat_offset
