"""EdgeWorkerSupervisor -> IngestSupervisor migration.

Every edge assertion this file used to carry has one of four dispositions
(reported in full to the orchestrator alongside this change); this file keeps
only the one guarantee that has no existing worker-side test:

- one-offline-camera isolation, RTSP degraded->ready recovery, and the
  "runner never touches the capture thread" guarantee are all superseded by
  tests/test_worker_ingest_lifecycle.py and tests/test_worker_camera_pipeline_pump.py
  (IngestSupervisor wires ingest and pump loops as physically separate threads,
  so extraction structurally cannot run on the capture thread -- a stronger
  guarantee than edge's callback-based `test_runner_called_by_scheduler_only`).
- the two-bed-boxes-not-misread-as-pose-pair guarantee is now structural:
  worker/pipeline/analytics/merge.py dispatches by `module_name` through a
  typed `_RESULT_MERGERS` table (singledispatch registered against each
  concrete RunnerResult), so edge's isinstance-based branch-guessing bug class
  cannot recur.
- heartbeat gating (only a READY transition can reach the relay) is ported
  below directly against `HeartbeatReporter`, which now owns that guarantee.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

import worker.runtime.worker as worker_module
from shared.events.evidence_http_transport import HttpResult
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.worker import HeartbeatReporter


def _config() -> tuple[WorkerConfig, CameraRuntimeConfig]:
    worker_config = WorkerConfig.model_validate(
        {
            "version": 3,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://example.test/camera-1",
                    "heartbeat_interval_sec": 30.0,
                }
            ],
        }
    )
    return worker_config, worker_config.cameras[0]


def test_heartbeat_reporter_sends_only_on_the_ready_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors edge's `_send_heartbeats` READY-only gate
    (edge/runtime/edge_worker_supervisor.py:147-154). Worker moved the same
    guarantee onto `HeartbeatReporter` itself: `mark_starting`/`mark_degraded`
    are no-ops (worker/runtime/worker.py:114-116,144-145), so only `mark_ready`
    can ever reach the relay.
    """
    requests: list[str] = []

    def bounded_request(
        url: str,
        _method: str,
        _headers: dict[str, str],
        _data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        requests.append(url)
        return 204, {}, b""

    monkeypatch.setattr(worker_module, "bounded_request", bounded_request)
    worker_config, camera = _config()
    reporter = HeartbeatReporter(worker_config, camera)

    reporter.mark_starting("camera-1")
    reporter.mark_degraded("camera-1", category="rtsp_reconnecting")
    assert requests == []

    reporter.mark_ready("camera-1")
    assert requests == ["http://relay.test/api/v1/relay/heartbeat"]
