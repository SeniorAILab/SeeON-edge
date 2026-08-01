"""YAML-runtime residue from the legacy `edge.runtime.edge_worker_config` /
`edge.runtime.edge_worker` test file of the same name, rewritten off `edge.*`
imports onto `worker.runtime.config`.

Its sibling, `test_ml_worker_yaml_config.py`, was already migrated (commit
651ca5e) onto `worker.runtime.config.loader.load_worker_config` and covers the
YAML config-schema contract: domain enable/disable shapes, night-window
validation, non-LSTM fall-model rejection, malformed/JSON config rejection,
etc. Comparing this file's original tests against that migrated coverage:

  - `test_worker_exits_nonzero_when_lstm_artifact_missing` is genuinely
    uncovered: `test_ml_worker_yaml_config.py` never exercises
    `FallModelConfig._validate_artifact_contract`'s check that
    weights/architecture/metadata files actually exist under `artifact_dir`.
    Ported below.

  - `test_worker_yaml_lstm_runtime_emits_fall_event` (full YAML config ->
    real LSTM artifact -> `edge_worker._build_supervisor` ->
    `supervisor.run(max_frames_per_camera=...)` -> asserts one emitted fall
    event) and `test_domain_detectors_inject_bed_exit_night_window` /
    `test_domain_detectors_leave_fall_without_time_gate` (domain detector
    construction from a loaded config via `edge_worker._domain_detectors`)
    exercise composition-root behavior that now lives in
    `worker/runtime/worker.py` (out of scope here -- owned by another
    migration lane) and `worker/domains/registry.py`. No equivalent
    "config -> constructed detectors / emitted event" test was found in
    tests/test_worker_fall_model_wiring.py, tests/test_worker_composition.py,
    tests/test_worker_per_camera_fall_state.py, or
    tests/test_worker_pr9_night_window.py (the last covers night-window
    *resolution*, i.e. `resolve_runtime_config`, and live `BedExitMonitor`
    updates, not construction from a full `WorkerConfig`). This is an open
    coverage gap, flagged rather than silently dropped or guessed at against
    composition-root code under active edit elsewhere.

  - `test_video_file_source_is_still_available_for_rtsp_harness_input` (a
    bare `VideoFileSource is not None` check) IS superseded:
    tests/test_sources_frame_source.py:32-76 exercises
    `worker.pipeline.ingest.video_file.VideoFileSource` with real behavioral
    coverage (frame yielding, RGB HWC image shape, frame_stride, monotonic
    non-negative time_sec, `FrameSource` protocol conformance) that strictly
    supersedes the old existence-only assertion. Not ported.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.loader import load_worker_config


def test_worker_yaml_config_rejects_lstm_artifact_missing_model_file(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "ml-worker.yaml", artifact_dir=tmp_path / "missing")

    with pytest.raises(WorkerConfigError, match="model.pt"):
        load_worker_config(config_path)


def _write_config(path: Path, *, artifact_dir: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "relay": {
                    "url": "http://127.0.0.1:8000",
                    "token": "relay-token-1",
                },
                "models": {
                    "fall": {
                        "type": "lstm",
                        "framework": "pytorch",
                        "mode": "sequence",
                        "artifact_dir": str(artifact_dir),
                        "weights": "model.pt",
                        "architecture": "arch.json",
                        "metadata": "metadata.yaml",
                        "window": 3,
                        "stride": 1,
                        "input_shape": [3, 51],
                        "operating_threshold": 0.5,
                    }
                },
                "domains": {"fall": {"enabled": True}},
                "cameras": [
                    {
                        "camera_id": "camera-1",
                        "facility_id": "facility-1",
                        "resident_id": "resident-1",
                        "rtsp_url": "rtsp://camera-1.local/trackID=2",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path
