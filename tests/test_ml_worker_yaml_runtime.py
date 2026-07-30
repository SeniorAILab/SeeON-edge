from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
import torch
import yaml

from contracts.frame import Frame
from contracts.runner import bed_result, pose_result
from edge.runtime import edge_worker
from edge.runtime.edge_worker_config import EdgeWorkerConfigError, load_edge_worker_config
from edge.sources.video_file import VideoFileSource


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, event: dict[str, object]) -> None:
        self.events.append(event)


class _PoseRunner:
    def run(self, image: np.ndarray):
        del image
        keypoints = tuple((20.0, 20.0, 0.9) for _ in range(17))
        return pose_result((keypoints,), ((10, 10, 40, 60, 0.95),))




class _BedRunner:
    def run(self, image: np.ndarray):
        del image
        return bed_result(())


class _SlowFrameSource:
    def __init__(self, frames: tuple[Frame, ...]) -> None:
        self.frames = frames

    def __iter__(self):
        for frame in self.frames:
            time.sleep(0.02)
            yield frame


def test_worker_yaml_lstm_runtime_emits_fall_event(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = _write_lstm_artifact(tmp_path / "models" / "fall" / "lstm")
    config_path = _write_config(tmp_path / "ml-worker.yaml", artifact_dir=artifact_dir)
    config = load_edge_worker_config(config_path)
    monkeypatch.setenv("ML_WORKER_STATE_DIR", str(tmp_path / "worker-state"))
    sink = _Sink()
    frames = tuple(
        Frame(index=index, time_sec=float(index), image=np.zeros((80, 80, 3), dtype=np.uint8))
        for index in range(3)
    )

    monkeypatch.setattr(edge_worker, "RTSPSource", lambda _url, **_kwargs: _SlowFrameSource(frames))
    monkeypatch.setattr(
        edge_worker, "_relay_client", lambda _config, _camera, _config_version=0: sink
    )
    monkeypatch.setattr(edge_worker.DEFAULT_REGISTRY, "create", _create_runner)

    supervisor = edge_worker._build_supervisor(
        config, edge_worker.StatusStore(), device="cpu", decode="opencv"
    )
    assert supervisor.run(max_frames_per_camera=3, heartbeat_on_start=False) == {"camera-1": 3}

    assert len(sink.events) == 1
    event = dict(sink.events[0])
    assert event.pop("snapshot_jpeg", None) is not None
    assert UUID(str(event.pop("edge_event_id"))).version == 4
    assert str(event.pop("detected_at")).endswith("Z")
    assert event == {
        "domain": "fall",
        "event_type": "fall",
        "identity": 1,
        "probability": pytest.approx(0.99, abs=0.01),
        "time_sec": 2.0,
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "clip_id": None,
        "audit": {
            "clock_source": "edge_wall_clock",
            "model_version": "lstm",
            "detector_version": "worker-domain-detectors-v1",
            "operating_threshold": 0.5,
        },
    }


def test_worker_exits_nonzero_when_lstm_artifact_missing(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "ml-worker.yaml", artifact_dir=tmp_path / "missing")

    with pytest.raises(EdgeWorkerConfigError, match="model.pt"):
        load_edge_worker_config(config_path)


def test_domain_detectors_inject_bed_exit_night_window(tmp_path: Path) -> None:
    artifact_dir = _write_lstm_artifact(tmp_path / "models" / "fall" / "lstm")
    config_path = _write_config(tmp_path / "ml-worker.yaml", artifact_dir=artifact_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["domains"] = {
        "fall": {"enabled": True},
        "bed_exit": {
            "enabled": True,
            "night_window": {"start": "21:00", "end": "05:00", "tz": "Asia/Seoul"},
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_edge_worker_config(config_path)

    detectors = edge_worker._domain_detectors(config)

    assert tuple(detector.__class__.__name__ for detector in detectors) == (
        "FallEventLatch",
        "BedExitMonitor",
    )
    bed_exit = detectors[1]
    assert bed_exit._night_window is not None
    assert bed_exit._night_window.start == "21:00"
    assert bed_exit._night_window.end == "05:00"
    assert bed_exit._night_window.tz == "Asia/Seoul"


def test_domain_detectors_leave_fall_without_time_gate(tmp_path: Path) -> None:
    artifact_dir = _write_lstm_artifact(tmp_path / "models" / "fall" / "lstm")
    config_path = _write_config(tmp_path / "ml-worker.yaml", artifact_dir=artifact_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["domains"] = {"fall": {"enabled": True}}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_edge_worker_config(config_path)

    detectors = edge_worker._domain_detectors(config)

    assert tuple(detector.__class__.__name__ for detector in detectors) == ("FallEventLatch",)
    assert not hasattr(detectors[0], "_night_window")


def test_video_file_source_is_still_available_for_rtsp_harness_input() -> None:
    assert VideoFileSource is not None


def _create_runner(task: str, **kwargs: object) -> object:
    if task == "pose":
        return _PoseRunner()
    if task == "bed":
        return _BedRunner()
    return edge_worker.DEFAULT_REGISTRY.get_factory(task)(**kwargs)


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
                "domains": {"enabled": ["fall"]},
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


def _write_lstm_artifact(path: Path) -> Path:
    from edge.runners.torch_lstm_fall import build_lstm_module

    path.mkdir(parents=True)
    arch = {"hidden": 4, "layers": 1, "dropout": 0.0}
    module = build_lstm_module(hidden=4, layers=1, dropout=0.0)
    with torch.no_grad():
        module.fc.bias[0] = -5.0
        module.fc.bias[1] = 5.0
    torch.save(module.state_dict(), path / "model.pt")
    (path / "arch.json").write_text(json.dumps(arch), encoding="utf-8")
    (path / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "type": "lstm",
                "framework": "pytorch",
                "mode": "sequence",
                "artifact_dir": str(path),
                "weights": "model.pt",
                "architecture": "arch.json",
                "metadata": "metadata.yaml",
                "window": 3,
                "stride": 1,
                "input_shape": [3, 51],
                "operating_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    return path
