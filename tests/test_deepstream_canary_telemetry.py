from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import worker.runtime.deepstream.canary_telemetry as telemetry_module
import worker.tools.deepstream_canary.runner as runner_module
from worker.runtime.deepstream.canary_telemetry import NativeCanaryTelemetry
from worker.tools.deepstream_canary.collector import CollectionRequest, collect_recorded_telemetry
from worker.tools.deepstream_canary.execution_artifacts import (
    ExecutionArtifactSources,
    copy_windows,
)
from worker.tools.deepstream_canary.models import CanaryMode
from worker.tools.deepstream_canary.safety import (
    CanarySafetyError,
    LiveSnapshot,
    SafetyLimits,
)
from worker.tools.deepstream_canary.telemetry import (
    CopyWindowSample,
    NativeWindowSample,
    RuntimeGpuSample,
)


def test_native_telemetry_seals_non_overlapping_window_when_boundary_arrives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: deterministic source PTS/wall mapping and one skipped native publication.
    monotonic_values = iter((0.0, 5.0, 10.1, 10.1))
    wall_values = iter((1_000_000_000, 1_100_000_000, 11_200_000_000))
    monkeypatch.setattr(telemetry_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(telemetry_module.time, "time_ns", lambda: next(wall_values))
    output = tmp_path / "native.jsonl"
    telemetry = NativeCanaryTelemetry("loop-01", output)

    # When: the second decision crosses the exact 10-second window boundary.
    telemetry.record(0, 1_000_000_000, 1)
    telemetry.record(10_000_000_000, 11_000_000_000, 3)

    # Then: one immutable raw window carries measured latency and overwrite counts.
    record = json.loads(output.read_text())
    assert record["schema_version"] == 2
    assert record["decision_count"] == 2
    assert record["latency_samples_ms"] == [100.0, 200.0]
    assert record["metadata_published"] == 3
    assert record["metadata_overwritten"] == 1
    assert record["timestamp_discontinuities"] == 0


def test_collector_flushes_partial_camera_registry_on_abort(tmp_path: Path) -> None:
    # Given: source telemetry is absent but the expected camera registry is known.
    (tmp_path / "raw").mkdir()
    gpu = RuntimeGpuSample(
        child_pid=1,
        child_memory_mib=1.0,
        global_used_mib=1.0,
        total_mib=10.0,
        utilization=1.0,
    )

    # When: the first-fault path seals the rung before teardown.
    receipt = collect_recorded_telemetry(
        CollectionRequest(
            evidence_dir=tmp_path,
            rung="loopback",
            mode=CanaryMode.SHARED_HOST_SMOKE,
            clean_steady_seconds=900,
            camera_ids=("loop-01",),
            gpu_samples=(gpu,),
            native_windows=(),
            copy_windows=(),
            allow_partial=True,
            fault_windows=("native_source_ready_timeout",),
        )
    )

    # Then: declared identity and fault survive even though camera samples are empty.
    recorded = json.loads(receipt.read_text())
    assert recorded["camera_count"] == 1
    assert recorded["cameras"] == []
    assert recorded["fault_windows"] == ["native_source_ready_timeout"]


def test_collector_rejects_missing_expected_camera_identity(tmp_path: Path) -> None:
    # Given: a workload declares loop-01 but native telemetry belongs to no such camera.
    window = NativeWindowSample(
        schema_version=2,
        camera_id="wrong-camera",
        window_started_ns=1,
        window_ended_ns=10_000_000_001,
        decision_count=150,
        latency_samples_ms=(1.0,),
        metadata_published=150,
        metadata_overwritten=0,
        timestamp_discontinuities=0,
    )
    gpu = RuntimeGpuSample(
        child_pid=1,
        child_memory_mib=1.0,
        global_used_mib=1.0,
        total_mib=10.0,
        utilization=1.0,
    )

    # When / Then: receipt construction fails instead of emitting cameras=[].
    with pytest.raises(ValueError, match="parent/copy camera identities mismatch"):
        collect_recorded_telemetry(
            CollectionRequest(
                evidence_dir=tmp_path,
                rung="loopback",
                mode=CanaryMode.SHARED_HOST_SMOKE,
                clean_steady_seconds=900,
                camera_ids=("loop-01",),
                gpu_samples=(gpu,),
                native_windows=(window,),
                copy_windows=(),
            )
        )


def test_copy_windows_never_reads_child_records_from_parent_file(tmp_path: Path) -> None:
    parent_path = tmp_path / "native-telemetry.jsonl"
    parent_path.write_text(json.dumps(_copy_record()) + "\n", encoding="utf-8")

    assert copy_windows(parent_path) == ()


def _native_window(
    *,
    camera_id: str = "loop-01",
    started_ns: int = 1,
    ended_ns: int = 10_000_000_001,
    decisions: int = 150,
) -> NativeWindowSample:
    return NativeWindowSample(
        schema_version=2,
        camera_id=camera_id,
        window_started_ns=started_ns,
        window_ended_ns=ended_ns,
        decision_count=decisions,
        latency_samples_ms=(1.0,),
        metadata_published=decisions,
        metadata_overwritten=0,
        timestamp_discontinuities=0,
    )


def _copy_window(
    *,
    camera_id: str = "loop-01",
    started_ns: int = 1,
    ended_ns: int = 10_000_000_001,
    frames: int = 30,
    h2d_bytes_max: int = 0,
    d2h_bytes_max: int = 1,
    box_source: str = "pose",
    pool_wait_us_p95: float = 1.0,
    gpu_us_p95: float = 1.0,
    surface_drops: int = 0,
) -> CopyWindowSample:
    return CopyWindowSample(
        schema_version=1,
        camera_id=camera_id,
        window_started_ns=started_ns,
        window_ended_ns=ended_ns,
        frames=frames,
        h2d_bytes_max=h2d_bytes_max,
        d2h_bytes_max=d2h_bytes_max,
        box_source=box_source,  # type: ignore[arg-type]
        pool_wait_us_p95=pool_wait_us_p95,
        gpu_us_p95=gpu_us_p95,
        surface_drops=surface_drops,
    )


def test_collector_correlates_overlapping_parent_and_copy_windows_conservatively(
    tmp_path: Path,
) -> None:
    (tmp_path / "raw").mkdir()
    gpu = RuntimeGpuSample(
        child_pid=1,
        child_memory_mib=1.0,
        global_used_mib=1.0,
        total_mib=10.0,
        utilization=1.0,
    )
    receipt = collect_recorded_telemetry(
        CollectionRequest(
            evidence_dir=tmp_path,
            rung="loopback",
            mode=CanaryMode.SHARED_HOST_SMOKE,
            clean_steady_seconds=900,
            camera_ids=("loop-01",),
            gpu_samples=(gpu,),
            native_windows=(
                _native_window(),
                _native_window(started_ns=10_000_000_001, ended_ns=20_000_000_001),
            ),
            copy_windows=(
                _copy_window(
                    started_ns=5_000_000_001,
                    ended_ns=15_000_000_001,
                    frames=30,
                    d2h_bytes_max=100,
                    pool_wait_us_p95=2.0,
                    gpu_us_p95=3.0,
                ),
                _copy_window(
                    started_ns=15_000_000_001,
                    ended_ns=25_000_000_001,
                    frames=30,
                    d2h_bytes_max=200424,
                    pool_wait_us_p95=4.0,
                    gpu_us_p95=5.0,
                    surface_drops=1,
                ),
            ),
            allow_partial=True,
        )
    )

    camera = json.loads(receipt.read_text())["cameras"][0]
    assert camera["copy_window_frames"] == 60
    assert camera["frame_window_spans_seconds"] == [10.0, 10.0]
    assert camera["telemetry_coverage_seconds"] == 20.0
    assert camera["h2d_bytes_max"] == 0
    assert camera["d2h_bytes_max"] == 200424
    assert camera["pool_wait_us_p95"] == 4.0
    assert camera["gpu_us_p95"] == 5.0
    assert camera["surface_drops"] == 1


def _gpu_sample() -> RuntimeGpuSample:
    return RuntimeGpuSample(
        child_pid=1,
        child_memory_mib=1.0,
        global_used_mib=1.0,
        total_mib=10.0,
        utilization=1.0,
    )


def _ten_second_windows(
    count: int,
    *,
    final_duration_seconds: int = 10,
) -> tuple[tuple[NativeWindowSample, ...], tuple[CopyWindowSample, ...]]:
    parents: list[NativeWindowSample] = []
    copies: list[CopyWindowSample] = []
    started_ns = 1
    for index in range(count):
        duration_seconds = final_duration_seconds if index == count - 1 else 10
        ended_ns = started_ns + duration_seconds * 1_000_000_000
        parents.append(
            _native_window(
                started_ns=started_ns,
                ended_ns=ended_ns,
                decisions=150,
            )
        )
        copies.append(
            _copy_window(
                started_ns=started_ns,
                ended_ns=ended_ns,
                frames=30,
            )
        )
        started_ns = ended_ns
    return tuple(parents), tuple(copies)


def test_collector_derives_frame_span_from_child_frames_not_parent_decisions(
    tmp_path: Path,
) -> None:
    (tmp_path / "raw").mkdir()

    receipt = collect_recorded_telemetry(
        CollectionRequest(
            evidence_dir=tmp_path,
            rung="loopback",
            mode=CanaryMode.SHARED_HOST_SMOKE,
            clean_steady_seconds=900,
            camera_ids=("loop-01",),
            gpu_samples=(_gpu_sample(),),
            native_windows=(_native_window(decisions=150),),
            copy_windows=(_copy_window(frames=1),),
            allow_partial=True,
        )
    )

    camera = json.loads(receipt.read_text())["cameras"][0]
    assert camera["frame_window_spans_seconds"] == [300.0]
    assert camera["telemetry_coverage_seconds"] == 10.0


def test_collector_rejects_short_strict_coverage(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()

    with pytest.raises(ValueError, match="telemetry coverage is insufficient"):
        collect_recorded_telemetry(
            CollectionRequest(
                evidence_dir=tmp_path,
                rung="loopback",
                mode=CanaryMode.SHARED_HOST_SMOKE,
                clean_steady_seconds=900,
                camera_ids=("loop-01",),
                gpu_samples=(_gpu_sample(),),
                native_windows=(_native_window(),),
                copy_windows=(_copy_window(),),
            )
        )


def test_collector_rejects_strict_internal_window_gap(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    first_parent = _native_window()
    first_copy = _copy_window()
    second_started_ns = 11_000_000_002

    with pytest.raises(ValueError, match="not temporally contiguous"):
        collect_recorded_telemetry(
            CollectionRequest(
                evidence_dir=tmp_path,
                rung="loopback",
                mode=CanaryMode.SHARED_HOST_SMOKE,
                clean_steady_seconds=900,
                camera_ids=("loop-01",),
                gpu_samples=(_gpu_sample(),),
                native_windows=(
                    first_parent,
                    _native_window(
                        started_ns=second_started_ns,
                        ended_ns=second_started_ns + 10_000_000_000,
                    ),
                ),
                copy_windows=(
                    first_copy,
                    _copy_window(
                        started_ns=second_started_ns,
                        ended_ns=second_started_ns + 10_000_000_000,
                    ),
                ),
            )
        )


def test_collector_accepts_complete_strict_telemetry(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    parents, copies = _ten_second_windows(90)

    receipt = collect_recorded_telemetry(
        CollectionRequest(
            evidence_dir=tmp_path,
            rung="loopback",
            mode=CanaryMode.SHARED_HOST_SMOKE,
            clean_steady_seconds=900,
            camera_ids=("loop-01",),
            gpu_samples=(_gpu_sample(),),
            native_windows=parents,
            copy_windows=copies,
        )
    )

    camera = json.loads(receipt.read_text())["cameras"][0]
    assert camera["telemetry_coverage_seconds"] == 900.0


def test_collector_omits_zero_frame_child_from_partial_telemetry(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    request = CollectionRequest(
        evidence_dir=tmp_path,
        rung="loopback",
        mode=CanaryMode.SHARED_HOST_SMOKE,
        clean_steady_seconds=900,
        camera_ids=("loop-01",),
        gpu_samples=(_gpu_sample(),),
        native_windows=(_native_window(),),
        copy_windows=(_copy_window(frames=0),),
    )

    with pytest.raises(ValueError, match="positive child frames"):
        collect_recorded_telemetry(request)

    partial = collect_recorded_telemetry(
        request.model_copy(update={"allow_partial": True})
    )
    assert json.loads(partial.read_text())["cameras"] == []


def test_collector_accepts_exact_coverage_boundary_tolerance(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    parents, copies = _ten_second_windows(88, final_duration_seconds=8)

    receipt = collect_recorded_telemetry(
        CollectionRequest(
            evidence_dir=tmp_path,
            rung="loopback",
            mode=CanaryMode.SHARED_HOST_SMOKE,
            clean_steady_seconds=900,
            camera_ids=("loop-01",),
            gpu_samples=(_gpu_sample(),),
            native_windows=parents,
            copy_windows=copies,
        )
    )

    camera = json.loads(receipt.read_text())["cameras"][0]
    assert camera["telemetry_coverage_seconds"] == 878.0


@pytest.mark.parametrize(
    ("copies", "message"),
    (
        ((), "parent/copy camera identities mismatch"),
        (
            (_copy_window(), _copy_window(box_source="person")),
            "mixed copy box_source",
        ),
    ),
)
def test_collector_rejects_missing_child_identity_and_mixed_copy_source(
    tmp_path: Path, copies: tuple[CopyWindowSample, ...], message: str
) -> None:
    gpu = RuntimeGpuSample(
        child_pid=1,
        child_memory_mib=1.0,
        global_used_mib=1.0,
        total_mib=10.0,
        utilization=1.0,
    )

    with pytest.raises(ValueError, match=message):
        collect_recorded_telemetry(
            CollectionRequest(
                evidence_dir=tmp_path,
                rung="loopback",
                mode=CanaryMode.SHARED_HOST_SMOKE,
                clean_steady_seconds=900,
                camera_ids=("loop-01",),
                gpu_samples=(gpu,),
                native_windows=(_native_window(),),
                copy_windows=copies,
            )
        )


def _runner_request(tmp_path: Path) -> runner_module.ExecutionRequest:
    return runner_module.ExecutionRequest(
        compose_path=tmp_path / "compose.yaml",
        evidence_dir=tmp_path,
        baseline=LiveSnapshot(containers=(), xid_count=0),
        rung_durations=(("loopback", 0),),
        publisher_count=1,
        relay_token="test-token",
        safety_limits=SafetyLimits(
            minimum_gpu_slack_mib=1.0,
            maximum_gpu_utilization=100.0,
            require_live_status=False,
        ),
        mode=CanaryMode.SHARED_HOST_SMOKE,
        rungs=("loopback",),
        artifacts=ExecutionArtifactSources(
            worker_image="example@sha256:worker",
            support_images=(),
            gate_policy="policy",
            compose="compose",
        ),
    )


def _native_record() -> dict[str, object]:
    return {
        "schema_version": 2,
        "camera_id": "loop-01",
        "window_started_ns": 1,
        "window_ended_ns": 10_000_000_001,
        "decision_count": 1,
        "latency_samples_ms": [1.0],
        "metadata_published": 1,
        "metadata_overwritten": 0,
        "timestamp_discontinuities": 0,
    }


def _copy_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "camera_id": "loop-01",
        "window_started_ns": 1,
        "window_ended_ns": 10_000_000_001,
        "frames": 1,
        "h2d_bytes_max": 1,
        "d2h_bytes_max": 1,
        "box_source": "pose",
        "pool_wait_us_p95": 1.0,
        "gpu_us_p95": 1.0,
        "surface_drops": 0,
    }


def test_runner_precreates_distinct_child_copy_sidecar_and_collects_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "raw").mkdir()
    request = _runner_request(tmp_path)
    parent_path = tmp_path / "raw" / "native-telemetry.jsonl"
    child_path = tmp_path / "raw" / "native-telemetry.child-copy.jsonl"
    collected: list[CollectionRequest] = []
    sample = RuntimeGpuSample(
        child_pid=1,
        child_memory_mib=1.0,
        global_used_mib=1.0,
        total_mib=10.0,
        utilization=1.0,
    )

    def compose(_request: runner_module.ExecutionRequest, *arguments: str) -> object:
        assert parent_path.is_file()
        assert child_path.is_file()
        assert parent_path.stat().st_mode & 0o777 == 0o666
        assert child_path.stat().st_mode & 0o777 == 0o666
        if arguments[:2] == ("up", "-d") and "ml-worker" in arguments:
            parent_path.write_text(json.dumps(_native_record()) + "\n", encoding="utf-8")
            child_path.write_text(json.dumps(_copy_record()) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(runner_module, "_compose", compose)
    monkeypatch.setattr(runner_module, "generate_corpus", lambda _root: tmp_path / "corpus")
    monkeypatch.setattr(runner_module, "compare_live_snapshot", lambda *args: request.baseline)
    monkeypatch.setattr(runner_module, "gpu_sample", lambda _snapshot: sample)
    monkeypatch.setattr(runner_module, "_probe_preview_paths", lambda *_args: "http://preview")
    monkeypatch.setattr(runner_module, "_capture_preview_evidence", lambda *_args: None)
    monkeypatch.setattr(runner_module, "_wait_for_source_ready", lambda *_args: None)
    monkeypatch.setattr(runner_module, "collect_recorded_telemetry", collected.append)
    monkeypatch.setattr(runner_module, "emit_receipts", lambda *_args: ())

    assert runner_module.execute_canary(request) == 0
    assert len(collected) == 1
    assert collected[0].native_windows[0].schema_version == 2
    assert collected[0].copy_windows[0].schema_version == 1
    assert collected[0].copy_windows == copy_windows(parent_path)


def test_runner_partial_collection_uses_child_copy_sidecar_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "raw").mkdir()
    request = _runner_request(tmp_path)
    parent_path = tmp_path / "raw" / "native-telemetry.jsonl"
    child_path = tmp_path / "raw" / "native-telemetry.child-copy.jsonl"
    collected: list[CollectionRequest] = []
    sample = RuntimeGpuSample(
        child_pid=1,
        child_memory_mib=1.0,
        global_used_mib=1.0,
        total_mib=10.0,
        utilization=1.0,
    )

    def compose(_request: runner_module.ExecutionRequest, *arguments: str) -> object:
        if arguments[:2] == ("up", "-d") and "ml-worker" in arguments:
            parent_path.write_text(json.dumps(_native_record()) + "\n", encoding="utf-8")
            child_path.write_text(json.dumps(_copy_record()) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def collect(collection: CollectionRequest) -> Path:
        collected.append(collection)
        if not collection.allow_partial:
            raise CanarySafetyError("collection_failed", "test")
        return tmp_path / "raw" / "telemetry-loopback.json"

    monkeypatch.setattr(runner_module, "_compose", compose)
    monkeypatch.setattr(runner_module, "generate_corpus", lambda _root: tmp_path / "corpus")
    monkeypatch.setattr(runner_module, "compare_live_snapshot", lambda *args: request.baseline)
    monkeypatch.setattr(runner_module, "gpu_sample", lambda _snapshot: sample)
    monkeypatch.setattr(runner_module, "_probe_preview_paths", lambda *_args: "http://preview")
    monkeypatch.setattr(runner_module, "_capture_preview_evidence", lambda *_args: None)
    monkeypatch.setattr(runner_module, "_wait_for_source_ready", lambda *_args: None)
    monkeypatch.setattr(runner_module, "collect_recorded_telemetry", collect)
    monkeypatch.setattr(runner_module, "emit_receipts", lambda *_args: ())

    assert runner_module.execute_canary(request) == 1
    assert len(collected) == 2
    partial = collected[1]
    assert partial.allow_partial
    assert partial.native_windows[0].schema_version == 2
    assert partial.copy_windows[0].schema_version == 1
    assert partial.copy_windows == copy_windows(parent_path)
