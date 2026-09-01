from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "worker.tools.deepstream_canary", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _policy(path: Path) -> Path:
    source = Path("scripts/qa/deepstream-canary/gate-policy.v1.json")
    path.write_bytes(source.read_bytes())
    return path


def _passing_receipt(camera_count: int = 1) -> dict[str, object]:
    cameras = [
        {
            "camera_id": f"loop-{index:02d}",
            "fps_windows": [14.8, 15.0, 15.2],
            "copy_window_frames": 30,
            "frame_window_spans_seconds": [2.0],
            "telemetry_coverage_seconds": 1000.0,
            "h2d_bytes_max": 0,
            "d2h_bytes_max": 200424,
            "box_source": "pose",
            "pool_wait_us_p95": 1.0,
            "gpu_us_p95": 1.0,
            "surface_drops": 0,
            "latency_ms": {"p50": 90.0, "p95": 150.0, "p99": 190.0, "max": 220.0},
            "au_gaps": 0,
            "config_discontinuities": 0,
            "timestamp_discontinuities": 0,
            "metadata_published": 1000,
            "metadata_overwritten": 10,
            "event_evidence_parity": True,
            "preview_ok": True,
            "derivative_ok": True,
        }
        for index in range(camera_count)
    ]
    return {
        "schema_version": 2,
        "rung": "loopback" if camera_count == 1 else str(camera_count),
        "mode": "commissioning",
        "camera_count": camera_count,
        "clean_steady_seconds": 900 if camera_count == 1 else 7200,
        "cameras": cameras,
        "gpu": {
            "child_pid": 222,
            "warmup_peak_mib": 5000.0,
            "steady_p50_mib": 4500.0,
            "steady_p95_mib": 4700.0,
            "recovery_mib": 4300.0,
            "global_used_mib": 5000.0,
            "total_mib": 16384.0,
            "minimum_slack_mib": 11000.0,
            "utilization_p50": 35.0,
            "utilization_p95": 55.0,
            "utilization_max": 70.0,
            "new_xids": [],
        },
        "nvdec": {"hardware_branches": camera_count, "software_fallbacks": 0},
        "timeline": [
            {"kind": kind, "sha256": character * 64, "playable": True}
            for kind, character in (
                ("event", "a"),
                ("evidence", "b"),
                ("preview", "c"),
                ("derivative", "d"),
            )
        ],
        "live_protection": {
            "container_restarts": 0,
            "camera_stale_transitions": 0,
            "evidence_drop_increase": 0,
            "relay_sentinel_leaks": 0,
            "mount_intersections": 0,
            "kernel_faults": 0,
        },
        "fault_windows": [],
        "workload": {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "fps": 15.0,
            "gop": 30,
            "camera_phase_offsets_ms": [index * 67 for index in range(camera_count)],
        },
        "artifacts": {
            "worker_image": "sha256:" + "b" * 64,
            "support_images": ("sha256:" + "e" * 64,),
            "models_manifest": "f" * 64,
            "engine_manifest": "1" * 64,
            "corpus": "2" * 64,
            "gate_policy": "3" * 64,
            "compose": "4" * 64,
        },
    }


def test_cli_renders_isolated_zero_and_loopback_compose(tmp_path: Path) -> None:
    # Given: no facility authorization and a fresh run directory.
    evidence = tmp_path / "run"

    # When: the default non-facility contract is rendered without starting Docker.
    result = _run_cli(
        "run",
        "--rungs",
        "zero,loopback",
        "--evidence-dir",
        str(evidence),
        "--render-only",
        "--worker-image",
        "seeon-edge@sha256:" + "b" * 64,
    )

    # Then: compose is isolated, publisher-local, loopback-bound, and immutable.
    assert result.returncode == 0, result.stderr
    rendered = (evidence / "compose.rendered.yaml").read_text()
    assert "name: seeon-ds-canary" in rendered
    assert "internal: true" in rendered
    assert '127.0.0.1:18090:8090' in rendered
    assert "publisher-01:" in rendered
    assert "ml-api" in rendered
    assert (evidence / "receipt-manifest.json").is_file()
    compose_check = subprocess.run(
        ["docker", "compose", "-f", str(evidence / "compose.rendered.yaml"), "config", "-q"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CANARY_RELAY_TOKEN": "test-only-render-token"},
        timeout=30,
    )
    assert compose_check.returncode == 0, compose_check.stderr


def test_relay_stub_serves_current_release_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from shared.release_identity import (
        EDGE_DATABASE_FORMAT_IDENTITY,
        EDGE_DATABASE_SCHEMA_VERSION,
    )

    monkeypatch.setenv("CANARY_RELAY_TOKEN", "test-token")
    monkeypatch.setenv("CANARY_RECEIPT_DIR", str(tmp_path / "receipts"))
    monkeypatch.setenv("CANARY_WORKER_CONFIG_DIR", str(tmp_path / "configs"))
    monkeypatch.setenv("CANARY_ACTIVE_CONFIG", str(tmp_path / "active-config"))
    monkeypatch.setenv(
        "CANARY_EDGE_DATABASE_SCHEMA_VERSION",
        str(EDGE_DATABASE_SCHEMA_VERSION),
    )
    monkeypatch.setenv(
        "CANARY_EDGE_DATABASE_FORMAT_IDENTITY",
        EDGE_DATABASE_FORMAT_IDENTITY,
    )
    relay = runpy.run_path("scripts/qa/deepstream-canary/relay_stub.py")
    response: dict[str, int] = {}
    body = io.BytesIO()
    handler = SimpleNamespace(
        path="/health/release-identity",
        headers={},
        send_response=lambda status: response.update(status=status),
        send_error=lambda status: response.update(status=status),
        send_header=lambda _name, _value: None,
        end_headers=lambda: None,
        wfile=body,
    )

    relay["RelayHandler"].do_GET(handler)

    assert response == {"status": 200}
    assert json.loads(body.getvalue()) == {
        "edge_database_format_identity": EDGE_DATABASE_FORMAT_IDENTITY,
        "edge_database_schema_version": EDGE_DATABASE_SCHEMA_VERSION,
    }


def test_browser_evidence_authenticates_upstream_before_accepting_screenshot() -> None:
    script = Path("scripts/qa/deepstream_canary_browser.mjs").read_text(
        encoding="utf-8"
    )

    assert "--extra-headers=" not in script
    assert '"X-Edge-Relay-Token": token' in script
    assert "upstreamStatus === 200" in script
    assert "frameDelivered" in script
    assert '"content-type": "image/jpeg"' in script
    assert "upstream.destroy()" in script
    assert 'stdio: "ignore"' in script
    assert "setTimeout(resolveHold" not in script


def test_live_rung_refuses_without_authorization_before_docker(tmp_path: Path) -> None:
    # Given: a requested facility rung without an owner artifact.
    evidence = tmp_path / "unauthorized"

    # When: the operator requests rung 1.
    result = _run_cli(
        "run",
        "--rungs",
        "1",
        "--evidence-dir",
        str(evidence),
        "--render-only",
        "--worker-image",
        "seeon-edge@sha256:" + "b" * 64,
    )

    # Then: the request fails closed and records the refusal.
    assert result.returncode == 2
    refusal = json.loads((evidence / "authorization-refusal.json").read_text())
    assert refusal["code"] == "authorization_required"


def test_rung_13_requires_current_authorization_eight_pass_and_slack(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.authorization import AuthorizationRequest, authorize_rungs
    from worker.tools.deepstream_canary.models import AuthorizationArtifact, CanaryMode

    # Given: a current artifact lacking an 8-pass binding and projected slack.
    artifact = AuthorizationArtifact(
        schema_version=1,
        appliance_id="edge-lab-01",
        worker_image="sha256:" + "b" * 64,
        camera_ids=tuple(f"camera-{index:02d}" for index in range(13)),
        owner="owner@example.invalid",
        issue="https://github.com/SeniorAILab/SeeON-edge/issues/381",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        authorized_rungs=(1, 4, 8, 13),
        eight_pass_report_sha256=None,
        projected_slack_mib=3071.0,
    )

    request = AuthorizationRequest(
        rungs=(13,),
        mode=CanaryMode.COMMISSIONING,
        appliance_id="edge-lab-01",
        worker_image_digest="sha256:" + "b" * 64,
        camera_ids=tuple(f"camera-{index:02d}" for index in range(13)),
        now=datetime.now(UTC),
        minimum_projected_slack_mib=3072.0,
    )

    # When / Then: rung 13 is refused independently of Compose execution.
    with pytest.raises(ValueError, match="eight_pass_report_required"):
        authorize_rungs(request, artifact)


def test_mount_guard_rejects_live_volume_ancestor(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.safety import (
        LiveContainer,
        LiveSnapshot,
        refuse_mount_overlap,
    )

    # Given: a canary state directory nested under a live bind mount.
    live = tmp_path / "live"
    snapshot = LiveSnapshot((LiveContainer("live-1", 0, (live,)),), 0)

    # When / Then: mount admission fails before Compose starts.
    with pytest.raises(RuntimeError, match="live_mount_intersection"):
        refuse_mount_overlap((live / "canary",), snapshot)


def test_live_gpu_protection_attributes_container_and_allows_pid_churn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worker.tools.deepstream_canary.safety as safety_module
    from worker.tools.deepstream_canary.safety import (
        LiveContainer,
        LiveSnapshot,
        SafetyLimits,
        compare_live_snapshot,
    )

    # Given: one protected SeeON GPU container plus an unrelated host process.
    before = LiveSnapshot(
        containers=(LiveContainer("live-1", 0, (), True, ("123",)),),
        xid_count=0,
        gpu_processes=("123, python, 100", "999, desktop, 10"),
    )
    after = LiveSnapshot(
        containers=(LiveContainer("live-1", 0, (), True, ("456",)),),
        xid_count=0,
        gpu_processes=("456, ffmpeg, 100",),
    )
    monkeypatch.setattr(safety_module, "capture_live_snapshot", lambda: after)

    # When / Then: unrelated exit and attributed in-container PID churn are allowed.
    observed: dict[str, float] = {}
    compare_live_snapshot(before, SafetyLimits(0, 100, False), observed)
    assert observed == {}


def test_live_gpu_protection_applies_bounded_container_loss_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worker.tools.deepstream_canary.safety as safety_module
    from worker.tools.deepstream_canary.safety import (
        LiveContainer,
        LiveSnapshot,
        SafetyLimits,
        compare_live_snapshot,
    )

    # Given: a protected SeeON container temporarily has no attributed GPU process.
    before = LiveSnapshot((LiveContainer("live-1", 0, (), True, ("123",)),), 0)
    after = LiveSnapshot((LiveContainer("live-1", 0, (), True, ()),), 0)
    monkeypatch.setattr(safety_module, "capture_live_snapshot", lambda: after)
    clock = iter((0.0, 31.0))
    monkeypatch.setattr(safety_module.time, "monotonic", lambda: next(clock))
    observed: dict[str, float] = {}
    limits = SafetyLimits(0, 100, False, gpu_process_loss_grace_seconds=30)

    # When / Then: the first miss is tolerated, but sustained loss fails closed.
    compare_live_snapshot(before, limits, observed)
    with pytest.raises(RuntimeError, match="live_container_gpu_process_missing"):
        compare_live_snapshot(before, limits, observed)


def test_verifier_recomputes_pass_and_rejects_intentional_red(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.gates import evaluate_receipt
    from worker.tools.deepstream_canary.models import GatePolicy, RungReceipt

    # Given: raw per-camera signals and the versioned policy.
    policy = GatePolicy.model_validate_json(_policy(tmp_path / "policy.json").read_bytes())
    passing = RungReceipt.model_validate(_passing_receipt())
    baseline = RungReceipt.model_validate(_baseline_for(_passing_receipt()))

    # When: gate math is recomputed.
    report = evaluate_receipt(passing, policy, baseline)

    # Then: clean signals pass, while a missing PTS mapping is an intentional RED.
    assert report.verdict == "PASS"
    malformed = passing.model_dump_json().replace(
        '\"latency_ms\":{\"p50\":90.0,\"p95\":150.0,\"p99\":190.0,\"max\":220.0},',
        "",
    )
    with pytest.raises(ValueError):
        RungReceipt.model_validate_json(malformed)


def _baseline_for(receipt: dict[str, object]) -> dict[str, object]:
    baseline = copy.deepcopy(receipt)
    baseline["cameras"][0]["latency_ms"]["p95"] = 151.0  # type: ignore[index]
    return baseline


def _write_verified_rung(
    root: Path, camera_count: int = 1, latency_ms: float = 150.0
) -> dict[str, object]:
    from worker.tools.deepstream_canary.models import ArtifactBindings
    from worker.tools.deepstream_canary.telemetry import RecordedRungTelemetry, build_rung_receipt

    receipt = _passing_receipt(camera_count)
    receipt["cameras"] = [
        {
            **camera,
            "latency_ms": {
                "p50": latency_ms,
                "p95": latency_ms,
                "p99": latency_ms,
                "max": latency_ms,
            },
        }
        for camera in receipt["cameras"]  # type: ignore[index]
    ]
    raw = copy.deepcopy(receipt)
    artifacts = raw.pop("artifacts")
    raw["cameras"] = [
        {
            "camera_id": camera["camera_id"],
            "decision_window_counts": [148, 150, 152],
            "decision_window_seconds": [10.0, 10.0, 10.0],
            "telemetry_coverage_seconds": camera["telemetry_coverage_seconds"],
            "copy_window_frames": camera["copy_window_frames"],
            "frame_window_spans_seconds": camera["frame_window_spans_seconds"],
            "h2d_bytes_max": camera["h2d_bytes_max"],
            "d2h_bytes_max": camera["d2h_bytes_max"],
            "box_source": camera["box_source"],
            "pool_wait_us_p95": camera["pool_wait_us_p95"],
            "gpu_us_p95": camera["gpu_us_p95"],
            "surface_drops": camera["surface_drops"],
            "latency_samples_ms": [latency_ms],
            "au_gaps": camera["au_gaps"],
            "config_discontinuities": camera["config_discontinuities"],
            "timestamp_discontinuities": camera["timestamp_discontinuities"],
            "metadata_published": camera["metadata_published"],
            "metadata_overwritten": camera["metadata_overwritten"],
            "event_evidence_parity": camera["event_evidence_parity"],
            "preview_ok": camera["preview_ok"],
            "derivative_ok": camera["derivative_ok"],
        }
        for camera in receipt["cameras"]  # type: ignore[index]
    ]
    gpu = raw["gpu"]  # type: ignore[index]
    raw["gpu"] = {
        "child_pid": gpu["child_pid"],
        "warmup_memory_mib": [gpu["warmup_peak_mib"]],
        "steady_memory_mib": [gpu["steady_p95_mib"]],
        "recovery_memory_mib": [gpu["recovery_mib"]],
        "global_used_mib": gpu["global_used_mib"],
        "total_mib": gpu["total_mib"],
        "slack_samples_mib": [gpu["minimum_slack_mib"]],
        "utilization_samples": [gpu["utilization_p95"]],
        "new_xids": gpu["new_xids"],
    }
    recorded = RecordedRungTelemetry.model_validate(raw)
    rebuilt = build_rung_receipt(recorded, ArtifactBindings.model_validate(artifacts))
    (root / "raw" / f"telemetry-{rebuilt.rung}.json").write_text(recorded.model_dump_json())
    (root / "raw" / f"rung-{rebuilt.rung}.json").write_text(rebuilt.model_dump_json())
    return rebuilt.model_dump(mode="json")


def _failed_check_names(report: object) -> set[str]:
    return {check.name for check in report.checks if not check.passed}  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("field", "value", "check"),
    (
        ("h2d_bytes_max", 1, "camera.loop-00.h2d_bytes_max"),
        ("d2h_bytes_max", 200425, "camera.loop-00.d2h_bytes_max"),
        (
            "frame_window_spans_seconds",
            [2.16],
            "camera.loop-00.frame_window_span_max_seconds",
        ),
        ("surface_drops", 1, "camera.loop-00.surface_drops"),
    ),
)
def test_v4_absolute_copy_gates_reject_values_beyond_exact_boundaries(
    tmp_path: Path, field: str, value: int | list[float], check: str
) -> None:
    from worker.tools.deepstream_canary.gates import evaluate_receipt
    from worker.tools.deepstream_canary.models import GatePolicy, RungReceipt

    policy = GatePolicy.model_validate_json(_policy(tmp_path / "policy.json").read_bytes())
    candidate = _passing_receipt()
    candidate["cameras"][0][field] = value  # type: ignore[index]
    report = evaluate_receipt(
        RungReceipt.model_validate(candidate),
        policy,
        RungReceipt.model_validate(_baseline_for(_passing_receipt())),
    )

    assert check in _failed_check_names(report)


def test_v4_exact_copy_gate_boundaries_and_matched_baseline_pass(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.gates import evaluate_receipt
    from worker.tools.deepstream_canary.models import GatePolicy, RungReceipt

    policy = GatePolicy.model_validate_json(_policy(tmp_path / "policy.json").read_bytes())
    candidate_data = _passing_receipt()
    candidate_data["cameras"][0]["frame_window_spans_seconds"] = [2.15]  # type: ignore[index]
    candidate = RungReceipt.model_validate(candidate_data)
    baseline_data = _baseline_for(_passing_receipt())
    baseline_data["cameras"][0]["frame_window_spans_seconds"] = [2.15]  # type: ignore[index]
    baseline = RungReceipt.model_validate(baseline_data)

    assert policy.policy_id.endswith("-v4")
    assert (policy.h2d_bytes_max, policy.d2h_bytes_max, policy.frame_window_span_max_seconds) == (
        0,
        200424,
        2.15,
    )
    assert evaluate_receipt(candidate, policy, baseline).verdict == "PASS"
    invalid = _passing_receipt()
    invalid["unrecognized"] = True
    with pytest.raises(ValueError, match="unrecognized"):
        RungReceipt.model_validate(invalid)


def test_absolute_gate_rejects_fault_window_and_sparse_telemetry(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.gates import evaluate_absolute_receipt
    from worker.tools.deepstream_canary.models import GatePolicy, RungReceipt

    policy = GatePolicy.model_validate_json(_policy(tmp_path / "policy.json").read_bytes())
    receipt = _passing_receipt()
    receipt["fault_windows"] = ["native_child_exit"]
    receipt["cameras"][0]["telemetry_coverage_seconds"] = 877.0  # type: ignore[index]

    report = evaluate_absolute_receipt(RungReceipt.model_validate(receipt), policy)

    assert {"fault_windows", "camera.loop-00.telemetry_coverage_seconds"} <= _failed_check_names(
        report
    )


@pytest.mark.parametrize(
    ("name", "mutate"),
    (
        ("relative.baseline.rung", lambda baseline: baseline.update(rung="1")),
        (
            "relative.baseline.mode",
            lambda baseline: baseline.update(mode="shared-host-smoke"),
        ),
        (
            "relative.baseline.camera_count",
            lambda baseline: baseline.update(camera_count=2),
        ),
        (
            "relative.baseline.camera_ids",
            lambda baseline: baseline["cameras"][0].update(camera_id="other"),  # type: ignore[index]
        ),
        (
            "relative.baseline.clean_steady_seconds",
            lambda baseline: baseline.update(clean_steady_seconds=901),
        ),
        (
            "relative.baseline.workload",
            lambda baseline: baseline["workload"].update(fps=30.0),  # type: ignore[index]
        ),
        (
            "relative.baseline.corpus",
            lambda baseline: baseline["artifacts"].update(corpus="9" * 64),  # type: ignore[index]
        ),
    ),
)
def test_v4_relative_gate_rejects_mismatched_baseline_facts(
    tmp_path: Path, name: str, mutate: object
) -> None:
    from worker.tools.deepstream_canary.gates import evaluate_receipt
    from worker.tools.deepstream_canary.models import GatePolicy, RungReceipt

    policy = GatePolicy.model_validate_json(_policy(tmp_path / "policy.json").read_bytes())
    baseline = _baseline_for(_passing_receipt())
    mutate(baseline)  # type: ignore[operator]
    report = evaluate_receipt(
        RungReceipt.model_validate(_passing_receipt()),
        policy,
        RungReceipt.model_validate(baseline),
    )

    assert name in _failed_check_names(report)


def test_v4_relative_gate_requires_baseline_and_enforces_per_camera_regression(
    tmp_path: Path,
) -> None:
    from worker.tools.deepstream_canary.gates import evaluate_receipt
    from worker.tools.deepstream_canary.models import GatePolicy, RungReceipt

    policy = GatePolicy.model_validate_json(_policy(tmp_path / "policy.json").read_bytes())
    candidate = _passing_receipt()
    baseline = _baseline_for(_passing_receipt())
    baseline["cameras"][0]["fps_windows"] = [15.0, 15.2, 15.4]  # type: ignore[index]
    baseline["cameras"][0]["latency_ms"]["p95"] = 150.0  # type: ignore[index]

    missing = evaluate_receipt(RungReceipt.model_validate(candidate), policy)
    regression = evaluate_receipt(
        RungReceipt.model_validate(candidate), policy, RungReceipt.model_validate(baseline)
    )
    zero = _passing_receipt(camera_count=0)
    zero.update(
        rung="zero",
        clean_steady_seconds=120,
        nvdec={"hardware_branches": 0, "software_fallbacks": 0},
        timeline=[],
        workload={
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "fps": 15.0,
            "gop": 30,
            "camera_phase_offsets_ms": [],
        },
    )

    assert "relative.baseline" in _failed_check_names(missing)
    assert {
        "camera.loop-00.relative.fps_p05",
        "camera.loop-00.relative.fps_p50",
        "camera.loop-00.relative.latency_p95",
    } <= _failed_check_names(regression)
    assert evaluate_receipt(RungReceipt.model_validate(zero), policy).verdict == "PASS"


def test_product_image_explicitly_excludes_canary_module() -> None:
    # Given / When: Docker build inputs are inspected.
    dockerignore = Path(".dockerignore").read_text()
    dockerfile = Path("Dockerfile.edge").read_text()

    # Then: COPY worker cannot ship the operator-only harness.
    assert "worker/tools/deepstream_canary" in dockerignore
    assert "test ! -e /app/worker/tools/deepstream_canary" in dockerfile


def _verify(
    kind: str, root: Path, baseline_root: Path | None = None
) -> subprocess.CompletedProcess[str]:
    baseline_arguments = (
        ["--baseline-evidence-root", str(baseline_root)]
        if baseline_root is not None
        else []
    )
    return subprocess.run(
        [
            sys.executable,
            "scripts/qa/verify_deepstream_delivery.py",
            kind,
            "--evidence-root",
            str(root),
            *baseline_arguments,
            "--output",
            str(root / f"{kind}.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _render_evidence(root: Path) -> None:
    result = _run_cli(
        "run",
        "--rungs",
        "zero,loopback",
        "--evidence-dir",
        str(root),
        "--render-only",
        "--worker-image",
        "seeon-edge@sha256:" + "b" * 64,
    )
    assert result.returncode == 0


def test_canary_verifier_loads_baseline_evidence_and_keeps_absolute_gates_active(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    baseline_root = tmp_path / "baseline"
    for root in (candidate_root, baseline_root):
        (root / "raw").mkdir(parents=True)
    candidate = _write_verified_rung(candidate_root)
    _ = _write_verified_rung(baseline_root, latency_ms=151.0)
    policy_digest = hashlib.sha256(
        Path("scripts/qa/deepstream-canary/gate-policy.v1.json").read_bytes()
    ).hexdigest()
    for root in (candidate_root, baseline_root):
        (root / "run-request.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "requested_rungs": ["loopback"],
                    "policy_sha256": policy_digest,
                }
            )
            + "\n"
        )

    assert _verify("canary", candidate_root, baseline_root).returncode == 0

    candidate["cameras"][0]["h2d_bytes_max"] = 1  # type: ignore[index]
    (candidate_root / "raw" / "rung-loopback.json").write_text(json.dumps(candidate))
    assert _verify("canary", candidate_root, baseline_root).returncode == 1
    report = json.loads((candidate_root / "canary.json").read_text())
    assert "camera.loop-00.h2d_bytes_max" in {
        check["name"] for check in report["reports"][0]["checks"] if not check["passed"]
    }


def test_canary_verifier_rejects_missing_requested_rung(tmp_path: Path) -> None:
    # Given: immutable zero+loopback intent but only a valid zero receipt.
    root = tmp_path / "missing-loopback"
    (root / "raw").mkdir(parents=True)
    zero = _passing_receipt(camera_count=0)
    zero.update(
        {
            "rung": "zero",
            "clean_steady_seconds": 120,
            "nvdec": {"hardware_branches": 0, "software_fallbacks": 0},
            "timeline": [],
            "workload": {
                "codec": "h264",
                "width": 1280,
                "height": 720,
                "fps": 15.0,
                "gop": 30,
                "camera_phase_offsets_ms": [],
            },
        }
    )
    (root / "raw" / "rung-zero.json").write_text(json.dumps(zero))
    policy_digest = hashlib.sha256(
        Path("scripts/qa/deepstream-canary/gate-policy.v1.json").read_bytes()
    ).hexdigest()
    (root / "run-request.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "requested_rungs": ["zero", "loopback"],
                "policy_sha256": policy_digest,
            }
        )
        + "\n"
    )

    # When: the independent verifier evaluates the immutable requested set.
    result = _verify("canary", root)

    # Then: an available zero PASS cannot hide the absent loopback receipt.
    assert result.returncode == 1
    verdict = json.loads((root / "canary.json").read_text())
    assert "requested_rung_missing:loopback" in verdict["findings"]


def test_canary_verifier_rejects_low_per_camera_fps(tmp_path: Path) -> None:
    # Given: a raw receipt whose aggregate shape looks complete but camera FPS is below policy.
    root = tmp_path / "canary-red"
    (root / "raw").mkdir(parents=True)
    receipt = json.dumps(_passing_receipt()).replace(
        '"fps_windows": [14.8, 15.0, 15.2]', '"fps_windows": [1.0, 1.0, 1.0]'
    )
    (root / "raw" / "rung-loopback.json").write_text(receipt)

    # When: the independent canary verifier recomputes gate math.
    result = _verify("canary", root)

    # Then: intentional RED exits non-zero and cannot trust a self-reported PASS.
    assert result.returncode == 1
    assert json.loads((root / "canary.json").read_text())["verdict"] == "FAIL"


def test_compliance_verifier_rejects_receipt_digest_tamper(tmp_path: Path) -> None:
    # Given: rendered evidence whose tracked compose bytes are changed after manifesting.
    root = tmp_path / "compliance-red"
    _render_evidence(root)
    with (root / "compose.rendered.yaml").open("a") as target:
        _ = target.write("# tampered\n")

    # When: the independent compliance verifier hashes raw bytes.
    result = _verify("compliance", root)

    # Then: intentional RED is detected.
    assert result.returncode == 1


def test_scope_verifier_rejects_non_loopback_port(tmp_path: Path) -> None:
    # Given: a rendered compose with a public host bind.
    root = tmp_path / "scope-red"
    _render_evidence(root)
    compose = root / "compose.rendered.yaml"
    compose.write_text(compose.read_text().replace("127.0.0.1:18090", "0.0.0.0:18090"))

    # When: scope invariants are independently checked.
    result = _verify("scope", root)

    # Then: intentional RED is detected.
    assert result.returncode == 1


def test_quality_verifier_rejects_failed_gate_report(tmp_path: Path) -> None:
    # Given: an independently generated FAIL report from a low-FPS raw receipt.
    root = tmp_path / "quality-red"
    (root / "raw").mkdir(parents=True)
    receipt = json.dumps(_passing_receipt()).replace(
        '"fps_windows": [14.8, 15.0, 15.2]', '"fps_windows": [1.0]'
    )
    (root / "raw" / "rung-loopback.json").write_text(receipt)
    assert _verify("canary", root).returncode == 1

    # When: the quality verifier evaluates the canonical report.
    result = _verify("quality", root)

    # Then: intentional RED is detected.
    assert result.returncode == 1


def test_report_filename_is_content_addressed(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.report import write_canonical_report

    # Given: a machine-consumed report.
    report = {"schema_version": 1, "verdict": "PASS", "checks": []}

    # When: it is persisted.
    path = write_canonical_report(tmp_path, report)

    # Then: the filename digest equals the canonical bytes.
    assert path.name == f"gate-report.{hashlib.sha256(path.read_bytes()).hexdigest()}.json"
    assert os.stat(path).st_mode & 0o222 == 0
