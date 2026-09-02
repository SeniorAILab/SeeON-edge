# /// script
# requires-python = ">=3.11"
# ///
# --- How to run ---
# uv run python scripts/qa/verify_deepstream_delivery.py canary \
#   --evidence-root <dir> --output <file>

"""Independently recompute C8 canary and final F1/F2/F4 verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import ClassVar, Literal, assert_never, final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.qa.deepstream_final_verification import compliance, quality, scope  # noqa: E402
from worker.tools.deepstream_canary.gates import (  # noqa: E402
    evaluate_absolute_receipt,
    evaluate_receipt,
)
from worker.tools.deepstream_canary.models import (  # noqa: E402
    ArtifactBindings,
    AuthorizationArtifact,
    CameraSignals,
    CanaryMode,
    GatePolicy,
    GateReport,
    GpuSignals,
    LatencySignals,
    LiveProtectionSignals,
    NvdecSignals,
    RungReceipt,
    TimelineEntry,
    WorkloadFacts,
)
from worker.tools.deepstream_canary.report import write_canonical_report  # noqa: E402
from worker.tools.deepstream_canary.telemetry import (  # noqa: E402
    RecordedRungTelemetry,
    build_rung_receipt,
)

POLICY = Path("scripts/qa/deepstream-canary/gate-policy.v1.json")
LEGACY_BASELINE_POLICY_SHA256 = (
    "95bb74fd48e242ced4a640399f403321bd45df1d70408e163a284584dcbfefa5"
)


class RunRequestReceipt(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    requested_rungs: tuple[str, ...]
    policy_sha256: str
    worker_image: str | None = None
    worker_image_digest: str | None = None
    expected_revision: str | None = None
    appliance_id: str = "unbound-canary-appliance"
    camera_ids: tuple[str, ...] = ()
    authorization_sha256: str | None = None


class LegacyBaselineAuthorizationAttestation(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    kind: Literal["legacy_baseline_authorization"]
    appliance_id: str
    authorization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rung_receipts: dict[str, str]

    @field_validator("rung_receipts")
    @classmethod
    def rung_receipt_hashes_are_sha256(cls, values: dict[str, str]) -> dict[str, str]:
        if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in values.values()):
            raise ValueError("rung receipt hashes must be lowercase SHA-256 digests")
        return values


class DeliveryVerdict(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    verdict: Literal["PASS", "FAIL", "APPROVE", "REJECT"]
    verifier: Literal["canary", "compliance", "quality", "scope"]
    # Content identity of the gate semantics that produced this verdict: the
    # SHA-256 over this verifier plus gates.py, so a stored verdict names the
    # exact decision rules even when the policy JSON digest is unchanged.
    verifier_revision: str | None = None
    reports: tuple[GateReport, ...] = ()
    findings: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()


class V1RecordedCameraTelemetry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    camera_id: str
    decision_window_counts: tuple[int, ...] = Field(min_length=1)
    decision_window_seconds: float = Field(gt=0)
    latency_samples_ms: tuple[float, ...] = Field(min_length=1)
    au_gaps: int = Field(ge=0)
    config_discontinuities: int = Field(ge=0)
    timestamp_discontinuities: int = Field(ge=0)
    metadata_published: int = Field(gt=0)
    metadata_overwritten: int = Field(ge=0)
    event_evidence_parity: bool
    preview_ok: bool
    derivative_ok: bool


class V1RecordedGpuTelemetry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    child_pid: int = Field(gt=0)
    warmup_memory_mib: tuple[float, ...] = Field(min_length=1)
    steady_memory_mib: tuple[float, ...] = Field(min_length=1)
    recovery_memory_mib: tuple[float, ...] = Field(min_length=1)
    global_used_mib: float = Field(ge=0)
    total_mib: float = Field(gt=0)
    slack_samples_mib: tuple[float, ...] = Field(min_length=1)
    utilization_samples: tuple[float, ...] = Field(min_length=1)
    new_xids: tuple[int, ...]


class V1RecordedRungTelemetry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    rung: str
    mode: CanaryMode
    camera_count: int = Field(ge=0)
    clean_steady_seconds: int = Field(ge=0)
    cameras: tuple[V1RecordedCameraTelemetry, ...]
    gpu: V1RecordedGpuTelemetry
    nvdec: NvdecSignals
    timeline: tuple[TimelineEntry, ...]
    live_protection: LiveProtectionSignals
    fault_windows: tuple[str, ...]
    workload: WorkloadFacts


class V1CameraSignals(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    camera_id: str
    fps_windows: tuple[float, ...] = Field(min_length=1)
    latency_ms: LatencySignals
    au_gaps: int = Field(ge=0)
    config_discontinuities: int = Field(ge=0)
    timestamp_discontinuities: int = Field(ge=0)
    metadata_published: int = Field(gt=0)
    metadata_overwritten: int = Field(ge=0)
    event_evidence_parity: bool
    preview_ok: bool
    derivative_ok: bool


class V1RungReceipt(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    rung: str
    mode: CanaryMode
    camera_count: int = Field(ge=0)
    clean_steady_seconds: int = Field(ge=0)
    cameras: tuple[V1CameraSignals, ...]
    gpu: GpuSignals
    nvdec: NvdecSignals
    timeline: tuple[TimelineEntry, ...]
    live_protection: LiveProtectionSignals
    fault_windows: tuple[str, ...]
    workload: WorkloadFacts
    artifacts: ArtifactBindings


BaselineReceipt = RungReceipt | V1RungReceipt


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _build_v1_rung_receipt(
    recorded: V1RecordedRungTelemetry, artifacts: ArtifactBindings
) -> V1RungReceipt:
    return V1RungReceipt(
        schema_version=1,
        rung=recorded.rung,
        mode=recorded.mode,
        camera_count=recorded.camera_count,
        clean_steady_seconds=recorded.clean_steady_seconds,
        cameras=tuple(
            V1CameraSignals(
                camera_id=camera.camera_id,
                fps_windows=tuple(
                    count / camera.decision_window_seconds
                    for count in camera.decision_window_counts
                ),
                latency_ms=LatencySignals(
                    p50=_percentile(camera.latency_samples_ms, 0.50),
                    p95=_percentile(camera.latency_samples_ms, 0.95),
                    p99=_percentile(camera.latency_samples_ms, 0.99),
                    max=max(camera.latency_samples_ms),
                ),
                au_gaps=camera.au_gaps,
                config_discontinuities=camera.config_discontinuities,
                timestamp_discontinuities=camera.timestamp_discontinuities,
                metadata_published=camera.metadata_published,
                metadata_overwritten=camera.metadata_overwritten,
                event_evidence_parity=camera.event_evidence_parity,
                preview_ok=camera.preview_ok,
                derivative_ok=camera.derivative_ok,
            )
            for camera in recorded.cameras
        ),
        gpu=GpuSignals(
            child_pid=recorded.gpu.child_pid,
            warmup_peak_mib=max(recorded.gpu.warmup_memory_mib),
            steady_p50_mib=_percentile(recorded.gpu.steady_memory_mib, 0.50),
            steady_p95_mib=_percentile(recorded.gpu.steady_memory_mib, 0.95),
            recovery_mib=recorded.gpu.recovery_memory_mib[-1],
            global_used_mib=recorded.gpu.global_used_mib,
            total_mib=recorded.gpu.total_mib,
            minimum_slack_mib=min(recorded.gpu.slack_samples_mib),
            utilization_p50=_percentile(recorded.gpu.utilization_samples, 0.50),
            utilization_p95=_percentile(recorded.gpu.utilization_samples, 0.95),
            utilization_max=max(recorded.gpu.utilization_samples),
            new_xids=recorded.gpu.new_xids,
        ),
        nvdec=recorded.nvdec,
        timeline=recorded.timeline,
        live_protection=recorded.live_protection,
        fault_windows=recorded.fault_windows,
        workload=recorded.workload,
        artifacts=artifacts,
    )


def _load_rung_receipts(
    root: Path, *, allow_v1: bool = False
) -> tuple[tuple[BaselineReceipt, bytes], ...]:
    def load(path: Path) -> tuple[BaselineReceipt, bytes]:
        content = path.read_bytes()
        if allow_v1 and json.loads(content)["schema_version"] == 1:
            return V1RungReceipt.model_validate_json(content), content
        return RungReceipt.model_validate_json(content), content

    return tuple(
        load(path) for path in sorted((root / "raw").glob("rung-*.json"))
    )


def _receipt_evidence_findings(
    root: Path,
    request: RunRequestReceipt,
    receipts: dict[str, BaselineReceipt],
    prefix: str,
    rungs: tuple[str, ...] | None = None,
) -> list[str]:
    findings: list[str] = []
    for rung in rungs or request.requested_rungs:
        receipt = receipts.get(rung)
        if receipt is None:
            continue
        telemetry_path = root / "raw" / f"telemetry-{rung}.json"
        if not telemetry_path.is_file():
            findings.append(f"{prefix}raw_telemetry_missing:{rung}")
            continue
        try:
            if isinstance(receipt, V1RungReceipt):
                telemetry = V1RecordedRungTelemetry.model_validate_json(
                    telemetry_path.read_bytes()
                )
                rebuilt = _build_v1_rung_receipt(telemetry, receipt.artifacts)
            else:
                telemetry = RecordedRungTelemetry.model_validate_json(
                    telemetry_path.read_bytes()
                )
                rebuilt = build_rung_receipt(telemetry, receipt.artifacts)
        except (OSError, ValidationError, ValueError) as error:
            findings.append(f"{prefix}raw_telemetry_invalid:{rung}:{error}")
            continue
        if rebuilt != receipt:
            findings.append(f"{prefix}receipt_raw_mismatch:{rung}")
    return findings


def _v1_as_current(receipt: V1RungReceipt) -> RungReceipt:
    return RungReceipt(
        schema_version=2,
        rung=receipt.rung,
        mode=receipt.mode,
        camera_count=receipt.camera_count,
        clean_steady_seconds=receipt.clean_steady_seconds,
        cameras=tuple(
            CameraSignals(
                camera_id=camera.camera_id,
                fps_windows=camera.fps_windows,
                telemetry_coverage_seconds=receipt.clean_steady_seconds,
                copy_window_frames=1,
                frame_window_spans_seconds=(0.0,),
                h2d_bytes_max=0,
                d2h_bytes_max=0,
                box_source="pose",
                pool_wait_us_p95=0,
                gpu_us_p95=0,
                surface_drops=0,
                latency_ms=camera.latency_ms,
                au_gaps=camera.au_gaps,
                config_discontinuities=camera.config_discontinuities,
                timestamp_discontinuities=camera.timestamp_discontinuities,
                metadata_published=camera.metadata_published,
                metadata_overwritten=camera.metadata_overwritten,
                event_evidence_parity=camera.event_evidence_parity,
                preview_ok=camera.preview_ok,
                derivative_ok=camera.derivative_ok,
            )
            for camera in receipt.cameras
        ),
        gpu=receipt.gpu,
        nvdec=receipt.nvdec,
        timeline=receipt.timeline,
        live_protection=receipt.live_protection,
        fault_windows=receipt.fault_windows,
        workload=receipt.workload,
        artifacts=receipt.artifacts,
    )


def _evaluate_baseline(receipt: BaselineReceipt, policy: GatePolicy) -> GateReport:
    if isinstance(receipt, RungReceipt):
        return evaluate_absolute_receipt(receipt, policy)
    report = evaluate_absolute_receipt(_v1_as_current(receipt), policy)
    candidate_only = (
        ".h2d_bytes_max",
        ".d2h_bytes_max",
        ".copy_window_frames",
        ".frame_window_span_max_seconds",
        ".telemetry_coverage_seconds",
        ".surface_drops",
        "fault_windows",
    )
    checks = tuple(
        check
        for check in report.checks
        if not check.name.endswith(candidate_only) and check.name != "fault_windows"
    )
    verdict = "PASS" if all(check.passed for check in checks) else "FAIL"
    return report.model_copy(
        update={
            "verdict": verdict,
            "claim_eligible": verdict == "PASS"
            and receipt.mode is CanaryMode.COMMISSIONING,
            "checks": checks,
        }
    )


def _authorization_findings(
    root: Path, request: RunRequestReceipt, rungs: tuple[str, ...], prefix: str
) -> list[str]:
    live_rungs = tuple(rung for rung in rungs if rung.isdigit())
    authorization_path = root / "authorization.json"
    if not live_rungs:
        if request.authorization_sha256 is not None or authorization_path.exists():
            return [f"{prefix}unexpected_authorization_artifact"]
        return []
    if request.authorization_sha256 is None:
        return [f"{prefix}authorization_digest_missing"]
    if not authorization_path.is_file():
        return [f"{prefix}authorization_missing"]
    content = authorization_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != request.authorization_sha256:
        return [f"{prefix}authorization_digest_mismatch"]
    try:
        artifact = AuthorizationArtifact.model_validate_json(content)
    except ValidationError as error:
        return [f"{prefix}authorization_invalid:{error}"]
    findings: list[str] = []
    if artifact.appliance_id != request.appliance_id:
        findings.append(f"{prefix}authorization_appliance_id_mismatch")
    if artifact.worker_image != request.worker_image_digest:
        findings.append(f"{prefix}authorization_worker_image_mismatch")
    if artifact.camera_ids != request.camera_ids:
        findings.append(f"{prefix}authorization_camera_ids_mismatch")
    for rung in live_rungs:
        if int(rung) not in artifact.authorized_rungs:
            findings.append(f"{prefix}authorization_rung_unauthorized:{rung}")
    return findings


def _legacy_baseline_authorization_findings(
    root: Path,
    request: RunRequestReceipt,
    rungs: tuple[str, ...],
    receipt_records: tuple[tuple[BaselineReceipt, bytes], ...],
) -> list[str]:
    authorization_path = root / "authorization.json"
    attestation_path = root / "legacy-authorization-attestation.json"
    findings: list[str] = []
    if not authorization_path.is_file():
        findings.append("baseline_legacy_authorization_missing")
    if not attestation_path.is_file():
        findings.append("baseline_legacy_authorization_attestation_missing")
    if findings:
        return findings
    try:
        attestation = LegacyBaselineAuthorizationAttestation.model_validate_json(
            attestation_path.read_bytes()
        )
    except (OSError, ValidationError) as error:
        return [f"baseline_legacy_authorization_attestation_invalid:{error}"]

    authorization_content = authorization_path.read_bytes()
    request_content = (root / "run-request.json").read_bytes()
    if hashlib.sha256(authorization_content).hexdigest() != attestation.authorization_sha256:
        findings.append("baseline_legacy_authorization_authorization_sha256_mismatch")
    if hashlib.sha256(request_content).hexdigest() != attestation.run_request_sha256:
        findings.append("baseline_legacy_authorization_run_request_sha256_mismatch")
    receipt_contents = {receipt.rung: content for receipt, content in receipt_records}
    for rung in rungs:
        expected_digest = attestation.rung_receipts.get(rung)
        if expected_digest is None:
            findings.append(f"baseline_legacy_authorization_rung_receipt_missing:{rung}")
        elif hashlib.sha256(receipt_contents[rung]).hexdigest() != expected_digest:
            findings.append(
                f"baseline_legacy_authorization_rung_receipt_sha256_mismatch:{rung}"
            )
    try:
        artifact = AuthorizationArtifact.model_validate_json(authorization_content)
    except ValidationError as error:
        findings.append(f"baseline_legacy_authorization_invalid:{error}")
        return findings
    if artifact.appliance_id != attestation.appliance_id:
        findings.append("baseline_legacy_authorization_appliance_id_mismatch")
    if artifact.worker_image != request.worker_image_digest:
        findings.append("baseline_legacy_authorization_worker_image_mismatch")
    if not rungs or len(artifact.camera_ids) != max(int(rung) for rung in rungs):
        findings.append("baseline_legacy_authorization_camera_count_mismatch")
    for rung in rungs:
        if int(rung) not in artifact.authorized_rungs:
            findings.append(f"baseline_legacy_authorization_rung_unauthorized:{rung}")
    return findings


def _canary(
    root: Path, policy_path: Path, baseline_root: Path | None = None
) -> DeliveryVerdict:
    policy_content = policy_path.read_bytes()
    policy = GatePolicy.model_validate_json(policy_content)
    request = RunRequestReceipt.model_validate_json((root / "run-request.json").read_bytes())
    policy_digest = hashlib.sha256(policy_content).hexdigest()
    receipt_records = _load_rung_receipts(root)
    receipts = tuple(receipt for receipt, _ in receipt_records)
    by_rung = {receipt.rung: receipt for receipt in receipts}
    digit_rungs = tuple(rung for rung in request.requested_rungs if rung.isdigit())
    baseline_request = (
        RunRequestReceipt.model_validate_json((baseline_root / "run-request.json").read_bytes())
        if baseline_root is not None and digit_rungs
        else None
    )
    baseline_records = (
        _load_rung_receipts(baseline_root, allow_v1=True)
        if baseline_root is not None and digit_rungs
        else ()
    )
    baseline_by_rung = {
        receipt.rung: receipt for receipt, _ in baseline_records
    }
    baseline_digit_rungs = (
        tuple(
            rung for rung in baseline_request.requested_rungs if rung.isdigit()
        )
        if baseline_request is not None
        else ()
    )
    v1_baseline = bool(baseline_digit_rungs) and all(
        isinstance(baseline_by_rung.get(rung), V1RungReceipt)
        for rung in baseline_digit_rungs
    )
    findings = [
        f"requested_rung_missing:{rung}"
        for rung in request.requested_rungs
        if rung not in by_rung
    ]
    if request.policy_sha256 != policy_digest:
        findings.append("run_request_policy_digest_mismatch")
    findings.extend(_receipt_evidence_findings(root, request, by_rung, "candidate_"))
    findings.extend(
        _authorization_findings(root, request, request.requested_rungs, "candidate_")
    )
    if baseline_root is None:
        findings.extend(
            f"baseline_evidence_missing:{rung}"
            for rung in digit_rungs
        )
    if baseline_request is not None:
        if v1_baseline and baseline_request.policy_sha256 != LEGACY_BASELINE_POLICY_SHA256:
            findings.append("baseline_legacy_policy_digest_unapproved")
        elif not v1_baseline and baseline_request.policy_sha256 != policy_digest:
            findings.append("baseline_run_request_policy_digest_mismatch")
        findings.extend(
            f"baseline_requested_rung_missing:{rung}"
            for rung in baseline_request.requested_rungs
            if rung not in baseline_by_rung
        )
        findings.extend(
            f"baseline_requested_rung_missing:{rung}"
            for rung in digit_rungs
            if rung not in baseline_request.requested_rungs or rung not in baseline_by_rung
        )
        findings.extend(
            _receipt_evidence_findings(
                baseline_root,
                baseline_request,
                baseline_by_rung,
                "baseline_",
                digit_rungs,
            )
        )
        findings.extend(
            _legacy_baseline_authorization_findings(
                baseline_root,
                baseline_request,
                baseline_digit_rungs,
                baseline_records,
            )
            if v1_baseline
            else _authorization_findings(
                baseline_root,
                baseline_request,
                baseline_request.requested_rungs,
                "baseline_",
            )
        )
    identity_artifacts: list[str] = [
        f"policy:{policy_digest}",
        *(
            f"candidate-rung:{receipt.rung}:{hashlib.sha256(content).hexdigest()}"
            for receipt, content in receipt_records
        ),
        *(
            f"baseline-rung:{receipt.rung}:{hashlib.sha256(content).hexdigest()}"
            for receipt, content in baseline_records
        ),
    ]
    if request.worker_image_digest is not None:
        expected_suffix = f"@{request.worker_image_digest}"
        if request.worker_image is None or not request.worker_image.endswith(expected_suffix):
            findings.append("run_request_worker_image_identity_mismatch")
        identity_artifacts.append(f"worker-image:{request.worker_image_digest}")
        findings.extend(
            f"worker_image_digest_mismatch:{receipt.rung}"
            for receipt in receipts
            if receipt.artifacts.worker_image != request.worker_image_digest
        )
        if baseline_request is not None:
            findings.extend(
                f"baseline_worker_image_digest_mismatch:{receipt.rung}"
                for receipt in baseline_by_rung.values()
                if receipt.artifacts.worker_image != baseline_request.worker_image_digest
            )
    if request.expected_revision is not None:
        if re.fullmatch(r"[0-9a-f]{40}", request.expected_revision) is None:
            findings.append("run_request_expected_revision_invalid")
        identity_artifacts.append(f"expected-revision:{request.expected_revision}")
    baseline_reports = tuple(
        _evaluate_baseline(baseline_by_rung[rung], policy)
        for rung in digit_rungs
        if rung in baseline_by_rung
    )
    if baseline_request is not None:
        findings.extend(
            f"baseline_absolute_failure:{report.rung}"
            for report in baseline_reports
            if report.verdict != "PASS" or not report.claim_eligible
        )
    qualified_baselines = {
        report.rung: _v1_as_current(baseline_by_rung[report.rung])
        if isinstance(baseline_by_rung[report.rung], V1RungReceipt)
        else baseline_by_rung[report.rung]
        for report in baseline_reports
        if report.verdict == "PASS"
        and report.claim_eligible
        and not any(finding.startswith("baseline_") for finding in findings)
        and not any(finding.endswith(f":{report.rung}") for finding in findings)
    }
    reports = tuple(
        evaluate_receipt(by_rung[rung], policy, qualified_baselines.get(rung))
        for rung in request.requested_rungs
        if rung in by_rung
    )
    for report in reports:
        _ = write_canonical_report(root, report.model_dump(mode="json"))
    passed = (
        not findings
        and bool(reports)
        and len(reports) == len(request.requested_rungs)
        and all(report.verdict == "PASS" and report.claim_eligible for report in reports)
    )
    return DeliveryVerdict(
        verdict="PASS" if passed else "FAIL",
        verifier="canary",
        reports=reports,
        findings=tuple(findings),
        artifacts=tuple(identity_artifacts),
    )


def _legacy_scope(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    compose = (root / "compose.rendered.yaml").read_text(encoding="utf-8")
    findings: list[str] = []
    required = ("name: seeon-ds-canary", "internal: true", "127.0.0.1:18090:8090")
    findings.extend(f"missing:{token}" for token in required if token not in compose)
    if ".env.edge.prod" in compose:
        findings.append("production_env_reference")
    published: list[str] = re.findall(
        r"- [\"']?([^\n\"']+:[0-9]+:[0-9]+)[\"']?", compose
    )
    findings.extend(
        f"non_loopback_port:{port}" for port in published if not port.startswith("127.0.0.1:")
    )
    compose_digest = hashlib.sha256(compose.encode()).hexdigest()
    return tuple(findings), (f"compose.rendered.yaml:{compose_digest}",)


@final
class VerifyArguments(argparse.Namespace):
    verifier: Literal["canary", "compliance", "quality", "scope"] = "canary"
    evidence_root: Path | None = None
    baseline_evidence_root: Path | None = None
    output: Path = Path()
    policy: Path = POLICY
    plan: Path | None = None
    ledger: Path | None = None
    require_tasks: str = ""
    require_prs: int = 0
    inject: str | None = None
    base_sha: str | None = None
    head_sha: str | None = None
    allowed_plan: Path | None = None
    inject_forbidden_path: str | None = None


def _arguments() -> VerifyArguments:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("verifier", choices=("canary", "compliance", "quality", "scope"))
    _ = parser.add_argument("--evidence-root", type=Path)
    _ = parser.add_argument("--baseline-evidence-root", type=Path)
    _ = parser.add_argument("--output", type=Path, required=True)
    _ = parser.add_argument("--policy", type=Path, default=POLICY)
    _ = parser.add_argument("--plan", type=Path)
    _ = parser.add_argument("--ledger", type=Path)
    _ = parser.add_argument("--require-tasks", default="")
    _ = parser.add_argument("--require-prs", type=int, default=0)
    _ = parser.add_argument("--inject", choices=("missing-native-ctest",))
    _ = parser.add_argument("--base-sha")
    _ = parser.add_argument("--head-sha")
    _ = parser.add_argument("--allowed-plan", type=Path)
    _ = parser.add_argument("--inject-forbidden-path")
    arguments = VerifyArguments()
    _ = parser.parse_args(namespace=arguments)
    return arguments


def _final_verdict(
    verifier: Literal["compliance", "quality", "scope"],
    result: tuple[tuple[str, ...], tuple[str, ...]],
) -> DeliveryVerdict:
    findings, artifacts = result
    return DeliveryVerdict(
        verdict="APPROVE" if not findings else "REJECT",
        verifier=verifier,
        findings=findings,
        artifacts=artifacts,
    )


def main() -> int:
    arguments = _arguments()
    try:
        match arguments.verifier:
            case "canary":
                if arguments.evidence_root is None:
                    raise FileNotFoundError("--evidence-root is required for canary")
                verdict = _canary(
                    arguments.evidence_root,
                    arguments.policy,
                    arguments.baseline_evidence_root,
                )
            case "compliance":
                if arguments.evidence_root is None:
                    raise FileNotFoundError("--evidence-root is required for compliance")
                tasks = tuple(int(item) for item in arguments.require_tasks.split(",") if item)
                verdict = _final_verdict(
                    "compliance",
                    compliance(
                        arguments.evidence_root,
                        arguments.plan,
                        arguments.ledger,
                        tasks,
                        arguments.require_prs,
                    ),
                )
            case "quality":
                if arguments.evidence_root is None:
                    raise FileNotFoundError("--evidence-root is required for quality")
                verdict = _final_verdict(
                    "quality", quality(arguments.evidence_root, arguments.inject)
                )
            case "scope":
                if (
                    arguments.base_sha is not None
                    and arguments.head_sha is not None
                    and arguments.allowed_plan is not None
                ):
                    result = scope(
                        arguments.base_sha,
                        arguments.head_sha,
                        arguments.allowed_plan,
                        arguments.inject_forbidden_path,
                    )
                elif arguments.evidence_root is not None:
                    result = _legacy_scope(arguments.evidence_root)
                else:
                    raise FileNotFoundError("scope requires Git range inputs or --evidence-root")
                verdict = _final_verdict("scope", result)
            case unreachable:
                assert_never(unreachable)
    except (OSError, ValidationError, ValueError, KeyError, json.JSONDecodeError) as error:
        verdict = DeliveryVerdict(
            verdict="FAIL" if arguments.verifier == "canary" else "REJECT",
            verifier=arguments.verifier,
            findings=(str(error),),
        )
    revision = hashlib.sha256(
        Path(__file__).read_bytes()
        + (REPOSITORY_ROOT / "worker/tools/deepstream_canary/gates.py").read_bytes()
    ).hexdigest()
    verdict = verdict.model_copy(update={"verifier_revision": revision})
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(verdict.model_dump_json() + "\n", encoding="utf-8")
    print(verdict.model_dump_json())
    return 0 if verdict.verdict in {"PASS", "APPROVE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
