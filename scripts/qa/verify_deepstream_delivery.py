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

from pydantic import BaseModel, ConfigDict, ValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.qa.deepstream_final_verification import compliance, quality, scope  # noqa: E402
from worker.tools.deepstream_canary.gates import (  # noqa: E402
    evaluate_absolute_receipt,
    evaluate_receipt,
)
from worker.tools.deepstream_canary.models import (  # noqa: E402
    AuthorizationArtifact,
    GatePolicy,
    GateReport,
    RungReceipt,
)
from worker.tools.deepstream_canary.report import write_canonical_report  # noqa: E402
from worker.tools.deepstream_canary.telemetry import (  # noqa: E402
    RecordedRungTelemetry,
    build_rung_receipt,
)

POLICY = Path("scripts/qa/deepstream-canary/gate-policy.v1.json")


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


class DeliveryVerdict(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    verdict: Literal["PASS", "FAIL", "APPROVE", "REJECT"]
    verifier: Literal["canary", "compliance", "quality", "scope"]
    reports: tuple[GateReport, ...] = ()
    findings: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()


def _load_rung_receipts(root: Path) -> tuple[tuple[RungReceipt, bytes], ...]:
    return tuple(
        (RungReceipt.model_validate_json(content := path.read_bytes()), content)
        for path in sorted((root / "raw").glob("rung-*.json"))
    )


def _receipt_evidence_findings(
    root: Path,
    request: RunRequestReceipt,
    receipts: dict[str, RungReceipt],
    prefix: str,
) -> list[str]:
    findings: list[str] = []
    for rung in request.requested_rungs:
        receipt = receipts.get(rung)
        if receipt is None:
            continue
        telemetry_path = root / "raw" / f"telemetry-{rung}.json"
        if not telemetry_path.is_file():
            findings.append(f"{prefix}raw_telemetry_missing:{rung}")
            continue
        try:
            telemetry = RecordedRungTelemetry.model_validate_json(telemetry_path.read_bytes())
            rebuilt = build_rung_receipt(telemetry, receipt.artifacts)
        except (OSError, ValidationError, ValueError) as error:
            findings.append(f"{prefix}raw_telemetry_invalid:{rung}:{error}")
            continue
        if rebuilt != receipt:
            findings.append(f"{prefix}receipt_raw_mismatch:{rung}")
    return findings


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
    baseline_request = (
        RunRequestReceipt.model_validate_json((baseline_root / "run-request.json").read_bytes())
        if baseline_root is not None
        else None
    )
    baseline_records = _load_rung_receipts(baseline_root) if baseline_root is not None else ()
    baseline_by_rung = {
        receipt.rung: receipt for receipt, _ in baseline_records
    }
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
            for rung in request.requested_rungs
            if rung.isdigit()
        )
    if baseline_root is not None and baseline_request is not None:
        if baseline_request.policy_sha256 != policy_digest:
            findings.append("baseline_run_request_policy_digest_mismatch")
        findings.extend(
            f"baseline_requested_rung_missing:{rung}"
            for rung in baseline_request.requested_rungs
            if rung not in baseline_by_rung
        )
        findings.extend(
            f"baseline_requested_rung_missing:{rung}"
            for rung in request.requested_rungs
            if rung.isdigit()
            and (rung not in baseline_request.requested_rungs or rung not in baseline_by_rung)
        )
        findings.extend(
            _receipt_evidence_findings(
                baseline_root, baseline_request, baseline_by_rung, "baseline_"
            )
        )
        findings.extend(
            _authorization_findings(
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
        evaluate_absolute_receipt(baseline_by_rung[rung], policy)
        for rung in request.requested_rungs
        if rung in baseline_by_rung
    )
    if baseline_root is not None:
        findings.extend(
            f"baseline_absolute_failure:{report.rung}"
            for report in baseline_reports
            if report.verdict != "PASS" or not report.claim_eligible
        )
    qualified_baselines = {
        report.rung: baseline_by_rung[report.rung]
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
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(verdict.model_dump_json() + "\n", encoding="utf-8")
    print(verdict.model_dump_json())
    return 0 if verdict.verdict in {"PASS", "APPROVE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
