from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from backend.app.edge_db import CheckViolation, DataError, NotNullViolation
from backend.app.edge_db.authority import AuthorityFenced
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    empty_detail,
)
from backend.app.features.audit.http import audit_runtime, mutation_audit
from backend.app.features.audit.store import AuditEvent
from backend.app.features.audit.store import utc_now as audit_now
from backend.app.features.cameras.router import (
    acknowledge_applied_detection_policies,
    worker_config_snapshot,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.update_command import CameraUpdate
from backend.app.features.evidence.event_outbox import (
    AcceptedEvent,
    EventIdentityConflict,
    EventOutbox,
    OutboxCapacityExceeded,
)
from backend.app.features.evidence.outbox_delivery import DeliveryStatus, OutboxDelivery
from backend.app.features.evidence.outbox_dispatch import RELAY_OUTBOX_BUDGET, dispatch
from backend.app.features.evidence.postgres_relay_projection import (
    PostgresRelayEvidenceProjection,
)
from backend.app.features.evidence.relay_projection import (
    RelayEvent,
    RelayEvidenceProjectionConflict,
    RelayEvidenceProjectionError,
    RelayEvidenceProjectionMissingEvent,
    RelaySnapshot,
)
from backend.app.features.status.heartbeat_store import get_heartbeat_store
from backend.app.features.status.runtime_status_store import get_runtime_status_store
from backend.app.shared.http.relay_http import (
    RELAY_TOKEN_HEADER,
    authorize_relay,
    authorize_relay_body,
    bounded_body_route,
    camera_binding,
)
from contracts import AlertEventType
from contracts.decode_diagnostics import DECODE_BACKENDS, DECODE_FALLBACK_REASONS
from contracts.worker_config import RESTART_EPOCH_KEY
from shared.events import envelope_limits
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.relay_failure_log import RelayFailureLog

logger = logging.getLogger(__name__)

MAX_INLINE_SNAPSHOT_BYTES = 200 * 1024
MAX_INLINE_SNAPSHOT_BASE64_CHARS = 4 * ((MAX_INLINE_SNAPSHOT_BYTES + 2) // 3)
MAX_RELAY_REQUEST_BODY_BYTES = 512 * 1024
MAX_RELAY_HEARTBEAT_BODY_BYTES = 4 * 1024
MAX_RELAY_RUNTIME_STATUS_BODY_BYTES = 64 * 1024
MAX_RELAY_SNAPSHOT_ATTACHMENT_BODY_BYTES = 8 * 1024
MAX_RELAY_SNAPSHOT_DISPOSITION_BODY_BYTES = 8 * 1024

_MAX_BODY_BYTES_BY_SUFFIX: dict[str, int] = {
    "/alerts": MAX_RELAY_REQUEST_BODY_BYTES,
    "/heartbeat": MAX_RELAY_HEARTBEAT_BODY_BYTES,
    "/runtime-status": MAX_RELAY_RUNTIME_STATUS_BODY_BYTES,
    "/snapshot-attachments": MAX_RELAY_SNAPSHOT_ATTACHMENT_BODY_BYTES,
    "/snapshot-dispositions": MAX_RELAY_SNAPSHOT_DISPOSITION_BODY_BYTES,
}


_LOGGER = logging.getLogger(__name__)

_backend_ingest_alert_failures = RelayFailureLog(
    _LOGGER, channel="backend ingest alerts", method="POST"
)

router = APIRouter(
    prefix="/relay", tags=["relay"], route_class=bounded_body_route(_MAX_BODY_BYTES_BY_SUFFIX)
)


def require_relay_alert(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> None:
    authorize_relay_body(request, max_bytes=MAX_RELAY_REQUEST_BODY_BYTES, relay_token=relay_token)


def require_relay_heartbeat(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> None:
    authorize_relay_body(request, max_bytes=MAX_RELAY_HEARTBEAT_BODY_BYTES, relay_token=relay_token)


def require_relay_runtime_status(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    authorize_relay_body(
        request,
        max_bytes=MAX_RELAY_RUNTIME_STATUS_BODY_BYTES,
        relay_token=relay_token,
        authorization=authorization,
    )


def require_relay_snapshot_attachment(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> None:
    authorize_relay_body(
        request, max_bytes=MAX_RELAY_SNAPSHOT_ATTACHMENT_BODY_BYTES, relay_token=relay_token
    )


def require_relay_snapshot_disposition(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> None:
    authorize_relay_body(
        request, max_bytes=MAX_RELAY_SNAPSHOT_DISPOSITION_BODY_BYTES, relay_token=relay_token
    )


class RelayAuditEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: int | None = None
    model_version: str | None = None
    detector_version: str | None = None
    operating_threshold: float | None = None
    clock_source: str | None = None
    runtime_manifest_sha256: str | None = None
    decision_trace_id: str | None = None


class RelaySnapshotMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    mime_type: str = Field(min_length=1)
    captured_at: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    edge_event_id: str | None = None


class RelaySnapshotAttachmentRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "description": (
                "An immutable snapshot reference; snapshot bytes never cross this route."
            ),
        },
    )

    edge_event_id: str = Field(min_length=1, max_length=envelope_limits.EDGE_EVENT_ID_MAX_CHARS)
    snapshot_id: str = Field(min_length=1, max_length=envelope_limits.SNAPSHOT_ID_MAX_CHARS)
    sha256: str = Field(
        min_length=envelope_limits.SHA256_MAX_CHARS,
        max_length=envelope_limits.SHA256_MAX_CHARS,
        pattern=r"^[0-9a-f]{64}$",
    )
    media_reference: str = Field(min_length=1, max_length=envelope_limits.MEDIA_REFERENCE_MAX_CHARS)
    size_bytes: int = Field(ge=0, le=envelope_limits.SNAPSHOT_SIZE_BYTES_MAX)
    mime_type: str = Field(min_length=1, max_length=envelope_limits.MIME_TYPE_MAX_CHARS)
    audit: RelayAuditEnvelope | None = None


class RelaySnapshotDispositionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "description": "A terminal, explicit statement that a snapshot cannot be delivered.",
        },
    )

    edge_event_id: str = Field(min_length=1, max_length=envelope_limits.EDGE_EVENT_ID_MAX_CHARS)
    snapshot_id: str = Field(min_length=1, max_length=envelope_limits.SNAPSHOT_ID_MAX_CHARS)
    disposition: str = Field(min_length=1, max_length=envelope_limits.DISPOSITION_MAX_CHARS)
    reason: str = Field(min_length=1, max_length=envelope_limits.DISPOSITION_REASON_MAX_CHARS)
    audit: RelayAuditEnvelope | None = None


def _envelope_encodable(value: object) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except ValueError:
        return False
    return True


class RelayAlertEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    domain: StrictStr | None = None
    identity: StrictStr | StrictInt | None = None
    time_sec: StrictInt | StrictFloat | None = None
    person_id: StrictInt | None = None
    bed_id: StrictInt | None = None
    clip_id: StrictStr | None = None

    @model_validator(mode="wrap")
    @classmethod
    def unencodable_evidence_is_left_to_the_envelope(
        cls, data: Any, handler: ModelWrapValidatorHandler[RelayAlertEvidence]
    ) -> RelayAlertEvidence:
        if isinstance(data, dict) and not _envelope_encodable(data):
            evidence = cls.model_construct()
            evidence.__pydantic_extra__ = dict(data)
            return evidence
        return handler(data)


def _evidence_values(evidence: RelayAlertEvidence | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    known = evidence.model_fields_set & set(RelayAlertEvidence.model_fields)
    return {**{key: getattr(evidence, key) for key in known}, **(evidence.model_extra or {})}


_EVIDENCE_JSON = TypeAdapter(dict[str, Any])


class RelayAlertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_event_id: str | None = Field(default=None, pattern=r"^[0-9a-f-]{36}$")
    event_type: AlertEventType
    probability: float = Field(ge=0.0, le=1.0)
    detected_at: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    facility_id: str = Field(min_length=1)
    resident_id: str | None = None
    evidence: RelayAlertEvidence | None = None
    audit: RelayAuditEnvelope | None = None
    snapshot_jpeg_base64: str | None = Field(
        default=None, max_length=MAX_INLINE_SNAPSHOT_BASE64_CHARS
    )
    attempt_ordinal: int | None = Field(default=None, ge=1)
    snapshot: RelaySnapshotMetadata | None = None

    @model_validator(mode="after")
    def snapshot_matches_inline_evidence(self) -> RelayAlertRequest:
        snapshot_bytes = _decode_snapshot(self.snapshot_jpeg_base64)
        if self.snapshot is not None:
            if self.snapshot.camera_id != self.camera_id:
                raise ValueError("snapshot camera_id must match alert camera_id")
            if self.snapshot.edge_event_id != self.edge_event_id:
                raise ValueError("snapshot edge_event_id must match alert edge_event_id")
            if snapshot_bytes is not None:
                if self.edge_event_id is None or self.snapshot.edge_event_id is None:
                    raise ValueError("inline snapshot requires an edge_event_id")
                if self.snapshot.mime_type != "image/jpeg":
                    raise ValueError("inline snapshot MIME type must be image/jpeg")
                if self.snapshot.size_bytes <= 0 or self.snapshot.size_bytes != len(snapshot_bytes):
                    raise ValueError("inline snapshot size_bytes must exactly match decoded bytes")
                if self.snapshot.sha256 != hashlib.sha256(snapshot_bytes).hexdigest():
                    raise ValueError("inline snapshot sha256 must match decoded bytes")
        return self


class RelayHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1)
    facility_id: str = Field(min_length=1)
    config_version: int | None = None


class RelayDecodeDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: str = Field(min_length=1)
    selected: str | None = Field(default=None)
    fallback_count: int = Field(ge=0)
    last_reason: str | None = Field(default=None)
    updated_at_sec: float = Field()

    @field_validator("requested", "selected")
    @classmethod
    def valid_backend(cls, value: str | None) -> str | None:
        if value is not None and value not in DECODE_BACKENDS:
            raise ValueError("decode backend is invalid")
        return value

    @field_validator("last_reason")
    @classmethod
    def valid_last_reason(cls, value: str | None) -> str | None:
        if value is not None and value not in DECODE_FALLBACK_REASONS:
            raise ValueError("last_reason is not a decode fallback reason")
        return value


class RelayDetectionStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: bool = Field()
    inference_admitted: int = Field(ge=0)
    inference_succeeded: int = Field(ge=0)
    inference_overwritten: int = Field(ge=0)
    decision_completed: int = Field(ge=0)

    @model_validator(mode="after")
    def counters_are_ordered(self) -> RelayDetectionStatus:
        if self.inference_succeeded > self.inference_admitted:
            raise ValueError("inference_succeeded cannot exceed inference_admitted")
        if self.decision_completed > self.inference_succeeded:
            raise ValueError("decision_completed cannot exceed inference_succeeded")
        return self


class RelayRuntimeStatusCamera(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1)
    decode: RelayDecodeDiagnostics
    measured_fps: float | None = Field(default=None, ge=0.0)
    detection: RelayDetectionStatus | None = Field(default=None)


class RelayGpuStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nvml_available: bool = Field()
    cuda_context_ok: bool = Field()
    driver_version: str | None = Field(default=None)
    device_name: str | None = Field(default=None)
    captured_at_sec: float = Field()
    nvml_error: str | None = Field(default=None)


class RelayWorkerStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alive: bool = Field()
    pid: int | None = Field(default=None, ge=0)
    started_at_sec: float | None = Field(default=None)
    profile_boot_error: str | None = Field(default=None)


class RelayClipExportStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = Field()
    version: int = Field(ge=0)


class RelayClipRecorderStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = Field()
    dropped_frames: int | None = Field(default=None, ge=0)
    dropped_events: int | None = Field(default=None, ge=0)
    failed_writes: int | None = Field(default=None, ge=0)
    finalized_clips: int | None = Field(default=None, ge=0)
    video_unavailable_clips: int | None = Field(default=None, ge=0)
    active_clips: int | None = Field(default=None, ge=0)
    encoder: str | None = Field(default=None)


class RelayDeliveryQueueStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted_count: int = Field(ge=0)
    accepted_bytes: int = Field(ge=0)
    max_accepted_entries: int = Field(gt=0)
    max_accepted_bytes: int = Field(gt=0)
    by_kind: dict[str, int] = Field()
    dead_lettered_count: int = Field(default=0, ge=0)
    dead_lettered_bytes: int = Field(default=0, ge=0)
    oldest_event_accepted_at: str | None = Field(default=None)


class RelayRuntimeStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facility_id: str = Field(min_length=1)
    generation: int | None = Field(default=None, ge=0)
    seq: int = Field(ge=0)
    cameras: list[RelayRuntimeStatusCamera] = Field()
    clip_recorder: RelayClipRecorderStatus
    clip_export: RelayClipExportStatus | None = Field(default=None)
    gpu: RelayGpuStatus | None = Field(default=None)
    worker: RelayWorkerStatus | None = Field(default=None)
    delivery_queue: RelayDeliveryQueueStatus | None = Field(default=None)


class RelayRuntimeStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    generation: int


class BackendIngestClient(Protocol):
    def send_heartbeat(self) -> bool: ...


@router.get("/config")
def worker_config(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> dict[str, object]:
    authorize_relay(request, relay_token)
    return worker_config_snapshot(request, require_available=True)


@router.post("/restart", status_code=status.HTTP_202_ACCEPTED)
def bump_restart(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> dict[str, int]:
    authorize_relay(request, relay_token)
    request.app.state.restart_epoch = int(getattr(request.app.state, "restart_epoch", 0)) + 1
    return {RESTART_EPOCH_KEY: request.app.state.restart_epoch}


@router.post(
    "/alerts",
    status_code=status.HTTP_202_ACCEPTED,
    description=(
        "Commit the incident and its delivery obligation, then answer.\n"
        "\n"
        "Every response is built after the admission COMMIT, so a failure before it\n"
        "is never acknowledged. A worker retry after a lost response lands on the\n"
        "same committed row instead of creating a second incident."
    ),
)
def relay_alert(
    payload: RelayAlertRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_alert)],
) -> dict[str, str]:
    binding = camera_binding(request, payload.camera_id, payload.facility_id)
    bound_camera_id = binding.get("backend_camera_id")
    backend_camera_id = (
        bound_camera_id if isinstance(bound_camera_id, str) and bound_camera_id.strip() else None
    )
    if backend_camera_id is None:
        _LOGGER.warning(
            "relay alert: skipping backend ingest, camera %s has no Hub mapping yet",
            payload.camera_id,
            extra={"local_camera_id": payload.camera_id},
        )
    client = getattr(request.app.state, "backend_ingest_client", None)
    delivery = getattr(request.app.state, "event_outbox_delivery", None)
    forward = backend_camera_id is not None and client is not None
    if forward and not isinstance(delivery, OutboxDelivery):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend outbox delivery is not configured",
        )
    edge_event_id = _alert_event_id(payload)
    accepted = _accept_alert(
        request, payload, edge_event_id, backend_camera_id=backend_camera_id, forward=forward
    )
    if accepted.delivery_state == "LOCAL_ONLY":
        return _local_receipt(payload, edge_event_id)
    if not isinstance(delivery, OutboxDelivery):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend outbox delivery is not configured",
        )
    claim = None
    if client is not None and accepted.delivery_state in {"PENDING", "IN_FLIGHT"}:
        try:
            claim = delivery.claim_event(edge_event_id)
        except AuthorityFenced as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="edge authority is fenced",
            ) from error
    if claim is None:
        return _replayed_alert_response(payload, edge_event_id, delivery.status(edge_event_id))
    result = dispatch(
        client,
        claim,
        delivery,
        on_accepted=lambda accepted_at: _record_alert_latency(request, payload, accepted_at),
    )
    if isinstance(result, DeliveryFailure):
        _backend_ingest_alert_failures.record_failure(result, path="alerts")
        raise _delivery_failure_error(result)
    _backend_ingest_alert_failures.record_success(path="alerts")
    if not result.event_id:
        return _local_receipt(payload, edge_event_id)
    return _central_receipt(payload, edge_event_id, result.event_id)


_IDLESS_ALERT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "urn:seeon-edge:relay-alert")


def _alert_event_id(payload: RelayAlertRequest) -> str:
    if payload.edge_event_id is not None:
        return payload.edge_event_id
    fields = payload.model_dump(
        exclude={"edge_event_id", "attempt_ordinal", "snapshot_jpeg_base64", "evidence"},
        exclude_none=True,
        mode="json",
    )
    evidence = _evidence_values(payload.evidence)
    if evidence is not None:
        fields["evidence"] = _EVIDENCE_JSON.dump_python(evidence, mode="json")
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return str(uuid.uuid5(_IDLESS_ALERT_NAMESPACE, canonical))


def _accept_alert(
    request: Request,
    payload: RelayAlertRequest,
    edge_event_id: str,
    *,
    backend_camera_id: str | None,
    forward: bool,
) -> AcceptedEvent:
    runtime = audit_runtime(request)
    outbox = EventOutbox(
        runtime.database, runtime.authority, RELAY_OUTBOX_BUDGET, audit_runtime=runtime
    )
    runtime.require_mutation_admission(outbox)
    try:
        snapshot_bytes = _decode_snapshot(payload.snapshot_jpeg_base64)
        return outbox.accept(
            _relay_event(payload, edge_event_id),
            backend_camera_id=backend_camera_id,
            forward=forward,
            snapshot=_relay_snapshot(payload),
            snapshot_bytes=snapshot_bytes,
        )
    except (EventIdentityConflict, RelayEvidenceProjectionConflict) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except OutboxCapacityExceeded as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend outbox capacity exceeded",
            headers={"Retry-After": "5"},
        ) from error
    except AuthorityFenced as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="edge authority is fenced",
        ) from error
    except (ValueError, RelayEvidenceProjectionError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except _REJECTED_FACTS as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="relay alert violates a stored fact constraint",
        ) from error


_REJECTED_FACTS = (
    CheckViolation,
    NotNullViolation,
    DataError,
)


def _relay_event(payload: RelayAlertRequest, edge_event_id: str) -> RelayEvent:
    return RelayEvent(
        edge_event_id=edge_event_id,
        event_type=str(payload.event_type),
        probability=payload.probability,
        detected_at=payload.detected_at,
        camera_id=payload.camera_id,
        facility_id=payload.facility_id,
        resident_id=payload.resident_id,
        evidence=_evidence_values(payload.evidence),
        audit=None if payload.audit is None else payload.audit.model_dump(exclude_none=True),
    )


def _relay_snapshot(payload: RelayAlertRequest) -> RelaySnapshot | None:
    if payload.snapshot is None or payload.snapshot_jpeg_base64 is None:
        return None
    return RelaySnapshot(
        snapshot_id=payload.snapshot.snapshot_id,
        path=payload.snapshot.path,
        sha256=payload.snapshot.sha256,
        size_bytes=payload.snapshot.size_bytes,
        mime_type=payload.snapshot.mime_type,
        captured_at=payload.snapshot.captured_at,
    )


def _local_receipt(payload: RelayAlertRequest, edge_event_id: str) -> dict[str, str]:
    if payload.edge_event_id is None:
        return {"status": "accepted"}
    return {"status": "accepted_local", "edge_event_id": edge_event_id}


def _central_receipt(
    payload: RelayAlertRequest, edge_event_id: str, event_id: str
) -> dict[str, str]:
    if payload.edge_event_id is None:
        return {"status": "accepted"}
    return {"status": "accepted", "edge_event_id": edge_event_id, "event_id": event_id}


def _delivery_failure_error(result: DeliveryFailure) -> HTTPException:
    if result.disposition is DeliveryDisposition.RETRY:
        code = status.HTTP_503_SERVICE_UNAVAILABLE
        detail = f"backend ingest retryable failure: {result.code}"
    elif result.disposition is DeliveryDisposition.COMPATIBILITY:
        code = status.HTTP_404_NOT_FOUND
        detail = "backend ingest rejected alert"
    else:
        code = result.status_code or status.HTTP_502_BAD_GATEWAY
        detail = "backend ingest rejected alert"
    headers = None
    if result.retry_after_seconds is not None:
        headers = {"Retry-After": str(max(0, int(result.retry_after_seconds)))}
    return HTTPException(status_code=code, detail=detail, headers=headers)


def _replayed_alert_response(
    payload: RelayAlertRequest, edge_event_id: str, stored: DeliveryStatus | None
) -> dict[str, str]:
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend outbox delivery state unavailable",
        )
    if stored.state == "SENT" and stored.backend_event_id:
        return _central_receipt(payload, edge_event_id, stored.backend_event_id)
    if stored.state in {"LOCAL_ONLY", "EXHAUSTED"} or (
        stored.state == "REJECTED" and stored.reason == "ACCEPTED_LOCAL"
    ):
        return _local_receipt(payload, edge_event_id)
    if stored.state == "REJECTED":
        raise _delivery_failure_error(
            DeliveryFailure(
                DeliveryDisposition.PERMANENT,
                stored.reason or "REJECTED",
                status_code=stored.http_status,
            )
        )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="backend ingest delivery pending",
        headers={"Retry-After": "5"},
    )


@router.post(
    "/snapshot-attachments",
    status_code=status.HTTP_202_ACCEPTED,
    description="Record one immutable media reference without accepting media bytes.",
)
def relay_snapshot_attachment(
    payload: RelaySnapshotAttachmentRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_snapshot_attachment)],
) -> dict[str, str]:
    projection = _snapshot_projection(request)
    audit = mutation_audit(
        request,
        lambda: _relay_audit_event(AuditAction.RELAY_SNAPSHOT_ATTACHMENT, payload.snapshot_id),
    )
    with _snapshot_projection_errors():
        audit.apply(
            projection,
            lambda append: projection.attach_snapshot(
                edge_event_id=payload.edge_event_id,
                snapshot_id=payload.snapshot_id,
                sha256=payload.sha256,
                media_reference=payload.media_reference,
                size_bytes=payload.size_bytes,
                mime_type=payload.mime_type,
                after_write=append,
            ),
        )
    return {"status": "accepted"}


@router.post(
    "/snapshot-dispositions",
    status_code=status.HTTP_202_ACCEPTED,
    description="Durably record an unavailable or failed snapshot without touching its event.",
)
def relay_snapshot_disposition(
    payload: RelaySnapshotDispositionRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_snapshot_disposition)],
) -> dict[str, str]:
    projection = _snapshot_projection(request)
    audit = mutation_audit(
        request,
        lambda: _relay_audit_event(AuditAction.RELAY_SNAPSHOT_DISPOSITION, payload.snapshot_id),
    )
    with _snapshot_projection_errors():
        audit.apply(
            projection,
            lambda append: projection.record_snapshot_disposition(
                edge_event_id=payload.edge_event_id,
                snapshot_id=payload.snapshot_id,
                disposition=payload.disposition,
                reason=payload.reason,
                after_write=append,
            ),
        )
    return {"status": "accepted"}


def _snapshot_projection(request: Request) -> PostgresRelayEvidenceProjection:
    projection = getattr(request.app.state, "relay_snapshot_projection", None)
    if not isinstance(projection, PostgresRelayEvidenceProjection):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="snapshot projection is not configured",
        )
    return projection


@contextmanager
def _snapshot_projection_errors() -> Iterator[None]:
    try:
        yield
    except (RelayEvidenceProjectionMissingEvent, RelayEvidenceProjectionConflict) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except RelayEvidenceProjectionError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except AuthorityFenced as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="edge authority is fenced",
        ) from error
    except _REJECTED_FACTS as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="snapshot record violates a stored fact constraint",
        ) from error


def _relay_audit_event(action: AuditAction, target_id: str) -> AuditEvent:
    return AuditEvent(
        occurred_at=audit_now(),
        actor_id="worker-relay",
        action=action,
        target_id=target_id,
        detail=empty_detail(action),
        actor_type=AuditActorType.SERVICE,
        auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
    )


@router.post("/heartbeat", status_code=status.HTTP_202_ACCEPTED)
def relay_heartbeat(
    payload: RelayHeartbeatRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_heartbeat)],
) -> dict[str, str]:
    get_heartbeat_store(request.app).record(
        payload.camera_id,
        payload.facility_id,
        config_version=payload.config_version,
    )
    acknowledge_applied_detection_policies(
        request,
        facility_id=payload.facility_id,
        config_version=payload.config_version,
    )
    _clear_never_connected_on_first_heartbeat(request, payload.camera_id)
    binding = camera_binding(request, payload.camera_id, payload.facility_id)
    bound_camera_id = binding.get("backend_camera_id")
    if not isinstance(bound_camera_id, str) or not bound_camera_id.strip():
        _LOGGER.warning(
            "relay heartbeat: skipping backend ingest, camera %s has no Hub mapping yet",
            payload.camera_id,
            extra={"local_camera_id": payload.camera_id},
        )
        return {"status": "accepted"}
    client = _optional_backend_ingest_client(request, camera_id=bound_camera_id)
    if client is None:
        return {"status": "accepted"}
    if not client.send_heartbeat():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="backend ingest rejected heartbeat",
        )
    return {"status": "accepted"}


@router.post("/runtime-status", response_model=RelayRuntimeStatusResponse)
def relay_runtime_status(
    payload: RelayRuntimeStatusRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_runtime_status)],
) -> RelayRuntimeStatusResponse:
    _runtime_status_facility_binding(request, payload.facility_id)
    _log_unresolved_runtime_status_cameras(request, payload)
    data = payload.model_dump()
    if data["worker"] is not None and data["worker"]["profile_boot_error"] is None:
        data["worker"].pop("profile_boot_error")
    result = get_runtime_status_store(request.app).record(data)
    if not result.accepted:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=result.reason)
    return RelayRuntimeStatusResponse(accepted=True, generation=result.generation)


def _decode_snapshot(snapshot_jpeg_base64: str | None) -> bytes | None:
    if snapshot_jpeg_base64 is None:
        return None
    try:
        snapshot_bytes = base64.b64decode(snapshot_jpeg_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("snapshot_jpeg_base64 must be valid Base64") from exc
    if len(snapshot_bytes) > MAX_INLINE_SNAPSHOT_BYTES:
        raise ValueError(
            "snapshot_jpeg_base64 exceeds maximum decoded size of "
            f"{MAX_INLINE_SNAPSHOT_BYTES} bytes"
        )
    return snapshot_bytes


def _record_alert_latency(request: Request, payload: RelayAlertRequest, received_at: float) -> None:
    if payload.attempt_ordinal != 1:
        return
    get_runtime_status_store(request.app).record_latency(
        payload.facility_id, payload.detected_at, received_at=received_at
    )


def _runtime_status_facility_binding(request: Request, facility_id: str) -> None:
    del request, facility_id


def _log_unresolved_runtime_status_cameras(
    request: Request, payload: RelayRuntimeStatusRequest
) -> None:
    for camera in payload.cameras:
        try:
            camera_binding(request, camera.camera_id, payload.facility_id)
        except HTTPException as exc:
            logger.warning(
                "runtime-status camera unresolved (recorded anyway): camera_id=%s detail=%s",
                camera.camera_id,
                exc.detail,
            )


def _clear_never_connected_on_first_heartbeat(request: Request, camera_id: str) -> None:
    store = getattr(request.app.state, "camera_registry", None)
    if not isinstance(store, CameraRegistryStore):
        return
    record = _find_registry_record(store, camera_id)
    if record is None or record.get("never_connected") is not True:
        return
    local_id = record.get("id")
    if isinstance(local_id, str):
        store.update(local_id, CameraUpdate(never_connected=False))


def _find_registry_record(store: CameraRegistryStore, camera_id: str) -> dict[str, object] | None:
    snapshot = store.snapshot()
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, list):
        return None
    for record in cameras:
        if not isinstance(record, dict):
            continue
        if camera_id in {record.get("id"), record.get("backend_camera_id")}:
            return record
    return None


def _optional_backend_ingest_client(
    request: Request, *, camera_id: str
) -> BackendIngestClient | None:
    client: BackendIngestClient | None = getattr(request.app.state, "backend_ingest_client", None)
    if client is None:
        return None
    for_camera = getattr(client, "for_camera", None)
    if for_camera is not None:
        scoped: BackendIngestClient = for_camera(camera_id)
        return scoped
    return client


def _backend_ingest_client(request: Request, *, camera_id: str) -> BackendIngestClient:
    client = _optional_backend_ingest_client(request, camera_id=camera_id)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend ingest client is not configured",
        )
    return client


__all__ = ["router"]
