from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class AuditActorType(StrEnum):
    USER = "user"
    SERVICE = "service"
    SYSTEM = "system"


class AuditAuthMechanism(StrEnum):
    DASHBOARD_SESSION = "dashboard_session"
    RELAY_TOKEN = "relay_token"
    INTERNAL = "internal"


class AuditAction(StrEnum):
    AUTH_LOGIN = "auth.login"
    AUTH_SESSION_READ = "auth.session.read"
    AUTH_LOGOUT = "auth.logout"
    CREDENTIAL_ROTATE = "credential.rotate"
    CAMERA_CREATE = "camera.create"
    CAMERA_UPDATE = "camera.update"
    CAMERA_DELETE = "camera.delete"
    CAMERA_PROBE = "camera.probe"
    LOCATION_CREATE = "location.create"
    LOCATION_UPDATE = "location.update"
    LOCATION_DELETE = "location.delete"
    BED_ZONE_UPDATE = "bed-zone.update"
    CONNECTION_UPDATE = "connection.update"
    CONNECTION_SYNC = "connection.sync"
    TOPOLOGY_CONFIRM = "topology.confirm"
    CLIP_STORAGE_UPDATE = "clip-storage.update"
    DETECTION_SETTINGS_UPDATE = "detection-settings.update"
    RUNTIME_SETTINGS_UPDATE = "runtime-settings.update"
    POLICY_APPLY = "policy.apply"
    POLICY_ROLLBACK = "policy.rollback"
    INCIDENT_LIST = "incident.list"
    INCIDENT_DETAIL = "incident.detail"
    INCIDENT_REVIEW = "incident.review"
    CLIP_LIST = "clip.list"
    CLIP_DETAIL = "clip.detail"
    CLIP_PLAY = "clip.play"
    CLIP_THUMBNAIL = "clip.thumbnail"
    CLIP_ARTIFACT = "clip.artifact"
    EVIDENCE_RECEIPT = "evidence.receipt"
    AUDIT_LIST = "audit.list"
    AUDIT_DETAIL = "audit.detail"
    RELAY_ALERT = "relay.alert"
    RELAY_SNAPSHOT_ATTACHMENT = "relay.snapshot-attachment"
    RELAY_SNAPSHOT_DISPOSITION = "relay.snapshot-disposition"
    AUDIT_SESSION_START = "audit.session-start"
    AUDIT_SESSION_CLOSE = "audit.session-close"
    RECOVERY_FENCE = "audit.recovery-fence"


@dataclass(frozen=True, slots=True)
class AuditDetail:
    action: AuditAction
    version: int
    json: str


@dataclass(frozen=True, slots=True)
class AuditEvent:
    occurred_at: str
    actor_id: str
    action: AuditAction
    target_id: str
    detail: AuditDetail
    actor_type: AuditActorType = AuditActorType.USER
    auth_mechanism: AuditAuthMechanism = AuditAuthMechanism.DASHBOARD_SESSION


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "AuditAction",
    "AuditActorType",
    "AuditAuthMechanism",
    "AuditDetail",
    "AuditEvent",
    "utc_now",
]
