from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

from pydantic import JsonValue, TypeAdapter, ValidationError

from backend.app.shared.audit_values import AuditAction, AuditDetail

MAX_DETAIL_BYTES: Final = 16 * 1024
_DETAIL_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])


class AuditDetailKind(StrEnum):
    EMPTY = "empty"
    PROBE = "probe"
    SESSION = "session"
    RECOVERY = "recovery"


@dataclass(frozen=True, slots=True)
class AuditDetailDeclaration:
    action: AuditAction
    version: int
    kind: AuditDetailKind


class AuditDetailError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_EMPTY: Final = AuditDetailKind.EMPTY
_PROBE: Final = AuditDetailKind.PROBE
_SESSION: Final = AuditDetailKind.SESSION
_RECOVERY: Final = AuditDetailKind.RECOVERY
ACTION_DETAIL_CATALOG: Final = (
    *(
        AuditDetailDeclaration(action, 1, _EMPTY)
        for action in (
            AuditAction.AUTH_LOGIN,
            AuditAction.AUTH_SESSION_READ,
            AuditAction.AUTH_LOGOUT,
            AuditAction.CREDENTIAL_ROTATE,
            AuditAction.CAMERA_CREATE,
            AuditAction.CAMERA_UPDATE,
            AuditAction.CAMERA_DELETE,
            AuditAction.LOCATION_CREATE,
            AuditAction.LOCATION_UPDATE,
            AuditAction.LOCATION_DELETE,
            AuditAction.BED_ZONE_UPDATE,
            AuditAction.CONNECTION_UPDATE,
            AuditAction.CONNECTION_SYNC,
            AuditAction.TOPOLOGY_CONFIRM,
            AuditAction.CLIP_STORAGE_UPDATE,
            AuditAction.DETECTION_SETTINGS_UPDATE,
            AuditAction.RUNTIME_SETTINGS_UPDATE,
            AuditAction.POLICY_APPLY,
            AuditAction.POLICY_ROLLBACK,
            AuditAction.INCIDENT_LIST,
            AuditAction.INCIDENT_DETAIL,
            AuditAction.INCIDENT_REVIEW,
            AuditAction.CLIP_LIST,
            AuditAction.CLIP_DETAIL,
            AuditAction.CLIP_PLAY,
            AuditAction.CLIP_THUMBNAIL,
            AuditAction.CLIP_ARTIFACT,
            AuditAction.EVIDENCE_RECEIPT,
            AuditAction.AUDIT_LIST,
            AuditAction.AUDIT_DETAIL,
            AuditAction.RELAY_ALERT,
            AuditAction.RELAY_SNAPSHOT_ATTACHMENT,
            AuditAction.RELAY_SNAPSHOT_DISPOSITION,
        )
    ),
    AuditDetailDeclaration(AuditAction.CAMERA_PROBE, 1, _PROBE),
    AuditDetailDeclaration(AuditAction.AUDIT_SESSION_START, 1, _SESSION),
    AuditDetailDeclaration(AuditAction.AUDIT_SESSION_CLOSE, 1, _SESSION),
    AuditDetailDeclaration(AuditAction.RECOVERY_FENCE, 1, _RECOVERY),
)


def assert_catalog_complete(
    declarations: tuple[AuditDetailDeclaration, ...],
) -> None:
    actions = tuple(declaration.action for declaration in declarations)
    if len(actions) != len(set(actions)) or set(actions) != set(AuditAction):
        raise AuditDetailError("audit detail catalog is incomplete or duplicated")
    if any(declaration.version != 1 for declaration in declarations):
        raise AuditDetailError("audit detail catalog has an unsupported version")


assert_catalog_complete(ACTION_DETAIL_CATALOG)
_CATALOG: Final = {declaration.action: declaration for declaration in ACTION_DETAIL_CATALOG}
_FORBIDDEN_CLASSES: Final = frozenset(
    {
        "raw",
        "resident",
        "pose",
        "keypoint",
        "token",
        "password",
        "credential",
        "session",
        "cookie",
        "request",
        "response",
        "traceback",
        "path",
        "media",
        "bytes",
        "secret",
        "authorization",
        "rtsp",
        "image",
        "frame",
    }
)


def empty_detail(action: AuditAction) -> AuditDetail:
    return _encode(action, {"version": 1}, AuditDetailKind.EMPTY)


def camera_probe_detail(ok: bool, error_class: str | None) -> AuditDetail:
    if error_class not in {None, "timeout", "decode", "auth", "unsupported", "unavailable"}:
        raise AuditDetailError("camera probe error class is invalid")
    return _encode(
        AuditAction.CAMERA_PROBE,
        {"version": 1, "ok": ok, "error_class": error_class},
        AuditDetailKind.PROBE,
    )


def session_detail(action: AuditAction) -> AuditDetail:
    return _encode(action, {"version": 1}, AuditDetailKind.SESSION)


def recovery_detail(failure_code: str, ended_at: str) -> AuditDetail:
    return _encode(
        AuditAction.RECOVERY_FENCE,
        {"version": 1, "failure_code": failure_code, "ended_at": ended_at},
        AuditDetailKind.RECOVERY,
    )


def parse_detail_json(action: AuditAction, encoded: str) -> AuditDetail:
    try:
        raw = _DETAIL_ADAPTER.validate_json(encoded)
    except ValidationError as error:
        raise AuditDetailError("audit detail JSON is invalid") from error
    return _encode(action, raw, _CATALOG[action].kind)


def _encode(
    action: AuditAction, raw: dict[str, JsonValue], expected_kind: AuditDetailKind
) -> AuditDetail:
    declaration = _CATALOG[action]
    if declaration.kind is not expected_kind:
        raise AuditDetailError(f"detail variant is not registered for {action.value}")
    _inspect(raw)
    allowed = {
        AuditDetailKind.EMPTY: frozenset({"version"}),
        AuditDetailKind.PROBE: frozenset({"version", "ok", "error_class"}),
        AuditDetailKind.SESSION: frozenset({"version"}),
        AuditDetailKind.RECOVERY: frozenset({"version", "failure_code", "ended_at"}),
    }[declaration.kind]
    version = raw.get("version")
    if frozenset(raw) != allowed or type(version) is not int or version != declaration.version:
        raise AuditDetailError(f"detail fields/version are not registered for {action.value}")
    if declaration.kind is AuditDetailKind.PROBE:
        if type(raw.get("ok")) is not bool or raw.get("error_class") not in {
            None,
            "timeout",
            "decode",
            "auth",
            "unsupported",
            "unavailable",
        }:
            raise AuditDetailError("camera probe detail values are invalid")
    if declaration.kind is AuditDetailKind.RECOVERY:
        if not isinstance(raw.get("failure_code"), str) or not isinstance(raw.get("ended_at"), str):
            raise AuditDetailError("recovery detail values are invalid")
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode()) > MAX_DETAIL_BYTES:
        raise AuditDetailError("audit detail exceeds 16 KiB")
    return AuditDetail(action, declaration.version, encoded)


def _inspect(value: JsonValue) -> None:
    match value:
        case None | bool() | int() | float():
            return
        case str() as text:
            if _forbidden(text):
                raise AuditDetailError("audit detail contains a forbidden value class")
        case list() as items:
            for item in items:
                _inspect(item)
        case dict() as values:
            for key, item in values.items():
                if _forbidden(key):
                    raise AuditDetailError("audit detail contains a forbidden field class")
                _inspect(item)
        case unreachable:
            assert_never(unreachable)


def _forbidden(value: str) -> bool:
    normalized = "".join(character for character in value.casefold() if character.isalnum())
    return any(term in normalized for term in _FORBIDDEN_CLASSES)


__all__ = [
    "ACTION_DETAIL_CATALOG",
    "MAX_DETAIL_BYTES",
    "AuditDetailDeclaration",
    "AuditDetailError",
    "AuditDetailKind",
    "assert_catalog_complete",
    "camera_probe_detail",
    "empty_detail",
    "parse_detail_json",
    "recovery_detail",
    "session_detail",
]
