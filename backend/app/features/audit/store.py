from __future__ import annotations

from dataclasses import dataclass

from backend.app.shared.audit_values import AuditAction, AuditDetail, AuditEvent


@dataclass(frozen=True, slots=True)
class AuditRecord:
    audit_id: int
    occurred_at: str
    recorded_at: str
    actor_id: str
    action: AuditAction
    target_type: str
    target_id: str
    detail: AuditDetail
    previous_hash: str
    record_hash: str


def _target_type(action: AuditAction) -> str:
    return action.value.partition(".")[0]


def _payload(event: AuditEvent, recorded_at: str, previous_hash: str) -> dict[str, str | None]:
    return {
        "action": event.action.value,
        "actor_id": event.actor_id,
        "actor_type": event.actor_type.value,
        "auth_mechanism": event.auth_mechanism.value,
        "clock_quality": "trusted",
        "detail_json": event.detail.json,
        "hold_reference": None,
        "interaction_id": None,
        "occurred_at": event.occurred_at,
        "outcome": "success",
        "previous_hash": previous_hash,
        "reason": None,
        "recorded_at": recorded_at,
        "request_id": None,
        "retention_class": "standard",
        "target_id": event.target_id,
        "target_type": _target_type(event.action),
    }


__all__ = ["AuditRecord"]
