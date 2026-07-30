"""API-owned clip audit log and best-effort backend backup."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.features.clips.store import (
    API_LABEL_STORE_ENV,
    DEFAULT_LABEL_STORE_DIR,
    is_valid_clip_id,
)

API_AUDIT_LOG_ENV = "API_AUDIT_LOG"
API_BACKEND_CLIP_EVENTS_URL_ENV = "API_BACKEND_CLIP_EVENTS_URL"
API_BACKEND_FACILITY_TOKEN_ENV = "API_BACKEND_FACILITY_TOKEN"
API_FACILITY_TOKEN_ENV = "API_FACILITY_TOKEN"
DEFAULT_BACKEND_TIMEOUT_SEC = 0.5


class AuditLogStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @classmethod
    def from_env(cls) -> AuditLogStore:
        configured_path = os.environ.get(API_AUDIT_LOG_ENV)
        if configured_path:
            return cls(configured_path)
        label_root = os.environ.get(API_LABEL_STORE_ENV, DEFAULT_LABEL_STORE_DIR)
        return cls(Path(label_root) / "audit.jsonl")

    def append(self, *, actor: str, action: str, clip_id: str) -> dict[str, object]:
        if not is_valid_clip_id(clip_id):
            raise ValueError("invalid clip_id")
        entry = {
            "ts": utc_now_iso(),
            "actor": actor,
            "action": action,
            "clip_id": clip_id,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        post_backend_backup("clip_audit", entry)
        return entry

    def list_entries(self) -> list[dict[str, object]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        entries: list[dict[str, object]] = []
        for line in lines:
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)
        return entries


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def post_backend_backup(event_type: str, payload: dict[str, Any]) -> bool:
    url = os.environ.get(API_BACKEND_CLIP_EVENTS_URL_ENV)
    if not url:
        return False
    envelope = {"type": event_type, "payload": payload}
    body = json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    token = os.environ.get(API_BACKEND_FACILITY_TOKEN_ENV) or os.environ.get(API_FACILITY_TOKEN_ENV)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_BACKEND_TIMEOUT_SEC) as response:
            status = int(getattr(response, "status", 200))
    except Exception as exc:  # noqa: BLE001 - backend backup is explicitly best-effort
        print(f"clip backend backup failed: {exc}", file=sys.stderr)
        return False
    else:
        return 200 <= status < 300


__all__ = [
    "API_AUDIT_LOG_ENV",
    "API_BACKEND_CLIP_EVENTS_URL_ENV",
    "AuditLogStore",
    "post_backend_backup",
    "utc_now_iso",
]
