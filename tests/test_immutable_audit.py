from __future__ import annotations

import json

import pytest
from fastapi import FastAPI

from backend.app.audit_lifecycle import close_audit_session, configure_audit_readiness
from backend.app.features.audit.catalog import (
    AuditDetailError,
    JsonValue,
    parse_detail_json,
    recovery_detail,
)
from backend.app.postgres_root import PostgresRoot
from backend.app.shared.audit_values import AuditAction
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


def test_restart_verification_marks_corrupted_schema18_audit_unready(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    sandbox.admin.execute("DROP TRIGGER audit_events_immutable_delete ON audit_events")
    app = FastAPI()
    app.state.postgres_root = PostgresRoot(sandbox.database, sandbox.authority)

    healthy = configure_audit_readiness(app, clock=lambda: 0.0)

    try:
        assert healthy is False
        status = app.state.audit_runtime.snapshot()
        assert status.failure_code == "verification_failed"
        assert status.session_established is False
    finally:
        close_audit_session(app)


@pytest.mark.parametrize(
    "detail",
    [
        {"nested": {"PassWord": "redacted"}},
        {"nested": {"safe": "session-token"}},
        {"MediaBytes": "00ff"},
        {"rawPose": [1, 2]},
    ],
)
def test_detail_parser_rejects_recursive_privacy_aliases(
    detail: dict[str, JsonValue],
) -> None:
    with pytest.raises(AuditDetailError):
        parse_detail_json(AuditAction.CLIP_LIST, json.dumps({"version": 1, "nested": detail}))


def test_detail_parser_rejects_more_than_sixteen_kibibytes() -> None:
    with pytest.raises(AuditDetailError):
        recovery_detail("x" * 17000, "2026-08-24T00:01:00.000Z")


def test_detail_parser_canonicalizes_safe_registered_detail() -> None:
    detail = recovery_detail("SQLITE_FULL", "2026-08-24T00:01:00.000Z")

    assert detail.json == (
        '{"ended_at":"2026-08-24T00:01:00.000Z","failure_code":"SQLITE_FULL","version":1}'
    )
