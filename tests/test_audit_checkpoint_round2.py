from __future__ import annotations

import pytest

from backend.app.features.audit.catalog import AuditDetailError, parse_detail_json
from backend.app.shared.audit_values import AuditAction


def test_detail_version_is_exact_uncoerced_json_integer() -> None:
    assert parse_detail_json(AuditAction.CLIP_LIST, '{"version":1}').json == '{"version":1}'
    for encoded in (
        '{"version":true}',
        '{"version":false}',
        '{"version":1.0}',
        '{"version":1.00}',
        '{"version":"1"}',
        '{"version":null}',
        '{"version":2}',
    ):
        with pytest.raises(AuditDetailError, match="version"):
            parse_detail_json(AuditAction.CLIP_LIST, encoded)


def test_nested_detail_payload_does_not_coerce_version() -> None:
    encoded = '{"version":true,"safe":{"version":1}}'
    with pytest.raises(AuditDetailError):
        parse_detail_json(AuditAction.CLIP_LIST, encoded)
