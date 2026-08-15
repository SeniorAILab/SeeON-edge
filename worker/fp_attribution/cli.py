"""Standalone read-only production FP attribution CLI.

``python -m worker.fp_attribution`` opens an already-migrated edge SQLite file
existing-file-only, composes the committed cohort/evidence/classifier/metrics
seams, and prints one privacy-allowlisted JSON document to stdout. The only
behavior is dry-run/query-only; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

CLEAN_EXIT_CODE = 0
DB_UNAVAILABLE_EXIT_CODE = 2
DB_SCHEMA_INVALID_EXIT_CODE = 3
EXPORT_UNTRUSTED_EXIT_CODE = 4

REPORT_SCHEMA = "fp-attribution-report-v1"
REPORT_VERSION = 1
DB_UNAVAILABLE = "production-edge-db-unavailable"
DB_SCHEMA_INVALID = "production-edge-db-schema-invalid"
EXPORT_UNTRUSTED = "export-untrusted"

_CORRELATION_SCHEMA = "fp-correlation-v1"
_FAULT_TOKEN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ALERT_KEYS = frozenset({"alert_id", "edge_event_id"})
_DELIVERY_KINDS = frozenset({"BACKEND_OR_UI_DUPLICATE", "DELIVERY_RETRY"})
_KIND_KEYS = {
    "BACKEND_OR_UI_DUPLICATE": frozenset(
        {"edge_event_id", "kind", "schema", "user_visible_delivery_count"}
    ),
    "DELIVERY_RETRY": frozenset(
        {"edge_event_id", "kind", "schema", "user_visible_delivery_count"}
    ),
    "TRANSPORT_ONLY": frozenset(
        {"edge_event_id", "kind", "schema", "user_visible_delivery_count"}
    ),
    "CAMERA_LIGHTING_OR_DECODE": frozenset(
        {"edge_event_id", "kind", "schema", "typed_fault_code"}
    ),
}


class _ExportError(ValueError):
    """Allowlisted correlation/alert export failed closed."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m worker.fp_attribution",
        description=(
            "Read-only production false-positive attribution over an already-"
            "migrated edge SQLite file. Never writes, never trains, never uses "
            "GPU or network, never bootstraps the worker runtime."
        ),
    )
    parser.add_argument(
        "--edge-db",
        type=Path,
        required=True,
        help="Path to an already-migrated edge.sqlite3 (opened existing-file-only)",
    )
    parser.add_argument(
        "--correlation-export",
        type=Path,
        default=None,
        help="Optional typed fp-correlation-v1 JSON list (strict allowlist)",
    )
    parser.add_argument(
        "--alert-export",
        type=Path,
        default=None,
        help="Optional typed alert-id JSON list (strict allowlist)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run(args.edge_db, args.correlation_export, args.alert_export)
    except _ExportError:
        _emit(EXPORT_UNTRUSTED)
        return EXPORT_UNTRUSTED_EXIT_CODE


def _run(
    database_path: Path,
    correlation_path: Path | None,
    alert_path: Path | None,
) -> int:
    from worker.fp_attribution.attribution import classify_record
    from worker.fp_attribution.cohort import FalsePositiveCohortQuery
    from worker.fp_attribution.evidence import AttributionEvidenceQuery
    from worker.fp_attribution.metrics import (
        metric_event_from_record,
        metrics_machine_bytes,
        summarize_attribution_metrics,
    )

    proofs = None if correlation_path is None else _load_correlation(correlation_path)
    alerts = None if alert_path is None else _load_alerts(alert_path)
    try:
        cohort = FalsePositiveCohortQuery(database_path).load()
        evidence = AttributionEvidenceQuery(database_path).extract()
    except ValueError as exc:
        token = DB_SCHEMA_INVALID if _schema_failure(exc) else DB_UNAVAILABLE
        _emit(token)
        return (
            DB_SCHEMA_INVALID_EXIT_CODE
            if token == DB_SCHEMA_INVALID
            else DB_UNAVAILABLE_EXIT_CODE
        )
    except sqlite3.OperationalError:
        _emit(DB_SCHEMA_INVALID)
        return DB_SCHEMA_INVALID_EXIT_CODE
    except (OSError, sqlite3.Error):
        _emit(DB_UNAVAILABLE)
        return DB_UNAVAILABLE_EXIT_CODE

    metric_events = []
    records = []
    for record in evidence.records:
        export = None if proofs is None else proofs.get(record.edge_event_id)
        decision = classify_record(record, correlation_export=export)
        if decision.annotations.correlation_status == "rejected":
            raise _ExportError
        metric_events.append(metric_event_from_record(record, decision=decision))
        records.append(_record_payload(record, decision))
    summary = summarize_attribution_metrics(
        metric_events,
        exclusions=evidence.exclusions,
        alert_correlation_export=alerts,
    )
    if alerts is not None and summary.transport.unique_alert_id.status != "AVAILABLE":
        raise _ExportError
    payload = {
        "cohort": {
            "exclusion_census": summary.legacy_excluded_census,
            "members": [
                {
                    "current_review_version": member.current_review_version,
                    "decision_trace_id": member.decision_trace_id,
                    "edge_event_id": member.edge_event_id,
                    "incident_id": member.incident_id,
                }
                for member in cohort.members
            ],
        },
        "correlation": {
            "alert_ids": {
                "missing_reason": summary.transport.unique_alert_id.missing_reason,
                "status": summary.transport.unique_alert_id.status,
                "value": summary.transport.unique_alert_id.value,
            },
            "proof_export": "absent" if proofs is None else "present",
        },
        "metrics": json.loads(metrics_machine_bytes(summary)),
        "records": records,
        "schema": REPORT_SCHEMA,
        "source": {"kind": "edge-sqlite", "read_mode": "query_only"},
        "version": REPORT_VERSION,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    sys.stdout.write("\n")
    return CLEAN_EXIT_CODE


def _schema_failure(exc: ValueError) -> bool:
    message = str(exc)
    return "schema" in message or "v16" in message


def _emit(token: str) -> None:
    sys.stderr.write(token)
    sys.stderr.write("\n")


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise _ExportError
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _ExportError from exc


def _load_correlation(path: Path) -> dict[str, dict[str, object]]:
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise _ExportError
    indexed: dict[str, dict[str, object]] = {}
    for item in raw:
        parsed = _require_correlation(item)
        event_id = str(parsed["edge_event_id"])
        existing = indexed.get(event_id)
        if existing is not None and existing != parsed:
            raise _ExportError
        indexed[event_id] = parsed
    return indexed


def _require_correlation(item: object) -> dict[str, object]:
    if not isinstance(item, dict):
        raise _ExportError
    schema = item.get("schema")
    kind = item.get("kind")
    if schema != _CORRELATION_SCHEMA or not isinstance(kind, str) or kind not in _KIND_KEYS:
        raise _ExportError
    if set(item) != _KIND_KEYS[kind]:
        raise _ExportError
    event_id = item.get("edge_event_id")
    if not isinstance(event_id, str) or not event_id:
        raise _ExportError
    if kind == "CAMERA_LIGHTING_OR_DECODE":
        fault = item.get("typed_fault_code")
        if not isinstance(fault, str) or _FAULT_TOKEN.fullmatch(fault) is None:
            raise _ExportError
        return item
    count = item.get("user_visible_delivery_count")
    if type(count) is not int:
        raise _ExportError
    if kind in _DELIVERY_KINDS and count < 2:
        raise _ExportError
    if kind == "TRANSPORT_ONLY" and count != 1:
        raise _ExportError
    return item


def _load_alerts(path: Path) -> tuple[Mapping[str, str], ...]:
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise _ExportError
    rows: list[Mapping[str, str]] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != _ALERT_KEYS:
            raise _ExportError
        event_id = item.get("edge_event_id")
        alert_id = item.get("alert_id")
        if not isinstance(event_id, str) or not event_id:
            raise _ExportError
        if not isinstance(alert_id, str) or not alert_id:
            raise _ExportError
        rows.append({"alert_id": alert_id, "edge_event_id": event_id})
    return tuple(rows)


def _record_payload(record: object, decision: object) -> dict[str, object]:
    from worker.fp_attribution.attribution import AttributionDecision
    from worker.fp_attribution.evidence import AttributionEvidenceRecord

    if not isinstance(record, AttributionEvidenceRecord):
        raise TypeError("attribution record is invalid")
    if not isinstance(decision, AttributionDecision):
        raise TypeError("attribution decision is invalid")
    return {
        "associated_sibling_event_ids": list(record.associated_sibling_event_ids),
        "attempt_count": record.attempt_count,
        "backend_event_ids": list(record.backend_event_ids),
        "bed_changed": record.bed_changed,
        "bed_id": record.bed_id,
        "bed_missing_reason": record.bed_missing_reason,
        "boot_changed": record.boot_changed,
        "category": decision.category,
        "correlation_kind": decision.annotations.correlation_kind,
        "correlation_status": decision.annotations.correlation_status,
        "coverage_reason": record.coverage_reason,
        "coverage_status": record.coverage_status,
        "current_state": record.current_state,
        "decision_reason": record.decision_reason,
        "edge_event_id": record.edge_event_id,
        "epoch_changed": record.epoch_changed,
        "evidence_status": record.evidence_status,
        "expected_frames": record.expected_frames,
        "matched_predicate": decision.annotations.matched_predicate,
        "neighborhood_pruned": record.neighborhood_pruned,
        "prevented_eligible": record.prevented_eligible,
        "previous_state": record.previous_state,
        "retained_frames": record.retained_frames,
        "score": record.score,
        "score_missing_reason": record.score_missing_reason,
        "stream_epoch": record.stream_epoch,
        "threshold": record.threshold,
        "threshold_missing_reason": record.threshold_missing_reason,
        "track_changed": record.track_changed,
        "track_id": record.track_id,
        "track_missing_reason": record.track_missing_reason,
        "worker_boot_id": record.worker_boot_id,
    }


__all__ = [
    "CLEAN_EXIT_CODE",
    "DB_SCHEMA_INVALID_EXIT_CODE",
    "DB_UNAVAILABLE_EXIT_CODE",
    "EXPORT_UNTRUSTED_EXIT_CODE",
    "main",
]
