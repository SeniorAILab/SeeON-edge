from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI

from backend.app.edge_db.root import PostgresRoot
from backend.app.features.cameras.bed_zone_store import BedZoneStore
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.artifacts import CentralClipArtifactQuery
from backend.app.features.clips.catalog_indexer import ClipCatalogIndexer, PostgresClipCatalog
from backend.app.features.clips.storage_location_store import ClipStorageLocationStore
from backend.app.features.clips.store import ClipStore
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.detection_settings.policy_store import DetectionPolicyStore
from backend.app.features.detection_settings.store import DetectionSettingsStore
from backend.app.features.evidence.outbox_delivery import OutboxDelivery
from backend.app.features.evidence.outbox_dispatch import RELAY_DELIVERY_BUDGET
from backend.app.features.evidence.postgres_receipts import PostgresArtifactReceiptStore
from backend.app.features.evidence.postgres_relay_projection import (
    PostgresRelayEvidenceProjection,
)
from backend.app.features.evidence.record_store import (
    CentralEvidenceQuery,
    CentralEvidenceReviewStore,
)
from backend.app.features.runtime_settings.store import RuntimeSettingsStore
from backend.app.shared.postgres_dashboard_credentials import PostgresDashboardCredentialsStore


def _clip_root(app: FastAPI) -> Path:
    clip_store = getattr(app.state, "clip_store", None)
    return clip_store.root if isinstance(clip_store, ClipStore) else ClipStore.from_env().root


def install_postgres_stores(app: FastAPI, root: PostgresRoot) -> tuple[str, ...]:
    database, authority = root.database, root.authority
    installed: list[str] = []
    stores: dict[str, Callable[[], object]] = {
        "camera_registry": lambda: CameraRegistryStore(database, authority),
        "bed_zone_store": lambda: BedZoneStore(database, authority),
        "clip_storage_location_store": lambda: ClipStorageLocationStore(database, authority),
        "clip_catalog": lambda: PostgresClipCatalog(database),
        "clip_catalog_indexer": lambda: ClipCatalogIndexer(database, authority),
        "central_clip_artifact_query": lambda: CentralClipArtifactQuery(database),
        "connection_settings_store": lambda: ConnectionSettingsStore(database, authority),
        "detection_policy_store": lambda: DetectionPolicyStore(database, authority),
        "detection_settings_store": lambda: DetectionSettingsStore(database, authority),
        "runtime_settings_store": lambda: RuntimeSettingsStore(database, authority),
        "dashboard_credentials_store": lambda: PostgresDashboardCredentialsStore(
            database, authority
        ),
        "artifact_receipt_store": lambda: PostgresArtifactReceiptStore(
            database, authority, _clip_root(app)
        ),
        "central_evidence_review_store": lambda: CentralEvidenceReviewStore(database, authority),
        "central_evidence_query": lambda: CentralEvidenceQuery(database),
        "event_outbox_delivery": lambda: OutboxDelivery(database, authority, RELAY_DELIVERY_BUDGET),
        "relay_snapshot_projection": lambda: PostgresRelayEvidenceProjection(database, authority),
    }
    for name, build in stores.items():
        if getattr(app.state, name, None) is None:
            setattr(app.state, name, build())
            installed.append(name)
    return tuple(installed)


__all__ = ["install_postgres_stores"]
