#!/usr/bin/env python3
"""Idempotently load existing local clip-store sidecars into ml-api's catalog."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from backend.app.features.cameras.store import (
    API_CAMERA_STORE_ENV,
    DEFAULT_CAMERA_STORE,
)
from backend.app.features.clips.audit_log import AuditLogStore
from backend.app.features.clips.catalog import (
    API_CATALOG_STORE_ENV,
    DEFAULT_CATALOG_STORE,
    CatalogStore,
    sanitized_camera_payload,
    strict_camera_snapshot,
)
from backend.app.features.clips.store import ClipStore, LabelStore

parser = argparse.ArgumentParser()
parser.add_argument(
    "--catalog",
    type=Path,
    default=Path(os.environ.get(API_CATALOG_STORE_ENV, DEFAULT_CATALOG_STORE)),
)
parser.add_argument("--clip-store", type=Path, required=True)
parser.add_argument("--audit", type=Path)
parser.add_argument("--labels", type=Path)
parser.add_argument(
    "--cameras",
    type=Path,
    default=Path(os.environ.get(API_CAMERA_STORE_ENV, DEFAULT_CAMERA_STORE)),
)
args = parser.parse_args()
store = CatalogStore.open(args.catalog)
try:
    camera_snapshot = strict_camera_snapshot(args.cameras)
    store.backfill(
        ClipStore(args.clip_store),
        audit_log=AuditLogStore(args.audit) if args.audit else None,
        label_store=LabelStore(args.labels) if args.labels else None,
    )
    if camera_snapshot is not None:
        for camera in camera_snapshot["cameras"]:
            if isinstance(camera.get("id"), str) and camera["id"]:
                store.record("cameras", camera["id"], sanitized_camera_payload(camera))
    camera_source = str(args.cameras) if camera_snapshot is not None else "absent"
    print(
        f"catalog backfill complete camera_source={camera_source} "
        f"cameras={len(camera_snapshot['cameras']) if camera_snapshot else 0}"
    )
finally:
    store.close()
