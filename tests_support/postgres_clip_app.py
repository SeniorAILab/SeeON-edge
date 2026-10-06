from __future__ import annotations

from fastapi import FastAPI

from backend.app.features.clips.catalog_indexer import ClipCatalogIndexer, ReconcileOutcome
from backend.app.features.clips.store import ClipStore

MAX_INDEX_PASSES = 64


def app_clip_store(app: FastAPI) -> ClipStore:
    store = getattr(app.state, "clip_store", None)
    if not isinstance(store, ClipStore):
        store = ClipStore.from_env()
        app.state.clip_store = store
    return store


def index_clips(app: FastAPI) -> tuple[ReconcileOutcome, ...]:
    indexer = app.state.clip_catalog_indexer
    assert isinstance(indexer, ClipCatalogIndexer)
    store = app_clip_store(app)
    outcomes: list[ReconcileOutcome] = []
    for _ in range(MAX_INDEX_PASSES):
        outcome = indexer.reconcile(store)
        outcomes.append(outcome)
        if outcome.remaining == 0:
            return tuple(outcomes)
    raise AssertionError(f"clip catalogue did not converge in {MAX_INDEX_PASSES} passes")
