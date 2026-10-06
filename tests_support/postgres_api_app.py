from __future__ import annotations

from fastapi import FastAPI

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.main import create_app, no_lifespan
from backend.app.postgres_root import PostgresRoot, install_postgres_stores
from tests_support.postgres_sandbox import ProductSandbox


def postgres_api_app(sandbox: ProductSandbox, audit_runtime: PostgresAuditRuntime) -> FastAPI:
    app = create_app(lifespan=no_lifespan)
    install_postgres_stores(app, PostgresRoot(sandbox.database, sandbox.authority))
    app.state.audit_runtime = audit_runtime
    return app
