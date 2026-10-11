from __future__ import annotations

from fastapi import FastAPI

from backend.app.bootstrap.lifespan import install_feature_ports
from backend.app.bootstrap.stores import install_postgres_stores
from backend.app.edge_db.root import PostgresRoot
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.main import create_app, no_lifespan
from tests_support.postgres_sandbox import ProductSandbox


def postgres_api_app(sandbox: ProductSandbox, audit_runtime: PostgresAuditRuntime) -> FastAPI:
    app = create_app(lifespan=no_lifespan)
    install_postgres_stores(app, PostgresRoot(sandbox.database, sandbox.authority))
    install_feature_ports(app)
    app.state.audit_runtime = audit_runtime
    return app
