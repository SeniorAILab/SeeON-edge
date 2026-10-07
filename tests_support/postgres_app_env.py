from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.postgres_root import (
    API_POSTGRES_AUTHORITY_FILE_ENV,
    API_POSTGRES_DSN_FILE_ENV,
    API_POSTGRES_SCHEMA_ENV,
    PostgresRoot,
)
from tests_support.postgres_sandbox import ProductSandbox


def write_postgres_root_env(
    sandbox: ProductSandbox, directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dsn_file = directory / "postgres.dsn"
    dsn_file.write_text(sandbox.dsn, encoding="utf-8")
    dsn_file.chmod(0o600)
    authority_file = directory / "postgres-authority.json"
    authority_file.write_text(
        json.dumps(
            {
                "generation": sandbox.authority.generation,
                "writer_token": str(sandbox.authority.writer_token),
            }
        ),
        encoding="utf-8",
    )
    authority_file.chmod(0o600)
    monkeypatch.setenv(API_POSTGRES_DSN_FILE_ENV, str(dsn_file))
    monkeypatch.setenv(API_POSTGRES_AUTHORITY_FILE_ENV, str(authority_file))
    monkeypatch.setenv(API_POSTGRES_SCHEMA_ENV, sandbox.schema)


def inject_sandbox_root(
    app: FastAPI, sandbox: ProductSandbox, audit_runtime: PostgresAuditRuntime
) -> None:
    app.state.postgres_root = PostgresRoot(sandbox.database, sandbox.authority)
    app.state.audit_runtime = audit_runtime


@pytest.fixture
def postgres_app_env(
    postgres_product_sandbox: ProductSandbox,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ProductSandbox:
    write_postgres_root_env(postgres_product_sandbox, tmp_path / "postgres-secrets", monkeypatch)
    return postgres_product_sandbox
