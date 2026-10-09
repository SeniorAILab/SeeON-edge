from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from starlette.concurrency import run_in_threadpool

from backend.app.edge_db import DatabaseDriverError
from backend.app.edge_db.postgres import PostgresError
from backend.app.features.diagnostics.schemas import (
    ExecutionQueryParams,
    ExecutionQueryResponse,
    ExecutionRecordReceiptResponse,
    query_response_from_result,
)
from backend.app.features.diagnostics.store import ExecutionRecordStore
from backend.app.features.diagnostics.wire import ingest_batch_from_wire, wire_receipt_from_store
from backend.app.shared.http.dashboard_auth import authorize_dashboard
from backend.app.shared.http.relay_http import (
    RELAY_TOKEN_HEADER,
    authorize_relay_body,
    bounded_body_route,
)
from shared.events.execution_records import (
    MAX_EXECUTION_RECORD_BODY_BYTES,
    ExecutionRecordContractError,
    WireBatch,
)

DISABLED_DETAIL = "execution records disabled"
UNAVAILABLE_DETAIL = "diagnostics store unavailable: check PostgreSQL and run migration provision"

router = APIRouter(
    tags=["diagnostics"],
    route_class=bounded_body_route({"/execution-records": MAX_EXECUTION_RECORD_BODY_BYTES}),
)


def require_relay_execution_records(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    authorize_relay_body(
        request,
        max_bytes=MAX_EXECUTION_RECORD_BODY_BYTES,
        relay_token=relay_token,
        authorization=authorization,
    )


def execution_record_store(request: Request) -> ExecutionRecordStore:
    store = getattr(request.app.state, "execution_record_store", None)
    if not isinstance(store, ExecutionRecordStore):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=DISABLED_DETAIL,
        )
    return store


def backend_build_revision(request: Request) -> str:
    value = getattr(request.app.state, "backend_build_revision", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML_API_BUILD_REVISION is not configured",
        )
    return value


@router.post("/relay/execution-records", response_model=ExecutionRecordReceiptResponse)
async def ingest_execution_records(
    request: Request,
    _: Annotated[None, Depends(require_relay_execution_records)],
) -> dict[str, object]:
    store = execution_record_store(request)
    try:
        batch = WireBatch.from_json(await request.json())
    except ExecutionRecordContractError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    ingest = ingest_batch_from_wire(batch, backend_build_revision=backend_build_revision(request))
    try:
        receipt = await run_in_threadpool(store.ingest_batch, ingest)
    except (PostgresError, DatabaseDriverError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=UNAVAILABLE_DETAIL,
        ) from error
    return wire_receipt_from_store(receipt).to_json()


@router.get("/diagnostics/executions", response_model=ExecutionQueryResponse)
def query_execution_records(
    request: Request,
    _: Annotated[str, Depends(authorize_dashboard)],
    params: Annotated[ExecutionQueryParams, Query()],
) -> ExecutionQueryResponse:
    store = execution_record_store(request)
    try:
        result = store.query(
            params.camera_id,
            params.from_ns,
            params.to_ns,
            params.limit,
            params.cursor,
        )
    except (PostgresError, DatabaseDriverError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=UNAVAILABLE_DETAIL,
        ) from error
    return query_response_from_result(result)


__all__ = ["backend_build_revision", "execution_record_store", "router"]
