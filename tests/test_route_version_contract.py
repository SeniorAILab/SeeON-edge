from __future__ import annotations

from typing import Final

from fastapi.routing import APIRoute

from backend.app.core.config import get_settings
from backend.app.main import create_app, no_lifespan

UNVERSIONED_ALLOWLIST: Final = {
    "/health/live",
    "/health/ready",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}


def test_routes_are_versioned_except_explicit_unversioned_allowlist() -> None:
    app = create_app(lifespan=no_lifespan)
    prefix = get_settings().api_v1_prefix
    paths = {route.path for route in app.routes}
    api_route_paths = {route.path for route in app.routes if isinstance(route, APIRoute)}

    escaped = sorted(
        path
        for path in api_route_paths
        if path not in UNVERSIONED_ALLOWLIST and not path.startswith(prefix)
    )

    assert not escaped
    assert UNVERSIONED_ALLOWLIST.issubset(paths)
    assert f"{prefix}/relay/alerts" in api_route_paths
    assert f"{prefix}/status" in api_route_paths
    assert "/status" not in paths


def test_operator_explanation_route_is_versioned_get_only() -> None:
    # Given the real FastAPI app and the committed api_v1 prefix
    app = create_app(lifespan=no_lifespan)
    prefix = get_settings().api_v1_prefix
    path = f"{prefix}/events/{{edge_event_id}}/explanation"

    # When versioned API routes and OpenAPI methods are collected
    api_routes = [
        route for route in app.routes if isinstance(route, APIRoute) and route.path == path
    ]
    spec_paths = app.openapi()["paths"]

    # Then the operator explanation surface is exactly one versioned GET
    assert path.startswith(prefix)
    assert path in spec_paths
    assert len(api_routes) == 1
    assert api_routes[0].methods == {"GET"}
    assert set(spec_paths[path]) == {"get"}
    assert spec_paths[path]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/EventExplanationResponse"}
    assert [
        candidate for candidate in spec_paths if candidate.startswith(f"{prefix}/events/")
    ] == [path]
