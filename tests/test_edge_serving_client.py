"""Characterization: the in-process serving seam is a pure pass-through.

The edge inference plane obtains runners through a ``ServingClient`` so the GPU
serving can later be extracted into its own service without touching edge
orchestration/perception/domain code. ``InProcessServingClient`` is the identity
implementation; this test pins that it delegates to the local registry
identically (same task, same kwargs, same returned object).
"""

from __future__ import annotations

from edge.runners.registry import DEFAULT_REGISTRY
from edge.serving_client import InProcessServingClient, ServingClient


def test_in_process_serving_client_passes_through_to_registry() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    sentinel = object()

    class StubRegistry:
        def create(self, task: str, **kwargs: object) -> object:
            calls.append((task, kwargs))
            return sentinel

    client = InProcessServingClient(StubRegistry())

    # Provision exactly as edge_worker does (pose/bed with device, fall without).
    assert client.create("pose", device="cpu") is sentinel
    assert client.create("bed", device="cpu") is sentinel
    assert client.create("fall") is sentinel

    # Equivalence: identical delegation to the underlying registry.
    assert calls == [
        ("pose", {"device": "cpu"}),
        ("bed", {"device": "cpu"}),
        ("fall", {}),
    ]


def test_in_process_serving_client_defaults_to_default_registry() -> None:
    assert InProcessServingClient()._registry is DEFAULT_REGISTRY


def test_in_process_serving_client_satisfies_serving_client_interface() -> None:
    client: ServingClient = InProcessServingClient()
    assert callable(client.create)
