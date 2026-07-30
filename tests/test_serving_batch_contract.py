"""serving_client batch-input evolution contract (ADR-0002, acceptance 5).

The seam defines a typed batched-inference swap point (`BatchServingClient`) for
50-camera scale; the in-process client stays single-frame (`ServingClient`) and
does NOT implement the batch contract yet (batching backend deferred).
"""

from __future__ import annotations

from edge.serving_client import BatchServingClient, InProcessServingClient, ServingClient


def test_inprocess_satisfies_serving_client():
    assert isinstance(InProcessServingClient(), ServingClient)


def test_inprocess_does_not_yet_implement_batch_contract():
    # documents the deferral: batching backend not implemented in-process (ADR-0002)
    assert not isinstance(InProcessServingClient(), BatchServingClient)


def test_batch_serving_client_is_a_superset_of_serving_client():
    assert issubclass(BatchServingClient, ServingClient)
    assert hasattr(BatchServingClient, "infer_batch")


def test_batch_contract_shape_is_provisionable_by_a_future_client():
    class _FakeBatched:
        def create(self, task, **kwargs):  # noqa: ANN001, ARG002
            raise NotImplementedError

        def infer_batch(self, task, frames, **kwargs):  # noqa: ANN001, ARG002
            return [None for _ in frames]

    client = _FakeBatched()
    assert isinstance(client, BatchServingClient)
    assert client.infer_batch("pose", [1, 2, 3]) == [None, None, None]
