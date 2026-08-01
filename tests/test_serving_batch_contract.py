"""serving seam batch-input evolution contract (ADR-0002, acceptance 5).

The seam defines a typed batched-inference swap point (BatchServingClient) for
50-camera scale; the in-process client stays single-frame (ServingClient) and
does NOT implement the batch contract yet (batching backend deferred).

The edge original's other three tests are superseded, not ported — all three
protocol-satisfaction/subset checks are covered by
tests/test_worker_model_serving.py:99-102
(test_batch_serving_client_remains_a_pure_deferred_protocol), which asserts
issubclass(BatchServingClient, ServingClient), that BatchServingClient is
still a pure Protocol (``_is_protocol`` is True), and that "infer_batch" is
its own attribute (a stricter check than the edge original's
``hasattr(BatchServingClient, "infer_batch")``); and by
tests/test_worker_model_serving.py:56-61
(test_in_process_client_satisfies_only_the_single_item_protocol), which
asserts isinstance(InProcessServingClient(...), ServingClient) and
not isinstance(InProcessServingClient(...), BatchServingClient).

Only the fourth edge test survives: it exercises an independent, non-registry
fake client, proving BatchServingClient's runtime_checkable Protocol accepts
any structurally-matching object rather than only worker's own classes — a
distinct guarantee neither superseding test makes.
"""

from __future__ import annotations

from worker.interfaces.serving import BatchServingClient


def test_batch_contract_shape_is_provisionable_by_a_future_client() -> None:
    class _FakeBatched:
        def create(self, task, **kwargs):  # noqa: ANN001, ARG002
            raise NotImplementedError

        def infer_batch(self, task, frames, **kwargs):  # noqa: ANN001, ARG002
            return [None for _ in frames]

    client = _FakeBatched()
    assert isinstance(client, BatchServingClient)
    assert client.infer_batch("pose", [1, 2, 3]) == [None, None, None]
