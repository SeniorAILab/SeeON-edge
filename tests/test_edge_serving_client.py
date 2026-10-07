from __future__ import annotations

from contracts.runner import Image, RunnerProtocol, RunnerResult, pose_result
from worker.adapters.model.in_process import InProcessServingClient
from worker.adapters.model.registry import ModelOption, ModelRegistry


def test_in_process_serving_client_passes_through_to_registry() -> None:
    calls: list[tuple[str, dict[str, ModelOption]]] = []

    class _SentinelRunner:
        def __call__(self, _image: Image) -> RunnerResult:
            return pose_result((), ())

    sentinel = _SentinelRunner()

    class StubRegistry(ModelRegistry):
        def create(self, task: str, **kwargs: ModelOption) -> RunnerProtocol:
            calls.append((task, kwargs))
            return sentinel

    client = InProcessServingClient(StubRegistry())

    assert client.create("pose", device="cpu") is sentinel
    assert client.create("person", device="cpu") is sentinel
    assert client.create("bed", device="cpu") is sentinel
    assert client.create("fall") is sentinel

    assert calls == [
        ("pose", {"device": "cpu"}),
        ("person", {"device": "cpu"}),
        ("bed", {"device": "cpu"}),
        ("fall", {}),
    ]
