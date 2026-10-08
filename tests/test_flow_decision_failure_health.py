from __future__ import annotations

import threading
import uuid
from collections.abc import Callable

from backend.app.features.relay.router import RelayRuntimeStatusRequest
from backend.app.features.status.runtime_status_store import RuntimeStatusStore
from contracts.decode_diagnostics import DecodeSelection
from worker.pipeline.diagnostics.lanes import ExecutionRecordLanes
from worker.runtime.flow.metadata_slot import AcceptanceToken
from worker.runtime.flow.observation_coverage import ObservationCoverage
from worker.runtime.flow.policy_pump import NativePolicyPump
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.worker import WorkerRuntime
from worker.types.metadata import MetadataCounters, MetadataFrame, SourceBinding
from worker.types.perception_frame import (
    BedRegionChannel,
    ChannelState,
    HumanPoseChannel,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBoxChannel,
)

BINDING = SourceBinding(
    worker_boot_id="boot-a",
    child_instance_id="child-a",
    camera_id="camera-a",
    source_generation=1,
    stream_epoch=2,
    transform_id="transform-a",
)


def _frame(seq: int) -> MetadataFrame:
    return MetadataFrame(
        frame=PerceptionFrameV1(
            identity=PerceptionFrameIdentity(
                worker_boot_id="boot-a",
                camera_id="camera-a",
                stream_epoch=2,
                seq=seq,
                source_pts=seq * 100,
            ),
            person_box=PersonBoxChannel(ChannelState.INFERRED_EMPTY),
            human_pose=HumanPoseChannel(ChannelState.INFERRED_EMPTY),
            bed_region=BedRegionChannel(ChannelState.SKIPPED),
        ),
        source_generation=1,
        child_instance_id=uuid.uuid4(),
        native_publish_sequence=seq,
        transform_id="transform-a",
    )


def _flow_diagnostics() -> WorkerDiagnostics:
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode(
        "camera-a",
        DecodeSelection(
            requested="auto",
            selected="nvdec",
            fallback_count=0,
            last_reason=None,
            updated_at_sec=1000.0,
        ),
    )
    diagnostics.register_native_detection("camera-a")
    return diagnostics


def _pump(
    diagnostics: WorkerDiagnostics,
    process: Callable[[MetadataFrame], None],
    execution_records: ExecutionRecordLanes | None = None,
) -> NativePolicyPump:
    pump = object.__new__(NativePolicyPump)
    pump._binding = BINDING
    pump._observation_coverage = ObservationCoverage(BINDING)
    pump._diagnostics = diagnostics
    pump._execution_records = execution_records
    pump.failure_count = 0
    pump.processed_count = 0
    pump._process = process  # type: ignore[method-assign, assignment]
    return pump


def _consume_one(pump: NativePolicyPump, frame: MetadataFrame) -> None:
    class OneFrameSlot:
        def subscribe(self, binding: SourceBinding) -> AcceptanceToken:
            return AcceptanceToken(binding, 0)

        def wait_accepted(self, token: AcceptanceToken, *, timeout_sec: float) -> MetadataFrame:
            pump._stop.set()
            return frame

        def counters(self) -> MetadataCounters:
            return MetadataCounters()

    pump._stop = threading.Event()
    pump._metadata = OneFrameSlot()  # type: ignore[assignment]
    pump.run()


def _edge_detection_after(
    pump: NativePolicyPump, diagnostics: WorkerDiagnostics
) -> dict[str, object]:
    store = RuntimeStatusStore(stale_after_sec=1000.0)
    for seq, at in enumerate((0.0, 5.0, 10.0), start=1):
        _consume_one(pump, _frame(seq))
        payload = RelayRuntimeStatusRequest.model_validate(
            diagnostics.to_payload("facility-1", 1, seq)
        )
        assert store.record(payload.model_dump(), received_at=at).accepted
    facilities = store.snapshot(now=10.0)["facilities"]
    assert isinstance(facilities, dict)
    (camera,) = facilities["facility-1"]["cameras"]
    detection = camera["detection"]
    assert isinstance(detection, dict)
    return detection


def test_flow_frames_that_fail_the_decision_read_blind_decision_not_completing() -> None:
    diagnostics = _flow_diagnostics()

    def fail_decision(frame: MetadataFrame) -> None:
        raise ValueError("scripted decision failure")

    pump = _pump(diagnostics, fail_decision)

    detection = _edge_detection_after(pump, diagnostics)

    assert (detection["state"], detection["reason"]) == ("blind", "decision_not_completing")
    assert {
        key: detection[key]
        for key in ("inference_admitted", "inference_succeeded", "decision_completed")
    } == {
        "inference_admitted": 3,
        "inference_succeeded": 3,
        "decision_completed": 0,
    }
    assert (pump.processed_count, pump.failure_count) == (3, 3)


def test_flow_frames_that_complete_the_decision_read_healthy() -> None:
    diagnostics = _flow_diagnostics()

    def complete_decision(frame: MetadataFrame) -> None:
        diagnostics.record_detection_completed(frame.frame.identity.camera_id)

    pump = _pump(diagnostics, complete_decision)

    detection = _edge_detection_after(pump, diagnostics)

    assert (detection["state"], detection["reason"]) == ("healthy", None)
    assert {
        key: detection[key]
        for key in ("inference_admitted", "inference_succeeded", "decision_completed")
    } == {
        "inference_admitted": 3,
        "inference_succeeded": 3,
        "decision_completed": 3,
    }
    assert (pump.processed_count, pump.failure_count) == (3, 0)


def _fail_on(*failing_seqs: int) -> Callable[[MetadataFrame], None]:
    def process(frame: MetadataFrame) -> None:
        if frame.frame.identity.seq in failing_seqs:
            raise ValueError("scripted decision failure")

    return process


def test_policy_consume_record_counts_failed_frames_as_attempts() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=8)
    pump = _pump(_flow_diagnostics(), _fail_on(1), lanes)

    for seq in (1, 2, 3):
        _consume_one(pump, _frame(seq))

    drained = lanes.drain_for("camera-a", "boot-a", limit=8)
    assert drained is not None
    assert [
        (record.frame_seq, record.payload["processed_count"])
        for record in drained.records
        if record.record_kind == "policy.consume"
    ] == [(2, 2), (3, 3)]


def test_max_frames_cap_counts_failed_frames_as_attempts() -> None:
    pump = _pump(_flow_diagnostics(), _fail_on(1, 2))
    runtime = object.__new__(WorkerRuntime)
    runtime._max_frames_per_camera = 2
    runtime._native_policy_pumps = (pump,)

    _consume_one(pump, _frame(1))
    reached_after_one = runtime._max_frames_completion_check()
    _consume_one(pump, _frame(2))

    assert (reached_after_one, runtime._max_frames_completion_check()) == (False, True)
