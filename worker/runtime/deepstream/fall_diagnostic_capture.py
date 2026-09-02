from __future__ import annotations

from typing import Literal, Protocol

from worker.native.deepstream.ipc import MetadataFrame
from worker.pipeline.decision import EventAggregator
from worker.runtime.deepstream.fall_diagnostics import FallDiagnosticFrame, FallScoreSnapshot
from worker.types import DecisionInput


class NativeFallDiagnostics(Protocol):
    def record(self, frame: FallDiagnosticFrame) -> None: ...


def require_v1_fall_diagnostic_contract(decision: EventAggregator) -> None:
    for decider in decision.deciders:
        target: object | None = decider
        for _ in range(9):
            if target is None:
                break
            if hasattr(target, "last_trace_snapshots") and hasattr(
                target, "last_score_snapshots"
            ):
                return
            target = getattr(target, "decider", None)
    raise RuntimeError("fall diagnostics require the v1 score/trace contract")


def record_native_fall_diagnostic(
    recorder: NativeFallDiagnostics,
    metadata: MetadataFrame,
    decision_input: DecisionInput,
    decision: EventAggregator,
    track_ids: tuple[int, ...],
) -> None:
    score, previous_state, current_state, triggered = _fall_diagnostic_state(decision)
    frame = metadata.frame
    association = frame.association
    if association is None:
        return
    recorder.record(
        FallDiagnosticFrame(
            source_pts=frame.identity.source_pts or 0,
            source_seq=frame.identity.seq,
            native_publish_seq=metadata.native_publish_sequence,
            source_generation=metadata.source_generation,
            stream_epoch=frame.identity.stream_epoch,
            poses=decision_input.observation.keypoints,
            boxes=tuple(
                (box.x1, box.y1, box.x2, box.y2, box.confidence)
                for box in decision_input.observation.boxes
            ),
            track_ids=track_ids,
            live_track_ids=association.live_track_ids,
            score=score,
            previous_state=previous_state,
            current_state=current_state,
            triggered=triggered,
        )
    )


def _fall_diagnostic_state(
    decision: EventAggregator,
) -> tuple[
    FallScoreSnapshot | None,
    Literal["clear", "fall"],
    Literal["clear", "fall"],
    bool,
]:
    for decider in decision.deciders:
        target: object | None = decider
        for _ in range(9):
            if target is None:
                break
            traces = getattr(target, "last_trace_snapshots", ())
            scores = getattr(target, "last_score_snapshots", ())
            if any(
                getattr(item, "reason", "") == "outside-detection-window"
                for item in traces
            ):
                return None, "clear", "clear", False
            trace = next(
                (
                    item
                    for item in traces
                    if "fall_probability" in getattr(item, "values", {})
                    or getattr(item, "reason", "") == "score-missing"
                ),
                None,
            )
            if trace is not None:
                previous = getattr(trace, "previous_state", "clear")
                current = getattr(trace, "current_state", "clear")
                if previous not in {"clear", "fall"} or current not in {"clear", "fall"}:
                    return None, "clear", "clear", False
                if not scores and getattr(trace, "reason", "") != "score-missing":
                    target = getattr(target, "decider", None)
                    continue
                selected = next(
                    (
                        item
                        for item in scores
                        if item.track_id == getattr(trace, "track_id", None)
                    ),
                    None,
                )
                previous_state: Literal["clear", "fall"] = (
                    "fall" if previous == "fall" else "clear"
                )
                current_state: Literal["clear", "fall"] = (
                    "fall" if current == "fall" else "clear"
                )
                return selected, previous_state, current_state, bool(trace.triggered)
            target = getattr(target, "decider", None)
    return None, "clear", "clear", False


__all__ = [
    "NativeFallDiagnostics",
    "record_native_fall_diagnostic",
    "require_v1_fall_diagnostic_contract",
]
