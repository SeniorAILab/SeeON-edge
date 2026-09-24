"""Static and synthetic coverage for fall-model output contract admission."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from contracts.model_selection import (
    DatasetPublication,
    ModelPublication,
    ModelSelection,
    canonical_digest,
)
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.fall_output_contract import parse_fall_output_contract
from worker.adapters.model.ort_pose_bbox56 import (
    OrtPoseBbox56Runner,
    PoseBbox56Conformance,
)
from worker.interfaces.fall_model import (
    BINARY_FALL_OUTPUT_CONTRACT,
    THREE_STATE_FALL_OUTPUT_CONTRACT,
    BinaryFallScoreEvidence,
    ThreeStateFallScoreEvidence,
)
from worker.pipeline.diagnostics.emit_policy import model_score_record


class _StaticSession:
    def __init__(self, logits: tuple[float, ...]) -> None:
        self.logits = logits

    def run(self, _output_names: object, _input_feed: object) -> list[object]:
        return [[list(self.logits)]]


def _conformance() -> PoseBbox56Conformance:
    return PoseBbox56Conformance(
        relative_path="conformance/pose-bbox56-v1.json",
        preprocessing_identity="coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        vector_length=56,
        tail_indices=MappingProxyType(
            {"x1": 51, "y1": 52, "x2": 53, "y2": 54, "valid": 55}
        ),
        keypoint_order=(),
        confidence_gate=0.5,
        coordinate_system=MappingProxyType({}),
        window_frames=30,
        stride_frames=5,
        fps=15.0,
        document=MappingProxyType({}),
    )


def _runner(
    logits: tuple[float, ...],
    *,
    output_contract=BINARY_FALL_OUTPUT_CONTRACT,
    temperature: float = 1.0,
) -> OrtPoseBbox56Runner:
    return OrtPoseBbox56Runner(
        _StaticSession(logits),
        temperature,
        None,
        5,
        5,
        False,
        "a" * 64,
        "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        _conformance(),
        output_contract,
    )


def _selection(output_digest: str, *, output_class_count: int = 2) -> ModelSelection:
    return ModelSelection(
        model_publication=ModelPublication("facility/fall", "a" * 40, "b" * 64),
        bundle_members_digest="c" * 64,
        dataset_publication=DatasetPublication("facility/dataset", "d" * 40, "e" * 64),
        evaluation_receipt_digest="f" * 64,
        field_evaluation_receipt_digest="1" * 64,
        calibration_digest="2" * 64,
        conformance_digest="3" * 64,
        input_observation_schema="pose-bbox56.v1",
        output_class_count=output_class_count,
        output_class_semantics_digest=output_digest,
        policy_digest="4" * 64,
        runtime_format="onnxruntime",
        bundle_format="bundle-manifest/proxy-v0",
        preprocessing_identity="coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        transition_threshold=0.5,
        threshold_source="default",
    )


def _admitted_files(tmp_path) -> tuple[object, object, ModelSelection]:
    root = tmp_path / "bundle"
    root.mkdir()
    manifest = {"output_contract": BINARY_FALL_OUTPUT_CONTRACT.manifest_value()}
    payloads = {
        "model.onnx": b"synthetic-static-model",
        "calibration.json": b"{}",
        "bundle-manifest.json": json.dumps(manifest, sort_keys=True).encode(),
    }
    for relative_path, payload in payloads.items():
        (root / relative_path).write_bytes(payload)
    proof = SimpleNamespace(
        observed={
            "member_digests": {
                path: hashlib.sha256(payload).hexdigest()
                for path, payload in payloads.items()
            }
        }
    )
    selection = _selection(canonical_digest(manifest["output_contract"]))
    return root, proof, selection


def test_binary_contract_preserves_sigmoid_and_reports_fallen_unsupported() -> None:
    runner = _runner((1.25,), temperature=2.0)

    result = runner.predict(np.zeros((30, 56), dtype=np.float32))
    expected = float(1.0 / (1.0 + np.exp(-np.float32(1.25) / 2.0)))

    assert (result.background, result.fall_transition, result.fallen) == (
        1.0 - expected,
        expected,
        0.0,
    )
    assert result.model_evidence == BinaryFallScoreEvidence(
        raw_logit=1.25,
        applied_temperature=2.0,
    )
    record = model_score_record(
        camera_id="camera",
        worker_boot_id="boot",
        source_generation=0,
        stream_epoch=1,
        frame_seq=30,
        source_pts_ns=2_000_000_000,
        track_id=7,
        generation=0,
        probability=result,
        observed_at_ns=3_000_000_000,
    )
    assert record is not None
    assert record.payload["fallen"] == 0.0
    assert record.payload["output_contract_id"] == "fall.binary-transition-sigmoid"
    assert record.payload["output_contract_version"] == 1
    assert record.payload["supported_states"] == ["background", "fall_transition"]
    assert record.payload["state_support"]["fallen"] == "unsupported"


def test_static_three_state_contract_uses_softmax() -> None:
    runner = _runner(
        (1.0, 2.0, 3.0),
        output_contract=THREE_STATE_FALL_OUTPUT_CONTRACT,
    )

    result = runner.predict(np.zeros((30, 56), dtype=np.float32))
    expected = np.exp(np.array([1.0, 2.0, 3.0], dtype=np.float32) - 3.0)
    expected /= expected.sum()

    assert (result.background, result.fall_transition, result.fallen) == pytest.approx(
        tuple(float(value) for value in expected)
    )
    assert result.model_evidence == ThreeStateFallScoreEvidence(
        raw_logits=(1.0, 2.0, 3.0),
        applied_temperature=1.0,
    )


def test_output_contract_parser_returns_version_and_supported_states() -> None:
    contract, digest = parse_fall_output_contract(
        {"output_contract": BINARY_FALL_OUTPUT_CONTRACT.manifest_value()}
    )

    assert contract is BINARY_FALL_OUTPUT_CONTRACT
    assert contract.version == 1
    assert contract.supported_states == ("background", "fall_transition")
    assert contract.unsupported_states == ("fallen",)
    assert digest == canonical_digest(BINARY_FALL_OUTPUT_CONTRACT.manifest_value())


def test_output_contract_parser_rejects_missing_metadata() -> None:
    with pytest.raises(ModelLoadError, match="must declare output_contract"):
        parse_fall_output_contract({})


@pytest.mark.parametrize(
    "output_contract",
    [
        {
            "id": "fall.unknown",
            "version": 1,
            "supported_states": ["background", "fall_transition"],
        },
        {
            "id": "fall.binary-transition-sigmoid",
            "version": 2,
            "supported_states": ["background", "fall_transition"],
        },
    ],
)
def test_output_contract_parser_rejects_unknown_id_or_version(
    output_contract: dict[str, object],
) -> None:
    with pytest.raises(ModelLoadError, match="unsupported output_contract id/version"):
        parse_fall_output_contract({"output_contract": output_contract})


def test_output_contract_parser_rejects_supported_states_mismatch() -> None:
    output_contract = BINARY_FALL_OUTPUT_CONTRACT.manifest_value()
    output_contract["supported_states"] = ["background", "fall_transition", "fallen"]

    with pytest.raises(ModelLoadError, match="output_contract metadata mismatch"):
        parse_fall_output_contract({"output_contract": output_contract})


def test_selected_output_contract_digest_mismatch_fails_closed(tmp_path) -> None:
    root, proof, selection = _admitted_files(tmp_path)
    mismatched = _selection("9" * 64)

    with pytest.raises(ModelLoadError, match="output_class_semantics_digest"):
        OrtPoseBbox56Runner.from_admitted_bundle(
            root,
            proof,
            mismatched,
            session_factory=lambda _path, _providers: _StaticSession((0.0,)),
        )


def test_output_contract_tamper_after_admission_fails_closed(tmp_path) -> None:
    root, proof, selection = _admitted_files(tmp_path)
    manifest_path = root / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_contract"]["version"] = 2
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ModelLoadError, match="admitted bundle member changed"):
        OrtPoseBbox56Runner.from_admitted_bundle(
            root,
            proof,
            selection,
            session_factory=lambda _path, _providers: _StaticSession((0.0,)),
        )
