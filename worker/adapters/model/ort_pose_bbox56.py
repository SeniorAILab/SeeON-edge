from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar, Final, Literal, Protocol, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.model import POSE_BBOX56_PREPROCESSING_IDENTITY
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.pose_bbox56_bundle_support import (
    read_json,
    verify_bundle,
)
from worker.interfaces.fall_model import BinaryFallScoreEvidence, FallProbabilities
from worker.types import FallModelInput

_SHAPE: Final = (30, 56)
_CPU_PROVIDER: Final = ["CPUExecutionProvider"]
_TAIL_INDICES: Final = {"x1": 51, "y1": 52, "x2": 53, "y2": 54, "valid": 55}


class _OrtSession(Protocol):
    def run(
        self, output_names: Sequence[str] | None, input_feed: dict[str, np.ndarray]
    ) -> Sequence[object]: ...


SessionFactory = Callable[[str, list[str]], _OrtSession]


class _TemporalRule(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True)

    m: Annotated[int, Field(ge=1)]
    n: int

    @model_validator(mode="after")
    def _m_within_n(self) -> _TemporalRule:
        if self.n < self.m:
            raise ValueError("temporal_rule must satisfy 1 <= m <= n")
        return self


class _Calibration(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True)

    class_order: tuple[Literal["non_fall"], Literal["fall_transition_proxy"]]
    preprocessing_identity_digest: Annotated[
        str, Field(pattern=hashlib.sha256(POSE_BBOX56_PREPROCESSING_IDENTITY.encode()).hexdigest())
    ]
    temperature: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    temporal_rule: _TemporalRule
    threshold: Annotated[float, Field(ge=0, le=1)] | None = None
    promotion_eligible: bool


class _Vector(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True)

    length: Literal[56]
    tail_indices: dict[str, int]


class _Confidence(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True)

    gate: float


class _Temporal(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True)

    window_frames: Literal[30]
    stride_frames: int
    fps: float


class _ConformanceDocument(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True)

    preprocessing_identity: str
    vector: _Vector
    confidence: _Confidence
    temporal: _Temporal
    coordinate_system: dict[str, object]
    keypoint_order: tuple[str, ...]


@dataclass(frozen=True)
class PackagedFallBundle:
    runner: OrtPoseBbox56Runner
    published_weights_digest: str
    preprocessing_identity: str


@dataclass(frozen=True)
class PoseBbox56Conformance:
    relative_path: str
    preprocessing_identity: str
    vector_length: int
    tail_indices: Mapping[str, int]
    keypoint_order: tuple[str, ...]
    confidence_gate: float
    coordinate_system: Mapping[str, object]
    window_frames: int
    stride_frames: int
    fps: float
    document: Mapping[str, object]


class OrtPoseBbox56Runner:
    device: Final[str] = "cpu"

    def __init__(
        self,
        session: _OrtSession,
        calibration: _Calibration,
        artifact_digest: str,
        calibration_digest: str,
        conformance: PoseBbox56Conformance,
    ) -> None:
        self._session = session
        self._temperature = calibration.temperature
        self.receipt_threshold = calibration.threshold
        self.receipt_transition_votes = calibration.temporal_rule.m
        self.receipt_transition_window = calibration.temporal_rule.n
        self.promotion_eligible = calibration.promotion_eligible
        self.artifact_digest = artifact_digest
        self.calibration_digest = calibration_digest
        self.preprocessing_identity = conformance.preprocessing_identity
        self.conformance = conformance

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        *,
        session_factory: SessionFactory | None = None,
    ) -> OrtPoseBbox56Runner:
        root = Path(artifact_dir).expanduser().resolve()
        manifest = read_json(root / "bundle-manifest.json")
        verify_bundle(root, manifest)
        files = cast(list[dict[str, str]], cast(dict[str, object], manifest)["files"])
        digests = {item["relative_path"]: item["sha256"] for item in files}
        return cls._load(root, digests, session_factory)

    @classmethod
    def from_admitted_bundle(
        cls,
        artifact_dir: str | Path,
        member_digests: Mapping[str, str],
        *,
        session_factory: SessionFactory | None = None,
    ) -> OrtPoseBbox56Runner:
        return cls._load(Path(artifact_dir).expanduser().resolve(), member_digests, session_factory)

    @classmethod
    def _load(
        cls,
        root: Path,
        digests: Mapping[str, str],
        session_factory: SessionFactory | None,
    ) -> OrtPoseBbox56Runner:
        conformance_paths = _conformance_members(digests)
        if len(conformance_paths) != 1 or not {"model.onnx", "calibration.json"} <= digests.keys():
            raise ModelLoadError(
                "bundle needs model.onnx, calibration.json and exactly one conformance member; "
                f"found {sorted(digests)!r}"
            )
        try:
            calibration = _Calibration.model_validate_json((root / "calibration.json").read_bytes())
            raw = (root / conformance_paths[0]).read_bytes()
            parsed = _ConformanceDocument.model_validate_json(raw)
            conformance = _conformance(conformance_paths[0], parsed, json.loads(raw))
        except (OSError, ValueError) as exc:
            raise ModelLoadError(f"invalid calibration or conformance: {exc}") from exc
        if (
            conformance.preprocessing_identity != POSE_BBOX56_PREPROCESSING_IDENTITY
            or dict(conformance.tail_indices) != _TAIL_INDICES
        ):
            raise ModelLoadError(
                "conformance differs from the runner contract: "
                f"preprocessing_identity {conformance.preprocessing_identity!r}, "
                f"tail_indices {dict(conformance.tail_indices)!r}"
            )
        factory = _onnxruntime_session_factory if session_factory is None else session_factory
        try:
            session = factory(str(root / "model.onnx"), list(_CPU_PROVIDER))
        except Exception as exc:
            raise ModelLoadError(f"cannot load pose-bbox56 ONNX model: {exc}") from exc
        runner = cls(
            session, calibration, digests["model.onnx"], digests["calibration.json"], conformance
        )
        runner.warmup()
        return runner

    def predict(self, features: FallModelInput) -> FallProbabilities:
        values = np.asarray(features, dtype=np.float32)
        if values.shape != _SHAPE or not np.isfinite(values).all():
            raise ModelLoadError("pose-bbox56 input must be finite shape (30, 56)")
        try:
            (output,) = self._session.run(None, {"window": values[np.newaxis, ...]})
            logits = np.asarray(output, dtype=np.float32).reshape(1, 1)
        except Exception as exc:
            raise ModelLoadError(f"cannot run pose-bbox56 ONNX model: {exc}") from exc
        if not np.isfinite(logits).all():
            raise ModelLoadError("pose-bbox56 ONNX model returned a non-finite logit")
        fall_transition = float(1.0 / (1.0 + np.exp(-logits[0, 0] / self._temperature)))
        return FallProbabilities(
            background=1.0 - fall_transition,
            fall_transition=fall_transition,
            fallen=0.0,
            model_evidence=BinaryFallScoreEvidence(
                raw_logit=float(logits[0, 0]),
                applied_temperature=self._temperature,
            ),
        )

    def warmup(self) -> None:
        self.predict(np.zeros(_SHAPE, dtype=np.float32))


def load_packaged_fall_bundle(artifact_dir: Path) -> PackagedFallBundle:
    root = artifact_dir.expanduser().resolve()
    runner = OrtPoseBbox56Runner.from_artifact_dir(root)
    files = cast(
        list[dict[str, str]],
        cast(dict[str, object], read_json(root / "bundle-manifest.json"))["files"],
    )
    weights = next((i["sha256"] for i in files if i["relative_path"] == "model.pt"), None)
    if weights is None:
        raise ModelLoadError("bundle manifest does not list model.pt")
    return PackagedFallBundle(runner, weights, runner.preprocessing_identity)


def _onnxruntime_session_factory(model_path: str, providers: list[str]) -> _OrtSession:
    try:
        import onnxruntime
    except ImportError as exc:
        raise ModelLoadError("onnxruntime is required for pose-bbox56 ONNX bundles") from exc
    return onnxruntime.InferenceSession(model_path, providers=providers)


def _conformance_members(paths: Iterable[str]) -> list[str]:
    return [path for path in paths if Path(path).parent == Path("conformance")]


def _conformance(
    relative_path: str, parsed: _ConformanceDocument, document: Mapping[str, object]
) -> PoseBbox56Conformance:
    return PoseBbox56Conformance(
        relative_path=relative_path,
        preprocessing_identity=parsed.preprocessing_identity,
        vector_length=parsed.vector.length,
        tail_indices=MappingProxyType(parsed.vector.tail_indices),
        keypoint_order=parsed.keypoint_order,
        confidence_gate=parsed.confidence.gate,
        coordinate_system=MappingProxyType(parsed.coordinate_system),
        window_frames=parsed.temporal.window_frames,
        stride_frames=parsed.temporal.stride_frames,
        fps=parsed.temporal.fps,
        document=MappingProxyType(dict(document)),
    )


__all__ = [
    "OrtPoseBbox56Runner",
    "PackagedFallBundle",
    "PoseBbox56Conformance",
    "SessionFactory",
    "load_packaged_fall_bundle",
]
