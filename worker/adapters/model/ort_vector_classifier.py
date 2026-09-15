"""CPU ONNX runner for a binary classifier over one fixed-length feature vector.

The fall domain owns which features it sends and what the score means; this
adapter only loads the artifact, checks its declared shape, and returns the
positive-class probability.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from worker.adapters.model.errors import ModelLoadError

_CPU_PROVIDER: Final = ("CPUExecutionProvider",)
_PROBABILITY_OUTPUT: Final = "probabilities"


class _Session(Protocol):
    def get_inputs(self) -> Sequence[object]: ...

    def get_outputs(self) -> Sequence[object]: ...

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, object]
    ) -> Sequence[object]: ...


SessionFactory = Callable[[str, list[str]], _Session]


def _onnxruntime_session_factory(model_path: str, providers: list[str]) -> _Session:
    try:
        import onnxruntime  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - the image always ships it
        raise ModelLoadError("vector classifier requires onnxruntime") from exc
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(model_path, sess_options=options, providers=providers)


class OrtVectorClassifier:
    """Positive-class probability for one feature vector, on the CPU."""

    _session: _Session
    _input_name: str
    feature_dim: int
    artifact_digest: str

    def __init__(
        self,
        session: _Session,
        *,
        input_name: str,
        feature_dim: int,
        artifact_digest: str,
    ) -> None:
        self._session = session
        self._input_name = input_name
        self.feature_dim = feature_dim
        self.artifact_digest = artifact_digest

    @classmethod
    def from_model_path(
        cls,
        model_path: Path,
        *,
        feature_dim: int,
        session_factory: SessionFactory = _onnxruntime_session_factory,
    ) -> OrtVectorClassifier:
        path = model_path.expanduser()
        if not path.is_file():
            raise ModelLoadError(f"vector classifier not found at {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        session = session_factory(str(path), list(_CPU_PROVIDER))
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ModelLoadError("vector classifier must take exactly one input")
        shape = getattr(inputs[0], "shape", None)
        if not isinstance(shape, list) or len(shape) != 2 or shape[1] != feature_dim:
            raise ModelLoadError(
                f"vector classifier input must be (batch, {feature_dim}), got {shape}"
            )
        names = [str(getattr(output, "name", "")) for output in session.get_outputs()]
        if _PROBABILITY_OUTPUT not in names:
            raise ModelLoadError(
                f"vector classifier must expose a {_PROBABILITY_OUTPUT!r} output, got {names}"
            )
        return cls(
            session,
            input_name=str(getattr(inputs[0], "name", "X")),
            feature_dim=feature_dim,
            artifact_digest=digest,
        )

    def positive_probability(self, vector: Sequence[float]) -> float:
        if len(vector) != self.feature_dim:
            raise ModelLoadError(
                f"vector classifier expects {self.feature_dim} features, got {len(vector)}"
            )
        batch = np.asarray([vector], dtype=np.float32)
        outputs = self._session.run([_PROBABILITY_OUTPUT], {self._input_name: batch})
        probabilities = np.asarray(outputs[0], dtype=np.float32)
        if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
            raise ModelLoadError(
                f"vector classifier returned {probabilities.shape}, expected finite (1, 2)"
            )
        return float(probabilities[0, 1])


__all__ = ["OrtVectorClassifier", "SessionFactory"]
