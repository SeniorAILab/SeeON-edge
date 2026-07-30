from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, overload

import numpy as np
import torch
import torch.nn as nn
import yaml
from numpy.typing import NDArray

KPT_VECTOR_DIM = 51
CURRENT_SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
CURRENT_PREPROCESSING_IDENTITY = "coco17-xyc-frame-normalized-zero-fill-v1"
LEGACY_PREPROCESSING_IDENTITY = "legacy-coco17-xyc-frame-normalized-zero-fill-v1"
SUPPORTED_PREPROCESSING_IDENTITIES = frozenset(
    {CURRENT_PREPROCESSING_IDENTITY, LEGACY_PREPROCESSING_IDENTITY}
)
YamlValue: TypeAlias = (
    str | int | float | bool | None | list["YamlValue"] | dict[str, "YamlValue"]
)
StateDict: TypeAlias = Mapping[str, torch.Tensor]


class ModelLoadError(RuntimeError):
    pass


class _LstmNet(nn.Module):
    def __init__(self, *, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=KPT_VECTOR_DIM,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


@dataclass(frozen=True, slots=True)
class LstmMetadata:
    window: int
    stride: int
    mode: Literal["sequence"]


@dataclass(frozen=True, slots=True)
class LstmFallManifest:
    type: Literal["lstm"]
    framework: Literal["pytorch"]
    mode: Literal["sequence"]
    artifact_dir: Path
    weights: str
    architecture: str
    metadata: str
    window: int
    stride: int
    input_shape: tuple[int, int]
    operating_threshold: float
    schema_version: int
    preprocessing_identity: str

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> LstmFallManifest:
        root = Path(artifact_dir).expanduser().resolve()
        metadata_path = root / "metadata.yaml"
        if not metadata_path.exists():
            raise ModelLoadError(f"missing metadata.yaml at {metadata_path}")
        if (root / "metadata.json").exists():
            raise ModelLoadError(
                "LSTM runtime requires metadata.yaml; metadata.json fallback is not supported"
            )
        return cls.from_yaml(metadata_path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> LstmFallManifest:
        metadata_path = Path(path).expanduser().resolve()
        try:
            raw = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ModelLoadError(f"cannot read metadata.yaml: {metadata_path}") from exc
        except yaml.YAMLError as exc:
            raise ModelLoadError(f"metadata.yaml is not valid YAML: {metadata_path}") from exc
        if not isinstance(raw, dict):
            raise ModelLoadError("metadata.yaml must contain an object")

        required = (
            "type",
            "framework",
            "mode",
            "artifact_dir",
            "weights",
            "architecture",
            "metadata",
            "window",
            "stride",
            "input_shape",
            "operating_threshold",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ModelLoadError(f"metadata.yaml missing required fields: {', '.join(missing)}")

        artifact_dir = _resolve_artifact_dir(raw["artifact_dir"], metadata_path.parent)
        input_shape = _parse_input_shape(raw["input_shape"])
        window = _positive_int(raw["window"], "window")
        stride = _positive_int(raw["stride"], "stride")
        schema_version = _schema_version(raw.get("schema_version", LEGACY_SCHEMA_VERSION))
        preprocessing_identity = _preprocessing_identity(raw, schema_version)
        manifest = cls(
            type=_literal(raw["type"], "lstm", "type"),
            framework=_literal(raw["framework"], "pytorch", "framework"),
            mode=_literal(raw["mode"], "sequence", "mode"),
            artifact_dir=artifact_dir,
            weights=str(raw["weights"]),
            architecture=str(raw["architecture"]),
            metadata=str(raw["metadata"]),
            window=window,
            stride=stride,
            input_shape=input_shape,
            operating_threshold=float(raw["operating_threshold"]),
            schema_version=schema_version,
            preprocessing_identity=preprocessing_identity,
        )
        # Manifest and preprocessing validation completes before any weights are read.
        manifest._validate()
        return manifest

    @property
    def weights_path(self) -> Path:
        return self.artifact_dir / self.weights

    @property
    def architecture_path(self) -> Path:
        return self.artifact_dir / self.architecture

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / self.metadata

    def _validate(self) -> None:
        if self.input_shape != (self.window, KPT_VECTOR_DIM):
            raise ModelLoadError(f"input_shape must be [{self.window}, {KPT_VECTOR_DIM}]")
        if not 0.0 <= self.operating_threshold <= 1.0:
            raise ModelLoadError("operating_threshold must be between 0 and 1")
        if self.metadata != "metadata.yaml":
            raise ModelLoadError("metadata must be metadata.yaml")
        if self.schema_version not in (LEGACY_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION):
            raise ModelLoadError(
                f"unsupported schema_version {self.schema_version}; "
                f"supported versions are {LEGACY_SCHEMA_VERSION} and {CURRENT_SCHEMA_VERSION}"
            )
        if self.preprocessing_identity not in SUPPORTED_PREPROCESSING_IDENTITIES:
            raise ModelLoadError(
                "unsupported preprocessing_identity "
                f"{self.preprocessing_identity!r}; expected one of "
                f"{sorted(SUPPORTED_PREPROCESSING_IDENTITIES)!r}"
            )
        if (
            self.schema_version == CURRENT_SCHEMA_VERSION
            and self.preprocessing_identity != CURRENT_PREPROCESSING_IDENTITY
        ):
            raise ModelLoadError(
                "schema_version 2 requires preprocessing_identity "
                f"{CURRENT_PREPROCESSING_IDENTITY!r}"
            )
        for path in (self.weights_path, self.architecture_path, self.metadata_path):
            if not path.exists():
                raise ModelLoadError(f"missing {path.name} at {path}")


class LstmFallRunner:
    name = "fall-detector"
    version = "lstm"

    def __init__(
        self, manifest: LstmFallManifest, module: nn.Module, device: str = "cpu"
    ) -> None:
        self.manifest = manifest
        self.metadata = LstmMetadata(
            window=manifest.window,
            stride=manifest.stride,
            mode=manifest.mode,
        )
        self.operating_threshold = manifest.operating_threshold
        self.preprocessing_identity = manifest.preprocessing_identity
        self.schema_version = manifest.schema_version
        try:
            self.device = torch.device(device)
            self.model = module.to(self.device).eval()
        except (RuntimeError, TypeError) as exc:
            raise ModelLoadError(
                f"cannot place LSTM fall model on device {device!r}: {exc}"
            ) from exc

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        device: str = "cpu",
        *,
        expected_schema_version: int | None = None,
        expected_preprocessing_identity: str | None = None,
    ) -> LstmFallRunner:
        manifest = LstmFallManifest.from_artifact_dir(artifact_dir)
        _validate_expected_identity(
            manifest,
            schema_version=expected_schema_version,
            preprocessing_identity=expected_preprocessing_identity,
        )
        module = build_lstm_module_from_arch(manifest.architecture_path)
        state_dict = _load_state_dict(manifest.weights_path)
        try:
            module.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise ModelLoadError(f"cannot load model.pt state_dict: {exc}") from exc
        return cls(manifest, module, device=device)

    def predict(self, features: NDArray[np.float32]) -> float:
        sequence = np.asarray(features, dtype=np.float32)
        if sequence.shape != self.manifest.input_shape:
            raise ModelLoadError(
                "unexpected input shape: "
                f"expected {self.manifest.input_shape}, received {sequence.shape}"
            )
        with torch.no_grad():
            tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
        return float(np.clip(probabilities[0, 1].item(), 0.0, 1.0))


def _validate_expected_identity(
    manifest: LstmFallManifest,
    *,
    schema_version: int | None,
    preprocessing_identity: str | None,
) -> None:
    if schema_version is not None and manifest.schema_version != schema_version:
        raise ModelLoadError(
            f"artifact schema_version {manifest.schema_version} does not match configured "
            f"schema_version {schema_version}"
        )
    if (
        preprocessing_identity is not None
        and manifest.preprocessing_identity != preprocessing_identity
    ):
        raise ModelLoadError(
            "artifact preprocessing_identity "
            f"{manifest.preprocessing_identity!r} does not match configured "
            f"preprocessing_identity {preprocessing_identity!r}"
        )

def build_lstm_module(*, hidden: int, layers: int, dropout: float) -> nn.Module:
    return _LstmNet(hidden=int(hidden), layers=int(layers), dropout=float(dropout))


def build_lstm_module_from_arch(path: Path) -> nn.Module:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelLoadError(f"cannot read arch.json: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"arch.json is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ModelLoadError("arch.json must contain an object")
    return build_lstm_module(
        hidden=_positive_int(raw.get("hidden"), "hidden"),
        layers=_positive_int(raw.get("layers"), "layers"),
        dropout=float(raw.get("dropout", 0.0)),
    )


def _load_state_dict(path: Path) -> StateDict:
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ModelLoadError(f"cannot load model.pt: {path}") from exc


def _resolve_artifact_dir(raw: YamlValue, base: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@overload
def _literal(
    value: YamlValue, expected: Literal["lstm"], field_name: str
) -> Literal["lstm"]: ...


@overload
def _literal(
    value: YamlValue, expected: Literal["pytorch"], field_name: str
) -> Literal["pytorch"]: ...


@overload
def _literal(
    value: YamlValue, expected: Literal["sequence"], field_name: str
) -> Literal["sequence"]: ...


def _literal(value: YamlValue, expected: str, field_name: str) -> str:
    if value != expected:
        raise ModelLoadError(f"{field_name} must be {expected!r}")
    return expected


def _parse_input_shape(value: YamlValue) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ModelLoadError("input_shape must contain [window, 51]")
    return (_positive_int(value[0], "input_shape[0]"), _positive_int(value[1], "input_shape[1]"))


def _positive_int(value: YamlValue, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelLoadError(f"{field_name} must be a positive integer")
    return value


def _schema_version(value: YamlValue) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelLoadError("schema_version must be an integer")
    return value


def _preprocessing_identity(raw: dict[str, YamlValue], schema_version: int) -> str:
    value = raw.get("preprocessing_identity")
    if value is None and schema_version == LEGACY_SCHEMA_VERSION:
        # Schema v1 bundles did not record preprocessing identity. Their
        # documented compatibility tag is the only permitted implicit value.
        return LEGACY_PREPROCESSING_IDENTITY
    if not isinstance(value, str) or not value:
        raise ModelLoadError("preprocessing_identity must be a non-empty string")
    return value

__all__ = [
    "CURRENT_PREPROCESSING_IDENTITY",
    "CURRENT_SCHEMA_VERSION",
    "LEGACY_PREPROCESSING_IDENTITY",
    "LEGACY_SCHEMA_VERSION",
    "LstmFallManifest",
    "LstmFallRunner",
    "LstmMetadata",
    "ModelLoadError",
    "SUPPORTED_PREPROCESSING_IDENTITIES",
    "build_lstm_module",
    "build_lstm_module_from_arch",
]
