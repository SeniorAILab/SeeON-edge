from __future__ import annotations

from pathlib import Path


class OnnxShapeError(RuntimeError):
    ...


def input_dims(path: Path) -> tuple[int | str | None, ...]:
    try:
        import onnxruntime

        inputs = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"]).get_inputs()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise OnnxShapeError(f"cannot load ONNX input shape from {path}: {error}") from error
    if not inputs:
        raise OnnxShapeError(f"ONNX artifact has no inputs: {path}")
    return tuple(inputs[0].shape)


def batch_axis_is_dynamic(dims: tuple[int | str | None, ...]) -> bool:
    return not isinstance(dims[0], int)
