"""Float64 reference for the native C++ RGBA preprocessing oracle.

This deliberately mirrors ``src/preprocess_cpu.cpp`` rather than OpenCV.  In
particular, resize samples and blends in float64 before its sole FP32 cast.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from worker.native.deepstream.parity.geometry import (
    LETTERBOX_PAD_VALUE,
    AffineMetadata,
    letterbox_affine,
)


def preprocess_rgba_to_bgr_tensor(
    rgba: bytes | NDArray[np.uint8],
    *,
    width: int,
    height: int,
    stride: int,
    affine: AffineMetadata | None = None,
) -> NDArray[np.float32]:
    """Return the C++ oracle's planar BGR tensor from stride-aware RGBA bytes."""
    if width <= 0 or height <= 0 or stride < width * 4:
        raise ValueError("invalid RGBA geometry")
    source = np.frombuffer(rgba, dtype=np.uint8)
    if source.size != height * stride:
        raise ValueError("RGBA bytes do not match height * stride")
    rows = source.reshape(height, stride)[:, : width * 4].reshape(height, width, 4)
    metadata = letterbox_affine(height, width) if affine is None else affine
    plane = metadata.tensor_height * metadata.tensor_width
    tensor = np.full(
        (3, metadata.tensor_height, metadata.tensor_width),
        LETTERBOX_PAD_VALUE / 255.0,
        dtype=np.float32,
    )

    scale_x = float(width) / metadata.content_width
    scale_y = float(height) / metadata.content_height
    y_coordinates = (np.arange(metadata.content_height, dtype=np.float64) + 0.5) * scale_y - 0.5
    y_coordinates = np.clip(y_coordinates, 0.0, float(height - 1))
    y0 = y_coordinates.astype(np.intp)
    y1 = np.minimum(y0 + 1, height - 1)
    wy = y_coordinates - y0
    x_coordinates = (np.arange(metadata.content_width, dtype=np.float64) + 0.5) * scale_x - 0.5
    x_coordinates = np.clip(x_coordinates, 0.0, float(width - 1))
    x0 = x_coordinates.astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    wx = x_coordinates - x0

    destination = (
        slice(metadata.pad_top, metadata.pad_top + metadata.content_height),
        slice(metadata.pad_left, metadata.pad_left + metadata.content_width),
    )
    for channel, source_byte in enumerate((2, 1, 0)):
        top = (1.0 - wx) * rows[y0[:, None], x0[None, :], source_byte] + wx * rows[
            y0[:, None], x1[None, :], source_byte
        ]
        bottom = (1.0 - wx) * rows[y1[:, None], x0[None, :], source_byte] + wx * rows[
            y1[:, None], x1[None, :], source_byte
        ]
        tensor[channel][destination] = ((1.0 - wy[:, None]) * top + wy[:, None] * bottom) / 255.0
    assert tensor.size == 3 * plane
    return tensor


__all__ = ["preprocess_rgba_to_bgr_tensor"]
