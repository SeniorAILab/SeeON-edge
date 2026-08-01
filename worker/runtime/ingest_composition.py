from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from worker.adapters.decode.cpu_av.adapter import CpuAvAdapter
from worker.adapters.decode.cpu_av.models import CpuAvConfig
from worker.adapters.decode.nvdec_cuvid.adapter import NvdecCuvidAdapter
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig
from worker.interfaces.decode import DecodeAdapter
from worker.pipeline.bus.frame_bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import (
    CameraIngestLoop,
    CameraIngestPorts,
    CameraIngestSpec,
    CapturePolicy,
    IngestReporter,
)
from worker.pipeline.ingest.registry import ResolvedSource, SourceRecord, SourceRegistry
from worker.runtime.config import CameraRuntimeConfig
from worker.runtime.profile.registry import DecodePolicy

DecodeConfig: TypeAlias = CpuAvConfig | NvdecCuvidConfig


def build_camera_source_registry(cameras: tuple[CameraRuntimeConfig, ...]) -> SourceRegistry:
    """Allowlist exactly this worker's configured cameras as trusted live sources.

    ``SourceRegistry.resolve`` rejects raw ``rtsp://`` descriptors outright; a
    ``trusted_live`` record keyed by ``camera_id`` is the sanctioned way through
    that gate. The record's ``path`` is a placeholder -- the real, already
    Pydantic-validated RTSP URL comes from ``CameraRuntimeConfig`` directly (see
    ``compose_camera_ingest_loop``), never from the registry.
    """
    records = {
        camera.camera_id: SourceRecord(
            source_id=camera.camera_id,
            path=Path(camera.camera_id),
            duration_sec=0.0,
            mime_type="",
            kind="live",
            trusted_live=True,
        )
        for camera in cameras
    }
    return SourceRegistry(records=records)


def decoder_for(decode: DecodePolicy) -> DecodeAdapter[DecodeConfig]:
    """Build the real decode adapter for a boot-resolved decode token.

    Fail-fast per ADR-0002: an unrecognized token raises immediately rather
    than silently falling back to a default backend.
    """
    if decode == "opencv":
        return CpuAvAdapter()
    if decode == "nvdec":
        return NvdecCuvidAdapter()
    raise RuntimeError(f"unsupported decode policy: {decode!r}")


def _decode_config_factory(
    decode: DecodePolicy, camera: CameraRuntimeConfig
) -> Callable[[str, ResolvedSource], DecodeConfig]:
    def make(camera_id: str, resolved: ResolvedSource) -> DecodeConfig:
        del resolved  # the registry only gates which cameras may ingest
        if decode == "opencv":
            return CpuAvConfig(camera_id=camera_id, url=camera.inference_rtsp_url)
        if decode == "nvdec":
            return NvdecCuvidConfig(camera_id=camera_id, url=camera.inference_rtsp_url)
        raise RuntimeError(f"unsupported decode policy: {decode!r}")

    return make


def compose_camera_ingest_loop(
    camera: CameraRuntimeConfig,
    bus: BoundedFrameBus,
    reporter: IngestReporter,
    *,
    decode: DecodePolicy,
    registry: SourceRegistry,
) -> CameraIngestLoop[DecodeConfig]:
    """Compose the real per-camera ingest loop for the boot-resolved decode profile."""
    ports = CameraIngestPorts(
        registry=registry,
        decoder=decoder_for(decode),
        bus=bus,
        reporter=reporter,
    )
    spec = CameraIngestSpec(
        camera_id=camera.camera_id,
        source_id=camera.camera_id,
        make_decode_config=_decode_config_factory(decode, camera),
        policy=CapturePolicy(target_fps=camera.fps),
    )
    return CameraIngestLoop(spec, ports)


__all__ = [
    "DecodeConfig",
    "build_camera_source_registry",
    "compose_camera_ingest_loop",
    "decoder_for",
]
