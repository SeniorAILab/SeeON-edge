from __future__ import annotations

import cv2  # noqa: F401

from contracts.frame import Frame, FrameSource
from worker.pipeline.ingest.rtsp import RTSPSource
from worker.pipeline.ingest.video_file import VideoFileSource
from worker.pipeline.ingest.webcam import CameraSource

__all__ = ["CameraSource", "Frame", "FrameSource", "RTSPSource", "VideoFileSource"]
