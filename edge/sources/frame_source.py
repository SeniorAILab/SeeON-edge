from __future__ import annotations

import cv2  # noqa: F401

from contracts.frame import Frame, FrameSource
from edge.sources.rtsp import RTSPSource
from edge.sources.video_file import VideoFileSource
from edge.sources.webcam import CameraSource

__all__ = ["Frame", "FrameSource", "VideoFileSource", "CameraSource", "RTSPSource"]
