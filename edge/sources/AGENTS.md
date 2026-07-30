# SOURCES KNOWLEDGE BASE

Own edge frame intake: video files, webcams, RTSP, source probing, and edge source resolution.

## Local Ownership

- `frame_source.py`: compatibility exports for `FrameSource` implementations.
- `video_file.py`, `webcam.py`, `rtsp.py`, `rtsp_backend.py`: concrete frame sources.
- `camera_probe.py`: local camera discovery.
- `registry.py`: edge source registry for configured sources.

## Imports

Allowed: `contracts`, local `edge/sources`, OpenCV, and standard library helpers.

Forbidden: `edge/runners`, `edge/perception`, `edge/domains`, `edge` runtime orchestration, `shared.events`, `backend`, `training`.

## Focused Tests

- `tests/test_sources_frame_source.py`
- `tests/test_sources_camera.py`
- `tests/test_sources_camera_probe.py`
- `tests/test_sources_rtsp.py`
- `tests/test_sources_no_demo_dependency.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Gotchas

`SourceRegistry` rejects raw paths, traversal, numeric device indexes, and untrusted live descriptors. Keep raw live descriptors out of public/API-facing surfaces.
- Preserve source normalization and registry validation before any capture object is opened.
