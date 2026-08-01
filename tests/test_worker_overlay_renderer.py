from __future__ import annotations

import numpy as np

from contracts.frame import Frame
from contracts.observation import BedRegionDebugSnapshot, BoundingBox, FrameObservation
from worker.domains.bed_exit.detector import BedExitMonitor
from worker.domains.bed_exit.schema import BedExitDebugSnapshot, BedStatus
from worker.pipeline.output.overlay import OverlayRenderer
from worker.types import FramePacket


def _packet(image: np.ndarray) -> FramePacket:
    return FramePacket(
        camera_id="cam-1",
        frame=Frame(index=1, time_sec=1.0, image=image),
        pts=1.0,
        seq=1,
        width=image.shape[1],
        height=image.shape[0],
        decode_time_ms=0.0,
    )


def test_overlay_renderer_does_not_call_bed_exit_update(monkeypatch) -> None:
    # worker's BedExitMonitor collapsed edge's separate update_boxes() into
    # update() (worker/domains/bed_exit/detector.py:52-92 has no update_boxes
    # method at all), so only update() can be guarded here.
    def fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("renderer must not mutate bed-exit detector")

    monkeypatch.setattr(BedExitMonitor, "update", fail)
    bed = BoundingBox(1, 1, 20, 20, 0.9)
    person = BoundingBox(2, 2, 10, 10, 0.9)
    observation = FrameObservation(detections=((person,), ()), regions=((bed,), ()))
    snapshot = BedExitDebugSnapshot(
        frame_index=1,
        person_boxes=(person,),
        bed_boxes=(bed,),
        statuses=(BedStatus(0, bed, "occupied", person_id=1),),
        bed_region=BedRegionDebugSnapshot(source="fresh", age_frames=0),
    )

    rendered = OverlayRenderer().render(
        _packet(np.zeros((32, 32, 3), dtype=np.uint8)),
        observation,
        (snapshot,),
    )
    assert rendered.shape == (32, 32, 3)
    assert int(rendered.sum()) > 0


def test_overlay_renderer_encodes_jpeg() -> None:
    observation = FrameObservation()
    jpeg = OverlayRenderer().encode_jpeg(
        _packet(np.zeros((16, 16, 3), dtype=np.uint8)),
        observation,
        (),
    )
    assert jpeg.startswith(b"\xff\xd8")


def test_overlay_renderer_draws_bed_segmentation_polygon() -> None:
    # A bed box carrying a mask contour is rendered as a translucent segmentation
    # fill, not an axis-aligned box; person stays a bbox. The pixel-level fill
    # math itself is already covered directly on draw_region() by
    # tests/test_worker_overlay_primitives.py:26-31
    # (test_draw_region_fills_polygon_interior_only); this test additionally
    # confirms OverlayRenderer.render() wires bed regions through draw_region
    # with fill=True, which the primitives-level test does not exercise.
    polygon = ((4, 4), (28, 4), (28, 28), (4, 28))
    bed = BoundingBox(4, 4, 28, 28, 0.9, polygon)
    observation = FrameObservation(detections=((), ()), regions=((bed,), ()))

    rendered = OverlayRenderer().render(
        _packet(np.zeros((32, 32, 3), dtype=np.uint8)),
        observation,
        (),
    )

    # Interior pixel is tinted by the translucent mask fill (bed color BGR (255,0,0)).
    assert int(rendered[16, 16][0]) > 0
    # A pixel outside the polygon stays black (no full-frame fill / no box-only draw).
    assert int(rendered[1, 1].sum()) == 0
