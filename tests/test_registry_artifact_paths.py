from __future__ import annotations

import inspect
from pathlib import Path

from edge.runners.yolo_bed_seg import YoloBedSegRunner
from edge.runners.yolo_pose import YoloPoseRunner


def test_default_yolo_runners_use_canonical_worker_artifact_paths() -> None:
    pose_default = inspect.signature(YoloPoseRunner).parameters["model_path"].default
    bed_default = inspect.signature(YoloBedSegRunner).parameters["model_path"].default

    assert Path(pose_default).parts[-3:] == ("models", "pose", "yolo26n-pose.pt")
    assert Path(bed_default).parts[-3:] == ("models", "bed", "yolo26m-seg.pt")
