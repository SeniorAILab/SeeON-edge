from __future__ import annotations

import sys
import threading
from pathlib import Path

from worker.adapters.model.yolo_api import YoloLoadError, YoloModel, load_yolo_model


def _stuck_construct(_path: Path) -> YoloModel:
    threading.Event().wait()
    raise AssertionError("unreachable")


def main() -> None:
    artifact = Path(sys.argv[1])
    try:
        load_yolo_model(artifact, "pose", timeout_seconds=0.05, construct=_stuck_construct)
    except YoloLoadError:
        pass
    else:
        raise AssertionError("expected load_yolo_model to raise YoloLoadError")
    sys.exit(0)


if __name__ == "__main__":
    main()
