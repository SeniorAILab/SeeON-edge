from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from worker.domains import DOMAIN_REGISTRY, BedExitDomainDependencies
from worker.domains.bed_exit import BedExitConfig
from worker.interfaces.decision import Decider
from worker.interfaces.fall_model import FallProbabilities


class _FallModel:
    def predict(self, window: object) -> FallProbabilities:
        del window
        return FallProbabilities(background=1.0, fall_transition=0.0, fallen=0.0)


def test_domain_detectors_no_longer_require_an_enabled_attribute() -> None:
    fall = DOMAIN_REGISTRY["fall"].factory(
        {
            "model": _FallModel(),
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "boot_id": "boot-1",
            "stream_epoch": "1",
            "source_generation": 0,
        }
    )
    bed_exit = DOMAIN_REGISTRY["bed_exit"].factory(
        BedExitDomainDependencies(
            config=BedExitConfig(camera_id="camera-1", facility_id="facility-1"),
            clock=lambda: datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC")),
            boot_id="boot-1",
            stream_epoch="1",
            source_generation=0,
        )
    )

    assert isinstance(fall, Decider)
    assert isinstance(bed_exit, Decider)

    assert not hasattr(bed_exit, "enabled")

    for name in ("fall", "bed_exit"):
        assert DOMAIN_REGISTRY[name].enabled is True
