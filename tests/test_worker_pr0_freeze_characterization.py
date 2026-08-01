"""Worker-side successor of the PR-1 tagged runner-result freeze characterization.

Original edge coverage (edge/runtime/camera_worker.py) and its worker-side disposition:

- ``_run_runner`` unified invocation seam (callable-first, ``.run``-fallback, no
  legacy ``.predict`` probing) -> ported below against
  ``worker.pipeline.analytics.models._runner_call``, the direct successor. One
  behavioral change is intentional and documented at the call site: edge
  resolved the call target *per frame* and raised ``TypeError`` from the call
  itself; worker resolves it once at extractor *provision* time
  (``provision_extractors`` -> ``NamedExtractor``) and raises ``AttributeError``
  eagerly if neither ``__call__`` nor ``.run`` exists, so an unsupported runner
  fails fast at composition instead of on the first frame.
- ``_build_observation`` tagged-``RunnerResult.kind`` dispatch (detection/pose/
  person/bed routing, person-overrides-pose-raw-boxes, detection-clears-
  raw-boxes) -> superseded 1:1 by
  ``tests/test_worker_composite_extractor.py::test_composite_merges_named_results_once_and_preserves_pose_keypoints``,
  which exercises the real ``singledispatch``-based
  ``worker.pipeline.analytics.merge.merge_module_results`` on typed
  ``PoseRunnerResult``/``PersonRunnerResult``/``BedRunnerResult``/
  ``DetectionRunnerResult`` (contracts/runner.py) end to end through
  ``CompositeExtractor.process()``. Not duplicated here.
- ``_events_from_detector`` variant normalization (``None``/dict/list of dicts)
  and ``_with_camera_identity``'s ``setdefault`` enrichment of an untyped event
  dict -> obsolete by architecture change, not ported. Worker deciders never
  return untyped dicts: ``Decider.update()`` returns fully-typed
  ``BusinessEvent`` instances (worker/types/business_event.py) with
  ``camera_id``/``facility_id`` as mandatory constructor fields on every
  decider (e.g. ``worker/domains/fall/detector.py``'s ``FallEventLatch``
  always stamps ``camera_id=self.camera_id, facility_id=self.facility_id`` when
  it builds a ``BusinessEvent``; ``BedExitMonitor`` does the same). There is no
  post-hoc dict-shape normalization or ambient identity enrichment step left to
  characterize.
- Source-construction and mid-iteration failure both mark the camera
  offline/DEGRADED with a ``camera.offline`` ops-equivalent event; per-frame
  processing failure does not -> superseded by
  ``tests/test_worker_ingest_lifecycle.py::test_source_construction_failure_marks_only_that_camera_offline``
  (construction-time), ``::test_two_camera_supervisor_isolates_read_failure_and_masks_credentials``
  (mid-iteration read failure), and
  ``::test_publish_processing_failure_is_not_classified_as_camera_offline``
  (per-frame processing failure), all exercising the real
  ``worker.pipeline.ingest.lifecycle.CameraIngestLoop``. Not duplicated here.
"""

from __future__ import annotations

import numpy as np
import pytest

from contracts.runner import RunnerResult, person_result
from worker.pipeline.analytics.models import _runner_call


class _RunRunner:
    def run(self, image: object) -> RunnerResult:
        del image
        return person_result(((1, 2, 3, 4, 0.9),))


class _CallableRunner:
    def __call__(self, image: object) -> RunnerResult:
        del image
        return person_result(((5, 6, 7, 8, 0.8),))


class _LegacyPredictOnly:
    def predict(self, image: object) -> RunnerResult:
        del image
        return person_result(((9, 10, 11, 12, 0.7),))


class _NoInvocation:
    pass


def test_runner_call_resolves_a_bound_run_method() -> None:
    call = _runner_call(_RunRunner())  # type: ignore[arg-type]

    result = call(np.zeros((1, 1, 3), dtype=np.uint8))

    assert result.kind == "person"
    assert result.boxes == ((1, 2, 3, 4, 0.9),)


def test_runner_call_prefers_a_plain_callable_over_run() -> None:
    call = _runner_call(_CallableRunner())  # type: ignore[arg-type]

    result = call(np.zeros((1, 1, 3), dtype=np.uint8))

    assert result.kind == "person"
    assert result.boxes == ((5, 6, 7, 8, 0.8),)


def test_runner_call_does_not_probe_legacy_predict_methods() -> None:
    # A predict-only object is neither callable nor has `.run`; worker resolves
    # this eagerly at provision time rather than probing method names.
    with pytest.raises(AttributeError):
        _runner_call(_LegacyPredictOnly())  # type: ignore[arg-type]


def test_runner_call_without_supported_invocation_raises() -> None:
    with pytest.raises(AttributeError):
        _runner_call(_NoInvocation())  # type: ignore[arg-type]
