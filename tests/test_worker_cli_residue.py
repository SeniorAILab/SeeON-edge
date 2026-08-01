"""Residue from the legacy `edge.runtime.edge_worker` CLI
(`tests/test_edge_worker_cli.py`, now deleted) that is not covered by
`tests/test_worker_entrypoint.py`'s coverage of `worker/__main__.py`.

`edge.runtime.edge_worker._parse_args` hand-rolled its own argument parser and
caught unknown flags itself inside `main()`'s `try/except EdgeWorkerConfigError`,
returning exit code 2 as a normal function return value
(`test_edge_worker_cli_rejects_unknown_arguments`). `worker/__main__.py` uses
stdlib `argparse` instead: an unknown flag makes `_build_parser().parse_args(argv)`
call `parser.error(...)`, which raises `SystemExit(2)` directly out of `main()`'s
first line -- before its own `try/except WorkerConfigError` block ever runs. The
documented exit code (docs/architecture.md "Entrypoint", exit code 2 for
config/resolution errors) is preserved; only the delivery mechanism changed from
a return value to an uncaught `SystemExit`. `tests/test_worker_entrypoint.py`
exercises this same argparse-exit pattern for `--help` (`SystemExit(0)`) but has
no test for an unrecognized flag, so that case is ported here.

The legacy file's `test_edge_worker_cli_rejects_non_positive_max_frames`
(`main(["--max-frames-per-camera", "0"]) == 2`) gap is now closed:
`--max-frames-per-camera` is a real `worker/__main__.py` flag, reinstated as
production wiring (not the test-only harness knob the legacy
`edge.runtime.edge_worker_supervisor.EdgeWorkerSupervisor.run(max_frames_per_camera=...)`
was) and threaded end to end through the composition root:

- `worker/__main__.py:62` `_build_parser()` adds `--max-frames-per-camera`
  (`:83-90`), typed with `_positive_int` (`:45-59`, rejects non-integer/<=0
  via `ArgumentTypeError` -> `SystemExit(2)`).
- `worker/__main__.py:189` `main()` passes `max_frames_per_camera=
  args.max_frames_per_camera` into the `WorkerRuntime(...)` constructor.
- `worker/runtime/worker.py:421,435` `WorkerRuntime.__init__` stores it as
  `self._max_frames_per_camera`.
- `worker/runtime/worker.py:616-620` `_activate()` computes
  `completion_check = None if self._max_frames_per_camera is None else
  self._max_frames_completion_check` and passes it into
  `IngestSupervisor(loops, ..., completion_check=completion_check)`.
- `worker/runtime/worker.py:685` `_default_pump_factory` passes
  `max_frames=self._max_frames_per_camera` into each camera's
  `CameraPipelinePump(...)`.
- `worker/pipeline/camera_pipeline.py:98` the pump's own `run()` loop exits
  once `_frames_exhausted()` (`:114-115`) is true for that camera, independent
  of the supervisor.
- `worker/runtime/worker.py:688-700` `_max_frames_completion_check` uses
  `all(...)`, not `any(...)`, over `self.cameras` -- the worker only exits
  once *every* camera has reached the cap, matching edge's `_done` semantics,
  not "first camera to finish".
- `worker/pipeline/ingest/lifecycle.py:199-275` `IngestSupervisor`'s
  `_watch_completion` thread polls that all-cameras check and stops the
  supervisor once it is true, mirroring the pre-existing
  `restart_check`/`_watch_restart` pair structurally.

Its CLI-level coverage -- rejecting non-positive/non-integer values, `--help`
documenting the flag, and constructor passthrough -- lives in
`tests/test_worker_entrypoint.py`, not here, since this file is reserved for
residue with no current home; multi-camera "wait for all, not first-to-finish"
and per-pump self-termination coverage live in
`tests/test_worker_ingest_lifecycle.py`, `tests/test_worker_camera_pipeline_pump.py`,
and `tests/test_worker_max_frames_per_camera_composition.py`.
"""

from __future__ import annotations

import pytest

import worker.__main__ as worker_main


def test_unknown_argument_exits_with_config_error_code() -> None:
    with pytest.raises(SystemExit) as exc_info:
        worker_main.main(["--unknown-option"])

    assert exc_info.value.code == 2
