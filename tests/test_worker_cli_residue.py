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
(`main(["--max-frames-per-camera", "0"]) == 2`) is NOT ported: `--max-frames-per-camera`
was a test-only harness knob threaded through
`edge.runtime.edge_worker_supervisor.EdgeWorkerSupervisor.run(max_frames_per_camera=...)`
to bound how many frames a camera loop processes before returning, for use in
tests. `worker/__main__.py` has no such flag (see `_build_parser` in that file),
and `WorkerRuntime.run(self) -> None` (worker/runtime/worker.py) takes no
frame-count parameter -- the bounded-run harness mechanism was not carried
forward into the composition root. This is an open coverage gap (there is
nothing to port a CLI-level test against), not silently dropped: flagging it
here so it is not lost if the harness knob is reinstated elsewhere.
"""

from __future__ import annotations

import pytest

import worker.__main__ as worker_main


def test_unknown_argument_exits_with_config_error_code() -> None:
    with pytest.raises(SystemExit) as exc_info:
        worker_main.main(["--unknown-option"])

    assert exc_info.value.code == 2
