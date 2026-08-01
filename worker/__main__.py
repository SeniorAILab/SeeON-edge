"""Worker entrypoint: canonical `python -m worker` CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from worker.runtime.bootstrap import BootstrapStageError
from worker.runtime.config import load_worker_config
from worker.runtime.worker import WorkerRuntime

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, load config, and run the worker.

    Exit codes:
      0 — clean shutdown
      1 — generic runtime error
      2 — config/resolution error
      3 — refuse-to-start (bootstrap failure)
      4 — fatal accelerator (handled by WorkerRuntime, exits via os._exit)
    """
    parser = argparse.ArgumentParser(
        prog="python -m worker",
        description="Eldercare fall/bed-exit worker inference pipeline",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="YAML config file path (default: looks for ML_WORKER_CONFIG env or standard locations)",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate config and exit without starting cameras",
    )
    parser.add_argument(
        "--heartbeat-on-start",
        action="store_true",
        help="Send heartbeat POST on READY (used by deployment)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        config = load_worker_config(args.config)
    except Exception as exc:
        LOGGER.error("config resolution failed: %s", exc)
        return 2

    if args.check_config:
        LOGGER.info("config validation passed")
        return 0

    try:
        runtime = WorkerRuntime(config)
        runtime.run()
        return 0
    except BootstrapStageError as exc:
        LOGGER.error("bootstrap failure: %s", exc)
        return 3
    except Exception as exc:
        LOGGER.error("worker runtime error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
