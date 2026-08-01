"""Fatal accelerator fault containment (todo 25).

Responsible for:
- Persisting exactly one first-fault record atomically under ML_WORKER_STATE_DIR.
- Providing a FaultHandler that stops all camera ingest loops and exits with code 4.

The exception TYPE (FatalAcceleratorError) lives in worker.adapters.model.errors so
that worker.pipeline.ingest.lifecycle can import it without breaking the import-linter
contract "worker runtime is the sole composition root".
"""
from __future__ import annotations

from worker.runtime.faults.handler import FATAL_ACCELERATOR_EXIT_CODE, FaultHandler
from worker.runtime.faults.record import FirstFaultRecord, persist_first_fault

__all__ = [
    "FATAL_ACCELERATOR_EXIT_CODE",
    "FaultHandler",
    "FirstFaultRecord",
    "persist_first_fault",
]
