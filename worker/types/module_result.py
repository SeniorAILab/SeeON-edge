from __future__ import annotations

from dataclasses import dataclass, field

from contracts.runner import RunnerResult


@dataclass(frozen=True, slots=True)
class ModuleResult:
    module_name: str
    result: RunnerResult = field(hash=False)
    elapsed_ms: float
    output_adapter: str | None = None


__all__ = ["ModuleResult"]
