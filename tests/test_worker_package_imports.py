from __future__ import annotations

import importlib
import pkgutil

EXCLUDED_PREFIX = "worker.tools"


def test_every_worker_module_imports_cleanly() -> None:
    failures: list[tuple[str, str]] = []
    for module in pkgutil.walk_packages(["worker"], "worker."):
        if module.name.startswith(EXCLUDED_PREFIX):
            continue
        try:
            importlib.import_module(module.name)
        except Exception as error:  # noqa: BLE001
            failures.append((module.name, f"{type(error).__name__}: {error}"))

    assert not failures, "worker modules that fail to import:\n" + "\n".join(
        f"  {name}: {reason}" for name, reason in failures
    )
