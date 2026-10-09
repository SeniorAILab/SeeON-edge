from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_REFERENCE: Final = re.compile(
    r"^(?P<locator>[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)@(?P<revision>[0-9a-f]{40})$"
)


@dataclass(frozen=True)
class ModelReference:
    source_locator: str
    revision: str

    def bundle_dir(self, models_root: Path) -> Path:
        return models_root / "bundles" / self.revision

    def __str__(self) -> str:
        return f"{self.source_locator}@{self.revision}"


def parse_model_reference(raw: str) -> ModelReference:
    match = _REFERENCE.fullmatch(raw.strip())
    if match is None:
        raise ValueError(
            f"model reference must be '<owner>/<name>@<40-hex commit>', got {raw!r}; "
            "branch and tag names are not allowed"
        )
    return ModelReference(match["locator"], match["revision"])


__all__ = ["ModelReference", "parse_model_reference"]
