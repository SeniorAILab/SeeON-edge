from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBE = """\
import hashlib
import json
import warnings

from backend.app.main import create_app, no_lifespan

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    payload = create_app(lifespan=no_lifespan).openapi()
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    duplicates = [
        str(warning.message)
        for warning in caught
        if "Duplicate Operation ID" in str(warning.message)
    ]
print(hashlib.sha256(body).hexdigest())
print(len(body))
print(len(duplicates))
for path, _methods in sorted(
    (path, sorted(ops))
    for path, ops in payload["paths"].items()
    if "get" in ops and "head" in ops
):
    get_id = payload["paths"][path]["get"]["operationId"]
    head_id = payload["paths"][path]["head"]["operationId"]
    print(f"{path}\t{get_id}\t{head_id}")
"""


def _probe(seed: int) -> tuple[str, int, int, tuple[tuple[str, str, str], ...]]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=_REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line]
    digest = lines[0]
    length = int(lines[1])
    duplicate_count = int(lines[2])
    pairs = tuple(
        (path, get_id, head_id)
        for path, get_id, head_id in (line.split("\t") for line in lines[3:])
    )
    return digest, length, duplicate_count, pairs


def test_openapi_byte_identical_across_pythonhashseed_0_to_15() -> None:
    results = [_probe(seed) for seed in range(16)]
    digests = {digest for digest, _length, _dups, _pairs in results}
    assert len(digests) == 1
    for digest, length, duplicate_count, pairs in results:
        assert digest == results[0][0]
        assert length == results[0][1]
        assert duplicate_count == 0
        assert len(pairs) == 3
        assert pairs == results[0][3]
        for path, get_id, head_id in pairs:
            assert get_id != head_id, path
            assert get_id.endswith("_get"), (path, get_id)
            assert head_id.endswith("_head"), (path, head_id)
