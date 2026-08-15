#!/usr/bin/env python3
"""Parse the operator-private hash-only model receipt.

Shared by the pre-boot materializer, verifier, and receipt generator so the
written schema cannot drift from the consumed contract.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
sha_re = re.compile(r"^[0-9a-f]{64}$")
rev_re = re.compile(r"^[0-9a-f]{40}$")
name_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
root_re = re.compile(r"^/[A-Za-z0-9/._-]+$")
rel_re = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._/-]*$")
allowed_top = {"schemaVersion", "source", "artifacts", "sidecars"}
allowed_source = {"kind", "container", "imageDigest", "revision", "root"}
allowed_item = {"path", "sha256", "class"}


def die() -> None:
    raise SystemExit(1)


try:
    body = json.loads(receipt_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    die()
if not isinstance(body, dict) or set(body) != allowed_top:
    die()
if body.get("schemaVersion") != 1:
    die()
source = body.get("source")
if not isinstance(source, dict) or set(source) != allowed_source:
    die()
if source.get("kind") != "docker-cli":
    die()
container = source.get("container")
digest = source.get("imageDigest")
revision = source.get("revision")
root = source.get("root")
if not isinstance(container, str) or not name_re.fullmatch(container):
    die()
if not isinstance(digest, str) or not sha_re.fullmatch(digest):
    die()
if not isinstance(revision, str) or not rev_re.fullmatch(revision):
    die()
if not isinstance(root, str) or not root_re.fullmatch(root) or ".." in root.split("/"):
    die()
artifacts = body.get("artifacts")
sidecars = body.get("sidecars")
if not isinstance(artifacts, list) or not artifacts or not isinstance(sidecars, list):
    die()

seen: set[str] = set()
rows: list[tuple[str, str, str]] = []
for items, allowed_class in ((artifacts, {"weight", "provenance"}), (sidecars, {"sidecar"})):
    for item in items:
        if not isinstance(item, dict) or set(item) != allowed_item:
            die()
        path = item.get("path")
        digest_value = item.get("sha256")
        klass = item.get("class")
        if not isinstance(path, str) or not isinstance(digest_value, str):
            die()
        if not isinstance(klass, str):
            die()
        if not rel_re.fullmatch(path) or "//" in path or path.endswith("/"):
            die()
        if any(part in {".", ".."} for part in path.split("/")):
            raise SystemExit(3)
        if not sha_re.fullmatch(digest_value) or klass not in allowed_class:
            die()
        if path in seen:
            die()
        seen.add(path)
        rows.append((path, digest_value, klass))

(out_dir / "source.env").write_text(
    f"CONTAINER={container}\nIMAGE_DIGEST={digest}\nREVISION={revision}\nSOURCE_ROOT={root}\n",
    encoding="utf-8",
)
artifact_lines = []
sidecar_lines = []
for path, digest_value, klass in rows:
    line = f"{path}\t{digest_value}\t{klass}\n"
    if klass == "sidecar":
        sidecar_lines.append(line)
    else:
        artifact_lines.append(line)
(out_dir / "artifacts.tsv").write_text("".join(artifact_lines), encoding="utf-8")
(out_dir / "sidecars.tsv").write_text("".join(sidecar_lines), encoding="utf-8")
raise SystemExit(0)
