from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ML_API = "ml-api"
ML_WORKER = "ml-worker"
IMAGES = (ML_API, ML_WORKER)
BOTH = frozenset(IMAGES)
NEITHER: frozenset[str] = frozenset()

DOCKERFILES = {ML_API: "Dockerfile.backend", ML_WORKER: "Dockerfile.edge"}

TAG_GLOB = "seeon-edge-v*"

_ML_API_INPUTS = (
    "Dockerfile.backend",
    "backend/",
    "contracts/",
    "front/",
    "pyproject.toml",
    "scripts/ops/",
    "shared/",
    "uv.lock",
)

_ML_WORKER_INPUTS = (
    "Dockerfile.edge",
    "contracts/",
    "pyproject.toml",
    "shared/",
    "uv.lock",
    "worker/",
)

_SHARED_INPUTS = (
    ".dockerignore",
    ".github/workflows/edge-images.yml",
)

_NEUTRAL_INPUTS = (
    ".claude/",
    ".env.edge.prod.example",
    ".env.example",
    ".github/",
    ".gitignore",
    ".gitleaksignore",
    ".omo/",
    ".pre-commit-config.yaml",
    ".python-version",
    "AGENTS.md",
    "DESIGN.md",
    "LICENSE",
    "README.md",
    "artifacts/",
    "compose.edge.dev.yaml",
    "compose.edge.yaml",
    "docs/",
    "edge-env-inventory.json",
    "models/",
    "scripts/",
    "tests/",
    "tests_support/",
)


def _build_rules() -> tuple[tuple[str, frozenset[str]], ...]:
    collected: list[tuple[str, frozenset[str]]] = []
    collected.extend((prefix, BOTH) for prefix in _SHARED_INPUTS)
    collected.extend((prefix, frozenset({ML_API})) for prefix in _ML_API_INPUTS)
    collected.extend((prefix, frozenset({ML_WORKER})) for prefix in _ML_WORKER_INPUTS)
    collected.extend((prefix, NEITHER) for prefix in _NEUTRAL_INPUTS)

    merged: dict[str, frozenset[str]] = {}
    for prefix, images in collected:
        merged[prefix] = merged.get(prefix, NEITHER) | images
    return tuple(sorted(merged.items(), key=lambda item: -len(item[0])))


RULES = _build_rules()


def _matches(path: str, prefix: str) -> bool:
    if prefix.endswith("/"):
        return path == prefix.rstrip("/") or path.startswith(prefix)
    return path == prefix


def is_classified(path: str) -> bool:
    return any(_matches(path, prefix) for prefix, _ in RULES)


def affected_images(path: str) -> frozenset[str]:
    for prefix, images in RULES:
        if _matches(path, prefix):
            return images
    return BOTH


def _short(reference: str) -> str:
    if len(reference) == 40 and all(c in "0123456789abcdef" for c in reference):
        return reference[:12]
    return reference


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def previous_tag(tag: str) -> str | None:
    listed = _git("tag", "--list", TAG_GLOB, "--sort=-version:refname").splitlines()
    for candidate in listed:
        if candidate and candidate != tag:
            return candidate
    return None


def changed_paths(base: str, head: str) -> list[str]:
    return [line for line in _git("diff", "--name-only", f"{base}..{head}").splitlines() if line]


def decide(image: str, base: str, head: str) -> dict[str, object]:
    paths = changed_paths(base, head)
    triggers = [path for path in paths if image in affected_images(path)]
    unclassified = [path for path in triggers if not is_classified(path)]
    return {
        "image": image,
        "base": base,
        "head": head,
        "build": bool(triggers),
        "total_changed": len(paths),
        "input_changed": len(triggers),
        "triggers": triggers,
        "unclassified": unclassified,
    }


class DigestNotPreserved(RuntimeError):
    ...


def _run(argv: Iterable[str]) -> str:
    argv = list(argv)
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout


Runner = Callable[[list[str]], str]


def published_digest(repo: str, tag: str, run: Runner | None = None) -> str:
    run = run or _run
    return run(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            f"{repo}:{tag}",
            "--format",
            "{{.Manifest.Digest}}",
        ]
    ).strip()


def published_revision(repo: str, reference: str, run: Runner | None = None) -> str:
    run = run or _run
    payload = run(
        ["docker", "buildx", "imagetools", "inspect", reference, "--format", "{{json .Image}}"]
    )
    revisions = {
        labels.get("org.opencontainers.image.revision")
        for labels in _iter_label_maps(json.loads(payload))
        if labels
    }
    revisions.discard(None)
    if len(revisions) != 1:
        raise RuntimeError(
            f"{reference} carries {len(revisions)} distinct "
            f"org.opencontainers.image.revision labels ({sorted(revisions)}); "
            "cannot determine the commit that built it"
        )
    return str(revisions.pop())


def _iter_label_maps(node: object) -> Iterable[dict[str, str]]:
    if isinstance(node, dict):
        config = node.get("config")
        if isinstance(config, dict) and isinstance(config.get("Labels"), dict):
            yield config["Labels"]
        for value in node.values():
            yield from _iter_label_maps(value)


def retag_preserving_digest(repo: str, digest: str, tag: str, run: Runner | None = None) -> str:
    run = run or _run
    source = f"{repo}@{digest}"
    run(["docker", "buildx", "imagetools", "create", "-t", f"{repo}:{tag}", source])
    observed = published_digest(repo, tag, run=run)
    if observed != digest:
        raise DigestNotPreserved(
            f"re-tagging {source} as {repo}:{tag} changed the digest to {observed}. "
            "The source manifest was probably not an index, so buildx re-wrapped it. "
            "Refusing to record a 'reused' image under a digest that nothing built."
        )
    return observed


def commit_exists(commit: str) -> bool:
    try:
        _git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError:
        return False
    return True


def plan(
    image: str,
    head: str,
    release_tag: str,
    repo: str,
    reuse_eligible: bool,
    run: Runner | None = None,
    previous: str | None = None,
) -> dict[str, object]:
    def build(reason: str, **extra: object) -> dict[str, object]:
        return {
            "image": image,
            "build": True,
            "reuse": False,
            "reason": reason,
            "head": head,
            **extra,
        }

    if not reuse_eligible:
        return build(
            "this event always builds (reuse is a release-time decision, and the "
            "boot-smoke gate must exercise a freshly built image)"
        )

    if previous is None:
        previous = previous_tag(release_tag)
    if not previous:
        return build(f"no previous {TAG_GLOB} tag exists, so there is no digest to reuse")

    candidates = [previous]
    with contextlib.suppress(subprocess.CalledProcessError):
        candidates.append(_git("rev-list", "-n", "1", previous))

    digest = ""
    for reference in candidates:
        try:
            digest = published_digest(repo, reference, run=run)
        except (subprocess.CalledProcessError, OSError):
            continue
        if digest.startswith("sha256:"):
            break
        digest = ""
    if not digest:
        return build(
            f"the registry resolves none of {', '.join(f'{repo}:{ref}' for ref in candidates)}"
        )

    try:
        base = published_revision(repo, f"{repo}@{digest}", run=run)
    except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as error:
        return build(f"{repo}@{digest} has no usable revision label ({error})")

    if not commit_exists(base):
        return build(
            f"{repo}@{digest} names build commit {base}, which is not in this repository's history"
        )

    decision = decide(image, base, head)
    decision.update(
        {
            "reuse": not decision["build"],
            "previous_tag": previous,
            "previous_digest": digest,
            "reason": (
                f"{decision['input_changed']} input path(s) changed since {_short(base)}, "
                f"which built {previous}"
            )
            if decision["build"]
            else (
                f"no input path changed since {_short(base)}, which built {previous}; "
                "re-tagging that digest instead of rebuilding"
            ),
        }
    )
    return decision


def _emit(name: str, value: str) -> None:
    destination = os.environ.get("GITHUB_OUTPUT")
    if not destination:
        return
    with open(destination, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def render_decision(decision: dict[str, object]) -> str:
    verdict = "BUILD" if decision["build"] else "REUSE"
    lines = [f"{decision['image']}: {verdict} -- {decision['reason']}"]
    if decision.get("previous_tag"):
        lines.append(f"  previous release : {decision['previous_tag']}")
        lines.append(f"  published digest : {decision['previous_digest']}")
    if decision.get("base"):
        lines.append(f"  built at (base)  : {decision['base']}")
    lines.append(f"  releasing (head) : {decision['head']}")
    triggers = decision.get("triggers") or []
    if triggers:
        lines.append(
            f"  {decision['input_changed']} of {decision['total_changed']} changed "
            f"path(s) are inputs to {decision['image']}:"
        )
        lines.extend(f"    - {path}" for path in triggers[:25])  # type: ignore[index]
        if len(triggers) > 25:  # type: ignore[arg-type]
            lines.append(f"    ... and {len(triggers) - 25} more")  # type: ignore[arg-type]
    if decision.get("unclassified"):
        lines.append("  UNCLASSIFIED (fail-closed -> treated as affecting both images):")
        lines.extend(f"    ! {path}" for path in decision["unclassified"][:25])  # type: ignore[index]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-image edge release build plan.")
    sub = parser.add_subparsers(dest="command", required=True)

    paths = sub.add_parser("paths", help="print the input path set for an image")
    paths.add_argument("--image", choices=IMAGES, required=True)

    classify = sub.add_parser("classify", help="print which images a path affects")
    classify.add_argument("path")

    previous = sub.add_parser("previous-tag", help="print the previous release tag")
    previous.add_argument("--tag", required=True)

    decide_parser = sub.add_parser("decide", help="decide whether an image must be rebuilt")
    decide_parser.add_argument("--image", choices=IMAGES, required=True)
    decide_parser.add_argument(
        "--base", required=True, help="commit that built the currently published digest"
    )
    decide_parser.add_argument("--head", required=True, help="commit being released")

    plan_parser = sub.add_parser("plan", help="full per-image decision, registry lookup included")
    plan_parser.add_argument("--image", choices=IMAGES, required=True)
    plan_parser.add_argument("--head", required=True, help="commit being released")
    plan_parser.add_argument("--release-tag", required=True, help="the tag being released")
    plan_parser.add_argument("--repo", required=True, help="e.g. ghcr.io/<ns>/ml-api")
    plan_parser.add_argument(
        "--reuse-eligible",
        choices=("true", "false"),
        required=True,
        help="whether this event is allowed to reuse a published digest at all",
    )
    plan_parser.add_argument(
        "--json-out",
        help="write the decision as JSON here, for later steps to read verbatim",
    )

    retag = sub.add_parser("retag", help="give a published digest a new tag, digest preserved")
    retag.add_argument("--repo", required=True)
    retag.add_argument("--digest", required=True)
    retag.add_argument("--tag", required=True)

    args = parser.parse_args(argv)

    if args.command == "paths":
        for prefix, images in sorted(RULES):
            if args.image in images:
                print(prefix)
        return 0

    if args.command == "classify":
        images = affected_images(args.path)
        print(" ".join(sorted(images)) if images else "(neither)")
        return 0

    if args.command == "previous-tag":
        print(previous_tag(args.tag) or "")
        return 0

    if args.command == "retag":
        print(retag_preserving_digest(args.repo, args.digest, args.tag))
        return 0

    if args.command == "decide":
        decision = decide(args.image, args.base, args.head)
        decision["reason"] = f"inputs compared against {_short(args.base)}"
        print(render_decision(decision))
        return 0

    decision = plan(
        args.image,
        args.head,
        args.release_tag,
        args.repo,
        reuse_eligible=args.reuse_eligible == "true",
    )
    print(render_decision(decision))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    _emit("build", "true" if decision["build"] else "false")
    _emit("reason", str(decision["reason"]))
    _emit("previous-tag", str(decision.get("previous_tag") or ""))
    _emit("previous-digest", str(decision.get("previous_digest") or ""))
    _emit("base", str(decision.get("base") or ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
