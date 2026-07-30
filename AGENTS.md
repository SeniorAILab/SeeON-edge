# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-10
**Commit:** 0ea0269
**Branch:** main

Python/uv + React monorepo for fall and bed-exit detection, organized as three
deployable instances (`front`, `backend`, `edge`) plus a `shared` library, each
owning its subtree to 2 levels. Boundaries are enforced by import-linter
(`[tool.importlinter]` in `pyproject.toml`) — not a hand-rolled walker.

## Package Boundaries

| Instance / Package | Ownership |
| --- | --- |
| `front` | React/Vite SPA, feature-sliced: `src/features/*`, `src/shared/{ui,api}`, `src/app` |
| `backend` | FastAPI gateway, vertical-slice: `app/features/{cameras,clips,evidence,relay,status}` (router+store), `app/core`, `app/shared` |
| `edge` | RTSP inference worker, 2-depth: `sources/perception/domains/runners/evidence/runtime/serving_client/features` |
| `shared` | `shared.events` (backend↔edge wire code); the cross-instance library |
| `contracts` | ADR-0004 vendored from `eldercare-dataset-ops` (top-level shared leaf; `test_vendor_drift` firewall) |
| `tests` | pytest contracts and boundary coverage |

`backend` and `edge` are import-independent (they talk only over relay HTTP);
both may import `contracts` and `shared`. Training belongs to
`eldercare-dataset-ops`; local runtime artifacts belong under `models/` and are
never committed. GPU serving is in-process behind `edge/serving_client`; the seam
exposes a **batch-input contract** so a future networked batched serving service
(50-camera scale) can swap in without a rewrite (deferred; see ADR-0002). Required
GPU/NVDEC infra fails fast (no silent CPU/OpenCV fallback) per ADR-0002.

## Runtime Topology

- `backend` is the FastAPI control/status/relay gateway (`ml-api` image). Product
  routes live under `/api/v1`; health probes stay at `/health/live` / `/health/ready`.
- `edge` consumes configured RTSP streams, runs model/domain logic in-process via
  `edge/serving_client`, and relays facts to `backend` at `/api/v1/relay/*`.
- `edge` is not an RTSP server. Do not add a stream publisher, MediaMTX,
  FFmpeg publisher, or synthetic RTSP runtime surface.

## Deployment identity (dir ↔ image)

Directory/instance names are `front`/`backend`/`edge`. The deployment **images**
and the `ML_API_`/`ML_WORKER_` and `API_*`/`WORKER_*` env prefixes intentionally
keep their **legacy identity** — renaming would break `.env.edge.prod`, the GHCR
image names, and the vendored `contracts/worker_config.py` (ADR-0004):

| Instance | Image / service | Dockerfile | Env identity |
| --- | --- | --- | --- |
| `front` | served by `backend` at `/` | `Dockerfile.backend` (build stage) | — |
| `backend` | `ml-api` | `Dockerfile.backend` | `Settings(env_prefix="ML_API_")`, `API_*` |
| `edge` | `ml-worker` | `Dockerfile.edge` | `WORKER_*`, `ML_WORKER_*` |

## CODE MAP

| Surface | Entry / Owner | Role |
| --- | --- | --- |
| Backend bootstrap | `backend/app/main.py`, `backend/app/lifespan.py` | FastAPI control/status/relay gateway. |
| Backend slices | `backend/app/features/*` | Per-capability router + service + store. |
| Edge bootstrap | `edge/runtime/edge_worker.py` | CLI, runner bundle, supervisors, relay client. |
| Per-frame path | `edge/runtime/camera_worker.py`, `edge/perception/` | Frame → observation → domain signal. |
| Domains | `edge/domains/` | Fall and bed-exit interpretation/latching. |
| Serving seam | `edge/serving_client/` | ServingClient interface + in-process runner provisioning. |
| Model adapters | `edge/runners/` | Registry, device selection, warmup, inference adapters. |
| Event egress | `shared/events/edge_ingest_client.py` | Backend Event API facts and heartbeats. |
| Frontend | `front/src/` | Feature-sliced React SPA. |

## Layout

```text
front/            front instance (React/Vite SPA)
backend/          backend instance (FastAPI)
edge/             edge instance (RTSP inference worker)
shared/           shared library (shared.events)
contracts/        L0 interfaces (vendored)
models/           local, ignored model artifacts
scripts/          operator and edge tooling
tests/            pytest suite
```

## Commands

Run commands from the repository root:

```bash
uv sync
uv run pytest -q
uvx ruff check .
docker build -f Dockerfile.backend .
docker build -f Dockerfile.edge .
pnpm --dir front install --frozen-lockfile && pnpm --dir front test
```

Architecture boundaries are enforced by import-linter (`uv run --group lint lint-imports`,
also a pre-commit hook and CI step); contract-symbol exports by `tests/test_contract_symbol_exports.py`.
## DOCS & DECISIONS

- Architecture map: [`docs/architecture.md`](docs/architecture.md)
- Decision records (explicit-only): [`docs/decisions/`](docs/decisions/) — index in [`docs/decisions/README.md`](docs/decisions/README.md)
- Active plans: [`docs/exec-plan/active/`](docs/exec-plan/active/) · Research: [`docs/research/`](docs/research/) · Rules: [`docs/rules/`](docs/rules/)

## CONVENTIONS

- Keep `contracts` dependency-light and in sync with `eldercare-dataset-ops` (ADR-0004 vendoring); `edge/features` is edge-internal pure math.
- Keep the edge→backend boundary one-way over relay HTTP; do not share runtime state.
- Use `uv` and run `lint-imports` after any import-boundary change.

## ANTI-PATTERNS (THIS PROJECT)

- No `backend` import from `edge`, no `edge` import from `backend`; instances communicate only over relay HTTP.
- No RTSP publishing/runtime server surface; `edge` is an RTSP client only.
- No committed local model artifacts or training code.

## NOTES

- Training belongs to `eldercare-dataset-ops`; `models/` is local and ignored.
- Read the nearest scoped `AGENTS.md` before changing a package.

<!-- BEGIN CRAFT-SKILLS INIT DEVELOPMENT FLOW -->
## Development Flow Recipe

Use an issue-driven loop for all repository work:

1. Open or select one GitHub issue describing the change.
2. Before starting new work, confirm the base worktree has no tracked local changes, then run `git fetch origin` and `git pull --ff-only` on the default branch so the work starts from the latest remote commit. Never pull across uncommitted work or discard user changes; create/reuse the task worktree only after the base is current.
3. Never commit directly on `main`. Use a worktree when you need isolation: `git wt <name>` creates (or reuses) a named worktree off the updated default branch. Reuse a small fixed pool (e.g. `lane-1`~`lane-3`) rather than making a new one per issue.
4. Plan first for non-trivial work: write the intended change, affected files, verification, and rollback note before editing.
5. Fan out into small PRs when a change spans unrelated domains, mixes assets with logic, or needs independent review lanes.
6. Attach review evidence to each PR: tests or checks run, screenshots/transcripts for user-facing behavior, and the issue or planning links that justify the change.
7. Merge only after review. If the user explicitly asks to record a durable decision, hand off to the `document` skill and use `docs/decisions/` as the destination.

Conventions agents must follow:

- Keep each change scoped to its issue. When work — planning, a requirements interview, or implementation — surfaces an out-of-scope problem (a new topic, unrelated bug, or follow-up idea beyond the current issue), open a new GitHub issue for it with one Type label instead of expanding the current change.
- Plan before editing non-trivial code.
- Prefer fan-out PRs over broad mixed-purpose PRs.
- Include review evidence before requesting/performing review.
- Do not merge before review.
- Do not create or require ADRs unless the user explicitly asks for ADRs.
<!-- END CRAFT-SKILLS INIT DEVELOPMENT FLOW -->
