# TESTS KNOWLEDGE BASE

Own pytest coverage for the ML uv project, including dependency-boundary guards and fixtures.

## Local Ownership

- `test_contract_symbol_exports.py`: contract-symbol export checks (runner/tracker/worker_config). Import boundaries are enforced by import-linter (`uv run --group lint lint-imports`), not a pytest walker.
- `edge_worker_fixtures.py`, `demo_app_control_helpers.py`: shared test helpers.
- `test_*`: package-specific and cross-boundary tests.

## Imports

Allowed: any production package needed by the test under coverage, pytest helpers, and local fixtures.

Forbidden: importing private generated data/model artifacts as required test inputs, relying on camera hardware, network services, or uncommitted local files for default tests.

## Commands

```bash
uv run --group lint lint-imports
```

## Gotchas

Keep boundary tests small and explicit. When import policy changes, update the `[tool.importlinter]` contracts in `pyproject.toml` and the relevant AGENTS files in the same change.
## Test Boundary

- Keep default tests deterministic and hardware-free.
- Exercise both allowed and forbidden imports when changing the dependency ladder.
- Use package-focused tests before the full suite.
