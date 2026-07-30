"""Cross-suite compatibility fixtures."""

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def enable_legacy_dashboard_auth_for_pre_session_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep old API contract tests explicit while production fails closed by default."""

    monkeypatch.setenv("API_ALLOW_LEGACY_DASHBOARD_AUTH", "1")
    yield
