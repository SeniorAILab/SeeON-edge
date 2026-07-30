from __future__ import annotations

from edge.domains import DOMAIN_REGISTRY, enabled_domains, list_domains


def test_registry_contains_exactly_enabled_fall_and_bed_exit_domains() -> None:
    assert set(DOMAIN_REGISTRY) == {"fall", "bed_exit"}
    assert set(list_domains()) == {"fall", "bed_exit"}
    assert set(enabled_domains()) == {"fall", "bed_exit"}
    assert list_domains(enabled=False) == ()


def test_enabled_domain_factories_are_active() -> None:
    for registration in DOMAIN_REGISTRY.values():
        detector = registration.factory(None, None)
        assert registration.enabled is True
        assert detector.enabled is True
