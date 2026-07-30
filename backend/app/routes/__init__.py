"""App-level route modules (cross-cutting: health probes, model metadata)."""

from backend.app.routes import health, models

__all__ = ["health", "models"]
