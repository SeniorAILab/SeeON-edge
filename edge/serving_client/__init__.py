"""Serving-client seam for the edge inference plane."""

from edge.serving_client.base import BatchServingClient, ServingClient
from edge.serving_client.in_process import InProcessServingClient

__all__ = ["BatchServingClient", "InProcessServingClient", "ServingClient"]
