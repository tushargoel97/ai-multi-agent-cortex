"""Public training-backend API."""

from .capabilities import capabilities
from .registry import get_backend, registered_backends
from .state import AdapterMetadata, adapter_backend_id

__all__ = [
    "AdapterMetadata",
    "adapter_backend_id",
    "capabilities",
    "get_backend",
    "registered_backends",
]
