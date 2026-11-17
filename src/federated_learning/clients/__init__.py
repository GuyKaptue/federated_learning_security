"""Client-side components: local training on a client's private data."""
from .client import FederatedClient
from .local_training import local_train

__all__ = ["FederatedClient", "local_train"]
