# core/general_dataset/__init__.py

from .base    import GeneralizedDataset
from .collate import custom_collate_fn, worker_init_fn

# Optionally, define __all__
__all__ = [
    "GeneralizedDataset",
    "custom_collate_fn",
    "worker_init_fn",
]
