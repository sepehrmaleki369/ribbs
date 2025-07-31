import torch
from typing import Any, Dict, List, Optional
import random
import numpy as np
from core.general_dataset.logger import logger


# def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     Custom collate function with None filtering.
#     """
#     # Filter out None samples
#     batch = [sample for sample in batch if sample is not None]
    
#     # Handle empty batch case
#     if not batch:
#         logger.warning("Empty batch after filtering None values")
#         return {}  # Or return a default empty batch structure
    
#     # Original collation logic
#     collated: Dict[str, Any] = {}
#     for key in batch[0]:
#         items = []
#         for sample in batch:
#             value = sample[key]
#             if isinstance(value, np.ndarray):
#                 value = torch.from_numpy(value)
#             items.append(value)
#         if isinstance(items[0], torch.Tensor):
#             collated[key] = torch.stack(items)
#         else:
#             collated[key] = items
#     return collated

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = [b for b in batch if b is not None]
    if not batch:
        logger.warning("Empty batch after filtering None values")
        return {}

    collated: Dict[str, Any] = {}
    for key in batch[0]:
        items = [sample[key] for sample in batch]  # tensors already
        if torch.is_tensor(items[0]):              # stack on new batch dim
            collated[key] = torch.stack(items, dim=0)
        else:
            collated[key] = items                  # e.g. metadata strings
    return collated

_worker_rngs = {}
def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    base_seed = info.seed  # unique per worker
    # Seed all libs for completeness
    np.random.seed(base_seed % 2**32)
    random.seed(base_seed)
    torch.manual_seed(base_seed)

    # Make & store a dedicated numpy Generator for this worker
    _worker_rngs[worker_id] = np.random.default_rng(base_seed)