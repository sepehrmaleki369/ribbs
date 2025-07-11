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

def worker_init_fn(worker_id):
    """
    DataLoader worker initialization to ensure different random seeds for each worker.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)