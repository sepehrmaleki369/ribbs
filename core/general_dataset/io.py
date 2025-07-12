from pathlib import Path
from typing import Optional
import numpy as np
import rasterio
import logging
import warnings
from rasterio.errors import NotGeoreferencedWarning
import torch
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

logger = logging.getLogger(__name__)

def load_array_from_file(file_path: str) -> Optional[np.ndarray]:
    """
    Load an array from disk. Supports:
      - .npy      → numpy.load
      - .tif/.tiff → rasterio.open + read
    Returns:
      np.ndarray on success, or None if the file cannot be read.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    loaders = {
        '.npy': lambda p: np.load(p),
        '.tif': lambda p: rasterio.open(p).read().astype(np.float32),
        '.tiff': lambda p: rasterio.open(p).read().astype(np.float32),
    }
    
    loader = loaders.get(ext)
    if loader is None:
        logger.warning("Unsupported file extension '%s' for %s", ext, file_path)
        return None

    try:
        return loader(str(path))
    except Exception as e:
        # logger.warning("Failed to load '%s': %s", file_path, e)
        return None
    
def to_tensor(obj):
    """Convert numpy ↦ torch (shared memory) but keep others unchanged."""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)          # 0-copy, preserves shape/dtype
    return obj