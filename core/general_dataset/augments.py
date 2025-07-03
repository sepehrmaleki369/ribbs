
from typing import Any, Dict, List, Optional
import numpy as np
from core.general_dataset.logger import logger

import math
from scipy.ndimage import rotate

def get_augmentation_metadata(augmentations) -> Dict[str, Any]:
    """
    Generate random augmentation parameters for a patch.

    Returns:
        Dict[str, Any]: Augmentation metadata.
    """
    meta: Dict[str, Any] = {}
    if 'rotation' in augmentations:
        meta['angle'] = np.random.uniform(0, 360)
    if 'flip_h' in augmentations:
        meta['flip_h'] = np.random.rand() > 0.5
    if 'flip_v' in augmentations:
        meta['flip_v'] = np.random.rand() > 0.5
    return meta


def flip_h(full_array: np.ndarray) -> np.ndarray:
    return np.flip(full_array, axis=-1)

def flip_v(full_array: np.ndarray) -> np.ndarray:
    return np.flip(full_array, axis=-2)

def rotate_(full_array: np.ndarray, patch_meta: Dict[str, Any], patch_size) -> np.ndarray:
    """
    Rotate a patch using an expanded crop to avoid border effects.
    If the crop is too small, log a warning and return a zero patch.

    Args:
        full_array (np.ndarray): Full image array.
        patch_meta (Dict[str, Any]): Contains patch coordinates and angle.
    
    Returns:
        np.ndarray: Rotated patch.
    """
    L = int(np.ceil(patch_size * math.sqrt(2)))
    x = patch_meta["x"]
    y = patch_meta["y"]
    angle = patch_meta["angle"]

    cx = x + patch_size // 2
    cy = y + patch_size // 2
    half_L = L // 2
    x0 = max(0, cx - half_L)
    y0 = max(0, cy - half_L)
    x1 = min(full_array.shape[-1], cx + half_L)
    y1 = min(full_array.shape[-2], cy + half_L)

    if full_array.ndim == 3:
        crop = full_array[:, y0:y1, x0:x1]
        if crop.shape[1] < L or crop.shape[2] < L:
            logger.warning("Crop too small for 3D patch rotation; returning zero patch.")
            return np.zeros((full_array.shape[0], patch_size, patch_size), dtype=full_array.dtype)
        rotated_channels = [rotate(crop[c], angle, reshape=False, order=1) for c in range(full_array.shape[0])]
        rotated = np.stack(rotated_channels)
        start = (L - patch_size) // 2
        return rotated[:, start:start + patch_size, start:start + patch_size]
    elif full_array.ndim == 2:
        crop = full_array[y0:y1, x0:x1]
        if crop.shape[0] < L or crop.shape[1] < L:
            logger.warning("Crop too small for 2D patch rotation; returning zero patch.")
            return np.zeros((patch_size, patch_size), dtype=full_array.dtype)
        rotated = rotate(crop, angle, reshape=False, order=1)
        start = (L - patch_size) // 2
        return rotated[start:start + patch_size, start:start + patch_size]
    else:
        raise ValueError("Unsupported array shape")
    

def extract_condition_augmentations(imgs: Dict[str, np.ndarray], metadata: Dict[str, Any], patch_size, augmentations) -> Dict[str, np.ndarray]:
    """
    Extract a patch from the full image and apply conditional augmentations.

    Args:
        imgs (Dict[str, np.ndarray]): Full images for each modality.
        metadata (Dict[str, Any]): Metadata containing patch coordinates and augmentations.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary of extracted patches.
    """
    imgs_aug = imgs.copy()
    data = extract_data(imgs, metadata['x'], metadata['y'], patch_size)
    for key in imgs:
        if key.endswith("_patch"):
            modality = key.replace("_patch", "")
            if 'flip_h' in augmentations:
                imgs_aug[modality] = flip_h(imgs[modality])
                data[key] = flip_h(data[key])
            if 'flip_v' in augmentations:
                imgs_aug[modality] = flip_v(imgs[modality])
                data[key] = flip_v(data[key])
            if 'rotation' in augmentations:
                data[key] = rotate_(imgs_aug[modality], metadata, patch_size)
    return data

def extract_data(imgs: Dict[str, np.ndarray], x: int, y: int, patch_size) -> Dict[str, np.ndarray]:
    """
    Extract a patch from each modality starting at (x, y) with size patch_size.

    Args:
        imgs (Dict[str, np.ndarray]): Full images.
        x (int): x-coordinate.
        y (int): y-coordinate.
    
    Returns:
        Dict[str, np.ndarray]: Extracted patch for each modality.
    """
    data: Dict[str, np.ndarray] = {}
    for key, array in imgs.items():
        if array.ndim == 3:
            data[f"{key}_patch"] = array[:, y:y + patch_size, x:x + patch_size]
        elif array.ndim == 2:
            data[f"{key}_patch"] = array[y:y + patch_size, x:x + patch_size]
        else:
            raise ValueError("Unsupported array dimensions in _extract_data")
    return data