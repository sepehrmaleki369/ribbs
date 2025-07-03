
from typing import Any, Dict, List, Optional
import numpy as np
from core.general_dataset.logger import logger

import math
from scipy.ndimage import rotate

def get_augmentation_metadata(augmentations: List[str], data_dim: int) -> Dict[str, Any]:
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
    if 'flip_d' in augmentations and data_dim == 3:
        meta['flip_d'] = np.random.rand() > 0.5
    return meta


def flip_h(full_array: np.ndarray) -> np.ndarray:
    return np.flip(full_array, axis=-1)

def flip_v(full_array: np.ndarray) -> np.ndarray:
    return np.flip(full_array, axis=-2)

def flip_d(full_array: np.ndarray) -> np.ndarray:
    return np.flip(full_array, axis=-3)

def rotate_(
    full_array: np.ndarray,
    patch_meta: Dict[str, Any],
    patch_size_xy: int,
    patch_size_z: int,
    data_dim: int
) -> np.ndarray:
    """Rotate patch, preserving shape, for 2D or 3D."""
    angle = patch_meta['angle']
    # Compute L only in-plane
    L = int(np.ceil(patch_size_xy * math.sqrt(2)))
    x, y = patch_meta['x'], patch_meta['y']
    # for 3D also get z but we only rotate each slice independently
    cx, cy = x + patch_size_xy // 2, y + patch_size_xy // 2
    half_L = L // 2
    x0, x1 = max(0, cx-half_L), min(full_array.shape[-1], cx+half_L)
    y0, y1 = max(0, cy-half_L), min(full_array.shape[-2], cy+half_L)

    if data_dim == 3:
        z = patch_meta.get('z', 0)
        # crop a 3D block, but rotate per-slice
        block = full_array[:, z:z+patch_size_z, y0:y1, x0:x1]
        D, Hc, Wc = block.shape[1:]
        if Hc < L or Wc < L:
            logger.warning("Crop too small for 3D rotation; zero patch.")
            return np.zeros_like(block[..., :patch_size_xy, :patch_size_xy])
        rotated_slices = []
        for d in range(block.shape[1]):
            # rotate each C×H×W slice
            slice_ = block[:, d]
            rotated = rotate(slice_, angle, reshape=False, order=1)
            # center-crop back to patch_size_xy
            start = (L - patch_size_xy)//2
            cropped = rotated[start:start+patch_size_xy, start:start+patch_size_xy]
            rotated_slices.append(cropped)
        return np.stack(rotated_slices, axis=1)  # C×Z×XY×XY

    else:
        # 2D case: full_array is H×W or C×H×W
        crop = full_array[y0:y1, x0:x1] if full_array.ndim == 2 else full_array[:, y0:y1, x0:x1]
        Hc, Wc = crop.shape[-2], crop.shape[-1]
        if Hc < L or Wc < L:
            logger.warning("Crop too small for 2D rotation; zero patch.")
            shape = (patch_size_xy, patch_size_xy) if crop.ndim == 2 else (crop.shape[0], patch_size_xy, patch_size_xy)
            return np.zeros(shape, dtype=crop.dtype)
        rotated = rotate(crop, angle, reshape=False, order=1)
        start = (L - patch_size_xy)//2
        if rotated.ndim == 2:
            return rotated[start:start+patch_size_xy, start:start+patch_size_xy]
        else:
            return rotated[:, start:start+patch_size_xy, start:start+patch_size_xy]


def extract_condition_augmentations(
    imgs: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    patch_size_xy: int,
    patch_size_z: int,
    augmentations: List[str],
    data_dim: int
) -> Dict[str, np.ndarray]:
    """
    Extract a patch from the full image and apply conditional augmentations.

    Args:
        imgs (Dict[str, np.ndarray]): Full images for each modality.
        metadata (Dict[str, Any]): Metadata containing patch coordinates and augmentations.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary of extracted patches.
    """
    imgs_aug = imgs.copy()
    z = metadata.get('z', 0)
    data = extract_data(imgs,
                        metadata['x'], metadata['y'], z,
                        patch_size_xy,
                        patch_size_z)
    for key in imgs:
        if key.endswith("_patch"):
            modality = key.replace("_patch", "")
            if 'flip_h' in augmentations:
                imgs_aug[modality] = flip_h(imgs[modality])
                data[key] = flip_h(data[key])
            if 'flip_v' in augmentations:
                imgs_aug[modality] = flip_v(imgs[modality])
                data[key] = flip_v(data[key])
            if 'flip_d' in augmentations and data_dim == 3:
                imgs_aug[modality] = flip_d(imgs_aug[modality])
                data[key]          = flip_d(data[key])
            if 'rotation' in augmentations:
                data[key] = rotate_(imgs_aug[modality], metadata, patch_size_xy, patch_size_z, data_dim)
    return data

def extract_data(imgs: Dict[str, np.ndarray],
                 x: int, y: int, z: int,
                 patch_size_xy: int,
                 patch_size_z: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    If patch_size_z is given, extract a 3-D block; otherwise a 2-D tile.
    """
    data = {}
    for key, arr in imgs.items():
        if arr.ndim == 4:  # C, D, H, W
            psz = patch_size_z or patch_size_xy
            data[f"{key}_patch"] = arr[:, z:z+psz, y:y+patch_size_xy, x:x+patch_size_xy]
        elif arr.ndim == 3:
            if patch_size_z is not None:
                data[f"{key}_patch"] = arr[z:z+patch_size_z, y:y+patch_size_xy, x:x+patch_size_xy]
            else:
                data[f"{key}_patch"] = arr[y:y+patch_size_xy, x:x+patch_size_xy]
        elif arr.ndim == 2:
            data[f"{key}_patch"] = arr[y:y+patch_size_xy, x:x+patch_size_xy]
        else:
            raise ValueError("Unsupported array dims")
    return data
