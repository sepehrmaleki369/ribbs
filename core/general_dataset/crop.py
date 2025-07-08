import random
from typing import Dict, Any, Tuple, Callable
import numpy as np


def crop(
    data: Dict[str, np.ndarray],
    split: str,
    patch_size: int,
    data_dim: int,
    patch_size_z: int,
    pad_reflect: Callable[[np.ndarray, Tuple[int, ...], Tuple[int, ...]], np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Crop and pad modalities consistently for training, handling optional channel-first arrays.

    Args:
        data: dict of modality arrays (e.g., {'image': arr, 'label': arr}).
        split: dataset split name ('train', 'valid', 'test').
        patch_size: spatial size for height and width.
        data_dim: 2 for 2D, 3 for 3D.
        patch_size_z: depth for 3D; ignored if data_dim==2.
        pad_reflect: function to reflect-pad an array: pad_reflect(arr, pad_before, pad_after).

    Returns:
        Cropped (and possibly padded) data dict.
    """
    # Only crop during training
    if split != "train":
        return data

    # Infer spatial shape (drop channel if present)
    sample = next(iter(data.values()))
    has_channel = sample.ndim == data_dim + 1
    spatial_shape = sample.shape[-data_dim:]

    if data_dim == 2:
        H, W = spatial_shape
        ph = max(0, patch_size - H)
        pw = max(0, patch_size - W)

        # pad as needed along spatial dims, preserving channel axis
        if ph > 0 or pw > 0:
            for k, arr in data.items():
                # build pad widths: one tuple per axis
                if has_channel and arr.ndim == 3:
                    # (C, H, W)
                    pad_before = (0, ph // 2, pw // 2)
                    pad_after  = (0, ph - ph // 2, pw - pw // 2)
                else:
                    # (H, W)
                    pad_before = (ph // 2, pw // 2)
                    pad_after  = (ph - ph // 2, pw - pw // 2)
                data[k] = pad_reflect(arr, pad_before=pad_before, pad_after=pad_after)
            H += ph
            W += pw

        # choose random crop origin
        top  = random.randint(0, H - patch_size)
        left = random.randint(0, W - patch_size)

        # apply crop on spatial dims
        for k, arr in data.items():
            if has_channel and arr.ndim == 3:
                # arr shape (C, H, W)
                data[k] = arr[:, top:top + patch_size, left:left + patch_size]
            else:
                # arr shape (H, W)
                data[k] = arr[top:top + patch_size, left:left + patch_size]

    else:
        # 3D cropping
        D0, H0, W0 = spatial_shape
        patch_z = patch_size_z
        pd = max(0, patch_z - D0)
        ph = max(0, patch_size - H0)
        pw = max(0, patch_size - W0)

        if pd > 0 or ph > 0 or pw > 0:
            for k, arr in data.items():
                if has_channel and arr.ndim == 4:
                    # (C, D, H, W)
                    pad_before = (0, pd // 2, ph // 2, pw // 2)
                    pad_after  = (0, pd - pd // 2, ph - ph // 2, pw - pw // 2)
                else:
                    # (D, H, W)
                    pad_before = (pd // 2, ph // 2, pw // 2)
                    pad_after  = (pd - pd // 2, ph - ph // 2, pw - pw // 2)
                data[k] = pad_reflect(arr, pad_before=pad_before, pad_after=pad_after)
            D0 += pd
            H0 += ph
            W0 += pw

        # random 3D crop origin
        front = random.randint(0, D0 - patch_z)
        top   = random.randint(0, H0 - patch_size)
        left  = random.randint(0, W0 - patch_size)

        for k, arr in data.items():
            if has_channel and arr.ndim == 4:
                data[k] = arr[:,
                               front:front + patch_z,
                               top:top   + patch_size,
                               left:left + patch_size]
            else:
                data[k] = arr[
                    front:front + patch_z,
                    top:top   + patch_size,
                    left:left + patch_size
                ]

    return data
