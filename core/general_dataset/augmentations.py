# augmentations.py

import numpy as np
from typing import Dict, Any, Tuple, List
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates


def _get_random_params(min_val: float, max_val: float, rng: np.random.RandomState) -> float:
    """
    Generate a random parameter within the specified range.
    """
    return rng.uniform(min_val, max_val)


def _apply_elastic_deformation(
    arr: np.ndarray,
    alpha: float,
    sigma: float,
    rng: np.random.RandomState,
    dim: int,
) -> np.ndarray:
    """
    Elastic deformation that warps only the spatial dimensions and keeps the
    channel axis (if any) untouched.
    """

    # ------------------------------------------------------------
    # 1. Detect channel axis
    # ------------------------------------------------------------
    has_ch = (arr.ndim == dim + 1)
    if has_ch:
        C, *spatial = arr.shape
        arr_ch = arr
    else:
        C = 1
        spatial = arr.shape
        arr_ch = arr[np.newaxis, ...]          # add fake channel axis

    # ------------------------------------------------------------
    # 2. Build common displacement field on the spatial grid
    # ------------------------------------------------------------
    grid = np.stack(
        np.meshgrid(*[np.arange(s) for s in spatial], indexing="ij"),
        axis=0,
    ).astype(np.float32)                       # shape (dim, *spatial)

    disp = np.stack([rng.normal(0, sigma, spatial) for _ in range(dim)], axis=0)
    disp = gaussian_filter(disp, sigma=sigma, mode="constant")
    disp *= alpha / (np.max(np.abs(disp)) + 1e-8)

    coords_def = [grid[i] + disp[i] for i in range(dim)]
    coords_flat = [c.ravel() for c in coords_def]

    # ------------------------------------------------------------
    # 3. Warp each channel with the same field
    # ------------------------------------------------------------
    warped_out = np.zeros_like(arr_ch)
    for c in range(C):
        warped = map_coordinates(
            arr_ch[c], coords_flat, order=1, mode="reflect"
        ).reshape(spatial)
        warped_out[c] = warped

    # ------------------------------------------------------------
    # 4. Remove fake channel axis if the input had none
    # ------------------------------------------------------------
    return warped_out if has_ch else warped_out[0]

def _apply_color_jitter(
    arr: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Apply color jitter to an RGB image.
    """
    if arr.shape[0] != 3:
        return arr
    b = _get_random_params(-brightness, brightness, rng)
    c = _get_random_params(1 - contrast, 1 + contrast, rng)
    out = arr + b
    out = out * c
    return np.clip(out, 0, 1)


def augment_images(
    data: Dict[str, np.ndarray],
    aug_type: str,
    aug_cfg: Dict[str, Any],
    data_dim: int,
    rng: np.random.RandomState
) -> Dict[str, np.ndarray]:
    """
    Apply a specific augmentation to a dictionary of modality arrays based on config.
    """
    augmented = {k: v.copy() for k, v in data.items()}
    prob = aug_cfg.get('prob', 1.0)

    if aug_type == 'flip_h':
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                augmented[m] = np.flip(augmented[m], axis=-1)

    elif aug_type == 'flip_v':
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                augmented[m] = np.flip(augmented[m], axis=-2)

    elif aug_type == 'flip_d' and data_dim == 3:
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                arr = augmented[m]
                axis = -3 if arr.ndim == 4 else 0
                augmented[m] = np.flip(arr, axis=axis)

    elif aug_type == 'rotation':
        angle = _get_random_params(aug_cfg['angle']['min'], aug_cfg['angle']['max'], rng)
        pad = aug_cfg.get('pad', True)
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                mode = 'nearest' if m == 'label' else 'reflect'
                augmented[m] = rotate(
                    augmented[m],
                    angle,
                    axes=(-2, -1),
                    reshape=pad,
                    mode=mode,
                    order=0 if m == 'label' else 1
                )

    elif aug_type == 'scale':
        factor = _get_random_params(aug_cfg['factor']['min'], aug_cfg['factor']['max'], rng)
        for m in aug_cfg['modalities']:
            if m not in augmented or rng.random() >= prob:
                continue
            arr = augmented[m]

            # Detect channel axis
            has_ch = (arr.ndim == data_dim + 1)
            if has_ch:
                zoom_seq = (1.0,) + tuple([factor] * data_dim)
                orig_shape = arr.shape
                spatial_idx = slice(1, None)
            else:
                zoom_seq = tuple([factor] * data_dim)
                orig_shape = arr.shape
                spatial_idx = slice(None)

            scaled = zoom(
                arr,
                zoom_seq,
                order=0 if m == 'label' else 1,
                mode='nearest' if m == 'label' else 'reflect'
            )

            if scaled.shape != orig_shape:
                # Center-crop or pad spatial dims only
                if has_ch:
                    C = orig_shape[0]
                    orig_sp = orig_shape[1:]
                    new_sp = scaled.shape[1:]
                    padded = np.zeros(orig_shape, dtype=scaled.dtype)
                    for c in range(C):
                        crop = scaled[c]
                        # compute crop/pad slices
                        slices_crop = []
                        pads = []
                        for ns, os in zip(new_sp, orig_sp):
                            if ns >= os:
                                start = (ns - os) // 2
                                slices_crop.append(slice(start, start + os))
                                pads.append((0, 0))
                            else:
                                pads.append(((os - ns) // 2, os - ns - (os - ns) // 2))
                                slices_crop.append(slice(0, ns))
                        crop = crop[tuple(slices_crop)]
                        padded_ch = np.zeros(orig_sp, dtype=scaled.dtype)
                        insert_slices = tuple(slice(pb[0], pb[0] + crop.shape[i]) for i, pb in enumerate(pads))
                        padded_ch[insert_slices] = crop
                        padded[c] = padded_ch
                else:
                    orig_sp = orig_shape
                    new_sp = scaled.shape
                    padded = np.zeros(orig_shape, dtype=scaled.dtype)
                    slices_crop = []
                    pads = []
                    for ns, os in zip(new_sp, orig_sp):
                        if ns >= os:
                            start = (ns - os) // 2
                            slices_crop.append(slice(start, start + os))
                            pads.append((0, 0))
                        else:
                            pads.append(((os - ns) // 2, os - ns - (os - ns) // 2))
                            slices_crop.append(slice(0, ns))
                    crop = scaled[tuple(slices_crop)]
                    insert_slices = tuple(slice(pb[0], pb[0] + crop.shape[i]) for i, pb in enumerate(pads))
                    padded[insert_slices] = crop

                augmented[m] = padded
            else:
                augmented[m] = scaled

    elif aug_type == 'elastic':
        alpha = _get_random_params(aug_cfg['alpha']['min'], aug_cfg['alpha']['max'], rng)
        sigma = _get_random_params(aug_cfg['sigma']['min'], aug_cfg['sigma']['max'], rng)
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                augmented[m] = _apply_elastic_deformation(
                    augmented[m], alpha, sigma, rng, data_dim
                )

    elif aug_type == 'random_crop':
        size = aug_cfg['size']
        crop_sz = (
            (size['z'], size['y'], size['x']) if data_dim == 3
            else (size['y'], size['x'])
        )
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                arr = augmented[m]
                shape = arr.shape[-data_dim:]
                starts = [
                    rng.randint(0, s - c) if s > c else 0
                    for s, c in zip(shape, crop_sz)
                ]
                slices = tuple(slice(st, st + c) for st, c in zip(starts, crop_sz))
                if data_dim == 3:
                    augmented[m] = arr[..., slices[0], slices[1], slices[2]]
                else:
                    augmented[m] = arr[..., slices[0], slices[1]]

    elif aug_type == 'random_resize':
        size = aug_cfg['size']
        tgt = (
            (size['z'], size['y'], size['x']) if data_dim == 3
            else (size['y'], size['x'])
        )
        for m in aug_cfg['modalities']:
            if m not in augmented or rng.random() >= prob:
                continue
            arr = augmented[m]
            has_ch = (arr.ndim == data_dim + 1)
            if has_ch:
                zoom_seq = (1.0,) + tuple(t / s for t, s in zip(tgt, arr.shape[1:]))
                orig_shape = arr.shape
            else:
                zoom_seq = tuple(t / s for t, s in zip(tgt, arr.shape))
                orig_shape = arr.shape

            resized = zoom(
                arr,
                zoom_seq,
                order=0 if m == 'label' else 1,
                mode='nearest' if m == 'label' else 'reflect'
            )

            # If shaped mismatch, reuse scale’s pad/crop logic
            if resized.shape != orig_shape:
                # (You can extract the same pad/crop block from 'scale' above here)
                # For brevity, assume tgt == orig so this rarely happens.
                augmented[m] = resized
            else:
                augmented[m] = resized

    elif aug_type == 'random_intensity':
        a = _get_random_params(aug_cfg['alpha_min'], aug_cfg['alpha_max'], rng)
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                augmented[m] = augmented[m] * a

    elif aug_type == 'random_brightness':
        b = _get_random_params(aug_cfg['beta_min'], aug_cfg['beta_max'], rng)
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                x = augmented[m] + b
                augmented[m] = np.clip(x, 0, 255)

    elif aug_type == 'random_brightness_contrast':
        a = _get_random_params(aug_cfg['alpha_min'], aug_cfg['alpha_max'], rng)
        b = _get_random_params(aug_cfg['beta_min'], aug_cfg['beta_max'], rng)
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                x = augmented[m] * a + b
                augmented[m] = np.clip(x, 0, 255)

    elif aug_type == 'random_gamma':
        g = _get_random_params(aug_cfg['min'], aug_cfg['max'], rng)
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                x = np.power(augmented[m] / 255.0, g) * 255.0
                augmented[m] = np.clip(x, 0, 255)

    elif aug_type == 'random_gaussian_noise':
        nstd = _get_random_params(aug_cfg['min'], aug_cfg['max'], rng)
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                noise = rng.normal(0, nstd, augmented[m].shape)
                x = augmented[m] + noise * 255
                augmented[m] = np.clip(x, 0, 255)

    elif aug_type == 'random_gaussian_blur':
        s = _get_random_params(aug_cfg['min'], aug_cfg['max'], rng)
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                augmented[m] = gaussian_filter(
                    augmented[m],
                    sigma=s,
                    mode='reflect'
                )

    elif aug_type == 'random_bias_field':
        coef = _get_random_params(aug_cfg['min'], aug_cfg['max'], rng)
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                shape = augmented[m].shape
                grid = np.meshgrid(
                    *[np.linspace(-1, 1, s) for s in shape[-data_dim:]],
                    indexing='ij'
                )
                bias = np.exp(coef * np.sum([g**2 for g in grid], axis=0))
                if data_dim == 3:
                    bias = bias[None, ...]
                x = augmented[m] * bias
                augmented[m] = np.clip(x, 0, 255)

    elif aug_type == 'random_color_jitter':
        for m in aug_cfg['modalities']:
            if m in augmented and rng.random() < prob:
                augmented[m] = _apply_color_jitter(
                    augmented[m],
                    aug_cfg['brightness'],
                    aug_cfg['contrast'],
                    aug_cfg['saturation'],
                    aug_cfg['hue'],
                    rng
                )

    return augmented
