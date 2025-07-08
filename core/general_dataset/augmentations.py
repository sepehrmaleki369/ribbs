# augmentations.py

import numpy as np
from typing import Dict, Any, Tuple, List
from scipy.ndimage import rotate, zoom, gaussian_filter
from scipy.interpolate import RegularGridInterpolator

def _get_random_params(min_val: float, max_val: float, rng: np.random.RandomState) -> float:
    """
    Generate a random parameter within the specified range.

    Args:
        min_val (float): Minimum value of the range.
        max_val (float): Maximum value of the range.
        rng (np.random.RandomState): Random number generator.

    Returns:
        float: Random value between min_val and max_val.
    """
    return rng.uniform(min_val, max_val)

def _apply_elastic_deformation(arr: np.ndarray, alpha: float, sigma: float, rng: np.random.RandomState, dim: int) -> np.ndarray:
    """
    Apply elastic deformation to an array.

    Args:
        arr (np.ndarray): Input array to deform.
        alpha (float): Magnitude of deformation.
        sigma (float): Smoothness of deformation.
        rng (np.random.RandomState): Random number generator.
        dim (int): Dimensionality of the data (2 or 3).

    Returns:
        np.ndarray: Deformed array.
    """
    shape = arr.shape
    grid = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    displacement = np.stack([rng.normal(0, sigma, shape) for _ in range(dim)], axis=0)
    displacement = gaussian_filter(displacement, sigma=sigma, mode='constant')
    displacement *= alpha / np.max(np.abs(displacement))
    
    coords = np.array(grid) + displacement
    shape_arr = np.array(shape).reshape((dim,) + (1,) * (coords.ndim - 1))
    coords = np.clip(coords, 0, shape_arr - 1)
    
    interpolator = RegularGridInterpolator(
        [np.arange(s) for s in shape],
        arr,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    flat_coords = np.array([coords[i].ravel() for i in range(dim)]).T
    return interpolator(flat_coords).reshape(shape)

def _apply_color_jitter(arr: np.ndarray, brightness: float, contrast: float, saturation: float, hue: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Apply color jitter to an RGB image.

    Args:
        arr (np.ndarray): Input RGB array (shape: [3, H, W]).
        brightness (float): Range for brightness adjustment.
        contrast (float): Range for contrast adjustment.
        saturation (float): Range for saturation adjustment (not implemented).
        hue (float): Range for hue adjustment (not implemented).
        rng (np.random.RandomState): Random number generator.

    Returns:
        np.ndarray: Adjusted RGB array.
    """
    if arr.shape[0] != 3:  # Assume RGB with channels first
        return arr
    b = _get_random_params(-brightness, brightness, rng)
    arr = arr + b
    c = _get_random_params(1 - contrast, 1 + contrast, rng)
    arr = arr * c
    # Saturation and hue adjustments could be added here if needed
    return np.clip(arr, 0, 1)

def augment_images(
    data: Dict[str, np.ndarray],
    aug_type: str,
    aug_cfg: Dict[str, Any],
    data_dim: int,
    rng: np.random.RandomState
) -> Dict[str, np.ndarray]:
    """
    Apply a specific augmentation to a dictionary of modality arrays based on config.

    Args:
        data (Dict[str, np.ndarray]): Dictionary of modality arrays (e.g., {'image': arr, 'label': arr}).
        aug_type (str): Type of augmentation to apply (e.g., 'flip_h', 'rotation').
        aug_cfg (Dict[str, Any]): Configuration for the specific augmentation type.
        data_dim (int): Dimensionality of the data (2 for 2D, 3 for 3D).
        rng (np.random.RandomState): Random number generator for reproducibility.

    Returns:
        Dict[str, np.ndarray]: Dictionary of augmented arrays.
    """
    augmented = {k: arr.copy() for k, arr in data.items()}
    params = aug_cfg
    prob = params.get('prob', 1.0)

    if aug_type == 'flip_h':
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                augmented[modality] = np.flip(augmented[modality], axis=-1)

    elif aug_type == 'flip_v':
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                augmented[modality] = np.flip(augmented[modality], axis=-2)

    elif aug_type == 'flip_d' and data_dim == 3:
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                augmented[modality] = np.flip(augmented[modality], axis=0)

    elif aug_type == 'rotation':
        angle = _get_random_params(params['angle']['min'], params['angle']['max'], rng)
        pad = params.get('pad', True)
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                mode = 'nearest' if modality == 'label' else 'reflect'
                augmented[modality] = rotate(
                    augmented[modality],
                    angle,
                    axes=(-2, -1),
                    reshape=pad,
                    mode=mode,
                    order=0 if modality == 'label' else 1
                )

    elif aug_type == 'scale':
        factor = _get_random_params(params['factor']['min'], params['factor']['max'], rng)

        def make_zoom_sequence(arr: np.ndarray) -> List[float]:
            # If arr has no channel axis (ndim == data_dim), zoom each spatial dim
            # If channel-first (ndim == data_dim+1), leave channel untouched
            if arr.ndim == data_dim:
                return [factor] * data_dim
            elif arr.ndim == data_dim + 1:
                return [1.0] + [factor] * data_dim
            else:
                raise RuntimeError(f"Unexpected array ndim={arr.ndim} in scale")

        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                arr = augmented[modality]
                zoom_seq = make_zoom_sequence(arr)
                order = 0 if modality == 'label' else 1
                scaled = zoom(
                    arr,
                    zoom_seq,
                    order=order,
                    mode='nearest' if modality == 'label' else 'reflect'
                )
                # Crop or pad back to original shape
                orig_shape = data[modality].shape
                new_shape = scaled.shape
                if new_shape != orig_shape:
                    # compute the slices in scaled to extract or pad around center
                    extract_slices = []
                    pad_before = []
                    pad_after  = []
                    for new_s, orig_s in zip(new_shape, orig_shape):
                        if new_s >= orig_s:
                            start = (new_s - orig_s) // 2
                            extract_slices.append(slice(start, start + orig_s))
                            pad_before.append(0)
                            pad_after.append(0)
                        else:
                            extract_slices.append(slice(0, new_s))
                            pad_before.append((orig_s - new_s) // 2)
                            pad_after.append(orig_s - new_s - pad_before[-1])
                    # extract then pad
                    cropped = scaled[tuple(extract_slices)]
                    padded = np.zeros(orig_shape, dtype=scaled.dtype)
                    insert_slices = tuple(
                        slice(pb, pb + cropped.shape[i])
                        for i, pb in enumerate(pad_before)
                    )
                    padded[insert_slices] = cropped
                    augmented[modality] = padded
                else:
                    augmented[modality] = scaled

    elif aug_type == 'elastic':
        alpha = _get_random_params(params['alpha']['min'], params['alpha']['max'], rng)
        sigma = _get_random_params(params['sigma']['min'], params['sigma']['max'], rng)
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                augmented[modality] = _apply_elastic_deformation(
                    augmented[modality], alpha, sigma, rng, data_dim
                )

    elif aug_type == 'random_crop':
        crop_size = (
            (params['size']['z'], params['size']['y'], params['size']['x'])
            if data_dim == 3 else (params['size']['y'], params['size']['x'])
        )
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                shape = augmented[modality].shape
                start = tuple(
                    rng.randint(0, s - cs) if s > cs else 0
                    for s, cs in zip(shape[-data_dim:], crop_size)
                )
                slices = tuple(slice(st, st + cs) for st, cs in zip(start, crop_size))
                if data_dim == 3:
                    augmented[modality] = augmented[modality][..., slices[0], slices[1], slices[2]]
                else:
                    augmented[modality] = augmented[modality][..., slices[0], slices[1]]

    elif aug_type == 'random_resize':
        target_size = (
            (params['size']['z'], params['size']['y'], params['size']['x'])
            if data_dim == 3 else (params['size']['y'], params['size']['x'])
        )

        def make_resize_sequence(arr: np.ndarray) -> Tuple[float, ...]:
            # spatial zoom factors
            spatial = arr.shape[-data_dim:]
            factors = tuple(t / s for t, s in zip(target_size, spatial))
            if arr.ndim == data_dim:
                return factors
            elif arr.ndim == data_dim + 1:
                return (1.0, *factors)
            else:
                raise RuntimeError(f"Unexpected array ndim={arr.ndim} in random_resize")

        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                arr = augmented[modality]
                resize_seq = make_resize_sequence(arr)
                order = 0 if modality == 'label' else 1
                resized = zoom(
                    arr,
                    resize_seq,
                    order=order,
                    mode='nearest' if modality == 'label' else 'reflect'
                )
                augmented[modality] = resized


    elif aug_type == 'random_intensity':
        alpha = _get_random_params(params['alpha_min'], params['alpha_max'], rng)
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                augmented[modality] = augmented[modality] * alpha

    elif aug_type == 'random_brightness':
        beta = _get_random_params(params['beta_min'], params['beta_max'], rng)
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                augmented[modality] = augmented[modality] + beta
                augmented[modality] = np.clip(augmented[modality], 0, 255)

    elif aug_type == 'random_brightness_contrast':
        alpha = _get_random_params(params['alpha_min'], params['alpha_max'], rng)
        beta = _get_random_params(params['beta_min'], params['beta_max'], rng)
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                augmented[modality] = augmented[modality] * alpha + beta
                augmented[modality] = np.clip(augmented[modality], 0, 255)

    elif aug_type == 'random_gamma':
        gamma = _get_random_params(params['min'], params['max'], rng)
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                augmented[modality] = np.power(augmented[modality] / 255.0, gamma) * 255.0
                augmented[modality] = np.clip(augmented[modality], 0, 255)

    elif aug_type == 'random_gaussian_noise':
        noise_std = _get_random_params(params['min'], params['max'], rng)
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                noise = rng.normal(0, noise_std, augmented[modality].shape)
                augmented[modality] = augmented[modality] + noise * 255
                augmented[modality] = np.clip(augmented[modality], 0, 255)

    elif aug_type == 'random_gaussian_blur':
        sigma = _get_random_params(params['min'], params['max'], rng)
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                augmented[modality] = gaussian_filter(
                    augmented[modality],
                    sigma=sigma,
                    mode='reflect'
                )

    elif aug_type == 'random_bias_field':
        coef = _get_random_params(params['min'], params['max'], rng)
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                shape = augmented[modality].shape
                grid = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape[-data_dim:]], indexing='ij')
                bias = np.exp(coef * np.sum([g**2 for g in grid], axis=0))
                if data_dim == 3:
                    bias = bias[None, ...]
                augmented[modality] = augmented[modality] * bias
                augmented[modality] = np.clip(augmented[modality], 0, 255)

    elif aug_type == 'random_color_jitter':
        for modality in params['modalities']:
            if modality in augmented and rng.random() < prob:
                augmented[modality] = _apply_color_jitter(
                    augmented[modality],
                    params['brightness'],
                    params['contrast'],
                    params['saturation'],
                    params['hue'],
                    rng
                )

    return augmented