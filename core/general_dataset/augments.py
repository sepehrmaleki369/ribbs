# ------------------------------------
# augments.py  (complete version – supports 2‑D & 3‑D)
# ------------------------------------

from typing import Any, Dict, List, Optional
import math
import numpy as np
from scipy.ndimage import rotate, zoom as ndi_zoom, gaussian_filter, map_coordinates

from core.general_dataset.logger import logger

# -----------------------------------------------------------------------------
# Random‑parameter generator
# -----------------------------------------------------------------------------

def get_augmentation_metadata(
    augmentations: List[str],
    data_dim: int,
    aug_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Return sampled parameters based on `augmentation_params` from config."""
    params = aug_params or {}
    meta: Dict[str, Any] = {}

    # rotation
    if "rotation" in augmentations:
        rmin = params.get("rotation", {}).get("min", 0.0)
        rmax = params.get("rotation", {}).get("max", 360.0)
        meta["angle"] = float(np.random.uniform(rmin, rmax))

    # flips (no change)
    if "flip_h" in augmentations:
        meta["flip_h"] = bool(np.random.rand() > 0.5)
    if "flip_v" in augmentations:
        meta["flip_v"] = bool(np.random.rand() > 0.5)
    if "flip_d" in augmentations and data_dim == 3:
        meta["flip_d"] = bool(np.random.rand() > 0.5)

    # scale
    if "scale" in augmentations:
        smin = params.get("scale", {}).get("min", 0.8)
        smax = params.get("scale", {}).get("max", 1.2)
        meta["scale_factor"] = float(np.random.uniform(smin, smax))

    # elastic
    if "elastic" in augmentations:
        e = params.get("elastic", {})
        a_min, a_max = e.get("alpha_min",5.0), e.get("alpha_max",10.0)
        s_min, s_max = e.get("sigma_min",3.0), e.get("sigma_max",6.0)
        meta["alpha"] = float(np.random.uniform(a_min, a_max))
        meta["sigma"] = float(np.random.uniform(s_min, s_max))

    # brightness_contrast
    if "brightness_contrast" in augmentations:
        bc = params.get("brightness_contrast", {})
        ca, cb = bc.get("alpha_min",0.9), bc.get("alpha_max",1.1)
        ba, bb = bc.get("beta_min",-30.0), bc.get("beta_max",30.0)
        meta["bc_alpha"] = float(np.random.uniform(ca, cb))
        meta["bc_beta"]  = float(np.random.uniform(ba, bb))

    # gamma
    if "gamma" in augmentations:
        gmin = params.get("gamma", {}).get("min",0.7)
        gmax = params.get("gamma", {}).get("max",1.5)
        meta["gamma"] = float(np.random.uniform(gmin, gmax))

    # gaussian_noise
    if "gaussian_noise" in augmentations:
        n = params.get("gaussian_noise", {})
        meta["noise_sigma"] = float(np.random.uniform(n.get("min",0.01), n.get("max",0.03)))

    # gaussian_blur
    if "gaussian_blur" in augmentations:
        gb = params.get("gaussian_blur", {})
        meta["blur_sigma"] = float(np.random.uniform(gb.get("min",0.5), gb.get("max",1.5)))

    # bias_field
    if "bias_field" in augmentations:
        bf = params.get("bias_field", {})
        meta["bias_amp"] = float(np.random.uniform(bf.get("min",0.2), bf.get("max",0.4)))

    return meta

# -----------------------------------------------------------------------------
# Basic flip helpers (unchanged)
# -----------------------------------------------------------------------------

def flip_h(arr: np.ndarray) -> np.ndarray:
    """Flip left/right (axis = −1)."""
    return np.flip(arr, axis=-1)


def flip_v(arr: np.ndarray) -> np.ndarray:
    """Flip up/down (axis = −2)."""
    return np.flip(arr, axis=-2)


def flip_d(arr: np.ndarray) -> np.ndarray:
    """Flip depth (axis = −3). Only meaningful for 3‑D volumes."""
    return np.flip(arr, axis=-3)

# -----------------------------------------------------------------------------
# **DO NOT CHANGE** – original rotation implementation
# -----------------------------------------------------------------------------

def rotate_(
    full_array: np.ndarray,
    patch_meta: Dict[str, Any],
    patch_size_xy: int,
    patch_size_z: int,
    data_dim: int,
) -> np.ndarray:
    """Rotate patch, preserving shape, for 2‑D or 3‑D (slice‑wise)."""
    angle = patch_meta["angle"]
    L = int(np.ceil(patch_size_xy * math.sqrt(2)))
    x, y = patch_meta["x"], patch_meta["y"]
    cx, cy = x + patch_size_xy // 2, y + patch_size_xy // 2
    half_L = L // 2
    x0, x1 = max(0, cx - half_L), min(full_array.shape[-1], cx + half_L)
    y0, y1 = max(0, cy - half_L), min(full_array.shape[-2], cy + half_L)

    if data_dim == 3:
        z = patch_meta.get("z", 0)
        block = full_array[:, z : z + patch_size_z, y0:y1, x0:x1]
        if block.shape[-2] < L or block.shape[-1] < L:
            logger.warning("Crop too small for 3D rotation; zero patch.")
            return np.zeros_like(block[..., :patch_size_xy, :patch_size_xy])
        rotated_slices = []
        for d in range(block.shape[1]):
            slice_ = block[:, d]
            rotated = rotate(slice_, angle, reshape=False, order=1)
            start = (L - patch_size_xy) // 2
            cropped = rotated[start : start + patch_size_xy, start : start + patch_size_xy]
            rotated_slices.append(cropped)
        return np.stack(rotated_slices, axis=1)
    else:
        crop = (
            full_array[y0:y1, x0:x1]
            if full_array.ndim == 2
            else full_array[:, y0:y1, x0:x1]
        )
        if crop.shape[-2] < L or crop.shape[-1] < L:
            logger.warning("Crop too small for 2D rotation; zero patch.")
            shp = (patch_size_xy, patch_size_xy) if crop.ndim == 2 else (crop.shape[0], patch_size_xy, patch_size_xy)
            return np.zeros(shp, dtype=crop.dtype)
        rotated = rotate(crop, angle, reshape=False, order=1)
        start = (L - patch_size_xy) // 2
        if rotated.ndim == 2:
            return rotated[start : start + patch_size_xy, start : start + patch_size_xy]
        else:
            return rotated[:, start : start + patch_size_xy, start : start + patch_size_xy]

# -----------------------------------------------------------------------------
# New spatial helpers
# -----------------------------------------------------------------------------

def _center_crop_or_pad(arr: np.ndarray, out_shape: tuple) -> np.ndarray:
    """Return `arr` centered and cropped/padded to `out_shape`. Zeros are used for padding."""
    result = np.zeros(out_shape, dtype=arr.dtype)
    min_shape = tuple(min(s, o) for s, o in zip(arr.shape, out_shape))
    # slice positions in input and output
    in_starts = [(s - m) // 2 for s, m in zip(arr.shape, min_shape)]
    out_starts = [(o - m) // 2 for o, m in zip(out_shape, min_shape)]
    slices_in = tuple(slice(i, i + m) for i, m in zip(in_starts, min_shape))
    slices_out = tuple(slice(o, o + m) for o, m in zip(out_starts, min_shape))
    result[slices_out] = arr[slices_in]
    return result


def scale_patch(patch: np.ndarray, scale_factor: float, data_dim: int) -> np.ndarray:
    """Isotropic zoom of the *patch* followed by center‑crop/zero‑pad to original size."""
    orig_shape = patch.shape

    if data_dim == 3:
        zoom_factors = (1, scale_factor, scale_factor, scale_factor) if patch.ndim == 4 else (scale_factor, scale_factor, scale_factor)
    else:
        zoom_factors = (1, scale_factor, scale_factor) if patch.ndim == 3 else (scale_factor, scale_factor)

    scaled = ndi_zoom(patch, zoom_factors, order=1)
    return _center_crop_or_pad(scaled, orig_shape)


# -----------------------------------------------------------------------------
# Elastic deformation  (B‑spline‑like random field)
# -----------------------------------------------------------------------------

def elastic_deform_patch(patch: np.ndarray, alpha: float, sigma: float, data_dim: int) -> np.ndarray:
    """Apply elastic deformation to `patch` and return the deformed patch (same shape)."""

    if data_dim == 3:
        spatial_shape = patch.shape[-3:] if patch.ndim == 4 else patch.shape
        dz = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect") * alpha
        dy = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect") * alpha
        dx = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect") * alpha
        z, y, x = np.meshgrid(
            np.arange(spatial_shape[0]),
            np.arange(spatial_shape[1]),
            np.arange(spatial_shape[2]),
            indexing="ij",
        )
        indices = (z + dz, y + dy, x + dx)

        if patch.ndim == 4:  # C,Z,H,W
            deformed = [map_coordinates(patch[c], indices, order=1, mode="reflect") for c in range(patch.shape[0])]
            return np.stack(deformed, axis=0)
        else:
            return map_coordinates(patch, indices, order=1, mode="reflect")

    else:  # 2‑D
        spatial_shape = patch.shape[-2:] if patch.ndim == 3 else patch.shape
        dy = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect") * alpha
        dx = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect") * alpha
        y, x = np.meshgrid(
            np.arange(spatial_shape[0]),
            np.arange(spatial_shape[1]),
            indexing="ij",
        )
        indices = (y + dy, x + dx)

        if patch.ndim == 3:  # C,H,W
            deformed = [map_coordinates(patch[c], indices, order=1, mode="reflect") for c in range(patch.shape[0])]
            return np.stack(deformed, axis=0)
        else:
            return map_coordinates(patch, indices, order=1, mode="reflect")

# -----------------------------------------------------------------------------
# Intensity helpers
# -----------------------------------------------------------------------------

def adjust_brightness_contrast(patch: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Linear brightness/contrast: I' = alpha·I + beta"""
    return patch.astype(np.float32) * alpha + beta


def gamma_correction(patch: np.ndarray, gamma: float) -> np.ndarray:
    patch_f = patch.astype(np.float32)
    max_val = patch_f.max() if patch_f.max() > 0 else 1.0
    norm = np.clip(patch_f / max_val, 0.0, 1.0)
    return (norm ** gamma) * max_val


def add_gaussian_noise(patch: np.ndarray, sigma_factor: float) -> np.ndarray:
    patch_f = patch.astype(np.float32)
    dyn = patch_f.max() - patch_f.min()
    sigma = sigma_factor * (dyn if dyn > 0 else 1.0)
    noise = np.random.normal(0.0, sigma, patch_f.shape)
    return patch_f + noise


def gaussian_blur_patch(patch: np.ndarray, sigma: float, data_dim: int) -> np.ndarray:
    """Apply an isotropic Gaussian blur (σ in voxels). Preserves shape."""
    if data_dim == 3:
        if patch.ndim == 4:  # C,Z,H,W
            blurred = [gaussian_filter(patch[c], sigma, mode="reflect") for c in range(patch.shape[0])]
            return np.stack(blurred, axis=0)
        else:
            return gaussian_filter(patch, sigma, mode="reflect")
    else:
        if patch.ndim == 3:  # C,H,W
            blurred = [gaussian_filter(patch[c], sigma, mode="reflect") for c in range(patch.shape[0])]
            return np.stack(blurred, axis=0)
        else:
            return gaussian_filter(patch, sigma, mode="reflect")


def bias_field_patch(patch: np.ndarray, amplitude: float, data_dim: int) -> np.ndarray:
    """Multiply `patch` by a smooth, low‑frequency bias field."""
    if data_dim == 3:
        spatial_shape = patch.shape[-3:] if patch.ndim == 4 else patch.shape
    else:
        spatial_shape = patch.shape[-2:] if patch.ndim == 3 else patch.shape

    sigma = 0.25 * min(spatial_shape)  # fairly low‑freq
    field = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect")
    field -= field.mean()
    field /= np.max(np.abs(field)) + 1e-8  # → ∈ [‑1,1]
    field *= amplitude
    field = 1.0 + field  # multiplicative  (1±amp)

    if patch.ndim == len(spatial_shape):  # no channel dim
        return patch * field
    else:  # prepend channel axis broadcast
        return patch * field[np.newaxis, ...]

# -----------------------------------------------------------------------------
# Patch extraction + conditional augmentation
# -----------------------------------------------------------------------------

def extract_data(
    imgs: Dict[str, np.ndarray],
    x: int,
    y: int,
    z: int,
    patch_size_xy: int,
    patch_size_z: Optional[int],
    data_dim: int,
) -> Dict[str, np.ndarray]:
    """
    Slice a spatial patch from every modality in *imgs*.

    Parameters
    ----------
    imgs : Dict[str, np.ndarray]
        Each value is one of the following shapes
          ─ data_dim == 2 →  (H, W)  or  (C, H, W)
          ─ data_dim == 3 →  (D, H, W)  or  (C, D, H, W)
    x, y : int
        Upper-left corner of the XY window.
    z : int
        First slice index along the depth axis (ignored for 2-D).
    patch_size_xy : int
        Height and width of the square crop.
    patch_size_z : Optional[int]
        If given in 3-D mode, take this many slices; otherwise fall back to
        *patch_size_xy*.
    data_dim : int
        2 or 3 – tells the function how to interpret array ranks.

    Returns
    -------
    Dict[str, np.ndarray]
        For every key in *imgs* a new entry ``f"{key}_patch"`` containing the
        cropped patch (shape preserved except for the spatial dims).
    """
    data: Dict[str, np.ndarray] = {}
    psz_z = patch_size_z or patch_size_xy

    for key, arr in imgs.items():

        # ───────────────────────────── 3-D CASE ───────────────────────────── #
        if data_dim == 3:
            if arr.ndim == 4:                             # (C, D, H, W)
                patch = arr[:, z : z + psz_z,
                               y : y + patch_size_xy,
                               x : x + patch_size_xy]
            elif arr.ndim == 3:                           # (D, H, W)
                patch = arr[z : z + psz_z,
                             y : y + patch_size_xy,
                             x : x + patch_size_xy]
            else:
                raise ValueError(f"{key}: expected 3-D or 4-D array, got shape {arr.shape}")

        # ───────────────────────────── 2-D CASE ───────────────────────────── #
        elif data_dim == 2:
            if arr.ndim == 3:                             # (C, H, W)
                patch = arr[:, y : y + patch_size_xy,
                               x : x + patch_size_xy]
            elif arr.ndim == 2:                           # (H, W)
                patch = arr[y : y + patch_size_xy,
                             x : x + patch_size_xy]
            else:
                raise ValueError(f"{key}: expected 2-D or 3-D array, got shape {arr.shape}")

        else:
            raise ValueError(f"data_dim must be 2 or 3, got {data_dim}")

        data[f"{key}_patch"] = patch

    return data



# -----------------------------------------------------------------------------
# Main entry: extract + apply *conditional* augmentations
# -----------------------------------------------------------------------------

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
                        patch_size_z,
                        data_dim)
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
#######
            if "scale" in augmentations:
                sf: float = metadata.get("scale_factor")
                imgs_aug[modality] = scale_patch(imgs_aug[modality], sf, data_dim)
                patch = scale_patch(patch, sf, data_dim)

            if "elastic" in augmentations:
                alpha: float = metadata.get("alpha")
                sigma: float = metadata.get("sigma")
                imgs_aug[modality] = elastic_deform_patch(imgs_aug[modality], alpha, sigma, data_dim)
                patch = elastic_deform_patch(patch, alpha, sigma, data_dim)

            # -----------------------------
            # Intensity augmentations (patch‑only)
            # -----------------------------
            if "brightness_contrast" in augmentations:
                patch = adjust_brightness_contrast(
                    patch,
                    metadata.get("bc_alpha"),
                    metadata.get("bc_beta"),
                )
            if "gamma" in augmentations:
                patch = gamma_correction(patch, metadata.get("gamma"))
            if "gaussian_noise" in augmentations:
                patch = add_gaussian_noise(patch, metadata.get("noise_sigma"))
            if "gaussian_blur" in augmentations:
                patch = gaussian_blur_patch(patch, metadata.get("blur_sigma"), data_dim)
            if "bias_field" in augmentations:
                patch = bias_field_patch(patch, metadata.get("bias_amp"), data_dim)

            if 'rotation' in augmentations:
                data[key] = rotate_(imgs_aug[modality], metadata, patch_size_xy, patch_size_z, data_dim)

            # Store the fully augmented patch back
            data[key] = patch    
    
    return data


# -----------------------------------------------------------------------------
# Public re‑exports
# -----------------------------------------------------------------------------

__all__ = [
    "get_augmentation_metadata",
    "extract_condition_augmentations",
    "flip_h",
    "flip_v",
    "flip_d",
    "rotate_",  # unchanged
]

# "augmentations": [
#         "flip_h",              # horizontal flip
#         "flip_v",              # vertical flip
#         "flip_d",              # depth flip (3D only)
#         "rotation",            # 0–360° random rotation
#         "scale",               # isotropic zoom in/out
#         "elastic",             # B-spline–style elastic warp
#         "brightness_contrast", # random α·I + β
#         "gamma",               # power-law I^γ
#         "gaussian_noise",      # additive Gaussian noise
#         "gaussian_blur",       # isotropic blur
#         "bias_field",          # low-frequency multiplicative bias field
#     ],
