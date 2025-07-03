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

def get_augmentation_metadata(augmentations: List[str], data_dim: int) -> Dict[str, Any]:
    """Return a dict of randomly‑sampled parameters needed by the requested
    *per‑patch* augmentations.
    
    Parameters that are not required by a given transform are simply omitted.
    Existing rotation/flip keys are kept unchanged.
    """
    meta: Dict[str, Any] = {}

    # -------- spatial (existing) --------
    if "rotation" in augmentations:
        meta["angle"] = float(np.random.uniform(0.0, 360.0))
    if "flip_h" in augmentations:
        meta["flip_h"] = bool(np.random.rand() > 0.5)
    if "flip_v" in augmentations:
        meta["flip_v"] = bool(np.random.rand() > 0.5)
    if "flip_d" in augmentations and data_dim == 3:
        meta["flip_d"] = bool(np.random.rand() > 0.5)

    # -------- spatial (new) -------------
    if "scale" in augmentations:
        meta["scale_factor"] = float(np.random.uniform(0.8, 1.2))
    if "elastic" in augmentations:
        meta["alpha"] = float(np.random.uniform(5.0, 10.0))   # displacement (px)
        meta["sigma"] = float(np.random.uniform(3.0, 6.0))   # smoothing (px)

    # -------- intensity (new) -----------
    if "brightness_contrast" in augmentations:
        meta["bc_alpha"] = float(np.random.uniform(0.9, 1.1))   # contrast scale
        meta["bc_beta"]  = float(np.random.uniform(-30.0, 30.0)) # brightness shift (0‑255 space)
    if "gamma" in augmentations:
        meta["gamma"] = float(np.random.uniform(0.7, 1.5))
    if "gaussian_noise" in augmentations:
        meta["noise_sigma"] = float(np.random.uniform(0.01, 0.03))  # relative to dyn‑range
    if "gaussian_blur" in augmentations:
        meta["blur_sigma"] = float(np.random.uniform(0.5, 1.5))
    if "bias_field" in augmentations:
        meta["bias_amp"] = float(np.random.uniform(0.2, 0.4))       # ± amplitude (×mean)

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


def extract_data(imgs: Dict[str, np.ndarray], x: int, y: int, z: int, patch_size_xy: int, patch_size_z: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Existing helper – unchanged (copied verbatim to keep augments.py self‑contained)."""
    data: Dict[str, np.ndarray] = {}
    for key, arr in imgs.items():
        if arr.ndim == 4:  # C, D, H, W
            psz = patch_size_z or patch_size_xy
            data[f"{key}_patch"] = arr[:, z : z + psz, y : y + patch_size_xy, x : x + patch_size_xy]
        elif arr.ndim == 3:
            if patch_size_z is not None:
                data[f"{key}_patch"] = arr[z : z + patch_size_z, y : y + patch_size_xy, x : x + patch_size_xy]
            else:
                data[f"{key}_patch"] = arr[y : y + patch_size_xy, x : x + patch_size_xy]
        elif arr.ndim == 2:
            data[f"{key}_patch"] = arr[y : y + patch_size_xy, x : x + patch_size_xy]
        else:
            raise ValueError("Unsupported array dims")
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
    data_dim: int,
) -> Dict[str, np.ndarray]:
    """Extract a patch and apply the augmentations specified in `augmentations`.

    Spatial transforms that affect geometry (flip/scale/elastic/rotation) are applied
    to *all* modalities so they remain aligned.

    Intensity‑only transforms are applied **only** to the "image" modality.
    """

    # 1) pre‑extract the patch (no aug yet)
    z = metadata.get("z", 0)
    data = extract_data(imgs, metadata["x"], metadata["y"], z, patch_size_xy, patch_size_z)

    # 2) geometric augmentations ------------------------------------------------
    for key in list(data.keys()):
        if not key.endswith("_patch"):
            continue
        modality = key.replace("_patch", "")
        patch = data[key]

        # (a) flips – we also flip the *full* image so that rotation below sees the correct orientation
        if modality in imgs:
            if "flip_h" in augmentations and metadata.get("flip_h", False):
                imgs[modality] = flip_h(imgs[modality])
                patch = flip_h(patch)
            if "flip_v" in augmentations and metadata.get("flip_v", False):
                imgs[modality] = flip_v(imgs[modality])
                patch = flip_v(patch)
            if data_dim == 3 and "flip_d" in augmentations and metadata.get("flip_d", False):
                imgs[modality] = flip_d(imgs[modality])
                patch = flip_d(patch)

        # (b) rotation – uses *full* (possibly flipped) image to avoid holes
        if "rotation" in augmentations:
            patch = rotate_(imgs[modality], metadata, patch_size_xy, patch_size_z, data_dim)

        # (c) scale (zoom)
        if "scale" in augmentations:
            patch = scale_patch(patch, metadata["scale_factor"], data_dim)

        # (d) elastic warp
        if "elastic" in augmentations:
            patch = elastic_deform_patch(patch, metadata["alpha"], metadata["sigma"], data_dim)

        data[key] = patch  # write back

    # 3) intensity‑only transforms (image modality *only*) ----------------------
    if "image_patch" in data:
        img_patch = data["image_patch"].astype(np.float32)

        if "brightness_contrast" in augmentations:
            img_patch = adjust_brightness_contrast(img_patch, metadata["bc_alpha"], metadata["bc_beta"])
        if "gamma" in augmentations:
            img_patch = gamma_correction(img_patch, metadata["gamma"])
        if "gaussian_noise" in augmentations:
            img_patch = add_gaussian_noise(img_patch, metadata["noise_sigma"])
        if "gaussian_blur" in augmentations:
            img_patch = gaussian_blur_patch(img_patch, metadata["blur_sigma"], data_dim)
        if "bias_field" in augmentations:
            img_patch = bias_field_patch(img_patch, metadata["bias_amp"], data_dim)

        data["image_patch"] = img_patch.astype(data["image_patch"].dtype)

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