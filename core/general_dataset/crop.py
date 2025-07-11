from __future__ import annotations

"""Patch‑sampling utilities (channel‑aware).
Public API
~~~~~~~~~~
* **bigger_crop** - pad spatial dims, then return a random crop whose spatial
  size is ``ceil(√D · patch_size)``.
* **center_crop** - symmetric crop back to *patch_size* in spatial dims.

"""

from typing import Dict, Sequence, Tuple
import math
import random
import numpy as np

__all__ = [
    "bigger_crop",
    "center_crop",
]

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _channel_flags_and_spatial_shape(
    data: Dict[str, np.ndarray],
    dim: int,
) -> Tuple[Tuple[int, ...], Dict[str, bool]]:
    """Return common *spatial* shape and ``{key: has_channel_axis}`` mapping."""
    if not data:
        raise ValueError("`data` must contain at least one modality.")

    spatial_shape: Tuple[int, ...] | None = None
    ch_flag: Dict[str, bool] = {}
    for k, v in data.items():
        if v.ndim not in (dim, dim + 1):
            raise ValueError(f"'{k}' must have {dim} or {dim+1} dims, got {v.ndim}")
        cur_spatial = v.shape[-dim:]
        if spatial_shape is None:
            spatial_shape = cur_spatial
        elif cur_spatial != spatial_shape:
            raise ValueError(
                f"All modalities must share spatial shape {spatial_shape}, but '{k}' has {cur_spatial}."
            )
        ch_flag[k] = (v.ndim == dim + 1)
    return spatial_shape, ch_flag


def _compute_sizes(dim: int, patch_size: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    if len(patch_size) != dim:
        raise ValueError("`patch_size` length must match spatial dim")
    scale = math.sqrt(dim)
    patch_size = np.asarray(patch_size, dtype=int)
    big = np.ceil(scale * patch_size).astype(int)
    pad = ((big - patch_size) + 1) // 2  # ceil‑to‑left
    return pad, big


def _random_start(
    rng: random.Random | np.random.RandomState | np.random.Generator,
    full_shape: Sequence[int],
    crop_shape: Sequence[int],
) -> np.ndarray:
    """Random spatial corner so *crop_shape* fits inside *full_shape*."""
    max_start = np.array(full_shape) - np.array(crop_shape)
    if np.any(max_start < 0):
        raise ValueError("`crop_shape` larger than padded shape - bug in logic.")

    starts = []
    for m in max_start:
        if hasattr(rng, "integers"):
            starts.append(int(rng.integers(0, m + 1)))  # numpy Generator
        else:
            starts.append(int(rng.randint(0, int(m))))  # RandomState or random.Random
    return np.asarray(starts, dtype=int)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def bigger_crop(
    data: Dict[str, np.ndarray],
    patch_size: Sequence[int],
    *,
    pad_mode: str = "edge",
    rng: random.Random | np.random.RandomState | np.random.Generator | None = None,
) -> Dict[str, np.ndarray]:
    rng = rng or np.random.default_rng()
    dim = len(patch_size)

    _, has_ch = _channel_flags_and_spatial_shape(data, dim)
    pad, big = _compute_sizes(dim, patch_size)
    pad_spatial = [(int(p), int(p)) for p in pad]

    # Pad spatial dims
    padded = {}
    for k, v in data.items():
        pad_cfg = pad_spatial if not has_ch[k] else [(0, 0)] + pad_spatial
        padded[k] = np.pad(v, pad_cfg, mode=pad_mode)

    # One random crop applied to all modalities
    spatial_full = padded[next(iter(padded))].shape[-dim:]
    start = _random_start(rng, spatial_full, big)
    end = start + big
    spatial_slice = tuple(slice(int(s), int(e)) for s, e in zip(start, end))

    cropped = {}
    for k, v in padded.items():
        full_slice = (slice(None),) + spatial_slice if has_ch[k] else spatial_slice
        cropped[k] = v[full_slice]
    return cropped


def center_crop(
    data: Dict[str, np.ndarray],
    patch_size: Sequence[int],
) -> Dict[str, np.ndarray]:
    dim = len(patch_size)
    _, has_ch = _channel_flags_and_spatial_shape(data, dim)
    patch_sz_arr = np.asarray(patch_size, dtype=int)

    out = {}
    for k, v in data.items():
        spatial_shape = np.array(v.shape[-dim:])
        extra = spatial_shape - patch_sz_arr
        if np.any(extra < 0):
            raise ValueError("`patch_size` larger than input along some axis.")
        offset = extra // 2
        spatial_slice = tuple(slice(int(o), int(o + p)) for o, p in zip(offset, patch_sz_arr))
        full_slice = (slice(None),) + spatial_slice if has_ch[k] else spatial_slice
        out[k] = v[full_slice]
    return out
