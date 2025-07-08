import math
import random
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.ndimage import (
    rotate, zoom, map_coordinates, gaussian_filter
)

# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────

def _pad_to_square(img: np.ndarray, side: int) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
    """Sym-pad a 2-D image to (side, side); return padded + paddings."""
    h, w = img.shape
    top = (side - h) // 2
    bot = side - h - top
    lef = (side - w) // 2
    rig = side - w - lef
    return np.pad(img, ((top, bot), (lef, rig)), mode="reflect"), (top, bot), (lef, rig)


def _pad_to_cube(vol: np.ndarray, side: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Sym-pad a 3-D volume to a cube (side, side, side)."""
    d, h, w = vol.shape
    pad_d0 = (side - d) // 2
    pad_d1 = side - d - pad_d0
    pad_h0 = (side - h) // 2
    pad_h1 = side - h - pad_h0
    pad_w0 = (side - w) // 2
    pad_w1 = side - w - pad_w0
    return (
        np.pad(vol, ((pad_d0, pad_d1), (pad_h0, pad_h1), (pad_w0, pad_w1)), mode="reflect"),
        (pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1),
    )


def _crop_center(vol: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Centre-crop `vol` back to `target_shape`."""
    src_shape = vol.shape
    slices = tuple(
        slice((s - t) // 2, (s - t) // 2 + t) for s, t in zip(src_shape, target_shape)
    )
    return vol[slices]


def elastic_deformation_2d(
    img: np.ndarray, alpha: float, sigma: float, rng: np.random.RandomState
) -> np.ndarray:
    """Elastic deformation of a 2-D image (Simard et al.)."""
    h, w = img.shape
    dx = gaussian_filter((rng.rand(h, w) * 2 - 1), sigma, mode="constant") * alpha
    dy = gaussian_filter((rng.rand(h, w) * 2 - 1), sigma, mode="constant") * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = (y + dy).ravel(), (x + dx).ravel()
    return map_coordinates(img, indices, order=1, mode="reflect").reshape(img.shape)


def elastic_deformation_3d(
    vol: np.ndarray, alpha: float, sigma: float, rng: np.random.RandomState
) -> np.ndarray:
    """Elastic deformation of a 3-D volume (Cuadrado 3-D extension)."""
    d, h, w = vol.shape
    dx = gaussian_filter((rng.rand(d, h, w) * 2 - 1), sigma, mode="constant") * alpha
    dy = gaussian_filter((rng.rand(d, h, w) * 2 - 1), sigma, mode="constant") * alpha
    dz = gaussian_filter((rng.rand(d, h, w) * 2 - 1), sigma, mode="constant") * alpha
    z, y, x = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    indices = (z + dz).ravel(), (y + dy).ravel(), (x + dx).ravel()
    return map_coordinates(vol, indices, order=1, mode="reflect").reshape(vol.shape)


def _random_in(cfg: Dict, lo: str, hi: str) -> float:
    return random.uniform(cfg[lo], cfg[hi])


def _sanitize_input(img: np.ndarray, data_dim: int) -> np.ndarray:
    """Ensure correct dimensionality; drop singleton channel if present."""
    if data_dim == 2:
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            # (C,H,W) or (H,W,C); keep first channel
            if img.shape[0] <= 3 and img.shape[2] > 3:  # assume CHW
                return img[0]
            if img.shape[2] <= 3 and img.shape[0] > 3:  # assume HWC
                return img[..., 0]
        raise ValueError(
            f"Expected (H,W) or (C,H,W)/(H,W,C) with C=1 for 2-D; got {img.shape}"
        )
    else:  # data_dim == 3
        if img.ndim == 3:
            return img
        if img.ndim == 4 and img.shape[0] == 1:  # (1,D,H,W)
            return img[0]
        raise ValueError(
            f"Expected (D,H,W) or (1,D,H,W) for 3-D; got {img.shape}"
        )


# ─── main entry ───────────────────────────────────────────────

def augment_image(
    image: np.ndarray,
    aug_cfg: Dict,
    data_dim: int = 2,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """YAML-driven augmentor (pure NumPy/SciPy).

    New in v1.1 -- rotation padding control
    ---------------------------------------
    By default, 2-D and 3-D rotations are *padded* to avoid corner cropping
    (historical behaviour).  Set `"pad": false` inside the `rotation` block
    to rotate *in-place* without the √(dim)× oversizing.  This will keep the
    original array shape but inevitably crops the out-of-frame corners when
    the angle is large.

    Example YAML snippet::

        rotation:
            min: -20
            max:  20
            pad: false  # <- NEW

    The same flag is supported for 3-D volumes.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    arr = _sanitize_input(image, data_dim).astype(np.float32, copy=True)

    # draw random spatial parameters ───────────────────────────
    do_h = "flip_h" in aug_cfg and random.random() < aug_cfg["flip_h"]["prob"]
    do_v = "flip_v" in aug_cfg and random.random() < aug_cfg["flip_v"]["prob"]
    do_d = "flip_d" in aug_cfg and random.random() < aug_cfg["flip_d"]["prob"]

    # rotation parameters
    if "rotation" in aug_cfg:
        rot_cfg = aug_cfg["rotation"]
        angle = _random_in(rot_cfg, "min", "max")
        pad_rotate = rot_cfg.get("pad", True)
    else:
        angle = 0.0
        pad_rotate = True

    scale = _random_in(aug_cfg["scale"], "min", "max") if "scale" in aug_cfg else 1.0

    alpha = (
        _random_in(aug_cfg["elastic"], "alpha_min", "alpha_max")
        if "elastic" in aug_cfg
        else 0.0
    )
    sigma = (
        _random_in(aug_cfg["elastic"], "sigma_min", "sigma_max")
        if "elastic" in aug_cfg
        else 0.0
    )

    # ═════ Spatial transforms ═════
    if data_dim == 2 or arr.ndim == 2:
        h, w = arr.shape  # cache original size once

        # ── flips
        if do_h:
            arr = arr[:, ::-1]
        if do_v:
            arr = arr[::-1, :]
        if do_d:
            arr = arr.T

        # ── rotation (optional padding)
        if angle != 0.0:
            if pad_rotate:
                side = int(math.ceil(math.sqrt(2) * max(h, w)))
                padded, *_ = _pad_to_square(arr, side)
                rotated = rotate(padded, angle, reshape=False, order=1, mode="reflect")
                arr = _crop_center(rotated, (h, w))
            else:  # in-place rotate – shape preserved but corners may clip
                arr = rotate(arr, angle, reshape=False, order=1, mode="reflect")
                # no additional cropping needed – rotate keeps same shape when reshape=False

        # ── scaling (isotropic)
        if scale != 1.0:
            scaled = zoom(arr, scale, order=1)
            arr = (
                _crop_center(scaled, (h, w))
                if scale > 1.0
                else np.pad(
                    scaled,
                    (
                        ((h - scaled.shape[0]) // 2, h - scaled.shape[0] - (h - scaled.shape[0]) // 2),
                        ((w - scaled.shape[1]) // 2, w - scaled.shape[1] - (w - scaled.shape[1]) // 2),
                    ),
                    mode="reflect",
                )
            )

        # ── elastic
        if alpha > 0 and sigma > 0:
            arr = elastic_deformation_2d(arr, alpha, sigma, random_state)

    else:  # data_dim == 3 (volumetric)
        D, H, W = arr.shape  # original dims once

        # ── flips
        if do_h:
            arr = arr[:, :, ::-1]  # flip width
        if do_v:
            arr = arr[:, ::-1, :]  # flip height
        if do_d:
            arr = arr[::-1, :, :]  # flip depth

        # ── rotation around all three axes (optional padding)
        if angle != 0.0:
            if pad_rotate:
                side = int(math.ceil(math.sqrt(3) * max(D, H, W)))
                padded, *_ = _pad_to_cube(arr, side)
            else:
                padded = arr  # operate in-place when no padding

            # sample separate angles per axis using same cfg range
            ax = angle
            ay = _random_in(rot_cfg, "min", "max") if "rotation" in aug_cfg else 0.0
            az = _random_in(rot_cfg, "min", "max") if "rotation" in aug_cfg else 0.0

            rot = rotate(padded, ax, axes=(1, 2), reshape=False, order=1, mode="reflect")
            rot = rotate(rot, ay, axes=(0, 2), reshape=False, order=1, mode="reflect")
            rot = rotate(rot, az, axes=(0, 1), reshape=False, order=1, mode="reflect")

            if pad_rotate:
                arr = _crop_center(rot, (D, H, W))
            else:
                arr = rot  # already same shape

        # ── scaling
        if scale != 1.0:
            scaled = zoom(arr, scale, order=1)
            arr = (
                _crop_center(scaled, (D, H, W))
                if scale > 1.0
                else np.pad(
                    scaled,
                    tuple(
                        (
                            (t - s) // 2,
                            t - s - (t - s) // 2,
                        )
                        for s, t in zip(scaled.shape, (D, H, W))
                    ),
                    mode="reflect",
                )
            )

        # ── elastic
        if alpha > 0 and sigma > 0:
            arr = elastic_deformation_3d(arr, alpha, sigma, random_state)

    # ═════ Intensity transforms ═════
    # brightness / contrast
    if "brightness_contrast" in aug_cfg:
        bc = aug_cfg["brightness_contrast"]
        a = _random_in(bc, "alpha_min", "alpha_max")
        b = _random_in(bc, "beta_min", "beta_max")
        arr = arr * a + b

    # gamma
    if "gamma" in aug_cfg:
        g = _random_in(aug_cfg["gamma"], "min", "max")
        arr = np.sign(arr) * (np.abs(arr) ** g)

    # gaussian noise
    if "gaussian_noise" in aug_cfg:
        gn = aug_cfg["gaussian_noise"]
        std = _random_in(gn, "min", "max")
        arr += random_state.normal(0, std, size=arr.shape)

    # gaussian blur
    if "gaussian_blur" in aug_cfg:
        gb = aug_cfg["gaussian_blur"]
        sigma_blur = _random_in(gb, "min", "max")
        arr = gaussian_filter(arr, sigma=sigma_blur)

    # bias field
    if "bias_field" in aug_cfg:
        bf = aug_cfg["bias_field"]
        alpha_bf = _random_in(bf, "min", "max")
        field = gaussian_filter(random_state.rand(*arr.shape), sigma=max(arr.shape) / 4)
        field = (field - field.min()) / (field.max() - field.min()) * 2 - 1
        arr *= 1 + alpha_bf * field

    return arr.astype(image.dtype, copy=False)
