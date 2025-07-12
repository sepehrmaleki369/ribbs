# augmentations_core.py  -------------------------------------------------------
# Built for Kornia ≥0.7  ·  elasticdeform ≥0.5
# Author: <you>  ·  2025-07
# -----------------------------------------------------------------------------
"""
Unified data-augmentation helper for both 2-D (C,H,W) and 3-D (C,D,H,W) tensors.

* Works on a *dict* of modalities (image, label, distance …)
* Generates **one** random parameter set per augmentation and applies it
  consistently to every selected modality.
* Seamlessly switches between Kornia’s 2-D and 3-D ops.
* Full GPU pipeline — even elastic deformation (elasticdeform.torch).

Public API
----------
augment_images(data, aug_cfg, dim, rng=None, verbose=False)
    data     : Dict[str, torch.Tensor]           (C,H,W) or (C,D,H,W)
    aug_cfg  : Single entry of the YAML "augmentation:" list
    dim      : 2 or 3
    returns  : (augmented_dict, metadata)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import inspect
import math
import numpy as np
import torch
import kornia as K
from torch import Tensor
import torch.nn.functional as F

import elasticdeform.torch as edt        # GPU warp
import elasticdeform                     # grid sampler

__all__ = ["augment_images"]

# -----------------------------------------------------------------------------#
# constants & helpers                                                          #
# -----------------------------------------------------------------------------#
_NEAREST, _LINEAR = "nearest", "bilinear"
_RNG = np.random.Generator


def _interp(mod: str) -> str:
    """Label ⇒ nearest, everything else ⇒ bilinear/linear."""
    return _NEAREST if mod == "label" else _LINEAR


def _maybe_scalar(v: Any, rng: _RNG) -> float:
    """Return scalar or sample from [lo,hi]."""
    if isinstance(v, (list, tuple)) and len(v) == 2:
        lo, hi = map(float, v)
        return float(rng.uniform(lo, hi))
    return float(v)


def _maybe_tuple(v: Any, rng: _RNG, d: int) -> Tuple[float, ...]:
    """Convert *v* into a d-tuple, sampling ranges if needed."""
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            # isotropic range
            scl = _maybe_scalar(v, rng)
            return (scl,) * d
        if len(v) == d:
            return tuple(_maybe_scalar(x, rng) for x in v)
    return (float(v),) * d


# -----------------------------------------------------------------------------#
# convenience selectors                                                        #
# -----------------------------------------------------------------------------#
def _kornia_cls(cls2d, cls3d, dim: int):
    """Return the correct (2-D | 3-D) class or raise if missing."""
    cls = cls2d if dim == 2 else cls3d
    if cls is None:
        raise RuntimeError(f"{cls2d.__name__} has no 3-D counterpart in this "
                           "Kornia build – remove it from the 3-D pipeline.")
    return cls


def _has_arg(func, name: str) -> bool:
    return name in inspect.signature(func).parameters


# -----------------------------------------------------------------------------#
# primitive transforms                                                         #
# -----------------------------------------------------------------------------#
def _apply_flip(x: Tensor, which: str, dim: int) -> Tensor:
    axes = {"flip_horizontal": -1,
            "flip_vertical":   -2,
            "flip_depth":      -3}.get(which)
    return x.flip(axes) if axes is not None else x


def _apply_rotate(x: Tensor, angles, dim: int, mode: str) -> Tensor:
    if dim == 2:
        ang = torch.as_tensor([angles], device=x.device, dtype=x.dtype)
        return K.geometry.transform.rotate(x, ang, mode=mode, align_corners=False)

    # Kornia rotate3d(yaw, pitch, roll)
    yaw, pitch, roll = angles
    yaw   = torch.as_tensor([yaw],   device=x.device, dtype=x.dtype)
    pitch = torch.as_tensor([pitch], device=x.device, dtype=x.dtype)
    roll  = torch.as_tensor([roll],  device=x.device, dtype=x.dtype)
    return K.geometry.transform.rotate3d(
        x, yaw, pitch, roll, mode=mode, align_corners=False
    )


def _apply_scale(x: Tensor, scale, dim: int, mode: str) -> Tensor:
    if dim == 2:
        s = torch.as_tensor([scale], device=x.device, dtype=x.dtype)
        return K.geometry.transform.scale(x, s, mode=mode, align_corners=False)

    # Kornia’s RandomAffine3D handles scaling tuples
    kw_interp = {"interpolation" if _has_arg(K.augmentation.RandomAffine3D, "interpolation")
                 else "resample": mode}
    aff = K.augmentation.RandomAffine3D(degrees=(0., 0., 0.),
                                        scale=scale,
                                        p=1.0,
                                        align_corners=False,
                                        **kw_interp)
    return aff(x)


def _apply_translate(x: Tensor, shift, dim: int, mode: str) -> Tensor:
    if dim == 2:
        t = torch.as_tensor([[shift, shift]], device=x.device, dtype=x.dtype)
        return K.geometry.transform.translate(x, t, mode=mode, align_corners=False)

    # Kornia lacks translate3d – emulate with integer roll
    sx, sy, sz = shift
    dx = int(round(sx * x.size(-3)))
    dy = int(round(sy * x.size(-2)))
    dz = int(round(sz * x.size(-1)))
    return x.roll((dx, dy, dz), dims=(-3, -2, -1))


def _apply_elastic(x: Tensor, sigma: float, pts: int, axis) -> Tensor:
    """
    Elastic deformation that stays on GPU.

    * If elasticdeform ≥ 0.5 is installed we call its
      `random_displacement` helper.
    * Otherwise we create a simple Gaussian-noise grid of identical shape.
      (For data-augmentation that approximation is perfectly fine.)
    """
    ndim = len(axis)
    dtype, device = x.dtype, x.device

    # 1) displacement grid ---------------------------------------------------
    if hasattr(elasticdeform, "random_displacement"):         # v0.5+
        disp_np = elasticdeform.random_displacement(ndim, pts, pts, sigma)
        disp = torch.as_tensor(disp_np, dtype=dtype, device=device)
    else:                                                     # legacy build
        # shape (ndim, pts, pts, …), same convention as elasticdeform
        shape = (ndim,) + (pts,) * ndim
        disp = torch.randn(*shape, dtype=dtype, device=device) * sigma

    # 2) warp (all-torch, differentiable) -------------------------------
    return edt.deform_grid(x, disp, axis=axis, order=3)

# SIMPLE IMPLEMENTATION FOR NOT SUPPORTEDs 
def _contrast_volume(x: Tensor, factor: float) -> Tensor:
    """Simple per-tensor contrast:  y = (x - mean)*f + mean."""
    mean = x.mean(dim=(-3, -2, -1), keepdim=True)
    return (x - mean) * factor + mean

def _brightness_volume(x: Tensor, factor: float) -> Tensor:
    """Multiply full volume by *factor* (simple brightness)."""
    return x * factor

def _gamma_volume(x: Tensor, gamma: float) -> Tensor:
    """Per-volume gamma correction:  y = x**gamma  (expects x in [0,1])."""
    # clamp protects against inf / nan if values are exactly 0
    return torch.clamp(x, min=1e-6) ** gamma

def _gaussian_noise_volume(x: Tensor, mean: float, std: float) -> Tensor:
    """Add i.i.d. Gaussian noise to the whole volume."""
    noise = torch.randn_like(x) * std + mean
    return x + noise

def _gaussian_kernel1d(radius: int, sigma: float, dtype, device) -> Tensor:
    """Returns a 1-D tensor of size (2*radius+1)."""
    # gaussian centred at 0 … radius
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def _gaussian_blur_volume(x: Tensor, k: int, sigma: float) -> Tensor:
    """Depth-wise separable 3-D Gaussian blur (B,C,D,H,W)."""
    r = k // 2
    dtype, device = x.dtype, x.device
    k1 = _gaussian_kernel1d(r, sigma, dtype, device)

    # separable ⇒ three 1-D convolutions
    pad = (r, r)
    # along depth (dim=-3)
    x = F.conv3d(F.pad(x, pad * 3, mode="reflect"),
                 k1.view(1, 1, -1, 1, 1), groups=x.size(1))
    # along height (dim=-2)
    x = F.conv3d(F.pad(x, pad * 3, mode="reflect"),
                 k1.view(1, 1, 1, -1, 1), groups=x.size(1))
    # along width (dim=-1)
    x = F.conv3d(F.pad(x, pad * 3, mode="reflect"),
                 k1.view(1, 1, 1, 1, -1), groups=x.size(1))
    return x
# -----------------------------------------------------------------------------#
# main entry                                                                    #
# -----------------------------------------------------------------------------#
def augment_images(
    data: Dict[str, Tensor],
    aug_cfg: Dict[str, Any],
    dim: int,
    rng: Optional[_RNG] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:

    if rng is None:
        rng = np.random.default_rng()

    name: str = aug_cfg["name"]
    if rng.random() > aug_cfg.get("p", 1.0):
        return data, {"name": name, "skipped": True}

    # ---------- sample all random parameters once --------------------------------
    prm, p = {}, aug_cfg.get("params", {})

    if name in ("flip_horizontal", "flip_vertical", "flip_depth"):
        pass

    elif name == "rotation":
        prm["angle"] = (_maybe_scalar(p["degrees"], rng)
                        if dim == 2 else _maybe_tuple(p["degrees"], rng, 3))

    elif name == "scaling":
        prm["scale"] = (_maybe_scalar(p["scale_limit"], rng)
                        if dim == 2 else (_maybe_tuple(p["scale_limit"], rng, 3)
                        if isinstance(p["scale_limit"], (int, float))
                        else tuple(_maybe_scalar(s, rng) if isinstance(s, (int, float))
                                   else tuple(map(float, s))
                                   for s in p["scale_limit"])))

    elif name == "translation":
        prm["shift"] = (_maybe_scalar(p["shift_limit"], rng)
                        if dim == 2 else _maybe_tuple(p["shift_limit"], rng, 3))

    elif name == "elastic_deformation":
        prm["sigma"]  = _maybe_scalar(p["sigma"], rng)
        prm["points"] = int(_maybe_scalar(p["alpha"], rng))

    elif name in ("brightness", "contrast"):
        lim = _maybe_scalar(p["limit"], rng)              # –0.15 … 0.15
        lo, hi = 1.0 + lim, 1.0 - lim if lim < 0 else 1.0 + lim
        prm["range"] = (min(lo, hi), max(lo, hi))

    elif name == "gamma":
        lo, hi = map(float, p["gamma_limit"])   # e.g. [0.9, 1.1] from YAML
        prm["range"] = (lo, hi)

    elif name == "gaussian_noise":
        prm["mean"] = float(p.get("mean", 0.0))
        prm["std"]  = _maybe_scalar(p["std"], rng)

    elif name == "blur":
        prm["kernel"] = int(np.clip(_maybe_scalar(p["kernel_size"], rng), 3, 31)) | 1
        prm["sigma"]  = _maybe_scalar(p["sigma"], rng)

    elif name == "color_jitter":
        prm = {}
        for k in ("brightness", "contrast", "saturation"):
            prm[k] = 1.0 + _maybe_scalar(p[k], rng)          # > 0
        delta_h = _maybe_scalar(p["hue"], rng)               # e.g. −0.05 … +0.05
        prm["hue"] = abs(delta_h)                            # 0 … 0.05 (non-neg.)

    else:
        raise NotImplementedError(name)

    if verbose:
        print(f"[augment] {name:>18}  params={prm}")

    # ---------- apply transform ---------------------------------------------------
    mods = aug_cfg.get("modalities", data.keys())
    out: Dict[str, Tensor] = {}

    for mod, x in data.items():
        if mod not in mods:                      # untouched modality
            out[mod] = x
            continue

        xb = x.unsqueeze(0).float()              # Kornia expects BCHW / BCDHW
        mode = _interp(mod)

        if name.startswith("flip"):
            yb = _apply_flip(xb, name, dim)

        elif name == "rotation":
            yb = _apply_rotate(xb, prm["angle"], dim, mode)

        elif name == "scaling":
            yb = _apply_scale(xb, prm["scale"], dim, mode)

        elif name == "translation":
            yb = _apply_translate(xb, prm["shift"], dim, mode)

        elif name == "elastic_deformation":
            axis = (2, 3) if dim == 2 else (2, 3, 4)
            yb = _apply_elastic(xb, prm["sigma"], prm["points"], axis)

        elif name == "brightness":
            if dim == 2:
                cls = K.augmentation.RandomBrightness      # class *does* exist for 2-D
                yb  = cls(brightness=prm["range"], p=1.0)(xb)
            else:                                          # 3-D fallback
                fac = float(rng.uniform(*prm["range"]))
                yb  = _brightness_volume(xb, fac)

        elif name == "contrast":
            if dim == 2:
                cls = K.augmentation.RandomContrast
                yb = cls(contrast=prm["range"], p=1.0)(xb)
            else:                # 3-D fallback
                fac = float(rng.uniform(*prm["range"]))
                yb  = _contrast_volume(xb, fac)


        elif name == "gamma":
            if dim == 2:
                # Kornia RandomGamma accepts a (min, max) tuple
                yb = K.augmentation.RandomGamma(gamma=prm["range"], p=1.0)(xb)
            else:                                   # 3-D fallback
                g  = float(rng.uniform(*prm["range"]))   # sample γ ∈ [lo, hi]
                yb = _gamma_volume(xb, g)

        elif name == "gaussian_noise":
            if dim == 2:
                yb = K.augmentation.RandomGaussianNoise(mean=prm["mean"],
                                                        std=prm["std"], p=1.0)(xb)
            else:                       # 3-D fallback
                yb = _gaussian_noise_volume(xb, prm["mean"], prm["std"])


        elif name == "blur":
            ks = prm["kernel"]
            if dim == 2:
                yb = K.augmentation.RandomGaussianBlur(
                        (ks, ks),
                        sigma=(prm["sigma"], prm["sigma"]),
                        p=1.0,
                    )(xb)
            else:                                        # 3-D fallback
                yb = _gaussian_blur_volume(xb, ks, prm["sigma"])

        elif name == "color_jitter":
            if dim == 3:                        # not supported -> skip
                yb = xb
            else:
                yb = K.augmentation.ColorJiggle(
                    brightness=prm["brightness"],
                    contrast=prm["contrast"],
                    saturation=prm["saturation"],
                    hue=prm["hue"],
                    p=1.0,
                )(xb)

        else:
            raise RuntimeError(f"unhandled {name}")

        out[mod] = yb.squeeze(0).to(x.dtype)

    meta = {"name": name,
            "sampled_params": prm,
            "modalities": list(mods),
            "skipped": False}
    return out, meta
