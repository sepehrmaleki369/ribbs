"""
augmentations.py
Unified augmentation helper for both 2-D and 3-D data.

* Any numeric YAML parameter may be
  - a single value            ➔ used as-is;
  - a 2-element list/tuple    ➔ uniformly sampled in [lo, hi].

* Interpolation is chosen automatically:
    nearest  → labels
    bilinear → images / continuous maps
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import numpy as np
import torch
import kornia as K

_INTERP_NEAREST, _INTERP_LINEAR = "nearest", "bilinear"

# -----------------------------------------------------------------------------#
# Small helpers                                                                
# -----------------------------------------------------------------------------#
def _select_interp(mod: str) -> str:
    return _INTERP_NEAREST if mod == "label" else _INTERP_LINEAR


def _interp_kwargs(cls, interp: str) -> Dict[str, str]:
    sig = inspect.signature(cls)
    for k in ("interpolation", "resample", "mode"):
        if k in sig.parameters:
            return {k: interp}
    return {}


def _tuple_if_range(v: Any) -> Any:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return tuple(v)
    return v


def _maybe_sample(v: Any, rng: np.random.Generator) -> Any:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        lo, hi = v
        return float(rng.uniform(lo, hi))
    return v


def _safe_single_number(x: Any) -> Any:
    """Ensure single numbers passed to ColorJiggle are non-negative."""
    return abs(x) if isinstance(x, float) else _tuple_if_range(x)


def _pair(x: float) -> tuple[float, float]:
    return (x, x)


def _triple(x: float) -> tuple[float, float, float]:
    return (x, x, x)


# -----------------------------------------------------------------------------#
# Factory                                                                      
# -----------------------------------------------------------------------------#
def _make_kornia_transform(name: str, p: Dict[str, Any], dim: int, interp: str):
    T = K.augmentation

    # ============================== 2-D ===================================== #
    if dim == 2:
        if name == "flip_horizontal":
            return T.RandomHorizontalFlip(p=1.0)
        if name == "flip_vertical":
            return T.RandomVerticalFlip(p=1.0)

        if name == "rotation":
            return T.RandomRotation(
                _tuple_if_range(p["degrees"]), p=1.0, **_interp_kwargs(T.RandomRotation, interp)
            )

        if name == "scaling":
            sc = p["scale_limit"]
            sc = _pair(sc) if isinstance(sc, float) else _tuple_if_range(sc)
            return T.RandomAffine(degrees=0, scale=sc, p=1.0, **_interp_kwargs(T.RandomAffine, interp))

        if name == "translation":
            # user may give a single ±limit or a [-limit, +limit] range
            lim = p["shift_limit"]
            if isinstance(lim, float):                      # e.g. 0.1 or −0.1
                mag = abs(lim)
            else:                                           # e.g. [−0.1, 0.1]
                lo, hi = lim
                mag = max(abs(lo), abs(hi))                 # 0.1
            sh = _pair(mag)                                 # (0.1, 0.1)
            return T.RandomAffine(degrees=0, translate=sh, p=1.0,
                                   **_interp_kwargs(T.RandomAffine, interp))

        if name == "elastic_deformation":
            alp = p["alpha"]; sig = p["sigma"]
            alp = _pair(alp) if isinstance(alp, float) else _tuple_if_range(alp)
            sig = _pair(sig) if isinstance(sig, float) else _tuple_if_range(sig)
            return T.RandomElasticTransform(alpha=alp, sigma=sig, p=1.0)

        if name == "brightness":
            return T.RandomBrightness(_safe_single_number(p["limit"]), p=1.0)
        if name == "contrast":
            return T.RandomContrast(_safe_single_number(p["limit"]), p=1.0)
        if name == "gamma":
            g = p["gamma_limit"]; g = _pair(abs(g)) if isinstance(g, float) else _tuple_if_range(g)
            return T.RandomGamma(g, p=1.0)

        if name == "gaussian_noise":
            return T.RandomGaussianNoise(
                mean=p.get("mean", 0.0), std=_safe_single_number(p["std"]), p=1.0
            )

        if name == "blur":
            k = int(np.clip(p["kernel_size"], 3, 31)) | 1  # odd integer
            raw = p.get("sigma", (0.1, 2.0))
            if isinstance(raw, float):                       # eg 0.03
                sig = (raw, raw)
            else:                                            # tuple/range
                sig = _tuple_if_range(raw)
            return T.RandomGaussianBlur((k, k), sigma=sig, p=1.0)


        if name == "color_jitter":
            return T.ColorJiggle(
                brightness=_safe_single_number(p["brightness"]),
                contrast=_safe_single_number(p["contrast"]),
                saturation=_safe_single_number(p["saturation"]),
                hue=_safe_single_number(p["hue"]),
                p=1.0,
            )

    # ============================== 3-D ===================================== #
    else:
        if name == "flip_horizontal":
            return T.RandomHorizontalFlip3D(p=1.0)
        if name == "flip_vertical":
            return T.RandomVerticalFlip3D(p=1.0)
        if name == "flip_depth":
            for cand in ("RandomDepthFlip3D", "RandomDepthicalFlip3D"):
                if hasattr(T, cand):
                    return getattr(T, cand)(p=1.0)
            raise RuntimeError("Depth-flip transform not found in this Kornia.")

        if name == "rotation":
            return T.RandomRotation3D(
                _tuple_if_range(p["degrees"]), p=1.0, **_interp_kwargs(T.RandomRotation3D, interp)
            )

        if name == "scaling":
            sc = p["scale_limit"]
            sc = _triple(sc) if isinstance(sc, float) else _tuple_if_range(sc)
            return T.RandomAffine3D(degrees=0, scale=sc, p=1.0, **_interp_kwargs(T.RandomAffine3D, interp))

        if name == "translation":
            lim = p["shift_limit"]
            if isinstance(lim, float):
                mag = abs(lim)
            else:
                lo, hi = lim
                mag = max(abs(lo), abs(hi))
            sh = _triple(mag)
            return T.RandomAffine3D(degrees=0, translate=sh, p=1.0,
                                     **_interp_kwargs(T.RandomAffine3D, interp))
        
        if name == "elastic_deformation":
            alp = p["alpha"]; sig = p["sigma"]
            alp = _triple(alp) if isinstance(alp, float) else _tuple_if_range(alp)
            sig = _triple(sig) if isinstance(sig, float) else _tuple_if_range(sig)
            return T.RandomElasticTransform3D(alpha=alp, sigma=sig, p=1.0)

        for nm, cls in [("brightness", T.RandomBrightness), ("contrast", T.RandomContrast), ("gamma", T.RandomGamma)]:
            if name == nm:
                key = "limit" if nm != "gamma" else "gamma_limit"
                return cls(_safe_single_number(p[key]), p=1.0)

        if name == "gaussian_noise":
            return T.RandomGaussianNoise(
                mean=p.get("mean", 0.0), std=_safe_single_number(p["std"]), p=1.0
            )

        if name == "blur":
            k = int(np.clip(p["kernel_size"], 3, 31)) | 1
            sig = _safe_single_number(p.get("sigma", (0.1, 2.0)))
            raw = p.get("sigma", (0.1, 2.0))
            sig = (raw, raw, raw) if isinstance(raw, float) else _tuple_if_range(raw)
            return T.RandomGaussianBlur((k, k, k), sigma=sig, p=1.0)

        if name == "color_jitter":
            return T.ColorJiggle(
                brightness=_safe_single_number(p["brightness"]),
                contrast=_safe_single_number(p["contrast"]),
                saturation=_safe_single_number(p["saturation"]),
                hue=_safe_single_number(p["hue"]),
                p=1.0,
            )

    raise NotImplementedError(f"{name} not handled for {dim}-D")


# -----------------------------------------------------------------------------#
# Public entry                                                                 
# -----------------------------------------------------------------------------#
def augment_images(
    data: Dict[str, torch.Tensor],
    aug_cfg: Dict[str, Any],
    dim: int,
    rng: Optional[np.random.Generator] = None,
    verbose: bool=False
) -> Dict[str, torch.Tensor]:
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() > aug_cfg.get("p", 1.0):
        return data  # skip

    sampled = {k: _maybe_sample(v, rng) for k, v in aug_cfg.get("params", {}).items()}
    if verbose:
        print(f"[augment]{aug_cfg['name']:>18} params={sampled}")

    mods = aug_cfg.get("modalities", list(data.keys()))
    first_mod = next(m for m in mods if m in data)
    base_interp = _select_interp(first_mod)
    base_aug = _make_kornia_transform(aug_cfg["name"], sampled, dim, base_interp)

    out: Dict[str, torch.Tensor] = {}
    cache = None
    for mod, x in data.items():
        if mod not in mods:
            out[mod] = x
            continue

        interp = _select_interp(mod)
        aug = base_aug if interp == base_interp else _make_kornia_transform(
            aug_cfg["name"], sampled, dim, interp
        )
        x_ = x.unsqueeze(0).float()          # Kornia expects B,C,* tensors
        y  = aug(x_, params=cache) if cache is not None else aug(x_)
        cache = aug._params if cache is None else cache
        out[mod] = y.squeeze(0).to(x.dtype)
    return out
