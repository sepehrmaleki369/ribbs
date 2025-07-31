"""
normalizations.py

Unified normalization utilities that work seamlessly with **both** NumPy
arrays and PyTorch tensors.  

Pass in a NumPy array → you get a NumPy array back.  
Pass in a Torch tensor → you get a Torch tensor back.

All math happens in the backend that the input came from.
"""
from __future__ import annotations

from typing import Optional, Union, List

import numpy as np
import torch
from core.general_dataset.logger import logger

TensorOrArray = Union[np.ndarray, torch.Tensor]

# -----------------------------------------------------------------------------#
# Helper utilities                                                             #
# -----------------------------------------------------------------------------#
def _is_torch(x: TensorOrArray) -> bool:           
    """Return *True* if *x* is a :class:`torch.Tensor`."""
    return isinstance(x, torch.Tensor)


def _validate_input(x: TensorOrArray) -> None:
    """Validate input is a proper numpy array or torch tensor."""
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError(f"Expected numpy array or torch tensor, got {type(x)}")
    
    # Fix bug #1: Properly check for empty tensors/arrays
    if _is_torch(x):
        if x.numel() == 0:
            raise ValueError("Empty arrays/tensors are not supported")
    else:
        if x.size == 0:
            raise ValueError("Empty arrays/tensors are not supported")


def _to_float(x: TensorOrArray) -> TensorOrArray: 
    """Cast to ``float32`` **without** switching backend."""
    _validate_input(x)
    return x.float() if _is_torch(x) else x.astype(np.float32, copy=False)


def _full_like(x: TensorOrArray, value: float) -> TensorOrArray:   
    """Backend-aware ``full_like`` that preserves dtype & device."""
    return torch.full_like(x, value) if _is_torch(x) else np.full_like(x, value)


def _zeros_like(x: TensorOrArray) -> TensorOrArray:   
    """Backend-aware ``zeros_like``."""
    return torch.zeros_like(x) if _is_torch(x) else np.zeros_like(x)


def _clamp(x: TensorOrArray, lo: float, hi: float) -> TensorOrArray:   
    """Backend-aware clamp/clip with identical semantics."""
    return torch.clamp(x, lo, hi) if _is_torch(x) else np.clip(x, lo, hi)


def _quantile(x: TensorOrArray, q: Union[float, List[float]]) -> Union[float, List[float]]:              
    """
    Backend-aware quantile.

    Always returns ``float`` if *q* is scalar or ``list[float]`` if *q* is
    iterable, never tensors/arrays.
    """
    if _is_torch(x):
        # Fix bug #3: Use same dtype as input tensor to avoid dtype mismatch
        if isinstance(q, (list, tuple)):
            q_tensor = torch.tensor(q, dtype=x.dtype, device=x.device)
        else:
            q_tensor = torch.tensor([q], dtype=x.dtype, device=x.device)
        
        qt = torch.quantile(x, q_tensor)
        
        if isinstance(q, (list, tuple)):
            return [float(v) for v in qt.cpu().tolist()]
        else:
            return float(qt.item())

    qt = np.quantile(x, q)
    if isinstance(q, (list, tuple)):
        return [float(v) for v in np.asarray(qt).tolist()]
    else:
        return float(qt)


def _is_binary_data(x: TensorOrArray) -> bool:
    """Check if data is already binary (contains only 0s and 1s)."""
    # Fix bug #2: Add early guard for empty tensors
    if _is_torch(x):
        if x.numel() == 0:
            return False
        unique_vals = torch.unique(x)
        return len(unique_vals) <= 2 and torch.all((unique_vals == 0) | (unique_vals == 1))
    else:
        if x.size == 0:
            return False
        unique_vals = np.unique(x)
        return len(unique_vals) <= 2 and np.all((unique_vals == 0) | (unique_vals == 1))


def _has_nan_or_inf(x: TensorOrArray) -> bool:
    """Check if array/tensor contains NaN or Inf values."""
    if _is_torch(x):
        return torch.isnan(x).any() or torch.isinf(x).any()
    else:
        return np.isnan(x).any() or np.isinf(x).any()


# -----------------------------------------------------------------------------#
# Normalization functions                                                      #
# -----------------------------------------------------------------------------#
def min_max_normalize(
    image: TensorOrArray,
    new_min: float = 0.0,
    new_max: float = 1.0,
    old_min: Optional[float] = None,
    old_max: Optional[float] = None,
) -> TensorOrArray:
    """Rescale intensities to **[new_min, new_max]** while preserving backend."""
    img = _to_float(image)
    
    if _has_nan_or_inf(img):
        logger.warning("Min-Max normalization: input contains NaN or Inf values")
    
    lo = old_min if old_min is not None else float(img.min().item() if _is_torch(img) else img.min())
    hi = old_max if old_max is not None else float(img.max().item() if _is_torch(img) else img.max())

    if hi <= lo:
        logger.error(
            "Min-Max normalization: invalid range old_min=%s, old_max=%s; "
            "returning new_min.", lo, hi
        )
        return _full_like(img, new_min)

    scaled = (img - lo) / (hi - lo)
    return scaled * (new_max - new_min) + new_min


def z_score_normalize(image: TensorOrArray, eps: float = 1e-8) -> TensorOrArray:
    """Subtract mean and divide by std; backend preserved."""
    img = _to_float(image)
    
    if _has_nan_or_inf(img):
        logger.warning("Z-Score normalization: input contains NaN or Inf values")
    
    mean = img.mean()
    std = img.std()
    std_val = float(std.item() if _is_torch(std) else std)

    if std_val < eps:
        logger.error("Z-Score normalization: low variance, returning zeros.")
        return _zeros_like(img)

    return (img - mean) / std


def robust_normalize(
    image: TensorOrArray, lower_q: float = 0.05, upper_q: float = 0.95
) -> TensorOrArray:
    """Clip to quantiles then min-max scale (backend preserved)."""
    img = _to_float(image)
    
    if _has_nan_or_inf(img):
        logger.warning("Robust normalization: input contains NaN or Inf values")
    
    low, high = _quantile(img, [lower_q, upper_q])

    if high == low:
        logger.error("Robust normalization: quantiles equal, returning zeros.")
        return _zeros_like(img)

    clipped = _clamp(img, low, high)
    return (clipped - low) / (high - low)


def percentile_normalize(
    image: TensorOrArray, q_low: float = 1.0, q_high: float = 99.0
) -> TensorOrArray:
    """Percentile wrapper around :func:`robust_normalize`."""
    return robust_normalize(image, q_low / 100.0, q_high / 100.0)


def clip_normalize(
    image: TensorOrArray, min_val: float, max_val: float
) -> TensorOrArray:
    """Clip to ``[min_val,max_val]`` then scale to **[0,1]** (backend preserved)."""
    img = _to_float(image)
    
    if max_val == min_val:
        logger.error("Clip normalization: min_val == max_val, returning zeros.")
        return _zeros_like(img)

    clipped = _clamp(img, min_val, max_val)
    return (clipped - min_val) / (max_val - min_val)


def hard_clip(image: TensorOrArray, min_val: float, max_val: float) -> TensorOrArray:
    """Hard clip to ``[min_val,max_val]`` **without** rescaling (backend preserved)."""
    img = _to_float(image)
    if max_val == min_val:
        logger.error("Hard clip: min_val == max_val, returning constant value.")
        return _full_like(img, min_val)
    return _clamp(img, min_val, max_val)


def divide_by(image: TensorOrArray, threshold: float) -> TensorOrArray:
    """Element-wise division by *threshold* (backend preserved)."""
    if threshold == 0.0:
        raise ValueError("Cannot divide by zero")
    
    img = _to_float(image)
    
    # Fix bug #5: Add NaN/Inf guard like other helpers
    if _has_nan_or_inf(img):
        logger.warning("Divide by: input contains NaN or Inf values")
    
    return img / threshold


def binarize(
    image: TensorOrArray,
    threshold: Union[float, int],
    *,
    greater_is_road: bool = True,
    return_bool: bool = True,
    verbose: bool = False,
) -> TensorOrArray:
    """
    Convert label / probability map to binary mask (**backend preserved**).

    * If the data already looks binary (all values 0/1) it is returned
      as-is (optionally type-converted).
    """
    if not isinstance(image, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f"image must be numpy.ndarray or torch.Tensor, got {type(image)}"
        )

    img = _to_float(image)

    # Fast path for already-binary data
    if _is_binary_data(img):
        if _is_torch(img):
            mask = img.bool()
        else:
            mask = img.astype(bool)
        
        if return_bool:
            return mask
        else:
            return mask.to(torch.uint8) if _is_torch(mask) else mask.astype(np.uint8)

    # Apply threshold
    if greater_is_road:
        mask_bool = img > threshold
    else:
        mask_bool = img <= threshold

    if verbose:
        info = {
            "backend": "torch" if _is_torch(image) else "numpy",
            "dtype": str(image.dtype),
            "shape": tuple(image.shape),
            "min": float(img.min().item() if _is_torch(img) else img.min()),
            "max": float(img.max().item() if _is_torch(img) else img.max()),
            "threshold": float(threshold),
            "greater_is_road": greater_is_road,
        }
        print("[binarize]", info)

    if return_bool:
        return mask_bool
    else:
        return mask_bool.to(torch.uint8) if _is_torch(mask_bool) else mask_bool.astype(np.uint8)


def boolean(image: TensorOrArray) -> TensorOrArray:
    """Cast any numeric array/tensor to **uint8** 0-1 representation."""
    # Fix bug #6: Skip _to_float for boolean input to avoid redundant conversion
    if _is_torch(image):
        if image.dtype == torch.bool:
            return image.to(torch.uint8)
        else:
            img = image.float()
            return img.bool().to(torch.uint8)
    else:
        if image.dtype == bool:
            return image.astype(np.uint8)
        else:
            img = image.astype(np.float32, copy=False)
            return img.astype(bool).astype(np.uint8)


# -----------------------------------------------------------------------------#
# Dispatcher                                                                   #
# -----------------------------------------------------------------------------#
def normalize(
    image: TensorOrArray,
    method: str = "minmax",
    **kwargs,
) -> TensorOrArray:
    """
    Backend-agnostic dispatcher.  
    Returns the **same type** it was given.
    """
    method = method.lower()
    
    # Validate required parameters for each method
    if method == "minmax":
        return min_max_normalize(image, **kwargs)
    elif method == "zscore":
        return z_score_normalize(image, **kwargs)
    elif method == "robust":
        return robust_normalize(image, **kwargs)
    elif method == "percentile":
        return percentile_normalize(image, **kwargs)
    elif method == "clip":
        if 'min_val' not in kwargs or 'max_val' not in kwargs:
            raise ValueError("clip method requires 'min_val' and 'max_val' parameters")
        return hard_clip(image, **kwargs)
    elif method == "clip_normalize":
        if 'min_val' not in kwargs or 'max_val' not in kwargs:
            raise ValueError("clip_normalize method requires 'min_val' and 'max_val' parameters")
        return clip_normalize(image, **kwargs)
    elif method == "divide_by":
        if 'threshold' not in kwargs:
            raise ValueError("divide_by method requires 'threshold' parameter")
        return divide_by(image, **kwargs)
    elif method == "binarize":
        if 'threshold' not in kwargs:
            raise ValueError("binarize method requires 'threshold' parameter")
        return binarize(image, **kwargs)
    elif method == "boolean":
        return boolean(image)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# Back-compat alias
normalize_image = normalize

# Export all public functions
__all__ = [
    'TensorOrArray',
    'min_max_normalize',
    'z_score_normalize', 
    'robust_normalize',
    'percentile_normalize',
    'clip_normalize',
    'hard_clip',  # Fix bug #4: Expose hard_clip publicly
    'divide_by',
    'binarize',
    'boolean',
    'normalize',
    'normalize_image',
]

# Example normalization configurations
normalization_config = {
    "minmax": {
        "method": "minmax",
        "new_min": 0.0,    # lower bound of output range
        "new_max": 1.0     # upper bound of output range
    },
    "zscore": {
        "method": "zscore",
        "eps": 1e-8        # small constant to avoid division by zero
    },
    "robust": {
        "method": "robust",
        "lower_q": 0.05,   # clip everything below 5th quantile
        "upper_q": 0.95    # clip everything above 95th quantile
    },
    "percentile": {
        "method": "percentile",
        "q_low": 1.0,      # clip below 1st percentile
        "q_high": 99.0     # clip above 99th percentile
    },
    "clip": {
        "method": "clip",
        "min_val": 0.0,    # hard clamp lower bound (no scaling)
        "max_val": 200.0   # hard clamp upper bound (no scaling)
    },
    "clip_normalize": {
        "method": "clip_normalize", 
        "min_val": 0.0,    # clamp then scale: lower bound
        "max_val": 200.0   # clamp then scale: upper bound
    },
    "divide_by": {
        "method": "divide_by",
        "threshold": 255.0  # division factor
    },
    "binarize": {
        "method": "binarize",
        "threshold": 0.5,           # binarization threshold
        "greater_is_road": True,    # threshold direction
        "return_bool": True         # return boolean or uint8
    }
}
