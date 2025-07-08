"""
normalizations.py

Various image normalization routines for 2D/3D numpy arrays.
Define multiple strategies and a unified `normalize` dispatcher, plus a binarization helper.
"""
import numpy as np
from core.general_dataset.logger import logger
from typing import Optional
import numpy as np
from typing import Union

def min_max_normalize(
    image: np.ndarray,
    new_min: float = 0.0,
    new_max: float = 1.0,
    old_min: Optional[float] = None,
    old_max: Optional[float] = None,
) -> np.ndarray:
    """
    Scale image intensities to [new_min, new_max].
    If old_min/old_max are provided, use them as the original bounds;
    otherwise compute from the image itself.
    """
    img = image.astype(np.float32)
    lo = old_min if old_min is not None else float(img.min())
    hi = old_max if old_max is not None else float(img.max())
    if hi <= lo:
        logger.warning(
            "Min-Max normalization: invalid range old_min=%s, old_max=%s; returning new_min.",
            lo, hi
        )
        return np.full_like(img, new_min)
    scaled = (img - lo) / (hi - lo)
    return scaled * (new_max - new_min) + new_min


def z_score_normalize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Subtract mean and divide by standard deviation (per-image).
    """
    img = image.astype(np.float32)
    mean = img.mean()
    std = img.std()
    if std < eps:
        logger.warning("Z-Score normalization: low variance, returning zeros.")
        return np.zeros_like(img)
    return (img - mean) / std


def robust_normalize(image: np.ndarray, lower_q: float = 0.05, upper_q: float = 0.95) -> np.ndarray:
    """
    Clip intensities to [lower_q, upper_q] quantiles and then apply min-max scaling.
    """
    img = image.astype(np.float32)
    low, high = np.quantile(img, [lower_q, upper_q])
    if high == low:
        logger.warning("Robust normalization: quantiles equal, returning zeros.")
        return np.zeros_like(img)
    clipped = np.clip(img, low, high)
    return (clipped - low) / (high - low)


def percentile_normalize(image: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    """
    Similar to robust but in percent (q_low and q_high are percents).
    Clip to percentiles then min-max scale to [0,1].
    """
    return robust_normalize(image, q_low/100.0, q_high/100.0)


def clip_normalize(image: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Clip intensities to [min_val, max_val] then scale to [0,1].
    """
    img = image.astype(np.float32)
    clipped = np.clip(img, min_val, max_val)
    if max_val == min_val:
        logger.warning("Clip normalization: min_val == max_val, returning zeros.")
        return np.zeros_like(img)
    return (clipped - min_val) / (max_val - min_val)

def clip_(image: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Clip intensities to [min_val, max_val] then scale to [0,1].
    """
    img = image.astype(np.float32)
    clipped = np.clip(img, min_val, max_val)
    if max_val == min_val:
        logger.warning("Clip normalization: min_val == max_val, returning zeros.")
        return np.zeros_like(img)
    return clipped

def divide_by(image: np.ndarray, threshold: float) -> np.ndarray:
    img = image / threshold
    return img


def binarize(
    image: np.ndarray,
    threshold: Union[float, int],
    *,
    greater_is_road: bool = True,
    return_bool: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Convert a label / probability map to a binary road mask.

    Parameters
    ----------
    image : np.ndarray
        Input array (any dtype convertible to ``np.float32``). Can be 2‑D or 3‑D; the
        function is applied element‑wise so channels / depth are preserved.
    threshold : float or int
        Threshold value.  Pixels **strictly greater** than the threshold become
        road (1) if ``greater_is_road`` is ``True``; otherwise the inequality is
        reversed.
    greater_is_road : bool, default=True
        Direction of the inequality: if *True*  → ``mask = img > thr``  else
        ``mask = img <= thr``.
    return_bool : bool, default=False
        If *True* the mask is returned as ``bool``; otherwise ``uint8`` (0/1).
    verbose : bool, default=False
        Print a small Rich table with min/max, dtype, chosen threshold, etc.  If
        the *rich* library is not installed this falls back to plain ``print``.

    Returns
    -------
    np.ndarray
        Binary mask of the same shape as *image* and dtype ``bool`` or
        ``uint8``.

    Notes
    -----
    * If *image* already looks binary (all values in {0,1}), it is returned as
     ‑is (with optional dtype conversion).
    * ``threshold`` is **not** auto‑scaled; pass in a value consistent with the
      actual range of the array.
    """

    # ------------------------------------------------------------
    # 0. Type & range sanity‑check
    # ------------------------------------------------------------
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image)}")

    if image.dtype.kind not in "iu":  # not int/uint/float
        raise TypeError(
            "Unsupported dtype: {} (expected int/uint/float types)".format(image.dtype)
        )

    # Identity fast‑path: already binary
    if image.min() >= 0 and image.max() <= 1:
        mask_bool = image.astype(bool) if image.dtype != bool else image  # cheap
        return mask_bool if return_bool else mask_bool.astype(np.uint8)

    # ------------------------------------------------------------
    # 1. Thresholding
    # ------------------------------------------------------------
    img_f32 = image.astype(np.float32, copy=False)
    if greater_is_road:
        mask_bool = img_f32 > threshold
    else:
        mask_bool = img_f32 <= threshold

    # ------------------------------------------------------------
    # 2. Optional pretty print with Rich
    # ------------------------------------------------------------
    if verbose:
        msg = {
            "dtype": str(image.dtype),
            "shape": image.shape,
            "min": float(image.min()),
            "max": float(image.max()),
            "threshold": float(threshold),
            "greater_is_road": greater_is_road,
        }
        print("[binarize]", msg)

    # ------------------------------------------------------------
    # 3. Dtype of the output
    # ------------------------------------------------------------
    if return_bool:
        return mask_bool
    return mask_bool.astype(np.uint8)


def normalize(
    image: np.ndarray,
    method: str = "minmax",
    **kwargs
) -> np.ndarray:
    # print('image.min(), image.max() before norm', image.min(), image.max())
    """
    Dispatch to a normalization method. Supported methods:
      - 'minmax'    : min_max_normalize
      - 'zscore'    : z_score_normalize
      - 'robust'    : robust_normalize
      - 'percentile': percentile_normalize
      - 'clip'      : clip_normalize (requires min_val, max_val args)
      - 'binarize'  : binarize (requires threshold, optional greater_is_road)
    """
    method = method.lower()
    if method == "minmax":
        return min_max_normalize(image, **kwargs)
    elif method == "zscore":
        return z_score_normalize(image, **kwargs)
    elif method == "robust":
        return robust_normalize(image, **kwargs)
    elif method == "percentile":
        return percentile_normalize(image, **kwargs)
    elif method == "clip":
        return clip_(image, **kwargs)
    elif method == "divide_by":
        return divide_by(image, **kwargs)
    elif method == "clip_normalize":
        return clip_normalize(image, **kwargs)
    elif method == "binarize":
        return binarize(image, **kwargs)
    else:
        logger.error(f"Unknown normalization method '{method}', using minmax.")
        return min_max_normalize(image, **kwargs)

# Alias for backward compatibility
normalize_image = normalize



# normalization_config = {
#     "minmax": {
#         "method": "minmax",
#         "new_min": 0.0,    # lower bound of output range
#         "new_max": 1.0     # upper bound of output range
#     },
#     "zscore": {
#         "method": "zscore",
#         "eps": 1e-8        # small constant to avoid division by zero
#     },
#     "robust": {
#         "method": "robust",
#         "lower_q": 0.05,   # clip everything below 5th quantile
#         "upper_q": 0.95    # clip everything above 95th quantile
#     },
#     "percentile": {
#         "method": "percentile",
#         "q_low": 1.0,      # clip below 1st percentile
#         "q_high": 99.0     # clip above 99th percentile
#     },
#     "clip": {
#         "method": "clip",
#         "min_val": 0.0,    # lower clipping threshold
#         "max_val": 200.0   # upper clipping threshold
#     }
# }