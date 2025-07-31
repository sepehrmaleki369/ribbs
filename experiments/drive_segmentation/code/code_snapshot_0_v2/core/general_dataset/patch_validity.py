from typing import Any, Dict, List, Optional
import numpy as np
from core.general_dataset.logger import logger


def check_min_thrsh_road(label_patch: np.ndarray, patch_size, threshold) -> bool:
    """
    Check if the label patch has at least a minimum percentage of road pixels.

    Args:
        label_patch (np.ndarray): The label patch.
    
    Returns:
        bool: True if the patch meets the minimum threshold; False otherwise.
    """
    patch = label_patch
    if patch.max() > 1:
        patch = (patch > 127).astype(np.uint8)
    road_percentage = np.sum(patch) / (patch_size * patch_size)
    return road_percentage >= threshold


def check_small_window(image_patch: np.ndarray, small_window_size) -> bool:
    """
    Check that no small window in the image patch is entirely black or white.

    Args:
        image_patch (np.ndarray): Input patch (H x W) or (C x H x W)

    Returns:
        bool: True if valid, False if any window is all black or white.
    """
    sw = small_window_size

    # Ensure image has shape (C, H, W)
    if image_patch.ndim == 2:
        image_patch = image_patch[None, :, :]  # Add channel dimension

    C, H, W = image_patch.shape
    if H < sw or W < sw:
        return False

    # Set thresholds
    max_val = image_patch.max()
    if max_val > 1.0:
        high_thresh = 255
        low_thresh = 0
    else:
        high_thresh = 255 / 255.0
        low_thresh = 0 / 255.0

    # Slide window over spatial dimensions
    for c in range(C):
        for y in range(0, H - sw + 1):
            for x in range(0, W - sw + 1):
                window = image_patch[c, y:y + sw, x:x + sw]
                window_var = np.var(window)
                if window_var < 0.01:
                    return False
                # print(window)
                if np.all(window >= high_thresh):
                    return False  # Found an all-white window
                if np.all(window <= low_thresh):
                    return False  # Found an all-black window

    return True  # All windows passed


