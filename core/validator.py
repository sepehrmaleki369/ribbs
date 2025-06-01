# core/validator.py
"""
Validator module for handling chunked inference in validation/test phases.
Implements the exact Road_2D_EEF approach with robust size handling.
"""

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import process_in_chuncks


class Validator:
    """
    Validator class for handling chunked inference during validation/testing.
    Uses the Road_2D_EEF process_in_chunks approach with robust size handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Validator.
        
        Args:
            config: Configuration dictionary with inference parameters such as 
                   patch_size and patch_margin
        """
        self.config = config
        self.patch_size = config.get("patch_size", [512, 512])
        self.patch_margin = config.get("patch_margin", [32, 32])
        self.logger = logging.getLogger(__name__)
        
        # Convert to tuples if provided as lists
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
            
        if isinstance(self.patch_margin, list):
            self.patch_margin = tuple(self.patch_margin)
            
        # Ensure patch_size and patch_margin have the same dimensions
        if len(self.patch_size) != len(self.patch_margin):
            raise ValueError(f"patch_size {self.patch_size} and patch_margin {self.patch_margin} "
                             f"must have the same number of dimensions")
    
    def _pad_to_valid_size(self, image: torch.Tensor, divisor: int = 16) -> tuple:
        """
        Pad image to ensure dimensions are divisible by divisor.
        
        Args:
            image: Input tensor (N, C, H, W)
            divisor: Divisor for dimension constraint (default: 16 for UNet with 3-4 levels)
            
        Returns:
            Tuple of (padded_image, (pad_h, pad_w)) where pad_h and pad_w are padding amounts
        """
        N, C, H, W = image.shape
        
        # Calculate required padding
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor
        
        if pad_h > 0 or pad_w > 0:
            # Pad with reflection to avoid artifacts
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
            # self.logger.debug(f"Padded image from ({H}, {W}) to ({H + pad_h}, {W + pad_w})")
        
        return image, (pad_h, pad_w)
    
    def _remove_padding(self, output: torch.Tensor, padding: tuple) -> torch.Tensor:
        """
        Remove padding from output tensor.
        
        Args:
            output: Padded output tensor (N, C, H, W)
            padding: Tuple of (pad_h, pad_w) padding amounts
            
        Returns:
            Unpadded output tensor
        """
        pad_h, pad_w = padding
        
        if pad_h > 0 or pad_w > 0:
            if pad_h > 0 and pad_w > 0:
                output = output[:, :, :-pad_h, :-pad_w]
            elif pad_h > 0:
                output = output[:, :, :-pad_h, :]
            elif pad_w > 0:
                output = output[:, :, :, :-pad_w]
        
        return output
    
    def run_chunked_inference(
        self,
        model: nn.Module,
        image: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Run inference on a large image by processing it in chunks.
        Pads the full image up to a multiple of 16, then
        splits into overlapping patches with margin, runs each
        chunk through the model, and stitches back together.
        """
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        image = image.to(device)

        # 1) pad the full image so H and W are divisible by 16
        padded_image, (pad_h, pad_w) = self._pad_to_valid_size(image, divisor=16)
        N, C, H, W = padded_image.shape

        # 2) figure out how many output channels the model produces
        #    by running one patch—but *also* pad that patch to 16!
        with torch.no_grad():
            # define a “test patch” of size (patch_size + 2*margin)
            test_h = min(H, self.patch_size[0] + 2 * self.patch_margin[0])
            test_w = min(W, self.patch_size[1] + 2 * self.patch_margin[1])
            test_patch = padded_image[:, :, :test_h, :test_w]
            # pad it so its dims are multiples of 16
            test_patch, _ = self._pad_to_valid_size(test_patch, divisor=16)
            test_out = model(test_patch)
            out_channels = test_out.shape[1]

        # 3) allocate a big “canvas” for all of our outputs
        output = torch.zeros((N, out_channels, H, W), device=device, dtype=test_out.dtype)

        # 4) define a helper that just calls the model
        def _process(chunk: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return model(chunk)

        # 5) chunk & stitch
        with torch.no_grad():
            output = process_in_chuncks(
                padded_image,
                output,
                _process,
                list(self.patch_size),
                list(self.patch_margin),
            )

        # 6) remove exactly the same padding we added at the top
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, : image.shape[2], : image.shape[3]]

        return output
