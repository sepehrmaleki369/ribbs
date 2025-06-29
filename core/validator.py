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
        self.patch_size   = config.get("patch_size",   [512, 512])
        self.patch_margin = config.get("patch_margin", [32,  32])
        self.logger = logging.getLogger(__name__)
        
        # Convert to tuples if provided as lists
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if isinstance(self.patch_margin, list):
            self.patch_margin = tuple(self.patch_margin)
            
        # Ensure patch_size and patch_margin have the same dimensions
        if len(self.patch_size) != len(self.patch_margin):
            raise ValueError(f"patch_size {self.patch_size} and patch_margin "
                             f"{self.patch_margin} must have the same number "
                             f"of dimensions")
    
    # --------------------------------------------------------------------- #
    # helpers                                                               #
    # --------------------------------------------------------------------- #
    def _pad_to_valid_size(self, image: torch.Tensor, divisor: int = 16) -> tuple:
        """
        Pad image to ensure dimensions are divisible by `divisor`.
        
        Returns
        -------
        image        : padded tensor
        (pad_h, pad_w): how much was added on bottom / right
        """
        N, C, H, W = image.shape
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        return image, (pad_h, pad_w)
    
    # --------------------------------------------------------------------- #
    # main entry                                                            #
    # --------------------------------------------------------------------- #
    def run_chunked_inference(
        self,
        model : nn.Module,
        image : torch.Tensor,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Full-image inference with overlapping tiles.

        1) Pad by `patch_margin` on all four sides (reflect).
        2) Pad further so H and W are divisible by 16.
        3) Slide windows of size `patch_size` with stride
           `patch_size – 2*patch_margin`, call `model`, keep only the
           inner (centre) region, and stitch into a canvas.
        4) Remove the /16 pad, then remove the initial margin pad.
        """
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        image = image.to(device)

        # -------------------------------------------------------------- #
        # (A) FIRST pad by the desired margins so borders get context    #
        # -------------------------------------------------------------- #
        mh, mw = self.patch_margin                                   
        if mh or mw:                                                 
            image = F.pad(                                           
                image,                                               
                pad=(mw, mw, mh, mh),  # (left, right, top, bottom)  
                mode="reflect",                                      
            )                                                        

        # -------------------------------------------------------------- #
        # (B) SECOND, pad to make H and W divisible by 16               #
        # -------------------------------------------------------------- #
        padded_image, (pad_h16, pad_w16) = self._pad_to_valid_size(image, 16)
        N, C, Hpad, Wpad = padded_image.shape

        # -------------------------------------------------------------- #
        # (C) Determine #output channels with a dummy forward           #
        # -------------------------------------------------------------- #
        with torch.no_grad():
            test_h = min(Hpad, self.patch_size[0] + 2 * mh)
            test_w = min(Wpad, self.patch_size[1] + 2 * mw)
            test_patch = padded_image[:, :, :test_h, :test_w]
            test_patch, _ = self._pad_to_valid_size(test_patch, 16)
            out_channels = model(test_patch).shape[1]

        # Allocate output canvas (same size as padded_image)
        output = torch.zeros(
            (N, out_channels, Hpad, Wpad),
            device=device,
            dtype=padded_image.dtype,
        )

        # -------------------------------------------------------------- #
        # (D) Sliding-window inference                                  #
        # -------------------------------------------------------------- #
        def _process(chunk: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return model(chunk)
        
        with torch.no_grad():
            output = process_in_chuncks(
                padded_image,
                output,
                _process,
                list(self.patch_size),
                list(self.patch_margin),
            )

        # -------------------------------------------------------------- #
        # (E) Remove the /16 pad                                         #
        # -------------------------------------------------------------- #
        if pad_h16 or pad_w16:
            output = output[:, :, : -pad_h16 if pad_h16 else None,
                                   : -pad_w16 if pad_w16 else None]

        # -------------------------------------------------------------- #
        # (F) Remove the initial margin pad                              #
        # -------------------------------------------------------------- #
        if mh or mw:                                                 
            output = output[:, :, mh : output.shape[2] - mh,         
                                   mw : output.shape[3] - mw]       

        return output
