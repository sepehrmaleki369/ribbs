"""
Validator module for handling chunked inference in validation/test phases.

This module provides functionality to perform validation with full-size images
by processing them in patches with overlap.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import the existing process_in_chuncks function
from core.utils import process_in_chuncks


class Validator:
    """
    Validator class for handling chunked inference during validation/testing.
    
    This class enables processing large images that don't fit in GPU memory by
    splitting them into overlapping chunks, processing each chunk, and reassembling
    the results with proper handling of overlapping regions.
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
    
    def run_chunked_inference(
        self, 
        model: nn.Module, 
        image: torch.Tensor, 
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Run inference on a large image by processing it in chunks.
        
        Args:
            model: The model to use for inference
            image: Input image tensor (N, C, H, W)
            device: Device to run inference on (default: None, uses model's device)
            
        Returns:
            Output tensor with predictions for the full image
        """
        if device is None:
            device = next(model.parameters()).device
            
        model.eval()
        image = image.to(device)
        
        # Get image dimensions
        N, C, H, W = image.shape
        
        # Create empty output tensor (assuming model output has same spatial dimensions as input)
        # We need to run a test inference to get the output channel dimension
        with torch.no_grad():
            # Create a small test patch to determine output channels
            test_patch = image[:, :, :min(H, self.patch_size[0]), :min(W, self.patch_size[1])]
            test_output = model(test_patch)
            out_channels = test_output.shape[1]
            
        # Initialize output tensor (N, out_channels, H, W)
        output = torch.zeros((N, out_channels, H, W), device=device)
        
        # Process function to wrap model inference
        def process_fn(chunk):
            with torch.no_grad():
                return model(chunk)
        
        # Run chunked inference
        output = process_in_chuncks(image, output, process_fn, self.patch_size, self.patch_margin)
        
        return output