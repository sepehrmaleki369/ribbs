"""
Mixed loss module that switches between primary and secondary loss functions based on epoch.

This module provides functionality to mix multiple loss functions with configurable weights
and a switchover epoch for the secondary loss.
"""

from typing import Any, Dict, List, Optional, Union, Callable

import torch
import torch.nn as nn


class MixedLoss(nn.Module):
    """
    A loss module that mixes multiple loss functions with configurable weights.
    
    The secondary loss will only be activated after a specified epoch.
    
    Attributes:
        primary_loss: The primary loss function (used from epoch 0)
        secondary_loss: The secondary loss function (activated after start_epoch)
        alpha: Weight for the secondary loss (between 0 and 1)
        start_epoch: Epoch to start using the secondary loss
        current_epoch: Current training epoch (updated externally)
    """
    
    def __init__(
        self, 
        primary_loss: Union[nn.Module, Callable], 
        secondary_loss: Optional[Union[nn.Module, Callable]] = None,
        alpha: float = 0.5,
        start_epoch: int = 0
    ):
        """
        Initialize the MixedLoss module.
        
        Args:
            primary_loss: The primary loss function (used from epoch 0)
            secondary_loss: The secondary loss function (used after start_epoch)
            alpha: Weight for the secondary loss (between 0 and 1)
            start_epoch: Epoch to start using the secondary loss
        """
        super().__init__()
        self.primary_loss = primary_loss
        self.secondary_loss = secondary_loss
        self.alpha = alpha
        self.start_epoch = start_epoch
        self.current_epoch = 0
    
    def update_epoch(self, epoch: int) -> None:
        """
        Update the current epoch.
        
        Args:
            epoch: The current epoch number
        """
        self.current_epoch = epoch
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate the mixed loss.
        
        Args:
            y_pred: Predicted values
            y_true: Ground truth values
            
        Returns:
            Tensor containing the calculated loss
        """
        p = self.primary_loss(y_pred, y_true)
        if self.secondary_loss is None or self.current_epoch < self.start_epoch:
            s = torch.tensor(0.0, device=p.device, dtype=p.dtype)
        else:
            s = self.secondary_loss(y_pred, y_true)
        m = (1 - self.alpha) * p + self.alpha * s
        return {"primary": p, "secondary": s, "mixed": m}