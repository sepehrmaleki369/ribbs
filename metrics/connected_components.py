"""
Connected Components Quality (CCQ) metric for segmentation.

This module provides a metric that evaluates segmentation quality based on
" "the quality of connected components in the prediction compared to ground truth.
" "Supports both binary masks and continuous distance-map outputs via a threshold.
"""

import torch
import torch.nn as nn
import numpy as np
from skimage import measure
from typing import List, Tuple, Set


class ConnectedComponentsQuality(nn.Module):
    """
    Connected Components Quality (CCQ) metric for evaluating segmentation quality.
    
    This metric considers both detection and shape accuracy of connected components
    in the predicted segmentation compared to the ground truth. It supports binary
    outputs as well as continuous-valued maps via a configurable threshold.
    """
    
    def __init__(
        self,
        min_size: int = 5,
        tolerance: int = 2,
        alpha: float = 0.5,
        threshold: float = 0.5,
        greater_is_one=True,
        eps: float = 1e-8,
    ):
        """
        Initialize the CCQ metric.
        
        Args:
            min_size: Minimum component size to consider
            tolerance: Pixel tolerance for component matching
            alpha: Weight between detection score and shape score (0 to 1)
            threshold: Scalar threshold for binarizing predictions and ground truth
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.min_size = min_size
        self.tolerance = tolerance
        self.alpha = alpha
        self.threshold = threshold
        self.eps = eps
        self.greater_is_one = bool(greater_is_one)
    
    def _bin(self, arr: np.ndarray) -> np.ndarray:
        return (arr >  self.threshold) if self.greater_is_one else \
               (arr <  self.threshold)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the CCQ metric between predicted and ground truth masks.
        
        Args:
            y_pred: Predicted maps (B, 1, H, W), e.g., logits, probability maps,
                    or signed/unsigned distance maps
            y_true: Ground truth masks or continuous maps (B, 1, H, W)
        
        Returns:
            Tensor containing the CCQ score (higher is better)
        """
        # Process each item in the batch
        batch_size = y_pred.shape[0]
        scores = []
        
        for i in range(batch_size):
            # Binarize predictions and ground truth via threshold
            pred = self._bin(y_pred[i, 0].detach().cpu().numpy()).astype(np.uint8)
            true = self._bin(y_true[i, 0].detach().cpu().numpy()).astype(np.uint8)
            
            # Skip empty ground truth masks
            if np.sum(true) == 0:
                if np.sum(pred) == 0:
                    scores.append(1.0)  # Both empty - perfect match
                else:
                    scores.append(0.0)  # True empty but pred not - no match
                continue
            
            # Find connected components
            true_labels = measure.label(true, connectivity=2)
            pred_labels = measure.label(pred, connectivity=2)
            
            true_props = measure.regionprops(true_labels)
            pred_props = measure.regionprops(pred_labels)
            
            # Filter out small components
            true_props = [prop for prop in true_props if prop.area >= self.min_size]
            pred_props = [prop for prop in pred_props if prop.area >= self.min_size]
            
            # Handle cases with no significant components
            if not true_props:
                if not pred_props:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
                continue
            if not pred_props:
                scores.append(0.0)
                continue
            
            # Match components
            matches = self._match_components(true_props, pred_props)
            
            # Detection score = TP / (TP + FP + FN)
            tp = len(matches)
            fp = max(0, len(pred_props) - tp)
            fn = max(0, len(true_props) - tp)
            detection_score = tp / (tp + fp + fn + self.eps)
            
            # Shape score = mean IoU of matched components
            shape_scores = []
            for true_idx, pred_idx in matches:
                true_mask = (true_labels == true_props[true_idx].label).astype(np.uint8)
                pred_mask = (pred_labels == pred_props[pred_idx].label).astype(np.uint8)
                intersection = np.sum(true_mask & pred_mask)
                union = np.sum(true_mask | pred_mask)
                iou = intersection / (union + self.eps)
                shape_scores.append(iou)
            
            shape_score = np.mean(shape_scores) if shape_scores else 0.0
            
            # Combined CCQ score
            combined_score = self.alpha * detection_score + (1 - self.alpha) * shape_score
            scores.append(combined_score)
        
        # Return mean score over batch
        return torch.tensor(sum(scores) / batch_size, device=y_pred.device)
    
    def _match_components(
        self,
        true_props: List[object],
        pred_props: List[object]
    ) -> List[Tuple[int, int]]:
        """
        Match predicted components to ground truth components.
        
        This uses a greedy approach based on centroid distance.
        
        Args:
            true_props: List of ground truth region properties
            pred_props: List of predicted region properties
        
        Returns:
            List of (true_idx, pred_idx) matches
        """
        matches = []
        used_pred: Set[int] = set()
        
        for true_idx, true_prop in enumerate(true_props):
            best_dist = float('inf')
            best_pred_idx = None
            true_centroid = true_prop.centroid
            
            for pred_idx, pred_prop in enumerate(pred_props):
                if pred_idx in used_pred:
                    continue
                pred_centroid = pred_prop.centroid
                
                # Calculate Euclidean distancebetween centroids
                dist = np.sqrt(
                    (true_centroid[0] - pred_centroid[0])**2 + 
                    (true_centroid[1] - pred_centroid[1])**2
                )
                if dist < best_dist and dist <= self.tolerance:
                    best_dist = dist
                    best_pred_idx = pred_idx
            
            if best_pred_idx is not None:
                matches.append((true_idx, best_pred_idx))
                used_pred.add(best_pred_idx)
        
        return matches
