import torch
import torch.nn as nn
import numpy as np
from skimage import measure
from typing import List, Tuple, Set

class ConnectedComponentsQuality(nn.Module):
    """
    Connected Components Quality (CCQ) metric for segmentation.

    Evaluates detection+shape accuracy of connected components in binary
    (or thresholded) predictions vs. ground truth, on 2-D or 3-D volumes.
    """
    def __init__(
        self,
        data_dim: int,
        min_size: int = 5,
        tolerance: float = 2.0,
        alpha: float = 0.5,
        threshold: float = 0.5,
        greater_is_road: bool = True,
        eps: float = 1e-8,
    ):
        """
        Args:
            data_dim: 2 for (H,W) images, 3 for (D,H,W) volumes.
            min_size: minimum component area/volume to keep.
            tolerance: max centroid distance (pixels/voxels) to match.
            alpha: weight [0–1] blending detection vs. shape scores.
            threshold: binarization cutoff.
            greater_is_road: direction of thresholding.
            eps: stability constant.
        """
        super().__init__()
        if data_dim not in (2, 3):
            raise ValueError("data_dim must be 2 or 3")
        self.data_dim        = data_dim
        self.min_size        = int(min_size)
        self.tolerance       = float(tolerance)
        self.alpha           = float(alpha)
        self.threshold       = float(threshold)
        self.greater_is_road = bool(greater_is_road)
        self.eps             = float(eps)

    def _bin(self, arr: np.ndarray) -> np.ndarray:
        return (arr > self.threshold) \
               if self.greater_is_road else (arr <= self.threshold)

    def _ensure_channel(self, t: torch.Tensor) -> torch.Tensor:
        # if (B, H, W) or (B, D, H, W), insert channel at axis=1
        if t.dim() == self.data_dim + 1:
            return t.unsqueeze(1)
        # else assume channel present: (B,1,H,W) or (B,1,D,H,W)
        return t

    def _label_connectivity(self) -> int:
        # skimage: connectivity=2 for 2-D 8-nbr, =1 for full 3-D adjacency
        return 2 if self.data_dim == 2 else 1

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (B, …) logits/prob maps or distance maps.
            y_true: (B, …) ground truth mask or continuous map.
        Returns:
            scalar CCQ score (higher is better).
        """
        # must have at least (B,H,W)
        if y_pred.dim() < self.data_dim + 1 or y_true.dim() < self.data_dim + 1:
            raise ValueError(f"Inputs must be at least {(self.data_dim+1)}-D")
        
        # unify channel: after this both are (B,1,...) 
        y_pred = self._ensure_channel(y_pred)
        y_true = self._ensure_channel(y_true)
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")

        B, C = y_pred.shape[:2]
        if C != 1:
            raise ValueError(f"CCQ only supports binary masks; got C={C}")

        conn = self._label_connectivity()
        scores: List[float] = []

        for b in range(B):
            pred_np = self._bin(y_pred[b,0].detach().cpu().numpy()).astype(np.uint8)
            true_np = self._bin(y_true[b,0].detach().cpu().numpy()).astype(np.uint8)

            # both empty?
            if true_np.sum() == 0:
                scores.append(1.0 if pred_np.sum()==0 else 0.0)
                continue

            # label components
            true_lbl = measure.label(true_np,  connectivity=conn)
            pred_lbl = measure.label(pred_np,  connectivity=conn)

            true_props = [p for p in measure.regionprops(true_lbl)
                          if p.area >= self.min_size]
            pred_props = [p for p in measure.regionprops(pred_lbl)
                          if p.area >= self.min_size]

            # no significant GT
            if not true_props:
                scores.append(1.0 if not pred_props else 0.0)
                continue
            # no significant pred
            if not pred_props:
                scores.append(0.0)
                continue

            # match by centroid
            matches = self._match_components(true_props, pred_props)

            tp = len(matches)
            fp = len(pred_props) - tp
            fn = len(true_props) - tp
            detection = tp / (tp + fp + fn + self.eps)

            # shape: avg IoU over matches
            shape_scores = []
            for t_idx, p_idx in matches:
                t_mask = (true_lbl == true_props[t_idx].label)
                p_mask = (pred_lbl == pred_props[p_idx].label)
                inter = np.logical_and(t_mask, p_mask).sum()
                union = np.logical_or(t_mask, p_mask).sum()
                shape_scores.append(inter / (union + self.eps))
            shape = float(np.mean(shape_scores)) if shape_scores else 0.0

            scores.append(self.alpha * detection + (1-self.alpha) * shape)

        return torch.tensor(float(np.mean(scores)), device=y_pred.device)

    def _match_components(
        self,
        true_props: List[measure._regionprops.RegionProperties],
        pred_props: List[measure._regionprops.RegionProperties],
    ) -> List[Tuple[int,int]]:
        matches: List[Tuple[int,int]] = []
        used_pred: Set[int] = set()
        for t_idx, t_prop in enumerate(true_props):
            tx, ty = t_prop.centroid[:2]
            best_dist, best_idx = float('inf'), None
            for p_idx, p_prop in enumerate(pred_props):
                if p_idx in used_pred:
                    continue
                px, py = p_prop.centroid[:2]
                d = np.hypot(tx-px, ty-py)
                if d <= self.tolerance and d < best_dist:
                    best_dist, best_idx = d, p_idx
            if best_idx is not None:
                matches.append((t_idx, best_idx))
                used_pred.add(best_idx)
        return matches
