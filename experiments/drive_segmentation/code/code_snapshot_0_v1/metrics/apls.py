import numpy as np
import torch
import torch.nn as nn
from metrics.apls_core import apls
from typing import Tuple

class APLS(nn.Module):
    """
    Average Path Length Similarity (APLS) metric for road–network segmentation.

    Works on 2-D (H,W) or 3-D (D,H,W) masks, with or without an explicit channel.

    Args:
        data_dim: 2 for 2-D masks, 3 for 3-D volumes.
        threshold: cutoff to binarize predictions.
        angle_range: passed to apls().
        max_nodes: passed to apls().
        max_snap_dist: passed to apls().
        allow_renaming: passed to apls().
        min_path_length: passed to apls().
        greater_is_road: direction of thresholding.
    """
    def __init__(
        self,
        data_dim: int = 2,
        threshold: float = 0.5,
        angle_range: Tuple[int,int] = (135, 225),
        max_nodes: int = 500,
        max_snap_dist: int = 4,
        allow_renaming: bool = True,
        min_path_length: int = 10,
        greater_is_road: bool = True,
    ):
        super().__init__()
        if data_dim not in (2, 3):
            raise ValueError("data_dim must be 2 or 3")
        self.data_dim       = data_dim
        self.threshold      = float(threshold)
        self.angle_range    = angle_range
        self.max_nodes      = int(max_nodes)
        self.max_snap_dist  = int(max_snap_dist)
        self.allow_renaming = bool(allow_renaming)
        self.min_path_length= int(min_path_length)
        self.greater_is_road= bool(greater_is_road)

    def _binarize(self, arr: np.ndarray) -> np.ndarray:
        return (arr > self.threshold).astype(np.uint8) \
               if self.greater_is_road else (arr <= self.threshold).astype(np.uint8)

    def _unify(self, m: torch.Tensor) -> np.ndarray:
        """
        Turn inputs of shape
          - (H,W), (1,H,W), (B,H,W), (B,1,H,W)  [2-D] or
          - (D,H,W), (1,D,H,W), (B,D,H,W), (B,1,D,H,W) [3-D]
        into a numpy array of shape (B, *spatial).
        """
        d = self.data_dim
        shp = m.shape
        # collapse channel if present
        if m.dim() == d + 2 and shp[1] == 1:
            arr = m[:, 0, ...]
        else:
            arr = m
        # ensure batch
        if arr.dim() == d:
            arr = arr[None, ...]
        return arr.detach().cpu().numpy()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # unify shapes → (B, *spatial)
        gt = self._unify(y_true)
        pr = self._unify(y_pred)
        if gt.shape != pr.shape:
            raise ValueError(f"GT {gt.shape} vs pred {pr.shape}")

        B = gt.shape[0]
        scores = np.zeros(B, dtype=np.float32)

        for i in range(B):
            gt_bin = self._binarize(gt[i])
            pr_bin = self._binarize(pr[i])

            # both empty
            if gt_bin.sum() == 0:
                scores[i] = 1.0 if pr_bin.sum()==0 else 0.0
                continue

            try:
                scores[i] = apls(
                    gt_bin,
                    pr_bin,
                    angle_range=self.angle_range,
                    max_nodes=self.max_nodes,
                    max_snap_dist=self.max_snap_dist,
                    allow_renaming=self.allow_renaming,
                    min_path_length=self.min_path_length
                )
            except Exception:
                scores[i] = 0.0

        return torch.tensor(float(scores.mean()), device=y_pred.device)
