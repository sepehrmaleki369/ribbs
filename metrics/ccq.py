import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

__all__ = ["ThresholdedCCQMetric"]

class ThresholdedCCQMetric(nn.Module):
    """
    Relaxed Correctness–Completeness–Quality (CCQ) metric for binary masks.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        slack: int = 3,
        eps: float = 1e-12,
        greater_is_road: bool = True,
        data_dim: int = 2,        # 2 = H,W   · 3 = D,H,W
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.slack = int(slack)
        self.eps = float(eps)
        self.greater_is_road = bool(greater_is_road)
        self.data_dim = int(data_dim)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _binarize(self, x: torch.Tensor) -> np.ndarray:
        """Torch → NumPy Bool with threshold & polarity."""
        x_np = x.detach().cpu().numpy()
        return (x_np > self.threshold) if self.greater_is_road else (x_np <= self.threshold)

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        return skeletonize(mask)

    def _ensure_channel(self, t: torch.Tensor) -> torch.Tensor:
        """Insert missing channel dim → (B,1,...) so we can strip it later."""
        spatial_dims = 2 + self.data_dim      # B + C + spatial
        if t.dim() == spatial_dims - 1:
            return t.unsqueeze(1)
        return t

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> np.ndarray:
        y_pred = self._ensure_channel(y_pred)
        y_true = self._ensure_channel(y_true)

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: pred {y_pred.shape}  vs  true {y_true.shape}")
        if y_pred.shape[1] != 1:
            raise ValueError("CCQ expects binary masks (C = 1)")

        # ---- to NumPy once ----
        pred_np = self._binarize(y_pred)[:, 0]      # (B, … spatial …)
        gt_np   = self._binarize(y_true)[:, 0]

        batch_scores = []

        for p_bin, g_bin in zip(pred_np, gt_np):
            p_bin = self._skeletonize(p_bin.astype(bool))
            g_bin = self._skeletonize(g_bin.astype(bool))

            # Distance maps on the inverse (background == True)
            dist_gt   = ndimage.distance_transform_edt(~g_bin)
            dist_pred = ndimage.distance_transform_edt(~p_bin)

            tp_area = dist_gt   <= self.slack
            fp_area = dist_gt   >  self.slack
            fn_area = dist_pred >  self.slack

            TP = np.logical_and(tp_area, p_bin).sum()
            FP = np.logical_and(fp_area, p_bin).sum()
            FN = np.logical_and(fn_area, g_bin).sum()

            # Empty–empty special-case  → perfect score
            if TP + FP + FN == 0:
                batch_scores.append([1.0, 1.0, 1.0])
                continue

            correctness  = TP / (TP + FP + self.eps)
            completeness = TP / (TP + FN + self.eps)
            quality      = TP / (TP + FP + FN + self.eps)
            batch_scores.append([correctness, completeness, quality])

        return np.mean(batch_scores, axis=0)
