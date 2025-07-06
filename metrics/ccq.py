import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

__all__ = ["ThresholdedCCQMetric"]

class ThresholdedCCQMetric(nn.Module):
    """
    Compute Relaxed Correctness, Completeness, and Quality (CCQ) for binary predictions.

    Args:
        threshold (float): cutoff for binarization (default: 0.5).
        slack (int): maximum pixel distance for a true positive (default: 3).
        eps (float): small constant to avoid division by zero (default: 1e-12).
        greater_is_road (bool): if True, values > threshold are foreground.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        slack: int = 3,
        eps: float = 1e-12,
        greater_is_road: bool = True,
        data_dim: int = 2  # 2D or 3D data
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.slack = int(slack)
        self.eps = float(eps)
        self.greater_is_road = bool(greater_is_road)
        self.data_dim = int(data_dim)

    def _binarize(self, x: torch.Tensor) -> np.ndarray:
        """
        Binarize tensor according to threshold.
        """
        if self.greater_is_road:
            return np.array(x.detach().cpu().numpy() > self.threshold)
        else:
            return np.array(x.detach().cpu().numpy() <= self.threshold)

    def _skeletonize(self, x_bin: np.ndarray) -> np.ndarray:
        """
        Apply binary skeletonization to the input tensor.
        This is a placeholder; actual implementation may vary.
        """
        x_bin = skeletonize(x_bin.astype(bool))
        if self.data_dim == 2:  # skeletonize the binary mask
            return x_bin
        elif self.data_dim == 3:
            return x_bin // 255
        return x_bin

    def _ensure_channel(self, t: torch.Tensor) -> torch.Tensor:
        """
        Ensure a channel dimension for 2D or 3D data.
        For 2D: (B,H,W) -> (B,1,H,W)
        For 3D: (B,D,H,W) -> (B,1,D,H,W)
        Leaves (B,1,H,W) or (B,1,D,H,W) unchanged.
        """
        expected_dim = 2 + self.data_dim  # batch + channel + spatial
        if t.dim() == expected_dim - 1:
            # Missing channel dimension
            return t.unsqueeze(1)
        return t

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> np.ndarray:
        """
        Returns a tensor of shape (3,) with [correctness, completeness, quality]
        averaged over the batch.
        """
        # Binarize inputs
        y_pred = self._ensure_channel(y_pred)
        y_true = self._ensure_channel(y_true)

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: pred {y_pred.shape} vs true {y_true.shape}")

        # Only binary mode: expect C=1
        if y_pred.shape[1] != 1:
            raise ValueError(f"CCQMetric supports binary masks only (C=1), got C={y_pred.shape[1]}")

        # Move to CPU numpy for distance transform
        p_np = (self._binarize(y_pred)[:, 0]).astype(bool)  # (B, ...) spatial
        g_np = (self._binarize(y_true)[:, 0]).astype(bool)

        batch_metrics = []
        for p_bin, g_bin in zip(p_np, g_np):
            g_bin = self._skeletonize(g_bin)  # skeletonize ground truth if needed
            p_bin = self._skeletonize(p_bin)  # skeletonize prediction if needed

            # print(f"p_bin shape: {p_bin.shape}, g_bin shape: {g_bin.shape}")
            # print(f"p_bin max: {p_bin.max()}, g_bin max: {g_bin.max()}")
            # distance maps
            dist_gt = ndimage.distance_transform_edt(~g_bin)
            dist_pred = ndimage.distance_transform_edt(~p_bin)
            
            # define areas
            tp_area = dist_gt <= self.slack
            fp_area = dist_gt > self.slack
            fn_area = dist_pred > self.slack

            # counts
            TP = np.logical_and(tp_area, p_bin).sum()
            FP = np.logical_and(fp_area, p_bin).sum()
            FN = np.logical_and(fn_area, g_bin).sum()

            # compute metrics
            correctness = TP / (TP + FP + self.eps)
            completeness = TP / (TP + FN + self.eps)
            quality = TP / (TP + FP + FN + self.eps)

            batch_metrics.append([correctness, completeness, quality])

        # average over batch
        avg = np.mean(batch_metrics, axis=0)
        return avg
