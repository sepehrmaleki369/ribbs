import numpy as np
import torch
import torch.nn as nn
from metrics.apls_core import apls


def compute_batch_apls(
    gt_masks,
    pred_masks,
    threshold: float = 0.5,
    angle_range=(135, 225),
    max_nodes=500,
    max_snap_dist=4,
    allow_renaming=True,
    min_path_length=10,
    greater_is_one=True
):
    def _bin(x): return (x > threshold) if greater_is_one else (x < threshold)

    # --- convert to numpy if needed ---
    if torch.is_tensor(gt_masks):
        gt_masks = gt_masks.detach().cpu().numpy()
    if torch.is_tensor(pred_masks):
        pred_masks = pred_masks.detach().cpu().numpy()

    # --- unify shapes to (B, H, W) ---
    def _unify(m):
        if m.ndim == 2:
            m = m[None, ...]
        if m.ndim == 3:
            return m
        if m.ndim == 4 and m.shape[1] == 1:
            return m[:,0,...]
        raise ValueError(f"Unsupported mask shape {m.shape}")

    gt = _unify(gt_masks)
    pr = _unify(pred_masks)
    if gt.shape != pr.shape:
        raise ValueError(f"GT shape {gt.shape} != pred shape {pr.shape}")

    B = gt.shape[0]
    scores = np.zeros(B, dtype=np.float32)

    for i in range(B):
        # Binarize with threshold
        gt_bin = _bin(gt[i]).astype(np.uint8)
        pr_bin = _bin(pr[i]).astype(np.uint8)

        # Skip empty ground truth
        if gt_bin.sum() == 0:
            scores[i] = 1.0 if pr_bin.sum() == 0 else 0.0
            continue

        try:
            scores[i] = apls(
                gt_bin,
                pr_bin,
                angle_range=angle_range,
                max_nodes=max_nodes,
                max_snap_dist=max_snap_dist,
                allow_renaming=allow_renaming,
                min_path_length=min_path_length
            )
        except Exception:
            scores[i] = 0.0

    return scores


class APLS(nn.Module):
    """
    Average Path Length Similarity (APLS) metric for road network segmentation.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        angle_range=(135, 225),
        max_nodes=500,
        max_snap_dist=4,
        allow_renaming=True,
        min_path_length=10,
        greater_is_one=True
    ):
        super().__init__()
        self.threshold = threshold
        self.angle_range = angle_range
        self.max_nodes = max_nodes
        self.max_snap_dist = max_snap_dist
        self.allow_renaming = allow_renaming
        self.min_path_length = min_path_length
        self.greater_is_one = bool(greater_is_one)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        scores = compute_batch_apls(
            y_true,
            y_pred,
            threshold=self.threshold,
            angle_range=self.angle_range,
            max_nodes=self.max_nodes,
            max_snap_dist=self.max_snap_dist,
            allow_renaming=self.allow_renaming,
            min_path_length=self.min_path_length,
            greater_is_one=self.greater_is_one,
        )
        return torch.tensor(scores.mean(), device=y_pred.device)
