import numpy as np
import torch
from metrics.apls_core import apls
import torch
import torch.nn as nn
import numpy as np

def compute_batch_apls(
    gt_masks,
    pred_masks,
    angle_range=(135, 225),
    max_nodes=500,
    max_snap_dist=4,
    allow_renaming=True,
    min_path_length=10
):
    # --- convert to numpy if needed ---
    if torch.is_tensor(gt_masks):
        gt_masks = gt_masks.detach().cpu().numpy()
    if torch.is_tensor(pred_masks):
        pred_masks = pred_masks.detach().cpu().numpy()

    # --- unify shapes to (B, H, W) ---
    def _unify(m):
        if m.ndim == 2:
            m = m[None, ...]           # (H,W) -> (1,H,W)
        if m.ndim == 3:
            return m                  # already (B,H,W) or (1,H,W)
        if m.ndim == 4 and m.shape[1] == 1:
            return m[:,0,...]        # (B,1,H,W) -> (B,H,W)
        raise ValueError(f"Unsupported mask shape {m.shape}")

    gt = _unify(gt_masks)
    pr = _unify(pred_masks)
    if gt.shape != pr.shape:
        raise ValueError(f"GT shape {gt.shape} != pred shape {pr.shape}")

    B, H, W = gt.shape
    scores = np.zeros(B, dtype=np.float32)

    for i in range(B):
        gt_bin = (gt[i] > 0.5).astype(np.uint8)
        pr_bin = (pr[i] > 0.5).astype(np.uint8)

        # handle empty‐GT cases
        if gt_bin.sum() == 0:
            # perfect if pred is also empty, else zero
            scores[i] = 1.0 if pr_bin.sum() == 0 else 0.0
            continue

        # now GT non‐empty ⇒ delegate to core APLS
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
            # fallback for degenerate graphs
            scores[i] = 0.0

    return scores


class APLS(nn.Module):
    """
    Average Path Length Similarity (APLS) metric for road network segmentation.
    """
    
    def __init__(
        self,
        min_segment_length=10,
        max_nodes=500,
        sampling_ratio=0.1,
        angle_range=(135, 225),
        max_snap_dist=4,
        allow_renaming=True,
    ):
        super().__init__()
        self.min_segment_length = min_segment_length
        self.max_nodes = max_nodes
        self.sampling_ratio = sampling_ratio
        self.angle_range = angle_range
        self.max_snap_dist = max_snap_dist
        self.allow_renaming = allow_renaming
    
    def forward(self, y_pred, y_true):
        """
        Compute the APLS metric between predicted and ground truth road networks.
        """
        scores = compute_batch_apls(
            y_true,
            y_pred,
            angle_range=self.angle_range,
            max_nodes=self.max_nodes,
            max_snap_dist=self.max_snap_dist,
            allow_renaming=self.allow_renaming,
            min_path_length=self.min_segment_length
        )
        
        # Convert numpy array to torch tensor and return mean
        return torch.tensor(scores.mean(), device=y_pred.device)