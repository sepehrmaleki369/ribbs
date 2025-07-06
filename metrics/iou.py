import torch
import torch.nn as nn
from typing import Tuple

class ThresholdedIoUMetric(nn.Module):
    r"""
    Compute thresholded Intersection-over-Union for binary **or** multi-class
    predictions in **any** spatial dimensionality.

    Acceptable input shapes
    -----------------------
    * 2-D with channel:  (B, C, H, W)
    * 2-D no channel:   (B, H, W)                → channel added
    * 3-D with channel: (B, C, D, H, W)
    * 3-D no channel:   (B, D, H, W)             → channel added

    Args
    ----
    threshold (float): Binarization cutoff.
    eps (float): Small constant to avoid division by zero.
    multiclass (bool): If *True*, macro-average across channels.
    zero_division (float): IoU to return when both pred & true are empty.
    greater_is_road (bool): If *True*, voxels **above** threshold are “road”
                            (foreground); otherwise “road” = `≤ threshold`.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        eps: float = 1e-6,
        multiclass: bool = False,
        zero_division: float = 1.0,
        greater_is_road: bool = True,
        data_dim: int = 2  # 2D or 3D data
    ):
        super().__init__()
        self.threshold       = float(threshold)
        self.eps             = float(eps)
        self.multiclass      = bool(multiclass)
        self.zero_division   = float(zero_division)
        self.greater_is_road = bool(greater_is_road)
        self.data_dim = int(data_dim)
    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _binarize(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.threshold).float() if self.greater_is_road else (x <= self.threshold).float()
    
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

    # --------------------------------------------------------------------- #
    # forward
    # --------------------------------------------------------------------- #
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.ndim < 3:
            raise ValueError("Expected at least 3 dims: (B, …, spatial)")

        # insert channel if needed
        y_pred = self._ensure_channel(y_pred)
        y_true = self._ensure_channel(y_true)

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")

        N, C = y_pred.shape[:2]
        if not self.multiclass and C != 1:
            raise ValueError(f"Binary mode expects 1 channel, got {C}")

        # binarize
        y_pred_bin = self._binarize(y_pred)
        y_true_bin = self._binarize(y_true)

        # gather all spatial dimensions (2, 3, …)
        spatial_dims: Tuple[int, ...] = tuple(range(2, y_pred_bin.ndim))

        # intersection & union
        inter = (y_pred_bin * y_true_bin).sum(dim=spatial_dims)          # (B, C)
        sum_pred = y_pred_bin.sum(dim=spatial_dims)
        sum_true = y_true_bin.sum(dim=spatial_dims)
        union = sum_pred + sum_true - inter                              # (B, C)

        # IoU with ε for numerical stability
        iou = (inter + self.eps) / (union + self.eps)                    # (B, C)

        # handle empty masks → user-defined value
        zero_mask = union == 0
        if zero_mask.any():
            fill = torch.full_like(iou, self.zero_division)
            iou = torch.where(zero_mask, fill, iou)

        # reduce batch then, optionally, channels
        iou_per_class = iou.mean(dim=0)                                  # (C,)
        return iou_per_class if (self.multiclass and C > 1) else iou_per_class.squeeze(0)



if __name__ == "__main__":
    # Example usage
    metric = ThresholdedIoUMetric(threshold=0.5, multiclass=True)
    y_pred = torch.rand(4, 3, 32, 32)  # Example prediction tensor
    y_true = torch.randint(0, 2, (4, 3, 32, 32))  # Example ground truth tensor
    iou = metric(y_pred, y_true)
    print("IoU per class:", iou)

    metric = ThresholdedIoUMetric(multiclass=False)

    # 2D no-channel → perfect match → IoU == 1
    y2 = torch.rand(2, 32, 32)
    assert metric(y2, y2) == 1.0

    # 2D with channel (explicit C=1)
    y2c = y2.unsqueeze(1)  # shape (2,1,32,32)
    assert metric(y2c, y2c) == 1.0

    # 3D no-channel → perfect match → IoU == 1
    y3 = torch.rand(2, 8, 32, 32)
    assert metric(y3, y3) == 1.0

    # 3D with channel (explicit C=1)
    y3c = y3.unsqueeze(1)  # shape (2,1,8,32,32)
    assert metric(y3c, y3c) == 1.0

    # multiclass mode with C>1
    metric_mc = ThresholdedIoUMetric(multiclass=True)
    y2m = torch.rand(2, 3, 32, 32)
    assert metric_mc(y2m, y2m) == 1.0
