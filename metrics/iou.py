import torch
import torch.nn as nn

class ThresholdedIoUMetric(nn.Module):
    """
    Threshold each channel into binary masks, then compute Intersection-over-Union.

    Args:
        threshold (float): cut-off for binarization.
        eps (float): small constant to stabilize non-zero cases.
        multiclass (bool): if True, macro-average over channels.
        zero_division (float): value to return when both pred and true are empty (union=0).
    """
    def __init__(
        self,
        threshold = 0.5,
        eps = 1e-6,
        multiclass = False,
        zero_division = 1.0,
        greater_is_road=True
    ):
        super().__init__()
        self.threshold     = float(threshold)
        self.eps           = float(eps)
        self.multiclass    = bool(multiclass)
        self.zero_division = float(zero_division)
        self.greater_is_road = bool(greater_is_road)

    def _binarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.greater_is_road:
            return (x >  self.threshold).float()
        else:
            return (x <=  self.threshold).float()
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure shape (N, C, H, W)
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)

        N, C, *_ = y_pred.shape
        if not self.multiclass and C != 1:
            raise ValueError(f"[ThresholdedIoUMetric] Binary mode expects 1 channel, got {C}")

        # Binarize
        y_pred_bin = self._binarize(y_pred)
        y_true_bin = self._binarize(y_true)
        
        # Flatten
        y_pred_flat = y_pred_bin.view(N, C, -1)
        y_true_flat = y_true_bin.view(N, C, -1)

        # Intersection and union
        inter = (y_pred_flat * y_true_flat).sum(-1)                    # (N, C)
        sum_pred = y_pred_flat.sum(-1)
        sum_true = y_true_flat.sum(-1)
        union = sum_pred + sum_true - inter                           # (N, C)

        # IoU per sample/class with ε for stability
        iou = (inter + self.eps) / (union + self.eps)                  # (N, C)

        # Handle zero-union explicitly: when union == 0, set to zero_division
        zero_mask = (union == 0)
        if zero_mask.any():
            iou = torch.where(zero_mask,
                              torch.tensor(self.zero_division, device=iou.device),
                              iou)

        # Mean over batch → (C,)
        iou_per_class = iou.mean(0)

        if not self.multiclass or C == 1:
            # Binary: return scalar
            return iou_per_class.squeeze(0)

        # Multiclass: macro-average
        return iou_per_class.mean()
