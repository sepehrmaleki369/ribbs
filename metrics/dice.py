import torch
import torch.nn as nn

class ThresholdedDiceMetric(nn.Module):
    """
    Threshold each channel into a binary mask, then compute standard Dice.

    Args:
        threshold (float or str): cut‐off for binarization.
        eps (float or str): small constant to avoid zero‐division.
        multiclass (bool or str): if True, macro‐average over channels.
        zero_division (float or str): returned value when both pred & GT empty.
    """
    def __init__(
        self,
        threshold=0.5,
        eps=1e-6,
        multiclass=False,
        zero_division=1.0,
        greater_is_one=True
    ):
        super().__init__()
        # Force numerical types
        self.threshold     = float(threshold)
        self.eps           = float(eps)
        self.multiclass    = bool(multiclass)
        self.zero_division = float(zero_division)
        self.greater_is_one = bool(greater_is_one)

    def _binarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.greater_is_one:
            return (x >  self.threshold).float()
        else:
            return (x <  self.threshold).float()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # ensure (N, C, H, W)
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)

        N, C, *_ = y_pred.shape
        if not self.multiclass and C != 1:
            raise ValueError(f"[ThresholdedDiceMetric] Binary mode expects 1 channel, got {C}")

        # binarize
        y_pred_bin = self._binarize(y_pred)
        y_true_bin = self._binarize(y_true)
        # flatten
        y_pred_flat = y_pred_bin.view(N, C, -1)
        y_true_flat = y_true_bin.view(N, C, -1)

        # intersection and sums
        inter = (y_pred_flat * y_true_flat).sum(-1)           # (N, C)
        sums  = y_pred_flat.sum(-1) + y_true_flat.sum(-1)     # (N, C)

        # build eps and zero_division as tensors
        device = y_pred.device
        eps_tensor = torch.tensor(self.eps, device=device, dtype=inter.dtype)
        zd_tensor = torch.tensor(self.zero_division, device=device, dtype=inter.dtype)

        # dice per sample/class with ε for stability
        dice = (2 * inter + eps_tensor) / (sums + eps_tensor)     # (N, C)

        # override exact-zero cases
        zero_mask = (sums == 0)
        if zero_mask.any():
            dice = torch.where(zero_mask, zd_tensor, dice)

        # mean over batch → (C,)
        dice_per_class = dice.mean(0)

        if not self.multiclass or C == 1:
            return dice_per_class.squeeze(0)

        return dice_per_class.mean()
