import torch
import torch.nn as nn
from typing import Tuple

class ThresholdedDiceMetric(nn.Module):
    r"""
    Compute thresholded Dice coefficient for binary **or** multi-class
    predictions in **any** spatial dimensionality.

    Acceptable input shapes
    -----------------------
    * 2-D with channel:  (B, C, H, W)
    * 2-D no channel:   (B, H, W)                → channel added
    * 3-D with channel: (B, C, D, H, W)
    * 3-D no channel:   (B, D, H, W)             → channel added

    Args
    ----
    threshold (float): cut-off for binarization.
    eps (float): small constant to avoid division by zero.
    multiclass (bool): if True, macro-average over channels.
    zero_division (float): value to return when both pred & GT are empty.
    greater_is_road (bool): if True, values > threshold are “positive”.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        eps: float = 1e-6,
        multiclass: bool = False,
        zero_division: float = 1.0,
        greater_is_road: bool = True,
    ):
        super().__init__()
        self.threshold      = float(threshold)
        self.eps            = float(eps)
        self.multiclass     = bool(multiclass)
        self.zero_division  = float(zero_division)
        self.greater_is_road = bool(greater_is_road)

    def _binarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.greater_is_road:
            return (x > self.threshold).float()
        else:
            return (x <= self.threshold).float()

    def _ensure_channel(self, t: torch.Tensor) -> torch.Tensor:
        # (B, H, W) → (B,1,H,W); (B, D, H, W) → (B,1,D,H,W)
        if t.dim() == 3:
            return t.unsqueeze(1)
        if t.dim() == 4:
            # could be (B, C, H, W) or (B, D, H, W)
            # if binary mode & C==1, assume it's already channel
            if not self.multiclass and t.shape[1] == 1:
                return t
            return t.unsqueeze(1)
        # dims == 5: (B, C, D, H, W) already good
        return t

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # must have at least (B, H, W)
        if y_pred.dim() < 3 or y_true.dim() < 3:
            raise ValueError("Expected inputs with at least 3 dims: (B, …, spatial)")
        # add channel if missing
        y_pred = self._ensure_channel(y_pred)
        y_true = self._ensure_channel(y_true)

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: pred {y_pred.shape} vs true {y_true.shape}")

        N, C = y_pred.shape[:2]
        if not self.multiclass and C != 1:
            raise ValueError(f"Binary mode expects C=1 channel, got C={C}")

        # threshold
        p = self._binarize(y_pred)
        g = self._binarize(y_true)

        # sum over all spatial dims
        spatial = tuple(range(2, p.dim()))
        inter = (p * g).sum(dim=spatial)           # (B, C)
        sums  = p.sum(dim=spatial) + g.sum(dim=spatial)  # (B, C)

        # stable dice
        eps_t = torch.tensor(self.eps, device=inter.device, dtype=inter.dtype)
        zd_t  = torch.tensor(self.zero_division, device=inter.device, dtype=inter.dtype)
        dice  = (2*inter + eps_t) / (sums + eps_t)       # (B, C)

        # handle empty
        empty = (sums == 0)
        if empty.any():
            dice = torch.where(empty, zd_t, dice)

        # reductions
        per_class = dice.mean(dim=0)            # → (C,)
        if self.multiclass and C > 1:
            return per_class.mean()             # macro-avg over classes
        return per_class.squeeze(0)             # scalar for binary

if __name__ == "__main__":
    # Example usage
    metric = ThresholdedDiceMetric(threshold=0.5, multiclass=True)
    y_pred = torch.rand(4, 3, 64, 64)  # Example prediction (B, C, H, W)
    y_true = torch.randint(0, 2, (4, 3, 64, 64))  # Example ground truth (B, C, H, W)
    
    dice_score = metric(y_pred, y_true)
    print("Dice Score:", dice_score)

    metric = ThresholdedDiceMetric(multiclass=False)

    # 2D no-channel
    y2 = torch.rand(2, 32, 32)
    assert torch.isclose(metric(y2, y2), torch.tensor(1.0))

    # 2D with channel
    y2c = y2.unsqueeze(1)
    assert torch.isclose(metric(y2c, y2c), torch.tensor(1.0))

    # 3D no-channel
    y3 = torch.rand(2, 8, 32, 32)
    assert torch.isclose(metric(y3, y3), torch.tensor(1.0))

    # 3D with channel
    y3c = y3.unsqueeze(1)
    assert torch.isclose(metric(y3c, y3c), torch.tensor(1.0))

    # multiclass
    metric_mc = ThresholdedDiceMetric(multiclass=True)
    y_mc = torch.rand(2, 3, 16, 16)
    assert torch.isclose(metric_mc(y_mc, y_mc), torch.tensor(1.0))
