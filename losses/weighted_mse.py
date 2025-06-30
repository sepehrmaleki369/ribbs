import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """
    Per-pixel MSE with class-dependent weights (e.g. give roads > background).

    Args
    ----
    road_weight : float
        Weight applied to squared errors on road pixels.
    bg_weight   : float
        Weight applied to squared errors on background pixels.
    threshold   : float
        Threshold that separates “road” from “background” in the SDF.
    greater_is_road : bool
        If True, pixels **>= threshold** are treated as road.
        If False, pixels **<  threshold** are treated as road (default for SDF where roads are negative).
    reduction : {'mean', 'sum', 'none'}
        • 'mean' – divide the *weighted* SSE by the total number of elements  
          (so when both weights are 1 you recover standard MSE, and
          changing the class weights doesn’t blow up the loss scale).  
        • 'sum'  – return the weighted sum of squared errors.  
        • 'none' – return the full per-pixel tensor.
    """

    def __init__(
        self,
        road_weight: float = 5.0,
        bg_weight:   float = 1.0,
        threshold:   float = 0.0,
        greater_is_road: bool = False,
        reduction: str = "mean",
    ):
        super().__init__()
        self.road_weight   = float(road_weight)
        self.bg_weight     = float(bg_weight)
        self.threshold     = float(threshold)
        self.greater_is_road = bool(greater_is_road)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction     = reduction

    # ------------------------------------------------------------------ #
    # forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(self, y_pred: torch.Tensor, y_true_sdf: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred      : Tensor (N, 1, H, W)  – model output SDF
        y_true_sdf  : Tensor (N, 1, H, W)  – ground-truth signed-distance map

        Returns
        -------
        Tensor
            A scalar (for 'mean' / 'sum') or a tensor shaped like the input (for 'none').
        """

        # 1) build per-pixel weights
        if self.greater_is_road:
            is_road = (y_true_sdf > self.threshold)
        else:
            is_road = (y_true_sdf <=  self.threshold)
        weight = torch.where(is_road, self.road_weight, self.bg_weight).to(y_pred.dtype)

        # 2) weighted squared error
        wse = (y_pred - y_true_sdf) ** 2 * weight

        # 3) reduction
        if self.reduction == "mean":
            # divide by the *number of elements* so scale matches plain MSE
            return wse.sum() / y_pred.numel()
        elif self.reduction == "sum":
            return wse.sum()
        else:                       # 'none'
            return wse
