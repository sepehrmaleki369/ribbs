import torch
import torch.nn as nn

class FixedLUTWeightedMSELoss(nn.Module):
    """
    MSE loss weighted by a fixed, precomputed LUT.

    Args:
        lut_path (str or torch.Tensor):
            Path to a .pt file containing a 1D tensor of length n_bins,
            or a torch.Tensor of weights itself.
        sdf_min (float): lower clamp for SDF values
        sdf_max (float): upper clamp for SDF values
        n_bins (int): number of histogram bins (must match LUT length)
        reduction (str): 'mean', 'sum', or 'none'
    """
    def __init__(
        self,
        lut_path,
        sdf_min: float = -7.0,
        sdf_max: float = 7.0,
        n_bins: int = 256,
        reduction: str = 'mean',
    ):
        super().__init__()
        # basic args
        self.register_buffer('sdf_min', torch.tensor(float(sdf_min)))
        self.register_buffer('sdf_max', torch.tensor(float(sdf_max)))
        self.register_buffer('scale', 1.0 / (self.sdf_max - self.sdf_min))
        self.n_bins = int(n_bins)
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

        # load LUT
        if isinstance(lut_path, str):
            lut = torch.load(lut_path, map_location='cpu')
        elif torch.is_tensor(lut_path):
            lut = lut_path
        else:
            raise ValueError("lut_path must be a filename or a torch.Tensor")

        if lut.numel() != self.n_bins:
            raise ValueError(f"LUT length ({lut.numel()}) does not match n_bins ({self.n_bins})")

        # register as buffer so it moves with the module
        self.register_buffer('lut', lut.to(torch.float32))
        # always ready
        self._lut_ready = True

    @torch.no_grad()
    def _bin_indices(self, sdf: torch.Tensor) -> torch.LongTensor:
        clamped = torch.clamp(sdf, self.sdf_min, self.sdf_max)
        unit = (clamped - self.sdf_min) * self.scale
        idx = torch.round(unit * (self.n_bins - 1)).long()
        return idx

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # look up per‐pixel weight from fixed LUT
        idx = self._bin_indices(y_true)
        w = self.lut[idx].to(dtype=y_pred.dtype, device=y_pred.device)

        # compute weighted squared error
        wse = w * (y_pred - y_true).pow(2)

        # reduction
        if self.reduction == 'mean':
            return wse.sum() / y_pred.numel()
        elif self.reduction == 'sum':
            return wse.sum()
        else:  # 'none'
            return wse

    def extra_repr(self) -> str:
        return (
            f"sdf_min={self.sdf_min.item()}, sdf_max={self.sdf_max.item()}, "
            f"n_bins={self.n_bins}, reduction={self.reduction}"
        )
