import torch
import torch.nn as nn

class LIFWeightedMSELoss(nn.Module):
    """
    Log-Inverse-Frequency weighted MSE loss with optional global LUT freezing.

    Each pixel weight = 1 / log(1 + eps + freq_k), where freq_k is the relative
    frequency of the pixel's SDF bin across the current batch or a frozen dataset.

    Args:
        sdf_min (float): lower clamp for SDF values (e.g. -d_max).
        sdf_max (float): upper clamp for SDF values (e.g. +d_max).
        n_bins (int): number of histogram bins.
        eps (float): small constant inside log to avoid division-by-zero.
        freeze_after_first (bool): if True, build LUT once at first forward and reuse.
        reduction (str): 'mean', 'sum', or 'none'.
    """
    def __init__(
        self,
        sdf_min: float = -7.0,
        sdf_max: float = 7.0,
        n_bins: int = 256,
        eps: float = 0.02,
        freeze_after_first: bool = False,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        if sdf_max <= sdf_min:
            raise ValueError('sdf_max must be > sdf_min')

        # store range and scale as buffers
        self.register_buffer('sdf_min', torch.tensor(float(sdf_min)))
        self.register_buffer('sdf_max', torch.tensor(float(sdf_max)))
        self.register_buffer('scale', 1.0 / (self.sdf_max - self.sdf_min))

        self.n_bins = int(n_bins)
        self.eps = float(eps)
        self.freeze_after_first = bool(freeze_after_first)
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

        # LUT registered as buffer, persistent to allow checkpointing
        self.register_buffer('_lut', torch.ones(self.n_bins), persistent=True)
        self._lut_ready = False

    @torch.no_grad()
    def freeze(self, data_loader) -> None:
        """
        Build a global LUT from ground-truth SDFs in data_loader and freeze it.
        Subsequent forwards will reuse this LUT.
        """
        if self._lut_ready:
            return
        device = self.sdf_min.device
        counts = torch.zeros(self.n_bins, device=device)
        total = 0
        for batch in data_loader:
            sdf = batch.to(device)
            idx = self._bin_indices(sdf)
            counts += torch.bincount(idx.flatten(), minlength=self.n_bins)
            total += idx.numel()
        if total == 0:
            raise RuntimeError('freeze received empty data_loader')
        freq = counts.float() / total
        self._lut = 1.0 / torch.log1p(self.eps + freq)
        self._lut_ready = True

    @torch.no_grad()
    def _bin_indices(self, sdf: torch.Tensor) -> torch.LongTensor:
        clamped = torch.clamp(sdf, self.sdf_min, self.sdf_max)
        unit = (clamped - self.sdf_min) * self.scale
        idx = torch.round(unit * (self.n_bins - 1)).long()
        return idx

    @torch.no_grad()
    def _build_lut(self, sdf: torch.Tensor) -> torch.Tensor:
        idx = self._bin_indices(sdf)
        freq = torch.bincount(idx.flatten(), minlength=self.n_bins).float()
        freq /= idx.numel()
        return 1.0 / torch.log1p(self.eps + freq)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # build or reuse LUT
        if not self._lut_ready:
            with torch.no_grad():
                self._lut = self._build_lut(y_true)
                if self.freeze_after_first:
                    self._lut_ready = True

        # gather weights
        idx = self._bin_indices(y_true)
        w = self._lut[idx].to(dtype=y_pred.dtype)

        # weighted squared error
        wse = w * (y_pred - y_true).pow(2)

        # reduction
        if self.reduction == 'mean':
            return wse.sum() / y_pred.numel()
        if self.reduction == 'sum':
            return wse.sum()
        return wse

    def extra_repr(self) -> str:
        return (
            f'sdf_min={self.sdf_min.item()}, sdf_max={self.sdf_max.item()}, '
            f'n_bins={self.n_bins}, eps={self.eps}, '
            f'freeze_after_first={self.freeze_after_first}, reduction={self.reduction}'
        )
