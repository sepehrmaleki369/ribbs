import torch
import torch.nn as nn

class FixedLUTWeightedMSELoss(nn.Module):
    """
    Same weighting formula as LIFWeightedMSELoss but with a *frozen* LUT that
    is loaded from disk (or passed as a tensor).

    Args
    ----
    lut_path (str | Tensor): 1-D tensor of length n_bins or path to .pt
    sdf_min / sdf_max (float): clamp range used when the LUT was built
    n_bins (int)            : number of histogram bins (must match LUT length)
    reduction ('mean'|'sum'|'none')
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
        # ---- common buffers -------------------------------------------------
        self.register_buffer('sdf_min', torch.tensor(float(sdf_min)))
        self.register_buffer('sdf_max', torch.tensor(float(sdf_max)))
        self.register_buffer('scale', 1.0 / (self.sdf_max - self.sdf_min))

        self.n_bins = int(n_bins)
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

        # ---- load LUT -------------------------------------------------------
        if isinstance(lut_path, str):
            lut = torch.load(lut_path, map_location='cpu')
        elif torch.is_tensor(lut_path):
            lut = lut_path
        else:
            raise TypeError("lut_path must be filename or Tensor")

        if lut.numel() != self.n_bins:
            raise ValueError(
                f"LUT length {lut.numel()} ≠ n_bins {self.n_bins}"
            )

        # *Same buffer name as dynamic class*
        self.register_buffer('_lut', lut.to(torch.float32), persistent=True)
        self._lut_ready = True        # already frozen

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _bin_indices(self, sdf: torch.Tensor) -> torch.LongTensor:
        clamped = torch.clamp(sdf, self.sdf_min, self.sdf_max)
        unit    = (clamped - self.sdf_min) * self.scale
        return torch.round(unit * (self.n_bins - 1)).long()

    # -------------------------------------------------------------------------
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        idx = self._bin_indices(y_true)
        w   = self._lut[idx].to(dtype=y_pred.dtype, device=y_pred.device)
        wse = w * (y_pred - y_true).pow(2)

        if self.reduction == 'mean':
            return wse.sum() / y_pred.numel()
        if self.reduction == 'sum':
            return wse.sum()
        return wse

    # -------------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"sdf_min={self.sdf_min.item()}, sdf_max={self.sdf_max.item()}, "
            f"n_bins={self.n_bins}, reduction={self.reduction}"
        )

    # -------------------------------------------------------------------------
    # Optional: load both old ('_lut') and new ('lut') keys seamlessly
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        key = prefix + "_lut"
        if key in state_dict and state_dict[key].numel() != self._lut.numel():
            print(
                f"⚠️  Replacing {key}: ckpt {tuple(state_dict[key].shape)} "
                f"→ current {tuple(self._lut.shape)}"
            )
            # put *our* 15-bin LUT into the state-dict
            state_dict[key] = self._lut.detach().cpu()

        # now let the normal loader run — sizes match, no missing keys
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )