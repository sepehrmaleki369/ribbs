from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch, torch.nn as nn, torch.nn.functional as F

# ---------------------------- helpers ----------------------------

def _prep_sdf(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 4:  # [B,1,H,W]
        return t.squeeze(1)
    if t.dim() == 3:  # [B,H,W]
        return t
    raise ValueError(f"Expected [B,1,H,W] or [B,H,W], got {t.shape}")


def _zero_crossings_lin_interp(
    sdf: torch.Tensor,
    iso: float = 0.0,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sub‑pixel iso-surface extraction along rows/cols.
    Count a crossing if:
      • sign change OR
      • either endpoint within ±eps of iso.
    Returns:
        P : (K,3) -> [b, row, col]
        counts : (B,)
    """
    sdf = sdf - iso
    B, H, W = sdf.shape
    dev = sdf.device
    sign = torch.sign(sdf)

    def _interp_axis(a1, a2, axis):
        # a1,a2 : tensors sharing B except along 'axis'
        # axis=0 (vertical) => rows interp; axis=1 (horizontal) => cols interp
        denom = (a1.abs() + a2.abs()).clamp_min(1e-8)
        alpha = (a1.abs() / denom).clamp(0.0, 1.0)
        return alpha

    # vertical (rows)
    s1v, s2v = sign[:, :-1, :], sign[:, 1:, :]
    v1,  v2  = sdf[:, :-1, :],  sdf[:, 1:, :]
    mask_v = ((s1v * s2v) < 0) | (v1.abs() <= eps) | (v2.abs() <= eps)
    alpha_v = _interp_axis(v1, v2, axis=0)
    r_v = torch.arange(H - 1, device=dev, dtype=torch.float32).view(1, -1, 1).expand(B, -1, W)
    c_v = torch.arange(W,     device=dev, dtype=torch.float32).view(1, 1, -1).expand(B, H - 1, -1)
    rows_v = r_v + alpha_v
    cols_v = c_v

    # horizontal (cols)
    s1h, s2h = sign[:, :, :-1], sign[:, :, 1:]
    v1h, v2h = sdf[:, :, :-1],  sdf[:, :, 1:]
    mask_h = ((s1h * s2h) < 0) | (v1h.abs() <= eps) | (v2h.abs() <= eps)
    alpha_h = _interp_axis(v1h, v2h, axis=1)
    r_h = torch.arange(H, device=dev, dtype=torch.float32).view(1, -1, 1).expand(B, H, W - 1)
    c_h = torch.arange(W - 1, device=dev, dtype=torch.float32).view(1, 1, -1).expand(B, H, -1)
    rows_h = r_h
    cols_h = c_h + alpha_h

    def _pack(mask, rows, cols):
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return torch.empty(0, 3, device=dev), torch.zeros(B, dtype=torch.long, device=dev)
        b = idx[:, 0]
        r = rows[idx[:, 0], idx[:, 1], idx[:, 2]]
        c = cols[idx[:, 0], idx[:, 1], idx[:, 2]]
        pts = torch.stack([b.float(), r, c], dim=1)
        cnt = torch.bincount(b, minlength=B)
        return pts, cnt

    Pv, cnt_v = _pack(mask_v, rows_v, cols_v)
    Ph, cnt_h = _pack(mask_h, rows_h, cols_h)

    return torch.cat([Pv, Ph], dim=0), cnt_v + cnt_h


def _compute_normals(sdf: torch.Tensor) -> torch.Tensor:
    H, W = sdf.shape
    grad_r = torch.zeros_like(sdf)
    grad_c = torch.zeros_like(sdf)
    grad_r[1:-1] = (sdf[2:] - sdf[:-2]) * 0.5
    grad_r[0]    = sdf[1] - sdf[0]
    grad_r[-1]   = sdf[-1] - sdf[-2]
    grad_c[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) * 0.5
    grad_c[:, 0]    = sdf[:, 1] - sdf[:, 0]
    grad_c[:, -1]   = sdf[:, -1] - sdf[:, -2]
    return torch.stack([grad_r, grad_c], dim=-1)


def _sample_normals(normals: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    H, W, _ = normals.shape
    r, c = pos[:, 0], pos[:, 1]
    r0 = r.floor().long().clamp(0, H - 1)
    c0 = c.floor().long().clamp(0, W - 1)
    r1 = (r0 + 1).clamp(0, H - 1)
    c1 = (c0 + 1).clamp(0, W - 1)
    dr = (r - r0.float()).unsqueeze(1)
    dc = (c - c0.float()).unsqueeze(1)
    Ia = normals[r0, c0]
    Ib = normals[r0, c1]
    Ic = normals[r1, c0]
    Id = normals[r1, c1]
    return Ia * (1 - dr) * (1 - dc) + Ib * (1 - dr) * dc + Ic * dr * (1 - dc) + Id * dr * dc


def _splat_bilinear(buf: torch.Tensor, pos: torch.Tensor, val: torch.Tensor) -> None:
    if val.dtype != buf.dtype:
        val = val.to(buf.dtype)
    H, W = buf.shape
    r, c = pos[:, 0], pos[:, 1]
    r0 = r.floor().long().clamp(0, H - 1)
    c0 = c.floor().long().clamp(0, W - 1)
    r1 = (r0 + 1).clamp(0, H - 1)
    c1 = (c0 + 1).clamp(0, W - 1)
    wr1 = r - r0.float(); wr0 = 1 - wr1
    wc1 = c - c0.float(); wc0 = 1 - wc1
    w00 = wr0 * wc0; w01 = wr0 * wc1; w10 = wr1 * wc0; w11 = wr1 * wc1
    buf.index_put_((r0, c0), val * w00, accumulate=True)
    buf.index_put_((r0, c1), val * w01, accumulate=True)
    buf.index_put_((r1, c0), val * w10, accumulate=True)
    buf.index_put_((r1, c1), val * w11, accumulate=True)


# ---------------------------- loss ----------------------------
class SDFChamferLoss(nn.Module):
    """
    Chamfer-style loss for SDFs with an L1 term.

    Parameters
    ----------
    weight_sdf : float
        Weight for the pixel-wise L1(pred, gt).
    band, reduction, use_squared, update_scale, normalize_normals,
    iso, eps : same as before.
    """
    def __init__(
        self,
        weight_sdf:      float = 1.0,
        band:            Optional[float] = 3.0,
        reduction:       str = "mean",
        use_squared:     bool = False,
        update_scale:    float = 1.0,
        normalize_normals: bool = True,
        iso:             float = 0.0,
        eps:             float = 1e-3,
    ):
        super().__init__()
        assert reduction in {"mean", "sum"}
        self.weight_sdf  = weight_sdf
        self.band   = band
        self.red    = reduction
        self.use_sq = use_squared
        self.scale  = update_scale
        self.norm_n = normalize_normals
        self.iso    = iso
        self.eps    = float(eps)

    # ------------------------------------------------------------------
    def forward(
        self,
        pred_sdf_in: torch.Tensor,
        gt_sdf_in:   torch.Tensor,
        *, return_parts: bool = False,
    ):
        pred_sdf = _prep_sdf(pred_sdf_in)
        gt_sdf   = _prep_sdf(gt_sdf_in)
        B, _, _  = pred_sdf.shape
        dev, dt  = pred_sdf.device, pred_sdf.dtype

        # ---------- iso-curve extraction (no grad) ----------
        P_all, P_cnt = _zero_crossings_lin_interp(pred_sdf.detach(), self.iso, self.eps)
        G_all, G_cnt = _zero_crossings_lin_interp(gt_sdf.detach(),   self.iso, self.eps)

        if P_all.numel() == 0 and G_all.numel() == 0:
            # fall-back to pure L1 if nothing to match
            l1 = F.l1_loss(pred_sdf, gt_sdf, reduction=self.red)
            loss_out = self.weight_sdf * l1
            return (loss_out, 0.0, l1, torch.tensor(0.0, device=dev)) if return_parts else loss_out

        chamfers, pseudos = [], []
        p_off = g_off = 0
        for b in range(B):
            p_n, g_n = P_cnt[b].item(), G_cnt[b].item()
            P = P_all[p_off:p_off+p_n, 1:] if p_n else torch.empty(0, 2, device=dev)
            G = G_all[g_off:g_off+g_n, 1:] if g_n else torch.empty(0, 2, device=dev)
            p_off += p_n; g_off += g_n

            if p_n == 0 or g_n == 0:
                chamfers.append(torch.zeros((), dtype=dt, device=dev))
                pseudos .append(torch.zeros((), dtype=dt, device=dev))
                continue

            d = torch.cdist(P, G, p=2)

            if self.band is not None:
                maskP = d.min(1).values <= self.band
                maskG = d.min(0).values <= self.band
            else:
                maskP = torch.ones(p_n, dtype=torch.bool, device=dev)
                maskG = torch.ones(g_n, dtype=torch.bool, device=dev)

            dP = d[maskP].min(1).values if maskP.any() else torch.zeros(0, device=dev, dtype=dt)
            dG = d[:, maskG].min(0).values if maskG.any() else torch.zeros(0, device=dev, dtype=dt)
            if self.use_sq:
                dP = dP.pow(2); dG = dG.pow(2)
            chamfers.append(dP.mean() + dG.mean())

            # pseudo-grad
            if maskP.any():
                normals  = _compute_normals(pred_sdf[b])
                valid_P  = P[maskP]
                idx      = d[maskP].argmin(1)
                matchedG = G[idx]
                dl_dp    = matchedG - valid_P
                n        = _sample_normals(normals, valid_P)
                if self.norm_n:
                    n = n / (n.norm(1, keepdim=True) + 1e-8)
                proj     = (dl_dp * n).sum(1) * self.scale
                grad_map = torch.zeros_like(pred_sdf[b], dtype=proj.dtype, device=dev)
                _splat_bilinear(grad_map, valid_P, proj)
                pseudos.append( - (pred_sdf[b] * grad_map).sum() )
            else:
                pseudos.append(torch.zeros((), dtype=dt, device=dev))

        chamfers = torch.stack(chamfers)   # no grad
        pseudos  = torch.stack(pseudos)    # carries grad

        # ---------- dense L1 term ----------
        l1 = F.l1_loss(pred_sdf, gt_sdf, reduction=self.red)

        # ---------- final scalar ----------
        opt_loss = pseudos + self.weight_sdf * l1        # has grad
        log_val  = chamfers.abs() + self.weight_sdf * l1 # what you print

        opt_loss = opt_loss.mean() if self.red == "mean" else opt_loss.sum()
        log_val  = log_val.mean()  if self.red == "mean" else log_val.sum()

        if return_parts:
            grad_mag = pseudos.abs().mean()
            return opt_loss, log_val, chamfers.mean(), l1, grad_mag
        return opt_loss