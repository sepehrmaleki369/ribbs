import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ==========================================================
# Utility functions
# ==========================================================

def subsample_points(pts: torch.Tensor, max_points: int) -> torch.Tensor:
    """Randomly subsample points if they exceed max_points."""
    if pts.numel() == 0 or pts.size(0) <= max_points:
        return pts
    idx = torch.randperm(pts.size(0), device=pts.device)[:max_points]
    return pts[idx]


def chunked_min_pairwise(a: torch.Tensor,
                         b: torch.Tensor,
                         chunk: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute min pairwise distances a->b and b->a in chunks to avoid OOM.
    a: (Na,2), b: (Nb,2)
    Returns:
        mins_ab: (Na,)  each element is min distance from a[i] to any point in b
        mins_ba: (Nb,)  same but from b[j] to any point in a
    """
    if a.numel() == 0 or b.numel() == 0:
        dev = a.device if a.numel() else b.device
        na = a.size(0)
        nb = b.size(0)
        mins_ab = torch.full((na,), float('inf'), device=dev)
        mins_ba = torch.full((nb,), float('inf'), device=dev)
        return mins_ab, mins_ba

    mins_ab = []
    for s in range(0, a.size(0), chunk):
        d = torch.cdist(a[s:s + chunk], b)
        mins_ab.append(d.min(dim=1).values)
    mins_ab = torch.cat(mins_ab, dim=0)

    mins_ba = []
    for s in range(0, b.size(0), chunk):
        d = torch.cdist(b[s:s + chunk], a)
        mins_ba.append(d.min(dim=1).values)
    mins_ba = torch.cat(mins_ba, dim=0)

    return mins_ab, mins_ba


def bilinear_sample(image: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    Bilinear sample a 2D tensor image(H,W) at N floating positions pos(N,2).
    pos[:,0] = row (y), pos[:,1] = col (x).
    Returns (N,).
    """
    r, c = pos[:, 0], pos[:, 1]
    r0 = torch.floor(r).long()
    c0 = torch.floor(c).long()
    r1 = r0 + 1
    c1 = c0 + 1

    H, W = image.shape
    r0 = r0.clamp(0, H - 1)
    r1 = r1.clamp(0, H - 1)
    c0 = c0.clamp(0, W - 1)
    c1 = c1.clamp(0, W - 1)

    dr = (r - r0.float()).unsqueeze(1)
    dc = (c - c0.float()).unsqueeze(1)

    Ia = image[r0, c0].unsqueeze(1)
    Ib = image[r0, c1].unsqueeze(1)
    Ic = image[r1, c0].unsqueeze(1)
    Id = image[r1, c1].unsqueeze(1)

    val = (Ia * (1 - dr) * (1 - dc) +
           Ib * (1 - dr) * dc +
           Ic * dr * (1 - dc) +
           Id * dr * dc)
    return val.squeeze(1)


def finite_diff_normals(sdf: torch.Tensor) -> torch.Tensor:
    """
    Compute normals via central differences.
    sdf: (H,W)
    returns: (H,W,2) -> grad_row, grad_col
    """
    H, W = sdf.shape
    grad_row = torch.zeros_like(sdf)
    grad_col = torch.zeros_like(sdf)

    grad_row[1:-1] = (sdf[2:] - sdf[:-2]) * 0.5
    grad_col[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) * 0.5

    grad_row[0] = sdf[1] - sdf[0]
    grad_row[-1] = sdf[-1] - sdf[-2]
    grad_col[:, 0] = sdf[:, 1] - sdf[:, 0]
    grad_col[:, -1] = sdf[:, -1] - sdf[:, -2]

    return torch.stack([grad_row, grad_col], dim=2)


def bilinear_sample_normals(normals: torch.Tensor,
                            pos: torch.Tensor,
                            normalize: bool = True) -> torch.Tensor:
    """
    Bilinearly sample normals(H,W,2) at positions(N,2).
    returns (N,2).
    """
    H, W, _ = normals.shape
    r, c = pos[:, 0], pos[:, 1]
    r0 = torch.floor(r).long()
    c0 = torch.floor(c).long()
    r1 = r0 + 1
    c1 = c0 + 1

    r0 = r0.clamp(0, H - 1)
    r1 = r1.clamp(0, H - 1)
    c0 = c0.clamp(0, W - 1)
    c1 = c1.clamp(0, W - 1)

    dr = (r - r0.float()).unsqueeze(1)
    dc = (c - c0.float()).unsqueeze(1)

    Ia = normals[r0, c0]
    Ib = normals[r0, c1]
    Ic = normals[r1, c0]
    Id = normals[r1, c1]

    n = (Ia * (1 - dr) * (1 - dc) +
         Ib * (1 - dr) * dc +
         Ic * dr * (1 - dc) +
         Id * dr * dc)

    if normalize:
        n = n / (torch.norm(n, dim=1, keepdim=True) + 1e-8)
    return n


def extract_zero_crossings(sdf: torch.Tensor) -> torch.Tensor:
    """
    Fully vectorized zero-crossing extraction for a single 2D SDF image.
    sdf: (H,W)
    returns: (N,2) tensor of subpixel coordinates (row, col).
    """
    device = sdf.device
    dtype = sdf.dtype
    eps = 1e-8

    H, W = sdf.shape

    # vertical sign changes between rows
    v1 = sdf[:-1, :]     # (H-1,W)
    v2 = sdf[1:, :]      # (H-1,W)
    sign_change_v = (v1 * v2) < 0
    nonzero_v1 = (v1 == 0)
    nonzero_v2 = (v2 == 0)

    idx_v = sign_change_v.nonzero(as_tuple=False)
    # idx_v: (Nv, 2) -> [r, c] where r in [0,H-2]
    if idx_v.numel():
        r = idx_v[:, 0]
        c = idx_v[:, 1]
        a = torch.abs(v1[r, c])
        b = torch.abs(v2[r, c])
        alpha = a / (a + b + eps)
        rows = r.float() + alpha
        cols = c.float()
        vert_pts = torch.stack([rows, cols], dim=1)
    else:
        vert_pts = torch.empty((0, 2), device=device, dtype=dtype)

    # handle exact zero in vertical neighbors
    z_idx_v1 = nonzero_v1.nonzero(as_tuple=False)
    z_idx_v2 = nonzero_v2.nonzero(as_tuple=False)
    if z_idx_v1.numel():
        vert_zero1 = torch.stack([z_idx_v1[:, 0].float(),
                                  z_idx_v1[:, 1].float()], dim=1)
        vert_pts = torch.cat([vert_pts, vert_zero1], dim=0)
    if z_idx_v2.numel():
        vert_zero2 = torch.stack([z_idx_v2[:, 0].float() + 1,
                                  z_idx_v2[:, 1].float()], dim=1)
        vert_pts = torch.cat([vert_pts, vert_zero2], dim=0)

    # horizontal sign changes between columns
    h1 = sdf[:, :-1]     # (H,W-1)
    h2 = sdf[:,  1:]     # (H,W-1)
    sign_change_h = (h1 * h2) < 0
    nonzero_h1 = (h1 == 0)
    nonzero_h2 = (h2 == 0)

    idx_h = sign_change_h.nonzero(as_tuple=False)
    if idx_h.numel():
        r = idx_h[:, 0]
        c = idx_h[:, 1]
        a = torch.abs(h1[r, c])
        b = torch.abs(h2[r, c])
        alpha = a / (a + b + eps)
        rows = r.float()
        cols = c.float() + alpha
        horiz_pts = torch.stack([rows, cols], dim=1)
    else:
        horiz_pts = torch.empty((0, 2), device=device, dtype=dtype)

    # handle exact zero in horizontal neighbors
    z_idx_h1 = nonzero_h1.nonzero(as_tuple=False)
    z_idx_h2 = nonzero_h2.nonzero(as_tuple=False)
    if z_idx_h1.numel():
        horiz_zero1 = torch.stack([z_idx_h1[:, 0].float(),
                                   z_idx_h1[:, 1].float()], dim=1)
        horiz_pts = torch.cat([horiz_pts, horiz_zero1], dim=0)
    if z_idx_h2.numel():
        horiz_zero2 = torch.stack([z_idx_h2[:, 0].float(),
                                   z_idx_h2[:, 1].float() + 1], dim=1)
        horiz_pts = torch.cat([horiz_pts, horiz_zero2], dim=0)

    # concatenate and remove duplicates (optional)
    if vert_pts.numel() == 0 and horiz_pts.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=dtype)

    pts = torch.cat([vert_pts, horiz_pts], dim=0)
    # optional uniqueness (could be expensive); usually fine without:
    # pts = torch.unique(torch.round(pts * 1000) / 1000, dim=0)

    return pts


def make_dSDF(pred_sdf: torch.Tensor,
              pred_zc: torch.Tensor,
              gt_zc: torch.Tensor,
              update_scale: float,
              dist_threshold: float,
              normalize_normals: bool,
              max_points: int,
              clip_val: Optional[float]) -> torch.Tensor:
    """
    Create the additive gradient field dSDF to be injected.
    """
    if pred_zc.numel() == 0 or gt_zc.numel() == 0:
        return torch.zeros_like(pred_sdf)

    # optional subsample for safety
    pred_zc = subsample_points(pred_zc, max_points)
    gt_zc   = subsample_points(gt_zc,   max_points)

    normals = finite_diff_normals(pred_sdf)
    n_at_pts = bilinear_sample_normals(normals, pred_zc, normalize=normalize_normals)

    # Compute nearest GT point for each pred point
    # We'll do it chunked:
    device = pred_sdf.device
    dSDF = torch.zeros(
        pred_sdf.shape,
        device=pred_sdf.device,
        dtype=pred_sdf.dtype
    )

    # Move pred_zc to CPU for loop? We keep it on GPU; we only iterate minimal times
    # but do operations in vectorized chunk fashion:
    chunk = 2048
    for s in range(0, pred_zc.size(0), chunk):
        p_chunk = pred_zc[s:s + chunk]          # (C,2)
        n_chunk = n_at_pts[s:s + chunk]         # (C,2)
        dist = torch.cdist(p_chunk, gt_zc)      # (C, G)
        min_dist, idx = dist.min(dim=1)         # (C,)
        mask = min_dist <= dist_threshold
        if not mask.any():
            continue

        valid_p = p_chunk[mask]
        valid_n = n_chunk[mask]
        valid_g = gt_zc[idx[mask]]

        # direction from pred to gt
        dl_dp = (valid_g - valid_p)             # (M,2)
        if normalize_normals:
            valid_n = valid_n / (torch.norm(valid_n, dim=1, keepdim=True) + 1e-8)

        dot_val = torch.sum(dl_dp * valid_n, dim=1) * update_scale  # (M,)

        # scatter into dSDF with bilinear weights
        r = valid_p[:, 0]
        c = valid_p[:, 1]
        r0 = torch.floor(r).long().clamp(0, pred_sdf.shape[0] - 1)
        c0 = torch.floor(c).long().clamp(0, pred_sdf.shape[1] - 1)
        r1 = (r0 + 1).clamp(0, pred_sdf.shape[0] - 1)
        c1 = (c0 + 1).clamp(0, pred_sdf.shape[1] - 1)

        dr = (r - r0.float())
        dc = (c - c0.float())
        w00 = (1 - dr) * (1 - dc)
        w01 = (1 - dr) * dc
        w10 = dr * (1 - dc)
        w11 = dr * dc

        # accumulation
        incr00 = (dot_val * w00).to(dSDF.dtype)
        incr01 = (dot_val * w01).to(dSDF.dtype)
        incr10 = (dot_val * w10).to(dSDF.dtype)
        incr11 = (dot_val * w11).to(dSDF.dtype)

        dSDF.index_put_((r0, c0), incr00, accumulate=True)
        dSDF.index_put_((r0, c1), incr01, accumulate=True)
        dSDF.index_put_((r1, c0), incr10, accumulate=True)
        dSDF.index_put_((r1, c1), incr11, accumulate=True)

    if clip_val is not None:
        dSDF = torch.clamp(dSDF, -clip_val, clip_val)

    return dSDF


# ==========================================================
# Main loss
# ==========================================================

class ChamferBoundarySDFLoss(nn.Module):
    """
    Final loss = pixel_weight * L1(pred_sdf, gt_sdf).

    Additionally, a custom gradient term (dSDF) is injected during backward
    to align the predicted zero-level set to the ground truth boundary.

    Parameters
    ----------
    pixel_weight       : weight of the pixel L1 term
    chamfer_weight     : scales the injected gradient (hook)
    update_scale       : multiplier inside dSDF construction
    dist_threshold     : maximum distance (in px) to consider for Chamfer pairing
    max_points         : subsample count of zero-crossing points (pred & gt)
    normalize_normals  : normalize normals before dot product
    use_hook           : True => inject gradient via register_hook
    clip_sdf_to        : clamp both pred and gt SDF to [lo,hi] before loss
    clip_dSDF          : clamp dSDF values to [-clip_dSDF, +clip_dSDF]
    warmup_factor      : 0..1 external scalar to scale chamfer_weight (set with set_epoch or manually)
    """

    def __init__(self,
                 pixel_weight: float = 1.0,
                 chamfer_weight: float = 1.0,
                 update_scale: float = 100.0,
                 dist_threshold: float = 1.0,
                 max_points: int = 4096,
                 normalize_normals: bool = True,
                 use_hook: bool = True,
                 clip_sdf_to: Optional[Tuple[float, float]] = None,
                 clip_dSDF: Optional[float] = 1.0):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.chamfer_weight = chamfer_weight
        self.update_scale = update_scale
        self.dist_threshold = dist_threshold
        self.max_points = max_points
        self.normalize_normals = normalize_normals
        self.use_hook = use_hook
        self.clip_sdf_to = clip_sdf_to
        self.clip_dSDF = clip_dSDF

        self._warmup = 1.0  # external multiplier you can change per epoch

        # Logging holders
        self.last_pixel = torch.tensor(0.0)
        self.last_chamfer = torch.tensor(0.0)

    # Optional: allow external warmup control
    def set_warmup_factor(self, factor: float):
        self._warmup = float(factor)

    def forward(self, pred_logits: torch.Tensor, gt_sdf: torch.Tensor) -> torch.Tensor:
        """
        pred_logits: (B,1,H,W) raw network output interpreted as SDF
        gt_sdf     : (B,1,H,W) or (B,H,W)
        """
        if pred_logits.ndim != 4 or pred_logits.size(1) != 1:
            raise ValueError("pred_logits must be (B,1,H,W)")

        pred_sdf = pred_logits[:, 0]  # (B,H,W)
        if gt_sdf.ndim == 4 and gt_sdf.size(1) == 1:
            gt_sdf = gt_sdf[:, 0]
        elif gt_sdf.ndim != 3:
            raise ValueError("gt_sdf must be (B,1,H,W) or (B,H,W)")

        if self.clip_sdf_to is not None:
            lo, hi = self.clip_sdf_to
            pred_sdf = pred_sdf.clamp(lo, hi)
            gt_sdf = gt_sdf.clamp(lo, hi)

        # Pixel-wise L1
        pixel_loss = F.l1_loss(pred_sdf, gt_sdf)

        chamfer_vals = []
        if self.use_hook and self._warmup > 0 and self.chamfer_weight > 0:
            # build dSDF per image and attach hook
            for b in range(pred_sdf.size(0)):
                psdf = pred_sdf[b]
                gsdf = gt_sdf[b]

                with torch.no_grad():
                    gt_zc = subsample_points(extract_zero_crossings(gsdf), self.max_points)
                    pred_zc = subsample_points(extract_zero_crossings(psdf), self.max_points)

                # Only for logging
                if pred_zc.numel() and gt_zc.numel():
                    d_ab, d_ba = chunked_min_pairwise(pred_zc, gt_zc)
                    chamfer_vals.append(-d_ab.mean() + d_ba.mean())
                else:
                    chamfer_vals.append(psdf.new_tensor(0.0))

                if pred_zc.numel() and gt_zc.numel():
                    dSDF = make_dSDF(psdf,
                                     pred_zc,
                                     gt_zc,
                                     update_scale=self.update_scale,
                                     dist_threshold=self.dist_threshold,
                                     normalize_normals=self.normalize_normals,
                                     max_points=self.max_points,
                                     clip_val=self.clip_dSDF)

                    # Inject gradient
                    def _hook(grad, d=dSDF):
                        d_cast = d.to(grad.dtype)
                        return grad + self._warmup * self.chamfer_weight * d_cast

                    psdf.register_hook(_hook)
        else:
            # no hook => chamfer_vals still trackable for logs but not used
            chamfer_vals = [pred_sdf.new_tensor(0.0) for _ in range(pred_sdf.size(0))]

        self.last_pixel = pixel_loss.detach()
        self.last_chamfer = torch.stack(chamfer_vals).mean().detach()

        return self.pixel_weight * pixel_loss
