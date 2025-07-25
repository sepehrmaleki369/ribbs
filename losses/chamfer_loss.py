import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

# -----------------------------------------------------------------------------
# Utility helpers (identical to your two base notebooks)
# -----------------------------------------------------------------------------

def sample_pred_at_positions(pred: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Bilinear‑interpolate *pred* at fractional row/col **positions** (N×2)."""
    r, c = positions[:, 0], positions[:, 1]
    r0, c0 = r.floor().long(), c.floor().long()
    r1, c1 = r0 + 1, c0 + 1
    dr, dc = (r - r0.float()).unsqueeze(1), (c - c0.float()).unsqueeze(1)
    H, W   = pred.shape

    r0, r1 = r0.clamp(0, H - 1), r1.clamp(0, H - 1)
    c0, c1 = c0.clamp(0, W - 1), c1.clamp(0, W - 1)

    Ia = pred[r0, c0].unsqueeze(1)
    Ib = pred[r0, c1].unsqueeze(1)
    Ic = pred[r1, c0].unsqueeze(1)
    Id = pred[r1, c1].unsqueeze(1)

    val = (Ia * (1 - dr) * (1 - dc) +
           Ib * (1 - dr) * dc        +
           Ic * dr        * (1 - dc) +
           Id * dr        * dc)
    return val.squeeze(1)


def compute_normals(sdf: torch.Tensor) -> torch.Tensor:
    """Central‑difference gradient (H×W×2)."""
    H, W = sdf.shape
    grad_r = torch.zeros_like(sdf)
    grad_c = torch.zeros_like(sdf)

    grad_r[1:-1]   = (sdf[2:]   - sdf[:-2]) / 2.0
    grad_c[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / 2.0
    grad_r[0]      =  sdf[1]  - sdf[0]
    grad_r[-1]     =  sdf[-1] - sdf[-2]
    grad_c[:, 0]   =  sdf[:, 1] - sdf[:, 0]
    grad_c[:, -1]  =  sdf[:, -1] - sdf[:, -2]
    return torch.stack([grad_r, grad_c], dim=2)


def sample_normals_at_positions(normals: torch.Tensor, positions: torch.Tensor, *, normalize=True) -> torch.Tensor:
    H, W, _ = normals.shape
    r, c    = positions[:, 0], positions[:, 1]
    r0, c0  = r.floor().long(), c.floor().long()
    r1, c1  = r0 + 1, c0 + 1
    dr, dc  = (r - r0.float()).unsqueeze(1), (c - c0.float()).unsqueeze(1)

    r0, r1 = r0.clamp(0, H - 1), r1.clamp(0, H - 1)
    c0, c1 = c0.clamp(0, W - 1), c1.clamp(0, W - 1)

    Ia = normals[r0, c0]
    Ib = normals[r0, c1]
    Ic = normals[r1, c0]
    Id = normals[r1, c1]

    out = (Ia * (1 - dr) * (1 - dc) +
           Ib * (1 - dr) * dc        +
           Ic * dr        * (1 - dc) +
           Id * dr        * dc)

    if normalize:
        out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-8)
    return out


def extract_zero_crossings_interpolated_positions(sdf: torch.Tensor, *, requires_grad: bool = False) -> torch.Tensor:
    """Return Nx2 tensor of (row, col) fractional coordinates where the signed‑distance
    field crosses zero (bilinear interpolation along grid edges)."""
    eps = 1e-8
    H, W = sdf.shape
    sdf_np = sdf.detach().cpu().numpy()
    pos = []

    # vertical neighbours
    for i in range(H - 1):
        for j in range(W):
            v1, v2 = sdf_np[i, j], sdf_np[i + 1, j]
            if v1 == 0:
                pos.append([i, j])
            elif v2 == 0:
                pos.append([i + 1, j])
            elif v1 * v2 < 0:
                a = abs(v1) / (abs(v1) + abs(v2) + eps)
                pos.append([i + a, j])

    # horizontal neighbours
    for i in range(H):
        for j in range(W - 1):
            v1, v2 = sdf_np[i, j], sdf_np[i, j + 1]
            if v1 == 0:
                pos.append([i, j])
            elif v2 == 0:
                pos.append([i, j + 1])
            elif v1 * v2 < 0:
                a = abs(v1) / (abs(v1) + abs(v2) + eps)
                pos.append([i, j + a])

    if not pos:
        return torch.empty((0, 2), dtype=torch.float32, device=sdf.device, requires_grad=requires_grad)
    return torch.tensor(pos, dtype=torch.float32, device=sdf.device, requires_grad=requires_grad)


def compute_chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    if p1.numel() == 0 or p2.numel() == 0:
        return torch.tensor(float('inf'), device=p1.device)
    diff  = p1.unsqueeze(1) - p2.unsqueeze(0)
    dists = torch.norm(diff, dim=2)
    m1,_  = torch.min(dists, dim=1)
    m2,_  = torch.min(dists, dim=0)
    return -torch.mean(m1) + torch.mean(m2)


def manual_chamfer_grad(pred_sdf: torch.Tensor,
                        pred_zc: torch.Tensor,
                        gt_zc:   torch.Tensor,
                        *, update_scale: float = 1.0,
                        dist_threshold: float = 3.0) -> torch.Tensor:
    """Pixel‑wise gradient used by the first (toy) base notebook."""
    dSDF = torch.zeros_like(pred_sdf)
    if pred_zc.numel() == 0 or gt_zc.numel() == 0:
        return dSDF

    normals = compute_normals(pred_sdf)
    n_at_p  = sample_normals_at_positions(normals, pred_zc)  # N×2

    gt_cpu   = gt_zc.detach().cpu()
    pred_cpu = pred_zc.detach().cpu()

    for i in range(pred_zc.shape[0]):
        p = pred_cpu[i]
        dists, idx = torch.norm(gt_cpu - p, dim=1, keepdim=False).min(dim=0)
        if dists > dist_threshold:
            continue
        dl_dp = gt_cpu[idx] - p  # direction
        n     = n_at_p[i]
        dot   = torch.dot(dl_dp.to(n.device), n) * update_scale

        r, c  = p[0].item(), p[1].item()
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = r0 + 1, c0 + 1
        wr1, wr0 = r - r0, 1 - (r - r0)
        wc1, wc0 = c - c0, 1 - (c - c0)
        H, W = dSDF.shape

        if 0 <= r0 < H and 0 <= c0 < W: dSDF[r0, c0] += dot * wr0 * wc0
        if 0 <= r0 < H and 0 <= c1 < W: dSDF[r0, c1] += dot * wr0 * wc1
        if 0 <= r1 < H and 0 <= c0 < W: dSDF[r1, c0] += dot * wr1 * wc0
        if 0 <= r1 < H and 0 <= c1 < W: dSDF[r1, c1] += dot * wr1 * wc1

    return dSDF

# -----------------------------------------------------------------------------
# Autograd function: |CD| forward  +  manual gradient backward (optional)
# -----------------------------------------------------------------------------

class _ChamferManualFn(Function):

    @staticmethod
    def forward(ctx, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor,
                weight: float, update_scale: float, dist_thr: float):
        pred_zc = extract_zero_crossings_interpolated_positions(pred_sdf)
        gt_zc   = extract_zero_crossings_interpolated_positions(gt_sdf)

        cd = compute_chamfer_distance(pred_zc, gt_zc)
        loss = weight * torch.abs(cd)

        # save for backward if a non‑zero update_scale is requested
        ctx.save_for_backward(pred_sdf, pred_zc, gt_zc)
        ctx.update_scale = update_scale
        ctx.dist_thr     = dist_thr
        ctx.has_grad     = update_scale != 0.0
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        if not ctx.has_grad:
            return grad_out * 0, None, None, None, None  # zero grad w.r.t. pred_sdf

        pred_sdf, pred_zc, gt_zc = ctx.saved_tensors
        dSDF = manual_chamfer_grad(pred_sdf, pred_zc, gt_zc,
                                   update_scale=ctx.update_scale,
                                   dist_threshold=ctx.dist_thr)
        return grad_out * dSDF, None, None, None, None

# -----------------------------------------------------------------------------
# Loss module
# -----------------------------------------------------------------------------

class SDFChamferLoss(nn.Module):
    """A *single* PyTorch‑Lightning/Lit‑friendly loss that covers both of your
    reference notebooks:

    • **loss_zc**  – sum of predicted SDF values at their own zero‑crossings.
      Enabled when *weight_zc* > 0.

    • **loss_sdf** – per‑pixel L1 between predicted and GT SDFs (*weight_sdf*).

    • **loss_cd**  – |Chamfer| distance between the zero‑crossing point sets
      (*weight_chamfer*).  If *update_scale* ≠ 0 we inject the *manual* Chamfer
      gradient used in the toy notebook; if it is 0 we let the term be
      non‑differentiable, matching the second notebook where only L1 provides
      gradients.
    """

    def __init__(self,
                 weight_sdf: float = 1.0,
                 weight_chamfer: float = 1.0,
                 weight_zc: float = 0.0,
                 update_scale=None,
                 dist_threshold: float = 3.0):
        super().__init__()
        self.w_sdf   = float(weight_sdf)
        self.w_cd    = float(weight_chamfer)
        self.w_zc    = float(weight_zc)
        self.dist_thr = float(dist_threshold)
        self.update_scale = float(update_scale) if update_scale is not None else float(weight_chamfer)

    # ------------------------------------------------------------------
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        """*y_pred* & *y_true* shape: **[B, 1, H, W]**."""
        pred = y_pred.squeeze(1)  # [B,H,W]
        gt   = y_true.squeeze(1)

        total = 0.0
        for p, g in zip(pred, gt):
            # 1) optional zero‑crossing value term -------------------------
            if self.w_zc != 0.0:
                zc = extract_zero_crossings_interpolated_positions(p)
                loss_zc = self.w_zc * sample_pred_at_positions(p, zc).sum()
            else:
                loss_zc = 0.0

            # 2) optional L1 SDF term -------------------------------------
            loss_sdf = self.w_sdf * F.l1_loss(p, g)

            # 3) Chamfer term (+ optional manual gradient) ----------------
            loss_cd = _ChamferManualFn.apply(
                p, g,
                self.w_cd,
                self.update_scale,
                self.dist_thr
            )

            total += loss_zc + loss_sdf + loss_cd

        return total / pred.size(0)

# -----------------------------------------------------------------------------
# Expose helpers (so your smoke‑test can import them)
# -----------------------------------------------------------------------------
__all__ = [
    "SDFChamferLoss",
    "extract_zero_crossings_interpolated_positions",
]
