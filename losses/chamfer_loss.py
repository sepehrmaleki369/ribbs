import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================== Helpers (exact copies) ===============================

def sample_pred_at_positions(pred, positions):
    r = positions[:, 0]
    c = positions[:, 1]
    r0 = r.floor().long(); c0 = c.floor().long()
    r1 = r0 + 1;           c1 = c0 + 1
    dr = (r - r0.float()).unsqueeze(1)
    dc = (c - c0.float()).unsqueeze(1)
    H, W = pred.shape
    r0 = r0.clamp(0, H - 1); r1 = r1.clamp(0, H - 1)
    c0 = c0.clamp(0, W - 1); c1 = c1.clamp(0, W - 1)
    Ia = pred[r0, c0].unsqueeze(1)
    Ib = pred[r0, c1].unsqueeze(1)
    Ic = pred[r1, c0].unsqueeze(1)
    Id = pred[r1, c1].unsqueeze(1)
    return (Ia*(1-dr)*(1-dc) + Ib*(1-dr)*dc + Ic*dr*(1-dc) + Id*dr*dc).squeeze(1)

def compute_normals(sdf):
    H, W = sdf.shape
    grad_row = torch.zeros_like(sdf)
    grad_col = torch.zeros_like(sdf)
    sdf_smoothed = sdf
    grad_row[1:-1]   = (sdf_smoothed[2:]   - sdf_smoothed[:-2]) / 2.0
    grad_col[:,1:-1] = (sdf_smoothed[:,2:] - sdf_smoothed[:,:-2]) / 2.0
    grad_row[0]      =  sdf_smoothed[1]  - sdf_smoothed[0]
    grad_row[-1]     =  sdf_smoothed[-1] - sdf_smoothed[-2]
    grad_col[:,0]    =  sdf_smoothed[:,1] - sdf_smoothed[:,0]
    grad_col[:,-1]   =  sdf_smoothed[:,-1]- sdf_smoothed[:,-2]
    return torch.stack([grad_row, grad_col], dim=2)

def sample_normals_at_positions(normals, positions, normalize=True):
    H, W, _ = normals.shape
    r = positions[:, 0]; c = positions[:, 1]
    r0 = r.floor().long(); c0 = c.floor().long()
    r1 = r0 + 1;           c1 = c0 + 1
    dr = (r - r0.float()).unsqueeze(1)
    dc = (c - c0.float()).unsqueeze(1)
    r0 = r0.clamp(0, H-1); r1 = r1.clamp(0, H-1)
    c0 = c0.clamp(0, W-1); c1 = c1.clamp(0, W-1)
    Ia = normals[r0, c0]; Ib = normals[r0, c1]
    Ic = normals[r1, c0]; Id = normals[r1, c1]
    normal_interp = (Ia*(1-dr)*(1-dc) + Ib*(1-dr)*dc + Ic*dr*(1-dc) + Id*dr*dc)
    if normalize:
        norm_val = torch.norm(normal_interp, dim=1, keepdim=True) + 1e-8
        normal_interp = normal_interp / norm_val
    return normal_interp

def extract_zero_crossings_interpolated_positions(sdf_tensor, requires_grad=False):
    epsilon = 1e-8
    positions = []
    H, W = sdf_tensor.shape
    sdf_np = sdf_tensor.detach().cpu().numpy()

    for i in range(H - 1):
        for j in range(W):
            v1 = sdf_np[i, j]; v2 = sdf_np[i + 1, j]
            if v1 == 0: positions.append([i, j])
            elif v2 == 0: positions.append([i + 1, j])
            elif v1 * v2 < 0:
                alpha = abs(v1) / (abs(v1) + abs(v2) + epsilon)
                positions.append([i + alpha, j])

    for i in range(H):
        for j in range(W - 1):
            v1 = sdf_np[i, j]; v2 = sdf_np[i, j + 1]
            if v1 == 0: positions.append([i, j])
            elif v2 == 0: positions.append([i, j + 1])
            elif v1 * v2 < 0:
                alpha = abs(v1) / (abs(v1) + abs(v2) + epsilon)
                positions.append([i, j + alpha])

    if positions:
        return torch.tensor(positions, dtype=torch.float32,
                            device=sdf_tensor.device, requires_grad=requires_grad)
    else:
        return torch.empty((0, 2), dtype=torch.float32,
                           device=sdf_tensor.device, requires_grad=requires_grad)

def compute_chamfer_distance(points1, points2):
    if points1.numel() == 0 or points2.numel() == 0:
        return torch.tensor(float('inf'), device=points1.device)
    diff  = points1.unsqueeze(1) - points2.unsqueeze(0)
    dists = torch.norm(diff, dim=2)
    min_dists1, _ = torch.min(dists, dim=1)
    min_dists2, _ = torch.min(dists, dim=0)
    return -torch.mean(min_dists1) + torch.mean(min_dists2)

def manual_chamfer_grad(pred_sdf, pred_zc, gt_zc, update_scale=1.0, dist_threshold=3.0):
    dSDF = torch.zeros_like(pred_sdf)
    normals = compute_normals(pred_sdf)
    sampled_normals = sample_normals_at_positions(normals, pred_zc)
    gt_zc_cpu   = gt_zc.detach().cpu()
    pred_zc_cpu = pred_zc.detach().cpu()

    for i in range(pred_zc.shape[0]):
        p = pred_zc_cpu[i]
        diff = gt_zc_cpu - p
        dist = torch.norm(diff, dim=1)
        if dist.numel() == 0:
            continue
        min_dist, min_index = torch.min(dist, dim=0)
        if min_dist > dist_threshold:
            continue
        matched_gt = gt_zc_cpu[min_index]
        dl_dp = matched_gt - p
        n = sampled_normals[i]
        n = n / (torch.norm(n) + 1e-8)
        dot_val = torch.dot(dl_dp.to(n.device), n) * update_scale

        r, c = p[0].item(), p[1].item()
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = r0 + 1, c0 + 1
        wr1, wr0 = r - r0, 1 - (r - r0)
        wc1, wc0 = c - c0, 1 - (c - c0)
        H, W = dSDF.shape

        if 0 <= r0 < H and 0 <= c0 < W: dSDF[r0, c0] += dot_val * wr0 * wc0
        if 0 <= r0 < H and 0 <= c1 < W: dSDF[r0, c1] += dot_val * wr0 * wc1
        if 0 <= r1 < H and 0 <= c0 < W: dSDF[r1, c0] += dot_val * wr1 * wc0
        if 0 <= r1 < H and 0 <= c1 < W: dSDF[r1, c1] += dot_val * wr1 * wc1

    return dSDF

# =============================== Loss class ===============================

class SDFChamferLoss(nn.Module):
    """
    EXACT same behavior as your training loop:

    batch_loss = λ_chamfer * |Chamfer(pred_zc, gt_zc)| + λ_sdf * L1(pred_sdf, gt_sdf)

    No hidden tricks. No algorithm change.
    """

    def __init__(self,
                 weight_sdf:     float = 1.0,
                 weight_chamfer: float = 1.0):
        super().__init__()
        self.w_sdf  = float(weight_sdf)
        self.w_ch   = float(weight_chamfer)

    def forward(self, y_pred, y_true):
        pred = y_pred.squeeze(1)
        gt   = y_true.squeeze(1)
        total = 0.0
        for p, g in zip(pred, gt):
            p_zc = extract_zero_crossings_interpolated_positions(p)
            g_zc = extract_zero_crossings_interpolated_positions(g)

            # if either set is empty, skip it
            if p_zc.numel() == 0 or g_zc.numel() == 0:
                loss_ch = 0.0
            else:
                cd = compute_chamfer_distance(p_zc, g_zc)
                loss_ch = torch.abs(cd)

            loss_sdf = F.l1_loss(p, g)
            total += self.w_ch * loss_ch + self.w_sdf * loss_sdf

        return total / pred.size(0)

# expose helpers
__all__ = [
    "SDFChamferLoss",
    "extract_zero_crossings_interpolated_positions",
    "compute_chamfer_distance",
    "manual_chamfer_grad",
    "sample_pred_at_positions",
    "compute_normals",
    "sample_normals_at_positions",
]
