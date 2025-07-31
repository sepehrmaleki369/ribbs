import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  
import matplotlib.pyplot as plt
import numpy as np

def sample_pred_at_positions(pred, positions):
    r = positions[:, 0]
    c = positions[:, 1]
    r0 = r.floor().long()
    c0 = c.floor().long()
    r1 = r0 + 1
    c1 = c0 + 1
    dr = (r - r0.float()).unsqueeze(1)
    dc = (c - c0.float()).unsqueeze(1)
    H, W = pred.shape
    r0 = r0.clamp(0, H - 1)
    r1 = r1.clamp(0, H - 1)
    c0 = c0.clamp(0, W - 1)
    c1 = c1.clamp(0, W - 1)
    Ia = pred[r0, c0].unsqueeze(1)
    Ib = pred[r0, c1].unsqueeze(1)
    Ic = pred[r1, c0].unsqueeze(1)
    Id = pred[r1, c1].unsqueeze(1)
    val = (Ia * (1 - dr) * (1 - dc) +
           Ib * (1 - dr) * dc +
           Ic * dr * (1 - dc) +
           Id * dr * dc)
    return val.squeeze(1)

def compute_normals(sdf):
    H, W = sdf.shape
    grad_row = torch.zeros_like(sdf)
    grad_col = torch.zeros_like(sdf)
    # blur = torchvision.transforms.GaussianBlur((3,3), sigma=(1,1))
    # sdf_smoothed = blur(sdf[None])[0]
    sdf_smoothed = sdf
    grad_row[1:-1] = (sdf_smoothed[2:] - sdf_smoothed[:-2]) / 2.0
    grad_col[:, 1:-1] = (sdf_smoothed[:, 2:] - sdf_smoothed[:, :-2]) / 2.0
    grad_row[0]    = sdf_smoothed[1] - sdf_smoothed[0]
    grad_row[-1]   = sdf_smoothed[-1] - sdf_smoothed[-2]
    grad_col[:, 0] = sdf_smoothed[:, 1] - sdf_smoothed[:, 0]
    grad_col[:, -1] = sdf_smoothed[:, -1] - sdf_smoothed[:, -2]
    normals = torch.stack([grad_row, grad_col], dim=2)
    return normals

def sample_normals_at_positions(normals, positions, normalize=True):
    H, W, _ = normals.shape
    r = positions[:, 0]
    c = positions[:, 1]
    r0 = r.floor().long()
    c0 = c.floor().long()
    r1 = r0 + 1
    c1 = c0 + 1
    dr = (r - r0.float()).unsqueeze(1)
    dc = (c - c0.float()).unsqueeze(1)
    r0 = r0.clamp(0, H - 1)
    r1 = r1.clamp(0, H - 1)
    c0 = c0.clamp(0, W - 1)
    c1 = c1.clamp(0, W - 1)
    Ia = normals[r0, c0]
    Ib = normals[r0, c1]
    Ic = normals[r1, c0]
    Id = normals[r1, c1]
    normal_interp = (Ia * (1 - dr) * (1 - dc) +
                     Ib * (1 - dr) * dc +
                     Ic * dr * (1 - dc) +
                     Id * dr * dc)
    
    if normalize:
        norm_val = torch.norm(normal_interp, dim=1, keepdim=True) + 1e-8
        normal_interp = normal_interp / norm_val

    return normal_interp

def extract_zero_crossings_interpolated_positions(sdf_tensor, requires_grad=False):
    """
    Extract zero crossings for a specified direction.
    
    - 'horizontal': Compare vertical neighbors (changes along rows).
    - 'vertical': Compare horizontal neighbors (changes along columns).
    - 'main_diagonal': Compare (i, j) with (i+1, j+1).
    - 'anti_diagonal': Compare (i, j) with (i+1, j-1).
    """
    epsilon = 1e-8
    positions = []
    H, W = sdf_tensor.shape
    sdf_np = sdf_tensor.detach().cpu().numpy()
    
    # Compare vertical neighbors
    for i in range(H - 1):
        for j in range(W):
            v1 = sdf_np[i, j]
            v2 = sdf_np[i + 1, j]
            if v1 == 0:
                positions.append([i, j])
            elif v2 == 0:
                positions.append([i + 1, j])
            elif v1 * v2 < 0:
                # alpha = min(abs(v1), abs(v2)) / (abs(v1) + abs(v2) + epsilon)
                # alpha = abs(v1) / abs(v2)
                alpha = abs(v1) / (abs(v1) + abs(v2) + epsilon)
                row_interp = i + alpha
                positions.append([row_interp, j])
    # Compare horizontal neighbors
    for i in range(H):
        for j in range(W - 1):
            v1 = sdf_np[i, j]
            v2 = sdf_np[i, j + 1]
            if v1 == 0:
                positions.append([i, j])
            elif v2 == 0:
                positions.append([i, j + 1])
            elif v1 * v2 < 0:
                # alpha = min(abs(v1), abs(v2)) / (abs(v1) + abs(v2) + epsilon)
                # alpha = abs(v1) / abs(v2)
                alpha = abs(v1) / (abs(v1) + abs(v2) + epsilon)

                col_interp = j + alpha
                positions.append([i, col_interp])
                    
    if positions:
        return torch.tensor(positions, dtype=torch.float32, device=sdf_tensor.device, requires_grad=requires_grad)
    else:
        return torch.empty((0, 2), dtype=torch.float32, device=sdf_tensor.device, requires_grad=requires_grad)
    
def compute_chamfer_distance(points1, points2):
    if points1.numel() == 0 or points2.numel() == 0:
        return torch.tensor(float('inf'), device=points1.device)
    diff = points1.unsqueeze(1) - points2.unsqueeze(0)
    dists = torch.norm(diff, dim=2)
    min_dists1, _ = torch.min(dists, dim=1)
    min_dists2, _ = torch.min(dists, dim=0)
    return -torch.mean(min_dists1) + torch.mean(min_dists2)


def manual_chamfer_grad(pred_sdf, pred_zc, gt_zc, update_scale=1.0, dist_threshold=3.0):
    """
    Compute a 'manual' gradient for Chamfer-like boundary alignment.
    We ignore any predicted zero crossings that are too far from
    all ground-truth zero crossings, i.e., they are considered spurious.
    
    Args:
        pred_sdf (Tensor): [H, W], the predicted SDF.
        pred_zc  (Tensor): [N, 2], predicted zero-crossing positions (row, col).
        gt_zc    (Tensor): [M, 2], ground-truth zero-crossing positions (row, col).
        update_scale (float): scaling factor for the gradient magnitude.
        dist_threshold (float): maximum distance for a predicted ZC to be considered valid.
    
    Returns:
        dSDF (Tensor): [H, W], the gradient w.r.t. pred_sdf for the Chamfer loss.
    """
    # Initialize gradient buffer
    dSDF = torch.zeros_like(pred_sdf)
    
    # Compute normals on the entire SDF
    normals = compute_normals(pred_sdf)  # shape: [H, W, 2]
    sampled_normals = sample_normals_at_positions(normals, pred_zc)  # shape: [N, 2]
    
    # Move data to CPU for distance computations
    gt_zc_cpu = gt_zc.detach().cpu()
    pred_zc_cpu = pred_zc.detach().cpu()

    # Loop over each predicted zero crossing
    for i in range(pred_zc.shape[0]):
        # p is (row, col)
        p = pred_zc_cpu[i]

        # Find nearest ground-truth crossing
        diff = gt_zc_cpu - p  # shape: [M, 2]
        dist = torch.norm(diff, dim=1)  # [M]
        min_dist, min_index = torch.min(dist, dim=0)

        # If the predicted ZC is too far from every GT ZC, skip it (spurious)
        if min_dist > dist_threshold:
            continue

        # Otherwise, compute the usual chamfer update
        matched_gt = gt_zc_cpu[min_index]
        dl_dp = matched_gt - p  # direction from predicted ZC to GT ZC

        # Project onto local normal at p
        n = sampled_normals[i]
        n_norm = torch.norm(n) + 1e-8
        n = n / n_norm
        dot_val = torch.dot(dl_dp.to(n.device), n) * update_scale

        # Distribute the gradient bilinearly among neighboring pixels
        r, c = p[0].item(), p[1].item()
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = r0 + 1, c0 + 1
        w_r1 = r - r0
        w_r0 = 1 - w_r1
        w_c1 = c - c0
        w_c0 = 1 - w_c1

        H, W = dSDF.shape
        if 0 <= r0 < H and 0 <= c0 < W:
            dSDF[r0, c0] += dot_val * w_r0 * w_c0
        if 0 <= r0 < H and 0 <= c1 < W:
            dSDF[r0, c1] += dot_val * w_r0 * w_c1
        if 0 <= r1 < H and 0 <= c0 < W:
            dSDF[r1, c0] += dot_val * w_r1 * w_c0
        if 0 <= r1 < H and 0 <= c1 < W:
            dSDF[r1, c1] += dot_val * w_r1 * w_c1

    return dSDF

def update_pred_sdf(gt_sdf, pred_sdf, gt_zc, pred_zc, optimizer, iter_num, update_scale=1.0, dist_threshold=3.0):
    """
    Updated function that uses a MeshSDF-like trick with bilinear gradient distribution for the Chamfer term,
    combined with a pixel-wise SDF loss.
    """
    pred_sdf.requires_grad_(True)
    optimizer.zero_grad()

    dSDF_chamfer = manual_chamfer_grad(pred_sdf, pred_zc, gt_zc, update_scale=update_scale, dist_threshold=dist_threshold)
    with torch.no_grad():
        pred_sdf.grad = dSDF_chamfer.clone()

    zc_values = sample_pred_at_positions(pred_sdf, pred_zc)
    loss_pred = (zc_values).sum()
    loss_pred.backward(retain_graph=True)
    
    normals = compute_normals(pred_sdf)
    sampled_normals = sample_normals_at_positions(normals, pred_zc)
    dl_dx_fake = torch.zeros_like(pred_zc)
    dl_ds_per_point = - torch.sum(sampled_normals * dl_dx_fake, dim=1)
    combined_loss = torch.sum(dl_ds_per_point)
    combined_loss.backward()

    optimizer.step()

    return pred_sdf


