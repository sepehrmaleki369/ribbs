import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_pred_at_positions(pred, positions):
    r, c = positions[:, 0], positions[:, 1]
    r0, c0 = r.floor().long(), c.floor().long()
    r1, c1 = r0 + 1, c0 + 1
    dr, dc = (r - r0.float()).unsqueeze(1), (c - c0.float()).unsqueeze(1)
    H, W = pred.shape
    r0, r1 = r0.clamp(0, H - 1), r1.clamp(0, H - 1)
    c0, c1 = c0.clamp(0, W - 1), c1.clamp(0, W - 1)
    Ia, Ib = pred[r0, c0].unsqueeze(1), pred[r0, c1].unsqueeze(1)
    Ic, Id = pred[r1, c0].unsqueeze(1), pred[r1, c1].unsqueeze(1)
    val = (Ia * (1 - dr) * (1 - dc) +
           Ib * (1 - dr) * dc +
           Ic * dr * (1 - dc) +
           Id * dr * dc)
    return val.squeeze(1)

def compute_normals(sdf):
    H, W = sdf.shape
    grad_row, grad_col = torch.zeros_like(sdf), torch.zeros_like(sdf)
    sdf_smoothed = sdf  # add GaussianBlur here if you wish
    grad_row[1:-1] = (sdf_smoothed[2:] - sdf_smoothed[:-2]) * 0.5
    grad_col[:, 1:-1] = (sdf_smoothed[:, 2:] - sdf_smoothed[:, :-2]) * 0.5
    grad_row[0]  = sdf_smoothed[1] - sdf_smoothed[0]
    grad_row[-1] = sdf_smoothed[-1] - sdf_smoothed[-2]
    grad_col[:, 0]  = sdf_smoothed[:, 1] - sdf_smoothed[:, 0]
    grad_col[:, -1] = sdf_smoothed[:, -1] - sdf_smoothed[:, -2]
    return torch.stack([grad_row, grad_col], dim=2)  # (H,W,2)

def sample_normals_at_positions(normals, positions, normalize=True):
    H, W, _ = normals.shape
    r, c = positions[:, 0], positions[:, 1]
    r0, c0 = r.floor().long(), c.floor().long()
    r1, c1 = r0 + 1, c0 + 1
    dr, dc = (r - r0.float()).unsqueeze(1), (c - c0.float()).unsqueeze(1)
    r0, r1 = r0.clamp(0, H-1), r1.clamp(0, H-1)
    c0, c1 = c0.clamp(0, W-1), c1.clamp(0, W-1)
    Ia, Ib = normals[r0, c0], normals[r0, c1]
    Ic, Id = normals[r1, c0], normals[r1, c1]
    n_interp = (Ia * (1 - dr) * (1 - dc) +
                Ib * (1 - dr) * dc +
                Ic * dr * (1 - dc) +
                Id * dr * dc)
    if normalize:
        n_interp = n_interp / (torch.norm(n_interp, dim=1, keepdim=True) + 1e-8)
    return n_interp  # (N,2)

def extract_zero_crossings_interpolated_positions(sdf_tensor, requires_grad=False):
    eps, pos, H, W = 1e-8, [], *sdf_tensor.shape
    sdf_np = sdf_tensor.detach().cpu().numpy()
    # vertical neighbours
    for i in range(H-1):
        for j in range(W):
            v1, v2 = sdf_np[i, j], sdf_np[i+1, j]
            if v1 == 0:                pos.append([i, j])
            elif v2 == 0:              pos.append([i+1, j])
            elif v1 * v2 < 0:
                alpha = abs(v1) / (abs(v1) + abs(v2) + eps)
                pos.append([i + alpha, j])
    # horizontal neighbours
    for i in range(H):
        for j in range(W-1):
            v1, v2 = sdf_np[i, j], sdf_np[i, j+1]
            if v1 == 0:                pos.append([i, j])
            elif v2 == 0:              pos.append([i, j+1])
            elif v1 * v2 < 0:
                alpha = abs(v1) / (abs(v1) + abs(v2) + eps)
                pos.append([i, j + alpha])
    return (torch.tensor(pos, dtype=torch.float32, device=sdf_tensor.device,
                         requires_grad=requires_grad)
            if pos else
            torch.empty((0, 2), dtype=torch.float32, device=sdf_tensor.device,
                        requires_grad=requires_grad))

def compute_chamfer_distance(points1, points2):
    if points1.numel() == 0 or points2.numel() == 0:
        return torch.tensor(float('inf'), device=points1.device)
    dists = torch.norm(points1.unsqueeze(1) - points2.unsqueeze(0), dim=2)
    return -torch.min(dists, dim=1).values.mean() + torch.min(dists, dim=0).values.mean()

def manual_chamfer_grad(pred_sdf, pred_zc, gt_zc, update_scale=1.0, dist_threshold=3.0):
    # ---------- early‑exit guards ----------
    if pred_zc.numel() == 0 or gt_zc.numel() == 0:
        return torch.zeros_like(pred_sdf)           # no gradient injection
    # ---------------------------------------

    dSDF = torch.zeros_like(pred_sdf)
    normals = compute_normals(pred_sdf)
    n_at_pts = sample_normals_at_positions(normals, pred_zc)  # (N,2)

    gt_cpu   = gt_zc.detach().cpu()
    pred_cpu = pred_zc.detach().cpu()

    for i, p in enumerate(pred_cpu):
        # distance to the *nearest* GT point
        dists = torch.norm(gt_cpu - p, dim=1)
        min_dist, idx = torch.min(dists, dim=0)     # now safe
        if min_dist > dist_threshold:
            continue
        dl_dp = gt_cpu[idx] - p                       # (2,)
        n = n_at_pts[i] / (torch.norm(n_at_pts[i]) + 1e-8)
        dot_val = torch.dot(dl_dp.to(n.device), n) * update_scale
        r, c = p.tolist()
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = r0 + 1, c0 + 1
        w_r1, w_c1 = r - r0, c - c0
        w_r0, w_c0 = 1 - w_r1, 1 - w_c1
        H, W = dSDF.shape
        if 0 <= r0 < H and 0 <= c0 < W: dSDF[r0, c0] += dot_val * w_r0 * w_c0
        if 0 <= r0 < H and 0 <= c1 < W: dSDF[r0, c1] += dot_val * w_r0 * w_c1
        if 0 <= r1 < H and 0 <= c0 < W: dSDF[r1, c0] += dot_val * w_r1 * w_c0
        if 0 <= r1 < H and 0 <= c1 < W: dSDF[r1, c1] += dot_val * w_r1 * w_c1
    return dSDF



class ChamferBoundarySDFLoss(nn.Module):
    """
    Final loss =  pixel_weight * L1(pred, gt)  with an additional boundary
    gradient injected (chamfer_weight * manual_chamfer_grad).
    """

    def __init__(self,
                 pixel_weight: float   = 1.0,
                 chamfer_weight: float = 1.0,
                 update_scale: float   = 100.0,
                 dist_threshold: float = 1.5):
        super().__init__()
        self.pixel_w      = pixel_weight
        self.chamfer_w    = chamfer_weight
        self.update_scale = update_scale
        self.dist_thresh  = dist_threshold

    # ------------------------------------------------------------------ #
    def forward(self, pred_logits: torch.Tensor, gt_sdf: torch.Tensor) -> torch.Tensor:
        """
        pred_logits : (B,1,H,W)  raw UNet outputs (interpreted as SDF)
        gt_sdf      : (B,1,H,W) or (B,H,W)  ground‑truth SDF
        returns     : scalar loss
        """
        if pred_logits.ndim != 4 or pred_logits.size(1) != 1:
            raise ValueError("pred_logits must be (B,1,H,W)")
        pred_sdf = pred_logits[:, 0]          # (B,H,W)

        if gt_sdf.ndim == 4 and gt_sdf.size(1) == 1:
            gt_sdf = gt_sdf[:, 0]             # (B,H,W)
        elif gt_sdf.ndim != 3:
            raise ValueError("gt_sdf must be (B,1,H,W) or (B,H,W)")

        # 1) pixel loss (fully differentiable)
        pixel_loss = F.l1_loss(pred_sdf, gt_sdf)

        # 2) inject boundary gradient via register_hook
        chamfer_vals = []
        for b in range(pred_sdf.size(0)):
            psdf, gsdf = pred_sdf[b], gt_sdf[b]

            with torch.no_grad():
                gt_zc   = extract_zero_crossings_interpolated_positions(gsdf, False)
                pred_zc = extract_zero_crossings_interpolated_positions(psdf, True)

            dSDF = manual_chamfer_grad(psdf, pred_zc, gt_zc,
                                       update_scale=self.update_scale,
                                       dist_threshold=self.dist_thresh)

            psdf.register_hook(lambda g, d=dSDF: g + self.chamfer_w * d)
            chamfer_vals.append(compute_chamfer_distance(pred_zc, gt_zc))

        # 3) composite value (only pixel_loss contributes numerically)
        loss = self.pixel_w * pixel_loss

        # store for optional logging
        self.last_pixel   = pixel_loss.detach()
        self.last_chamfer = (torch.stack(chamfer_vals).mean().detach()
                             if chamfer_vals else torch.tensor(0.0))

        return loss

# TODO: Making truely vectorized later


"""

### 1. **`w_inject`**

This is the weight applied to the **“injection”** term:

```python
inj = torch.sum(pred * dSDF.detach())
```

* **What it measures**: how well the predicted SDF aligns its zero-level set to the ground-truth boundary **along the local normal direction**.
* **Interpretation**: you're “injecting” boundary-normal corrections into the predicted field.
* **Effect of a larger `w_inject`**: you force the network to pay more attention to getting the *orientation and sharpness* of the boundary right.

---

### 2. **`w_pixel`**

This is the weight applied to the **“pixel” (point-based Chamfer) term**:

```python
vals = sample_pred_at_positions(pred, pred_zc)
pix = vals.sum()
```

* **What it measures**: for each zero-crossing point in your prediction, you sample the SDF value *at* that exact subpixel location and sum them.
* **Interpretation**: you're penalizing predicted boundary points that lie *far* from any true boundary—i.e. a point-to-set Chamfer distance.
* **Effect of a larger `w_pixel`**: you encourage the network to place its zero-crossing points *exactly* on the ground-truth boundary, reducing geometric offset.

---

### Putting it together

```python
total_loss = w_inject * inject_term   +   w_pixel * pixel_term
```

* If you set both weights to 1.0, you give equal importance to matching boundary orientation (`inject`) and exact boundary location (`pixel`).
* If your logs show the **pixel term** is numerically much smaller, but you still want it to matter, crank up `w_pixel`.
* Similarly, boost `w_inject` if you need the normals-based alignment to dominate.

---

**In practice** you'll often sweep over a few values (e.g. `w_inject=10, 50, 100`) to see which gives the best final segmentation boundary quality.

"""