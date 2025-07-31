import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """Ensure the tensor is 2‑D (H×W). If it has a leading singleton dimension
    – e.g. (1,H,W) or (B,1,H,W) after indexing over B – squeeze it. Raises if
    more than one channel is present."""
    if t.dim() == 3:
        # (C,H,W) – expect C==1
        if t.size(0) != 1:
            raise ValueError("SDF tensor has more than one channel; please select the channel to use.")
        return t.squeeze(0)
    if t.dim() != 2:
        raise ValueError(f"Expected a 2‑D grid; got shape {tuple(t.shape)}")
    return t


def bilinear_sample(img: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Vectorised bilinear sampling of a single‑channel image.

    Args:
        img    (H×W)  : SDF or any 2‑D tensor.
        coords (N×2)  : [row, col] floating‑point coordinates.

    Returns:
        (N,) tensor – sampled values.
    """
    H, W = img.shape
    r, c = coords[:, 0], coords[:, 1]
    r0 = torch.floor(r).long().clamp(0, H - 1)
    c0 = torch.floor(c).long().clamp(0, W - 1)
    r1 = (r0 + 1).clamp(0, H - 1)
    c1 = (c0 + 1).clamp(0, W - 1)
    ar = r - r0.float()
    ac = c - c0.float()

    Ia = img[r0, c0]
    Ib = img[r0, c1]
    Ic = img[r1, c0]
    Id = img[r1, c1]
    return Ia * (1 - ar) * (1 - ac) + Ib * (1 - ar) * ac + Ic * ar * (1 - ac) + Id * ar * ac


def compute_normals(sdf: torch.Tensor) -> torch.Tensor:
    """Central‑difference normals (H×W×2)."""
    grad_r = torch.zeros_like(sdf)
    grad_c = torch.zeros_like(sdf)
    grad_r[1:-1] = 0.5 * (sdf[2:] - sdf[:-2])
    grad_r[0] = sdf[1] - sdf[0]
    grad_r[-1] = sdf[-1] - sdf[-2]
    grad_c[:, 1:-1] = 0.5 * (sdf[:, 2:] - sdf[:, :-2])
    grad_c[:, 0] = sdf[:, 1] - sdf[:, 0]
    grad_c[:, -1] = sdf[:, -1] - sdf[:, -2]
    return torch.stack((grad_r, grad_c), dim=-1)


# -----------------------------------------------------------------------------
# Zero‑crossing extraction (fully vectorised)
# -----------------------------------------------------------------------------

def extract_zero_crossings(sdf: torch.Tensor, *, eps: float = 1e-8, requires_grad: bool = False) -> torch.Tensor:
    """Return (N×2) sub‑pixel positions of the 0‑level set using bilinear interpolation."""
    sdf = _to_2d(sdf)
    H, W = sdf.shape
    device = sdf.device

    # vertical edges: between rows
    v1, v2 = sdf[:-1, :], sdf[1:, :]
    mask_v = (v1 * v2) < 0
    alpha_v = v1.abs() / (v1.abs() + v2.abs() + eps)
    rs_v = torch.arange(H - 1, device=device).unsqueeze(1).expand(H - 1, W).float() + alpha_v
    cs_v = torch.arange(W, device=device).unsqueeze(0).expand(H - 1, W).float()
    pts_v = torch.stack((rs_v[mask_v], cs_v[mask_v]), dim=1)

    # horizontal edges: between columns
    h1, h2 = sdf[:, :-1], sdf[:, 1:]
    mask_h = (h1 * h2) < 0
    alpha_h = h1.abs() / (h1.abs() + h2.abs() + eps)
    rs_h = torch.arange(H, device=device).unsqueeze(1).expand(H, W - 1).float()
    cs_h = torch.arange(W - 1, device=device).unsqueeze(0).expand(H, W - 1).float() + alpha_h
    pts_h = torch.stack((rs_h[mask_h], cs_h[mask_h]), dim=1)

    # exact zeros
    mask_z = (sdf == 0)
    if mask_z.any():
        rz, cz = torch.where(mask_z)
        pts_z = torch.stack((rz.float(), cz.float()), dim=1)
        pts = torch.cat((pts_z, pts_v, pts_h), dim=0)
    else:
        pts = torch.cat((pts_v, pts_h), dim=0)

    if pts.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device, requires_grad=requires_grad)
    return pts.requires_grad_(requires_grad)


# -----------------------------------------------------------------------------
# Vectorised Chamfer gradient
# -----------------------------------------------------------------------------

def chamfer_grad_vectorised(pred: torch.Tensor, pred_zc: torch.Tensor, gt_zc: torch.Tensor,
                            *, update_scale: float = 1.0, dist_threshold: float = 3.0) -> torch.Tensor:
    """Vectorised replacement for manual_chamfer_grad."""

    pred2d = _to_2d(pred)  # ensure (H×W)

    if pred_zc.numel() == 0 or gt_zc.numel() == 0:
        return torch.zeros_like(pred2d)

    H, W = pred2d.shape
    device = pred2d.device

    # 1. normals at pred zero‑crossings (bilinear‑interpolated)
    normals = compute_normals(pred2d)  # H×W×2
    r, c = pred_zc[:, 0], pred_zc[:, 1]
    r0 = torch.floor(r).long().clamp(0, H - 1)
    c0 = torch.floor(c).long().clamp(0, W - 1)
    r1 = (r0 + 1).clamp(0, H - 1)
    c1 = (c0 + 1).clamp(0, W - 1)
    ar = r - r0.float()
    ac = c - c0.float()

    n00 = normals[r0, c0]
    n01 = normals[r0, c1]
    n10 = normals[r1, c0]
    n11 = normals[r1, c1]
    n = (
        n00 * (1 - ar).unsqueeze(1) * (1 - ac).unsqueeze(1)
        + n01 * (1 - ar).unsqueeze(1) * ac.unsqueeze(1)
        + n10 * ar.unsqueeze(1) * (1 - ac).unsqueeze(1)
        + n11 * ar.unsqueeze(1) * ac.unsqueeze(1)
    )
    n = n / (n.norm(dim=1, keepdim=True) + 1e-8)  # N × 2 (unit vectors)

    # 2. nearest GT point for each pred point
    diff = gt_zc.unsqueeze(0) - pred_zc.unsqueeze(1)  # Np × Ng × 2 (gt − pred)
    dist = diff.norm(dim=-1)                          # Np × Ng
    min_dist, idx = dist.min(dim=1)                   # length Np
    mask = min_dist <= dist_threshold                 # ignore far matches
    dir_vec = diff[torch.arange(pred_zc.size(0), device=device), idx]  # Np × 2

    dot = (dir_vec * n).sum(dim=1) * update_scale
    dot = dot * mask.float()

    # 3. accumulate into dSDF using scatter‑add (bilinear weights)
    w00 = (1 - ar) * (1 - ac)
    w01 = (1 - ar) * ac
    w10 = ar * (1 - ac)
    w11 = ar * ac

    flat_index = lambda rr, cc: rr * W + cc
    idx00 = flat_index(r0, c0)
    idx01 = flat_index(r0, c1)
    idx10 = flat_index(r1, c0)
    idx11 = flat_index(r1, c1)

    indices = torch.cat((idx00, idx01, idx10, idx11), dim=0)  # 4N
    contribs = torch.cat((dot * w00, dot * w01, dot * w10, dot * w11), dim=0)

    dflat = torch.zeros(H * W, device=device).index_add(0, indices, contribs)
    return dflat.view(H, W)


# -----------------------------------------------------------------------------
# Vectorised loss module
# -----------------------------------------------------------------------------

class ChamferBoundarySDFLossVec(nn.Module):
    """Drop‑in vectorised replacement for ChamferBoundarySDFLoss."""

    def __init__(self, *, update_scale: float = 1.0, dist_threshold: float = 3.0,
                 w_inject: float = 1.0, w_pixel: float = 1.0):
        super().__init__()
        self.update_scale = update_scale
        self.dist_threshold = dist_threshold
        self.w_inject = w_inject
        self.w_pixel = w_pixel
        self.latest: dict[str, float] = {}

    def forward(self, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> torch.Tensor:
        # Expect inputs (B,H,W) or (B,1,H,W) or (H,W)
        if pred_sdf.dim() == 2:
            pred_sdf = pred_sdf.unsqueeze(0)
            gt_sdf = gt_sdf.unsqueeze(0)
        if pred_sdf.dim() == 4 and pred_sdf.size(1) == 1:
            pred_sdf = pred_sdf[:, 0]  # drop channel dim → (B,H,W)
            gt_sdf = gt_sdf[:, 0]

        inject_terms, pixel_terms = [], []

        for pred, gt in zip(pred_sdf, gt_sdf):
            pred2d = _to_2d(pred)
            gt2d = _to_2d(gt)

            gt_zc = extract_zero_crossings(gt2d)
            pred_zc = extract_zero_crossings(pred2d.detach())  # keep graph small
            dSDF = chamfer_grad_vectorised(pred2d, pred_zc, gt_zc,
                                            update_scale=self.update_scale,
                                            dist_threshold=self.dist_threshold)
            inject_terms.append(torch.sum(pred2d * dSDF.detach()))
            if pred_zc.numel():
                pixel_terms.append(bilinear_sample(pred2d, pred_zc).sum())
            else:
                pixel_terms.append(torch.tensor(0., device=pred.device))

        inject = torch.stack(inject_terms).mean()
        pixel = torch.stack(pixel_terms).mean()
        total = self.w_inject * inject + self.w_pixel * pixel

        self.latest = {"inject": inject.item(), "pixel": pixel.item()}
        return total
