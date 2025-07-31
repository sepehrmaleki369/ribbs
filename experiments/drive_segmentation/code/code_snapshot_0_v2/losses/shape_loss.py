import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import measure
from scipy.ndimage import distance_transform_edt as edt

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def mask_to_sdf(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a binary mask to a signed distance field (SDF).
    """
    a = mask.cpu().numpy().astype(np.uint8)
    sdf_np = edt(1 - a) - edt(a)
    return torch.from_numpy(sdf_np).float().to(mask.device)


def interp_bilinear(img: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """
    Bilinear interpolation of values from `img` at floating-point coords (`ys`, `xs`).
    """
    H, W = img.shape
    y0 = ys.floor().long().clamp(0, H - 2)
    x0 = xs.floor().long().clamp(0, W - 2)
    y1, x1 = y0 + 1, x0 + 1
    wy1, wx1 = ys - y0.float(), xs - x0.float()
    wy0, wx0 = 1 - wy1, 1 - wx1
    return (
        img[y0, x0] * (wy0 * wx0) +
        img[y0, x1] * (wy0 * wx1) +
        img[y1, x0] * (wy1 * wx0) +
        img[y1, x1] * (wy1 * wx1)
    )


# -----------------------------------------------------------------------------
# Differentiable marching squares core
# -----------------------------------------------------------------------------
class IsoCurve2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sdf: torch.Tensor, contour_level: float, pad_width: int):
        """
        Extract contour vertices at `contour_level` with `pad_width` for gradient.
        """
        b, _, H, W = sdf.shape
        verts_out, meta = [], []
        cpu = sdf.detach().cpu().numpy()
        for i in range(b):
            vb, ib, off = [], [], 0
            for cont in measure.find_contours(cpu[i, 0], contour_level):
                pts = torch.from_numpy(cont).float().to(sdf.device)
                y, x = pts[:, 0], pts[:, 1]
                K = len(pts)
                ib.append((y, x, off, off + K))
                vb.append(torch.stack([x, y], 1))
                off += K
            verts_out.append(torch.cat(vb, 0) if vb else torch.empty(0, 2, device=sdf.device))
            meta.append(ib)
        ctx.save_for_backward(sdf)
        ctx.meta = meta
        ctx.pad = pad_width
        return tuple(verts_out)

    @staticmethod
    def backward(ctx, grad_vs):
        sdf, = ctx.saved_tensors
        b, _, H, W = sdf.shape
        # create gradient tensor matching sdf dtype (handles half precision)
        grad_sdf = torch.zeros_like(sdf)
        p = F.pad(sdf, (ctx.pad,)*4, mode='replicate')
        dx = 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2])
        dy = 0.5 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
        for i, ib in enumerate(ctx.meta):
            gv = grad_vs[i]
            if gv is None:
                continue
            for y, x, s, e in ib:
                gseg = gv[s:e]
                if gseg.numel() == 0:
                    continue
                dx_v = interp_bilinear(dx[i, 0], y, x)
                dy_v = interp_bilinear(dy[i, 0], y, x)
                normals = F.normalize(torch.stack([dx_v, dy_v], 1), dim=1, eps=1e-6)
                scal = -(gseg * normals).sum(1)
                # accumulate into grad_sdf, casting values to match dtype
                y0 = y.floor().long().clamp(0, H - 2)
                x0 = x.floor().long().clamp(0, W - 2)
                y1, x1 = y0 + 1, x0 + 1
                wy1, wx1 = y - y0.float(), x - x0.float()
                wy0, wx0 = 1 - wy1, 1 - wx1
                for (rr, cc, ww) in [
                        (y0, x0, wy0 * wx0), (y0, x1, wy0 * wx1),
                        (y1, x0, wy1 * wx0), (y1, x1, wy1 * wx1)
                    ]:
                    val = (scal * ww).to(dtype=grad_sdf.dtype)
                    grad_sdf[i, 0].index_put_((rr, cc), val, accumulate=True)
        return (grad_sdf, None, None)


def extract_curve(sdf: torch.Tensor, contour_level: float = 0.0, pad_width: int = 1):
    """
    Wrapper to extract curves; passes pad_width for backward.
    """
    return IsoCurve2D.apply(sdf, contour_level, pad_width)


# -----------------------------------------------------------------------------
# ShapeLoss: fully parameterized
# -----------------------------------------------------------------------------
class ShapeLoss(nn.Module):
    def __init__(
        self,
        chamfer_weight: float = 1.0,
        occupancy_weight: float = 2.0,
        eikonal_weight: float = 0.05,
        occ_band: float = 1.5,
        softplus_beta: float = 1.0,
        softplus_threshold: float = 10.0,
        pad_width: int = 1,
        contour_level: float = 0.0,
    ):
        super().__init__()
        self.w_ch = chamfer_weight
        self.w_occ = occupancy_weight
        self.w_eik = eikonal_weight
        self.band = occ_band
        self.beta = softplus_beta
        self.threshold = softplus_threshold
        self.pad = pad_width
        self.contour_level = contour_level

    def forward(self, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> torch.Tensor:
        # Extract vertex sets
        p_v, = extract_curve(pred_sdf, self.contour_level, self.pad)
        g_v, = extract_curve(gt_sdf, self.contour_level, self.pad)

        # Chamfer term
        cham = torch.tensor(0.0, device=pred_sdf.device, dtype=pred_sdf.dtype)
        if p_v.numel() and g_v.numel():
            D = torch.cdist(p_v, g_v)
            cham = D.min(1).values.mean() + D.min(0).values.mean()

        # Occupancy term
        inside = torch.sigmoid(-gt_sdf / self.band)
        soft_out = F.softplus(pred_sdf, beta=self.beta, threshold=self.threshold)
        soft_in = F.softplus(-pred_sdf, beta=self.beta, threshold=self.threshold)
        occ = (soft_out * inside).mean() + (soft_in * (1 - inside)).mean()

        # Eikonal term
        p = F.pad(pred_sdf, (self.pad,)*4, mode='replicate')
        gx = 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2])
        gy = 0.5 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
        mag = torch.sqrt(gx.square() + gy.square() + 1e-6)   # or use .clamp(min=1e-6)
        eik = (mag.sub(1.0)).abs().mean()

        return cham * self.w_ch + occ * self.w_occ + eik * self.w_eik


# -----------------------------------------------------------------------------
# Example smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    H, W = 64, 64
    gt_mask = torch.zeros(H, W)
    gt_mask[H//2-8:H//2+8, :] = 1
    pred_mask = gt_mask.clone()
    pred_mask[:, W//2-3:W//2+3] = 0

    gt_sdf = mask_to_sdf(gt_mask)[None, None]
    pred_sdf = nn.Parameter(mask_to_sdf(pred_mask)[None, None])

    loss_fn = ShapeLoss()
    opt = torch.optim.Adam([pred_sdf], lr=0.5)
    for it in range(5):
        opt.zero_grad()
        loss = loss_fn(pred_sdf, gt_sdf)
        loss.backward()
        opt.step()
        print(f"step {it:02d}   loss = {loss.item():.4f}")
