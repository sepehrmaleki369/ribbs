import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------
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
    r0 = r0.clamp(0, H - 1); r1 = r1.clamp(0, H - 1)
    c0 = c0.clamp(0, W - 1); c1 = c1.clamp(0, W - 1)
    Ia = pred[r0, c0].unsqueeze(1)
    Ib = pred[r0, c1].unsqueeze(1)
    Ic = pred[r1, c0].unsqueeze(1)
    Id = pred[r1, c1].unsqueeze(1)
    return (Ia * (1 - dr) * (1 - dc)
            + Ib * (1 - dr) * dc
            + Ic * dr * (1 - dc)
            + Id * dr * dc).squeeze(1)


def compute_normals(sdf):
    H, W = sdf.shape
    grad_r = torch.zeros_like(sdf)
    grad_c = torch.zeros_like(sdf)
    grad_r[1:-1] = (sdf[2:] - sdf[:-2]) / 2.0
    grad_r[0]    = sdf[1] - sdf[0]
    grad_r[-1]   = sdf[-1] - sdf[-2]
    grad_c[:,1:-1] = (sdf[:,2:] - sdf[:,:-2]) / 2.0
    grad_c[:,0]    = sdf[:,1] - sdf[:,0]
    grad_c[:,-1]   = sdf[:,-1] - sdf[:,-2]
    return torch.stack([grad_r, grad_c], dim=2)


def extract_zero_crossings_interpolated_positions(sdf, requires_grad=False):
    eps = 1e-8
    H, W = sdf.shape
    arr = sdf.detach().cpu().numpy()
    pts = []
    # vertical
    for i in range(H-1):
        for j in range(W):
            v1,v2 = arr[i,j], arr[i+1,j]
            if v1==0: pts.append([i,j])
            elif v2==0: pts.append([i+1,j])
            elif v1*v2<0:
                alpha = abs(v1)/(abs(v1)+abs(v2)+eps)
                pts.append([i+alpha,j])
    # horizontal
    for i in range(H):
        for j in range(W-1):
            v1,v2 = arr[i,j], arr[i,j+1]
            if v1==0: pts.append([i,j])
            elif v2==0: pts.append([i,j+1])
            elif v1*v2<0:
                alpha = abs(v1)/(abs(v1)+abs(v2)+eps)
                pts.append([i,j+alpha])
    if pts:
        return torch.tensor(pts, dtype=torch.float32, device=sdf.device, requires_grad=requires_grad)
    return torch.empty((0,2), dtype=torch.float32, device=sdf.device, requires_grad=requires_grad)


def manual_chamfer_grad(pred, pred_zc, gt_zc, update_scale=1.0, dist_threshold=3.0):
    if pred_zc.numel() == 0 or gt_zc.numel() == 0:
        return torch.zeros_like(pred)
    
    dSDF = torch.zeros_like(pred)
    normals = compute_normals(pred)
    sampled = []
    for p in pred_zc:
        r,c = p[0].item(), p[1].item()
        r0,c0 = int(r), int(c)
        r1,c1 = r0+1, c0+1
        H,W = pred.shape
        r0 = max(0,min(r0,H-1));   c0 = max(0,min(c0,W-1))
        r1 = max(0,min(r1,H-1));   c1 = max(0,min(c1,W-1))
        ar,ac = r-r0, c-c0
        Ia = normals[r0,c0]; Ib = normals[r0,c1]
        Ic = normals[r1,c0]; Id = normals[r1,c1]
        n = Ia*(1-ar)*(1-ac) + Ib*(1-ar)*ac + Ic*ar*(1-ac) + Id*ar*ac
        sampled.append(n/(n.norm()+1e-8))
    sampled = torch.stack(sampled,0) if sampled else torch.empty((0,2),device=pred.device)
    gt_pts=gt_zc.detach().cpu(); pr_pts=pred_zc.detach().cpu()
    for i,p in enumerate(pr_pts):
        # diffs = gt_pts-p; dists = torch.norm(diffs,dim=1)
        # md,idx = torch.min(dists,0)
        diffs = gt_pts - p
        if diffs.shape[0] == 0:
            continue
        dists = torch.norm(diffs, dim=1)
        md, idx = torch.min(dists, 0)
        if md>dist_threshold: continue
        dir = (gt_pts[idx]-p).to(pred.device)
        n = sampled[i]
        dot = torch.dot(dir,n)*update_scale
        r,c=p[0].item(),p[1].item()
        r0,c0=int(r),int(c); r1,c1=r0+1,c0+1; ar,ac=r-r0,c-c0
        for rr,cc,w in [(r0,c0,(1-ar)*(1-ac)),(r0,c1,(1-ar)*ac),(r1,c0,ar*(1-ac)),(r1,c1,ar*ac)]:
            if 0<=rr<dSDF.shape[0] and 0<=cc<dSDF.shape[1]:
                dSDF[rr,cc]+=dot*w
    return dSDF

# ---------------------------------------------------------------------
# Loss Class
# ---------------------------------------------------------------------
class ChamferBoundarySDFLoss(nn.Module):
    def __init__(self, update_scale=1.0, dist_threshold=3.0, w_inject=1.0, w_pixel=1.0):
        super().__init__()
        self.update_scale, self.dist_threshold = update_scale, dist_threshold
        self.w_inject, self.w_pixel = w_inject, w_pixel
        self.latest = {}
    def forward(self, pred_sdf, gt_sdf):
        # [B,1,H,W] -> [B,H,W]
        if pred_sdf.dim() == 4:
            pred_sdf = pred_sdf.squeeze(1)
        if gt_sdf.dim() == 4:
            gt_sdf = gt_sdf.squeeze(1)
        if pred_sdf.dim()==2:
            pred_sdf,gt_sdf = pred_sdf.unsqueeze(0), gt_sdf.unsqueeze(0)
        batch_inj,batch_pix=[],[]
        for pred,gt in zip(pred_sdf,gt_sdf):
            gt_zc = extract_zero_crossings_interpolated_positions(gt)
            pred_zc = extract_zero_crossings_interpolated_positions(pred.detach())
            dSDF = manual_chamfer_grad(pred,pred_zc,gt_zc,self.update_scale,self.dist_threshold)
            inj = torch.sum(pred * dSDF.detach())
            vals=sample_pred_at_positions(pred,pred_zc)
            pix = vals.sum() if vals.numel() else torch.tensor(0.,device=pred.device)
            batch_inj.append(inj); batch_pix.append(pix)
        inject = torch.stack(batch_inj).mean()
        pixel  = torch.stack(batch_pix).mean()
        total  = self.w_inject*inject + self.w_pixel*pixel
        self.latest={"inject":inject.item(),"pixel":pixel.item()}
        return total

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