from typing import Any, Dict, List, Optional, Union

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt


def _gather_from_outputs(batch, outputs, pl_module):
    """
    Extract input, ground‑truth and prediction tensors from ``batch`` and
    ``outputs`` without re‑running the model.
    """
    x = batch[pl_module.input_key].float()
    y = batch[pl_module.target_key].float()
    if y.dim() == 3:
        y = y.unsqueeze(1)  # (B, 1, H, W)
    preds = outputs.get("predictions").float()
    return x.cpu(), y.cpu(), preds.detach().cpu()


def _signed_scale(arr: torch.Tensor, pos_max: float, neg_min: float) -> torch.Tensor:
    """Scale *signed* ``arr`` so that

    * 0 → 0
    * (arr > 0) are mapped linearly onto ``(0,  +1]`` where the *largest* value
      becomes +1.
    * (arr < 0) are mapped linearly onto ``[−1, 0)`` where the *most‑negative*
      value becomes −1.

    Positive and negative parts are treated independently so that sign symmetry
    is preserved.
    """
    if pos_max <= 0 and neg_min >= 0:  # all‑zero tensor
        return torch.zeros_like(arr)

    scaled = arr.clone()
    if pos_max > 0:
        pos_mask = scaled > 0
        scaled[pos_mask] = scaled[pos_mask] / pos_max
    if neg_min < 0:  # remember: neg_min is ≤ 0
        neg_mask = scaled < 0
        scaled[neg_mask] = scaled[neg_mask] / abs(neg_min)
    return scaled


class SamplePlotCallback(pl.Callback):
    """Log side‑by‑side *input | ground‑truth | prediction* panels during
    training/validation with **independent colour scaling** for ground‑truth and
    prediction maps.

    Parameters
    ----------
    num_samples:
        Maximum number of examples to visualise each epoch.
    cmap:
        Colormap passed to ``matplotlib.pyplot.imshow`` for signed maps
        (default: ``"coolwarm"``).
    """

    def __init__(self, num_samples: int = 5, cmap: str = "coolwarm"):
        super().__init__()
        self.num_samples = num_samples
        self.cmap = cmap
        self._reset()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _reset(self):
        self._images, self._gts, self._preds = [], [], []
        self._count = 0

    # epoch hooks --------------------------------------------------------
    def on_train_epoch_start(self, *_):
        self._reset()

    def on_validation_epoch_start(self, *_):
        self._reset()

    # batch hooks --------------------------------------------------------
    def _collect(self, batch, outputs, pl_module):
        if self._count >= self.num_samples:
            return
        x, y, preds = _gather_from_outputs(batch, outputs, pl_module)
        take = min(self.num_samples - self._count, x.size(0))
        self._images.append(x[:take])
        self._gts.append(y[:take])
        self._preds.append(preds[:take])
        self._count += take

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        self._collect(batch, outputs, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        self._collect(batch, outputs, pl_module)

    # plotting -----------------------------------------------------------
    def _plot_and_log(self, tag: str, trainer):
        imgs = torch.cat(self._images, 0)   # (N, C, H, W)
        gts  = torch.cat(self._gts, 0)      # (N, 1, H, W)
        preds = torch.cat(self._preds, 0)   # (N, 1, H, W)
        n = imgs.size(0)

        # --- independent signed scaling for GT and prediction -------------
        pos_max_gt,  neg_min_gt  = float(gts.max()),  float(gts.min())
        pos_max_pr,  neg_min_pr  = float(preds.max()), float(preds.min())

        gts_scaled   = _signed_scale(gts,   pos_max_gt, neg_min_gt)
        preds_scaled = _signed_scale(preds, pos_max_pr, neg_min_pr)

        # --- plotting -----------------------------------------------------
        fig, axes = plt.subplots(n, 3, figsize=(9, n * 3), tight_layout=True)
        if n == 1:
            axes = axes[None, :]  # always treat as 2‑D array [row, col]

        for i in range(n):
            img = imgs[i].permute(1, 2, 0)  # CHW → HWC
            gt  = gts_scaled[i, 0]
            pr  = preds_scaled[i, 0]

            # input
            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 0].set_title("input")
            axes[i, 0].axis("off")

            # ground‑truth (own scale)
            axes[i, 1].imshow(gt, cmap=self.cmap, vmin=-1, vmax=1)
            axes[i, 1].set_title("gt (ind. scaled)")
            axes[i, 1].axis("off")

            # prediction (own scale)
            axes[i, 2].imshow(pr, cmap=self.cmap, vmin=-1, vmax=1)
            axes[i, 2].set_title("pred (ind. scaled)")
            axes[i, 2].axis("off")

        trainer.logger.experiment.add_figure(
            f"{tag}_samples", fig, global_step=trainer.current_epoch
        )
        plt.close(fig)

    # epoch completion ---------------------------------------------------
    def on_train_epoch_end(self, trainer, pl_module):
        if self._count > 0:
            self._plot_and_log("train", trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._count > 0:
            self._plot_and_log("validation", trainer)

class SamplePlot3DCallback(SamplePlotCallback):
    """Extends SamplePlotCallback to also handle 3D volumes by projecting
    along Z into XY, XZ, YZ slices (via max/min/mean)."""

    def __init__(
        self,
        num_samples: int = 5,
        cmap: str = "coolwarm",
        projection: str = "max",  # one of "max","min","mean"
    ):
        super().__init__(num_samples=num_samples, cmap=cmap)
        assert projection in ("max", "min", "mean")
        # map name → torch reduction
        self._proj_fn = {
            "max": lambda x, dim: torch.max(x, dim=dim)[0],
            "min": lambda x, dim: torch.min(x, dim=dim)[0],
            "mean": lambda x, dim: torch.mean(x, dim=dim),
        }[projection]

    def _project_volume(self, vol: torch.Tensor) -> Dict[str, torch.Tensor]:
        # vol: (C,Z,H,W) or (Z,H,W)
        if vol.ndim == 4:
            # collapse channels first via max
            vol = torch.max(vol, dim=0)[0]
        # now vol is (Z,H,W)
        xy = self._proj_fn(vol, dim=0)           # → (H,W)
        xz = self._proj_fn(vol, dim=1)           # → (Z,W)
        yz = self._proj_fn(vol, dim=2).transpose(0, 1)  # → (H,Z)
        return {"XY": xy, "XZ": xz, "YZ": yz}

    def _plot_and_log(self, tag: str, trainer):
        imgs = torch.cat(self._images, 0)
        gts  = torch.cat(self._gts, 0)
        preds = torch.cat(self._preds, 0)

        # detect 3D volumes by rank
        if imgs.dim() == 5:  # N×C×Z×H×W
            n = imgs.size(0)
            fig, axes = plt.subplots(
                n, 9, figsize=(9 * 3, n * 3), tight_layout=True
            )
            if n == 1:
                axes = axes[None, :]

            for i in range(n):
                # project each modality
                mods = {
                    "input": imgs[i].cpu(),
                    "gt":    gts[i].cpu().squeeze(0),
                    "pred":  preds[i].cpu().squeeze(0),
                }
                projs = {m: self._project_volume(v) for m, v in mods.items()}

                for col, m in enumerate(("input", "gt", "pred")):
                    for row, view in enumerate(("XY", "XZ", "YZ")):
                        ax = axes[i, col*3 + row]
                        arr = projs[m][view].numpy()
                        cmap = "gray" if m == "input" else self.cmap
                        ax.imshow(arr, cmap=cmap)
                        ax.set_title(f"{m}-{view}")
                        ax.axis("off")

            trainer.logger.experiment.add_figure(
                f"{tag}_3d_samples", fig, global_step=trainer.current_epoch
            )
            plt.close(fig)

        else:
            # fallback to the original 2D behavior
            super()._plot_and_log(tag, trainer)
