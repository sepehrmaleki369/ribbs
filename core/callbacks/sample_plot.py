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

class SamplePlot3DCallback(pl.Callback):
    """
    Callback to log sample slices from 3D volumes (Z×H×W) during training/validation.

    Args:
        num_samples (int): number of samples to log each epoch.
        projection_view (str): one of 'XY', 'XZ', 'YZ' to project on.
        cfg (Optional[Dict[str, Dict[str, Any]]]):
            per-modality settings, e.g.: 
            {
              'input': {'cmap': 'gray', 'projection': 'max'},
              'gt':    {'cmap': 'viridis', 'projection': 'min'},
              'pred':  {'cmap': 'plasma', 'projection': 'min'},
            }
    """
    def __init__(self, config):
        super().__init__()

        self.num_samples = config['num_samples']
        self.projection_view = config.get('projection_view', 'YZ')
        self.cfg = config.get('cfg')
        # default settings per modality
        self.default_modals = {
            'input': {'cmap': 'gray', 'projection': 'max'},
            'gt':    {'cmap': 'gray', 'projection': 'min'},
            'pred':  {'cmap': 'gray', 'projection': 'min'},
        }
        # map view to axis: XY->Z(0), XZ->Y(1), YZ->X(2)
        self.axis_map = {'XY': 0, 'XZ': 1, 'YZ': 2}

        # **FIX**: initialize buffers immediately so _images always exists
        self._reset()

    def _reset(self):
        self._images: List[torch.Tensor] = []
        self._gts:    List[torch.Tensor] = []
        self._preds:  List[torch.Tensor] = []
        self._count:  int = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset()

    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        self._collect(batch, outputs, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        self._collect(batch, outputs, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._count > 0:
            self._plot_and_log('train', trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._count > 0:
            self._plot_and_log('val', trainer, pl_module)

    def _collect(self, batch, outputs, pl_module):
        if self._count >= self.num_samples:
            return
        x, y, preds = _gather_from_outputs(batch, outputs, pl_module)
        take = min(self.num_samples - self._count, x.size(0))
        self._images.append(x[:take])
        self._gts.append(y[:take])
        self._preds.append(preds[:take])
        self._count += take

    def _project(self, volume: torch.Tensor, modal: str) -> torch.Tensor:
        """
        Project a 3D tensor onto 2D by reducing along the chosen axis,
        using the right projection type for this modality.
        """
        axis = self.axis_map[self.projection_view]
        modal_cfg = self.cfg.get(modal, {})
        proj_type = modal_cfg.get('projection', self.default_modals[modal]['projection'])
        if proj_type == 'min':
            return volume.min(dim=axis)[0]
        else:
            return volume.max(dim=axis)[0]

    def _plot_and_log(self, tag: str, trainer, pl_module):
        imgs  = torch.cat(self._images,  0)  # N × C × Z × H × W
        gts   = torch.cat(self._gts,     0)
        preds = torch.cat(self._preds,   0)

        if imgs.dim() != 5:
            # fallback to 2D callback if implemented upstream
            super_hook = getattr(super(), f"on_{tag}_epoch_end", None)
            if callable(super_hook):
                super_hook(trainer, pl_module)
            return

        n = imgs.size(0)
        fig, axes = plt.subplots(n, 3, figsize=(12, n * 4), tight_layout=True)
        if n == 1:
            axes = axes[None, :]  # shape (1,3) even for single sample

        for i in range(n):
            data = {
                'input': imgs[i].squeeze(0),
                'gt':    gts[i].squeeze(0),
                'pred':  preds[i].squeeze(0),
            }
            for col, m in enumerate(['input', 'gt', 'pred']):
                arr = self._project(data[m], m)
                cmap = self.cfg.get(m, {}).get('cmap', self.default_modals[m]['cmap'])
                ax = axes[i, col]
                ax.imshow(arr.numpy(), cmap=cmap)
                ax.set_title(f"{tag}:{m}-{self.projection_view}")
                ax.axis('off')

        trainer.logger.experiment.add_figure(
            f"{tag}_3d_samples", fig, global_step=trainer.current_epoch
        )
        plt.close(fig)
