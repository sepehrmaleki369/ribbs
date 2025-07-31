# core/callbacks/grad_plot.py
import matplotlib
matplotlib.use("Agg")          # head‑less, no Qt

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

plt.ioff()                     # make sure we never open a window


class GradPlotCallback(pl.Callback):
    """
    ▸ Per‑epoch (every `every_n_epochs`) logs:
        • Figure  : grid of input – grad maps – prediction – GT   → tag  train_grad_samples
        • Scalars : mean |∂loss/∂input| per loss key             → tags grad_mean/primary, …
    """

    # ------------------------------------------------------------------ #
    def __init__(self,
                 input_key: str = "image_patch",
                 every_n_epochs: int = 5,
                 max_samples: int = 4,
                 cmap: str = "turbo"):
        super().__init__()
        self.k        = input_key
        self.freq     = every_n_epochs
        self.max_samp = max_samples
        self.cmap     = cmap
        self._buf: List[Tuple[Dict[str, torch.Tensor],
                              torch.Tensor,
                              torch.Tensor,
                              torch.Tensor]] = []

    # ------------------------------------------------------------------ #
    def on_train_epoch_start(self, *_):          # type: ignore[override]
        self._buf.clear()

    # ------------------------------------------------------------------ #
    def on_train_batch_end(                     # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        if (trainer.current_epoch + 1) % self.freq != 0:
            return
        if len(self._buf) >= self.max_samp:
            return

        # ---- take a tiny slice & require grad wrt *input* --------------
        x = batch[self.k][: self.max_samp].clone().to(pl_module.device)
        y = batch[pl_module.target_key][: self.max_samp].to(pl_module.device)
        x.requires_grad_(True)

        with torch.enable_grad():
            y_hat     = pl_module(x)
            loss_dict = pl_module.loss_fn(y_hat, y)

        has_sec = pl_module.loss_fn.secondary_loss is not None
        loss_keys = ["primary"] + (["secondary", "mixed"] if has_sec else [])

        grads = {k: torch.autograd.grad(loss_dict[k], x, retain_graph=True)[0].detach().cpu()
                 for k in loss_keys}

        # store CPU copies only
        self._buf.append((grads,
                          x.detach().cpu(),
                          y_hat.detach().cpu(),
                          y.detach().cpu()))

    # ------------------------------------------------------------------ #
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: E501
        if not self._buf:
            return

        has_sec   = "secondary" in self._buf[0][0]
        grad_keys = ["primary"] + (["secondary", "mixed"] if has_sec else [])

        # ------------------------ Figure --------------------------------
        cols = 3 + len(grad_keys)                # input + grads + pred + gt
        rows = len(self._buf)
        fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), tight_layout=True)
        if rows == 1:
            ax = ax[None, :]

        for r, (gdict, inp, pred, gt) in enumerate(self._buf):
            # input
            img = inp[0].permute(1, 2, 0) if inp.size(1) > 1 else inp[0, 0]
            ax[r, 0].imshow(img, cmap="gray")
            ax[r, 0].set_title("input"); ax[r, 0].axis("off")

            # gradients
            for c, k in enumerate(grad_keys, start=1):
                gmap = gdict[k][0].abs().sum(0)      # [H,W]
                ax[r, c].imshow(gmap, cmap=self.cmap)
                ax[r, c].set_title(f"‖∂{k}/∂x‖"); ax[r, c].axis("off")

            # prediction
            p_col = 1 + len(grad_keys)
            ax[r, p_col].imshow(pred[0, 0], cmap="coolwarm")
            ax[r, p_col].set_title("prediction"); ax[r, p_col].axis("off")

            # ground truth
            g_col = p_col + 1
            ax[r, g_col].imshow(gt[0, 0], cmap="coolwarm")
            ax[r, g_col].set_title("ground‑truth"); ax[r, g_col].axis("off")

        trainer.logger.experiment.add_figure(
            "train_grad_samples", fig, global_step=trainer.current_epoch
        )
        plt.close(fig)

        # ------------------------ Scalars -------------------------------
        for k in grad_keys:
            all_g = torch.cat([gdict[k] for gdict, *_ in self._buf], dim=0)  # [N,C,H,W]
            mean_val = all_g.abs().mean().item()
            trainer.logger.experiment.add_scalar(f"grad_mean/{k}",
                                                 mean_val,
                                                 trainer.current_epoch)
