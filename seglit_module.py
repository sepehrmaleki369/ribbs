# seglit_module.py
"""
SegLitModule: a PyTorch LightningModule for segmentation that
  • uses dynamic batch‐dict keys for x/y,
  • wraps all metrics in a ModuleDict so they get .to(device()) automatically,
  • supports MixedLoss + chunked inference + PL schedulers.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Dict

from core.mix_loss import MixedLoss
from core.validator import Validator


class SegLitModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: MixedLoss,
        metrics: Dict[str, nn.Module],
        optimizer_config: Dict[str, Any],
        inference_config: Dict[str, Any],
        input_key: str = "image_patch",
        target_key: str = "label_patch",
    ):
        super().__init__()
        # avoid dumping large modules into hparams.yaml
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'metrics'])

        self.model      = model
        self.loss_fn    = loss_fn
        # wrap metrics so Lightning moves them to the same device
        self.metrics    = nn.ModuleDict(metrics)
        self.opt_cfg    = optimizer_config
        self.validator  = Validator(inference_config)
        self.input_key  = input_key
        self.target_key = target_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self):
        if isinstance(self.loss_fn, MixedLoss):
            self.loss_fn.update_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        x = batch[self.input_key].float()
        y = batch[self.target_key].float()
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.input_key].float()
        y = batch[self.target_key].float()
        # full‐res chunked inference
        y_hat = self.validator.run_chunked_inference(self.model, x)

        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        y_int = y.long()
        for name, metric in self.metrics.items():
            val = metric(y_hat, y_int)
            if isinstance(val, torch.Tensor):
                val = val.detach()
            self.log(f"val_{name}", val, prog_bar=True, on_epoch=True)

        return {"predictions": y_hat, "val_loss": loss}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        Opt = getattr(torch.optim, self.opt_cfg["name"])
        optimizer = Opt(self.model.parameters(), **self.opt_cfg.get("params", {}))

        sched_cfg = self.opt_cfg.get("scheduler", None)
        if not sched_cfg:
            return optimizer

        name   = sched_cfg["name"]
        params = sched_cfg.get("params", {}).copy()

        if name == "ReduceLROnPlateau":
            # extract monitor, leave other params (patience, factor, mode, min_lr)
            monitor = params.pop("monitor", "val_loss")
            SchedulerClass = getattr(torch.optim.lr_scheduler, name)
            scheduler = SchedulerClass(optimizer, **params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False   # ← ignore missing val_loss early on
                }
            }

        if name == "LambdaLR":
            decay = params.get("lr_decay_factor")
            if decay is None:
                raise ValueError("LambdaLR requires 'lr_decay_factor' in params")
            lr_lambda = lambda epoch: 1.0 / (1.0 + epoch * decay)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        # generic scheduler
        SchedulerClass = getattr(torch.optim.lr_scheduler, name)
        scheduler = SchedulerClass(optimizer, **params)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
