# seglit_module.py
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
        train_metrics_every_n_epochs: int = 1,
        val_metrics_every_n_epochs: int = 1,
        train_metric_frequencies: Dict[str, int] = None,  # Per-metric train frequencies
        val_metric_frequencies: Dict[str, int] = None,    # Per-metric val frequencies
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model','loss_fn','metrics'])

        self.model = model
        self.loss_fn = loss_fn
        self.metrics = nn.ModuleDict(metrics)
        self.opt_cfg = optimizer_config
        self.validator = Validator(inference_config)
        self.input_key = input_key
        self.target_key = target_key

        # Global default frequencies
        self.train_freq = train_metrics_every_n_epochs
        self.val_freq = val_metrics_every_n_epochs
        
        # Per-metric frequencies (override defaults when specified)
        self.train_metric_frequencies = train_metric_frequencies or {}
        self.val_metric_frequencies = val_metric_frequencies or {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self):
        if isinstance(self.loss_fn, MixedLoss):
            self.loss_fn.update_epoch(self.current_epoch)
        # Reset metrics at the start of each epoch
        for m in self.metrics.values():
            if hasattr(m, 'reset'):
                m.reset()

    def training_step(self, batch, batch_idx):
        x = batch[self.input_key].float()
        y = batch[self.target_key].float()
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Always log training loss per epoch
        self.log("train_loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))

        # Compute metrics using per-metric frequencies
        y_int = y.long()
        for name, metric in self.metrics.items():
            # Get specific frequency for this metric or fall back to default
            freq = self.train_metric_frequencies.get(name, self.train_freq)
            
            # Only compute and log if it's time for this metric
            if self.current_epoch % freq == 0:
                val = metric(y_hat, y_int)
                self.log(f"train_{name}", val,
                         prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def on_validation_epoch_start(self):
        # Always reset metrics at the start of validation
        for m in self.metrics.values():
            if hasattr(m, 'reset'):
                m.reset()

    def validation_step(self, batch, batch_idx):
        x = batch[self.input_key].float()
        y = batch[self.target_key].float()
        y_hat = self.validator.run_chunked_inference(self.model, x)

        # Always log validation loss
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))

        # Compute metrics using per-metric frequencies
        y_int = y.long()
        for name, metric in self.metrics.items():
            # Get specific frequency for this metric or fall back to default
            freq = self.val_metric_frequencies.get(name, self.val_freq)
            
            # Only compute and log if it's time for this metric
            if self.current_epoch % freq == 0:
                val = metric(y_hat, y_int)
                # Only show on progress bar if it's time to compute it
                self.log(f"val_{name}", val,
                         prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        
        return {"predictions": y_hat, "val_loss": loss}

    def on_test_epoch_start(self):
        # Reset metrics at the start of test epoch
        for m in self.metrics.values():
            if hasattr(m, 'reset'):
                m.reset()

    def test_step(self, batch, batch_idx):
        x = batch[self.input_key].float()
        y = batch[self.target_key].float()
        y_hat = self.validator.run_chunked_inference(self.model, x)

        # Log test loss
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))

        # Compute and log test metrics - always compute all metrics during testing
        y_int = y.long()
        for name, metric in self.metrics.items():
            val = metric(y_hat, y_int)
            self.log(f"test_{name}", val,
                     prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        
        return {"predictions": y_hat, "test_loss": loss}

    def configure_optimizers(self):
        Opt = getattr(torch.optim, self.opt_cfg["name"])
        optimizer = Opt(self.model.parameters(), **self.opt_cfg.get("params", {}))
        sched_cfg = self.opt_cfg.get("scheduler", None)
        if not sched_cfg:
            return optimizer
        
        name, params = sched_cfg["name"], sched_cfg.get("params", {}).copy()
        
        if name == "ReduceLROnPlateau":
            monitor = params.pop("monitor", "val_loss")
            SchedulerClass = getattr(torch.optim.lr_scheduler, name)
            scheduler = SchedulerClass(optimizer, **params)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler,
                                     "monitor": monitor,
                                     "interval": "epoch",
                                     "frequency": 1,
                                     "strict": False}}
        
        if name == "LambdaLR":
            decay = params.get("lr_decay_factor")
            if decay is None:
                raise ValueError("LambdaLR requires 'lr_decay_factor'")
            lr_lambda = lambda epoch: 1.0 / (1.0 + epoch * decay)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        SchedulerClass = getattr(torch.optim.lr_scheduler, name)
        scheduler = SchedulerClass(optimizer, **params)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}