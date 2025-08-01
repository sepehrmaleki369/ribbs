# seglit_module.py
# ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Dict
import numpy as np
import logging

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
        train_metric_frequencies: Dict[str, int] = None,
        val_metric_frequencies: Dict[str, int] = None,
        divisible_by: int = 16,
        compute_val_loss=False
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'metrics'])

        self.model = model
        self.loss_fn = loss_fn
        self.metrics = nn.ModuleDict(metrics)
        self.opt_cfg = optimizer_config
        self.validator = Validator(inference_config)
        self.input_key = input_key
        self.target_key = target_key

        self.train_freq = train_metrics_every_n_epochs
        self.val_freq = val_metrics_every_n_epochs
        self.train_metric_frequencies = train_metric_frequencies or {}
        self.val_metric_frequencies = val_metric_frequencies or {}

        self.divisible_by = divisible_by

        self.compute_val_loss = compute_val_loss

        self.py_logger = logging.getLogger(__name__)
    
    def log(self, name, value, *args, **kwargs):
        """
        Override LightningModule.log so that every time you call self.log(...)
        it also does a Python-log to console/file.
        """
        # 1) call Lightning's logger
        super().log(name, value, *args, **kwargs)

        # 2) extract a scalar out of value (torch.Tensor or numeric)
        try:
            val = value.item()
        except Exception:
            val = value

        # 3) log via Python logging
        #    you can format this how you like; here we print epoch/step context
        ep = getattr(self, 'current_epoch', None)
        st = getattr(self.trainer, 'global_step', None)
        ctx = f"[ep={ep} step={st}]" if ep is not None and st is not None else ""
        self.py_logger.info(f"{ctx} {name} = {val:.4f}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        During training, run the raw model on incoming patches.
        During validation / testing / prediction, automatically
        pad & chunk the input so that full-image inference
        goes through your Validator (with proper 16-divisibility).
        """
        # if we're in train() mode, just do raw patch-based forward
        if self.training:
            return self.model(x)

        # otherwise (val/test/predict), run full-image chunked inference
        # (validator will pad to divisible-by-16 under the hood)
        with torch.no_grad():
            y_hat = self.validator.run_chunked_inference(self.model, x, self.divisible_by)
        return y_hat
    
    def on_train_epoch_start(self):
        dm = self.trainer.datamodule
        if dm and hasattr(dm, "train_dataset") and hasattr(dm.train_dataset, "set_epoch"):
            dm.train_dataset.set_epoch(self.current_epoch)
            
        self._train_preds = []
        self._train_gts   = []

        if isinstance(self.loss_fn, MixedLoss):
            self.loss_fn.update_epoch(self.current_epoch)
        for m in self.metrics.values():
            if hasattr(m, 'reset'):
                m.reset()

    def training_step(self, batch, batch_idx):
        im = batch["image_patch"]
        # print('img shape:', im.shape)
        x = batch[self.input_key].float()
        y = batch[self.target_key].float()
        y_hat = self(x)
        # loss = self.loss_fn(y_hat, y)
        loss_dict = self.loss_fn(y_hat, y)
        # combined
        self.log("train_loss", loss_dict["mixed"],
                prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        # components
        self.log("train_loss/primary",   loss_dict["primary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train_loss/secondary", loss_dict["secondary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))


        # self._train_preds.append(y_hat.detach().flatten())
        # self._train_gts.append(  y.detach().flatten())
        # print(f"[Train]  x.shape={x.shape}, y.shape={y.shape}")
        # print(f"[Train] y_hat.shape={y_hat.shape}")
        # print(f"[Train] loss computed on y_hat of shape {y_hat.shape} and y of shape {y.shape}")

        y_int = y
        for name, metric in self.metrics.items():
            freq = self.train_metric_frequencies.get(name, self.train_freq)
            if (self.current_epoch + 1) % freq != 0:
                continue
            result = metric(y_hat, y_int)
            if isinstance(result, dict):
                for subname, value in result.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    self.log(f"train_metrics/{name}_{subname}", value, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)
            else:
                if isinstance(result, torch.Tensor):
                    result = result.item()
                self.log(f"train_metrics/{name}", result, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)

        
        # ——— Log GT / Pred stats for TensorBoard ———
        # flatten tensors
        pred_flat = y_hat.flatten()
        gt_flat   = y.flatten()
        # GT statistics
        self.log("train_mmm/gt_min",   torch.min(gt_flat),   on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train_mmm/gt_max",   torch.max(gt_flat),   on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train_mmm/gt_mean",  torch.mean(gt_flat),  on_step=False, on_epoch=True, batch_size=x.size(0))
        # Pred statistics
        self.log("train_mmm/pred_min", torch.min(pred_flat), on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train_mmm/pred_max", torch.max(pred_flat), on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train_mmm/pred_mean",torch.mean(pred_flat),on_step=False, on_epoch=True, batch_size=x.size(0))


        return {"loss": loss_dict["mixed"], "predictions": y_hat, "gts": y}
    
    # def on_train_epoch_end(self):
    #     # concatenate everything
    #     preds = torch.cat(self._train_preds, dim=0)
    #     gts   = torch.cat(self._train_gts,   dim=0)
    #     # compute RMSE
    #     rmse = torch.sqrt(F.mse_loss(preds, gts))
    #     self.log("train_rmse", rmse, prog_bar=True, on_epoch=True)

    def on_validation_epoch_start(self):
        self._val_preds = []
        self._val_gts   = []

    def validation_step(self, batch, batch_idx):
        x = batch[self.input_key].float()
        y = batch[self.target_key].float()
        if x.dim() == 3:
            x, y = x.unsqueeze(0), y.unsqueeze(0)

        # chunked inference (with built-in padding)
        with torch.no_grad():
            y_hat = self.validator.run_chunked_inference(self.model, x, self.divisible_by)

        # loss = self.loss_fn(y_hat, y)
        # loss_dict = self.loss_fn(y_hat, y)
        loss_dict = self._safe_loss(y_hat, y)
        # combined
        self.log("val_loss", loss_dict["mixed"],
                prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        # components
        self.log("val_loss/primary",   loss_dict["primary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_loss/secondary", loss_dict["secondary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))


        # self._val_preds.append(y_hat.detach().flatten())
        # self._val_gts  .append(y.detach().flatten())
        # print(f"[Val]  x.shape={x.shape}, y.shape={y.shape}")
        # print(f"[Val] y_hat.shape={y_hat.shape}")
        # print(f"[Val] loss computed on y_hat of shape {y_hat.shape} and y of shape {y.shape}")

        y_int = y
        for name, metric in self.metrics.items():
            freq = self.val_metric_frequencies.get(name, self.val_freq)
            if (self.current_epoch+1) % freq == 0:
                # print('y_int.shape', y_hat.shape)
                # print('y_int.shape', y_hat.shape)
                # print(name, metric(y_hat, y_int))
                result = metric(y_hat, y_int)
                if isinstance(result, dict):
                    for subname, value in result.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        self.log(f"val_metrics/{name}_{subname}", value, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)
                else:
                    if isinstance(result, torch.Tensor):
                        result = result.item()
                    self.log(f"val_metrics/{name}", result, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)

        # ——— Log GT / Pred stats for TensorBoard ———
        # flatten tensors
        pred_flat = y_hat.flatten()
        gt_flat   = y.flatten()
        # GT statistics
        self.log("val_mmm/gt_min",   torch.min(gt_flat),   on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_mmm/gt_max",   torch.max(gt_flat),   on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_mmm/gt_mean",  torch.mean(gt_flat),  on_step=False, on_epoch=True, batch_size=x.size(0))
        # Pred statistics
        self.log("val_mmm/pred_min", torch.min(pred_flat), on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_mmm/pred_max", torch.max(pred_flat), on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_mmm/pred_mean",torch.mean(pred_flat),on_step=False, on_epoch=True, batch_size=x.size(0))

        return {"predictions": y_hat, "val_loss": loss_dict["mixed"], "gts": y}
    
    # def on_validation_epoch_end(self):
    #     preds = torch.cat(self._val_preds, dim=0)
    #     gts   = torch.cat(self._val_gts,   dim=0)
    #     rmse = torch.sqrt(F.mse_loss(preds, gts))
    #     self.log("val_rmse", rmse, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # same as validation but logs under test_
        x = batch[self.input_key].float()
        y = batch[self.target_key].float()
        if x.dim() == 3:
            x, y = x.unsqueeze(0), y.unsqueeze(0)

        with torch.no_grad():
            y_hat = self.validator.run_chunked_inference(self.model, x, self.divisible_by)

        # loss = self.loss_fn(y_hat, y)
        # loss_dict = self.loss_fn(y_hat, y)
        loss_dict = self._safe_loss(y_hat, y)
        # combined
        self.log("test_loss", loss_dict["mixed"],
                prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        # components
        self.log("test_loss/primary",   loss_dict["primary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("test_loss/secondary", loss_dict["secondary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))

        y_int = y
        for name, metric in self.metrics.items():
            result = metric(y_hat, y_int)
            if isinstance(result, dict):
                for subname, value in result.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    self.log(f"test_metrics/{name}_{subname}", value, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)
            else:
                if isinstance(result, torch.Tensor):
                    result = result.item()
                self.log(f"test_metrics/{name}", result, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)

        return {"predictions": y_hat, "test_loss": loss_dict["mixed"], "gts": y}

    def configure_optimizers(self):
        Opt = getattr(torch.optim, self.opt_cfg["name"])
        optimizer = Opt(self.model.parameters(), **self.opt_cfg.get("params", {}))
        sched_cfg = self.opt_cfg.get("scheduler", None)
        if not sched_cfg:
            return optimizer

        name, params = sched_cfg["name"], sched_cfg.get("params", {}).copy()
        if name == "ReduceLROnPlateau":
            monitor = params.pop("monitor", "val_loss")
            Scheduler = getattr(torch.optim.lr_scheduler, name)
            scheduler = Scheduler(optimizer, **params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,
                }
            }

        if name == "LambdaLR":
            decay = params.get("lr_decay_factor")
            if decay is None:
                raise ValueError("LambdaLR requires 'lr_decay_factor'")
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + epoch * decay)
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        Scheduler = getattr(torch.optim.lr_scheduler, name)
        scheduler = Scheduler(optimizer, **params)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Try a strict load first; if it fails because of unexpected or
        missing keys, fall back to strict=False so training can resume
        without losing optimizer / scheduler / epoch states.
        """
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as err:
            self.py_logger.warning(f"Strict load failed: {err}\nRetrying with strict=False.")
            return super().load_state_dict(state_dict, strict=False)
    
        # ──────────────────────────────────────────────────────────────
    def _safe_loss(self, y_hat: torch.Tensor, y: torch.Tensor):
        """
        Try to compute the configured loss; if it fails (e.g. because no
        grad info is available during val/test), return zeros so the run
        keeps going and log a warning.
        """
        if not self.compute_val_loss:
            z = torch.tensor(0.0, device=y_hat.device)
            return {"mixed": z, "primary": z, "secondary": z}
        try:
            return self.loss_fn(y_hat, y)
        except RuntimeError as err:
            self.py_logger.warning(f"Loss calculation skipped: {err}")
            z = torch.tensor(0.0, device=y_hat.device)
            return {"mixed": z, "primary": z, "secondary": z}
