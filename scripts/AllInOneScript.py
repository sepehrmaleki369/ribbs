# ------------------------------------
# train.py
# ------------------------------------
# train.py
"""
Training script for segmentation experiments.

This script orchestrates the training process by loading configurations,
setting up models, losses, metrics, and dataloaders, and running training.
"""

import os
import argparse
import logging
from typing import Any, Dict, List

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from core.loaders.callback_loader import load_callbacks
from core.loaders.model_loader import load_model
from core.loaders.loss_loader import load_loss
from core.loaders.metric_loader import load_metrics
from core.loaders.dataloader import SegmentationDataModule
from core.mix_loss import MixedLoss

from core.logger import setup_logger
from core.checkpoint import CheckpointManager
from core.utils import yaml_read, mkdir

from seglit_module import SegLitModule

# Silence noisy loggers
for lib in ('rasterio', 'matplotlib', 'PIL', 'tensorboard', 'urllib3'):
    logging.getLogger(lib).setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger('core').setLevel(logging.INFO)
logging.getLogger('__main__').setLevel(logging.INFO)
logging.getLogger('seglit_module').setLevel(logging.INFO)


def load_config(config_path: str) -> Dict[str, Any]:
    return yaml_read(config_path)


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", type=str, default="configs/main.yaml",
                        help="Path to main configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--test", action="store_true",
                        help="Run testing instead of training")
    args = parser.parse_args()

    # --- load configs ---
    main_cfg      = load_config(args.config)
    dataset_cfg   = load_config(os.path.join("configs", "dataset",   main_cfg["dataset_config"]))
    model_cfg     = load_config(os.path.join("configs", "model",     main_cfg["model_config"]))
    loss_cfg      = load_config(os.path.join("configs", "loss",      main_cfg["loss_config"]))
    metrics_cfg   = load_config(os.path.join("configs", "metrics",   main_cfg["metrics_config"]))
    inference_cfg = load_config(os.path.join("configs", "inference", main_cfg["inference_config"]))
    callbacks_cfg = load_config(os.path.join("configs", "callbacks", main_cfg["callbacks_config"]))

    # --- prepare output & logger ---
    output_dir = main_cfg.get("output_dir", "outputs")
    mkdir(output_dir)
    logger = setup_logger(os.path.join(output_dir, "training.log"))
    logger.info(f"Output dir: {output_dir}")

    # --- trainer params from YAML ---
    trainer_cfg                = main_cfg.get("trainer", {})

    train_metrics_every_n      = trainer_cfg["train_metrics_every_n_epochs"]
    val_metrics_every_n        = trainer_cfg["val_metrics_every_n_epochs"]

    # --- model, loss, metrics ---
    logger.info("Loading model...")
    model = load_model(model_cfg)

    logger.info("Loading losses...")
    prim = load_loss(loss_cfg["primary_loss"])
    sec  = load_loss(loss_cfg["secondary_loss"]) if loss_cfg.get("secondary_loss") else None
    mixed_loss = MixedLoss(
        prim, sec,
        alpha=loss_cfg.get("alpha", 0.5),
        start_epoch=loss_cfg.get("start_epoch", 0),
    )

    logger.info("Loading metrics...")
    metric_list = load_metrics(metrics_cfg["metrics"])

    # --- data module ---
    logger.info("Setting up data module...")
    dm = SegmentationDataModule(dataset_cfg)
    dm.setup()
    logger.info(f"Train set size:      {len(dm.train_dataset)} samples")
    logger.info(f"Validation set size: {len(dm.val_dataset)} samples")
    logger.info(f"Test set size:       {len(dm.test_dataset)} samples")

    # --- lightning module ---
    input_key  = main_cfg.get("target_x", "image_patch")
    target_key = main_cfg.get("target_y", "label_patch")
    lit = SegLitModule(
        model=model,
        loss_fn=mixed_loss,
        metrics=metric_list,
        optimizer_config=main_cfg["optimizer"],
        inference_config=inference_cfg,
        input_key=input_key,
        target_key=target_key,
        train_metrics_every_n_epochs=train_metrics_every_n,
        val_metrics_every_n_epochs=val_metrics_every_n,
        train_metric_frequencies=metrics_cfg.get("train_frequencies", {}),
        val_metric_frequencies=metrics_cfg.get("val_frequencies", {}),
        divisible_by=inference_cfg.get('chunk_divisible_by', 16)
    )

    # --- callbacks ---
    callbacks = load_callbacks(
        callbacks_cfg["callbacks"],
        output_dir=output_dir,
        resume=args.resume,
        skip_valid_until_epoch=trainer_cfg["skip_validation_until_epoch"],
        save_gt_pred_val_test_every_n_epochs=trainer_cfg.get("save_gt_pred_val_test_every_n_epochs", 5),
        save_gt_pred_val_test_after_epoch=trainer_cfg.get("save_gt_pred_val_test_after_epoch", 0),
        save_gt_pred_max_samples=trainer_cfg.get("save_gt_pred_max_samples", None),
        project_root=os.path.dirname(os.path.abspath(__file__)),
    )
    
    # --- trainer & logger setup ---
    tb_logger     = TensorBoardLogger(save_dir=output_dir, name="logs")
    trainer_kwargs = dict(trainer_cfg.get("extra_args", {}))

    # apply only those keys you defined in YAML
    trainer_kwargs.update({
        "max_epochs":                trainer_cfg["max_epochs"],
        "num_sanity_val_steps":      trainer_cfg["num_sanity_val_steps"],
        "check_val_every_n_epoch":   trainer_cfg["check_val_every_n_epoch"],
        "log_every_n_steps":         trainer_cfg["log_every_n_steps"],
    })

    trainer_kwargs["callbacks"]         = callbacks
    trainer_kwargs["logger"]            = tb_logger
    trainer_kwargs.setdefault("default_root_dir", output_dir)

    trainer = pl.Trainer(**trainer_kwargs)

    batch = next(iter(dm.train_dataloader()))
    logger.info("image_patch shape:", batch["image_patch"].shape)   # (B, C, H, W)
    logger.debug("UNet expects    :", lit.model.in_channels)


    # --- run ---
    if args.test:
        logger.info("Running test...")
        if args.resume:
            logger.info(f"Loading checkpoint for testing: {args.resume}")
            trainer.test(lit, datamodule=dm, ckpt_path=args.resume)
        else:
            trainer.test(lit, datamodule=dm)
    else:
        logger.info("Running training...")
        if args.resume:
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            trainer.fit(lit, datamodule=dm, ckpt_path=args.resume)
        else:
            logger.info("Starting training from scratch...")
            trainer.fit(lit, datamodule=dm)

        mgr = CheckpointManager(
            checkpoint_dir=os.path.join(output_dir, "checkpoints"),
            metrics=list(metric_list.keys()),
            default_mode="max"
        )
        best_metric, best_ckpt, best_val = mgr.get_best_checkpoint()
        logger.info(f"Best ckpt: {best_ckpt} ({best_metric}={best_val:.4f})")
        trainer.test(lit, datamodule=dm, ckpt_path=best_ckpt)


if __name__ == "__main__":
    main()


# ------------------------------------
# seglit_module.py
# ------------------------------------
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
        divisible_by: int = 16
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
        loss_dict = self.loss_fn(y_hat, y)
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
        loss_dict = self.loss_fn(y_hat, y)
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
                    self.log(f"train_metrics/{name}_{subname}", value, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)
            else:
                if isinstance(result, torch.Tensor):
                    result = result.item()
                self.log(f"train_metrics/{name}", result, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)

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



# ------------------------------------
# core/logger.py
# ------------------------------------
import os
import sys
import logging
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """
    Formatter class that adds colors to console log output.
    """
    
    # ANSI color codes
    GREEN = "\033[32;1m"
    YELLOW = "\033[33;1m"
    RED = "\033[31;1m"
    RESET = "\033[0m"
    
    def __init__(self, fmt: str = None):
        if fmt is None:
            fmt = "%(asctime)s [%(name)s] %(message)s"
        super().__init__(fmt)
    
    def format(self, record: logging.LogRecord) -> str:
        # Save original format
        original_fmt = self._style._fmt
        
        # Apply color based on log level
        if record.levelno == logging.INFO:
            self._style._fmt = f"{self.GREEN}{original_fmt}{self.RESET}"
        elif record.levelno == logging.WARNING:
            self._style._fmt = f"{self.YELLOW}{original_fmt}{self.RESET}"
        elif record.levelno >= logging.ERROR:
            self._style._fmt = f"{self.RED}{original_fmt}{self.RESET}"
        
        # Format with color
        result = super().format(record)
        
        # Restore original format
        self._style._fmt = original_fmt
        
        return result


def setup_logger(
    log_file: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    logger_name: Optional[str] = None
) -> logging.Logger:
    """
    Configure a logger with console and optional file output.
    
    Args:
        log_file: Path to log file (if None, file logging is disabled)
        console_level: Logging level for console output
        file_level: Logging level for file output
        logger_name: Name for the logger (if None, uses root logger)
        
    Returns:
        Configured logger
    """
    # Get logger
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    
    logger.setLevel(logging.DEBUG)  # Allow all levels to pass through to handlers
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = ColoredFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_file is not None:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# ------------------------------------
# core/validator.py
# ------------------------------------
"""
Validator module for handling chunked inference in validation/test phases.
Now supports **both 2‑D (N, C, H, W)** and **3‑D (N, C, D, H, W)** inputs.
Implemented with robust size handling
and automatic dimensionality detection.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import process_in_chuncks  # unchanged – must support N‑D windows


class Validator:
    """Chunked, overlap‑tiled inference for 2‑D **or** 3‑D data.

    • Works with arbitrary batch size and channel count.
    • Pads the sample so every spatial dimension is divisible by a given
      *divisor* (default: 16) before tiling, then removes the pad.
    • Uses `patch_size` and `patch_margin` to create overlapping tiles.
      Only the *centre* region of each model prediction is kept and
      stitched together.

    Parameters
    ----------
    config : dict
        Dictionary with at least the keys:
            ``patch_size``   – tuple/list[int] (len == 2 or 3)
            ``patch_margin`` – tuple/list[int] same length as
                                 ``patch_size``
        Any other keys are ignored by this class.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patch_size: Tuple[int, ...] = tuple(config.get("patch_size", (512, 512)))
        self.patch_margin: Tuple[int, ...] = tuple(config.get("patch_margin", (32, 32)))
        self.logger = logging.getLogger(__name__)

        if len(self.patch_size) != len(self.patch_margin):
            raise ValueError(
                "patch_size %s and patch_margin %s must have the same number of dimensions"
                % (self.patch_size, self.patch_margin)
            )
        if len(self.patch_size) not in (2, 3):
            raise ValueError("Only 2‑D or 3‑D data are supported (got %d‑D)" % len(self.patch_size))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _calc_div16_pad(size: int, divisor: int = 16) -> int:
        """Return how many voxels/pixels must be *added to the right* so that
        *size* becomes divisible by *divisor* (default 16)."""
        return (divisor - size % divisor) % divisor

    def _pad_to_valid_size(
        self, image: torch.Tensor, divisor: int = 16
    ) -> Tuple[torch.Tensor, List[int]]:
        """Pad *image* so *all* spatial dims are divisible by ``divisor``.

        Only **right/bottom/back** padding is applied (no shift of origin).

        Parameters
        ----------
        image : torch.Tensor
            ``(N, C, H, W)`` or ``(N, C, D, H, W)`` tensor.
        divisor : int, optional
            The divisor (default 16).

        Returns
        -------
        image_padded : torch.Tensor
        pads         : list[int]
            Per‑dimension pad added (*same order as image spatial dims*).
        """

        spatial = image.shape[2:]
        pad_each: List[int] = [self._calc_div16_pad(s, divisor) for s in spatial]

        # Build pad tuple for F.pad – must be *reversed* order, one (left,right)
        # pair per dim starting with the last spatial dim.
        pad_tuple: List[int] = []
        for p in reversed(pad_each):
            pad_tuple.extend([0, p])  # (left = 0, right = p)

        if any(pad_each):
            try:
                image = F.pad(image, pad_tuple, mode="reflect")
            except RuntimeError:
                # "reflect" not implemented for 5‑D – fall back gracefully.
                image = F.pad(image, pad_tuple, mode="replicate")
        return image, pad_each

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------
    def run_chunked_inference(
        self,
        model: nn.Module,
        image: torch.Tensor,
        divisible_by: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Full‑image/volume inference with overlapping tiles.

        Workflow (N‑D):
        1) **Margin pad** by ``patch_margin`` (reflect/replicate).
        2) **Div‑16 pad** so every spatial dim is divisible by 16.
        3) **Sliding‑window** inference:
            • window      = ``patch_size``
            • window step = ``patch_size − 2*patch_margin``
            • model is applied on each window; only the *centre* region
              is placed into the output canvas.
        4) **Remove** the div‑16 pad.
        5) **Remove** the initial margin pad.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(N, out_channels, *original_spatial*)``.
        """

        if device is None:
            device = next(model.parameters()).device

        model.eval()
        image = image.to(device)

        ndim = len(self.patch_size)  # 2 or 3
        if image.dim() != ndim + 2:
            raise ValueError(
                f"Input tensor dim {image.dim()} does not match patch_size ndim {ndim}"
            )

        # ----------------------------------------------------------
        # (A) First pad by the desired margins so borders get context
        # ----------------------------------------------------------
        if any(self.patch_margin):
            # Build pad tuple [ ... (left,right) per dim … ]
            pad_tuple: List[int] = []
            for m in reversed(self.patch_margin):
                pad_tuple.extend([m, m])
            try:
                image = F.pad(image, pad_tuple, mode="reflect")
            except RuntimeError:
                image = F.pad(image, pad_tuple, mode="replicate")

        # ----------------------------------------------------------
        # (B) Second pad until all dims divisible by 
        # ----------------------------------------------------------
        padded_image, pad_div16 = self._pad_to_valid_size(image, divisible_by)
        N, C, *spatial_pad = padded_image.shape

        # ----------------------------------------------------------
        # (C) Dummy forward to figure out #out channels
        # ----------------------------------------------------------
        with torch.no_grad():
            test_sizes = [
                min(s, p + 2 * m)
                for s, p, m in zip(spatial_pad, self.patch_size, self.patch_margin)
            ]
            slices: List[slice] = [slice(None), slice(None)] + [slice(0, t) for t in test_sizes]
            test_patch = padded_image[tuple(slices)]
            test_patch, _ = self._pad_to_valid_size(test_patch, divisible_by)
            out_channels = model(test_patch).shape[1]

        # Allocate output canvas
        output_shape = (N, out_channels, *spatial_pad)
        output = torch.zeros(output_shape, device=device, dtype=padded_image.dtype)

        # ----------------------------------------------------------
        # (D) Sliding‑window inference
        # ----------------------------------------------------------
        def _process(chunk: torch.Tensor) -> torch.Tensor:  # noqa: D401
            with torch.no_grad():
                return model(chunk)

        with torch.no_grad():
            output = process_in_chuncks(
                padded_image,
                output,
                _process,
                list(self.patch_size),
                list(self.patch_margin),
            )

        # ----------------------------------------------------------
        # (E) Remove the div‑16 pad (right/bottom/back only)
        # ----------------------------------------------------------
        if any(pad_div16):
            slices: List[slice] = [slice(None), slice(None)]
            for p in pad_div16:
                slices.append(slice(None, -p if p else None))
            output = output[tuple(slices)]

        # ----------------------------------------------------------
        # (F) Remove the initial margin pad (all sides)
        # ----------------------------------------------------------
        if any(self.patch_margin):
            slices = [slice(None), slice(None)]
            for i, m in enumerate(self.patch_margin):
                end = output.shape[2 + i] - m
                slices.append(slice(m, end))
            output = output[tuple(slices)]

        return output


# ------------------------------------
# core/checkpoint.py
# ------------------------------------
"""
Checkpoint utilities for managing best metric checkpoints.

This module provides helper functionality for managing checkpoints 
based on the best values of multiple metrics.
"""

import os
import logging
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple


class CheckpointMode(Enum):
    """
    Enumeration of checkpoint modes.
    
    Attributes:
        MIN: Save checkpoint when metric reaches a new minimum
        MAX: Save checkpoint when metric reaches a new maximum
    """
    MIN = "min"
    MAX = "max"


class CheckpointManager:
    """
    Manager for tracking best metric values and checkpoint paths.
    
    This class tracks the best values for multiple metrics and manages
    checkpoint paths for each metric.
    """
    
    def __init__(
        self, 
        checkpoint_dir: str,
        metrics: Union[List[str], Dict[str, str]],
        default_mode: str = "min"
    ):
        """
        Initialize the CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints to
            metrics: List of metric names or dict mapping metrics to modes ("min"/"max")
            default_mode: Default mode ("min" or "max") for metrics
        """
        self.checkpoint_dir = checkpoint_dir
        self.logger = logging.getLogger(__name__)
        
        # Create directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup metric modes
        self.modes: Dict[str, CheckpointMode] = {}
        if isinstance(metrics, list):
            for metric in metrics:
                self.modes[metric] = CheckpointMode(default_mode)
        else:
            for metric, mode in metrics.items():
                self.modes[metric] = CheckpointMode(mode)
        
        # Initialize best values
        self.best_values: Dict[str, float] = {}
        for metric, mode in self.modes.items():
            self.best_values[metric] = float('inf') if mode == CheckpointMode.MIN else float('-inf')
    
    def is_better(self, metric: str, value: float) -> bool:
        """
        Check if a new metric value is better than the current best.
        
        Args:
            metric: Name of the metric
            value: New metric value
            
        Returns:
            True if the new value is better, False otherwise
        """
        if metric not in self.modes:
            raise ValueError(f"Unknown metric: {metric}")
            
        mode = self.modes[metric]
        current_best = self.best_values[metric]
        
        if mode == CheckpointMode.MIN:
            return value < current_best
        else:
            return value > current_best
    
    def update_best(self, metric: str, value: float) -> None:
        """
        Update the best value for a metric.
        
        Args:
            metric: Name of the metric
            value: New best value
        """
        if metric not in self.modes:
            raise ValueError(f"Unknown metric: {metric}")
            
        self.best_values[metric] = value
    
    def get_checkpoint_path(self, metric: str) -> str:
        """
        Get the checkpoint path for a metric.
        
        Args:
            metric: Name of the metric
            
        Returns:
            Path to the checkpoint file
        """
        if metric not in self.modes:
            raise ValueError(f"Unknown metric: {metric}")
            
        return os.path.join(self.checkpoint_dir, f"best_{metric}.ckpt")
    
    def get_last_checkpoint_path(self) -> str:
        """
        Get the path for the last checkpoint.
        
        Returns:
            Path to the last checkpoint file
        """
        return os.path.join(self.checkpoint_dir, "last.ckpt")
    
    def get_best_checkpoint(self) -> Tuple[str, str, float]:
        """
        Get the best checkpoint across all metrics.
        
        This method determines which metric has the 'most improved' value
        (relative to typical values for that metric).
        
        Returns:
            Tuple of (metric_name, checkpoint_path, best_value)
        """
        # Find the metric that has improved the most (as a percentage)
        best_metric = None
        best_improvement = 0.0
        
        for metric, mode in self.modes.items():
            initial_value = float('inf') if mode == CheckpointMode.MIN else float('-inf')
            current_value = self.best_values[metric]
            
            # Skip metrics that haven't been updated
            if current_value == initial_value:
                continue
                
            # Calculate improvement (always as a positive percentage)
            if mode == CheckpointMode.MIN:
                # For MIN mode, improvement is decrease from infinity (use a large number instead)
                improvement = 1.0 - (current_value / 1000.0)
            else:
                # For MAX mode, improvement is increase from negative infinity
                improvement = (current_value + 1000.0) / 1000.0
                
            if improvement > best_improvement:
                best_improvement = improvement
                best_metric = metric
        
        if best_metric is None:
            # If no metrics have been updated, return the last checkpoint
            return ("last", self.get_last_checkpoint_path(), 0.0)
        else:
            return (
                best_metric, 
                self.get_checkpoint_path(best_metric),
                self.best_values[best_metric]
            )

# ------------------------------------
# core/utils.py
# ------------------------------------
import os
import numpy as np
import json
import yaml
import pickle
import torch
import logging
import multiprocessing
import itertools
import math


torch_version_major = int(torch.__version__.split('.')[0])
torch_version_minor = int(torch.__version__.split('.')[1])

class Dummysink(object):
    def write(self, data):
        pass # ignore the data
    def __enter__(self): return self
    def __exit__(*x): pass

torch_no_grad = Dummysink() if torch_version_major==0 and torch_version_minor<4 else torch.no_grad()

def to_torch(ndarray, volatile=False):
    if torch_version_major>=1:
        return torch.from_numpy(ndarray)
    else:
        from torch.autograd import Variable
        return Variable(torch.from_numpy(ndarray), volatile=volatile)

def from_torch(tensor, num=False):
    return tensor.data.cpu().numpy()
    '''
    if num and torch_version_major==0 and torch_version_minor<4:
        return tensor.data.cpu().numpy()[0]
    else:
        return tensor.data.cpu().numpy()
    '''
def sigmoid(x):
    e = np.exp(x)
    return e/(e + 1)

def softmax(x):
    e_x = np.exp(x - np.max(x,0))
    return e_x / e_x.sum(axis=0)

def rgb2gray(image):
    dtype = image.dtype
    gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    return gray.astype(dtype)

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:
            data = json.load(f)
        return data
    except:
        raise ValueError("Unable to read JSON {}".format(filename))

def json_write(data, filename):
    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(data, f, indent=2)
    except:
        raise ValueError("Unable to write JSON {}".format(filename))

def yaml_read(filename):
    try:
        with open(filename, 'r') as f:
            try:
                data = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError as e:
                data = yaml.load(f)
        return data
    except:
        raise ValueError("Unable to read YAML {}".format(filename))

def yaml_write(data, filename):
    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), 'w') as f:
            yaml.dump(data, f, default_flow_style=False, width=1000)
    except:
        raise ValueError("Unable to write YAML {}".format(filename))

def pickle_read(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def isnan(x):
    return x!=x


def downsample_image(img, size, msigma=1.0, interpolation='area'):
    import cv2

    scale_h = size[0]/img.shape[0]
    scale_w = size[1]/img.shape[1]

    if interpolation == 'cubic':
        interpolation=cv2.INTER_CUBIC
    elif interpolation == 'area':
        interpolation=cv2.INTER_AREA
    elif interpolation == 'linear':
        interpolation=cv2.INTER_LINEAR

    if msigma is not None:
        img = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=1.0/scale_w*msigma, sigmaY=1.0/scale_h*msigma)
    img = cv2.resize(img, dsize=(size[1], size[0]), interpolation=interpolation)
    return img

def upsample_image(img, size, interpolation='cubic'):
    import cv2

    if interpolation == 'cubic':
        interpolation=cv2.INTER_CUBIC
    elif interpolation == 'area':
        interpolation=cv2.INTER_AREA
    elif interpolation == 'linear':
        interpolation=cv2.INTER_LINEAR

    img = cv2.resize(img, dsize=(size[1], size[0]), interpolation=interpolation)
    return img

def inbounds(coords, image_shape):
    """
    Returns a mask of the coordinates that are inside an image with the given
    shape.

    coords: [x,y]
    image_shape: [height, width]

    x->width
    y->height
    """

    mask = np.ones(len(coords))
    image_shape = np.array(image_shape)[[1,0]]

    for coords_i, sh_i in zip(coords.T, image_shape):
        aux = np.logical_and(coords_i >= 0, coords_i < sh_i)
        mask = np.logical_and(mask, aux)

    return mask

class Parallel(object):

    def __init__(self, threads=8):
        self.threads = threads
        self.p = multiprocessing.Pool(threads)

    def __call__(self, f, iterable, *arg):

        if len(arg):
            res = self.p.starmap(f, itertools.product(iterable, *[[x] for x in arg]))
        else:
            res = self.p.map(f, iterable)

        self.p.close()
        self.p.join()
        return res

    @staticmethod
    def split_iterable(iterable, n):

        if isinstance(iterable, (list,tuple)):
            s = len(iterable)//n
            return [iterable[i:i + s] for i in range(0, len(iterable), s)]
        elif isinstance(iterable, np.ndarray):
            return np.array_split(iterable, n)

    @staticmethod
    def join_iterable(iterable):

        if isinstance(iterable, (list,tuple)):
            return list(itertools.chain.from_iterable(iterable))
        elif isinstance(iterable, np.ndarray):
            return np.concatenate(iterable)




class BinCounter(object):
    """Counter of elements in NumPy arrays."""

    def __init__(self, minlength=0, x=None, weights=None):

        self.minlength = minlength
        self.counts = np.zeros(minlength, dtype=np.int_)

        if x is not None and len(x) > 0:
            self.update(x, weights)

    def update(self, x, weights=None):
        if weights is not None:
            weights = weights.flatten()

        current_counts = np.bincount(np.ravel(x), weights=weights, minlength=self.minlength)
        current_counts[:len(self.counts)] += self.counts

        self.counts = current_counts

    def frequencies(self):
        return self.counts / np.float_(np.sum(self.counts))

def invfreq_lossweights(labels, num_classes):

    bc = BinCounter(num_classes + 1)
    for labels_i in labels:
        bc.update(labels_i)
    class_weights = 1.0 / (num_classes * bc.frequencies)[:num_classes]
    class_weights = np.hstack([class_weights, 0])
    class_weights[np.isinf(class_weights)] = np.max(class_weights)
    return np.float32(class_weights)




def noCrops(inSize, cropSize, marginSize, startDim=0):
  # inSize 
  # cropSize - can be shorter than inSize, if not all dims are cropped
  #            in this case startDim > 0
  # marginSize - same length as cropSize; stores size of a single margin;
  #              the resulting overlap between crops is 2*marginSize
  # startDim - all dimensions starting from this one are cropped;
  #            for example, if dim 0 indexes batches and dim 1 indexes channels
  #            startDim would typically equal 2
  nCrops=1
  for dim in range(startDim, len(inSize)):
    relDim=dim-startDim
    nCropsPerDim=(inSize[dim]-2*marginSize[relDim])/ \
                 (cropSize[relDim]-2*marginSize[relDim])
    if nCropsPerDim<=0:
      nCropsPerDim=1
    nCrops*=math.ceil(nCropsPerDim)
  return nCrops

def noCropsPerDim(inSize,cropSize,marginSize,startDim=0):
  # nCropsPerDim - number of crops per dimension, starting from startDim
  # cumNCropsPerDim - number of crops for one index step along a dimension
  #                   starting from startDim-1; i.e. it has one more element
  #                   than nCropsPerDim, and is misaligned by a difference
  #                   in index of 1
  nCropsPerDim=[]
  cumNCropsPerDim=[1]
  for dim in reversed(range(startDim,len(inSize))):
    relDim=dim-startDim
    nCrops=(inSize[dim]-2*marginSize[relDim])/ \
           (cropSize[relDim]-2*marginSize[relDim])
    if nCrops<=0:
      nCrops=1 
    nCrops=math.ceil(nCrops)
    nCropsPerDim.append(nCrops)
    cumNCropsPerDim.append(nCrops*cumNCropsPerDim[len(inSize)-dim-1])
  nCropsPerDim.reverse()
  cumNCropsPerDim.reverse()
  return nCropsPerDim, cumNCropsPerDim

def cropInds(cropInd, cumNCropsPerDim):
    # given a single index into the crops of a given data chunk
    # this function returns indexes of the crop along all its dimensions
    assert cropInd<cumNCropsPerDim[0]
    rem=cropInd
    cropInds=[]
    for dim in range(1,len(cumNCropsPerDim)):
        cropInds.append(rem//cumNCropsPerDim[dim])
        rem=rem%cumNCropsPerDim[dim]
    return cropInds

def coord(cropInd,cropSize,marg,inSize):
    # this function maps an index of a volume crop
    # to the starting and end coordinate of a crop
    # it is meant to be used for a single dimension
    assert inSize>=cropSize
    startind=cropInd*(cropSize-2*marg) #starting coord of the crop in the big vol
    startValidInd=marg                 #starting coord of valid stuff in crop
    endValidInd=cropSize-marg
    if startind >= inSize-cropSize:
        startValidInd=cropSize+startind-inSize+marg
        startind=inSize-cropSize
        endValidInd=cropSize
    if cropInd==0:
        startValidInd=0
    return slice(int(startind),int(startind+cropSize)), \
         slice(int(startValidInd),int(endValidInd))
         
def coords(cropInds,cropSizes,margs,inSizes,startDim):
    # this function maps a table of crop indeces
    # to the starting and end coordinates of the crop
    cropCoords=[]
    validCoords=[]
    for i in range(startDim):
        cropCoords. append(slice(0,inSizes[i]))
        validCoords.append(slice(0,inSizes[i]))
    for i in range(startDim,len(inSizes)):
        reli=i-startDim
        c,d=coord(cropInds[reli],cropSizes[reli],margs[reli],inSizes[i])
        cropCoords.append(c)
        validCoords.append(d)
    return cropCoords, validCoords

def cropCoords(cropInd, cropSize, marg, inSize, startDim):
    # a single index in, a table of crop coordinates out
    nCropsPerDim, cumNCropsPerDim = noCropsPerDim(inSize, cropSize, marg, startDim)
    cropIdx = cropInds(cropInd, cumNCropsPerDim)
    cropCoords, validCoords = coords(cropIdx, cropSize, marg, inSize, startDim)
    return cropCoords, validCoords

def split_with_margin(size, crop_size, margin):
    
    # some checking
    assert len(crop_size)==len(margin), "crop_size and margin must have same length!"
    for crop,marg in zip(crop_size, margin):
        assert crop>(marg*2), "margin is bigger than crop_size!"

    # get number of crops
    n_crops = noCrops(size, crop_size, margin, 0) 

    # pack list of slices into a convenient format
    source_coords, valid_coords, destin_coords = [],[],[]
    for i in range(n_crops):
        source_slices, valid_slices = cropCoords(i, crop_size, margin, size, 0)
        destin_slices = [slice(s.start+v.start, s.start+v.stop) 
                             for s, v in zip(source_slices, valid_slices)]

        source_coords.append(tuple(source_slices))
        valid_coords.append(tuple(valid_slices))
        destin_coords.append(tuple(destin_slices))

    return source_coords, valid_coords, destin_coords

def process_in_chuncks(image, output, process, patch_size, patch_margin):
    """
    N,C,D1,D2,...,Dn
    """
    # print('process_in_chuncks',image.shape, output.shape, patch_size, patch_margin)
    assert len(image.shape)==len(output.shape), f'{len(image.shape)}=?{len(output.shape)}'
    assert (len(image.shape)-2)==len(patch_size), f'{(len(image.shape)-2)}?={len(patch_size)} - image.shape:{image.shape}, patch_size:{patch_size}'
    assert len(patch_margin)==len(patch_size)

    chunck_coords = split_with_margin(image.shape[2:], patch_size, patch_margin)

    semicol = (slice(None,None),) # this mimicks :
    
    for source_c, valid_c, destin_c in zip(*chunck_coords):

        crop = image[semicol+semicol+source_c]
        proc_crop = process(crop)
        # print(image.shape, crop.shape, proc_crop.shape)
        # print('proc_crop', proc_crop.detach().cpu().numpy())
        # print('crop', proc_crop.detach().cpu().numpy())
        ########### Changed by Fayzad:
        if len(proc_crop.shape) == 3:  
            proc_crop = proc_crop.unsqueeze(1)  # Convert [1, H, W] -> [1, 1, H, W]
        ###############################
        
        output[semicol+semicol+destin_c] = proc_crop[semicol+semicol+valid_c]
    # print('process_in_chuncks done', output.shape, output.dtype, output.device)
    return output


# ------------------------------------
# core/mix_loss.py
# ------------------------------------
import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Union


class MixedLoss(nn.Module):
    """
    Primary loss always active; secondary loss kicks in
    *only* during training (module.train()) and after
    `start_epoch`.  Nothing extra is required in val/test.
    """

    def __init__(
        self,
        primary_loss: Union[nn.Module, Callable],
        secondary_loss: Optional[Union[nn.Module, Callable]] = None,
        alpha: float = 0.5,
        start_epoch: int = 0,
    ):
        super().__init__()
        self.primary_loss = primary_loss
        self.secondary_loss = secondary_loss
        self.alpha = alpha
        self.start_epoch = start_epoch
        self.current_epoch = 0  # call update_epoch() each epoch

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------
    def update_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, y_pred, y_true) -> Dict[str, torch.Tensor]:
        p = self.primary_loss(y_pred, y_true)

        use_secondary = (
            self.secondary_loss is not None
            and self.training                       # train() vs eval()
            and self.current_epoch >= self.start_epoch
        )
        if use_secondary:
            s = self.secondary_loss(y_pred, y_true)
        else:
            s = torch.tensor(0.0, device=p.device, dtype=p.dtype)

        m = p + self.alpha * s
        return {"primary": p, "secondary": s, "mixed": m}

# ------------------------------------
# core/__init__.py
# ------------------------------------
"""
Module __init__ file for the core package.

This file imports and exposes the main components from the core package.
"""

from core.loaders.model_loader import load_model
from core.loaders.loss_loader import load_loss
from core.loaders.callback_loader import load_callbacks
from core.loaders.metric_loader import load_metric, load_metrics
from core.loaders.dataloader import SegmentationDataModule
from core.mix_loss import MixedLoss
from core.validator import Validator
from core.logger import setup_logger, ColoredFormatter
from core.checkpoint import CheckpointManager, CheckpointMode
from core.utils import *

from core.general_dataset import GeneralizedDataset, custom_collate_fn, worker_init_fn

__all__ = [
    'load_model',
    'load_loss',
    'load_metric',
    'load_metrics',
    'load_callbacks',
    'SegmentationDataModule',
    'MixedLoss',
    'Validator',
    'setup_logger',
    'ColoredFormatter',
    'CheckpointManager',
    'CheckpointMode',
    'GeneralizedDataset',
    'custom_collate_fn',
    'worker_init_fn',
]


# ------------------------------------
# core/loaders/model_loader.py
# ------------------------------------
"""
Model loader for loading models dynamically based on configuration.

This module provides functionality to load models dynamically based on their path and class name.
"""

import os
import sys
import importlib
from typing import Any, Dict, Optional, Type, Union

import torch
import torch.nn as nn


def load_model(config: Dict[str, Any]) -> nn.Module:
    """
    Load a model based on configuration.

    Args:
        config: Dictionary containing model configuration with keys:
            - path: Path to the module containing the model class
            - class: Name of the model class to instantiate
            - params: Parameters to pass to the model constructor

    Returns:
        Instantiated model

    Raises:
        ImportError: If the model module cannot be imported
        AttributeError: If the model class cannot be found in the module
        Exception: If the model cannot be instantiated
    """
    model_path = config.get("path")
    model_class_name = config.get("class")
    model_params = config.get("params", {})

    if not model_path or not model_class_name:
        raise ValueError("Model configuration must contain 'path' and 'class'")

    try:
        # If model_path is a direct path to a Python file
        if model_path.endswith(".py"):
            module_name = os.path.basename(model_path)[:-3]  # Remove .py extension
            spec = importlib.util.spec_from_file_location(module_name, model_path)
            if not spec or not spec.loader:
                raise ImportError(f"Could not load module spec from {model_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # If model_path is a module path (e.g., 'models.unet')
            module = importlib.import_module(model_path)

        model_class = getattr(module, model_class_name)
        model = model_class(**model_params)
        return model
    except ImportError as e:
        raise ImportError(f"Error importing model module from {model_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Could not find class {model_class_name} in module {model_path}: {e}")
    except Exception as e:
        raise Exception(f"Error instantiating model {model_class_name} from {model_path}: {e}")

# ------------------------------------
# core/loaders/metric_loader.py
# ------------------------------------
"""
Metric loader for loading evaluation metrics dynamically based on configuration.

This module provides functionality to load metric functions dynamically based on their path and class name.
"""

import os
import sys
import importlib
from typing import Any, Dict, Optional, Type, Union, Callable, List

import torch
import torch.nn as nn


def load_metric(config: Dict[str, Any]) -> Union[nn.Module, Callable]:
    """
    Load a metric based on configuration.

    Args:
        config: Dictionary containing metric configuration with keys:
            - path: Path to the module containing the metric class
            - class: Name of the metric class to instantiate
            - params (optional): Parameters to pass to the metric constructor
            - alias: A friendly name for the metric

    Returns:
        Instantiated metric function/module

    Raises:
        ImportError: If the metric module cannot be imported
        AttributeError: If the metric class cannot be found in the module
        Exception: If the metric cannot be instantiated
    """
    metric_path = config.get("path")
    metric_class_name = config.get("class")
    metric_params = config.get("params", {})

    if not metric_path or not metric_class_name:
        raise ValueError("Metric configuration must contain 'path' and 'class'")

    try:
        # If metric_path is a direct path to a Python file
        if metric_path.endswith(".py"):
            module_name = os.path.basename(metric_path)[:-3]  # Remove .py extension
            spec = importlib.util.spec_from_file_location(module_name, metric_path)
            if not spec or not spec.loader:
                raise ImportError(f"Could not load module spec from {metric_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # If metric_path is a module path (e.g., 'metrics.dice')
            module = importlib.import_module(metric_path)

        metric_class = getattr(module, metric_class_name)
        metric = metric_class(**metric_params)
        return metric
    except ImportError as e:
        raise ImportError(f"Error importing metric module from {metric_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Could not find class {metric_class_name} in module {metric_path}: {e}")
    except Exception as e:
        raise Exception(f"Error instantiating metric {metric_class_name} from {metric_path}: {e}")


def load_metrics(metrics_config: List[Dict[str, Any]]) -> Dict[str, Union[nn.Module, Callable]]:
    """
    Load multiple metrics based on configuration.

    Args:
        metrics_config: List of metric configuration dictionaries

    Returns:
        Dictionary mapping metric aliases to instantiated metrics
    """
    metrics = {}
    for metric_config in metrics_config:
        alias = metric_config.get("alias")
        if not alias:
            raise ValueError("Each metric configuration must contain an 'alias'")
        metrics[alias] = load_metric(metric_config)
    return metrics

# ------------------------------------
# core/loaders/dataloader.py
# ------------------------------------
import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from core.general_dataset.base import GeneralizedDataset
from core.general_dataset.collate import custom_collate_fn, worker_init_fn


class SegmentationDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for segmentation tasks wrapping GeneralizedDataset.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config.copy()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Dataloader parameters
        self.batch_size = {
            'train': config.get('train_batch_size', 8),
            'val':   config.get('val_batch_size', 1),
            'test':  config.get('test_batch_size', 1)
        }
        self.num_workers = config.get('num_workers', 4)
        self.pin_memory = config.get('pin_memory', True)

    def setup(self, stage: Optional[str] = None):
        """
        Instantiate datasets for training, validation, and testing.
        """
        def make_cfg(split: str) -> Dict[str, Any]:
            cfg = self.config.copy()
            cfg['split'] = split
            return cfg

        if stage in ('fit', None):
            self.train_dataset = GeneralizedDataset(make_cfg('train'))
            self.val_dataset = GeneralizedDataset(make_cfg('valid'))

        if stage in ('test', None):
            self.test_dataset = GeneralizedDataset(make_cfg('test'))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size['train'],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size['val'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size['test'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )


# ------------------------------------
# core/loaders/loss_loader.py
# ------------------------------------
"""
Loss loader for loading loss functions dynamically based on configuration.

This module provides functionality to load loss functions dynamically based on their path and class name.
"""

import os
import sys
import importlib
from typing import Any, Dict, Optional, Type, Union, Callable

import torch
import torch.nn as nn


def load_loss(config: Dict[str, Any]) -> Union[nn.Module, Callable]:
    """
    Load a loss function based on configuration.

    Args:
        config: Dictionary containing loss configuration with keys:
            - path (optional): Path to the module containing the loss class
            - class: Name of the loss class or built-in loss to instantiate
            - params (optional): Parameters to pass to the loss constructor

    Returns:
        Instantiated loss function

    Raises:
        ImportError: If the loss module cannot be imported
        AttributeError: If the loss class cannot be found in the module
        Exception: If the loss cannot be instantiated
    """
    loss_class_name = config.get("class")
    loss_params = config.get("params", {})

    if not loss_class_name:
        raise ValueError("Loss configuration must contain 'class'")

    # Check for built-in PyTorch losses
    if hasattr(nn, loss_class_name):
        return getattr(nn, loss_class_name)(**loss_params)

    # If not a built-in loss, try to load from custom path
    loss_path = config.get("path")
    if not loss_path:
        raise ValueError(f"Loss {loss_class_name} is not a built-in PyTorch loss. "
                         f"Please specify 'path' in the loss configuration.")

    try:
        # If loss_path is a direct path to a Python file
        if loss_path.endswith(".py"):
            module_name = os.path.basename(loss_path)[:-3]  # Remove .py extension
            spec = importlib.util.spec_from_file_location(module_name, loss_path)
            if not spec or not spec.loader:
                raise ImportError(f"Could not load module spec from {loss_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # If loss_path is a module path (e.g., 'losses.custom_loss')
            module = importlib.import_module(loss_path)

        loss_class = getattr(module, loss_class_name)
        loss = loss_class(**loss_params)
        return loss
    except ImportError as e:
        raise ImportError(f"Error importing loss module from {loss_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Could not find class {loss_class_name} in module {loss_path}: {e}")
    except Exception as e:
        raise Exception(f"Error instantiating loss {loss_class_name} from {loss_path}: {e}")

# ------------------------------------
# core/loaders/callback_loader.py
# ------------------------------------
import os
import logging
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from core.callbacks import (
    BestMetricCheckpoint,
    PeriodicCheckpoint,
    SamplePlotCallback,
    SamplePlot3DCallback,
    PredictionSaver,
    PredictionLogger,
    ConfigArchiver,
    SkipValidation,
)

__all__ = ["load_callbacks"]


def load_callbacks(
    callback_configs: List[Dict[str, Any]],
    output_dir: str,
    resume: Optional[str] = None,
    skip_valid_until_epoch: int = 0,
    save_gt_pred_val_test_every_n_epochs: int = 5,
    save_gt_pred_val_test_after_epoch: int = 0,
    save_gt_pred_max_samples: Optional[int] = None,
    project_root: Optional[str] = None,
) -> List[pl.Callback]:
    """
    Instantiate a list of Lightning callbacks from configuration.

    Args:
        callback_configs: List of dicts, each with keys:
            - name: Name of the callback class (one of the supported callbacks)
            - params: Optional dict of parameters for the constructor
        output_dir: Base directory for outputs
        resume: Resume checkpoint path; if provided, skip ConfigArchiver
        skip_valid_until_epoch: If >0, add SkipValidation
        save_gt_pred_val_test_every_n_epochs: frequency for PredictionSaver
        save_gt_pred_val_test_after_epoch: start epoch for PredictionSaver
        save_gt_pred_max_samples: max samples for PredictionSaver
        project_root: Root directory for ConfigArchiver

    Returns:
        List of instantiated pytorch_lightning.callbacks.Callback
    """
    callbacks: List[pl.Callback] = []

    for cfg in callback_configs:
        name = cfg.get("name")
        params = cfg.get("params", {}) or {}

        if name == "BestMetricCheckpoint":
            dirpath = os.path.join(output_dir, params.get("dirpath", "checkpoints"))
            callbacks.append(
                BestMetricCheckpoint(
                    dirpath=dirpath,
                    metric_names=params["metric_names"],
                    mode=params.get("mode", "max"),
                    save_last=params.get("save_last", True),
                    last_k=params.get("last_k", 1),
                    filename_template=params.get("filename_template", "best_{metric}"),
                )
            )

        elif name == "PeriodicCheckpoint":
            dirpath = os.path.join(output_dir, params.get("dirpath", "backup_checkpoints"))
            callbacks.append(
                PeriodicCheckpoint(
                    dirpath=dirpath,
                    every_n_epochs=params.get("every_n_epochs", 5),
                    prefix=params.get("prefix", "epoch"),
                )
            )

        elif name == "SamplePlotCallback":
            callbacks.append(
                SamplePlotCallback(
                    num_samples=params.get("num_samples", 5),
                    cmap=params.get("cmap", "coolwarm"),
                )
            )

        elif name == "SamplePlot3DCallback":
            callbacks.append(
                SamplePlot3DCallback(
                    params
                )
            )

        elif name == "PredictionSaver":
            save_dir = os.path.join(output_dir, params.get("save_dir", "saved_predictions"))
            callbacks.append(
                PredictionSaver(
                    save_dir=save_dir,
                    save_every_n_epochs=save_gt_pred_val_test_every_n_epochs,
                    save_after_epoch=save_gt_pred_val_test_after_epoch,
                    max_samples=save_gt_pred_max_samples,
                )
            )

        elif name == "PredictionLogger":
            log_dir = os.path.join(output_dir, params.get("log_dir", "prediction_logs"))
            callbacks.append(
                PredictionLogger(
                    log_dir=log_dir,
                    log_every_n_epochs=params.get("log_every_n_epochs", 1),
                    max_samples=params.get("max_samples", 4),
                    cmap=params.get("cmap", "coolwarm"),
                )
            )

        elif name == "ConfigArchiver":
            if resume is None:
                archive_dir = os.path.join(output_dir, params.get("output_dir", "code"))
                callbacks.append(
                    ConfigArchiver(
                        output_dir=archive_dir,
                        project_root=project_root or os.getcwd(),
                    )
                )

        elif name == "SkipValidation":
            if skip_valid_until_epoch > 0:
                callbacks.append(
                    SkipValidation(skip_until_epoch=skip_valid_until_epoch)
                )

        elif name == "LearningRateMonitor":
            callbacks.append(
                LearningRateMonitor(
                    logging_interval=params.get("logging_interval", "epoch")
                )
            )

        else:
            logging.getLogger(__name__).warning(
                f"Unrecognized callback '{name}', skipping."
            )

    return callbacks


# ------------------------------------
# core/callbacks/periodic_ckpt.py
# ------------------------------------
import os
import pytorch_lightning as pl

class PeriodicCheckpoint(pl.Callback):
    """
    Save the trainer / model state every `every_n_epochs` epochs.

    Args
    ----
    dirpath : str
        Where the *.ckpt* files will be written.
    every_n_epochs : int
        Save interval.
    prefix : str
        Filename prefix (default: "epoch").
    """

    def __init__(self, dirpath: str, every_n_epochs: int = 5, prefix: str = "epoch"):
        super().__init__()
        self.dirpath = dirpath
        self.every_n_epochs = every_n_epochs
        self.prefix = prefix
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch + 1  # epochs are 0-indexed internally
        if epoch % self.every_n_epochs == 0:
            filename = f"{self.prefix}{epoch:06d}.ckpt"
            ckpt_path = os.path.join(self.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            # optional: log the path so you can grep it later
            pl_module.logger.experiment.add_text("checkpoints/saved", ckpt_path, epoch)


# ------------------------------------
# core/callbacks/config_archiver.py
# ------------------------------------
import os
import shutil
import logging
import zipfile
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class ConfigArchiver(Callback):
    """
    Callback to archive configuration files and source code at the start of training.

    This callback creates:
      - A ZIP archive of config and source
      - A parallel folder copy of the same files (optional)
    """

    def __init__(
        self,
        output_dir: str,
        project_root: str,
        copy_folder: bool = True
    ):
        """
        Initialize the ConfigArchiver callback.

        Args:
            output_dir: Directory to save archives and/or copies
            project_root: Root directory of the project containing the code to archive
            copy_folder: Whether to also copy files into a folder alongside the ZIP
        """
        super().__init__()
        self.output_dir = output_dir
        self.project_root = project_root
        self.copy_folder = copy_folder
        self.logger = logging.getLogger(__name__)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Archive configuration and source code at the start of training.

        Creates a ZIP archive and, if enabled, copies the files to a folder.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        os.makedirs(self.output_dir, exist_ok=True)
        # Use epoch and timestamp for uniqueness
        timestamp = trainer.logger.experiment.current_epoch if hasattr(trainer.logger.experiment, 'current_epoch') else pl_module.current_epoch
        base_name = f"code_snapshot_{timestamp}"

        # Create ZIP archive
        zip_path = os.path.join(self.output_dir, f"{base_name}.zip")
        version = 1
        while os.path.exists(zip_path):
            zip_path = os.path.join(self.output_dir, f"{base_name}_v{version}.zip")
            version += 1

        self.logger.info(f"Creating ZIP archive at {zip_path}")
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            # Archive directories
            for folder in ['configs', 'core', 'models', 'losses', 'metrics', 'scripts', 'callbacks']:
                src_dir = os.path.join(self.project_root, folder)
                if os.path.isdir(src_dir):
                    for root, _, files in os.walk(src_dir):
                        for fname in files:
                            if fname.endswith(('.py', '.yaml', '.yml')):
                                full_path = os.path.join(root, fname)
                                arcname = os.path.relpath(full_path, self.project_root)
                                zipf.write(full_path, arcname)
            # train.py
            train_py = os.path.join(self.project_root, 'train.py')
            if os.path.exists(train_py):
                zipf.write(train_py, 'train.py')
            # seglit_module.py
            seglit_py = os.path.join(self.project_root, 'seglit_module.py')
            if os.path.exists(seglit_py):
                zipf.write(seglit_py, 'seglit_module.py')
        self.logger.info(f"ZIP archive created: {zip_path}")

        # Optionally create a folder copy
        if self.copy_folder:
            copy_path = os.path.join(self.output_dir, base_name)
            if os.path.exists(copy_path):
                copy_path = f"{copy_path}_v{version - 1}"  # same version count
            self.logger.info(f"Copying files to folder {copy_path}")
            os.makedirs(copy_path, exist_ok=True)
            for folder in ['configs', 'core', 'models', 'losses', 'metrics']:
                src_dir = os.path.join(self.project_root, folder)
                dst_dir = os.path.join(copy_path, folder)
                if os.path.isdir(src_dir):
                    shutil.copytree(src_dir, dst_dir)
            # train.py
            if os.path.exists(train_py):
                shutil.copy2(train_py, copy_path)
            # seglit_module.py
            if os.path.exists(seglit_py):
                shutil.copy2(seglit_py, copy_path)
            self.logger.info(f"Folder copy created at: {copy_path}")


# ------------------------------------
# core/callbacks/pred_logger.py
# ------------------------------------

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

class PredictionLogger(Callback):
    """
    Validation-only: accumulates up to `max_samples` and writes one PNG per epoch.
    Now uses *separate* vmin/vmax for GT vs. prediction.
    """
    def __init__(self,
                 log_dir: str,
                 log_every_n_epochs: int = 1,
                 max_samples: int = 4,
                 cmap: str = "coolwarm"):
        super().__init__()
        self.log_dir = log_dir
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples
        self.cmap = cmap
        self.logger = pl.utilities.logger.get_logs_dir_logger()
        self._reset_buffers()

    def _reset_buffers(self):
        self._images = []
        self._gts = []
        self._preds = []
        self._collected = 0
        self._logged_this_epoch = False

    def on_validation_epoch_start(self, trainer, pl_module):
        if (trainer.current_epoch+1) % self.log_every_n_epochs == 0:
            self._reset_buffers()
        else:
            self._logged_this_epoch = True

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx=0):
        if self._logged_this_epoch \
           or ((trainer.current_epoch+1) % self.log_every_n_epochs != 0):
            return

        x       = batch[pl_module.input_key].detach().cpu()
        y_true  = batch[pl_module.target_key].detach().cpu()
        y_pred  = outputs["predictions"].detach().cpu()

        take = min(self.max_samples - self._collected, x.shape[0])
        self._images.append(x[:take])
        self._gts   .append(y_true[:take])
        self._preds .append(y_pred[:take])
        self._collected += take

        if self._collected < self.max_samples:
            return

        imgs  = torch.cat(self._images, dim=0)
        gts   = torch.cat(self._gts,    dim=0)
        preds = torch.cat(self._preds,  dim=0)

        # separate signed limits
        vlim_gt   = float(gts.abs().max())
        vlim_pred = float(preds.abs().max())

        os.makedirs(self.log_dir, exist_ok=True)
        filename = os.path.join(
            self.log_dir,
            f"pred_epoch_{trainer.current_epoch:06d}.png"
        )

        fig, axes = plt.subplots(self.max_samples, 3,
                                 figsize=(12, 4 * self.max_samples),
                                 tight_layout=True)

        for i in range(self.max_samples):
            # Input
            ax = axes[i, 0]
            if imgs.shape[1] == 1:
                ax.imshow(imgs[i, 0], cmap='gray')
            else:
                im = torch.clamp(imgs[i].permute(1,2,0), 0, 1)
                ax.imshow(im)
            ax.set_title('Input')
            ax.axis('off')

            # Ground truth
            ax = axes[i, 1]
            ax.imshow(gts[i, 0],
                      cmap=self.cmap,
                      vmin=-vlim_gt,
                      vmax=vlim_gt)
            ax.set_title('Ground Truth')
            ax.axis('off')

            # Prediction
            ax = axes[i, 2]
            ax.imshow(preds[i, 0],
                      cmap=self.cmap,
                      vmin=-vlim_pred,
                      vmax=vlim_pred)
            ax.set_title('Prediction')
            ax.axis('off')

        plt.savefig(filename, dpi=150)
        plt.close(fig)

        self.logger.info(f"Saved prediction visualization: {filename}")
        self._logged_this_epoch = True


# ------------------------------------
# core/callbacks/sample_plot.py
# ------------------------------------
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


# ------------------------------------
# core/callbacks/best_metric_ckpt.py
# ------------------------------------

import os
import logging
from glob import glob
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class BestMetricCheckpoint(Callback):
    """
    Callback to save checkpoints for the best value of each metric.
    """
    
    def __init__(
        self, 
        dirpath: str, 
        metric_names: List[str],
        mode: Union[str, Dict[str, str]] = "min",
        save_last: bool = True,
        last_k: int = 5,  # Save last checkpoint every k epochs
        filename_template: str = "best_{metric}"
    ):
        """
        Initialize the BestMetricCheckpoint callback.
        
        Args:
            dirpath: Directory to save checkpoints to
            metric_names: List of metrics to monitor
            mode: Either "min", "max", or a dict mapping metric names to "min"/"max"
            save_last: Whether to save the last checkpoint
            last_k: Save last checkpoint every k epochs (reduce I/O)
            filename_template: Template for checkpoint filenames
        """
        super().__init__()
        self.dirpath = dirpath
        self.metric_names = metric_names
        self.last_k = last_k
        
        # Setup mode for each metric (min or max)
        self.mode = {}
        if isinstance(mode, str):
            for metric in metric_names:
                self.mode[metric] = mode
        else:
            self.mode = mode
            # Ensure all metrics have a mode
            for metric in metric_names:
                if metric not in self.mode:
                    self.mode[metric] = "min"
                    
        self.save_last = save_last
        self.filename_template = filename_template
        self.best_values = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize best values
        for metric in metric_names:
            if self.mode[metric] == "min":
                self.best_values[metric] = float('inf')
            else:
                self.best_values[metric] = float('-inf')
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Check metrics at the end of validation and save checkpoint if needed.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.dirpath, exist_ok=True)

        # print(f"Saving best metric checkpoints to {self.dirpath}")
        # print(f"Current epoch: {trainer.current_epoch}, Max epochs: {trainer.max_epochs}")
        # print(f"Metrics being monitored: {self.metric_names}")
        # print(f"Mode for metrics: {self.mode}")
        # print(f"Best values so far: {self.best_values}")

        # Check each metric
        for metric in self.metric_names:
            metric_key = f"val_metrics/{metric}"
            # print('trainer.callback_metrics:', trainer.callback_metrics)
            if metric_key in trainer.callback_metrics:
                current_value = trainer.callback_metrics[metric_key].item()
                self.logger.info(f"Current value for {metric}: {current_value}")
                is_better = False
                if self.mode[metric] == "min" and current_value < self.best_values[metric]:
                    is_better = True
                    self.best_values[metric] = current_value
                elif self.mode[metric] == "max" and current_value > self.best_values[metric]:
                    is_better = True
                    self.best_values[metric] = current_value
                
                if is_better:
                    filename = f"{self.filename_template.format(metric=metric)}.ckpt"
                    filepath = os.path.join(self.dirpath, filename)
                    self.logger.info(f"Saving best {metric} checkpoint: {filepath}")
                    trainer.save_checkpoint(filepath)
        
        # Save last checkpoint if requested (with reduced frequency)
        if self.save_last and (
            (trainer.current_epoch+1) % self.last_k == 0 or  # Every k epochs
            trainer.current_epoch == trainer.max_epochs - 1  # Last epoch
        ):
            filename = "last.ckpt"
            filepath = os.path.join(self.dirpath, filename)
            trainer.save_checkpoint(filepath)
            self.logger.info(f"Saved last checkpoint at epoch {trainer.current_epoch}")





# ------------------------------------
# core/callbacks/pred_saver.py
# ------------------------------------
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl

class PredictionSaver(pl.Callback):
    """
    Save model predictions and ground truths on train, validation, and test.
    Works with Lightning 2.x using *_batch_end hooks.
    """
    def __init__(
        self,
        save_dir: str,
        save_every_n_epochs: int = 1,
        save_after_epoch: int = 0,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.every = save_every_n_epochs
        self.after = save_after_epoch
        self.max_samples = max_samples
        self._counter = 0
        # Per-epoch switch
        self._save_gts_this_epoch = False
        # Ever-saved guard to ensure GTs only once
        self._gts_already_saved = False

    def _should_save(self, epoch: int) -> bool:
        # print('epoch, self.every', epoch, self.every)
        return epoch >= self.after and (epoch + 1) % self.every == 0

    def _save_tensor(
        self,
        array: np.ndarray,
        split: str,
        epoch: int,
        batch_idx: int,
        sample_idx: int,
        which: str
    ):
        fname = f"{split}_e{epoch}_b{batch_idx}_i{sample_idx}_{which}.npy"
        folder = os.path.join(self.save_dir, split, f"epoch={epoch}")
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, fname), array)

    # # ——— TRAINING HOOKS ———
    # def on_train_epoch_start(self, trainer, pl_module):
    #     self._counter = 0

    # def on_train_batch_end(
    #     self,
    #     trainer: pl.Trainer,
    #     pl_module: pl.LightningModule,
    #     outputs: Dict[str, Any],
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx: int = 0,
    # ):
    #     epoch = trainer.current_epoch
    #     if not self._should_save(epoch):
    #         return

    #     preds = outputs.get("predictions")
    #     gts   = outputs.get("gts")
    #     if preds is None or gts is None:
    #         return

    #     preds = preds.detach().cpu().numpy()
    #     gts   = gts.detach().cpu().numpy()
    #     for i in range(preds.shape[0]):
    #         if self.max_samples is not None and self._counter >= self.max_samples:
    #             return
    #         self._save_tensor(preds[i], "train", epoch, batch_idx, i, "pred")
    #         self._save_tensor(gts[i],   "train", epoch, batch_idx, i, "gt")
    #         self._counter += 1

    # ——— VALIDATION HOOKS ———
    def on_validation_epoch_start(self, trainer, pl_module):
        self._counter = 0
        # only enable GT saving if this is the first matching epoch
        epoch = trainer.current_epoch
        self._save_gts_this_epoch = self._should_save(epoch) and not self._gts_already_saved

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        epoch = trainer.current_epoch
        if not self._should_save(epoch):
            return

        preds = outputs.get("predictions")
        gts   = outputs.get("gts")
        if preds is None or gts is None:
            return

        # preds = preds.detach().cpu().numpy()
        # gts   = gts.detach().cpu().numpy()
        # for i in range(preds.shape[0]):
        #     if self.max_samples is not None and self._counter >= self.max_samples:
        #         return
        #     self._save_tensor(preds[i], "val", epoch, batch_idx, i, "pred")
        #     self._save_tensor(gts[i],   "val", epoch, batch_idx, i, "gt")
        #     self._counter += 1

        # always save preds as before
        preds_np = preds.detach().cpu().numpy()
        for i in range(preds_np.shape[0]):
            if self.max_samples is not None and self._counter >= self.max_samples:
                break
            self._save_tensor(preds_np[i], "val", epoch, batch_idx, i, "pred")
            self._counter += 1

        # save gts for every batch—but only in that one epoch
        # if self._save_gts_this_epoch:
        #     gts_np = gts.detach().cpu().numpy()
        #     for i in range(gts_np.shape[0]):
        #         self._save_tensor(gts_np[i], "val", epoch, batch_idx, i, "gt")

    def on_validation_epoch_end(self, trainer, pl_module):
        # mark that we've done our one-time GT dump
        if self._save_gts_this_epoch:
            self._gts_already_saved = True

    # ——— TEST HOOKS ———
    def on_test_epoch_start(self, trainer, pl_module):
        self._counter = 0

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        epoch = trainer.current_epoch
        if not self._should_save(epoch):
            return

        preds = outputs.get("predictions")
        gts   = outputs.get("gts")
        if preds is None or gts is None:
            return

        preds = preds.detach().cpu().numpy()
        gts   = gts.detach().cpu().numpy()
        for i in range(preds.shape[0]):
            if self.max_samples is not None and self._counter >= self.max_samples:
                return
            self._save_tensor(preds[i], "test", epoch, batch_idx, i, "pred")
            self._save_tensor(gts[i],   "test", epoch, batch_idx, i, "gt")
            self._counter += 1


# ------------------------------------
# core/callbacks/__init__.py
# ------------------------------------
# core/callbacks/__init__.py

from core.callbacks.best_metric_ckpt    import BestMetricCheckpoint
from core.callbacks.periodic_ckpt       import PeriodicCheckpoint
from core.callbacks.pred_saver          import PredictionSaver
from core.callbacks.sample_plot         import SamplePlotCallback, SamplePlot3DCallback
from core.callbacks.skip_validation     import SkipValidation
from core.callbacks.config_archiver     import ConfigArchiver
from core.callbacks.pred_logger   import PredictionLogger   # <<–– add this line


# ------------------------------------
# core/callbacks/skip_validation.py
# ------------------------------------

import logging
from pytorch_lightning.callbacks import Callback

class SkipValidation(Callback):
    """
    Skip the entire validation loop until a given epoch by zeroing out
    `trainer.limit_val_batches`. Restores the original setting once the
    epoch threshold is reached.
    """
    def __init__(self, skip_until_epoch: int = 0):
        super().__init__()
        self.skip_until_epoch = skip_until_epoch
        self._original_limit_val_batches = None
        self.logger = logging.getLogger(__name__)

    def on_fit_start(self, trainer, pl_module):
        # Capture the user's configured limit_val_batches
        self._original_limit_val_batches = trainer.limit_val_batches

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.skip_until_epoch:
            if trainer.limit_val_batches != 0:
                self.logger.info(
                    f"Skipping validation until epoch {self.skip_until_epoch} "
                    f"(current: {trainer.current_epoch})"
                )
                trainer.limit_val_batches = 0
        else:
            # Restore the original setting once we've reached the target epoch
            if trainer.limit_val_batches == 0:
                trainer.limit_val_batches = self._original_limit_val_batches
                self.logger.info(
                    f"Resuming validation from epoch {trainer.current_epoch}"
                )


# ------------------------------------
# losses/chamfer_class.py
# ------------------------------------
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

# ------------------------------------
# losses/lif_weighted_mse.py
# ------------------------------------
import torch
import torch.nn as nn

class LIFWeightedMSELoss(nn.Module):
    """
    Log-Inverse-Frequency weighted MSE loss with optional global LUT freezing.

    Each pixel weight = 1 / log(1 + eps + freq_k), where freq_k is the relative
    frequency of the pixel's SDF bin across the current batch or a frozen dataset.

    Args:
        sdf_min (float): lower clamp for SDF values (e.g. -d_max).
        sdf_max (float): upper clamp for SDF values (e.g. +d_max).
        n_bins (int): number of histogram bins.
        eps (float): small constant inside log to avoid division-by-zero.
        freeze_after_first (bool): if True, build LUT once at first forward and reuse.
        reduction (str): 'mean', 'sum', or 'none'.
    """
    def __init__(
        self,
        sdf_min: float = -7.0,
        sdf_max: float = 7.0,
        n_bins: int = 256,
        eps: float = 0.02,
        freeze_after_first: bool = False,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        if sdf_max <= sdf_min:
            raise ValueError('sdf_max must be > sdf_min')

        # store range and scale as buffers
        self.register_buffer('sdf_min', torch.tensor(float(sdf_min)))
        self.register_buffer('sdf_max', torch.tensor(float(sdf_max)))
        self.register_buffer('scale', 1.0 / (self.sdf_max - self.sdf_min))

        self.n_bins = int(n_bins)
        self.eps = float(eps)
        self.freeze_after_first = bool(freeze_after_first)
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

        # LUT registered as buffer, persistent to allow checkpointing
        self.register_buffer('_lut', torch.ones(self.n_bins), persistent=True)
        self._lut_ready = False

    @torch.no_grad()
    def freeze(self, data_loader) -> None:
        """
        Build a global LUT from ground-truth SDFs in data_loader and freeze it.
        Subsequent forwards will reuse this LUT.
        """
        if self._lut_ready:
            return
        device = self.sdf_min.device
        counts = torch.zeros(self.n_bins, device=device)
        total = 0
        for batch in data_loader:
            sdf = batch.to(device)
            idx = self._bin_indices(sdf)
            counts += torch.bincount(idx.flatten(), minlength=self.n_bins)
            total += idx.numel()
        if total == 0:
            raise RuntimeError('freeze received empty data_loader')
        freq = counts.float() / total
        self._lut = 1.0 / torch.log1p(self.eps + freq)
        self._lut_ready = True

    @torch.no_grad()
    def _bin_indices(self, sdf: torch.Tensor) -> torch.LongTensor:
        clamped = torch.clamp(sdf, self.sdf_min, self.sdf_max)
        unit = (clamped - self.sdf_min) * self.scale
        idx = torch.round(unit * (self.n_bins - 1)).long()
        return idx

    @torch.no_grad()
    def _build_lut(self, sdf: torch.Tensor) -> torch.Tensor:
        idx = self._bin_indices(sdf)
        freq = torch.bincount(idx.flatten(), minlength=self.n_bins).float()
        freq /= idx.numel()
        return 1.0 / torch.log1p(self.eps + freq)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # build or reuse LUT
        if not self._lut_ready:
            with torch.no_grad():
                self._lut = self._build_lut(y_true)
                if self.freeze_after_first:
                    self._lut_ready = True

        # gather weights
        idx = self._bin_indices(y_true)
        w = self._lut[idx].to(dtype=y_pred.dtype)

        # weighted squared error
        wse = w * (y_pred - y_true).pow(2)

        # reduction
        if self.reduction == 'mean':
            return wse.sum() / y_pred.numel()
        if self.reduction == 'sum':
            return wse.sum()
        return wse

    def extra_repr(self) -> str:
        return (
            f'sdf_min={self.sdf_min.item()}, sdf_max={self.sdf_max.item()}, '
            f'n_bins={self.n_bins}, eps={self.eps}, '
            f'freeze_after_first={self.freeze_after_first}, reduction={self.reduction}'
        )


# ------------------------------------
# losses/vectorized_chamfer.py
# ------------------------------------
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


# ------------------------------------
# losses/fixed_lif_weighted_mse.py
# ------------------------------------
import torch
import torch.nn as nn

class FixedLUTWeightedMSELoss(nn.Module):
    """
    Same weighting formula as LIFWeightedMSELoss but with a *frozen* LUT that
    is loaded from disk (or passed as a tensor).

    Args
    ----
    lut_path (str | Tensor): 1-D tensor of length n_bins or path to .pt
    sdf_min / sdf_max (float): clamp range used when the LUT was built
    n_bins (int)            : number of histogram bins (must match LUT length)
    reduction ('mean'|'sum'|'none')
    """
    def __init__(
        self,
        lut_path,
        sdf_min: float = -7.0,
        sdf_max: float = 7.0,
        n_bins: int = 256,
        reduction: str = 'mean',
    ):
        super().__init__()
        # ---- common buffers -------------------------------------------------
        self.register_buffer('sdf_min', torch.tensor(float(sdf_min)))
        self.register_buffer('sdf_max', torch.tensor(float(sdf_max)))
        self.register_buffer('scale', 1.0 / (self.sdf_max - self.sdf_min))

        self.n_bins = int(n_bins)
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

        # ---- load LUT -------------------------------------------------------
        if isinstance(lut_path, str):
            lut = torch.load(lut_path, map_location='cpu')
        elif torch.is_tensor(lut_path):
            lut = lut_path
        else:
            raise TypeError("lut_path must be filename or Tensor")

        if lut.numel() != self.n_bins:
            raise ValueError(
                f"LUT length {lut.numel()} ≠ n_bins {self.n_bins}"
            )

        # *Same buffer name as dynamic class*
        self.register_buffer('_lut', lut.to(torch.float32), persistent=True)
        self._lut_ready = True        # already frozen

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _bin_indices(self, sdf: torch.Tensor) -> torch.LongTensor:
        clamped = torch.clamp(sdf, self.sdf_min, self.sdf_max)
        unit    = (clamped - self.sdf_min) * self.scale
        return torch.round(unit * (self.n_bins - 1)).long()

    # -------------------------------------------------------------------------
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        idx = self._bin_indices(y_true)
        w   = self._lut[idx].to(dtype=y_pred.dtype, device=y_pred.device)
        wse = w * (y_pred - y_true).pow(2)

        if self.reduction == 'mean':
            return wse.sum() / y_pred.numel()
        if self.reduction == 'sum':
            return wse.sum()
        return wse

    # -------------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"sdf_min={self.sdf_min.item()}, sdf_max={self.sdf_max.item()}, "
            f"n_bins={self.n_bins}, reduction={self.reduction}"
        )

    # -------------------------------------------------------------------------
    # Optional: load both old ('_lut') and new ('lut') keys seamlessly
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        key = prefix + "_lut"
        if key in state_dict and state_dict[key].numel() != self._lut.numel():
            print(
                f"⚠️  Replacing {key}: ckpt {tuple(state_dict[key].shape)} "
                f"→ current {tuple(self._lut.shape)}"
            )
            # put *our* 15-bin LUT into the state-dict
            state_dict[key] = self._lut.detach().cpu()

        # now let the normal loader run — sizes match, no missing keys
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

# ------------------------------------
# losses/simple_binary_weighted_mse.py
# ------------------------------------
import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """
    Per-pixel MSE with class-dependent weights (e.g. give roads > background).

    Args
    ----
    road_weight : float
        Weight applied to squared errors on road pixels.
    bg_weight   : float
        Weight applied to squared errors on background pixels.
    threshold   : float
        Threshold that separates “road” from “background” in the SDF.
    greater_is_road : bool
        If True, pixels **>= threshold** are treated as road.
        If False, pixels **<  threshold** are treated as road (default for SDF where roads are negative).
    reduction : {'mean', 'sum', 'none'}
        • 'mean' – divide the *weighted* SSE by the total number of elements  
          (so when both weights are 1 you recover standard MSE, and
          changing the class weights doesn’t blow up the loss scale).  
        • 'sum'  – return the weighted sum of squared errors.  
        • 'none' – return the full per-pixel tensor.
    """

    def __init__(
        self,
        road_weight: float = 5.0,
        bg_weight:   float = 1.0,
        threshold:   float = 0.0,
        greater_is_road: bool = False,
        reduction: str = "mean",
    ):
        super().__init__()
        self.road_weight   = float(road_weight)
        self.bg_weight     = float(bg_weight)
        self.threshold     = float(threshold)
        self.greater_is_road = bool(greater_is_road)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction     = reduction

    # ------------------------------------------------------------------ #
    # forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(self, y_pred: torch.Tensor, y_true_sdf: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred      : Tensor (N, 1, H, W)  – model output SDF
        y_true_sdf  : Tensor (N, 1, H, W)  – ground-truth signed-distance map

        Returns
        -------
        Tensor
            A scalar (for 'mean' / 'sum') or a tensor shaped like the input (for 'none').
        """

        # 1) build per-pixel weights
        if self.greater_is_road:
            is_road = (y_true_sdf > self.threshold)
        else:
            is_road = (y_true_sdf <=  self.threshold)
        weight = torch.where(is_road, self.road_weight, self.bg_weight).to(y_pred.dtype)

        # 2) weighted squared error
        wse = (y_pred - y_true_sdf) ** 2 * weight

        # 3) reduction
        if self.reduction == "mean":
            # divide by the *number of elements* so scale matches plain MSE
            return wse.sum() / y_pred.numel()
        elif self.reduction == "sum":
            return wse.sum()
        else:                       # 'none'
            return wse


# ------------------------------------
# losses/custom_loss.py
# ------------------------------------
"""
Example custom loss for segmentation.

This module demonstrates how to create a custom loss for the seglab framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologicalLoss(nn.Module):
    """
    TopologicalLoss: A loss function that incorporates topological features.
    
    This example loss combines binary cross-entropy with a term that penalizes
    topological errors like incorrect connectivity.
    """
    
    def __init__(
        self,
        topo_weight: float = 0.5,
        smoothness: float = 1.0,
        connectivity_weight: float = 0.3
    ):
        """
        Initialize the TopologicalLoss.
        
        Args:
            topo_weight: Weight for the topological component
            smoothness: Smoothness parameter for gradient computation
            connectivity_weight: Weight for the connectivity component
        """
        super().__init__()
        self.topo_weight = topo_weight
        self.smoothness = smoothness
        self.connectivity_weight = connectivity_weight
        self.bce_loss = nn.BCELoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the topological loss.
        
        Args:
            y_pred: Predicted segmentation masks
            y_true: Ground truth segmentation masks
            
        Returns:
            Tensor containing the calculated loss
        """
        # Binary cross-entropy component
        bce = self.bce_loss(y_pred, y_true)
        
        # Compute gradients for topology
        gradients_pred = self._compute_gradients(y_pred)
        gradients_true = self._compute_gradients(y_true)
        
        # Compute gradient loss
        gradient_loss = F.mse_loss(gradients_pred, gradients_true)
        
        # Compute connectivity loss (simplified example)
        connectivity_loss = self._compute_connectivity_loss(y_pred, y_true)
        
        # Combine losses
        topo_loss = gradient_loss + self.connectivity_weight * connectivity_loss
        total_loss = (1 - self.topo_weight) * bce + self.topo_weight * topo_loss
        
        return total_loss
    
    def _compute_gradients(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial gradients of the input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor of spatial gradients
        """
        # Ensure input is at least 4D: [batch, channels, height, width]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Apply Sobel filters
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=x.device).view(1, 1, 3, 3).repeat(1, x.shape[1], 1, 1)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=x.device).view(1, 1, 3, 3).repeat(1, x.shape[1], 1, 1)
        
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])
        
        # Compute gradient magnitude
        gradients = torch.sqrt(grad_x**2 + grad_y**2 + self.smoothness**2)
        
        return gradients
    
    def _compute_connectivity_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute connectivity loss between prediction and ground truth.
        
        This is a simplified example that penalizes disconnected regions.
        
        Args:
            y_pred: Predicted segmentation masks
            y_true: Ground truth segmentation masks
            
        Returns:
            Tensor containing the connectivity loss
        """
        # Apply morphological operations to find connected components
        # This is a simplified approximation for demonstration purposes
        
        # Convert to binary
        y_pred_bin = (y_pred > 0.5).float()
        y_true_bin = (y_true > 0.5).float()
        
        # Use dilated difference to approximate connectivity errors
        kernel_size = 3
        dilated_pred = F.max_pool2d(y_pred_bin, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        dilated_true = F.max_pool2d(y_true_bin, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
        # Connectivity error is higher when dilated regions differ
        connectivity_error = F.mse_loss(dilated_pred, dilated_true)
        
        return connectivity_error

# ------------------------------------
# losses/cape/loss.py
# ------------------------------------
import torch
from torch import nn
import numpy as np
import skimage.graph
import random
import cv2
from skimage.morphology import skeletonize
from .utils.graph_from_skeleton_3D import graph_from_skeleton as graph_from_skeleton_3D
from .utils.graph_from_skeleton_2D import graph_from_skeleton as graph_from_skeleton_2D
from .utils.crop_graph import crop_graph_2D, crop_graph_3D
from skimage.draw import line_nd
from scipy.ndimage import binary_dilation, generate_binary_structure
import networkx as nx


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CAPE(nn.Module):
    def __init__(self, window_size=128, three_dimensional=False, dilation_radius=10, shifting_radius=5, is_binary=False, distance_threshold=20, single_edge=False):
        super().__init__()
        """
        Initialize the CAPE loss module.

        Args:
            window_size (int): Size of the square patch (window) to process at a time.
            three_dimensional (bool): If True, operate in 3D mode; otherwise, operate in 2D.
            dilation_radius (int): Radius used to dilate ground-truth paths for masking.
            shifting_radius (int): Radius for refining start/end points to lowest-cost nearby pixels.
            is_binary (bool): If True, treat inputs as binary maps (invert predictions/ground truth).
            distance_threshold (float): Maximum value used for clipping ground-truth distance maps.
            single_edge (bool): If True, sample a single edge at a time; otherwise, sample a path.
        """
        self.window_size = window_size
        self.three_dimensional = three_dimensional
        self.dilation_radius = dilation_radius
        self.shifting_radius = shifting_radius
        self.is_binary = is_binary
        self.distance_threshold = distance_threshold
        self.single_edge = single_edge
        self.data_dim = 3 if three_dimensional else 2
    
    def _ensure_no_channel(self, t: torch.Tensor) -> torch.Tensor:
        """
        Remove a channel dimension for 2D or 3D data if it's a singleton.

        For 2D:
        (B,1,H,W) -> (B,H,W)
        For 3D:
        (B,1,D,H,W) -> (B,D,H,W)
        Leaves (B,H,W) or (B,D,H,W) unchanged.
        """
        expected_dim = 1 + self.data_dim  # batch + spatial
        if t.dim() != expected_dim and t.size(1) == 1:
            return t.squeeze(1)
        return t
        

    def _random_connected_pair(self, G):
        """
        Pick two distinct nodes that are in the same connected component.
        """
        node1 = random.choice(list(G.nodes()))
        reachable = list(nx.node_connected_component(G, node1))
        if len(reachable) == 1:
            return self._random_connected_pair(G)
        node2 = random.choice([n for n in reachable if n != node1])
        return node1, node2


    def _dilate_path_2D(self, shape, path_pts, radius):
        """
        Rasterise a poly-line into a thick 2D mask.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        for p, q in zip(path_pts[:-1], path_pts[1:]):
            cv2.line(mask,
                    (int(p[0]), int(p[1])),
                    (int(q[0]), int(q[1])),
                    1, int(radius))
        return mask
    
    
    def _dilate_path_3D(self, shape, path_positions, radius):
        """
        Rasterise a poly-line into a thick 3D mask.
        """
        mask = np.zeros(shape, dtype=np.uint8)

        for p1, p2 in zip(path_positions[:-1], path_positions[1:]):
            temp = np.zeros(shape, dtype=np.uint8)
            
            rr, cc, zz = line_nd(tuple(map(int, p1)), tuple(map(int, p2)))
            temp[zz, cc, rr] = 1

            struct = generate_binary_structure(3, 1)
            dilated_segment = binary_dilation(temp, structure=struct, iterations=int(radius))

            mask = np.logical_or(mask, dilated_segment)

        return mask.astype(np.uint8)
        
        
    def draw_line_with_thickness_3D(self, volume, start_point, end_point, value=1, thickness=1):
        """
        Draw a 3D line with specified thickness between two points in a volume using dilation.
        """
        rr, cc, zz = line_nd(start_point, end_point)
        volume[zz, cc, rr] = value
        
        struct = generate_binary_structure(3, 1)
        dilated_volume = binary_dilation(volume, structure=struct, iterations=thickness)
        
        return dilated_volume
      

    def find_min_in_radius_2D(self, array: np.ndarray, center: tuple, radius: float):
        """
        Finds the coordinates of the minimum value inside a given radius from a center point in a 2D array.
        """
        x0, y0 = center
        height, width = array.shape

        y_min, y_max = max(0, y0 - int(radius)), min(height, y0 + int(radius) + 1)
        x_min, x_max = max(0, x0 - int(radius)), min(width, x0 + int(radius) + 1)

        sub_image = array[y_min:y_max, x_min:x_max]
        
        min_idx = np.unravel_index(np.argmin(sub_image), sub_image.shape)

        min_coords = (y_min + min_idx[0], x_min + min_idx[1])
        return min_coords
    
    
    def find_min_in_radius_3D(self, array: np.ndarray, center: tuple, radius: float):
        """
        Finds the coordinates of the minimum value inside a given radius from a center point in a 3D array.
        """
        x0, y0, z0 = center
        depth, height, width = array.shape

        z_min, z_max = max(0, z0 - int(radius)), min(depth, z0 + int(radius) + 1)
        y_min, y_max = max(0, y0 - int(radius)), min(height, y0 + int(radius) + 1)
        x_min, x_max = max(0, x0 - int(radius)), min(width, x0 + int(radius) + 1)

        sub_volume = array[z_min:z_max, y_min:y_max, x_min:x_max]
        
        
        min_idx = np.unravel_index(np.argmin(sub_volume), sub_volume.shape)
        
        min_coords = (z_min + min_idx[0], y_min + min_idx[1], x_min + min_idx[2])
        return min_coords
    
    
    def path_cost_2D(self, cost_tensor, pred_cost_map, start_point, end_point, dilation_radius=20, extra_path=None):  
        """
        Compute the shortest path cost in 2D using Dijkstra's algorithm.
        """
        start_point = (int(start_point[0]), int(start_point[1]))
        end_point   = (int(end_point[0]), int(end_point[1]))
        dilation_radius = int(dilation_radius)

        if extra_path is None:                            
            dilated_image = np.zeros_like(pred_cost_map, dtype=np.uint8)
            cv2.line(dilated_image, start_point, end_point,
                    color=1, thickness=int(dilation_radius))
        else:                                             
            dilated_image = self._dilate_path_2D(pred_cost_map.shape,
                                                extra_path,
                                                dilation_radius)

        pred_cost_map = self.distance_threshold - pred_cost_map
        dilated_image = dilated_image * pred_cost_map
        dilated_image = self.distance_threshold - dilated_image
        dilated_image = np.where(dilated_image == self.distance_threshold, float('inf'), dilated_image)
        path_cost = torch.tensor(0.0, requires_grad=True).to(device)
        
        start_refined = self.find_min_in_radius_2D(dilated_image, start_point, radius=self.shifting_radius)
        end_refined = self.find_min_in_radius_2D(dilated_image, end_point, radius=self.shifting_radius)

        dilated_image = np.maximum(dilated_image, 0) + 0.00001
        
        try:

            path_coords, _ = skimage.graph.route_through_array(
                dilated_image, start=start_refined, end=end_refined, fully_connected=True, geometric=True)

            path_coords = np.transpose(np.array(path_coords), (1, 0))
            path_cost = torch.sum(cost_tensor[path_coords[0], path_coords[1]] ** 2).to(device)
            
            return path_cost
        
        except Exception as e:

            return path_cost
        
        
    def path_cost_3D(self, cost_tensor, pred_cost_map, start_point, end_point, dilation_radius=5, extra_path=None):
        """
        Compute the shortest path cost in 3D using Dijkstra's algorithm.
        """
        if extra_path is None:
            dilated_image = self.draw_line_with_thickness_3D(
                np.zeros_like(pred_cost_map, dtype=np.uint8),
                start_point, end_point, value=1, thickness=dilation_radius)
            
        else:                                           
            dilated_image = self._dilate_path_3D(pred_cost_map.shape,
                                                extra_path,
                                                dilation_radius)
        
        
        dilated_image = dilated_image.astype(np.uint8)
        dilated_image = np.where(dilated_image, 1, 0)
        
        pred_cost_map_temp = self.distance_threshold - pred_cost_map
        dilated_image = dilated_image * pred_cost_map_temp
        dilated_image = self.distance_threshold - dilated_image
        dilated_image = np.where(dilated_image == self.distance_threshold, float('inf'), dilated_image)
        
        start_refined = self.find_min_in_radius_3D(dilated_image, start_point, radius=self.shifting_radius)
        end_refined = self.find_min_in_radius_3D(dilated_image, end_point, radius=self.shifting_radius)
        
        dilated_image = np.maximum(dilated_image, 0) + 0.00001
        
        try:
            path_coords, _ = skimage.graph.route_through_array(
                dilated_image, start=start_refined, end=end_refined, fully_connected=True, geometric=True
            )
            path_coords = np.array(path_coords).T
            path_cost = torch.sum((cost_tensor[path_coords[0], path_coords[1], path_coords[2]]) ** 2).to(device)
            
            return path_cost
        
        except Exception as e:
            return torch.tensor(0.0, requires_grad=True).to(device)
        

    def forward(self, predictions, ground_truths):
        """
        Compute the average CAPE loss over a batch of predictions and ground truths.

        The method splits each prediction volume/mask into patches (windows), extracts
        or receives a graph representation of the skeletonized ground truth in each window,
        samples paths from the graph, computes the squared-distance sum along each predicted path,
        and accumulates these costs to return the mean loss per batch.

        Args:
            predictions (torch.Tensor): Distance maps of shape
                - (batch, H, W) for 2D
                - (batch, D, H, W) for 3D
            ground_truths (Union[nx.Graph, np.ndarray, torch.Tensor]):
                - Graph objects for direct skeleton-based sampling,
                - Or binary masks (arrays or tensors) to skeletonize.

        Returns:
            torch.Tensor: Scalar tensor representing the mean CAPE loss over the batch.
        """
        predictions = self._ensure_no_channel(predictions)
        ground_truths = self._ensure_no_channel(ground_truths)

        batch_size = predictions.size(0)
        total_loss = 0.0

        if isinstance(ground_truths[0], nx.Graph):
            gt_type = 0
        
        elif isinstance(ground_truths, np.ndarray):
            gt_type = 1
            
        elif isinstance(ground_truths, torch.Tensor):
            ground_truths = ground_truths.detach().cpu().numpy()
            gt_type = 1
        
        
        if self.is_binary:
            
            predictions = 1 - predictions
            
            if gt_type == 1:
                ground_truths = 1 - ground_truths
                
            self.distance_threshold = 1
        
        
        
        # ── 2D MODE ───────────────────────────────────────────────────────────────
        
        if self.three_dimensional == False:

            for b in range(batch_size):

                full_prediction_map = predictions[b]
                
                # NO GRAPH INPUT
                if gt_type == 1:
                    full_ground_truth_mask = (ground_truths[b] == 0).astype(np.uint8)
                
                # GRAPH INPUT    
                elif gt_type == 0:
                    complete_graph = ground_truths[b]

                assert predictions.shape[-1] % self.window_size == 0, "Width must be divisible by window size"
                assert predictions.shape[-2] % self.window_size == 0, "Height must be divisible by window size"

                num_windows_height = predictions.shape[-2] // self.window_size
                num_windows_width = predictions.shape[-1] // self.window_size

                crop_loss_sum = 0.0
                

                for i in range(num_windows_height):
                    for j in range(num_windows_width):
                        window_pred = full_prediction_map[i * full_prediction_map.shape[0] // num_windows_height:(i + 1) * full_prediction_map.shape[0] // num_windows_height, :]
                        window_pred = window_pred[:, j * full_prediction_map.shape[1] // num_windows_width:(j + 1) * full_prediction_map.shape[1] // num_windows_width]
                        
                        # NO GRAPH INPUT
                        if gt_type == 1:
                            
                            window_gt = full_ground_truth_mask[i * full_prediction_map.shape[0] // num_windows_height:(i + 1) * full_prediction_map.shape[0] // num_windows_height, :]
                            window_gt = window_gt[:, j * full_prediction_map.shape[1] // num_windows_width:(j + 1) * full_prediction_map.shape[1] // num_windows_width]

                            skeleton = skeletonize(window_gt)
                            graph = graph_from_skeleton_2D(skeleton, angle_range=(175,185), verbose=False)
                        
                        # GRAPH INPUT
                        elif gt_type == 0:
                            graph = crop_graph_2D(complete_graph,
                                                ymin=i * full_prediction_map.shape[0] // num_windows_height,
                                                xmin=j * full_prediction_map.shape[1] // num_windows_width,
                                                ymax=(i + 1) * full_prediction_map.shape[0] // num_windows_height,
                                                xmax=(j + 1) * full_prediction_map.shape[1] // num_windows_width)

                        window_loss = 0.0
                        
                        if self.single_edge == False:
                        
                            while list(graph.edges()):
                                n1, n2 = self._random_connected_pair(graph)

                                path_nodes = nx.shortest_path(graph, n1, n2)
                                path_pos   = [graph.nodes[n]['pos'] for n in path_nodes]

                                single_loss = self.path_cost_2D(
                                    cost_tensor=window_pred,
                                    pred_cost_map=window_pred.detach().cpu().numpy(),
                                    start_point=path_pos[0], end_point=path_pos[-1],
                                    dilation_radius=self.dilation_radius,
                                    extra_path=path_pos
                                )

                                graph.remove_edges_from(zip(path_nodes[:-1], path_nodes[1:]))
                                window_loss += single_loss                            
                            
                        else:
                            
                            edges = list(graph.edges())
                            
                            while list(graph.edges()):
                                edges = list(graph.edges())
        
                                edge = random.choice(edges)
                                node_1 = edge[0]
                                node_2 = edge[1]
                                
                                node_idx1 = list(graph.nodes).index(node_1)
                                node_idx2 = list(graph.nodes).index(node_2)

                                node1 = list(graph.nodes)[node_idx1]
                                node2 = list(graph.nodes)[node_idx2]

                                node1_pos = graph.nodes()[node1]['pos']
                                node2_pos = graph.nodes()[node2]['pos']

                                single_loss = self.path_cost_2D(
                                    cost_tensor=window_pred,
                                    pred_cost_map=window_pred.detach().cpu().numpy(),
                                    start_point=node1_pos, end_point=node2_pos,
                                    dilation_radius=self.dilation_radius
                                )
                                
                                graph.remove_edge(*edge)
                                window_loss += single_loss
                            
                        crop_loss_sum += window_loss
                        
                total_loss += crop_loss_sum
                
            return total_loss / batch_size if batch_size > 0 else 0

        # ── 3D MODE ───────────────────────────────────────────────────────────────

        else:
            
            for b in range(batch_size):
                full_prediction_map = predictions[b]
                
                # NO GRAPH INPUT
                if gt_type == 1:
                    full_ground_truth_mask = (ground_truths[b] == 0).astype(np.uint8)
                
                # GRAPH INPUT    
                elif gt_type == 0:
                    complete_graph = ground_truths[b]
                    
                assert predictions.shape[-3] % self.window_size == 0, "Depth must be divisible by window size"
                assert predictions.shape[-2] % self.window_size == 0, "Height must be divisible by window size"
                assert predictions.shape[-1] % self.window_size == 0, "Width must be divisible by window size"

                num_windows_depth = predictions.shape[-3] // self.window_size
                num_windows_height = predictions.shape[-2] // self.window_size
                num_windows_width = predictions.shape[-1] // self.window_size
                
                crop_loss_sum = 0.0
                for d in range(num_windows_depth):
                    for i in range(num_windows_height):
                        for j in range(num_windows_width):
                            window_pred = full_prediction_map[
                                d * self.window_size:(d + 1) * self.window_size,
                                i * self.window_size:(i + 1) * self.window_size,
                                j * self.window_size:(j + 1) * self.window_size
                            ]
                            
                            # NO GRAPH INPUT
                            if gt_type == 1:
                                
                                window_gt = full_ground_truth_mask[
                                    d * self.window_size:(d + 1) * self.window_size,
                                    i * self.window_size:(i + 1) * self.window_size,
                                    j * self.window_size:(j + 1) * self.window_size
                                ]
                                skeleton = skeletonize(window_gt)
                                graph = graph_from_skeleton_3D(skeleton, angle_range=(175,185), verbose=False)
                            
                            # GRAPH INPUT
                            elif gt_type == 0:
                                graph = crop_graph_3D(       
                                        complete_graph,
                                        xmin=j * self.window_size,
                                        ymin=i * self.window_size,
                                        zmin=d * self.window_size,
                                        xmax=j * self.window_size + self.window_size,
                                        ymax=i * self.window_size + self.window_size,
                                        zmax=d * self.window_size + self.window_size)
                            
                            window_loss = 0.0
                            
                            if self.single_edge == False:
                            
                                while list(graph.edges()):
                                    n1, n2 = self._random_connected_pair(graph)

                                    path_nodes = nx.shortest_path(graph, n1, n2)
                                    path_pos   = [graph.nodes[n]['pos'] for n in path_nodes]

                                    single_loss = self.path_cost_3D(
                                        cost_tensor=window_pred,
                                        pred_cost_map=window_pred.detach().cpu().numpy(),
                                        start_point=path_pos[0], end_point=path_pos[-1],
                                        dilation_radius=self.dilation_radius,
                                        extra_path=path_pos
                                    )
                                    
                                    graph.remove_edges_from(zip(path_nodes[:-1], path_nodes[1:]))
                                    window_loss += single_loss
                               
                            else: 
                                
                                while list(graph.edges()):
                                    edge = random.choice(list(graph.edges()))
                                    node1, node2 = edge
                                    node1_pos = graph.nodes[node1]['pos']
                                    node2_pos = graph.nodes[node2]['pos']

                                    single_loss = self.path_cost_3D(
                                        cost_tensor=window_pred,
                                        pred_cost_map=window_pred.detach().cpu().numpy(),
                                        start_point=node1_pos, end_point=node2_pos,
                                        dilation_radius=self.dilation_radius
                                    )
                                    
                                    graph.remove_edge(*edge)
                                    window_loss += single_loss
                            
                            crop_loss_sum += window_loss
                            
                total_loss += crop_loss_sum
                
            return total_loss / batch_size if batch_size > 0 else 0



# ------------------------------------
# losses/cape/__init__.py
# ------------------------------------


# ------------------------------------
# losses/cape/utils/graph_from_skeleton_2D.py
# ------------------------------------
import numpy as np
import networkx as nx
import time
import copy

def pixel_graph(skeleton):

    _skeleton = copy.deepcopy(np.uint8(skeleton))
    _skeleton[0,:] = 0
    _skeleton[:,0] = 0
    _skeleton[-1,:] = 0
    _skeleton[:,-1] = 0
    G = nx.Graph()

    # add one node for each active pixel
    xs,ys = np.where(_skeleton>0)
    G.add_nodes_from([(int(x),int(y)) for i,(x,y) in enumerate(zip(xs,ys))])

    # add one edge between each adjacent active pixels
    for (x,y) in G.nodes():
        patch = _skeleton[x-1:x+2, y-1:y+2]
        patch[1,1] = 0
        for _x,_y in zip(*np.where(patch>0)):
            if not G.has_edge((x,y),(x+_x-1,y+_y-1)):
                G.add_edge((x,y),(x+_x-1,y+_y-1))

    for n,data in G.nodes(data=True):
        data['pos'] = np.array(n)[::-1]

    return G

def compute_angle_degree(c, p0, p1):
    p0c = np.sqrt((c[0] - p0[0]) ** 2 + (c[1] - p0[1]) ** 2)
    p1c = np.sqrt((c[0] - p1[0]) ** 2 + (c[1] - p1[1]) ** 2)
    p0p1 = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)

    # Prevent division by zero
    denominator = 2 * p1c * p0c
    if denominator == 0:
        return 0  # or some default value like 180

    cos_value = (p1c**2 + p0c**2 - p0p1**2) / denominator

    # Clip values to prevent NaN issues
    cos_value = np.clip(cos_value, -1.0, 1.0)

    return np.arccos(cos_value) * 180 / np.pi


def distance_point_line(c,p0,p1):
    return np.linalg.norm(np.cross(p0-c, c-p1))/np.linalg.norm(p1-p0)

def decimate_nodes_angle_distance(G, angle_range=(110,240), dist=0.3, verbose=True):

    H = copy.deepcopy(G)

    def f():
        start = time.time()
        nodes = list(H.nodes())
        np.random.shuffle(nodes)
        changed = False
        for n in nodes:

            ajacent_nodes = list(nx.neighbors(H, n))
            if n in ajacent_nodes:
                ajacent_nodes.remove(n)
            if len(ajacent_nodes)==2:
                angle = compute_angle_degree(n, *ajacent_nodes)
                d = distance_point_line(np.array(n), np.array(ajacent_nodes[0]), np.array(ajacent_nodes[1]))
                if d<dist or (angle>angle_range[0] and angle<angle_range[1]):
                    H.remove_node(n)
                    H.add_edge(*ajacent_nodes)
                    changed = True
        return changed

    while True:
        if verbose:
            print("Remaining nodes:", len(H.nodes()))
        if not f():
            break

    if verbose:
        print("Finished. Remaining nodes:", len(H.nodes()))

    return H

def remove_close_nodes(G, dist=10, verbose=True):

    H = copy.deepcopy(G)
    def _remove_close_nodes():
        edges = list(H.edges())
        changed = False
        for (s,t) in edges:
            if H.has_node(s) and H.has_node(t):
                d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2)
                if d<dist:
                    if len(H.edges(s))==2:
                        ajacent_nodes = list(nx.neighbors(H, s))
                        if s in ajacent_nodes:
                            ajacent_nodes.remove(s)
                        if t in ajacent_nodes:
                            ajacent_nodes.remove(t)
                        if len(ajacent_nodes)==1:
                            d = np.sqrt((s[0]-ajacent_nodes[0][0])**2+(s[1]-ajacent_nodes[0][1])**2)
                            if d<dist:
                                H.remove_node(s)
                                H.add_edge(ajacent_nodes[0], t)
                                changed = True
                    elif len(H.edges(t))==2:
                        ajacent_nodes = list(nx.neighbors(H, t))
                        if s in ajacent_nodes:
                            ajacent_nodes.remove(s)
                        if t in ajacent_nodes:
                            ajacent_nodes.remove(t)
                        if len(ajacent_nodes)==1:
                            d = np.sqrt((t[0]-ajacent_nodes[0][0])**2+(t[1]-ajacent_nodes[0][1])**2)
                            if d<dist:
                                H.remove_node(t)
                                H.add_edge(ajacent_nodes[0], s)
                                changed = True
        return changed

    while True:
        if verbose:
            print("Remaining nodes:", len(H.nodes()))
        if not _remove_close_nodes():
            break

    if verbose:
        print("Finished. Remaining nodes:", len(H.nodes()))

    return H

def remove_small_dangling(G, length=10, verbose=True):

    H = copy.deepcopy(G)
    edges = list(H.edges())
    for (s,t) in edges:
        d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2)
        if d<length:
            edge_count_s = len(H.edges(s))
            edge_count_t = len(H.edges(t))
            if edge_count_s==1:
                H.remove_node(s)
            if edge_count_t==1:
                H.remove_node(t)

    return H

def merge_close_intersections(G, dist=10, verbose=True):

    H = copy.deepcopy(G)
    def _merge_close_intersections():
        edges = list(H.edges())
        changed = False
        for (s,t) in edges:
            d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2)
            if d<dist:
                if len(H.edges(s))>2 and len(H.edges(t))>2:
                    ajacent_nodes = list(nx.neighbors(H, s))
                    if t in ajacent_nodes:
                        ajacent_nodes.remove(t)
                    H.remove_node(s)
                    for n in ajacent_nodes:
                        H.add_edge(n, t)
                    changed = True
                else:
                    pass
        return changed

    while True:
        if verbose:
            print("Remaining nodes:", len(H.nodes()))
        if not _merge_close_intersections():
            break

    if verbose:
        print("Finished. Remaining nodes:", len(H.nodes()))

    return H

def graph_from_skeleton(skeleton, angle_range=(135,225), dist_line=3,
                        dist_node=10, verbose=True, max_passes=20, relabel=True):
    """
    Parameters
    ----------
    skeleton : numpy.ndarray
        binary skeleton
    angle_range : (min,max) in degree
        two connected edges are merged into one if the angle between them
        is in this range
    dist_line : pixels
        two connected edges are merged into one if the distance between
        the central node to the line connecting the external nodes is
        lower then this value.
    dist_node : pixels
        two nodes that are connected by an edge are "merged" if their distance is
        lower than this value.
    """
    if verbose: print("Creation of densly connected graph.")
    G = pixel_graph(skeleton)

    for i in range(max_passes):

        if verbose: print("Pass {}:".format(i))

        n = len(G.nodes())

        if verbose: print("\tFirst decimation of nodes.")
        G = decimate_nodes_angle_distance(G, angle_range, dist_line, verbose)

        if verbose: print("\tFirst removing close nodes.")
        G = remove_close_nodes(G, dist_node, verbose)


        if verbose: print("\tRemoving short danglings.")
        G = remove_small_dangling(G, length=dist_node)

        if verbose: print("\tMerging close intersections.")
        G = merge_close_intersections(G, dist_node, verbose)

        if n==len(G.nodes()):
            break

    if relabel:
        mapping = dict(zip(G.nodes(), range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)

    return G


# ------------------------------------
# losses/cape/utils/graph_from_skeleton_3D.py
# ------------------------------------
import numpy as np
import networkx as nx
import time
import copy

def pixel_graph(skeleton):

    _skeleton = copy.deepcopy(np.uint8(skeleton))
    _skeleton[0,:,:] = 0
    _skeleton[:,0,:] = 0
    _skeleton[:,:,0] = 0
    _skeleton[-1,:,:] = 0
    _skeleton[:,-1,:] = 0
    _skeleton[:,:,-1] = 0
    G = nx.Graph()

    # add one node for each active pixel
    xs,ys,zs = np.where(_skeleton>0)
    G.add_nodes_from([(int(x),int(y),int(z)) for i,(x,y,z) in enumerate(zip(xs,ys,zs))])

    # add one edge between each adjacent active pixels
    for (x,y,z) in G.nodes():
        patch = _skeleton[x-1:x+2, y-1:y+2, z-1:z+2]
        patch[1,1,1] = 0
        for _x,_y,_z in zip(*np.where(patch>0)):
            if not G.has_edge((x,y,z),(x+_x-1,y+_y-1,z+_z-1)):
                G.add_edge((x,y,z),(x+_x-1,y+_y-1,z+_z-1))

    for n,data in G.nodes(data=True):
        data['pos'] = np.array(n)[::-1]

    return G

def compute_angle_degree(c, p0, p1):
    p0c = np.sqrt((c[0]-p0[0])**2+(c[1]-p0[1])**2+(c[2]-p0[2])**2)
    p1c = np.sqrt((c[0]-p1[0])**2+(c[1]-p1[1])**2+(c[2]-p1[2])**2)
    p0p1 = np.sqrt((p1[0]-p0[0])**2+(p1[1]-p0[1])**2+(p1[2]-p0[2])**2)
    return np.arccos((p1c*p1c+p0c*p0c-p0p1*p0p1)/(2*p1c*p0c))*180/np.pi
    # cos_val = (p1c*p1c + p0c*p0c - p0p1*p0p1) / (2 * p1c * p0c)
    ## clamp to [-1, 1] to guard against floating-point drift
    # cos_val = np.clip(cos_val, -1.0, 1.0)
    # angle_rad = np.arccos(cos_val)
    # return angle_rad * 180.0 / np.pi

def distance_point_line(c,p0,p1):
    return np.linalg.norm(np.cross(p0-c, c-p1))/np.linalg.norm(p1-p0)

def decimate_nodes_angle_distance(G, angle_range=(110,240), dist=0.3, verbose=True):

    H = copy.deepcopy(G)

    def f():
        start = time.time()
        nodes = list(H.nodes())
        np.random.shuffle(nodes)
        changed = False
        for n in nodes:

            ajacent_nodes = list(nx.neighbors(H, n))
            if n in ajacent_nodes:
                ajacent_nodes.remove(n)
            if len(ajacent_nodes)==2:
                angle = compute_angle_degree(n, *ajacent_nodes)
                d = distance_point_line(np.array(n), np.array(ajacent_nodes[0]), np.array(ajacent_nodes[1]))
                if d<dist or (angle>angle_range[0] and angle<angle_range[1]):
                    H.remove_node(n)
                    H.add_edge(*ajacent_nodes)
                    changed = True
        return changed

    while True:
        if verbose:
            print("Remaining nodes:", len(H.nodes()))
        if not f():
            break

    if verbose:
        print("Finished. Remaining nodes:", len(H.nodes()))

    return H

def remove_close_nodes(G, dist=10, verbose=True):

    H = copy.deepcopy(G)
    def _remove_close_nodes():
        edges = list(H.edges())
        changed = False
        for (s,t) in edges:
            if H.has_node(s) and H.has_node(t):
                d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2+(s[2]-t[2])**2)
                if d<dist:
                    if len(H.edges(s))==2:
                        ajacent_nodes = list(nx.neighbors(H, s))
                        if s in ajacent_nodes:
                            ajacent_nodes.remove(s)
                        if t in ajacent_nodes:
                            ajacent_nodes.remove(t)
                        if len(ajacent_nodes)==1:
                            d = np.sqrt((s[0]-ajacent_nodes[0][0])**2+(s[1]-ajacent_nodes[0][1])**2+(s[2]-ajacent_nodes[0][2])**2)
                            if d<dist:
                                H.remove_node(s)
                                H.add_edge(ajacent_nodes[0], t)
                                changed = True
                    elif len(H.edges(t))==2:
                        ajacent_nodes = list(nx.neighbors(H, t))
                        if s in ajacent_nodes:
                            ajacent_nodes.remove(s)
                        if t in ajacent_nodes:
                            ajacent_nodes.remove(t)
                        if len(ajacent_nodes)==1:
                            d = np.sqrt((t[0]-ajacent_nodes[0][0])**2+(t[1]-ajacent_nodes[0][1])**2+(t[2]-ajacent_nodes[0][2])**2)
                            if d<dist:
                                H.remove_node(t)
                                H.add_edge(ajacent_nodes[0], s)
                                changed = True
        return changed

    while True:
        if verbose:
            print("Remaining nodes:", len(H.nodes()))
        if not _remove_close_nodes():
            break

    if verbose:
        print("Finished. Remaining nodes:", len(H.nodes()))

    return H

def remove_small_dangling(G, length=10, verbose=True):

    H = copy.deepcopy(G)
    edges = list(H.edges())
    for (s,t) in edges:
        d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2+(s[2]-t[2])**2)
        if d<length:
            edge_count_s = len(H.edges(s))
            edge_count_t = len(H.edges(t))
            if edge_count_s==1:
                H.remove_node(s)
            if edge_count_t==1:
                H.remove_node(t)

    return H

def merge_close_intersections(G, dist=10, verbose=True):

    H = copy.deepcopy(G)
    def _merge_close_intersections():
        edges = list(H.edges())
        changed = False
        for (s,t) in edges:
            d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2+(s[2]-t[2])**2)
            if d<dist:
                if len(H.edges(s))>2 and len(H.edges(t))>2:
                    ajacent_nodes = list(nx.neighbors(H, s))
                    if t in ajacent_nodes:
                        ajacent_nodes.remove(t)
                    H.remove_node(s)
                    for n in ajacent_nodes:
                        H.add_edge(n, t)
                    changed = True
                else:
                    pass
        return changed

    while True:
        if verbose:
            print("Remaining nodes:", len(H.nodes()))
        if not _merge_close_intersections():
            break

    if verbose:
        print("Finished. Remaining nodes:", len(H.nodes()))

    return H

def graph_from_skeleton(skeleton, angle_range=(135,225), dist_line=3,
                        dist_node=10, verbose=True, max_passes=20, relabel=True):
    """
    Parameters
    ----------
    skeleton : numpy.ndarray
        binary skeleton
    angle_range : (min,max) in degree
        two connected edges are merged into one if the angle between them
        is in this range
    dist_line : pixels
        two connected edges are merged into one if the distance between
        the central node to the line connecting the external nodes is
        lower then this value.
    dist_node : pixels
        two nodes that are connected by an edge are "merged" if their distance is
        lower than this value.
    """
    if verbose: print("Creation of densly connected graph.")
    G = pixel_graph(skeleton)

    for i in range(max_passes):

        if verbose: print("Pass {}:".format(i))

        n = len(G.nodes())

        if verbose: print("\tFirst decimation of nodes.")
        G = decimate_nodes_angle_distance(G, angle_range, dist_line, verbose)

        if verbose: print("\tFirst removing close nodes.")
        G = remove_close_nodes(G, dist_node, verbose)


        if verbose: print("\tRemoving short danglings.")
        G = remove_small_dangling(G, length=dist_node)

        if verbose: print("\tMerging close intersections.")
        G = merge_close_intersections(G, dist_node, verbose)

        if n==len(G.nodes()):
            break

    if relabel:
        mapping = dict(zip(G.nodes(), range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)

    return G


# ------------------------------------
# losses/cape/utils/crop_graph.py
# ------------------------------------
import numpy as np
from shapely.geometry import box, LineString, Point
import networkx as nx

def crop_graph_2D(graph, xmin, ymin, xmax, ymax, precision=8):
    
    bounding_box = box(xmin, ymin, xmax, ymax)
    
    cropped_graph = nx.Graph()
    node_positions = {}      
    inside_nodes = {}        
    coord_to_node = {}      
    
    for n, data in graph.nodes(data=True):
        pos = data['pos']
        node_positions[n] = pos
        x, y = pos
        if xmin <= x <= xmax and ymin <= y <= ymax:
            new_pos = (x - xmin, y - ymin)
            inside_nodes[n] = new_pos
            cropped_graph.add_node(n, pos=new_pos)
    
            key = (round(new_pos[0], precision), round(new_pos[1], precision))
            coord_to_node[key] = n
    
    max_node_index = max(graph.nodes()) if graph.nodes else 0

    for u, v, data in graph.edges(data=True):
        u_pos = node_positions[u]
        v_pos = node_positions[v]
        line = LineString([u_pos, v_pos])
        
        if u in inside_nodes and v in inside_nodes:
            if u != v:
                cropped_graph.add_edge(u, v, **data)
            continue
        
        if not bounding_box.intersects(line):
            continue

        intersection = bounding_box.intersection(line)
        if intersection.is_empty:
            continue

        if intersection.geom_type == 'Point':
            pts = [(intersection.x, intersection.y)]
        elif intersection.geom_type == 'MultiPoint':
            pts = [(pt.x, pt.y) for pt in intersection.geoms]
        elif intersection.geom_type == 'LineString':
            pts = list(intersection.coords)
        else:
            continue

        pts.sort(key=lambda pt: line.project(Point(pt)))
        
        new_nodes = []
        for pt in pts:
            new_pos = (pt[0] - xmin, pt[1] - ymin)
            key = (round(new_pos[0], precision), round(new_pos[1], precision))
            if key in coord_to_node:
                node_id = coord_to_node[key]
            else:
                max_node_index += 1
                node_id = max_node_index
                cropped_graph.add_node(node_id, pos=new_pos)
                coord_to_node[key] = node_id
            new_nodes.append(node_id)
        
        endpoints = []
        if u in inside_nodes:
            endpoints.append(u)
        endpoints.extend(new_nodes)
        if v in inside_nodes:
            endpoints.append(v)

        for i in range(len(endpoints) - 1):
            if endpoints[i] != endpoints[i+1]:
                cropped_graph.add_edge(endpoints[i], endpoints[i+1], **data)
    
    return cropped_graph


def crop_graph_3D(graph, xmin, ymin, zmin, xmax, ymax, zmax, precision=8):
    
    def _to_voxel(u, lo, hi):
        v = int(np.floor(u - lo))          
        return max(0, min(v, hi - lo - 1)) 

    def _segment_box_intersections(p0, p1):

        pts = []
        (x0, y0, z0), (x1, y1, z1) = p0, p1
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

        for plane, (k0, k1, p) in (
            ("x", (x0, dx, xmin)), ("x", (x0, dx, xmax)),
            ("y", (y0, dy, ymin)), ("y", (y0, dy, ymax)),
            ("z", (z0, dz, zmin)), ("z", (z0, dz, zmax)),
        ):
            k0, dk, plane_val = k0, k1, p
            if dk == 0:                        
                continue
            t = (plane_val - k0) / dk
            if 0 < t < 1:                      
                x = x0 + t * dx
                y = y0 + t * dy
                z = z0 + t * dz
                if xmin - 1e-6 <= x <= xmax + 1e-6 and \
                ymin - 1e-6 <= y <= ymax + 1e-6 and \
                zmin - 1e-6 <= z <= zmax + 1e-6:
                    pts.append((x, y, z))

        pts.sort(key=lambda pt: (pt[0]-x0)**2 + (pt[1]-y0)**2 + (pt[2]-z0)**2)
        return pts

    cropped = nx.Graph()
    inside_nodes, pos_cache, coord2id = {}, {}, {}

    for n, d in graph.nodes(data=True):
        x, y, z = d["pos"]
        pos_cache[n] = (x, y, z)
        if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
            vx, vy, vz = (_to_voxel(x, xmin, xmax),
                        _to_voxel(y, ymin, ymax),
                        _to_voxel(z, zmin, zmax))
            
            inside_nodes[n] = (vx, vy, vz)
            cropped.add_node(n, pos=inside_nodes[n])
            coord2id[(vx, vy, vz)] = n

    next_id = (max(graph.nodes()) if graph.nodes else 0) + 1

    for u, v, edata in graph.edges(data=True):
        p0, p1 = pos_cache[u], pos_cache[v]

        if (u not in inside_nodes) and (v not in inside_nodes):

            if (p0[0] < xmin and p1[0] < xmin) or (p0[0] > xmax and p1[0] > xmax) \
            or (p0[1] < ymin and p1[1] < ymin) or (p0[1] > ymax and p1[1] > ymax) \
            or (p0[2] < zmin and p1[2] < zmin) or (p0[2] > zmax and p1[2] > zmax):
                continue

        split_pts = _segment_box_intersections(p0, p1)

        node_chain = []
        if u in inside_nodes:
            node_chain.append(u)

        for pt in split_pts:
            
            vz = _to_voxel(pt[2], zmin, zmax)
            vy = _to_voxel(pt[1], ymin, ymax)
            vx = _to_voxel(pt[0], xmin, xmax)
        
            key = (vx, vy, vz)
            if key in coord2id:
                node_id = coord2id[key]
            else:
                node_id = next_id
                next_id += 1
                cropped.add_node(node_id, pos=(vx, vy, vz))
                coord2id[key] = node_id
            node_chain.append(node_id)

        if v in inside_nodes:
            node_chain.append(v)

        for a, b in zip(node_chain[:-1], node_chain[1:]):
            if a != b:
                cropped.add_edge(a, b, **edata)

    return cropped



# ------------------------------------
# models/base_models.py
# ------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BIAS = True

nn_Conv      = lambda three_dimensional: nn.Conv3d if three_dimensional else nn.Conv2d
nn_ConvTrans = lambda three_dimensional: nn.ConvTranspose3d if three_dimensional else nn.ConvTranspose2d
nn_BatchNorm = lambda three_dimensional: nn.BatchNorm3d if three_dimensional else nn.BatchNorm2d
# nn_GroupNorm = lambda three_dimensional: nn.BatchNorm3d if three_dimensional else nn.BatchNorm2d
nn_Dropout   = lambda three_dimensional: nn.Dropout3d if three_dimensional else nn.Dropout2d
nn_MaxPool   = lambda three_dimensional: nn.MaxPool3d if three_dimensional else nn.MaxPool2d
nn_AvgPool   = lambda three_dimensional: nn.AvgPool3d if three_dimensional else nn.AvgPool2d

def verify_input_size(n_levels, size):

    assert(isinstance(size, int))

    is_not_pair = lambda x: x%2!=0

    current_size = size
    for n in range(n_levels):
        if is_not_pair(current_size):
            return False
        current_size = current_size/2.0
    return True

def possible_input_size(n_levels, step=2, range=(2,1024)):
    good_in_sizes = []
    for in_size in np.arange(range[0],range[1], step):
        if verify_input_size(n_levels, int(in_size)):
            good_in_sizes.append(in_size)
    return np.array(good_in_sizes)

class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_first=False, n_convs=2,
                 dropout=0.3, norm_type='batch', num_groups=8, pooling="max", three_dimensional=False):
        super().__init__()

        _3d = three_dimensional

        layers = []

        if not is_first:
            in_channels = out_channels//2
            if pooling=="max":
                layers.append(nn_MaxPool(_3d)(2))
            elif pooling=="avg":
                layers.append(nn_AvgPool(_3d)(2))
            else:
                raise ValueError("Unrecognized option pooling=\"{}\"".format(pooling))

        layers.append(nn_Conv(_3d)(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=BIAS))
        # if batch_norm:
        #     # layers.append(nn_BatchNorm(_3d)(out_channels))
        #     layers.append(nn.GroupNorm(8, out_channels))
        if norm_type=='batch':
            layers.append(nn_BatchNorm(_3d)(out_channels))
        elif norm_type=='group':
            layers.append(nn.GroupNorm(num_groups, out_channels))
        # elif norm_type=='instance':
        #     layers.append(nn.InstanceNorm3d(out_channels) if _3d else nn.InstanceNorm2d(out_channels))
        # else:  # norm_type is None → no normalization
        #     pass
        layers.append(nn.ReLU(inplace=True))
        for i in range(n_convs-1):
            layers.append(nn_Conv(_3d)(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=BIAS))
#             if batch_norm:
# #                 layers.append(nn_BatchNorm(_3d)(out_channels))
#                 layers.append(nn.GroupNorm(8, out_channels))
            if norm_type=='batch':
                layers.append(nn_BatchNorm(_3d)(out_channels))
            elif norm_type=='group':
                layers.append(nn.GroupNorm(num_groups, out_channels))
            # elif norm_type=='instance':
            #     layers.append(nn.InstanceNorm3d(out_channels) if _3d else nn.InstanceNorm2d(out_channels))
            # else:  # norm_type is None → no normalization
            #     pass
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn_Dropout(_3d)(p=dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

class UpBlock(nn.Module):

    def __init__(self, in_channels, n_convs=2, dropout=0.3,
                 norm_type='batch', num_groups=8, upsampling='deconv', three_dimensional=False, align_corners=False):
        super().__init__()

        _3d = three_dimensional
        out_channels = in_channels//2

        if upsampling=='deconv':
            self.upsampling = nn_ConvTrans(_3d)(in_channels, out_channels, kernel_size=2, stride=2, bias=BIAS)
            '''
            self.upsampling = nn.Sequential(
                                nn_ConvTrans(_3d)(in_channels, out_channels, kernel_size=2, stride=2, bias=BIAS),
                                nn_BatchNorm(_3d)(out_channels),
                                nn.ReLU(inplace=True))
            '''
        elif upsampling in ['nearest', 'bilinear', 'trilinear']:
            mode = upsampling
            if _3d and upsampling == 'bilinear':
                mode = 'trilinear'
            # align_corners only applies to non-nearest
            uc = {} if mode=='nearest' else {'align_corners': align_corners}
            self.upsampling = nn.Sequential(
                                nn.Upsample(size=None, scale_factor=2, mode=mode, **uc),
                                nn_Conv(_3d)(in_channels, out_channels, kernel_size=1, padding=0, bias=BIAS))
        else:
            raise ValueError("Unrecognized upsampling option {}".fomrat(upsampling))
        

        layers = []
        layers.append(nn_Conv(_3d)(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=BIAS))
#         if batch_norm:
# #             layers.append(nn_BatchNorm(_3d)(out_channels))
#             layers.append(nn.GroupNorm(8, out_channels))
        if norm_type=='batch':
            layers.append(nn_BatchNorm(_3d)(out_channels))
        elif norm_type=='group':
            layers.append(nn.GroupNorm(num_groups, out_channels))
        # elif norm_type=='instance':
        #     layers.append(nn.InstanceNorm3d(out_channels) if _3d else nn.InstanceNorm2d(out_channels))
        # else:  # norm_type is None → no normalization
        #     pass
        layers.append(nn.ReLU(inplace=True))
        for i in range(n_convs-1):
            layers.append(nn_Conv(_3d)(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=BIAS))
#             if batch_norm:
# #                 layers.append(nn_BatchNorm(_3d)(out_channels))
#                 layers.append(nn.GroupNorm(8, out_channels))
            if norm_type=='batch':
                layers.append(nn_BatchNorm(_3d)(out_channels))
            elif norm_type=='group':
                layers.append(nn.GroupNorm(num_groups, out_channels))
            # elif norm_type=='instance':
            #     layers.append(nn.InstanceNorm3d(out_channels) if _3d else nn.InstanceNorm2d(out_channels))
            # else:  # norm_type is None → no normalization
            #     pass
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn_Dropout(_3d)(p=dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, input, to_stack):

        x = self.upsampling(input)

        x = torch.cat([to_stack, x], dim=1)

        return self.layers(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, m_channels=64, out_channels=2, n_convs=1,
                 n_levels=3, dropout=0.0, norm_type='batch', num_groups=8, upsampling='bilinear',
                 pooling="max", three_dimensional=False, apply_final_relu=True):
        super().__init__()

        assert n_levels>=1
        assert n_convs>=1

        self.in_channels = in_channels
        self.m_channels = m_channels
        self.out_channels = out_channels
        self.n_convs = n_convs
        self.n_levels = n_levels
        self.dropout = dropout
        self.norm_type = norm_type 
        self.num_groups = num_groups
        self.upsampling = upsampling
        self.pooling = pooling
        self.three_dimensional = three_dimensional
        _3d = three_dimensional
        self.apply_final_relu = apply_final_relu

        channels = [2**x*m_channels for x in range(0, self.n_levels+1)]

        down_block = lambda inch, outch, is_first: DownBlock(inch, outch, is_first, n_convs,
                                                             dropout, norm_type, num_groups, pooling,
                                                             three_dimensional)
        up_block = lambda inch: UpBlock(inch, n_convs, dropout, norm_type, num_groups,
                                        upsampling, three_dimensional)

        # === Down path ===
        down_path = []
        down_path.append(down_block(in_channels, channels[0], True))
        for i, ch in enumerate(channels[1:-1]):
            down_path.append(down_block(None, ch, False))

        # === Bottom ===
        bottom = []
        bottom.append(down_block(None, channels[-1], False))

        # === Up path ===
        up_path = []
        for i, ch in enumerate(reversed(channels[1:])):
            up_path.append(up_block(ch))

        self.down_path = nn.Sequential(*down_path)
        self.bottom = nn.Sequential(*bottom)
        self.up_path = nn.Sequential(*up_path)
        self.last_layer = nn_Conv(_3d)(channels[0], out_channels, kernel_size=1, padding=0, bias=BIAS)

    def forward(self, input):

        try:
            # === Down path ===
            fmaps = [self.down_path[0](input)]
            for i in range(1, len(self.down_path._modules)):
                fmaps.append(self.down_path[i](fmaps[-1]))

            # === Bottom ===
            x = self.bottom(fmaps[-1])

            # === Up path ===
            for i in range(0, len(self.up_path._modules)):
                x = self.up_path[i](x, fmaps[len(fmaps)-i-1])

            # === Last ===
            x = self.last_layer(x)

            if self.apply_final_relu:
                x = F.relu(x)
            
        except Exception as e:
            for i,dim in enumerate(input.shape[2:]):
                if not verify_input_size(self.n_levels+1, size=dim):
                    print("Dimension {} of input tensor is not divisible by 2 in each step.".format(i+2))
                    possible_sizes = possible_input_size(self.n_levels+1, step=1, range=(dim//2,dim*2))
                    print("Valid input sizes {}".format(possible_sizes))
            raise

        return x

class UNetReg(nn.Module):
    def __init__(self, in_channels=1, m_channels=64, out_channels=1, n_convs=1,
                 n_levels=3, dropout=0.0, batch_norm=False, upsampling='bilinear',
                 pooling="max", three_dimensional=False):
        super().__init__()

        assert n_levels>=1
        assert n_convs>=1

        self.in_channels = in_channels
        self.m_channels = m_channels
        self.out_channels = out_channels
        self.n_convs = n_convs
        self.n_levels = n_levels
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.upsampling = upsampling
        self.pooling = pooling
        self.three_dimensional = three_dimensional
        _3d = three_dimensional

        channels = [2**x*m_channels for x in range(0, self.n_levels+1)]

        down_block = lambda inch, outch, is_first: DownBlock(inch, outch, is_first, n_convs,
                                                             dropout, batch_norm, pooling,
                                                             three_dimensional)
        up_block = lambda inch: UpBlock(inch, n_convs, dropout, batch_norm,
                                        upsampling, three_dimensional)

        # === Down path ===
        down_path = []
        down_path.append(down_block(in_channels, channels[0], True))
        for i, ch in enumerate(channels[1:-1]):
            down_path.append(down_block(None, ch, False))

        # === Bottom ===
        bottom = []
        bottom.append(down_block(None, channels[-1], False))

        # === Up path ===
        up_path = []
        for i, ch in enumerate(reversed(channels[1:])):
            up_path.append(up_block(ch))

        self.down_path = nn.Sequential(*down_path)
        self.bottom = nn.Sequential(*bottom)
        self.up_path = nn.Sequential(*up_path)
        self.last_layer = nn_Conv(_3d)(channels[0], out_channels, kernel_size=1, padding=0, bias=BIAS)

    def forward(self, input):

        try:
            # === Down path ===
            fmaps = [self.down_path[0](input)]
            for i in range(1, len(self.down_path._modules)):
                fmaps.append(self.down_path[i](fmaps[-1]))

            # === Bottom ===
            x = self.bottom(fmaps[-1])

            # === Up path ===
            for i in range(0, len(self.up_path._modules)):
                x = self.up_path[i](x, fmaps[len(fmaps)-i-1])

            # === Last ===
            x = self.last_layer(x)
            x = F.relu(x)

        except Exception as e:
            for i,dim in enumerate(input.shape[2:]):
                if not verify_input_size(self.n_levels+1, size=dim):
                    print("Dimension {} of input tensor is not divisible by 2 in each step.".format(i+2))
                    possible_sizes = possible_input_size(self.n_levels+1, step=1, range=(dim//2,dim*2))
                    print("Valid input sizes {}".format(possible_sizes))
            raise

        return x



class UNetBin(nn.Module):
    def __init__(self, in_channels=1, m_channels=64, out_channels=2, n_convs=1,
                 n_levels=3, dropout=0.0, batch_norm=False, upsampling='bilinear',
                 pooling="max", three_dimensional=False):
        super().__init__()

        assert n_levels>=1
        assert n_convs>=1

        self.in_channels = in_channels
        self.m_channels = m_channels
        self.out_channels = out_channels
        self.n_convs = n_convs
        self.n_levels = n_levels
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.upsampling = upsampling
        self.pooling = pooling
        self.three_dimensional = three_dimensional
        _3d = three_dimensional

        channels = [2**x*m_channels for x in range(0, self.n_levels+1)]

        down_block = lambda inch, outch, is_first: DownBlock(inch, outch, is_first, n_convs,
                                                             dropout, batch_norm, pooling,
                                                             three_dimensional)
        up_block = lambda inch: UpBlock(inch, n_convs, dropout, batch_norm,
                                        upsampling, three_dimensional)

        # === Down path ===
        down_path = []
        down_path.append(down_block(in_channels, channels[0], True))
        for i, ch in enumerate(channels[1:-1]):
            down_path.append(down_block(None, ch, False))

        # === Bottom ===
        bottom = []
        bottom.append(down_block(None, channels[-1], False))

        # === Up path ===
        up_path = []
        for i, ch in enumerate(reversed(channels[1:])):
            up_path.append(up_block(ch))

        self.down_path = nn.Sequential(*down_path)
        self.bottom = nn.Sequential(*bottom)
        self.up_path = nn.Sequential(*up_path)
        self.last_layer = nn_Conv(_3d)(channels[0], out_channels, kernel_size=1, padding=0, bias=BIAS)

    def forward(self, input):

        try:
            # === Down path ===
            fmaps = [self.down_path[0](input)]
            for i in range(1, len(self.down_path._modules)):
                fmaps.append(self.down_path[i](fmaps[-1]))

            # === Bottom ===
            x = self.bottom(fmaps[-1])

            # === Up path ===
            for i in range(0, len(self.up_path._modules)):
                x = self.up_path[i](x, fmaps[len(fmaps)-i-1])

            # === Last ===
            x = self.last_layer(x)
            x = torch.sigmoid(x)

        except Exception as e:
            for i,dim in enumerate(input.shape[2:]):
                if not verify_input_size(self.n_levels+1, size=dim):
                    print("Dimension {} of input tensor is not divisible by 2 in each step.".format(i+2))
                    possible_sizes = possible_input_size(self.n_levels+1, step=1, range=(dim//2,dim*2))
                    print("Valid input sizes {}".format(possible_sizes))
            raise

        return x

