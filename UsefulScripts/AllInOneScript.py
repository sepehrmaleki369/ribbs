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

from core.model_loader import load_model
from core.loss_loader import load_loss
from core.mix_loss import MixedLoss
from core.metric_loader import load_metrics
from core.dataloader import SegmentationDataModule
from core.callbacks import (
    BestMetricCheckpoint,
    PredictionLogger,
    ConfigArchiver,
    SkipValidation,
    SamplePlotCallback,
    PredictionSaver,
    PeriodicCheckpoint
)
from core.logger import setup_logger
from core.checkpoint import CheckpointManager
from core.utils import yaml_read, mkdir

from seglit_module import SegLitModule

# Silence noisy loggers
for lib in ('rasterio', 'matplotlib', 'PIL', 'tensorboard', 'urllib3'):
    logging.getLogger(lib).setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger('core').setLevel(logging.DEBUG)
logging.getLogger('__main__').setLevel(logging.DEBUG)


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

    # --- prepare output & logger ---
    output_dir = main_cfg.get("output_dir", "outputs")
    mkdir(output_dir)
    logger = setup_logger(os.path.join(output_dir, "training.log"))
    logger.info(f"Output dir: {output_dir}")

    # --- trainer params from YAML ---
    trainer_cfg                = main_cfg.get("trainer", {})
    skip_valid_until_epoch     = trainer_cfg["skip_validation_until_epoch"]

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
    )

    # --- callbacks ---
    callbacks: List[pl.Callback] = []
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    mkdir(ckpt_dir)

    callbacks.append(BestMetricCheckpoint(
        dirpath=ckpt_dir,
        metric_names=list(metric_list.keys()),
        mode="max",
        save_last=True,
        last_k=1,
    ))

    backup_ckpt_dir = os.path.join(output_dir, "backup_checkpoints")
    mkdir(backup_ckpt_dir)

    callbacks.append(PeriodicCheckpoint(               # <-- add this block
        dirpath=backup_ckpt_dir,
        every_n_epochs=trainer_cfg.get("save_checkpoints_every_n_epochs", 5)
    ))

    callbacks.append(SamplePlotCallback(
        num_samples=trainer_cfg["num_samples_plot"],
        cmap=trainer_cfg["cmap_plot"]
    ))

    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    if not args.resume:
        code_dir = os.path.join(output_dir, "code")
        mkdir(code_dir)
        callbacks.append(ConfigArchiver(
            output_dir=code_dir,
            project_root=os.path.dirname(os.path.abspath(__file__))
        ))

    if skip_valid_until_epoch > 0:
        callbacks.append(SkipValidation(skip_until_epoch=skip_valid_until_epoch))

    pred_save_dir = os.path.join(output_dir, "saved_predictions")
    mkdir(pred_save_dir)
    callbacks.append(PredictionSaver(
        save_dir=pred_save_dir,
        save_every_n_epochs=trainer_cfg["save_gt_pred_val_test_every_n_epochs"],
        save_after_epoch=trainer_cfg["save_gt_pred_val_test_after_epoch"],
        max_samples=trainer_cfg.get('save_gt_pred_max_samples', None),
    ))
    logging.getLogger("core.callbacks.PredictionSaver").setLevel(logging.DEBUG)

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
            checkpoint_dir=ckpt_dir,
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
            y_hat = self.validator.run_chunked_inference(self.model, x)
        return y_hat
    
    def on_train_epoch_start(self):
        self._train_preds = []
        self._train_gts   = []

        if isinstance(self.loss_fn, MixedLoss):
            self.loss_fn.update_epoch(self.current_epoch)
        for m in self.metrics.values():
            if hasattr(m, 'reset'):
                m.reset()

    def training_step(self, batch, batch_idx):
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
            if (self.current_epoch+1) % freq == 0:
                self.log(f"train_metrics/{name}", metric(y_hat, y_int),
                         prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        
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
            y_hat = self.validator.run_chunked_inference(self.model, x)

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
                self.log(f"val_metrics/{name}", metric(y_hat, y_int),
                         prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        
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
            y_hat = self.validator.run_chunked_inference(self.model, x)

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
            self.log(f"test_metrics/{name}", metric(y_hat, y_int),
                     prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

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
Now supports **both 2‑D (N, C, H, W)** and **3‑D (N, C, D, H, W)** inputs.
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
      *divisor* (default: 16) before tiling, then removes the pad.
    • Uses `patch_size` and `patch_margin` to create overlapping tiles.
      Only the *centre* region of each model prediction is kept and
      stitched together.

    Parameters
    ----------
    config : dict
        Dictionary with at least the keys:
            ``patch_size``   – tuple/list[int] (len == 2 or 3)
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
            ``(N, C, H, W)`` or ``(N, C, D, H, W)`` tensor.
        divisor : int, optional
            The divisor (default 16).

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
            pad_tuple.extend([0, p])  # (left = 0, right = p)

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
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Full‑image/volume inference with overlapping tiles.

        Workflow (N‑D):
        1) **Margin pad** by ``patch_margin`` (reflect/replicate).
        2) **Div‑16 pad** so every spatial dim is divisible by 16.
        3) **Sliding‑window** inference:
            • window      = ``patch_size``
            • window step = ``patch_size − 2*patch_margin``
            • model is applied on each window; only the *centre* region
              is placed into the output canvas.
        4) **Remove** the div‑16 pad.
        5) **Remove** the initial margin pad.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(N, out_channels, *original_spatial*)``.
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
        # (B) Second pad until all dims divisible by 16
        # ----------------------------------------------------------
        padded_image, pad_div16 = self._pad_to_valid_size(image, 16)
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
            test_patch, _ = self._pad_to_valid_size(test_patch, 16)
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
# core/model_loader.py
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
# core/metric_loader.py
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
# core/dataloader.py
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
            # disable sampling and augmentations for val/test
            if split in ('valid', 'test'):
                cfg['validate_road_ratio'] = False
                cfg['max_attempts'] = 1
                cfg['augmentations'] = []  # no augmentations
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
# core/loss_loader.py
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
        
    return output


# ------------------------------------
# core/mix_loss.py
# ------------------------------------
"""
Mixed loss module that switches between primary and secondary loss functions based on epoch.

This module provides functionality to mix multiple loss functions with configurable weights
and a switchover epoch for the secondary loss.
"""

from typing import Any, Dict, List, Optional, Union, Callable

import torch
import torch.nn as nn


class MixedLoss(nn.Module):
    """
    A loss module that mixes multiple loss functions with configurable weights.
    
    The secondary loss will only be activated after a specified epoch.
    
    Attributes:
        primary_loss: The primary loss function (used from epoch 0)
        secondary_loss: The secondary loss function (activated after start_epoch)
        alpha: Weight for the secondary loss (between 0 and 1)
        start_epoch: Epoch to start using the secondary loss
        current_epoch: Current training epoch (updated externally)
    """
    
    def __init__(
        self, 
        primary_loss: Union[nn.Module, Callable], 
        secondary_loss: Optional[Union[nn.Module, Callable]] = None,
        alpha: float = 0.5,
        start_epoch: int = 0
    ):
        """
        Initialize the MixedLoss module.
        
        Args:
            primary_loss: The primary loss function (used from epoch 0)
            secondary_loss: The secondary loss function (used after start_epoch)
            alpha: Weight for the secondary loss (between 0 and 1)
            start_epoch: Epoch to start using the secondary loss
        """
        super().__init__()
        self.primary_loss = primary_loss
        self.secondary_loss = secondary_loss
        self.alpha = alpha
        self.start_epoch = start_epoch
        self.current_epoch = 0
    
    def update_epoch(self, epoch: int) -> None:
        """
        Update the current epoch.
        
        Args:
            epoch: The current epoch number
        """
        self.current_epoch = epoch
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate the mixed loss.
        
        Args:
            y_pred: Predicted values
            y_true: Ground truth values
            
        Returns:
            Tensor containing the calculated loss
        """
        p = self.primary_loss(y_pred, y_true)
        if self.secondary_loss is None or self.current_epoch < self.start_epoch:
            s = torch.tensor(0.0, device=p.device, dtype=p.dtype)
        else:
            s = self.secondary_loss(y_pred, y_true)
        m = (1 - self.alpha) * p + self.alpha * s
        return {"primary": p, "secondary": s, "mixed": m}

# ------------------------------------
# core/__init__.py
# ------------------------------------
"""
Module __init__ file for the core package.

This file imports and exposes the main components from the core package.
"""

from core.model_loader import load_model
from core.loss_loader import load_loss
from core.mix_loss import MixedLoss
from core.metric_loader import load_metric, load_metrics
from core.dataloader import SegmentationDataModule
from core.validator import Validator
from core.callbacks import (
    BestMetricCheckpoint,
    PredictionLogger,
    ConfigArchiver,
    SkipValidation,
    PredictionSaver
)
from core.logger import setup_logger, ColoredFormatter
from core.checkpoint import CheckpointManager, CheckpointMode
from core.utils import *

from core.general_dataset import GeneralizedDataset, custom_collate_fn, worker_init_fn

__all__ = [
    'load_model',
    'load_loss',
    'MixedLoss',
    'load_metric',
    'load_metrics',
    'SegmentationDataModule',
    'Validator',
    'BestMetricCheckpoint',
    'PredictionLogger',
    'ConfigArchiver',
    'SkipValidation',
    'setup_logger',
    'ColoredFormatter',
    'CheckpointManager',
    'PredictionSaver',
    'CheckpointMode',
    'GeneralizedDataset',
    'custom_collate_fn',
    'worker_init_fn'
]

# ------------------------------------
# metrics/dice.py
# ------------------------------------
import torch
import torch.nn as nn

class ThresholdedDiceMetric(nn.Module):
    """
    Threshold each channel into a binary mask, then compute standard Dice.

    Args:
        threshold (float or str): cut‐off for binarization.
        eps (float or str): small constant to avoid zero‐division.
        multiclass (bool or str): if True, macro‐average over channels.
        zero_division (float or str): returned value when both pred & GT empty.
    """
    def __init__(
        self,
        threshold=0.5,
        eps=1e-6,
        multiclass=False,
        zero_division=1.0,
        greater_is_road=True
    ):
        super().__init__()
        # Force numerical types
        self.threshold     = float(threshold)
        self.eps           = float(eps)
        self.multiclass    = bool(multiclass)
        self.zero_division = float(zero_division)
        self.greater_is_road = bool(greater_is_road)

    def _binarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.greater_is_road:
            return (x >  self.threshold).float()
        else:
            return (x <=  self.threshold).float()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # ensure (N, C, H, W)
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)

        N, C, *_ = y_pred.shape
        if not self.multiclass and C != 1:
            raise ValueError(f"[ThresholdedDiceMetric] Binary mode expects 1 channel, got {C}")

        # binarize
        y_pred_bin = self._binarize(y_pred)
        y_true_bin = self._binarize(y_true)
        # flatten
        y_pred_flat = y_pred_bin.view(N, C, -1)
        y_true_flat = y_true_bin.view(N, C, -1)

        # intersection and sums
        inter = (y_pred_flat * y_true_flat).sum(-1)           # (N, C)
        sums  = y_pred_flat.sum(-1) + y_true_flat.sum(-1)     # (N, C)

        # build eps and zero_division as tensors
        device = y_pred.device
        eps_tensor = torch.tensor(self.eps, device=device, dtype=inter.dtype)
        zd_tensor = torch.tensor(self.zero_division, device=device, dtype=inter.dtype)

        # dice per sample/class with ε for stability
        dice = (2 * inter + eps_tensor) / (sums + eps_tensor)     # (N, C)

        # override exact-zero cases
        zero_mask = (sums == 0)
        if zero_mask.any():
            dice = torch.where(zero_mask, zd_tensor, dice)

        # mean over batch → (C,)
        dice_per_class = dice.mean(0)

        if not self.multiclass or C == 1:
            return dice_per_class.squeeze(0)

        return dice_per_class.mean()


# ------------------------------------
# metrics/iou.py
# ------------------------------------
import torch
import torch.nn as nn

class ThresholdedIoUMetric(nn.Module):
    """
    Threshold each channel into binary masks, then compute Intersection-over-Union.

    Args:
        threshold (float): cut-off for binarization.
        eps (float): small constant to stabilize non-zero cases.
        multiclass (bool): if True, macro-average over channels.
        zero_division (float): value to return when both pred and true are empty (union=0).
    """
    def __init__(
        self,
        threshold = 0.5,
        eps = 1e-6,
        multiclass = False,
        zero_division = 1.0,
        greater_is_road=True
    ):
        super().__init__()
        self.threshold     = float(threshold)
        self.eps           = float(eps)
        self.multiclass    = bool(multiclass)
        self.zero_division = float(zero_division)
        self.greater_is_road = bool(greater_is_road)

    def _binarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.greater_is_road:
            return (x >  self.threshold).float()
        else:
            return (x <=  self.threshold).float()
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure shape (N, C, H, W)
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)

        N, C, *_ = y_pred.shape
        if not self.multiclass and C != 1:
            raise ValueError(f"[ThresholdedIoUMetric] Binary mode expects 1 channel, got {C}")

        # Binarize
        y_pred_bin = self._binarize(y_pred)
        y_true_bin = self._binarize(y_true)
        
        # Flatten
        y_pred_flat = y_pred_bin.view(N, C, -1)
        y_true_flat = y_true_bin.view(N, C, -1)

        # Intersection and union
        inter = (y_pred_flat * y_true_flat).sum(-1)                    # (N, C)
        sum_pred = y_pred_flat.sum(-1)
        sum_true = y_true_flat.sum(-1)
        union = sum_pred + sum_true - inter                           # (N, C)

        # IoU per sample/class with ε for stability
        iou = (inter + self.eps) / (union + self.eps)                  # (N, C)

        # Handle zero-union explicitly: when union == 0, set to zero_division
        zero_mask = (union == 0)
        if zero_mask.any():
            iou = torch.where(zero_mask,
                              torch.tensor(self.zero_division, device=iou.device),
                              iou)

        # Mean over batch → (C,)
        iou_per_class = iou.mean(0)

        if not self.multiclass or C == 1:
            # Binary: return scalar
            return iou_per_class.squeeze(0)

        # Multiclass: macro-average
        return iou_per_class.mean()


# ------------------------------------
# metrics/apls.py
# ------------------------------------
import numpy as np
import torch
import torch.nn as nn
from metrics.apls_core import apls


def compute_batch_apls(
    gt_masks,
    pred_masks,
    threshold: float = 0.5,
    angle_range=(135, 225),
    max_nodes=500,
    max_snap_dist=4,
    allow_renaming=True,
    min_path_length=10,
    greater_is_road=True
):
    def _bin(x): return (x > threshold) if greater_is_road else (x <= threshold)

    # --- convert to numpy if needed ---
    if torch.is_tensor(gt_masks):
        gt_masks = gt_masks.detach().cpu().numpy()
    if torch.is_tensor(pred_masks):
        pred_masks = pred_masks.detach().cpu().numpy()

    # --- unify shapes to (B, H, W) ---
    def _unify(m):
        if m.ndim == 2:
            m = m[None, ...]
        if m.ndim == 3:
            return m
        if m.ndim == 4 and m.shape[1] == 1:
            return m[:,0,...]
        raise ValueError(f"Unsupported mask shape {m.shape}")

    gt = _unify(gt_masks)
    pr = _unify(pred_masks)
    if gt.shape != pr.shape:
        raise ValueError(f"GT shape {gt.shape} != pred shape {pr.shape}")

    B = gt.shape[0]
    scores = np.zeros(B, dtype=np.float32)

    for i in range(B):
        # Binarize with threshold
        gt_bin = _bin(gt[i]).astype(np.uint8)
        pr_bin = _bin(pr[i]).astype(np.uint8)

        # Skip empty ground truth
        if gt_bin.sum() == 0:
            scores[i] = 1.0 if pr_bin.sum() == 0 else 0.0
            continue

        try:
            scores[i] = apls(
                gt_bin,
                pr_bin,
                angle_range=angle_range,
                max_nodes=max_nodes,
                max_snap_dist=max_snap_dist,
                allow_renaming=allow_renaming,
                min_path_length=min_path_length
            )
        except Exception:
            scores[i] = 0.0

    return scores


class APLS(nn.Module):
    """
    Average Path Length Similarity (APLS) metric for road network segmentation.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        angle_range=(135, 225),
        max_nodes=500,
        max_snap_dist=4,
        allow_renaming=True,
        min_path_length=10,
        greater_is_road=True
    ):
        super().__init__()
        self.threshold = threshold
        self.angle_range = angle_range
        self.max_nodes = max_nodes
        self.max_snap_dist = max_snap_dist
        self.allow_renaming = allow_renaming
        self.min_path_length = min_path_length
        self.greater_is_road = bool(greater_is_road)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        scores = compute_batch_apls(
            y_true,
            y_pred,
            threshold=self.threshold,
            angle_range=self.angle_range,
            max_nodes=self.max_nodes,
            max_snap_dist=self.max_snap_dist,
            allow_renaming=self.allow_renaming,
            min_path_length=self.min_path_length,
            greater_is_road=self.greater_is_road,
        )
        return torch.tensor(scores.mean(), device=y_pred.device)


# ------------------------------------
# metrics/ccq.py
# ------------------------------------
"""
Connected Components Quality (CCQ) metric for segmentation.

This module provides a metric that evaluates segmentation quality based on
" "the quality of connected components in the prediction compared to ground truth.
" "Supports both binary masks and continuous distance-map outputs via a threshold.
"""

import torch
import torch.nn as nn
import numpy as np
from skimage import measure
from typing import List, Tuple, Set


class ConnectedComponentsQuality(nn.Module):
    """
    Connected Components Quality (CCQ) metric for evaluating segmentation quality.
    
    This metric considers both detection and shape accuracy of connected components
    in the predicted segmentation compared to the ground truth. It supports binary
    outputs as well as continuous-valued maps via a configurable threshold.
    """
    
    def __init__(
        self,
        min_size: int = 5,
        tolerance: int = 2,
        alpha: float = 0.5,
        threshold: float = 0.5,
        greater_is_road=True,
        eps: float = 1e-8,
    ):
        """
        Initialize the CCQ metric.
        
        Args:
            min_size: Minimum component size to consider
            tolerance: Pixel tolerance for component matching
            alpha: Weight between detection score and shape score (0 to 1)
            threshold: Scalar threshold for binarizing predictions and ground truth
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.min_size = min_size
        self.tolerance = tolerance
        self.alpha = alpha
        self.threshold = threshold
        self.eps = eps
        self.greater_is_road = bool(greater_is_road)
    
    def _bin(self, arr: np.ndarray) -> np.ndarray:
        return (arr >  self.threshold) if self.greater_is_road else \
               (arr <=  self.threshold)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the CCQ metric between predicted and ground truth masks.
        
        Args:
            y_pred: Predicted maps (B, 1, H, W), e.g., logits, probability maps,
                    or signed/unsigned distance maps
            y_true: Ground truth masks or continuous maps (B, 1, H, W)
        
        Returns:
            Tensor containing the CCQ score (higher is better)
        """
        # Process each item in the batch
        batch_size = y_pred.shape[0]
        scores = []
        
        for i in range(batch_size):
            # Binarize predictions and ground truth via threshold
            pred = self._bin(y_pred[i, 0].detach().cpu().numpy()).astype(np.uint8)
            true = self._bin(y_true[i, 0].detach().cpu().numpy()).astype(np.uint8)
            
            # Skip empty ground truth masks
            if np.sum(true) == 0:
                if np.sum(pred) == 0:
                    scores.append(1.0)  # Both empty - perfect match
                else:
                    scores.append(0.0)  # True empty but pred not - no match
                continue
            
            # Find connected components
            true_labels = measure.label(true, connectivity=2)
            pred_labels = measure.label(pred, connectivity=2)
            
            true_props = measure.regionprops(true_labels)
            pred_props = measure.regionprops(pred_labels)
            
            # Filter out small components
            true_props = [prop for prop in true_props if prop.area >= self.min_size]
            pred_props = [prop for prop in pred_props if prop.area >= self.min_size]
            
            # Handle cases with no significant components
            if not true_props:
                if not pred_props:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
                continue
            if not pred_props:
                scores.append(0.0)
                continue
            
            # Match components
            matches = self._match_components(true_props, pred_props)
            
            # Detection score = TP / (TP + FP + FN)
            tp = len(matches)
            fp = max(0, len(pred_props) - tp)
            fn = max(0, len(true_props) - tp)
            detection_score = tp / (tp + fp + fn + self.eps)
            
            # Shape score = mean IoU of matched components
            shape_scores = []
            for true_idx, pred_idx in matches:
                true_mask = (true_labels == true_props[true_idx].label).astype(np.uint8)
                pred_mask = (pred_labels == pred_props[pred_idx].label).astype(np.uint8)
                intersection = np.sum(true_mask & pred_mask)
                union = np.sum(true_mask | pred_mask)
                iou = intersection / (union + self.eps)
                shape_scores.append(iou)
            
            shape_score = np.mean(shape_scores) if shape_scores else 0.0
            
            # Combined CCQ score
            combined_score = self.alpha * detection_score + (1 - self.alpha) * shape_score
            scores.append(combined_score)
        
        # Return mean score over batch
        return torch.tensor(sum(scores) / batch_size, device=y_pred.device)
    
    def _match_components(
        self,
        true_props: List[object],
        pred_props: List[object]
    ) -> List[Tuple[int, int]]:
        """
        Match predicted components to ground truth components.
        
        This uses a greedy approach based on centroid distance.
        
        Args:
            true_props: List of ground truth region properties
            pred_props: List of predicted region properties
        
        Returns:
            List of (true_idx, pred_idx) matches
        """
        matches = []
        used_pred: Set[int] = set()
        
        for true_idx, true_prop in enumerate(true_props):
            best_dist = float('inf')
            best_pred_idx = None
            true_centroid = true_prop.centroid
            
            for pred_idx, pred_prop in enumerate(pred_props):
                if pred_idx in used_pred:
                    continue
                pred_centroid = pred_prop.centroid
                
                # Calculate Euclidean distancebetween centroids
                dist = np.sqrt(
                    (true_centroid[0] - pred_centroid[0])**2 + 
                    (true_centroid[1] - pred_centroid[1])**2
                )
                if dist < best_dist and dist <= self.tolerance:
                    best_dist = dist
                    best_pred_idx = pred_idx
            
            if best_pred_idx is not None:
                matches.append((true_idx, best_pred_idx))
                used_pred.add(best_pred_idx)
        
        return matches


# ------------------------------------
# losses/chamfer_class.py
# ------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation


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
        diffs = gt_pts-p; dists = torch.norm(diffs,dim=1)
        md,idx = torch.min(dists,0)
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
# models/custom_model.py
# ------------------------------------
"""
Example custom model for segmentation.

This module demonstrates how to create a custom model for the seglab framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopoTokens(nn.Module):
    """
    TopoTokens: A custom segmentation model that incorporates topological features.
    
    This is a simple example for illustration - to be replaced with actual implementation.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_channels: int = 64,
        decoder_channels: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.2
    ):
        """
        Initialize the TopoTokens model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            encoder_channels: Base number of encoder channels
            decoder_channels: Base number of decoder channels
            num_blocks: Number of encoder/decoder blocks
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, encoder_channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(encoder_channels)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        current_channels = encoder_channels
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(current_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(current_channels * 2, current_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(current_channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            self.encoder_blocks.append(block)
            current_channels *= 2
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(current_channels * 2, current_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True),
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.ConvTranspose2d(current_channels, current_channels // 2, kernel_size=2, stride=2),
                nn.Conv2d(current_channels // 2, current_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(current_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(current_channels // 2, current_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(current_channels // 2),
                nn.ReLU(inplace=True)
            )
            self.decoder_blocks.append(block)
            current_channels //= 2
        
        # Final convolution
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TopoTokens model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Encoder path with skip connections
        skip_connections = []
        for block in self.encoder_blocks:
            skip_connections.append(x)
            x = block(x)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder path with skip connections
        for block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = block(x)
            # Add handling for size mismatches if needed
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip  # Skip connection
        
        # Final convolution
        x = self.final_conv(x)
        
        return torch.sigmoid(x)  # Apply sigmoid for binary segmentation

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
nn_GroupNorm = lambda three_dimensional: nn.BatchNorm3d if three_dimensional else nn.BatchNorm2d
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
                 dropout=0.3, batch_norm=True, pooling="max", three_dimensional=False):
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
        if batch_norm:
            # layers.append(nn_BatchNorm(_3d)(out_channels))
            layers.append(nn.GroupNorm(8, out_channels))
        layers.append(nn.ReLU(inplace=True))
        for i in range(n_convs-1):
            layers.append(nn_Conv(_3d)(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=BIAS))
            if batch_norm:
#                 layers.append(nn_BatchNorm(_3d)(out_channels))
                layers.append(nn.GroupNorm(8, out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn_Dropout(_3d)(p=dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

class UpBlock(nn.Module):

    def __init__(self, in_channels, n_convs=2, dropout=0.3,
                 batch_norm=True, upsampling='deconv', three_dimensional=False):
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
        elif upsampling in ['nearest', 'bilinear']:
            self.upsampling = nn.Sequential(
                                nn.Upsample(size=None, scale_factor=2, mode=upsampling),
                                nn_Conv(_3d)(in_channels, out_channels, kernel_size=1, padding=0, bias=BIAS))
        else:
            raise ValueError("Unrecognized upsampling option {}".fomrat(upsampling))

        layers = []
        layers.append(nn_Conv(_3d)(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=BIAS))
        if batch_norm:
#             layers.append(nn_BatchNorm(_3d)(out_channels))
            layers.append(nn.GroupNorm(8, out_channels))
        layers.append(nn.ReLU(inplace=True))
        for i in range(n_convs-1):
            layers.append(nn_Conv(_3d)(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=BIAS))
            if batch_norm:
#                 layers.append(nn_BatchNorm(_3d)(out_channels))
                layers.append(nn.GroupNorm(8, out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn_Dropout(_3d)(p=dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, input, to_stack):

        x = self.upsampling(input)

        x = torch.cat([to_stack, x], dim=1)

        return self.layers(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, m_channels=64, out_channels=2, n_convs=1,
                 n_levels=3, dropout=0.0, batch_norm=False, upsampling='bilinear',
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
        self.batch_norm = batch_norm
        self.upsampling = upsampling
        self.pooling = pooling
        self.three_dimensional = three_dimensional
        _3d = three_dimensional
        self.apply_final_relu = apply_final_relu

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

