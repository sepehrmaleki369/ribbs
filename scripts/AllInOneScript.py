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
            if (self.current_epoch+1) % freq == 0:
                # print('y_int.shape', y_hat.shape)
                # print('y_int.shape', y_hat.shape)
                # print(name, metric(y_hat, y_int))
                if name == "ccq":
                    ccq_value = metric(y_hat, y_int)
                    correctness = ccq_value[0].item() if isinstance(ccq_value, torch.Tensor) else ccq_value[0]
                    completeness = ccq_value[1].item() if isinstance(ccq_value, torch.Tensor) else ccq_value[1]
                    quality = ccq_value[2].item() if isinstance(ccq_value, torch.Tensor) else ccq_value[2]
                    self.log(f"train_metrics/{name}/correctness", correctness,
                             prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
                    self.log(f"train_metrics/{name}/completeness", completeness,
                             prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
                    self.log(f"train_metrics/{name}/quality", quality,
                             prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
                else:
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
                if name == "ccq":
                    ccq_value = metric(y_hat, y_int)
                    correctness = ccq_value[0].item() if isinstance(ccq_value, torch.Tensor) else ccq_value[0]
                    completeness = ccq_value[1].item() if isinstance(ccq_value, torch.Tensor) else ccq_value[1]
                    quality = ccq_value[2].item() if isinstance(ccq_value, torch.Tensor) else ccq_value[2]
                    self.log(f"val_metrics/{name}/correctness", correctness,
                             prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
                    self.log(f"val_metrics/{name}/completeness", completeness,
                             prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
                    self.log(f"val_metrics/{name}/quality", quality,
                             prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
                else:
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
            if name == "ccq":
                ccq_value = metric(y_hat, y_int)
                correctness = ccq_value[0].item() if isinstance(ccq_value, torch.Tensor) else ccq_value[0]
                completeness = ccq_value[1].item() if isinstance(ccq_value, torch.Tensor) else ccq_value[1]
                quality = ccq_value[2].item() if isinstance(ccq_value, torch.Tensor) else ccq_value[2]
                self.log(f"test_metrics/{name}/correctness", correctness,
                            prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
                self.log(f"test_metrics/{name}/completeness", completeness,
                            prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
                self.log(f"test_metrics/{name}/quality", quality,
                            prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
            else:
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
        divisible_by: int,
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
# core/general_dataset/augments.py
# ------------------------------------
# ------------------------------------
# augments.py  (complete version – supports 2‑D & 3‑D)
# ------------------------------------

from typing import Any, Dict, List, Optional
import math
import numpy as np
from scipy.ndimage import rotate, zoom as ndi_zoom, gaussian_filter, map_coordinates

from core.general_dataset.logger import logger

# -----------------------------------------------------------------------------
# Random‑parameter generator
# -----------------------------------------------------------------------------

def get_augmentation_metadata(
    augmentations: List[str],
    data_dim: int,
    aug_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Return sampled parameters based on `augmentation_params` from config."""
    params = aug_params or {}
    meta: Dict[str, Any] = {}

    # rotation
    if "rotation" in augmentations:
        rmin = params.get("rotation", {}).get("min", 0.0)
        rmax = params.get("rotation", {}).get("max", 360.0)
        meta["angle"] = float(np.random.uniform(rmin, rmax))

    # flips (no change)
    if "flip_h" in augmentations:
        meta["flip_h"] = bool(np.random.rand() > 0.5)
    if "flip_v" in augmentations:
        meta["flip_v"] = bool(np.random.rand() > 0.5)
    if "flip_d" in augmentations and data_dim == 3:
        meta["flip_d"] = bool(np.random.rand() > 0.5)

    # scale
    if "scale" in augmentations:
        smin = params.get("scale", {}).get("min", 0.8)
        smax = params.get("scale", {}).get("max", 1.2)
        meta["scale_factor"] = float(np.random.uniform(smin, smax))

    # elastic
    if "elastic" in augmentations:
        e = params.get("elastic", {})
        a_min, a_max = e.get("alpha_min",5.0), e.get("alpha_max",10.0)
        s_min, s_max = e.get("sigma_min",3.0), e.get("sigma_max",6.0)
        meta["alpha"] = float(np.random.uniform(a_min, a_max))
        meta["sigma"] = float(np.random.uniform(s_min, s_max))

    # brightness_contrast
    if "brightness_contrast" in augmentations:
        bc = params.get("brightness_contrast", {})
        ca, cb = bc.get("alpha_min",0.9), bc.get("alpha_max",1.1)
        ba, bb = bc.get("beta_min",-30.0), bc.get("beta_max",30.0)
        meta["bc_alpha"] = float(np.random.uniform(ca, cb))
        meta["bc_beta"]  = float(np.random.uniform(ba, bb))

    # gamma
    if "gamma" in augmentations:
        gmin = params.get("gamma", {}).get("min",0.7)
        gmax = params.get("gamma", {}).get("max",1.5)
        meta["gamma"] = float(np.random.uniform(gmin, gmax))

    # gaussian_noise
    if "gaussian_noise" in augmentations:
        n = params.get("gaussian_noise", {})
        meta["noise_sigma"] = float(np.random.uniform(n.get("min",0.01), n.get("max",0.03)))

    # gaussian_blur
    if "gaussian_blur" in augmentations:
        gb = params.get("gaussian_blur", {})
        meta["blur_sigma"] = float(np.random.uniform(gb.get("min",0.5), gb.get("max",1.5)))

    # bias_field
    if "bias_field" in augmentations:
        bf = params.get("bias_field", {})
        meta["bias_amp"] = float(np.random.uniform(bf.get("min",0.2), bf.get("max",0.4)))

    return meta

# -----------------------------------------------------------------------------
# Basic flip helpers (unchanged)
# -----------------------------------------------------------------------------

def flip_h(arr: np.ndarray) -> np.ndarray:
    """Flip left/right (axis = −1)."""
    return np.flip(arr, axis=-1)


def flip_v(arr: np.ndarray) -> np.ndarray:
    """Flip up/down (axis = −2)."""
    return np.flip(arr, axis=-2)


def flip_d(arr: np.ndarray) -> np.ndarray:
    """Flip depth (axis = −3). Only meaningful for 3‑D volumes."""
    return np.flip(arr, axis=-3)

# -----------------------------------------------------------------------------
# **DO NOT CHANGE** – original rotation implementation
# -----------------------------------------------------------------------------

def rotate_(
    full_array: np.ndarray,
    patch_meta: Dict[str, Any],
    patch_size_xy: int,
    patch_size_z: int,
    data_dim: int,
) -> np.ndarray:
    """Rotate patch, preserving shape, for 2‑D or 3‑D (slice‑wise)."""
    angle = patch_meta["angle"]
    L = int(np.ceil(patch_size_xy * math.sqrt(2)))
    x, y = patch_meta["x"], patch_meta["y"]
    cx, cy = x + patch_size_xy // 2, y + patch_size_xy // 2
    half_L = L // 2
    x0, x1 = max(0, cx - half_L), min(full_array.shape[-1], cx + half_L)
    y0, y1 = max(0, cy - half_L), min(full_array.shape[-2], cy + half_L)

    if data_dim == 3:
        z = patch_meta.get("z", 0)
        block = full_array[:, z : z + patch_size_z, y0:y1, x0:x1]
        if block.shape[-2] < L or block.shape[-1] < L:
            logger.warning("Crop too small for 3D rotation; zero patch.")
            return np.zeros_like(block[..., :patch_size_xy, :patch_size_xy])
        rotated_slices = []
        for d in range(block.shape[1]):
            slice_ = block[:, d]
            rotated = rotate(slice_, angle, reshape=False, order=1)
            start = (L - patch_size_xy) // 2
            cropped = rotated[start : start + patch_size_xy, start : start + patch_size_xy]
            rotated_slices.append(cropped)
        return np.stack(rotated_slices, axis=1)
    else:
        crop = (
            full_array[y0:y1, x0:x1]
            if full_array.ndim == 2
            else full_array[:, y0:y1, x0:x1]
        )
        if crop.shape[-2] < L or crop.shape[-1] < L:
            logger.warning("Crop too small for 2D rotation; zero patch.")
            shp = (patch_size_xy, patch_size_xy) if crop.ndim == 2 else (crop.shape[0], patch_size_xy, patch_size_xy)
            return np.zeros(shp, dtype=crop.dtype)
        rotated = rotate(crop, angle, reshape=False, order=1)
        start = (L - patch_size_xy) // 2
        if rotated.ndim == 2:
            return rotated[start : start + patch_size_xy, start : start + patch_size_xy]
        else:
            return rotated[:, start : start + patch_size_xy, start : start + patch_size_xy]

# -----------------------------------------------------------------------------
# New spatial helpers
# -----------------------------------------------------------------------------

def _center_crop_or_pad(arr: np.ndarray, out_shape: tuple) -> np.ndarray:
    """Return `arr` centered and cropped/padded to `out_shape`. Zeros are used for padding."""
    result = np.zeros(out_shape, dtype=arr.dtype)
    min_shape = tuple(min(s, o) for s, o in zip(arr.shape, out_shape))
    # slice positions in input and output
    in_starts = [(s - m) // 2 for s, m in zip(arr.shape, min_shape)]
    out_starts = [(o - m) // 2 for o, m in zip(out_shape, min_shape)]
    slices_in = tuple(slice(i, i + m) for i, m in zip(in_starts, min_shape))
    slices_out = tuple(slice(o, o + m) for o, m in zip(out_starts, min_shape))
    result[slices_out] = arr[slices_in]
    return result


def scale_patch(patch: np.ndarray, scale_factor: float, data_dim: int) -> np.ndarray:
    """Isotropic zoom of the *patch* followed by center‑crop/zero‑pad to original size."""
    orig_shape = patch.shape

    if data_dim == 3:
        zoom_factors = (1, scale_factor, scale_factor, scale_factor) if patch.ndim == 4 else (scale_factor, scale_factor, scale_factor)
    else:
        zoom_factors = (1, scale_factor, scale_factor) if patch.ndim == 3 else (scale_factor, scale_factor)

    scaled = ndi_zoom(patch, zoom_factors, order=1)
    return _center_crop_or_pad(scaled, orig_shape)


# -----------------------------------------------------------------------------
# Elastic deformation  (B‑spline‑like random field)
# -----------------------------------------------------------------------------

def elastic_deform_patch(patch: np.ndarray, alpha: float, sigma: float, data_dim: int) -> np.ndarray:
    """Apply elastic deformation to `patch` and return the deformed patch (same shape)."""

    if data_dim == 3:
        spatial_shape = patch.shape[-3:] if patch.ndim == 4 else patch.shape
        dz = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect") * alpha
        dy = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect") * alpha
        dx = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect") * alpha
        z, y, x = np.meshgrid(
            np.arange(spatial_shape[0]),
            np.arange(spatial_shape[1]),
            np.arange(spatial_shape[2]),
            indexing="ij",
        )
        indices = (z + dz, y + dy, x + dx)

        if patch.ndim == 4:  # C,Z,H,W
            deformed = [map_coordinates(patch[c], indices, order=1, mode="reflect") for c in range(patch.shape[0])]
            return np.stack(deformed, axis=0)
        else:
            return map_coordinates(patch, indices, order=1, mode="reflect")

    else:  # 2‑D
        spatial_shape = patch.shape[-2:] if patch.ndim == 3 else patch.shape
        dy = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect") * alpha
        dx = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect") * alpha
        y, x = np.meshgrid(
            np.arange(spatial_shape[0]),
            np.arange(spatial_shape[1]),
            indexing="ij",
        )
        indices = (y + dy, x + dx)

        if patch.ndim == 3:  # C,H,W
            deformed = [map_coordinates(patch[c], indices, order=1, mode="reflect") for c in range(patch.shape[0])]
            return np.stack(deformed, axis=0)
        else:
            return map_coordinates(patch, indices, order=1, mode="reflect")

# -----------------------------------------------------------------------------
# Intensity helpers
# -----------------------------------------------------------------------------

def adjust_brightness_contrast(patch: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Linear brightness/contrast: I' = alpha·I + beta"""
    return patch.astype(np.float32) * alpha + beta


def gamma_correction(patch: np.ndarray, gamma: float) -> np.ndarray:
    patch_f = patch.astype(np.float32)
    max_val = patch_f.max() if patch_f.max() > 0 else 1.0
    norm = np.clip(patch_f / max_val, 0.0, 1.0)
    return (norm ** gamma) * max_val


def add_gaussian_noise(patch: np.ndarray, sigma_factor: float) -> np.ndarray:
    patch_f = patch.astype(np.float32)
    dyn = patch_f.max() - patch_f.min()
    sigma = sigma_factor * (dyn if dyn > 0 else 1.0)
    noise = np.random.normal(0.0, sigma, patch_f.shape)
    return patch_f + noise


def gaussian_blur_patch(patch: np.ndarray, sigma: float, data_dim: int) -> np.ndarray:
    """Apply an isotropic Gaussian blur (σ in voxels). Preserves shape."""
    if data_dim == 3:
        if patch.ndim == 4:  # C,Z,H,W
            blurred = [gaussian_filter(patch[c], sigma, mode="reflect") for c in range(patch.shape[0])]
            return np.stack(blurred, axis=0)
        else:
            return gaussian_filter(patch, sigma, mode="reflect")
    else:
        if patch.ndim == 3:  # C,H,W
            blurred = [gaussian_filter(patch[c], sigma, mode="reflect") for c in range(patch.shape[0])]
            return np.stack(blurred, axis=0)
        else:
            return gaussian_filter(patch, sigma, mode="reflect")


def bias_field_patch(patch: np.ndarray, amplitude: float, data_dim: int) -> np.ndarray:
    """Multiply `patch` by a smooth, low‑frequency bias field."""
    if data_dim == 3:
        spatial_shape = patch.shape[-3:] if patch.ndim == 4 else patch.shape
    else:
        spatial_shape = patch.shape[-2:] if patch.ndim == 3 else patch.shape

    sigma = 0.25 * min(spatial_shape)  # fairly low‑freq
    field = gaussian_filter(np.random.randn(*spatial_shape), sigma, mode="reflect")
    field -= field.mean()
    field /= np.max(np.abs(field)) + 1e-8  # → ∈ [‑1,1]
    field *= amplitude
    field = 1.0 + field  # multiplicative  (1±amp)

    if patch.ndim == len(spatial_shape):  # no channel dim
        return patch * field
    else:  # prepend channel axis broadcast
        return patch * field[np.newaxis, ...]

# -----------------------------------------------------------------------------
# Patch extraction + conditional augmentation
# -----------------------------------------------------------------------------

def extract_data(
    imgs: Dict[str, np.ndarray],
    x: int,
    y: int,
    z: int,
    patch_size_xy: int,
    patch_size_z: Optional[int],
    data_dim: int,
) -> Dict[str, np.ndarray]:
    """
    Slice a spatial patch from every modality in *imgs*.

    Parameters
    ----------
    imgs : Dict[str, np.ndarray]
        Each value is one of the following shapes
          ─ data_dim == 2 →  (H, W)  or  (C, H, W)
          ─ data_dim == 3 →  (D, H, W)  or  (C, D, H, W)
    x, y : int
        Upper-left corner of the XY window.
    z : int
        First slice index along the depth axis (ignored for 2-D).
    patch_size_xy : int
        Height and width of the square crop.
    patch_size_z : Optional[int]
        If given in 3-D mode, take this many slices; otherwise fall back to
        *patch_size_xy*.
    data_dim : int
        2 or 3 – tells the function how to interpret array ranks.

    Returns
    -------
    Dict[str, np.ndarray]
        For every key in *imgs* a new entry ``f"{key}_patch"`` containing the
        cropped patch (shape preserved except for the spatial dims).
    """
    data: Dict[str, np.ndarray] = {}
    psz_z = patch_size_z or patch_size_xy

    for key, arr in imgs.items():

        # ───────────────────────────── 3-D CASE ───────────────────────────── #
        if data_dim == 3:
            if arr.ndim == 4:                             # (C, D, H, W)
                patch = arr[:, z : z + psz_z,
                               y : y + patch_size_xy,
                               x : x + patch_size_xy]
            elif arr.ndim == 3:                           # (D, H, W)
                patch = arr[z : z + psz_z,
                             y : y + patch_size_xy,
                             x : x + patch_size_xy]
            else:
                raise ValueError(f"{key}: expected 3-D or 4-D array, got shape {arr.shape}")

        # ───────────────────────────── 2-D CASE ───────────────────────────── #
        elif data_dim == 2:
            if arr.ndim == 3:                             # (C, H, W)
                patch = arr[:, y : y + patch_size_xy,
                               x : x + patch_size_xy]
            elif arr.ndim == 2:                           # (H, W)
                patch = arr[y : y + patch_size_xy,
                             x : x + patch_size_xy]
            else:
                raise ValueError(f"{key}: expected 2-D or 3-D array, got shape {arr.shape}")

        else:
            raise ValueError(f"data_dim must be 2 or 3, got {data_dim}")

        data[f"{key}_patch"] = patch

    return data



# -----------------------------------------------------------------------------
# Main entry: extract + apply *conditional* augmentations
# -----------------------------------------------------------------------------

def extract_condition_augmentations(
    imgs: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    patch_size_xy: int,
    patch_size_z: int,
    augmentations: List[str],
    data_dim: int
) -> Dict[str, np.ndarray]:
    """
    Extract a patch from the full image and apply conditional augmentations.

    Args:
        imgs (Dict[str, np.ndarray]): Full images for each modality.
        metadata (Dict[str, Any]): Metadata containing patch coordinates and augmentations.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary of extracted patches.
    """
    imgs_aug = imgs.copy()
    z = metadata.get('z', 0)
    data = extract_data(imgs,
                        metadata['x'], metadata['y'], z,
                        patch_size_xy,
                        patch_size_z,
                        data_dim)
    for key in imgs:
        if key.endswith("_patch"):
            modality = key.replace("_patch", "")
            if 'flip_h' in augmentations:
                imgs_aug[modality] = flip_h(imgs[modality])
                data[key] = flip_h(data[key])
            if 'flip_v' in augmentations:
                imgs_aug[modality] = flip_v(imgs[modality])
                data[key] = flip_v(data[key])
            if 'flip_d' in augmentations and data_dim == 3:
                imgs_aug[modality] = flip_d(imgs_aug[modality])
                data[key]          = flip_d(data[key])
#######
            if "scale" in augmentations:
                sf: float = metadata.get("scale_factor")
                imgs_aug[modality] = scale_patch(imgs_aug[modality], sf, data_dim)
                patch = scale_patch(patch, sf, data_dim)

            if "elastic" in augmentations:
                alpha: float = metadata.get("alpha")
                sigma: float = metadata.get("sigma")
                imgs_aug[modality] = elastic_deform_patch(imgs_aug[modality], alpha, sigma, data_dim)
                patch = elastic_deform_patch(patch, alpha, sigma, data_dim)

            # -----------------------------
            # Intensity augmentations (patch‑only)
            # -----------------------------
            if "brightness_contrast" in augmentations:
                patch = adjust_brightness_contrast(
                    patch,
                    metadata.get("bc_alpha"),
                    metadata.get("bc_beta"),
                )
            if "gamma" in augmentations:
                patch = gamma_correction(patch, metadata.get("gamma"))
            if "gaussian_noise" in augmentations:
                patch = add_gaussian_noise(patch, metadata.get("noise_sigma"))
            if "gaussian_blur" in augmentations:
                patch = gaussian_blur_patch(patch, metadata.get("blur_sigma"), data_dim)
            if "bias_field" in augmentations:
                patch = bias_field_patch(patch, metadata.get("bias_amp"), data_dim)

            if 'rotation' in augmentations:
                data[key] = rotate_(imgs_aug[modality], metadata, patch_size_xy, patch_size_z, data_dim)

            # Store the fully augmented patch back
            data[key] = patch    
    
    return data


# -----------------------------------------------------------------------------
# Public re‑exports
# -----------------------------------------------------------------------------

__all__ = [
    "get_augmentation_metadata",
    "extract_condition_augmentations",
    "flip_h",
    "flip_v",
    "flip_d",
    "rotate_",  # unchanged
]

# "augmentations": [
#         "flip_h",              # horizontal flip
#         "flip_v",              # vertical flip
#         "flip_d",              # depth flip (3D only)
#         "rotation",            # 0–360° random rotation
#         "scale",               # isotropic zoom in/out
#         "elastic",             # B-spline–style elastic warp
#         "brightness_contrast", # random α·I + β
#         "gamma",               # power-law I^γ
#         "gaussian_noise",      # additive Gaussian noise
#         "gaussian_blur",       # isotropic blur
#         "bias_field",          # low-frequency multiplicative bias field
#     ],


# ------------------------------------
# core/general_dataset/logger.py
# ------------------------------------
import logging
import sys

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.ERROR)

fmt = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(fmt)

# (re)attach handler
if not logger.handlers:
    logger.addHandler(handler)

# ------------------------------------
# core/general_dataset/collate.py
# ------------------------------------
import torch
from typing import Any, Dict, List, Optional
import random
import numpy as np
from core.general_dataset.logger import logger


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function with None filtering.
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]
    
    # Handle empty batch case
    if not batch:
        logger.warning("Empty batch after filtering None values")
        return {}  # Or return a default empty batch structure
    
    # Original collation logic
    collated: Dict[str, Any] = {}
    for key in batch[0]:
        items = []
        for sample in batch:
            value = sample[key]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            items.append(value)
        if isinstance(items[0], torch.Tensor):
            collated[key] = torch.stack(items)
        else:
            collated[key] = items
    return collated

def worker_init_fn(worker_id):
    """
    DataLoader worker initialization to ensure different random seeds for each worker.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

# ------------------------------------
# core/general_dataset/modalities.py
# ------------------------------------
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from typing import Any, Dict, List, Optional
from core.general_dataset.logger import logger




def compute_distance_map(lbl: np.ndarray, distance_threshold: Optional[float]) -> np.ndarray:
    """
    Compute a distance map from a label image.

    Args:
        lbl (np.ndarray): Input label image.
        distance_threshold (Optional[float]): Maximum distance value.
    
    Returns:
        np.ndarray: Distance map.
    """
    lbl_bin = (lbl > 127).astype(np.uint8) if lbl.max() > 1 else (lbl > 0).astype(np.uint8)
    distance_map = distance_transform_edt(lbl_bin == 0)
    if distance_threshold is not None:
        np.minimum(distance_map, distance_threshold, out=distance_map)
    return distance_map

def compute_sdf(lbl: np.ndarray, sdf_iterations: int, sdf_thresholds: List[float]) -> np.ndarray:
    """
    Compute the signed distance function (SDF) for a label image.

    Args:
        lbl (np.ndarray): Input label image.
        sdf_iterations (int): Number of iterations for dilation.
        sdf_thresholds (List[float]): [min, max] thresholds for the SDF.
    
    Returns:
        np.ndarray: The SDF computed.
    """
    lbl_bin = (lbl > 127).astype(np.uint8) if lbl.max() > 1 else (lbl > 0).astype(np.uint8)
    dilated = binary_dilation(lbl_bin, iterations=sdf_iterations)
    dist_out = distance_transform_edt(1 - dilated)
    dist_in  = distance_transform_edt(lbl_bin)
    sdf = dist_out - dist_in
    if sdf_thresholds is not None:
        sdf = np.clip(sdf, sdf_thresholds[0], sdf_thresholds[1])
    return sdf

# ------------------------------------
# core/general_dataset/base.py
# ------------------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import os
import json
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

from core.general_dataset.io          import load_array_from_file
from core.general_dataset.modalities  import compute_distance_map, compute_sdf
from core.general_dataset.patch_validity       import check_min_thrsh_road
from core.general_dataset.augments    import get_augmentation_metadata, extract_condition_augmentations
from core.general_dataset.collate     import custom_collate_fn, worker_init_fn
from core.general_dataset.normalizations import normalize_image
from core.general_dataset.visualizations import visualize_batch_2d, visualize_batch_3d
from core.general_dataset.splits import Split
from core.general_dataset.logger import logger


class GeneralizedDataset(Dataset):
    """
    PyTorch Dataset for generalized remote sensing or segmentation datasets.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.split: str = config.get("split", "train")
        self.patch_size: int = config.get("patch_size", 128)
        self.small_window_size: int = config.get("small_window_size", 8)
        self.threshold: float = config.get("threshold", 0.05)
        self.max_images: Optional[int] = config.get("max_images")
        self.max_attempts: int = config.get("max_attempts", 10)
        self.validate_road_ratio: bool = config.get("validate_road_ratio", False)
        self.seed: int = config.get("seed", 42)
        self.fold = config.get("fold")
        self.num_folds = config.get("num_folds")
        self.verbose: bool = config.get("verbose", False)
        self.augmentations: List[str] = config.get("augmentations", ["flip_h", "flip_v", "rotation"])
        self.augmentation_params: Dict[str, Any] = config.get("augmentation_params", {})
        self.distance_threshold: Optional[float] = config.get("distance_threshold")
        self.sdf_iterations: int = config.get("sdf_iterations")
        self.sdf_thresholds: List[float] = config.get("sdf_thresholds")
        self.num_workers: int = config.get("num_workers", 4)
        self.split_ratios: Dict[str, float] = config.get("split_ratios", {"train":0.7,"valid":0.15,"test":0.15})
        self.source_folder: str = config.get("source_folder", "")
        self.save_computed: bool = config.get("save_computed", False)
        self.base_modalities = config.get('base_modalities')
        self.compute_again_modalities = config.get('compute_again_modalities', False)
        self.data_dim    = config.get("data_dim", 2)
        self.split_cfg = config["split_cfg"]
        self.split_cfg['seed'] = self.seed
        self.norm_cfg: Dict[str, Optional[Dict[str, Any]]] = config.get("normalization", {})
        
        if self.patch_size is None:
            raise ValueError("patch_size must be specified in the config.")
        if self.data_dim not in (2, 3):
            raise ValueError(f"data_dim must be 2 or 3, got {self.data_dim}")
        if self.data_dim == 3 and config.get("patch_size_z", 1) < 2:
            raise ValueError("patch_size_z must > 1 for 3D")

        random.seed(self.seed)
        np.random.seed(self.seed)

        splitter = Split(self.split_cfg, self.base_modalities)
        self.modality_files: Dict[str, List[str]] = splitter.get_split(self.split)
        self.modalities = list(self.modality_files.keys())
        if 'image' not in self.modality_files or 'label' not in self.modality_files:
            raise ValueError("Split must define both 'image' and 'label' modalities in split_cfg.")
        assert len(self.modality_files['image']) == len(self.modality_files['label']), (
            f"len(images): {len(self.modality_files['image'])}, len(labels): {len(self.modality_files['label'])}")

        if self.max_images is not None:
            for key in self.modality_files:
                self.modality_files[key] = self.modality_files[key][:self.max_images]
                # print(key, 'Max Data Point:', len(self.modality_files[key]))
        # Precompute additional modalities if requested
        if self.save_computed:
            for key in [m for m in self.modalities if m not in ['image', 'label']]:
                logger.info(f"Generating {key} modality maps...")
                for file_idx, _ in tqdm(list(enumerate(self.modality_files['label'])),
                                        total=len(self.modality_files['label']),
                                        desc=f"Processing {key} maps"):
                    lbl = load_array_from_file(self.modality_files['label'][file_idx])
                    modality_path = self.modality_files[key][file_idx]
                    os.makedirs(os.path.dirname(modality_path), exist_ok=True)
                    if key == 'distance':
                        if not os.path.exists(modality_path) or self.compute_again_modalities:
                            processed = compute_distance_map(lbl, None)
                            np.save(modality_path, processed)
                    elif key == 'sdf':
                        if not os.path.exists(modality_path) or self.compute_again_modalities:
                            processed = compute_sdf(lbl, self.sdf_iterations, None)
                            np.save(modality_path, processed)
                    else:
                        raise ValueError(f"Unsupported modality {key}")

    def _load_datapoint(self, file_idx: int) -> Optional[Dict[str, np.ndarray]]:
        imgs: Dict[str, np.ndarray] = {}
        for key in self.modalities:
            path = self.modality_files[key][file_idx]
            if os.path.exists(path):
                arr = load_array_from_file(path)
                if arr is None:
                    return None
            else:
                lbl = load_array_from_file(self.modality_files['label'][file_idx])
                if key == 'distance':
                    arr = compute_distance_map(lbl, None)
                elif key == 'sdf':
                    arr = compute_sdf(lbl, self.sdf_iterations, None)
                else:
                    raise ValueError(f"Unsupported modality {key}")
            imgs[key] = arr
        # if self.data_dim == 3:
        #     for k, v in imgs.items():
        #         if v.ndim == 3:                # (D, H, W)  →  (1, D, H, W)
        #             imgs[k] = v[np.newaxis, ...].astype(np.float32)
        if self.verbose:
            print('imgs[key] shape:', imgs[key].shape)
        return imgs

    def _postprocess_patch(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Binarize, clip, normalize per modality
        for key, arr in list(data.items()):
            if not key.endswith('_patch'):
                continue
            modality = key[:-6]
            patch = arr.astype(np.float32)
            if modality == 'label':
                thresh = 127 if patch.max() > 1 else 0
                data[key] = (patch > thresh).astype(np.uint8)
            elif modality == 'distance' and self.distance_threshold is not None:
                data[key] = np.clip(patch, 0, self.distance_threshold)
            elif modality == 'sdf' and self.sdf_thresholds is not None:
                lo, hi = self.sdf_thresholds
                data[key] = np.clip(patch, lo, hi)
            else:
                data[key] = patch
        # Normalization
        for key, arr in list(data.items()):
            if not key.endswith('_patch'):
                continue
            modality = key[:-6]
            cfg = self.norm_cfg.get(modality)
            if isinstance(cfg, dict):
                method = cfg.get('method', 'minmax')
                params = {k: v for k, v in cfg.items() if k != 'method'}
                data[key] = normalize_image(arr, method=method, **params)

        for key, arr in list(data.items()):
            if not key.endswith("_patch") or not isinstance(arr, np.ndarray):
                continue
            # 2-D image  -> (1, H, W)
            if arr.ndim == 2:
                data[key] = arr[np.newaxis, ...]
            # 3-D volume -> (1, D, H, W)
            elif arr.ndim == 3 and self.data_dim == 3:
                data[key] = arr[np.newaxis, ...]
        return data

    def __len__(self) -> int:
        return len(self.modality_files['image'])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        imgs = self._load_datapoint(idx)
        if imgs is None:
            return self.__getitem__((idx + 1) % len(self))

        img = imgs['image']
        # Determine shape for slicing
        if self.data_dim == 3:
            if img.ndim == 4:
                _, D, H, W = img.shape
            elif img.ndim == 3:
                D, H, W = img.shape
            else:
                raise ValueError(f"Expected 3D data for data_dim=3; got {img.ndim}D")
        else:
            if img.ndim == 2:
                C, H, W = 1, *img.shape
            elif img.ndim == 3:
                C, H, W = img.shape
            else:
                raise ValueError(f"Expected 2D or 3-channel image for data_dim=2; got {img.ndim}D")

        # Validation/test: return full image as one patch
        if self.split != 'train':
            data = {}
            for key, array in imgs.items():
                data[f"{key}_patch"] = array
            data['metadata'] = {"image_idx": idx, "x": -1, "y": -1}
            return self._postprocess_patch(data)

        # Training: random crop
        attempts = 0
        while attempts < self.max_attempts:
            x = random.randint(0, W - self.patch_size)
            y = random.randint(0, H - self.patch_size)
            patch_meta: Dict[str, Any] = {"image_idx": idx, "x": x, "y": y}
            if self.augmentations:
                patch_meta.update(get_augmentation_metadata(self.augmentations, self.data_dim, self.augmentation_params))
            # Extract patch
            if self.data_dim == 2:
                data: Dict[str, np.ndarray] = {}
                for key, array in imgs.items():
                    if array.ndim == 2:
                        patch = array[y:y+self.patch_size, x:x+self.patch_size]
                    else:
                        patch = array[:, y:y+self.patch_size, x:x+self.patch_size]
                    data[f"{key}_patch"] = patch
                data['metadata'] = patch_meta
                data = self._postprocess_patch(data)
            else:
                data = extract_condition_augmentations(
                    imgs, patch_meta,
                    patch_size_xy=self.patch_size,
                    patch_size_z=self.config.get('patch_size_z', 1),
                    augmentations=self.augmentations,
                    data_dim=self.data_dim
                )
                data['metadata'] = patch_meta
                data = self._postprocess_patch(data)

            # Validate road ratio if needed
            if not self.validate_road_ratio or check_min_thrsh_road(data['label_patch'], self.patch_size, self.threshold):
                return data
            attempts += 1

        logger.warning("No valid patch found after %d attempts on image %d; skipping.", self.max_attempts, idx)
        return self.__getitem__((idx + 1) % len(self))

if __name__ == "__main__":
    split_cfg = {
        "seed": 42,
        "sources": [
            {
                "type": "folder",
                "path": "/home/ri/Desktop/Projects/Datasets/RRRR/dataset",
                "layout": "folders",
                "modalities": {
                    "image":    {"folder": "sat"},
                    "label":    {"folder": "label"},
                    "distance": {"folder": "distance"},
                    "sdf":      {"folder": "sdf"},
                },
                "splits": {
                    "train": "train",
                    "valid": "valid",
                    "test":  "test",
                }
            }
        ]
    }

    # ------------------------------------
    # full config for GeneralizedDataset
    # ------------------------------------
    config = {
        # which split to load
        "split": "train",

        # pass the splitter cfg here
        "split_cfg": split_cfg,

        # patch extraction params
        "patch_size": 256,
        "patch_size_z": 1,             # keep at 1 for 2D data

        # optional small‐window check
        "small_window_size": 8,

        # require a minimum fraction of road pixels in each patch
        "validate_road_ratio": True,
        "threshold": 0.025,

        # if you want to limit to just N images (for debugging), set here
        "max_images": 5,

        # random seeds & workers
        "seed": 42,
        "num_workers": 4,
        "verbose": True,

        # train‐time augmentations
        "augmentations": ["flip_h", "flip_v", "rotation"],

        # distance & SDF modalities: compute if missing
        "distance_threshold": 15.0,
        "sdf_iterations": 3,
        "sdf_thresholds": [-7, 7],
        "compute_again_modalities": False,
        "save_computed": True,

        # only used if you ever switch to ratio‐based splitting
        "split_ratios": {"train": 0.7, "valid": 0.15, "test": 0.15},

        # these are the “base” modalities that must all be present
        "base_modalities": ["image", "label"],

        # 2D vs 3D
        "data_dim": 2,

        "normalization": {
            "image":    {"method":  "minmax",
                    "old_min": 0,     # your supplied min
                    "old_max": 255.0,   # your supplied max
                    "new_min": 0.0,
                    "new_max": 1.0},

            # "distance": {"method": "zscore"},
            "sdf":      {},            # empty → no normalization
            "label":    None,          # or null in JSON → skip
        },
        "augmentation_params": {
            "scale":             {"min": 1.5,  "max": 5.8},
            # "elastic":           {"alpha_min": 5.0,  "alpha_max": 10.0,
                                # "sigma_min": 3.0,  "sigma_max": 6.0},
            "brightness_contrast":{"alpha_min": 0.9,"alpha_max": 1.1,
                                "beta_min": -30.0,"beta_max": 30.0},
            "gamma":             {"min": 0.7,  "max": 1.5},
            # "gaussian_noise":    {"min": 0.01, "max": 0.03},
            # "gaussian_blur":     {"min": 0.5,  "max": 1.5},
            # "bias_field":        {"min": 0.2,  "max": 0.4},
            # "rotation":          {"min": 0.0,  "max": 360.0},
            # flips and flip_d remain Bernoulli(0.5), no extra params needed
        },
    }

    # split_cfg = {
    #     "seed": 42,
    #     "sources": [
    #         {
    #             "type": "ratio",
    #             "path": "/home/ri/Desktop/Projects/Datasets/AL175",
    #             "layout": "flat",
    #             "modalities": {
    #                 "image":    {"pattern": r"^cube_(.*)\.npy$"},
    #                 "label":    {"pattern": r"^lbl_(.*)\.npy$"},
    #                 "distance": {"pattern": r"^distlbl_(.*)\.npy$"},
    #             },
    #             "ratios": {
    #                 "train": 0.7,
    #                 "valid": 0.15,
    #                 "test":  0.15,
    #             }
    #         },
    #         {
    #             "path": "/home/ri/Desktop/Projects/Datasets/AL175",
    #             "layout": "flat",
    #             "modalities": {
    #                 "image":    {"pattern": r"^cube_(.*)\.npy$"},
    #                 "label":    {"pattern": r"^lbl_(.*)\.npy$"},
    #                 "distance": {"pattern": r"^distlbl_(.*)\.npy$"},
    #             },

    #             "type": "kfold",
    #             "num_folds": 5,
    #             "fold_idx": 0,
    #             "test_source":{
    #                 "type": "ratio",
    #                 "path": "/home/ri/Desktop/Projects/Datasets/AL175",
    #                 "layout": "flat",
    #                 "modalities": {
    #                     "image":    {"pattern": r"^cube_(.*)\.npy$"},
    #                     "label":    {"pattern": r"^lbl_(.*)\.npy$"},
    #                     "distance": {"pattern": r"^distlbl_(.*)\.npy$"},
    #                 },
    #                 "ratios": {
    #                     "train": 0.01,
    #                     "valid": 0.9,
    #                     "test":  0.01,
    #                 }
    #             }
    #         }
    #     ]
    # }

    # config = {
    #     "split": "train",                # one of "train","valid","test"
    #     "split_cfg": split_cfg,

    #     "data_dim": 3,                   # your .npy volumes are 3D
    #     "patch_size": 90,                # XY window size
    #     "patch_size_z": 90,              # Z-depth

    #     "augmentations": ["flip_h","flip_v","flip_d","rotation"],
    #     "validate_road_ratio": False,    # set True if you want to enforce label coverage
    #     "threshold": 0.05,

    #     "distance_threshold": None,      # clip distance map if desired
    #     "sdf_iterations": 3,             # only if you compute SDF
    #     "sdf_thresholds": [-5, 5],       # ditto

    #     "save_computed": True,          # we already have distlbl_*.npy
    #     "compute_again_modalities": False,

    #     "max_images": None,
    #     "max_attempts": 10,
    #     "seed": 42,
    #     "num_workers": 4,
    #     "verbose": True,
    #     "base_modalities": ['image', 'label']
    # }

    import yaml
    # with open('/home/ri/Desktop/Projects/Codebase/configs/dataset/main.yaml', 'w') as f_out:
        # yaml.dump(config, f_out)
    with open('./configs/dataset/AL175_15.yaml', 'r') as f:
    # with open('/home/ri/Desktop/Projects/Codebase/configs/dataset/massroads.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create dataset and dataloader.
    dataset = GeneralizedDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        # worker_init_fn=worker_init_fn
    )
    logger.info('len(dataloader): %d', len(dataloader))
    for epoch in range(10):
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue
            logger.info("Batch keys: %s", batch.keys())
            logger.info("Image shape: %s", batch["image_patch"].shape)
            logger.info("Label shape: %s", batch["label_patch"].shape)
            if config["data_dim"] == 2:
                visualize_batch_2d(batch, num_per_batch=2)
            else:
                visualize_batch_3d(batch, projection='max', num_per_batch=2)

            # break  # Uncomment to visualize only one batch.


# ------------------------------------
# core/general_dataset/splits.py
# ------------------------------------
import os
import re
import random
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.model_selection import KFold

def _is_junk(fname: str) -> bool:
    """Ignore index.html or config.json files (case insensitive)."""
    return fname.lower() in ("index.html", "config.json")


def _listdir_safe(folder: str) -> List[str]:
    """Safely list directory contents; return empty list if path doesn't exist."""
    try:
        return os.listdir(folder)
    except Exception:
        return []

def filename_from_pattern(pattern: str, stem: str) -> str:
    # 1) strip regex anchors
    if pattern.startswith("^"):
        pattern = pattern[1:]
    if pattern.endswith("$"):
        pattern = pattern[:-1]

    # 2) split on the literal "(.*)"
    parts = pattern.split("(.*)")
    if len(parts) != 2:
        raise ValueError(f"Pattern must contain exactly one '(.*)' slot: {pattern!r}")
    prefix, suffix = parts

    # 3) un-escape any escaped chars (e.g. "\." → ".", "\(" → "(", etc.)
    unescape = lambda s: re.sub(r"\\(.)", r"\1", s)
    prefix = unescape(prefix)
    suffix = unescape(suffix)

    # 4) re-join
    return f"{prefix}{stem}{suffix}"

def _filter_complete(records, required_modalities):
    by_group = defaultdict(list)
    for rec in records:
        key = (rec["split"], rec["stem"])
        by_group[key].append(rec)

    complete = []
    for (split, stem), recs in by_group.items():
        mods = {r["modality"] for r in recs}
        if mods >= set(required_modalities):
            complete.extend(recs)
    return complete

def _filter_complete_no_split(records: List[Dict[str, str]], required_modalities: List[str]) -> List[Dict[str, str]]:
    """
    Given a flat list of records without split information,
    drop any stems missing one of the required_modalities.
    """
    # Group records by stem
    by_stem: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for rec in records:
        stem = rec["stem"]
        by_stem[stem].append(rec)

    # Keep only those groups whose modalities cover the required set
    req_set = set(required_modalities)
    complete: List[Dict[str, str]] = []
    for stem, recs in by_stem.items():
        mods = {r["modality"] for r in recs}
        if mods >= req_set:
            complete.extend(recs)

    return complete

def _pivot_views(records):
    st_md_sp = defaultdict(lambda: defaultdict(dict))
    md_st_sp = defaultdict(lambda: defaultdict(dict))
    st_sp_md = defaultdict(lambda: defaultdict(dict))
    sp_st_md = defaultdict(lambda: defaultdict(dict))

    for r in records:
        s, m, t, p = r["split"], r["modality"], r["stem"], r["path"]
        st_md_sp[t][m][s] = p
        md_st_sp[m][t][s] = p
        st_sp_md[t][s][m] = p
        sp_st_md[s][t][m] = p

    return [st_md_sp, md_st_sp, st_sp_md, sp_st_md]

def _pivot_views_no_split(records):
    st_md = defaultdict(lambda: defaultdict(dict))
    md_st = defaultdict(lambda: defaultdict(dict))

    for r in records:
        m, t, p = r["modality"], r["stem"], r["path"]
        st_md[t][m] = p
        md_st[m][t] = p

    return [st_md, md_st]

def _collect_datapoints_from_source(src: Dict[str, Any], base_modalities, rng):
    root       = src["path"]
    layout     = src.get("layout", "flat")
    modalities = src.get("modalities", {})
    splits     = src.get("splits", {})

    base_records = []
    if layout == "folders":
        for split, subfolder in splits.items():
            split_dir = os.path.join(root, subfolder)
            for mod, meta in modalities.items():
                if mod not in base_modalities:
                    continue
                subfolder = meta.get("folder")
                if not subfolder:
                    continue
                folder = os.path.join(split_dir, subfolder)
                for fname in _listdir_safe(folder):
                    if _is_junk(fname):
                        continue
                    stem = os.path.splitext(fname)[0]
                    path = os.path.join(folder, fname)
                    base_records.append({
                        "split":    split,
                        "modality": mod,
                        "stem":     stem,
                        "path":     path,
                    })

        # 2) drop any (split,stem) groups missing a base modality
        base_records = _filter_complete(base_records, base_modalities)
        # extract all the valid stems per split
        stems_by_split = defaultdict(set)
        for rec in base_records:
            stems_by_split[rec["split"]].add(rec["stem"])

        # 3) start your final list with the complete base records
        records = list(base_records)

        # 4) now add any non-base modalities, if the file exists  
        for split, valid_stems in stems_by_split.items():
            split_dir = os.path.join(root, split)
            for mod, meta in modalities.items():
                if mod in base_modalities:
                    continue
                subfolder = meta.get("folder")
                if not subfolder:
                    continue
                folder = os.path.join(split_dir, subfolder)
                for stem in valid_stems:
                    fname = f"{stem}_{mod}.npy"
                    records.append({
                        "split":    split,
                        "modality": mod,
                        "stem":     stem,
                        "path":     os.path.join(folder, fname),
                    })

        # print(_pivot_views(records)[-1])
        
    else:
        base_records = []
        for fname in _listdir_safe(root):
            if _is_junk(fname):
                continue
            for mod, meta in modalities.items():
                if mod not in base_modalities:
                    continue
                pat = meta.get("pattern")
                if not pat:
                    continue
                m = re.fullmatch(pat, fname)
                if not m:
                    continue

                # extract stem
                if m.groups():
                    stem = m.group(1)
                else:
                    grp_pat = pat.replace(".*", "(.*)")
                    m2 = re.fullmatch(grp_pat, fname)
                    if m2:
                        stem = m2.group(1)
                    else:
                        stem = os.path.splitext(fname)[0]
                
                base_records.append({
                        "modality": mod,
                        "stem":     stem,
                        "path":     os.path.join(root, fname),
                    })
        

        # Filter out incomplete groups
        complete_base = _filter_complete_no_split(base_records, base_modalities)
        valid_stems = {rec["stem"] for rec in complete_base}

        # Add complete base records
        records = list(complete_base)
        # print(_pivot_views_no_split(records)[0])
        # raise

        # Now append non-base modalities
        for mod, meta in modalities.items():
            if mod in base_modalities:
                continue
            pat = meta.get("pattern")
            if not pat:
                continue
            for stem in valid_stems:
                escaped_stem = re.escape(stem)
                full_pat = pat.replace("(.*)", f"({escaped_stem})")
                fname   = filename_from_pattern(pat, stem)
                # print(mod, stem)
                # print(fname)
                records.append({
                    "modality": mod,
                    "stem":     stem,
                    "path":     os.path.join(root, fname),
                })
        # print(_pivot_views_no_split(records)[0])
        records = split_records(src, records, rng)
    
    return records

def split_records(src, records, rng):
    """
    Assigns a 'split' field to each record in `records` based on the split strategy in `src`.

    - For 'ratio': groups by stem, shuffles, and slices by the provided ratios.
    - For 'kfold': performs K-fold cross-validation on stems, using fold_idx as the held-out fold.
    """
    split_type = src.get('type')
    if split_type == 'folder':  
        return records
    
    # RATIO-BASED SPLIT
    if split_type == 'ratio':
        # Collect unique stems
        stems = sorted({r['stem'] for r in records})
        # Shuffle with seed if provided
        # seed = src.get('seed', None)
        # if seed is not None:
        #     random.Random(seed).shuffle(stems)
        # else:
        #     random.shuffle(stems)
        rng.shuffle(stems)

        ratios = src.get('ratios', {})
        train_ratio = ratios.get('train', 0)
        valid_ratio = ratios.get('valid', 0)
        # test_ratio implied
        n = len(stems)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        # Ensure all accounted for
        n_test = n - n_train - n_valid

        train_stems = set(stems[:n_train])
        valid_stems = set(stems[n_train:n_train + n_valid])
        test_stems  = set(stems[n_train + n_valid:])

        # Assign splits
        for rec in records:
            s = rec['stem']
            if s in train_stems:
                rec['split'] = 'train'
            elif s in valid_stems:
                rec['split'] = 'valid'
            else:
                rec['split'] = 'test'
        return records

    # K-FOLD SPLIT
    elif split_type == 'kfold':
        num_folds = src.get('num_folds')
        fold_idx  = src.get('fold_idx', 0)
        # Collect unique stems
        stems = sorted({r['stem'] for r in records})
        # Prepare KFold
        kf = KFold(n_splits=num_folds,
                   shuffle=True,
                   random_state=src.get('seed', None))
        # Find the train/valid split for the requested fold
        for idx, (train_idx, valid_idx) in enumerate(kf.split(stems)):
            if idx == fold_idx:
                train_stems = {stems[i] for i in train_idx}
                valid_stems = {stems[i] for i in valid_idx}
                break

        # Assign splits
        for rec in records:
            if rec['stem'] in train_stems:
                rec['split'] = 'train'
            elif rec['stem'] in valid_stems:
                rec['split'] = 'valid'
                
        return records

    else:
        raise ValueError(f"Unsupported split type: {split_type}")

class Split:
    """
    Unified splitter with support for 'folder', 'ratio', and 'kfold' sources.
    Adds base_modalities intersection and path validation.
    """
    def __init__(self, cfg: Dict[str, Any], base_modalities: List[str]) -> None:
        self.cfg = cfg
        self.seed = cfg.get("seed", 0)
        self.index_save_pth = cfg.get("index_save_pth", None)

        if self.index_save_pth:
            os.makedirs(self.index_save_pth, exist_ok=True)
            
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.base_modalities = base_modalities
        # Make the RNG deterministic but *local* to this instance
        self._rng = random.Random(self.seed)
        np.random.seed(self.seed)

        # Filled lazily by _build_splits()
        self._splits_built = False
        self._split2mod2files: Dict[str, Dict[str, List[str]]] = {}

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def get_split(self, split: str) -> Dict[str, List[str]]:
        """
        Args
        ----
        split : str
            One of "train", "valid", "test".

        Returns
        -------
        Dict[str, List[str]]
            { modality : [file paths] }  (lists are *already sorted*).
        """
        if split not in ("train", "valid", "test"):
            raise ValueError("split must be 'train', 'valid' or 'test'")

        if not self._splits_built:
            self._build_splits()

        # `.get` so an empty dict is returned if this split doesn't exist
        return self._split2mod2files.get(split, {})

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _build_splits(self) -> None:
        """
        One-shot scan of every source in the config → populate
        self._split2mod2files.
        """
        all_records: List[Dict[str, str]] = []

        # 1) collect and (if needed) split each source
        for src in self.cfg.get("sources", []):
            # recs = _collect_datapoints_from_source(src, self.base_modalities)
            recs = self._collect_datapoints_for(src)
            all_records.extend(recs)

        # 2) bucket by split / modality
        split2mod2files = defaultdict(lambda: defaultdict(list))
        for rec in all_records:
            sp, mod, path = rec["split"], rec["modality"], rec["path"]
            split2mod2files[sp][mod].append(path)

        # 3) deterministic order: sort the lists so paired modalities stay aligned
        for sp in split2mod2files:
            for mod in split2mod2files[sp]:
                split2mod2files[sp][mod].sort()

        # ---- print a summary of each split ----
        for sp, mod2files in split2mod2files.items():
            # count “stems” via the first base modality
            if self.base_modalities:
                base = self.base_modalities[0]
                stem_count = len(mod2files.get(base, []))
            else:
                stem_count = sum(len(v) for v in mod2files.values()) // max(1, len(mod2files))
            print(f"→ Split '{sp}': {stem_count} stems")
            for mod, files in mod2files.items():
                print(f"     {mod:8s}: {len(files)} files")
        # ---- end summary ----

        self._split2mod2files = split2mod2files
        self._splits_built    = True

        if self.index_save_pth:
            self._save_indices()

    def _save_indices(self) -> None:
        """
        Save the file list for each modality in each split as JSON.
        """
        import json
        for split, mod2files in self._split2mod2files.items():
            index = {mod: files for mod, files in mod2files.items()}
            fname = os.path.join(self.index_save_pth, f"{split}_index.json")
            with open(fname, 'w') as f:
                json.dump(index, f, indent=2)
            print(f"Saved index for split '{split}' to {fname}")

    def _collect_datapoints_for(self, src: Dict[str, Any]) -> List[Dict[str, str]]:
        records = _collect_datapoints_from_source(src, self.base_modalities, self._rng)
        if src.get('type')=='kfold':
            test_src = src.get('test_source')
            test_records = _collect_datapoints_from_source(test_src, self.base_modalities, self._rng)  
            records.extend([rec for rec in test_records if rec['split']=='test'])
        return records
    


def main():
    split_cfg = {
        "seed": 42,
        "sources": [
            {
                "type": "folder",
                "path": "/data/folder1",
                "layout": "folders",
                "modalities": {
                    "image": {"folder": "imgs"},
                    "label": {"folder": "lbls"}
                },
                "splits": {"train": "train_dir", "valid": "val_dir", "test": "test_dir"}
            },
            {
                "type": "ratio",
                "path": "/data/flat2",
                "layout": "flat",
                "modalities": {
                    "image": {"pattern": r".*\\.jpg$"},
                    "mask":  {"pattern": r".*\\.png$"}
                },
                "ratios": {"train": 0.7, "valid": 0.2, "test": 0.1}
            },
            {
                "type": "kfold",
                "num_folds": 5,
                "fold_idx": 0,
                "path": "/data/flat_tv",
                "layout": "flat",
                "modalities": {
                    "image": {"pattern": r".*\.npy$"},
                    "seg":   {"pattern": r".*\.npy$"}
                },
                "test_source":{
                    # this can only be type ratio 
                }
            }
        ]
    }

    splitter = Split(split_cfg)
    for split_name in ("train", "valid", "test"):
        mappings = splitter.get_split(split_name)
        print(f"=== {split_name.upper()} ===")
        for mod, files in mappings.items():
            print(f"  {mod}: {len(files)} files")

if __name__ == "__main__":
    main()


# ------------------------------------
# core/general_dataset/normalizations.py
# ------------------------------------
"""
normalizations.py

Various image normalization routines for 2D/3D numpy arrays.
Define multiple strategies and a unified `normalize` dispatcher, plus a binarization helper.
"""
import numpy as np
from core.general_dataset.logger import logger
from typing import Optional


def min_max_normalize(
    image: np.ndarray,
    new_min: float = 0.0,
    new_max: float = 1.0,
    old_min: Optional[float] = None,
    old_max: Optional[float] = None,
) -> np.ndarray:
    """
    Scale image intensities to [new_min, new_max].
    If old_min/old_max are provided, use them as the original bounds;
    otherwise compute from the image itself.
    """
    img = image.astype(np.float32)
    lo = old_min if old_min is not None else float(img.min())
    hi = old_max if old_max is not None else float(img.max())
    if hi <= lo:
        logger.warning(
            "Min-Max normalization: invalid range old_min=%s, old_max=%s; returning new_min.",
            lo, hi
        )
        return np.full_like(img, new_min)
    scaled = (img - lo) / (hi - lo)
    return scaled * (new_max - new_min) + new_min


def z_score_normalize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Subtract mean and divide by standard deviation (per-image).
    """
    img = image.astype(np.float32)
    mean = img.mean()
    std = img.std()
    if std < eps:
        logger.warning("Z-Score normalization: low variance, returning zeros.")
        return np.zeros_like(img)
    return (img - mean) / std


def robust_normalize(image: np.ndarray, lower_q: float = 0.05, upper_q: float = 0.95) -> np.ndarray:
    """
    Clip intensities to [lower_q, upper_q] quantiles and then apply min-max scaling.
    """
    img = image.astype(np.float32)
    low, high = np.quantile(img, [lower_q, upper_q])
    if high == low:
        logger.warning("Robust normalization: quantiles equal, returning zeros.")
        return np.zeros_like(img)
    clipped = np.clip(img, low, high)
    return (clipped - low) / (high - low)


def percentile_normalize(image: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    """
    Similar to robust but in percent (q_low and q_high are percents).
    Clip to percentiles then min-max scale to [0,1].
    """
    return robust_normalize(image, q_low/100.0, q_high/100.0)


def clip_normalize(image: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Clip intensities to [min_val, max_val] then scale to [0,1].
    """
    img = image.astype(np.float32)
    clipped = np.clip(img, min_val, max_val)
    if max_val == min_val:
        logger.warning("Clip normalization: min_val == max_val, returning zeros.")
        return np.zeros_like(img)
    return (clipped - min_val) / (max_val - min_val)


def binarize(
    image: np.ndarray,
    threshold: float,
    greater_is_road: bool = True
) -> np.ndarray:
    """
    Binarize the image based on a threshold.

    Args:
        image:    Input array.
        threshold: Value to threshold at.
        greater_is_road: If True, pixels > threshold become 1; else pixels < threshold become 1.

    Returns:
        A uint8 array of 0s and 1s.
    """
    img = image.astype(np.float32)
    if greater_is_road:
        mask = img > threshold
    else:
        mask = img <= threshold
    return mask.astype(np.uint8)


def normalize(
    image: np.ndarray,
    method: str = "minmax",
    **kwargs
) -> np.ndarray:
    """
    Dispatch to a normalization method. Supported methods:
      - 'minmax'    : min_max_normalize
      - 'zscore'    : z_score_normalize
      - 'robust'    : robust_normalize
      - 'percentile': percentile_normalize
      - 'clip'      : clip_normalize (requires min_val, max_val args)
      - 'binarize'  : binarize (requires threshold, optional greater_is_road)
    """
    method = method.lower()
    if method == "minmax":
        return min_max_normalize(image, **kwargs)
    elif method == "zscore":
        return z_score_normalize(image, **kwargs)
    elif method == "robust":
        return robust_normalize(image, **kwargs)
    elif method == "percentile":
        return percentile_normalize(image, **kwargs)
    elif method == "clip":
        return clip_normalize(image, **kwargs)
    elif method == "binarize":
        return binarize(image, **kwargs)
    else:
        logger.error(f"Unknown normalization method '{method}', using minmax.")
        return min_max_normalize(image, **kwargs)

# Alias for backward compatibility
normalize_image = normalize



# normalization_config = {
#     "minmax": {
#         "method": "minmax",
#         "new_min": 0.0,    # lower bound of output range
#         "new_max": 1.0     # upper bound of output range
#     },
#     "zscore": {
#         "method": "zscore",
#         "eps": 1e-8        # small constant to avoid division by zero
#     },
#     "robust": {
#         "method": "robust",
#         "lower_q": 0.05,   # clip everything below 5th quantile
#         "upper_q": 0.95    # clip everything above 95th quantile
#     },
#     "percentile": {
#         "method": "percentile",
#         "q_low": 1.0,      # clip below 1st percentile
#         "q_high": 99.0     # clip above 99th percentile
#     },
#     "clip": {
#         "method": "clip",
#         "min_val": 0.0,    # lower clipping threshold
#         "max_val": 200.0   # upper clipping threshold
#     }
# }

# ------------------------------------
# core/general_dataset/augments_old.py
# ------------------------------------
from typing import Any, Dict, List, Optional
import numpy as np
from core.general_dataset.logger import logger

import math
from scipy.ndimage import rotate


def get_augmentation_metadata(augmentations: List[str], data_dim: int) -> Dict[str, Any]:
    """
    Generate random augmentation parameters for a patch.

    Returns:
        Dict[str, Any]: Augmentation metadata.
    """
    meta: Dict[str, Any] = {}
    if 'rotation' in augmentations:
        meta['angle'] = np.random.uniform(0, 360)
    if 'flip_h' in augmentations:
        meta['flip_h'] = np.random.rand() > 0.5
    if 'flip_v' in augmentations:
        meta['flip_v'] = np.random.rand() > 0.5
    if 'flip_d' in augmentations and data_dim == 3:
        meta['flip_d'] = np.random.rand() > 0.5
    return meta


def flip_h(full_array: np.ndarray) -> np.ndarray:
    return np.flip(full_array, axis=-1)

def flip_v(full_array: np.ndarray) -> np.ndarray:
    return np.flip(full_array, axis=-2)

def flip_d(full_array: np.ndarray) -> np.ndarray:
    return np.flip(full_array, axis=-3)

def rotate_(
    full_array: np.ndarray,
    patch_meta: Dict[str, Any],
    patch_size_xy: int,
    patch_size_z: int,
    data_dim: int
) -> np.ndarray:
    """Rotate patch, preserving shape, for 2D or 3D."""
    angle = patch_meta['angle']
    # Compute L only in-plane
    L = int(np.ceil(patch_size_xy * math.sqrt(2)))
    x, y = patch_meta['x'], patch_meta['y']
    # for 3D also get z but we only rotate each slice independently
    cx, cy = x + patch_size_xy // 2, y + patch_size_xy // 2
    half_L = L // 2
    x0 = max(0, cx - half_L)
    y0 = max(0, cy - half_L)
    x1 = min(full_array.shape[-1], cx + half_L)
    y1 = min(full_array.shape[-2], cy + half_L)
    x0, x1 = max(0, cx-half_L), min(full_array.shape[-1], cx+half_L)
    y0, y1 = max(0, cy-half_L), min(full_array.shape[-2], cy+half_L)

    if data_dim == 3:
        z = patch_meta.get('z', 0)
        # crop a 3D block, but rotate per-slice
        block = full_array[:, z:z+patch_size_z, y0:y1, x0:x1]
        D, Hc, Wc = block.shape[1:]
        if Hc < L or Wc < L:
            logger.warning("Crop too small for 3D rotation; zero patch.")
            return np.zeros_like(block[..., :patch_size_xy, :patch_size_xy])
        rotated_slices = []
        for d in range(block.shape[1]):
            # rotate each C×H×W slice
            slice_ = block[:, d]
            rotated = rotate(slice_, angle, reshape=False, order=1)
            # center-crop back to patch_size_xy
            start = (L - patch_size_xy)//2
            cropped = rotated[start:start+patch_size_xy, start:start+patch_size_xy]
            rotated_slices.append(cropped)
        return np.stack(rotated_slices, axis=1)  # C×Z×XY×XY
    
    else:
        # 2D case: full_array is H×W or C×H×W
        crop = full_array[y0:y1, x0:x1] if full_array.ndim == 2 else full_array[:, y0:y1, x0:x1]
        Hc, Wc = crop.shape[-2], crop.shape[-1]
        if Hc < L or Wc < L:
            logger.warning("Crop too small for 2D rotation; zero patch.")
            shape = (patch_size_xy, patch_size_xy) if crop.ndim == 2 else (crop.shape[0], patch_size_xy, patch_size_xy)
            return np.zeros(shape, dtype=crop.dtype)
        rotated = rotate(crop, angle, reshape=False, order=1)
        start = (L - patch_size_xy)//2
        if rotated.ndim == 2:
            return rotated[start:start+patch_size_xy, start:start+patch_size_xy]
        else:
            return rotated[:, start:start+patch_size_xy, start:start+patch_size_xy]
        

def extract_condition_augmentations(
    imgs: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    patch_size_xy: int,
    patch_size_z: int,
    augmentations: List[str],
    data_dim: int
) -> Dict[str, np.ndarray]:
    """
    Extract a patch from the full image and apply conditional augmentations.

    Args:
        imgs (Dict[str, np.ndarray]): Full images for each modality.
        metadata (Dict[str, Any]): Metadata containing patch coordinates and augmentations.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary of extracted patches.
    """
    imgs_aug = imgs.copy()
    z = metadata.get('z', 0)
    data = extract_data(imgs,
                        metadata['x'], metadata['y'], z,
                        patch_size_xy,
                        patch_size_z)
    for key in imgs:
        if key.endswith("_patch"):
            modality = key.replace("_patch", "")
            if 'flip_h' in augmentations:
                imgs_aug[modality] = flip_h(imgs[modality])
                data[key] = flip_h(data[key])
            if 'flip_v' in augmentations:
                imgs_aug[modality] = flip_v(imgs[modality])
                data[key] = flip_v(data[key])
            if 'flip_d' in augmentations and data_dim == 3:
                imgs_aug[modality] = flip_d(imgs_aug[modality])
                data[key]          = flip_d(data[key])
            if 'rotation' in augmentations:
                data[key] = rotate_(imgs_aug[modality], metadata, patch_size_xy, patch_size_z, data_dim)
    return data


def extract_data(imgs: Dict[str, np.ndarray],
                 x: int, y: int, z: int,
                 patch_size_xy: int,
                 patch_size_z: Optional[int] = None) -> Dict[str, np.ndarray]:
    
    data = {}
    for key, arr in imgs.items():
        if arr.ndim == 4:  # C, D, H, W
            psz = patch_size_z or patch_size_xy
            data[f"{key}_patch"] = arr[:, z:z+psz, y:y+patch_size_xy, x:x+patch_size_xy]
        elif arr.ndim == 3:
            if patch_size_z is not None:
                data[f"{key}_patch"] = arr[z:z+patch_size_z, y:y+patch_size_xy, x:x+patch_size_xy]
            else:
                data[f"{key}_patch"] = arr[y:y+patch_size_xy, x:x+patch_size_xy]
        elif arr.ndim == 2:
            data[f"{key}_patch"] = arr[y:y+patch_size_xy, x:x+patch_size_xy]
        else:
            raise ValueError("Unsupported array dimensions in _extract_data")
    return data

# ------------------------------------
# core/general_dataset/patch_validity.py
# ------------------------------------
from typing import Any, Dict, List, Optional
import numpy as np
from core.general_dataset.logger import logger


def check_min_thrsh_road(label_patch: np.ndarray, patch_size, threshold) -> bool:
    """
    Check if the label patch has at least a minimum percentage of road pixels.

    Args:
        label_patch (np.ndarray): The label patch.
    
    Returns:
        bool: True if the patch meets the minimum threshold; False otherwise.
    """
    patch = label_patch
    if patch.max() > 1:
        patch = (patch > 127).astype(np.uint8)
    road_percentage = np.sum(patch) / (patch_size * patch_size)
    return road_percentage >= threshold


def check_small_window(image_patch: np.ndarray, small_window_size) -> bool:
    """
    Check that no small window in the image patch is entirely black or white.

    Args:
        image_patch (np.ndarray): Input patch (H x W) or (C x H x W)

    Returns:
        bool: True if valid, False if any window is all black or white.
    """
    sw = small_window_size

    # Ensure image has shape (C, H, W)
    if image_patch.ndim == 2:
        image_patch = image_patch[None, :, :]  # Add channel dimension

    C, H, W = image_patch.shape
    if H < sw or W < sw:
        return False

    # Set thresholds
    max_val = image_patch.max()
    if max_val > 1.0:
        high_thresh = 255
        low_thresh = 0
    else:
        high_thresh = 255 / 255.0
        low_thresh = 0 / 255.0

    # Slide window over spatial dimensions
    for c in range(C):
        for y in range(0, H - sw + 1):
            for x in range(0, W - sw + 1):
                window = image_patch[c, y:y + sw, x:x + sw]
                window_var = np.var(window)
                if window_var < 0.01:
                    return False
                # print(window)
                if np.all(window >= high_thresh):
                    return False  # Found an all-white window
                if np.all(window <= low_thresh):
                    return False  # Found an all-black window

    return True  # All windows passed




# ------------------------------------
# core/general_dataset/io.py
# ------------------------------------
from pathlib import Path
from typing import Optional
import numpy as np
import rasterio
import logging
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

logger = logging.getLogger(__name__)

def load_array_from_file(file_path: str) -> Optional[np.ndarray]:
    """
    Load an array from disk. Supports:
      - .npy      → numpy.load
      - .tif/.tiff → rasterio.open + read
    Returns:
      np.ndarray on success, or None if the file cannot be read.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    loaders = {
        '.npy': lambda p: np.load(p),
        '.tif': lambda p: rasterio.open(p).read().astype(np.float32),
        '.tiff': lambda p: rasterio.open(p).read().astype(np.float32),
    }
    
    loader = loaders.get(ext)
    if loader is None:
        logger.warning("Unsupported file extension '%s' for %s", ext, file_path)
        return None

    try:
        return loader(str(path))
    except Exception as e:
        # logger.warning("Failed to load '%s': %s", file_path, e)
        return None

# ------------------------------------
# core/general_dataset/__init__.py
# ------------------------------------
# core/general_dataset/__init__.py

from .base    import GeneralizedDataset
from .collate import custom_collate_fn, worker_init_fn

# Optionally, define __all__
__all__ = [
    "GeneralizedDataset",
    "custom_collate_fn",
    "worker_init_fn",
]


# ------------------------------------
# core/general_dataset/visualizations.py
# ------------------------------------
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

def visualize_batch_2d(batch: Dict[str, Any], num_per_batch: Optional[int] = None) -> None:
    """
    Visualizes patches in a batch: image, label, distance, and SDF (if available).

    Args:
        batch (Dict[str, Any]): Dictionary containing batched patches.
        num_per_batch (Optional[int]): Maximum number of patches to visualize.
    """
    print('batch["image_patch"].shape:', batch["image_patch"].shape)
    num_to_plot = batch["image_patch"].shape[0]
    if num_per_batch:
        num_to_plot = min(num_to_plot, num_per_batch)
    for i in range(num_to_plot):
        sample_image = batch["image_patch"][i].numpy()
        if sample_image.shape[0] == 3:  # CHW to HWC
            sample_image = sample_image.transpose(1, 2, 0)
        elif sample_image.shape[0] == 1:
            sample_image = sample_image[0]  # grayscale
        else:
            sample_image = sample_image.transpose(1, 2, 0)

        sample_label = np.squeeze(batch["label_patch"][i].numpy())
        sample_distance = batch["distance_patch"][i].numpy() if "distance_patch" in batch else None
        sample_sdf = batch["sdf_patch"][i].numpy() if "sdf_patch" in batch else None

        print(f'Patch {i}')
        print('  image:', sample_image.min(), sample_image.max())
        print('  label:', sample_label.min(), sample_label.max())
        if sample_distance is not None:
            print('  distance:', sample_distance.min(), sample_distance.max())
        if sample_sdf is not None:
            print('  sdf:', sample_sdf.min(), sample_sdf.max())

        num_subplots = 3 + (1 if sample_sdf is not None else 0)
        fig, axs = plt.subplots(1, num_subplots, figsize=(12, 4))
        axs[0].imshow(sample_image, cmap='gray' if sample_image.ndim == 2 else None)
        axs[0].set_title("Image")
        axs[0].axis("off")
        axs[1].imshow(sample_label, cmap='gray')
        axs[1].set_title("Label")
        axs[1].axis("off")
        if sample_distance is not None:
            axs[2].imshow(sample_distance[0], cmap='gray')
            axs[2].set_title("Distance")
            axs[2].axis("off")
        else:
            axs[2].text(0.5, 0.5, "No Distance", ha='center', va='center')
            axs[2].axis("off")
        if sample_sdf is not None:
            axs[3].imshow(sample_sdf[0], cmap='coolwarm')
            axs[3].set_title("SDF")
            axs[3].axis("off")
        plt.tight_layout()
        plt.show()


def visualize_batch_3d(
    batch: Dict[str, Any],
    projection: str = "max",
    num_per_batch: Optional[int] = None
) -> None:
    """
    Visualizes 3D patches in a batch by projecting along the Z axis.
    
    Args:
        batch: dict with keys image_patch, label_patch, etc., each a Tensor [B,C,Z,H,W] or [B,Z,H,W]
        projection: one of "max", "min", "mean"
        num_per_batch: how many samples to plot
    """
    assert projection in ("max","min","mean"), "projection must be max, min, or mean"
    
    # which modalities?
    print('batch', batch.keys())
    mods = [k.replace("_patch","") for k in batch if k.endswith("_patch")]
    print('batch["image_patch"].shape:', batch["image_patch"].shape)
    num_to_plot = batch["image_patch"].shape[0]
    if num_per_batch:
        num_to_plot = min(num_to_plot, num_per_batch)
    
    for i in range(num_to_plot):
        # gather each modality’s volume
        vols = {}
        for mod in mods:
            arr = batch[f"{mod}_patch"][i].numpy()
            # arr is either (Z,H,W) or (C,Z,H,W)
            if arr.ndim == 4:
                # collapse channels by max
                arr = arr.max(axis=0)
            vols[mod] = arr  # now Z×H×W
        
        # set up subplot grid: rows=modalities, cols=1
        nrows = len(mods)
        ncols = 3
        fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
        project = {
            "max": np.max,
            "min": np.min,
            "mean": np.mean
        }[projection]
        print(f'Patch {i} ({projection}-projection)')
        for row, mod in enumerate(mods):
            vol = vols[mod]
            # Compute projections along axes:
            #   XY: collapse Z axis -> (H, W)
            #   XZ: collapse Y axis -> (Z, W)
            #   YZ: collapse X axis -> (Z, H) then transpose for display (H, Z)
            proj_xy = project(vol, axis=0)
            proj_xz = project(vol, axis=1)
            proj_yz = project(vol, axis=2).T

            projs = {"XY": proj_xy, "XZ": proj_xz, "YZ": proj_yz}

            for col, (name, proj) in enumerate(projs.items()):
                ax = axs[row, col] if nrows > 1 else axs[col]
                ax.imshow(proj, cmap='gray')
                ax.set_title(f"{mod} - {name} view")
                ax.axis("off")

                # Print projection stats
                print(f'  {mod} {name}: min={proj.min():.3f}, max={proj.max():.3f}, mean={proj.mean():.3f}')

        plt.tight_layout()
        plt.show()

# ------------------------------------
# metrics/connected_components.py
# ------------------------------------
import torch
import torch.nn as nn
import numpy as np
from skimage import measure
from typing import Any, List, Tuple, Set


class ConnectedComponentsQuality(nn.Module):
    """
    Connected Components Quality (CCQ) metric for segmentation.

    Evaluates detection+shape accuracy of connected components in binary
    (or thresholded) predictions vs. ground truth, on 2-D or 3-D volumes.
    """
    def __init__(
        self,
        data_dim: int = 2,
        min_size: int = 5,
        tolerance: float = 2.0,
        alpha: float = 0.5,
        threshold: float = 0.5,
        greater_is_road: bool = True,
        eps: float = 1e-8,
    ):
        """
        Args:
            data_dim: 2 for (H,W) images, 3 for (D,H,W) volumes.
            min_size: minimum component area/volume to keep.
            tolerance: max centroid distance (pixels/voxels) to match.
            alpha: weight [0–1] blending detection vs. shape scores.
            threshold: binarization cutoff.
            greater_is_road: direction of thresholding.
            eps: stability constant.
        """
        super().__init__()
        if data_dim not in (2, 3):
            raise ValueError("data_dim must be 2 or 3")
        self.data_dim        = data_dim
        self.min_size        = int(min_size)
        self.tolerance       = float(tolerance)
        self.alpha           = float(alpha)
        self.threshold       = float(threshold)
        self.greater_is_road = bool(greater_is_road)
        self.eps             = float(eps)

    def _bin(self, arr: np.ndarray) -> np.ndarray:
        return (arr > self.threshold) if self.greater_is_road else (arr <= self.threshold)

    def _ensure_channel(self, t: torch.Tensor) -> torch.Tensor:
        # If t is (B, H, W) or (B, D, H, W), insert channel at axis=1
        if t.dim() == self.data_dim + 1:
            return t.unsqueeze(1)
        # Else assume channel present: (B,1,H,W) or (B,1,D,H,W)
        return t

    def _label_connectivity(self) -> int:
        # skimage: connectivity=2 for 2-D 8-nbr, =1 for full 3-D adjacency
        return 2 if self.data_dim == 2 else 1

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (B, …) logits/prob maps or distance maps.
            y_true: (B, …) ground truth mask or continuous map.
        Returns:
            scalar CCQ score (higher is better).
        """
        # check for at least (B,H,W) or (B,D,H,W)
        if y_pred.dim() < self.data_dim + 1 or y_true.dim() < self.data_dim + 1:
            raise ValueError(f"Inputs must be at least {(self.data_dim+1)}-D")

        # unify channel dimension
        y_pred = self._ensure_channel(y_pred)
        y_true = self._ensure_channel(y_true)
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")

        B, C = y_pred.shape[:2]
        if C != 1:
            raise ValueError(f"CCQ only supports binary masks; got C={C}")

        conn = self._label_connectivity()
        scores: List[float] = []

        for b in range(B):
            pred_np = self._bin(y_pred[b,0].detach().cpu().numpy()).astype(np.uint8)
            true_np = self._bin(y_true[b,0].detach().cpu().numpy()).astype(np.uint8)

            # If ground truth is empty
            if true_np.sum() == 0:
                scores.append(1.0 if pred_np.sum() == 0 else 0.0)
                continue

            # label connected components
            true_lbl = measure.label(true_np, connectivity=conn)
            pred_lbl = measure.label(pred_np, connectivity=conn)

            true_props = [p for p in measure.regionprops(true_lbl) if p.area >= self.min_size]
            pred_props = [p for p in measure.regionprops(pred_lbl) if p.area >= self.min_size]

            # no significant GT components
            if not true_props:
                scores.append(1.0 if not pred_props else 0.0)
                continue
            # no significant predicted components
            if not pred_props:
                scores.append(0.0)
                continue

            # match components by centroid
            matches = self._match_components(true_props, pred_props)

            tp = len(matches)
            fp = len(pred_props) - tp
            fn = len(true_props) - tp
            detection = tp / (tp + fp + fn + self.eps)

            # shape: mean IoU over matched pairs
            shape_scores = []
            for t_idx, p_idx in matches:
                t_mask = (true_lbl == true_props[t_idx].label)
                p_mask = (pred_lbl == pred_props[p_idx].label)
                inter = np.logical_and(t_mask, p_mask).sum()
                union = np.logical_or(t_mask, p_mask).sum()
                shape_scores.append(inter / (union + self.eps))
            shape = float(np.mean(shape_scores)) if shape_scores else 0.0

            scores.append(self.alpha * detection + (1 - self.alpha) * shape)

        avg = float(np.mean(scores))
        return torch.tensor(avg, device=y_pred.device)

    def _match_components(
        self,
        true_props: List[Any],
        pred_props: List[Any],
    ) -> List[Tuple[int, int]]:
        matches: List[Tuple[int, int]] = []
        used_pred: Set[int] = set()

        for t_idx, t_prop in enumerate(true_props):
            tx, ty = t_prop.centroid[:2]
            best_dist, best_idx = float('inf'), None
            for p_idx, p_prop in enumerate(pred_props):
                if p_idx in used_pred:
                    continue
                px, py = p_prop.centroid[:2]
                d = np.hypot(tx - px, ty - py)
                if d <= self.tolerance and d < best_dist:
                    best_dist, best_idx = d, p_idx
            if best_idx is not None:
                matches.append((t_idx, best_idx))
                used_pred.add(best_idx)

        return matches


# ------------------------------------
# metrics/dice.py
# ------------------------------------
import torch
import torch.nn as nn
from typing import Tuple

class ThresholdedDiceMetric(nn.Module):
    r"""
    Compute thresholded Dice coefficient for binary **or** multi-class
    predictions in **any** spatial dimensionality.

    Acceptable input shapes
    -----------------------
    * 2-D with channel:  (B, C, H, W)
    * 2-D no channel:   (B, H, W)                → channel added
    * 3-D with channel: (B, C, D, H, W)
    * 3-D no channel:   (B, D, H, W)             → channel added

    Args
    ----
    threshold (float): cut-off for binarization.
    eps (float): small constant to avoid division by zero.
    multiclass (bool): if True, macro-average over channels.
    zero_division (float): value to return when both pred & GT are empty.
    greater_is_road (bool): if True, values > threshold are “positive”.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        eps: float = 1e-6,
        multiclass: bool = False,
        zero_division: float = 1.0,
        greater_is_road: bool = True,
        data_dim: int = 2  # 2D or 3D data
    ):
        super().__init__()
        self.threshold      = float(threshold)
        self.eps            = float(eps)
        self.multiclass     = bool(multiclass)
        self.zero_division  = float(zero_division)
        self.greater_is_road = bool(greater_is_road)
        self.data_dim = int(data_dim)

    def _binarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.greater_is_road:
            return (x > self.threshold).float()
        else:
            return (x <= self.threshold).float()

    def _ensure_channel(self, t: torch.Tensor) -> torch.Tensor:
        """
        Ensure a channel dimension for 2D or 3D data.
        For 2D: (B,H,W) -> (B,1,H,W)
        For 3D: (B,D,H,W) -> (B,1,D,H,W)
        Leaves (B,1,H,W) or (B,1,D,H,W) unchanged.
        """
        expected_dim = 2 + self.data_dim  # batch + channel + spatial
        if t.dim() == expected_dim - 1:
            # Missing channel dimension
            return t.unsqueeze(1)
        return t

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # must have at least (B, H, W)
        if y_pred.dim() < 3 or y_true.dim() < 3:
            raise ValueError("Expected inputs with at least 3 dims: (B, …, spatial)")
        # add channel if missing
        y_pred = self._ensure_channel(y_pred)
        y_true = self._ensure_channel(y_true)

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: pred {y_pred.shape} vs true {y_true.shape}")

        N, C = y_pred.shape[:2]
        if not self.multiclass and C != 1:
            raise ValueError(f"Binary mode expects C=1 channel, got C={C}")

        # threshold
        p = self._binarize(y_pred)
        g = self._binarize(y_true)

        # sum over all spatial dims
        spatial = tuple(range(2, p.dim()))
        inter = (p * g).sum(dim=spatial)           # (B, C)
        sums  = p.sum(dim=spatial) + g.sum(dim=spatial)  # (B, C)

        # stable dice
        eps_t = torch.tensor(self.eps, device=inter.device, dtype=inter.dtype)
        zd_t  = torch.tensor(self.zero_division, device=inter.device, dtype=inter.dtype)
        dice  = (2*inter + eps_t) / (sums + eps_t)       # (B, C)

        # handle empty
        empty = (sums == 0)
        if empty.any():
            dice = torch.where(empty, zd_t, dice)

        # reductions
        per_class = dice.mean(dim=0)            # → (C,)
        if self.multiclass and C > 1:
            return per_class.mean()             # macro-avg over classes
        return per_class.squeeze(0)             # scalar for binary

if __name__ == "__main__":
    # Example usage
    metric = ThresholdedDiceMetric(threshold=0.5, multiclass=True)
    y_pred = torch.rand(4, 3, 64, 64)  # Example prediction (B, C, H, W)
    y_true = torch.randint(0, 2, (4, 3, 64, 64))  # Example ground truth (B, C, H, W)
    
    dice_score = metric(y_pred, y_true)
    print("Dice Score:", dice_score)

    metric = ThresholdedDiceMetric(multiclass=False)

    # 2D no-channel
    y2 = torch.rand(2, 32, 32)
    assert torch.isclose(metric(y2, y2), torch.tensor(1.0))

    # 2D with channel
    y2c = y2.unsqueeze(1)
    assert torch.isclose(metric(y2c, y2c), torch.tensor(1.0))

    # 3D no-channel
    y3 = torch.rand(2, 8, 32, 32)
    assert torch.isclose(metric(y3, y3), torch.tensor(1.0))

    # 3D with channel
    y3c = y3.unsqueeze(1)
    assert torch.isclose(metric(y3c, y3c), torch.tensor(1.0))

    # multiclass
    metric_mc = ThresholdedDiceMetric(multiclass=True)
    y_mc = torch.rand(2, 3, 16, 16)
    assert torch.isclose(metric_mc(y_mc, y_mc), torch.tensor(1.0))


# ------------------------------------
# metrics/iou.py
# ------------------------------------
import torch
import torch.nn as nn
from typing import Tuple

class ThresholdedIoUMetric(nn.Module):
    r"""
    Compute thresholded Intersection-over-Union for binary **or** multi-class
    predictions in **any** spatial dimensionality.

    Acceptable input shapes
    -----------------------
    * 2-D with channel:  (B, C, H, W)
    * 2-D no channel:   (B, H, W)                → channel added
    * 3-D with channel: (B, C, D, H, W)
    * 3-D no channel:   (B, D, H, W)             → channel added

    Args
    ----
    threshold (float): Binarization cutoff.
    eps (float): Small constant to avoid division by zero.
    multiclass (bool): If *True*, macro-average across channels.
    zero_division (float): IoU to return when both pred & true are empty.
    greater_is_road (bool): If *True*, voxels **above** threshold are “road”
                            (foreground); otherwise “road” = `≤ threshold`.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        eps: float = 1e-6,
        multiclass: bool = False,
        zero_division: float = 1.0,
        greater_is_road: bool = True,
        data_dim: int = 2  # 2D or 3D data
    ):
        super().__init__()
        self.threshold       = float(threshold)
        self.eps             = float(eps)
        self.multiclass      = bool(multiclass)
        self.zero_division   = float(zero_division)
        self.greater_is_road = bool(greater_is_road)
        self.data_dim = int(data_dim)
    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _binarize(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.threshold).float() if self.greater_is_road else (x <= self.threshold).float()
    
    def _ensure_channel(self, t: torch.Tensor) -> torch.Tensor:
        """
        Ensure a channel dimension for 2D or 3D data.
        For 2D: (B,H,W) -> (B,1,H,W)
        For 3D: (B,D,H,W) -> (B,1,D,H,W)
        Leaves (B,1,H,W) or (B,1,D,H,W) unchanged.
        """
        expected_dim = 2 + self.data_dim  # batch + channel + spatial
        if t.dim() == expected_dim - 1:
            # Missing channel dimension
            return t.unsqueeze(1)
        return t

    # --------------------------------------------------------------------- #
    # forward
    # --------------------------------------------------------------------- #
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.ndim < 3:
            raise ValueError("Expected at least 3 dims: (B, …, spatial)")

        # insert channel if needed
        y_pred = self._ensure_channel(y_pred)
        y_true = self._ensure_channel(y_true)

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")

        N, C = y_pred.shape[:2]
        if not self.multiclass and C != 1:
            raise ValueError(f"Binary mode expects 1 channel, got {C}")

        # binarize
        y_pred_bin = self._binarize(y_pred)
        y_true_bin = self._binarize(y_true)

        # gather all spatial dimensions (2, 3, …)
        spatial_dims: Tuple[int, ...] = tuple(range(2, y_pred_bin.ndim))

        # intersection & union
        inter = (y_pred_bin * y_true_bin).sum(dim=spatial_dims)          # (B, C)
        sum_pred = y_pred_bin.sum(dim=spatial_dims)
        sum_true = y_true_bin.sum(dim=spatial_dims)
        union = sum_pred + sum_true - inter                              # (B, C)

        # IoU with ε for numerical stability
        iou = (inter + self.eps) / (union + self.eps)                    # (B, C)

        # handle empty masks → user-defined value
        zero_mask = union == 0
        if zero_mask.any():
            fill = torch.full_like(iou, self.zero_division)
            iou = torch.where(zero_mask, fill, iou)

        # reduce batch then, optionally, channels
        iou_per_class = iou.mean(dim=0)                                  # (C,)
        return iou_per_class if (self.multiclass and C > 1) else iou_per_class.squeeze(0)



if __name__ == "__main__":
    # Example usage
    metric = ThresholdedIoUMetric(threshold=0.5, multiclass=True)
    y_pred = torch.rand(4, 3, 32, 32)  # Example prediction tensor
    y_true = torch.randint(0, 2, (4, 3, 32, 32))  # Example ground truth tensor
    iou = metric(y_pred, y_true)
    print("IoU per class:", iou)

    metric = ThresholdedIoUMetric(multiclass=False)

    # 2D no-channel → perfect match → IoU == 1
    y2 = torch.rand(2, 32, 32)
    assert metric(y2, y2) == 1.0

    # 2D with channel (explicit C=1)
    y2c = y2.unsqueeze(1)  # shape (2,1,32,32)
    assert metric(y2c, y2c) == 1.0

    # 3D no-channel → perfect match → IoU == 1
    y3 = torch.rand(2, 8, 32, 32)
    assert metric(y3, y3) == 1.0

    # 3D with channel (explicit C=1)
    y3c = y3.unsqueeze(1)  # shape (2,1,8,32,32)
    assert metric(y3c, y3c) == 1.0

    # multiclass mode with C>1
    metric_mc = ThresholdedIoUMetric(multiclass=True)
    y2m = torch.rand(2, 3, 32, 32)
    assert metric_mc(y2m, y2m) == 1.0


# ------------------------------------
# metrics/apls.py
# ------------------------------------
import numpy as np
import torch
import torch.nn as nn
from metrics.apls_core import apls
from typing import Tuple

class APLS(nn.Module):
    """
    Average Path Length Similarity (APLS) metric for road–network segmentation.

    Works on 2-D (H,W) or 3-D (D,H,W) masks, with or without an explicit channel.

    Args:
        data_dim: 2 for 2-D masks, 3 for 3-D volumes.
        threshold: cutoff to binarize predictions.
        angle_range: passed to apls().
        max_nodes: passed to apls().
        max_snap_dist: passed to apls().
        allow_renaming: passed to apls().
        min_path_length: passed to apls().
        greater_is_road: direction of thresholding.
    """
    def __init__(
        self,
        data_dim: int = 2,
        threshold: float = 0.5,
        angle_range: Tuple[int,int] = (135, 225),
        max_nodes: int = 500,
        max_snap_dist: int = 4,
        allow_renaming: bool = True,
        min_path_length: int = 10,
        greater_is_road: bool = True,
    ):
        super().__init__()
        if data_dim not in (2, 3):
            raise ValueError("data_dim must be 2 or 3")
        self.data_dim       = data_dim
        self.threshold      = float(threshold)
        self.angle_range    = angle_range
        self.max_nodes      = int(max_nodes)
        self.max_snap_dist  = int(max_snap_dist)
        self.allow_renaming = bool(allow_renaming)
        self.min_path_length= int(min_path_length)
        self.greater_is_road= bool(greater_is_road)

    def _binarize(self, arr: np.ndarray) -> np.ndarray:
        return (arr > self.threshold).astype(np.uint8) \
               if self.greater_is_road else (arr <= self.threshold).astype(np.uint8)

    def _unify(self, m: torch.Tensor) -> np.ndarray:
        """
        Turn inputs of shape
          - (H,W), (1,H,W), (B,H,W), (B,1,H,W)  [2-D] or
          - (D,H,W), (1,D,H,W), (B,D,H,W), (B,1,D,H,W) [3-D]
        into a numpy array of shape (B, *spatial).
        """
        d = self.data_dim
        shp = m.shape
        # collapse channel if present
        if m.dim() == d + 2 and shp[1] == 1:
            arr = m[:, 0, ...]
        else:
            arr = m
        # ensure batch
        if arr.dim() == d:
            arr = arr[None, ...]
        return arr.detach().cpu().numpy()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # unify shapes → (B, *spatial)
        gt = self._unify(y_true)
        pr = self._unify(y_pred)
        if gt.shape != pr.shape:
            raise ValueError(f"GT {gt.shape} vs pred {pr.shape}")

        B = gt.shape[0]
        scores = np.zeros(B, dtype=np.float32)

        for i in range(B):
            gt_bin = self._binarize(gt[i])
            pr_bin = self._binarize(pr[i])

            # both empty
            if gt_bin.sum() == 0:
                scores[i] = 1.0 if pr_bin.sum()==0 else 0.0
                continue

            try:
                scores[i] = apls(
                    gt_bin,
                    pr_bin,
                    angle_range=self.angle_range,
                    max_nodes=self.max_nodes,
                    max_snap_dist=self.max_snap_dist,
                    allow_renaming=self.allow_renaming,
                    min_path_length=self.min_path_length
                )
            except Exception:
                scores[i] = 0.0

        return torch.tensor(float(scores.mean()), device=y_pred.device)


# ------------------------------------
# metrics/ccq.py
# ------------------------------------
import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

__all__ = ["ThresholdedCCQMetric"]

class ThresholdedCCQMetric(nn.Module):
    """
    Relaxed Correctness–Completeness–Quality (CCQ) metric for binary masks.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        slack: int = 3,
        eps: float = 1e-12,
        greater_is_road: bool = True,
        data_dim: int = 2,        # 2 = H,W   · 3 = D,H,W
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.slack = int(slack)
        self.eps = float(eps)
        self.greater_is_road = bool(greater_is_road)
        self.data_dim = int(data_dim)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _binarize(self, x: torch.Tensor) -> np.ndarray:
        """Torch → NumPy Bool with threshold & polarity."""
        x_np = x.detach().cpu().numpy()
        return (x_np > self.threshold) if self.greater_is_road else (x_np <= self.threshold)

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        return skeletonize(mask)

    def _ensure_channel(self, t: torch.Tensor) -> torch.Tensor:
        """Insert missing channel dim → (B,1,...) so we can strip it later."""
        spatial_dims = 2 + self.data_dim      # B + C + spatial
        if t.dim() == spatial_dims - 1:
            return t.unsqueeze(1)
        return t

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> np.ndarray:
        y_pred = self._ensure_channel(y_pred)
        y_true = self._ensure_channel(y_true)

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: pred {y_pred.shape}  vs  true {y_true.shape}")
        if y_pred.shape[1] != 1:
            raise ValueError("CCQ expects binary masks (C = 1)")

        # ---- to NumPy once ----
        pred_np = self._binarize(y_pred)[:, 0]      # (B, … spatial …)
        gt_np   = self._binarize(y_true)[:, 0]

        batch_scores = []

        for p_bin, g_bin in zip(pred_np, gt_np):
            p_bin = self._skeletonize(p_bin.astype(bool))
            g_bin = self._skeletonize(g_bin.astype(bool))

            # Distance maps on the inverse (background == True)
            dist_gt   = ndimage.distance_transform_edt(~g_bin)
            dist_pred = ndimage.distance_transform_edt(~p_bin)

            tp_area = dist_gt   <= self.slack
            fp_area = dist_gt   >  self.slack
            fn_area = dist_pred >  self.slack

            TP = np.logical_and(tp_area, p_bin).sum()
            FP = np.logical_and(fp_area, p_bin).sum()
            FN = np.logical_and(fn_area, g_bin).sum()

            # Empty–empty special-case  → perfect score
            if TP + FP + FN == 0:
                batch_scores.append([1.0, 1.0, 1.0])
                continue

            correctness  = TP / (TP + FP + self.eps)
            completeness = TP / (TP + FN + self.eps)
            quality      = TP / (TP + FP + FN + self.eps)
            batch_scores.append([correctness, completeness, quality])

        return np.mean(batch_scores, axis=0)


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

