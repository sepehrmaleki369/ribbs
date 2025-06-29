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
        loss = self.loss_fn(y_hat, y)

        # self._train_preds.append(y_hat.detach().flatten())
        # self._train_gts.append(  y.detach().flatten())
        # print(f"[Train]  x.shape={x.shape}, y.shape={y.shape}")
        # print(f"[Train] y_hat.shape={y_hat.shape}")
        # print(f"[Train] loss computed on y_hat of shape {y_hat.shape} and y of shape {y.shape}")

        self.log("train_loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))

        y_int = y
        for name, metric in self.metrics.items():
            freq = self.train_metric_frequencies.get(name, self.train_freq)
            if self.current_epoch % freq == 0:
                self.log(f"train_metrics/{name}", metric(y_hat, y_int),
                         prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        
        # ——— Log GT / Pred stats for TensorBoard ———
        # flatten tensors
        pred_flat = y_hat.flatten()
        gt_flat   = y.flatten()
        # GT statistics
        self.log("train_mmm/gt_min",   torch.min(gt_flat),   on_step=False, on_epoch=True)
        self.log("train_mmm/gt_max",   torch.max(gt_flat),   on_step=False, on_epoch=True)
        self.log("train_mmm/gt_mean",  torch.mean(gt_flat),  on_step=False, on_epoch=True)
        # Pred statistics
        self.log("train_mmm/pred_min", torch.min(pred_flat), on_step=False, on_epoch=True)
        self.log("train_mmm/pred_max", torch.max(pred_flat), on_step=False, on_epoch=True)
        self.log("train_mmm/pred_mean",torch.mean(pred_flat),on_step=False, on_epoch=True)


        return {"loss": loss, "predictions": y_hat, "gts": y}
    
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

        loss = self.loss_fn(y_hat, y)

        # self._val_preds.append(y_hat.detach().flatten())
        # self._val_gts  .append(y.detach().flatten())
        # print(f"[Val]  x.shape={x.shape}, y.shape={y.shape}")
        # print(f"[Val] y_hat.shape={y_hat.shape}")
        # print(f"[Val] loss computed on y_hat of shape {y_hat.shape} and y of shape {y.shape}")
        self.log("val_loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        y_int = y
        for name, metric in self.metrics.items():
            freq = self.val_metric_frequencies.get(name, self.val_freq)
            if self.current_epoch % freq == 0:
                self.log(f"val_metrics/{name}", metric(y_hat, y_int),
                         prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        
        # ——— Log GT / Pred stats for TensorBoard ———
        # flatten tensors
        pred_flat = y_hat.flatten()
        gt_flat   = y.flatten()
        # GT statistics
        self.log("val_mmm/gt_min",   torch.min(gt_flat),   on_step=False, on_epoch=True)
        self.log("val_mmm/gt_max",   torch.max(gt_flat),   on_step=False, on_epoch=True)
        self.log("val_mmm/gt_mean",  torch.mean(gt_flat),  on_step=False, on_epoch=True)
        # Pred statistics
        self.log("val_mmm/pred_min", torch.min(pred_flat), on_step=False, on_epoch=True)
        self.log("val_mmm/pred_max", torch.max(pred_flat), on_step=False, on_epoch=True)
        self.log("val_mmm/pred_mean",torch.mean(pred_flat),on_step=False, on_epoch=True)

        return {"predictions": y_hat, "val_loss": loss, "gts": y}
    
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

        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        y_int = y
        for name, metric in self.metrics.items():
            self.log(f"test_metrics/{name}", metric(y_hat, y_int),
                     prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        return {"predictions": y_hat, "test_loss": loss, "gts": y}

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
# core/general_dataset.py
# ------------------------------------
import os
import json
import warnings
import logging
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import KFold
from scipy.ndimage import distance_transform_edt, rotate, binary_dilation
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
assumptions:
    - lbl can be binary or int (thresholded by 127)
    - roads are 1 on lbl
    - image modality must be defined 
    - label modality is not required necessarily (even it's possible to define one folder to more than one modality) 
    - for modalities other that label and image: 
        - file names must be in this format: modality_filename = f"{base}_{key}.npy"
        - if the computed modality's folder does not contain file "config.json" it will be computed again
    - if use_splitting then there is two options:
        - 1) kfold -> source_folder, num_folds and fold is required
        - 2) split_ratio -> source_folder, ratios are required
    - there is two optiona overall: 
        - 1) setting stride -> so dataloader extracts all valid patches per image (removed in this version)
        - 2) if stride was None -> extracts just one patch per image
"""

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

def _is_readable_tiff(path: str) -> bool:
    """
    Stubbed-out for tests (and general use):
    never drop .tif/.tiff files based on rasterio.
    """
    return True

def load_array_from_file(file_path: str) -> Optional[np.ndarray]:
    """
    Load an array from disk.  If the file is unreadable, return None.
    """
    try:
        if file_path.endswith(".npy"):
            return np.load(file_path)
        else:
            with rasterio.open(file_path) as src:
                return src.read().astype(np.float32)
    except Exception:
        return None

class GeneralizedDataset(Dataset):
    """
    PyTorch Dataset for generalized remote sensing or segmentation datasets.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.root_dir: str = config.get("root_dir")
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
        self.distance_threshold: Optional[float] = config.get("distance_threshold")
        self.sdf_iterations: int = config.get("sdf_iterations")
        self.sdf_thresholds: List[float] = config.get("sdf_thresholds")
        self.num_workers: int = config.get("num_workers", 4)
        self.split_ratios: Dict[str, float] = config.get("split_ratios", {"train":0.7,"valid":0.15,"test":0.15})
        self.use_splitting: bool = config.get("use_splitting", False)
        self.modalities: Dict[str, str] = config.get("modalities", {"image":"sat","label":"map"})
        self.source_folder: str = config.get("source_folder", "")
        self.save_computed: bool = config.get("save_computed", False)

        if self.root_dir is None:
            raise ValueError("root_dir must be specified in the config.")
        if self.patch_size is None:
            raise ValueError("patch_size must be specified in the config.")

        random.seed(self.seed)
        np.random.seed(self.seed)

        split_dir = os.path.join(self.root_dir, self.split)
        if self.use_splitting and self.split != 'test':
            split_dir = os.path.join(self.root_dir, self.source_folder)
        self.data_dir: str = split_dir

        self.modality_dirs: Dict[str, str] = {}
        self.modality_files: Dict[str, List[str]] = {}
        exts = ['.tiff', '.tif', '.png', '.jpg', '.npy']

        # Process "image" modality.
        folder_name = self.modalities['image']
        mod_dir = os.path.join(self.data_dir, folder_name)
        if not os.path.isdir(mod_dir):
            raise ValueError(f"Modality directory {mod_dir} not found.")
        self.modality_dirs['image'] = mod_dir
        files = sorted(
            f for f in os.listdir(mod_dir)
            if any(f.endswith(ext) for ext in exts)
        )
        self.modality_files['image'] = files

        if self.max_images is not None:
            self.modality_files['image'] = self.modality_files['image'][:self.max_images]

        # Process "label" modality.
        if "label" in self.modalities:
            folder_name = self.modalities['label']
            mod_dir = os.path.join(self.data_dir, folder_name)
            if not os.path.isdir(mod_dir):
                raise ValueError(f"Modality directory {mod_dir} not found.")
            self.modality_dirs['label'] = mod_dir
            files = sorted(
                f for f in os.listdir(mod_dir)
                if any(f.endswith(ext) for ext in exts)
            )
            self.modality_files['label'] = files

            if self.max_images is not None:
                self.modality_files['label'] = self.modality_files['label'][:self.max_images]

        # Precompute additional modalities (e.g., distance, sdf).
        for key in [modal for modal in self.modalities if modal not in ['image', 'label']]:
            folder_name = self.modalities[key]
            mod_dir = os.path.join(self.data_dir, folder_name)
            os.makedirs(mod_dir, exist_ok=True)
            sdf_comp_again = False
            config_path = os.path.join(mod_dir, "config.json")
            # if os.path.exists(config_path):
            #     with open(config_path, "r") as config_file:
            #         saved_config = json.load(config_file)
            #         sdf_comp_again = self.sdf_iterations != saved_config.get("sdf_iterations", None)

            logger.info(f"Generating {key} modality maps...")
            for file_idx, file in tqdm(enumerate(self.modality_files["label"]),
                                       total=len(self.modality_files["label"]),
                                       desc=f"Processing {key} maps"):
                lbl_path = os.path.join(self.modality_dirs['label'], self.modality_files["label"][file_idx])
                lbl = load_array_from_file(lbl_path)
                base, _ = os.path.splitext(file)
                modality_filename = f"{base}_{key}.npy"
                modality_path = os.path.join(mod_dir, modality_filename)
                if self.save_computed:
                    if key == "distance":
                        if not os.path.exists(modality_path):
                            processed_map = compute_distance_map(lbl, None)
                            np.save(modality_path, processed_map)
                    elif key == "sdf":
                        if not os.path.exists(modality_path) or sdf_comp_again:
                            processed_map = compute_sdf(lbl, self.sdf_iterations, None)
                            np.save(modality_path, processed_map)
                    else:
                        raise ValueError(f"Modality {key} not supported.")
            with open(config_path, "w") as config_file:
                json.dump(self.config, config_file, indent=4)
            self.modality_dirs[key] = mod_dir
            files = sorted([f for f in os.listdir(mod_dir) if any(f.endswith(ext) for ext in exts)])
            self.modality_files[key] = files

        # Perform dataset splitting if requested.
        if self.use_splitting:
            if self.fold is not None and self.num_folds is not None:
                # ----- KFold Splitting -----
                files = self.modality_files["image"]
                kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
                splits = list(kf.split(files))
                if self.split == "train":
                    selected_indices = splits[self.fold][0].tolist()
                elif self.split == "valid":
                    selected_indices = splits[self.fold][1].tolist()
                elif self.split == "test":
                    selected_indices = [i for i in range(len(files))]
                else:
                    raise ValueError("For KFold splitting, split must be 'train' or 'valid' or 'test'.", self.split)
                for key in self.modality_files:
                    all_files = self.modality_files[key]
                    self.modality_files[key] = [all_files[i] for i in selected_indices]
            else:
                # ----- Split by Ratios -----
                files = self.modality_files["label"]
                num_files = len(files)
                indices = np.arange(num_files)
                np.random.shuffle(indices)
                train_count = int(num_files * self.split_ratios["train"])
                valid_count = int(num_files * self.split_ratios["valid"])
                if self.split == "train":
                    selected_indices = indices[:train_count]
                elif self.split == "valid":
                    selected_indices = indices[train_count:train_count + valid_count]
                elif self.split == "test":
                    selected_indices = indices[train_count + valid_count:]
                else:
                    raise ValueError("For an 'entire' folder, split must be one of 'train', 'valid', or 'test'.")
                for key in self.modality_files:
                    all_files = self.modality_files[key]
                    self.modality_files[key] = [all_files[i] for i in selected_indices]

        if self.max_images is not None:
            for key in self.modality_files:
                self.modality_files[key] = self.modality_files[key][:self.max_images]

    def _load_datapoint(self, file_idx: int) -> Optional[Dict[str, np.ndarray]]:
        imgs: Dict[str, np.ndarray] = {}
        for key in self.modalities:
            if file_idx < len(self.modality_files[key]):
                file_path = os.path.join(self.modality_dirs[key], self.modality_files[key][file_idx])
                arr = load_array_from_file(file_path)
                if arr is None:            # <- corrupted TIFF
                    return None            # signal caller to skip this index
            else:
                lbl_path = os.path.join(self.modality_dirs['label'], self.modality_files["label"][file_idx])
                lbl = load_array_from_file(lbl_path)
                if key == "distance":
                    arr = compute_distance_map(lbl, None)
                elif key == "sdf":
                    arr = compute_sdf(lbl, self.sdf_iterations, None)
                else:
                    raise ValueError(f"Modality {key} not supported.")
                
            imgs[key] = arr
        return imgs

    def _check_small_window(self, image_patch: np.ndarray) -> bool:
        """
        Check that no small window in the image patch is entirely black or white.

        Args:
            image_patch (np.ndarray): Input patch (H x W) or (C x H x W)

        Returns:
            bool: True if valid, False if any window is all black or white.
        """
        sw = self.small_window_size

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


    def _check_min_thrsh_road(self, label_patch: np.ndarray) -> bool:
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
        road_percentage = np.sum(patch) / (self.patch_size * self.patch_size)
        return road_percentage >= self.threshold

    def __len__(self) -> int:
        return len(self.modality_files['image'])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        imgs = self._load_datapoint(idx)
        if imgs is None:           # corrupted file detected
            # pick a different index (cyclic) so DataLoader doesn’t crash
            return self.__getitem__((idx + 1) % len(self))

        if imgs['image'].ndim == 3:
            _, H, W = imgs['image'].shape
        elif imgs['image'].ndim == 2:
            H, W = imgs['image'].shape
        else:
            raise ValueError("Unsupported image dimensions")

        if self.split != 'train':
            data = {}
            patch_meta = {"image_idx": idx, "x": -1, "y": -1}
            for key, array in imgs.items():
                if array.ndim == 3:
                    data[f"{key}_patch"] = array
                elif array.ndim == 2:
                    data[f"{key}_patch"] = array
                else:
                    raise ValueError("Unsupported array dimensions in _extract_data")
            data['metadata'] = patch_meta
            data = self._postprocess_patch(data)
            
            return data

        valid_patch_found = False
        attempts = 0
        while not valid_patch_found and attempts < self.max_attempts:
            x = np.random.randint(0, W - self.patch_size + 1)
            y = np.random.randint(0, H - self.patch_size + 1)
            patch_meta = {"image_idx": idx, "x": x, "y": y}
            if self.augmentations:
                patch_meta.update(self._get_augmentation_metadata())
            data = self._extract_condition_augmentations(imgs, patch_meta)
            if self.validate_road_ratio:
                if self._check_min_thrsh_road(data['label_patch']):
                    valid_patch_found = True
                    data['metadata'] = patch_meta
                    data = self._postprocess_patch(data)
                    return data
            else:
                valid_patch_found = True
                data['metadata'] = patch_meta
                data = self._postprocess_patch(data)
                return data
            
            attempts += 1

        # If a valid patch isn't found after max_attempts, fallback to the last sampled patch
        if not valid_patch_found:
            logger.warning("No valid patch found after %d attempts; trying next image", self.max_attempts)
            return self.__getitem__((idx + 1) % len(self))
        return None
        
    def _get_augmentation_metadata(self) -> Dict[str, Any]:
        """
        Generate random augmentation parameters for a patch.

        Returns:
            Dict[str, Any]: Augmentation metadata.
        """
        meta: Dict[str, Any] = {}
        if 'rotation' in self.augmentations:
            meta['angle'] = np.random.uniform(0, 360)
        if 'flip_h' in self.augmentations:
            meta['flip_h'] = np.random.rand() > 0.5
        if 'flip_v' in self.augmentations:
            meta['flip_v'] = np.random.rand() > 0.5
        return meta

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        return image / 255.0 if image.max() > 1.0 else image

    def _postprocess_patch(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize image patch values and binarize label patch if needed.

        Args:
            data (Dict[str, np.ndarray]): Dictionary containing patches.
        
        Returns:
            Dict[str, np.ndarray]: Postprocessed patches.
        """
        for key in data:
            if key == "image_patch":
                data[key] = self._normalize_image(data[key])
            elif key == "label_patch":
                if data[key].max() > 1:
                    data[key] = (data[key] > 127).astype(np.uint8)
            elif key == "distance_patch":
                if self.distance_threshold:
                    data[key] = np.clip(data[key], 0, self.distance_threshold)
            elif key == "sdf_patch":
                if self.sdf_thresholds:
                    data[key] = np.clip(data[key], self.sdf_thresholds[0], self.sdf_thresholds[1])
        return data

    def _extract_condition_augmentations(self, imgs: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract a patch from the full image and apply conditional augmentations.

        Args:
            imgs (Dict[str, np.ndarray]): Full images for each modality.
            metadata (Dict[str, Any]): Metadata containing patch coordinates and augmentations.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of extracted patches.
        """
        imgs_aug = imgs.copy()
        data = self._extract_data(imgs, metadata['x'], metadata['y'])
        for key in imgs:
            if key.endswith("_patch"):
                modality = key.replace("_patch", "")
                if 'flip_h' in self.augmentations:
                    imgs_aug[modality] = self._flip_h(imgs[modality])
                    data[key] = self._flip_h(data[key])
                if 'flip_v' in self.augmentations:
                    imgs_aug[modality] = self._flip_v(imgs[modality])
                    data[key] = self._flip_v(data[key])
                if 'rotation' in self.augmentations:
                    data[key] = self._rotate(imgs_aug[modality], metadata)
        return data

    def _extract_data(self, imgs: Dict[str, np.ndarray], x: int, y: int) -> Dict[str, np.ndarray]:
        """
        Extract a patch from each modality starting at (x, y) with size self.patch_size.

        Args:
            imgs (Dict[str, np.ndarray]): Full images.
            x (int): x-coordinate.
            y (int): y-coordinate.
        
        Returns:
            Dict[str, np.ndarray]: Extracted patch for each modality.
        """
        data: Dict[str, np.ndarray] = {}
        for key, array in imgs.items():
            if array.ndim == 3:
                data[f"{key}_patch"] = array[:, y:y + self.patch_size, x:x + self.patch_size]
            elif array.ndim == 2:
                data[f"{key}_patch"] = array[y:y + self.patch_size, x:x + self.patch_size]
            else:
                raise ValueError("Unsupported array dimensions in _extract_data")
        return data

    def _flip_h(self, full_array: np.ndarray) -> np.ndarray:
        return np.flip(full_array, axis=-1)
    
    def _flip_v(self, full_array: np.ndarray) -> np.ndarray:
        return np.flip(full_array, axis=-2)
    
    def _rotate(self, full_array: np.ndarray, patch_meta: Dict[str, Any]) -> np.ndarray:
        """
        Rotate a patch using an expanded crop to avoid border effects.
        If the crop is too small, log a warning and return a zero patch.

        Args:
            full_array (np.ndarray): Full image array.
            patch_meta (Dict[str, Any]): Contains patch coordinates and angle.
        
        Returns:
            np.ndarray: Rotated patch.
        """
        patch_size = self.patch_size
        L = int(np.ceil(patch_size * math.sqrt(2)))
        x = patch_meta["x"]
        y = patch_meta["y"]
        angle = patch_meta["angle"]

        cx = x + patch_size // 2
        cy = y + patch_size // 2
        half_L = L // 2
        x0 = max(0, cx - half_L)
        y0 = max(0, cy - half_L)
        x1 = min(full_array.shape[-1], cx + half_L)
        y1 = min(full_array.shape[-2], cy + half_L)

        if full_array.ndim == 3:
            crop = full_array[:, y0:y1, x0:x1]
            if crop.shape[1] < L or crop.shape[2] < L:
                logger.warning("Crop too small for 3D patch rotation; returning zero patch.")
                return np.zeros((full_array.shape[0], patch_size, patch_size), dtype=full_array.dtype)
            rotated_channels = [rotate(crop[c], angle, reshape=False, order=1) for c in range(full_array.shape[0])]
            rotated = np.stack(rotated_channels)
            start = (L - patch_size) // 2
            return rotated[:, start:start + patch_size, start:start + patch_size]
        elif full_array.ndim == 2:
            crop = full_array[y0:y1, x0:x1]
            if crop.shape[0] < L or crop.shape[1] < L:
                logger.warning("Crop too small for 2D patch rotation; returning zero patch.")
                return np.zeros((patch_size, patch_size), dtype=full_array.dtype)
            rotated = rotate(crop, angle, reshape=False, order=1)
            start = (L - patch_size) // 2
            return rotated[start:start + patch_size, start:start + patch_size]
        else:
            raise ValueError("Unsupported array shape")

        
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


def visualize_batch(batch: Dict[str, Any], num_per_batch: Optional[int] = None) -> None:
    """
    Visualizes patches in a batch: image, label, distance, and SDF (if available).

    Args:
        batch (Dict[str, Any]): Dictionary containing batched patches.
        num_per_batch (Optional[int]): Maximum number of patches to visualize.
    """
    import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    config = {
        "root_dir": "/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/",  # Update with your dataset path.
        "split": "train",
        # "split": "valid",
        "patch_size": 256,
        "small_window_size": 8,
        "validate_road_ratio": True,
        "threshold": 0.025,
        "max_images": 1,  # For quick testing.
        "seed": 42,
        "fold": None,
        "num_folds": None,
        "verbose": True,
        "augmentations": ["flip_h", "flip_v", "rotation"],
        "distance_threshold": 100.0,
        "sdf_iterations": 3,
        "sdf_thresholds": [-20, 20],
        "num_workers": 4,
        "use_splitting": False,
        "split_ratios": {
            "train": 0.7,
            "valid": 0.15,
            "test": 0.15
        },
        "modalities": {
            "image": "sat",
            "label": "map",
            "distance": "distance",
            "sdf": "sdf"
        }
    }

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
            visualize_batch(batch)
            # break  # Uncomment to visualize only one batch.


# ------------------------------------
# core/validator.py
# ------------------------------------
# core/validator.py
"""
Validator module for handling chunked inference in validation/test phases.
Implements the exact Road_2D_EEF approach with robust size handling.
"""

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import process_in_chuncks


class Validator:
    """
    Validator class for handling chunked inference during validation/testing.
    Uses the Road_2D_EEF process_in_chunks approach with robust size handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Validator.
        
        Args:
            config: Configuration dictionary with inference parameters such as 
                   patch_size and patch_margin
        """
        self.config = config
        self.patch_size = config.get("patch_size", [512, 512])
        self.patch_margin = config.get("patch_margin", [32, 32])
        self.logger = logging.getLogger(__name__)
        
        # Convert to tuples if provided as lists
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
            
        if isinstance(self.patch_margin, list):
            self.patch_margin = tuple(self.patch_margin)
            
        # Ensure patch_size and patch_margin have the same dimensions
        if len(self.patch_size) != len(self.patch_margin):
            raise ValueError(f"patch_size {self.patch_size} and patch_margin {self.patch_margin} "
                             f"must have the same number of dimensions")
    
    def _pad_to_valid_size(self, image: torch.Tensor, divisor: int = 16) -> tuple:
        """
        Pad image to ensure dimensions are divisible by divisor.
        
        Args:
            image: Input tensor (N, C, H, W)
            divisor: Divisor for dimension constraint (default: 16 for UNet with 3-4 levels)
            
        Returns:
            Tuple of (padded_image, (pad_h, pad_w)) where pad_h and pad_w are padding amounts
        """
        N, C, H, W = image.shape
        
        # Calculate required padding
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor
        
        if pad_h > 0 or pad_w > 0:
            # Pad with reflection to avoid artifacts
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
            # self.logger.debug(f"Padded image from ({H}, {W}) to ({H + pad_h}, {W + pad_w})")
        
        return image, (pad_h, pad_w)
    
    def _remove_padding(self, output: torch.Tensor, padding: tuple) -> torch.Tensor:
        """
        Remove padding from output tensor.
        
        Args:
            output: Padded output tensor (N, C, H, W)
            padding: Tuple of (pad_h, pad_w) padding amounts
            
        Returns:
            Unpadded output tensor
        """
        pad_h, pad_w = padding
        
        if pad_h > 0 or pad_w > 0:
            if pad_h > 0 and pad_w > 0:
                output = output[:, :, :-pad_h, :-pad_w]
            elif pad_h > 0:
                output = output[:, :, :-pad_h, :]
            elif pad_w > 0:
                output = output[:, :, :, :-pad_w]
        
        return output
    
    def run_chunked_inference(
        self,
        model: nn.Module,
        image: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Run inference on a large image by processing it in chunks.
        Pads the full image up to a multiple of 16, then
        splits into overlapping patches with margin, runs each
        chunk through the model, and stitches back together.
        """
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        image = image.to(device)

        # 1) pad the full image so H and W are divisible by 16
        padded_image, (pad_h, pad_w) = self._pad_to_valid_size(image, divisor=16)
        N, C, H, W = padded_image.shape

        # 2) figure out how many output channels the model produces
        #    by running one patch—but *also* pad that patch to 16!
        with torch.no_grad():
            # define a “test patch” of size (patch_size + 2*margin)
            test_h = min(H, self.patch_size[0] + 2 * self.patch_margin[0])
            test_w = min(W, self.patch_size[1] + 2 * self.patch_margin[1])
            test_patch = padded_image[:, :, :test_h, :test_w]
            # pad it so its dims are multiples of 16
            test_patch, _ = self._pad_to_valid_size(test_patch, divisor=16)
            test_out = model(test_patch)
            out_channels = test_out.shape[1]

        # 3) allocate a big “canvas” for all of our outputs
        output = torch.zeros((N, out_channels, H, W), device=device, dtype=test_out.dtype)

        # 4) define a helper that just calls the model
        def _process(chunk: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return model(chunk)

        # 5) chunk & stitch
        with torch.no_grad():
            output = process_in_chuncks(
                padded_image,
                output,
                _process,
                list(self.patch_size),
                list(self.patch_margin),
            )

        # 6) remove exactly the same padding we added at the top
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, : image.shape[2], : image.shape[3]]

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
"""
Dataloader module for wrapping GeneralizedDataset.

This module provides functionality to load and configure datasets for training, validation, and testing.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# Import the existing GeneralizedDataset
from core.general_dataset import GeneralizedDataset, custom_collate_fn, worker_init_fn


class SegmentationDataModule((pl.LightningDataModule)):
    """
    DataModule for segmentation tasks that wraps GeneralizedDataset.
    
    This class handles dataset creation and dataloader configuration for training,
    validation, and testing, ensuring that validation/testing use full images without cropping.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SegmentationDataModule.
        
        Args:
            config: Configuration dictionary with dataset parameters
        """
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.logger = logging.getLogger(__name__)
        
        # Extract important config parameters
        self.root_dir = config.get("root_dir")
        self.split_mode = config.get("split_mode", "folder")  # "folder" or "kfold"
        self.fold = config.get("fold", 0)
        self.num_folds = config.get("num_folds", 5)
        self.batch_size = {
            "train": config.get("train_batch_size", 8),
            "val": config.get("val_batch_size", 1),
            "test": config.get("test_batch_size", 1)
        }
        self.num_workers = config.get("num_workers", 4)
        self.pin_memory = config.get("pin_memory", True)
        
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for the specified stage.
        
        Args:
            stage: Stage to set up ('fit' or 'test')
        """
        if stage == 'fit' or stage is None:
            # Training dataset
            train_config = self._get_dataset_config("train")
            self.train_dataset = GeneralizedDataset(train_config)
            
            # Validation dataset - ensure no cropping in validation
            val_config = self._get_dataset_config("valid")
            val_config["validate_road_ratio"] = False  # Don't filter patches by road content
            # For validation, we want to process full images, not crops
            self.val_dataset = GeneralizedDataset(val_config)
            
        if stage == 'test' or stage is None:
            # Test dataset - ensure no cropping in test
            test_config = self._get_dataset_config("test")
            test_config["validate_road_ratio"] = False  # Don't filter patches by road content
            # For testing, we want to process full images, not crops
            self.test_dataset = GeneralizedDataset(test_config)
    
    def _get_dataset_config(self, split: str) -> Dict[str, Any]:
        """
        Get configuration for a specific dataset split.
        
        Args:
            split: Dataset split ('train', 'valid', or 'test')
            
        Returns:
            Configuration dictionary for the specified split
        """
        config = self.config.copy()
        config["split"] = split
        
        # Set split-specific parameters
        if split in ("valid", "test"):
            # For validation and test, we want to process full images when possible
            config["validate_road_ratio"] = False
        
        if self.split_mode == "kfold":
            config["use_splitting"] = True
            config["fold"] = self.fold
            config["num_folds"] = self.num_folds
        
        return config
    
    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader.
        
        Returns:
            DataLoader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Create validation dataloader.
        
        Returns:
            DataLoader for validation
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        Create test dataloader.
        
        Returns:
            DataLoader for testing
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=self.pin_memory,
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
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the mixed loss.
        
        Args:
            y_pred: Predicted values
            y_true: Ground truth values
            
        Returns:
            Tensor containing the calculated loss
        """
        primary_loss_value = self.primary_loss(y_pred, y_true)
        
        # If secondary loss is not configured or hasn't been activated yet, return only primary loss
        if self.secondary_loss is None or self.current_epoch < self.start_epoch:
            return primary_loss_value
        
        # Calculate mixed loss
        secondary_loss_value = self.secondary_loss(y_pred, y_true)
        return (1 - self.alpha) * primary_loss_value + self.alpha * secondary_loss_value

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
# metrics/connected_components.py
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
        greater_is_one=True,
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
        self.greater_is_one = bool(greater_is_one)
    
    def _bin(self, arr: np.ndarray) -> np.ndarray:
        return (arr >  self.threshold) if self.greater_is_one else \
               (arr <  self.threshold)
    
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
        greater_is_one=True
    ):
        super().__init__()
        # Force numerical types
        self.threshold     = float(threshold)
        self.eps           = float(eps)
        self.multiclass    = bool(multiclass)
        self.zero_division = float(zero_division)
        self.greater_is_one = bool(greater_is_one)

    def _binarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.greater_is_one:
            return (x >  self.threshold).float()
        else:
            return (x <  self.threshold).float()

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
        greater_is_one=True
    ):
        super().__init__()
        self.threshold     = float(threshold)
        self.eps           = float(eps)
        self.multiclass    = bool(multiclass)
        self.zero_division = float(zero_division)
        self.greater_is_one = bool(greater_is_one)

    def _binarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.greater_is_one:
            return (x >  self.threshold).float()
        else:
            return (x <  self.threshold).float()
        
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
    greater_is_one=True
):
    def _bin(x): return (x > threshold) if greater_is_one else (x < threshold)

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
        greater_is_one=True
    ):
        super().__init__()
        self.threshold = threshold
        self.angle_range = angle_range
        self.max_nodes = max_nodes
        self.max_snap_dist = max_snap_dist
        self.allow_renaming = allow_renaming
        self.min_path_length = min_path_length
        self.greater_is_one = bool(greater_is_one)

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
            greater_is_one=self.greater_is_one,
        )
        return torch.tensor(scores.mean(), device=y_pred.device)


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
# configs/main.yaml
# ------------------------------------
# Main configuration file for segmentation experiments
# This file references all sub-configs and sets high-level training parameters

# Sub-config references
dataset_config: "massroads.yaml"
model_config: "baseline.yaml"
loss_config: "mixed_topo.yaml"
metrics_config: "segmentation.yaml"
inference_config: "chunk.yaml"

# Output directory
output_dir: "outputs/baseline_unet_massroads"

# Trainer configuration
trainer:
  num_sanity_val_steps: 0
  max_epochs: 10000
  check_val_every_n_epoch: 10      # Validate every N epochs (this is redundant with val_check_interval=1.0)
  skip_validation_until_epoch: 0  # Skip validation until this epoch
  log_every_n_steps: 1            # log metrics every step
  train_metrics_every_n_epochs: 1 # compute/log train metrics every epoch
  val_metrics_every_n_epochs: 1  # compute/log val   metrics every epoch
  save_gt_pred_val_test_every_n_epochs: 10  # Save GT+pred every 10 epochs
  save_gt_pred_val_test_after_epoch: 0      # Start saving after epoch 0
  # save_gt_pred_max_samples: 3            # No limit on samples (or set an integer)
  save_checkpoints_every_n_epochs: 1

  num_samples_plot: 5
  cmap_plot: "coolwarm"

  extra_args:
    accelerator: "auto" 
    precision: 32  
    deterministic: false
    # gradient_clip_val: 1.0
    # accumulate_grad_batches: 1
    # max_time: "24:00:00"



# Optimizer configuration
optimizer:
  name: "Adam"
  params:
    lr: 0.0001
    weight_decay: 0.0001
  
  # Optional learning rate scheduler
  scheduler:
    # name: "ReduceLROnPlateau"
    # params:
    #   patience: 10
    #   factor: 0.5
    #   monitor: "val_loss"
    #   mode: "min"
    #   min_lr: 0.00001
    name: "LambdaLR"
    params:
      lr_decay_factor: 0.00001

target_x: "image_patch"
target_y: "sdf_patch"



# ------------------------------------
# configs/metrics/segmentation.yaml
# ------------------------------------
# ----------------------------------------
# Segmentation-metrics configuration
# ----------------------------------------

metrics:

  # Dice
  - alias: dice
    path: metrics.dice
    class: ThresholdedDiceMetric
    params:
      threshold: 0          # 0 splits neg/pos
      greater_is_one: false # neg < 0  -> road = 1
      eps: 1e-6
      multiclass: false
      zero_division: 1.0

  # IoU
  - alias: iou
    path: metrics.iou
    class: ThresholdedIoUMetric
    params:
      threshold: 0
      greater_is_one: false
      eps: 1e-6
      multiclass: false     
      zero_division: 1.0

  # Connected-components quality
  - alias: ccq
    path: metrics.connected_components
    class: ConnectedComponentsQuality
    params:
      min_size: 5
      tolerance: 5           # centroid distance in px
      threshold: 0
      greater_is_one: false

  # APLS
  - alias: apls
    path: metrics.apls
    class: APLS
    params:
      threshold: 0
      greater_is_one: false
      angle_range: [135, 225]
      max_nodes: 1000
      max_snap_dist: 25
      allow_renaming: true
      min_path_length: 10

# How often to compute each metric
train_frequencies:   {dice: 1,  iou: 1,  ccq: 10, apls: 25}
val_frequencies:     {dice: 1,  iou: 1,  ccq: 5,  apls: 10}


# ------------------------------------
# configs/loss/mixed_topo.yaml
# ------------------------------------
# Mixed Topological Loss Configuration

# Primary loss (used from the beginning)
primary_loss:
  class: "MSELoss"  # Built-in PyTorch loss
  params: {}

# # Secondary loss (activated after start_epoch)
secondary_loss:
  path: "losses.chamfer_class"  # Path to the module containing the loss
  class: "ChamferBoundarySDFLoss"  # Name of the loss class
  params:
    update_scale: 1.0
    dist_threshold: 3.0
    w_inject: 1.0
    w_pixel: 1.0

# # Mixing parameters
alpha: 1  # Weight for the secondary loss (0 to 1)
start_epoch: 10000  # Epoch to start using the secondary loss



 

# ------------------------------------
# configs/inference/chunk.yaml
# ------------------------------------
# Chunked Inference Configuration

# Patch size for inference [height, width]
# Patches of this size will be processed independently 
# and then reassembled to form the full output
patch_size: [256, 256]

# Patch margin [height, width]
# Margin of overlap between patches to avoid edge artifacts
# The effective stride will be (patch_size - 2*patch_margin)
patch_margin: [50, 50]


# ------------------------------------
# configs/dataset/massroads.yaml
# ------------------------------------
# Massachusetts Roads Dataset Configuration

# Dataset root directory
root_dir: "/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset"
# root_dir: "/cvlabdata2/cvlab/home/oner/Elyar/datasets/dataset"

# Dataset split mode: "folder" or "kfold"
split_mode: "folder"  # Uses folder structure for splits

# K-fold configuration (used if split_mode is "kfold")
# fold: 0  # Current fold
# num_folds: 3  # Total number of folds

# Split ratios (used if split_mode is "folder" with "source_folder")
# split_ratios:
  # train: 0.7
  # valid: 0.15
  # test: 0.15

use_splitting: false

# Source folder (used if split_mode is "folder" with split_ratios)
# source_folder: 'train'

# Batch sizes
train_batch_size: 1 #64
val_batch_size: 1  # Usually 1 for full-image validation
test_batch_size: 1  # Usually 1 for full-image testing

# Patch and crop settings
patch_size: 512  # Size of training patches
small_window_size: 8  # Size of window to check for variation
validate_road_ratio: false  # Validate patch has enough road content
threshold: 0.05  # Minimum road ratio threshold

# Data loading settings
num_workers: 4
pin_memory: true

# Modality settings
modalities:
  image: "sat"  # Satellite imagery folder
  label: "label"  # Road map folder
  # distance: "distance"  # Distance transform folder
  sdf: "sdf"  # Signed distance function folder

# Distance transform settings
# distance_threshold: 20.0

# Signed distance function settings
sdf_iterations: 3
sdf_thresholds: [-20, 20]

# Augmentation settings
augmentations:
  - "flip_h"
  - "flip_v"
  - "rotation"

# Misc settings
# max_images: null  # No limit (set a number to limit images loaded)
max_images: 2 #200  # No limit (set a number to limit images loaded)
# max_attempts: 10  # Maximum attempts for finding valid patches
save_computed: true  # Save computed distance maps and SDFs
verbose: false  # Verbose output
seed: 42  # Random seed

# ------------------------------------
# configs/model/baseline.yaml
# ------------------------------------
# TopoTokens Model Configuration

# Model path and class
path: "models.base_models"  # Path to the module containing the model
class: "UNet"  # Name of the model class

# Model parameters
params:
  three_dimensional: False
  m_channels: 32
  n_convs: 2
  n_levels: 3
  dropout: 0.1
  batch_norm: True
  upsampling: "bilinear"
  pooling: "max"
  in_channels: 3
  out_channels: 1
  apply_final_relu: False
  

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

