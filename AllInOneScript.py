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
    PredictionSaver
)
from core.logger import setup_logger
from core.checkpoint import CheckpointManager
from core.utils import yaml_read, mkdir

from seglit_module import SegLitModule


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

    # --- trainer params ---
    trainer_cfg            = main_cfg.get("trainer", {})
    max_epochs             = trainer_cfg.get("max_epochs", 100)
    val_check_interval     = trainer_cfg.get("val_check_interval", 1.0)
    skip_valid_until_epoch = trainer_cfg.get("skip_validation_until_epoch", 0)
    
    # Track metrics frequency from config (for consistent visualization even if not all shown in progress bar)
    train_metrics_every_n_epochs = trainer_cfg.get("train_metrics_every_n_epochs", 1)
    val_metrics_every_n_epochs = trainer_cfg.get("val_metrics_every_n_epochs", 1)
    
    # Get per-metric frequencies if defined
    train_metric_frequencies = metrics_cfg.get("train_frequencies", {})
    val_metric_frequencies = metrics_cfg.get("val_frequencies", {})

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

    # --- dynamic batch keys ---
    input_key  = main_cfg.get("target_x", "image_patch")
    target_key = main_cfg.get("target_y", "label_patch")

    # --- lightning module ---
    logger.info("Creating Lightning module...")
    lit = SegLitModule(
        model=model,
        loss_fn=mixed_loss,
        metrics=metric_list,
        optimizer_config=main_cfg["optimizer"],
        inference_config=inference_cfg,
        input_key=input_key,
        target_key=target_key,
        train_metrics_every_n_epochs=train_metrics_every_n_epochs,
        val_metrics_every_n_epochs=val_metrics_every_n_epochs,
        train_metric_frequencies=train_metric_frequencies,
        val_metric_frequencies=val_metric_frequencies,
    )

    # --- callbacks ---
    callbacks: List[pl.Callback] = []
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    mkdir(ckpt_dir)
    
    # Add BestMetricCheckpoint callback to save best models for each metric
    callbacks.append(BestMetricCheckpoint(
        dirpath=ckpt_dir,
        metric_names=list(metric_list.keys()),
        mode="max",
        save_last=True,
    ))
    
    # Add PredictionLogger to visualize predictions
    callbacks.append(PredictionLogger(
        log_dir=os.path.join(output_dir, "predictions"),
        log_every_n_epochs=trainer_cfg.get("log_every_n_epochs", 1),
        max_samples=4
    ))
    
    # Add SamplePlotCallback to monitor sample predictions during training
    callbacks.append(SamplePlotCallback(
        num_samples=trainer_cfg.get("num_samples_plot", 3)
    ))
    
    # Add LearningRateMonitor to track learning rate changes
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    
    # Archive code if not resuming
    if not args.resume:
        code_dir = os.path.join(output_dir, "code")
        mkdir(code_dir)
        callbacks.append(ConfigArchiver(
            output_dir=code_dir,
            project_root=os.path.dirname(os.path.abspath(__file__))
        ))
    
    # Skip validation for early epochs if needed
    if skip_valid_until_epoch > 0:
        callbacks.append(SkipValidation(skip_until_epoch=skip_valid_until_epoch))

    pred_save_dir = os.path.join(output_dir, "saved_predictions")
    mkdir(pred_save_dir)
    callbacks.append(PredictionSaver(
        save_dir=pred_save_dir,
        save_every_n_epochs=trainer_cfg.get("save_gt_pred_val_test_every_n_epochs", 5),
        save_after_epoch=trainer_cfg.get("save_gt_pred_val_test_after_epoch", 0),
        max_samples=trainer_cfg.get("save_gt_pred_max_samples", 4)
    ))

    # --- trainer & logger ---
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    trainer_kwargs = {
        "max_epochs": max_epochs,
        "callbacks": callbacks,
        "logger": tb_logger,
        "val_check_interval": val_check_interval,
        "check_val_every_n_epoch": trainer_cfg.get("val_every_n_epochs", 1),  # Validate every N epochs
        **trainer_cfg.get("extra_args", {}),
    }
    if args.resume:
        trainer_kwargs["resume_from_checkpoint"] = args.resume

    trainer = pl.Trainer(**trainer_kwargs)

    # --- run ---
    if args.test:
        logger.info("Running test...")
        trainer.test(lit, datamodule=dm)
    else:
        logger.info("Running training...")
        trainer.fit(lit, datamodule=dm)

        # test best checkpoint
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

        self.log("train_loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))

        y_int = y.long()
        for name, metric in self.metrics.items():
            freq = self.train_metric_frequencies.get(name, self.train_freq)
            if self.current_epoch % freq == 0:
                self.log(f"train_{name}", metric(y_hat, y_int),
                         prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.input_key].float()
        y = batch[self.target_key].float()
        if x.dim() == 3:
            x, y = x.unsqueeze(0), y.unsqueeze(0)

        # chunked inference (with built-in padding)
        with torch.no_grad():
            y_hat = self.validator.run_chunked_inference(self.model, x)

        # resize if needed
        if y_hat.shape[2:] != y.shape[2:]:
            y_hat = F.interpolate(y_hat, size=y.shape[2:], mode='bilinear', align_corners=False)

        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        y_int = y.long()
        for name, metric in self.metrics.items():
            freq = self.val_metric_frequencies.get(name, self.val_freq)
            if self.current_epoch % freq == 0:
                self.log(f"val_{name}", metric(y_hat, y_int),
                         prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        return {"predictions": y_hat, "val_loss": loss}

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

        y_int = y.long()
        for name, metric in self.metrics.items():
            self.log(f"test_{name}", metric(y_hat, y_int),
                     prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

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

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    pass

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
            if os.path.exists(config_path):
                with open(config_path, "r") as config_file:
                    saved_config = json.load(config_file)
                    sdf_comp_again = self.sdf_iterations != saved_config.get("sdf_iterations", None)

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
            self.logger.debug(f"Padded image from ({H}, {W}) to ({H + pad_h}, {W + pad_w})")
        
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
        Uses the exact Road_2D_EEF process_in_chunks approach with robust size handling.
        
        Args:
            model: The model to use for inference
            image: Input image tensor (N, C, H, W)
            device: Device to run inference on (default: None, uses model's device)
            
        Returns:
            Output tensor with predictions for the full image
        """
        if device is None:
            device = next(model.parameters()).device
            
        model.eval()
        image = image.to(device)
        
        # Store original dimensions
        original_shape = image.shape
        
        # Pad image to ensure valid dimensions for UNet
        # Use 16 as divisor for UNet with 3-4 levels (2^4 = 16)
        padded_image, padding = self._pad_to_valid_size(image, divisor=16)
        
        # Get padded image dimensions
        N, C, H, W = padded_image.shape
        
        # Create empty output tensor - determine output channels first
        with torch.no_grad():
            # Create a small test patch to determine output channels
            test_h = min(H, self.patch_size[0] + 2 * self.patch_margin[0])
            test_w = min(W, self.patch_size[1] + 2 * self.patch_margin[1])
            test_patch = padded_image[:, :, :test_h, :test_w]
            test_output = model(test_patch)
            out_channels = test_output.shape[1]
            
        # Initialize output tensor (N, out_channels, H, W)
        output = torch.zeros((N, out_channels, H, W), device=device, dtype=test_output.dtype)
        
        # Define the process function that will be called for each chunk
        def process_chunk(chunk):
            with torch.no_grad():
                return model(chunk)
        
        # Use the original process_in_chuncks function
        with torch.no_grad():
            output = process_in_chuncks(
                padded_image, 
                output, 
                process_chunk, 
                list(self.patch_size), 
                list(self.patch_margin)
            )
        
        # Remove padding to restore original dimensions
        output = self._remove_padding(output, padding)
        
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
# core/callbacks.py
# ------------------------------------
"""
Callbacks module for PyTorch Lightning training.

This module provides custom callbacks for checkpointing, visualization, 
code archiving, and validation control.
"""

import os
import io
import logging
import shutil
import zipfile
from glob import glob
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class SamplePlotCallback(pl.Callback):
    """
    After each training & validation epoch, log up to `num_samples`
    examples as figures:
      - train/samples: [input | ground-truth | prediction]
      - val/samples:   [input | ground-truth | prediction]
    """
    def __init__(self, num_samples: int = 5):
        super().__init__()
        self.num_samples = num_samples
        self._reset_buffers()

    def _reset_buffers(self):
        self._images   = []   # list of torch.Tensor batches
        self._gts      = []
        self._preds    = []
        self._collected = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset_buffers()

    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset_buffers()

    def _capture(self, batch, pl_module):
        x = batch[pl_module.input_key].float().to(pl_module.device)
        y = batch[pl_module.target_key].float().to(pl_module.device)
        if y.ndim == 3:
            y = y.unsqueeze(1)
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.sigmoid(logits)
        return x.cpu(), y.cpu(), preds.cpu()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        if self._collected >= self.num_samples:
            return
        x, y, preds = self._capture(batch, pl_module)
        remaining = self.num_samples - self._collected
        take = min(remaining, x.size(0))
        self._images.append(x[:take])
        self._gts.append(y[:take])
        self._preds.append(preds[:take])
        self._collected += take

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        if self._collected >= self.num_samples:
            return
        x, y, preds = self._capture(batch, pl_module)
        remaining = self.num_samples - self._collected
        take = min(remaining, x.size(0))
        self._images.append(x[:take])
        self._gts.append(y[:take])
        self._preds.append(preds[:take])
        self._collected += take

    def _plot_and_log(self, tag, trainer):
        # concatenate everything we gathered
        imgs  = torch.cat(self._images, 0)
        gts   = torch.cat(self._gts,    0)
        preds = torch.cat(self._preds,  0)
        n = imgs.size(0)

        fig, axes = plt.subplots(n, 3, figsize=(3*3, n*3), tight_layout=True)
        if n == 1:
            axes = axes[None, :]

        for i in range(n):
            img = imgs[i].permute(1,2,0)
            gt  = gts[i,0]
            pr  = preds[i,0]

            axes[i,0].imshow(img); axes[i,0].set_title("input"); axes[i,0].axis("off")
            axes[i,1].imshow(gt,  cmap="gray");       axes[i,1].set_title("gt");    axes[i,1].axis("off")
            axes[i,2].imshow(pr,  cmap="gray");       axes[i,2].set_title("pred");  axes[i,2].axis("off")

        trainer.logger.experiment.add_figure(f"{tag}/samples", fig, global_step=trainer.current_epoch)
        plt.close(fig)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._collected > 0:
            self._plot_and_log("train", trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._collected > 0:
            self._plot_and_log("val", trainer)


class PredictionLogger(Callback):
    """
    Callback to log input/prediction/ground truth visualization during validation.
    Accumulates up to `max_samples` across batches and saves one grid per epoch.
    """
    def __init__(self, log_dir: str, log_every_n_epochs: int = 1, max_samples: int = 4):
        super().__init__()
        self.log_dir = log_dir
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)
        self._reset_buffers()

    def _reset_buffers(self):
        self._images = []
        self._gts = []
        self._preds = []
        self._collected = 0
        self._logged_this_epoch = False

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self._reset_buffers()
        else:
            self._logged_this_epoch = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self._logged_this_epoch:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Pull images and labels by dynamic key:
        x = batch[pl_module.input_key]
        y_true = batch[pl_module.target_key]
        y_pred = outputs["predictions"]

        x = x.detach().cpu()
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()

        remaining = self.max_samples - self._collected
        take = min(remaining, x.shape[0])

        self._images.append(x[:take])
        self._gts.append(y_true[:take])
        self._preds.append(y_pred[:take])
        self._collected += take

        if self._collected >= self.max_samples:
            imgs = torch.cat(self._images, dim=0)
            gts  = torch.cat(self._gts,    dim=0)
            preds= torch.cat(self._preds,  dim=0)

            os.makedirs(self.log_dir, exist_ok=True)
            filename = os.path.join(
                self.log_dir,
                f"pred_epoch_{trainer.current_epoch:06d}.png"
            )

            fig, axes = plt.subplots(
                self.max_samples, 3,
                figsize=(12, 4 * self.max_samples)
            )

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
                if gts.shape[1] == 1:
                    ax.imshow(gts[i, 0], cmap='gray')
                else:
                    mask = torch.argmax(gts[i], dim=0)
                    ax.imshow(mask, cmap='tab20')
                ax.set_title('Ground Truth')
                ax.axis('off')

                # Prediction
                ax = axes[i, 2]
                if preds.shape[1] == 1:
                    ax.imshow(preds[i, 0], cmap='gray')
                else:
                    pmask = torch.argmax(preds[i], dim=0)
                    ax.imshow(pmask, cmap='tab20')
                ax.set_title('Prediction')
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close(fig)

            self.logger.info(f"Saved prediction visualization: {filename}")
            self._logged_this_epoch = True


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
        
        # Check each metric
        for metric in self.metric_names:
            metric_key = f"val_{metric}"
            if metric_key in trainer.callback_metrics:
                current_value = trainer.callback_metrics[metric_key].item()
                
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
            trainer.current_epoch % self.last_k == 0 or  # Every k epochs
            trainer.current_epoch == trainer.max_epochs - 1  # Last epoch
        ):
            filename = "last.ckpt"
            filepath = os.path.join(self.dirpath, filename)
            trainer.save_checkpoint(filepath)
            self.logger.info(f"Saved last checkpoint at epoch {trainer.current_epoch}")




class ConfigArchiver(Callback):
    """
    Callback to archive configuration files and source code at the start of training.
    
    This callback creates a zip file containing:
    - All configuration files
    - Source code files (train.py, core/, models/, losses/, metrics/)
    """
    
    def __init__(
        self,
        output_dir: str,
        project_root: str
    ):
        """
        Initialize the ConfigArchiver callback.
        
        Args:
            output_dir: Directory to save the archive to
            project_root: Root directory of the project containing the code to archive
        """
        super().__init__()
        self.output_dir = output_dir
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
    
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Archive configuration and source code at the start of training.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = pl_module.current_epoch
        archive_name = os.path.join(self.output_dir, f"code_snapshot_{timestamp}.zip")
        
        self.logger.info(f"Creating code archive: {archive_name}")
        
        with zipfile.ZipFile(archive_name, 'w') as zipf:
            # Archive configurations
            config_dir = os.path.join(self.project_root, 'configs')
            for root, _, files in os.walk(config_dir):
                for file in files:
                    if file.endswith('.yaml'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.project_root)
                        zipf.write(file_path, arcname)
            
            # Archive source code
            # train.py
            train_path = os.path.join(self.project_root, 'train.py')
            if os.path.exists(train_path):
                zipf.write(train_path, 'train.py')
            
            # Core modules
            core_dir = os.path.join(self.project_root, 'core')
            for root, _, files in os.walk(core_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.project_root)
                        zipf.write(file_path, arcname)
            
            # Models
            models_dir = os.path.join(self.project_root, 'models')
            if os.path.exists(models_dir):
                for root, _, files in os.walk(models_dir):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, self.project_root)
                            zipf.write(file_path, arcname)
            
            # Losses
            losses_dir = os.path.join(self.project_root, 'losses')
            if os.path.exists(losses_dir):
                for root, _, files in os.walk(losses_dir):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, self.project_root)
                            zipf.write(file_path, arcname)
            
            # Metrics
            metrics_dir = os.path.join(self.project_root, 'metrics')
            if os.path.exists(metrics_dir):
                for root, _, files in os.walk(metrics_dir):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, self.project_root)
                            zipf.write(file_path, arcname)
        
        self.logger.info(f"Code archive created: {archive_name}")


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

class PredictionSaver(Callback):
    """
    Callback to save ground truth and prediction tensors as NumPy arrays.
    Only saves after a specified starting epoch and at a specified frequency.
    """
    
    def __init__(
        self,
        save_dir: str,
        save_every_n_epochs: int = 5,
        save_after_epoch: int = 0,
        max_samples: int = 4
    ):
        """
        Initialize the PredictionSaver callback.
        
        Args:
            save_dir: Directory to save prediction data
            save_every_n_epochs: How often to save (every N epochs)
            save_after_epoch: Only start saving after this epoch
            max_samples: Maximum number of samples to save per epoch
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.save_after_epoch = save_after_epoch
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)
        
        # Buffers to collect samples
        self._reset_buffers()
    
    def _reset_buffers(self):
        """Reset the internal buffers that collect samples."""
        self._gts = []
        self._preds = []
        self._collected = 0
        self._saved_this_epoch = False
    
    def _should_save_this_epoch(self, epoch):
        """Determine if we should save data for this epoch."""
        if epoch < self.save_after_epoch:
            return False
            
        return (epoch - self.save_after_epoch) % self.save_every_n_epochs == 0
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Reset buffers at the start of a validation epoch."""
        current_epoch = trainer.current_epoch
        if self._should_save_this_epoch(current_epoch):
            self._reset_buffers()
        else:
            # Mark as already done for non-saving epochs
            self._saved_this_epoch = True
    
    def on_test_epoch_start(self, trainer, pl_module):
        """Reset buffers at the start of a test epoch."""
        # Always save during test
        self._reset_buffers()
    
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0
    ):
        """Collect validation batch results for later saving."""
        # Skip if we already collected enough samples this epoch or it's not a saving epoch
        if self._saved_this_epoch or self._collected >= self.max_samples:
            return
        
        current_epoch = trainer.current_epoch
        if not self._should_save_this_epoch(current_epoch):
            return
        
        # Get ground truth and prediction
        y_true = batch[pl_module.target_key]
        y_pred = outputs["predictions"]
        
        # Move to CPU and detach
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()
        
        # How many more samples we need
        remaining = self.max_samples - self._collected
        take = min(remaining, y_pred.shape[0])
        
        # Append the slices
        self._gts.append(y_true[:take])
        self._preds.append(y_pred[:take])
        self._collected += take
    
    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0
    ):
        """Collect test batch results for later saving."""
        # Skip if we already collected enough samples
        if self._saved_this_epoch or self._collected >= self.max_samples:
            return
        
        # Get ground truth and prediction
        y_true = batch[pl_module.target_key]
        y_pred = outputs["predictions"]
        
        # Move to CPU and detach
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()
        
        # How many more samples we need
        remaining = self.max_samples - self._collected
        take = min(remaining, y_pred.shape[0])
        
        # Append the slices
        self._gts.append(y_true[:take])
        self._preds.append(y_pred[:take])
        self._collected += take
    
    def _save_data(self, trainer, phase="val"):
        """Save collected data as NumPy arrays."""
        if not self._collected:
            return
            
        current_epoch = trainer.current_epoch
        
        # Concatenate buffers
        gts = torch.cat(self._gts, dim=0)
        preds = torch.cat(self._preds, dim=0)
        
        # Create output directory
        phase_dir = os.path.join(self.save_dir, phase)
        epoch_dir = os.path.join(phase_dir, f"epoch_{current_epoch:06d}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Convert to NumPy and save
        gts_numpy = gts.numpy()
        preds_numpy = preds.numpy()
        
        # Save as .npy files
        np.save(os.path.join(epoch_dir, "ground_truth.npy"), gts_numpy)
        np.save(os.path.join(epoch_dir, "predictions.npy"), preds_numpy)
        
        self.logger.info(f"Saved {self._collected} {phase} tensors to {epoch_dir}")
        self._saved_this_epoch = True
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Save data at the end of the validation epoch if conditions are met."""
        if not self._saved_this_epoch:
            self._save_data(trainer, "val")
            self._reset_buffers()
    
    def on_test_epoch_end(self, trainer, pl_module):
        """Save data at the end of the test epoch."""
        if not self._saved_this_epoch:
            self._save_data(trainer, "test")
            self._reset_buffers()

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
            worker_init_fn=worker_init_fn
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
            worker_init_fn=worker_init_fn
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
            worker_init_fn=worker_init_fn
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
the quality of connected components in the prediction compared to ground truth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import measure
from typing import Dict, List, Tuple, Set, Optional


class ConnectedComponentsQuality(nn.Module):
    """
    Connected Components Quality (CCQ) metric for evaluating segmentation quality.
    
    This metric considers both detection and shape accuracy of connected components
    in the predicted segmentation compared to the ground truth.
    """
    
    def __init__(
        self,
        min_size: int = 5,
        tolerance: int = 2,
        alpha: float = 0.5,
        eps: float = 1e-8
    ):
        """
        Initialize the CCQ metric.
        
        Args:
            min_size: Minimum component size to consider
            tolerance: Pixel tolerance for component matching
            alpha: Weight between detection score and shape score (0 to 1)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.min_size = min_size
        self.tolerance = tolerance
        self.alpha = alpha
        self.eps = eps
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the CCQ metric between predicted and ground truth masks.
        
        Args:
            y_pred: Predicted segmentation masks (B, 1, H, W)
            y_true: Ground truth segmentation masks (B, 1, H, W)
            
        Returns:
            Tensor containing the CCQ score (higher is better)
        """
        # Process each item in the batch
        batch_size = y_pred.shape[0]
        scores = []
        
        for i in range(batch_size):
            # Convert tensors to numpy arrays
            pred = (y_pred[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
            true = (y_true[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
            
            # Skip empty masks
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
            
            # Filter small components
            true_props = [prop for prop in true_props if prop.area >= self.min_size]
            pred_props = [prop for prop in pred_props if prop.area >= self.min_size]
            
            # If no components left after filtering
            if not true_props:
                if not pred_props:
                    scores.append(1.0)  # Both have no significant components
                else:
                    scores.append(0.0)  # True has no components but pred does
                continue
                
            if not pred_props:
                scores.append(0.0)  # Pred has no components but true does
                continue
            
            # Match components
            matches = self._match_components(true_props, pred_props)
            
            # Calculate detection score (TP / (TP + FP + FN))
            tp = len(matches)
            fp = max(0, len(pred_props) - tp)
            fn = max(0, len(true_props) - tp)
            
            detection_score = tp / (tp + fp + fn + self.eps)
            
            # Calculate shape score based on IoU of matched components
            shape_scores = []
            for true_idx, pred_idx in matches:
                true_mask = (true_labels == true_props[true_idx].label).astype(np.uint8)
                pred_mask = (pred_labels == pred_props[pred_idx].label).astype(np.uint8)
                
                intersection = np.sum(true_mask & pred_mask)
                union = np.sum(true_mask | pred_mask)
                
                iou = intersection / (union + self.eps)
                shape_scores.append(iou)
            
            # Average shape score
            shape_score = np.mean(shape_scores) if shape_scores else 0.0
            
            # Combined score
            combined_score = self.alpha * detection_score + (1 - self.alpha) * shape_score
            scores.append(combined_score)
        
        # Convert scores to tensor and return mean
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
        used_pred = set()
        
        for true_idx, true_prop in enumerate(true_props):
            best_dist = float('inf')
            best_pred_idx = None
            
            true_centroid = true_prop.centroid
            
            for pred_idx, pred_prop in enumerate(pred_props):
                if pred_idx in used_pred:
                    continue
                
                pred_centroid = pred_prop.centroid
                
                # Calculate Euclidean distance between centroids
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
# metrics/apls.py
# ------------------------------------
import numpy as np
import torch
import torch.nn as nn
from metrics.apls_core import apls

def compute_batch_apls(
    gt_masks,
    pred_masks,
    angle_range=(135, 225),
    max_nodes=500,
    max_snap_dist=4,
    allow_renaming=True,
    min_path_length=10
):
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
        gt_bin = (gt[i] > 0.5).astype(np.uint8)
        pr_bin = (pr[i] > 0.5).astype(np.uint8)

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
        angle_range=(135, 225),
        max_nodes=500,
        max_snap_dist=4,
        allow_renaming=True,
        min_path_length=10
    ):
        super().__init__()
        self.angle_range = angle_range
        self.max_nodes = max_nodes
        self.max_snap_dist = max_snap_dist
        self.allow_renaming = allow_renaming
        self.min_path_length = min_path_length

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        scores = compute_batch_apls(
            y_true,
            y_pred,
            angle_range=self.angle_range,
            max_nodes=self.max_nodes,
            max_snap_dist=self.max_snap_dist,
            allow_renaming=self.allow_renaming,
            min_path_length=self.min_path_length
        )
        return torch.tensor(scores.mean(), device=y_pred.device)


# ------------------------------------
# UsefulScripts/extract_paths.py
# ------------------------------------
import os
import pickle
import networkx as nx
import numpy as np
import itertools
import tempfile
import time
from multiprocessing import Pool, cpu_count

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_graph_txt(filename):
    G = nx.Graph()
    nodes = []
    edges = []
    i = 0
    switch = True
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 and switch:
                switch = False
                continue
            if switch:
                x, y = line.split(' ')
                G.add_node(i, pos=(float(x), float(y)))
                i += 1
            else:
                idx_node1, idx_node2 = line.split(' ')
                G.add_edge(int(idx_node1), int(idx_node2))
    return G

def save_graph_txt(G, filename):
    mkdir(os.path.dirname(filename))
    nodes = list(G.nodes())
    with open(filename, "w+") as file:
        for n in nodes:
            file.write("{:.6f} {:.6f}\r\n".format(G.nodes[n]['pos'][0], G.nodes[n]['pos'][1]))
        file.write("\r\n")
        for s, t in G.edges():
            file.write("{} {}\r\n".format(nodes.index(s), nodes.index(t)))

def txt_to_graph(filecontent):
    G = nx.Graph()
    lines = filecontent.strip().splitlines()
    switch = True  
    node_index = 0
    
    for line in lines:
        line = line.strip()
        if len(line) == 0 and switch:
            switch = False
            continue
        
        if switch:
            try:
                x, y = line.split()
                G.add_node(node_index, pos=(float(x), float(y)))
                node_index += 1
            except ValueError:
                raise ValueError(f"Error parsing node line: {line}")
        else:
            try:
                idx_node1, idx_node2 = line.split()
                G.add_edge(int(idx_node1), int(idx_node2))
            except ValueError:
                raise ValueError(f"Error parsing edge line: {line}")
    
    return G

def process_single_graph(graph_file, graphs_folder, temp_dir):
    start_time = time.time()  # Start time for each graph
    temp_file = os.path.join(temp_dir, f"{graph_file}.pkl")

    if os.path.exists(temp_file):
        with open(temp_file, "rb") as f:
            result = pickle.load(f)
        elapsed_time = time.time() - start_time
        print(f"Loaded cached graph {graph_file} in {elapsed_time:.2f} seconds.")
        return result

    graph_path = os.path.join(graphs_folder, graph_file)
    G = load_graph_txt(graph_path)

    for n, data in G.nodes(data=True):
        if 'pos' not in data and 'x' in data and 'y' in data:
            data['pos'] = (data['x'], data['y'])

    paths = []
    nodes = list(G.nodes())
    for s, t in itertools.combinations(nodes, 2):
        try:
            sp = nx.shortest_path(G, source=s, target=t, weight='length')
            s_coords = np.array(G.nodes[s].get('pos', (G.nodes[s].get('x'), G.nodes[s].get('y'))))
            t_coords = np.array(G.nodes[t].get('pos', (G.nodes[t].get('x'), G.nodes[t].get('y'))))
            paths.append({
                's_gt': s_coords,
                't_gt': t_coords,
                'shortest_path_gt': sp
            })
        except nx.NetworkXNoPath:
            continue

    result = (graph_file, paths)
    with open(temp_file, "wb") as f:
        pickle.dump(result, f)

    elapsed_time = time.time() - start_time
    print(f"Processed {graph_file} in {elapsed_time:.2f} seconds.")

    return result

def handle_graph_processing(dataset_name, base_dir, num_cores=4):
    start_total_time = time.time()  # Start measuring total time

    dataset_path = os.path.join(base_dir, dataset_name)
    graphs_folder = os.path.join(dataset_path, "graphs")
    output_path = os.path.join(dataset_path, f"{dataset_name}_gt_paths.pkl")
    temp_dir = os.path.join(tempfile.gettempdir(), f"{dataset_name}_temp")
    os.makedirs(temp_dir, exist_ok=True)

    if not os.path.exists(graphs_folder):
        print(f"Graphs folder not found at {graphs_folder}.")
        return

    graph_files = [f for f in os.listdir(graphs_folder) if f.endswith('.txt')]
    processed_files = {f for f in os.listdir(temp_dir) if f.endswith('.pkl')}
    remaining_files = [f for f in graph_files if f"{f}.pkl" not in processed_files]

    print(f"Processing {len(remaining_files)} remaining graphs in parallel using {num_cores} CPU cores...")

    gt_paths_dict = {}
    with Pool(processes=num_cores) as pool:
        results = pool.starmap(process_single_graph, [(gf, graphs_folder, temp_dir) for gf in remaining_files])

    for gf, paths in results:
        gt_paths_dict[gf] = paths

    with open(output_path, "wb") as f:
        pickle.dump(gt_paths_dict, f)

    total_elapsed_time = time.time() - start_total_time
    print(f"Saved all processed ground truth paths to {output_path}")
    print(f"Total processing time: {total_elapsed_time:.2f} seconds.")

# Usage with limited CPU cores
BaseDirDataset = '/home/ri/Desktop/Projects/ProcessedDatasets'
Dataset_name = 'CREMI'
handle_graph_processing(Dataset_name, BaseDirDataset, num_cores=4)

# Uncomment to process DRIVE dataset as well
# Dataset_name = 'DRIVE'
# handle_graph_processing(Dataset_name, BaseDirDataset, num_cores=4)


# ------------------------------------
# UsefulScripts/mass_clean_labels.py
# ------------------------------------
import os
import numpy as np
from tqdm import tqdm
import warnings

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    RASTERIO_AVAILABLE = True
except ImportError:
    from PIL import Image
    RASTERIO_AVAILABLE = False


def read_image(path):
    if RASTERIO_AVAILABLE:
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))  # -> (H, W, C)
    else:
        img = np.array(Image.open(path).convert("RGB"))
    return img


def read_label(path):
    if RASTERIO_AVAILABLE:
        with rasterio.open(path) as src:
            lbl = src.read(1)  # read first band as label
    else:
        lbl = np.array(Image.open(path))
    return lbl


def save_label(label_array, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if RASTERIO_AVAILABLE:
        height, width = label_array.shape
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': str(label_array.dtype),
        }
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(label_array, 1)
    else:
        from PIL import Image
        Image.fromarray(label_array).save(out_path)


def clean_labels(image_dir, label_dir, output_dir, window_size=8, white_threshold=250):
    """
    For each label pixel block (window_size x window_size),
    if the corresponding image block is 'white' (above white_threshold),
    set those label pixels to 0.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Gather image and label files by base name, ignoring extension differences
    image_files = {os.path.splitext(f)[0]: os.path.join(image_dir, f)
                   for f in os.listdir(image_dir)
                   if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))}

    label_files = {os.path.splitext(f)[0]: os.path.join(label_dir, f)
                   for f in os.listdir(label_dir)
                   if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))}

    common_keys = sorted(set(image_files.keys()) & set(label_files.keys()))

    for key in tqdm(common_keys, desc=f"Cleaning labels from {image_dir}"):
        image_path = image_files[key]
        label_path = label_files[key]
        output_path = os.path.join(output_dir, f"{key}.tif")  # force .tif output

        try:
            image = read_image(image_path)
            label = read_label(label_path)
        except Exception as e:
            print(f"Failed to load {key}: {e}")
            continue

        # If you have an alpha channel, ignore it
        if image.shape[-1] == 4:
            image = image[..., :3]

        H, W = image.shape[:2]
        cleaned = label.copy()

        for y in range(0, H, window_size):
            for x in range(0, W, window_size):
                y1 = min(y + window_size, H)
                x1 = min(x + window_size, W)
                patch = image[y:y1, x:x1]  # shape ~ (8, 8, 3)

                # is_every_pixel_white? => (pixel_value > white_threshold) in all channels
                if np.all(patch > white_threshold):
                    cleaned[y:y1, x:x1] = 0

        save_label(cleaned, output_path)

    print(f"Finished cleaning {len(common_keys)} matched label-image pairs.")


if __name__ == "__main__":
    # Example usage:
    clean_labels(
        image_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/train/sat",
        label_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/train/map",
        output_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/train/label",
        window_size=8,
        white_threshold=250  # can tune to ~240-255
    )

    clean_labels(
        image_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/valid/sat",
        label_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/valid/map",
        output_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/valid/label",
        window_size=8,
        white_threshold=250
    )

    clean_labels(
        image_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/test/sat",
        label_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/test/map",
        output_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/test/label",
        window_size=8,
        white_threshold=250
    )


# ------------------------------------
# UsefulScripts/mass_roads_dataset_downloader.py
# ------------------------------------
import os
import requests
def download_file(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"Downloaded: {save_path}")
        else:
            print(f"Failed to download: {url}")
    else:
        print(f"File already exists: {save_path}")

from bs4 import BeautifulSoup

def extract_hrefs_from_html(html_file):
    """
    Extracts all href attributes from <a> tags in an HTML file.
    
    Args:
        html_file (str): Path to the HTML file.
        output_file (str, optional): Path to save the extracted hrefs. If None, no file is saved.
    
    Returns:
        list: A list of href strings extracted from the HTML file.
    """
    # Load the HTML file
    with open(html_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')

    # Extract all href attributes from <a> tags
    hrefs = [a['href'] for a in soup.find_all('a', href=True)]

    return hrefs
    
dataset = {
    ("train","sat"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html',
    ("train","map"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html',
    ("valid","sat"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html',
    ("valid","map"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html',
    ("test", "sat"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/index.html',
    ("test", "map"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/index.html',
  
} 
BASE = "dataset"
for folders, url in dataset.items():
  os.makedirs(BASE, exist_ok=True)
  f1 = os.path.join(BASE, folders[0]) 
  f2 = os.path.join(f1, folders[1])
  os.makedirs(f1, exist_ok=True) 
  os.makedirs(f2, exist_ok=True)
  index = os.path.join(f2, "index.html")
  download_file(url, index)
  hrefs = extract_hrefs_from_html(index)
  for href in hrefs:
    download_file(href, os.path.join(f2, href.split('/')[-1]))


# ------------------------------------
# UsefulScripts/tlts.py
# ------------------------------------
import numpy as np
import networkx as nx
import queue
import itertools
from scipy.spatial.distance import euclidean

def find_connectivity(img, x, y, stop=None):
    
    _img = img.copy()   
    _img2 = img.copy()
    
    dy = [0, 0, 1, 1, 1, -1, -1, -1]
    dx = [1, -1, 0, 1, -1, 0, 1, -1]
    xs = []
    ys = []
    cs = []
    q = queue.Queue()
    if _img[y,x] == True:
        q.put((y,x))
    i = 0
    while q.empty() == False:
        i+=1
        v,u = q.get()
        xs.append(u)
        ys.append(v)
        adjacent = [(u,v)]
        if stop is not None and i==stop:
            return xs, ys, cs
        for k in range(8):
            yy = v + dy[k]
            xx = u + dx[k]            
            if _img[yy, xx] == True:
                _img[yy, xx] = False
                q.put((yy, xx))               
            if _img2[yy, xx] == True:
                adjacent.append((xx,yy))
        cs.append(adjacent)
    return xs, ys, cs 

def find_connectivity_3d(img, x, y, z, stop=None):
    
    _img = img.copy()   
    _img2 = img.copy()
    
    dx = [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dy = [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1]
    dz = [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]     
    xs = []
    ys = []
    zs = []
    cs = []
    q = queue.Queue()
    if _img[y,x,z] == True:
        q.put((x,y,z))
    i = 0
    while q.empty() == False:
        i+=1
        u,v,w = q.get()
        xs.append(u)
        ys.append(v)
        zs.append(w)
        adjacent = [(u,v,w)]
        if stop is not None and i==stop:
            return xs, ys, zs, cs
        for k in range(26):            
            xx = u + dx[k]  
            yy = v + dy[k]
            zz = w + dz[k]            
            if _img[yy, xx, zz] == True:
                _img[yy, xx, zz] = False
                q.put((xx,yy,zz))               
            if _img2[yy, xx, zz] == True:
                adjacent.append((xx,yy,zz))
        cs.append(adjacent)
    return xs, ys, zs, cs
    
def create_graph(skeleton):
    
    _skeleton = skeleton.copy()>0

    # make sure no pixel are active on the borders
    _skeleton[ 0, :] = False
    _skeleton[-1, :] = False
    _skeleton[ :, 0] = False
    _skeleton[ :,-1] = False

    css = []
    while True:
        ys, xs = np.where(_skeleton)
        if len(ys)==0:
            break
        _xs, _ys, _cs = find_connectivity(_skeleton, xs[0], ys[0], stop=None)
        css += _cs
        _skeleton[_ys,_xs] = False       
    
    graph = nx.Graph()

    for cs in css:
        for pos in cs:
            if not graph.has_node(pos):
                graph.add_node(pos, pos=np.array(pos))  
        '''   
        if len(cs)==2:
            us,vs = [cs[0]],[cs[1]]
            distances = [euclidean(cs[0],cs[1])]
        elif len(cs)==3:
            d1 = euclidean(cs[0],cs[1])
            d2 = euclidean(cs[1],cs[2])
            d3 = euclidean(cs[0],cs[2])
            if d1>d2 and d1>d3:
                us,vs = [cs[1],cs[0]],[cs[2],cs[2]] 
                distances = [d2,d3]
            if d2>d1 and d2>d3:
                us,vs = [cs[0],cs[0]],[cs[1],cs[2]]  
                distances = [d1,d3]
            if d3>d1 and d3>d2:
                us,vs = [cs[0],cs[1]],[cs[1],cs[2]]
                distances = [d1,d2]
        else:
        '''
        us,vs = [],[]
        for u,v in itertools.combinations(cs, 2):
            us += [u]
            vs += [v]                    
        distances = [euclidean(u,v) for u,v in zip(us,vs)] 

        for u,v,d in zip(us,vs,distances):
            if not graph.has_edge(u,v):
                '''
                if False:
                    try:
                        length = nx.shortest_path_length(graph,u,v)
                        if length>5:
                            graph.add_edge(u,v)
                    except:
                        graph.add_edge(u,v)      
                else:
                '''
                if d<1.42:
                    graph.add_edge(u,v)

    return graph

def create_graph_3d(skeleton):
    
    _skeleton = skeleton.copy()>0
    
    # make sure no pixel are active on the borders
    _skeleton[ 0, :, :] = False
    _skeleton[-1, :, :] = False
    _skeleton[ :, 0, :] = False
    _skeleton[ :,-1, :] = False   
    _skeleton[ :, :, 0] = False
    _skeleton[ :, :,-1] = False     

    css = []
    while True:
        ys, xs, zs = np.where(_skeleton)
        if len(ys)==0:
            break
        _xs, _ys, _zs, _cs = find_connectivity_3d(_skeleton, xs[0], ys[0], zs[0], stop=None)
        css += _cs
        _skeleton[_ys,_xs,_zs] = False       
    
    graph = nx.Graph()

    for cs in css:
        for pos in cs:
            if not graph.has_node(pos):
                graph.add_node(pos, pos=np.array(pos))  

        us,vs = [],[]
        for u,v in itertools.combinations(cs, 2):
            us += [u]
            vs += [v]                    
        distances = [euclidean(u,v) for u,v in zip(us,vs)] 

        for u,v,d in zip(us,vs,distances):
            if not graph.has_edge(u,v):
                if d<2.1:
                    graph.add_edge(u,v)

    return graph

def extract_gt_paths(graph_gt, N=100, min_path_length=10):

    cc_graphs = list(graph_gt.subgraph(c) for c in nx.connected_components(graph_gt))
    n_subgraph = len(cc_graphs)  
    
    total = 0

    paths = []
    for _ in range(N*1000):
        
        idx_sub = np.random.choice(np.arange(n_subgraph), 1)[0]
        graph = cc_graphs[idx_sub]
        
        nodes_gt = list(graph.nodes()) 
        n_nodes = len(nodes_gt)
        if n_nodes < 2:
            continue
    
        # randomly pick two node in the GT
        idx_s,idx_t = np.random.choice(np.arange(n_nodes), 2, replace=False)
        s_gt, t_gt = nodes_gt[idx_s], nodes_gt[idx_t]
        
        # search shortest path in GT
        try:
            shortest_path_gt = list(nx.shortest_path(graph, tuple(s_gt), tuple(t_gt)))
            #shortest_path_gt = list(nx.astar_path(graph, tuple(s_gt), tuple(t_gt)))
            length_line_gt = len(shortest_path_gt)
        except:
            # path not found
            continue 
            
        if length_line_gt<min_path_length:
            continue
            
        paths.append({"s_gt":s_gt, "t_gt":t_gt, "shortest_path_gt":shortest_path_gt})
        
        total += 1
        
        if total==N:
            break

    return paths
    
def toolong_tooshort_score(paths_gt, graph_pred, radius_match=5, length_deviation=0.05):
    """
    A higher-order CRF model for road network extraction
    Jan D. Wegner, Javier A. Montoya-Zegarra, Konrad Schindler
    2013
    
    These are
    computed in the following way: we randomly sample two
    points which lie both on the true and the estimated road
    network, and check whether the shortest path between the
    two points has the same length in both networks (up to a
    deviation of 5% to account for geometric uncertainty). We
    then keep repeating this procedure with different random
    points and record the percentages of correct, too short, too
    long and infeasible paths, until these percentages have converged. 
    Infeasible and too long paths indicate missing links,
    whereas too short ones indicate hallucinated connections.  
    
    """
      
    nodes_pred = np.array(graph_pred.nodes())
    idxs_pred = np.arange(len(nodes_pred))     

    counter_correct = 0
    counter_toolong = 0
    counter_tooshort = 0
    counter_infeasible = 0
    
    res = []  
    for path in paths_gt:
    
        # unpack GT path
        s_gt, t_gt = np.array(path["s_gt"]), np.array(path["t_gt"])
        shortest_path_gt = path["shortest_path_gt"]
        length_line_gt = len(shortest_path_gt)

        # match GT nodes in prediction
        nodes_radius_s = nodes_pred[np.linalg.norm(nodes_pred-s_gt[None], axis=1)<radius_match]
        nodes_radius_t = nodes_pred[np.linalg.norm(nodes_pred-t_gt[None], axis=1)<radius_match]
        if len(nodes_radius_s)==0 or len(nodes_radius_t)==0:
            counter_infeasible += 1
            res.append({"line_gt":shortest_path_gt, 
                        "line_pred":None,
                        "s_gt":s_gt,"t_gt":t_gt,
                        "s_pred":None, "t_pred":None,
                        "tooshort":False, "toolong":False,
                        "correct":False, "infeasible":True})
            continue
        s_pred = nodes_radius_s[np.linalg.norm(nodes_radius_s-s_gt[None], axis=1).argmin()]
        t_pred = nodes_radius_t[np.linalg.norm(nodes_radius_t-t_gt[None], axis=1).argmin()]
        
        # find shortest path in prediction
        try:
            shortest_path_pred = list(nx.shortest_path(graph_pred, tuple(s_pred), tuple(t_pred)))
            #shortest_path_pred = list(nx.astar_path(graph_pred, tuple(s_pred), tuple(t_pred)))
            length_line_pred = len(shortest_path_pred)
        except:
            # path not found
            counter_infeasible += 1
            res.append({"line_gt":shortest_path_gt, 
                        "line_pred":None,
                        "s_gt":s_gt,"t_gt":t_gt,
                        "s_pred":s_pred, "t_pred":t_pred,
                        "tooshort":False, "toolong":False,
                        "correct":False, "infeasible":True})            
            continue 

        # compare path lengths
        toolong, tooshort, correct = False,False,False        
        if length_line_pred>length_line_gt*(1+length_deviation):
            toolong = True
            counter_toolong += 1
        elif length_line_pred<length_line_gt*(1-length_deviation): 
            tooshort = True
            counter_tooshort += 1                   
        else:
            correct = True
            counter_correct += 1
            
        res.append({"line_gt":shortest_path_gt, 
                    "line_pred":shortest_path_pred,
                    "s_gt":s_gt,"t_gt":t_gt,
                    "s_pred":s_pred, "t_pred":t_pred,
                    "tooshort":tooshort, "toolong":toolong,
                    "correct":correct, "infeasible":False}) 
            
    total = len(paths_gt)
        
    return total, counter_correct/total, counter_tooshort/total, counter_toolong/total, counter_infeasible/total, res    

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
# tests/test_helpers.py
# ------------------------------------
"""
Unit tests for critical helper functions in the segmentation framework.
Run with: pytest -xvs test_helpers.py
"""

import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# Add parent directory to sys.path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the functions to test
from core.general_dataset import compute_distance_map, compute_sdf, custom_collate_fn
from core.utils import (
    noCrops,
    noCropsPerDim,
    cropInds,
    coord,
    coords,
    cropCoords,
    process_in_chuncks,
)


class TestDistanceMap:
    """Tests for compute_distance_map function"""

    def test_basic_functionality(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        distance = compute_distance_map(mask, None)

        assert distance[2, 2] == 0
        for i, j in [(0, 0), (0, 4), (4, 0), (4, 4)]:
            assert 2.8 < distance[i, j] < 2.9
        for i, j in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            assert distance[i, j] == 1

    def test_thresholding(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[3, 3] = 1
        distance = compute_distance_map(mask, 2.0)

        assert np.max(distance) <= 2.0
        assert distance[3, 3] == 0
        assert distance[2, 3] == 1
        assert distance[4, 3] == 1

    def test_binary_formats(self):
        mask_01 = np.zeros((5, 5), dtype=np.uint8); mask_01[2, 2] = 1
        mask_0255 = np.zeros((5, 5), dtype=np.uint8); mask_0255[2, 2] = 255
        d1 = compute_distance_map(mask_01, None)
        d2 = compute_distance_map(mask_0255, None)
        assert np.array_equal(d1, d2)

    def test_empty_mask(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        distance = compute_distance_map(mask, None)
        assert np.all(distance > 0)

    def test_full_mask(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        distance = compute_distance_map(mask, None)
        assert np.all(distance == 0)


class TestSignedDistanceFunction:
    """Tests for compute_sdf function"""

    def test_basic_functionality(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[3, 3] = 1
        sdf = compute_sdf(mask, sdf_iterations=1, sdf_thresholds=None)

        # Inside (where mask==1) should be negative
        assert sdf[3, 3] < 0
        # Far away should be positive
        assert sdf[0, 0] > 0

    def test_iterations_parameter(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[4, 4] = 1

        sdf1 = compute_sdf(mask, sdf_iterations=1, sdf_thresholds=None)
        sdf3 = compute_sdf(mask, sdf_iterations=3, sdf_thresholds=None)

        neg1 = np.sum(sdf1 < 0)
        neg3 = np.sum(sdf3 < 0)
        # More iterations should not shrink the "inside"—allow equal or larger
        assert neg3 >= neg1

    def test_thresholds_parameter(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[4, 4] = 1

        sdf = compute_sdf(mask, sdf_iterations=1, sdf_thresholds=[-2, 2])
        assert np.all(sdf >= -2)
        assert np.all(sdf <= 2)

    def test_binary_formats(self):
        mask_01 = np.zeros((5, 5), dtype=np.uint8); mask_01[2, 2] = 1
        mask_0255 = np.zeros((5, 5), dtype=np.uint8); mask_0255[2, 2] = 255
        s1 = compute_sdf(mask_01, sdf_iterations=1, sdf_thresholds=None)
        s2 = compute_sdf(mask_0255, sdf_iterations=1, sdf_thresholds=None)
        assert np.array_equal(s1, s2)


class TestCropFunctions:
    """Tests for the crop‐related utility functions"""

    def test_noCrops_basic(self):
        assert noCrops([100, 100], [50, 50], [5, 5], startDim=0) == 9

    def test_noCrops_tiny_image(self):
        assert noCrops([10, 10], [10, 10], [3, 3], startDim=0) == 1

    def test_noCropsPerDim(self):
        per, cum = noCropsPerDim([100, 200], [50, 50], [5, 5], startDim=0)
        assert per == [3, 5]
        assert cum == [15, 5, 1]

    def test_cropInds(self):
        cum = [12, 3, 1]
        assert cropInds(0, cum) == [0, 0]
        assert cropInds(3, cum) == [1, 0]
        assert cropInds(11, cum) == [3, 2]

    def test_coord(self):
        c, v = coord(2, 30, 5, 100)
        assert (c.start, c.stop) == (40, 70)
        assert (v.start, v.stop) == (5, 25)

        c, v = coord(4, 30, 5, 100)
        assert (c.start, c.stop) == (70, 100)
        # when hitting edge, valid region is trimmed (start=15) to avoid going out of bounds
        assert (v.start, v.stop) == (15, 30)

    def test_coords(self):
        cc, vc = coords([1, 2], [30, 30], [5, 5], [100, 100], 0)
        assert (cc[0].start, cc[0].stop) == (20, 50)
        assert (cc[1].start, cc[1].stop) == (40, 70)
        assert (vc[0].start, vc[0].stop) == (5, 25)
        assert (vc[1].start, vc[1].stop) == (5, 25)

    def test_cropCoords(self):
        cc, vc = cropCoords(7, [30, 40], [5, 5], [100, 200], 0)
        for sl in [*cc, *vc]:
            assert isinstance(sl, slice)
        assert 0 <= cc[0].start < cc[0].stop <= 100
        assert 0 <= cc[1].start < cc[1].stop <= 200
        assert 0 <= vc[0].start < vc[0].stop <= 30
        assert 0 <= vc[1].start < vc[1].stop <= 40


class TestProcessInChunks:
    """Tests for process_in_chuncks"""

    def test_basic_processing(self):
        inp = torch.ones((1, 3, 10, 10))
        out = torch.zeros((1, 1, 10, 10))
        def fn(x): return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]))
        res = process_in_chuncks(inp, out, fn, [5, 5], [1, 1])
        assert torch.all(res == 1)

    def test_chunking_logic(self):
        inp = torch.zeros((1, 3, 20, 20))
        out = torch.zeros((1, 1, 20, 20))
        def fn(x):
            b, c, h, w = x.shape
            t = torch.zeros((b, 1, h, w))
            for i in range(h):
                for j in range(w):
                    t[0, 0, i, j] = i + j
            return t
        res = process_in_chuncks(inp, out, fn, [10, 10], [2, 2])
        assert res[0, 0, 0, 0] == 0
        assert res[0, 0, 6, 0] == 6

    def test_shape_handling(self):
        inp = torch.ones((1, 3, 10, 10))
        out = torch.zeros((1, 1, 10, 10))
        def fn(x): return torch.ones((x.shape[0], x.shape[2], x.shape[3]))
        res = process_in_chuncks(inp, out, fn, [5, 5], [1, 1])
        assert torch.all(res == 1)

    def test_margin_handling(self):
        inp = torch.zeros((1, 1, 20, 20))
        for i in range(20):
            for j in range(20):
                inp[0, 0, i, j] = i * 100 + j
        out = torch.zeros((1, 1, 20, 20))
        def fn(x): return x
        res = process_in_chuncks(inp, out, fn, [10, 10], [2, 2])
        assert torch.all(res == inp)


class TestDataSplitting:
    """Tests for data splitting logic (ratio and k-fold)"""

    @pytest.fixture
    def mock_file_structure(self, tmp_path):
        """
        Create a mock dataset structure for testing:
        
        dataset/
          source/
            sat/  (images)
            map/  (labels)
        """
        root = tmp_path / "dataset"
        img_dir = root / "source" / "sat"
        lbl_dir = root / "source" / "map"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create 10 dummy .tif files in each
        for i in range(10):
            (img_dir / f"img_{i}.tif").write_text("dummy")
            (lbl_dir / f"img_{i}.tif").write_text("dummy")

        return root

    def test_ratio_splitting(self, mock_file_structure, monkeypatch):
        """Test ratio-based splitting logic"""
        from core.general_dataset import GeneralizedDataset

        # Prevent any random shuffle
        monkeypatch.setattr(np.random, "shuffle", lambda x: x)

        base_cfg = {
            "root_dir": str(mock_file_structure),
            "use_splitting": True,
            "source_folder": "source",
            "split_ratios": {"train": 0.6, "valid": 0.2, "test": 0.2},
            "modalities": {"image": "sat", "label": "map"},
        }

        # Create each split explicitly
        train_ds = GeneralizedDataset({**base_cfg, "split": "train"})
        valid_ds = GeneralizedDataset({**base_cfg, "split": "valid"})
        test_ds  = GeneralizedDataset({**base_cfg, "split": "test"})

        # Should be 6,2,2 images respectively
        assert len(train_ds.modality_files["image"]) == 6
        assert len(valid_ds.modality_files["image"]) == 2
        assert len(test_ds.modality_files["image"])  == 2

        # No overlap
        t = set(train_ds.modality_files["image"])
        v = set(valid_ds.modality_files["image"])
        s = set(test_ds.modality_files["image"])
        assert t.isdisjoint(v)
        assert t.isdisjoint(s)
        assert v.isdisjoint(s)

    def test_kfold_splitting(self, mock_file_structure, monkeypatch):
        """Test k-fold splitting logic"""
        from core.general_dataset import GeneralizedDataset

        # Mock KFold to return a single fixed split (first 8 train, last 2 valid)
        class MockKFold:
            def __init__(self, n_splits, shuffle, random_state):
                pass
            def split(self, X):
                return [(np.arange(8), np.arange(8, 10))]

        monkeypatch.setattr("sklearn.model_selection.KFold", MockKFold)

        base_cfg = {
            "root_dir": str(mock_file_structure),
            "use_splitting": True,
            "split_mode": "kfold",
            "num_folds": 5,
            "source_folder": "source",
            "modalities": {"image": "sat", "label": "map"},
        }

        # Train fold 0
        tr_cfg = {**base_cfg, "split": "train", "fold": 0}
        train_ds = GeneralizedDataset(tr_cfg)
        assert len(train_ds.modality_files["image"]) == 8

        # Valid fold 0
        va_cfg = {**base_cfg, "split": "valid", "fold": 0}
        valid_ds = GeneralizedDataset(va_cfg)
        assert len(valid_ds.modality_files["image"]) == 2


class TestCustomCollate:
    """Tests for custom_collate_fn"""

    def test_basic(self):
        batch = [
            {"image": torch.ones(3,10,10), "label": torch.zeros(1,10,10)},
            {"image": torch.ones(3,10,10), "label": torch.zeros(1,10,10)},
        ]
        c = custom_collate_fn(batch)
        assert c["image"].shape == (2,3,10,10)
        assert c["label"].shape == (2,1,10,10)

    def test_mixed_types(self):
        batch = [
            {"image": torch.ones(3,10,10), "meta": {"id":1}},
            {"image": torch.ones(3,10,10), "meta": {"id":2}},
        ]
        c = custom_collate_fn(batch)
        assert isinstance(c["meta"], list)
        assert c["meta"][0]["id"] == 1

    def test_filter_none(self):
        def imp(batch):
            batch = [b for b in batch if b is not None]
            if not batch:
                return {"image": torch.zeros(0,3,10,10)}
            return custom_collate_fn(batch)
        b = [None, {"image": torch.ones(3,10,10)}]
        c1 = imp(b)
        assert c1["image"].shape[0] == 1
        c2 = imp([None,None])
        assert c2["image"].shape[0] == 0


# ------------------------------------
# tests/smoke_test.py
# ------------------------------------
"""
Integration smoke test for the entire segmentation pipeline.
This creates a minimal synthetic dataset and runs a few training steps.
Run with: python smoke_test.py
"""

import os
import shutil
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core modules
from core.general_dataset import GeneralizedDataset, custom_collate_fn, worker_init_fn
from core.model_loader import load_model
from core.loss_loader import load_loss
from core.mix_loss import MixedLoss
from core.metric_loader import load_metrics
from core.dataloader import SegmentationDataModule
from seglit_module import SegLitModule


def create_synthetic_dataset(root_dir, num_samples=10, img_size=32):
    """Create a synthetic dataset with roads for testing"""
    logger.info(f"Creating synthetic dataset in {root_dir} with {num_samples} samples")
    
    # Create directory structure
    os.makedirs(os.path.join(root_dir, 'train', 'sat'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'train', 'map'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'valid', 'sat'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'valid', 'map'), exist_ok=True)
    
    # Create test images with simple road patterns
    for split in ['train', 'valid']:
        for i in range(num_samples):
            # Create image with random noise
            img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            
            # Create binary mask with horizontal and vertical lines (roads)
            mask = np.zeros((img_size, img_size), dtype=np.uint8)
            
            # Horizontal road
            h_pos = np.random.randint(5, img_size-5)
            mask[h_pos-2:h_pos+2, :] = 1
            
            # Vertical road
            v_pos = np.random.randint(5, img_size-5)
            mask[:, v_pos-2:v_pos+2] = 1
            
            # Add brightness to the roads in the image for realism
            for c in range(3):
                img[:, :, c] = np.where(mask > 0, 
                                        np.minimum(img[:, :, c] + 100, 255),
                                        img[:, :, c])
            
            # Save files
            np.save(os.path.join(root_dir, split, 'sat', f'img_{i}.npy'), img)
            np.save(os.path.join(root_dir, split, 'map', f'img_{i}.npy'), mask * 255)  # 0/255 format
    
    logger.info(f"Created {num_samples} samples each for train and validation")


class SimpleBinaryUNet(nn.Module):
    """A very simple UNet for testing purposes"""
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, stride=2)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 2, stride=2)
        )
        
        # Final layer
        self.final = nn.Conv2d(8, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        # Decoder
        d1 = self.dec1(e2)
        d2 = self.dec2(d1)
        
        # Final layer
        out = torch.sigmoid(self.final(d2))
        return out


def create_configs(dataset_path):
    """Create configuration dictionaries for testing"""
    # Dataset configuration
    dataset_config = {
        "root_dir": dataset_path,
        "split_mode": "folder",
        "patch_size": 16,
        "small_window_size": 2,
        "validate_road_ratio": False,  # Don't filter patches for quick testing
        "train_batch_size": 2,
        "val_batch_size": 1,
        "num_workers": 0,  # Use 0 for easier debugging
        "pin_memory": False,
        "modalities": {
            "image": "sat",
            "label": "map"
        }
    }
    
    # Model configuration using our simple test model
    model_config = {
        "simple_unet": True,  # Flag for our smoke test
        "in_channels": 3,
        "out_channels": 1
    }
    
    # Loss configuration
    loss_config = {
        "primary_loss": {
            "class": "BCELoss",
            "params": {}
        },
        "alpha": 1.0  # Only use primary loss
    }
    
    # Metrics configuration
    metrics_config = {
        "metrics": [
            {
                "alias": "dice",
                "path": "torchmetrics.classification",
                "class": "Dice",
                "params": {
                    "threshold": 0.5,
                    "zero_division": 1.0
                }
            }
        ]
    }
    
    # Inference configuration
    inference_config = {
        "patch_size": [16, 16],
        "patch_margin": [2, 2]
    }
    
    return dataset_config, model_config, loss_config, metrics_config, inference_config


def run_smoke_test():
    """Run a complete smoke test of the segmentation pipeline"""
    logger.info("Starting smoke test")
    
    # Create temporary directory for dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create synthetic dataset
        create_synthetic_dataset(tmp_dir, num_samples=5, img_size=32)
        
        # Create configurations
        dataset_config, model_config, loss_config, metrics_config, inference_config = create_configs(tmp_dir)
        
        # Set up data module
        logger.info("Setting up data module")
        data_module = SegmentationDataModule(dataset_config)
        data_module.setup()
        
        # Create model manually for smoke test
        logger.info("Creating model")
        if model_config.get("simple_unet", False):
            model = SimpleBinaryUNet(
                in_channels=model_config["in_channels"],
                out_channels=model_config["out_channels"]
            )
        else:
            model = load_model(model_config)
        
        # Create loss function
        logger.info("Creating loss function")
        primary_loss = load_loss(loss_config["primary_loss"])
        secondary_loss = None
        if "secondary_loss" in loss_config:
            secondary_loss = load_loss(loss_config["secondary_loss"])
        
        mixed_loss = MixedLoss(primary_loss, secondary_loss, loss_config.get("alpha", 0.5), 
                              loss_config.get("start_epoch", 0))
        
        # Create metrics
        logger.info("Loading metrics")
        metrics = load_metrics(metrics_config.get("metrics", []))
        
        # Create optimizer config
        optimizer_config = {
            "name": "Adam",
            "params": {"lr": 0.001}
        }
        
        # Create Lightning module
        logger.info("Creating Lightning module")
        lit_module = SegLitModule(
            model=model,
            loss_fn=mixed_loss,
            metrics=metrics,
            optimizer_config=optimizer_config,
            inference_config=inference_config
        )
        
        # Create a simple trainer for testing
        logger.info("Creating trainer")
        trainer = pl.Trainer(
            max_epochs=2,
            log_every_n_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True,
            accelerator="cpu"
        )
        
        # Run a few training steps
        logger.info("Running training")
        trainer.fit(lit_module, datamodule=data_module)
        
        logger.info("Smoke test complete!")


def inspect_dataset_samples():
    """Create and inspect dataset samples for debugging"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create synthetic dataset
        create_synthetic_dataset(tmp_dir, num_samples=3, img_size=32)
        
        # Create configurations
        dataset_config, _, _, _, _ = create_configs(tmp_dir)
        
        # Override for detailed inspection
        dataset_config["train_batch_size"] = 1
        
        # Create dataset directly
        train_config = dataset_config.copy()
        train_config["split"] = "train"
        
        # Fix the None sample issue in __getitem__
        from core.general_dataset import GeneralizedDataset
        
        # Patch the class to fix the None return issue
        original_getitem = GeneralizedDataset.__getitem__
        
        def safe_getitem(self, idx):
            result = original_getitem(self, idx)
            if result is None:
                # Try another index
                logger.warning(f"Got None for index {idx}, trying next index")
                return self.__getitem__((idx + 1) % len(self))
            return result
        
        # Apply the monkey patch
        GeneralizedDataset.__getitem__ = safe_getitem
        
        # Create and inspect dataset
        train_dataset = GeneralizedDataset(train_config)
        
        # Check dataset length
        logger.info(f"Dataset length: {len(train_dataset)}")
        
        # Iterate through a few samples
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            logger.info(f"Sample {i} keys: {sample.keys()}")
            
            # Check shapes
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"  {key} shape: {value.shape}, dtype: {value.dtype}, range: [{value.min()}, {value.max()}]")
        
        # Create dataloader and inspect a batch
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=0
        )
        
        # Get a batch
        batch = next(iter(train_loader))
        logger.info(f"Batch keys: {batch.keys()}")
        
        # Check shapes
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key} shape: {value.shape}, dtype: {value.dtype}")


if __name__ == "__main__":
    # First inspect dataset
    logger.info("Inspecting dataset samples...")
    inspect_dataset_samples()
    
    # Then run full smoke test
    logger.info("\nRunning full smoke test...")
    run_smoke_test()

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
  max_epochs: 100
  val_check_interval: 1.0  # Validate once per epoch
  skip_validation_until_epoch: 0  # Skip validation for the first 5 epochs
  val_every_n_epochs: 10
  log_every_n_epochs: 2  # Log predictions every 2 epochs
  log_every_n_steps: 1
  train_metrics_every_n_epochs: 1    # compute/log train metrics once every epoch
  val_metrics_every_n_epochs: 1      # compute/log val   metrics once every epoch

  save_gt_pred_val_test_every_n_epochs: 5  # Save every 5 epochs
  save_gt_pred_val_test_after_epoch: 10    # Start saving after epoch 10
  save_gt_pred_max_samples: 4              # Save up to 4 samples per epoch
  
  # Extra arguments passed directly to PyTorch Lightning Trainer
  extra_args:
    accelerator: "auto"  # Use GPU if available
    precision: 32  # Mixed precision training
    gradient_clip_val: 1.0
    accumulate_grad_batches: 1
    deterministic: false

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
target_y: "label_patch"

# ------------------------------------
# configs/metrics/segmentation.yaml
# ------------------------------------
# Segmentation Metrics Configuration

# List of metrics to evaluate
metrics:
  # Dice coefficient
  - alias: "dice"  # Shorthand name for the metric
    path: "torchmetrics.classification"
    class: "Dice"
    params:
      threshold: 0.5
      zero_division: 1.0

  # IoU (Jaccard index)
  - alias: "iou"
    path: "torchmetrics.classification"
    class: "JaccardIndex"
    params:
      task: "binary"
      threshold: 0.5
      num_classes: 2  # Binary segmentation

  # Connected Components Quality
  - alias: "ccq"
    path: "metrics.connected_components"
    class: "ConnectedComponentsQuality"
    params:
      min_size: 5
      tolerance: 2

  # Average Path Length Similarity
  - alias: "apls"
    path: "metrics.apls"
    class: "APLS"
    params:
      angle_range: [175, 185]
      max_nodes: 1000
      max_snap_dist: 25
      allow_renaming: true
      min_path_length: 10

# Per-metric frequencies for training - how often to compute each metric
train_frequencies:
  dice: 1    # Compute every epoch (lightweight metric)
  iou: 1     # Compute every epoch (lightweight metric)
  ccq: 1    # Compute every 10 epochs (moderately expensive)
  apls: 1   # Compute every 25 epochs (very computationally expensive)

# Per-metric frequencies for validation - how often to compute each metric
val_frequencies:
  dice: 1    # Compute every epoch
  iou: 1     # Compute every epoch
  ccq: 1     # Compute every 5 epochs
  apls: 1   # Compute every 10 epochs

# ------------------------------------
# configs/loss/mixed_topo.yaml
# ------------------------------------
# Mixed Topological Loss Configuration

# Primary loss (used from the beginning)
primary_loss:
  class: "BCELoss"  # Built-in PyTorch loss
  params: {}

# # Secondary loss (activated after start_epoch)
# secondary_loss:
#   path: "losses.custom_loss"  # Path to the module containing the loss
#   class: "TopologicalLoss"  # Name of the loss class
#   params:
#     topo_weight: 0.5  # Weight for the topological component
#     smoothness: 1.0  # Smoothness parameter for gradient computation
#     connectivity_weight: 0.3  # Weight for the connectivity component

# # Mixing parameters
# alpha: 0.4  # Weight for the secondary loss (0 to 1)
# start_epoch: 10  # Epoch to start using the secondary loss

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

# Dataset split mode: "folder" or "kfold"
split_mode: "folder"  # Uses folder structure for splits

# K-fold configuration (used if split_mode is "kfold")
fold: 0  # Current fold
num_folds: 3  # Total number of folds

# Split ratios (used if split_mode is "folder" with "source_folder")
# split_ratios:
  # train: 0.7
  # valid: 0.15
  # test: 0.15

use_splitting: false

# Source folder (used if split_mode is "folder" with split_ratios)
# source_folder: 'train'

# Batch sizes
train_batch_size: 8
val_batch_size: 1  # Usually 1 for full-image validation
test_batch_size: 1  # Usually 1 for full-image testing

# Patch and crop settings
patch_size: 256  # Size of training patches
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
  # sdf: "sdf"  # Signed distance function folder

# Distance transform settings
# distance_threshold: 20.0

# Signed distance function settings
# sdf_iterations: 3
# sdf_thresholds: [-20, 20]

# Augmentation settings
augmentations:
  - "flip_h"
  - "flip_v"
  - "rotation"

# Misc settings
# max_images: null  # No limit (set a number to limit images loaded)
max_images: 10  # No limit (set a number to limit images loaded)
# max_attempts: 10  # Maximum attempts for finding valid patches
save_computed: false  # Save computed distance maps and SDFs
verbose: false  # Verbose output
seed: 42  # Random seed

# ------------------------------------
# configs/model/baseline.yaml
# ------------------------------------
# TopoTokens Model Configuration

# Model path and class
path: "models.base_models"  # Path to the module containing the model
class: "UNetBin"  # Name of the model class

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

