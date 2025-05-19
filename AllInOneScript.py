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

# ------------------------------------
# requirements.txt
# ------------------------------------
# Python ≥3.8, PyTorch Lightning ≥1.9 requirements file

# Core dependencies
torch==2.1.0
torchvision==0.16.0
pytorch-lightning==2.1.2
torchmetrics==1.2.0

# Data processing
numpy==1.24.3
scipy==1.10.1
scikit-image==0.21.0
scikit-learn==1.2.2
rasterio==1.3.8
tqdm==4.66.1
pandas==2.0.3

# Visualization
matplotlib==3.7.2
tensorboard==2.14.0

# Configuration
pyyaml==6.0.1

# Graph processing (for APLS metric)
networkx==3.1

# Utilities
pillow==10.0.1

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

This module provides functionality to perform validation with full-size images
by processing them in patches with overlap.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import the existing process_in_chuncks function
from core.utils import process_in_chuncks


class Validator:
    """
    Validator class for handling chunked inference during validation/testing.
    
    This class enables processing large images that don't fit in GPU memory by
    splitting them into overlapping chunks, processing each chunk, and reassembling
    the results with proper handling of overlapping regions.
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
    
    def run_chunked_inference(
        self, 
        model: nn.Module, 
        image: torch.Tensor, 
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Run inference on a large image by processing it in chunks.
        
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
        
        # Get image dimensions
        N, C, H, W = image.shape
        
        # Create empty output tensor (assuming model output has same spatial dimensions as input)
        # We need to run a test inference to get the output channel dimension
        with torch.no_grad():
            # Create a small test patch to determine output channels
            test_patch = image[:, :, :min(H, self.patch_size[0]), :min(W, self.patch_size[1])]
            test_output = model(test_patch)
            out_channels = test_output.shape[1]
            
        # Initialize output tensor (N, out_channels, H, W)
        output = torch.zeros((N, out_channels, H, W), device=device)
        
        # Process function to wrap model inference
        def process_fn(chunk):
            with torch.no_grad():
                return model(chunk)
        
        # Run chunked inference
        output = process_in_chuncks(image, output, process_fn, self.patch_size, self.patch_margin)
        
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
    After each training & validation epoch, log the first `num_samples`
    examples as figures:
      - train/samples: [input | ground-truth | prediction]
      - val/samples:   [input | ground-truth | prediction]

    Captures the very first batch during on_*_batch_end,
    then at epoch end renders once.
    """

    def __init__(self, num_samples: int = 5):
        super().__init__()
        self.num_samples = num_samples
        self._train_sample = None  # tuple(x, y, preds)
        self._val_sample   = None

    def _capture(self, batch, pl_module):
        x = batch["image_patch"].float().to(pl_module.device)
        y = batch["label_patch"].float().to(pl_module.device)
        if y.ndim == 3:
            y = y.unsqueeze(1)
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.sigmoid(logits)
        return x.cpu(), y.cpu(), preds.cpu()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args
    ):
        # only capture first batch
        if batch_idx == 0 and self._train_sample is None:
            self._train_sample = self._capture(batch, pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args
    ):
        if batch_idx == 0 and self._val_sample is None:
            self._val_sample = self._capture(batch, pl_module)

    def _plot_and_log(self, sample, tag, trainer):
        x, y, preds = sample
        n = min(self.num_samples, x.size(0))

        fig, axes = plt.subplots(n, 3, figsize=(3 * 3, n * 3), tight_layout=True)
        if n == 1:
            axes = axes[None, :]

        for i in range(n):
            img = x[i].permute(1, 2, 0)
            gt  = y[i, 0]
            pr  = preds[i, 0]

            axes[i, 0].imshow(img)
            axes[i, 0].set_title("input")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gt, cmap="gray")
            axes[i, 1].set_title("gt")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pr, cmap="gray")
            axes[i, 2].set_title("pred")
            axes[i, 2].axis("off")

        trainer.logger.experiment.add_figure(
            f"{tag}/samples", fig, global_step=trainer.current_epoch
        )
        plt.close(fig)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._train_sample is not None:
            self._plot_and_log(self._train_sample, "train", trainer)
            self._train_sample = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._val_sample is not None:
            self._plot_and_log(self._val_sample, "val", trainer)
            self._val_sample = None



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

class PredictionLogger(Callback):
    """
    Callback to log input/prediction/ground truth visualization during validation.
    Accumulates up to `max_samples` across batches and saves one grid per epoch.
    """

    def __init__(
        self,
        log_dir: str,
        log_every_n_epochs: int = 1,
        max_samples: int = 4
    ):
        super().__init__()
        self.log_dir = log_dir
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)

        # Buffers to accumulate per-epoch
        self._reset_buffers()

    def _reset_buffers(self):
        self._images = []
        self._gts = []
        self._preds = []
        self._collected = 0
        self._logged_this_epoch = False

    def on_validation_epoch_start(self, trainer, pl_module):
        # Only prepare to log on epochs matching frequency
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self._reset_buffers()
        else:
            # Mark as already done for non-logging epochs
            self._logged_this_epoch = True

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0
    ):
        # Skip if we already logged this epoch or it's not a logging epoch
        if self._logged_this_epoch:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Extract raw tensors (up to max needed)
        x = batch["image_patch"]
        y_true = batch["label_patch"]
        y_pred = outputs["predictions"]

        # Move to CPU and detach
        x = x.detach().cpu()
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()

        # How many more we need
        remaining = self.max_samples - self._collected
        take = min(remaining, x.shape[0])

        # Append the slice
        self._images.append(x[:take])
        self._gts.append(y_true[:take])
        self._preds.append(y_pred[:take])
        self._collected += take

        # Once we've got enough, render & save
        if self._collected >= self.max_samples:
            # Concatenate buffers
            imgs = torch.cat(self._images, dim=0)
            gts  = torch.cat(self._gts,    dim=0)
            preds= torch.cat(self._preds,  dim=0)

            # Ensure directory exists
            os.makedirs(self.log_dir, exist_ok=True)
            filename = os.path.join(
                self.log_dir,
                f"pred_epoch_{trainer.current_epoch:06d}.png"
            )

            # Create grid
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
from metrics.apls_core import apls
import torch
import torch.nn as nn
import numpy as np

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
            m = m[None, ...]           # (H,W) -> (1,H,W)
        if m.ndim == 3:
            return m                  # already (B,H,W) or (1,H,W)
        if m.ndim == 4 and m.shape[1] == 1:
            return m[:,0,...]        # (B,1,H,W) -> (B,H,W)
        raise ValueError(f"Unsupported mask shape {m.shape}")

    gt = _unify(gt_masks)
    pr = _unify(pred_masks)
    if gt.shape != pr.shape:
        raise ValueError(f"GT shape {gt.shape} != pred shape {pr.shape}")

    B, H, W = gt.shape
    scores = np.zeros(B, dtype=np.float32)

    for i in range(B):
        gt_bin = (gt[i] > 0.5).astype(np.uint8)
        pr_bin = (pr[i] > 0.5).astype(np.uint8)

        # handle empty‐GT cases
        if gt_bin.sum() == 0:
            # perfect if pred is also empty, else zero
            scores[i] = 1.0 if pr_bin.sum() == 0 else 0.0
            continue

        # now GT non‐empty ⇒ delegate to core APLS
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
            # fallback for degenerate graphs
            scores[i] = 0.0

    return scores


class APLS(nn.Module):
    """
    Average Path Length Similarity (APLS) metric for road network segmentation.
    """
    
    def __init__(
        self,
        min_segment_length=10,
        max_nodes=500,
        sampling_ratio=0.1,
        angle_range=(135, 225),
        max_snap_dist=4,
        allow_renaming=True,
    ):
        super().__init__()
        self.min_segment_length = min_segment_length
        self.max_nodes = max_nodes
        self.sampling_ratio = sampling_ratio
        self.angle_range = angle_range
        self.max_snap_dist = max_snap_dist
        self.allow_renaming = allow_renaming
    
    def forward(self, y_pred, y_true):
        """
        Compute the APLS metric between predicted and ground truth road networks.
        """
        scores = compute_batch_apls(
            y_true,
            y_pred,
            angle_range=self.angle_range,
            max_nodes=self.max_nodes,
            max_snap_dist=self.max_snap_dist,
            allow_renaming=self.allow_renaming,
            min_path_length=self.min_segment_length
        )
        
        # Convert numpy array to torch tensor and return mean
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
# configs/main.yaml
# ------------------------------------
# Main configuration file for segmentation experiments
# This file references all sub-configs and sets high-level training parameters

# Sub-config references
dataset_config: "massroads.yaml"
model_config: "topotokens.yaml"
loss_config: "mixed_topo.yaml"
metrics_config: "segmentation.yaml"
inference_config: "chunk.yaml"

# Output directory
output_dir: "outputs/experiment_1"

# Trainer configuration
trainer:
  max_epochs: 100
  val_check_interval: 1.0  # Validate once per epoch
  skip_validation_until_epoch: 0  # Skip validation for the first 5 epochs
  val_every_n_epochs: 5
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
    lr: 0.001
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
      lr_decay_factor: 0.0001

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
      min_segment_length: 10
      max_nodes: 500
      sampling_ratio: 0.1

# Per-metric frequencies for training - how often to compute each metric
train_frequencies:
  dice: 1    # Compute every epoch (lightweight metric)
  iou: 1     # Compute every epoch (lightweight metric)
  ccq: 10    # Compute every 10 epochs (moderately expensive)
  apls: 25   # Compute every 25 epochs (very computationally expensive)

# Per-metric frequencies for validation - how often to compute each metric
val_frequencies:
  dice: 1    # Compute every epoch
  iou: 1     # Compute every epoch
  ccq: 5     # Compute every 5 epochs
  apls: 10   # Compute every 10 epochs

# ------------------------------------
# configs/loss/mixed_topo.yaml
# ------------------------------------
# Mixed Topological Loss Configuration

# Primary loss (used from the beginning)
primary_loss:
  class: "BCELoss"  # Built-in PyTorch loss
  params: {}

# Secondary loss (activated after start_epoch)
secondary_loss:
  path: "losses.custom_loss"  # Path to the module containing the loss
  class: "TopologicalLoss"  # Name of the loss class
  params:
    topo_weight: 0.5  # Weight for the topological component
    smoothness: 1.0  # Smoothness parameter for gradient computation
    connectivity_weight: 0.3  # Weight for the connectivity component

# Mixing parameters
alpha: 0.4  # Weight for the secondary loss (0 to 1)
start_epoch: 10  # Epoch to start using the secondary loss

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

# Additional inference settings
use_tta: false  # Test-time augmentation
tta_merge_mode: "mean"  # How to merge TTA results (mean, max, etc.)

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

use_splitting: true

# Source folder (used if split_mode is "folder" with split_ratios)
source_folder: 'train'

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
  label: "map"  # Road map folder
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
max_images: 10  # No limit (set a number to limit images loaded)
# max_attempts: 10  # Maximum attempts for finding valid patches
save_computed: false  # Save computed distance maps and SDFs
verbose: false  # Verbose output
seed: 42  # Random seed

# ------------------------------------
# configs/model/topotokens.yaml
# ------------------------------------
# TopoTokens Model Configuration

# Model path and class
path: "models.custom_model"  # Path to the module containing the model
class: "TopoTokens"  # Name of the model class

# Model parameters
params:
  in_channels: 3  # RGB input
  out_channels: 1  # Binary segmentation output
  encoder_channels: 64  # Initial number of encoder channels
  decoder_channels: 32  # Initial number of decoder channels
  num_blocks: 4  # Number of encoder/decoder blocks
  dropout: 0.2  # Dropout rate

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
                 n_levels=3, dropout=0.0, batch_norm=False, upsampling='deconv',
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
                 n_levels=3, dropout=0.0, batch_norm=False, upsampling='deconv',
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
                 n_levels=3, dropout=0.0, batch_norm=False, upsampling='deconv',
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

