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
    Callback to log input/prediction/ground truth visualization during training.
    
    This callback creates a grid visualization of input images, ground truth masks,
    and model predictions at configurable intervals.
    """
    
    def __init__(
        self,
        log_dir: str,
        log_every_n_epochs: int = 1,
        max_samples: int = 4
    ):
        """
        Initialize the PredictionLogger callback.
        
        Args:
            log_dir: Directory to save visualizations to
            log_every_n_epochs: Frequency of logging in epochs
            max_samples: Maximum number of samples to visualize
        """
        super().__init__()
        self.log_dir = log_dir
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        """
        Log predictions at the end of a validation batch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            outputs: Batch outputs (including predictions)
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: Dataloader index
        """
        # Only log on specified epochs and for the first batch
        if batch_idx != 0 or trainer.current_epoch % self.log_every_n_epochs != 0:
            return
            
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Extract data
        x = batch["image_patch"][:self.max_samples]
        y_true = batch["label_patch"][:self.max_samples]
        y_pred = outputs["predictions"][:self.max_samples]
        
        # Ensure we're working with CPU tensors
        x = x.detach().cpu()
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()
        
        # Create figure
        fig, axes = plt.subplots(self.max_samples, 3, figsize=(12, 4 * self.max_samples))
        
        for i in range(min(self.max_samples, x.shape[0])):
            # Display input image
            if x.shape[1] == 1:  # Grayscale
                axes[i, 0].imshow(x[i, 0], cmap='gray')
            else:  # RGB
                # Convert from (C, H, W) to (H, W, C) for matplotlib
                img = x[i].permute(1, 2, 0)
                # Clip to [0, 1] range
                img = torch.clamp(img, 0, 1)
                axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')
            
            # Display ground truth
            if y_true.shape[1] == 1:  # Binary mask
                axes[i, 1].imshow(y_true[i, 0], cmap='gray')
            else:  # Multi-class mask
                # Use argmax for multi-class segmentation
                mask = torch.argmax(y_true[i], dim=0)
                axes[i, 1].imshow(mask, cmap='tab20')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Display prediction
            if y_pred.shape[1] == 1:  # Binary mask
                axes[i, 2].imshow(y_pred[i, 0], cmap='gray')
            else:  # Multi-class mask
                # Use argmax for multi-class segmentation
                pred = torch.argmax(y_pred[i], dim=0)
                axes[i, 2].imshow(pred, cmap='tab20')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.log_dir, f"pred_epoch_{trainer.current_epoch:06d}.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        
        self.logger.info(f"Saved prediction visualization: {filename}")


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