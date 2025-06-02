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
    
    Fixed to properly save ALL samples when max_samples=None.
    """
    
    def __init__(
        self,
        save_dir: str,
        save_every_n_epochs: int = 5,
        save_after_epoch: int = 0,
        max_samples: int = None  # None means save all samples
    ):
        """
        Initialize the PredictionSaver callback.
        
        Args:
            save_dir: Directory to save prediction data
            save_every_n_epochs: How often to save (every N epochs)
            save_after_epoch: Only start saving after this epoch
            max_samples: Maximum number of samples to save per epoch (None = save all)
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.save_after_epoch = save_after_epoch
        self.max_samples = max_samples  # None means no limit
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self._reset_buffers()
        
        # Log initialization once at INFO level
        self.logger.info(f"PredictionSaver initialized:")
        self.logger.info(f"  save_dir: {save_dir}")
        self.logger.info(f"  save_every_n_epochs: {save_every_n_epochs}")
        self.logger.info(f"  save_after_epoch: {save_after_epoch}")
        if max_samples is None:
            self.logger.info(f"  max_samples: unlimited (save all)")
        else:
            self.logger.info(f"  max_samples: {max_samples}")
    
    def _reset_buffers(self):
        """Reset the internal buffers that collect samples."""
        self._gts = []
        self._preds = []
        self._collected = 0
        # Note: Don't reset _saved_this_epoch here - it's managed per epoch
    
    def _should_save_this_epoch(self, epoch):
        """Determine if we should save data for this epoch."""
        if epoch < self.save_after_epoch:
            return False
        self.logger.debug(f'Line 498: + epoch:{epoch} save_after:{self.save_after_epoch} every:{self.save_every_n_epochs} check: {(epoch - self.save_after_epoch)}')
        return (epoch - self.save_after_epoch) % self.save_every_n_epochs == 0
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Reset state at the start of a validation epoch."""
        current_epoch = trainer.current_epoch
        
        # Always reset buffers and state for new epoch
        self._reset_buffers()
        self._saved_this_epoch = False
        
        # Check if we should save this epoch
        if self._should_save_this_epoch(current_epoch):
            self.logger.info(f"Will save predictions for epoch {current_epoch}")
        else:
            # Mark as done for non-saving epochs to skip processing
            self._saved_this_epoch = True
            self.logger.debug(f"Skipping epoch {current_epoch} (not a saving epoch)")
    
    def on_test_epoch_start(self, trainer, pl_module):
        """Reset buffers at the start of a test epoch."""
        self.logger.info(f"Starting test prediction collection for epoch {trainer.current_epoch}")
        self._reset_buffers()
        self._saved_this_epoch = False
    
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
        # Early return if we've already saved this epoch
        if self._saved_this_epoch:
            return
        
        current_epoch = trainer.current_epoch
        if not self._should_save_this_epoch(current_epoch):
            return
        
        # FIXED: Only check max_samples limit if it's actually set
        # When max_samples is None, we want to collect ALL samples
        if self.max_samples is not None and self._collected >= self.max_samples:
            return
        
        # Extract data with additional safety checks
        try:
            if outputs is None or not isinstance(outputs, dict):
                return
            if "predictions" not in outputs:
                return
                
            y_true = batch[pl_module.target_key]
            y_pred = outputs["predictions"]
            
            # Validate tensor shapes
            if y_true.numel() == 0 or y_pred.numel() == 0:
                return
                
        except (KeyError, AttributeError, RuntimeError) as e:
            self.logger.debug(f"Skipping batch due to extraction error: {e}")
            return
        
        # Move to CPU and detach to prevent memory leaks
        try:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.detach().cpu()
        except RuntimeError as e:
            self.logger.debug(f"Error moving tensors to CPU: {e}")
            return
        
        # FIXED: Calculate how many samples to take
        if self.max_samples is None:
            # Take ALL samples from the batch when max_samples is None
            take = y_pred.shape[0]
        else:
            # Take limited samples when max_samples is set
            remaining = self.max_samples - self._collected
            take = min(remaining, y_pred.shape[0])
        
        if take > 0:
            # Append the slices
            self._gts.append(y_true[:take])
            self._preds.append(y_pred[:take])
            self._collected += take
            
            # FIXED: Improved logging for unlimited collection
            if self.max_samples is None:
                if batch_idx % 20 == 0:  # Log every 20 batches when saving all
                    self.logger.debug(f"Collected {self._collected} samples so far (batch {batch_idx})")
            else:
                # Log when we reach max samples for limited collection
                if self._collected >= self.max_samples:
                    self.logger.info(f"Collected maximum {self.max_samples} samples for epoch {current_epoch}")
                elif batch_idx % 10 == 0:  # Log progress every 10 batches
                    self.logger.debug(f"Collected {self._collected}/{self.max_samples} samples (batch {batch_idx})")
    
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
        # Early return if we've already saved this epoch
        if self._saved_this_epoch:
            return
        
        # FIXED: Only check max_samples limit if it's actually set
        # When max_samples is None, we want to collect ALL samples
        if self.max_samples is not None and self._collected >= self.max_samples:
            return
        
        # Extract data
        try:
            if outputs is None or not isinstance(outputs, dict):
                return
            if "predictions" not in outputs:
                return
                
            y_true = batch[pl_module.target_key]
            y_pred = outputs["predictions"]
            
            # Validate tensor shapes
            if y_true.numel() == 0 or y_pred.numel() == 0:
                return
                
        except (KeyError, AttributeError, RuntimeError) as e:
            self.logger.debug(f"Skipping test batch due to extraction error: {e}")
            return
        
        # Move to CPU and detach
        try:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.detach().cpu()
        except RuntimeError as e:
            self.logger.debug(f"Error moving tensors to CPU: {e}")
            return
        
        # FIXED: Calculate how many samples to take
        if self.max_samples is None:
            # Take ALL samples from the batch when max_samples is None
            take = y_pred.shape[0]
        else:
            # Take limited samples when max_samples is set
            remaining = self.max_samples - self._collected
            take = min(remaining, y_pred.shape[0])
        
        if take > 0:
            self._gts.append(y_true[:take])
            self._preds.append(y_pred[:take])
            self._collected += take
            
            # Log progress for test phase
            if self.max_samples is None:
                if batch_idx % 20 == 0:  # Log every 20 batches when saving all
                    self.logger.debug(f"Collected {self._collected} test samples so far (batch {batch_idx})")
            else:
                if batch_idx % 10 == 0:  # Log progress every 10 batches
                    self.logger.debug(f"Collected {self._collected}/{self.max_samples} test samples (batch {batch_idx})")
    
    def _save_data(self, trainer, phase="val"):
        """Save collected data as NumPy arrays."""
        if self._collected == 0:
            self.logger.debug(f"No data collected to save for {phase}")
            return False
            
        current_epoch = trainer.current_epoch
        
        try:
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
            gt_path = os.path.join(epoch_dir, "ground_truth.npy")
            pred_path = os.path.join(epoch_dir, "predictions.npy")
            
            np.save(gt_path, gts_numpy)
            np.save(pred_path, preds_numpy)
            
            # FIXED: Better logging message
            samples_info = f"{self._collected} samples" if self.max_samples is None else f"{self._collected}/{self.max_samples} samples"
            self.logger.info(f"Saved {samples_info} for {phase} epoch {current_epoch} "
                           f"(shapes: gt={gts_numpy.shape}, pred={preds_numpy.shape})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save {phase} data for epoch {current_epoch}: {e}")
            return False
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Save data at the end of the validation epoch if conditions are met."""
        current_epoch = trainer.current_epoch
        
        # Only attempt to save if we haven't already saved and we collected data
        if not self._saved_this_epoch and self._collected > 0:
            success = self._save_data(trainer, "val")
            if success:
                self._saved_this_epoch = True
        
        # Clean up buffers to free memory
        self._gts.clear()
        self._preds.clear()
        
        final_count = self._collected
        self._collected = 0
        
        self.logger.debug(f"Validation epoch {current_epoch} end: "
                         f"saved={self._saved_this_epoch}, collected={final_count} samples")
    
    def on_test_epoch_end(self, trainer, pl_module):
        """Save data at the end of the test epoch."""
        current_epoch = trainer.current_epoch
        
        if not self._saved_this_epoch and self._collected > 0:
            success = self._save_data(trainer, "test")
            if success:
                self._saved_this_epoch = True
        
        # Clean up buffers
        self._gts.clear()
        self._preds.clear()
        
        final_count = self._collected
        self._collected = 0
        
        self.logger.debug(f"Test epoch {current_epoch} end: collected={final_count} samples")