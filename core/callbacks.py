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

def _gather_from_outputs(batch, outputs, pl_module):
    """
    Extract input, gt, and prediction tensors from batch and outputs without re-running the model.
    """
    x = batch[pl_module.input_key].float()
    y = batch[pl_module.target_key].float()
    if y.dim() == 3:
        y = y.unsqueeze(1)
    preds = outputs.get("predictions").float()
    return x.cpu(), y.cpu(), preds.detach().cpu()

class SamplePlotCallback(pl.Callback):
    """
    After each training & validation epoch, log up to `num_samples`
    examples as figures:
      - train/samples: [input | ground-truth | prediction]
      - validation/samples: [input | ground-truth | prediction]

    Now uses *separate* vmin/vmax for GT vs. prediction.
    """
    def __init__(self, num_samples: int = 5, cmap: str = "coolwarm"):
        super().__init__()
        self.num_samples = num_samples
        self.cmap = cmap
        self._reset()

    def _reset(self):
        self._images = []
        self._gts    = []
        self._preds  = []
        self._count  = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset()

    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset()

    def _collect(self, batch, outputs, pl_module):
        if self._count >= self.num_samples:
            return
        x, y, preds = _gather_from_outputs(batch, outputs, pl_module)
        take = min(self.num_samples - self._count, x.size(0))
        self._images.append(x[:take])
        self._gts   .append(y[:take])
        self._preds .append(preds[:take])
        self._count += take

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **_):
        self._collect(batch, outputs, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **_):
        self._collect(batch, outputs, pl_module)

    def _plot_and_log(self, tag, trainer):
        imgs  = torch.cat(self._images, 0)
        gts   = torch.cat(self._gts,    0)
        preds = torch.cat(self._preds,  0)
        n = imgs.size(0)

        # compute *separate* signed limits
        vlim_gt   = float(gts.abs().max())
        vlim_pred = float(preds.abs().max())

        fig, axes = plt.subplots(n, 3, figsize=(9, n*3), tight_layout=True)
        if n == 1:
            axes = axes[None, :]

        for i in range(n):
            img = imgs[i].permute(1,2,0)
            gt  = gts[i,0]
            pr  = preds[i,0]

            # input
            axes[i,0].imshow(img, cmap='gray')
            axes[i,0].set_title("input")
            axes[i,0].axis("off")

            # ground truth
            axes[i,1].imshow(gt, cmap=self.cmap, vmin=-vlim_gt, vmax=vlim_gt)
            axes[i,1].set_title("gt (signed)")
            axes[i,1].axis("off")

            # prediction
            axes[i,2].imshow(pr, cmap=self.cmap, vmin=-vlim_pred, vmax=vlim_pred)
            axes[i,2].set_title("pred (signed)")
            axes[i,2].axis("off")

        trainer.logger.experiment.add_figure(f"{tag}/samples",
                                             fig,
                                             global_step=trainer.current_epoch)
        plt.close(fig)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._count > 0:
            self._plot_and_log("train", trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._count > 0:
            self._plot_and_log("validation", trainer)


class PredictionLogger(Callback):
    """
    Validation‐only: accumulates up to `max_samples` and writes one PNG per epoch.
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
        if trainer.current_epoch % self.log_every_n_epochs == 0:
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
           or (trainer.current_epoch % self.log_every_n_epochs != 0):
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

    def _should_save(self, epoch: int) -> bool:
        return epoch >= self.after and (epoch - self.after) % self.every == 0

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

    # ——— TRAINING HOOKS ———
    def on_train_epoch_start(self, trainer, pl_module):
        self._counter = 0

    def on_train_batch_end(
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
            self._save_tensor(preds[i], "train", epoch, batch_idx, i, "pred")
            self._save_tensor(gts[i],   "train", epoch, batch_idx, i, "gt")
            self._counter += 1

    # ——— VALIDATION HOOKS ———
    def on_validation_epoch_start(self, trainer, pl_module):
        self._counter = 0

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

        preds = preds.detach().cpu().numpy()
        gts   = gts.detach().cpu().numpy()
        for i in range(preds.shape[0]):
            if self.max_samples is not None and self._counter >= self.max_samples:
                return
            self._save_tensor(preds[i], "val", epoch, batch_idx, i, "pred")
            self._save_tensor(gts[i],   "val", epoch, batch_idx, i, "gt")
            self._counter += 1

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
