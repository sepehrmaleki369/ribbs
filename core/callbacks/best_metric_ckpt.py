
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
        
        # Check each metric
        for metric in self.metric_names:
            metric_key = f"val_metric/{metric}"
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
            (trainer.current_epoch+1) % self.last_k == 0 or  # Every k epochs
            trainer.current_epoch == trainer.max_epochs - 1  # Last epoch
        ):
            filename = "last.ckpt"
            filepath = os.path.join(self.dirpath, filename)
            trainer.save_checkpoint(filepath)
            self.logger.info(f"Saved last checkpoint at epoch {trainer.current_epoch}")



