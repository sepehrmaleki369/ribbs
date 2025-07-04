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
                    num_samples=params.get("num_samples", 5),
                    cmap=params.get("cmap", "coolwarm"),
                    projection=params.get("projection", "max"),
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
