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


logging.getLogger('rasterio').setLevel(logging.WARNING)
logging.getLogger('rasterio.env').setLevel(logging.WARNING)
logging.getLogger('rasterio._io').setLevel(logging.WARNING)
logging.getLogger('rasterio._env').setLevel(logging.WARNING)
logging.getLogger('rasterio._base').setLevel(logging.WARNING)


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
        last_k=1,  # Save last checkpoint every epoch instead of every 5 epochs
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
    
    # Prepare trainer kwargs - FIXED: Remove resume_from_checkpoint
    trainer_kwargs = {
        "max_epochs": max_epochs,
        "callbacks": callbacks,
        "logger": tb_logger,
        "val_check_interval": val_check_interval,
        "check_val_every_n_epoch": trainer_cfg.get("val_every_n_epochs", 1),  # Validate every N epochs
        **trainer_cfg.get("extra_args", {}),
    }
    
    # Don't add resume_from_checkpoint to trainer_kwargs anymore
    trainer = pl.Trainer(**trainer_kwargs)

    # --- run ---
    if args.test:
        logger.info("Running test...")
        # For testing, load from checkpoint if provided
        if args.resume:
            logger.info(f"Loading checkpoint for testing: {args.resume}")
            trainer.test(lit, datamodule=dm, ckpt_path=args.resume)
        else:
            trainer.test(lit, datamodule=dm)
    else:
        logger.info("Running training...")
        # For training, use ckpt_path parameter in fit() method
        if args.resume:
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            trainer.fit(lit, datamodule=dm, ckpt_path=args.resume)
        else:
            logger.info("Starting training from scratch...")
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