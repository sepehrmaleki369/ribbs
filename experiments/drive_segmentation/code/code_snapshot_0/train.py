"""
Training script for segmentation experiments.

This script orchestrates the training process by loading configurations,
setting up models, losses, metrics, and dataloaders, and running training.
"""

import os
import argparse
import logging
from typing import Any, Dict
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

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
for lib in ("rasterio", "matplotlib", "PIL", "tensorboard", "urllib3"):
    logging.getLogger(lib).setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger("core").setLevel(logging.INFO)
logging.getLogger("__main__").setLevel(logging.INFO)
logging.getLogger("seglit_module").setLevel(logging.INFO)


def load_config(path: str) -> Dict[str, Any]:
    return yaml_read(path)


def build_lit(
    model,
    mixed_loss,
    metric_list,
    main_cfg,
    inference_cfg,
    input_key,
    target_key,
    train_metrics_every_n,
    val_metrics_every_n,
    metrics_cfg,
    *, compute_val_loss=True
):
    """Always create a *fresh* LightningModule – Trainer will load the checkpoint later."""
    return SegLitModule(
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
        divisible_by=inference_cfg.get("chunk_divisible_by", 16),
        compute_val_loss=compute_val_loss, 
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--test", action="store_true", help="Run testing instead of training")
    args = parser.parse_args()

    # --- load configs -------------------------------------------------------
    main_cfg = load_config(args.config)
    dataset_cfg = load_config(os.path.join("configs", "dataset", main_cfg["dataset_config"]))
    model_cfg = load_config(os.path.join("configs", "model", main_cfg["model_config"]))
    loss_cfg = load_config(os.path.join("configs", "loss", main_cfg["loss_config"]))
    metrics_cfg = load_config(os.path.join("configs", "metrics", main_cfg["metrics_config"]))
    inference_cfg = load_config(os.path.join("configs", "inference", main_cfg["inference_config"]))
    callbacks_cfg = load_config(os.path.join("configs", "callbacks", main_cfg["callbacks_config"]))

    # --- output & logger ----------------------------------------------------
    output_dir = main_cfg.get("output_dir", "outputs")
    mkdir(output_dir)
    logger = setup_logger(os.path.join(output_dir, "training.log"))
    logger.info(f"Output dir: {output_dir}")

    # --- trainer params -----------------------------------------------------
    trainer_cfg = main_cfg.get("trainer", {})
    train_metrics_every_n = trainer_cfg["train_metrics_every_n_epochs"]
    val_metrics_every_n = trainer_cfg["val_metrics_every_n_epochs"]
    compute_val_loss        = trainer_cfg.get("compute_val_loss", True)

    # --- model / loss / metrics --------------------------------------------
    logger.info("Loading model…")
    model = load_model(model_cfg)

    logger.info("Loading losses…")
    prim = load_loss(loss_cfg["primary_loss"])
    sec = load_loss(loss_cfg["secondary_loss"]) if loss_cfg.get("secondary_loss") else None
    mixed_loss = MixedLoss(
        prim,
        sec,
        alpha=loss_cfg.get("alpha", 0.5),
        start_epoch=loss_cfg.get("start_epoch", 0),
    )

    logger.info("Loading metrics…")
    metric_list = load_metrics(metrics_cfg["metrics"])

    # --- data module --------------------------------------------------------
    logger.info("Setting up data module…")
    dm = SegmentationDataModule(dataset_cfg)
    dm.setup()
    logger.info(f"Train set size:      {len(dm.train_dataset)} samples")
    logger.info(f"Validation set size: {len(dm.val_dataset)} samples")
    logger.info(f"Test set size:       {len(dm.test_dataset)} samples")

    # --- lightning module ---------------------------------------------------
    input_key = main_cfg.get("target_x", "image_patch")
    target_key = main_cfg.get("target_y", "label_patch")
    lit = build_lit(
        model,
        mixed_loss,
        metric_list,
        main_cfg,
        inference_cfg,
        input_key,
        target_key,
        train_metrics_every_n,
        val_metrics_every_n,
        metrics_cfg,
        compute_val_loss = compute_val_loss
    )

    # --- callbacks ----------------------------------------------------------
    callbacks = load_callbacks(
        callbacks_cfg["callbacks"],
        output_dir=output_dir,
        resume=args.resume,
        skip_valid_until_epoch=trainer_cfg["skip_validation_until_epoch"],
        save_gt_pred_val_test_every_n_epochs=trainer_cfg.get("save_gt_pred_val_test_every_n_epochs", 5),
        save_gt_pred_val_test_after_epoch=trainer_cfg.get("save_gt_pred_val_test_after_epoch", 0),
        save_gt_pred_max_samples=trainer_cfg.get("save_gt_pred_max_samples"),
        project_root=os.path.dirname(os.path.abspath(__file__)),
    )

    # --- trainer ------------------------------------------------------------
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    trainer_kwargs = dict(trainer_cfg.get("extra_args", {}))
    trainer_kwargs.update(
        {
            "max_epochs": trainer_cfg["max_epochs"],
            "num_sanity_val_steps": trainer_cfg["num_sanity_val_steps"],
            "check_val_every_n_epoch": trainer_cfg["check_val_every_n_epoch"],
            "log_every_n_steps": trainer_cfg["log_every_n_steps"],
            "callbacks": callbacks,
            "logger": tb_logger,
            "default_root_dir": output_dir,
        }
    )
    trainer = pl.Trainer(**trainer_kwargs)

    # quick sanity print
    batch = next(iter(dm.train_dataloader()))
    logger.info(f"image_patch shape: {batch['image_patch'].shape}")
    logger.debug(f"UNet expects: {lit.model.in_channels}")

    # --- run ----------------------------------------------------------------
    ckpt = args.resume or None
    if args.test:
        logger.info("Running test…")
        trainer.test(lit, datamodule=dm, ckpt_path=ckpt)
    else:
        logger.info("Running training…")
        trainer.fit(lit, datamodule=dm, ckpt_path=ckpt)

        # evaluate best checkpoint after training
        mgr = CheckpointManager(
            checkpoint_dir=os.path.join(output_dir, "checkpoints"),
            metrics=list(metric_list.keys()),
            default_mode="max",
        )
        best_metric, best_ckpt, best_val = mgr.get_best_checkpoint()
        logger.info(f"Best ckpt: {best_ckpt} ({best_metric}={best_val:.4f})")
        trainer.test(lit, datamodule=dm, ckpt_path=best_ckpt)


if __name__ == "__main__":
    main()
