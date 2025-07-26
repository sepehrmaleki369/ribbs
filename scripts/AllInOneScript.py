# ------------------------------------
# train.py
# ------------------------------------
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
import logging

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
        divisible_by: int = 16,
        compute_val_loss=False
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

        self.divisible_by = divisible_by

        self.compute_val_loss = compute_val_loss

        self.py_logger = logging.getLogger(__name__)
    
    def log(self, name, value, *args, **kwargs):
        """
        Override LightningModule.log so that every time you call self.log(...)
        it also does a Python-log to console/file.
        """
        # 1) call Lightning's logger
        super().log(name, value, *args, **kwargs)

        # 2) extract a scalar out of value (torch.Tensor or numeric)
        try:
            val = value.item()
        except Exception:
            val = value

        # 3) log via Python logging
        #    you can format this how you like; here we print epoch/step context
        ep = getattr(self, 'current_epoch', None)
        st = getattr(self.trainer, 'global_step', None)
        ctx = f"[ep={ep} step={st}]" if ep is not None and st is not None else ""
        self.py_logger.info(f"{ctx} {name} = {val:.4f}")

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
            y_hat = self.validator.run_chunked_inference(self.model, x, self.divisible_by)
        return y_hat
    
    def on_train_epoch_start(self):
        dm = self.trainer.datamodule
        if dm and hasattr(dm, "train_dataset") and hasattr(dm.train_dataset, "set_epoch"):
            dm.train_dataset.set_epoch(self.current_epoch)
            
        self._train_preds = []
        self._train_gts   = []

        if isinstance(self.loss_fn, MixedLoss):
            self.loss_fn.update_epoch(self.current_epoch)
        for m in self.metrics.values():
            if hasattr(m, 'reset'):
                m.reset()

    def training_step(self, batch, batch_idx):
        im = batch["image_patch"]
        # print('img shape:', im.shape)
        x = batch[self.input_key].float()
        y = batch[self.target_key].float()
        y_hat = self(x)
        # loss = self.loss_fn(y_hat, y)
        loss_dict = self.loss_fn(y_hat, y)
        # combined
        self.log("train_loss", loss_dict["mixed"],
                prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        # components
        self.log("train_loss/primary",   loss_dict["primary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train_loss/secondary", loss_dict["secondary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))


        # self._train_preds.append(y_hat.detach().flatten())
        # self._train_gts.append(  y.detach().flatten())
        # print(f"[Train]  x.shape={x.shape}, y.shape={y.shape}")
        # print(f"[Train] y_hat.shape={y_hat.shape}")
        # print(f"[Train] loss computed on y_hat of shape {y_hat.shape} and y of shape {y.shape}")

        y_int = y
        for name, metric in self.metrics.items():
            freq = self.train_metric_frequencies.get(name, self.train_freq)
            if (self.current_epoch + 1) % freq != 0:
                continue
            result = metric(y_hat, y_int)
            if isinstance(result, dict):
                for subname, value in result.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    self.log(f"train_metrics/{name}_{subname}", value, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)
            else:
                if isinstance(result, torch.Tensor):
                    result = result.item()
                self.log(f"train_metrics/{name}", result, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)

        
        # ——— Log GT / Pred stats for TensorBoard ———
        # flatten tensors
        pred_flat = y_hat.flatten()
        gt_flat   = y.flatten()
        # GT statistics
        self.log("train_mmm/gt_min",   torch.min(gt_flat),   on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train_mmm/gt_max",   torch.max(gt_flat),   on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train_mmm/gt_mean",  torch.mean(gt_flat),  on_step=False, on_epoch=True, batch_size=x.size(0))
        # Pred statistics
        self.log("train_mmm/pred_min", torch.min(pred_flat), on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train_mmm/pred_max", torch.max(pred_flat), on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train_mmm/pred_mean",torch.mean(pred_flat),on_step=False, on_epoch=True, batch_size=x.size(0))


        return {"loss": loss_dict["mixed"], "predictions": y_hat, "gts": y}
    
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
            y_hat = self.validator.run_chunked_inference(self.model, x, self.divisible_by)

        # loss = self.loss_fn(y_hat, y)
        # loss_dict = self.loss_fn(y_hat, y)
        loss_dict = self._safe_loss(y_hat, y)
        # combined
        self.log("val_loss", loss_dict["mixed"],
                prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        # components
        self.log("val_loss/primary",   loss_dict["primary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_loss/secondary", loss_dict["secondary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))


        # self._val_preds.append(y_hat.detach().flatten())
        # self._val_gts  .append(y.detach().flatten())
        # print(f"[Val]  x.shape={x.shape}, y.shape={y.shape}")
        # print(f"[Val] y_hat.shape={y_hat.shape}")
        # print(f"[Val] loss computed on y_hat of shape {y_hat.shape} and y of shape {y.shape}")

        y_int = y
        for name, metric in self.metrics.items():
            freq = self.val_metric_frequencies.get(name, self.val_freq)
            if (self.current_epoch+1) % freq == 0:
                # print('y_int.shape', y_hat.shape)
                # print('y_int.shape', y_hat.shape)
                # print(name, metric(y_hat, y_int))
                result = metric(y_hat, y_int)
                if isinstance(result, dict):
                    for subname, value in result.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        self.log(f"val_metrics/{name}_{subname}", value, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)
                else:
                    if isinstance(result, torch.Tensor):
                        result = result.item()
                    self.log(f"val_metrics/{name}", result, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)

        # ——— Log GT / Pred stats for TensorBoard ———
        # flatten tensors
        pred_flat = y_hat.flatten()
        gt_flat   = y.flatten()
        # GT statistics
        self.log("val_mmm/gt_min",   torch.min(gt_flat),   on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_mmm/gt_max",   torch.max(gt_flat),   on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_mmm/gt_mean",  torch.mean(gt_flat),  on_step=False, on_epoch=True, batch_size=x.size(0))
        # Pred statistics
        self.log("val_mmm/pred_min", torch.min(pred_flat), on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_mmm/pred_max", torch.max(pred_flat), on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_mmm/pred_mean",torch.mean(pred_flat),on_step=False, on_epoch=True, batch_size=x.size(0))

        return {"predictions": y_hat, "val_loss": loss_dict["mixed"], "gts": y}
    
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
            y_hat = self.validator.run_chunked_inference(self.model, x, self.divisible_by)

        # loss = self.loss_fn(y_hat, y)
        # loss_dict = self.loss_fn(y_hat, y)
        loss_dict = self._safe_loss(y_hat, y)
        # combined
        self.log("test_loss", loss_dict["mixed"],
                prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        # components
        self.log("test_loss/primary",   loss_dict["primary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("test_loss/secondary", loss_dict["secondary"],
                prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))

        y_int = y
        for name, metric in self.metrics.items():
            result = metric(y_hat, y_int)
            if isinstance(result, dict):
                for subname, value in result.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    self.log(f"test_metrics/{name}_{subname}", value, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)
            else:
                if isinstance(result, torch.Tensor):
                    result = result.item()
                self.log(f"test_metrics/{name}", result, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0),)

        return {"predictions": y_hat, "test_loss": loss_dict["mixed"], "gts": y}

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

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Try a strict load first; if it fails because of unexpected or
        missing keys, fall back to strict=False so training can resume
        without losing optimizer / scheduler / epoch states.
        """
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as err:
            self.py_logger.warning(f"Strict load failed: {err}\nRetrying with strict=False.")
            return super().load_state_dict(state_dict, strict=False)
    
        # ──────────────────────────────────────────────────────────────
    def _safe_loss(self, y_hat: torch.Tensor, y: torch.Tensor):
        """
        Try to compute the configured loss; if it fails (e.g. because no
        grad info is available during val/test), return zeros so the run
        keeps going and log a warning.
        """
        if not self.compute_val_loss:
            z = torch.tensor(0.0, device=y_hat.device)
            return {"mixed": z, "primary": z, "secondary": z}
        try:
            return self.loss_fn(y_hat, y)
        except RuntimeError as err:
            self.py_logger.warning(f"Loss calculation skipped: {err}")
            z = torch.tensor(0.0, device=y_hat.device)
            return {"mixed": z, "primary": z, "secondary": z}


# ------------------------------------
# core/validator.py
# ------------------------------------
"""
Validator module for handling chunked inference in validation/test phases.
Now supports **both 2‑D (N, C, H, W)** and **3‑D (N, C, D, H, W)** inputs.
Implemented with robust size handling
and automatic dimensionality detection.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import process_in_chuncks  # unchanged – must support N‑D windows


class Validator:
    """Chunked, overlap‑tiled inference for 2‑D **or** 3‑D data.

    • Works with arbitrary batch size and channel count.
    • Pads the sample so every spatial dimension is divisible by a given
      *divisor* (default: 16) before tiling, then removes the pad.
    • Uses `patch_size` and `patch_margin` to create overlapping tiles.
      Only the *centre* region of each model prediction is kept and
      stitched together.

    Parameters
    ----------
    config : dict
        Dictionary with at least the keys:
            ``patch_size``   – tuple/list[int] (len == 2 or 3)
            ``patch_margin`` – tuple/list[int] same length as
                                 ``patch_size``
        Any other keys are ignored by this class.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patch_size: Tuple[int, ...] = tuple(config.get("patch_size", (512, 512)))
        self.patch_margin: Tuple[int, ...] = tuple(config.get("patch_margin", (32, 32)))
        self.logger = logging.getLogger(__name__)

        if len(self.patch_size) != len(self.patch_margin):
            raise ValueError(
                "patch_size %s and patch_margin %s must have the same number of dimensions"
                % (self.patch_size, self.patch_margin)
            )
        if len(self.patch_size) not in (2, 3):
            raise ValueError("Only 2‑D or 3‑D data are supported (got %d‑D)" % len(self.patch_size))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _calc_div16_pad(size: int, divisor: int = 16) -> int:
        """Return how many voxels/pixels must be *added to the right* so that
        *size* becomes divisible by *divisor* (default 16)."""
        return (divisor - size % divisor) % divisor

    def _pad_to_valid_size(
        self, image: torch.Tensor, divisor: int = 16
    ) -> Tuple[torch.Tensor, List[int]]:
        """Pad *image* so *all* spatial dims are divisible by ``divisor``.

        Only **right/bottom/back** padding is applied (no shift of origin).

        Parameters
        ----------
        image : torch.Tensor
            ``(N, C, H, W)`` or ``(N, C, D, H, W)`` tensor.
        divisor : int, optional
            The divisor (default 16).

        Returns
        -------
        image_padded : torch.Tensor
        pads         : list[int]
            Per‑dimension pad added (*same order as image spatial dims*).
        """

        spatial = image.shape[2:]
        pad_each: List[int] = [self._calc_div16_pad(s, divisor) for s in spatial]

        # Build pad tuple for F.pad – must be *reversed* order, one (left,right)
        # pair per dim starting with the last spatial dim.
        pad_tuple: List[int] = []
        for p in reversed(pad_each):
            pad_tuple.extend([0, p])  # (left = 0, right = p)

        if any(pad_each):
            try:
                image = F.pad(image, pad_tuple, mode="reflect")
            except RuntimeError:
                # "reflect" not implemented for 5‑D – fall back gracefully.
                image = F.pad(image, pad_tuple, mode="replicate")
        return image, pad_each

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------
    def run_chunked_inference(
        self,
        model: nn.Module,
        image: torch.Tensor,
        divisible_by: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Full‑image/volume inference with overlapping tiles.

        Workflow (N‑D):
        1) **Margin pad** by ``patch_margin`` (reflect/replicate).
        2) **Div‑16 pad** so every spatial dim is divisible by 16.
        3) **Sliding‑window** inference:
            • window      = ``patch_size``
            • window step = ``patch_size − 2*patch_margin``
            • model is applied on each window; only the *centre* region
              is placed into the output canvas.
        4) **Remove** the div‑16 pad.
        5) **Remove** the initial margin pad.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(N, out_channels, *original_spatial*)``.
        """

        if device is None:
            device = next(model.parameters()).device

        model.eval()
        image = image.to(device)

        ndim = len(self.patch_size)  # 2 or 3
        if image.dim() != ndim + 2:
            raise ValueError(
                f"Input tensor dim {image.dim()} does not match patch_size ndim {ndim}"
            )

        # ----------------------------------------------------------
        # (A) First pad by the desired margins so borders get context
        # ----------------------------------------------------------
        if any(self.patch_margin):
            # Build pad tuple [ ... (left,right) per dim … ]
            pad_tuple: List[int] = []
            for m in reversed(self.patch_margin):
                pad_tuple.extend([m, m])
            try:
                image = F.pad(image, pad_tuple, mode="reflect")
            except RuntimeError:
                image = F.pad(image, pad_tuple, mode="replicate")

        # ----------------------------------------------------------
        # (B) Second pad until all dims divisible by 
        # ----------------------------------------------------------
        padded_image, pad_div16 = self._pad_to_valid_size(image, divisible_by)
        N, C, *spatial_pad = padded_image.shape

        # ----------------------------------------------------------
        # (C) Dummy forward to figure out #out channels
        # ----------------------------------------------------------
        with torch.no_grad():
            test_sizes = [
                min(s, p + 2 * m)
                for s, p, m in zip(spatial_pad, self.patch_size, self.patch_margin)
            ]
            slices: List[slice] = [slice(None), slice(None)] + [slice(0, t) for t in test_sizes]
            test_patch = padded_image[tuple(slices)]
            test_patch, _ = self._pad_to_valid_size(test_patch, divisible_by)
            out_channels = model(test_patch).shape[1]

        # Allocate output canvas
        output_shape = (N, out_channels, *spatial_pad)
        output = torch.zeros(output_shape, device=device, dtype=padded_image.dtype)

        # ----------------------------------------------------------
        # (D) Sliding‑window inference
        # ----------------------------------------------------------
        def _process(chunk: torch.Tensor) -> torch.Tensor:  # noqa: D401
            with torch.no_grad():
                return model(chunk)

        with torch.no_grad():
            output = process_in_chuncks(
                padded_image,
                output,
                _process,
                list(self.patch_size),
                list(self.patch_margin),
            )

        # ----------------------------------------------------------
        # (E) Remove the div‑16 pad (right/bottom/back only)
        # ----------------------------------------------------------
        if any(pad_div16):
            slices: List[slice] = [slice(None), slice(None)]
            for p in pad_div16:
                slices.append(slice(None, -p if p else None))
            output = output[tuple(slices)]

        # ----------------------------------------------------------
        # (F) Remove the initial margin pad (all sides)
        # ----------------------------------------------------------
        if any(self.patch_margin):
            slices = [slice(None), slice(None)]
            for i, m in enumerate(self.patch_margin):
                end = output.shape[2 + i] - m
                slices.append(slice(m, end))
            output = output[tuple(slices)]

        return output


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
# core/mix_loss.py
# ------------------------------------
import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Union


class MixedLoss(nn.Module):
    """
    Primary loss always active; secondary loss kicks in
    *only* during training (module.train()) and after
    `start_epoch`.  Nothing extra is required in val/test.
    """

    def __init__(
        self,
        primary_loss: Union[nn.Module, Callable],
        secondary_loss: Optional[Union[nn.Module, Callable]] = None,
        alpha: float = 0.5,
        start_epoch: int = 0,
    ):
        super().__init__()
        self.primary_loss = primary_loss
        self.secondary_loss = secondary_loss
        self.alpha = alpha
        self.start_epoch = start_epoch
        self.current_epoch = 0  # call update_epoch() each epoch

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------
    def update_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, y_pred, y_true) -> Dict[str, torch.Tensor]:
        p = self.primary_loss(y_pred, y_true)

        use_secondary = (
            self.secondary_loss is not None
            and self.training                       # train() vs eval()
            and self.current_epoch >= self.start_epoch
        )
        if use_secondary:
            s = self.secondary_loss(y_pred, y_true)
        else:
            s = torch.tensor(0.0, device=p.device, dtype=p.dtype)

        m = p + self.alpha * s
        return {"primary": p, "secondary": s, "mixed": m}

# ------------------------------------
# core/__init__.py
# ------------------------------------
"""
Module __init__ file for the core package.

This file imports and exposes the main components from the core package.
"""

from core.loaders.model_loader import load_model
from core.loaders.loss_loader import load_loss
from core.loaders.callback_loader import load_callbacks
from core.loaders.metric_loader import load_metric, load_metrics
from core.loaders.dataloader import SegmentationDataModule
from core.mix_loss import MixedLoss
from core.validator import Validator
from core.logger import setup_logger, ColoredFormatter
from core.checkpoint import CheckpointManager, CheckpointMode
from core.utils import *

from core.general_dataset import GeneralizedDataset, custom_collate_fn, worker_init_fn

__all__ = [
    'load_model',
    'load_loss',
    'load_metric',
    'load_metrics',
    'load_callbacks',
    'SegmentationDataModule',
    'MixedLoss',
    'Validator',
    'setup_logger',
    'ColoredFormatter',
    'CheckpointManager',
    'CheckpointMode',
    'GeneralizedDataset',
    'custom_collate_fn',
    'worker_init_fn',
]


# ------------------------------------
# core/loaders/model_loader.py
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
# core/loaders/dataloader.py
# ------------------------------------
import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from core.general_dataset.base import GeneralizedDataset
from core.general_dataset.collate import custom_collate_fn, worker_init_fn


class SegmentationDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for segmentation tasks wrapping GeneralizedDataset.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config.copy()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Dataloader parameters
        self.batch_size = {
            'train': config.get('train_batch_size', 8),
            'val':   config.get('val_batch_size', 1),
            'test':  config.get('test_batch_size', 1)
        }
        self.num_workers = config.get('num_workers', 4)
        self.pin_memory = config.get('pin_memory', True)

    def setup(self, stage: Optional[str] = None):
        """
        Instantiate datasets for training, validation, and testing.
        """
        def make_cfg(split: str) -> Dict[str, Any]:
            cfg = self.config.copy()
            cfg['split'] = split
            return cfg

        if stage in ('fit', None):
            self.train_dataset = GeneralizedDataset(make_cfg('train'))
            self.val_dataset = GeneralizedDataset(make_cfg('valid'))

        if stage in ('test', None):
            self.test_dataset = GeneralizedDataset(make_cfg('test'))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size['train'],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size['val'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size['test'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )


# ------------------------------------
# core/loaders/loss_loader.py
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
# core/loaders/callback_loader.py
# ------------------------------------
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
    GradPlotCallback
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
                    params
                )
            )

        elif name == "GradPlotCallback":
            callbacks.append(
                GradPlotCallback(
                    input_key=params.get("input_key", "image_patch"),
                    every_n_epochs=params.get("every_n_epochs", 5),
                    max_samples=params.get("max_samples", 4),
                    cmap=params.get("cmap", "turbo"),
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
# nn_GroupNorm = lambda three_dimensional: nn.BatchNorm3d if three_dimensional else nn.BatchNorm2d
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
                 dropout=0.3, norm_type='batch', num_groups=8, pooling="max", three_dimensional=False):
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
        # if batch_norm:
        #     # layers.append(nn_BatchNorm(_3d)(out_channels))
        #     layers.append(nn.GroupNorm(8, out_channels))
        if norm_type=='batch':
            layers.append(nn_BatchNorm(_3d)(out_channels))
        elif norm_type=='group':
            layers.append(nn.GroupNorm(num_groups, out_channels))
        # elif norm_type=='instance':
        #     layers.append(nn.InstanceNorm3d(out_channels) if _3d else nn.InstanceNorm2d(out_channels))
        # else:  # norm_type is None → no normalization
        #     pass
        layers.append(nn.ReLU(inplace=True))
        for i in range(n_convs-1):
            layers.append(nn_Conv(_3d)(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=BIAS))
#             if batch_norm:
# #                 layers.append(nn_BatchNorm(_3d)(out_channels))
#                 layers.append(nn.GroupNorm(8, out_channels))
            if norm_type=='batch':
                layers.append(nn_BatchNorm(_3d)(out_channels))
            elif norm_type=='group':
                layers.append(nn.GroupNorm(num_groups, out_channels))
            # elif norm_type=='instance':
            #     layers.append(nn.InstanceNorm3d(out_channels) if _3d else nn.InstanceNorm2d(out_channels))
            # else:  # norm_type is None → no normalization
            #     pass
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn_Dropout(_3d)(p=dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

class UpBlock(nn.Module):

    def __init__(self, in_channels, n_convs=2, dropout=0.3,
                 norm_type='batch', num_groups=8, upsampling='deconv', three_dimensional=False, align_corners=False):
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
        elif upsampling in ['nearest', 'bilinear', 'trilinear']:
            mode = upsampling
            if _3d and upsampling == 'bilinear':
                mode = 'trilinear'
            # align_corners only applies to non-nearest
            uc = {} if mode=='nearest' else {'align_corners': align_corners}
            self.upsampling = nn.Sequential(
                                nn.Upsample(size=None, scale_factor=2, mode=mode, **uc),
                                nn_Conv(_3d)(in_channels, out_channels, kernel_size=1, padding=0, bias=BIAS))
        else:
            raise ValueError("Unrecognized upsampling option {}".fomrat(upsampling))
        

        layers = []
        layers.append(nn_Conv(_3d)(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=BIAS))
#         if batch_norm:
# #             layers.append(nn_BatchNorm(_3d)(out_channels))
#             layers.append(nn.GroupNorm(8, out_channels))
        if norm_type=='batch':
            layers.append(nn_BatchNorm(_3d)(out_channels))
        elif norm_type=='group':
            layers.append(nn.GroupNorm(num_groups, out_channels))
        # elif norm_type=='instance':
        #     layers.append(nn.InstanceNorm3d(out_channels) if _3d else nn.InstanceNorm2d(out_channels))
        # else:  # norm_type is None → no normalization
        #     pass
        layers.append(nn.ReLU(inplace=True))
        for i in range(n_convs-1):
            layers.append(nn_Conv(_3d)(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=BIAS))
#             if batch_norm:
# #                 layers.append(nn_BatchNorm(_3d)(out_channels))
#                 layers.append(nn.GroupNorm(8, out_channels))
            if norm_type=='batch':
                layers.append(nn_BatchNorm(_3d)(out_channels))
            elif norm_type=='group':
                layers.append(nn.GroupNorm(num_groups, out_channels))
            # elif norm_type=='instance':
            #     layers.append(nn.InstanceNorm3d(out_channels) if _3d else nn.InstanceNorm2d(out_channels))
            # else:  # norm_type is None → no normalization
            #     pass
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn_Dropout(_3d)(p=dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, input, to_stack):

        x = self.upsampling(input)

        x = torch.cat([to_stack, x], dim=1)

        return self.layers(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, m_channels=64, out_channels=2, n_convs=1,
                 n_levels=3, dropout=0.0, norm_type='batch', num_groups=8, upsampling='bilinear',
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
        self.norm_type = norm_type 
        self.num_groups = num_groups
        self.upsampling = upsampling
        self.pooling = pooling
        self.three_dimensional = three_dimensional
        _3d = three_dimensional
        self.apply_final_relu = apply_final_relu

        channels = [2**x*m_channels for x in range(0, self.n_levels+1)]

        down_block = lambda inch, outch, is_first: DownBlock(inch, outch, is_first, n_convs,
                                                             dropout, norm_type, num_groups, pooling,
                                                             three_dimensional)
        up_block = lambda inch: UpBlock(inch, n_convs, dropout, norm_type, num_groups,
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

