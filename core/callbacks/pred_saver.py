import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

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
        self._saved_val_gts = False

    def _should_save(self, epoch: int) -> bool:
        print('epoch, self.every', epoch, self.every)
        return epoch >= self.after and (epoch+1) % self.every == 0

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

    # # ——— TRAINING HOOKS ———
    # def on_train_epoch_start(self, trainer, pl_module):
    #     self._counter = 0

    # def on_train_batch_end(
    #     self,
    #     trainer: pl.Trainer,
    #     pl_module: pl.LightningModule,
    #     outputs: Dict[str, Any],
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx: int = 0,
    # ):
    #     epoch = trainer.current_epoch
    #     if not self._should_save(epoch):
    #         return

    #     preds = outputs.get("predictions")
    #     gts   = outputs.get("gts")
    #     if preds is None or gts is None:
    #         return

    #     preds = preds.detach().cpu().numpy()
    #     gts   = gts.detach().cpu().numpy()
    #     for i in range(preds.shape[0]):
    #         if self.max_samples is not None and self._counter >= self.max_samples:
    #             return
    #         self._save_tensor(preds[i], "train", epoch, batch_idx, i, "pred")
    #         self._save_tensor(gts[i],   "train", epoch, batch_idx, i, "gt")
    #         self._counter += 1

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

        # preds = preds.detach().cpu().numpy()
        # gts   = gts.detach().cpu().numpy()
        # for i in range(preds.shape[0]):
        #     if self.max_samples is not None and self._counter >= self.max_samples:
        #         return
        #     self._save_tensor(preds[i], "val", epoch, batch_idx, i, "pred")
        #     self._save_tensor(gts[i],   "val", epoch, batch_idx, i, "gt")
        #     self._counter += 1

        # always save preds as before
        preds_np = preds.detach().cpu().numpy()
        for i in range(preds_np.shape[0]):
            if self.max_samples is not None and self._counter >= self.max_samples:
                break
            self._save_tensor(preds_np[i], "val", epoch, batch_idx, i, "pred")
            self._counter += 1

        # only save gts once ever
        if gts is not None and not self._saved_val_gts:
            gts_np = gts.detach().cpu().numpy()
            # you might only need to save one batch (or all of them here)
            for i in range(gts_np.shape[0]):
                self._save_tensor(gts_np[i], "val", epoch, batch_idx, i, "gt")
            self._saved_val_gts = True

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
