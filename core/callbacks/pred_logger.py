
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

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
        if (trainer.current_epoch+1) % self.log_every_n_epochs == 0:
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
           or ((trainer.current_epoch+1) % self.log_every_n_epochs != 0):
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
