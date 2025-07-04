import os
import pytorch_lightning as pl

class PeriodicCheckpoint(pl.Callback):
    """
    Save the trainer / model state every `every_n_epochs` epochs.

    Args
    ----
    dirpath : str
        Where the *.ckpt* files will be written.
    every_n_epochs : int
        Save interval.
    prefix : str
        Filename prefix (default: "epoch").
    """

    def __init__(self, dirpath: str, every_n_epochs: int = 5, prefix: str = "epoch"):
        super().__init__()
        self.dirpath = dirpath
        self.every_n_epochs = every_n_epochs
        self.prefix = prefix
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch + 1  # epochs are 0-indexed internally
        if epoch % self.every_n_epochs == 0:
            filename = f"{self.prefix}{epoch:06d}.ckpt"
            ckpt_path = os.path.join(self.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            # optional: log the path so you can grep it later
            pl_module.logger.experiment.add_text("checkpoints/saved", ckpt_path, epoch)
