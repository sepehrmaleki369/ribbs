import os
import shutil
import logging
import zipfile
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class ConfigArchiver(Callback):
    """
    Callback to archive configuration files and source code at the start of training.

    This callback creates:
      - A ZIP archive of config and source
      - A parallel folder copy of the same files (optional)
    """

    def __init__(
        self,
        output_dir: str,
        project_root: str,
        copy_folder: bool = True
    ):
        """
        Initialize the ConfigArchiver callback.

        Args:
            output_dir: Directory to save archives and/or copies
            project_root: Root directory of the project containing the code to archive
            copy_folder: Whether to also copy files into a folder alongside the ZIP
        """
        super().__init__()
        self.output_dir = output_dir
        self.project_root = project_root
        self.copy_folder = copy_folder
        self.logger = logging.getLogger(__name__)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Archive configuration and source code at the start of training.

        Creates a ZIP archive and, if enabled, copies the files to a folder.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        os.makedirs(self.output_dir, exist_ok=True)
        # Use epoch and timestamp for uniqueness
        timestamp = trainer.logger.experiment.current_epoch if hasattr(trainer.logger.experiment, 'current_epoch') else pl_module.current_epoch
        base_name = f"code_snapshot_{timestamp}"

        # Create ZIP archive
        zip_path = os.path.join(self.output_dir, f"{base_name}.zip")
        version = 1
        while os.path.exists(zip_path):
            zip_path = os.path.join(self.output_dir, f"{base_name}_v{version}.zip")
            version += 1

        self.logger.info(f"Creating ZIP archive at {zip_path}")
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            # Archive directories
            for folder in ['configs', 'core', 'models', 'losses', 'metrics', 'scripts', 'callbacks']:
                src_dir = os.path.join(self.project_root, folder)
                if os.path.isdir(src_dir):
                    for root, _, files in os.walk(src_dir):
                        for fname in files:
                            if fname.endswith(('.py', '.yaml', '.yml')):
                                full_path = os.path.join(root, fname)
                                arcname = os.path.relpath(full_path, self.project_root)
                                zipf.write(full_path, arcname)
            # train.py
            train_py = os.path.join(self.project_root, 'train.py')
            if os.path.exists(train_py):
                zipf.write(train_py, 'train.py')
            # seglit_module.py
            seglit_py = os.path.join(self.project_root, 'seglit_module.py')
            if os.path.exists(seglit_py):
                zipf.write(seglit_py, 'seglit_module.py')
        self.logger.info(f"ZIP archive created: {zip_path}")

        # Optionally create a folder copy
        if self.copy_folder:
            copy_path = os.path.join(self.output_dir, base_name)
            if os.path.exists(copy_path):
                copy_path = f"{copy_path}_v{version - 1}"  # same version count
            self.logger.info(f"Copying files to folder {copy_path}")
            os.makedirs(copy_path, exist_ok=True)
            for folder in ['configs', 'core', 'models', 'losses', 'metrics']:
                src_dir = os.path.join(self.project_root, folder)
                dst_dir = os.path.join(copy_path, folder)
                if os.path.isdir(src_dir):
                    shutil.copytree(src_dir, dst_dir)
            # train.py
            if os.path.exists(train_py):
                shutil.copy2(train_py, copy_path)
            # seglit_module.py
            if os.path.exists(seglit_py):
                shutil.copy2(seglit_py, copy_path)
            self.logger.info(f"Folder copy created at: {copy_path}")
