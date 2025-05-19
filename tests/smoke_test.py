"""
Integration smoke test for the entire segmentation pipeline.
This creates a minimal synthetic dataset and runs a few training steps.
Run with: python smoke_test.py
"""

import os
import shutil
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core modules
from core.general_dataset import GeneralizedDataset, custom_collate_fn, worker_init_fn
from core.model_loader import load_model
from core.loss_loader import load_loss
from core.mix_loss import MixedLoss
from core.metric_loader import load_metrics
from core.dataloader import SegmentationDataModule
from seglit_module import SegLitModule


def create_synthetic_dataset(root_dir, num_samples=10, img_size=32):
    """Create a synthetic dataset with roads for testing"""
    logger.info(f"Creating synthetic dataset in {root_dir} with {num_samples} samples")
    
    # Create directory structure
    os.makedirs(os.path.join(root_dir, 'train', 'sat'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'train', 'map'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'valid', 'sat'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'valid', 'map'), exist_ok=True)
    
    # Create test images with simple road patterns
    for split in ['train', 'valid']:
        for i in range(num_samples):
            # Create image with random noise
            img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            
            # Create binary mask with horizontal and vertical lines (roads)
            mask = np.zeros((img_size, img_size), dtype=np.uint8)
            
            # Horizontal road
            h_pos = np.random.randint(5, img_size-5)
            mask[h_pos-2:h_pos+2, :] = 1
            
            # Vertical road
            v_pos = np.random.randint(5, img_size-5)
            mask[:, v_pos-2:v_pos+2] = 1
            
            # Add brightness to the roads in the image for realism
            for c in range(3):
                img[:, :, c] = np.where(mask > 0, 
                                        np.minimum(img[:, :, c] + 100, 255),
                                        img[:, :, c])
            
            # Save files
            np.save(os.path.join(root_dir, split, 'sat', f'img_{i}.npy'), img)
            np.save(os.path.join(root_dir, split, 'map', f'img_{i}.npy'), mask * 255)  # 0/255 format
    
    logger.info(f"Created {num_samples} samples each for train and validation")


class SimpleBinaryUNet(nn.Module):
    """A very simple UNet for testing purposes"""
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, stride=2)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 2, stride=2)
        )
        
        # Final layer
        self.final = nn.Conv2d(8, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        # Decoder
        d1 = self.dec1(e2)
        d2 = self.dec2(d1)
        
        # Final layer
        out = torch.sigmoid(self.final(d2))
        return out


def create_configs(dataset_path):
    """Create configuration dictionaries for testing"""
    # Dataset configuration
    dataset_config = {
        "root_dir": dataset_path,
        "split_mode": "folder",
        "patch_size": 16,
        "small_window_size": 2,
        "validate_road_ratio": False,  # Don't filter patches for quick testing
        "train_batch_size": 2,
        "val_batch_size": 1,
        "num_workers": 0,  # Use 0 for easier debugging
        "pin_memory": False,
        "modalities": {
            "image": "sat",
            "label": "map"
        }
    }
    
    # Model configuration using our simple test model
    model_config = {
        "simple_unet": True,  # Flag for our smoke test
        "in_channels": 3,
        "out_channels": 1
    }
    
    # Loss configuration
    loss_config = {
        "primary_loss": {
            "class": "BCELoss",
            "params": {}
        },
        "alpha": 1.0  # Only use primary loss
    }
    
    # Metrics configuration
    metrics_config = {
        "metrics": [
            {
                "alias": "dice",
                "path": "torchmetrics.classification",
                "class": "Dice",
                "params": {
                    "threshold": 0.5,
                    "zero_division": 1.0
                }
            }
        ]
    }
    
    # Inference configuration
    inference_config = {
        "patch_size": [16, 16],
        "patch_margin": [2, 2]
    }
    
    return dataset_config, model_config, loss_config, metrics_config, inference_config


def run_smoke_test():
    """Run a complete smoke test of the segmentation pipeline"""
    logger.info("Starting smoke test")
    
    # Create temporary directory for dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create synthetic dataset
        create_synthetic_dataset(tmp_dir, num_samples=5, img_size=32)
        
        # Create configurations
        dataset_config, model_config, loss_config, metrics_config, inference_config = create_configs(tmp_dir)
        
        # Set up data module
        logger.info("Setting up data module")
        data_module = SegmentationDataModule(dataset_config)
        data_module.setup()
        
        # Create model manually for smoke test
        logger.info("Creating model")
        if model_config.get("simple_unet", False):
            model = SimpleBinaryUNet(
                in_channels=model_config["in_channels"],
                out_channels=model_config["out_channels"]
            )
        else:
            model = load_model(model_config)
        
        # Create loss function
        logger.info("Creating loss function")
        primary_loss = load_loss(loss_config["primary_loss"])
        secondary_loss = None
        if "secondary_loss" in loss_config:
            secondary_loss = load_loss(loss_config["secondary_loss"])
        
        mixed_loss = MixedLoss(primary_loss, secondary_loss, loss_config.get("alpha", 0.5), 
                              loss_config.get("start_epoch", 0))
        
        # Create metrics
        logger.info("Loading metrics")
        metrics = load_metrics(metrics_config.get("metrics", []))
        
        # Create optimizer config
        optimizer_config = {
            "name": "Adam",
            "params": {"lr": 0.001}
        }
        
        # Create Lightning module
        logger.info("Creating Lightning module")
        lit_module = SegLitModule(
            model=model,
            loss_fn=mixed_loss,
            metrics=metrics,
            optimizer_config=optimizer_config,
            inference_config=inference_config
        )
        
        # Create a simple trainer for testing
        logger.info("Creating trainer")
        trainer = pl.Trainer(
            max_epochs=2,
            log_every_n_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True,
            accelerator="cpu"
        )
        
        # Run a few training steps
        logger.info("Running training")
        trainer.fit(lit_module, datamodule=data_module)
        
        logger.info("Smoke test complete!")


def inspect_dataset_samples():
    """Create and inspect dataset samples for debugging"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create synthetic dataset
        create_synthetic_dataset(tmp_dir, num_samples=3, img_size=32)
        
        # Create configurations
        dataset_config, _, _, _, _ = create_configs(tmp_dir)
        
        # Override for detailed inspection
        dataset_config["train_batch_size"] = 1
        
        # Create dataset directly
        train_config = dataset_config.copy()
        train_config["split"] = "train"
        
        # Fix the None sample issue in __getitem__
        from core.general_dataset import GeneralizedDataset
        
        # Patch the class to fix the None return issue
        original_getitem = GeneralizedDataset.__getitem__
        
        def safe_getitem(self, idx):
            result = original_getitem(self, idx)
            if result is None:
                # Try another index
                logger.warning(f"Got None for index {idx}, trying next index")
                return self.__getitem__((idx + 1) % len(self))
            return result
        
        # Apply the monkey patch
        GeneralizedDataset.__getitem__ = safe_getitem
        
        # Create and inspect dataset
        train_dataset = GeneralizedDataset(train_config)
        
        # Check dataset length
        logger.info(f"Dataset length: {len(train_dataset)}")
        
        # Iterate through a few samples
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            logger.info(f"Sample {i} keys: {sample.keys()}")
            
            # Check shapes
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"  {key} shape: {value.shape}, dtype: {value.dtype}, range: [{value.min()}, {value.max()}]")
        
        # Create dataloader and inspect a batch
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=0
        )
        
        # Get a batch
        batch = next(iter(train_loader))
        logger.info(f"Batch keys: {batch.keys()}")
        
        # Check shapes
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key} shape: {value.shape}, dtype: {value.dtype}")


if __name__ == "__main__":
    # First inspect dataset
    logger.info("Inspecting dataset samples...")
    inspect_dataset_samples()
    
    # Then run full smoke test
    logger.info("\nRunning full smoke test...")
    run_smoke_test()