"""
Dataloader module for wrapping GeneralizedDataset.

This module provides functionality to load and configure datasets for training, validation, and testing.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# Import the existing GeneralizedDataset
from core.general_dataset import GeneralizedDataset, custom_collate_fn, worker_init_fn


class SegmentationDataModule((pl.LightningDataModule)):
    """
    DataModule for segmentation tasks that wraps GeneralizedDataset.
    
    This class handles dataset creation and dataloader configuration for training,
    validation, and testing, ensuring that validation/testing use full images without cropping.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SegmentationDataModule.
        
        Args:
            config: Configuration dictionary with dataset parameters
        """
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.logger = logging.getLogger(__name__)
        
        # Extract important config parameters
        self.root_dir = config.get("root_dir")
        self.split_mode = config.get("split_mode", "folder")  # "folder" or "kfold"
        self.fold = config.get("fold", 0)
        self.num_folds = config.get("num_folds", 5)
        self.batch_size = {
            "train": config.get("train_batch_size", 8),
            "val": config.get("val_batch_size", 1),
            "test": config.get("test_batch_size", 1)
        }
        self.num_workers = config.get("num_workers", 4)
        self.pin_memory = config.get("pin_memory", True)
        
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for the specified stage.
        
        Args:
            stage: Stage to set up ('fit' or 'test')
        """
        if stage == 'fit' or stage is None:
            # Training dataset
            train_config = self._get_dataset_config("train")
            self.train_dataset = GeneralizedDataset(train_config)
            
            # Validation dataset - ensure no cropping in validation
            val_config = self._get_dataset_config("valid")
            val_config["validate_road_ratio"] = False  # Don't filter patches by road content
            # For validation, we want to process full images, not crops
            self.val_dataset = GeneralizedDataset(val_config)
            
        if stage == 'test' or stage is None:
            # Test dataset - ensure no cropping in test
            test_config = self._get_dataset_config("test")
            test_config["validate_road_ratio"] = False  # Don't filter patches by road content
            # For testing, we want to process full images, not crops
            self.test_dataset = GeneralizedDataset(test_config)
    
    def _get_dataset_config(self, split: str) -> Dict[str, Any]:
        """
        Get configuration for a specific dataset split.
        
        Args:
            split: Dataset split ('train', 'valid', or 'test')
            
        Returns:
            Configuration dictionary for the specified split
        """
        config = self.config.copy()
        config["split"] = split
        
        # Set split-specific parameters
        if split in ("valid", "test"):
            # For validation and test, we want to process full images when possible
            config["validate_road_ratio"] = False
        
        if self.split_mode == "kfold":
            config["use_splitting"] = True
            config["fold"] = self.fold
            config["num_folds"] = self.num_folds
        
        return config
    
    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader.
        
        Returns:
            DataLoader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Create validation dataloader.
        
        Returns:
            DataLoader for validation
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        Create test dataloader.
        
        Returns:
            DataLoader for testing
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn
        )