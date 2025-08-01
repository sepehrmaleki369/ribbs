import os
import numpy as np
import torch
from torch.utils.data import Dataset
from core.general_dataset.io import load_array_from_file, to_tensor as _to_tensor
from core.general_dataset.normalizations import normalize_image
from core.general_dataset.augmentations import augment_images
import random

class RegressionDataset(Dataset):
    """Custom dataset for regression training with distance maps"""
    
    def __init__(self, config, split='train'):
        super().__init__()
        self.config = config
        self.split = split
        self.data_dim = config.get("data_dim", 2)
        self.seed = config.get("seed", 42)
        
        # Set up random number generator
        self.rng = np.random.RandomState(self.seed)
        
        # Load file lists
        self.image_files = []
        self.distance_map_files = []
        
        # Define paths
        image_dir = "drive/training/images_npy"
        distance_map_dir = "drive/training/discrete_distance_maps"  # Changed to discrete distance maps
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
        
        # Create corresponding distance map file names
        for img_file in image_files:
            stem = img_file.replace('.npy', '')
            # Extract the number from the training file name (e.g., "21_training" -> "21")
            number = stem.split('_')[0]
            distance_file = f"{number}_manual1_discrete_distance_map.npy"  # Fixed naming pattern
            
            # Check if distance map file exists
            distance_path = os.path.join(distance_map_dir, distance_file)
            if os.path.exists(distance_path):
                self.image_files.append(os.path.join(image_dir, img_file))
                self.distance_map_files.append(distance_path)
        
        print(f"Found {len(self.image_files)} image-distance map pairs")
        
        # Validation split
        if split == 'valid':
            # Use last 20% for validation
            split_idx = int(len(self.image_files) * 0.8)
            self.image_files = self.image_files[split_idx:]
            self.distance_map_files = self.distance_map_files[split_idx:]
        elif split == 'train':
            # Use first 80% for training
            split_idx = int(len(self.image_files) * 0.8)
            self.image_files = self.image_files[:split_idx]
            self.distance_map_files = self.distance_map_files[:split_idx]
        
        print(f"{split} split: {len(self.image_files)} samples")
        
        # Normalization settings
        self.norm_cfg = config.get("normalization", {})
        
        # Augmentation settings
        self.aug_cfg = config.get("augmentation", [])
        self.augment = split == 'train'
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and distance map
        image = load_array_from_file(self.image_files[idx])
        distance_map = load_array_from_file(self.distance_map_files[idx])
        
        # Ensure distance map is 2D
        if distance_map.ndim == 3:
            distance_map = distance_map[:, :, 0]  # Take first channel if 3D
        
        # Create data dictionary
        data = {
            'image': image,
            'distance_map': distance_map
        }
        
        # Skip normalization and augmentation as requested
        # Convert to tensors with correct dimensions
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if key == 'image':
                    # Convert (H, W, C) to (C, H, W) for PyTorch
                    result[key] = torch.from_numpy(value.transpose(2, 0, 1)).float()
                else:
                    # Distance map is already (H, W), add channel dimension
                    result[key] = torch.from_numpy(value).float()
            else:
                result[key] = value
        
        return result 