# Test Discrete Distance Map Dataset
# Verify the discrete distance map dataset is working

import torch
import yaml
from core.regression_dataset import RegressionDataset
import numpy as np
import matplotlib.pyplot as plt

def test_discrete_dataset():
    """Test the discrete distance map dataset"""
    
    print("=== TESTING DISCRETE DISTANCE MAP DATASET ===")
    
    # Load configuration
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = RegressionDataset(config, split='train')
    val_dataset = RegressionDataset(config, split='valid')
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ No training samples found!")
        return
    
    # Test loading a sample
    print("\nTesting sample loading...")
    sample = train_dataset[0]
    
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Distance map shape: {sample['distance_map'].shape}")
    print(f"Image dtype: {sample['image'].dtype}")
    print(f"Distance map dtype: {sample['distance_map'].dtype}")
    
    # Check distance map values
    distance_map = sample['distance_map'].numpy()
    unique_values = np.unique(distance_map)
    print(f"Distance map unique values: {len(unique_values)}")
    print(f"Distance map range: [{distance_map.min():.1f}, {distance_map.max():.1f}]")
    print(f"Sample unique values: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")
    
    # Check if values are integers
    is_integer = all(isinstance(val, (int, np.integer)) for val in unique_values)
    has_decimals = any(val % 1 != 0 for val in unique_values)
    
    print(f"All integer values: {'✅' if is_integer else '❌'}")
    print(f"Has decimal values: {'❌' if not has_decimals else '⚠️'}")
    
    # Create visualization
    create_discrete_visualization(sample)
    
    print(f"\n✅ Discrete dataset test completed!")

def create_discrete_visualization(sample):
    """Create a test visualization of the discrete sample"""
    
    # Convert tensors to numpy for visualization
    image = sample['image'].numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    distance_map = sample['distance_map'].numpy()
    
    # Normalize image for display
    if image.max() > 1:
        image = image / 255.0
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image (RGB)')
    axes[0].axis('off')
    
    # Distance map
    im = axes[1].imshow(distance_map, cmap='hot')
    axes[1].set_title('Discrete Distance Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Distance map histogram
    axes[2].hist(distance_map.flatten(), bins=50, alpha=0.7)
    axes[2].set_title('Discrete Distance Map Distribution')
    axes[2].set_xlabel('Distance Value')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('predictions/discrete_dataset_test.png', dpi=300, bbox_inches='tight')
    print(f"Discrete test visualization saved as predictions/discrete_dataset_test.png")

if __name__ == "__main__":
    test_discrete_dataset() 