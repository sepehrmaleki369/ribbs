import yaml
import torch
from torch.utils.data import DataLoader
from core.regression_dataset import RegressionDataset

def test_custom_regression_dataset():
    """Test the custom regression dataset"""
    
    print("=== TESTING CUSTOM REGRESSION DATASET ===")
    
    # Load the regression dataset configuration
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded successfully")
    
    # Create training dataset
    print("\nCreating training dataset...")
    train_dataset = RegressionDataset(config, split='train')
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Test loading a few samples
    print("\nTesting data loading...")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"Sample {i}:")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Distance map shape: {sample['distance_map'].shape}")
        print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print(f"  Distance map range: [{sample['distance_map'].min():.3f}, {sample['distance_map'].max():.3f}]")
        print()
    
    # Create validation dataset
    print("\nCreating validation dataset...")
    val_dataset = RegressionDataset(config, split='valid')
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Test DataLoader
    print("\nTesting DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Distance map shape: {batch['distance_map'].shape}")
        print(f"  Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        print(f"  Distance map range: [{batch['distance_map'].min():.3f}, {batch['distance_map'].max():.3f}]")
        break
    
    print("\n✅ Custom regression dataset test completed successfully!")

if __name__ == "__main__":
    test_custom_regression_dataset() 