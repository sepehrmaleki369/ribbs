import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from core.regression_dataset import RegressionDataset
from models.base_models import UNet
import os

def save_current_checkpoint():
    """Save the current training state as a checkpoint"""
    
    print("=== SAVING CURRENT CHECKPOINT ===")
    
    # Load configuration
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = UNet(
        in_channels=3,
        m_channels=64,
        out_channels=1,
        n_convs=2,
        n_levels=2,
        dropout=0.1,
        norm_type='batch',
        upsampling='bilinear',
        pooling="max",
        three_dimensional=False,
        apply_final_relu=False
    ).to(device)
    
    # Load the current model state
    current_model_path = 'checkpoints_regression/final_continued_model.pth'
    if os.path.exists(current_model_path):
        model.load_state_dict(torch.load(current_model_path, map_location=device))
        print(f"✅ Loaded current model from {current_model_path}")
    else:
        print(f"❌ Current model not found at {current_model_path}")
        return
    
    # Save as a new checkpoint
    new_checkpoint_path = 'checkpoints_regression/current_training_checkpoint.pth'
    torch.save(model.state_dict(), new_checkpoint_path)
    print(f"✅ Current checkpoint saved to {new_checkpoint_path}")
    
    print(f"\n📁 Checkpoint saved successfully!")
    print(f"   Path: {new_checkpoint_path}")
    print(f"   Model: {current_model_path}")
    print(f"\n🛑 Training can be stopped now.")

if __name__ == "__main__":
    save_current_checkpoint() 