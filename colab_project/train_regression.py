import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from core.regression_dataset import RegressionDataset
from models.base_models import UNet
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def train_regression():
    """Train regression model for distance map prediction"""
    
    print("=== TRAINING REGRESSION MODEL ===")
    
    # Load configuration
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = RegressionDataset(config, split='train')
    val_dataset = RegressionDataset(config, split='valid')
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model using existing UNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use the existing UNet with appropriate parameters for regression
    model = UNet(
        in_channels=3,      # RGB images
        m_channels=64,      # Base number of channels
        out_channels=1,     # Single channel for distance map regression
        n_convs=2,          # Number of convolutions per block
        n_levels=2,         # Number of levels (depth) - reduced to 2 for compatibility
        dropout=0.1,        # Dropout rate
        norm_type='batch',  # Normalization type
        upsampling='bilinear',  # Upsampling method
        pooling="max",      # Pooling method
        three_dimensional=False,  # 2D data
        apply_final_relu=False   # No ReLU for regression output
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in train_pbar:
            images = batch['image'].to(device)
            targets = batch['distance_map'].to(device)
            
            # Pad images to make them divisible by 4 (for 2 levels)
            # Calculate padding needed
            h, w = images.shape[2], images.shape[3]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            
            if pad_h > 0 or pad_w > 0:
                images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                targets = torch.nn.functional.pad(targets, (0, pad_w, 0, pad_h), mode='reflect')
            
            # Forward pass
            outputs = model(images)
            
            # Remove padding from outputs and targets for loss calculation
            if pad_h > 0 or pad_w > 0:
                outputs = outputs[:, :, :h, :w]
                targets = targets[:, :h, :w]
            
            # Compute loss
            loss = criterion(outputs, targets.unsqueeze(1))  # Add channel dimension
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch in val_pbar:
                images = batch['image'].to(device)
                targets = batch['distance_map'].to(device)
                
                # Pad images to make them divisible by 4 (for 2 levels)
                # Calculate padding needed
                h, w = images.shape[2], images.shape[3]
                pad_h = (4 - h % 4) % 4
                pad_w = (4 - w % 4) % 4
                
                if pad_h > 0 or pad_w > 0:
                    images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                    targets = torch.nn.functional.pad(targets, (0, pad_w, 0, pad_h), mode='reflect')
                
                outputs = model(images)
                
                # Remove padding from outputs and targets for loss calculation
                if pad_h > 0 or pad_w > 0:
                    outputs = outputs[:, :, :h, :w]
                    targets = targets[:, :h, :w]
                
                loss = criterion(outputs, targets.unsqueeze(1))
                
                val_loss += loss.item()
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save model
    os.makedirs('checkpoints_regression', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints_regression/best_regression_model.pth')
    print("Model saved to checkpoints_regression/best_regression_model.pth")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved as regression_training_curves.png")
    
    # Test prediction on a sample
    print("\nTesting prediction on a sample...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        images = sample_batch['image'].to(device)
        targets = sample_batch['distance_map'].to(device)
        
        outputs = model(images)
        
        # Visualize results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i in range(2):  # Show 2 samples
            # Original image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 0].imshow(img / 255.0)
            axes[i, 0].set_title(f'Sample {i+1}: Original Image')
            axes[i, 0].axis('off')
            
            # Target distance map
            target = targets[i].cpu().numpy()
            im1 = axes[i, 1].imshow(target, cmap='hot')
            axes[i, 1].set_title(f'Sample {i+1}: Target Distance Map')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1])
            
            # Predicted distance map
            pred = outputs[i, 0].cpu().numpy()
            im2 = axes[i, 2].imshow(pred, cmap='hot')
            axes[i, 2].set_title(f'Sample {i+1}: Predicted Distance Map')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2])
        
        plt.tight_layout()
        plt.savefig('regression_predictions.png', dpi=300, bbox_inches='tight')
        print("Predictions visualization saved as regression_predictions.png")
    
    print("\n✅ Regression training completed successfully!")

if __name__ == "__main__":
    train_regression() 