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

def continue_training():
    """Continue training from checkpoint for 20 more epochs"""
    
    print("=== CONTINUING TRAINING FROM CHECKPOINT ===")
    
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
    
    # Load existing checkpoint
    checkpoint_path = 'checkpoints_regression/final_50_epochs_model.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded existing model from {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop for 10 more epochs
    num_epochs = 10
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {num_epochs} more epochs...")
    print(f"Validation will be performed every 5 epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in train_pbar:
            images = batch['image'].to(device)
            targets = batch['distance_map'].to(device)
            
            # Pad images to make them divisible by 4
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
            loss = criterion(outputs, targets.unsqueeze(1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            
            with torch.no_grad():
                for batch in val_pbar:
                    images = batch['image'].to(device)
                    targets = batch['distance_map'].to(device)
                    
                    # Pad images
                    h, w = images.shape[2], images.shape[3]
                    pad_h = (4 - h % 4) % 4
                    pad_w = (4 - w % 4) % 4
                    
                    if pad_h > 0 or pad_w > 0:
                        images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                        targets = torch.nn.functional.pad(targets, (0, pad_w, 0, pad_h), mode='reflect')
                    
                    outputs = model(images)
                    
                    # Remove padding
                    if pad_h > 0 or pad_w > 0:
                        outputs = outputs[:, :, :h, :w]
                        targets = targets[:, :h, :w]
                    
                    loss = criterion(outputs, targets.unsqueeze(1))
                    
                    val_loss += loss.item()
                    val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoints_regression/continued_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Plot extended training curves
    plt.figure(figsize=(12, 8))
    
    # Create x-axis for total epochs (50 + 10 = 60)
    total_epochs = 60
    x_epochs = np.arange(1, total_epochs + 1)
    
    # Previous training losses (estimated from the final loss we saw)
    # We'll use the final loss from the previous training as a starting point
    previous_train_losses = [191.41, 178.93, 173.65, 169.12, 163.43, 158.53, 155.20, 150.40, 145.95, 141.55, 138.00, 130.73, 127.58, 122.36, 118.20, 113.22, 108.52, 104.38, 100.27, 95.80, 91.27, 85.91, 81.62, 77.97, 76.46, 71.03, 69.15, 67.54, 65.14, 64.49, 63.62, 61.06, 59.52, 58.76, 57.87, 57.44, 55.23, 56.21, 56.32, 55.12, 58.04, 56.79, 55.40, 55.18, 53.70, 53.03, 52.59, 52.47, 52.82, 52.13]
    previous_val_losses = [244.37, 215.12, 245.24, 194.72, 182.98, 129.71, 120.15, 124.84, 148.66, 136.31, 145.01, 125.72, 95.06, 91.99, 102.30, 94.53, 99.47, 118.10, 100.51, 83.54, 74.92, 100.53, 84.22, 89.09, 63.13, 64.11, 59.95, 71.26, 63.48, 60.27, 87.96, 61.21, 57.34, 56.34, 54.90, 55.61, 63.81, 70.43, 58.44, 55.20, 66.79, 55.98, 54.57, 54.16, 53.69, 55.67, 53.04, 51.09, 54.92, 50.71]
    
    # Combine previous and new losses
    all_train_losses = previous_train_losses + train_losses
    all_val_losses = previous_val_losses + val_losses
    
    plt.plot(x_epochs[:len(all_train_losses)], all_train_losses, label='Training Loss', linewidth=2)
    plt.plot(x_epochs[:len(all_val_losses)], all_val_losses, label='Validation Loss', linewidth=2)
    
    # Add vertical line to show where continued training started
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Continued Training Start')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Extended Training Curves (50 + 10 epochs = 60 total)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('extended_training_curves_60_epochs.png', dpi=300, bbox_inches='tight')
    print("Extended training curves saved as extended_training_curves_60_epochs.png")
    
    # Save final model
    final_model_path = 'checkpoints_regression/final_60_epochs_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Print summary
    print("\n=== TRAINING SUMMARY ===")
    print(f"Previous training (50 epochs): Final loss = {previous_train_losses[-1]:.4f}")
    print(f"Continued training (10 epochs): Final loss = {train_losses[-1]:.4f}")
    print(f"Total improvement: {previous_train_losses[-1] - train_losses[-1]:.4f}")
    print(f"Final model saved: {final_model_path}")
    print(f"Extended curves saved: extended_training_curves_60_epochs.png")
    
    print("\n✅ Extended training completed successfully!")

if __name__ == "__main__":
    continue_training() 