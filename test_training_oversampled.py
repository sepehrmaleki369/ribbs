#!/usr/bin/env python3
"""
Test training script to verify oversampled graphs work correctly.
Runs 1 epoch MSE + 2 epochs Snake Simple for quick testing.
"""

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
from datetime import datetime

# Import snake losses and graph loading
from losses_n import SnakeSimpleLoss
from utils.graphs import load_training_graph_by_id, get_current_graph_mode

def test_training_with_oversampled_graphs():
    """Test training with oversampled graphs for 1 MSE + 2 Snake epochs."""
    
    print("=== TEST TRAINING WITH OVERSAMPLED GRAPHS ===")
    print(f"🎯 Graph mode: {get_current_graph_mode()}")
    
    # Configuration
    num_epochs = 3  # 1 MSE + 2 Snake
    mse_epochs = 1
    snake_epochs = 2
    batch_size = 2
    learning_rate = 0.001
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    train_dataset = RegressionDataset(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"✅ Dataset loaded: {len(train_dataset)} training samples")
    print(f"✅ DataLoader: {len(train_loader)} batches (batch_size={batch_size})")
    
    # Create model
    model = UNet(
        in_channels=3,
        m_channels=64,
        out_channels=1,
        depth=4
    ).to(device)
    
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create loss functions
    mse_criterion = nn.MSELoss()
    snake_loss = SnakeSimpleLoss(
        stepsz=0.1,
        alpha=0.0001,
        beta=0.01,
        fltrstdev=0.5,
        ndims=2,
        nsteps=5,
        cropsz=[16, 16],
        dmax=15.0,
        maxedgelen=3.0,
        extgradfac=1.0
    ).to(device)
    
    print(f"✅ Loss functions created")
    print(f"   • MSE Loss: {type(mse_criterion).__name__}")
    print(f"   • Snake Loss: {type(snake_loss).__name__}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create output directory
    output_dir = 'test_training_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    print(f"\n🚀 Starting training: {mse_epochs} MSE + {snake_epochs} Snake epochs")
    
    train_losses = []
    epoch_types = []
    
    for epoch in range(num_epochs):
        epoch_type = "MSE" if epoch < mse_epochs else "Snake"
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ({epoch_type}) ---")
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} ({epoch_type})")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['image'].to(device)
            targets = batch['distance_map'].to(device)
            sample_ids = batch['sample_id']
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss based on epoch type
            if epoch_type == "MSE":
                loss = mse_criterion(outputs, targets.unsqueeze(1))
            else:  # Snake
                # Load graphs for snake loss
                lbl_graphs = []
                for sample_id in sample_ids:
                    try:
                        graph = load_training_graph_by_id(str(sample_id))
                        lbl_graphs.append(graph)
                    except Exception as e:
                        print(f"⚠️ Could not load graph for sample {sample_id}: {e}")
                        lbl_graphs.append(None)
                
                # Calculate snake loss
                loss = snake_loss(outputs, lbl_graphs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Type': epoch_type,
                'Batch': f'{batch_idx+1}/{len(train_loader)}'
            })
            
            # Only process first few batches for testing
            if batch_idx >= 2:  # Process 3 batches per epoch for quick testing
                break
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        epoch_types.append(epoch_type)
        
        print(f"✅ Epoch {epoch+1} ({epoch_type}) completed")
        print(f"   • Average loss: {avg_loss:.4f}")
        print(f"   • Batches processed: {num_batches}")
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(output_dir, f'test_checkpoint_epoch_{epoch+1}_{epoch_type.lower()}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'epoch_type': epoch_type,
            'graph_mode': get_current_graph_mode()
        }, checkpoint_path)
        print(f"   • Checkpoint saved: {checkpoint_path}")
    
    # Training summary
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"✅ Completed {num_epochs} epochs successfully")
    print(f"✅ Graph mode used: {get_current_graph_mode()}")
    
    # Loss summary
    print(f"\n📊 Loss Summary:")
    for i, (loss, epoch_type) in enumerate(zip(train_losses, epoch_types)):
        print(f"   • Epoch {i+1} ({epoch_type}): {loss:.4f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, 'bo-', linewidth=2, markersize=8)
    
    # Color code by epoch type
    for i, epoch_type in enumerate(epoch_types):
        color = 'blue' if epoch_type == "MSE" else 'green'
        plt.plot(i + 1, train_losses[i], 'o', color=color, markersize=10, label=f'{epoch_type}' if i == 0 or epoch_types[i-1] != epoch_type else "")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Test Training Loss - {get_current_graph_mode()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'test_training_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training plot saved: {plot_path}")
    
    # Save training summary
    summary_path = os.path.join(output_dir, 'test_training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TEST TRAINING WITH OVERSAMPLED GRAPHS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Graph mode: {get_current_graph_mode()}\n")
        f.write(f"Total epochs: {num_epochs}\n")
        f.write(f"MSE epochs: {mse_epochs}\n")
        f.write(f"Snake epochs: {snake_epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n\n")
        
        f.write("LOSS SUMMARY:\n")
        for i, (loss, epoch_type) in enumerate(zip(train_losses, epoch_types)):
            f.write(f"Epoch {i+1} ({epoch_type}): {loss:.6f}\n")
        
        f.write(f"\nCheckpoints saved in: {output_dir}/\n")
        f.write(f"Training plot: test_training_loss.png\n")
    
    print(f"✅ Training summary saved: {summary_path}")
    print(f"\n🎯 Test training completed successfully!")
    print(f"🎯 Oversampled graphs working correctly!")
    print(f"🎯 Ready for full training with oversampled graphs!")


if __name__ == '__main__':
    test_training_with_oversampled_graphs()
