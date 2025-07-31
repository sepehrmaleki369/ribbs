import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import yaml
from core.regression_dataset import RegressionDataset
from models.base_models import UNet
from torch.utils.data import DataLoader
import os

def show_improved_predictions():
    """Show predictions using the improved model (30 epochs total)"""
    
    print("=== SHOWING IMPROVED PREDICTIONS ===")
    
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
    
    # Load the improved model (60 epochs total)
    model_path = 'checkpoints_regression/final_60_epochs_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded improved model from {model_path}")
    else:
        print(f"❌ Improved model not found at {model_path}")
        return
    
    model.eval()
    
    # Create validation dataset
    val_dataset = RegressionDataset(config, split='valid')
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Make predictions
    print("\nMaking predictions with improved model...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            targets = batch['distance_map'].to(device)
            
            # Pad images to make them divisible by 4
            h, w = images.shape[2], images.shape[3]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            
            if pad_h > 0 or pad_w > 0:
                images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
            
            # Forward pass
            outputs = model(images)
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                outputs = outputs[:, :, :h, :w]
            
            # Visualize results
            fig, axes = plt.subplots(2, 5, figsize=(25, 10))
            fig.suptitle(f'Improved Regression Predictions (60 epochs) - Batch {batch_idx+1}', fontsize=16)
            
            for i in range(2):  # Show 2 samples
                # Original image
                img = images[i].cpu().permute(1, 2, 0).numpy()
                axes[i, 0].imshow(img / 255.0)
                axes[i, 0].set_title(f'Sample {i+1}: Original Image')
                axes[i, 0].axis('off')
                
                # Target distance map
                target = targets[i].cpu().numpy()
                im1 = axes[i, 1].imshow(target, cmap='hot')
                axes[i, 1].set_title(f'Sample {i+1}: Target Distance Map\nRange: [{target.min():.1f}, {target.max():.1f}]')
                axes[i, 1].axis('off')
                plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
                
                # Predicted distance map
                pred = outputs[i, 0].cpu().numpy()
                im2 = axes[i, 2].imshow(pred, cmap='hot')
                axes[i, 2].set_title(f'Sample {i+1}: Predicted Distance Map\nRange: [{pred.min():.1f}, {pred.max():.1f}]')
                axes[i, 2].axis('off')
                plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
                
                # Difference (Target - Prediction)
                diff = target - pred
                im3 = axes[i, 3].imshow(diff, cmap='RdBu_r', vmin=-diff.max(), vmax=diff.max())
                axes[i, 3].set_title(f'Sample {i+1}: Difference (Target - Pred)\nMAE: {np.abs(diff).mean():.2f}')
                axes[i, 3].axis('off')
                plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
                
                # Absolute error
                abs_error = np.abs(diff)
                im4 = axes[i, 4].imshow(abs_error, cmap='hot')
                axes[i, 4].set_title(f'Sample {i+1}: Absolute Error\nMax: {abs_error.max():.2f}')
                axes[i, 4].axis('off')
                plt.colorbar(im4, ax=axes[i, 4], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(f'predictions/regression_predictions_60_epochs.png', dpi=300, bbox_inches='tight')
            print(f"Predictions saved as predictions/regression_predictions_60_epochs.png")
            
            # Calculate metrics
            mae = torch.nn.functional.l1_loss(outputs, targets.unsqueeze(1)).item()
            mse = torch.nn.functional.mse_loss(outputs, targets.unsqueeze(1)).item()
            rmse = torch.sqrt(torch.nn.functional.mse_loss(outputs, targets.unsqueeze(1))).item()
            
            print(f"\nBatch {batch_idx+1} Metrics (Improved Model):")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            
            # Only process first batch for visualization
            break
    
    # Create comparison with previous model
    print("\nCreating comparison with previous model...")
    
    # Load previous model (10 epochs)
    previous_model = UNet(
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
    
    previous_model_path = 'checkpoints_regression/best_regression_model.pth'
    if os.path.exists(previous_model_path):
        previous_model.load_state_dict(torch.load(previous_model_path, map_location=device))
        previous_model.eval()
        
        # Get a sample batch
        sample_batch = next(iter(val_loader))
        images = sample_batch['image'].to(device)
        targets = sample_batch['distance_map'].to(device)
        
        # Pad images
        h, w = images.shape[2], images.shape[3]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        
        if pad_h > 0 or pad_w > 0:
            images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Get predictions from both models
        with torch.no_grad():
            previous_outputs = previous_model(images)
            improved_outputs = model(images)
            
            if pad_h > 0 or pad_w > 0:
                previous_outputs = previous_outputs[:, :, :h, :w]
                improved_outputs = improved_outputs[:, :, :h, :w]
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Model Comparison: 10 epochs vs 60 epochs', fontsize=16)
        
        for i in range(2):  # Show 2 samples
            # Original image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 0].imshow(img / 255.0)
            axes[i, 0].set_title(f'Sample {i+1}: Original Image')
            axes[i, 0].axis('off')
            
            # Previous model prediction (10 epochs)
            prev_pred = previous_outputs[i, 0].cpu().numpy()
            im1 = axes[i, 1].imshow(prev_pred, cmap='hot')
            axes[i, 1].set_title(f'Sample {i+1}: 10 epochs\nRange: [{prev_pred.min():.1f}, {prev_pred.max():.1f}]')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # Improved model prediction (60 epochs)
            imp_pred = improved_outputs[i, 0].cpu().numpy()
            im2 = axes[i, 2].imshow(imp_pred, cmap='hot')
            axes[i, 2].set_title(f'Sample {i+1}: 60 epochs\nRange: [{imp_pred.min():.1f}, {imp_pred.max():.1f}]')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            # Difference between models
            model_diff = imp_pred - prev_pred
            im3 = axes[i, 3].imshow(model_diff, cmap='RdBu_r', vmin=-model_diff.max(), vmax=model_diff.max())
            axes[i, 3].set_title(f'Sample {i+1}: Model Difference\n(60 - 10 epochs)')
            axes[i, 3].axis('off')
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig('predictions/model_comparison_10_vs_60_epochs.png', dpi=300, bbox_inches='tight')
        print("Model comparison saved as predictions/model_comparison_10_vs_60_epochs.png")
        
        # Calculate improvement metrics
        target = targets[0].cpu().numpy()
        prev_error = np.abs(target - prev_pred[0]).mean()
        imp_error = np.abs(target - imp_pred[0]).mean()
        improvement = prev_error - imp_error
        
        print(f"\nImprovement Analysis:")
        print(f"  Previous model (10 epochs) MAE: {prev_error:.4f}")
        print(f"  Improved model (60 epochs) MAE: {imp_error:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement/prev_error*100:.1f}%)")
    
    print("\n✅ Improved predictions visualization completed!")
    print("\n📊 Summary:")
    print(f"  - Improved predictions: predictions/regression_predictions_60_epochs.png")
    print(f"  - Model comparison: predictions/model_comparison_10_vs_60_epochs.png")
    print(f"  - Extended training curves: extended_training_curves_60_epochs.png")

if __name__ == "__main__":
    show_improved_predictions() 