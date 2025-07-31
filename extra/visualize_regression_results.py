import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import yaml
from core.regression_dataset import RegressionDataset
from models.base_models import UNet
from torch.utils.data import DataLoader
import os

def visualize_training_results():
    """Visualize training results and make predictions"""
    
    print("=== VISUALIZING REGRESSION RESULTS ===")
    
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
    
    # Load trained model
    model_path = 'checkpoints_regression/best_regression_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded trained model from {model_path}")
    else:
        print(f"❌ Model not found at {model_path}")
        return
    
    model.eval()
    
    # Create validation dataset
    val_dataset = RegressionDataset(config, split='valid')
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Make predictions
    print("\nMaking predictions on validation samples...")
    
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
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'Regression Predictions - Batch {batch_idx+1}', fontsize=16)
            
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
            
            plt.tight_layout()
            plt.savefig(f'regression_predictions_batch_{batch_idx+1}.png', dpi=300, bbox_inches='tight')
            print(f"Predictions saved as regression_predictions_batch_{batch_idx+1}.png")
            
            # Calculate metrics
            mae = torch.nn.functional.l1_loss(outputs, targets.unsqueeze(1)).item()
            mse = torch.nn.functional.mse_loss(outputs, targets.unsqueeze(1)).item()
            rmse = torch.sqrt(torch.nn.functional.mse_loss(outputs, targets.unsqueeze(1))).item()
            
            print(f"Batch {batch_idx+1} Metrics:")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            
            # Only process first batch for visualization
            break
    
    # Create detailed analysis plots
    print("\nCreating detailed analysis plots...")
    
    # Load training curves
    if os.path.exists('regression_training_curves.png'):
        print("✅ Training curves available: regression_training_curves.png")
    
    # Create histogram of prediction errors
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error histogram
    all_errors = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = batch['distance_map'].to(device)
            
            # Pad images
            h, w = images.shape[2], images.shape[3]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            
            if pad_h > 0 or pad_w > 0:
                images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
            
            outputs = model(images)
            
            if pad_h > 0 or pad_w > 0:
                outputs = outputs[:, :, :h, :w]
            
            errors = (targets.unsqueeze(1) - outputs).cpu().numpy().flatten()
            all_errors.extend(errors)
    
    axes[0].hist(all_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title('Prediction Error Distribution')
    axes[0].set_xlabel('Error (Target - Prediction)')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0].grid(True, alpha=0.3)
    
    # Error statistics
    all_errors = np.array(all_errors)
    axes[1].text(0.1, 0.9, f'Error Statistics:', fontsize=14, fontweight='bold', transform=axes[1].transAxes)
    axes[1].text(0.1, 0.8, f'Mean: {all_errors.mean():.4f}', fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.7, f'Std: {all_errors.std():.4f}', fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.6, f'Min: {all_errors.min():.4f}', fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.5, f'Max: {all_errors.max():.4f}', fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.4, f'MAE: {np.abs(all_errors).mean():.4f}', fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.3, f'RMSE: {np.sqrt(np.mean(all_errors**2)):.4f}', fontsize=12, transform=axes[1].transAxes)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('regression_error_analysis.png', dpi=300, bbox_inches='tight')
    print("Error analysis saved as regression_error_analysis.png")
    
    print("\n✅ Visualization completed!")
    print("\n📊 Summary:")
    print(f"  - Training curves: regression_training_curves.png")
    print(f"  - Predictions: regression_predictions_batch_1.png")
    print(f"  - Error analysis: regression_error_analysis.png")
    print(f"  - Model: {model_path}")

if __name__ == "__main__":
    visualize_training_results() 