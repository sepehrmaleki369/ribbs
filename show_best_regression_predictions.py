import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.base_models import UNet
from core.regression_dataset import RegressionDataset
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm

def show_best_regression_predictions():
    """Show predictions using the best regression model on VALIDATION data"""
    
    print("=== SHOWING BEST REGRESSION MODEL PREDICTIONS (VALIDATION) ===")
    
    # Load configuration
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create VALIDATION dataset
    print("Creating VALIDATION dataset...")
    dataset = RegressionDataset(config, split='valid')
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # One image at a time for visualization
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Load the best model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
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
    
    # Load best model weights
    model_path = 'checkpoints_regression/best_model.pth'
    print(f"Loading model from: {model_path}")
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Available models in checkpoints_regression/:")
        if os.path.exists('checkpoints_regression/'):
            for file in os.listdir('checkpoints_regression/'):
                if file.endswith('.pth'):
                    print(f"  - {file}")
        else:
            print("  No checkpoints_regression directory found")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Create predictions directory
    os.makedirs('predictions/best_model_validation_predictions', exist_ok=True)
    
    # Process all validation samples for visualization
    num_samples = len(dataset)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    fig.suptitle('Best Regression Model Predictions (VALIDATION DATA)', fontsize=16)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Generating validation predictions")):
            # Get data
            images = batch['image'].to(device)
            targets = batch['distance_map'].to(device)
            
            # Print shapes for debugging
            print(f"Validation Sample {i+1} - Images shape: {images.shape}, Targets shape: {targets.shape}")
            
            # Pad images if needed
            h, w = images.shape[2], images.shape[3]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            
            if pad_h > 0 or pad_w > 0:
                images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                targets = torch.nn.functional.pad(targets, (0, pad_w, 0, pad_h), mode='reflect')
            
            # Get prediction
            predictions = model(images)
            
            # Remove padding if it was added
            if pad_h > 0 or pad_w > 0:
                predictions = predictions[:, :, :h, :w]
                # Handle different target dimensions
                if targets.dim() == 4:
                    targets = targets[:, :, :h, :w]
                else:
                    targets = targets[:, :h, :w]
                images = images[:, :, :h, :w]
            
            # Convert to numpy
            image_np = images[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
            
            # Handle different target shapes
            if targets.dim() == 4:
                target_np = targets[0, 0].cpu().numpy()  # (H, W)
            else:
                target_np = targets[0].cpu().numpy()  # (H, W)
            
            if predictions.dim() == 4:
                prediction_np = predictions[0, 0].cpu().numpy()  # (H, W)
            else:
                prediction_np = predictions[0].cpu().numpy()  # (H, W)
            
            # Normalize image for display
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
            
            # Plot results
            if num_samples == 1:
                ax_row = axes
            else:
                ax_row = axes[i]
            
            # Original image
            ax_row[0].imshow(image_np)
            ax_row[0].set_title(f'Validation Sample {i+1}\nOriginal Image')
            ax_row[0].axis('off')
            
            # Ground truth distance map
            im1 = ax_row[1].imshow(target_np, cmap='hot')
            ax_row[1].set_title(f'Ground Truth\nRange: [{target_np.min():.1f}, {target_np.max():.1f}]\nMean: {target_np.mean():.1f}')
            ax_row[1].axis('off')
            plt.colorbar(im1, ax=ax_row[1], fraction=0.046, pad=0.04)
            
            # Predicted distance map
            im2 = ax_row[2].imshow(prediction_np, cmap='hot')
            ax_row[2].set_title(f'Prediction\nRange: [{prediction_np.min():.1f}, {prediction_np.max():.1f}]\nMean: {prediction_np.mean():.1f}')
            ax_row[2].axis('off')
            plt.colorbar(im2, ax=ax_row[2], fraction=0.046, pad=0.04)
            
            # Error map
            error_map = np.abs(target_np - prediction_np)
            im3 = ax_row[3].imshow(error_map, cmap='hot')
            ax_row[3].set_title(f'Absolute Error\nRange: [{error_map.min():.1f}, {error_map.max():.1f}]\nMean: {error_map.mean():.1f}')
            ax_row[3].axis('off')
            plt.colorbar(im3, ax=ax_row[3], fraction=0.046, pad=0.04)
            
            # Save individual prediction
            fig_ind, axes_ind = plt.subplots(1, 4, figsize=(20, 5))
            
            axes_ind[0].imshow(image_np)
            axes_ind[0].set_title('Original Image (Validation)')
            axes_ind[0].axis('off')
            
            im1_ind = axes_ind[1].imshow(target_np, cmap='hot')
            axes_ind[1].set_title(f'Ground Truth\nRange: [{target_np.min():.1f}, {target_np.max():.1f}]')
            axes_ind[1].axis('off')
            plt.colorbar(im1_ind, ax=axes_ind[1], fraction=0.046, pad=0.04)
            
            im2_ind = axes_ind[2].imshow(prediction_np, cmap='hot')
            axes_ind[2].set_title(f'Prediction\nRange: [{prediction_np.min():.1f}, {prediction_np.max():.1f}]')
            axes_ind[2].axis('off')
            plt.colorbar(im2_ind, ax=axes_ind[2], fraction=0.046, pad=0.04)
            
            im3_ind = axes_ind[3].imshow(error_map, cmap='hot')
            axes_ind[3].set_title(f'Absolute Error\nRange: [{error_map.min():.1f}, {error_map.max():.1f}]')
            axes_ind[3].axis('off')
            plt.colorbar(im3_ind, ax=axes_ind[3], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(f'predictions/best_model_validation_predictions/validation_sample_{i+1}_prediction.png', dpi=300, bbox_inches='tight')
            plt.close(fig_ind)
    
    plt.tight_layout()
    plt.savefig('predictions/best_model_validation_predictions_summary.png', dpi=300, bbox_inches='tight')
    print(f"\nValidation predictions saved to predictions/best_model_validation_predictions/")
    print(f"Summary plot saved as predictions/best_model_validation_predictions_summary.png")
    
    # Calculate validation statistics
    calculate_validation_statistics(model, dataloader, device)
    
    print(f"\n✅ Best regression model validation predictions completed!")

def calculate_validation_statistics(model, dataloader, device):
    """Calculate validation prediction statistics"""
    
    print(f"\n=== CALCULATING VALIDATION STATISTICS ===")
    
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating validation statistics"):
            images = batch['image'].to(device)
            targets = batch['distance_map'].to(device)
            
            # Pad images if needed
            h, w = images.shape[2], images.shape[3]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            
            if pad_h > 0 or pad_w > 0:
                images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                targets = torch.nn.functional.pad(targets, (0, pad_w, 0, pad_h), mode='reflect')
            
            predictions = model(images)
            
            # Remove padding if it was added
            if pad_h > 0 or pad_w > 0:
                predictions = predictions[:, :, :h, :w]
                # Handle different target dimensions
                if targets.dim() == 4:
                    targets = targets[:, :, :h, :w]
                else:
                    targets = targets[:, :h, :w]
            
            # Calculate metrics
            mse = torch.nn.functional.mse_loss(predictions, targets)
            mae = torch.nn.functional.l1_loss(predictions, targets)
            
            total_mse += mse.item()
            total_mae += mae.item()
            num_samples += 1
    
    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples
    
    print(f"Validation Average MSE: {avg_mse:.4f}")
    print(f"Validation Average MAE: {avg_mae:.4f}")
    print(f"Total validation samples evaluated: {num_samples}")

if __name__ == "__main__":
    show_best_regression_predictions() 