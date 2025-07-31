import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import yaml
from models.base_models import UNet
import os
from core.general_dataset.io import load_array_from_file
from tqdm import tqdm

def test_on_test_images():
    """Test the trained model on test images"""
    
    print("=== TESTING MODEL ON TEST IMAGES ===")
    
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
    
    # Load the trained model (60 epochs)
    model_path = 'checkpoints_regression/final_60_epochs_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded trained model from {model_path}")
    else:
        print(f"❌ Model not found at {model_path}")
        return
    
    model.eval()
    
    # Get test image paths
    test_images_dir = 'drive/test/images'
    test_image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.tif')])
    
    print(f"Found {len(test_image_files)} test images")
    
    # Create output directory for test predictions
    os.makedirs('predictions/test_predictions', exist_ok=True)
    
    # Process each test image
    for i, image_file in enumerate(tqdm(test_image_files, desc="Processing test images")):
        image_path = os.path.join(test_images_dir, image_file)
        
        # Load image
        image = load_array_from_file(image_path)
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        
        # Pad image to make it divisible by 4
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        
        if pad_h > 0 or pad_w > 0:
            image_tensor = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Make prediction
        with torch.no_grad():
            prediction = model(image_tensor)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            prediction = prediction[:, :, :h, :w]
        
        # Convert to numpy
        prediction_np = prediction[0, 0].cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image / 255.0)
        axes[0].set_title(f'Test Image: {image_file}')
        axes[0].axis('off')
        
        # Predicted distance map
        im1 = axes[1].imshow(prediction_np, cmap='hot')
        axes[1].set_title(f'Predicted Distance Map\nRange: [{prediction_np.min():.1f}, {prediction_np.max():.1f}]')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Histogram of distance values
        axes[2].hist(prediction_np.flatten(), bins=50, alpha=0.7, color='red')
        axes[2].set_title('Distance Map Histogram')
        axes[2].set_xlabel('Distance Value')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save prediction
        output_filename = f"predictions/test_predictions/test_{i+1:02d}_{image_file.replace('.tif', '')}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save distance map as numpy file
        distance_map_filename = f"predictions/test_predictions/test_{i+1:02d}_{image_file.replace('.tif', '_distance_map.npy')}"
        np.save(distance_map_filename, prediction_np)
    
    # Create a summary plot with all predictions
    print("\nCreating summary plot...")
    
    # Load a few sample predictions for summary
    sample_predictions = []
    sample_images = []
    
    for i in range(min(6, len(test_image_files))):
        image_path = os.path.join(test_images_dir, test_image_files[i])
        image = load_array_from_file(image_path)
        
        # Get prediction
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        
        # Pad
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        
        if pad_h > 0 or pad_w > 0:
            image_tensor = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        with torch.no_grad():
            prediction = model(image_tensor)
        
        if pad_h > 0 or pad_w > 0:
            prediction = prediction[:, :, :h, :w]
        
        sample_images.append(image)
        sample_predictions.append(prediction[0, 0].cpu().numpy())
    
    # Create summary plot
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle('Test Images and Their Predicted Distance Maps (60 epochs model)', fontsize=16)
    
    for i in range(6):
        # Original image
        axes[0, i].imshow(sample_images[i] / 255.0)
        axes[0, i].set_title(f'Test {i+1}')
        axes[0, i].axis('off')
        
        # Predicted distance map
        im = axes[1, i].imshow(sample_predictions[i], cmap='hot')
        axes[1, i].set_title(f'Distance Map {i+1}\nRange: [{sample_predictions[i].min():.1f}, {sample_predictions[i].max():.1f}]')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('predictions/test_predictions_summary.png', dpi=300, bbox_inches='tight')
    
    print("\n✅ Test predictions completed!")
    print(f"📊 Summary:")
    print(f"  - Individual predictions: predictions/test_predictions/")
    print(f"  - Summary plot: predictions/test_predictions_summary.png")
    print(f"  - Distance maps saved as .npy files")
    print(f"  - Processed {len(test_image_files)} test images")

if __name__ == "__main__":
    test_on_test_images() 