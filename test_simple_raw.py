# Simple Test - Just Load Images and Run Model
# No normalization, no preprocessing, just raw model inference

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.base_models import UNet
import os
from tqdm import tqdm
import sys

print("=== SIMPLE TEST ON DRIVE TEST DATA ===")

# Device setup
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

# Load trained model
model_path = 'checkpoints_regression/best_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from {model_path}")
else:
    print(f"❌ Model not found at {model_path}")
    print("Available models:")
    if os.path.exists('checkpoints_regression/'):
        for file in os.listdir('checkpoints_regression/'):
            if file.endswith('.pth'):
                print(f"  - {file}")
    sys.exit(1)

model.eval()

# Test data paths
test_images_dir = 'drive/test/images_npy/'
test_images = [f for f in os.listdir(test_images_dir) if f.endswith('.npy')]
test_images.sort()

print(f"Found {len(test_images)} test images")

# Create output directory
os.makedirs('predictions/test_predictions', exist_ok=True)

# Process each test image
all_predictions = []
all_images = []

for i, image_file in enumerate(tqdm(test_images, desc="Processing test images")):
    # Load test image
    image_path = os.path.join(test_images_dir, image_file)
    image = np.load(image_path)
    
    # Convert to tensor and convert to float32 (no normalization)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Pad if needed
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    
    if pad_h > 0 or pad_w > 0:
        image_tensor = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    # Predict
    with torch.no_grad():
        prediction = model(image_tensor)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            prediction = prediction[:, :, :h, :w]
    
    # Convert to numpy
    pred_np = prediction[0, 0].cpu().numpy()
    
    # Save prediction
    pred_filename = f"test_{image_file.replace('.npy', '_distance_map.npy')}"
    pred_path = os.path.join('predictions/test_predictions', pred_filename)
    np.save(pred_path, pred_np)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f'Test Image: {image_file}')
    axes[0].axis('off')
    
    # Predicted distance map
    im1 = axes[1].imshow(pred_np, cmap='hot')
    axes[1].set_title(f'Predicted Distance Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    
    # Save individual plot
    plot_filename = f"test_{image_file.replace('.npy', '.png')}"
    plot_path = os.path.join('predictions/test_predictions', plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Store for summary
    all_predictions.append(pred_np)
    all_images.append(image)

# Create summary plot
print("Creating summary plot...")
n_images = len(all_images)
n_cols = 5
n_rows = (n_images + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(20, 4 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)

for i in range(n_images):
    row = i // n_cols
    col = (i % n_cols) * 2
    
    # Original image
    axes[row, col].imshow(all_images[i])
    axes[row, col].set_title(f'Test {i+1}')
    axes[row, col].axis('off')
    
    # Predicted distance map
    im = axes[row, col + 1].imshow(all_predictions[i], cmap='hot')
    axes[row, col + 1].set_title(f'Distance Map {i+1}')
    axes[row, col + 1].axis('off')
    plt.colorbar(im, ax=axes[row, col + 1])

# Hide empty subplots
for i in range(n_images, n_rows * n_cols):
    row = i // n_cols
    col = (i % n_cols) * 2
    axes[row, col].axis('off')
    axes[row, col + 1].axis('off')

plt.tight_layout()
plt.savefig('predictions/test_predictions_summary.png', dpi=300, bbox_inches='tight')

# Print statistics
print("\n=== TEST RESULTS ===")
print(f"Processed {len(test_images)} test images")
print(f"Predictions saved to: predictions/test_predictions/")
print(f"Summary plot saved to: predictions/test_predictions_summary.png")

# Calculate statistics
all_pred_flat = np.concatenate([pred.flatten() for pred in all_predictions])
print(f"\nDistance Map Statistics:")
print(f"  Mean: {np.mean(all_pred_flat):.4f}")
print(f"  Std: {np.std(all_pred_flat):.4f}")
print(f"  Min: {np.min(all_pred_flat):.4f}")
print(f"  Max: {np.max(all_pred_flat):.4f}")
print(f"  Range: {np.max(all_pred_flat) - np.min(all_pred_flat):.4f}")

print("\n✅ Test completed successfully!") 