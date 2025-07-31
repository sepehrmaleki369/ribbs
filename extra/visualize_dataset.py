import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from core.general_dataset.io import load_array_from_file
import seaborn as sns

def analyze_and_visualize_dataset():
    """Analyze and visualize the drive dataset"""
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory for plots
    os.makedirs("dataset_analysis", exist_ok=True)
    
    # Analyze training data
    print("=== TRAINING DATA ANALYSIS ===")
    train_images_dir = "drive/training/images"
    train_masks_dir = "drive/training/mask"
    
    train_image_files = sorted([f for f in os.listdir(train_images_dir) if f.endswith('.npy')])
    train_mask_files = sorted([f for f in os.listdir(train_masks_dir) if f.endswith('.npy')])
    
    print(f"Number of training images: {len(train_image_files)}")
    print(f"Number of training masks: {len(train_mask_files)}")
    
    # Analyze test data
    print("\n=== TEST DATA ANALYSIS ===")
    test_images_dir = "drive/test/images"
    test_masks_dir = "drive/test/mask"
    
    test_image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.tif')])
    test_mask_files = sorted([f for f in os.listdir(test_masks_dir) if f.endswith('.gif')])
    
    print(f"Number of test images: {len(test_image_files)}")
    print(f"Number of test masks: {len(test_mask_files)}")
    
    # Data range analysis
    print("\n=== DATA RANGE ANALYSIS ===")
    
    # Training images analysis
    train_image_ranges = []
    train_mask_ranges = []
    
    for i, (img_file, mask_file) in enumerate(zip(train_image_files[:5], train_mask_files[:5])):  # Analyze first 5
        img_path = os.path.join(train_images_dir, img_file)
        mask_path = os.path.join(train_masks_dir, mask_file)
        
        img = load_array_from_file(img_path)
        mask = load_array_from_file(mask_path)
        
        train_image_ranges.append({
            'file': img_file,
            'min': img.min(),
            'max': img.max(),
            'mean': img.mean(),
            'std': img.std(),
            'shape': img.shape
        })
        
        train_mask_ranges.append({
            'file': mask_file,
            'min': mask.min(),
            'max': mask.max(),
            'mean': mask.mean(),
            'std': mask.std(),
            'shape': mask.shape,
            'unique_values': np.unique(mask)
        })
        
        print(f"Training Image {i+1}: {img_file}")
        print(f"  Shape: {img.shape}, Range: [{img.min():.2f}, {img.max():.2f}], Mean: {img.mean():.2f}")
        print(f"  Mask: {mask_file}")
        print(f"  Shape: {mask.shape}, Range: [{mask.min():.2f}, {mask.max():.2f}], Mean: {mask.mean():.2f}")
        print(f"  Unique values: {np.unique(mask)}")
        print()
    
    # Test images analysis
    test_image_ranges = []
    test_mask_ranges = []
    
    for i, (img_file, mask_file) in enumerate(zip(test_image_files[:5], test_mask_files[:5])):  # Analyze first 5
        img_path = os.path.join(test_images_dir, img_file)
        mask_path = os.path.join(test_masks_dir, mask_file)
        
        img = load_array_from_file(img_path)
        mask = load_array_from_file(mask_path)
        
        test_image_ranges.append({
            'file': img_file,
            'min': img.min(),
            'max': img.max(),
            'mean': img.mean(),
            'std': img.std(),
            'shape': img.shape
        })
        
        test_mask_ranges.append({
            'file': mask_file,
            'min': mask.min(),
            'max': mask.max(),
            'mean': mask.mean(),
            'std': mask.std(),
            'shape': mask.shape,
            'unique_values': np.unique(mask)
        })
        
        print(f"Test Image {i+1}: {img_file}")
        print(f"  Shape: {img.shape}, Range: [{img.min():.2f}, {img.max():.2f}], Mean: {img.mean():.2f}")
        print(f"  Mask: {mask_file}")
        print(f"  Shape: {mask.shape}, Range: [{mask.min():.2f}, {mask.max():.2f}], Mean: {mask.mean():.2f}")
        print(f"  Unique values: {np.unique(mask)}")
        print()
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Drive Dataset Visualization - Training Data (First 5 samples)', fontsize=16)
    
    for i in range(5):
        if i < len(train_image_files):
            # Load data
            img_path = os.path.join(train_images_dir, train_image_files[i])
            mask_path = os.path.join(train_masks_dir, train_mask_files[i])
            
            img = load_array_from_file(img_path)
            mask = load_array_from_file(mask_path)
            
            # Handle different channel orders
            if img.ndim == 3 and img.shape[2] == 3:  # (H, W, C)
                img_display = img
            elif img.ndim == 3 and img.shape[0] == 3:  # (C, H, W)
                img_display = img.transpose(1, 2, 0)
            else:
                img_display = img
            
            # Normalize image for display
            if img_display.max() > 1:
                img_display = img_display / 255.0
            
            # Plot original image
            axes[0, i].imshow(img_display)
            axes[0, i].set_title(f'Image {i+1}\nRange: [{img.min():.0f}, {img.max():.0f}]')
            axes[0, i].axis('off')
            
            # Plot mask
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].set_title(f'Mask {i+1}\nRange: [{mask.min():.0f}, {mask.max():.0f}]')
            axes[1, i].axis('off')
            
            # Plot overlay
            overlay = img_display.copy()
            if overlay.ndim == 3:
                # Create red overlay for mask
                mask_overlay = np.zeros_like(overlay)
                mask_overlay[:, :, 0] = mask / mask.max()  # Red channel
                overlay = np.clip(overlay + mask_overlay * 0.5, 0, 1)
            else:
                overlay = np.stack([overlay, overlay, overlay], axis=-1)
                mask_overlay = np.zeros_like(overlay)
                mask_overlay[:, :, 0] = mask / mask.max()
                overlay = np.clip(overlay + mask_overlay * 0.5, 0, 1)
            
            axes[2, i].imshow(overlay)
            axes[2, i].set_title(f'Overlay {i+1}')
            axes[2, i].axis('off')
            
            # Plot histogram
            axes[3, i].hist(img.flatten(), bins=50, alpha=0.7, label='Image', color='blue')
            axes[3, i].hist(mask.flatten(), bins=20, alpha=0.7, label='Mask', color='red')
            axes[3, i].set_title(f'Histogram {i+1}')
            axes[3, i].legend()
            axes[3, i].set_xlabel('Pixel Value')
            axes[3, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis/training_data_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create test data visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Drive Dataset Visualization - Test Data (First 5 samples)', fontsize=16)
    
    for i in range(5):
        if i < len(test_image_files):
            # Load data
            img_path = os.path.join(test_images_dir, test_image_files[i])
            mask_path = os.path.join(test_masks_dir, test_mask_files[i])
            
            img = load_array_from_file(img_path)
            mask = load_array_from_file(mask_path)
            
            # Handle different channel orders
            if img.ndim == 3 and img.shape[2] == 3:  # (H, W, C)
                img_display = img
            elif img.ndim == 3 and img.shape[0] == 3:  # (C, H, W)
                img_display = img.transpose(1, 2, 0)
            else:
                img_display = img
            
            # Normalize image for display
            if img_display.max() > 1:
                img_display = img_display / 255.0
            
            # Plot original image
            axes[0, i].imshow(img_display)
            axes[0, i].set_title(f'Image {i+1}\nRange: [{img.min():.0f}, {img.max():.0f}]')
            axes[0, i].axis('off')
            
            # Plot mask
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].set_title(f'Mask {i+1}\nRange: [{mask.min():.0f}, {mask.max():.0f}]')
            axes[1, i].axis('off')
            
            # Plot overlay
            overlay = img_display.copy()
            if overlay.ndim == 3:
                # Create red overlay for mask
                mask_overlay = np.zeros_like(overlay)
                mask_overlay[:, :, 0] = mask / mask.max()  # Red channel
                overlay = np.clip(overlay + mask_overlay * 0.5, 0, 1)
            else:
                overlay = np.stack([overlay, overlay, overlay], axis=-1)
                mask_overlay = np.zeros_like(overlay)
                mask_overlay[:, :, 0] = mask / mask.max()
                overlay = np.clip(overlay + mask_overlay * 0.5, 0, 1)
            
            axes[2, i].imshow(overlay)
            axes[2, i].set_title(f'Overlay {i+1}')
            axes[2, i].axis('off')
            
            # Plot histogram
            axes[3, i].hist(img.flatten(), bins=50, alpha=0.7, label='Image', color='blue')
            axes[3, i].hist(mask.flatten(), bins=20, alpha=0.7, label='Mask', color='red')
            axes[3, i].set_title(f'Histogram {i+1}')
            axes[3, i].legend()
            axes[3, i].set_xlabel('Pixel Value')
            axes[3, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis/test_data_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary statistics plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training image statistics
    train_img_mins = [r['min'] for r in train_image_ranges]
    train_img_maxs = [r['max'] for r in train_image_ranges]
    train_img_means = [r['mean'] for r in train_image_ranges]
    
    axes[0, 0].bar(range(len(train_img_mins)), train_img_mins, alpha=0.7, label='Min')
    axes[0, 0].bar(range(len(train_img_maxs)), train_img_maxs, alpha=0.7, label='Max')
    axes[0, 0].plot(range(len(train_img_means)), train_img_means, 'ro-', label='Mean')
    axes[0, 0].set_title('Training Images - Value Ranges')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Pixel Value')
    axes[0, 0].legend()
    
    # Training mask statistics
    train_mask_mins = [r['min'] for r in train_mask_ranges]
    train_mask_maxs = [r['max'] for r in train_mask_ranges]
    train_mask_means = [r['mean'] for r in train_mask_ranges]
    
    axes[0, 1].bar(range(len(train_mask_mins)), train_mask_mins, alpha=0.7, label='Min')
    axes[0, 1].bar(range(len(train_mask_maxs)), train_mask_maxs, alpha=0.7, label='Max')
    axes[0, 1].plot(range(len(train_mask_means)), train_mask_means, 'ro-', label='Mean')
    axes[0, 1].set_title('Training Masks - Value Ranges')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Pixel Value')
    axes[0, 1].legend()
    
    # Test image statistics
    test_img_mins = [r['min'] for r in test_image_ranges]
    test_img_maxs = [r['max'] for r in test_image_ranges]
    test_img_means = [r['mean'] for r in test_image_ranges]
    
    axes[1, 0].bar(range(len(test_img_mins)), test_img_mins, alpha=0.7, label='Min')
    axes[1, 0].bar(range(len(test_img_maxs)), test_img_maxs, alpha=0.7, label='Max')
    axes[1, 0].plot(range(len(test_img_means)), test_img_means, 'ro-', label='Mean')
    axes[1, 0].set_title('Test Images - Value Ranges')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Pixel Value')
    axes[1, 0].legend()
    
    # Test mask statistics
    test_mask_mins = [r['min'] for r in test_mask_ranges]
    test_mask_maxs = [r['max'] for r in test_mask_ranges]
    test_mask_means = [r['mean'] for r in test_mask_ranges]
    
    axes[1, 1].bar(range(len(test_mask_mins)), test_mask_mins, alpha=0.7, label='Min')
    axes[1, 1].bar(range(len(test_mask_maxs)), test_mask_maxs, alpha=0.7, label='Max')
    axes[1, 1].plot(range(len(test_mask_means)), test_mask_means, 'ro-', label='Mean')
    axes[1, 1].set_title('Test Masks - Value Ranges')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Pixel Value')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('dataset_analysis/data_ranges_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comprehensive summary
    print("\n=== COMPREHENSIVE DATASET SUMMARY ===")
    print(f"Training Images: {len(train_image_files)} files")
    print(f"Training Masks: {len(train_mask_files)} files")
    print(f"Test Images: {len(test_image_files)} files")
    print(f"Test Masks: {len(test_mask_files)} files")
    
    print("\nTraining Image Statistics:")
    print(f"  Min value across all: {min([r['min'] for r in train_image_ranges])}")
    print(f"  Max value across all: {max([r['max'] for r in train_image_ranges])}")
    print(f"  Mean value across all: {np.mean([r['mean'] for r in train_image_ranges]):.2f}")
    
    print("\nTraining Mask Statistics:")
    print(f"  Min value across all: {min([r['min'] for r in train_mask_ranges])}")
    print(f"  Max value across all: {max([r['max'] for r in train_mask_ranges])}")
    print(f"  Mean value across all: {np.mean([r['mean'] for r in train_mask_ranges]):.2f}")
    print(f"  Unique values found: {set().union(*[set(r['unique_values']) for r in train_mask_ranges])}")
    
    print("\nTest Image Statistics:")
    print(f"  Min value across all: {min([r['min'] for r in test_image_ranges])}")
    print(f"  Max value across all: {max([r['max'] for r in test_image_ranges])}")
    print(f"  Mean value across all: {np.mean([r['mean'] for r in test_image_ranges]):.2f}")
    
    print("\nTest Mask Statistics:")
    print(f"  Min value across all: {min([r['min'] for r in test_mask_ranges])}")
    print(f"  Max value across all: {max([r['max'] for r in test_mask_ranges])}")
    print(f"  Mean value across all: {np.mean([r['mean'] for r in test_mask_ranges]):.2f}")
    print(f"  Unique values found: {set().union(*[set(r['unique_values']) for r in test_mask_ranges])}")
    
    print(f"\nVisualization saved to: dataset_analysis/")

if __name__ == "__main__":
    analyze_and_visualize_dataset() 