import numpy as np
import matplotlib.pyplot as plt
import os
from core.general_dataset.io import load_array_from_file
from scipy.ndimage import distance_transform_edt
import seaborn as sns

def test_distance_transform():
    """Apply distance transform to labels and visualize results"""
    
    print("=== TESTING DISTANCE TRANSFORM ===")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs("distance_transform_analysis", exist_ok=True)
    
    labels_dir = "drive/training/inverted_labels"
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.npy')])
    
    print(f"Found {len(label_files)} label files")
    
    # Test on first few samples
    num_samples = min(6, len(label_files))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    fig.suptitle('Distance Transform Analysis', fontsize=16)
    
    for i in range(num_samples):
        label_file = label_files[i]
        label_path = os.path.join(labels_dir, label_file)
        label = load_array_from_file(label_path)
        
        # Convert to binary (0 for background, 1 for vessels)
        binary_label = (label > 0).astype(np.uint8)
        
        # Apply distance transform
        # Distance from background to vessels
        dist_to_vessels = distance_transform_edt(binary_label == 0)
        
        # Distance from vessels to background
        dist_from_vessels = distance_transform_edt(binary_label == 1)
        
        # Signed distance transform (positive inside vessels, negative outside)
        signed_dist = dist_from_vessels - dist_to_vessels
        
        print(f"\nSample {i+1}: {label_file}")
        print(f"  Original label range: [{label.min()}, {label.max()}]")
        print(f"  Binary label range: [{binary_label.min()}, {binary_label.max()}]")
        print(f"  Distance to vessels range: [{dist_to_vessels.min():.2f}, {dist_to_vessels.max():.2f}]")
        print(f"  Distance from vessels range: [{dist_from_vessels.min():.2f}, {dist_from_vessels.max():.2f}]")
        print(f"  Signed distance range: [{signed_dist.min():.2f}, {signed_dist.max():.2f}]")
        
        # Plot original label
        axes[i, 0].imshow(label, cmap='gray')
        axes[i, 0].set_title(f"Original Label\nRange: [{label.min()}, {label.max()}]")
        axes[i, 0].axis('off')
        
        # Plot binary label
        axes[i, 1].imshow(binary_label, cmap='gray')
        axes[i, 1].set_title(f"Binary Label\nRange: [{binary_label.min()}, {binary_label.max()}]")
        axes[i, 1].axis('off')
        
        # Plot distance to vessels
        im1 = axes[i, 2].imshow(dist_to_vessels, cmap='hot')
        axes[i, 2].set_title(f"Distance to Vessels\nRange: [{dist_to_vessels.min():.1f}, {dist_to_vessels.max():.1f}]")
        axes[i, 2].axis('off')
        plt.colorbar(im1, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Plot signed distance
        im2 = axes[i, 3].imshow(signed_dist, cmap='RdBu_r', vmin=-50, vmax=50)
        axes[i, 3].set_title(f"Signed Distance\nRange: [{signed_dist.min():.1f}, {signed_dist.max():.1f}]")
        axes[i, 3].axis('off')
        plt.colorbar(im2, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('distance_transform_analysis/distance_transform_results.png', dpi=300, bbox_inches='tight')
    print("Distance transform results saved as: distance_transform_analysis/distance_transform_results.png")
    
    # Create histogram analysis
    print("\n=== CREATING HISTOGRAM ANALYSIS ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distance Transform Histograms', fontsize=16)
    
    for i in range(min(6, len(label_files))):
        label_file = label_files[i]
        label_path = os.path.join(labels_dir, label_file)
        label = load_array_from_file(label_path)
        
        binary_label = (label > 0).astype(np.uint8)
        dist_to_vessels = distance_transform_edt(binary_label == 0)
        dist_from_vessels = distance_transform_edt(binary_label == 1)
        signed_dist = dist_from_vessels - dist_to_vessels
        
        row = i // 3
        col = i % 3
        
        # Plot histogram
        axes[row, col].hist(dist_to_vessels.flatten(), bins=50, alpha=0.7, label='Distance to Vessels')
        axes[row, col].hist(signed_dist.flatten(), bins=50, alpha=0.7, label='Signed Distance')
        axes[row, col].set_title(f'Sample {i+1}: Distance Distributions')
        axes[row, col].set_xlabel('Distance Value')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig('distance_transform_analysis/distance_histograms.png', dpi=300, bbox_inches='tight')
    print("Distance histograms saved as: distance_transform_analysis/distance_histograms.png")
    
    # Analyze distance statistics
    print("\n=== DISTANCE STATISTICS ANALYSIS ===")
    
    all_dist_to_vessels = []
    all_dist_from_vessels = []
    all_signed_dist = []
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        label = load_array_from_file(label_path)
        
        binary_label = (label > 0).astype(np.uint8)
        dist_to_vessels = distance_transform_edt(binary_label == 0)
        dist_from_vessels = distance_transform_edt(binary_label == 1)
        signed_dist = dist_from_vessels - dist_to_vessels
        
        all_dist_to_vessels.append(dist_to_vessels)
        all_dist_from_vessels.append(dist_from_vessels)
        all_signed_dist.append(signed_dist)
    
    # Calculate statistics
    all_dist_to_vessels_array = np.concatenate([d.flatten() for d in all_dist_to_vessels])
    all_dist_from_vessels_array = np.concatenate([d.flatten() for d in all_dist_from_vessels])
    all_signed_dist_array = np.concatenate([d.flatten() for d in all_signed_dist])
    
    print(f"Distance to vessels statistics:")
    print(f"  Min: {all_dist_to_vessels_array.min():.2f}")
    print(f"  Max: {all_dist_to_vessels_array.max():.2f}")
    print(f"  Mean: {all_dist_to_vessels_array.mean():.2f}")
    print(f"  Std: {all_dist_to_vessels_array.std():.2f}")
    
    print(f"\nDistance from vessels statistics:")
    print(f"  Min: {all_dist_from_vessels_array.min():.2f}")
    print(f"  Max: {all_dist_from_vessels_array.max():.2f}")
    print(f"  Mean: {all_dist_from_vessels_array.mean():.2f}")
    print(f"  Std: {all_dist_from_vessels_array.std():.2f}")
    
    print(f"\nSigned distance statistics:")
    print(f"  Min: {all_signed_dist_array.min():.2f}")
    print(f"  Max: {all_signed_dist_array.max():.2f}")
    print(f"  Mean: {all_signed_dist_array.mean():.2f}")
    print(f"  Std: {all_signed_dist_array.std():.2f}")
    
    # Create summary visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distance Transform Summary Statistics', fontsize=16)
    
    axes[0].hist(all_dist_to_vessels_array, bins=100, alpha=0.7)
    axes[0].set_title('Distance to Vessels Distribution')
    axes[0].set_xlabel('Distance Value')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(all_dist_from_vessels_array, bins=100, alpha=0.7)
    axes[1].set_title('Distance from Vessels Distribution')
    axes[1].set_xlabel('Distance Value')
    axes[1].set_ylabel('Frequency')
    
    axes[2].hist(all_signed_dist_array, bins=100, alpha=0.7)
    axes[2].set_title('Signed Distance Distribution')
    axes[2].set_xlabel('Distance Value')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('distance_transform_analysis/summary_statistics.png', dpi=300, bbox_inches='tight')
    print("Summary statistics saved as: distance_transform_analysis/summary_statistics.png")
    
    print(f"\nAll distance transform analysis files saved to: distance_transform_analysis/")

if __name__ == "__main__":
    test_distance_transform() 